#!/usr/bin/env python3
"""Route the SM12x FP8 paged-MQA logits around DeepGEMM's missing page geometry.

This closes the kit's **second** DeepGEMM blocker, and the one that survives
``--block-size 64``: the engine now boots and warms up, then dies on the first
long prompt.

The failure
-----------
``model_executor/layers/sparse_attn_indexer.py`` computes the indexer logits for
decode with

    fp8_fp4_paged_mqa_logits((q, None), kv_cache, weights, seq_lens,
                             decode_metadata.block_table, schedule_metadata,
                             max_model_len, clean_logits=False, indices=None)

and ``kv_cache`` there is the indexer's compressed pool in the 4D
``[num_blocks, num_states, 1, head_dim + 4]`` view that
``kv_cache_as_quant_view`` builds. DeepGEMM takes the page's state count as the
``block_kv`` compile-time template parameter, and on arch 12 (SM120/SM121) its
FP8 logits kernel admits **exactly 64**:

    Assertion error (csrc/apis/attention.hpp:320):
    (arch_major == 10 and (block_kv == 32 or block_kv == 64 or block_kv == 128))
    or (arch_major == 9 and (block_kv == 32 or block_kv == 64))
    or (arch_major == 12 and ((is_fp4 and (block_kv == 32 or block_kv == 64))
                              or (not is_fp4 and block_kv == 64)))

``num_states`` is ``block_size // tokens_per_state`` and V4.1's compressed caches
use ``tokens_per_state = compress_ratio``. This checkpoint carries ratios 1 and
2, so ``--block-size 64`` makes the ratio-1 pages 64 states and the ratio-2 pages
**32**. The 32 is what aborts, and it aborts only after a long prompt reaches
decode: warmup exercises the *metadata* builder, whose own assert
(``attention.hpp:247``/``:262``) accepts both 32 and 64.

There is no manager block size that avoids this. 64 gives {64, 32}; 128 gives
{128, 64} and 128 is not legal for FP8 on arch 12 either. Both of this
checkpoint's ratios cannot be simultaneously servable by DeepGEMM's arch-12 FP8
kernel. This is vLLM issue **#53635** (open); its option (c) is a non-DeepGEMM
fallback for the small-page geometries, which is what this patch installs.

Why a Triton fallback, and why it is safe
-----------------------------------------
``overlay/sm12x_mqa.py`` vendors the FP8 paged MQA-logits Triton kernels from
vLLM PR **#41834** (jasl, "Add SM12x support for DeepSeek V4 Flash with
essential fixes"), scoped to the paged path only. Three measured facts license
using them here, all from ``for_review/v41-block-size/
DEEPGEMM-ARCH12-FALLBACK-PROBE.txt`` on the real GB10:

* At 64 states -- the geometry both accept -- the Triton kernel matches
  DeepGEMM on the valid columns to ``max_rel 7.8e-07``. It is numerically the
  same operator, not an approximation of it. (``tl.dot`` on fp8 values widened
  to fp32 with ``input_precision="tf32"`` is *exact*: an e4m3 mantissa is a
  subset of tf32's, so the products match fp8 MMA and only the fp32 summation
  order differs.)
* At 32 states it matches an independently unpacked FP32 reference to
  ``max_abs 1.2e-04`` on ``|ref|max ~1.1e3``, i.e. one ulp of fp32 at that
  magnitude.
* The kernels read the packed page directly, and their value/scale addressing
  is byte-for-byte the layout DeepGEMM itself derives -- values at page offset
  0 with state stride ``head_dim``, scales at byte ``num_states * head_dim``
  with stride ``page_bytes / 4``. This is a *segregated* page (all values, then
  all scales), not an interleaved one; interleaving misreads every scale. The
  overlay helper re-derives both tensors from the same offsets DeepGEMM asserts.

One behaviour deliberately differs. With ``clean_logits=False`` and a 2D
``context_lens``, DeepGEMM never writes the beyond-context columns at all: they
are raw ``torch.empty`` contents, not a contract (measured: 0 of 84344 are
``-inf``). The Triton kernel writes ``-inf`` on all of them. Cleaning is a
**strict superset** of the contract the caller asked for -- the same argument
jasl/vllm#44 established for the contiguous sibling of this kernel -- and
nothing downstream can observe a value it was not already required to mask off.
The pinned scoring, candidate-filtering and top-k sequence is untouched.

Scope
-----
Two purely additive edits to ``utils/deep_gemm.py``: the hook definition, and
its call at the very top of ``fp8_fp4_paged_mqa_logits``'s body -- before
``_lazy_init()`` and before the ``_missing()`` guard, so the fallback also works
where the DeepGEMM implementation is absent.

Nothing else is patched. In particular ``v1/attention/backends/mla/indexer.py``
is left alone: its ``get_paged_mqa_logits_metadata(seq_lens,
kv_cache_spec.num_states, num_sms)`` call is **legal** on arch 12 at both 32 and
64 states (measured), so there is no zero-filled-schedule hazard of the kind
PR #53522 has to avoid, and suppressing a call that succeeds would buy nothing.
Only the paged wrapper is hooked: the contiguous/prefill path does not take its
``block_kv`` from the cache geometry, never hits this restriction, and stays on
DeepGEMM -- routing it to Triton would be unreviewed risk for no benefit.

Out of scope, deliberately:

* **MXFP4 Q** (``use_fp4_cache``, ``q[1] is not None``). DeepGEMM's arch-12
  MXFP4 logits kernel accepts both 32 and 64, so it has no such gap, and the
  overlay kernel is FP8-only. PR #41834's own review flags the reverse mistake
  (dropping the DeepGEMM requirement for the FP4 path).
* **A DeepGEMM-less install.** ``sparse_attn_indexer.py`` still raises when
  ``is_cuda() and not has_deep_gemm()``, and the rest of V4.1 leans on DeepGEMM
  far beyond this one call.

See `RESEARCH.md` §11.9.17.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

MARK = "dsv41-sm12x-indexer"

DEEP_GEMM_REL = Path("utils/deep_gemm.py")

# DeepGEMM's arch-12 FP8 paged-MQA logits kernel admits exactly this many states
# per page; anything else has no instantiation at all.
_DEEPGEMM_ARCH12_FP8_BLOCK_KV = 64

# The indexer's metadata builder asserts this set on arch 12 and runs before any
# logits call, so a page outside it never reaches the hook. Reaching it anyway
# means an invariant this overlay has not reasoned about, and it fails closed.
_METADATA_BLOCK_KV = (32, 64)

# The injected module-level constants. Interpolated from the two above so the
# injected source and this script cannot disagree; a separate small f-string,
# because the hook body below contains braces of its own.
_HOOK_CONSTS = f"""# Overlay constants (overlay/patch_sm12x_indexer.py). DeepGEMM gates the
# paged-MQA path twice and the two sets differ: the FP8 logits kernel is
# templated for exactly _DSV41_ARCH12_FP8_BLOCK_KV states per page, while the
# metadata builder that runs first admits _DSV41_METADATA_BLOCK_KV.
_DSV41_ARCH12_FP8_BLOCK_KV = {_DEEPGEMM_ARCH12_FP8_BLOCK_KV}
_DSV41_METADATA_BLOCK_KV = {_METADATA_BLOCK_KV}


"""

# Inserted immediately before `def fp8_fp4_paged_mqa_logits(`. Anchored on the
# def so the hook cannot land inside another function.
_HOOK_OLD = "def fp8_fp4_paged_mqa_logits(\n"

_HOOK_BODY = '''def _dsv41_sm12x_fp8_paged_mqa_logits(
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
    indices: torch.Tensor | None,
) -> torch.Tensor | None:
    """Overlay hook: FP8 paged MQA logits on arch 12 for pages DeepGEMM rejects.

    Returns ``None`` to leave the call on DeepGEMM, or the logits when the
    overlay's Triton kernel served it instead. Installed by
    ``overlay/patch_sm12x_indexer.py``; see that file for the full rationale and
    for the hardware receipts behind every claim here.

    DeepGEMM's arch-12 FP8 paged-MQA logits kernel is templated for exactly
    ``block_kv == 64`` states per page, but this checkpoint's ratio-2 indexer
    layers present 32 at ``--block-size 64``, and the call aborts during decode
    on the first long prompt. The page's state count is the template parameter,
    so it cannot be worked around from the call site -- only the kernel can
    change.
    """
    # Arch 12 only, and only the FP8 cache. Everywhere else this is a no-op and
    # DeepGEMM stays the default: this is a fallback, not a replacement.
    if not current_platform.is_cuda():
        return None
    if not current_platform.is_device_capability_family(120):
        return None
    q_values, q_scale = q
    if q_scale is not None:
        # MXFP4 Q (``use_fp4_cache``). Unreachable on arch 12: vLLM resolves
        # ``indexer_kv_dtype`` to mxfp4 only on capability family 100 and raises
        # outright on sm_120 (``dsa_indexer_uses_fp4``), so this kit's FP8 cache
        # is structural, not a configuration choice. Kept as a guard because
        # that gate is vLLM's to widen: the overlay kernel is FP8 only, and
        # DeepGEMM's arch-12 MXFP4 kernel takes 32 and 64 alike, so there is no
        # gap to fill here either way.
        return None

    # The 4D packed page that ``kv_cache_as_quant_view`` builds. Its dim 1 *is*
    # the ``block_kv`` DeepGEMM would be handed, which is the whole problem.
    if kv_cache.dim() != 4 or kv_cache.shape[2] != 1:
        raise RuntimeError(
            "dsv41-sm12x-indexer: expected the 4D packed indexer page "
            f"[num_blocks, num_states, 1, head_dim + 4], got shape "
            f"{tuple(kv_cache.shape)}. The overlay's FP8 fallback reads that "
            "layout's value and scale offsets directly; refusing rather than "
            "guessing."
        )
    block_kv = kv_cache.shape[1]
    if block_kv == _DSV41_ARCH12_FP8_BLOCK_KV:
        # The geometry DeepGEMM's arch-12 FP8 kernel does have.
        return None
    if block_kv not in _DSV41_METADATA_BLOCK_KV:
        raise RuntimeError(
            f"dsv41-sm12x-indexer: indexer page holds {block_kv} states, which "
            "is outside the {32, 64} the indexer's own metadata builder admits "
            "on arch 12. That call runs before this one, so reaching here means "
            "an invariant this overlay was not written against. Refusing rather "
            "than dispatching to a kernel whose geometry was never reviewed."
        )
    if indices is not None:
        # ``indices`` selects DeepGEMM's varlen path, which is family-100 only
        # and is never set on arch 12 (``_supports_varlen_paged_mqa_logits()``
        # requires family 100, so ``decode_indices`` stays None). The overlay
        # kernel has no varlen mode, so a non-None value is unreviewed state.
        raise RuntimeError(
            "dsv41-sm12x-indexer: the FP8 fallback does not implement DeepGEMM's "
            "varlen (per-request ``indices``) path. This is unreachable while "
            "``_supports_varlen_paged_mqa_logits()`` is family-100 gated; if that "
            "gate has widened, this fallback needs re-review before it can serve "
            "arch 12."
        )

    # Imported here rather than at module scope: `vllm.utils.deep_gemm` is
    # imported by platform probes that must not acquire an overlay dependency,
    # and this path is arch-12-only anyway. A missing overlay module therefore
    # fails loudly at the first 32-state decode rather than at import time, and
    # the build-time suite pins its presence.
    from sm12x_mqa import fp8_paged_mqa_logits_sm12x

    return fp8_paged_mqa_logits_sm12x(
        q_values,
        kv_cache,
        weights,
        context_lens,
        block_tables,
        max_model_len,
    )


'''

# The anchor's own `def` line survives as the last line of the replacement: these
# edits may only insert lines, never rewrite one, and `Target.apply` enforces it.
_HOOK_NEW = _HOOK_CONSTS + _HOOK_BODY + _HOOK_OLD

# The hook call, inserted as the first statements of the function body. Anchored
# on the `_lazy_init()` + impl-guard pair, which is unique to this function
# (`get_paged_mqa_logits_metadata` has its own, different guard).
_CALL_OLD = """    _lazy_init()
    if _fp8_fp4_paged_mqa_logits_impl is None:
        return _missing()
"""

_CALL_NEW = """    # Arch-12 FP8 pages DeepGEMM has no kernel for: served by the overlay's
    # Triton fallback. Runs before `_lazy_init()` so it also holds where the
    # DeepGEMM implementation is absent, and returns None on every other
    # platform, dtype and geometry so DeepGEMM stays the default.
    _dsv41_logits = _dsv41_sm12x_fp8_paged_mqa_logits(
        q,
        kv_cache,
        weights,
        context_lens,
        block_tables,
        max_model_len,
        indices,
    )
    if _dsv41_logits is not None:
        return _dsv41_logits
    _lazy_init()
    if _fp8_fp4_paged_mqa_logits_impl is None:
        return _missing()
"""


def _is_ordered_subsequence(needle: list[str], haystack: list[str]) -> bool:
    """Whether every line of `needle` appears in `haystack`, in order."""
    it = iter(haystack)
    return all(any(line == other for other in it) for line in needle)


class Target:
    """One exact, purely additive replacement in an installed source file."""

    def __init__(self, name: str, rel: Path, old: str, new: str) -> None:
        self.name = name
        self.rel = rel
        self.old = old
        self.new = new
        self.delta = len(new.splitlines()) - len(old.splitlines())

    def classify(self, text: str) -> str:
        """Return 'patched', 'original', or fail closed on anything else."""
        if self.new in text:
            return "patched"
        if self.old in text:
            return "original"
        raise SystemExit(
            f"[{MARK}] {self.name}: unexpected anchor state — neither the stock "
            "nor the patched form is present. Upstream source drifted, an "
            "unrecognised partial edit is present, or this image was built from "
            "an already patched base. Re-derive the anchor before building."
        )

    def apply(self, path: Path) -> None:
        text = path.read_text()
        if text.count(self.old) != 1:
            raise SystemExit(
                f"[{MARK}] {self.name}: stock anchor not unique in {path} "
                f"(found {text.count(self.old)})"
            )
        if self.delta <= 0 or not _is_ordered_subsequence(
            self.old.splitlines(), self.new.splitlines()
        ):
            raise SystemExit(
                f"[{MARK}] {self.name}: the edit is not purely additive (delta "
                f"{self.delta:+d}); these replacements must only insert lines, so "
                "no other platform's behaviour can change"
            )
        patched = text.replace(self.old, self.new)
        if self.classify(patched) != "patched":
            raise SystemExit(f"[{MARK}] {self.name}: rewrite did not take; refusing")
        path.write_text(patched)
        print(f"[{MARK}] {self.name}: patched {path} ({self.delta:+d} lines)")


TARGETS = [
    # The hook must exist before the call site refers to it; Python resolves the
    # global at call time, so order here is for readability, not correctness.
    Target(name="SM12x FP8 fallback hook", rel=DEEP_GEMM_REL, old=_HOOK_OLD, new=_HOOK_NEW),
    Target(name="paged-MQA logits call site", rel=DEEP_GEMM_REL, old=_CALL_OLD, new=_CALL_NEW),
]


def vllm_root() -> Path:
    """Return the vLLM package directory to patch."""
    override = os.environ.get("DSV41_VLLM_ROOT")
    if override:
        return Path(override)
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit(f"[{MARK}] cannot locate the vllm package; set DSV41_VLLM_ROOT")
    return Path(next(iter(spec.submodule_search_locations)))


def main() -> int:
    root = vllm_root()

    # Preflight every target before writing any of them.
    planned: list[tuple[Target, Path, str]] = []
    for target in TARGETS:
        path = root / target.rel
        if not path.is_file():
            raise SystemExit(f"[{MARK}] {target.name}: missing file {path}")
        planned.append((target, path, target.classify(path.read_text())))

    if all(state == "patched" for *_, state in planned):
        print(f"[{MARK}] already applied to {root}")
        return 0

    for target, path, state in planned:
        if state == "patched":
            print(f"[{MARK}] {target.name}: already patched, leaving {path.name} alone")
            continue
        target.apply(path)

    print(f"[{MARK}] applied to {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
