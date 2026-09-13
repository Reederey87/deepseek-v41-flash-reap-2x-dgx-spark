#!/usr/bin/env python3
"""Make the DeepSeek-V4.1 sparse-MLA backends accept a 64-token kernel block on SM12x.

This closes the kit's boot blocker: at ``--block-size 128`` the engine dies during
``compile_or_warm_up_model`` on the sparse indexer's DeepGEMM paged-MQA call.

The failure
-----------
``v1/attention/backends/mla/indexer.py`` builds the decode metadata with

    get_paged_mqa_logits_metadata(seq_lens, self.kv_cache_spec.num_states, ...)

and ``num_states`` is ``block_size // tokens_per_state``
(``v1/kv_cache_interface.py``, ``get_num_kernel_states``). V4.1's compressed
caches use ``tokens_per_state = compress_ratio``, and this checkpoint carries
ratios 1 and 2, so ``--block-size 128`` makes ``num_states`` 128 for the ratio-1
layers and 64 for the ratio-2 ones.

DeepGEMM's SM120 paged-MQA path takes ``block_kv`` as a compile-time template
parameter and gates it *twice*, with different sets. The metadata builder
(``get_paged_mqa_logits_metadata``) asserts ``block_kv == 32 or block_kv == 64``;
the FP8 logits kernel itself (``fp8_fp4_paged_mqa_logits``) has no instantiation
for anything but 64. 128 is in neither set, so warmup aborts:

    RuntimeError: Assertion error (.../csrc/apis/attention.hpp:262):
    block_kv == 32 or block_kv == 64

The gap between the two sets — 32, which the metadata builder admits and the
logits kernel does not — is the second half of this blocker and is *not* closed
by the block size. It is closed by ``overlay/patch_sm12x_indexer.py``, which
routes a 32-state FP8 page to this kit's Triton fallback. Reading the assert
above as the whole constraint is what let an earlier version of the pre-flight
call ``--block-size 64`` servable when the ratio-2 pages still had no kernel.

This is vLLM issue **#53635** (open, 2026-08-24). Its own analysis rules out the
alternatives: PR #53522's ``num_states`` gate removes the assert but leaves
``schedule_metadata`` zero-filled while the decode path still launches
``fp8_fp4_paged_mqa_logits``, which then reads a garbage schedule and takes an
async illegal memory access — a sticky ``CUDA_ERROR_ILLEGAL_ADDRESS`` that
surfaces in an unrelated kernel. There is no non-DeepGEMM CUDA escape hatch
either: ``model_executor/layers/sparse_attn_indexer.py`` raises outright when
``is_cuda() and not has_deep_gemm()``. That gate still stands — the overlay's
fallback is dispatched from *inside* ``vllm/utils/deep_gemm.py``, so it does not
remove the DeepGEMM requirement, it only replaces the logits call.

Why 64 is the fix, not a workaround
-----------------------------------
``block_kv`` is the number of states in one *physical* page, so the manager block
size has to land the compressed pool on 32 or 64. V4.1's ratios are 1 and 2, so
``--block-size 64`` gives exactly {64, 32} — both in the *metadata builder's* set,
and one page per kernel block for each ratio. The 32 additionally needs
``overlay/patch_sm12x_indexer.py``, because it is outside the FP8 logits kernel's
set. 128 gives {128, 64}: 128 is outside the metadata set, so that geometry cannot
boot at all. (V4-Flash is stuck here: its ratio is 128, so ``num_states`` is
``block_size / 128`` and no usable block size reaches 32 or 64.)

Changing the block size alone is not enough, because the kernel block size must
also be *negotiated*: ``prepare_kernel_block_sizes`` intersects the advertised
lists of every backend sharing a KV cache group, and the V4.1 MLA and indexer
specs are uniform with each other (``MLAAttentionSpec.is_uniform_with_collection``),
so they land in one group and ``select_common_block_size`` runs over both. On
SM121 all three V4.1 backends currently advertise 128:

* ``models/deepseek_v4_1/nvidia/flashinfer_sparse.py`` — ``[128]`` (hardcoded;
  this is the class SM12 actually uses)
* ``models/deepseek_v4_1/sparse_mla.py`` — ``[64 if family(90) else 128]``
* ``v1/attention/backends/mla/indexer.py`` — ``[64 if family(90) else 128]``

so ``select_common_block_size(64, ...)`` would raise ``ValueError: No common
block size for 64`` at engine init. Each gains a capability-family-120 clause
returning ``[64]``.

Family 120 is ``major == 12`` (``is_device_capability_family`` divides by ten),
so GB10's (12, 1) matches. Every other family keeps its exact current value:
the clause is added, nothing is rewritten. This follows upstream PR **#53425**
("SM12x FlashInfer sparse MLA kernel block size 64"), which returns ``[64]`` on
capability family 120 and leaves SM100 at its larger block.

Scope: three exact method-body replacements, one per file. Idempotent and
fail-closed: every target is classified before any is written, so a drifted
anchor or an unrecognised mixed state exits non-zero without touching the tree.
Each edit is purely additive — every stock line survives, verified as an ordered
subsequence — so the only behaviour that can change is the family-120 branch.
Rebuilding on top of an *already patched* image is refused rather than
half-applied; the base image is the stock V4.1 image, so the anchors are stock
there.

See `RESEARCH.md` §11.9.15.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

MARK = "dsv41-v41-kernel-block-size"

FLASHINFER_SPARSE_REL = Path("models/deepseek_v4_1/nvidia/flashinfer_sparse.py")
SPARSE_MLA_REL = Path("models/deepseek_v4_1/sparse_mla.py")
INDEXER_REL = Path("v1/attention/backends/mla/indexer.py")

# The rationale is repeated in each file rather than factored out: these are
# three independently readable edits to upstream source, and a reader of any one
# of them should not have to find the other two.
_WHY = """        # SM12x runs the FlashInfer SM120 sparse decoder, whose pages are 64
        # tokens, and DeepGEMM's paged MQA kernel, which is templated for
        # block_kv 32 or 64. A 128-token kernel block would make the ratio-1
        # compressed pool 128 states per page, which DeepGEMM cannot serve
        # (vLLM #53635). 64 gives 64 and 32, one page per kernel block.
"""

# `flashinfer_sparse.py` does not import `current_platform` at module scope, and
# a function-local import keeps this edit a single method-body replacement. The
# file already uses function-local imports for the same reason.
_FLASHINFER_SPARSE_OLD = """    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [128]
"""
_FLASHINFER_SPARSE_NEW = (
    """    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        from vllm.platforms import current_platform

"""
    + _WHY
    + """        if current_platform.is_device_capability_family(120):
            return [64]
        return [128]
"""
)

# The base and indexer backends already import `current_platform` and already
# special-case family 90, so the family-120 clause slots in ahead of it and the
# existing expression is left byte-identical.
_FAMILY_OLD = """    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [64 if current_platform.is_device_capability_family(90) else 128]
"""
_FAMILY_NEW = (
    """    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
"""
    + _WHY
    + """        if current_platform.is_device_capability_family(120):
            return [64]
        return [64 if current_platform.is_device_capability_family(90) else 128]
"""
)


def _is_ordered_subsequence(needle: list[str], haystack: list[str]) -> bool:
    """Whether every line of `needle` appears in `haystack`, in order."""
    it = iter(haystack)
    return all(any(line == other for other in it) for line in needle)


class Target:
    """One exact method-body replacement.

    Classification is on the whole `old`/`new` block, and the two are disjoint:
    `old` is the method header immediately followed by its old return, while
    `new` inserts lines between them, so neither is a substring of the other.
    That is what makes the block test unambiguous even though the old return
    statement survives verbatim inside the new one.
    """

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
            "nor the patched form of this method is present. Upstream source "
            "drifted, an unrecognised partial edit is present, or this image was "
            "built from an already patched base. Re-derive the anchor before "
            "building."
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
    Target(
        name="FlashInfer sparse-MLA backend",
        rel=FLASHINFER_SPARSE_REL,
        old=_FLASHINFER_SPARSE_OLD,
        new=_FLASHINFER_SPARSE_NEW,
    ),
    Target(
        name="sparse-MLA backend base",
        rel=SPARSE_MLA_REL,
        old=_FAMILY_OLD,
        new=_FAMILY_NEW,
    ),
    Target(
        name="V4 indexer backend",
        rel=INDEXER_REL,
        old=_FAMILY_OLD,
        new=_FAMILY_NEW,
    ),
]


def vllm_root() -> Path:
    """Return the vLLM package directory to patch."""
    override = os.environ.get("DSV41_VLLM_ROOT")
    if override:
        return Path(override)
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit(
            f"[{MARK}] cannot locate the vllm package; set DSV41_VLLM_ROOT"
        )
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
