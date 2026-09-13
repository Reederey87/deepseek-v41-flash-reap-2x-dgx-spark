#!/usr/bin/env python3
"""Fail-closed pre-flight for DeepSeek-V4.1 sparse MLA on SM120 (GB10).

Three things are checked, because they fail in three different places and only
one of them is covered by vLLM's own init-time gate.

1. **The decode dispatch pair.** FlashInfer's SM120 sparse-MLA decode switch is
   keyed on `(num_heads, topk)`, where `topk = indices.shape[-1]` is taken from
   the *main* index tensor (`flashinfer/mla/_sparse_mla_sm120.py`, `topk =
   indices.shape[-1]`). Our vLLM passes the SWA segment as that tensor and the
   compressed segment separately as `extra_sparse_indices`
   (`models/deepseek_v4_1/nvidia/flashinfer_sparse.py` decode and prefill), so the
   key is `(num_attention_heads // tp, <SWA index width>)` — 32 and 128 at TP=2
   *only when the SWA rows are `sliding_window` wide*.

   That width is not always `sliding_window`. `attention.py` and `sparse_swa.py`
   size the prefill SWA rows at `sliding_window + vision_max_n_token`, so a
   checkpoint whose config declares `vision_config.max_image_tokens = 1024`
   produces 128 + 1024 = **1152**-wide rows, and that whole table is what
   reaches the kernel as `indices`. The dispatch sets are fixed buckets —
   {128, 192, 256, 512, 1024} for decode, the same plus 2048 for the single
   prefill dispatcher, and `dispatch_dsv4_dual` (the path this tree takes)
   requires exactly 128 — and 1152 is in none of them, so the engine dies at the
   first attention call. There is no valid width to round 1152 to.

   The widening is deliberate for vision (the image span is attended
   bidirectionally, and vLLM sizes the span by the per-image token cap), so the
   fix is to collapse the *budget*, not the layer count: this kit's `start.sh`
   passes `--hf-overrides '{"vision_max_n_token": 0}'` whenever
   `LANGUAGE_MODEL_ONLY=1`, which restores the rows to `sliding_window` — the
   width upstream vLLM allocates. Note that `{"vision_n_layers": 0}` also
   collapses the width but is **wrong**: the MoE gate creates its `bias_vl`
   parameter only when `vision_n_layers > 0`, so zeroing it makes the loader
   raise `KeyError: 'layers.0.ffn.gate.bias_vl'` on this checkpoint's weights.
   Pass the effective budget with `--max-image-tokens` so this check verifies
   the width the engine will really use, not the checkpoint's declared one.

   The recipe for this checkpoint also lists `(32, 640)` and `(32, 1152)`, the
   *combined* SWA+compressed widths (`index_topk * ratio + window`). Those apply
   only to the single-index call shape, which the recipe's own bind-mounted
   `flashinfer_sparse.py` can use and our tree does not — the prefill entry
   takes `dispatch_dsv4_dual` when `extra_KV_cache != nullptr`, and that
   function requires `topk == 128` exactly. They are reported as candidates, not
   required.

2. **The compressed pool's page block, read from the installed prefill source.**
   FlashInfer's dual-cache prefill dispatcher accepts a compressed cache only
   when `extra_page_block_size` is one of the values it tests, and that value is
   read from the compressed KV tensor's own page block. In vLLM that pool's page
   block is `block_size // compress_ratio` (`v1/kv_cache_interface.py`,
   `get_num_kernel_states`), and V4.1 uses both ratio 1 and ratio 2, so the two
   ratios want different block sizes.

   `--block-size 64` is the checkpoint's native SM12x geometry, and the same
   number has to satisfy DeepGEMM: the indexer passes `num_states` to
   `get_paged_mqa_logits_metadata`, and DeepGEMM's SM120 paged-MQA kernel is
   templated for `block_kv` 32 or 64 (vLLM #53635). 64 gives {64, 32} — both
   supported, one page per kernel block; 128 gives {128, 64} and dies at warmup.
   That is why `start.sh` pins 64 and why the backend patch exists at all.

   Stock FlashInfer instantiates only 64 and 2 (the 2 is V4's ratio-128 page),
   so the ratio-2 page of 32 needs `overlay/patch_flashinfer_prefill_pbsx.py`.
   The accepted set is **parsed from the installed `.cu`** rather than assumed:
   this check must fail if the image was built without the patch, and pass once
   it is in place.

   The two dispatch paths are then modelled **separately**, because they fail
   differently and a union of instantiated values hides a half-applied patch.
   Stock's full-tile branch ends in a bare `else { PBSX(2) }`, so admitting a
   value without adding that branch runs the 2-token kernel on it — wrong
   addresses, not a clean failure. The patched ladder ends in an explicit
   `else { return false; }` instead, and the non-full-tile ladder ends in
   `return false`, so a missing entry there is a clean but fatal boot failure —
   and it is the branch this tree takes. Both must be identity dispatches for
   every page block the config needs.

3. **The AOT trap.** `JitSpec.build_and_load()` returns the prebuilt AOT `.so`
   whenever it exists, so a patched `.cu` is never compiled. Whenever the config
   needs an instantiation that stock FlashInfer lacks, an AOT binary still being
   present means the patch is being ignored; the image masks it
   (`overlay/mask_flashinfer_aot.py`) and `scripts/build_flashinfer.sh` compiles
   the patched source into the persistent cache.

Usage:
    check_flashinfer_dsv41.py --config <ckpt>/config.json --tp 2 [--block-size 64]
                              [--max-image-tokens <effective>]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
from collections.abc import Iterable
from pathlib import Path

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_MISSING = 3
EXIT_NO_FLASHINFER = 4
EXIT_PAGE_BLOCK = 5
EXIT_INDEXER_BLOCK_KV = 6

# vLLM pads q heads to one of these before the sparse-MLA kernel sees them.
SUPPORTED_Q_HEADS = (8, 16, 32, 64, 128)

# The compressed page block stock FlashInfer instantiates. Anything else needs
# the overlay patch; the accepted set is read from the source, not from here.
STOCK_EXTRA_PAGE_BLOCKS = frozenset({64, 2})

# DeepGEMM gates the paged-MQA path twice, and the two sets are NOT the same.
# Conflating them is what once made this check bless a 32-state page that boots,
# warms up, and then dies on the first long prompt. Hardware receipts for both
# sets: for_review/v41-block-size/DEEPGEMM-ARCH12-FALLBACK-PROBE.txt.
#
# The *metadata* builder is the first gate, and the indexer calls it while
# building decode metadata — before any logits call. On arch 12 it admits 32 or
# 64 states per page (vLLM #53635; observed here as `Assertion error
# (.../csrc/apis/attention.hpp:262): block_kv == 32 or block_kv == 64`). A page
# outside this set aborts warmup outright and nothing downstream can rescue it.
DEEPGEMM_METADATA_BLOCK_KV = frozenset({32, 64})

# The *logits* kernel is the second gate and it is stricter: on arch 12 it
# admits `block_kv == 64` exactly for the FP8 cache (32 and 64 for MXFP4), which
# is `Assertion error (.../csrc/apis/attention.hpp:320)`. A page inside the
# metadata set but outside this one gets all the way to decode and then kills
# the engine. `overlay/patch_sm12x_indexer.py` installs a Triton fallback that
# serves exactly those pages, so whether they are servable depends on whether
# that fallback is actually installed — which is checked, not assumed.
DEEPGEMM_FP8_LOGITS_BLOCK_KV = frozenset({64})

PREFILL_REL = Path("data/csrc/sparse_mla_sm120_prefill.cu")

# Every `extra_page_block_size == N` comparison, wherever it appears.
_ACCEPT_RE = re.compile(r"extra_page_block_size\s*==\s*(\d+)")

# One dispatch branch: a header followed by exactly one macro call. Matching the
# pair — not just the call — is what makes the two ladders distinguishable and
# lets each be modelled as an ordered chain.
_BRANCH_RE = re.compile(
    r"(?:if|else\s+if)\s*\(\s*extra_page_block_size\s*==\s*(\d+)\s*\)\s*\{\s*"
    r"(DISPATCH_(FULLTILE_)?BY_NH_PBSX)\((\d+)\)\s*;\s*\}"
    r"|else\s*\{\s*(DISPATCH_(FULLTILE_)?BY_NH_PBSX)\((\d+)\)\s*;\s*\}",
    re.DOTALL,
)

# A `(condition, instantiated page block)` chain. `condition` is None for a bare
# `else`, which matches anything the earlier entries did not.
Chain = list[tuple[int | None, int]]

# The prefill path this tree takes. With the SWA segment as `indices` and the
# compressed segment as `extra_sparse_indices`, `extra_KV_cache != nullptr`, so
# `sparse_mla_prefill_dispatch` calls `dispatch_dsv4_dual` — whose guard is
# `topk == 128` and whose only instantiation is `DISPATCH_DUAL_MG_FULLTILE(NH,
# 128, ...)`. The widths `dispatch_dsv4_single` carries do not apply.
DUAL_REQUIRED_TOPK = 128

# The `dispatch_dsv4_single` width ladder: `if (topk == N) DISPATCH_BY_NH_CM(..)`
# followed by zero or more `else if` arms. Anchored on the macro name, which is
# unique to that function, so the dual dispatcher's `topk == 128` guard — which
# belongs to a different contract — is not mistaken for a member of the set.
_SINGLE_TOPK_RE = re.compile(
    r"if\s*\(\s*topk\s*==\s*\d+\s*\)\s*DISPATCH_BY_NH_CM\s*\([^;]*?\)\s*;"
    r"(?:\s*else\s+if\s*\(\s*topk\s*==\s*\d+\s*\)\s*DISPATCH_BY_NH_CM\s*\([^;]*?\)\s*;)*"
)
_TOPK_LITERAL_RE = re.compile(r"topk\s*==\s*(\d+)")


def _text_config(config: dict) -> dict:
    return config.get("text_config", config) or {}


def pad_q_heads(num_heads: int) -> int:
    """The head count the kernel actually sees (vLLM `_pad_to_supported_q_heads`)."""
    for supported in SUPPORTED_Q_HEADS:
        if num_heads <= supported:
            return supported
    raise ValueError(
        f"{num_heads} heads exceeds the sparse-MLA kernel's supported set "
        f"{SUPPORTED_Q_HEADS}"
    )


def config_vision_n_layers(config: dict) -> int:
    """The checkpoint's declared `vision_config.num_hidden_layers` (0 if absent)."""
    vision = config.get("vision_config") or {}
    if not isinstance(vision, dict):
        return 0
    value = vision.get("num_hidden_layers", 0)
    return int(value) if isinstance(value, int) and value > 0 else 0


def config_vision_max_image_tokens(config: dict) -> int:
    """The checkpoint's `vision_config.max_image_tokens` (1024 default, 0 if none)."""
    vision = config.get("vision_config") or {}
    if not isinstance(vision, dict):
        return 0
    value = vision.get("max_image_tokens", 1024)
    return int(value) if isinstance(value, int) and value > 0 else 0


def declared_image_budget(config: dict) -> int:
    """The per-image token budget the checkpoint declares (0 for a text model)."""
    if config_vision_n_layers(config) == 0:
        return 0
    return config_vision_max_image_tokens(config)


def swa_index_width(config: dict, max_image_tokens: int | None = None) -> tuple[int, str]:
    """`(width, why)` for the SWA index rows the sparse-MLA kernels will see.

    `attention.py` and `sparse_swa.py` size the prefill SWA rows at
    `sliding_window + vision_max_n_token`, so the width follows the *effective*
    per-image token budget — not the checkpoint's declared one. This kit serves
    text-only and collapses that budget to 0 with
    `--hf-overrides '{"vision_max_n_token": 0}'`, which restores the rows to
    `sliding_window`.
    """
    text = _text_config(config)
    window = int(text.get("sliding_window", 0))
    if window <= 0:
        raise ValueError("config has no sliding_window")
    declared = declared_image_budget(config)
    budget = declared if max_image_tokens is None else max_image_tokens
    if budget < 0:
        raise ValueError(f"max_image_tokens must be >= 0, got {budget}")
    if budget == 0:
        if declared == 0:
            return window, f"window {window} (the checkpoint declares no image span)"
        return window, (
            f"window {window} (effective image budget 0, overriding the declared "
            f"{declared}; the prefill SWA rows are not widened)"
        )
    return window + budget, (
        f"window {window} + max_image_tokens {budget} = {window + budget} "
        "(an image span widens the prefill SWA rows)"
    )


def required_pairs(
    config: dict, tp_size: int, max_image_tokens: int | None = None
) -> dict[tuple[int, int], str]:
    """The `(num_heads, topk)` pairs the decode dispatch must carry.

    `topk` is the main index width, which is the SWA row width — `sliding_window`
    on the text-only posture, and `sliding_window + max_image_tokens` whenever an
    image span can appear.
    """
    text = _text_config(config)
    heads = int(text.get("num_attention_heads", 0))
    if heads <= 0:
        raise ValueError("config has no num_attention_heads")
    if tp_size <= 0:
        raise ValueError(f"tp_size must be positive, got {tp_size}")
    if heads % tp_size:
        raise ValueError(f"num_attention_heads={heads} is not divisible by tp={tp_size}")
    width, why = swa_index_width(config, max_image_tokens)
    return {(pad_q_heads(heads // tp_size), width): f"main/SWA index width: {why}"}


def candidate_combined_pairs(config: dict, tp_size: int) -> dict[tuple[int, int], str]:
    """The combined-width pairs the recipe's patch adds, if the call shape is combined.

    Reported for information only. `index_topk * compress_ratio + sliding_window`.
    """
    text = _text_config(config)
    heads = int(text.get("num_attention_heads", 0)) // tp_size
    window = int(text.get("sliding_window", 0))
    index_topk = int(text.get("index_topk", 0))
    ratios = sorted({int(r) for r in (text.get("compress_ratios") or [])})
    out: dict[tuple[int, int], str] = {}
    if index_topk <= 0:
        return out
    for ratio in ratios:
        if ratio <= 0:
            continue
        out[(pad_q_heads(heads), index_topk * ratio + window)] = (
            f"combined width at compress_ratio {ratio} "
            f"({index_topk} x {ratio} + {window})"
        )
    return out


def prefill_source() -> Path | None:
    """The installed SM120 prefill dispatcher, or None when it is not present."""
    override = os.environ.get("DSV41_FLASHINFER_ROOT")
    if override:
        candidate = Path(override) / PREFILL_REL
        return candidate if candidate.is_file() else None
    spec = importlib.util.find_spec("flashinfer")
    if spec is None or not spec.submodule_search_locations:
        return None
    candidate = Path(next(iter(spec.submodule_search_locations))) / PREFILL_REL
    return candidate if candidate.is_file() else None


def _ladders(cu_text: str) -> tuple[Chain, Chain]:
    """The `(full-tile, non-full-tile)` dispatch chains, in source order."""
    full: Chain = []
    plain: Chain = []
    for match in _BRANCH_RE.finditer(cu_text):
        if match.group(2) is not None:
            cond, macro, value = match.group(1), match.group(2), match.group(4)
        else:
            cond, macro, value = None, match.group(5), match.group(7)
        entry = (int(cond) if cond is not None else None, int(value))
        (full if "FULLTILE" in macro else plain).append(entry)
    return full, plain


def _first_fulltile_offset(cu_text: str) -> int | None:
    """Where the full-tile chain starts, i.e. where the acceptance test ends."""
    for match in _BRANCH_RE.finditer(cu_text):
        if "FULLTILE" in (match.group(2) or match.group(5)):
            return match.start()
    return None


def _dispatch(chain: Chain, value: int) -> int | None:
    """The page block `chain` instantiates for `value`; None if it falls through."""
    for cond, instantiated in chain:
        if cond is None or cond == value:
            return instantiated
    return None


def acceptance_page_blocks(cu_text: str) -> frozenset[int]:
    """The compressed page blocks the full-tile acceptance test admits.

    Read from the acceptance test alone, not from the whole file. The branch
    headers below it compare the same variable, so a union would report a value
    as accepted when only a branch names it — and that is exactly how a
    half-applied patch hides (adding a branch without widening the test, or the
    reverse, is invisible to a union).
    """
    at = _first_fulltile_offset(cu_text)
    if at is None:
        return frozenset()
    head = cu_text[:at]
    anchor = head.rfind("topk == 128")
    if anchor == -1:
        return frozenset()
    open_paren = head.rfind("if (", 0, anchor)
    if open_paren == -1:
        return frozenset()
    return frozenset(int(n) for n in _ACCEPT_RE.findall(head[open_paren:]))


def dispatch_defects(cu_text: str, needed: Iterable[int]) -> list[tuple[int, str]]:
    """Why each `needed` page block cannot be served, as `(page_block, reason)`.

    Two independent requirements, because the dispatcher has two paths and they
    fail differently:

    * **Full-tile** must be an identity dispatch for every page block the
      acceptance test admits. Stock ends that chain in a bare `else { PBSX(2) }`,
      so admitting 128 without adding a branch runs the 2-token kernel on
      128-granular slot ids — wrong addresses, not a clean failure. This is the
      dangerous case.
    * **Non-full-tile** must instantiate the page block it is given. Falling off
      its end is `return false`: a clean failure, but still a boot failure, and
      it is the branch this tree actually takes because vLLM passes non-null
      `extra_sparse_topk_lens`.

    Comparing the two *unions* of instantiated values catches neither partial
    application, because each ladder covers for the other.
    """
    accepted = acceptance_page_blocks(cu_text)
    full, plain = _ladders(cu_text)
    defects: list[tuple[int, str]] = []
    for value in sorted(set(needed)):
        if value in accepted:
            got = _dispatch(full, value)
            if got != value:
                ran = "nothing" if got is None else f"PBSX({got})"
                defects.append(
                    (
                        value,
                        f"the full-tile branch runs {ran} instead of "
                        f"PBSX({value}) while the acceptance test admits it, so "
                        f"{value}-granular slot ids would be divided by the "
                        "wrong page block",
                    )
                )
        got = _dispatch(plain, value)
        if got != value:
            did = "returns false" if got is None else f"dispatches PBSX({got})"
            defects.append(
                (
                    value,
                    f"the non-full-tile ladder {did} instead of PBSX({value}); "
                    "compressed prefill would fail on this checkpoint's live "
                    "call shape",
                )
            )
    return defects


def completeness_defects(cu_text: str) -> list[str]:
    """Why the accepted set and the two ladders do not enumerate the same values.

    The patch's three edits are coterminous by design: the acceptance test is the
    guard, and each ladder enumerates exactly what the guard admits, with nothing
    extra. Stock satisfies this trivially — `{64, 2}` in all three — so the check
    is a statement about the *patch*, and it is the only thing that catches a
    partial application whose reachable path happens to work.

    That case is real and `dispatch_defects` cannot see it: applying edit 3 alone
    (the non-full-tile ladder) leaves the acceptance test at `{64, 2}` while the
    plain ladder carries `{64, 2, 128, 32}`. Every page block this checkpoint
    needs is then served correctly — the guard rejects 32, so the full-tile path
    is never taken for it — and a per-value reachability check returns no defect.
    The image is still only one third patched, which is the state the verifier
    exists to refuse.
    """
    accepted = acceptance_page_blocks(cu_text)
    full_set = frozenset(value for _cond, value in _ladders(cu_text)[0])
    plain_set = frozenset(value for _cond, value in _ladders(cu_text)[1])
    if accepted == full_set == plain_set:
        return []
    return [
        "the acceptance test and the two dispatch ladders do not enumerate the "
        f"same compressed page blocks: acceptance {sorted(accepted)}, full-tile "
        f"{sorted(full_set)}, non-full-tile {sorted(plain_set)}. The patch's "
        "three edits are coterminous by design, so a disagreement means only "
        "some of them landed. Re-apply "
        "overlay/patch_flashinfer_prefill_pbsx.py to the stock base image."
    ]


def prefill_page_blocks(cu_text: str) -> tuple[frozenset[int], frozenset[int]]:
    """`(accepted, instantiated)` compressed page blocks, as a summary.

    `accepted` is what the acceptance test admits; `instantiated` is the union
    of both ladders' calls. The union is reported for a human, not used as the
    verdict — `dispatch_defects` is what decides, because a union cannot see a
    half-applied patch.
    """
    instantiated = frozenset(
        value for chain in _ladders(cu_text) for _cond, value in chain
    )
    return acceptance_page_blocks(cu_text), instantiated


def prefill_single_topk_values(cu_text: str) -> frozenset[int]:
    """The `topk` widths `dispatch_dsv4_single` instantiates, read from the `.cu`.

    Informational: this tree's call shape does not reach that function, so the
    set is reported to explain *why* a wider width does not help.
    """
    out: set[int] = set()
    for match in _SINGLE_TOPK_RE.finditer(cu_text):
        out.update(int(n) for n in _TOPK_LITERAL_RE.findall(match.group(0)))
    return frozenset(out)


def prefill_width_defect(width: int, cu_text: str) -> str | None:
    """Why the prefill path cannot serve SWA rows `width` wide, or None if it can."""
    if width == DUAL_REQUIRED_TOPK:
        return None
    single = prefill_single_topk_values(cu_text)
    carried = (
        f"the single dispatcher carries {sorted(single)}, but that function is "
        "only reached when no compressed segment is passed"
        if single
        else "the single dispatcher's width set could not be read"
    )
    return (
        f"the prefill path calls dispatch_dsv4_dual (extra_KV_cache != nullptr), "
        f"which requires topk == {DUAL_REQUIRED_TOPK} exactly, but the SWA rows "
        f"are {width} wide; {carried}"
    )


def compressed_page_blocks(config: dict, block_size: int) -> list[tuple[int, int]]:
    """`(compress_ratio, compressed page block)` for each compressed ratio.

    The page block is the compressed pool's states per page, which is what
    FlashInfer reads back out of the cache tensor's shape.
    """
    ratios = sorted(
        {int(r) for r in (_text_config(config).get("compress_ratios") or [])}
    )
    return [(r, max(1, block_size // r)) for r in ratios if r > 0]


def indexer_block_kv_defects(
    rows: Iterable[tuple[int, int]],
    fallback_active: bool,
) -> list[tuple[int, int, str]]:
    """`(compress_ratio, page_block, gate)` triples DeepGEMM cannot serve.

    The page block *is* the `num_states` the indexer hands DeepGEMM as
    `block_kv`, at both gates, so this is the check that fails fastest and most
    loudly. Which gate fails is reported because the two fail very differently:
    a metadata-set miss aborts during `compile_or_warm_up_model`, after the
    weights are loaded, while a logits-set miss survives warmup and takes the
    engine down on the first long prompt.

    `fallback_active` is whether `overlay/patch_sm12x_indexer.py` is installed
    in the tree under test. It only lifts the logits gate: the fallback replaces
    the logits call, not the metadata build, so a page outside the metadata set
    is unservable either way.
    """
    defects: list[tuple[int, int, str]] = []
    for ratio, page_block in rows:
        if page_block not in DEEPGEMM_METADATA_BLOCK_KV:
            defects.append((ratio, page_block, "metadata"))
        elif not fallback_active and page_block not in DEEPGEMM_FP8_LOGITS_BLOCK_KV:
            defects.append((ratio, page_block, "logits"))
    return defects


def sm12x_fallback_installed() -> bool:
    """Whether the overlay's SM12x FP8 paged-MQA logits fallback is wired in.

    Two additive edits install it, and both must be present: the helper
    definition, and a reference to it inside the wrapper. Checking only that the
    helper exists accepts a tree carrying the first edit alone, where every
    32-state request still reaches DeepGEMM and aborts exactly as before — the
    same false acceptance this split exists to remove.

    The second half is read off the wrapper's code object rather than its source
    text, so a rename or a reformat cannot satisfy it by accident. It is a
    *wiring marker*, not proof of execution: a wrapper that merely referenced the
    helper without calling it would pass. That is acceptable because the wrapper's
    body is a reviewed, fixed artefact whose anchors the patch pins and whose
    content `tests/test_sm12x_indexer.py` checks at build time. This check exists
    to catch a *missing* edit, which it does.

    Read from the installed module rather than inferred from this kit's own
    source: the fallback is precisely what makes a non-64 FP8 page servable.

    This covers the dispatch hook and its wiring. The other half — that
    `sm12x_mqa`, which the hook imports at call time, is present and parses — is
    pinned at build time by `tests/test_sm12x_indexer.py`, which is the right
    place for it: this script runs from the overlay directory, so importing
    `sm12x_mqa` here would test *this* script's import path rather than the
    served process's.
    """
    try:
        from vllm.utils.deep_gemm import (
            _dsv41_sm12x_fp8_paged_mqa_logits,
            fp8_fp4_paged_mqa_logits,
        )
    except Exception:  # noqa: BLE001 — absent, unimportable, or no vLLM here
        return False
    if not callable(_dsv41_sm12x_fp8_paged_mqa_logits):
        return False
    code = getattr(fp8_fp4_paged_mqa_logits, "__code__", None)
    if code is None:
        return False
    return "_dsv41_sm12x_fp8_paged_mqa_logits" in code.co_names


def load_dispatch():
    """FlashInfer's decode dispatch table, or None when it is unavailable."""
    try:
        from flashinfer.mla import _sparse_mla_sm120 as mod
    except Exception:
        return None, None
    return (
        getattr(mod, "_DECODE_DSV4_DISPATCH", None),
        getattr(mod, "_DECODE_DSV4_PAGE_BLOCK_SIZE", None),
    )


def aot_status() -> tuple[bool | None, str]:
    """(is_aot, path) for the sparse_mla_sm120 module; None when unreadable.

    Unreadable is distinct from "not AOT": this is the introspection that says
    whether a patched `.cu` would be compiled at all, so a failed lookup cannot
    be reported as a clean bill of health. The caller fails closed on None when
    the config needs a page block the stock binary cannot serve.
    """
    try:
        from flashinfer.mla._sparse_mla_sm120 import gen_sparse_mla_sm120_module

        spec = gen_sparse_mla_sm120_module()
        return bool(spec.is_aot), str(spec.aot_path)
    except Exception as exc:  # noqa: BLE001
        return None, f"unreadable ({exc!r})"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument(
        "--block-size",
        type=int,
        default=64,
        help=(
            "vLLM --block-size; drives the compressed pool's page block. 64 is the "
            "only value that also lands every ratio on DeepGEMM's block_kv set "
            "{32, 64}, so it is what start.sh pins"
        ),
    )
    ap.add_argument(
        "--expect-page",
        type=int,
        default=64,
        help="required SWA page size; the SM120 DSV4 decode kernels require 64",
    )
    ap.add_argument(
        "--max-image-tokens",
        type=int,
        default=None,
        help=(
            "effective per-image token budget the engine will run with (0 for the "
            "text-only posture start.sh serves via --hf-overrides "
            "'{\"vision_max_n_token\": 0}'); default: the checkpoint's declared value"
        ),
    )
    ap.add_argument(
        "--allow-missing",
        action="store_true",
        help="report a missing decode pair but exit 0",
    )
    args = ap.parse_args()

    if not args.config.is_file():
        print(f"ERROR: no config at {args.config}", file=sys.stderr)
        return EXIT_USAGE
    config = json.loads(args.config.read_text())
    declared_budget = declared_image_budget(config)
    effective_budget = (
        declared_budget if args.max_image_tokens is None else args.max_image_tokens
    )
    try:
        pairs = required_pairs(config, args.tp, effective_budget)
        candidates = candidate_combined_pairs(config, args.tp)
        width, _why = swa_index_width(config, effective_budget)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_USAGE

    print(f"DeepSeek-V4.1 sparse MLA pre-flight (tp={args.tp}, block-size={args.block_size})")
    if effective_budget != declared_budget:
        print(
            f"  effective image-token budget={effective_budget} "
            f"(checkpoint declares {declared_budget}); SWA index width = {width}"
        )
    else:
        print(f"  image-token budget={declared_budget}; SWA index width = {width}")
    print("  required decode dispatch keys (main index width):")
    for (heads, topk), why in sorted(pairs.items()):
        print(f"    (num_heads={heads}, topk={topk})  <- {why}")
    if candidates:
        print("  candidate combined-width keys (single-index call shape only; not this tree):")
        for (heads, topk), why in sorted(candidates.items()):
            print(f"    (num_heads={heads}, topk={topk})  <- {why}")

    dispatch, page = load_dispatch()
    if dispatch is None:
        print(
            "ERROR: flashinfer.mla._sparse_mla_sm120._DECODE_DSV4_DISPATCH is not "
            "readable; cannot verify the sparse-MLA instantiations",
            file=sys.stderr,
        )
        return EXIT_NO_FLASHINFER

    is_aot, aot_path = aot_status()
    print(f"  AOT binary in use: {'unknown' if is_aot is None else is_aot}  ({aot_path})")
    if page is not None and int(page) != int(args.expect_page):
        print(
            f"  WARNING: FlashInfer expects page_block_size={page}, not "
            f"{args.expect_page}; check the SWA page patch (RESEARCH.md 11.6)",
            file=sys.stderr,
        )

    have = {(int(h), int(k)) for (h, k) in dispatch}
    missing = sorted(p for p in pairs if p not in have)
    if missing:
        print("\nMISSING decode specializations:", file=sys.stderr)
        for heads, topk in missing:
            print(f"  (num_heads={heads}, topk={topk})  <- {pairs[(heads, topk)]}", file=sys.stderr)
        print(
            "\nThe engine would fail at the first decode of those layers "
            '("SM120 sparse-MLA has no decode kernel for this shape").',
            file=sys.stderr,
        )

    source = prefill_source()
    if source is None:
        print(
            f"\nERROR: cannot read the installed prefill dispatcher "
            f"(flashinfer/{PREFILL_REL}); the compressed page block cannot be "
            "verified",
            file=sys.stderr,
        )
        return EXIT_PAGE_BLOCK
    cu_text = source.read_text()
    accepted, instantiated = prefill_page_blocks(cu_text)
    rows = compressed_page_blocks(config, args.block_size)
    defects = dispatch_defects(cu_text, [page_block for _r, page_block in rows])
    unservable = {value for value, _reason in defects}
    incomplete = completeness_defects(cu_text)
    print(f"  prefill source: {source}")
    print(f"    accepts compressed page blocks: {sorted(accepted)}")
    print(f"    instantiates (both ladders):    {sorted(instantiated)}")
    print(
        f"    acceptance and both ladders agree: "
        f"{'yes' if not incomplete else 'NO'}"
    )
    print(f"    single dispatcher topk widths:  {sorted(prefill_single_topk_values(cu_text))}")

    # The SWA row width is the *first* thing to fail on a boot, before any page
    # block matters: it keys the decode dispatch and it is the dual dispatcher's
    # `topk == 128` guard. Report it ahead of the page-block verdict so the
    # reader sees the actual blocker.
    width_defect = prefill_width_defect(width, cu_text)
    if width_defect is not None:
        print(
            "\nERROR: the prefill path cannot serve the SWA index width this "
            "checkpoint produces:",
            file=sys.stderr,
        )
        print(f"  width {width}: {width_defect}", file=sys.stderr)
        print(
            "\nA checkpoint that can carry an image span widens the SWA prefill "
            "rows to sliding_window + vision_config.max_image_tokens, and no "
            "sparse-MLA dispatcher instantiates that width. Serve the model "
            "text-only: run with LANGUAGE_MODEL_ONLY=1 and "
            "TEXT_ONLY_VISION_OVERRIDE=1 so start.sh passes --hf-overrides "
            "'{\"vision_max_n_token\": 0}', and pass the resulting effective "
            "budget here with --max-image-tokens 0. Do NOT zero vision_n_layers "
            "instead — the MoE gate only creates bias_vl when it is positive, so "
            "the checkpoint's bias_vl weights would fail to load.",
            file=sys.stderr,
        )
        return EXIT_MISSING

    print("  compressed pool page block per ratio:")
    fallback_active = sm12x_fallback_installed()
    indexer_defects = indexer_block_kv_defects(rows, fallback_active)
    indexer_unservable = {page_block for _ratio, page_block, _gate in indexer_defects}
    for ratio, page_block in rows:
        marks = []
        if page_block in indexer_unservable:
            marks.append("NOT SERVABLE (DeepGEMM indexer)")
        if page_block in unservable:
            marks.append("NOT SERVABLE (FlashInfer prefill)")
        print(
            f"    compress_ratio {ratio}: block_size//ratio = {page_block}  "
            f"{'ok' if not marks else '; '.join(marks)}"
        )
    if not indexer_unservable:
        print(
            "  DeepGEMM indexer page blocks: ok  "
            f"(SM12x FP8 logits fallback {'installed' if fallback_active else 'not needed'})"
        )

    # DeepGEMM's block_kv constraint is reported before the FlashInfer one: it is
    # what the manager block size has to satisfy, and the fix (`--block-size 64`)
    # is also what decides the FlashInfer page blocks above.
    if indexer_defects:
        print(
            "\nERROR: the sparse indexer's DeepGEMM paged-MQA path cannot serve "
            f"this checkpoint at --block-size {args.block_size}:",
            file=sys.stderr,
        )
        for ratio, page_block, gate in indexer_defects:
            if gate == "metadata":
                print(
                    f"  compress_ratio {ratio}: num_states = block_size // ratio "
                    f"= {page_block}; the metadata builder accepts only "
                    f"{sorted(DEEPGEMM_METADATA_BLOCK_KV)}",
                    file=sys.stderr,
                )
            else:
                print(
                    f"  compress_ratio {ratio}: num_states = block_size // ratio "
                    f"= {page_block}; the FP8 logits kernel accepts only "
                    f"{sorted(DEEPGEMM_FP8_LOGITS_BLOCK_KV)} and the SM12x "
                    "fallback is NOT installed",
                    file=sys.stderr,
                )
        print(
            "\nThe indexer passes `kv_cache_spec.num_states` straight to both "
            "gates: get_paged_mqa_logits_metadata while building decode metadata, "
            "and fp8_fp4_paged_mqa_logits for the logits themselves. On arch 12 "
            "they accept different sets — the metadata builder takes 32 or 64 "
            "(`Assertion error (.../csrc/apis/attention.hpp:262): block_kv == 32 "
            "or block_kv == 64`), the FP8 logits kernel takes 64 only "
            "(`.../attention.hpp:320`) (vLLM #53635) — and a page outside the "
            "metadata set aborts warmup while one outside the logits set boots, "
            "warms up, and dies on the first long prompt.\n"
            "V4.1 compresses at ratio 1 and 2, so --block-size 64 gives {64, 32}: "
            "the 32 needs overlay/patch_sm12x_indexer.py, which routes it to the "
            "overlay's Triton fallback. --block-size 128 gives {128, 64} and "
            "cannot boot at all. Serve with --block-size 64, plus "
            "overlay/patch_v41_kernel_block_size.py so the backends accept it and "
            "overlay/patch_sm12x_indexer.py for the 32-state pages.",
            file=sys.stderr,
        )
        return EXIT_INDEXER_BLOCK_KV

    # A partial application is the silent wrong-kernel case: stock's full-tile
    # branch ends in a bare `else`, so admitting a value without adding the
    # branch runs the 2-token kernel on it. The completeness check is the third
    # leg: it catches a partial application whose *reachable* path happens to
    # work (edit 3 alone), which the per-value checks above cannot see.
    if defects or incomplete:
        print(
            "\nERROR: the installed prefill dispatcher cannot serve this "
            "checkpoint's compressed page blocks:",
            file=sys.stderr,
        )
        for value, reason in defects:
            print(f"  page block {value}: {reason}", file=sys.stderr)
        for reason in incomplete:
            print(f"  incomplete patch: {reason}", file=sys.stderr)
        print(
            "\nRe-apply overlay/patch_flashinfer_prefill_pbsx.py (all three edits: "
            "the acceptance test, the full-tile branch, and the non-full-tile "
            "ladder). See RESEARCH.md 11.9.11 and 11.9.16.",
            file=sys.stderr,
        )
        return EXIT_PAGE_BLOCK

    # An AOT binary makes FlashInfer ignore the patched .cu. That matters
    # whenever the config needs a page block stock FlashInfer does not carry,
    # because the AOT binary is stock and cannot serve it.
    needs_patched = any(
        page_block not in STOCK_EXTRA_PAGE_BLOCKS for _ratio, page_block in rows
    )
    if is_aot is None and needs_patched:
        print(
            "\nERROR: this config needs a compressed page block outside stock "
            f"{sorted(STOCK_EXTRA_PAGE_BLOCKS)}, but whether the prebuilt AOT "
            "binary is still present could not be determined. FlashInfer returns "
            "that binary without compiling whenever it exists, so the patch "
            "cannot be assumed to be in effect. Check the FlashInfer install "
            "before booting.",
            file=sys.stderr,
        )
        return EXIT_PAGE_BLOCK

    if is_aot and needs_patched:
        print(
            "\nERROR: this config needs a compressed page block outside stock "
            f"{sorted(STOCK_EXTRA_PAGE_BLOCKS)}, but the prebuilt AOT binary is "
            "still present. FlashInfer returns it without compiling, so a patched "
            ".cu is ignored and the stock binary cannot serve that page block. "
            "The image must apply "
            "overlay/patch_flashinfer_prefill_pbsx.py and run "
            "overlay/mask_flashinfer_aot.py, then scripts/build_flashinfer.sh "
            "(RESEARCH.md 11.9.3).",
            file=sys.stderr,
        )
        return EXIT_PAGE_BLOCK

    if missing:
        return EXIT_OK if args.allow_missing else EXIT_MISSING
    print("\nOK: decode keys present and every compressed page block is servable")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
