#!/usr/bin/env python3
"""Shared fixture: a minimal stand-in for FlashInfer's SM120 prefill dispatcher.

Not a test file. Only the three regions the overlay patch anchors on are
reproduced, and they are taken from the patch module's own constants so the
fixture cannot drift away from the anchors it exercises. This keeps vendored
NVIDIA source out of the repo.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "overlay"))

import patch_flashinfer_prefill_pbsx as pfx  # noqa: E402

_HEAD = """// Minimal stand-in for FlashInfer's SM120 prefill dispatcher. Only the three
// regions the overlay patch anchors on are reproduced.
inline bool dispatch_dsv4_dual(int num_heads, int topk, int topk_extra, int extra_page_block_size,
                               const bf16* Q, const uint8_t* KV, const int32_t* indices,
                               const uint8_t* KV_extra, const int32_t* idx_extra,
                               const float* attn_sink, bf16* output, float* out_lse, float sm_scale,
                               int num_tokens, size_t stride_kv_block, size_t stride_kv_block_extra,
                               const int* topk_length_ptr, const int* topk_length_extra_ptr,
                               cudaStream_t stream) {
  if (topk == 128 && topk_length_ptr == nullptr && topk_length_extra_ptr == nullptr &&
"""

_TAIL = """  }

  if (topk != 128) return false;
"""

# The single dispatcher's width ladder, reproduced because the pre-flight reads
# it to explain why a wide SWA table cannot be served even though these widths
# look like they would cover it. This tree never reaches it: with a compressed
# segment present, `sparse_mla_prefill_dispatch` takes the dual path above.
_SINGLE_LADDER = """
inline bool dispatch_dsv4_single(int num_heads, int topk) {
  if (topk == 128)
    DISPATCH_BY_NH_CM(BF16, 128);
  else if (topk == 192)
    DISPATCH_BY_NH_CM(BF16, 192);
  else if (topk == 256)
    DISPATCH_BY_NH_CM(BF16, 256);
  else if (topk == 512)
    DISPATCH_BY_NH_CM(FP8, 512);
  else if (topk == 1024)
    DISPATCH_BY_NH_CM(FP8, 1024);
  else if (topk == 2048)
    DISPATCH_BY_NH_CM(FP8, 2048);
  else
    return false;

#undef DISPATCH_BY_NH_CM
  return false;
}
"""


def stock_cu() -> str:
    """Stock: accepts and instantiates only 64 and 2."""
    return (
        _HEAD
        + pfx.ACCEPT_OLD_LINE
        + "\n"
        + pfx.FULLTILE_OLD
        + _TAIL
        + pfx.PLAIN_OLD
        + "  return false;\n}\n"
        + _SINGLE_LADDER
    )


def patched_cu() -> str:
    """Stock with the overlay patch applied."""
    return (
        stock_cu()
        .replace(pfx.ACCEPT_OLD_LINE, pfx.ACCEPT_NEW_LINE)
        .replace(pfx.FULLTILE_OLD, pfx.FULLTILE_NEW)
        .replace(pfx.PLAIN_OLD, pfx.PLAIN_NEW)
    )


def trap_cu() -> str:
    """The dangerous intermediate: the acceptance test widened, the ladders not.

    Stock's full-tile branch is `if (== 64) ... else { PBSX(2) }`, so widening
    only the acceptance test admits page blocks that the bare `else` silently
    serves with the 2-token instantiation.
    """
    return stock_cu().replace(pfx.ACCEPT_OLD_LINE, pfx.ACCEPT_NEW_LINE)


def no_fulltile_branch_cu() -> str:
    """Edits 1 and 3 applied, edit 2 omitted.

    The acceptance test admits every page block and the non-full-tile ladder
    instantiates them, but the full-tile branch still ends in the bare `else`, so
    a wide page would run the 2-token kernel whenever the full-tile fast path is
    taken.
    """
    return trap_cu().replace(pfx.PLAIN_OLD, pfx.PLAIN_NEW)


def no_plain_branch_cu() -> str:
    """Edits 1 and 2 applied, edit 3 omitted.

    The full-tile branch is correct, but the non-full-tile ladder has no wide
    entry and ends in `return false` — and that is the branch this tree takes.
    """
    return (
        stock_cu()
        .replace(pfx.ACCEPT_OLD_LINE, pfx.ACCEPT_NEW_LINE)
        .replace(pfx.FULLTILE_OLD, pfx.FULLTILE_NEW)
    )


def branch_only_cu() -> str:
    """Edit 3 alone: the wide entries in the non-full-tile ladder, nothing else.

    Correct but asymmetric — the acceptance test still rejects them, so the
    full-tile path is never taken for them and the plain ladder serves them. This
    is the case a union-based check cannot distinguish from a real fix.
    """
    return stock_cu().replace(pfx.PLAIN_OLD, pfx.PLAIN_NEW)


def dual_guard_only_cu() -> str:
    """The dual dispatcher's `topk == 128` guard, with no single-dispatcher ladder.

    Used to prove the width parse is anchored on the single ladder's macro name:
    a loose `topk == N` scan would read this guard as a 128-wide instantiation and
    wrongly conclude the prefill path can serve a 128-wide table via the single
    dispatcher.
    """
    return _HEAD + """  return false;\n}\n"""
