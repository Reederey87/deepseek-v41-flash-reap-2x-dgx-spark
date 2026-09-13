#!/usr/bin/env python3
"""Teach FlashInfer's SM120 sparse-MLA prefill about 32-token compressed pages.

This closes the kit's boot blocker on the FlashInfer side. It is a FlashInfer
source edit, not a vLLM one: at ``--block-size 64`` every compressed-cache
address in vLLM is already correct for a 64- or 32-state page, and the only
missing piece is a prefill kernel instantiation.

Why the block size is 64
------------------------
The indexer feeds DeepGEMM's paged-MQA kernel ``kv_cache_spec.num_states`` as
``block_kv``, and that kernel's host dispatcher asserts ``block_kv == 32 or
block_kv == 64`` (vLLM #53635). ``num_states`` is ``block_size //
tokens_per_state``, and V4.1's compressed caches use ``tokens_per_state =
compress_ratio`` with ratios 1 and 2, so the manager block size decides:

* ``--block-size 128`` -> {128, 64} -> 128 is not a kernel the host has.
* ``--block-size 64``  -> {64, 32}  -> both are.

64 is therefore the checkpoint's native geometry on SM12x, and it is what
``overlay/patch_v41_kernel_block_size.py`` and ``start.sh`` pin. That patch is
what makes the MLA and indexer backends accept 64 at all.

Why this patch is still needed
------------------------------
The compressed pool's page block is the number of states per page, read from the
cache tensor's own shape (``flashinfer/mla/_sparse_mla_sm120.py``,
``_packed_kv_page_block_size``: NHD ``[num_pages, page_block_size, 1, bytes]`` ->
``shape[1]``). vLLM builds that tensor as ``[pages, states_per_page, 584]`` with
``states_per_page = block_size // compress_ratio``
(``models/deepseek_v4_1/nvidia/flashinfer_sparse.py``,
``compressed_block_size = attn_metadata.block_size // self.compress_ratio``),
so at ``--block-size 64`` ratio 1 yields **64** and ratio 2 yields **32**.

FlashInfer's dual-cache prefill dispatcher accepts only 64 or 2
(``csrc/sparse_mla_sm120_prefill.cu``, ``dispatch_dsv4_dual``). The 64 is the
ratio-1 page and is already stock; the **2 is V4's** ratio-128 geometry, and
nothing here produces it. So ratio-2 prefill has no instantiation and the engine
dies on the first long prompt. This patch adds 32.

Everything else about a 32-state page is already consistent, which is why the
kernel is the only thing that has to change:

* The slot ids vLLM computes for ratio 2 are ``block_table[req, p // 32] * 32 +
  p % 32``, and the block table is manager-block granular, so the page index is
  the manager block id.
* The kernel's only use of the page block is as the divisor/modulus that turns a
  slot id into ``(block_idx, local_idx)``, with the page byte stride passed in by
  the caller: ``io_bulk_gather_tile`` and ``io_gather_scales``
  (``common/kv_cache_io.cuh``) and ``prefill_kv_entry_base``
  (``prefill_kernel.cuh``). 32 is the correct value for those divisions.
* The scale region sits at ``page_block_size * 576`` inside the page, exactly
  where vLLM's writer puts it (``CACHE_BLOCK * 576`` in
  ``fused_compress_quant_cache.py``). At 32 that is 18,432 bytes of data plus 256
  bytes of scales, well inside the 74,880-byte page, so the ``alignment=576``
  padding is harmless tail space and needs no alignment change.
* There is no ``static_assert`` or bound on the page block, and the kernel's
  shared-memory footprint (``SmemLayoutMG``) does not depend on it, so a 32
  instantiation is not a new register/smem tier. Only ``page_block_size == 2``
  has a special path (``USE_WFP8_ROW_XOR``), which 32 does not take.
* The decode side needs no companion change: ``_decode_dsv4_dispatchable``
  checks the *primary* (SWA) cache's page, which ``patch_swa_page_size.py``
  already pins to 64, and the extra page block reaches the decode kernel as a
  runtime divisor rather than a template parameter.

Why 128 is kept
---------------
The 128 instantiation this patch previously added is retained. At
``--block-size 64`` it is unreachable, so it is inert; it costs compile time
only, and it keeps the 128-token geometry reachable if upstream ever teaches
DeepGEMM's paged-MQA kernel a 128-wide ``block_kv`` (option (a) in #53635). The
accepted set and the branch sets are kept **coterminous** — the acceptance test
enumerates exactly {64, 2, 128, 32} and so does each ladder — because a
dispatcher whose guard admits a value that no branch handles is the dangerous
half-applied state this file exists to prevent.

The edits
---------
1. ``dispatch_dsv4_dual``'s full-tile acceptance test gains ``== 32``.
2. Its full-tile branch becomes a fully explicit ladder. Stock is ``if (== 64)
   ... else { (2) }`` with a bare ``else``; the bare ``else`` is what made the
   original 128 edit a trap, because admitting a value without adding a branch
   ran the ``PBSX(2)`` instantiation on it and computed ``idx / 2``. The ladder
   now ends in ``else { return false; }``, so a value the guard admits but no
   branch handles is a clean ``false`` rather than a wrong divisor.
3. The non-full-tile ladder gains ``else if (== 32)``. This is the branch our
   tree actually takes: vLLM passes ``extra_sparse_topk_lens``, so
   ``topk_length_extra_ptr`` is non-null and the full-tile fast path is skipped.

The AOT trap is handled separately by ``mask_flashinfer_aot.py``; a patched
``.cu`` is ignored while the prebuilt ``.so`` exists.

Scope: three exact source edits to one vendored file. Idempotent and fail-closed:
each edit is classified on its whole `old`/`new` block before anything is
written, so a drifted anchor or an unrecognised mixed state exits non-zero
without touching the file. A *recognised* partial application — one edit already
in place, the others not — is deliberately **completed** rather than refused,
because widening the acceptance test without adding the branch is the dangerous
intermediate state and finishing the job is the repair. The result is
re-classified before it is written, so a partial write cannot happen. Note that
the *previous* form of this patch (which added 128 and ended the full-tile
ladder in a bare ``else``) is not a recognised state: rebuilding on top of an
image that already carries it fails closed and says so. Rebuild from the stock
base image.

See `RESEARCH.md` §11.9.15 and §11.9.16.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

MARK = "dsv41-fi-prefill-pbsx"

PRELUDE_REL = Path("data/csrc/sparse_mla_sm120_prefill.cu")

# Identity check: the file must be the SM120 prefill dispatcher we expect.
IDENTITY = "dispatch_dsv4_dual"


class Edit:
    """One exact source edit.

    Classification runs on the whole `old`/`new` block rather than on a single
    value line, because two of these edits are structural — a bare `else`
    becomes `else if`, and a branch is appended — so the stock text partly
    survives inside the patched form. `new` is never a substring of `old` and
    vice versa, which is what makes the block-level test unambiguous.
    """

    def __init__(self, name: str, old: str, new: str) -> None:
        self.name = name
        self.old = old
        self.new = new

    def classify(self, text: str) -> str:
        """Return 'patched', 'original', or fail closed on anything else."""
        old_n = text.count(self.old)
        new_n = text.count(self.new)
        if old_n == 1 and new_n == 0:
            return "original"
        if old_n == 0 and new_n == 1:
            return "patched"
        raise SystemExit(
            f"[{MARK}] {self.name}: unexpected anchor state "
            f"(stock form x{old_n}, patched form x{new_n}); refusing to write. "
            "The vendored FlashInfer source drifted, an unrecognised partial "
            "edit is present, or this image already carries the previous "
            "128-only form of this patch — re-derive the anchor before building."
        )

    def apply(self, text: str) -> str:
        if text.count(self.old) != 1:
            raise SystemExit(
                f"[{MARK}] {self.name}: replacement anchor not unique "
                f"(found {text.count(self.old)})"
            )
        patched = text.replace(self.old, self.new)
        if patched.count(self.new) != 1:
            raise SystemExit(f"[{MARK}] {self.name}: rewrite did not take; refusing")
        return patched


# 1. The full-tile acceptance test. Kept on one line so the vendored file's line
#    numbering shifts only by the branch lines below.
ACCEPT_OLD_LINE = (
    "      topk_extra % BI == 0 && (extra_page_block_size == 64 || "
    "extra_page_block_size == 2)) {"
)
ACCEPT_NEW_LINE = (
    "      topk_extra % BI == 0 && (extra_page_block_size == 64 || "
    "extra_page_block_size == 2 || extra_page_block_size == 128 || "
    "extra_page_block_size == 32)) {"
)

# 2. The full-tile branch. Stock's bare `else` must become an explicit ladder
#    with a fail-closed tail, so no value the acceptance test admits can reach a
#    PBSX instantiation that is not its own.
FULLTILE_OLD = """    if (extra_page_block_size == 64) {
      DISPATCH_FULLTILE_BY_NH_PBSX(64);
    } else {
      DISPATCH_FULLTILE_BY_NH_PBSX(2);
    }
"""
FULLTILE_NEW = """    if (extra_page_block_size == 64) {
      DISPATCH_FULLTILE_BY_NH_PBSX(64);
    } else if (extra_page_block_size == 2) {
      DISPATCH_FULLTILE_BY_NH_PBSX(2);
    } else if (extra_page_block_size == 128) {
      DISPATCH_FULLTILE_BY_NH_PBSX(128);
    } else if (extra_page_block_size == 32) {
      DISPATCH_FULLTILE_BY_NH_PBSX(32);
    } else {
      return false;
    }
"""

# 3. The non-full-tile branch (the one this tree takes, because it passes
#    extra_sparse_topk_lens). Its stock tail is already an explicit `return
#    false`, so this edit only appends the 32 arm.
PLAIN_OLD = """  if (extra_page_block_size == 64) {
    DISPATCH_BY_NH_PBSX(64);
  } else if (extra_page_block_size == 2) {
    DISPATCH_BY_NH_PBSX(2);
  }
"""
PLAIN_NEW = """  if (extra_page_block_size == 64) {
    DISPATCH_BY_NH_PBSX(64);
  } else if (extra_page_block_size == 2) {
    DISPATCH_BY_NH_PBSX(2);
  } else if (extra_page_block_size == 128) {
    DISPATCH_BY_NH_PBSX(128);
  } else if (extra_page_block_size == 32) {
    DISPATCH_BY_NH_PBSX(32);
  }
"""

EDITS = [
    Edit(
        name="full-tile acceptance test",
        old=ACCEPT_OLD_LINE,
        new=ACCEPT_NEW_LINE,
    ),
    Edit(
        name="full-tile dispatch branch",
        old=FULLTILE_OLD,
        new=FULLTILE_NEW,
    ),
    Edit(
        name="non-full-tile dispatch branch",
        old=PLAIN_OLD,
        new=PLAIN_NEW,
    ),
]

# The compressed page blocks the acceptance test must admit. Pinned here so the
# tests and the pre-flight agree with the patch on one list rather than three.
ACCEPTED_PAGE_BLOCKS = frozenset({64, 2, 128, 32})


def flashinfer_root() -> Path:
    """Return the installed flashinfer package directory."""
    override = os.environ.get("DSV41_FLASHINFER_ROOT")
    if override:
        return Path(override)
    spec = importlib.util.find_spec("flashinfer")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit(
            f"[{MARK}] cannot locate the flashinfer package; "
            "set DSV41_FLASHINFER_ROOT"
        )
    return Path(next(iter(spec.submodule_search_locations)))


def main() -> int:
    path = flashinfer_root() / PRELUDE_REL
    if not path.is_file():
        raise SystemExit(f"[{MARK}] missing vendored source {path}")

    text = path.read_text()
    if IDENTITY not in text:
        raise SystemExit(
            f"[{MARK}] {path} does not define {IDENTITY}; refusing to patch an "
            "unrecognised file"
        )

    # Classify every edit before writing any of them.
    states = [edit.classify(text) for edit in EDITS]
    if all(state == "patched" for state in states):
        print(f"[{MARK}] already applied to {path}")
        return 0

    patched = text
    for edit, state in zip(EDITS, states):
        if state == "patched":
            print(f"[{MARK}] {edit.name}: already patched, leaving alone")
            continue
        patched = edit.apply(patched)
        print(f"[{MARK}] {edit.name}: patched")

    # Re-classify the result: every edit must now read as applied.
    for edit in EDITS:
        if edit.classify(patched) != "patched":
            raise SystemExit(
                f"[{MARK}] {edit.name}: post-write classification failed; "
                "refusing to write a partial result"
            )

    path.write_text(patched)
    print(f"[{MARK}] applied to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
