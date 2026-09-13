#!/usr/bin/env python3
"""Force 64-token SWA pages for DeepSeek-V4.1 on SM120 (GB10).

The pinned vLLM builds every ``DeepseekV4SWACache`` with a 32-token page:
``deepseek_v4_1/attention.py`` passes ``block_size=32`` into the constructor, so
the SWA KV-cache group's manager page is 32 and ``swa_metadata.block_size`` is
32. ``--block-size`` cannot change this — the SWA spec takes its page from that
constructor argument, not from ``cache_config.block_size``.

FlashInfer's SM120 DeepSeek-V4 sparse decoder requires a 64-token page.
``flashinfer/mla/_sparse_mla_sm120.py`` defines
``_DECODE_DSV4_PAGE_BLOCK_SIZE = 64`` and ``_decode_dsv4_dispatchable()`` accepts
only ``page_block_size == 64``; the page size is read from the primary cache
tensor's shape, and vLLM passes the SWA cache as that primary tensor. On
``cc == (12, 0)`` or ``(12, 1)``, ``_resolve_dsv4_sparse_mla_backend()`` resolves
``backend="auto"`` to ``"sparse"`` before any shape check, and the TRTLLM-gen
alternative requires SM100/SM103. With a 32-token page the decode specialization
does not match, the paged orchestrator is prefill-only (it asserts
``num_tokens > 64`` in C++), and a decode-sized call raises::

    ValueError: SM120 sparse-MLA has no decode kernel for this shape: ...
    page_block_size=32 ...

The 32 is deliberate upstream, not drift: vLLM #56228 ("DeepSeek-V4.1-Flash Model
Definitions") added the parameter, removed an older hardcoded ``= 64``, relaxed
the backend to ``MultipleOf(32)``, and noted that "the sparse decode kernels take
the page size at runtime". That holds for the FlashMLA path, which
``DeepseekV4SparseMLABackend.supports_compute_capability()`` restricts to
``major in [9, 10]`` — so it never runs on GB10. On SM12 the only V4.1 sparse-MLA
path is FlashInfer, where the page size is a dispatch key rather than a runtime
argument. Current upstream ``main`` still passes 32, so no vLLM revision
available today fixes this; the two values below are the whole fix.

Scope: two exact source edits, nothing else.

* ``vllm/models/deepseek_v4_1/attention.py`` — the SWA-cache constructor call.
* ``vllm/v1/attention/backends/mla/sparse_swa.py`` — the SWA backend's advertised
  kernel block sizes, moved back to ``MultipleOf(64)`` to match the page.

The main MLA pool's advertised kernel block size is **not** touched here: FlashInfer
validates only the primary (SWA) cache's page size, so this patch is complete
without it. That pool is handled separately, because the kit's pinned
``--block-size 64`` requires all three V4.1 backends to advertise 64
(``overlay/patch_v41_kernel_block_size.py``, `RESEARCH.md` §11.9.15).

Idempotent, and fail-closed: both files are classified before either is written,
so a drifted anchor or a half-applied patch exits non-zero without touching the
tree. See `RESEARCH.md` §11.6.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

MARK = "dsv41-swa-page-64"

ATTENTION_REL = Path("models/deepseek_v4_1/attention.py")
SPARSE_SWA_REL = Path("v1/attention/backends/mla/sparse_swa.py")

# The SWA-cache construction site. Anchored on the two lines that identify it so
# a same-valued `block_size=32,` elsewhere in the file cannot be hit.
ATTENTION_OLD = """            backend_cls=self.swa_backend_cls,
            block_size=32,
"""
ATTENTION_NEW = """            backend_cls=self.swa_backend_cls,
            block_size=64,
"""

SPARSE_SWA_OLD = """    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(32)]
"""
SPARSE_SWA_NEW = """    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(64)]
"""


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


class Target:
    """One exact source edit.

    `anchor_old`/`anchor_new` carry the replacement. `old_line`/`new_line` are
    the single distinguishing lines, and classification runs on those so that a
    duplicated or hand-edited value line is caught even when the surrounding
    anchor no longer matches.
    """

    def __init__(
        self,
        name: str,
        rel: Path,
        old_line: str,
        new_line: str,
        anchor_old: str,
        anchor_new: str,
    ) -> None:
        self.name = name
        self.rel = rel
        self.old_line = old_line
        self.new_line = new_line
        self.anchor_old = anchor_old
        self.anchor_new = anchor_new

    def classify(self, text: str) -> str:
        """Return 'patched', 'original', or fail closed on anything else."""
        old_n = text.count(self.old_line)
        new_n = text.count(self.new_line)
        if old_n == 1 and new_n == 0:
            return "original"
        if old_n == 0 and new_n == 1:
            return "patched"
        raise SystemExit(
            f"[{MARK}] {self.name}: unexpected anchor state "
            f"(original value x{old_n}, patched value x{new_n}); refusing to "
            "write. Upstream source drifted, or a partial edit is present — "
            "re-derive the anchor before building."
        )

    def apply(self, path: Path) -> None:
        text = path.read_text()
        if text.count(self.anchor_old) != 1:
            raise SystemExit(
                f"[{MARK}] {self.name}: replacement anchor not unique in {path}"
            )
        patched = text.replace(self.anchor_old, self.anchor_new)
        if self.classify(patched) != "patched":
            raise SystemExit(f"[{MARK}] {self.name}: rewrite did not take; refusing")
        old_lines = text.splitlines()
        new_lines = patched.splitlines()
        if len(old_lines) != len(new_lines):
            raise SystemExit(f"[{MARK}] {self.name}: line count changed; refusing")
        changed = sum(1 for a, b in zip(old_lines, new_lines) if a != b)
        if changed != 1:
            raise SystemExit(
                f"[{MARK}] {self.name}: {changed} lines changed, expected 1"
            )
        path.write_text(patched)
        print(f"[{MARK}] {self.name}: patched {path}")


TARGETS = [
    Target(
        name="SWA cache page",
        rel=ATTENTION_REL,
        old_line="            block_size=32,",
        new_line="            block_size=64,",
        anchor_old=ATTENTION_OLD,
        anchor_new=ATTENTION_NEW,
    ),
    Target(
        name="SWA backend page",
        rel=SPARSE_SWA_REL,
        old_line="        return [MultipleOf(32)]",
        new_line="        return [MultipleOf(64)]",
        anchor_old=SPARSE_SWA_OLD,
        anchor_new=SPARSE_SWA_NEW,
    ),
]


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
