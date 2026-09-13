#!/usr/bin/env python3
"""Mask the prebuilt sparse_mla_sm120 AOT module so the patched .cu is used.

FlashInfer's ``JitSpec.build_and_load()`` short-circuits on the prebuilt AOT
binary::

    if self.is_aot:
        return self.load(self.aot_path)

``is_aot`` is ``FLASHINFER_AOT_DIR/<name>/<name>.so`` existing, and in this base
image that file is present at
``/usr/local/lib/python3.12/dist-packages/flashinfer_jit_cache/jit_cache/
sparse_mla_sm120/sparse_mla_sm120.so``. While it exists, the source edit made by
``patch_flashinfer_prefill_pbsx.py`` is **silently ignored** — the module loads
from the AOT binary and the 32-state ratio-2 prefill still has no
instantiation. This is the trap recorded in RESEARCH.md §11.9.3.

Removing the module directory leaves ``flashinfer_jit_cache`` itself importable,
so ``FLASHINFER_AOT_DIR`` still resolves to ``.../jit_cache`` and only this one
module loses its AOT shortcut; every other prebuilt module keeps working. The
patched ``.cu`` is then compiled by ``scripts/build_flashinfer.sh`` into the
persistent FlashInfer cache, which the launcher mounts at
``/root/.cache/flashinfer``.

Idempotent: a missing module directory means the mask is already in place and is
reported as such. Fail-closed: anything other than "the expected .so is there" or
"the directory is gone" exits non-zero.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from pathlib import Path

MARK = "dsv41-fi-aot-mask"

# The JIT module whose AOT binary must go.
MODULE = "sparse_mla_sm120"


def _package_dir(name: str) -> Path | None:
    """Locate an installed package without importing it."""
    spec = importlib.util.find_spec(name)
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(next(iter(spec.submodule_search_locations)))


def aot_dir() -> Path:
    """Return FlashInfer's AOT directory.

    ``flashinfer.jit.env`` computes this as
    ``flashinfer_jit_cache.get_jit_cache_dir()`` when that package is installed
    — which its ``__init__`` defines as ``Path(__file__).parent / "jit_cache"``
    — and ``flashinfer/data/aot`` otherwise. Both are handled.

    The path is derived from the installed package locations rather than by
    importing ``flashinfer.jit.env``, because that import pulls in torch and a
    CUDA context and this script runs during a GPU-less ``docker build``.
    """
    override = os.environ.get("DSV41_FLASHINFER_AOT_DIR")
    if override:
        return Path(override)
    jit_cache = _package_dir("flashinfer_jit_cache")
    if jit_cache is not None:
        return jit_cache / "jit_cache"
    flashinfer = _package_dir("flashinfer")
    if flashinfer is None:
        raise SystemExit(
            f"[{MARK}] neither flashinfer nor flashinfer_jit_cache is importable"
        )
    return flashinfer / "data" / "aot"


def main() -> int:
    root = aot_dir()
    module_dir = root / MODULE
    aot_so = module_dir / f"{MODULE}.so"

    if not root.is_dir():
        raise SystemExit(f"[{MARK}] AOT directory {root} does not exist")
    if not module_dir.exists():
        print(f"[{MARK}] already masked: {module_dir} is absent")
        return 0
    if not module_dir.is_dir():
        raise SystemExit(f"[{MARK}] {module_dir} exists but is not a directory")
    if not aot_so.is_file():
        # No .so means is_aot is already False for this module; leave whatever
        # else is there alone rather than deleting an unexpected tree.
        print(f"[{MARK}] already masked: {aot_so} is absent")
        return 0

    size = aot_so.stat().st_size
    shutil.rmtree(module_dir)
    if module_dir.exists():
        raise SystemExit(f"[{MARK}] {module_dir} still present after removal")
    print(f"[{MARK}] masked {aot_so} ({size} bytes); patched .cu will be compiled")
    return 0


if __name__ == "__main__":
    sys.exit(main())
