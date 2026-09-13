#!/usr/bin/env python3
"""Install the V4.1 overlay, then hand off to `vllm serve` in this process."""

from __future__ import annotations

import os
import sys
from pathlib import Path

OVERLAY = Path(__file__).resolve().parent
if str(OVERLAY) not in sys.path:
    sys.path.insert(0, str(OVERLAY))

from install_runtime import install  # noqa: E402


def _verify_overlay() -> None:
    """Refuse to serve unless the Engram overlay is actually in place.

    The `.pth` hook installs at interpreter startup, but a `.pth` failure is
    only printed, never raised. Re-check here so the serving process cannot
    silently fall back to the stock loader and materialize the NVMe tables.
    """
    import engram_install

    if os.environ.get("DSV41_OVERLAY_FAILED"):
        raise SystemExit(f"Engram overlay failed at boot: {os.environ['DSV41_OVERLAY_FAILED']}")
    if not engram_install._INSTALLED:
        raise SystemExit("Engram overlay is not installed; refusing to serve")


def main() -> None:
    install()
    _verify_overlay()
    from vllm.entrypoints.cli.main import main as vllm_main

    if len(sys.argv) == 1 or sys.argv[1] != "serve":
        sys.argv = [sys.argv[0], "serve", *sys.argv[1:]]
    vllm_main()


if __name__ == "__main__":
    main()
