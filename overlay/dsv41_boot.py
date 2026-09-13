# SPDX-License-Identifier: Apache-2.0
"""Persistent boot hook so every interpreter installs the overlay.

`vllm serve` forces `VLLM_WORKER_MULTIPROC_METHOD=spawn`
(`vllm/entrypoints/serve/utils/api_utils.py::cli_env_setup`), and a spawned
EngineCore / TP worker re-imports the `__main__` module without running its
`if __name__ == "__main__":` block. An install that only runs in `serve.py`
therefore never reaches the process that constructs `Engram` and streams the
checkpoint, and the stock path materializes the 91 GiB tables.

A site `.pth` (`dsv41_engram.pth`) imports this module at every interpreter
startup, which is how the existing deployment on these boxes persists its
overlay. The install is idempotent and only acts when disk offload is enabled.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


def install() -> None:
    from install_runtime import install as _install

    _install()


def _strict() -> bool:
    """Fail closed by default.

    A `.pth` cannot raise: `site.addpackage` prints the traceback and carries
    on, so a broken overlay would leave a process that constructs `Engram`
    and then streams the 91 GiB tables into unified memory. That is exactly
    the failure this kit exists to prevent, so a failed install must kill the
    process rather than warn. Set `DSV41_ENGRAM_STRICT=0` for a debugging
    shell, or `DSV41_ENGRAM_DISK=0` to skip the overlay entirely.
    """
    return os.environ.get("DSV41_ENGRAM_STRICT", "1") == "1"


if os.environ.get("DSV41_ENGRAM_DISK", "1") == "1":
    try:
        install()
    except BaseException as exc:  # noqa: BLE001 - reported, never swallowed
        os.environ["DSV41_OVERLAY_FAILED"] = f"{type(exc).__name__}: {exc}"
        print(
            f"dsv41_boot: Engram overlay install FAILED: {exc!r}",
            file=sys.stderr,
        )
        sys.stderr.flush()
        if _strict():
            os._exit(70)
