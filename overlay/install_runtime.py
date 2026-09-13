# SPDX-License-Identifier: Apache-2.0
"""Container boot hook: NVMe Engram.

Imported by the inner `vllm serve` wrapper. Does not rewrite vLLM sources
as text (no sitecustomize string patches).

This kit serves a natively-quantized checkpoint (MXFP4/FP8), so it installs no
quantization method. It deliberately does not import any EXL3 code: this hook is
on the fail-closed path (`dsv41_boot` exits 70 when it raises), and an unrelated
import failure must not be able to kill a REAP boot. The EXL3 project lives in
the sibling `deepseek-v4.1-flash-exl3-2dgx-sparks` kit.
"""

from __future__ import annotations

from pathlib import Path
import sys

OVERLAY = Path(__file__).resolve().parent
if str(OVERLAY) not in sys.path:
    sys.path.insert(0, str(OVERLAY))


def install() -> None:
    import engram_install

    engram_install.install()
    if not engram_install._INSTALLED:
        raise RuntimeError("engram_install.install() returned without installing")


if __name__ == "__main__":
    install()
    print("dsv41 runtime overlay installed")
