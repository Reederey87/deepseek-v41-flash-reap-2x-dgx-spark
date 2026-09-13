#!/usr/bin/env python3
"""Checks for `overlay/patch_swa_page_size.py` that do not need GB10.

The patch forces 64-token SWA pages for DeepSeek-V4.1 on SM120. It must be
exact (two values, nothing else), idempotent, and fail-closed: a drifted anchor
or a half-applied patch has to stop the build instead of shipping an image that
cannot decode.

When this workspace's `vllm/` tree is present the test runs against the real
files, which also pins the anchors to the pinned revision. In the image (no
sibling tree) it falls back to synthesized fixtures carrying the same anchors.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "overlay" / "patch_swa_page_size.py"

ATTENTION_REL = Path("models/deepseek_v4_1/attention.py")
SPARSE_SWA_REL = Path("v1/attention/backends/mla/sparse_swa.py")
FLASHINFER_REL = Path("models/deepseek_v4_1/nvidia/flashinfer_sparse.py")

REAL_VLLM = ROOT.parent / "vllm" / "vllm"

ATTENTION_FIXTURE = '''\
class DeepseekV4Attention:
    def __init__(self):
        self.swa_cache_layer = DeepseekV4SWACache(
            head_dim=self.head_dim,
            window_size=self.window_size,
            dtype=self.kv_cache_torch_dtype,
            prefix=f"{prefix}.swa_cache",
            cache_config=cache_config,
            backend_cls=self.swa_backend_cls,
            block_size=32,
        )
        self._o_proj_block_size = (
            32 if getattr(self.wo_a, "weight_block_size", None) == [1, 32] else 128
        )
'''

SPARSE_SWA_FIXTURE = '''\
class DeepseekSparseSWABackend(AttentionBackend):
    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(32)]

    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:
        return 256
'''


def build_tree(dest: Path) -> bool:
    """Populate `dest` with the two targets. True when the real files were used."""
    real = (REAL_VLLM / ATTENTION_REL).is_file() and (
        REAL_VLLM / SPARSE_SWA_REL
    ).is_file()
    for rel, fixture in (
        (ATTENTION_REL, ATTENTION_FIXTURE),
        (SPARSE_SWA_REL, SPARSE_SWA_FIXTURE),
    ):
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if real:
            shutil.copyfile(REAL_VLLM / rel, target)
        else:
            target.write_text(fixture)
    if real and (REAL_VLLM / FLASHINFER_REL).is_file():
        extra = dest / FLASHINFER_REL
        extra.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REAL_VLLM / FLASHINFER_REL, extra)
    return real


def run_patch(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(PATCH)],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "DSV41_VLLM_ROOT": str(root)},
    )


def snapshot(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): p.read_text()
        for p in sorted(root.rglob("*.py"))
    }


def test_patch_forces_64_and_touches_nothing_else() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        real = build_tree(root)
        before = snapshot(root)
        result = run_patch(root)
        assert result.returncode == 0, result.stderr
        after = snapshot(root)

        # Exactly one line per target changes.
        for rel in (ATTENTION_REL, SPARSE_SWA_REL):
            old_lines = before[str(rel)].splitlines()
            new_lines = after[str(rel)].splitlines()
            assert len(old_lines) == len(new_lines), rel
            diff = [i for i, (a, b) in enumerate(zip(old_lines, new_lines)) if a != b]
            assert len(diff) == 1, f"{rel}: {len(diff)} lines changed, expected 1"

        assert "block_size=64," in after[str(ATTENTION_REL)]
        assert "block_size=32," not in after[str(ATTENTION_REL)]
        assert "return [MultipleOf(64)]" in after[str(SPARSE_SWA_REL)]
        assert "return [MultipleOf(32)]" not in after[str(SPARSE_SWA_REL)]

        # The 32 that picks the MXFP8 o_proj path is a different value and must
        # survive; the main MLA pool stays at [128] for FlashInfer.
        assert "weight_block_size" in after[str(ATTENTION_REL)]
        if real and str(FLASHINFER_REL) in after:
            assert "return [128]" in after[str(FLASHINFER_REL)]
            assert after[str(FLASHINFER_REL)] == before[str(FLASHINFER_REL)]


def test_patch_is_idempotent() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        build_tree(root)
        assert run_patch(root).returncode == 0
        once = snapshot(root)
        second = run_patch(root)
        assert second.returncode == 0, second.stderr
        assert "already applied" in second.stdout
        assert snapshot(root) == once, "second run modified the tree"


def test_drifted_anchor_fails_closed_without_writing() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        build_tree(root)
        # Drift the SWA page anchor only. The other target is still patchable,
        # so this also proves the preflight happens before any write.
        attention = root / ATTENTION_REL
        attention.write_text(
            attention.read_text().replace("block_size=32,", "block_size=32,".replace("32", "48"))
        )
        before = snapshot(root)
        result = run_patch(root)
        assert result.returncode != 0, "drifted anchor must fail the build"
        assert "unexpected anchor state" in result.stderr, result.stderr
        assert snapshot(root) == before, "failed run must not write anything"


def test_partial_state_fails_closed() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        build_tree(root)
        # Both spellings present in one file: a partial or duplicated edit.
        sparse = root / SPARSE_SWA_REL
        sparse.write_text(
            sparse.read_text().replace(
                "        return [MultipleOf(32)]\n",
                "        return [MultipleOf(32)]\n        return [MultipleOf(64)]\n",
            )
        )
        before = snapshot(root)
        result = run_patch(root)
        assert result.returncode != 0, "mixed anchor state must fail the build"
        assert snapshot(root) == before


def test_missing_file_fails_closed() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        build_tree(root)
        (root / SPARSE_SWA_REL).unlink()
        result = run_patch(root)
        assert result.returncode != 0
        assert "missing file" in result.stderr, result.stderr


if __name__ == "__main__":
    test_patch_forces_64_and_touches_nothing_else()
    test_patch_is_idempotent()
    test_drifted_anchor_fails_closed_without_writing()
    test_partial_state_fails_closed()
    test_missing_file_fails_closed()
    used_real = (REAL_VLLM / ATTENTION_REL).is_file()
    source = "pinned vllm tree" if used_real else "synthesized fixtures"
    print(f"swa page-size patch tests OK (anchors checked against {source})")
