#!/usr/bin/env python3
"""Checks for `overlay/patch_v41_kernel_block_size.py`.

Four things are pinned here:

1. The patch applies exactly the three intended method-body replacements, is
   idempotent, is purely additive, and fails closed on a drifted or partially
   applied source.
2. The patched methods really do return `[64]` on capability family 120 and
   leave every other family exactly as it was — executed, not string-matched.
3. The *installed* vLLM, when present (inside the image), is either stock or
   fully patched — never a mix — and the patch behaves correctly against it.
   The Dockerfile applies the patch before this suite runs, so the image path is
   the patched one. Skipped on this Mac, where vLLM is not installed.
4. The Dockerfile applies the patch before running this suite.

The stock fixtures are assembled from the patch module's own anchor constants, so
they cannot drift away from the anchors they exercise.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / "overlay"
PATCH = OVERLAY / "patch_v41_kernel_block_size.py"

sys.path.insert(0, str(OVERLAY))

import patch_v41_kernel_block_size as kb  # noqa: E402

_PRELUDE = """from __future__ import annotations


class Backend:
"""

_FAMILY_CLASS = """from __future__ import annotations


class FamilyBackend:
"""


class _Capability:
    """A stand-in for `current_platform` at a fixed major version.

    `is_device_capability_family` divides by ten, so family 120 means major 12 —
    which is what GB10's (12, 1) reports.
    """

    def __init__(self, major: int) -> None:
        self.major = major

    def is_device_capability_family(self, capability: int) -> bool:
        return capability // 10 == self.major


def _stock_text(target_name: str) -> str:
    """A minimal stock module carrying exactly one of the patch's anchors."""
    for target in kb.TARGETS:
        if target.name != target_name:
            continue
        prelude = _PRELUDE if "FlashInfer" in target_name else _FAMILY_CLASS
        return prelude + target.old
    raise AssertionError(f"no such target: {target_name}")


def _write_stock(root: Path) -> None:
    for target in kb.TARGETS:
        path = root / target.rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_stock_text(target.name))


def _apply(root: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(PATCH)],
        capture_output=True,
        text=True,
        env={**os.environ, "DSV41_VLLM_ROOT": str(root)},
    )


def _supported(path: Path, major: int, cls: str) -> list:
    """The patched method's return value, with the platform stubbed at `major`.

    `flashinfer_sparse.py`'s patched method imports `current_platform` from
    `vllm.platforms` inside the body, so that import is satisfied with a stub
    module rather than skipped — the import is part of what is being tested.
    """
    capability = _Capability(major)
    fake_vllm = types.ModuleType("vllm")
    fake_platforms = types.ModuleType("vllm.platforms")
    fake_platforms.current_platform = capability
    fake_vllm.platforms = fake_platforms
    saved = {name: sys.modules.get(name) for name in ("vllm", "vllm.platforms")}
    sys.modules["vllm"] = fake_vllm
    sys.modules["vllm.platforms"] = fake_platforms
    try:
        namespace: dict = {"current_platform": capability}
        exec(compile(path.read_text(), str(path), "exec"), namespace)  # noqa: S102
        # The call must happen while the stub is installed: the patched
        # `flashinfer_sparse.py` method imports `vllm.platforms` in its body.
        return namespace[cls].get_supported_kernel_block_sizes()
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _class_of(target) -> str:
    return "Backend" if "FlashInfer" in target.name else "FamilyBackend"


def test_patch_covers_three_files() -> None:
    rels = sorted(str(t.rel) for t in kb.TARGETS)
    assert rels == [
        "models/deepseek_v4_1/nvidia/flashinfer_sparse.py",
        "models/deepseek_v4_1/sparse_mla.py",
        "v1/attention/backends/mla/indexer.py",
    ], rels


def test_every_edit_is_purely_additive() -> None:
    """No stock line may be dropped, so no other platform's behaviour can shift."""
    for target in kb.TARGETS:
        old = target.old.splitlines()
        new = target.new.splitlines()
        assert kb._is_ordered_subsequence(old, new), target.name
        assert target.delta > 0, target.name


def test_every_edit_names_family_120_and_returns_64() -> None:
    for target in kb.TARGETS:
        assert "is_device_capability_family(120)" in target.new, target.name
        assert "return [64]" in target.new, target.name


def test_anchors_are_disjoint_from_their_replacements() -> None:
    """The classification is a block test; it is only sound if the blocks differ."""
    for target in kb.TARGETS:
        assert target.old not in target.new, target.name
        assert target.new not in target.old, target.name


def test_patch_applies_and_is_idempotent() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_stock(root)

        first = _apply(root)
        assert first.returncode == 0, (first.returncode, first.stdout, first.stderr)
        once = {t.rel: (root / t.rel).read_text() for t in kb.TARGETS}
        for target in kb.TARGETS:
            assert once[target.rel].count(target.new) == 1, target.name

        second = _apply(root)
        assert second.returncode == 0, second.stderr
        assert "already applied" in second.stdout, second.stdout
        for rel, text in once.items():
            assert (root / rel).read_text() == text, "a second run must not change files"


def test_patched_methods_return_64_on_family_120_only() -> None:
    """The behaviour, executed: 64 on SM12x, untouched everywhere else."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_stock(root)
        assert _apply(root).returncode == 0

        for target in kb.TARGETS:
            path = root / target.rel
            cls = _class_of(target)
            # Family 120 is SM120/SM121 — the only target of this kit.
            assert _supported(path, 12, cls) == [64], target.name
            # Family 90 keeps its own 64; SM100/110 keep 128.
            if cls == "FamilyBackend":
                assert _supported(path, 9, cls) == [64], target.name
            for major in (10, 11):
                assert _supported(path, major, cls) == [128], (target.name, major)


def test_stock_fixture_returns_128_on_family_120() -> None:
    """The fixture must actually reproduce the bug, or the test above proves nothing."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_stock(root)
        for target in kb.TARGETS:
            got = _supported(root / target.rel, 12, _class_of(target))
            assert got == [128], (target.name, got)


def test_patch_refuses_a_drifted_anchor() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_stock(root)
        victim = kb.TARGETS[0]
        drifted = (root / victim.rel).read_text().replace("return [128]", "return [256]")
        (root / victim.rel).write_text(drifted)

        r = _apply(root)
        assert r.returncode != 0, (r.returncode, r.stdout)
        assert "unexpected anchor state" in r.stderr, r.stderr
        assert (root / victim.rel).read_text() == drifted, (
            "a refused patch must not touch the file"
        )
        # The other two must not have been written either: all targets are
        # classified before any is applied.
        for target in kb.TARGETS[1:]:
            assert (root / target.rel).read_text() == _stock_text(target.name), target.name


def test_patch_fails_closed_on_a_missing_file() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_stock(root)
        (root / kb.TARGETS[1].rel).unlink()
        r = _apply(root)
        assert r.returncode != 0, (r.returncode, r.stdout)
        assert "missing file" in r.stderr, r.stderr


def test_patch_refuses_an_already_patched_base() -> None:
    """Rebuilding on an image that already carries the patch must refuse.

    The previous form of the tree advertised 128 on SM12x, so an image built
    before this patch is not a recognised state — it is neither stock nor
    patched, and silently applying on top of it is what the anchor test prevents.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_stock(root)
        victim = kb.TARGETS[0]
        (root / victim.rel).write_text(
            _stock_text(victim.name).replace("return [128]", "return [64]")
        )
        r = _apply(root)
        assert r.returncode != 0, (r.returncode, r.stdout)
        assert "unexpected anchor state" in r.stderr, r.stderr


def _real_vllm_installed() -> bool:
    spec = importlib.util.find_spec("vllm")
    return spec is not None and bool(spec.submodule_search_locations)


def test_installed_vllm_is_stock_or_fully_patched() -> None:
    """In the image: the installed sources must be stock, or fully patched.

    The Dockerfile applies this patch *before* it runs this suite, so inside the
    image the installed files are already patched and the stock anchors are gone
    by design — requiring them there is what would break the build. Both states
    are therefore legitimate, and each is checked for what it means:

    * **stock** — the anchors are present, and applying the patch to a copy of
      the real files produces the patched form;
    * **patched** — the patched form is present exactly once, re-running is a
      clean no-op that does not rewrite the file, and the file still parses.

    A mix (some targets stock, some patched) is refused: that is a half-applied
    patch, and it is the state this test exists to catch. So is a file that is
    neither form, which is what a base-image bump that reshapes these methods
    looks like.

    Skipped where vLLM is absent (this Mac), so the check runs exactly where it
    means something.
    """
    if not _real_vllm_installed():
        print("  (skipped: vllm not installed)")
        return
    spec = importlib.util.find_spec("vllm")
    installed = Path(next(iter(spec.submodule_search_locations)))

    states: dict[str, str] = {}
    for target in kb.TARGETS:
        text = (installed / target.rel).read_text()
        if target.new in text:
            states[str(target.rel)] = "patched"
        elif target.old in text:
            states[str(target.rel)] = "stock"
        else:
            raise AssertionError(
                f"{target.rel} is neither the stock nor the patched form of this "
                "patch's anchor; the base image drifted, so the anchor must be "
                "re-derived before building"
            )
    assert len(set(states.values())) == 1, (
        f"the installed vLLM is half-patched: {states}"
    )
    state = next(iter(states.values()))

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "vllm"
        for target in kb.TARGETS:
            dst = root / target.rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(installed / target.rel, dst)

        r = _apply(root)
        assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)

        if state == "patched":
            assert "already applied" in r.stdout, r.stdout
        for target in kb.TARGETS:
            text = (root / target.rel).read_text()
            assert text.count(target.new) == 1, target.name
            # A no-op must not have rewritten the installed bytes.
            if state == "patched":
                assert text == (installed / target.rel).read_text(), target.name
            # The patched real module must still compile, and the method it
            # patches must still be the only one in the file.
            compile(text, str(root / target.rel), "exec")
            assert text.count("def get_supported_kernel_block_sizes") == (
                (installed / target.rel).read_text().count(
                    "def get_supported_kernel_block_sizes"
                )
            ), target.name

    print(f"  (installed vLLM was {state}; checked accordingly)")


def dockerfile_ordering_defects(root: Path) -> list[str]:
    """Why `root`'s Dockerfile does not apply this patch before running this suite."""
    path = root / "Dockerfile"
    if not path.is_file():
        return [f"no Dockerfile at {path}; the ordering cannot be verified"]
    text = path.read_text()
    defects = []
    patch_at = text.find("patch_v41_kernel_block_size.py")
    test_at = text.find("test_v41_kernel_block_size_patch.py")
    if patch_at == -1:
        defects.append("the Dockerfile does not apply the kernel-block-size patch")
    if test_at == -1:
        defects.append("the Dockerfile does not run this test suite")
    if not defects and patch_at > test_at:
        defects.append(
            "patch_v41_kernel_block_size.py must run before "
            "test_v41_kernel_block_size_patch.py"
        )
    return defects


def test_dockerfile_orders_the_patch_before_the_tests() -> None:
    defects = dockerfile_ordering_defects(ROOT)
    assert defects == [], defects


if __name__ == "__main__":
    test_patch_covers_three_files()
    test_every_edit_is_purely_additive()
    test_every_edit_names_family_120_and_returns_64()
    test_anchors_are_disjoint_from_their_replacements()
    test_patch_applies_and_is_idempotent()
    test_patched_methods_return_64_on_family_120_only()
    test_stock_fixture_returns_128_on_family_120()
    test_patch_refuses_a_drifted_anchor()
    test_patch_fails_closed_on_a_missing_file()
    test_patch_refuses_an_already_patched_base()
    test_installed_vllm_is_stock_or_fully_patched()
    test_dockerfile_orders_the_patch_before_the_tests()
    print("v41 kernel block size patch tests OK")
