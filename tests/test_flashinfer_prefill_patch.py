#!/usr/bin/env python3
"""Checks for the SM120 sparse-MLA prefill patch and the AOT mask.

Three things are pinned here:

1. ``overlay/patch_flashinfer_prefill_pbsx.py`` applies exactly the three
   intended edits, is idempotent, and fails closed on a drifted or partially
   applied source.
2. ``overlay/mask_flashinfer_aot.py`` removes the AOT module that would otherwise
   make FlashInfer ignore the patch.
3. When a real FlashInfer is installed (inside the image) its **installed**
   dispatcher really does accept and instantiate the 32-state compressed page
   that V4.1's ratio-2 layers need at ``--block-size 64`` — the assertion that
   the patch took effect rather than merely being applied to a fixture.

The stock fixture is assembled from the patch module's own anchor constants plus
a minimal C++ skeleton, so it cannot drift away from the anchors it is meant to
exercise and no vendored NVIDIA source is carried in this repo.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / "overlay"
PATCH = OVERLAY / "patch_flashinfer_prefill_pbsx.py"
MASK = OVERLAY / "mask_flashinfer_aot.py"
CHECK = OVERLAY / "check_flashinfer_dsv41.py"

sys.path.insert(0, str(ROOT / "overlay"))
sys.path.insert(0, str(ROOT / "tests"))

import check_flashinfer_dsv41 as chk  # noqa: E402
import patch_flashinfer_prefill_pbsx as pfx  # noqa: E402
from _fi_fixture import patched_cu, stock_cu, trap_cu  # noqa: E402


def test_fixture_is_the_stock_shape() -> None:
    """The fixture must reproduce the stock dispatcher, or the tests prove nothing."""
    accepted, instantiated = chk.prefill_page_blocks(stock_cu())
    assert accepted == frozenset({64, 2}), accepted
    assert instantiated == frozenset({64, 2}), instantiated
    assert pfx.IDENTITY in stock_cu()


def test_patch_admits_and_instantiates_every_needed_page_block() -> None:
    """The patched dispatcher's two sets must agree, and cover both V4.1 ratios.

    At ``--block-size 64`` the compressed pool presents 64 (ratio 1) and 32
    (ratio 2); 128 is retained for the geometry DeepGEMM may one day serve.
    """
    accepted, instantiated = chk.prefill_page_blocks(patched_cu())
    assert accepted == pfx.ACCEPTED_PAGE_BLOCKS, accepted
    assert instantiated == pfx.ACCEPTED_PAGE_BLOCKS, instantiated
    assert frozenset({64, 32}) <= accepted, "both V4.1 ratios must be servable"


def test_patch_applies_exactly_three_edits() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cu = Path(tmp) / chk.PREFILL_REL
        cu.parent.mkdir(parents=True)
        cu.write_text(stock_cu())
        before = cu.read_text()

        r = subprocess.run(
            [sys.executable, str(PATCH)],
            capture_output=True,
            text=True,
            env={**os.environ, "DSV41_FLASHINFER_ROOT": tmp},
        )
        assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
        after = cu.read_text()

        # The result must be exactly the expected patched text: three edits, no
        # other change. The acceptance test is one line either way, the full-tile
        # ladder grows by 6 lines (three more `else if` arms plus the fail-closed
        # `else { return false; }`), and the non-full-tile ladder by 4 (two more
        # arms).
        assert after == patched_cu(), after
        assert len(after.splitlines()) - len(before.splitlines()) == 10, after
        for edit in pfx.EDITS:
            assert after.count(edit.new) == 1, edit.name


def test_patch_is_idempotent() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cu = Path(tmp) / chk.PREFILL_REL
        cu.parent.mkdir(parents=True)
        cu.write_text(stock_cu())
        env = {**os.environ, "DSV41_FLASHINFER_ROOT": tmp}

        first = subprocess.run([sys.executable, str(PATCH)], capture_output=True, text=True, env=env)
        assert first.returncode == 0, first.stderr
        once = cu.read_text()

        second = subprocess.run(
            [sys.executable, str(PATCH)], capture_output=True, text=True, env=env
        )
        assert second.returncode == 0, second.stderr
        assert "already applied" in second.stdout, second.stdout
        assert cu.read_text() == once, "a second run must not change the file"


def test_patch_refuses_a_drifted_anchor() -> None:
    """A changed anchor must fail the build, not silently skip the fix."""
    with tempfile.TemporaryDirectory() as tmp:
        cu = Path(tmp) / chk.PREFILL_REL
        cu.parent.mkdir(parents=True)
        # Drift the acceptance test so its anchor no longer matches.
        drifted = stock_cu().replace("topk_extra % BI == 0", "topk_extra % BI != 0")
        cu.write_text(drifted)

        r = subprocess.run(
            [sys.executable, str(PATCH)],
            capture_output=True,
            text=True,
            env={**os.environ, "DSV41_FLASHINFER_ROOT": tmp},
        )
        assert r.returncode != 0, (r.returncode, r.stdout)
        assert "refusing to write" in r.stderr or "not unique" in r.stderr, r.stderr
        assert cu.read_text() == drifted, "a refused patch must not touch the file"


def test_patch_completes_a_partial_application() -> None:
    """A half-applied patch is completed, not refused and not double-applied.

    Widening the acceptance test without adding the branch is the dangerous
    intermediate state (the bare-else trap), so the patch must finish the job
    rather than leave it or bail out.
    """
    with tempfile.TemporaryDirectory() as tmp:
        cu = Path(tmp) / chk.PREFILL_REL
        cu.parent.mkdir(parents=True)
        partial = stock_cu().replace(pfx.ACCEPT_OLD_LINE, pfx.ACCEPT_NEW_LINE)
        cu.write_text(partial)
        # The intermediate state really is inconsistent, so the test is not vacuous.
        accepted, instantiated = chk.prefill_page_blocks(partial)
        assert accepted != instantiated, "the partial state must be inconsistent"

        r = subprocess.run(
            [sys.executable, str(PATCH)],
            capture_output=True,
            text=True,
            env={**os.environ, "DSV41_FLASHINFER_ROOT": tmp},
        )
        assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
        assert cu.read_text() == patched_cu(), cu.read_text()
        accepted, instantiated = chk.prefill_page_blocks(cu.read_text())
        assert accepted == instantiated == pfx.ACCEPTED_PAGE_BLOCKS


def test_bare_else_trap_is_detected() -> None:
    """Widening the acceptance test without the branches must be reported, not run.

    This is the failure the patch exists to prevent: stock's full-tile branch is
    `if (== 64) ... else { PBSX(2) }`, so widening only the acceptance test would
    admit wide page blocks and run the 2-token instantiation on them.
    """
    trap = trap_cu()
    accepted, instantiated = chk.prefill_page_blocks(trap)
    assert accepted == pfx.ACCEPTED_PAGE_BLOCKS, accepted
    assert instantiated == frozenset({64, 2}), instantiated
    assert accepted != instantiated, "the trap must be visible as a set mismatch"


def test_mask_removes_only_the_module() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        aot = Path(tmp) / "jit_cache"
        module = aot / "sparse_mla_sm120"
        module.mkdir(parents=True)
        (module / "sparse_mla_sm120.so").write_bytes(b"pretend aot binary")
        other = aot / "other_module"
        other.mkdir()
        (other / "other_module.so").write_bytes(b"keep me")

        env = {**os.environ, "DSV41_FLASHINFER_AOT_DIR": str(aot)}
        r = subprocess.run([sys.executable, str(MASK)], capture_output=True, text=True, env=env)
        assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
        assert not module.exists(), "the AOT module must be gone"
        assert other.exists(), "other AOT modules must survive"


def test_mask_is_idempotent() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        aot = Path(tmp) / "jit_cache"
        aot.mkdir(parents=True)
        env = {**os.environ, "DSV41_FLASHINFER_AOT_DIR": str(aot)}

        r = subprocess.run([sys.executable, str(MASK)], capture_output=True, text=True, env=env)
        assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
        assert "already masked" in r.stdout, r.stdout


def test_mask_fails_closed_on_a_missing_root() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        r = subprocess.run(
            [sys.executable, str(MASK)],
            capture_output=True,
            text=True,
            env={**os.environ, "DSV41_FLASHINFER_AOT_DIR": str(Path(tmp) / "nope")},
        )
        assert r.returncode != 0, (r.returncode, r.stdout)
        assert "does not exist" in r.stderr, r.stderr


def _real_flashinfer_installed() -> bool:
    spec = importlib.util.find_spec("flashinfer")
    return spec is not None and bool(spec.submodule_search_locations)


def test_installed_dispatcher_really_instantiates_32() -> None:
    """In the image: assert the patch took effect on the real installed file.

    Skipped where FlashInfer is absent (this Mac, and any host without the
    image), so the assertion runs exactly where it means something. The 32 is the
    point: it is the ratio-2 page at ``--block-size 64`` and the one value stock
    FlashInfer does not carry.
    """
    if not _real_flashinfer_installed():
        print("  (skipped: flashinfer not installed)")
        return
    source = chk.prefill_source()
    assert source is not None, "flashinfer is installed but the prefill source is missing"
    cu_text = source.read_text()
    accepted, instantiated = chk.prefill_page_blocks(cu_text)
    assert accepted == instantiated, (
        f"installed dispatcher accepts {sorted(accepted)} but instantiates "
        f"{sorted(instantiated)}"
    )
    assert 32 in accepted, (
        f"installed dispatcher does not accept a 32-state compressed page "
        f"(accepts {sorted(accepted)}); overlay/patch_flashinfer_prefill_pbsx.py "
        "did not take effect in this image"
    )
    assert 32 in instantiated, sorted(instantiated)
    # Both V4.1 ratios at --block-size 64 must be servable, through both ladders.
    for needed in (64, 32):
        defects = chk.dispatch_defects(cu_text, [needed])
        assert defects == [], (needed, defects)


def dockerfile_ordering_defects(root: Path) -> list[str]:
    """Why `root`'s Dockerfile does not apply the fix before testing it.

    Takes the tree root rather than using the module constant, so the same check
    can be exercised against a reproduction of the image layout — where the
    Dockerfile has to be present too, or the build fails on the test instead of
    on the thing it is testing.
    """
    path = root / "Dockerfile"
    if not path.is_file():
        return [f"no Dockerfile at {path}; the ordering cannot be verified"]
    text = path.read_text()
    defects = []
    patch_at = text.find("patch_flashinfer_prefill_pbsx.py")
    mask_at = text.find("mask_flashinfer_aot.py")
    test_at = text.find("test_flashinfer_prefill_patch.py")
    if patch_at == -1:
        defects.append("the Dockerfile does not apply the FlashInfer prefill patch")
    if mask_at == -1:
        defects.append("the Dockerfile does not mask the AOT module")
    if test_at == -1:
        defects.append("the Dockerfile does not run this test suite")
    if not defects:
        if patch_at > test_at:
            defects.append(
                "patch_flashinfer_prefill_pbsx.py must run before "
                "test_flashinfer_prefill_patch.py"
            )
        if mask_at > test_at:
            defects.append("the AOT mask must run before this test suite")
    return defects


def test_dockerfile_orders_the_patch_before_the_tests() -> None:
    """Guard the ordering in the Dockerfile: the patch must precede the tests.

    If the image ever ran this test suite before the patch, the assertion above
    would fail the build — which is the intent, but this makes the requirement
    explicit rather than implied by step order. `tests/test_image_layout.py`
    covers the other half: that everything these tests read is actually copied.
    """
    defects = dockerfile_ordering_defects(ROOT)
    assert defects == [], defects


if __name__ == "__main__":
    test_fixture_is_the_stock_shape()
    test_patch_admits_and_instantiates_every_needed_page_block()
    test_patch_applies_exactly_three_edits()
    test_patch_is_idempotent()
    test_patch_refuses_a_drifted_anchor()
    test_patch_completes_a_partial_application()
    test_bare_else_trap_is_detected()
    test_mask_removes_only_the_module()
    test_mask_is_idempotent()
    test_mask_fails_closed_on_a_missing_root()
    test_installed_dispatcher_really_instantiates_32()
    test_dockerfile_orders_the_patch_before_the_tests()
    print("flashinfer prefill patch tests OK")
