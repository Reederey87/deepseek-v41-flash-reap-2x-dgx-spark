#!/usr/bin/env python3
"""Checks for the compiled-module pre-flight (`overlay/check_flashinfer_cache.py`).

The image masks FlashInfer's prebuilt AOT binary, so the compiled module is a
required artifact and its absence or staleness is a boot failure. A filename
match is not evidence — a module keyed on another FlashInfer version is never
looked up, a cache from an earlier image is stale, and an empty file matches any
glob — so these tests pin the receipt checks that replace it.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "overlay" / "check_flashinfer_cache.py"
sys.path.insert(0, str(ROOT / "overlay"))

import check_flashinfer_cache as cc  # noqa: E402

IMAGE_ID = "sha256:" + "a" * 64
OTHER_IMAGE_ID = "sha256:" + "b" * 64
VERSION = "0.6.17"
ARCH = "12_1a"
REL = Path(VERSION) / ARCH / "cached_ops" / "sparse_mla_sm120" / "sparse_mla_sm120.so"
# A real module is a sub-megabyte object, not a tens-of-megabytes one: the stock
# AOT build in the base image is 967,520 bytes and the patched per-node prebuild
# is 972,952. The payload here just has to clear the placeholder floor.
PAYLOAD = b"\x7fELF" + b"pretend a compiled flashinfer module" * 65536
# The two sizes actually observed on the cluster. A floor above these refuses a
# correct build, which is what happened on the first boot attempt.
STOCK_AOT_BYTES = 967_520
PATCHED_BUILD_BYTES = 972_952


def _cache(tmp: str, *, image_id: str = IMAGE_ID, version: str = VERSION,
           rel: Path | None = None, payload: bytes = PAYLOAD,
           bytes_field: str | None = None, sha_field: str | None = None,
           drop: tuple[str, ...] = ()) -> Path:
    """A cache tree with a receipt, as `scripts/build_flashinfer.sh` leaves it.

    `rel=None` uses the path the stub FlashInfer resolves, i.e. the one a correct
    receipt must name; pass an explicit `rel` to model a receipt that names
    something else (another arch, a missing component, an extra one).
    """
    cache = Path(tmp) / "flashinfer"
    cache.mkdir(parents=True, exist_ok=True)
    rel = REL if rel is None else rel
    module = cache / rel
    module.parent.mkdir(parents=True, exist_ok=True)
    module.write_bytes(payload)
    fields = {
        "image_ref": "dsv41-reap-selfbuild",
        "image_id": image_id,
        "flashinfer_version": version,
        "module": str(rel),
        "bytes": bytes_field if bytes_field is not None else str(len(payload)),
        "sha256": sha_field if sha_field is not None else hashlib.sha256(payload).hexdigest(),
    }
    for key in drop:
        fields.pop(key, None)
    text = "".join(f"{k}={v}\n" for k, v in fields.items())
    (cache / cc.RECEIPT_NAME).write_text("# written by scripts/build_flashinfer.sh\n" + text)
    return cache


def _run(cache: Path, image_id: str = IMAGE_ID, *, stub: str | None = None):
    args = [sys.executable, str(SCRIPT), "--cache", str(cache), "--image-id", image_id]
    env = {**os.environ}
    if stub:
        # PREPEND, so the stub shadows the real install that the image has, and
        # so anything the image already puts on PYTHONPATH keeps working.
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = stub + (os.pathsep + existing if existing else "")
    return subprocess.run(args, capture_output=True, text=True, env=env)


def _stub(tmp: str, *, version: str | None = VERSION, arch: str = ARCH,
          is_aot: bool | None = False, resolved: Path | None = None) -> str:
    """A minimal stand-in for the installed flashinfer package.

    Reproduces the two things the checker asks FlashInfer for: the cache root,
    and `JitSpec.jit_library_path` — the exact path the runtime loads.

    `version=None` omits `__version__` and the distribution metadata (an
    unreadable install). `is_aot=None` omits the module generator entirely (an
    introspection failure). `resolved` overrides the path FlashInfer would
    resolve, which is how a runtime that computes a different key is modelled.

    The stub shadows the **distribution metadata** as well as the package. The
    image has a real FlashInfer installed, so shadowing only the package left
    `importlib.metadata.version("flashinfer-python")` reading the real 0.6.18 and
    this suite failed at image build time — the version case is only hermetic if
    both are shadowed. A `dist-info` directory earlier on `sys.path` wins, and a
    METADATA with no `Version:` line makes the lookup return None rather than
    falling through to the real install.
    """
    root = Path(tmp) / "stub"
    fi = root / "flashinfer"
    mla = fi / "mla"
    jit = fi / "jit"
    mla.mkdir(parents=True)
    jit.mkdir(parents=True)
    (fi / "__init__.py").write_text(
        "" if version is None else f"__version__ = {version!r}\n"
    )
    (mla / "__init__.py").write_text("")
    (jit / "__init__.py").write_text("")
    dist = root / f"flashinfer_python-{version or '0.0.0'}.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text(
        "Name: flashinfer-python\n"
        + ("" if version is None else f"Version: {version}\n")
    )
    cache_dir = Path("/root/.cache/flashinfer")
    resolved = (
        resolved
        if resolved is not None
        else cache_dir
        / (version or "unknown")
        / arch
        / "cached_ops"
        / "sparse_mla_sm120"
        / "sparse_mla_sm120.so"
    )
    (jit / "env.py").write_text(f"from pathlib import Path\nFLASHINFER_CACHE_DIR = Path({str(cache_dir)!r})\n")
    generator = (
        ""
        if is_aot is None
        else "import types\n"
        "def gen_sparse_mla_sm120_module():\n"
        "    return types.SimpleNamespace(\n"
        f"        is_aot={bool(is_aot)!r},\n"
        f"        jit_library_path=__import__('pathlib').Path({str(resolved)!r}),\n"
        "    )\n"
    )
    (mla / "_sparse_mla_sm120.py").write_text(generator)
    return str(root)


def test_parse_receipt_ignores_comments_and_blanks() -> None:
    fields = cc.parse_receipt("# a comment\n\nimage_id=x\nmodule = a/b.so\n")
    assert fields == {"image_id": "x", "module": "a/b.so"}, fields


def test_a_verified_cache_passes() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cache = _cache(tmp)
        r = _run(cache, stub=_stub(tmp))
        assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
        assert "will be reused" in r.stdout, r.stdout


def test_a_missing_receipt_is_refused() -> None:
    """No receipt means the module was never verified on this node."""
    with tempfile.TemporaryDirectory() as tmp:
        cache = _cache(tmp)
        (cache / cc.RECEIPT_NAME).unlink()
        r = _run(cache, stub=_stub(tmp))
        assert r.returncode == 5, (r.returncode, r.stdout, r.stderr)
        assert "no prebuild receipt" in r.stderr, r.stderr
        assert "build_flashinfer.sh" in r.stderr, r.stderr


def test_a_real_sized_artifact_is_accepted() -> None:
    """The floor must sit below the real artifact, not above it.

    Both sizes here are measured, not invented: the stock AOT module shipped in
    the base image, and the patched module `build_flashinfer.sh` produced on the
    cluster. A floor set from the assumption that such a module is tens of
    megabytes refused a correct build on the first boot attempt, which is a
    false negative on the critical path.
    """
    for size, label in (
        (STOCK_AOT_BYTES, "stock AOT module"),
        (PATCHED_BUILD_BYTES, "patched prebuild"),
    ):
        payload = b"\x7fELF" + b"\0" * (size - 4)
        with tempfile.TemporaryDirectory() as tmp:
            cache = _cache(tmp, payload=payload)
            r = _run(cache, stub=_stub(tmp))
            assert r.returncode == 0, (label, r.returncode, r.stdout, r.stderr)
    # And the floor must still be a floor: one byte below it is refused.
    with tempfile.TemporaryDirectory() as tmp:
        cache = _cache(tmp, payload=b"\x7fELF" + b"\0" * (cc.MIN_MODULE_BYTES - 5))
        r = _run(cache, stub=_stub(tmp))
        assert r.returncode == 5, (r.returncode, r.stdout, r.stderr)
        assert "is only" in r.stderr, r.stderr
    assert cc.MIN_MODULE_BYTES < STOCK_AOT_BYTES, (
        "the floor is above the real artifact size and will refuse a good build"
    )


def test_a_receipt_missing_fields_is_refused() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cache = _cache(tmp, drop=("sha256", "bytes"))
        r = _run(cache, stub=_stub(tmp))
        assert r.returncode == 5, (r.returncode, r.stdout, r.stderr)
        assert "is missing" in r.stderr and "sha256" in r.stderr, r.stderr


def test_a_cache_from_another_image_is_refused() -> None:
    """A rebuilt image invalidates the cache; the receipt pins the image."""
    with tempfile.TemporaryDirectory() as tmp:
        cache = _cache(tmp, image_id=OTHER_IMAGE_ID)
        r = _run(cache, stub=_stub(tmp))
        assert r.returncode == 5, (r.returncode, r.stdout, r.stderr)
        assert "the cache is stale" in r.stderr, r.stderr


def test_a_cache_keyed_on_another_version_is_refused() -> None:
    """FlashInfer keys its cache on the version; a mismatched key is never read."""
    with tempfile.TemporaryDirectory() as tmp:
        other = Path("0.6.18.post1") / ARCH / "cached_ops" / "sparse_mla_sm120" / "sparse_mla_sm120.so"
        cache = _cache(tmp, version="0.6.18.post1", rel=other)
        r = _run(cache, stub=_stub(tmp))
        assert r.returncode == 5, (r.returncode, r.stdout, r.stderr)
        assert "would never be looked up" in r.stderr, r.stderr


def test_a_receipt_naming_another_arch_is_refused() -> None:
    """A module built for another arch is never looked up, so it is a cache miss."""
    with tempfile.TemporaryDirectory() as tmp:
        wrong = Path(VERSION) / "wrong-arch" / "cached_ops" / "sparse_mla_sm120" / "sparse_mla_sm120.so"
        cache = _cache(tmp, rel=wrong)
        r = _run(cache, stub=_stub(tmp))
        assert r.returncode == 5, (r.returncode, r.stdout, r.stderr)
        assert "but the runtime loads" in r.stderr, r.stderr


def test_a_receipt_missing_the_arch_component_is_refused() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        wrong = Path(VERSION) / "cached_ops" / "sparse_mla_sm120" / "sparse_mla_sm120.so"
        cache = _cache(tmp, rel=wrong)
        r = _run(cache, stub=_stub(tmp))
        assert r.returncode == 5, (r.returncode, r.stdout, r.stderr)
        assert "but the runtime loads" in r.stderr, r.stderr


def test_a_receipt_with_an_extra_component_is_refused() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        wrong = (
            Path(VERSION) / "extra" / ARCH / "cached_ops" / "sparse_mla_sm120" / "sparse_mla_sm120.so"
        )
        cache = _cache(tmp, rel=wrong)
        r = _run(cache, stub=_stub(tmp))
        assert r.returncode == 5, (r.returncode, r.stdout, r.stderr)
        assert "but the runtime loads" in r.stderr, r.stderr


def test_a_receipt_naming_the_path_outside_the_cache_is_refused() -> None:
    """A path escaping the cache root is not the runtime's key."""
    with tempfile.TemporaryDirectory() as tmp:
        cache = _cache(tmp, rel=Path("../elsewhere/sparse_mla_sm120.so"))
        r = _run(cache, stub=_stub(tmp))
        assert r.returncode == 5, (r.returncode, r.stdout, r.stderr)
        assert "but the runtime loads" in r.stderr, r.stderr


def test_a_runtime_that_resolves_another_path_is_refused() -> None:
    """The check follows the runtime, so a different runtime key fails.

    This is the property that makes the check about reuse rather than about a
    filename: if FlashInfer would resolve a different path than the receipt
    names — a different arch, a different cache root layout — it is a miss.
    """
    with tempfile.TemporaryDirectory() as tmp:
        cache = _cache(tmp)  # receipt names the canonical path
        other = Path("/root/.cache/flashinfer") / VERSION / "12_0a" / "cached_ops" / "sparse_mla_sm120" / "sparse_mla_sm120.so"
        r = _run(cache, stub=_stub(tmp, resolved=other))
        assert r.returncode == 5, (r.returncode, r.stdout, r.stderr)
        assert "but the runtime loads" in r.stderr, r.stderr


def test_an_unreadable_runtime_path_is_refused() -> None:
    """If the runtime key cannot be determined, the receipt cannot be checked."""
    with tempfile.TemporaryDirectory() as tmp:
        cache = _cache(tmp)
        r = _run(cache, stub=_stub(tmp, is_aot=None))
        assert r.returncode == 5, (r.returncode, r.stdout, r.stderr)
        assert "cannot be determined" in r.stderr, r.stderr


def test_the_stub_shadows_a_real_flashinfer_install() -> None:
    """The suite must not depend on flashinfer being absent.

    This ran green on macOS, where there is no flashinfer, and failed at image
    build time, where there is a real 0.6.18: the stub shadowed the package but
    not the distribution metadata, so the version case read the real install. A
    version that no real install would report is the check that the shadowing
    covers both.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fake = "0.0.0-stub"
        rel = Path(fake) / ARCH / "cached_ops" / "sparse_mla_sm120" / "sparse_mla_sm120.so"
        cache = _cache(tmp, version=fake, rel=rel)
        r = _run(cache, stub=_stub(tmp, version=fake))
        assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
    # And the unreadable case must not fall through to a real install either.
    # A separate stub root, because two dist-info dirs for the same distribution
    # in one directory resolve in filesystem order.
    with tempfile.TemporaryDirectory() as tmp:
        cache = _cache(tmp)
        r2 = _run(cache, stub=_stub(tmp, version=None))
        assert r2.returncode == 5, (r2.returncode, r2.stdout, r2.stderr)
        assert "version cannot be read" in r2.stderr, r2.stderr


def test_a_missing_artifact_is_refused() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cache = _cache(tmp)
        (cache / REL).unlink()
        r = _run(cache, stub=_stub(tmp))
        assert r.returncode == 5, (r.returncode, r.stdout, r.stderr)
        assert "which is not a file" in r.stderr, r.stderr


def test_an_empty_artifact_is_refused() -> None:
    """The reviewer's reproduction: an empty file matched the old filename glob."""
    with tempfile.TemporaryDirectory() as tmp:
        cache = _cache(tmp, payload=b"")
        r = _run(cache, stub=_stub(tmp))
        assert r.returncode == 5, (r.returncode, r.stdout, r.stderr)
        assert "is only 0 bytes" in r.stderr, r.stderr


def test_a_truncated_artifact_is_refused() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cache = _cache(tmp, bytes_field=str(len(PAYLOAD) * 2))
        r = _run(cache, stub=_stub(tmp))
        assert r.returncode == 5, (r.returncode, r.stdout, r.stderr)
        assert "truncated or was replaced" in r.stderr, r.stderr


def test_a_replaced_artifact_is_refused() -> None:
    """Same size, different bytes: only the digest catches it."""
    with tempfile.TemporaryDirectory() as tmp:
        cache = _cache(tmp, sha_field="0" * 64)
        r = _run(cache, stub=_stub(tmp))
        assert r.returncode == 5, (r.returncode, r.stdout, r.stderr)
        assert "hashes to" in r.stderr, r.stderr


def test_a_present_aot_binary_is_refused() -> None:
    """The AOT binary would be returned instead of the compiled module."""
    with tempfile.TemporaryDirectory() as tmp:
        cache = _cache(tmp)
        r = _run(cache, stub=_stub(tmp, is_aot=True))
        assert r.returncode == 5, (r.returncode, r.stdout, r.stderr)
        assert "mask_flashinfer_aot.py" in r.stderr, r.stderr


def test_an_unreadable_aot_state_is_refused() -> None:
    """An AOT state that cannot be read is not a clean bill of health."""
    with tempfile.TemporaryDirectory() as tmp:
        cache = _cache(tmp)
        r = _run(cache, stub=_stub(tmp, is_aot=None))
        assert r.returncode == 5, (r.returncode, r.stdout, r.stderr)
        assert "cannot be determined" in r.stderr, r.stderr


def test_an_unreadable_version_is_refused() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cache = _cache(tmp)
        r = _run(cache, stub=_stub(tmp, version=None))
        assert r.returncode == 5, (r.returncode, r.stdout, r.stderr)
        assert "version cannot be read" in r.stderr, r.stderr


def test_a_missing_cache_directory_is_refused() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        r = _run(Path(tmp) / "nope", stub=_stub(tmp))
        assert r.returncode == 5, (r.returncode, r.stdout, r.stderr)
        assert "no cache directory" in r.stderr, r.stderr


def test_the_build_script_writes_what_the_checker_reads() -> None:
    """The two halves must agree on the receipt name and the fields.

    `build_flashinfer.sh` is shell and the checker is Python, so nothing else
    keeps them in step; the shared name and the required keys are asserted here.
    """
    script = (ROOT / "scripts" / "build_flashinfer.sh").read_text()
    assert f'RECEIPT_NAME="{cc.RECEIPT_NAME}"' in script, cc.RECEIPT_NAME
    for key in cc.REQUIRED_KEYS:
        # The container prints the artifact fields; the host records `image_id`
        # itself, because it is the one field the container cannot know.
        assert (
            f'print(f"{key}=' in script or f"printf '{key}=" in script
        ), key
    # The build must identify the artifact through FlashInfer's own resolution,
    # not by globbing: a glob can pick up a module for another arch.
    assert "spec.jit_library_path" in script, script
    assert "rglob" not in script, script


def test_the_launcher_runs_this_checker_on_both_nodes() -> None:
    """A one-node check would let a stale worker cache through."""
    launcher = (ROOT / "start.sh").read_text()
    assert "check_flashinfer_cache.py" in launcher, launcher
    assert "--image-id" in launcher, launcher
    assert "for label in head worker" in launcher, launcher


if __name__ == "__main__":
    test_parse_receipt_ignores_comments_and_blanks()
    test_a_verified_cache_passes()
    test_a_missing_receipt_is_refused()
    test_a_receipt_missing_fields_is_refused()
    test_a_cache_from_another_image_is_refused()
    test_a_cache_keyed_on_another_version_is_refused()
    test_a_receipt_naming_another_arch_is_refused()
    test_a_receipt_missing_the_arch_component_is_refused()
    test_a_receipt_with_an_extra_component_is_refused()
    test_a_receipt_naming_the_path_outside_the_cache_is_refused()
    test_a_runtime_that_resolves_another_path_is_refused()
    test_an_unreadable_runtime_path_is_refused()
    test_the_stub_shadows_a_real_flashinfer_install()
    test_a_missing_artifact_is_refused()
    test_an_empty_artifact_is_refused()
    test_a_real_sized_artifact_is_accepted()
    test_a_truncated_artifact_is_refused()
    test_a_replaced_artifact_is_refused()
    test_a_present_aot_binary_is_refused()
    test_an_unreadable_aot_state_is_refused()
    test_an_unreadable_version_is_refused()
    test_a_missing_cache_directory_is_refused()
    test_the_build_script_writes_what_the_checker_reads()
    test_the_launcher_runs_this_checker_on_both_nodes()
    print("flashinfer cache pre-check tests OK")
