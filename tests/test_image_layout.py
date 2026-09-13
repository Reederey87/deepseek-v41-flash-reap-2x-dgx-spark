#!/usr/bin/env python3
"""Check that the image layout the Dockerfile creates is actually workable.

The build runs several test files *inside* the image, from `/opt/dsv41`, which
contains only what the Dockerfile `COPY`s. A build-time test can therefore read
only what was copied — and a test that reads a missing file does not fail with a
clear message, it fails with `FileNotFoundError` and the image cannot be built at
all. That happened twice: first with `Dockerfile`, then with `start.sh`.

Two checks, both derived from the artefacts rather than hand-maintained lists,
because a hand-maintained list is what let the second one through:

1. **Input coverage.** Every repo-relative path any file under `tests/` reaches
   for through its `ROOT / "..."` anchor must be covered by a `COPY`.
2. **Layout reproduction.** Materialize exactly the `COPY`ed content in a
   temporary directory and run every build-time test suite there, from that
   directory, requiring success. This is the same thing the image does, so a
   missing input fails here instead of during a 40-minute image build.

It also guards the base image: the kit shipped one whose vLLM has no
`models/deepseek_v4_1`, so the build could not run at all. See RESEARCH.md
§11.9.12.

This suite is host-side: it needs the recipe and the pre-Dockerfile tree, so it
is deliberately not part of the in-image test run.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "Dockerfile"

# Where the image puts the kit.
IMAGE_PREFIX = "/opt/dsv41/"

# `ROOT / "overlay" / "x.py"` and `ROOT / "start.sh"` both start like this.
INPUT_RE = re.compile(r'ROOT\s*/\s*"([^"]+)"')

# Local test modules a build-time test pulls in, e.g. `from _fi_fixture import ...`.
LOCAL_IMPORT_RE = re.compile(r"^\s*(?:from|import)\s+(_?[a-z][\w]*)\b", re.MULTILINE)

# The Dockerfile's test RUN, e.g. `python3 /opt/dsv41/tests/test_x.py`.
TEST_RUN_RE = re.compile(r"python3\s+" + re.escape(IMAGE_PREFIX) + r"(tests/[\w./-]+\.py)")

# `ARG BASE=<ref>`, the base image the kit overlays.
BASE_RE = re.compile(r"^ARG BASE=(\S+)", re.MULTILINE)

# The base this kit must **not** inherit: a fork build whose vLLM has no
# `deepseek_v4_1`, so the build-time V4.1 guard fails on the first RUN. Verified
# on the cluster; see RESEARCH.md 11.9.12.
NOT_A_BASE = "vllm/vllm-openai:glm53-flash-arm64-cu130"


def copies(dockerfile_text: str) -> dict[str, str]:
    """`{image-relative destination: repo-relative source}` from the COPY lines."""
    found: dict[str, str] = {}
    for line in dockerfile_text.splitlines():
        line = line.strip()
        if not line.startswith("COPY "):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        src, dest = parts[1], parts[2]
        if not dest.startswith(IMAGE_PREFIX):
            continue
        found[dest[len(IMAGE_PREFIX) :].rstrip("/")] = src
    return found


def build_time_tests(dockerfile_text: str) -> list[str]:
    """The test files the image runs at build time, image-relative."""
    return TEST_RUN_RE.findall(dockerfile_text)


def is_copied(path: Path, copied: dict[str, str]) -> bool:
    """Whether a repo-relative path arrives in the image through some COPY."""
    as_posix = path.as_posix()
    for dest in copied:
        if as_posix == dest or as_posix.startswith(dest + "/"):
            return True
    return False


def reachable_inputs(dockerfile_text: str) -> set[Path]:
    """Every first path component the *build-time* test tree reaches for.

    Scoped to the files the image actually runs plus the local modules they
    import, so a host-only file (this one) cannot make the scan vacuous or
    noisy. The whole tree is covered anyway by running the suites in the
    reproduced layout below.
    """
    root = ROOT / "tests"
    pending = [Path(rel).name for rel in build_time_tests(dockerfile_text)]
    seen: set[str] = set()
    found: set[Path] = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        path = root / name
        if not path.is_file():
            continue
        text = path.read_text()
        for target in INPUT_RE.findall(text):
            found.add(Path(target))
        for module in LOCAL_IMPORT_RE.findall(text):
            if (root / f"{module}.py").is_file():
                pending.append(f"{module}.py")
    return found


def materialize(destination: Path, copied: dict[str, str]) -> None:
    """Reproduce the image's `/opt/dsv41` from the COPY map."""
    for dest, src in copied.items():
        source = ROOT / src
        target = destination / dest
        if source.is_dir():
            shutil.copytree(
                source, target, ignore=shutil.ignore_patterns("__pycache__")
            )
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def test_every_input_a_build_time_test_reads_is_copied() -> None:
    """A test that cannot read its input fails the build, so the inputs must arrive."""
    copied = copies(DOCKERFILE.read_text())
    missing = sorted(
        str(path)
        for path in reachable_inputs(DOCKERFILE.read_text())
        if not is_copied(path, copied)
    )
    assert missing == [], (
        f"these paths are read by a file under tests/ but no COPY brings them "
        f"into the image: {missing}; copied destinations are {sorted(copied)}"
    )


def test_the_dockerfile_runs_the_test_files_this_suite_reproduces() -> None:
    """Guard against the reproduction silently covering nothing."""
    tests = build_time_tests(DOCKERFILE.read_text())
    assert tests, "the Dockerfile no longer runs any test file at build time"
    for rel in tests:
        assert (ROOT / rel).is_file(), f"{rel} is run at build time but is not in the repo"


def test_the_build_time_suites_pass_in_the_image_layout() -> None:
    """Run every build-time suite against exactly what the image will contain.

    This is the check that would have caught both missing-COPY defects: the test
    suite is executed from the reproduced layout, so a file the suite reads but
    the Dockerfile does not copy fails here rather than mid-build.
    """
    copied = copies(DOCKERFILE.read_text())
    tests = build_time_tests(DOCKERFILE.read_text())
    assert copied and tests

    with tempfile.TemporaryDirectory() as tmp:
        layout = Path(tmp) / "dsv41"
        layout.mkdir()
        materialize(layout, copied)

        # The reproduction must be a *subset* of the repo: anything the tests
        # reach for outside the COPY map has to fail, or the test is vacuous.
        assert not (layout / ".venv").exists()
        assert (layout / "tests").is_dir()

        failures = []
        for rel in tests:
            proc = subprocess.run(
                [sys.executable, str(layout / rel)],
                capture_output=True,
                text=True,
                cwd=layout,
                env={"PATH": "/usr/bin:/bin", "HOME": tmp},
            )
            if proc.returncode != 0:
                failures.append((rel, proc.returncode, proc.stdout, proc.stderr))
        assert failures == [], "\n".join(
            f"{rel} exit {rc}\n--- stdout ---\n{out}\n--- stderr ---\n{err}"
            for rel, rc, out, err in failures
        )


def test_the_base_is_a_pinned_image_that_carries_v4_1() -> None:
    """The kit shipped a base whose vLLM has no V4.1, so the build could not run.

    `vllm/vllm-openai:glm53-flash-arm64-cu130` is a pulled fork build
    (`0.1.dev20051+g487ecf187`) whose `vllm/models/` has `deepseek_v32`,
    `deepseek_v4` and `glm5next` and **no** `deepseek_v4_1`, so the build-time
    guard below the `FROM` failed on the first `RUN`. The base is now the public
    day-0 V4.1 image. This cannot be checked by running it here, so it is checked
    structurally: one pinned base, and not that one.
    """
    text = DOCKERFILE.read_text()
    bases = BASE_RE.findall(text)
    assert len(bases) == 1, f"expected exactly one `ARG BASE=`, found {bases}"
    base = bases[0]
    assert base != NOT_A_BASE, "the GLM fork base carries no V4.1; see RESEARCH.md 11.9.12"
    assert "deepseekv41" in base, f"the base must be a V4.1 image: {base}"
    # Unpinned would follow a moving tag, which the whole receipt chain keys on.
    assert "@sha256:" in base, f"the base must be digest-pinned: {base}"
    # The header must state the two facts an operator needs in order to pick a
    # base: that it has to carry V4.1, and that the arm64 manifest exists. The
    # previous version of this test also required the header to name the wrong
    # base; that name is gone from the docs on purpose, and the guard that
    # matters is the `base != NOT_A_BASE` assertion above.
    assert "deepseek_v4_1" in text, "the header must state the V4.1 requirement"
    assert "arm64" in text, "the header must record the arm64 manifest"


def test_the_guard_the_base_must_satisfy_is_still_in_the_dockerfile() -> None:
    """The V4.1 assertion is the thing that makes a wrong base fail loudly."""
    text = DOCKERFILE.read_text()
    assert "models' / 'deepseek_v4_1'" in text or "deepseek_v4_1" in text, text
    assert "assert p.is_dir()" in text, text


if __name__ == "__main__":
    test_every_input_a_build_time_test_reads_is_copied()
    test_the_dockerfile_runs_the_test_files_this_suite_reproduces()
    test_the_base_is_a_pinned_image_that_carries_v4_1()
    test_the_guard_the_base_must_satisfy_is_still_in_the_dockerfile()
    test_the_build_time_suites_pass_in_the_image_layout()
    print("image layout tests OK")
