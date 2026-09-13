#!/usr/bin/env python3
"""Check the weight-staging script against the contract it actually has to meet.

`scripts/verify_weights.py` ends a failed verification by telling the operator to
run `scripts/fetch_weights.sh`, and for a long time there was no such file. It
now exists, and it is the piece that turns a 385.5 GiB two-repository staging job
into one command, so it is tested by running it rather than by reading it.

The downloader is stubbed. That is deliberate: the defects this guards against
are all in *how the script drives the CLI*, not in the CLI:

* `--include a b` looks like two patterns and is not — the second name becomes a
  positional filter and the pattern is dropped, so only one shard arrives;
* `hf download` picks its own cache unless `--cache-dir` is passed, so a custom
  `HF_HOME_HOST` downloads somewhere the script then cannot find;
* `df` on a cache directory that does not exist yet fails, and under
  `set -o pipefail` that aborts the script before its own fallback;
* a revision pinned for one repository must not be requested from the other;
* a re-run must reuse what is staged instead of re-pulling 385 GiB.

The stub records its argv and materializes real safetensors shards, so the run
ends by actually executing `verify_weights.py` over the staged tree.
"""

from __future__ import annotations

import json
import os
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FETCH = ROOT / "scripts" / "fetch_weights.sh"
VERIFY = ROOT / "scripts" / "verify_weights.py"
START = ROOT / "start.sh"

REAP = "LibertAIDAI/DeepSeek-V4.1-Flash-REAP-256E"
BASE = "deepseek-ai/DeepSeek-V4.1-Flash"
ENGRAM_SHARDS = (
    "model-00047-of-00048.safetensors",
    "model-00048-of-00048.safetensors",
)
# 46 published shards plus the two that must come from the base checkpoint.
SHARDS = [f"model-{i:05d}-of-00048.safetensors" for i in range(1, 49)]
PUBLISHED = SHARDS[:46]
FROM_BASE = SHARDS[46:]

SCRIPT_REF_RE = re.compile(r"scripts/([\w.-]+\.(?:sh|py))")
CODE_GLOBS = ("overlay/*.py", "scripts/*.py", "scripts/*.sh", "tests/*.py", "tests/*.sh")


def _safetensors(keys: list[str]) -> bytes:
    """A minimal but structurally valid safetensors file."""
    header = {
        key: {"dtype": "F32", "shape": [1], "data_offsets": [i * 4, (i + 1) * 4]}
        for i, key in enumerate(keys)
    }
    raw = json.dumps(header).encode()
    raw += b" " * ((8 - len(raw) % 8) % 8)
    return struct.pack("<Q", len(raw)) + raw + b"\0" * (4 * len(keys))


# The stub downloader: records argv, then materializes what the script asked for.
HF_STUB = '''#!/usr/bin/env python3
import json, os, struct, sys
from pathlib import Path

args = sys.argv[1:]
with open(os.environ["HF_STUB_LOG"], "a") as fh:
    fh.write(json.dumps(args) + "\\n")

if args[0] != "download":
    print(f"STUB ERROR: expected the download subcommand, got {args[0]!r}", file=sys.stderr)
    raise SystemExit(5)
repo = args[1]
rest = args[2:]
cache = local = None
revision = "main"
positional = []
i = 0
while i < len(rest):
    a = rest[i]
    if a == "--cache-dir":
        cache = rest[i + 1]; i += 2; continue
    if a == "--local-dir":
        local = rest[i + 1]; i += 2; continue
    if a == "--revision":
        revision = rest[i + 1]; i += 2; continue
    if a.startswith("-"):
        i += 1; continue
    positional.append(a); i += 1
# The real CLI rejects these together: "Cannot use both `--local-dir` and
# `--cache-dir` at the same time". Modelling it is what makes the tests able to
# catch the combination, which is how it was found on the node.
if local is not None and cache is not None:
    print("STUB ERROR: Cannot use both `--local-dir` and `--cache-dir`", file=sys.stderr)
    raise SystemExit(6)


def write(path, keys):
    header = {k: {"dtype": "F32", "shape": [1], "data_offsets": [n * 4, (n + 1) * 4]}
              for n, k in enumerate(keys)}
    raw = json.dumps(header).encode()
    raw += b" " * ((8 - len(raw) % 8) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\\0" * (4 * len(keys)))


if "REAP-256E" in repo:
    if cache is None:
        print("STUB ERROR: the cached download needs --cache-dir", file=sys.stderr)
        raise SystemExit(3)
    snap = Path(cache) / "models--LibertAIDAI--DeepSeek-V4.1-Flash-REAP-256E" / "snapshots" / revision
    snap.mkdir(parents=True, exist_ok=True)
    weight_map = {}
    total = 0
    for n in range(1, 47):
        shard = f"model-{n:05d}-of-00048.safetensors"
        key = f"published.{n}"
        write(snap / shard, [key])
        weight_map[key] = shard
        total += 4
    for n, shard in ((47, "model-00047-of-00048.safetensors"),
                     (48, "model-00048-of-00048.safetensors")):
        key = f"engram.{n}"
        weight_map[key] = shard
        total += 4
    (snap / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map})
    )
    (snap / "config.json").write_text("{}")
    print(f"stub: published 46 shards at {snap}")
else:
    if local is None:
        print("STUB ERROR: base download without --local-dir", file=sys.stderr)
        raise SystemExit(4)
    # Only the requested filenames, which is the point of the positional form.
    for name in positional:
        n = int(name.split("-")[1])
        write(Path(local) / name, [f"engram.{n}"])
    print(f"stub: staged {positional} into {local}")
'''


def _run(tmp: str, *, extra_env: dict | None = None, args: tuple[str, ...] = ()):
    """Run fetch_weights.sh with a stub downloader and a nested fresh cache."""
    bindir = Path(tmp) / "bin"
    bindir.mkdir(exist_ok=True)
    stub = bindir / "hf"
    stub.write_text(HF_STUB)
    stub.chmod(0o755)
    log = Path(tmp) / "hf-argv.log"
    cache = Path(tmp) / "cache" / "nested" / "deep"  # does not exist yet
    env = {
        **os.environ,
        "PATH": f"{bindir}:{os.environ.get('PATH', '')}",
        "HF_STUB_LOG": str(log),
        "HF_HOME_HOST": str(cache),
        # Small, overridable budgets so the gate is exercised without 385 GiB free.
        "REAP_BYTES_GIB": "1",
        "ENGRAM_SHARD_GIB": "1",
        "HEADROOM_GIB": "1",
        # The stand-in shards are a few bytes, so the "is this a real shard"
        # floor is lowered for the test only.
        "ENGRAM_MIN_BYTES": "1",
        "WORKER_USER": "nvidia",
        "WORKER_IP": "192.168.177.11",
        "WORKER_HOME": "/home/nvidia",
        **(extra_env or {}),
    }
    proc = subprocess.run(
        ["bash", str(FETCH), *args],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
    )
    calls = []
    if log.is_file():
        calls = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    return proc, calls, cache


def _base_calls(calls):
    """Calls naming the base checkpoint. argv[0] is the subcommand."""
    return [c for c in calls if len(c) > 1 and BASE in c[1]]


def test_a_full_staging_run_succeeds_and_verifies() -> None:
    """The whole path: both repositories, then the verifier over the result."""
    with tempfile.TemporaryDirectory() as tmp:
        proc, calls, cache = _run(tmp)
        assert proc.returncode == 0, (proc.stdout, proc.stderr)
        # The launcher resolves $HF_HOME_HOST/hub/models--<org>--<name>/snapshots/<rev>.
        snap = (
            cache / "hub" / f"models--{REAP.replace('/', '--')}" / "snapshots" / "main"
        )
        assert snap.is_dir(), snap
        for shard in SHARDS:
            assert (snap / shard).is_file(), shard
        assert "checkpoint OK" in proc.stdout, proc.stdout
        # The two shards the REAP repo does not publish came from the base.
        base = _base_calls(calls)
        assert len(base) == 1, base
        assert "staged" in proc.stdout and ENGRAM_SHARDS[0] in proc.stdout, proc.stdout


def test_the_engram_shards_are_positional_not_include_patterns() -> None:
    """`--include a b` silently downloads only the second shard.

    The CLI takes one value for `--include`, so the second name becomes a
    positional filter and the pattern is dropped. Reproduced with
    huggingface-hub 1.29/1.30; the positional form is the one that works.
    """
    with tempfile.TemporaryDirectory() as tmp:
        proc, calls, _ = _run(tmp)
        assert proc.returncode == 0, (proc.stdout, proc.stderr)
        (base,) = _base_calls(calls)
        assert "--include" not in base, base
        for shard in ENGRAM_SHARDS:
            assert shard in base, base
        # Both must be positional: after the subcommand, the repo id and the
        # option values are stripped, exactly the two filenames remain.
        opts_with_values = {"--revision", "--cache-dir", "--local-dir"}
        i, positional = 2, []
        while i < len(base):
            if base[i] in opts_with_values:
                i += 2
                continue
            if base[i].startswith("-"):
                i += 1
                continue
            positional.append(base[i])
            i += 1
        assert positional == list(ENGRAM_SHARDS), positional


def test_each_download_uses_exactly_one_destination_flag() -> None:
    """`--cache-dir` and `--local-dir` are mutually exclusive, and each is needed.

    The cache flag is what makes a custom `HF_HOME_HOST` land where the launcher
    looks. The local-dir flag is what keeps a half-written Engram shard out of
    the snapshot. Passing both is a hard error from the CLI, which is how this
    was found on the node.
    """
    with tempfile.TemporaryDirectory() as tmp:
        proc, calls, cache = _run(tmp)
        assert proc.returncode == 0, (proc.stdout, proc.stderr)
        (reap,) = [c for c in calls if len(c) > 1 and REAP in c[1]]
        assert "--cache-dir" in reap, reap
        assert "--local-dir" not in reap, reap
        assert str(cache / "hub") in reap, reap
        (base,) = _base_calls(calls)
        assert "--local-dir" in base, base
        assert "--cache-dir" not in base, base
        # And the stub rejects the combination, so the assertions above bite.
        assert "STUB ERROR" not in proc.stderr, proc.stderr


def test_a_fresh_nested_cache_directory_does_not_abort_the_script() -> None:
    """`df` on a missing directory fails; the script must create it first."""
    with tempfile.TemporaryDirectory() as tmp:
        proc, _, cache = _run(tmp)
        assert proc.returncode == 0, (proc.stdout, proc.stderr)
        assert cache.is_dir(), cache
        # The failure mode was silent: no download attempt at all.
        assert "staging" in proc.stdout, proc.stdout


def test_the_disk_gate_refuses_with_a_named_reason() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        proc, calls, _ = _run(tmp, extra_env={"HEADROOM_GIB": "100000000"})
        assert proc.returncode != 0
        assert "GiB" in proc.stderr and "free" in proc.stderr, proc.stderr
        # Nothing should have been fetched once the gate refused.
        assert calls == [], calls


def test_a_rerun_reuses_the_staged_engram_shards() -> None:
    """Re-running on a complete checkpoint must not re-pull 189 GiB."""
    with tempfile.TemporaryDirectory() as tmp:
        first, calls1, cache = _run(tmp)
        assert first.returncode == 0, (first.stdout, first.stderr)
        assert len(_base_calls(calls1)) == 1
        # Second run, same cache: the Engram shards are in place, so the base
        # repository must not be contacted at all.
        bindir = Path(tmp) / "bin"
        log = Path(tmp) / "hf-argv-2.log"
        env = {
            **os.environ,
            "PATH": f"{bindir}:{os.environ.get('PATH', '')}",
            "HF_STUB_LOG": str(log),
            "HF_HOME_HOST": str(cache),
            "REAP_BYTES_GIB": "1",
            "ENGRAM_SHARD_GIB": "1",
            "HEADROOM_GIB": "1",
            "ENGRAM_MIN_BYTES": "1",
        }
        second = subprocess.run(
            ["bash", str(FETCH)], capture_output=True, text=True, cwd=ROOT, env=env
        )
        assert second.returncode == 0, (second.stdout, second.stderr)
        assert "already staged" in second.stdout, second.stdout
        calls2 = (
            [json.loads(x) for x in log.read_text().splitlines() if x.strip()]
            if log.is_file()
            else []
        )
        assert _base_calls(calls2) == [], calls2


def test_the_engram_estimate_is_not_added_to_the_index_total() -> None:
    """The index already lists the Engram shards; adding them again over-states
    the need by 189 GiB. The gate reports its own arithmetic, so the reported
    need is what is asserted."""
    with tempfile.TemporaryDirectory() as tmp:
        first, _, cache = _run(tmp)
        assert first.returncode == 0, (first.stdout, first.stderr)
        snap = (
            cache / "hub" / f"models--{REAP.replace('/', '--')}" / "snapshots" / "main"
        )
        # Remove the two Engram shards but keep the index, so the index-derived
        # remaining total is the whole requirement.
        for shard in ENGRAM_SHARDS:
            (snap / shard).unlink()
        bindir = Path(tmp) / "bin"
        env = {
            **os.environ,
            "PATH": f"{bindir}:{os.environ.get('PATH', '')}",
            "HF_STUB_LOG": str(Path(tmp) / "hf-argv-3.log"),
            "HF_HOME_HOST": str(cache),
            "REAP_BYTES_GIB": "1",
            "ENGRAM_SHARD_GIB": "1",
            "HEADROOM_GIB": "1",
            "ENGRAM_MIN_BYTES": "1",
        }
        second = subprocess.run(
            ["bash", str(FETCH)], capture_output=True, text=True, cwd=ROOT, env=env
        )
        assert second.returncode == 0, (second.stdout, second.stderr)
        # Index-derived remainder rounds to 0 GiB, so the need is the headroom
        # alone. Double-counting would make it 1 + 2 + 1.
        assert "need 1 GiB" in second.stdout, second.stdout
        assert "index-derived" in second.stdout, second.stdout


def test_the_base_checkpoint_has_its_own_revision() -> None:
    """A revision pinned for REAP must not be requested from the base repo."""
    with tempfile.TemporaryDirectory() as tmp:
        proc, calls, _ = _run(
            tmp, extra_env={"MODEL_REVISION": "deadbeefcafe", "BASE_MODEL_REVISION": "main"}
        )
        assert proc.returncode == 0, (proc.stdout, proc.stderr)
        (base,) = _base_calls(calls)
        i = base.index("--revision")
        assert base[i + 1] == "main", base
        reap = [c for c in calls if len(c) > 1 and REAP in c[1]]
        assert reap, calls
        j = reap[0].index("--revision")
        assert reap[0][j + 1] == "deadbeefcafe", reap[0]


def test_the_printed_copy_command_targets_the_worker_account_and_path() -> None:
    """Both ranks mount the head's model cache directory, so the copy must land
    as the worker's own account at that same path — and scoped to that one
    directory, not the whole cache root, which would ship every other model."""
    with tempfile.TemporaryDirectory() as tmp:
        proc, _, cache = _run(tmp)
        assert proc.returncode == 0, (proc.stdout, proc.stderr)
        line = [ln for ln in proc.stdout.splitlines() if "rsync" in ln]
        assert line, proc.stdout
        src = str(cache / "hub" / f"models--{REAP.replace('/', '--')}")
        dst = (
            "nvidia@192.168.177.11:/home/nvidia/.cache/huggingface/hub/"
            f"models--{REAP.replace('/', '--')}/"
        )
        assert f"{src}/ {dst}" in line[0], line[0]
        assert "reederey" not in line[0], line[0]
        # The whole cache root must not be the source: that would copy every
        # unrelated model on the node as well.
        assert f"{cache}/ nvidia@" not in line[0], line[0]


def test_every_script_a_code_path_names_exists() -> None:
    """A named remedy has to be a file."""
    files = [START]
    for pattern in CODE_GLOBS:
        files.extend(sorted(ROOT.glob(pattern)))
    missing: list[str] = []
    for path in files:
        for name in SCRIPT_REF_RE.findall(path.read_text()):
            if not (ROOT / "scripts" / name).is_file():
                missing.append(f"{path.relative_to(ROOT)} names scripts/{name}")
    assert missing == [], "named but absent:\n  " + "\n  ".join(sorted(set(missing)))


def test_the_verifier_names_the_staging_script_that_now_exists() -> None:
    text = VERIFY.read_text()
    assert "scripts/fetch_weights.sh" in text, "the verifier no longer names the remedy"
    assert FETCH.is_file(), f"{FETCH} is named by the verifier but does not exist"


def test_the_staging_script_targets_the_directory_the_launcher_resolves() -> None:
    """Both sides must compute the same snapshot path from the same variables."""
    fetch = FETCH.read_text()
    start = START.read_text()
    assert 'MODEL_CACHE="${HF_HOME_HOST}/hub/models--${MODEL//\\//--}"' in start, start
    assert 'MODEL_DIR_HOST="${MODEL_CACHE}/snapshots/$(tr -d' in start, start
    assert 'MODEL_CACHE="$HF_HOME_HOST/hub/models--${MODEL//\\//--}"' in fetch, fetch
    assert 'echo "$cache/snapshots/$(tr -d' in fetch, fetch
    for name in ("HF_HOME_HOST", "MODEL", "MODEL_REVISION"):
        assert re.search(rf'^{name}="\$\{{{name}:-', fetch, re.MULTILINE), name
        assert re.search(rf'^{name}="\$\{{{name}:-', start, re.MULTILINE), name
    assert 'die "weights missing at $MODEL_DIR_HOST (no download from this script)"' in start


def test_the_staging_script_does_not_leave_a_half_written_shard_in_place() -> None:
    """A partial shard must never appear under the name the loader reads."""
    fetch = FETCH.read_text()
    assert 'stage="$HF_HOME_HOST/hub/.dsv41-engram-stage"' in fetch, fetch
    assert 'mv -f "$src" "$dst"' in fetch, fetch


def test_the_staging_script_creates_the_cache_root_before_measuring_it() -> None:
    """`mkdir -p` must precede the first `df`, or a fresh cache aborts."""
    fetch = FETCH.read_text()
    mkdir_at = fetch.index('mkdir -p "$CACHE_DIR"')
    df_at = fetch.index("free_gib() {")
    call_at = fetch.index('avail="$(free_gib')
    assert mkdir_at < call_at, "the cache root must be created before it is measured"
    assert df_at < mkdir_at, "free_gib must be defined before it is called"


def test_the_staging_script_is_executable() -> None:
    assert FETCH.stat().st_mode & 0o111, f"{FETCH} is not executable"


if __name__ == "__main__":
    test_a_full_staging_run_succeeds_and_verifies()
    test_the_engram_shards_are_positional_not_include_patterns()
    test_each_download_uses_exactly_one_destination_flag()
    test_a_fresh_nested_cache_directory_does_not_abort_the_script()
    test_the_disk_gate_refuses_with_a_named_reason()
    test_a_rerun_reuses_the_staged_engram_shards()
    test_the_engram_estimate_is_not_added_to_the_index_total()
    test_the_base_checkpoint_has_its_own_revision()
    test_the_printed_copy_command_targets_the_worker_account_and_path()
    test_every_script_a_code_path_names_exists()
    test_the_verifier_names_the_staging_script_that_now_exists()
    test_the_staging_script_targets_the_directory_the_launcher_resolves()
    test_the_staging_script_does_not_leave_a_half_written_shard_in_place()
    test_the_staging_script_creates_the_cache_root_before_measuring_it()
    test_the_staging_script_is_executable()
    print("weight staging tests OK")
