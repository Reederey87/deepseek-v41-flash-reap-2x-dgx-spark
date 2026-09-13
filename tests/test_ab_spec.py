#!/usr/bin/env python3
"""Tests for the DSpark A/B harness and its decode bench.

Two things here can silently produce a wrong verdict, so both are pinned:

1. the bench's coherence gate — if it cannot see a NaN row or a looping draft it
   will report a healthy tok/s for a broken arm, which is the exact failure mode
   this wave exists to catch;
2. the arm table — an arm that quietly resolves to `SPEC=none`, or to a spec
   depth the engine rejects, would make every speculative row a duplicate of the
   baseline.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "bench" / "bench_decode.py"
AB = ROOT / "scripts" / "ab_spec.sh"

fails = 0


def check(ok: bool, label: str, detail: str = "") -> None:
    global fails
    if ok:
        print(f"ok   {label}")
    else:
        print(f"FAIL {label} {detail}")
        fails += 1


def load_bench():
    spec = importlib.util.spec_from_file_location("bench_decode", BENCH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


# --- coherence gate -----------------------------------------------------------
bench = load_bench()

ok, why = bench.coherence("A hash map stores key-value pairs and resolves collisions.")
check(ok, "coherent prose passes", why)

ok, why = bench.coherence("")
check(not ok, "empty output fails", why)

ok, why = bench.coherence("the value is nan and the answer is nan")
check(not ok, "a NaN marker fails", why)

# A collapsed drafter repeats itself; prose does not. The window check must fire
# well before a human would call it "a bit repetitive".
looped = "and then the next token follows " * 40
ok, why = bench.coherence(looped)
check(not ok, "a looping draft fails", why)

# A legitimate repetition inside an otherwise normal answer must NOT trip it: a
# word-level check would have flagged this.
ok, why = bench.coherence(
    "Repeat after me: the quick brown fox jumps over the lazy dog. " * 2
    + "Now explain why that sentence is used for typing practice, in detail."
)
check(ok, "ordinary repetition passes", why)

# --- the bench end to end against a stub server -------------------------------
class Stub(BaseHTTPRequestHandler):
    mode = "good"
    # The bench reads acceptance as a delta across the measured window, so the
    # stub has to move its counters between the before and after scrape — a
    # static value yields a zero delta and correctly reports no acceptance.
    metrics_calls = 0

    def log_message(self, *a):  # noqa: D102
        pass

    def do_GET(self):  # noqa: N802
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
            return
        if self.path == "/metrics":
            Stub.metrics_calls += 1
            first = Stub.metrics_calls == 1
            # Upstream's own worked example (vllm PR #16665): 5 drafts, 15 draft
            # tokens, 9 accepted, with a per-position vector of [5, 3, 1]. The
            # before scrape is all zeros, so the delta *is* the example. The scale
            # is 1 for that reason — scaling it would scale the deltas too.
            scale = 0.0 if first else 1.0
            body = (
                "# TYPE vllm:spec_decode_num_accepted_tokens_total counter\n"
                f"vllm:spec_decode_num_accepted_tokens_total {9.0 * scale}\n"
                f"vllm:spec_decode_num_draft_tokens_total {15.0 * scale}\n"
                f"vllm:spec_decode_num_drafts_total {5.0 * scale}\n"
                "# TYPE vllm:spec_decode_num_accepted_tokens_per_pos_total counter\n"
                f'vllm:spec_decode_num_accepted_tokens_per_pos_total{{position="0"}} {5.0 * scale}\n'
                f'vllm:spec_decode_num_accepted_tokens_per_pos_total{{position="1"}} {3.0 * scale}\n'
                f'vllm:spec_decode_num_accepted_tokens_per_pos_total{{position="2"}} {1.0 * scale}\n'
            ).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        if self.mode == "truncated":
            # Coherent text, then an SSE error, then nothing: no usage, no
            # logprobs on the error path, no [DONE]. This is the shape that used
            # to be averaged in as a fast run.
            for w in ["A hash map ", "stores pairs "]:
                obj = {
                    "choices": [
                        {
                            "delta": {"content": w},
                            "logprobs": {"content": [{"logprob": -0.5, "token": w}]},
                            "finish_reason": None,
                        }
                    ]
                }
                self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
                self.wfile.flush()
            self.wfile.write(
                f"data: {json.dumps({'error': {'message': 'engine died'}})}\n\n".encode()
            )
            self.wfile.flush()
            return
        if self.mode == "good":
            words = ["A hash map ", "stores pairs ", "and resolves ", "collisions."]
            logprob = -0.5
        else:
            words = ["nan "] * 4
            logprob = float("nan")
        for i, w in enumerate(words):
            obj = {
                "choices": [
                    {
                        "delta": {"content": w},
                        "logprobs": {"content": [{"logprob": logprob, "token": w}]},
                        "finish_reason": None,
                    }
                ]
            }
            self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
            self.wfile.flush()
        obj = {
            "choices": [{"delta": {}, "finish_reason": "length"}],
            "usage": {"completion_tokens": len(words), "prompt_tokens": 8},
        }
        self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def run_bench(mode: str, tmp: Path) -> tuple[int, dict]:
    Stub.mode = mode
    Stub.metrics_calls = 0
    srv = HTTPServer(("127.0.0.1", 0), Stub)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    out = tmp / f"{mode}.json"
    env = {
        "DSV41_BASE": f"http://127.0.0.1:{port}",
        "DSV41_MODEL": "stub",
        "PATH": "/usr/bin:/bin",
    }
    p = subprocess.run(
        [sys.executable, str(BENCH), "--phase", mode, "--out", str(out), "--runs", "1"],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    srv.shutdown()
    return p.returncode, json.loads(out.read_text()) if out.exists() else {}


import tempfile  # noqa: E402

with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    rc, rec = run_bench("good", tmp)
    check(rc == 0, "bench exits 0 on a clean run", f"rc={rc}")
    check(rec.get("coherent") is True, "receipt marks a clean run coherent")
    check(rec.get("any_nan") is False, "receipt marks a clean run NaN-free")
    check(rec.get("tok_s_median") is not None, "receipt carries a tok/s")
    check(
        rec.get("accept_ratio_median") == 0.6,
        "acceptance is read from the engine counters",
        str(rec.get("accept_ratio_median")),
    )
    # 1 + accepted/drafts = 1 + 9/5 = 2.8, not 1 + accept_ratio = 1.6. Quoting the
    # wrong one understates the win, and it is the number a reader compares across
    # arms, so pin it against upstream's definition.
    check(
        rec.get("mean_acceptance_length") == 2.8,
        "mean acceptance length is 1 + accepted/drafts",
        str(rec.get("mean_acceptance_length")),
    )
    # The per-position vector is accepted_per_pos/drafts and is cumulative: 5/5,
    # 3/5, 1/5. If the label is dropped the three series collapse onto one key and
    # only the last survives, which is the bug this pins.
    check(
        rec.get("acceptance", {}).get("per_position") == [1.0, 0.6, 0.2],
        "per-position acceptance keeps all three series",
        str(rec.get("acceptance", {}).get("per_position")),
    )
    check(
        rec.get("acceptance", {}).get("drafts") == 5.0,
        "the window delta carries the draft count",
        str(rec.get("acceptance", {}).get("drafts")),
    )
    # The raw scrape must keep the position label, or the vector above is built
    # from a single overwritten value.
    check(
        len([k for k in rec.get("spec_metrics_after", {}) if "position=" in k]) == 3,
        "the raw scrape keeps per-position series distinct",
        str(sorted(rec.get("spec_metrics_after", {}))),
    )

    rc, rec = run_bench("bad", tmp)
    check(rc == 1, "bench exits 1 on a NaN run", f"rc={rc}")
    check(rec.get("any_nan") is True, "receipt marks a NaN run")
    check(rec.get("coherent") is False, "receipt marks a NaN run incoherent")

    # A coherent-looking prefix followed by an SSE error must not read as a pass.
    # Without this, a stream that dies mid-generation is averaged in as a row.
    rc, rec = run_bench("truncated", tmp)
    check(rc == 1, "bench exits 1 on a truncated stream", f"rc={rc}")
    check(rec.get("ok") is False, "a truncated stream is not ok")
    check(
        rec.get("tok_s_median") is None,
        "a truncated stream yields no measurement",
        str(rec.get("tok_s_median")),
    )
    problems = " ".join(rec["runs"][0].get("problems") or [])
    check("sse error" in problems, "the SSE error is recorded", problems)
    check("without [DONE]" in problems, "the missing [DONE] is recorded", problems)
    check("no usage" in problems, "the missing usage is recorded", problems)

    # A clean run must be marked ok, or the verdict logic has no positive case.
    rc, rec = run_bench("good", tmp)
    check(rec.get("ok") is True, "a clean run is marked ok")
    check(
        rec.get("valid_runs") == 1 and rec.get("invalid_runs") == 0,
        "a clean run counts one valid row",
        f"{rec.get('valid_runs')}/{rec.get('invalid_runs')}",
    )

# --- the arm table ------------------------------------------------------------
p = subprocess.run(["bash", str(AB), "list"], capture_output=True, text=True)
check(p.returncode == 0, "ab_spec.sh list runs", p.stderr[:200])
lines = [ln for ln in p.stdout.splitlines()[1:] if ln.strip()]
check(len(lines) >= 4, "arm table has the expected arms", str(len(lines)))
names = [ln.split()[0] for ln in lines]
check("baseline" in names, "baseline arm present", str(names))
check("greedy5" in names, "greedy5 arm present", str(names))
check("probabilistic5" in names, "probabilistic arm present", str(names))
# An arm that cannot boot is not an arm: adaptive verification needs full CUDA
# graphs, which this checkpoint cannot use, so `spec_flags` refuses it.
check("adaptive5" not in names, "no arm requests adaptive verification", str(names))

# The documented invocation is `./scripts/ab_spec.sh`, so the file must be
# executable in the published tree.
check(os.access(AB, os.X_OK), "ab_spec.sh is executable")

# The baseline must be SPEC=none, or every speculative row has no control.
baseline = next(ln for ln in lines if ln.startswith("baseline"))
check("none" in baseline, "baseline is SPEC=none", baseline)

# Every speculative arm must name a spec method, or the row duplicates baseline.
for ln in lines:
    if ln.startswith("baseline"):
        continue
    check("dspark" in ln, f"arm names a spec method: {ln.split()[0]}", ln)

# The spec depths in the table must be ones the engine accepts for a block of 5:
# at most 5, or a multiple of it.
for ln in lines:
    parts = ln.split()
    if parts[1] != "dspark":
        continue
    n = int(parts[2])
    check(
        n <= 5 or n % 5 == 0,
        f"spec depth {n} is on the DSpark block grid",
        ln,
    )

p = subprocess.run(["bash", str(AB), "bogus"], capture_output=True, text=True)
check(p.returncode != 0, "an unknown command fails closed")

# --- the arm lifecycle, against a stand-in launcher ---------------------------
# Two things in `run_arm` can silently invalidate a comparison, and neither is
# visible from the arm table: the arm `.env` must start as a copy of the
# operator's configuration (appending to a missing file would boot on start.sh's
# defaults and drop a custom PORT/IMAGE/MIN_FREE_GIB), and the restore must be
# per-invocation (a fixed backup name would let a second run restore the *first*
# run's copy and destroy an edit made in between). Both are exercised here with a
# stub launcher, because the real one needs docker and a GPU.
KIT_ENV = (
    "# operator configuration\n"
    "PORT=8123\n"
    "IMAGE=custom-tested-image\n"
    "MIN_FREE_GIB=99\n"
    "SERVED_MODEL_NAME=custom-model\n"
    "SPEC_METHOD=none\n"
)
STUB_START = """#!/usr/bin/env bash
# Records the .env it was handed, so the test can see what the arm booted with.
set -u
case "${1:-}" in
  start) cp .env "$RECORD/env.start" ;;
  stop|smoke) ;;
esac
exit 0
"""
STUB_BENCH = """#!/usr/bin/env python3
import json, sys
out = sys.argv[sys.argv.index("--out") + 1]
json.dump({"ok": True, "coherent": True, "any_nan": False,
           "tok_s_median": 1.0, "ttft_median_s": 1.0}, open(out, "w"))
print("stub bench")
"""


def run_arm_in_stub_kit(tmp: Path, arm: str) -> tuple[int, str]:
    env = dict(os.environ)
    env["AB_SETTLE_SECONDS"] = "0"
    env["RECORD"] = str(tmp / "record")
    p = subprocess.run(
        ["bash", str(tmp / "kit" / "scripts" / "ab_spec.sh"), "run", arm],
        capture_output=True, text=True, env=env, timeout=120, cwd=str(tmp / "kit"),
    )
    return p.returncode, p.stdout + p.stderr


with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    kit = tmp / "kit"
    (kit / "scripts").mkdir(parents=True)
    (kit / "bench").mkdir(parents=True)
    (tmp / "record").mkdir()
    shutil.copy(AB, kit / "scripts" / "ab_spec.sh")
    (kit / ".env").write_text(KIT_ENV)
    (kit / "start.sh").write_text(STUB_START)
    (kit / "start.sh").chmod(0o755)
    (kit / "bench" / "bench_decode.py").write_text(STUB_BENCH)

    rc, log = run_arm_in_stub_kit(tmp, "greedy5")
    check(rc == 0, "the arm completes against a stub launcher", log[-300:])

    seen = (tmp / "record" / "env.start")
    booted = seen.read_text() if seen.exists() else ""
    for key, want in (("PORT", "8123"), ("IMAGE", "custom-tested-image"),
                      ("MIN_FREE_GIB", "99"),
                      ("SERVED_MODEL_NAME", "custom-model")):
        check(f"{key}={want}" in booted,
              f"the arm boots with the operator's {key}", booted[:300])
    check("SPEC_METHOD=dspark" in booted, "the arm override is applied")
    check("FIRST_BOOT=0" in booted, "the arm lifts the first-boot shape")
    # The arm override must win over the operator's value, not lose to it.
    check(booted.rindex("SPEC_METHOD=dspark") > booted.index("SPEC_METHOD=none"),
          "the override is appended after the operator's value")

    check((kit / ".env").read_text() == KIT_ENV,
          "the operator's .env is restored verbatim",
          (kit / ".env").read_text()[:200])
    leftovers = sorted(f.name for f in kit.iterdir() if f.name.startswith(".env."))
    check(not leftovers, "no arm or snapshot file is left behind", str(leftovers))

    # An edit made between two runs must survive the second run: a fixed backup
    # name would restore the first run's copy and silently revert it.
    edited = KIT_ENV.replace("MIN_FREE_GIB=99", "MIN_FREE_GIB=77")
    (kit / ".env").write_text(edited)
    rc, log = run_arm_in_stub_kit(tmp, "baseline")
    check(rc == 0, "a second arm run completes", log[-300:])
    check((kit / ".env").read_text() == edited,
          "an operator edit between runs survives",
          (kit / ".env").read_text()[:200])
    booted2 = (tmp / "record" / "env.start").read_text()
    check("MIN_FREE_GIB=77" in booted2,
          "the second arm boots with the edited value", booted2[:300])

print()
if fails:
    print(f"ab-spec tests FAILED ({fails})")
    sys.exit(1)
print("ab-spec tests OK")
