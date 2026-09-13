#!/usr/bin/env bash
# ab_spec.sh — A/B arms for the DSpark speculative-decode wave.
#
# One arm per boot. An arm is a set of env overrides on top of `.env`; the script
# stops the cluster, starts it with those overrides, gates the boot on a health
# wait, runs the NaN/coherence smoke, takes a decode row, and appends a receipt.
#
# Why one boot per arm rather than an in-place change: the speculative config is
# read at engine init, and the weights are re-read anyway, so there is no cheaper
# way to switch arms. Why a receipt per arm: this checkpoint's known failure mode
# is fluent NaN carrying a healthy throughput number, so a tok/s figure without
# its smoke result is not evidence.
#
#   ./scripts/ab_spec.sh list
#   ./scripts/ab_spec.sh run <arm> [more arms...]
#   ./scripts/ab_spec.sh report
#
# Arms are declared in ARMS below; `baseline` is the serving configuration, so
# every other arm is a delta against it.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
RECEIPTS="${RECEIPTS:-$SCRIPT_DIR/results/ab-spec}"
PHASE_TOKENS="${PHASE_TOKENS:-200}"
PHASE_RUNS="${PHASE_RUNS:-3}"
# The interlock refuses to start while another GPU container is up; give the
# unified pool a moment to come back before the next boot claims it. Overridable
# so the tests can exercise the arm lifecycle without waiting.
AB_SETTLE_SECONDS="${AB_SETTLE_SECONDS:-15}"
# Read from the operator's `.env` when it is there, so the harness drives the same
# port and model the cluster is actually serving rather than a copy of a default.
if [ -f "$SCRIPT_DIR/.env" ]; then
  PORT="$(sed -n 's/^PORT=//p' "$SCRIPT_DIR/.env" | tail -1)"
  SERVED_MODEL_NAME="$(sed -n 's/^SERVED_MODEL_NAME=//p' "$SCRIPT_DIR/.env" | tail -1)"
fi
PORT="${PORT:-8001}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-deepseek-v4.1-flash-reap256}"

log() { printf '\033[1;36m[ab]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[ab]\033[0m ERROR: %s\n' "$*" >&2; exit 1; }

# name|SPEC_METHOD|SPEC_NUM_TOKENS|extra env...
#
# `baseline` is the deployed configuration (SPEC=none, the measured 13.80 tok/s
# control) so every speculative row has a same-day control. `greedy5` is the
# adopted winner. `greedy10` tests whether the acceptance curve keeps climbing
# past the first block — it does not. `probabilistic5` is the higher-acceptance
# option that upstream recorded degenerating silently; it is here to be judged by
# the coherence check, not by tok/s.
#
# There is no `adaptive5` arm: `enable_adaptive_verification` needs full CUDA
# graphs, which this checkpoint cannot use, so `spec_flags` refuses it. An arm
# that cannot boot is not an arm.
ARMS=(
  "baseline|none||"
  "greedy5|dspark|5|SPEC_DRAFT_SAMPLE_METHOD=greedy"
  "greedy10|dspark|10|SPEC_DRAFT_SAMPLE_METHOD=greedy"
  "probabilistic5|dspark|5|SPEC_DRAFT_SAMPLE_METHOD=probabilistic"
)

arm_field() {  # arm_field <arm> <field-index>
  local want="$1" i="$2" spec
  for spec in "${ARMS[@]}"; do
    [ "${spec%%|*}" = "$want" ] || continue
    printf '%s' "$spec" | cut -d'|' -f"$((i + 1))"
    return 0
  done
  return 1
}

list_arms() {
  printf '%-18s %-8s %-6s %s\n' ARM SPEC TOKENS OVERRIDES
  local spec
  for spec in "${ARMS[@]}"; do
    IFS='|' read -r name method tokens extra <<<"$spec"
    printf '%-18s %-8s %-6s %s\n' "$name" "${method:-none}" "${tokens:--}" "$extra"
  done
}

health_ok() {
  python3 - "$PORT" <<'PY'
import sys, urllib.request as u
try:
    print(u.urlopen(f"http://127.0.0.1:{sys.argv[1]}/health", timeout=30).status)
except Exception as e:
    print("ERR", e)
PY
}

run_arm() {
  local arm="$1"
  local method tokens extra
  method="$(arm_field "$arm" 1)" || die "unknown arm '$arm' (see: $0 list)"
  tokens="$(arm_field "$arm" 2)"
  extra="$(arm_field "$arm" 3)"
  method="${method:-none}"

  mkdir -p "$RECEIPTS"
  local out="$RECEIPTS/$arm.json"
  log "arm=$arm method=$method tokens=${tokens:--} extra='$extra'"

  log "stopping the cluster"
  ( cd "$SCRIPT_DIR" && ./start.sh stop ) >/dev/null 2>&1 || true
  sleep "$AB_SETTLE_SECONDS"

  # An arm cannot be passed through the environment: `start.sh` sources `.env`
  # with `set -a` *after* the environment exists, so a value in `.env` wins over
  # an exported one. So build an arm `.env` and install it with an atomic rename.
  # This runs while no `start.sh` is executing, which is what makes it safe.
  #
  # The arm file must START as a copy of the operator's `.env` and then have the
  # overrides appended — later assignments win in a sourced file. Appending to a
  # missing file would create one containing *only* the arm lines, and the boot
  # would then silently run on `start.sh`'s defaults instead of the operator's
  # configuration (a custom PORT, IMAGE or MIN_FREE_GIB would all be dropped),
  # which would invalidate the comparison the arm exists to make.
  #
  # The snapshot is per-invocation, not a fixed name: reusing one backup file
  # means a second run restores the *first* run's copy and destroys any edit the
  # operator made in between.
  local snap="$SCRIPT_DIR/.env.ab-snapshot.$$"
  cp "$SCRIPT_DIR/.env" "$snap" || die "cannot snapshot $SCRIPT_DIR/.env"
  local armfile="$SCRIPT_DIR/.env.arm.$$"
  cp "$snap" "$armfile" || die "cannot seed the arm env from $SCRIPT_DIR/.env"
  {
    printf '\n# --- arm %s (written by scripts/ab_spec.sh) ---\n' "$arm"
    # `FIRST_BOOT=1` refuses speculative decode on purpose: the first boot of a
    # new kit is meant to be the conservative shape. An A/B arm is the deliberate
    # ramp, so every arm — including the control — boots with FIRST_BOOT=0. That
    # only *permits* the four first-boot switches; the argv still honors whatever
    # the rest of `.env` says, so the serving shape is unchanged and the sole
    # delta between arms is the speculative config.
    printf 'FIRST_BOOT=0\n'
    printf 'SPEC_METHOD=%s\n' "$method"
    [ -n "$tokens" ] && printf 'SPEC_NUM_TOKENS=%s\n' "$tokens"
    [ -n "$extra" ] && printf '%s\n' "$extra" | tr ' ' '\n'
    # `baseline` has neither tokens nor extra, so both `[ -n ]` tests short-circuit
    # false and would make the group return 1 — which the `|| die` below would
    # read as a write failure. The explicit success is what makes the guard
    # meaningful instead of firing on the control arm.
    :
  } >> "$armfile" || die "cannot write the arm env"
  mv "$armfile" "$SCRIPT_DIR/.env" || die "cannot install the arm env"

  # Restore the operator's configuration on *any* exit — a failed start, a
  # signal, or an interrupt — because leaving an arm installed would make the
  # next boot silently speculative.
  restore_env() {
    if [ -f "$snap" ]; then
      cp "$snap" "$SCRIPT_DIR/.env" && rm -f "$snap" "$armfile" 2>/dev/null
    fi
  }
  trap restore_env EXIT INT TERM HUP

  log "starting the cluster"
  local rc=0
  ( cd "$SCRIPT_DIR" && ./start.sh start ) || rc=$?
  # Restore explicitly and clear the trap: every later step reads the port and
  # the model name from the running server, not from the file, so leaving an arm
  # in place would only mislead the next boot.
  restore_env
  trap - EXIT INT TERM HUP

  if [ "$rc" -ne 0 ]; then
    log "arm=$arm START FAILED rc=$rc — recording and continuing"
    python3 - "$out" "$arm" "$method" "$tokens" "$extra" <<'PY'
import json, sys, time
out, arm, method, tokens, extra = sys.argv[1:6]
json.dump({"arm": arm, "spec_method": method, "spec_tokens": tokens,
           "overrides": extra, "start_rc": 1, "ts": time.time(),
           "verdict": "start-failed"}, open(out, "w"), indent=2)
PY
    return 1
  fi

  log "health"
  local h
  h="$(health_ok)"
  log "health=$h"

  log "smoke (logprobs must be finite)"
  local smoke_rc=0
  smoke_out="$( cd "$SCRIPT_DIR" && ./start.sh smoke 2>&1 )" || smoke_rc=$?
  printf '%s\n' "$smoke_out" | tail -5

  log "decode row (phase=$arm runs=$PHASE_RUNS)"
  local bench_rc=0 bench_out=""
  bench_out="$( cd "$SCRIPT_DIR" && \
      DSV41_BASE="http://127.0.0.1:$PORT" DSV41_MODEL="$SERVED_MODEL_NAME" \
      python3 bench/bench_decode.py --phase "$arm" --runs "$PHASE_RUNS" \
      --max-tokens "$PHASE_TOKENS" --out "$out" 2>&1 )" || bench_rc=$?
  printf '%s\n' "$bench_out" | tail -8

  python3 - "$out" "$arm" "$method" "$tokens" "$extra" "$smoke_rc" "$bench_rc" "$h" <<'PY'
import json, os, sys, time
out, arm, method, tokens, extra, smoke_rc, bench_rc, health = sys.argv[1:9]
# `bench_decode.py` already wrote the full row — per-run results, usage, the
# engine's spec counters and the head of each generation. Annotate that file
# rather than replacing it: the per-run detail and the generated text are the
# evidence that a median alone cannot supply, and `bench/README.md` promises
# they are in the receipt.
try:
    rec = json.load(open(out))
    if not isinstance(rec, dict):
        rec = {"bench_payload": rec}
except Exception as e:
    rec = {"bench_payload_error": str(e)}
rec.update({"arm": arm, "spec_method": method, "spec_tokens": tokens,
            "overrides": extra, "start_rc": 0, "smoke_rc": int(smoke_rc),
            "bench_rc": int(bench_rc), "health": health, "ts": time.time()})
# The bench decides `ok` for itself, and it is the only thing that knows whether
# a row actually completed with usage, logprobs and a finish reason. Trust its
# verdict rather than re-deriving one from `coherent` alone: a stream that dies
# after coherent text, or that never carried logprobs, is not a passing arm.
rec["verdict"] = ("ok" if rec.get("smoke_rc") == 0 and rec.get("bench_rc") == 0
                  and rec.get("ok") is True else "check")
json.dump(rec, open(out, "w"), indent=2)
print(f"[ab] {arm}: {rec['verdict']}")
PY
  return 0
}

report() {
  [ -d "$RECEIPTS" ] || die "no receipts at $RECEIPTS"
  python3 - "$RECEIPTS" <<'PY'
import glob, json, os, sys
rows = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.json"))):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    if not isinstance(d, dict):
        continue
    # Current receipts carry the bench row at the top level, annotated with the
    # arm metadata. Early receipts nested it under "decode"; accept both so an
    # older file still reports rather than silently dropping out of the table.
    dec = d.get("decode") if isinstance(d.get("decode"), dict) else d
    rows.append((d.get("arm") or os.path.basename(f)[:-5],
                 d.get("spec_method", "?"), d.get("spec_tokens", "-"),
                 d.get("verdict", "?"), dec.get("tok_s_median"),
                 dec.get("ttft_median_s"), dec.get("coherent"),
                 dec.get("any_nan"), dec.get("accept_ratio_median")))
if not rows:
    print("no receipts")
    raise SystemExit(1)
hdr = ("arm", "spec", "tok", "verdict", "tok/s", "ttft_s", "coherent", "nan", "accept")
print(f"{hdr[0]:<18}{hdr[1]:<8}{hdr[2]:<5}{hdr[3]:<8}{hdr[4]:<9}{hdr[5]:<9}{hdr[6]:<10}{hdr[7]:<6}{hdr[8]}")
for r in rows:
    fmt = lambda v: "-" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))
    print(f"{r[0]:<18}{r[1]:<8}{str(r[2]):<5}{r[3]:<8}{fmt(r[4]):<9}{fmt(r[5]):<9}"
          f"{str(r[6]):<10}{str(r[7]):<6}{fmt(r[8])}")
PY
}

case "${1:-list}" in
  list) list_arms ;;
  run)
    shift
    [ "$#" -ge 1 ] || die "usage: $0 run <arm> [more arms...]"
    rc=0
    for a in "$@"; do run_arm "$a" || rc=1; done
    exit "$rc"
    ;;
  report) report ;;
  *) die "usage: $0 list|run <arm>|report" ;;
esac
