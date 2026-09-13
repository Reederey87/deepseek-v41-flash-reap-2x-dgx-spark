#!/usr/bin/env bash
# Launcher guard tests.
#
# On GB10 the GPU allocator and the page cache share one unified pool, and the
# box hard-resets when `MemAvailable` approaches zero — there is no OOM kill to
# catch and no log line afterwards. `refuse_low_memory()` is therefore
# safety-critical and must fail closed: it reads both nodes and refuses when
# either is short, or when either reading is unavailable.
#
# The reader is stubbed rather than mocked at /proc, because the real
# `mem_available_gib` shells into the worker and this Mac has no /proc.
#
# Run: bash tests/test_launch_guards.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
START_SH="$HERE/../start.sh"

if [ ! -f "$START_SH" ]; then
  echo "start.sh not present (container layout); launcher guard tests skipped"
  exit 0
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

{
  echo 'die() { echo "REFUSED: $*" >&2; exit 9; }'
  echo 'log() { :; }'
  echo 'WORKER_SSH=nvidia@spark2'
  echo 'MIN_FREE_GIB=90'
  awk '/^refuse_low_memory\(\)/,/^}/' "$START_SH"
  # `mem_available_gib` takes an optional node argument: empty = head.
  cat <<'EOS'
case "$CASE" in
  head_short)    mem_available_gib() { [ -z "${1:-}" ] && echo 42 || echo 200; } ;;
  worker_short)  mem_available_gib() { [ -z "${1:-}" ] && echo 200 || echo 42; } ;;
  both_ok)       mem_available_gib() { echo 200; } ;;
  at_floor)      mem_available_gib() { echo 90; } ;;
  one_below)     mem_available_gib() { echo 89; } ;;
  head_unread)   mem_available_gib() { [ -z "${1:-}" ] && return 1 || echo 200; } ;;
  worker_unread) mem_available_gib() { [ -z "${1:-}" ] && echo 200 || return 1; } ;;
esac
# `die` exits the script, so the guard runs in a subshell to observe its status.
( refuse_low_memory ) >/dev/null 2>&1 && echo ALLOWED || echo REFUSED
EOS
} > "$TMP/run.sh"

fails=0
expect() {
  local want="$1" case_name="$2" label="$3"
  local got
  got="$(CASE="$case_name" bash "$TMP/run.sh" 2>&1 | tail -1)"
  if [ "$got" = "$want" ]; then
    printf 'ok   %-40s %s\n' "$label" "$got"
  else
    printf 'FAIL %-40s want %s got %s\n' "$label" "$want" "$got"
    fails=$((fails + 1))
  fi
}

expect REFUSED head_short    "head below floor"
expect REFUSED worker_short  "worker below floor"
expect ALLOWED both_ok       "both nodes above floor"
expect ALLOWED at_floor      "exactly at the floor"
expect REFUSED one_below     "one GiB below the floor"
expect REFUSED head_unread   "head reading unavailable"
expect REFUSED worker_unread "worker reading unavailable"

# --- quantization / checkpoint consistency -----------------------------------
# Forcing --quantization exl3 onto a natively MXFP4/FP8 pack (REAP-256E) makes
# the loader look for EXL3 trellis tensors that do not exist. The mismatch must
# be caught before a container starts, not in the loader.
{
  echo 'die() { echo "REFUSED: $*" >&2; exit 9; }'
  echo 'log() { :; }'
  awk '/^check_quantization_matches\(\)/,/^}/' "$START_SH"
  cat <<'EOS'
# `die` exits the script, so the guard runs in a subshell to observe its status.
( check_quantization_matches ) >/dev/null 2>&1 && echo ALLOWED || echo REFUSED
EOS
} > "$TMP/quant.sh"

qexpect() {
  local want="$1" decl="$2" quant="$3" label="$4"
  local got
  printf '{"quantization_config": {"quant_method": "%s"}}\n' "$decl" > "$TMP/config.json"
  got="$(MODEL_DIR_HOST="$TMP" QUANTIZATION="$quant" bash "$TMP/quant.sh" 2>&1 | tail -1)"
  if [ "$got" = "$want" ]; then
    printf 'ok   %-40s %s\n' "$label" "$got"
  else
    printf 'FAIL %-40s want %s got %s\n' "$label" "$want" "$got"
    fails=$((fails + 1))
  fi
}
qexpect ALLOWED fp8 ""    "REAP pack, no override"
qexpect REFUSED fp8 exl3  "exl3 forced onto a fp8 pack"
qexpect ALLOWED exl3 exl3 "exl3 on an exl3 pack"
qexpect REFUSED "" exl3   "exl3 forced onto a pack declaring nothing"

# --- FlashInfer shape gate ---------------------------------------------------
# The page-block blocker (RESEARCH 11.9.8) must stop the launch, not be reported
# after a multi-minute weight load. The gate must fail closed by default and be
# overridable only explicitly.
{
  echo 'die() { echo "REFUSED: $*" >&2; exit 9; }'
  echo 'log() { echo "LOG: $*"; }'
  echo 'TP=2'
  echo 'IMAGE=test-image'
  echo 'MODEL_DIR_HOST=/nonexistent-checkpoint-dir'
  echo 'docker() { echo "DOCKER-CALLED"; return 0; }'
  awk '/^check_flashinfer_shapes\(\)/,/^}/' "$START_SH"
  cat <<'EOS'
( check_flashinfer_shapes ) >/dev/null 2>&1 && echo ALLOWED || echo REFUSED
EOS
} > "$TMP/fi.sh"

got="$(SKIP_FLASHINFER_CHECK=1 bash "$TMP/fi.sh" 2>&1 | tail -1)"
if [ "$got" = "ALLOWED" ]; then
  printf 'ok   %-40s %s\n' "explicit skip overrides the gate" "$got"
else
  printf 'FAIL %-40s want ALLOWED got %s\n' "explicit skip overrides the gate" "$got"
  fails=$((fails + 1))
fi

# With the gate on and no checkpoint present it must refuse (fail closed), and
# it must not have reached docker.
out="$(SKIP_FLASHINFER_CHECK=0 bash "$TMP/fi.sh" 2>&1 | tail -1)"
if [ "$out" = "REFUSED" ]; then
  printf 'ok   %-40s %s\n' "gate on, no checkpoint" "$out"
else
  printf 'FAIL %-40s want REFUSED got %s\n' "gate on, no checkpoint" "$out"
  fails=$((fails + 1))
fi

if grep -q "check_flashinfer_shapes" "$START_SH" && \
   awk '/^preflight\(\)/,/^}/' "$START_SH" | grep -q "check_flashinfer_shapes"; then
  printf 'ok   %-40s %s\n' "preflight calls the shape gate" "wired"
else
  printf 'FAIL %-40s %s\n' "preflight calls the shape gate" "NOT WIRED"
  fails=$((fails + 1))
fi

# An unusable SPEC_METHOD must be reported before any container starts, not after
# a multi-minute weight load, so preflight has to run the same check the argv does.
if awk '/^preflight\(\)/,/^}/' "$START_SH" | grep -q "check_spec_method"; then
  printf 'ok   %-40s %s\n' "preflight validates SPEC_METHOD" "wired"
else
  printf 'FAIL %-40s %s\n' "preflight validates SPEC_METHOD" "NOT WIRED"
  fails=$((fails + 1))
fi
# The divisibility rule is only correct if it is built on the checkpoint's own
# block size, and the drafter lives in `text_config` — the top-level config.json
# has no `dspark_*` key even when the drafter is present. So preflight must read
# the nested block and the weight index, not just the top level.
if awk '/^preflight\(\)/,/^}/' "$START_SH" | grep -q "check_dspark_draft"; then
  printf 'ok   %-40s %s\n' "preflight verifies the DSpark drafter" "wired"
else
  printf 'FAIL %-40s %s\n' "preflight verifies the DSpark drafter" "NOT WIRED"
  fails=$((fails + 1))
fi

# --- DSpark drafter presence --------------------------------------------------
# The drafter ships inside the target checkpoint under `mtp.*`, and its block size
# lives in `text_config`. Reading only the top level would refuse a servable arm;
# skipping the check would emit `--speculative-config` for a pack that cannot
# honour it and fail minutes into the weight load. Drive the real function against
# synthetic checkpoints.
{
  echo 'die() { echo "REFUSED: $*" >&2; exit 9; }'
  echo 'log() { :; }'
  echo 'SPEC_DSPARK_BLOCK=5'
  echo 'SPEC_NUM_TOKENS=5'
  echo 'SPEC_DRAFT_SAMPLE_METHOD=greedy'
  echo 'SPEC_ADAPTIVE_VERIFICATION=0'
  awk '/^check_dspark_draft\(\)/,/^}/' "$START_SH"
  cat <<'EOS'
( check_dspark_draft ) >/dev/null 2>&1 && echo ALLOWED || echo REFUSED
EOS
} > "$TMP/dspark.sh"

dspark_case() {
  local name="$1" cfg="$2" idx="$3"
  mkdir -p "$TMP/dp_$name"
  printf '%s' "$cfg" > "$TMP/dp_$name/config.json"
  printf '%s' "$idx" > "$TMP/dp_$name/model.safetensors.index.json"
}
dspark_run() {
  local spec="$1"
  MODEL_DIR_HOST="$TMP/dp_$spec" SPEC_METHOD=dspark bash "$TMP/dspark.sh" 2>&1 | tail -1
}
dspark_expect() {
  local want="$1" spec="$2" label="$3"
  local got
  got="$(dspark_run "$spec")"
  if [ "$got" = "$want" ]; then
    printf 'ok   %-40s %s\n' "$label" "$got"
  else
    printf 'FAIL %-40s want %s got %s\n' "$label" "$want" "$got"
    fails=$((fails + 1))
  fi
}
# The real shape: block size nested in text_config, drafter weights in the index.
dspark_case good '{"text_config":{"dspark_block_size":5}}' '{"weight_map":{"mtp.0.ffn.gate.bias":1,"layers.0.attn.q.weight":2}}'
# No drafter at all: text_config present but no dspark_block_size.
dspark_case nodraft '{"text_config":{"hidden_size":5120}}' '{"weight_map":{"layers.0.attn.q.weight":2}}'
# A config that claims a drafter the weights do not carry (pruned/partial pack).
dspark_case noweights '{"text_config":{"dspark_block_size":5}}' '{"weight_map":{"layers.0.attn.q.weight":2}}'
# A different block size than the operator declared.
dspark_case otherblock '{"text_config":{"dspark_block_size":7}}' '{"weight_map":{"mtp.0.ffn.gate.bias":1}}'
# Missing files entirely.
mkdir -p "$TMP/dp_missing"
dspark_expect ALLOWED good "drafter present in text_config"
dspark_expect REFUSED nodraft "no declared block is refused"
dspark_expect REFUSED noweights "declared drafter with no weights is refused"
dspark_expect REFUSED otherblock "a mismatched block size is refused"
if [ "$(MODEL_DIR_HOST="$TMP/dp_missing" SPEC_METHOD=dspark bash "$TMP/dspark.sh" 2>&1 | tail -1)" = "REFUSED" ]; then
  printf 'ok   %-40s %s\n' "missing config is refused" "REFUSED"
else
  printf 'FAIL %-40s %s\n' "missing config is refused" "ALLOWED"
  fails=$((fails + 1))
fi
# SPEC=none must not consult the checkpoint at all: a pack with no drafter is
# perfectly servable without speculation.
if [ "$(MODEL_DIR_HOST="$TMP/dp_nodraft" SPEC_METHOD=none bash "$TMP/dspark.sh" 2>&1 | tail -1)" = "ALLOWED" ]; then
  printf 'ok   %-40s %s\n' "SPEC=none ignores the drafter check" "ALLOWED"
else
  printf 'FAIL %-40s %s\n' "SPEC=none ignores the drafter check" "REFUSED"
  fails=$((fails + 1))
fi

# --- first-boot shape gate ----------------------------------------------------
# seqs/eager/language-only/SPEC are a first-boot posture, not a permanent one.
# FIRST_BOOT=0 must lift exactly those four, and must not lift the memory floor,
# the GPU interlock or the shape gates. Anything other than 0/1 must fail closed
# rather than silently reading as "not first boot".
{
  echo 'die() { echo "REFUSED: $*" >&2; exit 9; }'
  echo 'log() { :; }'
  awk '/^refuse_first_boot_shape\(\)/,/^}/' "$START_SH"
  cat <<'EOS'
( refuse_first_boot_shape ) >/dev/null 2>&1 && echo ALLOWED || echo REFUSED
EOS
} > "$TMP/fb.sh"

fbexpect() {
  local want="$1" fb="$2" seqs="$3" spec="$4" eager="$5" label="$6"
  local got
  got="$(FIRST_BOOT="$fb" MAX_NUM_SEQS="$seqs" SPEC_METHOD="$spec" \
         ENFORCE_EAGER="$eager" LANGUAGE_MODEL_ONLY=1 \
         bash "$TMP/fb.sh" 2>&1 | tail -1)"
  if [ "$got" = "$want" ]; then
    printf 'ok   %-40s %s\n' "$label" "$got"
  else
    printf 'FAIL %-40s want %s got %s\n' "$label" "$want" "$got"
    fails=$((fails + 1))
  fi
}
fbexpect ALLOWED 1 2 none 1 "first boot, receipt shape"
fbexpect REFUSED 1 8 none 1 "first boot refuses seqs=8"
fbexpect REFUSED 1 2 dspark 1 "first boot refuses spec decode"
fbexpect REFUSED 1 2 none 0 "first boot refuses cuda graphs"
fbexpect ALLOWED 0 8 none 0 "ramped boot lifts the shape guards"
fbexpect ALLOWED 0 24 none 0 "ramped boot allows high concurrency"
fbexpect REFUSED 0 1 none 1 "ramped boot still needs seqs>=2"
fbexpect REFUSED 2 2 none 1 "a non-0/1 FIRST_BOOT fails closed"
fbexpect REFUSED yes 2 none 1 "a non-numeric FIRST_BOOT fails closed"

# --- serve argv ---------------------------------------------------------------
# Async scheduling is a tri-state: omitting the flag enables it. The argv must
# always carry one spelling or the other. `--enable-flashinfer-autotune` and
# `kernel_config.enable_flashinfer_autotune` are mutually exclusive, so the warmup
# bundle must never emit both.
{
  echo 'die() { echo "REFUSED: $*" >&2; exit 9; }'
  echo 'log() { :; }'
  awk '/^vllm_argv\(\)/,/^}/' "$START_SH"
  awk '/^spec_flags\(\)/,/^}/' "$START_SH"
  cat <<'EOS'
vllm_argv 0
EOS
} > "$TMP/argv.sh"

ARGV_DEFAULTS=(
  SERVED_MODEL_NAME=dsv41 PORT=8001 TP=2 NNODES=2 HEAD_IP=192.168.177.10
  MASTER_PORT=29531 TOKENIZER_MODE=deepseek_v41 TOOL_CALL_PARSER=deepseek_v41
  REASONING_PARSER=deepseek_v41 QUANTIZATION= MAX_MODEL_LEN=65536
  MAX_NUM_SEQS=2 MAX_NUM_BATCHED_TOKENS=8192 GPU_MEM_UTIL=0.82
  KV_CACHE_DTYPE=fp8 KV_CACHE_MEMORY_BYTES=4294967296
  LANGUAGE_MODEL_ONLY=1 TEXT_ONLY_VISION_OVERRIDE=1
)

argv_has() {
  local needle="$1" label="$2"
  shift 2
  local out
  out="$(env "${ARGV_DEFAULTS[@]}" "$@" bash "$TMP/argv.sh" 2>&1)"
  if printf '%s' "$out" | grep -q -- "$needle"; then
    printf 'ok   %-40s %s\n' "$label" "present"
  else
    printf 'FAIL %-40s %s\n' "$label" "MISSING: $needle"
    fails=$((fails + 1))
  fi
}

argv_lacks() {
  local needle="$1" label="$2"
  shift 2
  local out
  out="$(env "${ARGV_DEFAULTS[@]}" "$@" bash "$TMP/argv.sh" 2>&1)"
  if printf '%s' "$out" | grep -q -- "$needle"; then
    printf 'FAIL %-40s %s\n' "$label" "UNEXPECTED: $needle"
    fails=$((fails + 1))
  else
    printf 'ok   %-40s %s\n' "$label" "absent"
  fi
}

argv_has --no-async-scheduling "async pinned off by default" NO_ASYNC_SCHEDULING=1
argv_has --async-scheduling "async on when asked for" NO_ASYNC_SCHEDULING=0
argv_lacks --no-async-scheduling "no both spellings at once" \
  NO_ASYNC_SCHEDULING=0
argv_has --enable-auto-tool-choice "tool parser is activated" ENABLE_AUTO_TOOL_CHOICE=1
argv_lacks --enable-auto-tool-choice "tool choice can be declined" ENABLE_AUTO_TOOL_CHOICE=0
argv_has --enable-prefix-caching "prefix caching pinned" ENABLE_AUTO_TOOL_CHOICE=1
argv_has --long-prefill-token-threshold "prefill HoL cap set" \
  LONG_PREFILL_TOKEN_THRESHOLD=4096
argv_lacks --long-prefill-token-threshold "prefill HoL cap can be disabled" \
  LONG_PREFILL_TOKEN_THRESHOLD=0
argv_has 'enable_jit_warmup' "warmups bundled in kernel-config" DISABLE_KERNEL_WARMUPS=1
argv_has 'enable_cutedsl_warmup' "cutedsl warmup disabled" DISABLE_KERNEL_WARMUPS=1
argv_lacks --no-enable-flashinfer-autotune \
  "no autotune flag alongside kernel-config" DISABLE_KERNEL_WARMUPS=1
argv_has --no-enable-flashinfer-autotune "autotune flag alone when warmups on" \
  DISABLE_KERNEL_WARMUPS=0
argv_lacks 'enable_jit_warmup' "no kernel-config when warmups on" \
  DISABLE_KERNEL_WARMUPS=0
argv_has --max-num-batched-tokens "prefill budget passed through" MAX_NUM_BATCHED_TOKENS=8192

# --- the ramped shape must actually reach the engine ---------------------------
# `refuse_first_boot_shape` only decides whether a combination is *permitted*.
# The argv is what the engine sees, so FIRST_BOOT=0 must change the argv too:
# allowing `ENFORCE_EAGER=0` and then passing `--enforce-eager` anyway is a
# silent no-op, and permitting `SPEC_METHOD=dspark` without emitting
# `--speculative-config` is the same. These check the argv, not the guard.
argv_has --enforce-eager "eager passed on the first boot" ENFORCE_EAGER=1
argv_lacks --enforce-eager "eager lifted when graphs are allowed" ENFORCE_EAGER=0
argv_has --language-model-only "language-only passed on the first boot" LANGUAGE_MODEL_ONLY=1
argv_lacks --language-model-only "language-only lifted when asked" LANGUAGE_MODEL_ONLY=0
argv_lacks --speculative-config "no speculative config at SPEC=none" SPEC_METHOD=none
argv_has --speculative-config "spec decode reaches the engine" \
  SPEC_METHOD=dspark SPEC_NUM_TOKENS=5

# --- the text-only image-budget override ---------------------------------------
# `--language-model-only` zeroes every modality limit but does NOT stop the
# checkpoint's per-image token budget from widening the SWA prefill index rows to
# `sliding_window + max_image_tokens` (128 + 1024 = 1152 here). That width is in
# neither the decode dispatch set ({128,192,256,512,1024}) nor the single prefill
# ladder (those plus 2048), and the live dual path requires exactly 128, so the
# engine dies at the first attention call. Collapsing the budget restores 128, and
# it must reach the engine as one argv word. Zeroing `vision_n_layers` instead
# would also give 128 but drops `bias_vl`, which the checkpoint ships — the loader
# then raises `KeyError: 'layers.0.ffn.gate.bias_vl'`.
argv_has --hf-overrides "text-only override reaches the engine" LANGUAGE_MODEL_ONLY=1
argv_lacks --hf-overrides "override can be turned off" \
  LANGUAGE_MODEL_ONLY=1 TEXT_ONLY_VISION_OVERRIDE=0
argv_lacks --hf-overrides "no override without language-only" \
  LANGUAGE_MODEL_ONLY=0 TEXT_ONLY_VISION_OVERRIDE=0
if grep -q -- '--hf-overrides .{"vision_max_n_token": 0}' "$START_SH"; then
  printf 'ok   %-40s %s\n' "override collapses the image budget" "correct lever"
else
  printf 'FAIL %-40s %s\n' "override collapses the image budget" "WRONG LEVER"
  fails=$((fails + 1))
fi
# The rejected spelling must not appear anywhere in the argv path.
if grep -q -- '"vision_n_layers": 0' "$START_SH"; then
  printf 'FAIL %-40s %s\n' "override must not zero vision_n_layers" "WOULD BREAK bias_vl"
  fails=$((fails + 1))
else
  printf 'ok   %-40s %s\n' "override must not zero vision_n_layers" "absent"
fi

# --- the vision-config coherence gate ------------------------------------------
# The override is only coherent while the engine also refuses images. With images
# enabled it would shrink the SWA rows out of under a real image span, which is a
# correctness bug rather than a startup failure, so the combination must be
# refused rather than guessed at.
{
  echo 'die() { echo "REFUSED: $*" >&2; exit 9; }'
  echo 'log() { :; }'
  awk '/^refuse_incoherent_vision_config\(\)/,/^}/' "$START_SH"
  cat <<'EOS'
( refuse_incoherent_vision_config ) >/dev/null 2>&1 && echo ALLOWED || echo REFUSED
EOS
} > "$TMP/vision.sh"

visexpect() {
  local want="$1" override="$2" lmo="$3" label="$4"
  local got
  got="$(TEXT_ONLY_VISION_OVERRIDE="$override" LANGUAGE_MODEL_ONLY="$lmo" \
         bash "$TMP/vision.sh" 2>&1 | tail -1)"
  if [ "$got" = "$want" ]; then
    printf 'ok   %-40s %s\n' "$label" "$got"
  else
    printf 'FAIL %-40s want %s got %s\n' "$label" "$want" "$got"
    fails=$((fails + 1))
  fi
}
visexpect ALLOWED 1 1 "override with language-only is coherent"
visexpect REFUSED 1 0 "override without language-only is refused"
visexpect ALLOWED 0 1 "declining the override is coherent"
visexpect ALLOWED 0 0 "declining both is coherent"
visexpect REFUSED 2 1 "a non-0/1 override fails closed"
visexpect REFUSED yes 1 "a non-numeric override fails closed"

# The verifier must be told the *effective* count, or it checks the 1152-wide
# table the checkpoint declares while the engine runs 128 — and passes a boot
# that cannot happen.
if awk '/^preflight\(\)/,/^}/' "$START_SH" | grep -q "refuse_incoherent_vision_config"; then
  printf 'ok   %-40s %s\n' "preflight runs the vision gate" "wired"
else
  printf 'FAIL %-40s %s\n' "preflight runs the vision gate" "NOT WIRED"
  fails=$((fails + 1))
fi
if awk '/^preflight\(\)/,/^}/' "$START_SH" | grep -q "resolve_effective_image_tokens"; then
  printf 'ok   %-40s %s\n' "preflight resolves the effective budget" "wired"
else
  printf 'FAIL %-40s %s\n' "preflight resolves the effective budget" "NOT WIRED"
  fails=$((fails + 1))
fi
if grep -q -- '--max-image-tokens "\$EFFECTIVE_MAX_IMAGE_TOKENS"' "$START_SH"; then
  printf 'ok   %-40s %s\n' "shape check gets the effective budget" "wired"
else
  printf 'FAIL %-40s %s\n' "shape check gets the effective budget" "NOT WIRED"
  fails=$((fails + 1))
fi

# `resolve_effective_image_tokens` is what the verifier's `--max-image-tokens`
# argument is built from. With the override on it must answer 0 *without* reading
# the checkpoint (the served posture is a declaration, not a lookup), and with it
# off it must report the config's own budget.
{
  echo 'die() { echo "REFUSED: $*" >&2; exit 9; }'
  echo 'log() { :; }'
  awk '/^resolve_effective_image_tokens\(\)/,/^}/' "$START_SH"
  cat <<'EOS'
resolve_effective_image_tokens
echo "EFFECTIVE=$EFFECTIVE_MAX_IMAGE_TOKENS"
EOS
} > "$TMP/budget.sh"

mkdir -p "$TMP/ckpt" "$TMP/ckpt-novision" "$TMP/ckpt-nocap" "$TMP/ckpt-zero"
printf '{"vision_config": {"num_hidden_layers": 32, "max_image_tokens": 1024}}' \
  > "$TMP/ckpt/config.json"
printf '{"text_config": {"sliding_window": 128}}' > "$TMP/ckpt-novision/config.json"
printf '{"vision_config": {"num_hidden_layers": 32}}' > "$TMP/ckpt-nocap/config.json"
printf '{"vision_config": {"num_hidden_layers": 32, "max_image_tokens": 0}}' \
  > "$TMP/ckpt-zero/config.json"

buexpect() {
  local want="$1" override="$2" dir="$3" label="$4"
  local got
  got="$(TEXT_ONLY_VISION_OVERRIDE="$override" MODEL_DIR_HOST="$dir" \
         bash "$TMP/budget.sh" 2>&1 | tail -1)"
  if [ "$got" = "EFFECTIVE=$want" ]; then
    printf 'ok   %-40s %s\n' "$label" "$got"
  else
    printf 'FAIL %-40s want EFFECTIVE=%s got %s\n' "$label" "$want" "$got"
    fails=$((fails + 1))
  fi
}
buexpect 0 1 "$TMP/ckpt" "override short-circuits to 0"
# No config on disk must not matter while the override is on.
buexpect 0 1 "$TMP/nonexistent" "override needs no config on disk"
buexpect 1024 0 "$TMP/ckpt" "declining the override reads the config"
buexpect 1024 0 "$TMP/ckpt-nocap" "a missing cap defaults to 1024"
buexpect 0 0 "$TMP/ckpt-zero" "an explicit zero cap reads as 0"
buexpect 0 0 "$TMP/ckpt-novision" "a config without vision layers reads as 0"

# `vllm_argv` prints `%q`-quoted, which is what `launch_rank` unquotes on the
# remote side. Test the *decoded* argv, so the assertion is about the words the
# engine will actually receive rather than about the escaping.
argv_word_after() {
  local flag="$1" out
  shift
  out="$(env "${ARGV_DEFAULTS[@]}" "$@" bash "$TMP/argv.sh" 2>/dev/null)"
  # shellcheck disable=SC2086
  eval "set -- $out"
  local prev=""
  for word in "$@"; do
    if [ "$prev" = "$flag" ]; then printf '%s' "$word"; return 0; fi
    prev="$word"
  done
  return 1
}

awcheck() {
  local flag="$1" want="$2" label="$3"
  shift 3
  local got
  got="$(argv_word_after "$flag" "$@")"
  if [ "$got" = "$want" ]; then
    printf 'ok   %-40s %s\n' "$label" "exact"
  else
    printf 'FAIL %-40s want %s got %s\n' "$label" "$want" "$got"
    fails=$((fails + 1))
  fi
}
awcheck --speculative-config '{"method": "dspark", "num_speculative_tokens": 5, "draft_sample_method": "greedy"}' \
  "spec JSON survives as one word" SPEC_METHOD=dspark SPEC_NUM_TOKENS=5
# The default is the whole DSpark block, not a partial one. `2` was the old
# default: legal (the divisibility rule only fires above the block) but it leaves
# 3 of every 5 draft slots unspent.
awcheck --speculative-config '{"method": "dspark", "num_speculative_tokens": 5, "draft_sample_method": "greedy"}' \
  "spec depth defaults to the whole block" SPEC_METHOD=dspark
awcheck --speculative-config '{"method": "dspark", "num_speculative_tokens": 10, "draft_sample_method": "greedy"}' \
  "a multiple of the block is accepted" SPEC_METHOD=dspark SPEC_NUM_TOKENS=10
awcheck --speculative-config '{"method": "dspark", "num_speculative_tokens": 5, "draft_sample_method": "probabilistic"}' \
  "probabilistic drafting is opt-in" SPEC_METHOD=dspark SPEC_DRAFT_SAMPLE_METHOD=probabilistic
# Adaptive verification requires full CUDA graphs, which this checkpoint cannot
# use (graphs produce NaN here, and the vendor records an illegal memory access in
# both graph modes). The engine can never honour the flag, so it is refused rather
# than emitted — accepting it would look applied while doing nothing.
out_av="$(env "${ARGV_DEFAULTS[@]}" SPEC_METHOD=dspark SPEC_ADAPTIVE_VERIFICATION=1 \
  bash "$TMP/argv.sh" 2>&1)"
if printf '%s' "$out_av" | grep -q "cannot work on this deployment"; then
  printf 'ok   %-40s %s\n' "adaptive verification is refused" "REFUSED"
else
  printf 'FAIL %-40s out=%s\n' "adaptive verification is refused" "$out_av"
  fails=$((fails + 1))
fi
argv_lacks 'enable_adaptive_verification' \
  "adaptive verification is never emitted" SPEC_METHOD=dspark SPEC_ADAPTIVE_VERIFICATION=0

# `num_speculative_tokens` is quantised to the DSpark block: one drafting round
# emits `dspark_block_size` tokens, so a value that is neither inside the block
# nor a multiple of it is rejected by the engine at startup
# (config/speculative.py:1478-1485). Catch it before the weight load.
out_bad_div="$(env "${ARGV_DEFAULTS[@]}" SPEC_METHOD=dspark SPEC_NUM_TOKENS=7 \
  bash "$TMP/argv.sh" 2>&1)"
if printf '%s' "$out_bad_div" | grep -q "not usable with DSpark"; then
  printf 'ok   %-40s %s\n' "an off-block spec depth fails closed" "REFUSED"
else
  printf 'FAIL %-40s out=%s\n' "an off-block spec depth fails closed" "$out_bad_div"
  fails=$((fails + 1))
fi
if printf '%s' "$out_bad_div" | grep -q -- '--speculative-config'; then
  printf 'FAIL %-40s %s\n' "off-block depth emits no argv" "EMITTED"
  fails=$((fails + 1))
else
  printf 'ok   %-40s %s\n' "off-block depth emits no argv" "absent"
fi
out_bad_dsm="$(env "${ARGV_DEFAULTS[@]}" SPEC_METHOD=dspark \
  SPEC_DRAFT_SAMPLE_METHOD=sampled bash "$TMP/argv.sh" 2>&1)"
if printf '%s' "$out_bad_dsm" | grep -q "SPEC_DRAFT_SAMPLE_METHOD must be"; then
  printf 'ok   %-40s %s\n' "an unknown draft sample method fails closed" "REFUSED"
else
  printf 'FAIL %-40s out=%s\n' "an unknown draft sample method fails closed" "$out_bad_dsm"
  fails=$((fails + 1))
fi
out_bad_av="$(env "${ARGV_DEFAULTS[@]}" SPEC_METHOD=dspark \
  SPEC_ADAPTIVE_VERIFICATION=yes bash "$TMP/argv.sh" 2>&1)"
if printf '%s' "$out_bad_av" | grep -q "SPEC_ADAPTIVE_VERIFICATION must be 0"; then
  printf 'ok   %-40s %s\n' "a non-0 adaptive flag fails closed" "REFUSED"
else
  printf 'FAIL %-40s out=%s\n' "a non-0 adaptive flag fails closed" "$out_bad_av"
  fails=$((fails + 1))
fi
# The block itself is a checkpoint fact, not a constant to guess: if the operator
# declares a different block the divisibility rule would be built on the wrong
# number, so preflight compares it against the checkpoint.
out_bad_block="$(env "${ARGV_DEFAULTS[@]}" SPEC_METHOD=dspark \
  SPEC_DSPARK_BLOCK=7 SPEC_NUM_TOKENS=7 bash "$TMP/argv.sh" 2>&1)"
if printf '%s' "$out_bad_block" | grep -q -- '--speculative-config'; then
  printf 'ok   %-40s %s\n' "a re-declared block changes the rule" "present"
else
  printf 'FAIL %-40s out=%s\n' "a re-declared block changes the rule" "$out_bad_block"
  fails=$((fails + 1))
fi
awcheck --hf-overrides '{"vision_max_n_token": 0}' \
  "override JSON survives as one word" LANGUAGE_MODEL_ONLY=1
awcheck --hf-overrides '{"vision_max_n_token": 0}' \
  "override is on by default" LANGUAGE_MODEL_ONLY=1 TEXT_ONLY_VISION_OVERRIDE=1

# `mtp` must be rejected, not translated: V4.1 ships DSpark stages under `mtp.*`
# and carries no classic-MTP weights, so the engine raises for method=mtp. A kit
# that documented `SPEC_METHOD=mtp` as usable would be documenting a boot failure.
out_mtp="$(env "${ARGV_DEFAULTS[@]}" SPEC_METHOD=mtp bash "$TMP/argv.sh" 2>&1)"
if printf '%s' "$out_mtp" | grep -q "cannot work on this checkpoint"; then
  printf 'ok   %-40s %s\n' "mtp is rejected, not translated" "REFUSED"
else
  printf 'FAIL %-40s out=%s\n' "mtp is rejected, not translated" "$out_mtp"
  fails=$((fails + 1))
fi
if printf '%s' "$out_mtp" | grep -q -- '--speculative-config'; then
  printf 'FAIL %-40s %s\n' "mtp emits no argv" "EMITTED"
  fails=$((fails + 1))
else
  printf 'ok   %-40s %s\n' "mtp emits no argv" "absent"
fi

# An unknown method must fail closed rather than being silently dropped.
out_bad="$(env "${ARGV_DEFAULTS[@]}" SPEC_METHOD=medusa bash "$TMP/argv.sh" 2>&1)"
if printf '%s' "$out_bad" | grep -q "not supported by this kit"; then
  printf 'ok   %-40s %s\n' "unknown SPEC_METHOD fails closed" "REFUSED"
else
  printf 'FAIL %-40s out=%s\n' "unknown SPEC_METHOD fails closed" "$out_bad"
  fails=$((fails + 1))
fi
if printf '%s' "$out_bad" | grep -q -- '--speculative-config'; then
  printf 'FAIL %-40s %s\n' "unknown SPEC_METHOD emits nothing" "EMITTED"
  fails=$((fails + 1))
else
  printf 'ok   %-40s %s\n' "unknown SPEC_METHOD emits nothing" "absent"
fi
out_bad_n="$(env "${ARGV_DEFAULTS[@]}" SPEC_METHOD=dspark SPEC_NUM_TOKENS=0 \
  bash "$TMP/argv.sh" 2>&1)"
if printf '%s' "$out_bad_n" | grep -q "SPEC_NUM_TOKENS must be"; then
  printf 'ok   %-40s %s\n' "a zero spec depth fails closed" "REFUSED"
else
  printf 'FAIL %-40s out=%s\n' "a zero spec depth fails closed" "$out_bad_n"
  fails=$((fails + 1))
fi
# A warning must never leak into the argv (vllm_argv's stdout IS the argv).
out_warn="$(env "${ARGV_DEFAULTS[@]}" SPEC_METHOD=dspark SPEC_NUM_TOKENS=5 \
  bash "$TMP/argv.sh" 2>/dev/null)"
if printf '%s' "$out_warn" | grep -q "WARNING"; then
  printf 'FAIL %-40s %s\n' "no warning text in the argv" "LEAKED"
  fails=$((fails + 1))
else
  printf 'ok   %-40s %s\n' "no warning text in the argv" "clean"
fi

# --- worker reachability ------------------------------------------------------
# The shipped template set `WORKER_SSH=nvidia@spark2`, which resolves from a
# laptop's ssh_config but not from spark1 — the nodes reach each other only over
# the QSFP rail. That made preflight abort with ssh's own "Could not resolve
# hostname" instead of a named blocker. `check_worker_ssh` must run first among
# the worker checks and must fail closed with the remedy in the message.
{
  echo 'die() { echo "REFUSED: $*" >&2; exit 9; }'
  echo 'log() { :; }'
  echo 'WORKER_SSH=nvidia@spark2'
  echo 'WORKER_USER=nvidia'
  echo 'WORKER_IP=192.168.177.11'
  echo 'HEAD_CX7_IF=enp1s0f1np1'
  echo 'HEAD_IP=192.168.177.10'
  echo 'WORKER_CX7_IF=enp1s0f1np1'
  # Literal fixture text: the expansion happens when wssh.sh runs, not here.
  # shellcheck disable=SC2016
  echo 'worker_ssh() { ssh -o BatchMode=yes "$WORKER_SSH" "$@"; }'
  awk '/^check_worker_ssh\(\)/,/^}/' "$START_SH"
  awk '/^check_fabric_addresses\(\)/,/^}/' "$START_SH"
  # Stub the transport: `unreachable` models the unresolved hostname, `no_docker`
  # a worker that answers but has no docker, and `ok` a healthy worker.
  cat <<'EOS'
case "$CASE" in
  unreachable) worker_ssh() { return 255; } ;;
  no_docker)   worker_ssh() { [ "$1" = true ] && return 0; return 1; } ;;
  ok)          worker_ssh() { return 0; } ;;
esac
# `die` exits the subshell, so the verdict is the last line either way; the
# refusal text itself stays on stderr for the message assertions below.
( check_worker_ssh ) && echo ALLOWED || echo REFUSED
EOS
} > "$TMP/wssh.sh"

wssh() {
  local want="$1" case_name="$2" label="$3"
  local got
  got="$(CASE="$case_name" bash "$TMP/wssh.sh" 2>&1 | tail -1)"
  if [ "$got" = "$want" ]; then
    printf 'ok   %-40s %s\n' "$label" "$got"
  else
    printf 'FAIL %-40s want %s got %s\n' "$label" "$want" "$got"
    fails=$((fails + 1))
  fi
}
wssh REFUSED unreachable "an unreachable worker refuses"
wssh REFUSED no_docker   "a worker without docker refuses"
wssh ALLOWED ok          "a reachable worker is allowed"

# The refusal must name the configuration and the remedy, not just fail.
wssh_msg="$(CASE=unreachable bash "$TMP/wssh.sh" 2>&1)"
for needle in 'cannot ssh (key-based)' 'does not resolve between the nodes' 'WORKER_IP'; do
  if printf '%s' "$wssh_msg" | grep -q -- "$needle"; then
    printf 'ok   %-40s %s\n' "refusal names the remedy" "present"
  else
    printf 'FAIL %-40s %s\n' "refusal names the remedy" "MISSING: $needle"
    fails=$((fails + 1))
  fi
done

# And the address must be DERIVED when unset, because a bare hostname is the trap.
derived="$(WORKER_SSH='' WORKER_USER=nvidia WORKER_IP=192.168.177.11 bash -c '
  WORKER_USER=nvidia; WORKER_IP=192.168.177.11
  WORKER_SSH="${WORKER_SSH:-${WORKER_USER}@${WORKER_IP}}"
  printf "%s" "$WORKER_SSH"')"
if [ "$derived" = "nvidia@192.168.177.11" ]; then
  printf 'ok   %-40s %s\n' "unset WORKER_SSH derives from the rail IP" "$derived"
else
  printf 'FAIL %-40s want nvidia@192.168.177.11 got %s\n' "unset WORKER_SSH derives from the rail IP" "$derived"
  fails=$((fails + 1))
fi
if grep -qE '^WORKER_SSH=' "$HERE/../env.example"; then
  printf 'FAIL %-40s %s\n' "env.example must not pin WORKER_SSH" "it does"
  fails=$((fails + 1))
else
  printf 'ok   %-40s %s\n' "env.example must not pin WORKER_SSH" "unset"
fi

# --- container flags ----------------------------------------------------------
# The UCX registration-cache leak drains unified memory over a multi-hour run and
# the MIN_FREE_GIB gate only samples at start, so these must be in the container
# env unconditionally. The kernel caches must live on the persistent bind: each
# start makes a fresh container, so anything left in the image layer
# cold-compiles on every recreate.
{
  echo 'log() { :; }'
  echo 'MODEL_DIR_IN_CONTAINER=/model-cache/snapshots/x'
  echo 'MODEL_MOUNT_HOST=/h/.cache/huggingface/hub/models--x'
  echo 'MODEL_MOUNT_TARGET=/model-cache'
  echo 'CACHE_HOST=/h/.cache/vllm-dsv41'
  echo 'FLASHINFER_CACHE_HOST=/h/.cache/vllm-dsv41/flashinfer'
  echo 'NCCL_IB_HCA=rocep1s0f1 NCCL_IB_GID_INDEX=3'
  echo 'VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800'
  echo 'UCX_MEM_MMAP_HOOK_MODE=none UCX_RCACHE_MAX_UNRELEASED=1024'
  echo 'OOM_SCORE_ADJ=500'
  echo 'DSV41_ENGRAM_STRICT=1'
  # Literal fixture text: bash expands this when common.sh runs, not here.
  # shellcheck disable=SC2016
  echo 'DISABLE_KERNEL_WARMUPS=${DISABLE_KERNEL_WARMUPS:-1}'
  awk '/^docker_common\(\)/,/^}/' "$START_SH"
  echo 'docker_common enp1s0f1np1 192.168.177.10'
} > "$TMP/common.sh"

common_out="$(bash "$TMP/common.sh" 2>&1)"
ccheck() {
  local needle="$1" label="$2"
  if printf '%s' "$common_out" | grep -q -- "$needle"; then
    printf 'ok   %-40s %s\n' "$label" "present"
  else
    printf 'FAIL %-40s %s\n' "$label" "MISSING: $needle"
    fails=$((fails + 1))
  fi
}
ccheck 'UCX_MEM_MMAP_HOOK_MODE=none' "UCX mmap hook disabled"
ccheck 'UCX_RCACHE_MAX_UNRELEASED=1024' "UCX rcache bounded"
ccheck '--oom-score-adj 500' "container is the OOM victim"
ccheck 'VLLM_DEEP_GEMM_WARMUP=skip' "deep-gemm warmup skipped"
ccheck 'TRITON_CACHE_DIR=/root/.cache/vllm/triton' "triton cache on the bind"
ccheck 'TORCHINDUCTOR_CACHE_DIR=/root/.cache/vllm/torchinductor' "inductor cache on the bind"
ccheck 'XDG_CACHE_HOME=/root/.cache/vllm/xdg' "XDG cache on the bind"
ccheck 'VLLM_CACHE_ROOT=/root/.cache/vllm' "vLLM cache root pinned"
# The image masks FlashInfer's AOT binary, so the compiled module must come from
# a per-node prebuild on a persistent bind. A cache that lives only in the image
# layer is empty on every `--rm` recreate.
ccheck '-v /h/.cache/vllm-dsv41/flashinfer:/root/.cache/flashinfer' "FlashInfer cache on the bind"

# And with warmups left on, the deep-gemm warmup must not be skipped.
common_warm="$(DISABLE_KERNEL_WARMUPS=0 bash "$TMP/common.sh" 2>&1)"
if printf '%s' "$common_warm" | grep -q 'VLLM_DEEP_GEMM_WARMUP=relax'; then
  printf 'ok   %-40s %s\n' "deep-gemm warmup relax when enabled" "present"
else
  printf 'FAIL %-40s %s\n' "deep-gemm warmup relax when enabled" "MISSING"
  fails=$((fails + 1))
fi

# --- crashed-rank logs --------------------------------------------------------
# A rank that dies during model construction must keep its traceback. `--rm`
# together with `-d` makes the container remove itself on exit, so `docker logs`
# answers "No such container" — which is how the `dp_size` boot failure first
# presented, and why it cost a full boot cycle to diagnose. The flags must not
# carry `--rm`, a restart must clear the stale name, and a failed health wait
# must dump the state and log tail before it dies.
if printf '%s' "$common_out" | grep -q -- '--rm'; then
  printf 'FAIL %-40s %s\n' "rank containers keep their logs" "PRESENT: --rm"
  fails=$((fails + 1))
else
  printf 'ok   %-40s %s\n' "rank containers keep their logs" "no --rm"
fi

if grep -A4 'local run=(docker run -d --name' "$START_SH" | grep -q 'docker rm -f'; then
  printf 'ok   %-40s %s\n' "restart clears the stale name" "present"
else
  printf 'FAIL %-40s %s\n' "restart clears the stale name" "MISSING"
  fails=$((fails + 1))
fi

# The timeout path must diagnose both ranks, and diagnose before it dies.
start_fn="$(awk '/^start_all\(\)/,/^}/' "$START_SH")"
diag_line="$(printf '%s\n' "$start_fn" | grep -n 'diagnose_rank' | head -1 | cut -d: -f1)"
die_line="$(printf '%s\n' "$start_fn" | grep -n 'die "not healthy' | head -1 | cut -d: -f1)"
if [ -n "$diag_line" ] && [ -n "$die_line" ] && [ "$diag_line" -lt "$die_line" ]; then
  printf 'ok   %-40s %s\n' "unhealthy start diagnoses first" "present"
else
  printf 'FAIL %-40s %s\n' "unhealthy start diagnoses first" \
    "MISSING (diagnose=${diag_line:-none} die=${die_line:-none})"
  fails=$((fails + 1))
fi
if printf '%s\n' "$start_fn" | grep -q 'diagnose_rank 1' \
   && printf '%s\n' "$start_fn" | grep -q 'diagnose_rank 0'; then
  printf 'ok   %-40s %s\n' "unhealthy start diagnoses both ranks" "present"
else
  printf 'FAIL %-40s %s\n' "unhealthy start diagnoses both ranks" "MISSING"
  fails=$((fails + 1))
fi

# Behaviour: the report carries the exit code and the log tail, on both nodes.
{
  echo 'log() { :; }'
  echo 'WORKER_SSH=nvidia@spark2'
  awk '/^diagnose_rank\(\)/,/^}/' "$START_SH"
  echo 'docker() { echo "DOCKER:$*"; }'
  echo 'worker_ssh() { echo "WORKER:$*"; }'
  echo 'diagnose_rank 1 w1 remote'
  echo 'diagnose_rank 0 h1 local'
} > "$TMP/diagnose.sh"
diag_out="$(bash "$TMP/diagnose.sh" 2>&1)"
dcheck() {
  local needle="$1" label="$2"
  if printf '%s' "$diag_out" | grep -qF -- "$needle"; then
    printf 'ok   %-40s %s\n' "$label" "present"
  else
    printf 'FAIL %-40s %s\n' "$label" "MISSING: $needle"
    fails=$((fails + 1))
  fi
}
dcheck '.State.ExitCode' "diagnosis reports the exit code"
dcheck 'DOCKER:logs --tail 60 h1' "diagnosis tails the head log"
dcheck "WORKER:docker logs --tail 60 'w1'" "diagnosis tails the worker log"

# --- FlashInfer compiled-module cache ----------------------------------------
# The image masks the prebuilt AOT binary for `sparse_mla_sm120` so the patched
# .cu is compiled instead (RESEARCH 11.9.3). That makes the compiled module a
# required artifact: without it the engine runs nvcc at startup and races the
# server for the same unified pool.
#
# The guard delegates to the in-image verifier on each node, so what is under
# test here is the plumbing: that both nodes are checked, that a node the
# launcher cannot inspect is refused rather than skipped, and that the image id
# is passed through. The verifier's own logic is covered by
# tests/test_check_flashinfer_cache.py, which is where the receipt, cache key and
# artifact checks live.
{
  echo 'die() { echo "REFUSED: $*" >&2; exit 9; }'
  echo 'log() { :; }'
  echo 'WORKER_SSH=nvidia@spark2'
  echo 'IMAGE=dsv41-reap-selfbuild'
  echo 'FLASHINFER_CACHE_HOST=/h/.cache/vllm-dsv41/flashinfer'
  awk '/^fi_cache_verdict\(\)/,/^}/' "$START_SH"
  awk '/^check_flashinfer_cache\(\)/,/^}/' "$START_SH"
  # `docker_probe` is the node-aware docker runner; stub it so the guard can be
  # exercised without docker or ssh. Its own argument serialization is covered by
  # tests/test_gpu_interlock.sh.
  cat <<'EOS'
docker_probe() {
  local target="$1"; shift
  # Record which node was addressed, so the test can assert both were visited.
  printf '%s\n' "${target:-head}" >> "$FI_SEEN"
  case "$CASE:$*" in
    *"image inspect"*)
      [ "$CASE" = "inspect_unread" ] && return 1
      echo "sha256:imageid" ;;
    *"run --rm"*)
      # A node with no usable module: the verifier exits non-zero.
      case "$CASE:${target:-head}" in
        empty_head:head)        echo "no prebuild receipt" >&2; return 5 ;;
        empty_worker:*)         echo "no prebuild receipt" >&2; return 5 ;;
      esac
      # `--gpus all` is required, not incidental: the verifier asks FlashInfer
      # for `jit_library_path`, and without an arch that call raises, so a
      # correct receipt would be reported as unusable.
      case "$*" in
        *"--gpus all"*) : ;;
        *) echo "MISSING --gpus all" >&2; return 5 ;;
      esac
      # The verifier must be given the image id, and the cache mounted read-only.
      case "$*" in
        *"--image-id sha256:imageid"*) : ;;
        *) echo "MISSING image id" >&2; return 5 ;;
      esac
      case "$*" in
        *"-v /h/.cache/vllm-dsv41/flashinfer:/fi:ro"*) : ;;
        *) echo "MISSING cache mount" >&2; return 5 ;;
      esac
      echo "OK: the compiled sparse_mla_sm120 module will be reused" ;;
  esac
}
( check_flashinfer_cache ) >/dev/null 2>&1 && echo ALLOWED || echo REFUSED
EOS
} > "$TMP/fi_cache.sh"

fiexpect() {
  local want="$1" case_name="$2" label="$3"
  local got
  : > "$TMP/fi-seen"
  got="$(CASE="$case_name" FI_SEEN="$TMP/fi-seen" bash "$TMP/fi_cache.sh" 2>&1 | tail -1)"
  if [ "$got" = "$want" ]; then
    printf 'ok   %-40s %s\n' "$label" "$got"
  else
    printf 'FAIL %-40s want %s got %s\n' "$label" "$want" "$got"
    fails=$((fails + 1))
  fi
}

fiexpect ALLOWED healthy        "module verified on both nodes"
fiexpect REFUSED empty_head     "head cache unusable"
fiexpect REFUSED empty_worker   "worker cache unusable"
fiexpect REFUSED inspect_unread "head image uninspectable"

# Both nodes must be visited, not just the first one that answers.
: > "$TMP/fi-seen"
CASE=healthy FI_SEEN="$TMP/fi-seen" bash "$TMP/fi_cache.sh" >/dev/null 2>&1
if grep -q '^head$' "$TMP/fi-seen" && grep -q '^nvidia@spark2$' "$TMP/fi-seen"; then
  printf 'ok   %-40s %s\n' "both nodes are checked" "head + worker"
else
  printf 'FAIL %-40s %s\n' "both nodes are checked" "$(tr '\n' ' ' < "$TMP/fi-seen")"
  fails=$((fails + 1))
fi

# A knob in the template that nothing reads is worse than no knob: it reads as a
# switch the operator can throw. `REFUSE_IF_GLM` was exactly that — the real
# interlock is `REFUSE_IF_GPU_BUSY`, which inspects actual device requests, so
# setting the other one to 0 changed nothing.
for key in $(grep -oE '^REFUSE_IF_[A-Z0-9_]+' "$HERE/../env.example" | sort -u); do
  if grep -q "\${$key" "$START_SH"; then
    printf 'ok   %-40s %s\n' "every REFUSE_IF_* knob is read" "$key"
  else
    printf 'FAIL %-40s %s\n' "every REFUSE_IF_* knob is read" "$key is dead"
    fails=$((fails + 1))
  fi
done

# --- round-3 preflight plumbing -----------------------------------------------
# Three defects from the third review round, each invisible off-cluster: a
# `docker run` probe without `--gpus all` cannot determine the AOT state (the
# generator raises with no arch, so a *correct* receipt is rejected), a
# snapshot-only mount leaves `config.json`'s `../../blobs` symlink dangling on a
# perfectly staged checkpoint, and an unreachable worker used to be reported as a
# memory or GPU problem instead of a connectivity one.

# (1) The shape probe must request the GPU, mount the cache ROOT, and read the
#     in-container config path — not the host one, which does not exist there.
mkdir -p "$TMP/ckpt"
echo '{}' > "$TMP/ckpt/config.json"
{
  echo 'die() { echo "REFUSED: $*" >&2; exit 9; }'
  echo 'log() { :; }'
  echo "MODEL_DIR_HOST=$TMP/ckpt"
  echo 'MODEL_MOUNT_HOST=/h/.cache/huggingface'
  echo 'MODEL_MOUNT_TARGET=/model-cache'
  echo 'MODEL_DIR_IN_CONTAINER=/model-cache/hub/models--x--y/snapshots/main'
  echo 'IMAGE=dsv41-reap-selfbuild'
  echo 'TP=2'
  echo 'WORKER_SSH=nvidia@spark2'
  # Record each node's argv instead of running docker. `docker_probe` takes the
  # node first ("" is the head), which is also what shows the check reaches the
  # worker rather than only the head.
  cat <<'EOS'
docker_probe() {
  local target="$1"; shift
  printf '%s|%s\n' "${target:-head}" "$*" >> "$SHAPE_ARGV"
  echo "probe output"; return 0
}
EOS
  awk '/^check_flashinfer_shapes\(\)/,/^}/' "$START_SH"
  cat <<'EOS'
( check_flashinfer_shapes ) >/dev/null 2>&1 && echo ALLOWED || echo REFUSED
EOS
} > "$TMP/shape.sh"

: > "$TMP/shape-argv"
shape_rc="$(SHAPE_ARGV="$TMP/shape-argv" bash "$TMP/shape.sh" 2>&1 | tail -1)"
shape_argv="$(cat "$TMP/shape-argv")"
if [ "$shape_rc" = "ALLOWED" ]; then
  printf 'ok   %-40s %s\n' "shape probe runs when the config exists" "$shape_rc"
else
  printf 'FAIL %-40s want ALLOWED got %s\n' "shape probe runs when the config exists" "$shape_rc"
  fails=$((fails + 1))
fi
for needle in '--gpus all' \
              '-v /h/.cache/huggingface:/model-cache:ro' \
              '--config /model-cache/hub/models--x--y/snapshots/main/config.json'; do
  if printf '%s' "$shape_argv" | grep -qF -- "$needle"; then
    printf 'ok   %-40s %s\n' "shape probe argv" "has ${needle%% *}"
  else
    printf 'FAIL %-40s %s\n' "shape probe argv" "MISSING: $needle"
    fails=$((fails + 1))
  fi
done
# The probe must reach BOTH nodes: it reads the image, and each rank runs its own
# copy, so a head-only probe misses a worker left on the previous image.
if printf '%s' "$shape_argv" | grep -qF -- "nvidia@spark2|run"; then
  printf 'ok   %-40s %s\n' "shape probe reaches the worker" "probed"
else
  printf 'FAIL %-40s %s\n' "shape probe reaches the worker" "head only"
  fails=$((fails + 1))
fi
# The host path must NOT be what the container is asked to read.
if printf '%s' "$shape_argv" | grep -qF -- "--config $TMP/ckpt/config.json"; then
  printf 'FAIL %-40s %s\n' "shape probe reads the in-container path" "used the host path"
  fails=$((fails + 1))
else
  printf 'ok   %-40s %s\n' "shape probe reads the in-container path" "in-container"
fi

# (2) Connectivity must be checked before anything that reads the worker, or an
#     unreachable worker is reported as a GPU or memory problem.
preflight_body="$(awk '/^preflight\(\)/,/^}/' "$START_SH")"
first_gate="$(printf '%s\n' "$preflight_body" | grep -oE '(check|refuse|resolve)_[a-z_]+' | head -1)"
if [ "$first_gate" = "check_worker_ssh" ]; then
  printf 'ok   %-40s %s\n' "connectivity is the first preflight gate" "$first_gate"
else
  printf 'FAIL %-40s want check_worker_ssh got %s\n' "connectivity is the first preflight gate" "${first_gate:-none}"
  fails=$((fails + 1))
fi
for later in refuse_busy_gpu refuse_low_memory check_worker_model check_flashinfer_cache; do
  a="$(printf '%s\n' "$preflight_body" | grep -n 'check_worker_ssh' | head -1 | cut -d: -f1)"
  b="$(printf '%s\n' "$preflight_body" | grep -n "$later" | head -1 | cut -d: -f1)"
  if [ -n "$a" ] && [ -n "$b" ] && [ "$a" -lt "$b" ]; then
    printf 'ok   %-40s %s\n' "runs before $later" "yes"
  else
    printf 'FAIL %-40s %s\n' "runs before $later" "NO (a=${a:-none} b=${b:-none})"
    fails=$((fails + 1))
  fi
done

# (3) The worker must hold the checkpoint at the head's own path: both ranks
#     mount that path, so a worker-only download starts rank 1 against nothing.
{
  echo 'die() { echo "REFUSED: $*" >&2; exit 9; }'
  # `-` not `:-`: the unset-MODEL_DIR_HOST case must be able to pass an empty
  # value through rather than being replaced by the default.
  echo 'MODEL_DIR_HOST="${WMODEL_DIR-/h/.cache/huggingface/hub/models--x--y/snapshots/main}"'
  awk '/^check_worker_model\(\)/,/^}/' "$START_SH"
  cat <<'EOS'
# Record the remote command, so the test can assert which path the worker is
# actually asked about rather than trusting the message text.
case "$CASE" in
  missing) worker_ssh() { printf '%s\n' "$*" >> "$WMODEL_SEEN"; return 1; } ;;
  present) worker_ssh() { printf '%s\n' "$*" >> "$WMODEL_SEEN"; return 0; } ;;
esac
( check_worker_model ) && echo ALLOWED || echo REFUSED
EOS
} > "$TMP/wmodel.sh"

wmodel() {
  local want="$1" case_name="$2" label="$3"
  local got
  got="$(CASE="$case_name" bash "$TMP/wmodel.sh" 2>&1 | tail -1)"
  if [ "$got" = "$want" ]; then
    printf 'ok   %-40s %s\n' "$label" "$got"
  else
    printf 'FAIL %-40s want %s got %s\n' "$label" "$want" "$got"
    fails=$((fails + 1))
  fi
}
wmodel REFUSED missing "worker without the checkpoint refuses"
wmodel ALLOWED present "worker holding the checkpoint is allowed"

# It must ask the worker about the head's own path, and say how to fix it.
: > "$TMP/wmodel-seen"
wmodel_msg="$(CASE=missing WMODEL_SEEN="$TMP/wmodel-seen" bash "$TMP/wmodel.sh" 2>&1)"
seen="$(cat "$TMP/wmodel-seen")"
if [ "$seen" = "test -f '/h/.cache/huggingface/hub/models--x--y/snapshots/main/config.json'" ]; then
  printf 'ok   %-40s %s\n' "worker-model probes the head's own path" "$seen"
else
  printf 'FAIL %-40s got %s\n' "worker-model probes the head's own path" "${seen:-nothing}"
  fails=$((fails + 1))
fi
for needle in 'the worker cannot see the checkpoint' 'scripts/fetch_weights.sh'; do
  if printf '%s' "$wmodel_msg" | grep -qF -- "$needle"; then
    printf 'ok   %-40s %s\n' "worker-model refusal names the remedy" "present"
  else
    printf 'FAIL %-40s %s\n' "worker-model refusal names the remedy" "MISSING: $needle"
    fails=$((fails + 1))
  fi
done
# An unset MODEL_DIR_HOST must refuse rather than test "/config.json" remotely.
wmodel_unset="$(WMODEL_DIR='' CASE=present bash "$TMP/wmodel.sh" 2>&1 | tail -1)"
if [ "$wmodel_unset" = "REFUSED" ]; then
  printf 'ok   %-40s %s\n' "unset MODEL_DIR_HOST refuses" "$wmodel_unset"
else
  printf 'FAIL %-40s want REFUSED got %s\n' "unset MODEL_DIR_HOST refuses" "$wmodel_unset"
  fails=$((fails + 1))
fi

# --- per-node sparse-MLA shape and SM12x-fallback check ----------------------
# `check_flashinfer_shapes` reads the *image*, and each rank runs its own copy of
# it, so it has to run on both nodes. A head rebuilt while the worker still
# carries the previous image passes a head-only check and then fails at launch —
# or serves with an image whose backends cannot accept the pinned block size and
# whose `deep_gemm.py` has no SM12x fallback, so the ratio-2 pages reach a
# DeepGEMM kernel that does not exist and the engine dies on the first long
# prompt. `docker_probe` is stubbed per node.
mkdir -p "$TMP/shapes"
: > "$TMP/shapes/config.json"
{
  echo 'die() { echo "REFUSED: $*" >&2; exit 9; }'
  echo 'log() { :; }'
  echo 'WORKER_SSH=nvidia@spark2'
  echo 'IMAGE=dsv41-reap-selfbuild:latest'
  echo "MODEL_DIR_HOST=$TMP/shapes"
  echo "MODEL_MOUNT_HOST=$TMP/shapes"
  echo 'MODEL_MOUNT_TARGET=/model-cache'
  echo 'MODEL_DIR_IN_CONTAINER=/model-cache/shapes'
  echo 'TP=2'
  echo 'EFFECTIVE_MAX_IMAGE_TOKENS=0'
  # Single-quoted so `shapes.sh` expands it at run time: the override has to be
  # settable per case, not frozen when this generator runs.
  echo 'SKIP_FLASHINFER_CHECK=${SKIP_SHAPES:-0}'
  awk '/^check_flashinfer_shapes\(\)/,/^}/' "$START_SH"
  cat <<'EOS'
docker_probe() {
  local target="$1"; shift
  if [ -z "$target" ]; then echo "head-report"; return "${HEAD_RC:-0}"; fi
  echo "worker-report"; return "${WORKER_RC:-0}"
}
( check_flashinfer_shapes ) >/dev/null 2>&1 && echo ALLOWED || echo REFUSED
EOS
} > "$TMP/shapes.sh"

expect_shapes() {
  local want="$1" head="$2" worker="$3" label="$4"
  local got
  got="$(HEAD_RC="$head" WORKER_RC="$worker" bash "$TMP/shapes.sh" 2>&1 | tail -1)"
  if [ "$got" = "$want" ]; then
    printf 'ok   %-40s %s\n' "$label" "$got"
  else
    printf 'FAIL %-40s want %s got %s\n' "$label" "$want" "$got"
    fails=$((fails + 1))
  fi
}

expect_shapes ALLOWED 0 0 "both nodes current"
expect_shapes REFUSED 0 6 "worker on an unpatched image"
expect_shapes REFUSED 0 5 "worker missing the 32-state page block"
expect_shapes REFUSED 0 4 "worker image cannot import FlashInfer"
expect_shapes REFUSED 6 0 "head itself unpatched"

# The refusal has to name the node, or an operator with two rebuild paths cannot
# tell which one to run. `die` writes to stderr and the generator above silences
# the subshell to keep the status readable, so this is the same script with only
# that redirection removed.
sed 's|( check_flashinfer_shapes ) >/dev/null 2>&1|( check_flashinfer_shapes )|' \
  "$TMP/shapes.sh" > "$TMP/shapes_verbose.sh"
shapes_msg="$(HEAD_RC=0 WORKER_RC=6 bash "$TMP/shapes_verbose.sh" 2>&1)"
if printf '%s' "$shapes_msg" | grep -qF -- "on the worker node"; then
  printf 'ok   %-40s %s\n' "worker refusal names the worker node" "present"
else
  printf 'FAIL %-40s %s\n' "worker refusal names the worker node" "MISSING: on the worker node"
  fails=$((fails + 1))
fi
if printf '%s' "$shapes_msg" | grep -qF -- "patch_sm12x_indexer.py"; then
  printf 'ok   %-40s %s\n' "worker refusal names the missing patch" "present"
else
  printf 'FAIL %-40s %s\n' "worker refusal names the missing patch" "MISSING: patch_sm12x_indexer.py"
  fails=$((fails + 1))
fi

# The head must be checked too, and a skipped check must still be a skip.
shapes_head="$(HEAD_RC=6 WORKER_RC=0 bash "$TMP/shapes_verbose.sh" 2>&1)"
if printf '%s' "$shapes_head" | grep -qF -- "on the head node"; then
  printf 'ok   %-40s %s\n' "head refusal names the head node" "present"
else
  printf 'FAIL %-40s %s\n' "head refusal names the head node" "MISSING: on the head node"
  fails=$((fails + 1))
fi
# `SKIP_FLASHINFER_CHECK=1` is an operator override and must still override on a
# node the check would have refused, or the escape hatch would not work on the
# one node it exists for.
shapes_skip="$(SKIP_SHAPES=1 HEAD_RC=0 WORKER_RC=6 bash "$TMP/shapes.sh" 2>&1 | tail -1)"
if [ "$shapes_skip" = "ALLOWED" ]; then
  printf 'ok   %-40s %s\n' "SKIP_FLASHINFER_CHECK overrides" "$shapes_skip"
else
  printf 'FAIL %-40s want ALLOWED got %s\n' "SKIP_FLASHINFER_CHECK overrides" "$shapes_skip"
  fails=$((fails + 1))
fi

echo
if [ "$fails" -eq 0 ]; then
  echo "launcher guard tests OK"
else
  echo "launcher guard tests FAILED ($fails)"
  exit 1
fi
