#!/usr/bin/env bash
# start.sh — 2× DGX Spark (GB10) runtime for DeepSeek-V4.1-Flash REAP-256E.
#
# Launcher for this cluster. The fabric pins are carried over from the
# production deployment on these boxes. Engram stays on NVMe. First boot is
# eager, text-only, seqs=2, SPEC=none. This script never stops another model:
# it refuses to start while one is holding the GPU.
#
#   ./start.sh validate|preflight|start|stop|status|logs|smoke
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
CMD="${1:-start}"

log() { printf '\033[1;36m[dsv41-reap]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[dsv41-reap]\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31m[dsv41-reap]\033[0m ERROR: %s\n' "$*" >&2; exit 1; }

if [ ! -f "$SCRIPT_DIR/.env" ]; then
    if [ "$CMD" = "validate" ] || [ "$CMD" = "preflight" ]; then
        die "missing .env; $CMD does not create configuration"
    fi
    [ -f "$SCRIPT_DIR/env.example" ] || die "missing env.example"
    cp "$SCRIPT_DIR/env.example" "$SCRIPT_DIR/.env"
    log "wrote .env from env.example"
fi

set -a
# shellcheck disable=SC1091
source "$SCRIPT_DIR/.env"
set +a

MODEL="${MODEL:-LibertAIDAI/DeepSeek-V4.1-Flash-REAP-256E}"
MODEL_REVISION="${MODEL_REVISION:-main}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-deepseek-v4.1-flash-reap256}"
IMAGE="${IMAGE:-dsv41-reap-selfbuild}"
PORT="${PORT:-8001}"
MASTER_PORT="${MASTER_PORT:-29531}"
TP="${TP:-2}"
NNODES="${NNODES:-2}"
HEAD_IP="${HEAD_IP:-192.168.177.10}"
WORKER_IP="${WORKER_IP:-192.168.177.11}"
HEAD_CX7_IF="${HEAD_CX7_IF:-enp1s0f1np1}"
WORKER_CX7_IF="${WORKER_CX7_IF:-enp1s0f1np1}"
HEAD_CX7_IB="${HEAD_CX7_IB:-rocep1s0f1}"
WORKER_CX7_IB="${WORKER_CX7_IB:-rocep1s0f1}"
NCCL_IB_HCA="${NCCL_IB_HCA:-rocep1s0f1}"
NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
# The configured GID index is a HINT. preflight re-resolves each rank's slot from
# the fabric (see resolve_fabric_gids) because the slot is a runtime table entry,
# not a property of the address, and it has already drifted across reboots on this
# cluster. A configured value that disagrees is reported as stale and overridden.
HEAD_GID_CONFIGURED="${HEAD_GID:-}"
WORKER_GID_CONFIGURED="${WORKER_GID:-}"
HEAD_GID="${HEAD_GID:-$NCCL_IB_GID_INDEX}"
WORKER_GID="${WORKER_GID:-$NCCL_IB_GID_INDEX}"
NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
NCCL_CROSS_NIC="${NCCL_CROSS_NIC:-0}"
WORKER_USER="${WORKER_USER:-nvidia}"
WORKER_HOME="${WORKER_HOME:-/home/${WORKER_USER}}"
WORKER_SSH="${WORKER_SSH:-${WORKER_USER}@${WORKER_IP}}"
# Default to the shape the receipt was measured at, not the checkpoint's 1M
# maximum: 64K is what "1,437,437 KV tokens" describes, and the 1M default is a
# silent 16x pressure increase on a 2x121 GiB pool. env.example pins 65536 too;
# this default only matters for a hand-edited .env.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
# FIRST_BOOT=1 holds the conservative shape (seqs=2, eager, language-only,
# SPEC=none). FIRST_BOOT=0 ramps those four deliberately after a successful boot;
# the memory, GPU-exclusivity, quantization and FlashInfer gates stay on in both
# modes. See refuse_first_boot_shape.
FIRST_BOOT="${FIRST_BOOT:-1}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
LONG_PREFILL_TOKEN_THRESHOLD="${LONG_PREFILL_TOKEN_THRESHOLD:-4096}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.82}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-4294967296}"
# Default to no --quantization override: the REAP-256E checkpoint is natively
# MXFP4/FP8 and declares it, so the engine must read that rather than be told
# `exl3`. Setting QUANTIZATION=exl3 is only correct for an EXL3-repacked pack,
# and `check_quantization_matches` refuses the mismatch.
QUANTIZATION="${QUANTIZATION:-}"
LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-1}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
# Text-only serving must also collapse the per-image token budget, or the
# checkpoint's image-span width (sliding_window + max_image_tokens = 1152) has no
# SM120 sparse-MLA kernel. See the `--hf-overrides` note in `vllm_argv`.
TEXT_ONLY_VISION_OVERRIDE="${TEXT_ONLY_VISION_OVERRIDE:-1}"
# Filled by `resolve_effective_image_tokens` during preflight; 0 here keeps
# `set -u` happy for the paths that never reach it.
EFFECTIVE_MAX_IMAGE_TOKENS="${EFFECTIVE_MAX_IMAGE_TOKENS:-0}"
SPEC_METHOD="${SPEC_METHOD:-none}"
# DSpark drafts a *block* of `dspark_block_size` tokens in one parallel pass, so
# `num_speculative_tokens` is quantised to that block: vLLM sets
# `n_predict = dspark_block_size` for deepseek_v41 and raises unless the value is
# at most the block or a multiple of it (`config/speculative.py:1436`,
# `:1478-1485`). This checkpoint's block is 5, so 5 is the smallest value that
# spends a whole block, and the previous default of 2 — while legal, because the
# divisibility rule only fires above the block — leaves 3 of every 5 draft slots
# unused.
SPEC_NUM_TOKENS="${SPEC_NUM_TOKENS:-5}"
# The block size the divisibility rule above is built on, and the value
# `check_dspark_draft` verifies the checkpoint actually declares.
SPEC_DSPARK_BLOCK="${SPEC_DSPARK_BLOCK:-5}"
# DSpark's own default is `greedy` (`config/speculative.py:588`). `probabilistic`
# is the higher-acceptance option, but upstream recorded it degrading into
# junk/loops with *inflated* acceptance rates until two bugfixes landed
# (vllm#46995) — a failure mode no throughput number can see, so it is opt-in and
# must be judged by a long-generation coherence check, not by tok/s.
SPEC_DRAFT_SAMPLE_METHOD="${SPEC_DRAFT_SAMPLE_METHOD:-greedy}"
# Adaptive verification scores every draft slot by survival probability and
# admits the best ones under a budget profiled at startup, so one setting holds
# across concurrency instead of one tuned `num_speculative_tokens`. DSpark-only
# (`config/speculative.py:1564` rejects it for any other method).
#
# **Refused, not merely defaulted off.** It requires full CUDA-graph support, and
# CUDA graphs cannot be used with this checkpoint: they produce NaN here, and the
# vendor's own card records an illegal memory access in both PIECEWISE and
# FULL_AND_PIECEWISE modes with a reduced capture set not helping. So the engine
# can never honour `1`, and `spec_flags` fails closed on it rather than emitting a
# setting that would look applied while doing nothing.
SPEC_ADAPTIVE_VERIFICATION="${SPEC_ADAPTIVE_VERIFICATION:-0}"
TOKENIZER_MODE="${TOKENIZER_MODE:-deepseek_v41}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-deepseek_v41}"
REASONING_PARSER="${REASONING_PARSER:-deepseek_v41}"
# vLLM only activates the tool parser behind this flag, and silently does nothing
# when it is absent.
ENABLE_AUTO_TOOL_CHOICE="${ENABLE_AUTO_TOOL_CHOICE:-1}"
# Async scheduling is a tri-state: omitting the flag resolves to ENABLED for a
# generative model on the mp executor. Default to off, explicitly.
NO_ASYNC_SCHEDULING="${NO_ASYNC_SCHEDULING:-1}"
DISABLE_KERNEL_WARMUPS="${DISABLE_KERNEL_WARMUPS:-1}"
UCX_MEM_MMAP_HOOK_MODE="${UCX_MEM_MMAP_HOOK_MODE:-none}"
UCX_RCACHE_MAX_UNRELEASED="${UCX_RCACHE_MAX_UNRELEASED:-1024}"
OOM_SCORE_ADJ="${OOM_SCORE_ADJ:-500}"
# Give vLLM time to tear NCCL/UCX down on `docker rm -f` and on unit restart.
# The default 10 s truncates the teardown of a 2-rank mp group.
STOP_TIMEOUT="${STOP_TIMEOUT:-60}"
# The checkpoint is a read-only local mount and the tokenizer ships with it, so
# a Hub round trip is pure latency and a new failure mode on an air-gapped box.
# The production deployment on these nodes runs offline; so does this one.
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
# The image sets TORCH_CUDA_ARCH_LIST=12.1a, which does NOT cover the CuTe-DSL
# and TileLang JIT paths. The 2-node recipe sets CUTE_DSL_ARCH explicitly and
# puts the TileLang cache on the persistent bind; a wrong arch there is a
# runtime JIT failure, not a build error.
CUTE_DSL_ARCH="${CUTE_DSL_ARCH:-sm_121a}"
# Runtime JIT fan-out competes with the page cache for the same unified pool; a
# 22-job FlashInfer GEMM build reset every node of the 4-Spark V4.1 recipe. The
# image already pins 2; re-assert it so a container env cannot widen it.
MAX_JOBS="${MAX_JOBS:-2}"
DSV41_ENGRAM_DISK="${DSV41_ENGRAM_DISK:-1}"
READY_TIMEOUT="${READY_TIMEOUT:-4800}"
VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1800}"
HEAD_NAME="${HEAD_NAME:-dsv41-reap-head}"
WORKER_NAME="${WORKER_NAME:-dsv41-reap-worker}"
HF_HOME_HOST="${HF_HOME_HOST:-$HOME/.cache/huggingface}"
MODEL_CACHE="${HF_HOME_HOST}/hub/models--${MODEL//\//--}"
if [ -z "${MODEL_DIR_HOST:-}" ]; then
    _ref="${MODEL_CACHE}/refs/${MODEL_REVISION}"
    if [ -f "$_ref" ]; then
        MODEL_DIR_HOST="${MODEL_CACHE}/snapshots/$(tr -d '[:space:]' < "$_ref")"
    else
        MODEL_DIR_HOST="${MODEL_CACHE}/snapshots/${MODEL_REVISION}"
    fi
fi
CACHE_HOST="${CACHE_HOST:-$HOME/.cache/vllm-dsv41}"
# FlashInfer's JIT cache. The image masks the prebuilt AOT binary for
# `sparse_mla_sm120` so the patched .cu is used (RESEARCH 11.9.3), which means
# the module must be compiled once per node by `scripts/build_flashinfer.sh`
# into this directory. It is mounted at /root/.cache/flashinfer, FlashInfer's
# default cache root, so no env override is needed.
FLASHINFER_CACHE_HOST="${FLASHINFER_CACHE_HOST:-$CACHE_HOST/flashinfer}"

# HF snapshots are symlink farms: `snapshots/<rev>/config.json` points at
# `../../blobs/<sha>`. Mounting only the snapshot dir leaves those links
# dangling inside the container, so mount the `models--...` cache root and
# point the server at the in-container snapshot path instead.
if [ "${MODEL_DIR_HOST#"$MODEL_CACHE"/}" != "$MODEL_DIR_HOST" ]; then
    MODEL_MOUNT_HOST="$MODEL_CACHE"
    MODEL_MOUNT_TARGET=/model-cache
    MODEL_DIR_IN_CONTAINER="/model-cache/${MODEL_DIR_HOST#"$MODEL_CACHE"/}"
else
    MODEL_MOUNT_HOST="$MODEL_DIR_HOST"
    MODEL_MOUNT_TARGET=/model
    MODEL_DIR_IN_CONTAINER=/model
fi

worker_ssh() { ssh -o BatchMode=yes "$WORKER_SSH" "$@"; }

# `ssh host cmd args...` does not forward argv: OpenSSH joins the arguments
# with spaces and the remote login shell re-parses the result, so quotes the
# local shell already consumed are gone. A Go template containing a space
# (`{{json .HostConfig.DeviceRequests}}`) therefore splits into two words and
# `docker inspect -f` fails, which made every worker probe look like a docker
# outage. Serialize the whole command and let the remote shell unquote it.
docker_probe() {
    local target="$1"
    shift
    if [ -z "$target" ]; then
        docker "$@"
    else
        ssh -o BatchMode=yes "$target" "$(printf '%q ' docker "$@")"
    fi
}

# Names of running containers that actually requested GPUs (fail closed).
gpu_container_names() {
    local target="${1:-}"
    local ids dr name
    ids="$(docker_probe "$target" ps -q 2>/dev/null)" || return 1
    for id in $ids; do
        dr="$(docker_probe "$target" inspect -f '{{json .HostConfig.DeviceRequests}}' "$id" 2>/dev/null)" || return 1
        case "$dr" in ""|null|"[]"|"<no value>") continue ;; esac
        name="$(docker_probe "$target" inspect -f '{{.Name}}' "$id" 2>/dev/null)" || return 1
        printf '%s\n' "${name#/}"
    done
}

# ------------------------- fabric GID resolution ----------------------------
# `NCCL_IB_GID_INDEX` is a runtime table SLOT on one NIC, not a property of the
# address. The usable entry is the one whose GID is *this node's own* fabric
# IPv4 with type "RoCE v2". A populated-but-wrong slot passes a naive
# "is it all-zero" check and then kills that rank ~60 s in with
# `ibv_modify_qp errno 61` ("No data available"). The community 2-node recipe
# pins 5, which on these boxes is an empty slot on `rocep1s0f1`; the existing
# deployment on this same fabric resolves it per node because it has already
# drifted across reboots. Resolve rather than trust the pin, and fail closed on
# ambiguity.

# The IPv4-mapped IPv6 form the kernel writes into `.../gids/<idx>`.
ipv4_mapped_gid() {
    local ip="$1" a b c d hex
    IFS=. read -r a b c d <<<"$ip"
    case "${a:-x}${b:-x}${c:-x}${d:-x}" in *[!0-9]*) return 1 ;; esac
    [ -n "$a" ] && [ -n "$b" ] && [ -n "$c" ] && [ -n "$d" ] || return 1
    hex="$(printf '%02x%02x%02x%02x' "$a" "$b" "$c" "$d")" || return 1
    [ "${#hex}" -eq 8 ] || return 1
    printf '0000:0000:0000:0000:0000:ffff:%s:%s' "${hex:0:4}" "${hex:4:4}"
}

# "index<TAB>gid<TAB>type" for every populated GID on the local port.
gid_table() {
    local ib_dev="$1" idx gid type
    for idx in $(seq 0 15); do
        gid="$(cat "/sys/class/infiniband/${ib_dev}/ports/1/gids/${idx}" 2>/dev/null)" || continue
        [ -n "$gid" ] || continue
        [ "$gid" != "0000:0000:0000:0000:0000:0000:0000:0000" ] || continue
        type="$(cat "/sys/class/infiniband/${ib_dev}/ports/1/gid_attrs/types/${idx}" 2>/dev/null)" || continue
        printf '%s\t%s\t%s\n' "$idx" "$gid" "$type"
    done
}

# The same table from the worker, in one ssh round trip. Matching happens
# locally so the selection logic lives in exactly one place.
worker_gid_table() {
    local ib_dev="$1"
    worker_ssh "for i in \$(seq 0 15); do g=\$(cat /sys/class/infiniband/${ib_dev}/ports/1/gids/\$i 2>/dev/null); [ -n \"\$g\" ] || continue; [ \"\$g\" = 0000:0000:0000:0000:0000:0000:0000:0000 ] && continue; t=\$(cat /sys/class/infiniband/${ib_dev}/ports/1/gid_attrs/types/\$i 2>/dev/null); printf '%s\t%s\t%s\n' \"\$i\" \"\$g\" \"\$t\"; done"
}

# Echo the single RoCE v2 index whose GID is $ip; on failure echo the reason and
# return 1. Exactly one match is required: two slots for the same address (the
# v1/v2 pair, or a dual-rail NIC) means the choice is not ours to guess.
gid_index_for_ip() {
    local table="$1" ip="$2" want matches count
    want="$(ipv4_mapped_gid "$ip")" || { printf 'cannot parse fabric IP %s' "$ip"; return 1; }
    matches="$(printf '%s\n' "$table" | awk -F'\t' -v w="$want" '$2 == w && $3 ~ /RoCE v2/ { print $1 }')"
    if [ -z "$matches" ]; then
        printf 'no RoCE v2 GID matches %s (%s)' "$ip" "$want"
        return 1
    fi
    count="$(printf '%s\n' "$matches" | wc -l | tr -d ' ')"
    if [ "$count" -ne 1 ]; then
        printf 'ambiguous: %s RoCE v2 GIDs match %s: %s' \
            "$count" "$ip" "$(printf '%s' "$matches" | tr '\n' ' ')"
        return 1
    fi
    printf '%s' "$matches"
}

dump_gid_tables() {
    warn "head ${HEAD_CX7_IB} (want ${HEAD_IP}):"
    printf '%s\n' "$1" | sed 's/^/    head   gid/' >&2
    warn "worker ${WORKER_CX7_IB} (want ${WORKER_IP}):"
    printf '%s\n' "$2" | sed 's/^/    worker gid/' >&2
}

# Resolve HEAD_GID / WORKER_GID from the fabric, then say so if the configured
# hint was stale. Each rank gets its own index, so a drift on one node does not
# have to be papered over by editing .env for the other.
resolve_fabric_gids() {
    local head_table worker_table head_idx worker_idx
    head_table="$(gid_table "$HEAD_CX7_IB")"
    if [ -z "$head_table" ]; then
        die "no populated GIDs on $HEAD_CX7_IB (head); is the fabric up?"
    fi
    worker_table="$(worker_gid_table "$WORKER_CX7_IB")"
    if [ -z "$worker_table" ]; then
        dump_gid_tables "$head_table" "$worker_table"
        die "no populated GIDs on $WORKER_CX7_IB (worker); is the fabric up?"
    fi
    if ! head_idx="$(gid_index_for_ip "$head_table" "$HEAD_IP")"; then
        dump_gid_tables "$head_table" "$worker_table"
        die "head: $head_idx"
    fi
    if ! worker_idx="$(gid_index_for_ip "$worker_table" "$WORKER_IP")"; then
        dump_gid_tables "$head_table" "$worker_table"
        die "worker: $worker_idx"
    fi
    if [ -n "$HEAD_GID_CONFIGURED" ] && [ "$HEAD_GID_CONFIGURED" != "$head_idx" ]; then
        warn "HEAD_GID=$HEAD_GID_CONFIGURED is stale; the head's RoCE v2 slot for $HEAD_IP is $head_idx"
    fi
    if [ -n "$WORKER_GID_CONFIGURED" ] && [ "$WORKER_GID_CONFIGURED" != "$worker_idx" ]; then
        warn "WORKER_GID=$WORKER_GID_CONFIGURED is stale; the worker's RoCE v2 slot for $WORKER_IP is $worker_idx"
    fi
    HEAD_GID="$head_idx"
    WORKER_GID="$worker_idx"
    log "fabric GID OK head=$HEAD_CX7_IB/$HEAD_GID worker=$WORKER_CX7_IB/$WORKER_GID (RoCE v2, own IPv4)"
}

refuse_busy_gpu() {
    [ "${REFUSE_IF_GPU_BUSY:-1}" = "1" ] || return 0
    local head_gpu worker_gpu others
    head_gpu="$(gpu_container_names)" \
        || die "cannot inspect local containers (docker down?) — refusing to start"
    others="$(printf '%s\n' "$head_gpu" | grep -vxE "${HEAD_NAME}|${WORKER_NAME}" | grep -v '^$' || true)"
    if [ -n "$others" ]; then
        die "another GPU container is running on the head:
$others
This kit does not stop it. Stop it yourself if you have a window."
    fi
    worker_gpu="$(gpu_container_names "$WORKER_SSH")" \
        || die "cannot inspect worker containers (ssh/docker down?) — refusing to start"
    others="$(printf '%s\n' "$worker_gpu" | grep -vxE "${WORKER_NAME}" | grep -v '^$' || true)"
    if [ -n "$others" ]; then
        die "another GPU container is running on the worker:
$others
This kit does not stop it."
    fi
}

refuse_engram_pin() {
    [ "$DSV41_ENGRAM_DISK" = "1" ] || die "DSV41_ENGRAM_DISK must be 1. Pinned Engram is the GB10 UMA trap."
}

# The conservative serving shape is a first-boot posture, not a permanent one:
# seqs=2 / eager / language-only / SPEC=none are what makes the first multi-minute
# weight load predictable. `FIRST_BOOT=0` lifts exactly those four so the shape can
# be ramped once a boot has succeeded. The memory floor, the GPU-exclusivity
# interlock, the quantization check and the FlashInfer gate are unconditional.
refuse_first_boot_shape() {
    if [ "$FIRST_BOOT" != "1" ]; then
        [ "$FIRST_BOOT" = "0" ] || die "FIRST_BOOT must be 0 or 1, got '$FIRST_BOOT'."
        [ "$MAX_NUM_SEQS" -ge 2 ] || die "MAX_NUM_SEQS=$MAX_NUM_SEQS with FIRST_BOOT=0; the receipt shape is 2."
        log "WARNING: FIRST_BOOT=0 — the first-boot shape guards are lifted (seqs=$MAX_NUM_SEQS spec=$SPEC_METHOD eager=$ENFORCE_EAGER language_only=$LANGUAGE_MODEL_ONLY)"
        return 0
    fi
    [ "$MAX_NUM_SEQS" -le 2 ] || die "MAX_NUM_SEQS=$MAX_NUM_SEQS; first boot is seqs=2 (NVMe Engram). Set FIRST_BOOT=0 to ramp it deliberately."
    [ "$SPEC_METHOD" = "none" ] || die "SPEC_METHOD=$SPEC_METHOD; first boot is SPEC=none."
    [ "$ENFORCE_EAGER" = "1" ] || die "ENFORCE_EAGER must be 1 on first boot."
    [ "$LANGUAGE_MODEL_ONLY" = "1" ] || die "LANGUAGE_MODEL_ONLY must be 1 on first boot."
}

# The text-only override is only coherent while the engine also refuses images.
# With images enabled, collapsing the per-image budget would shrink the SWA index
# rows out from under a real image span, which is a correctness bug rather than
# a startup failure — so refuse the combination instead of guessing.
refuse_incoherent_vision_config() {
    case "$TEXT_ONLY_VISION_OVERRIDE" in
        0|1) ;;
        *) die "TEXT_ONLY_VISION_OVERRIDE must be 0 or 1, got '$TEXT_ONLY_VISION_OVERRIDE'." ;;
    esac
    if [ "$TEXT_ONLY_VISION_OVERRIDE" = "1" ] && [ "$LANGUAGE_MODEL_ONLY" != "1" ]; then
        die "TEXT_ONLY_VISION_OVERRIDE=1 with LANGUAGE_MODEL_ONLY=$LANGUAGE_MODEL_ONLY: the override zeroes the per-image token budget, which would narrow the SWA index rows while the engine still accepts images. Set LANGUAGE_MODEL_ONLY=1, or TEXT_ONLY_VISION_OVERRIDE=0 and accept that the SM120 sparse-MLA backend has no kernel for the resulting 1152-wide SWA segment."
    fi
    if [ "$TEXT_ONLY_VISION_OVERRIDE" = "1" ]; then
        log "text-only: zeroing the per-image token budget so the SWA index rows stay at sliding_window"
    fi
}

# The SWA index rows are `sliding_window + max_image_tokens` wide, and that width
# is what the sparse-MLA dispatchers must instantiate. The shape verifier has to
# be told the *effective* budget — zero when the override is on, the config's own
# value when it is off — or it would keep checking the checkpoint's declared
# 1152-wide table while the engine actually runs 128, and pass a boot that cannot
# happen.
resolve_effective_image_tokens() {
    if [ "$TEXT_ONLY_VISION_OVERRIDE" = "1" ]; then
        EFFECTIVE_MAX_IMAGE_TOKENS=0
        return 0
    fi
    local cfg="$MODEL_DIR_HOST/config.json"
    [ -f "$cfg" ] || die "no config.json at $cfg; cannot determine the effective image-token budget"
    EFFECTIVE_MAX_IMAGE_TOKENS="$(python3 - "$cfg" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
vision = cfg.get("vision_config") or {}
layers = vision.get("num_hidden_layers", 0) if isinstance(vision, dict) else 0
# A text-only checkpoint carries no image span, so the budget is 0 whatever the
# key says; the model code gates on the layer count the same way.
if not isinstance(layers, int) or layers <= 0:
    print(0)
else:
    budget = vision.get("max_image_tokens", 1024)
    print(budget if isinstance(budget, int) and budget > 0 else 0)
PY
)" || die "cannot parse $cfg"
    case "$EFFECTIVE_MAX_IMAGE_TOKENS" in
        ''|*[!0-9]*) die "could not read the image-token budget from $cfg (got '$EFFECTIVE_MAX_IMAGE_TOKENS')" ;;
    esac
    log "image-token budget: $EFFECTIVE_MAX_IMAGE_TOKENS (from the checkpoint; the SWA index rows will be sliding_window + that)"
}

# `MemAvailable` in GiB on a node ("" if unreadable).
mem_available_gib() {
    local target="${1:-}" raw
    if [ -z "$target" ]; then
        raw="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo 2>/dev/null)"
    else
        raw="$(ssh -o BatchMode=yes "$target" "awk '/^MemAvailable:/ {print \$2}' /proc/meminfo" 2>/dev/null)"
    fi
    case "$raw" in ''|*[!0-9]*) return 1 ;; esac
    printf '%s' $((raw / 1024 / 1024))
}

# On GB10 the GPU allocator and the page cache draw on one pool, and the box
# hard-resets when MemAvailable approaches zero — there is no OOM kill to
# catch. Refuse to start below MIN_FREE_GIB on *both* nodes, and fail closed if
# either reading is unavailable.
refuse_low_memory() {
    local want="${MIN_FREE_GIB:-90}" head_gib worker_gib
    head_gib="$(mem_available_gib)" \
        || die "cannot read MemAvailable on the head — refusing to start"
    [ "$head_gib" -ge "$want" ] \
        || die "head has ${head_gib} GiB MemAvailable, need ${want}. On GB10 that is a hard-reset risk, not a slowdown."
    worker_gib="$(mem_available_gib "$WORKER_SSH")" \
        || die "cannot read MemAvailable on the worker — refusing to start"
    [ "$worker_gib" -ge "$want" ] \
        || die "worker has ${worker_gib} GiB MemAvailable, need ${want}."
    log "memory OK head=${head_gib}GiB worker=${worker_gib}GiB floor=${want}GiB"
}

# Reaching the worker is the first thing that can fail, and it fails with ssh's
# own message ("Could not resolve hostname") rather than a named blocker. That is
# not a hypothetical: the shipped template set `WORKER_SSH=nvidia@spark2`, which
# resolves from a laptop's ssh_config but not from spark1, where this runs — the
# nodes reach each other only over the QSFP rail. Report it as a configuration
# problem, with the remedy, before any other worker check can mislabel it.
check_worker_ssh() {
    worker_ssh true 2>/dev/null \
        || die "cannot ssh (key-based) to '${WORKER_SSH}' — set up passwordless ssh for it, or point WORKER_SSH at an address that resolves ON THE HEAD. A bare hostname like 'spark2' does not resolve between the nodes on this cluster; WORKER_IP (${WORKER_IP}) does. Unset WORKER_SSH to derive ${WORKER_USER}@${WORKER_IP}."
    worker_ssh "command -v docker >/dev/null" \
        || die "docker missing on worker ${WORKER_SSH}"
}

check_fabric_addresses() {
    command -v docker >/dev/null || die "docker missing on head"
    ip -4 addr show "$HEAD_CX7_IF" | grep -q "$HEAD_IP" \
        || die "$HEAD_CX7_IF does not own $HEAD_IP"
    worker_ssh "ip -4 addr show '$WORKER_CX7_IF' | grep -q '$WORKER_IP'" \
        || die "worker $WORKER_CX7_IF does not own $WORKER_IP"
}

# Both ranks mount the HEAD's absolute path (`docker_common` uses
# `$MODEL_MOUNT_HOST` for rank 0 and rank 1 alike), so a checkpoint staged only
# on the head starts rank 1 against an empty directory. The failure surfaces
# minutes into the load as a missing shard, so check it here instead.
check_worker_model() {
    [ -n "${MODEL_DIR_HOST:-}" ] || die "MODEL_DIR_HOST is unset; cannot locate the checkpoint"
    worker_ssh "test -f '$MODEL_DIR_HOST/config.json'" \
        || die "the worker cannot see the checkpoint at $MODEL_DIR_HOST (no config.json there). Both ranks mount the head's path, so the worker needs the same tree at the same path — copy it over the rail rather than downloading it twice; scripts/fetch_weights.sh prints the exact command."
}

preflight() {
    # Connectivity FIRST: every other gate below reads the worker, and each one
    # would otherwise report its own symptom (a container-inspection failure, a
    # short-memory reading) instead of the actual cause — an unreachable worker.
    check_worker_ssh
    refuse_busy_gpu
    refuse_engram_pin
    refuse_first_boot_shape
    refuse_incoherent_vision_config
    refuse_low_memory
    check_fabric_addresses
    check_worker_model
    resolve_fabric_gids
    check_kv_cache_budget
    check_quantization_matches
    resolve_effective_image_tokens
    check_flashinfer_cache
    check_flashinfer_shapes
    check_spec_method
    check_dspark_draft
    log "preflight OK fabric $HEAD_IP <-> $WORKER_IP disk=$DSV41_ENGRAM_DISK seqs=$MAX_NUM_SEQS eager=$ENFORCE_EAGER max_image_tokens=$EFFECTIVE_MAX_IMAGE_TOKENS"
}

# `spec_flags` is what builds the argv, and it fails closed on a method the engine
# would reject. Run it here too, so an unusable SPEC_METHOD is reported before any
# container starts instead of after a multi-minute weight load. The output is
# discarded; only the exit status and the warning matter.
check_spec_method() {
    SPEC_EXTRA=()
    spec_flags "${SPEC_METHOD:-none}"
    SPEC_EXTRA=()
}

# The DSpark drafter ships *inside* the target checkpoint under the `mtp.*` tensor
# prefix, and its configuration lives in `text_config` — the top-level
# `config.json` carries no `dspark_*` key at all. A check that read only the top
# level would conclude the drafter is absent and refuse a perfectly servable arm;
# a check that skipped the question entirely would emit `--speculative-config`
# for a pack that cannot honour it and fail minutes into the load. So read the
# nested block, confirm the block size the divisibility rule is built on, and
# confirm the weights are actually in the index.
check_dspark_draft() {
    [ "${SPEC_METHOD:-none}" = "dspark" ] || return 0
    local cfg="$MODEL_DIR_HOST/config.json"
    [ -f "$cfg" ] || die "no config.json at $cfg; cannot verify the DSpark draft"
    local idx="$MODEL_DIR_HOST/model.safetensors.index.json"
    [ -f "$idx" ] || die "no model.safetensors.index.json at $idx; cannot verify the DSpark draft weights"
    local block n_mtp
    block="$(python3 - "$cfg" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
text = cfg.get("text_config") or {}
print(text.get("dspark_block_size", ""))
PY
)" || die "cannot parse $cfg"
    [ -n "$block" ] || die "SPEC_METHOD=dspark but $cfg declares no text_config.dspark_block_size, so this checkpoint ships no DSpark drafter. Serve it with SPEC_METHOD=none, or point MODEL at a pack that carries the drafter. (Look in text_config: the top-level config.json has no dspark_* key even when the drafter is present.)"
    [ "$block" = "$SPEC_DSPARK_BLOCK" ] || die "SPEC_DSPARK_BLOCK=$SPEC_DSPARK_BLOCK but $cfg declares text_config.dspark_block_size=$block. Set SPEC_DSPARK_BLOCK=$block so the SPEC_NUM_TOKENS divisibility rule matches the checkpoint."
    n_mtp="$(python3 - "$idx" <<'PY'
import json, sys
wm = json.load(open(sys.argv[1]))["weight_map"]
print(sum(1 for k in wm if k.startswith("mtp.")))
PY
)" || die "cannot parse $idx"
    [ "$n_mtp" -gt 0 ] || die "SPEC_METHOD=dspark but $idx lists no 'mtp.*' tensors. The config declares a drafter the weights do not carry (a pruned or partially staged pack); serving it would fail in the loader. Re-stage the checkpoint or serve with SPEC_METHOD=none."
    log "DSpark draft OK: block=$block mtp_tensors=$n_mtp spec_tokens=$SPEC_NUM_TOKENS sample=$SPEC_DRAFT_SAMPLE_METHOD adaptive=$SPEC_ADAPTIVE_VERIFICATION"
}

# The checkpoint declares its own quantization. Forcing `--quantization exl3`
# onto a natively MXFP4/FP8 pack (REAP-256E) makes the loader look for EXL3
# trellis tensors that do not exist; leaving it unset on an EXL3 pack makes the
# loader try to read the trellis as plain weights. Either way the failure is
# late and confusing, so compare the two up front.
check_quantization_matches() {
    local cfg="$MODEL_DIR_HOST/config.json"
    [ -f "$cfg" ] || die "no config.json at $cfg; cannot verify the quantization setting"
    local declared
    declared="$(python3 - "$cfg" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
q = cfg.get("quantization_config") or {}
print(q.get("quant_method", "") if isinstance(q, dict) else "")
PY
)" || die "cannot parse $cfg"
    if [ -z "$QUANTIZATION" ]; then
        log "quantization: engine reads the checkpoint's own declaration (${declared:-none})"
        return 0
    fi
    if [ "$QUANTIZATION" != "$declared" ]; then
        die "QUANTIZATION=$QUANTIZATION but $cfg declares '${declared:-none}'. \
Set QUANTIZATION= (empty) to let the engine read the checkpoint, or point MODEL at a pack \
that really is $QUANTIZATION."
    fi
    log "quantization OK: $QUANTIZATION matches the checkpoint"
}

# `--kv-cache-memory-bytes` and `--gpu-memory-utilization` are not alternatives,
# and they are not interchangeable either. `gpu_worker.request_memory()` always
# applies the utilization cap (`v1/worker/utils.py:534-548`, called at
# `v1/worker/gpu_worker.py:449`), but when the byte count is set
# `determine_available_memory()` returns it *before* any profiling run and says
# so ("skipped memory profiling. This does not respect the gpu_memory_utilization
# config", `gpu_worker.py:539-561`). So the utilization number keeps gating the
# total, while the byte count fixes the KV share and disables the fit that would
# otherwise derive it. Print which one is in control: a reader who assumes the
# utilization number sizes the KV pool will misread every capacity number that
# follows.
check_kv_cache_budget() {
    local bytes="${KV_CACHE_MEMORY_BYTES:-}"
    if [ -z "$bytes" ]; then
        log "KV budget: auto-fit from --gpu-memory-utilization $GPU_MEM_UTIL (profiled)"
        return 0
    fi
    case "$bytes" in ''|*[!0-9]*) die "KV_CACHE_MEMORY_BYTES must be a positive integer or empty, got '$bytes'" ;; esac
    [ "$bytes" -gt 0 ] || die "KV_CACHE_MEMORY_BYTES must be > 0 or empty, got $bytes"
    warn "KV budget: pinned to $bytes bytes ($((bytes / 1048576)) MiB) per rank. This SKIPS the profile-based fit, so --gpu-memory-utilization $GPU_MEM_UTIL no longer sizes the KV pool (it still caps total memory). Empty the variable to auto-fit."
}

# The image masks FlashInfer's prebuilt AOT binary so the patched `.cu` is
# actually compiled, which makes the compiled module a *required* artifact
# rather than an optional cache. Without it the engine runs nvcc at startup,
# competing with the server for the same unified memory pool — the failure the
# offline prebuild exists to avoid.
#
# A matching filename is not evidence: a module built for another FlashInfer
# version or arch is never looked up, a cache left from an earlier image is
# stale, and an empty file matches any glob. So the receipt `build_flashinfer.sh`
# writes beside the module is verified inside the image — image identity, the
# runtime cache key, and the artifact's size and digest — on both nodes.

# Run the in-image verifier on one node. `target` empty = head.
#
# `--gpus all` is REQUIRED, not incidental: the verifier asks FlashInfer for
# `jit_library_path`, and `gen_sparse_mla_sm120_module()` derives the cache key
# from `CompilationContext().TARGET_CUDA_ARCHS`. With no GPU it raises
# "No supported CUDA architectures found for major versions [12]", the key
# cannot be determined, and a perfectly good receipt is rejected. Verified on
# the cluster: the same image reports `<version>/cached_ops` without a GPU and
# `<version>/121a/cached_ops` with one.
fi_cache_verdict() {
    local target="$1" label="$2" image_id
    image_id="$(docker_probe "$target" image inspect -f '{{.Id}}' "$IMAGE" 2>/dev/null)" \
        || { printf '%s\n' "cannot inspect image '$IMAGE' on the $label node"; return 1; }
    [ -n "$image_id" ] \
        || { printf '%s\n' "image '$IMAGE' is not present on the $label node"; return 1; }
    docker_probe "$target" run --rm --gpus all --entrypoint python3 \
        -v "$FLASHINFER_CACHE_HOST:/fi:ro" \
        "$IMAGE" /opt/dsv41/overlay/check_flashinfer_cache.py \
        --cache /fi --image-id "$image_id" 2>&1
}

check_flashinfer_cache() {
    local out rc target label
    for label in head worker; do
        if [ "$label" = head ]; then target=""; else target="$WORKER_SSH"; fi
        rc=0
        out="$(fi_cache_verdict "$target" "$label")" || rc=$?
        [ -z "$out" ] || printf '%s\n' "$out"
        [ "$rc" = 0 ] || die "the compiled FlashInfer module is not usable on the $label node. Run scripts/build_flashinfer.sh on that node in a deploy window (RESEARCH.md 11.9.3)."
    done
}

# The compressed-KV page-block constraint is fixed in the image by
# overlay/patch_flashinfer_prefill_pbsx.py, which adds the 32-state compressed
# page V4.1's ratio-2 layers need at `--block-size 64`. Stock FlashInfer
# instantiates only 64 and 2 (the 2 is V4's ratio-128 page). This runs before
# either container starts so a missing or half-applied patch is a clear
# pre-flight message instead of an engine traceback after a multi-minute weight
# load.
#
# Exit codes: 0 ok, 3 missing decode key, 4 FlashInfer unreadable, 5 page-block
# mismatch (patch absent, half-applied, or shadowed by the AOT binary), 6 the
# manager block size lands a compressed page on a `block_kv` DeepGEMM has no
# template for (vLLM #53635).
# SKIP_FLASHINFER_CHECK=1 overrides, and says so loudly.
check_flashinfer_shapes() {
    if [ "${SKIP_FLASHINFER_CHECK:-0}" = "1" ]; then
        log "WARNING: SKIP_FLASHINFER_CHECK=1 — not verifying FlashInfer sparse-MLA shapes"
        return 0
    fi
    [ -n "${MODEL_DIR_HOST:-}" ] || die "MODEL_DIR_HOST is unset; cannot locate the checkpoint config"
    local cfg="$MODEL_DIR_HOST/config.json"
    [ -f "$cfg" ] || die "no config.json at $cfg; cannot verify FlashInfer shapes"
    local out rc target label
    # Mount the HF cache ROOT, not the snapshot dir, for the same reason the
    # server does: `snapshots/<rev>/config.json` is a symlink to
    # `../../blobs/<sha>`, so a snapshot-only bind leaves it dangling and the
    # checker reports a missing config on a perfectly staged checkpoint. The
    # path it is asked to read is the in-container one, not the host one.
    #
    # `--gpus all` for the same reason as the cache check above: the AOT state
    # is read through `gen_sparse_mla_sm120_module()`, which needs an arch.
    #
    # BOTH nodes, not just the head. This check reads the *image*, and each rank
    # runs its own copy of it: a head rebuilt while the worker still carries the
    # previous image passes here and then fails at launch — or serves with an
    # image whose backends cannot accept the pinned block size and whose
    # `deep_gemm.py` has no SM12x fallback. The FlashInfer cache check below runs
    # per node for the same reason; the worker sees the head's paths (see
    # check_worker_model), so the same mount is correct on both.
    for label in head worker; do
        if [ "$label" = head ]; then target=""; else target="$WORKER_SSH"; fi
        log "verifying sparse-MLA shapes and the SM12x fallback on the $label node"
        rc=0
        out="$(docker_probe "$target" run --rm --gpus all --entrypoint python3 \
            -v "$MODEL_MOUNT_HOST:$MODEL_MOUNT_TARGET:ro" \
            "$IMAGE" /opt/dsv41/overlay/check_flashinfer_dsv41.py \
            --config "$MODEL_DIR_IN_CONTAINER/config.json" --tp "$TP" --block-size 64 \
            --max-image-tokens "$EFFECTIVE_MAX_IMAGE_TOKENS" 2>&1)" || rc=$?
        [ -z "$out" ] || printf '%s\n' "$out"
        case "$rc" in
            0) log "sparse-MLA shapes and the SM12x fallback OK on the $label node" ;;
            6) die "the manager block size lands a compressed page on a DeepGEMM block_kv with no kernel on the $label node (RESEARCH 11.9.15, 11.9.17). V4.1 compresses at ratio 1 and 2, so --block-size must be 64: that gives compressed pages of 64 and 32. DeepGEMM instantiates only 64 — its metadata builder also admits 32, which is why an unpatched image passes warmup and dies on the first long prompt — so the 32 is served by overlay/patch_sm12x_indexer.py with overlay/sm12x_mqa.py. --block-size 128 gives {128, 64} and cannot boot at all. Do not change --block-size; rebuild the $label node's image." ;;
            5) die "FlashInfer cannot serve this checkpoint's compressed page blocks on the $label node (RESEARCH 11.9.16). The image must carry the 32-state prefill instantiation from overlay/patch_flashinfer_prefill_pbsx.py with the AOT binary masked (overlay/mask_flashinfer_aot.py). Rebuild the $label node's image; do not deploy." ;;
            3) die "FlashInfer is missing a decode specialization for this checkpoint on the $label node. See RESEARCH 11.9.2." ;;
            4) die "FlashInfer is not importable in the $label node's image; cannot verify sparse-MLA shapes." ;;
            *) die "the sparse-MLA shape check failed on the $label node with exit $rc" ;;
        esac
    done
}

vllm_argv() {
    local rank="$1"
    local extra=()
    extra+=(--served-model-name "$SERVED_MODEL_NAME")
    extra+=(--host 127.0.0.1 --port "$PORT")
    extra+=(--tensor-parallel-size "$TP" --nnodes "$NNODES")
    extra+=(--node-rank "$rank")
    extra+=(--master-addr "$HEAD_IP" --master-port "$MASTER_PORT")
    extra+=(--distributed-executor-backend mp)
    extra+=(--tokenizer-mode "$TOKENIZER_MODE")
    extra+=(--tool-call-parser "$TOOL_CALL_PARSER")
    extra+=(--reasoning-parser "$REASONING_PARSER")
    # A tool parser without this flag is inert and reports nothing.
    [ "$ENABLE_AUTO_TOOL_CHOICE" = "1" ] && extra+=(--enable-auto-tool-choice)
    # The tri-state resolves to ENABLED when omitted; say it out loud.
    if [ "$NO_ASYNC_SCHEDULING" = "1" ]; then
        extra+=(--no-async-scheduling)
    else
        extra+=(--async-scheduling)
    fi
    # Prefix caching is already the resolved default for a hybrid generative
    # model, so this is a drift guard rather than a change: pin it so a future
    # default flip cannot silently cost the 92-94% hit rate the community agent
    # lanes measure.
    extra+=(--enable-prefix-caching)
    # Cap a running long prefill so a cold 64K prompt cannot hold the step.
    [ "${LONG_PREFILL_TOKEN_THRESHOLD:-0}" -gt 0 ] 2>/dev/null \
        && extra+=(--long-prefill-token-threshold "$LONG_PREFILL_TOKEN_THRESHOLD")
    # Only pass --quantization when explicitly set; see check_quantization_matches.
    [ -n "$QUANTIZATION" ] && extra+=(--quantization "$QUANTIZATION")
    # Lazy safetensors is the only loader that skips the Engram tables before
    # they are read; the overlay refuses whole-shard loaders while disk is on.
    extra+=(--load-format safetensors)
    extra+=(--safetensors-load-strategy lazy)
    # 64, not 128, and not left to the negotiation. The sparse indexer hands
    # DeepGEMM's paged-MQA path `kv_cache_spec.num_states` as `block_kv`, and
    # DeepGEMM gates that twice with different sets: the metadata builder takes
    # 32 or 64, the FP8 logits kernel only 64 (vLLM #53635). `num_states` is
    # `block_size // compress_ratio` and V4.1 compresses at ratio 1 and 2, so 64
    # gives compressed pages of 64 and 32 — one page per kernel block, the 64 on
    # DeepGEMM and the 32 on the overlay's Triton fallback
    # (overlay/patch_sm12x_indexer.py, RESEARCH 11.9.17) — while 128 gives 128
    # and 64, where 128 is outside both sets and the engine cannot boot at all.
    #
    # Every backend sharing this KV cache group must also *accept* 64. The
    # sparse-MLA and indexer specs are uniform with each other, and on SM121 all
    # three V4.1 backends advertised 128 until
    # overlay/patch_v41_kernel_block_size.py gave them a capability-family-120
    # clause returning [64]; without that patch this line raises "No common
    # block size for 64" at engine init, after the weights have loaded.
    #
    # The sparse-SWA backend accepts any MultipleOf(64) and its cache page is
    # pinned to 64 by overlay/patch_swa_page_size.py, so it agrees either way.
    extra+=(--block-size 64)
    extra+=(--max-model-len "$MAX_MODEL_LEN")
    extra+=(--max-num-seqs "$MAX_NUM_SEQS")
    extra+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
    extra+=(--gpu-memory-utilization "$GPU_MEM_UTIL")
    extra+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
    extra+=(--kv-cache-memory-bytes "$KV_CACHE_MEMORY_BYTES")
    extra+=(--engram-config.cpu_offload false)
    # The four first-boot posture switches must be *honored*, not just allowed
    # through the gate: `refuse_first_boot_shape` only decides whether the
    # combination is permitted, and the argv is what the engine actually sees.
    # `--enforce-eager` and `--language-model-only` are store-true flags
    # (`ModelConfig.enforce_eager = False`), so the false side is expressed by
    # omitting them rather than by a negated spelling.
    [ "$LANGUAGE_MODEL_ONLY" = "1" ] && extra+=(--language-model-only)
    [ "$ENFORCE_EAGER" = "1" ] && extra+=(--enforce-eager)
    # `--language-model-only` makes every modality's limit 0
    # (`config/multimodal.py`, `get_limit_per_prompt`), so no image can ever
    # reach this engine — but it does NOT stop the checkpoint's image-span
    # budget from widening the SWA prefill index rows. `attention.py` and
    # `sparse_swa.py` size those rows at `sliding_window + vision_max_n_token`,
    # which for this checkpoint's config is 128 + 1024 = 1152. That whole table
    # is handed to FlashInfer as the SWA segment, and the SM120 sparse-MLA
    # dispatchers instantiate topk in {128, 192, 256, 512, 1024} (decode) and
    # those plus 2048 (prefill) — while the dual path this tree actually takes
    # requires topk == 128 exactly. So a 1152-wide SWA segment has no kernel on
    # any path and the engine dies at the first attention call with "SM120
    # sparse-MLA has no decode kernel for this shape".
    #
    # Collapse the *budget*, not the layer count. Zeroing `vision_n_layers`
    # also produces 128, but it is wrong: the MoE gate creates its `bias_vl`
    # parameter only when `vision_n_layers > 0`
    # (`models/deepseek_v4/nvidia/model.py`), so the checkpoint's bias_vl
    # weights would have no destination and the loader raises
    # `KeyError: 'layers.0.ffn.gate.bias_vl'`. Setting the per-image budget to 0
    # keeps the model structurally identical to the checkpoint while restoring
    # the index rows to `sliding_window` (128), which both dispatchers carry and
    # which upstream vLLM allocates. Set TEXT_ONLY_VISION_OVERRIDE=0 only to
    # serve images, which this kit does not support on SM120.
    if [ "$LANGUAGE_MODEL_ONLY" = "1" ] && [ "$TEXT_ONLY_VISION_OVERRIDE" = "1" ]; then
        extra+=(--hf-overrides '{"vision_max_n_token": 0}')
    fi
    # `spec_flags` appends into SPEC_EXTRA because the speculative JSON contains
    # spaces and must stay one argv word. The if-form (not `[ ... ] &&`) keeps
    # bash 3.2's `set -u` out of the empty-array case.
    SPEC_EXTRA=()
    spec_flags "${SPEC_METHOD:-none}"
    if [ "${#SPEC_EXTRA[@]}" -gt 0 ]; then
        extra+=("${SPEC_EXTRA[@]}")
    fi
    # `--enable-flashinfer-autotune` and `kernel_config.enable_flashinfer_autotune`
    # are mutually exclusive, so the three warmup switches travel together in one
    # JSON config rather than one flag plus a JSON config.
    if [ "$DISABLE_KERNEL_WARMUPS" = "1" ]; then
        extra+=(--kernel-config '{"enable_flashinfer_autotune": false, "enable_cutedsl_warmup": false, "enable_jit_warmup": false}')
    else
        extra+=(--no-enable-flashinfer-autotune)
    fi
    printf '%q ' "${extra[@]}"
}

# Translate SPEC_METHOD into `--speculative-config`, or refuse. The kit supports
# only what the pinned engine will actually initialize for this checkpoint and
# rejects everything else rather than emitting a flag the engine would reject at
# startup.
#
# `dspark` is the only method available here. `mtp` is NOT: V4.1 ships DSpark
# stages under the `mtp.*` prefix and carries no classic-MTP weights
# (`e_proj`/`h_proj`/`enorm`/`hnorm`/`hc_head`), so `SpeculativeConfig` raises
# "DeepSeek V4.1 has no classic-MTP draft ... Use speculative method 'dspark'
# instead of 'mtp'." (`config/speculative.py:1347`). `none` is the first-boot
# default.
#
# `draft_sample_method` and `enable_adaptive_verification` are emitted only as
# configured; `greedy` is DSpark's own default and is what the A/B baseline uses,
# because `probabilistic` has an upstream history of silent degeneration that a
# throughput number cannot detect.
#
# Appends to the caller's SPEC_EXTRA array. Warnings go to stderr, never stdout:
# `vllm_argv`'s stdout *is* the argv.
spec_flags() {
    # Defaults are re-stated here so the function is self-contained: the guard
    # tests extract it on its own (start.sh's `:=` block does not run), and a
    # caller that sources only this function should still get the shipped shape.
    local method="$1" n
    local block="${SPEC_DSPARK_BLOCK:-5}"
    local dsm="${SPEC_DRAFT_SAMPLE_METHOD:-greedy}"
    local av="${SPEC_ADAPTIVE_VERIFICATION:-0}"
    case "$method" in
        none|"") return 0 ;;
        dspark) ;;
        mtp) die "SPEC_METHOD=mtp cannot work on this checkpoint: DeepSeek V4.1 has no classic-MTP draft and SpeculativeConfig rejects it at startup (config/speculative.py:1347). Use SPEC_METHOD=dspark." ;;
        *) die "SPEC_METHOD=$method is not supported by this kit. Use 'none' (default) or 'dspark' — and read the SPEC section of env.example before changing anything, because the drafter's block size constrains SPEC_NUM_TOKENS." ;;
    esac
    n="${SPEC_NUM_TOKENS:-5}"
    case "$n" in ''|*[!0-9]*) die "SPEC_NUM_TOKENS must be a positive integer, got '$n'" ;; esac
    [ "$n" -ge 1 ] || die "SPEC_NUM_TOKENS must be >= 1, got $n"
    # One DSpark round emits a block of `block` draft tokens, and the engine
    # rejects a value that is neither inside that block nor a multiple of it.
    # Catch it here rather than after the multi-minute weight load.
    case "$block" in ''|*[!0-9]*) die "SPEC_DSPARK_BLOCK must be a positive integer, got '$block'" ;; esac
    [ "$block" -ge 1 ] || die "SPEC_DSPARK_BLOCK must be >= 1, got $block"
    if [ "$n" -gt "$block" ] && [ $((n % block)) -ne 0 ]; then
        die "SPEC_NUM_TOKENS=$n is not usable with DSpark: one drafting round emits a block of $block tokens, so the value must be at most $block or a multiple of it (the engine raises otherwise — config/speculative.py:1478-1485). Valid: 1-$block, $((block * 2)), $((block * 3)), ..."
    fi
    case "$dsm" in
        greedy|probabilistic) ;;
        *) die "SPEC_DRAFT_SAMPLE_METHOD must be 'greedy' or 'probabilistic', got '$dsm'" ;;
    esac
    case "$av" in
        0) ;;
        1) die "SPEC_ADAPTIVE_VERIFICATION=1 cannot work on this deployment: adaptive verification needs full CUDA-graph support, and CUDA graphs are off for this checkpoint (they produce NaN here, and the vendor's card records an illegal memory access in both PIECEWISE and FULL_AND_PIECEWISE modes). The engine cannot honour the setting, so accepting it would only look like it had been applied. Leave it 0." ;;
        *) die "SPEC_ADAPTIVE_VERIFICATION must be 0, got '$av'. 1 is refused because it requires full CUDA graphs, which this checkpoint cannot use." ;;
    esac
    local json
    json="{\"method\": \"$method\", \"num_speculative_tokens\": $n, \"draft_sample_method\": \"$dsm\"}"
    warn "SPEC_METHOD=$method num_speculative_tokens=$n draft_sample_method=$dsm — check logprobs for NaN (scripts/smoke_chat.py) before believing any throughput number, and check a long generation for loops if draft_sample_method=probabilistic."
    SPEC_EXTRA+=(--speculative-config "$json")
}

docker_common() {
    local iface="$1" ip="$2" gid="${3:-$NCCL_IB_GID_INDEX}"
    # No `--rm`: with `-d` a rank that dies during model construction would be
    # removed by its own exit, taking the only copy of its traceback with it.
    # `stop_all` and `launch_rank` remove the containers explicitly instead.
    cat <<EOF
--network host --ipc=host --privileged
--ulimit memlock=-1
--oom-score-adj ${OOM_SCORE_ADJ}
--stop-timeout ${STOP_TIMEOUT}
-e NVIDIA_VISIBLE_DEVICES=all
-e DSV41_ENGRAM_DISK=1
-e DSV41_ENGRAM_STRICT=${DSV41_ENGRAM_STRICT:-1}
-e DSV41_MODEL_DIR=${MODEL_DIR_IN_CONTAINER}
-e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS}
-e VLLM_WORKER_MULTIPROC_METHOD=spawn
-e NCCL_IB_HCA=${NCCL_IB_HCA}
-e NCCL_IB_GID_INDEX=${gid}
-e NCCL_CUMEM_ENABLE=0
-e NCCL_CROSS_NIC=${NCCL_CROSS_NIC}
-e NCCL_P2P_DISABLE=0
-e VLLM_HOST_IP=${ip}
-e GLOO_SOCKET_IFNAME=${iface}
-e NCCL_SOCKET_IFNAME=${iface}
-e TORCH_CUDA_ARCH_LIST=12.1a
-e CUTE_DSL_ARCH=${CUTE_DSL_ARCH}
-e MAX_JOBS=${MAX_JOBS}
-e HF_HUB_OFFLINE=${HF_HUB_OFFLINE}
-e TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE}
-e UCX_MEM_MMAP_HOOK_MODE=${UCX_MEM_MMAP_HOOK_MODE}
-e UCX_RCACHE_MAX_UNRELEASED=${UCX_RCACHE_MAX_UNRELEASED}
-e VLLM_DEEP_GEMM_WARMUP=$( [ "$DISABLE_KERNEL_WARMUPS" = "1" ] && echo skip || echo relax )
-e VLLM_CACHE_ROOT=/root/.cache/vllm
-e TRITON_CACHE_DIR=/root/.cache/vllm/triton
-e TORCHINDUCTOR_CACHE_DIR=/root/.cache/vllm/torchinductor
-e XDG_CACHE_HOME=/root/.cache/vllm/xdg
-e TILELANG_CACHE_DIR=/root/.cache/vllm/tilelang
-v ${MODEL_MOUNT_HOST}:${MODEL_MOUNT_TARGET}:ro
-v ${CACHE_HOST}:/root/.cache/vllm
-v ${FLASHINFER_CACHE_HOST}:/root/.cache/flashinfer
EOF
}

launch_rank() {
    local rank="$1" name="$2" host="$3" iface="$4" ip="$5" gid="$6"
    local argv inner
    argv="$(vllm_argv "$rank")"
    inner="python3 /opt/dsv41/overlay/serve.py ${MODEL_DIR_IN_CONTAINER} $argv"
    if [ "$rank" = "1" ]; then
        inner="python3 /opt/dsv41/overlay/serve.py ${MODEL_DIR_IN_CONTAINER} $argv --headless"
    fi
    local run=(docker run -d --name "$name" --gpus all)
    # The container is kept on exit (no `--rm`), so a stale one would hold the
    # name and fail the next start. Clear it first; its logs are superseded.
    if [ "$host" = "local" ]; then
        docker rm -f "$name" >/dev/null 2>&1 || true
    else
        worker_ssh "docker rm -f '$name'" >/dev/null 2>&1 || true
    fi
    # shellcheck disable=SC2207
    local flags=( $(docker_common "$iface" "$ip" "$gid") )
    # The image ENTRYPOINT is the vLLM CLI; override it or the `bash -lc`
    # below becomes a vLLM argument and both ranks fail to parse.
    run+=(--entrypoint bash "${flags[@]}" "$IMAGE" -lc "$inner")
    if [ "$host" = "local" ]; then
        "${run[@]}"
    else
        worker_ssh "$(printf '%q ' "${run[@]}")"
    fi
}

diagnose_rank() {
    local rank="$1" name="$2" host="$3"
    local state logs
    local fmt='{{.State.Status}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}}'
    if [ "$host" = "local" ]; then
        state="$(docker inspect -f "$fmt" "$name" 2>&1)" || state="no container"
        logs="$(docker logs --tail 60 "$name" 2>&1)" || true
    else
        state="$(worker_ssh "docker inspect -f '$fmt' '$name'" 2>&1)" \
            || state="no container"
        logs="$(worker_ssh "docker logs --tail 60 '$name'" 2>&1)" || true
    fi
    printf '\033[1;31m[dsv41-reap]\033[0m rank %s (%s) %s\n' "$rank" "$name" "$state" >&2
    printf '%s\n' "$logs" >&2
}

stop_all() {
    docker rm -f "$HEAD_NAME" 2>/dev/null || true
    worker_ssh "docker rm -f '$WORKER_NAME'" 2>/dev/null || true
    log "stopped $HEAD_NAME / $WORKER_NAME (nothing else touched)"
}

status() {
    docker ps --filter "name=$HEAD_NAME" --format '{{.Names}} {{.Status}}' || true
    worker_ssh "docker ps --filter name=$WORKER_NAME --format '{{.Names}} {{.Status}}'" || true
    curl -sf "http://127.0.0.1:${PORT}/health" && echo " health=ok" || echo " health=down"
}

smoke() {
    python3 "$SCRIPT_DIR/scripts/smoke_chat.py" --port "$PORT"
}

start_all() {
    preflight
    [ -f "$MODEL_DIR_HOST/config.json" ] || die "weights missing at $MODEL_DIR_HOST (no download from this script)"
    mkdir -p "$CACHE_HOST"
    worker_ssh "mkdir -p '$CACHE_HOST'"
    mkdir -p "$FLASHINFER_CACHE_HOST"
    worker_ssh "mkdir -p '$FLASHINFER_CACHE_HOST'"
    log "starting worker rank 1 first"
    launch_rank 1 "$WORKER_NAME" remote "$WORKER_CX7_IF" "$WORKER_IP" "$WORKER_GID"
    sleep 3
    log "starting head rank 0"
    launch_rank 0 "$HEAD_NAME" local "$HEAD_CX7_IF" "$HEAD_IP" "$HEAD_GID"
    log "waiting for /health (timeout ${READY_TIMEOUT}s)"
    local t=0
    while [ "$t" -lt "$READY_TIMEOUT" ]; do
        if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null; then
            log "ready"
            return 0
        fi
        sleep 5
        t=$((t + 5))
    done
    # A rank that died during model construction is the usual cause; without
    # this the operator only sees "not healthy" and has to guess.
    diagnose_rank 1 "$WORKER_NAME" remote
    diagnose_rank 0 "$HEAD_NAME" local
    die "not healthy after ${READY_TIMEOUT}s"
}

case "$CMD" in
    validate|preflight) preflight ;;
    start) start_all ;;
    stop) stop_all ;;
    restart) stop_all; start_all ;;
    status) status ;;
    logs)
        if [ "${2:-head}" = "worker" ]; then
            worker_ssh "docker logs -f $WORKER_NAME"
        else
            docker logs -f "$HEAD_NAME"
        fi
        ;;
    smoke) smoke ;;
    *) die "usage: $0 validate|preflight|start|stop|restart|status|logs|smoke" ;;
esac
