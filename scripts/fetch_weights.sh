#!/usr/bin/env bash
# Stage the REAP-256E checkpoint, including the two Engram shards the REAP repo
# does not publish.
#
# `LibertAIDAI/DeepSeek-V4.1-Flash-REAP-256E` ships 46 of the 48 shards its own
# `model.safetensors.index.json` lists. The two it omits are the two Engram
# layers (`layers.1.engram.*`, `layers.14.engram.*`), which REAP never touched —
# 189.1 GiB, 94.56 GiB each — and they must be staged from the base checkpoint
# `deepseek-ai/DeepSeek-V4.1-Flash` into the same directory, under the same
# filenames. A directory missing them passes every "the shards I downloaded are
# there" check and then fails inside the loader, minutes into a 385.5 GiB load.
#
# Run it on the node that will serve, in a window where the download is welcome:
#
#     scripts/fetch_weights.sh              # on the head
#     scripts/fetch_weights.sh --verify-only
#
# Then get it to the other node over the QSFP rail rather than downloading it
# twice. `start.sh` mounts the HEAD's model cache directory on both ranks, so the
# worker must hold the same tree at the same path, as the account the launcher
# uses there:
#
#     rsync -a --info=progress2 \
#         ~/.cache/huggingface/hub/models--LibertAIDAI--DeepSeek-V4.1-Flash-REAP-256E/ \
#         nvidia@192.168.177.11:/home/nvidia/.cache/huggingface/hub/models--LibertAIDAI--DeepSeek-V4.1-Flash-REAP-256E/
#
# The exact command is printed at the end, derived from WORKER_USER, WORKER_IP
# and WORKER_HOME, so it lands where the worker's launcher looks.
#
# It stages into exactly the directory `start.sh` resolves — `$HF_HOME_HOST/hub/
# models--<org>--<name>/snapshots/<rev>` — and then runs `verify_weights.py`,
# which reads only the safetensors headers and refuses unless every indexed key
# is present in the shard the index assigns it to and every shard's declared
# extent matches its size. `start.sh` will not download anything itself.
set -euo pipefail

KIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "$KIT_DIR/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    . "$KIT_DIR/.env"
    set +a
fi

MODEL="${MODEL:-LibertAIDAI/DeepSeek-V4.1-Flash-REAP-256E}"
MODEL_REVISION="${MODEL_REVISION:-main}"
# The shards the REAP repo omits, and the checkpoint that has them. The base has
# its own revision: reusing MODEL_REVISION would request the REAP commit from an
# unrelated repository the moment someone pins REAP to a SHA.
BASE_MODEL="${BASE_MODEL:-deepseek-ai/DeepSeek-V4.1-Flash}"
BASE_MODEL_REVISION="${BASE_MODEL_REVISION:-main}"
ENGRAM_SHARDS="model-00047-of-00048.safetensors model-00048-of-00048.safetensors"
EXPECT_SHARDS="${EXPECT_SHARDS:-48}"
# `start.sh` resolves the same root and refuses to start without `config.json`
# under it, so a mismatch here is a 386 GiB mistake.
HF_HOME_HOST="${HF_HOME_HOST:-$HOME/.cache/huggingface}"

# Where the worker must end up holding the same tree. `start.sh` mounts the
# head's absolute path on both ranks, so this is the worker's own cache path.
WORKER_USER="${WORKER_USER:-nvidia}"
WORKER_IP="${WORKER_IP:-192.168.177.11}"
WORKER_HOME="${WORKER_HOME:-/home/nvidia}"
WORKER_HF_HOME_HOST="${WORKER_HF_HOME_HOST:-$WORKER_HOME/.cache/huggingface}"

# Space needed BEYOND what is already staged. The 385.5 GiB total is 196.4 GiB of
# REAP shards plus 189.1 GiB of Engram; a re-run on a complete checkpoint needs
# only the headroom, so the gate is on the remaining need rather than the total.
REAP_BYTES_GIB="${REAP_BYTES_GIB:-197}"
ENGRAM_SHARD_GIB="${ENGRAM_SHARD_GIB:-95}"
HEADROOM_GIB="${HEADROOM_GIB:-25}"
# A staged Engram shard counts as present only if it is big enough to be real; a
# partial write from an interrupted run must be re-fetched, not trusted. The
# threshold is overridable only so the staging tests can drive this path with
# tiny stand-in shards — `verify_weights.py` still checks every shard's declared
# extent against its size at the end, so a truncated file cannot pass unnoticed.
ENGRAM_MIN_BYTES="${ENGRAM_MIN_BYTES:-$((1024 * 1024 * 1024))}"

VERIFY_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --verify-only) VERIFY_ONLY=1 ;;
        -h | --help)
            sed -n '2,32p' "${BASH_SOURCE[0]}"
            exit 0
            ;;
        *)
            echo "unknown argument: $arg" >&2
            exit 2
            ;;
    esac
done

log() { printf '\033[1;36m[dsv41-weights]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[dsv41-weights]\033[0m %s\n' "$*" >&2; exit 1; }

MODEL_CACHE="$HF_HOME_HOST/hub/models--${MODEL//\//--}"
CACHE_DIR="$HF_HOME_HOST/hub"

# `hf` is a uv tool on the Sparks, in ~/.local/bin, which a non-login shell does
# not have on PATH. Resolve it rather than assuming.
hf_cli() {
    if command -v hf >/dev/null 2>&1; then
        command -v hf
        return
    fi
    for candidate in "$HOME/.local/bin/hf" /usr/local/bin/hf; do
        if [ -x "$candidate" ]; then
            echo "$candidate"
            return
        fi
    done
    if [ -x "$HOME/.local/bin/uvx" ]; then
        echo "$HOME/.local/bin/uvx --from huggingface-hub hf"
        return
    fi
    if command -v uvx >/dev/null 2>&1; then
        echo "uvx --from huggingface-hub hf"
        return
    fi
    die "cannot find the 'hf' CLI; install it with 'uv tool install huggingface-hub'"
}

HF="$(hf_cli)"
# shellcheck disable=SC2086
hf() { $HF "$@"; }

# The snapshot dir the launcher will mount, resolved the way `start.sh` resolves
# it: through `refs/<rev>` when that exists, else the literal revision.
snapshot_dir() {
    local cache="$1" revision="$2"
    local ref="$cache/refs/$revision"
    if [ -f "$ref" ]; then
        echo "$cache/snapshots/$(tr -d '[:space:]' < "$ref")"
    else
        echo "$cache/snapshots/$revision"
    fi
}

# Gibibytes of REAP data still to fetch: the index's own total minus what is
# already on disk. Falls back to the whole checkpoint when there is no index yet.
reap_remaining_gib() {
    local snapshot="$1"
    local index="$snapshot/model.safetensors.index.json"
    if [ ! -f "$index" ]; then
        echo "$REAP_BYTES_GIB"
        return
    fi
    python3 - "$index" "$snapshot" <<'PY'
import json
import pathlib
import sys

index = pathlib.Path(sys.argv[1])
snapshot = pathlib.Path(sys.argv[2])
payload = json.loads(index.read_text())
total = int((payload.get("metadata") or {}).get("total_size") or 0)
shards = sorted(set(payload["weight_map"].values()))
have = 0
for shard in shards:
    path = snapshot / shard
    if path.is_file():
        have += path.stat().st_size
missing = max(total - have, 0)
print(f"{missing / 1024**3:.1f}")
PY
}

# A staged Engram shard counts as present only if it is big enough to be real; a
# partial write from an interrupted run must be re-fetched, not trusted.
engram_staged() {
    local path="$1"
    [ -f "$path" ] || return 1
    [ "$(wc -c < "$path")" -ge "$ENGRAM_MIN_BYTES" ]
}

free_gib() {
    df -Pk "$1" | awk 'NR==2 {print int($4/1048576)}'
}

# ------------------------------- main ---------------------------------------
# Create the roots before anything reads them: `df` on a nonexistent path fails,
# and under `set -o pipefail` that aborts the script before its own fallback.
mkdir -p "$CACHE_DIR"

if [ "$VERIFY_ONLY" -eq 0 ]; then
    reap_snapshot="$(snapshot_dir "$MODEL_CACHE" "$MODEL_REVISION")"
    need_reap="$(reap_remaining_gib "$reap_snapshot")"

    # Which Engram shards still have to come down. Only these are fetched, so a
    # re-run after an interruption does not re-pull a shard already in place.
    missing_engram=""
    need_engram=0
    for shard in $ENGRAM_SHARDS; do
        if ! engram_staged "$reap_snapshot/$shard"; then
            missing_engram="$missing_engram $shard"
            need_engram=$((need_engram + ENGRAM_SHARD_GIB))
        fi
    done
    missing_engram="${missing_engram# }"

    # The index lists all 48 shards, the two Engram shards among them, so once it
    # exists its remaining total ALREADY covers them — adding the Engram estimate
    # on top double-counts 189 GiB. Before the first download there is no index,
    # nothing is staged, and both parts have to be added.
    if [ -f "$reap_snapshot/model.safetensors.index.json" ]; then
        need=$(( ${need_reap%.*} + HEADROOM_GIB ))
        need_parts="${need_reap} GiB missing (index-derived, Engram included) + ${HEADROOM_GIB} GiB headroom"
    else
        need=$(( ${need_reap%.*} + need_engram + HEADROOM_GIB ))
        need_parts="${need_reap} GiB REAP + ${need_engram} GiB Engram + ${HEADROOM_GIB} GiB headroom"
    fi
    avail="$(free_gib "$CACHE_DIR" 2>/dev/null || true)"
    [ -n "$avail" ] || die "cannot read free space for $CACHE_DIR"
    if [ "$avail" -lt "$need" ]; then
        die "need ${need} GiB more at $CACHE_DIR (${need_parts}) but only ${avail} GiB is free. Already-staged data is reused; free space or stage elsewhere."
    fi
    log "$(hostname): ${avail} GiB free, need ${need} GiB (${need_parts}), staging $MODEL@$MODEL_REVISION"

    # 1. The REAP repo: 46 shards plus the config, index and tokenizer.
    #    `--cache-dir` is explicit: the paths below are computed from
    #    HF_HOME_HOST, and letting the CLI pick its own default would put the
    #    data somewhere this script then fails to find.
    hf download "$MODEL" --revision "$MODEL_REVISION" --cache-dir "$CACHE_DIR"

    reap_snapshot="$(snapshot_dir "$MODEL_CACHE" "$MODEL_REVISION")"
    [ -d "$reap_snapshot" ] \
        || die "download finished but there is no snapshot at $reap_snapshot"
    [ -f "$reap_snapshot/model.safetensors.index.json" ] \
        || die "no model.safetensors.index.json in $reap_snapshot; the revision is wrong"

    # 2. The two Engram shards, from the base checkpoint. Staged in a sibling
    #    directory and then renamed into the snapshot: `hf download --local-dir`
    #    leaves a `.cache` tree and partial files behind, and a half-written
    #    shard must never appear under the name the loader will read.
    if [ -n "$missing_engram" ]; then
        stage="$HF_HOME_HOST/hub/.dsv41-engram-stage"
        mkdir -p "$stage"
        # Positional filenames, NOT `--include a b`: the CLI takes one value for
        # `--include`, so the second name becomes a positional filter and the
        # pattern is dropped — on huggingface-hub 1.29 that silently downloads
        # only the second shard and then fails looking for the first.
        #
        # No `--cache-dir` here: the CLI rejects it together with `--local-dir`
        # ("Cannot use both"), and `--local-dir` already makes the destination
        # explicit. The stage keeps its own `.cache/huggingface` metadata, so a
        # re-run reuses a shard already fetched into it.
        # shellcheck disable=SC2086
        hf download "$BASE_MODEL" --revision "$BASE_MODEL_REVISION" \
            --local-dir "$stage" \
            $missing_engram

        for shard in $missing_engram; do
            src="$stage/$shard"
            dst="$reap_snapshot/$shard"
            [ -f "$src" ] || die "$BASE_MODEL did not produce $shard in $stage"
            if [ -f "$dst" ] && [ "$(wc -c < "$src")" = "$(wc -c < "$dst")" ]; then
                log "$shard already staged"
            else
                # Same filesystem, so this is a rename: atomic, and no second copy.
                mv -f "$src" "$dst"
                log "staged $shard from $BASE_MODEL"
            fi
        done
        rm -rf "$stage"
    else
        log "both Engram shards already staged"
    fi
else
    reap_snapshot="$(snapshot_dir "$MODEL_CACHE" "$MODEL_REVISION")"
    [ -d "$reap_snapshot" ] || die "nothing staged at $reap_snapshot"
fi

log "verifying $reap_snapshot"
python3 "$KIT_DIR/scripts/verify_weights.py" \
    --model-dir "$reap_snapshot" \
    --expect-shards "$EXPECT_SHARDS"

log "done on $(hostname)"
log "model dir: $reap_snapshot"
if [ "$VERIFY_ONLY" -eq 0 ]; then
    # The launcher mounts the HEAD's absolute path on both ranks, so the worker
    # needs the same tree at the same path — and as the account the launcher
    # uses on the worker, which is not the account you are logged in as here.
    #
    # Scoped to the model's own cache directory, which is what `docker_common`
    # mounts. Copying the whole cache root would also ship every other model on
    # the node, which on these boxes is another ~170 GiB of unrelated weights.
    log "copy it to the other node over the QSFP rail instead of downloading twice:"
    log "  rsync -a --info=progress2 $MODEL_CACHE/ $WORKER_USER@$WORKER_IP:${WORKER_HF_HOME_HOST}/hub/models--${MODEL//\//--}/"
fi
