#!/usr/bin/env bash
# Prebuild FlashInfer's patched sparse_mla_sm120 module, on the node you run it on.
#
# The image masks the prebuilt AOT binary (overlay/mask_flashinfer_aot.py) so the
# patched .cu is used, which means the module must be compiled once before the
# engine starts. Doing it here — offline, in its own container — is deliberate:
# an in-engine nvcc build races the server for the same unified memory pool, and
# a 22-job FlashInfer build has already reset every node of a 4-Spark V4.1
# recipe. RESEARCH.md §11.9.3.
#
# Run it on BOTH nodes, in a deploy window with the nodes free:
#
#     scripts/build_flashinfer.sh              # on the head
#     ssh spark2 ~/dsv41/scripts/build_flashinfer.sh
#
# The compiled module lands under ${FLASHINFER_CACHE_HOST}, which the launcher
# mounts at /root/.cache/flashinfer for both ranks, and a receipt is written
# beside it. `start.sh preflight` refuses to start unless that receipt, the
# runtime cache key and the artifact all verify on both nodes.
#
# Configuration comes from the same `.env` the launcher reads, and the cache path
# resolves the same way, so the two cannot silently disagree about where the
# module lives.
set -euo pipefail

KIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "$KIT_DIR/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    . "$KIT_DIR/.env"
    set +a
fi

IMAGE="${IMAGE:-dsv41-reap-selfbuild}"
CACHE_HOST="${CACHE_HOST:-$HOME/.cache/vllm-dsv41}"
# Same empty-value default as start.sh: env.example ships this empty on purpose.
FLASHINFER_CACHE_HOST="${FLASHINFER_CACHE_HOST:-$CACHE_HOST/flashinfer}"
MAX_JOBS="${MAX_JOBS:-2}"
MIN_FREE_GIB="${MIN_FREE_GIB:-90}"
# Must match the image ENV and docker_common, or the compiled module lands under
# a different arch directory than the one the engine looks in.
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.1a}"
CUTE_DSL_ARCH="${CUTE_DSL_ARCH:-sm_121a}"
RECEIPT_NAME="dsv41-prebuild-receipt"

log() { printf '\033[1;36m[dsv41-fi-build]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[dsv41-fi-build]\033[0m %s\n' "$*" >&2; exit 1; }

docker image inspect "$IMAGE" >/dev/null 2>&1 \
    || die "image '$IMAGE' is not present on $(hostname); build it first"
IMAGE_ID="$(docker image inspect -f '{{.Id}}' "$IMAGE")"
[ -n "$IMAGE_ID" ] || die "cannot read the image id for '$IMAGE'"

# The GPU allocator and the page cache share one pool on GB10, and the box
# hard-resets as MemAvailable approaches zero. nvcc is CPU/IO heavy, so require
# the same floor the launcher does.
avail_kib="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
[ -n "$avail_kib" ] || die "cannot read MemAvailable"
avail_gib=$((avail_kib / 1024 / 1024))
if [ "$avail_gib" -lt "$MIN_FREE_GIB" ]; then
    die "only ${avail_gib} GiB available, need ${MIN_FREE_GIB} GiB (MIN_FREE_GIB). Free the nodes first — see the deploy runbook."
fi
log "$(hostname): ${avail_gib} GiB available, image $IMAGE_ID, building"

# Create the cache root on the host so the receipt is written into a directory
# this user owns; the container creates only the version/arch subdirectories.
mkdir -p "$FLASHINFER_CACHE_HOST"

# The container's job: confirm the AOT shortcut is masked, compile, then report
# the exact path the runtime will load. If `is_aot` is True the patch is being
# ignored, so fail instead of reporting success on a stale binary.
#
# The artifact is identified through FlashInfer's own resolution —
# `JitSpec.get_library_path()` -> `FLASHINFER_JIT_DIR/<name>/<name>.so`, with the
# version and arch components computed by FlashInfer — rather than by globbing.
# Globbing would let a module built for another arch, or a stale one from an
# earlier run, stand in for the module this build produced.
read -r -d '' PY <<'PYEOF' || true
import hashlib
import pathlib

from flashinfer.jit import env as jit_env
from flashinfer.mla._sparse_mla_sm120 import gen_sparse_mla_sm120_module

spec = gen_sparse_mla_sm120_module()
print("aot_present:", spec.is_aot, flush=True)
if spec.is_aot:
    raise SystemExit(
        "AOT binary is still present, so the patched .cu would be ignored. "
        "The image must run overlay/mask_flashinfer_aot.py at build time."
    )
module = spec.build_and_load()
print("built_and_loaded:", module, flush=True)

built = pathlib.Path(spec.jit_library_path)
root = pathlib.Path(jit_env.FLASHINFER_CACHE_DIR)
if not built.is_file():
    raise SystemExit(f"the runtime would load {built}, which the build did not produce")
rel = built.relative_to(root)

try:
    import flashinfer

    version = str(flashinfer.__version__)
except Exception:
    from importlib.metadata import version as _dist_version

    version = _dist_version("flashinfer-python")

digest = hashlib.sha256()
with built.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)

print("module:", rel, flush=True)
print(f"size: {built.stat().st_size / 1024 / 1024:.1f} MiB", flush=True)
print("DSV41_RECEIPT_BEGIN")
print(f"flashinfer_version={version}")
print(f"module={rel}")
print(f"bytes={built.stat().st_size}")
print(f"sha256={digest.hexdigest()}")
print("DSV41_RECEIPT_END")
PYEOF

out="$(docker run --rm --gpus all \
    -v "$FLASHINFER_CACHE_HOST:/root/.cache/flashinfer" \
    -e "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST" \
    -e "CUTE_DSL_ARCH=$CUTE_DSL_ARCH" \
    -e "MAX_JOBS=$MAX_JOBS" \
    -e PYTHONUNBUFFERED=1 \
    --entrypoint python3 "$IMAGE" -c "$PY")" || die "the FlashInfer build failed"
printf '%s\n' "$out"

receipt="$(printf '%s\n' "$out" \
    | sed -n '/^DSV41_RECEIPT_BEGIN$/,/^DSV41_RECEIPT_END$/p' \
    | sed '1d;$d')"
[ -n "$receipt" ] \
    || die "the build did not describe its artifact; refusing to record an unverified cache"

{
    printf '# written by scripts/build_flashinfer.sh on %s\n' "$(hostname)"
    printf 'image_ref=%s\n' "$IMAGE"
    printf 'image_id=%s\n' "$IMAGE_ID"
    printf '%s\n' "$receipt"
} > "$FLASHINFER_CACHE_HOST/$RECEIPT_NAME"

log "receipt: $FLASHINFER_CACHE_HOST/$RECEIPT_NAME"
log "done on $(hostname) — run this on the other node too"
