# DeepSeek-V4.1-Flash REAP-256E + NVMe Engram on 2× DGX Spark (GB10).
#
# Target: `LibertAIDAI/DeepSeek-V4.1-Flash-REAP-256E` — the base checkpoint with
# the routed-expert pool pruned 384→256 by REAP and nothing re-quantized (experts
# MXFP4, dense FP8, Engram untouched).
#
# Base: the public day-0 V4.1 image, pinned by manifest-index digest. Its
# manifest index carries **both** linux/amd64 and linux/arm64; the arm64 manifest
# is sha256:d84a123255b822fc22508635218000187221794f59c0694c33b0650d1e377d58 and
# `docker pull --platform linux/arm64` resolves it. Verified on the cluster
# (RESEARCH.md 11.9.12): vLLM `0.1.dev20904+g179dd0fa9` **with**
# `vllm/models/deepseek_v4_1`, FlashInfer **0.6.18** (the version this tree pins,
# newer than the GLM base's 0.6.17), torch 2.13.0+cu130, arch list including 12.1.
# Both kernel patches below apply to it at the same line numbers as the pinned
# tree (prefill `.cu` 325/354/395; `attention.py:456`, `sparse_swa.py:139`).
#
# This tree has FLASHINFER_MLA_SPARSE_DSV41 with a [128] main pool, but its V4.1
# SWA cache is built with a 32-token page while FlashInfer's SM120 sparse decoder
# requires 64 — `patch_swa_page_size.py` below is that fix. See RESEARCH.md §11.6.
#
# Layout: the repo's `overlay/` and `tests/` trees are preserved under
# /opt/dsv41 so the build-time tests resolve `../overlay` exactly as they do
# in the working tree. The overlay dir is on PYTHONPATH; `dsv41_boot` is
# imported by a site `.pth` so spawned worker processes get the hooks too.

ARG BASE=vllm/vllm-openai:deepseekv41-flash-0909@sha256:00d577a6a63281e15336029d5bcee4e9a2cf182214a4f20ba6111b1c8e79893d
FROM ${BASE}

ENV TORCH_CUDA_ARCH_LIST=12.1a
ENV FLASHINFER_CUDA_ARCH_LIST=12.1a
# Any runtime JIT that fans out to every core competes with the page cache for
# the same unified pool; a 22-job FlashInfer GEMM build exhausted host memory
# and the watchdogs reset all four nodes of the 4x Spark V4.1 recipe (its boot
# 3). Prebuild kernels where possible and keep the fan-out small.
ENV MAX_JOBS=2
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV DSV41_ENGRAM_DISK=1
# Plain assignment, not `${PYTHONPATH}`: the base image does not define it, so
# referencing it expands to a trailing colon, and an empty PYTHONPATH entry
# means the current directory — which would put `/vllm-workspace` on `sys.path`
# ahead of the installed packages. Set it, do not append to it.
ENV PYTHONPATH=/opt/dsv41/overlay
# vLLM forces spawn in cli_env_setup; pin it so the early `.pth` import cannot
# cache a "fork" default before cli_env_setup runs.
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn

# This kit overlays a vLLM tree; it does not rebuild vLLM. The base must
# already contain the V4.1 model, which is why BASE above is the V4.1 day-0
# image and not the GLM fork image. Fail the build loudly if it does not.
RUN python3 -c "import pathlib, vllm; p = pathlib.Path(vllm.__file__).parent / 'models' / 'deepseek_v4_1'; assert p.is_dir(), f'base image vLLM lacks V4.1: {p}'; print('base vLLM V4.1 OK', vllm.__version__, p)"

# The `.pth` hook cannot raise, so a failed overlay install must exit instead
# of warning. Keep the strict default in the image; a debugging shell can opt
# out with `-e DSV41_ENGRAM_STRICT=0`.
ENV DSV41_ENGRAM_STRICT=1

COPY overlay/ /opt/dsv41/overlay/
COPY tests/ /opt/dsv41/tests/
# The build-time suite reads the recipe, the prebuild script and the launcher —
# it asserts the kernel patches run before it does, that the receipt
# `build_flashinfer.sh` writes is the one the pre-flight reads, and that the
# launcher verifies it on both nodes. Copy all three in rather than letting those
# assertions skip: a skipped check is the one thing they cannot be.
# `tests/test_image_layout.py` derives this list from the tests themselves and
# fails when a build-time input is not copied, so a new one cannot be forgotten.
COPY scripts/ /opt/dsv41/scripts/
COPY start.sh /opt/dsv41/start.sh
COPY Dockerfile /opt/dsv41/Dockerfile

# Kernel-contract fix. Run first so a drifted anchor fails the build in seconds.
# The SWA cache page must be 64 tokens: FlashInfer's SM120 DeepSeek-V4 sparse
# decode specialization accepts only 64, and SM12 has no FlashMLA fallback.
# `--block-size` cannot reach this.
RUN python3 /opt/dsv41/overlay/patch_swa_page_size.py

# Second kernel-contract fix, same class as the one above. FlashInfer's SM120
# sparse-MLA prefill instantiates only 64- and 2-state compressed pages (the 2 is
# V4's ratio-128 page), and at `--block-size 64` this checkpoint's ratio-2 layers
# present 32; so the kernel gains a 32 instantiation. The 128 instantiation the
# previous form of this patch added is kept: at 64 it is unreachable, so it costs
# compile time only and leaves the 128 geometry reachable if DeepGEMM ever gains
# a 128-wide block_kv (RESEARCH 11.9.16). This is a FlashInfer edit rather than a
# vLLM one because every compressed-cache address vLLM computes is already
# correct for a 32-state page.
#
# `mask_flashinfer_aot.py` then removes the prebuilt AOT binary: FlashInfer
# returns it without compiling, so it would silently ignore the edit above
# (RESEARCH 11.9.3). `scripts/build_flashinfer.sh` compiles the patched source
# once per node, offline, before the engine starts.
RUN python3 /opt/dsv41/overlay/patch_flashinfer_prefill_pbsx.py \
    && python3 /opt/dsv41/overlay/mask_flashinfer_aot.py

# Third: the V4.1 sparse-MLA backends must *accept* the 64-token kernel block
# everything above assumes. On SM121 they all advertise 128, and the MLA and
# indexer specs share one KV cache group, so `select_common_block_size(64, ...)`
# would raise `ValueError: No common block size for 64` at engine init. This is
# the vLLM-side half of the `--block-size 64` decision (RESEARCH 11.9.15).
RUN python3 /opt/dsv41/overlay/patch_v41_kernel_block_size.py

# Fourth: `--block-size 64` lands the ratio-2 indexer layers on 32 states per
# compressed page, and DeepGEMM's arch-12 FP8 paged-MQA *logits* kernel is
# templated for exactly 64 — its metadata builder accepts 32, its logits kernel
# does not. So the engine boots, warms up, and then dies on the first long
# prompt (`attention.hpp:320`), which is what the block-size pin above cannot
# fix: 64 gives {64, 32} and 128 gives {128, 64}, and neither ratio pair is
# wholly servable. This routes those pages to the overlay's Triton fallback,
# vendored from PR #41834 and scoped to the paged path. Applied after the
# block-size patch because it is the *second* half of the same decision
# (RESEARCH 11.9.17).
RUN python3 /opt/dsv41/overlay/patch_sm12x_indexer.py

RUN python3 /opt/dsv41/tests/test_engram_disk.py \
    && python3 /opt/dsv41/tests/test_check_flashinfer_dsv41.py \
    && python3 /opt/dsv41/tests/test_check_flashinfer_cache.py \
    && python3 /opt/dsv41/tests/test_swa_page_patch.py \
    && python3 /opt/dsv41/tests/test_v41_kernel_block_size_patch.py \
    && python3 /opt/dsv41/tests/test_sm12x_indexer.py \
    && python3 /opt/dsv41/tests/test_flashinfer_prefill_patch.py

# Persistent install for every interpreter (EngineCore + TP workers included).
# Written last so it cannot fire during the build steps above.
RUN python3 -c "from pathlib import Path; Path('/usr/local/lib/python3.12/dist-packages/dsv41_engram.pth').write_text('import dsv41_boot\n')"

WORKDIR /vllm-workspace
ENTRYPOINT ["python3", "-m", "vllm.entrypoints.cli.main"]
