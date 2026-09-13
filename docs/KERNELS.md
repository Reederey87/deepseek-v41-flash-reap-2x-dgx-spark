# Kernel contracts for this 2-Spark V4.1 kit

This kit does not vendor CUDA. It binds this workspace's kernels.

## Engram

Stock: `vllm/models/deepseek_v4_1/common/engram.py`

| Kernel | Role | Why the disk overlay exists |
|---|---|---|
| `_hash_ids_kernel` / `_write_hash_cache_kernel` | n-gram hash → global row ids | Unchanged. Disk gather consumes those ids. |
| `_engram_lookup_kernel` | gather fp8 + ue8m0 → bf16 | Assumes resident/UVA storage. On GB10 `pin_memory=True` still occupies UMA. |
| `_fused_engram_post_wkv_kernel` | gate residual with looked-up KV | Unchanged. Runs after `prepare_embeddings()`. |

Disk path (`overlay/engram_disk.py`): `os.pread` of owned rows, CPU ue8m0 dequant
matching the Triton math (`scale_u8 << 23` as fp32), then H2D into `staged_rows`.
Not Direct I/O, not CUDA mmap. Later work: `O_DIRECT` aligned reads, a small
row cache, optional `cudaHostRegister` of a sliding window that is still
much smaller than the 189 GiB tables.

Row-id contract: `_engram_lookup_kernel` treats `ids` as **global** file row
ids. It computes `owned = index in [vocab_start, vocab_end)` and reads
`local = index - vocab_start`. The disk table uses the same global ids and the
same owned window, so the two paths agree head-for-head.

The skip must happen **before materialization**. vLLM's lazy iterator calls
`safe_open(...).get_tensor(name)` for every key and only then yields, so a
filter on the yielded `(name, tensor)` still reads the full table. The overlay
wraps `weight_utils.safe_open` and drops Engram names from `keys()`, so
`get_tensor` is never called for them. Consequences for the loader:

- `--load-format safetensors --safetensors-load-strategy lazy` is required;
  the launcher sets both.
- The multi-thread loader (`load_file`, whole shard into RAM) is replaced by
  the lazy iterator while disk is on.
- `fastsafetensors` / `instanttensor` / `runai` iterators raise, because they
  read whole shards before the filter can run.
- The tables live in shards 47 and 48 of 48, i.e. after most weights are
  resident; a post-yield filter would fault on a 91 GiB transient alloc.

Process model. `vllm serve` forces `VLLM_WORKER_MULTIPROC_METHOD=spawn`
(`cli_env_setup`), and a spawned EngineCore / TP worker re-imports the
`__main__` module without running its `if __name__ == "__main__":` block. The
Engram hooks are runtime monkeypatches, so an install that only runs in
`serve.py` never reaches the worker that constructs `Engram` and streams the
checkpoint. The image persists a site hook instead: `dsv41_engram.pth` imports
`dsv41_boot`, which calls `install_runtime.install()` at every interpreter
startup. That is the mechanism the existing deployment on these boxes uses.
`VLLM_WORKER_MULTIPROC_METHOD=spawn` is pinned in the image too, so the early
import cannot cache a `fork` default before `cli_env_setup` runs.

Fail-closed boot. `site.addpackage` prints a `.pth` traceback and keeps going,
so a boot hook cannot report an overlay failure by raising. A process that
continues without the hooks constructs `Engram` on the stock path and streams
the tables into UMA — the exact failure the kit exists to prevent. `dsv41_boot`
therefore prints the failure and calls `os._exit(70)` when
`DSV41_ENGRAM_STRICT=1` (the image and launcher default). `install_runtime`
raises if `engram_install._INSTALLED` is still false after `install()`, and
`serve.py` re-checks the sentinel and `DSV41_OVERLAY_FAILED` before handing off
to `vllm serve`, so a per-process opt-out cannot silently serve the stock
loader. `DSV41_ENGRAM_STRICT=0` is for a debugging shell only.

## Sparse MLA (this tree, SM12) — SWA page patched to 64 by the overlay

`_select_dsv4_attn_cls` already returns `DeepseekV4FlashInferSM120Attention` on
capability major 12 (SM121). That part is fine.
`DeepseekV4FlashInferMLASparseBackend.get_supported_kernel_block_sizes()` returns
`[128]`, which is not — see the kernel-block-size section below.

The SWA page size is not. `DeepseekV4SWACache.__init__` *declares*
`block_size: int = 64`, but the model overrides it: `attention.py:456` passes
`block_size=32` into every SWA cache, so the SWA group's manager page is 32 and
`swa_metadata.block_size` is 32. `--block-size` does **not** change this — the
SWA spec takes its page from the constructor, not from `cache_config.block_size`.

FlashInfer's SM120 DSV4 sparse decoder requires 64-token pages:
`flashinfer/mla/_sparse_mla_sm120.py` sets `_DECODE_DSV4_PAGE_BLOCK_SIZE = 64`
and `_decode_dsv4_dispatchable()` accepts only `page_block_size == 64`. On
SM120/SM121, `_resolve_dsv4_sparse_mla_backend()` in `flashinfer/mla/_core.py`
resolves `backend="auto"` **unconditionally to `"sparse"`** (there is no
trtllm-gen fallback — that path requires SM100/SM103). With a 32-token page the
decode specialization does not match, the prefill orchestrator is prefill-only
(it asserts `num_tokens > 64` in C++), and for any decode-sized call FlashInfer
raises `ValueError: SM120 sparse-MLA has no decode kernel for this shape: …
page_block_size=32 …`. First decode step on GB10 fails.

Upstream chose the 32 deliberately, not by drift: #56228 ("DeepSeek-V4.1-Flash
Model Definitions", `9b959b8657`, 2026-09-10) added the `block_size` parameter,
removed an older hardcoded `self.block_size = 64`, relaxed the backend to
`MultipleOf(32)`, and wrote "Any multiple of 32; the sparse decode kernels take
the page size at runtime". That is true for the FlashMLA path, which
`DeepseekV4SparseMLABackend.supports_compute_capability()` restricts to
`major in [9, 10]` — so it never runs on GB10. On SM12 the only V4.1 sparse-MLA
path is FlashInfer, where the page size is a dispatch key. Current `origin/main`
still passes 32, so a pin bump does not fix this.

Nothing in this tree tests the 32: the SWA tests use 64
(`test_deepseek_v4_swa_visible.py:34`, `test_flashmla_sparse.py:478`), and no test
references `DeepseekV4FlashInferSM120Attention` or constructs a
`DeepseekV4SWACache`.

So this is either an upstream inconsistency for V4.1-on-SM120, or a FlashInfer
build that differs from main; it is derived from source, not from execution.

**Contract, as built.** The image carries the 64-token page:
`overlay/patch_swa_page_size.py` rewrites `attention.py:456` to `block_size=64`
and `sparse_swa.py:139` to `MultipleOf(64)`, and the `Dockerfile` runs it first so
a drifted anchor fails the build in seconds. It touches exactly those two lines,
leaves the main MLA pool's advertised block size to the separate patch below, is
idempotent, and fails the build if an anchor drifts or a partial edit is present
(`tests/test_swa_page_patch.py`). The patch is a kernel-contract change and goes
through `cuda-kernel-reviewer`. No launcher flag reaches this. The probe below is
still the way to confirm 64 is sufficient, not just necessary — see `RESEARCH.md`
§11.6.

**Still to observe on hardware.** The `ValueError` above is read from FlashInfer
source, not seen: FlashInfer is not installed on the Mac. When a Spark window
opens, run a one-step decode (or the same call at `seqs=2`) against an image
built without the patch to confirm the failure, then with it to confirm the fix.

## Kernel block size — 64 on SM12, and all three backends must agree

The SWA page above is a FlashInfer dispatch key. The *manager* block size is a
separate constraint, and on this checkpoint it can only be 64.

vLLM's sparse indexer passes `kv_cache_spec.num_states` to DeepGEMM's paged-MQA
kernel as `block_kv`, and that kernel is templated for 32 or 64 only
(`csrc/apis/attention.hpp` asserts it, and the SM120 `.cuh` static-asserts
`SPLIT_KV == BLOCK_KV * kNumGroups`). `num_states` is
`get_num_kernel_states(block_size) = block_size // tokens_per_state`, V4.1's
compressed caches use `tokens_per_state = compress_ratio`, and this checkpoint
carries ratios 1 and 2. So 128 gives {128, 64} — 128 has no kernel, and warmup
aborts with `block_kv == 32 or block_kv == 64` — while 64 gives {64, 32}, both of
which do.

Pinning 64 is not sufficient on its own: `prepare_kernel_block_sizes` intersects
the advertised lists of every backend sharing a KV cache group, and the V4.1 MLA
and indexer specs are uniform with each other, so all three backends must accept
64. On SM121 they advertised 128 — `deepseek_v4_1/nvidia/flashinfer_sparse.py`
hardcoded, `deepseek_v4_1/sparse_mla.py` and
`v1/attention/backends/mla/indexer.py` via `64 if is_device_capability_family(90)
else 128` — so 64 alone raised `ValueError: No common block size for 64` at engine
init. `overlay/patch_v41_kernel_block_size.py` adds a capability-family-120 clause
returning `[64]` to each. `is_device_capability_family` divides by ten, so family
120 is `major == 12` and GB10's (12, 1) matches; family 90 and SM100/110 are
untouched. `start.sh` pins `--block-size 64` and the pre-flight reports the
DeepGEMM verdict ahead of the FlashInfer one. See `RESEARCH.md` §11.9.15.

## Indexer

Qwen already excludes SM120 family for cooperative/persistent topk in places.
Do not copy a source rewrite. First boot is eager + seqs=2; verify the
indexer path on GB10 before raising concurrency.


## Improvement order (after a real boot, not now)

1. Measure NVMe gather vs fused MoE vs sparse MLA on a C1 2k decode.
2. Direct I/O Engram + small host row cache if lookup dominates.
3. Only then consider graphs, DSpark, or a fat-expert kernel at 5120×1152.
4. Watch the pruned expert pool (256 vs 384) for quality before tuning throughput.
