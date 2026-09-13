# DeepSeek-V4.1-Flash REAP-256E on 2× DGX Spark — research and evidence

The reasoning and the measured evidence behind every constraint in this kit.
Superseded lanes, other projects' packs and abandoned approaches have been
pruned; where a wrong turn is still instructive it is kept, marked as a wrong
turn, with the reason it failed.

**Subsection numbers are stable anchors.** The launcher's error messages, the
overlay docstrings, the tests and `docs/KERNELS.md` all cite `§11.x` by number, so
the numbers that exist here are the ones they cite. Gaps in the sequence are
sections that were pruned; their surviving content is in §1–§8.

**Cluster:** 2× GB10, 121 GiB unified memory each (~242 GiB total), TP=2 over the
QSFP rail, 3.7 TB NVMe per node.
**Target:** `LibertAIDAI/DeepSeek-V4.1-Flash-REAP-256E`, native MXFP4/FP8, Engram
tables on NVMe. Nothing re-quantized.
**Base image:** `vllm/vllm-openai:deepseekv41-flash-0909`, pinned by manifest-index
digest, arm64 manifest.

---

## 1. What V4.1-Flash is

Released ~2026-09-10. It is **not** a drop-in of `DeepSeek-V4-Flash-0731`; the
architecture changed.

| Piece | V4-Flash-0731 | V4.1-Flash |
|---|---|---|
| Class | `DeepseekV4ForCausalLM` | `DeepseekV41ForCausalLM` |
| Backbone | 284B MoE, 43 layers | 552B nominal, 40 layers CED (20+20) |
| Active params | ~13B/token | 8B prefill / 16B decode |
| Experts | 256 routed, top-6 | **384 routed, top-6, 1 shared** |
| Attention | sparse MLA, compress 4/128 | **CSA2**, compress **1/2 only** |
| KV | ~3.5 KB/token | **890 B/token** (CSA2 + FP4 main KV) |
| Context | 1M | 1M |
| Engram | none | **~196B n-gram tables at layers 1 and 14** |
| Draft | DSpark, 3 stages, block 5 | DSpark, layers 37–39, 128 experts, top-3, block 5 |
| Vision | VL sidecar | native DeepSeek-ViT + MLP projector |
| Tokenizer | `deepseek_v4` | **`deepseek_v41`** (no Jinja) |

`config.json`: `quant_method: fp8`, `weight_block_size: [32,32]`, `scale_fmt:
ue8m0`, **`expert_dtype: fp4`**. The routed experts already ship packed E2M1, two
4-bit values per byte, so re-quantizing the experts "to NVFP4" buys almost
nothing. That is why this kit does not requantize anything.

**Storage, base checkpoint:**

| Component | Stored |
|---|---:|
| Routed + DSpark experts (MXFP4) | 259.5 GiB |
| Engram tables (MXFP8) | 188.8 GiB |
| Attention / norms / routers (MXFP8) | 5.6 GiB |
| Embedding + LM head + misc | 5.8 GiB |
| Quantization scales | 21.9 GiB |
| **Total** | **~476 GiB** |

Engram is `[384,006,168 × 256]` and `[384,016,682 × 256]` FP8 at layers 1 and 14.
Each token looks up 3 n-gram orders × 8 heads × 2 layers = **48 rows**, ~12.4 KB
of FP8 traffic per token. It is a **lookup table, not a GEMM** — DeepSeek designed
it for sparse gather with prefetch. That is why serving it from NVMe is
architecturally sound rather than a hack.

The official recipe asks for 614 GB of VRAM and is verified on TP4 GB200 NVL4 or
8× H200. Not this hardware.

## 2. Fit on this 2× Spark cluster

Unified memory, not discrete VRAM: pinned "host" allocations still count against
the 121 GiB GPU pool.

With Engram off the GPU entirely, the resident backbone is ~300 GiB (experts
259.5 + dense 5.6 + embed 5.8 + scales 21.9 + draft ~7). Per node, 121 GiB minus
~10 GiB runtime/OS minus the dense/embed share leaves **~90 GiB for experts**, so
~180 GiB cluster-wide against 259.5 GiB of native experts. **It does not fit.**
LibertAIDAI's own card says the same thing plainly: everything except the two
Engram tables is 302.3 GiB and must be resident, and two 120 GB unified boxes
cannot hold it with or without Engram on disk.

KV is not the problem here: at 890 B/token, 1M context is ~0.85 GiB per sequence.
Weights and runtime are. So there are only two routes — requantize the experts
smaller, or drop some — and this kit takes the second, because it needs no second
quantization and therefore costs no re-encoding error. (The requantization route
is a separate project and is not measured here.)

## 3. The GB10 gap, and why this kit exists

The pinned vLLM tree has the V4.1 model. What it lacks is a working SM120 (GB10)
path for two of the kernels the model needs. Both are kernel-contract problems,
and both are fixed by patching **FlashInfer**, not vLLM. §11.6 and §11.9.8 below
are those two fixes; §11.9.3 is what makes them take effect at all.

The kit **overlays a vLLM tree; it does not rebuild vLLM**, which is why §11.9.12
(the base image must already carry V4.1) is a hard constraint rather than a
preference.

## 4. The target checkpoint, and what the pruning costs

REAP-256E is the base checkpoint with the routed-expert pool cut 384 → 256 per
layer, chosen by saliency. Backbone parameters fall ~552B → ~371B. **Active
parameters per token are unchanged** (6 routed + 1 shared), so pruning buys memory
and not decode speed. On disk the checkpoint is **385.5 GiB**, down from 475.2.

**256 is an engine constraint, not a preference.** vLLM's fused MoE router
dispatches on a fixed expert-count table and rejects anything outside it with
`Unsupported expert number: 272`. The 272-expert checkpoint is the better model and
**does not load**.

Held-out perplexity, from the model card:

| experts kept, ranked by | text ppl | vs unpruned | caption ppl | vs unpruned |
|---|---:|---:|---:|---:|
| 384 (unpruned) | 3.6552 | — | 24.307 | — |
| 272 | 3.8060 | +4.1% | 24.769 | +1.9% |
| 256, mean-saliency | 3.8557 | +5.5% | 30.959 | **+27.4%** |
| **256, balanced (this kit)** | 4.1723 | **+14.1%** | 26.479 | +8.9% |

**+14.1% text / +8.9% captions is the cost of serving this checkpoint.**

Two consequences:

- A **text-only** deployment should re-prune with `--mode mean` for +5.5% text, and
  should then not use the vision path at all.
- **Do not rank experts on text saliency alone.** The model carries two router
  biases (`gate.bias`, `gate.bias_vl`) whose correlation is −0.05, so expert
  specialization is modality-dependent. At 272 kept, a text-only ranking is *worse
  than random* on captions (+15.9% against +7.2%) while its text perplexity looks
  healthy.

These are perplexity figures only; the card ran no downstream task benchmarks, so
treat them as a floor on the uncertainty.

**Served receipt on exactly this hardware** (2× GB10, TP2, Engram on NVMe):
**13.7 tok/s** at concurrency 1, **1,437,437 KV tokens at 64K context**, ~94 GiB
weights per rank, 47.2 GiB Engram per rank on disk, `--gpu-memory-utilization
0.90`. The KV pool is ~22× the first-boot concurrency, so it is mostly idle until
you ramp.

## 5. Engram on NVMe

`--engram-config.cpu_offload` does **not** save a Spark. In
`vllm/models/deepseek_v4_1/common/engram.py`, `cpu_offload=True` allocates with
`device="cpu", pin_memory=True` and lookups go through UVA — and pinned host memory
on GB10 is still the same 121 GiB pool. A sibling deployment already proved the
class on Qwen3.8-Flash-Next: `VLLM_PLE_CPU_OFFLOAD=1` is rejected at `nnodes=2`,
and even in concept that table is UMA. V4.1 needs **file-backed mmap or Direct I/O
row gather**, not `pin_memory`.

What the kit does instead (`overlay/engram_disk.py`, `engram_install.py`): serve the
tables from NVMe with `pread`, dequantize on CPU with a ue8m0 path matching this
tree's Triton kernel, hand the loader 1-row placeholders, and **never**
`pin_memory`. The Engram names are removed at `safe_open`, before `get_tensor` can
read the 91 GiB tables, and whole-shard loaders are refused while disk is on.

The hooks are runtime monkeypatches, so they must exist in **every** process, not
just the API server: `vllm serve` forces `VLLM_WORKER_MULTIPROC_METHOD=spawn`, and
a spawned EngineCore/TP worker re-imports `__main__` without running its
`__main__` block. `overlay/dsv41_boot.py` is installed by a site `.pth` so it runs
at every interpreter start.

A `.pth` cannot raise, and a process that keeps running without the hooks
materializes the tables into UMA — so a failed overlay install calls
`os._exit(70)` instead of warning. `DSV41_ENGRAM_STRICT=0` opts out for a debugging
shell.

## 6. Runtime posture, and why each setting is not optional

- **`MIN_FREE_GIB=90`, checked on both nodes, fail-closed.** On GB10 the GPU
  allocator and the page cache share one pool and the node hard-resets as
  `MemAvailable` approaches zero. There is no OOM kill and no log line afterwards.
- **`--async-scheduling` is a tri-state and omitting it means *enabled*.** `None`
  resolves to on for a generative model on the `mp` executor. Async doubles the
  sliding-window reservation and so halves the *reported* KV pool (#47728), the
  wrong trade on a KV-starved two-node fit, so the launcher always emits an
  explicit spelling and defaults to off.
- **`--tool-call-parser` is inert without `--enable-auto-tool-choice`**, silently:
  the parser is registered, nothing parses, no error is raised. Both are set. The
  chain-of-thought field is `reasoning`, not `reasoning_content`.
- **One `--kernel-config` JSON.** `--enable-flashinfer-autotune` and
  `kernel_config.enable_flashinfer_autotune` are mutually exclusive and vLLM
  raises if both are given; `VLLM_DEEP_GEMM_WARMUP=skip` travels with them. Decode
  here is bandwidth-bound, so the warmups buy little and a JIT fan-out competes
  with the page cache.
- **`UCX_MEM_MMAP_HOOK_MODE=none` and `UCX_RCACHE_MAX_UNRELEASED=1024`.** UCX is the
  RDMA transport under NCCL and its registration cache grows without bound by
  default. Because host RAM and GPU memory are one pool, that leak drains UMA until
  the node wedges. The community A/B at 256K/5-concurrent lost 10 → 6 GiB of free
  UMA and OOM-aborted in ~18 minutes without these. **`MIN_FREE_GIB` cannot catch
  it** — it samples once at start and this accumulates during serving.
- **`--oom-score-adj 500`** makes the container the preferred OOM victim, so an
  overrun reclaims the engine rather than starving `sshd`.
- **Never `--memory`.** On unified memory it caps GPU allocations too, and any
  arena auto-sizer silently shrinks to fit it.
- **All cache roots inside the persistent bind** (Triton, Inductor, XDG, vLLM). The
  container is `--rm`, so anything left in the image layer cold-compiles on every
  recreate.
- **`FIRST_BOOT=1`** holds `MAX_NUM_SEQS=2`, `--enforce-eager`,
  `--language-model-only`, `SPEC_METHOD=none`. Those four are a first-boot posture,
  not a permanent one; `FIRST_BOOT=0` lifts exactly those and nothing else. A value
  that is neither `0` nor `1` fails closed.
- **`SPEC_METHOD=mtp` is rejected on purpose.** V4.1 ships DSpark stages under the
  `mtp.*` prefix and has no classic-MTP draft, so `SpeculativeConfig` raises
  (`config/speculative.py:1347`). `dspark` is the only accepted method.
- **Thinking must be off for the smoke** (`17*19` → `323`); thinking-on with a small
  `max_tokens` returns empty content.
- **The smoke requests logprobs and fails on any non-finite value.** With CUDA
  graphs on, this checkpoint can serve output that reads fluently and is NaN all
  the way down, with a real throughput number attached. A text-only smoke passes
  straight through it, and Python's `json` parses bare `NaN`, so the guard tests
  values.

## 7. Staging the weights

The REAP repo publishes **46 of the 48 shards** its own
`model.safetensors.index.json` lists. The two it omits are the two Engram layers,
94.56 GiB each (189.1 GiB total), which REAP never touched. A directory missing
them passes every "the shards I downloaded are there" check and then fails inside
the loader, minutes into a 385.5 GiB load. `scripts/fetch_weights.sh` stages them
from the base checkpoint into the same snapshot under the same filenames, then runs
`scripts/verify_weights.py`, which reads every shard header and refuses unless
every indexed key is present in the shard the index assigns it to and every shard's
declared extent matches its size.

Four behaviours here are version- or environment-dependent, and each fails
**quietly** if got wrong:

- **`--include a b` is not two patterns.** The CLI takes one value for `--include`,
  so the second filename becomes a positional filter and the pattern is dropped.
  On `huggingface-hub` 1.29 (what the nodes run) that downloads only the second
  shard and then fails looking for the first. Observed in production:
  `Ignoring --include since filenames have been explicitly set`, then `did not
  produce model-00047-of-00048.safetensors`. The shards are passed **positionally**.
- **`--local-dir` and `--cache-dir` are mutually exclusive.** The CLI rejects the
  combination outright ("Cannot use both"). The cached download takes `--cache-dir`
  so a custom `HF_HOME_HOST` lands where the launcher looks; the Engram fetch takes
  `--local-dir` so a half-written shard never appears under the name the loader
  reads, and drops the cache flag.
- **`df` on a cache directory that does not exist fails**, and under
  `set -o pipefail` that aborts the script before its own fallback — on a fresh
  node, before a single byte is fetched. The cache root is created before it is
  measured.
- **The disk gate is on the *remaining* need, not the total.** The index already
  lists the Engram shards, so once it exists its remaining total includes them;
  adding a separate Engram estimate double-counted 189 GiB. On a fresh node there
  is no index and both parts are added.

The base checkpoint has its own `BASE_MODEL_REVISION`, because pinning REAP to a
commit SHA would otherwise request that commit from an unrelated repository.
`start.sh` never downloads anything: it resolves the snapshot path itself and
refuses to start if the weights are not there.

---

# 11. Findings, with the numbers the code cites

## 11.6 The SWA cache page must be 64, not 32

`deepseek_v4_1/attention.py:456` passes `block_size=32` to every
`DeepseekV4SWACache` (the class default of 64 is never used), so the
sliding-window pool's page is 32 and **`--block-size` cannot reach it**.
FlashInfer's SM120 DSV4 sparse decoder requires exactly 64
(`_DECODE_DSV4_PAGE_BLOCK_SIZE = 64`), `backend="auto"` resolves to `"sparse"`
unconditionally at `cc == (12,1)`, and the prefill orchestrator is prefill-only.
So a decode-sized call raises `ValueError: SM120 sparse-MLA has no decode kernel …
page_block_size=32`.

The 32 is deliberate upstream, not drift: #56228 added the parameter, removed an
older hardcoded 64, and relaxed the backend constraint to `MultipleOf(32)` — and
that relaxation is valid only for the FlashMLA path, which
`supports_compute_capability()` limits to major 9/10 and which therefore never runs
on GB10. **Current main still passes 32, so a pin bump does not fix this.**

`overlay/patch_swa_page_size.py` forces 64 at `attention.py:456` and
`MultipleOf(64)` at `v1/attention/backends/mla/sparse_swa.py:139`, leaving the main
MLA pool's `[128]` alone. It is exact, idempotent, preflights both targets before
writing either, and fails the build on a drifted anchor or a half-applied state.
The recipe repo independently requires the same thing, which is corroboration
rather than proof; the proof is §12, where the patched image reports both page
blocks servable.

## 11.9.2 The decode dispatch needs no patch — an earlier reading was wrong

An earlier reading claimed V4.1 needed decode keys `(32,640)`/`(32,1152)` and
therefore a FlashInfer patch. That reading is wrong for our call shape.

FlashInfer derives the decode key as `topk = indices.shape[-1]` from the **main**
index tensor. Our vLLM passes the **SWA segment** (width 128) as that tensor, with
the compressed segment separate in `extra_sparse_indices`, so the required key is
`(32,128)` — which FlashInfer 0.6.18 already carries. The `(32,640)`/`(32,1152)`
keys apply only to the single-index call shape that the recipe's own bind-mounted
`flashinfer_sparse.py` uses. **No decode patch is needed**, and the kit carries
none. The pre-flight still reports the candidate combined-width keys alongside the
required one, because it is cheap and it keeps the distinction visible.

## 11.9.3 The prebuilt AOT binary has to be masked

FlashInfer's `build_and_load()` returns the prebuilt `.so` whenever it exists, so
after the two patches above it would **silently ignore** the patched `.cu`.
`overlay/mask_flashinfer_aot.py` removes
`flashinfer_jit_cache/jit_cache/sparse_mla_sm120`, which turns the compiled module
into a **required artifact**: `scripts/build_flashinfer.sh` compiles it once per
node, offline, into a persistent bind mounted at `/root/.cache/flashinfer`.

Offline is deliberate. A parallel nvcc fan-out inside the engine races the server
for the same unified memory pool, and a 22-job FlashInfer build has already reset
every node of a 4-Spark V4.1 recipe. The image sets `MAX_JOBS=2` for the same
reason.

**The receipt verifies more than a filename.** A matching name proves nothing: a
module keyed on another FlashInfer version is never looked up, a cache from an
earlier image is stale, and an empty file matches any glob. `build_flashinfer.sh`
writes a receipt (image id, FlashInfer version, relative cache path, size, sha256)
and `overlay/check_flashinfer_cache.py` checks all of it inside the image on both
nodes before the launcher will start.

**The cache key is derived, never constructed.** The arch component is computed by
FlashInfer from `CompilationContext().TARGET_CUDA_ARCHS` and is *absent* until arch
detection succeeds: with no GPU the same install reports a 4-component path and
`gen_sparse_mla_sm120_module()` raises `No supported CUDA architectures found for
major versions [12]`. A receipt that hard-coded the component count or the arch
spelling would be wrong in one of the two states, so `runtime_spec()` asks
FlashInfer for `jit_library_path.relative_to(FLASHINFER_CACHE_DIR)` and fails
closed when it cannot.

**A rebuilt image invalidates the receipt**, because the receipt pins the image id.
Any image change means re-running the prebuild on each node before preflight will
pass.

## 11.9.8 The compressed pool's page block needs a 128-token instantiation

> **Superseded by §11.9.15 and §11.9.16.** The geometry below is real, but it is
> the geometry of `--block-size 128`, and that block size cannot boot this
> checkpoint at all: the indexer's DeepGEMM call aborts on it first (vLLM #53635).
> The kit now pins `--block-size 64`, which gives compressed pages of 64 and 32 —
> the second of which is what the retargeted FlashInfer patch adds. The
> `storage_block_size` analysis below still holds, and is why the kernel rather
> than the cache is the layer that changes.

This was the kit's last boot blocker, and the fix is in FlashInfer's prefill
dispatcher.

FlashInfer's SM120 sparse-MLA **prefill** accepts a compressed cache only when
`extra_page_block_size` is 64 or 2. That value is the compressed pool's own page
block, which in vLLM is `block_size // compress_ratio`. V4.1 uses **both** ratios:
at `--block-size 128`, ratio 1 gives 128 and ratio 2 gives 64. So `--block-size 128`
breaks ratio 1 and `--block-size 64` breaks ratio 2 — **no block size satisfies
both** as long as stock FlashInfer is the only source of instantiations, and the
prefill patch is unavoidable. The retarget in §11.9.16 adds 32, which is what makes
64 satisfiable; the DeepGEMM constraint in §11.9.15 is what makes 64 mandatory.

`overlay/patch_flashinfer_prefill_pbsx.py` — named `patch_flashinfer_prefill_pbsx128.py`
at the time of this section, see §11.9.16 — makes three exact edits to
`data/csrc/sparse_mla_sm120_prefill.cu`:

1. the full-tile acceptance test gains `|| extra_page_block_size == 128`;
2. the full-tile branch's bare `else` becomes `else if (== 2) … else {
   DISPATCH_FULLTILE_BY_NH_PBSX(128); }`;
3. the non-full-tile ladder gains `else if (== 128)`.

**Edit 2 is the whole point.** Stock is `if (== 64) … else { (2) }`, so adding 128
to the acceptance test alone would route a 128-token page into the `PBSX(2)`
instantiation, which computes `idx / 2` — wrong addresses, not a clean failure.
Edit 3 is the branch this tree actually takes, because vLLM passes
`extra_sparse_topk_lens`.

**Why the vLLM-side route is a dead end.** The obvious alternative is
`storage_block_size=64`. It cannot be applied where it is needed:
`create_kv_cache_views` asserts `kernel_block_size is None or kernel_block_size ==
spec.block_size` under "Padded KV pages do not support kernel block splitting"
(`kv_cache_interface.py:326`), and the `fp8_ds_mla` compressed page **is** padded
(64 states → 37,440 B against 37,376 unpadded; 128 → 74,880 against 74,752, with
`alignment=576`). Worse, `prepare_kernel_block_sizes` never reads
`storage_block_size` — it returns 128 (`flashinfer_sparse.py:112`) — and that same
list feeds both the block table and the views, so a 64-granular view over a
128-granular block table would alias two ratio-1 pages onto one manager block. The
QSA and GLM precedents confirm `storage_block_size` must equal the block-table
granularity.

**Why the kernel fix is sufficient.** The compressed slot mapping
(`compressor_utils.py:150-206`, called with `block_size = num_states`) gives
`block_ids == pos // 128` in both ratios, so ratio 1 is already 128-state and only
the kernel's page divisor is missing. `PAGE_BLOCK_SIZE` is used only as that
divisor/modulus; there is no `static_assert` or shared-memory dependence on it, and
the `alignment=576` page padding is harmless tail space the kernel never indexes —
so no alignment change is needed either. `--block-size 128` stands.

## 11.9.11 Closing §11.9.8

The fix above was implemented as a FlashInfer-side kernel instantiation, with the
AOT binary masked so the patched source is what gets compiled, and a per-node
offline prebuild. `start.sh preflight` no longer refuses for the §11.9.8 reason; it
refuses when the fix is missing from the image
(`check_flashinfer_shapes`) or when the compiled module is absent, stale or
unverifiable (`check_flashinfer_cache`). Both were confirmed on the cluster — see
§12.

**The boot still failed**, on a constraint this section did not model. The
FlashInfer-side machinery here — the AOT mask, the per-node offline prebuild, the
receipt — survives unchanged; only the instantiation it adds changed, from 128 to
32, and the manager block size with it. See §11.9.15 and §11.9.16.

## 11.9.12 The base image must already carry V4.1

The kit overlays a vLLM tree and does not rebuild vLLM, so the base must contain
`vllm/models/deepseek_v4_1`. The Dockerfile asserts it and fails the build
otherwise.

An earlier version of this kit used an image that looked interchangeable and is
not: a fork build with **no** `deepseek_v4_1`, so the guard failed on the first
`RUN`. Every image on both nodes was swept and none carried V4.1.

The base is now the public day-0 V4.1 image
`vllm/vllm-openai:deepseekv41-flash-0909`, pinned by manifest-index digest
`sha256:00d577a6…893d`. Its index carries **both** `linux/amd64` and `linux/arm64`
(the arm64 manifest is `sha256:d84a1232…7d58`), so `--platform linux/arm64`
resolves it. Verified by execution against the pulled arm64 image, not by reading:

| | value |
|---|---|
| vLLM | `0.1.dev20904+g179dd0fa9`, `models/deepseek_v4_1` present |
| FlashInfer | **0.6.18** (the version this tree pins) |
| torch / CUDA | 2.13.0+cu130 |
| arch list | includes 12.1 |

Both kernel patches were run against the image's **own** extracted files and apply
at the same line numbers as the pinned tree (`.cu` 325/354/395; `attention.py:456`;
`sparse_swa.py:139`), idempotently. The live JIT path is
`/root/.cache/flashinfer/0.6.18/121a/cached_ops/sparse_mla_sm120/sparse_mla_sm120.so`
with `is_aot=True`, which is why the mask is load-bearing here too.
`FLASHINFER_CACHE_DIR` is `$HOME/.cache/flashinfer`, the path the launcher mounts.

Because the base is a separate image rather than a shared one, a vLLM pin refresh
means a new day-0 image plus a re-run of the two anchor tests, which fail closed if
the line numbers moved.

## 11.9.13 The staging script, and the first image build

`scripts/verify_weights.py` had always ended a failed verification by naming
`scripts/fetch_weights.sh`, and there was no such file — the named remedy for a
385.5 GiB problem did not exist. It does now; §7 covers its four quiet failure
modes.

The image built end to end on the head node with all five in-image suites passing
inside the build. That is the first evidence the `COPY` map is complete, and it
caught a defect of its own: the in-image cache test shadowed the `flashinfer`
package but not its **distribution metadata**, so a version case read the real
0.6.18 and passed only on the Mac, where no FlashInfer is installed to be read. The
stub now shadows `dist-info` too and a case pins the version. A test that passes
because the thing under test is absent is worse than no test.

## 11.9.14 Deployment, and the floor that refused a correct build

The first real deployment, in order, with what it taught.

**Weights.** Staged on the head, then copied to the worker over the QSFP rail at
~355 MB/s in ~19 minutes, scoped to the model's own cache directory — the
directory `docker_common` actually mounts — so the transfer did not also ship the
~170 GiB of unrelated models in the head's cache root. `verify_weights.py` was
re-run **on the worker**: 48/48 shards, 385.5 GiB measured against 385.5 GiB
declared. Copying is not verifying, which is why the script exists.

**Deployment order matters, and one step was done in the wrong order.** The worker
image build was started while the previous model still held the GPU, leaving only
**9 GiB** of `MemAvailable` — the exact condition `MIN_FREE_GIB=90` exists to
refuse. The build survived because it is light (patches and Python tests, no
compile), but the correct order is: free both nodes, then build, then prebuild
FlashInfer, then preflight. Stopping the previous model took `MemAvailable` from
9 GiB to 114 GiB on both nodes.

**A `PYTHONPATH` wart, found by the build's own lint.** `ENV
PYTHONPATH=/opt/dsv41/overlay:${PYTHONPATH}` produced `UndefinedVar: Usage of
undefined variable '$PYTHONPATH'`, because the base image does not define it. The
expansion therefore ends in a **trailing colon**, and an empty `PYTHONPATH` entry
means *the current directory*, so `/vllm-workspace` would have been on `sys.path`.
It is now a plain assignment.

**The prebuild was right; the check was wrong.** `build_flashinfer.sh` compiled the
patched module on both nodes and wrote a per-node receipt. Preflight then refused
both nodes, on the stated belief that a compiled sparse-MLA module is "tens of
megabytes". That belief had never been measured. It is now:

| artifact | bytes |
|---|---:|
| stock AOT module shipped in the base image | 967,520 |
| patched per-node prebuild | 972,952 |

A sparse-MLA module is a **sub-megabyte** object, so the `1 << 20` floor rejected a
correct build and called it a placeholder. Our artifact being ~5 KB *larger* than
stock is the expected sign of the added compressed-page instantiations. The floor
is now 256 KiB, with both measured sizes recorded beside it and a test that accepts
both, refuses one byte below the floor, and asserts the floor stays below the real
artifact — a floor above the real size is a false negative on the critical path.

The fix had to be rebuilt **into the image**, because preflight runs this verifier
inside it. The two per-node artifacts have different digests (they are built
independently and embed build paths), which is why the receipt is per-node and the
verifier compares it against the local image id and the local artifact rather than
a shared expectation.

---

## 11.9.15 The manager block size must be 64, and the backends must accept it

This is the constraint §11.9.8 did not model, and it is the one that actually
blocked the boot. With the §11.9.8 fix in the image, the engine still died — but
earlier, during warmup, and for an unrelated reason:

```
indexer.py:1268 -> deep_gemm.py:608 -> RuntimeError: Assertion error
(.../csrc/apis/attention.hpp:262): block_kv == 32 or block_kv == 64
```

raised inside `kernel_warmup -> deepseek_v4_sparse_mla_attention_warmup ->
run_mixed_prefill_decode_warmup`. Warmup runs decode-shaped forwards regardless of
`--enforce-eager`, so eager mode is not an escape hatch.

**Root cause, established by reading the installed image rather than the docs.**
The indexer's decode metadata builder calls

```python
get_paged_mqa_logits_metadata(seq_lens, self.kv_cache_spec.num_states, self.num_sms, ...)
```

and `num_states` is `block_size // tokens_per_state`
(`v1/kv_cache_interface.py`, `get_num_kernel_states`). V4.1's compressed caches use
`tokens_per_state = compress_ratio`, and this checkpoint carries ratios 1 and 2, so
`num_states` is `block_size // ratio`. DeepGEMM's SM120 paged-MQA kernel takes
`block_kv` as a **compile-time template parameter** and its host dispatcher accepts
only 32 or 64, so:

| `--block-size` | ratio 1 | ratio 2 | boot |
|---|---:|---:|---|
| 128 | 128 | 64 | **no** — 128 has no kernel |
| 64 | 64 | 32 | yes — both have one |

64 is therefore the checkpoint's native SM12x geometry, not a workaround: it is the
only block size that puts both ratios on DeepGEMM's set, and it does so with
exactly one page per kernel block. (`--block-size 32` gives {32, 16} and is below
the floor; the verifier sweeps this and asserts 64 is the only value.)

**The alternatives are dead ends, and #53635 says so.** vLLM issue **#53635**
(open, 2026-08-24) is this exact failure on 2× GB10. It reports that PR **#53522**'s
`num_states` gate is *insufficient*: gating removes the assert but leaves
`schedule_metadata` zero-filled while the decode path still launches
`fp8_fp4_paged_mqa_logits`, which then reads a garbage schedule and takes an
asynchronous illegal memory access that surfaces in an unrelated kernel. Confirmed
in this image: the launch at `sparse_attn_indexer.py:647` is unconditional on the
non-XPU path, and `sparse_attn_indexer.py:860` raises outright when
`is_cuda() and not has_deep_gemm()` — so there is no non-DeepGEMM CUDA fallback to
fall back *to*. The megapatch's short-row decode top-k subsystem (#41834) has not
been sliced upstream.

**The vLLM half: three backends, one KV cache group.** `--block-size 64` is not
sufficient on its own. `prepare_kernel_block_sizes` intersects the advertised lists
of every backend sharing a KV cache group, and `MLAAttentionSpec.is_uniform_with_collection`
makes the V4.1 MLA and indexer specs uniform with each other, so they land in one
group and `select_common_block_size` runs over both. On SM121 all three V4.1
backends advertised 128:

| file | advertised | after the patch |
|---|---|---|
| `models/deepseek_v4_1/nvidia/flashinfer_sparse.py` | `[128]` (hardcoded) | family 120 → `[64]` |
| `models/deepseek_v4_1/sparse_mla.py` | `[64 if family(90) else 128]` | family 120 → `[64]` |
| `v1/attention/backends/mla/indexer.py` | `[64 if family(90) else 128]` | family 120 → `[64]` |

so `select_common_block_size(64, ...)` raised `ValueError: No common block size for
64` at engine init. `overlay/patch_v41_kernel_block_size.py` adds a
capability-family-120 clause to each, returning `[64]`. The edits are purely
additive — every stock line survives, verified as an ordered subsequence — so no
other platform's behaviour changes. `is_device_capability_family` divides by ten,
so family 120 is `major == 12` and GB10's (12, 1) matches.

Upstream PR **#53425** is the same direction on the same line: it returns `[64]` on
capability family 120 and leaves SM100 at its larger block, removing the hardcoded
override that `FLASHINFER_MLA_SPARSE_DSV4` carried. It keeps `--block-size 256`
because V4's C128 storage is `block_size / 128 = 2` states, which is a *different*
geometry: no block size can put 2 states on a 32- or 64-wide kernel, so V4 needs
the megapatch route. V4.1's ratios of 1 and 2 are what make the block size the
lever here.

`start.sh` now pins `--block-size 64` and the pre-flight grew a DeepGEMM check
(exit 6) so this failure is reported before the multi-minute weight load instead of
during warmup.

## 11.9.16 The prefill patch retargets from 128 to 32, and keeps 128

At `--block-size 64` the compressed pool presents **64** states per page for ratio
1 and **32** for ratio 2 (the same `block_size // compress_ratio` arithmetic, read
back out of the cache tensor's shape by `_packed_kv_page_block_size`). Stock
FlashInfer instantiates only 64 and 2 — the 2 being V4's ratio-128 page — so ratio
2 has no prefill instantiation and the engine dies on the first long prompt.

`overlay/patch_flashinfer_prefill_pbsx.py` (renamed from
`patch_flashinfer_prefill_pbsx128.py`) now makes three edits that leave the
acceptance test and both dispatch ladders **coterminous** over
`{64, 2, 128, 32}`:

1. the full-tile acceptance test gains `|| extra_page_block_size == 32`;
2. the full-tile ladder gains `else if (== 32)`, and its stock bare `else` becomes
   `else { return false; }`;
3. the non-full-tile ladder gains `else if (== 32)`.

**Edit 2 removes the trap rather than repeating it.** The previous form ended the
full-tile ladder in a bare `else { PBSX(128) }`, which is safe only as long as the
acceptance test and the ladder enumerate the same set — add a value to one and the
other silently runs the wrong divisor. The ladder now enumerates every value the
guard admits and fails closed on anything else, which is the same shape the
non-full-tile ladder already had (`return false`).

**The 128 instantiation is kept.** At 64 it is unreachable, so it is inert; it
costs compile time only, and it leaves the 128 geometry reachable if upstream ever
teaches DeepGEMM's paged-MQA kernel a 128-wide `block_kv` (option (a) in #53635).

**The 32 instantiation is not a new resource tier, by construction — but that is
not measured.** The shared-memory layout does not depend on the page block
(`smem_layout.cuh`'s `static_assert`s are on `TOTAL`, `N_HG` and `D_ROPE`), and
`PAGE_BLOCK_SIZE_EXTRA` appears only as the divisor/modulus in
`io_bulk_gather_tile`, `io_gather_scales`, `prefill_kv_entry_base` and
`xv_rope_mma_mg`, plus the `USE_WFP8_ROW_XOR = (PAGE_BLOCK_SIZE_EXTRA == 2)` flag
that 32 does not take. The source therefore offers no mechanism for a different
register or shared-memory tier. No register or shared-memory figure for the 32
variant has been measured, though, so none is claimed here: the per-node
`scripts/build_flashinfer.sh` compile is the receipt, and the numbers belong in
this section once it has run.

**The decode side needs no companion change.** `_decode_dsv4_dispatchable` checks
only the *primary* (SWA) cache's page, which `patch_swa_page_size.py` already pins
to 64, and the extra page block reaches the decode kernel as a runtime divisor
rather than a template parameter. The prefill kernel's primary `PAGE_BLOCK_SIZE` is
a hardcoded 64 in every launch macro, so it too is unaffected.

**The verifier now models all of it.** `check_flashinfer_dsv41.py` parses the
accepted set from the installed `.cu` and the instantiated set from both ladders,
checks each ladder independently, requires the acceptance test and both ladders to
enumerate the **same** set, reports the DeepGEMM `block_kv` verdict ahead of the
FlashInfer one, and fails closed with a new exit 6 when the manager block size
lands a page on a `block_kv` DeepGEMM has no kernel for.

The set-agreement leg is the one that took a review round to see. Checking each
ladder independently is not enough on its own: applying the non-full-tile edit
*alone* leaves the acceptance test at `{64, 2}` and the plain ladder at
`{64, 2, 128, 32}`, and every page block this checkpoint needs is then served
correctly — the guard rejects 32, so the full-tile path is never taken for it.
The per-ladder checks return clean, and the image is one third patched. Requiring
all three to agree is what refuses it; stock and the complete patch both satisfy
it, so the check is not vacuous at either endpoint.

---

## 11.9.17 The 32-state FP8 page has no DeepGEMM kernel, so the kit carries one

§11.9.15 read the DeepGEMM constraint as a single set, `block_kv ∈ {32, 64}`, and
concluded that `--block-size 64` (which yields pages 64 and 32) was servable. It
is not, and the difference is what this section adds: DeepGEMM gates the paged-MQA
path **twice, with different sets**, and the assert in §11.9.15 is the *first* of
them.

* `get_paged_mqa_logits_metadata`, building the decode schedule, asserts
  `block_kv == 32 or block_kv == 64` (`csrc/apis/attention.hpp:262`).
* `fp8_fp4_paged_mqa_logits`, producing the logits, has no instantiation for
  anything but **64** (`attention.hpp:320`).

So a 32-state page passes the metadata build and then reaches a logits kernel that
does not exist. On the cluster this is not a warmup failure: the engine boots,
warms up, and dies on the **first long prompt**, which is precisely the symptom
that started this investigation. The two gates also fail differently enough that
the distinction is operationally important — a metadata-set miss aborts during
`compile_or_warm_up_model`, after the weights are loaded, while a logits-set miss
survives every startup check and fails in production.

**No `--block-size` avoids it.** The page block is
`block_size // compress_ratio` and this checkpoint's ratios are 1 and 2, so 64
gives `{64, 32}` and 128 gives `{128, 64}`. Every reachable geometry either lands
a page on 32 or lands one on 128, and 128 is outside *both* sets. There is no
block size that puts both ratios on the logits kernel's single accepted value.

**Route C: replace the logits call, not the metadata build.** Three routes were on
the table: teach the arch-12 kernel a 32-state instantiation (a DeepGEMM rebuild
this kit cannot ship), slice the megapatch's short-row decode top-k (rejected by
that patch's own review as a P1 for the SM120 FP4 indexer path), and route the
unservable page to a non-DeepGEMM kernel. The third is what the upstream issue
itself lists as an option, and it is the one this kit took: the metadata build is
legal at 32, so only the logits call has to move.

`overlay/sm12x_mqa.py` vendors the FP8 paged MQA-logits Triton kernels (the paged
generic path, the rowwise path, both wrappers, and the layout helper), with
`fp8_paged_mqa_logits_sm12x` as the entry point. For this checkpoint's shape
(`H=32`, `D=128`) the **rowwise** path is taken (`head_dim % 64 == 0` and
`num_heads % 4 == 0`), launched at `BLOCK_N=128, BLOCK_D=64, BLOCK_H=8,
num_warps=4`. `overlay/patch_sm12x_indexer.py` installs a hook in
`vllm/utils/deep_gemm.py` that runs **before** `_lazy_init()`, so it also holds in
a tree where the DeepGEMM implementation is absent, and declines the call unless
*all* of these hold: CUDA, capability family 120, `q[1] is None` (FP8, not MXFP4),
the 4D packed page, and `block_kv != 64`. Anything else stays on DeepGEMM. The
patch is purely additive — no stock line is rewritten — so no other platform's
behaviour can shift, and `indexer.py` is untouched because the metadata build needs
no change.

It **fails closed** on unreviewed state rather than guessing: a page outside the
metadata set, a non-`None` `indices` (DeepGEMM's varlen path, which the overlay has
no mode for), or a page whose shape is not the 4D packed layout all raise. That
last one matters because the fallback reads the page's value and scale offsets
directly, so a layout change would silently misread every scale rather than fail.

**The page is segregated, not interleaved.** `kv_cache_as_quant_view`'s FP8 branch
is `kv_cache.unsqueeze(-2)`, giving `[num_blocks, num_states, 1, head_dim + 4]`,
and within a page all FP8 values precede the per-state fp32 scales. This was
checked against DeepGEMM's own asserted packed-page contract, not assumed; an
earlier probe indexed it the other way and disagreed with the reference.

**Receipts, on the hardware** (`for_review/v41-block-size/`,
`DEEPGEMM-ARCH12-FALLBACK-PROBE.txt`, GB10 sm_121a, capability 12.1, 48 SMs):

| Check | Result |
|---|---|
| Metadata builder, pages 32 and 64 | accepted |
| FP8 logits kernel, page 64 | accepted |
| FP8 logits kernel, page 32 | aborts at `attention.hpp:320` |
| Triton vs independent FP32 reference, 64 states | `max_abs 1.221e-04`, `max_rel 3.665e-05` |
| Triton vs independent FP32 reference, 32 states | `max_abs 1.221e-04`, `max_rel 3.459e-06` |
| Triton vs DeepGEMM, valid columns, 64 states | `max_rel 7.777e-07` |

The last row is the one that matters for the substitution: where DeepGEMM *can*
run the same geometry, the overlay agrees with it to seven significant figures.
The beyond-context columns differ, and deliberately: DeepGEMM writes no `-inf`
there at all, leaving whatever the output buffer held, while the overlay writes
`-inf`. The receipt records zero entries equal to `-inf` on DeepGEMM's side and
84344 on the overlay's, and the overlay's `-inf` mask equals the beyond-context
mask exactly. That is the strict superset of the `clean_logits=False` contract
that makes the fallback safe for the indexer's prefill path, where `clean_logits`
is always `False`: those columns are undefined under the contract, so a caller
that honours it cannot observe the difference, and a caller that reads them
anyway is strictly better off with `-inf` than with uninitialized memory.

**The FP8 posture is structural, not a setting.** The hook's MXFP4 branch is
unreachable here for a stronger reason than "the overlay is FP8 only": vLLM
resolves `indexer_kv_dtype` to `mxfp4` only on capability family 100 and raises
outright on sm_120 (`dsa_indexer_uses_fp4`, `v1/attention/backends/mla/indexer.py`),
and the default is `fp8`. So on GB10 the indexer cache is FP8 by construction, no
kit-level pin is needed, and the fallback is on the critical path rather than
dormant.

**One reviewer risk that does not apply.** The megapatch's P1 was that a fallback
must not bypass DeepGEMM's fused direct-top-k path. The pinned `deep_gemm.py` has
no `topk_indices`/`fp8_fp4_mqa_topk_indices` entry point at all, so there is
nothing for the hook to bypass.

**The pre-flight defect this exposed (P1).** `check_flashinfer_dsv41.py` had one
set, `{32, 64}`, and therefore called `--block-size 64` **clean** while the
ratio-2 pages had no kernel. That is the same class of false acceptance as the bug
being fixed, and worse than a missing check: it would have passed a boot straight
into the abort. It now splits the sets (`DEEPGEMM_METADATA_BLOCK_KV = {32, 64}`,
`DEEPGEMM_FP8_LOGITS_BLOCK_KV = {64}`), reports *which* gate rejected a page, and
lifts the logits gate only when `sm12x_fallback_installed()` finds the hook in the
installed `deep_gemm.py` — read from the installed module, never inferred from this
kit's own source. The other half of that contract, that `overlay/sm12x_mqa.py` is
in the image and its directory is on `PYTHONPATH`, is pinned at build time by
`tests/test_sm12x_indexer.py`, because the pre-flight runs from the overlay
directory and so cannot test the served process's import path.

**A review round then tightened that gate twice more.** `sm12x_fallback_installed()`
first looked only for the helper's *name*, which accepts a tree carrying the first
of the patch's two edits alone: the helper exists and imports cleanly, but nothing
calls it, so every 32-state page still reaches DeepGEMM and aborts as before — the
same false acceptance one edit further along. It now requires both edits *and their
connection*, read off the wrapper's `__code__.co_names` rather than its source
text, so neither a rename nor a reformat can satisfy it by accident. The gate also
runs **per node**, not on the head alone: it reads the *image*, each rank runs its
own copy, and a head rebuilt while the worker keeps the previous image would
otherwise pass here and then fail at launch. The build-time suite gained the
matching regressions, including a three-way installed-source classification, since
a fully patched file still contains its own anchors: the edits are additive, so
testing `old` independently reads the patched file as both states at once and the
check refuses the very output the Dockerfile produces.

---

## 12. What has been verified on the cluster

Everything here was executed on the hardware, not derived from source.

**Kernel contracts, from `./start.sh preflight` on the built image** — this receipt
is from the `--block-size 128` build, before §11.9.15; the retargeted build has not
been re-verified yet, and is expected to print `[2, 32, 64, 128]` with pages 64 and
32:

```
AOT binary in use: False  (…/flashinfer_jit_cache/jit_cache/sparse_mla_sm120/sparse_mla_sm120.so)
prefill source: …/flashinfer/data/csrc/sparse_mla_sm120_prefill.cu
  accepts compressed page blocks: [2, 64, 128]
  instantiates (both ladders):    [2, 64, 128]
compressed pool page block per ratio:
  compress_ratio 1: block_size//ratio = 128  ok
  compress_ratio 2: block_size//ratio = 64  ok
required decode dispatch keys (main index width):
  (num_heads=32, topk=128)  <- main/SWA index width (window 128)
OK: decode keys present and every compressed page block is servable
```

The acceptance set and the instantiation set agree at `[2, 64, 128]`, so the
prefill patch is applied and complete and the AOT binary is masked. The FlashInfer
cache receipt, its runtime cache key and the artifact all verify on **both** nodes.
Both sets are now checked for agreement *per ladder* rather than as a union, and the
DeepGEMM `block_kv` verdict is reported alongside them (§11.9.16).

**Weights:** 385.5 GiB staged and verified on both nodes, 48/48 shards.
**Images:** built on both nodes, all five in-image suites passing inside the build.

## 13. What is *not* verified yet

- The **boot itself**: the 385.5 GiB load, the Engram NVMe gather under real
  traffic, and the first `/health`.
- The **§11.9.15, §11.9.16 and §11.9.17 changes**, which are validated against the
  pinned source, the installed image source and reconstructed stock text, but have
  not yet been rebuilt, preflighted or booted on the cluster. In particular the
  ratio-2 (32-state) prefill path has not run a real long prompt.
- The **SM12x fallback under real traffic.** Its numerics are measured against an
  independent FP32 reference and against DeepGEMM (§11.9.17), and its dispatch
  contract is executed in the build-time suite, but it has only ever run as a
  kernel-level probe: no long prompt has yet been served through the hook, and the
  hook's fail-closed paths have not been exercised on hardware.
- **Throughput on this kit.** The 13.7 tok/s figure is the published receipt from
  the same hardware and the same checkpoint, not a measurement of this kit.
- The **`_DECODE_DSV4_DISPATCH` reading** was taken from the upstream 0.6.18 tag
  rather than a live install; the pre-flight re-reads it on-node and fails closed
  if they disagree.
- **Quality**: the perplexity figures are the model card's. Nothing has been
  measured against a real workload here.

## 14. Review provenance

- `cuda-kernel-reviewer` — **APPROVED** for the kernel-contract changes (the SWA
  page patch and the FlashInfer prefill instantiation), re-deriving the dispatch
  analysis against 0.6.18 rather than inheriting the earlier 0.6.17 reading. Its
  one recorded caveat, the decode table having been verified against the upstream
  tag, is resolved on-node: the pre-flight reads it live.
- `cuda-kernel-reviewer` — **APPROVED** for §11.9.17 (the vendored Triton fallback
  and the dispatch hook that changes which kernel serves a page). It verified
  independently that both patch anchors are unique in the pinned stock file, that
  the deltas are the claimed `+100`/`+15`, that apply/idempotency/compilation hold,
  and that the six vendored functions are byte-exact against pinned revision
  `9ad62027bc84`. It recorded three P3 items, all accepted as recorded rather than
  fixed: the hook's parameter list is a positional subset of the wrapper's, so
  `schedule_metadata` and `clean_logits` are dropped (safe because relative order
  is preserved); the `-inf` superset argument holds only because no downstream
  consumer reads beyond-context columns, which the cluster long-prompt gate should
  close empirically; and the function-local `sm12x_mqa` import fails closed only at
  the first 32-state decode, with the build-time Dockerfile/`PYTHONPATH` check as
  the gate.
- `final-reviewer` — multiple rounds, every finding fixed. The substantive ones are
  recorded where they belong rather than as a list: the `--gpus all` requirement on
  both FlashInfer probes, the cache-root mount for the shape check, the staging
  defects in §7, and the placeholder floor in §11.9.14.
- `final-reviewer`, round 4 — **CHANGES REQUIRED**, then **APPROVED** on the fix
  diff. Three required findings, all real and all reproduced by the reviewer before
  being fixed: the installed-source classifier read a fully patched `deep_gemm.py`
  as both states at once, because the edits are additive and `new` embeds `old`
  (that would have failed the image build at the test step, and the Mac suite hid
  it by skipping); `sm12x_fallback_installed()` accepted the helper's name alone,
  so the patch's first edit in isolation passed a check for a fallback that nothing
  called; and the shape gate ran on the head only although it reads the image each
  rank runs its own copy of. Its new recorded P3 is that the `co_names` test is a
  *wiring marker*, not proof of execution — a wrapper that referenced the helper
  without calling it would pass — which the docstring now states rather than
  overclaims. It confirmed both specialist-reviewed kernel artefacts unchanged by
  hash across the fix round.
