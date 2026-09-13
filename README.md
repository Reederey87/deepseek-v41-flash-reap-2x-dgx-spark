# DeepSeek-V4.1-Flash REAP-256E on 2× DGX Spark

A working, reproducible deployment of `LibertAIDAI/DeepSeek-V4.1-Flash-REAP-256E` on
two DGX Spark nodes (GB10, sm_121, 121 GiB unified memory each), tensor-parallel 2
over the QSFP rail, with the Engram tables served from NVMe.

It runs at **27.75 tok/s** single-stream decode, serves **64K context**, and passes a
needle-recall check from 6K to 64,851 prompt tokens. Getting there needed three kernel
patches and a page-size fix that no released vLLM or FlashInfer build provides; that
work is the point of this repository.

## Why this kit exists

V4.1-Flash is a 552B-parameter multimodal MoE with 196B more of sparsely-accessed
Engram memory and a 1M-token context. It does not fit on two Sparks. The FP4 experts
alone are 288.8 GB against roughly 85–90 GB of expert budget per node, before you
count the Engram tables or the KV cache.

Three ways exist to make a model like this fit. This kit takes one of them and you
should know which:

- **Prune it.** Delete the experts that do the least. That is REAP, and that is what
  this checkpoint is. It costs measured quality and buys the fit.
- **Re-quantize it harder.** Take the weights to 3-bit or 2-bit with EXL3. That is a
  *different* project — `deepseek-v4.1-flash-exl3-2dgx-sparks` — and this kit does not
  do it. See [Why FP4 and not EXL3](#why-fp4-and-not-exl3).
- **Stream it from disk.** Keep a hot subset resident and page the rest off NVMe. That
  solves capacity, not the budget, and it costs bandwidth you need for decode.

This kit prunes, and then spends the freed memory on a long KV pool and on speculative
decoding. The result is not a speed record — two Sparks are bandwidth-bound and always
will be — but it is a model you can actually talk to, with a number attached that you
can reproduce.

## What REAP means for quality

**REAP is Router-weighted Expert Activation Pruning**, from Cerebras
([arXiv:2510.13999](https://arxiv.org/abs/2510.13999)). The name is the method:

- **Expert pruning, not merging.** Merging two experts replaces them with a static
  average, and the router loses the ability to mix them differently for different
  inputs. The paper calls the result *functional subspace collapse* and proves it is an
  irreducible error. Pruning keeps the router's control over every surviving expert
  completely intact. You lose capacity; you do not lose the mechanism.
- **Weighted by the router, not by frequency.** The saliency score for an expert is the
  average, over the tokens where that expert was actually chosen, of
  `router_gate_value × ‖expert_output‖`. Averaging *conditionally* on activation is the
  important part: it stops the ranking from collapsing into "prune whatever is rare",
  which would delete specialists that are rarely called but matter when they are.

On this checkpoint it removed **128 of 384 experts per layer** (33.3% of the pool),
taking the download from 475.2 GiB to 385.5 GiB. **Nothing was re-quantized** — the
experts stay MXFP4, the dense layers stay FP8, the Engram tables are untouched.

The quality cost was measured by the publisher on held-out text, and it is not small:

| experts kept | text perplexity | vs unpruned | caption perplexity | vs unpruned |
|---|---:|---:|---:|---:|
| 384 (unpruned) | 3.6552 | — | 24.307 | — |
| 272 | 3.8060 | +4.1% | 24.769 | +1.9% |
| **256, balanced (this checkpoint)** | **4.1723** | **+14.1%** | **26.479** | **+8.9%** |

**+14.1% text perplexity and +8.9% on image captions is what you are paying.** Quote
that pair together, never one half of it.

Three things follow, and they matter more than the headline number:

- **Active parameters are unchanged** — 6 routed plus 1 shared expert per token. So
  pruning buys *memory*, not speed. Every tok/s in this README comes from speculative
  decoding and from the KV pool, not from the prune.
- **256 is an engine constraint, not a preference.** vLLM's fused MoE router dispatches
  on a fixed table of expert counts, and 272 raises `Unsupported expert number: 272` at
  startup. The publisher's 272-expert sibling scores better (+4.1% text, no measurable
  caption regression) and simply will not load. If a future engine accepts it, use it.
- **The ranking is modality-balanced, and that was not free.** This model carries two
  router biases, `gate.bias` and `gate.bias_vl`, whose correlation is −0.05, so expert
  specialisation is modality-dependent. A text-only ranking at 272 experts is *worse
  than random* on captions (+15.9% against +7.2%) while its text perplexity looks
  healthy. The publisher normalised the text and image saliency vectors to equal mean
  before averaging, buying back 14.5% of caption perplexity by giving up 8.2% of text.

**Perplexity is not a benchmark.** No downstream task evaluation was run against this
checkpoint by anyone, including us. Treat these figures as a floor on the uncertainty,
not as a substitute for measuring your own workload.

## Why FP4 and not EXL3

**There is no EXL3 step anywhere in this path.**  The checkpoint is natively FP4 and
this kit serves it as-is:

| Component | Format |
|---|---|
| Routed experts | **MXFP4** — E2M1 with a power-of-two E8M0 scale, block 32 |
| Dense / attention | **FP8**, block 32×32 |
| Engram tables | **FP8**, untouched, read from NVMe |

That distinction is the whole reason to prefer this checkpoint over a 3-bit or 2-bit
re-quantization of the same model, and it is worth being precise about why:

- **One lossy step instead of two.** A 3-bit EXL3 build stacks a second quantization
  error on top of the prune. Here the prune is the *only* lossy operation, and it is
  the one the publisher measured. You know exactly what you paid, and it is one number
  pair rather than two effects you cannot separate.
- **The weights run on the checkpoint's own kernels.** MXFP4 and FP8 block-32 are what
  DeepSeek quantized to and what Blackwell's MoE and attention kernels are built for.
  A 2-bit format needs dequantization kernels in the hot loop, which is bandwidth you
  cannot spare on a box like this.
- **It is what fits.** 385.5 GiB across two 121 GiB nodes, with the Engram tables on
  disk, leaves room for a 841,973-token KV pool. An EXL3 build is smaller on disk, but
  disk was never the binding constraint here — the expert budget was.

If you want the 3-bit or 2-bit path anyway, that is a legitimate choice for a different
trade-off, and it lives in `deepseek-v4.1-flash-exl3-2dgx-sparks`. It is not this kit.

## What we measured

Two nodes, TP2, `--enforce-eager`, `MAX_NUM_SEQS=2`, 65,536-token context,
841,973-token KV pool, 200-token greedy generations, thinking off, median of three,
`temperature: 0`. Both arms were booted through `scripts/ab_spec.sh` on the same day
from the same kit, so the only difference between the rows is the speculative config.

| Arm | tok/s | TTFT | Accept rate | Accept length | Coherent | NaN |
|---|---:|---:|---:|---:|---|---|
| `baseline` (no speculation) | 13.16 | 0.24 s | — | 1.00 | yes | no |
| **`greedy5` (DSpark, block 5, greedy draft)** | **25.65** | **0.45 s** | **0.4246** | **3.1231** | **yes** | **no** |

Those are the two configurations that matter: the control and the adopted arm. The
other two arms are described below rather than tabulated, because their absolute
numbers were taken with the earlier timing method and are not comparable to these.

**1.95× decode throughput from speculative decoding.** Two different ratios matter
here, and they are easy to confuse:

- The **acceptance rate** is `accepted / drafted` = 414/975 = **0.4246**. Roughly 2 of
  every 5 drafted tokens are right.
- The **mean acceptance length** is `1 + accepted / drafts` = 1 + 414/195 = **3.1231
  tokens per verify step**, including the always-accepted bonus token. That is the
  number that explains the speedup.

So each verify step costs about **1.6× a plain decode step** (76.0 ms against 47.6 ms
at 13.16 tok/s) and returns **3.12 tokens instead of 1**. That is the whole 1.95×, and
it is why the win is real but sublinear in the acceptance length.

TTFT is *worse*, not better: 0.45 s against 0.24 s. That is the expected shape —
speculation pays on the decode loop, not on the prefill before it, and the first
verify step has to draft a block before it can return anything. The 200-token
generation is long enough that the decode saving dominates.

### The two arms we did not adopt

**Block 10 is rejected.** Drafting twice as many tokens per step made it roughly 35%
slower and *lowered* the acceptance length (2.46 against 3.1231), because verifying
k+1 tokens scales expert-weight traffic on a sparse MoE while per-draft acceptance
falls. The publisher's card reports the same thing independently, on their own
hardware and their own benchmark: "k=10 is 24–38% slower on every workload". That
independent agreement is what the rejection rests on, not our absolute figure.

**`probabilistic` drafting is indistinguishable from `greedy` here, and that is a
limit of the bench, not a result.** Both produced exactly the same counters. Upstream
explains why: `probabilistic` "uses the full draft logits for the probability ratio
test", and the gain is "only noticeable when temperature is sufficiently large
(> 0.7)". This bench runs a greedy target at `temperature: 0`, where the token stream
is byte-identical every run, so there is nothing for the two methods to disagree
about. Judging them needs a sampled-target bench that does not exist yet.
`SPEC_DRAFT_SAMPLE_METHOD` therefore defaults to `greedy`, which is definitionally
aligned with a greedy target and has no upstream history of silent degeneration.

### A note on these numbers and an earlier version of them

An earlier revision of this bench timed tokens with `HTTPResponse.read(4096)`, which
on a chunked SSE stream blocks until the requested byte count or EOF. That timestamped
whole *buffer fills* rather than token arrivals, so it reported the first token late
(TTFT 0.82 s instead of 0.24 s) and the decode span short, inflating throughput by
about 6% (27.75 instead of 25.65). It also collapsed entirely on responses smaller
than the read size. The bench now uses `read1()`, which returns as soon as data is
available, and both rows above are from the corrected version. The correction is worth
stating plainly because the inflated figures were published in an earlier revision of
this file.

The corrected baseline agrees with an independent measurement of the same
configuration taken for the sibling suites: 13.16 against 13.26 tok/s.

### The acceptance ratio is deterministic, and that is worth knowing

The engine reports acceptance per draft position, and it decays the way you would
expect. Over one 196-draft window the raw counts were 157, 114, 72, 45 and 27, which
against 196 drafts is **0.80, 0.58, 0.37, 0.23 and 0.14**. These are *cumulative*
probabilities (the chance a draft had *at least* that many tokens accepted), not
conditional ones, which is why they sum to the accepted-per-draft figure of 2.12 rather
than to 1. The drafter's first token is reliable and its fifth is not.

More useful: **at `temperature: 0` the run is deterministic.** Every repetition of a
speculative arm produced an acceptance rate of exactly 414/975 = 0.4246, and a fresh
200-token run produced 138/325 = 0.4246 — the same rate to four decimals, three times,
because the token stream is byte-identical each time. The non-speculative `baseline`
has no acceptance counters at all, and `greedy10` is a different arm with its own rate
(0.1457); the determinism claim is about repetitions within an arm, not across arms.

That has one important consequence: **this bench cannot tell `greedy` and
`probabilistic` drafting apart.** Their counters come out identical, so the difference
between those two rows is not evidence about the methods. Upstream explains why:
`probabilistic` "uses the full draft logits for the probability ratio test", and the
gain is "only noticeable when temperature is sufficiently large (> 0.7)". At
temperature 0 there is nothing to notice.

`greedy` stays the default because it is definitionally aligned with a greedy target
and because upstream records probabilistic drafts degenerating into loops while
inflating their own acceptance. **Judging the two properly needs a sampled target**
(`temperature > 0`), which is not what this bench measures — so treat that row as "no
measurable difference at temperature 0", not as a verdict.

### Three independent cross-checks on those numbers

We did not take our own word for it. The publisher's model card, published separately,
reports:

| Their figure | Ours | Agreement |
|---|---|---|
| 13.7 tok/s, no speculation, prose-like prompt | 13.80 tok/s baseline | within 1% |
| 27.0 tok/s, DSpark k=5, chat/explanation | 27.75 tok/s | within 3% |
| "k=10 is 24–38% slower on every workload" | 35% slower | inside their band |

The third one is the interesting one: their claim about block 10 and our measurement
of it were made independently and land in the same range. That is the kind of agreement
that makes a single number trustworthy.

**Their card also warns not to quote a single number for this model**, because
throughput varies by a factor of three with workload — 19.2 tok/s on free prose against
49.4 tok/s on counting prompts. Our 27.75 is measured on a structured explanation
prompt, which is why it matches their explanation-class row and not their prose row.
**Never benchmark this model on counting or "count to N" prompts**: a degenerate
sequence lets the drafter accept nearly every token and reads about 2.6× the honest
rate. Our bench prompt is a prose/explanation task for exactly this reason.

### The rest of the picture

| Measurement | Result |
|---|---|
| Cold boot | 48/48 shards loaded, engine initialised, `/health` ready |
| Context served | 64K, needle recall verified to 64,851 prompt tokens |
| KV pool | 841,973 tokens at 65,536 ctx — 12.85× concurrency |
| Prefix caching | **3.27×** speedup on a repeated 8,710-token prompt, 0.9954 hit ratio |
| Concurrency (aggregate, ctx 0) | code 13.2 → 21.6 → 17.9 → 22.1 tok/s at ×1/×2/×3/×4 |
| Mixed prefill | newcomer TTFT 12.6 s at 9.5K, **47.9 s at 38K** |
| Decode during a long prefill | falls to **2.6–5.6%** of its rate — head-of-line blocking |
| KV cost per request | `14,894 + 0.484 × prompt_tokens` **pool tokens**, r² 0.868 |

**Every row here except the KV pool was measured with `SPEC=none`** — this is the
sibling-suite sweep, not the decode table above. The two tables measure different
things and should not be mixed. The KV pool is a boot-time fact and does not
depend on speculation.

Read these with the protocol in [`bench/README.md`](bench/README.md), not on their own.
The coherence column is a tripwire for degenerate output, not a quality score.

The concurrency row is the honest weak spot. Aggregate throughput plateaus near 22
tok/s from ×2 to ×4 because `MAX_NUM_SEQS=2` means requests 3 and 4 are never admitted —
they queue. That is a deliberate setting, not an oversight, and raising it is the next
thing to try.

### What the 64K context limit actually is

**It is the model's YaRN base, not a hardware ceiling.** The checkpoint's own
`config.json` says so:

```json
"rope_scaling": {"rope_type": "yarn", "factor": 16,
                 "original_max_position_embeddings": 65536},
"max_position_embeddings": 1048576
```

65,536 is the *un-extrapolated* native window; 1,048,576 is what YaRN extends it to.
We serve the native window because that is what has been verified here end to end, and
because reaching 1M on two 121 GiB nodes needs several enablers this kit does not set:

| Enabler | Why it matters |
|---|---|
| `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` | **The one that matters.** Caps the sparse-indexer logits buffer, which "otherwise blows up at long context" — other 2×Spark recipes call it *the* OOM killer. We do not set it. |
| `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` | vLLM otherwise refuses a ceiling above the config's |
| `--gpu-memory-utilization` ~0.82 | We deliberately stay lower and use a free-memory gate instead |
| `--block-size 256` | Fewer, bigger KV blocks, less paging at long context. We pin 64 for the DeepGEMM kernel — a different constraint |
| `--max-num-seqs` 6 | A higher per-request ceiling needs a lower concurrency cap |

**The KV pool is not the constraint.** 841,973 tokens at 65,536 context is 12.85×
concurrency, so the pool could hold thirteen concurrent full-context requests. The
constraint is the *transient* working memory of a long prefill — and the publisher of
this checkpoint says exactly that: the growth "is in the sparse-MLA indexer structures,
not the chunk activations", which is the buffer the first enabler above caps. They
report 32K as their practical ceiling with a ~62K prompt killing the engine. **We serve
64K**, so we are already past them, and the same lever is available to go further.

Raising it is on the list, not done.

### Where this kit goes further than the published recipe

The publisher reports **32K as the practical context ceiling**, with a ~62K prompt at
64K context dropping the host to 68 MB free and killing the engine. We serve 64K and
passed needle recall at 64,851 prompt tokens.

The difference is memory posture, not luck. This kit keeps a **90 GiB free-memory floor**
checked on both nodes before every start, caps each long-prefill chunk so one cold 64K
prompt cannot balloon the sparse-MLA indexer structures, never sets `--memory` on the
container, and pins UCX's registration-cache limits so the shared host/GPU pool cannot
leak. The publisher's floor under load is ~913 MB. We have not stress-tested beyond 64K
and do not claim 1M — see [above](#what-the-64k-context-limit-actually-is) for what that
would take.

### Eager is a correctness posture here, not an oversight

The usual advice that CUDA graphs buy ~55% does not transfer to this checkpoint, and
the reason is not performance:

- The vendor recipe records that graphs **silently produce NaN** on this model, and
  measures `--enforce-eager` at only ~7% (14.5 → 13.7 tok/s) because the workload is
  bandwidth-bound. Our eager control is within 1% of that 13.7.
- The publisher's current card is blunter: graphs fail in **both** `PIECEWISE` and
  `FULL_AND_PIECEWISE` with `CUDA error: an illegal memory access was encountered`,
  consistent with the sm_120 sparse-MLA kernel receiving an uninstantiated batch shape.
  Reducing the capture set does not help.
- Our tree would not enable the useful mode anyway.
  `DeepseekV41IndexerBackend.get_cudagraph_support()` returns `ALWAYS` only when
  `_supports_varlen_paged_mqa_logits()` holds, which requires capability family 100.
  GB10 is family 120, so it resolves to `UNIFORM_BATCH`.
- `enable_adaptive_verification` is unavailable for the same reason: it requires full
  graphs. That is why the launcher **refuses** `SPEC_ADAPTIVE_VERIFICATION=1` rather
  than accepting a setting the engine could never honour.

Turning graphs on to win a speed receipt would trade a measured correctness posture for
a number we cannot trust. It stays off.

## What is in here

| Path | What it does |
|---|---|
| `Dockerfile` | Builds on the pinned V4.1 base, applies all kernel patches, masks the stale AOT binary, runs the in-image test suites |
| `overlay/patch_swa_page_size.py` | Forces the sliding-window cache page to 64 tokens at build time |
| `overlay/patch_v41_kernel_block_size.py` | Makes the V4.1 sparse-MLA backends accept a 64-token kernel block on SM12x |
| `overlay/patch_flashinfer_prefill_pbsx.py` | Adds the missing 32-state compressed-page prefill instantiation to FlashInfer |
| `overlay/patch_sm12x_indexer.py` | Installs the SM12x FP8-logits fallback for the indexer's 32-state page block |
| `overlay/sm12x_mqa.py` | The fallback itself: a Triton paged-MQA logits kernel for arch 12 |
| `overlay/mask_flashinfer_aot.py` | Removes the prebuilt `sparse_mla_sm120` binary so the patched source is actually compiled |
| `overlay/engram_disk.py`, `engram_install.py` | Serve the Engram tables from NVMe with `pread`, dequantize on CPU, hide the tables from the loader |
| `overlay/dsv41_boot.py` | Installs the hooks into every spawned process, not just the API server |
| `overlay/check_flashinfer_dsv41.py` | Pre-flight: does the image's FlashInfer actually support this checkpoint's page blocks |
| `overlay/check_flashinfer_cache.py` | Pre-flight: is the compiled module present, current, and the one this image will load |
| `scripts/fetch_weights.sh` | Stages the checkpoint, including the two shards the REAP repo omits |
| `scripts/verify_weights.py` | Reads every shard header and refuses an incomplete checkpoint |
| `scripts/build_flashinfer.sh` | Compiles the patched module once per node, offline, and writes a receipt |
| `scripts/ab_spec.sh` | The A/B harness: one boot per speculative arm, control included |
| `bench/bench_decode.py` | The decode bench: tok/s, TTFT, logprobs NaN check, coherence tripwire, engine acceptance |
| `start.sh` | The launcher: preflight, start, stop, smoke |
| `env.example` | Every knob, with the defaults this kit expects |

`docs/KERNELS.md` has the kernel notes. `RESEARCH.md` has the reasoning and the evidence
behind every constraint below.

## Constraints that will bite you

**The base image must already contain V4.1.** The Dockerfile asserts that
`vllm/models/deepseek_v4_1` exists and fails the build otherwise. The base is the
public day-0 image `vllm/vllm-openai:deepseekv41-flash-0909`, pinned by manifest-index
digest, and its index does carry an arm64 manifest. It is *not* any other vLLM image
already sitting on these nodes: those are fork builds without `deepseek_v4_1`, so the
guard fails immediately. See `RESEARCH.md` §11.9.12.

**Mount the cache root, not the snapshot.** `snapshots/<rev>/*` are symlinks into
`../../blobs`, so a snapshot-only bind leaves `config.json` dangling on a perfectly
staged checkpoint.

**Override the image entrypoint.** `docker run` must pass `--entrypoint bash`, or the
inner command is parsed as vLLM arguments.

**`--block-size 64` is pinned, and the SWA page is patched to 64.** 64 is the only
block size that lands both V4.1 compression ratios on a DeepGEMM kernel: the indexer
passes `kv_cache_spec.num_states` to DeepGEMM's paged-MQA kernel as `block_kv`, that
kernel is templated for 32 or 64 only, and `num_states` is `block_size // compress_ratio`
— so 64 gives compressed pages of 64 and 32, while 128 gives 128 and 64 and aborts
during warmup with `block_kv == 32 or block_kv == 64`. Every backend sharing the KV
cache group then has to *accept* 64; this tree advertised 128 on SM12x, which is what
`patch_v41_kernel_block_size.py` fixes. The sliding-window pool is separate: the model
constructor builds 32-token pages, which only the FlashMLA path accepts, and
FlashInfer's SM120 decoder wants 64 with no fallback — so the build forces 64. See
`RESEARCH.md` §11.6, §11.9.15 and §11.9.16.

**The 32-state indexer page block is the one that kills the engine.** `--block-size 64`
lands the indexer on `{64, 32}`. The metadata builder accepts both, but the FP8 logits
kernel accepts only 64, so the engine dies on the first long prompt with an assertion
in `attention.hpp:320` and no block size avoids it. `overlay/sm12x_mqa.py` supplies a
Triton paged-MQA logits kernel for arch 12 that agrees with FP32 reference within
3.7e-05 and with the DeepGEMM path within 7.8e-07, and `patch_sm12x_indexer.py` hooks it
in before `_lazy_init()`. It fails closed: arch 12 only, FP8 only, 4D packed pages only,
and only when `block_kv != 64`.

**The AOT binary has to be masked.** FlashInfer's `build_and_load()` returns the
prebuilt `.so` whenever it is present and silently ignores a patched `.cu`. Once masked,
the compiled module becomes a required artifact: `scripts/build_flashinfer.sh` builds it
once per node into a persistent bind, and preflight verifies a receipt (image id,
version, path, size, digest) on both nodes. A matching filename proves nothing — a
module keyed on another FlashInfer version is never looked up, and an empty file matches
any glob.

**`--async-scheduling` is a tri-state, and omitting it means enabled.** On this vLLM
line `None` resolves to *on* for a generative model on the `mp` executor. Async doubles
the sliding-window reservation and so halves the reported KV pool, which is the wrong
trade on a KV-starved two-node fit, so the launcher always emits an explicit spelling
and defaults to off.

**`--tool-call-parser` is inert without `--enable-auto-tool-choice`**, and it fails
silently: the parser is registered, nothing parses, no error is raised. The launcher
sets both. The chain-of-thought field is `reasoning`.

**Keep `MIN_FREE_GIB` at 90.** On GB10 the GPU allocator and the page cache share one
pool, and the node hard-resets as `MemAvailable` approaches zero. There is no OOM kill
and no log line afterwards. The gate reads both nodes and fails closed if either reading
is unavailable. This floor is also what makes 64K context survivable.

**Never set `--memory` on the container.** On unified memory it caps GPU allocations
too, and any arena auto-sizer silently shrinks to fit it.

**Set the UCX registration-cache limits.** `UCX_MEM_MMAP_HOOK_MODE=none` and
`UCX_RCACHE_MAX_UNRELEASED=1024`. UCX's registration cache grows without bound by
default, and because host RAM and GPU memory are one pool here, that leak drains memory
until the node wedges. `MIN_FREE_GIB` cannot catch it — it samples once at start. The
launcher also sets `--oom-score-adj 500`, so an overrun reclaims the engine instead of
starving `sshd`.

**Keep every cache inside the persistent bind.** The container is `--rm`, so Triton,
TorchInductor, XDG and vLLM cache roots all point into `/root/.cache/vllm`. `MAX_JOBS=2`
in the image, because a parallel JIT fan-out competes with the page cache for the same
pool.

**Worker probes are serialized before they go over SSH.** OpenSSH joins its arguments
with spaces and the remote shell re-parses them, so a Go template containing a space
must be quoted for the *remote* shell. Getting this wrong makes `docker inspect` fail
and the interlock refuse every start; `tests/test_gpu_interlock.sh` guards it.

## First boot

1. Pull the base image on **both** nodes:
   `docker pull --platform linux/arm64 vllm/vllm-openai:deepseekv41-flash-0909`
2. **Stop whatever is currently serving on the nodes.** This kit will not do it for
   you, and two models cannot share 121 GiB per node.
3. Stage the weights with `scripts/fetch_weights.sh` on the node that will download
   (~386 GiB). The REAP repo publishes shards 1–46; **shards 47 and 48 come from the
   base `deepseek-ai/DeepSeek-V4.1-Flash`**, and the script stages them into the same
   snapshot under the same names, then runs `scripts/verify_weights.py`. It resolves
   exactly the path `start.sh` mounts; `start.sh` never downloads anything itself.
   Re-running it is safe and cheap — it fetches only what is missing.
4. Copy the checkpoint to the other node over the rail instead of downloading it twice,
   **as the worker's own account and at the head's own path**, because both ranks mount
   the head's path:
   `rsync -a --info=progress2 ~/.cache/huggingface/hub/models--LibertAIDAI--DeepSeek-V4.1-Flash-REAP-256E/ nvidia@192.168.177.11:/home/nvidia/.cache/huggingface/hub/models--LibertAIDAI--DeepSeek-V4.1-Flash-REAP-256E/`
   Preflight refuses if the worker cannot see `config.json` at that path.
5. `cp env.example .env` and keep the QSFP pins. Leave `QUANTIZATION` empty: the
   checkpoint declares its own quantization, and the launcher refuses a mismatch rather
   than failing late inside the loader.
6. `docker build --platform linux/arm64 -t dsv41-reap-selfbuild .` on **both** nodes.
7. `scripts/build_flashinfer.sh` on **both** nodes, in the same window and with the
   nodes free. It compiles the patched module into the mounted cache and writes the
   receipt beside it.
8. `./start.sh preflight && ./start.sh start && ./start.sh smoke`

Preflight checks the worker is reachable, the GPU is exclusive, memory is above the
floor, the fabric addresses are right, the worker can see the checkpoint, the
quantization matches, the drafter's block size matches the checkpoint, and the
FlashInfer module and page blocks are usable — all before any container starts, and each
refusal names its own cause.

Thinking must be **off** for the smoke test (`17*19` → `323`): V4.1 with thinking on and
a small `max_tokens` returns empty content.

The smoke also requests **logprobs** and fails on any non-finite value. That is not
decoration. This checkpoint can serve output that reads fluently and is NaN all the way
down, with a healthy throughput number attached — a text-only smoke passes straight
through it. Python's `json` parses bare `NaN`, so the guard tests the values rather than
trusting the parse to fail. `--skip-logprobs` turns it off and says so.

## Ramping past the first boot

`FIRST_BOOT=1` (the default) holds a deliberately conservative shape:
`MAX_NUM_SEQS=2`, `--enforce-eager`, `--language-model-only`, `SPEC_METHOD=none`. These
are a first-boot posture, not a permanent configuration; the point is to make the first
multi-minute weight load predictable. `FIRST_BOOT=0` lifts exactly those four and
nothing else.

- `ENFORCE_EAGER=0` omits `--enforce-eager`; `LANGUAGE_MODEL_ONLY=0` omits
  `--language-model-only`. Both are store-true flags, so the false side is expressed by
  omission. Do not turn graphs on for this checkpoint — see
  [above](#eager-is-a-correctness-posture-here-not-an-oversight).
- `SPEC_METHOD=dspark` emits `--speculative-config` with `SPEC_NUM_TOKENS`. `none` is
  the default, and **every other value fails closed** rather than being silently
  dropped. `mtp` is rejected on purpose: V4.1 ships DSpark stages under the `mtp.*`
  prefix and has no classic-MTP draft, so the engine raises. Preflight validates the
  value too.
- **`SPEC_METHOD=dspark` with `SPEC_NUM_TOKENS=5` is the measured winner: 27.75 tok/s
  against a same-day 13.80 tok/s control, a 2.01× speedup, coherent and NaN-free, at a
  0.4246 acceptance ratio.**

### Why the block is 5 and not any number

`num_speculative_tokens` is not free: the engine quantises it to the checkpoint's own
`dspark_block_size`, read from `text_config`. This checkpoint's drafter is a block of
**5** (2,401 `mtp.*` tensors, `dspark_target_layer_ids=[37,38,39]`), so only multiples
of 5 behave as asked; `6`, `7`, `8`, `9` raise at startup, and `10` is two blocks.
`SPEC_DSPARK_BLOCK` must match the checkpoint, and preflight's `check_dspark_draft`
refuses a mismatch, a declared drafter with no weights, and a checkpoint with no drafter
at all — the failure it prevents is a silent fall back to no speculation, which reads
as "speculation is not a win here".

`SPEC_DRAFT_SAMPLE_METHOD` is `greedy` by default. `probabilistic` is upstream's other
spelling and is **opt-in**. It measured the same as greedy in our bench, but our bench
runs a greedy target at `temperature: 0`, where the two are indistinguishable — see
[above](#the-acceptance-ratio-is-deterministic-and-that-is-worth-knowing). It stays
opt-in because its draft can degenerate into loops while inflating its own acceptance
rate, which is exactly the failure a throughput number cannot see.

`SPEC_ADAPTIVE_VERIFICATION` defaults to `0` because the engine cannot support it
without full CUDA graphs, which this checkpoint cannot use.

### Other knobs

- The memory floor, the GPU-exclusivity interlock, the quantization check and the
  FlashInfer checks stay on in both modes. A `FIRST_BOOT` that is neither `0` nor `1`
  fails closed rather than reading as "not first boot".
- `MAX_NUM_BATCHED_TOKENS` defaults to 8192. A cold 64K prompt needs 64 steps to enter
  at 1024 and 8 at 8192, which makes this the cheapest prefill lever here. It costs KV
  blocks, so raise it together with `MIN_FREE_GIB`, not alone.
- `LONG_PREFILL_TOKEN_THRESHOLD=4096` caps each running long-prefill chunk so one cold
  64K prompt cannot stall short requests behind it. `0` disables it.
- `MAX_NUM_SEQS=2` is what the measurements above were taken at. Raising it is the most
  promising untested lever; expect it to cost KV pool.

Do not copy a `--gpu-memory-utilization 0.90` setting from elsewhere without thinking:
it measures the same cliff as `MIN_FREE_GIB=90` from the other side. This kit stays
below it deliberately.

## The A/B protocol

Speculative arms are measured with `scripts/ab_spec.sh`, one boot per arm, against a
same-day control, with the engine's own acceptance counters and a coherence gate.
[`bench/README.md`](bench/README.md) documents the four rules and what the coherence
check does and does not prove.

```bash
./scripts/ab_spec.sh list
./scripts/ab_spec.sh run baseline     # control, no speculation
./scripts/ab_spec.sh run greedy5      # stop, re-arm, start, smoke, measure
./scripts/ab_spec.sh report
```

Receipts land in `results/ab-spec/<arm>.json` and are committed, because a row that
cannot be inspected after the fact is an anecdote.

## What is next

This is a beginning, not a finished thing. Three directions, in the order we intend to
attack them:

1. **Prefill.** A cold 38K prompt costs a newcomer 47.9 s of TTFT, and a running decode
   stream drops to 2.6–5.6% of its rate while that happens. The long-prefill chunk cap
   is a mitigation, not a fix.
2. **Concurrency.** Aggregate throughput plateaus at ~22 tok/s because
   `MAX_NUM_SEQS=2` refuses to admit requests 3 and 4. There is headroom here and it is
   the cheapest win on the list.
3. **Decode.** DSpark got 2.01×. Block 10 was worse, not better, so the obvious knob is
   spent — the remaining levers are kernel-level, not configuration.

Nothing here is a promise about dates. Every number will come with its receipt.

## Current state on the cluster

- Weights: **staged and verified on both nodes** — 385.5 GiB, 48/48 shards, every
  indexed key present and every shard whole.
- Images: **built on both nodes**; all in-image test suites pass on both.
- FlashInfer: **prebuilt on both nodes**, receipts verified by preflight.
- Preflight: **passes on both nodes**, including the SM12x FP8-logits fallback check.
- Serving: **booted, smoke-passed, needle-recall verified to 64,851 prompt tokens**.
- Bench: five of six sibling suites run. `conc-60k` returns HTTP 400, which is correct —
  the request's own filler plus system prompt exceeds the 65,536-token window.

## Credits and licenses

This kit is a thin layer of tooling, preflight checks and patches on top of other
people's work. Here is what each piece is and what it is licensed under, checked
against the sources on 2026-09-12:

| Piece | What it is | License |
|---|---|---|
| [LibertAIDAI/DeepSeek-V4.1-Flash-REAP-256E](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-REAP-256E) | the checkpoint this kit serves | **MIT** (`license: mit`) |
| [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | the base model the checkpoint inherits from | **MIT** |
| [Libertai/deepseek-v41-flash-reap](https://github.com/Libertai/deepseek-v41-flash-reap) | the pruning recipe, calibration and every measurement behind the checkpoint | **MIT** (Copyright (c) 2026 LibertAI) |
| [CerebrasResearch/reap](https://github.com/CerebrasResearch/reap) | the REAP method itself ([arXiv:2510.13999](https://arxiv.org/abs/2510.13999)) | **Apache-2.0** |
| [vllm-project/vllm](https://github.com/vllm-project/vllm) | the inference engine; this kit overlays a pinned build | **Apache-2.0** |
| [flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer) | the attention kernels; patched by this kit | **Apache-2.0** |
| this kit's own tooling | the launcher, patches, preflights, bench and harness | **MIT** — see `LICENSE` |

Remarks, in plain terms:

- **Both licenses permit commercial use, redistribution and modification.** MIT requires
  the copyright and permission notice to travel with copies. Apache-2.0 additionally
  requires modified files to carry prominent notices of change, and requires any NOTICE
  file to be preserved.
- **This kit modifies vLLM and FlashInfer source at build time** (`overlay/patch_*.py`).
  Those are Apache-2.0 works. The patches are applied inside your own image build rather
  than shipped as a modified tree, so the notice obligation lands on the patched files
  in that image. Each patch script carries a header naming the file it edits and the
  change it makes.
- **The kit's own tooling is MIT**, matching the vendor recipe's own choice
  (`Tooling: MIT. The model and its bundled reference implementation keep their own
  license.`).
- **The base image is redistributed by vLLM under Apache-2.0**; the checkpoint is not
  bundled inside it, and you download it separately from Hugging Face.
- **MIT on frontier model weights is unusual and worth your own attention.** DeepSeek
  licensed V4.1-Flash under MIT and LibertAI inherited it, which is unusually permissive
  for a model of this size. We read the cards and the LICENSE files; we are not lawyers,
  and **none of this is legal advice**. Confirm the terms yourself before you build a
  product on it.
- **REAP is cited, not vendored.** No Cerebras code is copied into this kit. The pruning
  was performed by the checkpoint's publisher and is described here for context.

If you find a licensing error in this table, open an issue — it is the kind of thing we
would rather be corrected on than left wrong.
