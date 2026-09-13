# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Vendored FP8 paged MQA-logits Triton kernels for DeepGEMM's arch-12 gap.

Scoped, byte-faithful vendoring of the paged (decode) FP8 MQA-logits Triton
kernels from vLLM PR #41834 ("SM12x support", author jasl):

    repository:  https://github.com/vllm-project/vllm  (PR #41834)
    source file: vllm/models/deepseek_v4/nvidia/ops/sm12x_mqa.py
    revision:    9ad62027bc84ca0ccbcc40853179312de770220c  (branch pr41834-sm12x)

Why this file exists
--------------------
DeepGEMM's FP8 MQA-logits kernel for arch 12 (SM120/SM121, i.e. the GB10 in the
DGX Spark) does not accept the indexer's cache geometry: the compressed page
holds fewer FP8 states than that kernel requires, so the packed-page cache
cannot be fed to it. These Triton kernels read the packed ``uint8`` page
storage directly (FP8 values followed by per-state fp32 scales) and are used
only when that geometry applies. They are a fallback, not a replacement: the
DeepGEMM path stays the default wherever it accepts the shapes.

Scope
-----
Only the paged/decode path is vendored:

  * ``_bucketed_logits_buffer``
  * ``_view_packed_fp8_paged_mqa_kv_cache``
  * ``_fp8_paged_mqa_logits_kernel``            (generic 2D-tiled paged kernel)
  * ``_fp8_paged_mqa_logits_rowwise_kernel``    (per-row paged kernel)
  * ``fp8_paged_mqa_logits_rowwise_triton``
  * ``fp8_paged_mqa_logits_triton``             (paged dispatcher)

The contiguous/prefill kernels (``_fp8_mqa_logits_kernel``,
``fp8_mqa_logits_triton``) are intentionally NOT vendored: the contiguous path
does not take its ``block_kv`` from the cache geometry, so it never hits the
arch-12 restriction and stays on DeepGEMM. The HyperConnection TF32 prenorm
GEMM and all direct-top-k / fused top-k code are likewise unrelated and
omitted.

The function bodies below are copied verbatim from the revision above; do not
edit them in place. The only addition is ``fp8_paged_mqa_logits_sm12x`` at the
end of the file, a thin overlay entry point that does not shadow the vendored
``fp8_paged_mqa_logits_triton``.
"""

import torch

from vllm.triton_utils import tl, triton


def _bucketed_logits_buffer(
    num_rows: int, row_width: int, device: torch.device
) -> torch.Tensor:
    """fp32 (num_rows, row_width) output, allocated in power-of-two buckets.

    During a chunked prefill the logits width (the compressed KV length) grows
    monotonically with every chunk, so a fresh exact-size ``torch.empty`` per
    call means no request can ever be served from a cached block: the caching
    allocator maps a new segment each chunk and ``memory_reserved`` ratchets
    toward the SUM of the distinct sizes. On unified-memory devices (GB10)
    that reserve is system RAM -- ~2.6-3 GiB per 32K needle prompt
    (jasl/vllm#31). Rounding the flat allocation up to a power of two makes
    consecutive chunks share a bucket, bounding the transient footprint at
    ~2x the largest live buffer instead of the running sum. The trailing view
    keeps the tensor contiguous, so kernel stride assumptions are unchanged.
    """
    numel = num_rows * row_width
    if numel == 0:
        return torch.empty((num_rows, row_width), device=device, dtype=torch.float32)
    alloc = 1 << (numel - 1).bit_length()
    flat = torch.empty((alloc,), device=device, dtype=torch.float32)
    return flat[:numel].view(num_rows, row_width)


def _view_packed_fp8_paged_mqa_kv_cache(
    kv_cache: torch.Tensor,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return FP8 values and fp32 scales from indexer cache block storage."""
    if kv_cache.dtype != torch.uint8:
        raise TypeError(f"Expected uint8 kv_cache, got {kv_cache.dtype}")
    if kv_cache.dim() == 3:
        num_blocks, block_size, head_dim_with_scale = kv_cache.shape
        num_kv_heads = 1
    elif kv_cache.dim() == 4:
        num_blocks, block_size, num_kv_heads, head_dim_with_scale = kv_cache.shape
    else:
        raise ValueError(f"Expected 3D or 4D kv_cache, got {kv_cache.dim()} dimensions")
    if num_kv_heads != 1:
        raise ValueError(f"Expected one KV head, got {num_kv_heads}")

    scale_bytes = head_dim_with_scale - head_dim
    if scale_bytes <= 0 or scale_bytes % torch.float32.itemsize != 0:
        raise ValueError(
            "Expected kv_cache last dimension to contain FP8 values followed "
            f"by fp32 scale bytes; got head_dim={head_dim}, "
            f"last_dim={head_dim_with_scale}"
        )

    block_stride = kv_cache.stride(0)
    base_storage_offset = kv_cache.storage_offset()
    scale_elems = scale_bytes // torch.float32.itemsize
    kv_values = torch.as_strided(
        kv_cache,
        size=(num_blocks, block_size, 1, head_dim),
        stride=(block_stride, head_dim, head_dim, 1),
        storage_offset=base_storage_offset,
    ).view(torch.float8_e4m3fn)
    kv_scale = torch.as_strided(
        kv_cache,
        size=(num_blocks, block_size, 1, scale_bytes),
        stride=(block_stride, scale_bytes, scale_bytes, 1),
        storage_offset=base_storage_offset + block_size * head_dim,
    ).view(torch.float32)
    return kv_values, kv_scale[..., :scale_elems]


@triton.jit
def _fp8_paged_mqa_logits_kernel(
    q_ptr,
    kv_ptr,
    scale_ptr,
    weights_ptr,
    context_lens_ptr,
    block_tables_ptr,
    logits_ptr,
    token_start,
    num_rows,
    logits_width,
    next_n: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    stride_qb: tl.constexpr,
    stride_qn: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kvb: tl.constexpr,
    stride_kvs: tl.constexpr,
    stride_kvd: tl.constexpr,
    stride_sb: tl.constexpr,
    stride_ss: tl.constexpr,
    stride_wm: tl.constexpr,
    stride_wh: tl.constexpr,
    stride_clb: tl.constexpr,
    stride_cln: tl.constexpr,
    stride_btb: tl.constexpr,
    stride_btk: tl.constexpr,
    stride_lm,
    stride_ln: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_local_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_n = token_start + offs_local_n
    offs_d = tl.arange(0, BLOCK_D)

    valid_m = offs_m < num_rows
    valid_n = offs_local_n < logits_width
    batch = offs_m // next_n
    q_pos = offs_m - batch * next_n
    context_len = tl.load(
        context_lens_ptr + batch * stride_clb + q_pos * stride_cln,
        mask=valid_m,
        other=0,
    )
    context_mask = valid_n[None, :] & (offs_n[None, :] < context_len[:, None])

    block_rank = offs_n // block_size
    block_offset = offs_n - block_rank * block_size
    block_idx = tl.load(
        block_tables_ptr
        + batch[:, None] * stride_btb
        + block_rank[None, :] * stride_btk,
        mask=valid_m[:, None] & valid_n[None, :],
        other=0,
    )

    logits = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    scale = tl.load(
        scale_ptr
        + block_idx.to(tl.int64) * stride_sb
        + block_offset[None, :] * stride_ss,
        mask=context_mask,
        other=0.0,
    )
    for h in tl.range(0, num_heads):
        scores = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for d0 in tl.range(0, head_dim, BLOCK_D):
            d = d0 + offs_d
            q = tl.load(
                q_ptr
                + batch[:, None] * stride_qb
                + q_pos[:, None] * stride_qn
                + h * stride_qh
                + d[None, :] * stride_qd,
                mask=valid_m[:, None] & (d[None, :] < head_dim),
                other=0.0,
            ).to(tl.float32)
            k = tl.load(
                kv_ptr
                + block_idx[:, :, None].to(tl.int64) * stride_kvb
                + block_offset[None, :, None] * stride_kvs
                + d[None, None, :] * stride_kvd,
                mask=context_mask[:, :, None] & (d[None, None, :] < head_dim),
                other=0.0,
            ).to(tl.float32)
            scores += tl.sum(q[:, None, :] * k, axis=2)
        weighted = tl.maximum(scores * scale, 0.0)
        weight = tl.load(
            weights_ptr + offs_m * stride_wm + h * stride_wh,
            mask=valid_m,
            other=0.0,
        )
        logits += weighted * weight[:, None]

    store_mask = valid_m[:, None] & valid_n[None, :]
    logits = tl.where(context_mask & store_mask, logits, float("-inf"))
    tl.store(
        logits_ptr + offs_m[:, None] * stride_lm + offs_local_n[None, :] * stride_ln,
        logits,
        mask=store_mask,
    )


@triton.jit
def _fp8_paged_mqa_logits_rowwise_kernel(
    q_ptr,
    kv_ptr,
    scale_ptr,
    weights_ptr,
    context_lens_ptr,
    block_tables_ptr,
    logits_ptr,
    token_start,
    num_rows,
    logits_width,
    next_n: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    stride_qb: tl.constexpr,
    stride_qn: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kvb: tl.constexpr,
    stride_kvs: tl.constexpr,
    stride_kvd: tl.constexpr,
    stride_sb: tl.constexpr,
    stride_ss: tl.constexpr,
    stride_wm: tl.constexpr,
    stride_wh: tl.constexpr,
    stride_clb: tl.constexpr,
    stride_cln: tl.constexpr,
    stride_btb: tl.constexpr,
    stride_btk: tl.constexpr,
    stride_lm,
    stride_ln: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Per-row paged-MQA logits kernel optimised for long ``token_count``.

    Each Triton program handles one logical row (``batch * next_n + q_pos``)
    across a ``BLOCK_N``-wide window of token positions. Q is loaded once per
    head tile and reused for every K element in the window, which preserves
    L2 / register locality and avoids the M-axis padding waste of the
    generic 2D-tiled kernel at long contexts (mt-bench c=1 MTP=2 num_rows=3
    with token_count=131072 launches 12k programs of 128 logits each rather
    than 8k programs of 64 logits with 25 % M-axis waste).
    """
    row = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_local_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_n = token_start + offs_local_n
    offs_d = tl.arange(0, BLOCK_D)

    valid_row = row < num_rows
    valid_n = offs_local_n < logits_width
    batch = row // next_n
    q_pos = row - batch * next_n
    context_len = tl.load(
        context_lens_ptr + batch * stride_clb + q_pos * stride_cln,
        mask=valid_row,
        other=0,
    )
    if token_start + pid_n * BLOCK_N >= context_len:
        logits = tl.full((BLOCK_N,), float("-inf"), dtype=tl.float32)
        tl.store(
            logits_ptr + row * stride_lm + offs_local_n * stride_ln,
            logits,
            mask=valid_row & valid_n,
        )
        return
    context_mask = valid_n & (offs_n < context_len)

    block_rank = offs_n // block_size
    block_offset = offs_n - block_rank * block_size
    block_idx = tl.load(
        block_tables_ptr + batch * stride_btb + block_rank * stride_btk,
        mask=valid_row & context_mask,
        other=0,
    )

    scale = tl.load(
        scale_ptr + block_idx.to(tl.int64) * stride_sb + block_offset * stride_ss,
        mask=context_mask,
        other=0.0,
    )
    logits = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for h0 in tl.range(0, num_heads, BLOCK_H):
        heads = h0 + tl.arange(0, BLOCK_H)
        valid_h = heads < num_heads
        scores = tl.zeros((BLOCK_H, BLOCK_N), dtype=tl.float32)
        for d0 in tl.range(0, head_dim, BLOCK_D):
            d = d0 + offs_d
            q = tl.load(
                q_ptr
                + batch * stride_qb
                + q_pos * stride_qn
                + heads[:, None] * stride_qh
                + d[None, :] * stride_qd,
                mask=valid_row & valid_h[:, None] & (d[None, :] < head_dim),
                other=0.0,
            ).to(tl.float32)
            k = tl.load(
                kv_ptr
                + block_idx[None, :].to(tl.int64) * stride_kvb
                + block_offset[None, :] * stride_kvs
                + d[:, None] * stride_kvd,
                mask=context_mask[None, :] & (d[:, None] < head_dim),
                other=0.0,
            ).to(tl.float32)
            scores += tl.dot(q, k, input_precision="tf32")

        weighted = tl.maximum(scores * scale[None, :], 0.0)
        weight = tl.load(
            weights_ptr + row * stride_wm + heads * stride_wh,
            mask=valid_row & valid_h,
            other=0.0,
        )
        logits += tl.sum(weighted * weight[:, None], axis=0)

    logits = tl.where(context_mask & valid_row, logits, float("-inf"))
    tl.store(
        logits_ptr + row * stride_lm + offs_local_n * stride_ln,
        logits,
        mask=valid_row & valid_n,
    )


def fp8_paged_mqa_logits_rowwise_triton(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
    token_start: int = 0,
    token_count: int | None = None,
) -> torch.Tensor:
    """Rowwise paged-MQA logits wrapper.

    Pre-condition: ``head_dim % 64 == 0`` and ``num_heads % 4 == 0`` so the
    ``tl.dot`` inside ``_fp8_paged_mqa_logits_rowwise_kernel`` lands on
    tensor-core friendly tile shapes. DSv4-Flash (head_dim=128,
    num_heads=64) satisfies both and is the only model that exercises this
    path today; the generic 2D kernel below remains the fallback for
    misaligned shapes.
    """
    batch_size, next_n, num_heads, head_dim = q.size()
    kv_values, kv_scale = _view_packed_fp8_paged_mqa_kv_cache(kv_cache, head_dim)
    _, block_size, _, _ = kv_values.size()
    num_rows = batch_size * next_n
    if token_count is None:
        token_count = max_model_len - token_start
    assert token_start >= 0
    assert token_count >= 0
    assert token_start + token_count <= max_model_len
    logits = _bucketed_logits_buffer(num_rows, token_count, q.device)
    if num_rows == 0 or token_count == 0:
        return logits

    context_lens_2d = context_lens.reshape(batch_size, -1)
    if context_lens_2d.shape[1] == 1 and next_n != 1:
        context_lens_2d = context_lens_2d.expand(batch_size, next_n).contiguous()
    block_n = 128
    grid = (num_rows, triton.cdiv(token_count, block_n))
    _fp8_paged_mqa_logits_rowwise_kernel[grid](
        q,
        kv_values,
        kv_scale,
        weights,
        context_lens_2d,
        block_tables,
        logits,
        token_start,
        num_rows,
        token_count,
        next_n,
        num_heads,
        head_dim,
        block_size,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        kv_values.stride(0),
        kv_values.stride(1),
        kv_values.stride(3),
        kv_scale.stride(0),
        kv_scale.stride(1),
        weights.stride(0),
        weights.stride(1),
        context_lens_2d.stride(0),
        context_lens_2d.stride(1),
        block_tables.stride(0),
        block_tables.stride(1),
        logits.stride(0),
        logits.stride(1),
        BLOCK_N=block_n,
        BLOCK_D=64,
        BLOCK_H=8,
        num_warps=4,
    )
    return logits


def fp8_paged_mqa_logits_triton(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
    token_start: int = 0,
    token_count: int | None = None,
) -> torch.Tensor:
    batch_size, next_n, num_heads, head_dim = q.size()
    # Aligned head shapes (DSv4-Flash and any future MQA model with
    # ``head_dim % 64 == 0`` and ``num_heads % 4 == 0``) get the rowwise
    # kernel, which keeps long-context decode (>100K tokens) on a per-row
    # grid that re-uses Q across the full token window. The generic 2D
    # kernel below still handles misaligned shapes and remains the canonical
    # reference for the rowwise variant.
    if head_dim % 64 == 0 and num_heads % 4 == 0:
        return fp8_paged_mqa_logits_rowwise_triton(
            q,
            kv_cache,
            weights,
            context_lens,
            block_tables,
            max_model_len,
            token_start=token_start,
            token_count=token_count,
        )

    kv_values, kv_scale = _view_packed_fp8_paged_mqa_kv_cache(kv_cache, head_dim)
    _, block_size, _, _ = kv_values.size()
    num_rows = batch_size * next_n
    if token_count is None:
        token_count = max_model_len - token_start
    assert token_start >= 0
    assert token_count >= 0
    assert token_start + token_count <= max_model_len
    logits = _bucketed_logits_buffer(num_rows, token_count, q.device)
    if num_rows == 0 or token_count == 0:
        return logits

    context_lens_2d = context_lens.reshape(batch_size, -1)
    if context_lens_2d.shape[1] == 1 and next_n != 1:
        context_lens_2d = context_lens_2d.expand(batch_size, next_n).contiguous()
    # Adaptive BLOCK_M: the kernel masks off positions >= num_rows, so a fixed
    # BLOCK_M=4 wastes ~75% of M-axis work in the common single-stream decode
    # case (num_rows=1). Pick the smallest power-of-2 tile that still covers
    # num_rows so we keep one grid-program for typical decode while still
    # benefiting from larger tiles when batch / MTP push num_rows higher.
    if num_rows <= 1:
        block_m = 1
    elif num_rows <= 2:
        block_m = 2
    elif num_rows <= 4:
        block_m = 4
    else:
        block_m = 8
    grid = (triton.cdiv(num_rows, block_m), triton.cdiv(token_count, 64))
    _fp8_paged_mqa_logits_kernel[grid](
        q,
        kv_values,
        kv_scale,
        weights,
        context_lens_2d,
        block_tables,
        logits,
        token_start,
        num_rows,
        token_count,
        next_n,
        num_heads,
        head_dim,
        block_size,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        kv_values.stride(0),
        kv_values.stride(1),
        kv_values.stride(3),
        kv_scale.stride(0),
        kv_scale.stride(1),
        weights.stride(0),
        weights.stride(1),
        context_lens_2d.stride(0),
        context_lens_2d.stride(1),
        block_tables.stride(0),
        block_tables.stride(1),
        logits.stride(0),
        logits.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=64,
        BLOCK_D=64,
        num_warps=4,
    )
    return logits


def fp8_paged_mqa_logits_sm12x(
    q_values: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    """Thin entry point used by the overlay dispatch. See module docstring.

    ``q_values`` must be the FP8 quantized q (no scale); ``kv_cache`` must be
    the packed uint8 page storage (FP8 values followed by fp32 per-state
    scales). The returned logits have shape ``(batch * next_n, max_model_len)``
    with ``-inf`` outside each row's valid interval.
    """
    return fp8_paged_mqa_logits_triton(
        q_values, kv_cache, weights, context_lens, block_tables, max_model_len
    )
