# SPDX-License-Identifier: Apache-2.0
"""Install NVMe Engram hooks onto this tree's V4.1 modules.

Does not rewrite `engram.py` as text. Wraps:

- `Engram.__init__` so the constructing layer id is visible,
- `ParallelEngramEmbedding.__init__` so the 189–203 GiB tables become
  1-row placeholders with `cpu_offload=False`,
- `ParallelEngramEmbedding.lookup` so gather is `pread` + CPU dequant + H2D,
- the safetensors iterators so `.engram.embed.weight/.scale` are never read.

Pin/UVA remains available if `DSV41_ENGRAM_DISK=0`, but the Spark launcher
refuses that path: pinned Engram is the GB10 UMA trap.
"""

from __future__ import annotations

import contextvars
from collections.abc import Generator, Iterable
from typing import Any

import torch
from torch import nn

from vllm.logger import init_logger

from engram_disk import (
    ENGRAM_DISK_ENV,
    EngramDiskTable,
    disk_offload_enabled,
    is_engram_table_tensor,
    resolve_model_dir,
    skip_engram_tables,
)

logger = init_logger(__name__)

_LAYER_ID: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "dsv41_engram_layer_id", default=None
)
_INSTALLED = False


def _disk_placeholder_loader(
    param: torch.nn.Parameter, loaded_weight: torch.Tensor
) -> None:
    del param, loaded_weight
    return None


def _wrap_engram_init(original):
    def wrapped(
        self,
        config,
        quant_config,
        layout,
        layer_hash_index: int,
        use_sequence_parallel: bool,
        prefix: str,
    ):
        layer_id = layout.layer_ids[layer_hash_index]
        token = _LAYER_ID.set(int(layer_id))
        try:
            original(
                self,
                config,
                quant_config,
                layout,
                layer_hash_index,
                use_sequence_parallel,
                prefix,
            )
        finally:
            _LAYER_ID.reset(token)
        embed = self.embed_tokens
        embed.cpu_offload = False
        embed._views = None
        embed._view_src = None
        if disk_offload_enabled() and not hasattr(embed, "_disk_table"):
            raise RuntimeError(
                f"Engram disk table missing after init for layer {layer_id}; "
                f"set {ENGRAM_DISK_ENV}=1 and DSV41_MODEL_DIR to the checkpoint"
            )
        return None

    return wrapped


def _resolve_engram_sharding() -> tuple[int, int, int]:
    """`(tp_size, dp_size, head_shard_rank)` as the installed vLLM defines them.

    The installed class splits the hash heads over `TP x DP`, where DP is the
    group of co-located replicas that share one table. `get_engram_dp_size` and
    `engram_head_shard_rank` do not exist on the pinned tree, whose window is
    TP-only; the fallbacks reproduce that arithmetic exactly. Both are looked up
    on the module rather than imported, so the overlay loads on either tree.
    """
    from vllm.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )
    from vllm.models.deepseek_v4_1.common import engram as engram_mod

    tp_size = int(get_tensor_model_parallel_world_size())
    dp_size = 1
    head_shard_rank = int(get_tensor_model_parallel_rank())

    get_dp_size = getattr(engram_mod, "get_engram_dp_size", None)
    if get_dp_size is not None:
        dp_size = int(get_dp_size())
    head_shard = getattr(engram_mod, "engram_head_shard_rank", None)
    if head_shard is not None:
        head_shard_rank = int(head_shard())
    return tp_size, dp_size, head_shard_rank


def _engram_shard_window(
    n_hash_cols: int,
    head_sizes: tuple[int, ...],
    tp_size: int,
    dp_size: int,
    head_shard_rank: int,
) -> tuple[int, int, int, int]:
    """Rows this rank owns under the stock `TP x DP` head split.

    Mirrors `ParallelEngramEmbedding.__init__`: `n_hash_cols` heads are split
    over `tp_size * dp_size` shards, and the shard index is TP-major
    (`tp_rank * dp_size + dp_rank`), so a DP gather brings in contiguous heads
    and the following TP gather completes the head order. The NVMe table must
    cover exactly the rows stock would have owned, or the gather returns the
    wrong heads.

    Returns `(part_n_hash_cols, head_start, vocab_start, vocab_end)`.
    """
    num_shards = tp_size * dp_size
    # Same as `triton.cdiv`; kept as integer math so the window is testable
    # without a Triton import.
    part_n_hash_cols = (n_hash_cols + num_shards - 1) // num_shards
    if (num_shards - 1) * part_n_hash_cols >= n_hash_cols:
        raise ValueError(
            f"Engram sharding leaves ranks without hash heads: "
            f"{n_hash_cols} heads over TP={tp_size} x DP={dp_size}"
        )
    head_start = head_shard_rank * part_n_hash_cols
    head_end = head_start + part_n_hash_cols
    return (
        part_n_hash_cols,
        head_start,
        sum(head_sizes[:head_start]),
        sum(head_sizes[:head_end]),
    )


def _init_disk_embed(
    self,
    num_embeddings: int,
    dim: int,
    head_sizes: tuple[int, ...],
    block_size: int,
) -> None:
    """Same TP x DP window as stock ParallelEngramEmbedding, 1-row params.

    Do not call the stock constructor: it would allocate the full shard on
    CUDA (cpu_offload=False) or pin it in UMA (cpu_offload=True).
    """
    from vllm.model_executor.utils import set_weight_attrs

    nn.Module.__init__(self)
    layer_id = _LAYER_ID.get()
    if layer_id is None:
        raise RuntimeError(
            "ParallelEngramEmbedding constructed without an Engram layer id"
        )
    if not head_sizes or not all(size > 0 for size in head_sizes):
        raise ValueError("Engram head_sizes must be positive")
    if sum(head_sizes) > num_embeddings:
        raise ValueError(
            f"Engram head sizes sum to {sum(head_sizes)} > {num_embeddings} rows"
        )
    n_hash_cols = len(head_sizes)
    tp_size, dp_size, head_shard_rank = _resolve_engram_sharding()
    part_n_hash_cols, head_start, vocab_start, vocab_end = _engram_shard_window(
        n_hash_cols, head_sizes, tp_size, dp_size, head_shard_rank
    )
    self.num_embeddings = num_embeddings
    self.dim = dim
    self.block_size = block_size
    self.n_hash_cols = n_hash_cols
    self.part_n_hash_cols = part_n_hash_cols
    self.head_start = head_start
    self.vocab_start_idx = vocab_start
    self.vocab_end_idx = vocab_end
    self.part_num_embeddings = vocab_end - vocab_start
    self.tp_size = tp_size
    # `Engram.__init__` sizes its staging buffer by `embed_tokens.dp_size`, so
    # the attribute is part of the contract this replacement has to keep. It is
    # 1 whenever no Engram DP group exists (any `data_parallel_size == 1` run).
    self.dp_size = dp_size
    self.cpu_offload = False
    self._views = None
    self._view_src = None
    self._num_sms = torch.cuda.get_device_properties(
        torch.accelerator.current_device_index()
    ).multi_processor_count
    self.weight = nn.Parameter(
        torch.empty(1, dim, dtype=torch.float8_e4m3fn, device="cpu"),
        requires_grad=False,
    )
    self.weight_scale_inv = nn.Parameter(
        torch.empty(1, dim // block_size, dtype=torch.uint8, device="cpu"),
        requires_grad=False,
    )
    for param in (self.weight, self.weight_scale_inv):
        set_weight_attrs(
            param,
            {
                "weight_loader": _disk_placeholder_loader,
                "engram_vocab_start": self.vocab_start_idx,
            },
        )
    self._disk_table = EngramDiskTable(
        resolve_model_dir(),
        layer_id=int(layer_id),
        vocab_start=int(self.vocab_start_idx),
        vocab_end=int(self.vocab_end_idx),
        dim=int(dim),
        block_size=int(block_size),
    )
    logger.info(
        "Engram layer %s: NVMe lookup, placeholder params 1x%s, "
        "owned rows [%s, %s) of %s, heads [%s, %s) of %s over TP=%s x DP=%s",
        layer_id,
        dim,
        self.vocab_start_idx,
        self.vocab_end_idx,
        num_embeddings,
        self.head_start,
        self.head_start + self.part_n_hash_cols,
        self.n_hash_cols,
        self.tp_size,
        self.dp_size,
    )


def _wrap_embed_init(original):
    def wrapped(
        self,
        num_embeddings: int,
        dim: int,
        head_sizes: tuple[int, ...],
        block_size: int = 32,
        cpu_offload: bool = False,
    ):
        if disk_offload_enabled():
            _init_disk_embed(self, num_embeddings, dim, head_sizes, block_size)
            return
        original(
            self,
            num_embeddings,
            dim,
            head_sizes,
            block_size=block_size,
            cpu_offload=cpu_offload,
        )

    return wrapped


def _wrap_lookup(original):
    def wrapped(
        self,
        indices: torch.Tensor,
        out: torch.Tensor,
        background: bool = False,
    ) -> None:
        table = getattr(self, "_disk_table", None)
        if table is None:
            return original(self, indices, out, background)
        tokens = int(indices.shape[0])
        local_heads = int(self.part_n_hash_cols)
        if tokens == 0 or local_heads == 0:
            return None
        head_end = self.head_start + local_heads
        local_ids = indices[:, self.head_start : head_end].reshape(-1)
        gathered = table.gather_dequant(local_ids)
        dest = out[:tokens, :local_heads]
        dest.copy_(
            gathered.view(tokens, local_heads, self.dim).to(
                device=dest.device, dtype=dest.dtype, non_blocking=True
            )
        )
        return None

    return wrapped


def _wrap_iterator(original):
    def wrapped(*args, **kwargs):
        for name, tensor in original(*args, **kwargs):
            if disk_offload_enabled() and is_engram_table_tensor(name):
                continue
            yield name, tensor

    return wrapped


def _engram_placeholder():
    """Tiny stand-in so a direct `get_tensor` call never reads table bytes."""
    import torch

    return torch.empty(0, dtype=torch.uint8)


class _EngramFilteringHandle:
    """Hide Engram tables at the safetensors handle, before materialization.

    `safetensors.safe_open(...).get_tensor(name)` is where a row of the
    checkpoint actually enters process memory. vLLM's lazy iterator calls it
    for every key before yielding, so a post-yield filter is too late: the
    91 GiB table would already be resident. Removing the names from `keys()`
    keeps the stock iterator from ever reaching `get_tensor`, and the
    `get_tensor` guard covers a caller that looks the name up directly.
    """

    __slots__ = ("_handle",)

    def __init__(self, handle: Any) -> None:
        self._handle = handle

    def __getattr__(self, item: str) -> Any:
        return getattr(self._handle, item)

    def __enter__(self) -> "_EngramFilteringHandle":
        self._handle.__enter__()
        return self

    def __exit__(self, *exc: Any) -> Any:
        return self._handle.__exit__(*exc)

    def keys(self):
        return [n for n in self._handle.keys() if not is_engram_table_tensor(n)]

    def get_tensor(self, name: str):
        if is_engram_table_tensor(name):
            return _engram_placeholder()
        return self._handle.get_tensor(name)


def _wrap_safe_open(original):
    def wrapped(filename, framework="pt", device="cpu"):
        handle = original(filename, framework=framework, device=device)
        if not disk_offload_enabled():
            return handle
        return _EngramFilteringHandle(handle)

    return wrapped


def _engram_aware_multi_thread(
    hf_weights_files, use_tqdm_on_load, max_workers=4
):
    """Lazy substitute for the multi-thread loader.

    The stock multi-thread loader calls `load_file`, which reads whole shards
    into RAM before yielding, so Engram tables would be materialized. While
    disk offload is on, fall back to the per-shard lazy iterator, whose
    `safe_open` handle is filtered above.
    """
    from vllm.model_executor.model_loader.weight_utils import (
        safetensors_weights_iterator,
    )

    if disk_offload_enabled():
        logger.info(
            "Engram disk offload: using the lazy safetensors iterator instead of "
            "the multi-thread loader (whole-shard loads would materialize tables)."
        )
    yield from safetensors_weights_iterator(hf_weights_files, use_tqdm_on_load)


def _guard_unfilterable(name: str, original):
    """Refuse loaders that read whole shards before the Engram filter can run."""

    def wrapped(*args, **kwargs):
        if disk_offload_enabled():
            raise RuntimeError(
                f"{name} materializes whole shards before yielding, which would "
                "read the Engram tables into memory. While DSV41_ENGRAM_DISK=1, "
                "use load_format=safetensors with the lazy strategy."
            )
        return original(*args, **kwargs)

    return wrapped


# Strategies that read whole shards into RAM before yielding. `eager` does
# `load(f.read())` per shard (weight_utils.py:958-963), which materializes the
# Engram tables before any post-yield filter can see them.
_WHOLE_SHARD_STRATEGIES = frozenset({"eager"})


def _guard_load_strategy(name: str, original):
    """Refuse a whole-shard `safetensors_load_strategy` while disk is on.

    The lazy strategy is the only one whose `safe_open` handle the Engram filter
    can intercept before the table bytes are read. The strategy arrives as a
    *keyword argument* of the same iterator function, so guarding the iterator
    is not enough: `eager` reaches `load(f.read())` regardless of the wrapper.
    """

    def wrapped(*args, **kwargs):
        strategy = kwargs.get("safetensors_load_strategy")
        if strategy is None and len(args) > 2:
            strategy = args[2]
        if disk_offload_enabled() and strategy in _WHOLE_SHARD_STRATEGIES:
            raise RuntimeError(
                f"{name}(safetensors_load_strategy={strategy!r}) reads whole "
                "shards before yielding, so the Engram tables would be "
                "materialized into unified memory. While DSV41_ENGRAM_DISK=1, "
                "use --safetensors-load-strategy lazy."
            )
        return original(*args, **kwargs)

    return wrapped


def _wrap_model_load_weights(original):
    def wrapped(self, weights):
        if disk_offload_enabled():
            weights = skip_engram_tables(weights)
        return original(self, weights)

    return wrapped


def _wrap_default_loader(original):
    """Keep 1-row placeholders from being overwritten by a real table row."""

    def wrapped(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        if (
            disk_offload_enabled()
            and getattr(param, "engram_vocab_start", None) is not None
            and tuple(param.shape) == (1, param.shape[-1])
            and loaded_weight.shape[0] > 1
        ):
            return None
        return original(param, loaded_weight)

    return wrapped


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from vllm.models.deepseek_v4_1.common import engram as engram_mod
    from vllm.model_executor.model_loader import default_loader, weight_utils

    engram_mod.Engram.__init__ = _wrap_engram_init(engram_mod.Engram.__init__)
    engram_mod.ParallelEngramEmbedding.__init__ = _wrap_embed_init(
        engram_mod.ParallelEngramEmbedding.__init__
    )
    engram_mod.ParallelEngramEmbedding.lookup = _wrap_lookup(
        engram_mod.ParallelEngramEmbedding.lookup
    )

    iterator_names = (
        "safetensors_weights_iterator",
        "multi_thread_safetensors_weights_iterator",
        "np_cache_weights_iterator",
        "fastsafetensors_weights_iterator",
        "pt_weights_iterator",
        "multi_thread_pt_weights_iterator",
        "runai_safetensors_weights_iterator",
        "instanttensor_weights_iterator",
    )
    for attr in iterator_names:
        fn = getattr(weight_utils, attr, None)
        if callable(fn):
            wrapped = _wrap_iterator(fn)
            setattr(weight_utils, attr, wrapped)
            if hasattr(default_loader, attr):
                setattr(default_loader, attr, wrapped)

    # The strategy is an argument of the same iterator, so the post-yield filter
    # above cannot stop `eager` from reading a whole shard first. Guard it.
    for mod in (weight_utils, default_loader):
        for attr in (
            "safetensors_weights_iterator",
            "multi_thread_safetensors_weights_iterator",
        ):
            fn = getattr(mod, attr, None)
            if callable(fn):
                setattr(mod, attr, _guard_load_strategy(attr, fn))

    # Filter at the safetensors handle so the tables are never read.
    if callable(getattr(weight_utils, "safe_open", None)):
        weight_utils.safe_open = _wrap_safe_open(weight_utils.safe_open)
    if callable(getattr(default_loader, "safe_open", None)):
        default_loader.safe_open = _wrap_safe_open(default_loader.safe_open)

    multi_thread = getattr(
        weight_utils, "multi_thread_safetensors_weights_iterator", None
    )
    if callable(multi_thread):
        setattr(
            weight_utils,
            "multi_thread_safetensors_weights_iterator",
            _engram_aware_multi_thread,
        )
        if hasattr(default_loader, "multi_thread_safetensors_weights_iterator"):
            setattr(
                default_loader,
                "multi_thread_safetensors_weights_iterator",
                _engram_aware_multi_thread,
            )

    for attr in (
        "fastsafetensors_weights_iterator",
        "instanttensor_weights_iterator",
        "runai_safetensors_weights_iterator",
    ):
        fn = getattr(weight_utils, attr, None)
        if callable(fn):
            guarded = _guard_unfilterable(attr, fn)
            setattr(weight_utils, attr, guarded)
            if hasattr(default_loader, attr):
                setattr(default_loader, attr, guarded)

    if hasattr(weight_utils, "default_weight_loader"):
        weight_utils.default_weight_loader = _wrap_default_loader(
            weight_utils.default_weight_loader
        )
    from vllm.models.deepseek_v4_1.nvidia import model as v41_model
    from vllm.models.deepseek_v4_1.nvidia import vl_model as v41_vl

    for mod in (v41_model, v41_vl):
        for cls_name in (
            "DeepseekV41LLMForCausalLM",
            "DeepseekV41ForCausalLM",
        ):
            cls = getattr(mod, cls_name, None)
            if cls is not None and hasattr(cls, "load_weights"):
                cls.load_weights = _wrap_model_load_weights(cls.load_weights)

    _INSTALLED = True
    logger.info(
        "V4.1 Engram NVMe overlay installed (disk=%s). Stock Triton UVA "
        "lookup is unused while disk tables are bound.",
        disk_offload_enabled(),
    )


def filter_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Generator[tuple[str, Any], None, None]:
    yield from skip_engram_tables(weights)
