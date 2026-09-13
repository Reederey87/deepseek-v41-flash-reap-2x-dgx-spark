# SPDX-License-Identifier: Apache-2.0
"""NVMe Engram lookup for DeepSeek-V4.1 on GB10 unified memory.

Stock vLLM (`vllm.models.deepseek_v4_1.common.engram.ParallelEngramEmbedding`)
keeps the n-gram tables either in HBM or in pinned host memory and gathers
through Triton/`UVA`. On GB10 that pin still occupies the 121 GiB UMA pool.

This module never string-patches `engram.py`. It:

1. skips `.engram.embed.weight` / `.engram.embed.scale` in the safetensors
   iterator so the 189–203 GiB tables never enter the process,
2. shrinks the module parameters to 1-row placeholders so `Engram` still
   constructs,
3. replaces `lookup()` with host `os.pread` + CPU ue8m0 dequant + H2D.

Row ids passed into `lookup()` are the **global file ids** produced by
`NgramHashState` (this tree's Triton hash kernel). Owned rows are those in
`[vocab_start_idx, vocab_end_idx)` computed by the stock constructor.

Not Direct I/O and not CUDA mmap. Bound by NVMe + memcpy. First boot is
`--max-num-seqs 2` for that reason.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Iterator

import logging

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

ENGRAM_DISK_ENV = "DSV41_ENGRAM_DISK"
MODEL_DIR_ENV = "DSV41_MODEL_DIR"
SKIP_SUFFIXES = (".engram.embed.weight", ".engram.embed.scale")
QUANT_BLOCK = 32
FP8_E4M3_ITEMSIZE = 1


def disk_offload_enabled() -> bool:
    return os.environ.get(ENGRAM_DISK_ENV, "1") == "1"


def is_engram_table_tensor(name: str) -> bool:
    return name.endswith(SKIP_SUFFIXES)


def skip_engram_tables(
    weights: Iterable[tuple[str, Any]],
) -> Iterator[tuple[str, Any]]:
    for name, tensor in weights:
        if is_engram_table_tensor(name):
            continue
        yield name, tensor


def _ue8m0_to_fp32(scale_u8: "torch.Tensor") -> "torch.Tensor":
    """One ue8m0 byte is the IEEE-754 exponent field of an fp32 power of two."""
    import torch

    return (scale_u8.to(torch.int32) << 23).view(torch.float32)


def dequant_fp8_ue8m0(
    weight_u8: "torch.Tensor",
    scale_u8: "torch.Tensor",
    dim: int,
    block_size: int = QUANT_BLOCK,
) -> "torch.Tensor":
    """CPU path matching `_engram_lookup_kernel` in this tree.

    `weight_u8` is fp8 e4m3 bytes `[N, dim]`. `scale_u8` is ue8m0 `[N, dim/block]`.
    Returns bf16 `[N, dim]`.
    """
    import torch

    if dim % block_size:
        raise ValueError(f"Engram head_dim {dim} is not a multiple of {block_size}")
    weight = weight_u8.view(torch.float8_e4m3fn).to(torch.float32)
    scale = _ue8m0_to_fp32(scale_u8).repeat_interleave(block_size, dim=-1)
    return (weight * scale).to(torch.bfloat16)


def _read_u64(path: Path, offset: int) -> int:
    with path.open("rb") as handle:
        data = os.pread(handle.fileno(), 8, offset)
    if len(data) != 8:
        raise RuntimeError(f"short read at {path}:{offset}")
    return int.from_bytes(data, "little")


def parse_safetensors_header(path: Path) -> tuple[int, dict[str, Any]]:
    header_len = _read_u64(path, 0)
    with path.open("rb") as handle:
        raw = os.pread(handle.fileno(), header_len, 8)
    header = json.loads(raw)
    header.pop("__metadata__", None)
    return 8 + header_len, header


@dataclass(frozen=True)
class DiskTensor:
    path: Path
    name: str
    dtype: str
    shape: tuple[int, ...]
    data_start: int
    itemsize: int

    @property
    def row_bytes(self) -> int:
        if len(self.shape) != 2:
            raise ValueError(f"{self.name} is not a 2D table: {self.shape}")
        return int(self.shape[1]) * self.itemsize

    def byte_offset(self, row: int) -> int:
        if row < 0 or row >= self.shape[0]:
            raise IndexError(f"{self.name} row {row} out of {self.shape[0]}")
        return self.data_start + row * self.row_bytes


_ITEMSIZE = {
    "F8_E4M3": 1,
    "U8": 1,
    "F8_E8M0": 1,
}


def _open_disk_tensor(path: Path, name: str) -> DiskTensor:
    data_base, header = parse_safetensors_header(path)
    info = header.get(name)
    if not info:
        raise KeyError(f"{path} has no tensor {name}")
    dtype = str(info["dtype"])
    itemsize = _ITEMSIZE.get(dtype)
    if itemsize is None:
        raise TypeError(f"{name} dtype {dtype} is not an Engram table dtype")
    start, _end = info["data_offsets"]
    shape = tuple(int(x) for x in info["shape"])
    return DiskTensor(
        path=path,
        name=name,
        dtype=dtype,
        shape=shape,
        data_start=data_base + int(start),
        itemsize=itemsize,
    )


def load_weight_index(model_dir: Path) -> dict[str, str]:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(
            f"Engram disk overlay needs {index_path} to locate table shards"
        )
    payload = json.loads(index_path.read_text())
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"{index_path} has no weight_map")
    return {str(k): str(v) for k, v in weight_map.items()}


def resolve_model_dir() -> Path:
    raw = os.environ.get(MODEL_DIR_ENV, "").strip()
    if not raw:
        raise RuntimeError(
            f"{MODEL_DIR_ENV} is required when {ENGRAM_DISK_ENV}=1 so Engram "
            "tables can be opened from NVMe without pinning them"
        )
    path = Path(raw)
    if not path.is_dir():
        raise FileNotFoundError(f"{MODEL_DIR_ENV}={path} is not a directory")
    return path


class EngramDiskTable:
    """One TP-rank's window onto a full-file Engram table."""

    def __init__(
        self,
        model_dir: Path,
        layer_id: int,
        vocab_start: int,
        vocab_end: int,
        dim: int,
        block_size: int = QUANT_BLOCK,
    ) -> None:
        if vocab_end <= vocab_start:
            raise ValueError(f"empty Engram shard [{vocab_start}, {vocab_end})")
        weight_name = f"layers.{layer_id}.engram.embed.weight"
        scale_name = f"layers.{layer_id}.engram.embed.scale"
        weight_map = load_weight_index(model_dir)
        try:
            weight_file = model_dir / weight_map[weight_name]
            scale_file = model_dir / weight_map[scale_name]
        except KeyError as exc:
            raise KeyError(
                f"checkpoint index is missing Engram tensor {exc.args[0]}"
            ) from exc
        self.weight = _open_disk_tensor(weight_file, weight_name)
        self.scale = _open_disk_tensor(scale_file, scale_name)
        if self.weight.shape[0] != self.scale.shape[0]:
            raise ValueError(
                f"Engram weight/scale row count mismatch at layer {layer_id}: "
                f"{self.weight.shape} vs {self.scale.shape}"
            )
        if self.weight.shape[1] != dim:
            raise ValueError(
                f"Engram dim mismatch at layer {layer_id}: file {self.weight.shape[1]} "
                f"vs module {dim}"
            )
        if self.scale.shape[1] != dim // block_size:
            raise ValueError(
                f"Engram scale width {self.scale.shape[1]} != {dim}//{block_size}"
            )
        self.vocab_start = vocab_start
        self.vocab_end = min(vocab_end, int(self.weight.shape[0]))
        self.dim = dim
        self.block_size = block_size
        self.layer_id = layer_id
        self._fd_weight = os.open(self.weight.path, os.O_RDONLY)
        self._fd_scale = os.open(self.scale.path, os.O_RDONLY)
        # Rows are picked at random from a 91 GiB file, so the readahead the
        # kernel would otherwise do is pure waste. Buffered `pread` is correct
        # here (Direct I/O on 264-byte rows is all overhead), but tell the
        # kernel the pattern.
        for fd in (self._fd_weight, self._fd_scale):
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
            except (AttributeError, OSError):  # not POSIX, or fd not seekable
                pass
        logger.info(
            "Engram NVMe table layer=%s file=%s rows=%s local=[%s, %s) dim=%s",
            layer_id,
            self.weight.path.name,
            self.weight.shape[0],
            self.vocab_start,
            self.vocab_end,
            dim,
        )

    def close(self) -> None:
        for fd in (self._fd_weight, self._fd_scale):
            try:
                os.close(fd)
            except OSError:
                pass

    def gather_dequant(self, global_ids: "torch.Tensor") -> "torch.Tensor":
        """`global_ids` int64 `[N]` → bf16 `[N, dim]`. Unowned / negative → 0."""
        import torch

        ids = global_ids.detach().to(dtype=torch.int64, device="cpu").reshape(-1)
        n = int(ids.numel())
        out = torch.zeros((n, self.dim), dtype=torch.bfloat16)
        if n == 0:
            return out
        owned = (ids >= self.vocab_start) & (ids < self.vocab_end)
        if not bool(owned.any()):
            return out
        owned_ids = ids[owned]
        unique, inverse = torch.unique(owned_ids, return_inverse=True)
        unique_list = [int(x) for x in unique.tolist()]
        packed_w = bytearray()
        packed_s = bytearray()
        w_row = self.weight.row_bytes
        s_row = self.scale.row_bytes
        for row in unique_list:
            packed_w += os.pread(self._fd_weight, w_row, self.weight.byte_offset(row))
            packed_s += os.pread(self._fd_scale, s_row, self.scale.byte_offset(row))
        weight = torch.frombuffer(memoryview(packed_w), dtype=torch.uint8).reshape(
            len(unique_list), self.dim
        )
        scale = torch.frombuffer(memoryview(packed_s), dtype=torch.uint8).reshape(
            len(unique_list), self.dim // self.block_size
        )
        dequant = dequant_fp8_ue8m0(weight, scale, self.dim, self.block_size)
        out[owned] = dequant[inverse]
        return out
