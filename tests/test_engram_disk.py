#!/usr/bin/env python3
"""Engram disk overlay tests. Dequant/gather needs torch; skip-path does not."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

OVERLAY = Path(__file__).resolve().parents[1] / "overlay"
sys.path.insert(0, str(OVERLAY))

from engram_disk import (  # noqa: E402
    is_engram_table_tensor,
    skip_engram_tables,
)

INSTALL = (OVERLAY / "engram_install.py").read_text()
BOOT = (OVERLAY / "dsv41_boot.py").read_text()


def test_engram_filtered_before_materialization() -> None:
    # vLLM's lazy iterator calls get_tensor(name) before yielding, so a
    # post-yield filter still reads the 91 GiB table. The overlay must hide the
    # names at the safetensors handle and refuse whole-shard loaders.
    assert "_EngramFilteringHandle" in INSTALL
    assert "def keys(self)" in INSTALL
    assert "not is_engram_table_tensor(n)" in INSTALL
    assert "safe_open = _wrap_safe_open" in INSTALL
    assert "_engram_aware_multi_thread" in INSTALL
    assert "_guard_unfilterable" in INSTALL


def test_whole_shard_strategy_is_refused() -> None:
    """`safetensors_load_strategy=eager` must be refused, not merely filtered.

    The strategy is a keyword argument of the same iterator the post-yield
    filter wraps, and it reaches `load(f.read())` per shard
    (`weight_utils.py:958-963`) before anything is yielded — so wrapping the
    iterator cannot stop it. The guard has to inspect the argument.
    """
    assert "_guard_load_strategy" in INSTALL
    assert "_WHOLE_SHARD_STRATEGIES" in INSTALL
    assert '"eager"' in INSTALL
    # It must be applied to both iterator names, on both modules.
    assert INSTALL.count("_guard_load_strategy(") >= 2, INSTALL.count("_guard_load_strategy(")

    # Behaviour: the guard refuses eager and passes lazy through, only while
    # disk offload is on.
    src = INSTALL
    ns: dict = {}
    start = src.index("def _guard_load_strategy")
    end = src.index("\ndef _wrap_model_load_weights")
    import os as _os

    _os.environ["DSV41_ENGRAM_DISK"] = "1"
    ns["disk_offload_enabled"] = lambda: True
    ns["_WHOLE_SHARD_STRATEGIES"] = frozenset({"eager"})
    exec(compile(src[start:end], "<engram_install>", "exec"), ns)  # noqa: S102
    guard = ns["_guard_load_strategy"]

    def original(*args, **kwargs):
        return ("called", args, kwargs)

    wrapped = guard("safetensors_weights_iterator", original)
    try:
        wrapped([], False, "eager")
    except RuntimeError as exc:
        assert "materialized into unified memory" in str(exc), exc
    else:
        raise AssertionError("eager must be refused while disk offload is on")
    try:
        wrapped([], False, safetensors_load_strategy="eager")
    except RuntimeError:
        pass
    else:
        raise AssertionError("eager passed as a keyword must also be refused")
    # lazy is allowed, positionally and by keyword.
    assert wrapped([], False, "lazy")[0] == "called"
    assert wrapped([], False, safetensors_load_strategy="lazy")[0] == "called"
    assert wrapped([], False, None)[0] == "called"
    # With disk offload off, eager is allowed again.
    ns["disk_offload_enabled"] = lambda: False
    assert guard("x", original)([], False, "eager")[0] == "called"


def test_engram_shard_window_matches_stock_tp_dp_split() -> None:
    """The NVMe placeholders must cover the rows stock would have owned.

    The installed class splits the hash heads over `TP x DP` and indexes the
    shard TP-major (`tp_rank * dp_size + dp_rank`). The disk overlay replaces
    that constructor, so its window has to agree with the stock arithmetic for
    every rank — a mismatch gathers the wrong heads.
    """
    src = INSTALL
    start = src.index("def _engram_shard_window")
    end = src.index("\ndef _init_disk_embed")
    ns: dict = {}
    exec(compile(src[start:end], "<engram_install>", "exec"), ns)  # noqa: S102
    window = ns["_engram_shard_window"]

    def stock(
        n_hash_cols: int,
        head_sizes: tuple[int, ...],
        tp_size: int,
        dp_size: int,
        rank: int,
    ) -> tuple[int, int, int, int]:
        # vllm/models/deepseek_v4_1/common/engram.py ParallelEngramEmbedding.__init__
        part = -(-n_hash_cols // (tp_size * dp_size))
        head_start = rank * part
        head_end = head_start + part
        return part, head_start, sum(head_sizes[:head_start]), sum(head_sizes[:head_end])

    head_sizes = tuple(range(1001, 1001 + 36))
    for tp_size in (1, 2, 4):
        for dp_size in (1, 2, 3):
            for tp_rank in range(tp_size):
                for dp_rank in range(dp_size):
                    rank = tp_rank * dp_size + dp_rank
                    got = window(len(head_sizes), head_sizes, tp_size, dp_size, rank)
                    assert got == stock(
                        len(head_sizes), head_sizes, tp_size, dp_size, rank
                    ), (tp_size, dp_size, tp_rank, dp_rank, got)

    # TP-major ordering: under TP=2 x DP=2 shard 2 is the second TP rank's first
    # DP shard, so a DP gather brings in contiguous heads.
    assert window(36, head_sizes, 2, 2, 2)[1] == 2 * 9
    assert window(36, head_sizes, 2, 2, 1)[1] == 1 * 9

    # The windows partition the heads and the rows, with no gap between ranks.
    for tp_size, dp_size in ((1, 1), (2, 1), (2, 2), (4, 2)):
        parts = [
            window(36, head_sizes, tp_size, dp_size, r)
            for r in range(tp_size * dp_size)
        ]
        assert parts[0][2] == 0
        assert parts[-1][3] == sum(head_sizes)
        assert sum(p[0] for p in parts) >= 36  # cdiv may pad the last shard
        for a, b in zip(parts, parts[1:]):
            assert a[3] == b[2], (tp_size, dp_size, a, b)

    # A split that would leave a rank with no heads is refused, as stock asserts.
    try:
        window(3, head_sizes[:3], 2, 2, 0)
    except ValueError as exc:
        assert "leaves ranks without hash heads" in str(exc), exc
    else:
        raise AssertionError("3 heads over TP=2 x DP=2 must be refused")


def test_disk_embed_keeps_the_dp_size_attribute() -> None:
    """`Engram.__init__` reads `embed_tokens.dp_size`; the disk path must set it.

    Without it the worker rank dies with `AttributeError:
    'ParallelEngramEmbedding' object has no attribute 'dp_size'` while
    `staged_rows` is sized (`engram.py:1058` in the installed wheel).
    """
    body = INSTALL[
        INSTALL.index("def _init_disk_embed") : INSTALL.index("def _wrap_embed_init")
    ]
    assert "self.tp_size = tp_size" in body
    assert "self.dp_size = dp_size" in body
    # The window must come from the shared helper, not a bare TP rank, or it
    # silently diverges the moment an Engram DP group exists.
    assert "_engram_shard_window(" in body
    assert "tp_rank * self.part_n_hash_cols" not in body


def test_engram_sharding_resolves_with_and_without_dp() -> None:
    """The pinned tree has no DP helpers; the installed wheel does."""
    src = INSTALL
    start = src.index("def _resolve_engram_sharding")
    end = src.index("\ndef _engram_shard_window")
    ns: dict = {}
    exec(compile(src[start:end], "<engram_install>", "exec"), ns)  # noqa: S102
    resolve = ns["_resolve_engram_sharding"]

    import types

    names = (
        "vllm",
        "vllm.distributed",
        "vllm.models",
        "vllm.models.deepseek_v4_1",
        "vllm.models.deepseek_v4_1.common",
        "vllm.models.deepseek_v4_1.common.engram",
    )
    saved = {k: sys.modules.get(k) for k in names}

    def install_fake(engram_attrs: dict, tp_size: int, tp_rank: int) -> None:
        for name in names[:5]:
            mod = types.ModuleType(name)
            mod.__path__ = []
            sys.modules[name] = mod
        dist = types.ModuleType("vllm.distributed")
        dist.get_tensor_model_parallel_world_size = lambda: tp_size
        dist.get_tensor_model_parallel_rank = lambda: tp_rank
        sys.modules["vllm.distributed"] = dist
        sys.modules["vllm"].distributed = dist
        engram = types.ModuleType("vllm.models.deepseek_v4_1.common.engram")
        for key, value in engram_attrs.items():
            setattr(engram, key, value)
        sys.modules["vllm.models.deepseek_v4_1.common.engram"] = engram
        sys.modules["vllm.models.deepseek_v4_1.common"].engram = engram

    try:
        # Pinned tree: neither helper exists, so the window stays TP-only.
        install_fake({}, tp_size=2, tp_rank=1)
        assert resolve() == (2, 1, 1)
        # Installed tree with an Engram DP group: DP size and TP-major rank.
        install_fake(
            {"get_engram_dp_size": lambda: 2, "engram_head_shard_rank": lambda: 3},
            tp_size=2,
            tp_rank=1,
        )
        assert resolve() == (2, 2, 3)
    finally:
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


def test_boot_hook_reaches_spawned_processes() -> None:
    # `vllm serve` forces spawn, so an install that only runs in serve.py never
    # reaches the EngineCore/TP workers that build Engram. A site `.pth` must
    # import this module, which installs at interpreter startup.
    assert "install_runtime" in BOOT
    assert "DSV41_ENGRAM_DISK" in BOOT


def test_boot_hook_fails_closed() -> None:
    # site.addpackage prints a `.pth` traceback and continues, so an overlay
    # failure must exit the process instead of warning: a process that keeps
    # running without the overlay materializes the 91 GiB NVMe tables.
    assert "os._exit(70)" in BOOT
    assert "DSV41_ENGRAM_STRICT" in BOOT
    serve = (OVERLAY / "serve.py").read_text()
    assert "_verify_overlay" in serve
    assert "_INSTALLED" in serve
    assert "spawn" in BOOT


def test_skip_tables() -> None:
    items = [
        ("layers.1.engram.embed.weight", object()),
        ("layers.1.engram.embed.scale", object()),
        ("layers.1.engram.wkv.weight", object()),
        ("layers.2.ffn.experts.0.w1.trellis", object()),
    ]
    kept = list(skip_engram_tables(items))
    assert [n for n, _ in kept] == [
        "layers.1.engram.wkv.weight",
        "layers.2.ffn.experts.0.w1.trellis",
    ]
    assert is_engram_table_tensor("layers.14.engram.embed.weight")
    assert not is_engram_table_tensor("layers.14.engram.wkv.weight")


def test_ue8m0_dequant_and_gather() -> None:
    try:
        import torch
    except ImportError:
        print("torch missing; skip dequant/gather")
        return

    from engram_disk import EngramDiskTable, dequant_fp8_ue8m0

    dim = 32
    weight = torch.tensor([[1.0] * dim], dtype=torch.float8_e4m3fn).view(torch.uint8)
    scale = torch.tensor([[127]], dtype=torch.uint8)
    out = dequant_fp8_ue8m0(weight, scale, dim)
    assert out.shape == (1, dim)
    assert out.dtype == torch.bfloat16
    assert torch.allclose(out.float(), torch.ones(1, dim), atol=0.05)

    def fp8_bytes(values: list[float]) -> bytes:
        t = torch.tensor(values, dtype=torch.float32).reshape(1, dim).to(
            torch.float8_e4m3fn
        )
        return bytes(t.contiguous().view(torch.uint8).reshape(-1).tolist())

    n_rows = 8
    weight_rows = [fp8_bytes([1.0 if i == 3 else 0.0] * dim) for i in range(n_rows)]
    scale_rows = [bytes([127]) for _ in range(n_rows)]

    def write_safetensors(path: Path, tensors: dict) -> None:
        header: dict = {}
        blob = bytearray()
        offset = 0
        for name, (dtype, shape, data) in tensors.items():
            header[name] = {
                "dtype": dtype,
                "shape": list(shape),
                "data_offsets": [offset, offset + len(data)],
            }
            blob += data
            offset += len(data)
        raw = json.dumps(header).encode("utf-8")
        path.write_bytes(len(raw).to_bytes(8, "little") + raw + bytes(blob))

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        shard = root / "model-00001-of-00001.safetensors"
        write_safetensors(
            shard,
            {
                "layers.1.engram.embed.weight": (
                    "F8_E4M3",
                    (n_rows, dim),
                    b"".join(weight_rows),
                ),
                "layers.1.engram.embed.scale": (
                    "U8",
                    (n_rows, 1),
                    b"".join(scale_rows),
                ),
            },
        )
        (root / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {
                        "layers.1.engram.embed.weight": shard.name,
                        "layers.1.engram.embed.scale": shard.name,
                    }
                }
            )
        )
        table = EngramDiskTable(root, layer_id=1, vocab_start=2, vocab_end=6, dim=dim)
        ids = torch.tensor([0, 3, 7, 3], dtype=torch.int64)
        gathered = table.gather_dequant(ids)
        assert gathered.shape == (4, dim)
        assert torch.count_nonzero(gathered[0]) == 0
        assert torch.count_nonzero(gathered[2]) == 0
        assert torch.allclose(gathered[1].float(), torch.ones(dim), atol=0.05)
        assert torch.equal(gathered[1], gathered[3])
        table.close()


if __name__ == "__main__":
    test_skip_tables()
    test_engram_filtered_before_materialization()
    test_whole_shard_strategy_is_refused()
    test_engram_shard_window_matches_stock_tp_dp_split()
    test_disk_embed_keeps_the_dp_size_attribute()
    test_engram_sharding_resolves_with_and_without_dp()
    test_boot_hook_reaches_spawned_processes()
    test_boot_hook_fails_closed()
    test_ue8m0_dequant_and_gather()
    print("engram disk tests OK")
