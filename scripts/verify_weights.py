#!/usr/bin/env python3
"""Verify a staged DeepSeek-V4.1-Flash-REAP-256E checkpoint against its own index.

Why this exists: the checkpoint is 385.5 GiB across 48 shards, and the REAP Hub
repo publishes only 46 of them. `model.safetensors.index.json` still lists 48, so
the two missing shards are not optional and are not obvious — they hold the two
Engram layers (`layers.1.engram.*` and `layers.14.engram.*`, six keys each, four
of which are ordinary weights the model needs). A directory that is missing them
passes every filename check that stops at "the shards I downloaded are present"
and then fails inside the loader, minutes into a weight load.

Three failure modes are checked, all of which a partial 385 GiB transfer can
produce:

1. **A shard the index references is absent.** Reported per shard, with the key
   names that live there, so the operator can see *what* is missing rather than
   just how many files.
2. **A shard is truncated.** Every shard is validated by comparing the file size
   against the extent its own header declares (`8 + header_len + max end`). A
   truncated write is the classic symptom of an interrupted download, and it is
   invisible to a size-only sanity check because the file is "there" and non-zero.
3. **A shard holds the wrong tensors.** Every key in the index must appear in the
   header of the shard the index assigns it to. This is what catches the
   base-model substitution going wrong: shards 47/48 come from
   `deepseek-ai/DeepSeek-V4.1-Flash`, and if that source were ever re-sharded or
   renamed, the files would still exist and still be the right *size* while
   carrying the wrong names.

Reads only the safetensors headers (a few MB), never the tensor data, so it is
cheap enough to run before every boot.

Usage:
    verify_weights.py --model-dir <dir> [--expect-shards 48] [--quiet]
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

INDEX_NAME = "model.safetensors.index.json"
INDEX_SHARD_KEYS = ("weight_map",)
# Shards the REAP repo does not publish and that must be staged from the base
# checkpoint. Named so the failure message can say where to get them.
BASE_MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"


class ShardError(Exception):
    """A shard that cannot be read as safetensors at all."""


def read_header(path: Path) -> tuple[int, dict]:
    """Return (data_start, header) for a safetensors file.

    The format is an 8-byte little-endian header length followed by that many
    bytes of JSON. `data_start` is the absolute offset of tensor byte 0.
    """
    size = path.stat().st_size
    with path.open("rb") as handle:
        raw_len = handle.read(8)
        if len(raw_len) != 8:
            raise ShardError(f"file is {size} B, too short for a header")
        (header_len,) = struct.unpack("<Q", raw_len)
        if header_len <= 0 or header_len > 256 * 1024 * 1024:
            raise ShardError(f"implausible header length {header_len}")
        if 8 + header_len > size:
            raise ShardError(
                f"header claims {header_len} B but the file is only {size} B "
                "(truncated download)"
            )
        raw = handle.read(header_len)
        if len(raw) != header_len:
            raise ShardError("short read of the header")
    try:
        header = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ShardError(f"header is not JSON: {exc}") from None
    header.pop("__metadata__", None)
    return 8 + header_len, header


def declared_extent(header: dict) -> int:
    """The furthest byte any tensor in this header occupies."""
    end = 0
    for name, info in header.items():
        offsets = info.get("data_offsets")
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise ShardError(f"tensor {name!r} has no usable data_offsets")
        end = max(end, int(offsets[1]))
    return end


def load_index(model_dir: Path) -> dict:
    index_path = model_dir / INDEX_NAME
    if not index_path.is_file():
        raise SystemExit(f"no {INDEX_NAME} in {model_dir}")
    payload = json.loads(index_path.read_text())
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise SystemExit(f"{index_path} has no weight_map")
    payload["_path"] = index_path
    return payload


def verify(model_dir: Path, expect_shards: int | None, quiet: bool) -> int:
    payload = load_index(model_dir)
    weight_map: dict[str, str] = payload["weight_map"]
    declared_total = int((payload.get("metadata") or {}).get("total_size") or 0)

    by_shard: dict[str, list[str]] = {}
    for key, shard in weight_map.items():
        by_shard.setdefault(shard, []).append(key)

    problems: list[str] = []
    missing_shards: list[str] = []
    measured_total = 0
    checked = 0

    for shard in sorted(by_shard):
        path = model_dir / shard
        if not path.is_file():
            missing_shards.append(shard)
            continue
        try:
            data_start, header = read_header(path)
        except ShardError as exc:
            problems.append(f"{shard}: unreadable ({exc})")
            continue

        size = path.stat().st_size
        extent = declared_extent(header)
        if data_start + extent != size:
            problems.append(
                f"{shard}: file is {size} B but its header declares "
                f"{data_start + extent} B "
                f"({'truncated by ' + str(data_start + extent - size) + ' B' if size < data_start + extent else 'longer than declared'})"
            )
            continue

        wanted = by_shard[shard]
        absent = [key for key in wanted if key not in header]
        if absent:
            problems.append(
                f"{shard}: index expects {len(wanted)} keys, "
                f"{len(absent)} absent from the shard header "
                f"(first: {', '.join(sorted(absent)[:3])})"
            )
            continue

        measured_total += extent
        checked += 1
        if not quiet:
            print(
                f"  ok   {shard:38s} {size / 1024**3:7.2f} GiB "
                f"{len(wanted):6d} keys"
            )

    if expect_shards is not None and len(by_shard) != expect_shards:
        problems.append(
            f"index references {len(by_shard)} shards, expected {expect_shards}"
        )

    print()
    print(
        f"shards: {checked}/{len(by_shard)} verified, "
        f"{len(by_shard) - checked - len(missing_shards)} unreadable, "
        f"{len(missing_shards)} missing"
    )
    print(f"tensor bytes measured: {measured_total / 1024**3:.1f} GiB")
    if declared_total:
        print(f"index declares:        {declared_total / 1024**3:.1f} GiB")

    if missing_shards:
        print(file=sys.stderr)
        print("MISSING SHARDS:", file=sys.stderr)
        for shard in missing_shards:
            keys = sorted(by_shard[shard])
            print(f"  {shard}  ({len(keys)} keys)", file=sys.stderr)
            for key in keys[:6]:
                print(f"      {key}", file=sys.stderr)
            if len(keys) > 6:
                print(f"      ... +{len(keys) - 6} more", file=sys.stderr)
        print(
            f"\nThe REAP repo publishes only the shards that changed. Any shard "
            f"named here must be staged from the base checkpoint "
            f"`{BASE_MODEL}` into this same directory, under exactly these "
            f"filenames. `scripts/fetch_weights.sh` does that.",
            file=sys.stderr,
        )

    if problems:
        print(file=sys.stderr)
        print("SHARD PROBLEMS:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)

    if missing_shards or problems:
        print("\ncheckpoint INCOMPLETE", file=sys.stderr)
        return 1
    if declared_total and measured_total != declared_total:
        print(
            f"\ncheckpoint complete but tensor bytes ({measured_total}) do not "
            f"match the index total_size ({declared_total})",
            file=sys.stderr,
        )
        return 1
    print("\ncheckpoint OK: every indexed key is present and every shard is whole")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument(
        "--expect-shards",
        type=int,
        default=None,
        help="refuse unless the index references exactly this many shards",
    )
    ap.add_argument("--quiet", action="store_true", help="print only the summary")
    args = ap.parse_args()

    if not args.model_dir.is_dir():
        print(f"ERROR: no directory at {args.model_dir}", file=sys.stderr)
        return 2
    return verify(args.model_dir, args.expect_shards, args.quiet)


if __name__ == "__main__":
    sys.exit(main())
