#!/usr/bin/env python3
"""Fail-closed pre-flight for the compiled FlashInfer sparse-MLA module.

The image masks FlashInfer's prebuilt AOT binary (`overlay/mask_flashinfer_aot.py`)
so the patched `.cu` is compiled instead, which turns the compiled module from an
optional cache into a **required artifact**. `scripts/build_flashinfer.sh`
compiles it once per node and writes a receipt next to it.

A matching filename is not evidence that the engine will reuse the module:

* a module built for another FlashInfer version or another arch is never looked
  up, so the engine would compile again at startup — the fan-out the offline
  prebuild exists to avoid;
* a cache left over from an earlier image is stale even if it is loadable;
* an empty or truncated file matches any glob.

So the receipt is verified rather than the filename: the image identity, the
runtime cache key (version component included), and the artifact's size and
digest. Run it inside the image, on each node, with the cache mounted:

    check_flashinfer_cache.py --cache /fi --image-id sha256:...

Exit codes: 0 usable, 2 usage, 5 not usable (named reason on stderr).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_NOT_USABLE = 5

RECEIPT_NAME = "dsv41-prebuild-receipt"

REQUIRED_KEYS = (
    "image_id",
    "flashinfer_version",
    "module",
    "bytes",
    "sha256",
)

# The digest below is the real integrity check; this floor only has to catch an
# empty or placeholder file, which is exactly what a filename match cannot
# distinguish from a real build.
#
# Calibrated against the real artifacts, not guessed. The stock AOT module in the
# base image is **967,520 bytes**, and the per-node prebuild of the patched source
# is **972,952 bytes** — so a sparse-MLA module is a *sub*-megabyte object, and a
# 1 MiB floor rejects a perfectly good build. (It did, on the first boot attempt:
# preflight refused a correct artifact and named it a placeholder.) 256 KiB sits
# far below any real build and far above an empty file or a stub.
MIN_MODULE_BYTES = 1 << 18


def runtime_spec() -> tuple[Path, bool] | None:
    """`(cache-relative path the runtime loads, aot_present)`, or None.

    FlashInfer resolves the module through
    ``JitSpec.get_library_path()`` -> ``FLASHINFER_JIT_DIR/<name>/<name>.so``,
    where the version and arch components are computed by FlashInfer itself from
    the installed version and the target arch list
    (``jit/env.py``: ``FLASHINFER_CACHE_DIR / flashinfer_version / arch``, arch =
    ``"_".join(f"{major}{minor}" for major, minor in sorted(TARGET_CUDA_ARCHS))``).

    Asking FlashInfer for that path, instead of checking components by hand, is
    what makes a wrong-arch, missing-arch or extra-component receipt impossible
    to accept: anything that is not the exact path the runtime loads is a cache
    miss, and a cache miss means the engine compiles at startup.
    """
    try:
        from flashinfer.jit import env as jit_env
        from flashinfer.mla._sparse_mla_sm120 import gen_sparse_mla_sm120_module

        spec = gen_sparse_mla_sm120_module()
        return (
            spec.jit_library_path.relative_to(jit_env.FLASHINFER_CACHE_DIR),
            bool(spec.is_aot),
        )
    except Exception:  # noqa: BLE001
        return None


def parse_receipt(text: str) -> dict[str, str]:
    """Key/value lines, ignoring blanks and `#` comments."""
    fields: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        fields[key.strip()] = value.strip()
    return fields


def installed_version() -> str | None:
    """The installed FlashInfer version, or None when it cannot be read."""
    try:
        import flashinfer

        version = getattr(flashinfer, "__version__", None)
        if version:
            return str(version)
    except Exception:  # noqa: BLE001
        return None
    try:
        from importlib.metadata import version as dist_version

        return dist_version("flashinfer-python")
    except Exception:  # noqa: BLE001
        return None


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def problems(cache: Path, image_id: str) -> list[str]:
    """Everything wrong with the cache, as named reasons."""
    receipt = cache / RECEIPT_NAME
    if not receipt.is_file():
        return [
            f"no prebuild receipt at {receipt}: the compiled module has not been "
            "verified on this node. Run scripts/build_flashinfer.sh here "
            "(RESEARCH.md 11.9.3)"
        ]

    fields = parse_receipt(receipt.read_text())
    missing = [key for key in REQUIRED_KEYS if not fields.get(key)]
    if missing:
        return [f"{receipt} is missing {missing}; re-run scripts/build_flashinfer.sh"]

    found: list[str] = []

    if fields["image_id"] != image_id:
        found.append(
            f"the receipt was written from image {fields['image_id']} but this "
            f"container is {image_id}. The image changed, so the cache is stale: "
            "re-run scripts/build_flashinfer.sh"
        )

    want_version = fields["flashinfer_version"]
    have_version = installed_version()
    if have_version is None:
        found.append("the installed FlashInfer version cannot be read")
    elif have_version != want_version:
        found.append(
            f"the receipt is for FlashInfer {want_version} but {have_version} is "
            "installed. FlashInfer keys its cache on the version, so this module "
            "would never be looked up: re-run scripts/build_flashinfer.sh"
        )

    # The decisive check: the receipt must name the exact path the runtime loads.
    spec = runtime_spec()
    if spec is None:
        found.append(
            "the cache path FlashInfer would load cannot be determined, so the "
            "receipt cannot be checked against it"
        )
        return found
    expected, is_aot = spec

    recorded = Path(fields["module"])
    if recorded != expected:
        found.append(
            f"the receipt names {recorded} but the runtime loads {expected}. "
            "FlashInfer resolves the module from the installed version and the "
            "target arch list, so anything else is a cache miss and the engine "
            "would compile at startup: re-run scripts/build_flashinfer.sh"
        )
        return found

    module = cache / recorded
    if not module.is_file():
        found.append(f"the receipt names {module}, which is not a file")
        return found

    size = module.stat().st_size
    if size < MIN_MODULE_BYTES:
        found.append(
            f"{module} is only {size} bytes; a compiled sparse-MLA module is "
            "tens of megabytes, so this is not a real build artifact"
        )
    if str(size) != fields["bytes"]:
        found.append(
            f"{module} is {size} bytes but the receipt says {fields['bytes']}: "
            "the artifact is truncated or was replaced"
        )
    digest = sha256_of(module)
    if digest != fields["sha256"]:
        found.append(
            f"{module} hashes to {digest} but the receipt says {fields['sha256']}"
        )

    if is_aot:
        found.append(
            "the prebuilt AOT binary is still present, so FlashInfer would return "
            "it instead of this module. The image must run "
            "overlay/mask_flashinfer_aot.py"
        )
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", required=True, type=Path, help="mounted FlashInfer cache root")
    ap.add_argument("--image-id", required=True, help="`docker image inspect -f '{{.Id}}'`")
    args = ap.parse_args()

    if not args.cache.is_dir():
        print(f"ERROR: no cache directory at {args.cache}", file=sys.stderr)
        return EXIT_NOT_USABLE

    print(f"FlashInfer module cache pre-flight ({args.cache})")
    found = problems(args.cache, args.image_id)
    if found:
        print("\nERROR: the compiled FlashInfer module cache is not usable:", file=sys.stderr)
        for reason in found:
            print(f"  - {reason}", file=sys.stderr)
        return EXIT_NOT_USABLE

    print("  receipt, runtime cache key and artifact all verified")
    print("\nOK: the compiled sparse_mla_sm120 module will be reused")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
