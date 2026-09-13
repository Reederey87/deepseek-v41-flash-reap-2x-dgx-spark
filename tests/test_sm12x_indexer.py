#!/usr/bin/env python3
"""Checks for `overlay/patch_sm12x_indexer.py` and `overlay/sm12x_mqa.py`.

What is pinned here:

1. The patch makes exactly the two intended additive edits to
   `utils/deep_gemm.py`, is idempotent, and fails closed on a drifted,
   half-applied, or already-patched source.
2. The hook runs **before** `_lazy_init()` and before the `_missing()` guard, so
   it also holds where the DeepGEMM implementation is absent.
3. The dispatch contract, **executed** rather than string-matched: a non-64 FP8
   page on capability family 120 reaches the overlay kernel and bypasses
   DeepGEMM; a 64-state page, an MXFP4 Q, another capability family, and a CPU
   platform all stay on DeepGEMM; and the inputs the overlay was not written
   against (an unreviewed page geometry, a varlen `indices`, an unexpected cache
   layout) fail closed instead of dispatching.
4. The vendored overlay module stays scoped to the paged path — the contiguous
   kernels, the HyperConnection GEMM, and any direct-top-k code are absent, so
   the pinned scoring/filtering/top-k sequence is untouched.
5. The Dockerfile applies the patch before this suite runs.

The stock fixture is assembled from the patch module's own anchor constants, so
it cannot drift away from the anchors it exercises. This suite is host-side:
`sm12x_mqa.py` is only parsed, never imported, because importing it needs vLLM.
"""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / "overlay"
PATCH = OVERLAY / "patch_sm12x_indexer.py"
MQA = OVERLAY / "sm12x_mqa.py"

sys.path.insert(0, str(OVERLAY))

import patch_sm12x_indexer as si  # noqa: E402

DEEP_GEMM_REL = Path("utils/deep_gemm.py")

# A page holding this many states per page, which is the value DeepGEMM's arch-12
# FP8 logits kernel is templated for.
LEGAL_BLOCK_KV = 64
# What this checkpoint's ratio-2 indexer layers present at --block-size 64, and
# the whole reason this patch exists.
UNSERVABLE_BLOCK_KV = 32


class _Platform:
    """A stand-in for `current_platform` at a fixed capability family.

    `is_device_capability_family` divides by ten, so family 120 means major 12 —
    what GB10's (12, 1) reports.
    """

    def __init__(self, major: int, cuda: bool = True) -> None:
        self.major = major
        self.cuda = cuda

    def is_cuda(self) -> bool:
        return self.cuda

    def is_device_capability_family(self, capability: int) -> bool:
        return self.cuda and capability // 10 == self.major


class _Tensor:
    """Just enough of a tensor for the hook's shape checks."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = tuple(shape)

    def dim(self) -> int:
        return len(self.shape)


def _page(block_kv: int) -> _Tensor:
    """The 4D packed indexer page the caller builds for a given state count."""
    return _Tensor((8, block_kv, 1, 128 + 4))


# ---------------------------------------------------------------- fixtures


_STOCK_HEAD = '''from __future__ import annotations

from vllm.platforms import current_platform

lazy_init_calls = 0
deepgemm_calls = 0
_fp8_fp4_paged_mqa_logits_impl = None


def _lazy_init():
    global lazy_init_calls, _fp8_fp4_paged_mqa_logits_impl
    lazy_init_calls += 1
    # The real module binds the implementation here, which is what makes the
    # `_fp8_fp4_paged_mqa_logits_impl is None` guard below mean "not yet
    # initialised" rather than "no DeepGEMM on this host".
    _fp8_fp4_paged_mqa_logits_impl = object()


def _missing(*args, **kwargs):
    raise RuntimeError("DeepGEMM is not available")


'''

_STOCK_SIG = """    q,
    kv_cache,
    weights,
    context_lens,
    block_tables,
    schedule_metadata,
    max_model_len,
    clean_logits,
    indices=None,
):
"""

# The patch's own anchor, so the fixture's body is the shape the patch expects.
_STOCK_BODY = si._CALL_OLD + """    global deepgemm_calls
    deepgemm_calls += 1
    return "deepgemm"
"""


def _stock_text() -> str:
    return _STOCK_HEAD + si._HOOK_OLD + _STOCK_SIG + _STOCK_BODY


def _write_stock(root: Path) -> None:
    for target in si.TARGETS:
        path = root / target.rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_stock_text())


def _apply(root: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(PATCH)],
        capture_output=True,
        text=True,
        env={**os.environ, "DSV41_VLLM_ROOT": str(root)},
    )


def _patched_text(root: Path) -> str:
    return (root / DEEP_GEMM_REL).read_text()


def _sm12x_stub(calls: list[int]) -> types.ModuleType:
    """A stand-in for the overlay module, recording that it was reached."""
    module = types.ModuleType("sm12x_mqa")

    def fp8_paged_mqa_logits_sm12x(
        q_values, kv_cache, weights, context_lens, block_tables, max_model_len
    ):
        calls.append(max_model_len)
        return "sm12x"

    module.fp8_paged_mqa_logits_sm12x = fp8_paged_mqa_logits_sm12x
    return module


@contextlib.contextmanager
def _loaded(text: str, platform: _Platform, calls: list[int]):
    """Exec the module with `vllm.platforms` and `sm12x_mqa` stubbed.

    The stubs must stay installed for the duration of the `with`, not just the
    exec: the hook's import of `sm12x_mqa` is function-local, so it resolves at
    call time.
    """
    fake_vllm = types.ModuleType("vllm")
    fake_platforms = types.ModuleType("vllm.platforms")
    fake_platforms.current_platform = platform
    fake_vllm.platforms = fake_platforms

    names = ("vllm", "vllm.platforms", "sm12x_mqa")
    saved = {name: sys.modules.get(name) for name in names}
    sys.modules["vllm"] = fake_vllm
    sys.modules["vllm.platforms"] = fake_platforms
    sys.modules["sm12x_mqa"] = _sm12x_stub(calls)
    try:
        namespace: dict = {}
        exec(compile(text, "<deep_gemm>", "exec"), namespace)  # noqa: S102
        yield namespace
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _call(namespace: dict, *, block_kv: int, q_scale=None, indices=None, page=None):
    return namespace["fp8_fp4_paged_mqa_logits"](
        (object(), q_scale),
        _page(block_kv) if page is None else page,
        object(),
        object(),
        object(),
        object(),
        4096,
        False,
        indices=indices,
    )


def _patched_root() -> tuple[tempfile.TemporaryDirectory, Path]:
    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name)
    _write_stock(root)
    assert _apply(root).returncode == 0
    return tmp, root


# ------------------------------------------------------------------ the patch


def test_patch_touches_only_the_paged_wrapper() -> None:
    rels = sorted({str(t.rel) for t in si.TARGETS})
    assert rels == [str(DEEP_GEMM_REL)], rels
    assert len(si.TARGETS) == 2, [t.name for t in si.TARGETS]


def test_every_edit_is_purely_additive() -> None:
    """No stock line may be dropped, so no other platform's behaviour can shift."""
    for target in si.TARGETS:
        assert si._is_ordered_subsequence(
            target.old.splitlines(), target.new.splitlines()
        ), target.name
        assert target.delta > 0, target.name


def test_injected_constants_are_interpolated_from_this_scripts_values() -> None:
    """The hook reads module-level names, so those names must be injected.

    They were not, at first: the hook referenced constants that existed only in
    this patch script, so the patched module raised `NameError` at call time and
    every legal-geometry request died. Interpolating from the script's own
    constants keeps the injected source and this script from drifting.
    """
    assert (
        f"_DSV41_ARCH12_FP8_BLOCK_KV = {si._DEEPGEMM_ARCH12_FP8_BLOCK_KV}"
        in si._HOOK_NEW
    ), si._HOOK_NEW
    assert (
        f"_DSV41_METADATA_BLOCK_KV = {si._METADATA_BLOCK_KV}" in si._HOOK_NEW
    ), si._HOOK_NEW
    # And the body must actually read them, rather than inlining literals that
    # could then drift from the constants.
    assert "_DSV41_ARCH12_FP8_BLOCK_KV" in si._HOOK_BODY, si._HOOK_BODY
    assert "_DSV41_METADATA_BLOCK_KV" in si._HOOK_BODY, si._HOOK_BODY


def test_hook_runs_before_lazy_init_and_before_the_missing_guard() -> None:
    """The reviewer's required shape: intercept before DeepGEMM is initialised.

    Returning before `_lazy_init()` is what lets the fallback also serve a tree
    where the DeepGEMM implementation is absent, and returning before the
    `_missing()` guard is what keeps a missing implementation from raising first.
    """
    body = si._CALL_NEW
    # Matched as whole statements: the rationale comment above the hook also
    # contains the text `_lazy_init()`, and a bare substring search would find
    # the comment instead of the call.
    lazy = "\n    _lazy_init()\n"
    missing = "\n    if _fp8_fp4_paged_mqa_logits_impl is None:\n"
    assert lazy in body and missing in body, body
    assert body.index("_dsv41_logits = ") < body.index(lazy), body
    assert body.index("return _dsv41_logits") < body.index(missing), body


def test_classification_distinguishes_stock_from_patched() -> None:
    """`classify` is a block test, so the two blocks must be tellable apart.

    Deliberately *not* the disjointness the block-size patch's suite requires.
    These edits may only insert lines, so each replacement necessarily still
    contains its own anchor — the hook edit even ends with it. What makes the
    test sound is that the replacement carries content the stock form cannot
    contain (88 lines of helper, or the `_dsv41_logits` call), so `new in text`
    remains a real discriminator. Both directions are executed, not argued.
    """
    for target in si.TARGETS:
        assert target.new != target.old, target.name
        assert target.old in target.new, target.name
        assert target.classify(_stock_text()) == "original", target.name
        patched = _stock_text().replace(target.old, target.new)
        assert target.classify(patched) == "patched", target.name


def test_patch_applies_and_is_idempotent() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_stock(root)

        first = _apply(root)
        assert first.returncode == 0, (first.returncode, first.stdout, first.stderr)
        once = _patched_text(root)
        for target in si.TARGETS:
            assert once.count(target.new) == 1, target.name
        compile(once, str(root / DEEP_GEMM_REL), "exec")

        second = _apply(root)
        assert second.returncode == 0, second.stderr
        assert "already applied" in second.stdout, second.stdout
        assert _patched_text(root) == once, "a second run must not change the file"


def test_patch_refuses_a_drifted_anchor() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_stock(root)
        drifted = _stock_text().replace("return _missing()", "return _missing(1)")
        (root / DEEP_GEMM_REL).write_text(drifted)

        r = _apply(root)
        assert r.returncode != 0, (r.returncode, r.stdout)
        assert "unexpected anchor state" in r.stderr, r.stderr
        assert _patched_text(root) == drifted, "a refused patch must not touch the file"


def test_patch_fails_closed_on_a_missing_file() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        r = _apply(root)
        assert r.returncode != 0, (r.returncode, r.stdout)
        assert "missing file" in r.stderr, r.stderr


# --------------------------------------------------------------- the dispatch


def test_fallback_serves_a_non_64_fp8_page_on_family_120() -> None:
    """The failing configuration, executed: the overlay kernel serves it."""
    tmp, root = _patched_root()
    with tmp:
        calls: list[int] = []
        with _loaded(_patched_text(root), _Platform(12), calls) as ns:
            got = _call(ns, block_kv=UNSERVABLE_BLOCK_KV)
        assert got == "sm12x", got
        assert calls == [4096], calls


def test_fallback_runs_before_lazy_init_at_runtime() -> None:
    """Not just textually: the DeepGEMM init must not have happened."""
    tmp, root = _patched_root()
    with tmp:
        with _loaded(_patched_text(root), _Platform(12), []) as ns:
            _call(ns, block_kv=UNSERVABLE_BLOCK_KV)
            assert ns["lazy_init_calls"] == 0, "the fallback must precede _lazy_init()"
            assert ns["deepgemm_calls"] == 0, "DeepGEMM must be bypassed entirely"
            # And the 64-state page must still initialise and use DeepGEMM.
            assert _call(ns, block_kv=LEGAL_BLOCK_KV) == "deepgemm"
            assert ns["lazy_init_calls"] == 1
            assert ns["deepgemm_calls"] == 1


def test_deepgemm_stays_the_default_everywhere_else() -> None:
    """This is a fallback, not a replacement: everything else is untouched."""
    tmp, root = _patched_root()
    with tmp:
        cases = {
            "the geometry DeepGEMM has (64 states)": dict(block_kv=LEGAL_BLOCK_KV),
            "MXFP4 Q": dict(block_kv=UNSERVABLE_BLOCK_KV, q_scale=object()),
        }
        for label, kwargs in cases.items():
            calls: list[int] = []
            with _loaded(_patched_text(root), _Platform(12), calls) as ns:
                assert _call(ns, **kwargs) == "deepgemm", label
            assert calls == [], label

        # Other capability families, and a non-CUDA platform, must not be routed.
        for major, cuda, label in (
            (10, True, "family 100 (datacenter Blackwell)"),
            (9, True, "family 90 (Hopper)"),
            (12, False, "no CUDA"),
        ):
            calls = []
            with _loaded(_patched_text(root), _Platform(major, cuda=cuda), calls) as ns:
                assert _call(ns, block_kv=UNSERVABLE_BLOCK_KV) == "deepgemm", label
            assert calls == [], label


def test_stock_fixture_takes_the_deepgemm_path() -> None:
    """The fixture must actually reproduce the bug, or the tests above prove nothing.

    Unpatched, a 32-state page reaches DeepGEMM — which is the call that aborts
    on real hardware with `attention.hpp:320`.
    """
    calls: list[int] = []
    with _loaded(_stock_text(), _Platform(12), calls) as ns:
        assert _call(ns, block_kv=UNSERVABLE_BLOCK_KV) == "deepgemm"
    assert calls == []
    assert "def _dsv41_sm12x_fp8_paged_mqa_logits" not in _stock_text()


def test_dispatch_fails_closed_on_unreviewed_inputs() -> None:
    """Never guess: an input the overlay was not written against must raise."""
    tmp, root = _patched_root()
    with tmp:
        # A page outside the metadata builder's own set cannot be reached in
        # practice; if it is, the geometry was never reviewed.
        with _loaded(_patched_text(root), _Platform(12), []) as ns:
            try:
                _call(ns, block_kv=16)
            except RuntimeError as exc:
                assert "invariant" in str(exc), exc
            else:
                raise AssertionError("an unreviewed page geometry must fail closed")

        # varlen `indices` is family-100 only and has no overlay implementation.
        with _loaded(_patched_text(root), _Platform(12), []) as ns:
            try:
                _call(ns, block_kv=UNSERVABLE_BLOCK_KV, indices=object())
            except RuntimeError as exc:
                assert "varlen" in str(exc), exc
            else:
                raise AssertionError("varlen indices must fail closed")

        # An unexpected cache layout would silently misread every scale.
        with _loaded(_patched_text(root), _Platform(12), []) as ns:
            try:
                _call(ns, block_kv=UNSERVABLE_BLOCK_KV, page=_Tensor((8, 32, 132)))
            except RuntimeError as exc:
                assert "packed indexer page" in str(exc), exc
            else:
                raise AssertionError("an unexpected page layout must fail closed")


# ------------------------------------------------- the vendored overlay module


def _defs(text: str) -> set[str]:
    return {
        node.name
        for node in ast.walk(ast.parse(text))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def test_overlay_module_is_scoped_to_the_paged_path() -> None:
    """The port must not drag in the contiguous, HC, or direct-top-k paths.

    Those either take no `block_kv` from the cache geometry (so they never hit
    the restriction) or would change the pinned scoring / candidate-filtering /
    top-k sequence. Their absence is what keeps this patch a fallback rather
    than a behavioural change.
    """
    text = MQA.read_text()
    names = _defs(text)

    assert "fp8_paged_mqa_logits_sm12x" in names
    for vendored in (
        "_bucketed_logits_buffer",
        "_view_packed_fp8_paged_mqa_kv_cache",
        "_fp8_paged_mqa_logits_kernel",
        "_fp8_paged_mqa_logits_rowwise_kernel",
        "fp8_paged_mqa_logits_rowwise_triton",
        "fp8_paged_mqa_logits_triton",
    ):
        assert vendored in names, vendored

    for excluded in (
        "_fp8_mqa_logits_kernel",
        "fp8_mqa_logits_triton",
        "_fp8_mqa_logits_block_m",
        "_tf32_hc_prenorm_gemm_kernel",
        "tf32_hc_prenorm_gemm_triton",
    ):
        assert excluded not in names, excluded

    # No direct-top-k / fused top-k code, in any spelling.
    lowered = text.lower()
    assert "topk" not in lowered, "fused top-k code must not be vendored"
    assert "top_k" not in lowered, "fused top-k code must not be vendored"


def test_overlay_entry_point_delegates_to_the_paged_dispatcher() -> None:
    """The entry point must be a thin delegate, not a second implementation."""
    tree = ast.parse(MQA.read_text())
    entry = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "fp8_paged_mqa_logits_sm12x"
    )
    body = [node for node in entry.body if not isinstance(node, ast.Expr)]
    assert len(body) == 1, "the entry point must have exactly one statement"
    call = body[0]
    assert isinstance(call, ast.Return) and isinstance(call.value, ast.Call)
    assert call.value.func.id == "fp8_paged_mqa_logits_triton", ast.dump(call.value.func)


# ------------------------------------------------------------- the Dockerfile


def dockerfile_defects(root: Path) -> list[str]:
    """Why `root`'s Dockerfile does not install this patch before running this suite."""
    path = root / "Dockerfile"
    if not path.is_file():
        return [f"no Dockerfile at {path}; the ordering cannot be verified"]
    text = path.read_text()
    defects = []
    patch_at = text.find("patch_sm12x_indexer.py")
    test_at = text.find("test_sm12x_indexer.py")
    if patch_at == -1:
        defects.append("the Dockerfile does not apply the SM12x indexer patch")
    if test_at == -1:
        defects.append("the Dockerfile does not run this test suite")
    if not defects and patch_at > test_at:
        defects.append(
            "patch_sm12x_indexer.py must run before test_sm12x_indexer.py"
        )

    # The hook imports `sm12x_mqa` at call time, so the overlay directory has to
    # be importable in the served process: the tree must be copied into the
    # image, and that same directory must be on PYTHONPATH. The pre-flight cannot
    # check this half — it only reads the installed `deep_gemm.py` — so this is
    # the gate for it. Without both, the fallback raises `ModuleNotFoundError` on
    # the first 32-state decode instead of serving it, which is the very failure
    # it was written to remove.
    dests = [
        line.split()[2].rstrip("/")
        for line in text.splitlines()
        if line.startswith("COPY overlay/")
    ]
    if not dests:
        defects.append(
            "the Dockerfile does not COPY the overlay tree, so "
            "overlay/sm12x_mqa.py is not in the image the hook imports from"
        )
    else:
        pythonpath = [
            entry
            for line in text.splitlines()
            if line.startswith("ENV PYTHONPATH=")
            for entry in line[len("ENV PYTHONPATH=") :].split(":")
        ]
        if not any(dest in pythonpath for dest in dests):
            defects.append(
                "the overlay directory is not on PYTHONPATH, so the hook's "
                "function-local `from sm12x_mqa import ...` cannot resolve in "
                "the served process"
            )
    return defects


_DOCKERFILE_OK = """FROM base
ENV PYTHONPATH=/opt/dsv41/overlay
COPY overlay/ /opt/dsv41/overlay/
RUN python3 /opt/dsv41/overlay/patch_sm12x_indexer.py
RUN python3 /opt/dsv41/tests/test_sm12x_indexer.py
"""


def test_dockerfile_installs_the_patch_before_this_suite() -> None:
    defects = dockerfile_defects(ROOT)
    assert defects == [], defects


def test_dockerfile_defects_report_each_way_the_install_can_break() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        dockerfile = root / "Dockerfile"

        def defects(text: str) -> list[str]:
            dockerfile.write_text(text)
            return dockerfile_defects(root)

        assert defects(_DOCKERFILE_OK) == [], defects(_DOCKERFILE_OK)

        got = defects(_DOCKERFILE_OK.replace("patch_sm12x_indexer.py", "x.py", 1))
        assert any("does not apply" in d for d in got), got

        got = defects(
            _DOCKERFILE_OK.replace(
                "RUN python3 /opt/dsv41/tests/test_sm12x_indexer.py\n", ""
            )
        )
        assert any("does not run this test suite" in d for d in got), got

        # The suite before the patch: it would be asserting on a stock tree.
        got = defects(
            "FROM base\n"
            "ENV PYTHONPATH=/opt/dsv41/overlay\n"
            "COPY overlay/ /opt/dsv41/overlay/\n"
            "RUN python3 /opt/dsv41/tests/test_sm12x_indexer.py\n"
            "RUN python3 /opt/dsv41/overlay/patch_sm12x_indexer.py\n"
        )
        assert any("must run before" in d for d in got), got

        got = defects(
            _DOCKERFILE_OK.replace("COPY overlay/ /opt/dsv41/overlay/\n", "")
        )
        assert any("does not COPY the overlay tree" in d for d in got), got

        got = defects(
            _DOCKERFILE_OK.replace("ENV PYTHONPATH=/opt/dsv41/overlay\n", "")
        )
        assert any("not on PYTHONPATH" in d for d in got), got

        # Copied somewhere else and never put on the path: the import would
        # still fail, so the check must compare the two rather than just
        # looking for the word "overlay".
        got = defects(
            _DOCKERFILE_OK.replace(
                "COPY overlay/ /opt/dsv41/overlay/", "COPY overlay/ /opt/elsewhere/"
            )
        )
        assert any("not on PYTHONPATH" in d for d in got), got


def _real_vllm_installed() -> bool:
    spec = importlib.util.find_spec("vllm")
    return spec is not None and bool(spec.submodule_search_locations)


def _target_states(text: str) -> dict[str, str]:
    """Per-target state of a `deep_gemm.py`-shaped text: patched, stock or drift.

    The edits are purely additive, so `new` embeds `old` and a *patched* file
    still contains its own anchor. `new` therefore has to win: testing `old`
    independently reads a fully patched file as both states at once, and the
    check then refuses its own build output. That is not hypothetical — the
    Dockerfile patches the tree before this suite runs, so the ordering makes a
    fully patched file the *expected* one inside the image.
    """
    states: dict[str, str] = {}
    for target in si.TARGETS:
        if target.new in text:
            states[target.name] = "patched"
        elif target.old in text:
            states[target.name] = "stock"
        else:
            states[target.name] = "drift"
    return states


def installed_source_verdict(text: str) -> str:
    """`patched`, `stock`, or why the installed `deep_gemm.py` is refused."""
    states = _target_states(text)
    if "drift" in states.values():
        return "drift"
    if len(set(states.values())) != 1:
        return "half"
    return next(iter(states.values()))


def test_installed_source_verdict_tells_the_four_states_apart() -> None:
    """Every state this check must distinguish, exercised without vLLM present.

    The installed-source check only runs inside the image, which is exactly
    where a mistake in it breaks the build, so the classification is separated
    out and driven here on all four states: stock, fully patched, and each edit
    applied alone.
    """
    stock = _stock_text()
    assert installed_source_verdict(stock) == "stock", _target_states(stock)

    hook = next(t for t in si.TARGETS if "hook" in t.name)
    call = next(t for t in si.TARGETS if "call" in t.name)

    full = stock.replace(hook.old, hook.new).replace(call.old, call.new)
    assert installed_source_verdict(full) == "patched", _target_states(full)

    # Each edit alone: the helper without its call site is inert, and the call
    # site without the helper is a NameError. Both are half-applied.
    edit1 = stock.replace(hook.old, hook.new)
    assert installed_source_verdict(edit1) == "half", _target_states(edit1)
    edit2 = stock.replace(call.old, call.new)
    assert installed_source_verdict(edit2) == "half", _target_states(edit2)

    # Neither form: a base-image bump that reshapes the anchors.
    assert installed_source_verdict("def unrelated():\n    pass\n") == "drift"


def test_installed_vllm_is_stock_or_fully_patched() -> None:
    """In the image: the installed `deep_gemm.py` is stock or fully patched.

    The Dockerfile applies this patch *before* it runs this suite, so inside the
    image the patched form is the expected one. Both states are legitimate and
    each is checked for what it means; a half-applied file, or one that is
    neither form, is a base-image drift and must be re-derived before building.
    Skipped on this Mac, where vLLM is absent, so it runs exactly where it
    means something.
    """
    if not _real_vllm_installed():
        print("  (skipped: vllm not installed)")
        return
    spec = importlib.util.find_spec("vllm")
    installed = Path(next(iter(spec.submodule_search_locations)))
    text = (installed / DEEP_GEMM_REL).read_text()

    verdict = installed_source_verdict(text)
    if verdict == "drift":
        raise AssertionError(
            f"{DEEP_GEMM_REL} is neither the stock nor the patched form of this "
            f"patch's anchors ({_target_states(text)}); the base image drifted, "
            "so the anchor must be re-derived before building"
        )
    assert verdict != "half", (
        f"the installed vLLM is half-patched: {_target_states(text)}"
    )
    patched = verdict == "patched"

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "vllm"
        dst = root / DEEP_GEMM_REL
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(installed / DEEP_GEMM_REL, dst)

        r = _apply(root)
        assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
        if patched:
            assert "already applied" in r.stdout, r.stdout
        got = dst.read_text()
        for target in si.TARGETS:
            assert got.count(target.new) == 1, target.name
        compile(got, str(dst), "exec")
        if patched:
            assert got == text, "a no-op must not rewrite the installed bytes"

    print(f"  (installed {DEEP_GEMM_REL} was {verdict})")


if __name__ == "__main__":
    test_patch_touches_only_the_paged_wrapper()
    test_every_edit_is_purely_additive()
    test_hook_runs_before_lazy_init_and_before_the_missing_guard()
    test_classification_distinguishes_stock_from_patched()
    test_patch_applies_and_is_idempotent()
    test_patch_refuses_a_drifted_anchor()
    test_patch_fails_closed_on_a_missing_file()
    test_fallback_serves_a_non_64_fp8_page_on_family_120()
    test_fallback_runs_before_lazy_init_at_runtime()
    test_deepgemm_stays_the_default_everywhere_else()
    test_stock_fixture_takes_the_deepgemm_path()
    test_dispatch_fails_closed_on_unreviewed_inputs()
    test_overlay_module_is_scoped_to_the_paged_path()
    test_overlay_entry_point_delegates_to_the_paged_dispatcher()
    test_dockerfile_installs_the_patch_before_this_suite()
    test_dockerfile_defects_report_each_way_the_install_can_break()
    test_installed_source_verdict_tells_the_four_states_apart()
    test_installed_vllm_is_stock_or_fully_patched()
    print("sm12x indexer patch tests OK")
