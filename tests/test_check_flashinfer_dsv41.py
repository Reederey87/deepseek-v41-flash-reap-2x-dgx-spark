#!/usr/bin/env python3
"""Checks for the SM120 sparse-MLA pre-flight (`overlay/check_flashinfer_dsv41.py`).

The pre-flight reports four independent things: the decode dispatch keys, the
DeepGEMM `block_kv` the indexer will ask for, the compressed pool's page block
per ratio (read from the **installed** prefill source), and the AOT state. These
tests pin the derivations, the fail-closed behaviour, and the acceptance of the
32-state compressed page that `overlay/patch_flashinfer_prefill_pbsx.py` adds
(RESEARCH 11.9.15, 11.9.16).

The FlashInfer side is stubbed through PYTHONPATH so the assertions hold both in
the image, where FlashInfer is present, and on a machine where it is not.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "overlay" / "check_flashinfer_dsv41.py"
sys.path.insert(0, str(ROOT / "overlay"))
sys.path.insert(0, str(ROOT / "tests"))

import check_flashinfer_dsv41 as chk  # noqa: E402
from _fi_fixture import (  # noqa: E402
    branch_only_cu,
    dual_guard_only_cu,
    no_fulltile_branch_cu,
    no_plain_branch_cu,
    patched_cu,
    stock_cu,
    trap_cu,
)

# The real REAP-256E shape: 40 layers, ratios 0/1/2, index_topk 512, window 128,
# and a vision tower (32 layers, 1024-token images) whose per-image budget
# widens the SWA index rows to 1152 unless the engine is told there is no image.
V41_CONFIG = {
    "architectures": ["DeepseekV41ForCausalLM"],
    "text_config": {
        "num_attention_heads": 64,
        "sliding_window": 128,
        "index_topk": 512,
        "compress_ratios": [0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0],
    },
    "vision_config": {"num_hidden_layers": 32, "max_image_tokens": 1024},
}

# The posture `start.sh` serves: `--hf-overrides '{"vision_max_n_token": 0}'`,
# which collapses the SWA index rows back to `sliding_window`.
TEXT_ONLY = 0

# The width the checkpoint's own config produces: 128 + 1024.
VISION_WIDTH = 1152

# The manager block size `start.sh` pins: the only one that lands both V4.1
# ratios on DeepGEMM's `block_kv` set, giving compressed pages of 64 and 32.
SERVED_BLOCK_SIZE = 64

# The compressed page blocks the patched dispatcher must admit and instantiate.
PATCHED_PAGE_BLOCKS = frozenset({64, 2, 128, 32})


def test_required_key_is_the_window_width() -> None:
    """Our tree passes the SWA segment as the main index, so the key is 128."""
    pairs = chk.required_pairs(V41_CONFIG, 2, TEXT_ONLY)
    assert list(pairs) == [(32, 128)], pairs
    assert "main/SWA" in pairs[(32, 128)]


def test_image_budget_widens_the_required_key_past_every_instantiation() -> None:
    """The blocker this fix removes: the declared config wants a 1152-wide key.

    `attention.py` and `sparse_swa.py` size the prefill SWA rows at
    `sliding_window + vision_max_n_token`, and 1152 is in neither the decode set
    ({128,192,256,512,1024}) nor the single prefill ladder (those plus 2048); the
    dual path this tree takes requires exactly 128.
    """
    declared = chk.required_pairs(V41_CONFIG, 2)
    assert list(declared) == [(32, VISION_WIDTH)], declared
    assert chk.swa_index_width(V41_CONFIG)[0] == VISION_WIDTH
    assert chk.swa_index_width(V41_CONFIG, TEXT_ONLY)[0] == 128


def test_zeroing_the_image_budget_is_what_collapses_the_width() -> None:
    """The lever is the per-image token budget, not the vision layer count.

    `{"vision_n_layers": 0}` also yields 128, but the MoE gate only creates
    `bias_vl` when the layer count is positive, so zeroing it makes this
    checkpoint's `layers.*.ffn.gate.bias_vl` weights fail to load
    (`KeyError`). The width must therefore follow `max_image_tokens` alone.
    """
    assert chk.declared_image_budget(V41_CONFIG) == 1024
    # A checkpoint with no vision tower declares no image span at all.
    no_vision = json.loads(json.dumps(V41_CONFIG))
    no_vision["vision_config"]["num_hidden_layers"] = 0
    assert chk.declared_image_budget(no_vision) == 0
    assert chk.swa_index_width(no_vision)[0] == 128
    # Zeroing the budget keeps the layer count, and with it bias_vl.
    assert chk.config_vision_n_layers(V41_CONFIG) == 32
    assert chk.swa_index_width(V41_CONFIG, 0)[0] == 128


def test_swa_width_ignores_vision_without_a_max_image_tokens() -> None:
    """Vision layers alone do not widen anything; the image cap is what does."""
    cfg = json.loads(json.dumps(V41_CONFIG))
    del cfg["vision_config"]["max_image_tokens"]
    assert chk.swa_index_width(cfg)[0] == VISION_WIDTH, (
        "max_image_tokens defaults to 1024 in the v4.1 config class"
    )
    cfg["vision_config"]["max_image_tokens"] = 0
    assert chk.swa_index_width(cfg)[0] == 128


def test_effective_budget_is_what_is_checked() -> None:
    """`--max-image-tokens` must override the checkpoint's declared budget.

    Otherwise the verifier would keep checking the 1152-wide table the
    checkpoint declares while the engine actually runs 128, and pass a boot that
    cannot happen.
    """
    assert chk.config_vision_n_layers(V41_CONFIG) == 32
    assert chk.config_vision_n_layers({}) == 0
    assert chk.config_vision_n_layers({"vision_config": {"num_hidden_layers": -1}}) == 0
    try:
        chk.required_pairs(V41_CONFIG, 2, -1)
    except ValueError as exc:
        assert "max_image_tokens" in str(exc), exc
    else:
        raise AssertionError("a negative effective budget must be rejected")


def test_required_key_is_per_rank_and_padded() -> None:
    """FlashInfer keys on the padded per-rank head count."""
    assert (32, 128) in chk.required_pairs(V41_CONFIG, 2, TEXT_ONLY)
    assert (16, 128) in chk.required_pairs(V41_CONFIG, 4, TEXT_ONLY)
    assert (64, 128) in chk.required_pairs(V41_CONFIG, 1, TEXT_ONLY)
    # 64/6 is not integral, and 40 heads would pad to 64, not stay 40.
    cfg = json.loads(json.dumps(V41_CONFIG))
    cfg["text_config"]["num_attention_heads"] = 48
    assert (64, 128) in chk.required_pairs(cfg, 1, TEXT_ONLY), chk.required_pairs(
        cfg, 1, TEXT_ONLY
    )
    assert chk.pad_q_heads(40) == 64
    assert chk.pad_q_heads(32) == 32


def test_indivisible_heads_is_an_error() -> None:
    try:
        chk.required_pairs(V41_CONFIG, 5, TEXT_ONLY)
    except ValueError as exc:
        assert "not divisible" in str(exc), exc
    else:
        raise AssertionError("a non-divisible head count must be rejected")


def test_combined_widths_are_candidates_not_requirements() -> None:
    """640/1152 are reported as candidates: the recipe's call shape, not ours."""
    cands = chk.candidate_combined_pairs(V41_CONFIG, 2)
    assert set(cands) == {(32, 640), (32, VISION_WIDTH)}, cands
    assert "combined width" in cands[(32, 640)]
    assert "compress_ratio 1" in cands[(32, 640)]
    assert "compress_ratio 2" in cands[(32, VISION_WIDTH)]
    # They must never appear in the required set of the served posture.
    assert not (set(cands) & set(chk.required_pairs(V41_CONFIG, 2, TEXT_ONLY)))


def test_no_index_topk_yields_no_combined_candidates() -> None:
    cfg = json.loads(json.dumps(V41_CONFIG))
    cfg["text_config"]["index_topk"] = 0
    assert chk.candidate_combined_pairs(cfg, 2) == {}
    # The required key does not depend on index_topk.
    assert (32, 128) in chk.required_pairs(cfg, 2, TEXT_ONLY)


def test_single_ladder_widths_are_read_from_the_source() -> None:
    """The single dispatcher's ladder is parsed for the operator's benefit."""
    assert chk.prefill_single_topk_values(stock_cu()) == frozenset(
        {128, 192, 256, 512, 1024, 2048}
    )
    # The dual dispatcher's `topk == 128` guard is a different contract and must
    # not be mistaken for a member of that ladder: a bare guard alone reads as
    # empty, which is what distinguishes the two.
    assert chk.prefill_single_topk_values(dual_guard_only_cu()) == frozenset()


def test_prefill_width_defect_accepts_only_the_dual_width() -> None:
    """The live prefill path requires exactly 128, whatever the single ladder has."""
    assert chk.prefill_width_defect(128, stock_cu()) is None
    for width in (1152, 640, 192, 1024, 2048):
        defect = chk.prefill_width_defect(width, stock_cu())
        assert defect is not None, width
        assert "dispatch_dsv4_dual" in defect, defect
        assert "128" in defect, defect


def test_page_block_parsing_reads_the_source() -> None:
    """The accepted and instantiated sets come from the installed .cu."""
    stock = chk.prefill_page_blocks(stock_cu())
    assert stock == (frozenset({64, 2}), frozenset({64, 2})), stock
    patched = chk.prefill_page_blocks(patched_cu())
    assert patched == (PATCHED_PAGE_BLOCKS, PATCHED_PAGE_BLOCKS), patched
    # A widened acceptance test with a bare `else` is the mismatch case.
    trap = chk.prefill_page_blocks(trap_cu())
    assert trap == (PATCHED_PAGE_BLOCKS, frozenset({64, 2})), trap


def test_compressed_page_blocks_are_block_size_over_ratio() -> None:
    """The page block is `block_size // compress_ratio`, per compressed ratio.

    This is also the value the indexer hands DeepGEMM as `block_kv`, which is why
    only one of the two block sizes below can boot.
    """
    assert chk.compressed_page_blocks(V41_CONFIG, 128) == [(1, 128), (2, 64)]
    assert chk.compressed_page_blocks(V41_CONFIG, 64) == [(1, 64), (2, 32)]


def test_deepgemm_metadata_and_logits_sets_are_distinct() -> None:
    """DeepGEMM gates paged MQA twice, and the two sets are not the same.

    The metadata builder takes 32 or 64 on arch 12; the FP8 logits kernel takes
    64 only. Treating the metadata set as the logits set is what previously made
    this check accept a 32-state page that boots, warms up, and then kills the
    engine on the first long prompt.
    """
    assert chk.DEEPGEMM_METADATA_BLOCK_KV == frozenset({32, 64})
    assert chk.DEEPGEMM_FP8_LOGITS_BLOCK_KV == frozenset({64})
    # The whole point: 32 passes the first gate and fails the second.
    assert 32 in chk.DEEPGEMM_METADATA_BLOCK_KV
    assert 32 not in chk.DEEPGEMM_FP8_LOGITS_BLOCK_KV


def test_indexer_defects_report_which_gate_rejected_the_page() -> None:
    """A defect names its gate, because the two fail very differently.

    Without the fallback, the kit's pinned {64, 32} is a *logits* defect: it
    survives warmup and dies in decode. With the fallback installed the 32 is
    served and there is no defect at all. A page outside the metadata set is a
    metadata defect either way — the fallback replaces the logits call, not the
    metadata build.
    """
    pinned = [(1, 64), (2, 32)]
    assert chk.indexer_block_kv_defects(pinned, fallback_active=False) == [
        (2, 32, "logits")
    ]
    assert chk.indexer_block_kv_defects(pinned, fallback_active=True) == []

    # 128 is outside the metadata set, so no fallback can rescue it.
    wide = [(1, 128), (2, 64)]
    assert chk.indexer_block_kv_defects(wide, fallback_active=False) == [
        (1, 128, "metadata")
    ]
    assert chk.indexer_block_kv_defects(wide, fallback_active=True) == [
        (1, 128, "metadata")
    ]


def test_only_block_size_64_lands_both_ratios_on_the_deepgemm_set() -> None:
    """The reason `start.sh` pins 64, stated as a property rather than a comment.

    64 is the only block size that puts both V4.1 ratios inside the metadata set
    without leaving one of them on the logits gate: 64 gives {64, 32}, whose 32
    needs the SM12x fallback; 128 gives {128, 64} and cannot boot at all; 32
    gives {32, 16} and 16 is below the metadata floor. Sweeping is what makes the
    pin self-justifying, and it must hold with the fallback installed, because
    that is the configuration this kit ships.
    """
    servable = [
        block_size
        for block_size in (32, 64, 128, 256)
        if not chk.indexer_block_kv_defects(
            chk.compressed_page_blocks(V41_CONFIG, block_size),
            fallback_active=True,
        )
    ]
    assert servable == [SERVED_BLOCK_SIZE], servable

    # And without the fallback, 64 is *not* servable — which is exactly the
    # failure the preflight previously reported as clean.
    assert chk.indexer_block_kv_defects(
        chk.compressed_page_blocks(V41_CONFIG, 64), fallback_active=False
    ) == [(2, 32, "logits")]


def test_acceptance_set_ignores_the_branch_headers() -> None:
    """The acceptance test is read on its own, not unioned with the branches.

    `branch_only_cu()` adds wide entries to the non-full-tile ladder but leaves
    the acceptance test alone. A union would report them as accepted and hide
    that asymmetry; reading the acceptance test separately reports the truth.
    """
    assert chk.acceptance_page_blocks(stock_cu()) == frozenset({64, 2})
    assert chk.acceptance_page_blocks(branch_only_cu()) == frozenset({64, 2})
    assert chk.acceptance_page_blocks(trap_cu()) == PATCHED_PAGE_BLOCKS
    assert chk.acceptance_page_blocks(patched_cu()) == PATCHED_PAGE_BLOCKS


def test_dispatch_defects_are_branch_local() -> None:
    """Each ladder is checked independently; the union of both is not the verdict."""
    needed = [64, 32]
    assert chk.dispatch_defects(stock_cu(), needed), "stock cannot serve 32"
    assert chk.dispatch_defects(patched_cu(), needed) == []
    # Edit 3 alone is *reachable*: the acceptance test still rejects 32, so the
    # full-tile path is never taken for it and the plain ladder serves it. That
    # makes it the case reachability cannot see — it is still only one of the
    # patch's three edits, and `completeness_defects` is what refuses it. See
    # the next two tests.
    assert chk.dispatch_defects(branch_only_cu(), needed) == []
    # Ratio 1 alone needs only 64, which stock serves.
    assert chk.dispatch_defects(stock_cu(), [64]) == []


def test_completeness_defects_catches_every_incomplete_combination() -> None:
    """Set agreement is the third leg, and it is not vacuous on either endpoint.

    Stock and the complete patch both enumerate `{64, 2}` and `{64, 2, 128, 32}`
    respectively in all three places, so they pass. Every partial combination
    disagrees — including edit 3 alone, whose reachable path works.
    """
    assert chk.completeness_defects(stock_cu()) == []
    assert chk.completeness_defects(patched_cu()) == []
    for name, cu in (
        ("edit 1 only (the trap)", trap_cu()),
        ("edits 1+3 (no full-tile branch)", no_fulltile_branch_cu()),
        ("edits 1+2 (no plain branch)", no_plain_branch_cu()),
        ("edit 3 only (the reachable one)", branch_only_cu()),
    ):
        assert chk.completeness_defects(cu), name
    # The disagreement is reported with all three sets, so the reader can see
    # which edit is missing.
    reason = chk.completeness_defects(branch_only_cu())[0]
    assert "do not enumerate the same compressed page blocks" in reason, reason
    assert "non-full-tile [2, 32, 64, 128]" in reason, reason
    assert "acceptance [2, 64]" in reason, reason


def test_partial_patch_of_only_the_plain_ladder_is_rejected() -> None:
    """Edit 3 alone serves this checkpoint and must still be refused.

    Every page block this checkpoint needs is served: the guard rejects 32, so
    the full-tile path is never taken for it, and the plain ladder — the branch
    this tree actually uses — instantiates it correctly. The image is still only
    one of the patch's three edits, and set agreement is the only check that
    sees it.
    """
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _config(tmp, V41_CONFIG)
        r = _run(
            cfg,
            block_size=SERVED_BLOCK_SIZE,
            stub=_stub(tmp, dispatch=[(32, 128)], cu=branch_only_cu()),
        )
        assert r.returncode == 5, (r.returncode, r.stdout, r.stderr)
        assert "acceptance and both ladders agree: NO" in r.stdout, r.stdout
        assert "incomplete patch:" in r.stderr, r.stderr
        assert "do not enumerate the same compressed page blocks" in r.stderr, r.stderr


def test_missing_fulltile_branch_is_a_wrong_kernel_defect() -> None:
    """Edits 1+3 without edit 2: 32 is admitted and run as PBSX(2).

    This is the case a union of instantiated values cannot see, because the
    non-full-tile ladder's new 32 entry covers for the full-tile branch.
    """
    cu = no_fulltile_branch_cu()
    assert chk.prefill_page_blocks(cu)[1] == PATCHED_PAGE_BLOCKS, (
        "the union must look complete, or this test is vacuous"
    )
    defects = chk.dispatch_defects(cu, [64, 32])
    assert [value for value, _reason in defects] == [32], defects
    assert "full-tile branch runs PBSX(2)" in defects[0][1], defects[0][1]


def test_missing_plain_branch_is_a_boot_failure_defect() -> None:
    """Edits 1+2 without edit 3: the live branch has no 32 entry."""
    cu = no_plain_branch_cu()
    assert chk.prefill_page_blocks(cu)[1] == PATCHED_PAGE_BLOCKS, (
        "the union must look complete, or this test is vacuous"
    )
    defects = chk.dispatch_defects(cu, [64, 32])
    assert [value for value, _reason in defects] == [32], defects
    assert "non-full-tile ladder returns false" in defects[0][1], defects[0][1]


def test_stock_block_set_is_the_vendored_one() -> None:
    """The stock set is a claim about FlashInfer; keep it pinned to the source."""
    assert chk.STOCK_EXTRA_PAGE_BLOCKS == frozenset({64, 2})
    assert chk.STOCK_EXTRA_PAGE_BLOCKS == chk.prefill_page_blocks(stock_cu())[0]


def test_block_size_128_is_refused_by_the_deepgemm_check() -> None:
    """The failure that blocked the last boot, caught before the weight load.

    At `--block-size 128` the ratio-1 pool is 128 states per page, which is
    outside the *metadata* builder's set as well as the logits kernel's, so the
    engine dies during `compile_or_warm_up_model` (vLLM #53635) and no logits
    fallback can help. The pre-flight must say so and exit before any FlashInfer
    verdict.
    """
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _config(tmp, V41_CONFIG)
        r = _run(
            cfg,
            block_size=128,
            stub=_stub(tmp, dispatch=[(32, 128)], cu=patched_cu()),
        )
        assert r.returncode == 6, (r.returncode, r.stdout, r.stderr)
        assert (
            "compress_ratio 1: num_states = block_size // ratio = 128" in r.stderr
        ), r.stderr
        assert "the metadata builder accepts only [32, 64]" in r.stderr, r.stderr
        assert "block_kv == 32 or block_kv == 64" in r.stderr, r.stderr
        assert "NOT SERVABLE (DeepGEMM indexer)" in r.stdout, r.stdout
        assert "OK: decode keys present" not in r.stdout, r.stdout


def test_ratio_two_page_block_is_refused_when_the_sm12x_fallback_is_absent() -> None:
    """The P1 defect, stated as a test: {64, 32} is NOT servable without the patch.

    At `--block-size 64` the kit's pinned geometry passes the metadata builder
    and fails the FP8 logits kernel. The previous form of this check called that
    clean and exited 0, which is the false acceptance that let a boot proceed
    into a guaranteed decode abort. With the fallback absent the pre-flight must
    refuse it, and must name the *logits* gate rather than the metadata one, so
    the operator knows which patch is missing.
    """
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _config(tmp, V41_CONFIG)
        r = _run(
            cfg,
            block_size=SERVED_BLOCK_SIZE,
            stub=_stub(
                tmp, dispatch=[(32, 128)], cu=patched_cu(), fallback="none"
            ),
        )
        assert r.returncode == 6, (r.returncode, r.stdout, r.stderr)
        assert (
            "compress_ratio 2: num_states = block_size // ratio = 32" in r.stderr
        ), r.stderr
        assert "the FP8 logits kernel accepts only [64]" in r.stderr, r.stderr
        assert "the SM12x fallback is NOT installed" in r.stderr, r.stderr
        assert "patch_sm12x_indexer.py" in r.stderr, r.stderr
        assert "NOT SERVABLE (DeepGEMM indexer)" in r.stdout, r.stdout
        assert "OK: decode keys present" not in r.stdout, r.stdout


def test_the_helper_without_its_call_site_is_not_a_fallback() -> None:
    """The half-applied patch: the helper is defined, but nothing calls it.

    The two edits are separate, so a tree can carry the first alone. The helper
    then exists and imports cleanly, but every 32-state page still reaches
    DeepGEMM and aborts exactly as before. A check that only looked for the
    helper's name would call this servable — which is the same false acceptance
    as the missing-patch case, one edit further along.
    """
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _config(tmp, V41_CONFIG)
        r = _run(
            cfg,
            block_size=SERVED_BLOCK_SIZE,
            stub=_stub(
                tmp, dispatch=[(32, 128)], cu=patched_cu(), fallback="helper"
            ),
        )
        assert r.returncode == 6, (r.returncode, r.stdout, r.stderr)
        assert "the SM12x fallback is NOT installed" in r.stderr, r.stderr
        assert "NOT SERVABLE (DeepGEMM indexer)" in r.stdout, r.stdout


def test_stock_image_refuses_the_ratio_two_page_block() -> None:
    """Without the FlashInfer patch, ratio-2 prefill is the blocker (exit 5)."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _config(tmp, V41_CONFIG)
        r = _run(
            cfg,
            block_size=SERVED_BLOCK_SIZE,
            stub=_stub(tmp, dispatch=[(32, 128)], cu=stock_cu()),
        )
        assert r.returncode == 5, (r.returncode, r.stdout, r.stderr)
        assert "compress_ratio 1: block_size//ratio = 64  ok" in r.stdout, r.stdout
        assert "compress_ratio 2: block_size//ratio = 32" in r.stdout, r.stdout
        assert "NOT SERVABLE" in r.stdout, r.stdout
        assert "cannot serve" in r.stderr, r.stderr
        assert "11.9.16" in r.stderr, r.stderr


def test_patched_image_accepts_the_whole_checkpoint() -> None:
    """The fix's acceptance test: both ratios pass at --block-size 64.

    This is the case that could not be satisfied before the patch — ratio 1 wants
    a 64-state compressed page and ratio 2 a 32-state one, from the same run, and
    stock FlashInfer has no 32.
    """
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _config(tmp, V41_CONFIG)
        r = _run(
            cfg,
            block_size=SERVED_BLOCK_SIZE,
            stub=_stub(tmp, dispatch=[(32, 128)], cu=patched_cu()),
        )
        assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
        assert "compress_ratio 1: block_size//ratio = 64  ok" in r.stdout, r.stdout
        assert "compress_ratio 2: block_size//ratio = 32  ok" in r.stdout, r.stdout
        assert "accepts compressed page blocks: [2, 32, 64, 128]" in r.stdout, r.stdout
        assert "OK: decode keys present" in r.stdout, r.stdout


def test_widened_acceptance_without_instantiation_is_rejected() -> None:
    """The bare-else trap must be a hard failure, not a pass.

    Stock's full-tile branch is `if (== 64) ... else { PBSX(2) }`, so a patch that
    widened only the acceptance test would admit 32 and run the 2-token
    instantiation on 32-granular slot ids.
    """
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _config(tmp, V41_CONFIG)
        r = _run(
            cfg,
            block_size=SERVED_BLOCK_SIZE,
            stub=_stub(tmp, dispatch=[(32, 128)], cu=trap_cu()),
        )
        assert r.returncode == 5, (r.returncode, r.stdout, r.stderr)
        assert "full-tile branch runs PBSX(2)" in r.stderr, r.stderr
        assert "wrong page block" in r.stderr, r.stderr


def test_partial_patch_missing_the_fulltile_branch_is_rejected() -> None:
    """Edits 1+3 without edit 2 must not pass, though the union looks complete."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _config(tmp, V41_CONFIG)
        r = _run(
            cfg,
            block_size=SERVED_BLOCK_SIZE,
            stub=_stub(tmp, dispatch=[(32, 128)], cu=no_fulltile_branch_cu()),
        )
        assert r.returncode == 5, (r.returncode, r.stdout, r.stderr)
        assert "instantiates (both ladders):    [2, 32, 64, 128]" in r.stdout, r.stdout
        assert "full-tile branch runs PBSX(2)" in r.stderr, r.stderr


def test_partial_patch_missing_the_plain_branch_is_rejected() -> None:
    """Edits 1+2 without edit 3 must not pass, though the union looks complete."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _config(tmp, V41_CONFIG)
        r = _run(
            cfg,
            block_size=SERVED_BLOCK_SIZE,
            stub=_stub(tmp, dispatch=[(32, 128)], cu=no_plain_branch_cu()),
        )
        assert r.returncode == 5, (r.returncode, r.stdout, r.stderr)
        assert "instantiates (both ladders):    [2, 32, 64, 128]" in r.stdout, r.stdout
        assert "non-full-tile ladder returns false" in r.stderr, r.stderr


def test_aot_binary_shadowing_the_patch_is_rejected() -> None:
    """A present AOT binary cannot serve a non-stock page block; fail."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _config(tmp, V41_CONFIG)
        r = _run(
            cfg,
            block_size=SERVED_BLOCK_SIZE,
            stub=_stub(tmp, dispatch=[(32, 128)], cu=patched_cu(), is_aot=True),
        )
        assert r.returncode == 5, (r.returncode, r.stdout, r.stderr)
        assert "AOT binary is" in r.stderr and "still present" in r.stderr, r.stderr
        assert "mask_flashinfer_aot.py" in r.stderr, r.stderr


def test_aot_binary_is_fine_when_no_patch_is_needed() -> None:
    """A ratio-2-only config at block size 128 is served by the stock binary.

    Both its compressed page (128 // 2 = 64) and its `block_kv` are in stock
    FlashInfer and DeepGEMM respectively, so the AOT binary is not a problem and
    must not be reported as one.
    """
    with tempfile.TemporaryDirectory() as tmp:
        only2 = json.loads(json.dumps(V41_CONFIG))
        only2["text_config"]["compress_ratios"] = [0, 2, 2, 0]
        cfg = _config(tmp, only2)
        r = _run(
            cfg,
            block_size=128,
            stub=_stub(tmp, dispatch=[(32, 128)], cu=stock_cu(), is_aot=True),
        )
        assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
        assert "AOT binary in use: True" in r.stdout, r.stdout


def test_unreadable_aot_status_fails_closed_when_a_patch_is_needed() -> None:
    """A failed AOT lookup is not a clean bill of health.

    If the introspection that decides whether a patched `.cu` is compiled at all
    cannot be read, the pre-flight cannot claim the patch is in effect. For a
    config that needs a non-stock page block that must refuse, not pass.
    """
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _config(tmp, V41_CONFIG)
        r = _run(
            cfg,
            block_size=SERVED_BLOCK_SIZE,
            stub=_stub(tmp, dispatch=[(32, 128)], cu=patched_cu(), is_aot=None),
        )
        assert r.returncode == 5, (r.returncode, r.stdout, r.stderr)
        assert "AOT binary in use: unknown" in r.stdout, r.stdout
        assert "could not be determined" in r.stderr, r.stderr


def test_unreadable_aot_status_is_fine_when_no_patch_is_needed() -> None:
    """The same failure is harmless when the stock page blocks already serve it."""
    with tempfile.TemporaryDirectory() as tmp:
        only2 = json.loads(json.dumps(V41_CONFIG))
        only2["text_config"]["compress_ratios"] = [0, 2, 2, 0]
        cfg = _config(tmp, only2)
        r = _run(
            cfg,
            block_size=128,
            stub=_stub(tmp, dispatch=[(32, 128)], cu=stock_cu(), is_aot=None),
        )
        assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
        assert "AOT binary in use: unknown" in r.stdout, r.stdout


def test_missing_decode_key_exits_3() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _config(tmp, V41_CONFIG)
        r = _run(
            cfg,
            block_size=SERVED_BLOCK_SIZE,
            stub=_stub(tmp, dispatch=[], cu=patched_cu()),
        )
        assert r.returncode == 3, (r.returncode, r.stdout, r.stderr)
        assert "MISSING decode specializations" in r.stderr, r.stderr
        assert "(num_heads=32, topk=128)" in r.stderr, r.stderr


def test_allow_missing_downgrades_the_decode_key() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _config(tmp, V41_CONFIG)
        r = _run(
            cfg,
            block_size=SERVED_BLOCK_SIZE,
            stub=_stub(tmp, dispatch=[], cu=patched_cu()),
            extra=["--allow-missing"],
        )
        assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)


def test_flashinfer_unavailable_exits_4() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _config(tmp, V41_CONFIG)
        blocker = Path(tmp) / "block"
        (blocker / "flashinfer").mkdir(parents=True)
        (blocker / "flashinfer" / "__init__.py").write_text("raise ImportError('blocked')\n")
        r = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--config",
                str(cfg),
                "--tp",
                "2",
                "--block-size",
                str(SERVED_BLOCK_SIZE),
            ],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": str(blocker)},
        )
        assert r.returncode == 4, (r.returncode, r.stdout, r.stderr)
        assert "cannot verify" in r.stderr, r.stderr


def test_missing_config_fails_closed() -> None:
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--config", "/nonexistent/config.json"],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 2, r.stdout + r.stderr
    assert "no config at" in r.stderr, r.stderr


def test_derivations_are_documented_in_the_script() -> None:
    src = SCRIPT.read_text()
    # The required key and its source line must be stated, not implied.
    assert "topk = indices.shape[-1]" in src
    assert "sliding_window" in src
    # The image-span widening and the override that undoes it must be stated,
    # including why the layer count is the wrong lever.
    assert "vision_max_n_token" in src
    assert "max_image_tokens" in src
    assert "hf-overrides" in src
    assert "bias_vl" in src
    # The DeepGEMM block_kv constraint and the issue it comes from must be
    # stated, because it is what fixes the manager block size. Both gates are
    # named, because conflating them is the bug this check had.
    assert "block_kv == 32 or block_kv == 64" in src
    assert "53635" in src
    assert "DEEPGEMM_METADATA_BLOCK_KV" in src
    assert "DEEPGEMM_FP8_LOGITS_BLOCK_KV" in src
    assert "sm12x_fallback_installed" in src
    assert "patch_sm12x_indexer.py" in src
    # The page-block constraint and its source must be stated.
    assert "extra_page_block_size" in src
    assert "block_size // compress_ratio" in src
    # The AOT trap must be reported, not silently tolerated.
    assert "is_aot" in src
    assert "mask_flashinfer_aot.py" in src
    # Both patches must be named so an operator can find the fix.
    assert "patch_flashinfer_prefill_pbsx.py" in src
    assert "patch_v41_kernel_block_size.py" in src


def test_declared_vision_config_is_refused_without_the_override() -> None:
    """The real checkpoint without `--max-image-tokens 0` must fail, not pass.

    This is the false green the verifier used to produce: it checked the
    128-wide key the *served* posture needs while never looking at the width the
    engine would actually build.
    """
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _config(tmp, V41_CONFIG)
        r = _run(
            cfg,
            block_size=SERVED_BLOCK_SIZE,
            stub=_stub(tmp, dispatch=[(32, 128)], cu=patched_cu()),
            max_image_tokens=None,
        )
        assert r.returncode == 3, (r.returncode, r.stdout, r.stderr)
        assert f"SWA index width = {VISION_WIDTH}" in r.stdout, r.stdout
        assert "cannot serve the SWA index width" in r.stderr, r.stderr
        assert "dispatch_dsv4_dual" in r.stderr, r.stderr
        assert "hf-overrides" in r.stderr, r.stderr
        # The page-block verdict must not be reached, because the width is the
        # blocker that actually fires first.
        assert "OK: decode keys present" not in r.stdout, r.stdout


def test_text_only_override_is_reported_and_accepted() -> None:
    """The served posture: effective budget 0, width 128, both ratios servable."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _config(tmp, V41_CONFIG)
        r = _run(
            cfg,
            block_size=SERVED_BLOCK_SIZE,
            stub=_stub(tmp, dispatch=[(32, 128)], cu=patched_cu()),
        )
        assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
        assert "effective image-token budget=0 (checkpoint declares 1024)" in r.stdout, (
            r.stdout
        )
        assert "SWA index width = 128" in r.stdout, r.stdout
        assert "single dispatcher topk widths:  [128, 192, 256, 512, 1024, 2048]" in (
            r.stdout
        ), r.stdout
        assert "OK: decode keys present" in r.stdout, r.stdout


def test_a_width_the_single_ladder_carries_is_still_refused() -> None:
    """192 is in the single ladder and in no decode set; it must still be refused.

    The dual path requires exactly 128, so a width that looks dispatchable is
    not: a loose check would accept it.
    """
    with tempfile.TemporaryDirectory() as tmp:
        cfg = json.loads(json.dumps(V41_CONFIG))
        cfg["vision_config"]["max_image_tokens"] = 64
        path = _config(tmp, cfg)
        r = _run(
            path,
            block_size=SERVED_BLOCK_SIZE,
            stub=_stub(tmp, dispatch=[(32, 192)], cu=patched_cu()),
            max_image_tokens=None,
        )
        assert r.returncode == 3, (r.returncode, r.stdout, r.stderr)
        assert "SWA index width = 192" in r.stdout, r.stdout
        assert "dispatch_dsv4_dual" in r.stderr, r.stderr


def _config(tmp: str, config: dict) -> Path:
    cfg = Path(tmp) / "config.json"
    cfg.write_text(json.dumps(config))
    return cfg


def _run(
    cfg: Path,
    *,
    block_size: int,
    stub: str,
    extra: list | None = None,
    max_image_tokens: int | None = TEXT_ONLY,
):
    """Run the pre-flight in the served posture unless a test says otherwise."""
    budget = (
        []
        if max_image_tokens is None
        else ["--max-image-tokens", str(max_image_tokens)]
    )
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--config",
            str(cfg),
            "--tp",
            "2",
            "--block-size",
            str(block_size),
            *budget,
            *(extra or []),
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": stub},
    )


def _stub(
    tmp: str,
    *,
    dispatch: list,
    cu: str,
    is_aot: bool | None = False,
    fallback: str = "full",
) -> str:
    """A minimal stand-in for the installed flashinfer package.

    Carries both the decode dispatch table and the prefill source, because the
    pre-flight reads the page-block acceptance out of the source file. Passing
    `is_aot=None` omits the module generator entirely, which is how an
    introspection failure (a renamed attribute, a changed API) looks from the
    pre-flight's side.

    `fallback` models the three states of the overlay patch in a stub
    `vllm.utils.deep_gemm`, which is exactly what `sm12x_fallback_installed()`
    reads, so these tests exercise the real detection path rather than a
    monkeypatch:

    * `"full"` — both edits, the helper and the call to it. The shipped image,
      where the Dockerfile applies `overlay/patch_sm12x_indexer.py` before this
      suite runs.
    * `"helper"` — the first edit alone. The helper exists but nothing calls it,
      so every 32-state page still reaches DeepGEMM.
    * `"none"` — stock, no overlay.

    The stub `vllm` shadows the real one for the subprocess, which is harmless
    here: the pre-flight imports nothing else from it.
    """
    root = Path(tmp) / "stub"
    fi = root / "flashinfer"
    mla = fi / "mla"
    csrc = fi / "data" / "csrc"
    mla.mkdir(parents=True)
    csrc.mkdir(parents=True)
    (fi / "__init__.py").write_text("")
    (mla / "__init__.py").write_text("")
    generator = (
        ""
        if is_aot is None
        else "def gen_sparse_mla_sm120_module():\n"
        "    return types.SimpleNamespace(is_aot=%r, aot_path='stub')\n"
        % bool(is_aot)
    )
    (mla / "_sparse_mla_sm120.py").write_text(
        "import types\n"
        "_DECODE_DSV4_DISPATCH = frozenset(%r)\n"
        "_DECODE_DSV4_PAGE_BLOCK_SIZE = 64\n"
        "%s" % (set(tuple(x) for x in dispatch), generator)
    )
    (csrc / chk.PREFILL_REL.name).write_text(cu)

    hook = (
        "def _dsv41_sm12x_fp8_paged_mqa_logits(*args, **kwargs):\n"
        "    return None\n"
    )
    stock_wrapper = (
        "def fp8_fp4_paged_mqa_logits(*args, **kwargs):\n"
        "    return None\n"
    )
    wired_wrapper = (
        "def fp8_fp4_paged_mqa_logits(*args, **kwargs):\n"
        "    result = _dsv41_sm12x_fp8_paged_mqa_logits(*args, **kwargs)\n"
        "    if result is not None:\n"
        "        return result\n"
        "    return None\n"
    )
    if fallback == "full":
        body = hook + "\n" + wired_wrapper
    elif fallback == "helper":
        body = hook + "\n" + stock_wrapper
    elif fallback == "none":
        body = stock_wrapper
    else:
        raise ValueError(f"unknown fallback state {fallback!r}")

    vllm = root / "vllm"
    utils = vllm / "utils"
    utils.mkdir(parents=True)
    (vllm / "__init__.py").write_text("")
    (utils / "__init__.py").write_text("")
    (utils / "deep_gemm.py").write_text(body)
    return str(root)


if __name__ == "__main__":
    test_required_key_is_the_window_width()
    test_image_budget_widens_the_required_key_past_every_instantiation()
    test_zeroing_the_image_budget_is_what_collapses_the_width()
    test_swa_width_ignores_vision_without_a_max_image_tokens()
    test_effective_budget_is_what_is_checked()
    test_required_key_is_per_rank_and_padded()
    test_indivisible_heads_is_an_error()
    test_combined_widths_are_candidates_not_requirements()
    test_no_index_topk_yields_no_combined_candidates()
    test_single_ladder_widths_are_read_from_the_source()
    test_prefill_width_defect_accepts_only_the_dual_width()
    test_page_block_parsing_reads_the_source()
    test_compressed_page_blocks_are_block_size_over_ratio()
    test_deepgemm_metadata_and_logits_sets_are_distinct()
    test_indexer_defects_report_which_gate_rejected_the_page()
    test_only_block_size_64_lands_both_ratios_on_the_deepgemm_set()
    test_acceptance_set_ignores_the_branch_headers()
    test_dispatch_defects_are_branch_local()
    test_completeness_defects_catches_every_incomplete_combination()
    test_partial_patch_of_only_the_plain_ladder_is_rejected()
    test_missing_fulltile_branch_is_a_wrong_kernel_defect()
    test_missing_plain_branch_is_a_boot_failure_defect()
    test_stock_block_set_is_the_vendored_one()
    test_block_size_128_is_refused_by_the_deepgemm_check()
    test_ratio_two_page_block_is_refused_when_the_sm12x_fallback_is_absent()
    test_the_helper_without_its_call_site_is_not_a_fallback()
    test_stock_image_refuses_the_ratio_two_page_block()
    test_patched_image_accepts_the_whole_checkpoint()
    test_widened_acceptance_without_instantiation_is_rejected()
    test_partial_patch_missing_the_fulltile_branch_is_rejected()
    test_partial_patch_missing_the_plain_branch_is_rejected()
    test_aot_binary_shadowing_the_patch_is_rejected()
    test_aot_binary_is_fine_when_no_patch_is_needed()
    test_unreadable_aot_status_fails_closed_when_a_patch_is_needed()
    test_unreadable_aot_status_is_fine_when_no_patch_is_needed()
    test_missing_decode_key_exits_3()
    test_allow_missing_downgrades_the_decode_key()
    test_flashinfer_unavailable_exits_4()
    test_missing_config_fails_closed()
    test_derivations_are_documented_in_the_script()
    test_declared_vision_config_is_refused_without_the_override()
    test_text_only_override_is_reported_and_accepted()
    test_a_width_the_single_ladder_carries_is_still_refused()
    print("flashinfer dsv41 pre-check tests OK")
