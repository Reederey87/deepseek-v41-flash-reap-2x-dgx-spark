#!/usr/bin/env python3
"""Checks for the smoke script's NaN guard (`scripts/smoke_chat.py`).

The checkpoint's recipe records a failure mode where the engine serves
fluent-looking output that is NaN all the way down, with a real throughput
number attached. A text-only smoke cannot see it; asking for logprobs can. These
tests pin the guard, including the case Python's `json` hides: it parses bare
`NaN`/`Infinity` literals happily, so the values must be tested, not the parse.

`scripts/` is not copied into the image, so skip cleanly there rather than
reporting a failure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "smoke_chat.py"

if not SCRIPT.is_file():
    print("scripts/smoke_chat.py not present (container layout); smoke guard tests skipped")
    raise SystemExit(0)

sys.path.insert(0, str(ROOT / "scripts"))

import smoke_chat  # noqa: E402


def _payload(content_logprobs: list) -> dict:
    return {"choices": [{"logprobs": {"content": content_logprobs}}]}


def test_finite_logprobs_pass() -> None:
    seen, bad = smoke_chat._finite_logprobs(
        _payload(
            [
                {
                    "token": "3",
                    "logprob": -0.01,
                    "top_logprobs": [{"token": "3", "logprob": -0.01}],
                },
                {
                    "token": "2",
                    "logprob": -0.02,
                    "top_logprobs": [{"token": "2", "logprob": -0.02}],
                },
            ]
        )
    )
    assert seen == 4, seen
    assert bad == [], bad


def test_nan_is_caught_even_though_json_parses_it() -> None:
    """The whole point: `json.loads` accepts bare NaN, so values must be tested."""
    raw = (
        '{"choices":[{"logprobs":{"content":[{"token":"3","logprob":NaN,'
        '"top_logprobs":[{"token":"3","logprob":NaN}]}]}}]}'
    )
    payload = json.loads(raw)
    seen, bad = smoke_chat._finite_logprobs(payload)
    assert seen == 2, seen
    assert bad, "a NaN logprob must be reported"
    assert "nan" in bad[0].lower(), bad


def test_infinity_is_caught() -> None:
    payload = json.loads(
        '{"choices":[{"logprobs":{"content":[{"token":"x","logprob":-Infinity}]}}]}'
    )
    seen, bad = smoke_chat._finite_logprobs(payload)
    assert seen == 1, seen
    assert bad, bad


def test_non_numeric_is_caught() -> None:
    seen, bad = smoke_chat._finite_logprobs(
        _payload([{"token": "x", "logprob": "not-a-number"}])
    )
    assert seen == 1, seen
    assert bad, bad


def test_absent_logprobs_count_zero() -> None:
    """No logprobs at all must not read as 'clean' — main() fails on seen == 0."""
    seen, bad = smoke_chat._finite_logprobs(_payload([]))
    assert seen == 0, seen
    assert bad == [], bad
    # A response with no logprobs field at all behaves the same way.
    seen, bad = smoke_chat._finite_logprobs({"choices": [{"message": {}}]})
    assert seen == 0, seen
    assert bad == [], bad


def test_none_logprob_is_skipped() -> None:
    """A null logprob is not a NaN and must not be counted as one."""
    seen, bad = smoke_chat._finite_logprobs(
        _payload([{"token": "x", "logprob": None}])
    )
    assert seen == 0, seen
    assert bad == [], bad


def test_guard_is_wired_into_the_request_and_the_verdict() -> None:
    """The guard only helps if main() asks for logprobs and checks the count."""
    src = SCRIPT.read_text()
    assert '"logprobs"] = True' in src
    assert '"top_logprobs"] = 1' in src
    assert "math.isfinite" in src
    assert "did not actually run" in src
    assert "--skip-logprobs" in src
    # The eager requirement must be named where the failure is reported.
    assert "enforce-eager" in src


if __name__ == "__main__":
    test_finite_logprobs_pass()
    test_nan_is_caught_even_though_json_parses_it()
    test_infinity_is_caught()
    test_non_numeric_is_caught()
    test_absent_logprobs_count_zero()
    test_none_logprob_is_skipped()
    test_guard_is_wired_into_the_request_and_the_verdict()
    print("smoke chat guard tests OK")
