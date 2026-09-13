#!/usr/bin/env python3
"""Thinking-off arithmetic smoke, plus a NaN guard on the logprobs.

Two independent checks, because they fail differently:

1. `17*19` must return `323`. Thinking must be off; with it on and a small
   `max_tokens` the `content` field comes back empty, so empty content is a fail.

2. Every returned logprob must be finite. This is the tell for the failure mode
   the checkpoint's own recipe records: with CUDA graphs enabled the engine
   loads, serves, and returns fluent-looking bytes that are **NaN all the way
   down**, so the throughput number is real and the output is garbage. The
   give-away is `logprobs` failing with "Out of range float values are not JSON
   compliant: nan". A text-only smoke passes straight through that, which is why
   this one asks for logprobs.

Python's `json` accepts bare `NaN`/`Infinity` literals, so the guard has to test
the values rather than rely on the parse failing.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.request


def _finite_logprobs(payload: dict) -> tuple[int, list[str]]:
    """(count, offenders) over every logprob in the response."""
    seen = 0
    bad: list[str] = []
    for choice in payload.get("choices") or []:
        lp = (choice.get("logprobs") or {}).get("content") or []
        for entry in lp:
            for candidate in [entry] + list(entry.get("top_logprobs") or []):
                value = candidate.get("logprob")
                if value is None:
                    continue
                seen += 1
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    bad.append(f"{candidate.get('token')!r}={value!r}")
    return seen, bad


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--model", default="deepseek-v4.1-flash-reap256")
    p.add_argument(
        "--skip-logprobs",
        action="store_true",
        help="do not ask for logprobs; the NaN guard is then not exercised",
    )
    args = p.parse_args()

    request: dict = {
        "model": args.model,
        "messages": [
            {"role": "user", "content": "What is 17*19? Reply with the number only."}
        ],
        "max_tokens": 32,
        "temperature": 0.0,
        "chat_template_kwargs": {"thinking": False},
    }
    if not args.skip_logprobs:
        request["logprobs"] = True
        request["top_logprobs"] = 1

    body = json.dumps(request).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{args.port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        raw = resp.read().decode()
    payload = json.loads(raw)
    content = payload["choices"][0]["message"].get("content") or ""
    print(content)

    failed = False
    if "323" not in content:
        print(
            "FAIL: expected 323 in content (thinking-off required; empty content "
            "means thinking was on with too small a max_tokens)",
            file=sys.stderr,
        )
        failed = True

    if not args.skip_logprobs:
        seen, bad = _finite_logprobs(payload)
        if bad:
            print(
                "FAIL: non-finite logprobs — the engine is producing NaN. This is "
                "the CUDA-graph failure mode: output looks fluent and throughput "
                "is real. Check that --enforce-eager is in effect.",
                file=sys.stderr,
            )
            for offender in bad[:8]:
                print(f"  {offender}", file=sys.stderr)
            failed = True
        elif seen == 0:
            print(
                "FAIL: logprobs were requested but none came back, so the NaN "
                "guard did not actually run.",
                file=sys.stderr,
            )
            failed = True
        else:
            print(f"logprobs finite over {seen} values")

    if failed:
        return 1
    print("smoke OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
