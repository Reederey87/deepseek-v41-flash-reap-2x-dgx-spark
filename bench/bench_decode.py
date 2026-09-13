#!/usr/bin/env python3
"""Streaming decode bench + coherence gate for DeepSeek-V4.1-Flash REAP-256E.

One boot, one arm. Drives the live OpenAI endpoint and reports the engine's own
numbers, because on this checkpoint a throughput figure on its own is not
evidence: the known failure mode is fluent output that is NaN all the way down
with a healthy tok/s attached, and the speculative path adds a second one, a
draft that degenerates into loops while *inflating* its acceptance rate. So every
row carries `any_nan` and `coherent`, and the harness treats a row without them
as unfinished.

Decode tok/s = (completion_tokens - 1) / (last_chunk_time - first_token_time),
taken from the server's usage block rather than SSE chunk counts.

  python3 bench/bench_decode.py --phase baseline --out results/ab-spec/baseline.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = os.environ.get("DSV41_BASE", "http://127.0.0.1:8001").rstrip("/")
MODEL = os.environ.get("DSV41_MODEL", "deepseek-v4.1-flash-reap256")

# Long enough that a draft that degenerates has room to show it, and prose rather
# than markup because markup is where acceptance is highest and a broken drafter
# looks best.
PROMPT = (
    "Write a detailed step-by-step explanation of how a hash map works, "
    "including collision handling, resizing, and time complexity. Be thorough."
)

# The spec-decode counters the engine exposes, so acceptance is read from the
# server rather than inferred from token counts.
SPEC_RE = re.compile(r"^(vllm:spec_decode_[a-zA-Z0-9_]+)\{?([^}]*)\}?\s+(\S+)$")
NAN_RE = re.compile(r"\bnan\b|locklock", re.I)


def _req(path: str, body: dict | None = None, timeout: float = 900.0):
    url = BASE + path
    if body is None:
        return urllib.request.urlopen(url, timeout=timeout)
    return urllib.request.urlopen(
        urllib.request.Request(
            url,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        ),
        timeout=timeout,
    )


def health() -> tuple[int, str]:
    try:
        with _req("/health", timeout=20) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except Exception as e:  # noqa: BLE001 - reported, not swallowed
        return 0, str(e)


def spec_counters() -> dict:
    """Acceptance-length counters, so a tok/s row can be explained.

    The `position` label is part of the key. Without it, the per-position series
    all collapse onto one name and the last one parsed wins — which silently
    threw away four of five series and made the per-position diagnostic, the
    thing that explains *why* an arm is fast, unavailable. Other labels are
    ignored; they do not vary within a single server.
    """
    out: dict[str, float] = {}
    try:
        with _req("/metrics", timeout=30) as r:
            text = r.read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 - metrics are optional
        return out
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        m = SPEC_RE.match(line.strip())
        if m:
            name, labels, value = m.group(1), m.group(2) or "", m.group(3)
            pos = re.search(r'position="(\d+)"', labels)
            key = f"{name}{{position={pos.group(1)}}}" if pos else name
            try:
                out[key] = float(value)
            except ValueError:
                continue
    return out


def acceptance_delta(before: dict, after: dict) -> dict:
    """Acceptance over the measured window, in the engine's own definitions.

    The formulas are vLLM's (`vllm/v1/spec_decode/metrics.py`): the acceptance
    rate is `accepted / draft_tokens`, the mean acceptance length is
    `1 + accepted / drafts` (the +1 is the always-accepted bonus token), and the
    per-position vector is `accepted_per_pos / drafts`, which is the probability
    that a draft had *at least* that many tokens accepted, not exactly that many.

    A delta rather than the process lifetime, so the numbers describe these runs.
    """
    def d(key: str) -> float:
        return after.get(key, 0.0) - before.get(key, 0.0)

    n_drafts, n_draft_tokens, n_accepted = (
        d("vllm:spec_decode_num_drafts_total"),
        d("vllm:spec_decode_num_draft_tokens_total"),
        d("vllm:spec_decode_num_accepted_tokens_total"),
    )
    if n_drafts <= 0 or n_draft_tokens <= 0:
        return {"accept_ratio": None, "mean_acceptance_length": None,
                "per_position": None, "drafts": n_drafts,
                "draft_tokens": n_draft_tokens, "accepted_tokens": n_accepted}
    per_pos = []
    for i in range(64):  # bounded; the engine reports num_speculative_tokens
        key = f"vllm:spec_decode_num_accepted_tokens_per_pos_total{{position={i}}}"
        if key not in after:
            break
        per_pos.append(round(d(key) / n_drafts, 4))
    return {
        "accept_ratio": round(n_accepted / n_draft_tokens, 4),
        "mean_acceptance_length": round(1 + n_accepted / n_drafts, 4),
        "per_position": per_pos or None,
        "drafts": n_drafts,
        "draft_tokens": n_draft_tokens,
        "accepted_tokens": n_accepted,
    }


def stream_chat(prompt: str, max_tokens: int, timeout: float = 900.0) -> dict:
    """One greedy, thinking-off streaming completion."""
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "logprobs": True,
        "top_logprobs": 1,
        "chat_template_kwargs": {"thinking": False},
    }
    started = time.perf_counter()
    first: float | None = None
    last = started
    chunks: list[str] = []
    usage: dict | None = None
    finish: str | None = None
    logprob_values: list[float] = []
    done_seen = False
    stream_error: str | None = None

    with _req("/v1/chat/completions", body, timeout=timeout) as resp:
        status = resp.status
        buf = b""
        # `read()` on a chunked response blocks until the requested byte count or
        # EOF, so it would timestamp whole buffer fills rather than token
        # arrivals: with 4 tokens arriving 0.2 s apart it reports ~50,000 tok/s.
        # `read1()` returns as soon as any data is available, which is what makes
        # the timestamps mean what they claim. The fallback keeps a Python
        # without `read1` working, at the cost of the coarser timing.
        reader = getattr(resp, "read1", None) or resp.read
        while True:
            piece = reader(4096)
            if not piece:
                break
            buf += piece
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    done_seen = True
                    continue
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if obj.get("error"):
                    stream_error = json.dumps(obj["error"])[:300]
                if obj.get("usage"):
                    usage = obj["usage"]
                for ch in obj.get("choices") or []:
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
                    delta = ch.get("delta") or {}
                    content = delta.get("content")
                    if content:
                        if first is None:
                            first = time.perf_counter()
                        last = time.perf_counter()
                        chunks.append(content)
                    for lp in (ch.get("logprobs") or {}).get("content") or []:
                        v = lp.get("logprob")
                        if isinstance(v, (int, float)):
                            logprob_values.append(float(v))

    text = "".join(chunks)
    completion = (usage or {}).get("completion_tokens")
    elapsed = (last - first) if first is not None else 0.0
    tok_s = ((completion - 1) / elapsed) if (first is not None and elapsed > 0 and completion and completion > 1) else None
    nonfinite = [v for v in logprob_values if v != v or v in (float("inf"), float("-inf"))]
    # A row is only usable if the stream actually completed and carried the
    # evidence the row claims. A coherent-looking prefix followed by an SSE error,
    # or a stream that ends without usage or logprobs, must not be silently
    # averaged in as a fast run.
    problems: list[str] = []
    if status != 200:
        problems.append(f"http {status}")
    if stream_error:
        problems.append(f"sse error: {stream_error}")
    if not done_seen:
        problems.append("stream ended without [DONE]")
    if finish is None:
        problems.append("no finish_reason")
    if not usage:
        problems.append("no usage block")
    elif completion is None:
        problems.append("usage without completion_tokens")
    if not logprob_values:
        problems.append("no logprobs in the stream")
    if nonfinite:
        problems.append(f"{len(nonfinite)} non-finite logprobs")
    if tok_s is None:
        problems.append("no usable timing")
    return {
        "http": status,
        "ttft_s": (first - started) if first is not None else None,
        "completion_tokens": completion,
        "finish_reason": finish,
        "tok_s": tok_s,
        "logprob_count": len(logprob_values),
        "nonfinite_logprobs": len(nonfinite),
        "text_chars": len(text),
        "text": text,
        "elapsed_s": elapsed,
        "done": done_seen,
        "problems": problems,
        "valid": not problems,
    }


def coherence(text: str) -> tuple[bool, str]:
    """Cheap structural checks. Not a quality metric; a degeneration tripwire."""
    if not text.strip():
        return False, "empty"
    if NAN_RE.search(text):
        return False, "nan-or-locklock-marker"
    # A drafter that has collapsed repeats itself far more than prose does. Count
    # the longest run of a repeated 20-char window rather than trusting a
    # dictionary word, which a legitimate repetition would also match.
    window, seen, worst = 20, {}, 0
    for i in range(0, max(0, len(text) - window), 5):
        key = text[i : i + window]
        seen[key] = seen.get(key, 0) + 1
        worst = max(worst, seen[key])
    if worst >= 8:
        return False, f"repeated-window x{worst}"
    return True, "ok"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", required=True, help="arm name, recorded in the receipt")
    p.add_argument("--out", required=True)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--prompt", default=PROMPT)
    p.add_argument("--skip-coherence", action="store_true")
    args = p.parse_args()

    code, body = health()
    print(f"[bench] health {code}")
    if code != 200:
        print(f"[bench] server not ready: {body[:200]}", file=sys.stderr)
        return 2

    before = spec_counters()
    runs = []
    for i in range(args.runs):
        print(f"[bench] run {i + 1}/{args.runs} phase={args.phase}", flush=True)
        try:
            r = stream_chat(args.prompt, args.max_tokens)
        except Exception as e:  # noqa: BLE001 - the row must record the failure
            print(f"[bench] run {i + 1} failed: {e}", file=sys.stderr)
            runs.append({"error": str(e)})
            continue
        ok, why = coherence(r["text"]) if not args.skip_coherence else (True, "skipped")
        r["coherent"] = ok
        r["coherence_note"] = why
        r["nan"] = r["nonfinite_logprobs"] > 0 or not ok
        print(json.dumps({k: r[k] for k in
                          ("tok_s", "ttft_s", "completion_tokens", "finish_reason",
                           "coherent", "nan", "coherence_note")}))
        runs.append(r)

    after = spec_counters()
    # Only rows that completed and carried their evidence count toward the
    # median. An invalid row is kept in the receipt so the failure is visible,
    # but it must not become the number the arm is judged on.
    good = [r for r in runs if r.get("valid")]
    invalid = [r for r in runs if not r.get("valid")]
    toks = [r["tok_s"] for r in good]
    ttfts = [r["ttft_s"] for r in good if r.get("ttft_s") is not None]

    # Acceptance over the measured window, so it describes these runs rather
    # than the process lifetime. `accept_ratio_median` is kept as a flat alias
    # because the A/B report and earlier receipts read it by that name.
    acceptance = acceptance_delta(before, after)

    receipt = {
        "schema_version": 1,
        "phase": args.phase,
        "base": BASE,
        "model": MODEL,
        "args": {"runs": args.runs, "max_tokens": args.max_tokens},
        "health_code": code,
        "health_code_after": health()[0],
        "tok_s_median": statistics.median(toks) if toks else None,
        "tok_s_min": min(toks) if toks else None,
        "tok_s_max": max(toks) if toks else None,
        "ttft_median_s": statistics.median(ttfts) if ttfts else None,
        "completion_tokens_median": statistics.median(
            [r["completion_tokens"] for r in good]
        )
        if good
        else None,
        "valid_runs": len(good),
        "invalid_runs": len(invalid),
        "any_nan": any(r.get("nan") for r in runs),
        "coherent": all(r.get("coherent") for r in runs) if runs else False,
        "accept_ratio_median": acceptance["accept_ratio"],
        "mean_acceptance_length": acceptance["mean_acceptance_length"],
        "acceptance": acceptance,
        "spec_metrics_before": before,
        "spec_metrics_after": after,
        "runs": [
            {k: v for k, v in r.items() if k != "text"} | {"text_head": (r.get("text") or "")[:400]}
            for r in runs
        ],
        "ts": time.time(),
    }
    # No valid row means no measurement, whatever the text looked like.
    receipt["ok"] = bool(good) and receipt["coherent"] and not receipt["any_nan"]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        f"[bench] median {receipt['tok_s_median']} tok/s | ttft "
        f"{receipt['ttft_median_s']} s | coherent {receipt['coherent']} | "
        f"nan {receipt['any_nan']} | accept {receipt['accept_ratio_median']} | "
        f"accept-len {receipt['mean_acceptance_length']} | "
        f"valid {receipt['valid_runs']}/{args.runs}"
    )
    if invalid:
        for r in invalid:
            print(f"[bench] INVALID row: {'; '.join(r.get('problems') or ['unknown'])}",
                  file=sys.stderr)
    print(f"[bench] wrote {args.out}")
    return 0 if receipt["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
