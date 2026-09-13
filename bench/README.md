# bench/ — the A/B harness and the rules it refuses to break

`scripts/ab_spec.sh` runs one boot per arm; `bench/bench_decode.py` takes the row.

```bash
./scripts/ab_spec.sh list              # the arm table
./scripts/ab_spec.sh run baseline      # the control: SPEC=none
./scripts/ab_spec.sh run greedy5       # stop, re-arm, start, smoke, measure
./scripts/ab_spec.sh report            # every receipt side by side
```

## Why one boot per arm

The speculative config is read at engine init and the weights are re-read anyway,
so there is no cheaper way to switch arms. Each arm is written as an appended
`.env` — not an environment variable, because `start.sh` sources `.env` with
`set -a` *after* the environment exists, so a value in `.env` wins over an
exported one. The operator's `.env` is backed up to `.env.ab-backup` and restored
immediately after the boot.

Every arm boots with `FIRST_BOOT=0`, including the control. `FIRST_BOOT=1`
refuses speculative decode on purpose — the first boot of a new kit is the
conservative shape — and `FIRST_BOOT=0` only *permits* the four first-boot
switches; the argv still honors the rest of `.env`. So the serving shape is
identical across arms and the sole delta is the speculative config.

## The four rules

**1. A tok/s number without its coherence result is not a row.**
This checkpoint's known failure mode is fluent output that is NaN all the way
down with a healthy throughput number attached, and the speculative path adds a
second: a draft that degenerates into loops while *inflating* its acceptance
rate. Every receipt therefore carries `any_nan` and `coherent`, and the bench
exits non-zero when either fails.

**2. Every arm has a same-day control.**
`baseline` is SPEC=none. A speculative number quoted against a baseline taken on
a different day, or under a different `--max-num-seqs` or KV pool, is not
comparable.

**3. Acceptance comes from the engine, not from the token counts.**
The receipt carries an `acceptance` block read as a delta of the engine's own
counters across the measured window, so it describes these runs rather than the
process lifetime. It is what explains a tok/s figure; without it, a speedup is
unattributable.

The three numbers use vLLM's own definitions
(`vllm/v1/spec_decode/metrics.py`), and they are easy to confuse:

| Field | Formula | Meaning |
|---|---|---|
| `accept_ratio` | `accepted / draft_tokens` | fraction of *drafted* tokens accepted |
| `mean_acceptance_length` | `1 + accepted / drafts` | tokens emitted per verify step, bonus token included |
| `per_position` | `accepted_per_pos[i] / drafts` | probability a draft had **at least** `i+1` tokens accepted |

`mean_acceptance_length` is the one that explains throughput, and it is not
`1 + accept_ratio` — for the block-5 arm it is 3.1231 against a 0.4246 acceptance
rate.
The per-position vector is cumulative, so it sums to `accepted / drafts`, not to 1.

The `position` label is part of the counter key. Dropping it collapses all five
per-position series onto one name, which silently discards four of them; the
tests pin this against upstream's published worked example.

**4. One repetition is not a measurement.**
`--runs 3`, and the median is the row. The first run after a boot is slower than
the rest — the KV cache is cold and the page cache is not yet holding the
weights — which is visible in every receipt as a low first `tok_s`.

One caveat this protocol cannot fix: the bench runs a **greedy target at
`temperature: 0`**, where the token stream is byte-identical every time and the
acceptance rate is exactly reproducible. That is good for comparing arms and bad
for comparing *draft sampling methods*, which only diverge at higher
temperatures. If you need that comparison, it is a different bench.

## What the coherence gate does and does not do

It is a **tripwire, not a quality metric**. It fails on empty output, on a NaN
marker in the text, and on a 20-character window repeated eight or more times
(the signature of a collapsed drafter). It deliberately does not fail on ordinary
repetition, because a word-level check would flag legitimate prose.

Passing it means *this output is not obviously degenerate*. It does not mean the
model is correct, and no number in this directory should be read as a quality
claim. For a real quality gate, use a task benchmark — none has been run here.

## Receipts

`results/ab-spec/<arm>.json`, one per arm, each carrying the engine's own
acceptance counters, the per-run rows, and the first 400 characters of each
generation. They are committed because a row that cannot be inspected after the
fact is an anecdote.
