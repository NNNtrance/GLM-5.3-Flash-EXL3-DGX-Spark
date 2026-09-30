# Multi-turn agentic gate — contributed tool, and the run that motivated it

Contributed by **@jdecker76**. Tool: [`scripts/multiturn-gate.py`](../../../../scripts/multiturn-gate.py).
Related: issue #1 §4 (where this was first suggested), [HELP-WANTED](../../../../HELP-WANTED.md) §12.

## Why

Every gate in this repository is **single-turn** — correctness probe, code exam,
tool-call gate, needle-lite. We ran a GLM-5.3-Flash stack in production on real
multi-user agentic coding traffic that passed all of them (10/10, 12/12, 8/8,
5/5 vision, cold and warm) while users hit **looping, premature end-of-turn and
degraded long-session quality** badly enough that we rolled the engine back. The
gates were not lying; the symptom family lives in warm multi-turn sessions,
where no single-turn gate can reach it.

## What it checks

| | Check | Method |
|---|---|---|
| A | **Reasoning retention** | Same 5-turn conversation twice — once echoing assistant turns **verbatim** (what real clients send), once with `<think>` stripped by the harness — comparing prompt-token growth between the arms. |
| B | **Degenerate repetition** | Longest repeated-line run plus a repeated-phrase detector, per response. |
| C | **Early stop** | `finish_reason == "stop"` at implausibly short length on prompts that explicitly ask for substantive output. |

Check A is **differential** on purpose: it measures the rendered result rather
than reading configuration, so it is engine- and template-agnostic and reports
what the server actually did.

## The run that motivated it

Settings: GLM-5.3-Flash, TP=3 + expert parallel on 3× DGX Spark (GB10), NVFP4
(`modelopt_mixed`, the sibling recipe's format — not this repo's EXL3), fp8 KV,
DFlash2 k=7, `--block-size 256`, `gpu-memory-utilization` 0.88, `max-model-len`
1000000, `max-num-seqs` 8, temperature 1.0 / top_p 0.95, `reasoning_effort` low,
5 turns × 2 arms, 9 September 2026. `[measured-here, raw not published]` —
"here" being our cluster, not this repo's.

| Arm | prompt_tokens by turn | growth | ratio |
|---|---|---:|---:|
| launcher defaults carried `clear_thinking:true`, request sent its own `chat_template_kwargs` | 52 · 518 · 1166 · 1549 · 2275 | +2223 | **2.24** |
| template itself defaults `clear_thinking` to true | 52 · 309 · 704 · 937 · 1511 | +1459 | **1.13** |

**The finding worth carrying over:** engine-level `--default-chat-template-kwargs`
are **not** a guard. A per-request `chat_template_kwargs` block **replaces** the
defaults rather than merging, so any client sending `{"reasoning_effort":"low"}`
silently drops `clear_thinking` and retention returns. We had "fixed" this a day
earlier by adding the key to launcher defaults; this gate is what showed us we
had not. That is the case for checking rendered behaviour over configuration.

Both arms: no degenerate repetition, no early stops.

## On HELP-WANTED §12 — what we can and cannot offer

§12 asks for a flake baseline (10 cold + 10 after a ~49,000-token soak, per
item). **We have not run that**, and these two data points are not a baseline:
on the boot that restored our NVFP4 configuration the code exam returned
**11/12 on its first run and 12/12 on the immediate repeat**, and the item that
failed was **`matrix`** — the same item named in §12, on a different stack and a
different quantization. Two sites, two stacks, same item `[measured-here, raw
not published]`. Offered only as corroboration that §12 is pointing at something
real, not as the measurement it asks for.

## Not tested

- Beyond 5 turns; retention shows by turn 3, but a longer horizon is untested `[not tested]`.
- Tool-call turns inside the multi-turn arm — the conversation is plain text `[not tested]`.
- Thresholds (`--retention-ratio 1.35`, `--min-stop-tokens 120`, `--max-repeat 4`) are
  judgement calls from one cluster, **not** measured flake baselines. Given §12 and
  `9496fcc`, anyone using this to adjudicate a change should establish their own
  baseline first.
