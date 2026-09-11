# Tool-call corruption with the structural-tag grammar ON — `strict: true` A/B (11 September 2026)

Follow-up to [`toolcall-gate-11sep.md`](toolcall-gate-11sep.md), answering the second finding in
issue #7: in vLLM the sampling-layer grammar that makes malformed tool-call syntax *unsamplable* is
gated off for `tool_choice="auto"` unless at least one tool carries `strict: true`. No agentic client
sets that field, so in practice the grammar never activates and the tool-call parser is the only line
of defence.

This page measures what happens when the gate is opened — **without touching the engine**.

## Method

A shim, [`scripts/strict-proxy.py`](../../scripts/strict-proxy.py), sits in front of the production
server and sets `tools[i].function.strict = true` on every function tool in
`POST /v1/chat/completions`. Everything else (SSE streaming, `/v1/models`, `/tokenize`, `/metrics`) is
forwarded byte for byte; one JSON line per request is logged with sizes and timings. The control arm
uses the same proxy with `--passthrough` (identical plumbing, no injection). The engine — EXL3
turboderp-4.05bpw, TP=3 + EP, fail-closed tool parser on, DFlash2 draft at k=7, `reasoning_effort: low`,
`clear_thinking: true` — was not restarted, reconfigured or patched for this run.

### The gate is real on this build

Verified read-only inside the running container, on the pinned build
(`0.1.dev20051+g487ecf187`):

| Check | Finding |
|---|---|
| `structural_tag_registry.py:113` | `if tool_choice == "auto" and not _any_tool_strict(tools): return None` |
| `glm_4_7` in `XGRAMMAR_BUILTIN_STRUCTURAL_TAG_MODELS` | yes |
| `VLLM_ENFORCE_STRICT_TOOL_CALLING` | defaults to `True` — and does **not** lift this gate |
| `strict` survives request validation | yes — a declared field on `FunctionDefinition`, not an ignored extra; after injection `_any_tool_strict` is `True` and `get_model_structural_tag("glm_4_7", …, "auto", …)` returns a `StructuralTag` |

### Protocol identity

Both arms use [`scripts/toolcall-gate.py`](../../scripts/toolcall-gate.py) at 4 sessions × 30 turns,
concurrency 2, `--max-prompt-tokens 120000`, effort low, `clear_thinking` on. The baseline arm was run
from a translated copy of the script; `TOOLS` (30), `TASKS` (16), `SYSTEM_PROMPT` and the fixture
repository were hash-compared and are byte-identical — the 157 differing lines are comment and
label translation only.

## Result

| Metric | A — no `strict` (08:45) | B — `strict: true` via proxy (16:24) |
|---|---|---|
| Turns | 120 | 120 |
| Tool calls | 118 | **133** |
| Well-formed (schema-valid) | 117 | **133 — all of them** |
| Fail-closed rejections (malformed XML in `content`) | 0 | **0** |
| Parsed but out-of-schema | 1 (the harness's own 2048-token completion cap truncating a long `write`) | **0** |
| JSON parse errors | 0 | 0 |
| Empty turns / repeats / `finish_reason=length` | 0 / 0 / 0 | 0 / 0 / 0 |
| Highest context reached | 100,522 | 91,294 |
| Turn latency median / mean / p90 / max | 3.33 / 8.59 / 23.7 / 72.2 s | 3.21 / 6.32 / 18.4 / 42.0 s |
| Wall clock | 1,030 s | 375 s |

### B, per-bucket

| context_bucket | turns | tool_calls | wellformed | rejected(b) | schema_violation(c) | json_error | empty_turn(d) | repeat(e) | length(f) |
|---|---|---|---|---|---|---|---|---|---|
| 0–10k | 33 | 38 | 38 | 0 | 0 | 0 | 0 | 0 | 0 |
| 10–20k | 16 | 19 | 19 | 0 | 0 | 0 | 0 | 0 | 0 |
| 20–30k | 31 | 36 | 36 | 0 | 0 | 0 | 0 | 0 | 0 |
| 30–40k | 25 | 26 | 26 | 0 | 0 | 0 | 0 | 0 | 0 |
| 40–50k | 3 | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| 50–60k | 10 | 10 | 10 | 0 | 0 | 0 | 0 | 0 | 0 |
| 90–100k | 2 | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| TOTAL | 120 | 133 | 133 | 0 | 0 | 0 | 0 | 0 | 0 |

### The two arms did not walk the same path

This matters for reading the latency rows. The script's context ramp pastes fixture files **only on
turns where the model makes no tool call**. With the grammar on the model called a tool more often
(1.11 calls/turn vs 0.98) and went quiet less often, so the ramp fired less, and arm B topped out at
91k instead of 100.5k with a different bucket distribution (25 turns in 30–40k vs 3). **B is therefore
not a like-for-like depth or timing comparison of A**, and most of the wall-clock difference is that,
not the grammar. A ramp driven by a token target rather than by quiet turns would fix this.

## Grammar compilation, priced two independent ways

| Instrument | 5 tools | 15 tools | 30 tools | 55 tools |
|---|---|---|---|---|
| Through the proxy: first request − cache-warm request | 0.05 s | 0.06 s | **0.18 s** | — |
| In-container, model-free `GrammarCompiler.compile_structural_tag` | — | — | **0.171 s** | 0.070 s\* |
| Same tag compiled a second time (cache) | — | — | 0.000 s | 0.001 s |

\* the 55-tool set is 30 distinct schemas plus 25 renamed duplicates, so shared sub-grammars make it
cheaper than the 30-tool set — it is a lower bound for 55 *distinct* tools.
One-time `TokenizerInfo` + `GrammarCompiler` warm-up: **2.56 s**, per process, independent of toolset.

The gate run agrees: the very first request of the whole run cost 12.40 s with strict vs 10.98 s
without (+1.4 s), and the first requests of sessions 2–4 show no penalty at all (4.34 vs 4.86,
3.04 vs 3.22, 2.45 vs 2.39 s).

**This does not reproduce the ~27 s first-request compile reported in issue #7 for 55 tools** — on this
stack per-toolset compilation is a fifth of a second. The likeliest explanation (inference, not
measured on the other cluster) is that the 27 s was the one-time per-process xgrammar/tokenizer
warm-up; the server measured here had been serving grammar-constrained traffic for hours already,
because another client on the same engine uses structured output with a strict JSON schema — which
means the grammar path had in fact been live in production all along, just never for tool calls.

## Streaming and decode under the grammar

| Arm | total | TTFT | decode | finish_reason | tool_call deltas | arguments |
|---|---|---|---|---|---|---|
| essay, no tools | 6.80 s | 2.28 s | 38.7 tok/s | stop | — | — |
| essay, tools, no strict | 7.14 s | 3.01 s | 41.7 tok/s | stop | — | — |
| essay, tools + **strict** | 8.56 s | 3.12 s | 38.9 tok/s | stop | — | — |
| tool call, no strict | 2.12 s | 2.12 s | — | tool_calls | 1 | valid JSON |
| tool call, **strict** | 2.23 s | 2.23 s | — | tool_calls | 1 | valid JSON |

No measurable decode cost from a loaded grammar (38.7 / 41.7 / 38.9 tok/s is inside this cluster's
noise band, and other traffic shared the engine during these probes). Streaming stayed clean under
grammar + speculative decoding: no swallowed deltas, arguments parse as JSON, correct `finish_reason`.
For the record, the proxy flipped **the flag only** — schemas were not rewritten to OpenAI strict style
(a `--schema-strict` mode exists in the proxy and was not needed).

## Engine-side noise — open question

Across a 10-minute window that covers both strict runs, with no other client on the engine:

| Log line | Count |
|---|---|
| `backend_xgrammar.py:166 … Failed to advance FSM … Please file an issue.` (**ERROR**) | **200**, across 26 distinct requests (max 16 per request) |
| `grammar_matcher.cc:612 Warning: The matcher has terminated after accepting the stop token` | 245 |
| Any other `ERROR` | 0 |

All 240 turns of the two strict runs completed correctly and the corruption counters are zero, so these
lines are **not fatal**. But issue #7 reported only the benign `grammar_matcher` warning; this build
additionally logs an **ERROR-level FSM advance failure**, and the shape (several token ids per burst)
points at draft tokens from speculative decoding proposing past the stop token. This build predates the
xgrammar backports #52805 / #53046, which is exactly this territory. Hundreds of ERROR lines an hour is
a real operational cost even when nothing breaks.

`/metrics` before and after: `num_requests_running` 0 at both ends; prefix-cache queries
1.680→1.722 × 10⁸ and hits 1.540→1.576 × 10⁸ (≈4.1M queries / 3.6M hits attributable to the run, 88%);
KV usage back to 0.

## Reading

Opening the gate is **safe and cheap on this build, and its measured benefit is small** — because the
baseline was already clean. Arm B turned 133 tool calls into 133 schema-valid tool calls and removed
the one deviation arm A showed, at a cost of 0.17 s of one-off grammar compilation per toolset, no
decode penalty and no streaming damage. The honest summary is that the grammar is a cheap second
belt, not a rescue: on the fail-closed production build there was nothing left to rescue in 240 turns.
The thing that would change this verdict is real agentic traffic at high effort, which this fixture
is not.

Two routes to turn it on, both client-side, neither needing an engine change: set `strict: true` in the
client that builds the `tools` list, or keep a shim like this proxy in front. An engine-side env that
treats `tool_choice="auto"` as strict would buy the same thing at the price of a patch, and is not
recommended on that basis. What should be settled before any of it becomes default: the ERROR-level
FSM interaction with speculative decoding above.

## Limits, stated plainly

One run per arm, one reasoning-effort setting (low), a synthetic fixture repository; this counts
corruption **events**, it does not establish a **rate**. The two arms did not reach the same context
depth (91k vs 100.5k, different bucket profile), so the latency rows are not a controlled comparison,
and the arms ran at different times of day. `--schema-strict` was never exercised. The FSM-error bursts
were shown harmless across 240 turns only — not under soak, not at high effort. The 27 s-warm-up
explanation is inference about another cluster, not a measurement of it.

Raw output (`turns.jsonl`, `summary.json`, the proxy's request log and the compile ladder) is not
committed; this page is the summary. Re-running `scripts/strict-proxy.py` plus
`scripts/toolcall-gate.py --api http://127.0.0.1:8011` reproduces it.
