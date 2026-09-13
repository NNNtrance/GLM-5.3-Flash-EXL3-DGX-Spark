# Concurrency at agent workloads: one reader stalls every writer, and the chunk size is not the fix (13 September 2026)

**Applies to: TP=3.** Nothing here was run on the TP=2 track `[not tested]`.

This page answers three questions an agent harness runs into on this stack: what the engine gives a
single stream against eight, why prose is a third of code, and what happens to the streams that are
*writing* while another stream's long prompt is being *read*. The last one turned out to be the
number that explains "5 tok/s per agent" in a live multi-agent run, and the obvious engine knob for
it — `--max-num-batched-tokens` — was measured in both directions and left where it was.

**Settings.** Three DGX Spark (GB10, sm\_121) nodes over the ConnectX-7 mesh, image
`exl3-zeus:754421f` (vLLM `0.1.dev20051+g487ecf187`), checkpoint `turboderp/GLM-5.3-Flash-exl3` branch
`4.05bpw` (full scope), **TP=3 + expert parallelism**, DFlash2 draft at k=7 with an fp8 draft cache,
KV dtype fp8, `gpu-memory-utilization` **0.88**, `max-model-len` **1,000,000**, `--block-size 256`,
`--max-num-batched-tokens 2048` unless the row says otherwise, `--max-num-seqs 8` (5 from §5 on),
`enforce_eager`, vision tower on, `index_topk` 8192, `clear_thinking: true`, `reasoning_effort: low`.
Measured 12–13 September 2026 with nothing else on the cluster. Every prompt is **fresh** (distinct
real-project file bodies per stream, so no two streams share a prefix); numbers are single runs unless
a spread is given, so read differences under ~10 % as noise.

---

## 1. Code against prose, C1 → C8 (short prompts, `bench-sweep.py`, 256 output tokens)

| C | code, aggregate / per stream | prose, aggregate / per stream | draft acceptance code / prose |
|---|---|---|---|
| 1 | 66.0 / 69.7 | 28.0 / 29.0 | 60 % / 15 % |
| 2 | 94.9 / 52.3 | 40.7 / 21.3 | 60 % / 16 % |
| 4 | 140.9 / 39.8 | 55.4 / 14.4 | 66 % / 15 % |
| 6 | 160.0 / 30.4 | 71.0 / 12.5 | 61 % / 15 % |
| 8 | **186.8** / 27.8 | **82.7** / 10.9 | 62 % / 15 % |

`[measured-here]`, 13 September 05:42, 12 prompts per set (`hizset-v2` for code; twelve
design-document prompts of the same length for prose), one round each. The prose column is the one
an agent harness lives in — design documents, reports, reviews — and it is **44 % of code** at every
concurrency, because the DFlash2 draft accepts 15 % of its proposals on English prose against ~60 %
on code ([docs/10](../../docs/10-results-and-roofline.md) §1.2). Aggregate throughput keeps rising to
C8; per-stream decode falls to about 40 % of C1 at C8 in both columns.

## 2. The same at long context, 300 output tokens (`eszaman-olc.py`, 12 September)

| C × prompt | decode per stream, production | DFlash2 off | TTFT mean / max (production) |
|---|---|---|---|
| 1 × 4k | 33.3 | 32.4 | 3.3 s |
| 1 × 24k | 30.7 | 32.3 | 13.0 s |
| 4 × 24k | 9.9 | 10.8 | 38.5 / 62.4 s |
| 6 × 24k | 7.2 | 7.0 | 51.5 / 86.9 s |
| 4 × 64k | 7.0 | 8.6 | 101.5 / 166.2 s |
| 6 × 64k | **4.9** | — | 151.1 / 256.8 s |

`[measured-here]`, 12 September 21:00–21:30, two arms in one session, engine restarted between them
with gates. Three readings: speculative decoding neither helps nor hurts at concurrency (C1 gains
~5 %); per-stream decode at 4 streams is a third of C1 and at 6 streams a quarter; and the TTFT
column is the important one — **prefill is serialized.** Four 24k prompts arriving together take 62 s
until the last first token, which is 96k tokens at ~1.5k tok/s: the reading rate of the engine is a
fixed budget and N readers share it, each waiting for the ones ahead. Reading does not scale with
concurrency the way decode does.

## 3. One reader stalls every writer

`scripts/prefill-interference.py` (§7): three prose streams writing (short prompt, 1,500 output
tokens), and at t = 20 s one 82,384-token prompt with `max_tokens 2` is fired at the same engine.
Tokens delivered per stream per 5-second window:

| phase | stream 1 | stream 2 | stream 3 |
|---|---|---|---|
| before the read (0–20 s) | 7.8–8.6 | 7.4–8.6 | 7.8–8.6 |
| **while the read runs (20–80 s)** | **0.8** | **0.8** | **0.8** |
| after (80–125 s) | 7.0–8.6 | 7.6–8.6 | 7.0–8.6 |

`[measured-here]`, 13 September 07:25 and repeated at 08:19 (same numbers). The read took 60 s
(**1,364 tok/s** prefill with three decoders alive) and for that minute every writer ran at **a tenth**
of its speed, then recovered instantly. The window counts SSE chunks, so absolute values are chunks
rather than tokens (a chunk carries the accepted draft tokens of one step); the ratio is what matters.

Mechanism: chunked prefill. Each scheduler step carries up to `max-num-batched-tokens` (2,048) prompt
tokens plus one verify batch per decoding sequence; a 2,048-token chunk step takes ~1.5 s at this
context, and a decoding sequence advances by one step per step. In a multi-agent harness every tool
result of every agent is a read, so "someone is reading" is the normal state, and this dip — not the
C8 aggregate — is the per-agent speed an agent sees. A live 12 September run (591 tool calls, 4–6
workers) read 700–1,600 tok/s of prompt throughput in most minutes and 4–5 tok/s per worker.

## 4. `--max-num-batched-tokens` in both directions: 2,048 stays

The chunk size is the only scheduler knob that changes how a step is shared between the reader and the
writers, so it was measured — 1,024 and 512 on 13 September (`chunk-ab.sh`: three-node restart per arm,
health, probe, the §3 instrument, then a C1 / C4 × 24k baseline; production restored and the 2,048 row
re-measured in the same session), 8,192 on 30 August.

| chunk | writers during the read (per stream) | 82k read | prefill rate | C1 × 24k TTFT | C4 × 24k TTFT mean / decode | KV pool |
|---|---|---|---|---|---|---|
| **2,048 (production)** | 0.8 | 60.4 s | 1,364 tok/s | 14.2 s | 23.1 s / 10.5 | 7,060,606 |
| 1,024 | 1.6 (2×) | 72.0 s (+19 %) | 1,145 | 16.8 s (+18 %) | 34.5 s (+49 %) / 10.9 | 7,166,197 (+1.5 %) |
| 512 | 2.7 (3.4×) | 120.5 s (2×) | 683 | 26.0 s (+83 %) | 64.2 s (2.8×) / 13.2 | 7,304,843 (+3.5 %) |
| 8,192 (30 Aug) | — | — | no change within the spread | — | C8 TTFT 3.70 → 4.44 s | **−28 %** (activation workspace) |

`[measured-here]`. The per-step fixed cost during prefill is ~0.45–0.5 s at this context (40 steps in
60 s, 80 in 72 s, 161 in 120 s), an order of magnitude above a bare decode step's ~30 ms: the sparse
indexer's per-chunk work, the TP collectives, the expert all-to-all and the draft verification are all
paid once per step whatever the chunk holds. Halving the chunk doubles the writers' rate during a read
and adds that fixed cost to every one of twice as many steps; on a workload where the engine is reading
most of the time, total throughput falls. Going up costs KV pool (the workspace scales with the chunk)
and halves the writers again. **2,048 is the measured optimum in both directions**, and the lever for
"writers stall while someone reads" is therefore not in the scheduler: it is reading fewer tokens.

## 5. `--max-num-seqs` 8 → 5: no effect on its own

Lowered on 13 September 08:55 (4 harness workers plus one interactive slot), three-node restart, gates
re-probed: KV pool 7,060,606 → **7,074,380** (+0.2 %: the pool is sized by memory, not by the sequence
cap), C1 × 24k 23.8 → 23.2 tok/s, C4 × 24k 10.5 → 10.1 — noise. The cap buys nothing by itself; only
running fewer streams does. Left at 5 because the harness runs 4.

## 6. What this means for an agent harness on this stack

- Budget per-agent decode at **prose C4–C6 with a reader present**: 5–10 tok/s, not the C8 aggregate.
- Prefix caching is the lever that works: the lifetime hit rate on this engine is 89 %
  (`vllm:prefix_cache_hits_total / queries_total`), so a stable system-prompt + tool-schema prefix per
  profile means only the new turn's tokens are read. A system prompt that embeds the current time
  would defeat it entirely.
- Concurrency above 4–5 workers raises aggregate throughput but lengthens every card and, at 6, this
  engine left one request without a chunk for 15 minutes while serving the others at 5 tok/s (the
  harness's stale-stream guard then reset the card).
- Reading fewer tokens is the only remaining speed lever, and it trades against what the agents know.
  We chose not to cap tool output for that reason.

## 7. Instruments

`scripts/prefill-interference.py` — the §3 measurement, self-contained: `--corpus DIR` builds the
long prompt from any directory of text files (a synthetic filler is used when none is given), writes
the window table and a JSON file. `scripts/bench-sweep.py` (§1) and the long-context concurrency
sweep of §2 are described in [docs/09](../../docs/09-measurement-protocol.md).
