# Speed map of the corrected engine — what correct routing costs, and where the step time goes (7 October 2026)

**Applies to: TP=3, the main (vLLM `main`) stack, bias-fix build.** This page replaces the speed figures of
the previous stack as a basis for comparison. Those were measured with the MoE router bias missing (see the
README's correctness notice): correct routing makes the engine read more experts, so the old numbers were
fast partly because they selected the wrong ones.

## 0. Settings

| | |
|---|---|
| Image | `a4g-21d93d0d8-1451b1a` (bias-fix build), one A/B arm on `a4f-21d93d0d8-f3e184a` (pre-fix, router bias zero) with identical software otherwise. Production adds the SwiGLU clamp (`448f1d6`), which did not move speed ([kl-health.md](kl-health.md) §6) |
| Cluster | 3× DGX Spark (GB10), TP=3 + expert parallelism, fast-load LOAD path, `NCCL_MESH_TIMEOUT_SEC=180`, JIT cache gate passed (193 files equal on all nodes) |
| Engine | `gpu-memory-utilization` **0.75** (the production fraction is 0.84; not used here), KV `fp8`, `--max-model-len 1000000`, `--block-size 256`, `HAREM_SW_BLOCK_SIZE=256`, `--max-num-batched-tokens 2048`, **`--max-num-seqs 5`**, FlashKDA prefill, `--attention-backend CUSTOM`, DFlash2 draft **k=7** (fp8 draft KV; draft attention = FlashInfer, so **CUDA graphs resolve to NONE**, [rejected-draft-cuda-graph.md](rejected-draft-cuda-graph.md)), no per-batch-size schedule, no NODROP, 16 images / 4 videos per request, vision on |
| Sampling | temperature 0, reasoning effort **low** (server default, also set in each request), 256 generated tokens per request (128 for the short-Turkish set) |
| Protocol | each arm in its own boot **after a triple reboot of the cluster** (a reboot restores the memory bandwidth, [bandwidth-fragmentation.md](bandwidth-fragmentation.md)); step time as the primary quantity because token rates move ±5–12 % per cell with the draft's acceptance and ±1–3 % in step time; two sweep rounds with **round 2 counted** for the per-content and `hizset-v2` tables; three 8-second windows per scenario for the step-time driver. **This is fewer than the five-round rule for comparing engine arms** ([docs/09](../../docs/09-measurement-protocol.md)): the routing effect in §1 (+11 % / +20 %) is four to ten times the step-time noise (±1–3 %) and was predicted independently from the bytes read, so we accept it; no smaller effect is claimed anywhere in this file |
| Raw | [`raw/speed-by-content.csv`](raw/speed-by-content.csv) (40 cells), [`raw/routing-step-time.csv`](raw/routing-step-time.csv) |

All rows `[measured-here]`; the harness (step-time driver, expert counter hook, profiler classifier) is
not in this repository `[measured-here, private harness]` except where the repository's own
`scripts/bench-sweep.py` and `hizset-v2.jsonl` are named.

---

## 1. Correct routing costs 11 % of a single-user step and 20 % of a four-user step

Same session, same software, two consecutive boots; the only difference is whether the router bias is
loaded. A read-only counter hook recorded, for every MoE layer and rank, how many **distinct experts** the
step touched.

| Scenario | Step, bias missing | Step, bias loaded | Δ | Distinct experts / rank / layer, missing → loaded | Experts' extra bytes / step | Predicted Δ at 213 GB/s |
|---|---|---|---|---|---|---|
| One user, code prompt (8 verification tokens) | 70.3 ms | **78.2 ms** | **+7.9 ms (+11 %)** | 10.53 → **13.50** (+28 %) | +1.57 GB | +7.4 ms |
| Four users, mixed (32 tokens) | 139.4 ms | **167.0 ms** | **+27.6 ms (+20 %)** | 29.54 → **40.77** (+38 %) | +5.95 GB | +27.9 ms |
| One user, plain English | 67.9 ms | 75.6 ms | +7.7 ms (+11 %) | 9.50 → 12.24 (+29 %) | | |

The prediction is the extra expert bytes (12.62 MB per expert) divided by the MoE kernel's measured
throughput (~213 GB/s, 89 % of the 240 GB/s practical ceiling): it lands within 0.5 ms of both measured
deltas. **The pre-fix build reproduces the previous stack's speed to the digit**: four-user step 139.4 ms against
139.4 ms in the previous stack's 0.75 baseline, one-user 70.3 against 70.5 in the earlier pre-fix boot. So
the new software is not slower; the correct selection is. (Mis-routing had concentrated the step's tokens
on a few "popular" experts, so it read fewer of them.)

What the user sees, same-session pair: aggregate decode at four users **−12 %** (111.9 → 98.3 tok/s),
single stream **−5 %** (45.7 → 43.5 tok/s; the draft's acceptance recovered about 3 points of the 11 % step
cost). Against the pre-fix build's `hizset-v2` sweep from earlier the same day (a different boot, so weaker
evidence) the new build reads −8.6 % at C1, −13 % at C2, −16 % at C4. Prefill did not change (§4).

Verification width drives expert reads almost linearly: distinct experts per rank and layer, by tokens in
the step, **M=8: 12.4–13.5, M=16: 21.7, M=24: 32.4, M=32: 40.8** `[measured-here]`. That is the lever the
[draft schedule](draft-schedule.md) pulls.

**What this cost** — this is the price of correctness, not a regression to hunt: the previous stack's
decode figures should not be quoted as this stack's baseline, and this stack's should not be quoted as an
improvement over them.

---

## 2. Single-stream speed depends on what is being written

Fixed prompts, five content types × four prompts, concurrency 1/2/4/5, **round 2**, per-stream decode
tok/s (median over streams) / aggregate tok/s. Draft k=7 as above. The harness also records time to first
token and per-position draft acceptance. These are realistic prompts of each kind, not the synthetic count-up ceiling
of [docs/09](../../docs/09-measurement-protocol.md); but there are only 4 per type and 256 tokens each, so they are a
ruler for engine changes, not a usage forecast.

| Content | C1 | C2 | C4 | C5 | Draft acceptance (positions 1–7) |
|---|---|---|---|---|---|
| English prose (explanations, essays) | 35.1 / 34.3 | 24.5 / 44.2 | 17.1 / 63.9 | 16.1 / 74.1 | 23.5 % (74/45/24/12/6/3/1 %) |
| Code (2 Python, 2 GDScript) | 59.0 / 57.7 | 41.7 / 68.5 | 29.6 / 93.6 | 28.9 / 103.2 | 51.3 % (87/70/58/48/41/31/25 %) |
| Tool call / JSON (strict tools, 4.1k-token prompt) | 48.2 / 36.8 | 34.7 / 50.2 | 20.3 / 60.7 | 20.3 / 66.2 | 40.6 % |
| Mathematics | 70.3 / 66.4 | 53.1 / 97.3 | 34.4 / 123.2 | 31.0 / 142.7 | 66.6 % (92/82/74/66/58/50/45 %) |
| Short Turkish answer (128 tokens) | 22.5 / 21.6 | 15.7 / 30.1 | 12.4 / 44.2 | 10.8 / 50.4 | **10.1 %** (46/18/5/2/0/0/0 %) |

Time to first token, median (worst), seconds: prose 0.18 (0.20) at C1, 0.62 at C5; code 0.26 / 0.63;
**tool call / JSON 1.40 (2.51) at C1, 4.63 (7.45) at C5**; Turkish 0.34 / 0.46. The tool-call row is slow
because a 4.1k-token prompt is re-read whole each time on the drafted engine — the prefix-cache grain
finding of [prefix-hits-nodrop.md](prefix-hits-nodrop.md) §1, which the NODROP setting later fixed.

Sub-types: Python 75.6 against **GDScript 49.4 tok/s (−35 %)** at C1, 36.6 against 25.6 at C4; the draft
guesses GDScript worse. Thinking at effort `low` is nearly absent (14.2 % of generated tokens on maths,
0.2 % on JSON, 0 elsewhere).

### Is the draft helping? Drafted against draft-less, same prompts

Per-stream tok/s, drafted / draft-less. The draft-less arm runs with CUDA graphs and the drafted arm cannot,
which favours the draft-less arm:

| Content | C1 | C2 | C4 | C5 |
|---|---|---|---|---|
| English prose | 35.1 / 32.5 (1.08×) | 24.5 / 25.1 (0.98×) | 17.1 / 18.3 (0.93×) | 16.1 / 16.5 (0.97×) |
| Code | 59.0 / 32.6 (**1.81×**) | 41.7 / 25.0 (1.67×) | 29.6 / 18.4 (1.61×) | 28.9 / 16.5 (1.76×) |
| Tool call / JSON | 48.2 / 32.4 (1.49×) | 34.7 / 24.7 (1.40×) | 20.3 / 18.1 (1.12×) | 20.3 / 15.8 (1.28×) |
| Mathematics | 70.3 / 32.5 (**2.16×**) | 53.1 / 24.9 (2.14×) | 34.4 / 18.2 (1.89×) | 31.0 / 16.5 (1.87×) |
| Short Turkish answer | 22.5 / 32.7 (**0.69×**) | 15.7 / 24.8 (0.63×) | 12.4 / 19.6 (0.63×) | 10.8 / 17.4 (0.62×) |

Draft-less step time is content-independent (C1 30.6–30.8 ms, C2 40.0–40.9, C4 51.9–56.0, C5 58.0–63.0), so
the draft pays only when it buys more than **2.4–2.8 tokens per step** (drafted step 74–79 ms at C1, 2.4–2.6×
longer). Acceptance of 10 % buys 1.7 on Turkish: **the draft is a loss there at every concurrency**. On
aggregate throughput prose and tool-call JSON also lose at C4–C5 (0.89–0.92×, 0.90–0.91×). Against the
eager draft-less step (44.9 ms, ~22 tok/s) the Turkish loss would be nil `[estimate]`, so the
drafted-eager comparison and the graphed draft-less one disagree about whether this is a loss: the graph is
what the drafted engine is missing here ([rejected-draft-cuda-graph.md](rejected-draft-cuda-graph.md)).

---

## 3. Where the step time goes

Profiler on the live server, all three ranks, kernel time classified per stream (see §6 for the classifier
bug found on the way), one-user step 76.1 ms and four-user step 163.5 ms. Ideal = bytes the step must read
per rank ÷ 240 GB/s (byte model, `[estimate]` for the ideal column; the times are measured).

| Item (one user) | ms | Share | Ideal | Of ceiling |
|---|---|---|---|---|
| MoE experts (rank mean) | 33.3 | 43.8 % | 29.8 | **89 %** |
| Dense GEMM (EXL3 + bf16) | 15.3 | 20.1 % | 9.7 | 64 % |
| Draft model | 5.4 | 7.1 % | 4.5 | 83 % |
| mHC | 2.3 | 3.0 % | 0.6 | 26 % |
| KDA | 1.6 | 2.1 % | 0.5 | 30 % |
| MLA / DSA + indexer | 0.7 | 0.9 % | 0.2 | 23 % |
| Sampling / logits | 0.7 | 0.9 % | 0.7 | 95 % |
| Communication wire (105 collectives, 39.7 µs each) | 4.1 | 5.4 % | 4.1 | 100 % |
| **Rank wait (imbalance)** | **7.7** | **10.1 %** | 6.5 | 85 % |
| GPU idle (from `nvidia-smi` use, understates CPU launch loss) | 4.6 | 6.1 % | 0 | |
| Unexplained residual | 1.4 | 1.8 % | | |
| **Total** | **76.1** | | **56.5** | **74 %** (66 % of the 273 GB/s paper figure) |

Four users: MoE 101.6 ms (62 %, 89 % of ceiling), dense GEMM 17.0, **KDA 8.5 ms (23 % of ceiling — the
largest small-operation item)**, mHC 3.75, draft 9.0, rank wait 11.0 ms (7 %), total 163.5 against an ideal
124.4 (76 % / 67 % of paper).

Rank wait is 7.7 / 11.0 ms per step; **37 % (one user) and 43 % (four users) of it is layer-specific** — in
a given layer the popular experts keep landing on the same rank — and could be attacked by per-layer expert
placement or hot-expert replication (`[estimate]` ~2.8 ms at C1, ~4.7 ms at C4; not built). The rest is
random: a different rank trails each step.

Where the previous stack's worst imbalance (one node doing 16–20 % extra expert work) went: it is **not
present** in three boots of this stack (nodes within 1–2 % of each other, also on the pre-fix build) and the
old reading was specific to one boot; do not carry it over.

---

## 4. Concurrency and prefill

`hizset-v2` (12 short English code prompts, 256 tokens, temperature 0), `scripts/bench-sweep.py`
method, round 2, `--max-num-seqs 5`:

| C | Draft acceptance | Tokens / step | Aggregate tok/s | TTFT median (s) |
|---|---|---|---|---|
| 1 | 61.1 % | 5.27 | **64.3** | 0.31 |
| 2 | 61.8 % | 5.32 | 88.0 | 0.45 |
| 4 | 63.6 % | 5.45 | 120.1 | 0.68 |
| 5 | 60.8 % | 5.25 | 133.3 | 0.74 |
| 6 | 62.5 % | 5.38 | 136.5 | 1.03 |
| 8 | 63.0 % | 5.41 | 136.6 | 5.63 |

C6 and C8 are slow by configuration, not by engine fault: with five slots, the sixth and later requests wait.
Sampled at 0.5 s from `/metrics`, mean queue time was **0 s up to C5, 1.41 s at C6, 4.46 s at C8**; the
queue accounts for 88 % of the C8 TTFT, and TTFT with the queue removed equals C5's. Aggregate saturates at
C5 (133 → 136 → 137). The previous recipe's C6/C8 figures were taken at `--max-num-seqs 8`; this is a
setting, not a regression.

**Prefill** (cold prompts, not in the prefix cache; the step driver on 2,048-token chunks, not the
repository's `bench/prefill-fresh.py`): 32k tokens **1,809 and 1,816 tok/s** (17.7 / 17.6 s), 90k **1,791
tok/s** (50.3 s). Correct routing reads **41 % more expert bytes** per prefill chunk (77.7 against 55.1 of 96
local experts), yet prefill speed moved only −0.7 % to −1.9 %: prefill's MoE is not bandwidth-bound
`[measured-here]`. The repository's own fresh-prompt prefill harness has not been re-run on this stack
`[not tested]`.

---

## 5. What we tried and rejected, open problems, retracted

**Rejected, with evidence** (each `[measured-here]`):

- *Draft KV in bf16 to unlock CUDA graphs.* Graph opens, step −4.0 % at one user, but the KV pool shrinks
  11.5 % and acceptance does not rise: no net gain. Superseded by the cleaner attempt in
  [rejected-draft-cuda-graph.md](rejected-draft-cuda-graph.md).
- *Draft-less decode as the default.* It loses 45–55 % of the speed on code and mathematics (§2) and only
  wins on Turkish.

**Open problems**

1. Small operations plus CPU launch overhead (~9 ms, 12 % at one user; ~21 ms, 13 % at four users): KDA
   multi-sequence decode update first (8.5 ms at four users, 4.4× its ideal), then mHC decode and
   norm/elementwise fusion.
2. The Turkish and prose rows (§2): draft length by content. The per-batch-size schedule
   ([draft-schedule.md](draft-schedule.md)) takes the multi-user part; the single-user part is open.
3. Dense EXL3 GEMM at tiny M (64 % of ceiling at one user).
4. Why prefill MoE is not bandwidth-bound while reading 41 % more expert bytes.
5. Draft acceptance on a ~90k-token document-reading prompt read 45.7 % and 52.4 % with correct routing
   against 63 % in one earlier run (the previous stack had read 43 % on another). A comparison of the
   generated text, two samples per arm, traced the difference to the correctly routed model writing tighter
   records (no repetition or garbage in either arm: 4/8/16-gram repeat ≈ 0). Sample size small.

**Retracted / superseded**

- The previous stack's absolute decode figures and its "one node does 16–20 % extra work" reading, as a baseline
  for this stack (README correctness notice; [`docs/11`](../../docs/11-open-issues.md) §1.15).

---

## 6. Instrument notes

- The profiler classifier assigned a GEMM to "MoE" if it followed the **last seen** `had_in`, held in one
  variable for all streams. The shared expert runs on its own CUDA stream, so its `had_in` landed between the
  main stream's MoE `had_in` and GEMM and moved ~15.7 ms of MoE into "dense". Fixed by keeping the variable
  per stream; the earlier traces were re-analysed with the fixed tool and **their conclusions did not
  change**.
- `nvidia-smi` utilisation reads 93 % on an eager draft-less step that a CUDA graph shortens by 31 %; it does
  not see CPU launch loss and so understates it.
- Profiling inflated the step by +4.6 % (one user) / +1 % (four) on this stack, against +16–19 % on the
  previous one, so the itemised table is closer to the unprofiled step than before.
