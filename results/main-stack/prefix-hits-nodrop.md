# Agent turns re-read half as much: `disable_eagle_block_drop` ("NODROP") and its prefix-cache gates (8 October 2026)

**Applies to: TP=3, the main (vLLM `main`) stack, DFlash2 draft.** One speculative-config key,
`"disable_eagle_block_drop": true`, shortens the first token of every follow-up turn in a multi-turn
conversation by about half and cuts the tokens re-read from scratch by 61 %. In production since
8 October. In the previous recipe this option was carried only as a diagnostic arm
([`tracks/tp3/patches/prefix-hit-and-kpool-tail/`](../../tracks/tp3/patches/prefix-hit-and-kpool-tail/README.md));
on this stack it earned its place, and §4 says why the two verdicts differ.

```
--speculative-config '{"method":"dflash","model":"<draft>","num_speculative_tokens":7,"kv_cache_dtype":"fp8","disable_eagle_block_drop":true}'
```

The option is upstream's: vLLM [#53388](https://github.com/vllm-project/vllm/pull/53388) by
[@ZeldaHuang](https://github.com/ZeldaHuang), merged 1 September 2026, documented there as experimental for
acceptance. On the pinned tree `21d93d0d8` it is `config/speculative.py` and the scheduler logs
`EAGLE trailing prefix-cache block dropping is disabled` at start (the boot evidence we gate on).

## 1. The mechanism

On this hybrid model the scheduler's block is **3,328 tokens** (the least common multiple of the KV-group
page sizes `[3328, 16, 3328 ×4, 256]`) and the hash block is **256** (the greatest common divisor): fine-grained
256-token cache hits are enabled. The DFlash2 draft's KV group is flagged as an "EAGLE group" by
`HAREM_PREFIX_HIT=1`, and for an EAGLE group vLLM by default **drops the last matched block** (the draft group
looks one 256-block further and gives it back) and shifts the KDA partial tail back by 256.
Together with the 3,328 scheduler grain, a follow-up turn's cache hit lands about one to two big blocks short
of where it could.

- Default: hits on a repeat follow `(⌊(L−1)/3,328⌋ − 1) × 3,328` — five of five probe prompts matched
  (4,095 tokens: 0 hit; 12,575: 6,656; 28,819: 23,296) `[measured-here]`. Prompts shorter than 6,656 tokens
  get nothing.
- NODROP: the hit on an agent turn is **⌊previous prompt / 256⌋ × 256** — **60 of 60** turn-2-and-later
  requests matched on each NODROP boot (30 serial + 30 wave), and the control never matched it. The loss per
  turn is `P mod 256`, about 128 tokens on average.
- Why EAGLE needs the drop and DFlash2 does not (**code reading, not measured**): an EAGLE draft's KV at
  position t depends on token t+1, so the last cached position depends on the continuation of the request
  that wrote it. A DFlash2 draft's context KV at position t depends only on the target's hidden state at t.
  Measured consequence in §3: no change in draft acceptance.

## 2. Settings

| | |
|---|---|
| Image | `a4g-21d93d0d8-1451b1a` for the first A/B (§3); `a4i-21d93d0d8-448f1d6` for the long confirmation (§5). Fast-load LOAD sidecar, **triple reboot of the cluster before every arm** in the long confirmation |
| Engine | TP=3 + EP, `gpu-memory-utilization` **0.75**, KV `fp8`, `--max-num-seqs 5`, `--max-num-batched-tokens 2048`, `--block-size 256`, `HAREM_SW_BLOCK_SIZE=256`, `HAREM_PREFIX_HIT=1`, draft k=7 (fp8 draft KV), `--enable-prompt-tokens-details`, temperature 0 (seed 1234), reasoning effort `low`. The only variable between arms: the NODROP key |
| Arms | **ABBA**: control a1, NODROP b1, NODROP b2, control a2 (the A/A pair is the instrument's self-test) |
| Raw | [`raw/prefix-hit-agent-turns.csv`](raw/prefix-hit-agent-turns.csv) |

All rows `[measured-here, private harness]` except where the bundled `scripts/` are named; the agent-turn
workload driver is not in this repository, but its shape is below.

**Workload.** Five sessions × 7 turns. Every session starts with the same ~4.1k-token prefix — a system prompt plus the
definitions of 8 real agent-harness tools; turn-1 prompts are 4,050 tokens — so the first turn of later
sessions can hit the shared prefix.
Sessions 1 and 3 carry an additional document, ending at 61,780 and 105,306 prompt tokens; each later turn
appends 300–1,500 tokens (a tool result and a reply). Two modes: **serial** (one request at a time) and
**wave** (each turn issued for all five sessions at once). The workload is byte-identical across arms
(sha256 `2c8cce36…`).

## 3. A/B: first-token time and re-reading

Turn 2 onward, 30 requests per mode per arm.

**Serial**

| Arm | TTFT median | Mean | Worst | Tokens re-read | Hit rate | Pooled draft acceptance | Decode tok/s (median) |
|---|---|---|---|---|---|---|---|
| a1 control | 2.34 s | 2.49 | 4.72 | 111,245 | 90.9 % | 40.4 % | 46.4 |
| b1 NODROP | **1.24 s** | 1.23 | 1.96 | **43,149** | 96.5 % | 42.6 % | 48.1 |
| b2 NODROP | **1.14 s** | 1.20 | 1.96 | **43,149** | 96.5 % | 40.6 % | 47.6 |
| a2 control | 2.35 s | 2.50 | 4.74 | 111,245 | 90.9 % | 42.2 % | 47.3 |
| **NODROP / control** | **−49 %** (median), −51 % (mean), −59 % (worst) | | | **−61 %** | +5.6 pt | +0.3 pt | ≈ |

**Wave** (five sessions at once): TTFT median 7.25 / 3.83 / 3.65 / 7.23 s for a1 / b1 / b2 / a2 — **−48 %**;
worst 11.81 / 5.75 / 5.89 / 11.84 s — −51 %; tokens re-read 96,551 against 39,975 — **−59 %**.

- **Determinism of the instrument:** a1 and a2 have **identical per-request hit counts**, as do b1 and b2;
  the median TTFT of a1 against a2 differs by 0.45 %. The ruler is stable.
- **Turn 1 bonus:** with NODROP four of five sessions' first turns got the shared 3,840-token prefix from
  the cache (the control: 0). TTFT of a newly opened agent session: **2.5 s → 0.4 s**.
- **The simulation held.** A pre-run CPU simulation with the real scheduler classes predicted −56…−61 % in
  re-read tokens and roughly halved TTFT; measured −61 % and −49 %. The simulated hit formula matched the
  engine on 60/60 requests.

### Is a cached continuation as correct as a cold one?

The new path — continuing from a KDA state saved at a 256-multiple rather than a 3,328-multiple — had not been
used in production before. For every NODROP boot: teacher-forced re-scoring of a greedy continuation, cold
twice and warm (from cache) twice; first divergence position; the target's argmax agreement; and
|Δ log-probability| cold-to-cold (the instrument's own noise) against warm-to-cold. Seven points, including
three **edge points** chosen to break it:

| Point | Previous prompt P | Hit (expected) | Off the 3,328 grid? | Cold-cold \|Δlp\| | Warm-cold \|Δlp\| | Teacher agreement cold / warm |
|---|---|---|---|---|---|---|
| session 1 turn 5 | ~68.2k | 68,096 (68,096) | yes | 0.01005 | 0.00967 | 252 / 254 |
| session 0 turn 6 | ~9.3k | 9,216 (9,216) | yes | 0.01972 | 0.02229 | 251 / 254 |
| the same two, second boot | | | | 0.00542 / 0.03505 | 0.00429 / 0.03490 | 254 / 253, 251 / 251 |
| edge: P mod 256 = 1 | 7,169 | 7,168 (P−1) | yes | 0.01715 | 0.01424 | 253 / 254 |
| edge: P mod 256 = 255 | 8,447 | 8,192 (P−255) | yes | 0.04129 | 0.04635 | 248 / 253 |
| edge: P mod 3,328 = 164 | 13,476 | 13,312 (4 × 3,328) | no (full block) | 0.02117 | 0.02080 | 251 / 251 |

The warm-to-cold deviation sits at the cold-to-cold noise (ratio **0.79–1.13** over the seven points);
greedy divergences start at the 2nd–22nd token even between two **cold** runs (this engine is not
deterministic) and the warm ones sit in the same band. A stale or wrong KDA state would have raised the warm
|Δlp| to multiples of baseline and dropped warm agreement; a deliberately broken mock engine scored 5/256
agreement and Δ 0.5. **Not seen.**

**Not tested:** a previous prompt whose length is an exact multiple of 256 (P mod 256 = 0). A simulation says
that case drops the hit for that turn (about one turn in 256; it falls back to the previous turn's hit).

## 4. Pre-registration, a flawed rule, and the confirmatory arm

The rule (written between 00:50 and 01:00 on 8 October, before any data; the A/A pair must show no "gain"),
seven clauses named by what they test: *evidence* (the boot log shows the right mode); *self-test* (a1 and
a2 have identical hits; their median TTFT differ by less than θ = max(0.20, 3 × their gap)); *hits*
(re-read tokens ≤ 60 % of the control; b1 = b2; the simulation holds on ≥ 90 % of requests); *TTFT* (≤ (1−θ)
× control); *acceptance* (≥ control − max(3 pt, 2 × A/A gap)); *correctness* (every NODROP boot, with an
off-grid point: warm agreement ≥ cold − 0.02; \|Δlp\| warm-cold ≤ 1.5 × cold-cold + 0.005; earliest warm
divergence ≥ half the earliest cold one); *pool* (**KV pool within the control's range ±0.2 %**).

**Literal verdict: reject.** Evidence, self-test, hits, TTFT and acceptance passed; two clauses failed.

1. **The correctness clause was not measurable on b1 (a tool-ordering defect, not an engine fault).** The cold "teachers" wrote the
   prompt into the cache first; the warm teacher then got a cache hit that extended *past the first row to be
   scored* (hit 68,864 > scoring start 68,744; 12,032 > 11,893), and vLLM's prompt-logprob scorer drops such a
   request, so the warm value came back empty. The tool ran the warm teacher after the cold ones; it now runs
   it first.
2. **The pool clause failed (4,645,776 against a lower bound of 4,652,801).** The ±0.2 % band came from a single earlier
   observation (0.1 %). Over six boots of this one setting that night the pool spread **0.59 %** — and the
   spread follows the free memory of the lowest rank (33.25–33.45 GiB, ~139.7k tokens per GiB); the flag
   does not enter the pool arithmetic (code reading).

**What we did.** The literal verdict stands in the log as reject, neither applied nor reversed. The two
causes were diagnosed and written down. A corrected rule — correctness on seven points
including three off-grid edges with the warm teacher run first; hits identical across b1, b2 and an extra
boot; pool within the *observed six-boot* range 4,641,924–4,682,370 — was committed to the log
**before the extra boot's data existed**, and one more NODROP boot (b1d) was run with the fixed tool and the
edge points. All seven correctness points passed, hits were identical across the three NODROP boots, and the
pool stayed in range.
Verdicts side by side: literal — reject; diagnosis — two instrument defects, not engine; corrected,
confirmatory — gain.

Boot-to-boot KV pool at gmu 0.75, six boots: a1 4,673,024 · a2 4,662,125 · b2 4,664,850 · b1 4,645,776 ·
b1d first boot 4,648,501 · b1d 4,656,675. The 0.59 % spread is the boot noise to use for KV claims, not
0.2 %.

**The earlier rejection.** On 6 October, on the previous stack (broken-routing build, our own patch rather
than the upstream option), the same idea was rejected on: same-prompt divergence lens 1.81 / 1.78 against
control 1.12 / 0.58, strict tool-call gate 1 corrupted + 1 rejected in 120 turns, mean accepted length on hit
turns 2.15 against 2.50, four-user step −4.1 %. **None of those reproduced on this stack** (§5). That
rejection is superseded for this stack; it was not evidence about NODROP itself.

## 5. The long confirmation on the production build (`a4i`), and a second flawed rule

Because the earlier rejection's gates were never run on this stack, the five old gates were rebuilt for it
and run as an ABBA on `a4i`: **M** same-prompt divergence lens, **T** strict tool-call gate (120 turns per
arm), **K** accepted length per hit turn against a cold twin, **R** resume mismatches on hit turns (standard
and edge points), **H** speed (step-time driver at one and four users, content-fast mode). Rules were
written first and tested against the old raw data: they rejected the old NODROP arm and reported "no
difference" for the old control against itself; three of the original plan's thresholds were too tight and
were changed *before* data (K: fixed −0.10 → 2 standard errors, false-alarm rate on a control against itself
25 % → 1.7 %; R: a per-arm count → one-sided Fisher on hit turns only, 23 % → 3.6 %; fast-mode four-user
speed demoted to report-only, because it moves a few per cent between identical boots).

| Gate | Control (a1, a2) | NODROP (b1, b2) | Rule | Result |
|---|---|---|---|---|
| **T** strict tool-call, turns / corrupted / rejected | 120/0/0, 120/0/0 | 120/0/0, 120/0/0 | NODROP: 0 and 0 | **pass — 480 turns, 0 corrupted, 0 rejected, 0 errors** |
| **K** accepted length per hit turn | 2.102 / 2.066 (cold twin 2.143 / 2.107) | 2.109 / 2.075 (cold twin 2.092 / 2.125) | B − A ≥ −max(0.10, 2 SE) | **pass**: B − A **+0.009** (SE 0.024) |
| **R** resume mismatches on hit turns | 2/72 | 5/72 | one-sided Fisher p < 0.05 | **pass** (p = 0.221) |
| **H** four-user step | 161.35 / 161.35 ms | 161.35 / 162.23 | B/A ≤ 1.02 | pass (**1.0027**) |
| **H** one-user step | 76.07 / 76.80 ms | 77.09 / 77.39 | B/A ≤ 1.01 | **fail: 1.0106** |
| **H** content-fast one-user | 74.67 / 74.61 | 74.88 / 74.87 | B/A ≤ 1.01 | pass (1.0032) |
| **H** acceptance (two sources) | 40.71 % | 40.91 % | B ≥ A − 3 pt | pass (+0.20) |
| **M** same-prompt lens (two prompts) | 1.15 / 0.77, **1.90 / 0.85** | 0.68 / 1.74, 0.68 / 1.20 | mean B ≤ max(1.5, mean A + 0.5); **A/A self-test** | B raw pass (1.21 ≤ 1.50, 0.94 ≤ 1.88); **self-test failed → gate invalid** |

Hit counts: `⌊P/256⌋ × 256` on every hit turn (standard 10/10, edges 25/25); the two independent hit
sources (`/metrics` and `cached_tokens`) agreed on all 44/44 turns of the four arms. 16 of 16 lens outputs
identical and 16 of 16 expected tool calls in every arm.

**Literal verdict: reject**, on one gate: the one-user step ratio 1.0106 against a 1.01 bar (0.06 points
over). The A/A control against itself read 1.0096, 0.04 points inside the bar, and the bar is narrower
than the spread between identical boots: control boots of one setting read 0.96 % apart (a1 against a2), the
draft-schedule chain 0.9 % (one step), the production-clamp chain 1.85 % (the step driver's resolution is
0.9 % per step at one user). **The threshold, not the engine, was the problem**, and the lens gate was not
valid (A/A: 1.90 against 0.85).

**Speed-only confirmatory chain.** Six boots a b a b a b, a triple reboot before each, the speed driver
only (3,200-token one-user windows of 40 s, 30 s four-user windows), pooled ratio, rule written before data
(≤ 1.010 pass, ≥ 1.020 reject, between: gray):

| | A (a1, a2, a3) | B (b1, b2, b3) | B/A | |
|---|---|---|---|---|
| One-user step | 76.834 ms (9 runs) | 76.932 ms (9 runs) | **1.0013** | pass |
| Four-user step | 161.691 (8 valid runs) | 161.585 (8 valid) | **0.9993** | pass |

(Two four-user runs were discarded as pre-registered, one per arm: three streams still active at window end.
Boot-to-boot spread of the controls' one-user step: 0.52 %; four-user 0.90 %.) **Verdict: pass — NODROP is
neutral on speed (+0.13 % and −0.07 %), and the full battery passes, with the lens gate recorded as invalid
rather than passed.**

Content-fast mode, report-only: four-user step 156.60 / 156.36 ms → 146.86 / 149.05 ms (**B/A 0.9455**) —
tool-call JSON, which re-read 4.1k tokens each time before, went 178–180 ms → 150–151 ms (**−16 %**).

On the production configuration (gmu 0.84, with the draft schedule) the next boot measured tool-call JSON TTFT
at one user **1.38 s → 0.35 s** and at four users **2.30 s → 0.58–0.83 s** (memory-fraction trial,
[memory-fraction-084.md](memory-fraction-084.md)).

## 6. What this cost

- **One extra pre-fill step per continuation turn** (the scheduler stops at the tail block; a pre-registered
  risk): inside the measured gain, not separately isolated.
- **Draft acceptance:** none (+0.3 pt in the A/B, +0.20 pt in the long run; the bar was −3 pt).
- **Speed:** none (one-user step ×1.0013, four-user ×0.9993 over six boots).
- **KV pool:** none from the flag; boot-to-boot ±0.3 % (§4).
- **Correctness exposure:** a code path never used in production before — resume from a KDA state saved at a
  256-grain. Tested on seven points including three off-grid edges, 480 strict tool-call turns and the soak
  ([soak-8h.md](soak-8h.md)); the `P mod 256 = 0` edge is untested.
- **Dependence on an option upstream labels experimental** for acceptance.

## 7. Rejected, open, retracted

**Rejected:** nothing in this file was rejected; the option was kept. The earlier stack's rejection is
superseded (§4).

**Open**

- The `P mod 256 = 0` edge (a turn whose previous prompt ends exactly on a 256-multiple).
- A single-process (TP=1) engine: vLLM #55601 (KDA seed index in the wrong unit on a prefix hit; merged
  7 October) `[reported]`. The multi-process executor used here has the 3,328 block size in the worker and is
  believed unaffected, and a 256-grain hit lands the seed in the right block, but a TP=1 deployment should
  take that fix or avoid hits `[not tested]`.
- Hit retention under load: how much of finished conversations stays cached is measured in
  [soak-8h.md](soak-8h.md) §3 (~0.49 M tokens), not isolated per conversation shape.
- Multi-modal prompts with NODROP were exercised in the soak and the 16-image trial, not as a hit-accuracy
  test.
- The lens gate's A/A instability (1.90 against 0.85 on one prompt) is unexplained; the isolated cause is that
  hits resemble each other more than they resemble cold fills (median divergence hit-to-cold 0.0172, hit-to-hit
  0.0091, cold-to-cold ~0.0135), i.e. the ratio rises because the *numerator* is unusually small, not
  because the hit path is wrong.

**Retracted:** none. The 6 October rejection was a decision on another build, not a published number.
