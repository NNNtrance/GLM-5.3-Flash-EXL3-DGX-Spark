# Eight hours under mixed sampled load: 7,515 requests, 0 errors, 6 of 6 criteria (8–9 October 2026)

**Applies to: TP=3, the main (vLLM `main`) stack, production configuration, served by the production
systemd unit on the production port.** The soak asks only whether the engine stays up, stays within memory
and stays as fast at the end as at the start under a realistic mix of work at the configured concurrency. It
does not measure how good the answers are; answers are checked and reported, but are not a criterion.

## 0. Settings

| | |
|---|---|
| Engine | production configuration: image `a4i-21d93d0d8-448f1d6`, TP=3 + EP, `gpu-memory-utilization` **0.84**, `--max-model-len 1000000`, `--max-num-seqs 5`, `--max-num-batched-tokens 2048`, KV `fp8`, draft k=7 with NODROP and the `[[1,1,7],[2,8,3]]` schedule, vision 16 images / 4 videos, `--tool-call-parser glm47` (fail-closed), default `reasoning_effort` `low`, started from power-on by the autostart unit, watchdog active ([boot-and-watchdog.md](boot-and-watchdog.md)); the engine containers had been up since 17:06 on 8 October (right after the watchdog drill), four minutes before the capacity phase and 20 before the soak |
| Load | a client container on one worker node, limited to 1 GiB and 2 CPUs so it cannot disturb the engine; **sustained 5 concurrent requests** (the slot count): **four agent sessions** (multi-turn, tool calls, strict tools, fabricated tool results) + **one mixed worker** (code, JSON through tools and through a schema, English prose, Turkish, mathematics). **Temperature 0.7, top-p 0.95**, reasoning effort `low` on 70 % of requests and `high` on 30 % |
| Heavy jobs | an **image request** (2–4 images, 800×600 to 1600×1200, generated on the spot as coloured squares to be counted) about every 10 min; a **long-context request** (~100k tokens, two needles) about every 75 min; the intervals jitter ±40 %; no new heavy job in the final 10 minutes |
| Samplers | read-only recorders on each node: memory and swap every second, GPU errors (`Xid`) every minute, container state, GPU temperature and power; a bare flag file under 512 MiB. They never touch the engine |
| Dates | capacity phase 8 Oct 17:10–17:26, soak 8 Oct 17:26:17 → 9 Oct 01:26:30 (8.0 h), report written by the harness at 01:28:45 |
| Raw | [`raw/soak-hourly.csv`](raw/soak-hourly.csv) |

All rows `[measured-here, private harness]` (the load generator and reporter are not in this repository).
The reporter was dry-run against eight synthetic scenarios before use — pass, speed drop, low memory, `Xid`,
restart, swap growth, many errors, unfinished — and gave the expected verdict for each.

## 1. The six criteria (written before the run)

| # | Criterion | Result | Measured |
|---|---|---|---|
| 1 | error rate < 1 % | **pass** | **0 of 7,515** requests failed (0.00 %) |
| 2 | engine did not die, restart or hang 10 minutes | **pass** | all three containers ran uninterrupted, one start each |
| 3 | every node's free memory floor ≥ 1 GiB | **pass** | head **4.67 GiB**, worker-1 **6.93 GiB**, worker-2 **7.02 GiB** |
| 4 | swap does not grow without bound (last-2-hours slope ≤ 256 MiB/h, peak < 14 GiB) | **pass** | peak 0.0 GiB on all three, slope +0 MiB/h |
| 5 | no GPU error (`Xid`) | **pass** | +0 on all three nodes |
| 6 | speed not more than 10 % below the first hour | **pass** | worst hour **1.00×** of hour 1 |

GPU maximum temperature / power: head 76 °C / 59 W, worker-1 74 °C / 56 W, worker-2 78 °C / 61 W. Sampler
gaps: none (longest 0 s).

**Independent second source (written afterwards from the engine, not from the harness):** the three engine
containers started 8 Oct 17:06 (after the watchdog drill), `RestartCount` 0, no out-of-memory kill; the
engine logs held **0** `Traceback`, `ERROR`, `CUDA error` and `NCCL WARN` lines from soak start to end;
`/metrics` counted **8,451** successful completions (7,760 stopped + 691 hit their length limit), **0**
errors, **0** aborts, **0** preemptions — which agrees with 7,515 soak requests + ~360 capacity-phase requests
+ ~620 watchdog probes (8 tokens, length-limited) to within the approximations in the last two terms.

## 2. Speed, hour by hour

Criterion 6 uses the **median over six content categories of each category's median decode tok/s**, per hour,
normalised to hour 1 (requests overlapping a heavy job excluded). The harness reports seven complete hours.

| Hour | Ratio to hour 1 | Requests | First-token median (s) | Errors |
|---|---|---|---|---|
| 1 | 1.006 | 952 | 1.050 | 0 |
| 2 | 1.038 | 872 | 1.021 | 0 |
| 3 | 0.996 | 992 | 1.031 | 0 |
| 4 | 1.052 | 927 | 1.010 | 0 |
| 5 | 1.002 | 901 | 1.036 | 0 |
| 6 | 0.996 | 978 | 1.044 | 0 |
| 7 | 1.004 | 911 | 1.069 | 0 |

(7,515 total includes the eighth hour, not tabulated; hours 1–7 sum to 6,533.)

First-hour per-request decode rate by category, median, **under mixed concurrent load at five slots — not
a clean speed table**: plain prose 17.2, agent turns 21.1, code 22.1, JSON 20.6, Turkish 11.5, mathematics
25.2 tok/s. Use [speed-map.md](speed-map.md) for single-purpose speeds.

## 3. Answer checks (information, not a criterion)

| Category | Requests | Failed | Cut at length limit | Checked | Wrong |
|---|---|---|---|---|---|
| agent | 6,389 | 0 | 26 | 6,363 | **0** |
| plain prose | 153 | 0 | 0 | 153 | 0 |
| JSON | 199 | 0 | **34 (17 %)** | 165 | 0 |
| code | 275 | 0 | 0 | 275 | 0 |
| mathematics | 210 | 0 | 0 | 210 | **2** |
| images | 54 | 0 | 0 | 54 | **2** |
| Turkish | 217 | 0 | 0 | 217 | 0 |
| long context (needles) | 18 | 0 | 0 | 12 | 0 |

"Wrong" means: the computed result, the JSON schema, the tool call's validity, the count of coloured squares,
the needle in the long document, the language of a Turkish answer. The wrong-answer rate per category was
0.0 % in the first hour and in the last hour for every category that had enough samples to compare
(agent 0/844 → 0/763; prose 0/23 → 0/22; JSON 0/21 → 0/24; code 0/32 → 0/42; mathematics 0/18 → 0/27;
Turkish 0/29 → 0/21). The 2 wrong mathematics results therefore fell in neither the first nor the last hour;
the hours of the 2 wrong image counts were not reported. Causes were not analysed `[not tested]`.

**The 17 % of JSON requests cut at the length limit are a property of the test, not of the engine:** the
soak sets a small `max_tokens` that the thinking budget at effort `high` can consume. It says something worth
knowing about structured output at a tight output budget, but it was not measured further.

## 4. Prefix-cache capacity (measured before the soak, same boot)

56 sessions of ~24k tokens each (six turns, each session unique) were filled once, then 24 of them were
probed from newest to oldest; hits counted from both `cached_tokens` and `/metrics`:

| | |
|---|---|
| Probed / full hit / partial / miss | 24 / **21** / 0 / 3 |
| Finished history still served from cache | **0.47 M tokens** |
| Capacity | **~0.49 M tokens** (upper bound 0.51 M) |

That is roughly 8 % of the 6.19 M KV pool: the cache keeps about 20 finished 24k-token conversations. An
older estimate for the previous stack (0.60–0.65 M) used another method and is not comparable.

## 5. What this cost, and what it does not cover

- **Cost:** eight hours of the whole cluster, and the first sampled-load exposure of this configuration: an
  upstream report describes a memory overflow in sampled long runs of the DFlash2 draft (#55279 `[reported]`).
  Not seen: no floor approached 1 GiB and swap stayed at zero. The lowest values were reached at 01:06 (head,
  40 minutes before the end), 21:05 (worker-1) and 00:05 (worker-2); a slow downward drift of the head's
  floor late in the run cannot be excluded from three minima alone, and the per-minute series is not
  tabulated here.
- **Not covered:** the **16-large-image** worst case (the soak's images are small; that case is in
  [memory-fraction-084.md](memory-fraction-084.md)); 1M-token contexts (the long jobs are ~100k); a
  continuous single-user session of many hours; load above five concurrent requests; any restart mid-soak
  (the watchdog drill, which is separate, is in [boot-and-watchdog.md](boot-and-watchdog.md)).
- The speed criterion is coarse on purpose: cells wander ±5–12 % with draft acceptance, so a 10 % threshold
  detects a collapse, not a 3 % drift. Hour 4's 1.052 is faster than hour 1, not slower.
- Answer quality is a spot check, not an evaluation.

## 6. Rejected, open, retracted

**Rejected:** none.

**Open**

- Whether anything degrades beyond eight hours — the next run should be longer and include the 16-image load.
- The 4 wrong answers (2 mathematics, 2 image counts) were not analysed for cause.
- The harness's low `max_tokens` for JSON makes its JSON category weak evidence about structured output.

**Retracted:** none.
