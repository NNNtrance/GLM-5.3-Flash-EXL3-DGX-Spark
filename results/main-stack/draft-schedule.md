# Draft length by number of running requests — `[[1,1,7],[2,8,3]]` (8 October 2026)

**Applies to: TP=3, the main (vLLM `main`) stack, DFlash2 draft.** One speculative-config setting that
shortens the draft from 7 tokens to 3 whenever two or more requests are in the same step. In production
since 8 October.

```
--speculative-config '{"method":"dflash","model":"<draft>","num_speculative_tokens":7,"kv_cache_dtype":"fp8","num_speculative_tokens_per_batch_size":[[1,1,7],[2,8,3]]}'
```

`num_speculative_tokens_per_batch_size` is vLLM's own dynamic-speculation table: a list of
`[first, last, K]` ranges over the number of requests **scheduled in that step**; here "1 request: K=7;
2 to 8 requests: K=3".

## 0. What the table does on this stack (code facts, pinned tree `21d93d0d8`)

- The counted quantity is requests **scheduled in this step** (prefill chunks count, waiting requests do not),
  not requests running. Ranges are inclusive, must start at 1, must not overlap; K is **silently clipped** to
  `num_speculative_tokens` (7); the table length is `--max-num-seqs` (5), so `[2,8,3]` is accepted although
  8 > 5. An invalid table fails at engine-core start, which is fail-closed.
- It is applied **one step late** under the async scheduler: placeholders for K tokens are put in at step t,
  so verification width at step t+1 is `1 + K_t` per request.
- **The DFlash2 draft always proposes its full 8-token block**; the engine truncates to the first K
  afterwards (`draft_tokens[:, K:] = -1`). So draft cost does **not** fall with K; **verification width
  does**: M = C × (1 + K) tokens per step (one user, K=3: M=4; four users: 16 instead of 32).
- Expert reads grow almost linearly with M ([speed-map.md](speed-map.md) §1: distinct experts per rank and
  layer 12.4–13.5 at M=8, 21.7 at 16, 32.4 at 24, 40.8 at 32), and with correct routing that is most of the
  step. Shorter verification is therefore worth more than before the router fix.
- The KDA/GDN state buffers are always 8 columns wide, so K changes only how many columns are written
  (1+K instead of 8) — a side benefit, not a hazard (code reading, checked by the transition test in §4).
- K=0 is valid but the draft still runs and its result is discarded; K=1 is valid (width 2).

## 1. Settings

| | |
|---|---|
| Image | `a4g-21d93d0d8-1451b1a` (bias-fix build), fast-load LOAD sidecar, one **triple reboot of the cluster before every arm** ([bandwidth-fragmentation.md](bandwidth-fragmentation.md)), memory band 233.7–245.4 GB/s in every arm (floor 230) |
| Engine | TP=3 + EP, `gpu-memory-utilization` **0.75**, KV `fp8`, `--max-num-seqs 5`, `--max-num-batched-tokens 2048`, `--block-size 256`, draft k=7 baseline (fp8 draft KV, FlashInfer draft attention, CUDA graphs NONE), no NODROP, vision on |
| Sampling | temperature 0, reasoning effort `low`, 256 generated tokens per request |
| Content set | the fixed per-content prompt set of [speed-map.md](speed-map.md) §2 (English prose, code, tool-call JSON, mathematics, short Turkish), concurrency 1/2/4/5, round 2 counted, plus the one- and four-user step-time driver (8–40 s windows) and a teacher-forced agreement check |
| Raw | [`raw/draft-schedule-arms.csv`](raw/draft-schedule-arms.csv) |

Weights for the pooled content rate (the jobs this cluster actually runs): code 0.35, English prose 0.30,
JSON/tool call 0.25, mathematics 0.05, Turkish 0.05, combined as a **weighted harmonic mean** (finishing
time is proportional to Σ wᵢ/vᵢ). The weights are ours and say so; §3 shows how much the answer depends on
them.

All rows `[measured-here]`; the chain runner, content driver and decision script are not in this repository
`[measured-here, private harness]`.

## 2. Calibration chain: how step time falls with K

Arms k=7 (control), 3, 5, 1, and a second k=7 at the end (drift control), each after its own triple reboot.

| Arm | Schedule | KV pool (tokens) | Boot | One-user step (ms) | Four-user step (ms) | Agreement / checked | Large-margin divergences |
|---|---|---|---|---|---|---|---|
| control k=7 | `[[1,8,7]]` | 4,670,299 | 185 s | **74.9** | **160.2** | 4,316 / 4,352 | 13 |
| k=3 | `[[1,8,3]]` | 4,651,226 | 186 s | **59.8 (−20.2 %)** | **116.1 (−27.5 %)** | 4,155 / 4,188 | 14 |
| k=5 | `[[1,8,5]]` | 4,656,675 | 185 s | 67.3 (−10.1 %) | 140.6 (−12.3 %) | 4,314 / 4,352 | 11 |
| k=1 | `[[1,8,1]]` | 4,645,776 | 185 s | 55.7 (−25.7 %) | 84.4 (−47.4 %) | 4,292 / 4,352 | 26 |
| drift control k=7 | `[[1,8,7]]` | 4,664,850 | 170 s | 75.6 (+0.9 %) | 160.3 (+0.0 %) | 4,310 / 4,352 | 21 |

The step curve answers the question that decided whether this was worth doing: one user, K=3 gave **59.8
ms**; the "expert reads dominate" hypothesis had predicted 56 ms and the "per-sequence work dominates"
hypothesis 71 ms. About 80 % of the predicted shortening happened. The curve is steeper from K=7 to 5 to 3
(~3.8 ms per K at one user) than from 3 to 1 (~2.1 ms per K): below K=3 it flattens into the draft's cost
and fixed work.

Pooled content rate against the k=7 control (weighted harmonic mean of per-stream tok/s):

| C | K=3 | K=5 | K=1 |
|---|---|---|---|
| 1 | +3.6 % | +4.8 % | **−29.3 %** |
| 2 | +13.5 % | +11.1 % | −5.2 % |
| 4 | +6.0 % | +2.3 % | −6.4 % |
| 5 | +11.1 % | +4.2 % | −7.9 % |

K=3 by content, against the control (C1 / C2 / C4 / C5):

| Content | C1 | C2 | C4 | C5 |
|---|---|---|---|---|
| English prose | +14.5 % | +26.6 % | +20.6 % | +19.4 % |
| Turkish | +23.8 % | +29.9 % | +32.6 % | +37.5 % |
| Code | −5.6 % | +6.9 % | −2.9 % | +0.7 % |
| JSON / tool call | −3.3 % | +2.7 % | −7.4 % | +8.3 % |
| Mathematics | −20.4 % | −14.8 % | −13.1 % | −17.4 % |

Where the draft guesses well (maths, code) a shorter draft gives away accepted tokens; where it guesses
badly (prose, Turkish: acceptance 10–24 %) it wins. The best K at **one** user therefore depends on the
content (prose and Turkish prefer K=3, maths and code K=7), which is the argument for a content-adaptive K
(§6).

## 3. The pre-registration, its flaw, and the confirmatory arm

**Rule, written before any data** (six clauses per concurrency, named here by what they test): *stream rate*
— pooled stream-rate ratio ≥ 1.05; *step model* — tokens-per-step ÷ ms-per-step ratio ≥ 1.025, as a second
source; *per-content step* — per-content step ratio ≤ 0.97; *driver agreement* — the step-time drivers agree
with the content driver within 3 points; **the agreement clause — the teacher-forced agreement count must be
at least the control's minus 30 tokens, and large-margin divergences ≤ max(2× the control, control + 10)**;
*robustness* — the verdict must survive halving or ×1.5 on every content weight. Choose the highest-scoring
passing K per concurrency, else K=7. The thresholds were first applied to the baseline against itself (it
must not report a gain) and did not.

**Literal verdict: `[[1,1,7],[2,2,5],[3,8,7]]` — a different table from the one that shipped, and it was not
applied.** The agreement clause counted *tokens*. The K=3 arm generated 4,188 tokens (one prompt ended early on an end-of-text
token at token 92 — after a near-tie of margin 1.13 at token 79, and this engine is not deterministic, so
the greedy path differs between arms) against the control's 4,352. A shorter arm lost 30 "agreement tokens"
for generating fewer tokens. As a *ratio*, K=3 agreed on **99.21 %** against the control's **99.17 %** —
equal. The rule was flawed, not the arm.

**What we did and did not do.**

1. The literal verdict was *neither applied nor reversed*; it was recorded as "undecided".
2. The diagnosis (which gate, why, with the numbers above) was written down.
3. A corrected rule — **the agreement clause as a ratio: agreement ≥ control − 0.5 points, and large
   divergences ≤ max(2× the control, control + 10)** — was written to the log **before any new data**, with the
   confirmatory arm's definition: the proposed table, run on a fresh boot beside a freshly booted control of
   the *same chain* (so a drifting baseline cannot manufacture a gain), plus a transition test (§4), a
   one-user guard (±3 % — nothing may change when K stays 7), and the stream-rate, step-model and robustness clauses as before.
4. As a diagnosis only, the corrected clause was also applied to the first chain's data: it selects `[7, 3, 3, 3, 3]`, i.e.
   the table that shipped (K=1 fails it at 98.62 % against a 98.67 % bar and a large-divergence count equal
   to its limit).
5. The confirmatory chain was then run.

**Confirmatory chain** (fresh control `k7c`, then the proposed table, each after a triple reboot):

| Gate | Result | Bar |
|---|---|---|
| Agreement, ratio form | **99.01 %** (control 99.22 %) | ≥ 98.72 % — pass |
| Large-margin divergences | 24 | ≤ 34 — pass |
| One user (K stays 7 in both) | pooled rate **+2.4 %**, step 75.6 → 74.9 ms | within ±3 % — pass |
| C2 / C4 / C5 pooled stream rate | **+10.7 % / +15.6 % / +7.1 %** | ≥ +5 % — pass |
| Step-model ratio J′ (second source) | +11.0 % / +12.6 % / +9.7 % | ≥ +2.5 % — pass |
| Weight perturbation (each weight ×0.5 and ×1.5), worst case | +7.8 % / +12.3 % / +2.8 % | ≥ 0 — pass |
| Boot / KV pool | 169 s / 4,659,400 tokens (control 4,656,675) | |
| Four-user step | 160.2 → **114.5 ms (−28.6 %)** | |

**Verdict: gain, all pre-registered gates passed.** Three verdicts side by side, as run: literal — table not
applied; diagnosis — `[[1,1,7],[2,8,3]]`; confirmatory — gain.

**How much of the +15.6 % at four users to believe.** The two k=7 controls booted two hours apart read
**−9.2 % apart at four users** in pooled stream rate (and 2.2 % at most in step time): the tool-call and
prose cells' acceptance wanders. Against the *first* chain's control the same table reads C2 +14.1 %, C4
**+5.0 %** (exactly at the bar), C5 +7.2 %; against the geometric mean of the two controls +12.4 % / +10.1 %
/ +7.1 %. The proposed table also agrees with the K=3 calibration arm to within ±3.5 % at C2, C4 and C5. The
robust finding is the **step time: −24 % to −26 % at two or more users against either control**. The stream
rate gain is real in sign and probably 7–15 % in size; the +15.6 % headline is the favourable control.

By content at two or more users, against the fresh control: C2 code −7.0 %, prose +25.8 %, JSON +10.4 %,
maths −11.6 %, Turkish +32.7 %; C4 +1.8 / +32.1 / +10.1 / −16.6 / +33.5 %; C5 −6.8 / +29.8 / −7.8 / −12.7 /
+31.3 %. At one user (K=7 in both) the cells are scatter: −3.9 % to +5.1 %.

## 4. Transitions between K values

Mixed schedules have a known upstream failure class (a transition bug, vLLM #50021 `[reported]`), so the
confirmatory chain included a transition test: four long prompts alone (calm), then the same prompts with
3-request bursts of short requests arriving behind them, moving the step between 1 and 4+ requests. Every
sequence is re-scored teacher-forced.

| | Calm | During transitions | Bar |
|---|---|---|---|
| Large-margin divergence rate | 1.24 % | **0.54 %** | ≤ max(2× calm, calm + 0.5 pt) = 2.47 % |
| Mean draft length | 7.0 | **4.0** | between 3 and 7 — the schedule really switched |
| Errors / garbage | 0 / none | 0 / none | |

34 sequences in the transition phase (4 long + 30 burst), 2,976 tokens. Of the calm phase's 19 large-margin
divergences, 17 were in a single short Turkish sequence on the *baseline* K=7 single-user path (greedy path
branches in a long generation); the same sequence diverged 3 times in the transition phase. That is
baseline behaviour, unrelated to the schedule.

## 5. What this cost

- **Mathematics loses 12–17 %** at two or more users (−11.6 % to −17.4 % across both chains); code is
  −7.0 % at C2 and −6.8 % at C5 (+1.8 % at C4) in the confirmatory arm; JSON/tool-call is mixed (−7.8 % to
  +10.4 %). The schedule is a bet on the job mix, and the weights in §1 encode that bet. A maths-heavy
  deployment should not take it.
- **Single-user speed is unchanged by design** (K stays 7), so Turkish single-user, where the draft is a
  net loss ([speed-map.md](speed-map.md) §2), is **not** improved: the biggest per-content problem is
  untouched.
- **KV pool and boot:** none (4,659,400 against 4,656,675 tokens, +0.06 %; 169 s against 169 s).
- **Risk carried:** the mixed-K transition class (§4) and an upstream report of a memory overflow in
  DFlash2 during long sampled runs (#55279 `[reported]`, unverified here). Both are the reason the schedule
  went through an 8-hour sampled-load run before production ([soak-8h.md](soak-8h.md)).
- **Measurement price:** two controls booted two hours apart disagree by 9 % on a content-weighted rate at
  four users (§3). Any further K work needs a baseline in the *same* chain.

## 6. Rejected, open, retracted

**Rejected:** a uniform K=3 schedule `[[1,8,3]]` (loses on code and maths at one user and costs 20 % of
single-user maths speed); K=1 (fails the agreement rule, −29 % at one user); K=5 (a smaller gain than K=3 at
every concurrency above one).

**Open**

- **Content-adaptive K** — K by recent acceptance for the request, which is what the content table asks for
  at one user (prose and Turkish K=3, code and maths K=7). A draft patch exists and passed 23/23 CPU unit
  tests; it has never run on a GPU `[not tested]`.
- K=4 was not run (+25 minutes) and concurrency 3 was not measured (the table takes the larger neighbour's
  K).
- The weights are ours; a different job mix needs a different table.
- The upstream "skip the draft when K=0" proposal (#53426) is not in the pinned tree.

**Retracted:** none by this work; the pre-registered agreement clause was flawed and is superseded by its ratio form.
