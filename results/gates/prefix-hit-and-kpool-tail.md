# The repeated prefix and the one-row tail — what we measured

**8 September 2026.** Two backports into the pinned vLLM (`487ecf187`), measured against a control
arm that is the same tree with both knobs unset — the patch scripts are applied unconditionally by
the prelude so the control runs the same bytes as the candidate, which is the arrangement
[docs/09](../../docs/09-measurement-protocol.md) §2 asks for and the one the indexer-workspace A/B
used. Patch page:
[`tracks/tp3/patches/prefix-hit-and-kpool-tail/`](../../tracks/tp3/patches/prefix-hit-and-kpool-tail/README.md).

Settings for every number on this page unless stated otherwise: image `exl3-zeus:754421f`,
checkpoint `turboderp/GLM-5.3-Flash-exl3@4.05bpw` (full scope), TP=3 + expert parallel, DFlash2
`k=7` with draft KV at fp8, `gpu-memory-utilization 0.88`, `--block-size 256`,
`HAREM_SW_BLOCK_SIZE=256`, `--max-num-batched-tokens 2048`, `--max-num-seqs 8`,
`NCCL_MAX_NCHANNELS=8`, `max_model_len` 1,000,000, vision on (4 images + 2 videos per request),
CUDA graphs off, `reasoning_effort: low`, temperature 0.

---

## 0. The instrument, and why the ceiling is in every table

Hit ratios come from the engine's own Prometheus counters — `vllm:prefix_cache_hits_total` and
`vllm:prefix_cache_queries_total`, read either side of each request, one request at a time, so every
delta belongs to exactly one request. The API's `cached_tokens` field would need the server started
with `--enable-cache-report`, which production is not.

A request of `n` tokens can never hit all of them: at least one token has to be recomputed. Hits
also land on the coordinator's alignment granularity, which on this stack is the target MLA group's
block, **3,328 tokens** — measured, not assumed: every hit this page reports is an exact multiple of
3,328. So the best achievable ratio is

```
ceiling(n) = floor((n - 1) / 3328) * 3328 / n
```

which is **83 % at 8,000 tokens** and **99.8 % at 60,000**. A raw "83 %" on an 8K prompt is a perfect
score. Every table therefore carries the ceiling beside the ratio, and the acceptance criterion is
the quotient, not the raw number.

---

## 1. What the production configuration was doing, before anything changed

Measured against the untouched production engine (configuration 13, vision on) before the patch tree
existed at all `[measured-here]`:

| Scenario | n tokens | hit | hit tokens | ceiling | of ceiling |
|---|---|---|---|---|---|
| exact repeat | 6,677 | 49.8 % | 3,328 | 99.7 % (2 blocks) | 50.0 % |
| exact repeat | 49,765 | 86.9 % | 43,264 | 93.6 % (14 blocks) | 92.9 % |
| four-turn agent conversation | ~6,7xx | 49.3 % | 3,328 | ~99 % | 50.0 % |

**The loss is exactly one 3,328-token block, on every repeat, in every scenario.** That is what the
flag-all fallback costs, and it is the number the first patch is aimed at.


---

## 2. The A/B: control against both knobs, same tree, same session

Four fast-load boots on the same patch tree, differing only in environment knobs. Every arm's boot
is timed from `docker run` to `/health` 200.

| Arm | Knobs | Boot | KV pool |
|---|---|---|---|
| **off** — control | none | 294 s | 7,046,831 |
| det0 — evidence | `HAREM_KPOOL_TAIL_BOUNDS=1` | 293 s | — |
| det1 — evidence | `HAREM_KPOOL_TAIL_FIX=1 HAREM_KPOOL_TAIL_BOUNDS=1` | 314 s | — |
| **ab** — both knobs | `HAREM_PREFIX_HIT=1 HAREM_KPOOL_TAIL_FIX=1` | 293 s | 6,936,639 |
| ab, second boot | the same | 294 s | — |
| armA | `HAREM_PREFIX_HIT=1` | 315 s | 7,013,774 |
| armB — **what ships** | `HAREM_KPOOL_TAIL_FIX=1` | 294 s | 6,914,600 |

`[measured-here]`

### 2.1 Exact-repeat hit ratio

| Scenario | n tokens | control | both knobs | ceiling | of ceiling |
|---|---|---|---|---|---|
| exact repeat | 8,008 | 41.56 % | **83.12 %** | 83.12 % | **100.0 %** |
| four-turn agent conversation | 8,045 | 41.25 % | **82.51 %** | 82.51 % | **100.0 %** |
| exact repeat | 59,910 | 94.43 % | 94.43 % | 99.99 % | 94.4 % |

`[measured-here]` — mean over the repeats of each scenario; the first repeat and the later ones agree
to within 0.2 points on every arm, so there is no warm-up term.

**Two of the three land exactly on the ceiling.** The 8K repeat goes from one cached block to two,
and the four-turn conversation — which is the shape an agent actually produces, history re-sent every
turn — goes with it. **The 60K repeat does not move**, and §2.2 is why.

### 2.2 TTFT of the follow-up requests

| Scenario | control | both knobs | change |
|---|---|---|---|
| exact repeat, 8,008 tokens | 2.830 s | **1.076 s** | **−62.0 %** |
| four-turn agent conversation | 2.841 s | **1.110 s** | **−60.9 %** |
| exact repeat, 59,910 tokens | 2.165 s | 2.358 s | +8.9 % (the hit did not change; this is one sample either side) |

`[measured-here]` — time to the first streamed token, thinking on at `reasoning_effort: low`.

### 2.3 Speed, median of three sweep rounds

`scripts/hizset-v2.jsonl`, 12 short English code prompts, `--think low`, temperature 0.

| Concurrency | control tok/s | both knobs tok/s | change | band (docs/09 §1.2) |
|---|---|---|---|---|
| C1 | 66.97 | 69.82 | +4.3 % | ±4 % (within-arm spread on the control was 66.7–68.2) |
| C2 | 97.85 | 98.85 | +1.0 % | — |
| C4 | 141.57 | 141.11 | −0.3 % | ±9 % |
| C6 | 175.58 | 172.92 | −1.5 % | — |
| C8 | 192.12 | 193.54 | +0.7 % | ±3 % |
| prefill, fresh unseen ~8.5K | 1,753 | 1,765 | +0.7 % | ±3 % |

Draft acceptance, median per level: control 59.6 / 61.6 / 64.0 / 63.2 / 60.6 %, candidate 62.8 /
61.4 / 63.5 / 62.4 / 60.7 %. Averaged over the five levels: **61.80 % against 62.15 %, +0.35
points** — inside the ±2-point gate, and inside the control's own round-to-round spread at C1 alone
(59.1–61.2). `[measured-here]`

**Nothing is outside its band.** The C1 gain is in the favourable direction and does not survive the
within-arm spread, so this page does not claim it.

### 2.4 KV pool

7,046,831 on the control against 6,936,639 on the both-knobs arm, **−1.56 %**. Neither patch allocates
anything, and the same tree with the same knobs unset produced **6,914,600** on its dump boot, below
both. The pool is decided by what the profile run peaks at, and the boot-to-boot spread on this
configuration now spans 6,914,600–7,143,250 (**3.2 %**) across five boots of the same code. **Read as
unchanged**, and the spread is wider than the 7,024,793–7,126,721 this repository quoted from three
boots on 7 September. `[measured-here]`

### 2.5 Gates

| Gate | control cold | both knobs cold | both knobs after the soak, boot 1 |
|---|---|---|---|
| correctness probe | 10/10 | 10/10 | 10/10 |
| code exam | 12/12 | 12/12 | 12/12 |
| tool-call gate | 8/8 | 8/8 | 8/8 |
| needle-lite, six depths at ~54.7K | 6/6 | 6/6 | **5/6** — see §4.3 |
| vision K2 (four images) + K4 (video) | 4/4 + PASS | 4/4 + PASS | — |

`[measured-here]`. Free host RAM after the whole benchmark: 2.0 / 3.9 / 3.9 GiB on the control and
2.7 / 4.6 / 4.6 GiB on the candidate; swap 0.03 GiB on rank 0 and zero elsewhere on both.

---

## 3. The K-pool tail — what ships

### 3.1 The detector, either side of the knob

`HAREM_KPOOL_TAIL_BOUNDS=1` counts, after the correction and on the mapping the engine actually
uses, how many tail tokens address a block that is not the request's own. Two arms, identical
workload (four concurrent 1,024-token generations plus one of 2,048):

| | fix off (`det0`) | fix on (`det1`) |
|---|---|---|
| steps where `positions` reached the tail builder | **0 / 896** | **896 / 896** |
| tail tokens counted | 15,630 | 16,806 |
| **used mapping: wrong block** | **15,578 — 99.67 %** | **0 — 0.00 %** |
| worst block written | **332** | **275** |
| largest block any request owned | 275 | 275 |
| generic mapping: wrong block | 15,578 | 16,754 (unchanged, and no longer used) |
| row overruns | 0 | 0 |

`[measured-here]`. Two readings matter. **`positions` is None on every single call without the fix**
— that is the whole bug, exactly as reported. And the worst block written without the fix is **332
against a largest-owned block of 275**: those writes are not landing in another request's ring, they
are landing outside every ring any request holds.

**One detail of the published diagnosis does not hold on this build.** vcruz305 describes the tail's
block-table row as one entry wide, so that `pos >= block_size` reads past it. Our runner sizes that
row `cdiv(max_model_len, block_size)` = **250,016 entries**, not from
`KpoolTailSpec.max_num_blocks_per_req()` (which returns 1) — so there is **no out-of-row read at
all**, and the detector says so: zero row overruns while the mapping is wrong for 99.67 % of tokens.
The harm is the wrong block, not the overrun. This is the second independent reason a clamp is the
wrong layer; the first is his own measurement that the clamp changed nothing.

### 3.2 Model-free, before any of that

Inside the serving image, CPU only, no model: `kpool-tail-unit-test.py` **4/4** `[measured-here]`.
It pins `block_table[req, 0] * kpool + pos % kpool` for a decode batch and for a 600-token prefill
batch whose positions cross the row width, and it shows the generic mapping putting three requests
that own blocks 10, 11 and 12 all onto block **0**. `annotate-unit-test.py` **3/3**: the drafter's
group is flagged, the target's is not, and both fail-closed refusals fire.

---

## 4. The prefix-cache half — measured, and not in the recipe

### 4.1 What it bought

Sections 2.1 and 2.2 are the numbers: an 8,008-token exact repeat goes from **41.56 %** to
**83.12 %** of its tokens served from cache — from one cached 3,328-token block to two, which is
**100.0 % of the ceiling** — and the follow-up TTFT falls from **2.830 s to 1.076 s, −62.0 %**. The
four-turn agent conversation, which is the shape that actually matters, goes the same way: 41.25 % to
82.51 %, TTFT 2.841 s to 1.110 s. Nothing else moved: every concurrency level inside its band, draft
acceptance +0.35 points averaged over five levels, gates full, KV pool inside the boot-to-boot
spread. `[measured-here]`

### 4.2 Why the 60K repeat did not move — the offset decides it

A request of `n` tokens sits `n mod 3328` tokens past its last aligned boundary. The drafter's
sliding-window group still takes the EAGLE drop, gives back one of its own 256-token blocks, and then
re-aligns to the 3,328-token granularity — so it needs a whole 256-block *past* the boundary to give
back. The 60K prompt's offset is **6 tokens**. Measured on the control at four offsets, holding
everything else fixed:

| n tokens | offset past the boundary | hit | ceiling | blocks hit / ceiling |
|---|---|---|---|---|
| 55,189 | 1,941 | 90.45 % | 96.48 % | 15 / 16 |
| 55,862 | 2,614 | 89.36 % | 95.32 % | 15 / 16 |
| 56,623 | 47 | 94.04 % | 99.92 % | 16 / 17 |
| 57,379 | 803 | 92.80 % | 98.60 % | 16 / 17 |

`[measured-here]` — the control is **exactly one block short of the ceiling at every offset**, which
is the flag-all fallback and is offset-independent. The residual after the patch is the drafter's own
drop, which *is* offset-dependent: it is free when the offset exceeds 256 tokens and costs a whole
3,328-token block when it does not. About **7.7 %** of prompt lengths fall in that window. The
patched arm was not re-run across the four offsets `[not tested]` — the sweep above establishes the
control's behaviour and the mechanism; the patched sweep is the obvious next measurement.

### 4.3 The failure that stopped it, and the reproduction that did not

| Arm | knobs | needle-lite runs on that boot |
|---|---|---|
| control | none | 6/6 · 6/6 · 6/6 (cold, warm, warm) |
| armA | `HAREM_PREFIX_HIT=1` | 6/6 · 6/6 · 6/6 |
| armB | `HAREM_KPOOL_TAIL_FIX=1` | 6/6 · 6/6 · 6/6 · 6/6 concurrent |
| **ab, boot 1** | both | 6/6 cold · **5/6** after a 48,483-token soak · **5/6** |
| ab, boot 2 | both | 6/6 · 6/6 · 6/6 · 6/6 concurrent · 6/6 after a 48,910-token soak · 6/6 |

`[measured-here]`. The failing needle is the earliest of six in a 54,694-token haystack; the model
returned a plausible invented code, and a *different* invented code on the second failure. **Nine
runs of the both-knobs configuration: seven pass, two fail, both inside one boot.**

**Part of why this is unsettled is a design error of ours.** The failing arm was the only one in the
whole session that ran needle-lite *after* a soak. The control's needle runs had no soak in front of
them, so the comparison that raised the alarm was not like for like. The reproduction attempt put an
identical soak in front of the gate on the second boot and came back 6/6 twice.

We are not calling it noise and we are not calling it a bug. It is one boot, and
[docs/09](../../docs/09-measurement-protocol.md) §1 says a single boot settles nothing — which
applies to a failure as much as to a gain. The patch stays in the tree with its knob unset.

---

## 5. The K-pool-only arm, and the gate flake that ended the night

### 5.1 The K-pool half on its own boot

| Measure | control | K-pool only | change |
|---|---|---|---|
| C1 tok/s | 66.97 | 69.34 | +3.5 % |
| C2 | 97.85 | 99.53 | +1.7 % |
| C4 | 141.57 | 143.26 | +1.2 % |
| C6 | 175.58 | 170.68 | −2.8 % |
| C8 | 192.12 | 191.59 | −0.3 % |
| prefill, fresh | 1,753 | 1,757 | +0.2 % |
| draft acceptance, five-level average | 61.80 % | 62.62 % | +0.8 pt |
| KV pool | 7,046,831 | 6,914,600 | inside the spread |
| exact-repeat hit, 8,008 tokens | 41.56 % | **41.56 %** | **identical** |
| cold gates | 10/10 · 12/12 · 8/8 · 6/6 | 10/10 · 12/12 · 8/8 · 6/6 | — |
| vision K2 + K4 | pass | 5/5 pass | — |
| soak | — | 49,152 tokens, engine alive, all coherent | — |
| needle after the soak, sequential and concurrent | — | 6/6 and 6/6 | — |
| **battery after the soak** | **clean** | **code exam 11/12** (`matrix`) | see §5.2 |

`[measured-here]`. The identical hit ratio is the check that matters here: the K-pool half touches
nothing the scheduler does, and the number says so.

### 5.2 The gate that judged them has a flake rate, and nobody had measured it

The rollback was decided on this: three post-soak batteries on patched arms each lost one item, and
the control's post-soak battery was clean. Twenty minutes later, with production restored and
**nothing patched at all**, the first code exam on the restored engine came back **11/12 —
`matrix`, the same `AssertionError`**. Four more runs on the same engine: **12/12, 12/12, 12/12,
12/12**. `[measured-here]`

So `matrix` failed **2 of the 12 code exams** run in this session, one of them on a configuration
with no patch tree in it. The K-pool arm's 11/12 is a flaky item, not evidence, and it is withdrawn
as evidence here.

What survives is narrower: **one boot** of the both-knobs arm returning needle 5/6 twice, against six
later runs of that configuration returning 6/6 — including concurrently and after an identical soak
— and against 6/6 on the control, the prefix-hit-only arm and the K-pool-only arm, three runs each.

**Nothing was promoted.** `.env.tp3` was verified byte-identical to its dated backup on all three
nodes; the production tree and its sidecar were never touched. Restoring production was `systemctl
start`: `/health` 200 at **255 s**, KV pool **7,066,115**, probe 10/10, tool-call 8/8, needle 6/6,
vision 5/5, and the code exam 11/12 on its first run and 12/12 on the four after it.

**The lesson, and it is the useful one:** we spent a night measuring two patches against gates whose
own flake rate we had never measured. Two accidental data points on that flake rate were enough to
overturn half of a decision. A gate baseline — every gate, ten runs, cold and post-soak, on the
production configuration — is now the cheapest useful measurement on this stack's list.
