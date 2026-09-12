# 09 — Measurement protocol

**Applies to: both tracks.** If you read one page of this repository, read this one.

Four different ways to measure a lie on this stack, and the protocol that stops each one. Every
retraction in [11](11-open-issues.md) is a case of one of these, and every one of them was published
before it was caught.

If you take one thing from this repository, take this page rather than any number on it.

---

## 1. Sweep rounds: five and discard two, or three if the tuner cache is warm

**The trap.** The MLA decode tuner keys on the batch shape and re-tunes when it sees a new one, at
about 15 ms per event, and every event is marked evicted. A boot mints 11 events before the server is
even up, and more as the sweep presents new shapes. **The warm-up window is sometimes longer than two
sweep rounds** — long enough to put a whole benchmark arm inside it `[measured-here]`.

**What that looked like.** An arm we were testing measured prefill 1,373 → 698 tok/s, prefill-fresh
1,672 → 936, C8 140 → 105. It looked like a disaster. Re-measured on the *same running engine* a few
minutes later, with nothing changed but the clock: prefill 1,483–1,515, prefill-fresh 1,720, C8
152.6 / 150.3 / 148.6. That arm was the winner by 12.6 %.

**The protocol.** Five `bench-sweep` rounds per arm. **Discard rounds 1 and 2.** Report the median of
rounds 3–5, and report the spread.

```
python3 scripts/bench-sweep.py --out sweep-1.json --label arm-round1 --think low
```

Repeat five times, then take medians. Rounds 3–5 of a settled engine are tight: on the 8-channel
configuration, C8 over five rounds read 151.7 / 150.1 / 152.4 / 150.8 / 147.0 `[measured-here]`.

**The shorter protocol, and what earns it.** Upstream now persists the tuner cache across processes
(`CUDA_EXL3_TUNE_CACHE`, [12](12-tuner-cache.md)). With it warm, a boot mints **zero** tune events
before serving and **zero** during a full sweep, and round 1 stops being a penalty: cold cache, C8
round 1 → round 3 is −3.4 %; warm cache it is +2.7 %, i.e. unordered noise `[measured-here]`. On an
image that has the cache **and a warm cache file on disk**, the protocol drops to **three rounds,
median of three** — about 15 minutes saved per arm.

Three conditions, all of them, or go back to five:

1. the image carries the persisted cache and `CUDA_EXL3_TUNE_CACHE` points at a directory that
   survives the container;
2. the cache file already exists from a previous boot of **this** image — the boot that writes it is
   still a five-round boot;
3. the three rounds agree within about 5 %. If they do not, that is a signal, not a number.

Every arm in this repository up to *fast boot S4* is a five-round median with two discarded; the ones
after it are three-round medians. The tables say which, and the two are not interchangeable.

### 1.1 Our own harness disagreed with its own documentation

The quick-arm script carried the header "5 rounds, the first 2 discarded" and its body ran **three**.
Applied literally, the rule would have left a median of one `[retracted]`. Nothing published here was
computed that way — the three-round arms were reported as medians of three, which is what they are —
but the discrepancy stood in the script for a day and is exactly the class of thing that quietly
turns into a wrong number.

Two things were checked before fixing it in the direction of the code rather than the header. First,
**the warm-up ramp is gone on a warm cache**: across the three settled arms of 5 September, round 1
was the *fastest* round twice. There is no ordered penalty left to discard. Second, the round-to-round
spread inside an arm is what §1.2 measures, and it is larger than most of the differences we have been
calling results.

**The quick tier is now one warm-up round plus three measured rounds, median of the three** — four
rounds in total, ~4 minutes more than the old three and the only version in which the "discard the
warm-up" sentence is true.

### 1.2 The noise band, per metric, and the rule that follows from it

Round-to-round spread within a single settled arm, three arms, warm tuner cache, nothing changed
between rounds `[measured-here]`:

| arm (rounds) | C1 | C2 | C4 | C6 | C8 |
|---|---|---|---|---|---|
| production 6 (3) | 57.24 / 56.85 / 56.10 → **1.9 %** | 84.43 / 82.95 / 84.17 → 1.7 % | 114.94 / 125.07 / 118.51 → **8.5 %** | 142.78 / 146.44 / 142.91 → 2.6 % | 167.31 / 170.22 / 168.95 → 1.7 % |
| mesh arm (3) | 56.15 / 56.35 / 58.51 → 4.2 % | 79.40 / 84.44 / 83.77 → 6.0 % | 123.46 / 125.11 / 119.32 → 4.7 % | 151.42 / 143.27 / 143.30 → 5.7 % | 175.03 / 171.07 / 170.45 → 2.7 % |
| draft page 256 (5) | 54.62…52.34 → 4.4 % | — | — | — | 167.83…162.00 → 3.6 % |

**The declared bands: C1 ±4 %, C2 ±6 %, C4 ±9 %, C6 ±6 %, C8 ±3 %.** C4 is the noisiest column on
this stack and C8 the quietest, which is the opposite of the intuition that more streams means more
variance.

Two rules follow, and they apply to every table in this repository:

- **A difference of 3 % or less is written down as "equal".** Report the numbers, do not report a
  winner. This is a floor, not the whole test.
- **Above the floor, a difference still has to clear its own metric's band** before it is called a
  result — and if the decision matters, §2 applies as well: repeat it on a second boot.

Judgements already on record that this rule reclassifies as "equal": the combine-staging arm's
+2.3 % at C4, and patch 0007's −0.9…+4.2 %. One survives it: production 5 → 6 is +4.4 % at C1
(inside C1's band) but **+5.6 % at C8**, which clears C8's ±3 %, so that conclusion stands.

None of this replaces §2. The bands above are *within one boot*; boot-to-boot is 15.9 %.

## 2. Boot-to-boot variance is 16 %, so two rounds decide nothing

**The trap.** The same image, the same environment file, two separate boots `[measured-here]`:

| | boot 1 round 1 | boot 1 round 2 | boot 2 round 1 |
|---|---|---|---|
| C1 | 47.71 | 48.33 | 49.62 |
| C2 | 63.68 | 71.67 | 70.60 |
| C4 | 90.37 | 100.66 | 96.83 |
| C6 | 110.14 | 115.68 | 117.13 |
| C8 | **118.44** | **137.23** | **135.56** |

**15.9 % spread on C8 with nothing changed at all.** We once published a 10 % gap between two kernel
builds as an unexplained regression, and drew a kernel conclusion from it. It was this
`[retracted]`.

**The protocol.** A difference under about 5 % on this stack is not a result. Where a decision hangs
on one, take the means over five rounds **and** repeat on a second boot before writing it down.

## 3. Prefill measured on a repeated prompt is not a prefill measurement

**The trap.** `scripts/prefill-7k.py` uses two fixed seeds. Run it twice inside one boot and the
second run reads two whole 3,328-token blocks out of the prefix cache and reports up to **1,596
tok/s where the honest number is 1,025** `[measured-here]`.

**The protocol.** For any prefill claim, use `bench/prefill-fresh.py`, which draws a new seed per
request so the engine has never seen the prompt. Report the median of at least three, and say which
one you used. Both numbers appear in this repository and they are labelled: "prefill 7k" is the warm
repeated-prompt figure, "prefill-fresh" is the honest one.

```
python3 bench/prefill-fresh.py
```

## 4. Model-free first — the engine is the last place to look

Every conclusion in this repository that survived was reached with the engine down, on a
micro-benchmark using the real GLM-5.3-Flash shapes. Every conclusion drawn from two engine sweeps
alone has since been retracted.

The arithmetic is blunt: an engine boot on this stack takes about 11 minutes, and a full A/B arm
about 30. The model-free work that produced the two strongest results in this repository — rejecting
a protocol setting outright, and the per-kernel comparison that retired one of our own patches —
cost **five minutes between them** `[measured-here]`.

The harness is in `bench/`; `bench/validate.sh <image-tag>` runs the whole battery.

### 4.1 When the engine is the only place to look: profile it, and verify the profiler

Some questions have no model-free form. "Where does a step go" is one: it needs real routing, real
collectives and the real scheduler, and every attempt to answer it by carrying model-free ratios onto
a wall-clock total produced numbers that a later trace corrected — two of them by more than 10×
([10](10-results-and-roofline.md) §5). Four things make that measurement cheap and correct.

**1. Set the flag before you need it.** This vLLM takes the profiler as `--profiler-config`, **not**
as `VLLM_TORCH_PROFILER_DIR` (`vllm/config/profiler.py`). With the environment variable alone the
`/start_profile` route is never attached and the endpoint answers **404** — and since the engine
cannot be reconfigured without a boot, that 404 is what postponed this stack's step-time breakdown by
a week. `scripts/start-tp3.sh` now carries the arm, off unless `PROFILER_DIR` is set:

```bash
PROFILER_DIR=/cache/prof ./start-tp3.sh 0
# then, against the running server, with no restart:
curl -X POST http://<head>:8001/start_profile      # -> 200
#   ... drive the window: a fresh long prompt, or N seconds of steady decode ...
curl -X POST http://<head>:8001/stop_profile       # -> 200
```

Each rank drops its own `*.pt.trace.json.gz` plus a `profiler_out_0.txt` under that directory. An
unset `PROFILER_DIR` costs nothing at run time, so there is no reason for a production boot not to
carry it. `torch_profiler_with_stack: false` keeps the traces to tens of MB and costs only the Python
frame names, which the `cpu_op` names (`vllm::all_reduce`, `vllm::moe_forward_shared`,
`Torch-Compiled Region`, `aten::pin_memory`) largely replace.

**2. Measure the profiler's own cost, in the same windows.** Run each window twice, once with the
profiler off. On this stack the overhead is **0 % on prefill, +2.5 % on C1, +1.3 % on C8**, which is
what licenses carrying the *shares* onto a profiler-free engine.

**3. Subtract CUPTI before you believe an idle number.** The overhead is not spread evenly — it lands
on kernel boundaries, at roughly **1 µs per boundary**. A decode step here launches ~2,300 kernels
across ~1,873 gaps, so the tracer alone manufactures ~2.0 ms of apparent GPU idle. Uncorrected, a
3.75 % idle budget reads as 5.8 %, and a target that is worth +1.5–2 % reads as worth +6 %. **Take
`busy(union)` from the trace, take the wall from the profiler-off run, and let the difference be the
idle** — do not read the idle out of the trace directly. Both of this repository's published
retractions about GPU idle come from skipping that step ([10](10-results-and-roofline.md) §5.8).

**4. Two trace-reading traps, both of which produce confident nonsense.** In decode the per-step
`gpu_user_annotation` arrives as an **overlapping pair** — merge them, or your step count doubles and
you will report a 50 % GPU bubble. And when matching a gap to the launch that follows it, include
**`cuda_driver`** launches, not only `cuda_runtime`: triton and inductor kernels are launched through
the driver API, and filtering them mislabels the two largest gaps in a step as device-side waits.

A useful accident on this stack: the drafter runs **outside** the step annotation, so the annotation
boundary is an exact target-versus-draft split, free of charge. That is what corrected the k=7
drafter's cost from 19.5 % of a C1 step to 11.4 %.

**What it costs.** About six minutes of GPU time for a baseline plus three windows, ~120 MB of trace
per node, and — measured, and unreturned until the container restarts — **7–8 GiB of host RAM per
node**, most plausibly kineto's activity buffers inside the worker process. On a stack whose rule is
never to go below 4 GiB free, plan the profile run when the host has room, and watch the free-RAM
line afterwards `[measured-here]`.

---

### 4.2 A working set that fits the cache flatters every synthetic bench, the same way three times

Three model-free benches in this recipe looked better than the engine for one reason: the synthetic
working set was small enough to be served from cache, and production's is not.

| bench | what fit | what it hid | how it was caught |
|---|---|---|---|
| MoE GEMM with uniform-random routing | eight experts' blocks per step stayed in L2 | the real router concentrates on hot experts and streams the rest | the engine profile disagreed with the bench at small M |
| MLA attention on a 200 k-row KV pool | the whole pool | a 5 M-token pool is a cold read every step | the pool-size sweep bent where L2 ran out |
| Dense/KDA GEMM timed on one weight tensor | a 0.7–33 MB weight in a 24–128 MB L2 | the trellis exists to save bytes that a resident weight never pays for | the ruler read 210 % of the card's peak bandwidth; the `cuda-exl3` author later caught the same artefact in his own table and withdrew a claim |

The rule: **before trusting a model-free ratio, size the working set against L2 and rotate over a
bank at least three times larger** (`bench/kda_gate_bench.py` and `bench/ruler_check.py` do this),
and read the achieved bandwidth — a number above the machine's measured peak is the cache talking,
not the kernel. A warm ratio is not wrong, it is a different question: it answers "what if the weight
were resident", which at decode it is not.

**And banked on the wrong card is the same as not banked at all** — row 3 above is in this table
twice, because the bank that caught it was sized for the large shapes on a 101 MB-L2 workstation and
left the 0.72 MB arm the conclusion was actually about resident from first replay to last, producing a
ratio that reversed when it was re-taken cold on the 24 MiB-L2 part production runs
([11](11-open-issues.md) §1.11). Three consequences, all cheap: size the bank **per shape** and
against **the card you will serve on**; rotate **both** arms over it, since a 4-bit bank is a quarter
of a bf16 one and the artefact's *sign* depends on which arm fits, so it does not cancel in a ratio;
and where a production trace already holds the same call, use it as the referee — cold reproduced ours
to ±20 %, warm was out by 2.2–2.4×.

## 5. The gates come before the numbers

Two cheap tests, run **cold** after every boot and again **after** the full benchmark. Nothing else
in this repository means anything without them.

```
python3 scripts/correctness-probe.py http://192.0.2.10:8001
```

```
python3 scripts/code-exam.py http://192.0.2.10:8001
```

Expected: correctness probe **10/10** (client-visible 9/9, empty content 0) and code exam
**12/12 PASS, 0 FAIL**.

**The warm run is the one that carries weight**, and this is not a formality. The class of defect
this stack has actually produced — a kernel writing rows nothing initialises, and a combine summing
them — *hides on a fresh engine*, because a fresh caching allocator hands out zeroed pages. After a
full benchmark the allocator is recycling used memory, so the retired rows would hold real garbage.
Full marks in that state is direct evidence, not a formality `[measured-here]`.

**Two counts are reported on purpose.** With thinking on, this model sometimes puts the whole answer
in the reasoning field and leaves the content field empty — same prompt, same settings, temperature
0. So the probe reports `content` (what a plain client sees) and `both` (what the model actually
knows). A drop in the first with the second intact is a chat-template or parser problem, not a model
problem.

### 5.1 The acceptance list a production change actually gets, in the order it is run

The two gates above are the floor, not the list. Below is the one production configuration 9 was
promoted on ([13](13-full-scope-checkpoint.md) §7), written out because the **order** is the part
that transfers: everything model-free comes before the boot, every gate comes before a speed number,
the pool is read from a load boot, and the long benchmark is last because it is the only thing that
costs half an hour. Nineteen points were written down before the arm and all nineteen were taken.

| # | test | gate | why it is where it is |
|---|---|---|---|
| 1 | both patch scripts in `--check` mode, engine **down** | every anchor matches **exactly once** | costs seconds; a drifted anchor found here is not found four minutes into a boot |
| 2 | **meta-device name-set dump, all three ranks** | 0 unmapped / 0 unfilled on **every** rank; the EXL3 and BF16 module sets match the list predicted beforehand | §11.3. Rank 2 is mandatory at TP=3: the pad lives there |
| 3 | memory arithmetic, model-free | settled before spending a boot | the item that had forced the TP=2 arm down to a 31k pool |
| 4 | preflight on the padded sidecar | vocab 155,136 → 51,712/rank (404 × 128); shared expert 2,304 → 768 (6 × 128); MLA and KDA 66 → 22; expert parallel mandatory | shape arithmetic needs no GPU and no weights |
| 5 | **image capability gate**, in the prelude | `[padload] ... =yes` on all three capabilities, or **exit 23** | it runs *before the weights*: both failure modes leave a half-loaded stack |
| 6 | patch anchors, at boot | `anchors 1/1: A1 ... A10`, plus the patch `sha256` in the log | says which code is running, not which code is on disk |
| 7–10 | asserts 1–4 | **silent** | shard order, `suh` collapse, `conv1d` axis, mapping/shard-id agreement |
| 11 | **assert 5, the pad audit** | `285 EXL3 pad site(s) audited, 285 padded on this rank, all whole 128-blocks and exactly zero` | the count is one an independent model-free run predicted; **absence of the line is a failure** |
| 12 | `CUDA_EXL3_DEBUG_NAMES` tally | **203 EXL3 / 113 bf16**, and nothing heavy in the `-> unquantized` list | read **negatively**: the gate is which modules are *absent* from the miss list |
| 13 | KV pool, from the boot log | reported; `≥ 3.4 M` expected, below 3 M stop | free with the boot, and on an arm that moves a page size it **is** the result |
| 14 | free memory and swap | ≥ 4 GiB free on every node, swap flat | |
| 15–16 | correctness probe, code exam | **10/10 and 12/12, cold** | before any speed number, always |
| 17 | speed, **3 rounds, median** | C1 ≥ the control's, C8 ≥ the control's, prefill-fresh ≥ the control's | §1; three rounds only because the tuner cache is warm |
| 18 | draft acceptance | ≥ 60 % | a speed win paid for out of acceptance is not a win |
| 19 | MMLU sample (57 × 35, 0-shot) | inside the control's error bar | ~26 min; last, because it is the expensive one |
| 20 | gates again, **warm** | 10/10 · 12/12 after the full benchmark | the run that carries weight — see above |

Four properties of that list are worth stealing even if none of the specific tests apply to you.

**Half of it costs no GPU.** Points 1–4 run in a throwaway CPU container beside an idle engine and
between them catch every failure that loads cleanly and computes the wrong thing (§9, §11.3).

**Two of the boot gates are read negatively.** Point 12's question is not "did anything print" but
"is the module I care about *absent* from the miss list", and point 11's is "did the audit line
appear at all". A gate whose passing state is silence has to be checked for presence, because silence
is also what a gate that never ran looks like — which is exactly how the same diagnostic failed on
the TP=2 arm ([13](13-full-scope-checkpoint.md) §6.3).

**Every count in it was predicted before it was measured.** 285 pad sites, 203 EXL3 linears, 113 BF16,
404 × 128 per rank: each came out of a model-free derivation first, so agreeing with the log is
evidence rather than an observation about the log.

**The list ends where it began.** Point 20 is point 15 run again on a used allocator, and point 13 is
only trustworthy because point 3 settled the arithmetic first.

---

## 6. Realistic, synthetic, and fresh

Three kinds of speed number, never mixed in one table.

| Kind | What it is | What it is for |
|---|---|---|
| **Realistic** | `scripts/hizset-v2.jsonl` — 12 short English code prompts, and the four category sets (prose / code / math / JSON) | Everything we publish. This is what a user experiences. |
| **Synthetic** | "count from 1 to 200" and friends | The speculative-decoding **ceiling**. Always labelled. It will disappoint you in real use. |
| **Fresh** | prompts the engine has never seen | Prefill only. See §3. |

The gap is not small. On this model family a synthetic counting prompt runs at roughly 1.7× the
realistic single-stream rate, because the draft model predicts a counting sequence almost perfectly
and predicts real code about half the time. Publishing a synthetic number without the label is the
single easiest way to mislead someone about this hardware.

**Prompt language is a lever too.** On our NVFP4 sibling stack, the same task asked in English ran at
54–63 tok/s with 57–69 % draft acceptance and in another language at 41–47 tok/s with 41–48 %
`[measured-here]`. The engine is identical; the draft model simply predicts English far better. Every
speed table in this repository was measured with English prompts, so treat them as English-workload
numbers.

---

## 7. All benchmarks are at reasoning effort `low`

Temperature 0, `enable_thinking` true, `reasoning_effort: low`, unless a table says otherwise.

We did not run anything at max effort. On this model the low-to-max token ratio is large enough that
a full benchmark suite would take days of cluster time, and we spent that time on the stack instead.
So: **nothing in this repository is a max-effort number, and none of it should be quoted as one.**
What max effort would change is an `[estimate]` wherever we make one, and it is marked.

There is no way to turn thinking off on this model. `enable_thinking: false` does not do what its
name suggests here; the effort levels are the control surface.

Two template facts learned from the field on 10 September 2026 ([docs/14](14-troubleshooting.md)
§9.11–9.12): `enable_thinking` is not read by this template at all — the launchers sent it out of
habit and no longer do — and `clear_thinking` **is** read, defaulting to *retain* prior turns'
reasoning in the 27 August and later upstream templates. Both launchers now default it to `true`.
Nothing in this repository changes: every measurement here is single-turn, where the flag is a
no-op. Any multi-turn measurement must state which way it was set.

---

## 8. What a complete measurement report looks like

Every number we publish carries: the image tag (which kernel commit), TP and EP, quantization, KV
dtype, draft method and `k`, `gpu-memory-utilization`, `--block-size`, `HAREM_SW_BLOCK_SIZE`,
`--max-num-batched-tokens`, `--max-num-seqs`, `NCCL_MAX_NCHANNELS`, temperature, reasoning effort,
`max_tokens`, concurrency, prompt type, the number of sweep rounds and which were discarded, and the
date. If a table does not carry those in its caption or a settings block, it is incomplete.

And one line that is easy to leave out and must not be: **what the gain cost.** Speed, quality and
memory together. If a change genuinely cost nothing, say that it was looked for and not found, and
give the numbers that show it.

---

---

## 9. Three tiers of test, and picking the cheapest one that can answer

A boot on this stack costs about 11 minutes and a full A/B arm about half an hour, so the question
before every measurement is which of these can actually settle it. Working down from the cheapest:

| Tier | What it runs | Cost | Answers |
|---|---|---|---|
| **A — model-free** | `bench/`: the real shapes, engine **down**. `bench/validate.sh <image>`, `bench/ar_bench.py`, `bench/mesh-multilink-sweep.sh`, `bench/moe_stage_bench.py`, `bench/topk_bench.py`, `bench/mhc_bench.py`, `bench/zerokv_bench.py`, and always `bench/bw.py` + `bench/gemmpeak.py` in the same run | seconds to ~45 min, no boot | kernel and collective questions; rejecting a setting outright; anything with a mechanism |
| **B — quick arm** | one boot → gates cold → `soguk-c1` → `prefill-7k` → **`prefill-fresh`** → **1 warm-up + 3 measured** C1–C8 rounds → gates warm → free RAM and swap | ~17–21 min | does a model-free win survive the engine, and did it cost anything |
| **C — full arm** | tier B with **5** measured rounds, plus `category-speed.py` and `mixed-load-probe.py` | ~30–40 min | a configuration that is going into production, or a claim about content types or latency under load |

The tier B row is a measured cost, not an estimate: two arms on 5 September ran 07:38→07:55 and
09:20→09:36, of which the boot is about 5 minutes `[measured-here]`. Its KV pool line comes free —
`GPU KV cache size` is in the boot log — and on any arm that changes a page size or a batch budget
**that line is the result**, not a footnote.

A fourth thing, which is not a tier and is cheaper than all of them: **the live server's own
metrics**. `bench/live-step.py` and `bench/live-decode.py` get a prefill ladder, a chunk-boundary
probe and an exact ms-per-engine-step out of a *running* engine without restarting it, because
`vllm:iteration_tokens_total_count` and `vllm:spec_decode_num_drafts_total` make the step count exact
rather than inferred. When the question is "how long is a step", that is the whole measurement.

**Tier A first, always.** Every conclusion in this repository that survived was reached with the
engine down; every conclusion drawn from two engine sweeps alone has since been retracted (§4). The
protocol setting we rejected without spending a boot, and the per-kernel comparison that retired one
of our own patches, cost five minutes between them.

Tier A is not only for kernels. **A change to the weight loader has its own tier-A test** — a
meta-device instantiation of the real model class, walked through the real `load_weights` against the
checkpoint's index, counting unmapped and unfilled tensors. It needs no GPU, no weights and no boot,
and it is the only cheap test that catches the failure that loads cleanly and computes the wrong
thing. §11.3.

**Tier B is the workhorse, and it has a gap you must state.** It does not run the category probe or
the mixed-load probe, so an arm measured at tier B has **no** prose/code/math/JSON numbers and **no**
mixed-load numbers. Say `[not tested]` rather than carrying the previous arm's figures forward
silently — the results tables in this repository do exactly that for the last two arms.

**Tier C before anything becomes production**, and before any claim that a change moved the balance
between content types or the behaviour under a long prompt.

Two rules that cut across all three. Every tier ends with the **gates cold and warm** (§5) — a speed
number without them is not a result. And every tier reports **what the gain cost**: speed, quality
and memory together, with the KV pool line filled in, because several changes on this stack have paid
for throughput out of the pool without anyone noticing for a boot or two.

### 9.1 A fourth, narrower tier: copy fidelity at long context

None of the three tiers above catches the defect in [docs/14](14-troubleshooting.md) §9.15 — it does
not show at all until tens of thousands of tokens of agentic history have built up, which is past
what a short, repeatable prompt or a five-round sweep ever reaches. It needs its own instrument:
[`scripts/longctx-copy-fidelity.py`](../scripts/longctx-copy-fidelity.py) builds a seeded, fully
fictional agent transcript (a fake project with near-duplicate absolute paths, read/write/search/
terminal tool calls and their results) up to a target prompt size, then runs real turns that each
require copying one exact path out of that history without it being restated, and classifies every
path-like tool-call argument as an exact copy, an edit-distance slip, a splice, a decoy, or
unrelated.

Its first version is a scaffold, not yet a yardstick: at `index_topk` 2048 it read 0/30 where the
private replay reads 11/30 ([results/gates/index-topk-8192-12sep.md](../results/gates/index-topk-8192-12sep.md)
§10), so the fixture still lacks whatever makes real agent context hard to copy from — most likely the
volume of code and logs between the mentions of a path. Treat it the way §6 asks: read a trend across configurations, never an absolute rate. The corpus is
still a stress fixture — near-duplicate paths at a far higher density than an organic session would
produce — but it is generated clean on every run, so it does not carry the self-imitation effect §2
of [results/gates/index-topk-8192-12sep.md](../results/gates/index-topk-8192-12sep.md) documents for
the original session replay (the model copying its own earlier mistakes); that should make this
instrument's absolute numbers milder than the original's, not worse. A 30-turn, 35k-token run still
costs tens of minutes rather than seconds, because the first scored turn cannot fire until the
fixture has built a cold boot's worth of context inside the run itself.

## 10. One measurement at a time: the GPU and the fabric are a lock

This is a protocol rule, not a courtesy. On a three-node cluster with one engine, a model-free
micro-benchmark, a fabric sweep and a serving benchmark all want the same hardware, and any two of
them running at once produce numbers that are quietly wrong: a throwaway container next to the live
engine competes for memory bandwidth and SM occupancy, and a three-node NCCL sweep next to a live
engine competes for RDMA queue-pair resources — which does not slow the engine down, it **fails** its
next collective.

So: **one measurement holds the cluster at a time**, and it says so in a file.

```
echo "what, and when" > ~/exl3-zeus/ENGINE-BUSY
```

The rule around that file is four lines:

1. **Write it before the first `docker run` that touches the GPU or the fabric**, with what you are
   doing and the timestamp. Remove it after the last one — not after the analysis, after the last
   command that could contend.
2. **Check it first, and wait rather than interleave.** If it is held, your measurement is not
   cheaper for starting now; it is worthless.
3. **A model-free container beside an idle engine is allowed and is not free.** Everything measured
   this way in this repository ran in a throwaway container (`--rm`) with a memory cap
   (`--memory=8g`), on a **cpuset disjoint from the engine's**, and with its own JIT cache
   directories so it could not warm or poison the engine's. The engine is pinned to `CPUSET=5-9,15-19`
   ([tracks/tp3/env.tp3.example](../tracks/tp3/env.tp3.example)), so the bench gets `--cpuset-cpus 0-4,10-14` and
   the two do not share a core. Write both sets down in the report. This is only defensible while the
   engine is idle, and it is still not free: the GPU and its memory bandwidth are shared no matter how
   the cores are split, which is why §10 exists at all.
4. **Three-node NCCL work needs the engine down, not idle** `[measured-here]`. This is the one hard
   line: a fabric sweep beside a running engine can exhaust queue-pair resources and take the engine's
   next collective with it. Every mesh number on this stack was taken with all three containers
   stopped.

And one habit that has repeatedly paid for itself: **name every command that ever touched the live
engine**, in the report, including the harmless ones. One measurement in this repository read the
engine container's cpuset with `docker inspect` while checking for a CPU collision; that is a
configuration read with no effect on a running container, and it is written down anyway. The value is
not the confession, it is that the next person debugging an odd number can rule the engine in or out
without guessing.

## 11. Changing a patch changes the sidecar: budget the dump boot into the arm

This is a planning rule, and it has cost us an hour of downtime once already.

The fast-load sidecar's identity hash covers **every `patch-*.py` in the patch directory and the
prelude script**, not just the patches that touch a weight ([08](08-fast-boot.md) §4). So any arm
that adds, edits or reorders a patch — including a patch that only writes a log line, only changes a
KV page size, or is gated off by an environment variable it does not set — **invalidates the sidecar,
and the preflight refuses the boot**. That is the design working. It is also a 682-second dump boot
on every node before the arm can start `[measured-here]`.

Three consequences for how an arm is planned:

1. **Read the arm's patch list before you cost it.** A "one boot, ~17 minutes" quick arm that touches
   the patch directory is a **dump boot plus a load boot**, closer to 45 minutes, and the dump boot's
   own numbers are not usable: its KV pool reads low because 56 GiB per node goes out through the
   page cache, and its weights+non-torch ledger is not the production one either. **Never record a
   dump boot's pool as a result** ([08](08-fast-boot.md) §5).
2. **A dump boot is still a real arm for everything that is not memory.** Gates, draft acceptance,
   tok/s and TTFT are all valid on it — the draft-fp8 arm was validated exactly that way
   ([07](07-kv-and-draft-page.md) §7, [11](11-open-issues.md) §2.18) — so if the question is
   "is this safe", the dump boot answers it and the load boot only supplies the pool number.
3. **Do not touch the patch directory while a boot is in progress.** It is mounted live into the
   container, so an edit mid-boot changes the identity underneath the running dump. This has happened
   once and it cost the boot.

The narrower gate this argues for — hashing only the patches that can affect a weight, and hashing
the prelude's ordered list of calls rather than its full text — is written up as an open item in
[11](11-open-issues.md) §2.21. Until it exists, the rule is the one above: budget the dump boot, or
do not touch the patches.

### 11.2 Between a dump and a load, the patch directory is frozen — including additions

§11 is about *editing* a patch. This is the case that caught us, and it is narrower and easier to
walk into: **adding a file.**

An experimental patch was written for a side arm and placed in the TP=3 patch directory, with an
environment-gated call added to the prelude. Neither changes production behaviour by a byte — the
hook does nothing with the flag unset, and the patch is never called at TP=3. Both change the
**identity**. The next production restore was refused on all three nodes:

```text
preflight-fastload: sidecar stale - boot refused
  patches.patch-fullscope-tp2.py: recorded='<none>' now='bca9a201...'
  patches.tp3-prelude.sh:         recorded='e17d46a4...' now='5eba79f3...'
```

**That is the design working, and it should not be softened.** A sidecar is pre-processed weights
belonging to a specific set of code; using one whose code has changed is exactly how a stack serves
fluent wrong answers. But it cost a restore, at the end of an arm, when production was already down.

The rule, in three lines:

1. **Experimental patches live in their own directory**, not the one the prelude hashes. If two
   places need the same file, **hard-link it** rather than copying it — one inode, no drift, and
   removing one directory entry does not delete the file.
2. **Nothing is added to, removed from or edited in the patch directory between a dump and a load.**
   Not a file that is gated off. Not a file that is never called. The gate hashes the directory, not
   the call graph.
3. **Back up the prelude before you hook it** (`cp` to a `.bak-` name) and restore from that backup
   rather than reversing the edit by hand — the hash has to match to the byte, and an edit that is
   semantically identical is not.

### 11.3 The cheapest acceptance test for a loader change is a meta-device name-set check

A loader change — a new checkpoint layout, a packed-module mapping, a module split — has a failure
mode that no speed or quality number catches quickly: **it loads without error and the numbers are
subtly wrong.** The test that does catch it costs no GPU, no weights and no boot.

Instantiate the real model class on `torch.device("meta")`, walk the checkpoint's
`model.safetensors.index.json` weight map through the real `load_weights`, and count two things:

- **unmapped** — checkpoint tensors with no parameter to receive them (a `KeyError` at boot);
- **unfilled** — parameters no checkpoint tensor ever writes (the silent half, and the dangerous one).

Both must be **0**, and the module census — which modules resolved to the quantized method and which
stayed BF16 — must match the list you predicted before you looked. Three properties make this worth
more than it looks:

- **It is verifiable against a known-good case.** Run the same derivation against the checkpoint that
  already serves; if it does not come back 0/0, the instrument is wrong and its reading about the new
  checkpoint means nothing. That check is what turned this from an argument into a measurement
  ([13](13-full-scope-checkpoint.md) §2).
- **It must run at the real rank count.** A single-rank run cannot see a tensor-parallel bug. Our
  two-rank run caught a replicated layer carrying the global TP rank, which would have overrun on
  `narrow` on rank 1 and could not have appeared at TP=1 ([13](13-full-scope-checkpoint.md) §3.3).
  For a padded TP=3 arrangement the *last* rank is mandatory: the pad lives there.
- **It runs beside an idle engine** under §10's rules — a throwaway container, a memory cap, a cpuset
  disjoint from the engine's, no GPU device, no weights read except safetensors headers.

It is a tier-A test in §9's table, and on the full-scope arm it did the job the intended boot-time
gate could not: the plugin's own `CUDA_EXL3_DEBUG_NAMES` diagnostic printed nothing, because it logs
at `info` and this image configures only vLLM's logger. **An acceptance gate has to be verified before
the arm, the same as any other instrument** — the flag was set, the container had it, and the output
was silence that looked exactly like a clean run.

### 11.1 The profile-baseline rule: settle the host, then read the pool

A dump boot is only the loudest case of a more general problem, and the general case cost us a real
finding before it was understood.

**The KV pool is a difference between two readings of `/proc/meminfo`, not a reading of memory.** On
this integrated-GPU part vLLM's "free GPU memory" *is* `MemAvailable`, and "consumed memory
(weights + non-torch)" is `MemAvailable(just after NCCL init) − MemAvailable(after the profile run)`
([07](07-kv-and-draft-page.md) §1.1). Two consequences that a measurement protocol has to carry:

1. **The instrument runs backwards.** A node that starts with less memory available computes itself a
   *larger* pool. Whatever the host has not finished reclaiming — the previous container, page cache,
   dirty writeback — is booked as engine memory it never took.
2. **It is systematic, not random.** The ranks start worker-2 → worker-1 → head, so the head is always
   the node given least time to reclaim ~90 GiB, and its number is always the inflated one. Read as an
   allocation, that produced a "the head has 8.2 GiB of stranded memory" claim that survived a day and
   was false ([11](11-open-issues.md) §2.3).

The protocol rule, in three parts:

- **Settle the host before the container.** `scripts/start-tp3.sh` waits after `docker rm -f` until
  `MemAvailable` is back over `SETTLE_MIN_GIB`. It buys **zero** tokens; it buys a ruler that does not
  move ([08](08-fast-boot.md) §5.1).
- **Read the pool from a load boot only.** Dump boots read about 6.7 % low, and the number is not
  merely noisy, it is biased.
- **Verify the gate on the boot you are about to quote**, the same way any other instrument is
  verified before its reading is used: all three ranks within **1 GiB** on both the
  `Free memory on device` and the `Actual usage ... consumed memory` lines. Fail that check and the
  boot still produced a valid speed and quality result — and no pool result.

**State the exposure honestly, in both directions.** No pool figure published in this repository is
known to be wrong: the pool takes the minimum over ranks, and on every boot with a ledger the polluted
node was the head, which was not binding. That is luck — the polluted node is whichever starts last —
and the amount at stake is **27 % of a rank's KV allowance**. Meanwhile recent pool figures span
**4,231,404 → 4,484,848, 6.0 %** `[measured-here]`, each step of which has a candidate explanation in
[08](08-fast-boot.md) §5, with comparable post-fix load boots agreeing to 0.4 %. The cost was never a
proven error; it was that **an explanation and an artefact were indistinguishable**, in the one metric
this stack has spent the most arms on, at exactly the few-percent scale its decisions are made at.

The general form, and the one worth stealing: **an instrument that measures a difference between two
moments is only as good as your control over both moments.** Verify the ruler, not just the reading —
the same lesson the bandwidth ruler taught in [10](10-results-and-roofline.md) §4.1, arriving by a
completely different route.

### 11.4 "Boot time" is three different measurements, and autostart is the third

This repository publishes 251 s, 620 s and 315 s and they are not versions of one number:

| What it measures | Starts at | Ours |
|---|---|---|
| **Fast-load boot** | `docker run` on a warm, settled host | **251 s** (weights 58 s) |
| **Dump boot** | the same, on a boot that also writes the 53 GB sidecar | **620 s** |
| **Boot from power-on** | `reboot`, with the autostart unit enabled | **315 s** (see below) |

The third exists because it is the only one that includes the layer the other two assume: the fabric.
Measured once, on production configuration 10, all three nodes rebooted together `[measured-here]`:
ssh and `ibv_devinfo` 4/4 at **+29 / +30 / +31 s**; the units log `Finished` at **+98 / +98 / +103 s**,
which is the whole of `ExecStartPre` (docker, ConnectX-7 4/4, fabric pings, `drop_caches`) plus the
settle gate; `/health` returns 200 at 22:28:21, **212 s after the last unit finished**. The pool read
5,652,892 and the gates read 10/10 and 12/12 afterwards
([`results/boot/boot-ledger.md`](../results/boot/boot-ledger.md),
[systemd](../systemd/README.md)).

**And it comes with its own instrument lesson, which is why it is on this page.** The harness printed
`health 200 +242s`; the wall-clock stamps in the same log give **315 s**, and 242 s before the health
check lands on no event the log records. We cannot reconstruct which instant the counter started
from, so **both are published and the larger is the one to plan with**. The rule this repeats, from
§11.1 and from [10](10-results-and-roofline.md) §4.1: a number that is a difference between two
moments is only as good as your control over both, and a harness that prints an elapsed time is
itself an instrument that has to agree with a clock before it is quoted.

Two protocol notes for anyone repeating it. **One trial is one trial** — systemd starts the three
units with no ordering between them, and what carried this boot is the workers' rendezvous retrying
until rank 0 appears, not a guarantee. The test has since been repeated on production configurations
11 and 12, reading 312 s and 311 s against this boot's 315 s; three trials are three trials, and the
ordering is still not enforced. And **a reboot is the cleanest baseline a pool number can
have**, which is what makes the +0.6 % agreement with a settled `docker run` the strongest check on
the settle gate we have run.

---

## 12. What is next

[10 — Results and roofline](10-results-and-roofline.md) — the numbers this protocol produced, and how
close to the hardware they are. [12 — The tuner cache](12-tuner-cache.md) — why §1 has two protocols
in it.
