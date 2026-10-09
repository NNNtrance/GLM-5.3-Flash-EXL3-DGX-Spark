# Quality gates on the new stack — what was measured on which build, and what is still withdrawn

**Applies to: TP=3, the main (vLLM `main`) stack.** This page exists to prevent one misreading. A four-hour
quality battery ran on the new stack on 7 October, and every test in it passed — **on a build whose MoE
router bias was missing.** Tests whose answers are self-evident (copy this passage, fix this bug, say
whether you fixed it) passed on the wrong router as they would on the right one; they could not see the
defect that the KL check found within hours ([kl-health.md](kl-health.md)). So that battery is recorded
below as what it is, and the benchmark scores in it are **not** claims about the production engine.

All runs: reasoning effort **low** unless stated, temperature 0 unless stated. Why low: the cluster's max
effort would take days for the benchmark sets listed; where the effort would change a conclusion this page
says so (the previous recipe measured `high` as +90 % completion tokens for +4.8 points on tool-eval-bench,
[`results/gates/quality-battery-production-13.md`](../gates/quality-battery-production-13.md) — that was the
previous stack and is itself withdrawn with the rest of its quality numbers).

## 0. The builds

| Name here | Image tag | `cuda-exl3` | What it is |
|---|---|---|---|
| **pre-fix** | `a4f-21d93d0d8-f3e184a` | `f3e184a` | vLLM `21d93d0d8` + plugin port + 14 patches. **Router bias never loaded.** No SwiGLU clamp on routed experts |
| **bias-fix** | `a4g-21d93d0d8-1451b1a` | `1451b1a` | + router bias loaded, fail-closed boot gate. No clamp |
| **production** | `a4i-21d93d0d8-448f1d6` | `448f1d6` | + routed-expert SwiGLU clamp at `swiglu_limit` 10, GEMM-tuner knobs |

## 1. The four-hour battery — on the pre-fix build (7 October 2026, 12:53–16:21)

Three nodes, TP=3 + EP, `gpu-memory-utilization` 0.75, fast-load LOAD, draft k=7, KV `fp8`, vision on;
uninterrupted for four hours, swap-out at most 2 pages, no crash.

| Test | Result on the **pre-fix** build | Reading |
|---|---|---|
| Correctness probe (`scripts/correctness-probe.py`) | 10/10, content-only 9/9 | an "is it open" check |
| Code exam (`scripts/code-exam.py`) | 12/12, three times | same |
| Copy a passage from a 48–65k-token context | **128/128** exact (warm 96/96, cold 32/32); copy margin median 4.25; warm-versus-cold McNemar p = 1 | the answer is known in advance |
| Same-prompt divergence lens (consistency, 42k and 46k contexts) | 32/32 expected tool calls, 0 corrupted, each; **lens 0.80 / 0.58** | a clean engine reads ~1, the broken one we had before read 4.5 |
| Fix a bug in a Godot project, with hidden tests | short **24/24**, long (40k prefix) **32/32**; **false "fixed" claims 0/56**; tests touched 0 | self-evident |
| Honesty (does it say it fixed what it did not) | **0 / 12 dishonest** (blocked 6, fixed-not-verified 4, not-fixed 2) | self-evident |
| Vision, ten items | 10/10 (the script said 9/10: its keyword list lacked "moving right"/"moving down"; the model's answers were right) | instrument false alarm |
| MMLU sample, 57 tasks, 1,995 questions | 86.32 ± 0.74 | **withdrawn with the other benchmark scores** |
| tool-eval-bench, T=0, hardmode, 8 trials | 85.25 (583/736) | **withdrawn** |
| tool-eval-bench, T=1.0, 4 seeds | 88.5 (308/368) | **withdrawn** |
| Prefix-cache continuation, long documents | same document continued from cache 3–4 s against 24–70 s cold; results the same | pre-fix, but a structural property |
| Fast-load LOAD against a normal load | weights bit-identical (1,500 of 1,500 sampled tensors per rank; draft 93 of 94, the 94th an uninitialised unused buffer); logits inside the measured repeat noise | structural, not affected by the router |

`[measured-here, private harness]` for the in-house tests (copy, Godot, honesty, lens) and
`[measured-here]` for those with a public script in `scripts/`; raw is not in this repository for any of
them. The benchmark rows are `[retracted]` as quality evidence for the production engine: they were measured on
the wrong router (see the README notice and [`docs/11`](../../docs/11-open-issues.md) §1.15) and have not been replaced.

**Why this is not validation.** The router bug moved the assistant-token KL to the reference from 0.0084 to
0.103 ([kl-health.md](kl-health.md)) and 11× the engine's own noise. None of the tests above moved with it.
That is the "an easy exam is not evidence" rule in numbers: copy, a unit-test fix and an honesty check have
one correct answer regardless of which eight experts fire. The only check on this page that *did* separate
the broken engine from the fixed one is the distribution-level KL.

## 2. Quality evidence on the corrected builds

| Check | Build | Result | Where |
|---|---|---|---|
| Assistant-token KL to the reference, six texts | bias-fix / production | **0.00841 / 0.00828** against floors 0.0067–0.0081; perplexity 2.222 against the reference's 2.224 | [kl-health.md](kl-health.md) |
| Expert selection equals the official formula | bias-fix | 96 / 96 (0 / 96 before) | [kl-health.md](kl-health.md) §4 |
| Router bias boot gate | all since | `GECTI` 42 / 42 layers × 3 ranks at every boot of the bias-fix and production builds that we logged | [kl-health.md](kl-health.md) §4 |
| Drafted output equals the target's own argmax (4,352 greedy tokens, teacher-forced) | bias-fix (two arms), bias-fix + draft-graph patch, production (and its same-session control) | 98.92 % and 99.40 %; 99.10 %; 99.20 % (control 98.90 %) | [kl-health.md](kl-health.md) §6, [rejected-draft-cuda-graph.md](rejected-draft-cuda-graph.md) §2 |
| Same-prompt lens, two prompts, with and without NODROP | production | 16 / 16 outputs identical in every arm; lens ratio 0.68–1.90 (one A/A control pair unstable: that gate was declared invalid) | [prefix-hits-nodrop.md](prefix-hits-nodrop.md) §5 |
| Strict tool-call gate | production candidate / production | **480 turns, 0 corrupted, 0 rejected, 0 errors** (4 arms × 120); then 30 turns in the compliance exam | [prefix-hits-nodrop.md](prefix-hits-nodrop.md) §5, [boot-and-watchdog.md](boot-and-watchdog.md) §2 |
| Cached continuation against cold, seven points including three off-grid edges | production candidate | \|Δ log-prob\| ratio 0.79–1.13 of the cold-to-cold noise | [prefix-hits-nodrop.md](prefix-hits-nodrop.md) §3 |
| Needle at 60k, depths 10 / 50 / 90 % | production | 3 / 3 | [boot-and-watchdog.md](boot-and-watchdog.md) §2 |
| Vision: 2 images, 1 video, both together; 8 and 16 large images | production | all correct; 16 large images in 84.7 s, all of 1…16 identified | [boot-and-watchdog.md](boot-and-watchdog.md), [memory-fraction-084.md](memory-fraction-084.md) |
| Eight-hour sampled load, answer checks | production | wrong answers: agent 0 / 6,363, prose 0 / 153, JSON 0 / 165, code 0 / 275, Turkish 0 / 217, needles 0 / 12; **mathematics 2 / 210, image counts 2 / 54** | [soak-8h.md](soak-8h.md) §3 |

None of these is an accuracy benchmark. They say that the corrected engine computes the same function as the
reference implementation to within the noise of either, and that the features we rely on (tool calls, prefix
resume, vision, long context, a long sampled run) behave.

## 3. What has not been re-measured on the corrected engine `[not tested]`

- **GSM8K, IFEval, the MMLU sample and the full MMLU, tool-eval-bench at `low` and at `high`, the 1M-token
  needle, the long-context stress runs, ExtractBench.** All were withdrawn on 8 October and none has been
  re-run on a corrected build as of this writing.
- **The probe and the code exam** (`scripts/correctness-probe.py`, `scripts/code-exam.py`) on the bias-fix or
  production build.
- **A BF16 or official-API reference** for the quality of the *model* rather than of the engine.

Expectation, labelled as such: with assistant-token KL at the repeat floor and perplexity equal to the
reference's, benchmark scores should land within noise of what the reference implementation scores on the same
checkpoint — and they may differ from the withdrawn figures in either direction. `[estimate]`

## 4. What this cost

This page reports no gain, so the price is of the evidence itself: four hours of the whole cluster for a
battery that could not distinguish the broken engine from the fixed one; a set of benchmark scores we had
quoted for weeks that cannot be quoted now; and a re-measurement bill — GSM8K, IFEval, MMLU, tool-eval-bench at
two efforts, the 1M needle and the long-context runs — that is still unpaid. The distribution-level check that
did find the defect cost about an hour of cluster time per build after one-off preparation.

## 5. What we tried and rejected, open problems, retracted

**Rejected:** treating the four-hour battery as evidence of correctness (§1).

**Open**

- The re-measurement above; it is the first thing to do before any quality number is quoted.
- A text set that exercises the routed-expert clamp ([kl-health.md](kl-health.md) §6).
- Two wrong mathematics results and two wrong image counts in the soak were not analysed.
- The model-written fail rate on JSON at a tight output budget (17 % cut at the soak's budget) is a property of
  the test, not measured further.

**Retracted:** every quality figure in the pre-fix battery above that is a benchmark score, as quality
evidence for the production engine. Tests with self-evident answers are kept as recorded, with their
pre-fix label.

## 6. Where our pre-registered rules were wrong in this directory

The ledger of every rule we wrote down before seeing data and then had to treat as flawed, so that none
is quoted as a clean pass:

| Where | Flawed rule | What happened | File |
|---|---|---|---|
| KL check | absolute thresholds (mean KL 0.0345, top-1 98 %) | exceeded by the reference's own jittered copy; literal verdict stays "problem", relative gates and the assistant subset reported beside it | [kl-health.md](kl-health.md) §7 |
| Draft length | agreement as an absolute count | penalised an arm that generated fewer tokens; literal verdict not applied or reversed; ratio rule written before the confirmatory chain | [draft-schedule.md](draft-schedule.md) §3 |
| NODROP A/B | pool band ±0.2 % from one observation; a tool-ordering defect | literal verdict reject; corrected rule written before the extra boot's data; confirmatory arm passed | [prefix-hits-nodrop.md](prefix-hits-nodrop.md) §4 |
| NODROP long run | one-user speed bar 1 % narrower than boot-to-boot spread; lens gate failed its own A/A test | literal verdict reject; speed-only six-boot arm passed; lens gate recorded invalid | [prefix-hits-nodrop.md](prefix-hits-nodrop.md) §5 |
| Production clamp build | agreement as a count; validity band around two arms | both corrected before data; passed on the corrected rules | [kl-health.md](kl-health.md) §6 |
| Memory fraction | 0.5× token-rate bar in the image window | failed for a product behaviour, not memory; ruled on the memory gates alone, no confirmatory arm | [memory-fraction-084.md](memory-fraction-084.md) §3 |
| Draft CUDA graph | one-user ≥ 2 % against a stale baseline; a second rule written after seeing the ruler | three rules disagreed; the holistic one, which rejected it, was applied | [rejected-draft-cuda-graph.md](rejected-draft-cuda-graph.md) §4 |
