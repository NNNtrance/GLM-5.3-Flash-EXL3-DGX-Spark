# Rejected: CUDA graphs for the draft model through FlashInfer's XQA kernel (7–8 October 2026)

**Applies to: TP=3, the main (vLLM `main`) stack with an fp8-KV DFlash2 draft.** The reason this engine
runs with CUDA graphs off in speculative mode was found on 6 September and filed upstream
([`docs/11`](../../docs/11-open-issues.md) §2.29: "the patch is not written"). This page is that patch, written
and verified on the new stack, measured — and not shipped. It records the measurement, the three decision
rules that disagreed, and why the one that rejected it was applied.

## 0. The cause, in one paragraph

With the draft's KV cache in fp8, the draft attention backend is FlashInfer, and its builder picks the **XQA**
decode kernel using the *draft's own* head counts (12 query / 3 KV heads per rank). vLLM then asks
`FlashInferMetadataBuilder.get_cudagraph_support`, a **class method**, which reads the head count from the
*target model's* config instead: 22 query heads per rank at TP=3. The draft group has 3 KV heads, and
`22 % 3 ≠ 0`, so the class method answers `UNIFORM_SINGLE_TOKEN_DECODE`, and with speculative decoding on,
vLLM turns full graph capture off (`setting cudagraph_mode=NONE`). The kernel the builder actually chose has
no such restriction. The mechanism, the head-count arithmetic (TP=2: 32 % 4 = 0; TP=3: 22 % 3 = 1) and the
upstream filing, vllm#55581 (ours, open), are in [`docs/11`](../../docs/11-open-issues.md) §2.29; the earlier
recipe measured the only route then available, a bf16 draft KV, at +1.75 % at one user on its own engine and
left graphs off. Graphs are off at TP=3 for this reason and on at TP=2, where the division is clean. The new
stack shows the same gate line, so nothing about the cause has changed; what is new here is the patch and
its measurement.

## 1. Settings

| | |
|---|---|
| Image | `a4h-21d93d0d8-1451b1a` = the bias-fix build `a4g` plus one extra patch; control `a4g-21d93d0d8-1451b1a` on the same nodes |
| The patch | in `vllm/v1/worker/gpu/attn_utils.py::get_attn_cg_support` (the V2 model runner), where the group-level answer is formed: when the class method says "single-token only" **and** the builder selected XQA **and** DCP is off **and** the group's per-rank query head count differs from the model's — the exact condition of the bug — report `UNIFORM_BATCH`. Every other group (MLA, KDA, indexer, any group answering NEVER/ALWAYS/UNIFORM_BATCH) is unchanged. Knob `HAREM_DRAFT_XQA_CG=0` returns upstream behaviour; one log line per selected group |
| Engine | TP=3 + EP, `gpu-memory-utilization` **0.75**, KV `fp8`, `--max-num-seqs 5`, `--max-num-batched-tokens 2048`, `--block-size 256`, draft k=7 (fp8 draft KV, FlashInfer draft attention), fast-load LOAD, no NODROP, no per-batch-size schedule |
| Protocol | each arm after the night's usual boot; ~25 minutes between the test and the same-night control (§4); step-time driver 8 s windows ×3, two-round content set (round 2), a teacher-forced agreement check, a prefix-state check, a profile |
| Raw | none beyond the tables here `[measured-here, raw lost]` for the per-window files, which stay with the private harness |

## 2. What it did

| Evidence | Control `a4g` | `a4h` |
|---|---|---|
| Graph mode | `…FlashInferBackend (UNIFORM_SINGLE_TOKEN_DECODE); setting cudagraph_mode=NONE` | **`FULL_AND_PIECEWISE`**, no warning; `Capturing dflash2 CUDA graphs (FULL)` **5 of 5**; target graphs: full (2 + 5) + piecewise at 13 sizes (1…80) |
| Log line | none | on each rank, four times: `builder selected XQA with 12 q / 3 kv heads per rank (model: 22 q heads), no DCP -> UNIFORM_BATCH` |
| Graph memory, actual / estimated | | 0.47 / 2.04 GiB (and 0.50 / 1.95) — vLLM subtracts the estimate from the KV pool |
| **KV pool** | 4,659,400 tokens / 33.85 GiB | **4,367,847 / 31.75 GiB: −6.3 %** (−2.10 GiB); the dump boot −6.6 % |
| Profile, launches per step | 2,219 single launches | **2 graph + 51 single** |
| Profile, all-reduce wait per step (head) | 15.84 ms | **13.19 ms (−2.65 ms)** |
| Profile, GPU kernel sum | 79.07 ms | 77.54 ms |
| Profile, trace wall time per step | 81.67 ms | 76.19 ms |
| Main MoE GEMM per step | 33.44 ms (31.84 ms in the fresh-node baseline) | 34.20 ms |

So the graph did what graphs do — the CPU launch load collapsed and the ranks synchronised better — and
the GPU kernel work did not shrink.

**Correctness: three gates, all passed.**

| Check | Result |
|---|---|
| Assistant KL to reference, all texts | **0.00857** (bias-fix build 0.00841; its own repeat 0.00813): ×1.02 of the baseline, ×1.05 of the repeat — noise (prefill only: the graph path matters for decode) |
| Teacher-forced agreement, 4,352 tokens | 4,313 (**99.10 %**), 17 large-margin divergences; the two bias-fix arms read 4,305 (98.92 %, 23) and 4,326 (99.40 %, 12): **inside** their band |
| Prefix-state trace (vLLM #55601 class): the same 13.5k-token document cold and warm | greedy first divergence cold↔cold 34, cold↔warm 40 and 43 — the same band; teacher-forced \|Δ log-probability\| cold↔cold 0.035 mean, cold↔warm 0.030 (second sequence 0.019 / 0.021); **no trace** of a wrong restored state |

(The large teacher-forced margins in that check all had the end-of-turn token as the teacher's argmax,
where the generator had suppressed it with `min_tokens`: a measurement condition, not an engine fault.)

## 3. How much it gained, with the baseline problem

| Comparison | One-user step | Four-user step | Content-fast C1 / C4 |
|---|---|---|---|
| `a4h` vs baseline taken the evening before on fresh nodes | **−0.6 %** (75.60 vs 76.06 ms) | +0.7 % (164.66 vs 163.54) | −3.4 % / −1.6 % (vs a draft-less-run base) |
| `a4h` vs **a control booted 25 minutes later** on the same (aged) nodes | **−2.56 %** (75.60 vs 77.58) | **−1.35 %** (164.66 vs 166.92) | **−3.74 % / −1.46 %** |

Two different baselines gave two different answers because memory bandwidth on the test nodes had fallen
7–11 % between them ([bandwidth-fragmentation.md](bandwidth-fragmentation.md)): the test and the same-night
control both ran on degraded nodes (probes before and after the pair within 0.5 %), the evening baseline did
not. **The same-night control is the valid comparison.** Content set, round 2, against the evening base: C1
aggregate +2.6 % (63.65 vs 62.89 → 65.93 vs 64.26 tok/s), C4 −2.9 %; draft acceptance unchanged (60.4 % → 62.1 %
at C1, 65.3 % → 63.4 % at C4).

## 4. Three decision rules, written at different times, that disagreed

| Rule | Written | Content | Result |
|---|---|---|---|
| **1. The task's criterion** | before data | one-user step ≥ 2 % shorter than the baseline **and** four-user ≤ +1 % **and** acceptance not lower **and** accuracy gates | one-user −0.6 % → **reject** |
| **2. Second rule** (written by whoever ran the arm) | after seeing the baseline was stale, before the control data | against the same-night control: one-user step ≤ −2 %, four-user step ≤ +1 %, content-fast one-user ≤ −2 %, probe shift ±2 % | −2.56 %, −1.35 %, −3.74 %, <0.5 % → **gain** |
| **3. Holistic rule** (the person deciding) | 00:33, before the control data | gain against the same-night control: one-user step ≥ **3 %** and four-user ≥ **2 %**; otherwise the costs (first-token time, KV pool, swap at boot) outweigh it | one-user −2.56 % (fast −3.74 %), four-user −1.35 % (fast −1.46 %) → **reject (park)** |

Rules 2 and 3 conflicted. Rule 3 was applied: the patch was moved out of the patch set into an archive (not
deleted), the cluster went back to the previous build with its boot evidence (graph mode `NONE`, KV
4,651,226, 14 patches no-change, gate 42 × 3). **Honest note:** the 3 % and 2 % bars of rule 3 are judgement,
not derived from noise; the result misses them by 0.4 and 0.65 points, so this was a close decision decided by
the costs, not by the speed.

## 5. What it would have cost (and why that decided it)

- **KV pool −6.3 %** (2.10 GiB), caused by vLLM's own graph-memory estimate (2.0 GiB estimated, 0.47 actual),
  which a re-balanced `gpu-memory-utilization` of about +0.017 would give back (vLLM states "0.75 = 0.7332
  without graphs").
- **First-token time of short prompts +0.05 to +0.25 s** (graph mode makes the prefill steps run without a
  graph; +70 to +330 ms measured; bf16-draft-KV graphs showed the same).
- **Boot swap-out on the head 9,570 pages against 1,731** (1 s burst at KV allocation; free-memory floor 17.1
  GiB; nothing in the measurement windows). The control booted the same night swapped 1,731 pages against 21–24
  in earlier nights, so the base is already abnormal ([bandwidth-fragmentation.md](bandwidth-fragmentation.md)).
- **One extra patch** carried against the pinned tree, and one more thing to re-verify at each vLLM bump.
- Against: one-user step −2.6 % (−3.7 % content-fast), four-user −1.4 %: **the gain is small because the
  GPU is already ~98 % busy at four users and the draft is a small fraction of the step.**

## 6. The earlier attempt, and what was learned

Before the fix, the same graph path was reached by putting the **draft KV in bf16** (draft attention then runs
on FlashAttention, whose graph declaration has no head check): graphs open, one-user step 76.1 → 73.1 ms
(**−4.0 %**), but the KV pool shrinks **11.5 %** (4,659,400 → 4,125,340; the draft cache is twice the bytes
plus a 2 GiB graph-memory over-estimate), boot 184 → 217 s, acceptance level (62.5 % against 62.7 % pooled),
agreement 99.40 %: no net gain, not recommended. The XQA patch is the same win without the doubled draft
cache, and it too netted out against its costs.

## 7. Rejected, open, retracted

**Rejected:** the draft-KV-bf16 route (§6) and the XQA patch (§4); the patch and its 22 CPU tests are archived.

**Open**

- **Retry condition (set by the person deciding):** after the per-batch-size draft schedule
  ([draft-schedule.md](draft-schedule.md)) is in, because a shorter verification width raises the CPU's share of
  the step and so the graph's value. The patch, its tests, the image and the sidecar still exist and need no
  rebuild. Not retried `[not tested]`; look at the KV re-balance (+0.017) and the short-prompt first-token cost
  first.
- Upstream vllm#55581 is open; if it merges, the patch retires.
- A single dense GEMM shape picked a different tuner configuration between two boots (+2.4 ms on that item),
  a source of boot-to-boot variation unrelated to the graph.
- The explanation of the 5.5× larger boot swap with graphs was not isolated.

**Retracted:** none by this page. (The earlier "22 % 4 head-count" explanation for graphs being off was
already withdrawn on 6 September, in the entry of [`docs/11`](../../docs/11-open-issues.md) headed "The 36/9
drafter sidecar removed the `22 % 4` head-count obstacle"; this page does not change it.)
