# `gpu-memory-utilization` 0.84, not 0.85 — the worst-case vision load (8 October 2026)

**Applies to: TP=3, the main (vLLM `main`) stack, production configuration.** The production fraction is
**0.84**. At 0.85 the KV pool was 2.7 % larger, and the first 16-large-image request on the same run took the
head node below its 1 GiB safety floor, and the watchdog stopped the engine.

## 0. Settings

| | |
|---|---|
| Image | `a4i-21d93d0d8-448f1d6` (production build) |
| Engine | TP=3 + EP, KV `fp8`, `--max-model-len 1000000`, `--max-num-seqs 5`, `--max-num-batched-tokens 2048`, `--block-size 256`, draft k=7 with NODROP ([prefix-hits-nodrop.md](prefix-hits-nodrop.md)) and the `[[1,1,7],[2,8,3]]` schedule ([draft-schedule.md](draft-schedule.md)), fast-load LOAD sidecar, `NCCL_MESH_TIMEOUT_SEC=180` |
| Vision | `--limit-mm-per-prompt '{"image":16,"video":4}'`, `--mm-encoder-tp-mode data`, `--mm-processor-kwargs '{"max_pixels":12544000,"max_image_tokens":8000}'`, `--mm-processor-cache-gb 0`, `--skip-mm-profiling`. The 16-image limit is a product requirement of the deployment (a client chat that accumulates more than four images was otherwise rejected for good); capacity is not cut to make memory fit |
| Arms | gmu **0.85**, then gmu **0.84**, each after a triple reboot of the cluster; both booted with the three KL API flags (`--enable-scale-out`, `--max-logprobs 512`, `--enable-prompt-tokens-details`), which do not touch compute; the one correctness read was taken in the 0.85 arm only |
| Watchdog | 1 s sampler per node, kills the engine if `MemAvailable` < **1 GiB** (the default of the test harness is 4 GiB; at 0.85 the head node's quiet floor is about 4.7 GiB, so the default would have fired in or next to normal use) |
| Load per arm | warm-up → **five users of mixed text** for 5 minutes (temperature 0.7, top-p 0.95) → content-fast mode → [0.85 only: KL read] → **8 large images** in one request with the five users in the background → **16 large images** in one request with the five users in the background (last, so an engine death cannot cost the earlier data) |

All rows `[measured-here, private harness]`; raw in [`raw/memory-fraction-arms.csv`](raw/memory-fraction-arms.csv).

## 1. Why 0.84: the arithmetic, then the two measurements

On GB10 the GPU pool **is** host memory. Every +0.01 of the fraction adds **1.216 GiB** of KV (0.01 × 121.63
GiB) and removes the same from the node's free memory. From the 0.75 boot (consumed by weights and non-torch
allocations 55.89 GiB, peak activation 1.66 GiB, KV 33.68 GiB = 4,656,675 tokens, **138.3k tokens/GiB**):

| Fraction | KV pool predicted | KV pool measured | Head node's quiet free-memory floor, predicted `[estimate]` → measured (0.84 prediction interpolated) |
|---|---|---|---|
| 0.75 | 4,656,675 | 4,656,675 (the boot it was derived from; six boots 4,641,924–4,682,370) | 16.9 → 16.2–16.9 GiB |
| 0.83 | ~6.00 M | not run | ~7.1 GiB |
| **0.84** | **~6.18 M** `[estimate]` | **6,188,010** | ~5.9 → **5.98–6.01** GiB |
| **0.85** | **~6.34 M** `[estimate]` | **6,354,223** | 4.7 → **4.76–4.85** GiB |

The prediction held to 0.1 % and 0.2 %.

**The deployment's two requirements** (stated by the operator, not derived): the pool must be **above 6
million tokens**, and the weakest node must keep **about 1 GiB free** at the worst load. Swap is not forbidden
as such; harm is the criterion, and a few GiB of swap on the most marginal scenario is acceptable if it
settles. Together they bound the choice from both sides, and 0.84 is the smallest step that satisfies both.

## 2. Results

| | gmu 0.85 | gmu 0.84 |
|---|---|---|
| KV pool (tokens) | **6,354,223** | **6,188,010** |
| Boot, `docker run` to `/health` | 169 s | 186 s |
| Head node, five users of text, 5 min: min / median `MemAvailable` | 4,759 / 4,846 MiB | **5,975 / 6,009 MiB** |
| Head node, content-fast mode: min / median | 4,740 / 4,768 MiB | 5,908 / 5,939 MiB |
| Head node, 8 large images + five users: min / median | 1,544 / 1,569 MiB | 3,064 / 3,153 MiB |
| Head node, **16 large images + five users**: min / median | **882 MiB → watchdog stopped the engine 8 s into the request** | **1,621 / 1,695 MiB** |
| Swap-out, pages (quiet / 8 images / 16 images) | 0 / 231 / 50 (+9,807 at the stop) | 0 / 1 / 2 |
| Swap use, 60 s after the 16-image request | not reached | **+0 MiB** (settled) |
| 16-image request | not completed | **HTTP 200, all of 1…16 identified, 84.7 s** |
| Five users, quiet window / during the image windows | | 42 requests, 0 errors, median 22.22 tok/s per request / 15 background requests, 0 errors, median 8.17 tok/s |
| `Xid` count before / after, watchdog events | | 0 / 0 on all three nodes, 0 events |
| Worker nodes, quiet median `MemAvailable` | ~6.6 GiB | ~7.9 GiB; lowest under images **7.29 GiB** |
| Correctness read (0.85 arm only) | assistant KL **0.00806** (bar 0.00908) | not run: numerical path identical |

- The image load lands almost entirely on the **head** node (API, front end and image preprocessing live
  there): ~0.37–0.40 GiB of transient memory per large image, so about 3.0 GiB for 8 and 4.4 GiB for 16. The
  workers' lowest value under images (7.29 GiB) is only 0.6 GiB under their quiet median.
- **The two troughs are not strictly comparable.** The fraction step alone predicts a 1.2 GiB difference in
  the head's trough (it did between the quiet windows: 4.76–4.85 against 5.98–6.01 GiB); the 16-image
  troughs differ by 0.74 GiB (882 against 1,621 MiB). At 0.85 the 16-image request started before the
  8-image request's memory had fully come back (2.8 GiB free at its start); whether the 0.84 arm began from
  a fuller recovery was not recorded. The 0.84 trough is therefore a single observation of a
  sequence-dependent minimum, not a bound.
- Production autostart at 0.84 (a later boot, without the KL flags) reads KV 6,190,735 tokens and
  `MemAvailable` 7.4 / 9.2 / 9.2 GiB (head / worker-1 / worker-2) just after boot.

## 3. The pre-registration, two changes made before data, and a verdict that was not literal

The gates were written into the log first. Before any data, two criteria were changed to match the
operator's rules:

| Gate | First wording | As run (written before data) |
|---|---|---|
| Pool | ≥ 0.98 × the arithmetic (≥ 6,211,598 at 0.85) | **≥ 6,000,000** (the operator's number) |
| Memory | median free memory ≥ 3 GiB in the quiet window, swap-out ≤ 256 pages, swap increase ≤ 6 GiB in the image window | **removed** as gates (values still reported); replaced by: floor ≥ 1 GiB at all times (watchdog event 0), swap increase ≤ 128 MiB in the second half of the quiet window **and** in the 60 s after the 16-image request ("swap must settle when the load ends") |
| Image | | 16-image request HTTP 200, 1…16 all identified, `/health` 200 after, ≤ 600 s, five users 0 errors, five-user token rate during the window ≥ **0.5×** the quiet rate, `Xid` unchanged |
| Speed | | fast-mode step time ≤ 1.05× the draft-schedule chain's reference; first-token median ≤ max(1.5×, +0.2 s) of it |

**0.85: literal verdict fail** — not on any speed gate: the watchdog stopped the engine on the floor, the
chain halted on its own safety rule, and the lock and the stop flag were left for inspection (a safety stop
does not roll on to the next arm by itself). The next rung, **0.84**, was chosen by hand with a rule written
before its data: 0.83 would run only if 0.84 failed for a *memory* reason. It did not, so 0.83 never ran.

**0.84: the memory gates passed; two other gates failed for reasons independent of memory.**

1. *The five-user token rate during the 16-image window* was **8.17 against 22.22 tok/s per request
   (0.37×)** against a 0.5× bar. The cause is not memory: a 16-image request is 63k–127k tokens of vision
   pre-fill, and the engine shares its steps between that pre-fill and everyone else's decode. A plain text
   prompt of the same length does the same. At 0.85 the 8-image window already read 9.77 against 22.61
   (0.43×). It is a product behaviour — **while a very large image request is being processed (about 85 s
   here), other users' writing speed falls to about 37 %** — and the 0.5× bar did not separate that from
   memory.
2. *One content-fast first-token cell* (code, four users) read 0.535 s against a reference of 0.265 s. At
   0.85 a different cell (plain prose, four users) read 0.516 against 0.285; the same cells read ~0.26 s in
   the other arm. That is single-cell noise inside a historical range of 0.26–0.56 s. The rest of the speed
   gates held within ±1.4 % of reference.

**Ruling.** The literal verdict for 0.84 was "fail". No corrected-rule confirmatory arm was run: the
operator's note arrived (the pool is above 6 M and the safe floor is 1 GiB; if both hold at 0.85, 0.85 is
final), 0.85 broke the floor, 0.84 held both, and the two failing gates were assigned to behaviours
independent of the fraction (they read the same at 0.85). **0.84 was ruled final on the two memory
requirements alone**, and the image-window slowdown was recorded as a separate product finding. The 0.83 arm
was not run, so 0.84 is the **first fraction that worked**, not a measured optimum.

## 4. What this cost

- **166,213 tokens of KV pool (−2.6 %)** against the 0.85 that did not survive — 6.19 M against 6.35 M,
  about 6.2 simultaneous 1M-token requests instead of 6.4.
- **Nothing in speed** (step times within ±1.4 % across the fraction arms; the fraction does not enter the
  compute path).
- The image-window slowdown is **not** a cost of this fraction; it is the same at 0.85.
- **Residual risk:** under 16 large images the head's floor is **1.62 GiB**, 0.6 GiB above the safety line.
  A larger single image, a video clip in the same window, or a different resolution setting was not tested.
  The eight-hour sampled load ([soak-8h.md](soak-8h.md)) ran 54 image requests of 2–4 small images, not the
  16-large-image worst case.

## 5. Rejected, open, retracted

**Rejected:** 0.85 (floor broken, stopped by the watchdog at 882 MiB); the earlier wording that gated on a
3 GiB quiet median (it would have rejected a fraction the operator's own rule accepts).

**Open**

- **0.83 and fractions between 0.84 and 0.85 were not measured**; 0.845 may work and was not tried.
- The new stack uses about 2 GiB more host memory on the head than the previous stack did at the same
  fraction (cross-check against that stack's own ladder: its head floor at 0.85 was 6.0 GiB, this stack's
  predicted 4.7 and measured 4.76–4.85). The source was not isolated (the KL flags and the new API process
  are candidates) `[not tested]`.
- Large-image decode memory is a head-node cost that grows with the image limit; the limit was not reduced.
- No larger-than-3400² image or video-plus-16-images combined load was run.

**Retracted:** the earlier recipe's memory-fraction ladder (0.88 at gmu with a 7.04 M pool, [memory/ladder-6sep.md](../memory/ladder-6sep.md))
is another stack's and does not carry over: its KV pool, vision limit and node floor were different.
