# results/main-stack — measurements of the vLLM-`main` stack (7–9 October 2026)

The files here are the measured results of the rewrite announced in the README's correctness notice: vLLM
`main` @ `21d93d0d8`, our `cuda-exl3` fork (`NNNtrance/cuda-exl3`, branch `tp3-vllm-main`, production build
`448f1d6`) and 14 build-time patches, on three DGX Spark (GB10) at TP=3 with expert parallelism. The older
`results/` files were measured on a stack whose MoE router bias was never loaded; those quality numbers are
withdrawn and those speeds are not a baseline for this one (README notice; [`docs/11`](../../docs/11-open-issues.md)
§1.15). Nothing in this directory depends on them.

Everything below was measured 7–9 October 2026. Each file states its settings, its evidence tier on every
claim, a "what this cost" line for every gain, and the rejected / open / retracted items. Where a rule we wrote
before seeing the data turned out to be flawed, the file says so and says what was done; the list is in
[quality-gates.md](quality-gates.md) §6.

## Builds

| Name | Image tag | `cuda-exl3` | Router bias | SwiGLU clamp on routed experts |
|---|---|---|---|---|
| pre-fix | `a4f-21d93d0d8-f3e184a` | `f3e184a` | **not loaded** | no |
| bias-fix | `a4g-21d93d0d8-1451b1a` | `1451b1a` | loaded; boot gate | no |
| **production** | `a4i-21d93d0d8-448f1d6` | `448f1d6` | loaded; boot gate | **yes** (`swiglu_limit` 10) |

## Production configuration (since 8 October 2026)

TP=3 + expert parallelism over three nodes (`head`, `worker-1`, `worker-2`, mesh transport as in
[docs/06](../../docs/06-nccl-mesh.md)); checkpoint `turboderp/GLM-5.3-Flash-exl3` branch `4.05bpw`;
`--gpu-memory-utilization 0.84`, `--max-model-len 1000000`, `--max-num-seqs 5`,
`--max-num-batched-tokens 2048`, `--block-size 256`, `HAREM_SW_BLOCK_SIZE=256`, KV `fp8`,
`--kda-prefill-backend flashkda`, `--hf-overrides '{"index_topk":2048}'`; DFlash2 draft k=7 with fp8 draft KV,
**`disable_eagle_block_drop: true`** and **`num_speculative_tokens_per_batch_size: [[1,1,7],[2,8,3]]`**;
vision `{"image":16,"video":4}` with `--mm-encoder-tp-mode data`; reasoning effort `low` by default;
started at boot by a systemd unit; a watchdog on `head` reboots all three nodes after three failed probes.

## Headline numbers

Settings are abbreviated; the file named in the last column gives them in full. Speeds are per the build
and gmu named there — they are **not** all production-configuration figures.

| # | Metric | Value | Settings | Date | Evidence | File |
|---|---|---|---|---|---|---|
| 1 | Assistant-token KL to the official reference, **router bias missing** | **0.1033** (engine repeat floor 0.0094; reference jitter 0.0067) | pre-fix, TP3+EP, gmu 0.75, 6 texts, 21,066 positions | 7 Oct | `[measured-here]` | [kl-health.md](kl-health.md) |
| 2 | Same, **bias loaded** (production build) | **0.00841** bias-fix, **0.00828** production; perplexity 2.222 against the reference's 2.224 | as above, KV fp8, effort low | 7–8 Oct | `[measured-here]` | [kl-health.md](kl-health.md) |
| 3 | Expert selection equal to the official formula | **0 / 96 → 96 / 96** positions | TP=1 truncated six-layer test, layer 3 | 7 Oct | `[measured-here]` | [kl-health.md](kl-health.md) §4 |
| 4 | Decode step cost of correct routing | **+11 %** (70.3 → 78.2 ms, 1 user), **+20 %** (139.4 → 167.0 ms, 4 users); the cause is 28 % / 38 % more distinct experts read per step | bias-fix against pre-fix, same session, gmu 0.75, k=7, 3 windows | 7 Oct | `[measured-here]` | [speed-map.md](speed-map.md) §1 |
| 5 | Single-stream decode by content, per stream | code **59.0**, mathematics **70.3**, tool-call JSON **48.2**, English prose **35.1**, short Turkish **22.5** tok/s; draft acceptance 51 / 67 / 41 / 24 / **10** % | bias-fix, k=7, 4 prompts per type, round 2, `--max-num-seqs 5`, gmu 0.75 | 7 Oct | `[measured-here]` | [speed-map.md](speed-map.md) §2 |
| 6 | Aggregate decode, 12 short code prompts | C1 **64.3**, C4 **120.1**, C5 **133.3**, C8 **136.6** tok/s (saturated at C5: slots, not engine) | bias-fix, `hizset-v2`, k=7, 256 tokens, effort low, T 0, gmu 0.75, round 2 | 7 Oct | `[measured-here]` | [speed-map.md](speed-map.md) §4 |
| 7 | Draft length by running requests: four-user step | **160.2 → 114.5 ms (−28.6 %)**; pooled rate C2 / C4 / C5 **+10.7 / +15.6 / +7.1 %**, one user unchanged (+2.4 %); mathematics −12…−17 % | bias-fix, `[[1,1,7],[2,8,3]]` against a fresh k=7 control, gmu 0.75, triple reboot before each arm | 8 Oct | `[measured-here]` | [draft-schedule.md](draft-schedule.md) |
| 8 | NODROP: first-token time of agent follow-up turns | **2.34 → 1.24 s** serial (−49 %), **7.25 → 3.83 s** with five sessions at once; tokens re-read **−61 %**; speed ×1.0013 / ×0.9993 over six boots; 480 strict tool-call turns, **0 corrupted** | 5 sessions × 7 turns, up to 105k tokens, `disable_eagle_block_drop`, gmu 0.75, T 0 | 8 Oct | `[measured-here, private harness]` | [prefix-hits-nodrop.md](prefix-hits-nodrop.md) |
| 9 | KV pool and the memory fraction | **6,188,010** tokens at **0.84** (6,190,735 at autostart); 0.85 gave 6,354,223 but the head fell to 882 MiB free under 16 large images and the 1 GiB watchdog stopped the engine; 0.84 held 1,621 MiB | production build, `{"image":16,"video":4}`, five users, gmu 0.85 and 0.84 | 8 Oct | `[measured-here, private harness]` | [memory-fraction-084.md](memory-fraction-084.md) |
| 10 | Eight-hour sampled soak | **7,515 requests, 0 errors, 6 / 6 criteria**; free-memory floors 4.67 / 6.93 / 7.02 GiB; swap 0; hourly speed ratio 0.996–1.052; `/metrics` agrees (8,451 completions, 0 errors) | production, 5 concurrent, T 0.7, top-p 0.95, effort low 70 % / high 30 % | 8–9 Oct | `[measured-here, private harness]` | [soak-8h.md](soak-8h.md) |
| 11 | Boot from power-on, and recovery from a killed engine | **274 s** to `/health` 200; kill → triple reboot → `/health` 200 in **8.2 min** with no person involved | production unit, autostart, watchdog on `head` | 8 Oct | `[measured-here, private harness]` | [boot-and-watchdog.md](boot-and-watchdog.md) |
| 12 | GB10 memory bandwidth after engine sessions | **−6.6 / −10.4 / −10.2 %** after ~10 sessions (234.3 / 244.7 / 244.1 → 218.9 / 219.3 / 219.2 GB/s); −4 to −7 % after one; **a reboot restores it, `compact_memory` does not** | continuous matrix-vector probe, 34 samples × 3 nodes | 8 Oct | `[measured-here, private harness]` | [bandwidth-fragmentation.md](bandwidth-fragmentation.md) |
| 13 | Draft CUDA graph through FlashInfer XQA (patch ours) | step **−2.56 %** (1 user) and **−1.35 %** (4 users) against the same-night control; KV pool **−6.3 %**; first token of short prompts +0.05–0.25 s → **rejected** | `a4h`, gmu 0.75, fast-load, draft k=7 | 8 Oct | `[measured-here]` | [rejected-draft-cuda-graph.md](rejected-draft-cuda-graph.md) |
| 14 | Cold prefill | **1,809 / 1,816 tok/s** at 32k tokens, **1,791** at 90k (no prefix hit); correct routing reads +41 % expert bytes per chunk yet prefill moves only −0.7 to −1.9 % | bias-fix, 2,048-token chunks, gmu 0.75 | 7 Oct | `[measured-here, private harness]` | [speed-map.md](speed-map.md) §4 |
| 15 | Routed-expert SwiGLU clamp (production build) | **no measurable effect**: assistant KL 0.00828 inside the 0.00808–0.00841 band; step 75.84 → 74.44 ms (−1.85 %, 2 steps of resolution); four-user +0.01 % | production against bias-fix, same session, each after a triple reboot | 8 Oct | `[measured-here]` | [kl-health.md](kl-health.md) §6 |

## Files

| File | What it is |
|---|---|
| [kl-health.md](kl-health.md) | The comparison with the official reference implementation that found the missing router bias: method, floors, layer curve, cause, fix, padding audit, SwiGLU clamp, the flawed absolute thresholds and the instrument defects |
| [speed-map.md](speed-map.md) | What correct routing costs, speed by content type, drafted against draft-less, where the step time goes, concurrency, prefill |
| [draft-schedule.md](draft-schedule.md) | `num_speculative_tokens_per_batch_size`: how draft length falls with running requests; the calibration chain, the flawed rule, the confirmatory arm, the transition test |
| [prefix-hits-nodrop.md](prefix-hits-nodrop.md) | `disable_eagle_block_drop`: the cache-grain mechanism, the agent-turn A/B, cached-versus-cold accuracy at seven points, the flawed rules and the confirmatory arms, the six-boot speed check |
| [memory-fraction-084.md](memory-fraction-084.md) | Why 0.84 and not 0.85: the KV arithmetic, the 16-large-image worst case, the watchdog stop, the gates and what was ruled |
| [soak-8h.md](soak-8h.md) | Eight hours at five concurrent requests with sampling: the six criteria, hour-by-hour, answer checks, cache-retention capacity |
| [boot-and-watchdog.md](boot-and-watchdog.md) | Autostart in 274 s, the compliance exam, the three-node-reboot watchdog and the real-failure drill |
| [bandwidth-fragmentation.md](bandwidth-fragmentation.md) | The 7–11 % bandwidth loss after engine sessions, the probe, what restores it, and the measurement protocol it forced |
| [rejected-draft-cuda-graph.md](rejected-draft-cuda-graph.md) | The XQA graph patch: correct, measured, and not shipped, with the three decision rules that disagreed |
| [quality-gates.md](quality-gates.md) | Which quality evidence belongs to which build, what is still withdrawn, what has not been re-measured, and the ledger of flawed pre-registered rules |
| [`raw/`](raw/) | The numbers behind the tables as CSV (below) |

### `raw/`

| File | Rows | Backs |
|---|---|---|
| `kl-pairs.csv` | 304 | every KL pair, build and scope: n, mean, median, p90, p95, p99, max, top-1 |
| `kl-layer-curve.csv` | 45 | relative residual difference per layer, reference jitter / pre-fix / after fix |
| `speed-by-content.csv` | 40 | per-content, per-concurrency speed, first-token time, acceptance, step time; drafted and draft-less |
| `routing-step-time.csv` | 3 | the correct-versus-missing-bias step times and expert counts |
| `draft-schedule-arms.csv` | 7 | the calibration and confirmatory arms: KV pool, boot, step times, agreement |
| `prefix-hit-agent-turns.csv` | 8 | NODROP A/B per arm and mode: first-token, tokens re-read, acceptance |
| `memory-fraction-arms.csv` | 2 | the 0.85 and 0.84 arms: KV pool, free-memory minima by load window |
| `soak-hourly.csv` | 7 | hour-by-hour speed ratio, requests, first-token time |
| `bandwidth-probe-series.csv` | 102 | every probe sample with the engine sessions since the last reboot |

All CSV were rebuilt from the run files: numbers only, node names mapped to `head`/`worker-1`/`worker-2`,
no prompts, no log lines. `kl-pairs.csv` is the largest at about 51 kB.

## Reading order and what is not here

Start with [kl-health.md](kl-health.md) (why the stack was rebuilt) and [speed-map.md](speed-map.md) (what
it runs at), then the three configuration findings ([draft-schedule.md](draft-schedule.md),
[prefix-hits-nodrop.md](prefix-hits-nodrop.md), [memory-fraction-084.md](memory-fraction-084.md)), then the
operational files. [quality-gates.md](quality-gates.md) before quoting any quality number.

Not in this directory: the measurement harnesses (reference runner, KL comparer, load generators, watchdog
and install scripts — described, not published; the watchdog itself is published in [`tracks/tp3-main/ops/watchdog/`](../../tracks/tp3-main/ops/watchdog/README.md)), engine and boot logs (they carry host names and addresses),
profiler traces, per-window raw JSON. The tables rebuild from the CSV above; the patches are in
[`tracks/`](../../tracks/README.md).

## Not yet measured on this stack

GSM8K, IFEval, MMLU, tool-eval-bench, the 1M-token needle and the long-context stress runs (withdrawn on the
pre-fix routing, [quality-gates.md](quality-gates.md) §3); the repository's own fresh-prefill harness
(`bench/prefill-fresh.py`); a five-round sweep of the production configuration (`scripts/bench-sweep.py`
protocol, [docs/09](../../docs/09-measurement-protocol.md)); anything at `max` or `high` reasoning effort; any
two-node (TP=2) measurement on this stack `[not tested]`.
