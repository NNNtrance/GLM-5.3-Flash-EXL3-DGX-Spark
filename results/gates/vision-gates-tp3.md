# The vision gates at three ranks — ten gates, three engine windows, and the promotion boot

**7 September 2026 `[measured-here]`.** The gate table behind
[docs/18](../../docs/18-vision-at-three-ranks.md). Three trial windows and one promotion boot, all on
the same three nodes, all against a same-session production configuration 12 reference.

Settings, every arm: image `exl3-zeus:754421f` (`cuda-exl3` `754421f`), the
[`tracks/tp3/patches/`](../../tracks/tp3/patches/) tree **plus**
[`patches/vision/`](../../tracks/tp3/patches/vision/README.md), TP=3 + expert parallel,
`turboderp/GLM-5.3-Flash-exl3` at 4.05 bpw (full scope), `kv-cache-dtype fp8` and an fp8 draft cache,
DFlash2 draft at k=7, `--block-size 256`, `HAREM_SW_BLOCK_SIZE=256`, `--max-num-batched-tokens 2048`,
`--max-num-seqs 8`, `NCCL_MAX_NCHANNELS=8`, `gpu-memory-utilization` **0.88**, the sm_12x correctness
set, the indexer workspace bound to 513 MB, `max_model_len 1,000,000`, warm MLA tuner cache,
**temperature 0**, thinking on at reasoning effort **`low`**. Vision on top of that:
`--mm-encoder-tp-mode data`, `--mm-processor-kwargs {"max_pixels":12544000,"max_image_tokens":8000}`,
`--limit-mm-per-prompt {"image":4,"video":2}`, `--mm-processor-cache-gb 0`, `HAREM_VISION=1`,
`CUDA_EXL3_PACKED_MAPPING={"qkv_proj":["q_proj","k_proj","v_proj"]}`.

Rounds 1–3 booted **without** a fast-load sidecar (the configuration-12 sidecar is refused by a
tower-carrying engine, correctly); the promotion boot has its own.

---

## 1. The four windows

| Window | Engine time | What it established |
|---|---|---|
| Round 1 | 30 min 54 s | Tower loads and images work end to end; **the first video killed the engine core on all three ranks** |
| Round 2 | 12 min 34 s | The crash trace taken from all three ranks — identical, same second; `--mm-processor-cache-gb 1` tried on host memory and found insufficient |
| Round 3 | 40 min 53 s | VS4 + VS6 + VS7 and the geometry gate: **all ten gates pass**; speed, 1M-token and acceptance measured |
| Promotion | 06:19 → 07:01 | Sidecar dump (53 GB/node), load boot 247 s, worst-case stress, environment switch, unit start 244 s |
| Reboot test | after 07:01 | All three nodes rebooted together, autostart enabled: `/health` 200 at **318 s** |

Production configuration 12 was verified healthy after **every** window: `/health` 200, KV pool
inside its 6.62–7.46 M band, cold gates 10/10 and 12/12, `HAREM-VISION` line count 0, all three units
`active + enabled`, `ibv_devinfo` 4/4 on each node, and no whole-cluster reboot needed at any point.

---

## 2. The gate table

Round 1's six blocked gates are the ones the video crash prevented from running at all.

| Gate | What it asks | Round 1 | Round 3 | Promotion |
|---|---|---|---|---|
| K0 | boot evidence + KV pool line | **PASS** — 8 lines, KV 7,005,509 | **PASS** — 9 lines, KV 6,994,490 | **PASS** — chain of 5, KV 7,143,250 |
| K1a | text correctness probe | **PASS** 10/10 | **PASS** 10/10 | **PASS** 10/10 |
| K1b | text code exam | **PASS** 12/12 | **PASS** 12/12 | **PASS** 12/12 |
| K2 | four single images | **PASS ×4** | **PASS ×4** | **PASS ×4** |
| K3 | four images, one request | **PASS** — 2,612 prompt tokens | **PASS** | **PASS** |
| K4 | one video | **FAIL — engine dead**, HTTP 500 | **PASS** — 1,722 prompt tokens | **PASS** |
| K4b | two videos, one request | blocked | **PASS** — 3,415 prompt tokens | **PASS** |
| K5 | four images + two videos | blocked | **PASS ×4** — 5,991 prompt tokens | **PASS** |
| K6 | a third video is refused | blocked (500, engine dead) | **PASS** — HTTP 400, engine alive | **PASS** |
| K7a | KV pool inside band | **PASS** 98.6 % | **PASS** 99.6 % | **PASS** 100.2 % |
| K7b | one 1M-token text request | blocked | **PASS** — 999,052 tokens, needle found | **PASS** — 999,052, 635 s |
| K8 | C1 and C8, text only | blocked | **PASS** — −1.6 % / +0.0 % | **PASS** — +0.1 % / +1.5 % |

**Ten out of ten from round 3 onwards.** K1 runs first and text-only on purpose: "the camera went in
and the speech broke" is the failure this arm most had to disprove.

### 2.1 The model's own answers

| Gate | Answer |
|---|---|
| K2 | "Red circle" · "Blue square" · "Green triangle" · "ELEPHANT" |
| K3 | "red circle blue square green triangle ELEPHANT" — all four, **in order** |
| K4 | "A red circular shape (two overlapping red circles) moves from left to right across the frame." |
| K4b | "In the first video, a red circular shape moves steadily to the right across the screen. In the second video, a blue rectangular shape moves steadily downward. The two differ in the object's colour an…" |
| K5 | "Images: 1. Red circle 2. Blue square 3. Green triangle 4. The word \"ELEPHANT\" Videos: 1. Two overlapping red circles moving right 2. A blue rectangle moving down Difference: The first video moves horizontally to the rig…" |
| K6 | HTTP 400, `"At most 2 video(s) may be provided in one prompt. (parameter=video)"`, `engine_alive_after=True` |
| K7b | needle **"ZEPHYR-4417"** at 999,052 prompt tokens |

K4b is the most informative single answer: two videos described separately **and their difference
stated**, which is what proves VS4's time-axis accounting and VS7's lifted limit at once. K3's
"ELEPHANT" is the second: reading text out of an image proves the tower is loaded with the correct
tensors and bound to the language model correctly, not merely constructed.

### 2.2 Prompt-token arithmetic, which closes on the wire

| Request | Measured `prompt_tokens` | Arithmetic |
|---|---:|---|
| one image | 638 | grid `[1,44,58]` → 638 |
| four images | 2,612 | 4 × 638 + prompt |
| one video | 1,722 | 1,656 + prompt |
| two videos | 3,415 | 2 × 1,656 + prompt |
| four images + two videos | 5,991 | 4 × 638 + 2 × 1,656 + prompt |

1,656 is exactly `T × H × W // merge²` for `video_grid_thw = [4, 36, 46]` — the number the CPU bench
predicted before the engine was started.

---

## 3. Cost against production configuration 12

Same-session reference in every column.

| | Configuration 12 | Round 3 (no sidecar) | Promotion (sidecar) | Band |
|---|---:|---:|---:|---|
| KV pool | 7,024,793 / 7,126,721 | 6,994,490 (99.6 %) | **7,143,250** (100.2 %) | floor 94 % |
| C1 aggregate tok/s | 69.90 | 68.8 (−1.6 %) | **70.0** (+0.1 %) | ±4 % |
| C1 rounds | — | 67.2 / 68.8 / 71.0 | 67.9 / 69.95 / 70.17 | — |
| C8 aggregate tok/s | 195.78 | 195.8 (+0.0 %) | **198.8** (+1.5 %) | ±3 % |
| C8 rounds | — | 191.7 / 195.8 / 197.5 | 198.76 / 202.30 / 190.97 | — |
| TTFT C1 / C8 | 0.28 / 0.83 s | 0.28 / 0.82 s | 0.246–0.285 / 0.819–0.826 s | — |
| DFlash2 acceptance | ~62 % | 60.4–64.4 (C1) · 62.5–63.1 (C8) | 60.4–63.9 | — |
| Peak activation, rank 0 | 1.66 GiB | 1.69 GiB | 1.69 GiB, and **1.66** on the boot now serving (§7) | — |
| Consumed, rank 0 | 55.28 GiB | 55.37 GiB | — | — |
| CUDA-graph pool | 0.0 GiB | 0.0 GiB | 0.0 GiB | same regime |
| Indexer workspace | 513 MB | 513 MB | 513 MB | unchanged |

**Round 3's −1.6 % at C1 was the cost of booting without a sidecar, not the cost of the tower.** The
promotion boot, with a sidecar and otherwise identical, crossed the line in the other direction.

**K7b's wall clock is not comparable between the two rows.** Round 3 read 279 s, the promotion boot
635 s, and the second is the honest one: the promotion run was **cold** — the prefix cache reported
19,968 / 1,038,987 = 1.9 % hits — and 999,052 tokens in ~625 s is **~1,598 tok/s**, inside this
configuration's own documented prefill band (fresh 1,739 · 7K 1,620 · sweep 1,744). Round 3's 279 s
implies 3,581 tok/s, more than double the documented rate and not achievable on a cold 1M prefill.
**Round 3's K7b wall clock is withdrawn as a reference figure** `[retracted]`.

---

## 4. The brake, measured

Same session, one 1080p clip, three arms `[measured-here]`:

| `--mm-processor-kwargs` | `video_grid_thw` | vision tokens | peak encoder activation |
|---|---|---:|---:|
| none | (4, 78, 138) | 10,764 | 0.657 GiB |
| `{"max_pixels":12544000}` | (4, 78, 138) | **10,764 — unchanged** | 0.657 GiB |
| `{"max_image_tokens":8000}` | (4, 66, 118) | **7,788** | 0.475 GiB |

Images are **bit-identical** in all three arms — grid `[1,44,58]`, 638 tokens, pixel tensor sha256
`4d0c137b…` — because the checkpoint already budgets images at 8,000 tokens. `max_pixels` is not a
brake on video; it is not accepted by `Glm5NextVideoProcessorKwargs` and the engine logs that it is
ignored. See [docs/18](../../docs/18-vision-at-three-ranks.md) §5.

### 4.1 The sampler mismatch measured across clip durations

Model-free, CPU, from the same bench that reproduced the crash. The ratio
`len(timestamps) / grid_t` is **1 only by coincidence**, in one duration band:

| Clip duration | ratio | Consequence |
|---|---:|---|
| below 30 s | **3.0** | three times too many placeholders — the crash we hit |
| 30–300 s | **1.0** | the two samplers agree; nothing fails |
| above 300 s | **0.5 and falling** | too *few* placeholders — fatal in the other direction |

Every fixture in the gate table is under 30 s, so the gates exercise the worst band and not the
lucky one. Filed with the issue,
[vllm#55644](https://github.com/vllm-project/vllm/issues/55644); the fix and its unit test are
[vllm#55647](https://github.com/vllm-project/vllm/pull/55647) `[measured-here]`.

---

## 5. Host memory on the API-server rank

The one measurement that took three rounds. All figures from a 5 s sampler over `/proc/meminfo` and
`vmstat`, rank 0 only; the other two nodes swapped **zero** in every window of every round.

| Round | `mm_processor_cache_gb` | Idle MemAvailable | Under load |
|---|---|---:|---|
| 1 | 4 (default) | **1.10 GiB** | swap-out 1,252.1 MiB over the window |
| 2 | 1 | **1.84 GiB** | swap-out 2,554.6 MiB over the window |
| 3 | **0** | **3.42 GiB** | swap-out 412.6 MiB (video gates), 773.3 MiB (1M token) |

Container resident set, round 3: rank 0 **7.60 GiB idle / 8.07 GiB under load**, against 6.75/6.85
and 6.54/6.63 on the other two nodes — **about 1.2–1.4 GiB more**, the API server plus the multimodal
frontend. This figure could not be taken in rounds 1 and 2 at all, because the arm crashed and took
the container with it.

### 5.1 The promotion boot's worst-case stress

Deliberately adversarial on rank 0, at `gpu-memory-utilization` **0.88** `[measured-here]`:

| Window | Duration | MemAvailable min | swap-in | swap-out | SwapUsed start → end |
|---|---:|---:|---:|---:|---|
| 1M-token request | 677 s | **1.47 GiB** | 67.0 MiB | 281.6 MiB | 1.31 → **1.57 GiB** |
| C1 + C8 speed rounds | 220 s | 1.70 GiB | 5.1 MiB | **0** | 1.56 → 1.56 GiB |
| C8 **with** a 4-image + 2-video request concurrently | 60 s | 1.78 GiB | 0.1 MiB | **0** | 1.56 → 1.56 GiB |

Four acceptance criteria, in order: no error, OOM or engine death (`EngineDeadError|Traceback|CUDA
out of memory` count **0**, `/health` 200); MemAvailable never near 1 GiB (**minimum 1.47 GiB**); no
slowdown outside the bands (C1 +0.1 %, C8 +1.5 %, TTFT at production values); and swap **settling**
rather than growing (1.31 → 1.57 GiB and then flat, with the two later windows paging out nothing).
All four met, so **0.88 ships and no KV pool was given up**. A 0.87 rung was prepared and never run.

**The measured price of the concurrent window, which has no reference value.** The text arm ran at
**173.53 tok/s** against 198.8 alone (**−12.7 %**), TTFT 0.868 s against 0.823; the multimodal
request answered correctly in 14.42 s. This is contention — `--max-num-seqs 8`, so a ninth request
queues and a 5,991-token prefill takes slots — and **how much of it is multimodal rather than simply
a long ninth prefill was not attributed** `[not tested]`. Production configuration 12 cannot sit this
exam at all: it refuses the request.

---

## 6. Temperature and power

Peak values over the load window, round 3 `[measured-here]`:

| Node | GPU peak | Power peak | SM clock peak | Throttle |
|---|---|---|---|---|
| head | **84 °C** (1M-token gate) | 81.4 W | 2,554 MHz | **0x0 — no throttling** |
| worker-1 | 78 °C | 79.8 W | 2,593 MHz | **0x0** |
| worker-2 | 83 °C | 81.6 W | 2,476 MHz | **0x0** |

Rounds 1 and 2 reported 49–53 °C, and that was **misleading**: no real load ever ran in those
windows, because the engine died on the first video. Round 3's 1M-token prefill and C8 rounds worked
the GPUs properly and the throttle bit never lit even at 84 °C. (`0x4` appears only in idle samples;
it is the "GPU idle" flag.)

---

## 7. The promotion boot

| | |
|---|---|
| Sidecar dump | 50.66 GiB in 22 shards per node, 86–95 s at 570–631 MB/s; main model load 165–172 s per rank; **53 GB per node** on disk |
| Sidecar contents | **6,454** names against configuration 12's 5,858 — **+596 = the tower**, +0.42 GiB. Drafter sidecar 94 names / 2.04 GiB in both |
| `visual.*` in the manifest | 99 `.trellis`, 99 `.mul1`/`.suh`/`.svh`/`.bias`, 100 `.weight` (all *norm*), 1 `cos_sin_cache`, and **no `attn.qkv.weight`** |
| Load boot, by hand | **247 s** to `/health` 200 (dump boot 454 s; round 3 without a sidecar 365 s; configuration 12 ~251 s) |
| Restore | `restored 6454 tensors, 50.66 GiB from 22 shards in 55.8 s (974 MB/s)` |
| Environment switch | exactly **6 lines** against configuration 12; a diff against the kept backup reported **0 unexpected changes** on all three nodes. `gpu-memory-utilization` and `FASTLOAD_MODE` did **not** move |
| systemd unit | **unchanged** and not reloaded — `ExecStart` points at the launcher and `TP3_DIR` comes from the environment file |
| Boot under the unit | `systemctl start` on all three (2 → 1 → 0) → `/health` 200 at **244 s**; units `active + enabled` on all three; KV pool **7,033,057** (98.7 % of configuration 12, inside band); gates **8/8**, text 10/10 and 12/12; K2 four images correct; K4 video correct |
| **Whole-cluster reboot**, all three together, unit enabled | `/health` 200 at **318 s** by the wall clock from the `reboot` command — configuration 12 read **311 s**, and this stack's three earlier trials span 311–315 s. KV pool **7,016,528** (98.5 %). Gates **5/5**, text 10/10 and 12/12 |
| Memory profile after the reboot, rank 0 | consumed **55.12 GiB** (configuration 12: 55.28), peak activation **1.66 GiB** — *equal*, not the 1.69 of the sidecar-free trial boot — CUDA-graph pool 0.0, indexer workspace 513 MB, `cudagraph_mode=NONE` |
| Disk after promotion | **546–547 GB free per node** with both sidecars present. The text-only sidecar was **not deleted**, so the rollback is one environment file and no dump boot |

**Three vision boots against configuration 12's own boot-to-boot range.** Configuration 12 read
7,024,793–7,126,721 across boots; the three vision boots read 7,143,250 (by hand), 7,033,057 (unit)
and 7,016,528 (after the reboot). All three sit inside that spread, which is the honest form of the
claim: **the tower has no measurable KV-pool cost**, rather than a single flattering ratio.

**The `HAREM-VISION: tower loaded` line is absent on a fast-load boot**, by mechanism: VS3's audit
lives inside the tower's `load_weights` and `harem_fastload` skips `load_weights` in load mode. The
patch was **not** edited to restore it — its hash is part of the sidecar identity, so editing it
would have invalidated the sidecar just written and shipped a different tree from the one the gates
passed. The replacement evidence chain, and the five-scenario counter-test that proved the rewritten
K0 gate can still fail, are in [docs/18](../../docs/18-vision-at-three-ranks.md) §8.1.

---

## 8. Three tooling faults, all ours

Recorded because each one cost an engine window, and because
[docs/09](../../docs/09-measurement-protocol.md) gained a rule from the last of them.

1. **VS1's anchor was written against the stock file** while the prelude patches that block earlier.
   Zero matches, exit 21 on all three ranks — fail-closed working correctly. The model-free gate said
   PASS because it chained full-scope → vision and never reproduced the boot's patch order.
   `verify-cpu.sh` now runs the full prelude order.
2. **The same gate returned 0 through a pipe.** Every check ends in `| tail -1` and the script had no
   `pipefail`, so a run printing `[vision] FAIL` exited 0. Fixed, then confirmed by watching a
   failing gate actually fail.
3. **A header-derived count asserted as an invariant.** "48 dead fused tensors must be dropped" is
   true of the safetensors headers and false of the load, which delivers 0 or 1. It killed a boot
   whose tower was entirely correct. Counts are now reported; the invariant that is asserted is
   99 EXL3 linears / 0 unquantized / no dense fused `qkv.weight`.

And the fourth, which is the expensive one: **the install-day gates only checked weight loading.**
The tower loaded perfectly and the engine still died on the first video, because nothing checked
geometry. That gate now exists, is model-free, takes under a second, and would have caught the fault
before any of the three engine windows.

Two harness faults, neither a regression of the arm: K7b built its 1M-token prompt from a
500-line probe (3-digit line numbers) and sent 20,000 lines (5-digit), overshooting the limit by one
token twice — it now measures with the server's own `/tokenize` endpoint; and K7b's `max_tokens=64`
was spent entirely in the reasoning channel, leaving `content` empty and the gate reporting "needle
not found" on a correctly served request. Raised to 256.
