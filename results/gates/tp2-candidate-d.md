# TP=2 candidate D — the vision tower and the two backports at two ranks, one boot

**8 September 2026 `[measured-here]`.** The raw record behind
[docs/15](../../docs/15-tp2-track.md) §5.10 and [docs/19](../../docs/19-vision-at-two-ranks.md). Two
DGX Spark (GB10) nodes — `head` (rank 0, serves the API) and `worker-1` — one dump boot, one
fast-load boot through the autostart unit, and one control boot with the two knobs off.

**Settings.** TP=2, **expert parallelism off**, image `exl3-zeus:754421f`, the two-node tree
([`tracks/tp2/patches/`](../../tracks/tp2/patches/README.md)) plus `patch-vllm-tp3.py`,
`patch-vision-tp3.py`, `patch-prefixhit-tp3.py`, `patch-kpooltail-tp3.py`, launcher
[`scripts/start-tp2full.sh`](../../scripts/start-tp2full.sh) with its `MemAvailable ≥ 112 GiB` settle
gate, `turboderp/GLM-5.3-Flash-exl3` at 4.05 bpw, KV `fp8` and an fp8 draft cache, DFlash2 at k=7,
`--attention-backend CUSTOM`, `--block-size 256`, `HAREM_SW_BLOCK_SIZE=256`, `--max-num-seqs 8`,
`--max-num-batched-tokens 2048`, `--max-model-len 1000000`, **`gpu-memory-utilization 0.85`**,
`HAREM_INDEXER_WS_MODE=bound`, `HAREM_DISABLE_PERSISTENT_TOPK=1`, `NCCL_MAX_NCHANNELS=8`, mesh plugin
with both cables and `NCCL_PTR_CUDA`, per-rank fast-load sidecar, warm `CUDA_EXL3_TUNE_CACHE`,
`--safetensors-load-strategy eager`, `--no-enable-flashinfer-autotune`, temperature 0, thinking on at
reasoning effort **low**, `max_tokens` 256, prompts `scripts/hizset-v2.jsonl`. Vision:
`HAREM_VISION=1`, `CUDA_EXL3_PACKED_MAPPING={"qkv_proj":["q_proj","k_proj","v_proj"]}`,
`LANGUAGE_MODEL_ONLY=0`, `--mm-encoder-tp-mode data`,
`--mm-processor-kwargs {"max_pixels":12544000,"max_image_tokens":8000}`,
`--limit-mm-per-prompt {"image":4,"video":2}`, `--mm-processor-cache-gb 0`. Backports:
`HAREM_PREFIX_HIT=1`, `HAREM_KPOOL_TAIL_FIX=1`.

---

## 1. The boots

| | |
|---|---|
| Model-free gate before any boot | `verify-cpu.sh` ran the **whole** two-node prelude order in a throwaway CPU container: prefix-hit and kpool-tail applied, `[fullscope] applied: S1 S2 S3`, `[vision] applied: VS1 VS2 VS3 VS4 VS6 VS7`, `[vision-map] PASS 99/99`, `[vision-names] PASS 959`, `[video-geom] PASS 5 geometries`, `preflight RESULT: PASS — arithmetic is sound for tp=2` |
| Dump boot (attempt 1) | **died at 4 minutes** on the drafter: `qwen3_dflash.py:186 assert self.total_num_kv_heads % tp_size == 0`. The shared drafter directory carried the three-node 36/9 pad. [docs/14](../../docs/14-troubleshooting.md) §10.5 |
| Dump boot (attempt 2) | **1,013 s.** Target model 75.59 GiB in 29 shards, 570 s on rank 0 and 653 s on rank 1; drafter 2.71 GiB in 2 shards, 30 s. KV pool on the dump path 2,353,571 |
| Sidecar | **79 GB per rank, 32 files** (candidate C: 78 GB, 32 files) |
| Fast-load boot, **through `harem-exl3-tp2.service`** | **/health 200 at 280 s.** Weight restore 84.2 s at 964 MB/s; drafter 2.6 s at 1,110 MB/s |
| KV pool | **2,585,714** tokens, maximum concurrency 2.59× at 1M |
| Available KV memory | rank 0 **20.15 GiB**, rank 1 **19.54 GiB** |
| CUDA graphs | **19 PIECEWISE + 8 FULL + 8 DFlash2 FULL** — still captured at two ranks with a tower in the model |
| Control boot (knobs off, same tree, same sidecar) | 297 s by hand, KV **2,592,857** |

Boot-log evidence lines, all present: `patch-prefixhit: applied` ×2, `patch-kpooltail: applied` ×2,
`HAREM-TP3 prefix-hit: 1 drafter group(s) flagged is_eagle_group; target groups left unflagged`,
`[vision] applied: VS1 … VS7`, the three vision gates, and
`HAREM-VISION: tower loaded, data_parallel=True, vit_attn_backend=FLASH_ATTN, EXL3 linears=99,
unquantized linears=0`. `language_model_only` does **not** appear. The EXL3 module audit reads
**302 EXL3 / 113 bf16** against candidate B's documented 203/113 — the difference is the tower's 99.

---

## 2. Text gates, cold

| Gate | Result |
|---|---|
| `scripts/correctness-probe.py` | **10/10** |
| `scripts/code-exam.py` | **12/12**, first attempt (the `matrix` item, a measured flake on this stack, did not fail) |
| Tool-call gate | **8/8** |
| Needle-lite, 6 depths, ~55K prompts, three runs | **6/6 · 6/6 · 6/6** |

---

## 3. The six vision gates

One request at a time, the fixtures of [docs/18](../../docs/18-vision-at-three-ranks.md) §6.

| Gate | Result |
|---|---|
| K2, one image, four fixtures | **4/4** — `Red circle`, `Blue square`, `Green triangle`, `ELEPHANT` |
| K3, four images in one request | **PASS** — all four, **in order**, 2,612 prompt tokens |
| K4, one video | **PASS** — colour, shape and motion, 1,722 prompt tokens |
| K4b, two videos in one request | **PASS** — both, in order, 3,415 prompt tokens |
| **K5, 4 images + 2 videos** (the declared limit) | **PASS** — all six items, 5,991 prompt tokens |
| K6, three videos | **HTTP 400** (`At most 2 video(s) may be provided in one prompt`), engine alive afterwards |

---

## 4. The prefix-hit A/B — one environment line apart, same tree, same sidecar

**Block granularity, measured rather than copied:** gcd(4,608, 55,296) = **4,608 tokens** at two
ranks, against 3,328 at three. Ceiling = `floor((n − 1) / G) × G / n`.

| Scenario | n tokens | knobs **off** | knobs **on** | ceiling | on / ceiling |
|---|---:|---:|---:|---:|---:|
| Exact repeat | 8,008 | **0 (0.0 %)** | **4,608 (57.54 %)** | 57.54 % | **100 %** |
| Four-turn agent, every turn | 8,024–8,089 | **0.0 %** | 57.28 % | 57.28 % | **100 %** |
| Exact repeat | 59,910 | 55,296 (92.30 %) | 55,296 (92.30 %) | 99.99 % | 92.3 % |
| Repeat TTFT, 8K | | 5.60 s | **2.55 s** | | −54 % |
| KV pool | | 2,592,857 | 2,585,714 | | −0.28 % |
| Draft acceptance (probe filler, prose regime) | | 23.08 % | 23.08 % | | equal |
| Preemptions | | 0 | 0 | | |

**At two ranks the defect costs the whole hit rather than half of it.** The lost quantity is one
block either way; one block is 4,608 tokens here, and an 8,008-token prompt has exactly one block of
hit available.

**The 59,910-token row does not move at either rank count.** That prompt ends **6 tokens** past
59,904, which is 18 × 3,328 *and* 13 × 4,608. The three-node remainder hypothesis, tested here at a
different granularity, holds.

---

## 5. The cached-path equality test — 24/24

24 needle-style prompts, eight at each of three size classes, eight depths, each with its own filler
seed; each asked **cold and then repeated byte-for-byte**, with the prefix-cache counters read either
side so the repeat is proved to be a hit. Temperature 0, reasoning effort `low`, thinking on.

| Size class | n tokens | cold hit | repeat hit | cold → repeat | identical | correct |
|---|---|---|---|---|---|---|
| ~11K | 11,213–11,366 | 0.0 % | **81.1–82.2 %** (the ceiling) | 7.9 s → **1.7 s** | 8/8 | 8/8 |
| ~83K | 82,859–83,359 | 0.0 % | **94.2–94.4 %** | 55.3 s → **3.8 s** | 8/8 | 8/8 |
| ~177K | 176,763–177,436 | 0.0 % | **98.7–99.1 %** | 118.3 s → **2.0 s** | 8/8 | 8/8 |

| | |
|---|---|
| identical and correct | **24/24** |
| differing but both correct | 0 |
| wrong on either pass | 0 |
| repeat that did not hit the cache | 0 |
| mean hit ratio | cold **0.0 %**, repeat **91.7 %** |

---

## 6. Speed, three sweep rounds

Aggregate output tok/s on `scripts/hizset-v2.jsonl`, medians of three rounds on a warm tuner cache.
Candidate C's column is **a different session**, so read the bands, not the signs.

| | Candidate C | **Candidate D** | Δ | band |
|---|---:|---:|---:|---|
| C1 aggregate | 60.08 | **59.45** | −1.0 % | ±4 % |
| C1 per stream | 65.96 | **64.45** | −2.3 % | ±4 % |
| C2 aggregate | — | 81.37 | — | |
| C4 aggregate | — | 115.06 | — | ±9 % |
| C6 aggregate | — | 134.35 | — | |
| C8 aggregate | 157.71 | **155.47** | −1.4 % | ±3 % |
| TTFT median, C1 / C8 | 0.381 / 1.054 s | 0.375 / 1.080 s | equal | |
| Acceptance · accepted tokens per step, C1 | ~60.4 % | 60.33 % · 5.22 | equal | ±2 pt |
| Acceptance · accepted tokens per step, C8 | ~61.3 % | 62.59 % · 5.38 | equal | ±2 pt |
| Prefill, 3 fresh unseen ~8.4K prompts | 1,414 | **1,413** and **1,403** on two runs | equal | ±3 % |

Per-round spread: C1 58.78–60.07 (2.2 %), C4 111.63–116.71 (4.6 %), C8 152.44–155.75 (2.2 %) — all
inside the declared bands.

**One reading was taken wrong and is recorded rather than dropped.** The first fresh-prefill run read
**1,120 tok/s**, taken while another probe was hitting the same engine. Alone, twice: 1,413 and
1,403. One measurement at a time, or the number is fiction
([docs/09](../../docs/09-measurement-protocol.md)).

**`prefill-7k.py` is no longer a prefill measurement on this configuration.** It times the *second*
copy of one prompt, and with `HAREM_PREFIX_HIT=1` that copy now comes largely out of the cache: it
read **2,602 tok/s** against a fresh-prompt 1,413, and **1,276 tok/s** on a run where the cache had
been flushed by other traffic. Use [`bench/prefill-fresh.py`](../../bench/prefill-fresh.py).

---

## 7. The K-pool soak, and host memory in the worst case

Four concurrent 4,096-token generations, host memory sampled every 5 s on both nodes throughout
(582 samples):

| | |
|---|---|
| Generations | 4/4 filled their budget, all coherent, **no error**, engine alive; 16,384 tokens in 3.7 min |
| Rank 0 `MemAvailable` floor | **4.97 GiB** |
| Rank 0 swap used / pages in / pages out | 0.059 GiB flat / +2 / **0** |
| Rank 1 | 7.28 GiB floor, 0.001 GiB swap, zero paging |

Then the adversarial case — **a 4-image + 2-video request fired during a C8 text sweep**, which puts
the multimodal frontend and eight decode streams on rank 0 at once:

| | |
|---|---|
| Text arm during the multimodal request | **141.03** tok/s against 155.47 alone — **−9.3 %** (three ranks: −12.7 % on the same test) |
| TTFT median in that window | 1.073 s; acceptance 62.37 % |
| Multimodal request | correct, **10.71 s**, 5,991 prompt tokens |
| Rank 0 `MemAvailable` floor | **4.90 GiB** |
| Rank 0 swap used / pages in / pages out | 0.059 GiB flat / **0** / **0** |
| Rank 1 | 7.28 GiB floor, 0.001 GiB swap, zero paging |

**Harm means slowdown, errors or leaving the safe zone — not the presence of swap.** None of the
three happened. The −9.3 % is contention and it is attributable: `--max-num-seqs 8` means a ninth
request queues and a 5,991-token prefill takes slots. **Not attributed:** how much of it is
*multimodal* rather than "a ninth request with a long prefill" — a text request of the same prefill
length was not run `[not tested]`.

---

## 8. What is not here

One boot of candidate D, so no boot-to-boot spread at two ranks and no way to separate the −4.0 %
pool row into "the tower" and "this boot". No MMLU. No reboot test. No sliced vision tower. No
sustained multimodal load. [docs/15](../../docs/15-tp2-track.md) §6 is the full list with reasons.

**One raw file was lost and it was our doing.** The concurrent-stress harness writes its JSON beside
itself, and running it at two ranks overwrote the three-node run's copy. The three-node numbers
survive in [docs/18](../../docs/18-vision-at-three-ranks.md) §8.2 (173.53 tok/s against 198.8,
−12.7 %, multimodal request 14.42 s); the raw file does not `[measured-here, raw lost]`.
