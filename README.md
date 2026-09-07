# GLM-5.3-Flash (EXL3 4bpw) on NVIDIA DGX Spark — two- and three-node recipes (vLLM + cuda-exl3)

> **Most recipes stop at the flags. This one goes below the engine.** When three nodes ran slower
> than two, we did not swap a setting — we found the kernel dispatch bug behind it, together with the
> `cuda-exl3` author, and it is fixed upstream. We carry our own patch in the MoE combine kernel, we
> patched the NCCL transport plugin so the second cable on every ConnectX-7 carries traffic, we built
> the loading path that puts a **fully quantized** checkpoint into dimensions vLLM had padded (its
> kernel side written by the author to our specification), we found the page counter that was
> hiding 45 % of the KV pool, and we turned a vision tower back on that our own recipe said could not
> run at three ranks — which meant finding the upstream frame-sampler mismatch that had been killing
> the engine core on the first video request. Every change was measured on the hardware, and every number we got wrong
> is withdrawn in public. Assembling a working cluster from other people's parts is where this work
> starts, not where it ends.

> **Status: release.** Numbers, flags and patches are current as of **7 September 2026** and describe
> **production configuration 13** — configuration 12 with the **vision tower on**, which is what our
> three nodes serve, start at boot and were rebooted into as a whole cluster with the gates read
> afterwards. Configuration 12 is configuration 11 at `gpu-memory-utilization` 0.88 with the
> sparse-indexer K-gather workspace bound to its real ceiling, and every speed, pool and quality
> figure below is measured on it and holds on 13 — the tower is not on the decode path and the three
> vision boots land inside configuration 12's own boot-to-boot spread
> ([docs/18](docs/18-vision-at-three-ranks.md)). Every analysis section is configuration 9's and
> applies unchanged, because 9 through 13 differ by a memory fraction, two guard patches, one buffer
> size and the tower. The stack is still moving and this file is overwritten in place when it does.
> Sections marked *open* in [docs/11-open-issues.md](docs/11-open-issues.md) are the honest edge of
> what we know; [docs/11 §1](docs/11-open-issues.md) is what we published and then had to withdraw,
> and [audit/](audit/README.md) §6 indexes it. **Read the retractions before you quote a number.**

## How many nodes do you have?

One question decides which half of this repository is yours, and
**[docs/00 — Start here](docs/00-start-here.md)** answers it in a page. Every `docs/NN` page carries
an **Applies to** badge on its first line, and [`tracks/`](tracks/README.md) holds the files that
differ between the two arrangements — the environment templates, the patch trees and the autostart
units — so they cannot be mixed by accident.

| You have | What this repository gives you |
|---|---|
| **3 DGX Spark** | The **TP=3 track**: the production recipe, the quick start below, [`tracks/tp3`](tracks/tp3/README.md) |
| **2 DGX Spark** | The **TP=2 track**: [docs/15](docs/15-tp2-track.md) and [`tracks/tp2`](tracks/tp2/README.md). At two ranks nothing needs padding, so it is a *shorter* recipe — fourteen patch files against twenty-three — rather than a cut-down one |
| **1 DGX Spark** | No serving recipe: 153.8 GiB of weights against 121.6 GiB of unified memory. Still yours — the image build, the GB10 kernel fixes, the measurement protocol, the model-free benches and the failure index. [docs/00 §1](docs/00-start-here.md) |
| **4 DGX Spark** | Nothing measured `[not tested]`. The padding and expert-parallel arithmetic, the cabling problem and what we would want reported are in [HELP-WANTED.md](HELP-WANTED.md) §1 |

### Three nodes — production configuration 12

Three DGX Spark nodes, one 4-bit EXL3 checkpoint, realistic prompts, temperature 0, reasoning effort
`low`, 6 September 2026 `[measured-here]`. Speed is the pool of six sweep rounds over **two boots** of
this configuration:

| | |
|---|---|
| Single-stream decode (C1) | **69.7** tok/s aggregate (**75.6** per stream) |
| Aggregate at 8 concurrent streams (C8) | **196.1** tok/s |
| Prefill, fresh unseen ~8K prompts | **1,744** tok/s |
| Sustained aggregate under load, stopwatch, **vision enabled** (7 September) | decode **205** tok/s with 12 code requests submitted at once (8 running + 4 queued; one-second peaks to 261) · prefill **1,914** tok/s with 24 fresh ~7K prompts in flight — 60 s counter deltas, nothing else on the cluster; [`results/speed/stopwatch-production-7sep.md`](results/speed/stopwatch-production-7sep.md) `[measured-here]` |
| KV pool at `max_model_len` 1,000,000 | **7,041,322** tokens at `gpu-memory-utilization 0.88` — about 7.0 concurrent 1M-token requests |
| Quality gates | correctness probe **10/10**, code exam **12/12** cold and warm; tool-call gate **8/8**; needle-lite **6/6** at 64K and 128K; MMLU sample (1,995 q) **86.47 ±0.74**, carried from production 9 `[not tested]` on this configuration |
| Quality benchmarks, against the NVFP4 sibling recipe | GSM8K **97.5 %** (sibling 94.0), IFEval **80.0 % prompt / 86.0 % instruction** (78.9 / 85.1), tool-eval-bench **85.5 ±1.3** (**87.8 ±0.9** — behind, and it is four scenarios out of 88; the largest is a grader ordering rule, the template explanation was tested and refuted, and the build/checkpoint confound is still open), 6–7 September 2026 `[measured-here]`. The 1M needle and the full MMLU were **deferred on time** and are not run `[not tested]`. [`results/gates/quality-battery-production-12.md`](results/gates/quality-battery-production-12.md) |
| Long-context stress | one **969,468-token** request correct in 569.6 s; **eight concurrent ~128K lanes** 8/8, 640,904 prompt tokens in 227.5 s |
| Cold boot, `docker run` → API ready | **272 s** (the one-off dump boot that writes the sidecar is 590 s) |
| **Boot from power-on**, all three nodes rebooted together, autostart unit enabled | `/health` 200 at **311 s** by the wall clock, timed from the reboot command; **318 s** with the vision tower ([systemd](systemd/README.md), [`results/boot/boot-ledger.md`](results/boot/boot-ledger.md)) |
| **Multimodal: 4 images + 2 videos per request** (configuration 13, 7 September) | KV pool 7,143,250 / 7,033,057 / 7,016,528 across three boots — **inside configuration 12's own 7,024,793–7,126,721 spread**; C1 **+0.1 %**, C8 **+1.5 %**, TTFT and draft acceptance unchanged; ten gates out of ten, text gates 10/10 and 12/12. The tower is 6-bit EXL3 in the checkpoint and costs 0.416 GiB per rank. [docs/18](docs/18-vision-at-three-ranks.md), [`results/gates/vision-gates-tp3.md`](results/gates/vision-gates-tp3.md) `[measured-here]` |

### Two nodes — the TP=2 production candidate

Two DGX Spark nodes, TP=2, **expert parallelism off**, the same full-scope checkpoint,
`gpu-memory-utilization` **0.85** — the three-node rung is not transferable and the two-node
ladder has never been derived — same harness and same protocol, 6 September 2026 `[measured-here]`.
**Candidate C** (the indexer-workspace bound carried to two ranks, [docs/15 §5.9](docs/15-tp2-track.md)) is
the published configuration since 6 September evening; its speed rows are a different session from
candidate B's and are inside the bands:

| | |
|---|---|
| Single-stream decode (C1) | **60.08** tok/s aggregate (**65.96** per stream) |
| Aggregate at 8 concurrent streams (C8) | **157.71** tok/s |
| Prefill, fresh unseen ~8.4K prompts | **1,414** tok/s |
| KV pool at `max_model_len` 1,000,000 | **2,692,857** tokens — about 2.7 concurrent 1M-token requests (candidate B: 2,128,571; **+26.5 %** from the workspace bound) |
| Quality | correctness probe **10/10**, code exam **12/12** cold and warm, tool-call **8/8**, needle-lite **6/6**, one ~1M-token request and 8 × ~128K concurrent lanes correct; MMLU sample (1,995 q) **86.02 ±0.75** is candidate B's, not re-run on C |
| Cold boot, fast-load | **272 s** (the one-off dump boot that writes the sidecar is 956 s) |
| Autostart unit → `/health` 200 | **261 s**, `systemctl start` on both nodes — **candidate B, not re-run on C**. Candidate C's unit is the same unit pointing at the same launcher with a different env file and sidecar, and was not re-timed under it. **No reboot test yet** `[not tested]` |

**Two ranks are 80–86 % of the speed on 38 % of the pool, at quality that is inside one error bar.**
The side-by-side comparison, and why the third node wins on latency as well as on memory, is further
down this page and in [docs/15](docs/15-tp2-track.md) §7. A second candidate, on the
routed-experts-only checkpoint, is kept for anyone who already has those 164 GB: it is slower on every
concurrency with a 1,500,000-token pool ([docs/15](docs/15-tp2-track.md) §5).

A reproducible recipe for serving **`zai-org/GLM-5.3-Flash`** as an **EXL3 4-bit** checkpoint on three
DGX Spark (GB10) nodes with vLLM and the `cuda-exl3` kernels: the two-layer image build, expert
parallelism over 288 experts, the TP=3 shape padding, the DFlash2 speculative-decoding port, the
kernel bugs we found and what fixed them, the NCCL mesh cliff, the second cable nothing was using,
the PCIe wall behind it, the KV-pool surgery, a 251-second cold boot, a measured breakdown of where a
prefill and a decode step actually go, the loader work that got a **fully quantized** checkpoint to
load — and then to load into dimensions vLLM had padded, which is what put it in production — and
what is still broken. Written so that a person **or their AI coding agent** can follow it step by
step.

This is the EXL3 sibling of our NVFP4 recipe,
[`NNNtrance/GLM-5.3-Flash-NVFP4-TP3-3x-DGX-Spark`](https://github.com/NNNtrance/GLM-5.3-Flash-NVFP4-TP3-3x-DGX-Spark),
which serves the same model on the same three nodes through a different quantization path. The two
share a cluster, a fabric and a set of memory rules, but **this repository is self-contained, at
both node counts**: [docs/00](docs/00-hardware-and-os.md) is a complete environment record down to
firmware, the hotplug fix, the PCIe ceiling and every OS-level setting we did and did not change, and
[docs/14](docs/14-troubleshooting.md) indexes all 86 failures we hit by symptom with the exact log
line, each tagged with the node counts it can happen on. You do not need the sibling to follow
either track. **Each track is complete on its own** — its own environment template, launcher, patch
tree, fast-load sidecar and autostart unit, in [`tracks/`](tracks/README.md) — so the two-node recipe
is not a cut-down of the three-node one and no page asks you to mentally subtract a rank. At two
ranks nothing needs padding, so it is genuinely *shorter*: the patch tree is fourteen files rather
than twenty-three.

> **About the name "HAREM".** HAREM is simply the name we gave our three-node setup. It is hardcoded
> in several places in the stack — patch markers (`HAREM-TP3`, `HAREM-GB10-TOPK`), environment
> variables (`HAREM_SW_BLOCK_SIZE`, `HAREM_DISABLE_PERSISTENT_TOPK`, `HAREM_FASTLOAD_*`,
> `HAREM_EP_ZERO_MODE`), function names (`_harem_*`), one module (`harem_fastload.py`), some image
> tags and log lines. You can keep them. If you rename them, grep the whole repository first: several
> of these strings are matched exactly by the patch scripts, which fail closed when an anchor stops
> matching.

## Headline results

The two at-a-glance tables are above, under [How many nodes do you have?](#how-many-nodes-do-you-have)
This section is the three-node track in full: what each configuration changed, what it cost, and the
measurements that overturned four of our own targets.

Production 10 is production 9 with `gpu-memory-utilization` 0.80 → **0.83**: **+8.7 % of KV pool**, no
speed number outside its band, gates full, swap flat under load.

**Production 11 is production 10 with exactly two changes**, taken in one boot on 6 September:
`gpu-memory-utilization` 0.83 → **0.87** and the **sm_12x correctness set**
([docs/11](docs/11-open-issues.md) §2.27) behind `HAREM_SM12_ITEMS=pdl,kpool`. Against a same-session
production 10 reference: **KV pool +12.1 %**, C1 **+0.6 %**, C8 **+1.8 %**, prefill equal, every gate
full cold and warm on both boots, tool-call 8/8, needle-lite 6/6, and swap traffic **exactly zero** on
all three nodes on the clean boot.

**Production 12 is production 11 with exactly two changes**, again in one boot, later the same day:
`gpu-memory-utilization` 0.87 → **0.88** — the rung the ladder below measured, passed and *declined* —
and the **sparse-indexer K-gather workspace bound**
([`tracks/tp3/patches/indexer-workspace/`](tracks/tp3/patches/indexer-workspace/)) behind
`HAREM_INDEXER_WS_MODE=bound`, which takes a buffer that upstream sizes at `40 × max_model_len`
entries from **5,036.40 MB to 513.00 MB**. Against a same-session production 11 reference:
**KV pool +10.3 %** (7,041,322 tokens on the reboot boot, 7,170,798 on the best of three), C1
**+0.08 %** and C8 **−0.08 %** pooled over six rounds and two boots — no measurable speed cost —
every gate full cold and warm, tool-call 8/8, needle-lite 6/6, one **969,468-token** request correct,
**eight concurrent ~128K lanes** 8/8, and swap *in* exactly zero in every sample on every node.
Everything below and every analysis page is production 9's and applies unchanged.

**What the second change had to survive to ship, and did.** A workspace bounded at 2.03× the
scheduler's ceiling is only safe if the ceiling is right, so promotion was gated on the two cases that
can actually crowd that buffer — a single ~1M-token prefill and eight long-context prefills at once —
and on the buffer never growing after `lock_workspace()`. After both stress cases every rank still
logs **exactly one** workspace resize, `0.00 → 513.00 MB`, and zero assertions: none of the patch's
four safety layers fired. `VLLM_DEBUG_WORKSPACE=1` is carried in production for exactly this reason —
it is upstream's own variable, it costs three INFO lines a boot and nothing in the hot path, and it is
what makes that check readable from the engine's own log on every future boot rather than only in an
A/B.

**Production 13 is production 12 with the vision tower on**, promoted 7 September. Six environment
lines and six patch anchors; nothing else moves — same image, same checkpoint, same memory fraction,
same batching, same sampling. The recipe had said this was impossible at three ranks and had shipped
a patch to *prevent* the tower being built; that claim is retracted in
[docs/03](docs/03-tp3-padding-and-sidecars.md) §3 and replaced by
[docs/18](docs/18-vision-at-three-ranks.md). **Against a same-session production 12 reference:
4 images and 2 videos per request, C1 +0.1 %, C8 +1.5 %, TTFT and draft acceptance unchanged, ten
gates out of ten, a whole-cluster reboot at 318 s against 311 s, and a KV pool inside configuration
12's own boot-to-boot spread.** The tower is 6-bit EXL3 inside the checkpoint (0.557 GiB in total,
0.416 GiB per replicated rank), not the 1.05 GiB of BF16 this repository used to quote — that figure
described the other checkpoint and is withdrawn.

**And the thing that actually cost the three engine windows was not ours.** vLLM's GLM-5-Next port
runs a video through **two independent frame samplers** — the pixels through GLM-5-Next's, the prompt
placeholders through GLM-4.6V's — so a four-second clip produced 4,968 placeholders for 1,656
encoder rows and killed the engine core on all three ranks with
`Attempted to assign 1656 = 1656 multimodal tokens to 1728 placeholders`. It has nothing to do with
three ranks, EXL3, expert parallelism or anything else here: it kills **every** GLM-5-Next video
request. We found it on a CPU in under a second once we stopped guessing, fixed it in four lines, and
added a model-free geometry gate that would have caught it before the first boot. It is **not yet
filed upstream** — [docs/18](docs/18-vision-at-three-ranks.md) §12 is the draft.

**Two changes in one boot on purpose.** The memory fraction is not part of the fast-load manifest
identity but the patch files are, so adding the patches forces a dump boot (590 s, ~53 GB per node)
that the rung can ride along on for free ([docs/08](docs/08-fast-boot.md) §4).

**Before you read a few-percent difference anywhere in this repository as a result, read
[docs/10 §1.1](docs/10-results-and-roofline.md)**: on this stack C1 boot medians span 1.1 %, C8 2.5 %
and **C4 7.4 %** — and production 11 is the case in point. Its load boot read C4 at 135.0 against the
reference's 144.2, the single number that moved; the clean boot after a whole-cluster reboot read
**145.9**. The six-round spread at C4 on this configuration is **11.1 %**, against 2.5 % at C1 and
5.3 % at C8. The low reading is boot noise and it is printed rather than smoothed. Production 12
repeated the lesson from the other side: its load boot read C6 at 167.8 and C4 at 140.8, its clean
boot 177.4 and 151.6 — 5.7 % and 7.7 % apart — while C1 and C8 pooled to within a tenth of a percent
of the reference. **C1 and C8 are the levels that carry a verdict on this stack.**

Settings for every row: image `exl3-zeus:754421f`, TP=3 + expert parallel, **full-scope** EXL3
weights (`turboderp/GLM-5.3-Flash-exl3` at 4.05 bpw), `kv-cache-dtype fp8` **and an fp8 draft cache**
(`HAREM_DRAFT_KV_DTYPE=fp8`), DFlash2 draft at k=7, `--block-size 256`, `HAREM_SW_BLOCK_SIZE=256`,
`--max-num-batched-tokens 2048`, `--max-num-seqs 8`, `NCCL_MAX_NCHANNELS=8`,
`gpu-memory-utilization 0.80`, per-rank pre-sliced sidecar, warm MLA tuner cache, mesh plugin with
both cables per peer and `NCCL_PTR_CUDA`, the launcher's memory settle gate, temperature 0, reasoning
effort **low**, 5 September 2026. This is **production configuration 9**. Speed is the median of
three sweep rounds — the persisted tuner cache is what makes three enough; on an image without it the
rule is still five rounds with two discarded ([docs/09](docs/09-measurement-protocol.md),
[docs/12](docs/12-tuner-cache.md)). The production-8 column is the pool of two same-day runs of the
same script, because that arm's documented run-to-run spread is about 7 %.

| What | **Production 9** | Production 8 | Notes |
|---|---|---|---|
| Quality gates | correctness probe **10/10**, code exam **12/12** | 10/10 · 12/12 | cold **and** after a full benchmark, both arms `[measured-here]` |
| MMLU sample (35 questions per subject, 1,995 q) | **86.47 ±0.74** | 86.4 ±0.7 | measured at TP=3 on the production checkpoint; 0.07 points apart, a tenth of either bar `[measured-here]` |
| Speed, realistic (12 short English code prompts) | C1 **69.9** tok/s total (**75.9** per stream) · C8 **197.2** total (28.6 per stream) | 56.9 / 62.4 · 175.4 / 26.7 | **+22.9 % / +21.7 % at C1, +12.5 % at C8** `[measured-here]` |
| Draft acceptance · accepted tokens per step | 61.9 % · 5.34 | 64.4 % · 5.50 | the gate is ≥60 %. **The 2.4-point gap is our own harness, not the checkpoint** — pooled by draft token over five concurrency levels and three boots it is **+0.18 points** `[retracted]`, see below |
| Prefill, fresh unseen ~8.5K prompts | **1,738** tok/s (warm repeated 7K prompt: 1,575) | 1,776 (1,537) | both inside the ±3 % equality band: **equal** `[measured-here]` |
| TTFT | **0.280** s at C1, **0.826** s at C8 | 0.344 / 0.906 | −18.6 % and −8.8 % `[measured-here]` |
| KV pool | **5,165,289 tokens** at `gpu-memory-utilization 0.80` | 4,696,969 | **+10.0 %**, about 5.2 concurrent 1M-token requests; read from a load boot with a settled baseline `[measured-here]` |
| Consumed memory per node (weights + non-torch) | **58.3 – 59.1 GiB** | 62.1 – 62.4 GiB | **−3.4 GiB**, which is where the pool came from `[measured-here]` |
| Cold boot, container start → API ready | **251 s** (weights 58 s) | 264 s (weights 73 s) | a fast-load boot; the one-off dump boot that produces the sidecar is 620 s `[measured-here]` |
| Free host RAM at rest / swap | 12.1 / 13.5 / 13.4 GiB · ~0.1 GiB | 12.3 / 13.5 / 13.5 · ~0.1 | rule: never below 4 GiB free `[measured-here]` |
| Speed by category, C1 | code **61.7** · math **79.6** · JSON **72.8** · prose **29.1** tok/s | code 47.9 · math 59.0 · JSON 57.7 · prose 22.4 | acceptance 46 / 58 / 54 / **13 %** — every category +30–35 %, and prose is still where a k=7 draft is wasted `[measured-here]` |
| Speed by category, C1, re-measured on **production 12** (7 September) | code **61.5** · math **76.2** · JSON **73.1** · prose **29.0** tok/s | code 61.7 · math 79.6 · JSON 72.8 · prose 29.1 (production 9, the row above) | acceptance 46 / 57 / 53 / **13 %**, unchanged. Every category inside its own round-to-round spread, so **+37.8 % of KV pool cost no category any tokens per second**. Prose is the slow row because the draft accepts one token in eight there, no flag changes that, and **the newest drafter revision does not change it either** — measured, same night: [docs/10 §1.2](docs/10-results-and-roofline.md), [`results/speed/category-speeds-production-12.md`](results/speed/category-speeds-production-12.md) `[measured-here]` |
| KV pool at the 0.83 rung (**production 10**) | **5,619,834 tokens** | — | +8.7 % again, one line changed, no speed number outside its band; swap flat `[measured-here]` |
| KV pool at the 0.87 rung (**production 11**) | **6,382,920 tokens** | — | **+12.1 %** against a same-session 0.83 reference `[measured-here]` |
| KV pool at 0.88 **with the indexer workspace bound** (**production 12**) | **7,041,322 tokens** | — | **+10.3 %** again, against a same-session 0.87 reference; three boots read 7,170,798 / 7,088,154 / 7,041,322 `[measured-here]` |
| Consumed memory per node, production 12 | **54.3 – 54.6 GiB** | 58.3 – 59.1 (prod 9–11) | the 4.42 GiB the workspace bound releases, measured on the far side `[measured-here]` |
| Host memory at 0.83 under load | `MemAvailable` **8–10 GB** per node · `MemFree` **0.9–1.2 GiB** · swap ~0.1 GB, flat | 12–13 GB `MemAvailable` at 0.80 | the rule and why it still passes: see below `[measured-here]` |
| Host memory at 0.87 under load (**production 11**) | `MemAvailable` **3.4 / 4.7 / 4.7 GiB** · swap traffic **0** | 8–10 GB at 0.83 | that headroom is the cost, and it is what 0.88 was first declined over `[measured-here]` |
| Host memory at 0.88 under load (**production 12**) | `MemAvailable` **1.5 / 3.4 / 3.4 GiB** on the arm that also ran both stress cases, **3.2 / 4.5 / 4.5** on the clean boot · swap **in** exactly 0 | 3.4 / 4.7 / 4.7 at 0.87 | the cost of the rung, taken deliberately; swap *out* 1,340 / 5 / 10 KiB in 8 of 353 samples against the reference's own 10 / 5 / 5 `[measured-here]` |

**The memory rule, said plainly, because production 11 sits against it.** On a GB10 the GPU shares
host memory, so the host's free share *is* the safety margin — but **`MemFree` is the wrong ruler for
it**, because most of what the kernel holds at that moment is reclaimable page cache. This page used a
2 GiB `MemFree` floor until 6 September and it has been replaced by two figures: **`MemAvailable`**,
and **swap traffic under load** (`si`/`so` per second from `vmstat`, summed over the benchmark
window). Swap *used* is a stock and sits at ~0.04 GiB at every rung including the failing one; it
discriminates nothing.

The ladder was then climbed rung by rung, in one session, against a same-session 0.83 reference
([`results/memory/ladder-6sep.md`](results/memory/ladder-6sep.md)):

| | 0.83 (prod 10) | 0.85 | 0.87 (prod 11) | **0.88 (prod 12)** | 0.90 |
|---|---:|---:|---:|---:|---:|
| KV pool | 5,674,931 | 6,016,528 | 6,363,636 | **6,542,699** | 6,870,523 |
| Swap traffic under load | ~0 | **0** | **0** | 4 KiB | **si 143 MiB + so 1,519 MiB** |
| `MemAvailable` min, head node | 8.35 GiB | 5.99 | 3.49 | **1.86** | 1.04 |
| Speed | reference | in band | in band | in band | **in band** |
| Verdict | — | pass | shipped (prod 11) | **pass, thin — shipped as prod 12** | **rejected** |

The 0.88 column is the ladder's own arm, measured on the production-10 stack; production 12 later
took that rung together with the workspace bound, and the pool it reads there is 7.04–7.17 M rather
than 6.54 M because the two gains add **in absolute tokens**. 0.89 was never measured, and the
workspace bound does not change that — it gives memory back on the GPU side, not the host side.

**Read the bottom two rows together.** At 0.90 every concurrency level is still inside its band while
the head node pages 1.5 GB out and reads 143 MB back for 250 of 598 seconds — a ladder judged on
tok/s would have taken it. What surfaces at the client is not a lower median but the arm's first
prefill going **5.0 → 9.8 s**. And **0.88 passed and was not taken on the day it was measured**: it leaves 1.86 GiB of
`MemAvailable` and 2.59 GiB of page cache on the head node, which is the budget anything running
beside the engine lives in; 0.87 leaves 3.49 GiB and gives up 2.8 % of pool for it. Every 1 % of the
fraction costs about **1.2 GiB** of host headroom here. It **was** taken, hours later, as production
12 — a decision about what the cluster is for, not a new measurement: with 0.88 in production the
budget for anything running beside the engine is about 2 GiB, and a profiling run has to stop the
engine first
([docs/11 §2.4](docs/11-open-issues.md), [docs/00 §11](docs/00-hardware-and-os.md),
[docs/07 §6](docs/07-kv-and-draft-page.md)).

**Production configuration 9 is a different checkpoint, and it is the largest single move this stack
has made.** Every configuration from 1 to 8 served `scope: glm53_routed_experts_only` weights:
4-bit routed experts beside a BF16 attention stack, shared expert and `lm_head`, and that dense half
measured **45.3 % of a single-stream decode step** — larger than everything in the `cuda-exl3` column
of our target table put together. Production 9 serves a **full-scope** checkpoint instead, so the
dense path is 5–6 bit EXL3 as well. Step time goes **88.2 → 70.3 ms**, and the arithmetic is the
result rather than the tok/s: acceptance and accepted length moved the *wrong* way by about 3 % and
the arm still wins by 22 %, so none of the gain is drafter behaviour
([13](docs/13-full-scope-checkpoint.md), [10](docs/10-results-and-roofline.md) §1).

**Two fears from the TP=2 dress rehearsal did not reproduce, and one small cost did.** At two ranks
the same checkpoint measured ~10 GiB *heavier* per node and left a 31k-token pool that closed the
long-prompt path; at three ranks it is **3.4 GiB lighter** and the pool grows **10 %**, so
`max_model_len 1,000,000` was never in question. A cold-probe signal that draft acceptance collapses
did not reproduce either — it is flat on that probe and 2.4 points lower on the sweep. What it does
cost, and the line is not left empty: **2.4 points of acceptance, 3 % of accepted tokens per step, a
second patch tree to keep in step with the first, and a second 53 GB fast-load sidecar per node**
([13](docs/13-full-scope-checkpoint.md) §7.4, [11](docs/11-open-issues.md) §2.24).

**The "cost" line in that table was wrong, and the correction is a lesson about the instrument rather
than about the stack** `[retracted]`. We published 2.4 points of lost draft acceptance as the price of
production 9. It is an artefact of our own sweep harness: `bench-sweep.py` cycles `prompts[i % 12]`,
so **C1 and C2 see only the first eight of the twelve prompts** while C4–C8 see all twelve, and the
two groups differ by about 8 points of acceptance. Pooled by draft token across all five levels and
three independent boots, production 9 reads **62.27 %** against production 8's **62.09 %** — **+0.18
points**, inside that arm's own ±1.4-point boot-to-boot spread. The sign even reverses at C6 (+1.35).
A 700-token cold probe is identical on both arms (42.53 % against 42.51 %). And because
`accept_len = 1 + k × acceptance` holds on all 90 rows to ±0.005, "acceptance −2.4 points" and
"tokens per step −3 %" were never two costs — they are **the same number written twice**. Net effect
on throughput: **+0.24 %**. The real costs of production 9 are the ones that remain: a second patch
tree, a second 53 GB sidecar per node, and a **prefill** regression on the dense path
([docs/14](docs/14-troubleshooting.md) §7.8, [11](docs/11-open-issues.md) §2.26).

**None of it was about quantization.** The checkpoint would not load for three reasons — a missing
`packed_modules_mapping`, two lines that pin the attention stack to BF16 whatever the weights hold,
and a KDA factorisation the reader does not expect — and one of those three fails identically on a
BF16 copy of the same checkpoint. TP=3 then needed three more things: two launcher constants moved
from `lcm(64, tp)` to `lcm(128, tp)` (vocab `padding_size` 192 → **384**, shared expert 2112 →
**2304**), one patch of ours (A9: split a pre-fused checkpoint tensor by the *checkpoint's* widths,
not the module's padded ones), and a **padded-load path** on the plugin side, which the author built
in `f3e3090` and `754421f` once the quality gate passed. A post-load audit confirms the invariant it
rests on: **285 padded EXL3 sites on the rank that owns the padding, every pad a whole number of
128-column Hadamard blocks and exactly zero** ([13](docs/13-full-scope-checkpoint.md) §7,
[03](docs/03-tp3-padding-and-sidecars.md) §1).

The configurations before it still hold. Production 8 moved the image to `62f53e6` and nothing else,
and every column stayed inside its own band, which is exactly what a change priced at 0.2–0.3 % of
prefill wall should look like ([11](docs/11-open-issues.md) §2.19); production 7 bought **+5.6 % of
pool** by putting the DFlash2 drafter's own cache at fp8 ([07](docs/07-kv-and-draft-page.md) §7).

**Where a step goes is now measured, and the measurement deleted two of our own targets.** A torch
profiler ran against the *live* server, all three ranks, no restart — one launcher flag that should
have been set a week earlier ([09](docs/09-measurement-protocol.md) §4.1). Per steady prefill chunk,
which turns out to be **1,792 tokens rather than 2,048**: MoE trellis GEMM 28.5 %, dense BF16 GEMM
17.4 %, NCCL 14.5 %, hyper-connection mixing 12.0 %, MLA 8.3 %, KDA 8.1 %. Per C1 decode step: dense
BF16 GEMM **45.3 %** and the k=7 drafter **11.4 %**. Per C8 step: the MoE stage **51.6 %**. Against
that, the reconciliation it replaces had `exl3_moe_combine` at 1.5 % of a chunk — **the kernel does
not exist in this build** — and `_zero_kv_blocks` at 1.3 %, which measures **0.09 %**. Both had been
published as ranked targets. The NCCL class is **100 % exposed**: measured comm/compute overlap is
0.00 ms per prefill step. And the "5.45 ms of GPU idle at C1" that looked like a CUDA-graph
opportunity is **3.47 ms** once the profiler's own ~1 µs per kernel boundary is subtracted, of which
77 % is per-kernel dispatch rather than the host — so graph coverage is worth +1.5–2 % single-stream,
not the +6 % we had claimed ([10](docs/10-results-and-roofline.md) §5,
[`results/profile/measured-prod7.md`](results/profile/measured-prod7.md)). Both rulers those
percentages are against were measured on the device — 225 GB/s and 97.3 TFLOP/s, not the 273 and ~125
the datasheet implies.

**The other half of production 7 buys no tokens at all, and it is the part worth copying.** The KV
pool number this stack has spent a dozen arms on turned out to be a *difference* between two readings
of `/proc/meminfo`, taken minutes apart — and it runs backwards: a node that starts with less memory
free awards itself a **larger** pool. Because the launcher killed a ~90 GiB container and started the
next one immediately, and the nodes start in a fixed order, the last node started was systematically
9 GiB short — **27 % of a rank's KV allowance** sitting inside the measurement. No published figure
here is known to be wrong: the pool takes the minimum over ranks and the polluted node happened never
to be the binding one, which is luck rather than design. A host-side wait for memory to settle, before
`docker run`, took the per-rank spread from 9 GiB to 1.4 GiB and made the difference between an
explanation and an artefact checkable. It also refuted this repository's
largest open item, which had claimed 8.2 GiB per worker was stranded and that equalising the ranks was
worth 8–26 % of pool; acting on it would have over-committed the head node
([07](docs/07-kv-and-draft-page.md) §1.1, [08](docs/08-fast-boot.md) §5.1,
[11](docs/11-open-issues.md) §2.3).

**That breakdown is production 7's, and production 9 was built to delete its largest row — and
production 9 has now been profiled too `[measured-here]`.** The 45.3 % dense-BF16 column is what a
`routed_experts_only` checkpoint costs, and the arm that removed it took ~18 ms off an 88.2 ms step.
The same protocol was then run against the production-9 server, live, all three ranks, no restart:
the dense stage is **25.9 % of a C1 step (21.90 ms) against production 7's 45.3 % (42.90 ms)** — that
21 ms *is* the whole of the +22 %, since acceptance moved the wrong way. The new C1 ranking is MoE
trellis GEMM **32.5 %**, NCCL **26.1 %**, dense EXL3 GEMM **15.0 %**, remaining BF16 linears
**10.3 %** (of which nearly half is the drafter's). Per prefill chunk: MoE trellis GEMM **28.1 %**,
NCCL **14.0 %**, dense EXL3 GEMM **13.4 %**, hyper-connection mixing **11.9 %**, MLA **8.3 %**, KDA
**8.0 %**. At C8 the MoE stage is **56.3 %**. The structural findings carry over unchanged: the NCCL
class is still **100 % exposed** (measured comm/compute overlap 0.00 ms) and the C1 idle budget is
still small. **Read two of those numbers with care** — at C1 the NCCL and CPU-gap rows are inflated by
the profiler itself, because production 9 launches 2,738 kernels per step; with the profiler off the
two together are ≤17.19 ms rather than 29.1 `[measured-here]`. Full tables and every class in
[`results/profile/step-breakdown.csv`](results/profile/step-breakdown.csv) and
[`charts/step-breakdown-prod9.svg`](charts/step-breakdown-prod9.svg)
([10](docs/10-results-and-roofline.md) §5, [11](docs/11-open-issues.md) §2.23).

One thing the re-profile changed that the arithmetic did not predict: **the full-scope checkpoint made
the dense stage in *prefill* slower, not faster** — 184.73 ms against production 7's 167.39 ms,
**+10.4 %** — while wall-clock prefill stayed inside the ±3 % equality band. The gain is a decode
gain, and it is confined to decode.

**Behind it, two `NCCL_ALGO` arms and one memory rung, and all three are now closed by measurement.**
`Tree` is rejected on this fabric — 4–6× slower on the sizes that matter. `Ring,Tree` came back inside
the model-free sweep's own repeat spread, so it got the five-round engine arm it needed: **every
concurrency level inside ±1 % of `Ring`**, identical TTFT, acceptance inside its own spread, full
gates. It leaves 1.5 %
more KV pool through NCCL's per-algorithm buffer sizing and that is its only real difference;
**`Ring` stays** ([06](docs/06-nccl-mesh.md) §12.3). The `gpu-memory-utilization 0.83` rung was
predicted at +11 % of pool from production 7's geometry and **measured at +8.7 %** on production 9's —
the method held, the base did not — and it is production 10 ([11](docs/11-open-issues.md) §2.4).
Upstream, the `cuda-exl3` MoE stage is **closed**: `62f53e6`
bounded what was left of `exl3_moe_had_in` at half-ALU work worth ≤2 % of prefill and unreachable in
practice, and the author's own bench closed MLA prefill at 86–89 % of achievable — so every remaining
prefill lever on this stack belongs to vLLM, to the fabric, or to us.

For reference, our NVFP4 stack on the same three nodes reaches C1 57–60, C8 150, prefill 1,585 and a
KV pool of 4.32M at `gpu-memory-utilization 0.88` — against this stack's **69.9 / 197.2 / 1,738 and
5.17M at 0.80**. **EXL3 at TP=3 is now ahead on single-stream decode, aggregate throughput, memory and
boot, and level on prefill.** Read that with two caveats attached. Three of the changes that got it
there — the channel cap, the second cable and `NCCL_PTR_CUDA` — are **fabric-level, not
format-level**, they use the same plugin over the same wiring, and none of them has been applied to
the NVFP4 stack `[not tested]`. And the single largest one, production 9, is **checkpoint-level**: an
NVFP4 checkpoint that quantized the same dense path would collect the same 18 ms per step, and none
exists that we know of `[not tested]`. Neither stack has been measured at max reasoning effort.

**All benchmark numbers here are at reasoning effort `low`, temperature 0.** Max effort would cost
5–12× the tokens and days of cluster time; we did not spend them. Nothing on this page is a max-effort
number, and none of it should be quoted as one.

**Synthetic versus realistic.** Every speed number above is realistic — real prompts, mixed content,
a draft model that sometimes misses. Synthetic prompts ("count from 1 to 200") measure the
speculative-decoding *ceiling* and run far faster; they are labelled as such wherever they appear and
they will disappoint you in real use. See [docs/09](docs/09-measurement-protocol.md).

### If you have two Sparks — the two-node track beside the three

The two-node track is a complete measured recipe of its own, not a cut-down of the above. Same
harness, same protocol, same day; two nodes, TP=2, **expert parallelism off**, the same full-scope
checkpoint, `gpu-memory-utilization` **0.85** (the three-node rung is not transferable and the
two-node ladder has never been derived — the three-node column is now **0.88**, so the memory row
below understates the two-node share by a rung rather than overstating it), 6 September 2026
`[measured-here]`.

**This table pairs candidate B with production 11 `[history]`**, and it is kept in that pairing
because those two arms are same-day and directly comparable. The published configurations are now
two-node **candidate C** and three-node **production 12**; their headline tables are at the top of
this page.

| | **TP=2 candidate B** (2 nodes) | TP=3 production 11 (3 nodes) | two-node share |
|---|---|---|---|
| Single-stream decode (C1) | **58.5** tok/s aggregate (**62.6** per stream) | 69.6 (74.7) | 84 % / 84 % |
| Aggregate at 8 concurrent streams (C8) | **155.8** tok/s | 201.1 | 77 % |
| Prefill, fresh unseen ~8.4K prompts | **1,400** tok/s | 1,760 | 80 % |
| KV pool at `max_model_len` 1,000,000 | **2,128,571** tokens — 2.1 concurrent 1M-token requests | 6,382,920 | 33 % |
| TTFT, C1 / C8 | 0.407 / 1.077 s | 0.278 / 0.806 s | +46 % / +34 % worse |
| Quality | correctness **10/10**, code exam **12/12** cold and warm, tool-call **8/8**, needle-lite **6/6**; MMLU sample **86.02 ±0.75** | 10/10 · 12/12, tool-call 8/8, needle-lite 6/6; MMLU 86.47 ±0.74 | **equal** |
| Cold boot, `docker run` → API ready | **272 s** (fast-load; the one-off dump boot that writes the sidecar is 998 s) | 271 s (dump boot 590 s) | — |
| Autostart unit → `/health` 200 | **261 s**, `systemctl start` on both nodes, no reboot test yet | 312 s from power-on, reboot-tested twice | — |
| Consumed memory per node | 84.8 GiB | 58.3–59.1 GiB | the whole story |

**Two ranks are 77–84 % of the speed on a third of the pool, at identical quality.** What the third node
buys is memory first and bandwidth second — a decode step here is weight-bandwidth bound, so a third
rank cuts each rank's weight traffic by a third and the collective it costs in exchange does not
repay it. See [docs/15](docs/15-tp2-track.md) §7.

**The one setting a two-node reader must not miss** is `HAREM_SW_BLOCK_SIZE=256`. Without it the
two-node pool is 601,562 tokens and a **6,253-token prompt is never scheduled at all** — the engine
sits at `Running: 0, Waiting: 1, GPU KV cache usage: 0.0 %` indefinitely, because one request wants
640 of the pool's 385 blocks ([docs/15](docs/15-tp2-track.md) §4).

## Read in this order

0. [**00 — Start here**](docs/00-start-here.md) — **one page, one question: how many nodes do you have.** What still applies at one node and what does not, which track two and three go to, what a fourth node would change, and the table of which of the twenty documents belongs to which track. Read it first if you are not sure this repository is about your hardware.
1. [00 — Hardware, firmware and OS](docs/00-hardware-and-os.md) — **the complete environment record.** Three Sparks and their firmware, the ring cabling and what the fabric ceiling really is, every version we ran, the hotplug fix that stops a single-node reboot killing the fabric, the six OS-level changes we made and the three we deliberately did not, and the memory rules. Read it even if you think you know this layer.
2. [01 — Model and license](docs/01-model-and-license.md) — the two EXL3 checkpoints, their pinned revisions, and two licences, one of which is not one you have seen before.
3. [02 — Image build](docs/02-image-build.md) — the two-layer Docker recipe, pinned to a `cuda-exl3` commit.
4. [03 — TP=3 padding and sidecars](docs/03-tp3-padding-and-sidecars.md) — why an EXL3 tensor cannot be split three ways, and the shape surgery that makes it possible anyway.
5. [04 — The DFlash2 port](docs/04-dflash2-port.md) — porting a speculative decoder into an image that had never seen one.
6. [05 — Expert parallel and the cuda-exl3 fixes](docs/05-expert-parallel-and-cuda-exl3-fixes.md) — the one-line kernel bug that cost 45 % of the MoE stage.
7. [06 — The NCCL mesh](docs/06-nccl-mesh.md) — 0.6 GB/s in the middle of the size range and the one environment variable that fixed it; then the second cable of every pair, which had never carried a packet; then the PCIe slot that was the ceiling all along, and the transport rewrite that proved it.
8. [07 — KV pool and the draft page](docs/07-kv-and-draft-page.md) — why the pool was capped by a counter, not by memory.
9. [08 — Fast boot](docs/08-fast-boot.md) — 618 s → 274 s → 251 s, the bit-identity proof that makes it safe, and why two patch trees means two sidecars.
10. [09 — Measurement protocol](docs/09-measurement-protocol.md) — five rounds, discard two; and four ways to measure a lie.
11. [10 — Results and roofline](docs/10-results-and-roofline.md) — the full tables, the rulers measured rather than quoted, and where a prefill and a decode step actually go, class by class, from a profiler trace of the live server.
12. [11 — Open issues](docs/11-open-issues.md) — what is unresolved, what we retracted, and what we never ran.
13. [12 — The MLA tuner cache](docs/12-tuner-cache.md) — the measurement tax a process-local cache was charging, and the shorter protocol that removes it.
14. [13 — The full-scope checkpoint](docs/13-full-scope-checkpoint.md) — the three independent reasons a fully quantized checkpoint would not load, none of them about quantization; the loader patch; the TP=3 padded-load port; and what the dense stage is worth, measured twice. **This is the production recipe.**
15. [14 — Troubleshooting](docs/14-troubleshooting.md) — **all 86 failures we hit, indexed by symptom, with the exact log line.** A triage order at the top, and a ranked index of the twenty that produced no error message at all. If something is wrong right now, start here.
16. [15 — Running this recipe at TP=2](docs/15-tp2-track.md) — **the two-node track, and it is now a complete recipe rather than a set of arms.** Why two ranks need no padding at all; the env file, launcher, patch tree, fast-load sidecar and autostart unit that make up a named **TP=2 production candidate**, measured end to end on 6 September 2026; the draft KV page, without which two ranks silently refuse an 8K prompt; and the two findings this page had to retract — including "full-scope at two ranks is a rig, not a serving configuration", which was an unsettled boot rather than a property of the stack.
17. [16 — Comparison with other published recipes](docs/16-comparison-with-published-recipes.md) — a dozen other public GLM-5.3-Flash EXL3 DGX Spark recipes, quoted exactly as they publish them with their own stated conditions, beside our numbers at the matching node count. Read the conditions column before you read the numbers. It also contains the two most useful outside findings we know of: **two other people quantized this model's dense path independently, one at two nodes and one at three, and both measured a gain in the same band as ours** — and a four-node recipe's soak hang that we have never looked for on three.
18. [17 — The memory ledger](docs/17-memory-ledger.md) — **where the 121.6 GiB on each node actually goes**, read from the logs rather than estimated: the per-node ledger down to the driver's fixed reserve, the KV block's anatomy, the KDA state slots that speculation costs, a ranked give-back list with six items already closed at zero, the two-node column beside it, and one pair of our own boots that does not reconcile.
19. [18 — Vision at three ranks](docs/18-vision-at-three-ranks.md) — **the vision tower, on.** Why the recipe said it could not run at three ranks and why that was wrong; the two loader blockers that a flag does not clear, because this checkpoint's tower is 6-bit EXL3 and vLLM builds it unquantized; **the upstream video sampler bug that kills the engine core on the first video request at any parallel size**, with the arithmetic that found it on a CPU in under a second; the brake and the activation ceiling; ten gates with the model's own answers; and the cost, which is inside the boot-to-boot noise on every axis.
20. [audit/](audit/README.md) — a post-install self-check with our own numbers beside each step, the provenance table for every headline figure, and the retraction index. Run `audit/run-audit.sh` before you conclude anything about your install.
21. [charts/](charts/) — four figures generated from the CSVs in [`results/`](results/README.md) by [`charts/make-charts.py`](charts/make-charts.py), standard library only, so you can regenerate them and check the bars against the rows.
22. [systemd](systemd/README.md) — **the autostart units and their preflights**, real, installed and — at three nodes — reboot-tested, plus the hazard that comes first: if you also run the NVFP4 sibling, or the other track's unit, whichever is enabled wins a reboot. Exactly one may be enabled. Read this before a reboot.
23. [tracks/](tracks/README.md) — **the files that differ between the two node counts**, one folder each: the environment template, the patch tree and the autostart unit. Everything else in this repository is shared, and this page says why each shared thing is shared. It also carries the one trap the move introduces: the directory name here is not the directory name on your nodes.
24. [**HELP-WANTED.md**](HELP-WANTED.md) — **what a second cluster could settle, ranked, with the expected effort on every item.** Four nodes, a two-node reboot test, the memory ladder at two ranks, other checkpoints, the mesh plugin's small-message latency floor, the KDA state slots, the KDA GEMM gap, a one-bench falsification of a kernel closure that was corrected the day it was measured, an upstream vLLM issue we measured and confirmed on someone else's thread rather than duplicating, a second one we filed ourselves (the CUDA-graph support gate at three ranks, [vllm#55581](https://github.com/vllm-project/vllm/issues/55581)), and the four largest items from docs/11. It also says, per item, what a contributor with fewer nodes can and cannot check.
25. [CREDITS](CREDITS.md) · [LICENSES](LICENSES.md) · [CHANGELOG](CHANGELOG.md) · [CONTRIBUTING](CONTRIBUTING.md) · [STYLE-GUIDE](STYLE-GUIDE.md) · [`.github/`](.github/) — three issue templates and a pull request template, all of them the measurement protocol in [docs/09](docs/09-measurement-protocol.md) turned into checklists

## The four figures

| | |
|---|---|
| [Decode throughput by production configuration](charts/speed-by-configuration.svg) | C1 and C8 across all twelve configurations, with the `cuda-exl3` commit under each. 10, 11 and 12 are memory rungs and move the pool rather than the speed |
| [KV pool by configuration](charts/kv-pool-progression.svg) | every rung to production 12's 7,041,322, including the one we measured and rejected (0.90) and the one whose rejection we later retracted (0.85) |
| [Where a step actually goes](charts/step-breakdown-prod9.svg) | production 9, profiled on the live server, prefill and both decode regimes |
| [The one number production 9 was built to move](charts/dense-stage-prod7-vs-prod9.svg) | the dense stage, 45.3 % → 25.9 % of a single-stream step |

## Quick start — eleven steps and one optional, for a person or their AI coding agent

Each step ends in a **check**. Do not go on until it passes: on this stack the expensive failures are
the silent ones, and every check below exists because something got past us once.

```text
 1. HARDWARE AND OS.  Read docs/00 end to end. Bring all three nodes to SBIOS 0104+ with
    fwupdmgr, remove /etc/nvidia/cx7-hotplug-enabled on all three, set multi-user.target,
    leave vm.swappiness at 60, and reboot ALL THREE TOGETHER (never one).
    CHECK: `ibv_devinfo | grep -c PORT_ACTIVE` prints 4 on every node, and
           `sudo lspci -vv | grep -A2 ConnectX` reports Speed 32GT/s, Width x4. If it says
           x2, your SBIOS is too old and you are measuring a different machine.

 2. FABRIC.  Bring up the six /24 fabric links, MTU 9000, ipv4.never-default yes. Pick a
    private range that does NOT collide with your LAN -- NVIDIA's playbook uses 192.168.0-5
    and on a typical home network that takes your router down (docs/00 section 4.5).
    CHECK: ping across each second link, bound to the interface, from BOTH ends. Six pings.
           Every node ends with four fabric neighbours resolved. If one fails, STOP.

 3. MODEL.  Download turboderp/GLM-5.3-Flash-exl3, branch 4.05bpw, revision 2a30229e
    (165 GB, MIT). That is the production checkpoint: full scope, so the dense path and the
    head are quantized too. brandonmusic/GLM-5.3-Flash-tr3-4bpw at b20c49ba is the fallback
    if your image predates the padded-load path; its licence is the ShapleyMCG License 1.0,
    not MIT, so read LICENSES.md before you use it.
    CHECK: sha256 23/23 against the repository's own metadata, on each node independently.

 4. DRAFTER.  Download incoai/GLM-5.3-Flash-DFlash2. It is CC BY-NC-ND 4.0 and OUR PERMISSION
    DOES NOT TRANSFER TO YOU (LICENSES.md). The recipe runs without it, about 2.6x slower at
    a single stream: set SPEC_METHOD= empty.
    CHECK: you have read the licence and it permits what you intend to do.

 5. MESH PLUGIN.  Build autoscriptlabs/nccl-mesh-plugin at 19924dcc on EVERY node, with
    patches/kernel/0004, 0005 and 0006 applied. Do not skip docs/06: the default channel
    count costs 13 % of C8, and unpatched the plugin puts every channel on one link of each
    pair. Set NCCL_MAX_NCHANNELS=8 -- 16 is 2.5x WORSE on the decode-sized message.
    CHECK: `make test-unit` gives test_routing 13/13, and after your first benchmark ALL FOUR
           port_xmit_data counters per node have moved, not two.

 6. IMAGE.  Build on every node from the same source tarball (tar, not git archive -- it drops
    the untracked Dockerfile silently), pinned to cuda-exl3 754421f or later. The padded-load
    path (f3e3090 + 754421f) is NOT optional at TP=3 with this checkpoint. docs/02.
    CHECK: verify by BEHAVIOUR, not by hash -- the upstream pytest suite, expect
           44 passed / 41 skipped, exit 0. Binary hashes differ between identical builds on
           this toolchain (docs/14 section 7.10).

 7. SIDECARS.  tracks/tp3/patches/pad-tp3full.py for the model -- it writes the padded config.json
    AND the rewritten quantization_config.json carrying the packed mapping; patches/tp3/pad-tp3.py
    does not write the second one and the load then fails. patches/tp3/pad-tp3.py --draft for
    the drafter. They are symlink trees, not copies (docs/03).
    CHECK: each sidecar is mounted at its OWN host path inside the container. Mount one
           anywhere else and the relative symlinks dangle, reported as "no safetensors found"
           rather than as a mount error (docs/14 section 2.11).

 8. CONFIGURE.  Copy scripts/ and tracks/tp3/patches/ to ~/exl3-zeus/ on every node; hard-link
    tp3full-prelude.sh to the name tp3-prelude.sh inside that directory. Derive each node's
    env from tracks/tp3/env.tp3-full.example WITH SED, per node -- never copy the file between
    nodes. The launcher directory and the patch-tree directory are two things; the examples
    use tp3full/ for both for brevity. Set TP3_DIR (and OVERLAY_DIR) to wherever you put the
    patch tree -- any name works, and it is the one the sidecar identity hashes.
    Point CUDA_EXL3_TUNE_CACHE at a directory under the /cache mount BEFORE the first
    boot, or every benchmark you run measures the tuner rather than your change (docs/12).
    CHECK: MASTER_ADDR is rank 0's MANAGEMENT address, never a fabric one. A fabric address
           hangs the rendezvous silently; scripts/start-tp3.sh refuses one outright.

 8b. IMAGES AND VIDEO, or skip to 9 for a text-only stack. Copy tracks/tp3/patches/vision/
    into the SAME patch tree and paste prelude-vision-hook.sh into tp3full-prelude.sh after
    the full-scope block. Then LANGUAGE_MODEL_ONLY=0 and five settings, all already in
    env.tp3-full.example: HAREM_VISION=1 and CUDA_EXL3_PACKED_MAPPING (EXTRA_ENV), and
    --mm-encoder-tp-mode data, --mm-processor-kwargs {"max_pixels":12544000,
    "max_image_tokens":8000}, --limit-mm-per-prompt {"image":4,"video":2} and
    --mm-processor-cache-gb 0 (EXTRA_ARGS). NO SPACES in any of that JSON -- both variables
    are word-split. The brake is max_image_tokens; max_pixels does NOT reach the video
    processor and is there for vLLM's budget estimate only. docs/18.
    CHECK: run tracks/tp3/patches/vision/verify-cpu.sh FIRST -- a throwaway CPU container,
           no --gpus, safe while the engine is up. It must print five PASS/applied lines,
           the last of which is the video geometry gate. Then at boot: nine log lines
           (docs/18 section 8) and NO 'language_model_only'. Adding these files changes the
           fast-load identity, so budget one dump boot and a SECOND sidecar directory --
           never the same name, or you overwrite your rollback.

 9. BOOT.  worker-2 first, then worker-1, then head. Read the gates BEFORE any number: the
    [padload] line, the ten patch anchors, the assert-5 pad audit, and the
    CUDA_EXL3_DEBUG_NAMES tally (expect 203 EXL3 / 113 bf16). Then the quality gates,
    scripts/correctness-probe.py and scripts/code-exam.py, COLD and again WARM.
    CHECK: 10/10 and 12/12 in BOTH states. The defect class this stack produces hides on a
           fresh allocator, so a cold-only pass proves nothing (docs/09 section 5).

10. VERIFY AND MEASURE.  Run audit/run-audit.sh and compare against audit/README.md. Before
    you trust the KV pool, confirm all three ranks' "Free memory on device" lines agree within
    1 GiB -- the pool is a delta between two /proc/meminfo readings and it runs BACKWARDS
    (docs/14 section 5.8). Optional but recommended: take one FASTLOAD_MODE=dump boot (~620 s)
    and switch to load, which cuts every later boot to 251 s (docs/08).
    CHECK: the audit's numbers land in the bands in audit/README.md. Where they do not,
           docs/14-troubleshooting.md indexes every failure we hit by symptom.

11. AUTOSTART, ONLY ONCE STEPS 1-10 PASS.  Install tracks/tp3/harem-exl3.service and
    tracks/tp3/motor-onkosul-exl3.sh on every node (systemd/README.md has the four commands).
    Edit the FABRIC_PEERS case in the preflight to YOUR addresses first. If you also run the
    NVFP4 sibling recipe, `systemctl disable --now harem-motor.service` in the SAME change --
    its unit wins a reboot and brings up the other engine on the same memory, healthily and
    silently (docs/14 section 10.1). Then reboot ALL THREE TOGETHER, never one (docs/00
    section 3.4): the preflight checks only its own node's fabric, so a single-node reboot
    passes it and starts a rank into a cluster whose peers are gone.
    CHECK: `/health` returns 200 within about 5 minutes of power-on, the unit logs Finished
           at roughly +100 s (less means the preflight did not run), and the quality gates
           pass again afterwards. Ours: gates 10/10 and 12/12, KV pool within 0.6 % of the
           same configuration started by hand.
```

**If a step fails, go to [docs/14](docs/14-troubleshooting.md) before you go to an issue tracker.**
Its §0 is a triage order and its §1 indexes the exact error strings.

Evidence tiers used throughout: `[measured-here]`, `[measured-here, raw lost]`, `[reported]`,
`[estimate]`, `[not tested]`, `[retracted]`.
