# FlashKDA for KDA chunked prefill — the A/B that promoted it (10 September 2026)

**Applies to: TP=3.** Two nodes have not been measured — [HELP-WANTED](../../HELP-WANTED.md) §15.

A port of vLLM [#55737](https://github.com/vllm-project/vllm/pull/55737) onto the pinned vLLM this
stack serves: GLM-5.3-Flash's KDA **chunked prefill** through the fused `vllm._flashkda_C` kernel
instead of the ten-kernel Triton `chunk_kda_with_fused_gate` chain. **Adopted**: sustained prefill
**+6.5 %**, TTFT at 7K **−5.1 %**, decode and KV pool unmoved, four gates full.

The patch, the knob, the prelude line and the operational price are
[`tracks/tp3/patches/flashkda/`](../../tracks/tp3/patches/flashkda/README.md). This page is what the
numbers were.

**Settings, every row below.** Three DGX Spark (GB10, sm\_121) nodes over the ConnectX-7 mesh, image
`exl3-zeus:754421f` (vLLM `0.1.dev20051+g487ecf187`), checkpoint `turboderp/GLM-5.3-Flash-exl3` branch
`4.05bpw` (full scope, 150.2 GiB), **TP=3 + expert parallelism** (96 of 288 experts per rank), DFlash2
draft at **k=7** with an fp8 draft cache, KV dtype **fp8**, `gpu-memory-utilization` **0.88**,
`max-model-len` **1,000,000**, `--block-size 256`, `HAREM_SW_BLOCK_SIZE=256`,
`--max-num-batched-tokens 2048`, `--max-num-seqs 8`, `NCCL_MAX_NCHANNELS=8`, `enforce_eager`
(compilation mode NONE), vision tower on (4 images + 2 videos per request), `HAREM_PREFIX_HIT=1`,
`HAREM_KPOOL_TAIL_FIX=1`, `HAREM_SM12_ITEMS=pdl,kpool`, `HAREM_INDEXER_WS_MODE=bound`, warm MLA tuner
cache, temperature 0, thinking on at `reasoning_effort: low`. That is production configuration 13 with
one environment variable added. Measured 10 September 2026, 02:55–04:00 UTC, nothing else on the
cluster, an `ENGINE-BUSY` lock held on all three nodes throughout
([docs/09](../../docs/09-measurement-protocol.md)).

---

## 0. The instrument, and why a prefill number needs its counters

Prefill throughput here is the delta of `vllm:prompt_tokens_total` over a **60 s window that opens
20 s after the load starts**, with **24 requests in flight**. Every prompt is a fresh ~7,000-token
English document whose **first bytes are a 24-hex-digit nonce**, so not even the first prefix-cache
block can be shared between two requests; `max_tokens=8` at temperature 0, so the engine is
prefill-bound and decode during the window is ~2 tok/s. The prefix-cache queries and hits are sampled
over the **same** window and printed with the result.

**Hits in every window of every arm on this page: 0.** That matters because this stack has published
a prefill number off a repeated prompt before and it read 56 % high
([docs/14](../../docs/14-troubleshooting.md) §8.3). A freshness claim that is asserted rather than
counted is not evidence, and the counters are what make this one checkable.

The instrument is [`loadgen.py`](../../tracks/tp3/patches/flashkda/loadgen.py) — stopwatch, TTFT and
single-stream decode modes. It is a **throughput** measurement and therefore a different number from
`bench/prefill-fresh.py`, which times one request at a time; the two are not interchangeable and the
tables say which was used. The same instrument measured both arms and the confirmation run, which is
the only reason the three are comparable at all: the 7 September figure of 1,914 tok/s in
[`../speed/stopwatch-production-7sep.md`](../speed/stopwatch-production-7sep.md) came from a
**different** generator and must not be read against these.

## 1. The A/B: two sidecar-less boots, one flag apart

The fast-load sidecar's manifest identity hashes every `patch-*.py` in the patch directory and the
full text of the prelude ([docs/08](../../docs/08-fast-boot.md) §4), so registering a new patch
invalidates every sidecar on every node and the preflight refuses a `FASTLOAD_MODE=load` boot.
A first attempt at this A/B died there and measured nothing.

So **both arms were booted with the sidecar disabled** — `FASTLOAD_MODE=` empty, which makes the
launcher skip the whole fast-load block and the prelude never call the preflight (`preflight-fastload`
lines in both boot logs: **0**). The patch is registered in **both** arms; `HAREM_KDA_FLASHKDA=1`
against `=0` is the only difference between them. The arms were taken back to back in one session,
with the env file edited per node by `sed` and never copied between nodes.

**Which kernel actually ran is in the log, on all three ranks of both arms:**

```
[HAREM-FLASHKDA] kda_prefill_backend=flashkda   (HAREM_KDA_FLASHKDA='1')
[HAREM-FLASHKDA] kda_prefill_backend=triton     (HAREM_KDA_FLASHKDA='0')
```

PR #55737 prints nothing equivalent. Without that line a registration mistake — and there were two,
both in [`tracks/tp3/patches/flashkda/`](../../tracks/tp3/patches/flashkda/README.md) §3 — produces a
healthy boot that quietly runs the old path, and the A/B compares nothing with nothing. Tracebacks in
either arm: **0**.

## 2. The results

`[measured-here]`, two stopwatch runs per arm, five TTFT samples per arm, one decode run per arm:

| | Triton (control) | FlashKDA | delta |
|---|---|---|---|
| Prefill, sustained, run 1 | 1,753.5 tok/s | **1,867.4** | **+6.50 %** |
| Prefill, sustained, run 2 | 1,754.0 tok/s | **1,868.8** | **+6.55 %** |
| TTFT, single stream, fresh ~7K prompt, median of 5 | 4.192 s | **3.978 s** | **−5.10 %** |
| Decode, single stream, 512 tokens | 61.15 tok/s | 61.28 | +0.21 % |
| KV pool at `max_model_len` 1,000,000 | 7,077,134 tokens | 7,099,173 | +0.31 % |
| Prefix-cache hits in window (run 1 / run 2) | 0 / 0 | 0 / 0 | — |
| Requests completed (run 1 / run 2) | 42 / 43 | 42 / 44 | — |
| Request errors | 0 | 0 | — |
| Mean prompt tokens (run 1 / run 2) | 7,015.8 / 7,008.9 | 7,006.5 / 7,011.3 | — |
| `/health` 200 after the restart, sidecar-less | 315 s | 285 s | — |

**The prefill delta clears everything this stack asks of a difference.** Both repeats of both arms
land within 0.3 % of their own twin, the two arms do not overlap, and +6.5 % is above the ±3 %
"written down as equal" floor and above every declared band in
[docs/09](../../docs/09-measurement-protocol.md) §1.2. The repeatability is the part worth keeping:
the control read 1,753.5 and 1,754.0 here, and the same instrument on the production configuration
an hour earlier — on the **sidecar-on** boot path — read 1,750.5 and 1,755.6. Four readings inside
0.3 %, which also says the sidecar-less boot does not move prefill the way §2.1 shows it moving
decode.

**TTFT, every sample** `[measured-here]`. The distributions do not overlap:

| arm | samples (s) | median | mean |
|---|---|---|---|
| Triton | 4.106 · 4.164 · 4.192 · 4.213 · 4.220 | 4.192 | 4.179 |
| FlashKDA | 3.899 · 3.957 · 3.978 · 3.978 · 3.994 | **3.978** | 3.961 |

PR #55737's author measured **−8 … −13 %** mean TTFT on 4× GB300 at TP=4 `[reported]`. Our −5.1 %
is the same direction and smaller, which is what this stack's own prefill profile predicts: the EXL3 MoE
trellis GEMMs dominate our prefill (23.4 % and 11.9 % of GPU-busy time for two kernels) and the KDA
share is small beside them.

### 2.1 The decode number a single arm would have published, and why it is a phantom

Against the previous night's baseline — the same configuration on the **sidecar-on** production boot
path — decode read 58.53 tok/s, so the FlashKDA arm's 61.28 looks like **+4.7 %**. It is not. The
matched control, booted the same sidecar-less way, reads **61.15**. The whole difference is an
artefact of the boot configuration, and FlashKDA does not touch the decode path at all.

Without a matched control this page would have claimed a decode gain that does not exist. That is
[docs/09](../../docs/09-measurement-protocol.md) §2 in one paragraph, and it is the reason both arms
were re-booted rather than one arm compared against yesterday.

### 2.2 Boot time, on two bases, because they disagree

The figures above are wall clock from the harness's own mark to `/health` 200. The units' journal puts
the same two boots at **282 s** and **312 s** from `systemctl restart` — 3 s earlier on each arm, so
the 30 s gap between arms is the same on either basis. **Neither is the production boot path**: both
are sidecar-less, where production loads a pre-sliced sidecar and boots in 165–234 s. The internal
record of this session wrote the FlashKDA arm's boot as 276 s, counted from a later mark than the
control's; on the same basis the pair is 285 / 315 and that is what is printed here. Boot time was
not an acceptance criterion and 30 s across one boot each settles nothing either way.

## 3. Gates

Cold, immediately after `/health` 200, on the FlashKDA arm `[measured-here]`:

| gate | result |
|---|---|
| Correctness probe | **10/10** (content-only 9/9, requests with empty content **0**) |
| Code exam | **12/12**, first run, no rerun — including `matrix`, the item with a known flake |
| needle-lite, six depths of a ~54,700-token haystack | **6/6** (`either` 6/6) |
| Vision, synthetic red circle on white, radius jittered per run | **PASS** |

The control arm ran the probe only — **10/10**, empty content 0 — because a control that has the
patch registered and the behaviour switched off is production configuration 13, whose full battery is
[`quality-battery-production-13.md`](quality-battery-production-13.md). The flake baseline these gates
still lack is [HELP-WANTED](../../HELP-WANTED.md) §12.

## 4. Telemetry

Rank 0, sampled every 5 s across each arm's whole speed suite `[measured-here]`:

| arm | samples | GPU temp min / mean / max (°C) | power mean / peak (W) | SM clock min / mean / max (MHz) |
|---|---|---|---|---|
| FlashKDA | 72 | 60 / 82.6 / 87 | 76.3 / 80.0 | 2,392 / 2,425 / 2,489 |
| Triton | 74 | 58 / 80.0 / 86 | 76.0 / 79.0 | 2,411 / 2,438 / 2,528 |

No throttling signature on either arm, and the arms are thermally indistinguishable: the FlashKDA arm
is 2.6 °C warmer in the mean and holds a *lower* mean clock, which is the wrong direction for a
thermal explanation of its speed. Both start from an idle first sample (60 °C / 14.06 W) and end under
load.

## 5. Where the gain should have come from, and where it did not

**Model-free first.** Both halves of the prediction were measured before the engine was touched.

**The kernel, dense inputs**, 22 heads per rank, head\_dim 128, bf16, `lower_bound` −5, one sequence,
median of seven after three warm-ups, one arm per process `[measured-here]`:

| tokens | Triton chain | FlashKDA | ratio |
|---|---|---|---|
| 512 | 0.449 ms | 0.152 ms | 2.94× |
| 1,024 | 1.064 ms | 0.326 ms | 3.26× |
| **2,048** | **2.238 ms** | **0.673 ms** | **3.33×** |
| 4,096 | 4.462 ms | 1.350 ms | 3.31× |
| 8,192 | 9.437 ms | 2.682 ms | 3.52× |

**The kernel at the production shape**, which is the number that belongs in an estimate: q/k/v are
strided views of one fused qkv buffer, so FlashKDA's three `.contiguous()` copies are inside its timed
region `[measured-here]`:

| tokens | Triton, strided | FlashKDA + copies | ratio | copies |
|---|---|---|---|---|
| 512 | 0.468 ms | 0.214 ms | 2.19× | 0.034 ms |
| 1,024 | 1.158 ms | 0.466 ms | 2.49× | 0.164 ms |
| **2,048** | **2.467 ms** | **0.967 ms** | **2.55×** | 0.314 ms |

**One instrument defect, found and published.** Run both arms in one process beside a serving engine —
about 1.6 GiB free — and the Triton arm at 8,192 tokens reads **32.8 ms** instead of 9.44 ms. That is
allocator pressure, a 3.5× artefact, and it would have been reported as a 12× speedup. One arm per
process, and `--min-free-mib` armed so the bench refuses rather than measures.

**The share, from an existing profile rather than a new one.** The rank-0 prefill trace of production 9
(5 September, same TP=3 stack, 18,318 kernel events on a single stream, so no overlap to untangle):
**GPU busy 4,724.6 ms** against a 4,794.0 ms wall, i.e. prefill is **98.5 % GPU-bound** and a GPU-time
saving passes to TTFT about 1:1. The call counts confirm the identification — 204 = 34 KDA layers × 6
prefill steps `[measured-here]`:

| kernel FlashKDA replaces | ms | % of GPU busy | calls |
|---|---|---|---|
| `chunk_gla_fwd_kernel_o` | 60.107 | 1.27 | 204 |
| `recompute_w_u_fwd_kernel` | 55.911 | 1.18 | 204 |
| `chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter` | 44.160 | 0.93 | 204 |
| `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 43.359 | 0.92 | 204 |
| `chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_intra` | 36.342 | 0.77 | 204 |
| `kda_gate_cumsum_fwd_kernel` | 24.072 | 0.51 | 204 |
| `l2norm_fwd_kernel2` | 14.360 | 0.30 | 408 |
| `merge_16x16_to_64x64_inverse_kernel` | 13.318 | 0.28 | 204 |
| `triton_poi_fused__to_copy_sigmoid_0` (the beta cast) | 0.715 | 0.02 | 204 |
| **total** | **292.34** | **6.19** | |
| the same plus the fp32 fill buffers it drops | 306.57 | 6.49 | |

What it does **not** replace, for scale: `_causal_conv1d_fwd_kernel` 48.5 ms, `layer_norm_gated` 12.2,
state gather/scatter 5.9, and the NCCL all-reduce at **673.7 ms (14.3 %)**. Cross-check: the trace says
1.433 ms per layer-step and the micro-benchmark predicts 1.69 ms for the same length mix — 18 % high,
the same order, so the bench represents production.

**The prediction, and the miss.** 6.19 % of GPU time at a 0.39–0.46 production-shape ratio gives
**−3.4 … −4.0 % of prefill GPU time**, i.e. **+3.5 … +4.1 % of prefill speed** `[estimate]`.
Measured: **+6.5 %**, roughly **1.6×** the prediction.

**The excess is not explained by the data in hand, and this page does not explain it.** The prediction
counted the kernel substitution and nothing else. A plausible mechanism — the fused path drops several
fp32 fill buffers and asks less of the shared workspace arena, which may overlap better with the
dominant MoE GEMMs — is recorded as a hypothesis, not a claim, and this stack's own rule is that a
mechanism nobody measured is not an explanation. It is open as
[docs/11](../../docs/11-open-issues.md) §2.35. The honest reading of the adoption is therefore: the
gain is real, repeated and gated, and **we do not fully know why it is as large as it is** — which
also means we cannot say how it will scale to another machine.

## 6. The confirmation on the production boot path

The A/B arms are sidecar-less and production is not, so the promoted configuration was re-measured on
the real path: a `FASTLOAD_MODE=dump` boot into a **new** `FASTLOAD_DIR` (394 s, 53 GiB per rank),
then `load` (181 s), with `preflight-fastload: OK` and `kda_prefill_backend=flashkda` on all three
ranks and 0 tracebacks.

| run | prefill | prefix hits / queries in window | requests | errors |
|---|---|---|---|---|
| Production, new sidecar, first window | **1,865.7** tok/s | 0 / 111,811 | 43 | 0 |
| Production, new sidecar, independent second window 8 minutes later | **1,866.6** tok/s | 0 / 112,013 | 44 | 0 |

Against the arm's 1,867.4 / 1,868.8 that is **−0.1 %**: the gain survives the fast-load path, which is
the one thing a sidecar-less A/B cannot tell you. Correctness probe on the final production state:
**10/10**, empty content 0.

## 7. Two anomalies, both printed

**The 18-minute outage.** The first dump-mode restart was refused by `ExecStartPre` on all three nodes
and all three units went to `failed`: the sidecar `MANIFEST.json` check ignored `FASTLOAD_MODE` and
skipped only when `FASTLOAD_DIR` was empty, so pointing it at a new directory could never boot — the
mode that creates the directory required the directory to exist. The engine was down from 03:26:28 to
03:44:18 UTC, **17 min 50 s**. The gate was **narrowed, not removed** (enforced only for
`FASTLOAD_MODE=load`), backed up per node, and `bash -n` checked on all three.
[docs/14](../../docs/14-troubleshooting.md) §10.6 is the entry; the shipped fix is in
[`tracks/tp3/motor-onkosul-exl3.sh`](../../tracks/tp3/motor-onkosul-exl3.sh). No production artefact
was overwritten and the previous sidecars were intact throughout.

**The final KV pool is the low reading of the session.** Production came up at **7,030,303** tokens,
**0.4 % below** the 7.06–7.10M this session had been reading. The five pools of the session, in order:
7,063,360 · 7,104,683 · 7,099,173 · 7,077,134 · **7,030,303** — a **1.06 %** spread with the memory
fraction, the checkpoint, the image and the patch set all constant. It is read as allocator variance
at 0.88 rather than as a FlashKDA cost, and the reason is in the A/B itself: the FlashKDA arm read
*higher* than the control, not lower. **It was not re-booted for a nicer number** — that would be
cherry-picking — so the low reading stands as the production figure and the inference is labelled as
one. The whole series is inside the 6,873,278–7,143,250 boot-to-boot range this configuration has
accumulated.

## 8. What it cost

- **Speed:** nothing measurable. Decode +0.21 %, KV +0.31 %, TTFT better, all inside their bands.
- **Memory:** **61.45 MiB per rank** of workspace at our settings — `get_workspace_size(2048, 22, 8)`
  is 39.45 MiB, the `(8, 22, 128, 128)` fp32 recurrent state is 11.00 MiB and the
  `(1, 2048, 22, 128)` bf16 output buffer another 11.00 MiB — and all 34 KDA layers **share** the
  arena rather than each holding one. The measured pool went up, not down.
- **Quality:** four gates full cold, and the two kernels agree to about one bf16 ULP on the attention
  output (mean |Δ| 1.2e-5, max 7.3e-4 against a reference |mean| of 2.5e-3; state mean |Δ| 1.0e-4
  against 2.9e-2) `[measured-here]`.
- **Operationally, and this is the real bill:** one more `patch-*.py` in the fast-load identity, which
  costs a **394 s dump boot and a fresh 53 GiB-per-rank sidecar**, and the previous sidecar kept on
  disk rather than deleted because it is the way back — 53 GiB × 3. Node free space 546 G →
  **493 G**.
- **18 minutes of downtime**, paid to a gate of ours that had never been exercised in dump mode
  against a new directory.

## 9. What is not measured

- **Two nodes.** Not run `[not tested]`. The mechanism is rank-independent — the head count per rank
  changes (22 at TP=3) and nothing else — but the MoE GEMMs that dilute the KDA share are a different
  fraction of a two-rank prefill, so the *size* of the gain should not be copied.
  [HELP-WANTED](../../HELP-WANTED.md) §15.
- **A newer FlashKDA.** The `_flashkda_C` in this image is built from FlashKDA `b5d11010`
  (28 July 2026), the tag the pinned vLLM's `cmake/external_projects/flashkda.cmake` names; upstream
  was seven commits further on at `3b225bf` (2 September 2026) when the PR was opened, including a
  TMA proxy-fence fix and an fp16 Neumann-inverse replacement. Every number here is on the July
  kernel `[not tested]` for anything later.
- **Spec-decode steps.** FlashKDA takes the non-spec prefill segment only; a step carrying accepted
  draft tokens keeps the Triton recurrent kernel. Not separated in any measurement here.
- **Longer prefill chunks.** Our step budget is 2,048 tokens, where the production-shape ratio is
  2.55×. The dense ratio keeps climbing to 8,192 and `--max-num-batched-tokens 4096` costs 28.5 % of
  the KV pool ([docs/14](../../docs/14-troubleshooting.md) §5.9), so the interaction is unmeasured and
  stays that way.
- **A soak.** Everything here is minutes of load. The longest continuous uptime on this stack is about
  an hour ([docs/11](../../docs/11-open-issues.md) §3).

---

**Raw:** [`flashkda-ab-10sep/`](flashkda-ab-10sep/) — both arms' stopwatch, TTFT and decode JSON, the
gate outputs with the model's own answers, the per-arm GPU telemetry at 5 s resolution, the four boot
logs with the KV line each boot printed, the production confirmation window, and the dense-shape
micro-benchmark sweep with its numerical-agreement fields. The instruments are
[`tracks/tp3/patches/flashkda/`](../../tracks/tp3/patches/flashkda/README.md).
