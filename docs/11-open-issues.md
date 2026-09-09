# 11 — Open issues, retractions, and what we never ran

**Applies to: both tracks.** The retractions are the whole stack's.

This stack is not finished. This page is the honest edge of it: what is unresolved, what we published
and then withdrew, and what we simply have not measured. Nothing here is hidden in a footnote
elsewhere.

---

## 1. Retracted

**Thirty-eight retractions.** That is the number this repository quotes everywhere, and this is what
it counts: the **32 rows** of the audit table in §1.9, plus the **six** withdrawn findings of §1 that
are not rows of that table — §1.5, the first bullet of §1.8, §1.11, §1.12, §1.13 and §1.14. A
retracted *correction* is a retraction in its own right, which is why §1.12, §1.13 and §1.14 are
counted rather than folded into the rows they correct. The other numbered subsections are the long-form accounts of rows
that are already in the table and are not counted twice.

Seven things we wrote down as findings and later measured properly, plus two smaller ones, plus the
step-time breakdown that a real profiler run corrected in three places (§1.10), plus a kernel ratio
withdrawn the same night it was published (§1.11), plus a full audit of every claim of ours that a
later measurement overturned (§1.9 — **32 rows**, as of the 5 September pass), plus a CUDA-graph
reason that was right about the fact and wrong about the cause (§1.12), plus a failure mode we had
published upstream and someone else corrected (§1.13). Each was published — in a report, an upstream
issue, or both — before it was corrected. Two of them (§1.6 and §1.7) are the same number, corrected
twice, in opposite directions. **§1.13 is the first that a reader outside this stack caught**, and it
is the one we would least have found alone. **§1.14 is the largest in scope**: it withdraws a
capability claim rather than a number, and it stood for two days because nobody tested it against the
checkpoint this stack actually serves.

### 1.1 "The missing `n_rows` also costs the non-expert-parallel path"

We reported that a kernel bug affecting expert parallelism also cost an ordinary tensor-parallel
arrangement, reasoning that with no expert map the surplus tail is a valid local expert on every
rank. **Wrong** `[retracted]`. Running the alignment directly on our own build shows the tail is
`-1` everywhere when there is no map; only `expert_map[expert_ids]` converts it, by negative
indexing. The kernel author's reading was right. Corrected in
[05](05-expert-parallel-and-cuda-exl3-fixes.md) §2, and corrected upstream in the issue thread.

### 1.2 "The extra masking pass is under 1 % of the MoE layer"

Our estimate assumed the pass covered the routed rows; it covered the allocated rows. Measured share:
2.9 % at M=8 rising to **15.8 % at M=2048** `[retracted]`.

### 1.3 "One upstream build is ~10 % slower end to end, and we do not know why"

It does not reproduce. What we measured was boot-to-boot and warm-up variance — **15.9 % spread on
C8 with nothing changed at all**, on the same image and the same environment file `[retracted]`. We
had drawn a kernel conclusion from a single pair of sweeps. This produced the five-round protocol in
[09](09-measurement-protocol.md) §1–2, which is the most useful thing in this repository.

### 1.4 "NCCL is choosing the LL protocol at 16 MB and leaving half the link on the table"

Read off a profiler kernel name. Forcing LL at 16 MB costs 20,114 µs against auto's 1,787 µs, so the
tuner plainly is not choosing it `[retracted]`. The real finding underneath was a different one
entirely — see [06](06-nccl-mesh.md).

### 1.5 "The MLA tuner re-tunes on every prefill chunk, so 2.6 % of prefill is wasted"

Measured on both sides of the relevant upstream fix: the tuner mints **4–5 events per set of fresh
prefills**, each triggering roughly 350 eviction calls — a one-off cost that settles as batch shapes
repeat, not continuous re-tuning `[retracted]`. We withdrew the proposal to disable tuning. The axis
that actually varies on this model is the batch size, not the top-k the fix bucketed.

### 1.6 "The mesh ceiling is ~13 GB/s against a 25 GB/s link, and GPUDirect is the fix we cannot make"

Two errors in one sentence, both published here, both corrected on 5 September `[retracted]`. The
"25 GB/s link" is a **pair of cables at 50 GB/s**, and the plugin was using one of them — the second
cable of every pair had transmitted exactly zero bytes since driver load, on all three nodes. And
recovering the device-pointer path was **two lines** in `getProperties` plus a real flush, not the
receiver-advertised-FIFO redesign we had priced it at, because the plugin's `regMr` already
registered CUDA pointers and was simply never handed one.

What replaced it: the all-reduce reaches **20.84 GB/s** at 64 MB against 12.08 before, and the engine
went from C8 159.9 to **168.9**. See [06](06-nccl-mesh.md) §6–§7. The lesson we would rather have
learned earlier is in §4 of this page.

### 1.7 "The pair of cables is worth 50 GB/s, so the collective is at 28 % of the fabric"

The correction in §1.6 was itself wrong, in the other direction, and lasted about eighteen hours
`[retracted]`. 50 GB/s is the **wire**: two 200 Gb/s ports per card. Each ConnectX-7 sits in a
**PCIe Gen5 x4** slot — `LnkSta: Speed 32GT/s, Width x4` — and carries about **15 GB/s** no matter
what its ports advertise, so a node's real fabric ceiling is **~30 GB/s** and the collective at
~20 GB/s is at about **70 %** of it, not 28 %.

Everything measured stays measured; only the roof moved. What changes is the priority: remaining
fabric headroom is **≤30 %**, worth **2–4 % of prefill**, not the 12–17 % an estimate of ours carried.
The old 13 GB/s ceiling was never "half a link" either — it was one card's PCIe limit at 87 % of it,
which is why the second cable, on the second card, took it to 20 and not to 40. Full account in
[06](06-nccl-mesh.md) §9. Two retractions of the same number in two days is its own lesson: we
computed a ceiling twice from a datasheet and never once from the machine.

### 1.8 Two smaller ones

- **"The gemm should zero a retired tile rather than return."** We shipped it; upstream's design is
  10.7 % cheaper per MoE layer at M=2048. Our patch is retired in writing
  (`patches/kernel/0002-RETIRED.md`) `[retracted]`.
- **"`--no-enable-flashinfer-autotune` is worth 34 s of provably empty work."** It is worth about
  **3.5 s**: the autotune was also doing JIT and kernel warm-up, which simply moved into graph
  capture `[retracted]`. See [08](08-fast-boot.md).

### 1.10 Three rows of the step-time breakdown, and the target list built on them

The step-time breakdown in [10](10-results-and-roofline.md) §5 was, until 5 September, a
**reconciliation**: per-class ratios from an older trace normalised onto a newer wall clock, with a
2.8 % residual booked to NCCL. It was honest about being that. It was also wrong in three places,
each of which had been published as a ranked target `[retracted]`:

- **`exl3_moe_combine`, "1.5 % of a prefill chunk", ranked target #10.** The kernel **does not exist**
  in this build — it is fused into the down-projection epilogue. The class is 0 %. What produced the
  1.5 % was a model-free bench that still contains a `combine6` entry point, measured in isolation and
  then assumed to be on the production path.
- **`_zero_kv_blocks`, "14.7 ms per chunk, 1.3 %", ranked target #8.** Measured on the live production
  configuration: **0.857 ms, 0.09 %**. A 16× overestimate. The model-free reconstruction reproduced a
  *call's* geometry faithfully and was then priced against a pool and page mix the production
  configuration does not run.
- **The DFlash2 drafter, "18.5 ms, 19.5 % of a C1 step".** Measured: **10.78 ms, 11.4 %** — 1.7× too
  high. The old figure came from segmenting a step by "the span of the MoE GEMM calls is the target";
  the profiler's own step annotation excludes the drafter exactly, and the heuristic did not.

A fourth correction is about a *lever* rather than a class, and it went upstream before it was
checked. We reported **5.45 ms of GPU idle at C1 (5.8 %)** and estimated CUDA-graph coverage of the
8-token verify batch at **+6 % single-stream**. Both numbers are wrong in the same direction:
**~2.0 ms of that idle is CUPTI itself** (~1 µs per kernel boundary × ~1,873 boundaries), so the real
budget is **3.47 ms = 3.75 %**; and **77 % of what remains is per-kernel dispatch**, not host latency,
so "CPU gap" was a misnomer — the host runs 3.9 ms *ahead* of the GPU at C1. Graph capture is worth
**1.4–1.9 ms, +1.5–2.1 %**, and it removes neither the step head's `prepare_inputs` nor the one
blocking sync. The corrected figure was posted to the same thread the original went to
([10](10-results-and-roofline.md) §5.8) `[retracted]`.

**The lesson is a protocol one and it is now written down** ([09](09-measurement-protocol.md) §4.1):
the profiler flag costs nothing when unset, so carry it in production; measure the profiler's own
overhead in the same windows; and never read an idle figure straight out of a trace — take
`busy(union)` from the trace and the wall from a profiler-off run.

### 1.11 "EXL3 is 1.58–1.76× slower than BF16 on the KDA shapes at M=8"

The sentence that closed §2.25, published here and posted upstream, and **withdrawn the same night**
`[retracted]`. It was measured on a workstation GPU against a ~300 MB weight bank — a bank that is
three times a 101 MB L2 for the *large* shapes in the table and irrelevant to the 0.72 MB arm the
sentence is about, which stayed cache-resident throughout. Re-measured on the **target** GPU with the
same shapes and the same `cuda-exl3` `754421f` build, rotating **both** arms over at least 4× L2,
`f_b_proj` at M=8 reads **1.023**; GB10's own warm arm reproduces the withdrawn number at **1.605**,
which is the proof of what was being measured. Seven of nine shapes reverse sign, and the family
verdict moves from **−0.584 ms/step** to **+0.050**.

Two more sentences from the same pass go with it: that GB10's ratios would be **worse** than the
workstation's (every family came out **better**), and that these arms are "not bandwidth-bound on any
machine" *because of bytes* — they are not bandwidth-bound, and the cost is **two dependent kernel
launches**, which points at a fusion rather than at a bit width.

**The conclusion did not move: the arms stay BF16 and the item stays closed** — now because
quantizing them is worth nothing rather than because it costs something, and with prefill
unre-measured, so it is re-scoped rather than passed. Full account in §2.25, tables in
[`../results/kernels/kda-gate-bench-gb10.md`](../results/kernels/kda-gate-bench-gb10.md).

**Why it belongs in this list rather than in a footnote.** It is the *third* time on this stack that
a ruler produced the finding (§1.6, §1.7, §1.10 row 4), and the first time the instrument was one we
had already caught once: the workstation ruler read **210 % of peak**, we added a bank, and we sized
it against the wrong shape on the wrong card. The person who found the same artefact in his own table
and withdrew his own claim over it is the kernel author, in the thread ours had been posted to
([issue #5](https://github.com/Zeuss5/cuda-exl3/issues/5)) — we ran his check because he asked, and it
corrected us and not him. The rule that came out of it is in
[09](09-measurement-protocol.md) §4.2: **size the bank against the card you will run on, per shape,
and check the achieved bandwidth of both arms** — the artefact's *sign* depends on which arm fits the
cache, so it is not a constant offset that cancels in a ratio.

### 1.9 The full audit: every claim of ours that a measurement overturned

On 5 September the whole stack was re-read against its own raw data, and every published claim was
checked against the evidence behind it. Thirty-two did not survive — three of them to a single
profiler run the same evening, after the step-time breakdown they came from had stood for a week, and
four more to the full-scope arm that closed the same day, the last of them to that arm's own
promotion to TP=3 a few hours later. The ten above are the ones with a story
worth telling; this table is the complete list, so that nothing is quietly dropped and so the shape of
the mistakes is visible in one place `[retracted]`.

| # | What we claimed | What the measurement says | Where |
|---|---|---|---|
| 1 | Memory ruler is 273 GB/s | Achievable read is **225 GB/s**; every roofline published before was ~22 % optimistic | [10](10-results-and-roofline.md) §4.1 |
| 2 | "A 25 GB/s link" per neighbour | The cable is not the ceiling; the card's PCIe Gen5 x4 slot is, at ~15 GB/s | §1.6 |
| 3 | "A 50 GB/s pair of cables", so the collective is at 28 % | Same number wrong again, other direction; ~30 GB/s per node, collective at ~70 % | §1.7 |
| 4 | The 13 GB/s ceiling is because the device-pointer path is off | Turning it on added **nothing** at 16 MB (1,611 → 1,606 µs); the ceiling was PCIe | §1.6, [06](06-nccl-mesh.md) §9 |
| 5 | Expert-stationary scheduling is worth 14–27 % of MoE traffic | Doubling blocks per expert costs 1.11×, not 2×; the trellis stays L2-resident | §2.12 |
| 6 | The KV-zeroing gate is worth −1.2…1.4 % of prefill | 85.5 % of those bytes are Mamba/KDA state; safe remainder **−0.19 %** | §2.13 |
| 7 | NCCL picks `LL` at 16 MB and leaves half the link unused | Forcing LL there is **11× worse**; the tuner is not choosing it | §1.4 |
| 8 | No CUDA graphs on this stack / the `22 % 4` head-count obstacle | **Our correction was itself wrong (§1.12).** Graphs captured on that boot because the draft KV was bf16 (FlashAttention, whose declaration has no head check); the 36/9 sidecar removed nothing — at three ranks it makes the gate's division 22 % 3 — and since production 7 (fp8 draft KV → FlashInfer) graphs are **off** at TP=3 and **on** at TP=2 on the same image. The gate is wrong, filed as vllm#55581 | §1.12, §2.29 |
| 9 | One upstream build is ~10 % slower end to end | Boot-to-boot variance is 15.9 % | §1.3 |
| 10 | Build `1699c89` costs 16 % at C1 | Single round. Over five rounds there is no loss (+1.2 %) | §1.3 |
| 11 | Disabling flashinfer autotune saves 34 s | ~3.5 s | §1.8 |
| 12 | Draft KV at fp8 is worth +0.3 % of the pool | That was the pre-256 geometry; today the same knob is **+4.7 %** | §2.18 |
| 13 | Fusing the hyper-connection pair saves 40 ms, −3.6 % of prefill | Measured ceiling −30 ms, −2.7 %; the kernel that exists delivers −1.0…1.1 % | §2.16 |
| 14 | The existing fused HC kernel is the better route at small M | At M=8 the fused route is 1.6 % **worse**; at M=2048 it is +32 % worse | [10](10-results-and-roofline.md) §5.5 |
| 15 | The MoE trellis GEMM is target number one | Against the corrected ruler it runs at 81–96 %. The two largest prefill items are **outside** the EXL3 kernel library | [10](10-results-and-roofline.md) §6 |
| 16 | The quick-arm harness runs five rounds and discards two | Its body ran three. Applied literally the rule left a median of one | [09](09-measurement-protocol.md) §1.1 |
| 17 | The chat template we serve comes from the base model | Its md5 matches **neither** checkpoint on disk. It has never been verified against a named source | [tracks/tp3/env.tp3.example](../tracks/tp3/env.tp3.example) |
| 18 | `NCCL_BUFFSIZE` is an open lever worth trying | It had already been measured and eliminated ("no difference"); `NCCL_P2P_NET_CHUNKSIZE` only affects point-to-point. We listed a closed item as open twelve hours later | [06](06-nccl-mesh.md) §12 |
| 19 | `NCCL_MAX_NCHANNELS=8` had already been tried and eliminated | That arm set 8 channels **together with** `NCCL_PROTO=LL` and was never written up. Tried cleanly it is +13 % at C8 | [06](06-nccl-mesh.md) §8 |
| 20 | The extra masking pass is under 1 % of the MoE layer | 2.9 % at M=8 rising to 15.8 % at M=2048 | §1.2 |
| 21 | The missing `n_rows` also costs the non-expert-parallel path | With no expert map the tail is `-1` everywhere; the author's reading was right | §1.1 |
| 22 | `--language-model-only` stops the vision tower being built | It only stops it being *run*, and at TP=3 `divide(16, 3)` asserts before the flag can matter. **The "1.05 GiB wasted at TP=2" half of this row is itself withdrawn — §1.14** | [03](03-tp3-padding-and-sidecars.md), §1.14 |
| 23 | The TP=2 KV collapse came from the draft's page layout | The dominant cause was memory scarcity at two nodes; the page layout was second-order | [07](07-kv-and-draft-page.md) §4 |
| 24 | Overlapping the collective with compute is worth −10…13 % of prefill and −6…10 % of decode | That estimate counted the hideable collective and not the second MoE weight stream the split pays for. Corrected: prefill −6.3…+8.0 %, decode +6…+38 % (worse) | §2.17 |
| 25 | 8.2 GiB per worker is stranded, and equalising the ranks would grow the pool 8–26 % | Not an allocation. vLLM's "non-torch memory" is a delta between two `MemAvailable` readings, and the last node started is the one the kernel has had least time to reclaim for. Acting on it would have over-committed the head by ~8 GiB | §2.3 |
| 26 | `exl3_moe_combine` is 1.5 % of a prefill chunk, ranked target #10 | The kernel **does not exist** in this build; it is fused into the down-projection epilogue. 0 % | §1.10 |
| 27 | `_zero_kv_blocks` costs 14.7 ms per chunk, ranked target #8 | **0.857 ms, 0.09 %** on the live production configuration — 16× over | §1.10 |
| 28 | 5.45 ms of C1 GPU idle, and CUDA graphs would return ~6 % of single-stream | ~2.0 ms of it is the profiler; the real budget is 3.75 % and graphs are worth **1.5–2.1 %**. 77 % of the idle is per-kernel dispatch, not the host | §1.10 |
| 29 | The checkpoint is `routed_experts_only` because attention and the head are quality-sensitive, so the dense stage is a **quality choice** | It was a **loader limitation**. Two lines in vLLM's `glm5next` model file pin the attention stack to BF16 whatever the weights hold (`model.py:331`, `kda.py:171-174`), locking **72.8 %** of the dense traffic, so no checkpoint of any scope could have used it | §2.22, [13](13-full-scope-checkpoint.md) §2.2 |
| 30 | Draft acceptance drops on a quantized target — an early full-scope probe read 45.5 → 39.2 % and 48.0 → 34.4 %, "which would eat part of the speed gain" | **It does not.** Over three sweep rounds acceptance is 61–65 % against the control's 62–63 %, and accepted length 5.3–5.6 against 5.3–5.4. The signal was a cold single-prompt probe with a sample of one; it was reported upstream before the sweep ran | §2.22, [13](13-full-scope-checkpoint.md) §4.1 |
| 31 | Full scope is "+9.5 % aggregate / +25 % per stream" against a control C1 of "54.7 / 54.3" | Transcription, not measurement: 54.7 is the control's **per-stream** median and 54.3 its round-3 per-stream value; the aggregate median is **47.40**. Like for like it is **+26.4 % aggregate, +24.3 % per stream**. Posted upstream in the wrong form | [10](10-results-and-roofline.md) §2.2 |
| 32 | A full-scope checkpoint is ~10 GiB **heavier** per node, so the TP=3 pool would fall 4.70 M → ~3.4 M, −27 %, and the 1M-context claim was at risk | **Not reproduced.** At TP=3 it is **3.4 GiB lighter** and the pool is **10 % larger**. The TP=2 pair was confounded — different checkpoints *and* different `max_model_len`, on an arm whose own report showed free KV scaling with `max_model_len`. Recorded as not reproduced rather than explained: the mechanism was never isolated | §2.22, [13](13-full-scope-checkpoint.md) §6.2, §7.5 |

Read the shape rather than the rows. **Seven of the thirty-two rows are a ruler we quoted instead of
measured** (1, 2, 3, 4, 15, 25, and the roofline percentages that followed). **Six are a single pair
of sweeps, a single probe, or a confounded pair treated as a result** (9, 10, 16, 19, 30, 32). **Three are an arithmetic
model that a bench refuted** (5, 6, 24). **Three are a model-free measurement carried onto the
production path without checking that the path still had it** (26, 27, and the drafter row behind 28).
**Two are our own tooling disagreeing with our own documentation** (16, 18). One (29) is a mechanism
we attributed to somebody else's judgement without checking whether the code would even allow the
alternative. Only one (17) is a provenance claim, and it is still open.

Row 31 is the only one in the table that is not a measurement error at all — the raw sweep was right
and the summary was wrong. It is here because a number that leaves this repository is a published
number whatever produced it, and because the failure it represents is common and cheap to avoid:
**two columns of a table with the same unit in the header and different units in the cells.**

And row 28 adds a category of its own with exactly one member so far: **the instrument's own cost
booked as a finding.** A profiler that charges 1 µs per kernel boundary, on a step with 1,873 of
them, manufactures 2 ms of GPU idle out of nothing — and a stack that had just finished learning to
verify its bandwidth ruler and its memory ruler published the number anyway.

Row 25 is the one to read twice, because it is the largest class in a different disguise: the ruler
there was not a bandwidth figure we had copied off a datasheet but **a number the engine itself
printed**, in its own log, about its own memory. It was still a ruler, it was still unverified, and
it was still wrong.

---

### 1.12 "The 36/9 drafter sidecar removed the `22 % 4` head-count obstacle"

Row 8 of §1.9 corrected an earlier "no CUDA graphs on this stack" claim with that sentence. The fact
was right for the boot it described — graphs captured — and the reason was wrong. They captured
because the draft KV was **bf16**, which puts the drafter on FlashAttention, whose cudagraph
declaration has no head check at all. The sidecar changed nothing about the gate: at three ranks it
makes FlashInfer's division 22 % 3 = 1, and the moment production 7 moved the draft cache to fp8
(FlashInfer) the graphs went off at TP=3 and stayed off — while TP=2, where the same division is
32 % 4 = 0, captures them on the same image. §2.29 has the code and the arithmetic; the declaration
is filed upstream as [vllm#55581](https://github.com/vllm-project/vllm/issues/55581). Retracted 6 September 2026 `[retracted]`.

### 1.13 "A too-small indexer workspace ends in upstream's locked-workspace `AssertionError`"

We wrote this in §2.28, in [HELP-WANTED](../HELP-WANTED.md) §9, in the patch's own docstring and
[in the #55221 thread](https://github.com/vllm-project/vllm/issues/55221#issuecomment-5559336290),
and built a four-layer safety story on it in which upstream's assertion was the load-bearing layer.
**Wrong** `[retracted]`. The issue author, [@drakosha](https://github.com/drakosha),
[corrected it](https://github.com/vllm-project/vllm/issues/55221#issuecomment-5561194190): both
indexers ask the `WorkspaceManager` for a **static** size and only then slice per chunk, so the
request never grows, `_ensure_workspace_size` sees the same `required_bytes` every step, and the
assertion we linked **cannot fire from this path at all**. Our own A/B had said so and we did not read
it — exactly one resize event per rank, for the life of the engine.

Verified against the code we run (image `exl3-zeus:754421f`, vLLM base `487ecf187`), where two things
came out:

- **Their correction of the request path holds exactly.** `total_seq_lens` is the op argument bound
  once at construction to `get_max_prefill_buffer_size(vllm_config)` = `max_model_len × 40`; the
  profiling branch and the run-time branch pass the same value; the per-chunk narrowing is the slice
  `k_quant_full[: chunk.total_seq_lens]` (and `[: chunk.max_local_total_seq_lens]` in the sibling
  file). A Python slice longer than its tensor clamps without raising, so **the failure is silent**.
  The fp8 scale buffer does share the allocation — `get_simultaneous` packs every requested tensor
  into one `uint8` buffer at 256-byte-aligned offsets — so the hazard they describe is real in kind.
- **One detail of the correction does not hold on our build, in the safer direction.** Their
  replacement symptom is an out-of-bounds *write* by `cp_gather_indexer_k_quant_cache`. That kernel is
  **destination-bounded**: `int num_tokens = dst_k.size(0);`, a grid of
  `ceil(num_tokens / BLOCK_Y_SIZE)`, and a per-thread `token_idx >= num_tokens` early return carrying
  upstream's comment "num_tokens may be an allocation upper bound when Python avoids a D2H sync".
  `cu_seq_lens` chooses which in-range rows are valid, not how many are written. True at `808f8cd3ac`
  — the ref their line numbers cite — and at every commit sampled back to `22a58640b4`. So the rows
  past the clamp are never gathered; the wrongness then travels to `fp8_fp4_mqa_logits`, which is
  handed `cu_seqlen_ks`/`ke` describing the unclamped extent alongside a shortened `k_quant`.

**Net: the real failure mode is a silent clamp producing wrong answers, and the load-bearing layer is
ours, not upstream's** — the startup refusal below the one-request floor. Corrected in
[`results/memory/indexer-workspace-ab.md`](../results/memory/indexer-workspace-ab.md) §2, which also
carries the reachability arithmetic, in §2.28 below, in
[HELP-WANTED](../HELP-WANTED.md) §9 and in the patch docstring. The thread is answered.
Retracted 6 September 2026.

### 1.14 "The vision tower cannot run at three ranks, and it wastes 1.05 GiB at two"

Until 7 September 2026 this repository said, in [03](03-tp3-padding-and-sidecars.md) §3,
[14](14-troubleshooting.md) §2.7, the README quick start, `docs/00` and the TP=3 environment
template, that the GLM-5.3-Flash vision tower cannot be served at TP=3 — and shipped a patch whose
purpose was to stop it being built. **The whole claim is withdrawn** `[retracted]`, and it was
withdrawn by running it: production configuration 13 serves 4 images and 2 videos per request at
three ranks, with the text gates unmoved, C1 +0.1 %, C8 +1.5 %, and a KV pool inside configuration
12's own boot-to-boot spread ([18](18-vision-at-three-ranks.md)).

Three separate things were wrong, and the shape of the error is worth more than any of them.

- **"It cannot be split three ways, so it cannot run."** True of splitting and false of running. The
  flag that replicates the encoder instead — `--mm-encoder-tp-mode data` — exists in this vLLM
  revision and the model declares `supports_encoder_tp_data = True`. It was named in §3 of that page
  and dismissed in the same sentence, on the strength of the second error below.
- **"1.05 GiB of BF16 vision weights on every rank at TP=2."** That is the **routed-experts-only**
  checkpoint, which configurations 1–8 served. On the full-scope checkpoint this stack has served
  since configuration 9 the tower is **6-bit EXL3: 172 modules, 1,007 tensors, 0.557 GiB in total and
  0.416 GiB per replicated rank** — measured from the safetensors headers on a CPU in seconds, at any
  time in the preceding two days. The number was carried forward across a checkpoint change without
  being re-derived.
- **"`--mm-encoder-tp-mode data` would clear the assert but keep carrying the 1.05 GiB."** Wrong in
  both halves: the figure is 0.416 GiB, and the flag on its own does not make the tower **load** at
  all, because vLLM builds it with `quant_config=None` while this checkpoint's tower is quantized.
  That was the real obstacle and it was never in the sentence.

A fourth, smaller and in the other direction: on image `exl3-zeus:754421f` the flag may cost nothing
even on the checkpoint the figure came from, because `_mark_tower_model` enters `no_init_weights` and
the tensors stay on the meta device. That is a reading of the source, not a boot-log measurement
`[not tested]`.

**What the failure was.** Not a mismeasurement — the tower's weight was never measured on the
checkpoint the claim was about. A capability was declared impossible from an assert message, a memory
figure was inherited from a checkpoint we had stopped serving, and the one flag that would have
falsified both was named and waved past. The counter-discipline this stack already has —
*model-free first, and a number is only a plan input if two independent sources agree* — would have
caught it from the safetensors headers alone, with no GPU and no engine window. Retracted and
replaced by [18](18-vision-at-three-ranks.md), 7 September 2026.

## 2. Open, with a known next step

### 2.1 The unquantized half of the model

**44.8 % of a C1 decode step** and **16.2 % of a prefill chunk** is BF16 dense GEMM, because the
checkpoint quantizes routed experts only ([10](10-results-and-roofline.md) §5.2–§5.3). Nothing in the
EXL3 kernel library touches it, and at TP=3 each rank's share is a third rather than a half, so those
kernels get *less* efficient as ranks are added. **This is the largest remaining structural item on
the stack**, and the step-time breakdown made it larger rather than smaller: at decode it is now
measured as the single biggest class, ahead of the trellis GEMM.

There is a second, smaller half to it that is not about the checkpoint at all: those kernels are
Ampere-class `cutlass_80_*` on an sm_121 part, and at the engine's own shapes they reach **79 %** of
what `torch.matmul` gets on the same device (63.6 against 80.4 TFLOP/s). Closing that gap is
somebody's cuBLAS/vLLM work and is worth ~3.1 % of prefill `[measured-here]`.

The obvious answer — a checkpoint that also quantizes attention — used to be filed as "the one that
cannot run at TP=3", because with attention quantized there is no unquantized dimension left to split
three ways ([01](01-model-and-license.md) §3.1). **It is now what production serves.** Production
configuration 9, since 5 September evening: **+21.7 % per stream, +12.5 % at C8, KV pool +10.0 %,
quality unchanged** `[measured-here]`, and the acceptance cost we first billed it for was our own
harness (§2.26). None of the three blockers
was about parallelism: the model class declares no `packed_modules_mapping`, the model file pins
attention to BF16, and the KDA block is factorised differently. **This item is closed** — see §2.22
for what it left behind, and [13](13-full-scope-checkpoint.md) for the whole account.

**The percentages above are production 7's trace and have not been re-measured since**
`[not tested]`. The class they name is the one production 9 removed, so they describe the ranking
that produced the change rather than the one that follows it. The second half of this item — that
those kernels are Ampere-class `cutlass_80_*` at 79 % of `torch.matmul` — applies to whatever dense
work remains, which is now the 113 BF16 linears in §2.25 rather than 45 % of a step.

### 2.2 The fabric: what is actually left, now that the ceiling is right

Mostly **closed, and smaller than it looked**. Three things settled it on 5 September:

- The ceiling is the cards' **PCIe Gen5 x4** slots, ~15 GB/s each and ~30 GB/s per node, not the
  50 GB/s of wire the ports advertise (§1.7, [06](06-nccl-mesh.md) §9). At ~20 GB/s of bus bandwidth
  the collective is at about **70 %** of what the machine can carry, so the remaining headroom is
  **≤30 %** — worth **2–4 % of prefill**, not the 12–17 % we had priced it at `[estimate]`.
- The receiver-advertised FIFO with `RDMA_WRITE`, which this section used to call "the deeper fix,
  upstream's to make", **was written and measured** (`patches/kernel/0007`). It removes RNR retries
  and out-of-buffer events to **exactly zero** at every message size — and changes throughput by
  nothing, at any FIFO depth, with or without the flush. The engine arm is inside boot spread.
  Not adopted, kept as an option, deliberately not offered upstream ([06](06-nccl-mesh.md) §10)
  `[measured-here]`.
- The collective's share is no longer an inference: **16.5 % of a prefill chunk** (14–17 % allowing
  for the reconciliation residual) and **10–15 % of a C1 decode step**, down from 20.1 % and 17.5 %
  ([10](10-results-and-roofline.md) §5) `[measured-here]`.

What is still genuinely open here is small and specific:

1. **The engine-side RNR counters have not been re-read since the patches** `[not tested]`. The
   42,000-events-per-node reading was taken on the single-cable, host-bounce-buffer configuration.
   Model-free the retries fell from ~15 per operation to ~3 with `NCCL_PTR_CUDA` and to 0 on the
   one-sided transport — which bought nothing, so this item is now mostly of forensic interest.
2. **`NCCL_MAX_NCHANNELS=12` over two cables** `[not tested]`. Indistinguishable from 8 on one cable,
   never taken to the engine, and 16 is 2.5× worse on the decode-sized message over two
   ([06](06-nccl-mesh.md) §8.1), so the arithmetic changed and the equivalence cannot be carried
   forward.
3. **`patches/kernel/0004` has never been isolated on the engine** `[not tested]`. It rides in the
   production build with 0005 and 0006 at `NCCL_MESH_MIN_RNR_TIMER=1`.
4. **Nobody has tried to make one collective use both cards at once.** Channels alternate between the
   two NICs; whether a single large all-reduce can saturate both PCIe paths at the same time is
   unmeasured, and it is where the remaining ≤30 % would have to come from `[not tested]`.
5. **`NCCL_ALGO` — CLOSED. `Tree` is dead, `Ring,Tree` is equal, `Ring` stays** `[measured-here]`.
   The sweep this item asked for has been run: three arms, two repetitions, production plugin and
   production environment ([06](06-nccl-mesh.md) §12.2, raw in
   [`../results/mesh/algo-sweep.md`](../results/mesh/algo-sweep.md)). **`Tree` is rejected** — 4–6×
   slower than Ring at 16 and 64 MB all-reduce, 23–96 % worse on the decode-step proxy, RNR retries an
   order of magnitude higher, and the port counters show it redistributing traffic asymmetrically
   across nodes that have no hierarchy to reward it. The step-count arithmetic that motivated the item
   (`2(w−1) = 4` ring steps against `~2·log₂3 ≈ 3.2`) is real and is outweighed by what a step costs
   here. **`Ring,Tree` needed the engine, and it has now had it.** Model-free it was 3.6 % better on
   the step proxy and at 4 MB, worse on `sendrecv` at 64 MB, with arms swapping places at 1 MB and a
   repeat-to-repeat spread (up to 1.7× at 1 MB) larger than the effect. The five-round engine arm was
   run on production configuration 10 against the same-day `Ring` boot: **C1 70.6 / C4 143.4 / C8 195.6
   against 70.5 / 144.6 / 194.0 — every level inside ±1 %**, with identical TTFT, acceptance inside its
   own spread and full gates ([06](06-nccl-mesh.md) §12.3). The proxy's −3.6 % did not survive contact with a step
   that overlaps none of its collectives. The one structural difference is **+1.5 % of KV pool**
   (5,702,479 against 5,619,834) from NCCL's per-algorithm buffer sizing — real, and not worth pinning
   production to an algorithm list for. **`Ring` stays; the item is closed.**

`NCCL_BUFFSIZE` and `NCCL_P2P_NET_CHUNKSIZE` are **not** on this list, and were briefly put back on
it by mistake: the first was measured and made no difference, the second only affects point-to-point
transfers ([06](06-nccl-mesh.md) §12, §1.9 row 18).

**Retracted here, twice** `[retracted]`: this section once said the disabled GPUDirect path "holds
the ceiling at ~13 GB/s against a 25 GB/s link", and then said the true ceiling was a 50 GB/s pair of
cables. Both are wrong; see §1.6 and §1.7.

### 2.3 The rank memory imbalance — closed, it was the instrument

**Closed, and retracted.** This section used to say that non-torch memory was 1.50 GiB on rank 0 and
9.48–9.72 GiB on the workers, that **8.2 GiB per worker was stranded**, and that equalising it would
grow the pool by 8–26 % — "larger than every kernel item left on this page". None of that was true.
There is no stranded KV. The claim was a property of the measuring instrument, and the diagnosis that
overturned it is the most useful thing in this section `[retracted]`.

**What the number actually is.** On this integrated-GPU part vLLM's "free GPU memory" is
`/proc/meminfo` `MemAvailable`, and "consumed memory (weights + non-torch)" is not an allocation at
all — it is `MemAvailable(just after NCCL init) − MemAvailable(after the profile run)`
([07](07-kv-and-draft-page.md) §1.1). Three checks settle it `[measured-here]`:

- **The three ranks finish in the same place.** Startup availability 104.10 / 113.07 / 113.10 GiB, a
  **9.00 GiB** spread; after the profile 47.74 / 48.73 / 48.52 GiB, a **0.99 GiB** spread. A
  difference that vanishes by the end was never an allocation. Every candidate we had listed — mesh
  plugin buffers, the 32 GB shared-memory segment, `--headless`, the dumper — is present on all three
  ranks anyway.
- **The direction is impossible.** Read live 24 minutes into the same boot, rank 0 had **1.7–1.9 GiB
  less** free than the workers, because it also carries the API server and the engine core. The
  profile had given it **8.2 GiB more**. Two readings of the same machine ten GiB apart is not a
  finding about memory; it is a finding about the reading.
- **The cause is the boot order.** The launcher kills the previous ~90 GiB container and starts the
  new one immediately; the nodes start worker-2 → worker-1 → head, at 69 / 58 / **47** seconds of
  reclaim time before the snapshot. The node given least time is exactly the node that started 9 GiB
  low, and a low start inflates the pool, because the formula subtracts a *delta*.

**The pool was never mis-sized.** It is built on the minimum over ranks, which is the workers'
~31 GiB — the correct figure — and the arithmetic confirms it: at 141,247 tokens per GiB the binding
rank's 31.03 GiB gives the 4.38M the engine reported, where rank 0's phantom 39.26 GiB would have
given 5.55M. **Acting on the old claim would have over-committed the head node by ~8 GiB**, which is
precisely the 0.85 boot that hit 1.9 GiB free with 1.6 GB of swap (§2.4).

**The real cost, stated without inflating it.** No published pool figure is known to be wrong — the
minimum-over-ranks rule protected every boot we have a ledger for, because the polluted node was always
the head and the head was never binding. That is luck: the polluted node is whichever starts last, and
the amount sitting in the measurement is **27 % of a rank's KV allowance**. Recent pool figures span
**4,231,404 → 4,484,848, 6.0 %**, each step with a candidate explanation in [08](08-fast-boot.md) §5
and post-fix load boots agreeing to 0.4 % — the cost was that an explanation and an artefact could not
be told apart, at the few-percent scale this stack decides at. A host-side settle gate in the launcher
(wait for `MemAvailable` ≥ 112 GiB after `docker rm -f`) removed the ambiguity: per-rank startup free
memory went from a **9 GiB** spread to **1.4 GiB**, at a cost of seconds of boot and zero tokens
([08](08-fast-boot.md) §5.1, [09](09-measurement-protocol.md) §11.1). The rules it leaves behind are
the durable part: **read a pool number only from a load boot**, and **check all three ranks agree
within 1 GiB before quoting one**.

Sub-items left open by the closure, both small `[not tested]`:

- **`--kv-cache-memory` as a pin** skips the profile entirely, so it is immune to both the spread and
  the drift, and all three ranks take the same byte figure. Order still matters: ladder first, pin
  last (§2.4).
- **Dump boots could also `fadvise` the shards they write.** `harem_fastload.py` drops the page cache
  of the shards it *reads*; the 56 GiB it writes in dump mode is still dirty during the profile, which
  is most of the 6.7 % a dump boot reads low. Worth ~2.3 GiB of pool on dump boots only, so it changes
  no production number and is low priority.

### 2.4 The memory ladder above 0.80 — climbed to the top, shipped at 0.87 and then at 0.88 (production 11, then 12)

**The full per-node accounting this rung is spent against is [17](17-memory-ledger.md)**: what fills
the 121.6 GiB, what the fraction is a fraction *of* (the total, not the free), and the six give-back
candidates that measure zero.

**CLOSED, and taken** `[measured-here]`. The rung below was designed and costed here and then run on
the evening of 5 September as **production configuration 10** — production 9 with a single line
changed, `GPU_MEMORY_UTILIZATION` 0.80 → 0.83:

| | production 9 @ 0.80 | production 10 @ 0.83 |
|---|---|---|
| KV pool at `max_model_len` 1,000,000 | 5,168,044 | **5,619,834** (+8.7 %) |
| C1 / C4 / C8 tok/s | 69.8 / 134.6 / 192.4 | 70.5 / 144.6 / 194.0 — inside the bands |
| Prefill, fresh | 1,745–1,774 | 1,687 / 1,769 / 1,779 |
| Quality gates, cold and warm | full | full |
| Swap under load, three nodes | ~0.1 GB | ~0.1 GB, **flat through the rounds** |
| `MemAvailable` after the rounds | 12–13 GB | **8–10 GB** |
| `MemFree` after the rounds | — | 0.9–1.2 GiB, reclaimable page cache |

**+8.7 % rather than the +11 % predicted below**, which is the useful part: the prediction assumed
production 7's pool and geometry, and production 9's is different. The prediction's *method* held; its
base did not.

**What it cost, and the line is not left empty.** `MemFree` now sits below the 2 GiB figure this page
has used as a floor. The number that decides whether that matters is **swap growth, and it did not
move** — 0.85's rejection was never about `MemFree` in the abstract, it was about 1.6 GB of swap
appearing under load. `MemAvailable` at 8–10 GB per node is the honest headroom figure and it is well
clear.

**Superseded on 6 September 2026, including the sentence this item ended with.** "0.85 will not be
attempted on this stack" was written the day 0.83 shipped, and the whole ladder was climbed the next
morning: **0.85, 0.87 and 0.88 all pass; 0.90 is rejected on swap traffic**, and **0.87 shipped as
production configuration 11**. The rung that decided it on the day was not the highest that passed —
0.88 leaves 1.86 GiB of `MemAvailable` on the head node, 0.87 leaves 3.49 — and the criterion that
decided it was not the one this item used. **0.88 was then taken hours later as production
configuration 12**, together with the indexer workspace bound (§2.28): a decision about what the
cluster is for rather than a new measurement. Full tables in
[`../results/memory/ladder-6sep.md`](../results/memory/ladder-6sep.md); the ruler correction is in
[07](07-kv-and-draft-page.md) §6 and [00](00-hardware-and-os.md) §11.2.

| | 0.83 (prod 10) | 0.85 | **0.87 (prod 11)** | **0.88 (prod 12)** | 0.90 |
|---|---:|---:|---:|---:|---:|
| KV pool | 5,674,931 | 6,016,528 | **6,363,636** | **6,542,699** | 6,870,523 |
| Swap traffic under load | ~0 | 0 | **0** | 4 KiB | **si 143 MiB + so 1,519 MiB** |
| `MemAvailable` min, head node | 8.35 GiB | 5.99 | **3.49** | **1.86** | 1.04 |
| Verdict | reference | pass | **shipped (prod 11)** | **pass, thin — shipped as prod 12** | **rejected** |

**What the campaign is worth beyond the rung.** Speed was inside its band at every rung *including
the rejected one*, so a ladder judged on tok/s would have taken 0.90 and shipped a stack that pages
under load. The failure is visible in `vmstat` a whole rung before it is visible in a throughput
number, and it surfaces at the client as a first-prefill stall (5.0 → 9.8 s) rather than as a lower
median.

The design, the arithmetic and the reasoning that produced the 0.83 rung are kept below, because the
method is the transferable part.

<details>
<summary>The original item, as written before the rung was run</summary>

#### 2.4 (original) The memory ladder above 0.80 — 0.83 is designed and waiting on approval

0.85 was measured and rejected on the free-memory rule; 0.88 was never attempted. Three things have
changed since that verdict was recorded, and all of them argue for re-running it rather than carrying
it forward `[not tested]`:

- **The rejection predates the fast-load work.** That work removed a large page-cache spike during
  loading and added `MADV_DONTNEED` plus `malloc_trim` after it ([08](08-fast-boot.md) §5). The
  0.85 boot hit 1.9 GiB free with 1.6 GB of swap; the same production configuration today sits at
  11–12 GiB free with **zero** swap and 3.0–3.5 GiB of page cache. The rung at **0.82–0.83** was
  never tried at all.
- **`--kv-cache-memory` has never been used.** It sizes the pool in bytes rather than as a fraction
  of the device, which matters here because `gpu-memory-utilization` budgets a share of the **total**
  while a worker starts with 113 GiB actually free — and vLLM's own boot log says so, printing the
  byte figure it believes would fully utilise the device against the far smaller one we take.

- **The ruler was unpinned when the verdict was recorded.** The per-rank memory figures the pool is
  computed from carried up to 9 GiB of startup-baseline artefact — 27 % of a rank's allowance, and
  more than twice what this ladder step is worth. The settle gate closed that (§2.3). Climbing a
  ladder against a number whose baseline nobody controls settles nothing.

Order matters: **ladder first, pin last.** A byte pin removes exactly the headroom the 4 GiB
free-memory rule exists to protect, so it is the last rung, not a shortcut past the others. See
[07](07-kv-and-draft-page.md) §6.

**The next rung is specified and costed, and it has not been run** `[not tested]`. `0.80 → 0.83`:
`gpu-memory-utilization` multiplies `MemTotal`, so unlike everything else on the memory page it is
deterministic and immune to the baseline — **+0.03 × 121.63 = +3.65 GiB** of budget, landing whole on
the binding rank (~33.3 GiB at production 7's pool and this geometry's 141,247 tokens per GiB), so the
pool goes ~4.70M → about **5.2M, +11 %**, and concurrency 4.70× → ~5.2×. Stated cost: the OS share
falls from 20 % to 17 %, and the head node — narrowest, because it also runs the API server — goes
from ~12.3 GB free under load to about **8.4 GB**, still clear of the 4 GiB rule. 0.85 (another
+6.08 GiB of budget, head node at ~5.8 GB) is the rung after it and only after a soak. It is one boot
and it is reversible in one line, but it is a **production memory change** and it is **held for the
stack owner's decision**, not deferred for technical reasons.

</details>

### 2.5 `--max-num-batched-tokens 3072`

2048 and 4096 were both measured; the intermediate value was not `[not tested]`. It is the obvious
probe if you want some of the prefill back without all of the KV pool.

### 2.6 `block_m` under expert parallelism

The alignment needs the global expert count and the block-size heuristic is about rows per expert;
the two uses may want different numbers. Worth a sweep `[not tested]`.

### 2.7 The fast-load read path

The sidecar reads at 0.88–1.04 GB/s while a different loader on the same NVMe reaches 3.1 GB/s, so
roughly 3× is still on the table — boot 67 s of weight load could become ~20 s, and the total ~230 s.
An mmap arm exists in the code and has **never been run** `[not tested]`. Beyond that, a buffered
multi-threaded reader.

After that, the profile run at 67–73 s becomes the largest remaining item; its first step alone is
about 45 s and contains the first NCCL collective and the MLA tune.

### 2.8 A persisted MLA tuner cache — closed

**Done.** Upstream's `9bf594c` persists the cache behind `CUDA_EXL3_TUNE_CACHE`; we built the image,
wired it into the environment and measured it: **18 tune events before serving → 0**, no events during
a sweep, and round 1 is no longer a penalty. The sweep protocol dropped from five rounds to three.
Kept here as a closed item rather than deleted, because the *sub*-items it left behind are open:

- the cache key does not know about our kernel patches, so a patch that changed the candidate grid
  would need the file cleared by hand, and nothing enforces that `[not tested]`;
- the boot-time part of the saving was never isolated from phase 4 of the ledger `[not tested]`;
- three rounds is calibrated on **one** cold/warm pair at one configuration.

Full account in [12](12-tuner-cache.md).

### 2.9 The 2,304-padded alternative to expert parallelism

Encoding the routed experts at a padded 2,304 width would let them be tensor-sliced three ways
instead of distributed whole. It was designed, the sidecars were built and verified, and it was
**not adopted**: on the fixed kernel it wins only at M=1 (1.50×), is a wash at M=16, and loses 14 %
at M=2048, while costing +12.5 % expert bytes straight out of the KV pool `[measured-here]`. With a
k=7 draft the real decode batch is M=8 at one stream and M=64 at eight — both in the region where
expert parallelism wins. The sidecars exist on disk as an option and nothing uses them.

### 2.10 DMA-BUF registration on the mesh plugin

`NCCL_MESH_DMABUF=1` works — `ibv_reg_dmabuf_mr` accepts these buffers on this platform, which was an
open question — and is **slower** than plain `ibv_reg_mr` across the size range (64 MB all-reduce
18.08 against 20.84 GB/s) `[measured-here]`. Rejected on the measurement; not investigated further,
so the *why* is `[not tested]`.

### 2.11 The RDMA_READ flush

`NCCL_MESH_FLUSH=0` measures inside noise of flush-on, and better at a couple of sizes. We keep the
flush because coherence is not a noise-level decision, but "inside noise" over two repetitions is not
the same as "free", and it has never been taken to the engine either way `[not tested]`. On the
one-sided transport the same knob is again inside noise ([06](06-nccl-mesh.md) §10.2), and it is
again kept on, for the same reason.

### 2.12 Expert-stationary MoE scheduling — closed, it buys nothing

**Closed.** The large-M trellis GEMM runs 112 blocks over 96 local experts at M=2048 and 121.5 at
M=512, so *if* each block re-read its expert's weights, 17–27 % of the traffic would be avoidable and
an expert-stationary schedule would be worth ~3.7 % of prefill. The kernel author's own bench
(`bench_moe_expert_reread.py` at `9b17ea9`), run unmodified on this hardware three times, says the
trellis **stays resident**: doubling the blocks per expert costs 1.11×, not 2×; quadrupling costs
1.5×, not 4× `[measured-here]`. `moe_align_block_size` keeps an expert's blocks adjacent and 8.4 MB
per expert fits a 24 MiB L2. The same bench gives 1.16× on the author's 188-SM card, so it is
structural and not a property of 48 SMs. Full tables in [10](10-results-and-roofline.md) §5.4.

The traffic model that generated the 14–27 % estimate was simply wrong, and only a bench could say
so — the trace cannot see whether a read was served by L2.

### 2.13 The KV-zeroing gate — closed, the gain is not available on this model

**Closed.** `_zero_kv_blocks_kernel` writes 2.4–2.9 GB per prefill chunk where the new tokens' real
KV is ~3.4 MB, and it does it at 100 % of the memset roofline, so the only lever was to not run it.
vLLM runs it when the cache has Mamba layers **or** mixed precision; the zeroer skips Mamba layers,
which made "so the only live reason is mixed precision — move the draft to fp8 and the gate opens"
look correct. It is not: in this model's hybrid layout one tensor is **co-owned** by an MLA layer and
one Mamba layer per group, so a block moving from the Mamba group to the attention group carries
1.7 MB of raw SSM state. Measured per block, **85.5 %** of what is being zeroed is that co-owned
region `[measured-here]`.

A fail-closed gate was written anyway (`HAREM_ZERO_ATTENTION_KV=0`, off by default, three conditions
proved from the engine's own config, `raise` rather than warn) precisely so the conclusion is checked
by the machine rather than believed: on this model it **refuses to boot**, by design. The safe
remainder — indexer plus draft page, if the cache were uniform — is 0.19 % of prefill, and no partial
mode was written for it. The ceiling is recorded here so nobody prices it a third time.
[10](10-results-and-roofline.md) §5.6.

### 2.14 A cooperative (single-kernel, `grid.sync`) MoE stage — closed for production

**Closed.** On a 48-SM part the cooperative-launch grid tops out at 288 blocks — about 3.9× fewer
blocks have to reach the barrier than on the 188-SM card the idea came from — so the intuition that
"the sign flips on a small part" is correct: outside a CUDA graph a `grid.sync()` barrier beats three
separate launches at every size we measured, by up to 1.37 µs per phase boundary (33 % at 262,144
elements) `[measured-here]`, three independent runs agreeing.

**Inside a CUDA graph the sign flips back**, and production runs inside one. The graph erases almost
all of the separate-launch arm's launch cost (0.2576 → 0.1049 ms at the small size) and has little to
erase on the cooperative arm, leaving the barrier itself as a **+0.2 to +0.3 µs per boundary loss**
at small and medium sizes; at 4M elements both arms are bandwidth-bound and equal. So the detail that
decides it is not the SM count, it is the graph. Tool: [`bench/gridsync.cu`](../bench/gridsync.cu).

**And the premise turned out not to hold here — the kernel author said so himself, unprompted.**
Production does **not** run inside a CUDA graph: spec-decode plus the FlashInfer attention backend
forces `cudagraph_mode=NONE` (§2.23, [10](10-results-and-roofline.md) §5.8), so the arm that wins by
up to 1.37 µs per boundary is the one that is actually running. That reopens the item in principle and
closes it again on size: 42 MoE layers × 3 phase boundaries is about **0.2 ms of a 94.65 ms C1 step**,
0.2 %, behind every item in §2.23. **Reopened, re-priced, still last.** Worth recording because the
advice we acted on ("a barrier is never worth paying for inside a graph") was true on the author's
stack and false on ours, and neither of us checked which one we were reasoning about
`[reported]`.

### 2.15 The fused MoE input transform — closed upstream

**Closed.** Our A/B found the fusion worth +1–4 % end to end at small batch and a regression at large
batch on this hardware; upstream first narrowed the gate (`61a17bc`), then removed the feature
entirely (`76598b2`) and took the same win another way. Nothing to carry. Listed here because it was
an open arm for two days and because the outcome — upstream deleting 232 kernel lines rather than
tuning a threshold — is the useful part of the story ([02](02-image-build.md),
[10](10-results-and-roofline.md) §2).

### 2.16 The hyper-connection fusion kernel — written and measured; not adopted on its own

`hc_mult = 4` carries the residual stream in four copies, and a fused post+pre block touches
`residual_cur` three times per call: the first kernel writes it, the second reads it, the third reads
it. The second kernel's **entire** traffic is that one re-read. Fusing kernels one and two removes
30.4 % of that pair's bytes, which is a ceiling of **−2.5 to −2.7 % of prefill** `[measured-here]`.

**The kernel now exists.** A Triton kernel that grids over token tiles and reduces the post mapping
against the projection while the row is still in registers was written, swept over 33 configurations
and measured model-free in two independent runs: **−14.9 to −15.5 %** on the fused pair, **−9.0 to
−9.9 %** on the three-kernel route, **−1.0 to −1.1 % of the prefill wall** at M=2048. Correctness is
bit-identical on `residual_cur` at every M except one 7-element, 1-ulp difference at M=64;
`layer_input` differs by at most one bf16 ulp on 5.1 % of elements. Full tables:
[10](10-results-and-roofline.md) §5.5.1.

**It reached about 40 % of its own ceiling and it is not going in alone.** The loss is entirely
bytes-per-second — the fused kernel runs at 187.7 GB/s against the 229.5 the route it replaces gets —
and a full configuration sweep could not improve it, so it is not tiling or occupancy. Set against
adoption's price (Triton JIT inside the serving process, a warm-up before graph capture, and a
configuration surface that is a cliff rather than a slope: the winner reads 187.8 GB/s and its
neighbours 79.4 and 44.5), −1 % does not earn its own boot. **It rides the next image bundle with
`had_in` (§2.19), where one boot measures both.**

Two threads left open:

- **The one untried arm that would change the verdict** `[not tested]`: the `tl.dot` operand path
  transposes the projection inside the kernel through shared memory. Pre-transposing it once on the
  host would remove that; if it lifts the kernel to the band the first kernel already achieves, the
  result becomes −2.2 % of prefill and this stops being a bundle rider. Half an hour of work,
  not measured, not claimed.
- **The size threshold is measured now, and it was wrong.** Fusing *loses* below about M=1024 —
  +37.7 % at M=512 — because `residual_cur` at that size is 16.8 MB and fits the 24 MiB L2, so the
  re-read it deletes was never going to DRAM. The module shipped with the threshold at 256.

Two smaller, cheaper pieces of the same class are measured and unclaimed: passing two keyword
arguments the call site currently leaves at their defaults is −4.9 % on the first kernel, and one
constant in the third is −3.5 % — 0.4 % of prefill between them. And a one-pass kernel that reads the
residual once and writes both outputs once would be −5.3 % of prefill, but needs 32 KB of shared
memory per token and would collapse occupancy on 48 SMs; not measured, not recommended, recorded so
the ceiling is known `[estimate]`.

### 2.17 Overlapping the all-reduce with compute — dual-batch overlap is closed; one variant survives

The collective is **16.5 % of a prefill chunk** and GPU occupancy in that window is **99.3 %**, so the
all-reduce is **serialised** against the compute rather than hidden behind it. Making the fabric
faster can win at most 2–4 % of prefill (§2.2); making the collective *overlap* could reach for most
of the 16.5 %. That is why this item carried the largest number on the page. It is now much smaller,
and the reason is arithmetic rather than engineering.

**First, what is not available.** Every overlap mechanism vLLM ships behind a flag is dead on this
configuration, and each for a checkable reason `[measured-here]`, read out of the image with the
engine down: async tensor parallelism, the sequence-parallelism pass and the FlashInfer
all-reduce+RMSNorm fusion all require `torch.compile`, and this model family never enters it — it is
on a fixed list that forces `CompilationMode.NONE`. The FlashInfer path additionally supports world
sizes 2/4/8/16, custom all-reduce 2/4/6/8/16, NCCL symmetric memory a minimum of 4, and torch
symmetric memory neither this capability nor this world size — **`world_size = 3` is excluded by all
four**. DeepEP is installed but never engages, because with `data_parallel_size = 1` the all-to-all
path is not constructed at all. Nothing here is a tuning opportunity; they are all structural.

**Second, dual-batch overlap (DBO), which was the real candidate — closed.** The mechanism suits an
all-reduce: the yield primitives are stream-based and collective-agnostic, vLLM's TP all-reduce lands
on `current_stream()`, and the ping-pong between micro-batches is deterministic so every rank issues
in the same order. The patch is also **smaller than we estimated** — about 95–160 lines across five
files, and it does **not** touch the model file, because one bottleneck (`GroupCoordinator.all_reduce`)
covers all 102 collectives. Our earlier "medium-to-large vLLM patch, mHC state at risk" reading was
wrong on both counts: the mHC triple is thread-local dataflow, not a shared buffer `[retracted]`.

It fails on arithmetic instead. **Splitting the batch pays the MoE expert weight stream twice.** At
M ≥ ~128 every local expert is already touched, so two half-batches stream the same 96 experts per
rank twice: **+73 to +232 ms per chunk**, against **−135 ms** of collective that could be hidden at
best. The break-even is in the middle of that uncertainty band.

| arm (per 2,032-token chunk, f = 0.8 hidden) | ms/token | vs today |
|---|---|---|
| today, MNBT 2048 | 0.5458 | — |
| MNBT 4096, no DBO | 0.4584 | **−16.0 %** |
| MNBT 4096 + DBO | 0.4702 | −13.9 % |
| MNBT 2048 + DBO | 0.5113…0.5896 | **−6.3 %…+8.0 %** |

Decode is worse and not marginal: C1 **+38 %**, C8 +6 %, because the collectives are latency-bound
(102 → 204 of them, and halving the message does not halve the time), the drafter cannot be
micro-batched at all, and `use_ubatching` **disables the breakable CUDA graph for both the target
model and the drafter**. There is also a correctness hazard that would have to be solved first: 34 of
45 layers are KDA/Mamba, and the batch splitter does not align to request boundaries, so a fresh
prompt split in two has its second half start its recurrent state from zero and write it into the
same state slot — fluent, wrong, and silent. `VLLM_DBO_COMM_SMS` is a **no-op** here (the SM-reservation
setter is empty on this communicator and `deep_gemm` is not installed), so NCCL and the trellis GEMM
would compete for SMs unmanaged. **Verdict: do not build DBO.** Increasing the batched-token budget
buys more than overlapping does, for the same KV price and no code — and the budget is a decision we
have already made the other way (§2.5, [07](07-kv-and-draft-page.md) §5).

**Third, the one variant still open** `[not tested]`: **attention-scoped micro-batching**. Split the
chunk only across the attention block and rejoin before the MoE stage. It hides the 45 attention
`o_proj` all-reduces (~80 ms, 44 % of the collective) and pays only for attention weights streaming
twice (~15–25 ms, because the experts dominate the model). Expected **−3 to −6 % of prefill**, ~0 at
decode, ~120–160 lines in the model file and the runner. It carries the **same KDA state hazard**, so
it cannot be written before that is solved.

Before any of this, one model-free probe is worth more than the design work and costs nothing with
the engine down: **does an all-reduce on a second stream actually overlap a GEMM on this hardware**,
or does the plugin's proxy thread or SM contention serialise them? A probe is written and has not
been run; the gate is `overlap_frac ≥ 0.60` and `gemm_stretch ≤ 1.15` at 16.78 MB. If it fails, every
overlap variant above dies with it `[not tested]`.

Also closed while looking: model-level **sequence parallelism** would be a one-line gate change and is
a bad idea here — the bytes are identical (2 × 1.333 S per layer either way), the collective *count*
doubles from 90 to 180, and decode is latency-bound, so it is **+10…15 % worse at decode** for a
~10 % prefill compute saving. And **pipeline parallelism** removes ~98 % of the collective bytes at
the cost of ~3× single-stream decode latency. Both are written down because "just turn on SP" is the
obvious next reflex and it is wrong on this workload.

### 2.18 Draft KV at fp8 — closed, and in production

**Closed.** The DFlash draft's KV cache was `bf16` while the main cache was `fp8` — the *other* half of
the mixed-precision condition in §2.13, and it cost pool. A prelude patch overrides `SpeculativeConfig`
(`HAREM_DRAFT_KV_DTYPE=fp8`) without touching the launcher and is a no-op when the knob is unset.

Both risks are retired and the number that promotes it has arrived `[measured-here]`:

- **Safety**, on the validation (dump) boot: the DFlash sliding-window backend accepts an fp8 cache;
  draft acceptance **60.1–64.0 %** across every concurrency and round against production's 61–65 % and
  a gate that required the 60–65 band; gates 10/10 · 12/12 cold and warm. The mechanism is in the log
  rather than inferred — draft page 393,216 → **196,608 bytes**, per-block 21,917,440 → **20,934,400**,
  divisor unchanged at 363.
- **The pool**, on the ordinary load boot that followed, with the settle gate of §2.3 in place:
  4,449,035 → **4,699,724 tokens, +5.6 %**, against +4.7 % predicted. Speed unchanged inside the
  bands (C1 57.0, C8 175.1), TTFT better (0.34 / 0.91 s against 0.41 / 1.01), acceptance unchanged,
  gates full.

This is **production configuration 7**. The retraction that came with it stays on the record: the
**+0.3 %** figure in our earlier notes belongs to the pre-256 draft-page geometry and never applied to
this stack `[retracted]`. Tables in [10](10-results-and-roofline.md) §1 and
[07](07-kv-and-draft-page.md) §7.

### 2.19 The next image bundle: `exl3_moe_had_in`, and what rides with it

`had_in` was the only sub-roofline kernel left in the MoE stage at 37–57 % of the ruler. Upstream took
it in `a47da6e` by removing a 64-bit division in favour of deriving the index from the grid:
**−10 to −18 %** on that kernel, roofline 57 % → 63 % `[reported]`. On this stack that is ~0.2–0.3 %
of prefill wall, which does not justify an image rebuild on its own.

**It went in anyway, and the result is the honest kind of nothing.** Image `exl3-zeus:62f53e6` is
**production configuration 8** since 5 September afternoon — not because 0.3 % was worth a boot, but
because a build was being made and staying on upstream's head is cheaper than catching up later.
Measured against production 7 on three sweep rounds: C1 56.8 against 57.0, C8 172.8 against 175.1,
prefill-fresh 1,780 against 1,769, KV pool 4,674,931 against 4,699,724, gates full, acceptance 62–64 %
`[measured-here]`. Four signs, every one inside its own band, exactly as a sub-noise change should
read ([10](10-results-and-roofline.md) §1). **What the bundle rule bought here was the discipline not
to claim any of it.**

**And that is the end of the kernel-library item, not a pause in it.** The author has since bounded
what is left in `62f53e6`: the remaining gap on `had_in` is a **half-ALU** limit — a 128-point Hadamard
done with warp shuffles — so the rest is arithmetic that has to happen, not traffic that can be
removed, and the whole of it is worth **≤2 % of prefill** here and is unreachable in practice
`[reported]`. **The `cuda-exl3` MoE stage is closed as an optimisation target on this stack.** The two
things still worth a build are ours or vLLM's: the dense BF16 GEMM that the checkpoint does not
quantize (§2.1) and the hyper-connection mixing kernel (§2.16). What that means for this item is that
the bundle will not grow further from upstream — whatever is in it when it is built is what it is.

**What is left in the bundle is ours, not upstream's.** `had_in` has shipped; the hyper-connection
fusion kernel at −1.0…1.1 % (§2.16) has not, and it is now the only thing in the pile. If the
pre-transpose arm in §2.16 lands it is worth −2.2 %, at which point it earns its own arm; below that
it waits for company. The rule the bundle exists to enforce is unchanged: **do not spend a boot on a
sub-1 % change**; accumulate them and spend one boot on the pile. The pile is also, as of production
8, no longer growing from upstream.

### 2.20 Autostart — CLOSED: the unit is real, installed and reboot-tested. The watchdog is not

**Closed** `[measured-here]`. This item spent the life of the repository saying there was no systemd
unit, that the template in [`systemd/`](../systemd/) had three things wrong with it, and that the
sibling NVFP4 unit would win a reboot. All three are now false, and the honest half of the closure is
listed below with them.

The unit is [`tracks/tp3/harem-exl3.service`](../tracks/tp3/harem-exl3.service), installed and `enabled` on
all three nodes, and the preflight it calls is
[`tracks/tp3/motor-onkosul-exl3.sh`](../tracks/tp3/motor-onkosul-exl3.sh) — seven checks: docker answering,
`ibv_devinfo` 4/4, a ping to each fabric neighbour, `drop_caches`, then the env file, the image and
this rank's fast-load sidecar manifest. `ExecStop` names `exl3-tp3` rather than the NVFP4 container the
template named. `TimeoutStartSec` is **1200**, not 900, so a 620 s dump boot cannot time out mid-load.
`harem-motor.service` is `disabled` on all three nodes and the unit carries
`Conflicts=harem-motor.service` besides.

**Measured once, whole cluster, power-on to a served token.** `reboot` to all three; ssh and 4/4 at
+29 / +30 / +31 s; the units log `Finished` at +98 / +98 / +103 s, which is the preflight, the fabric
wait and the settle gate together; `/health` 200 at 22:28:21 — printed by the harness as +242 s and by
the log's own wall clock as **315 s**, both recorded because they disagree and the larger is the one to
plan with. KV pool **5,652,892** against 5,619,834 from a settled `docker run`, **+0.6 %**; gates 10/10
and 12/12 afterwards ([`results/boot/boot-ledger.md`](../results/boot/boot-ledger.md),
[systemd](../systemd/README.md), [09](09-measurement-protocol.md) §11.4).

**Three things this does not close, and they are the reason the item is not deleted.**

- **`--restart no` has not changed, and will not.** A half-started rank quietly retrying is exactly
  the "fluent and wrong" failure class this stack refuses. Autostart and restart policy were always
  two decisions and only one of them was a gap.
- **systemd still does not honour the worker-2 → worker-1 → head order.** The three units start
  independently. What carried the test is that the workers' rendezvous retries until rank 0 appears
  and `TimeoutStartSec` is roughly five times a normal boot — margin, not a guarantee, demonstrated
  **once**. A rank-dependent delay or a peer-port poll in `ExecStartPre` is the belt-and-braces
  version and is **not written** `[not tested]`.
- **The watchdog is still not written.** A unit solves reboots; it does not solve the failure we
  actually hit, which was the engine exiting while nobody was looking. A 60-second `docker ps` plus
  `/health` poll that records `docker logs --tail 40` when the container is gone is a few lines, needs
  no root and is independent of any unit. One outage during this work ran an hour purely because the
  only thing being watched was a benchmark log `[measured-here]`. **Not written** `[not tested]`.

**Reboot all three nodes together, or none** ([00](00-hardware-and-os.md) §3.4). The unit makes an
all-three reboot survivable. It does nothing for a single-node reboot except start an engine into half
a fabric.

### 2.21 The fast-load identity gate is stricter than it needs to be

The sidecar's identity hashes **every** `patch-*.py` in the patch directory and the full text of the
prelude script. Three patches that do not touch a single weight byte — a KV-page knob, a fail-loud
import guard and a draft-dtype override — refused a boot on that basis and cost an hour
`[measured-here]`. The gate is doing what it was built to do; it is simply drawing the line in the
wrong place, and it makes every experiment that adds a patch cost a dump boot
([09](09-measurement-protocol.md) §11).

The narrower gate, not written `[not tested]`: an explicit allow-list of the patches that can affect a
weight (the padding patch, the expert-parallel patch and its overlay, the drafter patch, the
expert-filter patch) plus the environment keys that decide the layout; a patch not on that list does
not change the identity, and adding a name to the list is then a deliberate act. The prelude would
contribute its **ordered list of patch invocations** rather than its full text, so a comment change
stops invalidating 56 GiB per node while a reordering or a skipped patch still fails. And the launcher
should snapshot the patch directory to a private path before mounting it, which turns "do not touch
the directory during a boot" from a rule into a property.

What must not change: the 32-tensor sha256 sample still runs on every boot. The identity gate is a
cheap early warning, not the proof — which is why it can be narrowed and must not be removed.

### 2.22 The full-scope checkpoint — CLOSED, and it is production

**This item is closed and the thing it asked for is what the stack now serves.** It was the largest
single item this repository carried for a week: dense BF16 GEMM at **45.3 % of a C1 decode step**
([10](10-results-and-roofline.md) §5.3), because every checkpoint through production 8 was
`scope: glm53_routed_experts_only`. Production configuration 9, promoted 5 September 18:40, serves
`turboderp/GLM-5.3-Flash-exl3` at 4.05 bpw at TP=3 with expert parallelism `[measured-here]`:

| | full scope (production 9) | experts-only (production 8) | delta |
|---|---|---|---|
| C1 total / per stream | **69.90 / 75.91** tok/s | 56.88 / 62.39 | **+22.9 % / +21.7 %** |
| C8 total | **197.20** | 175.37 | **+12.5 %** |
| TTFT, C1 / C8 | 0.280 / 0.826 s | 0.344 / 0.906 | −18.6 / −8.8 % |
| prefill, fresh / 7K repeat | 1,738 / 1,575 | 1,776 / 1,537 | equal both ways |
| KV pool at 0.80, 1M | **5,165,289** | 4,696,969 | **+10.0 %** |
| consumed memory per node | 58.3–59.1 GiB | 62.1–62.4 | **−3.4 GiB** |
| draft acceptance · tokens per step | 61.94 % · 5.34 | 64.36 % · 5.50 | **−2.4 pt · −3.0 %** |
| gates cold and warm · MMLU sample | 10/10 · 12/12 · **86.47 ±0.74** | 10/10 · 12/12 · 86.4 ±0.7 | equal |

Per step 88.2 → **70.3 ms**. The estimate this item carried was 42.9 ms → ~11 ms, about +34 %;
**17.8 ms arrived**, and the difference is the four families this checkpoint leaves in BF16 anyway
(§2.25). The +34 % was never refuted — it was an upper bound, and it is now bounded from both sides.
The whole account, including the loader work and the padded-load port, is
[13](13-full-scope-checkpoint.md).

**What closing it settled, beyond the tokens.**

- **The premise of this item was wrong and that is a retraction.** It was never a quality decision by
  the publisher. Two lines in vLLM's `glm5next` model file pin the attention stack to BF16 whatever
  the weights hold, locking **72.8 %** of the dense traffic, so `routed_experts_only` was the only
  scope that could load at all (§1.9 row 29).
- **The 6-bit `lm_head` cost nothing measurable.** It was the one place flagged for damage, and it
  has now been measured twice: 86.32 ±0.75 at TP=2 and 86.47 ±0.74 at TP=3.
- **The KV risk this item carried was the wrong sign.** It warned that the TP=2 arm's ~10 GiB of
  extra per-node memory might replicate at TP=3 and take the pool from 4.70 M to about 3.4 M, −27 %,
  putting the 1M-context claim in question. Measured: **3.4 GiB lighter and +10.0 % of pool.** The
  mechanism is still not isolated; the TP=2 reading is recorded as **not reproduced** rather than
  explained ([13](13-full-scope-checkpoint.md) §7.5).
- **The plugin author's side is done.** `f3e3090` (a padded output dim accepted when the pad is whole
  128-blocks, with `svh` allocated zeroed, plus a row-parallel `suh` load that copies what exists and
  zeros the rest), `754421f` (the vocab loaders fill a prefix) and `807d798` (the acceptance
  diagnostic prints at all). All three were required. A fourth question was answered exhaustively
  rather than built: all **65,536** trellis codes swept through the device decoder on all three
  codebooks, zero non-finite, bounded ranges `[reported]`.

**Three things it left behind, each with its own item:** the two patch trees and the second sidecar
(§2.24), the KDA gating arms that stay BF16 — closed by measurement, then closed again for a
different reason when that measurement was re-taken on the target GPU (§2.25) — and the 2.4 points
of draft acceptance, which was ours rather than the checkpoint's (§2.26).

### 2.23 The remaining C1 idle: 3.75 %, and the four things inside it

Superseding the "+6 % from CUDA graphs" line this page used to carry (§1.10). The corrected budget at
C1 is **3.477 ms of a 92.64 ms profiler-off step = 3.75 %**, and it is the *whole* ceiling — a perfect
fix returns 57 → 59.1 tok/s. Ranked by expected value against that base
([10](10-results-and-roofline.md) §5.8):

| # | fix | C1 gain | C8 gain | notes |
|---|---|---|---|---|
| 1 | **Fuse the glue kernels** (717 of the boundaries sit in front of norm / elementwise / copy) | 0.6–1.0 ms (0.6–1.1 %) | 0.5–0.9 ms | `compilation mode NONE` today — torch.compile is off entirely, so nothing is fused. The blocker is that PIECEWISE silently disables spec-decode (vLLM #53030) and the drafter's own speculator declares PIECEWISE unsupported, so this is an **experiment** (compile the target model, keep `cudagraph_mode=NONE`), not a config change |
| 2 | **De-serialise `prepare_inputs` at the step head** | 0.4–0.6 ms (0.45–0.65 %) | ~0 | **the cheapest.** The step's first 1.08 ms runs at **11.7 % GPU occupancy**: four rounds of eager `aten` ops with a pinned H2D each, then a `torch.compile` region that already operates on a deliberately *stale* confidence copy — so it has no dependency and can move into the previous step's shadow. No backend change, fp8 draft KV kept |
| 3 | **Pinned/async the verify→draft D2H, drop the `nonzero`** | 0.15–0.25 ms | 0.20–0.30 ms | one `Memcpy DtoH (Device → Pageable)` — synchronous by definition — then `cudaStreamSynchronize` and host bookkeeping. It is the single structural gap left at C8 (8.2 % of that idle) |
| 4 | **FULL CUDA graph for the 8-token verify batch** | 1.4–1.9 ms (1.5–2.1 %) | 1.0–1.6 ms | needs an attention backend declaring `UNIFORM_BATCH` **and** fp8 draft KV. Three routes: raise FlashInfer's fixed-`qo_len` paged-decode wrapper to `UNIFORM_BATCH` (upstream, unverified); give the drafter a Triton backend that supports fp8 KV and declares it (draft accuracy re-gated); or return to bf16 draft KV, which is ready, costs 5.6 % of pool, and **has already been tried — same 57 tok/s.** Comes *after* #1, since a graph replays whatever kernel count #1 leaves. **6 Sep:** the first route is moot — the drafter already runs the graphable XQA kernel; what is wrong is FlashInfer's *declaration* (it divides the target's 22 heads by the draft's 3 KV heads), filed as [vllm#55581](https://github.com/vllm-project/vllm/issues/55581); §2.29 has the arithmetic, the price and the A/B |
| — | draft collective overlap | up to 1.4 ms | up to 2.6 ms | not an idle item at all; it is §2.17's lever seen from the drafter's side (11 all-reduce/step at 133 µs, latency-bound, overlap measured at 0.014 ms) |

1–4 do not add cleanly (1 and 4 target the same boundaries). **Realistic total: 1.0–1.5 ms/step =
+1.1…1.6 % single-stream** `[estimate]`. Every one of them is vLLM-side work, and none is worth a boot
on its own — this is a bundle item like §2.19, and the honest summary was that single-stream on the
old checkpoint scope was close to its floor, with §2.22 the one lever that was not.

**§2.22 has since been taken, and that changes the size of this item rather than its content.** The
budget above is 3.477 ms of a **92.64 ms** step; production 9's step is **70.3 ms** (72.52 ms on the
re-profile's own boot), so the same absolute milliseconds — the boundaries, the dispatch, the one
synchronous D2H — are a larger share of a shorter step. The four fixes are unchanged and so are their
absolute gains; only the percentages move, and they move upward.

**Production 9 has now been profiled** ([10](10-results-and-roofline.md) §5) `[measured-here]`, and
the C1 idle row it prints is **8.43 %** — but that figure is **not** comparable to the 3.75 % above
and must not be quoted against it. Production 9 launches 2,738 kernels per step against production
7's count, so CUPTI's per-boundary cost lands much harder: the trace's wall is 84.44 ms where the
profiler-off step is 72.52 ms, and NCCL plus the CPU gap together come to **≤17.19 ms** with the
profiler off rather than the 29.1 ms printed. **Subtracting the instrument on this arm has not been
done properly** `[not tested]`, and until it is, the honest statement is that the idle budget is
small, per-kernel dispatch dominated, and of unknown exact size on production 9. This is the third
time on this stack that a number has had to be read against the cost of the tool that produced it.

### 2.24 Two patch trees and two fast-load sidecars, and the merge that is owed

Production 9 runs from [`tracks/tp3/patches/`](../tracks/tp3/patches/) while production 8's tree,
[`patches/tp3/`](../patches/tp3/), stays on every node as the rollback. **The code did not have to
diverge** — the two constants that differ are `lcm(128, tp)` against `lcm(64, tp)` and are provably
no-ops at TP≤2, so one tree could serve both — **the fast-load manifest identity did.** The
directory's file list and contents are hashed into the sidecar, so adding the full-scope patch to
`patches/tp3/` would have refused the next production boot on all three nodes, which is exactly what
happened twice on 5 September before the second tree existed ([09](09-measurement-protocol.md)
§11.2). Keeping the arm in its own directory is what let production 8 stay up, untouched, while
production 9 was built and measured beside it.

The price, stated plainly: **a fix that lands in one tree does not reach the other**, and two files
(`patch-vllm-tp3.py`, `preflight-tp3.py`) are deliberately divergent. The merge is technically
possible today and was not done, because doing it would change production 8's identity and cost a
dump boot on the arm we were keeping as the rollback. `tracks/tp3/patches/README.md` carries a
file-by-file `cmp` that turns "they have not drifted" from an assumption into a check; that is a
mitigation, not the fix. **Not written** `[not tested]`.

The disk half of it: **53 GB × 3** for production 9's sidecar on top of production 8's ~63 GB × 3, on
top of two 154+ GiB checkpoints × 3. One node had 51 G free before the arm and needed old sidecars
cleared first. There is no policy for retiring an old sidecar and there should be — the safe rule is
that a sidecar may be deleted only after its env file has been retired, because
`FASTLOAD_MODE=load` with the directory gone refuses the boot loudly and correctly
([08](08-fast-boot.md) §9 step 6). **Not written.**

### 2.25 The 113 linears that are still BF16 — closed, and **the reason we gave was wrong**

`CUDA_EXL3_DEBUG_NAMES` on the production boot reads **203 EXL3 / 113 bf16** per rank
`[measured-here]`. The 113 are four families and nothing else: KDA `f_b_proj` (34), `g_b_proj` (34),
`in_proj_bfg_a` (34) and MLA `kv_b_proj` (11). They are what makes the measured gain 17.8 ms rather
than the ~32 ms the estimate implied ([13](13-full-scope-checkpoint.md) §4.2).

This item asked what they were worth, answered it on 5 September with a model-free bench on a
**workstation** GPU — and got the answer's *reason* wrong. **The headline sentence is withdrawn:**

> **"On the KDA shapes, EXL3 is slower than BF16 at decode: 1.58–1.76× at M=8."** `[retracted]`

That is a **warm** number. Re-measured the same night on the target GPU with the same shapes and the
same `cuda-exl3` `754421f` build, rotating **both** arms over a bank of at least 4× L2, `f_b_proj` at
M=8 reads **1.023** — level. GB10's *warm* arm reproduces the withdrawn figure almost exactly
(**1.605** against the published 1.596), which is how we know what the workstation was measuring.
Seven of the nine shapes reverse sign. Tables, method and projection:
[`../results/kernels/kda-gate-bench-gb10.md`](../results/kernels/kda-gate-bench-gb10.md); raw under
[`../results/kernels/gb10-coldbench/`](../results/kernels/gb10-coldbench/); the script ships as
`bench/kda_gate_bench_gb10.py`. Posted to the same upstream thread the original went to
([issue #5](https://github.com/Zeuss5/cuda-exl3/issues/5)).

**The closure survives. It is re-scoped, not passed, and it now reads like this** `[measured-here]`:

1. **The checkpoint author left them BF16 on purpose, and said so in code.** Unchanged and
   independent of any bench. `exllamav3`'s `gated_delta_net.py` builds the five KDA gating arms with
   `qmap = None` and the comment names the reason — the model author's own FP8 release excludes those
   families from conversion, which it reads as a sensitivity signal. `kv_b_proj` is not even a
   `Linear`: attention never applies it as that GEMM, so it is carried unquantized by construction.
   The same author quantized `qkv_proj` and `o_proj` to 6 bit from the same exclusion list, so it was
   a discriminating decision rather than a copied one.
2. **The KDA gating arms stay BF16 because quantizing them is worth nothing, not because it costs.**
   Cold on GB10 the family moves from **−0.584 ms/step** to **+0.050 ms/step at 4 bit** —
   **0.07 % of C1**, and −0.036 ms at 6 bit. Neutral either way, and far too little to pay for a
   multiplicative quality risk on `f_b_proj`'s decay term. **The lever on these shapes is not the bit
   width.** Cold, `f_b_proj` reads **45.7 GB/s — 19 % of peak — in either format**: the arm is not
   bandwidth-bound, it is sitting on a ~5 µs floor made of **two dependent launches**
   (`exl3_had_in_kernel` + `exl3_gemm_m_kernel`) where bf16 has one. Fusing `had_in` into the GEMM at
   narrow inputs is the thing worth asking for, and it has been asked for `[not tested]`.
3. **The arithmetic no longer fails by an order of magnitude — it fails narrowly, and only on the
   part we measured.** The three measured families are **+1.368 ms/step at 4 bit** against a
   `Δ ≥ 1.5 ms/step` gate: 2.5× the workstation's +0.547, and still short. Applied to the whole
   **4.10 ms** the production-9 trace attributes to the target's four unquantized families, at the
   cost-weighted cold ratio 0.495, it is **+2.07 ms — which would pass** `[estimate]`. We do not claim
   the pass: **1.389 ms of that budget is MLA/DSA kernels this bench never touched.**
4. **The work list inverted rather than grew, and this is why the item stays shut.** **96 %** of the
   gain (+1.318 of +1.368) is still `kv_b_proj` alone, and `cuda-exl3` at `754421f` still has **no
   per-head batched EXL3 GEMM** and no `M`-threshold reconstruct path, both re-verified by source
   scan. The one item worth doing is a **kernel** job, not a quantization job.
5. **Prefill was not re-measured.** M = 1, 8 and 64 only; M = 1,792 was not run, and the gate is an
   **and**. The workstation prefill verdict is less exposed to the artefact — those shapes are
   compute-bound — but it stands unconfirmed `[not tested]`.

**What survives unchanged is not a quantization item at all.** The workstation bench found the MLA
strided-batched family running in **fp32** — 11 calls per step, 0.757 ms — and bf16 measures
**0.684×**, worth **+0.24 ms/step, about +0.3 % of C1**, with more than half of that family's prefill
cost as well. No quantization, no checkpoint change, no new kernel: a dtype. The cold re-measurement
does not touch it in direction or size, and it is still the cheapest item on the board. It is gated on
`needle` at 1M rather than on speed, because this is the tensor that decodes the KV latent and its
error would touch the whole of history. **Filed as future and minor** `[not tested]`.

**A second workstation claim goes with the first** `[retracted]`. That report warned its ratios were
*optimistic* for GB10 — "at M=8 GB10's fixed cost is larger, so its ratios will be worse" — and rested
the verdict on failing a gate with optimistic numbers. **Every family came out better on GB10.** This
part is a rare thing on this page: an error whose correction makes the stack look *better* and the
conclusion no weaker.

**The instrument lesson, restated, because ours moved as well as upstream's.** The original closure
already knew a resident weight lies — its own ruler read **210 % of the workstation's peak** before a
bank was added, and that is why a ~300 MB bank exists in `bench/kda_gate_bench.py` at all. It was
still not enough, for a reason worth carrying: **a bank sized against the wrong card is the same
mistake as no bank at all.** 300 MB is three times a 101 MB L2 for the *large* shapes and irrelevant
to a 0.72 MB one, which stayed resident throughout. And the artefact's **sign depends on which arm
fits**: on a 101–128 MiB L2 both arms fit and EXL3 reads slow; on GB10's 24 MiB only the trellis fits
and EXL3 reads fast; only cold is honest on either. Written into
[09](09-measurement-protocol.md) §4.2.

**What the re-measurement cost:** 90 s of one GPU, 1.47 GiB peak, in a throwaway container beside an
idle production engine. No restart, no configuration change, nothing on the cluster touched.

<details>
<summary>The original closure, as published on 5 September, before the target-GPU re-measurement</summary>

**This item used to ask what they were worth and note that we had not measured it. We have now, on a
model-free bench, and the answer is that quantizing them makes the engine slower** `[measured-here]`.
The bench and the arithmetic are in [13](13-full-scope-checkpoint.md) §4.4, the raw tables in
[`../results/kernels/kda-gate-bench.md`](../results/kernels/kda-gate-bench.md), and both scripts ship
(`bench/ruler_check.py`, `bench/kda_gate_bench.py`). The four things that close the item:

1. **The checkpoint author left them BF16 on purpose, and said so in code.** [unchanged — point 1
   above.]
2. **On the KDA shapes, EXL3 is slower than BF16 at decode.** Measured at the TP=3 per-rank widths,
   M=8, CUDA graphs on, against a weight bank three times L2 so nothing reads out of cache: EXL3 4-bit
   is **1.58–1.76×** the BF16 time on `f_b_proj`, `g_b_proj` and the `in_proj` split. These arms are
   0.72 MB and are **not bandwidth-bound on any machine**, so four-bit weights buy bytes that were
   never the cost while paying the Hadamard and trellis-decode fixed cost.
3. **The arithmetic cannot be rescued even at zero cost.** The whole KDA gating family is
   **0.851 ms of a 72.5 ms step**. If every one of those linears took no time at all it would be
   +1.2 % of C1 — under the gate the arm was given, and far under the noise floor in
   [10](10-results-and-roofline.md) §1.1.
4. **The one family that would genuinely gain needs a kernel that does not exist.** `kv_b_proj` is
   bandwidth-bound and measures **0.391×** at 4 bit, worth +1.13 ms/step — but `cuda-exl3` at `754421f`
   has no per-head batched EXL3 GEMM (`exl3_linear` is a single `[M,k] → [M,n]`), and no `M`-threshold
   reconstruct path either, both verified by source scan.

Of those four, **point 1 stands, point 4 stands and got larger (0.391 → 0.291), point 2 is
withdrawn, and point 3 is withdrawn as stated**: the ceiling argument was sound but it was built on
the KDA family alone, which is exactly the family that turned out to be worth nothing either way.
Two more sentences from the same pass are withdrawn with them: that these arms are "not
bandwidth-bound **on any machine**" — true, and the conclusion drawn from it was still wrong, because
the cost is launches rather than bytes — and that GB10's ratios would be *worse* than the
workstation's.

</details>

A second, smaller half of the same item survives from §2.1: whatever dense BF16 remains runs on
Ampere-class `cutlass_80_*` kernels on an sm_121 part, at **79 %** of what `torch.matmul` reaches on
the same device. That is cuBLAS/vLLM work and it is now worth proportionally less than it was.

### 2.26 The 2.4 points of draft acceptance production 9 cost — **RETRACTED: it was our harness**

**`[retracted]`. There is no acceptance regression.** Pooled by draft token across all five
concurrency levels and three independent boots, production 9 reads **62.27 %** against production 8's
**62.09 %** — **+0.18 points**, inside that arm's own ±1.4-point boot-to-boot spread. At TP=2 the
full-scope arm was +1.10 points. The 700-token cold probe is identical on both (42.53 % against
42.51 %). Net effect on throughput: **+0.24 %**.

**The mechanism, and it is worth more than the number.** `scripts/bench-sweep.py` cycles
`prompts[i % 12]`, so **C1 and C2 see only the first eight of the twelve prompts** while C4–C8 see
all twelve. The two groups differ: p0–p7 run −1.6 points against the mean and p8–p11 **+6.5**. A
two-group prompt-mix model explains all five levels at **R² = 0.97**, and the same sign pattern
appears independently in the TP=2 pair (R² = 0.91). So the *sign of the difference changes with
concurrency*: honest pooled C1 is −1.68 points at a permutation-test **p = 0.11** — not significant —
while **C6 is +1.35 the other way**. Reading a per-level median instead of pooling by draft token is
what produced the headline.

**And it was never two costs.** `accept_len = 1 + k × acceptance` holds on all 90 rows to ±0.005, so
"acceptance −2.4 points" and "tokens per step −3 %" are **the same number written twice**.

**Confound, stated honestly:** the two arms ran on different images (`754421f` against `62f53e6`).
Those intervening commits touch the loader and debug paths rather than decode arithmetic, and **the
drafter is byte-identical in both arms** — but it is not a single-variable experiment.

**Action: none.** There is nothing to repair on the model side. The thing to fix is the measuring
instrument: give `bench-sweep.py` a unique prompt per request so per-level acceptance becomes
comparable — at the cost of breaking comparison with every number already published here, which is
why it has not been done yet. **k stays at 7.**

The hypothesis this section used to carry is kept below, because it was reasonable and it was wrong,
and because the two observations that "supported" it are the ones that should have raised the alarm:
acceptance was flat on the cold probe and equal at C8. **A regression that only appears at one
concurrency level is a property of the harness until proven otherwise.**

<details>
<summary>The original hypothesis, as written before the raw data was re-read</summary>

**The hypothesis:** the DFlash2 drafter was trained against a BF16 head, and this checkpoint's
`lm_head` is 6-bit, so the target's logits are perturbed relative to what the drafter expects. Two
observations are consistent with it and neither proves it. Acceptance is **flat** on the cold
single-prompt script (42.5 / 42.2 / 42.9 % against the control's 40.4–43.9 %), which says the effect
is small and workload-dependent rather than structural; and acceptance at **C8 is equal** (62.59 %
against 61.74 %), where the verify batch is larger.

**What would settle it, cheapest first** `[not tested]`:

1. Serve the full-scope checkpoint with `lm_head` forced to BF16 — the checkpoint has no BF16 head,
   so this needs a synthesised one and is more work than it sounds.
2. Re-measure acceptance against the *fallback* checkpoint on the same image and tree, isolating the
   checkpoint from everything else that moved.
3. A drafter aligned to a 6-bit head, which is the real fix and is somebody else's training run.

Worth about 3 % of single-stream throughput if fully recovered `[estimate]`. It is not worth a boot
on its own; it is worth carrying into the next arm that touches the drafter.

</details>

### 2.27 The four sm_12x stack patches — three ride with production 11, one stays out

**Closed as a question, open as an option.** Four findings against this stack on sm_12x hardware
([Zeuss5/cuda-exl3 issue #6](https://github.com/Zeuss5/cuda-exl3/issues/6), found by
`tpurtell/glm-5.3-flash-ext3-2x-rtx`, Apache-2.0) were re-implemented against our own anchors, put
behind runtime knobs and run as a five-arm A/B in a diagnostic image on 6 September 2026
`[measured-here]`. Production configuration 10 was never modified and the patches were verified
absent from the live container afterwards.

**The item that mattered was item 1, the PDL gate.** vLLM's `is_arch_support_pdl()` returns
`major >= 9`, so Programmatic Dependent Launch is on for sm_121 — a part on which it was never
qualified — across every KDA and mHC launch site, and 34 of this model's 45 layers are KDA. The
upstream report is of KDA recurrent-state races on exactly that path. **We could not detect one.**
Byte identity has no power here (the fused MoE epilogue's bf16 `atomicAdd` alone puts a varying
4.34e-3 per layer under everything), so the probe measures the run-to-run floor first: **the
between-arm p95 came out *below* the within-arm p95** — 6.2–6.5e-02 against 7.7–8.0e-02 — with no
prompt above K = 4 and, in the first of the two comparisons, later token divergence between arms
than within one. On speed the gate is a wash on 48 SMs: **C1 +0.8 %, C8 +0.7 %** pooled over six
PDL-on and three PDL-off rounds, both inside the band. The single number outside its band, C1
per-stream decode at +5.4 %, was deleted by a repeat: one PDL-on arm read 69.75 (a nominal +10.8 %),
the identical configuration repeated read 76.17, between the two PDL-off readings.

**Items 2 and 3 are the write and the read of one array** — an uninitialised top-k buffer whose
positive residue expands into real token indices — and they cost nothing measurable, including the
prefill memset the preparation note had priced as "not free on this part". **Item 4 is smaller than
the issue implies**: triton 3.7.1 gives an int argument three specialisation classes, not one per
value, and the tree's own warmup already covers most of them, so at most one cold compile per engine
process is at stake — and across cold, freshly booted processes we could not see it from the client
at all.

**The decision, and what happened to it.** Items 2 and 3 are correctness-class: their value rests on
a mechanism, not on a measurement, and **this A/B did not show that mechanism firing**. What it
showed is that fitting the insurance costs nothing. Those are different statements, and a change
without a reason is not made to a configuration that has passed a three-node reboot test — so the
verdict on the day was "adoptable, but not on a boot of its own; ride with the next production
change".

**That next change came the same afternoon.** Items 1, 2 and 3 are in **production configuration
11**, adopted alongside the 0.87 memory rung so that one boot carries both, and they now live in the
track's own patch tree ([`../tracks/tp3/patches/`](../tracks/tp3/patches/)) behind
`HAREM_SM12_ITEMS=pdl,kpool`. Measured against a same-session production 10 reference, the combined
change is **C1 +0.9 %, C8 +0.8 %**, both gates full cold and warm, tool-call 8/8, needle-lite 6/6 —
so the second, larger sample says the same thing the A/B did `[measured-here]`. **Item 4 is not
adopted** and stays in
[`../tracks/tp3/patches-optional/sm12/`](../tracks/tp3/patches-optional/sm12/): what it removes could
not be measured from the client at all, so there is nothing to weigh against the risk of carrying it.
Full tables, arms, floors and the two instrument caveats:
[`../results/kernels/sm12-stack-patches-ab.md`](../results/kernels/sm12-stack-patches-ab.md).

**What the campaign's instrument closed elsewhere, which is the larger result.** Arm 1 also carried a
diagnostic hook that read back the token-granular selection buffer, producing the datum
[issue #5](https://github.com/Zeuss5/cuda-exl3/issues/5) had asked for: **median 2,049 selected keys
per query row and a median adjacent-row top-k overlap of 0.9258** over 7,168 rows `[measured-here]`.
That is about **152 keys turning over per row**, roughly 76× the ~2 keys of the kernel author's
"drifting" arm — so production did not sit between his two arms at all. He built a third arm
calibrated to our turnover, swept context, and corrected his own conclusion in **`5fd7299`**
(*"Correct the MLA prefill ceiling: at production overlap there is no gap"*) `[reported]`: at **262K
context** the production-pattern arm runs within **1.6 %** of the fully cache-resident arm — 2,422.8 µs
against 2,385.8 µs — while the independent arm needs 3,474 µs, because the live key set is the
**residence window** (~4,096 keys, ≈4.5 MiB) rather than the chunk footprint and therefore fits even
a 24 MiB L2. **MLA prefill is compute-bound at production overlap, and the "21–26 % overlap gap,
about 2 % of a prefill chunk" quoted before this measurement does not exist. The item closes at
zero**; the only lever left on that kernel is reducing its work, not its traffic. This repository
never carried the 21–26 % figure in a document, so there is nothing here to withdraw — but it was
live in the thread the datum was produced for, and the correction is worth more than the datum. A
cheap falsification on a 48-SM part was [HELP-WANTED](../HELP-WANTED.md) §8; we have now run it.

**We ran it the same day, on our own hardware, and the prediction held at both head counts.**
[`results/kernels/mla-prefill-falsification-gb10.md`](../results/kernels/mla-prefill-falsification-gb10.md)
`[measured-here]`: his fixture, verbatim, at his 16-head shape and at our own 22-head TP=3 shape, over
the same three contexts. At 262K, production sits **1.3 %** above drifting at our 22-head shape —
tighter than the **1.6 %** he measured on his own 188-SM card — and **5.4 %** above drifting at his
16-head shape run here, against an independent arm that needs **82.0 %** and **146.8 %** more time
respectively. Across all six cells (three contexts × two head counts) production closes **96.0–98.4 %**
of the independent→drifting distance, and its excess over drifting at 262K (223–575 µs) is below one
cold read of its own working set. **Item 6 — his own numbering — now closes at zero on both parts, not
just his.** The run also sharpened the test rather than weakening it: this card's independent/drifting
spread (1.82–2.47×) is wider than his (1.46×) because its bandwidth-to-compute ratio is roughly 6×
lower, so the same leaked traffic would have cost about six times as much relatively — and it still
landed on drifting.

**Closing this item produced one new one.** At 22 heads — this stack's own TP=3 per-rank count — the
kernel's compute floor costs **13–16 % more per head** than at his 16, reproducibly at every context
and isolated to the compute path rather than traffic: the first concrete candidate for the "reduce the
kernel's own work" lever `5fd7299` says is all that is left. **And one caveat travels forward rather
than closing here.** The correctness check gating this fixture's timings ran at a small 2-head shape;
the harness grew a matrix for the real 22-head shape that same afternoon and its CPU-only self-test
passed (24/24), but the GPU run itself — the kernel call at 22 heads against a torch reference — is
still queued for an engine-free window `[not tested]`. Both the 22-head correctness run and the
per-head cost anomaly are [HELP-WANTED](../HELP-WANTED.md) §8 now.

**Two process corrections came out of the campaign and both cost more than the result.** The
diagnostic image had been built with only the first of the production image's two stages, so the
DFlash2 port's files were missing and a fail-closed anchor stopped the first dump boot — the build
note recorded the single stage without recording what it cost. And the GB10 top-k overlay is
bind-mounted **read-only** over the file two of these patches target, so those two had to be applied
to a host-side copy of the overlay instead of by the prelude. That exception applies to any future
patch that lands on an overlaid file.

### 2.28 The sparse indexer's K-gather workspace — opened, measured the same day, and waiting on disk

**Open as a decision, closed as a question** `[measured-here]`. The 7.28 GiB "non-torch" line of
[17](17-memory-ledger.md) had never been broken down, and the largest thing inside it turned out to be
a single buffer nothing reports: vLLM sizes the sparse indexer's K-gather workspace as
`40 × max_model_len` **entries**, a constant chosen upstream against DeepSeek-V3.2's 163,840-token
context where it comes to 825 MB. At our `max_model_len` of 1,000,000 the same constant reserves
40,000,000 entries × 132 B = **4.92 GiB**, allocated during the profile run, locked for the life of
the engine, and — because it is neither weights nor KV — subtracted from the budget before the KV
pool is sized.

**The buffer holds one indexer chunk's compressed context and nothing else.** Its exact ceiling here
is `max_num_seqs × ceil((max_model_len + num_spec + 1) / index_kpool)` = 2,000,016 entries =
**251.8 MB**, and upstream's own DeepSeek-V4 call site divides by that `compress_ratio` where the
glm5next path does not. Bounded to 512 MB — 2.03× the ceiling, 16.3× the one-request correctness
floor — on a same-session A/B at the production-11 memory fraction:

| | control (knob off) | patched (`HAREM_INDEXER_WS_MODE=bound`) |
|---|---:|---:|
| Locked workspace, all three ranks | 5,036.40 MB | **513.00 MB** |
| **KV pool** | 6,289,256 | **6,933,884 (+10.25 %)** |
| C1 / C8 aggregate tok/s | 69.69 / 199.76 | 70.69 / **196.81** — both in band |
| Gates cold · warm, tool-call, needle-lite | 10/10 · 12/12, 8/8, 6/6 | 10/10 · 12/12, 8/8, 6/6 |
| One 969,468-token request · eight concurrent long-context lanes | not run | **PASS · 8/8 PASS** |
| Safety layers fired | — | **none** |

**The hypothesis was tested before the patch was, which is the part worth copying.** An upstream
environment variable, `VLLM_DEBUG_WORKSPACE=1`, prints every workspace resize with its caller. The
control arm showed **one** resize event on each rank, grown by `sparse_attn_indexer_kpool.py`, at
exactly the 5,036.40 MB the code reading predicted — so the gain was not capped by some other
consumer, which was the most likely way for the whole item to be worth nothing.

**What it cost.** Diagnostic margin: 20× the largest load the scheduler can produce, down to 2.03×
against that ceiling and 16.27× against the one-request floor, which is the only route to an over-size
chunk — held by a startup refusal at that floor plus two armed run-time assertions either side of the
slice. What an under-size would actually do is a **silent clamp producing wrong answers**, not the
out-of-bounds device write this page first described, and upstream's locked-workspace assertion,
which we had counted as the load-bearing layer, cannot fire from this path — §1.13, corrected by the
issue author. Speed: nothing measurable, and
it was looked for at five concurrency levels, in fresh prefill and in TTFT. Host headroom: nothing,
in either direction — the freed memory goes straight into the pool, so `MemAvailable` is unchanged
and this is **not** a lever on the rung where 0.90 failed (§2.4). It stacks with that ladder.

**Why it is not in production, and it is not doubt.** Installing it into the production tree changes
the fast-load manifest identity, so it forces a fresh sidecar of about **53 GB per node** — and two of
our three nodes have 36 and 39 GB free. An older sidecar has to be deleted first, which is the stack
owner's decision rather than a measurement's. The A/B itself was run from a **copy** of the tree with
fast-load off for exactly that reason, which is also the item's one open caveat: both arms are eager
boots, so the production pool figure with fast-load restored is an `[estimate]` (≈7.03M) until a
promoted boot prints it.

Full tables: [`../results/memory/indexer-workspace-ab.md`](../results/memory/indexer-workspace-ab.md).
The patch and its knobs:
[`../tracks/tp3/patches/indexer-workspace/`](../tracks/tp3/patches/indexer-workspace/).
The mechanism in its ledger context: [17](17-memory-ledger.md) §2.5. **The upstream half of it — a
sizing constant that grows linearly with `max_model_len` and skips a division the sibling path
applies — was reported upstream on 3 September, three days before we measured it, as part 2 of
[vllm-project/vllm#55221](https://github.com/vllm-project/vllm/issues/55221); our confirmation and
three additions are [in that thread](https://github.com/vllm-project/vllm/issues/55221#issuecomment-5559336290),
and what is still open is [HELP-WANTED](../HELP-WANTED.md) §9.**

---

### 2.29 CUDA graphs at three ranks — off because FlashInfer's support gate does the wrong division; filed upstream, and left off

**What the log says, and what it means.** Since production 7 (fp8 draft KV) every TP=3 boot prints

```
CUDAGraphMode.FULL_AND_PIECEWISE is not supported with spec-decode for attention backend
FlashInferBackend (support: AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE); setting cudagraph_mode=NONE
```

and until 6 September this page, [10](10-results-and-roofline.md) §5.8 and [14](14-troubleshooting.md)
§7.7 read that as a capability — "FlashInfer cannot capture an 8-token verify batch". It is not a
capability. `FlashInferMetadataBuilder.get_cudagraph_support()` (`flashinfer.py:985` on our image,
`:997` on `main` @ `52358e6e`) declares `UNIFORM_BATCH` only when `num_qo_heads % num_kv_heads == 0`,
and it takes `num_qo_heads` from the **target** model (`model_config.get_num_attention_heads`) while
`num_kv_heads` comes from the KV-cache group being asked about — on this engine the **drafter's**, since
the target's attention is on cuda-exl3's custom backend and never goes near FlashInfer. The builder
*instance* that actually runs takes its head count from the group's own layers (`flashinfer.py:759`,
"compatible with models with non-uniform per-layer head counts"), passes the same test, and selects the
XQA decode kernel — `decode_backend=xqa` is three lines above the warning in the same log, and the
`Using FlashInfer for draft model non-causal attention` line that follows is only printed when that
selection succeeded. Two functions in one file disagree, and the engine believes the wrong one
`[measured-here]`.

| per rank | TP=2 | **TP=3** |
|---|---|---|
| target query heads (what the gate uses) | 64/2 = 32 | 66/3 = **22** |
| draft query heads (what the builder uses) | 32/2 = 16 | 36/3 = **12** |
| draft KV heads (`spec.num_kv_heads`) | 8/2 = 4 | 9/3 = **3** |
| the gate's division | 32 % 4 = 0 | **22 % 3 = 1** |
| the builder's division | 16 % 4 = 0 | 12 % 3 = 0 |
| declared | `UNIFORM_BATCH` | `UNIFORM_SINGLE_TOKEN_DECODE` |
| result | `Graph capturing finished in 11 secs, took 1.10 GiB` | `cudagraph_mode=NONE`, `0.0 GiB for CUDAGraph memory` |

Gates, acceptance and MMLU are identical between the two tracks ([15](15-tp2-track.md) §5.8), which
is the other half of the evidence: the kernel the gate says cannot be graphed is the kernel serving
both. The "22-head" history on the NVFP4 sibling was a different thing — a decode kernel computing
wrong results at that shape; here 22 is only a number fed to a remainder.

**Upstream.** [vllm#55581](https://github.com/vllm-project/vllm/issues/55581) (ours, 6 September 2026): the fix is to derive the head count from the group's own
layers, as the builder already does — the call site holds `kv_cache_group.layer_names`. Related but not
the same: #49547 (FlashInfer's *native* decode path genuinely capped at single-token decode, measured
+16 % on GB10 — a different code path from XQA), #53030 (PIECEWISE + spec-decode silently rejects every
draft, which is why `cudagraph_mode=PIECEWISE` is not a workaround), and #40969 (GB10 hang with
`FULL_AND_PIECEWISE` + chunked prefill on DeepSeek-V4-Flash — the risk on the day the gate is fixed,
because TP=3 + expert parallel + graphs ran a full battery on 6 September without a hang, see §2.29;
TP=2 runs that mode for hours).

**What it is worth.** Two independent derivations ([10](10-results-and-roofline.md) §5.8's gap census
on production 7, and the production-9 launch count of 2,738 kernels/step) put the ceiling at
**+1.2–2.3 % single-stream, +0.5–0.8 % at C8, ≤0.5 % of prefill** `[estimate]` — every one inside the
band we grade in (±4 % / ±3 % / ±3 %, [09](09-measurement-protocol.md) §1.2), so the gain could not be
confirmed on our own bench even if it were real. The one graphs-on boot we have (production 6, bf16
draft KV) read the same single-stream tok/s as its graphs-off successor, consistent with that ceiling.

**What it costs.** The graph pool comes out of the KV budget. The two-node arrangement of this image
charges **1.10 GiB** per rank for it `[measured-here]`; the memory profiler's estimator reserves up to
**2.64 GiB** (its 0.85 → 0.8283 equivalent on that boot). At three ranks that predicted **−2 to −5 %
of the pool** `[estimate]`. **Both figures are now superseded by a three-rank measurement below**:
the capture charges 0.25–0.27 GiB per rank, the total price is −3.85 %, and the mechanism is the
activation reservation rather than the pool. The one env-only way to get graphs back today is to drop
`HAREM_DRAFT_KV_DTYPE=fp8` (drafter → FlashAttention → `UNIFORM_BATCH`): predicted −4.5 % of pool for
the bf16 cache and −6.5 to −9.5 % with the graph pool on top, **measured −7.67 %**.

| route | feasible | needs | risk | price |
|---|---|---|---|---|
| `cudagraph_mode=FULL_DECODE_ONLY` | **no** — `decode_mode()` is `FULL`, same gate | — | — | — |
| `cudagraph_mode=PIECEWISE` | no | — | #53030: every draft silently rejected | — |
| patch the gate (one anchor, `flashinfer.py:985`) | yes | a new `patch-*.py`, which **re-dumps the fast-load sidecar** ([08](08-fast-boot.md)) | low, now that EP + graphs have run a full battery here | KV **−3.85 %** `[measured-here]` |
| draft on `TRITON_ATTN` (declares `ALWAYS`) | possible | a selector patch, full gate battery | high: never measured, numerics unknown | unknown |
| draft KV back to bf16 | **one env line**, sidecar stays valid | nothing | low | KV **−7.67 %** `[measured-here]` |
| leave it | — | — | none | **+1.75 % at C1, unmeasurable on this bench** `[measured-here]` |

**The A/B that would settle it — designed, then run the same evening**: three arms on the
production-12 tree, all env-only, no dump. **A** = production 12. **B** = A without
`HAREM_DRAFT_KV_DTYPE=fp8` (graphs on). **C** = B plus `ENFORCE_EAGER=1` (same KV geometry as B,
graphs off), so **B − C is the graph lever alone** and A − C is the fp8-draft lever.
`ab-quick2-full.sh` per arm — three sweep rounds, TTFT, prefill, acceptance, cold + warm gates, the
tool-call gate. Kill criteria: any gate short; acceptance outside 60–65 % (an approach to 1.00 is the
#53030 signature); a hang or `EngineDeadError` (#40969); swap-in under load; and **B − C under +2 %
at C1 closes the item** without the patch being written.

### Run on 6 September: the lever is +1.75 %, the pool it costs is 3.85 %, and the item closes

One 53-minute engine window, arm A measured on the live production boot so only two boots were
spent, arms B and C booted from `.env.tp3`-derived files on each node `[measured-here]`. Neither knob
enters the fast-load manifest identity (`file_identity()` hashes the image tag, TP size, EP flag,
node rank, the filter suffixes and every `patch-*.py` plus the prelude — not which of them runs), so
all three arms loaded the production sidecar and booted in 217–275 s.

| median of 3 rounds | A = production 12 | B = bf16 draft, graphs **on** | C = bf16 draft, graphs off | **B − C** | band |
|---|---:|---:|---:|---:|---|
| C1 agg tok/s | 69.90 | 71.67 | 70.44 | **+1.75 %** | ±4 % |
| C2 | 100.05 | 103.30 | 102.10 | +1.18 % | ±6 % |
| C4 | 145.32 | 144.76 | 147.34 | −1.75 % | ±9 % |
| C6 | 171.89 | 168.47 | 177.70 | −5.19 % | ±6 % |
| C8 | 195.78 | 198.18 | 201.97 | −1.88 % | ±3 % |
| C1 per-stream decode | 73.64 | 79.51 | 76.12 | +4.45 % | — |
| TTFT C1 / C8 (s) | 0.283 / 0.796 | 0.334 / 0.861 | 0.272 / 0.797 | | |
| prefill-fresh / prefill7k | 1,746 / 1,552 | 1,755 / 1,629 | 1,736 / 1,607 | +1.1 % / +1.4 % | ±3 % |
| acceptance C1 / C8 | 62.1 / 60.5 % | 61.4 / 62.5 % | 60.6 / 61.9 % | | 60–65 % |
| **KV pool (tokens)** | **7,077,134** | 6,534,435 | 6,796,143 | **−3.85 %** | |
| graph pool, ranks 0/1/2 (GiB) | 0 / 0 / 0 | **0.27 / 0.25 / 0.00** | 0 / 0 / 0 | | |
| peak activation, rank 0 (GiB) | 1.66 | **2.62** | 1.66 | | |
| boot (s) | 217 | 275 | 255 | | |

**Every one of the five levels is inside its band and the signs are mixed** — C1 and C2 up, C4, C6
and C8 down. Two more numbers from the same run say why that is the whole answer: the **round-to-round
spread inside a single arm** was 3.52–10.69 % at C1 (per-stream 4.07–9.15 %), and B and C are separate
boots, where this stack's spread is 15.9 % ([09](09-measurement-protocol.md) §2). The signal is
smaller than either ruler. The per-stream +4.45 % is the one figure that grazes a band, and it is
buried by the same two.

**The graph pool at three ranks, measured for the first time — and it is a quarter of what this page
estimated.** `Graph capturing finished in 44 secs, took 0.27 GiB` (rank 0), 43 s / 0.25 GiB (rank 1),
43 s / **0.00 GiB** (rank 2) `[measured-here]`, against the 1.10 GiB carried over from TP=2 and the
2.64 GiB estimator figure above. The **−2 to −5 % of pool** estimate nevertheless came out right —
**−3.85 %**, 261,708 tokens — but for a different reason than it gave: the capture pool is nearly
free, and the cost is the **peak-activation reservation the capture forces up, 1.66 → 2.62 GiB on
rank 0**. Estimate right, mechanism wrong; recorded as a correction, not a confirmation.

**Where B's whole 7.67 % against production 12 goes.** Per-token cost 7,599 → 8,036 B (**+5.75 %**,
the bf16 draft page) and 2.35 % less memory left for KV (graph pool plus activation). **The pool is
lost to the bf16 draft cache, not to the graphs** — which reverses the emphasis of the row above.

**A − C, the fp8 draft cache, at equal graph state**: C1 −0.77 %, C2 −2.01 %, C4 −1.37 %, C6 −3.27 %,
C8 −3.06 %, per-stream C1 −3.26 % — 1–3 % slower for **+4.14 % of pool**. Read it as the whole draft
route, not as a dtype: A runs FlashInfer + the XQA kernel + fp8, C runs FlashAttention + bf16.

**Gates and the #40969 question.** Correctness 10/10 and code 12/12 cold and warm on all three arms,
tool-call 8/8 on all three; acceptance stayed in 59.4–65.6 % across every round of every arm, with no
approach to 1.00, so the #53030 signature did not appear. Arm A's first warm code run read 11/12
(`matrix`), passed 12/12 on an immediate re-run on the same engine, and is recorded as a single-item
flake rather than a rejection — the same re-run rule was in force for B and C and neither needed it.
Swap-in was 0 on all three nodes throughout, `MemAvailable` bottomed at 2–3 GiB, and the throttle
mask was `0x0` on every sample of every node, so no arm's numbers are thermal. **And the combination
this page listed as an unrun risk — TP=3 + expert parallel + `FULL_AND_PIECEWISE` + chunked prefill —
ran a full battery without a hang, an `EngineDeadError` or an error line.** That risk row can be
retired; the lever it guarded still cannot be measured.

**Verdict, 6 September: left off, now on a measurement rather than an estimate.** B − C at C1 is
+1.75 %, under the +2 % the design set as the closing threshold, and smaller than the spread between
rounds of one arm. The gate declaration stays filed as [vllm#55581](https://github.com/vllm-project/vllm/issues/55581); **the patch is not written**. What
changed on this page besides the result: §1.9 row 8's "the sidecar removed the obstacle" was wrong
and is retracted in §1.12, and the graph-pool price above is corrected downward in
[17](17-memory-ledger.md) §2.3.

### 2.30 The tool-eval gap against the NVFP4 sibling — the template is eliminated, the build and the weights are still varying together

**The measurement.** On production 12, tool-eval-bench (hardmode, 88 scenarios, 8 trials, seed 42,
temperature 0, effort `low`) scores **85.5 ±1.3** against the NVFP4 sibling recipe's **87.8 ±0.9** on
the same harness version and the same flags, 6 September `[measured-here]`. It is not trial noise —
Welch t = 4.02, exact permutation p = 0.0048 — and it is **four scenarios out of 88**: with TC-51,
TC-21, TC-74 and TC-87 removed, this stack is **+0.22 ahead** on the other 84. Nine of fourteen
categories are identical digit for digit; the only failure codes recorded on either side are
`wrong_args` and `missing_step`; malformed calls, timeouts, empty content, refusals and schema
violations are **zero in 1,408 scenario-runs on both stacks**. The same engine the same night scored
97.5 % on GSM8K against the sibling's 94.0 and beat it on both IFEval axes. Full breakdown:
[`../results/gates/quality-battery-production-12.md`](../results/gates/quality-battery-production-12.md).

**What is already settled, and it is most of the item.** TC-51 alone is 47 % of the gap and it is a
**grading rule, proven from the grader's source**: `call.turn <= valid_event.turn` fails a
notification issued in the *same* turn as the event it announces, and this stack batches those two
dependent calls into one turn with the plan, the arguments, the recipients and the date all correct.
The harness invites the batching — it sends `parallel_tool_calls: true` on every request. And the
**chat template hypothesis was tested and refuted the same night**: production 12 was rebooted with
the 27 August template (`04c4e9e9`) as the single changed line and TC-51 and TC-21 — 72 % of the gap
between them — stayed **0/8**, with the same trace and the same `parallel tool turns: 2` diagnostic.
The remaining block favours the old template by +1.12 points per trial at **p = 0.20**, CI
[−0.45, +2.70]; as a share of the gap that is a point estimate of **28 % with a 95 % interval of
[−11 %, +68 %]**, which contains zero. **The new template stays in production**: the gain is
unmeasurable and the old template carries three real rendering defects, all on an agent's hot path.

**What is open.** At least 72 %, and possibly all, of the −2.3 points sits with two variables that
changed together and have not been separated: the **checkpoint** and the **vLLM build**
(`0.1.dev20051+g487ecf187` against `0.1.dev0+lil.jovian.9c4dd0548`), the second of which carries a
different speculative-decoding implementation with it. Confidence that this stack's agentic quality is
not meaningfully below the sibling's is **high**; confidence about where the residual ~1.6 points
lives is **low**, and saying otherwise would be inventing a result.

**Next step — the build, not the weights.** Same checkpoint, same template, this cluster, the vLLM
build as the only variable, run on the eight scenarios of the template arm rather than the full 88:
about 12 minutes of engine time per arm plus two boots. A same-session NVFP4 control is **not**
available — that line is closed and its stack no longer stands on these nodes — so the split has to be
made from this side. Until it is, no line in this repository should attribute the gap to the
quantization, and none does.

**One behavioural note for agent users, because it reproduced under every arm.** This stack will batch
two *dependent* tool calls into a single turn — create the record and announce it, without waiting to
see whether the first call succeeded. It is efficient and usually right, and it is wrong wherever the
second call must not happen if the first one failed, or wherever a framework grades on turn ordering.
Constrain it in the system prompt or send `parallel_tool_calls: false`; that setting has never been
measured here `[not tested]`.

**Update, 8 September — a second reading that narrows the item without closing it.** The same engine,
nothing changed on it, scores **90.3 ±1.2** on the same 88 scenarios at reasoning effort `high`
(three trials, `final_score` 91, 160/176), which is **above** the sibling's `low`-effort 87.8 ±0.9 —
Welch t = 3.25, exact permutation p = 0.012 `[measured-here]`. Three of the four scenarios the gap
was concentrated in move: **TC-21 0.00 → 2.00** (it finds 5/5 validation errors instead of 1/5, on a
tool-free reasoning task), TC-87 1.25 → 2.00, TC-74 1.25 → 1.67. TC-51 goes 0.12 → 0.67 and stays the
largest hole, which is consistent with the grading rule above being the cause rather than the
reasoning. The effort was raised by a proxy that injects `reasoning_effort` into the harness's
requests, so the server was never restarted or reconfigured
([`../scripts/effort-proxy.py`](../scripts/effort-proxy.py),
[`../results/gates/quality-battery-production-13.md`](../results/gates/quality-battery-production-13.md)).
**What this does and does not do.** It says the `low`-effort deficit is a thinking-budget effect on
planning-shaped scenarios rather than a capability difference. It does **not** separate the build from
the weights — the sibling has not been run at `high` effort and cannot be, so the arms are not
symmetric — and the build A/B above is still the next step. The item stays open.

**Field corroboration, 8–9 September (issue #1).** A production deployment of this recipe — ten
users, four days, real agentic coding traffic at `reasoning_effort: low`, temperature 1.0 — reached
the same conclusion in the same direction without knowing this benchmark existed: a consistent user
preference for the NVFP4 sibling on identical work, severe enough to trigger a rollback. That window
predates the K-pool tail fix
([`../tracks/tp3/patches/prefix-hit-and-kpool-tail/`](../tracks/tp3/patches/prefix-hit-and-kpool-tail/README.md))
and ran with the reasoning-retention default of [docs/14](14-troubleshooting.md) §9.12, so it does
not isolate the effort effect; it does say the `low`-effort gap is not academic. The operational
advice follows from the `high` measurement above: **for agentic traffic make `high` the default and
reserve `low` for cost-sensitive, short-horizon clients**, with the price stated next to it — about
+90 % completion tokens and +24 % turn latency on the same scenarios. The build A/B remains the next
step; jdecker76 has offered a same-cluster NVFP4 control arm, which would be the first symmetric one.

### 2.31 The vision tower is in production, and five things about it are not settled

The tower ships ([18](18-vision-at-three-ranks.md), production configuration 13, 7 September 2026):
4 images and 2 videos per request, ten gates out of ten, no measurable cost on any speed or pool
axis. What is *not* closed, in the order we would take it:

1. **The upstream video sampler fix is filed and waiting on a maintainer.**
   [vllm#55644](https://github.com/vllm-project/vllm/issues/55644) is the issue,
   [vllm#55647](https://github.com/vllm-project/vllm/pull/55647) the fix plus a unit test, DCO-signed
   and **awaiting a ready label**. It is not ours and not conditional on anything here:
   `Glm5NextProcessingInfo` does not override `_get_video_second_idx_glm46v`, so the placeholder count
   comes from GLM-4.6V's frame sampler while the pixels come from GLM-5-Next's, and the mismatch kills
   the engine core on the **first video request** at any parallel size. The ratio moves with the
   clip's duration — **3.0 below 30 s, 1.0 between 30 and 300 s, 0.5 and lower above 300 s** — so the
   only band that works is a coincidence and long clips are fatal too. Two smaller items ride along in
   the issue: the `min(user, supported)` clamp that makes `--limit-mm-per-prompt {"video": 2}`
   silently mean 1, and the sampler that returns zero frames for a clip under half a second.
   **Next step: nothing on our side until it moves.** [18](18-vision-at-three-ranks.md) §12.
2. **The brake ladder has one rung.** `max_image_tokens` is at 8,000 tokens per item, chosen because
   the tower runs a video in one forward and the unbraked ceiling is ~14.6 GiB of SwiGLU buffer
   `[estimate]`. 16,000 would put the peak at 0.98 GiB and is the obvious next rung, with K4/K4b/K5/K7
   repeated at each. Not run `[not tested]`. Nothing at the current ceiling has been served either:
   every fixture is four seconds long.
3. **Image quality is unbenchmarked.** The gates prove the tower loads correctly and answers
   correctly on synthetic fixtures — a colour, a shape, a direction, a word. No MMMU, no DocVQA, no
   ChartQA, and no comparison of this 6-bit EXL3 tower against a BF16 one `[not tested]`. This is the
   largest gap on the page and it is a benchmark-time problem, not a design one.
4. **The tower audit line does not print on a fast-load boot.** VS3's audit lives inside
   `load_weights`, which the sidecar path skips. The replacement evidence chain runs on every boot
   and was counter-tested five ways ([18](18-vision-at-three-ranks.md) §8.1), but moving the audit
   past `process_weights_after_loading` would restore the direct line. It costs a sidecar dump, which
   is why it did not happen inside the promotion window `[not tested]`.
5. **The concurrency price is measured and unattributed.** A 4-image + 2-video request sent while a
   C8 round runs costs the text arm **−12.7 %** — contention at `--max-num-seqs 8`, where a ninth
   request queues with a 5,991-token prefill. How much of that is *multimodal* rather than simply a
   long ninth prefill was never separated, because the equivalent text request was not run
   `[not tested]`.

And one thing that is closed and worth stating as closed: **speculative decoding keeps working with
images and video.** The drafter never sees pixels (`supports_mm_inputs = False`) and reaches the
multimodal context only through the target's hidden states; acceptance measured 60.4–63.9 % against
production 12's ~62 %. The code permitted this and nothing had demonstrated it.

---

### 2.32 The hybrid prefix-cache hit — nothing flags the drafter, so everything gets flagged

**CLOSED for this stack, 8 September 2026: `HAREM_PREFIX_HIT=1` is in the recipe.** Still open
upstream, and still open for anyone on a vLLM without #52047. Patch, mechanism and full numbers:
[`tracks/tp3/patches/prefix-hit-and-kpool-tail/`](../tracks/tp3/patches/prefix-hit-and-kpool-tail/README.md),
[`results/gates/prefix-hit-and-kpool-tail.md`](../results/gates/prefix-hit-and-kpool-tail.md).

`KVCacheCoordinator.__init__` collects the KV cache groups carrying `is_eagle_group` and, when
speculative decoding is on and that set is empty, conservatively flags **all** of them. The only
function in this image that ever sets the flag is `_annotate_eagle_groups_deepseek_v4`, which
requires `model_version == "deepseek_v4"` and is reached from one branch of `get_kv_cache_groups`.
GLM-5.3-Flash with a DFlash2 drafter takes a different branch — the one this recipe added, which
groups the target and the draft separately ([docs/04](04-dflash2-port.md)) — so no group is ever
flagged and the fallback has fired on every boot this stack has run.

What it costs, measured on the production configuration before anything changed `[measured-here]`:
an exact repeat of an 8,008-token prompt hits **41.56 %** against a **83.12 %** ceiling, and of a
59,910-token prompt **94.43 %** against 99.99 %. The loss is **exactly one 3,328-token block**, in
every scenario, on every repeat. With the patch: 83.12 % at 8K — **the ceiling** — and follow-up TTFT
from 2.830 s to **1.076 s, −62.0 %**; a four-turn agent conversation goes the same way.

**Why it was held for several hours, and what promoted it.** The acceptance bar was a *raw* hit
ratio of 95 %, which the 3,328-token granularity makes unreachable at 8K; that bar is now restated in
ceiling terms, and by that reading the patch is perfect at every size tested. The 60K case genuinely
does not move — its offset past the aligned boundary is 6 tokens, so the drafter's own drop has no
block to give back and costs a full 3,328 on re-alignment (about 7.7 % of prompt lengths are in that
window), and that residual is unchanged by the promotion. The other reason was one boot of the
both-knobs arm returning needle-lite 5/6 twice. It was promoted after **two further boots through the
autostart unit with nine needle-lite runs at 6/6**, three code exams at 12/12 on the first attempt,
and a **cached-path equality test** built for that exact failure mode: 24 prompts at ~11K, ~84K and
~178K tokens, each asked cold and then repeated byte-for-byte with the cache counters read either
side, **24 of 24 answers byte-identical and correct**, mean repeat hit 94.7 % against 0.0 % cold
`[measured-here]`. The two 5/6 now stand at two failures in eighteen runs, both inside one boot.
[`results/gates/prefix-hit-and-kpool-tail.md`](../results/gates/prefix-hit-and-kpool-tail.md) §6.

**If this stack rebases onto a vLLM carrying #52047**, our anchor disappears and the right move is
#54041's marker — `non_causal_multi_token_decode` on the drafter's `SlidingWindowSpec` — not to
re-add ours. The patch fails closed on a drifted anchor, so it cannot pass silently.

### 2.33 The K-pool tail slot mapping — a real correctness bug, measured, and now shipped

**CLOSED for this stack, 8 September 2026: `HAREM_KPOOL_TAIL_FIX=1` is in the recipe.** Still open
upstream, and still open for every hybrid model on an unpatched vLLM.
**Not ours:** found, reproduced and fixed by [vcruz305](https://github.com/vcruz305)
([CREDITS](../CREDITS.md)) on the same base image in August. We had been serving it for a fortnight.

`KpoolTailSpec` is a one-block circular scratch cache — `max_num_blocks_per_req() == 1`,
`block_size == index_kpool == 4` — addressed as `block_table[req, 0] * kpool + pos % kpool`. Its slot
mapping is produced by the generic per-group paged kernel, which indexes
`block_table[req, pos // block_size]`; only column 0 of that row is ever written, so from `pos >= 4`
the mapping addresses a block that is not the request's own, and neither kpool write kernel
bounds-checks it. The corrected mapping is already in the image and is skipped because
`v1/worker/gpu/model_states/mamba_hybrid.py` calls `build_attn_metadata(...)` without `positions=`
while `model_states/default.py` passes it. Every hybrid model is affected.

Measured on our own engine with a counter on the mapping the engine actually uses `[measured-here]`:
`positions` reached the tail builder on **0 of 896** steps; **15,578 of 15,630** tail tokens —
**99.67 %** — addressed a block that was not the request's own; the worst block written was **332**
against a largest-owned block of **275**, so those writes land outside every ring any request holds.
With the fix: **0 of 16,806**, worst block 275, positions on 896 of 896 steps. A model-free unit test
in the serving image pins the same thing 4/4.

**One detail of the published diagnosis does not hold here.** vcruz305 describes the block-table row
as one entry wide, so that `pos >= block_size` reads past it. Our runner sizes that row
`cdiv(max_model_len, block_size)` = **250,016 entries**, so there is no out-of-row read at all and
our detector counts zero row overruns while the mapping is wrong for 99.67 % of tokens. The harm is
the wrong block, not the overrun — and on this build there is nothing for a clamp to clamp, which is
the second independent reason the clamp is the wrong layer (his own measurement, that it changed
nothing, is the first).

**What held it, and what promoted it.** Its own arm was clean on everything except the battery run
after a soak, which lost the code exam's `matrix` item — and that item then failed on the **restored,
unpatched production** on its first run and passed four times after, so it is a flaky item and was
withdrawn as evidence. It went in the same night alongside §2.32, on two boots through the autostart
unit with every gate full and a third boot to prove the unit brings the configuration up by itself.
Nothing was found that it costs: KV pools of 7,044,077 / 7,041,322 / 6,947,658 against a documented
6,914,600–7,143,250 spread, and speed unchanged on the same-session A/B that measured it
`[measured-here]`.

**What this session actually established, and it is not about either patch.** We adjudicated two
changes with gates whose own flake rate nobody had measured. `matrix` failed 2 of 12 code exams,
including once with no patch tree present. **A gate baseline — every gate, ten runs, cold and after a
soak, on the production configuration — is now the cheapest useful measurement on the list**
([HELP-WANTED](../HELP-WANTED.md) §12).

### 2.34 A repeat that arrives later hits nothing — every hit ratio here is a back-to-back number

**OPEN, measured 8 September 2026.** Mechanism `[not tested]`.

Every prefix-cache hit ratio this repository publishes — §2.32, the production baseline, all of
[`results/gates/prefix-hit-and-kpool-tail.md`](../results/gates/prefix-hit-and-kpool-tail.md) §1 and
§2.1 — comes from asking a prompt and then immediately asking it again. The first version of the
cached-path equality test asked 24 distinct prompts cold and then repeated all 24 in the same order.
**All 24 repeats hit 0.0 %**, and each took the same wall time as its cold pass — 97.5 s on a
177K-token prompt, against 1.1 s when the same prompt is repeated immediately `[measured-here]`.
Twenty-three long requests in between leave nothing of the first one behind, on a pool of 7,041,322
tokens against about 2.2 M of prompt.

So "the same prompt again, immediately" and "the same prompt again, later in a busy hour" are
different measurements on this stack, and only the first has ever been measured here. That matters
for reading the TTFT and hit numbers as a production expectation: an agent that re-sends its history
on the next turn is the first case, a user returning to a conversation after other traffic is the
second.

**What decides it is not measured.** The plausible reason is the hybrid model's non-caching groups —
the four `MambaSpec` KDA groups hold per-request state whose slot count is small and unrelated to the
token capacity of the pool, so a coordinator hit needs every group to still hold its part. That is a
hypothesis, not a measurement, and this stack's own rule is that a mechanism nobody measured is not
an explanation. What would settle it: vary the number of intervening requests and find where the hit
ratio falls off, and read the per-group block counts either side.

The model's answers were identical and correct in all 24 uncached repeats, so nothing here is about
output quality. Instrument: `cached-equality.py --order passes` in
[`tracks/tp3/patches/prefix-hit-and-kpool-tail/`](../tracks/tp3/patches/prefix-hit-and-kpool-tail/README.md).

## 3. Never run

| What | Why not |
|---|---|
| Anything at **max reasoning effort** | Days of cluster time. Everything published here is at `low`. |
| **MMLU at TP=3 on the fallback checkpoint** | Now run on the *production* checkpoint at TP=3 (86.47 ±0.74). The 86.4 ±0.7 the fallback carries is still a TP=2 figure, so the two are not a like-for-like pair; the gates are identical between arrangements, so there is no signal that justifies the hours — but it is an absence. **6 September: the fallback checkpoint and its image (`62f53e6`) were removed from our cluster to make room; we will not run this.** |
| ~~**IFEval, GSM8K, tool-eval-bench**~~ — **run on 6–7 September** | No longer an absence. All three ran on production 12 against the NVFP4 sibling's 3 September battery at the same harness settings: GSM8K **97.5 %** against 94.0, IFEval **80.0 / 86.0 %** against 78.9 / 85.1, tool-eval-bench **85.5 ±1.3** against 87.8 ±0.9. The tool-eval gap is four scenarios out of 88 and is open as §2.30. [`../results/gates/quality-battery-production-12.md`](../results/gates/quality-battery-production-12.md). |
| ~~**Needle-in-a-haystack at 1M**~~, and the full 14,042-question MMLU | Both were staged in the 6 September battery and **deferred on time** before either started. **The needle is now run — 20/20 on 8 September**, four haystack sizes to 997,952 tokens by five depths with every cell retrieved, 1 h 25 min 32 s against the sibling's identical 20/20 in 5,288.6 s ([`../results/gates/quality-battery-production-13.md`](../results/gates/quality-battery-production-13.md) §8) `[measured-here]`. The **full MMLU** is still outstanding; the sibling has 85.9 ±0.3 and what exists here is the **1,995-question sample at TP=3 on this checkpoint, 86.47 ±0.74** `[not tested]`. |
| **ExtractBench Short** | Never run on this stack `[not tested]` — and, checking the claim this row used to make, **never run on the NVFP4 sibling's shipped build either**: what that recipe carries is a 94.51 against an H100 FP8 reference's 96.46 on the 215-document set from an **earlier, different** build, which it says explicitly must not be quoted as a figure for the build it ships `[reported]`. So this is a dimension neither line has measured, not one where we are behind. |
| **The newer checkpoint revision** (`aba59d21`, four days newer than the one we pinned) | Not tested. |
| **`NCCL_MAX_NCHANNELS=8` on the NVFP4 stack** | Same plugin, same fabric, same TP=3, so it should transfer — one line per node, reversible. Not applied there. |
| **The mesh plugin patches on the NVFP4 stack** | The idle second cable and the host bounce buffer are properties of the fabric and the plugin, not of the quantization, so both should transfer and are worth more there than the channel cap. Not applied. |
| **KDA linear attention's efficiency** | 8.1 % of a prefill chunk and 8.0 % of a C8 step, triton chunked scans, never measured against a ruler. It inherits the empty-denominator slot MLA prefill just vacated. |
| **A profiler-off re-run of the same decode window** | The CUPTI subtraction that turns 5.45 ms of C1 idle into 3.47 ms is an inference from two walls (§1.10), not a direct measurement. Separating it cleanly needs the profiler off and the window re-opened: one boot, and the answer would move a 3.75 % figure by a fraction of itself. |
| **A watchdog on the running container** | A 60-second `docker ps` plus `/health` poll that dumps `docker logs --tail 40` when the container is gone. Few lines, no root, independent of the unit — and it would have caught the one-hour outage the autostart unit does nothing about. §2.20. |
| **A rank-ordered start under systemd** | The unit starts all three ranks concurrently and the reboot test passed on the rendezvous retrying, not on ordering. One trial. A rank-dependent delay or peer-port poll in `ExecStartPre` is the guarantee; not written. §2.20. |
| **Quantizing the four BF16 families ourselves** | Ruled out rather than untried, and the reason changed on re-measurement: cold on the target GPU the KDA gating arms are **neutral** (+0.050 ms/step, 0.07 % of C1) rather than 1.6–1.8× slower, so they stay BF16 for want of a gain rather than for a loss. The three measured families together are **+1.368 ms/step** against a 1.5 ms gate, **96 % of it in `kv_b_proj` alone**, which needs a batched EXL3 kernel that does not exist. Prefill was not re-measured and the gate is an *and*. §2.25, [13](13-full-scope-checkpoint.md) §4.4, [`../results/kernels/kda-gate-bench-gb10.md`](../results/kernels/kda-gate-bench-gb10.md). |
| **The MLA fp32 → bf16 dtype change** | The one lever that survived that bench, and the cold re-measurement did not touch it: +0.24 ms/step, about +0.3 % of C1, no quantization and no new kernel. Small, and gated on `needle` at 1M rather than on speed. Its own numbers have never been re-taken cold either. §2.25. |
| **The prefill half of the quantization gate, re-measured cold** (M = 1,792) | The decode half was re-taken on the target GPU and moved by 2.5×; the prefill half was not, so half of an **and** is carried on a workstation warm bench. Those shapes are compute-bound and so less exposed to the cache artefact, which is an argument, not a measurement. One throwaway container beside an idle engine, about two minutes. §2.25. |
| **Whether the full-scope arm's TP=2 memory reading has a mechanism at all** | It said ~10 GiB heavier; TP=3 says 3.4 GiB lighter. The confound is named (different checkpoints *and* different `max_model_len`) and the `ops.reserve` hypothesis is still untested. Recorded as not reproduced, not explained. §1.9 row 32. |
| **`mtp.safetensors`, the MIT-licensed MTP drafter that ships with that checkpoint** | 3.79 GB, not in the index, never read by vLLM. It is a candidate replacement for DFlash2 whose licence would transfer to a reader, unlike ours ([01](01-model-and-license.md) §4). Not evaluated. |
| ~~**`gpu-memory-utilization` above 0.83**~~ — **climbed on 6 September** | No longer an absence, and the sentence this row used to carry is retracted (§2.4): the ladder was climbed rung by rung against swap *traffic* rather than `MemFree`. 0.85, 0.87 and 0.88 all pass, 0.90 is rejected, 0.87 shipped as production 11 and 0.88 as production 12. **0.89 has never been measured**, and what this stack still needs at that end is a soak rather than another rung. [`../results/memory/ladder-6sep.md`](../results/memory/ladder-6sep.md). |
| **Whether a second-stream all-reduce overlaps a GEMM at all on this part** | The probe is written and has not been run. It gates every overlap variant in §2.17, and it costs one engine-down bench. |
| **A long unattended run** | The longest continuous uptime on record is about an hour between arms. Leaks, KV fragmentation, fabric drift and acceptance drift over 6–12 hours of mixed load are all unmeasured. |
| **`--max-num-seqs` above 8** | Chosen to match the TP=2 arrangement and never A/B'd, and C8 sits exactly on the cap. It does not enter the KV divisor, so the cost would be TTFT, not pool. |
| **Prefix caching** | With a 3,328-token attention block our benchmark prompts never fill one, so the hit rate is 0 % throughout and the benchmark measures nothing about it. |
| **Long-context behaviour under load** | KV usage never exceeded 13 %. The pool is insurance, not something we have stress-tested. |
| **Pipeline parallelism** | The other way to run a fully quantized checkpoint on three nodes. Not evaluated; its interaction with speculative decoding is unknown. |
| **The RoPE convention question in the drafter** | A helper exists upstream whose comment describes the exact symptom we suspected, and our acceptance rate refutes it — so a working, measured configuration was left alone. The patch is written and never applied. |

---

## 4. Things that are true and inconvenient

- **Boot-to-boot variance on this stack is up to 16 % on C8.** Any single-pair comparison anyone
  publishes about this hardware, including ours, should be read with that in mind.
- **Binary hashes cannot certify a build here.** The compiled extension differs between independent
  builds of provably identical source, in the embedded device code; neither whole-file hashes nor the
  ELF build id are stable. Only behaviour certifies a build. See [02](02-image-build.md).
- **One node's build wall-clock does not match its own build's internal step timer**, by 90 seconds,
  and clock skew, toolchain version and compiler caching were each checked and ruled out. Unexplained
  `[measured-here]`. It does not affect any conclusion, which rest on content hashes and on tests.
- **The drafter is the difference between ~20 and ~57 tok/s at a single stream, and it is the most
  restrictively licensed component in the stack.** See [../LICENSES.md](../LICENSES.md).
- **Half of this fabric had never carried a packet, and nothing told us.** Every link was `ACTIVE`,
  every subnet was configured, every benchmark ran, and the second cable of each pair had transmitted
  zero bytes since the driver loaded. The ceiling we spent a day reasoning against was half the real
  one. If you take one operational lesson from this repository, take that one: read the byte counters,
  not the link state ([06](06-nccl-mesh.md) §6).
- **We computed the same ceiling wrong twice, from two different datasheets, in two days.** First
  "a 25 GB/s link", then "a 50 GB/s pair of cables"; the answer was a PCIe slot both times (§1.6,
  §1.7). On the same day the memory roofline moved from a vendor 273 GB/s to a measured 225 GB/s and
  made every efficiency percentage in this repository ~22 % optimistic
  ([10](10-results-and-roofline.md) §4.1). Three of our rulers were brochures. **Measure the ruler,
  in the same process, and quote it beside the number.**
- **The ruler itself is not stable.** Three read-bandwidth measurements on the same idle machine the
  same morning gave 225.2, 239.6 and 240.9 GB/s — 6.5 % apart `[measured-here]`. Any efficiency
  figure in this repository that would change a decision at ±6 % is given as a band for that reason.
- **The best-designed change of the day did exactly what it was designed to do and was worth
  nothing.** The one-sided RDMA_WRITE transport takes RNR retries to zero, which is what it was
  built for, and moves throughput by zero, because the constraint was somewhere else entirely
  ([06](06-nccl-mesh.md) §10). A mechanism working is not a result.
- **A compile check answers "does it build", never "does it run".** A GPU-free ahead-of-time compile
  of 18 kernel configurations reported all 18 building and all 18 inside the shared-memory limit. At
  real launch **6 of the 18 failed with `OutOfResources`** — the ahead-of-time path reported 36,864
  bytes of shared memory where the launch needed 106,496 `[measured-here]`. Resource allocation only
  becomes visible when something is actually launched, and the static test had already been written
  up as a pass.
- **Our own harness disagreed with its own documentation for a day**, claiming five rounds with two
  discarded while running three ([09](09-measurement-protocol.md) §1.1). Nothing was published from
  the wrong reading, but a tool that describes itself incorrectly is the same failure class as a
  ruler that reads high: it will eventually be believed instead of read.
- **An engine's own log line about its own memory is still an unverified ruler.** "Non-torch memory:
  1.50 GiB on rank 0, 9.5 on the workers" was printed by vLLM, in production, every boot. It is not an
  allocation — it is the difference between two `/proc/meminfo` readings taken minutes apart, and it
  runs **backwards**: a node that starts dirty awards itself a bigger KV pool. We priced 8.2 GiB of
  "stranded" memory off it and were about to over-commit the head node by exactly the amount that had
  already produced a swapping boot (§2.3). Nothing about the number was a lie; we simply never asked
  what it measured.
- **Two of the thirty-two rows of §1.9 are us re-opening something we had already closed and
  measured.** `NCCL_BUFFSIZE` was listed as an open lever twelve hours after it had been eliminated,
  because the elimination lived in one report and the candidate list was written from another. The
  fix is not diligence, it is that a closed item has to be closed **in the place where the next
  person will look**, which is why §2 keeps closed items rather than deleting them.
- **The two changes designed as one turned out to be unrelated.** The KV-zeroing gate and the fp8
  draft cache were built together, on the reading that uniform precision would let the zeroing stop.
  The zeroing is bound by Mamba slot sharing, not precision, so one of the two is closed at zero and
  the other is a 5.6 % pool gain that never needed it (§2.13, §2.18).
- **The cheapest change of the day bought nothing and was still the most valuable one.** A host-side
  wait for memory to settle before starting the container adds seconds to a boot and **zero** tokens to
  the pool, and it corrects **no** published number. What it removes is 27 % of a rank's KV allowance
  sitting in the measurement, waiting for a different start order — and the state where an explanation
  and an artefact are indistinguishable. A measurement fix appears in no results table, which is
  exactly why it keeps not getting written.

---

## 5. Where to help

[../CONTRIBUTING.md](../CONTRIBUTING.md) lists the items above that a reader with comparable hardware
could close, in rough order of usefulness.
