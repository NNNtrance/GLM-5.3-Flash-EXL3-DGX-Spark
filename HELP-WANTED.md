# Help wanted — what a second cluster could settle, ranked

This is the list to pick from. Each item says what it is, **what it costs you in wall-clock time**,
what to measure and how to report it. Every one of them is something we could not run, not something
we could not be bothered to run, and the reason is given with each.

`CONTRIBUTING.md` has the general rules and its own twelve-item list, which this page ranks and
extends. `docs/00-start-here.md` says which pages apply at which node count. `STYLE-GUIDE.md`
governs anything written here, including the evidence tiers used below:
`[measured-here]` · `[measured-here, raw lost]` · `[reported]` · `[estimate]` · `[not tested]` ·
`[retracted]`.

**Three ways to send a result**, in increasing order of how much we can do with it: an issue using
the *Measurement contribution* template; a pull request adding your raw output under
`results/community/<your-handle>/<item>/` with a short Markdown summary; or a pull request that also
edits the page the item lives on. We credit contributors in `CREDITS.md`.

**Effort figures below are our own measured tier costs** (`docs/09-measurement-protocol.md` §9): a
fast-load boot is about 5 minutes and a cold one 6-7, a **tier A** model-free bench is seconds to 45
minutes with the engine down, a **tier B** quick arm is **17-21 minutes** (boot, gates cold, prefill,
one warm-up plus three measured C1-C8 rounds, gates warm), and a **tier C** full arm is **30-40
minutes**. A one-off `FASTLOAD_MODE=dump` boot is about 11 minutes and only has to happen once.

---

## 1. Four nodes, TP=4 — the whole track we cannot write

**Effort: days, not hours.** Cabling and a first boot, then the padding work below, then a tier C
arm. **We own three nodes and cannot start it** `[not tested]`.

We can hand you the arithmetic, and it is short, because **TP=4 needs almost none of the shape
surgery TP=3 does.** Read off `config.json` and our own `tracks/tp3/patches/preflight-tp3.py`, not
measured:

| Shape | Value | At TP=4 | At TP=3, for contrast |
|---|---|---|---|
| `num_attention_heads` | 64 | 16 per rank, whole | padded 64 → 66 |
| `num_key_value_heads` | 64 | 16 per rank, whole | 64 → 66 |
| KDA `linear_attn_config.num_heads` | 64 | 16 per rank, whole | 64 → 66 |
| shared expert intermediate | 2,048 | 512 per rank = **4 × 128**, whole | padded 2,048 → 2,304 |
| routed experts `moe_intermediate_size` | 2,048 | 2,048 % (128 × 4) = 0 → **tensor-sliceable** | not sliceable; EP is mandatory |
| DFlash2 drafter GQA | 32 / 8 | 8 / 2 per rank, whole | padded 32/8 → 36/9 |
| **`vocab_size`** | **154,880** | **38,720 per rank = 302.5 × 128 — half a Hadamard block** | `padding_size` 384 → 155,136, 404 × 128 per rank |

Two consequences, and the second is a defect in our own gate.

**Expert parallelism is optional at TP=4**, as at TP=2 and unlike TP=3: the trellis slices cleanly to
512 columns per rank. Under EP, 288 experts give **72 whole experts per rank**. Both arrangements are
legal, neither is measured, and which one wins at four ranks is a real question.

**The vocabulary is the only shape that needs padding, and `preflight-tp3.py` will tell you it does
not.** 154,880 is already 1,210 × 128 and divides 4, which is exactly the condition the preflight's
first branch returns "no patch needed" on — but it never checks the **per-rank** count, and per rank
it is 302.5 blocks. That is the same defect `lcm(64, 3) = 192` produced at TP=3 and that
`docs/03-tp3-padding-and-sidecars.md` §1.1 calls silently wrong. The unit that works is `128 × tp`:
at tp=4 that is **512**, giving a padded vocabulary of **155,136** — the same value TP=3 uses — and
38,784 = 303 × 128 per rank. `lcm(128, tp)` is right at tp=3 (384), right *by luck* at tp=2 (154,880
is already 605 × 128 per rank) and **wrong at tp=4**. A TP=4 port should change the constant in
`patch-vllm-tp3.py` to `128 × tp` and add the per-rank condition to the preflight's first branch.
None of this applies to the routed-experts-only fallback checkpoint, where nothing that gets padded
is EXL3 `[not tested]`.

**Cabling is the part we cannot reason our way past.** Each DGX Spark has **two** QSFP cages
(`docs/00-hardware-and-os.md` §4.1). Three nodes ring with three cables and **every pair has a direct
cable**. Four nodes and two cages each is a **ring of four cables covering four of the six node
pairs** — the two diagonal pairs have no direct link at all, and what NCCL and the mesh plugin do
about that is the first thing to report. `patches/kernel/0005` (the second logical link of a pair)
still applies to the directly-cabled pairs, because each cable carries two links on two PCI
endpoints; `patches/kernel/0006` applies regardless.

**What is already published at four nodes** `[reported]`:
`punkjazz-labs/glm-5.3-flash-exl3-4x-dgx-spark` runs the routed-experts-only checkpoint at TP=4 and
publishes a tuning campaign, and — more usefully — a **soak that hung the engine three times out of
three** in 150 minutes with 96K prompts in the mix. `docs/16-comparison-with-published-recipes.md`
§4.4 has the detail. If you have four nodes, that failure is the single thing we would most like
understood.

**What to measure:** the eleven checks in the README quick start, in order; the boot gates as
`tracks/tp3/patches/README.md` lists them (including the `assert 5` pad-audit line and the
`CUDA_EXL3_DEBUG_NAMES` tally); quality gates cold **and** warm; the KV pool from a settled load
boot; C1-C8 on `scripts/hizset-v2.jsonl`; prefill on **fresh** prompts; TTFT; and EP on against EP
off, which is the arrangement question only you can answer.

---

## 2. A two-node reboot and autostart test

**Effort: about an hour, one reboot.** This is the cheapest open item on the page and it closes a
`[not tested]` that a two-node owner hits on their first power cut.

`systemd/README.md` records **three** three-node trials, one per production configuration: `/health`
200 at **315 s** by the wall clock on configuration 10 (242 s by the harness's own counter, which is
why both are printed), **312 s** on 11 and **311 s** on 12, the unit finishing at +98 to +103 s, the
KV pool coming back within 0.6 % of a hand-started boot each time. Nothing enforces the start order —
the unit starts all ranks concurrently and the tests passed because the workers' rendezvous retries
until rank 0 appears, with `TimeoutStartSec=1200` about four times a normal boot. **That is tolerance,
not a guarantee, and it has been demonstrated three times rather than proved once.** Whether it still
holds with one worker instead of two is unmeasured.

The two-node edits are four, in `docs/15-tp2-track.md` §2.6: `WorkingDirectory` and `ExecStart` point
at your two-node tree, `FABRIC_PEERS` in the preflight becomes one address per node rather than two
(the ConnectX-7 check stays `4/4` — it counts ports, not peers), and the reboot rule becomes "reboot
**both** together, never one".

**What to measure:** the reboot instant; ssh answering again; `ibv_devinfo` 4/4; the unit's `Finished`
timestamp; `/health` 200 **by both clocks** (print the harness counter and the wall clock separately,
because ours disagree by 73 seconds and we could not reconstruct why); the KV pool on that boot
against a settled hand start of the same configuration; and the gates afterwards. Report whether the
preflight ran at all — a unit that finishes in well under a minute did not.

---

## 3. The memory-fraction ladder at two nodes

**Effort: 3-4 tier B arms, about two hours.** One boot per rung.

Every TP=2 arm in this repository ran `gpu-memory-utilization` **0.85**, and arm A recorded **3.5 GB
of swap growth on the head node during weight load** at that rung `[measured-here]`. Three-node
production runs **0.88**: the three-node ladder was climbed rung by rung on 6 September against swap
*traffic* under load, 0.85 and 0.87 and 0.88 all passed and 0.90 was rejected
(`docs/11-open-issues.md` §2.4). The two-node memory budget is a different problem: about **82 GiB of weights per
node** against roughly 55 at three, on a part where the GPU and the host share one 121.6 GiB pool.
**The three-node ladder was derived at three ranks and should be re-derived at two, not copied**
(`docs/11-open-issues.md` §2.4, `docs/15` §3.3).

**What to measure, per rung** (suggested: 0.80, 0.83, 0.85, and one rung below 0.80 if 0.80 swaps):

- KV pool from a **load** boot on a settled host, with all ranks' "Free memory on device" lines
  agreeing within about 1 GiB.
- `MemFree`, `MemAvailable` **and swap, sampled through every benchmark round**, not only at rest.
  The rung is decided on **swap growth under load**, not on free memory at idle: that is the number
  the three-node ladder was decided on, and at 0.83 our `MemFree` is already 0.9-1.2 GiB.
- C1-C8 and TTFT, so the pool gain can be priced.
- Gates cold and warm.

**Report the rung that swaps, not only the rung that works.** A ladder with no failing rung on it
does not say where the edge is.

---

## 4. Other checkpoints

**Effort: a download, plus one tier B arm each (~30 minutes of cluster time per arm).** The download
is the long pole: the production checkpoint is 165 GB.

Four separate questions, in order of how cheap they are to answer:

1. **The newer revision of the fallback.** We pinned `brandonmusic/GLM-5.3-Flash-tr3-4bpw` at
   `b20c49ba`; that repository's `main` moved to `aba59d21` four days later. A clean A/B with the
   correctness probe, the code exam and an MMLU sample `[not tested]`.
2. **`mtp.safetensors`, which ships inside the production checkpoint and is never read.** 3.79 GB,
   not in the safetensors index, so vLLM ignores it. It is an **MIT-licensed** MTP drafter, and that
   matters: our DFlash2 drafter is CC BY-NC-ND and **our permission does not transfer to you**
   (`LICENSES.md`). A drafter whose licence a reader can actually use is worth an arm on its own
   `[not tested]`.
3. **Higher-precision and other mixed-bitrate EXL3 packages.** The 4.05 bpw package is already mixed
   — 4-bit routed experts, 5-6 bit dense and attention, 6-bit head — so this is a refinement rather
   than an alternative, and nobody has run one `[not tested]`.
4. **The class that actually breaks loaders: a checkpoint where a tensor we have to *pad* is itself
   quantized.** That is the entire reason `tracks/tp3/patches`'s A9 and A10 and the plugin's padded-load
   path exist (`docs/13-full-scope-checkpoint.md` §7). Our stack handles it for EXL3. The same class
   exists in other quantization formats — for example RadixArk-style NVFP4 packs, where the shared
   expert is quantized rather than BF16, so the pad needs packed-U8 columns **plus** an FP8 block
   scale, which our NVFP4 sibling's pad code, being BF16-only, does not write `[not tested]`.

**If you bring a full-scope checkpoint from a different publisher, report these three lines before
any speed number:** the `CUDA_EXL3_DEBUG_NAMES` tally (ours is **203 EXL3 / 113 BF16** per rank), the
`HAREM-FULLSCOPE assert 5` pad-audit line (ours: 285 padded EXL3 sites, all whole 128-blocks and
exactly zero), and the `[padload]` capability line. If `assert 5` is absent, the audit did not run
and the boot proves nothing.

---

## 5. The mesh plugin's small-message latency floor

**Effort: one tier A bench, under an hour, engine down. Needs at least two nodes and a cable.**

The decode collective has no bandwidth term in it. On our production plugin build (0004 + 0005 +
0006, `NCCL_MAX_NCHANNELS=8`), an in-house `torch.distributed` all-reduce bench over three nodes on
6 September 2026 reads **74.7 µs at 8 KB** (0.146 GB/s of bus bandwidth), 86.4 µs at 64 KB, 275 µs at
1 MiB, **1,097 µs at 16 MiB — 20.40 GB/s, or 98.1 % of the 20.8 GB/s we measure on our own wire** —
and 3,955 µs at 64 MiB. **From 8 B to 32 KiB the curve is flat at 72-85 µs**: pure latency, no bytes
in it `[measured-here]`. Full tables, arms, isolation and raw:
[`results/mesh/nccl-latency-sweep.md`](results/mesh/nccl-latency-sweep.md); the tool is
[`bench/nccl-latency-bench.py`](bench/nccl-latency-bench.py) and
[`bench/README-nccl-latency.md`](bench/README-nccl-latency.md) says why it is not `nccl-tests`
(no MPI on these nodes, and `nccl-tests` requires it for multi-node).

**Start by settling our own instrument, because three of our harnesses disagree.** On the same
operation, at the same two sizes:

| harness | 8 KB | 64 KB |
|---|---:|---:|
| `bench/ar_bench.py`, pre-multilink configuration (`docs/06` §12.1) | **38.6 / 38.7 µs** | **61.3 / 61.5 µs** |
| `bench/nccl-latency-bench.py`, production configuration, 6 September | **74.68 µs** | **86.40 µs** |
| `bench/mesh_sweep.py` at `NCCL_MAX_NCHANNELS=8` (`results/mesh/all-reduce-sweep.md` §4) | — | **143 µs**, called latency-bound with ±100 µs of noise at the time |

A factor of two at 8 KB and about 40 % at 64 KB, between benches of our own. They differ in more than
one variable at once — iteration count, warmup, the timing loop, and for the first row a plugin build
and cabling state from before the dual-cable and `NCCL_PTR_CUDA` work — so **we cannot assign a cause
from the data that exists, and we have corrected none of them against the others.** Whoever runs this
should run **at least two** of the harnesses in one session, on one configuration, before drawing any
conclusion: this repository's standing rule is that the ruler gets measured too, and three of ours
turned out to be brochures (`docs/11` §4). **This is the cheapest useful item on this page and it
needs no new code** — all three tools are in `bench/`.

**Why it is worth the hour.** The C1 decode all-reduce is 64 KB and there are about 90 of them per
step; the profile says the NCCL class is **100 % exposed**, with measured comm/compute overlap of
**0.00 ms** (`docs/10-results-and-roofline.md` §5). A latency floor that is mostly software is the
one remaining fabric lever that is not capped by the cards' PCIe Gen5 x4 slots (`docs/06` §9). For
scale, the `cuda-exl3` author measures **13.6 µs** for an 8 KB NCCL all-reduce on his own hardware
against a 6.74 µs one-way flag-visibility floor `[reported]` — different fabric and different
topology, so the absolute number is not ours to borrow, but a gap that large on a message with no
bytes in it is the shape of a software floor rather than a wire.

**What a contributor with a single host can check** — and it is more than nothing:

- The **local half of the floor**: kernel-launch and flag-visibility cost on one GB10, which sets the
  floor everything else sits on top of.
- The plugin's own unit tests: `make test-unit`, expect `test_routing` 13/13.
- Any `bench/` micro-benchmark that does not cross a node boundary — `bench/bw.py`,
  `bench/gemmpeak.py`, `bench/topk_bench.py`, `bench/moe_stage_bench.py`.

**What they cannot check, and please do not report as if they had:** anything about the collective
itself. With one node there is no peer, `mesh_isend`/`mesh_irecv` never post, RNR flow control — which
is the whole subject of `docs/06` — cannot occur, and NCCL will not build a ring. This item needs two
nodes and a cable.

**One finding on this item does not transfer to a single host, and it should be said before someone
wastes a day on it.** `NCCL_MAX_NCHANNELS=8` is worth up to **11×** in the 128 KiB - 16 MiB band here
and **nothing at all** at decode sizes; the `cuda-exl3` author's channel sweep on his single-host
no-P2P box is **flat** `[reported]`. The channel effect is a property of a multi-NIC mesh fabric, not
of NCCL, and a one-host reader should not expect it.

**Report:** microseconds per all-reduce at 8 B / 8 KB / 64 KB / 512 KB / 16 MiB, the per-collective
`rnr_nak_retry_err` and `out_of_buffer` deltas beside each, the channel count, the plugin commit and
which of `patches/kernel/0004-0007` are applied, and which harness took each row.

---

## 6. ReplaySSM: the KDA state slots, 9 down to 2

**Effort: about 8 hours to port, then one A/B boot per arm. The first thing to run is not the port,
it is the speed measurement.** We have done the code reading and not the boot. The full working —
where the slots come from, what the ring costs, the four-arm validated pool model and the port
budget — is [docs/17](docs/17-memory-ledger.md) §5.3, inside the memory ledger it belongs to.

**The finding.** In the align path, `MambaSpec.max_memory_usage_bytes = page × (2 +
num_speculative_blocks)`, so with DFlash2 at k=7 **every KDA layer holds 9 state slots per request,
7 of them purely for speculation**. The engine prints it —
`MambaSpec: 9 layer(s), page 1,703,936 B, max/req 15,335,424 B -> 9 block(s)` — and the slot count
multiplies the **block count**, not the page, which is why a compact ring does not shrink the
3,328-token attention block ([docs/17](docs/17-memory-ledger.md) §4). That is 36 blocks per request — **9.9 % of the block counter at
TP=3** (4.25 GiB per rank) and **12.9 % at TP=2** `[measured-here]`. It is the largest single
give-back left in the pool, and it is the same class of defect as the draft KV page in
`docs/07-kv-and-draft-page.md` §3: a **counter**, not memory.

**What would fix it.** A third-party ReplaySSM speculative-decoding patch for vLLM
(`vllm-replayssm-spec.patch`, by `tpurtell`) replaces the slot array with a compact ring and takes 9
slots to **2** — two, not one: the align path keeps a pair. We do not carry it and it is not in this
repository.

**What we measured without starting an engine** — a read-only code review plus a throwaway CPU
container:

- The patch is 21 files and 127 hunks, **pure Python and Triton, zero CUDA or `csrc`**, and applies
  **127/127 clean** to our image. A GLM port script holds **9 of 11 anchors**; the two misses are one
  class-base line, a two-line fix. It touches nothing in `cuda-exl3`, EXL3, the GEMM path or MLA
  `[measured-here]`.
- A pool model that reproduces **four independent measured arms exactly** (TP=3 production 10 at
  5,619,834; TP=2 control 601,562; TP=2 with the page fix 1,303,571; TP=3 before the page fix
  2,428,769) projects **5,619,834 → 6,071,684, +8.0 % at TP=3** and **1,303,571 → 1,429,245, +9.6 %
  at TP=2** `[estimate]`. That is **lower** than the raw slot arithmetic suggests, because the ring
  lives inside the mamba page and raises the attention block from 3,328 to 4,096.

**What it would cost, which is why it is not in production here.** The replay loop runs
`tl.static_range(0, CACHE_BUF_LEN)` in full on every step, taking the sequential work from 8 to
**24 — three times**. KDA is about **8 % of a C8 step** on this stack, so the worst case is roughly
**+17 % on a C8 step**: 197 tok/s down to about 168 `[estimate]`. The author's own note says the
baseline is faster at a single stream and gives no figure.

**And correctness is unproven, which is the part that would have to be settled first.** The ring
holds `d` and `k` in **fp16** and the error accumulates across the replay, so the question is not
whether the mechanism works but how far the state drifts. Reading the upstream harness we could not
find a test that compares the ring against a baseline — the materializer test we did find reads the
same fp16 ring twice, so it would pass whether the ring is right or wrong. If such a test exists and
we missed it, we would rather be corrected than write our own.

**The A/B we would run, and would take from anyone.** One boot per arm, same image, the only
difference the patch. Then, in order: a logit-divergence probe that establishes the **within-arm**
noise floor before it compares arms, at 64 / 512 / 4,096 generated tokens; a needle-in-a-haystack run
at 1M context; the gates cold and warm; and **C1 and C8**. C8 is where the cost is, and an arm
measured only at C1 answers the wrong question.

**Our verdict, so you can disagree with it in the open: not now.** The pool is not the binding
constraint at three nodes — a bigger pool prevents preemption, it does not add tokens per second. It
*was* binding at two. This item becomes worth the risk if the pool binds, if two nodes become a real
serving configuration, or if the author publishes a numeric C1/C16 A/B.

---

## 7. The KDA GEMM, engine against standalone

**Effort: one tier A bench plus one live profile, about an hour. One node is enough — this needs no
fabric.**

A model-free micro-benchmark of the KDA `in_proj` GEMM on an idle GB10 reaches **214 GB/s**. The same
shape inside the running engine reads **150 GB/s** — a **37 % gap**, of which tensor alignment
accounts for only 3-5 % `[measured-here]`. That measurement was taken on our NVFP4 sibling stack
(`NNNtrance/GLM-5.3-Flash-NVFP4-TP3-3x-DGX-Spark`), where the dense BF16 path is 49.4 % of a
single-stream step and KDA `in_proj` is a single 15.9 ms GEMM inside it.

**It belongs on this page because it is a cuBLAS operand-and-heuristic question, not a quantization
one, and the same GEMM runs here.** KDA is **8.1 % of a prefill chunk and 8.0 % of a C8 step** on
this stack and **has never been measured against a ruler** (`docs/11` §3). It inherits the empty slot
MLA prefill vacated.

**What to measure.** The same shape twice, with `bench/bw.py` and `bench/gemmpeak.py` run **in the
same process** as rulers, because ours were brochures until we measured them (225 GB/s achievable,
not the datasheet's 273). Then a live profile with `record_shapes` on. **Report which cuBLAS kernel
each side picks** — that is the whole question. If the standalone bench picks a different kernel from
the engine for identical shapes, the gap is a heuristic and it is fixable at the call site.

---

## 8. Two loose ends from the MLA-prefill falsification: a correctness check and a cost anomaly

**The falsification itself is done — this section used to ask for it, and now reports it and asks for
two smaller follow-ups instead.** The `cuda-exl3` author corrected his own MLA-prefill ceiling in
commit `5fd7299` after our indexer datum ([`results/kernels/sm12-stack-patches-ab.md`](results/kernels/sm12-stack-patches-ab.md)
§8: median 2,049 selected keys per row, median adjacent-row overlap 0.9258) showed production was two
orders of magnitude from the low-turnover arm it had been compared against, and predicted that on a
48-SM part the production arm would land within a few percent of a fully cache-resident arm rather
than between it and a gather-bound one. We ran that on our own GB10:
[`results/kernels/mla-prefill-falsification-gb10.md`](results/kernels/mla-prefill-falsification-gb10.md)
`[measured-here]`. **The prediction held at both head counts** — at 262K context, +1.3 % at our own
22-head TP=3 shape (tighter than the +1.6 % he measured on his own 188-SM card) and +5.4 % at his
16-head shape, against an independent arm needing 82.0–146.8 % more time. **The item closes at zero on
both parts**; full account in [docs/11](docs/11-open-issues.md) §2.27. Two things fell out of closing
it, and neither has an answer yet.

### 8a. Numerically verify the kernel at the real 22-head TP=3 shape

**Effort: one tier A bench, a few minutes, engine down. One node, no fabric — and the tooling is
already written.**

The falsification's own correctness gate checked `mla_decode` against a torch reference only at a
small 2-head smoke shape; the 22-head shape every timed number above uses was **timed but never
numerically verified**. This stack has an unrelated kernel (`b12x`'s decode path) that silently
miscomputed at exactly the 22-head TP=3 shape because nobody had tested that shape, so this is not a
formality.

Nothing needs writing: [`bench/mla-prefill/mla_prefill_falsify.py`](bench/mla-prefill/mla_prefill_falsify.py)
`--check` runs a correctness matrix at the production shape — both head counts, both contexts, the
kpool-aligned selection grid, and a ragged-`seqlens` variant that exercises the masking path the
timing run never touches — and its `--self-test` (CPU only, no lock, no GPU) has already passed
24/24. What is missing is the GPU half: run `--check` inside the production image in an engine-free
window.

**Report:** pass/fail per cell of the matrix, the worst `max_rel` and `cos_min` across it, and — if
anything fails — which arm, context and variant, so a failure can be narrowed the way the b12x one
eventually was.

### 8b. Why 22 heads cost 13-16 % more per head than 16, in the compute-only arm

**Effort: unknown — a "what is going on" item, not a "run this command" one. Start with a tier A bench
(an hour); escalate to profiling only if that does not explain it.**

The falsification's `drifting` arm (cache-resident, so purely the kernel's compute/issue cost) scales
**super-linearly** with head count: 16 → 22 heads is 1.375× the work, but the measured time is
**1.557–1.589×**, reproducibly at all three contexts. The `independent` (traffic-bound) arm does the
opposite — 1.17–1.28×, *sub*-linear — so the penalty sits entirely on the compute path, not in memory
traffic, and it is invisible at the author's own 16-head shape. Since 22 heads per rank is what this
stack's TP=3 shape gives it, a TP=2 or TP=4 topology would not see this at all.

**What is not known:** whether this is a smooth per-head cost, a step at some occupancy or
register-pressure boundary between 16 and 22, or something specific to `head_dim=576` not dividing
evenly into whatever block/warp geometry the kernel uses at higher head counts. A cheap first step:
sweep head count finely (16, 18, 20, 22, and a couple of points past it) in the `drifting` arm only —
the effect is context-independent in the falsification's own data, so no context sweep is needed —
and report whether the cost curve is smooth or has a knee, and where. Past that, it needs `ncu`/`nsys`
on the kernel at two head counts and a report of which phase of `mla_decode` the extra time lands in.

**Report:** the per-head-count timing table from the sweep, "smooth" or "knee at N heads", and, if you
profile it, which sub-kernel or code path accounts for the gap.

---

## 9. The indexer workspace that grows with `max_model_len` — filed upstream, and the part still open

**Effort: none, if you only want the outcome. It is filed and it is measured** `[measured-here]`.
This item asked, until 6 September, for someone to carry our measurement into vLLM. It no longer
needs carrying: the same bug was reported upstream three days earlier from entirely different
hardware, and we have added our numbers to that thread rather than opening a second one.

**Where it lives now.**

| | |
|---|---|
| Issue | [vllm-project/vllm#55221](https://github.com/vllm-project/vllm/issues/55221) — part 2 of it is this finding |
| Our confirmation | [comment on #55221](https://github.com/vllm-project/vllm/issues/55221#issuecomment-5559336290) |
| Fix, model side | [#55222](https://github.com/vllm-project/vllm/pull/55222) — divides the glm5next call site by `index_kpool` |
| Fix, builder side | [#51252](https://github.com/vllm-project/vllm/pull/51252) — the same division in `DeepseekV32IndexerMetadataBuilder`, generically |

**The finding, unchanged.** `get_max_prefill_buffer_size()`
(`vllm/v1/attention/backends/mla/indexer.py`) returns `max_model_len * 40` **entries** for the sparse
indexer's K-gather workspace. Upstream's own comment gives the provenance: 40 was chosen against
DeepSeek-V3.2's `max_model_len` of 163,840, where 40 × 163,840 × 132 B = 825 MB. The constant is
linear in the context length and nothing else, so a long-context deployment pays it in full:
`zai-org/GLM-5.3-Flash` at `max_model_len` 1,000,000 reserves **40,000,000 entries = 4.92 GiB**,
during the profile run, locked for the life of the engine, and charged to the residual vLLM subtracts
before it sizes the KV cache. **It is not visible in any line an operator reads** — not in
`Model loading took`, not in the KV pool, only inside `consumed memory (weights + non-torch)`.

**And there is a second half, which is the part that makes it a bug rather than a tuning constant.**
The buffer holds one indexer chunk's **compressed** context: the chunker works in
`seq_lens // compress_ratio`, and `compress_ratio == index_kpool` (4 for this model). The
DeepSeek-V4 path divides the same function's result by `compress_ratio` at its call site
(`vllm/models/deepseek_v4/attention.py`); the `glm5next` path does not. So the buffer is sized at
four times the resolution of the unit that fills it, on top of a constant already 6.1× larger than
the model it was tuned for. The exact ceiling is
`max_num_seqs × ceil((max_model_len + num_spec + 1) / compress_ratio)` — **251.8 MB** at our
settings, against 4.92 GiB reserved.

**What we measured, so the thread does not have to take it on trust** — three-node TP=3, one A/B in
one session, everything else held: bounding the buffer to 512 MB (2.03× that ceiling) moves the KV
pool **6,289,256 → 6,933,884 tokens, +10.25 %**, with the gates full cold and warm, a 969,468-token
single request and eight concurrent long-context lanes all correct, and no speed level outside its
band. Tables: [`results/memory/indexer-workspace-ab.md`](results/memory/indexer-workspace-ab.md).
Our env-gated patch, which is a local workaround rather than a proposed fix, is
[`tracks/tp3/patches/indexer-workspace/`](tracks/tp3/patches/indexer-workspace/).

**Three things we put in the thread that were not already in it**, and any of them can be argued
with:

1. **#55222 and #51252 want to land together, not merely to compose.** #55222 divides the buffer's
   *shape* at the glm5next call site; the builder's admission budget still comes from the undivided
   value and is what the chunker compares compressed sequence lengths against. With one of the two,
   the chunker may admit 4× what the buffer it writes into can hold. **We first wrote that the net is
   upstream's own locked-workspace `AssertionError`, i.e. a loud failure. That was wrong**, and the
   issue author [@drakosha](https://github.com/drakosha)
   [corrected it](https://github.com/vllm-project/vllm/issues/55221#issuecomment-5561194190): both
   indexers ask the `WorkspaceManager` for a **static** size and slice per chunk, so the request never
   grows and that assertion cannot fire. The real net is **silent** —
   `k_quant_full[: chunk.total_seq_lens]` clamps without raising. Still unreachable at a 1M context
   with 32 sequences, but the shape permits the drift and the failure it permits is quiet. One detail
   of the correction did not hold on our build, in the safer direction: the gather kernel is
   destination-bounded, so there is no out-of-bounds *write* either
   ([docs/11](docs/11-open-issues.md) §1.13). **#51252 is the right generic fix and it is stalled**;
   we will not duplicate it into #55222.
2. **After the division the size is still linear in `max_model_len`** and blind to how many requests
   can exist. Of the 4.42 GiB we freed, the division accounts for 3.69 GiB and the
   `max_num_seqs × per-request` bound for the remaining 0.73 GiB. The second is the smaller half; it
   is also the half that removes the `max_model_len` term. We offered a PR for it.
3. **The 40 is documented, and its justification is borrowed.** We had written here that the comment
   "does not say what it is holding at bay" — that was **wrong**, or at least a version behind. On
   current `main` the comment derives 40 from the `flashmla_sparse` workspace, and *that* workspace
   justifies itself with "This fits nicely below the typical MoE workspace size of >2GB so this is
   'free'". Both statements are true at 163,840 tokens and neither survives 1M — and our control arm
   logged exactly **one** workspace resize event, grown by the indexer, which says there is no larger
   MoE workspace amortising it here. "Free" is a scale-dependent claim.

**What is still genuinely open, and where a reader can help.** We run one model on one part. Nobody
in the thread has checked whether the `max_num_seqs` bound is safe on the DeepSeek-V3.2 and V4 paths,
which is where the constant came from and where a chunker behaviour we have not seen may bind; and
part 1 of #55221 — an fp8 KV cache rejected by the SM90 sparse MLA `plan()` — is untested by us
because there is no SM90 hardware here. If you have either, **the thread is the place**, not a new
issue and not this page.

---

## 10. The open items already written up in docs/11

`docs/11-open-issues.md` §2 is "open, with a known next step", §3 is "never run", and
`CONTRIBUTING.md` lists twelve with their reasons. Rather than repeat them, the four that would move
the most:

- **Overlapping the all-reduce with compute** (`docs/11` §2.17). The collective is **16.5 % of a
  prefill chunk** at **99.3 % GPU occupancy**, so it is serialised rather than hidden — the biggest
  number on the board with nobody on it. Dual-batch overlap is **closed** (splitting the batch streams
  the MoE expert weights twice, and it would corrupt KDA state silently); attention-scoped
  micro-batching survives at an estimated −3 to −6 % of prefill. **Before any of it, one model-free
  probe: does an all-reduce on a second stream overlap a GEMM on this part at all?** The probe is
  written, has never been run, costs one engine-down bench, and if it fails every overlap variant
  dies with it `[not tested]`.
- **A soak.** Our longest continuous uptime on record is about **an hour** between benchmark arms.
  Leaks, KV fragmentation, fabric drift and acceptance drift over 6-12 hours of mixed load are all
  unmeasured — and the published four-node recipe hung three times out of three in a 150-minute soak
  with 96K prompts in the mix (§1). This needs no special hardware, only patience.
- **The large-M hyper-connection fusion kernel** (`docs/11` §2.16, `CONTRIBUTING.md` item 1).
  −2.5 to −2.7 % of prefill is available; our written kernel reaches about 40 % of it and a full
  configuration sweep could not close the gap. **The cheap half is unclaimed: two keyword arguments
  at the call site and one constant in the third kernel are −0.4 % between them, measured.**
- **The fast-load read path.** `HAREM_FASTLOAD_READ=mmap` exists in the code and **has never been
  run**. The sidecar reads at 0.88-1.04 GB/s where the same NVMe gives another loader 3.1 GB/s;
  weight load could go from 67 s to about 20 `[estimate]`.

Also short and unclaimed: `--max-num-batched-tokens 3072` (2048 and 4096 are both measured, the value
between them is not); `NCCL_MAX_NCHANNELS=12` over two cables (one boot, one token in `EXTRA_ENV`);
`block_m` under expert parallelism; MMLU at TP=3 on the fallback checkpoint; and **anything at max
reasoning effort** — everything published here is at `low`, and nothing on any page of this
repository should be quoted as a max-effort number.

---

## 11. CUDA graphs at three ranks — off because of a wrong declaration. **The A/B is done; what is left is a second boot**

> **Run on 6 September 2026 — closed.** We ran the three arms ourselves the evening this section was
> written. **B − C = +1.75 % at C1**, under the +2 % threshold the design set, with every level
> inside its band and mixed signs; the graph pool at three ranks is **0.25–0.27 GiB per rank**
> (a quarter of what we had estimated from the two-node track) and the total price is **−3.85 % of
> the KV pool**. Gates 10/10 and 12/12 cold and warm on all three arms, tool-call 8/8, acceptance
> 59.4–65.6 % throughout, no hang. **We did not write the patch and we are not asking anyone else
> to.** Full numbers: [docs/11](docs/11-open-issues.md) §2.29 and
> [docs/17](docs/17-memory-ledger.md) §2.3; raw rows in
> [`results/speed/concurrency-sweeps.csv`](results/speed/concurrency-sweeps.csv).
>
> **What is still worth an hour, if you have three nodes:** a **second boot of arms B and C**. Ours
> were one boot each, and boot-to-boot spread on this stack is 15.9 % at C8
> ([docs/09](docs/09-measurement-protocol.md) §2) — enough to swallow the whole result twice over.
> A repeat that lands on +1.75 % again would turn "unmeasurable" into "measured and small"; one that
> lands somewhere else would tell us the bench, not the lever, produced the number. Either is worth
> more than the patch.
>
> The section below is kept as written on the morning of 6 September, because the finding it
> describes — the wrong division — is unchanged and still unfixed upstream.

**Effort: nothing, if you only want the finding — it is filed. One engine hour on a three-node
cluster, if you want to repeat the price measurement on a second boot.**

At TP=3 this stack serves with `cudagraph_mode=NONE`, and the reason is not the one we published
until 6 September. FlashInfer's `get_cudagraph_support()` divides the *target* model's per-rank query
heads (22) by the *draft* group's KV heads (3) and, finding a remainder, declares the drafter unable
to capture the 8-token verify batch — while the builder that actually runs uses the draft's own 12
heads, passes the same test, and serves on the XQA kernel the gate says cannot be graphed. At two
ranks the division is 32 % 4 = 0 and the same image captures graphs. [docs/11](docs/11-open-issues.md)
§2.29 has the arithmetic; the report is ours:

| | |
|---|---|
| Issue | [vllm-project/vllm#55581](https://github.com/vllm-project/vllm/issues/55581) — the gate uses the wrong model's head count |
| Fix | one parameter: derive the head count from the group's own layers, as the builder does. Offered in the issue; no PR at the time of writing |
| What we did with it | **nothing** — the ceiling was +1.2–2.3 % single-stream (inside our ±4 % band), and on 6 September the measurement landed at **+1.75 %** |

**What settled the price** `[measured-here]`: three env-only arms on the production-12 tree —
**A** production 12, **B** = A without `HAREM_DRAFT_KV_DTYPE=fp8` (drafter → FlashAttention → graphs
on), **C** = B + `ENFORCE_EAGER=1`. B − C is the graph lever at equal KV geometry; A − C is the fp8
draft cache. `scripts/ab-quick2-full.sh` per arm, cold + warm gates and the tool-call gate mandatory
(a bad capture shows up as acceptance or gate failures, not as an error), about 15 minutes an arm.
The rule was that B − C under +2 % at C1 closes the item without anyone writing the patch; it came in
at +1.75 %, and the three-round median could not resolve it against a 3.5–10.7 % within-arm spread —
**which is the finding, and it is written down as one**. A fix on our side would still be a new patch
file, which re-dumps the fast-load sidecar ([docs/08](docs/08-fast-boot.md)); it is not being written.

## What we would rather you did not send

Repeated from `CONTRIBUTING.md` because it is the shortest way to save your afternoon:

- A single pair of sweep rounds as evidence for anything. Boot-to-boot spread here is up to **16 % on
  C8**. We published a kernel conclusion drawn from one pair and had to withdraw it.
- A three-round median from an image **without** a warm MLA tuner cache. Say which you ran.
- A prefill number measured on a repeated prompt. It reads the prefix cache and overstates by up to
  55 %.
- A speed number without the quality gates from the same boot, cold **and** after the benchmark.
- Host names, LAN addresses, user names, home paths or tokens, anywhere in anything you upload. Grep
  before you push; we do, before every commit.

**A pull request that withdraws one of our numbers is worth more here than one that adds a number.**
We keep the mistake and add the retraction rather than deleting the mistake — thirty-eight of them
are in `docs/11` §1, whose opening paragraph says what that number counts — and we do not ask for a
replacement figure as the price of a correction.
