# 10 — Results and roofline

**Applies to: TP=3.** The tables are the three-node ones. The two rulers, and the method that
measured them rather than quoting them, apply at any node count.

The full tables, the progression that produced them, and how close any of it is to what the hardware
can physically do.

**Settings for everything on this page unless a row says otherwise:** three DGX Spark (GB10) nodes,
TP=3 + expert parallel, `brandonmusic/GLM-5.3-Flash-tr3-4bpw` at revision `b20c49ba` (EXL3 4bpw),
image `exl3-zeus:62f53e6`, KV `fp8`, DFlash2 draft k=7, `--block-size 256`,
`HAREM_SW_BLOCK_SIZE=256`, `--max-num-batched-tokens 2048`, `--max-num-seqs 8`,
`NCCL_MAX_NCHANNELS=8`, CUDA graphs **off** (§5.8), `gpu-memory-utilization 0.80`, per-rank fast-load sidecar,
`CUDA_EXL3_TUNE_CACHE` warm, mesh plugin built from `19924dcc` + patches 0004/0005/0006 with
`NCCL_MESH_LINKS_PER_PEER=0 NCCL_MESH_PTR_CUDA=1 NCCL_MESH_FLUSH=1`, temperature 0, **reasoning
effort `low`**, 5 September 2026. Production configuration 7 adds `HAREM_DRAFT_KV_DTYPE=fp8` and the
launcher's memory settle gate; production 8 moves the image to `62f53e6` and changes nothing else.
CUDA graphs are **off** on both — not by `--enforce-eager`, but because FlashInfer's support gate *declares* the
drafter's group unable to capture an 8-token verify batch (§5.8; the declaration is wrong — it divides the target's
head count by the draft's — and is filed upstream as [vllm#55581](https://github.com/vllm-project/vllm/issues/55581)) — and it costs less than the boot log
suggests. Rows that predate production 7 are labelled in §2. Speed on this configuration is the **median of three sweep rounds**
— the persisted tuner cache is what makes three enough ([09](09-measurement-protocol.md) §1,
[12](12-tuner-cache.md)); the older arms in §2 are five-round medians with two discarded. Raw tables
in [`../results/`](../results/README.md).

**Rulers before roofline.** Every efficiency percentage on this page is against a bandwidth and a
GEMM peak **measured on this device in our own image**, not against a vendor figure — §4.1. The tools
are [`bench/bw.py`](../bench/bw.py) and [`bench/gemmpeak.py`](../bench/gemmpeak.py), they take
seconds, and running them in the same process as the thing being measured is not optional here: the
read-bandwidth ruler drifted 6.5 % between three runs on the same idle machine the same morning.

---

## 1. The production configuration

**Production configuration 9**, since 5 September 18:40: production 8 with the **checkpoint** changed
— `turboderp/GLM-5.3-Flash-exl3` at 4.05 bpw, full scope, on image `exl3-zeus:754421f` and the
`tracks/tp3/patches/` tree. It is the first entry on this page in which the model itself is different,
and it is the largest single move the stack has made ([13](13-full-scope-checkpoint.md)).

**Production configuration 10** was production 9 with one line changed: `gpu-memory-utilization`
0.80 → **0.83**. It bought **+8.7 % of KV pool** (5,168,044 → **5,619,834**) and moved no speed number
outside the spread in §1.1 — C1 70.5, C4 144.6, C8 194.0, gates full cold and warm, swap flat through
the rounds `[measured-here]`.

**Production configuration 11 is what actually ships**, since 6 September 13:25. It is production 10
with exactly two changes, taken in one boot: `gpu-memory-utilization` 0.83 → **0.87**, the top of a
ladder whose next rung is rejected on swap traffic
([`../results/memory/ladder-6sep.md`](../results/memory/ladder-6sep.md)), and the **sm_12x correctness
set** — `patch-pdl-gate.py` and `patch-kpool-init.py` behind `HAREM_SM12_ITEMS=pdl,kpool`
([11](11-open-issues.md) §2.27). Measured against a **same-session** production 10 reference, pooled
over six rounds and two boots: KV pool **6,382,920 (+12.1 %)**, C1 **69.6 (+0.6 %)**, C8 **201.1
(+1.8 %)**, prefill equal, gates 10/10 and 12/12 cold and warm on both boots, tool-call 8/8,
needle-lite 6/6, and swap traffic exactly zero on all three nodes on the clean boot `[measured-here]`.
The one number that moved is C4, and it moved back: 135.0 on the load boot, **145.9** on the clean
boot, against the reference's 144.2 — an 11.1 % six-round spread at the level §1.1 already names as
the noisiest.

Both configurations were reboot-tested as a whole cluster with the autostart unit enabled
([systemd](../systemd/README.md), [`../results/boot/boot-ledger.md`](../results/boot/boot-ledger.md)):
315 s to `/health` on production 10, **312 s** on production 11. Because 10 and 11 change nothing but
a memory fraction and two guard patches, **the whole of this page's analysis is production 9's and
applies unchanged**; the ladder, its cost and why 0.88 was declined are in [11](11-open-issues.md)
§2.4 and [07](07-kv-and-draft-page.md) §6.

**One caption note.** The settings block at the top of this page describes production 8 and still
governs every section from §4 onwards, which was measured on it. For §1 and §2.3, substitute the
full-scope checkpoint, image `754421f` and `TP3_DIR=.../tp3full`; everything else — TP=3 + EP, fp8 KV
and fp8 draft cache, DFlash2 k=7, block size 256, MNBT 2048, 8 sequences, 8 NCCL channels, 0.80, the
settle gate, temperature 0, effort low, medians of three rounds — is unchanged, which is what makes
the two columns comparable.

| | **production 9** | production 8 | production 7 |
|---|---|---|---|
| C1 aggregate / per stream | **69.9 / 75.9** tok/s `[measured-here]` | 56.9 / 62.4 | 57.0 / 64.0 |
| C2 / C4 / C6 / **C8** aggregate | 99.2 / 140.7 / 172.4 / **197.2** tok/s `[measured-here]` | 83.3 / 120.2 / 144.0 / 175.4 | 80.9 / 120.0 / 143.4 / 175.1 |
| TTFT, C1 / C8 | **0.280 / 0.826** s `[measured-here]` | 0.344 / 0.906 | 0.34 / 0.91 |
| Draft acceptance (per-concurrency medians) | **61.9–62.6 %** `[measured-here]` | 61.7–64.4 % | 60.8–64.3 % |
| Accepted tokens per step | **5.34** at C1 `[measured-here]` | 5.50 | 5.3–5.5 |
| Prefill, fresh unseen ~8.5K prompts (median of 3) | **1,738** tok/s `[measured-here]` | 1,776 | 1,769 |
| Prefill, 7K warm repeat | **1,575** tok/s `[measured-here]` | 1,537 | 1,529 |
| KV pool | **5,165,289** tokens (5.17 concurrent 1M-token requests) `[measured-here]` | 4,696,969 | 4,699,724 |
| Consumed memory per node (weights + non-torch) | **58.3–59.1 GiB** `[measured-here]` | 62.1–62.4 | — |
| Boot, container start → API ready | **251 s** (weights 57.9 s) `[measured-here]` | 264 s (weights 73.2) | ~274 s |
| Quality gates, cold and warm | 10/10 · 12/12 `[measured-here]` | 10/10 · 12/12 | 10/10 · 12/12 |
| MMLU sample (1,995 q) | **86.47 ±0.74** `[measured-here]` | 86.4 ±0.7 | 86.4 ±0.7 |

**The step arithmetic is the entry, not the tok/s.** 5.50 tokens per step at 62.39 per stream is an
88.2 ms step; 5.34 at 75.91 is **70.3 ms**. **17.8 ms saved, while acceptance and accepted length
moved the wrong way by about 3 %** — so the whole gain is the dense stage going from BF16 to 4–6 bit
and none of it is drafter behaviour. The same lever measured 20.7 ms at TP=2 on two nodes, which
makes this the first result on this page confirmed on two independent topologies
([13](13-full-scope-checkpoint.md) §7.3).

**What it cost, and the line is not empty:** a second patch tree to keep in step with the first, a
second 53 GB fast-load sidecar per node, and a **prefill** regression on the dense path (+17.3 ms,
+10.4 % at M=1,792, hidden inside an equality-band wall — §5). Quality was looked for and not found:
86.47 ±0.74 against 86.4 ±0.7 is 0.07 points, a tenth of either bar, with both gates full cold and
warm on the same engine instance in one session ([13](13-full-scope-checkpoint.md) §7.4).

**The draft-acceptance entry that used to head this list is withdrawn** `[retracted]`. We published
−2.4 points of acceptance and −3.0 % of accepted tokens per step as the price. Pooled by draft token
across five concurrency levels and three boots it is **+0.18 points**: the C1 median was an artefact of
`bench-sweep.py` cycling `prompts[i % 12]`, so C1 and C2 saw eight of twelve prompts and C4–C8 all
twelve. And `accept_len = 1 + k × acceptance` holds on all 90 rows, so the two entries were the same
number written twice. Net effect on throughput **+0.24 %**
([11](11-open-issues.md) §2.26, [13](13-full-scope-checkpoint.md) §7.4). The row in the table above is
kept because it is what the sweep measured; it is not a cost.

**Production 8's entry was that nothing moved, and it still reads that way.** The `had_in` commit was
priced at ~0.2–0.3 % of prefill wall before it was built ([11](11-open-issues.md) §2.19) — below the
noise floor of a serving benchmark — and it was adopted to keep the image on upstream's head rather
than to buy tokens. Four signs, every one inside its own band. **Production 7** bought the pool at
+5.6 % over production 6 and a better TTFT at both ends, from the fp8 draft cache
([07](07-kv-and-draft-page.md) §7), with its speed column unchanged in both directions.

Three rows need their footnote said out loud rather than hidden in a tier label.

**The KV pool** is read from an ordinary **load** boot with a settled baseline — the launcher waits
for the host's memory to settle before starting the container, which took the per-rank startup spread
from 9 GiB to 1.4 GiB, 27 % of a rank's KV allowance out of the measurement
([07](07-kv-and-draft-page.md) §1.1, [08](08-fast-boot.md) §5.1). A dump boot's pool number is not
usable ([09](09-measurement-protocol.md) §11).

**Boot** was itemised on the fast-boot arm ([08](08-fast-boot.md)): 617.9 s → **273.6 s**, of which
weight loading is 67.2 s. Nothing in the four configurations after it touches the loader, so 274 s is
carried forward rather than re-itemised, with two untimed additions on top: the settle gate's wait
(seconds, capped at 180 s) and the FlashInfer warm-up `[measured-here, raw lost]`. A full restart
driven from the workstation — stop all three, drop caches, staggered start, wait for `/health` —
measured 307 s wall on the production 7 arm, including the driver's own stop and stagger. Production 8
re-dumped its sidecar for the new image, which is a dump boot's cost and not this figure.

~~**Content types and mixed load have not been re-measured for four configurations.**~~ **Content
types were re-measured on production 12 on 7 September** `[measured-here]`: code **61.5**, math
**76.2**, JSON **73.1**, prose **29.0** tok/s at a single stream, acceptance 46 / 57 / 53 / **13 %**,
median of three rounds. Every category is inside its own round-to-round spread against production 9,
which means the three memory rungs and the indexer workspace bound bought **+37.8 % of KV pool
without costing any category tokens per second**. Full tables, C1 and C4:
[`../results/speed/category-speeds-production-12.md`](../results/speed/category-speeds-production-12.md);
why the prose row is what it is, §1.2 below. **Mixed load is still unrun on this configuration**
`[not tested]` — the last figure is the fast-boot arm's 7.0 tok/s with a 4.9 s TTFT for the long
prompt.

**And prefill has a third figure now, from a third instrument, so all three are written down with
their method rather than one of them being promoted over the others.** The rows in this repository are
not interchangeable `[measured-here]`:

| Prefill figure | Instrument and method | Date |
|---|---|---|
| **1,738** tok/s | `bench/prefill-fresh.py` — one fresh request at a time, a new seed per request, median of three. This is the row in the table above and in the README's headline | 5–6 September, production 9/12 |
| 1,575 tok/s | `scripts/prefill-7k.py` — the **warm repeated-prompt** figure, labelled as such everywhere it appears | 5 September, production 9 |
| 1,914 tok/s | 60 s counter delta under 24 fresh ~7K prompts, a generator **this repository does not ship**. Kept as published; not comparable with the row below | 7 September, production 13 |
| **1,867** tok/s | [`tracks/tp3/patches/flashkda/loadgen.py`](../tracks/tp3/patches/flashkda/loadgen.py) — `vllm:prompt_tokens_total` over a 60 s window opening 20 s after the load starts, 24 fresh nonce-prefixed ~7,000-token prompts in flight, `max_tokens=8`, **prefix-cache hits counted over the same window and zero**. The control arm of the same A/B, one environment variable apart, read **1,754**; the promoted configuration on the production fast-load path read **1,865.7** and **1,866.6** | 10 September, production 13 + FlashKDA |

The last row is a **sustained-throughput** measurement and the first is a per-request one; they answer
different questions and the gap between them is not a result. What *is* a result is the 1,754 → 1,867
inside the last row, because both halves came from the same instrument in the same session
([`../results/gates/flashkda-ab-10sep.md`](../results/gates/flashkda-ab-10sep.md)). The estimate that
preceded it was +3.5–4.1 % against a measured +6.5 %, and the excess is open as
[11](11-open-issues.md) §2.35.

### 1.1 The spread these numbers sit inside, and which reading the headline uses

Every headline figure on this page is **the median of that boot's sweep rounds** — three rounds with a
warm tuner cache, five with a cold one ([09](09-measurement-protocol.md) §1,
[12](12-tuner-cache.md)). That rule matters more than it looks, so here is the whole production 9 and
10 series, every round of every boot, aggregate tok/s `[measured-here]`:

| Boot | Configuration | C1 rounds | C1 median | C4 median | C8 median |
|---|---|---|---|---|---|
| `tp3full` | production 9, the promotion boot | 69.6 / 69.9 / 70.3 | **69.9** | 140.7 | 197.2 |
| `prod9r` | production 9, clean re-boot from `.env.tp3` | 69.8 / 70.4 / 68.6 | **69.8** | 134.6 | 192.4 |
| `L083` | production 10 (0.83) | 71.7 / 70.2 / 70.5 | **70.5** | 144.6 | 194.0 |
| `ALG5` | production 10 + `NCCL_ALGO=Ring,Tree`, five rounds | 70.9 / 68.8 / 71.5 / 70.6 / 67.7 | **70.6** | 142.6 | 195.7 |

Read down the columns rather than across the rows, because the two axes behave differently:

- **Round to round inside one boot**, C1 spans **1.0–5.7 %** peak to peak; the 5.7 % is the five-round
  arm, which is simply the arm with the most chances to show its tails. C4 spans **5.1–9.8 %** and C8
  **1.6–5.0 %**.
- **Boot to boot, comparing medians**, C1 is remarkably tight: **69.8 … 70.6, a 1.1 % span** across
  four boots and two memory settings. C8 spans **2.5 %** (192.4 … 197.2). **C4 spans 7.4 %**
  (134.6 … 144.6) and is the noisiest metric on this stack.

**Which means one number in this repository must not be read as a gain.** Production 10's C4 is
144.6 against production 9's 134.6, **+7.4 %** — and 7.4 % is exactly the boot-median span of C4 on
this configuration. A memory fraction does not buy 7 % of four-way throughput; the two arms are equal
at C4 and the table says so in its own footnote. The same guard applies in the other direction to
`prod9r`'s C4 of 134.6, which is the low end of the same spread rather than a regression from the
promotion boot.

**Anything under about 3 % at C1 or C8, or 8 % at C4, is not a result on one boot each.** That is why
the five-round rule exists and why single-boot A/B arms in this repository are reported as "inside the
band" rather than as small wins ([09](09-measurement-protocol.md) §1.2).

**One correction to our own working notes.** A "C1 spread of 59.9 … 70.6 across boots" appeared in the
field log for this configuration and it is wrong `[retracted]`: **59.9 is the TP=2 full-scope arm's C1
aggregate** ([13](13-full-scope-checkpoint.md) §4.1), not a TP=3 boot, and the other low reading
beside it was a production **8** restore. No production 9 or 10 boot has read below **67.7** even at
the level of a single round. Mixing a two-node arm into a three-node series would have doubled the
apparent noise floor and buried every real few-percent effect on this page underneath it.

### 1.2 Why prose is slow, and why it is the draft

The category row above has one number in it that looks like a different machine. On production 12,
single stream, median of three rounds `[measured-here]`:

| | prose | code | math | JSON |
|---|---|---|---|---|
| decode tok/s | **29.04** | 61.46 | 76.20 | 73.13 |
| draft acceptance | **13.02 %** | 45.94 % | 57.11 % | 52.90 % |
| accepted tokens per step | **1.91** | 4.22 | 5.00 | 4.70 |
| **step rate (decode ÷ tokens per step)** | **15.19 /s** | **14.58 /s** | **15.25 /s** | **15.55 /s** |

**Read the last row first.** With a k-deep draft the engine emits `L = 1 + k × acceptance` tokens per
target forward. Divide each category's decode rate by its own `L` and the step rate is the same in
all four columns — a 6.4 % spread, which is this probe's ordinary round-to-round noise — while the
row above it spans a factor of 2.6. **The engine steps at the same rate whatever the content. The
only thing that changes is how many tokens come out of each step, and that is the drafter's hit
rate.**

Three further measurements close the attribution:

- **Acceptance per category has not moved across ten boots** `[measured-here]`. Prose has read
  **12.1 – 13.1 %** across four image builds, both draft page sizes, four memory fractions, the mesh
  patches, the indexer workspace bound and the full-scope checkpoint promotion; code 44.6 – 46.9 %,
  math 55.0 – 58.0 %, JSON 52.8 – 55.2 %. Production 12 sits inside every one of those bands.
- **The checkpoint is not the variable.** The full-scope promotion (production 9, §2.3) raised the
  step rate from about 11.7 /s to about 15.3 /s — that is what a 4-bit dense stage bought — and left
  acceptance untouched in all four categories. It moved every category by the same 30 %; it did not
  change the ratio between them.
- **Speculation is barely paying on prose.** The one arm measured with *and* without the drafter is
  the two-node one ([04](04-dflash2-port.md) §1): 14.73 tok/s per stream unspeculated against 50.79
  at k=7, where the speculative arm advances 5.46 tokens per step. A speculative step therefore costs
  **1.58×** a plain step, break-even is **8.3 % acceptance**, and at 13.02 % prose gets about
  **×1.21** out of speculation where math gets ×3.16 `[estimate]` — the overhead factor is measured
  at two nodes, an unspeculated three-node arm is `[not tested]`. Prose runs close to what this
  engine does with no drafter at all, and pays the verify overhead to stay there.

**No configuration lever exists for it, and three were looked for.** `k=5` raises prose by 3.6 % at
C1 and costs every other category and every concurrency level 3.5 – 6.4 % ([04](04-dflash2-port.md)
§6). Per-request `k` is not available: `num_speculative_tokens` lives in `--speculative-config` and is
fixed at engine start, so one value serves every request. And a **newer published revision of the
same drafter was booted on production 12 on 7 September** and came back equal — acceptance 62.08 →
61.32 % at C1 and 60.53 → 61.61 % at C8, opposite signs, both inside the 60–65 % band, all gates full
first time; the ~5 % speed reading on that arm belongs to the boot rather than the draft, because
prefill fell by the same 5 % and prefill does not use the drafter ([04](04-dflash2-port.md) §8.1).
**Its per-category split was then run as well, and prose does not move**: acceptance 12.0 % on that
arm's measured round and 13.1 % on its warm one, a 1.1-point swing inside one boot that brackets
production 12's 12.97 % rather than beating it, while code and JSON are equal to a third of a point
`[measured-here]`. The one column that may have moved on that arm is **math**, 3.2 points of
acceptance above every previous arm — one boot, not carried by the aggregate, and left as a two-boot
question rather than a result
([`../results/speed/category-speeds-production-12.md`](../results/speed/category-speeds-production-12.md)
§2.5).

**So the plain statement.** Prose is slow because the DFlash2 draft agrees with the target model
about one token in eight on free-running prose. It is not the checkpoint, not the kernels, not the
mesh, not the KV geometry and not the memory fraction — ten arms of those moved prose acceptance by
one point in total. **A better draft checkpoint is the only thing that fixes this row**, and a
different drafter demonstrably can: an MTP head at k=3, measured in the same session as DFlash2 on
the two-node arm, *wins* prose at 21.3 against 18.5 tok/s while losing every other category by
20–40 % ([04](04-dflash2-port.md) §7). Until such a checkpoint exists, budget the prose row as
measured. Full evidence, every arm and both concurrency levels:
[`../results/speed/category-speeds-production-12.md`](../results/speed/category-speeds-production-12.md).

---

## 2. How it got there

Each row is a boot with its own gates. Aggregate tok/s, medians of rounds 3–5 `[measured-here]`.

| Arm | C1 | C4 | C8 | prefill-fresh | KV pool | what changed |
|---|---|---|---|---|---|---|
| TP=2 + DFlash2 (two nodes) | 42.9 | 83.9 | 114.6 | — | 825,000 | reference |
| TP=3 + EP, stock kernels | 40.8 | 59.2 | 91.9 | — | 2,587,828 | the third node; **slower** |
| + the `n_rows` kernel fix | 49.4 | 97.0 | 137.2 | 1,474 | 2,571,230 | [05](05-expert-parallel-and-cuda-exl3-fixes.md) §2 |
| upstream `bc0e0f6` + combine staging, MNBT 4096 | 51.1 | 104.8 | 150.9 | 1,761 | 1,627,170 | adopted upstream, retired our patch |
| `f4987cf`, MNBT 2048 | 51.9 | 107.0 | 153.8 | 1,645 | 2,428,769 | skip MoE padding rows; pool back |
| + `NCCL_MAX_NCHANNELS=8` | — | — | 150.8 | 1,728 | 1,648,621 | [06](06-nccl-mesh.md), measured on the 4096 arm |
| + draft page 256 | 52.8 | 117.1 | 162.8 | 1,508 | **4,413,223** | [07](07-kv-and-draft-page.md) |
| + fast-boot sidecar | 54.4 | 114.6 | 161.8 | 1,704 | 4,484,848 | [08](08-fast-boot.md); boot 618 → 274 s |
| + `9bf594c`, tuner cache warm | 54.5 | 112.0 | 159.9 | 1,709 | 4,429,752 | [12](12-tuner-cache.md); speed unchanged by design, protocol 5 rounds → 3 |
| + dual cable + `NCCL_PTR_CUDA` (production 6) | 56.9 | 118.5 | 168.9 | 1,792 | 4,449,035 | [06](06-nccl-mesh.md) §6–§8; the second cable of every pair had never carried a packet |
| + fp8 draft cache + settle gate (production 7) | 57.0 | 120.0 | 175.1 | 1,769 | 4,699,724 | [07](07-kv-and-draft-page.md) §7; speed unchanged, pool +5.6 %, the first pool number taken with a settled baseline |
| + image `62f53e6` (production 8) | 56.8 | 119.5 | 172.8 | 1,780 | 4,674,931 | upstream's `had_in` fix; every column inside its own band, which is what a 0.2–0.3 % change should look like |
| **+ the full-scope checkpoint (production 9)** | **69.9** | **140.7** | **197.2** | **1,738** | **5,165,289** | [13](13-full-scope-checkpoint.md); the dense stage goes from BF16 to 4–6 bit. Step 88.2 → 70.3 ms, pool +10 %, memory −3.4 GiB per node, acceptance −2.4 pt |

Four of those rows are the interesting ones. **The third node initially made the machine slower**, by
8–29 %, and that was a one-line kernel bug rather than a cost of the arrangement. **The largest single
jump in the KV pool cost no memory at all** — it was a per-request block counter, not bytes. The
**dual-cable row is not a tuning win at all**: half the fabric had never been used, by any workload,
since the cluster was built. And **the last row is not a tuning win either** — it is a different
checkpoint, and it moved every column further than every tuning row after the kernel fix put
together. Several rows move by less than their own spread and are in the table because they changed
the boot, the pool or the measurement protocol rather than the speed — the tuner cache, the fast-boot
sidecar, production 7 and production 8.

Rejected on the way, each with its own boot and gates:

| Arm | Why rejected |
|---|---|
| `NCCL_PROTO=Simple` | 2.8× worse at the C1 decode message, 4.4× at C8, no better at 16 MB. No boot spent — model-free `[measured-here]`. |
| draft depth k=5 | Higher acceptance rate, lower accepted tokens per step; −6.4 % at C1, −3.5 % at C8. Wins only in prose and at C4 `[measured-here]`. |
| `gpu-memory-utilization 0.85` | +19 % KV pool, no speed change, head node at 1.9 GiB free with 1.6 GB swap — breaks the 4 GiB rule `[measured-here]`. |
| `--max-num-batched-tokens 4096` | +9.5 % fresh prefill, −13 % mixed-load TTFT, −28.5 % KV pool. A judgement, reversible in one line `[measured-here]`. |
| MoE input-transform fusion (`61a17bc`) | +1–4 % end to end, but a later upstream commit takes the same win another way and upstream dropped the branch `[measured-here]`. |
| A 2,304-padded tensor-sliced arrangement instead of expert parallelism | On the fixed kernel it loses everywhere except M=1 and costs +12.5 % expert bytes out of the KV pool `[measured-here]`. |
| A one-sided RDMA_WRITE mesh transport (`patches/kernel/0007`) | Removes RNR retries to exactly zero and moves throughput by nothing: engine C1 56.4, C8 171.1, prefill-fresh 1,763 against production's 56.9 / 168.9 / 1,792 — differences in both directions, inside boot spread. The ceiling is the cards' PCIe slots, not the flow control ([06](06-nccl-mesh.md) §9–§10) `[measured-here]`. |

### 2.1 The draft cache at fp8: validated on a dump boot, promoted on a load boot

The DFlash2 drafter's own KV cache was `bf16` while the main cache was `fp8`. Moving it to fp8 shrinks
the drafter's page and grows the pool ([07](07-kv-and-draft-page.md) §7). The open question was never
the arithmetic — it was whether the draft's sliding-window backend accepts an fp8 cache at all, and
whether a drafter attending over fp8 still proposes as well.

It took **two boots** to answer, and the pair is a worked example of the dump-versus-load rule
([09](09-measurement-protocol.md) §11). The validation arm added three prelude patches, which
invalidates the fast-load sidecar and forces a dump boot, so its speed and quality lines are valid and
**its KV pool line is not**. The promotion arm is the ordinary load boot that followed, with the
launcher's settle gate in place `[measured-here]`:

| | production 6 | draft KV fp8, **dump** boot | **production 7**, load boot |
|---|---|---|---|
| C1 / C2 / C4 / C6 / C8 aggregate | 56.9 / 84.2 / 118.5 / 142.9 / 168.9 | 56.0 / 82.3 / 121.7 / 143.6 / 175.5 | **57.0 / 80.9 / 120.0 / 143.4 / 175.1** |
| TTFT, C1 / C8 | 0.41 / 1.01 s | 0.40 / 0.91 s | **0.34 / 0.91 s** |
| draft acceptance | 61–65 % | 60.1–64.0 % (one C1 round at 57.3 %) | **60.8–64.3 %** (per-concurrency medians; full round range 59.4–65.9 %) |
| accepted tokens per step | 5.3–5.5 | 5.2–5.5 | 5.3–5.5 (round range 5.16–5.61) |
| prefill-fresh (median of 3) | 1,792 | 1,790 | 1,769 |
| prefill 7K, warm repeat | 1,506 | 1,532 | 1,529 |
| gates, cold and warm | 10/10 · 12/12 | 10/10 · 12/12 | **10/10 · 12/12** |
| free RAM / swap | 11.3 / 12.6 / 12.5 GiB · ~0.1 | 14.5 / 15.8 / 15.8 GiB · ~0.1 | 12.3 / 13.5 / 13.3 GiB · ~0.1 |
| KV pool | 4,449,035 | 4,382,920 — **not usable** | **4,699,724 (+5.6 %)** |

**The claim this supports is "it costs nothing and buys pool", not "it is faster."** C8 reads +3.7 %
on the load boot, which clears that metric's ±3 % band ([09](09-measurement-protocol.md) §1.2) — and
C2 reads −3.9 % on the same three rounds, which clears its band in the other direction. Two boots with
opposite-signed excursions in different columns is what noise looks like, and there is no mechanism by
which a smaller draft cache would speed up decoding. The number that promoted the change is the pool,
and it needed the load boot: **+5.6 % against +4.7 % predicted**, on the first boot of this stack whose
memory baseline was pinned rather than inherited from whatever the previous container had not yet
released ([07](07-kv-and-draft-page.md) §1.1). Closed: [11](11-open-issues.md) §2.18.

The mechanism is confirmed in the engine's own log rather than inferred: the drafter's page goes
393,216 → **196,608 bytes**, per-block cost 21,917,440 → **20,934,400 bytes** (−4.5 %), and the
blocks-per-request divisor stays at 363 — which is the whole point, since that divisor is what
collapsed the pool before ([07](07-kv-and-draft-page.md) §1).

### 2.2 A full-scope checkpoint at TP=2: the dense-stage lever, measured

This arm is **not** in the progression above, because it was run at TP=2 on two nodes as the dress
rehearsal for the row that is — production 9, §2.3. It is the experiment that answered the largest
open item this repository carried ([11](11-open-issues.md) §2.22): what is the BF16 dense stage
actually worth, if the same layers were 4-bit. The full story — why the checkpoint would not load,
the loader patch, and the TP=3 port — is [13](13-full-scope-checkpoint.md).

Two of its readings did **not** survive TP=3 and are flagged where they appear: the memory figure
below (§6.2 of [13](13-full-scope-checkpoint.md)) reversed sign, and a cold-probe acceptance collapse
did not reproduce at all.

**Settings, both arms identical unless the row says otherwise:** **two** nodes, **TP=2**, expert
parallel **off**, image `exl3-zeus:62f53e6`, KV `fp8`, DFlash2 draft k=7,
`gpu-memory-utilization 0.85`, `--block-size 256` requested, `--max-num-seqs 8`,
`--max-num-batched-tokens 2048`, `NCCL_MAX_NCHANNELS=8`, tuner cache warm, **no** fast-load sidecar in
either arm, **no** `HAREM_SW_BLOCK_SIZE`, **no** fp8 draft cache in either arm, temperature 0,
reasoning effort `low`, medians of three rounds, 5 September 2026. Full scope is
`turboderp/GLM-5.3-Flash-exl3` at 4.05 bpw (`2a30229e`, MIT); the control is the production
checkpoint. **The one setting that is not identical:** the full-scope arm runs
`max_model_len 65,536` against the control's `1,000,000`, because at 1M it could not boot. That
changes the hybrid allocator's page size as well as the pool, so **only C1–C4 are a comparison** —
C6/C8, every prefill figure and the pool are not.

| metric | **full scope** | experts-only control | delta |
|---|---|---|---|
| **C1 per stream** | **68.00** tok/s | 54.69 | **+24.3 %** |
| **C1 aggregate** | **59.93** tok/s | 47.40 | **+26.4 %** |
| C2 aggregate | 83.02 | 68.03 | +22.0 % |
| C4 aggregate | 111.05 | 90.66 | +22.5 % |
| C6 / C8 aggregate | 109.75 / 110.03 | 110.12 / 133.57 | **KV-bound, void** |
| TTFT at C1 | 0.524 s | 0.615 s | −14.8 % |
| draft acceptance at C1 | 63.14 % | 64.08 % | equal |
| accepted tokens per step at C1 | 5.42 | 5.49 | equal |
| boot, cold, no sidecar | 355 s | 396 s | −10 % |
| gates, cold and warm | **10/10 · 12/12** | 10/10 · 12/12 | equal |
| MMLU sample (1,995 q) | **86.32 ±0.75** | 86.4 ±0.7 (not re-run — see below) | inside the bar |
| KV pool | 31,343 at 65,536 ctx | 665,625 at 1,000,000 ctx | not comparable |
| prefill, 7K and fresh | **not measurable** | 1,135 / 1,334 tok/s | see below |

All figures `[measured-here]`. **The step arithmetic is the result**, not the tok/s: 5.49 tokens per
step at 54.69 tok/s is a 100.4 ms step; 5.42 at 68.00 is **79.7 ms**. **20.7 ms saved per step, with
acceptance and accepted length unchanged** — the whole gain is arithmetic, none of it is drafter
behaviour. Against the estimate this page carried (42.90 ms of dense BF16 GEMM going to ~11 ms, about
+34 % single-stream), **65 % of it arrived**; the rest is the part this checkpoint leaves in BF16
anyway ([13](13-full-scope-checkpoint.md) §4.2). The estimate was an upper bound, and it is not
retracted so much as bounded.

**The control's MMLU was not re-run in this arm.** It is the same 86.4 ±0.7 quoted everywhere in this
repository — same checkpoint, same TP=2, measured earlier the same day with the MTP drafter rather
than DFlash2. A log-likelihood task does not go through the speculative decoder, so the number
transfers; it is flagged here so the table is not read as two fresh runs.

**The cost is context, and it is the reason this is not a serving configuration.** The full-scope
model is ~10 GiB *heavier* per node at TP=2 despite being 10 GiB smaller on disk (§6.2 of
[13](13-full-scope-checkpoint.md) leaves that contradiction standing, unexplained), so the pool comes
out at 31,343 tokens — about 6.8 pages once the hybrid allocator raises the attention block to 4,608.
Measured admission: 844- and 1,684-token prompts serve normally; **~2,800 tokens and above are never
scheduled at all** (`Running: 0, Waiting: 1`, KV usage 0 %, indefinitely). That is what voids C6/C8
and every prefill number here.

> **Correction to the version of this table posted upstream** `[retracted]`. The issue-thread copy
> gave the control's C1 as "54.7 / 54.3" aggregate / per stream and derived "+9.5 % / +25 %". 54.7 is
> the control's **per-stream** median and 54.3 its round-3 per-stream value; the aggregate median is
> **47.40**. The like-for-like deltas are the ones above. The full-scope column, the quality gate and
> the conclusion are unaffected ([11](11-open-issues.md) §1.9 row 31).

### 2.3 The same lever at TP=3, and what promoted it

Production 9. Same three nodes, same everything else, medians of three rounds, against a control that
is the **pool of two same-day runs** of the same script because that arm's run-to-run spread is about
7 % `[measured-here]`. Full table, boot ledger, per-round figures and the cost line:
[13](13-full-scope-checkpoint.md) §7.3–§7.4.

| metric | **full scope, TP=3** | experts-only, TP=3 (production 8) | delta |
|---|---|---|---|
| C1 total / per stream | **69.90 / 75.91** tok/s | 56.88 / 62.39 | **+22.9 % / +21.7 %** |
| C2 / C4 / C6 total | 99.17 / 140.72 / 172.40 | 83.31 / 120.22 / 144.03 | +19.0 / +17.1 / +19.7 % |
| C8 total | **197.20** | 175.37 | **+12.5 %** |
| TTFT, C1 / C8 | 0.280 / 0.826 s | 0.344 / 0.906 | −18.6 / −8.8 % |
| prefill, fresh / 7K repeat | 1,738 / 1,575 | 1,776 / 1,537 | **equal** both ways |
| draft acceptance at C1 · tokens per step | 61.94 % · 5.34 | 64.36 % · 5.50 | **−2.4 pt · −3.0 %** |
| consumed memory per node | 58.3–59.1 GiB | 62.1–62.4 | **−3.4 GiB** |
| KV pool at 0.80, 1M context | **5,165,289** | 4,696,969 | **+10.0 %** |
| boot, fast-load | 251 s (weights 57.9) | 264 s (weights 73.2) | −5 % |
| gates cold and warm · MMLU sample | 10/10 · 12/12 · **86.47 ±0.74** | 10/10 · 12/12 · 86.4 ±0.7 | equal |

**The two things this row demonstrates that §2.2 could not.** First, the TP=2 arm's two loudest
warnings were artefacts of the rig: at three ranks the arm is *lighter* rather than 10 GiB heavier,
and draft acceptance is flat on the probe that had shown it collapsing. Second, and more useful:
**the step-time saving reproduced across topologies to within a millisecond and a half** — 20.7 ms at
TP=2, 17.8 ms at TP=3 — which is the strongest evidence on this page that the mechanism was
understood rather than merely observed.

---

## 3. Against the NVFP4 sibling stack

Same three nodes, same model, same draft, same prompt set, different quantization path. The NVFP4
figures are from [`NNNtrance/GLM-5.3-Flash-NVFP4-TP3-3x-DGX-Spark`](https://github.com/NNNtrance/GLM-5.3-Flash-NVFP4-TP3-3x-DGX-Spark)
`[measured-here]`.

| | **EXL3 TP=3, production 9** | EXL3 TP=3, production 8 | NVFP4 TP=3 |
|---|---|---|---|
| C1 aggregate | **69.9** | 56.9 | 57–60 |
| C8 aggregate | **197.2** | 175.4 | 150 |
| prefill, fresh | **1,738** | 1,776 | 1,585 (7K) |
| TTFT C1 | **0.280 s** | 0.344 s | 0.38 s |
| KV pool | **5,165,289 @ 0.80** | 4,696,969 @ 0.80 | 4,321,739 @ 0.88 |
| consumed memory per node | **58.3–59.1 GiB** | 62.1–62.4 GiB | — |
| boot to serving | **251 s** | 264 s | ~300 s |
| gates | 10/10 · 12/12 | 10/10 · 12/12 | 10/10 · 12/12 |
| MMLU | **86.47 ±0.74** (1,995-question sample, TP=3) | 86.4 ±0.7 (sample, at TP=2) | 85.9 ±0.3 (full, 14,042 questions) |

**EXL3 is now ahead on single-stream decode, aggregate throughput, memory and boot, and level on
prefill.** Two days earlier this row read "behind on single-stream and prefill". The KV comparison
stays the sharper one: EXL3 reaches a **19 % larger** pool at `gpu-memory-utilization 0.80` than
NVFP4 reaches at 0.88, which is headroom NVFP4 does not have.

Four caveats that matter, and they all point the same way. The MMLU numbers are not comparable — one
is a full run, ours are 1,995-question samples. The prefill columns are not the same measurement:
ours is `prefill-fresh` on unseen prompts, the NVFP4 figure is a warm 7K prompt, and the honest
comparison would need both stacks measured the same way. **Three findings on this page are
fabric-level, not format-level** — `NCCL_MAX_NCHANNELS=8` (+13 % at C8), the idle second cable, and
`NCCL_PTR_CUDA` — all three use the same plugin over the same wiring and **none of them has been
applied to the NVFP4 stack** `[not tested]`. And **the largest single item, production 9, is
checkpoint-level rather than format-level**: an NVFP4 checkpoint that quantized the same dense path
would collect the same ~18 ms per decode step, and none is known to exist `[not tested]`. Read this
table as "this is what the EXL3 stack does today", not as "EXL3 beats NVFP4".

### 3.1 Quality, on the same comparison: three benchmarks, two won and one lost

The row above says "gates 10/10 · 12/12" on both stacks, which is true and thin. On 6–7 September
production 12 ran three of the sibling's benchmarks against its 3 September battery — same harness
version, same flags, same three nodes, temperature 0, effort `low` `[measured-here]`:

| | **EXL3 TP=3, production 12** | NVFP4 TP=3, 3 Sep |
|---|---|---|
| GSM8K, 200 questions, 8-shot CoT | **97.5 %** (195/200) | 94.0 % (188/200) |
| IFEval, 541 prompts | prompt **80.0 %**, instruction **86.0 %** | prompt 78.9 %, instruction 85.1 % |
| tool-eval-bench, hardmode, 88 × 8 trials | **85.5 ±1.3** (`final_score` 86) | **87.8 ±0.9** |
| MMLU | 86.47 ±0.74 (1,995-question sample, TP=3) | 85.9 ±0.3 (full, 14,042 questions) |
| needle at 1M | **20/20** (8 September, effective context 997,952 tokens, 1 h 25 min 32 s) | 20/20 (5,288.6 s) |

**The tool-eval line is the one to read properly, and it is not a quality collapse.** The −2.3 points
is real (permutation p = 0.0048) and it is **four scenarios out of 88** — on the other 84 this stack
is 0.2 ahead, nine of fourteen categories are identical digit for digit, and malformed calls,
timeouts, empty content and refusals are zero on both sides. TC-51 alone is 47 % of it and is a
**grader ordering rule**, not a wrong answer: this stack issues the calendar event and its
notification in the *same* turn, which the harness's own `parallel_tool_calls: true` invites and that
scenario's Python fails outright. The chat-template explanation was tested with the old template as
the single variable and **refuted**. What is still confounded is the checkpoint against the vLLM
build, which changed together; the next step is a build A/B.
[`../results/gates/quality-battery-production-12.md`](../results/gates/quality-battery-production-12.md)
has the scenario table, the statistics and the template arm; [11](11-open-issues.md) §2.30 keeps it
open.

**Since 8 September there is a second reading of that row, and it changes what the −2.3 points
means.** The same engine, unchanged, scores **90.3 ±1.2** on the same 88 scenarios at reasoning
effort `high` — above the sibling's `low`-effort 87.8 ±0.9, permutation p = 0.012 `[measured-here]`.
Three of the four scenarios the deficit was concentrated in go level or better, TC-21 from 0.00 to a
full 2.00. The effort was raised by a proxy that injects `reasoning_effort` into the harness's
requests, so the server was never restarted or reconfigured and production still serves at `low`.
Read together, the two rows say the `low`-effort gap was a **thinking-budget** difference on
planning-shaped scenarios rather than a capability difference in the quantisation or the engine —
and the build/checkpoint confound of §2.30 is untouched by it and stays open. The price is +90 %
completion tokens and +24 % median turn latency:
[`../results/gates/quality-battery-production-13.md`](../results/gates/quality-battery-production-13.md).

---

## 4. Roofline

### 4.1 Measure the ruler first, and this one moved

Every roofline percentage published in this repository before 5 September was against a **vendor**
number, and every one of them was about **22 % optimistic** `[retracted]`. Both rulers have now been
measured on this device, in our own image, in the same process that ran the benchmarks
`[measured-here]`:

| ruler | measured on GB10 | vendor / implied | achieved |
|---|---|---|---|
| device read bandwidth, bf16 `sum` over 4 GiB | **225.2 GB/s** | 273 GB/s | 82 % |
| device copy bandwidth, read+write, 4 GiB | 214.5 | — | — |
| device read bandwidth, 2 GiB buffer | 205.8 | — | — |
| **BF16 dense GEMM peak**, 8192³ `torch.matmul` | **97.3 TFLOP/s** | ~125 (1 PFLOP FP4 ÷ 8) | 78 % |
| BF16 GEMM at an engine shape, 2032×5632×4096 | 80.4 | — | — |
| BF16 GEMM at an engine shape, 2032×4096×4096 | 91.8 | — | — |

Two tools, both in `bench/`, both a few seconds: [`bench/bw.py`](../bench/bw.py) and
[`bench/gemmpeak.py`](../bench/gemmpeak.py). Run them **in the same binary and the same run** as
whatever you are measuring, and quote the result beside every efficiency claim.

**The ruler itself drifts.** Three independent measurements the same morning on the same idle machine
gave 225.2, 239.6 and 240.9 GB/s — a 6.5 % spread `[measured-here]`. Percentages below are therefore
given against a band, not a point, wherever the difference matters. A `memset` ruler (`.zero_()`)
measured 196.8–198.2 GB/s, which is the right comparison for a kernel that only writes.

### 4.2 The byte model

From the checkpoint's own shapes `[estimate]`: 45 layers of which 43 carry routed experts (3–45;
layer 45 is the MTP layer and carries its own 288); 288 experts, top-8, one shared expert; hidden
4096, routed intermediate 2048. A routed expert is 3 × 4096 × 2048 = 25.2M parameters, about
**12.9 MB** at 4 bpw with its scales. Under expert parallelism each node holds 96 experts of 288, so
43 × 96 × 12.9 MB ≈ 49.6 GiB of expert weight per node — and the measured figure is 54.86 GiB, which
leaves about **5.3 GiB of BF16 non-expert weight per node** (attention, KDA, the shared expert,
`lm_head`). That reconciliation is the reason to trust the rest of the table.

With a k=7 draft a decode step verifies 8 tokens per sequence, so the expected number of distinct
experts touched per layer is `288 × (1 − e^(−8·tokens/288))`.

| Regime | Measured | Bytes per step per node `[estimate]` | Effective bandwidth | Share of **225 GB/s** | Share of 97.3 TFLOP/s |
|---|---|---|---|---|---|
| Decode C1, DFlash2 k=7 | 89.1 ms per engine step (measured directly, §5.3) | 8 verify tokens → ~19 of 96 experts per layer: 10.6 GB expert + 5.3 GB dense ≈ **15.9 GB** | ≈ **178 GB/s** | ≈ **79 %** | ≈ 2 % |
| Decode C8, DFlash2 k=7 | 223 ms per engine step (measured directly, §5.3) | 64 verify tokens → ~80 of 96 experts per layer: 44.3 GB + 5.3 GB ≈ **49.6 GB** | ≈ **222 GB/s** | ≈ **99 %** | ≈ 6 % |
| Prefill, **1,792**-token chunk | **962.55 ms** per chunk, production 7 (mean of 3 ranks, measured directly, §5.2); production 9 measures **961.73 ms** — equal | every expert touched: 53.3 GB + 5.3 GB ≈ **58.6 GB** | ≈ **61 GB/s** | ≈ **27 %** | ≈ 24 % |

**The prefill row's chunk is 1,792 tokens, not 2,048** `[measured-here]`. With `--block-size 256` the
scheduler issues **7 × 256 = 1,792** of the 2,048 tokens `--max-num-batched-tokens` budgets, so
12.5 % of the budget is unused before it is argued about. This row previously read "2048-token chunk,
1.109 s" — a chunk size taken from the flag rather than the trace, and a wall from an earlier
configuration. Both are corrected here against the measured values in
[`../results/profile/measured-prod7.md`](../results/profile/measured-prod7.md) §1–§2 and the
production-9 profile summarised in §5 and
[`../results/profile/step-breakdown.csv`](../results/profile/step-breakdown.csv). The byte column is
unchanged, because at prefill every local expert is read once per layer whatever the chunk size — so
the effective bandwidth rises from the 53 GB/s this table used to print to about **61 GB/s**, while
the compute share barely moves (tokens and wall fell by 12.5 % and 13.2 %). Percentages elsewhere in
this document that were computed against a 1,109 ms chunk are corrected in place; §5.5 shows the
working for the one that mattered.

**Reading.** Decode with the draft is **at** the memory roof, not near it — the C8 row lands at 99 %
of the measured ruler, which is a way of saying the byte model and the measurement agree rather than
that the hardware is saturated to the last percent. Faster kernels buy nothing there; the levers are
fewer bytes (a checkpoint that also quantizes attention) or higher acceptance. **Prefill is far from
both roofs**, at a bit over a quarter of bandwidth and under a quarter of compute, and that is where
kernel work can still pay. §5 replaces this arithmetic with a measured breakdown.

Assumptions, and they are not small: KV reads are ignored (short contexts), the draft model's own
weights are ignored, expert-touch counts are expectations rather than measured histograms, and a
4 bpw expert is taken as 12.9 MB. **Treat every ratio as ±15 %.** What backs the byte model is that
the derived per-node weight total lands within 1 % of the measured one.

---

## 5. Where a step actually goes

§4 is arithmetic. This section is the **measurement** — a torch-profiler trace of the live server,
all three ranks, no restart and no reconfiguration. Full tables:
[`../results/profile/measured-prod7.md`](../results/profile/measured-prod7.md) `[measured-here]`.

**Read the version note first.** Until 5 September this section carried a *reconciliation*: the
per-class ratios of an older trace normalised onto a newer wall clock, with a 2.8 % residual booked
to NCCL and every class marked ±3 %. That table is superseded. The engine was relaunched with
`--profiler-config` set ([09](09-measurement-protocol.md) §4.1), `/start_profile` was called on the
running production configuration 7, and every row below is read out of the trace of the configuration
it describes. The reconciliation was good in aggregate — its per-token prefill cost was 1.5 % high —
and **two of its target rows were fiction**: `exl3_moe_combine`, priced at 1.5 % of a chunk, does not
exist as a kernel in this build, and `_zero_kv_blocks`, priced at 1.3 %, measures 0.09 %. Both had
ranked in the top ten targets `[retracted]`.

**Two traps in reading such a trace**, because anyone repeating this will hit them. The engine emits
one `record_function` per engine step, which kineto projects as a `gpu_user_annotation` named
`execute_context_N(T)_generation_M(T)`. In decode those arrive as **overlapping pairs** — merge them,
or the step count doubles and the trace appears to show a 50 % GPU bubble that is not there. And the
**drafter runs outside that annotation**, which is inconvenient for one paragraph and a gift for
everything after it: it gives an exact target-versus-draft split for free, and it is what corrected
the drafter's own cost from 19.5 % of a C1 step to 11.4 %.

**The profiler's cost was measured, not assumed**: the same windows with the profiler off read 0 %
different on prefill, +2.5 % on C1 and +1.3 % on C8, so every *share* below carries to a
profiler-free engine within 2.5 %. The absolute *idle* figures do not, and §5.7 subtracts them.

Production configuration **8** differs from this arm by one image (`62f53e6`) and moves no speed
number outside its band, so this breakdown is read as production 8's as well — with the `had_in` row
now a little smaller than the 5.6 % printed here `[not tested]`.

**Production configuration 9 has since been profiled with the same protocol, and the shares below
are superseded for that arm** `[measured-here]`. The run was made against the live production-9
server on the evening of 5 September 2026 — three ranks, `/start_profile` and `/stop_profile` only,
no restart and no reconfiguration — over the same three windows: a fresh unseen 8,497-token prompt
(six chunks, four of them 1,792 tokens), 94 steady C1 decode steps, and 79 C8 decode steps.

| Class | Prefill chunk | Decode C1 | Decode C8 |
|---|---|---|---|
| MoE trellis GEMM | **28.1 %** | **32.5 %** | **56.3 %** |
| NCCL collectives | 14.0 % | 26.1 % ‡ | — |
| Dense EXL3 GEMM | 13.4 % | 15.0 % | — |
| Hyper-connection mixing | 11.9 % | 2.6 % | — |
| MLA attention | 8.3 % | 0.8 % | — |
| KDA / GDN linear attention | 8.0 % | 1.8 % | — |
| Remaining BF16 linears | 3.8 % | 10.3 % | — |
| CPU gap (GPU idle) | 0.9 % | 8.4 % ‡ | — |
| **Dense stage total** (EXL3 GEMM + EXL3 Hadamard + remaining BF16) | **19.2 %** (184.73 ms) | **25.9 %** (21.90 ms) | — |

‡ **CUPTI-inflated at C1 and not comparable to the production-7 column.** Production 9 launches
2,738 kernels per step, so the profiler adds about 16 % to the step; with it off, NCCL and the CPU
gap together are **≤17.19 ms** rather than the 29.1 the trace prints. Wall per step was 84.44 ms
with the profiler on against 72.52 ms with it off. On prefill the same overhead is +1.4 %, so the
prefill column is safe to read at face value. Every class, with its milliseconds and call counts, is
in [`../results/profile/step-breakdown.csv`](../results/profile/step-breakdown.csv).

**What moved, and it is one row.** The dense stage went from **45.3 % / 42.90 ms** of a C1 step to
**25.9 % / 21.90 ms** — the 21 ms that is the whole of production 9's +22 %, since draft acceptance
did not move at all once it was pooled properly (+0.18 points, §1). **What did not move:** the NCCL
class is still 100 % exposed
(measured comm/compute overlap 0.00 ms per prefill step), the MoE trellis GEMM is still the largest
compute class in every window, and both of the targets we retracted in §5.8 are still absent.

**And one thing the arithmetic did not predict.** In *prefill* the full-scope dense stage is
**slower**, not faster: 184.73 ms against 167.39 ms, **+10.4 %**. Wall-clock prefill still landed
inside the ±3 % equality band, so no published prefill figure changes — but the full-scope gain is a
**decode** gain and it does not generalise to the prefill path.

The production-7 and production-8 shares below are kept as measured, because they are the control the
paragraph above is read against.

### 5.1 The totals, and what a day of fabric and cache work did

Live server, fresh unseen prompts, `max_tokens=1`, 52 requests over five length steps
`[measured-here]`:

| prompt tokens | wall (s) | tok/s |
|---|---|---|
| 1,032 | 0.773–0.840 | 1,192–1,341 |
| 2,026–2,090 | 1.208–1.258 | 1,655–1,677 |
| 3,980–4,088 | 2.260–2.292 | 1,761–1,784 |
| 6,142–6,311 | 3.519–3.598 | 1,745–1,754 |
| **8,423–8,427** | **4.669–4.680** | **1,801–1,804** |

A least-squares fit over all 52 points gives **0.5456 ms per prompt token marginal** and a 139 ms
intercept (HTTP, tokenise, sample, detokenise).

| | one configuration earlier | production 6 | change |
|---|---|---|---|
| steady 2,032-token prefill chunk | 1,262.3 ms | **1,109 ms** | **−12.2 %** |
| end-to-end per prompt token, 8.3–8.4K | 0.7892 ms | **0.5547 ms** | **−29.7 %** |
| prefill throughput, 8.4K fresh | 1,257 tok/s | **1,802** | **+43 %** |
| C1 decode, ms per engine step | ~108 | **89.1** | **−17.5 %** |
| C8 decode, ms per engine step | — | **223** | — |

The gap between −12 % *inside* a chunk and −30 % end to end is the MLA tuner cache
([12](12-tuner-cache.md)): the older run paid 411 ms extra on its first chunk and 665 ms on its
17-token tail pass, re-tuning on every new shape.

### 5.2 Prefill, one steady chunk — and the chunk is 1,792 tokens, not 2,048

The first thing the trace says is about the scheduler, not about a kernel. The 8,497-token prompt was
split `1792 · 1792 · 1792 · 1792 · 1024 · 305`, and **1,792 = 7 × 256 = 87.5 % of the batched-token
budget**: with `--block-size 256` the scheduler issues seven blocks of the eight-block budget and one
block goes unused on every chunk `[measured-here]`. The re-derived table assumed a 2,032–2,048-token
chunk, so its per-token arithmetic was against a chunk 12.5 % larger than the one that runs. Whether
that is a scheduler reservation or block alignment is **open** ([11](11-open-issues.md) §2.5); if it
closes, the MoE stage's per-token cost falls with it.

Per-token cost agrees anyway: 963.27 ms / 1,792 = **0.5375 ms per token**, against 0.5458 re-derived.

**wall 962.55 ms · GPU busy 954.4 ms · occupancy 99.2 % · launch gap 0.85 %** — prefill is entirely
kernel-bound and there is nothing to win on the host side. Mean of three ranks, four steady chunks
`[measured-here]`:

| class | ms | % of step | calls | re-derived % | error |
|---|---|---|---|---|---|
| **MoE trellis GEMM** (`exl3_gemm_m_kernel`) | **274.42** | **28.51 %** | 84 | 26.4 | +8.0 % |
| &nbsp;&nbsp;— gate/up (`w13`) | 177.77 | 18.47 % | 42 | | |
| &nbsp;&nbsp;— down (`w2`) | 96.65 | 10.04 % | 42 | | |
| Dense BF16 GEMM (cutlass / nvjet — the unquantized half) | 167.39 | 17.39 % | 457 | 16.2 | +7.3 % |
| **NCCL collectives** (102 all-reduce + 3 all-gather) | **139.27** | **14.47 %** | 105 | 16.5 | −12.3 % |
| Hyper-connection mixing (`mhc_*`, `hc_prenorm`) | 115.40 | 11.99 % | 275 | 11.7 | +2.5 % |
| MLA attention (`mla_decode_partial` + `reduce`) | 79.83 | 8.29 % | 59 | 8.2 | +1.1 % |
| KDA linear attention (triton chunked scans) | 77.86 | 8.09 % | 479 | 7.5 | +7.9 % |
| MoE `had_in` / `glu_had_in` | 53.62 | 5.57 % | 84 | 6.1 | −8.7 % |
| norm / elementwise / copy | 37.49 | 3.90 % | 1,040 | 3.7 | +5.4 % |
| DSA indexer (`fp8_mqa_logits`, `topKPerRowPrefill`) | 5.10 | 0.53 % | 77 | 0.6 | −12 % |
| memcpy / memset | 1.88 | 0.20 % | 94 | — | not modelled |
| MoE align / route | 1.87 | 0.19 % | 126 | 0.2 | −5 % |
| **`_zero_kv_blocks`** | **0.86** | **0.09 %** | 1 | 1.3 | **−93 %** |
| sampling / spec bookkeeping | 0.09 | 0.01 % | 24 | — | not modelled |
| **`exl3_moe_combine` / `build_inv`** | **—** | **0.00 %** | 0 | 1.5 | **−100 %** |
| **CPU gap (GPU idle)** | **8.16** | **0.85 %** | — | — | not modelled |
| **wall** | **962.55** | **100 %** | | | |

**16.2 ms of that (1.7 %) is the DFlash2 drafter**, which runs on prefill chunks too — dense GEMM
11.75 ms, collectives 2.56, the rest 1.90. The re-derived prefill table had no draft row at all.

Two rows are worth reading twice. **`exl3_moe_combine` does not exist as a kernel in this build** —
it is fused into the down-projection epilogue, so the 1.5 % attributed to it is not there and neither
is the target. And **`_zero_kv_blocks` is 0.86 ms, not 14.7** `[retracted]`: the model-free
reconstruction in §5.6 reproduced the *geometry* of a call correctly and then that geometry was
priced against a bigger pool and a different page mix than the production configuration actually
runs. The class is 16× smaller than published and the item was already closed on other grounds;
it is now closed on its own size ([11](11-open-issues.md) §2.13).

**The DSA indexer is 0.53 % of prefill**, a third independent confirmation that a
device-capability-120 persistent or filtered top-k would buy nothing here.

**A near-empty chunk is cheap; a small one is not.** Crossing a chunk boundary costs only the tokens
(2,041 → 2,058 adds 45 ms; 6,101 → 6,173 adds 45 ms), but a 128-token chunk costs **403 ms** —
105 collectives (205 ms) plus a nearly complete MoE weight stream (104 ms), because 128 tokens ×
top-8 already touches every expert `[measured-here]`.

### 5.3 Decode, measured at C1 (verify batch M=8) and C8 (M=64)

C1: **wall 94.65 ms · busy 89.2 ms · occupancy 94.2 %**, 93 steps × 3 ranks.
C8: **wall 216.52 ms · busy 212.2 ms · occupancy 98.0 %**, 63 steps × 3 ranks. `[measured-here]`

| class | **C1 ms** | **C1 %** | target | draft | **C8 ms** | **C8 %** |
|---|---|---|---|---|---|---|
| **Dense BF16 GEMM** | **42.90** | **45.33 %** | 34.58 | **8.32** | 45.61 | 21.07 % |
| **MoE trellis GEMM** | 28.11 | 29.69 % | 28.11 | — | **111.71** | **51.59 %** |
| &nbsp;&nbsp;— gate/up · down | 18.64 · 9.47 | 19.69 · 10.00 % | | | 72.73 · 38.98 | 33.59 · 18.00 % |
| **NCCL collectives** | 14.64 | 15.47 % | 12.89 | **1.75** | 25.26 | 11.67 % |
| **CPU gap (GPU idle)** | **5.45** | **5.76 %** | — | — | 4.32 | 2.00 % |
| KDA linear attention | 1.61 | 1.70 % | 1.61 | 0.00 | 17.32 | 8.00 % |
| hyper-connection mixing | 2.08 | 2.20 % | 2.08 | — | 3.19 | 1.48 % |
| MoE hadamard | 0.86 | 0.91 % | 0.86 | — | 3.88 | 1.79 % |
| MLA attention | 0.64 | 0.68 % | 0.50 | 0.14 | 3.05 | 1.41 % |
| norm / elementwise / copy | 1.79 | 1.89 % | 1.31 | 0.48 | 2.94 | 1.36 % |
| MoE align / route · DSA indexer | 0.97 | 1.02 % | 0.97 | — | 1.87 | 0.86 % |
| sampling / memcpy / `_zero_kv_blocks` | 0.12 | 0.12 % | 0.04 | 0.08 | 0.25 | 0.11 % |
| **step wall** | **94.65** | | | | **216.52** | |

**At decode the unquantized half of the checkpoint is the single largest item.** It gets *less*
efficient per rank as ranks are added, because each rank's shard of a BF16 matrix shrinks, and
nothing in the EXL3 kernel library touches it. §5.8 is about what that is worth.

**The k=7 drafter costs 10.78 ms (11.4 %) at C1 and 14.09 ms (6.5 %) at C8**, including its own
1.75 / 3.32 ms of collectives. The reconciliation put it at 18.5 ms and **19.5 %** of a C1 step: a
1.7× overestimate, and the largest single correction the trace made `[retracted]`. It was an artefact
of segmenting the step by "the span of the MoE GEMM calls is the target"; the annotation boundary is
exact and the heuristic was not.

**The C8 split, previously `[not tested]`, is here.** The MoE stage is **51.6 %** of a C8 step — the
older prose said "more than half", which was right, beside a "~130 ms" figure that is 16 % high.

**Why the columns sum past 100 %** (104.8 % at C1, 101.3 % at C8, 100.1 % at prefill): a second CUDA
stream runs part of the dense path concurrently, 5.5 ms/step at C1 and 5.3 at C8, so kernel durations
sum past the busy union. Wall is always `busy(union) + gap`. That overlap is real work being hidden,
and it is the only compute/compute overlap on this stack — the comm side has none (§5.7).

### 5.3.1 A model-free MoE benchmark overstates the small-batch cost by 1.5–1.7×

Same kernel, per MoE layer, per rank `[measured-here]`:

| M | model-free `gemm_w13` (§5.4's own bench) | **in-engine, measured** | ratio |
|---|---|---|---|
| 8 (C1) | 662 µs | **443.8 µs** | **1.49× faster in the engine** |
| 64 (C8) | 2,975 µs | **1,731.7 µs** | **1.72× faster** |
| 1,792–2,048 (prefill) | 4,404 µs @2048 | **4,232.6 µs @1792** | equal |

The model-free bench routes uniform-random top-8, which **maximises** the number of distinct experts
a batch touches; real routing is clustered, so at small M far fewer experts and far less trellis are
read. At large M all 96 local experts are touched either way and the two agree. **Do not carry a
small-M MoE cost from a uniform-random bench** — and the same trap one level down caught a synthetic
MLA measurement elsewhere whose 200k-row pool fitted L2 and duly reported a bandwidth above the DRAM
ruler. Both errors flatter small inputs, for the same reason: the synthetic working set is more
cache-resident than production's.

### 5.4 The MoE trellis GEMM is at the roof, and the traffic lever is closed

The largest single prefill class runs at **81–96 % of the measured 225 GB/s ruler**, model-free, EP
arm, one rank, per MoE layer (`w13 = [E, 4096, 4096]` at 4 bit = 8.389 MB per expert; traffic =
`local_blocks × 8.389 MB`) `[measured-here]`:

*Convention: each row is an independent bench call at that M (tokens fed to one call), not additive
into a chunk total, and — shown below — this class does **not** scale linearly with M (L2 residency),
so no rescaled row is given. The M=2048 row was measured at M=2048; production prefill chunk is
1,792 tokens, 7×256 (§5.2).*

| M | block_m | `local_blocks` | distinct local experts | `gemm_w13` µs | GB/s if per-block | GB/s if per-expert | % of 225 GB/s |
|---|---|---|---|---|---|---|---|
| 8 | 16 | 17 | ≤17 | 662 | 215 | 215 | **96 %** |
| 64 | 16 | 77.5 | ≈67 | 2,975 | 219 | 189 | 84–97 % |
| 512 | 16 | 121.5 | 96 | 3,831 | 266 | 210 | 93 – >100 % |
| 2048 | 64 | 112 | 96 | 4,404 | 213 | 183 | **81–95 %** |

The kernel is not slow; it is taking most of what the memory can deliver. The same figure now
exists **in the engine** rather than model-free, from the trace: per MoE layer per rank on a
1,792-token chunk, `gemm_w13` runs 4,232.6 µs against 805.3 MB of per-expert traffic = 190.3 GB/s =
**84.5 % of the ruler**, and `gemm_w2` 2,301.2 µs against 402.6 MB = 175.0 = **77.7 %**
`[measured-here]`. The `GB/s if per-block` column above is kept only because the arm below was built
to test it; it is **not a real alternative** and should not be quoted as a ceiling `[retracted]`.

That left exactly one lever — **read less**. At M=2048 the launch runs 112 blocks over 96 local experts and at M=512 it
runs 121.5, so *if* a block re-reads its expert's trellis, 17–27 % of that traffic is avoidable and
an expert-stationary schedule would be worth about 3.7 % of prefill wall.

**It does not re-read.** The kernel author added a bench for exactly this question
(`bench_moe_expert_reread.py`, `9b17ea9`); we ran it unmodified on GB10 against the production build,
three times, with the ruler in the same process — 96 experts, `block_m=16`, N = 1…4 blocks per expert
`[measured-here]`:

*Convention: N sweeps blocks read per expert at a fixed 96-expert configuration; it has no M or
chunk-length parameter, so the 1,792-vs-2,048-token chunk convention does not apply to this table.*

| N | blocks | rows | µs (run 1 / 2 / 3) | GB/s per expert | GB/s per block |
|---|---|---|---|---|---|
| 1 | 96 | 1,536 | 3,696 / 3,621 / 3,660 | 218–222 | 218–222 |
| 2 | 192 | 3,072 | 4,035 / 4,066 / 4,064 | 198–200 | 396–399 |
| 3 | 288 | 4,608 | 4,370 / 4,361 / 4,403 | 183–185 | 549–554 |
| 4 | 384 | 6,144 | 5,469 / 5,464 / 5,480 | 147 | 588–590 |

Doubling the block count costs **1.11×**, not 2×; quadrupling it costs 1.5×, not 4×. The trellis
**stays resident** across an expert's blocks — `moe_align_block_size` keeps an expert's blocks
adjacent and 8.4 MB per expert fits a 24 MiB L2. The 14–27 % traffic win the arm was built to find
**does not exist at this configuration, and the item is closed** `[measured-here]`. The reason the
per-block GB/s column climbs past the ruler is that an L2-resident re-read is cheap, not that DRAM is
being exceeded; the load-bearing evidence is the µs column being nearly independent of block count.
The same bench on the author's own 188-SM card gives 1.16×, so this is structural rather than a
property of 48 SMs.

That leaves `exl3_moe_had_in` as the only sub-roofline kernel in the MoE stage — 83–129 GB/s, **37–57 %
of the ruler**, 4.0 % of a prefill chunk. Upstream has since taken it (`a47da6e`, a 64-bit division
removed in favour of deriving the index from the grid): **−10 to −18 % on that kernel**, roofline
57 % → 63 %, which is worth ~0.2–0.3 % of prefill wall on this stack — real, and not worth an image
rebuild on its own `[reported]`. It goes into the next build bundle.

### 5.5 Hyper-connection mixing: 11.7 % of prefill, and one honest lever

`hc_mult = 4`, so the residual stream is carried in **four copies** — `(M, 4, 4096)` bf16, 32,768
bytes per token. Twice per layer (before attention and before the FFN) a fused post+pre block runs
**three kernels**: a post mapping that writes the new residual, a tf32 GEMM against a `(24, 16384)`
projection, and a fused sigmoid/softmax/Sinkhorn + RMSNorm that produces the layer input. Six
launches per layer, 275 per prefill chunk, 132 ms `[measured-here]`.

The analytic traffic is ~148,600 bytes per token, or **27.2 GB per chunk**; at 132 ms that is
**205.8 GB/s = 86–91 % of the ruler band**. Arithmetic intensity is 5.3 FLOP/byte against this
machine's balance point of ~405, so the class is memory-bound by a factor of 76 and the tf32 GEMM
inside it runs at 4.8 % of peak because there is nothing else for it to be. A model-free reproduction
in the same image landed within **0.8 %** of the trace ([`bench/mhc_bench.py`](../bench/mhc_bench.py)).

Three things this rules out, each of which looked plausible:

- **Not launch-bound.** With CUDA graphs on and off the 90-call sequence differs by **0.03 %** at
  M=2048 (it differs by 76 % at M=8, which is why decode runs in a graph) `[measured-here]`.
- **Not a bad TileLang configuration.** Sweeping the two tunables the call site leaves at their
  defaults is worth −4.9 % on the first kernel and −3.5 % on the third — **0.4 % of prefill**
  together. On the third kernel `threads > 96` does not compile at all (`no available layout`), so
  the production value is the widest one that builds, not a lazy default `[measured-here]`.
- **Not escapable via the torch fallback.** The reference implementation exists but is unreachable on
  CUDA — `forward_cuda` calls TileLang unconditionally, `enabled()` is overridden to `True`, and the
  model bypasses the `CustomOp` wrapper entirely by importing the TileLang functions directly. It is
  also 5.3–15.5× slower, adding ~900 ms to a chunk `[measured-here]`. It is a reference, not an
  escape hatch.

The one real lever is to **read the residual once less**: the second kernel's entire traffic is one
re-read of what the first kernel just wrote. Fusing them would save 30.5 % of that pair's bytes —
**−28 to −30 ms per chunk, −2.5 to −2.7 % of prefill**. The existing fused kernel does not do this
job: it is selected only at ≤16 tokens, it grids per token per n-tile, and forced at large M it is
**+32 % worse** at M=2048 and worse at every M we tried, including M=8 `[measured-here]`. The lever
needs a new large-M kernel that tiles over `block_m`, and it is vLLM/TileLang work, not EXL3 work.
An earlier estimate of ours put this at −3.6 %; measured, the ceiling is −2.7 %, so **that estimate
was 30 % optimistic** `[retracted]`.

### 5.5.1 That kernel was then written, and it reached 40 % of its own ceiling

A Triton kernel that grids over token tiles rather than over tokens — so the `(24, 16384)` projection
is shared across a tile and the post mapping is reduced against `fn` **while the row is still in
registers**, never landing in HBM for the second kernel to read back — was written, compiled and
measured model-free `[measured-here]`. Throwaway container, engine idle and untouched, disjoint
cpuset, both rulers measured inside each run ([09](09-measurement-protocol.md) §10), two independent
runs, median of 21 repetitions, CUDA graphs on, 90 calls = 45 layers × 2.

Winning configuration in both runs: `BLOCK_M=16, BLOCK_H=64, SPLIT_H=2, warps=4, stages=2`.

*Convention: per-call, per-layer cost at the fixed micro-benchmark M=2048 tokens — measured at
M=2048; production prefill chunk is 1,792 tokens, 7×256 (§5.2). This class is memory-bound and
near-linear in M (§5.5), so the 1,792-token chunk totals are recomputed in the next table.*

| M = 2048 | run 1 µs/call | GB/s | run 2 µs/call | GB/s |
|---|---|---|---|---|
| k1 `mhc_post` | 670.7 | 225.4 | 674.3 | 224.2 |
| k2 `hc_prenorm` | 311.6 | 221.1 | 313.9 | 219.4 |
| k3 `pre_big_fuse` | 431.3 | 195.4 | 434.7 | 193.8 |
| **kF, k1+k2 fused** | **815.7** | **187.7** | **815.2** | **187.9** |

*Convention: totals for 90 calls (45 layers × 2) at the fixed micro-benchmark M=2048 — measured at
M=2048, not the 1,792-token production chunk. The added column scales the mean of run 1/run 2 by
1792/2048 = 0.875 `[estimate]`, since this class is memory-bound and near-linear in M (§5.5); the
relative-savings columns (traffic %, time %) are unchanged because both arms scale together.*

| route, 90 calls | traffic | run 1 | run 2 | time | ≈1,792-tok chunk `[estimate]` |
|---|---|---|---|---|---|
| k1 + k2 (today) | 220.05 MB | 86.293 ms | 86.783 ms | — | 75.72 ms |
| **kF (fused)** | 153.14 MB (**−30.4 %**) | **73.416 ms** | **73.365 ms** | **−14.9 / −15.5 %** | 64.22 ms |
| R3 (production, 3 kernels) | 304.31 MB | 123.914 ms | 124.960 ms | — | 108.88 ms |
| **RF (fused, 2 kernels)** | 237.61 MB (−21.9 %) | **112.729 ms** | **112.649 ms** | **−9.0 / −9.9 %** | 98.60 ms |

**On the prefill wall that is 11.2–12.3 ms of a 1,109 ms chunk: −1.01 to −1.11 %** — both figures
assumed the old ~2,032–2,048-token chunk (§5.2). Scaled ×0.875 for the actual 1,792-token production
chunk, that is **9.8–10.8 ms of a 970 ms chunk**, still **−1.01 to −1.11 %** because the savings and
the chunk scale together `[estimate]`; against the real measured 962.55 ms prefill wall (§5.2) it is
−1.02 to −1.12 %, materially the same conclusion. A second, independent route to the same figure
agrees: this class is 11.7 % of a chunk (§5.2) and the route gets 9.4 % cheaper, which is −1.10 %.
**The target was −2.1 to −2.8 %; the kernel delivered a little under half of it.**

**Why it stopped there, and it is not the tiling.** The fused kernel runs at 187.7 GB/s where the
k1+k2 route it replaces runs at 229.5 (220.05 MB in 958.8 µs). Traffic ratio 0.696 ÷ bandwidth ratio
0.818 = 0.851, which is the measured −14.9 % exactly, so the traffic model is right and the loss is
entirely bytes-per-second. Had the fused kernel reached k1's own band it would be −29.1 % on the pair
and −19.3 % on the route — **−2.2 % of prefill**, the bottom of the original estimate. A 33-configuration
sweep across two M values could not improve the winner, so this is not occupancy or tile shape. The
one concrete untried arm is the `tl.dot` operand path: `fn` is transposed inside the kernel
(`tl.trans` on a 32×64 fp32 block, staged through shared memory), and pre-transposing it once on the
host would remove that. Half an hour of work, **not measured, not claimed** `[not tested]`.

**Fusing loses below M ≈ 1024, and the reason is the L2.** `residual_cur` at M=512 is 16.8 MB and
fits the 24 MiB L2, so k2's "re-read" was never going to DRAM and there is nothing to delete — the
fusion pays its own cost for no saving `[measured-here]`:

*Convention: each row is measured at its own M. The production prefill chunk is 1,792 tokens, 7×256
(§5.2) — between the 512 and 2,048 rows; the 1,792 row is not a fresh measurement, it scales the
M=2048 row (run 1) by 1792/2048 = 0.875 `[estimate]`.*

| M | k1+k2, 90 calls | kF | |
|---|---|---|---|
| 8 | 0.633 ms | 3.541 ms | +459 % |
| 64 | 0.803 ms | 4.458 ms | +455 % |
| 512 | 13.859 ms | 19.090 ms | **+37.7 %** |
| 1,792 `[estimate]` | 75.51 ms | 64.24 ms | **−14.9 %** |
| 2048 | 86.293 ms | 73.416 ms | **−14.9 %** |
| 4096 | 175.900 ms | 142.051 ms | **−19.2 %** |

That is the measured justification for a size threshold, and it puts it at **M ≥ 1024**, not the 256
the module shipped with. The M=4096 row also says the gain grows with the chunk: under
`--max-num-batched-tokens 4096` the route figure is −13.2 %, which extrapolates to about −1.5 % of
prefill — an `[estimate]`, since a 4,096-token chunk's own profile was never taken.

**Correctness.** `residual_cur` is **bit-identical** at M = 8, 512, 2048 and 4096 (0 differing
elements out of 33.5M and 67.1M); at M=64, 7 elements of 1,048,576 differ by one bf16 ulp (1.9e-6),
which is fp32 FMA contraction near zero. `layer_input` differs by **at most one bf16 ulp** on 5.1 %
of elements (max absolute difference exactly 3.125e-2 = 2⁻⁵, one ulp in the [4, 8) binade);
`post_mix` and `comb_mix` by 0.21 % and 0.35 % relative. Inside bf16 rounding, and **not zero**.

**What it cost, and why it is not in production on its own.** Adopting it puts **Triton JIT into the
serving process** — a `/root/.triton` mount and an explicit warm-up before graph capture, or the first
large-M call compiles inside the capture — which is a new failure surface for −1 %. The config surface
is a cliff, not a slope: the winner reads 187.8 GB/s and its neighbours 79.4 and 44.5, and **the
default the module shipped with was one of the bad ones** (79.4 GB/s), so every hardware or shape
change needs the sweep run again. Set against the levers beside it — the collective at 16.5 % and the
MoE GEMM at 26.4 % — a −1 % change does not earn its own boot and its own A/B cycle. **Written,
measured, not adopted standalone; it rides the next image bundle together with `had_in`**, where one
boot measures both ([11](11-open-issues.md) §2.16, §2.19).

**One lesson is worth more than the kernel.** A GPU-free ahead-of-time compile check reported 18 of 18
configurations compiling and every one of them under the 99 KB shared-memory limit. At real launch
**6 of the 18 failed with `OutOfResources`**, because `metadata.shared` from the AOT path
under-reports what a launch actually needs — 36,864 bytes reported against 106,496 required, in one
case 40,960 against 147,456 `[measured-here]`. **A compile check answers "does it build", never "does
it run".** No harm was done, because the bench caught them and carried on, but the static test had
been written up as a pass.

### 5.6 KV block zeroing: at the memset roof, and the gain is not available

> **Corrected by the trace, and the correction is large.** This section was written against
> `_zero_kv_blocks_kernel` at **14.7 ms per prefill chunk (1.3 %)**. Measured on the live production
> configuration it is **0.857 ms, 0.09 %** — one call per chunk, at a 4.72M-token pool with 256-token
> blocks. A 16× overestimate `[retracted]`. The mechanism below is still correct and is why the item
> was closed in the first place; only the size was wrong, and it was wrong in the direction that
> made a non-target look like a target. §5.2, [`../results/profile/measured-prod7.md`](../results/profile/measured-prod7.md) §4.

`_zero_kv_blocks_kernel` was costed at **14.7 ms in a single call per prefill chunk**, 1.3 %. vLLM zeroes
newly allocated blocks when the cache has Mamba layers **or** mixed precision, and this stack has
both: 34 KDA/Mamba layers, and a main cache at fp8 with the DFlash draft's own cache at bf16.

Reconstructing the live grid model-free — `[128, 720, 8]`, 32 KB pages —
reproduces the trace within 2.5 % and shows the kernel running at **100 % of the memset ruler**
(198 GB/s) `[measured-here]`. There is nothing to win in the kernel. What it is doing is zeroing
**2.4–2.9 GB per chunk** where the new tokens' real KV is about 3.4 MB.

The obvious lever — make the cache uniform-precision by moving the draft to fp8, then skip the
zeroing — **is not available on this model**, and finding out why is the useful part. The zeroer does
skip Mamba layers, which is what made "then the only remaining reason is mixed precision" look right.
But in this model's hybrid layout **one tensor is co-owned by an MLA layer and one Mamba layer from
each group**, so a block handed from the Mamba group to the attention group carries 1.7 MB of raw SSM
state. Measured per block: MLA pages co-owned with Mamba/KDA are **85.5 %** of the bytes being
zeroed, the indexer tail 5.5 %, the draft's sliding window 9.0 %. The Mamba half of vLLM's condition
is the binding one and it is independent of precision. The safe remainder — indexer plus draft, if
the cache were uniform — is worth **0.19 % of prefill**, which is not worth writing a partial mode
for `[measured-here]`. Item closed; the ceiling is recorded so nobody prices it again.

### 5.7 The collectives are 100 % exposed, and part of them is arrival skew

Two numbers the trace settles that no model-free bench could `[measured-here]`.

**Comm/compute overlap is zero.** All-reduce and compute share one CUDA stream: measured overlap is
**0.00 ms/step at prefill**, 0.014 at C1, 0.012 at C8. Every microsecond of NCCL is on the critical
path. Prefill runs 102 all-reduce (median 903 µs, p75 1,401, max 7,520) plus 3 all-gather per chunk;
C1 and C8 run the same counts at medians of 71 and 123 µs. Set that beside [06](06-nccl-mesh.md) §9:
the bandwidth lever is nearly spent at the PCIe wall, and the **overlap lever has never been touched**.
It is the largest untried item on the fabric side and it is vLLM's, not the plugin's
([11](11-open-issues.md) §2.17).

**Part of the collective class is not transport at all.** Step wall is identical across ranks
(963.3 / 962.6 / 961.7 ms at prefill), but time spent *inside* collectives is not:

| window | rank A | rank B | rank C | spread |
|---|---|---|---|---|
| prefill (1,792 tok) | 136.02 ms | **145.08 ms** | 136.69 ms | 6.5 % |
| decode C1 | 15.04 | 15.44 | **13.45** | 13.6 % |
| decode C8 | 24.61 | **26.96** | 24.20 | 10.9 % |

The rank that waits longest at prefill is the one doing **5.9 ms less MoE GEMM**, 1.6 ms less dense
GEMM and 1.3 ms less MoE hadamard — it finishes early and blocks at the barrier. So true wire time is
the **minimum** over ranks, ≈136.0 ms, and **6.5 % of the collective class (≈0.9 % of a step) is
expert-parallel load imbalance**. That is closed by expert placement, not by the plugin, the kernel
or a cable.

### 5.8 The C1 idle budget: 3.75 %, not 5.8 %, and "CPU gap" was the wrong name

The C1 row that reads **5.45 ms of GPU idle (5.76 %)** invites one conclusion — turn CUDA graphs on —
and the trace does not support it. Two corrections, in order of size.

**First, subtract the instrument.** GPU busy union is 89.17 ms against a **profiler-off wall of
92.64 ms**, so ~2.0 ms of that "idle" is CUPTI itself: 1.97 ms over 1,873 kernel boundaries ≈
**1.05 µs per boundary**, which is the known per-kernel cost of the tracer. At C8 it is 1.50 ms over
1,800 ≈ 0.83 µs. The real budget:

| window | profiled wall | busy (union) | profiled idle | unprofiled wall | **corrected idle** |
|---|---|---|---|---|---|
| decode C1 | 94.611 | 89.167 | 5.444 | **92.64** | **3.477 ms — 3.75 %** |
| decode C8 | 216.448 | 212.127 | 4.321 | **214.95** | **2.828 ms — 1.31 %** |

The two walls are `[measured-here]`; the split between them is `[estimate]`, and it assumes CUPTI
does not slow the kernels themselves. Separating those cleanly needs the profiler off and the same
window re-opened, which costs a boot and has not been spent.

**Second, the label.** Matching every gap to the launch of the kernel on its right — via
`correlation`, and **including `cuda_driver` launches**, since triton kernels go out that way and
filtering them mislabels the step's two largest gaps as device waits:

| cause | C1 ms/step | % of idle | gaps/step | mean |
|---|---|---|---|---|
| **device-side dispatch bubble** (target 3.328 + draft 0.858) | **4.186** | **76.9 %** | 1,830 | 2.3 µs |
| host-bound launch gap, mostly the step head's `prepare_inputs` | 0.838 | 15.4 % | 36.5 | 23 µs |
| **blocking sync / pageable D2H on the critical path** | 0.281 | 5.2 % | 1.0 | 281 µs |
| draft orchestration · NCCL launch · rest | 0.140 | 2.6 % | 4.1 | — |
| **TOTAL** | **5.444** | 100 % | 1,873 | |

So **77 % of the idle is per-kernel dispatch**: the step launches **2,332 kernels** and pays ~2.3 µs
at 80 % of their boundaries. Only 18 % is the host, and the host is not behind — it runs **3.9 ms
ahead** of the GPU at C1 and 24.8 ms ahead at C8, which is why C8's idle is smaller in both absolute
and relative terms. There are exactly **two gaps per step of ≥ 0.1 ms** (one at C8), together
0.48 ms; the rest is 1,871 micro-holes. Half of the idle sits in front of glue kernels — norm,
elementwise and copy account for **50.9 %** of it over 717 boundaries — and this build runs
`compilation mode NONE`, so torch.compile is off entirely and nothing is fused.

By phase, with the CUPTI correction applied in the last column:

| phase | phase ms | idle ms | occupancy | **corrected idle** |
|---|---|---|---|---|
| **P1 head: scheduler handoff + `prepare_inputs`** | 1.075 | 0.949 | **11.7 %** | 0.856 |
| P2 target forward body | 81.526 | 3.261 | 96.0 % | 1.754 |
| P3 logits + sample + accept | 2.959 | 0.055 | 98.1 % | 0.033 |
| **P4 draft loop (DFlash2 k=7)** | 9.051 | 1.178 | **87.0 %** | 0.834 |
| **TOTAL** | **94.611** | **5.444** | 94.25 % | **3.477** |

**What graph capture is actually worth: 1.4–1.9 ms/step, +1.5–2.1 % of C1**, and 0.5–0.7 % at C8. It
removes part of P2's 1,435 boundaries and, with its own capture, part of P4's; it removes **neither**
P1 — `prepare_inputs` is host code producing the graph's inputs, outside any captured region — nor
the blocking sync, because a graph replays kernels and does not remove a synchronise. **An earlier
claim of "+6 % single-stream" from graph coverage is retracted**: the whole idle budget is 3.75 % and
graphs are at most two-thirds of it `[retracted]`. This also explains an otherwise awkward
observation: the previous production configuration *did* capture graphs (bf16 draft KV →
FlashAttention, `Graph capturing finished` in the boot log) and read the same **57 tok/s**
single-stream as this one, which does not (fp8 draft KV → FlashInfer → `cudagraph_mode=NONE`). A
1.5–2 % difference is inside the noise of a tok/s reading. Graphs never brought back a lost 5.45 ms,
because there was never a 5.45 ms to bring back.

Why graphs are off is named in the engine's own log rather than inferred:

```
CUDAGraphMode.FULL_AND_PIECEWISE is not supported with spec-decode for attention backend
FlashInferBackend (support: AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE);
setting cudagraph_mode=NONE
```

FULL capture with spec-decode needs a backend declaring at least `UNIFORM_BATCH`; FlashInfer declares
`UNIFORM_SINGLE_TOKEN_DECODE`, and the drafter is on FlashInfer *because* its cache is fp8. The
ordered list of what would close the remaining budget — glue-kernel fusion first, then
`prepare_inputs`, then the pageable D2H, then graph coverage — is in
[11](11-open-issues.md) §2.23, with a realistic total of **+1.1…1.6 % single-stream**.

**And the drafter's collectives, measured**: 11 all-reduce + 3 all-gather per step at **133 µs each**
against the target's 146 µs. They are latency-bound, not size-bound, so batching is unavailable (they
are eleven sequentially dependent layers) and replicating the drafter costs more than it saves (its
dense GEMM is already 8.32 ms at C1). Overlap is the only lever, and overlap is 0.014 ms.

---

**Correction, 6 September — the declaration is wrong, and the kernel is not the limit** `[measured-here]`.
FlashInfer's `get_cudagraph_support()` decides the level by `num_qo_heads % num_kv_heads == 0`, and it
takes `num_qo_heads` from the **target** model (22 per rank at TP=3) while `num_kv_heads` belongs to the
group being asked about — the **drafter's** (3 per rank). 22 % 3 = 1, so it declares
`UNIFORM_SINGLE_TOKEN_DECODE`. The builder that actually runs takes its head count from the group's own
layers (12 per rank; 12 % 3 = 0), passes the same test, and selects the XQA decode kernel —
`decode_backend=xqa` sits three lines above the warning in the same boot log. The identical image at
TP=2 (32 % 4 = 0) captures graphs: `Graph capturing finished in 11 secs, took 1.10 GiB`. So the
paragraph above is right about *what* the engine does and wrong about *why*: FULL capture is not
impossible on this drafter, it is mis-declared. Filed upstream as [vllm#55581](https://github.com/vllm-project/vllm/issues/55581); the arithmetic, the price of
turning graphs back on (a graph pool of 1.1–2.6 GiB per rank, −2 to −5 % of KV, for a ceiling of
+1.2–2.3 % single-stream) and the A/B that would settle it are in [11](11-open-issues.md) §2.29. We
leave them off.

## 6. Ranked targets, and who owns them

Per steady **1,792-token** prefill chunk (962.55 ms measured), and per decode step at C1 (94.65 ms)
and C8 (216.52 ms), against the **measured** rulers of §4.1. Every share is now read out of a trace
rather than re-derived `[measured-here]`:

| # | target | prefill | C1 | C8 | achieved vs ruler | realistic gain | owner |
|---|---|---|---|---|---|---|---|
| 1 | **Dense BF16 GEMM — the unquantized half of the checkpoint** | 17.4 % | **45.3 %** | 21.1 % | 79 % of shape-matched achievable TFLOP/s | **CLOSED, and taken: +21.7 % per stream at TP=3, in production since 5 September evening** (§2.3, [13](13-full-scope-checkpoint.md)) | was **checkpoint scope + the vLLM model file**, not the kernels |
| 2 | MoE trellis GEMM, large M | 28.5 % | 29.7 % | **51.6 %** | 78–85 % of 225 GB/s in the engine; duplicate-read lever **closed** (§5.4) | ~0 | cuda-exl3 — **closed** |
| 3 | NCCL — **overlap** with compute, currently exactly 0 (§5.7) | 14.5 % | 15.5 % | 11.7 % | ~20 GB/s bus against a ~30 GB/s per-node PCIe ceiling | bandwidth ≤ −2…4 %; **overlap untried, ceiling is the whole class** | vLLM (the fabric side is spent) |
| 4 | Hyper-connection mixing, 3 passes over a 4× residual | 12.0 % | 2.2 % | 1.5 % | 86–91 % of the ruler | ceiling −2.5…2.7 %; the kernel exists and delivers **−1.0…1.1 %** (§5.5.1) | vLLM / TileLang |
| 5 | MLA prefill (`mla_decode_partial` runs at prefill too) | 8.3 % | 0.7 % | 1.4 % | **86–89 % of achievable** — measured by the kernel author at our shapes, 2 GB pool `[reported]` | ~0, **closed** | cuda-exl3 — **closed** |
| 6 | KDA linear attention (triton) | 8.1 % | 1.7 % | 8.0 % | **not measured** | ? | vLLM |
| 7 | `exl3_moe_had_in` / `glu_had_in` | 5.6 % | 0.9 % | 1.8 % | 37–57 % → 63 % after `a47da6e`; the remainder is half-ALU work, ≤2 % of prefill and unreachable (`62f53e6`) `[reported]` | −0.2…0.3 % | cuda-exl3 — **closed** |
| 8 | CUDA-graph coverage of the 8-token verify batch | 0.9 % | **3.75 % real idle** | 1.3 % | — | **+1.5…2.1 % of C1**, not the +6 % once claimed (§5.8) | vLLM |
| 9 | norm / elementwise / copy — the glue in front of half the idle | 3.9 % | 1.9 % | 1.4 % | — | −0.6…1.0 ms/step at C1 via fusion; `compilation mode NONE` today | vLLM |
| 10 | DSA indexer | 0.53 % | 0.37 % | 0.26 % | — | ~0, **closed** (third confirmation) | — |
| 11 | `_zero_kv_blocks` | **0.09 %** | ~0 | ~0 | 100 % of the memset ruler, and 16× smaller than published (§5.6) | ~0, **closed** | vLLM |
| 12 | `exl3_moe_combine` | **0 %** | 0 | 0 | — | **the kernel does not exist in this build** | — |

Four things a reader should take from that table, and the first three have changed since it was a
reconciliation. **Row 1 has since been taken**, so read the table as the ranking that *produced*
production 9 rather than as today's; the shares have not been re-measured on it (§5).

**The largest item is not a kernel and not the fabric — it is the checkpoint's scope, and two lines
in a model file.** Dense BF16 GEMM is 45 % of a C1 step because `scope: glm53_routed_experts_only`
leaves attention, the shared experts and `lm_head` unquantized, so at M=8 that stage streams 16-bit
weights beside a 4-bit routed half. The arithmetic, which is the kernel author's on our numbers: the
stage is weight-bandwidth-bound at M=8, so 4 bpw instead of 16 is ~4× less traffic — 42.9 ms →
~11 ms, **~32 ms off a 94.65 ms step, roughly +34 % single-stream** `[estimate]`. Everything in the
`cuda-exl3` column of this table comes to about 5 %.

**That is no longer an estimate, and it is no longer open.** A full-scope checkpoint was measured at
TP=2 on 5 September at +24.3 % per stream, and **promoted to production the same evening at TP=3:
+21.7 % per stream, +12.5 % at C8, KV pool +10.0 %, MMLU inside the bar** `[measured-here]` (§2.3).
Against the estimate, **17.8 ms of the ~32 ms arrived** — 65 % at TP=2 and 55 % at TP=3 of a step
that was itself shorter — and the difference is the layers this particular checkpoint leaves in BF16
anyway. **The +34 % was not refuted; it was an upper bound, and it is now bounded from both sides.**

Three things had to be fixed first, and none is about quantization: the model class declares no
`packed_modules_mapping`; the vLLM `glm5next` model file pins the attention stack to BF16 in two
places that between them lock **72.8 %** of the dense traffic; and the KDA block is factorised
differently in the checkpoint from what the reader expects. **The scoped checkpoint was not a quality
choice; it was the only scope that could load** ([13](13-full-scope-checkpoint.md),
[11](11-open-issues.md) §2.22).

**The `cuda-exl3` kernel library is closed as a target on this stack.** Rows 2, 5, 7, 10, 11 and 12
are at the roofline, bounded below what a rebuild is worth, or not kernels at all. Two of them closed
in the last day: MLA prefill, which this repository listed as "efficiency not measured" for a week,
came back at 86–89 % of achievable from the author's own bench; and `exl3_moe_combine`, ranked #10,
turned out never to have existed in this build.

**Two published targets were fiction and both were ours.** `exl3_moe_combine` at 1.5 % and
`_zero_kv_blocks` at 1.3 % were re-derived rather than measured, and they are 0 % and 0.09 %. That is
the whole argument for spending one boot on a launcher flag rather than a week on a reconciliation.

**There is still no single-digit-percent win left in prefill.** The honest list is 2–4 % from the
fabric's bandwidth (and an untried overlap lever worth up to 14.5 %), 2.5–2.7 % from a
hyper-connection kernel at half its ceiling, 3.1 % from somebody shipping Blackwell-class dense GEMM
for sm_121, and a scattering of fractions. The one **config** lever larger than all of them is the
batched-token budget — and the trace found a second one beside it: the scheduler issues **1,792 of
the 2,048** budgeted tokens per chunk, so 12.5 % of the budget is unused before the budget is even
argued about (§5.2, [11](11-open-issues.md) §2.5) `[measured-here]`.

---

## 7. What we did not measure

- **Anything at max reasoning effort** `[not tested]`. See [09](09-measurement-protocol.md) §7.
- **MMLU at TP=3** — run since 5 September on the production checkpoint: **86.47 ±0.74** on the 1,995-question sample (§1, §3); the earlier 86.4 ±0.7 figure was a TP=2 reading. The **full** 14,042-question MMLU at TP=3 is still not run (deferred, [results/gates/quality-battery-production-12.md](../results/gates/quality-battery-production-12.md)).
- ~~**IFEval, GSM8K, tool-eval-bench**~~ **Done on 6–7 September** `[measured-here]` — §3.1 and
  [`../results/gates/quality-battery-production-12.md`](../results/gates/quality-battery-production-12.md).
  **Needle-in-a-haystack at 1M is no longer missing: 20/20 on 8 September**, effective context
  997,952 tokens, level with the sibling and 3.0 % quicker
  ([`../results/gates/quality-battery-production-13.md`](../results/gates/quality-battery-production-13.md) §8)
  `[measured-here]`. Still missing: the **full MMLU** and **ExtractBench Short**, which neither this
  stack nor the NVFP4 sibling's shipped build has ever run `[not tested]`.
- **Prefix caching.** With a 3,328-token attention block, our benchmark prompts never fill one, so
  the prefix-cache hit rate is 0 % throughout and the benchmark says nothing about it
  `[measured-here]`.
- **Long-context behaviour at scale.** The KV pool supports about 4.4 concurrent million-token
  requests and we never drove it past 13 % usage `[measured-here]`.
- ~~**Content types and mixed load on the production configuration.**~~ **Content types: done on
  production 12 on 7 September** `[measured-here]` — §1.2 and
  [`../results/speed/category-speeds-production-12.md`](../results/speed/category-speeds-production-12.md).
  The newer drafter revision was split by category on the same night, and prose does not move.
  **Mixed load is still unrun** `[not tested]`: `scripts/mixed-load-probe.py` has not gone since the
  fast-boot arm, and §1 carries that arm's figure. Also still absent on the category side is an
  **unspeculated three-node arm** — one boot without `--speculative-config`, about 25 minutes, which
  would turn §1.2's per-category return-on-speculation column from an estimate into a measurement.
- ~~**A torch-profiler run on the production configuration.**~~ **Done** `[measured-here]`. It cost one
  launcher flag and no boot of its own, it settled both rows it was supposed to settle — the NCCL band
  came in at the bottom of 14–17 %, and the C8 split is in §5.3 — and it deleted two targets that had
  been carried for a week. The remaining absence is smaller and named in §5.8: the CUPTI subtraction
  is an inference from two walls rather than a direct measurement, and separating it cleanly costs a
  boot nobody has spent.
- ~~**The MLA prefill kernel's efficiency.**~~ **Closed, and not by us** `[reported]`. It is 8.3 % of a
  prefill chunk and our trace has no selected-key count, so there was no denominator. The kernel's
  author measured it at our shapes — top-k 2,176, head_dim 512, fp8 cache, a 2 GB pool chosen
  specifically so it could not sit in L2 — and got **86–89 % of achievable** (1,299–1,345 GB/s
  gathered against a 1,518 GB/s ruler in the same run). It sits with the trellis GEMM: near the roof,
  no traffic to remove. Worth recording is that his *first* cut used a 200k-row pool that fitted the
  card's L2 and reported a bandwidth above the DRAM ruler — caught only because the number was
  impossible. Same class of error as §5.3.1, one level down.
  **Closed a second time, from the other end, on 6 September** `[reported]`. We finally measured the
  selection this kernel attends to — median 2,049 keys per query row, median adjacent-row overlap
  **0.9258**, i.e. about 152 keys turning over per row
  ([`../results/kernels/sm12-stack-patches-ab.md`](../results/kernels/sm12-stack-patches-ab.md) §8) —
  which placed this configuration roughly 76× away from the low-turnover arm it had been compared
  against rather than between his two arms. He built a third arm at that turnover and corrected his
  own conclusion in `5fd7299`: at 262K context the production-pattern arm is within **1.6 %** of the
  fully cache-resident arm (2,422.8 against 2,385.8 µs) where the independent arm needs 3,474 µs,
  because the live key set is the ~4,096-key residence window rather than the chunk footprint. **MLA
  prefill is compute-bound at production overlap and the "21–26 % overlap gap ≈ 2 % of a chunk" does
  not exist**; the only lever left is less work, not less traffic. Not reproduced on a 48-SM part yet
  — [HELP-WANTED](../HELP-WANTED.md) §8 `[not tested]`.
- **KDA linear attention's efficiency** `[not tested]`. Now the largest class on the list with no
  denominator: 8.1 % of prefill and 8.0 % of a C8 step, triton chunked scans, never measured against a
  ruler. It inherits the slot MLA just vacated.
- **`--max-num-batched-tokens 4096` on the current configuration** `[not tested]`. §6 gives the
  arithmetic for why it is the largest single prefill lever left, and [07](07-kv-and-draft-page.md)
  gives the KV price it was rejected on two configurations ago. Nobody has re-run it since.

---

## 8. What is next

[11 — Open issues](11-open-issues.md): what is unresolved, what we retracted, and what we never ran.
