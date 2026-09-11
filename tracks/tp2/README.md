# tracks/tp2 — two nodes, TP=2, expert parallelism off

At two ranks this stack is a **shorter** recipe rather than a cut-down one: all five shapes that do
not divide by three do divide by two, and each leaves every rank a whole number of 128-column
Hadamard blocks. Nothing needs padding, so the shape surgery in
[docs/03](../../docs/03-tp3-padding-and-sidecars.md) and the padded-load path in
[docs/13](../../docs/13-full-scope-checkpoint.md) §7 are not needed at all.

**The track page is [docs/15](../../docs/15-tp2-track.md)** — why it works, the exact changes to the
env file, the launcher, the patch tree and the autostart unit, every two-node arm we ran with its
date and settings, and the honest list of what we never ran here. This page is the directory.

Three nodes instead of two: [tracks/tp3](../tp3/) and the [README quick start](../../README.md).

---

## What is here

| File | What it is |
|---|---|
| [`env.tp2-full.example`](env.tp2-full.example) | **The production-candidate template.** `NNODES=2`, `TP_SIZE=2`, `ENABLE_EP=0`, no padding sidecar, `gpu-memory-utilization` **0.85** rather than the three-node 0.88, and four settings that are not optional at two ranks |
| [`patches/`](patches/) | The in-container patch tree — **eighteen** files against the three-node tree's twenty-three, because nothing here has to pad anything. Four of the eighteen are the three-node track's files used byte for byte; [`patches/vision/`](patches/vision/README.md) and [`patches/prefix-hit-and-kpool-tail/`](patches/prefix-hit-and-kpool-tail/README.md) are pointers with install commands rather than second copies. [`patches/README.md`](patches/README.md) is the inventory |
| [`harem-exl3-tp2.service`](harem-exl3-tp2.service) | The autostart unit. It is a unit **of its own** — you do not edit the three-node one. Installed, started, health-checked and stopped on both nodes on 6 September 2026 and again on 8 September, where it brought candidate D to `/health` 200 in **280 s**. Left `disabled`, because exactly one of the two units may be enabled |
| [`motor-onkosul-exl3-tp2.sh`](motor-onkosul-exl3-tp2.sh) | Its preflight — `FABRIC_PEERS` is **one** address per node rather than two; the ConnectX-7 check stays `4/4`, because it counts ports on the node, not peers |

**Derive each node's environment file from the template with `sed`, on that node.** Never copy a
finished env file between nodes ([envs/README.md](../../envs/README.md)).

## What is not here, and where it is

| | |
|---|---|
| The launcher | [`scripts/start-tp2full.sh`](../../scripts/start-tp2full.sh) — every launcher in this repository lives in `scripts/`, one per track, so the harness and the probes sit beside them. **The prelude does not**: it is [`patches/tp2full-prelude.sh`](patches/tp2full-prelude.sh), inside the tree, because the tree's file list *and the full text of the prelude* are the fast-load sidecar's identity ([docs/08](../../docs/08-fast-boot.md) §4). `scripts/tp2-prelude.sh` is the **candidate-B era** copy and is kept as history only; it does not carry the indexer workspace bound, the vision block or the two backports. Two copies of a file are a coin flip unless something checks, and that one had already drifted |
| The `tp`-agnostic patches | [`patches/tp3/`](../../patches/tp3/) — `patch-swblock-tp3.py`, `patch-kvdiag-tp3.py`, `patch-draftkv-tp3.py`, `patch-epfilter-tp3.py` and `patch-fastload-tp3.py` are all gated on their own environment knobs and are used unchanged at two ranks |
| The mesh plugin patches | [`patches/kernel/`](../../patches/kernel/) — with one cable per pair set `NCCL_MESH_LINKS_PER_PEER=1`, which makes `0005` a no-op; `0006` is worth measuring either way |
| The GB10 top-k overlay | [`patches/indexer-overlay/`](../../patches/indexer-overlay/) — **mandatory**, and the failure that stopped our very first TP=2 boot |
| Everything measured | [`results/`](../../results/README.md), [`bench/`](../../bench/) |

**Keep the tree in its own directory.** A patch directory's file list and the full text of its
prelude are hashed into the fast-load sidecar's identity
([docs/08](../../docs/08-fast-boot.md) §4), and adding one file to a tree — even a file that is never
called — refuses the next boot on every node. That has happened to us twice, and once it was the
TP=2 patch dropped into the TP=3 tree that did it
([docs/13](../../docs/13-full-scope-checkpoint.md) §6.4).

---

## Four things that are mandatory here for reasons that have nothing to do with rank count

| | |
|---|---|
| `HAREM_DISABLE_PERSISTENT_TOPK=1` | vLLM's sparse-attention indexer picks `persistent_topk`, which cannot run on a GB10: 85 CTAs against 48 SMs, and the fallback wants ≥128 KB of shared memory where the part has 101,376 bytes |
| `--block-size 256` | With `index_kpool` 4 and fp8 KV, DeepGEMM's arch-12 path needs `block_kv` exactly 64 |
| `--kv-cache-dtype fp8` | The same kernel constraint |
| `HAREM_SW_BLOCK_SIZE=256` | **Mandatory in practice.** The drafter's KV group is allocated on a 16-token page, and at two ranks it takes 60.2 % of the blocks-per-request divisor against 53 % at three. Without the fix the pool is 601,562 tokens and **a 6,253-token prompt is never scheduled at all** ([docs/15](../../docs/15-tp2-track.md) §4) |

## What TP=2 buys, and what it does not

**It buys a node, and a shorter recipe.** No sidecar config, no image gate, no pad audit, no
128-block arithmetic to get wrong, and an image requirement that drops from `754421f` to any image
carrying the loader patch.

**It does not buy lower latency.** On this stack three nodes are faster per stream as well as in
aggregate, which is the opposite of what "fewer ranks, fewer collectives" predicts: a decode step
here is weight-bandwidth bound, and adding a rank cuts each rank's weight traffic by a third
([docs/15](../../docs/15-tp2-track.md) §4). Read that section before you plan around this track.

## The numbers

**The TP=2 recipe, measured 8 September 2026 — candidate D** (candidate C, the full-scope candidate
plus the sparse-indexer workspace bound, **plus the vision tower and the two upstream backports**;
[docs/15](../../docs/15-tp2-track.md) §5.10 and [docs/19](../../docs/19-vision-at-two-ranks.md)) — two nodes,
TP=2, EP off, image `exl3-zeus:754421f`, the full-scope checkpoint
(`turboderp/GLM-5.3-Flash-exl3` at 4.05 bpw), KV fp8 and an fp8 draft cache, DFlash2 k=7,
`--block-size 256`, `HAREM_SW_BLOCK_SIZE=256`, `--max-num-batched-tokens 2048`, `--max-num-seqs 8`,
`--max-model-len 1000000`, `gpu-memory-utilization 0.85`, `NCCL_MAX_NCHANNELS=8`, per-rank fast-load
sidecar, warm tuner cache, temperature 0, reasoning effort `low`, median of three sweep rounds,
**one boot** `[measured-here]`:

| | | candidate C, for comparison |
|---|---|---|
| Single-stream decode (C1) | **59.45** tok/s aggregate (**64.45** per stream) | 60.08 / 65.96 |
| Aggregate at 8 concurrent streams (C8) | **155.47** tok/s | 157.71 |
| Prefill, fresh unseen ~8.4K prompts | **1,413** tok/s | 1,414 |
| KV pool at `max_model_len` 1,000,000 | **2,585,714** tokens — about 2.6 concurrent 1M-token requests | 2,692,857; the tower costs −4.0 % |
| TTFT, C1 / C8 | **0.375** / **1.080** s | 0.381 / 1.054 |
| **4 images + 2 videos per request** | six gates, all passing, including all six items in **one** request and a third video refused with HTTP 400 | refuses the request |
| **The two backports** | an 8K exact repeat reads **57.5 %** out of the cache — **100 % of the ceiling**, against **0.0 %** with the knobs off | not present |
| Quality | probe **10/10**, code exam **12/12** first try, tool-call **8/8**, needle-lite **6/6 ×3**, cached-path equality **24/24**; MMLU sample (1,995 q) **86.02 ±0.75** | equal |
| Cold boot, fast-load, through the unit | **280 s** (the one-off dump boot that writes the sidecar is 1,013 s) | 272 s by hand / 956 s |
| Autostart unit → `/health` 200 | **280 s**, `systemctl start` on both nodes, **on this configuration**. **No reboot test yet** `[not tested]` | 261 s, candidate B's |
| Available KV memory per rank | **20.15 / 19.54 GiB** | 21.31 / 20.33 |

**MMLU is candidate B's**: neither candidate C nor D changes a language-model weight or a kernel, and
the short quality gates were taken as sufficient `[not tested]`.

**The block granularity is 4,608 tokens at two ranks**, not the three-node 3,328 — it is the one
rank-dependent number in either backport and it is measured, not copied
([docs/15](../../docs/15-tp2-track.md) §5.10).

**How candidate C was separated from candidate B.** Not by comparing the two tables above — those
are different sessions. A same-session A/B with **one environment line** between the arms, both
booting eagerly so that neither could reuse a sidecar: KV pool **1,800,000 → 2,378,571, +32.14 %**,
every gate full on both arms, and all five concurrency levels inside their declared bands.
[docs/15](../../docs/15-tp2-track.md) §5.9 has the table, the cost, and the one speed reading that is
recorded as unexplained rather than as noise.

**There are two candidates below this one and this is the recommended lineage.** Candidate A serves the
routed-experts-only checkpoint and is kept for anyone who already has those 164 GB: it is slower on
every concurrency and its pool is 1,500,000 rather than 2,128,571. The full-scope candidate B is
**+20.0 % at C1, +13.3 % at C8, +41.9 % of pool and 4.5 GiB lighter per node**, with MMLU 0.35
points apart — inside one error bar. The side-by-side table, every sweep round, the boot and sidecar
figures and every gate are [docs/15](../../docs/15-tp2-track.md) §5, raw in
[`results/speed/tp2-production-candidate.md`](../../results/speed/tp2-production-candidate.md).

**The earlier arms are kept rather than deleted** and they are **not** interchangeable with the
above — different images, different days, different stacks. They are [docs/15](../../docs/15-tp2-track.md)
§3, with their dates.

## Agentic context budget (read before pointing an agent at this endpoint)

The 1M-token figure is a **retrieval** number (needle-in-a-haystack passes at 1M). It is not the coherent
range for agentic sessions. Measured on this stack in September 2026 (our gates plus an independent
3-node deployment reporting on issue #7): first malformed tool calls appear from roughly **36k tokens**
of agentic history, the first behavioural loops around **70k**, and a full derailment was captured at
**378k** (with the sparse-MLA indexer's top-k, each token attends to well under 1% of such a history).
The healthy agentic range on this stack is roughly **50-80k tokens**; beyond ~100k is unmeasured
territory. Practical consequences: keep worker sessions short (one stage of work per card), compress
or reset well before 100k, cap completion tokens per request, and enable the structural-tag grammar
(`strict: true` on tools) together with the fail-closed parser and the xgrammar backports
(docs/14 §9.13-9.14) so that what *does* go wrong is refused rather than executed.
