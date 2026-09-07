# 00 — Start here: how many nodes do you have?

**Applies to: both tracks**, and to anyone holding one node or four.

This repository documents two working tracks over the same stack, plus a large amount of work that
belongs to neither. This page routes you to the right one in about a minute, and then says which of
the other pages apply to you.

**One question decides most of it.**

| You have | Go to | What you get |
|---|---|---|
| **1 DGX Spark** | §1 below | No serving recipe. A useful amount of everything else: the image build, the kernel fixes, the measurement protocol and the failure index |
| **2 DGX Spark** | **[15 — the TP=2 track](15-tp2-track.md)** | A *shorter* recipe than the three-node one: at two ranks nothing needs padding |
| **3 DGX Spark** | **[the README quick start](../README.md)**, then [03](03-tp3-padding-and-sidecars.md) and [13](13-full-scope-checkpoint.md) | The production recipe this repository was built around |
| **4 DGX Spark** | **[HELP-WANTED.md](../HELP-WANTED.md) §1** | Nothing measured. The padding and expert-parallel arithmetic, the cabling problem, and what we would want reported `[not tested]` |

Everything below the engine — the kernel patches, the transport plugin, the loader, the measurement
protocol — is shared by every node count and is worth reading whatever you own.

---

## 1. One node

**The model does not fit, and that is arithmetic rather than an opinion.** The production checkpoint
is **153.8 GiB** on disk and the fallback is larger still; one DGX Spark has **121.6 GiB** of unified
memory *in total*, shared between the GPU and the host, before the engine allocates anything
([00 — hardware](00-hardware-and-os.md), [01 — model](01-model-and-license.md)). There is no
single-node recipe in this repository and we are not going to imply one.

What does apply, unchanged, on one node:

| What | Where | Why it applies |
|---|---|---|
| **The image build** | [02](02-image-build.md) | Two Docker layers pinned to a `cuda-exl3` commit. Nothing in it is rank-dependent. Verify by **behaviour** — the upstream pytest suite, 44 passed / 41 skipped — because binary hashes differ between identical builds on this toolchain |
| **The GB10 kernel fixes** | [05](05-expert-parallel-and-cuda-exl3-fixes.md) §3.5, [`patches/indexer-overlay/`](../patches/indexer-overlay/) | The sparse-attention indexer's `persistent_topk` cannot run on this part — 85 CTAs against 48 SMs, and the fallback wants ≥128 KB of shared memory where the part has 101,376 bytes. That is a property of the GPU, not of the cluster |
| **The measurement protocol** | [09](09-measurement-protocol.md) | Four ways to measure a lie on this hardware. Tier A — model-free, engine down — is most of it, and it needs one GPU |
| **The model-free benches** | [`bench/`](../bench/) | `bw.py`, `gemmpeak.py`, `topk_bench.py`, `moe_stage_bench.py`, `mhc_bench.py`, `zerokv_bench.py` all run on a single node. The rulers matter: achievable read bandwidth here is **225 GB/s**, not the datasheet's 273 |
| **The failure index** | [14](14-troubleshooting.md) | Every failure we hit, by symptom, with the exact log line. The single-node ones are the build, the image and the kernel entries |
| **The open items you can still close** | [HELP-WANTED.md](../HELP-WANTED.md) §5 and §7 | The KDA GEMM engine-against-standalone gap needs **one** GB10. The local half of the mesh plugin's latency floor does too |

What does **not** apply on one node: everything in [03](03-tp3-padding-and-sidecars.md),
[06](06-nccl-mesh.md), [15](15-tp2-track.md) and [13](13-full-scope-checkpoint.md) §7. With no peer
there is no collective, no RNR flow control and no padding problem.

---

## 2. Two nodes — the TP=2 track

**[15 — Running this recipe at TP=2](15-tp2-track.md)** is the whole track: why two ranks need no
padding, the exact changes to the env file, the launcher, the patch tree and the autostart unit, our
two-node arms with their dates and settings, and the list of production features we never ran there.

Three things to know before you start, all of them on that page:

1. **It is a shorter recipe, not a cut-down one.** All five shapes that do not divide by three divide
   by two, and each leaves every rank a whole number of 128-column Hadamard blocks. No sidecar
   config, no pad audit, no padded-load path.
2. **Expert parallelism is optional at two ranks**, unlike at three where it is mandatory. Every
   TP=2 arm we ran had it off.
3. **Read the trade-off before you plan around it** ([15](15-tp2-track.md) §4). Two ranks are slower
   on every speed axis we measured, single-stream included, and the KV pool is a fraction of the
   three-node one. What TP=2 buys is a node and a shorter recipe, not latency.

---

## 3. Three nodes — the TP=3 track

The [README quick start](../README.md) is eleven steps and one optional, each ending in a check. Do not skip the
checks: on this stack the expensive failures are the silent ones.

The three pages that are TP=3's own are [03](03-tp3-padding-and-sidecars.md) (why an EXL3 tensor
cannot be split three ways, and the shape surgery that makes it possible anyway),
[13](13-full-scope-checkpoint.md) §7 (the padded-load port, which is the production recipe) and
[`tracks/tp3/patches/`](../tracks/tp3/patches/README.md).

A fourth since 7 September 2026: [18](18-vision-at-three-ranks.md), the **vision tower**, with its two-node counterpart [19](19-vision-at-two-ranks.md) since 8 September. The
production configuration accepts 4 images and 2 videos per request, and at three ranks that took a
replication flag rather than the padding this stack uses everywhere else, because nothing in the
tower divides by three.

---

## 4. Four nodes

Nothing here is measured at four `[not tested]`. What we can offer is the arithmetic, and it is
shorter than TP=3's: at four ranks the heads, the KDA heads, the shared expert, the routed experts
and the drafter all divide cleanly, expert parallelism becomes **optional**, and the **vocabulary is
the only shape that needs padding** — with our own preflight gate missing it, for a reason spelled
out in [HELP-WANTED.md](../HELP-WANTED.md) §1. The cabling is the part we cannot reason past: two
QSFP cages per node cover every pair at three nodes and only four of six at four.

One four-node recipe is already published by someone else, including the most useful negative result
we know of; it is in [16](16-comparison-with-published-recipes.md) §4.4 `[reported]`.

---

## 5. Which document applies to which track

Every `docs/NN-*.md` page carries an **Applies to** badge on its first line. This is the index.

| Document | Applies to | Note |
|---|---|---|
| [00 — Hardware, firmware and OS](00-hardware-and-os.md) | **both tracks** | The complete environment record. The fabric section describes a three-cable ring; two nodes are one pair of it |
| [01 — Model and licence](01-model-and-license.md) | **both tracks** | Same two checkpoints, same licences, at either rank count |
| [02 — Image build](02-image-build.md) | **both tracks** | TP=2 needs no image newer than the loader patch; TP=3 with the full-scope checkpoint needs the padded-load path |
| [03 — TP=3 padding and sidecars](03-tp3-padding-and-sidecars.md) | **TP=3 only** | Nothing needs padding at two ranks. This is the page TP=2 deletes |
| [04 — The DFlash2 port](04-dflash2-port.md) | **both tracks** | The drafter's GQA divides by two, so the 32/8 → 36/9 pad is a TP=3 detail inside a shared page |
| [05 — Expert parallel and the cuda-exl3 fixes](05-expert-parallel-and-cuda-exl3-fixes.md) | **both tracks** | EP is mandatory at three ranks and optional at two, and never measured on at two. §3.5, the GB10 top-k overlay, applies to any node count |
| [06 — The NCCL mesh](06-nccl-mesh.md) | **both tracks** | The channel cap is a plugin property, not a peer-count property. With one cable per pair set `NCCL_MESH_LINKS_PER_PEER=1`, which makes `patches/kernel/0005` a no-op |
| [07 — KV pool and the draft page](07-kv-and-draft-page.md) | **both tracks** | The draft page defect is **worse** at two ranks than at three, and both are measured |
| [08 — Fast boot](08-fast-boot.md) | **both tracks** | The sidecar is per rank, so two ranks means two sidecars. We never built one at TP=2 `[not tested]` |
| [09 — Measurement protocol](09-measurement-protocol.md) | **both tracks** | If you read one page, read this one |
| [10 — Results and roofline](10-results-and-roofline.md) | **TP=3** | The tables are the three-node ones. The two rulers and the method behind them apply to any node count |
| [11 — Open issues](11-open-issues.md) | **both tracks** | The retractions are the whole stack's |
| [12 — The MLA tuner cache](12-tuner-cache.md) | **both tracks** | It decides your sweep protocol at either rank count |
| [13 — The full-scope checkpoint](13-full-scope-checkpoint.md) | **both tracks** | §4-§6 are the TP=2 dress rehearsal; §7 is the TP=3 padded-load port and applies only at three |
| [14 — Troubleshooting](14-troubleshooting.md) | **both tracks** | Entries carry their own track tag where we know it |
| [15 — Running this recipe at TP=2](15-tp2-track.md) | **TP=2 only** | The two-node track |
| [16 — Comparison with other published recipes](16-comparison-with-published-recipes.md) | **both tracks** | §3 is two nodes, §4 is three, §4.4 is four |
| [17 — The memory ledger](17-memory-ledger.md) | **both tracks** | The ledger is the three-node arrangement; §6 is the two-node column, and the KDA state finding is worse at two ranks than at three |
| [18 — Vision at three ranks](18-vision-at-three-ranks.md) | **TP=3** | The vision tower, on: 4 images + 2 videos per request. The loader half is a checkpoint property and applies at any rank count; the replication flag is TP=3's own. §11 now points at the two-node page |
| [19 — Vision at two ranks](19-vision-at-two-ranks.md) | **TP=2** | The same tower on two nodes, measured 8 September 2026: what divides and why we replicated it anyway, the anchor dependency that made the two-node patch tree grow a file, and the drafter config that stops a two-node boot |

And the directories:

| Directory | Applies to | Note |
|---|---|---|
| [`bench/`](../bench/) | **both tracks** | Model-free, engine down. Most of it runs on one node |
| [`scripts/`](../scripts/) | **one launcher per track, both here** | `start-tp3.sh` + `tp3-prelude.sh` and `start-tp2full.sh` + `tp2-prelude.sh`. The benchmark harness, the probes and the prompt sets beside them are identical at either rank count |
| [`patches/kernel/`](../patches/kernel/) | **both tracks** | The mesh plugin patches. `0005` is a no-op with one cable per pair |
| [`patches/dflash2-port/`](../patches/dflash2-port/) | **both tracks** | The drafter port into the image |
| [`patches/indexer-overlay/`](../patches/indexer-overlay/) | **both tracks** | The GB10 top-k overlay. Mandatory at any node count |
| [`tracks/tp3/patches/vision/`](../tracks/tp3/patches/vision/README.md) | **TP=3 as written** | The vision patch, its three model-free gates and the prelude hook. The video half (VS4/VS6/VS7) fixes an upstream bug and is needed at **any** rank count |
| [`results/`](../results/README.md) | **both tracks** | Each file's header names its arm and its rank count |
| [`charts/`](../charts/) | **TP=3** | Generated from the three-node CSVs |
| [`audit/`](../audit/README.md) | **both tracks** | Compare your numbers against ours; the bands are three-node |
| [`tracks/`](../tracks/README.md) | **one folder per track** | The only files that differ between the two node counts: the environment template, the patch tree, the autostart unit and its preflight. [`tracks/README.md`](../tracks/README.md) says what stays shared and why |
| [`.github/`](../.github/) | **both tracks** | Three issue templates and a pull request template; the measurement one is [09](09-measurement-protocol.md) turned into a checklist |

---

## 6. What to read next, whatever you own

- [09 — Measurement protocol](09-measurement-protocol.md), before you measure anything. Boot-to-boot
  spread on this stack is **up to 16 % on C8** with nothing changed at all, so a difference under
  about 5 % on one boot is not a result.
- [11 — Open issues](11-open-issues.md) §1, before you quote a number. **Thirty-eight** claims of
  ours did not survive contact with their own raw data, and they are kept in place with what replaced
  them. §1's opening paragraph says exactly what that number counts.
- [HELP-WANTED.md](../HELP-WANTED.md), if you would like to close one of the gaps. It is ranked, it
  carries the expected effort for each item, and it says what a one-node owner can and cannot check.
