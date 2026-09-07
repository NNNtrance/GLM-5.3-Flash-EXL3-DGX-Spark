# tracks/tp3 — three nodes, TP=3 with expert parallelism

**This is the production track.** It is what our three nodes serve, start at boot, and were rebooted
into as a whole cluster with the quality gates read afterwards.

The recipe itself is the [README quick start](../../README.md) — eleven steps and one optional, each ending in a
check. This page is the directory: what is in it, what stays outside it, and the numbers this
arrangement produces.

Two nodes instead of three: [tracks/tp2](../tp2/) and [docs/15](../../docs/15-tp2-track.md).

---

## What is here

| File | What it is |
|---|---|
| [`env.tp3-full.example`](env.tp3-full.example) | **The production template — configuration 13.** Configuration 12 plus the vision tower: `LANGUAGE_MODEL_ONLY=0`, `HAREM_VISION=1`, `CUDA_EXL3_PACKED_MAPPING`, and four `--mm-*` arguments. The text-only fallback is the same file with those six lines reversed, and it is documented in place. Everything below is unchanged from configuration 12: Full-scope checkpoint, `gpu-memory-utilization` **0.88**, the sparse-indexer workspace bound (`HAREM_INDEXER_WS_MODE=bound`), the sm_12x correctness set (`HAREM_SM12_ITEMS=pdl,kpool`), fp8 KV and fp8 draft cache, `HAREM_SW_BLOCK_SIZE=256`, `NCCL_MAX_NCHANNELS=8`. Every variable carries a one-line reason and the ones with a real cost carry the measurement that decided them |
| [`env.tp3.example`](env.tp3.example) | The routed-experts-only template — production configurations 1 to 8, and the rollback |
| [`patches/`](patches/) | The in-container patch tree for the full-scope checkpoint: the ten-anchor loader patch, the image gate, the sidecar generator, the sm_12x correctness set, the prelude. [`patches/README.md`](patches/README.md) is the inventory |
| [`patches-optional/sm12/`](patches-optional/sm12/) | **Not part of the recipe** — one file. The sm_12x correctness set moved out of here into `patches/` with production 11; what is left is item 4, whose effect could not be measured from the client in either direction. The sidecar warning that copying anything into the production tree costs a dump boot lives here too |
| [`patches/vision/`](patches/vision/README.md) | **In the recipe since production configuration 13 (7 September 2026).** Six anchors and three model-free gates that turn the vision tower back on: **4 images + 2 videos per request**, with the KV pool inside configuration 12's own boot-to-boot spread, C1 +0.1 %, C8 +1.5 %, ten gates out of ten and a whole-cluster reboot at 318 s against 311 s. It cost one 590 s-class dump boot and a fresh ~53 GB-per-node sidecar, because the tower adds 596 tensor names to the identity ([docs/18](../../docs/18-vision-at-three-ranks.md)) |
| [`patches/indexer-workspace/`](patches/indexer-workspace/) | **In the recipe since production configuration 12 (6 September 2026).** One patch that bounds the sparse indexer's K-gather workspace from 4.92 GiB to 512 MB: **KV pool +10.25 %** at the same memory fraction, gates full, stress clean, no measured speed cost. It cost one 590 s dump boot and a fresh ~53 GB-per-node sidecar ([`results/memory/indexer-workspace-ab.md`](../../results/memory/indexer-workspace-ab.md)) |
| [`patches/prefix-hit-and-kpool-tail/`](patches/prefix-hit-and-kpool-tail/README.md) | **Not in the recipe — measured on 8 September 2026, both knobs default off.** Two upstream backports the pinned vLLM predates: the hybrid prefix-cache annotation (an exact repeat at 8K goes 41.56 % → **83.12 %**, the ceiling, and follow-up TTFT −62 %) and the K-pool tail slot mapping (**99.67 % → 0.00 %** of tail writes addressing the wrong block). Neither was promoted: one missed a raw-hit bar the block granularity makes unreachable, and one boot of the both-knobs arm lost a needle that six later runs did not reproduce. The gate that judged them then turned out to flake on unpatched production. [`results/gates/prefix-hit-and-kpool-tail.md`](../../results/gates/prefix-hit-and-kpool-tail.md), [docs/11](../../docs/11-open-issues.md) §2.32–2.33 |
| [`harem-exl3.service`](harem-exl3.service) | The autostart unit, installed and `enabled` on all three of our nodes |
| [`motor-onkosul-exl3.sh`](motor-onkosul-exl3.sh) | Its preflight: seven checks, at most ten minutes of waiting |

**Derive each node's environment file from the template with `sed`, on that node.** Never copy a
finished env file between nodes — two lines differ per node and a copied file produces a cluster
where two processes believe they are the same rank, with no error at all
([envs/README.md](../../envs/README.md)).

## What is not here, and where it is

| | |
|---|---|
| The launcher and prelude | [`scripts/start-tp3.sh`](../../scripts/start-tp3.sh) and [`scripts/tp3-prelude.sh`](../../scripts/tp3-prelude.sh) — every launcher in this repository lives in `scripts/`, one per track |
| The rollback patch tree | [`patches/tp3/`](../../patches/tp3/) — shared: it is production 1 to 8 here **and** the source of the `tp`-agnostic patches the two-node tree carries |
| The mesh plugin patches | [`patches/kernel/`](../../patches/kernel/) |
| The drafter port, the GB10 top-k overlay | [`patches/dflash2-port/`](../../patches/dflash2-port/), [`patches/indexer-overlay/`](../../patches/indexer-overlay/) |
| Everything measured | [`results/`](../../results/README.md), [`charts/`](../../charts/), [`bench/`](../../bench/) |
| The install commands and the reboot test | [`systemd/README.md`](../../systemd/README.md) |

**The directory name here is not the directory name on your nodes.** On a node this tree lives at
`~/exl3-zeus/tp3full/`, and that name is load-bearing three times over: the launcher mounts
`$TP3_DIR/tp3-prelude.sh` at `/start.sh`, `tp3-prelude.sh` is a **hard link** to
`tp3full-prelude.sh` rather than a copy, and the directory's file list and the full text of the
prelude are hashed into the fast-load sidecar's identity
([docs/08](../../docs/08-fast-boot.md) §4). See [tracks/README.md](../README.md) for the copy and the
link.

**The launcher directory and the patch-tree directory are two things**; the examples use `tp3full/`
for both for brevity, and on our own nodes the patch tree is a separate directory named after the
configuration (which is why [`env.tp3-full.example`](env.tp3-full.example) shows one). Set `TP3_DIR`
and `OVERLAY_DIR` to wherever you put the patch tree and the overlay; the systemd unit's `ExecStart`
follows the launcher instead.

---

## What this track measures

**Production configuration 12** — three DGX Spark nodes, one 4-bit EXL3 checkpoint, realistic
prompts, temperature 0, reasoning effort `low`, 6 September 2026 `[measured-here]`. Speed is the pool
of six sweep rounds over **two boots** of this configuration:

| | | production 10 `[history]` |
|---|---|---|
| Single-stream decode (C1) | **69.7** tok/s aggregate (**75.6** per stream) | 70.5 (76.9) |
| Aggregate at 8 concurrent streams (C8) | **196.1** tok/s | 194.0 |
| Prefill, fresh unseen ~8K prompts | **1,744** tok/s | 1,769 |
| KV pool at `max_model_len` 1,000,000 | **7,041,322** tokens at `gpu-memory-utilization` 0.88 | 5,619,834 at 0.83 |
| Quality | correctness probe **10/10**, code exam **12/12** cold and warm; tool-call gate **8/8**; needle-lite **6/6** at 64K and 128K; MMLU sample (1,995 q) **86.47 ±0.74**, carried from configuration 9 `[not tested]` here | 10/10 · 12/12; MMLU 86.47 ±0.74 |
| Cold boot, `docker run` → API ready | **272 s** (the dump boot that writes the sidecar is 590 s) | 251 s |
| Boot from power-on, all three nodes together | `/health` 200 at **311 s** by the wall clock, timed from the `reboot` command | 242 s by the harness's counter, **315 s** by the wall clock in the same log — both printed because they disagree, and 315 s was the figure to plan with |

Settings for every row: image `exl3-zeus:754421f`, TP=3 + expert parallel, full-scope EXL3 weights
(`turboderp/GLM-5.3-Flash-exl3` at 4.05 bpw), `kv-cache-dtype fp8` and an fp8 draft cache, DFlash2
draft at k=7, `--block-size 256`, `HAREM_SW_BLOCK_SIZE=256`, `--max-num-batched-tokens 2048`,
`--max-num-seqs 8`, `NCCL_MAX_NCHANNELS=8`, per-rank pre-sliced sidecar, warm MLA tuner cache, mesh
plugin with both links per peer and `NCCL_PTR_CUDA`, the launcher's memory settle gate. Configuration
12 adds the sm_12x correctness set (`HAREM_SM12_ITEMS=pdl,kpool`) and the sparse-indexer K-gather
workspace bound (`HAREM_INDEXER_WS_MODE=bound`), which configuration 10 predates. Configuration 10's
speed is the median of three sweep rounds, which the persisted tuner cache is what earns
([docs/09](../../docs/09-measurement-protocol.md), [docs/12](../../docs/12-tuner-cache.md)).

**Before you read a few-percent difference here as a result:** on this stack C1 boot medians span
1.1 %, C8 2.5 % and **C4 7.4 %**, and boot-to-boot on C8 has been measured at **15.9 % with nothing
changed at all**. The full tables are [docs/10](../../docs/10-results-and-roofline.md); the
configuration-by-configuration progression is
[`results/configs/production-configurations.csv`](../../results/configs/production-configurations.csv).

## What is TP=3's own, and why

Four things in this repository exist only because three ranks is not two:

1. **The padding.** Five shapes in GLM-5.3-Flash do not divide by three, and an EXL3 trellis cannot
   be zero-extended, so the shapes are padded to whole 128-column Hadamard blocks: heads 64 → 66,
   vocabulary `padding_size` 384 giving 155,136, shared expert 2,048 → 2,304, drafter GQA 32/8 →
   36/9 ([docs/03](../../docs/03-tp3-padding-and-sidecars.md)).
2. **Expert parallelism is mandatory**, not a tuning choice: 2,048/3 is not an integer, so the routed
   experts cannot be tensor-sliced at all and each rank must hold 96 whole experts
   ([docs/05](../../docs/05-expert-parallel-and-cuda-exl3-fixes.md)).
3. **The padded-load path**, which is what puts a *fully quantized* checkpoint into dimensions vLLM
   has padded. It needs `cuda-exl3` at `754421f` or later plus our A9 and A10 patches, and both are
   provable no-ops at TP≤2 ([docs/13](../../docs/13-full-scope-checkpoint.md) §7).
4. **The replicated vision encoder.** Nothing in the tower divides by three — not the 16 heads, not
   the 4096 MLP, not the merger's 4096 and 10240 — so it is `--mm-encoder-tp-mode data` rather than
   the padding this stack uses everywhere else. **At two ranks that flag is not needed**; the rest of
   [`patches/vision/`](patches/vision/README.md) is a property of the checkpoint and of upstream, not
   of the rank count ([docs/18](../../docs/18-vision-at-three-ranks.md) §11).

Everything else on this stack — the image, the kernels, the fabric work, the KV pool surgery, the
fast boot, the measurement protocol — belongs to both tracks.
