# 15 — Running this recipe at TP=2

**Applies to: TP=2 only.** This is the two-node track's own page. Everything else in the repository is
shared unless its first line says otherwise.

This repository is a three-node recipe and every default in it is a TP=3 default. It also runs on
**two** DGX Spark nodes, and at two ranks it is a *simpler* recipe rather than a cut-down one: nothing
needs padding, so the shape surgery in [03](03-tp3-padding-and-sidecars.md) and the padded-load path
in [13](13-full-scope-checkpoint.md) §7 are not needed at all.

**As of 6 September 2026 this page describes a complete, measured two-node configuration rather than
a set of bring-up arms.** There is a named TP=2 production candidate — a patch tree, an environment
file, a launcher, a fast-load sidecar and an autostart unit — and every row of it was measured on
the head node (rank 0) and worker-1 on 6 September 2026 with the protocol in
[09](09-measurement-protocol.md). §5 is that configuration. §3 is how we got there, arm by arm, with
dates. §6 is the shorter and more honest list of what is still untested.

**Two things on this page reversed on 6 September, and both had been published here as findings.**
The draft KV page (`HAREM_SW_BLOCK_SIZE=256`) was listed as untested at two ranks and is now the
single setting without which two ranks cannot serve an 8K prompt at all (§4). And the **full-scope
checkpoint**, which this page called "a rig and not a serving configuration" at two ranks because it
could not boot at `max_model_len` 1,000,000, boots there comfortably once the page fix and the
launcher's settle gate are both in place — and is the faster, larger-pool, recommended candidate
(§5). Both retractions are marked where they sit.

Every number below carries its date, its image and its settings. Evidence tiers are the ones in
[STYLE-GUIDE.md](../STYLE-GUIDE.md).

---

## 1. Why two ranks need no padding, and what that removes

### 1.1 The five shapes

[03](03-tp3-padding-and-sidecars.md) §1 lists the five numbers in GLM-5.3-Flash that do not divide by
three. **All five divide by two**, and — this is the part that matters for an EXL3 checkpoint — all
five leave every rank a whole number of 128-column Hadamard blocks:

| Shape | Value | At TP=2 | Whole 128-blocks per rank | At TP=3, for contrast |
|---|---|---|---|---|
| `num_attention_heads` | 64 | 32 per rank | MLA `qk`/`v` head_dim 256, so yes | padded to 66 → 22 per rank |
| `num_key_value_heads` | 64 | 32 per rank | yes | 66 → 22 |
| KDA `linear_attn_config.num_heads` | 64 | 32 per rank | head_dim 128, so yes | 66 → 22 |
| `vocab_size` | 154,880 | 77,440 per rank = **605 × 128** | yes | `padding_size` → 384, giving 155,136 |
| shared expert intermediate | 2,048 | 1,024 per rank = **8 × 128** | yes | padded to 2,304 → 768 per rank |
| routed experts `moe_intermediate_size` | 2,048 | 1,024 per rank = **8 × 128** | yes | not sliced at all; 96 whole experts per rank under EP |

Two consequences follow, and they are the whole reason a two-node track is worth writing down.

**First: expert parallelism is optional at TP=2.** At three ranks it is mandatory, because 2,048/3 is
not an integer and a trellis cannot be zero-extended ([03](03-tp3-padding-and-sidecars.md) §1.1). At
two ranks the routed experts tensor-slice cleanly to 1,024 = 8 × 128 per rank, so both arrangements
are legal. `tracks/tp2/patches/preflight-tp3.py` encodes exactly this rule and prints
`preflight, tp=2, ep=OFF (experts tensor-sliced)` while refusing `--ep 0` at `tp=3`: it requires
`moe_intermediate_size` to be a multiple of `128 × tp`, and 2,048 is a multiple of 256 but not of
384 `[measured-here]`. **Every TP=2 arm we have run had EP off.** We have never measured EP on at two
ranks `[not tested]`.

**Second: the padded-load path is not needed.** The full-scope checkpoint's TP=3 port exists because
`lm_head` had to be loaded narrow into a vocabulary vLLM had padded, and that needed three
capabilities the kernel author added in `f3e3090` and `754421f`
([13](13-full-scope-checkpoint.md) §7.2). At TP=2 there is no pad, so there is nothing to load
narrow into: our first full-scope TP=2 arm ran on **`exl3-zeus:62f53e6`**, an image that raises
`EXL3 weights cannot be zero-extended` at TP=3 `[measured-here]`. The production candidate runs
`754421f` only so that both tracks share one image.

The two launcher constants that production 9 moved from `lcm(64, tp)` to `lcm(128, tp)` are computed
at run time from `tp`, so `patch-vllm-tp3.py` is a **no-op at TP≤2** by arithmetic — `lcm(128, 2) =
128`, 154,880 is already a multiple of 128, and 2,048/2 = 1,024 = 8 × 128. We do not ship it in the
two-node tree anyway; see §2.3.

### 1.2 What TP=2 does not get out of

Everything on this list is a property of the part, the fabric or the model, not of the rank count, and
it applies unchanged at two nodes:

- **The GB10 top-k overlay.** vLLM's sparse-attention indexer picks `persistent_topk`, which cannot
  run on this part — 85 CTAs against 48 SMs, and the fallback wants ≥128 KB of shared memory where
  the part has 101,376 bytes. This is the failure that stopped our very first TP=2 boot
  `[measured-here]`, and `HAREM_DISABLE_PERSISTENT_TOPK=1` is how it is cleared
  ([14](14-troubleshooting.md), [05](05-expert-parallel-and-cuda-exl3-fixes.md) §3.5).
- **`--block-size 256`.** Required at two ranks for the same reason as at three: with `index_kpool`
  4 and fp8 KV, DeepGEMM's arch-12 path needs `block_kv` exactly 64. Without it, built-in MTP dies in
  `attention.hpp:320` `[measured-here]`.
- **fp8 KV**, and **the fp8 draft cache**. Same kernel constraint for the first; the second is a
  pool decision, measured at two ranks in §5.3.
- **The fabric work in [06](06-nccl-mesh.md).** Two nodes are one peer pair rather than three, but
  `NCCL_MAX_NCHANNELS=8` addresses the plugin's per-channel RNR behaviour, not the number of peers,
  and the second-cable patch (`patches/kernel/0005`) exists because a *pair* had two cables and only
  one was carrying packets. **Both cables of a single pair are now measured as carrying traffic at
  two ranks** (§5.4) `[measured-here]`. If your two nodes are joined by one cable, set
  `NCCL_MESH_LINKS_PER_PEER=1`, which makes 0005 a no-op.
- **The per-rank fast-load sidecar** ([08](08-fast-boot.md) §2.4). It is per rank, so two ranks means
  two sidecars — and because EP is off, each rank owns *half* of every tensor rather than a third,
  so a two-node sidecar is **78–83 GB per rank** against a three-node one's 53 GB (§5.2)
  `[measured-here]`.
- **The quality gates.** `scripts/correctness-probe.py` and `scripts/code-exam.py`, cold **and** warm.
  Both passed on every TP=2 arm we have ever run.

### 1.3 What it buys

- **A node.** Two Sparks instead of three.
- **A shorter recipe.** No `pad-tp3full.py` sidecar config, no `check-padload-tp3.py` image gate, no
  `svh = 0` pad audit, no 128-block arithmetic to get wrong. The five asserts that exist to catch a
  silently half-padded stack have nothing to catch, and the patch tree is 14 files rather than 22.
- **The loader work is visible on its own.** Our full-scope dress rehearsal ran at TP=2 first
  precisely because it answers "what is the dense stage worth" with no padding machinery on top
  ([13](13-full-scope-checkpoint.md) §4).

It does **not** buy lower single-stream latency on this stack. See §7.

---

## 2. To run this recipe at TP=2, change exactly these

Everything not listed here is unchanged, including the image build ([02](02-image-build.md)), the
checkpoints ([01](01-model-and-license.md)), the DFlash2 port ([04](04-dflash2-port.md)), the
measurement protocol ([09](09-measurement-protocol.md)) and the troubleshooting index
([14](14-troubleshooting.md)).

### 2.1 The env file

Start from [`tracks/tp2/env.tp2-full.example`](../tracks/tp2/env.tp2-full.example) — that template *is* the
production candidate of §5, and it already carries everything below. If you are converting a
three-node file instead, the changes are:

```
NNODES=2
TP_SIZE=2
ENABLE_EP=0
```

`ENABLE_EP=0` is the arrangement every one of our TP=2 measurements used. `ENABLE_EP=1` is also legal
at two ranks and we have never measured it `[not tested]`; the preflight will accept either.

- `MODEL_HOST_PATH` — point it at the checkpoint itself, not at a padding sidecar. There is no padded
  `config.json` to build at TP=2, so set `MODEL_PATH` to a plain container path and leave
  `MODEL_LINK_TARGET` unset; the identity-mount machinery has nothing to protect.
- `DRAFT_HOST_PATH` — likewise. The DFlash2 drafter's GQA is 32/8, which divides by two, so the
  32/8 → 36/9 pad that TP=3 needs is not needed and `patch-dflash-tp3.py` is not in the tree.
- **Keep `--hf-overrides` if you serve the full-scope checkpoint.** It is not about padding: that
  checkpoint's inlined quantization config carries no `tensor_storage`, so `cuda-exl3` has to be
  pointed at the standalone file. At TP=2 that file is the checkpoint's own, unrewritten:

```
EXTRA_ARGS='--block-size 256 --safetensors-load-strategy eager --no-enable-flashinfer-autotune --hf-overrides {"quantization_config_file":"/models/<your-checkpoint-dir>/quantization_config.json"}'
```

Single quotes and no spaces inside the JSON, for the reasons in
[14](14-troubleshooting.md). Note also what is **absent**: `--enable-ep-weight-filter`. That flag
needs expert parallelism, and EP is off here.

### 2.2 The launcher

Use [`scripts/start-tp2full.sh`](../scripts/start-tp2full.sh) rather than editing `start-tp3.sh`.
It is the three-node launcher with the EP requirement removed and four pieces kept that a bring-up
launcher does not have — every one of which we needed:

| Piece | Why it is in the two-node launcher |
|---|---|
| the F1 **settle gate** (`MemAvailable ≥ 112 GiB`, 60 polls) | the KV pool is computed from a `MemAvailable` snapshot, so a rank that starts on a dirty host awards itself memory it does not have ([07](07-kv-and-draft-page.md) §1.1). This is the single change that turns the full-scope arm from "cannot boot at 1M" into the production candidate (§5.5) |
| the **fastload block** (`FASTLOAD_MODE=dump\|load`, identity mount) | boot 997 s → **272 s** (§5.2) |
| `PROF_ARG` | this vLLM takes the torch profiler as `--profiler-config`; without the flag `/start_profile` answers 404 |
| `-e NODE_RANK -e TP_SIZE -e ENABLE_EP -e TP3_DIR` | the bring-up launcher passed none of them, which is why its prelude printed `rank=? tp=?` — and **the fastload sidecar cannot work without `NODE_RANK`** |

`TP3_DIR` is deliberately set to the *same* directory as `TP2_DIR`. `preflight-fastload.py` and
`harem_fastload.py` both read `TP3_DIR` to find the patch set they must hash; pointing it at the
two-node tree is what makes the sidecar identity cover the two-node patch set instead of a
three-node one that is not even mounted.

**Boot order is worker-1, then head** — the same rule as at three ranks, highest rank first.

### 2.3 The patch tree

Use [`tracks/tp2/patches/`](../tracks/tp2/patches/). It is the two-node tree, complete, and every file in
it is documented here:

| File | What it does at two ranks |
|---|---|
| `tp2full-prelude.sh` | the in-container prelude; applies the patches below, then `exec vllm serve`. **Install it under two names** — `tp2-prelude.sh`, which the launcher mounts at `/start.sh`, and `tp3-prelude.sh`, which is the name `harem_fastload_id.file_identity()` hashes. A **hard link**, not a copy, so the two cannot drift; without the second name the prelude's text silently falls out of the sidecar manifest, and `start-tp2full.sh` refuses to launch if either is missing. [`tracks/tp2/patches/README.md`](../tracks/tp2/patches/README.md) has the two-line install |
| `patch-kvdiag-tp3.py` | logging only: the per-group decomposition of the pool arithmetic. Every arm |
| `patch-swblock-tp3.py` | the draft KV page, 16 → 256 tokens. **Mandatory in practice** (§4) |
| `patch-draftkv-tp3.py` | the drafter's own cache at fp8. Gated on `HAREM_DRAFT_KV_DTYPE`; measured at two ranks in §5.3 |
| `patch-tilelang-failloud-tp3.py` | turns a silent `contextlib.suppress` around `import flashinfer.comm` into a named error. Gated |
| `patch-epfilter-tp3.py` | EP weight filter for `.trellis`. **Inert here** — it needs `--enable-ep-weight-filter`, which needs EP. Kept so an EP-on TP=2 arm needs no tree change, because a tree change means a new manifest and a new dump boot |
| `patch-fastload-tp3.py` | installs the sidecar hook into vLLM's loader. Inert unless `HAREM_FASTLOAD_MODE` is set |
| `harem_fastload.py`, `harem_fastload_id.py` | the dump/restore engine and the identity |
| `preflight-fastload.py` | refuses a stale sidecar in the prelude, before vLLM starts |
| `preflight-tp3.py` | the shape preflight. `tp`-parameterised; run it as `--tp 2 --ep 0` |
| `flashinfer-warmup.py` | imports `flashinfer.comm` once, CPU-side, ~2 s |
| `patch-fullscope-tp2.py` | S1 packed mapping, S2 stop hard-wiring MLA+KDA to bf16, S3 KDA refactorisation. Gated on `HAREM_EXL3_FULLSCOPE`. **This is the patch the recommended candidate runs** |

And what is deliberately **not** in it, because all of it exists only to serve a pad that does not
exist at two ranks:

| Not shipped | Why |
|---|---|
| `patch-vllm-tp3.py` | the zero-extend helper for a shard that runs past the stored dim. No pad at tp=2, so the branch can never fire. It is harmless to keep if you prefer one tree for both rank counts — but no two-node arm of ours has ever run it, and we do not ship unmeasured text |
| `patch-exl3-ep.py` + `overlay/cuda_exl3/` | the cuda-exl3 EP kernel fixes. EP is off |
| `patch-dflash-tp3.py` | makes the DFlash2 head check pad-aware over the 32/8 → 36/9 drafter pad |
| `patch-fullscope-tp3.py` | its A9 (split a fused tensor by checkpoint widths) and A10 (post-load pad audit) are the padded-load half. A9 is a no-op at TP≤2 and A10 has nothing to audit; `patch-fullscope-tp2.py` is the same S1–S3 without them |
| `check-padload-tp3.py` | gates a `cuda-exl3` capability only the padded-load path needs |
| `pad-tp3.py`, `pad-tp3full.py` | build padded sidecars. Nothing to pad |
| `patch-zerokv-tp3.py` | an optional arm, never in a production env |

**Keep the file list in its own directory.** The directory's file list and the full text of the
prelude are hashed into the fast-load manifest ([08](08-fast-boot.md) §4), and adding one file to a
tree — even a file that is never called — refuses the next boot on every node. That has happened to
us twice, and once it was **the TP=2 patch dropped into the TP=3 tree** that did it
([13](13-full-scope-checkpoint.md) §6.4). Finish the tree *before* the dump boot and add nothing
between dump and load.

### 2.4 The image

`exl3-zeus:62f53e6` is enough at TP=2 `[measured-here]`; the production candidate runs **`754421f`**
so that both tracks share one image and one sidecar-invalidation event. `754421f` adds only the
padded-load capability, which two ranks do not use.

### 2.5 Flags that stay, and the two that are not optional

| Flag / knob | At TP=2 |
|---|---|
| `HAREM_DISABLE_PERSISTENT_TOPK=1` | **mandatory**, same reason (§1.2) |
| `--block-size 256` | **mandatory**, same reason |
| `--kv-cache-dtype fp8` | **mandatory**, same reason |
| `HAREM_SW_BLOCK_SIZE=256` | **mandatory in practice** — without it the pool is 601,562 and a 6,253-token prompt is never scheduled at all (§4) `[measured-here]` |
| `HAREM_DRAFT_KV_DTYPE=fp8` | keep: **+15.1 % of pool at two ranks, speed and acceptance unchanged** (§5.3) `[measured-here]` |
| `NCCL_MAX_NCHANNELS=8` | keep; it is a plugin property, not a peer-count property |
| `NCCL_MESH_LINKS_PER_PEER` | `0` (auto) with two cables between the pair — **both cables measured carrying ~90 GB each across a sweep** (§5.4); `1` with one cable, which makes `patches/kernel/0005` a no-op |
| `NCCL_MESH_MIN_RNR_TIMER=1`, `NCCL_MESH_PTR_CUDA=1`, `NCCL_MESH_FLUSH=1` | keep |
| `CUDA_EXL3_TUNE_CACHE` | keep, and warm it before you measure anything ([12](12-tuner-cache.md)). Warm at two ranks it does what it does at three: round 1 is inside the round-to-round band (§5.6) |
| `GPU_MEMORY_UTILIZATION` | **0.85**, which is where every TP=2 arm of ours has run. It is *not* production 10's 0.83, and the ladder has never been derived at two ranks — read §6 before moving it |
| `MAX_MODEL_LEN` | **1,000,000 is reachable with either checkpoint** once the page fix and the settle gate are in place. That is a reversal; see §5.5 |

### 2.6 The autostart unit

[`tracks/tp2/harem-exl3-tp2.service`](../tracks/tp2/harem-exl3-tp2.service) and its preflight
[`tracks/tp2/motor-onkosul-exl3-tp2.sh`](../tracks/tp2/motor-onkosul-exl3-tp2.sh) are the two-node pair.
They differ from the three-node ones in four places and no more:

- `WorkingDirectory` and `ExecStart` point at `tp2full/` and `start-tp2full.sh`.
- `ExecStop` names `exl3-tp2`.
- `Conflicts=harem-exl3.service harem-motor.service` — **both**. Two engines can never run on one
  node's unified memory, and the three-node unit is the second engine here. Enable exactly one.
- the preflight's `FABRIC_PEERS` becomes **one** address per node. The ConnectX-7 check stays at
  `4/4` because that counts ports on the node, not peers, and a node whose other two ports are down
  has a hardware problem worth failing on even when this cluster does not use them.

**Installed, started, health-checked and stopped on both nodes on 6 September 2026**
`[measured-here]` — §5.7. It is installed **disabled**: the three-node unit remains the enabled
autostart on our cluster, and a two-node reboot test has still not been run `[not tested]` (§6).

And the reboot rule becomes "reboot **both** together, never one". The preflight checks only its own
node's fabric, so a single-node reboot passes it and starts a rank into a cluster whose peer is gone
([00](00-hardware-and-os.md) §3.4).

---

## 3. The arms, in order

Seven arms now, all on two DGX Spark nodes, all at temperature 0 and reasoning effort **low**, all
with the GB10 top-k overlay and `--block-size 256`. **Arms A–D are older stacks. Do not compare a
number in arm A against a number in candidate B.**

| Arm | Date | Image | Checkpoint | What it established | Section |
|---|---|---|---|---|---|
| **A** bring-up | 4 Sep | `serve` (`37330c9`) | experts-only | it runs at all; MTP k=3 beats no draft | §3.1 |
| **B** DFlash2 | 4–5 Sep | `dflash` | experts-only | DFlash2 k=7 is the fastest drafter, and it costs 58 % of the pool | §3.1 |
| **C** current stack, control | 5 Sep | `62f53e6` | experts-only | the two-node baseline the fabric work was measured against | §3.2 |
| **D** full-scope, first try | 5 Sep | `62f53e6` | full-scope | +24 % single stream — and, we thought, an unusable pool | §3.2 |
| **KV-fix control / fix** | 6 Sep | `62f53e6` | experts-only | the draft page is not optional at two ranks | §4 |
| **Candidate A** | 6 Sep | `754421f` | experts-only | the complete two-node configuration on the smaller checkpoint | §5 |
| **Candidate B** | 6 Sep | `754421f` | **full-scope** | **the recommended TP=2 production candidate** | §5 |

### 3.1 Arms A and B — bring-up, 4–5 September 2026

**Arm A.** Image `exl3-zeus:serve` (`cuda-exl3` `37330c9`), checkpoint
`brandonmusic/GLM-5.3-Flash-tr3-4bpw` revision `b20c49ba` (routed experts EXL3 4-bit, head 16-bit),
TP=2, EP off, `--attention-backend CUSTOM`, KV fp8, `gpu-memory-utilization 0.85`, no KV pin,
`--max-model-len 1000000`, `--max-num-seqs 8`, `--max-num-batched-tokens 2048`, mesh plugin,
**before** any of the fabric work in [06](06-nccl-mesh.md). Two speculative arrangements: none, and
the image's built-in MTP at k=3. Published as `Zeuss5/cuda-exl3` issue #2 `[measured-here]`.

| | no draft | MTP k=3 |
|---|---|---|
| Boot to serving | **471 s** (weights 312 s, 79.95 GiB per node) | 471 s |
| KV pool at 0.85, fp8, 1M context | **2,548,117** tokens | 1,987,179 (the MTP layer plus block 256: −22 %) |
| Quality gates | 10/10 correctness, 12/12 code exam, 0 empty | 10/10 · 12/12 · 0 empty |
| Single stream, 700-token code prompt | 14.6 tok/s | 30.3–30.6 tok/s, acceptance 70 % |
| `hizset-v2` C1 / C2 / C4 / C6 / C8, aggregate | 14.4 / 28.8 / 46.4 / 50.6 / **70.6** | 30.6 / 48.0 / 71.0 / 85.9 / **102.4** |
| Prefill, 7.4K prompt, cold cache | **1,137** tok/s | 1,131 tok/s |
| MMLU sample, 57 × 35 = 1,995 q, 0-shot | — | **86.4 ±0.7** |
| Free host RAM after load tests / swap | head 5.2 GB / **3.5 GB swap**; worker-1 7.1 GB / 2.8 GB | head 4.8 GB, worker-1 6.7 GB |

The swap in that last row appeared **during weight load** and was stable afterwards. It is the
clearest single reading of what two nodes cost, and it is the reason §6 still refuses to move the
memory rung without a derivation.

**Arm B — the DFlash2 port at two ranks**, image `exl3-zeus:dflash`, drafter
`incoai/GLM-5.3-Flash-DFlash2` at k=7, same checkpoint and env otherwise, mean of two rounds
`[measured-here]`. Three columns run back to back on one boot:

| | no draft | MTP k=3 | **DFlash2 k=7** |
|---|---|---|---|
| C1 aggregate | 14.42 | 30.49 | **42.91** |
| C1 per stream | 14.73 | 33.52 | **50.79** |
| C2 / C4 / C6 aggregate | 28.80 / 46.49 / 49.30 | 47.89 / 71.12 / 85.23 | **60.80 / 83.89 / 98.08** |
| C8 aggregate | 69.50 | 102.32 | **114.60** |
| acceptance · accepted tokens per step | — | 77.3 % · 3.31 | **62.4 % · 5.37** |
| KV pool | — | 1,987,179 | **825,000** |
| Gates | 10/10 · 12/12 | 10/10 · 12/12 | 10/10 · 12/12 |

**What that cost.** The KV pool falls from 1,987,179 to 825,000 — **−58 %** — because the drafter's
KV group is allocated on a 16-token page. That is the defect [07](07-kv-and-draft-page.md) §3
diagnoses; §4 is its measurement at two ranks. And prose gets *slower*, 21.3 → 18.5 tok/s: at k=7 the
wasted draft costs more than the occasional hit is worth on high-entropy text.

### 3.2 Arms C and D — the current stack, 5 September 2026

Both arms: two nodes, TP=2, EP off, image `exl3-zeus:62f53e6`, KV fp8, DFlash2 k=7,
`gpu-memory-utilization 0.85`, `--block-size 256`, `--max-num-seqs 8`,
`--max-num-batched-tokens 2048`, `NCCL_MAX_NCHANNELS=8`, mesh plugin with both cables per peer,
`CUDA_EXL3_TUNE_CACHE` warm, **no** fast-load sidecar, **no** `HAREM_SW_BLOCK_SIZE`, **no** fp8 draft
cache, **no settle gate**, medians of three rounds `[measured-here]`. Arm C is the experts-only
checkpoint; arm D is `turboderp/GLM-5.3-Flash-exl3` at 4.05 bpw. **Arm D ran at `max_model_len`
65,536 against arm C's 1,000,000**, because at 1M it could not boot.

| metric | **C — experts-only** | **D — full-scope** | delta, where it is one |
|---|---|---|---|
| C1 per stream (tok/s) | 54.69 | **68.00** | **+24.3 %** |
| C1 aggregate (tok/s) | 47.40 | **59.93** | **+26.4 %** |
| C2 / C4 aggregate | 68.03 / 90.66 | 83.02 / 111.05 | +22.0 % / +22.5 % |
| C6 / C8 aggregate | 110.12 / 133.57 | 109.75 / 110.03 | **void — arm D is KV-bound** |
| TTFT at C1 | 0.615 s | 0.524 s | −14.8 % |
| acceptance at C1 · tokens per step | 64.08 % · 5.49 | 63.14 % · 5.42 | equal |
| decode step time | 100.4 ms | 79.7 ms | −20.6 % |
| Boot, cold, no sidecar | 396 s | 355 s | −10 % |
| KV pool | **665,625** at 1M ctx | 31,343 at 65,536 ctx | **not comparable** |
| Prefill, three fresh 8,204-token prompts | **1,334** tok/s | not measurable | |
| Gates, cold and warm | 10/10 · 12/12 | 10/10 · 12/12 | equal |
| MMLU sample, 1,995 q, 0-shot | 86.4 ±0.7 | **86.32 ±0.75** | inside the error bar |

**Arm D's cost line said TP=2 full-scope was a rig and not a serving configuration. That judgement
is `[retracted]`.** At 1M it failed its own budget gate with `6.6 GiB KV needed for max seq len
1,000,000, available 0.73 GiB`; dropped to 65,536 it produced a 31,343-token pool — about 6.8 pages
— and a ~2,800-token prompt was never scheduled. **Both readings are real, and neither survives the
two changes that came after them:** the draft page fix cuts blocks-per-request from 640 to 280, and
the launcher's settle gate stops a rank measuring a dirty host. With both in place the same
checkpoint boots at `max_model_len` 1,000,000 with **16.07 GiB** of available KV memory and a
**2,128,571**-token pool (§5.5). The arithmetic of the reversal is decomposed there.

**Arm C's cost line is the plain two-node one.** 665,625 tokens against production 8's 4,699,724 on
the same day — because a 164 GiB model over two nodes leaves about 85 GiB of weights per node and
very little else. That figure is also **an unpinned baseline**: the same env file booted through a
settle gate on 6 September got 601,562, not 665,625 (§4).

### 3.3 One reading that did not survive, and is kept here

Arm D at `max_model_len=1,000,000` implied the full-scope checkpoint was about **10 GiB heavier per
node** than the experts-only one — while post-run free host memory on the same arm said its weights
were ~3.3 GiB per node *lighter*. Those two readings contradicted each other and we never isolated
the mechanism; at three ranks the sign reversed.

**6 September settles it, on one boot each with the same settle gate, the same image and the same
launcher** `[measured-here]`: consumed memory (weights + non-torch) is **89.3 / 89.2 GiB** per node
with the experts-only checkpoint and **84.8 / 84.5 GiB** with the full-scope one. The full-scope
checkpoint is **4.5–4.7 GiB lighter per node**, which agrees in sign and roughly in size with the
three-node reading of 3.4 GiB lighter. The "10 GiB heavier" arithmetic stays `[retracted]` and its
source is now named: an unsettled boot.

A separate finding from that investigation does stand: lowering `max_model_len` from 1,000,000 to
65,536 raised available KV memory from 0.73 to **5.41 GiB**, so roughly **4.7 GiB per node of
persistent, non-KV allocation scales with `max_model_len`** on this stack `[measured-here]`.

---

## 4. The draft KV page at two ranks — measured, 6 September 2026

This is the setting to get right before anything else. Two arms back to back, one boot each, **the
only difference one token in `EXTRA_ENV`** `[measured-here]`. Full raw record:
[`results/speed/tp2-draft-page.md`](../results/speed/tp2-draft-page.md).

**Settings, identical in both arms.** Two nodes, TP=2, EP off, image `exl3-zeus:62f53e6`,
experts-only checkpoint at `b20c49ba`, KV `fp8`, DFlash2 k=7, `--attention-backend CUSTOM`,
`--block-size 256`, `--max-num-seqs 8`, `--max-num-batched-tokens 2048`, `--max-model-len 1000000`,
`gpu-memory-utilization 0.85`, `HAREM_DISABLE_PERSISTENT_TOPK=1`, `NCCL_MAX_NCHANNELS=8`, mesh plugin
with both cables, `CUDA_EXL3_TUNE_CACHE` warm, no fast-load sidecar, no fp8 draft cache, temperature
0, reasoning effort **low**, medians of three rounds. Both arms booted through the host-side settle
gate (`MemAvailable ≥ 112 GiB`; they started at 116.9 and 117.1 GiB).

### 4.1 Why the defect is *worse* at two ranks

The engine's own per-group decomposition, printed by `patch-kvdiag-tp3.py` in both arms:

| group | page, control | blocks/request, control | page, fix | blocks/request, fix |
|---|---|---|---|---|
| `MLAAttentionSpec`, 22 layers | 152,064 B | 218 | 152,064 B | 218 |
| `KpoolTailSpec`, 11 layers | 152,064 B | 1 | 152,064 B | 1 |
| `MambaSpec` × 4 | 2,359,296 B | 9 + 9 + 9 + 9 | 2,359,296 B | 9 + 9 + 9 + 9 |
| **`SlidingWindowSpec` — the drafter, 5 layers** | **32,768 B** (16 tokens) | **385** | **524,288 B** (256 tokens) | **25** |
| **blocks per request** | | **640** | | **280** |
| `num_blocks` | | 385 | | 365 |
| **`GPU KV cache size` at 1M** | | **601,562** (0.60x) | | **1,303,571** (1.30x) |

**The drafter takes 60.2 % of the divisor at two ranks against 53 % at three.** Not because the
drafter changed — its 385 is identical — but because the target's share shrank: the platform raises
the attention block to **4,608** tokens at TP=2 where it is 3,328 at TP=3, and that cuts the MLA
group from 301 blocks per request to 218. A smaller denominator makes the same defect a larger share.

### 4.2 What it did

| metric | control — page 16 | **fix — page 256** | delta |
|---|---|---|---|
| **KV pool at 1M** | 601,562 | **1,303,571** | **+116.7 %** |
| the same, normalised to equal binding-rank KV memory | 601,562 | 1,260,714 | **+109.6 %** |
| **prefill-fresh, 3 unseen ~8.3K prompts** | **never scheduled** | **1,478 tok/s** | path opens |
| **prefill, 7,382 tokens, uncached** | **never scheduled** | **1,267 tok/s** | path opens |
| **largest prompt actually served** | **5,386 tokens** | **8,268**, every size served | path opens |
| C1 aggregate · per stream | 47.41 · 55.73 | 47.30 · 51.34 | equal · −7.9 % (see below) |
| C2 / C4 / C6 aggregate | 68.27 / 91.65 / 113.22 | 68.72 / 93.74 / 117.71 | +0.7 / +2.3 / +4.0 % |
| **C8 aggregate** | 127.54 | **135.59** | **+6.3 %** |
| **TTFT median, C1 / C8** | 0.621 / 1.703 s | **0.478 / 1.244 s** | **−23 % / −27 %** |
| acceptance · tokens per step, C1 | 64.48 % · 5.51 | 62.56 % · 5.38 | −1.9 points |
| gates, cold **and** warm | 10/10 · 12/12 · 0 empty | 10/10 · 12/12 · 0 empty | equal |
| boot, cold, no sidecar | 396 s | 375 s | −5 % |

The direction and size match three ranks where they should: there the same change bought +82 % of
pool, +6 % at C8 and 20–30 % off TTFT ([07](07-kv-and-draft-page.md) §3).

### 4.3 The cliff, which is the real finding

The control does not serve a long prompt *slowly*. It does not serve it at all. Walking prompt length
up one request at a time, 75-second budget `[measured-here]`:

| prompt tokens | control | fix |
|---|---|---|
| 913 / 1,786 / 2,759 / 3,586 / 4,444 / 5,386 | served, 1.5–4.1 s | served |
| **6,253** | **never scheduled** | served, 4.9 s |
| 7,329 / 8,268 | — | served, 5.5 / 6.3 s |

`Running: 0 reqs, Waiting: 1 reqs, GPU KV cache usage: 0.0 %`, indefinitely — one request wants 640
of the pool's 385 blocks, and block ids are global to one pool, so it is never admitted. After the
fix it wants 280 of 365.

**And the control sits on the edge of that cliff by luck.** Arm C got 426 blocks and did serve a
7,382-token prompt at 1,135 tok/s; the same env file booted through a settle gate on 6 September
gets 385 and cannot. The difference is the instrument, not the stack. **Without the page fix,
whether two ranks can serve an 8K prompt depends on the host's state at boot.**

### 4.4 What it cost

- **Per-block memory rises about 9.1 %** — the drafter's per-block cost goes 163,840 → 2,621,440 B.
  At three ranks the price was +9.2 %.
- **The draft group's prefix-cache matching unit coarsens 16 → 256 tokens.** Nothing measurable here:
  with a 4,608-token attention block the prefix-cache hit rate is 0 % in both arms.
- **Acceptance falls 1.9 points**, inside this stack's 60–65 % boot-to-boot band.
- **C1 per-stream decode reads 7.9 % lower**, the one number that moved the wrong way. The fix arm's
  own three rounds span 9.3 %, the C1 aggregate is equal and TTFT is 23 % better. No C1 gain, and no
  proven C1 loss. **Candidate A, one image later and on a second boot, reads 54.72 per stream** —
  back inside the control's band, which is what "not established" looked like.

---

## 5. The TP=2 production candidate — measured, 6 September 2026

Two complete candidates, identical in every respect except the checkpoint and the one patch and one
flag that serving it needs. **Candidate B, the full-scope one, is the better of the two**:
it is faster on every concurrency, has a 42 % larger pool, is lighter in host memory, and its
quality gates are indistinguishable. Candidate A is kept for anyone who has the 164 GB
routed-experts-only checkpoint and does not want to fetch the other one.

**§5.9 adds a third, and it is the one to run.** Candidate C is candidate B plus one environment
line — the sparse-indexer workspace bound — and it is **+26.5 % of pool** for no measured quality
cost. Everything §5.1–§5.8 says about candidate B applies to it unchanged; only the pool, the memory
rows and the sidecar directory move.

### 5.1 What both candidates are

Two nodes — the head node at rank 0, worker-1 at rank 1 — TP=2, **EP off**, image `exl3-zeus:754421f`,
`tracks/tp2/patches/`, `scripts/start-tp2full.sh` with its settle gate, KV `fp8`, **fp8 draft cache**,
DFlash2 k=7, `--attention-backend CUSTOM`, `--block-size 256`, `HAREM_SW_BLOCK_SIZE=256`,
`--max-num-seqs 8`, `--max-num-batched-tokens 2048`, **`--max-model-len 1000000`**,
`gpu-memory-utilization 0.85`, `HAREM_DISABLE_PERSISTENT_TOPK=1`, `NCCL_MAX_NCHANNELS=8`, mesh plugin
`patched2` with both cables and `NCCL_PTR_CUDA`, warm `CUDA_EXL3_TUNE_CACHE`, per-rank fast-load
sidecar, `--safetensors-load-strategy eager`, `--no-enable-flashinfer-autotune`, temperature 0,
reasoning effort **low**. Speed is the median of rounds 2–4 of a four-round sweep; round 1 is
reported separately as the tuner-cache check (§5.6).

The only differences between them:

| | **A — experts-only** | **B — full-scope (recommended)** |
|---|---|---|
| checkpoint | `brandonmusic/GLM-5.3-Flash-tr3-4bpw` @ `b20c49ba`, 164 GB | `turboderp/GLM-5.3-Flash-exl3` @ 4.05 bpw, 154 GB |
| `HAREM_EXL3_FULLSCOPE` | unset | `1` → `patch-fullscope-tp2.py` |
| `--hf-overrides` | absent | `{"quantization_config_file":"…/quantization_config.json"}` |
| what is 4-bit | routed experts only; attention, shared expert and `lm_head` stay BF16 | all of it — `CUDA_EXL3_DEBUG_NAMES=1` prints **203 EXL3 / 113 bf16** modules against A's 0 EXL3 in the same places |

### 5.2 The numbers

| | **A — experts-only** | **B — full-scope** | B vs A |
|---|---|---|---|
| **KV pool at `max_model_len` 1,000,000** | 1,500,000 (**1.50×**) | **2,128,571** (**2.13×**) | **+41.9 %** |
| `num_blocks` · blocks/request | 420 · 280 | **596** · 280 | +41.9 % |
| Available KV memory, rank 0 / rank 1 | 11.70 / 11.34 GiB | **16.07 / 16.23 GiB** | +37–43 % |
| Consumed memory per node (weights + non-torch) | 89.30 / 89.23 GiB | **84.77 / 84.51 GiB** | **−4.5 GiB** |
| **C1 aggregate** (tok/s) | 48.76 | **58.50** | **+20.0 %** |
| **C1 per stream** | 54.72 | **62.55** | **+14.3 %** |
| C2 aggregate | 68.24 | **82.45** | +20.8 % |
| C4 aggregate | 96.52 | **112.62** | +16.7 % |
| C6 aggregate | 115.69 | **137.37** | +18.7 % |
| **C8 aggregate** | 137.41 | **155.75** | **+13.3 %** |
| C8 per stream | 20.90 | 22.17 | +6.1 % |
| **TTFT median, C1 / C8** | 0.468 / 1.249 s | **0.407 / 1.077 s** | −13.0 % / −13.8 % |
| acceptance · tokens per step, C1 | 65.47 % · 5.58 | 59.98 % · 5.20 | −5.5 pt, see the cost line |
| acceptance · tokens per step, C4 / C8 | 63.60 · 5.45 / 62.12 · 5.35 | 63.67 · 5.46 / 62.84 · 5.40 | equal |
| cold first request | TTFT 0.78 s, 37.7 tok/s, acc 41.1 % | TTFT 0.76 s, **50.8** tok/s, acc 42.8 % | +35 % |
| **Prefill, 3 fresh unseen ~8.4K prompts** | **1,444** tok/s | 1,400 tok/s | −3.0 %, equal |
| Prefill, 7,382 tokens, uncached | 1,241 tok/s | **1,289** tok/s | +3.9 %, equal |
| **Boot, fast-load** | **272 s** | **272 s** | identical |
| Boot, one-off dump (writes the sidecar) | 997 s | 998 s | identical |
| Sidecar size per rank | 82.6 GiB (36 files) | **75.2 GiB** (32 files) | −9 % |
| weight restore time | 90.1 s, 952 MB/s | 88.0 s, 918 MB/s | equal |
| Correctness probe / code exam, **cold and warm** | 10/10 · 12/12 both | 10/10 · 12/12 both | equal |
| **Tool-call gate** | **8/8** | **8/8** | equal |
| **MMLU sample**, 57 × 35 = 1,995 q, 0-shot | **86.37 ±0.74** | **86.02 ±0.75** | −0.35 pt, **inside one error bar** |
| **Needle-lite**, 64K and 128K, three depths | **6/6** (80,113-token prompts served) | **6/6** (same prompts) | equal |
| `MemAvailable` / swap after the full run | 6.0 / 7.2 GiB · 0.03 / 0.02 GiB | 5.7 / 7.0 GiB · 0.03 / 0.02 GiB | swap flat in both |

### 5.3 What the fp8 draft cache did, isolated by arithmetic

`HAREM_DRAFT_KV_DTYPE=fp8` was `[not tested]` at two ranks until now. It is in both candidates, so it
is not isolated as an A/B arm — but the engine's own decomposition isolates it exactly, because it
changes one number and nothing else. `SlidingWindowSpec` bytes-per-block goes **2,621,440 →
1,310,720 B** while its 25 blocks-per-request and every other group stay identical, so
blocks-per-request stays 280 and the freed memory turns straight into `num_blocks`:

| | draft cache `auto` (§4 fix arm) | **draft cache fp8** (candidate A) |
|---|---|---|
| `SlidingWindowSpec` bytes/block | 2,621,440 B | **1,310,720 B** |
| blocks per request | 280 | 280 |
| `num_blocks` | 365 | **420** |
| **KV pool** | 1,303,571 | **1,500,000** |

**+15.1 % of pool** `[measured-here]`, against +5.6 % at three ranks — larger here for the same
reason the page defect was larger, a smaller denominator. Acceptance and accepted-tokens-per-step are
inside their band and the gates are unchanged. The two arms are one image apart (`62f53e6` →
`754421f`), so the *pool* attribution is exact arithmetic but the *speed* comparison across them is
not; we do not make one.

One behavioural note worth recording: at three ranks the fp8 draft cache pushed the drafter onto a
FlashInfer path and **CUDA graphs stopped capturing**. At two ranks they capture normally — 19
PIECEWISE and 8 FULL graphs in the boot log of both candidates `[measured-here]`. The asymmetry is
not FlashInfer's capability but its support gate: it divides the *target's* per-rank heads by the
*draft's* KV heads — 32 % 4 = 0 here, 22 % 3 = 1 at three ranks — while the kernel it actually runs
(XQA) is the same on both. Filed upstream as [vllm#55581](https://github.com/vllm-project/vllm/issues/55581) and priced in
[11](11-open-issues.md) §2.29. **When per-stream speed is compared across the two tracks, this is a
≤2 % term in TP=2's favour** `[estimate]`, and the 1.10 GiB graph pool is a term against its KV.

### 5.4 Both cables of the pair, measured

`port_xmit_data` deltas across the four-round C1–C8 sweep, read on both nodes
([06](06-nccl-mesh.md) §6 is the method) `[measured-here]`:

| node | device | candidate A | candidate B |
|---|---|---|---|
| head | `rocep1s0f0` | 92,299 MB | 91,736 MB |
| head | `roceP2p1s0f0` | 90,180 MB | 89,544 MB |
| head | `rocep1s0f1`, `roceP2p1s0f1` | 0.0 | 0.0 |
| worker-1 | `rocep1s0f1` | 92,061 MB | 91,495 MB |
| worker-1 | `roceP2p1s0f1` | 89,944 MB | 89,354 MB |
| worker-1 | `rocep1s0f0`, `roceP2p1s0f0` | 0.0 | 0.0 |

**Two devices per node move ~90 GB each and the split is 50.5 / 49.5.** That closes the question
[06](06-nccl-mesh.md) left open for a *single* peer pair: the dual-cable patch (`patches/kernel/0005`)
is not a three-node effect. The zero rows are the ports facing the third node, which is not in this
cluster — exactly what `FABRIC_PEERS` with one address per node expects.

### 5.5 The reversal: full-scope at `max_model_len` 1,000,000

Arm D could not boot there: `6.6 GiB KV needed for max seq len 1,000,000, available 0.73 GiB`.
Candidate B boots there with **16.07 GiB available** and a 2,128,571-token pool. The gap is 22×, so
it is worth decomposing rather than asserting `[measured-here]`:

| term | effect | what it does to arm D's failure |
|---|---|---|
| `HAREM_SW_BLOCK_SIZE=256` | blocks/request 640 → 280 | the 6.6 GiB requirement becomes ~2.9 GiB |
| the launcher's **settle gate** | `MemAvailable` 116.9–117.1 GiB at snapshot time on both nodes | 0.73 GiB available becomes 16.07 GiB |
| `HAREM_DRAFT_KV_DTYPE=fp8` | `num_blocks` +15.1 % at fixed memory | pool, not admission |

**The settle gate is the larger half, and it is not a tuning knob — it is a measurement bug fix.**
Arm D snapshotted a host that had not finished reclaiming the previous container, and
[07](07-kv-and-draft-page.md) §1.1 already says a rank that does that writes itself memory it does
not have. What is new is how large the error can be: it was the difference between "this
configuration cannot serve" and "this configuration is the recommended one".

### 5.6 The tuner cache at two ranks

Four rounds per candidate, warm `CUDA_EXL3_TUNE_CACHE`. Round 1 against the median of rounds 2–4
`[measured-here]`:

| | A, C1 | A, C8 | B, C1 | B, C8 |
|---|---|---|---|---|
| round 1 | 47.25 | 135.54 | 60.14 | 152.63 |
| median of rounds 2–4 | 48.76 | 137.41 | 58.50 | 155.75 |
| round 1 error | −3.1 % | −1.4 % | **+2.8 %** | −2.0 % |

Unordered and inside ±3 %, which is what [12](12-tuner-cache.md) §4 measured at three ranks — round 1
is no longer a penalty, and it is not a bonus either. Three tune events were logged on the whole
boot. **Three rounds is enough at two ranks too.** The observed spread of rounds 2–4 was 1.3–6.9 %
for candidate A and 2.0–3.1 % for candidate B, so the "treat differences under about 5 % as noise"
rule stands here as well.

### 5.7 The autostart unit, tested

**Every figure in this section is candidate B's.** Candidate C's unit is the same unit, pointing at
the same launcher with a different environment file and a different fast-load sidecar — one
`EXTRA_ENV` word between them — so nothing here changes for it, but **it was not re-timed under the
unit** `[not tested]`. The 272 s in §5.9 is a `docker run` fast-load boot, not a `systemctl start`.

`tracks/tp2/harem-exl3-tp2.service` and `tracks/tp2/motor-onkosul-exl3-tp2.sh` installed on both nodes,
`daemon-reload`, then `systemctl start` (worker-1 first) → `/health` → gates → `systemctl stop`, with
the three-node unit stopped throughout `[measured-here]`:

| | |
|---|---|
| `systemctl start` returned | **3 s / 6 s** — the preflight is 1 s |
| **`/health` 200** | **+261 s** from the first `systemctl start` |
| KV pool on that boot | **2,153,571**, +1.2 % against the hand-started boot of the same env |
| correctness probe · code exam | **10/10** · **12/12** |
| `systemctl stop` | clean on both; units `inactive`, no containers left |

**The first attempt failed, and it failed correctly.** The unit had been installed before its
`ENV_FILE` was pointed at the full-scope environment file, so the preflight looked for the *other*
candidate's fast-load sidecar — deleted to make disk room — and refused in one second with
`fast-load sidecar missing`, before docker was touched. That is check 7 of the preflight doing the
job it exists for `[measured-here]`.

It is deliberately left **`disabled`**: on our cluster `harem-exl3.service` (three nodes) is the
enabled autostart, and the two units carry `Conflicts=` for each other precisely so that only one can
ever run.

**We have not run a two-node reboot test** `[not tested]`. The three-node one is in
[systemd/README](../systemd/README.md).

### 5.8 Quality, side by side

| gate | A — experts-only | B — full-scope | TP=3 production 10, for reference |
|---|---|---|---|
| correctness probe, cold / warm | 10/10 · 10/10 | 10/10 · 10/10 | 10/10 |
| code exam, cold / warm | 12/12 · 12/12 | 12/12 · 12/12 | 12/12 |
| tool-call gate | **8/8** | **8/8** | — |
| MMLU sample, 1,995 q, 0-shot | **86.37 ±0.74** | **86.02 ±0.75** | 86.47 ±0.74 |
| needle-lite, 64K + 128K × 3 depths | **6/6** | **6/6** | 20/20 at 1M (the full suite) |

Every MMLU figure this stack has produced at either rank count — 86.02, 86.32, 86.37, 86.4, 86.47 —
sits inside a single error bar of every other. **Quality is equal, so the choice between the two
candidates is made on speed and pool, and both favour B.**

The needle probe serves **80,113-token prompts** at three depths in both candidates. That is the
long-prompt path of §4 open at full length rather than merely at 8K.

Full raw record: [`results/speed/tp2-production-candidate.md`](../results/speed/tp2-production-candidate.md).

---

### 5.9 Candidate C — the indexer workspace bound at two ranks, measured 6 September 2026

**The recommended configuration.** Candidate B plus `HAREM_INDEXER_WS_MODE=bound`, the patch the
three-node track adopted the same day ([`tracks/tp3/patches/indexer-workspace/`](../tracks/tp3/patches/indexer-workspace/),
[`results/memory/indexer-workspace-ab.md`](../results/memory/indexer-workspace-ab.md)). Nothing else
changes: same checkpoint, same image, same kernels, same 0.85.

**Why it was worth measuring separately rather than assumed.** vLLM sizes the sparse indexer's
K-gather workspace as `40 × max_model_len` **entries** — 4.92 GiB at `max_model_len` 1,000,000,
reserved during the profile run, locked for the life of the engine, and charged to the residual the
profiler subtracts *before* it sizes the KV pool. **Every term of that expression is per engine, not
per rank.** So two ranks reserve exactly the same 4.92 GiB per rank as three do — against a pool
that is a third the size. The bound is arithmetic, not a rank-count question, and the prediction
made before the run was **+27.5 %** against the three-node track's measured +10.25 %.

#### The workspace itself

`VLLM_DEBUG_WORKSPACE=1` is an upstream variable and needs no patch. One resize per rank, on both
ranks, in both arms — so no other consumer sets this buffer and the gain is not capped by somebody
else's requirement:

| Arm | `WorkspaceManager` line | Grown by |
|---|---|---|
| control | `0.00 MB -> `**`5036.40 MB`** | `sparse_attn_indexer_kpool.py:295` |
| bound | `0.00 MB -> `**`513.00 MB`** | `sparse_attn_indexer_kpool.py:295` |

**Both numbers are the three-node track's, to the decimal.** The patch's own startup line is
identical too — `max_model_len=1000000 compress_ratio=4 entry_bytes=132 max_num_seqs=8 num_spec=7`,
`per_request_floor=250002`, `scheduler_ceiling=2000016`, `headroom=2.03x`. The file needed no
two-node variant.

#### The A/B, one environment line apart `[measured-here]`

Both arms booted **eagerly** — no fast-load — because a new `patch-*.py` and a changed prelude
change the sidecar's manifest identity, and when the arms ran there was not enough free disk for a
second two-node sidecar. The comparison is therefore control-versus-bound on the same boot path,
never against the §5.2 table:

| | **control** (knob unset) | **bound** | delta |
|---|---:|---:|---:|
| Locked workspace | 5,036.40 MB | **513.00 MB** | −4.42 GiB |
| **KV pool at 1M** | 1,800,000 | **2,378,571** | **+578,571, +32.14 %** |
| Maximum concurrency at 1M | 1.80× | **2.38×** | +32 % |
| Available KV, rank 0 / rank 1 | 13.65 / 13.60 GiB | 18.20 / **17.96** GiB | +4.55 / **+4.36** |
| Consumed memory per node | 84.68 / 85.00 GiB | 79.50 / 80.40 GiB | −4.23 / −4.60 |
| Peak activation | 5.06 / 4.79 GiB | 4.74 / 5.02 GiB | unchanged |
| CUDAGraph memory | 1.07 / 1.10 GiB | 1.10 / 1.10 GiB | unchanged — graphs still capture here, vllm#55581 |
| Boot, eager | 333 s | 333 s | identical |
| C1 / C2 / C4 / C6 / C8 aggregate | 61.91 / 85.15 / 121.13 / 142.68 / 160.43 | 60.76 / 84.16 / 117.94 / 139.40 / 157.15 | −1.86 / −1.16 / −2.63 / −2.30 / −2.04 % |
| C1 per stream | 67.09 | 67.46 | +0.55 % |
| TTFT, C1 / C8 | 0.372 / 1.047 s | 0.381 / 1.036 s | equal |
| acceptance, C1 / C8 | 61.13 / 62.10 % | 60.39 / 61.29 % | equal |
| Prefill, 3 fresh unseen ~8.4K prompts | 1,413 tok/s | 1,403 tok/s | −0.7 % |
| Correctness / code, **cold and warm** | 10/10 · 12/12 both | 10/10 · 12/12 both | equal |
| Tool-call gate | 8/8 | 8/8 | equal |
| Needle-lite, 64K + 128K × 3 depths | 6/6 | 6/6 | equal |
| Swap used, peak | 0.00 GiB | 0.000 GiB | flat |

**The conversion checks out against an independently measured ruler.** 578,571 tokens ÷ 4.36 GiB on
the binding rank = **132,700 tokens/GiB**, against the 132,456 the candidate-B boot gives from
2,128,571 ÷ 16.07. Two derivations, one number.

**Why three times the three-node gain.** The memory freed is the same on both — about 4.4 GiB per
rank. The pool it is freed *into* is 13.6 GiB at two ranks and 44.5 GiB at three. The whole
difference between +32.14 % and +10.25 % is the denominator.

**A control worth having.** The control arm's 1,800,000 is within **0.98 %** of the 1,817,857 that
candidate B's own dump boot — the same eager path — produced hours earlier. The arms are comparable
and the control is not a one-off.

#### The stress the buffer actually has to survive

| Gate | Result |
|---|---|
| One **~1M-token** request | 969,468 prompt tokens, needle correct — see the harness note below |
| **Eight concurrent ~128K** prompts, each with its own needle | **8/8**, 640,904 prompt tokens, 288.0 s wall, 2,225 tok/s aggregate prefill |
| A resize after `lock_workspace()` | **none** — exactly one resize line per rank, the boot one |
| The patch's four safety layers | **none fired** — no `AssertionError`, no `HAREM-IDXWS refuses`, no `K-gather workspace too small` |
| `MemAvailable` floor / swap | 7.0 and 8.0 GiB / **0.000 GiB**, swap-out 9 and 7 pages |

The eight-lane gate is the one that matters: the buffer holds *one indexer chunk's* compressed
context for every prefill request the scheduler batches together, so many long prompts at once is
its worst case, not one long prompt. Each lane carried a **different** needle, so a gather that read
past its slice would surface as a **wrong code**, not as a plausible number. It did not.

#### The measuring instrument was wrong before the engine was `[measured-here]`

The 1M gate first read **FAIL**: 969,468 prompt tokens, 660.6 s, and an **empty** answer — not a
wrong one. The probe scored `message.content` only. At `reasoning_effort: low` this model sometimes
puts a short answer entirely in `reasoning_content` and leaves `content` empty; that is a measured
property of the extractor, not a miss. The identical request, same seed, same settings, re-run:
**PASS**, the code in `content`, 662.7 s.

Recorded honestly: **two attempts, one empty, one correct, and no wrong code in either.** The first
attempt's `reasoning_content` was discarded by the probe, so the artefact cannot be *proved* for that
specific request — only that the failure had the artefact's shape. The fixed harness scores both
fields and prints which one hit:
[`bench/needle-1m-bothfields.py`](../bench/needle-1m-bothfields.py). A probe that reads one of two
fields is a ruler with a missing tick, and this is the second time on this project that the
instrument, not the engine, was the thing that failed.

#### One reading recorded as unexplained, not as noise

All five concurrency levels moved the **same way**: −1.16 %, −1.86 %, −2.04 %, −2.30 %, −2.63 %,
mean −2.0 %. Every one is inside its declared band, and the per-stream C1, the TTFT and the
acceptance rows are all flat. But at three ranks the same patch gave **mixed** signs (+1.4, −1.0,
+0.8, +0.9, −1.5 %; mean +0.1 %), and five out of five in one direction is not what noise usually
looks like.

**We cannot attribute it, and we are not going to call it noise.** No clock, temperature or power
telemetry was sampled during either arm — only `MemAvailable`, swap and `vmstat`. What `vmstat`
shows is that neither arm was CPU-starved (87–93 % idle, run queue 1.4–2.7), but the two windows are
not the same length (23 minutes against 54, because the stress gates ran in the bound arm), so even
that is not a like-for-like reading. A second confound is **order**: the control ran first and the
bound arm started 25 minutes deeper into a loaded cluster. Thermal drift is a plausible hypothesis
and an unmeasured one.

**Verdict: same sign, inside band, unexplained.** If you repeat this, reverse the arm order and
sample clock, temperature and power per arm `[not tested]`.

#### The production configuration, with the sidecar back

Once the disk was found, candidate C got its own dump and a fast-load boot:

| | Candidate B | **Candidate C** | delta |
|---|---:|---:|---:|
| **KV pool at 1M** | 2,128,571 | **2,692,857** | **+564,286, +26.5 %** |
| Maximum concurrency at 1M | 2.13× | **2.69×** | +26.5 % |
| Available KV, rank 0 / rank 1 | 16.07 / 16.23 GiB | 21.31 / 20.33 GiB | +5.24 / +4.10 |
| Consumed memory per node | 84.77 / 84.51 GiB | 79.50 / 80.35 GiB | −5.3 / −4.2 |
| Boot, fast-load | 272 s | **272 s** | identical |
| Weight restore | 88.0 s, 918 MB/s | 92.4 / 86.4 s, 873 / 934 MB/s | equal |
| Sidecar per rank | 75.2 GiB, 32 files | 78 GB, 32 files | same content |
| One-off dump boot | 998 s | 956 s | equal |
| C1 aggregate · per stream | 58.50 · 62.55 | 60.08 · 65.96 | different session, §5.9 note |
| C8 aggregate | 155.75 | 157.71 | different session |
| Gates cold + warm · tool-call · needle-lite | 10/10 · 12/12 · 8/8 · 6/6 | 10/10 · 12/12 · 8/8 · 6/6 | equal |

**The pool figure was predicted before it was measured.** Candidate B's own eager→fast-load
difference is 2,128,571 − 1,817,857 = **+310,714** tokens; adding it to the bound arm's eager
2,378,571 predicts **2,689,285**. Measured: **2,692,857** — **+0.13 %**. The two speed columns are
**different sessions and not an A/B**; the clean comparison is the control-versus-bound table above.

#### What it cost, and the line is not left empty

1. **Diagnostic margin.** The indexer had 20× the largest load the scheduler can produce and now has
   **2.03×**, against a startup refusal and three run-time guards, none of which fired here.
2. **Disk and one dump boot.** A fresh 78 GB-per-rank sidecar and 956 s. Adding a file to a patch
   tree changes the fast-load manifest identity, so this is not optional — budget it.
3. **The unexplained speed sign** above. Not free until it is explained.
4. **MMLU was not re-run** `[not tested]`. Candidate C changes no weight and no kernel — only the
   size of a scratch buffer — and the short quality gates were taken as sufficient. Candidate B's
   86.02 ±0.75 on the same checkpoint and stack is the standing figure.

---

## 5b. The vision tower at two ranks — divisible, and still not free `[not tested]`

Two ranks is the easy case for *splitting* the tower and it changes nothing about *loading* it.

**Divisibility is clean.** Heads 16/2 = 8, `attn.proj` 512 = 4 × 128, the MLP 2048 = 16 × 128, the
merger 2048 = 16 × 128 and its context 5120 = 40 × 128. No padding, and
**`--mm-encoder-tp-mode data` is not needed** — that flag is TP=3's own, because at three ranks
nothing in the tower divides at all ([18](18-vision-at-three-ranks.md) §1.1).

**What you still need depends on the checkpoint, not the rank count.**

| Checkpoint | What it takes |
|---|---|
| `brandonmusic/GLM-5.3-Flash-tr3-4bpw` (routed experts only) — the tower is dense BF16 | `LANGUAGE_MODEL_ONLY=0` and the per-request limits, plus VS4/VS6/VS7 for video. Nothing else |
| `turboderp/GLM-5.3-Flash-exl3` at 4.05 bpw (full scope) — **the recommended two-node candidate**, §5 | The **same** VS1/VS2/VS3 and the same `CUDA_EXL3_PACKED_MAPPING` as at three ranks: the tower is 6-bit EXL3 and vLLM builds it with `quant_config=None` regardless of how many ranks you have |

**And the video half is needed at any rank count**, because the frame-sampler mismatch it fixes is
upstream's and has nothing to do with sharding ([18](18-vision-at-three-ranks.md) §4;
[vllm#55644](https://github.com/vllm-project/vllm/issues/55644)).

Neither row has been run. Both are readings of code against measured checkpoint facts `[not tested]`.
If you run one, [HELP-WANTED](../HELP-WANTED.md) says what we would want reported.

---

## 6. What is still **not** measured at two ranks

This list is much shorter than it was, and every row now carries a reason rather than an omission.

| Not run at TP=2 | Why not, and what it would take |
|---|---|
| **The `gpu-memory-utilization` ladder.** Every TP=2 arm ran **0.85** | This is the one deliberate refusal. KV maximisation is the last step in this project's order of work and it needs the cluster's owner, not an agent. It also has a two-node-specific warning attached: arm A recorded **3.5 GB of swap on the head during weight load** at this rung (§3.1), and the three-node ladder was climbed rung by rung on 6 September against swap *traffic* under load, which took it to 0.88 ([11](11-open-issues.md) §2.4). The three-node ladder must be **re-derived** at two ranks, not copied — the memory left after a fixed cost is not linear in the node count `[not tested]` |
| **A two-node reboot test** | The unit is installed and start/stop-tested (§5.7), but a power-on trial takes the cluster down and the three-node unit is the enabled production autostart. Reboot **both** nodes or neither: the preflight passes on a single node whose peer is gone `[not tested]` |
| **Expert parallelism at two ranks** | Legal (§1.1), never measured. It changes which kernel path the MoE stage takes ([05](05-expert-parallel-and-cuda-exl3-fixes.md)) and `tracks/tp2/patches/` already carries `patch-epfilter-tp3.py` so that trying it needs no tree change — and therefore no new dump boot `[not tested]` |
| **A second boot of each candidate**, and a boot-to-boot spread | Every number in §5 is a median of three rounds on **one** boot per candidate. At three ranks the boot-median spread is C1 1.1 %, C8 2.5 %, **C4 7.4 %**. The pool, memory, boot-time and gate rows are far too large a difference to be boot noise; the speed rows carry that uncertainty and the B-over-A margins (+13 to +21 %) clear it comfortably `[not tested]` |
| **`patch-vllm-tp3.py` at two ranks** | A no-op by arithmetic (§1.1), deliberately not shipped, therefore never measured. If you keep one tree for both rank counts you will run it; we do not expect a difference and we have not shown one `[not tested]` |
| **Why the bounded arm's five concurrency levels all moved the same way** (§5.9) | Inside band, mean −2.0 %, and mixed-signed at three ranks. No clock, temperature or power telemetry was sampled during either arm, and the arms ran in a fixed order 25 minutes apart, so neither thermal drift nor anything else can be ruled in or out. Repeat with the arm order reversed and per-arm telemetry `[not tested]` |
| **MMLU on candidate C** | It changes no weight and no kernel — only a scratch buffer's size — and the short gates were taken as sufficient. Candidate B's 86.02 ±0.75 on the same checkpoint and stack stands `[not tested]` |
| **Anything at one or four nodes** | Out of scope for this page; [00-start-here](00-start-here.md) says what we can and cannot say about other node counts |

Two rows that used to live here are gone: the fast-load sidecar and the dual-cable plugin patch are
both measured at two ranks now (§5.2, §5.4), and so are the tuner-cache protocol (§5.6), the fp8
draft cache (§5.3), the autostart unit (§5.7) and the draft KV page (§4).

---

## 7. The trade-off, measured

Three nodes still win. On this stack the third node is not only a bigger memory pool — it is faster,
per stream as well as in aggregate, and that is the opposite of what "fewer ranks, fewer collectives,
lower latency" would predict.

Like for like where it can be: both sides on the **full-scope** checkpoint, DFlash2 k=7, the same
harness and the same protocol. The rank counts run at different memory rungs — 0.85 at two nodes,
0.83 at three — because neither ladder is transferable; that difference is called out in the pool row
`[measured-here]`:

| | **TP=2 candidate C** (2 nodes, 0.85) | **TP=3 production 10** (3 nodes, 0.83) | two-node share |
|---|---|---|---|
| C1 aggregate | 60.08 | 70.5 | 85 % |
| C1 per stream | 65.96 | 76.9 | 86 % |
| C8 aggregate | 157.71 | 194.0 | 81 % |
| Prefill, fresh unseen ~8.4K prompts | 1,414 | 1,769 | 80 % |
| **KV pool at 1M** | **2,692,857** | **5,619,834** | **48 %** — and the two rungs differ |
| TTFT, C1 / C8 | 0.381 / 1.054 s | 0.280 / 0.826 s | +36 % / +28 % worse |
| Consumed memory per node | 79.5–80.4 GiB | 58.3–59.1 GiB | the whole story |
| Boot, fast-load | 272 s | 251 s | comparable |
| Sidecar per rank | 78 GB | 53 GiB | EP is why |
| Quality gates · tool-call | 10/10 · 12/12 · 8/8 | 10/10 · 12/12 | equal |
| MMLU sample | §5.8 (candidate B's, carried) | 86.47 ±0.74 | |

**One asymmetry to read the pool row with.** The two-node column carries the indexer workspace
bound (§5.9) and production 10 does not — it predates it. The three-node figure that does carry it
is production 12, whose headline pool is 7,041,322, against which the two-node share is **38 %** rather than
48 %. The speed rows are unaffected: the bound costs nothing measurable at either rank count.

**Why the third node also wins on latency.** A decode step here is weight-bandwidth bound, and the
profile says so: at production 7 the dense BF16 GEMM alone was 45.3 % of a single-stream step, and in
the engine the MoE trellis GEMM runs at **78–85 % of a measured 225 GB/s** ruler
([10](10-results-and-roofline.md) §5). Adding a rank cuts each rank's weight traffic by a third. The
collective it costs in exchange is real but bounded: on production 9, NCCL and the CPU gap
**together** are at most **17.19 ms of a 72.5 ms step** with the profiler off `[measured-here]`. The
bandwidth term wins, and it is not close.

**What changed on 6 September is the size of the gap, not its direction.** The two-node share of
single-stream throughput was 85–91 % before, but measured against a three-node arm on a *different*
checkpoint; on the same checkpoint and with both stacks carrying the settings this repository ships,
it is 81–83 % of speed and **38 % of pool** rather than 14 %. Two nodes went from a configuration
that silently refused 8K prompts to one that serves **2.7 concurrent million-token requests**.

**What TP=2 wins, plainly.** One node, and a recipe with the whole of
[03](03-tp3-padding-and-sidecars.md) and [13](13-full-scope-checkpoint.md) §7 taken out of it. That
is a real answer for anyone who owns two Sparks, and it is the whole reason this page exists — but it
is not a latency argument, and this repository will not make one.

---

## 8. If you run this at two ranks, tell us

[CONTRIBUTING](../CONTRIBUTING.md) has the format. The four things worth most from a two-node cluster,
in order:

1. **The memory ladder.** 0.85 is where all our arms ran and it is also where arm A found 3.5 GB of
   swap during weight load. Where the safe rung actually sits at two nodes is the largest open
   question on this page, and the one thing we deliberately did not touch.
2. **A second boot of candidate B**, so the speed rows in §5.2 stop resting on one boot each.
3. **A two-node reboot test** with `harem-exl3-tp2.service` enabled on both nodes.
4. **Expert parallelism on at two ranks.** Legal, never measured, and the tree is already ready
   for it.
