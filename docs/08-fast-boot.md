# 08 — Fast boot: 618 s to 274 s, and then to 251 s

**Applies to: both tracks.** The sidecar is per rank, so two ranks means two sidecars; we never built
one at TP=2 `[not tested]`.

Cold boot on this stack — container start to `Application startup complete` — took **617.9 s**. It now takes **251 s** on
production configuration 9, and it took **273.6 s** on the arm this page itemises; the weight-loading phase inside it went from
**426.3 s to 67.2 s** (6.3×), and to **57.9 s** once the checkpoint underneath it got smaller (§8.1). Speed, quality and the KV pool did not pay for it:
the gates are unchanged, the sweep is inside its own round-to-round spread, and the pool came out slightly **larger** than
before. This page is the account — how the boot was measured, the four changes in order, the proof that the fast path restores
bit-identical weights, the regression that had to be chased, what it costs, and how to undo any of it in one line.

Settings for every number here unless a row says otherwise: image `exl3-zeus:f4987cf`, TP=3 + expert parallel, EXL3 4bpw
checkpoint at HF revision `b20c49ba`, `kv-cache-dtype fp8`, DFlash2 draft at k=7, `--block-size 256`, `HAREM_SW_BLOCK_SIZE=256`,
`--max-num-batched-tokens 2048`, `--max-num-seqs 8`, `NCCL_MAX_NCHANNELS=8`, `--max-model-len 1000000`, temperature 0, reasoning
effort **low**, three nodes (`head` = rank 0, `worker-1`, `worker-2`), 5 September 2026. `gpu-memory-utilization` is **0.85** in
the baseline arm and **0.80** in every other arm; that difference is called out where it matters. Prerequisites: docs
[02](02-image-build.md), [03](03-tp3-padding-and-sidecars.md) and [07](07-kv-and-draft-page.md) working, plus
[09](09-measurement-protocol.md).

---

> **At TP=2 there is no sidecar and every boot on this stack has been a cold one** `[not tested]`. The
> sidecar is per rank, so two ranks means two of them at about 1.5× the size; the mechanism is
> rank-count-agnostic and should carry, but we never built one. It is most of why our two-node boots
> take 355–471 s against production 10's 251 s. The manifest-identity rule in §4 matters *more* at
> TP=2, not less: it was a TP=2 patch dropped into the TP=3 tree that refused a production boot.
> [15](15-tp2-track.md) §2.3 and §3.3.

## 1. Measure first: the boot ledger

Nothing was changed here before the boot was itemised, and you should not change anything either until you have your own ledger.
A boot that "feels slow" is not a finding; a phase table is. The ledger parses the container's own timestamped log against its
own `StartedAt` — so every number is the engine's clock, not the host's — anchoring on lines present in every arm (`Loading
weights took`, `GPU KV cache size:`, `Graph capturing finished`, `Application startup complete` and three more):

```
python3 scripts/boot-ledger.py --node head
```

Seven phases, four arms; all `[measured-here]`, raw ledgers in `results/boot/`. Headers carry each arm's memory share.

| Phase | Baseline (0.85) | BOOT1, S1+S2+S3 (0.80) | BOOT2, S4 verified (0.80) | **Production, S4 (0.80)** |
|---|---:|---:|---:|---:|
| 1 container + prelude + preflight + import + distributed init | 49.3 | 38.9 | 39.8 | **48.4** |
| 2 **main weight load** | **426.3** | **189.7** | **65.3** | **67.2** |
| 3 drafter (DFlash2) load + load close-out | 6.5 | 4.0 | 76.7 ¹ | 23.2 ² |
| 4 profile run (graph memory profile + first NCCL + MLA tune) | 67.3 | 67.2 | 67.5 | 73.1 |
| 5 KV pool → end of graph capture (warm-up + autotune + capture) | 53.8 | 50.3 | 48.6 | 49.3 |
| 5a  — of which FlashInfer autotune | *34.0* | *0 (disabled)* | *0* | *0* |
| 6 engine core close-out | 7.9 | 8.0 | 7.9 | 7.9 |
| 7 API server | 6.8 | 4.1 | 4.3 | 4.3 |
| **Total (s)** | **617.9** | **362.1** | **310.1** | **273.6** |
| `GPU KV cache size` (tokens) | 5,256,198 | 4,231,404 | 4,468,319 | **4,484,848** |
| Gates: correctness / code | 10/10 · 12/12 | 10/10 · 12/12 | 10/10 · 12/12 | 10/10 · 12/12 |

¹ BOOT2's phase 3 also carries the whole verification workload: a 64-tensor hash check on both models (16.8 + 14.8 s) and two
full post-processing hash dumps (23.0 + 18.7 s), about 73 s — paid for the proof in section 3, absent in production.
² Production: drafter restore 2.1 s plus a 32-tensor hash exam on each model (8.2 + 11.6 s). Gates were run cold **and** warm
in the three 0.80 arms.

Two caveats. The baseline ran at 0.85, so **its KV number is not comparable**; the KV argument in section 5 is made against a
0.80 reference arm carrying none of S1–S4, at **4,413,223** tokens. And phase 4 grew ~6 s in production, not understood and
listed in section 10. The ledger is also what makes the decision — phase 2 was **69 %** of a cold boot, so nothing else was
worth touching until it moved.

## 2. The four changes

Numbered in the order applied. Each: what it is, why, how to apply it, the line to look for in the boot log, what it cost. S1–S3
are flags, and the production env carries all three:

```
EXTRA_ARGS="--block-size 256 --enable-ep-weight-filter --safetensors-load-strategy eager --no-enable-flashinfer-autotune"
```

### 2.1 S1 — teach the EP weight filter about `.trellis`

**What and why.** `--enable-ep-weight-filter` plus a one-anchor prelude patch, `patches/tp3/patch-epfilter-tp3.py`. With expert
parallelism on, each rank owns 96 of 288 routed experts, so two thirds of every expert tensor it reads are discarded. vLLM has a
filter for exactly this, off by default — and switching it on still does nothing here, because upstream's `should_skip_weight`
only considers names ending in `.weight` or `.weight_packed`, while an EXL3 expert keeps **99.8 % of its bytes** in
`<proj>.trellis`. The patch adds that suffix, env-driven so the set can widen without a new patch (`HAREM_EP_FILTER_SUFFIXES`,
comma separated, default `.trellis` only). The tiny per-expert scale tensors `.suh` / `.svh` / `.mcg` are **deliberately still
read for every expert**: they are cheap, and a backend wanting a reduction over all experts' scales would break silently without
them. Single anchor, fails closed, idempotent.

**Evidence.** Upstream's line, then the patch's own counter, printed every 5,000 skips:

```
[default_loader.py:406] EP weight filter: ep_size=3, ep_rank=0, loading 96/288 experts
[harem-epfilter] skipped 20000 non-local expert tensors (suffixes ('.weight', '.weight_packed', '.trellis'))
```

**Cost.** Bytes read fall 163.58 → ~66.6 GiB (−59 %) and tensors touched 150,226 → ~51,100, about 24,800 skipped `.trellis`
tensors per rank, two thirds of the 37,152 in the checkpoint `[estimate]`; the log lines `[measured-here]`. **Clock gain: near
zero, and we say so.** The same idea at TP=2 skipped 99,072 expert tensors and the load time did not move — 424 s,
indistinguishable from the unfiltered 412–424 s `[measured-here, raw lost]` — and S1 went into BOOT1 with S2 and S3, never
isolated on its own clock. What it buys is page-cache pressure at the load peak cut to a third, which matters on unified memory
(section 5). Quality: none. S4 bypasses the checkpoint loader entirely, so S1 is **not on the production hot path**; it stays in
the env as the fallback.

### 2.2 S2 — `--safetensors-load-strategy eager`

**What and why.** One flag. The default path is `safe_open` + `get_tensor`, i.e. mmap: 4 KB page faults, one thread, no queue
depth; `eager` does a sequential `read()` of each shard into a buffer and deserialises from there. The baseline loader moved
163.58 GiB in 426.3 s = **393 MB/s** on an NVMe that another loader on the same machine drives at 3.1 GB/s. The `prefetch`
strategy is unavailable — the engine refuses it in the log, because the filesystem is not a recognised network FS and the
checkpoint exceeds 90 % of available RAM.

**Evidence, result and cost.** The phase-2 line itself, `[default_loader.py:430] Loading weights took 189.67 seconds`:
**426.3 s → 189.7 s** `[measured-here]`. This was also the experiment that settled the diagnosis: S2 is the hypothesis test for
"is the read path the bottleneck". It is; section 6 is the block-layer half of the same answer. Price: about +2.7 GiB of
transient host RAM at the load peak, one shard being 1.36 GiB `[estimate]`, plus 4.1 % of the KV pool — section 5, the
regression that had to be chased. Quality: none. Like S1 it is off the production hot path and stays as the fallback.

### 2.3 S3 — `--no-enable-flashinfer-autotune`, and a correction

**What, and why we expected free money.** One flag that skips FlashInfer's autotuning pass during warm-up. In the baseline that
pass took 34.0 s and its own log reported `Loaded 0 configs` and `Saved 0 configs (0 new, 0 from previous)`; with
`--attention-backend CUSTOM` (the `cuda-exl3` sparse MLA kernel) FlashInfer has nothing to tune, and the plan called this "−34.3
s of provably empty work".

**What the measurement said.** `[retracted]` — that judgement is wrong and this measurement replaces it. Turning autotune off
moved `Graph capturing finished` from **12 s to 44 s**: the autotuner was not only tuning, it was also doing JIT and kernel
warm-up that graph capture then had to do itself. Across the whole block (phase 5, KV pool → end of graph capture) the honest
figure is **53.8 s → 50.3 s, about 3.5 s**, not 34 s `[measured-here]`. The flag took effect where you would expect:
`[kernel_warmup.py:170] Skipping FlashInfer autotune because it is disabled.`

**Cost.** ~3.5 s gained, no measured loss: the speed gates (C1–C8) and the quality gates are clean in every arm carrying it. It
was kept because it is not negative, not because it is worth 34 s. The general lesson belongs in
[09](09-measurement-protocol.md): a stage that reports doing nothing may still be doing something for a later stage.

### 2.4 S4 — the per-rank pre-sliced sidecar (the real fix)

**What.** Once per `(model, TP, rank)`, the rank writes to disk exactly the tensors it holds when `load_weights` returns, in
canonical (row-major) byte order, plus a `MANIFEST.json`; every later boot copies them straight back instead of re-deriving them
from the full checkpoint. Files: `patches/tp3/harem_fastload.py` (dump/restore engine), `patches/tp3/harem_fastload_id.py` (the
identity, deliberately free of torch and vLLM imports so the engine and the preflight compute it with the *same* code),
`patches/tp3/patch-fastload-tp3.py` (installs both into the vLLM package, two anchors in `base_loader.py`),
`patches/tp3/preflight-fastload.py` (the early refusal). With `HAREM_FASTLOAD_MODE` unset the hook is **literally**
`self.load_weights(...)`, so the patch is inert unless the env asks for it.

**Why this design, and not a re-derivation of the slicing rules.** The slicing is not simple: heads 64 → 66, shared expert 2048
→ 2112, vocabulary padding, the EP map 96/288, and a `narrow` per projection (see [03](03-tp3-padding-and-sidecars.md)). A
script that re-derived all of that would be a second implementation of the loader, and second implementations drift. The sidecar
is instead the **output** of a normal full-checkpoint load, so "what the loader would produce" and "what is on disk" cannot
diverge by construction. That leaves exactly one open question — *is the restore identical* — and hashing answers it (section
3).

**What the manifest records.** Schema and `SIDECAR_FORMAT`; image tag; TP / EP / rank; the checkpoint identity (sha256 of its
small files, `SHA256SUMS` among them, the HF revision, and a hash of the 120 shards' name:size list, 175,642,157,752 bytes); the
sha256 of every `patch-*.py`, of `tp3-prelude.sh` and of the `cuda_exl3` overlay; the engine identity (vLLM
`0.1.dev20051+g487ecf187`, `cuda-exl3` 1.0.0, TP/PP/DP, EP, expert placement); each model's own configuration (path, dtype,
quantization, `hf_overrides`, revision); and **per tensor the shape, dtype, byte count, sha256 and shard name**, tied storages
stored once and referenced by `alias_of`.

**How.** Add four variables to `.env.tp3` on each node — derived per node with `sed`, never copied between nodes
([03](03-tp3-padding-and-sidecars.md)). The launcher translates `FASTLOAD_*` into `HAREM_FASTLOAD_*` inside the container and
mounts the rank directory at the **same path**: read-write for a dump, read-only for a load.

```
FASTLOAD_DIR=/var/tmp/glm53-exl3-tp3
FASTLOAD_MODE=load
FASTLOAD_VERIFY=32
FASTLOAD_SHARD_BYTES=2147483648
```

Produce the sidecar once with a dump boot (about 11 minutes), pass the gates, then set `FASTLOAD_MODE=load` the same way:

```
for h in head worker-1 worker-2; do ssh $h "sed -i 's/^FASTLOAD_MODE=.*/FASTLOAD_MODE=dump/' ~/exl3-zeus/.env.tp3"; done
```

**Evidence.** The prelude's preflight line, then the restore:

```
preflight-fastload: OK  rank=0 dir=/var/tmp/glm53-exl3-tp3-r0 models=['dflash2-draft-tp3', 'glm-5.3-flash-tr3-4bpw-tp3'] 55.53 GiB
[harem-fastload] restored 3741 tensors, 53.50 GiB from 23 shards in 64.3 s (893 MB/s, read=buffered)
[harem_fastload.py:594] Loading weights took 67.23 seconds
```

**Size and result.** Per node **56 GiB** — main model 53.50 GiB / 3,741 tensors / 23 shards, drafter 2.04 GiB / 94 tensors / 2
shards, `MANIFEST.json` 954 KB — leaving 370–374 G free. Phase 2 goes 189.7 s → **65.3 s** (BOOT2) and **67.2 s** in production
`[measured-here]`. The disk, the maintenance debt, the one-off dump boot and the per-boot verification are section 8.

## 3. The bit-identity proof

Two independent pieces of evidence, on all three ranks: the first says the bytes came back, the second says the bytes the
kernels actually read came back.

**(a) At load time, raw tensors.** A dedicated verification boot ran with `HAREM_FASTLOAD_VERIFY=all`, which re-hashes every
restored tensor against the manifest — and that manifest was written from a **full-checkpoint** load, not from the sidecar:

```
rank 0  verify OK 1475/1475 tensors re-hashed (53.50 GiB, 490.0 s)   restore 64.3 s
rank 1  verify OK 1475/1475 tensors re-hashed (53.50 GiB, 484.0 s)   restore 63.5 s
rank 2  verify OK 1475/1475 tensors re-hashed (53.50 GiB, 490.1 s)   restore 55.3 s
```

1,475 is the count of tensors with their own storage; the other 2,266 of the 3,741 names share a storage and are bound to the
same bytes through `alias_of`. So this is **all 53.50 GiB**, not a sample. `[measured-here]`

**(b) After `process_weights_after_loading`** — the tensors the kernels actually read, since that step repacks and fuses. A
full-checkpoint boot and a sidecar boot each wrote a hash dump of every parameter and buffer after it, and the two were
compared:

```
glm-5.3-flash-tr3-4bpw-tp3  rank 0/1/2: 400 tensors each (101 expert, 123 attention, 2 embedding, 83 norm, 91 other) -> NO DIFFERENCE
dflash2-draft-tp3           rank 0/1/2:  94 tensors each (66 attention, 3 embedding, 12 norm, 13 other)              -> NO DIFFERENCE
VERDICT: BIT-IDENTICAL
```

For the target that is a stratified sample of 400 out of 3,785 named entries per rank — stratified on purpose, so "we checked
400 tensors" cannot mean "we checked 400 layer norms". For the drafter it is all 94. `[measured-here]`

**(c) The ordinary gates.** `scripts/correctness-probe.py` **10/10** and `scripts/code-exam.py` **12/12**, cold and warm, in
every arm — baseline, BOOT1, BOOT2, production, and the dump boot that produced the sidecar. `[measured-here]`

## 4. Staleness protection, and the two times it fired

A derived artefact that silently goes stale is worse than no artefact, so two checks stand between a stale sidecar and a served
token and neither is a warning. **`preflight-fastload.py`** runs in the prelude, before the engine imports anything: it
recomputes the file half of the identity against the manifest and exits **30** when the sidecar is missing, empty or short a
shard, **31** on an identity mismatch, printing the first 20 differing fields; it is cheap by design, with no torch and no vLLM.
**`harem_fastload._restore`** repeats the comparison inside the engine with what only the engine knows (vLLM and `cuda-exl3`
versions, parallel layout, `hf_overrides`), and refuses on top of that for any difference in the tensor name set, shape, dtype
or byte count. Failure raises; there is no fall-back-to-slow-path branch, because a fall-back is how you end up serving weights
nobody checked. It fired twice during the work, and both times it was the design working:

1. **The patch directory was edited while a boot was running.** It is mounted read-only into the container and its `patch-*.py`
   hashes are part of the identity; the hash changed between the main model's dump and the drafter's, so the drafter's dump was
   refused as "a different build". The rule now: **do not touch the patch directory while a boot is in progress.** The identity
   definition was corrected too — the fastload modules' own hashes were removed from it (they are the tool, not the recipe) and
   replaced by a `SIDECAR_FORMAT` version, bumped by hand when the on-disk layout or the dump semantics change.
2. **The drafter's `hf_overrides` is a function object**, not a dict — `SpeculativeConfig`'s config override callable. Its
   `repr` carries a fresh heap address every boot, so the drafter's sidecar looked stale the instant it was written. The address
   is now stripped from **both** sides of the comparison, so a sidecar written before the scrubbing existed is not falsely
   condemned either. The target model, whose `hf_overrides` is a plain dictionary, was never affected.

Neither was a bug in the fail-closed design: in both cases the alternative behaviour — shrug and load something — is the outcome
the design exists to prevent.

It fired a third time on 12 September 2026, and that one is worth a sentence of its own because it was **not** a patch file:
adding `"index_topk":8192` to the launcher's `--hf-overrides` changed the target model's `hf_overrides`, which is part of the
identity above, so a `load` boot was refused with `harem-fastload: sidecar was written for a different model configuration`.
**A launcher argument can cost a dump boot exactly as a `patch-*.py` can.** Adoption took one dump boot of about 6 minutes into a
new directory and 53 GB per rank, after which `load` boots are 170 s again; exploratory arms that change the override run with
`FASTLOAD_MODE` empty instead ([14](14-troubleshooting.md) §9.15,
[`../results/gates/index-topk-8192-12sep.md`](../results/gates/index-topk-8192-12sep.md)).

## 5. The regression that had to be chased: KV −4.1 %

BOOT1 came back with a KV pool of **4,231,404** tokens against the 0.80 reference arm's **4,413,223** — a 4.1 % loss from a
boot-time change that should not touch memory at all.

| | Reference arm (0.80) | BOOT1 | Production |
|---|---|---|---|
| `Available KV cache memory` (head node) | 35.4 GiB | 33.39 GiB | 33.36 GiB |
| `Actual usage ... (weights + non-torch)` | 60.07 GiB | **62.22 GiB** | — |
| `Model loading took` (weights) | 54.86 GiB | 54.86 GiB | 54.83 GiB |
| `GPU KV cache size` | 4,413,223 | 4,231,404 | **4,484,848** |

`[measured-here]`. Read the `Available KV cache memory` row as an indicator only: it is the head node's line, and the pool is
sized by the **smallest** of the three ranks — see [07](07-kv-and-draft-page.md) for the pool arithmetic.

The weights are 54.86 GiB in both arms, so the whole difference sits in **non-torch** memory. The cause is GB10's unified
memory: the GPU pool *is* host memory, vLLM sizes the KV pool from what is free after loading, and page cache is not free
memory. Reading 163.58 GiB through it bills the KV pool for pages nothing will read again — and `eager` reads them harder than
mmap does. The remedy is two calls in `harem_fastload.py`, both on by default and both numerically inert:
`posix_fadvise(POSIX_FADV_DONTNEED)` over the checkpoint shards once loading is done (`HAREM_DROP_CKPT_CACHE`), and
`malloc_trim(0)` to return the loader's freed host arenas to the OS (`HAREM_MALLOC_TRIM`; measured on the dump boot, RSS 7.95 →
5.20 GiB). Result: **4,484,848 tokens, 1.6 % above the pre-change reference arm**, not merely recovered. `[measured-here]`
Closed.

### 5.1 The same mechanism from outside the container: the settle gate

Section 5 is about page cache the engine *itself* generated. There is a second source of exactly the same
error, and it arrives before the container does.

The launcher kills the previous container with `docker rm -f` and starts the new one immediately. That
container held about 90 GiB, and the kernel does not give it back instantly. vLLM's boot-time memory profile is a
**delta** against `MemAvailable` taken just after NCCL init ([07](07-kv-and-draft-page.md) §1.1), so whatever has
not been reclaimed at that instant is charged to `weights + non-torch` — **and the sign is backwards**: a node that
starts dirty computes itself a *larger* KV pool. Boot order makes it systematic rather than random. The nodes start
worker-2 → worker-1 → head, so the head is the node given the least reclaim time; on one boot the three ranks began
with 104.10 / 113.07 / 113.10 GiB available (a **9.00 GiB** spread) and finished the profile within **0.99 GiB** of
each other `[measured-here]`.

**The boot sequence therefore has a step before `docker run`.** `scripts/start-tp3.sh` calls `sync` and then waits
for `MemAvailable` to come back above `SETTLE_MIN_GIB` (112 by default), polling every 3 s up to 180 s, and logs the
result:

```
mem settle: MemAvailable=113 GiB (target 112) after 6s
```

It has to be on the host: `/proc/sys` is in the container's `ReadonlyPaths` and the container is unprivileged, so a
prelude cannot drop caches even if it wanted to. Optionally the gate can be preceded by
`sudo sh -c 'sync; echo 1 > /proc/sys/vm/drop_caches'` — our NVFP4 sibling's preflight does exactly that on these
same nodes — but the wait alone was enough here, and dropping caches on a node the engine is about to read 56 GiB
from is not free.

**What it changed, and what it did not** `[measured-here]`:

| | before the gate | with the gate |
|---|---|---|
| per-rank startup free memory | 104.10 / 113.07 / 113.10 GiB (spread **9.00**) | 111.65 / 113.06 / 113.07 GiB (spread **1.4**) |
| the spread as a share of a rank's KV allowance | **27 %** | ~4 % |
| KV tokens bought | — | **zero** |
| boot time | — | + the wait (measured: seconds, capped at 180 s) |

**No published pool figure here is known to be wrong**, and this section is not a retraction of one. The pool takes
the minimum over ranks, and on every boot with a ledger the polluted node was the head, which was not binding —
which is luck rather than design, since the polluted node is whichever one starts last. What the gate removes is
the *possibility*: 27 % of a rank's allowance sitting in the measurement, waiting for a different start order.

The gate is a **measurement** fix, not a performance one, and section 5 is why it matters. That section chased a
4.1 % pool regression to page cache and fixed it — a piece of work that is only sound if the pool number is stable
enough to attribute. Recent boots span 4,231,404 → 4,484,848 (**6.0 %**), every step of which has a candidate
explanation here, and comparable load boots after the page-cache fix agree to 0.4 %. With the baseline unpinned
there was no way to prove which was which. The acceptance test for a boot that intends to report a pool number is in
[07](07-kv-and-draft-page.md) §1.1: all three ranks within 1 GiB on both the `Free memory on device` and the
`consumed memory` lines, and the boot must be a **load** boot, never a dump boot.

## 6. Block-layer forensics, and the decision rule they support

Before choosing between "the read path is slow" and "the copy path is slow", the boot was recorded at the block layer:
`/sys/block/nvme0n1/stat` before and after, plus five-second sampling of the worker processes' `/proc/<pid>/io` and CPU.
Read-only, no code changes, attachable to any arm.

```
bash scripts/boot-forensics.sh BOOT1
```

| Arm | Bytes read from disk (head / worker-1 / worker-2) | Worker CPU, median |
|---|---|---|
| BOOT1 (eager + EP filter) | 193.8 / 166.8 / 166.9 GiB | 40–50 % (under one core) |
| **BOOT2 (S4)** | **57.1 / 56.6 / 56.7 GiB** | **90–100 %** |

`[measured-here]`, raw in `results/boot/`.

The decision rule, and the part worth stealing: **a full checkpoint's worth of bytes read while the worker CPU is idle means the
read path is the bottleneck**, and the cure is a better reader (S2, then S4); the same bytes with one core pinned at 100 % would
mean the *copy and slicing* path is, and the cure is to stop re-deriving the slices (S4). BOOT1 was unambiguously the first
pattern. Note that the head node read **more** than the checkpoint holds, 193.8 against 163.58 GiB: kernel readahead pulls in
skipped tensors anyway, which is exactly why S1's accounting argument ("we skipped 59 % of the bytes") was never a measurement.
After S4 the picture inverts — bytes read fall 66 % and the CPU saturates, so the fast path is now bound by reading and hashing
rather than copying, which is section 10's first item.

## 7. Speed and quality: nothing improved, nothing regressed

This section is not a win. It is the evidence that a 2.3× faster boot bought no speed and cost no speed. Five sweep rounds,
first two discarded as MLA-tuner warm-up, median of rounds 3–5 with [min–max]; realistic prompt set (12 short English code
prompts); reference arm is the same image at the same 0.80 with none of S1–S4, and [09](09-measurement-protocol.md) explains why
fewer than five rounds would be worthless here.

**Total tok/s** `[measured-here]`

| Concurrency | Reference (0.80) | BOOT1 | BOOT2 | BOOT2 vs reference |
|---|---|---|---|---|
| 1 | 52.81 [52.34–53.50] | 53.81 [52.78–54.50] | 54.37 [54.16–54.49] | +3.0 % |
| 2 | 80.97 [77.24–81.01] | 80.37 [78.47–80.42] | 80.06 [79.58–80.79] | −1.1 % |
| 4 | 117.08 [113.33–120.19] | 112.87 [109.72–114.73] | 114.61 [113.47–115.59] | −2.1 % |
| 6 | 135.64 [134.14–141.81] | 136.48 [136.24–136.63] | 136.82 [135.95–137.44] | +0.9 % |
| 8 | **162.85** [162.00–164.71] | 164.04 [160.75–166.09] | **161.82** [161.29–163.43] | −0.6 % |

Per stream: C1 61.07 → 61.77 tok/s (+1.1 %), C8 25.52 → 25.22 (−1.2 %); C2 / C4 / C6 −0.6 / +1.9 / +1.7 % — all inside the arms'
own round-to-round spread, with speculative acceptance steady in the **61–64 %** band. Prefill, 7K prompt, second (uncached)
request: reference 1,469 · BOOT1 1,470 · BOOT2 1,475 · production 1,447 tok/s; fresh, never-seen prompts: reference (warm) 1,703
· BOOT2 1,704 tok/s. Host memory and swap at the end of each arm:

| Arm | head | worker-1 | worker-2 |
|---|---|---|---|
| Reference (0.80) | free 11.6 G / swap 0.12 G | 12.9 G / 0.10 G | 12.6 G / 0.09 G |
| BOOT1 | 12.8 G / 0.12 G | 13.8 G / 0.10 G | 13.8 G / 0.09 G |
| BOOT2 | 11.1 G / 0.12 G | 12.3 G / 0.10 G | 12.3 G / 0.09 G |
| **Production** | **10.9 G / 0.12 G** | **12.2 G / 0.10 G** | **12.1 G / 0.09 G** |

Swap is a residue from an earlier boot and did not grow within any arm. BOOT1's roomier free memory is the small KV pool,
production's tighter figure the large one — the memory went where it was supposed to. The floor rule from
[07](07-kv-and-draft-page.md) still holds: never below 4 GiB free.

## 8. What this cost

No gain is published here without its price, and this one has four.

| Cost | Amount | Notes |
|---|---|---|
| Disk | **+56 GiB per node** of derived artefact | 370–374 G free afterwards `[measured-here]` |
| Maintenance debt | regenerate the sidecar whenever the checkpoint, image, patches or TP/EP layout change | closed by the manifest and the preflight, not by memory: a stale sidecar **refuses the boot** |
| One-off production | **~11 min** dump boot per node, engine otherwise down | 174 s load + 193 s dump + drafter + hash sample `[measured-here]` |
| Every boot | **~20 s** of hash verification | `FASTLOAD_VERIFY=32`, two models; `FASTLOAD_VERIFY=0` removes it and is **not recommended** |

The maintenance debt is the one that would bite an unattended cluster, so be explicit: the protection is not "remember to
regenerate", it is that the identity is recomputed on every boot and a mismatch is fatal, so the cost of forgetting is a refused
boot with a printed diff — the cheapest possible failure.

**It came due the same day.** Moving the image from `exl3-zeus:f4987cf` to `exl3-zeus:9bf594c` for the persisted MLA tuner cache
([12](12-tuner-cache.md)) invalidated the sidecar, the preflight refused the boot, and regenerating it cost one dump boot of
**682 s** wall on all three nodes `[measured-here]`. That is the design working, and it is also the real recurring price of this
page: **every kernel-image change now carries an 11-minute dump boot.** Put it in the plan for the image, not in the surprise
column. One trap with it: the dump boot's own KV pool reads 3,958,677 rather than ~4.48M, because writing 56 GiB per node goes
out through the page cache (§5). It returns on the next boot. Do not record a dump boot's pool as a result.

**And one thing that was not measured: the ceiling of the read path.** `[not tested]` The sidecar is read at 0.88–1.04 GB/s (893
/ 905 / 1039 MB/s on the three ranks) where another loader on the same NVMe reaches 3.1 GB/s, so roughly **3× is still on the
table** on top of S4. `HAREM_FASTLOAD_READ=mmap` exists in the code as a one-variable A/B that needs no re-dump, and it has
**never been run**.

---

### 8.1 Two patch trees means two sidecars, and the identity is what makes that safe

Production configuration 9 serves a different checkpoint from a different patch tree
([13](13-full-scope-checkpoint.md)), and this page is where that lands. **Four inputs to the manifest
identity moved at once** — the checkpoint, the sidecar `config.json`, the patch directory and the
image — so production 8's sidecar could not be reused under any circumstance, and the arm needed its
own dump boot.

Two things follow, and both are cheap only if you plan them.

**Give the new sidecar its own directory name.** `FASTLOAD_DIR` is mounted **read-write** in dump
mode, at the same path inside the container as outside, so reusing the name would have overwritten
production 8's sidecar during the dump — destroying the rollback while building its replacement. One
line in the env file, and it is the difference between an experiment and an outage.

**Keep the experimental patches in their own directory.** The identity hashes every `patch-*.py` in
the patch directory and the full text of the prelude (§4), so writing the full-scope patch into
`patches/tp3/` would have refused the next production boot — which is exactly what happened twice on
5 September before the second tree existed ([09](09-measurement-protocol.md) §11.2). The full-scope
arm was built in `tracks/tp3/patches/` instead, and **production 8 stayed up, untouched, throughout**.
The trees' relationship, the file-by-file check that they have not drifted, and the merge that is
still owed are in [`tracks/tp3/patches/README.md`](../tracks/tp3/patches/README.md).

The boot ledger for the new arm, both boots, all three ranks `[measured-here]`:

| arm | mode | wall | weights | drafter | init engine | KV pool |
|---|---|---|---|---|---|---|
| production 9, **dump** | cold read + dump | **620 s** | 163.9 / 159.2 / 161.6 s | 2.4 / 2.3 / 2.8 s | 66.0 s | 4,840,220 — **not usable** |
| production 9, **load** | fastload | **251 s** | **57.9 s** (50.24 GiB, 932 MB/s) + 5.4 s verify | 2.3 s | 66.8 s | **5,165,289** |
| production 8 (control) | fastload | 264 s | 73.2 s | 2.2 s | 65.2 s | 4,696,969 |

The dump wrote **50.24 GiB in 22 shards per rank** (219 / 202 / 241 s = 246 / 268 / 224 MB/s) plus a
2.04 GiB drafter sidecar, with `MANIFEST.json` present on all three (53 G, 25 files each).

Three things in that table are worth reading rather than skipping.

**The new arm restores 21 % faster** — 57.9 s against 73.2 s — for a reason that has nothing to do
with this page's work: its sidecar is simply smaller, 50.24 GiB per rank against about 63, because the
checkpoint behind it is smaller. Boot time followed the weights, as it should.

**The dump boot's pool reads 4,840,220 and the load boot's 5,165,289**, a 6.3 % gap on an otherwise
identical arm. That is §5's page-cache mechanism, measured again on a new arm, and it is the whole
reason for the rule: **never record a dump boot's pool as a result**
([09](09-measurement-protocol.md) §11.1). Read the other way round, the pair also prices fast-load
itself at about **2 GiB and ~325k KV tokens** on this arm.

**Both rows in the comparison are load boots**, which is what makes the +10.0 % pool figure a
like-for-like reading rather than an artefact of which boot each side happened to come from.

---

## 9. Rollback

Six independent one-line rollbacks; none depends on another, and none touches the image, because every patch runs in the
container layer from the prelude. Reboot the engine afterwards.

**1. Full revert to the pre-change production env** (no S1–S4):

```
for h in head worker-1 worker-2; do ssh $h "cp ~/exl3-zeus/.env.tp3.bak-sw256-080 ~/exl3-zeus/.env.tp3"; done
```

**2. Disable only S4**, keeping S1+S2+S3: the loader returns to the stock path, weight loading goes back to about 190 s, and the
sidecar stays on disk doing no harm:

```
for h in head worker-1 worker-2; do ssh $h "sed -i '/^FASTLOAD_MODE=/d' ~/exl3-zeus/.env.tp3"; done
```

**3. Remove one flag** from `EXTRA_ARGS` — the three are independent:

```
for h in head worker-1 worker-2; do ssh $h "sed -i 's/ --no-enable-flashinfer-autotune//' ~/exl3-zeus/.env.tp3"; done
```

**4. Remove the patches** — drop the `run` lines from `tp3-prelude.sh`, or restore the backups the install leaves beside it
(`tp3-prelude.sh.bak-preepf`, `tp3-prelude.sh.bak-prefastload`, `start-tp3.sh.bak-prefastload`):

```
for h in head worker-1 worker-2; do ssh $h "cp ~/exl3-zeus/tp3/tp3-prelude.sh.bak-prefastload ~/exl3-zeus/tp3/tp3-prelude.sh"; done
```

**5. Regenerate the sidecar** after any checkpoint, image, patch or layout change: remove the rank directories with the command
in step 6, run one dump boot as in section 2.4, then set `FASTLOAD_MODE=load` again. About 11 minutes.

**6. Reclaim the disk**, 56 GiB per node — but do step 2 first: with `FASTLOAD_MODE=load` still in the env and the sidecar gone,
the launcher and then the preflight refuse to start, correctly and loudly, and you will have turned a disk cleanup into an
outage.

```
for h in head worker-1 worker-2; do ssh $h "rm -rf /var/tmp/glm53-exl3-tp3-r*"; done
```

## 10. What is next, and was not done

1. **The ceiling of the read path.** The sidecar reads at 0.88–1.04 GB/s where the same disk gives 3.1 GB/s to another loader.
   `HAREM_FASTLOAD_READ=mmap` is already in the code and can be A/B'd **without regenerating the sidecar**; beyond that, a
   multi-threaded `pread` into pinned buffers. Ceiling: phase 2 from 67 s to roughly 20 s, total boot roughly 230 s.
   `[estimate]` `[not tested]`
2. **The profile run is now the second-largest item**, 67–73 s, and its **first step alone is ~45 s** (`Profiling CUDA graph
   memory (PIECEWISE): 1/19 [00:44<13:22, 44.56s/it]`) — inside that step the first NCCL collective is established and the MLA
   tuner fires. The NCCL part is unavoidable.
3. ~~**Persist the MLA tuner cache.**~~ **Done, and it is upstream's, not ours.** `cuda-exl3` kept the tune map in a
   process-local `static std::map`; commit `9bf594c` writes it into the already-mounted `/cache` behind
   `CUDA_EXL3_TUNE_CACHE`. Measured on this cluster: **18 tune events before serving → 0**, and the first sweep round stops
   being a penalty, which is what shortened the protocol in [09](09-measurement-protocol.md) from five rounds to three. The
   boot-time part of it — a few hundred milliseconds inside phase 4 — was **not** isolated, because the reason for doing it was
   measurement hygiene rather than boot time. Full account in [12](12-tuner-cache.md) `[measured-here]`.
4. **Re-test the memory ladder.** The pool is 4,484,848 at 0.80 now, and the page-cache remedy from section 5 was not in place
   when the 0.85 arm was last measured, so the higher rungs may be roomier. A separate measurement, not a claim. `[not tested]`

## 11. The prelude, start to finish

[`scripts/tp3-prelude.sh`](../scripts/tp3-prelude.sh) is mounted as the container's entrypoint and runs
every patch below, in this order, before `exec vllm serve "$@"`. Two invocation styles are mixed on
purpose: most patches are **called unconditionally** and are inert internally when their own knob is
unset (so a patch that drifts cannot be blamed on a shell `if` nobody remembered to update); the three
newest arms are **called conditionally**, from the shell script itself, so an env file that never asks
for one of them cannot be broken by an anchor that drifted in some other image. `patches/tp3/README.md`
has the full description of each patch's env knob and default; this table is only the order and the
gate.

This table is `patches/tp3/`'s prelude, which is production configurations 1–8 and the rollback path.
Production 9 runs `tracks/tp3/patches/tp3full-prelude.sh`, which is this list plus two steps at the end
of it — `patch-fullscope-tp3.py` and `check-padload-tp3.py`, both behind `HAREM_EXL3_FULLSCOPE`
([13](13-full-scope-checkpoint.md) §7). Everything below applies to both.

| # | Command | Invoked | Effect gated by | Introduced |
|---|---|---|---|---|
| 1 | `patch-vllm-tp3.py --root $VLLM_PY` | always | — (unconditional) | base TP=3 padding |
| 2 | `patch-exl3-ep.py --pkg $EXL3_PKG --overlay .../overlay/cuda_exl3/_harem_ep.py` | always | — (unconditional) | base EP support |
| 3 | `patch-dflash-tp3.py --root $VLLM_PY` | if `qwen3_dflash2.py` exists in the image | — (unconditional once invoked) | DFlash2 port |
| 4 | `patch-kvdiag-tp3.py --root $VLLM_PY` | always | — (logging only) | KV pool visibility |
| 5 | `patch-swblock-tp3.py --root $VLLM_PY` | always | `HAREM_SW_BLOCK_SIZE` | production 3 |
| 6 | `patch-epfilter-tp3.py --root $VLLM_PY` | always | `--enable-ep-weight-filter` (CLI) + `HAREM_EP_FILTER_SUFFIXES` | production 4 |
| 7 | `patch-fastload-tp3.py --root $VLLM_PY` | always | `HAREM_FASTLOAD_MODE` | production 4 |
| 8 | `patch-zerokv-tp3.py --root $VLLM_PY` | only if `HAREM_ZERO_ATTENTION_KV=0` | (the `if` itself; refuses at startup on this checkpoint — [07](07-kv-and-draft-page.md) §8) | shipped with production 7, not active in it |
| 9 | `patch-draftkv-tp3.py --root $VLLM_PY` | only if `HAREM_DRAFT_KV_DTYPE` is set | (the `if` itself) | production 7 |
| 10 | `patch-tilelang-failloud-tp3.py --root $VLLM_PY` | only if `HAREM_TILELANG_FAILLOUD=1` | (the `if` itself) | production 7 |
| 11 | `flashinfer-warmup.py` | always | `HAREM_FLASHINFER_WARMUP` (internal; default on) | production 7 |
| 12 | `preflight-tp3.py --model $1 --tp ... --ep ...` | if argv[1] is a directory | — | base |
| 13 | `preflight-fastload.py --model $1` | if `HAREM_FASTLOAD_MODE` is set and argv[1] is a directory | — | production 4 |
| 14 | `exec vllm serve "$@"` | always | — | — |

`patch-mhcfused-tp3.py` is **not** in this list. It is written, statically verified and kept in
`patches/tp3/`, but the prelude does not call it — see `patches/tp3/README.md` and
[11](11-open-issues.md) §2.16.

Two things this ordering is not: it is not the order the patches were written in (5, 6 and 7 predate 8,
9 and 10 by weeks), and it is not a claim that steps 1–7 are cheaper to change than 8–10. The fast-load
sidecar's identity hash covers every `patch-*.py` in the directory and the prelude script's full text,
so changing any one of these fourteen steps — including only editing a comment on a step whose knob is
never set — invalidates the sidecar on every node ([09](09-measurement-protocol.md) §11,
[11](11-open-issues.md) §2.21).

## 12. One launcher, one copy — and the switch it was missing

The prelude above is mounted from a directory; the launcher that mounts it is a single file, and for
two days there were **two of it**. A working copy on the nodes had grown a `--profiler-config` arm; a
second copy in the same tree, one day older, had not, and had never had the settle gate, the fast-load
mounts or the conditional prelude arms either. Nothing pointed at the older one, nothing depended on
it, and it stayed readable and plausible. The cost was not a wrong boot — it was that
[10](10-results-and-roofline.md) §5 spent a week as a *reconciliation* with a 2.8 % residual, because
`/start_profile` answered **404** on the running engine and reconfiguring it needed a boot nobody
would spend. When the flag finally went in, the trace deleted two of that section's ranked targets in
an afternoon.

Two rules, both cheap:

**One launcher, one copy.** Whatever machinery a stack keeps — patch directories, sidecar builders,
env templates — the file that assembles the `docker run` is the one that must never be duplicated. It
is the file most likely to be edited under time pressure, the one whose divergence is hardest to see
(both copies run; both produce a server), and the one whose drift is silently inherited by every
measurement afterwards. If a second copy exists, it is not a backup, it is a coin flip. The stale copy
here was retired rather than merged, and the arm was ported forward.

**Carry the profiler arm in production, off by default.** `scripts/start-tp3.sh` reads `PROFILER_DIR`
and, when it is set, passes:

```
--profiler-config {"profiler":"torch","torch_profiler_dir":"<dir>",
                   "torch_profiler_with_stack":false,"ignore_frontend":true}
```

Unset, it appends nothing and costs nothing. Set, `POST /start_profile` and `POST /stop_profile`
answer 200 on a **running** engine and each rank drops its own trace — so the most informative
measurement on this stack becomes a six-minute window rather than a boot. Note that this vLLM takes
the setting **only** as `--profiler-config`; `VLLM_TORCH_PROFILER_DIR` alone leaves the route
unattached and the endpoint answering 404, which is the exact failure that cost the week
([09](09-measurement-protocol.md) §4.1).

The general form is the same one the settle gate taught in §5.1 and the rulers taught in
[10](10-results-and-roofline.md) §4.1: **the instrument has to be in place before the question is
asked**, because by the time the question is interesting, the cost of installing the instrument is
exactly what stops you answering it.

Open items and retractions from this page are carried in [11](11-open-issues.md).
