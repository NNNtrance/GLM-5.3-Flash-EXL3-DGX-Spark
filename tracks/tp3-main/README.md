# tracks/tp3-main — three nodes, TP=3 with expert parallelism, on upstream vLLM `main`

**This is the stack we run since 8 October 2026.** It is the answer to the
[correctness notice](../../README.md) at the top of this repository: the old tracks
([`tracks/tp3`](../tp3/), [`tracks/tp2`](../tp2/)) never loaded the MoE router's load-balancing bias and
therefore picked the wrong experts in every layer, and nothing in them could have said so. This track runs
the same model on the same three DGX Spark with a different engine underneath: upstream vLLM `main` at
`21d93d0d8`, a `cuda-exl3` fork that loads the bias and **refuses to boot without it**, and fourteen
anchored patch scripts applied when the image is built.

Why it was rebuilt, how the defect was found and what correct routing costs:
[docs/20](../../docs/20-main-stack.md). What it measures:
[`results/main-stack/`](../../results/main-stack/README.md). This page is the recipe: what is pinned, how to
build it, how to boot it, which lines to read before you trust it, what each patch does, and how to go back.

**Two nodes are not covered** `[not tested]`: nothing here has been moved to a TP=2 arrangement.
**The quality benchmarks of the old tracks are withdrawn and have not been re-measured on this stack**
([`results/main-stack/quality-gates.md`](../../results/main-stack/quality-gates.md)); this page quotes none.

---

## Settings of this track

Every number in this track belongs to this arrangement unless it says otherwise.

| | |
|---|---|
| Nodes | three DGX Spark (GB10), `head` (rank 0, serves the API), `worker-1`, `worker-2`; mesh transport as in [docs/06](../../docs/06-nccl-mesh.md) |
| Parallelism | TP=3 + expert parallelism (mandatory at three ranks: [docs/05](../../docs/05-expert-parallel-and-cuda-exl3-fixes.md)) |
| Weights | `turboderp/GLM-5.3-Flash-exl3` 4.05 bpw, full scope ([docs/13](../../docs/13-full-scope-checkpoint.md)) |
| KV cache | `fp8`, `--block-size 256`, `HAREM_SW_BLOCK_SIZE=256`, `--max-model-len 1000000` |
| Memory | `--gpu-memory-utilization` **0.84**, KV pool 6,188,010 tokens (6,190,735 at autostart) `[measured-here]` ([`memory-fraction-084.md`](../../results/main-stack/memory-fraction-084.md)) |
| Scheduling | `--max-num-seqs 5`, `--max-num-batched-tokens 2048`, chunked prefill, prefix caching |
| Draft | DFlash2, `num_speculative_tokens` 7, fp8 draft KV; `disable_eagle_block_drop` **true**; schedule `[[1,1,7],[2,8,3]]` (the two switches below) |
| Prefill | `--kda-prefill-backend flashkda` |
| Vision | `{"image":16,"video":4}` per request, `--mm-encoder-tp-mode data`, `max_pixels` 12,544,000, `max_image_tokens` 8000, `--mm-processor-cache-gb 0` |
| Sparse indexer | `--hf-overrides '{"index_topk":2048}'` (the checkpoint's own value) |
| Defaults for clients | reasoning effort `low`, `clear_thinking` true; reasoning parser `deepseek_r1`, tool parser `glm47` (fail-closed, patched) |
| Network | `NCCL_MAX_NCHANNELS=8`, `NCCL_MESH_TIMEOUT_SEC=180`, port 8001, served name `glm-5.3-flash` |
| Autostart | `harem-exl3.service` on all three nodes, enabled |

## What is different from `tracks/tp3`

| | `tracks/tp3` (until 8 October) | `tracks/tp3-main` |
|---|---|---|
| vLLM | the model-launch image (vLLM `487ecf187`) plus a 23-file patch tree | upstream `main` @ `21d93d0d8`, 14 patch scripts |
| `cuda-exl3` | `754421f` and earlier upstream | fork `448f1d6` |
| Router bias | never loaded | loaded, with a boot gate that refuses to start without it |
| Routed-expert SwiGLU | unclamped | clamped at the model's `swiglu_limit` (10); no measurable effect |
| Sidecar identity | keyed on a tree of 23 files | keyed on 14 `yama-*.py`, the engine, the prelude and the plugin's `.py` sources |
| Patch tool | one script per edit, no common rules | one engine (`harem_yama.py`): base `sha256` pinned per file, anchor exactly once, idempotent, `--revert` |
| Quality figures | withdrawn | not yet re-measured |

---

## Pinned revisions

| Component | Revision | What we use it for |
|---|---|---|
| Base image | `vllm/vllm-openai@sha256:6f0d5e677145fb4003891a994f11fed993d022d141c71844157d3e84e55b1e0a` (the arm64 manifest of `nightly-21d93d0d8c0e9627900020382bfce4730e61cab7`; the multi-arch index of the same tag is `sha256:bcee860ca68b70b3e7f666ab3dedb5932c89d901f56d1a29b302ec08f9569884`). vLLM `0.30.1rc1.dev709+g21d93d0d8`, torch `2.13.0+cu130`, Python 3.12.3, CUDA 13.0.2, nvcc 13.0.88, transformers 5.18.0, FlashInfer 0.7.0.post1, NCCL 2.30.7. Apache-2.0; the NVIDIA libraries inside it carry NVIDIA's own terms, so we do not redistribute an image | the engine |
| vLLM | commit `21d93d0d8c0e9627900020382bfce4730e61cab7` (`main`, 6 October 2026 04:18 UTC, PR #58128) | what the patches are written against; their pinned base hashes are this commit's files |
| `cuda-exl3` fork | [`NNNtrance/cuda-exl3`](https://github.com/NNNtrance/cuda-exl3/tree/tp3-vllm-main) branch `tp3-vllm-main`, **production build `448f1d6b8964b96e7d4e299494538bfc240149ef`**. The branch tip `6f3b6dc9f3ffe11a956f21a4da1f75cc03c36e09` changes only a test default. Parent: `Zeuss5/cuda-exl3` master `6a1ffc34866e23f484574ce1922a8bca93eb33b2`. MIT | the EXL3 kernels and the vLLM plugin |
| Fork commit `f3e184a3848e5689c5ff36ea228d14c8f136628d` | the TP=3 production pieces folded into the plugin (padding, expert parallelism, mixed linear layers, the MLA KV layout), ported to vLLM `21d93d0d8` | |
| Fork commit `1451b1a12132b6b232b2b4bfa9e6c584f1d5161d` | loads the MoE router bias and adds the boot gate `[HAREM-MOE-KAPI]`; the 2-line fix is also offered upstream as [Zeuss5/cuda-exl3#8](https://github.com/Zeuss5/cuda-exl3/pull/8) | |
| Fork commit `42474b4b0d3056f71f98eddfa6f268a91319ecba` | clamps the routed experts' SwiGLU at `swiglu_limit` | |
| Fork commit `448f1d6b8964b96e7d4e299494538bfc240149ef` | two measurement knobs for the dense GEMM tuner (`CUDA_EXL3_GEMM_TUNE_REPS`, `CUDA_EXL3_GEMM_TUNE_VERBOSE`), off by default | |
| Image we run | tag `harem/vllm-exl3-tasima:a4i-21d93d0d8-448f1d6`, Id `sha256:8c4f560d92ba190ed84983bdf12e9975c5b4f1ec6dc380b5b93041b2a19ac261` (**our** build; yours has another Id, builds of this source are not bit-identical, [docs/14](../../docs/14-troubleshooting.md) §7.10) | the tag enters the fast-load identity; do not re-tag |
| Patches | [`patches/`](patches/) — 14 `yama-*.py`, the engine `harem_yama.py`, `harem-prelude.sh`, `harem_fastload.py`, `harem_fastload_id.py`, `harem_taslak_kapi.py`; hashes in [`patches/SHA256SUMS`](patches/SHA256SUMS). **These are not byte-identical to the files in the image we run:** for publication their comments and docstrings were translated to English and cleared of internal references, and seven string constants that land in comments, docstrings or one error message were edited. Every executable token is unchanged: applied to the pristine `21d93d0d8` files, both sets apply, re-apply as no-ops, pass `--check` and revert to the base, and the two patched trees differ only in that text `[measured-here]`. The fast-load identity hashes these files, so a sidecar written with one set is refused by the other — one dump boot | |
| Checkpoint | `turboderp/GLM-5.3-Flash-exl3`, branch `4.05bpw`, revision `2a30229e67012798ba9f0cd832bb78abf4c363d5`. MIT ([docs/01](../../docs/01-model-and-license.md)) | the weights |
| Draft model | `incoai/GLM-5.3-Flash-DFlash2`, revision `dc77ff1c99eeb2df044ee3d4f0094eb033fee410`. **CC BY-NC-ND 4.0, and our permission does not transfer to you** ([LICENSES.md](../../LICENSES.md)) | speculative decoding |
| Chat template | `chat_template.jinja` of `zai-org/GLM-5.3-Flash` at revision `690b7052` (4 September 2026), 10,950 bytes, sha256 `0c4099f3382d6c92700dfb99725025360966fd73032f0ecf32377c0d9e6309c5` ([docs/14](../../docs/14-troubleshooting.md) §9.11) | the served template; the checkpoint's own file is a different revision |
| NCCL mesh plugin | as in [docs/06](../../docs/06-nccl-mesh.md), unchanged by this track | the transport |

**Revisions we did not pin:** the full 40-digit commit of the chat template (`690b7052` is the short form we
recorded; the size and the sha256 above decide the question) `[not determined]`.

---

## What is in this directory

| Path | What it is |
|---|---|
| [`build.sh`](build.sh) | builds the image on one node: the base by digest, the fork at the pinned commit, kernels for `sm_121a`, the patches, and the gates; nothing is tagged until all pass |
| [`patches/`](patches/) | the 19 files of the patch tree, mounted read-only into the container at boot. **Do not edit them in place on a node**: the fast-load identity hashes them |
| [`env.tp3-main.example`](env.tp3-main.example) | the per-node environment template. Derive each node's file with `sed`, on that node; never copy a finished file between nodes ([envs/README.md](../../envs/README.md)) |
| [`bin/start-tp3-main.sh`](bin/start-tp3-main.sh), [`bin/preflight-tp3-main.sh`](bin/preflight-tp3-main.sh) | the launcher and its fail-closed preflight |
| [`harem-exl3.service`](harem-exl3.service), [`harem-exl3.service.d/tp3-main.conf`](harem-exl3.service.d/tp3-main.conf) | the autostart unit; and the drop-in for a node that already has the old `tracks/tp3` unit |
| [`tools/sidecar_kur.py`](tools/sidecar_kur.py) | builds the two sidecars (target and draft) the engine reads |
| [`tools/spec_extra.py`](tools/spec_extra.py) | validates and merges `SPEC_EXTRA_JSON` into `--speculative-config` |
| [`tools/dflash2_build_gate.py`](tools/dflash2_build_gate.py) | the CPU gate `build.sh` runs on the finished image |
| [`stack-dump/sitecustomize.py`](stack-dump/sitecustomize.py) | a `SIGUSR1` stack-dump hook for hang diagnosis (mounted into the container; inert in the production setting) |
| [`ops/jit-cache-sync.sh`](ops/jit-cache-sync.sh) | equalises the three nodes' JIT caches (the preflight's JIT gate) |
| [`ops/watchdog/`](ops/watchdog/README.md) | optional: a watchdog that reboots all three nodes when the engine dies or hangs |

---

## Quick start

Each step ends in a **check**; do not go on until it passes. The hardware, firmware, fabric and mesh-plugin
steps are not repeated here because nothing in them changed: do them as the old README's steps 1, 2 and 5
say ([docs/00](../../docs/00-hardware-and-os.md), [docs/06](../../docs/06-nccl-mesh.md)), including the rule
that you **reboot all three nodes together, or none**.

### 1. The base image, on one node

```
docker pull vllm/vllm-openai@sha256:6f0d5e677145fb4003891a994f11fed993d022d141c71844157d3e84e55b1e0a
```

**Check:** `docker image inspect vllm/vllm-openai@sha256:6f0d5e677145fb4003891a994f11fed993d022d141c71844157d3e84e55b1e0a --format '{{.Architecture}}'` prints `arm64`. `build.sh` refuses any other image.

### 2. Build the image, once

```
bash tracks/tp3-main/build.sh
```

It runs for a while (the kernel compile). Its checks, in the order it makes them: the patch files against
`patches/SHA256SUMS`; the nightly against its digest; the fork exported from the pinned commit; the
17 vLLM files the patches pin are byte-identical to the nightly's; the plugin built for `sm_121a` and for
nothing else, its `.py` files equal to the source, the SwiGLU clamp, the `[HAREM-MOE-KAPI]` gate and the
tuner knobs present; the prelude applied (14 results the first time, 0 the second) and `--check` on all 14;
the DFlash2 gate. Only then does it commit the image.

**Check:** the last lines name the tag and print its Id (also in `~/exl3-main-build/IMAGE_ID`). Behaviour,
not a hash, is what certifies a build of this source: [docs/14](../../docs/14-troubleshooting.md) §7.10.

### 3. The same image on all three nodes

```
docker save harem/vllm-exl3-tasima:a4i-21d93d0d8-448f1d6 | ssh worker-1 docker load
```

Repeat for `worker-2`. **Check:** `docker image inspect harem/vllm-exl3-tasima:a4i-21d93d0d8-448f1d6 --format '{{.Id}}'` prints the same Id on all three. The preflight compares it against `IMG_ID` on every boot.

### 4. Weights, draft and template, on every node

```
huggingface-cli download turboderp/GLM-5.3-Flash-exl3 --revision 2a30229e67012798ba9f0cd832bb78abf4c363d5 --local-dir /var/tmp/glm-5.3-flash-turboderp-4.05bpw
```

```
huggingface-cli download incoai/GLM-5.3-Flash-DFlash2 --revision dc77ff1c99eeb2df044ee3d4f0094eb033fee410 --local-dir /var/tmp/dflash2-draft
```

```
huggingface-cli download zai-org/GLM-5.3-Flash chat_template.jinja --revision 690b7052 --local-dir /tmp/glm-template
```

Read the draft's licence first ([docs/01](../../docs/01-model-and-license.md)). **Check:** `sha256sum -c SHA256SUMS`
in the checkpoint directory passes on each node; the template is the right one when
`python3 scripts/verify-chat-template.py /tmp/glm-template/chat_template.jinja` says so (its size and sha256
are in the table above). If the Hub refuses the short revision, take the file from that repository's 4 September 2026 revision by
any other means: its size and sha256 are the arbiter. Copy it to `~/exl3-main/chat_template.jinja` in step 5.

### 5. Install the files, and derive each node's environment

On every node, from the root of a clone of this repository:

```
mkdir -p ~/exl3-main/log && cp -r tracks/tp3-main/bin tracks/tp3-main/patches tracks/tp3-main/stack-dump tracks/tp3-main/tools ~/exl3-main/
```

```
cp /tmp/glm-template/chat_template.jinja ~/exl3-main/chat_template.jinja
```

```
mkdir -p /var/tmp/exl3-main/cache && chmod 755 /var/tmp/exl3-main /var/tmp/exl3-main/cache
```

Put the patched mesh plugin directory (the `libnccl-net-mesh.so` and its sibling, built per
[docs/06](../../docs/06-nccl-mesh.md)) at `~/exl3-main/nccl-mesh-patched2/`. Then derive the environment on that node. For the
head (rank 0); on `worker-1` and `worker-2` change the first four values to `1`/`worker-1`/`192.0.2.11` and
`2`/`worker-2`/`192.0.2.12`:

```
NODE_RANK=0 NODE_NAME=head HOST_IP=192.0.2.10 HEAD_IP=192.0.2.10 IFACE=eth0 IMAGE_ID=$(cat ~/exl3-main-build/IMAGE_ID)
```

```
sed -e "s#@NODE_RANK@#$NODE_RANK#; s#@NODE_NAME@#$NODE_NAME#; s#@HOST_IP@#$HOST_IP#; s#@HEAD_IP@#$HEAD_IP#; s#@IFACE@#$IFACE#; s#@HOME@#$HOME#g; s#@IMAGE_ID@#$IMAGE_ID#" tracks/tp3-main/env.tp3-main.example > ~/exl3-main/.env.tp3-main
```

Set `IFACE` to your **management** interface (`ip -br addr`), never a fabric one; `HEAD_IP` is the head's
management address on all three nodes. Set `MESH_SO_SHA` in the file to the first 16 hex digits of
`sha256sum ~/exl3-main/nccl-mesh-patched2/libnccl-net-mesh.so`, the same value on all three. Set
`FASTLOAD_MODE=dump` for the first boot. Finally seal the directory so the preflight can notice an edit:

```
cd ~/exl3-main && find bin patches stack-dump tools -type f -print0 | sort -z | xargs -0 sha256sum > MANIFEST.sha256
```

**Check:** `grep -nE '^[A-Z_]+=.*@' ~/exl3-main/.env.tp3-main` prints nothing (no unreplaced placeholder), and
the launcher prints a complete command: `ENV_FILE=$HOME/exl3-main/.env.tp3-main DRY_RUN=1 bash ~/exl3-main/bin/start-tp3-main.sh`.

### 6. The two sidecars, on every node

The sidecars are symlink trees next to the downloads: the checkpoint's KDA `qkv_proj` and `conv1d` tensors
split three ways offline (no requantisation), and the draft padded 32/8 → 36/9 and tagged. The tool runs
inside the image because the draft's padding plan lives in the plugin.

```
docker run --rm --entrypoint python3 -v /var/tmp:/var/tmp -v $HOME/exl3-main/tools:/tools:ro harem/vllm-exl3-tasima:a4i-21d93d0d8-448f1d6 /tools/sidecar_kur.py hedef --kaynak /var/tmp/glm-5.3-flash-turboderp-4.05bpw --cikti /var/tmp/glm-5.3-flash-turboderp-4.05bpw-harem
```

```
docker run --rm --entrypoint python3 -v /var/tmp:/var/tmp -v $HOME/exl3-main/tools:/tools:ro harem/vllm-exl3-tasima:a4i-21d93d0d8-448f1d6 /tools/sidecar_kur.py taslak --kaynak /var/tmp/dflash2-draft --cikti /var/tmp/dflash2-draft-tp3-harem --tp 3
```

(`hedef` = target, `taslak` = draft.) **Check:** both print one JSON line whose `durum` is `kuruldu`
(built) and the first one verifies itself byte for byte against the original before it returns; the draft's
`config.json` now carries `harem_tp_pad` with `"tp": 3`. Run `dogrula` (verify) with the same arguments to
repeat the target check. If the draft's `config.json` was already padded by an older tool, restore the
original and pass it with `--kaynak-config`; the tool refuses a source that is already padded.

### 7. First boot: the dump boot

The first boot of a new image, patch set or checkpoint reads the whole 163 GB checkpoint at every rank and
**writes the fast-load sidecar** (`FASTLOAD_DIR-r<rank>`, about 53 GB per node). It takes several minutes
longer than every later boot (380 s from `docker run` in our dump boot, against 169 to 187 s for a load boot
`[measured-here]`, [`boot-and-watchdog.md`](../../results/main-stack/boot-and-watchdog.md)). Start the ranks last
to first. By hand (the unit of step 9 runs the same two commands):

```
ENV_FILE=$HOME/exl3-main/.env.tp3-main bash ~/exl3-main/bin/preflight-tp3-main.sh && ENV_FILE=$HOME/exl3-main/.env.tp3-main bash ~/exl3-main/bin/start-tp3-main.sh
```

Run it on `worker-2`, then `worker-1`, then `head`. **Read the gate lines (next section) before you send a
single request.** Then check the API:

```
curl -s http://192.0.2.10:8001/v1/models
```

### 8. The fast boot

Stop the engine on all three nodes (`docker stop exl3-tp3`), set `FASTLOAD_MODE=load` in each env file,
equalise the JIT caches (below) and start again. A load boot restores the rank's tensors straight from the
sidecar and re-hashes a sample of them. **Reboot all three nodes together** before any measurement
([`bandwidth-fragmentation.md`](../../results/main-stack/bandwidth-fragmentation.md): after many engine
sessions on one boot the memory bandwidth was 7–11 % lower `[measured-here]`).

#### The JIT cache gate

JIT-compiled kernels are cached per node. A rank that must compile what its neighbours already have arrives
late at the first collective, the mesh plugin's operation timeout turns that into a boot that hangs without a
message, and `NCCL_MESH_TIMEOUT_SEC=180` only makes the window wider. So the head's preflight refuses to start
unless the three nodes hold **identical** JIT cache listings (file names and sizes under `triton`, `tilelang`,
`flashinfer` and `deep_gemm/cache`). On the very first boot all three are empty and equal; after the dump boot
they differ. Equalise them once, with the engine stopped, from any machine with key-based ssh to the three:

```
NODES="head worker-1 worker-2" CACHE_DIR=/var/tmp/exl3-main/cache bash tracks/tp3-main/ops/jit-cache-sync.sh sync
```

`jit-cache-sync.sh check` prints each node's size and says whether the three agree. The gate lists files with an
unprivileged `find`; if the cache directory is not readable by the service user it sees nothing on every node
and passes vacuously, so make `CACHE_DIR` world-readable. `JIT_GATE=0` in the env file switches it off.

### 9. Autostart

On every node:

```
sed "s#@USER@#$USER#; s#@HOME@#$HOME#g" tracks/tp3-main/harem-exl3.service | sudo tee /etc/systemd/system/harem-exl3.service
```

```
sudo systemctl daemon-reload && sudo systemctl enable harem-exl3.service
```

The unit is a oneshot with `RemainAfterExit`; the container runs with `--restart no` on purpose (a rank that
quietly retries by itself is the "fluent and wrong" failure class this stack refuses). `Conflicts=harem-motor.service`
stops the NVFP4 sibling's engine from running next to it; it does **not** stop that unit being enabled at boot,
so disable it ([systemd/README.md](../../systemd/README.md) explains the hazard). The launcher's settle gate
and the preflight both call `sudo -n tee /proc/sys/vm/drop_caches`; give the service user a passwordless
sudo line for it, or the drop is silently skipped.

**A node that already carries the old `tracks/tp3` unit** keeps its unit and its name and takes the drop-in
instead: `sed` it the same way into `/etc/systemd/system/harem-exl3.service.d/tp3-main.conf`, move any older
drop-in that sets `ENV_FILE` out of the way, `daemon-reload`. Anything that watches `harem-exl3` or the
container name `exl3-tp3` keeps working unchanged. To go back, delete the file.

**Check:** reboot all three nodes together, with the unit enabled and the sidecars in place. Our power-on to
`/health` 200 took 274 s `[measured-here]` ([`boot-and-watchdog.md`](../../results/main-stack/boot-and-watchdog.md)).

### 10. Optional: the watchdog

[`ops/watchdog/`](ops/watchdog/README.md). Read it before you enable it: it reboots three machines.

---

## The lines to read at boot

None of the lines below is decoration. Each is printed by a patch or the plugin that fails closed, and its
**absence** is the signal. Grep the container log of each rank:

```
docker logs exl3-tp3 2>&1 | grep -E 'harem-prelude|harem-taslak-kapi|HAREM|DFlash draft|KV cache size|EXL3 routed|EAGLE trailing|startup complete'
```

**Every boot (dump or load):**

| Line (the fixed parts) | From | Means |
|---|---|---|
| `[harem-prelude] rank=R tp=3 ep=1 vision=1 mambagrid=1 swblock=256 fastload=load` | prelude | the knobs the patches will read |
| 14 × `[yama-...] --apply: değişiklik yok (zaten uygulanmis)` | prelude | all 14 patches were found already applied in the image; a `FAILED:` here (exit 21) means an anchor stopped matching and the rank stopped |
| `[harem-prelude] sidecar: kaynak=... bolunen_kda=34 dusurulen=218` | prelude | the target sidecar is a HAREM sidecar: 34 KDA layers split, 218 fused tensors superseded |
| `[harem-taslak-kapi] taslak tp=3 ... harem_tp_pad={"num_attention_heads":[32,36],"num_key_value_heads":[8,9],"tp":3} -> GECTI` | draft-tag gate | the draft is tagged for tp=3 (`GECTI` = passed; `RED` = refused) |
| `HAREM TP pad: tp=3 num_attention_heads=64->66 num_key_value_heads=64->66 linear_num_heads=64->66 harem_shared_expert_intermediate_size=2048->2304` | plugin | the TP=3 padding plan |
| `EXL3 routed experts: mode=EP ep_size=3 experts_local=96/288 ...` | plugin | expert parallelism: 96 of 288 experts on this rank |
| **`[HAREM-MOE-KAPI] GECTI <layer>: 1152/1152 expert tensors written; router bias = checkpoint <name> (288 experts, max diff 0); swiglu_limit 10 (routed experts clamp)`** | plugin | **the gate this stack exists for.** One line per MoE layer, **42 per rank** (layers 3 to 44), on all three ranks. A zero or different bias, an unwritten expert tensor or a wrong shape raises and the rank does not boot |
| `DFlash draft: 5 KV layers kept in 1 independent cache group(s)` | `yama-dflash-kvgrup` | the drafter has its own KV group |
| `GPU KV cache size: N tokens ...` | vLLM | our pool: 6.19 M; an unexpectedly small N means the memory settle failed or `GMU` moved |
| `[HAREM-MAMBAGRID] split_grid=3328 (cache_config.block_size=256)` | `yama-mambagrid` | prefill chunks follow the KDA state grid |
| `[HAREM-GLM47-FAILCLOSED] failclosed=True ... required=True ...` | `yama-glm47` | the fail-closed tool-call parser is active |
| `EAGLE trailing prefix-cache block dropping is disabled.` (a warning) | vLLM | switch 1 is on, as intended |
| `Application startup complete.` on the head, `/health` 200 | vLLM | up |

**Not printed, and that is right:** `HAREM prefix-hit: N drafter group(s) flagged is_eagle_group`. It appears only
when `HAREM_PREFIX_HIT=1` and the drop is still on; with switch 1 set the annotation returns early.

**Fast boot only:** `[harem-fastload] restore start tag=...`, `ledger restored ... 76 module(s), 48792 part(s)
marked`, `restored 6456 tensors, 50.72 GiB from 22 shards in 64.9 s`, `verify OK ... 32/2716 tensors re-hashed`
(then the same for the draft), `page cache dropped`, `malloc_trim`. Any identity mismatch, missing or extra
name, wrong shape or hash raises: the sidecar is never trusted unchecked.

**Dump boot only** (the weights are loaded the slow way, so these patches speak): `HAREM-SIDECAR: N superseded
tensors skipped (...)`, `HAREM-TP3 pad: ...` lines, `HAREM-TP3 drafter pad verified on rank R/3: 32/8 -> 36/9 ...
exactly zero`, `HAREM-VISION: tower EXL3 linears=99, unquantized=0, ...`, and at the end `dump done tag=... 50.72 GiB in 22
shards`. A fast boot restores the tensors without calling the loader, so on a fast boot **these checks are
not repeated**; they were made when the sidecar was written, which is why a changed patch or checkpoint forces
a new dump boot. The numbers above are our boot of 8 October `[measured-here]`; the fixed parts must match yours.

### Reading the Turkish in the log lines

The scripts match these words exactly, so they were not translated.

| Word | Meaning | Word | Meaning |
|---|---|---|---|
| `uygulanmis` | applied | `yazıldı` | written |
| `değişiklik yok` | no change | `zaten` | already |
| `GECTI` | passed | `RED` | refused |
| `KURU` | dry (apply and verify, do not serve) | `taslak` | draft |
| `kaynak` | source | `hedef` | target |
| `dusurulen` | dropped / superseded | `bolunen_kda` | split KDA layers |
| `kapi` | gate | `etiket` | tag |
| `yama` | patch | `capa` | anchor |
| `taban` | base | `kopya` | copy |
| `ince` / `butce` | fine (grid) / budget | `devre disi` | disabled |
| `yazildi_defteri` | the written-parts ledger | | |

---

## The 14 patches

Each file is `yama-<name>.py` in [`patches/`](patches/). All share the engine `harem_yama.py`: the base
`sha256` of every file it touches is pinned to `21d93d0d8`; every anchor must occur **exactly once** in the
pristine file; the new text must occur exactly once in a patched one; a second `--apply` writes no byte;
`--revert` restores the base hash; writes are atomic and a Python file is compiled before it is written.
Anything else exits non-zero and the prelude stops the rank, because a half-patched stack is fluent and wrong.
"Upstream" below means vLLM `main` at `21d93d0d8`; "ours" means no upstream equivalent was known on 6 October 2026.

| Patch | Changes in vLLM | Why | Upstream status | Fails closed how |
|---|---|---|---|---|
| `tp3-dolgu` | `parameter.py`, `weight_utils.py`, `vocab_parallel_embedding.py`: `narrow` becomes `_harem_pad_then_narrow` at the six places a TP shard is cut from a checkpoint tensor; the vocabulary padding unit becomes 128 × tp | five shapes of GLM-5.3-Flash do not divide by three and an EXL3 trellis cannot be zero-extended ([docs/03](../../docs/03-tp3-padding-and-sidecars.md)) | ours, TP=3 specific | refuses a shard that would be padding only |
| `glm5next-tamkapsam` | `glm5next/common/model.py`, `kda.py`: the shared-expert width from the plugin's hook; the EXL3 quant config is handed to MLA and KDA | upstream builds MLA and KDA with `quant_config=None` (written for the fp8 checkpoint) | ours | the gate is `quant_config.get_name() == "exl3"`; any other checkpoint is untouched |
| `sidecar-dizin` | `default_loader.py`: a checkpoint index with `metadata.harem_sidecar` becomes authoritative | vLLM yields every tensor of every shard; the sidecar splits fused tensors offline | ours | a tensor in neither the index nor the superseded list is an error; the skipped count must equal the declared count |
| `mambagrid` | `scheduler.py`: align-mode prefill chunks end on the KDA state grid (3,328) instead of the 256 grid; a guard for a wide fine grid | a chunk ending off the state grid writes the wrong moment's state, and a prefix hit resumes from a stale state | the scheduler half of [vLLM #54076](https://github.com/vllm-project/vllm/pull/54076), **still open** on 9 October 2026 | `HAREM_MAMBA_GRID=1`; several mamba block sizes raise at startup |
| `kpool-init` | `sparse_indexer.py`, `kpool_compress.py`: the top-k buffer is `-1`-filled instead of uninitialised, and the reader is bounded above | stale memory expanded into real cache rows | [Zeuss5/cuda-exl3#6](https://github.com/Zeuss5/cuda-exl3/issues/6), fix after tpurtell (Apache-2.0); not in upstream (the original text is still there) | anchors |
| `epfilter` | `ep_weight_filter.py`: `.trellis` is skipped by the expert-parallel weight filter | 99.8 % of an expert's bytes are in `<proj>.trellis`; without it every rank reads all 288 experts ([docs/08](../../docs/08-fast-boot.md)) | ours | `HAREM_EP_FILTER_SUFFIXES` entries must start with `.` |
| `fastload` | `base_loader.py` plus two copied modules: per-rank weight sidecar, dump and restore | the slicing at TP=3 + EP is ~150,000 small strided copies; this is one sequential read ([docs/08](../../docs/08-fast-boot.md)) | ours | identity, name, shape, dtype, size or hash mismatch raises; no silent fall-back to a normal load |
| `swblock` | `attention.py`: `HAREM_SW_BLOCK_SIZE` pins the drafter's sliding-window kernel block | upstream now picks the largest block that fits (3,328 here), the geometry we rejected on the old stack | ours; upstream's choice changed direction at `21d93d0d8` | not a multiple of the backend's smallest block raises |
| `pdl` | `platforms/cuda.py`: PDL off on sm_12x | PDL is not qualified for GB10 and KDA state races were reported | [Zeuss5/cuda-exl3#6](https://github.com/Zeuss5/cuda-exl3/issues/6), tpurtell (Apache-2.0); upstream still returns `major >= 9` | `HAREM_PDL_SM12=1` restores upstream |
| `glm47` | `parser/glm47_moe.py`: the tool-call parser validates a call whole (name, keys, schema, required keys) and sends a rejected call to content | a salvaged malformed call is echoed back by the client and imitated by the model ([issue #7](https://github.com/NNNtrance/GLM-5.3-Flash-EXL3-TP3-3x-DGX-Spark/issues/7)) | ours; upstream salvages. Changes a visible API behaviour (a valid streaming call arrives as one delta) | `HAREM_GLM47_FAILCLOSED=0` is exactly upstream; prints a status line at import |
| `dflash-tp3` | `qwen3_dflash2.py`: wraps `load_weights` and proves the padded draft is exactly zero in its pad rows; refuses a quantised draft | the draft is padded 32/8 → 36/9 at tp=3 | ours; upstream has no such check | no escape hatch; any mismatch raises |
| `dflash-eagle3` | `glm5next/common/model.py`: the EAGLE3 interface on the GLM-5.3 target | upstream does not implement `SupportsEagle3` for GLM-5.3, and the DFlash driver asks for it ([docs/04](../../docs/04-dflash2-port.md) §3.1) | a merge of a vLLM fork's commit `e7097feb6`; not upstream | with no draft the upstream path is followed |
| `dflash-kvgrup` | `kv_cache_utils.py`: the drafter's layers get an independent KV group, are counted in the byte arithmetic, and (`HAREM_PREFIX_HIT=1`) only they are flagged `is_eagle_group` | the generic path cannot group a Mamba + MLA + sliding-window mix; flagging every group drops an aligned target block on each prefix hit | the idea of a vLLM fork's `_partition_dflash_draft_specs`; the flagging follows [#52047](https://github.com/vllm-project/vllm/pull/52047) and the idea of #54041 | dies if the drafter's groups are not all sliding-window or a target group is already flagged |
| `gorsel` | `glm5next/common/model.py`, `multimodal.py`: the EXL3 vision tower, the `_proj` weight spellings, a post-load audit (99 EXL3 linears, 0 unquantised), the per-prompt video limit, a front-end check that placeholders equal encoder rows | the tower is 6-bit EXL3 in the checkpoint; a placeholder mismatch on the GPU kills the engine on every rank | ours; one part retired because upstream has it ([#55647](https://github.com/vllm-project/vllm/pull/55647)) | a mismatch rejects one request with HTTP 400 instead of killing the engine; `HAREM_VISION` empty is upstream |

The other files: `harem-prelude.sh` runs the 14 in this order and then the draft-tag gate, and `exec`s `vllm serve`;
`harem_taslak_kapi.py` refuses to boot when the draft is untagged or tagged for another tp (it also runs in
load mode, where the engine's own check does not); `harem_fastload.py` and `harem_fastload_id.py` are the
modules `fastload` copies into vLLM (the identity covers the image tag, TP/EP, rank, the checkpoint files,
`yama-*.py`, the engine, the prelude, vLLM's and the plugin's versions and the plugin's `.py` sources).

**Patches of the old tracks that this base made unnecessary** (retired in writing, not deleted quietly):
the K-pool seed kernel [#57477](https://github.com/vllm-project/vllm/pull/57477), the tail ring
[#58454](https://github.com/vllm-project/vllm/pull/58454), the token-id guard
[#51795](https://github.com/vllm-project/vllm/pull/51795) and [#54196](https://github.com/vllm-project/vllm/pull/54196),
the XGrammar backports [#52805](https://github.com/vllm-project/vllm/pull/52805) and
[#53046](https://github.com/vllm-project/vllm/pull/53046), the TileLang fail-loud patch, FlashKDA
[#55737](https://github.com/vllm-project/vllm/pull/55737) and [#58846](https://github.com/vllm-project/vllm/pull/58846),
and the sparse-indexer workspace bound [#57701](https://github.com/vllm-project/vllm/pull/57701) with
[#55222](https://github.com/vllm-project/vllm/pull/55222) are in upstream at `21d93d0d8`; the
expert-parallel loader moved into the plugin; the drafter's fp8 KV cache is now a key of
`--speculative-config`. The old prefix-hit annotation was **not** retired: it is `dflash-kvgrup`. The old K-pool
tail backport (#57317, #57534) is in the base and was not ported `[not tested]` on its own on this stack.

---

## Two switches that are only configuration

Neither changes a line of code; both are keys in `--speculative-config`, carried by `SPEC_EXTRA_JSON` and
validated by [`tools/spec_extra.py`](tools/spec_extra.py) before any container starts. The measurements are in
`results/main-stack/`; this page states the mechanism and the price.

**`disable_eagle_block_drop: true`.** With a drafter on, vLLM drops the last matched prefix block on every
prefix-cache hit (the EAGLE convention), and on this stack the scheduler block is 3,328 tokens, so an agent's
follow-up turn re-read up to two whole blocks it had just computed, and a short tool prompt read nothing from
the cache at all. DFlash2's draft context is aligned token for token with the target, so the drop protects
nothing here (upstream added the key as [#53388](https://github.com/vllm-project/vllm/pull/53388)). With it
set the hit becomes ⌊P/256⌋·256: the first token of follow-up turns went from 2.34 to 1.24 s serially and from
7.25 to 3.83 s with five sessions at once, and tokens re-read fell 61 %. **What it cost:** nothing we found,
and it was looked for: speed over six boots ×1.0013 at one user and ×0.9993 at four, warm and cold answers
agree at seven probe points, 480 strict tool-call turns with 0 corrupted `[measured-here, private harness]`
([`prefix-hits-nodrop.md`](../../results/main-stack/prefix-hits-nodrop.md)). The boot prints the warning line
listed above.

**`num_speculative_tokens_per_batch_size: [[1,1,7],[2,8,3]]`.** Each triple is `[first batch size, last batch
size, k]`: the draft proposes 7 tokens while one request is running and 3 while two to eight are (vLLM's
dynamic speculation, `v1/spec_decode/dynamic/utils.py`; the tool rejects a `k` above `num_speculative_tokens`,
which vLLM would silently clip). A long draft is a bet that pays when the verifier is idle and loses when it is
shared: at four users the step fell from 160.2 to 114.5 ms, and the pooled output rate rose +10.7, +15.6 and
+7.1 % at two, four and five users, one user unchanged. **What it cost:** mathematics prompts run 12 to 17 %
slower at two or more users, where the draft accepts well and a shorter draft wastes it
`[measured-here]` ([`draft-schedule.md`](../../results/main-stack/draft-schedule.md)).

To run without either, empty `SPEC_EXTRA_JSON` in the env file and reboot all three nodes; no new dump boot is
needed (the identity does not cover the speculative configuration).

## Memory fraction

0.84 is where this stack runs, not a wall. 0.85 gave a KV pool of 6,354,223 tokens, but under the worst request
we allow (sixteen large images, 126,797 prompt tokens, five users) the head fell to 882 MiB free and our 1 GiB
safety line stopped the engine; 0.84 held 1,621 MiB on the same load. **What 0.84 costs:** 2.6 % of the pool
`[measured-here, private harness]` ([`memory-fraction-084.md`](../../results/main-stack/memory-fraction-084.md)).
We measured at 0.75, tried 0.85 and settled on 0.84. The way to find your own value is to start where the
setup boots with room to spare, move up in 0.01 steps while the free-memory floor stays above 1 GiB in the worst
case, and compare two values the way [docs/09](../../docs/09-measurement-protocol.md) says.
`GMU` is not part of the fast-load identity: changing it costs a reboot, not a dump boot.

---

## Rollback

From the smallest step to the largest. The first three keep the corrected routing.

| You want to | Do | Costs |
|---|---|---|
| lower the memory fraction | `GMU=0.83` in the env files, reboot all three | a smaller KV pool; no dump boot |
| run without the two switches | empty `SPEC_EXTRA_JSON`, reboot all three | the gains above; no dump boot |
| take the watchdog away | `sudo systemctl disable --now harem-watchdog.timer` | |
| undo the drop-in on a node that kept the old unit | delete `harem-exl3.service.d/tp3-main.conf`, `daemon-reload`, reboot all three | you are on the old stack |
| the old stack | the old tracks are still in this repository. **Their routing is wrong and their quality figures are withdrawn**: use them to compare, not to serve | |

Reboot **all three nodes together** after any of these ([docs/00](../../docs/00-hardware-and-os.md) §3.4).

## What this track cost

| Gain | Price |
|---|---|
| Correct expert routing (the point) | decode steps 11 % longer at one stream and 20 % at four, because correct routing reads 28 to 38 % more distinct experts per step `[measured-here]` ([`speed-map.md`](../../results/main-stack/speed-map.md)) |
| `disable_eagle_block_drop` | looked for, none found |
| The draft schedule | mathematics 12 to 17 % slower at two or more users |
| 0.84 instead of 0.85 | 2.6 % of the KV pool |
| Fail-closed parser | a valid streaming tool call arrives as one delta, not as a name delta plus argument deltas |
| The boot gates | a precondition script of 12 s on the head and 62 s on the workers at every boot `[measured-here]`; a refused boot instead of a fluent wrong one |
| A watchdog | three reboots, about 8 minutes out, for every real engine failure, and every in-flight request lost `[measured-here]` ([`boot-and-watchdog.md`](../../results/main-stack/boot-and-watchdog.md) §5) |

## What we tried and rejected

- **0.85**, above. `[measured-here]`.
- **A CUDA graph for the draft through FlashInfer XQA** (a patch of ours): correct and measured, step −2.56 % at one
  user and −1.35 % at four, but KV pool −6.3 % and a slower first token for short prompts, so it is not shipped
  `[measured-here]` ([`rejected-draft-cuda-graph.md`](../../results/main-stack/rejected-draft-cuda-graph.md)).
- **`index_topk` 8192**, which the old tracks promoted on 12 September: this track runs the checkpoint's 2048.
  Whether 8192 is still needed on the corrected stack has not been measured `[not tested]`
  ([docs/20](../../docs/20-main-stack.md) §2).

## Open problems

- Every quality benchmark of the old tracks is withdrawn and **not re-measured on this stack**; the 1M-token
  needle and the long-context stress runs likewise `[not tested]`.
- The watchdog's silent-hang path has been exercised only against a mock engine; the head has no watcher
  ([`boot-and-watchdog.md`](../../results/main-stack/boot-and-watchdog.md) §6).
- The mesh plugin does not abort on its own timeout: a rank more than the configured timeout behind hangs the
  cluster silently. `NCCL_MESH_TIMEOUT_SEC=180` and the JIT gate prevent the case we hit, not the behaviour.
- Builds are not bit-identical, so the image Id is yours and the preflight needs it set per cluster.
- `mambagrid` is a port of [#54076](https://github.com/vllm-project/vllm/pull/54076), still open on 9 October 2026; check whether it has merged before you carry it.
- No two-node arrangement has been built on this stack.

## Retracted

This track retracts nothing by itself. The retraction it answers is the routing defect, in
[docs/11](../../docs/11-open-issues.md) §1.15: every quality figure of the old tracks was measured on routing that
chose the wrong experts and is withdrawn until re-measured.

## Credits

Where a patch adapts someone else's idea or fix, the patch table names it: the kernel project [`Zeuss5/cuda-exl3`](https://github.com/Zeuss5/cuda-exl3) (and tpurtell's
sm_12x findings, Apache-2.0), the vLLM pull requests above, and a vLLM fork's DFlash2 commit; full entries in
[CREDITS.md](../../CREDITS.md). Our patches are written by us for this recipe; use them freely (Apache-2.0); a
credit is appreciated.
