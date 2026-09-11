# 14 — Troubleshooting: every failure we hit, by symptom

**Applies to: both tracks.** Every entry carries its own **Track** line — see the third convention
below.

Eighty-seven entries, with the exact text where one exists. This page is the reason the rest of the
repository can be short: if you are stuck, the answer is probably here, and if it is not, the shape of
the answer probably is. **Two of the eighty-seven are not failures we hit**, and each says so in its own
heading: §9.8 is a corruption the code blocks before it can happen, and §9.9 is a hypothesis a
measurement refuted before it cost us anything. They are counted and kept because this is where a
reader will look for them.

**Read §0 first if something is wrong right now.** Everything after it is organised by subsystem, and
§1 indexes it by what you actually see on your screen.

Three conventions. Nodes are `head` (rank 0, serves the API), `worker-1` and `worker-2`; addresses
are documentation addresses. **`no log line — silent`** means exactly that: the failure produced no
message at all — those are marked **[SILENT]** and collected in §9, because they are the expensive
ones.

And every entry opens with a **Track** line, because this repository documents two node counts
([docs/00 — Start here](00-start-here.md)). It reads one of four ways, and the third is the one worth
understanding:

| Track line | What it means |
|---|---|
| **both** | The failure is a property of the image, the part, the fabric, the checkpoint, the model or the harness. Nothing in it depends on how many ranks are running |
| **both, measured at TP=3 only** | The mechanism is rank-independent, but every reading behind the entry came from a three-node arm. Expect it at two ranks; do not expect our numbers |
| **TP=3 only** | It cannot happen at two ranks: it is about the padding, the padded load, or expert parallelism being mandatory |
| **TP=2 only** | We only ever hit it at two ranks, and the entry says why three did not |

Across the page, **45** entries are plain **both** and another **27** are both-but-measured-at-three
ranks. That is a useful finding on its own: most of what goes wrong on this stack does not care how
many nodes you have. Thirteen are TP=3 only and two are TP=2 only.

---

## 0. Triage — the order that saves the most time

Work down. Each step rules out everything after it.

| # | Check | Command | Expected |
|---|---|---|---|
| 1 | Is the fabric there? | `ibv_devinfo \| grep -c PORT_ACTIVE` | `4` on **every** node |
| 2 | Is the fabric *used*? | `for p in /sys/class/infiniband/*/ports/1/counters/port_xmit_data; do echo "$p $(cat $p)"; done` | all **four** counters move, not two (§4.3) |
| 3 | Is the slot what you think? | `sudo lspci -vv \| grep -A2 ConnectX` | `Speed 32GT/s, Width x4` (§4 intro) |
| 4 | Did the container even start? | `docker ps -a` | not `Exited (21)` / `(23)` / `(30)` / `(31)` — those are our own gates, §2.16, §3.1 |
| 5 | Did the boot gates print? | `docker logs <container> 2>&1 \| grep -E '\[padload\]\|assert 5\|EXL3 routed experts'` | all present (§7.3, §7.5) |
| 6 | Is it correct? | `python3 scripts/correctness-probe.py` and `scripts/code-exam.py` | 10/10 and 12/12, **cold and warm** |
| 7 | Is the pool real? | `docker logs <container> 2>&1 \| grep 'Free memory on device'` | all three ranks within **1 GiB** (§5.8) |
| 8 | Only now, speed | `audit/run-audit.sh` | see [audit/](../audit/README.md) |

**A fast engine that answers wrong is worthless, and a machine that is swapping produces speed numbers
that mean nothing.** That is why correctness and memory come before timing.

---

## 1. Index by symptom

| What you see | Go to |
|---|---|
| `persistent_topk would oversubscribe … requires >=128KB smem` | §2.1 |
| `Assertion error (…deepgemm…/attention.hpp:320)` | §2.2 |
| `RuntimeError: Model does not support EAGLE3 interface` | §2.5 |
| `page size is not divisible by the maximum page size and cannot be padded` | §2.6 |
| `AssertionError: 16 is not divisible by 3` | §2.7 |
| `num_attention_heads must be divisible by world_size` | §2.8 |
| `ValueError: 154880 is not divisible by 3` | §2.8 |
| `assert self.num_heads % self.tp_size == 0` (in `kda.py`) | §2.8 |
| `refusing to zero-extend … the entire shard would be padding` | §2.10 |
| `"no safetensors found"` although the weights are right there | §2.11 |
| `tensor parallelism would slice the routed-expert trellis to 682 columns per rank` | §2.12 |
| `gate_proj trellis has 128 tiles on dim 1, expected 384` | §2.12 |
| `KeyError: layers.2.self_attn.o_proj.mul1` | §2.13 |
| `KeyError: layers.1.self_attn.conv1d.weight` | §2.13 |
| `Tried to load weights of size torch.Size([1536]) to a parameter of size torch.Size([1, 1536])` | §2.14 |
| `RuntimeError: start (…) + length (…) exceeds dimension size` in a `narrow` | §2.15 |
| `EXL3 weights cannot be zero-extended` | §2.16 |
| `copy_` shape mismatch in `_vocab_loaders` | §2.16 |
| `ValueError: could not find tensor_storage` | §2.17 |
| `fatal error: cusolverDn.h: No such file or directory` | §2.21 |
| `preflight-fastload: sidecar stale - boot refused` | §3.1, §10.6 |
| Container exits immediately, `Exited (21)` on all three nodes | §3.1 |
| `fast-load sidecar missing: …` and all units `failed` before the container starts | §10.6 |
| A boot log with no `[HAREM-FLASHKDA] kda_prefill_backend=` line | §10.6 |
| `ValueError: 6.6 GiB KV needed for max seq len 1,000,000, available 0.73 GiB` | §5.3 |
| `Setting attention block size to 3328 tokens…` | §5.2 |
| Request accepted, then `Running: 0, Waiting: 1` forever | §5.4 |
| KV pool far smaller than expected, no error | §5.1, §5.8 |
| `undefined reference to ibv_event_type_str` | §6.8 |
| All-reduce collapses between 200 KB and 12 MB | §6.1 |
| `exl3_moe_gemm: svh n mismatch` | §7.2 |
| `OutOfResources: out of resource: shared memory, Required: 106496` | §7.3 |
| `min_reg_num < INT64_MAX is false` | §7.4 |
| `CUDAGraphMode.FULL_AND_PIECEWISE is not supported with spec-decode` | §7.7 |
| `POST /start_profile` returns 404 | §8.7 |
| Three nodes slower than two | §7.1 |
| Model serves and is confidently wrong | §2.10, §9 |
| Draft acceptance falls with no error | §9 (several) |

---

## 2. Boot and load

### 2.1 First EXL3 boot dies in the sparse-attention indexer's persistent top-k

**Track:** both — vLLM's own kernel against 48 SMs; a property of the part.

**Symptom.** The engine never reaches serving; engine core init fails on all ranks.

```
RuntimeError: launch_persistent_topk, /workspace/csrc/libtorch_stable/topk.cu:138, persistent_topk would oversubscribe and the FilteredTopK fallback requires >=128KB smem pe
```

(truncated at 220 characters by our log capture; the full form reads
`persistent_topk would oversubscribe and the FilteredTopK fallback requires >=128KB smem per block (have 101376). total_ctas=85 > num_sms*occupancy=48 (TopK=512, vec_size=4, ctas_per_group=85, smem=49152)`)
followed by

```
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}
```

**Cause.** With `select_k = index_topk / index_kpool = 2048/4 = 512`, `sparse_attn_indexer_kpool.py`
calls `torch.ops._C.persistent_topk`. GB10 has 48 SMs and 101,376 B of opt-in shared memory; the
fallback wants ≥128 KB. **This is vLLM's kernel, not `cuda-exl3`'s.**

**Fix.** The `HAREM-GB10-TOPK` overlay — two env-gated lines that skip the persistent path and fall
through to `top_k_per_row_decode`. Set `HAREM_DISABLE_PERSISTENT_TOPK=1`.

**Bonus.** The fallback is *faster* at our shapes (5.0 vs 7.0 µs at 8 rows / 512 pools). The
persistent path only wins above ~16K candidate pools ≈ 64K tokens of context.

### 2.2 MTP with fp8 KV dies in DeepGEMM unless `--block-size 256`

**Track:** both — the DeepGEMM arch-12 `block_kv` constraint, mandatory at two ranks too.

```
RuntimeError: Worker failed with error 'Assertion error (/workspace/.deps/deepgemm-src/csrc/apis/attention.hpp:320): (arch_major == 10 and (block_kv == 32 or block_kv == 64 or block_kv == 128)) or (a
```

**Cause.** For arch 12 with non-fp4 KV, DeepGEMM requires `block_kv` to be exactly 64. With
`index_kpool = 4` that means `--block-size 256`.

**Fix.** `--block-size 256`, which is load-bearing for a second reason too (§2.3).

### 2.3 `--block-size 256` is load-bearing for the multi-group prefix cache

**Track:** both — the prefix-cache hash-block rule plus the drafter's second KV group.

Not a failure we hit — a rule found by reading. `v1/core/kv_cache_coordinator.py` asserts that every
KV cache group in prefix caching satisfies `block_size % hash_block_size == 0`, and a DFlash drafter
adds a **second** group (sliding window, 2048). Keep 256; an incompatible value fails loudly.

### 2.4 The base image cannot run DFlash2 — the drafter is not in the registry

**Track:** both — the image ships DFlash v1; a registry property.

**Cause.** The base image ships DFlash **v1**; `DFlash2DraftModel` exists only in the fork.
**Fix.** The port layer, `patches/dflash2-port/`. See [04](04-dflash2-port.md).

### 2.5 Target side has no EAGLE3 interface

**Track:** both — the model file lacks `SupportsEagle3`; no rank term in it.

```
RuntimeError: Model does not support EAGLE3 interface
```

raised in `v1/worker/gpu/spec_decode/eagle/eagle3_utils.py`, **after** the weights have loaded.

**Cause.** DFlash is not self-contained: the drafter reads the target's intermediate layers, so the
target class must expose `SupportsEagle3`. The image's model file has zero aux-hidden-state support.
**DFlash v1 could not have worked here either.**

**Fix.** Three-way merge of the fork's change onto the model file. Proof at runtime — exactly
`target_layer_ids + 1`:

```
Using Eagle3 auxiliary layers from config: (6, 15, 25, 34, 43)
```

### 2.6 KV cache grouping — the indexer page cannot be padded

**Track:** both — the draft's sliding-window layers break the MLA-only grouping path.

```
NotImplementedError: Layer language_model.model.layers.3.self_attn.indexer.k_cache: page size is not divisible by the maximum page size and cannot be padded.
```

**Cause.** The indexer's `k_cache` is 33 B/token and can never reach 512 B/token by integer block
growth. GLM-5.3 normally takes a dedicated grouping path that never unifies pages, but that path
requires every attention spec to be an `MLAAttentionSpec`; the draft's sliding-window layers fail
that test and the model falls through to the generic path.

**Fix.** Partition the draft's KV layers from the target's, group each independently, concatenate.
Costs +0.6 % of per-block bytes. Proof at runtime:

```
DFlash draft: 5 KV layers kept in 1 independent cache group(s)
```

**This is also what later capped the KV pool** — see §5.1.

### 2.7 The vision tower asserts at TP=3, and `--language-model-only` does not stop it being built

**Track:** both — an assert at TP=3, a built-but-unused tower at TP=2.

```
AssertionError: 16 is not divisible by 3
```

in `Glm5NextVisionAttention.__init__`.

**Cause.** The model file builds the vision transformer **unconditionally**. `--language-model-only`
only makes the multimodal limit return 0 — no image can be *submitted*; the tower is still *built*.
At TP=2, `divide(16, 2)` succeeds and the boot continues. At TP=3 it asserts and the engine never
starts.

**Two fixes, and which one you want depends on what you are serving.**

- **You want images and video:** turn the tower on properly —
  [`tracks/tp3/patches/vision/`](../tracks/tp3/patches/vision/README.md) and
  [18](18-vision-at-three-ranks.md). That is the production recipe since 7 September 2026. The assert
  itself is cleared by `--mm-encoder-tp-mode data`, but that flag **on its own is not enough**: see
  §2.7a below, which is the failure you get next.
- **You want text only:** three anchors in `patch-vllm-tp3.py` build no tower at all. Confirmation:

```
HAREM-TP3: --language-model-only is set, so the GLM-5.3 vision tower is not built and its checkpoint tensors are skipped.
```

**`[retracted]` — this entry used to say the TP=2 cost is "1.05 GiB of unused BF16 vision weights per
rank, silently".** That figure belongs to the routed-experts-only checkpoint; on the full-scope
checkpoint the tower is 6-bit EXL3 at 0.416 GiB per rank, and on image `exl3-zeus:754421f` the flag
may well cost nothing at all, because the meta-device path leaves the tensors unmaterialised
`[not tested]`. [11](11-open-issues.md) §1.14.

### 2.7a `There is no module or parameter named visual.blocks.N.attn.proj.weight` **[after clearing the assert]**

**Track:** both — it is a property of the **checkpoint**, not of the rank count.

You set `--mm-encoder-tp-mode data`, the assert is gone, and the load dies looking for dense weights.

**Cause.** vLLM builds the tower with `quant_config=None` unconditionally
(`models/glm5next/nvidia/model.py`), with a comment that is correct for the zai-org **fp8** checkpoint
it was written against. In `turboderp/GLM-5.3-Flash-exl3` at 4.05 bpw the tower is **6-bit EXL3** —
`proj`, `mlp.*` and `merger.*` have no dense `.weight` on disk at all. Built dense, it asks for
tensors that do not exist.

**A second, quieter half.** With `HAREM_EXL3_FULLSCOPE=1` the full-scope patch replaces the class's
whole `packed_modules_mapping`, which is where the inherited `qkv_proj -> q/k/v_proj` entry lived.
Without it **24 of the 99 vision linears do not resolve and fall back to BF16 silently** —
`Exl3Config.get_quant_method` fails *open*, so this loads clean and answers wrong.

**Fix.** VS1 and VS2 of [`patches/vision/patch-vision-tp3.py`](../tracks/tp3/patches/vision/patch-vision-tp3.py)
for the first half; `CUDA_EXL3_PACKED_MAPPING={"qkv_proj":["q_proj","k_proj","v_proj"]}` — no patch,
merged **under** the class mapping so it can only add — for the second. Prove both model-free before
you boot, with `check-vision-mapping.py`. [18](18-vision-at-three-ranks.md) §3.

### 2.7b `Attempted to assign N = N multimodal tokens to M placeholders` **[KILLS THE ENGINE CORE]**

**Track:** both, and it is **upstream** — not ours, not EXL3's, not TP=3's.

```
File ".../vllm/v1/worker/gpu/mm/encoder_runner.py", line 285, in get_inputs_embeds
File ".../vllm/model_executor/models/interfaces.py", line 450, in embed_input_ids
File ".../vllm/model_executor/models/utils.py", line 657, in _merge_multimodal_embeddings
RuntimeError: shape mismatch: value tensor of shape [1656, 4096] cannot be broadcast to
              indexing result of shape [1728, 4096]
ValueError: Attempted to assign 1656 = 1656 multimodal tokens to 1728 placeholders
```

The worker dies, the engine core dies, every later request gets `EngineDeadError` and HTTP 500. All
ranks, same second. **The first video request does this.** Images are unaffected, which is exactly
why an all-green image gate proves nothing about video.

**Cause.** A video goes through **two independent frame samplers**: the pixels through
GLM-5-Next's (`target_fps = 2.0`, `extract_t = int(duration × fps)`), the prompt placeholders through
GLM-4.6V's (`target_fps = 3`, `extract_t = int(duration × fps × temporal_patch)`), because
`Glm5NextProcessingInfo` overrides four methods and not `_get_video_second_idx_glm46v`. A
four-second clip gives a grid of 4 and **12** timestamps: 4,968 placeholders for 1,656 encoder rows.

**Do not read the numbers in the trace as totals.** Under chunked prefill at
`--max-num-batched-tokens 2048` they are chunk-local counts (1,728 and 1,768 are pieces of 4,968).
Reading 1,728 as `4 × 36 × 48 / 4` and concluding the width rounded 46 → 48 is wrong: both sides read
`H` and `W` from the *same* grid tensor.

**Not the cause, and both were suspected first:** the `max_pixels` brake (it never reaches the video
processor at all) and the data-parallel encoder (it splits per item, and the encoder produced exactly
the right number of rows).

**It is not only short clips, and the ratio is not a constant.** The mismatch
`len(timestamps) / grid_t` tracks the clip's duration through GLM-4.6V's `DYNAMIC_FPS_THRES`:
**3.0 below 30 s**, **1.0 between 30 and 300 s** — the only band where the two samplers happen to
agree — and **0.5 or lower above 300 s**, where there are now *fewer* placeholders than encoder rows.
Both directions are fatal. A stack that has only ever been tested with a two-minute clip will look
fine and die on the first short one.

**Fix.** VS4 in [`patches/vision/`](../tracks/tp3/patches/vision/README.md) derives the timestamps
from the pixel path's own sampler, which makes the equality structural. VS6 makes any future mismatch
an HTTP 400 in the frontend instead of three dead ranks. And run `check-video-geometry.py` in the
prelude: model-free, under a second, and it refuses the boot. [18](18-vision-at-three-ranks.md) §4.

**Upstream:** filed as [vllm#55644](https://github.com/vllm-project/vllm/issues/55644) with the fix
and a unit test in [vllm#55647](https://github.com/vllm-project/vllm/pull/55647).

### 2.7c `At most 1 video(s) may be provided in one prompt` when you asked for 2 **[SILENT until you try]**

**Track:** both.

The boot log prints `'limit_mm_per_prompt': {'image': 4, 'video': 2}` and the limit is **1**.
`Glm4vProcessingInfo.get_supported_mm_limits` declares `{"image": None, "video": 1}` and
`multimodal/processing/context.py` takes `min(user_limit, supported_limit)`. The 1 is a conservative
default, not a real limit: videos are processed one at a time and the placeholder and mrope paths are
both per item. **Fix:** VS7, `HAREM_VISION_VIDEO_LIMIT` (default 2).

Related, and survivable: a clip shorter than about **0.5 s** makes `extract_t = int(duration × 2.0)`
collapse to 0, the sampler return an empty list, and the frontend raise `IndexError` — HTTP 400 with
the engine alive.

### 2.7d A patch anchor matched zero times and the rank stopped with exit 21 — while the model-free gate said PASS

**Track:** both, and the lesson is general: it is about **gate design**, not about vision.

A patch script that fails closed did exactly what it should — an anchor matched zero times, the
script refused, and all three ranks stopped with exit 21 before a byte of weight was read. The
confusing part was that the model-free gate had passed the same patch minutes earlier.

**Cause.** The gate ran `patch-fullscope-tp3.py` and then the vision patch, and called that the
chain. The **prelude** runs `patch-vllm-tp3.py` long before either, and one of its edits rewrites the
first two lines of the very block the vision anchor was written against. The gate and the boot were
patching different bytes.

**Fix, and the rule.** [`verify-cpu.sh`](../tracks/tp3/patches/vision/verify-cpu.sh) now runs the
**full prelude order**, with the same environment knobs and the same read-only overlay bind-mount as
the launcher. *A gate that does not reproduce the boot is not a gate.* Two companions, both real:

- **`set -eo pipefail` is not optional in a gate.** Every check in that script ends in `| tail -1`,
  and without `pipefail` a run printing `[vision] FAIL` exits **0**. Add it, then confirm by watching
  a failing gate actually fail — an untested gate is an assumption.
- **Do not assert a count you derived from metadata.** "48 dead fused tensors must be dropped" was
  true of the safetensors headers and false of the load, which delivers 0 or 1 because
  `AutoWeightsLoader` filters unexpected `.bias` suffixes first. It killed a boot whose tower was
  entirely correct. Report counts; assert invariants.

[18](18-vision-at-three-ranks.md) §13.

### 2.8 The four hard asserts at TP=3

**Track:** **TP=3 only** — the four asserts are the 64 -> 66 and 154,880/3 divisibility failures.

GLM-5.3-Flash has 64 attention heads, 64 KDA heads, a 154,880-token vocabulary and a 2,048-wide
shared expert. **None divides by three.**

| Line | Cause | Fix |
|---|---|---|
| `AssertionError: num_attention_heads must be divisible by world_size` (`model.py:670`) | config not padded | run `pad-tp3.py`; check the model path points at the **sidecar** |
| `assert self.num_heads % self.tp_size == 0` in `kda.py` | `linear_attn_config.num_heads` still 64 | pad the **nested dict**, not just the flat field |
| `ValueError: 154880 is not divisible by 3` in `vocab_parallel_embedding.py` | `padding_size` still 64 | `patch-vllm-tp3.py` edit 3 |
| `AssertionError` in `load_merged_column_weight` on `...shared_experts.gate_up_proj`, or `2048 is not divisible by 3` | shared-expert pad missing | `patch-vllm-tp3.py` edit 4 |

Sidecar `config.json` pads heads 64 → 66 (22/rank, both KDA mirrors); `padding_size` 64 → **384**;
shared expert 2,048 → **2,304**.

### 2.9 The 2,112 / 192 constants are silently wrong for a quantized checkpoint **[SILENT, half]**

**Track:** **TP=3 only** — `lcm(64, tp)` half-block pads; both constants are provable no-ops at TP<=2.

At production 9's checkpoint one path refuses loudly (the plugin's k-alignment refusal on `down_proj`
at k=704) and the other **corrupts quietly**: `gate_up_proj` gets a half-block output pad with only a
warning.

**Cause.** An EXL3 pad must occupy **whole 128-column Hadamard blocks**, because the Hadamard mixes
across each block before `svh` is applied. A pad sharing a block with real output does not merely
fail to be zero — **it corrupts the real columns beside it.** `lcm(64,3)` gives vocab 154,944 →
51,648 = **403.5** × 128 and shared expert 2,112 → 704 = **5.5** × 128 per rank.

**Fix.** Move both constants from `lcm(64, tp)` to `lcm(128, tp)`. **2,176 looks right and is not** —
it is 17 × 128, but 2176/3 is not an integer. Both changes are provable no-ops at TP≤2.

### 2.10 Padding heads to 96 instead of 66 — starts, serves, answers confidently wrong

**Track:** **TP=3 only** — padding heads to a multiple of the rank count exists only at three ranks.

```
ValueError: HAREM-TP3: refusing to zero-extend … the entire shard would be padding
```

**Cause.** Padding a head count to the next multiple of the TP size is safe **only while the last
rank still owns at least one real head.** 64 → 96 gives rank 2 heads 64–95, every one fabricated.

**Fix.** Pad to 66. `pad-tp3.py` refuses the 96 case by name. **[SILENT] without the check.**

### 2.11 The sidecar mount trap

**Track:** **TP=3 only** — the padding sidecar and the identity-mount check exist only at TP=3.

**Symptom.** Every weight link dangles, and the failure surfaces as a confusing `"no safetensors
found"` — **not as a mount error.**

**Cause.** The sidecars are trees of **relative** symlinks. Mount the sidecar anywhere else — an
obvious `/models/...` is the natural mistake — and `..` resolves to the sidecar itself.

**Fix.** **Identity mounting**: each sidecar mounted at its own host path inside the container.
`check_relative_sidecar()` in `scripts/start-tp3.sh` refuses any other arrangement and names the
reason. `pad-tp3.py --hardlink` is the alternative.

### 2.12 Expert parallelism not on every rank

**Track:** **TP=3 only** — expert parallelism is mandatory only at three ranks, because 2,048/3 is not an integer.

```
EXL3 <prefix>: tensor parallelism would slice the routed-expert trellis to 682 columns per rank…
```
```
EXL3 <prefix>: gate_proj trellis has 128 tiles on dim 1, expected 384
```

**Cause.** First: `ENABLE_EP=0` at TP=3 — 2048/3 is not an integer and a trellis column is meaningful
only on a 128-element Hadamard boundary. Second: the EP flag lost on **one** rank, or a
sidecar/original mismatch.

**Fix.** `ENABLE_EP=1`; all three ranks must carry `--enable-expert-parallel`; **tear all three down
before relaunching any of them.** The prelude refuses the first case up front.

### 2.13 The full-scope checkpoint's loader `KeyError`s

**Track:** both — S1-S3 are A1-A8, identical text in the TP=2 patch tree.

```
KeyError: layers.2.self_attn.o_proj.mul1
```
```
KeyError: layers.1.self_attn.conv1d.weight
```

**Cause.** Three independent layers, measured model-free on `torch.device("meta")` as **886 unmapped
tensors**:

- **S1** — the model class declares **no `packed_modules_mapping`**, so `Exl3Config` cannot invert
  vLLM's linear merges. Invisible for a routed-experts-only checkpoint, fatal for a full-scope one.
  886 → 526.
- **S2** — **the attention stack is pinned to BF16 in the model file.** One site passes
  `quant_config=None` with the comment "MLA projections are BF16 in checkpoint"; `kda.py` nulls
  `vllm_config.quant_config` for the duration of `super().__init__` and the base class captures it,
  so `self.quant_config` stays `None` **permanently**. Together they lock **72.8 %** of dense
  traffic. 526 → 170.
- **S3** — the KDA block is factorised differently: one EXL3 `qkv_proj` and one BF16
  `conv1d.weight` where `kda.py` expects three shards each. 170 → 0.

**The second `KeyError` is the informative one:** `conv1d` is a **BF16** tensor, so a BF16 copy of the
same checkpoint dies in the same place. **None of this is about quantization.**

**Fix.** `tracks/tp3/patches/patch-fullscope-tp3.py`, ten anchors, all gated on
`HAREM_EXL3_FULLSCOPE=1`; plus a rewritten `quantization_config.json` in the sidecar, or
`CUDA_EXL3_PACKED_MAPPING` from `fba9f27` onwards — **write the JSON with no spaces**, because the
launcher word-splits `EXTRA_ENV`.

### 2.14 `ReplicatedLinear` has no `weight_loader_v2` — a vLLM bug found on the meta device

**Track:** both — a vLLM replicated-linear bug, caught on a two-rank meta-device run.

```
AssertionError: Tried to load weights of size torch.Size([1536])
                to a parameter of size torch.Size([1, 1536])
```

**Cause.** The sparse indexer's `wq_b` is a `ReplicatedLinear`, and vLLM's `ReplicatedLinear` never
dispatches to `weight_loader_v2`; its only loader asserts `param.size() == loaded_weight.size()`,
which a per-shard `suh` of shape `(num_shards, k)` can never satisfy. **Any replicated linear served
by a v2-parameter quantization method fails at load.**

**Deeper.** `ReplicatedLinear` never calls `update_param_tp_status()`, so its parameters carry the
**global** TP rank — a replicated layer holds the whole tensor on every rank, so the copy has to pin
`tp_rank = 0` or rank 1 overruns on `narrow`. **A single-rank test could not have caught that; the
two-rank meta-device run did.**

**Fix.** Upstream fixed it better on the plugin side (`d19dee0`): the plugin now places a bare `(k,)`
into row 0 when the shapes say that is what happened.

### 2.15 The tuple-shard loader splits by the module's *padded* widths

**Track:** **TP=3 only** — A9: the padded `output_sizes` of 3 x 8,448 exist only at three ranks.

```
segment 0: narrow(start=0,     len=8448) -> 8448    correct
segment 1: narrow(start=8448,  len=8448) -> 16896   wrong (true start 8192)
segment 2: narrow(start=16896, len=8448) -> 25344   overflow -> RuntimeError
```

Generic form: `RuntimeError: start (…) + length (…) exceeds dimension size` inside a `narrow`.

**Cause.** vLLM's tuple-shard path slices the checkpoint's single 24,576-wide `qkv_proj` using offsets
built from the module's `output_sizes` — which at TP=3 are **padded**, 3 × 8,448 against a checkpoint
that is still 3 × 8,192.

**Fix.** Record the checkpoint's real per-shard width on the module, split by that, and let the
existing zero-pad path widen each rank's slice afterwards. **Read the tensor name in the traceback
and add it; do not widen the pad heuristic blindly.**

### 2.16 Padded EXL3 load needs three upstream commits

**Track:** **TP=3 only** — the padded load; the TP=2 full-scope arm ran on `62f53e6` without it.

- On anything older than `f3e3090`: `EXL3 weights cannot be zero-extended` (a hard refusal).
- On `f3e3090` alone: the boot passes `create_weights` and then **dies on a `copy_` shape mismatch in
  `_vocab_loaders`** — rank 2, 3,232 against 3,216 tiles.

**Cause.** An EXL3 tensor cannot be zero-extended (a trellis is not a dense tensor), and this
checkpoint has no BF16 head to fall back on.

**Fix.** All three of `f3e3090`, `754421f` and `807d798`. **No two of them are enough.** A prelude
gate reads all three capabilities out of the installed source and **exits 23 before a byte of weight
is read**:

```
[padload] cuda-exl3 padded-load support: padded-output-gate (f3e3090)=yes  vocab-loader-prefix (754421f)=yes  row-parallel-suh-pad (f3e3090)=yes
```

### 2.17 `Exl3Config` needs `_model_path_hint`

**Track:** both — explicitly not about padding; TP=2 needs the same `--hf-overrides`.

```
ValueError: could not find tensor_storage
```

**Fix.** Set the hint, or point it at the full file. With either it reads all **36,719** modules
cleanly — which is what proved the plugin side was sound and the problem was one layer above it.

### 2.18 `--safetensors-load-strategy prefetch` is refused

**Track:** both — the refusal keys on filesystem type and RAM fraction, not on rank.

The engine refuses it because the filesystem is not a recognised network FS and the checkpoint
exceeds 90 % of available RAM. **Fix.** Use `eager` (426.3 s → 189.7 s), then the fast-load sidecar
(→ 67.2 s). Note that `eager` has a memory cost — §5.7.

### 2.19 `git archive` silently drops the build recipe **[SILENT]**

**Track:** both — an untracked Dockerfile lost from the source tarball.

The upstream Dockerfile lives in the checkout's `docker/` directory and is **not tracked by git**.
`git archive` drops it silently and the build then fails on a missing file. **Fix.** Make the source
tarball with `tar`, excluding `.git`, `build/`, `__pycache__`, `*.egg-info`.

### 2.20 Upstream's `MAX_JOBS=12` does not fit under the build memory cap

**Track:** both — a build memory cap against a hardcoded `MAX_JOBS`.

Twelve parallel `nvcc` jobs do not fit under `--memory=4g`, and `MAX_JOBS=12` is hardcoded inside the
Dockerfile's `RUN` line — not an `ARG`, so it cannot be overridden. **Fix.** `sed` it to 3 **in the
extracted copy only**; editing the checkout makes every future build change silently while the
tarball sha stops describing what was compiled. Gate on ≥6 GiB free before starting.

### 2.21 The `cuda-exl3` build fails on a missing CUDA header

**Track:** both — a build-time CUDA include probe.

```
ATen/cuda/CUDAContextLight.h:16: fatal error: cusolverDn.h: No such file or directory
```

then `RuntimeError: Error compiling objects for extension`.

**Cause.** The include probe looks only for `cusparse.h`, but torch needs at least `cusolverDn.h` too.
**Fix (reported upstream, now fixed).** Probe for either, and add the wheel's include directory with
`-idirafter` so toolkit headers still win:

```
[cuda-exl3] toolkit lacks cusparse.h, cusolverDn.h; adding /usr/local/lib/python3.12/dist-packages/nvidia/cu13/include with -idirafter
```

---

## 3. The fast-load sidecar

### 3.1 The sidecar refuses the boot — and both times it was right

**Track:** both, measured at TP=3 only — the fast-load sidecar is per rank and was never built at TP=2.

**Symptom.** The container exits immediately on all three nodes; `docker ps -a` shows `Exited (21)`.

```
preflight-fastload: sidecar stale - boot refused
  patches.patch-fullscope-tp2.py: recorded='<none>' now='bca9a201...'
  patches.tp3-prelude.sh:         recorded='e17d46a4...' now='5eba79f3...'
```

The success form, for contrast:

```
preflight-fastload: OK  rank=0 dir=/var/tmp/glm53-exl3-tp3-r0 models=['dflash2-draft-tp3', 'glm-5.3-flash-tr3-4bpw-tp3'] 55.53 GiB
```

Exit codes: **30** when the sidecar is missing, empty or short a shard; **31** on an identity
mismatch, printing the first 20 differing fields.

**Cause.** The manifest identity hashes **every `patch-*.py` in the patch directory and the full text
of the prelude script**, not just the patches that touch a weight. Adding an experimental patch and an
env-gated prelude hook changed the identity without changing production behaviour by a byte.

**Fix.** Restore the prelude from its backup, unlink the experimental patch, reboot.

**Rules this produced.** Experimental patches live in **their own directory** and are **hard-linked**,
never copied. Nothing is added to, removed from or edited in the patch directory between a dump and a
load. Back up the prelude before hooking it and restore **from that backup** — the hash has to match
to the byte, and a semantically identical edit is not.

**Two earlier firings of the same gate.** (1) The patch directory was **edited while a boot was
running** — it is mounted live, so the hash changed between the model's dump and the drafter's. (2)
The drafter's `hf_overrides` is a **function object** whose `repr` carries a fresh heap address every
boot, so the drafter's sidecar looked stale the instant it was written; the address is now stripped
from both sides of the comparison.

### 3.2 The identity gate is stricter than it needs to be, and it cost an hour

**Track:** both, measured at TP=3 only — the same manifest identity, never exercised at two ranks.

Three patches that touch no weight byte — a KV-page knob, a fail-loud import guard, a draft-dtype
override — refused a boot, because **the gate hashes the directory, not the call graph.**
Cost: **one hour** of downtime and a **682-second** dump boot per node. The fix (an explicit
allow-list plus the prelude contributing its ordered list of patch invocations rather than its full
text) is designed and **not written**.

### 3.3 Any image change invalidates the sidecar

**Track:** both, measured at TP=3 only — the manifest records the image tag; no sidecar was ever built at TP=2.

The manifest records the image tag, so **every kernel-image change carries an 11-minute dump boot per
node.** Put it in the plan, not in the surprise column.

**The trap that rides with it:** a dump boot's own KV pool reads low — 3,958,677 against ~4.48M —
because writing 56 GiB per node goes out through the page cache. On production 9 the gap was
4,840,220 (dump) against 5,165,289 (load), 6.3 %. **Never record a dump boot's pool as a result.**

### 3.4 Reusing the sidecar directory name in dump mode destroys the rollback **[SILENT]**

**Track:** both, measured at TP=3 only — dump mode against an existing sidecar; none was ever built at TP=2.

`FASTLOAD_DIR` is mounted **read-write** in dump mode. Give a new arm's sidecar its own directory
name: one line in the env file, and it is the difference between an experiment and an outage.

Related, also unwritten: a sidecar may be deleted only **after** its env file is retired, because
`FASTLOAD_MODE=load` with the directory gone refuses the boot loudly and correctly — so a disk cleanup
in the wrong order becomes an outage.

---

## 4. Fabric, before NCCL

### 4.1 A single-node reboot kills the far end of its links

**Track:** both — the hotplug handler is per node, and the two-node reboot rule is identical.

Covered in full in [00](00-hardware-and-os.md) §3: the `dgx-spark-mlnx-hotplug` handler pulls the
ConnectX-7 off the PCI bus when a peer goes down. Symptom is `NCCL error: unhandled system error` and
`PORT_ACTIVE` reading 2 instead of 4 **on a node you did not reboot**. Remove
`/etc/nvidia/cx7-hotplug-enabled` on all three, and reboot all three together anyway.

### 4.2 `ip link` says UP and ping works, and RDMA is still dead

**Track:** both — IP-layer health never proves RDMA health, at any node count.

`ip -br link` can read `UP`, and ping can pass, on a link where `ibv_devinfo` says the port is not
`PORT_ACTIVE`. **IP-layer health does not prove RDMA health.** Always check `ibv_devinfo`.

### 4.3 Half the fabric had never carried a packet **[SILENT]**

**Track:** both, measured at TP=3 only — the counter table is three nodes; the defect is per peer pair.

**Symptom.** None. Every link `ACTIVE`, every subnet configured, every benchmark ran.

```
head      roceP2p1s0f0:0  roceP2p1s0f1:0  rocep1s0f0:2968259552956  rocep1s0f1:19338488069
worker-1  roceP2p1s0f0:0  roceP2p1s0f1:0  rocep1s0f0:2615320151240  rocep1s0f1:370647326623
worker-2  roceP2p1s0f0:0  roceP2p1s0f1:0  rocep1s0f0:2613834402438  rocep1s0f1:18498564601
```

The two `roceP2p1s0*` ports on every node — the second link of each pair — had transmitted **zero**.
Not "little". Zero.

**Cause.** The plugin's `mesh_connect()` opens with `(void)dev;` — it throws away the device index
NCCL hands it — then stops at the **first** address a local NIC can reach. Deterministic, so every
channel to a peer lands on the same link, every time.

**Fix.** `patches/kernel/0005`. 64 MB all-reduce 12.08 → **16.66 GB/s**. With
`NCCL_MESH_LINKS_PER_PEER=1` the selection is bit-identical to pre-patch.

**The most quotable line in this repository: read the byte counters, not the link state.**

### 4.4 The second link had no ARP neighbour — configuration is not a delivered packet

**Track:** both, measured at TP=3 only — the triangle ping and the four-neighbour rule are the three-node arrangement.

Before enabling the second link, there was **no ARP neighbour at all** on the second-link subnets on
any node, and `port_xmit_data = 0` on those ports. The netdevs had moved a few thousand packets each —
multicast and neighbour discovery only. **Unicast IP had never crossed those links.**

**Fix.** Six pings on a triangle, each bound to the interface under test, each node ending with four
resolved fabric neighbours. If one does not answer, **stop**: patch `0005` would route half the
channels at a peer that is not there, the handshake would time out and the engine would refuse to
boot. Set `NCCL_MESH_LINKS_PER_PEER=1`, keep only `0006`, and take it to whoever cabled the rack.
`bench/mesh-multilink-sweep.sh` refuses to run in that state.

---

## 5. Memory and the KV pool

### 5.1 The pool was halved by a page-size counter, not by memory **[SILENT]**

**Track:** both — measured at both rank counts, and worse at two: 60.2 % of the divisor.

**Symptom.** `GPU KV cache size: 2,428,769 tokens` where the memory said far more should fit.

```
HAREM-TP3 KV pool breakdown: num_blocks=1,756, blocks/request=723, max_model_len=1,000,000
  MLAAttentionSpec:  22 layer(s), page   109,824 B, max/req 33,057,024 B -> 301 block(s)
  KpoolTailSpec:     11 layer(s), page   109,824 B, max/req    109,824 B ->   1 block(s)
  MambaSpec:          9 layer(s), page 1,703,936 B, max/req 15,335,424 B ->   9 block(s)
  ...
  SlidingWindowSpec:  5 layer(s), page    24,576 B, max/req  9,461,760 B -> 385 block(s)   <-- the drafter
GPU KV cache size: 2,428,769 tokens, Maximum concurrency for 1,000,000 tokens per request: 2.43x
```

**Cause.** `num_blocks_per_request` is **summed over groups**. The drafter's page is 16 tokens, so
`cdiv(2047 + 4096, 16) + 1 = 385` — **53 % of the divisor for 0.6 % of the memory.** vLLM's own
comment says the smallest block is fine because a later unification step scales it up; true upstream,
false here, because the DFlash2 port keeps the draft's KV layers in their own independent group
(§2.6) and that path never reaches unification. **The 16 survives because of a change we had to make
for an unrelated reason.**

**Fix.** `patch-swblock-tp3.py` + `HAREM_SW_BLOCK_SIZE=256`. Pool 2,428,769 → **4,413,223 (+82 %)**.
Cost: per-block memory +9.2 %, and the draft group's prefix-cache matching unit coarsens 16 → 256.
Unpredicted bonus: the draft block table shrinks 16× → C4 +9 %, C8 +6 %, TTFT −20 to −30 %.
**Matching the target's page (3,328) is worse, not better: −7 %.**

Nothing in the log says the divisor is the binding term. The breakdown patch exists to make it
visible.

### 5.2 `--block-size 256` is silently raised to 3,328 (or 4,608)

**Track:** both — the entry names both blocks: 3,328 tokens at TP=3, 4,608 at TP=2.

```
Setting attention block size to 3328 tokens to ensure that attention page size is >= mamba page size.
Padding mamba page size by 5.79% to ensure that mamba page size and attention page size are exactly equal.
```

**Cause.** The mamba state has to fit in one block. **This cannot be changed and must not be** —
raising 3,328 grows the MLA page too and the block count falls. `--block-size 256` still matters: it
is the **prefix-cache hash block** (§2.3).

**Consequence, stated honestly:** with a 3,328-token block, prompts of a few hundred tokens never fill
one, so **prefix-cache hit rate is 0 % in every benchmark here.** Our benchmarks measure nothing
about prefix caching.

### 5.3 KV sizing refuses at `max_model_len 1000000`

**Track:** **TP=2 only** — at three ranks `max_model_len` 1,000,000 was never threatened.

```
ValueError: 6.6 GiB KV needed for max seq len 1,000,000, available 0.73 GiB
```

**Cause.** The TP=2 full-scope arm: two nodes carrying a 154 GiB model. About **4.7 GiB per node of
persistent, non-KV allocation scales with `max_model_len`** — lowering it to 65,536 raised available
KV memory from 0.73 to **5.41 GiB**. That buffer is paid in production too.

**Fix.** For an experiment only, lower `MAX_MODEL_LEN` per node, with a backup, and never carry it to
production. At TP=3 `max_model_len 1,000,000` was never threatened.

### 5.4 Long prompts are never scheduled — no error, no timeout **[SILENT]**

**Track:** **TP=2 only** — a pool small enough to starve the scheduler only ever occurred at two ranks.

**Symptom.** A request is accepted and then sits there. The engine reports `Running: 0, Waiting: 1`
with KV usage 0 %.

**Cause.** At TP=2 with the full-scope checkpoint the pool is **31,343 tokens ≈ 6.8 pages** at a
4,608-token block. With six or seven pages a long prompt cannot be scheduled at all.

**The measured admission ladder:** 844 tokens → served in 1.1 s; 1,684 → 1.7 s; ~2,800 → **never
scheduled**; 7,382 → never scheduled, client gave up at 600 s; 8,204 → three 900-second timeouts to
prove it.

**Fix.** TP=2 is a measurement rig, not a serving configuration. Every prefill figure and the C6/C8
aggregates in that arm are void.

**A lucky escape worth writing down:** MMLU 0-shot prompts sit below the ~2,000-token admission
ceiling. That was not calculated in advance, and on a benchmark with longer prompts the same arm would
have produced nothing at all.

### 5.5 `gpu-memory-utilization 0.85` starves the head node and swaps **[SILENT]**

**Track:** both, measured at TP=3 only — the memory ladder and its swap reading were derived at three ranks.

+19 % pool, no speed change, and the head node at **1.9 GiB free with 1.6 GB of swap** in use. Read
from `free` and `/proc/meminfo`, never from the engine.

**Cause.** Unified memory: every rung of the ladder is taken directly from the host. The swap was the
engine's own process pages being paged out.

**Fix.** Climb one rung at a time and check free memory **and** swap at each rung; never below 4 GiB
free. **Caveat, and it matters:** the 0.85 rejection predates the fast-load work, which removed a
large page-cache spike and added `MADV_DONTNEED` + `malloc_trim`. The same configuration today sits
at 11–12 GiB free with zero swap, and the 0.83 rung has since been measured and taken (see
[00](00-hardware-and-os.md) §11).

### 5.6 Do not take vLLM's own "to fully utilize gpu memory" hint

**Track:** both — vLLM's own hint arithmetic on a unified-memory part.

```
[gpu_worker.py:790] Free memory on device (104.1/121.63 GiB) on startup.
        Desired GPU memory utilization is (0.8, 97.3 GiB). Actual usage is 56.36 GiB for consumed
        memory (weights + non-torch), 1.69 GiB for peak activation, and 0.0 GiB for CUDAGraph
        memory. ... `--kv-cache-memory=49293662208` (45.91 GiB) to fully utilize gpu memory.
        Current kv cache memory in use is 39.26 GiB.
```

**Cause.** That offered figure is `MemAvailable(init) − non_kv_cache_memory − 150 MiB` — *all* the
memory the host had free at boot. On a unified-memory part it leaves the machine zero page cache,
which is §5.5's swap table by another route.

**Fix.** Ignore it. `gpu-memory-utilization` budgets a share of the **total**, deliberately.
`--kv-cache-memory` as a pin is a sharper instrument that skips the profile entirely — but **ladder
first, pin last**, because a byte pin removes exactly the headroom the 4 GiB rule protects. Never used
here.

### 5.7 `eager` loading billed 4.1 % of the KV pool to page cache

**Track:** both, measured at TP=3 only — the page-cache mechanism is general; the pool figures are three-node.

A boot-time change that should not touch memory cost **4,413,223 → 4,231,404** tokens.
`Available KV cache memory` 35.4 → 33.39 GiB, `consumed memory` 60.07 → **62.22 GiB**, with
`Model loading took` unchanged at 54.86 GiB.

**Cause.** The GPU pool *is* host memory, vLLM sizes the pool from what is free after loading, and
**page cache is not free memory.**

**Fix.** Two calls in `harem_fastload.py`, both on by default and both numerically inert:
`posix_fadvise(POSIX_FADV_DONTNEED)` over the shards once loading is done, and `malloc_trim(0)` to
return the loader's arenas (RSS 7.95 → 5.20 GiB on the dump boot). Result **4,484,848** — 1.6 %
*above* the pre-change reference.

### 5.8 The pool number is a delta between two `/proc/meminfo` readings, and it runs backwards **[SILENT]**

**Track:** both, measured at TP=3 only — the per-rank spread and the settle gate are all three-node readings.

**The highest-value silent failure on this stack: an engine log line about the engine's own memory
that is not a measurement of memory.**

**Symptom.** Rank 0 reports 1.50 GiB of "non-torch" memory and the workers 9.48–9.72 GiB, with
identical weights — `Model loading took 54.86 GiB` on all three. Read as an allocation, that says
8.2 GiB per worker is stranded. **No error is ever printed.**

**Cause.** On an integrated GPU, vLLM's "free GPU memory" **is** `MemAvailable`, and "consumed memory
(weights + non-torch)" is `MemAvailable(after NCCL init) − MemAvailable(after the profile run)`.
**The lower `MemAvailable` is when the engine starts, the larger the pool it computes.** The launcher
kills a ~90 GiB container and starts the next immediately, and the nodes start in a fixed order, so
the last one started is systematically the polluted one. One boot began at 104.10 / 113.07 / 113.10
GiB (**9.00 GiB** spread) and ended the profile at 47.74 / 48.73 / 48.52 (**0.99 GiB**). **A spread
that vanishes by the end was never an allocation.**

**Fix.** A host-side settle gate in `scripts/start-tp3.sh` — `sync`, then wait for `MemAvailable ≥
SETTLE_MIN_GIB` (112), polling every 3 s for up to 180 s:

```
mem settle: MemAvailable=113 GiB (target 112) after 6s
```

It **buys zero tokens.** It takes the per-rank spread from 9 GiB to 1.4 GiB. It has to be on the
host: `/proc/sys` is in the container's `ReadonlyPaths` and the container is unprivileged.

**Acceptance rule.** All three ranks' `Free memory on device` **and** `consumed memory` lines must
agree within **1 GiB**, and it must be a **load** boot. Fail that and the boot produced a speed result
and a quality result, and **no pool result**:

```
docker logs <container> 2>&1 | grep 'gpu_worker.py' | grep 'Free memory on device'
```

### 5.9 `--max-num-batched-tokens 4096` costs 28.5 % of the KV pool **[SILENT]**

**Track:** both, measured at TP=3 only — the batched-token budget never varied at two ranks; the divisor mechanism is general.

Visible only as `GPU KV cache size:` moving 2,427,385 → 1,736,465, because
`max_in_flight_tokens = max_concurrent_batches × max_num_batched_tokens` doubles and the draft group's
blocks-per-request term doubles with it (§5.1).

4096 buys +9.5 % fresh prefill, +24 % mixed-load decode and −13 % mixed-load TTFT. Production stays at
2048. **3072 was never tried** — the obvious probe, and it is item 3 in
[CONTRIBUTING](../CONTRIBUTING.md).

At 2048 the scheduler emits a harmless warning about `max_num_scheduled_tokens` versus the k=7 draft
slots. Cosmetic.

### 5.10 The profiler's kineto buffers take 7–8 GiB per node and do not come back **[SILENT]**

**Track:** both, measured at TP=3 only — profiling was only ever run on the three-node production arms.

Free RAM drops sharply after a profiling window and stays down until the container restarts. On
production 9's profile it drove host RAM to 2/4/4 GiB and the engine had to be restarted. **On a
stack whose rule is never to go below 4 GiB free, this is a real constraint.** Plan the run when the
host has room. Whole-run cost: ~6 min of GPU time, ~120 MB of trace per node.

---

## 6. NCCL and the mesh plugin

### 6.1 The all-reduce cliff — 0.6–1.9 GB/s in the middle of the size range

**Track:** both, measured at TP=3 only — the channel cap applies at any peer count; the cliff was measured on three nodes.

**Symptom.** The all-reduce column collapses between ~200 KB and ~12 MB while point-to-point over the
**same queue pairs** stays healthy — 11 GB/s at 3.6 MB where the all-reduce manages 0.83.

The one-line bug, in `mesh_connect_qp()` at the RTR transition:

```c
qp_attr.min_rnr_timer = 12;  // ~0.01ms min RNR NAK timer
```

**The comment is right about the intent and wrong about the value.** In the IBTA encoding **code 1 is
0.01 ms; code 12 is 0.64 ms.**

**Cause.** The plugin's data path is **two-sided**, so **RNR *is* the flow control**. Two multipliers
turn a rare miss into a systematic one: NCCL opens **64 channels** on this fabric, all serviced
round-robin by a **single proxy progress thread**, so the window in which a send can outrun its
receive grows with the channel count; and NCCL drives the chunk size to its **32 KiB floor**, so a
512 KB all-reduce becomes hundreds of sub-3 KB sends, each able to miss, each miss quantised up to
0.64 ms.

**Diagnostic.** The ConnectX-7 counters. `rnr_nak_retry_err` (requester) and `out_of_buffer`
(responder) track the cliff exactly; `packet_seq_err` and `local_ack_timeout_err` are **zero at every
size in every configuration** — nothing is lost, nothing times out.

**Fix that ships.** `NCCL_MAX_NCHANNELS=8`, environment only. C8 decode all-reduce **1,195 → 123 µs
(9.7×)**; one C8 step's collectives **91.7 → 9.9 ms**; engine **C8 +12.6 to +13.0 %**.

**Fix that is carried but was never worth a boot alone.** `patches/kernel/0004`, 26 added lines plus
`NCCL_MESH_MIN_RNR_TIMER`. On its own at 64 channels it is worth about **2.2×**, not the ~10× the cap
gives: a shorter timer makes each miss ~64× cheaper, while capping the channels stops the misses
happening.

**Honest shape.** The microbenchmark sees 9.3× on a decode step's collectives; the engine sees
+12.6 % at C8. **Nothing justifies quoting 9× as an end-to-end number.**

### 6.2 Sixteen channels is 2.5× worse on the message the engine decodes with **[SILENT]**

**Track:** both, measured at TP=3 only — the 16-channel arm is a three-node decode-step measurement.

A reasonable-looking change — 8 channels over 2 links → 16 to restore 8 per link — makes decode 2.5×
slower. One C8 decode step (90 × 512 KB): 8 channels **9.27 ms**, 16 channels **26.18 ms**.

**Cause.** The RNR mechanism does not care how many links the channels are spread over; it cares how
long the single proxy thread's round-robin lap is, and 16 makes it long again. Sixteen *does* win the
large-message columns (64 MB all-reduce 20.56 vs 16.66 GB/s) — exactly the trade the cap was chosen
against, because this engine's decode all-reduce is 512 KB, not 64 MB.

**Fix. Keep 8.** `NCCL_MAX_NCHANNELS=12` over two links has **never been measured**: the equivalence
with 8 was established on **one** link and the arithmetic has changed.

### 6.3 DMA-BUF registration works and is slower

**Track:** both, measured at TP=3 only — the DMA-BUF sweep was run on the three-node mesh only.

`NCCL_MESH_DMABUF=1` succeeds — answering the open question of whether `ibv_reg_dmabuf_mr` accepts
these buffers on this platform — and is slower across the range (64 MB all-reduce **18.08** against
20.84 GB/s). On a part where one physical memory is shared and `ibv_reg_mr` takes a device pointer
directly, DMA-BUF buys nothing and costs registration work. **Leave it off.**

A *predicted* failure string was written into our plan and never fired
(`regMr: failed to register ... Set NCCL_MESH_PTR_CUDA=0`). **That string is a prediction, not an
observed log line.**

### 6.4 An old env file carrying 8 channels *and* `NCCL_PROTO=LL` is not evidence **[SILENT]**

**Track:** both, measured at TP=3 only — the confounded env file and its protocol numbers are three-node sweeps.

`NCCL_PROTO=LL` costs **11×** at 16 MB (20,113.6 µs against auto's 1,787.3) and buries the channel
gain. That combination made `NCCL_MAX_NCHANNELS=8` look "already tried and rejected"; tried cleanly it
is +13 % at C8. **Leave `NCCL_PROTO` unset** — `Simple` is 2.8× worse at the C1 decode message and
4.4× at C8; `LL128` is 3.1× better at 4 MB and 2.1× worse at 16 MB.

**Re-run an arm one variable at a time.**

### 6.5 `NCCL_ALGO=Tree` is dead on this fabric **[SILENT]**

**Track:** both, measured at TP=3 only — the entry's own reasoning is "three nodes"; the five-round arm was TP=3.

4–6× slower than Ring at the two message sizes the engine uses in bulk; RNR retries an order of
magnitude higher; 23–96 % worse on the decode-step proxy. A tree wants a bandwidth-shaped topology to
pay for its lower step count, and this is three nodes each holding a pair of PCIe Gen5 x4 cards.

**`Ring` stays.** `Ring,Tree` looked 3.6 % better on the step proxy and worse on `sendrecv` at 64 MB,
inside a sweep whose own repeat spread reaches **1.7× at 1 MB** — model-free could not answer it. The
five-round engine arm did: **every concurrency level inside ±1 % of `Ring`**, identical TTFT,
acceptance inside its own spread, full gates. It leaves 1.5 % more KV pool through NCCL's buffer sizing and that is the only
real difference ([06](06-nccl-mesh.md) §12.3). Do not chase it.

### 6.6 The one-sided `RDMA_WRITE` transport works perfectly and buys nothing

**Track:** both, measured at TP=3 only — the RDMA_WRITE transport was measured on the three-node fabric only.

RNR and out-of-buffer go to **exactly zero** at every size, in both repetitions, and throughput does
not move at any FIFO depth. At ~20 GB/s the transfer is against a **PCIe** wall, not a flow-control
stall. `patches/kernel/0007` is kept as an option and **deliberately not offered upstream** —
reporting a transport rewrite as a win when our own measurement says it changes nothing would waste
the maintainer's time.

**Lesson: a mechanism working is not a result.**

### 6.7 A fabric sweep beside a running engine kills its next collective

**Track:** both, measured at TP=3 only — the sweep that killed the engine's next collective was the three-node one.

Not "slows" — **kills**. A three-node NCCL sweep competes for RDMA queue-pair resources and can
exhaust them. **Three-node NCCL work needs the engine down, not idle.** One measurement holds the
cluster at a time, and it says so in a file. A model-free container beside an *idle* engine is allowed
with `--rm`, a memory cap, a cpuset disjoint from the engine's, and its own JIT cache directories.

### 6.8 The plugin's own test target does not link on a clean tree

**Track:** both — the plugin Makefile omits `IBVERBS_LIBS`; a pre-existing build issue.

```
undefined reference to ibv_event_type_str
```

The plugin `Makefile` sets `TEST_LDFLAGS = -lpthread`, omitting `$(IBVERBS_LIBS)`. **Pre-existing, not
ours.** Linked by hand, stock and patched produce identical output: 60 passed, 6 failed, the 6 being
pre-existing string checks. `make test-unit` gives `test_routing` **13/13**.

---

## 7. Kernels and build

### 7.1 `n_rows` is not passed on the unsplit MoE launch — TP=3 was 8–29 % slower than TP=2 **[SILENT]**

**Track:** **TP=3 only** — the `expert_map` path needs expert parallelism, mandatory only at three ranks.

**Symptom.** Three nodes slower than two. C1 −7.8 %, C2 −16.5 %, **C4 −29.3 %**, C6 −25.2 %, C8
−19.7 %, prefill −5.0 %. Worst in the middle. Not memory, not the drafter, not correctness —
acceptance and tokens-per-step identical to three significant figures.

```c
if (split > 1)
    launch<..., true,  ...>(..., expert_ids, b_expert_stride, svh_expert_stride, n_rows);
else
    launch<..., false, ...>(..., expert_ids, b_expert_stride, svh_expert_stride);
                                                                      // ^ n_rows missing
```

which makes the guard `if (n_rows && m0 >= *n_rows) return;` dead on the unsplit path.

**Cause.** `moe_align_block_size` marks the surplus tail `-1` correctly — **the conversion happens one
step later**: vLLM maps the array through `expert_map`, and `-1` indexes the *last element* of that
map. On the rank owning the top of the global expert range that element is a live local expert, so
**every surplus block runs a full GEMM** — 38 % of the grid at M=2048 — and under expert parallelism
all three ranks wait for that one.

**Fix.** Pass `n_rows` in the `else` branch. Rank 2's MoE stage at M=2048: **18,401 → 10,107 µs
(−45 %)**. End to end C1 40.8 → 49.4, C4 59.5 → 99.6, C8 91.9 → 139.1, prefill 1,025 → 1,257.
**Upstream in `a95e809`.**

No error, no wrong output; **a rank quietly doing 1.2–2.0× the work.**

### 7.2 The micro-benchmark's tensor-parallel arm fails on a shape mismatch

**Track:** **TP=3 only** — 2 x 682 comes from 2,048/3; at two ranks 2 x 1,024 divides 16.

```
exl3_moe_gemm: svh n mismatch
```

The bench needs `2 × inter / tp` to be a multiple of 16 for the TP arm, and 2·682 is not. Use
`--inter 2304` for that arm, and read the **expert-parallel** rows as the ones describing production.

### 7.3 A GPU-free compile check reported a pass; 6 of 18 configurations failed at launch

**Track:** both — the launch-time shared-memory limit of the part.

```
OutOfResources: out of resource: shared memory, Required: 106496, Hardware limit: 101376. Reducing block sizes or `num_stages` may help.
```

and at other configurations `Required: 114688`, `147456`, `172032` against the same limit.

**Cause.** `metadata.shared` from the ahead-of-time path under-reports what a launch actually needs —
36,864 reported against 106,496 required in one case. Resource allocation only becomes visible when
something is launched.

**A compile check answers "does it build", never "does it run."**

### 7.4 TileLang will not compile the third HC kernel above 96 threads

**Track:** both — a TileLang compile limit, with no rank term.

```
min_reg_num < INT64_MAX is false
```

`threads=96` and `hidden_block=1024` are embedded in the kernel body with no kwarg. The fusion was
written in **Triton** instead: −14.9 to −15.5 % on the fused pair, −1.0…1.1 % of the prefill wall —
about 40 % of its own ceiling. **Not adopted standalone.**

### 7.5 TileLang's config surface is a cliff, and the module shipped with a bad default **[SILENT]**

**Track:** both — a kernel autotuning cliff and a bad shipped default.

A one-step configuration change moves the kernel from 187.8 GB/s to 79.4 or 44.5. **The default the
module shipped with was one of the bad ones (79.4).** Sweep, and re-sweep on every hardware or shape
change.

**A documentation trap rides with it:** `patch-mhcfused-tp3.py`'s docstring says the threshold default
is 256; the shipped module sets `MIN_M = _env_int("HAREM_MHC_FUSED_MIN_M", 1024)`. 256 was the value
measured and found to **lose** (+37.7 % at M=512). **Read the code, not the comment.**

### 7.6 TileLang's `flashinfer.comm` preload is swallowed **[SILENT]**

**Track:** both — a swallowed preload inside the TileLang module; image-level.

**Symptom.** An **illegal address at kernel launch much later, on a different rank, at a time that
looks random.**

The preload sits inside `contextlib.suppress(Exception)`; upstream's own comment says *"import order
is load-bearing"*. It preloads `flashinfer.comm` so its `CudaRTLibrary` binds the real `libcudart`
before `tilelang` maps a stub and shadows it. A transient failure is swallowed silently.

**Fix.** `patch-tilelang-failloud-tp3.py` (`HAREM_TILELANG_FAILLOUD=1`) replaces the suppression with
try/print/raise so a failed preload stops the rank in its first second with a named cause;
`flashinfer-warmup.py` imports `flashinfer.comm` on the CPU before any worker starts and warms the JIT
cache so N ranks do not race it. Default unset → upstream behaviour byte for byte.

### 7.7 CUDA graphs are off, and the log names the reason

**Track:** TP=3 — the rank term is the whole cause (below). TP=2 captures graphs on the same image.

```
CUDAGraphMode.FULL_AND_PIECEWISE is not supported with spec-decode for attention backend
FlashInferBackend (support: AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE);
setting cudagraph_mode=NONE
Skipping CUDA graph capture.
```

**Cause.** FULL capture with spec-decode needs a backend declaring at least `UNIFORM_BATCH`;
FlashInfer declares `UNIFORM_SINGLE_TOKEN_DECODE`, and the drafter is on FlashInfer **because its
cache is fp8** — which is production 7's own pool win. **It is not `--enforce-eager` that turns graphs
off.**

**Why it declares that, and why only at three ranks (6 September).** FlashInfer's
`get_cudagraph_support()` divides the *target* model's per-rank query heads by the *draft* group's KV
heads — 22 % 3 = 1 at TP=3, 32 % 4 = 0 at TP=2 — while the builder that actually runs uses the draft's
own 12 heads (12 % 3 = 0) and selects the XQA decode kernel: `decode_backend=xqa` is in the same boot
log, three lines above the warning. The declaration is wrong, the kernel is not; filed upstream as
[vllm#55581](https://github.com/vllm-project/vllm/issues/55581). At two ranks the same image prints `Graph capturing finished in 11 secs, took 1.10 GiB`.

**Fix: none taken.** Three routes exist and all cost something. Returning to bf16 draft KV has
already been tried and read the same tok/s, and costs 5.6 % of pool. The whole lever is worth
1.4–1.9 ms, **+1.5–2.1 %**. PIECEWISE also silently disables spec-decode (vLLM #53030). The upstream
fix is a one-parameter change to the gate; on our side it would be a new patch file, which re-dumps
the fast-load sidecar ([08](08-fast-boot.md)). Priced, with the A/B that would settle it, in
[11](11-open-issues.md) §2.29.

### 7.8 Full scope is a **prefill regression** on the dense path

**Track:** both, measured at TP=3 only — the M=1,792 dense profile is production 9's; TP=2 prefill was unmeasurable.

```
dense stage @ M=1792 : 167.39 ms (bf16, 457 calls)
                     : 184.73 ms (128.38 EXL3 GEMM + 19.89 had_in + 36.46 bf16, 695 calls)
```

**+17.3 ms, +10.4 %.** At M=1,792 the trellis GEMM plus its input Hadamard is more expensive than
cutlass/nvjet bf16 for the same shapes. The chunk wall stayed flat only because MoE Hadamard (−6.6 ms),
NCCL (−5.0) and MoE GEMM (−4.6) happened to give it back.

**Fix (proposed, not built).** An **M threshold on the dense path** — EXL3 below some batch size, bf16
above it. The plugin author confirmed the same shape on his own card (EXL3 dense 1.25–1.55× slower
than bf16 cuBLAS from M=64–256 upward) and **withdrew the "cuBLAS parity" claim from his README**.

**Caveat.** Not a clean single-variable A/B — the checkpoint changed too.

### 7.9 The NVFP4-era 22-head decode kernel bug — no EXL3 analogue, and that is deliberate

**Track:** **TP=3 only** — 22 heads is the TP=3 pad, and the guards are the pad checks.

On the NVFP4 stack a decode kernel silently computed the wrong thing at 22 heads (TP=3), because the
lab had tested TP=2 and TP=4 only. **An untested shape.** The exact error string was never recorded,
because there was no error — only fluent, confident nonsense.

**On this stack the failure mode is prevented rather than diagnosed:** the drafter refuses a quantized
checkpoint outright and requires head counts to divide the TP size; the pad check reads the
checkpoint's own safetensors headers and refuses five named traps; and the pad-zero proof reads the
fabricated rows after `load_weights`. **Both guards exist because of a failure mode already paid for
elsewhere.**

### 7.10 Binary hashes cannot certify this build

**Track:** both — cubin nondeterminism in the NVCC and ptxas toolchain.

The compiled extension is the same size everywhere and has a **different sha256 and a different ELF
Build-ID on each node**. The divergence is entirely in `.nv_fatbin`; `.text`, `.rodata`, `.data` and
`.comment` hash identically. Every plausible cause was proven identical across nodes.

**The decisive test:** a second independent `docker build --no-cache` **on the same machine** produced
another node's cubins and Build-ID. It is build-to-build nondeterminism in the NVCC/ptxas toolchain.

**Only behaviour certifies a build** — the upstream pytest suite (expect `44 passed, 41 skipped`,
exit 0), then the serving gates. **Check the count against your previous build's** rather than
assuming it: it moved once already in this series, and a suite that quietly gains or loses a test
between commits is itself the finding.

### 7.11 One node's build wall clock disagrees with its own build timer by 90 seconds

**Track:** both — an unexplained build-timer anomaly on one node.

`head`'s outer bracket for the serve layer was ~3 s while that build's own step timer reported
`#4 DONE 93.5s`. Clock skew, Docker version and ccache were all checked and ruled out. **Unexplained,
and recorded rather than rounded away** — in this series the anomalies that looked cosmetic were twice
the ones worth chasing.

---

## 8. Measurement and tooling

### 8.1 The MLA tuner's warm-up is longer than two sweep rounds, and it turned the winner into a disaster

**Track:** both, measured at TP=3 only — the tune-storm arm and its prefill and C8 figures are three-node.

**Symptom.** An arm measured prefill **1,373 → 698 tok/s**, C8 **140 → 105**. Re-measured on the
*same running engine* minutes later with nothing changed but the clock: prefill 1,483–1,515, C8
152.6 / 150.3 / 148.6. **That arm was the winner by 12.6 %.**

**Cause.** The MLA decode tuner keyed on batch shape and kept the winner in a **process-local
`std::map`**, so every boot re-tuned and every unseen batch shape bought a tune *while serving*. Our
sweeps present B = 25, 26, 27, 29, 470, 768 across C1–C8.

**Fix.** Upstream `9bf594c` persists it behind `CUDA_EXL3_TUNE_CACHE` — **18 tune events before
serving → 0.** Protocol drops from five rounds (discard two) to three (median of three).

**Check it took.** The cache file should have ~18 lines after a boot plus a sweep and **stop growing**
on the second boot. If it keeps growing, the directory is not surviving the container or the tag
changed — either way the engine is correct and only slower, which is the right failure mode.

**Open sub-item:** the cache key covers the device name and the upstream format tag but **not**
`patches/kernel/*`. A kernel patch that changed the candidate grid would need the file cleared by
hand, and nothing enforces that.

### 8.2 Boot-to-boot variance is 15.9 % on C8 with nothing changed

**Track:** both, measured at TP=3 only — boot-to-boot spread has only ever been measured at three ranks.

Same image, same environment file, two boots: C8 **118.44 / 137.23 / 135.56**.

**A difference under about 5 % is not a result.** Within one settled arm the declared bands are
**C1 ±4 %, C2 ±6 %, C4 ±9 %, C6 ±6 %, C8 ±3 %** — C4 is the noisiest column and C8 the quietest, the
opposite of intuition. **A difference of 3 % or less is written down as "equal."**

Cost of not knowing this: a published "one upstream build is ~10 % slower end to end" that does not
reproduce.

### 8.3 Prefill measured on a repeated prompt reads 56 % high **[SILENT]**

**Track:** both, measured at TP=3 only — two 3,328-token blocks; both figures come from a three-node arm.

Run the fixed-seed script twice inside one boot and the second run reads up to **1,596 tok/s where the
honest number is 1,025**. The prefix cache serves two whole 3,328-token blocks and nothing says so.
Use `bench/prefill-fresh.py`, which draws a new seed per request.

### 8.4 Our own harness disagreed with its own documentation for a day

**Track:** both — the harness header disagreed with its own body.

The quick-arm script carried the header "5 rounds, the first 2 discarded" and its body ran **three**.
Applied literally, the rule would have left a median of one. Nothing was published from the wrong
reading, **but a tool that describes itself incorrectly is the same failure class as a ruler that
reads high: it will eventually be believed instead of read.**

### 8.5 `CUDA_EXL3_DEBUG_NAMES` printed nothing, and silence looks exactly like a clean run **[SILENT]**

**Track:** both — the image configures only vLLM's logger; third-party INFO is discarded.

**Symptom.** The designed boot-time acceptance gate produced **no output at all**. The flag reached
the container and the code was right.

**Cause.** The plugin logs that line at `logger.info`, and the image configures only vLLM's own
logger, so third-party INFO goes nowhere.

**Fix.** Upstream now logs it at `warning` and reports the modules that resolved as well as those that
stayed BF16, with running tallies. **An acceptance gate has to be verified before the arm, the same as
any other instrument. A gate whose passing state is silence must be checked for *presence*.**

### 8.6 The expert-parallel evidence line was discarded by the logging configuration **[SILENT]**

**Track:** both — the same logger misconfiguration; the tensor-slice warning applies anywhere.

The line that should appear on every rank:

```
EXL3 routed experts: mode=EP ep_size=3 experts_local=96/288 hidden=4096 intermediate_local=2048 trellis=whole
```

(`intermediate_local=2048` and `trellis=whole` are the two words saying the trellis was never cut.)

**Cause.** The module created its logger under its own package name; the engine attaches a handler
only to the `vllm` logger. The fail-closed **raises** were unaffected — exceptions do not go through
logging — but the evidence line and a "routed experts are being TENSOR-sliced" **warning**, a safety
net, were invisible. **That is exactly backwards.**

### 8.7 The profiler answered 404 for a week, because of a second launcher copy **[SILENT]**

**Track:** both — launcher copies and `--profiler-config`; tooling, not rank.

**Symptom.** `POST /start_profile` returns **404** on a running engine.

**Cause, two layers.** This vLLM takes the profiler **only** as `--profiler-config`, not as
`VLLM_TORCH_PROFILER_DIR` — with the environment variable alone the route is never attached. **And
there were two copies of the launcher**: a working copy that had grown a `--profiler-config` arm, and
a second copy one day older that had not, and had never had the settle gate, the fast-load mounts or
the conditional prelude arms either. Nothing pointed at the older one, nothing depended on it, and it
stayed readable and plausible.

**Cost.** [10](10-results-and-roofline.md) §5 spent **a week as a reconciliation with a 2.8 %
residual**. When the flag finally went in, the trace deleted two of that section's ranked targets in
an afternoon.

**Fix.** Carry the profiler arm in production, off by default. And: **one launcher, one copy. If a
second copy exists, it is not a backup, it is a coin flip.**

### 8.8 CUPTI manufactures 2 ms of GPU idle out of nothing **[SILENT]**

**Track:** both, measured at TP=3 only — the CUPTI overhead figures come from the production 7 and 9 traces.

A trace showed **5.45 ms of C1 GPU idle (5.8 %)** and a CUDA-graph lever worth +6 % single-stream.
Both wrong in the same direction.

**Cause.** The profiler's overhead lands on **kernel boundaries** at roughly 1 µs each. A decode step
launches ~2,300 kernels across ~1,873 gaps, so the tracer alone manufactures **~2.0 ms** of apparent
idle. Real budget **3.477 ms = 3.75 %**. And **77 % of what remains is per-kernel dispatch**, not host
latency — the host runs 3.9 ms *ahead* of the GPU at C1, so "CPU gap" was a misnomer. Graph capture is
worth **1.4–1.9 ms, +1.5–2.1 %**.

**Fix. Take `busy(union)` from the trace and the wall from a profiler-off run, and let the difference
be the idle** — never read an idle figure straight out of a trace. Measure the profiler's own cost in
the same windows: here 0 % on prefill, +2.5 % on C1, +1.3 % on C8. On production 9, which launches
2,738 kernels per step, the same overhead is **+16 %** — so that arm's NCCL and idle rows are inflated
and must not be compared with production 7's.

**Two further trace-reading traps, both producing confident nonsense.** In decode the per-step
`gpu_user_annotation` arrives as an **overlapping pair** — merge them, or your step count doubles and
you report a 50 % GPU bubble. And when matching a gap to the launch that follows it, include
**`cuda_driver`** launches, not only `cuda_runtime` — Triton and Inductor kernels are launched through
the driver API, and filtering them mislabels the two largest gaps in a step as device-side waits.

**Useful accident:** the drafter runs **outside** the step annotation, so the annotation boundary is an
exact target-versus-draft split. That corrected the k=7 drafter's cost from 19.5 % of a C1 step to
11.4 %.

### 8.9 Both rulers were brochures, and every efficiency percentage was ~22 % optimistic

**Track:** both — single-device bandwidth and GEMM rulers.

Device read bandwidth **225.2 GB/s** against a vendor 273; BF16 dense GEMM peak **97.3 TFLOP/s**
against ~125 implied. Two tools, both in `bench/`, both a few seconds: `bench/bw.py` and
`bench/gemmpeak.py`. **Run them in the same binary and the same run as whatever you are measuring, and
quote the result beside every efficiency claim.**

**And the ruler itself drifts:** three independent measurements the same morning on the same idle
machine gave **225.2, 239.6 and 240.9 GB/s — 6.5 % apart.** Any efficiency figure that would change a
decision at ±6 % is given as a band. A `memset` ruler measures 196.8–198.2 GB/s and is the right
comparison for a kernel that only writes.

### 8.10 The model-free MoE bench overstates small-M by 1.5–1.7× **[SILENT]**

**Track:** both — a model-free bench against the real build's kernel set.

**Carry the *shape* of a model-free result to the engine, not the absolute number.** Two of this
repository's ranked targets were model-free measurements carried onto the production path without
checking that the path still had them: `exl3_moe_combine` (**the kernel does not exist in this
build** — it is fused into the down-projection epilogue, so the class is 0 %) and `_zero_kv_blocks`
(measured live at **0.857 ms, 0.09 %** against a published 14.7 ms, 1.3 % — a **16× overestimate**).

### 8.11 The real prefill chunk is 1,792 tokens, not 2,048

**Track:** both — 1,792 = 7 x 256 is a block granularity, and rank-independent.

Every "per chunk" conversion built on `--max-num-batched-tokens` is 12.5 % off. The real chunk is
**1,792 (7 × 256)** because of the 256-token block granularity, so 12.5 % of the budget is unused.
`MNBT 2304` would give a full 2,048-token chunk at a KV price that has not been measured.

### 8.12 Tier-B arms carry no category numbers, and the tables did it silently **[SILENT]**

**Track:** both — the quick-arm script omits the category step.

The quick-arm script does not include the category step at all, so results tables carried the previous
arm's category figures forward. **Say `[not tested]` rather than carrying figures forward.**

---

## 9. Silent-correctness — the failures with no error at all

### 9.1 `disable_tp` on the shared expert triples its contribution **[SILENT]**

**Track:** **TP=3 only** — it needs expert parallelism and the shared-expert pad, both TP=3 only.

The model stays fluent and the answers go wrong. The MoE runner all-reduces the combined MoE output
whenever `ep_size > 1`, and the model builds the shared MLP with `reduce_results=False` precisely
because the runner owns that reduction — so a replicated shared expert is summed once per rank, a
**3× shared contribution**. **Never `disable_tp` under expert parallelism.** Pad the shared expert
instead.

### 9.2 A padded drafter head holding allocator garbage **[SILENT]**

**Track:** **TP=3 only** — the 32/8 -> 36/9 drafter pad exists only at TP=3.

Acceptance falls, nothing crashes, and no log line says why. A padded head whose q/k/v rows are **zero**
produces a zero attention output and a zero `o_proj` contribution — an exact no-op. Garbage does not.
**A build gate cannot catch it: it depends on what the allocator happened to hand out at load time.**

The proof lines that must appear on the rank owning the padding:

```
HAREM-TP3 drafter pad: 32/8 (checkpoint) -> 36/9 (config) at tp=3; rank 2 owns 8 real + 4 padded query heads
HAREM-TP3 drafter pad verified zero on rank 2/3: 26,214,400 elements across 4 padded query head(s) and 1 padded KV head(s)
```

26,214,400 was **arrived at by arithmetic before the run**, which is what makes it evidence rather than
a log line agreeing with itself.

**The five traps divisibility alone misses:** a config that *shrinks* the real head count (24/6 — both
divide 3); a pad so wide the top rank owns no real head (48/12 — divides 3, starts, serves, answers
confidently wrong); a pad that changes the GQA ratio (36/12, 4:1 → 3:1); a checkpoint that cannot be
read (a dangling sidecar symlink — **"I could not check" and "it is fine" must never be the same
outcome**); and a config that is not `checkpoint + zero pad`.

`HAREM_TP3_DRAFT_PAD_CHECK=warn` exists for diagnosis only. **Do not serve with it.**

### 9.3 The target's EXL3 pad invariant was holding **by accident** **[SILENT]**

**Track:** **TP=3 only** — the A10 pad audit; the pad lives on the last of three ranks.

The audit line, **whose absence is a failure**:

```
HAREM-FULLSCOPE assert 5: 285 EXL3 pad site(s) audited, 285 padded on this rank, all whole 128-blocks and exactly zero
```

285 is the count a model-free meta-device run predicted for rank 2 **before** the boot.

**The audit caught its own bug on first writing, and that is the reusable part:** it called a
column-parallel module's *input* padded and rejected two modules on ranks 1 and 2. **Rank 0 passed.**
A single-rank test could not have seen it; **at TP=3 the last rank is mandatory, because the pad lives
there.**

### 9.4 The shard-index trap — a wrong index loads without error **[SILENT]**

**Track:** both — the tuple shard ids are the S3 KDA split, shared with the TP=2 tree.

**A tuple shard id must start at 0.** `weight_loader_v2` indexes the tuple *relatively*, so `(3,4,5)`
would load without error and write to the wrong slice. **vLLM's own "following weights were not
initialized" gate does not protect this**, because every parameter of every linear counts as loaded.

Four asserts carry it: three EXL3 shards of the expected width and total; the three identical `suh`
rows collapsed to **one** group; the three `conv1d` thirds **different from each other** (identical
thirds mean the split was on the wrong axis); and mapping order against shard ids. **All four were
silent on the live boot.**

Three things deliberately **not** in the mapping: `lm_head` (the trellis puts vocab on dim 1 where a
plain linear has it on dim 0 — putting it in the mapping *would load without error and be wrong*),
and two mixed-precision entries.

**The plugin's own README example was wrong** and resolved 0/34 here. Measured, reported, corrected.

### 9.5 A DFlash2 drafter on the V1 model runner degrades to DFlash1 **[SILENT]**

**Track:** both — a DFlash2 port property: the V1 runner against the V2 one.

The engine boots, produces correct-looking text, and quietly loses acceptance, because on V1 the same
checkpoint drafts through a proposer that never calls the candidate selector. Forced to the V2 runner
and guarded by a **build-time** gate that asserts every ported symbol resolves — **if any check fails,
no image is produced.** Expect one line: `DFLASH2 PORT BUILD GATE: OK`.

### 9.6 `draft_logits_spec()` with `torch.zeros` **[SILENT]**

**Track:** both — the drafter's logits hook; a port property.

DFlash2's kernel writes only the K candidate columns; the base's `torch.zeros` would leave
non-candidates at probability **0 rather than excluded**, so the walk and the sampler read different
distributions. DFlash2 overrides the hook to fp32 / `-inf`.

### 9.7 The KV-zeroing gate refuses to boot on this checkpoint, by design

**Track:** both — the hybrid layout's co-ownership of KV slots by MLA and Mamba/KDA layers.

With the zeroing disabled the engine `raise`s at startup rather than serving, naming the reason: the
region is **co-owned by attention and Mamba/KDA layers**. vLLM's zeroer visibly skips Mamba layers,
which made "the only live reason is the bf16 draft cache" look sound. It is wrong: in this hybrid
layout **one tensor is co-owned by an MLA layer and one Mamba layer per group**, and **85.5 %** of what
is being zeroed is that co-owned region.

**The fail-closed gate was written anyway — three conditions proved from the engine's own config,
`raise` rather than warn — precisely so the conclusion is checked by the machine rather than
believed.** The two changes had been designed as one and turned out unrelated: the zeroing is bound by
Mamba slot sharing, not precision, and draft KV at fp8 is a 5.6 % pool gain that never needed it.

### 9.8 Dual-batch overlap would corrupt KDA state, fluently and silently **[SILENT]**

**Track:** both — 34 of 45 layers are KDA; the batch splitter's alignment has no rank term.

34 of 45 layers are KDA/Mamba, and the batch splitter does not align to request boundaries, so a fresh
prompt split in two has its second half **start its recurrent state from zero and write it into the
same state slot.** vLLM's own code blocks it (`assert not should_ubatch`). **Do not build DBO.** It
also fails on arithmetic: splitting the batch pays the MoE expert weight stream twice, and decode is
**+38 %** worse.

**Also closed while looking:** model-level **sequence parallelism** is a one-line gate change and a bad
idea (identical bytes, collective count doubles, **+10…15 % worse at decode**); **pipeline
parallelism** removes ~98 % of collective bytes at ~3× single-stream decode latency. Both written down
because "just turn on SP" is the obvious next reflex and it is wrong on this workload.

### 9.9 The RoPE-convention hypothesis — on paper the bug, refuted by measurement

**Track:** both — the drafter-versus-target RoPE convention, refuted by acceptance.

A helper upstream carries a comment describing precisely the symptom we had gone looking for: the
mismatch is silent, *"acceptance collapses, the output stays correct"*. GLM-5.3's attention uses
`is_neox_style=False` and the base drafter takes the default `True`.

**No fix — the measurement refuted it.** Acceptance is 62.4 % at k=7, the same band as the known-good
stack running the same drafter at the same depth. **A working, measured configuration was not changed
on a hypothesis.** The patch is written and never applied.

Related, deliberately not ported: a fork commit decoupling the draft's Gumbel noise stream from the
target's. **It is the first place to look if sampling-mode acceptance ever looks strange** —
temperature 0 does not exercise that path.

### 9.10 Non-English prose collapses draft acceptance to 10–13 % **[SILENT]**

**Track:** both — a drafter-training property; low prose acceptance shows in the TP=2 arms too.

Single-stream prose at 29.1 tok/s against 61.7 for code, on production 9. The drafter predicts English
far better. **Every speed table in this repository was measured with English prompts; treat them as
English-workload numbers.** No fix available on this side — it is a drafter-training question.

Related: a synthetic counting prompt runs at roughly **1.7×** the realistic single-stream rate.
**Publishing a synthetic number without the label is the single easiest way to mislead someone about
this hardware.**

### 9.11 The chat template's provenance is unverified — **closed 10 September 2026**

**Track:** both — chat template provenance, at checkpoint level.

The served `chat_template.jinja` matches neither checkpoint on disk because it is neither
checkpoint's file: it is the **4 September 2026 revision of `zai-org/GLM-5.3-Flash`'s own
template**, 10,950 bytes, SHA-256 `0c4099f3…`, verified on the production node on 10 September.
The verification is by size and hash alone, and it is sufficient because of a finding we owe to
jdecker76 (issue #1): across the whole revision history of `zai-org/GLM-5.3-Flash`, the 62
safetensors, `config.json`, `generation_config.json`, `tokenizer.json`, `tokenizer_config.json` and
the weight index have **never changed**; the only content that has ever moved is
`chat_template.jinja`. Provenance therefore reduces to one table:

| Upstream revision | Size | SHA-256 (prefix) | `clear_thinking` default | Who ships it |
|---|---:|---|---|---|
| 26 Aug 2026 | 8,617 B | `41cff9af` | **clears** prior-turn reasoning | `turboderp/GLM-5.3-Flash-exl3` — snapshotted before the change |
| 27 Aug 2026 | 10,644 B | `34d5ee66` | **retains** it | `local-inference-lab/GLM-5.3-Flash-NVFP4`, the NVFP4 sibling |
| 4 Sep 2026, current | 10,950 B | `0c4099f3` | **retains** it | **this stack, both tracks** |

The 27 August revision flipped the default (line 4: `{%- set clear_thinking = clear_thinking if
clear_thinking is defined else false -%}`); that flip is the mechanism of §9.12. The 4 September
file is kept — it fixes the tool-call rendering defects recorded in [docs/11](11-open-issues.md)
§2.30, and jdecker76 independently reproduced that the 27 August file still renders a literal
`None` before a null-content tool call while the 4 September file does not — and the launcher now
guards the default instead (§9.12). **Closed.**

### 9.12 Prior turns' reasoning is re-rendered into every later prompt unless `clear_thinking` is sent **[SILENT]**

**Track:** both — a chat-template default; it affects any multi-turn client that echoes
`<think>…</think>` back inside `content`, which is what most agent frameworks do.

**Symptom.** Multi-turn sessions grow far faster than their visible text, and long agentic sessions
drift into repetition: what the model reasoned at turn N is fed back to it at turns N+1…, so one
degenerate turn primes the next. No single-turn gate can see it, and every gate in this repository
is single-turn. Reported from production by jdecker76 (issue #1) after four days of multi-user
agentic traffic on production configuration 13.

**Mechanism.** The served template (§9.11) defaults `clear_thinking` to `false`. When a client sends
the previous assistant turn back with its `<think>…</think>` block still inside `content`, template
lines 150–152 split the block out as `reasoning_content` and the retention branch renders it into
the prompt again. The `reasoning_content` **field** on an input message is dropped by vLLM before
the template sees it, so that path never fires; the inline path does.

**Measured on this stack** `[measured-here]`, 10 September, `/tokenize` on the production engine, a
three-message conversation whose one prior assistant turn carries about 900 tokens of reasoning:

| The prior assistant turn carries | default | `clear_thinking: true` |
|---|---:|---:|
| nothing | 30 | 30 |
| `<think>…</think>` inline in `content` | **1,771** | 30 |
| the same text in a `reasoning_content` field | 30 | 30 |

One prior reasoning turn is 59× the prompt, and it compounds with every turn.

**Fix, and it is in both launchers.** [`scripts/start-tp3.sh`](../scripts/start-tp3.sh) and
[`scripts/start-tp2full.sh`](../scripts/start-tp2full.sh) now always pass
`--default-chat-template-kwargs '{"clear_thinking":true, …}'`, with `reasoning_effort` appended when
`REASONING_EFFORT` is set. `enable_thinking`, which the launchers used to send, is not read by this
template at all and is no longer sent. With `clear_thinking: true` the template still renders the
reasoning of turns *after the last user message*, so a tool-call loop inside one turn keeps its own
reasoning; only earlier turns are cleared, which is the upstream API's behaviour.

**Two cautions.** On this build (vLLM `487ecf187`) a per-request `chat_template_kwargs` is **merged
over** the server default — `{"reasoning_effort":"high"}` alone keeps `clear_thinking: true` —
but jdecker76 measured the opposite (replacement) on their build, so a gateway that sends its own
`chat_template_kwargs` should carry `"clear_thinking": true` itself;
[`scripts/effort-proxy.py`](../scripts/effort-proxy.py) now does. And nothing measured in this
repository moves: every benchmark here is single-turn, where the flag is a no-op.

---

### 9.13 Malformed tool-call cascade — the parser salvages one bad call and the session ends in empty turns **[SILENT]**

**Track:** both — it is a parser default in the pinned vLLM, not a rank or a checkpoint property. It
needs agentic traffic to appear at all.

**Symptom, in the order users report it.** Long agentic coding sessions degrade and then stop: a tool
call that "stops halfway"; arguments with **missing or impossible keys**, so the client schema-rejects
them; the **same call repeated** identically two or three turns running; file paths and commands that
come back **mangled**; and finally a turn that is **completely empty** — the client says the turn
completed, and nothing happened. The engine log shows HTTP 200 and no parser error. In the captured
case 406 tokens were generated and **zero deltas reached the client**: one empty assistant role chunk,
`finish_reason=stop`, `stop_reason` 154829 (`<|observation|>`), usage, done. It gets worse **within** a
session, and no gate in this repository can see it, because every gate here is single-turn.

**Mechanism, and it is YuXiaoPan's (issue #7).** Four upstream lines in `vllm/parser/glm47_moe.py`
make a loop out of one rare event:

1. At ~60k+ tokens of agentic context the model occasionally **loses its own tool-call format** and
   emits malformed XML — including, in the captured vectors, Anthropic-style `</invoke>` and
   `<parameter name=` fragments, i.e. format loss rather than a typo.
2. `_glm47_arg_converter` (`glm47_moe.py:56-70`) **salvages** whatever its regexes match into a JSON
   dict and validates **nothing** about the keys — a real captured key is
   `print_code_snapshot</arg_value><arg_key>description`. Tool *names* are checked
   (`validate_tool_names=True`, `glm47_moe.py:171`) and `_is_valid_tool_name`
   (`parser_engine.py:392-397`) returns True unconditionally when the request carries no tools;
   argument keys are never checked. `stream_arg_deltas=True` (`glm47_moe.py:169`) has already streamed
   the fragments out, and `_build_extracted_result` (`parser_engine.py:1014-1065`) turns even a call
   that never closed into a tool call.
3. The client executes or schema-rejects it and **echoes the assistant message back into the history**,
   garbage `tool_calls` JSON included, which is what every agent framework does.
4. The chat template re-renders that garbage in canonical `<arg_key>` syntax — so the model is now
   reading **its own corruption as an in-context example**, and imitates it. Each round is worse until
   nothing parses and the whole turn is swallowed.

**How it was established, and the method is the transferable part.** A logging proxy in front of the API
caught the failing turn live, and the captured request body was then replayed **raw**: `/tokenize` on the
exact messages and tools, then `/v1/completions` on those token ids — no chat template, no parsers, so
nothing downstream of the model could be blamed or credited. That replay gave 5/5 corrupted calls at the
failing state, and then eliminated three explanations in turn: **not** the prefix cache (system message
salted, 0 % hits, still corrupted), **not** temperature (4/5, 4/5, 5/5 at 1.0 / 0.6 / 0.2 — deterministic
near zero, i.e. context-driven), and **not** depth as such (truncate the history to just before the first
corrupted call in it — 39 messages, 59.6k tokens — and the same model gives 8/8 well-formed calls at
temperature 1.0). A reproducible replay plus three eliminations is what turned "the model sometimes stops"
into a mechanism.

**The fix, in production since 11 September 2026.** A prelude patch makes the parser **fail closed**:
`stream_arg_deltas` off so a call is buffered until it closes; at `TOOL_CALL_END` the complete call is
validated (name identifier-shaped and in the request's tools; every argument key identifier-shaped and in
the tool's schema `properties`; a non-empty body the converter salvaged nothing from counts as truncated);
and a call that fails is surfaced as plain **content** with its slot cleared, so neither the streaming path
nor `_build_extracted_result` can emit it. **The cascade breaks at step 3** — nothing that looks like a tool
call re-enters the history. `HAREM_GLM47_FAILCLOSED`, default **on**, `=0` upstream byte for byte:
[`../tracks/tp3/patches/glm47-failclosed/`](../tracks/tp3/patches/glm47-failclosed/README.md), gates
[`../results/gates/failclosed-11sep.md`](../results/gates/failclosed-11sep.md).

**How to verify it on your own engine.** Three checks, in increasing order of effort:

- The boot log carries `[HAREM-GLM47-FAILCLOSED] failclosed=True (HAREM_GLM47_FAILCLOSED='')` once per
  process. **No line means the patch did not run** — almost always a missing `--in-place`, which makes
  the script a dry run that exits 0.
- A rejected call appears in the response as **raw `<tool_call>…</tool_call>` text inside `content`**,
  with one warning line in the API log naming the tool and the reason (never the argument body — that is
  user content). Garbage you can see is the working state; an empty turn was the broken one.
- A valid streaming tool call now arrives as **one** `tool_call` delta rather than a name delta plus
  argument deltas. That is the intended behaviour change and the one thing a client might notice.

**What is still open.** The patch does not touch the **rate** of the first corruption, only its
consequence — see [11](11-open-issues.md) §2.30, where whether the 4bpw EXL3 checkpoint raises that rate
against the NVFP4 sibling is still varying with the build. The gate that would catch this class before a
user does is [HELP-WANTED](../HELP-WANTED.md) §14, which now wants a **tool-call** transcript as well as a
text one.

---

## 10. Operations

### 10.1 A reboot brings up the sibling NVFP4 engine, not this one **[SILENT]**

**Track:** both — two engines on one node's memory; explicitly not a rank-count hazard.

The failure that made this entry: an unattended reboot did **not** leave the cluster down — it brought
up the **other engine**, on the same GPUs and the same unified memory, which is worse than nothing if
you were expecting this one. No error anywhere; `/health` on port 8001 simply never answers while a
different, healthy engine holds the machine.

**Fix, and it is now done on our nodes.** `harem-motor.service` is `disabled` on all three, and
[`tracks/tp3/harem-exl3.service`](../tracks/tp3/harem-exl3.service) carries `Conflicts=harem-motor.service`
besides. Whichever unit you install, disable the other **in the same change** — installing one without
disabling the other is not a partial fix, it is a worse state than having no unit at all.

The rest of the old entry is closed: the preflight
[`tracks/tp3/motor-onkosul-exl3.sh`](../tracks/tp3/motor-onkosul-exl3.sh) exists, `ExecStop` names
`exl3-tp3` rather than the NVFP4 container, and `TimeoutStartSec` is **1200** — enough for a 620 s dump
boot with the preflight and the settle gate in front of it. See §10.1a for what is still not solved.

### 10.1a Rebooting one node kills the fabric, and the unit will happily start into half of it

**Track:** both, measured at TP=3 only — the rule transfers to two nodes; only the three-node reboot test was run.

**Reboot all three nodes together, or none.** This is the oldest rule in this stack and the autostart
unit does not change it: bringing one node back takes down the far end of that node's links and the
pair does not heal (§4.1, [00](../docs/00-hardware-and-os.md) §3). Before the hotplug fix
(`/etc/nvidia/cx7-hotplug-enabled` removed on all three) the symptom was a pair going dead on the two
nodes that had **not** rebooted; with the fix in place the rule still stands, because the fix removes
the mechanism we identified and does not prove there is no second one.

What the unit adds to that rule is a way to get it wrong faster. The preflight waits for `ibv_devinfo`
4/4 **on its own node** and pings its two neighbours, so a single-node reboot into a healthy pair will
pass the preflight and start a rank into a cluster whose other two ranks are already gone — the
rendezvous then hangs until `TimeoutStartSec`, which is 1200 s. If you must bring one node back on its
own, **check the fabric on the other two before you let the engine start**, not just on the one you
rebooted.

Two more things the unit does not do, both stated in [`systemd/`](../systemd/README.md) and both
`[not tested]`:

- **It does not enforce the worker-2 → worker-1 → head start order.** systemd starts the three units
  independently. The all-three reboot test passed on the workers' rendezvous retrying until rank 0
  appears, plus a `TimeoutStartSec` about four times a normal boot — margin, not a guarantee, and
  three whole-cluster trials rather than a proof. If the rendezvous ever times out on a cold cluster,
  this is why.
- **Nothing watches the container once it is up.** `--restart no` is deliberate and unchanged. One
  outage during this work ran an hour purely because the only thing being watched was a benchmark
  log; a 60-second `docker ps` plus `/health` poll that dumps `docker logs --tail 40` when the
  container is gone is a few lines and is still not written.

**What a healthy boot from power-on looks like**, for comparison against yours `[measured-here]`: ssh
and `ibv_devinfo` 4/4 at about **+30 s**; the unit logs `Finished` at **+98 to +103 s**, which is the
whole preflight plus the settle gate; `/health` 200 roughly **three and a half minutes** after that.
If the unit finishes in seconds, the preflight did not run. If it sits in `activating` past ten
minutes, the preflight is waiting on the fabric and its own message will say which check.

### 10.2 Environment files must never be copied between nodes

**Track:** both — each node's env file carries node-specific values, at any rank count.

Each node's env file carries node-specific values. **Derive each node's file with `sed`, never `scp`
it.** Every env token in this repository is documented that way.

### 10.3 Do not touch the patch directory while a boot is in progress

**Track:** both, measured at TP=3 only — the manifest identity only exists where a fast-load sidecar was built.

It is mounted **live** into the container, so an edit changes the manifest identity underneath a
running dump. This has happened once and it cost the boot.

### 10.4 Two patch trees, and a fix in one does not reach the other **[SILENT]**

**Track:** both, measured at TP=3 only — the divergence is forced by the fast-load manifest, never built at TP=2.

`patches/tp3/` and `tracks/tp3/patches/` diverge in exactly two constants plus one patch. **The code did
not have to diverge** — the two constants are provable no-ops at TP≤2, so one tree could serve both.
**The fast-load manifest identity did.** Adding the full-scope patch to `patches/tp3/` would have
refused the next production boot on all three nodes, which is exactly what happened twice.

The merge is technically possible and was not done, because it would change production 8's identity
and cost a dump boot on the arm being kept as the rollback. `tracks/tp3/patches/README.md` carries a
file-by-file `cmp` — a mitigation, not the fix.

**The disk half:** 53 GB × 3 for production 9's sidecar on top of production 8's ~63 GB × 3, on top of
two 154+ GiB checkpoints × 3. One node had 51 GB free before the arm and needed old sidecars cleared
first.

### 10.5 A two-node boot dies on the drafter, because the TP=3 pad was written into its config

**Track:** **TP=2 only** — the pad it trips over exists only to serve three ranks.

Four minutes into a two-node boot, after the target model has already been read:

```
File ".../vllm/model_executor/models/qwen3_dflash.py", line 186, in __init__
    assert self.total_num_kv_heads % tp_size == 0
AssertionError
```

**The drafter is not the problem.** The DFlash2 drafter's own GQA is 32/8 and 8 divides by two;
[04](04-dflash2-port.md) says so in its first line. The problem is that the three-node track applies
its 32/8 → 36/9 pad by **rewriting `config.json` inside the drafter directory**, keeping the original
beside it as `config.json.orig`. Nine key-value heads do not divide by two.

**Fix — a directory, not a patch.** The weights are identical; only the config differs, so hard-link
the safetensors rather than copying 2.3 GB:

```
install -d /var/tmp/dflash2-draft-tp2
```

```
cp /var/tmp/dflash2-draft/config.json.orig /var/tmp/dflash2-draft-tp2/config.json
```

```
ln /var/tmp/dflash2-draft/model.safetensors /var/tmp/dflash2-draft-tp2/model.safetensors
```

Then point `DRAFT_HOST_PATH` at it. `scripts/start-tp2full.sh` has always defaulted to that path;
what was missing was the sentence saying why the directory has to exist separately.

**Do it before the dump boot.** The fast-load sidecar's identity hashes the drafter's `config.json`,
so fixing the config afterwards invalidates the sidecar and costs a second dump — which is exactly
what it cost us, 8 September 2026 `[measured-here]`. [19](19-vision-at-two-ranks.md) §3.

### 10.6 Registering a new prelude patch invalidates the sidecar — and the preflight gate demanded one in dump mode

**Track:** both, measured at TP=3 only — the `ExecStartPre` gate is in both tracks' preflight scripts
and only the three-node one has ever been exercised against a new sidecar directory.

**Symptom, part one.** A new `patch-*.py` is copied into the patch directory and the prelude is taught
to call it. The behaviour is env-gated and switched off, so nothing about the served model changes.
The next boot is refused on every node anyway:

```
preflight-fastload: sidecar stale - boot refused
  patches.patch-flashkda-tp3.py: recorded='<none>' now='4e9a0415...'
  patches.tp3-prelude.sh:        recorded='3b78d02d...' now='be712c25...'
```

**Symptom, part two — the expensive one.** The fix for part one is to dump a new sidecar into a new
directory. That restart never starts the container at all: all three units go straight to `failed`,
before any engine log exists.

```
fast-load sidecar missing: /var/tmp/glm53-exl3-flashkda-r0
```

**Mechanism.** `harem_fastload_id.file_identity()` hashes `glob($TP3_DIR/patch-*.py)`, the full text of
the prelude and the overlay module into the manifest ([08](08-fast-boot.md) §4), so the *presence* of a
file is part of the identity even when the code it contains never runs. §3.1 and §3.2 already say that.
What was new on 10 September is the second half: `motor-onkosul-exl3.sh`, the unit's `ExecStartPre`,
tested `$FASTLOAD_DIR-r$RANK/MANIFEST.json` **regardless of `FASTLOAD_MODE`** and skipped only when
`FASTLOAD_DIR` was empty. Dump mode is what *creates* that directory, so a new `FASTLOAD_DIR` could
never be booted: the mode whose job is to make the directory required the directory to already exist.

**What it cost.** The engine was down from 03:26:28 to 03:44:18 UTC — **17 min 50 s** — while a
chicken-and-egg in our own gate was diagnosed `[measured-here]`. No production artefact was
overwritten and the previous sidecars were intact throughout, which is the only reason it was 18
minutes and not an evening.

**Fix — narrow the gate, do not remove it.** One clause, and it is in the shipped
[`tracks/tp3/motor-onkosul-exl3.sh`](../tracks/tp3/motor-onkosul-exl3.sh):

```
FM=$(grep -E "^FASTLOAD_MODE=" "$ENVF" | cut -d= -f2)
[ -z "$FD" ] || [ "$FM" != load ] || test -f "$FD-r$R/MANIFEST.json" || { echo "fast-load sidecar missing: $FD-r$R"; exit 1; }
```

`load` is the only mode in which a missing sidecar is fatal. In `dump` the launcher creates the
directory itself; with the mode empty fast loading is not in play. The same edit is carried in
[`tracks/tp2/motor-onkosul-exl3-tp2.sh`](../tracks/tp2/motor-onkosul-exl3-tp2.sh) `[not tested]` at two
ranks.

**How to plan a patch A/B after this, in order.**

1. **Run the comparison sidecar-less.** Set `FASTLOAD_MODE=` empty in *both* arms — the launcher then
   skips the whole fast-load block and the prelude never calls the preflight — and register the patch
   in both, so one environment variable is the only difference. There is no "registered but switched
   off" state that keeps the sidecar valid, so this is not a preference; it is the only symmetric A/B
   available.
2. **Expect a slower boot and a different boot path.** Ours were 285 s and 315 s against production's
   165–234 s, and a sidecar-less boot moved single-stream decode by 4.7 % on its own — which a
   single-arm comparison would have published as a kernel gain
   ([`results/gates/flashkda-ab-10sep.md`](../results/gates/flashkda-ab-10sep.md) §2.1).
3. **Re-measure the winner on the real path.** Dump into a **new** `FASTLOAD_DIR` (§3.4 — reusing the
   name overwrites the rollback), switch to `load`, and repeat the headline measurement: ours read
   1,865.7 and 1,866.6 tok/s against the sidecar-less arm's 1,867.4 / 1,868.8.
4. **Keep the old sidecar until the configuration is retired.** 53 GB per node per sidecar. Check the
   free space first: ours went 546 G → 493 G.

**And print which arm ran.** Neither the sidecar nor the gate tells you whether the patch did anything.
A patch script that defaults to a dry run without `--in-place` produces a healthy boot that silently
keeps the old code path — so the patched code should say so once per process, as
`[HAREM-FLASHKDA] kda_prefill_backend=flashkda` does.
[`tracks/tp3/patches/flashkda/`](../tracks/tp3/patches/flashkda/README.md) §3.

---

---

## 11. The silent-failure index

Ranked by how expensive they are to discover late. **If you read one table on this page, read this
one.**

| # | Silent failure | What it looks like | §  |
|---|---|---|---|
| 1 | `GPU KV cache size` is a delta between two `/proc/meminfo` readings, and it runs backwards | a node that starts dirty awards itself a bigger pool | 5.8 |
| 2 | Half the fabric had never carried a packet | every link `ACTIVE`, every benchmark ran, `port_xmit_data = 0` | 4.3 |
| 3 | The `n_rows` omission | no error, correct output, one rank quietly doing 1.2–2.0× the work | 7.1 |
| 4 | `CUDA_EXL3_DEBUG_NAMES` printed nothing | a gate whose passing state is silence, and silence is also what a gate that never ran looks like | 8.5 |
| 5 | A wrong tuple shard index | loads without error, writes to the wrong slice | 9.4 |
| 6 | The 2,112 shared-expert pad | one half refuses loudly, the other is a half-block pad with **only a warning** | 2.9 |
| 7 | A padded head holding allocator garbage | a fluent drafter proposing wrong tokens; acceptance falls, nothing crashes | 9.2 |
| 8 | `disable_tp` on the shared expert | 3× shared contribution; fluent and wrong | 9.1 |
| 9 | DFlash2 on the V1 model runner | boots, correct-looking text, quietly loses acceptance | 9.5 |
| 10 | The swallowed `flashinfer.comm` preload | an illegal address later, on a different rank, at a random-looking time | 7.6 |
| 11 | The EP evidence line discarded by the logger | the raises worked, the evidence line and a safety-net **warning** did not | 8.6 |
| 12 | The vision tower at TP=2 | built and unused for the life of the stack — the "1.05 GiB" this row used to quote is withdrawn, [11](11-open-issues.md) §1.14 | 2.7 |
| 12a | The packed-mapping entry the full-scope patch shadows | 24 of 99 vision linears fall back to BF16 **silently** — `get_quant_method` fails open, so the tower loads clean and answers wrong | 2.7a |
| 12b | An anchor written against the stock file | the model-free gate says PASS and the boot stops at exit 21, because the gate did not reproduce the prelude's patch **order** | 2.7d |
| 13 | Long prompts never scheduled | no error, no timeout — `Running: 0, Waiting: 1` forever | 5.4 |
| 14 | Two launcher copies | both run, both produce a server, one silently lacks three features | 8.7 |
| 15 | The EXL3 pad invariant holding by accident | correct, unchecked, one arithmetic change from silently wrong | 9.3 |
| 16 | DBO splitting a request across micro-batches | recurrent state restarted from zero — fluent, wrong, silent | 9.8 |
| 17 | Prefill on a repeated prompt | 1,596 tok/s where the honest number is 1,025 | 8.3 |
| 18 | `git archive` dropping the untracked Dockerfile | the build fails later, on a missing file | 2.19 |
| 19 | The AOT compile check | 18/18 "pass", 6/18 die at launch | 7.3 |
| 20 | The sibling's systemd unit | a reboot brings up the wrong engine on the same GPUs, healthily | 10.1 |
| 21 | The salvaged malformed tool call | the client echoes it back, the model imitates it, and the session ends in empty turns reported as complete | 9.13 |

---

## 12. Where the retractions live

Things we published here and then disproved are **not** on this page — they are in
[11 — Open issues](11-open-issues.md) §1, with the measurement that overturned each, and indexed in
[audit/](../audit/README.md) §6. Two of them are the shape of every other mistake on this stack:

- **The ruler gets measured too.** Seven of the thirty-two audit rows are a number we quoted instead
  of measuring — a catalogue bandwidth, a wire capacity, and once **a figure the engine printed in its
  own log about its own memory**.
- **A pair of sweeps is not a result.** Six more are a single probe or a confounded pair treated as
  evidence, against a stack whose boot-to-boot spread reaches 15.9 %.

---

## 13. If you are about to report something to us

Include, or we cannot help:

- The output of §0 steps 1–7.
- Your image tag and the `cuda-exl3` commit it was built from. "The current build" is not a revision.
- Your full environment file, with addresses removed.
- `docker logs <container> 2>&1 | head -200` from **every** rank, not just the head.
- Whether the tuner cache was warm, and how many sweep rounds you ran.
- Whether the quality gates passed **cold and warm**.

And read [CONTRIBUTING](../CONTRIBUTING.md) first — several of the things a second cluster would most
usefully settle are listed there, with the reason we could not run them ourselves.
