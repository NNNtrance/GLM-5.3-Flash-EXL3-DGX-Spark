# 18 — Vision at three ranks: the tower comes back, and the video sampler bug behind it

**Applies to: the TP=3 track.** §2 explains why three ranks is the hard case and §11 is the two-node
note: at two ranks the tower divides cleanly, so only the flags and the video half of the patch are
needed — `[not tested]`.

**This page retracts a claim this repository made everywhere else.** Until 7 September 2026 the
recipe said the vision tower cannot run at TP=3 and served text only, with
`--language-model-only` and a three-anchor patch that stops the tower being built at all
([03](03-tp3-padding-and-sidecars.md) §3). That was true of the flags. It was not true of the
hardware, and it was never tested against the checkpoint we actually serve. Since 7 September the
production configuration accepts **4 images and 2 videos per request** and the text side is
unchanged.

What it took was one flag, six patch anchors, four environment settings and a new model-free gate —
and the largest single obstacle turned out to have nothing to do with three ranks, EXL3, expert
parallelism or anything else in this repository. It is an upstream accounting bug in vLLM's
GLM-5-Next port that kills the engine core on the **first video request** at any parallel size
(§4). It is filed as [vllm#55644](https://github.com/vllm-project/vllm/issues/55644), with the fix
and a unit test in [vllm#55647](https://github.com/vllm-project/vllm/pull/55647) — see §12.

**What this cost, in one line:** C1 **+0.1 %**, C8 **+1.5 %**, a KV pool inside configuration 12's
own boot-to-boot spread, draft acceptance and TTFT unchanged, a whole-cluster reboot at 318 s against
311 s — and, off the tables, one extra 53 GB-per-node sidecar dump boot and about 1.3 GiB more
resident host memory on the rank that serves the API, measured and measured to be harmless (§9)
`[measured-here]`.

The patch tree, the gates and the exact environment are
[`tracks/tp3/patches/vision/`](../tracks/tp3/patches/vision/README.md). The gate results are
[`results/gates/vision-gates-tp3.md`](../results/gates/vision-gates-tp3.md).

---

## 1. Why it was off, and what was actually wrong with the reason

### 1.1 The 16-head tower

`config.json` gives the tower as `Glm5NextForConditionalGeneration` → `vision_config`:

| Field | Value |
|---|---|
| `depth` | 24 |
| `hidden_size` | 1024 |
| `num_heads` | **16** (head dim 64) |
| `intermediate_size` | 4096 |
| `out_hidden_size` | 4096 |
| `projection_intermediate_size` | 10240 |
| `patch_size` | 14 |
| `spatial_merge_size` | 2 |
| `temporal_patch_size` | 2 |

Every linear in `vllm/models/glm5next/nvidia/multimodal.py` is built as a tensor-parallel linear and
divides its width by `tp_size`. `dist_utils.divide(16, 3)` asserts, and the engine never starts.
That much was known and correctly reported.

**What was not reported is that the head count is not the only thing that fails to divide.** At
three ranks *nothing* in the tower divides:

| Module | Width | Divides by 3 | Divides by 2 |
|---|---|---|---|
| `attn.qkv` head count | 16 | no | yes |
| `attn.proj` input | 1024 | no | yes |
| block `mlp.gate_up_proj` output | 4096 (×2) | no | yes |
| block `mlp.down_proj` input | 4096 | no | yes |
| `merger.proj` | 4096 | no | yes |
| `merger.gate_up_proj` output | 10240 (×2) | no | yes |
| `merger.down_proj` input | 10240 | no | yes |

This is the fact that decides §2: padding the heads 16 → 18, which is the pattern the text side uses
everywhere, would have been *four* padded shapes and a new pad audit rather than one.

### 1.2 `--language-model-only` never stopped the tower being built, and the reason is subtler than we wrote

The flag makes `MultiModalConfig.get_limit_per_prompt` return 0, so no image can be *submitted*. It
does not stop the tower being *constructed*. [03](03-tp3-padding-and-sidecars.md) §3 called this a
missing check. There is in fact a mechanism, and it runs too late:
`interfaces.py:296-334` `_mark_tower_model` is a **context manager**, and when every modality limit
is zero it enters `no_init_weights(...)`, which installs a module-registration hook and
`torch.device("meta")` — so `__init__` bodies still **run**, tensors merely stay on the meta device
and each assigned child is replaced by a `StageMissingLayer`. `divide(16, 3)` inside
`Glm5NextVisionAttention.__init__` is plain Python arithmetic. The meta device does not save it.

Our patch works: three anchors in `patch-vllm-tp3.py` — a builder that returns `None` when
`language_model_only` is set, the construction site routed through it, and a `load_weights` override
that skips the `visual.` prefix. It is still in the tree and is still the text-only fallback.

### 1.3 Retracted: "1.05 GiB of BF16 vision weights on every rank at TP=2"

[03](03-tp3-padding-and-sidecars.md) §3 and [14](14-troubleshooting.md) §2.7 both carried this
figure. It is wrong twice `[retracted]`.

- **It is the wrong checkpoint.** 347 dense BF16 vision tensors totalling 1.05 GiB describes
  `brandonmusic/GLM-5.3-Flash-tr3-4bpw`, the routed-experts-only checkpoint that configurations 1–8
  served, where the tower really is dense BF16. On the production checkpoint
  (`turboderp/GLM-5.3-Flash-exl3` at 4.05 bpw) **the tower is 6-bit EXL3 and weighs 0.557 GiB in
  total, 0.416 GiB per rank when replicated** — measured from the safetensors headers, §3.1.
- **On this image the flag probably does not spend it either.** With `StageMissingLayer` and the
  meta device (§1.2), the tower's tensors are never materialised and `AutoWeightsLoader` skips the
  module, so `--language-model-only` on image `exl3-zeus:754421f` should cost nothing at TP=2. That
  is a reading of the installed source, not a boot-log measurement, and it has not been confirmed
  `[not tested]`.

What replaces the claim: at TP=2 the flag's cost is **unmeasured and probably zero on this image**;
at TP=3 the cost was never memory in the first place, it was that the engine would not start.

---

## 2. Three ways to split a tower that does not divide by three, and why we took the first

### (a) Replicate it — `--mm-encoder-tp-mode data` — **taken**

The flag exists in this vLLM revision, and this model supports it. Six independent places say so:
the config field and its two values (`config/multimodal.py`), the CLI flag (`engine/arg_utils.py`),
the per-model capability flag defaulting to `False` (`interfaces.py:152`), the override
**`supports_encoder_tp_data = True`** on our superclass (`glm4_1v.py:1782`), the tower's own read of
it (`glm5next/nvidia/model.py:1069`), and the execution path
(`glm4_1v.py` → `run_dp_sharded_mrope_vision_model`, `vision.py:397-584`, with load balancing at
`vision.py:328` and an all-gather back at `:542`).

Every rank holds the whole tower: **0.416 GiB per rank**, one flag, no padded shapes, no new audit.

**The honest cost of (a):** one large item — a single image, a single video — runs on **one rank**
while the other two wait. A single-image request does not get a faster tower from three nodes. A
multi-item request (4 images + 2 videos) spreads across ranks.

**A one-line upstream fix we did not need.** `vision.py:144-161` `is_vit_use_data_parallel(num_heads)`
already warns and falls back to data parallelism when the head count does not divide — but
`glm5next/nvidia/multimodal.py` calls it with **no arguments** everywhere (lines 101, 140, 310, 375),
so the automatic path never fires and only the explicit flag works. Passing
`num_heads=vision_config.num_heads` is a one-line change. We did not make it: the flag does the job,
and a patch we do not need is a patch we have to keep in step.

### (b) Pad the heads 16 → 18 — **rejected, and the arithmetic is why**

Because §1.1's table has seven rows rather than one, this road is four padded shapes:

| Shape | Today | Padded for TP=3 | Per rank | 128-blocks |
|---|---|---|---|---|
| heads | 16 | 18 (1024 → 1152 columns) | 384 | 3 × 128 |
| block `mlp` intermediate | 4096 | 4608 | 1536 | 12 × 128 |
| `merger` d_model | 4096 | 4608 | 1536 | 12 × 128 |
| `merger` context | 10240 | 10752 | 3584 | 28 × 128 |

The arithmetic *works* — every padded width is a multiple of `lcm(128, 3) = 384`, the same rule the
text side's five shapes obey ([03](03-tp3-padding-and-sidecars.md) §1). So this is not impossible,
it is disproportionate: the tensors are EXL3 trellises, so each pad needs the `svh = 0` / `suh = 0`
narrow-load path and [03](03-tp3-padding-and-sidecars.md) §5.1's A10 pad audit rewritten for 24
blocks × 4 sites plus the merger — all to save **0.28 GiB per rank** (0.416 → 0.139), about 0.5 % of
the KV pool. `[estimate]`

### (c) A flag that pins the encoder to one rank — **does not exist here**

The module-level `disable_tp` parameter is real but is driven only by `is_vit_use_data_parallel()`,
and so is `use_data_parallel`. There is no separate CLI flag. Option (c) *is* option (a) in this
revision.

---

## 3. The two loader blockers, which is where the work actually was

`--mm-encoder-tp-mode data` clears the assert and `LANGUAGE_MODEL_ONLY=0` makes the launcher pass
the per-request limits. **Neither makes the tower load.** Two facts, both measured from the
checkpoint rather than assumed:

### 3.1 The tower in this checkpoint is 6-bit EXL3, not BF16

`[measured-here — safetensors headers only, no GPU, no engine]`. 19 shards, 148,046 tensors, of
which **1,007 are `model.visual.*`**, all in shard 19, **0.557 GiB** in total:

| Group | Count | Size | dtype | Example |
|---|---:|---:|---|---|
| block attn q/k/v/proj trellis | 96 | 72.00 MiB | I16 | `model.visual.blocks.N.attn.q_proj.trellis` `[64,64,96]` |
| block mlp gate/up/down trellis | 72 | 216.00 MiB | I16 | `...mlp.gate_proj.trellis` `[64,256,96]` |
| merger proj/gate/up/down trellis | 4 | 102.00 MiB | I16 | `model.visual.merger.down_proj.trellis` `[640,256,96]` |
| `suh` / `svh` / `bias` | 566 | ~1.4 MiB | F16 | `...q_proj.suh` `[1024]` |
| `mul1` (codebook multiplier) | 172 | ~0 | I32 | `...q_proj.mul1` `[]` |
| norm weights | 97 | ~0.1 MiB | BF16 | `...norm1.weight` `[1024]` |
| `downsample` (Conv2d) | 2 | 32.01 MiB | F16 | `[4096,1024,2,2]` |
| `patch_embed.proj` (Conv3d) | 2 | 2.30 MiB | F16 | `[1024,3,2,14,14]` |
| **residue — the pre-quantization fused dense qkv** | 48 | **144.14 MiB** | F16 | `...attn.qkv.weight` `[3072,1024]` |

The bit width is arithmetic, not a label: a 1024×1024 weight is stored as `64×64×96` I16 = 786,432
bytes = **6 bits per weight**, consistent with the text body at 4.05 bpw and `lm_head` at 6 bits.

And vLLM builds it dense anyway. `glm5next/nvidia/model.py:1074-1091` passes `quant_config=None`
**unconditionally**, with a comment explaining that the tower ships BF16 weights in "this fp8
checkpoint" and would produce NaN image features if it inherited the global fp8 config. That comment
is correct for the zai-org fp8 checkpoint it was written against and wrong for ours. Built dense, the
tower looks for `visual.blocks.N.attn.proj.weight`, `...mlp.*.weight` and `visual.merger.*.weight` —
**none of which exist on disk.** The 48 fused `attn.qkv.weight` tensors make it *look* as though both
paths are fed; they are not, because `proj`, `mlp.*` and `merger.*` have no dense counterpart at all.

**Fix: VS1.** Pass the real config through, gated on the knob *and* on the config being EXL3, so
every other checkpoint keeps upstream's `None`.

### 3.2 The EXL3 manifest does not list the tower — and the recovery path needs a union

`quantization_config.json`'s `tensor_storage` table has 37,032 entries and **zero** containing
"visual". (The 349 occurrences of "visual" in that file are in `modules_to_not_convert`, inherited
from the original fp8 release where the tower genuinely was BF16 — history, not instruction.)

`cuda_exl3/config.py:171` `_augment_from_checkpoint()` exists for exactly this: *"Some checkpoints
omit modules from quantization_config.json even though the tensors are present"*, and recovers them
from the safetensors headers. Run against the real config and index, on a CPU
`[measured-here]`:

```
modules in map: 36719
visual modules after augment: 172
model.visual.blocks.0.attn.q_proj   Exl3ModuleInfo(1024x1024, bits=6, cb=2)
model.visual.merger.down_proj       Exl3ModuleInfo(10240x4096, bits=6, cb=2)
```

Then `resolve()` has to map vLLM's fused module names onto those, and **neither packed mapping alone
is enough**:

| vLLM module | upstream Glm4v mapping | full-scope mapping | union |
|---|---|---|---|
| `visual.blocks.0.attn.qkv_proj` | resolves (q/k/v) | **None** | **resolves** |
| `visual.blocks.0.attn.proj` | resolves | resolves | resolves |
| `visual.blocks.0.mlp.gate_up_proj` | **None** | resolves (gate/up) | **resolves** |
| `visual.blocks.0.mlp.down_proj` | resolves | resolves | resolves |
| `visual.merger.proj` | resolves | resolves | resolves |
| `visual.merger.gate_up_proj` | **None** | resolves | **resolves** |
| `visual.merger.down_proj` | resolves | resolves | resolves |

With `HAREM_EXL3_FULLSCOPE=1`, `patch-fullscope-tp3.py`'s A2 **replaces** the class's
`packed_modules_mapping` wholesale, so the inherited Glm4v vision entries — including `qkv_proj` —
disappear. Harmless while the tower is off; fatal when it is on. Without the union, **24 of the 99
vision linears do not resolve and `get_quant_method` falls back to BF16 silently**: a tower that
loads clean and answers wrong.

**Fix: no patch.** `cuda_exl3/config.py:112-125` and `:381-386` merge `CUDA_EXL3_PACKED_MAPPING`
**under** the class mapping, so an environment entry can only *add*. The `.env` carries
`CUDA_EXL3_PACKED_MAPPING={"qkv_proj":["q_proj","k_proj","v_proj"]}` and
[`check-vision-mapping.py`](../tracks/tp3/patches/vision/check-vision-mapping.py) proves the union
model-free, in the boot log, before a weight moves.

### 3.3 Two smaller ones, VS2 and VS3

**VS2 — the spelling.** `Glm5NextVisionTransformer.hf_to_vllm_mapper`'s `orig_to_new_stacked`
dictionary maps `.attn.q.` → (`.attn.qkv.`, "q"), the GLM-OCR / GLM-4V spelling. This checkpoint
writes `.attn.q_proj.trellis`. Nothing matches, and the three EXL3 shards never reach the fused
module. Add the three `_proj` spellings; the destination stays `.attn.qkv.`, because the *attribute*
is `self.qkv` whatever the prefix says. The `.gate_proj` / `.up_proj` → `.gate_up_proj` entries
already exist and work for the merger too.

**VS3 — the dead tensors, and the count that must not be asserted.** The 48 residual fused
`attn.qkv.{weight,bias}` tensors have no home under EXL3 — the module has no `.weight` parameter —
so `AutoWeightsLoader` raises *"There is no module or parameter named ..."*. Filter them, then audit
the built tower.

The audit is the point; the count is not. **48 came from the headers and the real load delivers 0 or
1**, because `AutoWeightsLoader` already adds `.bias` to `ignore_unexpected_suffixes`
(`models/utils.py:404`) before this filter sees anything, and its prefix grouping consumes the rest.
Asserting 48 killed a boot whose tower was completely correct. The dropped count is now reported as
information. What is asserted is the invariant: **99 EXL3 linears, 0 unquantized, and no fused qkv
module left holding a dense `.weight`** — that second clause is what catches a module that fell back
to BF16, because `Exl3Config.get_quant_method` fails *open*.

Measured side finding, worth knowing before you write a stricter check:
`Glm5NextVisionTransformer.load_weights` is called **796 times** on one boot — the loader splits by
contiguous prefix runs — so "is the loaded name set complete" cannot be asked in one place.

---

## 4. The video sampler bug: two frame samplers, and a dead engine core

Rounds 1 and 2 of the trial arm loaded the tower perfectly and answered image questions correctly.
The **first video request killed the engine core on all three ranks**, identically, in the same
second.

```
File ".../vllm/v1/worker/gpu/mm/encoder_runner.py", line 285, in get_inputs_embeds
File ".../vllm/model_executor/models/interfaces.py", line 450, in embed_input_ids
File ".../vllm/model_executor/models/utils.py", line 657, in _merge_multimodal_embeddings
    inputs_embeds[is_multimodal] = mm_embeds_flat.to(dtype=input_dtype)
RuntimeError: shape mismatch: value tensor of shape [1656, 4096] cannot be broadcast to
              indexing result of shape [1728, 4096]
ValueError: Attempted to assign 1656 = 1656 multimodal tokens to 1728 placeholders
```

Worker dead → `EngineDeadError` → HTTP 500 on every subsequent request. A single malformed
multimodal request took down three nodes.

### 4.1 Two readings we published and then measured away `[retracted]`

- *"1,728 = 4 × 36 × 48 / 4, so the width was rounded to 48 instead of 46."* **Wrong.** 1,728 and
  1,768 are not totals; they are **chunk-local** placeholder counts under chunked prefill at
  `--max-num-batched-tokens 2048`, the pieces 4,968 breaks into. `H` and `W` are read on both sides
  from the **same** grid tensor, so a width disagreement is structurally impossible.
- *"The `max_pixels` brake is the first suspect."* **Wrong, and the brake is innocent** — it never
  reaches the video processor at all (§5.1).

### 4.2 The measured chain

`[measured-here — CPU container, no `--gpus`, engine down]`. Fixture: 640×480, 32 frames, 8 fps,
4.0 s, the same geometry as the clip that killed round 1.

```
LOADER    frames.shape = (32, 480, 640, 3)
          metadata = {'total_num_frames': 32, 'fps': 8.0, 'duration': 4.0, 'do_sample_frames': True}
PIXEL     sample_frames() -> 8 frames: [0, 4, 8, 13, 17, 22, 26, 31]
          video_grid_thw = [[4, 36, 46]]   pixel_values_videos = (6624, 1176)
          ENCODER ROWS = 4*36*46/4 = 1656
PLACEH    timestamps = 12  [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
          PLACEHOLDERS = 12 * 414 = 4968
VERDICT   encoder rows 1656   placeholders 4968   deficit 3312   ratio 3.0
```

`video_grid_thw = [[4, 36, 46]]` is bit-for-bit what the engine itself dumped in round 1, so the CPU
bench reproduces the serving path rather than approximating it.

| # | Where | What it does |
|---|---|---|
| 1 | `transformers_utils/processors/glm5next.py:597` `Glm5NextVideoProcessor.sample_frames` | `target_fps = fps_interval = 2.0`, `max_frame_count = 2048` |
| 2 | `glm5next.py:83` `glm_sample_frame_indices` | `extract_t = int(duration × target_fps)` = int(4 × 2.0) = **8** frames |
| 3 | `glm5next.py:675` `_preprocess` → `smart_resize` | → `grid_thw = (4, 36, 46)`, 6,624 patches |
| 4 | `models/glm4_1v.py:1903` `_process_video_input` | encoder output = `grid.prod() // merge²` = **1,656 rows** |
| 5 | `glm4_1v.py:1722` `get_video_replacement` | takes the **same** `video_grid_thw` |
| 6 | `glm4_1v.py:1452` | `num_tokens_per_frame = H*W // merge²` = **414** — still consistent |
| 7 | `glm4_1v.py:1456` | the loop runs over **`timestamps`, not `T`** |
| 8 | `glm4_1v.py:1265` `_get_video_second_idx_glm46v` | **a second, independent sampler**: GLM-4.6V's `DYNAMIC_FPS_THRES` gives `target_fps = 3`; `extract_t = int(dur × target_fps × temporal_patch_size)` = **24**; `[::2]` → **12 timestamps** |
| 9 | `encoder_runner.py:285` → `interfaces.py:450` → `models/utils.py:657/665` | 1,656 embeddings into 4,968 placeholder slots → worker dead |

**The gap in one sentence.** `Glm5NextProcessingInfo` overrides `get_hf_processor`,
`_get_vision_info`, `_get_image_max_pixels` and `_get_video_max_pixels` — but **not**
`_get_video_second_idx_glm46v` or `_construct_video_placeholder`. The pixel path was moved to
GLM-5-Next's own sampler and the placeholder path was left on GLM-4.6V's constants. Even the
`extract_t` formulas differ — one is `duration × fps`, the other `duration × fps × temporal_patch`
— so the two agreeing would take a coincidence.

**Images have no second accounting.** Their placeholders come straight from `image_grid_thw`. That
is exactly why the image gates passed perfectly on the same boot that the first video killed.

**And the ratio is not 3.0 — it is a function of the clip's duration.** GLM-4.6V's
`DYNAMIC_FPS_THRES` picks `target_fps` from the duration, so `len(timestamps) / grid_t` moves with
it:

| Clip duration | `len(timestamps) / grid_t` | What the engine sees |
|---|---:|---|
| below 30 s | **3.0** | three times too many placeholders |
| 30–300 s | **1.0** | they agree — **by coincidence**, and this is the only band that works |
| above 300 s | **0.5 and falling** | too *few* placeholders |

Both directions are fatal, and the middle band is the trap: a stack tested only with a two-minute
clip looks correct and dies on the first short one. This is why VS4's structural equality matters
more than any single fixture passing — a fix that made one duration agree would have been another
coincidence.

### 4.3 The fix — VS4, and why the equality is structural

VS4 overrides `_get_video_second_idx_glm46v` to build the timestamps from the **pixel path's own**
sampler. The count is then derived from the frames that path actually received:
`glm_sample_frame_indices` always returns an **even-length** list, the video processor takes
`grid_t = len(idx) // 2` from the same list, and `[::2]` therefore yields exactly `grid_t`
timestamps. It is not a constant that happens to line up.

Measured on CPU: **4 timestamps → 4 × 414 = 1,656 placeholders = 1,656 encoder rows. Difference
zero.** Measured live in round 3: `prompt_tokens` **1,722** for one video (1,656 + the prompt), and
**5,991** for four images and two videos = 4 × 638 + 2 × 1,656 + the prompt — the arithmetic closes
on the wire.

### 4.4 VS6 — the failure that must not kill the engine

Independently of whether VS4 is right, *one malformed multimodal request should not take down three
nodes.* VS6 compares placeholders against the grid in `_construct_video_placeholder`, in the
**frontend**. A mismatch is an HTTP 400 and the engine keeps serving. Round 3 confirmed the
mechanism on a different route: three videos in one request returns
`400 "At most 2 video(s) may be provided in one prompt. (parameter=video)"` with
`engine_alive_after = True`.

### 4.5 VS7 — the per-request video limit that was silently clamped to 1

Round 1's boot log printed `'limit_mm_per_prompt': {'image': 4, 'video': 2}`, and it was **not being
honoured**. `Glm4vProcessingInfo.get_supported_mm_limits` (`glm4_1v.py:986`) declares
`{"image": None, "video": 1}`, `Glm5Next` does not override it, and
`multimodal/processing/context.py:420-431` takes `allowed = min(user_limit, supported_limit)`. A
two-video request would have been refused with *"At most 1 video(s)…"*.

That 1 is a conservative default rather than a real limit — videos are processed one at a time
(`glm4_1v.py:1636`), `_construct_video_placeholder` takes an `item_idx`, and `iter_mm_grid_thw` /
`get_mrope_input_positions` walk every `mm_feature` by prompt offset. VS7 lifts it to
`HAREM_VISION_VIDEO_LIMIT` (default 2). Round 3 read two videos in one request correctly and
compared them, and refused a third cleanly.

### 4.6 One more defect, survivable, and named

A clip shorter than about 0.5 s makes `extract_t = int(duration × 2.0)` collapse to **0**, the
sampler return an empty list, and the frontend raise `IndexError`. That is an **HTTP 400 with the
engine alive** — measured on a 2-frame, 0.25 s fixture in round 2. Not fatal, but every video with
`duration × fps_interval < 1` fails, and a gate description should say so. Not fixed by us
`[not tested]` as to whether upstream considers it a bug.

---

## 5. The brake, the limits, and the activation arithmetic

### 5.1 The brake was on the wrong knob for three rounds

The design specified `--mm-processor-kwargs {"max_pixels":12544000}`. It does nothing to video.
`Glm5NextVideoProcessorKwargs` does not declare `max_pixels`, and the engine says so in its own log
on every boot: *"Keyword argument `max_pixels` is not a valid argument for this processor and will be
ignored."* The video budget comes from `_pixel_budget` via `min_image_tokens` / `max_image_tokens`,
whose video ceiling is `_MAX_VIDEO_TOKENS = 30,000` tokens = 47,040,000 pixels — far above anything
we send, so **no brake was engaged at all**.

The key that works is **`max_image_tokens`**. Measured on a 1080p clip, same session, three arms
`[measured-here]`:

| `--mm-processor-kwargs` | `video_grid_thw` | vision tokens | peak encoder activation |
|---|---|---:|---:|
| none | (4, 78, 138) | 10,764 | 0.657 GiB |
| `{"max_pixels":12544000}` | (4, 78, 138) | **10,764 — unchanged** | 0.657 GiB |
| `{"max_image_tokens":8000}` | (4, 66, 118) | **7,788** | 0.475 GiB |

**Images are bit-identical across all three arms** — grid `[1,44,58]`, 638 tokens, pixel tensor
sha256 `4d0c137b…` — because the checkpoint's own `processor_config.json` already gives images 8,000
tokens (12,544,000 pixels). The brake only ever touches video.

`max_pixels` stays in the line, and deleting it would cost something. Its one remaining reader is
vLLM's own budget estimate (`_get_video_max_pixels` → `get_mm_max_tokens_per_item` → the encoder
compute budget), which §5.3 turns into a hard guarantee: with it, the per-item budget is **10,242**
tokens; without it, **32,242**, and the encoder cache sized from that eats KV pool for nothing. Two
keys, two different jobs, both needed.

**So the fix was a `.env` line, not a patch**, and the anchor that would have carried it — VS5 — was
never written. It was specified, then measured away.

### 5.2 Why a brake is needed at all

The tower does **not** chunk a video: `multimodal.py:585-634` runs every patch in one forward. The
unbraked ceiling for this model is 239,400 vision tokens ≈ **960,000 patches**, and the block's
SwiGLU intermediate buffer alone is `960,000 × 8192 × 2 B ≈ 14.6 GiB`. That is a certain OOM on a
121.6 GiB node with 50 GiB of KV pool on it. `[estimate — arithmetic; never run]`

### 5.3 The activation arithmetic, re-derived from the measurement

One vision token is 4 patches. The measured peak SwiGLU buffer is ≈ **16,384 B per patch**
(gate + up, bf16, intermediate 4096), so **peak ≈ tokens × 64 KiB**. At the brake:

| | |
|---|---|
| Per-item ceiling | **8,000 vision tokens** → tower activation peak **0.49 GiB** |
| `get_mm_max_tokens_per_item["video"]` | 8,000 + 320 × (2 + 5) + 2 = **10,242** |
| Encoder compute budget | `max(max_num_batched_tokens, max_tokens_per_mm_item)` = `max(2048, 10,242)` = **10,242** |
| `max_encoder_items_per_batch` | 10,242 // 10,242 = **1** (`multimodal/encoder_budget.py:160`) |

**That last row is the guarantee.** Even a 4-image + 2-video request hands the encoder **one item at
a time**: the peak is 0.49 GiB, and two items encoded together would be 0.98 GiB — under the ~1 GiB
ceiling the design asked for.

Measured, on rank 0, against production configuration 12 in the same session: **peak activation
1.69 GiB versus 1.66 GiB** on the sidecar-free trial boot, and **1.66 GiB — equal** on the
configuration now serving. The brake did not raise the arm's peak memory in either reading; the
0.03 GiB is a cold-boot artefact rather than the tower's price.

### 5.4 `--max-num-batched-tokens 2048` stays

There is no rule that a multimodal item must fit in a batch. `scheduler.py:238-239` seeds
`max_num_encoder_input_tokens` and `encoder_cache_size` from `max_num_batched_tokens`, and
`v1/core/encoder_cache_manager.py:313-318` then silently raises both to
`max(that, max_tokens_per_mm_item)`. The hard error exists only under `disable_chunked_mm_input=True`,
which defaults to `False`. The encoder produces an item in one step; the language model's prefill
proceeds in 2,048-token chunks; the two are independent.

The encoder cache ceiling follows the brake: ~**0.061 GiB** at 8,000 tokens per item, against
~**1.85 GiB** at the unbraked 242,000 `[estimate]`. It is allocated on demand and is *not* reserved
during the profile run, so the KV pool is sized without knowing about it — which is why the brake
comes first and `--skip-mm-profiling` stays on.

### 5.5 A measurement-discipline note: images have their own prefix cache

Encoder outputs are cached separately from the KV prefix cache, keyed by `mm_hash`
(`gpu_model_runner.py:3062, 3285`). **The same image sent twice does not run the encoder the second
time.** This is the multimodal twin of the prefix-cache artefact in
[09](09-measurement-protocol.md): discard the first round of any image or video timing, or you are
comparing an encoder run against a cache hit.

---

## 6. The ten gates, with the model's own answers

Round 3, 7 September 2026, image `exl3-zeus:754421f`, full-scope EXL3 at 4.05 bpw, TP=3 + expert
parallel, `gpu-memory-utilization` 0.88, fp8 KV and fp8 draft cache, DFlash2 k=7, `--block-size 256`,
`HAREM_SW_BLOCK_SIZE=256`, `--max-num-batched-tokens 2048`, `--max-num-seqs 8`,
`NCCL_MAX_NCHANNELS=8`, temperature 0, reasoning effort `low`, no fast-load sidecar
`[measured-here]`. The full table with the promotion boot beside it is
[`results/gates/vision-gates-tp3.md`](../results/gates/vision-gates-tp3.md).

| Gate | What it asks | Result | The model's answer / the evidence |
|---|---|---|---|
| K0 | boot evidence + KV pool | **PASS** | nine required lines; `GPU KV cache size: 6,994,490 tokens` |
| K1a | text correctness probe | **PASS** | **10/10** (9/9 client-visible, empty `content` **0**) |
| K1b | text code exam | **PASS** | **12/12** |
| K2 | four single images | **PASS ×4** | "Red circle" · "Blue square" · "Green triangle" · "ELEPHANT" |
| K3 | four images in one request | **PASS** | "red circle blue square green triangle ELEPHANT" — all four, **in order**; `prompt_tokens` 2,612 |
| K4 | one video | **PASS** | "A red circular shape (two overlapping red circles) moves from left to right across the frame." — colour, shape and direction all correct; `prompt_tokens` **1,722** |
| K4b | two videos in one request | **PASS** | "In the first video, a red circular shape moves steadily to the right across the screen. In the second video, a blue rectangular shape moves steadily downward. The two differ in the object's colour an…" — both, in order; `prompt_tokens` **3,415** |
| K5 | four images + two videos | **PASS ×4** | "Images: 1. Red circle 2. Blue square 3. Green triangle 4. The word \"ELEPHANT\" Videos: 1. Two overlapping red circles moving right 2. A blue rectangle moving down…" — all six; `prompt_tokens` **5,991** |
| K6 | a third video must be refused | **PASS** | HTTP **400**, `"At most 2 video(s) may be provided in one prompt. (parameter=video)"`, `engine_alive_after=True` |
| K7a | KV pool inside its band | **PASS** | 6,994,490 / 7,024,793 = **99.6 %** (floor 94 %) |
| K7b | one 1M-token text request | **PASS** | `prompt_tokens` **999,052**, needle found, answer **"ZEPHYR-4417"** |
| K8 | C1 and C8, text only | **PASS** | C1 median **68.8** vs 69.90 = **−1.6 %** (band ±4 %); C8 median **195.8** vs 195.78 = **+0.0 %** (band ±3 %) |

**K1 first, and text-only, on purpose.** "We fitted the camera and broke the speech" is the failure
this arm most had to disprove, and it is the one gate whose failure would have closed the arm
outright regardless of any image result.

**The two answers that carry the most information.** K4b describes two videos separately and states
their *difference* — red circle rightward, blue square downward — which is visible proof that VS4's
time-axis accounting and VS7's limit are both working. K3 reads text out of an image ("ELEPHANT"),
which proves the tower is loaded with the right tensors and bound to the language model correctly,
not merely constructed.

**A quiet risk that stayed quiet.** `iter_mm_grid_thw` (`glm4_1v.py:2236`) walks a video frame by
frame when `len(embed_ranges) == t` and treats it as one contiguous 3-D block otherwise, in which
case mrope positions shift — a silent quality loss, not a crash. After VS4 the equality holds, and
the answers get every video's temporal direction right. Consistent with the reading, and not a
separate measurement `[not tested]`.

---

## 7. What it cost: KV, speed, TTFT, acceptance

Against production configuration 12, same session, same settings `[measured-here]`. Two rows of
evidence: round 3 without a fast-load sidecar, and the promotion boot with one.

| | Configuration 12 | Round 3 (no sidecar) | Promotion (sidecar) | Band |
|---|---:|---:|---:|---|
| KV pool | 7,024,793 / 7,126,721 | 6,994,490 (**99.6 %**) | **7,143,250** (**100.2 %**) | floor 94 % |
| C1 aggregate tok/s | 69.90 | 68.8 (**−1.6 %**) | **70.0** (**+0.1 %**) | ±4 % |
| C8 aggregate tok/s | 195.78 | 195.8 (**+0.0 %**) | **198.8** (**+1.5 %**) | ±3 % |
| TTFT, C1 / C8 | 0.28 / 0.83 s | 0.28 / 0.82 s | 0.246–0.285 / 0.819–0.826 s | — |
| DFlash2 acceptance | ~62 % | 62.5–63.1 % (C8) | 60.4–63.9 % | — |
| Peak activation, rank 0 | 1.66 GiB | 1.69 GiB | 1.69 GiB, **1.66** after the reboot | — |
| CUDA-graph pool | 0.0 GiB | 0.0 GiB | 0.0 GiB | same regime |
| Indexer workspace | 513 MB | 513 MB | 513 MB | unchanged |
| Whole-cluster reboot → `/health` 200 | 311 s | — | **318 s** | trials span 311–315 s |

**The tower is not on the decode path, and the measurement agrees with the theory for once.** C8
does not move at all; C1's −1.6 % on round 3 is inside a band that the promotion boot then crossed
in the other direction at +0.1 %.

**The KV pool did not pay either.** Round 3's 99.6 % was the cost of booting *without* the fast-load
sidecar, not the cost of the tower: with the sidecar the pool comes back at 100.2 % of configuration
12's. The tower's 0.416 GiB per rank is real and lands in the weights, not in the pool.

**Draft speculation keeps working with images and video, which the code allowed but nothing had
shown.** The drafter's `supports_mm_inputs = False`
(`v1/worker/gpu/spec_decode/dflash/speculator.py:42-43`) means it never sees pixels or vision
embeddings; multimodal context reaches it only through the target's hidden states, and the target
has already run its full forward including the tower. Acceptance is unchanged, measured.

**One landmine, disarmed by a setting we already had.** `v1/spec_decode/llm_base_proposer.py:1372-1406`
copies `image_token_index` into the drafter's config for multimodal targets, from a 14-model list
that does not include GLM-5.3-Flash — and `Glm5NextConfig` defines only `image_token_id`. On the V1
model runner, turning the tower on would raise `AttributeError` during model loading. We run V2
(forced for DFlash2 drafts by `config/vllm.py:629-658`), where that code does not exist. **Do not set
`VLLM_USE_V2_MODEL_RUNNER=0` with the tower on.**

---

## 8. The promotion, and the boot-log line that goes missing

Promoted 7 September 2026 as **production configuration 13** `[measured-here]`.

**The fast-load sidecar carries the tower.** This was checked rather than assumed:
`harem_fastload._entries()` walks `named_parameters(remove_duplicate=False)` plus `named_buffers`
with no prefix filter, so `visual.*` enters the dump automatically. Measured against the text-only
sidecar:

| Sidecar | Names | Own stored bytes | `visual.*` names |
|---|---:|---:|---:|
| configuration 12, text only | 5,858 | 50.24 GiB | **0** |
| configuration 13, with the tower | **6,454** | **50.66 GiB** | **596** |

**+596 names, +0.42 GiB** — the tower, exactly. It is not re-read from the checkpoint on every boot.
The drafter sidecar is 94 names / 2.04 GiB in both. On disk: **53 GB per node**, `/var/tmp` had
599 GB free on each.

**The dump boot cost one boot and a new directory name.** The sidecar's identity is an exact
tensor-name-set equality (`harem_fastload._restore`, `want != have` → `RuntimeError`), so the
configuration-12 sidecar is **refused loudly** by a tower-carrying engine — the correct behaviour,
and the reason a new `FASTLOAD_DIR` is mandatory rather than optional: reusing the name would have
overwritten the one-`cp` rollback. Dump boot 06:21 → 06:27, main model load 165–172 s per rank,
sidecar write 86–95 s per rank at 570–631 MB/s.

**Load boot: 247 s** to `/health` 200, against 454 s for the dump boot, 365 s for round 3's
sidecar-free boot and ~251 s for configuration 12. `restored 6454 tensors, 50.66 GiB from 22 shards
in 55.8 s (974 MB/s)`.

### 8.1 `HAREM-VISION: tower loaded` does not appear on a fast-load boot

Measured, and it is a mechanism rather than a defect: VS3's audit lives inside the tower's
`load_weights`, and `harem_fastload` in load mode **skips `load_weights` entirely** — tensors are
copied from the sidecar. Round 3 printed the line because that boot was cold.

**We did not "fix" the patch to restore the line, and the reason is a house rule.**
`patch-vision-tp3.py`'s hash is part of the fast-load manifest identity, so editing it invalidates
the sidecar just dumped **and** puts a different tree into production from the one the gates passed.
The bytes that were measured are the bytes that ship.

The replacement evidence chain is not weaker, and it runs on every boot:

| # | Evidence | What it proves |
|---|---|---|
| 1 | `preflight-fastload: OK ... dir=...vision...` | the right sidecar |
| 2 | `restored 6454 tensors` (configuration 12: 5,858) | the tower is in it — 596 names |
| 3 | `Using AttentionBackendEnum.FLASH_ATTN for MMEncoderAttention` | the encoder backend |
| 4 | `'mm_encoder_tp_mode': 'data'` | the tower is replicated per rank |
| 5 | `harem_fastload._restore`'s exact name-set equality | **a BF16-fallback tower could not have booted at all** |

(5) is the load-bearing one. Read out of the sidecar's own MANIFEST: the `visual.*` names are 99
`.trellis` plus 99 `.mul1`/`.suh`/`.svh`/`.bias`, 100 `.weight` (all *norm*), one `cos_sin_cache`,
and **no `attn.qkv.weight`**. The "99 EXL3 linears, 0 unquantized" invariant is written into the
sidecar itself and re-checked on every load. The K0 gate was rewritten against this chain and
**counter-tested**: five corrupted-log scenarios, five failures — restore count 5,858, missing
preflight line, missing encoder-backend line, encoder not data-parallel, preflight pointing at the
configuration-12 sidecar. The real log passes. A gate that cannot fail is not a gate.

### 8.2 Worst-case stress, and the memory decision it settled

The acceptance run was deliberately adversarial on rank 0: a 1M-token request, a C8 speed round, and
then **a 4-image + 2-video request sent while the C8 round was running**.

| Window | Duration | MemAvailable min | swap-in | swap-out |
|---|---:|---:|---:|---:|
| 1M-token request | 677 s | **1.47 GiB** | 67.0 MiB | 281.6 MiB |
| C1 + C8 speed | 220 s | 1.70 GiB | 5.1 MiB | **0** |
| C8 + (4 images + 2 videos) concurrently | 60 s | 1.78 GiB | 0.1 MiB | **0** |

No error, no OOM, no engine death (`EngineDeadError|Traceback|CUDA out of memory` count **0** across
the whole boot). Swap use rose from 1.31 to 1.57 GiB during the 1M-token window and **stayed there**;
the two later windows paged out nothing. The other two nodes swapped **zero** in every window and
held 2.8 GiB+ available.

`gpu-memory-utilization` stays at **0.88**. A step down to 0.87 was prepared and never run: it would
have cost about 1.2 GiB of KV pool to buy headroom against a cost that measurement then showed to be
absent.

**The measured price of the worst case, which has no reference value.** In the concurrent window the
text arm ran at **173.53 tok/s** against 198.8 alone — **−12.7 %** — with TTFT 0.868 s against 0.823.
The multimodal request answered correctly in 14.42 s. This is contention, not a defect:
`--max-num-seqs 8` means a ninth request queues, and a 5,991-token prefill takes slots from the
others. **Not attributed:** how much of that −12.7 % is *multimodal* rather than simply "a ninth
request with a long prefill" — a text request of the same prefill length was not run `[not tested]`.
Configuration 12 cannot sit this exam at all; it refuses the request.

### 8.3 The environment switch, and the unit

The production environment file on each node was derived from its own copy — never copied between
nodes — after the previous one was kept as a named backup. **The promotion is exactly six lines**,
and the claim was tested rather than asserted: a diff against the backup, ignoring comments and blank
lines, reported those six and **zero unexpected changes on all three nodes**.

```
LANGUAGE_MODEL_ONLY   1 -> 0
TP3_DIR               -> the vision patch tree
OVERLAY_DIR           -> the same overlay under that tree
FASTLOAD_DIR          -> the vision sidecar
EXTRA_ENV            += HAREM_VISION=1  CUDA_EXL3_PACKED_MAPPING=...
EXTRA_ARGS           += --mm-encoder-tp-mode data  --mm-processor-kwargs ...
                        --limit-mm-per-prompt {"image":4,"video":2}  --mm-processor-cache-gb 0
```

**`GPU_MEMORY_UTILIZATION` and `FASTLOAD_MODE` are not in that list** — both are identical in the two
files (0.88 and `load`). The memory fraction did not move.

`harem-exl3.service` needed **no change and was not reloaded**: its `ExecStart` points at the
launcher, `start-tp3.sh` is byte-identical in all three trees, and `TP3_DIR` comes from the
environment file.

**Boot under the unit: 244 s** to `/health` 200, started worker-2 → worker-1 → head; all three units
`active + enabled`; KV pool **7,033,057** (98.7 % of configuration 12, inside its 6.62–7.46 M band);
K0, K1a and K1b **pass / 10-10 / 12-12**; the four single images correct; one video correct — *"A red
circle moves from left to right across the frame."*

### 8.4 The whole-cluster reboot

All three nodes rebooted together, autostart unit enabled: **`/health` 200 at 318 s** by the wall
clock from the `reboot` command, against configuration 12's **311 s** — **+7 s**, which is inside the
spread of the three trials this stack has recorded (315 / 312 / 311 s,
[`results/boot/boot-ledger.md`](../results/boot/boot-ledger.md)). KV pool after the reboot
**7,016,528**, 98.5 % of configuration 12 and inside its own boot-to-boot range of
7,024,793–7,126,721. Post-reboot gates 5/5, text 10/10 and 12/12.

Memory profile on rank 0 at the boot now serving: consumed **55.12 GiB** (configuration 12: 55.28),
peak activation **1.66 GiB** — the *same* as configuration 12, not the 1.69 GiB of the sidecar-free
round-3 boot — CUDA-graph pool 0.0 GiB, indexer workspace 513 MB, `cudagraph_mode=NONE`. The tower's
0.03 GiB of extra peak activation is a cold-boot reading, and it does not survive into the
configuration that ships `[measured-here]`.

Disk after promotion: **546–547 GB free per node** with both sidecars present. The text-only sidecar
was **not deleted**, so the rollback is one environment file and no dump boot.

---

## 9. The host-memory finding, and the flag that fixed most of it

The tower's cost on the GPU is 0.416 GiB per rank and 0.03 GiB of peak activation. Its cost in
**host** memory, on the one rank that runs the API server, was larger and took three rounds to
locate.

`mm_processor_cache_gb` defaults to **4 GiB of host memory** in this image
(`config/multimodal.py:152`), charged as `mm_processor_cache_gb × (api_server_count +
data_parallel_size)` and living only where the API server lives. It is a **host-side** cache and the
KV pool calculation never sees it.

Idle MemAvailable on rank 0, three rounds, everything else equal:

| Cache setting | Idle MemAvailable, rank 0 |
|---|---:|
| default 4 GiB | 1.10 GiB |
| `--mm-processor-cache-gb 1` | 1.84 GiB |
| **`--mm-processor-cache-gb 0`** | **3.42 GiB** |

**And what turning it off costs:** a request that sends the same image twice runs the tower twice.
That is the whole price, it is small at four images per request, and it buys 2.3 GiB of host memory
on the rank everything else on the node has to share.

**And it was not the whole story.** With the cache off, rank 0 still paged out 412.6 MiB during the
video gates and 773.3 MiB during the 1M-token request, where configuration 12 on the same machine
pages nothing. The rest is the container's own resident set: **7.60 GiB idle / 8.07 GiB under load
on rank 0, against 6.75/6.85 and 6.54/6.63 on the other two** — about **1.2–1.4 GiB**, the API
server plus the multimodal frontend. That figure could not be taken in rounds 1 and 2 at all: the
arm crashed and took the container with it.

**The judgement, and whose it is.** The house rule had been "swap traffic under load must be
approximately zero", which would have forced 0.88 → 0.87. The owner overruled it: swap may be used
pragmatically as long as it does not slow the model, produce errors, or leave the safe region in the
worst realistic case — **no KV pool is given up before a real harm is measured.** §8.2 is that
measurement, and it found no harm. The number is written down either way.

---

## 10. What is not tested

- **Long videos at the brake.** Everything measured used four-second fixtures and one analytic 1080p
  case. `max_image_tokens = 8000` puts the ceiling at roughly 44 frames at 512×512 or 6 frames at
  1080p; a clip at the ceiling has never been served `[not tested]`.
- **The brake ladder.** 8,000 → 16,000 → 32,000 tokens per item, with K4/K4b/K5/K7 repeated at each
  rung, is the multimodal twin of the memory ladder. The peak activation at 16,000 would be 0.98 GiB
  `[estimate]`. Not run.
- **Image quality against BF16.** Nothing compares this 6-bit EXL3 tower's image understanding
  against a BF16 tower on the same prompts. The gates prove the tower is loaded correctly and answers
  correctly on synthetic fixtures — a colour, a shape, a direction, a word. They are not a
  vision benchmark `[not tested]`.
- **Any published multimodal benchmark.** No MMMU, no DocVQA, no ChartQA `[not tested]`.
- **Encoder CUDA graphs** (`--compilation-config cudagraph_mm_encoder=true`). Not enabled, not
  measured; it is a separate A/B `[not tested]`.
- **Moving VS3's audit past `process_weights_after_loading`**, so the tower line prints on a
  fast-load boot as well (§8.1). It is a small change and it costs a fresh sidecar dump, which is why
  it was not made inside the promotion window. Not written `[not tested]`.
- **Acceptance and TTFT on image-heavy sustained load.** Acceptance was measured on text sweeps
  while the tower was live, and on individual multimodal requests. A sustained multimodal load has
  not been swept `[not tested]`.
- **Where the −12.7 % of §8.2 comes from** — contention against multimodality, unattributed
  `[not tested]`.
- **TP=2 and TP=4** — §11.

---

## 11. Two ranks, and four

**At two ranks the tower divides and needs no padding and no data-parallel flag.** Every width in
§1.1's table is a clean multiple of 2 and of 128 per rank: heads 8, `attn.proj` 512 = 4 × 128, MLP
2048 = 16 × 128, merger 2048 = 16 × 128, merger context 5120 = 40 × 128.

**But the loader half is not about the rank count.** `quant_config=None` (§3.1) and the packed-mapping
union (§3.2) are properties of the *checkpoint*, so which of the two-node candidates you run decides
what you need:

| Checkpoint at TP=2 | What is needed |
|---|---|
| `brandonmusic/GLM-5.3-Flash-tr3-4bpw` (routed experts only) — the tower is dense BF16, 347 tensors, 1.05 GiB | `LANGUAGE_MODEL_ONLY=0` and the per-request limits. Nothing else `[not tested]` |
| `turboderp/GLM-5.3-Flash-exl3` @4.05bpw (full scope) — the two-node production candidate, [15](15-tp2-track.md) | The **same** VS1/VS2/VS3 and the same `CUDA_EXL3_PACKED_MAPPING`, plus VS4/VS6/VS7 for video. `--mm-encoder-tp-mode data` is **not** needed `[not tested]` |

The video half — VS4, VS6, VS7 — is needed at **any** parallel size, because the sampler mismatch has
nothing to do with sharding. Neither row above has been run; both are code readings against measured
checkpoint facts `[not tested]`. If you run one, [HELP-WANTED](../HELP-WANTED.md) says what we would
want reported.

**Four ranks:** nothing measured. 16/4 = 4 heads per rank divides, and the MLP and merger widths
divide too, so the tower may need only the loader half — but `attn.proj` at 256 per rank is 2 × 128
and the merger's 2560 is 20 × 128, so the 128-block rule survives on paper. Untested in every
respect `[not tested]`.

---

## 12. Upstream status

**The video sampler bug is filed, with a fix.**

| | |
|---|---|
| Issue | [vllm#55644](https://github.com/vllm-project/vllm/issues/55644) — the two-sampler mismatch, the chain with file and line, the CPU reproduction, and the duration table of §4.2 |
| Pull request | [vllm#55647](https://github.com/vllm-project/vllm/pull/55647) — the fix plus a unit test, DCO-signed, **awaiting a maintainer's ready label** |

It is not ours: `Glm5NextProcessingInfo` fails to override `_get_video_second_idx_glm46v`, in
upstream vLLM, on the stock image, and it kills the engine core on **every** GLM-5-Next video request
regardless of tensor parallel size, quantization, expert parallelism or anything else this repository
does. Our arm reproduced it three times and then measured it on a CPU in under a second.

The report carries the structural argument as well as the trace: deriving the timestamps from the
pixel path's own sampler is correct rather than merely sufficient (§4.3), and the duration table is
what shows why — a fix that made one clip length agree would only have moved the coincidence.

Two smaller items ride along in the issue: the `min(user, supported)` clamp that makes
`--limit-mm-per-prompt {"video": 2}` silently mean 1 (§4.5), and the zero-frame sampler on clips
under half a second (§4.6). A third needs no report because it is one line —
`is_vit_use_data_parallel(num_heads=...)` never being passed its argument (§2a).

Two other observations that are ours, not upstream's, and are recorded rather than filed: the fused
`attn.qkv.weight` residue in the published EXL3 checkpoint (144.14 MiB of tensors nothing reads,
§3.1), and the absence of the tower from that checkpoint's `tensor_storage` manifest (§3.2) — which
`cuda-exl3` already recovers from, so it costs a reader confusion rather than a failure.

---

## 13. The lesson this arm added to the protocol

Three boots died before the tower ever answered a question, and **all three were our own tooling**,
not the checkpoint:

1. **An anchor written against the stock file.** The prelude runs `patch-vllm-tp3.py` long before the
   vision patch, and its edit 4b rewrites the first two lines of the tower construction block. VS1
   matched zero times and stopped all three ranks with exit 21 — fail-closed working exactly as
   designed. The model-free gate said PASS, because it chained full-scope → vision and **never
   reproduced the boot's patch order**. `verify-cpu.sh` now runs the full prelude order. *A gate that
   does not reproduce the boot is not a gate.*
2. **The same gate returned 0 through a pipe.** Every check ends in `| tail -1` and the script had no
   `pipefail`, so a run printing `[vision] FAIL` exited 0. Added, and then confirmed by watching a
   failing gate actually fail.
3. **A header-derived count asserted as an invariant.** 48 dead tensors was true of the safetensors
   headers and false of the load (§3.3). It killed a correct boot. Counts are reported; invariants
   are asserted.

And the fourth, which is the expensive one: **the install-day gates only looked at weight loading.**
The tower loaded perfectly — 99/99 EXL3, 0 BF16, the right backend, data-parallel confirmed — and the
engine still died on the first video, because nobody was checking **geometry**. That gate now exists
([`check-video-geometry.py`](../tracks/tp3/patches/vision/check-video-geometry.py)), it is
model-free, it takes under a second, and it would have caught the fault before the first of the three
engine windows this arm spent finding it.
