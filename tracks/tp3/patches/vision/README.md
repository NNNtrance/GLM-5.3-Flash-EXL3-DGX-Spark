# `patches/vision` — the vision tower at three ranks

**In production as of 7 September 2026 — production configuration 13.** Six anchors and four
environment settings that turn the GLM-5.3-Flash vision tower back on at TP=3, on the **full-scope**
EXL3 checkpoint, for **4 images + 2 videos per request**. Measured cost against configuration 12 in
the same session: KV pool inside configuration 12's own boot-to-boot spread, C1 **+0.1 %**,
C8 **+1.5 %**, TTFT and draft acceptance unchanged, ten gates out of ten, and a whole-cluster reboot
at 318 s against 311 s `[measured-here]`.

**This directory is the whole change.** Nothing under [`patches/`](../) moves: with `HAREM_VISION`
unset the tree is production configuration 12 byte for byte, because every anchor reads the knob at
run time.

The full account — why the tower was off, the two loader blockers, the video sampler bug and its
arithmetic, the brake, the gates, the cost table — is [docs/18](../../../../docs/18-vision-at-three-ranks.md).
This page is the directory.

---

## What is here

| File | What it does |
|---|---|
| [`patch-vision-tp3.py`](patch-vision-tp3.py) | The patch. Six exact-match anchors across two files, all gated on `HAREM_VISION=1`. `--check` reports anchor counts and writes nothing |
| [`check-vision-mapping.py`](check-vision-mapping.py) | Model-free gate 1: with the class mapping and `CUDA_EXL3_PACKED_MAPPING` merged, do all 99 vision linears resolve to EXL3 modules? Reads `quantization_config.json` and safetensors headers only |
| [`check-vision-names.py`](check-vision-names.py) | Model-free gate 2: does every live vision tensor name in the checkpoint land on a parameter the built tower actually has? |
| [`check-video-geometry.py`](check-video-geometry.py) | Model-free gate 3, added in round 3: for each fixture video **and** an analytic 1080p case, does the placeholder count equal the encoder row count? This is the gate that rounds 1 and 2 did not have |
| [`prelude-vision-hook.sh`](prelude-vision-hook.sh) | The block to paste into [`tp3full-prelude.sh`](../tp3full-prelude.sh), after the full-scope block. Runs the patch, then the three gates, all fail-closed |
| [`verify-cpu.sh`](verify-cpu.sh) | Runs the **whole prelude order** plus the three gates in a throwaway CPU container — no `--gpus`, no engine, safe while production is serving. Must print five PASS/applied lines |

The fixture videos the geometry gate reads are not in this repository (they are two four-second
clips of a moving shape). The gate skips with a printed line when `$TP3_DIR/fixtures` is absent and
still runs its analytic 1080p case; make your own two clips, or run the gate with fixtures of your
own. `[measured-here]` for the clips we used.

---

## The six anchors

Naming note: **VS5 does not exist.** It was specified — patch the video processor to accept a brake
keyword — and then measured away: the brake reaches the processor already, under a different key
(below). A number that was never needed is left as a hole rather than renumbered, so the boot log
line `[vision] applied: VS1 VS2 VS3 VS4 VS6 VS7` means what it says.

### The loader half — VS1, VS2, VS3

| # | File | What it changes |
|---|---|---|
| **VS1** | `vllm/models/glm5next/nvidia/model.py` | The tower is built with `quant_config=None`, unconditionally. Pass the checkpoint's real EXL3 config through instead — but only when the knob is set **and** the config is EXL3, so every other checkpoint keeps upstream's `None` |
| **VS2** | `vllm/models/glm5next/nvidia/multimodal.py` | `Glm5NextVisionTransformer.hf_to_vllm_mapper` maps `.attn.q.` → (`.attn.qkv.`, "q"), the GLM-OCR spelling. This checkpoint writes `.attn.q_proj.trellis`. Add the three `_proj` spellings |
| **VS3** | same file | Drop the dead pre-quantization fused `attn.qkv.{weight,bias}` tensors the checkpoint still carries, then **audit** the built tower: 99 EXL3 linears, 0 that fell back to bf16, no fused qkv module left holding a dense `.weight` |

VS1's anchor is written against the **post-`patch-vllm-tp3.py`** text, not the stock file. That
matters and it is a scar: the prelude runs `patch-vllm-tp3.py` long before this script, and its
edit 4b rewrites the first two lines of the tower construction block, so a stock-file anchor matched
zero times and stopped all three ranks with exit 21. Correct behaviour, wrong gate — the model-free
gate had chained full-scope → vision and never saw that edit. [`verify-cpu.sh`](verify-cpu.sh) now
runs the full prelude order for exactly this reason.

### The video half — VS4, VS6, VS7

| # | File | What it changes |
|---|---|---|
| **VS4** | `.../multimodal.py`, `Glm5NextProcessingInfo._get_video_second_idx_glm46v` | Derive the placeholder timestamps from the **pixel path's own** frame sampler. Upstream inherits GLM-4.6V's, which samples the same clip differently; 12 timestamps against a grid of 4 meant 4,968 placeholders for 1,656 encoder rows and a dead engine core on all three ranks |
| **VS6** | `.../multimodal.py`, `_construct_video_placeholder` | Fail-closed: compare placeholders against the grid **in the frontend**. A future mismatch is an HTTP 400, never three dead workers |
| **VS7** | `.../multimodal.py`, `get_supported_mm_limits` | The inherited `{"video": 1}` silently clamps `--limit-mm-per-prompt {"video":2}` — `processing/context.py` takes the `min()` of the two. Nothing downstream needs the 1. Lifted to `HAREM_VISION_VIDEO_LIMIT`, default 2 |

VS4's equality is **structural, not arithmetic**: `glm_sample_frame_indices` always returns an
even-length list and the video processor derives `grid_t = len(idx) // 2` from that same list, so
taking every second timestamp gives exactly `grid_t` of them by construction. It is not a constant
that happens to match.

---

## Fail-closed rules — all four of them

1. **Every anchor must match exactly once.** Zero means the image drifted; more than one means the
   anchor is ambiguous. Either stops the rank, because a half-patched stack is the failure mode that
   serves fluent, wrong answers.
2. **A half-patched tree is refused.** If one of the two files already carries the `HAREM-VISION`
   marker and the other does not, the script refuses rather than completing the pair.
3. **The three gates stop the boot, not the request.** They run in the prelude, before a byte of
   weight is read, and each exits non-zero on failure. `HAREM_VISION_GATES=0` exists for a broken
   gate, never for a gate that is telling the truth.
4. **Counts are reported, thresholds are asserted.** The number of dead fused tensors dropped is
   printed, not checked: it was specified as 48 from the safetensors headers, and the real load
   delivers 0 or 1 of them, because `AutoWeightsLoader` ignores unexpected `.bias` suffixes before
   this filter sees them and its prefix grouping consumes the rest. Asserting 48 killed a boot whose
   tower was entirely correct. What **is** asserted is the invariant that matters: 99 EXL3 linears,
   0 unquantized, no dense fused `qkv.weight`.

---

## The environment, exactly

Six lines change against production configuration 12. Nothing else moves — same image, same
checkpoint, same memory fraction, same batching, same sampling.

```
LANGUAGE_MODEL_ONLY=0
```

```
EXTRA_ENV += HAREM_VISION=1 CUDA_EXL3_PACKED_MAPPING={"qkv_proj":["q_proj","k_proj","v_proj"]}
```

```
EXTRA_ARGS += --mm-encoder-tp-mode data --mm-processor-kwargs {"max_pixels":12544000,"max_image_tokens":8000} --limit-mm-per-prompt {"image":4,"video":2} --mm-processor-cache-gb 0
```

| Knob | Why |
|---|---|
| `LANGUAGE_MODEL_ONLY=0` | Makes the launcher pass `--skip-mm-profiling --limit-mm-per-prompt` instead of `--language-model-only`. The launcher's own limits are overridden by the `EXTRA_ARGS` copy — argparse `_StoreAction`, last one wins, and `EXTRA_ARGS` is appended after |
| `HAREM_VISION=1` | Arms all six anchors. Unset, the tree is configuration 12 byte for byte |
| `CUDA_EXL3_PACKED_MAPPING={"qkv_proj":["q_proj","k_proj","v_proj"]}` | The full-scope patch's A2 shadows the inherited Glm4v mapping, which is where `qkv_proj` came from. `cuda-exl3` merges this env value **under** the class mapping, so it can only add. **No spaces** — `EXTRA_ENV` is word-split |
| `--mm-encoder-tp-mode data` | Replicates the tower on every rank instead of slicing it. Nothing in the tower divides by three; the model declares `supports_encoder_tp_data = True` |
| `--mm-processor-kwargs {"max_pixels":12544000,"max_image_tokens":8000}` | The brake. `max_image_tokens` is the key that reaches the video processor; `max_pixels` reaches only vLLM's own budget estimate and is kept for that. **No spaces** |
| `--limit-mm-per-prompt {"image":4,"video":2}` | Per request. The `video: 2` half needs VS7 to survive the clamp. **No spaces** |
| `--mm-processor-cache-gb 0` | The multimodal processor cache defaults to **4 GiB of host memory** on the rank that runs the API server. Turning it off moved idle MemAvailable there from 1.10 GiB to 3.42 GiB |
| `HAREM_VISION_VIDEO_LIMIT` | Optional, default 2. Only read when `HAREM_VISION=1` |

The full template with every line's reasoning is
[`tracks/tp3/env.tp3-full.example`](../../env.tp3-full.example); the text-only fallback is the same
file with `LANGUAGE_MODEL_ONLY=1` and those six lines removed.

---

## What the boot log must say

Nine lines on a cold boot. On a **fast-load** boot the fifth is absent by design — `harem_fastload`
restores tensors directly and never calls `load_weights`, which is where VS3's audit lives — and the
chain that replaces it is in [docs/18](../../../../docs/18-vision-at-three-ranks.md) §8.

```
[tp3-prelude] VISION ARM: patch-vision-tp3.py sha256 65c147aee9588586
[vision] applied: VS1 VS2 VS3 VS4 VS6 VS7
[vision-map] PASS: union resolves 99/99 ... all 172 EXL3 vision modules
[vision-names] PASS: all 959 live vision tensors ...
[video-geom] PASS: 5 video geometries agree ... declared per-prompt video limit = 2
HAREM-VISION: tower loaded, data_parallel=True, vit_attn_backend=FLASH_ATTN, EXL3 linears=99, unquantized linears=0
Using AttentionBackendEnum.FLASH_ATTN for MMEncoderAttention
'limit_mm_per_prompt': {'image': 4, 'video': 2}
GPU KV cache size: 7,143,250 tokens
```

`language_model_only` must **not** appear. If it does, the launcher took the text-only branch and no
image will be accepted.

---

## What this cost

Measured against production configuration 12 in the same session, load boot, `gpu-memory-utilization`
0.88 `[measured-here]`:

| | Configuration 12 | With the tower | |
|---|---:|---:|---|
| KV pool | 7,024,793–7,126,721 across boots | **7,143,250 · 7,033,057 · 7,016,528** across three boots | inside configuration 12's own spread |
| C1 aggregate | 69.90 tok/s | **70.0** | +0.1 % (band ±4 %) |
| C8 aggregate | 195.78 tok/s | **198.8** | +1.5 % (band ±3 %) |
| Peak activation, rank 0 | 1.66 GiB | **1.66 GiB** | equal on the configuration that ships |
| Draft acceptance | ~62 % | 60.4–63.9 % | unchanged |
| Cold boot with fast-load | ~251 s | **247 s** by hand, **244 s** under the unit | equal |
| Whole-cluster reboot → `/health` 200 | 311 s | **318 s** | +7 s, inside a 311–315 s spread |
| Fast-load sidecar, per node | 50.24 GiB / 5,858 names | **50.66 GiB / 6,454 names** | +0.42 GiB, +596 names — the tower |

**And what it cost that is not in that table.** One extra 53 GB-per-node sidecar dump boot, because
the tower changes the tensor name list and the identity check is exact equality. Host memory on the
API-server rank is tighter under load than configuration 12's: the container's resident set is
1.2–1.4 GiB larger and the node pages out 282 MiB during a 1M-token request where configuration 12
pages out nothing — measured, and measured to be harmless (no error, no slowdown outside the bands,
swap use settles rather than growing). The full accounting, including what was **not** attributed,
is [docs/18](../../../../docs/18-vision-at-three-ranks.md) §9.
