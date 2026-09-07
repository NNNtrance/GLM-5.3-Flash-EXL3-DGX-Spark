#!/usr/bin/env python3
"""HAREM vision-tower patch for vLLM's glm5next at TP=3 -- env-gated on HAREM_VISION.

Turns the vision tower back on for the FULL-SCOPE EXL3 checkpoint
(`turboderp/GLM-5.3-Flash-exl3` @4.05bpw), which is what production 12 serves.

WHY A PATCH AT ALL.  `--mm-encoder-tp-mode data` clears the only thing everybody
knew about (`dist_utils.divide(16, 3)` asserts, so the tower cannot be built at
TP=3), and `LANGUAGE_MODEL_ONLY=0` makes start-tp3.sh pass the per-request
limits.  Neither of those makes the tower LOAD, because of two facts measured
from the checkpoint's safetensors headers, not guessed:

  1. The tower in this checkpoint is 6-bit EXL3, not bf16: 172 modules,
     1,007 tensors, all in shard 19, 0.557 GiB, in trellis/suh/svh/mul1 form.
     `proj`, `mlp.*` and `merger.*` have NO dense `.weight` on disk.
  2. vLLM builds the tower with `quant_config=None` unconditionally
     (model.py:1089).  The comment there is correct for the zai-org fp8
     checkpoint it was written against and wrong for ours.

So the tower builds as bf16 linears and the load dies looking for tensors that
do not exist.  Three exact-match anchors fix it:

  VS1  models/glm5next/nvidia/model.py -- the tower construction.  Pass the
       real quant config through, but only when the knob is set AND the config
       is EXL3.  Every other checkpoint keeps the upstream None.
  VS2  models/glm5next/nvidia/multimodal.py -- `Glm5NextVisionTransformer`'s
       stacked weight mapper.  Upstream maps `.attn.q.` -> (`.attn.qkv.`, "q"),
       the GLM-OCR / GLM-4V spelling.  This checkpoint writes
       `.attn.q_proj.trellis`, so nothing matches and the three EXL3 shards
       never reach the fused module.  Add the three `_proj` spellings.  The
       destination stays `.attn.qkv.`: the ATTRIBUTE is `self.qkv` whatever the
       prefix says (multimodal.py:150 only changes the prefix, and the prefix
       is what get_quant_method resolves -- the loader walks attributes).
  VS4  same file -- `Glm5NextProcessingInfo._get_video_second_idx_glm46v`.
       The VIDEO placeholder count came from GLM-4.6V's frame sampler while the
       pixels came from GLM-5-Next's; 12 timestamps against a grid of 4 meant
       4,968 placeholders for 1,656 encoder rows and a dead engine core on all
       three ranks (rounds 1 and 2).  Derive the timestamps from the pixel
       path's own sampler; the count is fixed by the frames that path receives,
       so the equality is structural.
  VS6  same file -- `Glm5NextProcessingInfo._construct_video_placeholder`.
       Fail-closed: compare placeholders against the grid IN THE FRONTEND, so a
       future mismatch is an HTTP 400 rather than three dead ranks.
  VS7  same file -- `Glm5NextProcessingInfo.get_supported_mm_limits`.  The
       inherited `{"video": 1}` silently CLAMPS `--limit-mm-per-prompt
       {"video":2}` (processing/context.py:420-431 takes the min).  Nothing
       downstream needs the 1; lift it to HAREM_VISION_VIDEO_LIMIT (default 2).

  VS3  same file -- `Glm5NextVisionTransformer.load_weights`.  The checkpoint
       also still carries the pre-quantization fused `attn.qkv.weight` /
       `.bias` (48 tensors, 144.14 MiB of dead F16).  Under EXL3 the module has
       no `.weight` parameter, so AutoWeightsLoader raises
       "There is no module or parameter named ...".  Drop them, report how many
       were dropped, and then AUDIT the built tower: 99 EXL3 linears, 0 that
       fell back to bf16, and no fused qkv module left carrying a dense
       `.weight` parameter.  The dropped COUNT is reported, not asserted --
       measured 7 September 2026 on the first real load, only 1 of the 48 header
       tensors reaches this filter, because AutoWeightsLoader already ignores
       unexpected `.bias` suffixes (models/utils.py:404) and its prefix grouping
       consumes the rest.  Asserting 48 killed a boot whose tower was correct.

WHAT IS NOT PATCHED, ON PURPOSE
  * The packed-modules mapping.  cuda_exl3 needs `qkv_proj -> q/k/v_proj` to
    resolve the fusion, and patch-fullscope-tp3.py's A2 shadows the inherited
    Glm4v mapping that carried it.  It comes back through the environment
    instead -- CUDA_EXL3_PACKED_MAPPING, merged UNDER the class mapping
    (cuda_exl3/config.py:112-125, :381-386), so it can only ADD.  No patch, and
    the union is proved model-free by check-vision-mapping.py.
  * The head-count padding (16 -> 18).  At TP=3 nothing in the tower divides by
    three -- not the heads, not the 4096 MLP, not the merger's 4096/10240 -- so
    that road is four padded shapes and a new pad audit for 0.28 GiB/rank.
    `--mm-encoder-tp-mode data` replicates the tower instead: one flag, no pad.
  * The video brake.  It is a processor argument, not code -- but NOT the one
    round 1 used.  `max_pixels` never reaches the video processor
    (Glm5NextVideoProcessorKwargs does not declare it; the engine logs
    "Keyword argument `max_pixels` is not a valid argument for this processor
    and will be ignored").  The key that does is `max_image_tokens`.  MEASURED
    on a 1080p clip: no kwarg -> 10,764 vision tokens; max_image_tokens=8000 ->
    7,788; images bit-identical either way (the checkpoint already gives images
    8,000 tokens).  So the .env carries
    --mm-processor-kwargs {"max_pixels":12544000,"max_image_tokens":8000}:
    the first key feeds vLLM's own budget estimate, the second is the brake.

ORDER.  Run AFTER patch-fullscope-tp3.py.  Fullscope's A3 anchor is the MLA
`quant_config=None,  # MLA projections are BF16 in checkpoint`; VS1's is the
vision one, a different string, and the two do not overlap.

VS1 IS ANCHORED ON THE POST-patch-vllm-tp3.py TEXT (re-anchored 7 September 2026,
first boot attempt).  The prelude runs patch-vllm-tp3.py long before this
script, and its edit 4b rewrites the first two lines of the tower construction
block:

    self.visual = Glm5NextVisionTransformer(      ->  self.visual = _harem_build_vision_tower(  # HAREM-TP3
        config.text_config,                             Glm5NextVisionTransformer,
                                                        multimodal_config,
                                                        config.text_config,

so an anchor written against the stock file matches ZERO times at boot and the
rank stops with exit 21 -- which is what happened, correctly, on all three
nodes.  The model-free gate had chained fullscope -> vision only and never saw
that edit; verify-cpu.sh now runs the full prelude order, so the gate and the
boot patch the same bytes.  The wrapper keeps `--language-model-only` working
(it returns None before the tower class is called); the kwargs, this one
included, are forwarded unchanged.

Inert unless HAREM_VISION=1.  Knob unset == upstream behaviour, byte for byte:
VS1's gate returns None, VS2's class attribute is never rebound, VS3's filter
and audit never run.  The patched code re-reads the environment at RUNTIME.

Every anchor must match EXACTLY once.  Zero means the image drifted; more than
one means the anchor is ambiguous.  Both stop the rank, because a half-patched
stack is the failure mode that serves fluent, wrong answers.

Usage:
    patch-vision-tp3.py --root /usr/local/lib/python3.12/dist-packages/vllm
    patch-vision-tp3.py --root ... --check     # report only, write nothing
"""

import argparse
import os
import py_compile
import sys

MARK = "HAREM-VISION"

MODEL_REL = ("models", "glm5next", "nvidia", "model.py")
MM_REL = ("models", "glm5next", "nvidia", "multimodal.py")


# ---------------------------------------------------------------------------
# VS1 -- model.py: give the tower the checkpoint's quant config
# ---------------------------------------------------------------------------

VS1_ANCHOR = '''        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = _harem_build_vision_tower(  # HAREM-TP3
                Glm5NextVisionTransformer,
                multimodal_config,
                config.text_config,
                config.vision_config,
                # Read eps from the VISION sub-config, not the top-level
                # `config.rms_norm_eps`: Glm5NextConfig.__getattribute__ mirrors
                # the latter onto text_config (1e-5), silently ignoring the
                # vision tower's own (1e-6) rms_norm_eps.
                norm_eps=config.vision_config.rms_norm_eps,
                # Vision tower ships BF16 weights in this fp8 checkpoint (no
                # weight_scale_inv for visual.*), so it must NOT inherit the
                # global fp8 quant_config -- doing so incorrectly quantizes
                # the tower
                # and yields NaN image features. Mirrors the MLA/KDA proj
                # pattern (quant_config=None for BF16 submodules).
                quant_config=None,
'''

VS1_REPL = '''        # HAREM-VISION VS1 (env-gated, 7 September 2026).  The upstream comment below
        # is TRUE of the zai-org fp8 checkpoint it was written against -- there
        # `visual.*` really is bf16 and inheriting the fp8 config would quantize
        # the tower into NaNs.  It is FALSE of turboderp/GLM-5.3-Flash-exl3
        # @4.05bpw, our production checkpoint: its tower is 6-bit EXL3 (172
        # modules / 1,007 tensors / shard 19 / 0.557 GiB, read off the
        # safetensors headers) and `proj`, `mlp.*` and `merger.*` have no dense
        # `.weight` on disk at all.  With quant_config=None the tower is built
        # out of bf16 linears and the load dies looking for tensors that were
        # never written.
        #
        # Pass the real config through -- but only for an EXL3 config with the
        # knob set.  Every other checkpoint keeps the upstream None, byte for
        # byte, so a patched image still serves them correctly.
        # Design and the checkpoint measurement: docs/18 section 3.
        import os as _harem_vision_os

        def _harem_vision_tower_quant(qc):
            if qc is None or _harem_vision_os.environ.get("HAREM_VISION") != "1":
                return None
            get_name = getattr(qc, "get_name", None)
            return qc if callable(get_name) and get_name() == "exl3" else None

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = _harem_build_vision_tower(  # HAREM-TP3
                Glm5NextVisionTransformer,
                multimodal_config,
                config.text_config,
                config.vision_config,
                # Read eps from the VISION sub-config, not the top-level
                # `config.rms_norm_eps`: Glm5NextConfig.__getattribute__ mirrors
                # the latter onto text_config (1e-5), silently ignoring the
                # vision tower's own (1e-6) rms_norm_eps.
                norm_eps=config.vision_config.rms_norm_eps,
                # Vision tower ships BF16 weights in this fp8 checkpoint (no
                # weight_scale_inv for visual.*), so it must NOT inherit the
                # global fp8 quant_config -- doing so incorrectly quantizes
                # the tower
                # and yields NaN image features. Mirrors the MLA/KDA proj
                # pattern (quant_config=None for BF16 submodules).
                quant_config=_harem_vision_tower_quant(vllm_config.quant_config),
'''


# ---------------------------------------------------------------------------
# VS2 -- multimodal.py: helpers + the stacked weight mapper
# ---------------------------------------------------------------------------

VS2_ANCHOR = '''class Glm5NextVisionTransformer(nn.Module):
    # Stacked-weight remap for the GLM-OCR/GLM-4V vision checkpoint layout.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_stacked={
            ".attn.q.": (".attn.qkv.", "q"),
            ".attn.k.": (".attn.qkv.", "k"),
            ".attn.v.": (".attn.qkv.", "v"),
            ".gate_proj": (".gate_up_proj", 0),
            ".up_proj": (".gate_up_proj", 1),
        }
    )
'''

VS2_REPL = '''# ---------------------------------------------------------------------------
# HAREM-VISION (7 September 2026) -- env-gated support for a FULL-SCOPE EXL3 vision
# tower.  HAREM_VISION unset == upstream behaviour, byte for byte.
# Design and the checkpoint measurement: docs/18 section 3.
# ---------------------------------------------------------------------------
import os as _harem_vision_os


def _harem_vision_env() -> bool:
    """The one knob.  Read at import time for the class attribute below and at
    call time in load_weights -- both see the same container environment."""
    return _harem_vision_os.environ.get("HAREM_VISION") == "1"


# The pre-quantization fused attention weight that turboderp left in the
# checkpoint beside the EXL3 q/k/v shards: 24 x (weight [3072,1024] + bias
# [3072]) = 48 tensors, 144.14 MiB of F16 nothing reads.  Under EXL3 the fused
# module has no `.weight` parameter at all, so AutoWeightsLoader would raise
# "There is no module or parameter named ...".  Both the weight and the bias go:
# the EXL3 path supplies both from the q/k/v shards.
_HAREM_VISION_DEAD_FUSED = (".attn.qkv.weight", ".attn.qkv.bias")


class Glm5NextVisionTransformer(nn.Module):
    # Stacked-weight remap for the GLM-OCR/GLM-4V vision checkpoint layout.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_stacked={
            ".attn.q.": (".attn.qkv.", "q"),
            ".attn.k.": (".attn.qkv.", "k"),
            ".attn.v.": (".attn.qkv.", "v"),
            ".gate_proj": (".gate_up_proj", 0),
            ".up_proj": (".gate_up_proj", 1),
        }
    )
    # HAREM-VISION VS2 (env-gated).  Upstream's three attention entries are the
    # GLM-OCR / GLM-4V spelling (`.attn.q.weight`).  turboderp's EXL3 checkpoint
    # writes `.attn.q_proj.trellis`, which matches none of them, so the shards
    # would never reach the fused module and the load would stop on a missing
    # weight.  Add the `_proj` spellings; keep the originals so an OCR-layout
    # checkpoint still loads.
    #
    # The destination stays `.attn.qkv.` even though the EXL3 prefix is
    # `...attn.qkv_proj`: multimodal.py's QKVParallelLinear takes
    # `prefix=f"{prefix}.qkv_proj" if quant_config else f"{prefix}.qkv"` but the
    # ATTRIBUTE is `self.qkv` either way.  The prefix is what get_quant_method
    # resolves; AutoWeightsLoader walks attributes.  Mixing the two loads
    # nothing and reports a missing module.
    #
    # Substring order is load-bearing: WeightsMapper mutates the key inside the
    # loop (utils.py:113-117), so an entry that matches the REWRITTEN name would
    # fire a second time.  `.attn.q_proj.` -> `.attn.qkv.` cannot re-match any
    # entry, and `.up_proj` cannot match `.gate_up_proj` (the character before
    # `up_proj` is `_`, not `.`) -- which is why upstream could write it that
    # way at all.  check-vision-names.py replays all 1,007 real checkpoint
    # names through this mapper and asserts every destination.
    if _harem_vision_env():
        hf_to_vllm_mapper = WeightsMapper(
            orig_to_new_stacked={
                ".attn.q.": (".attn.qkv.", "q"),
                ".attn.k.": (".attn.qkv.", "k"),
                ".attn.v.": (".attn.qkv.", "v"),
                ".gate_proj": (".gate_up_proj", 0),
                ".up_proj": (".gate_up_proj", 1),
                ".attn.q_proj.": (".attn.qkv.", "q"),
                ".attn.k_proj.": (".attn.qkv.", "k"),
                ".attn.v_proj.": (".attn.qkv.", "v"),
            }
        )
'''


# ---------------------------------------------------------------------------
# VS3 -- multimodal.py: drop the dead fused tensors, then audit the tower
# ---------------------------------------------------------------------------

VS3_ANCHOR = '''    def load_weights(self, weights) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
'''

VS3_REPL = '''    def _harem_vision_audit(self, dropped: int, loaded) -> None:
        """HAREM-VISION VS3 audit.  Say out loud what the tower ended up as.

        `Exl3Config.get_quant_method` FAILS OPEN (cuda_exl3/config.py:458-470):
        a prefix that does not resolve quietly becomes an UnquantizedLinearMethod
        and only dies later, on a missing weight, with a misleading message.  A
        tower that is half EXL3 and half bf16 is exactly the shape of bug that
        loads without error and answers wrongly, so count both sides and refuse
        to serve a mixed one.
        """
        exl3 = bf16 = 0
        for mod in self.modules():
            qm = getattr(mod, "quant_method", None)
            if qm is None:
                continue
            name = type(qm).__name__
            if name == "Exl3LinearMethod":
                exl3 += 1
            elif "Linear" in name:
                bf16 += 1
        dp = getattr(self, "tp_size", None) == 1
        # The encoder's attention backend is chosen by get_vit_attn_backend
        # (vision.py:101) from mm_encoder_attn_backend -- a DIFFERENT path from
        # the decoder's --attention-backend, so our CUSTOM MLA backend cannot
        # leak into it.  This image logs that choice NOWHERE, so print it here:
        # a silent fall back to TORCH_SDPA is a real risk and the arm's kill
        # criteria name it.
        backend = getattr(self, "attn_backend", None)
        msg = (
            f"HAREM-VISION: tower loaded, data_parallel={dp}, "
            f"vit_attn_backend={getattr(backend, 'name', backend)}, "
            f"EXL3 linears={exl3}, unquantized linears={bf16}, "
            f"dead fused qkv tensors dropped={dropped}"
        )
        try:
            from vllm.logger import init_logger

            init_logger(__name__).info(msg)
        except Exception:  # pragma: no cover - logging must never stop a boot
            print(msg, flush=True)
        depth = len(self.blocks)
        want_exl3 = depth * 4 + 3      # per block: qkv, proj, mlp gate_up, mlp down
        if exl3 != want_exl3 or bf16 != 0:
            raise RuntimeError(
                f"HAREM-VISION: expected {want_exl3} EXL3 linears and 0 "
                f"unquantized ones in the tower, got {exl3} and {bf16}. "
                "Some vision module did not resolve to its checkpoint tensors: "
                "check CUDA_EXL3_PACKED_MAPPING (the qkv_proj entry) and run "
                "check-vision-mapping.py."
            )

        # The dead-tensor COUNT is not an invariant -- MEASURED 7 September 2026, first
        # real load.  The safetensors headers carry 48 dead fused
        # `attn.qkv.{weight,bias}` tensors, but only 1 of them ever reaches this
        # filter: AutoWeightsLoader appends ".bias" to ignore_unexpected_suffixes
        # (models/utils.py:404) and the loader's own prefix grouping consumes the
        # rest before the tower's stream sees them.  Asserting 48 killed a boot
        # whose tower was, by every invariant that matters, correct.  So the count
        # is now REPORTED, and the invariant is checked BY NAME instead:
        #
        #   under EXL3 the fused qkv module has NO dense `.weight` parameter at
        #   all (it has .trellis/.suh/.svh/.mul1 and a .bias built from the three
        #   live per-projection biases).  A `...attn.qkv.weight` parameter on the
        #   built tower therefore means that module fell back to bf16 -- exactly
        #   the state in which a dead fused tensor could land silently and the
        #   tower would answer wrongly.  That is a hard failure.
        stray = sorted(
            n for n, _ in self.named_parameters()
            if n.endswith(".attn.qkv.weight")
        )
        if stray:
            raise RuntimeError(
                f"HAREM-VISION: {len(stray)} fused qkv module(s) still carry a "
                f"dense `.weight` parameter, e.g. {stray[:3]}. Under EXL3 they "
                "must not: such a module fell back to bf16 and the dead fused "
                "checkpoint tensor can land in it silently."
            )

        # Observational, never fatal: how much of the qkv bias side the loader
        # actually reported back on this call.  `load_weights` may be invoked more
        # than once (the loader groups by contiguous prefix runs), so a partial
        # set here is not by itself a fault -- it is written down so the next
        # reader has the number instead of a guess.
        try:
            loaded_set = set(loaded or ())
        except TypeError:
            loaded_set = set()
        qkv_bias = [n for n, _ in self.named_parameters()
                    if n.endswith(".attn.qkv.bias")]
        seen_bias = sum(1 for n in qkv_bias if n in loaded_set)
        try:
            from vllm.logger import init_logger

            init_logger(__name__).info(
                "HAREM-VISION: dead fused qkv tensors dropped=%d (header count "
                "48 is NOT an invariant, see the patch); qkv bias params=%d, "
                "reported loaded on this call=%d; loaded names on this call=%d",
                dropped, len(qkv_bias), seen_bias, len(loaded_set),
            )
        except Exception:  # pragma: no cover - logging must never stop a boot
            pass

    def load_weights(self, weights) -> set[str]:
        # HAREM-VISION VS3 (env-gated).  See _HAREM_VISION_DEAD_FUSED above.
        dropped = 0
        if _harem_vision_env():
            _dead = _HAREM_VISION_DEAD_FUSED

            def _harem_vision_filter(src):
                nonlocal dropped
                for name, data in src:
                    if name.endswith(_dead):
                        dropped += 1
                        continue
                    yield name, data

            weights = _harem_vision_filter(weights)
        loader = AutoWeightsLoader(self)
        loaded = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
        if _harem_vision_env():
            self._harem_vision_audit(dropped, loaded)
        return loaded
'''


# ---------------------------------------------------------------------------
# VS4 / VS6 / VS7 -- multimodal.py: the VIDEO half (7 September 2026, round 3)
#
# Rounds 1 and 2 both died on the FIRST video request, on all three ranks, with
# the same error:
#
#   models/utils.py:657  RuntimeError: shape mismatch: value tensor of shape
#                        [1656, 4096] cannot be broadcast to indexing result of
#                        shape [1728, 4096]
#   models/utils.py:665  ValueError: Attempted to assign 1656 = 1656 multimodal
#                        tokens to 1728 placeholders           -> EngineDeadError
#
# MEASURED root cause (CPU, no GPU, no engine -- the arithmetic is in docs/18
# section 4): a video passes through TWO INDEPENDENT frame samplers.
#
#   pixel path        transformers_utils/processors/glm5next.py:597
#                     Glm5NextVideoProcessor.sample_frames
#                       -> glm_sample_frame_indices(target_fps=fps_interval=2.0,
#                          extract_t = int(duration * target_fps))
#                       -> 8 frames -> video_grid_thw = [[4, 36, 46]]
#                       -> encoder rows = 4*36*46 // merge^2 = 1,656
#
#   placeholder path  model_executor/models/glm4_1v.py:1265
#                     Glm4vProcessingInfo._get_video_second_idx_glm46v
#                       -> GLM-4.6V constants: DYNAMIC_FPS_THRES={30:3,...},
#                          extract_t = int(duration*target_fps*temporal_patch_size)
#                       -> 24 frames -> [::2] -> 12 timestamps
#                     glm4_1v.py:1456 loops over TIMESTAMPS, not over grid T
#                       -> placeholders = 12 * 414 = 4,968
#
# The gap: Glm5NextProcessingInfo overrides get_hf_processor, _get_vision_info,
# _get_image_max_pixels and _get_video_max_pixels -- but NOT
# _get_video_second_idx_glm46v.  The pixel half was ported to GLM-5-Next's own
# sampler; the placeholder half stayed on GLM-4.6V's.  The two agreeing would be
# a coincidence -- the formulas are not even the same shape.
#
#   VS4  override _get_video_second_idx_glm46v so the timestamps come from the
#        PIXEL path.  Structurally exact: the COUNT is fixed by the number of
#        frames the pixel path receives (grid_t = ceil(n / temporal_patch_size),
#        because _preprocess pads the frame axis to a multiple of
#        temporal_patch_size at glm5next.py:717-722); only the VALUES come from
#        the sampler.  MEASURED on both fixtures: 4 timestamps -> 1,656
#        placeholders = 1,656 encoder rows, difference 0.
#
#   VS6  fail-closed.  Even with VS4 in place, ANY future disagreement kills the
#        engine core, because the two numbers first meet on the GPU inside
#        _merge_multimodal_embeddings.  So count the embed tokens in the
#        placeholder we just built and compare them with the grid the pixel path
#        produced, IN THE FRONTEND.  A mismatch becomes a rejected request
#        (HTTP 400) instead of three dead ranks.  Cost: one pass over a few
#        thousand ints on the API server's CPU.
#
#   VS7  the per-prompt VIDEO limit.  Glm4vProcessingInfo.get_supported_mm_limits
#        (glm4_1v.py:986) returns {"image": None, "video": 1}; GLM-5-Next
#        inherits it.  MEASURED consequence: multimodal/processing/context.py
#        :420-431 takes allowed = min(user_limit, supported_limit), so
#        `--limit-mm-per-prompt {"video":2}` is SILENTLY CLAMPED to 1 and a
#        two-video prompt is refused with "At most 1 video(s) may be provided in
#        one prompt."  Nothing downstream needs the 1: glm4_1v.py:1636 processes
#        video items one at a time in a loop, _construct_video_placeholder takes
#        an item index, and iter_mm_grid_thw / get_mrope_input_positions
#        (glm4_1v.py:2236, :2266) walk every mm_feature sorted by prompt offset
#        with no single-video assumption.  The 1 is a conservative default, and
#        VS7 lifts it to HAREM_VISION_VIDEO_LIMIT (default 2).
#
# THE BRAKE IS DELIBERATELY NOT HERE.  `--mm-processor-kwargs {"max_pixels":...}`
# never reaches the video processor -- Glm5NextVideoProcessorKwargs
# (glm5next.py:549) does not declare `max_pixels`, and the engine says so:
# "Keyword argument `max_pixels` is not a valid argument for this processor and
# will be ignored."  The key that DOES reach it is `max_image_tokens`
# (glm5next.py:646/660 -> _pixel_budget).  MEASURED on a 1080p clip: no kwarg ->
# grid (4,78,138) = 10,764 tokens; max_image_tokens=8000 -> (4,66,118) = 7,788
# tokens; images bit-identical either way, because the checkpoint already gives
# images 8,000 tokens / 12,544,000 pixels.  So the brake is one more key in the
# .env's --mm-processor-kwargs and needs no code.  `max_pixels` stays in that
# line because vLLM's OWN estimate (_get_video_max_pixels -> glm4_1v.py:1033
# get_mm_max_tokens_per_item -> the encoder compute/cache budget) is its only
# reader.
# ---------------------------------------------------------------------------

VS4_ANCHOR = '''    def _get_video_max_pixels(self) -> int:
        mm_kwargs = self.ctx.get_merged_mm_kwargs({})
        if (override := mm_kwargs.get("max_pixels")) is not None:
            return int(override)
        return self._processor_pixel_budget(self.get_hf_processor().video_processor)[1]
'''

VS4_REPL = '''    def _get_video_max_pixels(self) -> int:
        mm_kwargs = self.ctx.get_merged_mm_kwargs({})
        if (override := mm_kwargs.get("max_pixels")) is not None:
            return int(override)
        return self._processor_pixel_budget(self.get_hf_processor().video_processor)[1]

    def _get_video_second_idx_glm46v(self, metadata, total_frames: int) -> list[int]:
        """HAREM-VISION VS4 (env-gated): build the placeholder timestamps from
        the PIXEL path's own frame sampler.

        Upstream (glm4_1v.py:1265) uses GLM-4.6V constants -- target_fps from
        DYNAMIC_FPS_THRES and extract_t = duration * target_fps *
        temporal_patch_size -- while Glm5NextVideoProcessor.sample_frames uses
        fps_interval (2.0) and extract_t = duration * target_fps.  On the 4 s
        fixture that is 12 timestamps against a grid of 4: 4,968 placeholders
        for 1,656 encoder rows, and the engine core dies in
        _merge_multimodal_embeddings.

        The equality here is structural, not lucky.  _preprocess pads the frame
        axis up to a multiple of temporal_patch_size and then sets
        grid_t = frames // temporal_patch_size (glm5next.py:717-722), so the
        timestamp COUNT is fixed by the number of frames the pixel path
        receives.  Only the VALUES come from the sampler, and they keep
        upstream's convention: the first frame of each temporal patch,
        int(frame_index / fps) seconds.
        """
        if not _harem_vision_env():
            return super()._get_video_second_idx_glm46v(metadata, total_frames)

        from vllm.transformers_utils.processors.glm5next import (
            glm_sample_frame_indices,
        )

        vp = self.get_video_processor()
        tps = int(getattr(vp, "temporal_patch_size", 2) or 1)
        fps = float(metadata["fps"]) or 1.0

        if not metadata.get("do_sample_frames", True):
            # The loader already picked the frames: VideoMediaIO caps at
            # num_frames and then sets do_sample_frames=False
            # (multimodal/video.py:210).  The pixel path then uses the
            # delivered array as-is, so the frame COUNT is len(video_array) and
            # frames_indices only supplies the timestamp values.
            idx = [int(i) for i in metadata.get("frames_indices",
                                                range(int(total_frames)))]
            n_frames = int(total_frames)
        else:
            idx = [int(i) for i in glm_sample_frame_indices(
                int(metadata.get("total_num_frames", total_frames)),
                fps,
                float(metadata.get("duration") or 0),
                target_fps=vp.fps_interval,
                max_frame_count=vp.max_frame_count_dynamic,
                temporal_patch_size=tps,
            )]
            n_frames = len(idx)

        if n_frames <= 0 or not idx:
            # duration * fps_interval < 1 (a clip shorter than ~0.5 s) makes
            # glm_sample_frame_indices return []; the pixel path raises an
            # IndexError on the same input a moment later.  Say what happened
            # while there is still a request to reject.
            raise ValueError(
                "HAREM-VISION: the GLM-5-Next frame sampler selected no frames "
                f"for this video (fps={fps}, "
                f"duration={metadata.get('duration')!r}, "
                f"total_num_frames={metadata.get('total_num_frames')!r}). "
                "Clips shorter than ~0.5 s cannot be sampled by this processor."
            )

        grid_t = max((n_frames + (-n_frames % tps)) // tps, 1)
        last = len(idx) - 1
        return [int(idx[min(j * tps, last)] / fps) for j in range(grid_t)]
'''

VS6_ANCHOR = '''    def _processor_pixel_budget(self, proc) -> tuple[int, int]:
        from vllm.transformers_utils.processors.glm5next import _pixel_budget

        return _pixel_budget(
            proc.min_image_tokens,
            proc.max_image_tokens,
            proc.patch_size,
            proc.merge_size,
            proc.temporal_patch_size,
        )
'''

VS6_REPL = '''    def _processor_pixel_budget(self, proc) -> tuple[int, int]:
        from vllm.transformers_utils.processors.glm5next import _pixel_budget

        return _pixel_budget(
            proc.min_image_tokens,
            proc.max_image_tokens,
            proc.patch_size,
            proc.merge_size,
            proc.temporal_patch_size,
        )

    def _construct_video_placeholder(self, video_array, metadata, grid_thw):
        """HAREM-VISION VS6 (env-gated): turn a placeholder/embedding
        disagreement into a REJECTED REQUEST instead of a dead engine core.

        Today the two numbers first meet on the GPU, at
        model_executor/models/utils.py:657 -- `inputs_embeds[is_multimodal] =
        mm_embeds_flat` -- where a mismatch is a RuntimeError inside a worker,
        which kills the worker, which kills the engine core, on every rank at
        once.  One malformed multimodal request taking down all three nodes is
        not acceptable for production, and it is exactly what rounds 1 and 2
        did.

        The same equation can be checked in the frontend for the price of one
        pass over the placeholder list: the number of embed tokens we are about
        to write must equal the number of rows the vision tower will produce for
        the grid the pixel path already returned.  Raising here surfaces as an
        HTTP 400 with the engine untouched.
        """
        placeholder = super()._construct_video_placeholder(
            video_array, metadata, grid_thw
        )
        if not _harem_vision_env():
            return placeholder

        hf_processor = self.get_hf_processor()
        merge_length = hf_processor.image_processor.merge_size ** 2
        embed_token_id = self._get_video_frame_embed_token_id(hf_processor)
        got = sum(1 for tok in placeholder if tok == embed_token_id)
        want = int(grid_thw.prod()) // merge_length
        if got != want:
            grid = grid_thw.tolist() if hasattr(grid_thw, "tolist") else grid_thw
            msg = (
                "HAREM-VISION VS6: this video's prompt placeholders do not "
                f"match its encoder output -- {got} placeholder tokens for "
                f"{want} embedding rows (video_grid_thw={grid}). The request "
                "is rejected here, in the frontend; serving it would kill the "
                "engine core in _merge_multimodal_embeddings. That is the "
                "round-1/round-2 failure mode; see VS4 above."
            )
            try:
                from vllm.exceptions import VLLMValidationError
            except Exception:  # pragma: no cover - old images
                VLLMValidationError = None
            if VLLMValidationError is not None:
                raise VLLMValidationError(msg, parameter="video")
            raise ValueError(msg)
        return placeholder
'''

VS7_ANCHOR = '''    def get_hf_processor(self, **kwargs: object):
        proc = getattr(self, "_glm5_hf_processor", None)
        if proc is None:
'''

VS7_REPL = '''    def get_supported_mm_limits(self):
        """HAREM-VISION VS7 (env-gated): lift the inherited per-prompt VIDEO
        limit of 1.

        Glm4vProcessingInfo.get_supported_mm_limits (glm4_1v.py:986) returns
        {"image": None, "video": 1} and GLM-5-Next inherits it.  That number is
        not advisory: multimodal/processing/context.py:420-431 computes
        allowed = min(user_limit, supported_limit), so
        `--limit-mm-per-prompt {"video":2}` is silently clamped back to 1 and a
        two-video prompt is refused with "At most 1 video(s) may be provided in
        one prompt."  The flag looks honoured in the boot log and is not.

        Nothing below the declaration assumes a single video: glm4_1v.py:1636
        loops over the video items one at a time, _construct_video_placeholder
        takes an item index, and iter_mm_grid_thw / get_mrope_input_positions
        (glm4_1v.py:2236, :2266) walk every mm_feature sorted by prompt offset
        and accumulate positions across items.  With VS4 in place the video
        branch of iter_mm_grid_thw also takes its per-frame path
        (len(embed_ranges) == grid_t), which is the accurate one.

        Knob: HAREM_VISION_VIDEO_LIMIT (default 2).  Garbage stops the boot
        rather than quietly serving a different number than the .env asked for.
        """
        limits = dict(super().get_supported_mm_limits())
        if not _harem_vision_env():
            return limits
        raw = _harem_vision_os.environ.get("HAREM_VISION_VIDEO_LIMIT", "2")
        try:
            limit = int(raw)
        except ValueError:
            raise ValueError(
                f"HAREM-VISION: HAREM_VISION_VIDEO_LIMIT={raw!r} is not an "
                "integer."
            ) from None
        if limit < 1:
            raise ValueError(
                f"HAREM-VISION: HAREM_VISION_VIDEO_LIMIT={limit} must be >= 1."
            )
        limits["video"] = limit
        return limits

    def get_hf_processor(self, **kwargs: object):
        proc = getattr(self, "_glm5_hf_processor", None)
        if proc is None:
'''

# (tag, relative path, anchor, replacement)
ANCHORS = [
    ("VS1", MODEL_REL, VS1_ANCHOR, VS1_REPL),
    ("VS2", MM_REL, VS2_ANCHOR, VS2_REPL),
    ("VS3", MM_REL, VS3_ANCHOR, VS3_REPL),
    ("VS4", MM_REL, VS4_ANCHOR, VS4_REPL),
    ("VS6", MM_REL, VS6_ANCHOR, VS6_REPL),
    ("VS7", MM_REL, VS7_ANCHOR, VS7_REPL),
]

# What must be true of the written files.  (relative path, needle, count)
POST_CHECKS = [
    (MODEL_REL, "def _harem_vision_tower_quant(qc):", 1),
    (MODEL_REL, "quant_config=_harem_vision_tower_quant(vllm_config.quant_config),", 1),
    # The MLA tower construction must NOT have been touched: exactly one bare
    # `quant_config=None,` is gone (the vision one) and the MLA line stays.
    (MODEL_REL, "                quant_config=None,\n", 0),
    (MM_REL, "def _harem_vision_env() -> bool:", 1),
    (MM_REL, "_HAREM_VISION_DEAD_FUSED = ", 1),
    (MM_REL, '".attn.q_proj.": (".attn.qkv.", "q"),', 1),
    (MM_REL, '".attn.k_proj.": (".attn.qkv.", "k"),', 1),
    (MM_REL, '".attn.v_proj.": (".attn.qkv.", "v"),', 1),
    (MM_REL, "hf_to_vllm_mapper = WeightsMapper(", 2),
    (MM_REL, "def _harem_vision_audit(self, dropped: int, loaded) -> None:", 1),
    (MM_REL, "def _harem_vision_filter(src):", 1),
    (MM_REL, "self._harem_vision_audit(dropped, loaded)", 1),
    # VS4 / VS6 / VS7 -- the video half.
    (MM_REL, "def _get_video_second_idx_glm46v(self, metadata, total_frames: int)", 1),
    (MM_REL, "target_fps=vp.fps_interval,", 1),
    (MM_REL, "def _construct_video_placeholder(self, video_array, metadata, grid_thw):", 1),
    (MM_REL, "HAREM-VISION VS6:", 1),
    (MM_REL, "def get_supported_mm_limits(self):", 1),
    (MM_REL, "HAREM_VISION_VIDEO_LIMIT", 4),
    # The three inherited methods must be overridden EXACTLY once each: a second
    # copy would mean an anchor fired twice and the last definition would win
    # silently.
    (MM_REL, "def _get_video_max_pixels(self) -> int:", 1),
    (MM_REL, "def _processor_pixel_budget(self, proc) -> tuple[int, int]:", 1),
    (MM_REL, "def get_hf_processor(self, **kwargs: object):", 1),
]


def _path(root, rel):
    return os.path.join(root, *rel)


def fail(msg):
    print(f"[vision] FAIL: {msg}", file=sys.stderr)
    return 3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="vllm package root")
    ap.add_argument(
        "--check",
        action="store_true",
        help="report anchor counts only, change nothing",
    )
    a = ap.parse_args()

    files = {}
    for rel in (MODEL_REL, MM_REL):
        p = _path(a.root, rel)
        if not os.path.isfile(p):
            return fail(f"no such file {p}")
        with open(p) as f:
            files[rel] = f.read()

    already = [rel for rel, src in files.items() if MARK in src]
    if len(already) == len(files):
        print("[vision] already applied; nothing to do")
        return 0
    if already:
        return fail(
            f"half-patched tree: {[r[-1] for r in already]} carry {MARK}, the "
            "rest do not. Restore the image files before retrying."
        )

    out = dict(files)
    counts = []
    for tag, rel, anchor, repl in ANCHORS:
        n = out[rel].count(anchor)
        counts.append((tag, n))
        if n != 1:
            print(f"[vision] anchor counts: {counts}", file=sys.stderr)
            return fail(
                f"{tag}: anchor matched {n} times in {_path(a.root, rel)} "
                "(want exactly 1)"
            )
        out[rel] = out[rel].replace(anchor, repl, 1)

    print(f"[vision] anchors 1/1: {' '.join(t for t, _ in counts)}")

    if a.check:
        for rel, src in out.items():
            try:
                compile(src, rel[-1], "exec")
            except SyntaxError as e:
                return fail(f"patched {rel[-1]} does not compile: {e}")
        for rel, needle, want in POST_CHECKS:
            got = out[rel].count(needle)
            if got != want:
                return fail(
                    f"--check post-check {rel[-1]}: {needle!r} would appear "
                    f"{got} times, want {want}"
                )
        print(
            f"[vision] --check: {len(ANCHORS)}/{len(ANCHORS)} anchors match once, "
            f"both files compile, {len(POST_CHECKS)} post-checks hold; "
            "nothing written"
        )
        return 0

    for rel, src in out.items():
        path = _path(a.root, rel)
        tmp = path + ".harem-tmp"
        with open(tmp, "w") as f:
            f.write(src)
        try:
            py_compile.compile(tmp, doraise=True, cfile=tmp + ".pyc")
        except py_compile.PyCompileError as e:
            os.unlink(tmp)
            return fail(f"patched {path} does not compile: {e}")
        if os.path.exists(tmp + ".pyc"):
            os.unlink(tmp + ".pyc")
        os.replace(tmp, path)

    for rel, needle, want in POST_CHECKS:
        with open(_path(a.root, rel)) as f:
            got = f.read().count(needle)
        if got != want:
            return fail(
                f"post-check {rel[-1]}: {needle!r} appears {got} times, want {want}"
            )

    print(
        "[vision] applied: VS1 tower quant_config (EXL3 only), "
        "VS2 .attn.{q,k,v}_proj. -> .attn.qkv. stacked map, "
        "VS3 dead fused attn.qkv drop + post-load tower audit, "
        "VS4 video timestamps from the pixel-path sampler, "
        "VS6 frontend placeholder/embedding fail-closed (HTTP 400, not a dead "
        "engine), VS7 per-prompt video limit -> HAREM_VISION_VIDEO_LIMIT. "
        "Runtime knobs: HAREM_VISION=1, HAREM_VISION_VIDEO_LIMIT (default 2)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
