#!/usr/bin/env python3
"""HAREM-VISION: the full-scope EXL3 vision tower plus video safeguards (HAREM_VISION=1).

A rewrite to 21d93d0d8 of the previous stack's patch-vision-tp3.py (VS1-VS7). In the new
tree the GLM image processor was deleted from vLLM (#57387: the processor is now
transformers 5.18's ``Glm5NextProcessor``); the tower and ``Glm5NextProcessingInfo`` live
in ``models/glm5next/common/multimodal.py``.

  VS1  model.py: the tower is built with ``quant_config=None`` (right for the fp8
       checkpoint). turboderp's tower is 6-bit EXL3 (there are no dense proj/mlp/merger
       weights on disk) -> with HAREM_VISION=1 and an EXL3 config, the config is passed
       to the tower.
  VS2  multimodal.py: ``.attn.{q,k,v}_proj.`` spellings are added to the stacked-weight
       mapper (the checkpoint writes ``attn.q_proj.trellis``; upstream has only GLM-OCR's
       ``.attn.q.``). The target is ``.attn.qkv.`` (attribute ``self.qkv``).
  VS3  post-load audit: 24 x 4 + 3 = 99 EXL3 linears, 0 unquantized, no module carrying a
       dense ``.attn.qkv.weight``. (The dead fused ``attn.qkv.{weight,bias}`` tensors are
       now dropped by the sidecar index, see yama-sidecar-dizin.py; there is no filter in
       this patch.)
  VS4  RETIRED: video timestamps from the pixel path's sampler -> upstream (#55647,
       ``_get_video_second_idx_glm46v`` now comes from ``sample_frames``).
  VS6  front-end check, placeholder count = encoder rows: a mismatch kills the engine (all
       three ranks) on the GPU in ``_merge_multimodal_embeddings``; here it rejects the
       single request (VLLMValidationError -> HTTP 400). Upstream checks only the item
       count [verified, processor.py:1743-1776].
  VS7  per-prompt video limit: the inherited ``{"video": 1}`` silently clamps
       --limit-mm-per-prompt (context.py: min(user, supported)) -> HAREM_VISION_VIDEO_LIMIT
       (default 2; the production environment sets 4).
HAREM_VISION empty = exactly upstream (the knob is read at import and at run time).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harem_yama as Y  # noqa: E402

MODEL = "models/glm5next/common/model.py"
MM = "models/glm5next/common/multimodal.py"

ESLEYICI = '''class Glm5NextVisionTransformer(nn.Module):
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

VS2_YENI = '''# --- HAREM-VISION -----------------------------------------------------------
import os as _harem_vision_os


def _harem_vision_env() -> bool:
    return _harem_vision_os.environ.get("HAREM_VISION") == "1"


def _harem_vision_audit(tower) -> None:
    """VS3: an EXL3 tower must be all EXL3 (Exl3Config.get_quant_method fails
    open: an unresolved prefix silently becomes an unquantized linear)."""
    exl3 = other = 0
    for mod in tower.modules():
        name = type(getattr(mod, "quant_method", None)).__name__
        if name == "Exl3LinearMethod":
            exl3 += 1
        elif "Linear" in name:
            other += 1
    want = len(tower.blocks) * 4 + 3  # qkv, proj, gate_up, down; merger x3
    stray = [n for n, _ in tower.named_parameters() if n.endswith(".attn.qkv.weight")]
    if exl3 != want or other or stray:
        raise RuntimeError(
            f"HAREM-VISION: tower has {exl3} EXL3 linears (want {want}), {other} "
            f"unquantized, {len(stray)} dense attn.qkv weights; a module did not "
            "resolve to its EXL3 tensors (check the qkv_proj packed mapping)."
        )
    from vllm.logger import init_logger

    init_logger(__name__).info(
        "HAREM-VISION: tower EXL3 linears=%d, unquantized=0, vit_attn_backend=%s",
        exl3, getattr(tower.attn_backend, "name", tower.attn_backend))
# --- end HAREM-VISION --------------------------------------------------------


''' + ESLEYICI + '''    # HAREM-VISION VS2: turboderp's EXL3 tower writes `.attn.q_proj.trellis`;
    # add the `_proj` spellings (the destination attribute is `self.qkv`).
    if _harem_vision_env():
        hf_to_vllm_mapper = WeightsMapper(
            orig_to_new_stacked={
                **hf_to_vllm_mapper.orig_to_new_stacked,
                ".attn.q_proj.": (".attn.qkv.", "q"),
                ".attn.k_proj.": (".attn.qkv.", "k"),
                ".attn.v_proj.": (".attn.qkv.", "v"),
            }
        )
'''

VS67_YENI = '''    def get_supported_mm_limits(self):
        # HAREM-VISION VS7: the inherited {"video": 1} silently clamps
        # --limit-mm-per-prompt (context.py: min(user, supported)); videos are
        # processed item by item, so lift it to HAREM_VISION_VIDEO_LIMIT.
        limits = dict(super().get_supported_mm_limits())
        if _harem_vision_env():
            raw = _harem_vision_os.environ.get("HAREM_VISION_VIDEO_LIMIT", "2")
            if not raw.strip().isdigit() or int(raw) < 1:
                raise ValueError(
                    f"HAREM-VISION: HAREM_VISION_VIDEO_LIMIT={raw!r} must be an "
                    "integer >= 1"
                )
            limits["video"] = int(raw)
        return limits

    def _construct_video_placeholder(self, video_array, metadata, grid_thw):
        # HAREM-VISION VS6: placeholders vs encoder rows first meet on the GPU
        # (_merge_multimodal_embeddings) and a mismatch kills the engine on every
        # rank; check it here and reject the one request instead (HTTP 400).
        placeholder = super()._construct_video_placeholder(
            video_array, metadata, grid_thw
        )
        if _harem_vision_env():
            hf_processor = self.get_hf_processor()
            got = sum(1 for t in placeholder if t == hf_processor.image_token_id)
            want = int(grid_thw.prod()) // hf_processor.image_processor.merge_size**2
            if got != want:
                from vllm.exceptions import VLLMValidationError

                raise VLLMValidationError(
                    f"HAREM-VISION VS6: {got} video placeholder tokens for {want} "
                    f"encoder rows (video_grid_thw={grid_thw.tolist()}); rejected "
                    "in the frontend instead of killing the engine core.",
                    parameter="video",
                )
        return placeholder

    def get_image_size_with_most_features(self) -> ImageSize:
'''

D = Y.Duzen
DUZENLER = [
    D("VS1a-kapi", MODEL,
      "        with self._mark_tower_model(vllm_config, {\"image\", \"video\"}):\n"
      "            self.visual = Glm5NextVisionTransformer(\n",
      "        # HAREM-VISION VS1: turboderp's EXL3 checkpoint quantizes the vision\n"
      "        # tower (6-bit; no dense proj/mlp/merger weights on disk). With\n"
      "        # HAREM_VISION=1 an EXL3 config is passed through; otherwise None.\n"
      "        import os as _harem_os\n"
      "\n"
      "        _harem_tower_qc = vllm_config.quant_config\n"
      "        if not (\n"
      "            _harem_os.environ.get(\"HAREM_VISION\") == \"1\"\n"
      "            and _harem_tower_qc is not None\n"
      "            and _harem_tower_qc.get_name() == \"exl3\"\n"
      "        ):\n"
      "            _harem_tower_qc = None\n"
      "        with self._mark_tower_model(vllm_config, {\"image\", \"video\"}):\n"
      "            self.visual = Glm5NextVisionTransformer(\n"),
    D("VS1b-kule", MODEL,
      "                # pattern (quant_config=None for BF16 submodules).\n"
      "                quant_config=None,\n",
      "                # pattern (quant_config=None for BF16 submodules).\n"
      "                quant_config=_harem_tower_qc,  # HAREM-VISION VS1\n"),
    D("VS2-esleyici", MM, ESLEYICI, VS2_YENI),
    D("VS3-denetim", MM,
      "    def load_weights(self, weights) -> set[str]:\n"
      "        loader = AutoWeightsLoader(self)\n"
      "        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)\n",
      "    def load_weights(self, weights) -> set[str]:\n"
      "        loader = AutoWeightsLoader(self)\n"
      "        loaded = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)\n"
      "        if _harem_vision_env():\n"
      "            _harem_vision_audit(self)  # HAREM-VISION VS3\n"
      "        return loaded\n"),
    D("VS6-VS7-video", MM,
      "    def get_image_size_with_most_features(self) -> ImageSize:\n", VS67_YENI),
]

YAMA = Y.Yama(
    ad="yama-gorsel",
    aciklama="tam kapsam EXL3 görsel kulesi + video güvenceleri (HAREM_VISION)",
    duzenler=DUZENLER,
    taban={
        MODEL: "0cec23ba8181b8f9f80b3eee65eb76170c11a0907ba960d670ada95d4e17fe5c",
        MM: "1aa18a93d7f1cb4063eb0f190479dda5131d9869af10e49d82131992f6953a2d",
    },
)

if __name__ == "__main__":
    sys.exit(Y.main(YAMA, __file__))
