#!/usr/bin/env python3
"""HAREM DFlash2 target side: the EAGLE3 interface on GLM-5.3 (glm5next/common/model.py).

The 21d93d0d8 counterpart of the diff to ``vllm/models/glm5next/nvidia/model.py`` that
the previous stack's image carried in its DFlash2 port layer (+48 lines relative to
487ecf187; a three-way merge of the vLLM fork commit e7097feb6 "Support GLM-5.3 DFlash
speculation", see docs/04 section 3.1). Upstream 21d93d0d8 does not implement
``SupportsEagle3`` for GLM-5.3; the DFlash driver asks the target for auxiliary hidden
states and the boot dies with ``RuntimeError: Model does not support EAGLE3 interface``.

Edits (the same text as the previous port; only the file path and the surroundings
changed):
  E1  import: EagleModelMixin + SupportsEagle3
  E2  Glm5NextModel(nn.Module, EagleModelMixin) + ``dflash_capture``
  E3  ``_prepare_aux_hidden_state``: on an mHC layer, ``hc_post`` (the residual stream is
      [T, n, H]); for DFlash it is contracted to [T, H] with ``hc_contract``, for other
      methods flatten(1). Unchanged on a non-mHC layer or on the last layer (residual
      None).
  E4  forward loop: captures the output of the layer with
      ``layer_idx + 1 in aux_hidden_state_layers`` (the drafter's ``target_layer_ids``
      + 1 -- the upstream EAGLE3 contract); sp_all_gather under sequence parallelism.
  E5  returns (hidden_states, aux_hidden_states) when there is aux (the V2 runner expects
      the pair through ``use_aux_hidden_state_outputs``).
  E6  Glm5NextForCausalLM ... SupportsEagle3
  E7  Glm5NextForConditionalGeneration ... SupportsEagle3 (the served class; the protocol
      reaches Glm5NextModel through get_language_model().model)

Without a drafter (speculative_config None) aux_hidden_state_layers stays empty: forward
follows exactly the upstream path (no capture, no pair return). There is no environment
knob: this is a target-side interface hook, and behaviour changes only when a drafter is
enabled.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harem_yama as Y  # noqa: E402

MODEL = "models/glm5next/common/model.py"

D = Y.Duzen
DUZENLER = [
    D("E1-ice-aktarma", MODEL,
      "from vllm.model_executor.models.interfaces import (\n"
      "    HasInnerState,\n"
      "    IsHybrid,\n"
      "    MixtureOfExperts,\n"
      "    SupportsPP,\n"
      ")\n",
      "from vllm.model_executor.models.interfaces import (\n"
      "    EagleModelMixin,  # HAREM-DFLASH-EAGLE3\n"
      "    HasInnerState,\n"
      "    IsHybrid,\n"
      "    MixtureOfExperts,\n"
      "    SupportsEagle3,  # HAREM-DFLASH-EAGLE3\n"
      "    SupportsPP,\n"
      ")\n"),
    D("E2-model-sinifi", MODEL,
      "class Glm5NextModel(nn.Module):\n"
      "    def __init__(self, *, vllm_config: VllmConfig, prefix: str = \"\"):\n"
      "        super().__init__()\n"
      "\n"
      "        config = vllm_config.model_config.hf_text_config\n"
      "        _validate_supported_config(config)\n"
      "        self.config = config\n",
      "class Glm5NextModel(nn.Module, EagleModelMixin):  # HAREM-DFLASH-EAGLE3\n"
      "    def __init__(self, *, vllm_config: VllmConfig, prefix: str = \"\"):\n"
      "        super().__init__()\n"
      "\n"
      "        config = vllm_config.model_config.hf_text_config\n"
      "        _validate_supported_config(config)\n"
      "        self.config = config\n"
      "        # HAREM-DFLASH-EAGLE3: a DFlash drafter reads the target's mHC residual\n"
      "        # stream contracted back to hidden_size (see _prepare_aux_hidden_state).\n"
      "        speculative_config = vllm_config.speculative_config\n"
      "        self.dflash_capture = (\n"
      "            speculative_config is not None and speculative_config.use_dflash()\n"
      "        )\n"),
    D("E3-aux-hazirla", MODEL,
      "    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:\n"
      "        return self.embed_tokens(input_ids)\n"
      "\n"
      "    def forward(\n",
      "    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:\n"
      "        return self.embed_tokens(input_ids)\n"
      "\n"
      "    def _prepare_aux_hidden_state(\n"
      "        self,\n"
      "        layer: Glm5NextDecoderLayer,\n"
      "        hidden_states: torch.Tensor,\n"
      "        residual: torch.Tensor | None,\n"
      "        post: torch.Tensor | None,\n"
      "        comb: torch.Tensor | None,\n"
      "    ) -> torch.Tensor:\n"
      "        # HAREM-DFLASH-EAGLE3: an mHC layer defers its hc_post to the next\n"
      "        # layer's fused pre, so materialise it here for the drafter: the full\n"
      "        # residual stream [T, n, H] after this layer, contracted to [T, H] for\n"
      "        # DFlash. Non-mHC layers and the last mHC layer (which already\n"
      "        # contracted and returned residual=None) pass through unchanged.\n"
      "        if not layer.mhc or residual is None:\n"
      "            return hidden_states\n"
      "\n"
      "        assert post is not None and comb is not None\n"
      "        aux_hidden_state = layer.hc_post(hidden_states, residual, post, comb)\n"
      "        if self.dflash_capture:\n"
      "            return hc_contract(aux_hidden_state, layer.n)\n"
      "        return aux_hidden_state.flatten(1)\n"
      "\n"
      "    def forward(\n"),
    D("E4-yakalama", MODEL,
      "        for layer in self._active_layers:\n"
      "            hidden_states, residual, post, comb = layer(\n"
      "                positions, hidden_states, residual, post, comb\n"
      "            )\n",
      "        # HAREM-DFLASH-EAGLE3: collect the drafter's auxiliary hidden states\n"
      "        # (aux layer L = output of layer L-1, the upstream EAGLE3 convention).\n"
      "        aux_hidden_states: list[torch.Tensor] = []\n"
      "        if self.start_layer in self.aux_hidden_state_layers:\n"
      "            aux_hidden_states.append(hidden_states)\n"
      "\n"
      "        for layer_idx, layer in enumerate(self._active_layers, start=self.start_layer):\n"
      "            hidden_states, residual, post, comb = layer(\n"
      "                positions, hidden_states, residual, post, comb\n"
      "            )\n"
      "            if layer_idx + 1 in self.aux_hidden_state_layers:\n"
      "                aux_hidden_state = self._prepare_aux_hidden_state(\n"
      "                    layer, hidden_states, residual, post, comb\n"
      "                )\n"
      "                if self.is_sequence_parallel:\n"
      "                    aux_hidden_state = sp_all_gather(aux_hidden_state)[:full_num_tokens]\n"
      "                aux_hidden_states.append(aux_hidden_state)\n"),
    D("E5-donus", MODEL,
      "        hidden_states = self.norm(hidden_states)\n"
      "        return hidden_states\n",
      "        hidden_states = self.norm(hidden_states)\n"
      "        # HAREM-DFLASH-EAGLE3: hand the drafter its auxiliary hidden states.\n"
      "        if aux_hidden_states:\n"
      "            return hidden_states, aux_hidden_states\n"
      "        return hidden_states\n"),
    D("E6-causallm", MODEL,
      "class Glm5NextForCausalLM(\n"
      "    nn.Module, HasInnerState, SupportsPP, MixtureOfExperts, IsHybrid\n"
      "):\n",
      "class Glm5NextForCausalLM(\n"
      "    nn.Module,\n"
      "    HasInnerState,\n"
      "    SupportsPP,\n"
      "    MixtureOfExperts,\n"
      "    IsHybrid,\n"
      "    SupportsEagle3,  # HAREM-DFLASH-EAGLE3\n"
      "):\n"),
    D("E7-kosullu", MODEL,
      "class Glm5NextForConditionalGeneration(\n"
      "    Glm4vForConditionalGeneration, HasInnerState, IsHybrid, MixtureOfExperts\n"
      "):\n",
      "class Glm5NextForConditionalGeneration(\n"
      "    Glm4vForConditionalGeneration,\n"
      "    HasInnerState,\n"
      "    IsHybrid,\n"
      "    MixtureOfExperts,\n"
      "    SupportsEagle3,  # HAREM-DFLASH-EAGLE3\n"
      "):\n"),
]

YAMA = Y.Yama(
    ad="yama-dflash-eagle3",
    aciklama="GLM-5.3 hedefinde EAGLE3 aux gizli durum arayüzü (DFlash2; eski port model.py)",
    duzenler=DUZENLER,
    taban={
        MODEL: "0cec23ba8181b8f9f80b3eee65eb76170c11a0907ba960d670ada95d4e17fe5c",
    },
)

if __name__ == "__main__":
    sys.exit(Y.main(YAMA, __file__))
