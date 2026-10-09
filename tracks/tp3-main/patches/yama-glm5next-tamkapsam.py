#!/usr/bin/env python3
"""HAREM full-scope + TP3 model patches (glm5next/common; the vLLM 21d93d0d8 counterpart
of edit 4 of the previous stack's patch-vllm-tp3 and of S2 of its patch-fullscope-tp3).

The turboderp 4.05bpw checkpoint is "full-scope" EXL3: besides the routed experts, MLA,
KDA, the shared expert, the dense MLP and lm_head are EXL3 as well. Upstream glm5next
builds the MLA and KDA layers with ``quant_config=None`` (written for the fp8
checkpoint), so on EXL3 it would take that weight for bf16 and fail to load it. Three
edits:

  M1  shared-expert width: the plugin's TP3 hook writes the padded width to
      ``text_config.harem_shared_expert_intermediate_size`` (at tp=3: 2048 -> 2304 =
      3 x 6 x 128); without it (tp 1/2/4) the upstream product stays.
  M2  MLA: hand the EXL3 config back (o_proj, q_b_proj, fused_qkv_a_proj and the
      indexer's wq_b are EXL3; kv_b_proj and wk_weights_proj are bf16 -> not resolved,
      not quantized).
  K1  KDA: hand the EXL3 config back; ``in_proj_qkvbfg_a`` falls to the plugin's mixed
      method (EXL3 q/k/v + bf16 b/f_a/g_a), o_proj is EXL3, f_b/g_b are bf16.

Every other checkpoint (fp8, bf16) keeps the upstream ``None``: the gate is
``quant_config.get_name() == "exl3"`` and nothing else. There is no environment knob.

That the KDA q/k/v and conv1d sit fused in the checkpoint (qkv_proj, conv1d) is not
this patch's business but the offline sidecar's (tools/sidecar_kur.py).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harem_yama as Y  # noqa: E402

MODEL = "models/glm5next/common/model.py"
KDA = "models/glm5next/common/kda.py"

D = Y.Duzen
DUZENLER = [
    D("M1-paylasilan-uzman", MODEL,
      "            intermediate_size = config.moe_intermediate_size * config.n_shared_experts\n",
      "            intermediate_size = config.moe_intermediate_size * config.n_shared_experts\n"
      "            # HAREM-TP3: cuda_exl3._harem_tp3 records the shared expert's padded\n"
      "            # width (2048 -> 2304 = 3 x 6 x 128 at tp=3; absent at tp 1/2/4).\n"
      "            # The routed experts go expert-parallel and are not padded.\n"
      "            intermediate_size = getattr(\n"
      "                config, \"harem_shared_expert_intermediate_size\", intermediate_size\n"
      "            )\n"),
    D("M2-mla-quant", MODEL,
      "                quant_config=None,  # MLA projections are BF16 in checkpoint\n",
      "                # HAREM-FULLSCOPE: an EXL3 checkpoint quantizes o_proj, q_b_proj,\n"
      "                # fused_qkv_a_proj and indexer.wq_b (the rest resolve to\n"
      "                # unquantized); every other checkpoint keeps the upstream None.\n"
      "                quant_config=(\n"
      "                    quant_config\n"
      "                    if quant_config is not None and quant_config.get_name() == \"exl3\"\n"
      "                    else None\n"
      "                ),  # MLA projections are BF16 in checkpoint\n"),
    D("K1-kda-quant", KDA,
      "        try:\n"
      "            vllm_config.quant_config = None\n"
      "            super().__init__(config, vllm_config, prefix)\n"
      "        finally:\n"
      "            vllm_config.quant_config = saved_quant_config\n",
      "        try:\n"
      "            vllm_config.quant_config = None\n"
      "            super().__init__(config, vllm_config, prefix)\n"
      "        finally:\n"
      "            vllm_config.quant_config = saved_quant_config\n"
      "        # HAREM-FULLSCOPE: an EXL3 checkpoint quantizes q/k/v (in_proj_qkvbfg_a\n"
      "        # becomes cuda_exl3's mixed EXL3 + bf16 method) and o_proj; b/f_a/g_a and\n"
      "        # f_b/g_b resolve to unquantized. Other checkpoints keep the None above.\n"
      "        if saved_quant_config is not None and saved_quant_config.get_name() == \"exl3\":\n"
      "            self.quant_config = saved_quant_config\n"),
]

YAMA = Y.Yama(
    ad="yama-glm5next-tamkapsam",
    aciklama="GLM-5.3 tam kapsam EXL3 (MLA/KDA quant_config) + TP3 paylaşılan uzman genişliği",
    duzenler=DUZENLER,
    taban={
        MODEL: "0cec23ba8181b8f9f80b3eee65eb76170c11a0907ba960d670ada95d4e17fe5c",
        KDA: "fbf19f50e92653d2ed7ca7bb26a0028355802d2904b9a4273b7cbb48facb75e3",
    },
)

if __name__ == "__main__":
    sys.exit(Y.main(YAMA, __file__))
