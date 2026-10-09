#!/usr/bin/env python3
"""HAREM-TP3 loader padding and vocabulary unit (the vLLM 21d93d0d8 port of edits 1-3 of
the previous stack's patch-vllm-tp3.py).

At TP=3 the dimensions of GLM-5.3-Flash that do not divide by three are padded through
the config hook in the plugin (cuda_exl3._harem_tp3): MLA/KDA heads 64 -> 66, shared
expert 2048 -> 2304, DFlash2 drafter 32/8 -> 36/9. The shard of the top rank runs past
the end of the checkpoint tensor; those rows must be ZERO. This patch replaces
``narrow`` with ``_harem_pad_then_narrow`` at the six places where a TP shard is cut
from a checkpoint tensor (plain ``narrow`` when no padding is needed):

  parameter.py      column / merged column / qkv / row parallel (the v2 loader; EXL3
                    trellis+svh, bf16 weights, the bf16 half of the mixed method)
  weight_utils.py   row_parallel_weight_loader, sharded_weight_loader (KDA A_log,
                    dt_bias)
  vocab_parallel_embedding.py  padding unit 128 x tp (whole 128-blocks per rank):
                    154880 -> 155136 (3 x 404 x 128); unchanged at tp 1/2. The
                    previous stack's lcm(128, tp) rule left 302.5 blocks per rank at
                    tp=4 (a latent bug; it never ran at tp=4); tp 1-3 give the same.

EXL3 output padding is silenced by svh = 0 and input padding by suh = 0 (plugin); the
content of a zero-extended trellis is never read. Fails closed: it refuses when a whole
shard would be padding (the 64 -> 96 mistake).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harem_yama as Y  # noqa: E402

PARAM = "model_executor/parameter.py"
WU = "model_executor/model_loader/weight_utils.py"
VOCAB = "model_executor/layers/vocab_parallel_embedding.py"

YARDIMCI = '''

# --- HAREM-TP3 --------------------------------------------------------------
_HAREM_TP3_SEEN: set = set()


def _harem_pad_then_narrow(tensor, dim: int, start: int, length: int):
    """narrow() to this rank's TP shard, zero-extending the checkpoint tensor when
    a padded head / shared-expert count (cuda_exl3._harem_tp3) puts the shard past
    its stored end. Identical to tensor.narrow() when no padding is needed.
    Refuses a shard that would be padding only (the 64 -> 96 mistake)."""
    if tensor.dim() == 0:
        return tensor
    extra = start + length - tensor.size(dim)
    if extra <= 0:
        return tensor.narrow(dim, start, length)
    if extra >= length:
        raise ValueError(
            f"HAREM-TP3: refusing to zero-extend dim {dim} of a "
            f"{tuple(tensor.shape)} checkpoint tensor by {extra} for a shard of "
            f"{length} at {start}: the whole shard would be padding."
        )
    key = (tuple(tensor.shape), dim, extra, length)
    if key not in _HAREM_TP3_SEEN:
        _HAREM_TP3_SEEN.add(key)
        logger.info("HAREM-TP3 pad: %s dim %d +%d zeros (shard %d @ %d)",
                    tuple(tensor.shape), dim, extra, length, start)
    pads = [0, 0] * tensor.dim()
    pads[2 * (tensor.dim() - 1 - dim) + 1] = extra
    return torch.nn.functional.pad(tensor, tuple(pads)).narrow(dim, start, length)
# --- end HAREM-TP3 -----------------------------------------------------------
'''

D = Y.Duzen
DUZENLER = [
    D("P0-yardimci", PARAM,
      "logger = init_logger(__name__)\n\n\nclass BasevLLMParameter(Parameter):",
      "logger = init_logger(__name__)\n" + YARDIMCI + "\n\nclass BasevLLMParameter(Parameter):"),
    D("P1-sutun", PARAM,
      "    def load_column_parallel_weight(self, loaded_weight: torch.Tensor):\n"
      "        shard_size = self.data.shape[self.output_dim]\n"
      "        loaded_weight = loaded_weight.narrow(\n"
      "            self.output_dim, self.tp_rank * shard_size, shard_size\n"
      "        )\n",
      "    def load_column_parallel_weight(self, loaded_weight: torch.Tensor):\n"
      "        shard_size = self.data.shape[self.output_dim]\n"
      "        loaded_weight = _harem_pad_then_narrow(  # HAREM-TP3\n"
      "            loaded_weight, self.output_dim, self.tp_rank * shard_size, shard_size\n"
      "        )\n"),
    D("P2-birlesik-sutun", PARAM,
      "        param_data = param_data.narrow(self.output_dim, shard_offset, shard_size)\n"
      "        loaded_weight = loaded_weight.narrow(\n"
      "            self.output_dim, self.tp_rank * shard_size, shard_size\n"
      "        )\n",
      "        param_data = param_data.narrow(self.output_dim, shard_offset, shard_size)\n"
      "        loaded_weight = _harem_pad_then_narrow(  # HAREM-TP3\n"
      "            loaded_weight, self.output_dim, self.tp_rank * shard_size, shard_size\n"
      "        )\n"),
    D("P3-qkv", PARAM,
      "        param_data = param_data.narrow(self.output_dim, shard_offset, shard_size)\n"
      "        loaded_weight = loaded_weight.narrow(\n"
      "            self.output_dim, shard_id_int * shard_size, shard_size\n"
      "        )\n",
      "        param_data = param_data.narrow(self.output_dim, shard_offset, shard_size)\n"
      "        loaded_weight = _harem_pad_then_narrow(  # HAREM-TP3\n"
      "            loaded_weight, self.output_dim, shard_id_int * shard_size, shard_size\n"
      "        )\n"),
    D("P4-satir", PARAM,
      "    def load_row_parallel_weight(self, loaded_weight: torch.Tensor):\n"
      "        shard_size = self.data.shape[self.input_dim]\n"
      "        loaded_weight = loaded_weight.narrow(\n"
      "            self.input_dim, self.tp_rank * shard_size, shard_size\n"
      "        )\n",
      "    def load_row_parallel_weight(self, loaded_weight: torch.Tensor):\n"
      "        shard_size = self.data.shape[self.input_dim]\n"
      "        loaded_weight = _harem_pad_then_narrow(  # HAREM-TP3\n"
      "            loaded_weight, self.input_dim, self.tp_rank * shard_size, shard_size\n"
      "        )\n"),
    D("W1-satir-yukleyici", WU,
      "        start_idx = tp_rank * shard_size\n"
      "        loaded_weight = loaded_weight.narrow(shard_dim, start_idx, shard_size)\n",
      "        start_idx = tp_rank * shard_size\n"
      "        from vllm.model_executor.parameter import _harem_pad_then_narrow  # HAREM-TP3\n"
      "\n"
      "        loaded_weight = _harem_pad_then_narrow(\n"
      "            loaded_weight, shard_dim, start_idx, shard_size\n"
      "        )\n"),
    D("W2-dilimli-yukleyici", WU,
      "        start_idx = tp_rank * shard_size\n"
      "        loaded_weight = loaded_weight.narrow(shard_axis, start_idx, shard_size)\n",
      "        start_idx = tp_rank * shard_size\n"
      "        from vllm.model_executor.parameter import _harem_pad_then_narrow  # HAREM-TP3\n"
      "\n"
      "        loaded_weight = _harem_pad_then_narrow(\n"
      "            loaded_weight, shard_axis, start_idx, shard_size\n"
      "        )\n"),
    D("V1-sozcuk-birimi", VOCAB,
      "        self.num_embeddings = num_embeddings\n"
      "        self.padding_size = padding_size\n"
      "        self.org_vocab_size = org_num_embeddings or num_embeddings\n",
      "        self.num_embeddings = num_embeddings\n"
      "        self.padding_size = padding_size\n"
      "        # HAREM-TP3: every rank's vocab shard must be whole 128-blocks (an EXL3\n"
      "        # lm_head may not share a Hadamard block with its pad or another rank),\n"
      "        # so pad to a multiple of 128 x tp: 154880 -> 155136 = 3 x 404 x 128 at\n"
      "        # tp=3; GLM-5.3's vocab is unchanged at tp 1/2 (155136 at tp=4).\n"
      "        if self.tp_size > 1 and padding_size % (128 * self.tp_size):\n"
      "            from math import lcm as _harem_lcm\n"
      "\n"
      "            self.padding_size = _harem_lcm(padding_size, 128 * self.tp_size)\n"
      "        self.org_vocab_size = org_num_embeddings or num_embeddings\n"),
]

YAMA = Y.Yama(
    ad="yama-tp3-dolgu",
    aciklama="TP3 yükleyici sıfır uzatma + sözcük dağarcığı lcm(128, tp)",
    duzenler=DUZENLER,
    taban={
        PARAM: "1a4bbd7400fd1ba79e8e1e8d19666cd22881ec83c35e6e01b3fdc78faee23171",
        WU: "7d2573931b9684147b17018c1f6de318eea1c105556ffea03154201a72418afd",
        VOCAB: "01576027a0262d2135800aab0ddb2fe2701fae2177c33f8ad2426240b6e04073",
    },
)

if __name__ == "__main__":
    sys.exit(Y.main(YAMA, __file__))
