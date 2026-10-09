#!/usr/bin/env python3
"""HAREM-TP3 DFlash2 drafter guard (a rewrite of the previous stack's patch-dflash-tp3.py).

Half of the previous guard changed the head-count check of OUR DFlash2 port; upstream
``qwen3_dflash2.py`` has no such check (verified by hand). The drafter padding is now
offline and tagged: ``cuda_exl3._harem_tp3.apply_draft_plan`` writes
``harem_tp_pad = {"tp": 3, "num_attention_heads": [32, 36], "num_key_value_heads": [8, 9]}``
into the drafter's config (tools/sidecar_kur.py, mode ``taslak``); the plugin's config
hook refuses a drafter tagged for another tp.

This patch wraps ``DFlash2Qwen3ForCausalLM.load_weights``; when the tag is present:
  * the q_proj/k_proj row counts in the streaming checkpoint must equal the tag's stock
    head counts (x head_dim) -> "config = checkpoint + pad";
  * the GQA ratio must hold and every rank must keep at least one real head;
  * AFTER loading, this rank's pad rows (the q/k/v blocks of qkv_proj) and the pad input
    columns of o_proj must be EXACTLY ZERO (a proof, not an assumption).
Without the tag it is exactly upstream. There is deliberately no escape hatch (the old
HAREM_TP3_DRAFT_PAD_CHECK=warn).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harem_yama as Y  # noqa: E402

DF = "model_executor/models/qwen3_dflash2.py"

YARDIMCI = '''# --- HAREM-TP3 drafter pad guard --------------------------------------------
_HAREM_DRAFT_KEYS = ("self_attn.q_proj.weight", "self_attn.k_proj.weight")


def _harem_draft_watch(weights, seen: dict):
    """Record the drafter checkpoint's q/k row counts as the tensors stream by."""
    for name, tensor in weights:
        for key in _HAREM_DRAFT_KEYS:
            if name.endswith(key):
                seen.setdefault(key, set()).add(int(tensor.shape[0]))
        yield name, tensor


def _harem_draft_verify(model, tag: dict, seen: dict) -> None:
    """config == checkpoint + an exact zero pad, proven on this rank's weights."""
    from vllm.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )
    from vllm.logger import init_logger

    tp = get_tensor_model_parallel_world_size()
    rank = get_tensor_model_parallel_rank()
    cfg = model.config
    heads, kv = int(cfg.num_attention_heads), int(cfg.num_key_value_heads)
    stock_q = int(tag.get("num_attention_heads", [heads])[0])
    stock_kv = int(tag.get("num_key_value_heads", [kv])[0])
    hd = int(getattr(cfg, "head_dim", 0) or cfg.hidden_size // heads)
    want = {_HAREM_DRAFT_KEYS[0]: {stock_q * hd}, _HAREM_DRAFT_KEYS[1]: {stock_kv * hd}}
    problems = []
    if int(tag.get("tp", -1)) != tp:
        problems.append(f"padded for tp={tag.get('tp')}, serving tp={tp}")
    if seen != want:
        problems.append(f"checkpoint rows {seen} != tag stock {want}")
    if heads * stock_kv != kv * stock_q or heads < stock_q or kv < stock_kv:
        problems.append(f"{stock_q}/{stock_kv} -> {heads}/{kv} is not a GQA-keeping pad")
    q_loc, kv_loc = heads // tp, max(1, kv // tp)
    q_real = min(q_loc, max(0, stock_q - rank * q_loc))
    kv_real = min(kv_loc, max(0, stock_kv - rank * kv_loc))
    if not q_real or not kv_real:
        problems.append(f"rank {rank} would own padding only")
    for i, layer in enumerate(model.model.layers if not problems else ()):
        attn = layer.self_attn
        w = attn.qkv_proj.weight
        for label, base, real, total in (
            ("q", 0, q_real, q_loc),
            ("k", attn.q_size, kv_real, kv_loc),
            ("v", attn.q_size + attn.kv_size, kv_real, kv_loc),
        ):
            if real < total and int(w[base + real * hd: base + total * hd].count_nonzero()):
                problems.append(f"layer {i} qkv_proj.{label} pad rows non-zero")
        if q_real < q_loc and int(
            attn.o_proj.weight[:, q_real * hd: q_loc * hd].count_nonzero()
        ):
            problems.append(f"layer {i} o_proj pad columns non-zero")
    if problems:
        raise ValueError(
            f"HAREM-TP3 drafter pad check failed on rank {rank}/{tp} ({tag}): "
            + "; ".join(problems[:6])
        )
    init_logger(__name__).info(
        "HAREM-TP3 drafter pad verified on rank %d/%d: %d/%d -> %d/%d, "
        "%d padded q head(s) and %d padded kv head(s) here are exactly zero",
        rank, tp, stock_q, stock_kv, heads, kv, q_loc - q_real, kv_loc - kv_real)
# --- end HAREM-TP3 -------------------------------------------------------------


EntryClass = DFlash2Qwen3ForCausalLM
'''

D = Y.Duzen
DUZENLER = [
    D("DF1-sarmal", DF,
      "    def compute_candidates(\n"
      "        self, hidden_states: torch.Tensor\n"
      "    ) -> tuple[torch.Tensor, torch.Tensor]:\n",
      "    def load_weights(self, weights):\n"
      "        # HAREM-TP3: a drafter padded offline (cuda_exl3._harem_tp3,\n"
      "        # harem_tp_pad tag) must load its checkpoint heads + an exact zero pad.\n"
      "        tag = getattr(self.config, \"harem_tp_pad\", None)\n"
      "        if not tag:\n"
      "            return super().load_weights(weights)\n"
      "        seen: dict = {}\n"
      "        loaded = super().load_weights(_harem_draft_watch(weights, seen))\n"
      "        _harem_draft_verify(self, tag, seen)\n"
      "        return loaded\n"
      "\n"
      "    def compute_candidates(\n"
      "        self, hidden_states: torch.Tensor\n"
      "    ) -> tuple[torch.Tensor, torch.Tensor]:\n"),
    D("DF2-yardimci", DF, "EntryClass = DFlash2Qwen3ForCausalLM\n", YARDIMCI),
    # The previous DFlash2 port's _harem_check_port_assumptions (1) guard -- unconditionally
    # on in the previous stack; absent at 21d93d0d8 and in the first version of this patch.
    # This stack serves only the BF16 drafter (get_draft_quant_config -> None); a quantized
    # drafter would build the candidate-selector projection unquantized and the acceptance
    # would silently drop. A trip wire (behaviour does not change).
    D("DF3-nicem-reddi", DF,
      "        draft_config = self.config.dflash_config\n"
      "        self.input_embedding_scale = float(\n",
      "        # HAREM-DFLASH2: quantized drafter refused (DFlash2 port guard, production).\n"
      "        if self.quant_config is not None:\n"
      "            raise NotImplementedError(\n"
      "                \"HAREM-DFLASH2: quantized DFlash2 drafter refused (draft quant config \"\n"
      "                f\"{type(self.quant_config).__name__}); this stack serves only the BF16 \"\n"
      "                \"drafter -- the candidate-selector projection would run unquantized\"\n"
      "            )\n"
      "        draft_config = self.config.dflash_config\n"
      "        self.input_embedding_scale = float(\n"),
]

YAMA = Y.Yama(
    ad="yama-dflash-tp3",
    aciklama="DFlash2 taslağı TP3 dolgu bekçisi (etiket + sıfır satır kanıtı) + nicemli taslak reddi (4b)",
    duzenler=DUZENLER,
    taban={DF: "11ac6ba8c6db702375970e68cd042b98783a5c43b87f9bfa27e88463316bf499"},
)

if __name__ == "__main__":
    sys.exit(Y.main(YAMA, __file__))
