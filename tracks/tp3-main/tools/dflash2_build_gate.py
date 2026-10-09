"""HAREM DFlash2 image gate -- the new stack (vLLM 21d93d0d8 + 14 patches).

The new-tree counterpart of the previous stack image's ``/opt/harem/dflash2-gate.py``
(exl3-zeus:754421f). The old gate's header holds unchanged: if any hook checked here is
missing, the engine STARTS, produces correct-looking text, and only the acceptance rate
silently drops. That is why it runs when the image is built:
    docker run --rm --network none --entrypoint python3 <image> /opt/harem/dflash2-gate.py
It runs on the CPU (no GPU needed). The patches must already be APPLIED in the image (after
harem-prelude.sh in KURU mode).

Sections:
  A  upstream DFlash2 (the b389ac294 part of the previous port; on main at 21d93d0d8) -- the
     new paths of the old asserts
  B  target side (yama-dflash-eagle3) -- the glm5next/nvidia/model.py part of the previous
     port; structure + functional aux capture with dummy layers
  C  KV group (yama-dflash-kvgrup) -- the kv_cache_utils.py part of the previous port +
     prefixhit; structure + functional grouping / byte check with GLM-5.3-shaped specs
"""

import inspect
import os
from types import SimpleNamespace

# ------------------------------------------------------------------ A. upstream DFlash2
import vllm.model_executor.models.qwen3_dflash2 as m

for ad in ("DFlash2Qwen3ForCausalLM", "DFlash2Qwen3Model", "DFlash2Qwen3DecoderLayer",
           "CandidateSelector", "DFlashGroupedConv"):
    assert hasattr(m, ad), f"{ad} missing"
assert m.DFlash2Qwen3Model.decoder_layer_cls is m.DFlash2Qwen3DecoderLayer, "decoder_layer_cls hook not wired"
assert m.DFlash2Qwen3ForCausalLM.model_cls is m.DFlash2Qwen3Model, "model_cls hook not wired"
assert hasattr(m.DFlash2Qwen3ForCausalLM, "compute_candidates"), "compute_candidates missing"
# instead of the previous port's HAREM guard (_harem_check_port_assumptions): yama-dflash-tp3
#   (2) the TP3 pad guard (zero-row proof on a tagged draft) + (1) refusal of a quantized draft (DF3)
_src_q2 = inspect.getsource(m)
assert "HAREM-TP3 drafter pad" in _src_q2 and "_harem_draft_verify" in _src_q2, "yama-dflash-tp3 guard missing"
assert "harem_tp_pad" in inspect.getsource(m.DFlash2Qwen3ForCausalLM.load_weights), "pad guard not wired"
_init = inspect.getsource(m.DFlash2Qwen3Model.__init__)
assert "if self.quant_config is not None:" in _init and "NotImplementedError" in _init \
    and _init.index("quant_config is not None") < _init.index("CandidateSelector("), "quantized drafter not refused"

from vllm.model_executor.models.registry import _SPECULATIVE_DECODING_MODELS as R  # noqa: E402

assert R["DFlash2DraftModel"] == ("qwen3_dflash2", "DFlash2Qwen3ForCausalLM"), R.get("DFlash2DraftModel")
assert R["DFlashDraftModel"] == ("qwen3_dflash", "DFlashQwen3ForCausalLM"), "DFlash v1 entry disturbed"

import vllm.v1.worker.gpu.spec_decode.dflash2.speculator as s  # noqa: E402
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator  # noqa: E402

assert hasattr(s, "DFlash2Speculator"), "DFlash2Speculator missing"
assert hasattr(s, "_selector_walk_kernel"), "selector walk kernel missing"
assert hasattr(s, "_cache_draft_logits_kernel"), "draft logits cache kernel missing"
assert issubclass(s.DFlash2Speculator, DFlashSpeculator), "DFlash2Speculator must extend DFlashSpeculator"
assert s.DFlash2Speculator.draft_logits_spec is not DFlashSpeculator.draft_logits_spec, (
    "DFlash2 must override draft_logits_spec (fp32/-inf), else the selector walk "
    "and the rejection sampler read different distributions")
assert "aux_hidden_states" in inspect.signature(DFlashSpeculator.propose).parameters, "propose aux param"

from vllm.v1.worker.gpu.sample.gumbel import gumbel_noised_argmax  # noqa: E402,F401
from vllm.model_executor.layers.logits_processor import LogitsProcessor  # noqa: E402

assert hasattr(LogitsProcessor, "get_top_k_tokens"), "get_top_k_tokens missing"
_sig = inspect.signature(LogitsProcessor.get_top_k_tokens)
assert list(_sig.parameters)[:4] == ["self", "lm_head", "hidden_states", "k"], list(_sig.parameters)
assert _sig.parameters["return_log_probs"].default is False, "get_top_k_tokens default changed"
assert "return_log_probs" not in inspect.getsource(m.DFlash2Qwen3ForCausalLM.compute_candidates)
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator  # noqa: E402

assert hasattr(DraftModelSpeculator, "draft_logits_spec"), "draft_logits_spec hook missing"
from vllm.v1.worker.gpu.spec_decode import init_speculator  # noqa: E402

_src = inspect.getsource(init_speculator)
assert "DFlash2Speculator" in _src and "DFlash2DraftModel" in _src, "init_speculator does not dispatch to DFlash2"

# V2 model runner: the previous port forced it with _is_dflash2_draft; at 21d93d0d8 V2 is the
# default and a candidate-head DFlash draft is recognised by _is_dflash_candidate_draft. If it fell
# back to V1 the candidate selector would never be called (acceptance silently drops to DFlash1).
# Run-time evidence: "Using V2 Model Runner".
from vllm.config.vllm import VllmConfig  # noqa: E402

assert hasattr(VllmConfig, "use_v2_model_runner"), "use_v2_model_runner missing"
assert hasattr(VllmConfig, "_is_dflash_candidate_draft"), "_is_dflash_candidate_draft missing"
assert "DFlash2DraftModel" in inspect.getsource(VllmConfig._is_dflash_candidate_draft)
assert VllmConfig._is_dflash_candidate_draft(SimpleNamespace(speculative_config=SimpleNamespace(
    method="dflash", draft_model_config=SimpleNamespace(architectures=["DFlash2DraftModel"])))), \
    "DFlash2 draft not recognised as a candidate-head draft"
# The fall back to V1 is not SILENT: if V1 is chosen, a candidate-head DFlash draft is rejected with a ValueError
assert "_is_dflash_candidate_draft()" in inspect.getsource(VllmConfig._get_v1_model_runner_unsupported_features)
assert "raise ValueError" in inspect.getsource(VllmConfig._validate_v1_model_runner)
assert "self._validate_v1_model_runner()" in inspect.getsource(VllmConfig.__post_init__)
assert os.environ.get("VLLM_USE_V2_MODEL_RUNNER", "") not in ("0", "false", "False"), \
    "VLLM_USE_V2_MODEL_RUNNER=0 would drop DFlash2 to the V1 runner"

# #54373: the draft reads its RoPE layout from ITS OWN config (default neox). The previous port was
# neox as well (its optional rope patch was NOT applied). If this line changes, acceptance silently collapses.
import vllm.model_executor.models.qwen3_dflash as q1  # noqa: E402
import vllm.config.speculative as _sc  # noqa: E402

# the draft's quantization is NOT copied from the target (for dflash: quantization=self.quantization)
assert "quantization=self.quantization," in inspect.getsource(_sc)

assert 'is_neox_style = getattr(config, "is_neox_style", True)' in inspect.getsource(q1), \
    "DFlash draft RoPE layout source changed (#54373)"

# ------------------------------------------------------------------ B. target side
import torch  # noqa: E402

import vllm.models.glm5next.common.model as G  # noqa: E402
from vllm.model_executor.models.interfaces import EagleModelMixin, supports_eagle3  # noqa: E402

assert "EagleModelMixin" in [c.__name__ for c in G.Glm5NextModel.__mro__], \
    "Glm5NextModel must inherit EagleModelMixin (SupportsEagle3 asserts on it)"
assert hasattr(G.Glm5NextModel, "_set_aux_hidden_state_layers"), "mixin hook missing"
assert hasattr(G.Glm5NextModel, "_prepare_aux_hidden_state"), "aux hidden state builder missing"
for _cls in (G.Glm5NextForCausalLM, G.Glm5NextForConditionalGeneration):
    assert "SupportsEagle3" in [c.__name__ for c in _cls.__mro__], f"{_cls.__name__} must declare SupportsEagle3"
    assert hasattr(_cls, "set_aux_hidden_state_layers"), f"{_cls.__name__}.set_aux_hidden_state_layers missing"
assert hasattr(G.Glm5NextForConditionalGeneration, "get_language_model"), \
    "wrapper must expose get_language_model() for the EAGLE3 protocol to resolve"
assert "hc_contract(aux_hidden_state, layer.n)" in inspect.getsource(G.Glm5NextModel._prepare_aux_hidden_state), \
    "mHC aux must be contracted for DFlash"


def _ic(n=6):
    x = G.Glm5NextModel.__new__(G.Glm5NextModel)
    torch.nn.Module.__init__(x)
    x.start_layer, x.end_layer, x.is_sequence_parallel, x.dflash_capture = 0, n, False, True
    return x


_cg = G.Glm5NextForConditionalGeneration.__new__(G.Glm5NextForConditionalGeneration)
torch.nn.Module.__init__(_cg)
_lm = G.Glm5NextForCausalLM.__new__(G.Glm5NextForCausalLM)
torch.nn.Module.__init__(_lm)
_lm.model = _ic()
_cg.language_model = _lm
assert supports_eagle3(_cg)
from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import set_eagle3_aux_hidden_state_layers  # noqa: E402

set_eagle3_aux_hidden_state_layers(_cg, SimpleNamespace(draft_model_config=SimpleNamespace(
    hf_config=SimpleNamespace(dflash_config={"target_layer_ids": [5, 14, 24, 33, 42]}))))
assert _lm.model.aux_hidden_state_layers == (6, 15, 25, 34, 43), _lm.model.aux_hidden_state_layers


class _K:
    def __init__(self, i, son):
        g = torch.Generator().manual_seed(i)
        self.mhc, self.n, self.son = True, 4, son
        self.x = torch.randn(3, 8, generator=g)
        self.r = torch.randn(3, 4, 8, generator=g)
        self.p = torch.rand(3, 4, 1, generator=g)
        self.c = torch.rand(3, 4, 4, generator=g)

    def hc_post(self, x, r, p, c):
        from vllm.model_executor.kernels.mhc.torch import mhc_post_torch
        return mhc_post_torch(x, r, p, c)

    def __call__(self, *_):
        if self.son:
            return G.hc_contract(self.hc_post(self.x, self.r, self.p, self.c), 4), None, None, None
        return self.x, self.r, self.p, self.c


_orj_pp = G.get_pp_group
G.get_pp_group = lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True)
try:
    _m = _ic()
    _m._active_layers = [_K(i, i == 5) for i in range(6)]
    _m.norm = torch.nn.Identity()
    _m._set_aux_hidden_state_layers((2, 3, 4, 5, 6))
    _h, _aux = G.Glm5NextModel.forward(_m, None, torch.arange(3), None, inputs_embeds=torch.zeros(3, 8))
    assert len(_aux) == 5 and all(a.shape == (3, 8) for a in _aux)
    for _j, _L in enumerate((2, 3, 4, 5)):
        _k = _m._active_layers[_L - 1]
        assert torch.equal(_aux[_j], G.hc_contract(_k.hc_post(_k.x, _k.r, _k.p, _k.c), 4)), _L
    assert torch.equal(_aux[4], _h)
    _m2 = _ic()
    _m2._active_layers = _m._active_layers
    _m2.norm = torch.nn.Identity()
    assert isinstance(G.Glm5NextModel.forward(_m2, None, torch.arange(3), None,
                                              inputs_embeds=torch.zeros(3, 8)), torch.Tensor)
finally:
    G.get_pp_group = _orj_pp

# ------------------------------------------------------------------ C. KV group
import vllm.v1.core.kv_cache_utils as K  # noqa: E402
from vllm.v1.kv_cache_interface import KpoolTailSpec, MambaSpec, MLAAttentionSpec, SlidingWindowSpec  # noqa: E402
from vllm.v1.kv_cache_layout import KVCacheLayout  # noqa: E402

for ad in ("_harem_partition_dflash_draft_specs", "_harem_draft_bytes_per_block",
           "_harem_annotate_draft_eagle_groups"):
    assert hasattr(K, ad), f"{ad} missing"
assert "_harem_partition_dflash_draft_specs" in inspect.getsource(K.get_kv_cache_groups), \
    "get_kv_cache_groups does not partition the DFlash draft"
assert "draft_groups," in inspect.getsource(K._glm5_next_tensor_layout), \
    "_glm5_next_tensor_layout does not return draft groups"
for _fn in (K._get_kv_cache_bytes_per_block, K.get_kv_cache_config_from_groups,
            K._max_memory_usage_bytes_from_groups):
    assert "draft_groups" in inspect.getsource(_fn), f"{_fn.__name__} does not account for the draft group"

_B = 3328


def _mla(t=1):
    return MLAAttentionSpec(block_size=_B, num_kv_heads=1, head_size=512 if t == 1 else 132,
                            dtype=torch.uint8, tokens_per_state=t)


def _specler(taslak=True):
    sp = {}
    for i in range(6):
        p = f"language_model.model.layers.{i}.self_attn"
        if i == 3:
            sp[f"{p}.attn"] = _mla()
            sp[f"{p}.indexer.k_cache"] = _mla(4)
            sp[f"{p}.indexer.tail_cache"] = KpoolTailSpec(
                block_size=16, num_kv_heads=2, head_size=128, head_size_v=0,
                dtype=torch.bfloat16, sliding_window=16, dcp_sharded=False)
        else:
            sp[p] = MambaSpec(block_size=_B, shapes=((3, 4096), (8, 128, 128)),
                              dtypes=(torch.bfloat16, torch.float32), page_size_padded=_mla().page_size_bytes)
    if taslak:
        for j in range(5):
            sp[f"draft_model.model.layers.{6 + j}.self_attn.attn"] = SlidingWindowSpec(
                block_size=256, num_kv_heads=8, head_size=128, dtype=torch.float8_e4m3fn, sliding_window=2048)
    return sp


def _vc(yontem):
    spec = None if yontem is None else SimpleNamespace(
        method=yontem, use_eagle=lambda: True, use_eagle_block_drop=lambda: True,
        use_dflash=lambda: yontem == "dflash", num_speculative_tokens=7)
    return SimpleNamespace(
        speculative_config=spec,
        parallel_config=SimpleNamespace(pipeline_parallel_size=1, tensor_parallel_size=1,
                                        decode_context_parallel_size=1),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        model_config=SimpleNamespace(get_num_layers=lambda p: 6, get_total_num_hidden_layers=lambda: 6,
                                     max_model_len=16384, hf_config=SimpleNamespace(model_type="glm5_next")),
        cache_config=SimpleNamespace(num_gpu_blocks_override=None, prefix_cache_retention_interval=None,
                                     mamba_cache_mode="align", block_size=_B,
                                     get_resolved_kv_cache_layout=lambda: KVCacheLayout.LBHNC),
        attention_config=SimpleNamespace(hisparse_config=None))


_eski_env = os.environ.get("HAREM_PREFIX_HIT")
os.environ["HAREM_PREFIX_HIT"] = "1"
try:
    _hedef = K.get_kv_cache_groups(_vc(None), _specler(False))
    _hepsi = K.get_kv_cache_groups(_vc("dflash"), _specler(True))
finally:
    if _eski_env is None:
        os.environ.pop("HAREM_PREFIX_HIT", None)
    else:
        os.environ["HAREM_PREFIX_HIT"] = _eski_env
_nh = len(_hedef)
assert [(g.layer_names, g.kv_cache_spec) for g in _hepsi[:_nh]] == [(g.layer_names, g.kv_cache_spec) for g in _hedef], \
    "target grouping changed when a DFlash draft is present"
_tg = _hepsi[_nh:]
assert _tg and all(type(g.kv_cache_spec) is SlidingWindowSpec for g in _tg), "draft group missing"
assert [g.is_eagle_group for g in _hepsi] == [False] * _nh + [True] * len(_tg), "prefix-hit flags wrong"
_d = K._glm5_next_tensor_layout(_hepsi)
assert _d is not None and _d[8] == _tg, "GLM-5.3 layout not recognised with the draft group"
_dp = 5 * _tg[0].kv_cache_spec.page_size_bytes
assert K._get_kv_cache_bytes_per_block(_hepsi) == K._get_kv_cache_bytes_per_block(_hedef) + _dp
_c = K.get_kv_cache_config_from_groups(_vc("dflash"), _hepsi, 50 * K._get_kv_cache_bytes_per_block(_hepsi))
_isim = {t.layers[0] for t in _c.kv_cache_tensors}
assert all(n in _isim for g in _tg for n in g.layer_names), "draft tensors unallocated"

print("DFLASH2 BUILD GATE (21d93d0d8 + 14 yama): OK")
