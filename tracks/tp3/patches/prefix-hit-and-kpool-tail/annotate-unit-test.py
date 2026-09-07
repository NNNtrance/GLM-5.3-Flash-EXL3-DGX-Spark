import torch
from vllm.v1.core.kv_cache_utils import _harem_annotate_draft_eagle_groups as ann
from vllm.v1.kv_cache_interface import (KVCacheGroupSpec, SlidingWindowSpec,
                                        MLAAttentionSpec)
sw = SlidingWindowSpec(block_size=256, num_kv_heads=3, head_size=128,
                       dtype=torch.bfloat16, sliding_window=4096)
mla = MLAAttentionSpec(block_size=3328, num_kv_heads=1, head_size=576,
                       dtype=torch.bfloat16)
tgt = [KVCacheGroupSpec(["l0"], mla)]
drf = [KVCacheGroupSpec(["l45"], sw)]
ann(tgt, drf)
print("target flagged:", tgt[0].is_eagle_group, " draft flagged:", drf[0].is_eagle_group)
bad = [KVCacheGroupSpec(["lx"], mla)]
try:
    ann([KVCacheGroupSpec(["l0"], mla)], bad); print("FAIL: accepted a non-SWA drafter")
except ValueError:
    print("refused non-SWA drafter: OK")

# corrected third case: a TARGET group that is already flagged must be refused
t2 = [KVCacheGroupSpec(["l0"], mla)]
t2[0].is_eagle_group = True
try:
    ann(t2, [KVCacheGroupSpec(["l45"], sw)]); print("FAIL: accepted an already-flagged target")
except ValueError:
    print("refused already-flagged target: OK")
