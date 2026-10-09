#!/usr/bin/env python3
"""HAREM-TP3 swblock: block size of the independently grouped sliding window (the DFlash2
drafter).

The 21d93d0d8 port of the previous stack's patch-swblock-tp3.py (the call gained
kv_cache_spec; re-anchored). With ``HAREM_SW_BLOCK_SIZE=<n>`` the kernel block of the SW
group becomes n (it must be a multiple of the backend's smallest kernel block); empty =
upstream.

NOTE (the behaviour CHANGED direction at 21d93d0d8): 487ecf187, with no page budget,
chose the SMALLEST kernel block and the knob enlarged it to 256. The new upstream takes
the budget as primary block x token bytes and chooses the LARGEST of the sizes the
backend reports that fits (attention.py:_largest_kernel_block_within) [verified: code +
test]: Triton (MultipleOf(16)) picks 3,328 when the primary block is 3,328, FlashInfer on
SM12x ([16, 32, 64]) picks 64. 3,328 is the geometry of the "SW block 3328" arm we
REJECTED on the previous stack (prefix hit 93.5 % -> 89.4 %, KV 7.02 M -> 4.78 M; an old
record), and 64 lowers the fine grid from 256 to 64. The drafter's real backend only shows
at boot; the knob pins the production geometry (256) in both cases.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harem_yama as Y  # noqa: E402

ATT = "model_executor/layers/attention/attention.py"

D = Y.Duzen
DUZENLER = [
    D("SW1-blok", ATT,
      "            sw_block_size = _largest_kernel_block_within(\n"
      "                self.attn_backend,\n"
      "                sw_per_token,\n"
      "                page_budget,\n"
      "                block_size,\n"
      "                kv_cache_spec,\n"
      "            )\n",
      "            sw_block_size = _largest_kernel_block_within(\n"
      "                self.attn_backend,\n"
      "                sw_per_token,\n"
      "                page_budget,\n"
      "                block_size,\n"
      "                kv_cache_spec,\n"
      "            )\n"
      "            # HAREM-TP3: an independently grouped SW spec (DFlash2 drafter) is\n"
      "            # pinned to HAREM_SW_BLOCK_SIZE when set; it must be a multiple of\n"
      "            # the backend's smallest kernel block. Unset = upstream choice.\n"
      "            import os as _harem_os\n"
      "\n"
      "            _harem_sw = int(_harem_os.environ.get(\"HAREM_SW_BLOCK_SIZE\", \"0\") or 0)\n"
      "            if _harem_sw:\n"
      "                from vllm.v1.attention.backend import MultipleOf\n"
      "\n"
      "                _harem_base = min(\n"
      "                    s.base if isinstance(s, MultipleOf) else s\n"
      "                    for s in self.attn_backend.get_supported_kernel_block_sizes(\n"
      "                        kv_cache_spec\n"
      "                    )\n"
      "                )\n"
      "                if _harem_sw % _harem_base:\n"
      "                    raise ValueError(\n"
      "                        f\"HAREM_SW_BLOCK_SIZE={_harem_sw} is not a multiple of \"\n"
      "                        f\"{self.attn_backend.get_name()}'s kernel block {_harem_base}\"\n"
      "                    )\n"
      "                sw_block_size = _harem_sw\n"),
]

YAMA = Y.Yama(
    ad="yama-swblock",
    aciklama="DFlash2 taslağının SW grubu blok boyu HAREM_SW_BLOCK_SIZE ile",
    duzenler=DUZENLER,
    taban={ATT: "bde77677fd0e95ea09c6a580dfaf09b57741e8183e00cb106e2d1cc64d8e03f2"},
)

if __name__ == "__main__":
    sys.exit(Y.main(YAMA, __file__))
