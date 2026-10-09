#!/usr/bin/env python3
"""HAREM-SM12 items 2+3: make the K-pool top-k buffer and its reader agree.

The 21d93d0d8 port of the previous stack's patch-kpool-init.py (Zeuss5/cuda-exl3 issue #6;
the fix is tpurtell's port-glm53-sm12-stability.py, Apache-2.0). Upstream moved the kpool
indexer to ``models/glm5next/nvidia/sparse_indexer.py``; both ``torch.empty`` sites are
unchanged there [verified at :348 and :597].

  K2a/K2b  ``pool_topk = torch.empty(...)`` -> ``torch.full(..., -1)``: the top-k kernels
           write only min(select_k, valid) ids per row; the remaining words are stale
           memory that would expand into real cache rows.
  K3       ``_expand_pools_and_append_tail_kernel`` bounded the pool id only from below;
           the upper bound (pid < pool_len) is added.
The writer and the reader are in the same patch: neither is applied without the other.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harem_yama as Y  # noqa: E402

IDX = "models/glm5next/nvidia/sparse_indexer.py"
KP = "models/glm5next/nvidia/ops/kpool_compress.py"

D = Y.Duzen
DUZENLER = [
    D("K2a-prefill", IDX,
      "            if index_kpool > 1:\n"
      "                pool_topk = torch.empty(\n"
      "                    (num_rows, select_k), dtype=torch.int32, device=logits.device\n"
      "                )\n",
      "            if index_kpool > 1:\n"
      "                # HAREM-SM12 item 2: top_k_per_row_prefill writes only\n"
      "                # min(select_k, valid) ids per row; an empty() tail is stale\n"
      "                # memory that would expand into real cache rows.\n"
      "                pool_topk = torch.full(\n"
      "                    (num_rows, select_k), -1, dtype=torch.int32, device=logits.device\n"
      "                )\n"),
    D("K2b-decode", IDX,
      "        if index_kpool > 1:\n"
      "            pool_topk = torch.empty(\n"
      "                (num_rows, select_k), dtype=torch.int32, device=logits.device\n"
      "            )\n",
      "        if index_kpool > 1:\n"
      "            # HAREM-SM12 item 2: the decode top-k writes only min(select_k,\n"
      "            # valid) ids per row; see the prefill site above.\n"
      "            pool_topk = torch.full(\n"
      "                (num_rows, select_k), -1, dtype=torch.int32, device=logits.device\n"
      "            )\n"),
    D("K3-okuyucu", KP,
      "    hist_out = tl.where(pid >= 0, hist_val, -1)\n",
      "    # HAREM-SM12 item 3: bound the pool id from above too (pool_len is in\n"
      "    # scope); an out-of-range id must become -1, not a real token index.\n"
      "    hist_out = tl.where((pid >= 0) & (pid < pool_len), hist_val, -1)\n"),
]

YAMA = Y.Yama(
    ad="yama-kpool-init",
    aciklama="K-pool top-k tamponu -1 ile başlat + okuyucuya üst sınır",
    duzenler=DUZENLER,
    taban={
        IDX: "f2843c9beba54707b79004abd15c4970d47ef40f486e930935b498cb615b134d",
        KP: "adfbc6810c585580a6ddb073706302160a21468e15dfd47020350a7b1c3286d9",
    },
)

if __name__ == "__main__":
    sys.exit(Y.main(YAMA, __file__))
