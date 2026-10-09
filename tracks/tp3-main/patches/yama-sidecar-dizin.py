#!/usr/bin/env python3
"""HAREM sidecar index filter (new in the 21d93d0d8 port).

vLLM yields EVERY tensor of EVERY safetensors file in a checkpoint folder
(weight_utils.safetensors_weights_iterator; the index only filters the file list). The
HAREM sidecar (tools/sidecar_kur.py) symlinks the original shards and writes the tensors
that are split offline (KDA qkv_proj -> q/k/v_proj, conv1d -> q/k/v_conv1d) into a
separate file; the original fused tensors, and the dead fused ``attn.qkv`` weights of
the vision tower, remain inside the symlinked shards.

This patch makes the index AUTHORITATIVE for a checkpoint whose index carries
``metadata.harem_sidecar``:
  * a tensor present in the index's ``weight_map`` is yielded,
  * a tensor listed in the index's ``harem_sidecar.dusurulen`` ("superseded") list is
    skipped (and counted),
  * a tensor in neither is an ERROR (the sidecar and the shards disagree),
  * when the iteration ends, the skipped count must equal the declared count.
Exactly upstream for every checkpoint whose index has no ``harem_sidecar``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harem_yama as Y  # noqa: E402

DL = "model_executor/model_loader/default_loader.py"

YARDIMCI = '''

def _harem_sidecar_filter(weights_iterator, hf_folder, index_file):
    """HAREM-SIDECAR: make a converted HAREM checkpoint's index authoritative.

    Inert unless the index metadata carries "harem_sidecar" (written by
    tools/sidecar_kur.py). Then a tensor listed in weight_map is
    yielded, one declared in harem_sidecar.dusurulen is skipped, and anything
    else is an error; the skipped count must equal the declared one.
    """
    import json

    try:
        with open(os.path.join(hf_folder, index_file)) as f:
            index = json.load(f)
    except (OSError, ValueError, TypeError):
        return weights_iterator
    meta = (index.get("metadata") or {}).get("harem_sidecar")
    if not meta:
        return weights_iterator
    keep, drop = index["weight_map"], frozenset(meta["dusurulen"])

    def _filtered():
        skipped = 0
        for name, tensor in weights_iterator:
            if name in keep:
                yield name, tensor
            elif name in drop:
                skipped += 1
            else:
                raise ValueError(
                    f"HAREM-SIDECAR: {name} is neither in {index_file} nor declared "
                    "superseded; the sidecar does not match its shards."
                )
        if skipped != len(drop):
            raise ValueError(
                f"HAREM-SIDECAR: {skipped} superseded tensors seen, {len(drop)} "
                "declared; the sidecar does not match its shards."
            )
        logger.info("HAREM-SIDECAR: %d superseded tensors skipped (%s)",
                    skipped, meta.get("kaynak"))

    return _filtered()
'''

D = Y.Duzen
DUZENLER = [
    D("S1-yardimci", DL,
      "logger = init_logger(__name__)\n\n\nclass DefaultModelLoader(BaseModelLoader):",
      "logger = init_logger(__name__)\n" + YARDIMCI
      + "\n\nclass DefaultModelLoader(BaseModelLoader):"),
    D("S2-suzgec", DL,
      "            self.counter_before_loading_weights = time.perf_counter()\n"
      "        # Apply the prefix.\n",
      "            self.counter_before_loading_weights = time.perf_counter()\n"
      "        if use_safetensors:  # HAREM-SIDECAR: a converted checkpoint's index rules\n"
      "            weights_iterator = _harem_sidecar_filter(\n"
      "                weights_iterator, hf_folder, index_file\n"
      "            )\n"
      "        # Apply the prefix.\n"),
]

YAMA = Y.Yama(
    ad="yama-sidecar-dizin",
    aciklama="HAREM sidecar dizinini yetkili yap (çevrimdışı bölünmüş tensörler)",
    duzenler=DUZENLER,
    taban={DL: "d17ab54969489de2338bea88190b4af78c8a1f66e866796f612ff554eea3cca9"},
)

if __name__ == "__main__":
    sys.exit(Y.main(YAMA, __file__))
