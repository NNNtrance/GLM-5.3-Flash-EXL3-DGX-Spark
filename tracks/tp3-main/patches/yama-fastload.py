#!/usr/bin/env python3
"""HAREM-TP3 fastload: the per-rank weight sidecar hook (the 21d93d0d8 port of the previous
stack's patch-fastload-tp3.py; both anchors hold unchanged).

``harem_fastload.py`` (the ONLY difference from the previous stack's copy is the
written-parts ledger: the ledger of the plugin's mixed fused linear is written into the
manifest at dump, and at load only the recorded parts are marked) and
``harem_fastload_id.py`` (the identity now also covers yama-*.py, harem_yama.py,
harem-prelude.sh and the plugin's .py sources) are copied under
``model_executor/model_loader/``; ``BaseModelLoader.load_model`` goes through them. When
``HAREM_FASTLOAD_MODE`` is empty the hook calls ``self.load_weights(...)`` and then (as in
the previous stack, on by default) releases the checkpoint page cache
(HAREM_DROP_CKPT_CACHE) and calls malloc_trim (HAREM_MALLOC_TRIM); on GB10 both give
memory back to the KV pool.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harem_yama as Y  # noqa: E402

BL = "model_executor/model_loader/base_loader.py"

D = Y.Duzen
DUZENLER = [
    D("F1-yukle", BL,
      "            logger.debug(\"Loading weights on %s ...\", load_device)\n"
      "            self.load_weights(model, model_config)\n",
      "            logger.debug(\"Loading weights on %s ...\", load_device)\n"
      "            # HAREM-TP3 fastload: HAREM_FASTLOAD_MODE unset -> exactly\n"
      "            # self.load_weights(model, model_config); \"dump\" also writes the\n"
      "            # rank's post-load tensors, \"load\" restores them from the sidecar.\n"
      "            from vllm.model_executor.model_loader import (\n"
      "                harem_fastload as _harem_fastload,\n"
      "            )\n"
      "\n"
      "            _harem_fastload.load_weights_hook(self, model, model_config)\n"),
    D("F2-sonra", BL,
      "            process_weights_after_loading(model, model_config, target_device)\n"
      "\n"
      "        return model.eval()\n",
      "            process_weights_after_loading(model, model_config, target_device)\n"
      "            # HAREM-TP3 fastload: optional post-processing state hashes.\n"
      "            _harem_fastload.after_process_hook(model, model_config)\n"
      "\n"
      "        return model.eval()\n"),
]

YAMA = Y.Yama(
    ad="yama-fastload",
    aciklama="rank başı fastload sidecar kancası (HAREM_FASTLOAD_MODE)",
    duzenler=DUZENLER,
    taban={BL: "80010a6ce41704c38d110a24b1a59b6342982d972d10e56da7158137b44f90b3"},
    kopyalar=[
        Y.Kopya("harem_fastload.py", "model_executor/model_loader/harem_fastload.py"),
        Y.Kopya("harem_fastload_id.py", "model_executor/model_loader/harem_fastload_id.py"),
    ],
)

if __name__ == "__main__":
    sys.exit(Y.main(YAMA, __file__))
