#!/usr/bin/env python3
"""HAREM-SM12 item 1: Programmatic Dependent Launch (PDL) off on sm_12x.

The 21d93d0d8 port of the previous stack's patch-pdl-gate.py (the anchor is the whole
function and holds unchanged). Upstream ``is_arch_support_pdl`` still returns
``major >= 9`` (platforms/cuda.py:761); PDL is not qualified for GB10 (sm_121), and KDA
recurrent-state races were reported on SM12x (tpurtell, Apache-2.0; Zeuss5/cuda-exl3
issue #6).

  HAREM_PDL_SM12 empty or "0" -> PDL OFF on sm_12x (default; what production runs)
  HAREM_PDL_SM12="1"          -> upstream (major >= 9), the A/B arm
Capabilities 9 and 10 do not change in either direction. The knob must stay fixed for the
life of the process (the mHC tilelang path caches the value at import).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harem_yama as Y  # noqa: E402

CUDA = "platforms/cuda.py"

D = Y.Duzen
DUZENLER = [
    D("PDL1-kapi", CUDA,
      "    def is_arch_support_pdl(cls) -> bool:\n"
      "        try:\n"
      "            device = torch.cuda.current_device()\n"
      "            major, _ = torch.cuda.get_device_capability(device)\n"
      "        except Exception:\n"
      "            return False\n"
      "        return major >= 9\n",
      "    def is_arch_support_pdl(cls) -> bool:\n"
      "        try:\n"
      "            device = torch.cuda.current_device()\n"
      "            major, _ = torch.cuda.get_device_capability(device)\n"
      "        except Exception:\n"
      "            return False\n"
      "        # HAREM-SM12 PDL gate: PDL is not qualified on sm_12x (GB10 = sm_121)\n"
      "        # and KDA recurrent-state races were reported there; off by default.\n"
      "        # HAREM_PDL_SM12=1 restores upstream (major >= 9). Fixed per process.\n"
      "        if os.environ.get(\"HAREM_PDL_SM12\", \"0\") == \"1\":\n"
      "            return major >= 9\n"
      "        return major in (9, 10)\n"),
]

YAMA = Y.Yama(
    ad="yama-pdl",
    aciklama="sm_12x'te PDL varsayılan kapalı (HAREM_PDL_SM12)",
    duzenler=DUZENLER,
    taban={CUDA: "27bd20e6fe42dd70b72fe58ce12cb615a2385776f0ac705349646c67ca68bb26"},
)

if __name__ == "__main__":
    sys.exit(Y.main(YAMA, __file__))
