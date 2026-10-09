#!/usr/bin/env python3
"""HAREM-TP3: teach the EP weight filter EXL3's heavy expert tensor.

The 21d93d0d8 port of the previous stack's patch-epfilter-tp3.py (the anchors hold
unchanged). Upstream ``should_skip_weight`` skips only ``.weight`` / ``.weight_packed``
names (ep_weight_filter.py:83); EXL3 keeps 99.8 % of an expert's bytes in
``<proj>.trellis``, so even with ``--enable-ep-weight-filter`` every rank would read all
288 experts. The extra suffixes come from the environment:
HAREM_EP_FILTER_SUFFIXES=".trellis" (default; comma separated; "" = off). The small
scale tensors (.suh/.svh/.mul1) are deliberately read for every expert.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harem_yama as Y  # noqa: E402

EPF = "model_executor/model_loader/ep_weight_filter.py"

YARDIMCI = '''# HAREM-TP3: EXL3 stores the bulk of an expert in "<proj>.trellis" ----------
def _harem_skippable_suffixes() -> tuple[str, ...]:
    import os as _harem_os

    raw = _harem_os.environ.get("HAREM_EP_FILTER_SUFFIXES", ".trellis")
    extra = tuple(s.strip() for s in raw.split(",") if s.strip())
    bad = [s for s in extra if not s.startswith(".")]
    if bad:
        raise ValueError(f"HAREM_EP_FILTER_SUFFIXES entries must start with '.': {bad}")
    return (".weight", ".weight_packed") + extra


_HAREM_SKIPPABLE_SUFFIXES = _harem_skippable_suffixes()
# ---------------------------------------------------------------------------


def should_skip_weight('''

D = Y.Duzen
DUZENLER = [
    D("E0-yardimci", EPF, "def should_skip_weight(", YARDIMCI),
    D("E1-sonekler", EPF,
      "    if not weight_name.endswith((\".weight\", \".weight_packed\")):\n"
      "        return False\n",
      "    # HAREM-TP3: + HAREM_EP_FILTER_SUFFIXES (default \".trellis\", EXL3).\n"
      "    if not weight_name.endswith(_HAREM_SKIPPABLE_SUFFIXES):\n"
      "        return False\n"),
]

YAMA = Y.Yama(
    ad="yama-epfilter",
    aciklama="EP ağırlık süzgeci EXL3 .trellis'i de atlasın",
    duzenler=DUZENLER,
    taban={EPF: "c6e020abeed0d9d78eb6f6705376ea6b63abe95fbbadace6f7faec17c69396dc"},
)

if __name__ == "__main__":
    sys.exit(Y.main(YAMA, __file__))
