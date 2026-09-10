#!/usr/bin/env python3
"""Register patch-flashkda-tp3.py in this node's full-scope prelude.

Writes the new prelude through the EXISTING inode (inside the patch tree
tp3-prelude.sh is a HARD LINK to tp3full-prelude.sh, and the tree's own comment
says the two names must not drift), after taking a dated backup.  Idempotent: a
prelude that already calls the script is left alone.  `--undo` restores that
backup, which is the whole of the rollback on the prelude side.

    register-flashkda.py [--dir ~/exl3-zeus/tp3full] [--undo]

The published prelude in this repository already carries the block below, so
this script is for the case where yours does not -- and for `--undo`.
Measurements: results/gates/flashkda-ab-10sep.md.
"""
import argparse
import os
import shutil
import sys

ANCHOR = '  run python3 "$TP3_DIR/check-padload-tp3.py"\nfi\n'
BLOCK = '''
# --- FlashKDA KDA chunked prefill (10 September 2026, vLLM PR #55737 port) ---
# Applied UNCONDITIONALLY so the A/B control arm runs the same bytes as the
# candidate; the BEHAVIOUR is env-gated and default OFF.
#   HAREM_KDA_FLASHKDA unset / 0  -> the Triton chunk_kda_with_fused_gate chain,
#                                    byte for byte upstream (one env read per
#                                    KDA layer at construction, nothing else).
#   HAREM_KDA_FLASHKDA=1          -> the fused vllm._flashkda_C kernel for KDA
#                                    chunked prefill, and a loud refusal if this
#                                    hardware/dtype/head_dim cannot take it.
# The chosen backend is printed once per process as
#   [HAREM-FLASHKDA] kda_prefill_backend=triton|flashkda
# ORDER: after patch-fullscope-tp3.py, which is the other arm that edits
# glm5next/nvidia/kda.py (its anchors are in __init__ and weight loading, the
# five here are the import block, _cast_sigmoid, the end of __init__, forward's
# definition line and the chunked-prefill call -- no overlap).
# Same fail-closed `run` wrapper as every arm above: a drifted anchor stops the
# rank instead of serving a silently-wrong model.
# Evaluation, gates and the A/B that promoted it:
#   results/gates/flashkda-ab-10sep.md and tracks/tp3/patches/flashkda/README.md
# NOTE --root here is the DIST-PACKAGES root, not $VLLM_PY: this script's REL
# is "vllm/models/glm5next/nvidia/kda.py" (it was written to run against a
# throwaway tree), unlike the other patch scripts whose REL is relative to
# the vllm package.  And it is a DRY RUN without --in-place -- without the
# flag it prints "dry run OK", exits 0, and patches nothing.
run python3 "$TP3_DIR/patch-flashkda-tp3.py" \
    --root "$(dirname "$VLLM_PY")" --in-place
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.expanduser("~/exl3-zeus/tp3full"))
    ap.add_argument("--undo", action="store_true")
    a = ap.parse_args()
    p = os.path.join(a.dir, "tp3-prelude.sh")
    bak = p + ".bak-before-flashkda"
    s = open(p).read()

    if a.undo:
        if not os.path.exists(bak):
            print(f"register-flashkda: no backup {bak} -- nothing to undo")
            return 1
        with open(bak) as f:
            orig = f.read()
        with open(p, "w") as f:          # same inode: the hard link survives
            f.write(orig)
        print(f"register-flashkda: {p} restored from backup")
        return 0

    if "patch-flashkda-tp3.py" in s:
        print(f"register-flashkda: already registered in {p}")
        return 0
    n = s.count(ANCHOR)
    if n != 1:
        print(f"register-flashkda: anchor count={n} (expected 1) -- refusing",
              file=sys.stderr)
        return 3
    if not os.path.exists(bak):
        shutil.copy2(p, bak)
    with open(p, "w") as f:              # same inode: the hard link survives
        f.write(s.replace(ANCHOR, ANCHOR + BLOCK, 1))
    print(f"register-flashkda: registered in {p}; backup {bak}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
