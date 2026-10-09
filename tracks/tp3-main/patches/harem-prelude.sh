#!/bin/bash
# HAREM pre-container script (prelude) -- the new stack: official vLLM nightly 21d93d0d8
# (+ the cuda-exl3 plugin, installed into the image separately).
#
# It replaces the previous stack's tp3-prelude.sh (24 patch scripts + 2 overlays): here
# there are 14 anchored patches, all applied UNCONDITIONALLY, and their behaviour is
# switched by environment knobs (knob empty = upstream). Every patch uses the common engine
# (harem_yama.py): the base sha256 of each file is pinned, the anchor occurs exactly once,
# it is idempotent and is undone with --revert. If one anchor does not match, the rank
# STOPS (a half-patched stack is fluent but answers wrongly).
#
#   -v $TP3_DIR/harem-prelude.sh:/start.sh:ro  -v $TP3_DIR:/opt/harem-tp3:ro
#   --entrypoint bash IMAGE /start.sh <model path> <vllm serve arguments...>
#
# HAREM_PRELUDE_KURU=1: apply the patches + verify them, do NOT start vllm serve (a test).
# Behaviour knobs (the production values are in the environment file):
#   HAREM_MAMBA_GRID=1   HAREM_SW_BLOCK_SIZE=256   HAREM_PDL_SM12 (empty = PDL off)
#   HAREM_GLM47_FAILCLOSED (empty = on)   HAREM_VISION=1 HAREM_VISION_VIDEO_LIMIT=4
#   HAREM_EP_FILTER_SUFFIXES (empty = .trellis)   HAREM_FASTLOAD_MODE=dump|load
#   HAREM_TP_PAD (the plugin's TP3 hook; empty = on)
#   HAREM_PREFIX_HIT=1 (only the DFlash drafter KV groups get is_eagle_group; empty = upstream)
#
# Draft-tag gate (harem_taslak_kapi.py) -- right before vllm serve, does not run under KURU.
# With TP > 1, if the draft config does not carry the harem_tp_pad tag (or the tag's tp does
# not match), the boot is REFUSED (fail-closed; maintainers' decision of 7 October). The
# arguments are read with vLLM's own parser; the gate runs in fastload load mode too (the
# check inside the engine would not run in that mode).
set -euo pipefail

TP3_DIR="${TP3_DIR:-/opt/harem-tp3}"
VLLM_PY="${VLLM_PY:-/usr/local/lib/python3.12/dist-packages/vllm}"

run() {
  echo "[harem-prelude] $*"
  if "$@"; then return 0; fi
  echo "[harem-prelude] FAILED: $*" >&2
  exit 21
}

echo "[harem-prelude] rank=${NODE_RANK:-?} tp=${TP_SIZE:-?} ep=${ENABLE_EP:-?} vision=${HAREM_VISION:-0} mambagrid=${HAREM_MAMBA_GRID:-unset} swblock=${HAREM_SW_BLOCK_SIZE:-unset} fastload=${HAREM_FASTLOAD_MODE:-unset}"
python3 -c "import cuda_exl3" || { echo "[harem-prelude] FAILED: cuda_exl3 eklentisi imajda yok" >&2; exit 21; }

# --- TP3 + full-scope core --------------------------------------------------
run python3 "$TP3_DIR/yama-tp3-dolgu.py" --apply --root "$VLLM_PY"
run python3 "$TP3_DIR/yama-glm5next-tamkapsam.py" --apply --root "$VLLM_PY"
run python3 "$TP3_DIR/yama-sidecar-dizin.py" --apply --root "$VLLM_PY"
run python3 "$TP3_DIR/yama-mambagrid.py" --apply --root "$VLLM_PY"
# --- operational patches ----------------------------------------------------
run python3 "$TP3_DIR/yama-kpool-init.py" --apply --root "$VLLM_PY"
run python3 "$TP3_DIR/yama-epfilter.py" --apply --root "$VLLM_PY"
run python3 "$TP3_DIR/yama-fastload.py" --apply --root "$VLLM_PY"
run python3 "$TP3_DIR/yama-swblock.py" --apply --root "$VLLM_PY"
run python3 "$TP3_DIR/yama-pdl.py" --apply --root "$VLLM_PY"
# --- behaviour patches ------------------------------------------------------
run python3 "$TP3_DIR/yama-glm47.py" --apply --root "$VLLM_PY"
run python3 "$TP3_DIR/yama-dflash-tp3.py" --apply --root "$VLLM_PY"
run python3 "$TP3_DIR/yama-dflash-eagle3.py" --apply --root "$VLLM_PY"
run python3 "$TP3_DIR/yama-dflash-kvgrup.py" --apply --root "$VLLM_PY"
run python3 "$TP3_DIR/yama-gorsel.py" --apply --root "$VLLM_PY"

if [ "${HAREM_PRELUDE_KURU:-}" = "1" ]; then
  echo "[harem-prelude] KURU: yamalar uygulandı ve doğrulandı; vllm serve başlatılmadı"
  exit 0
fi

# Model directory argv[1]: it must be a HAREM sidecar (the index is authoritative).
if [ -d "${1:-}" ]; then
  python3 - "$1" <<'PY' || { echo "[harem-prelude] FAILED: model dizini HAREM sidecar'ı değil" >&2; exit 21; }
import json, sys
m = json.load(open(sys.argv[1] + "/model.safetensors.index.json"))["metadata"]["harem_sidecar"]
print(f"[harem-prelude] sidecar: kaynak={m['kaynak']} bolunen_kda={m['bolunen_kda']} dusurulen={len(m['dusurulen'])}")
PY
fi

# Draft-tag gate: an untagged draft with TP > 1 -> REFUSED.
python3 "$TP3_DIR/harem_taslak_kapi.py" "$@" || { echo "[harem-prelude] FAILED: taslak etiket kapisi reddetti" >&2; exit 21; }

echo "[harem-prelude] yamalar uygulandı; vllm serve başlıyor"
exec vllm serve "$@"
