#!/usr/bin/env bash
# Vision arm, model-free verification. CPU container, no --gpus, no engine.
# Safe to run while production is serving: it patches a THROWAWAY container's
# copy of vLLM and reads only safetensors headers.
#   ~/exl3-zeus/tp3vision/verify-cpu.sh
# Must print five PASS/applied lines. Anything else: do not boot a rank.
#
# 7 September 2026, ROUND 3: a fifth line joins them -- check-video-geometry.py. The
# first four look only at WEIGHT LOADING, and rounds 1 and 2 proved that is half
# a gate: the tower loaded perfectly and the engine still died on the first
# video, because the placeholder count and the encoder row count came from two
# different frame samplers. That fault was reproducible on a CPU in under a
# second and no gate was looking for it.
#
# 7 September 2026, AFTER THE FIRST BOOT ATTEMPT: this script used to chain
# fullscope -> vision only. That is NOT what the prelude does, and the
# difference hid a real fault: patch-vllm-tp3.py's edit 4b rewrites the first
# two lines of the tower construction block, so VS1's stock-file anchor matched
# zero times at boot (exit 21 on all three ranks) while this gate said PASS.
# It now runs the FULL prelude order, with the same env knobs and the same
# read-only overlay bind-mount as start-tp3.sh, so the gate and the boot patch
# the same bytes. A gate that does not reproduce the boot is not a gate.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE="${IMAGE:-exl3-zeus:754421f}"
MODEL="${MODEL:-/var/tmp/glm-5.3-flash-turboderp-4.05bpw-tp3}"
LINK="${LINK:-/var/tmp/glm-5.3-flash-turboderp-4.05bpw}"
OVERLAY="${OVERLAY_DIR:-$DIR/overlay-prod11}"
PM="${CUDA_EXL3_PACKED_MAPPING:-{\"qkv_proj\":[\"q_proj\",\"k_proj\",\"v_proj\"]\}}"
VP=/usr/local/lib/python3.12/dist-packages/vllm

docker run --rm --cpuset-cpus 0-2 --memory 6g \
  -v "$DIR:/opt/harem-tp3:ro" -v "$LINK:$LINK:ro" -v "$MODEL:$MODEL:ro" \
  -v "$OVERLAY/sparse_attn_indexer_kpool.py:$VP/model_executor/layers/sparse_attn_indexer_kpool.py:ro" \
  -e HAREM_VISION=1 -e HAREM_EXL3_FULLSCOPE=1 -e HAREM_DRAFT_KV_DTYPE=fp8 \
  -e HAREM_TILELANG_FAILLOUD=1 -e HAREM_SM12_ITEMS=pdl,kpool \
  -e HAREM_INDEXER_WS_MODE=bound \
  -e "CUDA_EXL3_PACKED_MAPPING=$PM" \
  -e "M=$MODEL" --entrypoint bash "$IMAGE" -c '
# pipefail matters: every check below ends in `| tail -1`, and without it a
# failing patch script exits 0 through the pipe (measured 7 September 2026).
set -eo pipefail
VP=/usr/local/lib/python3.12/dist-packages/vllm
EX=/usr/local/lib/python3.12/dist-packages/cuda_exl3
P=/opt/harem-tp3
# --- the prelude order, verbatim, up to the vision step ---------------------
python3 $P/patch-vllm-tp3.py  --root $VP                                     >/dev/null
python3 $P/patch-exl3-ep.py   --pkg  $EX --overlay $P/overlay/cuda_exl3/_harem_ep.py >/dev/null
if [ -f "$VP/model_executor/models/qwen3_dflash2.py" ]; then
  python3 $P/patch-dflash-tp3.py --root $VP                                  >/dev/null
fi
python3 $P/patch-kvdiag-tp3.py           --root $VP                          >/dev/null
python3 $P/patch-swblock-tp3.py          --root $VP                          >/dev/null
python3 $P/patch-epfilter-tp3.py         --root $VP                          >/dev/null
python3 $P/patch-fastload-tp3.py         --root $VP                          >/dev/null
python3 $P/patch-draftkv-tp3.py          --root $VP                          >/dev/null
python3 $P/patch-tilelang-failloud-tp3.py --root $VP                         >/dev/null
python3 $P/patch-pdl-gate.py             --root $VP                          >/dev/null
python3 $P/patch-kpool-init.py           --root $VP                          >/dev/null
python3 $P/patch-indexer-workspace-tp3.py --root $VP                         >/dev/null
# --- the two arms the vision configuration stands on -----------------------
python3 $P/patch-fullscope-tp3.py  --root $VP | tail -1
python3 $P/patch-vision-tp3.py     --root $VP | tail -1
python3 $P/check-vision-mapping.py --model $M --env-mapping "$CUDA_EXL3_PACKED_MAPPING" | tail -1
python3 $P/check-vision-names.py   --model $M | tail -1
python3 $P/check-video-geometry.py --model $M --fixtures $P/fixtures --max-video-tokens ${HAREM_VISION_MAX_VIDEO_TOKENS:-8000} | tail -1
' 2>&1 | grep -vE '^(INFO|WARNING) [0-9]|^W[0-9]{4} '
