#!/usr/bin/env bash
# TP=2 (tp2d arm): model-free verification of the WHOLE prelude order.
# CPU container, no --gpus, no engine, no weights read -- safetensors HEADERS
# only.  Must print six PASS/applied lines.  Anything else: do not boot a rank.
#
# This is tracks/tp3/patches/vision/verify-cpu.sh with the three-node-only steps
# removed (patch-exl3-ep, patch-dflash, the SM12 pair) and the two backports plus
# patch-vllm-tp3.py kept -- the last of those because VS1 anchors on its edit 4b,
# which is the whole reason the two-node tree now carries it.
# A gate that does not reproduce the boot is not a gate (docs/18 section 13).
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE="${IMAGE:-exl3-zeus:754421f}"
MODEL="${MODEL:-/var/tmp/glm-5.3-flash-turboderp-4.05bpw}"
OVERLAY="${OVERLAY_DIR:-$DIR/overlay}"
PM="${CUDA_EXL3_PACKED_MAPPING:-{\"qkv_proj\":[\"q_proj\",\"k_proj\",\"v_proj\"]\}}"
VP=/usr/local/lib/python3.12/dist-packages/vllm

docker run --rm --cpuset-cpus 0-2 --memory 6g \
  -v "$DIR:/opt/harem-tp2:ro" -v "$MODEL:$MODEL:ro" \
  -v "$OVERLAY/sparse_attn_indexer_kpool.py:$VP/model_executor/layers/sparse_attn_indexer_kpool.py:ro" \
  -e HAREM_VISION=1 -e HAREM_EXL3_FULLSCOPE=1 -e HAREM_DRAFT_KV_DTYPE=fp8 \
  -e HAREM_TILELANG_FAILLOUD=1 -e HAREM_INDEXER_WS_MODE=bound \
  -e HAREM_PREFIX_HIT=1 -e HAREM_KPOOL_TAIL_FIX=1 \
  -e TP_SIZE=2 -e ENABLE_EP=0 \
  -e "CUDA_EXL3_PACKED_MAPPING=$PM" \
  -e "M=$MODEL" --entrypoint bash "$IMAGE" -c '
# pipefail matters: every check below ends in `| tail -1`, and without it a
# failing patch script exits 0 through the pipe.
set -eo pipefail
VP=/usr/local/lib/python3.12/dist-packages/vllm
P=/opt/harem-tp2
# --- the tp2d prelude order, verbatim -------------------------------------
python3 $P/patch-vllm-tp3.py              --root $VP                        >/dev/null
python3 $P/patch-kvdiag-tp3.py            --root $VP                        >/dev/null
python3 $P/patch-swblock-tp3.py           --root $VP                        >/dev/null
python3 $P/patch-epfilter-tp3.py          --root $VP                        >/dev/null
python3 $P/patch-fastload-tp3.py          --root $VP                        >/dev/null
python3 $P/patch-draftkv-tp3.py           --root $VP                        >/dev/null
python3 $P/patch-tilelang-failloud-tp3.py --root $VP                        >/dev/null
python3 $P/patch-indexer-workspace-tp3.py --root $VP                        >/dev/null
python3 $P/patch-prefixhit-tp3.py         --root $VP | tail -1
python3 $P/patch-kpooltail-tp3.py         --root $VP | tail -1
python3 $P/patch-fullscope-tp2.py         --root $VP | tail -1
python3 $P/patch-vision-tp3.py            --root $VP | tail -1
python3 $P/check-vision-mapping.py --model $M --env-mapping "$CUDA_EXL3_PACKED_MAPPING" | tail -1
python3 $P/check-vision-names.py   --model $M | tail -1
python3 $P/check-video-geometry.py --model $M --fixtures $P/fixtures --max-video-tokens ${HAREM_VISION_MAX_VIDEO_TOKENS:-8000} | tail -1
# the shape preflight at two ranks, expert parallelism off
python3 $P/preflight-tp3.py --model $M --tp 2 --ep 0 | tail -1
' 2>&1 | grep -vE '^(INFO|WARNING) [0-9]|^W[0-9]{4} '
