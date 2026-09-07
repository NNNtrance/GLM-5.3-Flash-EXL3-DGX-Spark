#!/bin/bash
# In-container prelude for GLM-5.3-Flash EXL3 at TP=2 -- PRODUCTION CANDIDATE ARM.
#
# This is the two-node counterpart of tp3full/tp3-prelude.sh. It runs the same
# patch scripts, byte for byte, MINUS everything that exists only because three
# ranks need padding:
#
#   not here, and why
#     patch-vllm-tp3.py        the zero-extend helper for a shard that runs past
#                              the stored dim. At tp=2 nothing is padded (heads
#                              64/2, vocab 154880/2 = 605 x 128, shared expert
#                              2048/2 = 8 x 128), so the branch can never fire.
#                              Harmless to keep; not shipped, because we have
#                              never measured a two-node arm carrying it.
#     patch-exl3-ep.py         + overlay/: the cuda-exl3 EP kernel fixes. EP is
#                              off at two ranks (ENABLE_EP=0).
#     patch-dflash-tp3.py      makes the DFlash2 head check pad-aware over the
#                              32/8 -> 36/9 drafter pad. There is no such pad at
#                              tp=2: 32/8 divides by two.
#     patch-fullscope-tp3.py   its A9/A10 are the padded-load audit. The tp=2
#                              equivalent is patch-fullscope-tp2.py, below.
#     check-padload-tp3.py     gates a cuda-exl3 capability only the padded-load
#                              path needs.
#     pad-tp3.py/pad-tp3full.py  build padded sidecars. Nothing to pad.
#     patch-zerokv-tp3.py      an optional arm, never in production 10's env.
#
# Every optional patch is conditional on its own env knob, so an env file that
# asks for nothing behaves exactly like the unpatched image.
#
#   TP2_STRICT=0 downgrades a failed patch to a warning. Do not use it to get
#   past a broken anchor: a half-patched stack is the failure mode that serves
#   fluent, wrong answers.
#
# Mount this at /start.sh and launch the image with `--entrypoint bash /start.sh`.
# It is hard-linked to tp3-prelude.sh inside the same directory ON PURPOSE: the
# fastload identity (harem_fastload_id.file_identity) hashes the prelude under
# the name "tp3-prelude.sh", so the hard link is what keeps the prelude's text
# inside the sidecar manifest at two ranks. One inode, so the two names cannot
# drift apart.
set -euo pipefail

TP2_DIR="${TP2_DIR:-/opt/harem-tp2}"
VLLM_PY="${VLLM_PY:-/usr/local/lib/python3.12/dist-packages/vllm}"
STRICT="${TP2_STRICT:-1}"

run() {
  echo "[tp2full-prelude] $*"
  if "$@"; then return 0; fi
  echo "[tp2full-prelude] FAILED: $*" >&2
  [ "$STRICT" = "1" ] && exit 21
  return 0
}

echo "[tp2full-prelude] TP2D arm rank=${NODE_RANK:-?} tp=${TP_SIZE:-?} ep=${ENABLE_EP:-?} fullscope=${HAREM_EXL3_FULLSCOPE:-0} vision=${HAREM_VISION:-0} fastload=${HAREM_FASTLOAD_MODE:-off}"

# --- patch-vllm-tp3.py at TWO ranks (8 September 2026) -----------------------
# The two-node tree deliberately did NOT carry this file: its padding half is a
# no-op by arithmetic at tp<=2 (lcm(128, 2) = 128; 154,880 and 2,048 are already
# multiples of 128), and we do not ship text we have not measured.  The VISION
# arm changes that, because edit 4b of this script is what wraps the tower
# construction block:
#
#     self.visual = Glm5NextVisionTransformer(   ->  _harem_build_vision_tower(
#
# and patch-vision-tp3.py's VS1 anchor is written against the POST-4b text
# (a stock-file anchor matched zero times and stopped all three TP=3 ranks with
# exit 21 on 7 September -- docs/18 section 13).  Rather than fork the vision
# patch for two ranks, the two-node tree now carries the same file the three-node
# tree does, byte for byte, and the padding half stays the no-op it always was.
# Edit 4c (skip visual.* when the tower was not built) is what keeps
# LANGUAGE_MODEL_ONLY=1 working in this same tree.
run python3 "$TP2_DIR/patch-vllm-tp3.py" --root "$VLLM_PY"

# Logging only: print the per-group decomposition of the KV pool arithmetic, so
# "GPU KV cache size: N tokens" is an explained number rather than a mystery.
run python3 "$TP2_DIR/patch-kvdiag-tp3.py" --root "$VLLM_PY"

# The draft page fix (docs/07 sec.3, docs/15 sec.3.5). Gated on
# HAREM_SW_BLOCK_SIZE; unset == upstream behaviour byte for byte. Mandatory in
# practice at two ranks: without it a 6,253-token prompt is never scheduled.
run python3 "$TP2_DIR/patch-swblock-tp3.py" --root "$VLLM_PY"

# EXL3 keeps the heavy part of an expert in "<proj>.trellis"; upstream's EP
# weight filter only recognises ".weight"/".weight_packed". Inert unless
# --enable-ep-weight-filter is passed, which the two-node env does NOT do (the
# filter needs EP). Kept so an EP-on TP=2 arm needs no tree change -- and a
# tree change means a new fastload manifest, i.e. a new dump boot.
run python3 "$TP2_DIR/patch-epfilter-tp3.py" --root "$VLLM_PY"

# Per-rank fastload sidecar. Inert unless HAREM_FASTLOAD_MODE is dump|load;
# start-tp2full.sh sets that (and the mount) from FASTLOAD_MODE in the env file.
run python3 "$TP2_DIR/patch-fastload-tp3.py" --root "$VLLM_PY"

# --- Optional arms, each behind its own knob --------------------------------
#  HAREM_DRAFT_KV_DTYPE=fp8   put the DFlash2 drafter's KV at the main groups'
#                             precision (the launcher pins it to "auto" today).
#  HAREM_TILELANG_FAILLOUD=1  turn tilelang_kernels.py's silent
#                             contextlib.suppress around `import flashinfer.comm`
#                             into a named, immediate error.
if [ -n "${HAREM_DRAFT_KV_DTYPE:-}" ]; then
  run python3 "$TP2_DIR/patch-draftkv-tp3.py" --root "$VLLM_PY"
fi
if [ "${HAREM_TILELANG_FAILLOUD:-}" = "1" ]; then
  run python3 "$TP2_DIR/patch-tilelang-failloud-tp3.py" --root "$VLLM_PY"
fi

# --- The sparse indexer's K-gather workspace (6 September 2026) --------------
# Upstream sizes it as 40 * max_model_len ENTRIES -- a constant chosen against
# DeepSeek-V3.2's 163,840-token context, where it comes to 825 MB. At
# max_model_len 1,000,000 the same constant reserves 40,000,000 x 132 B =
# 4.92 GiB, during the profile run, locked by lock_workspace() for the life of
# the engine, and charged to the residual the profiler subtracts BEFORE it sizes
# the KV pool. The buffer only ever holds ONE indexer chunk's COMPRESSED
# context, so its real ceiling is
#     max_num_seqs * ceil((max_model_len + num_spec + 1) / index_kpool).
#
# EVERY TERM IS PER ENGINE, NOT PER RANK, so the same 4.92 GiB is reserved at
# two ranks as at three -- against a pool that is a third the size. Measured on
# both: +10.25 % of pool at three ranks, +32.14 % at two on the same eager-boot
# comparison, and +26.5 % against the previous two-node recipe once the fast-load
# sidecar is back. Measurements: results/memory/indexer-workspace-ab.md and
# docs/15 section 5.9. The file is the three-node track's, unchanged: the bound
# reads nothing that knows the rank count.
#
# The patch is applied UNCONDITIONALLY here and its BEHAVIOUR is env-gated,
# default OFF:
#   HAREM_INDEXER_WS_MODE unset / off / upstream  -> upstream sizing, byte for
#       byte, guards L2+L3 disarmed (one environment read at import).
#   HAREM_INDEXER_WS_MODE=bound                   -> real-bound sizing (512 MB)
#       and the two run-time guards armed. This is the two-node recipe.
# READ-ONLY OVERLAY: model_executor/layers/sparse_attn_indexer_kpool.py is
# bind-mounted read-only from $OVERLAY_DIR, so that half must be pre-applied to
# the host-side overlay copy; the script then reports "already patched" for it
# here and writes only the image's own indexer.py. Run it against a copy of the
# overlay, not against the overlay a running engine is mounting.
# Same fail-closed `run` wrapper as every arm above: a drifted anchor stops the
# rank instead of serving a silently-wrong model.
run python3 "$TP2_DIR/patch-indexer-workspace-tp3.py" --root "$VLLM_PY"

# --- prefix-hit + kpool-tail backports at TWO ranks (8 September 2026) -------
# Both files are the three-node track's, byte for byte: neither reads the rank
# count.  Applied UNCONDITIONALLY; BEHAVIOUR env-gated and default OFF, so the
# knobs unset == the two-node candidate C tree, byte for byte.
#
#  HAREM_PREFIX_HIT=1      flag ONLY the DFlash2 drafter's KV groups as EAGLE
#                          groups.  Unset, nothing is flagged and the
#                          coordinator falls back to flagging EVERY group, so
#                          the target MLA group drops a whole block off every
#                          exact-repeat prefix hit.  The BLOCK GRANULARITY is
#                          pool arithmetic and is NOT the three-node 3,328 --
#                          measure it (the GCD of the observed hits) with
#                          prefix-hit-probe.py before quoting a ceiling.
#  HAREM_KPOOL_TAIL_FIX=1  pass positions through the hybrid attention-metadata
#                          path so the K-pool tail's own circular slot mapping
#                          runs, and write it in place.  Rank-count independent
#                          (it is the hybrid/KDA model path).
#  HAREM_KPOOL_TAIL_BOUNDS=1  arm the tail wrong-block counter (log only).
run python3 "$TP2_DIR/patch-prefixhit-tp3.py" --root "$VLLM_PY"
run python3 "$TP2_DIR/patch-kpooltail-tp3.py" --root "$VLLM_PY"

# --- Full-scope EXL3 at two ranks -------------------------------------------
# S1 packed_modules_mapping, S2 stop hard-wiring MLA+KDA to bf16, S3 KDA
# refactorisation. No A9/A10: those are the padded-load audit and there is no
# pad at tp=2. Only for a FULL-SCOPE checkpoint (turboderp/GLM-5.3-Flash-exl3);
# unset == upstream image behaviour, so a patched image still serves the
# routed-experts-only checkpoint correctly.
if [ "${HAREM_EXL3_FULLSCOPE:-}" = "1" ]; then
  echo "[tp2full-prelude] patch-fullscope-tp2.py sha256 $(sha256sum "$TP2_DIR/patch-fullscope-tp2.py" | cut -c1-16)"
  run python3 "$TP2_DIR/patch-fullscope-tp2.py" --root "$VLLM_PY"
fi

# --- The vision tower at TWO ranks (8 September 2026) ------------------------
# Same six anchors as at three ranks, same file, same knob.  What two ranks
# change is NOT in this script:
#   * the tower divides cleanly at tp=2 (heads 16/2 = 8, attn.proj 512 = 4x128,
#     MLP and merger 2048 = 16x128, merger context 5120 = 40x128), so the
#     head-count problem that forces --mm-encoder-tp-mode data at three ranks
#     does not exist here.  We ship `data` anyway and say why in docs/19: it
#     REPLICATES the tower instead of slicing it, and slicing a 6-bit EXL3
#     tower across ranks is a path nothing in this repository has measured.
#     The tower is 0.557 GiB, so replication is the cheap side of that trade.
#   * everything else -- VS1/VS2/VS3 (the checkpoint is 6-bit EXL3 and vLLM
#     builds the tower with quant_config=None whatever the rank count) and
#     VS4/VS6/VS7 (the upstream frame-sampler mismatch, vllm#55644) -- is a
#     property of the CHECKPOINT and of upstream, not of sharding.
#
# ORDER: after patch-fullscope-tp2.py (whose A3 anchor is the MLA line, a
# different string) and after patch-vllm-tp3.py (VS1 anchors on its edit 4b).
# HAREM_VISION unset => nothing below runs => candidate C byte for byte.
if [ "${HAREM_VISION:-}" = "1" ]; then
  echo "[tp2full-prelude] VISION ARM: patch-vision-tp3.py sha256 $(sha256sum "$TP2_DIR/patch-vision-tp3.py" | cut -c1-16)"
  run python3 "$TP2_DIR/patch-vision-tp3.py" --root "$VLLM_PY"
  # Three model-free gates, in the boot log, before a byte of weight moves.
  # HAREM_VISION_GATES=0 only if a gate is itself broken -- never to get past a
  # gate that is telling the truth.
  if [ "${HAREM_VISION_GATES:-1}" = "1" ] && [ -d "${1:-}" ]; then
    run python3 "$TP2_DIR/check-vision-mapping.py" --model "$1" \
        --env-mapping "${CUDA_EXL3_PACKED_MAPPING:-}"
    run python3 "$TP2_DIR/check-vision-names.py" --model "$1"
    if [ -d "$TP2_DIR/fixtures" ]; then
      run python3 "$TP2_DIR/check-video-geometry.py" --model "$1" \
          --fixtures "$TP2_DIR/fixtures" \
          --max-video-tokens "${HAREM_VISION_MAX_VIDEO_TOKENS:-8000}"
    else
      echo "[tp2full-prelude] VISION: no fixtures under $TP2_DIR -- video geometry gate SKIPPED"
    fi
  fi
fi

# Import flashinfer.comm once, CPU-side, before any worker starts: prints the
# version into the boot log and warms flashinfer's JIT cache so the ranks do not
# race it. ~2 s. HAREM_FLASHINFER_WARMUP=0 skips it.
run python3 "$TP2_DIR/flashinfer-warmup.py"

# The model directory is argv[1]; run the shape preflight against whatever the
# launcher actually mounted, not against what the .env says it mounted. At tp=2
# it accepts --ep 0: moe_intermediate_size 2048 is a multiple of 128*2, so the
# routed experts tensor-slice cleanly and expert parallelism is optional.
if [ -d "${1:-}" ]; then
  run python3 "$TP2_DIR/preflight-tp3.py" --model "$1" --tp "${TP_SIZE:-2}" \
      --ep "${ENABLE_EP:-0}"
fi

# A fastload sidecar produced from another checkpoint / image / patch set must
# stop the rank here, not four minutes later with weights nobody checked.
# It reads TP3_DIR, which start-tp2full.sh points at this same directory.
if [ -n "${HAREM_FASTLOAD_MODE:-}" ] && [ -d "${1:-}" ]; then
  run python3 "$TP2_DIR/preflight-fastload.py" --model "$1"
fi

echo "[tp2full-prelude] patches applied (tp2d arm: candidate C + vision + prefix-hit + kpool-tail); starting vllm serve"
exec vllm serve "$@"
