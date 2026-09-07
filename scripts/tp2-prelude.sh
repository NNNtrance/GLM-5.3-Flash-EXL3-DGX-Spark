#!/bin/bash
#
# HISTORY ONLY -- this is the CANDIDATE-B prelude (5-6 September 2026).
# The prelude the two-node recipe runs is tracks/tp2/patches/tp2full-prelude.sh,
# which adds the sparse-indexer workspace bound (candidate C), the vision block
# and the two backports (candidate D).  This copy is kept because earlier arms
# in docs/15 section 3 ran it; do NOT install it.  A patch tree's prelude text is
# hashed into the fast-load sidecar identity, so the two are not interchangeable.
#
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

echo "[tp2full-prelude] rank=${NODE_RANK:-?} tp=${TP_SIZE:-?} ep=${ENABLE_EP:-?} fullscope=${HAREM_EXL3_FULLSCOPE:-0} fastload=${HAREM_FASTLOAD_MODE:-off}"

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

echo "[tp2full-prelude] patches applied (tp2full arm); starting vllm serve"
exec vllm serve "$@"
