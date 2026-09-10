#!/bin/bash
# In-container prelude for GLM-5.3-Flash EXL3 at TP=3 -- FULL-SCOPE ARM.
#
# Copy of tp3/tp3-prelude.sh. It lives in tp3full/ and runs tp3full/'s copies of
# the patch scripts, which differ from the production ones in exactly two
# constants (vocab padding_size lcm(128,tp)=384 instead of lcm(64,tp)=192;
# shared expert 2304 instead of 2112 -- both no-ops at tp<=2) plus one extra
# patch, patch-fullscope-tp3.py. ~/exl3-zeus/tp3/ is the production fastload
# manifest identity and is not touched: a new file there has already refused a
# production boot twice.
#
# Mount this at /start.sh and launch the image with `--entrypoint bash /start.sh`,
# passing the usual `vllm serve` arguments after it. It applies the three patch
# scripts and then execs the server, so a patch that no longer applies stops the
# rank instead of serving a silently-wrong model.
#
#   -v $TP3FULL/tp3-prelude.sh:/start.sh:ro   (hard link to this file)
#   -v $TP3FULL:/opt/harem-tp3:ro
# start-tp3.sh builds both mounts from TP3_DIR, which .env.tp3-full sets to
# $HOME/exl3-zeus/tp3full. The launcher mounts "$TP3_DIR/tp3-prelude.sh", so
# this file is hard-linked to that name inside tp3full/ -- one inode, so the
# two names cannot drift apart.
#
# The launcher directory and the patch-tree directory are two things; these
# comments use tp3full/ for both for brevity, and on our own nodes the patch
# tree is a separate directory named after the configuration. Any name works:
# set TP3_DIR to wherever you put the patch tree -- that is the directory whose
# file list and prelude text the fast-load manifest hashes.
#   --entrypoint bash "$IMAGE" /start.sh <model path> <vllm args...>
#
# TP3_STRICT=0 turns the failures into warnings. Do not use it to get past a
# broken anchor: a half-patched stack is exactly the failure mode that produces
# fluent, wrong answers.
set -euo pipefail

TP3_DIR="${TP3_DIR:-/opt/harem-tp3}"
VLLM_PY="${VLLM_PY:-/usr/local/lib/python3.12/dist-packages/vllm}"
EXL3_PKG="${EXL3_PKG:-/usr/local/lib/python3.12/dist-packages/cuda_exl3}"
STRICT="${TP3_STRICT:-1}"

run() {
  echo "[tp3-prelude] $*"
  if "$@"; then return 0; fi
  echo "[tp3-prelude] FAILED: $*" >&2
  [ "$STRICT" = "1" ] && exit 21
  return 0
}

echo "[tp3-prelude] TP3FULL arm rank=${NODE_RANK:-?} tp=${TP_SIZE:-?} ep=${ENABLE_EP:-?} fullscope=${HAREM_EXL3_FULLSCOPE:-0}"
echo "[tp3-prelude] tp3full constants: vocab padding_size lcm(128,tp), shared expert lcm(128,tp) -> 2304, A9 checkpoint-width fused split"
run python3 "$TP3_DIR/patch-vllm-tp3.py" --root "$VLLM_PY"
run python3 "$TP3_DIR/patch-exl3-ep.py" --pkg "$EXL3_PKG" \
    --overlay "$TP3_DIR/overlay/cuda_exl3/_harem_ep.py"

# The DFlash2 drafter runs on a padded sidecar config (32/8 -> 36/9 at tp=3).
# This makes the port's head check pad-aware and proves after load that the
# fabricated rows are zero. Skipped when the image carries no DFlash2 port.
if [ -f "$VLLM_PY/model_executor/models/qwen3_dflash2.py" ]; then
  run python3 "$TP3_DIR/patch-dflash-tp3.py" --root "$VLLM_PY"
fi

# Logging only: print the per-group decomposition of the KV pool arithmetic, so
# "GPU KV cache size: N tokens" is an explained number rather than a mystery.
run python3 "$TP3_DIR/patch-kvdiag-tp3.py" --root "$VLLM_PY"
run python3 "$TP3_DIR/patch-swblock-tp3.py" --root "$VLLM_PY"

# EXL3 keeps the heavy part of an expert in "<proj>.trellis"; upstream's EP
# weight filter only recognises ".weight"/".weight_packed", so without this it
# reads all 288 experts on every rank even with --enable-ep-weight-filter.
# The patch is inert unless that flag is passed.
run python3 "$TP3_DIR/patch-epfilter-tp3.py" --root "$VLLM_PY"

# Per-rank fastload sidecar. Inert unless HAREM_FASTLOAD_MODE is dump|load;
# start-tp3.sh sets that (and the mount) from FASTLOAD_MODE in the env file.
run python3 "$TP3_DIR/patch-fastload-tp3.py" --root "$VLLM_PY"

# --- Optional arms (5 September 2026) ----------------------------------------------
# Each of the three is applied ONLY when its own env knob is set, so an env file
# that never asks for one cannot be broken by an anchor that drifted in another
# image.  Unset knob == upstream behaviour, byte for byte.
#
#  HAREM_ZERO_ATTENTION_KV=0  fail-closed gate that skips the per-step ATTENTION
#                             KV memset (13.5-15.6 ms/chunk = 1.2-1.4 % prefill).
#                             It proves uniform KV precision AND that no Mamba/KDA
#                             layer shares an attention KVCacheTensor, or raises
#                             at startup instead of serving.
#  HAREM_DRAFT_KV_DTYPE=fp8   put the DFlash2 drafter's KV at the main groups'
#                             precision (start-tp3.sh pins it to "auto" today).
#  HAREM_TILELANG_FAILLOUD=1  turn tilelang_kernels.py:26's silent
#                             contextlib.suppress around `import flashinfer.comm`
#                             into a named, immediate error.
if [ "${HAREM_ZERO_ATTENTION_KV:-}" = "0" ]; then
  run python3 "$TP3_DIR/patch-zerokv-tp3.py" --root "$VLLM_PY"
fi
if [ -n "${HAREM_DRAFT_KV_DTYPE:-}" ]; then
  run python3 "$TP3_DIR/patch-draftkv-tp3.py" --root "$VLLM_PY"
fi
if [ "${HAREM_TILELANG_FAILLOUD:-}" = "1" ]; then
  run python3 "$TP3_DIR/patch-tilelang-failloud-tp3.py" --root "$VLLM_PY"
fi

# --- sm_12x stack patches (6 September 2026, Zeuss5/cuda-exl3 issue #6) -------
# Production configuration 11 carries the CORRECTNESS SET ONLY. Item 4 (the DSA
# indexer's Triton specialisation) and the diagnostic stats hook are deliberately
# NOT in this tree: measured on 6 September, neither a benefit nor a cost could be
# resolved for item 4, and the instrument stalls the step it fires in. Item 4 is
# kept in tracks/tp3/patches-optional/sm12/ for anyone who wants it.
#   HAREM_SM12_ITEMS unset  => no script runs => the tree behaves like production 10
#   pdl    item 1     the PDL gate; runtime knob HAREM_PDL_SM12=0|1 (0/unset = PDL OFF)
#   kpool  items 2+3  the K-pool top-k buffer init AND its reader's upper bound
#
# READ-ONLY OVERLAY: model_executor/layers/sparse_attn_indexer_kpool.py is bind-
# mounted read-only from $OVERLAY_DIR, and item 2 targets exactly that file, so it
# cannot be written from inside the container. Pre-apply item 2 to a host-side copy
# of the overlay directory and point OVERLAY_DIR at that copy; patch-kpool-init.py
# then reports "already applied" for it and writes only the image's own
# kpool_compress.py (item 3). This applies to ANY future patch on an overlaid file.
# Same fail-closed `run` wrapper as every arm above: a drifted anchor stops the rank.
if [ -n "${HAREM_SM12_ITEMS:-}" ]; then
  echo "[tp3-prelude] SM12 items=${HAREM_SM12_ITEMS} HAREM_PDL_SM12=${HAREM_PDL_SM12:-unset}"
  # Fail closed on a typo: an unrecognised item must stop the rank, not be skipped.
  # Silently skipping "kpol" would serve a stack the operator believes is patched.
  _sm12_rest=",${HAREM_SM12_ITEMS},"
  for _it in pdl kpool; do _sm12_rest="${_sm12_rest//,$_it,/,}"; done
  [ "$_sm12_rest" = "," ] || { echo "[tp3-prelude] FAILED: unknown HAREM_SM12_ITEMS entries: ${_sm12_rest}" >&2; exit 21; }
  case ",${HAREM_SM12_ITEMS}," in *,pdl,*)   run python3 "$TP3_DIR/patch-pdl-gate.py"   --root "$VLLM_PY" ;; esac
  case ",${HAREM_SM12_ITEMS}," in *,kpool,*) run python3 "$TP3_DIR/patch-kpool-init.py" --root "$VLLM_PY" ;; esac
fi

# --- indexer K-gather workspace bound (6 September 2026, production 12) ------
# Upstream sizes the sparse indexer's K-gather workspace as 40 * max_model_len
# ENTRIES -- a constant chosen against DeepSeek-V3.2's 163840 context, where it
# comes to 825 MB. At max_model_len = 1e6 that is 4.92 GiB, reserved during the
# profile run, locked by lock_workspace() for the life of the engine and charged
# to the residual the profiler subtracts before it sizes the KV pool. The buffer
# only ever holds ONE indexer chunk's COMPRESSED context, so its real ceiling is
# max_num_seqs * ceil((max_model_len + num_spec + 1) / index_kpool).
# Measurements: results/memory/indexer-workspace-ab.md.
#
# The patch is applied UNCONDITIONALLY here and its BEHAVIOUR is env-gated,
# default OFF:
#   HAREM_INDEXER_WS_MODE unset / off / upstream  -> upstream sizing, byte for
#       byte, guards L2+L3 disarmed (one environment read at import).
#   HAREM_INDEXER_WS_MODE=bound                   -> real-bound sizing (512 MB)
#       and the two run-time guards armed. This is production 12.
# READ-ONLY OVERLAY, again: model_executor/layers/sparse_attn_indexer_kpool.py is
# bind-mounted read-only from $OVERLAY_DIR -- the same file sm_12x item 2 lands
# on -- so that half is pre-applied to the host-side overlay copy and the script
# reports "already patched" for it here, writing only the image's own indexer.py.
# Same fail-closed `run` wrapper as every arm above: a drifted anchor stops the
# rank instead of serving a silently-wrong model.
run python3 "$TP3_DIR/patch-indexer-workspace-tp3.py" --root "$VLLM_PY"

# --- prefix-hit + kpool-tail backports (8 September 2026, production 13) ------
# Two upstream backports the pinned vLLM predates, both applied
# UNCONDITIONALLY with their BEHAVIOUR env-gated and default OFF, so the knobs
# unset == production configuration 12's tree byte for byte.  The production
# environment file sets both.
#
#  HAREM_PREFIX_HIT=1      flag ONLY the DFlash2 drafter's KV groups as EAGLE
#                          groups.  Unset, nothing is flagged and the
#                          coordinator falls back to flagging EVERY group, so
#                          the target MLA group drops a whole 3,328-token block
#                          off every exact-repeat prefix hit.
#  HAREM_KPOOL_TAIL_FIX=1  pass positions through the hybrid attention-metadata
#                          path so the K-pool tail's own circular slot mapping
#                          runs, and write it in place.  Without it 99.67 % of
#                          the tail's writes address a block that is not the
#                          request's own.
#  HAREM_KPOOL_TAIL_BOUNDS=1  arm the tail wrong-block counter (log only, an
#                          evidence arm rather than a production setting).
# Same fail-closed `run` wrapper as every arm above.
# Design, measurements and the two-boot promotion: patches/prefix-hit-and-kpool-tail/.
# NOTE the flat path: in this repository these two live in a subdirectory for a
# reader's benefit, but on a node every patch script sits directly in $TP3_DIR,
# because the sidecar identity hashes glob($TP3_DIR/patch-*.py) and a script in
# a subdirectory is neither hashed nor found (docs/08 section 4).
run python3 "$TP3_DIR/patch-prefixhit-tp3.py" --root "$VLLM_PY"
run python3 "$TP3_DIR/patch-kpooltail-tp3.py" --root "$VLLM_PY"

# --- Full-scope EXL3 (5 September 2026) ------------------------------------------
# One patch, three layers, one knob:
#   S1  packed_modules_mapping on both glm5next model classes
#   S2  stop hard-wiring the attention stack (MLA + KDA) to bf16
#   S3  KDA refactorisation: checkpoint `conv1d` -> q/k/v_conv1d, and
#       `qkv_proj` -> shards 0-2 of a split `in_proj_qkv`
#   A9  split a pre-fused checkpoint tensor by the CHECKPOINT widths, not the
#       module's padded ones (TP=3 head pad 64 -> 66)
#   A10 post-load audit: every EXL3 pad is whole 128-blocks and exactly zero
# Only for a FULL-SCOPE EXL3 checkpoint (turboderp/GLM-5.3-Flash-exl3@4.05bpw).
# HAREM_EXL3_FULLSCOPE unset == upstream image behaviour, byte for byte, and
# the patched code re-reads the knob at runtime, so a patched image still
# serves the routed-experts-only control checkpoint correctly.
# Design and measurements: docs/13 of the recipe repository.
# TP=3 pad arithmetic: docs/13 section 7.1.
# TP=3 needs a cuda-exl3 with the padded-load path: f3e3090 (padded output dim,
# row-parallel suh) AND 754421f (the vocab loaders fill a prefix). On an older
# image the lm_head load dies -- 62f53e6/5903248 raise "EXL3 weights cannot be
# zero-extended" in create_weights; f3e3090 alone passes that gate and then dies
# on a copy_ shape mismatch in _vocab_loaders. Both failures are loud.
if [ "${HAREM_EXL3_FULLSCOPE:-}" = "1" ]; then
  echo "[tp3-prelude] patch-fullscope-tp3.py sha256 $(sha256sum "$TP3_DIR/patch-fullscope-tp3.py" | cut -c1-16)"
  run python3 "$TP3_DIR/patch-fullscope-tp3.py" --root "$VLLM_PY"
  # Say which cuda-exl3 padded-load support is present, before the weights move.
  run python3 "$TP3_DIR/check-padload-tp3.py"
fi

# --- FlashKDA KDA chunked prefill (10 September 2026, vLLM PR #55737 port) ---
# Applied UNCONDITIONALLY so a control arm runs the same bytes as the candidate;
# the BEHAVIOUR is env-gated and default OFF.
#   HAREM_KDA_FLASHKDA unset / 0  -> the Triton chunk_kda_with_fused_gate chain,
#                                    byte for byte upstream (one environment
#                                    read per KDA layer at construction).
#   HAREM_KDA_FLASHKDA=1          -> the fused vllm._flashkda_C kernel for KDA
#                                    chunked prefill, and a loud refusal if this
#                                    hardware/dtype/head_dim cannot take it.
# The chosen backend is printed once per process as
#   [HAREM-FLASHKDA] kda_prefill_backend=triton|flashkda
# A boot log with no such line is a boot where this patch did not run.
# ORDER: after patch-fullscope-tp3.py, the other arm that edits
# glm5next/nvidia/kda.py (its anchors are in __init__ and weight loading; the
# five here are the import block, _cast_sigmoid, the end of __init__, forward's
# definition line and the chunked-prefill call -- no overlap). Against the
# vision block order is immaterial: that one edits model.py and multimodal.py.
# TWO THINGS ABOUT THIS LINE, and each cost a boot:
#   --root is the DIST-PACKAGES root, not $VLLM_PY -- this script's REL starts
#     with "vllm/", unlike every other patch script here, so --root "$VLLM_PY"
#     gives .../vllm/vllm/models/... and raises FileNotFoundError.
#   --in-place is required. Without it the script is a DRY RUN: it prints
#     "dry run OK", exits 0 and patches nothing, which is a healthy boot that
#     quietly keeps running the old path.
# Same fail-closed `run` wrapper as every arm above.
# Design, the A/B and the sidecar consequence: patches/flashkda/README.md and
# results/gates/flashkda-ab-10sep.md.
run python3 "$TP3_DIR/patch-flashkda-tp3.py" \
    --root "$(dirname "$VLLM_PY")" --in-place

# --- Vision tower -------------------------------------------------------------
# Production configuration 13 also runs the vision block, which is kept beside
# its own patch and gates rather than inlined here: paste
# patches/vision/prelude-vision-hook.sh in at this point. On our nodes it sits
# directly after the FlashKDA line above.

# Import flashinfer.comm once, CPU-side, before any worker starts: prints the
# version into the boot log and warms flashinfer's JIT cache so the ranks do not
# race it.  ~2 s.  Never fails the boot unless HAREM_TILELANG_FAILLOUD=1.
# HAREM_FLASHINFER_WARMUP=0 skips it.
run python3 "$TP3_DIR/flashinfer-warmup.py"


# The model directory is argv[1]; run the shape preflight against whatever the
# launcher actually mounted, not against what the .env says it mounted.
# The EP-vs-tensor-sliced decision is arithmetic on the mounted model, and
# preflight is the thing that owns it: --ep tells it which arrangement the
# launcher chose, and it refuses ENABLE_EP=0 unless moe_intermediate_size is a
# multiple of 128*tp AND the weights on disk agree with the config. The old
# blanket "TP=3 always needs EP" refusal was true of the 2048 checkpoint only.
if [ -d "${1:-}" ]; then
  run python3 "$TP3_DIR/preflight-tp3.py" --model "$1" --tp "${TP_SIZE:-3}" \
      --ep "${ENABLE_EP:-1}"
fi

# A fastload sidecar produced from another checkpoint / image / patch set must
# stop the rank here, not four minutes later with weights nobody checked.
if [ -n "${HAREM_FASTLOAD_MODE:-}" ] && [ -d "${1:-}" ]; then
  run python3 "$TP3_DIR/preflight-fastload.py" --model "$1"
fi

echo "[tp3-prelude] patches applied (tp3full arm); starting vllm serve"
exec vllm serve "$@"
