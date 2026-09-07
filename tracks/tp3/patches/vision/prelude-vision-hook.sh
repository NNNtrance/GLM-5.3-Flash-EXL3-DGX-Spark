# --- Vision tower (7 September 2026, production configuration 13) ------------
# PASTE THIS BLOCK into tp3full-prelude.sh, AFTER the full-scope block and
# BEFORE the final `exec`.  It is not a standalone script: it uses the
# prelude's own `run` wrapper, $TP3_DIR and $VLLM_PY.
#
# Turns the GLM-5.3-Flash vision tower back on at TP=3 for the FULL-SCOPE EXL3
# checkpoint.  Six anchors, one knob:
#   VS1  the tower is built with quant_config=None (model.py:1089).  That is
#        right for the zai-org fp8 checkpoint the comment was written for and
#        wrong for turboderp/GLM-5.3-Flash-exl3@4.05bpw, whose tower is 6-bit
#        EXL3 (172 modules, 1,007 tensors, shard 19, 0.557 GiB) with no dense
#        `.weight` for proj/mlp/merger at all.  Pass the EXL3 config through.
#   VS2  the stacked weight map expects the GLM-OCR spelling `.attn.q.`; this
#        checkpoint writes `.attn.q_proj.trellis`.  Add the `_proj` spellings.
#   VS3  drop the dead pre-quantization fused `attn.qkv.{weight,bias}`
#        tensors, then AUDIT the built tower: 99 EXL3 linears, 0 bf16 ones
#        (Exl3Config.get_quant_method fails OPEN, so a mixed tower is possible
#        and is exactly the bug that loads clean and answers wrong).
#   VS4  derive the VIDEO placeholder timestamps from the pixel path's own
#        frame sampler.  Upstream took them from GLM-4.6V's, and 12 timestamps
#        against a grid of 4 killed the engine core on all three ranks.
#   VS6  fail-closed: placeholders != encoder rows is an HTTP 400 in the
#        frontend, never a dead worker.
#   VS7  lift the inherited `{"video": 1}` cap, which silently clamped
#        --limit-mm-per-prompt.  HAREM_VISION_VIDEO_LIMIT, default 2.
#
# ORDER: after patch-fullscope-tp3.py.  VS1's anchor spans the whole tower
# construction block and must see it unedited; fullscope's A3 is the MLA line,
# a different string, and the two do not overlap.
#
# The packed-mapping half needs NO patch: cuda-exl3 merges
# CUDA_EXL3_PACKED_MAPPING UNDER the class mapping (config.py:112-125,
# :381-386), so the `qkv_proj` entry that full-scope's A2 shadows comes back
# from the env and cannot clobber anything.  The .env file carries it.
#
# HAREM_VISION unset => nothing below runs => this tree behaves exactly like
# production 12, byte for byte.  That is what makes it a safe place to stand.
# Design, measurements and gates: docs/18.
if [ "${HAREM_VISION:-}" = "1" ]; then
  echo "[tp3-prelude] VISION ARM: patch-vision-tp3.py sha256 $(sha256sum "$TP3_DIR/patch-vision-tp3.py" | cut -c1-16)"
  run python3 "$TP3_DIR/patch-vision-tp3.py" --root "$VLLM_PY"
  # Two model-free gates, in the boot log, before a single weight moves.  Both
  # read safetensors HEADERS only and build no model.  They exist because the
  # two failure modes here are quiet: a missing packed-mapping entry silently
  # produces a bf16 linear, and a mis-ordered mapper entry silently renames a
  # tensor onto a parameter that does not exist.  Skip with
  # HAREM_VISION_GATES=0 only if the gates themselves are broken -- never to
  # get past a gate that is telling the truth.
  if [ "${HAREM_VISION_GATES:-1}" = "1" ] && [ -d "${1:-}" ]; then
    run python3 "$TP3_DIR/check-vision-mapping.py" --model "$1" \
        --env-mapping "${CUDA_EXL3_PACKED_MAPPING:-}"
    run python3 "$TP3_DIR/check-vision-names.py" --model "$1"
    # Round 3: the GEOMETRY gate.  The two gates above look only at weight
    # loading, and rounds 1 and 2 showed that is half a gate -- the tower loaded
    # perfectly (99/99 EXL3, 0 bf16) and the engine core still died on the first
    # video request, because the prompt's placeholder count and the encoder's
    # row count come from two different frame samplers.  This one runs the real
    # pixel path over the fixture videos plus an analytic 1080p case and refuses
    # the boot unless placeholders == encoder rows.  It builds no model and
    # loads no weights; it needs the fixtures, which ride along in TP3_DIR.
    if [ -d "$TP3_DIR/fixtures" ]; then
      run python3 "$TP3_DIR/check-video-geometry.py" --model "$1" \
          --fixtures "$TP3_DIR/fixtures" \
          --max-video-tokens "${HAREM_VISION_MAX_VIDEO_TOKENS:-8000}"
    else
      echo "[tp3-prelude] VISION: no fixtures under $TP3_DIR -- video geometry gate SKIPPED"
    fi
  fi
fi
