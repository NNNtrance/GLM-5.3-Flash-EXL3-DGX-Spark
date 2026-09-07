#!/usr/bin/env python3
"""HAREM-TP3 kpool-tail: hand the K-pool tail builder the token positions it
needs, and let it write the tail group's own slot-mapping buffer in place.

WHAT IS WRONG WITHOUT THIS
--------------------------
``KpoolTailSpec`` is a one-block circular scratch cache: exactly one block per
request, ``block_size == index_kpool`` (4 on GLM-5.3-Flash), addressed as
``block_table[req, 0] * kpool + pos % kpool``.  The generic per-group slot
kernel in ``v1/worker/gpu/block_table.py`` instead computes
``block_table[req, pos // block_size] * block_size + pos % block_size``.  The
tail group's block-table row is 32 entries wide and only column 0 is ever
written, so from ``pos >= 4`` the mapping reads an unwritten column and every
request collapses onto physical tail block 0; from ``pos >= 128`` it reads past
the row entirely.  The two kpool write kernels (the prefill seed and the decode
update) then store through whatever block id came back, with no bounds check.

The image already contains the correct mapping --
``compute_kpool_tail_slot_mapping`` in ``v1/attention/backends/mla/indexer.py``,
called from ``KpoolTailMetadataBuilder.build``.  It is simply never reached:

    positions = common_attn_metadata.positions
    if positions is not None:          # silent fall-through
        slot_mapping = compute_kpool_tail_slot_mapping(...)

``positions`` is None on every call for this model, because GLM-5.3-Flash is a
hybrid (KDA) model and its attention metadata is built by
``v1/worker/gpu/model_states/mamba_hybrid.py``, which calls
``build_attn_metadata(...)`` WITHOUT ``positions=`` -- while the plain
transformer path in ``model_states/default.py`` passes
``positions=input_batch.positions``.  A guard that degrades silently to
incorrect addressing instead of failing.

WHAT THIS DOES
--------------
One knob, default OFF (unset == upstream behaviour, byte for byte):

  HAREM_KPOOL_TAIL_FIX=1
      K1  ``mamba_hybrid.py`` passes ``positions=input_batch.positions``,
          mirroring ``default.py``, so the correct mapping actually runs.
          ``common_attn_metadata.positions`` has exactly one other consumer on
          this platform (``cpu_attn.py``), so nothing else on the CUDA path
          changes behaviour.
      K2  ``compute_kpool_tail_slot_mapping`` writes the caller's buffer in
          place instead of returning ``slot_mapping.clone()``.  That buffer IS
          the tail group's persistent slot mapping, so in place is the correct
          semantics; a fresh clone is also what CUDA-graph capture pins at a
          transient address, which is read back stale on replay (Xid 13, Out Of
          Range Address).  This stack serves eager today, so K2 is latent here
          and is carried because K1 without it is a trap for anyone who turns
          graphs on.

  HAREM_KPOOL_TAIL_BOUNDS=1
      D   Detector: count, per step, how many tokens of the tail group's
          GENERIC mapping would address a block outside the group's own block
          table row, and how many of the mapping's slots fall outside the tail
          cache.  Prints a running total on the engine log.  Inert unless the
          knob is set.  See PROVENANCE for why a counter and not a crash test.

PROVENANCE
----------
Diagnosis, reproducer and first fix: **Victor Cruz (vcruz305)**,
``GLM-5.3-Flash-EXL3-K2-DGX-Spark-recipe``, ``docs/KPOOL_TAIL_BUG.md`` and
``scripts/patch_kpool_tail_positions.py`` (2026-08-30).  Both edits below are
his, applied to our tree behind an env gate; the anchors matched this image
byte for byte.  His notes record two things we did not have to rediscover:
clamping the block index inside the generic slot kernel changes nothing (48
overruns before, 48 after) because the tail mapping does not come from that
kernel once positions are present; and Python-side instrumentation inside a
CUDA-graph-captured op runs at capture time only, so only device-updated
counters are evidence.  The K-pool machinery itself comes from vLLM PR #53906
(ZJY0516, crediting JaredforReal), whose
``tests/v1/attention/test_kpool_tail_slot_mapping.py`` pins the intended
one-block circular addressing.  Reported independently by Suppressor72
(vLLM issue #53670) and UserHIJ.

FAIL-CLOSED
-----------
Three anchors, each required exactly once; a drifted anchor stops the rank.
"""

import argparse
import os
import sys

REL_HYBRID = "v1/worker/gpu/model_states/mamba_hybrid.py"
REL_INDEXER = "v1/attention/backends/mla/indexer.py"

MARK = "HAREM-TP3 kpool-tail"

# --- K1: the missing positions argument on the hybrid path -------------------
OLD_HYBRID = """            dcp_local_seq_lens=input_batch.dcp_local_seq_lens,
            model_specific_attn_metadata=mamba_attn_metadata,
            for_cudagraph_capture=for_capture,
            rswa_prefix_lens=input_batch.prompt_lens,
        )
"""

NEW_HYBRID = '''            dcp_local_seq_lens=input_batch.dcp_local_seq_lens,
            # HAREM-TP3 kpool-tail K1 (vcruz305): hybrid models never passed
            # positions here, unlike default.py.  The K-pool tail builder needs
            # them: without positions it skips compute_kpool_tail_slot_mapping
            # and uses the generic paged mapping against a row whose only
            # written column is 0, which addresses the tail cache out of
            # bounds.  Unset HAREM_KPOOL_TAIL_FIX == upstream behaviour.
            positions=(
                input_batch.positions if _HAREM_KPOOL_TAIL_FIX else None
            ),
            model_specific_attn_metadata=mamba_attn_metadata,
            for_cudagraph_capture=for_capture,
            rswa_prefix_lens=input_batch.prompt_lens,
        )
'''

HYBRID_CONST = '''
# HAREM-TP3 kpool-tail K1 gate.  Read once at import: the knob is a container
# environment variable and cannot change under a running engine.
_HAREM_KPOOL_TAIL_FIX = os.environ.get("HAREM_KPOOL_TAIL_FIX", "").strip() in (
    "1",
    "true",
    "True",
)
'''

# --- K2: write the tail slot mapping in place --------------------------------
OLD_INDEXER = """    out = slot_mapping.clone()
    if num_actual_tokens == 0:
        return out
"""

NEW_INDEXER = '''    # HAREM-TP3 kpool-tail K2 (vcruz305): slot_mapping IS the tail group's
    # persistent buffer, so writing it in place is the correct semantics.  A
    # fresh clone is captured by CUDA graphs at a transient address and read
    # back stale on replay (Xid 13, Out Of Range Address).  Unset
    # HAREM_KPOOL_TAIL_FIX == upstream behaviour (a clone).
    out = slot_mapping if _HAREM_KPOOL_TAIL_FIX else slot_mapping.clone()
    if num_actual_tokens == 0:
        return out
'''

# --- D: the detector, on the tail metadata builder ---------------------------
# The audit runs AFTER the correction and is handed BOTH mappings, because the
# fix does not repair the generic kernel -- it stops the tail from using it.
# A detector that only looked at the generic mapping would report the same
# number in both arms and prove nothing.
OLD_BUILD = """        slot_mapping = common_attn_metadata.slot_mapping
        positions = common_attn_metadata.positions
        if positions is not None:
            # Circular per-request layout; the generic kernel output collapses
            # onto tail block 0 for pos >= kpool (see compute_... docstring).
            slot_mapping = compute_kpool_tail_slot_mapping(
                slot_mapping,
                common_attn_metadata.block_table_tensor,
                common_attn_metadata.query_start_loc,
                positions,
                common_attn_metadata.num_actual_tokens,
                common_attn_metadata.num_reqs,
                self.kv_cache_spec.block_size,
            )
        return DeepseekV32IndexerMetadata(
"""

NEW_BUILD = '''        slot_mapping = common_attn_metadata.slot_mapping
        positions = common_attn_metadata.positions
        # HAREM-TP3 kpool-tail D: keep what the GENERIC per-group kernel
        # produced, before any correction, so the detector can compare the two.
        _harem_generic = slot_mapping.clone() if _HAREM_KPOOL_TAIL_BOUNDS else None
        if positions is not None:
            # Circular per-request layout; the generic kernel output collapses
            # onto tail block 0 for pos >= kpool (see compute_... docstring).
            slot_mapping = compute_kpool_tail_slot_mapping(
                slot_mapping,
                common_attn_metadata.block_table_tensor,
                common_attn_metadata.query_start_loc,
                positions,
                common_attn_metadata.num_actual_tokens,
                common_attn_metadata.num_reqs,
                self.kv_cache_spec.block_size,
            )
        if _HAREM_KPOOL_TAIL_BOUNDS:
            _harem_kpool_tail_audit(
                common_attn_metadata,
                self.kv_cache_spec,
                _harem_generic,
                slot_mapping,
                positions is not None,
            )
        return DeepseekV32IndexerMetadata(
'''

INDEXER_CONST = '''
# --- HAREM-TP3 kpool-tail: gate + detector -----------------------------------
# Read once at import: both knobs are container environment variables and
# cannot change under a running engine.
_HAREM_KPOOL_TAIL_FIX = os.environ.get("HAREM_KPOOL_TAIL_FIX", "").strip() in (
    "1",
    "true",
    "True",
)
_HAREM_KPOOL_TAIL_BOUNDS = os.environ.get(
    "HAREM_KPOOL_TAIL_BOUNDS", ""
).strip() in ("1", "true", "True")
_HAREM_KPOOL_AUDIT = {
    "calls": 0,
    "steps_with_positions": 0,
    "tokens": 0,
    "generic_wrong_block": 0,
    "generic_worst_block": 0,
    "used_wrong_block": 0,
    "used_worst_block": 0,
    "row_overruns": 0,
    "worst_col": 0,
    "own_max": 0,
    "width": 0,
}


def _harem_kpool_tail_audit(
    common_attn_metadata,
    kv_cache_spec,
    generic,
    final,
    had_positions,
) -> None:
    """Count tail slots that address a block other than the request's own.

    The tail cache is one block per request, addressed
    ``block_table[req, 0] * kpool + pos % kpool``.  Any token whose slot
    divides down to a different block is writing into another request's ring
    -- or, past the end of the cache, into another layer's tensor.  Three
    counters:

      used_wrong_block     tokens of the mapping the engine ACTUALLY uses this
                           step whose block is not the request's own.  This is
                           the number that must be zero, and it is the one that
                           differs between the arms.
      generic_wrong_block  the same count for what the generic per-group kernel
                           produced.  The fix does not repair that kernel, so
                           this number is the same in both arms by
                           construction; it is here to show what the tail was
                           being handed.
      row_overruns         tokens whose ``pos // block_size`` is at or past the
                           width of the tail's block-table row, i.e. the reads
                           the generic kernel makes outside the row.  Positions
                           are reconstructed from ``seq_lens`` and
                           ``query_start_loc`` so this is countable in both
                           arms, including the one where positions are None.

    Host-side reductions once per step.  This stack serves eager, so every step
    is really observed; under CUDA graphs a Python counter would see capture
    only, which is the trap vcruz305 documented -- do not arm this with graphs
    on and believe a zero.
    """
    import torch

    a = _HAREM_KPOOL_AUDIT
    n = int(common_attn_metadata.num_actual_tokens)
    if n <= 0 or final is None:
        return
    bt = common_attn_metadata.block_table_tensor
    num_reqs = int(common_attn_metadata.num_reqs)
    if num_reqs <= 0 or bt.numel() == 0:
        return
    bs = int(kv_cache_spec.block_size)
    dev = final.device
    qsl = common_attn_metadata.query_start_loc[: num_reqs + 1].to(torch.int64)
    tok = torch.arange(n, device=dev, dtype=torch.int64)
    req = (torch.searchsorted(qsl, tok, right=True) - 1).clamp_(0, num_reqs - 1)
    own = bt[:num_reqs, 0].to(torch.int64).index_select(0, req)

    a["calls"] += 1
    a["tokens"] += n
    a["steps_with_positions"] += int(bool(had_positions))
    a["width"] = int(bt.shape[1])
    a["own_max"] = max(a["own_max"], int(own.max().item()))

    def _wrong(mapping):
        m = mapping[:n].to(torch.int64)
        blk = torch.where(m >= 0, torch.div(m, bs, rounding_mode="floor"), own)
        return int((blk != own).sum().item()), int(blk.max().item())

    w, worst = _wrong(final)
    a["used_wrong_block"] += w
    a["used_worst_block"] = max(a["used_worst_block"], worst)
    if generic is not None:
        gw, gworst = _wrong(generic)
        a["generic_wrong_block"] += gw
        a["generic_worst_block"] = max(a["generic_worst_block"], gworst)

    # Positions as the generic kernel saw them, reconstructed so the count
    # exists in the arm where positions are None:
    #   pos(t) = seq_lens[r] - query_start_loc[r+1] + t   for t in request r
    seq = common_attn_metadata.seq_lens[:num_reqs].to(torch.int64)
    counts = (qsl[1:] - qsl[:-1]).clamp_min(0)
    if int(counts.sum().item()) == n:
        base = torch.repeat_interleave(seq - qsl[1:], counts)
        pos = base + tok
        col = torch.div(pos, bs, rounding_mode="floor")
        a["row_overruns"] += int((col >= a["width"]).sum().item())
        a["worst_col"] = max(a["worst_col"], int(col.max().item()))

    if a["calls"] % 128 == 0:
        logger.info(
            "HAREM-TP3 KPOOL_TAIL_BOUNDS fix=%s calls=%d (with positions %d) "
            "tokens=%d | USED wrong_block=%d worst_block=%d | GENERIC "
            "wrong_block=%d worst_block=%d | row_overruns=%d worst_col=%d "
            "row_width=%d own_max=%d",
            _HAREM_KPOOL_TAIL_FIX,
            a["calls"],
            a["steps_with_positions"],
            a["tokens"],
            a["used_wrong_block"],
            a["used_worst_block"],
            a["generic_wrong_block"],
            a["generic_worst_block"],
            a["row_overruns"],
            a["worst_col"],
            a["width"],
            a["own_max"],
        )
'''

ANCHOR_INDEXER_CONST_BEFORE = """def compute_kpool_tail_slot_mapping(
"""

ANCHOR_HYBRID_CONST_BEFORE = """@dataclass
class MambaHybridAttnMetadata(ModelSpecificAttnMetadata):
"""


def _need_import_os(text: str) -> bool:
    for line in text.splitlines():
        s = line.strip()
        # NOTE: `import os as <alias>` binds the ALIAS, not `os`.  The indexer
        # workspace patch inserts exactly that (`import os as _harem_idxws_os`),
        # and treating it as "os is available" cost one boot: the module raised
        # NameError at import and the prelude stopped the rank (8 Sep 2026).
        if s == "import os" or s.startswith("import os,"):
            return False
        if s.startswith("import os ") and " as " not in s:
            return False
    return True


def _add_import_os(text: str) -> str:
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.startswith("from __future__"):
            continue
        if line.startswith("import ") or line.startswith("from "):
            lines.insert(i, "import os\n")
            return "".join(lines)
    return "import os\n" + text


def _apply(path: str, rel: str, edits: list[tuple[str, str, str]]) -> None:
    src = open(path).read()
    if MARK in src:
        print(f"patch-kpooltail: already applied ({rel})")
        return
    for name, old, _new in edits:
        n = src.count(old)
        if n != 1:
            print(
                f"patch-kpooltail: ANCHOR {name} count={n} (expected 1) in {path}",
                file=sys.stderr,
            )
            sys.exit(3)
    out = src
    for _name, old, new in edits:
        out = out.replace(old, new, 1)
    if _need_import_os(out):
        out = _add_import_os(out)
    open(path, "w").write(out)
    print(f"patch-kpooltail: applied to {rel}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    a = ap.parse_args()

    hyb = os.path.join(a.root, REL_HYBRID)
    _apply(
        hyb,
        REL_HYBRID,
        [
            (
                "K1a",
                ANCHOR_HYBRID_CONST_BEFORE,
                HYBRID_CONST.lstrip("\n") + "\n" + ANCHOR_HYBRID_CONST_BEFORE,
            ),
            ("K1b", OLD_HYBRID, NEW_HYBRID),
        ],
    )

    idx_path = os.path.join(a.root, REL_INDEXER)
    _apply(
        idx_path,
        REL_INDEXER,
        [
            (
                "K2",
                ANCHOR_INDEXER_CONST_BEFORE,
                INDEXER_CONST.lstrip("\n") + "\n" + ANCHOR_INDEXER_CONST_BEFORE,
            ),
            ("K2b", OLD_INDEXER, NEW_INDEXER),
            ("D", OLD_BUILD, NEW_BUILD),
        ],
    )


if __name__ == "__main__":
    main()
