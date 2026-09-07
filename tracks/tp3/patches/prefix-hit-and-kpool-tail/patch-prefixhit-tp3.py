#!/usr/bin/env python3
"""HAREM-TP3 prefix-hit: annotate ONLY the DFlash2 drafter's KV cache groups as
EAGLE groups, and expose upstream's "do not drop the trailing block" switch.

WHAT IS WRONG WITHOUT THIS
--------------------------
This image's ``_annotate_eagle_groups_deepseek_v4`` flags a group only when the
merged spec says ``model_version == "deepseek_v4"``, and it is called only from
the ``group_and_unify_kv_cache_specs`` branch.  GLM-5.3-Flash with a DFlash2
drafter takes a different branch entirely -- ``_harem_partition_dflash_draft_specs``
groups the target and the draft separately and returns ``[*target, *draft]``
without ever reaching an annotation site.  So no group carries
``is_eagle_group`` and ``KVCacheCoordinator.__init__`` takes its conservative
fallback:

    if use_eagle and not self.eagle_group_ids:
        self.eagle_group_ids = set(range(len(kv_cache_config.kv_cache_groups)))

Every group then takes the EAGLE last-block drop: the target MLA group (block
3,328 tokens) and the four mamba groups as well as the drafter's sliding-window
group.  ``cache_blocks`` gives an eagle group one block of lookahead beyond the
aligned boundary, which is exactly what makes the drop free -- but the target
group only reaches that lookahead when the request is a whole block longer than
its aligned prefix, so in practice the drop costs a real 3,328-token block, and
the coordinator's fixed-point loop then re-runs and the sliding-window group's
re-alignment pop costs another.  Measured on an exact repeat of eight ~65k
prompts: 71 % (first repeat) / 89 % (later repeats) against a 97.3 % ceiling.

WHAT THIS DOES
--------------
Two knobs, both default OFF (unset == upstream behaviour, byte for byte):

  HAREM_PREFIX_HIT=1
      In the DFlash branch of ``get_kv_cache_groups``, flag the drafter's
      groups -- and only those -- with ``is_eagle_group = True``.  The
      coordinator then stops flagging everything, the target MLA group keeps
      its whole cached prefix, and the drafter's sliding-window group pops its
      one lookahead block and re-aligns back to exactly the aligned boundary.

  HAREM_EAGLE_BLOCK_DROP=0
      Drop nothing anywhere: ``eagle_group_ids`` stays empty even under
      speculative decoding.  This is upstream #53388's
      ``disable_eagle_block_drop`` reduced to the one place this stack needs it.
      DIAGNOSTIC ARM, not a production candidate: without the drop the drafter
      has no recomputed boundary state, so acceptance is the thing to watch.

PROVENANCE
----------
Ported from, and credited to, the vLLM upstream work:
  * #52047, okorzh-amd, merged 29 Aug 2026 -- generalises the DeepSeek-V4-only
    annotation into ``_annotate_eagle_groups`` driven by a spec-level marker,
    and warns when speculative decoding leaves every group unannotated.
  * #54041, positive666, 27 Aug 2026 (closed) -- the idea used here: mark the
    drafter's group as the EAGLE group only when *every* draft layer is
    sliding-window, and leave a mixed drafter on the conservative fallback.
  * #53388, ZeldaHuang, merged 1 Sep 2026 -- the ``disable_eagle_block_drop``
    speculative option.
  * Reported by Suppressor72 (vLLM issue #53670) and UserHIJ.
We do not carry #54041's plumbing of a ``non_causal_multi_token_decode`` marker
through ``Attention.__init__``: this stack already hands the grouping function
the drafter's groups as a separate list, so the marker would only be a longer
road to the same ``is_eagle_group``.  The all-sliding-window precondition of
#54041 is kept and enforced.

FAIL-CLOSED
-----------
Two anchors, each required exactly once.  At runtime the annotator refuses (a)
a drafter whose groups are not all sliding-window, and (b) a target group that
already carries the flag -- either means the topology moved under us and
guessing would silently change what is cached.
"""

import argparse
import os
import sys

REL_UTILS = "v1/core/kv_cache_utils.py"
REL_COORD = "v1/core/kv_cache_coordinator.py"

MARK = "HAREM-TP3 prefix-hit"

# --- anchor 1: the DFlash branch of get_kv_cache_groups ----------------------
OLD_UTILS = """        logger.info(
            "DFlash draft: %d KV layers kept in %d independent cache group(s)",
            len(draft_specs),
            len(draft_groups),
        )
        return [*target_groups, *draft_groups]
"""

NEW_UTILS = '''        logger.info(
            "DFlash draft: %d KV layers kept in %d independent cache group(s)",
            len(draft_specs),
            len(draft_groups),
        )
        _harem_annotate_draft_eagle_groups(target_groups, draft_groups)
        return [*target_groups, *draft_groups]
'''

HELPER = '''

def _harem_annotate_draft_eagle_groups(
    target_groups: list[KVCacheGroupSpec],
    draft_groups: list[KVCacheGroupSpec],
) -> None:
    """HAREM-TP3 prefix-hit: flag ONLY the drafter's groups as EAGLE groups.

    Port of vllm-project/vllm#52047 (okorzh-amd) plus the idea of #54041
    (positive666): the EAGLE last-block drop belongs to the drafter's own KV
    cache groups, not to every group in a hybrid layout.  Without this the
    GLM-5.3 + DFlash2 grouping path leaves every group unannotated and
    ``KVCacheCoordinator.__init__`` falls back to flagging all of them, which
    costs a whole aligned block of every exact-repeat prefix hit.

    ``HAREM_PREFIX_HIT`` unset or 0 -> upstream behaviour, byte for byte.

    Fails closed rather than guessing: a drafter that is not entirely
    sliding-window keeps the conservative fallback (#54041 leaves a mixed
    SWA/full drafter unmarked for the same reason), and a target group that
    already carries the flag means the topology changed under this patch.
    """
    if os.environ.get("HAREM_PREFIX_HIT", "").strip() not in ("1", "true", "True"):
        return
    if not draft_groups:
        raise ValueError(
            "HAREM_PREFIX_HIT: the DFlash branch produced no draft group"
        )
    already = [g for g in target_groups if getattr(g, "is_eagle_group", False)]
    if already:
        raise ValueError(
            "HAREM_PREFIX_HIT: a TARGET group is already flagged is_eagle_group "
            f"({[g.layer_names[0] for g in already]}); the grouping path changed "
            "-- refusing to annotate on top of it"
        )
    not_swa = [
        type(g.kv_cache_spec).__name__
        for g in draft_groups
        if not isinstance(g.kv_cache_spec, SlidingWindowSpec)
        or isinstance(g.kv_cache_spec, KpoolTailSpec)
    ]
    if not_swa:
        raise ValueError(
            "HAREM_PREFIX_HIT: the drafter's groups are not all sliding-window "
            f"({not_swa}); #54041 leaves a mixed drafter on the conservative "
            "flag-all fallback and so do we -- unset HAREM_PREFIX_HIT"
        )
    for g in draft_groups:
        g.is_eagle_group = True
    logger.info(
        "HAREM-TP3 prefix-hit: %d drafter group(s) flagged is_eagle_group; "
        "target groups left unflagged",
        len(draft_groups),
    )
'''

ANCHOR_HELPER_BEFORE = """def get_kv_cache_groups(
    vllm_config: VllmConfig, kv_cache_spec: dict[str, KVCacheSpec]
) -> list[KVCacheGroupSpec]:
"""

# --- anchor 2: the coordinator's flag-all fallback ---------------------------
OLD_COORD = """        # Conservatively fall back to flag all groups when no group is flagged.
        if use_eagle and not self.eagle_group_ids:
            self.eagle_group_ids = set(range(len(kv_cache_config.kv_cache_groups)))
"""

NEW_COORD = '''        # HAREM-TP3 prefix-hit: HAREM_EAGLE_BLOCK_DROP=0 is upstream #53388's
        # (ZeldaHuang) ``disable_eagle_block_drop`` reduced to the one decision
        # this stack needs -- speculation stays on, the trailing prefix-cache
        # block is not dropped anywhere.  DIAGNOSTIC ARM: the drafter then has
        # no recomputed boundary state, so watch acceptance.  Unset == upstream.
        if os.environ.get("HAREM_EAGLE_BLOCK_DROP", "").strip() == "0":
            logger.info(
                "HAREM-TP3 prefix-hit: HAREM_EAGLE_BLOCK_DROP=0 -- no group "
                "takes the EAGLE last-block drop (upstream #53388)"
            )
            self.eagle_group_ids = set()
        # Conservatively fall back to flag all groups when no group is flagged.
        elif use_eagle and not self.eagle_group_ids:
            self.eagle_group_ids = set(range(len(kv_cache_config.kv_cache_groups)))
'''


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


def patch_utils(root: str) -> None:
    p = os.path.join(root, REL_UTILS)
    src = open(p).read()
    if MARK in src:
        print(f"patch-prefixhit: already applied ({REL_UTILS})")
        return
    if src.count(OLD_UTILS) != 1:
        print(
            f"patch-prefixhit: ANCHOR-1 count={src.count(OLD_UTILS)} (expected 1) "
            f"in {p}",
            file=sys.stderr,
        )
        sys.exit(3)
    if src.count(ANCHOR_HELPER_BEFORE) != 1:
        print(
            f"patch-prefixhit: ANCHOR-1b count={src.count(ANCHOR_HELPER_BEFORE)} "
            f"(expected 1) in {p}",
            file=sys.stderr,
        )
        sys.exit(3)
    if "SlidingWindowSpec" not in src:
        print(
            "patch-prefixhit: SlidingWindowSpec is not imported in kv_cache_utils.py "
            "-- refusing",
            file=sys.stderr,
        )
        sys.exit(4)
    out = src.replace(OLD_UTILS, NEW_UTILS, 1)
    out = out.replace(ANCHOR_HELPER_BEFORE, HELPER.lstrip("\n") + "\n" + ANCHOR_HELPER_BEFORE, 1)
    if _need_import_os(out):
        out = _add_import_os(out)
    open(p, "w").write(out)
    print(f"patch-prefixhit: applied to {REL_UTILS} (HAREM_PREFIX_HIT honoured)")


def patch_coord(root: str) -> None:
    p = os.path.join(root, REL_COORD)
    src = open(p).read()
    if MARK in src:
        print(f"patch-prefixhit: already applied ({REL_COORD})")
        return
    if src.count(OLD_COORD) != 1:
        print(
            f"patch-prefixhit: ANCHOR-2 count={src.count(OLD_COORD)} (expected 1) "
            f"in {p}",
            file=sys.stderr,
        )
        sys.exit(3)
    out = src.replace(OLD_COORD, NEW_COORD, 1)
    if _need_import_os(out):
        out = _add_import_os(out)
    open(p, "w").write(out)
    print(
        f"patch-prefixhit: applied to {REL_COORD} (HAREM_EAGLE_BLOCK_DROP honoured)"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    patch_utils(a.root)
    patch_coord(a.root)


if __name__ == "__main__":
    main()
