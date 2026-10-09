#!/usr/bin/env python3
"""HAREM DFlash2 KV group: give the drafter's layers their own group on the GLM-5.3-specific
grouping path.

The 21d93d0d8 counterpart of two things in the previous stack's image: the
``vllm/v1/core/kv_cache_utils.py`` diff of its DFlash2 port layer (13 single-anchor edits
relative to 487ecf187; the idea comes from a vLLM fork's ``_partition_dflash_draft_specs``)
and the previous stack's patch-prefixhit-tp3.py (HAREM_PREFIX_HIT).

WHY (still true at 21d93d0d8, from reading the code):
  ``_get_kv_cache_groups_glm5_next`` requires every attention spec to be an
  ``MLAAttentionSpec``; the DFlash drafter adds sliding-window (SlidingWindowSpec)
  layers -> the path returns None -> the generic path is taken: LBHNC is not
  block-outside (the packed path returns None), ``unify_kv_cache_spec_page_size`` cannot
  unify the kpool indexer page (MLA, cannot be padded) -> NotImplementedError -> the
  full-separation fallback cannot reduce a Mamba + MLA + SW mix to one type -> the boot
  dies. Even if it did not die, GLM-5.3's slot sharing (mamba <-> MLA, kpool tail <->
  indexer) would be lost.

WHAT IT DOES (the same meaning as the previous port):
  P1  ``_harem_partition_dflash_draft_specs``: with method == "dflash", PP == 1 and the
      hybrid manager on, layers whose index is >= the target layer count count as the
      drafter. ``get_kv_cache_groups`` groups the target and the drafter SEPARATELY and
      joins them: the target's grouping stays bit for bit what it is without a drafter.
      Log: "DFlash draft: N KV layers kept in M independent cache group(s)".
  P2  ``_glm5_next_tensor_layout`` recognises the drafter groups (plain AttentionSpec
      groups that are neither UniformType nor Mamba) and returns them as the 9th element;
      with an unrecognised uniform group next to a drafter it fails LOUDLY.
  P3  Three consumers count the drafter's bytes: bytes per block
      (``_get_kv_cache_bytes_per_block`` -> null-block reserve and
      num_gpu_blocks_override), the KV configuration (a PRIVATE region for every drafter
      layer after the indexer region, in the same shared block-id space) and the peak
      memory estimate (as in the previous port: added to the multiplier).
  P4  HAREM_PREFIX_HIT=1 (set in the production environment): ONLY the drafter groups are
      flagged ``is_eagle_group``; otherwise KVCacheCoordinator flags all of them and
      drops an aligned target block on every exact-repeat prefix hit (previous
      measurement: 71 % / 89 % against a ceiling of 97.3 %). Empty = upstream.
      Fail-closed: it dies if the drafter groups are not all sliding-window or a target
      group is already flagged. It flags only when ``use_eagle_block_drop()`` is true (the
      same precondition as upstream's ``_annotate_eagle_groups``).

The previous port's HAREM_EAGLE_BLOCK_DROP=0 diagnostic arm was NOT ported: upstream now
has ``speculative_config.disable_eagle_block_drop`` (#53388). The previous production did
not set it; this track does (see the track README).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harem_yama as Y  # noqa: E402

KVU = "v1/core/kv_cache_utils.py"

YARDIMCILAR = '''def _harem_partition_dflash_draft_specs(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
) -> tuple[dict[str, KVCacheSpec], dict[str, KVCacheSpec]]:
    """HAREM-DFLASH-KVGRUP: split the DFlash draft's appended KV layers off.

    The DFlash head is built with ``start_layer_id = <target layer count>``, so
    its attention layers carry indices at or above that count. Grouping target
    and draft separately keeps the target on its specialised GLM-5.3 path
    (``_get_kv_cache_groups_glm5_next``: mamba layers co-own the MLA slots, the
    kpool tail co-owns the indexer slots) bit for bit as without a drafter; the
    draft's sliding-window layers would otherwise disqualify that path and the
    generic path cannot unify the kpool indexer page.
    """
    from vllm.model_executor.models.utils import extract_layer_index

    speculative_config = vllm_config.speculative_config
    if (
        speculative_config is None
        or speculative_config.method != "dflash"
        or vllm_config.parallel_config.pipeline_parallel_size > 1
        or vllm_config.scheduler_config.disable_hybrid_kv_cache_manager
    ):
        return kv_cache_spec, {}

    target_num_layers = vllm_config.model_config.get_num_layers(
        vllm_config.parallel_config
    )
    target_specs: dict[str, KVCacheSpec] = {}
    draft_specs: dict[str, KVCacheSpec] = {}
    for layer_name, spec in kv_cache_spec.items():
        try:
            layer_index = extract_layer_index(layer_name)
        except (AssertionError, IndexError, ValueError):
            target_specs[layer_name] = spec
            continue
        if layer_index >= target_num_layers:
            draft_specs[layer_name] = spec
        else:
            target_specs[layer_name] = spec
    return target_specs, draft_specs


def _harem_draft_bytes_per_block(
    draft_groups: list[KVCacheGroupSpec],
) -> int:
    """HAREM-DFLASH-KVGRUP: bytes one shared block id costs the draft's groups.

    Every group draws block ids from the same pool and each draft layer owns a
    private region, so the draft's pages add to the per-block cost exactly like
    the MLA and indexer pages. Leaving this out would over-report how many
    blocks fit and leave the draft's tensors unallocated.
    """
    return sum(
        len(group.layer_names) * group.kv_cache_spec.page_size_bytes
        for group in draft_groups
    )


def _harem_annotate_draft_eagle_groups(
    vllm_config: VllmConfig,
    target_groups: list[KVCacheGroupSpec],
    draft_groups: list[KVCacheGroupSpec],
) -> None:
    """HAREM-DFLASH-KVGRUP prefix-hit: flag ONLY the drafter's groups as EAGLE.

    Port of the production ``patch-prefixhit-tp3.py`` (after vllm#52047 and the
    idea of #54041). Upstream's ``_annotate_eagle_groups`` never runs on this
    path and has no rule for a sliding-window DFlash group, so
    ``KVCacheCoordinator`` would fall back to flagging every group and drop a
    whole aligned target block from every exact-repeat prefix hit.
    ``HAREM_PREFIX_HIT`` unset -> upstream behaviour. Fails closed.
    """
    if os.environ.get("HAREM_PREFIX_HIT", "").strip() not in ("1", "true", "True"):
        return
    spec_config = vllm_config.speculative_config
    if spec_config is None or not spec_config.use_eagle_block_drop():
        return
    if not draft_groups:
        raise ValueError("HAREM_PREFIX_HIT: the DFlash branch produced no draft group")
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
            f"({not_swa}); a mixed drafter stays on the conservative flag-all "
            "fallback -- unset HAREM_PREFIX_HIT"
        )
    for g in draft_groups:
        g.is_eagle_group = True
    logger.info(
        "HAREM prefix-hit: %d drafter group(s) flagged is_eagle_group; "
        "target groups left unflagged",
        len(draft_groups),
    )


'''

D = Y.Duzen
DUZENLER = [
    D("P2a-duzen-imza", KVU,
      "        list[str],\n"
      "        int,\n"
      "    ]\n"
      "    | None\n"
      "):\n"
      "    \"\"\"Recognize the GLM-5.3-Flash grouping after optional PP projection.\"\"\"\n"
      "    uniform_groups = [\n"
      "        group\n"
      "        for group in kv_cache_groups\n"
      "        if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)\n"
      "    ]\n"
      "    mamba_groups = [\n"
      "        group for group in kv_cache_groups if isinstance(group.kv_cache_spec, MambaSpec)\n"
      "    ]\n",
      "        list[str],\n"
      "        int,\n"
      "        list[KVCacheGroupSpec],  # HAREM-DFLASH-KVGRUP: draft groups\n"
      "    ]\n"
      "    | None\n"
      "):\n"
      "    \"\"\"Recognize the GLM-5.3-Flash grouping after optional PP projection.\"\"\"\n"
      "    uniform_groups = [\n"
      "        group\n"
      "        for group in kv_cache_groups\n"
      "        if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)\n"
      "    ]\n"
      "    mamba_groups = [\n"
      "        group for group in kv_cache_groups if isinstance(group.kv_cache_spec, MambaSpec)\n"
      "    ]\n"
      "    # HAREM-DFLASH-KVGRUP: a DFlash draft contributes its own independent\n"
      "    # group(s) (plain attention specs, neither UniformType nor Mamba). They take\n"
      "    # no part in the slot sharing but draw block ids from the shared pool.\n"
      "    draft_groups = [\n"
      "        group\n"
      "        for group in kv_cache_groups\n"
      "        if not isinstance(group.kv_cache_spec, (UniformTypeKVCacheSpecs, MambaSpec))\n"
      "    ]\n"),
    D("P2b-duzen-sayim", KVU,
      "    if attn_group is None or not mamba_groups:\n"
      "        return None\n"
      "    if len(uniform_groups) + len(mamba_groups) != len(kv_cache_groups):\n"
      "        return None\n",
      "    if attn_group is None or not mamba_groups:\n"
      "        return None\n"
      "    # HAREM-DFLASH-KVGRUP: only plain attention groups may ride alongside.\n"
      "    if len(uniform_groups) + len(mamba_groups) + len(draft_groups) != len(\n"
      "        kv_cache_groups\n"
      "    ):\n"
      "        return None\n"
      "    if not all(isinstance(g.kv_cache_spec, AttentionSpec) for g in draft_groups):\n"
      "        return None\n"
      "    if draft_groups and any(\n"
      "        g is not attn_group and g is not tail_group for g in uniform_groups\n"
      "    ):\n"
      "        raise ValueError(\n"
      "            \"HAREM-DFLASH-KVGRUP: an unrecognised uniform KV cache group rides \"\n"
      "            \"next to the DFlash draft; its pages would go unaccounted\"\n"
      "        )\n"),
    D("P2c-duzen-donus", KVU,
      "        tail_names,\n"
      "        tail_page,\n"
      "    )\n",
      "        tail_names,\n"
      "        tail_page,\n"
      "        draft_groups,  # HAREM-DFLASH-KVGRUP\n"
      "    )\n"),
    D("P3a-blok-bayt", KVU,
      "        _, _, mla_names, idx_names, mla_page, idx_page, _, _ = glm5_layout\n"
      "        return len(mla_names) * mla_page + len(idx_names) * idx_page\n",
      "        _, _, mla_names, idx_names, mla_page, idx_page, _, _, draft_groups = (\n"
      "            glm5_layout\n"
      "        )\n"
      "        return (\n"
      "            len(mla_names) * mla_page\n"
      "            + len(idx_names) * idx_page\n"
      "            + _harem_draft_bytes_per_block(draft_groups)  # HAREM-DFLASH-KVGRUP\n"
      "        )\n"),
    D("P3b-yapilandirma-bayt", KVU,
      "            tail_names,\n"
      "            _,\n"
      "        ) = glm5_layout\n"
      "        bytes_per_block = len(mla_names) * mla_page + len(idx_names) * idx_page\n",
      "            tail_names,\n"
      "            _,\n"
      "            draft_groups,\n"
      "        ) = glm5_layout\n"
      "        bytes_per_block = (\n"
      "            len(mla_names) * mla_page\n"
      "            + len(idx_names) * idx_page\n"
      "            + _harem_draft_bytes_per_block(draft_groups)  # HAREM-DFLASH-KVGRUP\n"
      "        )\n"),
    D("P3c-yapilandirma-tensor", KVU,
      "                add_tensor(tail_name, tail_specs[tail_name], offset)\n"
      "\n"
      "        return KVCacheConfig(\n",
      "                add_tensor(tail_name, tail_specs[tail_name], offset)\n"
      "\n"
      "        # HAREM-DFLASH-KVGRUP: each DFlash draft layer gets a private region\n"
      "        # after the indexer region, sized to the same shared block-id space.\n"
      "        draft_offset = idx_base + len(idx_names) * idx_page * num_blocks\n"
      "        for group in draft_groups:\n"
      "            for layer_name in group.layer_names:\n"
      "                add_tensor(layer_name, group.kv_cache_spec, draft_offset)\n"
      "                draft_offset += group.kv_cache_spec.page_size_bytes * num_blocks\n"
      "        assert draft_offset <= size, (\n"
      "            \"HAREM-DFLASH-KVGRUP: draft regions overrun the KV allocation\"\n"
      "        )\n"
      "\n"
      "        return KVCacheConfig(\n"),
    D("P3d-bellek-tahmini", KVU,
      "            tail_names,\n"
      "            _,\n"
      "        ) = glm5_layout\n"
      "        uniform_spec = cast(UniformTypeKVCacheSpecs, attn_group.kv_cache_spec)\n",
      "            tail_names,\n"
      "            _,\n"
      "            draft_groups,  # HAREM-DFLASH-KVGRUP\n"
      "        ) = glm5_layout\n"
      "        uniform_spec = cast(UniformTypeKVCacheSpecs, attn_group.kv_cache_spec)\n"),
    D("P3e-bellek-carpan", KVU,
      "        if tail_names:\n"
      "            total_blocks += 1\n"
      "        return total_blocks * (len(mla_names) * mla_page + len(idx_names) * idx_page)\n",
      "        if tail_names:\n"
      "            total_blocks += 1\n"
      "        return total_blocks * (\n"
      "            len(mla_names) * mla_page\n"
      "            + len(idx_names) * idx_page\n"
      "            + _harem_draft_bytes_per_block(draft_groups)  # HAREM-DFLASH-KVGRUP\n"
      "        )\n"),
    D("P1a-yardimcilar", KVU,
      "def get_kv_cache_groups(\n"
      "    vllm_config: VllmConfig,\n"
      "    kv_cache_spec: dict[str, KVCacheSpec],\n"
      ") -> list[KVCacheGroupSpec]:\n",
      YARDIMCILAR
      + "def get_kv_cache_groups(\n"
      "    vllm_config: VllmConfig,\n"
      "    kv_cache_spec: dict[str, KVCacheSpec],\n"
      ") -> list[KVCacheGroupSpec]:\n"),
    D("P1b-bolme", KVU,
      "        # attention free models.\n"
      "        return []\n"
      "\n"
      "    if hisparse_groups := get_hisparse_kv_cache_groups(vllm_config, kv_cache_spec):\n",
      "        # attention free models.\n"
      "        return []\n"
      "\n"
      "    # HAREM-DFLASH-KVGRUP: group a DFlash draft's KV layers independently of\n"
      "    # the target's, so the target keeps the specialised path it takes on its\n"
      "    # own (GLM-5.3: _get_kv_cache_groups_glm5_next).\n"
      "    target_specs, draft_specs = _harem_partition_dflash_draft_specs(\n"
      "        vllm_config, kv_cache_spec\n"
      "    )\n"
      "    if target_specs and draft_specs:\n"
      "        target_groups = get_kv_cache_groups(vllm_config, target_specs)\n"
      "        draft_groups = get_kv_cache_groups(vllm_config, draft_specs)\n"
      "        logger.info(\n"
      "            \"DFlash draft: %d KV layers kept in %d independent cache group(s)\",\n"
      "            len(draft_specs),\n"
      "            len(draft_groups),\n"
      "        )\n"
      "        _harem_annotate_draft_eagle_groups(vllm_config, target_groups, draft_groups)\n"
      "        return [*target_groups, *draft_groups]\n"
      "\n"
      "    if hisparse_groups := get_hisparse_kv_cache_groups(vllm_config, kv_cache_spec):\n"),
]

YAMA = Y.Yama(
    ad="yama-dflash-kvgrup",
    aciklama="DFlash2 taslak KV grubu GLM-5.3 özel düzeninde (eski port kv_cache_utils + prefixhit)",
    duzenler=DUZENLER,
    taban={
        KVU: "6cd5a3def8f7c9af4006e4c7f9302b160413e8ee049d278ed4edf62215338f1f",
    },
)

if __name__ == "__main__":
    sys.exit(Y.main(YAMA, __file__))
