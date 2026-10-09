#!/usr/bin/env python3
"""HAREM-TP3 mambagrid (R1): align-mode prefill chunks follow the KDA state grid.

The 21d93d0d8 port of the scheduler half of vLLM #54076 (OPEN; head b3fcf7c2 of 29
September, unchanged on 6 October); the same behaviour as the previous stack's
patch-mambagrid-tp3.py (plus the review fix E4). Only ``v1/core/sched/scheduler.py``.

PROBLEM: ``_mamba_block_aligned_split`` aligns chunk ends to ``cache_config.block_size``.
That value is the SMALLEST of the prefix-cached groups (the DFlash2 drafter's sliding-window
group, 256), while the KDA state grid is 3,328. A chunk that ends on the 256 grid but off
the 3,328 grid makes the worker write the WRONG moment's state into that slot; on a prefix
hit the KDA layers then continue from a stale state.

BEHAVIOUR (``HAREM_MAMBA_GRID=1``; empty = exactly upstream):
  G0/G1  the mamba groups' own block size (it must be unique, otherwise an error at
         startup) goes to ``self.mamba_state_block_size``; boot line [HAREM-MAMBAGRID].
  G2     chunk grid = state grid (including the internal-checkpoint computation, as in
         the PR).
  G3     every boundary crossed on the grid ends the chunk (the PR's unconditional stop;
         internal-checkpoint mode is exempt, as in the PR).
  G4     (NOT in the PR; review finding E4) when the grid is wider than the budget, a
         sub-block chunk end is lowered to the fine grid (cache_config.block_size): the
         kpool compressor does not write a pool when a chunk starts in the middle of it
         (common/attention.py: "chunked-prefill boundaries stay pool-aligned").
         GUARD: when the fine grid is WIDER than the budget (a boot without a drafter:
         there is no 256 sliding-window group, so the fine grid is the KDA block; TP1
         8704, TP3 3328 > 2048), E4 does NOT engage and upstream behaviour applies (a
         sub-block chunk end = start + budget). Without the guard E4 rounded the chunk
         end down to 0 and the request waited forever (found with a TP1 serve of a
         3,000-token prompt). Logged once: "[HAREM-MAMBAGRID] E4 devre disi: ince=X >
         butce=Y (upstream)". Production (TP3 + DFlash2: fine 256 <= 2048) behaviour is
         UNCHANGED.
The PR's speculative.py half (limit eagle block dropping to eagle/eagle3/mtp) was NOT
ported: DFlash2 flags the drafter KV groups with upstream's is_eagle_group
(kv_cache_utils.py:2222-2245); see yama-dflash-kvgrup (P4).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harem_yama as Y  # noqa: E402

SCHED = "v1/core/sched/scheduler.py"

YARDIMCI = '''

def _harem_mamba_state_block_size(kv_cache_config) -> int | None:
    """HAREM-TP3 mambagrid (port of vLLM #54076): the mamba groups' own state
    block when HAREM_MAMBA_GRID=1, else None (upstream: split on
    cache_config.block_size). Fails closed on several mamba block sizes."""
    import os as _harem_os

    sizes = {
        g.kv_cache_spec.block_size
        for g in kv_cache_config.kv_cache_groups
        if isinstance(g.kv_cache_spec, MambaSpec)
    }
    if len(sizes) > 1:
        raise ValueError(
            "HAREM_MAMBA_GRID: mamba align scheduling requires a single mamba "
            f"state block size, got {sorted(sizes)}"
        )
    if not sizes or _harem_os.environ.get("HAREM_MAMBA_GRID", "").strip() != "1":
        return None
    return next(iter(sizes))


_HAREM_E4_DEVRE_DISI_LOGLANDI = False


def _harem_e4_devre_disi(ince: int, butce: int) -> None:
    """HAREM-TP3 mambagrid E4 guard: log once that the fine-grid
    floor is skipped because the fine grid is wider than the prefill budget."""
    global _HAREM_E4_DEVRE_DISI_LOGLANDI
    if not _HAREM_E4_DEVRE_DISI_LOGLANDI:
        _HAREM_E4_DEVRE_DISI_LOGLANDI = True
        logger.info(
            "[HAREM-MAMBAGRID] E4 devre disi: ince=%d > butce=%d (upstream)",
            ince, butce,
        )
'''

D = Y.Duzen
DUZENLER = [
    D("G0-yardimci", SCHED,
      "logger = init_logger(__name__)\n\n\nclass Scheduler(SchedulerInterface):",
      "logger = init_logger(__name__)\n" + YARDIMCI + "\n\nclass Scheduler(SchedulerInterface):"),
    D("G1-baslat", SCHED,
      "        self.mamba_shared_prefix_checkpoint = (\n"
      "            self.mamba_partial_cache_hit\n"
      "            and self.kv_cache_manager.mamba_shared_prefix_checkpoint\n"
      "        )\n",
      "        self.mamba_shared_prefix_checkpoint = (\n"
      "            self.mamba_partial_cache_hit\n"
      "            and self.kv_cache_manager.mamba_shared_prefix_checkpoint\n"
      "        )\n"
      "        # HAREM-TP3 mambagrid: see _harem_mamba_state_block_size.\n"
      "        self.mamba_state_block_size = _harem_mamba_state_block_size(kv_cache_config)\n"
      "        if self.need_mamba_block_aligned_split:\n"
      "            logger.info(\n"
      "                \"[HAREM-MAMBAGRID] split_grid=%s (cache_config.block_size=%s)\",\n"
      "                self.mamba_state_block_size or self.cache_config.block_size,\n"
      "                self.cache_config.block_size,\n"
      "            )\n"),
    D("G2-izgara", SCHED,
      "        block_size = self.cache_config.block_size\n"
      "        # The last block-aligned position whose state can be cached. With\n",
      "        block_size = self.cache_config.block_size\n"
      "        # HAREM-TP3 mambagrid: chunk ends are where the worker writes a mamba\n"
      "        # state, so they follow the state grid when HAREM_MAMBA_GRID=1.\n"
      "        if getattr(self, \"mamba_state_block_size\", None) is not None:\n"
      "            block_size = self.mamba_state_block_size\n"
      "        # The last block-aligned position whose state can be cached. With\n"),
    D("G3-durak", SCHED,
      "            next_block_boundary\n"
      "            if start % block_size != 0 and not use_internal_checkpoint\n"
      "            else 0,\n",
      "            # HAREM-TP3 mambagrid (#54076): on the state grid every crossed\n"
      "            # boundary ends a chunk, whether the chunk started aligned or not.\n"
      "            next_block_boundary\n"
      "            if (\n"
      "                start % block_size != 0\n"
      "                or getattr(self, \"mamba_state_block_size\", None) is not None\n"
      "            )\n"
      "            and not use_internal_checkpoint\n"
      "            else 0,\n"),
    D("G4-ince-taban", SCHED,
      "            aligned_end = end // block_size * block_size\n"
      "            if aligned_end > start or block_size <= max_prefill_tokens:\n"
      "                end = aligned_end\n",
      "            aligned_end = end // block_size * block_size\n"
      "            if aligned_end > start or block_size <= max_prefill_tokens:\n"
      "                end = aligned_end\n"
      "            elif getattr(self, \"mamba_state_block_size\", None) is not None:\n"
      "                # HAREM-TP3 mambagrid (review fix E4, not in #54076): a sub-block\n"
      "                # chunk of the wide state grid still ends on the fine grid\n"
      "                # (cache_config.block_size, a multiple of index_kpool): the kpool\n"
      "                # prefill compressor drops a pool that straddles a chunk start.\n"
      "                # Guard: a fine grid wider than the budget (no\n"
      "                # draft SW group: fine == state grid) would floor the end to\n"
      "                # the chunk start forever -> keep upstream's sub-block chunk.\n"
      "                _harem_fine = self.cache_config.block_size\n"
      "                if _harem_fine <= max_prefill_tokens:\n"
      "                    end = end // _harem_fine * _harem_fine\n"
      "                else:\n"
      "                    _harem_e4_devre_disi(_harem_fine, max_prefill_tokens)\n"),
]

YAMA = Y.Yama(
    ad="yama-mambagrid",
    aciklama="R1: mamba align parçaları KDA durum ızgarasında (vLLM #54076 portu + E4)",
    duzenler=DUZENLER,
    taban={SCHED: "57fcd1a79dc37573025ba0a07d21d36d5f1d1789bf2f6071dc598d23eff13a83"},
)

if __name__ == "__main__":
    sys.exit(Y.main(YAMA, __file__))
