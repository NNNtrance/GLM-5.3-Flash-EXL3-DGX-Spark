# `tracks/tp2/patches/prefix-hit-and-kpool-tail` — the two backports at two ranks

**Applies to: TP=2 only.** In the two-node recipe as of 8 September 2026, the same night they were
promoted at three ranks `[measured-here]`.

**There are no patch files in this directory either.** Both scripts are the three-node track's and
neither reads the rank count:

| | |
|---|---|
| `patch-prefixhit-tp3.py` | `v1/core/kv_cache_utils.py` (the DFlash branch) and `v1/core/kv_cache_coordinator.py`. The defect is the coordinator's "`use_eagle` and no group flagged → flag **every** group" fallback, which is a property of how the hybrid model groups its KV, not of tensor parallelism |
| `patch-kpooltail-tp3.py` | `v1/worker/gpu/model_states/mamba_hybrid.py` and `v1/attention/backends/mla/indexer.py`. The missing `positions=` argument is on the **hybrid** model-state path, which this model takes at any rank count |

The mechanism, the unit tests, the in-engine detector and the three-node measurements are
[`tracks/tp3/patches/prefix-hit-and-kpool-tail/`](../../../tp3/patches/prefix-hit-and-kpool-tail/README.md)
and [`results/gates/prefix-hit-and-kpool-tail.md`](../../../../results/gates/prefix-hit-and-kpool-tail.md).
The two-rank numbers are [docs/15](../../../../docs/15-tp2-track.md) §5.10.

---

## Install

Before the dump boot, because a file added to a patch directory changes the fast-load manifest
identity ([docs/08](../../../../docs/08-fast-boot.md) §4):

```
cp tracks/tp3/patches/prefix-hit-and-kpool-tail/patch-prefixhit-tp3.py tracks/tp3/patches/prefix-hit-and-kpool-tail/patch-kpooltail-tp3.py "$TREE"/
```

[`tracks/tp2/patches/tp2full-prelude.sh`](../tp2full-prelude.sh) applies both **unconditionally**,
immediately after `patch-indexer-workspace-tp3.py` — the same slot as the three-node prelude — and
their behaviour is env-gated, default off:

```
EXTRA_ENV += HAREM_PREFIX_HIT=1 HAREM_KPOOL_TAIL_FIX=1
```

With both knobs unset the tree is candidate C byte for byte, which is what makes an A/B possible on
one sidecar.

## The one thing that is **not** transferable: the block granularity

The prefix-hit ceiling is `floor((n − 1) / G) × G / n`, where **G is the block granularity in
tokens** — and G is pool arithmetic, so it is **not** the three-node 3,328. Measure it before you
quote a ceiling: ask the same prompt twice, read
`vllm:prefix_cache_hits_total` either side, and take the **greatest common divisor of the observed
hit counts**. `prefix-hit-probe.py --granularity <G>` then reports against the right ceiling.

That is what the three-node handover note asked for and it is the only rank-dependent thing in
either patch. The measured two-rank value is in [docs/15](../../../../docs/15-tp2-track.md) §5.10.

## What to check in the boot log

```
patch-prefixhit: applied to v1/core/kv_cache_utils.py (HAREM_PREFIX_HIT honoured)
patch-prefixhit: applied to v1/core/kv_cache_coordinator.py (HAREM_EAGLE_BLOCK_DROP honoured)
patch-kpooltail: applied to v1/worker/gpu/model_states/mamba_hybrid.py
patch-kpooltail: applied to v1/attention/backends/mla/indexer.py
HAREM-TP3 prefix-hit: 1 drafter group(s) flagged is_eagle_group; target groups left unflagged
```

The last line is the one that matters: **one** drafter group flagged, the target groups left alone.
Zero flagged groups means the knob is off and the coordinator will flag all of them.
