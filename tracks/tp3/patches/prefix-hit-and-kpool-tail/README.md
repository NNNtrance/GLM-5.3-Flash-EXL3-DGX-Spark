# `patches/prefix-hit-and-kpool-tail` — the repeated prefix, and the one-row tail

Two backports into the pinned vLLM this stack serves (`487ecf187`, 25 August 2026), measured in the
same session against the same control. **Neither is in the recipe**, and why is the most useful
thing on this page: both do what they say, and a quality gate wobbled three times across four
patched arms while the control's identical battery came back clean.

| | Knob | What it does | Status |
|---|---|---|---|
| **K-pool tail slot mapping** | `HAREM_KPOOL_TAIL_FIX=1` | 99.67 % of the indexer tail's writes were addressing a block that is not the request's own; the fix takes that to **0.00 %** | **Measured, not shipped** — [§5](#5-why-neither-is-in-the-recipe) |
| **Hybrid prefix-cache hit** | `HAREM_PREFIX_HIT=1` | doubles the exact-repeat hit at 8K (41.56 % → 83.12 %, the ceiling) and cuts follow-up TTFT by **62 %** | **Measured, not shipped** — [§3](#3-why-the-prefix-cache-half-did-not-clear-its-bar), [§5](#5-why-neither-is-in-the-recipe) |

Both patch scripts are in the tree and are applied unconditionally by the prelude; only the
environment file decides which behaviour is armed. With both knobs unset the patched image is
upstream on both paths, byte for byte — which is what the control arm ran, and what the production
configuration runs today.

**This page is a negative result with two positive measurements inside it.** The bugs are real, the
fixes work, the instruments are here, and so is the evidence that stopped promotion — including the
part of that evidence which fell over twenty minutes after the decision. What is missing is boots,
and a flake baseline for the gates that judged them. Both are written up in
[HELP-WANTED](../../../../HELP-WANTED.md) §12.

Everything measured is in
[`results/gates/prefix-hit-and-kpool-tail.md`](../../../../results/gates/prefix-hit-and-kpool-tail.md).
Standing items: [docs/11](../../../../docs/11-open-issues.md) §2.32 and §2.33.

---

## What is here

| File | What it does |
|---|---|
| [`patch-prefixhit-tp3.py`](patch-prefixhit-tp3.py) | Two anchors. `vllm/v1/core/kv_cache_utils.py`: annotate the drafter's KV cache groups in the DFlash branch of `get_kv_cache_groups`. `vllm/v1/core/kv_cache_coordinator.py`: an env switch in front of the flag-all fallback. Default **off** — one environment read at grouping time and one at coordinator construction |
| [`patch-kpooltail-tp3.py`](patch-kpooltail-tp3.py) | Three anchors. `vllm/v1/worker/gpu/model_states/mamba_hybrid.py`: pass `positions=` into `build_attn_metadata`, as `default.py` already does. `vllm/v1/attention/backends/mla/indexer.py`: write the corrected tail mapping in place instead of returning a clone, and an opt-in out-of-bounds detector on the tail metadata builder. Default **off** — one environment read at import |
| [`kpool-tail-unit-test.py`](kpool-tail-unit-test.py) | Model-free, CPU-only: pins `block_table[req, 0] * kpool + pos % kpool` against the corrected mapping and shows the generic one collapsing three requests onto block 0. Runs inside the serving image in about ten seconds |
| [`annotate-unit-test.py`](annotate-unit-test.py) | Model-free: the drafter's group is flagged, the target's is not, and both fail-closed refusals fire |
| [`prefix-hit-probe.py`](prefix-hit-probe.py) | The measurement instrument: per-request prefix-cache hit ratio, TTFT and draft acceptance from the engine's own Prometheus counters, for exact repeats and a four-turn agent conversation |
| [`kpool-soak.py`](kpool-soak.py) | The long-generation soak: eight concurrent 4,096-token generations and two 8,192-token ones, with a coherence check on each. Position is what drives the tail bug, so what it stresses is generation length, not prompt length |
| [`offset-sweep.py`](offset-sweep.py) | Holds everything fixed and varies only `n mod 3328`, which is what decides whether the drafter's drop is free |
| [`needle-lite6.py`](needle-lite6.py) | Six needles at six depths of one ~54,700-token haystack, sequential or `concurrent`. Thinking stays on and `enable_thinking=false` is never sent; it scores `content` and reports `either` separately, as `correctness-probe.py` does |

## The knobs

| Variable | Values | Effect |
|---|---|---|
| `HAREM_PREFIX_HIT` | `1` | flag **only** the DFlash2 drafter's KV cache groups as EAGLE groups |
| | unset / `0` | **upstream behaviour, byte for byte**: no group is flagged and the coordinator flags all of them |
| `HAREM_EAGLE_BLOCK_DROP` | `0` | no group takes the last-block drop at all (#53388's `disable_eagle_block_drop`). **Diagnostic arm only** |
| | unset | upstream behaviour |
| `HAREM_KPOOL_TAIL_FIX` | `1` | pass `positions` through the hybrid path, and write the corrected tail mapping in place |
| | unset / `0` | **upstream behaviour, byte for byte** |
| `HAREM_KPOOL_TAIL_BOUNDS` | `1` | arm the tail out-of-bounds counter; it logs every 128 steps. Costs one clone and a handful of host-side reductions per step, so it is an evidence arm, not a production setting |
| | unset | inert |

---

## 1. The prefix-cache hit

### What the code does

`KVCacheCoordinator.__init__` collects the groups that carry `is_eagle_group` and then, if
speculative decoding is on and the set is empty, flags every group:

```python
self.eagle_group_ids = {i for i, g in enumerate(...) if g.is_eagle_group}
if use_eagle and not self.eagle_group_ids:
    self.eagle_group_ids = set(range(len(kv_cache_config.kv_cache_groups)))
```

The only thing in this image that ever sets `is_eagle_group` is
`_annotate_eagle_groups_deepseek_v4`, which requires `model_version == "deepseek_v4"` and is called
from one branch — `group_and_unify_kv_cache_specs`. GLM-5.3-Flash with a DFlash2 drafter takes a
different branch: `_harem_partition_dflash_draft_specs` groups the target and the draft separately
and returns `[*target_groups, *draft_groups]` ([docs/04](../../../../docs/04-dflash2-port.md) §3).
No annotation site is reached, so **every** group is flagged.

The layout it is flagging, read from the boot log:

| Group | Layers | Block | Role |
|---|---|---|---|
| `MLAAttentionSpec` | 22 | 3,328 tokens | the target's attention |
| `KpoolTailSpec` | 11 | 4 tokens | the indexer tail — opted out of prefix caching entirely |
| `MambaSpec` × 4 | 9 · 9 · 8 · 8 | — | the KDA layers |
| `SlidingWindowSpec` | 5 | 256 tokens | **the DFlash2 drafter** |

An EAGLE group matches one block past the aligned boundary and drops it; `cache_blocks` gives such a
group one block of lookahead so the drop nets out to zero. The drafter's group has that lookahead by
construction (its block is 256 and the alignment is 3,328, so its cached prefix always runs past the
boundary). The target's group does not: it caches whole 3,328-token blocks, so the block it is asked
to drop is the last one it has. The drop is a real loss, once per request.

### What the patch does

In the DFlash branch, flag the drafter's groups and only those:

```python
_harem_annotate_draft_eagle_groups(target_groups, draft_groups)
return [*target_groups, *draft_groups]
```

The helper refuses rather than guesses: it raises if the drafter's groups are not all
sliding-window (which is #54041's precondition — a mixed drafter keeps the conservative fallback),
and it raises if a target group already carries the flag, because that would mean the grouping path
moved under the patch.

We do not carry #54041's plumbing of a `non_causal_multi_token_decode` marker through
`Attention.__init__` and `get_kv_cache_spec`. Upstream needs a marker because its grouping function
only ever sees one flat list of specs; ours is handed the drafter's groups as a separate list, so
the marker would be a longer road to the same `is_eagle_group`. If this stack ever rebases onto a
vLLM that carries #52047, this patch's anchor disappears and the marker is the right way to do it —
see [docs/11](../../../../docs/11-open-issues.md) §2.32.

## 2. The K-pool tail

### What the code does

`KpoolTailSpec` declares a one-block circular scratch cache: `max_num_blocks_per_req() == 1`,
`block_size == index_kpool == 4`, addressed as `block_table[req, 0] * kpool + pos % kpool`, which is
what `kpool_compress.py` documents and what both write kernels assume. Only column 0 of the group's
block-table row is ever written. The generic per-group slot kernel computes

```python
block_indices = positions // block_size          # 0, 1, 2, ... for pos = 0, 4, 8, ...
block_numbers = tl.load(block_table_ptr + req * stride + block_indices)
```

so from `pos >= 4` it indexes a column no allocator ever wrote, and the slot lands on a block that is
not the request's own. Neither `_kpool_tail_seed_kernel` nor
`_kpool_decode_update_batched_kernel` bounds-checks the block id.

**One detail of the published diagnosis does not hold on this build, and it does not change the
conclusion.** vcruz305 describes the row as *one entry wide*, so that `pos >= block_size` reads past
it. On our runner the row is **250,016 entries** wide — it is sized `cdiv(max_model_len, block_size)`
rather than from `KpoolTailSpec.max_num_blocks_per_req()`, which returns 1 — so there is no
out-of-row read at all: our detector counts **zero** row overruns while the mapping is wrong for
99.7 % of tail tokens `[measured-here]`. The harm is the wrong block, not the overrun. This is the
second independent reason the clamp is the wrong layer: on this build there is nothing to clamp.

The correct mapping is already in the image, in `compute_kpool_tail_slot_mapping`, called from
`KpoolTailMetadataBuilder.build` — behind a guard that degrades silently:

```python
positions = common_attn_metadata.positions
if positions is not None:          # never true on a hybrid model
    slot_mapping = compute_kpool_tail_slot_mapping(...)
```

`v1/worker/gpu/model_states/mamba_hybrid.py` calls `build_attn_metadata(...)` without `positions=`;
`v1/worker/gpu/model_states/default.py`, three files away, passes `positions=input_batch.positions`.
GLM-5.3-Flash is a hybrid model because of its KDA layers, so it takes the first path, and the tail
builder is never handed positions.

### What the patch does

Both halves are vcruz305's, applied to this tree behind an env gate:

- **K1** — `mamba_hybrid.py` passes `positions=input_batch.positions`. On this platform
  `common_attn_metadata.positions` has exactly one other consumer (`cpu_attn.py`), so nothing else
  on the CUDA path changes.
- **K2** — `compute_kpool_tail_slot_mapping` writes the caller's buffer in place rather than
  returning `slot_mapping.clone()`. That buffer *is* the tail group's persistent slot mapping, so in
  place is the correct semantics; it also matters because CUDA-graph capture pins a clone's transient
  address and replays read it back after it has been freed (`Xid 13`). This stack serves eager
  ([docs/11](../../../../docs/11-open-issues.md) §2.29), so K2 is latent here and is carried because
  K1 without it is a trap for anyone who turns graphs on.

**What was tried and did not work, before we got here.** vcruz305 records two dead ends and we did
not have to repeat either: clamping the block index inside the generic slot kernel
(`tl.minimum(block_indices, stride - 1)`) changed nothing — 48 overrunning calls before and 48 after
— because once positions are present the tail mapping does not come from that kernel at all, and the
clamp only masks the garbage it produces; and synthesising positions at the tail builder from
`seq_lens` and `query_start_loc` made every write in bounds during warm-up and then died with
`Xid 13` a second after graph capture `[reported]`. The task that produced this page was briefed to
apply the clamp; it is on the record here that the clamp is the wrong layer.

### Detecting it on your own build

`HAREM_KPOOL_TAIL_BOUNDS=1` arms a counter on the tail metadata builder, **after** the correction, and
hands it both mappings — the generic one and the one the engine actually uses. That distinction is
the whole design: the fix does not repair the generic kernel, it stops the tail from using it, so a
detector that only looked at the generic mapping would report the same number in both arms and prove
nothing. It counts

- `used_wrong_block` — tokens of the mapping in force this step whose slot divides down to a block
  other than the request's own. **This is the number that must be zero.**
- `generic_wrong_block` — the same count for what the generic kernel produced, i.e. what the tail
  was being handed.
- `row_overruns` — tokens whose `pos // block_size` is at or past the row width. Positions are
  reconstructed from `seq_lens` and `query_start_loc`, so the count exists in both arms, including
  the one where `positions` is None.

The reductions are host-side, once per step. That is honest **only** because this stack serves
eager: under CUDA graphs a Python counter observes capture and never a replay, which is why
vcruz305's own detector accumulates in device tensors updated by the captured kernels. Do not arm
this with graphs on and believe a zero.

---

## 3. Why the prefix-cache half did not clear its bar

Separately from §5, which applies to both halves, this one also missed the bar it was given.
Neither point is "it did not work" — it works, and the numbers are in
[`results/gates/prefix-hit-and-kpool-tail.md`](../../../../results/gates/prefix-hit-and-kpool-tail.md).

**The bar was a raw hit ratio, and the ceiling makes that bar unreachable.** The bar was set in advance as *an exact
repeat must hit ≥ 95 %*. With the patch, an 8,008-token repeat hits **83.12 %** — which is
**100.0 % of the ceiling**, because `floor((8008-1)/3328) × 3328 / 8008` is 83.12 % and no
configuration can beat it — and a 59,910-token repeat hits **94.43 %**, unchanged. The raw bar is
not met at either size. The ceiling reading is the honest one and by that reading the patch is
perfect at 8K, but a bar is a bar, and the 60K case genuinely did not move.

**Why 60K did not move**, and this is a real property rather than a measurement problem: a request
of `n` tokens sits `n mod 3328` tokens past its last aligned boundary. The drafter's group takes the
EAGLE drop, gives back one of its own 256-token blocks, and then has to re-align to the 3,328-token
granularity — so it needs at least one whole 256-block *past* the boundary to give back. 59,910 mod
3,328 is **6 tokens**. There is no block to give, and the re-alignment pop costs a full 3,328-token
block instead. The patch moves the target group out of the way and the drafter's own drop remains.
About 7.7 % of prompt lengths land in that window.

**Reason two: one unexplained warm failure, in the arm where it was armed.** On the first boot of
the both-knobs arm, needle-lite came back **5/6** twice in a row — the earliest of six needles, in a
54,694-token haystack, answered with a plausible but invented code, twice with a *different* invented
code. The same gate was 6/6 cold on the same boot, and 6/6 on the control, on the prefix-hit-only arm
and on the K-pool-only arm, three runs each. A second boot of the both-knobs arm then returned
**6/6 six times**, including under concurrency and including after an identical 48,910-token soak —
so it does not reproduce. Nine runs of that configuration: seven pass, two fail, both failures inside
one boot.

That is not a verdict against the patch. It is also not a pass. This repository's own protocol says
a single boot does not settle anything ([docs/09](../../../../docs/09-measurement-protocol.md) §1),
and it applies to failures as well as to gains. **A measurement design error of ours is part of why
it is unsettled**: the failing arm was the only one that ran needle-lite *after* a soak, so the
comparison that produced the alarm was not like for like. The reproduction attempt corrected that and
came back clean.

**What would settle it**, for anyone with the hardware: three boots of the both-knobs arm, each
running the needle gate cold, warm, after a soak, and concurrently, on a pool that has been churned
by a full sweep first. If nine of nine pass, the patch is clear and the bar should be restated in
ceiling terms. See [HELP-WANTED](../../../../HELP-WANTED.md).

## 4. What the K-pool half cost, on its own arm

Nothing measurable in speed or memory, and it was looked for. On a boot with only
`HAREM_KPOOL_TAIL_FIX=1`: C1 **69.34** against the control's 66.97, C2 99.53 against 97.85, C4 143.26
against 141.57, C6 170.68 against 175.58, C8 191.59 against 192.12 — every level inside its band;
draft acceptance 62.62 % against 61.80 %; fresh prefill 1,757 against 1,753 tok/s; KV pool 6,914,600,
inside the spread; cold gates and vision full marks; and the exact-repeat hit ratio **identical to the
control at 41.6 %**, which is the check that the K-pool half touches nothing the scheduler does.
Its soak generated 49,152 tokens with the engine alive and every generation coherent. The one thing
it did not survive is the battery after that soak — §5. `[measured-here]` The KV pool, the sweep, the gates and the boot time are in
[`results/gates/prefix-hit-and-kpool-tail.md`](../../../../results/gates/prefix-hit-and-kpool-tail.md)
§3. The one real cost is a **fresh fast-load sidecar** — the patch scripts join the identity hash, so
promoting them re-dumps ~53 GB per node on a 495 s boot
([docs/08](../../../../docs/08-fast-boot.md)). We paid it once for both patches together, which is
why the prefix-hit script rides along in the tree with its knob unset: arming it later is an
environment-file edit, not another dump.

---

## 5. Why neither is in the recipe

The thing that stopped promotion is not in either patch's own numbers. It is a pattern across the
arms, and it took the whole night to see because the measurement that would have shown it early was
the one we did not run until last.

**Every quality battery run *after* a soak, on a patched arm, lost exactly one item. The control's
did not.**

| Arm | knobs | battery after a ~48,800-token soak |
|---|---|---|
| control | none | **clean** — probe 10/10, code 12/12, tool-call 8/8, needle 6/6 |
| both knobs, boot 1 | `HAREM_PREFIX_HIT=1 HAREM_KPOOL_TAIL_FIX=1` | needle **5/6**, twice in a row |
| both knobs, boot 2 | the same | needle 6/6, twice |
| K-pool only | `HAREM_KPOOL_TAIL_FIX=1` | code exam **11/12** (`matrix`, an `AssertionError` on the generated code) |

`[measured-here]`. Three wobbles across four patched batteries; zero across one control battery. That
is the table the rollback decision was made on, and **half of it fell over twenty minutes later.**

**The `matrix` item flakes on the untouched production configuration.** After rolling back, the very
first code exam on the restored production — no patch tree, no knobs, the configuration that has
served for a fortnight — came back **11/12, `matrix`, the same `AssertionError`**. Run four more
times on that same engine: **12/12, 12/12, 12/12, 12/12**. So the K-pool arm's post-soak 11/12 is a
known-flaky item, not evidence against anything, and it should never have been counted
`[measured-here]`.

What is left after that correction is **one boot** of the both-knobs arm returning needle 5/6 twice,
against six later runs of that same configuration returning 6/6 — including under concurrency and
including after an identical 48,910-token soak. Nothing else in the session, on any arm, is outside
its band.

**Four honest statements, in order of how much we would like them to be true.**

1. The rollback was decided on a table that was partly wrong, and we found that out by measuring the
   thing we should have measured first: the same gate, several times, on the configuration that was
   already in production.
2. What remains is one unreproduced boot. That is not a verdict against either patch, and this
   repository's own protocol says one boot settles nothing
   ([docs/09](../../../../docs/09-measurement-protocol.md) §1) — which applies to failures too.
3. It is also not a pass, and the burden of proof is on the change. Rolling back at midnight and
   reporting is cheap; discovering a fluent wrong answer in the morning is not.
4. Part of why the picture is thin is a design error of ours: for most of the session only the
   patched arms ran a battery after a soak, so the alarm came from a comparison that was not like for
   like. The control's post-soak battery — clean — was the last measurement of the night.

**Nothing was promoted. `.env.tp3` was never edited** — it was verified byte-identical to its dated
backup on all three nodes at the end — and the production patch tree and its fast-load sidecar were
never touched, so the rollback was "start the unit", which came back at **255 s** with the KV pool at
**7,066,115** and every gate full.

**What would settle it.** Two things, in this order.

First, **a flake baseline for the gates themselves**, which this session did not have and which is
the cheapest measurement on the list: run the correctness probe, the code exam, the tool-call gate
and needle-lite ten times each on the production configuration, cold and after a soak, and write down
the per-item pass rate. Tonight produced two data points by accident — `matrix` failed 2 of 12 code
exams — and they were enough to overturn half a decision. A gate whose flake rate is unknown cannot
adjudicate a patch.

Second, **three boots per arm** — control, K-pool only, prefix-hit only, both — each running the full
battery cold, then a ~49,000-token soak, then the full battery again. Twelve batteries, about four
hours. Judged against the baseline from the first step. If the arms match the baseline, both patches
are clear, and the K-pool half should go in on its own merits because it fixes a measured correctness
bug. The tree and its fast-load sidecar stay on disk, so arming an arm is one environment line and no
dump boot.

---

## Credits

- **Victor Cruz ([vcruz305](https://github.com/vcruz305))** — the K-pool tail diagnosis, the
  reproducer, the two dead ends, the detector design and the first fix
  (`GLM-5.3-Flash-EXL3-K2-DGX-Spark-recipe`, `docs/KPOOL_TAIL_BUG.md` and
  `scripts/patch_kpool_tail_positions.py`, 30 August 2026). Both edits in
  `patch-kpooltail-tp3.py` are his.
- **[okorzh-amd](https://github.com/okorzh-amd)** — vLLM
  [#52047](https://github.com/vllm-project/vllm/pull/52047), the generic
  `_annotate_eagle_groups`.
- **[positive666](https://github.com/positive666)** — vLLM
  [#54041](https://github.com/vllm-project/vllm/pull/54041), marking only an all-sliding-window
  drafter group.
- **[ZeldaHuang](https://github.com/ZeldaHuang)** — vLLM
  [#53388](https://github.com/vllm-project/vllm/pull/53388), `disable_eagle_block_drop`.
- **[ZJY0516](https://github.com/ZJY0516)**, crediting **JaredforReal** — vLLM
  [#53906](https://github.com/vllm-project/vllm/pull/53906), the K-pool machinery and the test that
  pins its intended addressing.
- **[Suppressor72](https://github.com/Suppressor72)** — vLLM issue
  [#53670](https://github.com/vllm-project/vllm/issues/53670), the throughput report behind the
  three prefix-cache pull requests — and **UserHIJ**, who reported the same class of failure.

Our own two patch scripts were written for this recipe; use freely (Apache-2.0), a credit is
appreciated. Full entries in [CREDITS.md](../../../../CREDITS.md).

---
