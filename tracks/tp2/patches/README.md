# tracks/tp2/patches — the two-node patch tree (TP=2 recipe, candidate D)

**Applies to: TP=2 only.** This is the tree the two-node production candidate of
[docs/15](../../../docs/15-tp2-track.md) §5 runs. It is [`tracks/tp3/patches/`](../../tp3/patches/) with every
file that exists only to serve a pad removed, plus the two-node full-scope loader patch.

**Updated 8 September 2026 — candidate D.** Candidate C's fourteen files plus four that come from
the three-node tree unchanged: `patch-vllm-tp3.py`, `patch-vision-tp3.py` and the two backports.
**Eighteen files against the three-node tree's twenty-three**, and the four new ones are *copies*
rather than forks — see [`vision/`](vision/README.md) and
[`prefix-hit-and-kpool-tail/`](prefix-hit-and-kpool-tail/README.md), which are pointers and install
commands rather than second copies of the scripts.

**Why a tree of its own and not a flag.** The directory's **file list and content are the fast-load
manifest identity** ([docs/08](../../../docs/08-fast-boot.md) §4): every `patch-*.py` in it and the full
text of the prelude are hashed into the sidecar, and adding one file to a tree — even a file that is
never called — refuses the next boot on every node. That has happened to us twice, and **once it was
a TP=2 patch dropped into the TP=3 tree** that did it. Finish this tree before the dump boot and add
nothing between dump and load.

The cost is the same one the three-node trees pay and it is written down rather than hidden: **the
trees have to be kept in step by hand.** Every shared file here is byte-identical with its
`tracks/tp3/patches/` copy today, and nothing enforces that tomorrow.

## What is in here

| File | Same as `tracks/tp3/patches/`? | What it does at two ranks |
|---|---|---|
| `tp2full-prelude.sh` | **new** | the in-container prelude: applies the patches below, then `exec vllm serve`. Install it under **two names** — see below |
| `patch-fullscope-tp2.py` | **new** (it is `tracks/tp2/patches/`'s) | S1 packed mapping, S2 stop hard-wiring MLA+KDA to bf16, S3 KDA refactorisation. Gated on `HAREM_EXL3_FULLSCOPE`. **The recommended candidate runs this** |
| `patch-kvdiag-tp3.py` | byte-identical | logging only: the per-group decomposition of the KV pool arithmetic. Every arm |
| `patch-swblock-tp3.py` | byte-identical | the draft KV page, 16 → 256 tokens. **Mandatory in practice** at two ranks ([docs/15](../../../docs/15-tp2-track.md) §4) |
| `patch-draftkv-tp3.py` | byte-identical | the drafter's own cache at fp8: **+15.1 % of pool** at two ranks |
| `patch-indexer-workspace-tp3.py` | byte-identical | bounds the sparse indexer's K-gather workspace, 4.92 GiB → 512 MB: **+32.14 % of pool** on an eager-boot A/B at two ranks, **+26.5 %** against the recipe it replaces once the sidecar is back. Gated on `HAREM_INDEXER_WS_MODE=bound`; unset is upstream byte for byte. The file needs no two-node variant — every term of the bound is per engine, not per rank ([docs/15](../../../docs/15-tp2-track.md) §5.9, [`tracks/tp3/patches/indexer-workspace/`](../../tp3/patches/indexer-workspace/) for the derivation and the four safety layers). **Half of it lands on a read-only overlay file** — pre-apply that half on the host, see the prelude |
| `patch-tilelang-failloud-tp3.py` | byte-identical | a silent `contextlib.suppress` becomes a named error. Gated |
| `patch-epfilter-tp3.py` | byte-identical | EP weight filter for `.trellis`. **Inert here** — it needs `--enable-ep-weight-filter`, which needs EP, and EP is off. Kept anyway, so that trying EP at two ranks needs no tree change and therefore no new dump boot |
| `patch-fastload-tp3.py` | byte-identical | installs the sidecar hook into vLLM's loader. Inert unless `HAREM_FASTLOAD_MODE` is set |
| `harem_fastload.py` | byte-identical | the dump/restore engine |
| `harem_fastload_id.py` | byte-identical | the sidecar identity, shared by the engine and the preflight so "stale" has one definition |
| `preflight-fastload.py` | byte-identical | refuses a stale sidecar in the prelude, in a second, before vLLM starts |
| `preflight-tp3.py` | byte-identical | the shape preflight; `tp`-parameterised. Run it `--tp 2 --ep 0` and it prints `ep=OFF (experts tensor-sliced)` |
| `flashinfer-warmup.py` | byte-identical | imports `flashinfer.comm` once, CPU-side, ~2 s |
| `patch-vllm-tp3.py` | byte-identical | **new in candidate D, and it is the surprise.** Its padding half is a no-op by arithmetic here (`lcm(128, 2) = 128`, and 154,880 and 2,048 are already multiples of it), which is why candidate C did not ship it. Its edits 4a/4b/4c are the `--language-model-only` wrapper — and `patch-vision-tp3.py`'s VS1 anchor is written against the text edit 4b leaves behind, so **vision at two ranks cannot work without this file**. [docs/19](../../../docs/19-vision-at-two-ranks.md) §2 |
| `patch-vision-tp3.py` | byte-identical | the tower: VS1/VS2/VS3 (the checkpoint is 6-bit EXL3) and VS4/VS6/VS7 (upstream's frame-sampler mismatch). Gated on `HAREM_VISION`. [`vision/`](vision/README.md) |
| `patch-prefixhit-tp3.py` | byte-identical | flag only the drafter's KV groups as EAGLE groups. Gated on `HAREM_PREFIX_HIT`. [`prefix-hit-and-kpool-tail/`](prefix-hit-and-kpool-tail/README.md) |
| `patch-kpooltail-tp3.py` | byte-identical | pass positions through the hybrid attention-metadata path so the K-pool tail's own slot mapping runs. Gated on `HAREM_KPOOL_TAIL_FIX`. [`prefix-hit-and-kpool-tail/`](prefix-hit-and-kpool-tail/README.md) |

Three model-free gates and one verifier ride along and are **not** hashed into the sidecar identity
(only `patch-*.py` and the prelude are): `check-vision-mapping.py`, `check-vision-names.py`,
`check-video-geometry.py` and [`verify-cpu.sh`](verify-cpu.sh), which runs the **whole** two-node
prelude order in a throwaway CPU container and must print six PASS/applied lines.

## Install the prelude under two names

```
install -m 0755 tracks/tp2/patches/tp2full-prelude.sh "$TREE/tp2-prelude.sh"
ln -f "$TREE/tp2-prelude.sh" "$TREE/tp3-prelude.sh"
```

Not a typo, and not a second copy. `scripts/start-tp2full.sh` mounts `tp2-prelude.sh` at `/start.sh`,
while `harem_fastload_id.file_identity()` hashes the prelude under the name **`tp3-prelude.sh`** —
so without the link the prelude's text silently falls out of the sidecar manifest and a changed
prelude would no longer invalidate a stale sidecar. A **hard** link is what keeps the two names on
one inode so they cannot drift apart. `start-tp2full.sh` refuses to launch if either name is missing.

The same launcher also sets `TP3_DIR` to the *same* directory as `TP2_DIR`, because
`preflight-fastload.py` and `harem_fastload.py` both read `TP3_DIR` to find the patch set they must
hash.

## What is deliberately **not** here

All of it exists to serve a pad, and at two ranks there is no pad: all five awkward shapes divide by
two and leave whole 128-column Hadamard blocks ([docs/15](../../../docs/15-tp2-track.md) §1.1).

**One row left this table on 8 September.** `patch-vllm-tp3.py` used to sit here, with the reason
"no two-node arm of ours has run it, and we do not ship text we have not measured". Both halves of
that sentence are still true of its *padding* edits and it is in the tree anyway, because the vision
patch anchors on a different edit of the same file. It has now been run at two ranks
`[measured-here]`.

| Not shipped | Why |
|---|---|
| `patch-exl3-ep.py` + `overlay/cuda_exl3/` | the `cuda-exl3` expert-parallel kernel fixes. EP is off at two ranks |
| `patch-dflash-tp3.py` | makes the DFlash2 head check pad-aware over the 32/8 → 36/9 drafter pad. 32/8 divides by two |
| `patch-fullscope-tp3.py` | its A1–A8 are `patch-fullscope-tp2.py`'s text exactly; A9 (split a fused checkpoint tensor by checkpoint widths) is a no-op at TP≤2 and A10 (post-load pad audit) has nothing to audit |
| `check-padload-tp3.py` | gates a `cuda-exl3` capability only the padded-load path needs |
| `pad-tp3.py`, `pad-tp3full.py`, `mk-env-tp3full.sh` | build padded sidecars and the env that points at them. Nothing to pad |
| `patch-zerokv-tp3.py` | an optional arm, never in a production environment file at either rank count |

## Order the prelude applies them

`patch-vllm` → `patch-kvdiag` → `patch-swblock` → `patch-epfilter` → `patch-fastload` →
(`patch-draftkv` if `HAREM_DRAFT_KV_DTYPE`) → (`patch-tilelang-failloud` if
`HAREM_TILELANG_FAILLOUD=1`) → `patch-indexer-workspace` → `patch-prefixhit` → `patch-kpooltail` →
(`patch-fullscope-tp2` if `HAREM_EXL3_FULLSCOPE=1`) → (`patch-vision` + its three gates if
`HAREM_VISION=1`) → `flashinfer-warmup` → `preflight-tp3` →
(`preflight-fastload` if `HAREM_FASTLOAD_MODE`) → `exec vllm serve`.

**This is the three-node order with the three-node-only steps taken out**, and the order is
load-bearing in two places: `patch-vision` must run after `patch-vllm` (VS1's anchor) and after
`patch-fullscope-tp2` (whose A2 shadows the mapping the environment then supplies back).

`TP2_STRICT=0` turns a failed patch into a warning. Do not use it to get past a broken anchor: a
half-patched stack is exactly the failure mode that serves fluent, wrong answers.
