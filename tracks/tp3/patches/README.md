# tracks/tp3/patches — the full-scope patch tree (production configuration 9)

This is the second patch tree. [`patches/tp3/`](../../../patches/tp3/) serves the
routed-experts-only checkpoint; this one serves the **full-scope** checkpoint
(`turboderp/GLM-5.3-Flash-exl3` at 4.05 bpw) at TP=3 with expert parallelism, and it has been
production since 5 September 2026 evening. The whole story is
[docs/13](../../../docs/13-full-scope-checkpoint.md); this page is the directory.

**Why two trees rather than one.** Not because the code has to differ — the two constants that
change derive from `tp` and 128 and are provably no-ops at TP≤2, so the trees *could* be merged
today. Because the directory's **file list and content are the fast-load manifest identity**
([docs/08](../../../docs/08-fast-boot.md) §4): every `patch-*.py` in it and the full text of the prelude
are hashed into the sidecar, and adding one file to `patches/tp3/` — even a file that is never
called — refuses the next production boot on every node. That happened twice in one day before this
tree existed ([docs/09](../../../docs/09-measurement-protocol.md) §11.2). Keeping the experimental arm
in its own directory is what let production 8 stay up, untouched, while production 9 was built and
measured beside it.

The cost is written down rather than hidden: **the two trees have to be kept in step by hand.** A fix
that lands in `patches/tp3/` does not reach here on its own. Merging them is the open item in
[docs/11](../../../docs/11-open-issues.md) §2.24.

## What is in here

| File | Relation to `patches/tp3/` |
|---|---|
| `patch-fullscope-tp3.py` | **new** — the ten-anchor loader patch, A1–A10 |
| `check-padload-tp3.py` | **new** — the image gate, run in the prelude before any weight is read |
| `pad-tp3full.py` | **new** — the sidecar generator: padded `config.json` **plus** a rewritten `quantization_config.json` carrying the packed mapping |
| `mk-env-tp3full.sh` | **new** — derives `.env.tp3-full` from *this node's own* `.env.tp3` with `sed` |
| `tp3full-prelude.sh` | **changed** — `tp3/tp3-prelude.sh` plus the two `HAREM_EXL3_FULLSCOPE` steps, the `HAREM_SM12_ITEMS` block, the indexer-workspace bound, the two 8 September backports and the FlashKDA line. The **vision** block is the one arm still kept outside it, as a paste-in hook beside its own gates ([`vision/prelude-vision-hook.sh`](vision/prelude-vision-hook.sh)); our production prelude carries it directly after the FlashKDA line |
| `patch-pdl-gate.py` | **new in production 11** — sm_12x item 1: Programmatic Dependent Launch off on sm_12x by default, `HAREM_PDL_SM12=1` restores upstream |
| `patch-kpool-init.py` | **new in production 11** — sm_12x items 2 + 3: the K-pool top-k buffer starts at −1 and its reader is bounded from above |
| `flashkda/patch-flashkda-tp3.py` | **new in production 13 + FlashKDA (10 September 2026)** — five anchors in `glm5next/nvidia/kda.py` that route KDA chunked prefill through the fused `vllm._flashkda_C` kernel, behind `HAREM_KDA_FLASHKDA`. Grouped in a subdirectory here for a reader; **on a node it sits directly in `$TP3_DIR`** with everything else, because the sidecar identity hashes `glob($TP3_DIR/patch-*.py)`. Its prelude line is the one in this tree with a different `--root` and a mandatory `--in-place`, and [`flashkda/README.md`](flashkda/README.md) §3 says why |
| `patch-vllm-tp3.py` | **changed** — two constants: vocab `padding_size` 192 → **384**, shared expert 2112 → **2304** |
| `preflight-tp3.py` | **changed** — the same two, as `lcm(128, tp)`, plus three new 128-alignment gates |
| everything else | **byte-identical copies** of the file of the same name in `patches/tp3/` |

Verify that last row rather than trusting it — two copies of a file are a coin flip unless something
checks ([docs/08](../../../docs/08-fast-boot.md) §12):

```
for f in tracks/tp3/patches/*.py tracks/tp3/patches/overlay/cuda_exl3/*.py; do b=${f#tracks/tp3/patches/}; [ -e "patches/tp3/$b" ] && { cmp -s "$f" "patches/tp3/$b" && echo "same $b" || echo "DIFFERS $b"; }; done
```

Expected: `DIFFERS` for `patch-vllm-tp3.py` and `preflight-tp3.py` only. The two `sm12` files have
no counterpart in `patches/tp3/` — they are full-scope-track only — so the loop skips them.

**`patch-kpool-init.py` writes a file the container cannot write.** One of its two targets,
`model_executor/layers/sparse_attn_indexer_kpool.py`, is bind-mounted **read-only** from
`$OVERLAY_DIR` by the launcher. Pre-apply it to a host-side copy of the overlay directory and point
`OVERLAY_DIR` at the copy; the prelude then reports "already applied" for that file and writes only
the image's own `kpool_compress.py`. Verify the copy with `diff -r` against the overlay it came from:
exactly one file should differ, by exactly item 2's two hunks. This applies to **any** future patch
that lands on an overlaid file.

**Two files that are on the nodes and deliberately not here.** `start-tp3.sh` — the launcher lives
once, in [`scripts/`](../../../scripts/start-tp3.sh); the nodes carry a copy inside each tree and that is
the duplication [docs/08](../../../docs/08-fast-boot.md) §12 warns about, not a second launcher to
publish. And `tp3-prelude.sh` — inside the directory it is a **hard link** to `tp3full-prelude.sh`,
one inode, because the launcher mounts `$TP3_DIR/tp3-prelude.sh` at `/start.sh` and two names that
can drift apart is exactly what a hard link prevents. Recreate it with
`ln tracks/tp3/patches/tp3full-prelude.sh ~/exl3-zeus/tp3full/tp3-prelude.sh`, not `cp`.

**The files here are scrubbed copies** — internal paths in comments were replaced by references to
these docs — so their `sha256` will not match ours. That is only worth saying because the manifest
records those hashes: you will dump your own sidecar against your own copies, and comparing hashes
with us would be meaningless.

## `patch-fullscope-tp3.py` — the ten anchors

Same style as the rest: exact-text anchors that must match **exactly once**, `py_compile` before the
write, atomic replace, idempotent, a `--check` mode, and a hard exit if anything does not match. Two
target files, `vllm/models/glm5next/nvidia/model.py` and `.../kda.py`, plus (A9)
`vllm/model_executor/layers/linear.py`. Everything is gated on `HAREM_EXL3_FULLSCOPE=1`; with the
knob unset the patched image is upstream byte for byte.

| # | File | Layer | What it does |
|---|---|---|---|
| A1 | `model.py` | S1 | helpers plus a conditional `packed_modules_mapping` on `Glm5NextForCausalLM` |
| A2 | `model.py` | S1 | the same conditional class attribute on `Glm5NextForConditionalGeneration` |
| A3 | `model.py` | S2 | the MLA `quant_config=None` becomes conditional on the method being EXL3 |
| A4 | `model.py` | S3b | replaces the six `.in_proj_qkvbfg_a` entries in `stacked_params_mapping` with four, then asserts there are four |
| A5 | `model.py` | S3a | the `conv1d` three-way split in `load_weights`, plus the `ReplicatedLinear` v2 routing |
| A6 | `kda.py` | S2 | the unconditional `quant_config` strip becomes conditional |
| A7 | `kda.py` | S3b | splits `in_proj_qkvbfg_a` into an EXL3 `in_proj_qkv` and a BF16 `in_proj_bfg_a`; records the checkpoint's real shard width on the module; installs asserts 1 and 2 |
| A8 | `kda.py` | S3b | `forward` becomes two calls when split, the upstream single call otherwise |
| **A9** | `linear.py` | TP=3 only | `_load_fused_module_from_checkpoint` splits by the **checkpoint's** shard widths, not the module's padded `output_sizes`. No-op at TP≤2 |
| **A10** | `model.py` | TP=3 only | post-load audit (assert 5): every padded EXL3 site is whole 128-blocks and exactly zero |

A1–A8 are the TP=2 patch, [`tracks/tp2/patches/patch-fullscope-tp2.py`](../../tp2/patches/patch-fullscope-tp2.py),
unchanged. A9 and A10 are what TP=3 added.

**A10 caught its own bug on the first writing**, and the way it did is the reusable part: it treated a
column-parallel module's *input* as padded and rejected `in_proj_qkv` on ranks 1 and 2. Rank 0
passed — a single-rank test could not have seen it. It now reads row-parallel-ness out of the
module's own geometry (`exl3_k * tp == input_size`) instead of inferring it from a width mismatch.

## The boot gates this tree prints

In order, and all ten held on the first production 9 boot
([docs/09](../../../docs/09-measurement-protocol.md) §5.1):

```
[tp3-prelude] TP3FULL arm rank=2 tp=3 ep=1 fullscope=1
[tp3-prelude] SM12 items=pdl,kpool HAREM_PDL_SM12=unset
patch-kpool-init: item 2 already applied (.../model_executor/layers/sparse_attn_indexer_kpool.py)
patch-kpool-init: applied item 3 to .../models/glm5next/nvidia/ops/kpool_compress.py (pid < pool_len bound)
[padload] cuda-exl3 padded-load support: padded-output-gate (f3e3090)=yes  vocab-loader-prefix (754421f)=yes  row-parallel-suh-pad (f3e3090)=yes
[fullscope] anchors 1/1: A1 A2 A3 A4 A5 A6 A7 A8 A9 A10
EXL3 language_model.lm_head: vocab padded 154880 -> 155136; the 256 pad columns are 2 whole Hadamard blocks and are zeroed through svh.
HAREM-FULLSCOPE assert 5: 285 EXL3 pad site(s) audited, 285 padded on this rank, all whole 128-blocks and exactly zero
```

The `assert 5` line is a gate, not a log line: **if it is absent, the audit did not run.** 285 is the
number the model-free meta-device run predicted for rank 2 before the boot, arrived at independently.

## Rollback

- **One line:** delete `HAREM_EXL3_FULLSCOPE=1` from `EXTRA_ENV`. The patch reads the knob at run
  time, so the patched image takes the upstream path and serves the routed-experts-only checkpoint.
- **The sm_12x set:** delete `HAREM_SM12_ITEMS=pdl,kpool` from `EXTRA_ENV`. With it unset no script
  runs and the tree behaves like production 10 — but the tree still *hashes* differently, so this
  rollback does not restore production 10's sidecar; keep pointing `FASTLOAD_DIR` at production 11's.
- **Production 10 whole:** start with `ENV_FILE` pointing at the production 10 env file (ours is
  `.env.tp3.bak-prod10-083`). That reverts the memory rung, the tree, the overlay and the sidecar in
  one move, because all four are named in the env file, and production 10's sidecar was never
  deleted. No dump boot.
- **Whole arm:** start with `ENV_FILE` pointing at the production 8 env file (ours is
  `.env.tp3.bak-prod8-62f53e6`). That reverts the checkpoint, the image, the tree and the sidecar in
  one move, because all four are named in the env file. `patches/tp3/` and production 8's sidecar
  were never modified.
- The **image never changes**: every patch here runs inside the container at boot, against the
  writable layer, and leaves no trace when the container is removed.
