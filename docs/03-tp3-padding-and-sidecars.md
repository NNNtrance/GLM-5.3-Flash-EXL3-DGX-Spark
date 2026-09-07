# 03 — TP=3 padding and sidecars

**Applies to: TP=3 only.** All five shapes divide by two and leave whole 128-blocks, so there is
nothing to pad at two ranks — [15](15-tp2-track.md) §1.1.

GLM-5.3-Flash has 64 attention heads, 64 KDA heads, a 154,880-token vocabulary, a 2,048-wide shared
expert and 2,048-wide routed experts. None of those five numbers divides by three. This page is how
the model runs on three nodes anyway: what gets padded, what must never be padded, where the padding
lives, and the checks that refuse the versions of this that produce fluent nonsense.

Everything here is applied at container start by the prelude, which runs the patch scripts and then
execs the server. **A patch whose anchor no longer matches stops the rank.** That is deliberate: a
half-patched stack is exactly the failure mode that serves confident, wrong answers.

**Two patch trees.** Production 9 serves the full-scope checkpoint from
[`tracks/tp3/patches/`](../tracks/tp3/patches/) with `tracks/tp3/patches/tp3full-prelude.sh`;
[`patches/tp3/`](../patches/tp3/) with `scripts/tp3-prelude.sh` is the routed-experts-only tree that
production 1–8 used and is the rollback. They differ in exactly two constants (§1.1) plus one extra
patch. Where this page names a file, it names the production one; the same file exists in `tp3/` with
the older constants. Why the two are not one file is
[`tracks/tp3/patches/README.md`](../tracks/tp3/patches/README.md) — it is about the fast-load manifest
identity, not about the code.

> **At TP=2, none of this page applies.** All five shapes divide by two *and* leave every rank a whole
> number of 128-column Hadamard blocks, so there is no sidecar `config.json`, no padded-load path, no
> `svh = 0` pad audit and no `check-padload-tp3.py` image gate. `patch-vllm-tp3.py` computes
> `lcm(128, tp)` at run time and is a provable no-op at two ranks, so leave it in place rather than
> editing it out. [15 — Running this recipe at TP=2](15-tp2-track.md) §1 and §2.3.

---

## 1. The five shapes, and what happens to each

| Shape | Value | At TP=2 | **At TP=3** | Where the change lives |
|---|---|---|---|---|
| `num_attention_heads` | 64 | 32 per rank | **padded to 66** → 22 per rank | sidecar `config.json` |
| `num_key_value_heads` | 64 | 32 | **66** → 22 | sidecar `config.json` |
| KDA `linear_attn_config.num_heads` (and its flat mirror `linear_num_heads`) | 64 | 32 | **66** → 22 | sidecar `config.json`, **both** places |
| `vocab_size` | 154,880 | pads to 154,880, /2 = 77,440 | `padding_size` 64 → **384**, giving 155,136, /3 = **51,712 = 404 × 128** | `tracks/tp3/patches/patch-vllm-tp3.py` |
| shared expert (`moe_intermediate_size × n_shared_experts`) | 2,048 | 1,024 per rank | **padded to 2,304** → **768 = 6 × 128** per rank | `tracks/tp3/patches/patch-vllm-tp3.py` |
| routed experts `moe_intermediate_size` | 2,048 | tensor-sliced to 1,024 | **not sliced at all** — 96 whole experts of 288 per rank | `--enable-expert-parallel` |

Two things that need no work: `intermediate_size` 12,288 (the three dense layers) divides by 3
cleanly, and `hidden_size` 4,096, `q_lora_rank` 1,536 and `kv_lora_rank` 512 are never split by rank
at all. The preflight checks each of those explicitly rather than trusting this paragraph.

### 1.1 Why 384 and 2,304 rather than 192 and 2,112 — the 128 rule

Those two rows carried **192** and **2,112** until production 9, and both were `lcm(64, 3)`. They
were correct for a checkpoint whose vocabulary and shared expert are BF16, and **silently wrong** for
one where they are EXL3. The production checkpoint quantizes both ([01](01-model-and-license.md) §1),
so the unit had to move from 64 to 128.

The reason is the Hadamard block. An EXL3 tensor cannot be zero-extended — a trellis is not a dense
tensor — but it **can** be loaded narrow into a parameter vLLM has padded, because `svh` scales
elementwise *after* the output Hadamard: zeroing `svh` on a pad column makes that column exactly zero
whatever the trellis behind it holds. The condition is that the pad must occupy **whole 128-column
blocks**, because the Hadamard mixes across each block before `svh` is applied. A pad that shares a
block with real output does not merely fail to be zero; it **corrupts the real columns beside it**.

So a padded width has to satisfy two conditions at once — divisible by `tp`, and a whole number of
128-blocks *per rank*:

| what | old (`lcm(64, 3)`) | per rank | verdict | new (`lcm(128, 3) = 384`) | per rank | verdict |
|---|---|---|---|---|---|---|
| vocab | 154,944 | 51,648 = **403.5** × 128 | half a block — corrupts | **155,136** | 51,712 = 404 × 128 | whole blocks on every rank |
| shared expert | 2,112 | 704 = **5.5** × 128 | half a block | **2,304** | 768 = 6 × 128 | whole blocks |

**2,176 is the wrong number and it is worth naming, because it looks right.** It is 17 × 128 — but
2176/3 is not an integer. The padded width must be a multiple of `lcm(128, 3) = 384`, so 2,304.

Today's 2,112 failed in **two different ways at once**, which is why one line fixes both: `down_proj`
at k=704 hits the plugin's explicit refusal (loud), while `gate_up_proj` gets a half-block output pad
where there is only a warning (**silent**). Both constants are `lcm(128, tp)` in
`tracks/tp3/patches/`, and at TP≤2 they are provably no-ops: `lcm(128, 2) = 128`, 154,880 is already a
multiple of 128, and 2048/2 = 1024 = 8 × 128.

**Every head pad was already legal, and that was checked rather than assumed** `[measured-here]`. MLA
`qk`/`v` head_dim is 256 and KDA head_dim is 128, so 64 → 66 heads gives whole 128-column blocks
everywhere: 512 columns on `q_b_proj` and on `o_proj`'s input, 256 per KDA shard. The `svh = 0`
mechanism was in fact **already running** on this stack for column-parallel modules before anyone
designed it — the TP=3 padding patch fills missing rows with zeros and `svh` goes through the same
path — so the invariant held **by accident and nothing checked it.** That is exactly how the 2,112
arithmetic would have produced silently wrong output. It is now audited at load time: see §5.1.

### Why 66 and not 96

Padding a head count to the next multiple of the TP size is safe **only while the last rank still
owns at least one real head**. 64 → 66 gives every rank 22 heads, of which rank 2 holds 20 real and
2 fabricated. 64 → 96 would give rank 2 heads 64–95, every one of them fabricated: the model loads,
serves, and produces confident garbage. `patches/tp3/pad-tp3.py` refuses that case by name rather
than letting it be discovered in an evaluation `[measured-here]`.

### Why the shared expert is padded and the routed experts are not

On the fallback checkpoint the shared expert is native BF16, so zero-extending it is a real no-op:
2,048 → 2,112 adds 64 zero columns and the maths is unchanged. On the production checkpoint it is
6-bit EXL3, so the pad is a **narrow load into a padded parameter with `svh = 0`** instead, and the
width has to be 2,304 (§1.1). Different mechanism, same launcher constant.

The routed experts are EXL3 trellis in **both** checkpoints, and they are not padded in either. A
trellis column is meaningful only on a 128-element Hadamard boundary, and **a zero-extended trellis
is not a zero-extended weight** — it decodes to noise. Nor is 2,048/3 an integer, so there is no pad
that would help. The routed experts are distributed whole under expert parallelism, never padded and
never sliced.

This distinction is the single most important idea on this page, and production 9 sharpened rather
than removed it: **zero-extension is the right tool for the unquantized parts of this model and the
wrong tool for the quantized parts. What works for a quantized part is a narrow load into a padded
parameter whose scale is zero on the pad — and only on whole 128-column blocks.**

### Do not use `disable_tp` on the shared expert

It looks like the easy alternative and it is poison under expert parallelism. The MoE runner
all-reduces the combined MoE output whenever `ep_size > 1`, and the model builds the shared MLP with
`reduce_results=False` precisely because the runner owns that reduction. A replicated shared expert
therefore gets summed once per rank — a 3× shared contribution. The model stays fluent and the
answers go wrong. Another three-node recipe hit exactly this and wrote it up; the mechanism is those
two lines `[reported]`.

---

## 2. Why the padding goes in a sidecar and not in `--hf-overrides`

`--hf-overrides` reaches the target model's top-level attributes. It does **not** reach
`text_config.linear_attn_config["num_heads"]`, a nested dict inside a nested config whose value wins
over the flat keyword; and `SpeculativeConfig` builds the drafter from its own config file, which
`--hf-overrides` never touches at all. Three override paths that all have to agree is worse than one
file that is right in every process.

So `tracks/tp3/patches/pad-tp3full.py` writes a **sidecar directory**: relative symlinks to every file
of the downloaded checkpoint, plus one rewritten `config.json` — and, for the full-scope checkpoint,
one rewritten `quantization_config.json`.

**That second file is not optional and it is not a copy.** `cuda-exl3` inverts vLLM's linear merges
through `packed_modules_mapping`, which vLLM copies off the *model class* — and `glm5next` declares
none, so `gate_up_proj.trellis` has nowhere to land ([13](13-full-scope-checkpoint.md) §2.1). Since
`cuda-exl3` `5903248` a **checkpoint** may declare its own fusions, and they merge *under* whatever
the model class declares, so this route and the patch's own mapping can both be live without
conflicting — belt and braces. The rewritten file is a full 48 MB copy rather than a small one
holding only the mapping, because `cuda-exl3` needs `tensor_storage` out of the same dict. From
`fba9f27` onwards the same JSON can travel in `CUDA_EXL3_PACKED_MAPPING` instead; write it with **no
spaces**, because the launcher word-splits `EXTRA_ENV`.

`patches/tp3/pad-tp3.py` writes the config but not the mapping, which is correct for the fallback
checkpoint and fails to load the production one.

- The downloaded checkpoint is never modified. No `.orig` backups, no in-place edit, and the
  repository's own `SHA256SUMS` stays verifiable.
- Reverting is `rm -rf` of the sidecar.
- Two mirrors of the KDA head count must both say 66, because the mamba state shape is derived from
  the flat `linear_num_heads` while the layer itself prefers the nested dict. If they disagree, the
  KDA state cache is sized for a different model from the one that runs. `pad-tp3.py` writes both.

Build both sidecars. The model one, on the production checkpoint:

```
tracks/tp3/patches/pad-tp3full.py /var/tmp/glm-5.3-flash-turboderp-4.05bpw /var/tmp/glm-5.3-flash-turboderp-4.05bpw-tp3 --tp 3
```

The drafter one, unchanged between the two arms:

```
patches/tp3/pad-tp3.py /var/tmp/dflash2-draft /var/tmp/dflash2-draft-tp3 --tp 3 --draft
```

On the fallback checkpoint the model sidecar is `patches/tp3/pad-tp3.py` against
`/var/tmp/glm-5.3-flash-tr3-4bpw`, with the same arguments.

The drafter's own shapes are 32 query heads and 8 KV heads, verified from the safetensors header
rather than from its config; at TP=3 those become a padded **36/9**. The checks that make that safe
are in [04-dflash2-port.md](04-dflash2-port.md) §5.

### The mount trap this creates

The sidecars are trees of **relative** symlinks
(`../glm-5.3-flash-tr3-4bpw/model-00001-of-00120.safetensors`). A sidecar therefore only resolves
while it keeps its position relative to its link target. Mount it anywhere else — an obvious
`/models/...` is the natural mistake — and `..` resolves to the sidecar itself, every weight link
dangles, and the failure surfaces as a confusing "no safetensors found", not as a mount error
`[measured-here]`.

The fix is **identity mounting**: each sidecar is mounted at its own host path, inside the container,
which is the one arrangement that is correct by construction. `check_relative_sidecar()` in
`scripts/start-tp3.sh` refuses any other arrangement and names the reason. `pad-tp3.py --hardlink` is
the alternative if you would rather mount one directory; it needs both to be on the same filesystem.

This cost us a boot before the first TP=3 attempt, found by reading rather than by running
`[measured-here]`.

---

## 3. The four hard asserts, and the patches that clear them

At TP=3, four places in the engine assert before anything useful happens:

| Where | Assert | At TP=3 | Cleared by |
|---|---|---|---|
| the model's top-level builder | `num_attention_heads % world_size == 0` | 64 % 3 = 1 | sidecar config (66) |
| the MLA attention module | `num_heads % tp_size == 0` | same | sidecar config (66) |
| the KDA module | `linear_attn_config.num_heads % tp_size == 0` | same | sidecar config (66, both mirrors) |
| `VocabParallelEmbedding` | `divide(num_embeddings_padded, tp_size)` | 154,880 % 3 = 2 | `patch-vllm-tp3.py`: `padding_size` 64 → 384 (§1.1) |

`patch-vllm-tp3.py` also carries the shared-expert pad and the vision-tower fix below. Each edit is a
single exact-match anchor; the script fails closed if an anchor matches zero times or more than once.

### The vision tower ignores `--language-model-only` — `[history]` since 7 September 2026

**This section is no longer the recipe.** Since 7 September the production configuration **builds
and serves the tower**: 4 images and 2 videos per request, at TP=3, on the full-scope checkpoint,
with the text side unchanged and no measurable KV-pool cost. The whole account — why the tower would
not load even with the assert cleared, the upstream video sampler bug that killed the engine core on
the first video request, the brake, the ten gates and the cost table — is
[18](18-vision-at-three-ranks.md). The patch tree is
[`tracks/tp3/patches/vision/`](../tracks/tp3/patches/vision/README.md).

What follows is kept because the `--language-model-only` path is still the **text-only fallback**,
and because two of its claims were wrong and the correction belongs beside them.

The model file reads `multimodal_config` — it even reads `mm_encoder_tp_mode` — and then builds the
vision transformer **unconditionally**. `--language-model-only` only makes the multimodal limit
return 0, so no image can be *submitted*; it never stops the tower being *built*. At TP=2,
`divide(16, 2)` succeeds and the engine starts anyway; at TP=3, `divide(16, 3)` asserts and it never
does `[measured-here]`.

The fix is three anchors: a builder that returns `None` when `language_model_only` is set and logs
why; the construction site routed through it; and a `load_weights` override that skips the `visual.`
prefix so the checkpoint's tower tensors are dropped rather than reported as unexpected. Confirmed in
the boot log:

```
HAREM-TP3: --language-model-only is set, so the GLM-5.3 vision tower is not built and its checkpoint tensors are skipped.
```

**Two corrections `[retracted]`.**

- *"Every rank built and loaded 347 unused tensors — 1.05 GiB of BF16 vision weights — for the whole
  life of the TP=2 stack."* Wrong twice. That describes the **routed-experts-only** checkpoint, where
  the tower really is dense BF16; on the production full-scope checkpoint the tower is **6-bit EXL3,
  0.557 GiB in total and 0.416 GiB per rank** ([18](18-vision-at-three-ranks.md) §3.1). And on image
  `exl3-zeus:754421f` the flag probably spends nothing at all: `_mark_tower_model` enters
  `no_init_weights`, so the tensors stay on the meta device and `AutoWeightsLoader` skips the module
  — a reading of the installed source, never confirmed from a boot log `[not tested]`. What the flag
  costs at TP=2 is therefore **unmeasured and probably zero**, not 1.05 GiB.
- *"`--mm-encoder-tp-mode data` would also clear the assert but would keep carrying the 1.05 GiB."*
  Wrong in both halves. It clears the assert and it is what production 13 runs, at 0.416 GiB per
  rank; and on its own it does **not** make the tower load at all, because the checkpoint's tower is
  EXL3 while vLLM builds it with `quant_config=None`
  ([18](18-vision-at-three-ranks.md) §3.1–§3.2).

Padding the tower's heads is still the wrong fix, but for a different reason than "the tower is never
executed": at three ranks **nothing** in the tower divides by three — not the 16 heads, not the 4096
MLP, not the merger's 4096 and 10240 — so it is four padded shapes and a new pad audit to save
0.28 GiB per rank ([18](18-vision-at-three-ranks.md) §2b).

---

## 4. Expert parallelism: what it does and what proves it happened

Under `--enable-expert-parallel`, vLLM reports `tp_size = 1, tp_rank = 0` to the MoE method and hands
it 96 local experts, so the placement narrows by nothing and every trellis stays 2,048 wide.
`patches/tp3/patch-exl3-ep.py` installs `patches/tp3/overlay/cuda_exl3/_harem_ep.py` and rewires the
call sites.

Two of the four edits this overlay originally carried are now upstream in `cuda-exl3` and have been
retired — see [05-expert-parallel-and-cuda-exl3-fixes.md](05-expert-parallel-and-cuda-exl3-fixes.md).
What remains, because upstream has no equivalent, is a pair of fail-closed load-time checks:

- `check_expert_shape()` refuses a trellis split that is not 128-aligned, and logs the evidence line.
- `check_trellis_slice()` refuses a mis-shaped trellis — which otherwise loads without error and
  decodes to noise.

The evidence line to look for, on **every** rank:

```
EXL3 routed experts: mode=EP ep_size=3 experts_local=96/288 hidden=4096 intermediate_local=2048 trellis=whole
```

`intermediate_local=2048` and `trellis=whole` are the two words that say the trellis was never cut.

**A logging trap worth knowing.** For the first TP=3 boots that line appeared nowhere, and the
absence was not evidence of anything: the module created its logger under its own package name, and
the engine's logging configuration attaches a handler only to the `vllm` logger while the root logger
has none — so every record from that module was discarded `[measured-here]`. The fail-closed *raises*
were unaffected (exceptions do not go through logging), but the evidence line and a
"routed experts are being TENSOR-sliced" warning — a safety net — were invisible. That is exactly
backwards. The logger is now created under the `vllm.` prefix.

---

## 5. Proving the arithmetic instead of assuming it

`tracks/tp3/patches/preflight-tp3.py` runs inside the container, against **whatever the launcher
actually mounted** rather than against what the environment file says it mounted, and it owns the
EP-versus-tensor-sliced decision:

- It refuses `ENABLE_EP=0` unless `moe_intermediate_size` is a multiple of `128 × tp` **and** the
  weights on disk agree with the config. (An earlier blanket rule of "TP=3 always needs EP" was true
  of this 2,048-wide checkpoint only, and it has been narrowed to the actual arithmetic.)
- It checks every shape in §1 individually, with the unit at `lcm(128, tp)` and three further gates
  that §1.1 is why: the padded vocab, the per-rank vocab and the per-rank shared-expert width must
  each be a whole number of 128s.
- It reads the drafter's real head counts out of the safetensors header.

Run it standalone against a sidecar before your first boot:

```
tracks/tp3/patches/preflight-tp3.py --model /var/tmp/glm-5.3-flash-turboderp-4.05bpw-tp3 --tp 3 --ep 1
```

Expected, and this is the whole of §1 in five lines:

```
vocab padding_size 384 -> 155,136 -> 51,712/rank (404 x 128); shared expert 2048 -> 2304 -> 768/rank (6 x 128)
MLA 66/22; KDA 66/22; routed experts 288 -> 96/rank; expert parallel mandatory
```

### 5.1 Prove the pad is zero, not just that it exists

Two of these run, and they are answering the same question about different tensors.

**For the target's EXL3 pads (assert 5, production 9).** `patch-fullscope-tp3.py`'s A10 wraps
`load_weights` and walks every EXL3 module while `suh` and `svh` are still raw. For an output pad
(column-parallel, and the vocab head) it requires the real width and the pad to be multiples of 128
and `svh[real:]` to be exactly zero; for an input pad (row-parallel only, decided from the module's
own geometry, `exl3_k * tp == input_size`) the same alignment gates and `suh[0, real:] == 0`. One line
comes out of it, and **its absence is a failure**: if the line is not in the log, the audit did not
run.

```
HAREM-FULLSCOPE assert 5: 285 EXL3 pad site(s) audited, 285 padded on this rank, all whole 128-blocks and exactly zero
```

285 is the count a model-free meta-device run predicted for rank 2 **before** the boot — 1 `lm_head`,
42 × 3 shared-expert sites, 34 × 3 `in_proj_qkv` shards, 34 KDA `o_proj` inputs, 11 MLA `o_proj`
inputs and 11 `q_b_proj` — so the reading agrees with an independent count rather than merely being
self-consistent `[measured-here]`.

**The audit caught its own bug on first writing, and that is the reusable part.** It called a
column-parallel module's *input* padded (`k_real != k_local * tp`) and rejected `in_proj_qkv` on
ranks 1 and 2. Rank 0 passed. **A single-rank test could not have seen it**, which is the same
lesson as §3's vision tower and [09](09-measurement-protocol.md) §11.3: run the check at the real
rank count, and at TP=3 the *last* rank is mandatory, because the pad lives there.

**For the drafter's BF16 pads.** The older check, and it runs on both checkpoints: read the fabricated rows of every layer's
`qkv_proj` and the matching input columns of `o_proj`, and fail if any is non-zero. A padded head
whose rows are zero produces a zero attention output and a zero `o_proj` contribution — an exact
no-op. A padded head holding allocator garbage is a **fluent drafter proposing slightly wrong
tokens**: acceptance falls, nothing crashes, and no log says why.

Measured at boot, on the only rank that has padding `[measured-here]`:

```
HAREM-TP3 drafter pad: 32/8 (checkpoint) -> 36/9 (config) at tp=3; rank 2 owns 8 real + 4 padded query heads
HAREM-TP3 drafter pad verified zero on rank 2/3: 26,214,400 elements across 4 padded query head(s) and 1 padded KV head(s)
```

26,214,400 is exactly `5 layers × (4×128 q + 128 k + 128 v) × 4096 + 5 × 4096 × 512` — the predicted
count, arrived at independently by arithmetic before the run.

---

## 6. What TP=3 bought, and what it cost

Against the same stack on two nodes, same image, same draft, same prompts `[measured-here]`:

| | TP=2 + DFlash2 | **TP=3 + EP + DFlash2** | change |
|---|---|---|---|
| weights per node | 81.53 GiB | **54.86 GiB** | −33 % |
| memory left for KV | 17.27 GiB | **39.86 GiB** | +131 % |
| KV pool | 825,000 tokens | **2,947,441** | +257 % |
| KV usage across the whole benchmark | 66–76 % at C8 | 2–13 %, queue always 0 | — |
| correctness probe / code exam | 10/10 · 12/12 | 10/10 · 12/12 (twice) | same |
| acceptance / tokens per step | 62.4 % · 5.37 | 62.9 % · 5.36 | same |

**Acceptance and tokens-per-step identical to three significant figures is the most important row in
that table.** It says the drafter and the target are both intact — the 96-head failure mode this
whole design exists to avoid did not happen, and any efficiency loss is efficiency, not a broken
kernel producing plausible-looking rubbish.

**What it cost, at first: throughput.** The initial TP=3 arm was −8 % at C1 and −29 % at C4 against
TP=2. That regression turned out to be a one-line bug in the `cuda-exl3` GEMM dispatch, not an
inherent cost of the arrangement; after the fix TP=3 beats TP=2 on every axis. The whole story is in
[05-expert-parallel-and-cuda-exl3-fixes.md](05-expert-parallel-and-cuda-exl3-fixes.md), and it is the
best argument in this repository for measuring model-free before believing an end-to-end number.

**What it costs permanently:** three nodes instead of two, three-way collectives instead of two-way
(small at decode, about 3 % of a prefill chunk), and — because each rank's share of the model's
unquantized half is a third rather than a half — those BF16 GEMMs get less efficient per rank
([01-model-and-license.md](01-model-and-license.md) §3.2).

---

## 7. Boot order, and the gates

Start the workers first and the head last. Tear all three down before relaunching any of them.

```
~/exl3-zeus/start-tp3.sh 2
```

```
~/exl3-zeus/start-tp3.sh 1
```

```
~/exl3-zeus/start-tp3.sh 0
```

Then, before believing any speed number, run the two quality gates
([09-measurement-protocol.md](09-measurement-protocol.md)). Expected: correctness probe **10/10**
(client-visible 9/9, empty content 0) and code exam **12/12**.

```
docker rm -f exl3-tp3
```

stops a rank; run it on all three.

---

## 8. What is next

[04 — The DFlash2 port](04-dflash2-port.md), which is the other half of what makes this
configuration fast, and [05](05-expert-parallel-and-cuda-exl3-fixes.md), which is what makes it
fast *enough*.
