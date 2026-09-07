# 13 — The full-scope checkpoint: five loader layers, a padded load, and the dense stage measured

**Applies to: both tracks.** §4–§6 are the TP=2 dress rehearsal; §7, the padded-load port, applies
only at three ranks.

**This is the production recipe.** Since 5 September 2026, 18:40 local, this stack serves
`turboderp/GLM-5.3-Flash-exl3` at 4.05 bpw at TP=3 with expert parallelism — production
configuration 9.

Every configuration before it served `scope: glm53_routed_experts_only`, so attention, the KDA
layers, the shared experts and `lm_head` streamed BF16 weights beside a 4-bit routed half. That dense
stage was **45.3 % of a single-stream decode step** ([10](10-results-and-roofline.md) §5.3) — the
largest single item this repository has carried, and larger than everything in the `cuda-exl3` column
of its target table put together.

This page is what happened when we removed it: the three independent reasons a full-scope checkpoint
would not load at all (§2), the loader patch (§3), the dress rehearsal at TP=2 (§4–§6), and the TP=3
port that put it in production (§7) — **+22.9 % total and +21.7 % per stream at C1, +12.5 % at C8,
KV pool +10.0 %, 3.4 GiB lighter per node, quality unchanged, and a draft-acceptance cost that we
published and then had to withdraw** (§7.4).

**Two reframings are worth reading even if you never load this checkpoint.**

The first: we had written the scope of the old checkpoint down as a quality decision by its
publisher. It was not. Two lines in the vLLM `glm5next` model file pin the whole attention stack to
BF16 regardless of what the weights contain, and they lock **72.8 %** of the dense traffic. Until
those lines are conditional, no checkpoint of any scope can put attention on EXL3 — so
`routed_experts_only` was not a choice about quality, it was the only thing that could load
([11](11-open-issues.md) §1.9 row 29).

The second: **an EXL3 tensor cannot be zero-extended, but it can be loaded narrow into a parameter
the engine has padded** — with `svh = 0` on the pad, and only when the pad occupies whole 128-column
Hadamard blocks. That single sentence is the whole of the TP=3 port, and it was already happening on
this stack, unchecked, before anyone designed it (§7.1).

---

## 1. The checkpoint

| | |
|---|---|
| Repository | `turboderp/GLM-5.3-Flash-exl3`, branch `4.05bpw` |
| Revision we ran | `2a30229e67012798ba9f0cd832bb78abf4c363d5` (short `2a30229e`, 28 August 2026) |
| Licence | **MIT** — the `LICENSE` file is the MIT text, "Copyright (c) 2026 Z.AI Co., Ltd", and the model card carries `license: mit` `[reported]`. More permissive than the checkpoint it replaced ([01](01-model-and-license.md) §2): no attribution condition, no exclusion clause |
| Size on disk | **165.2 GB / 153.8 GiB** — 19 shards (~8.59 GB each), plus `mtp.safetensors` 3.79 GB, `quantization_config.json` 47.9 MB and a 16.0 MB index `[measured-here]` |
| Verified | `sha256` 23/23 against the repository's own LFS metadata, independently on both nodes `[measured-here]` |
| Format | exl3 v1.4.4, codebook `mul1`, `bits: 4.05`, `head_bits: 6`, `out_scales: always`, calibration 250 rows × 2,048 columns |
| Architecture | `Glm5NextForConditionalGeneration` — **the vision tower is present and is itself 6-bit EXL3**: 172 modules, 1,007 tensors, all in shard 19, 0.557 GiB, in `trellis`/`suh`/`svh`/`mul1` form, with **no dense `.weight`** for `proj`, `mlp.*` or `merger.*`. It is **not** listed in `quantization_config.json`'s `tensor_storage` and is recovered by `cuda_exl3`'s `_augment_from_checkpoint()` from the safetensors headers — the first wall anyone else loading this checkpoint with the tower on will hit. Served since 7 September 2026 ([18](18-vision-at-three-ranks.md)); `--language-model-only` is the text-only fallback |
| Index | 148,046 tensors, of which **36,719 are `.trellis`**; `model.visual.*` (1,007) is quantized too; `mtp.safetensors` is **not** in the index, so vLLM never reads it |

**It is not smaller, and that surprised us.** The expectation going in was 55–60 GB. The routed
experts are already 4-bit in *both* checkpoints, so full scope only takes the remaining ~11 % of the
weights from BF16 down to 4–6 bits: 164 GiB → 154 GiB. **The gain is in decode traffic, not on disk**
— and at TP=3 the 10 GiB saved on disk is repaid again in host memory: 3.4 GiB less consumed per
node, which is +10.0 % of KV pool (§7.3). At TP=2 the same reading came out the other way and we
could not explain it (§6.2); it did not reproduce (§7.5).

### 1.1 What is quantized, and at what bitrate

Read from `quantization_config.json` (37,032 records) `[measured-here]`:

| module class | count | bits | codebook |
|---|---|---|---|
| `mlp.experts.*.{gate,up,down}_proj` (routed) | 12,096 × 3 | **4** | mul1 |
| `mlp.shared_experts.{gate,up,down}_proj` | 42 × 3 | 6 | mul1 |
| `mlp.{gate,up,down}_proj` (dense layers 0–2) | 3 × 3 | 5 | mul1 |
| `self_attn.qkv_proj` (KDA layers) | 34 | 6 | mul1 |
| `self_attn.o_proj` (every layer) | 45 | 6 | mul1 |
| `self_attn.{q_a,q_b,kv_a_proj_with_mqa}` (MLA) | 11 × 3 | 6 | mul1 |
| `self_attn.indexer.wq_b` | 11 | 6 | mul1 |
| **`lm_head`** | 1 | **6** | mul1 |
| stays BF16 | `embed_tokens`, every norm, `mlp.gate` (the router), KDA `b_proj`/`f_a_proj`/`f_b_proj`/`g_a_proj`/`g_b_proj`, `indexer.wk`/`weights_proj`, the three `conv1d` | — | — |

For comparison, the fallback checkpoint's `tensor_storage` holds **only** the 37,152 routed-expert
tensors (`bits 4`, codebook `mcg`, `head_bits 16`). The dense and attention path of the plugin had
therefore **never been exercised on this stack** before this arm.

What that turns into once loaded, read off `CUDA_EXL3_DEBUG_NAMES=1` on the live TP=3 boot rather
than inferred `[measured-here]`: **203 EXL3 linears and 113 BF16** per rank, plus 42
`Exl3MoEMethod` routed-expert layers. The 113 are exactly four families —
`self_attn.f_b_proj` (34), `self_attn.g_b_proj` (34), `self_attn.in_proj_bfg_a` (34) and MLA
`self_attn.kv_b_proj` (11) — the ones this checkpoint leaves unquantised. **The negative read is the
gate:** `o_proj`, `in_proj_qkv`, `q_b_proj`, `fused_qkv_a_proj`, `gate_up_proj`, `down_proj`,
`indexer.wq_b` and `lm_head` do **not** appear in the `-> unquantized` list. (`embed_tokens`,
`mlp.gate` and the three `conv1d` are BF16 too but never pass through the plugin's resolver, so they
are outside this tally; the meta-device run counted them separately at
`UnquantizedLinearMethod` 268 and `UnquantizedEmbeddingMethod` 1.)

### 1.2 What BF16 traffic there is to remove, and which layer removes it

Computed from the checkpoint's own tensor shapes `[measured-here]`. The total, 15.77 GB, agrees
independently with the ≈16.8 GiB the plugin's own `online.py` docstring quotes for this model — two
sources, one number.

| group | n | bits in this checkpoint | BF16 GB today | opened by |
|---|---|---|---|---|
| `self_attn.qkv_proj` (KDA) | 34 | 6 | **6.845** | **S3** |
| `self_attn.o_proj` | 45 | 6 | **3.758** | **S2** |
| `lm_head` | 1 | 6 | 1.269 | already resolvable |
| `shared_experts.{gate,up}_proj` | 84 | 6 | 1.410 | **S1** |
| `shared_experts.down_proj` | 42 | 6 | 0.705 | already resolvable |
| `self_attn.q_b_proj` | 11 | 6 | 0.554 | **S2** |
| dense MLP `{gate,up}_proj` | 6 | 5 | 0.604 | **S1** |
| dense MLP `down_proj` | 3 | 5 | 0.302 | already resolvable |
| `q_a_proj` + `kv_a_proj_with_mqa` | 22 | 6 | 0.184 | **S1 + S2** |
| `self_attn.indexer.wq_b` | 11 | 6 | 0.138 | **S2** |
| **total** | | | **15.768** | |

(`embed_tokens` adds 1.269 GB of BF16 that is gathered rather than read per token, and is not in the
lever.)

| arm | moved to EXL3 | dense traffic left | reduction |
|---|---|---|---|
| stock (does not boot) | 2.276 GB | 14.34 GB | 9.0 % |
| S1 + S2 | 8.924 GB (56.6 %) | 10.19 GB | 35.4 % |
| **S1 + S2 + S3 (what we ran)** | **15.768 GB (100 %)** | **5.92 GB** | **62.5 %** |

---

## 2. Why it did not load: three independent layers

Two boots died before any of this was understood, on `KeyError: layers.2.self_attn.o_proj.mul1` and
then on `KeyError: layers.1.self_attn.conv1d.weight`. The second one is the informative one:
`conv1d` is a **BF16** tensor, so the failure is not about quantization at all — a BF16 copy of the
same checkpoint would die in the same place.

The diagnosis was then done model-free, by instantiating the real vLLM model on
`torch.device("meta")` and reproducing `params_dict` analytically against the checkpoint's weight map
`[measured-here]`. "Unmapped" means a checkpoint tensor with no parameter to load into, which is a
`KeyError` at boot rather than a silent half-load.

| checkpoint tensor group | count | stock | + S1 | + S2 | + S3 |
|---|---|---|---|---|---|
| `mlp.shared_experts.gate_up_proj.*` | 168 | unmapped | **OK** | OK | OK |
| `mlp.gate_up_proj.*` (dense 0–2) | 12 | unmapped | **OK** | OK | OK |
| `self_attn.o_proj.*` | 180 | unmapped | unmapped | **OK** | OK |
| `self_attn.fused_qkv_a_proj.*` | 88 | unmapped | unmapped | **OK** | OK |
| `self_attn.q_b_proj.*` | 44 | unmapped | unmapped | **OK** | OK |
| `self_attn.indexer.wq_b.*` | 44 | unmapped | unmapped | **OK** | OK |
| `self_attn.qkv_proj.*` (KDA) | 136 | unmapped | unmapped | unmapped | **OK** |
| `self_attn.conv1d.weight` | 34 | unmapped | unmapped | unmapped | **OK** |
| **total unmapped** | | **886** | **526** | **170** | **0** |

**The same derivation gives 0 unmapped and 0 unfilled for the experts-only checkpoint**, which is the
check that it reproduces "loads fine" rather than merely producing a number.

### 2.1 S1 — the model class declares no `packed_modules_mapping`

vLLM merges several checkpoint tensors into one layer:

```text
.gate_up_proj      <- .gate_proj , .up_proj
.fused_qkv_a_proj  <- .q_a_proj  , .kv_a_proj_with_mqa
.wk_weights_proj   <- .wk        , .weights_proj
.in_proj_qkvbfg_a  <- .q_proj , .k_proj , .v_proj , .b_proj , .f_a_proj , .g_a_proj
```

`Exl3Config._candidate_names()` can invert that, but only through `self.packed_modules_mapping`,
which vLLM copies off the model class (`model_loader/utils.py:275,281`). `glm5next` declares none, so
the dictionary is `{}` and every merged linear fails to resolve. **This is invisible for a
routed-experts-only checkpoint** — those layers are BF16 anyway — **and fatal for a full-scope one.**

### 2.2 S2 — the attention stack is pinned to BF16 in the model file

This is the layer the first analysis missed, and it is the largest.

- **MLA:** `model.py:331` passes `quant_config=None` with the comment "MLA projections are BF16 in
  checkpoint", so 11 layers' `fused_qkv_a_proj`, `q_b_proj`, `kv_b_proj`, `o_proj` and `indexer.wq_b`
  get `UnquantizedLinearMethod`.
- **KDA:** `kda.py:171-174` sets `vllm_config.quant_config = None` for the duration of
  `super().__init__`, and the base class captures it (`mamba/gdn/base.py:41`), so
  `self.quant_config` stays `None` **permanently** and everything built after it —
  `in_proj_qkvbfg_a`, `f_b_proj`, `g_b_proj`, `o_proj` — is unquantized.

`Exl3Config.get_quant_method` is therefore **never called** for attention. An earlier note of ours
that "`o_proj` resolves 45/45" was correct and irrelevant: `resolve()` works, nothing asks it.
Together those two lines lock **72.8 % of the dense traffic**, and neither has anything to do with
packing. Both are legitimate for an fp8 checkpoint, where those projections really are BF16 — the
fix is to condition them on the quantization method, not to delete them.

### 2.3 S3 — the KDA block is factorised differently

The checkpoint serialises KDA attention the way the upstream Hugging Face release does; the NVIDIA
`glm5next` implementation expects a different factorisation.

| what `kda.py` builds | what this checkpoint has | what our production checkpoint has |
|---|---|---|
| `q_proj` / `k_proj` / `v_proj` as shards 0/1/2 of `in_proj_qkvbfg_a` | **one** EXL3 `self_attn.qkv_proj`, `[256, 1536, 96]` trellis = (4096/16, 24576/16, 16×6) | three separate BF16 tensors |
| `q_conv1d` / `k_conv1d` / `v_conv1d` | **one** `self_attn.conv1d.weight`, BF16 `[24576, 1, 4]` | three BF16 `[8192, 1, 4]` |
| `b_proj`, `f_a_proj`, `g_a_proj` as shards 3/4/5 | present, BF16 | present, BF16 |

So the producer of our production checkpoint did exactly two things upstream does not: split
`qkv_proj` three ways on the output dimension, and split `conv1d` three ways on dim 0. Pure slicing,
no re-quantization.

`packed_modules_mapping` cannot express either of them, and behind the naming sits a structural
problem: shards 0–2 of `in_proj_qkvbfg_a` would be EXL3 while shards 3–5 stay BF16, and
`Exl3LinearMethod` requires a linear to be **wholly** EXL3 or wholly BF16 (`resolve()` needs
`all(i is not None)`). A mixed-precision merged linear cannot be expressed today. The way out is to
split the module.

### 2.4 Rewriting the checkpoint offline instead — evaluated and rejected

The obvious alternative is to rewrite the 19 shards into the layout our production checkpoint uses
(pure name and slice rewriting, no re-quantization) so that no runtime patch is needed. It removes
**only S3**, and not even all of it:

- **S1 cannot be removed on disk.** `gate_proj` and `up_proj` were quantized separately, so their
  `suh` differ, and `suh` lives *inside* the Hadamard. They cannot be pre-fused; whatever the layout,
  the packed mapping is still required.
- **S2 cannot be removed on disk.** `quant_config=None` is the model's decision and has nothing to
  do with the checkpoint's names.
- **S3b cannot be removed on disk.** Even with `qkv_proj` pre-split, shards 0–2 are EXL3 and 3–5 are
  BF16, so `resolve()` still returns `None`. Splitting the module is unavoidable.

Against that: ~154 GiB read plus ~154 GiB written per node, on nodes at 95 % full; the loss of a
`sha256` 23/23 provenance match against the publisher; and a 165 GB re-download if it goes wrong.
**Rejected** — it is larger than the patch, not smaller, and it is one-way.

---

## 3. The patch

One file, in the same style as the rest of `patches/`: exact-text anchors that must match **exactly
once**, `py_compile` before the write, atomic replace, idempotent, a `--check` mode, a post-check, and
a hard exit if anything does not match. Two target files at TP=2:
`vllm/models/glm5next/nvidia/model.py` and `.../kda.py`; a third, `linear.py`, at TP=3.

There are two copies of it, and the difference is exactly A9 and A10:

- [`tracks/tp2/patches/patch-fullscope-tp2.py`](../tracks/tp2/patches/patch-fullscope-tp2.py) — eight anchors,
  A1–A8. This is the file the TP=2 measurement in §4 ran on. It is kept because it is the smaller,
  more readable statement of the same three layers, and because at TP≤2 nothing needs padding.
- [`tracks/tp3/patches/patch-fullscope-tp3.py`](../tracks/tp3/patches/patch-fullscope-tp3.py) — ten
  anchors, A1–A10, and the one production runs (§7).

| # | file | layer | what it does |
|---|---|---|---|
| A1 | `model.py` | S1 | module-level helpers plus `if <env>: packed_modules_mapping = …` in `Glm5NextForCausalLM` |
| A2 | `model.py` | S1 | the same conditional class attribute in `Glm5NextForConditionalGeneration` |
| A3 | `model.py` | S2 | MLA `quant_config=None` becomes conditional on the quant method being EXL3 |
| A4 | `model.py` | S3b | replaces the six `.in_proj_qkvbfg_a` entries in `stacked_params_mapping` with four new ones, then asserts there are four |
| A5 | `model.py` | S3a | the `conv1d` three-way split in `load_weights` (plus the `ReplicatedLinear` fix, §3.3) |
| A6 | `kda.py` | S2 | the unconditional `quant_config` strip becomes conditional |
| A7 | `kda.py` | S3b | splits `in_proj_qkvbfg_a` into an EXL3 `in_proj_qkv` and a BF16 `in_proj_bfg_a`; records the checkpoint's real shard width on the module; installs asserts 1 and 2 |
| A8 | `kda.py` | S3b | `forward` becomes two calls when split, the upstream single call otherwise |
| **A9** | `linear.py` | TP=3 | splits a pre-fused checkpoint tensor by the **checkpoint's** widths, not the module's padded `output_sizes` (§7.1). **No-op at TP≤2**, where the two lists are equal |
| **A10** | `model.py` | TP=3 | assert 5: the post-load audit that every EXL3 pad is whole 128-blocks and exactly zero (§7.1) |

**Everything is gated on one environment variable**, and with it unset the patched image is upstream
byte for byte. Measured rather than asserted: with the flag unset the class attribute is never
created (a bare `if` in a class body), so vLLM's `getattr(model_class, "packed_modules_mapping",
None)` returns `None`, both quant gates pass `None`, the KDA module is not split, and
`stacked_params_mapping` keeps its six upstream entries `[measured-here]`. The same patched image
serves the experts-only checkpoint with 0 unmapped and 0 unfilled.

One further safety property fell out of the design: **the split decision is taken from the
checkpoint, not from the environment** (`quant_config.resolve(f"{prefix}.in_proj_qkv") is not None`).
Booting the experts-only checkpoint with the flag **set** gives split inactive, 0 unmapped, 0
unfilled, everything BF16 `[measured-here]`. A wrong env file cannot silently corrupt a run.

### 3.1 The mapping, and the three things deliberately not in it

```json
{"gate_up_proj": ["gate_proj", "up_proj"],
 "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
 "in_proj_qkv": ["qkv_proj"]}
```

- **`lm_head` is absent on purpose.** It is a `VocabParallelEmbedding`: the trellis puts vocab on
  dim 1 where a plain linear has it on dim 0, and `cuda-exl3` loads the head through its own
  `_vocab_loaders`. Putting it in the mapping would load without error and be wrong.
- **`in_proj_qkvbfg_a` is absent on purpose.** Mixed precision; `resolve()` returns `None` for it
  whatever the mapping says. The fix is the module split, not a name.
- **`wk_weights_proj` is absent on purpose.** Both halves are BF16 in this checkpoint and the module
  is already built with `quant_config=None`.

The plugin's own README carried a different example, `{"in_proj_qkvbfg_a": ["qkv_proj", "conv1d"]}`,
which resolves **0/34** here: the group must be entirely EXL3 and `conv1d` is BF16 and additionally
split three ways. We measured it, reported it, and the author corrected the documentation and added
the rule that explains it ([CREDITS](../CREDITS.md)).

### 3.2 The shard-index trap, and the four asserts

The one genuinely dangerous failure mode is a wrong shard index: the load completes **without error**
and the numbers are wrong. The order of the sources in `packed_modules_mapping` must say the same
thing as the shard ids in `stacked_params_mapping`:

```text
(".in_proj_qkv",   ".qkv_proj",  (0, 1, 2))   # one checkpoint tensor, three vLLM shards
(".in_proj_bfg_a", ".b_proj",    0)
(".in_proj_bfg_a", ".f_a_proj",  1)
(".in_proj_bfg_a", ".g_a_proj",  2)
```

**A tuple shard id must start at 0.** `weight_loader_v2` indexes the tuple *relatively*
(`linear.py:901-916` builds `output_sizes` from the tuple, then `_load_fused_module_from_checkpoint`
enumerates 0,1,2), so `(3,4,5)` would load without error and write to the wrong slice. `in_proj_qkv`
uses `(0,1,2)`, which is the second reason to split the module rather than keep one module with a
BF16 tail.

Why the tuple is free rather than expensive: `suh` is a `PerTensorScaleParameter`, so the same row is
replayed for each index without narrowing, and `process_weights_after_loading` then collapses the
three identical `suh` rows into one group — the kernel sees a single fused GEMM again.

vLLM's own "following weights were not initialized" gate does **not** protect this, because
`process_weights_after_loading` is defined on the base class and every parameter of every linear
counts as loaded. So the patch carries its own gates:

| # | where | what it checks |
|---|---|---|
| 1 | after `in_proj_qkv` is created | three EXL3 shards of the expected width, and the expected total |
| 2 | around `process_weights_after_loading` | the three identical `suh` rows collapsed to **one** group; if not, different tensors reached different shards |
| 3 | while splitting `conv1d` | the three thirds are **different from each other** (identical thirds mean the split was on the wrong axis) and the row count divides by three |
| 4 | at the head of `load_weights` (no data needed) | mapping order against shard ids, tuples starting at 0 and contiguous, and `lm_head` **not** in the mapping |

**All four were silent on the live boot** `[measured-here]`.

### 3.3 A vLLM bug found on the way: `ReplicatedLinear` has no `weight_loader_v2`

The meta-device dry run raised something the design had not predicted:

```text
AssertionError: Tried to load weights of size torch.Size([1536])
                to a parameter of size torch.Size([1, 1536])
```

The sparse indexer's `wq_b` is a `ReplicatedLinear`, and vLLM's `ReplicatedLinear` never dispatches
to `weight_loader_v2` — its only loader asserts `param.size() == loaded_weight.size()`, which a
per-shard `suh` of shape `(num_shards, k)` can never satisfy. **Any replicated linear served by a
v2-parameter quantization method fails at load.** It had never shown up because `wq_b` is BF16 in the
experts-only checkpoint; S2 quantizes it for the first time. Leaving it BF16 is not an option — this
checkpoint has no BF16 copy of it, so 44 tensors would be orphaned and the boot would still stop.

Our fix routes that parameter through the v2 entry point with the shard index pinned at 0, and
limits itself by checking that the parameter actually carries the v2 interface. A second detail sits
underneath it: `ReplicatedLinear` never calls `update_param_tp_status()`, so its parameters carry the
**global** TP rank; a replicated layer holds the whole tensor on every rank, so the copy has to pin
`tp_rank = 0` or rank 1 overruns on `narrow`. **A single-rank test could not have caught that; the
two-rank meta-device run did** `[measured-here]`.

We reported it as a vLLM gap. The author's reading was better and he fixed it on his side instead:
`suh` is `(num_shards, k)`, the v1 path does not know that, so the plugin now places a bare `(k,)`
into row 0 when the shapes say that is what happened and delegates otherwise (`d19dee0`). Our
workaround can retire once that commit is in an image we run.

---

## 4. The measurement, at TP=2 — the dress rehearsal

This section and the two after it are the TP=2 arm, run earlier the same day. It is kept in full
because it is what decided the TP=3 port was worth building, because two of its readings did not
survive that port (§7.5), and because it is the only place the loader work can be seen without the
padding machinery on top of it. **The production numbers are in §7.3.**

**Why TP=2.** At two ranks nothing needs padding — 32 heads and 77,440 vocab rows per rank — so the
question "what is the dense stage worth" is answered with no new padding machinery. TP=3 needs a
padded-load path that does not exist yet (§7).

> **If you are here to run at TP=2 rather than to read the rehearsal**, the two-node track is
> [15](15-tp2-track.md): the exact file and flag changes, all four of our two-node arms with their
> dates and settings, and the list of production features we never ran there. Read §6.1 below first —
> at two ranks this checkpoint closes the long-prompt path, and that is not a TP=3 problem.

**Settings, both arms identical unless the row says otherwise:** two DGX Spark nodes, TP=2, expert
parallel **off**, image `exl3-zeus:62f53e6`, EXL3 weights, KV `fp8`, DFlash2 draft k=7,
`gpu-memory-utilization 0.85`, `--block-size 256` requested, `--max-num-seqs 8`,
`--max-num-batched-tokens 2048`, `NCCL_MAX_NCHANNELS=8`, mesh plugin with both cables per peer,
`CUDA_EXL3_TUNE_CACHE` warm, no fast-load sidecar in either arm, no `HAREM_SW_BLOCK_SIZE`, no fp8
draft cache in either arm, temperature 0, reasoning effort **low**, medians of three rounds,
5 September 2026. Control is `brandonmusic/GLM-5.3-Flash-tr3-4bpw` on the same stack.

**One setting is not identical and it matters:** the full-scope arm runs `max_model_len` **65,536**
against the control's **1,000,000**, because at 1M it could not boot (§6.1). That changes the hybrid
allocator's page size as well as the pool, so **only C1–C4 are treated as a comparison**; C6/C8, every
prefill figure and the pool are not.

### 4.1 Speed

| metric | **full-scope** | experts-only control | delta | verdict |
|---|---|---|---|---|
| **C1 per stream (tok/s)** | **68.00** | 54.69 | **+24.3 %** | the lever |
| **C1 aggregate (tok/s)** | **59.93** | 47.40 | **+26.4 %** | the lever |
| C2 aggregate | 83.02 | 68.03 | +22.0 % | faster |
| C4 aggregate | 111.05 | 90.66 | +22.5 % | faster |
| C6 aggregate | 109.75 | 110.12 | — | **KV-bound, void** |
| C8 aggregate | 110.03 | 133.57 | — | **KV-bound, void** |
| TTFT at C1 (s) | 0.524 | 0.615 | −14.8 % | faster |
| draft acceptance at C1 | 63.14 % | 64.08 % | equal | drafter unchanged |
| accepted tokens per step at C1 | 5.42 | 5.49 | equal | drafter unchanged |
| cold first request | TTFT 0.85 s, 47.5 tok/s | TTFT 1.44 s, 40.5 tok/s | +17 % | faster |
| boot, cold, no fast-load sidecar | 355 s | 396 s | −10 % | smaller checkpoint |
| gates, cold and warm | **10/10 · 12/12** | 10/10 · 12/12 | equal | pass |
| free host RAM after the run | 8.9 / 10.6 GiB | 6.6 / 7.7 GiB | +2.3…2.9 | lighter weights |
| KV pool | 31,343 at 65,536 ctx | 665,625 at 1,000,000 ctx | — | **not comparable** |
| prefill (7k and fresh) | **not measurable** | 1,135 / 1,334 tok/s | — | see §6.1 |

All three rounds of the full-scope arm, for the record `[measured-here]`:

| round | C1 aggregate | C1 per stream | C2 | C4 | C6 | C8 | acceptance at C1 |
|---|---|---|---|---|---|---|---|
| 1 | 60.21 | 69.28 | 83.02 | 110.15 | 109.75 | 110.26 | 63.66 % |
| 2 | 58.39 | 65.38 | 83.32 | 111.05 | 111.47 | 110.03 | 61.46 % |
| 3 | 59.93 | 68.00 | 81.64 | 111.43 | 106.18 | 108.71 | 63.14 % |

> **Correction to the figure we posted upstream.** The table sent to the kernel author's issue thread
> gave the control's C1 as "54.7 / 54.3" for aggregate / per stream and derived "+9.5 % / +25 %".
> 54.7 is the control's **per-stream** median and 54.3 is its round-3 per-stream value; its aggregate
> median is **47.40**. The like-for-like deltas are the ones in the table above: **+26.4 % aggregate
> and +24.3 % per stream** `[retracted]` ([11](11-open-issues.md) §1.9 row 31). The full-scope column
> and the conclusion are unaffected.

### 4.2 Where the gain comes from, and how much of the estimate arrived

```text
control  : 5.49 tokens/step ÷ 54.69 tok/s = 100.4 ms/step
full-scope: 5.42 tokens/step ÷ 68.00 tok/s =  79.7 ms/step
saved     : 20.7 ms/step  (step −20.6 %, throughput +24.3 %)
```

The profile attributed **42.90 ms of a 94.65 ms step** to dense BF16 GEMM
([10](10-results-and-roofline.md) §5.3), and the estimate was that 4-bit takes it to ~11 ms — about
32 ms, roughly **+34 % single-stream** ([11](11-open-issues.md) §2.22).

**We got 20.7 ms, 65 % of it.** The gap is explained rather than mysterious: this checkpoint leaves
`in_proj_bfg_a`, `f_b_proj`, `g_b_proj`, `kv_b_proj`, `indexer.wk_weights_proj` and the three
`conv1d` in BF16 anyway, and not every byte in the dense stage is bandwidth-bound. **The +34 % was
not refuted; it was an upper bound and it was optimistic** `[estimate]`.

**Acceptance and accepted tokens per step are unchanged**, so the entire gain is arithmetic and none
of it is drafter behaviour. That is also the strongest evidence the lever carries to TP=3: there is
nothing on the speculative side to re-tune, and at three ranks the per-rank dense traffic falls
further.

### 4.3 What actually resolved to EXL3

From the two-rank meta-device dump, per rank, TP=2 `[measured-here]`. Quant-method census:
`Exl3LinearMethod` 203, `Exl3MoEMethod` 42, `UnquantizedLinearMethod` 268,
`UnquantizedEmbeddingMethod` 1.

| resolved to EXL3 | n | | left BF16 | n |
|---|---|---|---|---|
| `mlp.experts` (routed) | 42 | | `model.embed_tokens` | 1 |
| `self_attn.o_proj` | 45 | | `mlp.gate` (router) | 42 |
| `mlp.gate_up_proj` (3 dense + 42 shared) | 45 | | `self_attn.f_b_proj` | 34 |
| `mlp.down_proj` (3 dense + 42 shared) | 45 | | `self_attn.g_b_proj` | 34 |
| **`self_attn.in_proj_qkv`** (KDA) | **34** | | **`self_attn.in_proj_bfg_a`** | **34** |
| `self_attn.q_b_proj` | 11 | | `self_attn.kv_b_proj` | 11 |
| `self_attn.fused_qkv_a_proj` | 11 | | `self_attn.indexer.wk_weights_proj` | 11 |
| `self_attn.indexer.wq_b` | 11 | | `q_conv1d` / `k_conv1d` / `v_conv1d` | 34 each |
| **`lm_head`** (6 bit) | **1** | | | |

On the control checkpoint the same dump gives EXL3 for the 42 routed-expert layers and nothing else,
which is what `scope: glm53_routed_experts_only` means.

### 4.4 The 113 that stayed BF16 — and why we are not going to quantize them

The right-hand column above is **113 linears per rank** and, once production 9 was serving, the
obvious next arm was to quantize them ourselves: the shapes are legal, the FP16 weights are already
inside the checkpoint, and a surgical pass over four families is hours rather than the 30–70 hours a
full requantization would take. **It was closed by measurement before any of that was spent** — and
then the measurement was re-taken on the target GPU the same night, **which left the closure standing
and destroyed the reason we had given for it** `[measured-here]`.

Read this section in that order: the author's decision, then the bench and what was wrong with it,
then what the corrected numbers say.

**First, the author's decision, which is in the source rather than inferred, and which no bench
touches.** `exllamav3` builds the five KDA gating projections with `qmap = None` — excluded from
quantization — and the comment gives the reason: the model author's own FP8 release lists those
families in `modules_to_not_convert`, which it reads as a sensitivity signal. `kv_b_proj` goes further
and is not a `Linear` at all: attention never applies it as that GEMM, its halves fold into the query
and output path at decode, so it is carried unquantized by construction. And the exclusion was applied
with judgement rather than copied — the same list also names KDA `qkv_proj` and `o_proj`, and those
**are** quantized at 6 bit in this checkpoint, with no measurable quality cost (§5). What is left BF16
is the subset the author decided was not worth the risk for the size.

**Second, the bench — model-free, on the real shapes, and measured twice.** `cuda-exl3` at `754421f`,
`exl3_linear` against `F.linear`, CUDA graphs on, at the TP=3 per-rank widths, `cb=2` (`mul1`, this
checkpoint's codebook). It was first run on a **workstation** GPU (101 MB L2, 170 SMs), median of 60
replays, and then re-run on **GB10** (24.0 MiB L2, 48 SMs) — the part production actually runs on —
with each shape timed twice: **warm**, the same weight every call, and **cold**, rotating both arms
over a bank of at least 4× L2.

The ruler was verified on both machines, and on the workstation it caught itself: a single-tensor read
of the `kv_b` shape reported **3,488 GB/s — 210 % of that machine's own peak** — so every arm there
ran against a **~300 MB weight bank**, at which the same shape reads 1,526 GB/s, 92 % of peak. **That
bank was still not enough**, and the reason is the whole of what follows: 300 MB is three times a
101 MB L2 for the *large* shapes and completely irrelevant to a **0.72 MB** KDA arm, which stayed
resident from the first replay to the last.

`ratio` is `exl3 / bf16`; **above 1 means EXL3 is slower**. At M=8, 4 bit `[measured-here]`:

| shape (TP=3 per rank) | k | n | workstation, as published | GB10 warm | **GB10 cold** |
|---|---:|---:|---:|---:|---:|
| `f_b_proj` | 128 | 2,816 | **1.596** | 1.605 | **1.023** |
| `g_b_proj` | 128 | 2,816 | **1.580** | 1.613 | **1.025** |
| `f_a_proj` / `g_a_proj` (replicated) | 4,096 | 128 | 1.171 | 1.270 | **0.853** |
| `in_proj_fg_a` (fused) | 4,096 | 256 | 1.060 | 1.139 | **0.655** |
| `kv_b_proj` (replicated) | 512 | 32,768 | **0.391** | 0.129 | **0.291** |

**The published column is a warm measurement, and GB10 proves it by reproducing it: 1.605 against
1.596.** Seven of the nine shapes in the full table reverse sign once both arms are rotated.
`[retracted]` — the withdrawn sentence, its two companions and the account of how it happened are in
[11](11-open-issues.md) §1.11 and §2.25; the corrected tables, the ruler, the trace referee and the
projection are in
[`../results/kernels/kda-gate-bench-gb10.md`](../results/kernels/kda-gate-bench-gb10.md).

**Which regime is real was decided by measurement, not by argument.** The same kernels' per-call costs
were already in the production-9 C1 trace, so the trace refereed the bench `[measured-here]`:

| family | trace µs/call | bench warm | bench cold | warm ÷ trace | **cold ÷ trace** |
|---|---:|---:|---:|---:|---:|
| `f_b_proj` + `g_b_proj` | 5.41 | 2.21 | 4.99 | 0.41 | **0.92** |
| `in_proj_bfg_a` | 14.20 | 6.52 | 15.51 | 0.46 | **1.09** |
| MLA A / `kv_b_proj` | 169.14 | 136.61 | 139.33 | 0.81 | **0.82** |

Cold reproduces the engine within ±20 %; warm is out by 2.2–2.4× on the small shapes. The mechanism
is plain: tens of GiB of weights stream through L2 per rank per decode step, so between two touches of
the same `f_b_proj` the whole active weight set has passed. **In production that tensor is never
resident.**

**Third, what the corrected arithmetic says — and it is a smaller change to the decision than to the
numbers.** Carried to the GB10 costs from the production-9 trace, against a 72.52 ms step:

| family | calls/step | ms/step now | cold ratio | projected | **Δ, + = faster** | as published |
|---|---:|---:|---:|---:|---:|---:|
| `f_b_proj` + `g_b_proj` | 68 | 0.368 | 1.023 | 0.376 | −0.008 | −0.218 |
| `in_proj_bfg_a` → split | 34 | 0.483 | 0.880 | 0.425 | +0.058 | −0.366 |
| **KDA gating arms** | 102 | **0.851** | 0.942 | 0.801 | **+0.050** | **−0.584** |
| `kv_b_proj` / MLA A | 11 | 1.860 | 0.291 | 0.542 | **+1.318** | +1.132 |
| **measured subtotal** | | **2.711** | 0.495 | 1.343 | **+1.368** | +0.547 |

The gate is `Δ ≥ 1.5 ms/step` at decode **and** `Δ ≤ +5 ms/chunk` at prefill. **The decode half now
fails narrowly rather than by an order of magnitude**, and applied to the whole **4.10 ms** the trace
attributes to the target's four unquantized families it would pass at **+2.07 ms** `[estimate]` —
which we do not claim, because 1.389 ms of that budget is MLA/DSA kernels the bench never touched.
**Prefill was not re-measured at all**, so the item is **re-scoped, not passed**.

**And the work list inverted rather than grew, which is why nothing is being built.** **96 %** of the
gain (+1.318 of +1.368) is `kv_b_proj` alone, and `cuda-exl3` at `754421f` still has **no per-head
batched EXL3 GEMM** (`exl3_linear` is a single `[M,k] → [M,n]`) and no `M`-threshold reconstruct path
— the mitigation `exllamav3` has (`AUTO_RECONSTRUCT_THRESHOLD = 144`) has no counterpart here. Both
re-verified by source scan. The one item worth doing is a **kernel** job. The KDA gating arms, the
families this section is named for, are not harmful and not useful: **+0.050 ms/step is 0.07 % of
C1**, which does not buy the multiplicative quality risk on `f_b_proj`'s decay term.

**The lever on those arms is not the bit width.** Cold, `f_b_proj` in bf16 reads 154 GB/s of a
235–240 GB/s peak; at 4 bit it reads **45.7 GB/s — 19 % of peak — and takes the same time**. It is not
bandwidth-bound in either format: it sits on a **~5 µs floor made of two dependent launches**,
`exl3_had_in_kernel` then `exl3_gemm_m_kernel`, where bf16 launches one. Fusing `had_in` into the GEMM
for narrow inputs is the thing to ask for, and it has been asked for `[not tested]`.

**One lever survived both passes, and it is not a quantization lever.** The workstation bench found
the MLA strided-batched family running in **fp32** — 11 calls per step, 0.757 ms — and bf16 measures
**0.684×**: **+0.24 ms/step, about +0.3 % of C1**, plus more than half of that family's prefill cost.
No requantization, no checkpoint change, no new kernel — a dtype. The cold re-measurement does not
touch it in direction or size. It is filed as future and minor, and gated on `needle` at 1M rather
than on speed, because this is the tensor that decodes the KV latent and an error in it touches the
whole of history `[not tested]`.

**Four transferable lessons, all about the instrument** — because none of this needed the engine, and
the two runs together cost about an hour of a workstation GPU and 90 s of one node's GPU against the
four to eight hours, two engine patches and blind quality risk they prevented:

1. **Bench a GEMM without a weight bank and it will read faster than the machine can physically
   fetch.** The number that gave it away was 210 % of peak; had the shapes been a little larger it
   would have read 95 % of peak and lied quietly.
2. **A bank sized against the wrong card, or the wrong shape, is the same mistake as no bank at all.**
   This is the one that cost us the sentence. 300 MB was a real bank for the large shapes on the
   machine it was written for, and no bank whatsoever for a 0.72 MB arm — which is precisely the arm
   the conclusion was about.
3. **The artefact's sign depends on which arm fits the cache, so it does not cancel in a ratio.** On a
   101–128 MiB L2 both the bf16 weight and the trellis fit, and EXL3 reads **slow**; on GB10's
   24 MiB only the trellis fits, and EXL3 reads **too fast** (0.129 warm against 0.291 cold). Only
   cold is honest on either card.
4. **The EXL3 prefill penalty is a property of shape, not of format.** A narrow input (k=128) makes
   EXL3 **faster** at M=1,792; a narrow output (n=128) makes it 3.5× slower. "EXL3 is expensive in
   prefill" is a sentence with a missing clause. This one is from the workstation prefill table, which
   **has not been re-measured cold** and is carried as it was `[not tested]`.

A fifth is not about the instrument: **"quantizing a small tensor makes it faster" is false whenever
the call was never bandwidth-bound** — the original bench said this and it is right, but its stated
reason (bytes that were never the cost) is only half of it. The cost is **launch count**, which is why
the remedy is a fusion.

Raw tables, the ruler checks and the projections: the workstation run in
[`../results/kernels/kda-gate-bench.md`](../results/kernels/kda-gate-bench.md), the target-GPU
re-measurement that supersedes its ratios in
[`../results/kernels/kda-gate-bench-gb10.md`](../results/kernels/kda-gate-bench-gb10.md) with the raw
logs beside it. All three scripts ship — `bench/ruler_check.py`, `bench/kda_gate_bench.py` and
`bench/kda_gate_bench_gb10.py` — and none of them needs a node or a checkpoint.
Tracked as closed in [11](11-open-issues.md) §2.25, retraction in §1.11.

---

## 5. Quality, at TP=2

The TP=3 production figure is in §7.3 (86.47 ±0.74). This is the dress rehearsal.

MMLU sample, 35 questions per subtask, 0-shot, concurrency 8, temperature 0, reasoning effort `low`
`[measured-here]`:

| arm | MMLU | CI | humanities | social sci. | other | STEM |
|---|---|---|---|---|---|---|
| **full-scope, TP=2** | **86.32 %** | ±0.75 | 86.81 | 89.05 | 85.27 | 84.96 |
| experts-only control, TP=2 | 86.4 % | ±0.7 | — | — | — | — |
| NVFP4 sibling stack, TP=3 (reference) | 86.7 % | — | — | — | — | — |

**The gate passes.** The difference is 0.08 points, about a tenth of either error bar. The one real
quality risk we had flagged was the **6-bit `lm_head`** at vocab 154,880, and it **cost nothing
measurable**. Second evidence in the same direction: correctness probe 10/10 and code exam 12/12,
cold and warm, on both arms.

The run was 7,980 requests in 26 minutes with no timeout. One piece of luck worth writing down: MMLU
0-shot prompts sit below the ~2,000-token admission ceiling this arm turned out to have (§6.1). That
was not calculated in advance, and on a benchmark with longer prompts the same arm would have
produced nothing.

**The control's 86.4 ±0.7 was not re-run in this arm** and it is the same figure the rest of this
repository quotes: measured earlier the same day, same checkpoint, same TP=2, but with the MTP
drafter rather than DFlash2. A log-likelihood task does not go through the speculative decoder at
all, so re-running it would have spent 20 minutes reproducing a known number. Say it plainly rather
than let the table imply two fresh runs `[measured-here]`.

---

## 6. What it cost, at TP=2

The production ledger is §7.4. This section is the rehearsal's, and two of its three big items
did not reproduce at three ranks (§7.5) — they are kept because a reading that did not survive is
still a reading, and because §6.3 and §6.4 are about instruments rather than about TP=2.

### 6.1 Context — the price, and it is not small

At TP=2 the full-scope model leaves so little KV that **the long-prompt path closes entirely**. The
engine's own log explains the mechanism: `Mamba cache mode is set to 'align'` →
`Setting attention block size to 4608 tokens to ensure that attention page size is >= mamba page
size`, and the pool comes out at **31,343 tokens — about 6.8 pages**. With six or seven pages, a long
prompt cannot be scheduled at all. Measured admission `[measured-here]`:

| prompt | result |
|---|---|
| 844 tokens | served, 1.1 s |
| 1,684 tokens | served, 1.7 s |
| ~2,800 tokens | **never scheduled** — 45 s later still `Running: 0, Waiting: 1`, KV usage 0 % |
| 7,382 tokens (`prefill-7k`) | **never scheduled**; the client gave up at 600 s |
| 8,204 tokens (`prefill-fresh`) | not run — same fate, and three 900-second timeouts to prove it |

That is why every prefill number and the C6/C8 aggregates are void in this arm, and it is also why
`--block-size 256` has no effect here: the hybrid allocator overrides it.

This is **the known context cost of two nodes made worse**, not a defect of full scope: our own TP=2
control pool is 665,625 tokens against production TP=3's 4,699,724 — **14 %** — because a 164 GiB
model over two nodes leaves ~85 GiB of weights per node and very little else. Full scope makes it
sharply worse rather than better (§6.2). **TP=2 is a measurement rig here, not a serving
configuration**, and that was true before this arm.

### 6.2 The memory reading we cannot explain

Two measurements about the same arm point in opposite directions, and both are left standing:

- At `max_model_len=1,000,000` the control arm had **4.4 GiB** of KV memory available and the
  full-scope arm **0.73 GiB** — the full-scope boot failed its own budget gate, needing 6.6 GiB.
  Consumed per node reads **95.5 GiB** against the control's ~85 GiB, i.e. **~10 GiB heavier**
  `[measured-here]`.
- Yet the checkpoint is 10 GiB **smaller** on disk, and post-run free host memory says its weights
  are ~3.3 GiB per node **lighter** `[measured-here]`.

Taken together that implies roughly **+15 GiB of non-weight allocation**, and the leading suspect is
`Exl3LinearMethod.process_weights_after_loading` calling `ops.reserve(...)` per EXL3 module: the
count of EXL3 dense linears per rank goes from **1 to 203**, and `lm_head` alone has `n_total`
51,712. The word "shared workspace" in that code implies a maximum rather than a sum, which would
make the hypothesis wrong — **it has not been measured** `[not tested]`. It was the single number to
settle before TP=3, and at TP=3 the **sign reversed**: 3.4 GiB *lighter* per node and the pool 10 %
*larger* (§7.3). The mechanism is still not isolated; the TP=2 reading is marked **not reproduced**
rather than explained (§7.5).

**A separate finding fell out of the same investigation.** Lowering `max_model_len` from 1,000,000 to
65,536 raised available KV memory from 0.73 to **5.41 GiB**: about **4.7 GiB per node of persistent,
non-KV allocation scales with `max_model_len`** on this stack `[measured-here]`. That buffer is paid
in production too, and it explains why the control arm at 1M could only find ~4.4 GiB for KV.

### 6.3 The acceptance test we designed did not work

The planned boot gate was `CUDA_EXL3_DEBUG_NAMES=1`, reading off which modules stayed unquantized. It
printed **nothing**. The flag reached the container (verified with `docker inspect`) and the code was
right; the plugin logs that line at `logger.info`, and this image configures only vLLM's own logger,
so third-party INFO goes nowhere. The same is true of the plugin's `EXL3: N quantized tensors` line,
which we had already caught being invisible earlier the same day and correctly refused to reason from
([11](11-open-issues.md) §1.9).

Three substitutes carried the gate instead, and they are the ones to reuse `[measured-here]`:

1. **Zero orphaned tensors.** With 36,719 `.trellis` in the checkpoint, any module that failed to
   resolve would leave its trellis with no parameter and stop the boot. It did not stop.
2. **Asserts 1 and 2 silent** — `in_proj_qkv` really was built as three EXL3 shards whose `suh`
   collapsed to one group. A module that had fallen back to BF16 would have failed assert 1.
3. **The two-rank meta-device dump** (§4.3), taken before the boot, with the engine down.

We reported the logging bug; the author fixed it at `warning` level and made it log the **hits** as
well as the misses, with running tallies, so a mapping that leaves half the attention stack in BF16
shows up as a climbing BF16 count (`807d798`).

### 6.4 A fast-load sidecar refused the production restore, and it was right

Restoring production after the arm failed on all three nodes with `Exited (21)`:

```text
preflight-fastload: sidecar stale - boot refused
  patches.patch-fullscope-tp2.py: recorded='<none>' now='bca9a201...'
  patches.tp3-prelude.sh:         recorded='e17d46a4...' now='5eba79f3...'
```

The sidecar records the identity of the patch set it was dumped against, including the `sha256` of
**every** `patch-*.py` in the patch directory and of the prelude ([08](08-fast-boot.md) §4). During
the arm the full-scope patch had been placed in the TP=3 patch directory and an env-gated hook added
to the TP=3 prelude. Neither changes production behaviour — the hook does nothing with the flag unset
and the patch is never called at TP=3 — but both change the **identity**, and a sidecar is
pre-processed weights belonging to a specific set of code. Using it silently could have served wrong
answers. **The refusal is the design working.**

The fix was to restore the prelude from its backup and unlink the patch from the TP=3 directory (the
file survives through its hard link in the TP=2 directory), then reboot: production came back in
264 s with a pool of 4,696,969 against 4,699,724 before the arm, 0.06 % `[measured-here]`.

**The rule this produced is in [09](09-measurement-protocol.md) §11.2:** between a dump and a load,
nothing is added to, removed from or edited in the patch directory — experimental patches live in
their own directory and are hard-linked, not copied, if two places need them.

### 6.5 The rest of the ledger

- Production was down **1 h 15 m**, planned, with the third node idle throughout.
- `max_model_len` was lowered to 65,536 **for the experiment only**, in each node's own env file with
  a backup; the production env files were not touched.
- Two nodes hold a second 154 GiB checkpoint and sit at ~95 % full. Nothing was deleted.
- No fast-load sidecar could be built for this arm (52–56 GB per node, no room), so both arms boot
  in 355–396 s rather than 274 s. Both arms are equally affected, so the comparison is fair.

---

## 7. TP=3: the port, and what it measured

**Done, promoted, in production since 5 September 18:40.** This section was a plan until that
afternoon; it is now an account. The projection it carried — "applying the measured per-stream ratio
projects roughly 74–75 tok/s" `[estimate]` — measured **75.91 tok/s per stream**, which is inside its
own band and is recorded here because a projection that lands is as much a check on the model behind
it as one that misses.

**Settings, both arms identical unless the row says otherwise:** three DGX Spark nodes, TP=3 +
expert parallel, DFlash2 draft k=7, KV `fp8` **and** an fp8 draft cache,
`gpu-memory-utilization 0.80`, `max_model_len 1,000,000`, `--block-size 256`,
`HAREM_SW_BLOCK_SIZE=256`, `--max-num-batched-tokens 2048`, `--max-num-seqs 8`,
`NCCL_MAX_NCHANNELS=8`, warm tuner cache, per-rank fast-load sidecar in **both** arms, the launcher's
settle gate, temperature 0, reasoning effort `low`, medians of three sweep rounds, 5 September 2026.
Full scope is `turboderp/GLM-5.3-Flash-exl3` at 4.05 bpw on image `exl3-zeus:754421f`; the control is
production configuration 8 — the experts-only checkpoint on `exl3-zeus:62f53e6`. **The control column
is the pool of two runs of the same script on the same day** (14:45 and 16:44, six rounds), because
that arm's documented run-to-run spread is about 7 % and a single run of it would have flattered
whichever side it favoured.

### 7.1 What the port needed on our side

Three things, and only one of them was code.

**Two launcher constants**, both `lcm(64, tp)` where they had to be `lcm(128, tp)`, because a
full-scope checkpoint quantizes the vocabulary head and the shared expert and every EXL3 pad has to
be a whole number of 128-column Hadamard blocks. Vocab `padding_size` 192 → **384** (154,880 →
155,136 = 3 × 404 × 128) and shared-expert intermediate 2,112 → **2,304** (768 = 6 × 128 per rank).
The arithmetic, including why 2,176 looks right and is not, is
[03](03-tp3-padding-and-sidecars.md) §1.1. Both are no-ops at TP≤2 and both live in
`tracks/tp3/patches/patch-vllm-tp3.py`, with matching gates in `preflight-tp3.py`.

**A9 — one patch, and it is TP=3 only.** vLLM's tuple-shard path (`(0, 1, 2)`) slices the
checkpoint's single 24,576-wide `qkv_proj` using offsets built from the module's `output_sizes` —
which at TP=3 are **padded**, 3 × 8,448 against a checkpoint that is still 3 × 8,192:

```text
segment 0: narrow(start=0,     len=8448) -> 8448    correct
segment 1: narrow(start=8448,  len=8448) -> 16896   wrong (true start 8192)
segment 2: narrow(start=16896, len=8448) -> 25344   overflow -> RuntimeError
```

The fix records the checkpoint's real per-shard width on the module when A7 builds it
(`quant_config.resolve(...)` gives `out_features / 3 = 8192`), splits by that, and lets the existing
zero-pad path widen each rank's slice afterwards. Three asserts: the lengths match, the three widths
are equal, and `pad % 128 == 0` with `sum(ckpt) + sum(pad) == sum(output_sizes)`. At TP=2 the two
lists are equal and A9 is a no-op. Measured on the meta device, all three ranks:
`output_sizes=[8448,8448,8448] ckpt=[8192,8192,8192] exl3_shards=[2816,2816,2816]`.

**A10 — the audit, because the invariant was already holding by accident.** The `svh = 0` mechanism
was in fact **already running** on this stack for column-parallel EXL3 modules before anyone designed
it: the TP=3 padding patch fills missing rows with zeros and `svh` goes through the same path. So the
padded load was already correct, and **nothing checked it** — which is precisely how the old 2,112
arithmetic would have produced silently wrong output rather than an error. A10 wraps `load_weights`
and, while `suh` and `svh` are still raw, walks every EXL3 module: for an output pad, `real % 128`,
`pad % 128`, `real > 0` and `svh[real:] == 0`; for an input pad — row-parallel only, decided from the
module's own geometry, `exl3_k * tp == input_size` — the same gates and `suh[0, real:] == 0`. One
line comes out, and **its absence is a failure**:

```text
HAREM-FULLSCOPE assert 5: 285 EXL3 pad site(s) audited, 285 padded on this rank, all whole 128-blocks and exactly zero
```

285 is what a model-free meta-device run predicted for rank 2 **before** the boot, from the pad table
below — so the reading agrees with an independent count rather than merely being self-consistent
`[measured-here]`.

| n | module | axis | real | local | pad | real/128 | pad/128 |
|---|---|---|---|---|---|---|---|
| 1 | `lm_head` | out | 51,456 | 51,712 | 256 | 402 | 2 |
| 42 | `mlp.shared_experts.down_proj` | **in** | 512 | 768 | 256 | 4 | 2 |
| 84 | `mlp.shared_experts.gate_up_proj[0,1]` | out | 512 | 768 | 256 | 4 | 2 |
| 102 | `self_attn.in_proj_qkv[0..2]` (KDA) | out | 2,560 | 2,816 | 256 | 20 | 2 |
| 34 | `self_attn.o_proj` (KDA) | **in** | 2,560 | 2,816 | 256 | 20 | 2 |
| 11 | `self_attn.o_proj` (MLA) | **in** | 5,120 | 5,632 | 512 | 40 | 4 |
| 11 | `self_attn.q_b_proj` | out | 5,120 | 5,632 | 512 | 40 | 4 |

**The audit caught its own bug on first writing, and that is the reusable part.** It called a
column-parallel module's *input* padded (`k_real != k_local * tp`) and rejected `in_proj_qkv` on
ranks 1 and 2. Rank 0 passed — **a single-rank test could not have seen it**
([09](09-measurement-protocol.md) §11.3).

**And two things that were not code at all:** the sidecar generator had to grow a rewritten
`quantization_config.json` (the packed mapping travelling with the checkpoint — a 48 MB copy, because
`cuda-exl3` needs `tensor_storage` out of the same dict; [03](03-tp3-padding-and-sidecars.md) §2),
and the whole tree had to move into `tracks/tp3/patches/` rather than into the production patch
directory, because that directory's file list is the fast-load manifest identity and adding one file
to it refuses the next production boot (§6.4, [09](09-measurement-protocol.md) §11.2).

### 7.2 What the plugin author built, and the image gate that checks for it

Three things were asked for in the issue thread, and all three were built the same afternoon once the
TP=2 quality gate passed. **All three were required; no two of them are enough.**

| commit | what it does | what fails without it |
|---|---|---|
| `f3e3090` | accepts a padded **output** dim when the pad is whole 128-blocks, allocating `svh` zeroed; and makes the row-parallel `suh` load copy what exists and zero the rest instead of narrowing past the end of the checkpoint | `create_weights` raises "EXL3 weights cannot be zero-extended" |
| `754421f` | the vocab loaders fill a **prefix** instead of `copy_`-ing an unpadded slice into a padded parameter | the gate passes and the load then dies on a `copy_` shape mismatch in `_vocab_loaders` (rank 2: 3,232 against 3,216 tiles) |
| `807d798` | makes `CUDA_EXL3_DEBUG_NAMES` print at all, and report the modules that **resolved** as well as those that stayed BF16 | the designed acceptance gate prints nothing and looks exactly like a clean run (§6.3) |

A fourth question was answered rather than built, and completely: the author swept **all 65,536**
16-bit trellis codes through the device decoder for all three codebooks — zero non-finite values,
bounded ranges (`3inst` [−3.9570, +3.9727], `mcg` [−3.9492, +3.9492], `mul1` [−3.4531, +3.3477])
`[reported]`. The decoder is total and bounded, so `svh = 0` on a pad column is exactly
`0 × finite = 0`. That is the full domain enumerated rather than an argument from construction, which
is the right standard underneath a padded-load path.

Both failure modes above are loud, and both leave a **half-loaded stack**, so the gate goes in front
of them rather than behind. `tracks/tp3/patches/check-padload-tp3.py` runs in the prelude, reads all
three capabilities out of the installed source by inspection (there is no version string to trust,
because the image is built from a git checkout), prints one line and exits **23** if any is missing —
before a byte of weight is read:

```text
[padload] cuda-exl3 padded-load support: padded-output-gate (f3e3090)=yes  vocab-loader-prefix (754421f)=yes  row-parallel-suh-pad (f3e3090)=yes
```

### 7.3 The result

Medians of three rounds, against the pooled production-8 control. Treating ≤3 % as equal
([09](09-measurement-protocol.md) §1.2) `[measured-here]`:

| metric | **full scope (production 9)** | experts-only (production 8) | delta | verdict |
|---|---|---|---|---|
| **C1 total** | **69.90** tok/s | 56.88 | **+22.9 %** | faster |
| **C1 per stream** | **75.91** tok/s | 62.39 | **+21.7 %** | faster |
| C2 total / per stream | 99.17 / 54.03 | 83.31 / 49.45 | +19.0 / +9.3 % | faster |
| C4 total / per stream | 140.72 / 42.61 | 120.22 / 37.25 | +17.1 / +14.4 % | faster |
| C6 total / per stream | 172.40 / 33.04 | 144.03 / 30.59 | +19.7 / +8.0 % | faster |
| **C8 total** | **197.20** tok/s | 175.37 | **+12.5 %** | faster |
| C8 per stream | 28.64 | 26.71 | +7.2 % | faster |
| TTFT at C1 / C8 | **0.280 / 0.826** s | 0.344 / 0.906 | −18.6 / −8.8 % | faster |
| prefill, fresh unseen | 1,738 tok/s | 1,776 | −2.2 % | **equal** |
| prefill, 7K warm repeat | 1,575 tok/s | 1,537 | +2.5 % | **equal** |
| draft acceptance at C1 | 61.94 % | 64.36 % | **−2.4 points** | lower, gate ≥60 % passed |
| draft acceptance at C8 | 62.59 % | 61.74 % | +1.4 % | equal |
| accepted tokens per step at C1 | 5.34 | 5.50 | −3.0 % | lower |
| consumed memory per rank | **58.28 / 59.08 / 59.01** GiB | 62.08 / 62.32 / 62.39 | **−3.4 GiB** | lighter |
| **KV pool** | **5,165,289** tokens (5.17× at 1M) | 4,696,969 (4.70×) | **+10.0 %** | larger |
| peak activation, free memory at startup | 1.66 GiB, 111.4 / 113.1 / 113.0 | 1.66 GiB, 111.9 / 113.0 / 113.1 | — | equal |
| boot, fast-load | **251 s** (weights 57.9 s) | 264 s (weights 73.2 s) | −5 % | faster |
| free host RAM / swap after the run | 12.1 / 13.5 / 13.4 GiB · 0.11 / 0.09 / 0.08 | 12.3 / 13.5 / 13.5 · same | — | equal, flat |
| gates cold and warm | **10/10 · 12/12** | 10/10 · 12/12 | — | pass |
| MMLU sample (57 × 35, 0-shot) | **86.47 ±0.74** | 86.4 ±0.7 | 0.07 points | pass |

All three full-scope rounds, for the record:

| round | C1 total / per stream | C2 | C4 | C6 | C8 | acceptance at C1 |
|---|---|---|---|---|---|---|
| 1 | 69.57 / 72.21 | 98.49 | 144.02 | 171.93 | 198.95 | 61.44 % |
| 2 | 69.90 / 78.37 | 100.01 | 140.72 | 173.35 | 190.30 | 61.94 % |
| 3 | 70.29 / 75.91 | 99.17 | 137.06 | 172.40 | 197.20 | 62.74 % |

**The step arithmetic is the result, not the tok/s:**

```text
production 8: 5.50 tokens/step ÷ 62.39 tok/s per stream = 88.2 ms/step
production 9: 5.34 tokens/step ÷ 75.91 tok/s per stream = 70.3 ms/step
gain        : 17.8 ms/step  (step -20.3 %, throughput +21.7 %)
```

The gain is **entirely** step time. Acceptance and tokens per step move the *wrong* way by about 3 %
and the arm still wins by 22 %, so none of it is drafter behaviour. This is the same lever measured
at TP=2 (20.7 ms/step, +24.3 %) reproduced at TP=3 to within a millisecond and a half: **the
dense-GEMM hypothesis is now confirmed on two independent topologies.**

A second, cleaner read on the same engine. `cold-warm-c1` runs one 700-token prompt cold, warm and
warm again — same prompt, same script, both arms, an hour apart `[measured-here]`:

| | full scope | production 8 (two runs) | delta |
|---|---|---|---|
| cold | **57.2** tok/s | 42.9 / 43.9 | **+33.3 %** |
| warm | **56.5** | 44.3 / 45.7 | +27.5 % |
| warm 2 | **57.5** | 44.8 / 43.3 | +28.3 % |
| TTFT cold | 1.36 s | 1.38 / 1.40 | equal |
| acceptance | 42.5 / 42.2 / 42.9 % | 40.4–43.9 % | **equal** |

**This settles the TP=2 acceptance scare** (§4.1, [11](11-open-issues.md) §1.9 row 30). At TP=2 the
same script showed acceptance falling 48 % → 34 % and it was reported upstream as "a new and real
gate". At TP=3 it is flat on this script, and the 2.4 points the sweep's C1 median showed turned out
to be the sweep's own prompt rotation rather than the checkpoint (§7.4). Speed is up 27–33 %.

### 7.4 What it cost — and the two entries we had to withdraw

No gain is published here without its price, and this one had six. **Two of them were not real, and
finding that out cost nothing but re-reading data we already had** `[retracted]`. The C1 median does
read 61.9 % against production 8's 64.4 %, but `bench-sweep.py` cycles `prompts[i % 12]`, so C1 and C2
see only the first **eight** of the twelve prompts while C4–C8 see all twelve, and the two groups
differ by about eight points of acceptance. Pooled by draft token across five concurrency levels and
three independent boots, production 9 reads **62.27 %** against production 8's **62.09 %** — **+0.18
points**, inside that arm's own ±1.4-point boot spread, with the sign reversing at C6 and the cold
probe identical on both arms (42.53 % against 42.51 %). And because `accept_len = 1 + k × acceptance`
holds on all 90 rows to ±0.005, the second entry was never a second cost. Net effect on throughput:
**+0.24 %** ([11](11-open-issues.md) §2.26).

The four costs that remain are the four that were never about the drafter.

| cost | size | note |
|---|---|---|
| draft acceptance | **none** `[retracted]` | we published −2.4 points here and it was our harness. See below |
| accepted tokens per step | **none** `[retracted]` | never a second cost: `accept_len = 1 + k × acceptance` holds on all 90 rows, so this was the same number written twice |
| prefill, fresh | −2.2 % | inside the ±3 % equality band; not a real loss, and prefill-7k moved +2.5 % the other way |
| maintenance | **a second patch tree**, `tracks/tp3/patches/`, whose `patch-vllm-tp3.py` and `preflight-tp3.py` diverge from `patches/tp3/` on purpose | a fix in one does not reach the other. Merging them is technically possible today — the constants derive from `tp` and 128 and are no-ops at TP≤2 — and was not done, so production 8's fast-load identity would keep working. [11](11-open-issues.md) §2.24 |
| disk | **53 GB × 3** for the second fast-load sidecar, on top of a second 154 GiB checkpoint × 3 | one node had 51 G free before this and needed the old sidecars cleared first |
| quality | **none found, and it was looked for** | MMLU 86.47 ±0.74 against 86.4 ±0.7 — 0.07 points, a tenth of either bar — plus both gates cold and warm on the same engine instance in one session |

Not on the list, because they were expected and did not happen: the **image is unchanged by the
patch** (it is applied inside the container at boot, to the writable layer, and leaves no trace when
the container is removed), and `MAX_MODEL_LEN=1,000,000` was never threatened.

### 7.5 Two TP=2 signals that did not reproduce, and the honest version of why

The dress rehearsal produced two warnings loud enough to be written into this page as risks. Neither
survived TP=3, and saying *why* matters more than saying *that*.

**"~10 GiB heavier per node", which forced TP=2 down to a 31k pool (§6.2).** At TP=3 the sign is
reversed: 3.4 GiB **lighter**, KV pool +10.0 %. We did not isolate the mechanism, and two things
differ. (a) At TP=2 the two arms ran on different checkpoints *and* different `max_model_len` values,
and that same report found free KV scaling with `max_model_len` (0.73 → 5.41 GiB going 1M → 64k), so
the "10 GiB" figure was measured across a confounded pair; here both arms are at 1M. (b) The leading
suspect, `Exl3LinearMethod.process_weights_after_loading`'s `ops.reserve`, scales with EXL3 **shard**
size, and at TP=3 every shard is a third rather than a half. We report the TP=3 numbers as measured
and mark the TP=2 delta **not reproduced**, not explained `[retracted]`.

**"Draft acceptance collapses on a quantized target."** That came from a cold single-prompt probe
with a sample of one, and it was reported upstream before the sweep ran. It is flat on that same
probe at TP=3, and the sweep's 2.4 points went the same way as the rest of that scare — withdrawn
(§7.4, [11](11-open-issues.md) §1.9 row 30, §2.26).

**Also measured, and useful for planning:** the dump boot (no fast-load) consumed 60.44 / 60.54 /
61.35 GiB and gave 4,840,220 KV tokens, so **fast-load itself is worth about 2 GiB and ~325k KV
tokens** on this arm. Both arms in the table above are load boots, so the comparison is like for like
([09](09-measurement-protocol.md) §11.1).

### 7.6 The acceptance list, as it was actually run

Nineteen points were written before the arm and every one of them was taken. The order matters more
than the list: everything model-free comes before the boot, every gate comes before a speed number,
and the pool is read from a load boot. It is reproduced in
[09](09-measurement-protocol.md) §5.1 as the general form.

If a gate had failed, the suspects in order were: the A9 split width (assert 6), rank 2's pad columns
(assert 5), the row-parallel `suh` narrowing on `o_proj`, the 6-bit `lm_head`, and the `q`/`k`/`v`
order of the `conv1d` split (assert 3).

### 7.7 Rollback

Three levels, none of which needs an image rebuild:

- **One line.** Delete `HAREM_EXL3_FULLSCOPE=1` from `EXTRA_ENV`. The patch reads the knob at run
  time, so the same patched image takes the upstream path.
- **The whole arm.** Start with `ENV_FILE` pointing at the production 8 env file — ours is
  `.env.tp3.bak-prod8-62f53e6`. That reverts the checkpoint, the image (`62f53e6`), the patch tree
  (`tp3/`) and the fast-load sidecar in one move, because all four are named in the env file.
  `patches/tp3/`, `.env.tp3`'s backup and production 8's sidecar were never modified during the arm.
- **The safety property underneath both.** The split decision is taken from the **checkpoint**, not
  from the environment (`quant_config.resolve(f"{prefix}.in_proj_qkv") is not None`), so booting the
  experts-only checkpoint with the flag *set* gives split inactive, 0 unmapped, 0 unfilled,
  everything BF16 `[measured-here]`. A wrong env file cannot silently corrupt a run.

## 8. What this changes

- **The largest item this stack carried is closed, and it is in production.** The dense stage is
  worth **+21.7 % per stream at TP=3** — 17.8 ms of an 88.2 ms decode step — with **no quality cost**,
  a **larger** KV pool and a **faster** boot, and the acceptance cost we billed it for was ours
  ([11](11-open-issues.md) §2.22).
- **`routed_experts_only` was never a quality decision.** Two lines in a model file made it the only
  loadable scope, and they would have done so for any checkpoint in any format.
- **The KDA factorisation mismatch is not an EXL3 problem.** `conv1d` is BF16; a BF16 copy of this
  checkpoint fails identically. Anyone loading an upstream-layout GLM-5.3-Flash checkpoint on the
  NVIDIA `glm5next` reader will hit it, whatever the quantization.
- **A quantized tensor can live in a padded parameter.** Not by zero-extension, which is meaningless
  for a trellis, but by a narrow load with the output scale zeroed on the pad — and only on whole
  128-column blocks. It is now checked at every load rather than holding by accident (§7.1).
- **TP=2 was a measurement rig and TP=3 is the serving configuration**, and the two disagreed about
  memory in a way we could not explain. Both readings are still on this page (§6.2, §7.5): the TP=2
  one is marked not reproduced rather than explained.
- **Two signals reported upstream before the sweep ran were wrong**, and both are in the retraction
  audit: the acceptance collapse (row 30) and the aggregate-versus-per-stream transcription (row 31).
  A number that leaves this repository is a published number whatever produced it.

---

## 9. What is next

[10 — Results and roofline](10-results-and-roofline.md) §1 for production 9 in the results
progression, [11 — Open issues](11-open-issues.md) §2.22 and §2.24–§2.25 for what this opened,
[03](03-tp3-padding-and-sidecars.md) §1.1 for the padding arithmetic, and
[`tracks/tp3/patches/README.md`](../tracks/tp3/patches/README.md) for the directory itself.
