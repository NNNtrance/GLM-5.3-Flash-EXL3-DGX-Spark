# 19 — Vision at two ranks: the tower divides, and the patch it needs is not the one that divides it

**Applies to: TP=2 only.** The three-node account is [18](18-vision-at-three-ranks.md); this page is
the two-node one, and it exists because two of the things [18](18-vision-at-three-ranks.md) §11 and
[15](15-tp2-track.md) §5b predicted about two ranks turned out to be right in a way that did not
help, and one thing neither page saw at all.

**In the two-node recipe as of 8 September 2026.** The GLM-5.3-Flash vision tower serves
**4 images + 2 videos per request** on two DGX Spark nodes at TP=2 on the full-scope EXL3 checkpoint,
alongside the two backports promoted at three ranks the same night. [15](15-tp2-track.md) §5.10 is
the configuration and its cost; this page is why it took the shape it did.

---

## 1. What both earlier pages predicted, and what it was worth

Both pages said the same thing about two ranks, and both were **correct**:

> At two ranks the tower divides and needs no padding and no data-parallel flag. Heads 16/2 = 8,
> `attn.proj` 512 = 4 × 128, the MLP 2048 = 16 × 128, the merger 2048 = 16 × 128 and its context
> 5120 = 40 × 128.

Every one of those numbers is right. **And it is not the part that mattered**, for two reasons.

**First, divisibility answers a question we chose not to ask.** Dividing the tower means *slicing a
6-bit EXL3 tower across ranks* — trellis, `suh`, `svh` and `mul1`, through `cuda-exl3`'s column- and
row-parallel paths. That path works for the language model at two ranks because we measured it there;
for the **tower** nothing in this repository has measured it at any rank count, because at three
ranks the tower cannot be sliced at all and `--mm-encoder-tp-mode data` was the only road. We shipped
`data` at two ranks as well: it **replicates** the 0.557 GiB tower on each rank and reuses the exact
path that is in three-node production, at a cost small enough to be inside the boot-to-boot spread.
The sliced tower stays `[not tested]`, and it is a real open item — see §6.

**Second, the loader half was never about the rank count and it is where all the work is.** VS1, VS2
and VS3 exist because the tower in `turboderp/GLM-5.3-Flash-exl3` is 6-bit EXL3 and vLLM builds it
with `quant_config=None` whatever the parallel size; VS4, VS6 and VS7 exist because upstream's
GLM-5-Next processing info inherits GLM-4.6V's frame sampler
([vllm#55644](https://github.com/vllm-project/vllm/issues/55644)). Both pages said so. What neither
page said is what §2 is about.

---

## 2. The finding: the vision patch needs a file the two-node tree deliberately did not ship

`patch-vision-tp3.py`'s **VS1** anchor is not written against the stock `model.py`. It is written
against the text `patch-vllm-tp3.py`'s **edit 4b** leaves behind — the wrapper that makes
`--language-model-only` actually stop the tower being built:

```
        self.visual = Glm5NextVisionTransformer(     ->    self.visual = _harem_build_vision_tower(  # HAREM-TP3
            config.text_config,                                 Glm5NextVisionTransformer,
                                                                multimodal_config,
                                                                config.text_config,
```

That re-anchoring is itself a scar from three ranks ([18](18-vision-at-three-ranks.md) §13): a
stock-file anchor matched zero times at boot and stopped all three ranks with exit 21. At **two**
ranks it produces a different and sharper problem, because [15](15-tp2-track.md) §2.3 lists
`patch-vllm-tp3.py` under *what is deliberately not shipped* — its padding half is a no-op by
arithmetic at TP≤2 and this repository does not ship text it has not measured.

So the two-node tree could not run the vision patch at all, and there were two ways out:

| | |
|---|---|
| **Fork the vision patch** — add a stock-text fallback anchor to `patch-vision-tp3.py` | **Rejected.** That file is in three-node *production*: its sha256 is printed in the boot log and its content is hashed into the production fast-load sidecar's identity. Editing it to serve two ranks invalidates a running three-node configuration, which is a real cost paid for a cosmetic tree property |
| **Ship the same file in both trees** | **Taken.** `patch-vllm-tp3.py` is now in the two-node tree, byte-identical with the three-node one |

**What that costs at two ranks, stated rather than assumed.** The file has four edits. Edits 1 and 2
(`_harem_pad_then_narrow`) only fire when a config pad puts a rank's shard past the stored dimension,
and at TP=2 no shape is padded, so the branch is unreachable. Edit 3 raises `padding_size` from 64 to
`lcm(128, 2) = 128`; the vocabulary is 154,880, a multiple of both, so the padded vocabulary is
identical either way. Edit 4's shared-expert rounding is gated on
`intermediate_size % tp != 0 or (intermediate_size // tp) % 128 != 0`, and 2,048/2 = 1,024 = 8 × 128
satisfies both, so it does not fire. **Edits 4a, 4b and 4c are the ones that do something**, and all
three are the `--language-model-only` wrapper: build the tower or do not, and skip the `visual.`
prefix when it was not built. That is behaviour the two-node track wanted anyway — without it,
`LANGUAGE_MODEL_ONLY=1` at two ranks builds and loads the tower regardless (which is what
[18](18-vision-at-three-ranks.md) §1.2 says upstream does, and it is not a rank-count property).

**The tree is therefore fifteen files rather than fourteen** before the vision and backport sets are
added. [15](15-tp2-track.md) §2.3 and [`tracks/tp2/patches/README.md`](../tracks/tp2/patches/README.md)
carry the corrected inventory.

---

## 3. The drafter directory: a TP=3 pad written in place, and a two-node boot that stops on it

The first dump boot of this arm reached the drafter and died:

```
File ".../vllm/model_executor/models/qwen3_dflash.py", line 186, in __init__
    assert self.total_num_kv_heads % tp_size == 0
AssertionError
```

**This is not a two-node defect and it is not upstream's.** The DFlash2 drafter's own GQA is 32/8,
which divides by two, and [04](04-dflash2-port.md) says so in its first line. What had happened is
that the TP=3 track's 32/8 → 36/9 pad was applied by **rewriting `config.json` inside the shared
drafter directory**, with the original kept beside it as `config.json.orig`. Nine key-value heads do
not divide by two, so a two-node reader who points `DRAFT_HOST_PATH` at "the drafter" gets a padded
config and an assertion, four minutes into a boot, after the target model has already been read.

**The fix is a directory, not a patch.** Give the two-node track its own drafter directory holding
the *unpadded* config and a **hard link** to the same `model.safetensors` — the weights are identical,
only the config differs, and a hard link costs nothing on disk:

```
install -d /var/tmp/dflash2-draft-tp2
cp /var/tmp/dflash2-draft/config.json.orig /var/tmp/dflash2-draft-tp2/config.json
ln /var/tmp/dflash2-draft/model.safetensors /var/tmp/dflash2-draft-tp2/model.safetensors
```

`scripts/start-tp2full.sh` has always defaulted `DRAFT_HOST_PATH` to `/var/tmp/dflash2-draft-tp2`;
what was missing was the sentence saying **why that directory has to exist separately**, and a
troubleshooting entry naming the assertion. Both are now in
[14](14-troubleshooting.md) §10.5 and [15](15-tp2-track.md) §2.1 `[measured-here]`.

**It costs a dump boot if you find it late.** The fast-load sidecar's identity hashes the drafter's
`config.json`, so correcting the config after a dump invalidates the sidecar. Fix the directory
*before* the dump boot.

---

## 4. What is the same as three ranks, and can be read there

Nothing in this section is repeated here, because none of it changed:

| | Where |
|---|---|
| Why the tower was off, and why `--language-model-only` never stopped it being built | [18](18-vision-at-three-ranks.md) §1 |
| VS1/VS2/VS3 — the 6-bit EXL3 tower, the `_proj` spellings, the dead fused `attn.qkv` residue and the post-load audit | [18](18-vision-at-three-ranks.md) §3 |
| VS4/VS6/VS7 — the two frame samplers, the dead engine core, the `min(user, supported)` clamp; [vllm#55644](https://github.com/vllm-project/vllm/issues/55644) and its PR | [18](18-vision-at-three-ranks.md) §4, §12 |
| The brake: `max_image_tokens` reaches the video processor and `max_pixels` does not | [18](18-vision-at-three-ranks.md) §5 |
| `CUDA_EXL3_PACKED_MAPPING` and why it is an environment value rather than a patch | [18](18-vision-at-three-ranks.md) §3.2 |
| `--mm-processor-cache-gb 0` and the 4 GiB of host memory it gives back on the API rank | [18](18-vision-at-three-ranks.md) §9 |
| The three model-free gates and what each one catches | [`tracks/tp3/patches/vision/README.md`](../tracks/tp3/patches/vision/README.md) |

**The patch files are the three-node track's, unchanged.** There is no two-rank copy of
`patch-vision-tp3.py` and there should not be one:
[`tracks/tp2/patches/vision/README.md`](../tracks/tp2/patches/vision/README.md) is a pointer and an
install command, not a second copy. Two copies of a file are a coin flip unless something checks
([08](08-fast-boot.md) §12).

---

## 5. What it did, and what it cost

**Settings.** Two DGX Spark (GB10) nodes — `head` (rank 0, serves the API) and `worker-1` — TP=2,
**expert parallelism off**, image `exl3-zeus:754421f`, the two-node tree of
[`tracks/tp2/patches/`](../tracks/tp2/patches/README.md) plus the vision set and the two backports,
launcher [`scripts/start-tp2full.sh`](../scripts/start-tp2full.sh) with its settle gate,
`turboderp/GLM-5.3-Flash-exl3` at 4.05 bpw (full scope), KV `fp8` and an fp8 draft cache, DFlash2 at
k=7, `--attention-backend CUSTOM`, `--block-size 256`, `HAREM_SW_BLOCK_SIZE=256`,
`--max-num-seqs 8`, `--max-num-batched-tokens 2048`, `--max-model-len 1000000`,
**`gpu-memory-utilization 0.85`** (candidate C's rung, untouched), the indexer workspace bound,
`NCCL_MAX_NCHANNELS=8`, per-rank fast-load sidecar, warm MLA tuner cache, temperature 0, thinking on
at reasoning effort **low**. Vision on top: `--mm-encoder-tp-mode data`,
`--mm-processor-kwargs {"max_pixels":12544000,"max_image_tokens":8000}`,
`--limit-mm-per-prompt {"image":4,"video":2}`, `--mm-processor-cache-gb 0`, `HAREM_VISION=1`,
`CUDA_EXL3_PACKED_MAPPING={"qkv_proj":["q_proj","k_proj","v_proj"]}`. 8 September 2026,
**one boot** `[measured-here]`.

### 5.1 The tower loaded on the first attempt

```
HAREM-VISION: tower loaded, data_parallel=True, vit_attn_backend=FLASH_ATTN,
              EXL3 linears=99, unquantized linears=0, dead fused qkv tensors dropped=0
```

`data_parallel=True` at **two** ranks is the flag doing what it says: the tower is replicated, not
sliced. The three model-free gates in the boot log agree with the three-node ones to the count —
99/99 vLLM vision prefixes onto 172 EXL3 checkpoint modules, 959 live vision tensors mapped and 48
dead fused ones dropped, five video geometries where placeholders equal encoder rows — because none
of those numbers is a function of the rank count.

One number does move, and it is the one that proves the tower is *in* the model rather than beside
it: the EXL3 module audit reads **302 EXL3 / 113 bf16** here against candidate C's documented
**203 / 113**. The difference is 99, which is the tower.

### 5.2 The six vision gates

The [runbook](18-vision-at-three-ranks.md) §6 cases, at two ranks `[measured-here]`:

| Gate | Result |
|---|---|
| **K2** one image, four fixtures | **4/4** — "Red circle", "Blue square", "Green triangle", "ELEPHANT" |
| **K3** four images in one request | **PASS** — all four described, **in order**, 2,612 prompt tokens |
| **K4** one video | **PASS** — colour, shape and motion all correct, 1,722 prompt tokens |
| **K4b** two videos in one request | **PASS** — both described and in order, 3,415 prompt tokens |
| **K5** 4 images + 2 videos, the declared limit | **PASS** — all six items, 5,991 prompt tokens |
| **K6** three videos | **HTTP 400**, engine alive afterwards — VS6 and VS7 doing their jobs |

K6 is the one worth reading twice. The failure it guards against is the one that killed three
engine cores at three ranks: a per-request limit that is *declared* but not *enforced in the
frontend* is a dead worker rather than a rejected request. At two ranks the rejection is a 400 with
the parameter named, and the engine answers the next request.

### 5.3 Text quality and speed: the tower costs nothing measurable

Cold gates: correctness probe **10/10**, code exam **12/12** on the first attempt, tool-call **8/8**,
needle-lite **6/6 three times**.

Speed, median of three sweep rounds on `scripts/hizset-v2.jsonl`, against candidate C's published
figures — **different sessions**, so the bands are what matters, not the signs:

| | Candidate C (6 Sep, no tower) | **With the tower and the two backports** | Δ | band |
|---|---:|---:|---:|---|
| C1 aggregate | 60.08 | **59.45** | −1.0 % | ±4 % |
| C1 per stream | 65.96 | **64.45** | −2.3 % | ±4 % |
| C2 aggregate | — | 81.37 | — | |
| C4 aggregate | — | 115.06 | — | ±9 % |
| C6 aggregate | — | 134.35 | — | |
| C8 aggregate | 157.71 | **155.47** | −1.4 % | ±3 % |
| TTFT, C1 / C8 | 0.381 / 1.054 s | 0.375 / 1.080 s | equal | |
| Draft acceptance, C1 / C8 | ~60.4 / 61.3 % | 60.33 / 62.59 % | equal | ±2 pt |
| Prefill, fresh unseen ~8.4K prompts | 1,414 tok/s | **1,413 · 1,403** | equal | ±3 % |

### 5.4 What it cost, and the line is not left empty

| | |
|---|---|
| **KV pool** | 2,692,857 → **2,585,714**, **−4.0 %** (−107,143 tokens). At the two-node conversion rate of ~132,700 tokens/GiB that is **0.81 GiB**, against a 0.557 GiB replicated tower plus its share of activation. **We are calling this a cost, not noise**: this configuration has one boot and candidate C's own boot-to-boot spread was never measured at two ranks. At three ranks the same tower landed inside the spread; here we cannot say that, so we do not |
| **Available KV memory** | 21.31 / 20.33 → **20.15 / 19.54** GiB per rank |
| **One extra dump boot** | The tower changes the tensor name list and the fast-load identity is exact equality. **1,013 s**, and a sidecar that grows 78 → **79 GB per rank** (32 files either way; the tower is 0.557 GiB of EXL3 and the count does not change because it rides inside existing shards) |
| **Boot** | fast-load through the autostart unit, **280 s** to `/health` 200, against candidate C's 272 s by hand and candidate B's 261 s under the unit. Weight restore 84.2 s at 964 MB/s |
| **Host memory** | see [15](15-tp2-track.md) §5.10 — measured under the worst case rather than at idle |
| **Speed** | nothing outside a band, in either direction |

**And what was looked for and not found.** CUDA graphs still capture — 19 PIECEWISE, 8 FULL and 8
DFlash2 FULL, exactly as candidate C — so the three-node graph loss to
[vllm#55581](https://github.com/vllm-project/vllm/issues/55581) still does not happen at two ranks
with a tower in the model.

---

## 6. What is **not** tested here

| | |
|---|---|
| **The sliced tower.** `--mm-encoder-tp-mode data` off at two ranks, so the tower is tensor-parallel across the pair | It should divide — every width is a multiple of 2 and of 128 per rank — and it would halve the tower's per-rank footprint, which is 0.28 GiB of a 0.81 GiB measured cost. It also means slicing a 6-bit EXL3 tower, which this repository has measured nowhere. `[not tested]`, and it is the first thing we would want from a second two-node cluster |
| **A second boot** | Every figure in §5 is one boot. The KV row in particular carries that: without a boot-to-boot spread at two ranks, −4.0 % cannot be separated into "the tower" and "this boot" `[not tested]` |
| **The memory ladder with a tower** | 0.85 is candidate C's rung and was not moved. Whether the tower changes where the safe rung sits at two ranks is unmeasured, and the ladder itself has never been derived at two ranks ([15](15-tp2-track.md) §6) `[not tested]` |
| **A two-node reboot test with vision** | The unit was installed, started and health-checked, and left **disabled**. No power-on trial `[not tested]` |
| **The routed-experts-only checkpoint with a tower at two ranks** | Its tower is dense BF16 and needs `LANGUAGE_MODEL_ONLY=0` and the limits and nothing else — still a code reading `[not tested]` |
| **Images and video under concurrency** | The six gates are one request at a time. Three ranks ran a concurrent stress; two ranks did not `[not tested]` |
