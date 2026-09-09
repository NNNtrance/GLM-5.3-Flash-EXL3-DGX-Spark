# Credits

Nothing in this recipe was built from nothing. Below is every external component we benefited from,
with the exact revision we ran, what we use it for, and its licence as we confirmed it (URL given).
Licences we could **not** confirm are marked as such, with the page we looked at.

Licences are summarised for readers in [LICENSES.md](LICENSES.md). This repository's own content is
Apache-2.0 ([LICENSE](LICENSE)).

Revision discipline: an entry without a commit, a Hugging Face sha or an image digest is not a
revision. Where an upstream project moved after we pinned it, we say so rather than silently pointing
at `main`.

---

## Model and weights

### `zai-org/GLM-5.3-Flash` — the base model

- **What we use it for:** the model this whole recipe serves. Everything downstream derives from its
  architecture: 45 layers, 288 routed experts, one shared expert, 64 attention / 64 KV / 64 KDA
  heads, `vocab_size` 154,880, `moe_intermediate_size` 2,048, `intermediate_size` 12,288. Those 64
  heads and that vocabulary are why TP=3 needs padding at all
  ([docs/03](docs/03-tp3-padding-and-sidecars.md)).
- **Link:** https://huggingface.co/zai-org/GLM-5.3-Flash
- **Licence:** **MIT** — confirmed on the model card metadata (`license: mit`).

### `brandonmusic/GLM-5.3-Flash-tr3-4bpw` — the checkpoint configurations 1–8 loaded, and the fallback

- **What we use it for:** the production checkpoint for configurations **1 through 8**, and now the
  documented fallback and rollback for images that predate the padded-load path. 175.6 GB across 120
  safetensors shards; EXL3
  trellis at 4 bits per weight with the `mcg` codebook; `quantization_config.scope:
  glm53_routed_experts_only`, so attention, KDA, the shared expert and `lm_head` stay BF16. We do not
  modify it on disk — the shape changes go into a sidecar directory of symlinks plus one rewritten
  `config.json`, so its `SHA256SUMS` stays verifiable.
- **Revision:** Hugging Face commit `b20c49ba9ecafb563099536e307d21c1310e1c49` (short `b20c49ba`,
  30 August 2026). **Not the current `main`** — as of 5 September 2026 `main` is `aba59d21`, four
  days newer, which we have not tested.
- **Link:** https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw
- **Licence:** **ShapleyMCG License 1.0** — the model card carries `license: other` with
  `license_name: shapleymcg-license-1.0`, and the licence text is in the repository's `LICENSE` file.
  Read it: it is short, it is not an OSI-approved open-source licence by its own statement, and it
  excludes one named individual from every right it grants. Detail in [LICENSES.md](LICENSES.md).
  **If you run production configuration 9 you do not need this licence at all** — the production
  checkpoint below is plain MIT.
- **What its card reports** `[reported]`: base model `zai-org/GLM-5.3-Flash-BF16`; KL divergence
  against BF16 of **0.0246 nats** over 51,175 positions.
- We do not mirror or redistribute these weights.

### `turboderp/GLM-5.3-Flash-exl3` at 4.05 bpw — **the production checkpoint**

- **What we use it for:** **the weights production configuration 9 serves**, at TP=3 with expert
  parallelism, since the evening of 5 September 2026. It began as the measurement arm that put a
  number on this stack's largest open item — what the unquantized dense stage is actually worth
  ([docs/13](docs/13-full-scope-checkpoint.md)) — and then became production once the loader work and
  the plugin's padded-load path (`f3e3090` + `754421f`) let it load into TP=3 shapes. At TP=2 it
  measured +24.3 % per stream with the MMLU sample inside the control's error bar; at TP=3 it is
  +21.7 % per stream, +12.5 % at C8, +10.0 % of KV pool, quality unchanged `[measured-here]`.
- **Revision:** branch `4.05bpw`, commit `2a30229e67012798ba9f0cd832bb78abf4c363d5` (28 August 2026).
  165.2 GB / 153.8 GiB across 19 shards; exl3 v1.4.4, `mul1` codebook, full scope — routed experts at
  4 bits, dense and attention at 5–6, `lm_head` at 6, calibrated on 250 rows × 2,048 columns. Verified
  here with `sha256` 23/23 against the repository's own metadata, independently on two nodes.
- **Link:** https://huggingface.co/turboderp/GLM-5.3-Flash-exl3
- **Licence:** **MIT** — the `LICENSE` file is the MIT text, "Copyright (c) 2026 Z.AI Co., Ltd", and
  the card carries `license: mit`. More permissive than the fallback checkpoint above: no
  attribution condition and no exclusion clause. Read it yourself; different publisher, different
  terms.
- We do not modify it on disk and we do not redistribute it. Everything needed to serve it is a
  runtime patch inside the container ([docs/13](docs/13-full-scope-checkpoint.md) §3), which is also
  why we rejected rewriting its shards: it would have cost the `sha256` match against the publisher.
- **Also in the repository and unused here:** `mtp.safetensors` (3.79 GB), an MTP drafter that is not
  in the safetensors index and that vLLM never reads. Under this checkpoint's licence it would
  transfer to a reader in a way our own draft model does not. Not evaluated `[not tested]`.

### `incoai/GLM-5.3-Flash-DFlash2` — the speculative draft model

- **What we use it for:** speculative decoding at k=7. 2.3 GB, BF16, 5 draft layers over 45 target
  layers. It is worth roughly 2.6× at a single stream on this stack
  ([docs/04](docs/04-dflash2-port.md)). We pad **our local copy** of its config from 32/8 to 36/9
  heads for TP=3, in a sidecar; the downloaded copy is untouched.
- **Link:** https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2
- **Licence:** **CC BY-NC-ND 4.0** — confirmed on the model card metadata
  (`license: cc-by-nc-nd-4.0`). NonCommercial, NoDerivatives, attribution required.
- **Important, and non-transferable:** we hold a project-specific written permission from the author
  for our own use. It covers us only, it is not sublicensed by this repository, and we do **not**
  redistribute the draft weights. Obtain your own. See [LICENSES.md](LICENSES.md).

---

## Engine and kernels

### `Zeuss5/cuda-exl3` — the EXL3 CUDA kernels and the vLLM plugin

This is the project this recipe is built on, and the one we owe the most to. EXL3 dense and MoE GEMM
kernels for Blackwell plus a fused sparse-MLA attention backend, with a vLLM plugin.

- **What we use it for:** everything EXL3. The routed-expert GEMM, the Hadamard input transforms, the
  combine, the MLA decode backend, the expert-parallel path, and the vLLM integration.
- **Link:** https://github.com/Zeuss5/cuda-exl3
- **Licence:** **MIT** — the repository's `LICENSE` file is the MIT text plus an attribution
  paragraph crediting `turboderp-org/exllamav3` for derived files. Note that GitHub's own licence
  classifier reports `other` / NOASSERTION for the repository, which is what a licence file with an
  added paragraph does to template matching; the file body is MIT terms. Both facts are recorded
  because they are not contradictory and a compliance reviewer will see the second one first.
- **Production commit:** `754421fc99919317b9a4cf2928797bae8f870040` — "Fill a prefix in the vocab
  loaders when the rank is padded", carrying `f3e3090` beneath it, since 5 September evening
  (production configuration 9). **That pair is not an optimisation, it is a prerequisite:** without
  it the full-scope checkpoint cannot be loaded at TP=3 at all, and `tracks/tp3/patches/check-padload-tp3.py`
  refuses the boot rather than letting it fail half way. The previous production commits were
  `62f53e676d3e416401c3a0716558e1454affa8ad`, "Bound what is left in `exl3_moe_had_in`", carrying
  `a47da6e` (production 8), `9bf594cd8b43a2a53db9c7d1d629794aa9365f1a`, "Persist the MLA tuner cache
  across processes" (production 5–7), and before it
  `f4987cf11806c7381c8a59cb388ab5863852679c`, "Do not fetch the MoE padding rows".

Commits this recipe built, measured or depends on:

| Short | Full | Subject | Why it matters here |
|---|---|---|---|
| `37330c9` | `37330c99b20b6199278b838b3d40bda413111a42` | Evict the tuner's cache by reading, not writing | the commit the first working image was built from |
| `e0a3975` | `e0a39752857ff4f4ce80f3811885d24ce4d323e3` | Fix the expert-parallel path | expert-parallel alignment against the global count |
| `77513d2` | `77513d267b78119003e1bd9f4448e3ee71465d76` | Size the MoE block from the global expert count | confirmed correct on GB10; the local count would have cost +32 % at M=2048 ([docs/05](docs/05-expert-parallel-and-cuda-exl3-fixes.md) §3.1) |
| `f906f00` | `f906f002e5e7c3256d236faa56754ae16758c308` | Skip unowned rows in the combine | the design we had arrived at independently, and the better one |
| `60d5349` | `60d5349b42ffad94bb337d7ade5882c42c383b19` | Add `bench_moe_ep.py` | the author's own expert-parallel benchmark, which we ran on both sides of the fix |
| `a95e809` | `a95e8098e65ae1430b6bda79cbbe42b665cc274f` | Pass `n_rows` on the unsplit MoE launch | **the fix for the regression that cost us 45 % of the MoE stage**; the line came from our report and the author's own diagnosis of the mechanism was the correct one |
| `bc0e0f6` | `bc0e0f699a826242a4cf917023d978ecb60ee667` | Bucket topk in the MLA tuner key | the tuner fix; on our model the varying axis turned out to be the batch, not the top-k |
| `61a17bc` | `61a17bcdcd5ec496bd963fd265dda7774225e671` | Do not fuse the input transform | the fusion arm we A/B'd |
| `76598b2` | `76598b22cb105b84c4edd6d164567be7a11e9462` | Drop the fused MoE input transform | fusion removed upstream |
| `f4987cf` | `f4987cf11806c7381c8a59cb388ab5863852679c` | Do not fetch the MoE padding rows | the production commit for configurations 2–4; every table dated before 5 September 06:45 |
| `e24f059` | `e24f059793c01b26418fee054484bcec4316567e` | Always skip the MoE padding rows | the follow-up; equivalent on this hardware, where the gate is always open. Rides along in the `9bf594c` checkout, still unmeasured on its own `[not tested]` |
| `5814c7f` | `5814c7fb09e8d1ffaef19506ef38ff02cd279f18` | Let the down projection finish the MoE | not built here |
| **`9bf594c`** | `9bf594cd8b43a2a53db9c7d1d629794aa9365f1a` | **Persist the MLA tuner cache across processes** | **the production commit.** Written by the author in answer to a measurement of ours; built, wired up and measured here — tune events before serving 18 → 0, and our sweep protocol dropped from five rounds to three ([docs/12](docs/12-tuner-cache.md)) |
| `3cad1d2` | `3cad1d2` | Document the environment variables that exist | not built here; read while preparing the bench below |
| `9b17ea9` | `9b17ea9` | Add the expert-reread bench, and close the duplicate-read question here | **the author's answer to our profile.** We ran his script unmodified on GB10 against the production build; it closed our own open item ([docs/10](docs/10-results-and-roofline.md) §5.4, [docs/11](docs/11-open-issues.md) §2.12) |
| `a47da6e` | `a47da6e` | Remove the 64-bit division in `had_in`, deriving the index from the grid | the follow-up to the same profile: **−10 to −18 %** on `exl3_moe_had_in`, roofline 57 % → 63 % `[reported]`. Worth ~0.2–0.3 % of prefill here. **In production since configuration 8**, and it read exactly as advertised: every serving column inside its own band ([docs/10](docs/10-results-and-roofline.md) §1) |
| **`62f53e6`** | `62f53e676d3e416401c3a0716558e1454affa8ad` | **Bound what is left in `had_in`** | **the production commit** since 5 September afternoon, and the answer to "is there more here": the remaining gap is a **half-ALU** limit — a 128-point Hadamard done with warp shuffles — so it is arithmetic that has to happen rather than traffic that can be removed, worth **≤2 % of prefill** on this stack and unreachable in practice `[reported]`. With it the **`cuda-exl3` MoE stage is closed as an optimisation target here** ([docs/10](docs/10-results-and-roofline.md) §6, [docs/11](docs/11-open-issues.md) §2.19) |
| `5903248` | `59032483e5e57b24a382231557d7f35daa1d2317` | Let a checkpoint declare its own packed-module fusions | written the same hour we reported that `glm5next` declares no `packed_modules_mapping`, so a fusion peculiar to one model can travel with the weights instead of in a fork of the model file. Checkpoint entries merge **under** the model class's, and a malformed entry is dropped rather than raised. We first recorded this as "not usable for our case" because a published checkpoint is not ours to edit — **that was too quick, and production 9 uses it** `[measured-here]`: the sidecar already rewrites files, so `pad-tp3full.py` writes a `quantization_config.json` carrying the mapping and the launcher points `--hf-overrides quantization_config_file` at it. It merges under the patch's own mapping, so both routes are live and neither conflicts — belt and braces ([docs/03](docs/03-tp3-padding-and-sidecars.md) §2) |
| `fba9f27` | `fba9f27daa8790e933a4bd55fc098480a58c40c1` | The same mapping from `CUDA_EXL3_PACKED_MAPPING` | the follow-up, because a published checkpoint is not ours to edit. Verified here against his `config.py` mounted read-only into a CPU container: `in_proj_qkv` 34/34, `gate_up_proj` 45/45, `fused_qkv_a_proj` 11/11 `[measured-here]`. It **is** in the production image (`754421f` is later), so this is a third live route to the same mapping and the one to prefer if you would rather not rewrite a 48 MB config. Write the JSON with **no spaces**: the launcher word-splits `EXTRA_ENV`. Not the route we run `[not tested]` |
| `d19dee0` | `d19dee00f55ad60ba2530bf937dc7184e17515e4` | Handle a bare `suh` on the v1 loader path, and fix the packed-mapping example | **our `ReplicatedLinear` workaround, done properly and on his side.** He infers the shard index from the shape where ours pins it at 0 — the same thing for a replicated module and not for anything else. It is in the production image, so **our A5 routing is now redundant**: it is kept because it runs first and is harmless, and retiring it would change the patch `sha256` and cost a dump boot on every node ([docs/08](docs/08-fast-boot.md) §4). Same commit corrects the README example we had measured at 0/34 and adds the rule that explains it: **every module in a packed group must be EXL3 in the checkpoint** |
| `807d798` | `807d798530d6c6c06039915e5610a0155e32bd54` | Make `CUDA_EXL3_DEBUG_NAMES` print, and report the hits too | the diagnostic he had recommended as our acceptance gate, which printed nothing on our image because it logged at `info`. Now `warning`, and it logs the modules that **resolved** as well as those that stayed BF16, with running tallies — the failure it was meant to catch (half the attention stack silently left in BF16) is now visible as a climbing count. **It is the boot gate production 9 was accepted on**: 203 EXL3 / 113 bf16, read negatively ([docs/09](docs/09-measurement-protocol.md) §5.1) |
| **`f3e3090`** | `f3e3090d0e8c388972ad4c1fa50824a84c01ea6f` | **Support loading EXL3 weights into a dim vLLM has padded** | **half of the production commit, and the larger half.** Three previous refusals become three supported paths: a padded **output** dim is accepted when the pad is whole 128-column blocks, with `svh` allocated zeroed — because `svh` scales elementwise *after* the output Hadamard, so zeroing it on the pad makes those columns exactly zero whatever the trellis holds; the gate refuses a pad that **shares** a 128-block with real output, and says which condition failed and by how much; and the **row-parallel `suh`** load stops narrowing past the end of the checkpoint, copying what exists and zeroing the rest. Resting under all three, checked exhaustively rather than argued: **all 65,536 codes decode finite in every codebook** — `3inst` [−3.957, +3.973], `mcg` [−3.949, +3.949], `mul1` [−3.453, +3.348] — so an uninitialised pad trellis cannot turn `svh = 0` into a NaN. He built it the same afternoon the TP=2 quality gate passed |
| **`754421f`** | `754421fc99919317b9a4cf2928797bae8f870040` | **Fill a prefix in the vocab loaders when the rank is padded** | **the other half, and the one that shows why "mostly built" is not built.** `f3e3090` let a padded vocab past the gate but did not teach the head's own loaders about it: both `copy_` the checkpoint slice into the whole parameter, which is a shape mismatch as soon as the parameter is the padded per-rank width. The gate would pass and the load would then die — a half-loaded stack. The trigger was a question rather than a bug report: we asked whether the vocab-parallel head goes through the same gate as an ordinary linear **before** building on it, and it did not |

Four things we reported to that project and their outcome:

- The expert-parallel alignment receiving the local expert count, and remote rows reaching the
  combine uninitialised — fixed upstream.
- `n_rows` not passed on the unsplit MoE launch — fixed upstream in `a95e809`. **Our claim that it
  also cost the non-expert-parallel path was wrong and we withdrew it**
  ([docs/11](docs/11-open-issues.md) §1.1).
- The fused input transform winning at small batch and regressing at large batch on this hardware —
  the fusion was subsequently removed upstream.
- **The MLA decode tuner's cache being process-local**, with the measurement of what that costs on
  GB10: tune events per boot, the batch shapes that mint new keys while serving, and the
  round-1-versus-round-3 spread it produces in a serving benchmark. We asked whether a persisted
  cache would be accepted and offered to write it; **the author wrote it instead**, in `9bf594c`,
  with a design better than our sketch — a device-name and format-tag keyed filename, `O_APPEND`
  sharing across the tensor-parallel ranks with no lock, and a runtime fallback for unseen keys. The
  code is theirs; ours was the evidence and the request. It removed the largest measurement tax on
  this stack ([docs/12](docs/12-tuner-cache.md)).

A fifth exchange, and the most useful one to record because **the author was right and we were
wrong**: we sent a full step-time breakdown of a 3× GB10 prefill and decode step, with the rulers
measured on the device, ranking what a change to `cuda-exl3` could still be worth
([issue #5](https://github.com/Zeuss5/cuda-exl3/issues/5), and the same content is
[docs/10](docs/10-results-and-roofline.md) §5). Two things came back within the hour. The author
pointed out that our largest proposed item — duplicate expert reads in the large-M trellis GEMM,
which we had priced at 14–27 % of the weight traffic — rests on a traffic model that a trace cannot
verify, wrote **`9b17ea9`, a bench that settles it**, and reported 1.16× on his own 188-SM card. We
ran that bench unmodified on GB10 and measured 1.11×: the trellis stays resident, the item is closed,
and our estimate was wrong because the model behind it was wrong. He then took the item that
*was* real — `exl3_moe_had_in` at 37–57 % of the ruler — in **`a47da6e`**.

The kernel-library half of that thread closed the same day, and it closed the whole stage rather than
the one kernel. In **`62f53e6`** the author bounded what remains in `had_in`: a 128-point Hadamard
implemented with warp shuffles is **ALU-bound at about half the unit's rate**, so the rest is work
that has to be done, not traffic that can be removed — **≤2 % of prefill on this stack, and
unreachable** `[reported]`. He also confirmed that `_zero_kv_blocks_kernel` is vLLM's, and that on
this model family the page is shared with Mamba/KDA state so the zeroing cannot be skipped
([docs/11](docs/11-open-issues.md) §2.13). **We have no open *performance* item against `cuda-exl3`**,
which is an unusual place for a dependency to end up: two of our three reports produced upstream
fixes, the third produced a bench that proved us wrong, and the fourth produced a bound that tells us
to stop looking. The one thing that was still outstanding on that side was not a kernel at all — the
padded-load path for a vocab-parallel EXL3 head — and it was **built the same day the quality gate
passed** (`f3e3090` + `754421f`, below). **We have no open item against `cuda-exl3` of any kind.**

**Then we sent a real profiler trace, and the thread turned into the most valuable exchange in this
repository.** Four things came back on it, and three of them make the author's side of the ledger
longer rather than ours:

- **MLA prefill, closed by him.** We had carried it for a week as "8.2 % of a chunk, efficiency not
  measured, no denominator, needs its own bench". He measured it at our shapes — top-k 2,176,
  head_dim 512, fp8 cache, and a **2 GB pool chosen specifically so it could not be L2-resident** —
  and got **86–89 % of achievable**, 1,299–1,345 GB/s gathered against a 1,518 GB/s ruler in the same
  run `[reported]`. Item closed with no work on our side. He also published that his *first* cut used
  a 200k-row pool that fitted the card's L2 and reported a bandwidth above the DRAM ruler, caught only
  because the number was impossible — the same failure our own model-free MoE bench had just
  committed at small M, on the same day, from the other direction. **Synthetic benches flatter small
  inputs**; that lesson is worth more than either measurement.
  **He then closed the same item a second time, against his own earlier answer, on 6 September.** We
  sent the selection statistics he had asked for — median 2,049 selected keys per query row and a
  median adjacent-row overlap of 0.9258, about 152 keys turning over per row — which showed this
  configuration sitting roughly 76× away from the low-turnover arm rather than between his two arms.
  He built a third arm calibrated to that turnover, swept context, and published `5fd7299`,
  *"Correct the MLA prefill ceiling: at production overlap there is no gap"*: at 262K context the
  production-pattern arm is within **1.6 %** of the fully cache-resident arm (2,422.8 against
  2,385.8 µs) while the independent arm needs 3,474 µs, because the live key set is the residence
  window (~4,096 keys, ≈4.5 MiB) rather than the chunk footprint `[reported]`. **The "21–26 % overlap
  gap" is withdrawn on his side and the item closes at zero.**
- **A conclusion of his own, withdrawn against his own advice.** He had told us a cooperative
  (`grid.sync`) MoE stage was never worth paying for, because "inside a CUDA graph a kernel boundary
  is cheap enough". Our trace says production runs with **graphs off** — spec-decode plus FlashInfer
  forces `cudagraph_mode=NONE` — so the premise did not hold on this stack, and our own uncaptured
  measurement had the cooperative arm winning by up to 1.4 µs per boundary. He said so unprompted, and
  then priced it honestly at ~0.2 ms of a 94.65 ms step and told us not to rush ([docs/11](docs/11-open-issues.md) §2.14).
- **The largest number anyone has put on this stack, and it is not his code.** Reading our C1 split he
  pointed at the row we had been treating as background: dense BF16 GEMM at **45 % of a decode step**,
  because our checkpoint is `scope: glm53_routed_experts_only` and that stage streams 16-bit weights
  at M=8. `Exl3LinearMethod` binds to any dense linear with EXL3 tensors present, so a broader-scope
  checkpoint would put those layers on the same kernel **with no code change on either side** —
  42.9 ms → ~11 ms, **~+34 % single-stream** `[estimate]`, against ~5 % for his whole column of our
  target table. He framed it as a quality question rather than a speed one, named the head at vocab
  154,880 as the place to look for damage first, and said he would want the numbers either way.
- **And a mechanism for the TP=3 half of it that we did not have.** Our objection was that 64 heads
  and a 154,880 vocab do not split three ways, and that `Exl3LinearMethod.create_weights` correctly
  refuses to zero-extend an EXL3 tensor. His answer is the 2,304-sidecar property one level up: in
  `had128_warp_out` the output Hadamard runs **first** and `svh` scales elementwise **afterwards**, so
  zeroing `svh` on a padded output column makes that column exactly zero whatever the trellis holds —
  the invariant already guarded by `test_exl3_moe_pad.py::test_padded_columns_are_exactly_zero`, and
  mirrored for padded *input* dims by `test_output_ignores_w2_padded_row_codes`. A full-scope
  checkpoint quantized **unpadded** can therefore be loaded into a padded parameter, bit-exact, with
  no re-quantization — **provided the pad occupies whole 128-column blocks**. He then checked our two
  tensors against that condition rather than asserting it: heads 64 → 66 = 256 columns = 2 whole
  blocks, **works**; vocab `padding_size=192` = 1.5 blocks, **corrupts the real rows sharing the
  block**. The fix is `padding_size=384`, one constant in our launcher. He has offered to add the
  padded-load path behind a flag once our quality gate says whether it is worth building
  ([docs/11](docs/11-open-issues.md) §2.22).

**Then the full-scope checkpoint went in, and the thread produced four commits in one afternoon.** The
work split cleanly and he drew the line himself: the packed-module mapping belongs in the plugin
because it is checkpoint layout, while the BF16 pinning and the KDA split are model structure and stay
in the model file — "that is the part you did not want to carry as a fork patch". Two of the four are
straight fixes to our reports, and one is a fix to something he had written:

- **`5903248` then `fba9f27`.** He put the packed mapping on his side within the hour, reading it from
  the checkpoint's own quantization config — and then, **before we could wire it up**, told us it was
  unusable for our case as written, because a published checkpoint is not ours to edit. The env-var
  route in `fba9f27` is the fix. Catching your own new feature's blind spot in the same thread, ahead
  of the person who would have hit it at boot, is worth recording.
- **`d19dee0`**, which is our `ReplicatedLinear` workaround done better: he infers the shard index
  from the shape where we pin it at 0 — identical for a replicated module, not identical in general.
  The same commit corrects a README example he had written from our prose without running it, and
  which we then measured at 0/34; he said so plainly and added the rule that explains it, which is the
  part that saves the next reader an afternoon.
- **`807d798`**, after we reported that the diagnostic he had recommended as our acceptance gate
  printed nothing on our image.

**And the one answer that unblocks TP=3 in principle.** We asked for a one-line confirmation that a
zero pad trellis multiplied by `svh = 0` cannot produce NaN. What came back was **the full domain
enumerated**: all 65,536 possible 16-bit trellis codes swept through the device decoder for all three
codebooks, zero non-finite values in every case, with the ranges reported (`3inst`
[−3.9570, +3.9727], `mcg` [−3.9492, +3.9492], `mul1` [−3.4531, +3.3477]) `[reported]`. He also
published that his first attempt reported an implausibly asymmetric range because it ordered floats by
their integer bits, and re-ran it rather than quoting a number he did not trust. **That is the right
standard underneath a padded-load path**, and it is a stronger answer than the question asked for.

The **commitment** that followed from it: once our quality gate decided the lever was worth building,
he would add the padded-load path behind a flag — accept an EXL3 tensor narrower than the parameter,
place it, zero `svh` on the remainder — covering the three cases we had checked our TP=3 geometry
against: the vocab-parallel `lm_head` (hard-refused at the time, and this checkpoint has no BF16 head
to fall back on), a "refuse unless the pad is 128-aligned" gate, and the input-dim case where
`Exl3SuhParameter.load_row_parallel_weight` narrows on its own and would overrun the last rank's
`o_proj`. He asked for the drafter's acceptance rate ahead of MMLU, on the grounds that a quantized
target losing acceptance would decide TP=3 before quality did — a better ordering than ours.

**The gate passed on 5 September, and the commitment was kept the same afternoon.** All three cases
are in `f3e3090`, and the fourth thing — which nobody had asked for and which is the reason the port
worked on the first attempt — is in `754421f`: he noticed, when we asked whether the vocab-parallel
head went through the same gate as an ordinary linear, that it did not, and that `f3e3090` alone
would pass the gate and then die in `_vocab_loaders`. **A half-built path is worse than none**,
because it fails after the weights have started moving. Four hours later the checkpoint was serving
production at TP=3: **+21.7 % per stream, MMLU 86.47 ±0.74, 285 padded EXL3 sites audited as whole
128-blocks and exactly zero** ([docs/13](docs/13-full-scope-checkpoint.md) §7). Two upstream commits,
two constants and one patch of ours, and no kernel changed.

The `f3e3090` gate is worth naming on its own, because it is the part a reader is most likely to get
wrong: it does not merely require the pad to be aligned, it refuses a pad that **shares** a
128-column block with real output — and says which condition failed and by how much. That is the
difference between "the pad is not zero" and "the real columns beside the pad are corrupted", and it
is the failure our own `lcm(64, tp)` arithmetic would have produced **silently**
([docs/03](docs/03-tp3-padding-and-sidecars.md) §1.1).

Three notes we owe that thread rather than the other way round: a 128-token prefill chunk costs 403 ms
because 128 tokens at top-8 already touch every expert, which is a batched-token-budget fact worth
knowing on any Spark; a uniform-random top-8 microbenchmark overstates small-M MoE cost by
**1.5–1.7×** against real clustered routing, which he accepted applies to his own `bench_moe_ep.py`
and padding-skip benches; and two corrections against our own published numbers —
`_zero_kv_blocks_kernel` from 14.7 ms to 0.86 ms, and the drafter from 19.5 % to 11.4 % of a C1 step —
which he pointed out is the harder direction to publish.

Our one surviving kernel change, `patches/kernel/0003-combine-smem-staging-on-a95e809.patch`, is
offered upstream with its description in `patches/kernel/0003-PR-DESCRIPTION.md`.

### `turboderp-org/exllamav3` — the EXL3 format

- **What we use it for:** the quantization format itself. The trellis layout, the codebooks and the
  128-element Hadamard block whose alignment rule decides that an EXL3 tensor cannot be split three
  ways ([docs/03](docs/03-tp3-padding-and-sidecars.md)).
- **Link:** https://github.com/turboderp-org/exllamav3 (note the organisation account; the
  personal-account path does not exist)
- **Licence:** **MIT** — confirmed on the repository.

### `vllm/vllm-openai` — base container image

- **What we use it for:** the bottom of our two-layer image: aarch64 + CUDA 13.0, the GLM-5.3 model
  implementation, and the vLLM the whole stack runs inside.
- **Revision:** tag `glm53-flash-arm64-cu130`, digest
  `sha256:905c02933be6021301db2dc284e24e3727467aa3a0f63b41d609885778a07bce`, linux/arm64, pushed
  26 August 2026. The vLLM inside reports `0.1.dev20051+g487ecf187`.
- **Links:** https://hub.docker.com/r/vllm/vllm-openai/tags · https://github.com/vllm-project/vllm
- **Licence:** **Apache-2.0** for vLLM — confirmed at
  https://github.com/vllm-project/vllm/blob/main/LICENSE . The NVIDIA CUDA, cuDNN and NCCL libraries
  inside the image carry NVIDIA's own redistribution terms, which is why we do not redistribute any
  image.

### vLLM upstream — the DFlash2 speculative-decoding support

- **What we use it for:** the DFlash2 delta that the base image predates, taken as a genuine
  three-way merge onto the image's own vLLM tree rather than hand-copied
  ([docs/04](docs/04-dflash2-port.md)).
- **Licence:** **Apache-2.0**, as above.
- **Reported upstream by us (6 September 2026):** [vllm-project/vllm#55581](https://github.com/vllm-project/vllm/issues/55581) —
  FlashInfer's `get_cudagraph_support()` checks the *target* model's head count against the *draft*
  group's KV heads, which turns CUDA graphs off at TP=3 while the same image captures them at TP=2
  ([docs/11](docs/11-open-issues.md) §2.29). No fix merged at the time of writing; we carry no patch
  for it, by choice.

### [@drakosha](https://github.com/drakosha) — the correction we did not find ourselves

- **What we owe him:** the retraction in [docs/11](docs/11-open-issues.md) §1.13. He opened
  [vllm-project/vllm#55221](https://github.com/vllm-project/vllm/issues/55221), the sparse-indexer
  K-gather workspace issue our own patch works around, and when we posted the failure mode we had
  built our safety story on — "a too-small workspace ends in upstream's locked-workspace
  `AssertionError`" — he
  [corrected it](https://github.com/vllm-project/vllm/issues/55221#issuecomment-5561194190): both
  indexers ask the `WorkspaceManager` for a **static** size and slice per chunk afterwards, so the
  request never grows and that assertion cannot fire from this path at all. The real failure mode is
  a **silent clamp** producing wrong answers, and the load-bearing safety layer is ours rather than
  upstream's. We had the evidence in our own A/B — one resize event per rank, for the life of the
  engine — and had not read it.
- **Why it is here rather than in a footnote:** it is the first of this repository's retractions that
  a reader outside this stack caught, and it is the one we would least have found alone. The
  correction is carried in [`results/memory/indexer-workspace-ab.md`](results/memory/indexer-workspace-ab.md)
  §2, [docs/11](docs/11-open-issues.md) §2.28, [HELP-WANTED](HELP-WANTED.md) §9 and the patch's own
  docstring, and the thread is answered.
- **Licence:** not applicable — we vendor nothing of his. What we took is a correction.

### [Victor Cruz (@vcruz305)](https://github.com/vcruz305) — the K-pool tail, diagnosed and fixed before we looked

- **What we owe him:** the whole of the K-pool tail item —
  [`tracks/tp3/patches/prefix-hit-and-kpool-tail/`](tracks/tp3/patches/prefix-hit-and-kpool-tail/README.md)
  §2 and [docs/11](docs/11-open-issues.md) §2.33. `KpoolTailSpec` allocates one block per request and
  is addressed circularly, but its slot mapping is produced by the generic paged kernel against a row
  whose only written column is 0, so concurrent requests collapse onto one tail block and, past the
  row width, the kernel reads outside it. He found it, wrote the reproducer, and established that the
  correct mapping was already in the tree and was being skipped because the hybrid
  attention-metadata path never passes `positions`. Both edits in our `patch-kpooltail-tp3.py` are
  his: the missing `positions=` argument in `model_states/mamba_hybrid.py`, and writing the corrected
  mapping in place rather than returning a clone whose address CUDA-graph capture pins.
- **What he saved us from doing wrong:** the brief we started from said to clamp the block index
  inside the generic slot kernel. His notes record that he tried exactly that and it changed nothing
  — 48 overrunning calls before and 48 after — because once positions are present the tail mapping
  does not come from that kernel at all `[reported]`. He also records the measurement trap: Python
  instrumentation inside a CUDA-graph-captured op runs at capture and never again, so only
  device-updated counters are evidence there. We serve eager, so our detector is host-side and says
  so.
- **Where:** `vcruz305/GLM-5.3-Flash-EXL3-K2-DGX-Spark-recipe`, `docs/KPOOL_TAIL_BUG.md` and
  `scripts/patch_kpool_tail_positions.py` / `scripts/patch_kpool_tail_detector.py`, 30 August 2026.
- **Licence:** we vendor nothing of his. What we took is a diagnosis and a two-line mechanism, both
  re-implemented against our own anchors and gated by our own environment knob.

### vLLM upstream — the hybrid prefix-cache hit

- **What we use it for:** the first half of
  [`tracks/tp3/patches/prefix-hit-and-kpool-tail/`](tracks/tp3/patches/prefix-hit-and-kpool-tail/README.md).
  Our pinned tree predates three changes, and the shape of our fix is theirs:
  - [#52047](https://github.com/vllm-project/vllm/pull/52047),
    [@okorzh-amd](https://github.com/okorzh-amd), merged 29 August 2026 — generalises the
    DeepSeek-V4-only `_annotate_eagle_groups_deepseek_v4` into a marker-driven
    `_annotate_eagle_groups`, and warns when speculative decoding leaves every group unannotated,
    which is precisely the state our grouping path was in.
  - [#54041](https://github.com/vllm-project/vllm/pull/54041),
    [@positive666](https://github.com/positive666), 27 August 2026, closed — the idea we took: mark
    the drafter's group as the EAGLE group **only** when every draft layer is sliding-window, and
    leave a mixed drafter on the conservative fallback. Our patch enforces that precondition.
  - [#53388](https://github.com/vllm-project/vllm/pull/53388),
    [@ZeldaHuang](https://github.com/ZeldaHuang), merged 1 September 2026 — the
    `disable_eagle_block_drop` speculative option, which we carry as a diagnostic arm rather than a
    production setting.
  - [#53906](https://github.com/vllm-project/vllm/pull/53906),
    [@ZJY0516](https://github.com/ZJY0516), crediting **JaredforReal** — the GLM-5.3-Flash support
    this whole stack is built on, including the K-pool machinery and
    `tests/v1/attention/test_kpool_tail_slot_mapping.py`, which pins the addressing our own unit test
    re-checks against this image's tree.
- **The reporters:** [@Suppressor72](https://github.com/Suppressor72), whose
  [issue #53670](https://github.com/vllm-project/vllm/issues/53670) is the throughput report behind
  the three prefix-cache pull requests, and **UserHIJ**, who reported the same class of failure.
- **Licence:** **Apache-2.0**, as above. We carry the mechanism, not the diff: our grouping path is
  not upstream's, so the anchors are ours.

### `autoscriptlabs/nccl-mesh-plugin` — the fabric transport

- **What we use it for:** NCCL over three direct ConnectX-7 links with no switch. Without it there is
  no three-node collective on this topology.
- **Revision:** commit `19924dcc7c571d6e260953724d394ae50bad82cf`.
- **Link:** https://github.com/autoscriptlabs/nccl-mesh-plugin
- **Licence:** **MIT** — confirmed on the repository.
- **Three findings reported upstream, all with patches, none of them the plugin author's fault to
  have missed** — two of the three only show up on a fabric with more than one cable per pair:
  1. A one-line flow-control setting (`min_rnr_timer = 12`, 0.64 ms, where the comment intends code 1
     at 0.01 ms) that costs about 10× on mid-size collectives.
     `patches/kernel/0004-min-rnr-timer.patch`. The plugin's own published benchmarks show the same
     curve on the author's hardware, so it is a property of the design, not of anyone's cabling.
  2. **`mesh_connect()` ignoring NCCL's `dev` index** (`(void)dev;`) and stopping at the first
     subnet match, so every channel to a peer rides one cable and the second cable of each pair never
     carries a byte — confirmed in hardware with `port_xmit_data == 0` on those ports after weeks of
     use. `patches/kernel/0005-device-aware-link-selection.patch`, ~30 lines, no wire-format change.
  3. **`ptrSupport` advertising `NCCL_PTR_HOST` only**, although the plugin's own `regMr` already
     registers CUDA pointers and plain `ibv_reg_mr` works for device pointers on this unified-memory
     part — so NCCL was staging every transfer through a host bounce buffer for nothing.
     `patches/kernel/0006-ptr-cuda-dmabuf-and-flush.patch`, which also makes `mesh_iflush()` a real
     RDMA_READ and `regMrDmaBuf` a real DMA-BUF registration.

  Findings 2 and 3 went upstream together as a follow-up on the plugin's issue thread (issue #58),
  with the sweep numbers and the port-counter evidence. All three patches now live on a public fork,
  [`NNNtrance/nccl-mesh-plugin`](https://github.com/NNNtrance/nccl-mesh-plugin), branch
  `gb10-dual-link-ptrcuda` on top of `19924dcc`, and are offered as
  [**pull request #59**](https://github.com/autoscriptlabs/nccl-mesh-plugin/pull/59). All three are
  env-gated (`NCCL_MESH_LINKS_PER_PEER`, `NCCL_MESH_PTR_CUDA`, `NCCL_MESH_FLUSH`,
  `NCCL_MESH_MIN_RNR_TIMER`) and default to today's behaviour when a peer has one cable. Details and
  numbers in [docs/06](docs/06-nccl-mesh.md) §6–§8. We do not ship the built `.so`.

- **A fourth patch, deliberately not offered** — `patches/kernel/0007-one-sided-fifo-rdma-write.patch`
  replaces the two-sided SEND/RECV data path with the receiver-advertised FIFO plus
  `RDMA_WRITE_WITH_IMM` that NCCL's own IB transport uses, which is the design the plugin's own
  roadmap names. It works: RNR retries and out-of-buffer events go to **exactly zero**. It is also
  worth **nothing** on this hardware, because the ceiling is the cards' PCIe Gen5 x4 slots and not the
  flow control. Sending a 977-line transport rewrite upstream on the strength of a mechanism that
  changes no number would cost the maintainer time we have no evidence is worth spending; it is kept
  in this repository as an option and described honestly in
  [docs/06](docs/06-nccl-mesh.md) §10. Credit where it belongs: the design is NCCL's, and the
  plugin's README had already named it as future work.

### NCCL 2.30.7 (inside the base image)

- **Licence: not confirmed for the binary we run.** The upstream project states most of it is
  Apache-2.0, parts keep an original BSD licence, and borrowed files carry their own text; we did not
  read the licence files inside the image. Treat it as part of the NVIDIA container stack under
  NVIDIA's terms: use it, do not redistribute the image.

### CUDA 13.0, cuDNN, NVIDIA driver 580.173.02, DGX OS 7.5.0

- **Licence: NVIDIA licence terms.** Not open source. Install them from NVIDIA on your own hardware.
  Do not redistribute them, and do not redistribute images containing them.

---

## Prior art we learned from

We vendor no files from either of these. What we took is practice and arithmetic, and we say which.

### `FlyCockpit/GLM-5.3-Flash-EXL3-3x-DGX-Sparks`

- **Licence:** **MIT** — confirmed. 1 star.
- **What we took:** the three-node arithmetic. The 64 → 66 head padding with 22 heads per rank; the
  vocabulary `padding_size` of `lcm(64, tp)`; padding the BF16 shared expert to 2,112 and slicing it
  rather than replicating it; the drafter's 32/8 → 36/9 pad; and the conditional-shard idea that lets
  a loader leave an already-correct tensor alone. Also, and importantly, its written-up finding that
  replicating the shared expert under expert parallelism produces a silently tripled contribution —
  a model that stays fluent and gets the answers wrong.
- We re-derived each of those from this checkpoint's own shapes and check them at load time rather
  than trusting them ([docs/03](docs/03-tp3-padding-and-sidecars.md) §5).

### `tpurtell/glm-5.3-flash-ext3-2x-rtx` — four sm_12x findings, and a ReplaySSM extension we did not take

- **Licence:** **Apache-2.0**.
- **What we took: four defect reports and the shape of their fixes**, reported to us as items 1–4 of
  [`Zeuss5/cuda-exl3` issue #6](https://github.com/Zeuss5/cuda-exl3/issues/6) and originally found in
  their own dual-Spark qualification. The PDL gate that turns Programmatic Dependent Launch on for
  unqualified sm_12x parts; the uninitialised K-pool top-k buffer and its unbounded reader, which are
  the write and the read of one array; and the indexer's Triton specialisation. **Credit for all four
  is theirs.** Our scripts in
  [`tracks/tp3/patches-optional/sm12/`](tracks/tp3/patches-optional/sm12/) are re-implementations
  against our own anchors — three images, three different line numbers and, for the K-pool buffer,
  three different site counts — and the Apache-2.0 attribution travels with them.
- **What we measured and reported back**, so the ledger is not one-sided: no PDL race is detectable
  on this three-node GB10 topology by a logit-divergence probe, PDL on/off is a wash on 48 SMs, item
  4 is smaller than the issue implies (triton 3.7.1 gives an int argument three specialisation
  classes, not one per value, and the tree's warmup already covers most of them), and the PDL gate
  has one call site on the EXL3 side that the issue does not list
  ([`results/kernels/sm12-stack-patches-ab.md`](results/kernels/sm12-stack-patches-ab.md)).
- **What we did not take:** `patches/vllm-replayssm-spec.patch`, which lifts upstream vLLM's refusal
  to run ReplaySSM alongside drafting and would take this stack's KDA state slots from 9 to 2. It
  applies 127/127 clean to our image and is pure Python and Triton. It is not adopted, the reasons
  are speed and an unproven fp16 replay rather than the code, and both are written down in
  [docs/17](docs/17-memory-ledger.md) §5.3 and [HELP-WANTED](HELP-WANTED.md) §6.

### `NNNtrance/GLM-5.3-Flash-NVFP4-TP3-3x-DGX-Spark` — our own sibling recipe

- **Licence:** **Apache-2.0**.
- The cluster setup, the fabric wiring and preflight, the "reboot all three" rule, the memory rules,
  the measurement scripts (`bench-sweep.py`, `correctness-probe.py`, `code-exam.py`,
  `category-speed.py`, `cold-warm-c1.py`, the `hizset-v2` prompt set) and the roofline hardware
  measurements are shared with it. Those scripts are reproduced here unchanged apart from the default
  port.

---

## Measurement tools

- **`bench/` in this repository** — the model-free harness (expert-parallel geometry, per-stage MoE
  timing, expert-map placement, mesh all-reduce with hardware counters, top-k comparison, fresh
  prefill, profiler driver and analyser). Written by us; Apache-2.0. The step-breakdown work added
  seven more, also ours: `bw.py` and `gemmpeak.py` (the two rulers — run them in the same process as
  whatever you are measuring), `live-step.py` and `live-decode.py` (prefill ladder, chunk-boundary
  probe and exact ms-per-engine-step against a running server, via its own metrics), `mhc_bench.py`
  (the hyper-connection kernels, three routes, graph on and off), `zerokv_bench.py` (vLLM's KV-zeroing
  kernel at the live engine's grid geometry) and `prof-analyze3.py` (the kernel taxonomy and pass
  segmentation the breakdown in [docs/10](docs/10-results-and-roofline.md) §5 is built from; the older
  `prof-analyze.py` is kept).
- **`bench_moe_expert_reread.py`** — **not ours**. Written by the `cuda-exl3` author in `9b17ea9` to
  settle a question we had raised the wrong way round; we ran it unmodified and it closed the item
  ([docs/11](docs/11-open-issues.md) §2.12). Under that project's MIT licence.
- **[`bench/mla-prefill/`](bench/mla-prefill/)** — `bench_mla_prefill_5fd7299.py` is **not ours**: a
  verbatim copy of the author's own `bench/bench_mla_prefill.py` at his commit `5fd7299`, kept unedited
  for provenance, under that project's MIT licence. `mla_prefill_falsify.py` is ours (Apache-2.0)
  except that its three arms and selection generator are copied verbatim from the same file under the
  same MIT terms, so the fixture it runs is his rather than a reimplementation of it. Used to falsify
  his `5fd7299` correction on our own GB10 — it held, tighter than on his card
  ([docs/11](docs/11-open-issues.md) §2.27,
  [`results/kernels/mla-prefill-falsification-gb10.md`](results/kernels/mla-prefill-falsification-gb10.md)).
- **PyTorch profiler**, via vLLM's `--profiler-config`. Note that on this vLLM the environment
  variable form does nothing and the endpoint returns 404 — the flag is the only way in
  ([docs/05](docs/05-expert-parallel-and-cuda-exl3-fixes.md)). Set it on the launcher **before** you
  need it: an engine already serving production cannot be profiled and should not be restarted to
  try ([docs/10](docs/10-results-and-roofline.md) §5).

---

## Our own patches

Everything under `patches/` was written by us for this recipe unless the file header says otherwise.
Use them freely under Apache-2.0; a credit is appreciated and not required. Where a patch implements
someone else's idea, the header says whose.

Two directories were added on 5 September evening with production configuration 9.
[`tracks/tp3/patches/`](tracks/tp3/patches/) is the production patch tree for the full-scope checkpoint —
`patches/tp3/` with two constants changed and `patch-fullscope-tp3.py` added. Why it is a second
directory rather than two more files in the first one is the fast-load manifest identity, and it is
explained in its own README. [`tracks/tp2/patches/patch-fullscope-tp2.py`](tracks/tp2/patches/patch-fullscope-tp2.py)
is the eight-anchor TP=2 form of the same patch, which [docs/13](docs/13-full-scope-checkpoint.md)
referenced as "not yet in repo" for a day; it is here now, and it is kept rather than superseded
because it is the smaller, more readable statement of the same three loader layers.

[`tracks/tp3/patches/indexer-workspace/patch-indexer-workspace-tp3.py`](tracks/tp3/patches/indexer-workspace/patch-indexer-workspace-tp3.py)
was added on 6 September. It is ours, the finding behind it is ours — read out of vLLM's own source
and then confirmed with an upstream debug switch — and it **shipped the same evening as production
configuration 12**: it bounds the sparse indexer's K-gather workspace, worth +10.25 % of KV pool at
three ranks and +32.14 % at two
([`results/memory/indexer-workspace-ab.md`](results/memory/indexer-workspace-ab.md)). The defect it
works around belongs upstream rather than here, where it was already reported three days before we
measured it; our numbers are on that thread, and so is the correction that became
[docs/11](docs/11-open-issues.md) §1.13
([vllm-project/vllm#55221](https://github.com/vllm-project/vllm/issues/55221),
[HELP-WANTED](HELP-WANTED.md) §9).

The published copies are **scrubbed** — internal paths in comments replaced by references to these
docs — so their `sha256` does not match the ones our nodes' manifests record. That only matters
because the manifest hashes them: you will dump your own sidecar against your own copies.

Two of them are not ours to keep:

- `patches/kernel/0002-harem-on-77513d2.patch` is **retired** — three of its four changes are
  upstream and the fourth was measurably the wrong choice. `patches/kernel/0002-RETIRED.md` records
  that in writing rather than deleting it quietly.
- `patches/kernel/0004-min-rnr-timer.patch`, `0005-device-aware-link-selection.patch` and
  `0006-ptr-cuda-dmabuf-and-flush.patch` are patches against someone else's project
  (`autoscriptlabs/nccl-mesh-plugin`), offered upstream as
  [PR #59](https://github.com/autoscriptlabs/nccl-mesh-plugin/pull/59) from the fork
  [`NNNtrance/nccl-mesh-plugin`](https://github.com/NNNtrance/nccl-mesh-plugin), and carried here so
  the findings are reproducible. If they land upstream we will retire our copies in writing, the way
  `0002-RETIRED.md` retires ours.
- `patches/kernel/0007-one-sided-fifo-rdma-write.patch` is against the same project and is **not**
  offered upstream and **not** in that PR branch, because our own measurement says it changes nothing
  on this hardware. It is kept, with the measurement that rejected it, rather than deleted
  ([docs/06](docs/06-nccl-mesh.md) §10).

## Field reports

### jdecker76 — issue #1, the first production report

Four days of multi-user agentic traffic on production configuration 13, side by side with the NVFP4
sibling on the same cluster; a root cause this repository had missed (the template's
`clear_thinking` default and the launcher that did not guard it); the provenance finding that closed
[docs/14](docs/14-troubleshooting.md) §9.11 (only `chat_template.jinja` has ever changed in
`zai-org/GLM-5.3-Flash`); the field evidence for the K-pool tail fix; and the design of the
multi-turn gate in [HELP-WANTED](HELP-WANTED.md) §14. Corrected their own report in public when a
measurement disagreed with it, which is the standard this repository tries to hold itself to.
