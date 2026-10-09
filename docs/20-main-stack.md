**Applies to: three nodes (TP=3).** The two-node track has not been moved to this stack `[not tested]`.

# 20 — The vLLM-`main` stack: why we rebuilt, what changed, and what it cost

From 8 October 2026 the three nodes serve GLM-5.3-Flash from a different engine than the one the rest of this
repository describes:

- upstream vLLM `main` at `21d93d0d8`;
- our `cuda-exl3` fork, [`NNNtrance/cuda-exl3`](https://github.com/NNNtrance/cuda-exl3/tree/tp3-vllm-main)
  branch `tp3-vllm-main`, production build `448f1d6`;
- fourteen build-time patches.

The reason is not speed. The old stack never loaded the MoE router's load-balancing bias, so every expert
choice it made was the wrong one, and nothing in it could have told us. This page covers:

- how that was found;
- what else the same audit found;
- what the new stack is and how production is configured;
- what each setting cost;
- what is still open.

The files to run it are in [`tracks/tp3-main/`](../tracks/tp3-main/README.md). The measurements are in
[`results/main-stack/`](../results/main-stack/README.md).

| | Old stack (configurations 1–13, until 8 October) | vLLM-`main` stack (production since 8 October) |
|---|---|---|
| vLLM | the model-launch image `vllm/vllm-openai:glm53-flash-arm64-cu130` (vLLM `0.1.dev20051+g487ecf187`), [02](02-image-build.md) | upstream `main` @ `21d93d0d8` (base image `vllm/vllm-openai@sha256:6f0d5e67…`) |
| `cuda-exl3` | `754421f` and earlier upstream | fork `448f1d6`: TP=3 pieces folded into the plugin, router bias loaded with a **boot gate**, routed-expert SwiGLU clamp |
| Router bias `e_score_correction_bias` | **never loaded** (zero in every MoE layer) | loaded; the engine refuses to start without it |
| Assistant-token KL to the official reference | **0.1033** | **0.00828** (repeat floor 0.0081, reference jitter 0.0067) |
| Expert set equal to the official top-8 | 0 / 96 positions | 96 / 96 |
| Quality benchmarks | withdrawn ([11](11-open-issues.md) §1.15) | **not yet re-measured** `[not tested]` |

---

## 1. How it was found: the model against its official definition

Every quality gate this repository ever published passed on the old stack:

- the correctness probe;
- the code exam;
- the tool-call gate;
- needle-lite;
- GSM8K and IFEval above the NVFP4 sibling.

None of them could see the defect, because each asks whether an answer is acceptable, not whether the
engine computes the model.

The check that saw it compares token probabilities with the checkpoint author's own implementation. The
reference was exllamav3 1.5.4, run on the same EXL3 checkpoint. The comparison covered six texts and
21,066 scored positions, and reports the KL divergence on assistant tokens. Details are in
[`results/main-stack/kl-health.md`](../results/main-stack/kl-health.md) `[measured-here]`.

- **The gap.** The old engine sat at **0.1033**. Its own repeat-noise floor was 0.0094, and the reference
  against a bf16-rounded copy of itself read 0.0067. So the gap was eleven times the engine's own noise.
  Changing the KV dtype, the KDA prefill kernel or the SwiGLU clamp did not move it.
- **The layer.** A layer-by-layer comparison put the jump at layer 3, the first MoE layer. The block
  differed 152× more than bf16 jitter does there, and the routed experts' output was 0.77× the
  official formula's.
- **The cause.** `Exl3MoEMethod.create_weights` installed its own weight loader on every direct parameter
  of the MoE layer. That included the router bias, which vLLM registers on the same layer before the
  method runs. That loader silently declines a tensor arriving without expert arguments. The bias stayed
  at zero; the checkpoint's has a mean of 7.76. vLLM's missing-weight check is off for quantized
  checkpoints, so nothing reported it.
- **The fix.** It is two lines: the loader goes only on the plugin's own parameters.
  - A boot gate now logs `[HAREM-MOE-KAPI]` per MoE layer and refuses to start if the bias is not loaded.
    It reads 42/42 on all three ranks.
  - With the bias loaded, KL is **0.00841**, perplexity is 2.222 against the reference's 2.224, and the
    expert sets match 96 / 96.
  - The fix is offered upstream as [Zeuss5/cuda-exl3#8](https://github.com/Zeuss5/cuda-exl3/pull/8). We
    no longer wait for it: this recipe points at our fork.
- **The SwiGLU clamp.** GLM-5.3 declares `swiglu_limit = 10`. The dense and shared-expert MLPs honoured
  it; the routed experts did not, because this MoE method is not a modular kernel and vLLM's kernel
  oracle never saw it.
  - The production build clamps them too (`42474b4`).
  - Its effect is below what we can measure: KL 0.00828 inside the 0.00808–0.00841 band, step time
    −1.85 % at two steps of resolution.
  - It is there for fidelity to the official definition, not to fix an error.

**What it cost.** Correct routing reads 28–38 % more distinct experts per rank and step.

- Decode steps grew by **11 %** at one stream and **20 %** at four.
- The aggregate a user sees fell about 5 % at one stream and 12 % at four.
- Prefill moved by −0.7 to −1.9 %.
- Part of the speed the old stack published came from reading the wrong, cheaper set of experts
  ([`speed-map.md`](../results/main-stack/speed-map.md) §1) `[measured-here]`.

## 2. What the same audit found first: four live correctness bugs on the old stack

The audit that ended in the router bias began on 3 October with a re-read of the running engine against
upstream. It found four defects that act on long agent sessions, each confirmed twice in the running bytes.
All four are upstream issues or pull requests; none is a bug of this recipe's design. `[measured-here]`

| | Defect | Upstream | Where it bites |
|---|---|---|---|
| R1 | With the drafter's 256-token sliding-window group, the scheduler's prefix grid is 256 while the KDA state slot is 3,328 tokens, so most prefix hits resumed 34 KDA layers from a state up to 1,536 tokens early | [#54076](https://github.com/vllm-project/vllm/pull/54076) (open) | every multi-turn session with a prefix hit |
| R2 | The K-pool seed kernel wrote with a dense stride: wrong pages overwritten, the request tail never seeded | [#57477](https://github.com/vllm-project/vllm/pull/57477) (merged 20 Sep) | contexts longer than `index_topk` |
| R3 | The four-slot tail ring overflowed at draft length 7: 40–65 % of decode pools carried a wrong summary key | [#58454](https://github.com/vllm-project/vllm/pull/58454) (merged 25 Sep) | contexts longer than `index_topk` |
| FK | FlashKDA kept the KDA prefill state in bf16 and lacked the NaN and fence fixes | [#58846](https://github.com/vllm-project/vllm/pull/58846) | KDA prefill |

The same audit established three things that were *not* defects:

- the TP=3 head padding is inert (padded lines carry no signal);
- the chat template is byte-identical to the official one;
- the harness of our own deployment was dropping the model's reasoning between tool turns (a client
  setting, not an engine fault).

**R2 and R3 act only when the context exceeds `index_topk`.** That is consistent with the 12 September
promotion of `index_topk` 8192 having reduced the long-session tool-call corruption of
[issue #7](https://github.com/NNNtrance/GLM-5.3-Flash-EXL3-TP3-3x-DGX-Spark/issues/7) without explaining it
([11](11-open-issues.md) §2.36). Whether R2/R3 and the wrong routing *are* that residual has not been
measured: the long-session replay has not been re-run on the corrected stack `[not tested]`.

On the old stack these four were fixed on 4 October and production went back to `index_topk` 2048 (the
model's default). That configuration was never published here, and the stack below replaces it.

## 3. The new stack

Upstream `main` already carries R2, R3 and the FlashKDA fixes. What a GB10 at TP=3 still needs is in the
fork and in fourteen patch scripts, applied at image build time by a prelude:

- the TP=3 head and shared-expert padding;
- the DFlash2 drafter at 36/9 heads;
- the expert-parallel weight filter;
- the full-scope checkpoint loader;
- the fast-load sidecar;
- the vision tower at three ranks;
- the fail-closed `glm47` tool-call parser;
- the R1 port;
- the K-pool initialisation.

Each patch matches its anchors exactly and fails closed when an anchor stops matching.

| Component | Revision |
|---|---|
| Base image | `vllm/vllm-openai@sha256:6f0d5e677145fb4003891a994f11fed993d022d141c71844157d3e84e55b1e0a` (arm64; vLLM `0.30.1rc1.dev709+g21d93d0d8`, torch 2.13.0+cu130, Python 3.12.3) |
| `cuda-exl3` | [`NNNtrance/cuda-exl3` `tp3-vllm-main`](https://github.com/NNNtrance/cuda-exl3/tree/tp3-vllm-main), production build `448f1d6` (four commits on upstream `6a1ffc3`; the branch tip `6f3b6dc` changes only a test default) |
| Patches | 14 scripts + prelude, [`tracks/tp3-main/patches/`](../tracks/tp3-main/README.md) |
| Checkpoint | `turboderp/GLM-5.3-Flash-exl3`, branch `4.05bpw` (full scope), as in [13](13-full-scope-checkpoint.md) |
| Drafter | DFlash2, padded to 36/9 at TP=3, as in [04](04-dflash2-port.md) |

The commits in the fork:

| Commit | What |
|---|---|
| `f3e184a` | The TP=3 production pieces folded into the plugin: padding, expert parallelism, mixed linear layers, the MLA KV layout. Ported to vLLM `21d93d0d8` |
| `1451b1a` | Load the MoE router bias, and refuse to start without it |
| `42474b4` | Clamp the routed experts' SwiGLU at the model's `swiglu_limit` |
| `448f1d6` | Two measurement knobs for the dense GEMM tuner: `CUDA_EXL3_GEMM_TUNE_REPS`, `CUDA_EXL3_GEMM_TUNE_VERBOSE`. Off by default, so the default build behaves as before |

The plugin's test suite reads **620 / 620** on the production image. The bias-fix tests are red on the old
`moe.py` (12 of 14) and green here.

## 4. Production settings, and why each one is what it is

Settings: `gpu-memory-utilization` 0.84, `max-model-len` 1,000,000, `max-num-seqs` 5,
`max-num-batched-tokens` 2048, block 256, fp8 KV, `index_topk` 2048, FlashKDA prefill, DFlash2 k=7, vision
16 images and 4 videos per request. The full launcher is in [`tracks/tp3-main/`](../tracks/tp3-main/README.md).

### 4.1 Memory fraction 0.84, not 0.85

- **0.85 gave a KV pool of 6,354,223 tokens.** Under the worst request we allow — sixteen large images,
  126,797 prompt tokens, with five users — the head node fell to **882 MiB** free. The watchdog's 1 GiB
  line stopped the engine.
- **0.84 gives 6,188,010 tokens** (6,190,735 at autostart). It held the head at 1,621 MiB on the same
  load, and the sixteen images came back correct in 84.7 s.
- **What it cost:** 2.6 % of the pool.
- The image load also slows the other users to 0.37–0.43× while it prefills. That is a product behaviour of
  one huge prefill, not memory, and it is the same at both fractions.
- Source: [`memory-fraction-084.md`](../results/main-stack/memory-fraction-084.md)
  `[measured-here, private harness]`.

### 4.2 `disable_eagle_block_drop: true` — prefix hits for agent turns

With a drafter on, vLLM drops the last matched prefix block, and here the scheduler block is 3,328 tokens.
An agent's follow-up turn therefore re-read up to two whole blocks it had just computed. On a short tool
prompt it read nothing from the cache at all.

DFlash2's draft context is aligned token-for-token with the target, so the drop protects nothing here:

- the hit becomes ⌊P/256⌋·256;
- the first token of follow-up turns went from **2.34 to 1.24 s** serial, and **7.25 to 3.83 s** with five
  sessions at once;
- tokens re-read fell **61 %**;
- warm and cold answers agree at seven probe points;
- a 480-turn strict tool-call run had **0** corrupted turns;
- speed over six boots was ×1.0013 at one user and ×0.9993 at four.

**What it cost:** none found; it was looked for. It is a configuration key, not a code change
([`prefix-hits-nodrop.md`](../results/main-stack/prefix-hits-nodrop.md)) `[measured-here, private harness]`.

### 4.3 Draft length by running requests: `[[1,1,7],[2,8,3]]`

`num_speculative_tokens_per_batch_size` drafts 7 tokens when one request is running and 3 when two to eight
are. At four users the step falls **160.2 → 114.5 ms**. The pooled output rate rises **+10.7 / +15.6 /
+7.1 %** at two, four and five users, and one user is unchanged.

| Content | Gain at 2+ users |
|---|---|
| English prose | +25 to +32 % |
| Short Turkish | +31 to +34 % |
| Code | −7 to +2 % |
| JSON | −8 to +10 % |
| Mathematics | **−12 to −17 %** |

**What it cost:** mathematics, which accepts long drafts and loses them at 3; we took that price
([`draft-schedule.md`](../results/main-stack/draft-schedule.md)) `[measured-here]`.

### 4.4 Vision

The vision settings:

- 16 images and 4 videos per request;
- `--mm-encoder-tp-mode data`;
- `max_pixels` 12,544,000 (about 8,000 tokens per image);
- `--mm-processor-cache-gb 0`.

The tower placement is as in [18](18-vision-at-three-ranks.md). Sixteen large images are the case that set
the memory fraction (4.1).

## 5. The measurement protocol changed: reboot before every arm

On GB10, each engine start and stop leaves unified memory more fragmented. The count of free 32 MB blocks
fell from about 3,700 to 240–370 after ten sessions, and read bandwidth fell with it: **−6.6 / −10.4 /
−10.2 %** on the three nodes, −4 to −7 % after a single session.

- `compact_memory` does not restore it. A reboot does, in about 100 s.
- An A/B run late in a session of restarts is therefore biased against whichever arm runs last.
- One rejected candidate (§6) was first measured against a stale baseline, and that made a −2.56 % gain look
  like −0.6 %.

The protocol for every comparison on this page:

- all three nodes rebooted before each arm;
- a bandwidth probe at the start and end of a chain;
- controls from the same session;
- source: [`bandwidth-fragmentation.md`](../results/main-stack/bandwidth-fragmentation.md)
  `[measured-here, private harness]`.

Production is not affected: it boots once.

The same night produced three pre-registered decision rules that turned out to be flawed:

- an absolute count where a ratio was meant;
- a band narrower than the boot-to-boot spread;
- a self-test that failed.

Each time the literal verdict was neither applied nor reversed. We wrote down the flaw, wrote a corrected
rule before looking at new data, and ran a confirmatory arm. All are listed in
[`quality-gates.md`](../results/main-stack/quality-gates.md) §6.

## 6. Tried and rejected

**A drafted CUDA graph through FlashInfer XQA.** This is our patch for the cause we reported as
[vllm#55581](https://github.com/vllm-project/vllm/issues/55581): the attention metadata class reads the
target's head count, 22, where the drafter's KV head count is 3.

- The patch was correct (22/22 tests).
- Against the same-night control it was **−2.56 %** step time at one user and **−1.35 %** at four.
- It cost **6.3 %** of the KV pool, 0.05–0.25 s on the first token of short prompts, and 5.5× the swap
  traffic at boot.
- Rejected, and kept in an archive in case the trade changes
  ([`rejected-draft-cuda-graph.md`](../results/main-stack/rejected-draft-cuda-graph.md)) `[measured-here]`.

## 7. Operations

- **Autostart.** A systemd unit and its preflight bring the engine up from power-on in **274 s** to
  `/health` 200 on all three nodes. The preflight checks docker, ConnectX-7 4/4, fabric peers and a memory
  settle. The MoE boot gate reads 42/42 × 3.
- **Watchdog (optional).** It runs on the head node every 60 s.
  - It probes `/health` plus an 8-token completion, and treats advancing token counters as busy rather
    than stuck.
  - After three consecutive failures it reboots the two workers and then the head. One node alone cannot
    be rebooted: its peer's fabric port would die.
  - It is rate-limited to once per 30 minutes and three times per six hours, then raises an alarm file.
  - Drill: an engine killed on a worker came back, unattended, **8.2 minutes** after the kill.
  - **What it costs:** a false positive reboots the whole cluster; the busy exemption and the rate limit
    are the guard.
  - Source: [`boot-and-watchdog.md`](../results/main-stack/boot-and-watchdog.md)
    `[measured-here, private harness]`.
- **Soak.** Eight hours at five concurrent requests, all types mixed with images and ~100k-token
  documents.
  - **7,515 requests, 0 errors**, the engine never restarted, free-memory floors 4.67 / 6.93 / 7.02 GiB,
    swap 0, no Xid.
  - Speed hour by hour stayed between 0.996 and 1.052 of the first hour.
  - The engine's own counters agree: 8,451 completions, 0 errors, 0 preemptions.
  - Source: [`soak-8h.md`](../results/main-stack/soak-8h.md) `[measured-here, private harness]`.

## 8. Open

- **Quality benchmarks on the corrected build** — GSM8K, IFEval, MMLU, tool-eval-bench, the 1M needle, the
  code exam, the probe `[not tested]`. Until then, the KL health check is the only quality evidence for this
  stack.
- **The long-session tool-argument corruption** ([issue #7](https://github.com/NNNtrance/GLM-5.3-Flash-EXL3-TP3-3x-DGX-Spark/issues/7))
  at `index_topk` 2048 on the corrected stack `[not tested]`. The 480-turn strict gate and a 60k needle
  passed, but neither is the replay that found it.
- **The two-node track** still runs the old stack, with the router bias missing. Moving it is not
  started.
- **GEMM tuner picks differ across ranks** for 25 of 113 dense shapes on one boot. That is why the
  measurement knobs exist; the consequence has not been measured.
- **The SwiGLU clamp's reference question.** Both builds sit nearer the unclamped reference than the
  clamped one on every row. The official modelling code's exact clamp placement for routed, shared and
  dense experts needs a second read.
- **The bias fix upstream:** [Zeuss5/cuda-exl3#8](https://github.com/Zeuss5/cuda-exl3/pull/8) is open.
