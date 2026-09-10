# `patches/flashkda` — the fused KDA prefill kernel, and the sidecar it invalidates

**In the recipe since 10 September 2026.** A port of vLLM
[#55737](https://github.com/vllm-project/vllm/pull/55737) onto the pinned vLLM this stack serves
(`487ecf187`, 25 August 2026): GLM-5.3-Flash's KDA **chunked prefill** runs through the fused
`vllm._flashkda_C` kernel instead of the ten-kernel Triton `chunk_kda_with_fused_gate` chain. One
file, five anchors, one environment knob, default off.

| | Knob | What it does | Status |
|---|---|---|---|
| **FlashKDA KDA prefill** | `HAREM_KDA_FLASHKDA=1` | the fused kernel for KDA chunked prefill: **+6.5 % sustained prefill**, **−5.1 % TTFT** at 7K, decode and KV pool unmoved | **In the recipe**, 10 September |
| | unset / `0` | **upstream behaviour, byte for byte** — one environment read per KDA layer at construction, nothing else | the A/B's control arm |

Everything measured is in
[`results/gates/flashkda-ab-10sep.md`](../../../../results/gates/flashkda-ab-10sep.md). The standing
open item — the gain is **1.6× the kernel share that predicts it** and nothing in hand explains the
excess — is [docs/11](../../../../docs/11-open-issues.md) §2.35.

**The most useful thing on this page is not the kernel.** It is that registering one new
`patch-*.py` in the patch directory invalidates every fast-load sidecar on every node, which means
an A/B of a new patch cannot be run on the production boot path at all — and that the `ExecStartPre`
gate guarding those sidecars demanded one in `dump` mode, the mode whose whole job is to create it.
That chicken-and-egg cost **18 minutes of downtime**. Both are §4 and §5 below, and
[docs/14](../../../../docs/14-troubleshooting.md) §10.6 carries them as a failure entry.

---

## What is here

| File | What it does |
|---|---|
| [`patch-flashkda-tp3.py`](patch-flashkda-tp3.py) | The patch. Five exact-text anchors in `vllm/models/glm5next/nvidia/kda.py`, each required **exactly once**: the import block, the backend resolver after `_cast_sigmoid`, the end of `__init__` (where the three workspace buffers are sized), the kernel wrapper before `forward`, and the chunked-prefill call itself. Gated on `HAREM_KDA_FLASHKDA`; `additional_config["kda_prefill_backend"]` is honoured too and wins, so upstream's own knob keeps working |
| [`register-flashkda.py`](register-flashkda.py) | Inserts the prelude block below through the prelude's **existing inode** — inside the patch tree `tp3-prelude.sh` is a hard link to `tp3full-prelude.sh`, one inode, precisely so the two names cannot drift ([`../README.md`](../README.md)). Idempotent; `--undo` restores the dated backup, which is the whole of the rollback on the prelude side |
| [`bench_flashkda_vs_chunk.py`](bench_flashkda_vs_chunk.py) | Model-free micro-benchmark at the real shapes: 22 KDA heads per rank, head\_dim 128, bf16, bounded gate, q/k/v **dense**. Runs with the engine up and idle, one short-lived process, and refuses to measure under allocator pressure (`--min-free-mib`) |
| [`bench_prodshape.py`](bench_prodshape.py) | The same comparison with q/k/v as **strided views of one fused qkv buffer**, which is what `_forward` actually hands over — so FlashKDA's three `.contiguous()` copies are inside its timed region. **This is the honest ratio**, and it is smaller than the dense one |
| [`loadgen.py`](loadgen.py) | The prefill stopwatch that produced every end-to-end number: 24 fresh nonce-prefixed ~7,000-token prompts in flight, throughput read as a `vllm:prompt_tokens_total` delta over a 60 s window, with the prefix-cache counters sampled over the same window so a hit cannot hide inside the figure. Also does single-stream TTFT and decode |

**Model-free first, and it paid.** Both benches ran beside the serving engine before anything was
registered anywhere: they established that the sm\_120 cubin in this image runs on GB10 (sm\_121) at
all, what the kernel is worth at the production shape, and that the two paths agree numerically —
output mean |Δ| **1.2e-5**, max **7.3e-4** against a reference whose own |mean| is 2.5e-3, i.e. about
one bf16 ULP; recurrent state mean |Δ| 1.0e-4 against 2.9e-2 `[measured-here]`. PR #55737's own
naive-reference comparison puts both paths at mean 2e-5 / max 4.8e-4 `[reported]`, the same order.

---

## 1. What the code does, and what the patch changes

`kda.py`'s chunked-prefill branch calls `chunk_kda_with_fused_gate(..., safe_gate=True)`, which is a
chain of about ten Triton kernels: the gate cumulative sum, two scaled-dot `kkt` passes, the `wy`
recompute, the chunk `h`/`o` pair, a 16×16 → 64×64 inverse, two l2-norms and a beta sigmoid cast.
`vllm._flashkda_C` is **already compiled into this image** — Kimi-K3 uses it — and implements the
same bounded-gate KDA recurrence, `lower_bound * sigmoid(exp(A_log) * (g + dt_bias))`, with the q/k
l2-norm in-kernel and raw beta logits, in one fused CUDA kernel. That is what makes it a drop-in
rather than a rewrite.

The patch resolves the backend **once per layer at construction**, sizes three workspace buffers from
`max_num_batched_tokens` and `max_num_seqs`, and routes the non-spec prefill segment through the fused
call. A step that also carries speculative-decode tokens keeps the Triton recurrent kernel for those
and writes the prefill segment into a workspace buffer that the existing merge scatters back; a
non-spec step writes **straight into the layer output buffer**, with no merge copy.

**Four ways the upstream diff does not apply as written**, and all four are in the patch's own
docstring:

1. **`torch.ops._flashkda_C.fwd` in this image takes 14 arguments** and ends at `cu_seqlens`. The
   PR's base has two more trailing optionals — `checkpoint_state`, `checkpoint_offsets` — and passes
   `None, None` for both. They are dropped here; passing them raises. This is also the evidence for
   §6's caveat.
2. The PR reads `additional_config["kda_prefill_backend"]`. Our launcher passes no
   `--additional-config`, so the knob would need the unit changed. Replaced by the environment gate,
   with `additional_config` still honoured and still winning when present.
3. Our `A_log` is `[1, 1, H, 1]` fp32 and `dt_bias` is `[H*D]` fp32; both are flattened with
   `.contiguous()` rather than `.view()`. The copy is free and survives a layout change.
4. Our recurrent state dtype is fp32 (`kda_state_dtype(bf16, "auto")` → `(bf16, fp32)`). Measured:
   FlashKDA accepts an fp32 initial and final state.

**Deliberately not ported.** The PR also rewrites the spec/non-spec merge to `index_copy_` straight
into `core_attn_out`, dropping one `torch.empty` and a copy. That is an independent micro-optimisation
of the Triton path as well, so it is left out and this patch changes exactly one thing.

**Why sm\_121 is in scope at all.** The extension in this image carries cubins for **sm\_90a,
sm\_100 and sm\_120 and no PTX** — there is no sm\_121 cubin and nothing to JIT from. It runs anyway
because `cmake/external_projects/flashkda.cmake` builds the 12.x target as a **family** binary
(`12.0f` under CUDA ≥ 13.0), and a family binary is valid across the whole 12.x family including
GB10. That is the reason from the source; the micro-benchmark is the reason from the hardware, and
both are needed — the resolver's `capability.major in (9, 10, 12)` test would have said yes either
way `[measured-here]`.

## 2. What it is worth, kernel and end to end

Kernel only, engine up and idle, one arm per process, 22 heads per rank, bf16, `lower_bound` −5,
one sequence, median of seven iterations after three warm-ups `[measured-here]`:

| tokens in the step | Triton chain | FlashKDA | ratio |
|---|---|---|---|
| 512 | 0.449 ms | 0.152 ms | 2.94× |
| 1,024 | 1.064 ms | 0.326 ms | 3.26× |
| **2,048** (our `--max-num-batched-tokens`) | **2.238 ms** | **0.673 ms** | **3.33×** |
| 4,096 | 4.462 ms | 1.350 ms | 3.31× |
| 8,192 | 9.437 ms | 2.682 ms | 3.52× |

**And the honest ratio is the one below it.** In production q/k/v are strided views of a single fused
qkv buffer; the Triton path eats those strides directly and FlashKDA hardcodes dense ones, so three
`.contiguous()` copies per layer per step go on FlashKDA's side of the ledger `[measured-here]`:

| tokens | Triton, strided input | FlashKDA + the three copies | ratio | of which copies |
|---|---|---|---|---|
| 512 | 0.468 ms | 0.214 ms | 2.19× | 0.034 ms |
| 1,024 | 1.158 ms | 0.466 ms | 2.49× | 0.164 ms |
| **2,048** | **2.467 ms** | **0.967 ms** | **2.55×** | 0.314 ms |

Sequence count barely moves it: at 2,048 tokens the dense ratio is 3.38× / 3.53× / 3.35× / 3.18× for
1 / 2 / 4 / 8 sequences.

**End to end, two matched sidecar-less boots one flag apart** `[measured-here]`:

| | Triton | FlashKDA | delta |
|---|---|---|---|
| Prefill, sustained, 24 fresh ~7K prompts, 60 s window (two runs) | 1,753.5 / 1,754.0 tok/s | **1,867.4 / 1,868.8** | **+6.50 % / +6.55 %** |
| TTFT, single stream, fresh 7K prompt, median of 5 | 4.192 s | **3.978 s** | **−5.10 %** |
| Decode, single stream, 512 tokens | 61.15 tok/s | 61.28 | +0.21 % |
| KV pool | 7,077,134 | 7,099,173 | +0.31 % |

Gates on the FlashKDA arm, cold: correctness probe **10/10** (content-only 9/9, empty content 0),
code exam **12/12** first run, needle-lite **6/6**, vision **PASS**. The full tables, the telemetry
and the two anomalies are in
[`results/gates/flashkda-ab-10sep.md`](../../../../results/gates/flashkda-ab-10sep.md).

**The prediction was wrong in our favour, and that is written down rather than celebrated.** A static
analysis of an existing rank-0 prefill trace puts the kernels FlashKDA replaces at **292.3 ms =
6.19 %** of GPU-busy time, with prefill 98.5 % GPU-bound; combined with the 2.55× production-shape
ratio that predicts **+3.5 … +4.1 %** of end-to-end prefill speed `[estimate]`. The measured figure
is **+6.5 %**. Nothing in hand explains the other 2.5 points.
[docs/11](../../../../docs/11-open-issues.md) §2.35 carries it open.

**What it cost.** Speed: nothing measurable anywhere — decode +0.21 % and TTFT better, both inside
their bands. Memory: the three workspace buffers are **61.45 MiB per rank** at our settings —
`get_workspace_size(2048, 22, 8)` is 39.45 MiB, the `(8, 22, 128, 128)` fp32 recurrent state is
11.00 MiB and the `(1, 2048, 22, 128)` bf16 output buffer another 11.00 MiB — and all 34 KDA layers
**share** the arena rather than each holding one, so the measured KV pool read *higher* on the
FlashKDA arm than on the control. Quality: four gates full, and the two kernels agree to about one
bf16 ULP. The real price is operational and it is §4: one more `patch-*.py` in the identity, which
means a fresh ~53 GiB-per-rank sidecar and a 394 s dump boot, and the old sidecar kept on disk
rather than deleted — 53 GiB × 3 held as the way back. Node free space went 546 G → **493 G**.

## 3. How it is registered, and the exact line

The prelude applies it **unconditionally**, after the full-scope block:

```
run python3 "$TP3_DIR/patch-flashkda-tp3.py" --root "$(dirname "$VLLM_PY")" --in-place
```

Both halves of that line are load-bearing, and each cost a boot to learn.

**`--root` is the dist-packages root, not `$VLLM_PY`.** This script's internal `REL` starts with
`vllm/`, unlike every other patch script in this tree, because it was written to run against a
throwaway copy of the source layout. With `--root "$VLLM_PY"` the path becomes
`.../vllm/vllm/models/glm5next/nvidia/kda.py` and it raises `FileNotFoundError`. The prelude's
fail-closed `run` wrapper then stopped the rank, which is exactly the right behaviour — the failure
was loud and it was ours.

**Without `--in-place` the script is a dry run.** It prints `dry run OK`, exits 0 and patches
nothing. Registered that way it produces a *healthy* boot that quietly keeps running the Triton
chain. An A/B against that is nothing compared with nothing, and nothing in the boot log would say
so — which is why the resolver prints its choice once per process:

```
[HAREM-FLASHKDA] kda_prefill_backend=flashkda   (HAREM_KDA_FLASHKDA='1')
[HAREM-FLASHKDA] kda_prefill_backend=triton     (HAREM_KDA_FLASHKDA='0')
```

Checked on **all three ranks in both arms**. PR #55737 prints nothing equivalent; without a line like
it there is no way to tell from a log which kernel ran. **A boot log with no such line is a boot
where this patch did not run.**

**Order.** After `patch-fullscope-tp3.py`, which is the other arm that edits `kda.py` — its anchors
are in `__init__` and weight loading, these five are the import block, `_cast_sigmoid`, the end of
`__init__`, `forward`'s definition line and the chunked-prefill call, and they do not overlap.
Against the vision block ([`../vision/prelude-vision-hook.sh`](../vision/prelude-vision-hook.sh))
order is immaterial — that one edits `model.py` and `multimodal.py` — and on our nodes FlashKDA comes
first.

## 4. Registering a new patch invalidates the fast-load sidecar — which is why the first A/B never ran

`harem_fastload_id.file_identity()` hashes `glob($TP3_DIR/patch-*.py)`, the **full text** of the
prelude and the overlay module into the sidecar's identity
([docs/08](../../../../docs/08-fast-boot.md) §4). So copying one new file into the patch directory
invalidates every sidecar on every node, and the production boot — `FASTLOAD_MODE=load` — is
refused:

```
preflight-fastload: sidecar stale - boot refused
  patches.patch-flashkda-tp3.py: recorded='<none>' now='4e9a0415...'
  patches.tp3-prelude.sh:        recorded='3b78d02d...' now='be712c25...'
```

The gate is right: that sidecar *was* produced from a different patch set. The same check runs a
second time inside the engine, and neither is a gate to skip. **There is no "leave it registered but
switched off" option**: the knob is read at run time, but the file's presence is hashed at boot.

**What that forces on an A/B.** Both arms were booted with the sidecar **disabled**
(`FASTLOAD_MODE=` empty — the launcher then skips the whole fast-load block and the prelude never
calls the preflight), so the patch is registered in both and `HAREM_KDA_FLASHKDA=1` against `=0` is
the only difference between them. The price is a slower boot in both arms and a boot path that is not
production's; the second is why the promoted configuration was re-measured on the real path
afterwards and read **1,865.7 tok/s**, against the arm's 1,867.4 / 1,868.8.

**And what it forces on promotion.** A new sidecar, in a **new directory**: one
`FASTLOAD_MODE=dump` boot (394 s, 53 GiB per rank) into `FASTLOAD_DIR=/var/tmp/glm53-exl3-flashkda`,
then back to `load` (181 s). The previous sidecar stays exactly where it is —
`/var/tmp/glm53-exl3-prefix-r{0,1,2}`, untouched, 53 GiB each — because reusing the name would
overwrite the rollback in place ([docs/14](../../../../docs/14-troubleshooting.md) §3.4).

## 5. The preflight gate demanded the sidecar in dump mode — 18 minutes down

The first dump-mode restart failed on every node before the container started. `ExecStartPre`
(`motor-onkosul-exl3.sh`, the last check of seven) required `$FASTLOAD_DIR-r$RANK/MANIFEST.json`
**regardless of `FASTLOAD_MODE`**, skipping only when `FASTLOAD_DIR` was empty:

```
fast-load sidecar missing: /var/tmp/glm53-exl3-flashkda-r0
```

Dump mode is precisely the mode that *creates* that directory, so pointing `FASTLOAD_DIR` at a new
path could never boot. All three units went to `failed` and the engine was down from 03:26:28 to
03:44:18 UTC — **17 min 50 s** — while it was diagnosed. No production artefact was overwritten and
the old sidecars were intact throughout.

**The gate was narrowed, not removed.** One clause, and it is in
[`tracks/tp3/motor-onkosul-exl3.sh`](../../motor-onkosul-exl3.sh) as shipped:

```
FM=$(grep -E "^FASTLOAD_MODE=" "$ENVF" | cut -d= -f2)
[ -z "$FD" ] || [ "$FM" != load ] || test -f "$FD-r$R/MANIFEST.json" || { echo "fast-load sidecar missing: $FD-r$R"; exit 1; }
```

`load` is the only mode in which a missing sidecar is fatal: in `dump` the launcher creates the
directory itself, and with the mode empty fast loading is not in play at all. The same one-line change
is in [`tracks/tp2/motor-onkosul-exl3-tp2.sh`](../../../tp2/motor-onkosul-exl3-tp2.sh), which carried
the identical clause — **not measured at two ranks** `[not tested]`, the same edit for the same
reason.

**How to plan a patch A/B after this.** Sidecar-less symmetric arms for the comparison; then one dump
boot into a **new** `FASTLOAD_DIR` for the winner; then `load`; keep the old sidecar until the
configuration is retired, and check the disk before you start — a dump costs ~53 GiB per node, one
rank each, on top of what is already there.

## 6. The caveat, and the tag we had wrong

**The `_flashkda_C` in this image is older than the upstream state PR #55737 was validated against,
and kernel-side fixes since then are not in our build.** The extension is dated **26 August 2026**,
and the vLLM revision the image reports — `0.1.dev20051+g487ecf187`, 25 August 2026 — pins it at
**`b5d11010`** (28 July 2026) in `cmake/external_projects/flashkda.cmake`. Upstream had moved to
`3b225bf` (2 September 2026) by the time the PR was opened: **seven commits**, including a missing
proxy-fence fix around the TMA accesses and the replacement of an fp16 Neumann inverse.

**An internal note had recorded our build as `3b225bf`, read off a newer checkout of the vLLM tree
rather than the pinned commit, and that was wrong.** The decisive evidence is in §1 item 1: our
`_flashkda_C::fwd` takes **14 arguments** and has no `checkpoint_state` / `checkpoint_offsets` tail,
and those parameters arrive in FlashKDA on 6 August — after `b5d11010` and well before `3b225bf`. So
the image cannot be carrying the later tag. Nothing published was affected, because nothing had been
published; it is recorded here because a revision read from the wrong checkout is exactly the class
of mistake this repository's own rule about naming commits exists to prevent.

What follows for a reader: this is measured on a **July** kernel. A newer build may be faster, may be
more correct, and has not been tried here `[not tested]`.

## 7. Rollback

- **One line:** delete `HAREM_KDA_FLASHKDA=1` from `EXTRA_ENV`. The resolver reads the knob at
  construction, so the patched image takes the Triton path — and **the sidecar stays valid**, because
  the knob is not part of the identity and the file has not moved.
- **The patch out of the prelude:** `register-flashkda.py --undo` restores the dated backup through
  the same inode. That changes the identity, so it needs the previous sidecar back as well: point
  `FASTLOAD_DIR` at `/var/tmp/glm53-exl3-prefix` and remove the file from the patch directory.
- **The whole configuration:** start with `ENV_FILE` pointing at the dated backup of the previous
  env file. It names the sidecar directory and the `EXTRA_ENV` line together, which is what makes the
  pair a single move. No dump boot.
- The **image never changes**: this patch runs inside the container at boot, against the writable
  layer, and leaves no trace when the container is removed.

---

## Credits

- **[JaredforReal](https://github.com/JaredforReal)** — vLLM
  [#55737](https://github.com/vllm-project/vllm/pull/55737), the change itself: the backend resolver,
  the FlashKDA prefill wrapper, the workspace-manager buffers and the spec/non-spec scatter scheme.
  Our five anchors are his diff adapted to a tree that predates it; his GB300 numbers and his
  accuracy runs are in the pull request. He is also the author of the GLM-5.3-Flash support this whole
  stack is built on (vLLM [#53906](https://github.com/vllm-project/vllm/pull/53906), submitted by
  [ZJY0516](https://github.com/ZJY0516)).
- **[vllm-project/FlashKDA](https://github.com/vllm-project/FlashKDA)** — the kernel. Our image's
  `vllm._flashkda_C` is built from tag **`b5d11010`** (28 July 2026) as pinned by
  `cmake/external_projects/flashkda.cmake` at vLLM `487ecf187`; upstream is at `3b225bf` (2 September
  2026). Nothing of it is vendored here — it was already compiled into the base image for Kimi-K3.

Our patch script, the registration script and the three instruments were written for this recipe; use
freely (Apache-2.0), a credit is appreciated. Full entries in
[CREDITS.md](../../../../CREDITS.md).

---
