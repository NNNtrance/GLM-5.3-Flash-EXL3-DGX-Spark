# `tracks/tp2/patches/vision` — the vision tower at two ranks

**Applies to: TP=2 only.** In the two-node recipe as of 8 September 2026: **4 images + 2 videos per
request** on the full-scope EXL3 checkpoint at two ranks `[measured-here]`.

**There are no patch files in this directory, and that is deliberate.** The vision patch and its
three model-free gates are the three-node track's files, and they are used at two ranks **byte for
byte**: nothing in any of them reads the rank count. Two copies of a file are a coin flip unless
something checks ([docs/08](../../../../docs/08-fast-boot.md) §12), so this page is a pointer and an
install command rather than a second copy.

The full two-rank account — what divides, why we replicated the tower anyway, and the one anchor
dependency that made the two-node tree grow a file — is
[docs/19](../../../../docs/19-vision-at-two-ranks.md). The three-node account, which is where all the
*mechanism* is, is [docs/18](../../../../docs/18-vision-at-three-ranks.md).

---

## Install

Copy the four files and the fixtures directory out of the three-node tree into your two-node tree,
**before the dump boot** — a file added to a patch directory changes the fast-load manifest identity
and refuses the next boot ([docs/08](../../../../docs/08-fast-boot.md) §4):

```
cp tracks/tp3/patches/vision/patch-vision-tp3.py tracks/tp3/patches/vision/check-vision-mapping.py tracks/tp3/patches/vision/check-vision-names.py tracks/tp3/patches/vision/check-video-geometry.py "$TREE"/
```

And one more, which is the part a two-node reader will not expect:

```
cp tracks/tp3/patches/patch-vllm-tp3.py "$TREE"/
```

**`patch-vllm-tp3.py` is required for vision at two ranks.** VS1's anchor is written against the text
that file's edit 4b leaves behind, not against the stock `model.py`. Its padding half stays the no-op
it always was at TP≤2 (`lcm(128, 2) = 128`; vocab 154,880 and the shared expert's 2,048 are already
multiples of 128); its edits 4a/4b/4c are the `--language-model-only` wrapper, which the two-node
track wants in its own right. The arithmetic, and why forking the vision patch instead was rejected,
is [docs/19](../../../../docs/19-vision-at-two-ranks.md) §2.

The video-geometry gate reads two short fixture clips that are not in this repository; make your own
two, or let the gate skip them and run its analytic 1080p case alone (it says which it did).

## The prelude block

[`tracks/tp2/patches/tp2full-prelude.sh`](../tp2full-prelude.sh) already carries it, after the
full-scope block and before `flashinfer-warmup.py` — the same slot the three-node prelude uses. It is
gated on `HAREM_VISION=1`; with the knob unset the tree is candidate C byte for byte.

## The environment, exactly

Six lines against the two-node candidate C template
([`tracks/tp2/env.tp2-full.example`](../../env.tp2-full.example) carries them):

```
LANGUAGE_MODEL_ONLY=0
```

```
EXTRA_ENV += HAREM_VISION=1 CUDA_EXL3_PACKED_MAPPING={"qkv_proj":["q_proj","k_proj","v_proj"]}
```

```
EXTRA_ARGS += --mm-encoder-tp-mode data --mm-processor-kwargs {"max_pixels":12544000,"max_image_tokens":8000} --limit-mm-per-prompt {"image":4,"video":2} --mm-processor-cache-gb 0
```

**No spaces inside any JSON** — `EXTRA_ENV` is word-split and the shell eats double-quoted JSON.

**`--mm-encoder-tp-mode data` is the one line this page argues about.** At three ranks it is
mandatory: nothing in the tower divides by three. At two ranks everything divides — heads 16/2 = 8,
`attn.proj` 512 = 4 × 128, the MLP and merger 2048 = 16 × 128, the merger's context 5120 = 40 × 128 —
so the flag is **optional**, and we shipped it anyway. It replicates the 0.557 GiB tower on every
rank instead of slicing a 6-bit EXL3 tower across ranks, which is a path this repository has measured
at no rank count. The sliced tower is `[not tested]` and is
[HELP-WANTED](../../../../HELP-WANTED.md) material; the reasoning is
[docs/19](../../../../docs/19-vision-at-two-ranks.md) §1.

**Decide `data` on or off before the dump boot.** The sidecar stores post-load per-rank tensors, and
a replicated tower and a sliced one are different tensors. The fast-load identity does not hash the
command line, so switching this flag after a dump is the kind of change that restores shapes nobody
checked.

## The model-free gate

[`tracks/tp2/patches/verify-cpu.sh`](../verify-cpu.sh) runs the **whole two-node prelude order** in a
throwaway CPU container — no `--gpus`, no engine, safetensors headers only — and must print six
PASS/applied lines. A gate that does not reproduce the boot's patch order is not a gate; that lesson
cost three boots at three ranks ([docs/18](../../../../docs/18-vision-at-three-ranks.md) §13).
