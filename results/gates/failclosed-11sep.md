# The fail-closed glm47 tool-call parser — the gates that adopted it (11 September 2026)

**Applies to: TP=3.** The patch is not registered in the TP=2 prelude and has not been run there
`[not tested]`.

The parser change itself, the cascade it ends and the prelude line are
[`tracks/tp3/patches/glm47-failclosed/`](../../tracks/tp3/patches/glm47-failclosed/README.md); the
mechanism was root-caused by YuXiaoPan in issue #7. This page is the acceptance run: **four gates
full, two tool-call smoke tests, one dump boot.** Nothing here is a speed measurement — this patch
touches the tool-call parser and no compute path, and no speed or KV number was expected to move or
was read as evidence.

**Settings.** Three DGX Spark (GB10, sm\_121) nodes over the ConnectX-7 mesh, image
`exl3-zeus:754421f` (vLLM `0.1.dev20051+g487ecf187`), checkpoint `turboderp/GLM-5.3-Flash-exl3`
branch `4.05bpw` (full scope), **TP=3 + expert parallelism**, DFlash2 draft at k=7 with an fp8 draft
cache, KV dtype fp8, `gpu-memory-utilization` **0.88**, `max-model-len` **1,000,000**,
`--block-size 256`, `--max-num-batched-tokens 2048`, `--max-num-seqs 8`, `enforce_eager`, vision tower
on, `HAREM_KDA_FLASHKDA=1`, `HAREM_PREFIX_HIT=1`, `HAREM_KPOOL_TAIL_FIX=1`,
`HAREM_SM12_ITEMS=pdl,kpool`, `HAREM_INDEXER_WS_MODE=bound`, `clear_thinking: true`,
`reasoning_effort: low`. That is the production configuration of 10 September with
`HAREM_GLM47_FAILCLOSED=1` added — and the knob defaults on, so the patch being present is the
change. Measured 11 September 2026, 08:22–08:26 local, nothing else on the cluster.

---

## 1. The gates

Run with the patch armed (`failclosed-on`), against the production engine on the API node:

| Gate | Result | Timestamp |
|---|---|---|
| `scripts/correctness-probe.py` — model knowledge, both fields | **10/10** | 08:22:47 |
| the same, content only, what a client actually sees | **9/9**, requests with empty content **0** | 08:22:47 |
| `scripts/code-exam.py` | **12/12 (100.0 %)** | 08:22:55 |
| `needle-lite6.py` — depths 0.60 / 0.80 / 0.97 at n=54,693 | **6/6** sequential, all three depths PASS | 08:23:07 |
| vision — red disc, radius 121 px | **PASS**, colour ✓ shape ✓ | 08:26:00 |

`GATES DONE 08:26:09`. The needle answers were `TV5-6630`, `BN2-3374` and `LP9-8105` at the three
depths; the vision arm returned "A large red circle centered on a white background" with both
assertions true.

**Read the second row, not just the first.** `correctness-probe.py` scores the model's knowledge by
reading `content` **and** `reasoning_content`; the content-only column is the one a client sees, and
it is the column this patch could have broken, because the patch changes what the parser puts into
`content`. **Zero requests with empty content** is the specific assertion that matters here: an empty
turn is the symptom issue #7 reported, and the gate now counts them rather than trusting that they do
not happen.

## 2. The two smoke tests

Gates above are single-turn and carry no tool calls at all, so they cannot say whether a **valid**
tool call still works. Both paths were checked by hand after the boot, and both passed:

- **Non-streaming.** A valid tool call returns as a `tool_calls` entry with schema-correct JSON
  arguments, unchanged from before the patch.
- **Streaming.** A valid tool call now arrives as **one** `tool_call` delta instead of a name delta
  followed by a run of argument deltas. **This is a visible API behaviour change, not a side effect** —
  it is how the patch prevents a call that later turns out to be malformed from having already
  half-reached the client. A client that renders partial argument text as it streams will see the
  arguments appear at once instead of growing; a client that parses completed tool calls sees no
  difference. If that matters to a deployment, `HAREM_GLM47_FAILCLOSED=0` is the restart-only way
  back.

The boot log carried the gate line once per process:

```
[HAREM-GLM47-FAILCLOSED] failclosed=True (HAREM_GLM47_FAILCLOSED='')
```

A boot without it is a boot where the patch did not run — the missing-`--in-place` dry-run trap of
the patch README §4.

Before any of this, `test_failclosed.py` passed **37/37** inside the image on a node, model-free, in a
throwaway container: the captured corruption vectors from issue #7 on both the streaming and
non-streaming paths, the unclosed-call shape that produced the empty turn, a hallucinated tool name, an
out-of-schema key, valid calls, plain text, the finish sweep, and the `=0` arm where the bug is
deliberately reproduced.

## 3. The boot

Registering a new `patch-*.py` invalidates every fast-load sidecar on every node
([docs/08](../../docs/08-fast-boot.md) §4, [docs/14](../../docs/14-troubleshooting.md) §10.6), so this
adoption cost a full load into a **new** sidecar directory: **380 s** in `FASTLOAD_MODE=dump`, after
which the engine went back to `load`. The previous (FlashKDA) sidecar is kept untouched, so the
rollback is one environment file and no second dump.

## 4. What these gates do not establish

**This patch cannot change the rate at which the model writes a first malformed tool call.** It is a
parser change, downstream of generation; the rate is a property of the checkpoint, the context length
and the sampling temperature. Whether the 4bpw EXL3 checkpoint raises that rate against the NVFP4
sibling is an **open measurement** — [docs/11](../../docs/11-open-issues.md) §2.30 — and issue #7's own
data cannot separate the build from the weights either.

What the patch changes is the consequence: one bad call used to seed a loop that ended the session, so
the rate had to be near zero to be survivable; now it costs one visibly ugly turn. **Only the cascade
is broken**, and no number on this page should be read as evidence about the first corruption.

**Since 12 September part of that rate has an address**: raising the sparse indexer's `index_topk` from
2048 to 8192 took a 30-turn replay at 30–41k prompt tokens from 11 bad turns to 2, with speculative
decoding, FlashKDA, fp8 KV and the grammar each cleared in their own arm
([`index-topk-8192-12sep.md`](index-topk-8192-12sep.md)). It does not settle the checkpoint question
above, which is a comparison against the other stack.

Every gate here is also still **single-turn**, which is the whole blind spot this failure lived in. The
multi-turn gate that would catch this class directly does not exist yet:
[HELP-WANTED](../../HELP-WANTED.md) §14, which now wants a tool-call variant as well as a text one.
