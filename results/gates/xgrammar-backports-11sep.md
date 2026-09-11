# The xgrammar backports (#52805 + #53046) — the gates that adopted them (11 September 2026)

**Applies to: TP=3.** The patch is not registered in the TP=2 prelude and has not been run there
`[not tested]`.

The patch, its provenance and the two upstream bugs are
[`tracks/tp3/patches/xgrammar-backports/`](../../tracks/tp3/patches/xgrammar-backports/README.md);
the mechanism is written up as [docs/14](../../docs/14-troubleshooting.md) §9.14. This page is the
acceptance run. **Nothing here is a speed measurement** — neither hunk touches a compute path, no
speed number was expected to move, and none was read as evidence.

**Settings.** Three DGX Spark (GB10, sm\_121) nodes over the ConnectX-7 mesh, image
`exl3-zeus:754421f` (vLLM `0.1.dev20051+g487ecf187`), checkpoint `turboderp/GLM-5.3-Flash-exl3`
branch `4.05bpw` (full scope), **TP=3 + expert parallelism**, DFlash2 draft at k=7 with an fp8 draft
cache, KV dtype fp8, `gpu-memory-utilization` **0.88**, `max-model-len` **1,000,000**,
`--block-size 256`, `--max-num-batched-tokens 2048`, `--max-num-seqs 8`, `enforce_eager`, vision tower
on, `HAREM_KDA_FLASHKDA=1`, `HAREM_PREFIX_HIT=1`, `HAREM_KPOOL_TAIL_FIX=1`,
`HAREM_GLM47_FAILCLOSED=1`, `HAREM_SM12_ITEMS=pdl,kpool`, `HAREM_INDEXER_WS_MODE=bound`,
`clear_thinking: true`, `reasoning_effort: low`. That is the production configuration of 11 September
morning with the two backports added — and the knob defaults on, so the patch being present is the
change. The **required-field** arm of the fail-closed parser (`HAREM_GLM47_REQUIRED`, default on) went
into production in the same boot. Measured 11 September 2026, 19:18–19:34 local, nothing else on the
cluster.

---

## 1. The engine log — the measurement this patch exists for

Same strict protocol both sides: `scripts/strict-proxy.py` in front of the engine setting
`tools[i].function.strict = true`, and `scripts/toolcall-gate.py` at 4 sessions × 30 turns,
concurrency 2, `--max-prompt-tokens 120000`, effort low.

| | **before** — 11 h production boot, 08:50–19:06 | **after** — 120-turn strict window, 19:25–19:34 |
|---|---|---|
| `Failed to advance FSM ... Please file an issue.` at `ERROR` | **776** | **0** |
| `grammar_matcher.cc:612` warnings | **872** | **0** |
| distinct requests affected | 102 | **0** |
| **total `ERROR` lines in the whole log** | 776 | **0** |

The before-window is longer and carries other traffic, so **the two counts are not a controlled
pair** — what is load-bearing is that the *after* number is **zero over the same protocol that
produced the bursts**, and that the warning text upstream prints ("the matcher has terminated after
accepting the stop token, but is trying to accept new token") names #52805's bug outright.

## 2. The gates

| Gate | Result | Time |
|---|---|---|
| `scripts/correctness-probe.py` — model knowledge, both fields | **10/10** | 19:28 |
| the same, content only, what a client actually sees | **9/9**, requests with empty content **0** | 19:28 |
| `scripts/code-exam.py` | **12/12 (100.0 %)** | 19:29 |
| `scripts/toolcall-gate.py` via `strict-proxy.py`, 4 × 30, concurrency 2 | see §3 | 19:25–19:34 |

Needle-lite and the vision arm were **not** re-run in this window `[not tested]`: neither hunk can
reach a KV path or the vision tower, the boot's KV pool is byte-identical to the boot they last passed
on, and the session was time-boxed. They remain part of the standard set and should be run before the
next promotion that *can* touch them.

## 3. The strict tool-call gate

| context_bucket | turns | tool_calls | wellformed | rejected | schema_violation | json_error | empty_turn | repeat | length |
|---|---|---|---|---|---|---|---|---|---|
| 0–10k | 13 | 15 | 15 | 0 | 0 | 0 | 0 | 0 | 0 |
| 10–20k | 35 | 40 | 40 | 0 | 0 | 0 | 0 | 0 | 0 |
| 20–30k | 51 | 56 | 56 | 0 | 0 | 0 | 0 | 0 | 0 |
| 40–50k | 12 | 11 | 11 | 0 | 0 | 0 | 0 | 0 | 0 |
| 50–60k | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| 60–70k | 5 | 4 | 4 | 0 | 0 | 0 | 0 | 0 | 0 |
| 70–80k | 2 | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| 90–100k | 1 | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 0 |
| **TOTAL** | **120** | **132** | **132** | **0** | **0** | **0** | **0** | **0** | **0** |

Sessions reaching first corruption **0 / 4**; harness errors 0; highest prompt_tokens **97,319**.
That is the same shape as the 11 September strict arm run *before* the backports (120 turns, 133
calls, 133 well-formed — [`toolcall-gate-11sep-strict.md`](toolcall-gate-11sep-strict.md)): the
backports were never expected to change tool-call correctness, and they did not. What changed is the
log.

## 4. Memory and boot

| | value |
|---|---|
| KV pool, sidecar-less boot | **7,063,360 tokens** (max concurrency 7.06× at 1 M) |
| KV pool before the change | 7,063,360 tokens — **unchanged** |
| block size / KV dtype / prefix caching | 256 / fp8 / on — unchanged |

Because the patch set changed, the existing fast-load sidecar was invalid, so the rollout cost the
documented three boots: **sidecar-less 5 min 08 s**, then a `dump` boot, then a `load` boot on the new
sidecar `/var/tmp/glm53-exl3-xgrammar-r{0,1,2}`. The previous sidecar
`/var/tmp/glm53-exl3-failclosed-r*` is **kept on disk** as the rollback target.

## 5. Rollback

1. Cheapest: `HAREM_XGRAMMAR_BACKPORT=0` in `EXTRA_ENV`, three-node restart. The sidecar stays valid
   (its identity is keyed on the `patch-*.py` set and the prelude text, not on `EXTRA_ENV`).
2. Full: restore `tp3-prelude.sh.bak-11eyl-xgrammar` and `.env.tp3.bak-11eyl-xgrammar` on all three
   nodes — which also restores `FASTLOAD_DIR=/var/tmp/glm53-exl3-failclosed` / `FASTLOAD_MODE=load` —
   and restart all three together. `patch-glm47-failclosed-tp3.py.bak-11eyl-required` reverts the
   required-field arm separately.

Never restart one node alone: it kills the fabric port on its peers.

## 6. What this run does not establish

- The before/after FSM counts are **not a controlled pair** (different window lengths, different
  traffic mixes). Zero is the claim; "776 → 0 under identical load" is not.
- Needle-lite and vision were not re-run `[not tested]`.
- No speed or acceptance-rate measurement was taken, so "speed unchanged" here is an **expectation
  from the code path**, not a measurement `[not measured]`.
- The #53046 hunk is covered behaviourally only by the production FSM count; its unit coverage in
  `test_xgrammar_backports.py` is structural.
