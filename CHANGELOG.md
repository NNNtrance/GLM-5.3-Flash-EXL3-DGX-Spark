# Changelog

Dated entries for the stack this repository documents. This is a working record, not a release
history — it is overwritten in place as the work continues.

Speed figures are aggregate output tok/s on `scripts/hizset-v2.jsonl` at `reasoning_effort: low`,
temperature 0. Entries up to and including *production configuration 4* are medians of sweep rounds
3–5 with rounds 1–2 discarded; from *production configuration 5* onwards they are medians of three
rounds, which is what the persisted MLA tuner cache bought — see
[docs/09](docs/09-measurement-protocol.md) §1 and [docs/12](docs/12-tuner-cache.md).

---

## 11 September 2026 — measured: 118 tool calls to 100.5k tokens on the fail-closed build, 0 malformed XML ([results/gates/toolcall-gate-11sep.md](results/gates/toolcall-gate-11sep.md))

---

## 11 September 2026 — issue #7: the malformed tool-call cascade, and a fail-closed glm47 parser in production

**The third and last mechanism of issue #1 is closed, and it was not ours to find.** YuXiaoPan caught
the residual "premature end-of-turn" on agentic traffic live behind a logging proxy and root-caused it
by replaying the captured request body raw — `/tokenize` on the exact messages and tools, then
`/v1/completions` on those token ids, no chat template and no parsers. At ~60k+ tokens of agentic
context GLM-5.3-Flash occasionally loses its own tool-call format; `_glm47_arg_converter`
(`vllm/parser/glm47_moe.py:56-70`) then **salvages** the malformed XML into a real `tool_calls` entry
without validating a single argument key, the client echoes that garbage back into the history, the
chat template re-renders it in canonical `<arg_key>` syntax, and the model imitates its own corruption
until nothing parses and the whole turn is swallowed — empty stream, `finish_reason=stop` at
`<|observation|>`, client shows "turn complete". Their replay also eliminated the prefix cache,
temperature and context depth as explanations, in that order.

**The fix is in production as of this morning** (`HAREM_GLM47_FAILCLOSED`, and it is **the one knob in
this stack whose default is ON**, because here the upstream default is the bug):
[`tracks/tp3/patches/glm47-failclosed/`](tracks/tp3/patches/glm47-failclosed/README.md) — one file, six
anchors in `glm47_moe.py`. A tool call is buffered until it closes; at `TOOL_CALL_END` the complete call
is validated (name identifier-shaped and in the request's tools, every argument key identifier-shaped
and in the tool's schema `properties`, a non-empty body the converter salvaged nothing from counts as
truncated); a call that fails is surfaced as plain **content** — the raw `<tool_call>…</tool_call>` text
— with its slot cleared, so neither the streaming path nor `_build_extracted_result` can emit it.
**The cascade breaks at its first link**: nothing that looks like a tool call re-enters the history.
`=0` restores upstream byte for byte, and `test_failclosed.py` proves it by reproducing the bug —
**37/37** model-free checks inside the image, against the corruption vectors captured in the issue.

**Gates after adoption** ([`results/gates/failclosed-11sep.md`](results/gates/failclosed-11sep.md)):
probe **10/10** (content-only 9/9, requests with empty content **0**), code exam **12/12**,
needle-lite6 **6/6**, vision **PASS**. Valid tool calls were checked on both paths: non-streaming
unchanged, and a streaming call now arrives as **one** `tool_call` delta instead of a name delta plus
argument deltas — a visible API behaviour change, and the price of not half-streaming a call that turns
out to be garbage. Registering the patch invalidated every fast-load sidecar again, so adoption cost one
`FASTLOAD_MODE=dump` boot of **380 s** into a new directory ([docs/08](docs/08-fast-boot.md) §4,
[docs/14](docs/14-troubleshooting.md) §10.6); the FlashKDA sidecar is kept as the way back.

**What this does not fix, and we are saying it in both places.** The patch cannot change the **rate** at
which the model writes a first malformed call — it is a parser change, downstream of generation. Whether
the 4bpw EXL3 checkpoint raises that rate against the NVFP4 sibling is still the open measurement of
[docs/11](docs/11-open-issues.md) §2.30, and issue #7's data cannot separate the build from the weights
either. What changes is the stakes: a bad call now costs one visibly ugly turn instead of the session.
New: [docs/14](docs/14-troubleshooting.md) §9.13 (silent-failure index row 21), a pointer at the end of
docs/11 §2.30, and [HELP-WANTED](HELP-WANTED.md) §14 now asks for a **tool-call** variant of the
multi-turn gate — jdecker76's PR #4 is text-only, and the tool call is this mechanism's whole carrier.

---

## 11 September 2026 — issue #2 closed (chat template pinned), issue #6 closed (patch ordering + tail-row docstring), and a gate field fix

**Issue #2 is closed.** The env examples now pin `CHAT_TEMPLATE_HOST` to a named revision instead of
leaving it to chance: `tracks/tp3/env.tp3.example`, `tracks/tp3/env.tp3-full.example` and
`tracks/tp2/env.tp2-full.example` each carry a comment next to the line naming
`zai-org/GLM-5.3-Flash @ 690b7052` (4 September 2026), 10,950 bytes,
`sha256 0c4099f3382d6c92700dfb99725025360966fd73032f0ecf32377c0d9e6309c5`, with
`python3 scripts/verify-chat-template.py "$CHAT_TEMPLATE_HOST"` to check it. Only this revision
carries the tool-calling fixes, and its `clear_thinking` default is the one both launchers already
guard ([docs/14](docs/14-troubleshooting.md) §9.11, §9.12). That script is jdecker76's **PR #5**,
merged as `44a9722`.

**Issue #6 is closed**, two follow-ups on the K-pool tail work, both reported by jdecker76:
(a) [`patches/prefix-hit-and-kpool-tail/README.md`](tracks/tp3/patches/prefix-hit-and-kpool-tail/README.md#ordering)
gets an **Ordering** note — `patch-kpooltail-tp3.py` and `patch-indexer-workspace-tp3.py` both edit
`v1/attention/backends/mla/indexer.py`, and the only sequence ever exercised runs kpooltail *after*
indexer-workspace, which is what `tp3full-prelude.sh` (lines 152 and 179) and `tp2full-prelude.sh`
(lines 138 and 159) already do. (b) `patch-kpooltail-tp3.py`'s module docstring gets one clarifying
clause: its "32 entries wide" names the tail group's own block-table row as this code path allocates
it (the same width `kpool-tail-unit-test.py`'s model-free harness pins), a different row from
[docs/11](docs/11-open-issues.md) §2.33's 250,016-entry figure, which is the runner's full-width row
(`cdiv(max_model_len, block_size)`) — both numbers are true, of different rows. Editing that
docstring still changes the file's hash, and the fast-load sidecar identity hashes every
`patch-*.py`, so the README now says redeploying it wants a sidecar-less boot, not a restart
([docs/08](docs/08-fast-boot.md) §4).

**`correctness-probe.py` and `code-exam.py` now read `reasoning_content`** (`1b0b93f`), the field
this engine actually returns on vLLM `487ecf187` — both were reading `reasoning` instead, which
`early-stop-ab.py` and `mixed-load-probe.py` already knew not to do. Found while reviewing PR #4; the
old key is kept as a fallback.

jdecker76's [CREDITS.md](CREDITS.md) entry now also names PR #5 and the issue #6 ordering report,
which the existing §9.11 / K-pool-tail lines there had left implicit.

---

## 10 September 2026 — FlashKDA for KDA chunked prefill: +6.5 % of prefill, and 18 minutes paid to one of our own gates

**The fused `vllm._flashkda_C` kernel now runs GLM-5.3-Flash's KDA chunked prefill** instead of the
ten-kernel Triton `chunk_kda_with_fused_gate` chain — a port of vLLM
[#55737](https://github.com/vllm-project/vllm/pull/55737) (JaredforReal) onto our pinned tree, five
anchors in one file, behind `HAREM_KDA_FLASHKDA=1` and default off.
[`tracks/tp3/patches/flashkda/`](tracks/tp3/patches/flashkda/README.md),
[`results/gates/flashkda-ab-10sep.md`](results/gates/flashkda-ab-10sep.md).

**Two matched sidecar-less boots, one environment variable apart** `[measured-here]`: sustained prefill
**1,753.5 / 1,754.0 → 1,867.4 / 1,868.8 tok/s**, **+6.50 % and +6.55 %** on the two repeats, with
prefix-cache hits counted at **0** in every window of both arms; TTFT on a fresh 7K prompt
**4.192 → 3.978 s** (−5.1 %, the five samples do not overlap); decode **+0.21 %** and KV pool
**+0.31 %**, both inside their bands. Gates cold on the candidate: probe 10/10, code exam **12/12**
first run, needle-lite 6/6, vision PASS. Confirmed afterwards on the production fast-load path at
**1,865.7** and **1,866.6** tok/s, which is the thing a sidecar-less A/B cannot tell you.

**The estimate was wrong in our favour and that is on the record, not in a footnote.** A rank-0 prefill
trace puts the replaced kernels at **292.34 ms = 6.19 %** of GPU-busy time with prefill 98.5 %
GPU-bound, and the micro-benchmark at the production shape — strided q/k/v, so the three
`.contiguous()` copies are charged to FlashKDA — puts the kernel at **2.55×**. Together that predicted
**+3.5 … +4.1 %**. Measured **+6.5 %**, about 1.6× the prediction, and **nothing in hand explains the
rest**: [docs/11](docs/11-open-issues.md) §2.35 carries it open with the things that were looked for
and are not there (zero cache hits, identical bytes, the backend named in all three ranks' logs, and
the faster arm holding the *lower* mean SM clock).

**The decode number a single arm would have published is a phantom.** Against the previous night's
sidecar-on baseline of 58.53 tok/s the candidate's 61.28 looks like +4.7 %. The matched control reads
**61.15**. The whole difference is the sidecar-less boot, and FlashKDA does not touch the decode path —
[docs/09](docs/09-measurement-protocol.md) §2 in one paragraph.

**Two operational findings, and the second cost an outage.** Registering one new `patch-*.py`
invalidates every fast-load sidecar, so there is **no "registered but switched off" state** and a
symmetric A/B has to run sidecar-less in both arms. And the unit's `ExecStartPre` demanded
`MANIFEST.json` **regardless of `FASTLOAD_MODE`**, so pointing `FASTLOAD_DIR` at a new directory could
never boot — the mode whose job is to create the directory required it to exist. All three units went
to `failed` and the engine was down **17 min 50 s**. The gate is now mode-aware, narrowed rather than
removed, in both tracks' preflight scripts; [docs/14](docs/14-troubleshooting.md) §10.6 is the entry and
carries the four-step plan for the next patch A/B.

**What it cost:** one 394 s dump boot, a fresh **53 GiB per rank** sidecar into
`/var/tmp/glm53-exl3-flashkda` with the previous one kept untouched as the rollback (546 G → 493 G free
per node), **61.45 MiB per rank** of workspace shared by all 34 KDA layers, and the 18 minutes. Speed,
quality and memory were all looked for and the only movement is the prefill gain. **One anomaly
printed rather than smoothed:** the final production boot's KV pool is **7,030,303**, the low reading of
a five-boot session whose spread is 1.06 % with nothing changed — read as allocator variance at 0.88,
because the FlashKDA arm read *higher* than the control, and **not re-booted for a better number**.

**A revision we had wrong, corrected before publication.** An internal note recorded this image's
FlashKDA as tag `3b225bf`; it is **`b5d11010`** (28 July 2026), the tag the pinned vLLM's
`cmake/external_projects/flashkda.cmake` names, and the proof is that our `_flashkda_C::fwd` takes 14
arguments with no `checkpoint_state` tail — parameters FlashKDA gained on 6 August. So every number
here is measured on a **July** kernel, seven commits behind what #55737 was validated against.

**Not measured: two nodes.** [HELP-WANTED](HELP-WANTED.md) §15.

---

## 10 September 2026 — issue #1: the reasoning-retention default, the template's provenance, and a field report on the K-pool fix

A production user (jdecker76, issue #1) ran production configuration 13 for four days of multi-user
agentic traffic and reported looping, premature end-of-turn and a long-session quality drop that
none of this repository's single-turn gates can see. Three things follow. **Both launchers now pass
`clear_thinking: true`**: the served template, the 4 September upstream file, defaults to
re-rendering every prior turn's `<think>` block into every later prompt, which on this engine turns
a 30-token three-message conversation into 1,771 tokens per prior reasoning turn `[measured-here]`
([docs/14](docs/14-troubleshooting.md) §9.12). **docs/14 §9.11 is closed**: the served template is
verified by size and hash as the 4 September revision of `zai-org/GLM-5.3-Flash`, whose template is
the only file in that repository that has ever changed; the `clear_thinking` default flipped between
the 26 and 27 August revisions, and turboderp's checkpoint ships the earlier one. **Field evidence
for the K-pool tail fix** (`008a730`): the reporter's window predates it, and their three symptoms
are the user-visible face of the 99.67 %-wrong-block finding — correlation, not attribution, and
recorded as such. [HELP-WANTED](HELP-WANTED.md) §14 asks for the multi-turn gate that would have
caught all of it; [docs/11](docs/11-open-issues.md) §2.30 carries the field corroboration of the
`low`-effort gap and the operational advice that follows from it. The early-stop A/B is in: 24 long-form prompts at temperature 1.0 / top_p 0.95, DFlash2 on and
off, **zero early stops in either arm** (48 requests, all to the 4,096-token cap, coherent at the
cut); the verifier applies top-k/top-p and temperature before rejection sampling, so the
"draft-proposed EOS" hypothesis has neither a code path nor a measurement behind it on this build
([`results/gates/early-stop-ab-10sep.md`](results/gates/early-stop-ab-10sep.md)).
---

## 8 September 2026 — the full MMLU: 85.30 ±0.29

All 14,042 questions, loglikelihood, 0-shot, on the current production configuration: **0.8530 ±0.0029**
(humanities 0.80, social sciences 0.91, STEM 0.86, other 0.87). The NVFP4 sibling's full run was 85.9;
this stack's 1,995-question sample was 86.47 ±0.74. Same class; the last `[not tested]` of the quality
battery is closed. [`results/gates/quality-battery-production-13.md`](results/gates/quality-battery-production-13.md) §10.

## 2026-09-08 — What the same engine scores when it is allowed to think: 90.3 against 85.5, and the four scenarios that were the gap

**tool-eval-bench at reasoning effort `high`: `final_score` 91, three-trial mean 90.3 ±1.2, median 91,
CI95 [89, 91], 160 of 176 points, 44 min 53 s** `[measured-here]`. The same 88 hardmode scenarios at
`low` on the same engine score **85.5 ±1.3**, and the NVFP4 sibling recipe scores **87.8 ±0.9**.
Welch t = 5.53 and 3.25; exact permutation over all 165 splits p = **0.006** and **0.012**.
[`results/gates/quality-battery-production-13.md`](results/gates/quality-battery-production-13.md).

**Nothing on the cluster changed.** The engine was not restarted and no flag on it moved. A 40-line
reverse proxy sits between the harness and the API and injects `chat_template_kwargs.reasoning_effort`
into `/chat/completions` bodies only; the harness points at the proxy. It is a measurement instrument,
not a deployment — production still serves at `low`, which is what every other quality number in this
repository is measured at. [`scripts/effort-proxy.py`](scripts/effort-proxy.py).

**It reframes the −2.3 points [docs/11](docs/11-open-issues.md) §2.30 has been carrying.** Three of
the four scenarios that gap was concentrated in go level or better at high effort — **TC-21 from 0.00
to a full 2.00** (5/5 validation errors instead of 1/5, on a tool-free reasoning task), TC-87 1.25 →
2.00, TC-74 1.25 → 1.67. TC-51 moves 0.12 → 0.67 and remains the largest hole, which is what the
grading rule §2.30 proved from the grader's source predicts. Twelve scenarios up, three down; four
categories carry all nine points, Safety & Boundaries 21 → **25/26** and Autonomous Planning 2 →
**4/6**. §2.30 stays open: the sibling has not been run at high effort, so the build and the weights
are still not separated.

**What it cost.** Completion tokens **+90 %** (19,072 → 36,152 per trial), mean scenario duration
+47 %, median turn latency +24 % (1,968 → 2,440 ms), harness `responsiveness` 65 → **58**. TTFT does
not move — this is a decode-length cost, not a prefill one. Looked for and not found: malformed calls,
timeouts, 5xx, empty content, safety warnings; `max_points` is 176 in both files, so no scenario was
dropped. Three trials rather than eight is the honest limit of the arm, and **TC-38** — 2.00 at `low`
over eight trials, `[0, 2, 2]` here — is recorded as unresolved rather than as a regression, because
three trials cannot tell a new failure mode from an unlucky draw.

**The 1M needle, deferred on 6 September, is run: 20/20** `[measured-here]`. Four haystack sizes up
to **997,952 tokens** by five depths, every cell retrieved, effective context the full largest
haystack; 1 h 25 min 32 s and 8.27M tokens against the NVFP4 sibling's identical 20/20 in 5,288.6 s —
**3.0 % quicker**, the same result. Run at the server's own `reasoning_effort: low` with the proxy
stopped. Thermals across the whole 85 minutes, 141 samples per node: head 83.4 °C mean / 87 peak,
workers 79.7 / 82.0 mean, SM clocks 2,398 / 2,450 / 2,356 MHz, no node out of band, peak power
82.4 W, swap traffic zero. Two readings that do not match what this repository has been repeating and
are left as measured: **the head is hotter than the workers by 3.7 °C, not by the wider margin
claimed, and it does not hold the lowest clock — worker-2 does.**

**The battery started from a whole-cluster reboot into this configuration**, all three nodes at the
same second, nothing started by hand: SSH back at 103 s, `ibv_devinfo` 4/4 on all three before the
engine, `/health` 200 at **321 s** against 318 s for the vision configuration. Gates cold: probe
10/10, code exam 12/12 first try, tool-call 8/8, needle-lite 6/6, vision 5/5. **One deviation, written
as one:** the KV pool came up at **6,873,278** tokens, **0.60 % below** the documented
6,914,600–7,143,250 boot-to-boot spread. That spread is a set of observations rather than a
specification and this boot widens its floor; nothing downstream moved, and the variance behind it is
still uncharacterised.

---

## 2026-09-08 (two nodes) — The two-node recipe catches up: the tower, both backports, and a block that is 4,608 tokens wide

**The TP=2 track's recommended configuration is now candidate D**: candidate C plus the **vision
tower** (4 images + 2 videos per request) and both upstream backports (`HAREM_PREFIX_HIT=1`,
`HAREM_KPOOL_TAIL_FIX=1`). Memory rung, checkpoint, image, kernels and every other setting are
candidate C's, untouched. [docs/15](docs/15-tp2-track.md) §5.10,
[docs/19](docs/19-vision-at-two-ranks.md), [`results/gates/tp2-candidate-d.md`](results/gates/tp2-candidate-d.md).

**Speed, one boot, medians of three rounds** `[measured-here]`: C1 **59.45** aggregate / 64.45 per
stream against candidate C's 60.08 / 65.96, C8 **155.47** against 157.71, fresh prefill **1,413**
against 1,414, TTFT and draft acceptance unchanged. Nothing outside a band in either direction.
Gates: probe 10/10, code exam 12/12 first try, tool-call 8/8, needle-lite **6/6 three times**.

**Six vision gates, all passing**, including **4 images + 2 videos in one request** (5,991 prompt
tokens, all six items described) and a third video refused with HTTP 400 while the engine stays up.
`HAREM-VISION: tower loaded, data_parallel=True, EXL3 linears=99, unquantized linears=0` at two
ranks, and the EXL3 module audit reads **302 EXL3 / 113 bf16** against candidate B's 203/113 — the
difference is the tower's 99.

**The number the three-node handover asked to be measured rather than copied: the prefix-cache block
is 4,608 tokens at two ranks, not 3,328.** gcd(4,608, 55,296). And at that granularity the defect is
worse than at three ranks rather than milder, which is the entry's most useful line: an 8,008-token
exact repeat reads **0.0 %** out of the cache with the knobs off and **57.54 %** with them on — which
is **100 % of the ceiling**, because one block is the whole hit available at that length. Repeat TTFT
5.60 s → **2.55 s**. Three ranks lost 41.56 points of 83.12; two ranks lose all 57.54. The A/B is one
environment line on the same tree and the same sidecar, so the KV difference between the arms
(−0.28 %) also says the backports are free.

**The cached-path equality test at two ranks: 24/24.** 24 prompts at ~11K, ~83K and ~177K tokens,
each asked cold then repeated byte-for-byte with the cache counters read either side: **all 24 repeat
answers byte-identical to their cold answers and correct on both passes**, mean repeat hit 91.7 %
against 0.0 % cold, every size class on its ceiling.

**What it cost.** KV pool **2,692,857 → 2,585,714, −4.0 %** — and the A/B above attributes that to
the tower rather than to the backports. Written down as a cost and not as noise, because candidate D
has one boot and candidate C's boot-to-boot spread at two ranks was never measured. One dump boot
(1,013 s) and a sidecar that grows 78 → 79 GB per rank. Boot through the autostart unit **280 s**,
the first time that unit has brought up the configuration this track recommends.

**Two findings that cost a boot each and are now written down so they cost nobody else one.**

1. **The vision patch needs a file the two-node tree deliberately did not ship.**
   `patch-vision-tp3.py`'s VS1 anchor is written against the text `patch-vllm-tp3.py`'s edit 4b
   leaves behind, and [docs/15](docs/15-tp2-track.md) §2.3 listed that file under *not shipped*
   because its padding half is a no-op at TP≤2. Forking the vision patch was rejected — it sits in
   three-node production, its sha256 is printed in that boot log and its content is hashed into that
   sidecar's identity — so the two-node tree now carries the same file, byte for byte. The tree is
   **eighteen** files rather than fourteen. [docs/19](docs/19-vision-at-two-ranks.md) §2 `[retracted]`
   against §5b's "the same VS1/VS2/VS3 … nothing else".
2. **A two-node boot dies on the drafter, four minutes in**, because the three-node 32/8 → 36/9 pad
   is applied by rewriting `config.json` **inside the shared drafter directory**. Nine key-value
   heads do not divide by two: `qwen3_dflash.py:186 assert self.total_num_kv_heads % tp_size == 0`.
   The fix is a directory with the unpadded config and a hard link to the same weights, built
   **before** the dump boot. New [docs/14](docs/14-troubleshooting.md) §10.5.

**Two instrument notes.** `prefill-7k.py` stops being a prefill measurement once `HAREM_PREFIX_HIT=1`
is on — it times the second copy of one prompt and read **2,602 tok/s** against a fresh-prompt 1,413.
And one of our own fresh-prefill readings (1,120 tok/s) was taken while a second probe was hitting
the same engine; alone it reads 1,413 and 1,403. Both are recorded rather than dropped.

**Host memory in the worst case**, which is what the swap rule actually asks about: a 4-image +
2-video request fired during a C8 sweep. Rank 0's `MemAvailable` floor **4.90 GiB**, swap flat at
0.059 GiB, **zero pages in or out**; the text arm ran at 141.03 tok/s against 155.47 alone, −9.3 %,
which is queue contention and is named as such. No error, no slowdown outside that, safe zone intact.

**Not measured**: the tower *sliced* across the two ranks (new [HELP-WANTED](HELP-WANTED.md) §13), a
second boot of candidate D, MMLU, a two-node reboot test, sustained multimodal load.

---

## 2026-09-08 (later) — Both backports promoted, on two boots and a test written for the one thing that was left

**The production configuration now carries `HAREM_PREFIX_HIT=1` and `HAREM_KPOOL_TAIL_FIX=1`.**
`.env.tp3` is the both-knobs arm on all three nodes, the systemd unit is unchanged — it reads
`ENV_FILE=$HOME/exl3-zeus/.env.tp3` and takes the tree, the overlay and the sidecar from it, so
promotion is a file swap — and the production tree is now the prefix-hit tree with its own fast-load
sidecar. The previous vision tree and its sidecar stay on disk as the rollback pair.
[`results/gates/prefix-hit-and-kpool-tail.md`](results/gates/prefix-hit-and-kpool-tail.md) §6.

**What the entry below decided, and why this one overturns it.** The rollback rested on one boot of
the both-knobs arm returning needle-lite 5/6 twice, after half the evidence for it had already
collapsed — the `matrix` code-exam item failed on unpatched production twenty minutes later. What
remained was one unreproduced observation. The answer to that is more boots and a directed test, not
a shelf.

**Two boots through the autostart unit, full battery on each** `[measured-here]`: `/health` at
**251 s** and **265 s**, KV pools **7,044,077** and **7,041,322** — inside this configuration's
documented 6,914,600–7,143,250 spread, and boot 2 within 0.01 % of the published 7,041,322. Boot 1
also took a fifteen-minute mixed soak (eight concurrent streams, code and prose, plus two
4,096-token generations; 124 requests, **132,867 tokens, engine alive, zero errors**) and a second
full battery after it. **Nine needle-lite runs, nine 6/6.** Three code exams, three **12/12 on the
first attempt** — `matrix` did not fail once. Probe 10/10 twice, tool-call 8/8 twice, vision 5/5
three times. A third boot proved the unit brings the promoted configuration up by itself: 266 s,
probe 10/10, vision 5/5.

**The cached-path equality test, which is the measurement that settled it.** A warm 5/6 would mean
one thing: a prompt served partly from cache answering differently from the same prompt computed in
full. So that was tested directly — 24 needle-style prompts at **~11K, ~84K and ~178K** tokens, eight
each at eight depths, every one with its own filler seed, each asked cold and then repeated
byte-for-byte with the engine's prefix-cache counters read either side so the repeat is *proved* to
be a hit. **24 of 24 repeat answers byte-identical to their cold answers, and all 24 correct on both
passes**; mean repeat hit **94.7 %** against 0.0 % cold; every size class on its ceiling (88.1 % at
11K where the ceiling is 88.1 %, 95.8–96.4 % at 84K, 98.1–99.8 % at 178K). New instrument,
`cached-equality.py`. `[measured-here]`

**A finding that came out of getting the test wrong first, and it is the more useful half.** The
original design asked all 24 prompts cold and then repeated all 24. **Every repeat hit 0.0 %** and
took its full cold time — 97.5 s on a 177K prompt against 1.1 s when repeated immediately. Twenty-three
long requests in between leave nothing behind. So every hit ratio this repository publishes is a
back-to-back number, and "the same prompt again, later in a busy hour" has never been measured here.
The mechanism is not measured either and is not asserted: [docs/11](docs/11-open-issues.md) §2.34.

**What it cost.** Nothing in memory (three KV pools inside the spread; host RAM and swap where
configuration 13 already sat) and nothing in quality that was looked for. Speed was **not** re-measured
on these boots `[not tested]` — it belongs to a session with a control arm, and the same-session A/B
below has it. The real price is disk: a second ~53 GB-per-node sidecar, kept alongside the first
because the pair is the rollback. One prose generation in the soak tripped the repetition heuristic
at 59.5 % and **its text had not been kept** — an instrument gap, now fixed; the same prompt came back
at 4.2 % alone and 1.9–18.7 % across eight concurrent generations, so it did not reproduce, and it is
recorded as one unexplained observation rather than explained away.

**What is not discharged.** The gates still have no measured flake rate. Three clean code exams are
not a baseline, and [HELP-WANTED](HELP-WANTED.md) §12 part one stands exactly as written.

---

## 2026-09-08 — Two upstream backports, measured; neither promoted, and the gate that judged them has a flake rate

**No change to the production configuration.** `.env.tp3` was never edited — verified byte-identical
to its dated backup on all three nodes at the end — and the production patch tree and its fast-load
sidecar were never touched. New patch tree, both knobs default off:
[`tracks/tp3/patches/prefix-hit-and-kpool-tail/`](tracks/tp3/patches/prefix-hit-and-kpool-tail/README.md).
Results: [`results/gates/prefix-hit-and-kpool-tail.md`](results/gates/prefix-hit-and-kpool-tail.md).
Standing items: [docs/11](docs/11-open-issues.md) §2.32 and §2.33.

**Two real bugs, found and measured.** The hybrid prefix-cache annotation: nothing on this model's
grouping path ever sets `is_eagle_group`, so the coordinator flags **every** group and the EAGLE
last-block drop costs **exactly one 3,328-token block off every exact repeat** — an 8,008-token
repeat hits 41.56 % where the ceiling is 83.12 %. The patch takes it to **83.12 %, the ceiling**, and
follow-up TTFT from **2.830 s to 1.076 s, −62.0 %**; a four-turn agent conversation goes with it.
The K-pool tail slot mapping: `positions` never reaches the tail's metadata builder on a hybrid
model, so **99.67 % of the tail's writes address a block that is not the request's own**, the worst
of them outside every block any request holds. The patch takes that to **0.00 %**. Both
`[measured-here]`.

**Neither was promoted, and the reasoning is worth more than the patches.** The prefix-cache half
missed its acceptance bar — a *raw* 95 % hit, which the 3,328-token granularity makes unreachable at
8K — and the 60K case did not move, because its offset past the aligned boundary is 6 tokens and the
drafter's own drop then costs a full block on re-alignment. Separately, one boot of the both-knobs
arm returned needle-lite **5/6** twice; six later runs of that configuration returned 6/6, including
under concurrency and after an identical 48,910-token soak.

**And then the gate turned out to be flaky.** The rollback was decided partly on the K-pool arm
losing the code exam's `matrix` item after a soak. Twenty minutes later, on the restored
**unpatched** production, the first code exam came back **11/12 — `matrix`, the same assertion** —
and the four runs after it were 12/12. `matrix` failed 2 of 12 code exams that session, once with no
patch tree present. That evidence is withdrawn, and the conclusion is not about either patch: **we
adjudicated two changes with gates whose own flake rate we had never measured.** A gate baseline is
now [HELP-WANTED](HELP-WANTED.md) §12, ahead of the reproduction it would serve.

**One correction to a published diagnosis, ours to make because it is our build.** vcruz305 describes
the K-pool tail's block-table row as one entry wide, so that `pos >= block_size` reads past it. Our
runner sizes that row `cdiv(max_model_len, block_size)` = **250,016 entries**: no out-of-row read
happens at all, and our detector counts zero row overruns while the mapping is wrong for 99.67 % of
tokens. The harm is the wrong block, not the overrun — and on this build there is nothing for a clamp
to clamp, which is the second independent reason the clamp is the wrong layer.

**What it cost:** one 495 s dump boot and a ~53 GB-per-node fast-load sidecar, which stays on disk
alongside the production one, so arming either arm later is an environment line and no dump.
Restoring production was `systemctl start`: `/health` 200 at **255 s**, KV pool **7,066,115**, probe
10/10, tool-call 8/8, needle 6/6, vision 5/5.

---

## 2026-09-07 — Vision at three ranks: the tower comes back, the video sampler bug, and 4 images + 2 videos per request

**Production configuration 13.** Configuration 12 with the vision tower **on**. Six environment lines
and six patch anchors; nothing else moves — same image, same checkpoint, same memory fraction, same
batching, same sampling. New page: [docs/18](docs/18-vision-at-three-ranks.md). New patch tree:
[`tracks/tp3/patches/vision/`](tracks/tp3/patches/vision/README.md). Gate results:
[`results/gates/vision-gates-tp3.md`](results/gates/vision-gates-tp3.md).

**This repository said this was impossible, and the claim is withdrawn.** docs/03 §3, docs/14 §2.7,
the README quick start, docs/00 and the TP=3 environment template all said the tower cannot be served
at TP=3, and the recipe shipped a patch whose purpose was to stop it being built. Three things were
wrong: the replication flag that clears the assert was named and dismissed in the same sentence; the
"1.05 GiB of BF16 vision weights per rank" belonged to the routed-experts-only checkpoint and was
carried across a checkpoint change without being re-derived (on the production checkpoint the tower is
**6-bit EXL3, 0.557 GiB total, 0.416 GiB per rank**, readable from the safetensors headers in
seconds); and the real obstacle — vLLM builds the tower with `quant_config=None` while this
checkpoint's tower is quantized — was never in the sentence at all. **The retraction count moves from
thirty-seven to thirty-eight**, [docs/11](docs/11-open-issues.md) §1.14, and every place that quotes
it is updated. docs/03 §3 becomes `[history]` with a pointer; docs/14 §2.7 keeps the text-only path
and carries the correction beside it.

**What it cost, measured against a same-session configuration 12 reference.** KV pool **7,143,250 ·
7,033,057 · 7,016,528** across three boots, all inside configuration 12's own 7,024,793–7,126,721
boot-to-boot spread — the tower has no measurable pool cost. C1 **70.0 (+0.1 %)**, C8 **198.8
(+1.5 %)**, both inside band; TTFT and DFlash2 acceptance (60.4–63.9 % against ~62 %) unchanged;
peak activation equal. Fast-load boot **247 s** by hand and **244 s** under the unit against ~251 s;
whole-cluster reboot **318 s** against 311 s. **Ten gates out of ten**, text gates 10/10 and 12/12 on
every boot. The memory fraction stays at **0.88** — a 0.87 rung was prepared and never run, because
the worst-case stress found no harm to pay for.

**What it cost that is not in that table.** One extra 53 GB-per-node sidecar dump boot (454 s): the
tower adds **596 tensor names and 0.42 GiB** to the sidecar, 5,858 → **6,454**, and the identity check
is exact name-set equality, so the text-only sidecar is *refused* by a tower-carrying engine — the
correct behaviour, and the reason a new `FASTLOAD_DIR` was mandatory rather than tidy. About
**1.2–1.4 GiB** more resident host memory on the rank that serves the API, and the multimodal
processor cache had to be turned off (`--mm-processor-cache-gb 0`, default **4 GiB of host memory**)
to bring idle MemAvailable there from 1.10 to 3.42 GiB. Under the worst case built for it — a
1M-token request, a C8 round, and a 4-image + 2-video request sent *while* the C8 round ran — that
node pages out 282 MiB once and then nothing, MemAvailable never falls below 1.47 GiB, and nothing
errors or slows outside its band. The concurrent window costs the text arm **−12.7 %**; that is
contention at `--max-num-seqs 8` and **how much of it is multimodal was not attributed**
`[not tested]`.

**And the thing that actually cost the three engine windows was not ours.** vLLM's GLM-5-Next port
runs a video through **two independent frame samplers** — pixels through GLM-5-Next's, prompt
placeholders through GLM-4.6V's — because `Glm5NextProcessingInfo` overrides four methods and not
`_get_video_second_idx_glm46v`. A four-second clip produced 4,968 placeholders for 1,656 encoder rows
and killed the engine core on all three ranks with `Attempted to assign 1656 = 1656 multimodal tokens
to 1728 placeholders`. It has nothing to do with three ranks, EXL3 or expert parallelism: it kills
**every** GLM-5-Next video request. And the ratio is not a constant — it tracks the clip's duration
(**3.0 below 30 s, 1.0 between 30 and 300 s, 0.5 and lower above 300 s**), so the one band that works
is a coincidence and long clips fail in the other direction. Filed as
[vllm#55644](https://github.com/vllm-project/vllm/issues/55644) with the fix and a unit test in
[vllm#55647](https://github.com/vllm-project/vllm/pull/55647), DCO-signed and awaiting a maintainer's
ready label.

**Two published readings of that crash were wrong and are withdrawn.** "The width rounded 46 → 48"
— 1,728 and 1,768 are chunk-local placeholder counts under chunked prefill, not totals, and both
sides read `H` and `W` from the same grid tensor. "The `max_pixels` brake is the first suspect" — it
never reaches the video processor at all; the key that does is **`max_image_tokens`**, measured on a
1080p clip at 10,764 → 7,788 vision tokens with images bit-identical. `max_pixels` stays in the line
because dropping it takes vLLM's own per-item budget from 10,242 to 32,242 tokens.

**Four gate lessons, all of them about our own tooling.** An anchor written against the stock file
matched zero times at boot while the model-free gate said PASS, because the gate chained two patches
instead of reproducing the prelude's order — *a gate that does not reproduce the boot is not a gate*,
and `verify-cpu.sh` now runs the full order. The same gate returned 0 through a `| tail -1` pipe for
want of `pipefail`. A count derived from safetensors headers (48 dead fused tensors) was asserted as
an invariant and killed a boot whose tower was entirely correct; counts are reported, invariants are
asserted. And the expensive one: the install-day gates checked only **weight loading**, so a tower
that loaded perfectly still died on the first video — `check-video-geometry.py` now checks
**geometry**, model-free, in under a second. New failure signatures in
[docs/14](docs/14-troubleshooting.md) §2.7a–§2.7d.

**One boot-log line is absent by design, and was not patched back.** `HAREM-VISION: tower loaded`
lives inside the tower's `load_weights`, which the fast-load path skips. Editing the patch to restore
it would have invalidated the sidecar just written **and** shipped a different tree from the one the
gates passed. The substitute chain runs on every boot — the restored-tensor count (6,454 against
5,858), the FLASH_ATTN encoder line, `mm_encoder_tp_mode: 'data'`, and the fail-closed name-set
equality that a BF16-fallback tower could not have satisfied — and the rewritten gate was
counter-tested against five corrupted-log scenarios, all five of which failed it.

**Still open**, [docs/11](docs/11-open-issues.md) §2.31: the upstream PR is waiting on a maintainer;
the brake ladder has one rung (`max_image_tokens` 8,000, and nothing at that ceiling has been served
— every fixture is four seconds long); **image quality is unbenchmarked** — no MMMU, no DocVQA, no
comparison against a BF16 tower; the audit line above; and the unattributed −12.7 %.

---

## 2026-09-07 — Consistency pass, part 2: the systemd page, the tree naming, the charts, and the retraction count

**Documentation only. No measurement was re-run and no claim was rewritten — one published figure was
wrong and is corrected, and everything else is a count, a label or a cross-reference catching up.**

**One corrected figure.** `results/configs/production-configurations.csv` row 11 carried
`boot_s` **251**, which is production **9**'s fast-load boot. Production 11's own note in the same row,
the README and `audit/` all say **271 s** (236 s from `systemctl start`). The column now reads 271 and
the note says where the 251 came from. Nothing else on the row moves.

**The systemd page was a configuration-10 page.** It documented one whole-cluster reboot and said the
start-order tolerance had been "demonstrated once". There have been **three** — configurations 10, 11
and 12, reading **315 s** (wall clock; 242 s by the harness counter, both printed), **312 s** and
**311 s** from the `reboot` command. The page now leads with all three and keeps configuration 10's
trial in full as `[history]`, because it is the only one whose log contradicts itself and the only one
with a phase decomposition. `results/boot/boot-ledger.md` gains configuration 12's row so the
citation resolves, and `HELP-WANTED` §2, `docs/09` §11, `docs/14` §10.1a and `tracks/README.md` no
longer say "one trial".

**The two-node autostart figure was candidate B's and is now labelled as such.** The 261 s
`systemctl start` → `/health` trial ran on candidate B; **candidate C was never re-timed under the
unit** `[not tested]`. Its unit is the same unit pointing at the same launcher with a different
environment file and sidecar, and the 272 s in `docs/15` §5.9 is a `docker run` fast-load boot, not a
`systemctl start`. Labelled in the README's candidate-C table, `tracks/tp2/README.md`, `docs/15` §5.7
and the systemd page.

**The launcher directory and the patch-tree directory are two things**, and five places said
`tp3full/` while the environment template pointed `TP3_DIR` at a configuration-named tree. Nothing was
renamed — the two are genuinely separate on our nodes, `ExecStart` follows the launcher and `TP3_DIR`
follows the patches, and only the second is hashed into the fast-load manifest identity. Said once in
each of `tracks/README.md`, `tracks/tp3/README.md`, `systemd/README.md`, quick-start step 8, the
prelude's header and the environment template's own comment.

**Both charts stopped at production 10 and one of them carried a retracted claim.**
`charts/speed-by-configuration.svg` now runs to configuration 12 (its data always came from the CSV;
the caption said "all 5 September" and quoted one memory fraction for twelve configurations).
`charts/kv-pool-progression.svg` had a hard-coded series ending at production 9 and a footer arguing
that 0.85 was rejected on free host RAM — **that rejection is retracted** (§2.4): the 6 September
ladder re-climbed the fraction against swap traffic under load, 0.85, 0.87 and 0.88 all pass, and
**0.90** is the rejected rung. The 0.85 bar is now grey and labelled as retracted, 0.90 is the red
one, and the series ends at production 12's **7,041,322**. `results/configs/kv-pool-progression.csv`
gains the four readings the new bars need, so the chart is still nothing but a view of a committed
CSV.

**The retraction count is stated once and is the same everywhere: thirty-seven.** It was quoted as
"thirty-two" in five places, "six" in the style guide and "twenty-five" in one paragraph of docs/11
itself. `docs/11` §1's opening paragraph now defines it — the **32 rows** of §1.9's audit table plus
the **five** withdrawn findings that are not rows of it (§1.5, the first bullet of §1.8, §1.11, §1.12
and §1.13) — and every other place quotes that number and points at the definition. **A retracted
correction is a retraction**, which is what §1.12 and §1.13 are; `audit/` §6 was missing both and now
indexes them.

**[@drakosha](https://github.com/drakosha) is credited in [CREDITS.md](CREDITS.md).** He opened
[vllm#55221](https://github.com/vllm-project/vllm/issues/55221) and corrected the failure mode we had
built a four-layer safety story on — §1.13, the first of our retractions a reader outside this stack
caught. The same page's claim that the workspace patch is "not in production" predated its promotion
and is corrected.

**Stale text found in passing, each against this repository's own later measurement:** `docs/11`
§2.29 still called TP=3 + expert parallelism + CUDA graphs a mode that "has never run here" after its
own 6 September battery ran it without a hang; `results/memory/indexer-workspace-ab.md`'s banner
tagged two ranks `[not tested]` while §6 of the same file carries the two-rank +32.14 %;
`CONTRIBUTING` item 5 asked for MMLU at TP=3, which has been a TP=3 figure since 5 September (the
open item is the full 14,042-question run), item 7 repeated the retracted "0.85 was rejected" and
item 10 told two-node readers they have no current figures; `docs/14` said "eighty-six failures"
where two of the eighty-six are entries we did not hit and say so; the three-node patch tree is
**23** files rather than twenty-two in four places; `tracks/tp3/README.md`'s numbers table was
production 10's; `results/speed/tp2-production-candidate.md` had a second top-level heading; and
five cross-references pointed at `docs/15` §3.5, which is `docs/15` §4 since that section was
promoted.

Link check: 1,445 relative links, none broken. Privacy grep clean over the whole tree.

---

## 2026-09-07 — Content types on production 12: the pool grew 38 % and no category noticed, and the prose row is decomposed down to the drafter

**The category probe had not run for four configurations, and it has now run on the production one.**
Single stream, six prompts per category, one warm-up round plus three measured rounds on a warm
production-12 engine with no boot inside the window: **code 61.5, math 76.2, JSON 73.1, prose 29.0
tok/s**, acceptance **46 / 57 / 53 / 13 %**, accepted tokens per step 4.22 / 5.00 / 4.70 / **1.91**
`[measured-here]`. Against production 9 — measured with the same prompt set at
`gpu-memory-utilization 0.80` — every category is inside its own round-to-round spread (prose −0.2 %,
code −0.4 %, JSON +0.5 %, math −4.3 % against an 8.4 % round spread). Between those two arms the KV
pool went **5,165,289 → 7,118,457 tokens, +37.8 %**, through three memory rungs and the indexer
workspace bound: **the pool was bought with memory, not with tokens per second.** C4 is on the page
too. New file:
[`results/speed/category-speeds-production-12.md`](results/speed/category-speeds-production-12.md).

**The prose row is 2.6× below math, and it decomposes exactly.** With a k-deep draft the engine emits
`L = 1 + k × acceptance` tokens per target forward. Divide each category's decode rate by its own `L`
and the **step rate is the same in all four columns — 14.58 to 15.55 /s, a 6.4 % spread** — while the
decode row above it spans a factor of 2.6 `[measured-here]`. The engine steps at one rate whatever
the content; only the tokens per step change, and that is the drafter's hit rate. Three
measurements finish the attribution: acceptance per category has not moved across **ten boots**
(prose 12.1 – 13.1 % through four image builds, four memory fractions, the mesh patches and the
full-scope promotion); the full-scope checkpoint raised the step rate ~30 % and left acceptance
untouched, so it is not the weights; and on the one arm measured with *and* without the drafter a
speculative step costs **1.58×** a plain one, putting break-even at **8.3 % acceptance** — prose, at
13 %, gets about **×1.21** out of speculation where math gets ×3.16 `[estimate]`.

**No configuration lever exists for it, and this is said plainly rather than left implied.** `k=5`
buys prose 3.6 % at C1 and costs every other category and every concurrency level 3.5 – 6.4 %;
per-request `k` does not exist, because `num_speculative_tokens` is fixed at engine start; and the
newer drafter revision in the entry below came back with acceptance equal — and then, an hour later,
**split by category on that same revision, prose still does not move**: 12.0 % acceptance on its
measured round and 13.1 % on its warm one, a 1.1-point swing inside one boot that brackets production
12's 12.97 % rather than beating it, with code and JSON equal to a third of a point `[measured-here]`.
The one column that may have moved there is **math**, 3.2 points of acceptance above every previous
arm on a single boot — not carried by the aggregate, and left as a two-boot question rather than a
result; the pin stays on `dc77ff1c`. **A better draft checkpoint is the only thing that moves the
prose row**, and a different drafter demonstrably can — an MTP head at k=3, in the same session, wins
prose 21.3 against 18.5 tok/s while losing every other category by 20–40 %. Written up as [docs/10 §1.2](docs/10-results-and-roofline.md); the two stale
`[not tested]` claims about content types in that page are corrected, and **mixed load is still
unrun** `[not tested]`, as is an unspeculated three-node arm, which would turn the
return-on-speculation column from an estimate into a measurement. Cost of the work: one 28-minute
engine window of request traffic, no boot, no configuration change, gates **10/10 · 12/12** cold and
again after the benchmark, swap use zero throughout.

---

## 2026-09-07 — The newer DFlash2 drafter revision, measured: acceptance does not move, and the pin does not either

**The one instrument a drafter swap actually moves is acceptance, and it did not move.**
`incoai/GLM-5.3-Flash-DFlash2` at **`bf582e4e`** (31 August, *"Checkpoint update"*) against the pinned
**`dc77ff1c`** — identical file size to the byte, byte-identical raw config, only the trained weights
differing — in one 23-minute engine window against the production-12 arm of the previous night's
CUDA-graph A/B, three sweep rounds each: **acceptance 61.32 / 61.61 % at C1 / C8 against 62.08 /
60.53**, under 1.1 points at all five concurrency levels **with mixed signs**, accepted tokens per step
level (5.29 / 5.31 against 5.35 / 5.24), gates **10/10 and 12/12 cold and warm on the first attempt**,
tool-call 8/8, KV pool −0.39 % `[measured-here]`. **No measurable difference; `dc77ff1c` stays** — and
this is the second negative trial of that checkpoint, on two stacks with two engines.

**Aggregate speed read ~5 % low, and it is written up as a boot rather than as a result.**
`prefill-fresh` fell by the same **5.04 %**, and the drafter's weight values cannot change prefill at
identical shapes, so the whole boot was uniformly slow. The reference arm's own round-to-round spread
was ±4.9 % at C1, as large as the difference, and the two arms are different boot regimes — a warm
sidecar-loaded engine against a fresh sidecar-less one — on a stack that spreads 15.9 % boot to boot.
Thermals level, no throttle bit on any node. The raw reading is left in the table rather than adjusted
away.

**Two by-products worth more than the arm.** A **sidecar-less TP=3 boot was timed for the first time —
275 s**, against an internal 8–10 minute estimate that was simply wrong. And the drafter zero-pad proof
in `patch-dflash-tp3.py` ran against an **independent** checkpoint for the first time and held:
26,214,400 padded elements verified zero on the rank that owns all of the padding.
[docs/04](docs/04-dflash2-port.md) §8.1 has the table and what promotion would have cost — two
environment lines *and* a mandatory sidecar re-dump under a new directory name, because the sidecar
stores the draft tensors and the draft path is part of the sidecar identity, so `preflight-fastload`
refuses rather than silently restoring the old drafter. [docs/01](docs/01-model-and-license.md) §4 now
names the revision we run.

---

## 2026-09-07 — The quality battery: two benchmarks won, one lost by four scenarios out of 88, and the explanation we liked was refuted the same night

**This repository had two cheap gates and one MMLU sample; it now has three benchmarks.** IFEval,
GSM8K and tool-eval-bench ran on production 12 on 6–7 September against the NVFP4 sibling recipe's
3 September battery — same harness build (`2.6.1.dev39`), same seed, same temperature, same effort
`low`, one engine, no restart between tests. **GSM8K 97.5 % against 94.0**, **IFEval 80.0 % prompt and
86.0 % instruction against 78.9 / 85.1**, and **tool-eval-bench 85.5 ±1.3 against 87.8 ±0.9**
`[measured-here]`. Two of the three rows in [docs/11](docs/11-open-issues.md) §3 that said "exists for
the sibling, never run here" close. New page:
[`results/gates/quality-battery-production-12.md`](results/gates/quality-battery-production-12.md).

**The number that moved the wrong way is the one worth the page, and it is four scenarios.** The
−2.3 points on tool-eval is not trial noise (Welch t = 4.02, exact permutation p = 0.0048) and it is
**not spread**: remove TC-51, TC-21, TC-74 and TC-87 and this stack is **+0.22 ahead on the other 84
scenarios**. Nine of fourteen categories are identical digit for digit. Across 1,408 scenario-runs on
both stacks, the only failure codes recorded are `wrong_args` and `missing_step` — **malformed tool
calls, timeouts, empty content, refusals and schema violations are all zero**, and `max_points` is 176
in both files, which is the direct proof that no scenario was dropped for an infrastructure failure.
The same engine the same night won GSM8K by 3.5 points.

**TC-51 is 47 % of the gap and it is a grading rule, read out of the grader's source.** The scenario
wants a calendar event created and the invitation sent; this stack does both correctly and issues them
**in the same assistant turn**, which `call.turn <= valid_event.turn` fails outright. The harness's own
request body sends `parallel_tool_calls: true` on every call, so the API layer invites exactly what
that scenario's Python forbids. Two of the other three are real and are written up as real — an
instruction violation on TC-74 and a filter inconsistency mid-pagination on TC-87.

**The explanation this repository preferred was tested as its own engine arm and refuted.** The two
batteries were served with different chat templates — the sibling's checkpoint ships `zai-org`'s
27 August template (`04c4e9e9`), this stack mounts the 4 September one (`690b7052`) — and every item in
the diff is in tool-call rendering. Production 12 was rebooted with the old template as the **single**
changed line (verified by hashing the file inside the container, not by reading a mount) and
**TC-51 and TC-21, 72 % of the gap between them, stayed 0/8** with the same trace and the same
`parallel tool turns: 2` diagnostic. The remaining block favours the old template by +1.12 points per
trial at **p = 0.20**, CI [−0.45, +2.70] — a template share of **28 %, 95 % CI [−11 %, +68 %]**, an
interval containing zero. **The new template stays in production**: the gain is unmeasurable and the
old one carries three real rendering defects, all on an agent's hot path. Boot, KV pool and gates were
unaffected on both arms and production was restored and re-verified.

**What stays open is the build against the weights, and it is filed rather than guessed at.** The
checkpoint and the vLLM build (with its different speculative-decoding implementation) changed
together and no measurement here separates them; the next step is a build A/B on the same eight
scenarios, about 12 minutes of engine time per arm. A same-session sibling control is not available.
[docs/11](docs/11-open-issues.md) §2.30, which also carries the behavioural note for agent users: this
stack batches two *dependent* tool calls into one turn, which is efficient and is wrong wherever the
second must not fire if the first failed.

**Two of the five staged tests did not run, and that is in the tables rather than in a footnote.** The
1M needle and the full 14,042-question MMLU were **deferred on time** after 3 h 24 min of engine
window, before either started. The 1,995-question MMLU sample at TP=3 on this checkpoint,
**86.47 ±0.74**, is unaffected and remains the only MMLU this stack has. Throttle mask was `0x0` on
every telemetry sample of every node in every test, so nothing on that page is thermal.

**And one row of ours was wrong in a way checking it fixed.** [docs/11](docs/11-open-issues.md) §3 said
ExtractBench "exists for the NVFP4 sibling recipe"; it does not. That recipe never ran it on the build
it ships either — the 94.51 it carries is from an earlier, different build, with its own note saying it
must not be quoted as one. ExtractBench is a dimension **neither** line has measured, which is not what
we had published.

---

## 2026-09-06 — Retracted: the indexer workspace does not fail loudly. The issue author corrected us, and checking it corrected one thing back

**We published a failure mode we had not read carefully, and the person we published it to caught
it.** Our confirmation on
[vllm#55221](https://github.com/vllm-project/vllm/issues/55221#issuecomment-5559336290) said that a
too-small sparse-indexer K-gather workspace ends in upstream's locked-workspace `AssertionError` — a
loud failure — and this repository built a four-layer safety story on that, with upstream's assertion
as the load-bearing layer. The issue author, [@drakosha](https://github.com/drakosha),
[replied](https://github.com/vllm-project/vllm/issues/55221#issuecomment-5561194190) that both
indexers ask the `WorkspaceManager` for a **static** size and only then slice per chunk, so the
request never grows and that assertion **cannot fire from this path**. Verified against the code we
run: they are right, and our own A/B had said so — exactly one resize event per rank. **The real
failure mode is a silent Python-slice clamp producing wrong answers**, which is the worse kind.
Retraction: [docs/11](docs/11-open-issues.md) §1.13.

**Checking it moved one detail back the other way.** Their replacement symptom is an out-of-bounds
device *write* by `cp_gather_indexer_k_quant_cache`. That kernel is destination-bounded —
`num_tokens = dst_k.size(0)` and a per-thread `token_idx >= num_tokens` early return, carrying
upstream's own comment about `num_tokens` being an allocation upper bound — at `808f8cd3ac`, the ref
their line numbers cite, at our image's base `487ecf187`, and at every commit sampled back to
`22a58640b4`. So no overrun from the gather: the rows past the clamp are never gathered, and the
wrongness travels to `fp8_fp4_mqa_logits`, handed `cu_seqlen_ks`/`ke` for an extent its `k_quant` no
longer has. The fp8 scale buffer *does* share the one `get_simultaneous` allocation, so that half of
the hazard stands.

**Nothing about the measurement or the configuration changes; the reason it is safe does.** The
load-bearing layer is **ours** — the startup refusal below the one-request floor — because the
splitter admits only while `new_n <= workspace_size` against the same bounded value, leaving the
`end == start` branch's single unsplittable request as the only route to an over-size chunk. Under
production 12 that needs an uncompressed context above **16,268,812 tokens** against a `max_model_len`
of 1,000,000: unreachable by construction, not by margin. Against the aggregate ceiling of
**2,000,016 entries (251.8 MB)** the 4,067,203-entry bound still holds **2.03×**. Arithmetic and the
code reading: [`results/memory/indexer-workspace-ab.md`](results/memory/indexer-workspace-ab.md) §2.

**And the transferable part of our note 2 is narrower than we wrote.** At `max_num_seqs` 32 the
`min(upstream, max(2 × ceiling, 512 MB))` formula our arm used frees nothing — drakosha's numbers,
and they are right. What transfers is the **exact ceiling without the 2× margin, plus the startup
guard below the one-request floor**; the win is dropping the `max_model_len` term, not the constant.
[#51252](https://github.com/vllm-project/vllm/pull/51252) is the right generic fix and is stalled; we
will not duplicate it into #55222. [HELP-WANTED](HELP-WANTED.md) §9 updated.

---

## 2026-09-06 — The CUDA-graph A/B: the lever is +1.75 % and closes the item, the graph pool at three ranks is a quarter of what we estimated

**We ran the A/B we had published as unrun, the same evening we published it.** Three env-only arms
on the production-12 tree in one 53-minute engine window: **A** production 12, **B** = A without
`HAREM_DRAFT_KV_DTYPE=fp8` (drafter → FlashAttention → `UNIFORM_BATCH` → graphs capture), **C** = B
plus `ENFORCE_EAGER=1`. **B − C is the graph lever at equal KV geometry, and it measured +1.75 % at
C1** `[measured-here]` — under the +2 % the design had set as the closing threshold. Every one of the
five concurrency levels is inside its band and **the signs are mixed** (C1 +1.75, C2 +1.18, C4 −1.75,
C6 −5.19, C8 −1.88 %). The item closes: **the patch is not written.** The wrong division stays filed
as [vllm#55581](https://github.com/vllm-project/vllm/issues/55581).
[docs/11](docs/11-open-issues.md) §2.29 has the run; raw rows in
[`results/speed/concurrency-sweeps.csv`](results/speed/concurrency-sweeps.csv).

**The finding is that our ruler cannot see it, and that is written down as the finding.** Two numbers
from the same run: the **round-to-round spread inside a single arm** was 3.52–10.69 % at C1
(per-stream 4.07–9.15 %), and arms B and C are separate boots, where this stack spreads 15.9 %
([docs/09](docs/09-measurement-protocol.md) §2). The signal is smaller than either. The one figure
that grazes a band — per-stream C1 **+4.45 %** — is buried by the same two, and is reported rather
than rounded away.

**The graph pool at three ranks, measured for the first time, is 0.25–0.27 GiB per rank.** `Graph
capturing finished in 44 secs, took 0.27 GiB` on rank 0, 0.25 on rank 1, **0.00 on rank 2**
`[measured-here]` — against the **1.10 GiB** this repository had carried over from the two-node track
and the estimator's 2.64 GiB. The **−2 to −5 % of pool** estimate came out right at **−3.85 %**
(6,796,143 → 6,534,435 tokens) **for the wrong reason**: the capture pool is nearly free and the
charge is the **peak-activation reservation the capture drives up, 1.66 → 2.62 GiB on rank 0**.
Estimate right, mechanism wrong, corrected in [docs/17](docs/17-memory-ledger.md) §2.3. And the
emphasis reverses on the full arm: B is −7.67 % of pool against production 12, of which +5.75 % is
the bf16 draft page (7,599 → 8,036 B per token) — **the pool goes to the draft cache, not to the
graphs.**

**A risk row retires.** The combination this repository listed as never run — TP=3 + expert parallel
+ `FULL_AND_PIECEWISE` + chunked prefill, the shape of [#40969](https://github.com/vllm-project/vllm/issues/40969) — ran a full battery with no hang,
no `EngineDeadError` and no error line. Gates were **10/10 and 12/12 cold and warm on all three
arms**, tool-call **8/8** on all three, acceptance **59.4–65.6 %** across every round with no approach
to 1.00 (so no [#53030](https://github.com/vllm-project/vllm/issues/53030) signature), swap-in 0 on
every node, and the throttle mask `0x0` on every telemetry sample of every node — so no arm's numbers
are thermal. Arm A's first warm code run read **11/12** on one item and **12/12** on an immediate
re-run on the same engine; recorded as a single-item flake under a re-run rule that was in force for
all three arms, not as a rejection.

**A − C, the fp8 draft cache at equal graph state**: 1–3 % slower (C1 −0.77 %, C8 −3.06 %) for
**+4.14 % of pool**. Read as the whole draft route rather than a dtype — A runs FlashInfer with the
XQA kernel and fp8, C runs FlashAttention with bf16. Production 12 is unchanged and was restored at
the end of the window: health 200, KV 7,030,303, gates 10/10 and 12/12.
[HELP-WANTED](HELP-WANTED.md) §11 keeps one ask, and it is a smaller one: **a second boot** of arms B
and C, because ours were one boot each.

---

## 2026-09-06 — The workspace bound at two ranks: +32.14 % of pool on an A/B, +26.5 % as shipped, and candidate C becomes the TP=2 recipe

**The same patch, the same file, three times the gain — and the reason is a denominator.** vLLM sizes
the sparse indexer's K-gather workspace as `40 × max_model_len` **entries**, which is 4.92 GiB at
`max_model_len` 1,000,000. **Every term of that expression is per engine, not per rank**, so two
nodes reserve exactly what three do, against a pool a third the size. The prediction written down
before the run was +27.5 % against the three-node track's measured +10.25 %; the measurement came in
at **+32.14 %**. New section: [docs/15](docs/15-tp2-track.md) §5.9; raw in
[`results/speed/tp2-production-candidate.md`](results/speed/tp2-production-candidate.md)
`[measured-here]`.

**A same-session A/B, one environment line apart, both arms booting eagerly** so that neither could
reuse a sidecar: locked workspace **5,036.40 → 513.00 MB** on both ranks — the three-node figures to
the decimal, and the patch's own startup line word for word, because it needed no two-node variant —
and KV pool **1,800,000 → 2,378,571**. Consumed memory per node 84.7/85.0 → 79.5/80.4 GiB. Gates
**10/10 and 12/12 cold and warm on both arms**, tool-call 8/8, needle-lite 6/6; one **969,468**-token
request correct; **eight concurrent ~128K lanes 8/8**, 640,904 prompt tokens at 2,225 tok/s of
prefill, each lane carrying its own needle; exactly one workspace resize per rank after the stress and
**none of the four safety layers fired**; swap 0.000 GiB.

**Shipped as candidate C, and the pool figure was predicted before it was measured.** A fresh 78 GB
per-rank sidecar and one 956 s dump boot later, the two-node recipe reads **2,128,571 → 2,692,857
tokens, +26.5 %**, at an identical 272 s fast-load boot. Candidate B's own eager→fast-load difference
is +310,714 tokens; adding it to the bound arm's eager figure predicted **2,689,285** against a
measured **2,692,857** — **0.13 % out**. [`tracks/tp2/`](tracks/tp2/) now carries the patch, the
prelude hook, the environment example and the unit; the previous recipe is one environment file away.

**Two things recorded as costs rather than rounded off.** First, **all five concurrency levels moved
the same way** — −1.16 to −2.63 %, mean −2.0 % — where the three-node arm read mixed signs. Every
level is inside its band and the per-stream, TTFT and acceptance rows are flat, but five out of five
in one direction is not what noise usually looks like, and **no clock, temperature or power telemetry
was sampled**; `vmstat` only shows that neither arm was CPU-starved, over windows of different
lengths, with the arms run in a fixed order 25 minutes apart. It is written down as **unexplained**.
Second, diagnostic margin: 20× the scheduler's largest possible load becomes 2.03×.

**The 1M gate failed first, and the engine was not the reason.** The probe scored `message.content`
alone; the first attempt returned an **empty** answer — not a wrong one — and at
`reasoning_effort: low` this model sometimes puts a short answer entirely into `reasoning_content`.
The identical request, same seed, passed on the next attempt with the code in `content`. Two
attempts, one empty, one correct, **no wrong code in either**, and the fixed probe now scores both
fields: [`bench/needle-1m-bothfields.py`](bench/needle-1m-bothfields.py). A gate that cannot tell an
empty answer from a wrong one is a ruler with a missing tick — the second time on this project that
the instrument, and not the engine, was the thing that failed.

---

## 2026-09-06 — CUDA graphs at three ranks are off because of a wrong declaration, not a kernel limit; filed upstream as vllm#55581, and left off

**No configuration changed.** The item the owner queued — "what did eager cost us?" — was answered by
reading code rather than by a boot, and the answer corrects three pages. Since production 7 every
TP=3 boot resolves `cudagraph_mode=NONE`, and this repository said that was because FlashInfer cannot
capture an 8-token verify batch. It can. FlashInfer's `get_cudagraph_support()` divides the *target*
model's per-rank query heads (22 at TP=3) by the *draft* group's KV heads (3) and, on the remainder,
declares `UNIFORM_SINGLE_TOKEN_DECODE`; the builder that actually runs uses the draft's own 12 heads
(12 % 3 = 0), passes the same test, and serves on the XQA decode kernel — `decode_backend=xqa` is in
the same boot log. The two-node arrangement of the same image, where the division is 32 % 4 = 0,
captures graphs (`Graph capturing finished in 11 secs, took 1.10 GiB`). Filed as
[vllm-project/vllm#55581](https://github.com/vllm-project/vllm/issues/55581) `[measured-here]`.

**Priced, and left off.** The ceiling is +1.2–2.3 % single-stream and +0.5–0.8 % at C8 — inside the
bands we grade in — and the graph pool is 1.1–2.6 GiB per rank out of the KV budget, −2 to −5 % of
the pool; the only env-only way to get graphs back (bf16 draft KV) costs −6.5 to −9.5 %. A fix on our
side would be a new patch file and therefore a sidecar re-dump. The three-arm A/B that would settle
the price is designed and listed as [HELP-WANTED](HELP-WANTED.md) §11.

**Corrected:** [docs/11](docs/11-open-issues.md) §1.9 row 8 — "the 36/9 drafter sidecar removed the
obstacle" — was wrong and is retracted as §1.12; the full item is §2.29. [docs/10](docs/10-results-and-roofline.md)
§5.8, [docs/14](docs/14-troubleshooting.md) §7.7, [docs/15](docs/15-tp2-track.md) §5.3 and
[docs/17](docs/17-memory-ledger.md) §2.3 carry the correction, and the TP=2/TP=3 comparison now names
the ≤2 % graph term in TP=2's favour. Both env examples say what `ENFORCE_EAGER=0` does and does not do.

---

## 2026-09-06 — The audit page grades against production 12, and the free-RAM floor leaves the script

**`audit/` had drifted one configuration behind the cluster.** Since 6 September its §1 said
production 11 was what the nodes serve while its §2 expectations were still production 9's, and the
promotion to 12 widened the gap: a reader grading a fresh boot against §2 would have compared a
7.04 M-token pool to a 5.17 M expectation and concluded something was wrong. **No measurement
changed here.** §2 is re-derived on production 12, production 9's figures are kept beside it in a
`[history]` column or a collapsed block, and the script's hard-coded constants follow, with the old
ones left in place as comments.

**Every speed row now carries the band it is graded in**, from the declared boot-to-boot spreads in
[docs/09](docs/09-measurement-protocol.md) §1.2 — C1 ±4 %, C2 ±6 %, C4 ±9 %, C6 ±6 %, C8 ±3 %, and
±6 % on the KV pool — rather than a bare figure a reader has to guess the tolerance for:

| | Production 12 | Pass range |
|---|---|---|
| KV pool | **7,041,322** (three boots: 7,170,798 / 7,088,154 / 7,041,322) | **6,618,843 – 7,463,801** |
| C1 aggregate / per stream | **69.72** / 75.55 tok/s | **66.93 – 72.51** |
| C8 aggregate / per stream | **196.06** / 28.44 tok/s | **190.18 – 201.94** |
| C2 · C4 · C6 | 101.20 · 146.14 · 176.00 | 95.13–107.27 · 132.99–159.29 · 165.44–186.56 |
| Prefill, fresh | 1,744 tok/s | 1,692 – 1,796 |
| TTFT C1 / C8 · acceptance | 0.25 / 0.80 s · 62.5 / 62.4 % at 5.35 per step | — |
| Consumed per node · available KV per rank | 54.62 / 54.48 / 54.28 GiB · 50.75 / 50.89 / 51.09 GiB | — |
| Boot: fast-load · `systemctl start` · power-on · dump | 272 · 205 · **311** · 590 s | — |

**The memory section is the one that needed rewriting rather than re-numbering.** It graded on
`MemFree` against a 4 GiB floor, and the script printed that floor as a RULE line; both are
`[retracted]` and both are gone. What replaces them is the criterion the 6 September ladder was
actually climbed with, with its thresholds stated rather than implied: swap **traffic** under load
(`si`/`so` from `vmstat -n -t 1`, summed over the window between "engine up" and "battery done"),
C1 inside ±4 % and C8 inside ±3 % of a same-session reference, and both gates full cold and warm —
three conditions together. Swap *used* is a stock, sits near 0.04 GiB at every fraction including the
one that failed, and discriminates nothing. Both ends of the scale are printed so "≈ 0" is readable:
production 12 passed with swap-**in** exactly zero everywhere and 1,340 / 5 / 10 KiB out in 8 of 353
samples (longest unbroken run 10 s), against the rejected 0.90 rung's 1,519 MiB out **and 143 MiB back
in**, 250 of 598 samples, 85 s unbroken. `MemAvailable` is published as what it is — a budget, about
2 GiB at 0.88, not a gate — with the 1.2 GiB per 1 % of fraction that sizes any rung you are
considering. The script grew a `pswpin`/`pswpout` snapshot on the same
"read it twice and diff" pattern the fabric counters already use.

**Two boot gates added, because production 11 and 12 each brought a guard patch.** `SM12
items=pdl,kpool`, and the `HAREM-IDXWS bound` line with its `headroom 2.03x`. Beside them, the check
that matters most on a bounded buffer: `VLLM_DEBUG_WORKSPACE=1` is upstream's own variable and prints
**exactly one** `0.00 MB -> 513.00 MB` resize per rank, independently of anything our patch claims
about itself. A second line on a rank means the buffer grew after `lock_workspace()` and the pool
just read is not the one that will be kept — so the section says to re-read it after a long-context
load, which is when it would happen.

**Two things are marked `[not tested]` rather than carried forward silently:** category speed, which
is production 9's, and the MMLU sample, which is production 9's 86.47 ±0.74 on 10, 11 and 12 alike.
A memory fraction and a buffer bound cannot touch what the drafter accepts — but that is an argument,
not a measurement.

**§5 and §6 follow.** The provenance table's "Production 12 candidate — **not in production**,
`production-configurations.csv` carries no row 12" was true when it was written that afternoon and
was false by the evening; it is corrected and three shipped rows are added (speed and pool, host
memory and swap traffic, quality and the whole-cluster reboot). Retraction 2.4 now records the whole
arc rather than half of it: the ladder passed 0.88 and declined it on diagnostic headroom, and the
same rung shipped hours later as production 12. **The measurement did not change; the decision about
what the cluster is for did.**

---

## 2026-09-06 — Production configuration 12: the workspace bound shipped, and the rung we had declined

**Production 12 is production 11 with exactly two changes, taken in one boot:**
`gpu-memory-utilization` 0.87 → **0.88** and the sparse-indexer K-gather workspace bound
(`HAREM_INDEXER_WS_MODE=bound`), which moves from `patches-optional/` into
[`tracks/tp3/patches/indexer-workspace/`](tracks/tp3/patches/indexer-workspace/). Measured against a
**same-session production-11 reference** — the same three-round battery on the running engine, no
restart `[measured-here]`.

**KV pool 6,385,674 → 7,041,322 tokens, +10.3 %**, and 7,170,798 on the best of three boots. Speed,
pooled over six rounds and two boots the way configuration 11 was: **C1 +0.08 %, C8 −0.08 %**, C2
+2.40 %, C4 +3.38 %, C6 −0.46 % — every level inside its band and the two banded ones within a tenth
of a percent. Prefill 1,739 → 1,737 and 1,750. Consumed memory per node **58.3–59.1 → 54.3–54.6 GiB**,
which is the 4.42 GiB the patch releases, read off the far side.

**The prediction was written down first and checked in absolute tokens.** The A/B priced the bound at
+644,628 tokens and the memory ladder priced the rung at +179,063; the sum predicts **7,209,365**
against a measured 7.04–7.17 M, inside the documented 6 % boot-to-boot pool spread. The two gains add
in tokens and **not** in percent, because the second lands on a base the first has already enlarged —
which is why the headline is +10.3 % rather than the +13.4 % a percentage sum would promise.

**What had to survive for it to ship.** A workspace bounded at 2.03× the scheduler's ceiling is only
safe if the ceiling is right, so promotion was gated on the two cases that can crowd that buffer, and
at the *higher* fraction where host memory is thinner: one **969,468-token** request, correct in
569.6 s, and **eight concurrent ~128K lanes**, 8/8, 640,904 prompt tokens in 227.5 s. After both,
every rank still logs **exactly one** workspace resize — `0.00 → 513.00 MB` — and zero assertions:
none of the patch's four safety layers fired. Gates 10/10 and 12/12 cold **and** warm on both boots
and again on the systemd-started engine; tool-call 8/8; needle-lite 6/6.

**The memory criterion, which is the whole argument for 0.88.** Swap **in** was exactly zero in every
sample on every node, in both windows. Swap *out* was 1,340 / 5 / 10 KiB on the arm that also ran the
two stress cases — 8 of 353 samples, longest unbroken run 10 s — against the reference's own
10 / 5 / 5, and 5 / 10 / 10 with swap **used** exactly 0.000 GiB on the clean boot. For scale, the
rejected 0.90 rung was 1,519 MiB out **and 143 MiB back in**, 250 of 598 samples, 85 s unbroken.
OOM-killer 0 on all three; the `NVRM ... NV_ERR_NO_MEMORY` lines in `dmesg` are the profiler's ceiling
probe and appear on every boot at every fraction.

**What it cost, and the line is not left empty.** `MemAvailable` under load falls to
**1.5 / 3.4 / 3.4 GiB** on the stress arm (3.2 / 4.5 / 4.5 on a clean boot). That is the budget
anything running *beside* the engine lives in, and at 0.88 it is about 2 GiB — a profiling run now has
to stop the engine first. This is exactly the headroom the 6 September ladder declined 0.88 over; the
decision changed, the measurement did not. Second, the indexer's own diagnostic margin goes from 20×
the largest load the scheduler can produce to **2.03×** — against a startup refusal, two armed
run-time guards and upstream's own post-lock assertion, none of which fired. Third, one 590 s dump
boot and 53 GB per node, paid for by deleting the production-9/10 sidecar; the production-11 sidecar
was kept, which is what makes the rollback a single `cp`.

**`VLLM_DEBUG_WORKSPACE=1` is now carried in production**, not just in the A/B. Upstream's own
variable, three INFO lines a boot, nothing in the hot path — and it is what makes "did the buffer
really reach 513 MB, and did it grow afterwards" readable from the engine's own log on every future
boot.

**Boot from power-on, all three nodes rebooted together, nothing touched:** `/health` 200 at **311 s**
against configuration 11's 312 s — unchanged. Units active and enabled on all three, fabric 4/4, gates
read afterwards. Fast-load container boot **272 s** (weights 80 s); `systemctl start` 205 s.

**Rollback** is `cp` of the production-11 environment file and a restart — its sidecar is still on
disk — or, to disarm just the patch, deleting one word from `EXTRA_ENV`, which costs no boot at all.

---

## 2026-09-06 — 4.92 GiB of dead buffer inside "non-torch": the indexer workspace, bounded, +10.25 % of KV pool

**The 7.28 GiB "non-torch" line of [docs/17](docs/17-memory-ledger.md) had never been broken down, and
the largest thing inside it was a single buffer nothing reports.** vLLM sizes the sparse indexer's
K-gather workspace as `40 × max_model_len` **entries** — a constant chosen upstream against
DeepSeek-V3.2's 163,840-token context, where it comes to 825 MB. At `max_model_len` 1,000,000 the same
constant reserves **4.92 GiB**, during the profile run, locked for the life of the engine, and charged
to the residual the profiler subtracts *before* it sizes the KV pool. It appears in no line an
operator reads. New page:
[`results/memory/indexer-workspace-ab.md`](results/memory/indexer-workspace-ab.md) `[measured-here]`;
the patch is [`tracks/tp3/patches/indexer-workspace/`](tracks/tp3/patches/indexer-workspace/).

**The hypothesis was tested before the patch was.** `VLLM_DEBUG_WORKSPACE=1` is an upstream variable
and needs no patch: the control arm printed **one** workspace resize per rank, grown by
`sparse_attn_indexer_kpool.py`, at exactly the **5,036.40 MB** the code reading had predicted. No
other consumer sets that buffer, so the gain was not capped by somebody else's requirement — which was
the most likely way for the whole item to be worth nothing.

**Bounded to 512 MB — 2.03× the exact ceiling — on a same-session A/B at production 11's 0.87, one
line between the arms:** locked workspace 5,036.40 → **513.00 MB**, KV pool **6,289,256 → 6,933,884
tokens, +10.25 %**; C1 69.69 → 70.69 and C8 199.76 → 196.81 aggregate, every level inside its band;
gates 10/10 and 12/12 cold **and** warm on both arms, tool-call 8/8, needle-lite 6/6; one
**969,468-token** request correct in 572.4 s; **eight concurrent long-context lanes**, 640,904 prompt
tokens, every lane's needle correct; swap 0.000 GiB and **none of the four safety layers fired**.

**What it cost, and the line is not left empty.** Diagnostic margin: the indexer had 20× the largest
load the scheduler can produce and now has 2.03×, against a startup refusal and two armed run-time
assertions. (This entry also credited upstream's locked-workspace assertion as a fourth layer; it
cannot fire from this path — retracted in the 6 September entry above.) Speed: nothing measurable,
looked for at
five concurrency levels, in fresh prefill and in TTFT. Host headroom: **nothing, in either direction**
— the freed memory goes straight into the pool, so `MemAvailable` is unchanged and this is not a lever
on the rung where 0.90 failed. It stacks with the ladder rather than competing with it.

**It is a candidate, not a configuration.** Installing it into the production tree changes the
fast-load manifest identity and forces a fresh ~53 GB-per-node sidecar, which does not fit until an
older one is deleted — an owner's decision. Both A/B arms therefore ran from a copy of the tree with
fast-load off, which is the item's one open caveat: the production pool figure with fast-load restored
is an `[estimate]` until a promoted boot prints it. Narrative:
[docs/11](docs/11-open-issues.md) §2.28; ledger context: [docs/17](docs/17-memory-ledger.md) §2.5.
**The upstream half — a sizing constant linear in `max_model_len` that skips the `compress_ratio`
division the DeepSeek-V4 call site applies — turned out to be filed already**, three days before we
measured it, as part 2 of [vllm-project/vllm#55221](https://github.com/vllm-project/vllm/issues/55221)
(fixes [#55222](https://github.com/vllm-project/vllm/pull/55222) and
[#51252](https://github.com/vllm-project/vllm/pull/51252)). We added our numbers and three points to
[that thread](https://github.com/vllm-project/vllm/issues/55221#issuecomment-5559336290) on 6
September rather than opening a second issue: the two fixes want to land together or the chunker's
budget stays 4× its buffer; after the division the size is still linear in `max_model_len`, which the
`max_num_seqs × per-request` bound removes (0.73 of our 4.42 GiB); and the 40's justification is
borrowed from the `flashmla_sparse` workspace, which calls itself "free" because it sits below a MoE
workspace — where our control arm logged exactly one resize event, the indexer's.
[HELP-WANTED](HELP-WANTED.md) §9 now carries what is still open.

---

## 2026-09-06 — the MLA-prefill ceiling, falsified on our own hardware: the kernel author's correction holds, tighter than on his card

**A prediction from someone else's issue thread, run on our own GB10 the same day it was made.** The
`cuda-exl3` author corrected his own MLA-prefill ceiling in commit `5fd7299` after our indexer datum
([`results/kernels/sm12-stack-patches-ab.md`](results/kernels/sm12-stack-patches-ab.md) §8: median
2,049 selected keys per row, adjacent-row overlap 0.9258) showed production sat two orders of
magnitude from the low-turnover arm his benchmark had been comparing it against, and predicted that on
a 48-SM part the production arm would land within a few percent of a fully cache-resident arm rather
than between it and a gather-bound one. New page:
[`results/kernels/mla-prefill-falsification-gb10.md`](results/kernels/mla-prefill-falsification-gb10.md)
`[measured-here]`; fixture in [`bench/mla-prefill/`](bench/mla-prefill/).

**The prediction held, at both head counts, and this card asked the question harder than his did.**
His fixture, copied verbatim, run against the production kernel (`cuda-exl3 754421f`) inside a window
where the engine was already down on all three nodes: 17 s, 1.65 GiB peak. At 262K context, production
sits **+1.3 %** over a fully cache-resident arm at our own 22-head TP=3 shape — tighter than the
**+1.6 %** he measured on his 188-SM, 128 MiB-L2 card — and **+5.4 %** at his 16-head shape run here,
against an independent arm needing **82.0–146.8 %** more time. Across all six cells (three contexts ×
two head counts) production closes **96.0–98.4 %** of the independent→drifting distance, and its
excess over drifting at 262K (223–575 µs) is below one cold read of its own 187 MiB working set —
three orders of magnitude under the full gather. **The "21–26 % overlap gap, about 2 % of a prefill
chunk" this item was carried at does not exist. It closes at zero on both parts.** GB10's own
bandwidth-to-compute ratio being roughly 6× lower than his made the test sharper, not weaker: this
card separates the two extreme arms by 1.82–2.47×, against his 1.46×, and the answer was the same
regardless.

**Closing it produced one new item.** At 22 heads — this stack's own TP=3 per-rank count — the
kernel's compute-only floor costs **13–16 % more per head** than at his 16, reproducibly at every
context and isolated to the compute path rather than traffic: the first concrete candidate for the
"reduce the kernel's own work" lever `5fd7299` says is what remains. **And one caveat travels forward
rather than closing here.** The correctness gate on this fixture's timings ran at a small 2-head
shape; a correctness matrix for the real 22-head shape was added the same afternoon and its CPU-only
self-test passed 24/24, but the GPU run itself is still queued for an engine-free window
`[not tested]`. Both the 22-head numeric check and the per-head cost anomaly are now
[HELP-WANTED](HELP-WANTED.md) §8, which used to ask for the falsification itself. Narrative and
closure: [docs/11](docs/11-open-issues.md) §2.27.

---

## 2026-09-06 — production configuration 11: the memory ladder to 0.87, and the sm_12x guards that ride with it

**Two changes in one boot, and the second one paid for the first.** `gpu-memory-utilization` 0.83 →
**0.87** and the **sm_12x correctness set** — [`patch-pdl-gate.py`](tracks/tp3/patches/patch-pdl-gate.py)
(item 1) and [`patch-kpool-init.py`](tracks/tp3/patches/patch-kpool-init.py) (items 2 + 3), behind
`HAREM_SM12_ITEMS=pdl,kpool`. The memory fraction is not part of the fast-load manifest identity but
the patch files are, so the patches force a dump boot (590 s, ~53 GB per node) that the rung rides
along on for free.

**Measured against a same-session production 10 reference**, pooled over six sweep rounds and two
boots: **KV pool 5,694,214 → 6,382,920, +12.1 %**; C1 69.2 → **69.6** (+0.6 %), C8 197.5 → **201.1**
(+1.8 %), prefill 1,742 → 1,760, TTFT unchanged at 0.278 / 0.806 s. Gates 10/10 and 12/12 cold **and**
warm on both boots, tool-call **8/8**, needle-lite **6/6** at 64K and 128K. The cost is host headroom:
`MemAvailable` under load 8.3 / 9.1 / 9.6 → **3.4 / 4.7 / 4.7 GiB**.

**The ladder, and the ruler that had to be replaced first.** New page:
[`results/memory/ladder-6sep.md`](results/memory/ladder-6sep.md). 0.85, 0.87 and 0.88 all pass; **0.90
is rejected**. The criterion is no longer `MemFree` against a 4 GiB floor — on a unified-memory part
most of what the kernel holds there is reclaimable page cache — but **swap traffic under load**,
`si`/`so` per second summed over the benchmark window. Swap *used* is a stock and sits at ~0.04 GiB at
every rung including the failing one.

**At 0.90 every concurrency level is still inside its band** while the head node pages 1,519 MiB out
and reads 143 MiB back, 250 of 598 load seconds are non-zero, and the longest unbroken run is 85 s. A
ladder judged on tok/s would have shipped it. What surfaces at the client is the arm's first prefill
going **5.0 → 9.8 s**. **0.88 passed and was not taken**: 1.86 GiB of `MemAvailable` and 2.59 GiB of
page cache left on the head node, against 0.87's 3.49 — and that headroom is what anything running
beside the engine lives in. Every 1 % of the fraction costs about **1.2 GiB** of host memory here.

**Two of our own verdicts are withdrawn.** "0.85 will not be attempted on this stack" and "0.88 was
never attempted" `[retracted]` — both were true of a boot that predates the fast-load page-cache fix,
and both were reached with the wrong ruler. Corrected in [README](README.md),
[docs/00](docs/00-hardware-and-os.md) §11, [docs/07](docs/07-kv-and-draft-page.md) §6,
[docs/11](docs/11-open-issues.md) §2.4, [docs/16](docs/16-comparison-with-published-recipes.md) §5.3
and in both `results/configs/` CSVs, where the superseded note is kept beside the correction rather
than overwritten. Note what §5.3 now says: **0.87 is where one published recipe crash-loops and two
others see swap.** A published rung is not transferable; only the method is.

**The sm_12x set was adopted, not re-litigated.** Yesterday's A/B closed with "adoptable, free, but a
change without a reason is not made to a configuration that passed a reboot test — ride with the next
production change" ([docs/11](docs/11-open-issues.md) §2.27). This was the next production change. The
two files moved from `patches-optional/sm12/` into the track's own patch tree; **item 4 did not** — what
it removes could not be measured from the client in either direction, so there is nothing to weigh.
The prelude's item list is now **fail-closed**: an unrecognised entry stops the rank instead of being
skipped, because silently ignoring `kpol` would serve a stack the operator believes is patched.

**One patch cannot be applied from inside the container**, and the pattern is written down because it
will recur. `patch-kpool-init.py` targets `sparse_attn_indexer_kpool.py`, which the launcher
bind-mounts **read-only** from `$OVERLAY_DIR`. The way through is a host-side copy of the overlay with
that item pre-applied; the prelude then reports "already applied" for it and writes only the image's
own file. Verify the copy with `diff -r`: exactly one file, exactly one patch's hunks.

**Reboot-tested as a whole cluster, and the reboot doubled as the noise check.** `/health` 200 at
**312 s** from the reboot command, against production 10's 315 s by the same wall clock — autostart
cost unchanged; all three units up on their own, ConnectX-7 4/4, KV 6,382,920, gates full cold and
warm, swap traffic exactly zero. It also settled the campaign's one open number: production 11's load
boot had read **C4 135.0** against the reference's 144.2, and the clean boot read **145.9**. C4's
six-round spread on this configuration is **11.1 %** against 2.5 % at C1 and 5.3 % at C8; the low
reading is boot noise and it is printed rather than smoothed.

---

## 2026-09-06 — the memory ledger: where 121.6 GiB per node goes, and the six give-backs that measure zero

**A read-only accounting of every gigabyte on a node, with the engine left running.** No boot, no
container `exec`, no configuration change, no lock: `docker inspect`/`logs`/`stats`, `/proc/meminfo`,
`ps`, `nvidia-smi` and the image's own source. New page: [docs/17](docs/17-memory-ledger.md), the
nineteenth document and both tracks'.

**The ledger.** Engine CUDA allocation **99.06 GiB** = weights 51.62 + KV pool 40.12 + non-torch 7.28
+ CUDA graph **0.00**; host anonymous 5.60; page cache 5.80; free 5.08; slab 2.36; and a **3.49 GiB**
residual that appears again at **3.12 GiB** with the engine down, which is what makes it the driver's
fixed reservation rather than a rounding error. Total genuinely unavailable at idle: **4.84 GiB**.

**"The driver takes 14.2 GiB" is stale by about 10 GiB.** That figure came from an NVFP4-era ceiling
scan whose best observed free was 107.43 of 121.63. Today the three ranks start at 112.01 / 112.97 /
113.15 and an idle node offers 116.79. What limits `gpu-memory-utilization` now is **the host's share
at run time** — worker RSS 4.71 GiB per rank plus API server 1.81 and EngineCore 1.56, plus page
cache — not the driver. This is a ruler correction, not a lever.

**What a KV block is.** 3,328 tokens, **20,934,400 B**: MLA latent fp8 **89.53 %**, indexer k fp8
5.77 %, DFlash2 draft 4.70 %. 40.12 GiB over that gives the 2,058 blocks the log prints. A 1M-token
request costs **363 blocks = 7.078 GiB on every node**, and the pool holds **5.67** of them — so at
three nodes the pool is not the binding constraint. At two it was.

**The largest avoidable line is a counter, not memory.** In the align path
`MambaSpec.max_memory_usage_bytes = page × (2 + num_speculative_blocks)`, so with DFlash2 at k=7
**every KDA layer holds 9 state slots per request, seven of them purely so a rejected speculative step
can be rolled back**: 36 blocks per request, **9.9 % of the divisor at TP=3 and 12.9 % at TP=2**, and
**5.61 GiB — 14 % of the pool — at 8-way concurrency**. The slot count multiplies the *block count*,
not the page, which is why the forced 3,328-token attention block (4,608 at TP=2, and it is
`platforms/interface.py` that forces it, silently overriding our `--block-size 256`) would not shrink
if the slots did.

**Six candidates are closed at zero, by arithmetic and reading rather than by boots.** The draft page
of 256 is the optimum — 384 is flat, 512 loses. The CUDA-graph estimator charges **0.00 GiB** here
where it was a 3.9 GiB item on the NVFP4 sibling, so that lever does not exist. The unused MTP layer
is not in the safetensors index and is never read. Rank duplication is ≈ 0. The fast-load path already
drops the checkpoint's page cache. `/dev/shm` holds 67 MiB of a 61 GiB tmpfs. **Shrinking the
attention block also closes**: it adds +0.2 % on top of a slot fix, because both dilute the same 62
fixed blocks.

**ReplaySSM, read in full and not adopted.** A third-party patch lifts upstream vLLM's own refusal to
run ReplaySSM alongside drafting and takes the slots from 9 to 2. It applies **127/127 hunks clean**
to our image, is pure Python and Triton with **zero CUDA**, touches nothing in `cuda-exl3`, EXL3, the
GEMM path or MLA, and needs about **8 hours** to port. A pool model that reproduces **four**
independent measured arms to the token projects **5,619,834 → 6,071,684, +8.0 % at TP=3** and
**1,303,571 → 1,429,245, +9.6 % at TP=2** — lower than the slot arithmetic suggests, because the ring
lives inside the mamba page and pushes the forced block from 3,328 to 4,096. Against that: the replay
loop is `tl.static_range` and runs its full length every step, taking the sequential work from 8 to
**24 — three times** — in a class that is 8.67 % of a C8 step, so the arithmetic worst case is
**C8 197 → about 168 tok/s**; and the ring holds its inputs in **fp16** where the baseline state is
fp32, with no test in the upstream harness that compares the ring against a baseline that we could
find. **Not now**, and if it is ever taken the first thing to run is one A/B boot at B=8 for speed
and gates only: if C8 loses more than 5 % the item closes there.

**One pair of our own boots does not reconcile, and it is written down rather than smoothed.** The
ledger boot logs 40.12 GiB over 20,934,400 B per block; the model that reproduces production 10's
pool uses 42.89 GiB over 22,572,800 B. Each is exact against its own boot and neither can be checked
against the other, **because the production-10 arm script grepped the pool figure out of the
container log and discarded the log**. The fix is ours: the three-node arm scripts must keep
`docker logs` as `kv.txt`, the way the two-node harness already does.

**And a correction to our own working note, kept because it changes what a fix looks like.** The nine
slots can be read as `1 + 8` off the non-align branch of that `if`, and our first pass did read it
that way. Align is the branch we run, so it is `2 + 7`. No total on the page moves — nine either way —
but it is why the compact-rollback alternative goes to **two** slots and not to one.

**The two-node column is published with its own warning.** It comes from the experts-only stack and
was read while the two-node full-scope candidate was taking a **dump** boot, whose pool figure reads
about 6.7 % low and must never be quoted as a result. Re-take it from that candidate's load boot.

---

## 2026-09-06 — the all-reduce latency floor, measured byte by byte, and three of our own rulers disagreeing

**The decode collective has no bandwidth term in it, and now there is a curve that says so.** A
byte-precise sweep — 8 B to 64 MiB doubling, 10 warmup and 50 timed iterations per size, `busBW =
algBW × 2(n−1)/n`, a correctness check at every size (**132/132 passed**) — on the production plugin
build and the production NCCL environment `[measured-here]`. Full page:
[`results/mesh/nccl-latency-sweep.md`](results/mesh/nccl-latency-sweep.md), raw beside it.

**Decode is latency-bound; prefill is at the wire.** 8 KiB **74.68 µs**, 64 KiB **86.40 µs**, 1 MiB
**275 µs** (5.08 GB/s), 16 MiB **1,096.79 µs** — **20.40 GB/s, or 98.1 % of the 20.8 GB/s we measure
on our own cable** — and 64 MiB 3,955 µs (22.62 GB/s). **From 8 B to 32 KiB, a 4,096× range of sizes,
the time never leaves 72–85 µs.** There is no software headroom left at the prefill end; the levers
that remain are fewer, larger, or overlapped collectives, none of which is a transport change.

**`NCCL_MAX_NCHANNELS=8` does nothing at decode and is worth up to 11× in the cliff.** At 8 and 64 KB
the difference against the default is inside noise, because a fixed base latency cannot be reduced by
adding channels. At 1 MiB it is 275 µs against 1,388, at 4 MiB 306 against 3,554, and at 64 MB the
difference disappears again because bandwidth is already saturated. **This is a property of a
multi-NIC mesh, not of NCCL** — the `cuda-exl3` author's channel sweep on a single-host no-P2P box is
flat `[reported]`, and a one-host reader should not expect the effect at all. `NCCL_PROTO` auto is at
least as good as every forced protocol at every size and clearly better between 128 KiB and 1 MiB,
where `Simple` is 4–7× slower — the same verdict he reaches on very different hardware.

**Three of our own harnesses disagree at small sizes, and all three readings stay.** At 64 KB:
`ar_bench.py` 61.3 µs, this new sweep 86.4 µs, `mesh_sweep.py` 143 µs. At 8 KB: 38.6 against 74.68 —
a factor of two. They differ in more than one variable at once (iteration count, warmup, the timing
loop, and for the oldest rows a plugin build and cabling state from before the dual-cable work), so
**no cause can be assigned from the data that exists and none of them is corrected against the
others**. Recorded in [docs/06](docs/06-nccl-mesh.md) §12.1 and in
[`results/mesh/all-reduce-sweep.md`](results/mesh/all-reduce-sweep.md) §5; a same-session comparison
is now the cheapest item on [HELP-WANTED](HELP-WANTED.md) §5 and it needs no new code, because all
three tools are in `bench/`. **The conclusions on the new page do not rest on the absolute value:**
the decode verdict comes from the flatness, which every harness shows, and the prefill verdict comes
from a 16 MiB point where they agree within a few percent.

**The tool is published with a warning label.** [`bench/nccl-latency-bench.py`](bench/nccl-latency-bench.py)
and its two runners reproduce `nccl-tests`' contract exactly — the same `-b/-e/-f` sweep convention,
the same `-w 10 -n 50`, the same busBW formula, the same `-c 1` check — but **the numbers do not come
from the `nccl-tests` binary**, because neither `nccl-tests` nor MPI exists on these nodes and
`nccl-tests` requires MPI for multi-node. [`bench/README-nccl-latency.md`](bench/README-nccl-latency.md)
says that first and says how to run it beside a live engine without touching it: a cpuset outside the
engine's cores, a memory cap, the rendezvous address set in the runner rather than sourced from the
engine's environment file, the measurement lock, and an idle check before the first byte moves.

**What this measurement cost.** Nothing. No engine restart, no configuration change, no code change:
the engine stayed up and idle throughout, the lock cleared after 720 s of waiting for another arm,
and the sweep held the fabric for 94 s.

**And one stale figure is buried properly.** The "13.9 GB/s against 50 GB/s = 28 %" that was quoted
upstream was already stale when it was quoted and is doubly wrong now: the 50 GB/s denominator was
retracted here in favour of the PCIe Gen5 x4 ceiling ([docs/11](docs/11-open-issues.md) §1.7), and
the numerator moved with two rounds of transport fixes. It is 20.40 against 20.8, or 98.1 %.

---

## 2026-09-06 — four sm_12x patches, measured: no race, no cost, and no reason to ship them

**A five-arm A/B on four reported sm_12x defects, and the answer is "nothing moved" in every
direction we could measure.** Production configuration 10 was never modified: the patched arms ran
in a separate image (`exl3-zeus:e7e345e-dflash`) and a separate in-container tree, and the patches
were verified absent from the live container file by file when the campaign ended. Full tables:
[`results/kernels/sm12-stack-patches-ab.md`](results/kernels/sm12-stack-patches-ab.md); the three
adoptable patches are in
[`tracks/tp3/patches-optional/sm12/`](tracks/tp3/patches-optional/sm12/), outside the tree the
prelude applies; narrative in [docs/11](docs/11-open-issues.md) §2.27.

**Byte identity was abandoned before it cost a boot, and that is the method half of this entry.** The
kernel author ran that experiment on his own hardware first and got 24 distinct completions out of 24
greedy runs **in both arms** — the fused MoE epilogue's bf16 `atomicAdd` alone puts a run-to-run
varying 4.34e-3 per layer under everything, and a TP=3 all-reduce is not bit-exact either. The
replacement, [`bench/logit-divergence/`](bench/logit-divergence/), measures the **within-arm** floor
first and only then looks above it: 12 fixed prompts, temperature 0, `max_tokens` 128,
`top_logprobs` 5, concurrency 1, a per-run `cache_salt` so run 2 recomputes its prefill, and a prompt
is an outlier only if it clears K = 4 times the pooled floor **or** diverges earlier than any
within-arm pair did.

**The PDL gate: no race found.** vLLM's `is_arch_support_pdl()` returns `major >= 9`, so PDL is on
for sm_121 across every KDA and mHC launch site, and 34 of this model's 45 layers are KDA. The
between-arm p95 came out **below** the within-arm p95 — 6.2–6.5e-02 against 7.7–8.0e-02, i.e. two
arms differ less than two runs of one arm — with no prompt above K = 4 and, in the first comparison,
later token divergence between arms than within one. On speed it is a wash on 48 SMs: **C1 +0.8 %,
C8 +0.7 %** pooled over six PDL-on and three PDL-off rounds. The one reading outside its band, C1
per-stream decode at +5.4 %, was deleted by a repeat — one arm read 69.75 (a nominal +10.8 % for PDL
off), the identical configuration repeated read 76.17, between the two PDL-off readings.

**Items 2–4 are free, and one of them is smaller than it was reported to be.** The K-pool buffer's
initialisation and its reader's missing upper bound cost nothing measurable, including the per-chunk
int32 memset the preparation note had priced as "not free on this part". Item 4 is at most **one**
cold Triton compile per engine process — triton 3.7.1 gives an int argument three specialisation
classes, not one per value, and the tree's own warmup already covers most of them — and across cold,
freshly booted processes it could not be seen from the client at all. Gates were 10/10 · 12/12 in
every arm, with one warm single-question flake that did not reproduce in three immediate re-runs.

**The decision, stated as a distinction rather than as a verdict.** Items 2 and 3 are
correctness-class: their value rests on a mechanism, not on a measurement, and this A/B **did not
show that mechanism firing** — it showed that fitting the insurance costs nothing. A configuration
that passed a three-node reboot test is not changed without a reason, so the patches ride with the
next production change or stay optional.

**The larger result came from the instrument, not the patches.** The campaign's diagnostic hook
produced the MLA-prefill datum `cuda-exl3` issue #5 had asked for: **median 2,049 selected keys per
query row and a median adjacent-row top-k overlap of 0.9258** over 7,168 rows — about **152 keys
turning over per row**, roughly **76×** the ~2 keys of the author's low-turnover arm, so production
was never between his two arms. He built a third arm at that turnover, swept context, and corrected
his own conclusion in `5fd7299`: at **262K context** the production-pattern arm is within **1.6 %**
of the fully cache-resident arm (2,422.8 against 2,385.8 µs) where the independent arm needs
3,474 µs, because the live key set is the **residence window** (~4,096 keys, ≈4.5 MiB) and not the
chunk footprint. **MLA prefill is compute-bound at production overlap and the "21–26 % overlap gap,
about 2 % of a prefill chunk" does not exist — the item closes at zero**, and the only lever left on
that kernel is less work rather than less traffic. Recorded in [docs/10](docs/10-results-and-roofline.md)
§7 and [CREDITS](CREDITS.md); the one-bench falsification on a 48-SM part is
[HELP-WANTED](HELP-WANTED.md) §8 and we have not run it.

**Two process corrections, both of which cost more than the result.** The diagnostic image had been
built with only the first of the production image's two stages, so the DFlash2 port's thirteen files
were missing and a fail-closed anchor stopped the first dump boot — the build note recorded the
single stage without recording what it cost. And the GB10 top-k overlay is bind-mounted **read-only**
over the file two of these patches target, so those two had to be applied to a host-side copy of the
overlay rather than by the prelude; that exception applies to any future patch landing on an overlaid
file.

**Also recorded: the instrument's own defect.** Its derived `context`/`expected`/`deficit` columns mix
pool-granular and token-granular units and are wrong; the two distributions published are counted
straight off the selection buffer and are unaffected. The instrument is not shipped here.

---

## 2026-09-06 — one repository, two tracks, and a front door in front of both

**This repository had grown a second working node count without growing a way to tell which half a
reader was in.** Nothing measured changed today. What changed is the shape: one folder per track for
the files that actually differ, a badge on every page saying which track it belongs to, a router that
asks how many nodes you have, and a ranked list of what a second cluster could settle.

**`tracks/`, and the rule for what goes in it.** A file belongs to a track only if it cannot be
shared: the environment template, the in-container patch tree, the autostart unit and its preflight.
`patches/tp3full/` is now [`tracks/tp3/patches/`](tracks/tp3/patches/) and `patches/tp2full/` is
[`tracks/tp2/patches/`](tracks/tp2/patches/), file for file, with the two units, the two preflights
and the three environment templates beside them. Everything else stays where it is and
[`tracks/README.md`](tracks/README.md) says why each shared thing is shared — `docs/`, `bench/`,
`scripts/` (both launchers, because the harness and the probes they sit beside are identical at
either rank count), `patches/kernel/`, `patches/dflash2-port/`, `patches/indexer-overlay/`,
`results/`, `charts/`, `audit/`, and the two directory pages `envs/README.md` and
`systemd/README.md`, which are rules rather than artefacts.

**One file was deleted and it was a duplicate.** `patches/tp2/patch-fullscope-tp2.py` was
**byte-identical** to the copy inside the two-node tree. This repository's own rule is that two
copies of a file are a coin flip unless something checks ([docs/08](docs/08-fast-boot.md) §12), so
the duplicate is gone and its five references now point at
[`tracks/tp2/patches/patch-fullscope-tp2.py`](tracks/tp2/patches/patch-fullscope-tp2.py). Nothing
else moved content; every other change in the move is a path.

**The trap the move introduces, said once and loudly.** The directory name in this repository is not
the directory name on your nodes. On a node the three-node tree is `~/exl3-zeus/tp3full/`, and that
name is load-bearing three times over: the launcher mounts `$TP3_DIR/tp3-prelude.sh` at `/start.sh`,
`tp3-prelude.sh` inside the directory is a **hard link** rather than a copy, and the directory's file
list and the full text of the prelude are hashed into the fast-load sidecar's identity
([docs/08](docs/08-fast-boot.md) §4). Renaming the tree on a node invalidates every sidecar and costs
a dump boot each. Renaming it here costs nothing, because the manifest hashes the directory you serve
from, not the one you cloned from.

**Every `docs/NN` page now opens with an "Applies to" line.** Three are not "both":
[03](docs/03-tp3-padding-and-sidecars.md) is TP=3 only, because all five shapes that do not divide by
three do divide by two; [10](docs/10-results-and-roofline.md) is TP=3, because its tables are the
three-node arms, though the two rulers it measured rather than quoted apply anywhere; and
[15](docs/15-tp2-track.md) is TP=2 only. The qualifiers on the "both" pages are the point of the
exercise — 05 applies to both but expert parallelism is mandatory at three ranks and optional at two,
08 applies to both but the sidecar is per rank, 13 applies to both but only its §7 is three ranks.

**[14](docs/14-troubleshooting.md) goes further: all 86 entries carry a track tag**, and the
distribution is the finding. 45 are plain **both**, 26 more are **both, measured at TP=3 only** —
the mechanism is rank-independent but every reading came from a three-node arm — so **71 of 86 are
things a two-node owner will meet**. Thirteen are TP=3 only and they cluster exactly where they
should: the four hard asserts, the 2,112/192 constants, padding heads to 96 instead of 66, the
sidecar mount trap, expert parallelism not on every rank, the tuple-shard loader, the three upstream
commits a padded load needs, the `n_rows` dispatch bug and its micro-benchmark, the 22-head guard,
and three silent-correctness entries that are all about a pad holding something it should not. Two
are TP=2 only — KV sizing refusing at `max_model_len` 1,000,000, and long prompts never being
scheduled — and at three ranks the pool was never small enough for either.

**New: [docs/00-start-here.md](docs/00-start-here.md)**, which asks how many nodes you have and
answers for all four cases. At **one** node there is no serving recipe and the page says so in
arithmetic rather than by implication — 153.8 GiB of weights against 121.6 GiB of unified memory —
and then lists what a single node still gets: the image build, the GB10 top-k overlay, the
measurement protocol, the model-free benches, the failure index, and the two `HELP-WANTED` items that
need one GPU and no fabric.

**New: [HELP-WANTED.md](HELP-WANTED.md)**, the ranked list, with the expected wall-clock effort on
every item and — where it matters — a line on what a contributor with fewer nodes than the item needs
can and cannot check. Eight items: four nodes at TP=4; a two-node reboot test, which is an hour and
closes a `[not tested]` a two-node owner meets on their first power cut; the memory ladder at two
ranks; other checkpoints, including the class where a tensor we have to pad is itself quantized; the
mesh plugin's small-message latency floor; the ReplaySSM compact rollback of the KDA state slots; the
KDA GEMM engine-against-standalone gap; and the four largest items already in
[11](docs/11-open-issues.md).

**Two of those items carry numbers that were not on any page here, and both say so.** The plugin's
small-message floor reads **74.7 µs at 8 KB** on the production build and is flat at 72–85 µs from
8 B to 32 KiB — pure latency, no bytes in it — while [06](docs/06-nccl-mesh.md) §5, taken with a
different harness on the pre-multilink configuration, reads **38.6 µs** for the same operation. A
factor of two between two of our own benches is not something to publish past, so the item asks for
both harnesses in one session before any conclusion is drawn. And the KDA state slots:
`MambaSpec.max_memory_usage_bytes = page × (2 + num_speculative_blocks)`, so at k=7 every KDA layer
holds **nine** state slots per request, seven of them purely for speculation — **9.9 % of the block
counter at TP=3 and 12.9 % at TP=2**. A pool model that reproduces four measured arms exactly
projects **+8.0 %** and **+9.6 %** from taking that to two. We have not run it and the reason is
written down: the replay loop triples the sequential work, KDA is about 8 % of a C8 step, and the
ring holds `d` and `k` in fp16 with no baseline comparison we could find in the upstream harness.

**A defect in one of our own gates, found while writing the TP=4 arithmetic and fixed here.**
`preflight-tp3.py`'s vocabulary check had three conditions and only tested two. 154,880 already
divides 4 and is already 1,210 × 128, so at `tp=4` the first branch returned "no patch needed" — but
**per rank** it is 38,720 = **302.5 × 128**, half a Hadamard block, which is exactly the defect
`lcm(64, 3) = 192` produces at three ranks and which [03](docs/03-tp3-padding-and-sidecars.md) §1.1
calls silently wrong. `lcm(128, tp)` is right at `tp=3` (384), right **by luck** at `tp=2` (154,880/2
is already 605 × 128) and wrong at `tp=4`, where the unit that works is `128 × tp` = 512, giving
155,136 and 303 × 128 per rank. The per-rank condition is now tested and a `tp=4` run is told what to
pad and why. **Repository copies only** — `tracks/tp3/patches/preflight-tp3.py` and
`tracks/tp2/patches/preflight-tp3.py`, which stay byte-identical to each other; the live trees on our
nodes were not touched. It is sidecar-safe by construction: `harem_fastload_id.py` hashes
`patch-*.py`, the prelude and the EP overlay, and `preflight-tp3.py` is none of those. Behaviour at
`tp=2` and `tp=3` is unchanged, verified by arithmetic; **`tp=4` is `[not tested]`** — we own three
nodes.

**Contributing, and four templates.** [CONTRIBUTING.md](CONTRIBUTING.md) gains a "how to contribute a
measurement" section naming three routes and what each becomes, and [`.github/`](.github/) gains
three issue templates and a pull request template that are [09](docs/09-measurement-protocol.md)
turned into checklists: rounds and which were discarded, the per-metric noise band a difference has
to clear, the prompt set named and labelled, the KV pool read from a load boot on a settled host, the
gates cold **and** warm, an evidence tier, the full settings block, and what the gain cost. Two
things are now said in the open rather than implied — a step that did **not** reproduce is as welcome
as a measurement, and a silent failure is worth more than a loud one; and a pull request that
withdraws one of our numbers is worth more than one that adds a number.

**Older entries on this page were rewritten in one respect and one only:** paths that moved now name
where the file actually is. The dates, the numbers and the reasoning are untouched.

**The repository was renamed** from `GLM-5.3-Flash-EXL3-TP3-3x-DGX-Spark` to
`GLM-5.3-Flash-EXL3-DGX-Spark`, because `TP3-3x` in the name was telling two-node owners this was not
for them. GitHub redirects the old URL; a clone with the old remote keeps working, and
`git remote set-url` is the tidy fix.

**No measurement changed today. Nothing was re-run and nothing is claimed that was not claimed
yesterday**, apart from the two `HELP-WANTED` figures above, which are labelled, and the preflight
gate, which is arithmetic.


## 2026-09-06 — a TP=2 production candidate, and the two-node page's second retraction

**The two-node track is now a complete recipe with a named production candidate, not a set of
bring-up arms.** A patch tree (`tracks/tp2/patches/`, thirteen files), an environment template
(`tracks/tp2/env.tp2-full.example`), a launcher (`scripts/start-tp2full.sh`), a per-rank fast-load sidecar
and an autostart unit (`tracks/tp2/harem-exl3-tp2.service`), all measured end to end on two nodes with
the protocol in [docs/09](docs/09-measurement-protocol.md) `[measured-here]`. Rewritten:
[docs/15](docs/15-tp2-track.md). Raw record:
[`results/speed/tp2-production-candidate.md`](results/speed/tp2-production-candidate.md).

**Two candidates were measured, identical except the checkpoint. The full-scope one wins every axis
and is the recommendation.** Both at `gpu-memory-utilization` 0.85, `max_model_len` 1,000,000, image
`exl3-zeus:754421f`, EP off, DFlash2 k=7, fp8 KV **and** fp8 draft cache, `HAREM_SW_BLOCK_SIZE=256`,
mesh plugin with both cables, warm tuner cache, fast-load sidecar, median of sweep rounds 2–4 of
four:

| | A — experts-only | **B — full-scope** | B vs A |
|---|---|---|---|
| KV pool at 1M | 1,500,000 | **2,128,571** | **+41.9 %** |
| C1 aggregate · per stream | 48.76 · 54.72 | **58.50 · 62.55** | +20.0 % · +14.3 % |
| C8 aggregate | 137.41 | **155.75** | +13.3 % |
| TTFT, C1 / C8 | 0.468 / 1.249 s | **0.407 / 1.077 s** | −13.0 % / −13.8 % |
| Consumed memory per node | 89.3 GiB | **84.8 GiB** | −4.5 GiB |
| Boot, fast-load | 272 s | 272 s | equal |
| Gates cold+warm · tool-call · needle-lite | 10/10 · 12/12 · 8/8 · 6/6 | 10/10 · 12/12 · 8/8 · 6/6 | equal |
| MMLU sample, 1,995 q | 86.37 ±0.74 | **86.02 ±0.75** | inside one error bar |

**`[retracted]`: "the full-scope checkpoint at two ranks is a rig and not a serving configuration".**
On 5 September it failed its KV budget gate at `max_model_len` 1,000,000 with
`6.6 GiB KV needed, available 0.73 GiB`, and at 65,536 produced a 31,343-token pool in which a
~2,800-token prompt was never scheduled ([docs/15](docs/15-tp2-track.md) §3.2). Neither reading
survives two changes made since. The draft page fix cuts blocks-per-request 640 → 280, so the 6.6 GiB
requirement becomes ~2.9 GiB; and the launcher's **settle gate** — a `MemAvailable ≥ 112 GiB` wait
before vLLM snapshots memory — turns 0.73 GiB of available KV memory into **16.07 GiB**. The settle
gate is the larger half, and it is a measurement bug fix rather than a tuning knob
([docs/07](docs/07-kv-and-draft-page.md) §1.1 already said a rank that snapshots a dirty host awards
itself memory it does not have; what is new is that the error was the difference between "cannot
serve" and "recommended").

**`[retracted]`: "the full-scope checkpoint is ~10 GiB heavier per node at two ranks".** Measured on
one boot each with the same settle gate, image and launcher: consumed memory (weights + non-torch) is
**89.3 / 89.2 GiB** experts-only against **84.8 / 84.5 GiB** full-scope. It is **4.5–4.7 GiB
lighter**, which agrees in sign and size with the three-node reading. The old figure came from a boot
with no settle gate.

**Four more rows left the two-node "not tested" list.** The **fast-load sidecar** at two ranks: boot
**997 s → 272 s**, sidecar 75–83 GB per rank (half the checkpoint, because EP is off, against a third
at three ranks with EP). The **fp8 draft cache**: **+15.1 % of pool**, isolated by arithmetic —
`SlidingWindowSpec` bytes/block halves, blocks-per-request is unchanged at 280, `num_blocks` goes
365 → 420 — and CUDA graphs still capture at two ranks, unlike at three. The **dual-cable mesh
patch** on a *single* peer pair: both devices per node moved ~90 GB across a sweep, split 50.5 / 49.5
— `patches/kernel/0005` is not a three-node effect. The **tuner-cache protocol**: round 1 is inside
±3 % of the median of rounds 2–4 at two ranks as well, so three rounds is enough here too.

**The autostart unit, installed and tested but left disabled.**
[`tracks/tp2/harem-exl3-tp2.service`](tracks/tp2/harem-exl3-tp2.service) with
[`motor-onkosul-exl3-tp2.sh`](tracks/tp2/motor-onkosul-exl3-tp2.sh) (one fabric peer per node instead of
two; the ConnectX-7 check stays 4/4 because it counts ports, not peers; `Conflicts=` **both** sibling
units). `systemctl start` on both nodes → `/health` 200 at **+261 s** → gates 10/10 · 12/12 →
`systemctl stop` clean. Its first attempt **failed correctly**: the preflight refused in one second
because the sidecar the environment file named was not there, before docker was touched. **No
two-node reboot test** `[not tested]`, and the three-node unit remains the enabled autostart.

**What is deliberately still open at two ranks:** the `gpu-memory-utilization` ladder (0.85 is where
every arm ran; it is where our first two-node arm found 3.5 GB of swap during weight load, and at
three ranks 0.85 was *rejected* — the ladder must be re-derived, not copied), a two-node reboot test,
expert parallelism on, and a second boot of each candidate. [docs/15](docs/15-tp2-track.md) §6.

---

## 2026-09-06 — the draft KV page at two ranks: the pool more than doubles, and the long-prompt path only exists with it

**A two-node owner read [docs/15](docs/15-tp2-track.md) and said the KV cache bug was the thing
standing between them and a usable configuration. They were right, and this page's own projection
was low.** `HAREM_SW_BLOCK_SIZE=256` — the draft KV page, 16 to 256 tokens — was the largest
untested item on the TP=2 page, carrying an `[estimate]` of "perhaps 0.8-1.2 M from arm C's
665,625". We ran the control and the fix back to back at two ranks on 6 September 2026, one boot
each, the only difference one token in `EXTRA_ENV`. New section
[docs/15](docs/15-tp2-track.md) §3.5 (now §4), raw record
[`results/speed/tp2-draft-page.md`](results/speed/tp2-draft-page.md) `[measured-here]`.

**KV pool 601,562 to 1,303,571, +116.7 %** (+109.6 % after normalising for the 3.4 % more KV memory
the fix arm's binding rank happened to get). C8 aggregate 127.54 to 135.59 (+6.3 %), C2/C4/C6
+0.7/+2.3/+4.0 %, TTFT −23 % at C1 and −27 % at C8, C1 aggregate equal. Quality gates 10/10 · 12/12
· 0 empty, cold and warm, on both arms. Settings otherwise identical: two nodes, TP=2, EP off, image
`exl3-zeus:62f53e6`, routed-experts-only checkpoint at `b20c49ba`, KV fp8, DFlash2 k=7,
`gpu-memory-utilization 0.85`, `--block-size 256`, `--max-num-batched-tokens 2048`,
`--max-model-len 1000000`, medians of three rounds, both booted through the host-side settle gate.

**The headline is not the pool, it is the cliff.** Without the fix the control does not serve a long
prompt slowly — it never schedules it: a 6,253-token prompt sits at `Running: 0, Waiting: 1, GPU KV
cache usage: 0.0 %` indefinitely, and both prefill benchmarks time out. The arithmetic is one line:
block ids are global to a single pool, and one request wants 640 of the pool's 385 blocks. With the
fix it wants 280 of 365, and 8,268 tokens serve in 6.3 s, with fresh-prompt prefill at 1,478 tok/s.

**Why the projection was low: the defect is worse at two ranks than at three.** The engine's own
per-group decomposition shows the drafter taking **60.2 %** of the blocks-per-request divisor at
TP=2 against 53 % at TP=3 — not because the drafter changed, its 385 blocks are identical, but
because the platform raises the attention block to 4,608 tokens at two ranks where it is 3,328 at
three, cutting the target's share from 301 blocks to 218. A smaller denominator makes the same
defect a larger share. [docs/07](docs/07-kv-and-draft-page.md) §4's "the page layout was the
*second-order* term" is corrected in place: second-order in memory, first-order in the counter, and
it grows as the node count falls.

**What it cost.** Per-block memory rises about 9.1 %, the same price as at three ranks. The draft
group's prefix-cache matching unit coarsens from 16 to 256 tokens, which costs nothing measurable
here because the hit rate is already 0 % behind a 4,608-token block. Acceptance falls 1.9 points,
inside this stack's boot-to-boot band. And C1 per-stream decode reads 7.9 % lower, which one boot
does not establish: the fix arm's own three rounds span 9.3 %, the C1 aggregate is equal and TTFT is
23 % better. No C1 gain, and no proven C1 loss.

**One retraction, and it is about our instrument rather than the stack.** The control arm
re-measured today gives **601,562** where 5 September published **665,625** for the same untouched
env file, −9.6 %. The 5 September TP=2 harness had no settle gate; this one waits for `MemAvailable`
to come back over 112 GiB on both nodes. That is exactly the bias
[docs/07](docs/07-kv-and-draft-page.md) §1.1 describes, in the direction it predicts. It matters
more than a percentage: 5 September's 426 blocks were just above the long-prompt cliff and served a
7,382-token prompt at 1,135 tok/s; today's cleaner 385 are just below it and cannot.
**Without the page fix, whether two ranks can serve an 8K prompt depends on the host's state at
boot.** The arm C prefill and pool figures in §3.2 and §4 are flagged as measured on an unpinned
baseline, and the "arm C had 4.4 GiB of available KV" figure that §3.4's retracted arithmetic rested
on is contradicted by today's 9.97 GiB; it is now marked an inference rather than a log line.

The patch is [`patches/tp3/patch-swblock-tp3.py`](patches/tp3/patch-swblock-tp3.py) unchanged — it
is `tp`-agnostic and gated on its own environment variable, so the control ran the same image with
the knob unset. `HAREM_SW_BLOCK_SIZE=256` moves from "keep, never run at TP=2" to **mandatory in
practice** in the §2.5 flag table. Still `[not tested]` at two ranks: a second boot, the fp8 draft
cache, fast load, the memory ladder, expert parallelism on, and the page fix on top of the
full-scope checkpoint.

---

## 2026-09-05 (after the release pass) — two documents: the TP=2 track, and what other published recipes report

**[docs/15 — Running this recipe at TP=2](docs/15-tp2-track.md).** This is a three-node recipe and
every default in it is a TP=3 default, but it runs on two nodes and at two ranks it is a *shorter*
recipe rather than a cut-down one: all five shapes that do not divide by three divide by two **and**
leave every rank a whole number of 128-column Hadamard blocks, so the whole of
[docs/03](docs/03-tp3-padding-and-sidecars.md) and the padded-load path in
[docs/13](docs/13-full-scope-checkpoint.md) §7 fall away. Expert parallelism becomes **optional** —
2,048/2 = 8 × 128 — and `preflight-tp3.py` already accepts `--ep 0` at two ranks while refusing it at
three. The page lists the exact nine changes (three env lines, three launcher edits, the
`tracks/tp2/patches/` tree, the image, the autostart unit), the four two-node arms we ran with their dates
and settings, and the eleven-row table of production features we have **never** run at two ranks:
`HAREM_SW_BLOCK_SIZE=256`, the fp8 draft cache, the fast-load sidecar, the memory ladder, expert
parallelism on, a two-node reboot test and a second boot among them `[not tested]`.

**The trade-off is not the one people expect, and the page says so in its second paragraph.** Like for
like on the same day, same image and same harness, two nodes reach **85–91 %** of three nodes' C1 per
stream, **75 %** of C8, **75 %** of prefill and **14 %** of the KV pool `[measured-here]`. TP=2 does
**not** win single-stream latency here: a decode step is weight-bandwidth bound, a third rank cuts each
rank's weight traffic by a third, and that beats the collective it costs. What two ranks buy is a node
and a shorter recipe. Boxed "At TP=2" notes now sit in
[03](docs/03-tp3-padding-and-sidecars.md), [05](docs/05-expert-parallel-and-cuda-exl3-fixes.md),
[06](docs/06-nccl-mesh.md), [07](docs/07-kv-and-draft-page.md), [08](docs/08-fast-boot.md),
[13](docs/13-full-scope-checkpoint.md) and [systemd](systemd/README.md), each pointing at the step that
differs. This partly closes [CONTRIBUTING](CONTRIBUTING.md) item 10; §3.3 says exactly which part is
still open.

**[docs/16 — Comparison with other published recipes](docs/16-comparison-with-published-recipes.md).**
Twelve other public GLM-5.3-Flash EXL3 DGX Spark recipes — nine at two nodes, two at three, one at
four — quoted **exactly as they publish them** with the conditions they state and tiered `[reported]`,
beside our figures at the matching node count. Nothing in their columns is re-derived, rescaled or
corrected by us, and where conditions differ the difference is a column rather than a footnote.
**There is no three-node recipe from `MiaAI-Lab`** — we looked on GitHub and on Hugging Face; its only
larger arrangement is an explicitly untested four-node script, and the two three-node EXL3 recipes are
`FlyCockpit/GLM-5.3-Flash-EXL3-3x-DGX-Sparks` (commit `9093765c`, 2026-08-29, one commit and unchanged
since) and `jakejharris/jspark3` (`v1.0.0`, 2026-09-03).

**The most valuable thing on the page is not a ranking — it is that two other people quantized this
model's dense path independently, and both measured what we measured.** `Alexbob0/glm53-flash-dense-exl3-tp2`
overlays the same 4.05 bpw pack's dense tensors onto a routed-experts-only checkpoint at **two** nodes
and reports **+22 % structured / +26 % prose** against its own control, with a torch-profiler step
anatomy whose top item and residual-BF16 story match ours `[reported]`; ours is +24.3 % per stream at
TP=2 through a different loader and a different kernel project. `jakejharris/jspark3` does it at
**three** nodes with an INT8 W8A16 Marlin overlay on 169 dense modules and reports +6.6–8.4 % against
its own control, along with a disclosed −21.2 % at C3 and −3.4 % on long prefill. Three stacks, three
routes, one direction. Their prose figure, 29.049 tok/s, also lands on our 29.1 — the only nearly
like-for-like row on the page.

**Three more things it establishes, and one it refuses to do.** The FlyCockpit padding constants
(`lcm(64,3) = 192`, shared expert 2,112) and ours (384, 2,304) are **both correct**, for different
checkpoints — the unit is 64 when those tensors are BF16 and 128 when they are EXL3, and our own
configurations 1–8 used theirs. The two-node KV pool gap is **ours**: our 665,625 is the smallest
two-node figure on the page because we never ran `HAREM_SW_BLOCK_SIZE=256` at two ranks, while the
`MiaAI-Lab` lineage solved the same drafter-page problem with a padded slot-share onto the MLA
tensors. And `punkjazz-labs/glm-5.3-flash-exl3-4x-dgx-spark` publishes a **soak hang** — three of
three, a 96K chunked prefill sharing steps with speculative streams, and a 282K cold prefill that
still stalled with the drafter off — which we have never looked for on three nodes `[not tested]` and
which is now the outside item we would most like settled about our own stack. What the page refuses to
do is convert a synthetic number into a realistic one: most of these recipes lead with a counting
prompt, the gap is roughly **1.7×** on this model family, and those rows stay labelled and apart.

**What we took**: from `FlyCockpit/GLM-5.3-Flash-EXL3-3x-DGX-Sparks`, the arithmetic already credited
in [CREDITS.md](CREDITS.md) and no files; from `MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks`,
**nothing** — no code, no configuration, no number, no technique; from everything else, nothing, and
all of it was found after production 9 shipped. They are named as the sources of their own figures and
for no other reason. Two search hits are excluded and deliberately not named: their documents are
copies of other repositories' files and their READMEs advertise a Windows executable for a machine
that cannot hold this model, so there is nothing of theirs to compare.

**Cost:** documentation only. No cluster time, no configuration change, no number in this repository
moved.

---

## 2026-09-05 (after the release pass) — the quantization gate bench, re-measured on the target GPU: the closure stands, the reason we gave for it does not

**`[retracted]`: "on the KDA shapes EXL3 is 1.58–1.76× slower than BF16 at M=8".** That sentence
closed the item above, and it went upstream. It is a **warm** number. The workstation bench did run
against a ~300 MB weight bank — the ruler had caught a 210 %-of-peak reading and that is why the bank
exists — but the bank was sized for the large shapes, and the arm the sentence is about is **0.72 MB**
and sat in that card's 101 MB L2 from the first replay to the last.

**Re-measured cold on GB10, same shapes, same `cuda-exl3` `754421f` build, both arms rotated over a
bank of at least 4× L2** `[measured-here]`. `f_b_proj` at M=8 reads **1.023**, not 1.596 — and GB10's
own *warm* arm reproduces the withdrawn figure at **1.605**, which is how we know what the workstation
was measuring. **Seven of nine shapes reverse sign**; `kv_b_proj` improves from 0.391 to **0.291**.
**The production trace refereed it**: per-call costs from the production-9 C1 profile are `f_b_proj`
5.41 µs against 4.99 cold and 2.21 warm, `in_proj_bfg_a` 14.20 against 15.51 and 6.52 — **cold
reproduces the engine to ±20 %, warm is out by 2.2–2.4×**, so which regime is honest is a measurement
and not a preference.

**What it changes, and what it does not.** The KDA gating arms move from "quantizing them costs
0.584 ms/step" to **+0.050 ms/step — neutral — 0.07 % of C1**. They stay BF16, now for want of a gain
rather than for a loss, and the lever on them is not the bit width at all: cold, `f_b_proj` reads
**45.7 GB/s, 19 % of peak, in either format**, because it sits on a ~5 µs floor made of **two
dependent kernel launches** where bf16 has one. Fusing `had_in` into the GEMM at narrow inputs is what
those shapes want, and it has been asked for upstream. The three measured families together go from
+0.547 to **+1.368 ms/step** against a 1.5 ms gate — a narrow miss rather than an order of magnitude —
and **96 % of that is still `kv_b_proj` alone**, which still needs the per-head batched EXL3 kernel
`754421f` does not have. **Prefill (M = 1,792) was not re-measured and the gate is an *and*, so the
item is re-scoped, not passed.** The MLA fp32 → bf16 lever is untouched and is still the cheapest
thing on the board.

**A second claim of the same report goes with the first** `[retracted]`: it warned its ratios were
*optimistic* for GB10. **Every family came out better on GB10** — machine balance about 416 flop/byte
against the workstation's 141 is exactly the condition a trellis is built for, and on the control
shapes EXL3 delivers 0.272 against the pure bit-ratio limit of 0.250.

**The lesson, and it is the third time this stack has learned a version of it.** The artefact's
**sign depends on which arm fits the cache**: on a 101–128 MiB L2 both fit and EXL3 reads slow, on
GB10's 24 MiB only the trellis fits and EXL3 reads too fast. It therefore does **not** cancel in a
ratio, and only cold is honest on either card. **A bank sized against the wrong card, or the wrong
shape, is the same mistake as no bank at all.** The rule is now in
[docs/09](docs/09-measurement-protocol.md) §4.2: size the bank per shape and against the card you will
serve on, rotate both arms over it, and let a production trace referee the regime where one exists.

**Cost:** 90 s of one node's GPU, 1.47 GiB peak, in a throwaway container beside an idle engine. No
restart, no configuration change, nothing on the cluster touched. Posted to the upstream thread the
original went to. [docs/11](docs/11-open-issues.md) §1.11 and §2.25,
[docs/13](docs/13-full-scope-checkpoint.md) §4.4,
[`results/kernels/kda-gate-bench-gb10.md`](results/kernels/kda-gate-bench-gb10.md) with its raw under
[`results/kernels/gb10-coldbench/`](results/kernels/gb10-coldbench/), the script as
`bench/kda_gate_bench_gb10.py`, and the retraction indexed in [`audit/`](audit/README.md) §6. The
workstation file is kept exactly as published, with a banner.

---

## 2026-09-05 (late) — release: autostart, three open items closed by measurement, and a spread we had mis-attributed

**Status moves from release candidate to release**, on production configuration 10.

**Autostart, and it has been through a reboot** `[measured-here]`. The `systemd/` directory stops
being a template with three things wrong with it: `tracks/tp3/harem-exl3.service` and
`tracks/tp3/motor-onkosul-exl3.sh` are the real unit and the real preflight, installed and `enabled` on
all three nodes. The preflight runs seven checks — docker answering, `ibv_devinfo` 4/4, a ping to each
fabric neighbour, `drop_caches`, then the env file, the image and this rank's fast-load sidecar
manifest, the last three each having cost us a silent boot. `ExecStop` names `exl3-tp3` rather than
the NVFP4 container the template named, `TimeoutStartSec` is **1200** so a 620 s dump boot cannot time
out mid-load, and the unit carries `Conflicts=harem-motor.service` while the sibling unit is now
`disabled` on all three nodes.

**The reboot test**, whole cluster, one trial: ssh and 4/4 at **+29 / +30 / +31 s**; units log
`Finished` at **+98 / +98 / +103 s** — that is the preflight, the fabric wait and the settle gate
together; `/health` 200 at 22:28:21, **212 s after the last unit finished**. KV pool **5,652,892**
against 5,619,834 from a settled `docker run` on the same configuration, **+0.6 %** — the cleanest
baseline a pool number can have, and the strongest check on the settle gate we have run. Gates 10/10
and 12/12 afterwards. **The elapsed figure is published twice because the log contradicts itself:**
the harness printed `+242 s`, the wall-clock stamps in the same file give **315 s**, 242 s before the
health check matches no recorded event, and the larger is the figure to plan with. What the unit does
**not** solve is stated with it: `--restart no` is unchanged and deliberate, systemd still does not
enforce the worker-2 → worker-1 → head order (the rendezvous retrying is margin, not a guarantee), and
the watchdog is still not written. **Reboot all three together or none** — the preflight checks only
its own node's fabric, so a single-node reboot passes it and starts a rank into a cluster whose peers
are gone. [`systemd/`](systemd/README.md), [docs/00](docs/00-hardware-and-os.md) §13a,
[docs/09](docs/09-measurement-protocol.md) §11.4, [docs/11](docs/11-open-issues.md) §2.20,
[docs/14](docs/14-troubleshooting.md) §10.1–§10.1a, README quick-start step 11.

**`NCCL_ALGO=Ring,Tree` — closed, equal, `Ring` stays** `[measured-here]`. The five-round engine arm
the model-free sweep could not substitute for: C1 70.6 / C2 99.9 / C4 143.4 / C6 175.2 / C8 195.6
against `Ring`'s 70.5 / 99.0 / 144.6 / 175.4 / 194.0 — **every level inside ±1 %**, TTFT identical,
acceptance inside its own spread, gates full. The proxy's −3.6 % did not survive a real step, which is
what should happen when a proxy times 90 collectives in isolation and the step overlaps none of them.
The one structural difference is **+1.5 % of KV pool** (5,702,479 against 5,619,834) from NCCL's
per-algorithm buffer sizing — real, and not worth pinning production to an algorithm list for.
[docs/06](docs/06-nccl-mesh.md) §12.3, [`results/mesh/algo-sweep.md`](results/mesh/algo-sweep.md).

**Quantizing the remaining BF16 KDA/MLA modules ourselves — closed, and it would be slower**
`[measured-here]`. The 113 linears per rank that production 9 leaves in BF16 were the obvious next
arm: the FP16 weights are already in the checkpoint, so it is a surgical pass rather than a
requantization. A model-free bench on a workstation GPU, `cuda-exl3` at the production commit, at the
TP=3 per-rank shapes, closed it before any of that was spent. **The checkpoint author left them BF16
by design** — `qmap = None` in `exllamav3` with the reason in the comment, and `kv_b_proj` is not a
`Linear` at all — and he applied that judgement discriminatingly, quantizing KDA `qkv_proj` and
`o_proj` to 6 bit from the same exclusion list. **On the KDA shapes EXL3 is 1.6–1.8× slower than BF16
at M=8**: those arms are 0.72 MB and were never bandwidth-bound, so four-bit weights buy bytes that
were not the cost while paying the Hadamard and trellis-decode fixed cost. The arithmetic cannot be
rescued even at zero cost — the whole family is **0.851 ms of a 72.5 ms step**, +1.2 % of C1 if it
were free. The one family that would genuinely gain, `kv_b_proj` at 0.391× and +1.13 ms/step, **needs
a batched EXL3 kernel that does not exist** (verified by source scan, along with the absence of any
M-threshold reconstruct path). **One lever survived and it is not a quantization lever:** the MLA
strided-batched family runs in fp32, and bf16 measures 0.684× — **+0.24 ms/step, about +0.3 % of C1**,
no checkpoint change and no new kernel, filed as future and minor and gated on `needle` at 1M rather
than on speed. Three instrument lessons came with it, including a ruler check that read **210 % of the
machine's own peak** before a weight bank was added. [docs/13](docs/13-full-scope-checkpoint.md) §4.4,
[docs/11](docs/11-open-issues.md) §2.25,
[`results/kernels/kda-gate-bench.md`](results/kernels/kda-gate-bench.md), and both scripts ship
(`bench/ruler_check.py`, `bench/kda_gate_bench.py`).
**Corrected the same night, and the entry above is the correction:** the 1.6–1.8× is a warm number
that the weight bank did not reach on a 0.72 MB arm, the item still closes, and it closes because
quantizing those arms is **neutral** rather than because it is a loss `[retracted]`.

**A spread we had mis-attributed, corrected against the raw sweeps** `[retracted]`. Our field log
recorded a "C1 run-to-run spread of 59.9 … 70.6 across boots" for production 9 and 10. It is wrong:
**59.9 is the TP=2 full-scope arm's C1 aggregate** and the other low reading beside it was a
production **8** restore. No production 9 or 10 boot has read below **67.7** even at the level of a
single round. Every round of all four boots is now printed in
[docs/10](docs/10-results-and-roofline.md) §1.1, and the real picture is that the two axes behave
differently: within a boot, C1 spans 1.0–5.7 % peak to peak and C4 5.1–9.8 %; across boot **medians**,
C1 spans **1.1 %**, C8 **2.5 %** and C4 **7.4 %**. **Which makes one published number a trap:**
production 10's C4 of 144.6 against production 9's 134.6 is **+7.4 %**, exactly C4's boot-median span
— a memory fraction does not buy 7 % of four-way throughput, and the arms are equal at C4. Every
headline figure is the median of one boot's rounds; the rule and the guard are now stated where the
numbers are.

**Stale claims the previous pass's retractions left behind, corrected.** `docs/10` §1 and `docs/13`
still billed production 9 for **−2.4 points of draft acceptance and −3.0 % of accepted tokens per
step** after the front page had withdrawn both; the four costs that remain there are now the four that
were never about the drafter. `results/gates/quality-gates.md` still said MMLU had not been run at
TP=3; it has, at 86.47 ±0.74 on the production checkpoint, and the page now separates that from the
86.4 ±0.7 TP=2 figure that configurations 7 and 8 carry forward. `docs/11` §3 still listed a
production-9 re-profile as never run.

**Also in this pass.** `results/speed/concurrency-sweeps.csv` gains the production 9, production 10
and `Ring,Tree` arms, so the provenance table's rows point at data that is actually in the file;
`results/configs/*.csv` gain the measured 0.83 rung in place of the "designed, not run" row, plus the
`Ring,Tree` and power-on boots; `audit/README.md` §5 gains five provenance rows and two more facts a
reviewer should not have to dig for, and its memory section now states the rule production 10 sits
against — **watch swap, not `MemFree`**, because `MemFree` is 0.9–1.2 GiB at 0.83 and swap is what
0.85 was actually rejected on. 0.85 will not be attempted; the rung after 0.83 is a soak.

---

## 2026-09-05 (night) — release pass: production configuration 10, two retractions, and the environment record

**Production configuration 10 = production 9 with one line changed**, `GPU_MEMORY_UTILIZATION`
0.80 → 0.83 `[measured-here]`. KV pool **5,168,044 → 5,619,834 (+8.7 %)**; C1/C4/C8 69.8/134.6/192.4
→ 70.5/144.6/194.0, all inside their bands; prefill-fresh 1,687/1,769/1,779 against 1,745–1,774;
quality gates full, cold and warm. **Swap stayed flat at ~0.1 GB per node through the rounds**, which
is the number 0.85 failed on. `MemAvailable` 8–10 GB, `MemFree` 0.9–1.2 GiB of reclaimable page
cache. `[docs/11](docs/11-open-issues.md)` §2.4 predicted +11 % from production 7's geometry and the
measured figure is +8.7 % — the method held, the base did not. **0.85 will not be attempted.**

**Two retractions, both of ours, both from re-reading raw data we already had:**

- **The "−2.4 points of draft acceptance" cost of production 9 does not exist** `[retracted]`. It is
  an artefact of `scripts/bench-sweep.py` cycling `prompts[i % 12]`, so C1 and C2 see only 8 of the
  12 prompts while C4–C8 see all twelve, and the two groups differ by about 8 points. Pooled by draft
  token over five levels and three boots: **+0.18 points**, and the sign reverses at C6. The cold
  probe is identical on both arms. `accept_len = 1 + k × acceptance` on all 90 rows, so the
  "acceptance" and "tokens per step" costs were one number written twice. Net throughput effect
  **+0.24 %**. Front page, [docs/11](docs/11-open-issues.md) §2.26 and
  [audit/](audit/README.md) §6 corrected.
- **Production 9 *was* re-profiled**, and this repository said it had not been `[retracted]`. The
  torch-profiler run against the live production-9 server (three ranks, no restart) landed 40 minutes
  after the README and `docs/10` were written. Dense stage **45.3 % → 25.9 %** of a C1 step
  (42.90 → 21.90 ms); new C1 ranking MoE trellis GEMM 32.5 %, NCCL 26.1 %‡, dense EXL3 GEMM 15.0 %,
  remaining BF16 10.3 %; prefill chunk MoE 28.1 %, NCCL 14.0 %, dense EXL3 13.4 %; C8 MoE 56.3 %.
  ‡ NCCL and CPU-gap at C1 are **CUPTI-inflated** — 2,738 kernel launches per step, so with the
  profiler off the two together are ≤17.19 ms rather than 29.1, and the step wall is 72.52 ms rather
  than 84.44. **And the arithmetic missed something:** the full-scope dense stage is **+17.3 ms,
  +10.4 % in *prefill*** at M=1,792. The wall stayed flat only because three other classes gave it
  back. The plugin author reproduced the shape on his own card and withdrew a "cuBLAS parity" claim
  from his README.

**New in the repository, for the release:**

- **[docs/00](docs/00-hardware-and-os.md) rewritten as a complete environment record** — firmware
  (SBIOS/EC/SoC/USB-C PD, and why below `0104` costs a quarter of the fabric silently), the
  `dgx-spark-mlnx-hotplug` root cause and the file to remove, the physical cabling (**three** cables
  carrying six links, not six cables — corrected), the PCIe Gen5 x4 ceiling read on 12 of 12
  endpoints, every version with the command that prints it, **the six OS-level changes we made and
  the three we deliberately did not**, the `swappiness=0` lock-up, the memory rules and the settle
  gate, and the mesh plugin's build and binary identities.
- **[docs/14](docs/14-troubleshooting.md)** — all 83 failures by symptom with the exact log line, a
  triage order, and a ranked index of the twenty that produced **no error message at all**.
- **[audit/](audit/README.md)** — a post-install self-check with our numbers beside each step, a
  provenance table for every headline figure, and the retraction index.
- **[charts/](charts/)** — four figures, generated from committed CSVs by a standard-library script.
- `results/configs/production-configurations.csv`, `results/configs/kv-pool-progression.csv`,
  `results/profile/step-breakdown.csv`.

**Corrections made in place:** `docs/10` §4.2 carried "2048-token chunk, 1.109 s" — a chunk size taken
from the flag rather than the trace and a wall from an earlier configuration. The measured values are
**1,792 tokens** (7 × 256) at **962.55 ms** (production 7) and **961.73 ms** (production 9), which
moves prefill's effective bandwidth from 53 to about **61 GB/s**. `LICENSES.md` and `CREDITS.md` named
the fallback checkpoint as the production one and pinned `cuda-exl3` at `f4987cf`; both now say
`turboderp/GLM-5.3-Flash-exl3` (MIT) and `754421f`. Category speeds are production 9's
(code 61.7 · math 79.6 · JSON 72.8 · prose 29.1) rather than five configurations old.

---

## 2026-09-05 (late evening) — production configuration 9: the checkpoint changed, and it is the largest move this stack has made

Every configuration from 1 to 8 served routed-experts-only weights beside a BF16 attention stack.
Production 9 does not. The item this repository had called its largest for a week — dense BF16 GEMM
at 45.3 % of a single-stream decode step — is closed, taken, and in production four hours after the
TP=2 arm that proved it was worth taking.

- **Production configuration 9** `[measured-here]`. `turboderp/GLM-5.3-Flash-exl3` at 4.05 bpw
  (`2a30229e`, MIT), full scope, at TP=3 with expert parallelism, on image `exl3-zeus:754421f` and the
  new `tracks/tp3/patches/` tree. Control is production 8 — the **pool of two runs of the same script on
  the same day**, because that arm's documented run-to-run spread is about 7 %. Medians of three
  rounds, everything else identical:

  | | **production 9** | production 8 | delta |
  |---|---|---|---|
  | C1 total / per stream | **69.90 / 75.91** tok/s | 56.88 / 62.39 | **+22.9 % / +21.7 %** |
  | C2 / C4 / C6 total | 99.17 / 140.72 / 172.40 | 83.31 / 120.22 / 144.03 | +19.0 / +17.1 / +19.7 % |
  | C8 total | **197.20** | 175.37 | **+12.5 %** |
  | TTFT, C1 / C8 | 0.280 / 0.826 s | 0.344 / 0.906 | −18.6 / −8.8 % |
  | prefill, fresh / 7K repeat | 1,738 / 1,575 | 1,776 / 1,537 | equal both ways |
  | KV pool at 0.80, 1M context | **5,165,289** | 4,696,969 | **+10.0 %** |
  | consumed memory per node | 58.3–59.1 GiB | 62.1–62.4 GiB | **−3.4 GiB** |
  | draft acceptance · tokens per step | 61.94 % · 5.34 | 64.36 % · 5.50 | **−2.4 pt · −3.0 %** |
  | boot, fast-load | 251 s (weights 57.9 s) | 264 s (73.2 s) | −5 % |
  | gates cold and warm · MMLU sample | 10/10 · 12/12 · **86.47 ±0.74** | 10/10 · 12/12 · 86.4 ±0.7 | equal |

  **The step arithmetic is the entry:** 88.2 → **70.3 ms**, 17.8 ms saved, while acceptance and
  accepted length moved the *wrong* way by about 3 %. The whole gain is the dense stage going from
  BF16 to 4–6 bit; none of it is drafter behaviour. The same lever measured 20.7 ms at TP=2 on two
  nodes, so it is now **confirmed on two independent topologies**.

- **What it cost, and the line is not empty** `[measured-here]`. Draft acceptance −2.4 points
  (gate ≥60 %, passed) and accepted tokens per step −3.0 %; a **second patch tree** whose
  `patch-vllm-tp3.py` and `preflight-tp3.py` diverge from production's on purpose and have to be kept
  in step by hand; a **second 53 GB fast-load sidecar** per node on top of a second 154 GiB
  checkpoint. Quality was looked for and not found: 86.47 ±0.74 against 86.4 ±0.7 is 0.07 points, a
  tenth of either error bar, with both gates full cold and warm on the same engine instance in one
  session. [docs/13](docs/13-full-scope-checkpoint.md) §7.4, [docs/11](docs/11-open-issues.md) §2.24.

- **The TP=3 port: two constants, one patch of ours, and a padded-load path from upstream.** The two
  launcher constants moved from `lcm(64, tp)` to `lcm(128, tp)` — vocab `padding_size` 192 → **384**
  (155,136 = 3 × 404 × 128) and shared expert 2112 → **2304** (768 = 6 × 128 per rank) — because a
  full-scope checkpoint quantizes both and every EXL3 pad has to be whole 128-column Hadamard blocks.
  **2,176 is 128-aligned and not divisible by three; the width must be a multiple of `lcm(128, 3)`.**
  Ours is **A9**: vLLM's tuple-shard loader slices the checkpoint's single 24,576-wide `qkv_proj`
  using the module's *padded* `output_sizes` (3 × 8,448), so segment 1 reads from the wrong offset and
  segment 2 overruns; it now splits by the checkpoint's real widths and lets the existing zero-pad
  widen each rank's slice. No-op at TP≤2. [docs/03](docs/03-tp3-padding-and-sidecars.md) §1.1.

- **A10, the pad audit — and the invariant it turned out to be checking had been holding by
  accident** `[measured-here]`. The `svh = 0` mechanism was already running on this stack for
  column-parallel EXL3 modules before anyone designed it, and **nothing verified it** — which is
  precisely how the old 2,112 arithmetic would have produced silently wrong output rather than an
  error. A10 now walks every EXL3 module after load and reports:

  ```text
  HAREM-FULLSCOPE assert 5: 285 EXL3 pad site(s) audited, 285 padded on this rank, all whole 128-blocks and exactly zero
  ```

  285 is what a model-free meta-device run predicted for rank 2 **before** the boot. The audit also
  caught its own bug on first writing — it called a column-parallel module's *input* padded and
  rejected `in_proj_qkv` on ranks 1 and 2 while rank 0 passed. **A single-rank test could not have
  seen it.**

- **Two TP=2 warnings did not reproduce, and one is a retraction** `[retracted]`. The dress rehearsal
  said the full-scope checkpoint was ~10 GiB *heavier* per node and projected the TP=3 pool falling
  4.70 M → ~3.4 M, −27 %, with the 1M-context claim at risk. Measured at three ranks: **3.4 GiB
  lighter, pool +10.0 %**, `MAX_MODEL_LEN` never touched. The TP=2 pair was confounded — different
  checkpoints *and* different `max_model_len` — and we record it as **not reproduced rather than
  explained**, because the mechanism was never isolated. Audit row 32. The second, a cold-probe
  signal that draft acceptance collapses, is flat on that same probe at TP=3 (row 30).

- **Ten boot gates, all held on the first attempt** `[measured-here]`: the image capability
  precheck (`[padload] ... =yes` on all three, or exit 23 before a byte of weight is read), the ten
  patch anchors with the patch `sha256`, the preflight arithmetic, asserts 1–4 silent, assert 5 at
  285/285, the padded `lm_head` line (154,880 → 155,136, two whole Hadamard blocks zeroed through
  `svh`), the `CUDA_EXL3_DEBUG_NAMES` tally at **203 EXL3 / 113 bf16** read negatively, the KV pool,
  free memory and swap, and both quality gates cold and warm. The acceptance list is written out in
  [docs/09](docs/09-measurement-protocol.md) §5.1, in the order it was run.

- **What stays BF16, measured rather than inferred.** The 113 are four families and nothing else:
  KDA `f_b_proj`, `g_b_proj`, `in_proj_bfg_a` and MLA `kv_b_proj`. They are why 17.8 ms arrived
  rather than the ~32 ms the estimate implied, and closing them is a **checkpoint-side** decision, not
  ours ([docs/11](docs/11-open-issues.md) §2.25).

- **Upstream.** Two more commits from the kernel author, and both were required — `f3e3090` (a padded
  output dim accepted when the pad is whole 128-blocks, `svh` allocated zeroed; and a row-parallel
  `suh` load that copies what exists and zeros the rest instead of narrowing off the end) and
  `754421f` (the vocab loaders fill a prefix). On `f3e3090` alone the boot clears `create_weights`
  and dies on a `copy_` shape mismatch in `_vocab_loaders`; on anything older it raises "EXL3 weights
  cannot be zero-extended". [CREDITS](CREDITS.md).

- **In the repository.** `tracks/tp3/patches/` (the whole production tree, with its own README),
  `tracks/tp2/patches/patch-fullscope-tp2.py` (the eight-anchor TP=2 patch, which had been referenced as
  "not yet in repo"), `tracks/tp3/env.tp3-full.example`, and `scripts/ab-quick2-full.sh` +
  `scripts/boot-only-full.sh` — the tier-B harness for the full-scope tree, which differs from the
  production one in a single line and was added rather than edited, for the same reason the patch
  tree was ([docs/09](docs/09-measurement-protocol.md) §11.2).

- **Rollback, one line or one file.** Delete `HAREM_EXL3_FULLSCOPE=1` from `EXTRA_ENV` and the same
  patched image takes the upstream path; or start with `ENV_FILE` pointing at the production 8 env
  file, which reverts the checkpoint, the image, the patch tree and the sidecar together because all
  four are named in it. `patches/tp3/` and production 8's sidecar were never modified.

---

## 2026-09-05 (evening) — the full-scope checkpoint loads, and the largest item on the stack is now a measurement

The item this repository has called its largest is no longer an estimate, and the reason it had been
out of reach turned out not to be quantization, parallelism or the weights.

- **A full-scope EXL3 checkpoint served for the first time on this stack, at TP=2**
  `[measured-here]`. `turboderp/GLM-5.3-Flash-exl3` at 4.05 bpw (`2a30229e`, **MIT**, 165 GB,
  `sha256` 23/23 verified on both nodes) against our production checkpoint as control, same image,
  same two nodes, expert parallel off, DFlash2 k=7, medians of three rounds:

  | | full scope | experts-only control | delta |
  |---|---|---|---|
  | C1 per stream / aggregate | **68.00 / 59.93** tok/s | 54.69 / 47.40 | **+24.3 % / +26.4 %** |
  | C2 / C4 aggregate | 83.02 / 111.05 | 68.03 / 90.66 | +22.0 % / +22.5 % |
  | TTFT at C1 | 0.524 s | 0.615 s | −14.8 % |
  | draft acceptance, accepted length | 63.14 %, 5.42 | 64.08 %, 5.49 | unchanged |
  | gates, cold and warm | 10/10 · 12/12 | 10/10 · 12/12 | equal |
  | MMLU sample (1,995 q) | **86.32 ±0.75** | 86.4 ±0.7 | inside the bar |

  Per decode step 100.4 → **79.7 ms**, 20.7 ms saved, with acceptance and accepted length flat — the
  entire gain is arithmetic, none of it is drafter behaviour. Against the estimate this stack carried
  (42.9 ms → ~11 ms, ~+34 %), **65 % arrived**; the rest is what this checkpoint leaves in BF16
  anyway. **The 6-bit `lm_head` at vocab 154,880, the one quality risk we had flagged, cost nothing
  measurable.** New page: [docs/13](docs/13-full-scope-checkpoint.md);
  [docs/10](docs/10-results-and-roofline.md) §2.2, [docs/11](docs/11-open-issues.md) §2.22.

- **Retracted: "the dense stage is a quality choice"** `[retracted]`. It was a **loader limitation**.
  vLLM's `glm5next` model file pins the attention stack to BF16 in two places — `quant_config=None`
  for the MLA projections and a `quant_config` strip in the KDA constructor — and between them they
  lock **72.8 % of the dense traffic** whatever the checkpoint holds. `routed_experts_only` was not
  the publisher's judgement about quality; it was the only scope that could load. Retraction 29.

- **Retracted: "draft acceptance drops on a quantized target"** `[retracted]`. A cold single-prompt
  probe read 45.5 → 39.2 % and 48.0 → 34.4 % and was reported upstream as an early signal that would
  "eat part of the speed gain". Over three sweep rounds acceptance is 61–65 % against the control's
  62–63 %. Sample of one, published. Retraction 30.

- **Corrected: the delta we posted upstream** `[retracted]`. The issue-thread table gave the control's
  C1 as "54.7 / 54.3" aggregate / per stream and derived "+9.5 % / +25 %"; 54.7 is the control's
  *per-stream* median and its aggregate median is 47.40. Like for like: **+26.4 % aggregate, +24.3 %
  per stream.** The conclusion is unaffected. Retraction 31 — the only row in that table produced by a
  summary rather than a measurement.

- **Three loader layers, one env-gated patch, and a vLLM bug** `[measured-here]`. The checkpoint would
  not load for three independent reasons, measured as an unmapped-tensor count going
  **886 → 526 → 170 → 0**: no `packed_modules_mapping` on the model class; the BF16 pinning above; and
  a KDA factorisation mismatch (one `qkv_proj` and one `conv1d` against the model's fused
  `in_proj_qkvbfg_a` and three `conv1d`). The last of those is **not an EXL3 problem** — `conv1d` is
  BF16, so a BF16 copy of the same checkpoint fails identically. Eight anchors, four asserts, all
  silent on the live boot, and with the flag unset the image is upstream byte for byte and still
  serves the control checkpoint at 0 unmapped / 0 unfilled. On the way: **`ReplicatedLinear` has no
  `weight_loader_v2` dispatch**, so any replicated linear served by a v2-parameter quantization method
  fails at load — reported, and fixed upstream in the plugin rather than worked around.

- **What it cost, and the line is not empty.** At TP=2 the full-scope model leaves a **31,343-token**
  KV pool, so prompts above ~2,000 tokens are **never scheduled** — every prefill figure and the
  C6/C8 aggregates are void in that arm, and TP=2 is a measurement rig rather than a serving
  configuration. It is also **~10 GiB heavier per node** despite being 10 GiB smaller on disk, and
  that contradiction is unexplained and left standing; it is the open risk for the TP=3 pool
  (4.70 M → possibly ~3.4 M) and the one number to measure before spending a boot.
  Production was down 1 h 15 m, planned, and came back at 4,696,969 tokens against 4,699,724, gates
  full.

- **Two instruments failed and are written down.** `CUDA_EXL3_DEBUG_NAMES=1` printed nothing — the
  plugin logs it at `info`, and this image configures only vLLM's logger — so the designed boot gate
  was unusable and three indirect checks carried it instead; the author has since fixed the level and
  made it log the hits as well as the misses. And a fast-load sidecar **refused the production
  restore** because an experimental patch had been placed in the hashed patch directory: correct
  behaviour, an hour late. New rules: [docs/09](docs/09-measurement-protocol.md) §11.2 (nothing is
  added to the patch directory between a dump and a load; experimental patches live elsewhere and are
  hard-linked) and §11.3 (the meta-device name-set check as the tier-A acceptance test for any loader
  change).

- **Upstream.** Four commits from the kernel author came out of this thread in one afternoon —
  a checkpoint-declared packed mapping (`5903248`), the same mapping from an environment variable
  (`fba9f27`), the `ReplicatedLinear` `suh` fix plus a corrected README example (`d19dee0`), and
  `CUDA_EXL3_DEBUG_NAMES` printing at all (`807d798`) — plus the one answer that unblocks TP=3 in
  principle: **all 65,536 trellis codes swept through the device decoder on all three codebooks, zero
  non-finite, bounded ranges**, so a zero pad trellis multiplied by `svh = 0` cannot produce NaN.
  [CREDITS](CREDITS.md).

- **What is open.** TP=3: `padding_size` 192 → **384**, shared expert 2112 → **2304** (not 2176 —
  128-aligned but not divisible by three), the A9 KDA split fix which is written down and not written,
  the checkpoint onto the third node, a re-dumped sidecar, and the padded-load path for the
  vocab-parallel head on the plugin side. [docs/11](docs/11-open-issues.md) §2.22,
  [docs/13](docs/13-full-scope-checkpoint.md) §7.

---

## 2026-09-05 (afternoon) — production configuration 8, and a profiler that deleted two of our own targets

Two entries in one day, and the interesting one is not the production change.

- **Production configuration 8** `[measured-here]`. Production 7 with the image moved to
  `exl3-zeus:62f53e6` — upstream's `had_in` commit `a47da6e` plus the note that bounds what is left of
  it. Sidecar re-dumped for the new image, `.env.tp3` promoted, **nothing else changed**:

  | | production 7 | **production 8** |
  |---|---|---|
  | KV pool @ 0.80 | 4,699,724 | 4,674,931 |
  | C1 / C2 / C4 / C6 / C8 aggregate tok/s | 57.0 / 80.9 / 120.0 / 143.4 / 175.1 | 56.8 / 83.5 / 119.5 / 146.0 / 172.8 |
  | prefill-fresh (3 rounds) | 1,733 / 1,769 / 1,788 | 1,769 / 1,780 / 1,789 |
  | draft acceptance | 60.8–64.3 % | 62–64 % |
  | gates cold and warm | 10/10 · 12/12 | 10/10 · 12/12 |

  **Every column is inside its own band and that is the entry.** The commit was priced at 0.2–0.3 % of
  prefill wall *before* it was built — deliberately below what a serving benchmark can see — and it was
  adopted to keep the image on upstream's head, not to buy tokens. Four signs in four directions, no
  mechanism, three rounds each. A stack that publishes 1 % changes as wins would have called two of
  these. [docs/10](docs/10-results-and-roofline.md) §1, [docs/11](docs/11-open-issues.md) §2.19.

- **The step-time breakdown is now a measurement, and it cost one launcher flag** `[measured-here]`.
  A torch profiler ran against the **live** production 7 server, all three ranks, with no restart and
  no reconfiguration — only `/start_profile`, `/stop_profile` and ordinary requests. This vLLM takes
  the profiler as `--profiler-config` and **not** as `VLLM_TORCH_PROFILER_DIR`; with the environment
  variable alone the route is never attached and the endpoint answers 404, which is why
  [docs/10](docs/10-results-and-roofline.md) §5 spent a week as a reconciliation with a 2.8 % residual.
  The flag is now in `scripts/start-tp3.sh` behind `PROFILER_DIR`, off by default and free when unset.
  Full tables: [`results/profile/measured-prod7.md`](results/profile/measured-prod7.md); how-to and the
  traps: [docs/09](docs/09-measurement-protocol.md) §4.1.

  What it found, in order of consequence:

  | | re-derived | **measured** |
  |---|---|---|
  | prefill chunk size | 2,032–2,048 tokens | **1,792** (7 × 256; 12.5 % of the budget unused) |
  | `exl3_moe_combine`, prefill | 1.5 % — ranked target #10 | **0 % — the kernel does not exist in this build** |
  | `_zero_kv_blocks`, prefill | 1.3 % — ranked target #8 | **0.09 %** (0.86 ms, not 14.7) |
  | DFlash2 drafter, C1 | 18.5 ms, 19.5 % | **10.78 ms, 11.4 %** |
  | NCCL, prefill | 14–17 % band | **14.47 %** — the bottom of it |
  | C8 decode split | `[not tested]` | MoE **51.6 %**, dense GEMM 21.1 %, NCCL 11.7 % |
  | comm/compute overlap | assumed some | **0.00 ms/step at prefill** — every microsecond of NCCL is exposed |

- **Retracted: "5.45 ms of C1 GPU idle" and "CUDA graphs would return ~6 % of single-stream"**
  `[retracted]`. Both were published, one of them upstream. **~2.0 ms of that idle is CUPTI itself** —
  about 1 µs per kernel boundary across ~1,873 of them — so the real budget is **3.47 ms = 3.75 %** of
  a profiler-off step. And "CPU gap" was the wrong name: **77 % of it is per-kernel dispatch**
  (2,332 kernels per step), 18 % the host, 5 % one blocking sync — with the host running 3.9 ms
  *ahead* of the GPU at C1. Graph capture is worth **1.4–1.9 ms, +1.5–2.1 %**, and removes neither the
  step head's `prepare_inputs` nor the sync. This also explains something that had been sitting
  awkwardly: the previous configuration *did* capture graphs and read the same 57 tok/s. Retractions
  26–28 in [docs/11](docs/11-open-issues.md) §1.9, story in §1.10.
  **The rule this adds to the protocol:** never read an idle figure out of a trace — take
  `busy(union)` from the trace and the wall from a profiler-off run of the same window.

- **Closed by the kernel author, not by us: MLA prefill** `[reported]`. It had been carried as
  "8.2 % of a chunk, efficiency not measured, no denominator" for a week. Measured at our shapes —
  top-k 2,176, head_dim 512, fp8 cache, a 2 GB pool chosen so it could not sit in L2 — it runs at
  **86–89 % of achievable**. It sits with the trellis GEMM: near the roof, no traffic to remove. Worth
  recording that his first cut used a 200k-row pool that fitted L2 and duly reported a bandwidth above
  the DRAM ruler; the same class of error as our own model-free MoE bench overstating small-M cost by
  1.5–1.7×. **Synthetic benches flatter small inputs, and both of us hit it the same day.**

- **New open item, and it is the largest one this repository has ever carried: the checkpoint's scope**
  `[estimate]`. Dense BF16 GEMM is **45.3 % of a C1 step** because we run
  `scope: glm53_routed_experts_only`. At 4 bpw that stage is bandwidth-bound on ~4× less traffic —
  42.9 ms → ~11 ms, **~+34 % single-stream** — against about 5 % for the whole `cuda-exl3` column of
  our target table. The TP=3 blocker is softer than [docs/01](docs/01-model-and-license.md) §3.1 said:
  an EXL3 tensor still cannot be zero-extended, but a checkpoint quantized *unpadded* can be loaded
  into a padded parameter with `svh = 0` on the pad — bit-exact, no re-quantization — **provided the
  pad occupies whole 128-column blocks**. Our head pad does (64 → 66 heads = 256 columns = 2 blocks);
  our vocab pad does not (`padding_size=192` = 1.5 blocks) and would at **`padding_size=384`**, which
  is one constant in our launcher. Ordering agreed upstream: **quality first** — a TP=2 trial of
  `turboderp/GLM-5.3-Flash-exl3` at 4.05 bpw against the MMLU sample, reported either way — then the
  padded-load path behind a flag if the gate holds. [docs/11](docs/11-open-issues.md) §2.22.

- **Docs.** `scripts/start-tp3.sh` gains the `PROFILER_DIR` arm — and with it a rule:
  **one launcher, one copy.** Two copies of this file existed for two days; the one the nodes ran had
  the profiler arm and the one in the tree did not, which is precisely the divergence that is invisible
  because both copies produce a working server. [docs/08](docs/08-fast-boot.md) §12.
  New: [docs/09](docs/09-measurement-protocol.md) §4.1 (profiling a live server, and verifying the
  profiler), [docs/10](docs/10-results-and-roofline.md) §5.7–§5.8,
  [docs/11](docs/11-open-issues.md) §1.10, §2.22, §2.23,
  [`results/profile/measured-prod7.md`](results/profile/measured-prod7.md).

---

## 2026-09-05 — production configuration 7: the draft cache at fp8, and a ruler that had been moving all along

The pool moved for the first time in three configurations, and the more useful half of the entry is
that we found out why it had been moving on its own.

- **Production configuration 7** `[measured-here]`. Production 6 plus an **fp8 DFlash2 draft cache**
  (`HAREM_DRAFT_KV_DTYPE=fp8`), a tilelang fail-loud guard, a FlashInfer warm-up, an idle profiler
  endpoint, the launcher's new memory settle gate and a fresh per-rank sidecar. Read from an ordinary
  load boot:

  | | production 6 | **production 7** |
  |---|---|---|
  | KV pool @ 0.80 | 4,449,035 | **4,699,724 (+5.6 %)** |
  | C1 / C2 / C4 / C6 / C8 aggregate tok/s | 56.9 / 84.2 / 118.5 / 142.9 / 168.9 | 57.0 / 80.9 / 120.0 / 143.4 / 175.1 |
  | TTFT C1 / C8 | 0.41 / 1.01 s | **0.34 / 0.91 s** |
  | draft acceptance (per-concurrency medians) | 61–65 % | 60.8–64.3 % |
  | prefill-fresh / warm 7K repeat | 1,792 / 1,506 | 1,769 / 1,529 |
  | gates cold and warm | 10/10 · 12/12 | 10/10 · 12/12 |
  | free RAM / swap | 11.3 / 12.6 / 12.5 GiB · ~0.1 | 12.3 / 13.5 / 13.3 GiB · ~0.1 |

  **Speed is unchanged and is not the claim.** C8 reads +3.7 % and C2 −3.9 % on the same three rounds:
  opposite signs, both inside their own bands, and no mechanism by which a smaller draft cache would
  make decoding faster. The claim is the pool, at **+5.6 % against +4.7 % predicted**, and a TTFT that
  improved at both ends. [docs/07](docs/07-kv-and-draft-page.md) §7,
  [docs/10](docs/10-results-and-roofline.md) §1 and §2.1, [docs/11](docs/11-open-issues.md) §2.18.

- **The KV pool number was never a memory reading** `[measured-here]`. On this integrated-GPU part vLLM's "free GPU memory" is `/proc/meminfo`
  `MemAvailable`, and "consumed memory (weights + non-torch)" is the *difference* between two such
  readings taken minutes apart. The instrument therefore runs backwards — a node that starts dirty
  computes itself a **larger** pool — and the launcher made it systematic by killing a ~90 GiB
  container and starting the next immediately, in a fixed node order, so the last node started was
  always ~9 GiB short — **27 % of a rank's KV allowance** sitting inside the measurement. **Fix: a
  host-side settle gate** in `scripts/start-tp3.sh`, which waits after `docker rm -f` until
  `MemAvailable` is back over `SETTLE_MIN_GIB` (112) before `docker run`. Per-rank startup free memory:
  spread **9 GiB → 1.4 GiB**. KV tokens bought: **zero**, and **no published figure is corrected** —
  the pool takes the minimum over ranks and the polluted node happened never to be the binding one,
  which is luck, not design. Two rules now stand with it: **read a pool only from a load boot**, and
  **check all three ranks agree within 1 GiB before quoting one**.
  [docs/07](docs/07-kv-and-draft-page.md) §1.1, [docs/08](docs/08-fast-boot.md) §5.1,
  [docs/09](docs/09-measurement-protocol.md) §11.1.

- **Retracted, and it was the largest open item on the page: "8.2 GiB per worker is stranded"**
  `[retracted]`. Yesterday's entry priced equalising the ranks at 8–26 % of pool. There is no stranded
  KV. The three ranks start 9.00 GiB apart and finish the profile **0.99 GiB** apart, so the gap was
  never an allocation; live readings during the same boot have the head node with *less* free memory
  than the workers, where the profile had given it 8.2 GiB more; and the pool is sized on the minimum
  over ranks, which is the correct figure. Acting on the claim would have over-committed the head node
  by ~8 GiB — the exact recipe for the 0.85 boot that swapped. This is retraction 25 in
  [docs/11](docs/11-open-issues.md) §1.9, and it is the same failure as the datasheet rulers with a
  better disguise: **the number was printed by the engine itself, about its own memory, and it was
  still an unverified ruler.** [docs/11](docs/11-open-issues.md) §2.3.

- **`NCCL_ALGO` swept model-free: `Tree` rejected, `Ring,Tree` unresolved** `[measured-here]`. Three
  arms, two repetitions, production plugin and environment. `Tree` is 4–6× slower than Ring at 16 and
  64 MB all-reduce (3.15/3.36 against 20.23/18.30 GB/s at 16 MB), 23–96 % worse on the decode-step
  proxy, its RNR retries are an order of magnitude higher, and the port counters show it
  redistributing traffic asymmetrically across three nodes that have no hierarchy to reward it.
  `Ring,Tree` is 3.6 % better on the step proxy and worse on `sendrecv` at 64 MB — **and the sweep's
  own repeat-to-repeat spread is up to 1.7× at 1 MB**, so model-free cannot settle it. Deferred to a
  five-round engine arm worth an expected −1…3 % of a decode step; `Ring` stays in the launcher.
  [docs/06](docs/06-nccl-mesh.md) §12.2, raw in
  [`results/mesh/algo-sweep.md`](results/mesh/algo-sweep.md).

- **The `cuda-exl3` MoE stage is closed upstream** `[reported]`. After taking `exl3_moe_had_in` in
  `a47da6e`, the author bounded the remainder in `62f53e6`: what is left is a half-ALU limit — a
  128-point Hadamard done with warp shuffles — so it is arithmetic that has to happen rather than
  traffic that can be removed, worth **≤2 % of prefill** here and unreachable in practice. He also
  confirmed our reading that `_zero_kv_blocks_kernel` belongs to vLLM rather than to the kernel
  library. Every remaining prefill lever on this stack now belongs to vLLM, to the fabric, or to us.
  [CREDITS](CREDITS.md), [docs/10](docs/10-results-and-roofline.md) §6,
  [docs/11](docs/11-open-issues.md) §2.19.

- **New open item, and it is a decision rather than a measurement:** `gpu-memory-utilization 0.83`.
  The knob multiplies `MemTotal`, so it is the one pool input the baseline cannot corrupt: +3.65 GiB
  of budget landing whole on the binding rank, pool ~4.70M → about **5.2M (+11 %)**. Price stated:
  the OS share falls 20 % → 17 % and the head node goes from ~12.3 to ~8.4 GB free under load, still
  clear of the 4 GiB rule. One boot, reversible in one line, **held for the stack owner's approval**
  `[not tested]`. [docs/11](docs/11-open-issues.md) §2.4, [docs/07](docs/07-kv-and-draft-page.md) §6.

---

## 2026-09-05 — a kernel that half-worked, an overlap that does not pay, and a stricter definition of "equal"

Production configuration 6 is **unchanged**, for the second entry running. One change is validated and
waiting on a boot; three designs were costed and rejected; and the measurement protocol got stricter
about what counts as a difference.

- **Draft KV at fp8: validated, not yet promoted** `[measured-here]`. Moving the DFlash2 drafter's own
  cache from bf16 to fp8 shrinks its page 393,216 → **196,608 bytes** and the per-block cost
  21,917,440 → **20,934,400**, with the blocks-per-request divisor unchanged at 363 — worth about
  **+4.7 %** of pool. The arm booted, which answered the only question a boot could: the DFlash
  sliding-window backend accepts an fp8 cache. Draft acceptance **60.1–64.0 %** against production's
  61–65 % and a gate that demanded the 60–65 band, gates 10/10 · 12/12 cold and warm, speed inside the
  noise bands. **It ran on a dump boot, so its KV pool figure (4,382,920) is meaningless** and
  production stays on bf16 until an ordinary load boot supplies the real number.
  [docs/07](docs/07-kv-and-draft-page.md) §7, [docs/10](docs/10-results-and-roofline.md) §2.1,
  [docs/11](docs/11-open-issues.md) §2.18.
- **The hyper-connection fusion kernel was written and it reached 40 % of its own ceiling.** A Triton
  kernel that tiles over tokens and reduces the post mapping in registers removes 30.4 % of the first
  two kernels' traffic and delivers **−14.9 to −15.5 %** on that pair, **−9.0 to −9.9 %** on the
  three-kernel route, and **−1.0 to −1.1 % of the prefill wall** — against a −2.1 to −2.8 % target.
  `residual_cur` is bit-identical at every M but one 7-element 1-ulp difference at M=64; `layer_input`
  is within one bf16 ulp on 5.1 % of elements. The shortfall is entirely bytes-per-second (187.7
  against the 229.5 GB/s the route it replaces gets), not tiling — a 33-configuration sweep could not
  improve the winner. **Not adopted standalone**: −1 % does not earn a boot when it also brings Triton
  JIT into the serving process and a configuration surface that is a cliff (the winner reads
  187.8 GB/s, its neighbours 79.4 and 44.5, and the shipped default was one of the bad ones). It rides
  the next image bundle with `had_in`. Also measured: fusing **loses** below M ≈ 1024 (+37.7 % at
  M=512), because the residual fits the 24 MiB L2 there and the re-read it deletes was never going to
  DRAM `[measured-here]`. [docs/10](docs/10-results-and-roofline.md) §5.5.1.
- **New lesson, and the more useful half of that kernel:** a GPU-free ahead-of-time compile check
  reported 18 of 18 configurations building and all inside the shared-memory limit; at real launch
  **6 of the 18 failed with `OutOfResources`**, the reported 36,864 bytes against 106,496 actually
  needed. **A compile check answers "does it build", never "does it run"**
  `[measured-here]`. [docs/11](docs/11-open-issues.md) §4.
- **Closed: dual-batch overlap (DBO).** The all-reduce is 16.5 % of a prefill chunk at 99.3 %
  occupancy, so overlapping it carried the largest number on the open-issues page. The mechanism does
  suit it and the patch is *smaller* than we estimated (~95–160 lines, five files, and it does not
  touch the model file — one bottleneck covers all 102 collectives, and the mHC state is thread-local,
  so our "medium-to-large patch, mHC at risk" reading was wrong on both counts `[retracted]`). The
  arithmetic kills it: splitting the batch pays the MoE expert weight stream **twice**, +73 to +232 ms
  per chunk against −135 ms of hideable collective. Prefill lands at **−6.3 % to +8.0 %** — a coin
  toss — and decode is a clear loss (**C1 +38 %**, C8 +6 %), with the drafter unable to micro-batch,
  the breakable CUDA graph disabled for both models, and a silent-corruption hazard where a split
  request restarts the KDA recurrent state from zero in 34 of 45 layers. Raising the batched-token
  budget beats every overlap variant for the same KV price and no code. **Do not build it.**
  [docs/11](docs/11-open-issues.md) §2.17.
- **Also dead, each for a checkable reason, read out of the image with the engine down:** async
  tensor parallelism, the sequence-parallelism pass and the FlashInfer all-reduce+RMSNorm fusion all
  need `torch.compile`, which this model family never enters; and `world_size = 3` is excluded by the
  supported-world-size lists of FlashInfer (2/4/8/16), custom all-reduce (2/4/6/8/16) and NCCL
  symmetric memory (minimum 4). DeepEP is installed and never engages at `data_parallel_size = 1`.
  Model-level sequence parallelism is a one-line gate and a bad idea: identical bytes, collective
  count 90 → 180, **+10…15 % worse at decode**.
- **One survivor in that class, and one free probe.** Attention-scoped micro-batching — split across
  the attention block only, rejoin before the MoE stage — hides 44 % of the collective while paying
  only for attention weights streaming twice: **−3 to −6 % of prefill** `[estimate]`, ~150 lines, and
  it inherits the same KDA hazard. Before any of it, a model-free probe should establish whether an
  all-reduce on a second stream overlaps a GEMM **at all** on this part. Written, not run
  `[not tested]`.
- **The measurement protocol got a floor.** Round-to-round spread inside a single settled arm is
  **C1 ±4 %, C2 ±6 %, C4 ±9 %, C6 ±6 %, C8 ±3 %** `[measured-here]` — C4 is the noisiest column, which
  is the opposite of the intuition. Two rules now apply to every table here: **a difference of 3 % or
  less is written down as "equal"**, and above that floor it still has to clear its own metric's band.
  This reclassifies the combine-staging arm's +2.3 % at C4 and patch 0007's −0.9…+4.2 % as equal;
  production 5 → 6 survives on its +5.6 % at C8. Also corrected: the quick-arm harness claimed five
  rounds with two discarded and ran three `[retracted]` — it is now one warm-up plus three measured
  rounds, and the warm-up ramp it was written for is gone anyway on a warm tuner cache.
  [docs/09](docs/09-measurement-protocol.md) §1.1–§1.2.
- **New planning rule: a patch change costs a dump boot.** The fast-load sidecar's identity covers
  *every* `patch-*.py` and the whole prelude, so three patches that touch no weight byte refused a
  boot and cost an hour. Budget the 682-second dump boot into any arm that adds a patch, and never
  record a dump boot's KV pool as a result. The narrower gate this argues for is written up but not
  written. [docs/09](docs/09-measurement-protocol.md) §11, [docs/11](docs/11-open-issues.md) §2.21.
- **Rank memory imbalance, now with a number and a diagnosis label.** The weights are identical on all
  three ranks (`Model loading took 54.86 GiB` ×3), yet non-torch memory reads **1.50 GiB on rank 0
  against 9.48–9.72 GiB on the workers** — about **8.2 GiB per worker stranded**, and the pool is sized
  by the worst rank, so equalising it would be worth **8–26 % of pool**. Larger than every kernel item
  left. Nobody knows yet what that memory is `[measured-here]`.
  [docs/11](docs/11-open-issues.md) §2.3.
  **Correction, same day** `[retracted]`: it is not memory, it is the instrument. "Non-torch" is a
  delta between two `MemAvailable` readings and the last node started is the one given least time to
  reclaim the previous container; the three ranks finish the profile 0.99 GiB apart. No KV was
  stranded, and equalising the ranks would have over-committed the head node by ~8 GiB. See the entry
  for production configuration 7.
- **The memory ladder is re-opened rather than settled.** The 0.85 rejection predates the fast-load
  work that removed the page-cache spike; the same configuration now sits at 11–12 GiB free with zero
  swap, and 0.82–0.83 was never tried. `--kv-cache-memory` — sizing the pool in bytes rather than as a
  fraction of the device — has never been used. Ladder first, pin last
  `[not tested]`. [docs/07](docs/07-kv-and-draft-page.md) §6.
- **`NCCL_ALGO=Ring,Tree` has never been run on this mesh**, and it is the cheapest untried thing in
  the repository: our launcher forces `Ring`, decode is latency-bound on a fixed 102 collectives per
  step, and a tree is ~3.2 steps against a ring's 4 at `world_size = 3`. Expected −1…3 % of a decode
  step; the sweep is model-free and costs nothing `[not tested]`.
  [docs/06](docs/06-nccl-mesh.md) §14 item 8.
- **New `systemd/` directory — a template, deliberately not installed.** With it, the hazard it
  exists to name: the NVFP4 sibling's `harem-motor.service` is `enabled` on all three of our nodes, so
  a reboot brings up the **other** engine on the same GPUs. The template's three unfinished pieces are
  named rather than fixed — its preflight script does not exist, systemd will not honour the
  worker-2 → worker-1 → head start order on its own, and its `ExecStop` names the wrong container.
  [systemd/README.md](systemd/README.md), [docs/11](docs/11-open-issues.md) §2.20.
- **A complete retraction audit.** Every published claim of ours was re-read against the raw data
  behind it; **24 did not survive** and all 24 are now in one table with what replaced them, including
  ones that had only been corrected in passing: the chat template we serve matches neither checkpoint
  on disk and its provenance is unverified; `NCCL_BUFFSIZE` was listed as an open lever twelve hours
  after being eliminated; `NCCL_MAX_NCHANNELS=8` had been "already tried" only in combination with
  `NCCL_PROTO=LL`; `--language-model-only` does not stop the vision tower being built, only run.
  Six of the 24 are a ruler we quoted instead of measured, four are a single pair of sweeps treated as
  a result, three are an arithmetic model a bench refuted, two are our own tooling disagreeing with
  our own documentation. [docs/11](docs/11-open-issues.md) §1.9.

## 2026-09-05 — where a step actually goes, and four items closed

Production configuration 6 is **unchanged**. Everything in this entry is measurement.

- **Both rulers were wrong, by ~22 %.** Measured on this device in our own image: achievable read
  bandwidth **225.2 GB/s** against a vendor 273, BF16 GEMM peak **97.3 TFLOP/s** against an implied
  ~125. Every roofline percentage published here before today was optimistic by that much
  `[retracted]`. Two new tools, `bench/bw.py` and `bench/gemmpeak.py`, seconds each; run them in the
  same process as the thing you are measuring. The read ruler itself drifted 6.5 % across three runs
  on the same idle machine the same morning, so percentages are now given as bands where it matters.
- **Step-time breakdown, per 2,048-token prefill chunk** (1,109 ms, occupancy 99.3 %): MoE trellis
  GEMM **26.4 %**, NCCL all-reduce **16.5 %**, dense BF16 GEMM **16.2 %**, hyper-connection mixing
  **11.7 %**, MLA 8.2 %, KDA 7.5 %, MoE `had_in` 6.1 %, KV zeroing 1.3 %, DSA indexer 0.6 %. Per C1
  decode step (89.1 ms): dense BF16 GEMM **44.8 %**, MoE trellis GEMM 29.3 %, all-reduce 10–15 %, and
  the k=7 drafter **19.5 %**. Since the previous configuration, prefill throughput at 8.4K is
  **+43 %** and the C1 step is **−17.5 %**.
- **Two corrections to our own earlier reading of the same trace** `[retracted]`: the `mhc_*` kernels
  are hyper-connection mixing, a class of their own worth 11.7 %, not dense GEMM and not the indexer;
  and MLA is 7.4 % in a steady chunk, not the 9 % a window average reported.
- **Method, stated because it has a caveat.** The running engine had no profiler endpoint and a
  restart was not available, so this is a reconciliation — structure from an earlier trace
  re-segmented per chunk, changed classes re-measured model-free, totals measured live — with a
  **2.8 % residual**. Read NCCL as a 14–17 % band. One profiling boot closes it.
- **Closed: expert-stationary MoE scheduling.** Our 14–27 % traffic estimate rested on a traffic
  model a trace cannot verify. The kernel author wrote a bench for it (`9b17ea9`); run unmodified on
  GB10, three times: doubling blocks per expert costs **1.11×**, not 2×. The trellis stays resident.
  Nothing to win `[measured-here]`.
- **Closed: the KV-zeroing gate.** The kernel runs at 100 % of the memset roofline and zeroes
  2.4–2.9 GB per chunk against ~3.4 MB of real new KV, so the only lever was to skip it. It cannot be
  skipped here: **85.5 %** of those bytes are MLA pages co-owned with Mamba/KDA state in this model's
  hybrid layout, which is the Mamba half of vLLM's condition and is independent of precision. A
  fail-closed gate was written so the machine checks the conclusion; on this model it refuses to boot,
  by design. Safe remainder 0.19 % of prefill; no partial mode written `[measured-here]`.
- **Closed: a cooperative (`grid.sync`) MoE stage.** Outside a CUDA graph the barrier wins on a 48-SM
  part, as predicted — up to 33 % at medium sizes. **Inside a graph, which is what production runs,
  the sign flips back** and it costs 0.2–0.3 µs per phase boundary. The deciding detail was the
  graph, not the SM count `[measured-here]`.
- **Closed: the DSA indexer**, at 0.6 % of prefill — the same conclusion an earlier micro-benchmark
  reached from the other side, now confirmed from the share itself.
- **Hyper-connection mixing measured**: 86–91 % of the ruler, memory-bound by a factor of 76, **not**
  launch-bound (CUDA graphs change it by 0.03 % at M=2048), **not** badly tuned (the two available
  knobs are worth 0.4 % of prefill, and the third kernel does not compile above 96 threads), and the
  torch fallback is unreachable on CUDA and 5–15× slower. One real lever remains: fusing the first two
  kernels to stop re-reading the residual, **−2.5 to −2.7 % of prefill**, which needs a new large-M
  kernel — forcing the existing fused one is +32 % worse. Our earlier −3.6 % estimate was 30 %
  optimistic `[retracted]`.
- Upstream took `exl3_moe_had_in` in `a47da6e` (−10…18 % on that kernel, ~0.2–0.3 % of prefill here).
  **Not in the production image**; queued for the next build.
- New protocol rule, learned the hard way: **one measurement holds the cluster at a time**, written in
  a lock file, and three-node NCCL work needs the engine **down** rather than idle — a fabric sweep
  beside a live engine can exhaust queue-pair resources and take its next collective with it.
  [docs/09](docs/09-measurement-protocol.md) §10.

## 2026-09-05 — the fabric ceiling is PCIe, and a transport rewrite that changed nothing

- **Retracted, one day after we published it** `[retracted]`: "the pair of cables is worth 50 GB/s,
  so the collective is at 28 % of the fabric". That is the **wire**. Each ConnectX-7 sits in a
  **PCIe Gen5 x4** slot (`LnkSta: Speed 32GT/s, Width x4`) and carries ~15 GB/s regardless of its two
  200 Gb/s ports, so the real ceiling is **~30 GB/s per node** and the collective at ~20 GB/s is at
  about **70 %** of it. The old 13 GB/s ceiling was never "half a link" either — it was one card's
  PCIe limit at 87 % of it, which is why the second cable, on the second card, took it to 20 and not
  to 40. **Remaining fabric headroom ≤30 %, worth 2–4 % of prefill**, against the 12–17 % we had
  priced it at. Two wrong ceilings in two days, both computed from a datasheet.
- **Patch 0007: one-sided transport (receiver-advertised FIFO + `RDMA_WRITE_WITH_IMM`), built,
  measured, not adopted.** +977/−16 lines over 0004–0006: FIFO in the sender's memory (one per cable
  after 0005), zero-byte RECV armed before the slot is advertised so **RNR becomes structurally
  impossible**, torn-slot double check, fail-closed ring overrun, a version handshake that keeps
  `mesh_qp_info` at 32 bytes and refuses a peer that does not speak the extension, and no dependence
  on NCCL's `*request = NULL` contract. Default `send` is byte-for-byte the old path.
- **It does exactly what it was designed to do and it is worth nothing here.** Six arms, two
  repetitions: every write arm reports **zero** RNR retries and **zero** out-of-buffer events at every
  size, against 1–9 per operation on the control — and throughput does not move. The gate (≥1.3× at
  ≥16 MB) was not met at any FIFO depth, with or without the flush, and all six arms sit within 0.17 ms
  of each other on the decode-sized message. Engine arm: C1 56.4, C8 171.1, prefill-fresh 1,763 against
  production's 56.9 / 168.9 / 1,792 — differences in both directions, inside boot spread, gates
  10/10 · 12/12 cold and warm `[measured-here]`.
- **Why**: at ~20 GB/s the transfer is against a PCIe wall, not a flow-control stall. Removing RNR
  from a path that was not waiting on RNR buys nothing.
- **Kept, not deleted, and not offered upstream.** The patch, its unit-test parity with the 0004–0006
  baseline and the measurement that rejected it are in the repository. Sending a transport rewrite
  upstream on the strength of a mechanism that moves no number would waste the maintainer's time.
- **Patches 0004, 0005 and 0006 are now on a public fork and offered as a pull request** —
  [`NNNtrance/nccl-mesh-plugin`](https://github.com/NNNtrance/nccl-mesh-plugin), branch
  `gb10-dual-link-ptrcuda` on `19924dcc`, [PR #59](https://github.com/autoscriptlabs/nccl-mesh-plugin/pull/59),
  referencing the issue thread the findings were reported on. 0007 is not in it.
- **Closed as an open item**: the collective's share of a step, which had been "the cheapest unspent
  measurement in this repository" through three separate changes. It is 16.5 % of prefill and 10–15 %
  of a C1 decode step. The C8 split still needs a profiling boot.
- **Newly written down as the larger lever**: the all-reduce is *serialised* against compute at 99.3 %
  occupancy. Making the fabric faster is worth ≤2–4 % of prefill; making the collective **overlap**
  reaches for most of 16.5 %. Nobody has tried. [docs/11](docs/11-open-issues.md) §2.17.

## 2026-09-05 — production configuration 6: both cables, and no host bounce buffer

- **Half the fabric had never carried a packet.** Two cables run between every pair of nodes;
  `mesh_connect()` discards NCCL's device index and stops at the first reachable peer address, so
  every channel to a peer rode one cable. `port_xmit_data` on the second cable of each pair read
  **exactly zero since driver load**, on all three nodes. The 13 GB/s we had been calling "the link"
  was one cable of a pair — and, as the next day's entry records, it was also one card's PCIe limit.
- `patches/kernel/0005-device-aware-link-selection.patch` (~30 lines, no wire-format change) picks
  `dev % usable` among the parallel links; with one cable, or `NCCL_MESH_LINKS_PER_PEER=1`, the
  selection is bit-identical to the stock plugin. 64 MB all-reduce **12.0 → 16.7 GB/s**.
- `patches/kernel/0006-ptr-cuda-dmabuf-and-flush.patch` advertises `NCCL_PTR_CUDA` — two lines; the
  plugin's `regMr` already registered CUDA pointers and NCCL was simply never handing it one — plus a
  real RDMA_READ `iflush` and a real DMA-BUF path. 64 MB all-reduce **16.7 → 20.8 GB/s**, RNR retries
  per operation 15 → 3.
- Engine, three rounds per arm: **C1 54.5 → 56.9, C4 112.0 → 118.5, C8 159.9 → 168.9** (+4–6 %),
  prefill-fresh 1,709 → **1,792**, TTFT C1 0.47 → 0.41 s, gates 10/10 · 12/12 cold and warm.
- **What it cost:** nothing measurable — the KV pool moved +0.4 % (inside boot noise), which was the
  line to watch because `NCCL_PTR_CUDA` moves NCCL's buffers into memory accounted under
  `gpu-memory-utilization`. The real price is not a number: a patched plugin to maintain.
- **`NCCL_MAX_NCHANNELS=16` rejected.** Restoring 8 channels per cable is the obvious follow-up and
  it is 2.5× slower on the decode-sized message (STEP90 9.3 → 26.2 ms), in both patch families. 8
  stays.
- **`NCCL_MESH_DMABUF=1` rejected.** It works — which settles whether `ibv_reg_dmabuf_mr` accepts
  these buffers here — and is slower than plain `ibv_reg_mr` (64 MB 18.1 against 20.8).
- Patch `0004-min-rnr-timer` is carried into the production build at `NCCL_MESH_MIN_RNR_TIMER=1`
  rather than staying on the shelf. Its isolated engine contribution is still unmeasured.
- **Retracted:** "the ceiling is ~13 GB/s against a 25 GB/s link, and the GPUDirect path needs a
  plugin redesign". The link is a pair of cables with one idle, and the device-pointer path was two
  lines. See [docs/11](docs/11-open-issues.md) §1.6. (The "50 GB/s pair" in this entry was itself
  retracted the next day — the ceiling is the cards' PCIe slots, ~30 GB/s per node; §1.7.)
- Reported upstream as a follow-up on the plugin's issue thread, with both patches offered.

## 2026-09-05 — production configuration 5: the MLA tuner cache stops charging us for measurement

- Image moved to `cuda-exl3` `9bf594c` ("Persist the MLA tuner cache across processes"), which adds
  `CUDA_EXL3_TUNE_CACHE`. The tuner's map had been process-local, so every boot re-tuned and every
  unseen batch shape bought a ~15 ms tune **while serving** — which is what polluted the first rounds
  of every A/B on this stack.
- Measured: **18 tune events before serving → 0**, none during a sweep, and round 1 stops being a
  penalty (cold cache C8 round 1 → round 3 −3.4 %; warm +2.7 %, i.e. unordered noise).
- **Protocol change:** five sweep rounds with two discarded → **three rounds, median of three**, on
  an image with the cache and a warm cache file. About 15 minutes saved per arm. Five rounds still
  stands everywhere else, including the boot that writes the cache.
- Speed unchanged by design: C1 54.5, C8 159.9 against 54.4 and 161.8, inside the spread. Gates
  10/10 · 12/12. KV pool 4,429,752.
- **What it cost:** a new image invalidates the fast-load sidecar, because the manifest records the
  image tag — the preflight refused the boot, correctly, and regenerating cost one **682 s** dump
  boot on all three nodes. That is now the standing price of every kernel-image change.
- The implementation is upstream's; ours was the measurement that asked for it. Credit in
  [CREDITS.md](CREDITS.md).
- Closes the open item in [docs/08](docs/08-fast-boot.md) §10, item 3, and
  [docs/11](docs/11-open-issues.md) §2.8. New page: [docs/12](docs/12-tuner-cache.md).

## 2026-09-05 — production configuration 4: fast boot

- Per-rank pre-sliced fast-load sidecar (`patches/tp3/harem_fastload.py` and friends): boot
  **618 s → 274 s**, weight load **426 s → 67 s**. Bit-identity proven twice — 1,475/1,475 tensors
  re-hashed against a manifest written from a full-checkpoint load, and a post-`process_weights` hash
  dump with no difference on any rank.
- `--enable-ep-weight-filter` plus a patch so the filter recognises EXL3's `.trellis` suffix: each
  rank now reads 96 of 288 experts instead of all of them.
- `--safetensors-load-strategy eager` and `--no-enable-flashinfer-autotune`.
- Page-cache remedy (`posix_fadvise(DONTNEED)` on checkpoint shards, `malloc_trim`) after a 4.1 % KV
  regression traced to reading 163 GB through the page cache on unified memory. Pool came back
  **above** the pre-change baseline: 4,484,848.
- Speed and quality unchanged, which was the intent. Gates 10/10 · 12/12 cold and warm.
- **Corrected:** `--no-enable-flashinfer-autotune` is worth about 3.5 s, not the 34 s it appears to
  be — most of that work moves into graph capture. See [docs/08](docs/08-fast-boot.md).

## 2026-09-05 — production configuration 3: the draft KV page

- Root cause found for a KV pool that was capped by a per-request **block counter**, not by memory:
  the DFlash2 draft's sliding-window group is given the backend's smallest kernel block (16 tokens),
  and because the port keeps that group independent it never reaches the unification step that would
  scale it up. The draft took **53 % of the blocks-per-request budget for 0.6 % of the memory**.
- `HAREM_SW_BLOCK_SIZE=256` (`patches/tp3/patch-swblock-tp3.py`, one anchor, env-gated):
  KV pool **2,428,769 → 4,413,223 (+82 %)**. 256 is the measured optimum; matching the target's
  3,328-token page makes the pool 7 % *worse* than doing nothing.
- Unpredicted bonus: the draft block table shrinks 16×, giving C4 +9 %, C8 +6 % and 20–30 % off TTFT.
- Cost: +9.2 % memory per block, and the draft group's prefix-cache matching unit coarsens from 16 to
  256 tokens.
- Memory ladder step to `gpu-memory-utilization 0.85` measured (+19 % pool, no speed change) and
  **rejected**: head node at 1.9 GiB free with 1.6 GB of swap in use, which breaks the 4 GiB rule.
  0.88 not attempted.

## 2026-09-05 — the NCCL mesh cliff

- Root cause for an all-reduce running at 0.6–1.9 GB/s between 128 KB and 4 MB while point-to-point
  over the same queue pairs stayed clean at 11–13 GB/s: the mesh plugin carries data with two-sided
  SEND/RECV, its only flow control is the receive-not-ready NAK, and it sets `min_rnr_timer = 12`
  (0.64 ms) where its own comment intends 0.01 ms (code 1). NCCL opens 64 channels on this fabric and
  one proxy thread services them round-robin, so the sender routinely outruns the receiver.
- **`NCCL_MAX_NCHANNELS=8`** adopted, environment only: model-free, the 512 KB decode all-reduce goes
  1,195 → 123 µs and one decode step's collectives 91.7 → 9.9 ms. In the engine, three boots and five
  rounds each: **C8 133.4 → 150.8 (+13.0 %)**, C1 +7.1 %, C4 +9.6 %, C6 +11.8 %; prefill, gates,
  acceptance and KV pool unchanged.
- Plugin patch (`patches/kernel/0004-min-rnr-timer.patch`) written, unit-tested and **not deployed** —
  with the channel cap its contribution is inside the noise.
- **Retracted:** an earlier reading that NCCL was choosing the LL protocol at 16 MB. Forcing LL there
  costs 20,114 µs against auto's 1,787 µs.
- **Protocol change:** five sweep rounds per arm, first two discarded. The tuner's warm-up window can
  be longer than two rounds and made the winning arm look 25–45 % worse on the first pass.

## 2026-09-05 — production configuration 2

- Image moved to `cuda-exl3` `f4987cf` ("do not fetch the MoE padding rows");
  `--max-num-batched-tokens` back to 2048 from 4096, recovering the KV pool (1,627,170 → 2,428,769)
  at a cost of ~5 % on fresh prefill.
- MoE input-transform fusion A/B (`61a17bc`): +1–4 % end to end over five rounds, but `f4987cf`
  reaches the same level another way and upstream dropped the fusion branch. Closed.
- k=7 versus k=5 A/B: k=5 raises the acceptance *rate* (63 → 73 %) and lowers accepted tokens per
  *step* (5.5 → 4.7); the second effect wins everywhere except prose and C4. **k=7 stays.**

## 2026-09-05 — upstream adopted, our patch retired

- Per-kernel comparison of four builds on identical shapes: our "zero the retired tile" choice is
  **10.7 % more expensive per MoE layer at M=2048** than upstream's "return, and let the combine skip
  those rows". Our `0002` patch retired in writing.
- Our one surviving kernel change kept and rebased: staging the combine's per-(token, k) facts in
  shared memory — combine −34 % at M=8, −36 % at M=64, −13 % at M=2048; end to end C6 +3.8 %,
  C8 +3.8 %.
- **Retracted:** the earlier report that an upstream build measured ~10 % slower end to end. It does
  not reproduce; what we had measured was **15.9 % boot-to-boot spread on C8 with nothing changed**.
- **Retracted:** the claim that the missing `n_rows` also costs the non-expert-parallel path. With no
  expert map the surplus tail is `-1` everywhere.
- `--max-num-batched-tokens 4096` adopted at this point: +9.5 % fresh prefill, −13 % mixed-load TTFT,
  −28.5 % KV pool. Later reverted, above.
- `NCCL_PROTO=Simple` rejected model-free without spending a boot: 2.8× worse at the C1 decode
  message, 4.4× at C8, no better at 16 MB.

## 2026-09-05 — the expert-parallel regression, root-caused

- TP=3 + expert parallel was **8–29 % slower than TP=2**. Cause: a one-line omission in the kernel's
  GEMM dispatch — the unsplit MoE launch never passed the live-row bound, so the surplus tail of
  `expert_ids`, which `expert_map` turns into a real local expert on one rank, ran a full GEMM over
  38 % of the grid at M=2048. Under expert parallelism every rank waits for that one.
- Per-rank MoE stage at M=2048: **18,401 → 10,107 µs (−45 %)**. End to end: C1 40.8 → 49.4,
  C4 59.5 → 99.6, C8 91.9 → 139.1, prefill 1,025 → 1,257. Reported upstream; fixed there in
  `a95e809`.
- Three of our own hypotheses refuted model-free with numbers (block padding, three-way collectives,
  the masking pass), and the GB10 top-k fallback we were pinned to turned out to be **faster** than
  the path it replaces below ~64K tokens of context.
- `gpu-memory-utilization` 0.85 → 0.80: cost nothing measurable and put all three nodes above the
  4 GiB free-memory rule for the first time.

## 2026-09-05 — TP=3 + expert parallel, first working boot

- Three nodes, 96 whole experts of 288 per rank, 64 → 66 head padding (22 per rank), vocabulary
  `padding_size` 192, shared expert padded to 2,112.
- Weights per node **81.53 → 54.86 GiB (−33 %)**; KV pool **825,000 → 2,947,441 (+257 %)**.
- Gates 10/10 · 12/12 cold **and after the full benchmark**; acceptance and accepted-tokens-per-step
  identical to TP=2 to three significant figures.
- Root causes fixed before the first boot by reading rather than running: the sidecar mount geometry
  (relative symlinks require identity mounts), and three divergent environment files.
- Root causes fixed during: the vision tower ignoring `--language-model-only` (fatal at TP=3, and it
  had been quietly carrying 1.05 GiB per rank at TP=2); a drafter head check that could not tell a
  legitimate pad from a disaster; an evidence log line being silently discarded because its logger
  sat outside the configured hierarchy; and KV pool arithmetic made visible.

## 2026-09-04 — DFlash2 ported to this stack (at TP=2)

- Speculative decoding brought into an image that had never run it, as a three-way git merge rather
  than a hand-copy. C1 aggregate **14.4 (no draft) → 30.5 (MTP k=3) → 42.9 (DFlash2 k=7)**;
  accepted tokens per step 3.31 → **5.37**.
- Two things the upstream delta did not cover: the target-side EAGLE3 interface, absent from this
  image entirely; and KV cache grouping, where the draft's sliding-window layers knock this model off
  its own grouping path. Fixed by giving the draft its own independent cache group — which is also
  what later capped the KV pool.
- Fail-closed additions: a quantized drafter is refused outright, head counts must divide the TP
  size, and the image is not produced if a ported symbol fails to resolve.

## 2026-09-04 — first EXL3 boot on this hardware

- `brandonmusic/GLM-5.3-Flash-tr3-4bpw` at revision `b20c49ba`, two nodes, TP=2, no expert
  parallelism. Boot 471 s after a first attempt died in the sparse-attention indexer's persistent
  top-k; the GB10 top-k overlay cleared it.
- Gates 10/10 · 12/12; MMLU sample 86.4 ±0.7.
