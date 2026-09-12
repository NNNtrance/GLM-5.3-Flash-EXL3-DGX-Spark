# Sparse-attention top-k 2048 → 8192 — the long-session tool-call corruption, bisected (12 September 2026)

**Applies to: TP=3.** The override was never set on the TP=2 track and nothing here was run there
`[not tested]`.

One value changed in production: `--hf-overrides` now carries `"index_topk": 8192` beside the
`quantization_config_file` entry it already had. Everything else is the 11 September configuration
untouched — DFlash2 at k=7, fp8 KV, FlashKDA, the prefix-hit and K-pool tail backports, the
fail-closed tool-call parser with its required-field arm, the two xgrammar backports,
`clear_thinking: true`, `reasoning_effort: low`.

It was adopted because it is the **only** thing we changed that moved the defect this page is about,
and the mechanism behind it is written up as [docs/14](../../docs/14-troubleshooting.md) §9.15. It is
not free: a 7K-prompt prefill reading fell **27 %**, and §7 prices that as honestly as one instrument
allows.

**Settings.** Three DGX Spark (GB10, sm\_121) nodes over the ConnectX-7 mesh, image
`exl3-zeus:754421f` (vLLM `0.1.dev20051+g487ecf187`), checkpoint `turboderp/GLM-5.3-Flash-exl3` branch
`4.05bpw` (full scope), **TP=3 + expert parallelism**, DFlash2 draft at k=7 with an fp8 draft cache,
KV dtype fp8, `gpu-memory-utilization` **0.88**, `max-model-len` **1,000,000**, `--block-size 256`,
`--max-num-batched-tokens 2048`, `--max-num-seqs 8`, `enforce_eager`, vision tower on,
`HAREM_KDA_FLASHKDA=1`, `HAREM_PREFIX_HIT=1`, `HAREM_KPOOL_TAIL_FIX=1`, `HAREM_GLM47_FAILCLOSED=1`
with the `required` arm on, `HAREM_XGRAMMAR_BACKPORT=1`, `HAREM_SM12_ITEMS=pdl,kpool`,
`HAREM_INDEXER_WS_MODE=bound`, `HAREM_DISABLE_PERSISTENT_TOPK=1`, `clear_thinking: true`,
`reasoning_effort: low`, `temperature 0.2` in the replay and temperature 0 in the gates. Measured
12 September 2026, 05:36–09:36 local, nothing else on the cluster except where §6 says otherwise.

The override is one JSON object with **no spaces** — `EXTRA_ARGS` is word-split by the launcher:

```
--hf-overrides {"quantization_config_file":"/var/tmp/glm-5.3-flash-turboderp-4.05bpw-tp3/quantization_config.json","index_topk":8192}
```

---

## 1. The symptom

In long agentic sessions — from roughly 32k prompt tokens onwards — tool-call **string arguments**
came back corrupted while the rest of the turn stayed coherent:

- single-character slips inside file paths, always in look-alike material: `projeler` → `rojeler`,
  `depo` → `dego`, `projeler` → `prodeler`;
- context text spliced onto the end of a path that started out correct;
- tool-call markup leaking into a value;
- occasionally a generation that ran away to `max_tokens`.

Short-context behaviour was flawless, and that is what made this hard to see: the 11 September soak
took **666/666** requests with zero errors and **493/493** well-formed tool calls over 420 agentic
turns ([`../soak-11sep.md`](../soak-11sep.md)), and every single-turn gate in this repository passes on
the unfixed engine. It was reported externally as issue #7 in this repository ("corruption 36k+").

## 2. The instrument, and what is wrong with it

A recorded 100-message agent session is replayed turn by turn against the live engine: 30 turns,
prompts rising 30k → 41k tokens, the real system prompt and 25 tool schemas attached, `temperature
0.2`, `reasoning_effort low`. Each turn's tool-call **path arguments** are scored against the paths
that exist in the fixture, and a turn counts as bad if any argument is wrong by any number of
characters. Two arms run on every configuration:

| arm | `cache_salt` | what it exercises |
|---|---|---|
| `fresh` | unique per turn | full prefill of the whole history, every turn |
| `session` | shared | the production path: prefix-cache reuse across turns |

**Three caveats, and they decide how the numbers may be read.**

1. **The corpus is a stress instrument, not a sample of healthy traffic.** It is a real session's
   history, so it contains that session's *own* earlier corrupted calls and the model's
   self-corrections of them. A model reading corrupted paths in its context imitates them — the same
   step-3 mechanism as §9.13 — so the **absolute** rates here are inflated against live traffic.
2. **Only arm-to-arm comparison is meaningful.** Every configuration below replays the same corpus
   with the same seedless sampling, so the columns are comparable to each other and to nothing else.
   A cleaned copy of the corpus (live errors repaired) was also tried and is **not** reported as a
   control: the upstream agent's own store truncates long arguments with a `...[truncated]` marker, so
   the cleaned history carries a *different* artefact the model copies just as readily. Fixing that
   needs request-level captures from the client, which we now have and had not yet when this ran.
3. **One run per configuration, 30 turns.** This counts corruption **events**; it does not establish a
   rate, and a difference of one or two turns is inside what a single run can produce.

## 3. The bisection

Bad turns out of 30, `fresh` / `session`. Each row is a three-node restart with exactly one thing
changed from production and then restored; the engine was never patched for this.

| arm | bad turns, `fresh` | bad turns, `session` | side effect | verdict |
|---|---|---|---|---|
| **production** (`index_topk` 2048, the checkpoint's own value) | **11** | **7** | — | the baseline |
| DFlash2 speculative decoding **off** | 12 | 10 | KV pool **+24 %** (no draft cache) | not the cause |
| FlashKDA **off** (Triton KDA prefill chain) | 8 | 7 | KV pool +1 % | not the cause |
| KV cache **bf16** instead of fp8 | 10 | 8 | KV pool **3,640,901** tokens, about half | not the cause |
| structural-tag grammar **off** (`strict` unset) | ≈9 (5 corrupt, plus 4 turns that fell to plain text) | — | — | not the cause |
| **`index_topk` 4096** | **6** | **4** | sidecar-less boot | about half the corruption |
| **`index_topk` 8192** | **2** | **3** | KV pool −1.3 % | adopted |
| `index_topk` 8192, final production boot (`load` mode) | **1** | — | — | one root slip, `/mnt/dego/…` |

Three readings of the three no-change rows: the defect survives the removal of speculative decoding,
of the fused prefill kernel and of fp8 KV quantization **independently**, it survives halving the KV
pool, and it survives turning the grammar off. The grammar row has a blind spot worth naming — with
`strict` unset a malformed call can be refused by the fail-closed parser and surface as content, so
four of those nine turns are "no tool call at all" rather than "a corrupted one", which is why it is
written as an approximation.

The `index_topk` rows are the only monotone column in the table, and they move in the direction the
mechanism predicts: widen the selection, lose the corruption.

## 4. Two limits found on the way, both hard

| attempt | what happens |
|---|---|
| `index_topk` **16384** and **65536** | the engine never serves: `CUDA error: invalid argument` during worker init. The kernel limit is therefore **between 8192 and 16384**, and a fully dense comparison is not available on this image |
| `index_topk: null` (dense MLA) | refused at backend selection: `Selected backend CUSTOM is not valid: 'non-sparse not supported'` — the EXL3 attention backend implements the **sparse-MLA** path only |

The first attempt at 65536 also failed for a second, unrelated reason that is worth having in writing:
with `FASTLOAD_MODE=load` the boot was refused with `harem-fastload: sidecar was written for a
different model configuration`, because `hf_overrides` is part of the sidecar identity
([docs/08](../../docs/08-fast-boot.md) §4). Every exploratory arm after that ran with `FASTLOAD_MODE`
empty — a full-checkpoint load, about 5 minutes — which is the rule for any experiment that changes
the identity, patch files included.

## 5. Interpretation

GLM-5.3-Flash's attention is DeepSeek-style sparse MLA: a lightning indexer selects
`index_topk` = 2048 key positions, pooled `index_kpool` = 4 deep, so the per-row top-k the kernel
actually runs is `index_topk / index_kpool` — 512 before, **2048** now. At 2048 selected positions a
token deep in a 35k-token agentic history attends to under 6 % of it, and the material that has to
come back **byte-exact** is the worst possible case for a lossy selection: near-duplicate path
strings that differ in one character and recur dozens of times in the history. Widening the selection
removes the corruption monotonically; nothing else we removed touched it.

**What that does not settle**, and we are not going to imply otherwise: whether the residual is the
model's own design limit at long context, or the **precision** of this engine's indexer — the indexer
runs in fp8 and the K-pool compresses four positions into one score, either of which could cost the
selection the one position that carries the literal. Separating those needs an indexer-precision arm
and a dense reference, and the dense reference does not exist on this image (§4). It is filed as
[docs/11](../../docs/11-open-issues.md) §2.36 and it is a question for upstream as much as for us.

## 6. The gates on the production boot

`FASTLOAD_MODE=load` on the new sidecar, 09:25 local. Boot to `/health` 200: **170 s**, sidecar
restore at 953 MB/s.

| Gate | Result |
|---|---|
| KV pool at `max_model_len` 1,000,000 | **7,033,057** tokens — inside the 7.03–7.10 M production band, about 0.3 % under the 11 September boot |
| [`scripts/correctness-probe.py`](../../scripts/correctness-probe.py), both fields | **10/10** (content only **9/9**, requests with empty content **0**) |
| [`scripts/code-exam.py`](../../scripts/code-exam.py) | **12/12** |
| [`needle-lite6.py`](../../tracks/tp3/patches/prefix-hit-and-kpool-tail/needle-lite6.py), six depths | **6/6** |
| [`scripts/toolcall-gate.py`](../../scripts/toolcall-gate.py) through [`strict-proxy.py`](../../scripts/strict-proxy.py), 2 × 30 rounds, prompts to 31k | **72 tool calls, 72 well-formed, 0 rejected, 0 out-of-schema, 0 JSON errors, 0 empty turns, 0 corrupt turns**, 1 identical repeat |
| the same gate **without** `strict` | **7 of 61** calls rejected by the fail-closed parser — tool-call markup left in the content — first at 11.8k and 20.8k tokens |
| the replay of §3 on this boot, `fresh` arm | **1/30** |

The last two rows are the operational result of this page after the `index_topk` one. **The
structural-tag grammar stays required in production**: on the same boot, with the same corpus, the
production client path (`strict: true` on every tool, which is how the grammar is armed at
`tool_choice="auto"` — [`toolcall-gate-11sep-strict.md`](toolcall-gate-11sep-strict.md)) produced a
clean 72/72, while the unarmed path left markup in seven calls for the parser to refuse. Those seven
are *refusals*, not derailments — that is the fail-closed patch doing its job — but they are seven
turns a user would have seen as ugly content. The two rows ran on the same engine minutes apart and
are the closest thing here to a controlled pair.

The dump boot that wrote the sidecar read a KV pool of **6,804,407** tokens. That number must not be
quoted as a result: a dump boot holds the serialisation buffers as well, and every pool figure in this
repository's dump rows is low for the same reason
([`../configs/kv-pool-progression.csv`](../configs/kv-pool-progression.csv)).

## 7. What it cost

| | before (11 September production) | after | |
|---|---|---|---|
| `scripts/prefill-7k.py`, 7K prompt | **1,868** tok/s | **1,366** tok/s | **−27 %** |
| per-turn prefill in the 30–41k replay, `fresh` arm, median | 22.7 s | 25.3 s | **≈ +11 %** |
| decode, `session` arm, median per turn | 3.4 s | 3.4 s | unchanged |
| KV pool | 7,052,341 | 7,033,057 | −0.3 %, in band |
| boot, `load` mode | 170–190 s | **170 s** | unchanged |
| sidecar | 53 GB per rank | 53 GB per rank, **new directory** | one dump boot, ~6 min |

**Read the −27 % carefully, because two instruments are involved.** `scripts/prefill-7k.py` reports a
single 7K-prompt figure and the 1,868 tok/s it is compared against is the **sustained** stopwatch
number from the FlashKDA promotion ([`flashkda-ab-10sep.md`](flashkda-ab-10sep.md)); the two are not
interchangeable and [docs/10](../../docs/10-results-and-roofline.md) §1 prints all four of this
stack's prefill figures with their methods. What is safe to say is that **short-prompt prefill got
materially slower in one direction**, and that the controlled per-turn number on a realistic 30–41k
history is **about +11 %**, which is the figure to plan agentic sessions with. The sweep
(`bench/prefill-fresh.py`, C1–C8, TTFT, draft acceptance) was **not** re-run `[not tested]`, so
`audit/README.md`'s prefill band is now pre-12-September and says so.

Why prefill and not decode, as an inference rather than a measurement `[not measured]`: the per-row
top-k the indexer runs goes 512 → 2048 (§5), and a prefill chunk pays that on 1,792 query rows at
once while a decode step pays it on one. A profile would settle it and none was taken.

**Nothing else moved.** Decode, KV pool and boot time are unchanged, and no gate regressed.

## 8. Rollback

Two steps, and the first is enough:

1. **The environment file.** Restore `.env.tp3.bak-12eyl-pre-topk8192` on all three nodes and restart
   all three together. It carries both halves of the change — the `EXTRA_ARGS` without `index_topk`
   **and** `FASTLOAD_DIR` pointing back at `/var/tmp/glm53-exl3-xgrammar`, which is **kept on disk**
   for exactly this. No dump boot.
2. The new sidecar directory (`/var/tmp/glm53-exl3-topk8192-r{0,1,2}`, 53 GB per rank) can then be
   removed at leisure. Check free disk **before** the next dump rather than after.

`index_topk` cannot be rolled back with an environment knob, unlike every `HAREM_*` gate in this
stack: it is a launcher argument and it is part of the fast-load identity, so changing it in either
direction is an env-file change plus a sidecar that matches. Never restart one node alone — it kills
the fabric port on its peers.

## 9. What this run does not establish

- **Absolute rates.** The replay corpus contains the original session's own corruption, so "11 bad
  turns in 30" is a property of this instrument, not of the engine in service. The 11 September soak's
  0 corrupt turns in 420 is the other end of the same truth: neither number is the rate.
- **One run per arm, 30 turns, one effort level** (`low`), one temperature (0.2), one corpus. The
  2 / 3 row and the 6 / 4 row are two turns apart in places where a single run can produce two turns
  of difference on its own; what carries the conclusion is the **monotone trend across three values**
  plus a fourth reading (1/30) on the production boot.
- **The remaining 1/30 is not zero.** A root-directory slip (`/mnt/dego/…`) survived at 8192 on the
  production boot. This is a reduction, not a proof of correctness, and a client that builds paths from
  model output still needs to validate them.
- **No dense reference exists on this image** (§4), so "more selection is better" is demonstrated over
  2048 → 4096 → 8192 and extrapolated nowhere.
- **The cause of the residual is open** — model design limit against indexer precision (§5,
  [docs/11](../../docs/11-open-issues.md) §2.36). Nothing here separates them.
- **The cost is one instrument deep.** No concurrency sweep, no TTFT series, no acceptance-rate
  reading, no telemetry `[not tested]`. A full speed re-characterisation of production at 8192 is owed.
- **TP=2 is untouched** `[not tested]`, and so is the question of whether a two-rank stack shows the
  same depth ladder at all.

Raw replay output, the per-arm drivers and the gate logs are **not in this repository** (they carry
request bodies from a live agent session, paths and prompts included); the per-turn scores and the
tables above are the summary, and `scripts/toolcall-gate.py` plus `scripts/strict-proxy.py` reproduce
the gate half of it on any stack.
