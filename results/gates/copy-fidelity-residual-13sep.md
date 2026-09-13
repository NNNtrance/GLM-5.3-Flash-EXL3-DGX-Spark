# The residual long-context copy slip at `index_topk` 8192: about 3 % per deep copy, and what it is not (13 September 2026)

**Applies to: TP=3** `[not tested at TP=2]`. Continues
[`index-topk-8192-12sep.md`](index-topk-8192-12sep.md) and [docs/14](../../docs/14-troubleshooting.md)
§9.15. Settings as on that page; the production boot (`index_topk` 8192, DFlash2 k=7, fp8 KV, strict
tool schemas, `clear_thinking: true`, effort low) unless a row says otherwise.

**Instrument.** A private depth probe (real 100-message agent history, real-density tool output as
filler, twelve absolute paths planted once each across the first 80 % of a sized context, twelve turns
that each demand one of them back byte-exact in a `read_file` call). The public
`scripts/longctx-copy-fidelity.py` is the same method without the private corpus and does **not** yet
reproduce the defect ([docs/09](../../docs/09-measurement-protocol.md) §9.1), so the rates below are
`[measured-here]` with raw JSON in our results but not reproducible from this repository alone.

## 1. The rate, and what does not move it

| arm (all at 100k–108k prompt tokens, 12 turns per stream) | bad turns | shape |
|---|---|---|
| production, 4 concurrent streams, temperature 0.2 | 2 / 48 | 1 slip (`projeler` → `rojeler`), 1 splice |
| DFlash2 off, 4 concurrent | 1 / 48 | 1 dropped directory segment |
| production, 6 concurrent (repeat) | 2 / 72 | 1 slip, 1 splice, same targets |
| production, **single stream** (3 × 12) | 2 / 36 | 2 slips, both at turn 11, same target |
| **temperature 0**, single stream (3 × 12) + 4 concurrent | 3 / 84 | 2 slips, 1 splice |

Single-stream and concurrent arms read the same; DFlash2 on and off read the same (12 September's
30-turn replay said so too); temperature 0 reads the same as 0.2. The residual is therefore **a
property of the model + engine at this selection budget** — about 3 % of turns in which a once-mentioned,
near-duplicate-rich string must be copied from ~100k tokens back — and not of sampling, concurrency or
speculative decoding. A dense-attention arm cannot be run on this image (§9.15: `index_topk: null` is
refused at backend selection, 16,384 fails in worker init), so the one engine-side lever left is a
selector kernel accepting a budget above 8,192, which is a kernel project and not a flag.

## 2. Numbers are more robust than paths

The same probe with twelve **numeric tuning values** planted once each in small config files (each
beside three near-duplicate keys with near-duplicate values: `1.175` next to `1.157` and `1.715`),
twelve turns that each require the exact value in a `patch` call, the source file "removed" so a
re-read returns not-found, three streams at 100k, temperature 0.2:

| class | turns |
|---|---|
| exact value | **31 / 36** |
| digit dropped / swapped / changed | **0** |
| a neighbouring file's value used instead | 1 |
| patch written without the number | 2 |
| empty first turn after the 100k prefill (no tool call, no content) | 2 |

`[measured-here]`, 13 September 07:45–07:52. No character-level slip in a number; the failure shapes are
"wrong neighbour" and "did not write it", which a reviewer comparing against the source catches. The
first-turn empty response after a fresh 100k prefill appeared in two of three streams and is a separate
symptom (§9.13's empty-turn family), not a copy error.

## 3. What absorbs the residual: repair the argument before the tool runs

Because the slip lands almost always in a tool-call **path argument** and the correct path exists on
disk, a `tool_request` middleware in our agent harness repairs it before execution: a known-root snap
(`/…/rojeler/…` → `/…/projeler/…`), a segment-by-segment walk accepting one confident directory-listing
match per level (case/diacritic fold, unique prefix completion, one adjacent transposition or one
dropped/extra character, or a difflib ratio ≥ 0.85 with a margin over the runner-up), a cut at
spliced text or leaked markup when the prefix is a real path, one-level re-insertion of a dropped
directory segment, and — for absolute paths inside shell commands — the same under stricter rules (only
under configured roots, the file name never changed, nothing after a heredoc marker, no command that
carries a destructive word). Every corrupted argument the probes recorded (five shapes) is repaired
against the real tree, 7 / 7; the offline suite of 51 checks has no false repair; the first live repair
was observed the same morning. Every repair is appended to a JSON-lines log, which is the live counter
of the residual from here on. The rule the middleware is built on: a failed tool call is recoverable, a
silent write to the wrong real file is not, so anything ambiguous is left exactly as the model sent it.

The middleware is harness-specific and lives outside this repository; the design above is the part to
copy.
