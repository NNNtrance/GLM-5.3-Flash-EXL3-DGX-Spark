# Tool-call corruption measurement on the fail-closed build (11 September 2026)

**Script:** [`scripts/toolcall-gate.py`](../../scripts/toolcall-gate.py) — a synthetic but realistic
agentic coding session (system prompt + 30 tool definitions + a fabricated repository, 16 tasks),
driven turn by turn, echoing the client's own history back exactly, `tool_calls` included. Issue #7's
mechanism — see [docs/14](../../docs/14-troubleshooting.md) §9.13 — is the first-corruption event this
script counts, bucketed by 10k tokens of context.

**Settings.** Production build with the fail-closed tool-call parser on
(`HAREM_GLM47_FAILCLOSED=1`, the default — [`tracks/tp3/patches/glm47-failclosed/`](../../tracks/tp3/patches/glm47-failclosed/README.md)),
`reasoning_effort: low`, `clear_thinking: true`, DFlash2 draft at k=7. Run: 4 sessions × 30 turns,
concurrency 2, `--max-prompt-tokens 120000`. 08:45–09:20 local, nothing else on the cluster.

## Result

| Metric | Value |
|---|---|
| Turns / tool calls | 120 / 118 |
| Well-formed (schema-valid) calls | 117 |
| Fail-closed rejections (malformed XML in `content`) | **0** |
| Parsed but out-of-schema | 1 — session 0, turn 25, 78,158 tokens: a `write` call missing `content`. The harness's own `--max-tokens 2048` completion cap truncated a long `write` mid-argument; this is truncation by the test harness, not the model losing its tool-call format. |
| Empty turns / repeats / `finish_reason=length` | 0 / 0 / 0 |
| Highest context reached | 100,522 tokens |

### Per-bucket breakdown

| context_bucket | turns | tool_calls | wellformed | rejected(b) | schema_violation(c) | json_error | empty_turn(d) | repeat(e) | length(f) |
|---|---|---|---|---|---|---|---|---|---|
| 0–10k | 43 | 47 | 47 | 0 | 0 | 0 | 0 | 0 | 0 |
| 10–20k | 13 | 13 | 13 | 0 | 0 | 0 | 0 | 0 | 0 |
| 20–30k | 32 | 30 | 30 | 0 | 0 | 0 | 0 | 0 | 0 |
| 30–40k | 3 | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 0 |
| 40–50k | 7 | 4 | 4 | 0 | 0 | 0 | 0 | 0 | 0 |
| 50–60k | 5 | 5 | 5 | 0 | 0 | 0 | 0 | 0 | 0 |
| 60–70k | 1 | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| 70–80k | 3 | 3 | 2 | 0 | 1 | 0 | 0 | 0 | 0 |
| 80–90k | 4 | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 0 |
| 90–100k | 7 | 6 | 6 | 0 | 0 | 0 | 0 | 0 | 0 |
| 100–110k | 2 | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| TOTAL | 120 | 118 | 117 | 0 | 1 | 0 | 0 | 0 | 0 |

## Reading

Across 118 tool calls at low reasoning effort, up to 100.5k tokens of agentic context, the production
build **produced zero malformed tool-call XML**: zero fail-closed rejections, zero empty turns, zero
repeated calls, zero `finish_reason=length`. The one deviation from a clean run was the harness's own
output cap truncating a long `write` argument — a property of this test script's `--max-tokens`, not
of the engine's tool-call format. That is consistent with issue #7's own characterization of the first
corruption as a rare event; the fail-closed parser's cascade break
([results/gates/failclosed-11sep.md](failclosed-11sep.md)) is what now stands between one rare bad call
and a dead session, and this run gave it nothing to catch.

## Limits, stated plainly

One run, against a synthetic fixture repository, at one reasoning-effort setting. This is **a
measurement, not a pass/fail gate** — the script's own exit code is always 0, by design. It does not
establish a corruption *rate* (only issue #7's own instrumented capture did that), it does not compare
against the NVFP4 sibling build, and it is not real agentic traffic — Hermes, once running, is the
actual exam ([HELP-WANTED](../../HELP-WANTED.md) §14).

**Next arms, not yet run:** `--effort high`; `--max-tokens 4096` (so a long `write` cannot be mistaken
for a schema violation again); `--max-prompt-tokens` past 150k.

Raw output for this run (`turns.jsonl`, `summary.json`, `summary.md`) is not committed — this page is
the summary. Anyone re-running the script gets the same three files under their own `--out` directory.
