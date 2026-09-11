# `patches/glm47-failclosed` — refusing a malformed tool call, and the cascade it ends

**In the recipe since 11 September 2026.** The glm47 tool-call parser in the pinned vLLM
(`487ecf187`) *salvages* malformed tool-call XML instead of refusing it. In a long agentic session
that salvage is the first link of a chain that ends in an empty turn — a turn the client reports as
complete, with zero bytes in it. This patch makes the parser **fail closed**: a call that does not
validate is surfaced as plain text, never as a tool call. One file, six anchors, one environment
knob, **default on**.

| | Knob | What it does | Status |
|---|---|---|---|
| **Fail-closed glm47 parser** | `HAREM_GLM47_FAILCLOSED=1`, or unset | a tool call is buffered until it closes, then validated against the request's tools and their schemas; an invalid call becomes `content`, not a `tool_call` | **In the recipe**, 11 September |
| | `0` | **upstream behaviour, byte for byte** — every override returns `super()` immediately and `stream_arg_deltas` goes back to `True` | the control arm, and the fastest rollback |
| **Required-field check** | `HAREM_GLM47_REQUIRED=1`, or unset | sub-gate of the above: a call missing any key in the tool's `parameters.required` is rejected too — presence only, an empty string value passes | **Not in production**, written 11 September, pending A/B + gates ([docs/14](../../../../docs/14-troubleshooting.md) §9.13) |
| | `0` | keys are shape- and `properties`-checked only, as they were before | the control arm for that A/B |

**The default is ON, and that is deliberate** — it is the only patch in this tree that defaults on.
Everywhere else an unset knob means *upstream*, because upstream is the safe side. Here upstream is
the bug, so an unset knob on a fresh node has to be the safe side instead.

**The mechanism is not ours.** It was captured live, replayed and root-caused by **YuXiaoPan** in
issue #7, down to the design of the fix; the implementation here is ours. §8.

The production gates after adoption are
[`results/gates/failclosed-11sep.md`](../../../../results/gates/failclosed-11sep.md). The question
this patch does **not** close — how often the model writes the *first* malformed call, and whether
the 4bpw EXL3 checkpoint raises that rate — is
[docs/11](../../../../docs/11-open-issues.md) §2.30, and §7 below says why the patch still helps
without it.

---

## What is here

| File | What it does |
|---|---|
| [`patch-glm47-failclosed-tp3.py`](patch-glm47-failclosed-tp3.py) | The patch. Six exact-text anchors in `vllm/parser/glm47_moe.py`, each required **exactly once**: the stdlib import block, the vLLM import block (where the module logger and the gate land), the parser class's config, the `TOOL_CALL_END` handler, the finish path, and the arg converter's call site. A missing or duplicated anchor exits non-zero instead of guessing; re-running is a no-op |
| [`test_failclosed.py`](test_failclosed.py) | **54 checks, CPU only, no model, no engine** — it imports the patched parser and drives it directly. The corruption vectors are the ones captured live in issue #7, not invented ones. 37 of the 54 are the set that passed before production adoption on 11 September; the other 17 cover the required-field check, which is **not in production yet**. §3 |

---

## 1. What upstream does, and what the patch changes

Target file: `vllm/parser/glm47_moe.py` in vLLM `487ecf187` (image `exl3-zeus:754421f`). Four lines
of upstream, between them, are the whole problem:

- **`_glm47_arg_converter` — `glm47_moe.py:56-70`.** It regex-scrapes every
  `<arg_key>K</arg_key><arg_value>V</arg_value>` pair out of the argument body into a dict and
  `.strip()`s the key. **Nothing else is checked.** A key like
  `print_code_snapshot</arg_value><arg_key>description` — a real one, echoed back in the captured
  history of issue #7 — becomes a real JSON argument name.
- **`validate_tool_names=True` — `glm47_moe.py:171`.** The **name** is validated and the arguments
  are not; and `ParserEngine._is_valid_tool_name`
  (`vllm/parser/engine/parser_engine.py:392-397`) returns `True` unconditionally when the request
  carries no tools at all.
- **`stream_arg_deltas=True` — `glm47_moe.py:169`.** Arguments are streamed out piece by piece, so a
  call that later turns out to be garbage has already reached the client.
- **`_build_extracted_result` — `parser_engine.py:1014-1065`.** A call that never closes produces no
  `TOOL_CALL_END`, so `_handle_tool_end` never runs — and this still turns the dangling slot into a
  tool call.

**The cascade those four make** (issue #7, every step of it replayed there): the model emits one
malformed call at long agentic context → the parser salvages it into a syntactically valid
`tool_calls` entry with a garbage argument key → the client executes or schema-rejects it and echoes
the assistant message, garbage included, back into the history → the 4 September chat template
re-renders it in canonical `<arg_key>` syntax → **the model is now reading its own corruption as an
in-context example** and imitates it. Each round is worse until a call is malformed enough that
nothing parses at all, and the whole turn is swallowed: empty stream, `finish_reason=stop` at
`<|observation|>`, client shows "turn complete". That shape is why the symptom only ever appears in
long, warm, multi-turn *agentic* sessions, gets worse within a session, and is invisible to every
single-turn gate in this repository.

**What the patch does**, when `HAREM_GLM47_FAILCLOSED` != `"0"`:

1. **Nothing partial leaves the parser.** `stream_arg_deltas` goes to `False` and `_emit_name_delta`
   is suppressed while a call is open, so a tool call is buffered and emitted as **one** delta when
   it closes. A half-formed call can no longer half-stream into a client.
2. **At `TOOL_CALL_END` the complete call is validated.** The name is identifier-shaped
   (`[A-Za-z_][A-Za-z0-9_.-]*`) and, when the request carries tools, present in them; every argument
   key is identifier-shaped and, when the tool carries a schema, present in its `properties`; and a
   non-empty argument body the converter could extract **nothing** usable from counts as truncated.
3. **A rejected call becomes content.** The raw `<tool_call>…</tool_call>` text is surfaced as plain
   `content` and the slot is cleared, so neither the streaming path nor `_build_extracted_result` can
   emit it as a tool call. The tool-call id is not spent either. The user sees the garbage as text;
   **nothing parseable-as-a-tool-call re-enters the history**, and the chain is broken at its first
   link.
4. **A call that never closes is caught twice.** The engine's `finish()`
   (`vllm/parser/engine/streaming_parser_engine.py:262-275`) synthesises a `TOOL_CALL_END` for an
   open call, so the validation above already rejects it as truncated. On top of that there is a
   **sweep**: any slot still open at finish is rejected. The second is depth — it is tested
   separately because defence that is never exercised is not defence.

One line per process at import, and it is the boot gate:

```
[HAREM-GLM47-FAILCLOSED] failclosed=True (HAREM_GLM47_FAILCLOSED='')
```

plus one warning per rejected call — the tool name truncated to 48 characters and the reason. **The
argument body is never logged**: it is user content. A boot log without the import line is a boot
where this patch did not run, which is the `--in-place` trap of §4.

## 2. The knob, and what `=0` is worth

```
HAREM_GLM47_FAILCLOSED unset or 1   -> fail closed (production)
HAREM_GLM47_FAILCLOSED=0            -> upstream behaviour, byte for byte
```

`=0` is not a best-effort disable. Every override returns `super()` on its first line and
`stream_arg_deltas` is restored to `True`, so the `=0` path executes upstream's code and nothing
else. `test_failclosed.py` asserts this by **reproducing the bug**: with `=0` the captured garbage
key comes back out as a tool-call argument, exactly as upstream leaves it. A claimed byte-for-byte
fallback that is not tested is a claim, not a fallback.

The knob is read at construction, not per token, so it costs one environment read per request
parser.

## 3. The tests, and how to run them

`test_failclosed.py` is CPU-only and model-free — it patches a copy of the parser inside a throwaway
container and drives it directly, so it needs no engine and cannot disturb one. Run it **inside the
production image**, which is the only place the parser it patches exists:

```bash
# on a node, with the patch and the test in /tmp/issue7
docker run --rm -v /tmp/issue7:/w --entrypoint bash exl3-zeus:754421f -c "
  python3 /w/patch-glm47-failclosed-tp3.py \
      --root /usr/local/lib/python3.12/dist-packages --in-place &&
  python3 /w/test_failclosed.py"
```

`--rm` means the in-place patch dies with the container; nothing on the node changes. **37/37 passed**
on 11 September 2026 before adoption, and the suite is now **54/54** — the 17 required-field checks added
later the same day were developed outside the image against a stub harness and then **run in the image on
11 September, 19:16, all passing**; the required-field arm went into production in the 19:18 boot
alongside the xgrammar backports ([`results/gates/xgrammar-backports-11sep.md`](../../../results/gates/xgrammar-backports-11sep.md)).

What the checks cover: the two corrupted `<arg_key>` sequences captured in issue #7 — one of them
carrying Anthropic-style `</invoke>` and `<parameter name=` fragments, i.e. the model having lost its
format rather than mistyped it — the unclosed call that produced the empty turn, a hallucinated tool
name, a key that parses cleanly but is outside the tool's schema, a valid call, a valid call with no
arguments, and plain text with no call in it. Each runs through **both** the streaming and the
non-streaming path, because they are different code. Then the sweep branch, and then the `=0` arm
described above.

**Note what is not in here.** These are unit tests against captured vectors. The *rate* at which the
model produces a first bad call is a serving measurement on a long warm session, and that is §7.

## 4. Deployment — the prelude line, and the sidecar it invalidates

The line, in [`../tp3full-prelude.sh`](../tp3full-prelude.sh), immediately after the FlashKDA block:

```bash
run python3 "$TP3_DIR/patch-glm47-failclosed-tp3.py" \
    --root "$(dirname "$VLLM_PY")" --in-place
```

**Two things about that line, the same two FlashKDA cost a boot each:**

- **`--root` is the dist-packages root, not `$VLLM_PY`.** This script's `REL` starts with `vllm/`,
  like `patch-flashkda-tp3.py` and unlike every other patch script in this directory, so
  `--root "$VLLM_PY"` resolves to `…/vllm/vllm/parser/…` and raises `FileNotFoundError`.
- **`--in-place` is required.** Without it the script is a **dry run**: it prints `dry run OK`, exits
  0 and patches nothing — a perfectly healthy boot that quietly keeps serving the salvage path. The
  `[HAREM-GLM47-FAILCLOSED]` line in the boot log is how you tell the two apart, and it is the gate.

Note the **flat path**: in this repository the script sits in a subdirectory for a reader's benefit,
but on a node every patch script sits directly in `$TP3_DIR`, because the fast-load sidecar identity
hashes `glob($TP3_DIR/patch-*.py)` and a script in a subdirectory is neither hashed nor found
([docs/08](../../../../docs/08-fast-boot.md) §4).

**The sidecar, and do not skip this.** That identity hash is exact equality over the patch set and
the full text of the prelude, so **adding this file invalidates every fast-load sidecar on every
node** and the preflight refuses a `FASTLOAD_MODE=load` boot. The sequence that works:

1. Point `FASTLOAD_DIR` at a **new** directory — never the live one; the identity check is exact
   equality, so the old and new sidecars can never be the same one.
2. Boot once with `FASTLOAD_MODE=dump`, which writes that sidecar. This boot is a full load: **380 s**
   on our three nodes, 11 September.
3. Switch `FASTLOAD_MODE` back to `load` and restart.

The chicken-and-egg inside step 2 — the `ExecStartPre` gate demanding a sidecar in the very mode
whose job is to create it — is [docs/14](../../../../docs/14-troubleshooting.md) §10.6, and it cost
18 minutes of downtime the first time. Reading it before the dump boot is cheaper than rediscovering
it.

**One note on the file itself.** Its docstring header still reads "NOT IN PRODUCTION yet", from the
morning it was written. It is left **byte-identical to the copy that booted production**, because the
text of this file is part of the identity hash above and editing a docstring would cost another
sidecar-less boot for nothing ([CHANGELOG](../../../../CHANGELOG.md), 11 September, the same lesson on
`patch-kpooltail-tp3.py`). The status line to trust is the one at the top of this page.

## 5. Rollback

- **Fastest, and it keeps the sidecar.** Set `HAREM_GLM47_FAILCLOSED=0` and restart the engine. The
  file stays in place, so the identity hash does not move and there is no re-dump. This is why the
  knob exists at all.
- **Full removal.** Delete the prelude line and the script. That changes the identity again, so the
  first boot after it is sidecar-less, then a new sidecar — the same three steps as §4.

## 6. Known limitations

Three, all on the error path, none of them silent:

- **Ordering inside one delta.** Plain text that arrives *after* a rejected call can land *before*
  the raw text in the same delta packet, because the base class prepends `_deferred_content`. Wrong
  order, no content lost, and only when a call has already been rejected.
- **A `</tool_call>` the model never wrote** is appended to the raw text of an unclosed call, because
  the engine's synthesised `TOOL_CALL_END` carries an empty value. It makes the surfaced text
  readable and it is not misleading, but it is ours, not the model's.
- **A request with no tools gets name-shape validation only.** There is nothing to check the name
  against, so upstream's behaviour is kept rather than guessed at. Argument keys are still
  shape-checked; schema checking needs a schema.

## 7. The open question this does not answer

The *first* corruption is a low-probability format derailment at ~60k+ tokens of agentic context at
temperature 1.0. **Whether the 4bpw EXL3 checkpoint raises that rate relative to the NVFP4 sibling
is not known** — it is precisely the checkpoint-versus-build question of
[docs/11](../../../../docs/11-open-issues.md) §2.30, and issue #7's data cannot separate them either.
This patch does not move that rate by a single token: it is a parser change, downstream of
generation.

What it changes is the **consequence**. Before it, one rare bad call seeded a loop that ended the
session, so the rate had to be near zero to be survivable. After it, a bad call costs one visibly
ugly turn and nothing more — so a low rate becomes an adequate defence instead of a required one.
Measuring the rate is still worth doing, and the design is the one issue #7 used: the same captured
request body, the same temperature, two checkpoints, corrupted calls per turn. A multi-turn gate that
would catch this class of failure in the first place is
[HELP-WANTED](../../../../HELP-WANTED.md) §14.

## 8. Credits

**The mechanism, the evidence and the design of the fix are YuXiaoPan's**
(issue #7). They caught
the empty turn live behind a logging proxy, then replayed the captured request body raw —
`/tokenize` on the exact messages and tools, then `/v1/completions` on those token ids, no chat
template and no parsers — which is what turned "the model sometimes stops" into a reproducible
5/5 corruption at a known state. The same replay eliminated the prefix cache (salted system message,
0 % hits, still corrupted), eliminated sampling temperature (corruption deterministic near zero), and
showed the clean-context model is fine at the same depth (8/8 well-formed at 59.6k tokens after
truncating the history to just before the first corrupted call in it). Four negatives and a positive,
from one instrument — that is the part worth copying.

The implementation in this directory, its tests and its gates are ours. We are glad to take their
own patch, tests or the captured transcript as a replay fixture, which is the one thing missing here:
these tests use the corruption vectors quoted in the issue, not the full session that produced them.
