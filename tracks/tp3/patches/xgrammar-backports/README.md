# `patch-xgrammar-backports-tp3.py` — upstream vLLM #52805 + #53046, backported

Backports two upstream **vllm-project/vllm** structured-output bugfixes into our
pinned build, which predates both merges.

| | upstream PR | merge commit | merged | file it touches |
|---|---|---|---|---|
| 1 | [#52805](https://github.com/vllm-project/vllm/pull/52805) — *Stop XGrammar token batches at termination* | `12f64b39d29282437e35be9aa5db432fb2a1a6e6` | 2026-08-18 | `vllm/v1/structured_output/backend_xgrammar.py` |
| 2 | [#53046](https://github.com/vllm-project/vllm/pull/53046) — *Avoid spurious FSM errors after speculative reasoning end* | `c6e19b3be24338759a443e03c8325d76da9ee202` | 2026-08-21 | `vllm/v1/structured_output/__init__.py` |

Our build: vLLM `0.1.dev20051+g487ecf187`, image `exl3-zeus:754421f`. Both target
files are byte-identical to the pre-PR upstream state, so the hunks map 1:1 and
the anchors are exact.

## Provenance and licence

Both hunks come from the **upstream Apache-2.0 repository**, fetched with

```bash
gh api repos/vllm-project/vllm/pulls/52805/files --jq '.[].patch'
gh api repos/vllm-project/vllm/pulls/53046/files --jq '.[].patch'
```

No third-party fork was used as a source. The upstream test
(`tests/v1/spec_decode/test_mtp_structured_output.py`) is **not** backported —
it needs the upstream test tree, which the image does not ship; `test_xgrammar_backports.py`
here covers the same behaviour against the installed tree.

## Why we need it

Production runs GLM-5.3-Flash with speculative decoding (DFlash2, k=7) and a
reasoning parser (`deepseek_r1`). Whenever structured output is active —
Hindsight runs `strict_schema`, and the 11 Sep strict tool-call A/B put grammar
on the tool-call path as well — the engine logged bursts of

```
ERROR [backend_xgrammar.py:166] Failed to advance FSM for request <id> for tokens <t>. Please file an issue.
[.../grammar_matcher.cc:612: Warning: The matcher has terminated after accepting the stop token, but is trying to accept new token with id <t>.
```

Measured on the API node over one 11-hour boot (11 Sep, 08:50–19:06 local, the fail-closed
production boot): **776 FSM `ERROR` / 872 `grammar_matcher` warnings across 102
distinct requests, and zero other `ERROR` lines**. Every affected turn still
completed correctly — this is log noise and wasted matcher work, not a wrong
answer — but the noise buries real errors and the `grammar_matcher` warning text
names the bug outright.

Two upstream causes:

1. **#52805.** `accept_tokens` advanced the matcher through a *whole* draft batch
   and only checked termination afterwards, so the drafts that follow the
   grammar's terminal token were fed to the matcher and rejected (that is the
   `grammar_matcher.cc:612` warning, verbatim). A terminated grammar also
   answered `accept_tokens` with `False`, which the caller reads as a failure
   rather than "nothing left to do" — that is the `ERROR` line. And `reset()`
   never cleared `_is_terminated`, so a reused grammar object stayed terminated
   for good.
2. **#53046.** When reasoning ends mid-window, the drafts after the `</think>`
   marker predate the bitmask and are not guaranteed grammar-valid. Upstream
   *advanced* the matcher with them and swallowed the rejection — but
   `accept_token` had already logged the `ERROR` by then. Validating first with
   `validate_tokens` (which rolls back) and advancing only on success keeps the
   state machine correct and silent.

Both are log/state hygiene: on a healthy path they do not change which tokens are
emitted. Neither touches a CUDA kernel, so speed and KV are expected unchanged —
and measured unchanged (see the gate report).

## Env gate — default ON

```
HAREM_XGRAMMAR_BACKPORT unset or 1  -> backported behaviour (recommended)
HAREM_XGRAMMAR_BACKPORT=0           -> pre-backport 487ecf187 behaviour, exactly
```

Default ON, like `patch-glm47-failclosed-tp3.py` and for the same reason: the
pre-backport default is the bug, so an unset knob on a fresh node must be the
safe one.

The gate means **rollback needs no re-patch and no new image** — add
`HAREM_XGRAMMAR_BACKPORT=0` to `EXTRA_ENV` and restart. The fastload sidecar
identity is keyed on the `patch-*.py` set plus the prelude text, *not* on
`EXTRA_ENV`, so flipping the gate does **not** invalidate the sidecar.

## Anchors

Six anchors across two files, each required exactly once in the installed file; a
missing or duplicated anchor exits non-zero instead of guessing. Each file is
marked (`HAREM-XGRAMMAR-BACKPORT`) independently, so re-running is a per-file
no-op.

| anchor | file | what it does |
|---|---|---|
| `B1-stdlib-imports` | `backend_xgrammar.py` | adds `import os` |
| `B2-gate` | `backend_xgrammar.py` | defines `_HAREM_XG_BACKPORT`, logs one line at import |
| `B3-accept-tokens` | `backend_xgrammar.py` | #52805: stop the batch at termination; terminated ⇒ `True` |
| `B4-validate-tokens` | `backend_xgrammar.py` | #52805: `[]` once terminated; stop probing past the terminal token |
| `B5-reset` | `backend_xgrammar.py` | #52805: `reset()` clears `_is_terminated` |
| `C1-stdlib-imports` | `structured_output/__init__.py` | adds `import os` |
| `C2-gate` | `structured_output/__init__.py` | defines `_HAREM_XG_BACKPORT`, logs one line at import |
| `C3-grammar-bitmask` | `structured_output/__init__.py` | #53046: validate-then-accept after the reasoning-end marker |

## How to invoke it

```bash
run python3 "$TP3_DIR/patch-xgrammar-backports-tp3.py" \
    --root "$(dirname "$VLLM_PY")" --in-place
```

`--root` is the **dist-packages** root (the `REL` paths start with `vllm/`), the
same convention as `patch-flashkda-tp3.py` and `patch-glm47-failclosed-tp3.py`.
Without `--in-place` it is a dry run: every anchor is validated, `dry run OK` is
printed, nothing is written.

In the prelude it sits **immediately after** `patch-glm47-failclosed-tp3.py`. No
real ordering constraint — no other HAREM patch touches
`vllm/v1/structured_output/` — but that is where the upstream reporter put it, so
the two recipes line up.

Two log lines prove it ran:

```
[HAREM-XGRAMMAR-BACKPORT] backend_xgrammar enabled=True (HAREM_XGRAMMAR_BACKPORT='')
[HAREM-XGRAMMAR-BACKPORT] structured_output enabled=True (HAREM_XGRAMMAR_BACKPORT='')
```

A boot log missing either line is a boot where this patch did not run.

## Test

`test_xgrammar_backports.py` — **36 checks, image-only**, against an
already-patched tree. It drives `XgrammarGrammar` with a fake xgrammar matcher:
no GPU, no model, no grammar compilation.

```bash
sudo docker run --rm \
  -v $PWD/patch-xgrammar-backports-tp3.py:/tmp/p.py:ro \
  -v $PWD/test_xgrammar_backports.py:/tmp/t.py:ro \
  --entrypoint bash exl3-zeus:754421f -c \
  'R=/usr/local/lib/python3.12/dist-packages; python3 /tmp/p.py --root $R --in-place && python3 /tmp/t.py'
```

Coverage: #52805 behaviourally in **both** gate positions (batch stops at the
terminal token; terminated grammar returns `True` and never touches the matcher;
`validate_tokens` returns `[]` and rolls its probe back exactly once; a genuinely
invalid token still fails loudly; `reset()` clears termination; `rollback()` still
recomputes it — and, with the gate off, each of those reverts to the documented
pre-backport bug). #53046 lives inside `StructuredOutputManager.grammar_bitmask`,
which needs a scheduler, a `VllmConfig` and spec-decode state to call, so it is
checked **structurally** here (validate-before-accept under
`post_reasoning_end_in_window`, two and only two `accept_tokens` call sites) and
**behaviourally in production** by the FSM-ERROR count in a strict window.

## Rollback

Cheapest first:

1. `HAREM_XGRAMMAR_BACKPORT=0` in `EXTRA_ENV`, then a three-node restart. Sidecar
   stays valid; behaviour returns to `487ecf187` exactly.
2. Full removal: restore `tp3-prelude.sh.bak-11eyl-xgrammar` and
   `.env.tp3.bak-11eyl-xgrammar` on all three nodes (which also puts
   `FASTLOAD_DIR=/var/tmp/glm53-exl3-failclosed` / `FASTLOAD_MODE=load` back —
   that sidecar is kept on disk) and restart all three together. Restoring the
   prelude changes the patch set back, so the old sidecar is valid again.

Never restart one node alone: a single-node restart kills the fabric port on its
peers.
