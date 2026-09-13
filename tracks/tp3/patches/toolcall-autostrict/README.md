# `patches/toolcall-autostrict` — `tool_choice="auto"` arms the tool-call grammar

vLLM builds the `glm_4_7` structural-tag grammar and then **discards it** for
almost every agentic request. `get_model_structural_tag`
(`vllm/tool_parsers/structural_tag_registry.py`) ends with:

```python
if tool_choice == "auto" and not _any_tool_strict(tools):
    return None
```

`glm_4_7` is in the builtin registry, so the grammar exists and is correct — it is
thrown away because clients do not set `strict: true` on their tool schemas, and
no agent harness we have seen does. Every agentic request therefore samples
tool-call syntax with **no sampling-layer constraint at all**, and a malformed
call is freely samplable.

This is the engine-side form of the lever measured in
[`results/gates/toolcall-gate-11sep-strict.md`](../../../../results/gates/toolcall-gate-11sep-strict.md),
where injecting `strict` at a proxy gave 133/133 well-formed calls, and named as
step 2 in issue #7: *"an opt-in engine env that treats `tool_choice="auto"` as
strict for clients that never set the field, published as a patch with the gate
results."*

## What is here

| file | what it is |
|---|---|
| `patch-toolcall-autostrict-tp3.py` | two anchors, idempotent, fail-closed |
| `test_toolcall_autostrict.py` | 5 model-free checks, CPU only |
| `README.md` | this file |

## Why the engine and not a proxy

A proxy works when the gateway fronts one model. On a gateway that also routes
to other providers, injecting `strict` there means per-model special-casing in a
shared routing layer, and it only covers clients that traverse that proxy. In
the engine the behaviour stays with the model it belongs to and applies to every
client.

## 1. What the patch changes

Two anchors in `vllm/tool_parsers/structural_tag_registry.py`:

1. `import os` beside the existing `collections.abc` import (the module does not
   import it today).
2. The discard branch gains one condition:

```python
if tool_choice == "auto" and not _any_tool_strict(tools):
    if os.environ.get("HAREM_TOOLCALL_AUTOSTRICT", "0").strip() != "1":
        return None
```

The environment is read **at call time**, not import time, so the knob needs no
rebuild. Nothing else in the selection path changes: `tool_choice="none"` still
discards, a `strict: true` tool is still armed with the gate off, and
`VLLM_ENFORCE_STRICT_TOOL_CALLING=0` still disables everything upstream of this.

## 2. The knob, and what `=0` is worth

```
HAREM_TOOLCALL_AUTOSTRICT unset or 0 -> upstream behaviour, byte for byte
HAREM_TOOLCALL_AUTOSTRICT=1          -> tool_choice="auto" arms the grammar
```

**Default OFF**, unlike `HAREM_GLM47_FAILCLOSED`. Upstream's choice here is a
defensible design decision rather than a bug, and this one carries a real trade
(§5). The knob is not part of the fast-load manifest identity; the patch *file*
is, so adding it costs one dump boot, and flipping it afterwards is a restart.

## 3. The failure it removes

On a 3× DGX Spark TP=3 deployment (full scope, DFlash2 k=7, `index_topk` 8192,
the prefix-hit / K-pool-tail / sm12 / indexer-workspace set, FlashKDA, the
fail-closed parser and the xgrammar backports all on), the raw text of rejected
calls showed one shape every time — sanitised, structure verbatim:

```
bash<arg_key>sed -n '1,20p' server.py | grep -n "ClassA\|ClassB"</arg_value>
bash<arg_key>export AWS_PROFILE=x\naws s3 ls s3://bucket/prefix/</arg_value>
bash<arg_key>def_someFunctionName;</arg_value>
```

`<arg_key>` → the **value** → `</arg_value>`: the key name and the middle tags
are skipped. A related shape put the whole call into the *name* slot
(`bash1635,1676p src/app/...`, a `sed` range welded on with no separator).

Neither is a legal production under the structural tag. Armed, the sampler never
offers those tokens; the fail-closed parser can only report them afterwards.

## 4. Before and after, production traffic

| build | window | rejected tool calls |
|---|---|---|
| fail-closed parser only | ~7 min | 6 |
| + reject reasons that name the offending key | ~30 min | 2 |
| **+ grammar armed** | **29 h, 322 requests** | **0** |

Alongside the zero: 322 `stop` / 0 `length` / 0 `abort` / 0 `error`, 0
preemptions, **0 FSM errors** (the xgrammar backports on this tree handle the
DFlash2-drafts-past-stop-token interaction completely), prefix cache 97.6%, mean
prompt 186k tokens.

**This is not a controlled A/B.** Traffic differs between the windows and only
the armed arm has a full day behind it. What is controlled is the change: the
commit that armed the grammar touched the patch, its registration, one env line
and a gate — nothing else — and the `<arg_key>VALUE</arg_value>` signature has
not recurred once since.

## 5. What it costs: the reasoning channel

Measured n=5 per cell inside one boot. The tag only applies when tools are
attached, so with-tools against without-tools is a clean within-boot control:

| | no tools | with tools (armed) |
|---|---|---|
| effort `high` | 0 ch | 29 ch |
| effort `max` | **1,724 ch** (1,301–2,475) | **161 ch** (143–184) |

Non-overlapping ranges, so the effect is real. Content length is unaffected
(387 → 415 ch) and `finish_reason` stays `stop`. Reasoning-then-**tool-call** is
untouched, because the glm47 parser closes reasoning at `<tool_call>`; the
affected population is reasoning variants on **text-answer** turns that have
tools attached. A deployment whose default is reasoning-off barely meets it.

### The mechanism is not the `excludes`, and that matters

The obvious reading — the `reasoning=False` tag lists `</think>` in `excludes`,
so the model cannot close the think block — does **not** survive the data. The
block closes cleanly and the reasoning parser splits it in every sample. GLM's
think tags are special token ids while the excludes are string-level, so they
never match.

What is actually there, in `vllm/parser/abstract_parser.py`, at the only call
site:

```python
structure_tag = self._tool_parser.get_structural_tag(
    request,
    reasoning=False,      # hardcoded
)
```

**`reasoning=False` is hardcoded.** vLLM never asks whether the request has
thinking enabled, so the `reasoning=True` shape —
`SequenceFormat([Tag(content=AnyText(...), end='</think>'), TriggeredTagsFormat(...)])`,
which is exactly what a pre-opened think block needs — is never selected for any
model, ever.

That reframes the `reasoning=True` result reported in #7 (the model rambling past
its budget without closing think): applied to requests that are *not* thinking,
that tag demands a `</think>` the model has no reason to emit. The correct value
looks per-request, read from `chat_template_kwargs`.

**This patch does not attempt that fix.** The failure mode if it is wrong is
runaway generation, which is worse than shortened reasoning. It is written up
here because it looks like a vLLM bug affecting every reasoning model with
tools, and because it suggests the cost in this table may be fixable rather than
inherent.

## 6. How it is registered, and the exact line

In `tp3full-prelude.sh`, after the xgrammar backports. Applied
**unconditionally** so a control arm runs the same bytes; the behaviour is the
env gate.

```bash
run python3 "$TP3_DIR/toolcall-autostrict/patch-toolcall-autostrict-tp3.py" \
    --root "$(dirname "$VLLM_PY")" --in-place
```

`--root` is the **dist-packages** root — `REL` starts with `vllm/`, the same
convention as `patch-flashkda-tp3.py` and `patch-glm47-failclosed-tp3.py` — and
without `--in-place` the script is a dry run that validates both anchors and
writes nothing. Ordering is immaterial: no other patch in this tree touches
`vllm/tool_parsers/`.

Then in `EXTRA_ENV`:

```
HAREM_TOOLCALL_AUTOSTRICT=1
```

## 7. Test

```bash
docker run --rm --entrypoint bash -v "$PWD":/work <image> -lc '
  D=/usr/local/lib/python3.12/dist-packages
  python3 /work/patch-toolcall-autostrict-tp3.py --root $D --in-place
  python3 /work/test_toolcall_autostrict.py'
```

Five checks, model-free, CPU only, `ALL-PASS` on a clean image. Use a throwaway
container — it patches the tree in place.

## 8. Rollback

`HAREM_TOOLCALL_AUTOSTRICT=0` and a three-node restart. The sidecar stays valid;
behaviour returns to upstream exactly, which the `=0` arm of the test asserts.
Full removal means restoring the prelude, which changes the patch set and so the
fast-load identity.

## 9. What this does not establish

- One deployment, one harness, one traffic shape; production effort level `low`.
- The before/after windows are not matched, and the 29-hour zero is a single arm.
- The grammar constrains **syntax**. It cannot touch the residual documented in
  [`copy-fidelity-residual-13sep.md`](../../../../results/gates/copy-fidelity-residual-13sep.md),
  where the syntax is valid and the copied *content* is wrong.
- Not tested at TP=2.
