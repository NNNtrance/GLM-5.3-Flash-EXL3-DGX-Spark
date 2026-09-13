#!/usr/bin/env python3
"""HAREM-TP3: treat tool_choice="auto" as strict, so the tool-call GRAMMAR arms.

WHY
---
vLLM builds a structural-tag grammar for GLM's tool-call syntax and then throws
it away for almost every real agentic request.  ``get_model_structural_tag``
(vllm/tool_parsers/structural_tag_registry.py) ends with:

    if tool_choice == "auto" and not _any_tool_strict(tools):
        return None

``glm_4_7`` IS in the builtin registry, so the grammar exists and is correct --
it is discarded because clients do not set ``strict: true`` on their tool
schemas, and essentially no agent harness does.  The result is that every
agentic request runs with NO sampling-layer constraint on tool-call syntax, and
malformed calls are freely samplable.

WHAT WAS MEASURED, which is why this patch exists
------------------------------------------------
Captured from production by logging the raw text of REJECTED calls only
(sanitised below; the structure is verbatim):

    bash<arg_key>sed -n '1,20p' server.py | grep -n "ClassA\\|ClassB"</arg_value>
    bash<arg_key>export AWS_PROFILE=x\\naws s3 ls s3://bucket/prefix/</arg_value>
    bash<arg_key>def_someFunctionName;</arg_value>

One shape every time: ``<arg_key>`` -> the VALUE -> ``</arg_value>``.  The model
opens the key tag, writes the argument value into it, and closes with the value
tag, skipping ``command</arg_key><arg_value>`` entirely.  That sequence is NOT a
legal production under the structural tag, so with the grammar armed the model
cannot emit it -- the sampler never offers the tokens.  A parser can only report
it afterwards, which is what we had been doing.

Ruled out first, so this is not a guess: the chat template (renders prior calls
correctly -- verified by rendering a real agent history through the live
template), truncation by token cap or client abort (finish_reason: 36 stop, 0
length, 0 abort, 0 error over the window), preemption (0), KV capacity (never
above 39%), and index_topk (already at its 8192 ceiling -- higher values die in
worker init).

Upstream's measurement of the same lever, same engine, minutes apart:
72 tool calls / 72 well-formed with strict injected, versus 7 of 61 carrying
tool-call markup without it.

THE COST, STATED UP FRONT
-------------------------
The ``reasoning=False`` structural tag puts ``<think>``/``</think>`` in its
``excludes``, and GLM's template force-opens the think block.  With the grammar
armed the model therefore cannot close reasoning with ``</think>`` on a turn
that produces plain text, so think-then-TEXT turns degrade: either reasoning is
skipped, or plan text lands in ``content``.  Reasoning-then-TOOL-CALL is
unaffected, because the glm47 parser closes reasoning at ``<tool_call>``.

The exposure is narrower than it first sounds: the structural tag only applies
when TOOLS ARE ATTACHED, and this deployment defaults reasoning OFF.  So the
affected population is the low/high/max reasoning variants on text-answer turns
inside tool-enabled sessions.  Measure it before and after; the knob is one line.

Upstream tried the obvious alternative (``reasoning=True``, sequence any_text ->
``</think>`` -> triggered tags) and found it WORSE: the model rambles past its
token budget without ever closing think.  Do not reach for it.

SPEC DECODE
-----------
With DFlash2 k=7 the drafter proposes tokens past a terminated grammar's stop
token, which pre-#52805/#53046 builds log as "Failed to advance FSM" at ERROR --
hundreds per hour.  This tree already carries those backports
(HAREM_XGRAMMAR_BACKPORT, default on), which takes that count to zero, so the
interaction is already handled here.  If FSM errors appear after
arming this, check that gate first.

ENV GATE -- DEFAULT OFF
-----------------------
  HAREM_TOOLCALL_AUTOSTRICT unset or 0 -> upstream behaviour, byte for byte.
  HAREM_TOOLCALL_AUTOSTRICT=1          -> tool_choice="auto" arms the grammar.

Default OFF deliberately, unlike the fail-closed parser: upstream's default here
is a defensible choice rather than a bug, and this one carries a real trade.

HOW TO INVOKE IT
----------------
    run python3 "$TP3_DIR/toolcall-autostrict/patch-toolcall-autostrict-tp3.py" \\
        --root "$(dirname "$VLLM_PY")" --in-place

``--root`` is the DIST-PACKAGES root (REL starts with "vllm/"), same convention
as patch-flashkda-tp3.py and patch-glm47-failclosed-tp3.py.  Without
``--in-place`` it is a DRY RUN: anchors are validated, nothing is written.

Two anchors, each required exactly once.  Re-running is a no-op.
"""

import argparse
import os
import sys

REL = "vllm/tool_parsers/structural_tag_registry.py"
MARK = "HAREM-TOOLCALL-AUTOSTRICT"

# --- anchor 1: stdlib import (the module does not import os today) ----------
A1_OLD = "from collections.abc import Callable, Sequence\n"
A1_NEW = ("import os  # HAREM-TOOLCALL-AUTOSTRICT\n"
          "from collections.abc import Callable, Sequence\n")

# --- anchor 2: the gate that discards the grammar ---------------------------
A2_OLD = """    if tool_choice == "auto" and not _any_tool_strict(tools):
        return None
"""
A2_NEW = '''    if tool_choice == "auto" and not _any_tool_strict(tools):
        # HAREM-TOOLCALL-AUTOSTRICT.  Upstream discards the grammar here unless a
        # tool declares strict:true, which no agent harness does -- so agentic
        # traffic samples tool-call syntax unconstrained.  =1 keeps the grammar.
        # Read at call time, not import time, so the knob needs no rebuild.
        if os.environ.get("HAREM_TOOLCALL_AUTOSTRICT", "0").strip() != "1":
            return None
'''

ANCHORS = [
    ("A1-os-import", A1_OLD, A1_NEW),
    ("A2-auto-strict-gate", A2_OLD, A2_NEW),
]


def apply(src: str, where: str) -> str:
    if MARK in src:
        print(f"patch-toolcall-autostrict: already applied ({where})")
        return src
    for name, old, new in ANCHORS:
        n = src.count(old)
        if n != 1:
            print(f"patch-toolcall-autostrict: anchor {name} found {n} times "
                  f"(expected 1) in {where} -- refusing", file=sys.stderr)
            sys.exit(3)
        src = src.replace(old, new, 1)
    return src


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dist-packages root")
    ap.add_argument("--in-place", action="store_true")
    ap.add_argument("--out", default="", help="write here instead of in place")
    a = ap.parse_args()
    path = os.path.join(a.root, REL)
    if not os.path.exists(path):
        print(f"patch-toolcall-autostrict: {path} not found", file=sys.stderr)
        return 4
    src = open(path, encoding="utf-8").read()
    out = apply(src, REL)
    if out == src and MARK in src:
        return 0
    dst = a.out or (path if a.in_place else "")
    if not dst:
        print("patch-toolcall-autostrict: dry run OK "
              "(pass --in-place or --out to write)")
        return 0
    os.makedirs(os.path.dirname(dst), exist_ok=True) if os.path.dirname(dst) else None
    open(dst, "w", encoding="utf-8").write(out)
    print(f"patch-toolcall-autostrict: applied to {dst} "
          f"(HAREM_TOOLCALL_AUTOSTRICT honoured, default OFF)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
