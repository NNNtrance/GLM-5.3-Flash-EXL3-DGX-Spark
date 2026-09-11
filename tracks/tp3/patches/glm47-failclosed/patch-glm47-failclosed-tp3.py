#!/usr/bin/env python3
"""HAREM-TP3 fail-closed glm47 tool-call parser for GLM-5.3-Flash.

NOT IN PRODUCTION yet.  Prepared 11 September 2026 against issue #7 on the
public recipe repo; deployment waits for the owner's approval.  Behaviour is
env-gated (``HAREM_GLM47_FAILCLOSED``, default ON -- see ENV GATE below), and
``=0`` leaves this code path byte for byte upstream.

Target: ``vllm/parser/glm47_moe.py`` in our pinned vLLM (487ecf187, image
exl3-zeus:754421f).

THE BUG IT CLOSES
-----------------
At ~60k+ tokens of agentic context GLM-5.3-Flash occasionally loses its own
tool-call format and emits malformed XML, e.g.

    <tool_call>Bash<arg_key>S1_latest.json</arg_value><arg_key>description</arg_key>...
    <tool_call>Bash<arg_key>SimpsonsF><tool_use_error>...</invoke>

Upstream then *salvages* it instead of refusing:

* ``_glm47_arg_converter`` (glm47_moe.py:56-70) regex-scrapes every
  ``<arg_key>K</arg_key><arg_value>V</arg_value>`` pair into a dict and
  ``.strip()``s the key.  Nothing else is checked, so a key like
  ``print_code_snapshot</arg_value><arg_key>description`` becomes a real JSON
  argument name.
* ``validate_tool_names=True`` (glm47_moe.py:171) checks the *name* only --
  and ``ParserEngine._is_valid_tool_name`` (parser_engine.py:392-397) returns
  True unconditionally when the request carries no tools.
* ``stream_arg_deltas=True`` (glm47_moe.py:169) half-streams the arguments, so
  a call that later turns out to be garbage has already reached the client.
* A call that never closes produces no ``TOOL_CALL_END`` at all, so
  ``_handle_tool_end`` never runs -- but ``_build_extracted_result``
  (parser_engine.py:1014-1065) still turns the dangling slot into a tool call.

The client echoes the salvaged garbage back into the history, the chat template
re-renders it as canonical ``<arg_key>`` syntax, and the model starts imitating
its own corruption.  Each round is worse until nothing parses and the whole
turn is swallowed (empty stream, finish=stop at ``<|observation|>``).  Refusing
the first bad call breaks that loop.

WHAT THE PATCH DOES
-------------------
When ``HAREM_GLM47_FAILCLOSED`` != "0":

(a) ``stream_arg_deltas=False`` and ``_emit_name_delta`` is suppressed while a
    call is still open, so a tool call is buffered and emitted as ONE delta
    when it closes.  Nothing partial can reach the client.
(b) At ``TOOL_CALL_END`` the complete call is validated:
      * the name is identifier-shaped and, when the request carries tools,
        present in them;
      * every argument key is identifier-shaped and, when the tool has a
        schema, present in its ``properties``;
      * every key in the tool's ``parameters.required`` is present in the
        parsed arguments (``HAREM_GLM47_REQUIRED``, default ON, see below);
      * a non-empty argument body that the converter extracted *nothing*
        usable from counts as truncated.
    A call that never closes is validated at finish and fails as truncated.
(c) A rejected call is surfaced as plain CONTENT -- the raw
    ``<tool_call>...</tool_call>`` text -- and its slot is cleared, so neither
    the streaming path nor ``_build_extracted_result`` can emit it as a tool
    call.  The user sees the garbage as text; nothing parseable-as-a-tool-call
    re-enters the history.
(d) ``HAREM_GLM47_FAILCLOSED=0`` restores upstream behaviour byte for byte:
    every override returns ``super()`` immediately and ``stream_arg_deltas``
    goes back to True.

ENV GATE -- DEFAULT ON
----------------------
  HAREM_GLM47_FAILCLOSED unset or 1 -> fail closed (recommended).
  HAREM_GLM47_FAILCLOSED=0          -> upstream behaviour, byte for byte.

Unlike the other HAREM patches the default is ON, because the upstream default
is the bug: an unset knob on a fresh node must be the safe one.

REQUIRED-FIELD GATE -- DEFAULT ON, UNDER THE ONE ABOVE
------------------------------------------------------
  HAREM_GLM47_REQUIRED unset or 1 -> missing required keys are rejected.
  HAREM_GLM47_REQUIRED=0          -> keys are shape/schema-checked only.

Added 11 September 2026, NOT IN PRODUCTION yet (pending A/B + gates).  In the
issue #7 gate run the *client* schema-rejected a salvaged call for a missing
required key at 35k tokens -- the parser had already let it through, because
``properties`` membership says nothing about what the schema demands.  The check
is presence of the key only: a key whose value is an empty string passes, since
an empty string is a legal value for a ``"type": "string"`` parameter and the
tool, not the parser, owns that judgement.  It never runs when
``HAREM_GLM47_FAILCLOSED=0``, because no validation runs at all there.

LOGGING
-------
One line at import::

    [HAREM-GLM47-FAILCLOSED] failclosed=True (HAREM_GLM47_FAILCLOSED='') required=True (HAREM_GLM47_REQUIRED='')

and one warning per rejected call (name, truncated to 48 chars, plus the
reason).  The argument body is never logged -- it is user content.
A boot log without the import line is a boot where this patch did not run.

HOW TO INVOKE IT
----------------
    run python3 "$TP3_DIR/patch-glm47-failclosed-tp3.py" \\
        --root "$(dirname "$VLLM_PY")" --in-place

``--root`` is the DIST-PACKAGES root (``REL`` below starts with ``vllm/``, like
patch-flashkda-tp3.py).  Without ``--in-place`` this is a DRY RUN: it prints
"dry run OK", exits 0 and patches nothing.

FAIL-CLOSED PATCHING
--------------------
Six anchors, each required exactly once in the installed file.  A missing or
duplicated anchor exits non-zero instead of guessing.  Re-running is a no-op.
"""

import argparse
import os
import sys

REL = "vllm/parser/glm47_moe.py"
MARK = "HAREM-GLM47-FAILCLOSED"

# --- anchor 1: stdlib imports ------------------------------------------------
A1_OLD = "import functools\nimport json\n"
A1_NEW = "import functools\nimport json\nimport os\n"

# --- anchor 2: vllm imports + module logger ---------------------------------
A2_OLD = """from vllm.parser.engine.parser_engine_config import (
    ParserEngineConfig,
    ParserState,
    Transition,
)
"""
A2_NEW = """from vllm.parser.engine.parser_engine_config import (
    ParserEngineConfig,
    ParserState,
    Transition,
)

# HAREM-GLM47-FAILCLOSED.  All three already sit in this module's import graph
# (parser_engine imports them), so there is no new cycle.
from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.logger import init_logger
from vllm.tool_parsers.utils import find_tool_name, find_tool_properties

logger = init_logger(__name__)
"""

# --- anchor 3: the gate + shape check, right after the arg converter --------
A3_OLD = "    return json.dumps(params, ensure_ascii=False)\n"
A3_NEW = '''    return json.dumps(params, ensure_ascii=False)


# HAREM-GLM47-FAILCLOSED ------------------------------------------------------
# A tool name or an argument key the model did not invent: a leading letter or
# underscore, then letters/digits/underscore/dot/dash.  Every captured
# corruption vector fails it on the first '<', '>', '/', '(', '=' or space.
_HAREM_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.\\-]{0,127}\\Z")


def _harem_failclosed() -> bool:
    """True unless HAREM_GLM47_FAILCLOSED is explicitly "0"."""
    return os.environ.get("HAREM_GLM47_FAILCLOSED", "1").strip() != "0"


def _harem_required() -> bool:
    """True unless HAREM_GLM47_REQUIRED is explicitly "0".

    Sub-gate of _harem_failclosed: when that one is off no validation runs at
    all, so this knob only ever narrows the fail-closed arm.
    """
    return os.environ.get("HAREM_GLM47_REQUIRED", "1").strip() != "0"


print(
    f"[{MARK}] failclosed={_harem_failclosed()} "
    f"(HAREM_GLM47_FAILCLOSED={os.environ.get('HAREM_GLM47_FAILCLOSED', '')!r}) "
    f"required={_harem_required()} "
    f"(HAREM_GLM47_REQUIRED={os.environ.get('HAREM_GLM47_REQUIRED', '')!r})",
    flush=True,
)
'''.replace("{MARK}", MARK)

# --- anchor 4: config signature ---------------------------------------------
A4_OLD = "def glm47_moe_config(thinking: bool = True) -> ParserEngineConfig:\n"
A4_NEW = """def glm47_moe_config(
    thinking: bool = True,
    harem_failclosed: bool | None = None,
) -> ParserEngineConfig:
    # HAREM-GLM47-FAILCLOSED.  Passed explicitly by Glm47MoeParser so the
    # functools.cache key covers the gate; None keeps upstream call sites working.
    if harem_failclosed is None:
        harem_failclosed = _harem_failclosed()
"""

# --- anchor 5: buffer the arguments instead of streaming them ---------------
A5_OLD = "        stream_arg_deltas=True,\n"
A5_NEW = "        stream_arg_deltas=not harem_failclosed,  # HAREM-GLM47-FAILCLOSED\n"

# --- anchor 6: parser state + the overrides ---------------------------------
A6_OLD = """        kwargs.setdefault(
            "parser_engine_config",
            glm47_moe_config(thinking=self.thinking_enabled),
        )
        super().__init__(tokenizer, tools, **kwargs)

    def _emit_name_delta(self, idx: int, deltas, name: str | None) -> None:
        if name is not None:
            name = name.strip()
        super()._emit_name_delta(idx, deltas, name)

    def _handle_tool_end(self, event, deltas) -> None:
        idx = event.tool_index
        if 0 <= idx < len(self._tool_slots):
            self._tool_slots[idx].name = self._tool_slots[idx].name.strip()
        super()._handle_tool_end(event, deltas)
"""
A6_NEW = '''        # HAREM-GLM47-FAILCLOSED.  Set before super().__init__ so the
        # overrides below are safe from the first event onwards.
        self._harem_failclosed = _harem_failclosed()
        # Sub-gate, read once per request parser like the one above.
        self._harem_required = _harem_required()
        self._harem_closed: set[int] = set()
        self._harem_reject_text: list[str] = []
        kwargs.setdefault(
            "parser_engine_config",
            glm47_moe_config(
                thinking=self.thinking_enabled,
                harem_failclosed=self._harem_failclosed,
            ),
        )
        super().__init__(tokenizer, tools, **kwargs)

    def _emit_name_delta(self, idx: int, deltas, name: str | None) -> None:
        # HAREM-GLM47-FAILCLOSED.  The only caller is _handle_arg_chunk, i.e.
        # the call is still open; holding the name back is what makes the call
        # emit as one delta from _handle_tool_end once it has been validated.
        if self._harem_failclosed:
            return
        if name is not None:
            name = name.strip()
        super()._emit_name_delta(idx, deltas, name)

    def _handle_tool_end(self, event, deltas) -> None:
        idx = event.tool_index
        in_range = 0 <= idx < len(self._tool_slots)
        if in_range and self._harem_failclosed:
            # HAREM-GLM47-FAILCLOSED.  Validate the COMPLETE call before any of
            # it becomes a tool call.
            self._harem_closed.add(idx)
            slot = self._tool_slots[idx]
            reason = self._harem_validate_call(slot.name, slot.args)
            if reason is not None:
                self._harem_reject(idx, reason, event.value or TOOL_CALL_END)
                return
        if in_range:
            self._tool_slots[idx].name = self._tool_slots[idx].name.strip()
        super()._handle_tool_end(event, deltas)

    # ── HAREM-GLM47-FAILCLOSED ────────────────────────────────────────────

    def _harem_validate_call(self, name: str, raw_args: str) -> str | None:
        """Return None if the call is sound, else a short reason.

        Mirrors what the model is allowed to have produced, not what the
        regexes can be made to match.
        """
        name = name.strip()
        if not name:
            return "empty tool name"
        if not _HAREM_IDENT_RE.match(name):
            return "tool name is not identifier-shaped"
        if self._tools and not find_tool_name(self._tools, name):
            return "tool name is not in the request tools"

        # HAREM_GLM47_REQUIRED.  Key names come from the request's own schema,
        # never from model output, so they are safe to name in the reason.
        required = self._harem_required_keys(name) if self._harem_required else set()

        if not raw_args.strip():
            # A no-argument call is legal -- unless the schema demands keys.
            if required:
                return "missing required key(s): " + ", ".join(sorted(required))
            return None

        converter = self._arg_converter
        try:
            parsed = json.loads(converter(raw_args, False)) if converter else None
        except (TypeError, ValueError):
            return "argument body did not convert to JSON"
        if not isinstance(parsed, dict):
            return "argument body did not convert to an object"
        if not parsed:
            # Tags opened but never closed: the converter matched no pair.
            return "non-empty argument body yielded no arguments (truncated)"

        properties = find_tool_properties(self._tools, name) if self._tools else {}
        for key in parsed:
            if not _HAREM_IDENT_RE.match(key):
                return f"argument key is not identifier-shaped ({len(key)} chars)"
            if properties and key not in properties:
                return "argument key is not in the tool schema"

        missing = sorted(required - set(parsed))
        if missing:
            return "missing required key(s): " + ", ".join(missing)
        return None

    def _harem_required_keys(self, name: str) -> set[str]:
        """The named tool's ``parameters.required``, or an empty set.

        find_tool_properties returns only ``properties``, so walk the request's
        tools here.  Every hop accepts a pydantic object OR a plain dict: the
        OpenAI entrypoint hands us ChatCompletionToolsParam, while MCP and
        test paths hand us raw dicts, and a malformed schema must read as "no
        required keys" rather than raise inside the parser.
        """
        for tool in self._tools or ():
            fn = (
                tool.get("function") if isinstance(tool, dict)
                else getattr(tool, "function", None)
            )
            if fn is None:
                continue
            fn_name = (
                fn.get("name") if isinstance(fn, dict)
                else getattr(fn, "name", None)
            )
            if fn_name != name:
                continue
            params = (
                fn.get("parameters") if isinstance(fn, dict)
                else getattr(fn, "parameters", None)
            )
            required = (
                params.get("required") if isinstance(params, dict)
                else getattr(params, "required", None)
            )
            if isinstance(required, (list, tuple, set)):
                return {k for k in required if isinstance(k, str) and k}
            return set()
        return set()

    def _harem_reject(self, idx: int, reason: str, closer: str) -> None:
        """Surface the raw tool-call text as content and clear the slot."""
        slot = self._tool_slots[idx]
        self._harem_reject_text.append(
            TOOL_CALL_START + slot.name + slot.args + closer
        )
        logger.warning(
            "[%s] rejected tool call idx=%d name=%r: %s",
            "''' + MARK + '''",
            idx,
            slot.name.strip()[:48],
            reason,
        )
        # A fresh slot is skipped by _build_extracted_result (no name, no args),
        # and no tool-call id was burned because _ensure_tool_id never ran.
        self._tool_slots[idx] = type(slot)()

    def _harem_finalize(self, delta, finished: bool):
        """Reject never-closed calls, then append rejected text as content."""
        if finished:
            for idx, slot in enumerate(self._tool_slots):
                if idx in self._harem_closed or (not slot.name and not slot.args):
                    continue
                self._harem_closed.add(idx)
                self._harem_reject(idx, "tool call never closed (truncated)", "")
        if not self._harem_reject_text:
            return delta
        text = "".join(self._harem_reject_text)
        self._harem_reject_text.clear()
        if delta is None:
            return DeltaMessage(content=text)
        delta.content = (delta.content or "") + text
        return delta

    def _events_to_delta(self, events, finished: bool = False):
        delta = super()._events_to_delta(events, finished)
        if not self._harem_failclosed:
            return delta
        return self._harem_finalize(delta, finished)

    def finish_streaming(self):
        # _events_to_delta already finalizes when it runs; _harem_finalize is
        # idempotent, and this covers the path where super() returns early
        # because there were neither events nor deferred content.
        delta = super().finish_streaming()
        if not self._harem_failclosed:
            return delta
        return self._harem_finalize(delta, True)

    def _reset(self, initial_state=None) -> None:
        super()._reset(initial_state=initial_state)
        self._harem_closed.clear()
        self._harem_reject_text.clear()
'''

ANCHORS = [
    ("A1-stdlib-imports", A1_OLD, A1_NEW),
    ("A2-vllm-imports", A2_OLD, A2_NEW),
    ("A3-gate", A3_OLD, A3_NEW),
    ("A4-config-signature", A4_OLD, A4_NEW),
    ("A5-stream-arg-deltas", A5_OLD, A5_NEW),
    ("A6-overrides", A6_OLD, A6_NEW),
]


def apply(src: str, where: str) -> str:
    if MARK in src:
        print(f"patch-glm47-failclosed: already applied ({where})")
        return src
    for name, old, _new in ANCHORS:
        n = src.count(old)
        if n != 1:
            print(
                f"patch-glm47-failclosed: {name} count={n} (expected 1) in "
                f"{where} -- refusing",
                file=sys.stderr,
            )
            sys.exit(3)
    for _name, old, new in ANCHORS:
        src = src.replace(old, new, 1)
    return src


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dist-packages root")
    ap.add_argument("--in-place", action="store_true")
    ap.add_argument("--out", default="", help="write here instead of in place")
    a = ap.parse_args()
    p = os.path.join(a.root, REL)
    src = open(p).read()
    out = apply(src, REL)
    if out is src:
        return
    dst = a.out or (p if a.in_place else "")
    if not dst:
        print("patch-glm47-failclosed: dry run OK (pass --in-place or --out to write)")
        return
    open(dst, "w").write(out)
    print(
        f"patch-glm47-failclosed: applied to {dst} "
        "(HAREM_GLM47_FAILCLOSED honoured, default ON)"
    )


if __name__ == "__main__":
    main()
