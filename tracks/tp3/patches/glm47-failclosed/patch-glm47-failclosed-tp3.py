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
reason).
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

# HAREM-GLM47-RETRY.  The retryable arm emits a real tool call instead of prose, so it
# needs the same four symbols the base engine builds its calls from. All four
# already sit in this module's import graph via parser_engine.
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaToolCall,
    FunctionCall,
    ToolCall,
)

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


# HAREM-GLM47-RETRY -------------------------------------------------------------------
# ACU-Serve delta over upstream, 2026-09-12.  Upstream surfaces a rejected call
# as CONTENT.  That ends the agent's turn: every agent loop reads "assistant
# produced text and no tool call" as "the turn is finished", so one malformed
# call silently terminates the session instead of being retried.  Observed in
# production the morning this landed -- name='bash' with an argument key
# 'review_expr' that is not in bash's schema; the user saw a fragment of markup
# and a dead turn.
#
# The cascade upstream was defending against did NOT come from emitting a tool
# call. It came from _glm47_arg_converter SALVAGING garbage into a PLAUSIBLE
# dict -- {"print_code_snapshot</arg_value><arg_key>description": "..."} -- which
# the client echoed back, the chat template re-rendered in canonical <arg_key>
# syntax, and the model then imitated. The defect is the plausibility, not the
# emission.
#
# So: emit the call, with arguments that cannot be mistaken for real ones.
#   {"__harem_invalid_tool_call__": "<why it was rejected>"}
# The key is in no tool's schema, so a schema-validating client ALWAYS rejects
# it and never half-executes -- it cannot succeed with arguments missing, which
# is what made "just drop the bad args" unsafe. The client turns that into an
# ordinary tool error, the error goes back as a tool message, and the model is
# told exactly what it got wrong and retries. Nothing imitable enters history:
# the echoed assistant message carries a sentinel key, not <arg_key> markup.
#
# ONLY when the tool NAME is valid and in the request's tools. A hallucinated or
# malformed NAME has no call to make, so those still fall back to content.
#
# HAREM_GLM47_RETRY=0 restores upstream's content-only behaviour exactly.
_HAREM_INVALID_KEY = "__harem_invalid_tool_call__"



def _harem_safe(text: str, limit: int) -> str:
    """HAREM-GLM47-RETRY: model output made safe to quote back at the model.

    The reason string is echoed into the conversation history, so anything from
    it that looks like tool-call markup could be imitated -- which is the
    cascade the fail-closed arm exists to stop.  Strip anything angle-bracketed,
    collapse whitespace, cap the length.  What survives is the plain text the
    model meant, which is the part it needs in order to correct itself.
    """
    text = re.sub(r"<[^>]*>", "", text)
    text = " ".join(text.split())
    return text[:limit] + ("..." if len(text) > limit else "")


def _harem_retry() -> bool:
    """True unless HAREM_GLM47_RETRY is explicitly "0".

    Sub-gate of _harem_failclosed, like _harem_required: with the fail-closed
    arm off nothing is ever rejected, so this knob cannot fire on its own.
    """
    return os.environ.get("HAREM_GLM47_RETRY", "1").strip() != "0"


print(
    f"[{MARK}] failclosed={_harem_failclosed()} "
    f"(HAREM_GLM47_FAILCLOSED={os.environ.get('HAREM_GLM47_FAILCLOSED', '')!r}) "
    f"required={_harem_required()} "
    f"(HAREM_GLM47_REQUIRED={os.environ.get('HAREM_GLM47_REQUIRED', '')!r}) "
    f"acu_retry={_harem_retry()} "
    f"(HAREM_GLM47_RETRY={os.environ.get('HAREM_GLM47_RETRY', '')!r})",
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
        # HAREM-GLM47-RETRY.  Read once per request parser, like the gates above.
        self._harem_retry = _harem_retry()
        # HAREM-GLM47-RETRY.  Rejections that carry a VALID tool name, held so the
        # non-streaming extractor can emit them after the base has run.
        self._harem_rejected: list[tuple[str, str]] = []
        self._harem_pending_stream: list = []
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
                # HAREM-GLM47-RETRY: deltas is passed so a retryable rejection can emit
                # its sentinel call on the streaming path.
                self._harem_reject(idx, reason, event.value or TOOL_CALL_END,
                                   deltas)
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
            return ("tool name is not one of the offered tools (offered: "
                    + ", ".join(sorted(self._harem_tool_names())[:12]) + ")")

        # HAREM_GLM47_REQUIRED.  Key names come from the request's own schema,
        # never from model output, so they are safe to name in the reason.
        required = self._harem_required_keys(name) if self._harem_required else set()

        if not raw_args.strip():
            # A no-argument call is legal -- unless the schema demands keys.
            if required:
                return ("missing required key(s): " + ", ".join(sorted(required))
                        + " -- the call carried no arguments at all")
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
            return ("the call was cut off before a complete key/value pair -- "
                    "write the key name, then the value, and close the call"
                    + (f"; valid keys for {name!r} are: "
                       + ", ".join(sorted(find_tool_properties(self._tools, name) or {}))
                       if self._tools and find_tool_properties(self._tools, name) else ""))

        properties = find_tool_properties(self._tools, name) if self._tools else {}
        # HAREM-GLM47-RETRY: the reason is quoted back to the MODEL, so it has to name
        # the mistake and the correction. "argument key is not in the tool
        # schema" told it nothing it could act on, and it repeated the same
        # error every turn -- observed in production 2026-09-12, where the model
        # called bash with 'prefix'/'path' over and over. Valid key names come
        # from the request's own schema and are always safe to quote; the
        # offending key is model output and goes through _harem_safe first.
        valid = ", ".join(sorted(properties)) if properties else ""
        tail = f"; valid keys for {name!r} are: {valid}" if valid else ""
        for key in parsed:
            if not _HAREM_IDENT_RE.match(key):
                return (f"the argument key name was omitted -- {_harem_safe(key, 60)!r} "
                        f"is a VALUE written where the key name belongs" + tail)
            if properties and key not in properties:
                return (f"argument key {_harem_safe(key, 40)!r} is not in the tool "
                        f"schema{tail}")

        missing = sorted(required - set(parsed))
        if missing:
            return ("missing required key(s): " + ", ".join(missing) + tail)
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

    def _harem_resolve_name(self, name: str) -> str | None:
        """HAREM-GLM47-RETRY: the tool this rejection can be re-offered against, or None.

        Exact match first.  Failing that, the LEADING TOKEN: at depth the model
        sometimes collapses the whole call into the name slot, losing the
        <arg_key> structure entirely.  Captured in production 2026-09-12 as
            name='bash generating-docs-animations-properly,119-133'
        -- a real tool with a mangled sed range welded onto it.  The leading
        token is the tool it meant, and re-offering it is safe for the same
        reason the sentinel is: the call is GUARANTEED to be refused, so the
        worst case is one wasted round trip, never a wrong execution.

        A name with no valid leading token is a genuine hallucination: there is
        no call to make, so it stays content.
        """
        if not self._harem_retry or not self._tools:
            return None
        name = name.strip()
        if not name:
            return None
        if _HAREM_IDENT_RE.match(name) and find_tool_name(self._tools, name):
            return name
        # Ask which KNOWN tool this name begins with, longest first -- do NOT
        # take the longest identifier-shaped run. The run rule looks right and
        # fails on the shape that matters: captured in production as
        #   name='bash1635,1676p src/app/modules/admin/intake-comp'
        # where a sed range is welded on with no separator, so the run is
        # 'bash1635' (digits are identifier characters), which is not a tool,
        # and the turn died. Matching against the tool list instead resolves it
        # to 'bash'. Longest-first so 'readFile' wins over 'read'.
        for cand in sorted(self._harem_tool_names(), key=len, reverse=True):
            if name.startswith(cand):
                return cand
        return None

    def _harem_tool_names(self) -> list[str]:
        """HAREM-GLM47-RETRY: the request's tool names, dict or pydantic."""
        out: list[str] = []
        for tool in self._tools or ():
            fn = (tool.get("function") if isinstance(tool, dict)
                  else getattr(tool, "function", None))
            if fn is None:
                continue
            n = (fn.get("name") if isinstance(fn, dict)
                 else getattr(fn, "name", None))
            if isinstance(n, str) and n:
                out.append(n)
        return out

    def _harem_reject(self, idx: int, reason: str, closer: str,
                      deltas=None) -> None:
        """Reject the call: as a sentinel tool call when the name is valid
        (HAREM-GLM47-RETRY), otherwise as content (upstream behaviour)."""
        slot = self._tool_slots[idx]
        name = slot.name.strip()
        # HAREM-GLM47-RETRY: may differ from `name` when a valid tool name had garbage
        # welded onto it and only the leading token survived.
        call_name = self._harem_resolve_name(name)
        retryable = call_name is not None
        if retryable:
            # HAREM-GLM47-RETRY.  Arguments the client cannot execute and cannot
            # mistake for real ones; the schema rejects them, the model is told
            # why, and the agent loop lives to take another turn.
            detail = reason
            if call_name != name:
                # Say so in the arguments too: the model is the audience, and
                # "the name itself was malformed" is the thing it must fix.
                # _harem_safe, not name[:80]: the malformed name is MODEL OUTPUT
                # and regularly carries tag fragments -- 'bash@1</arg_value>'
                # was caught in the gate. Quoting that back into the history is
                # precisely the markup-imitation cascade this arm exists to stop.
                detail = (f"{reason}; the tool name was malformed "
                          f"({_harem_safe(name, 80)!r}) and was read as {call_name!r}")
            args_json = json.dumps({_HAREM_INVALID_KEY: detail},
                                   ensure_ascii=False)
            self._harem_rejected.append((call_name, args_json))
            self._ensure_tool_id(slot, call_name)
            call = DeltaToolCall(
                index=idx,
                id=slot.id,
                type="function",
                function=DeltaFunctionCall(name=call_name, arguments=args_json),
            )
            if deltas is not None:
                deltas.append(call)
            else:
                # HAREM-GLM47-RETRY: reached from _harem_finalize (a never-closed call),
                # which runs after the event loop and has no deltas list. Held
                # so finalize can put it on the streaming path itself --
                # otherwise a truncated call would be recorded for the
                # non-streaming extractor and silently dropped when streaming.
                self._harem_pending_stream.append(call)
        else:
            self._harem_reject_text.append(
                TOOL_CALL_START + slot.name + slot.args + closer
            )
        logger.warning(
            "[%s] rejected tool call idx=%d name=%r: %s (%s)",
            "''' + MARK + '''",
            idx,
            name[:48],
            reason,
            (f"retryable as {call_name!r} -> sentinel tool call"
             if retryable else "-> content"),
        )
        # A fresh slot is skipped by _build_extracted_result (no name, no args),
        # and on the content path no tool-call id was burned because
        # _ensure_tool_id never ran.
        self._tool_slots[idx] = type(slot)()

    def _build_extracted_result(self, *deltas):
        """HAREM-GLM47-RETRY: the NON-STREAMING half.

        The base iterates _tool_slots, and _harem_reject has already emptied
        the rejected one, so the sentinel calls are appended here instead.
        """
        result = super()._build_extracted_result(*deltas)
        if not (self._harem_failclosed and self._harem_rejected):
            return result
        for name, args_json in self._harem_rejected:
            slot = type(self._tool_slots[0])() if self._tool_slots else None
            if slot is not None:
                self._ensure_tool_id(slot, name)
                tool_id = slot.id
            else:
                tool_id = ""
            result.tool_calls.append(
                ToolCall(
                    id=tool_id,
                    function=FunctionCall(name=name, arguments=args_json),
                )
            )
        self._harem_rejected.clear()
        result.tools_called = len(result.tool_calls) > 0
        return result

    def _harem_finalize(self, delta, finished: bool):
        """Reject never-closed calls, then append rejected text as content."""
        if finished:
            for idx, slot in enumerate(self._tool_slots):
                if idx in self._harem_closed or (not slot.name and not slot.args):
                    continue
                self._harem_closed.add(idx)
                # HAREM-GLM47-RETRY: no deltas list here (finalize runs after the event
                # loop), so a retryable one is recorded for the non-streaming
                # extractor and, on the streaming path, emitted below.
                self._harem_reject(idx, "tool call never closed (truncated)",
                                   "", None)
        # HAREM-GLM47-RETRY: put any sentinel call raised by the truncation sweep above
        # onto the streaming path. _build_extracted_result reads _tool_slots and
        # delta CONTENT only -- never delta tool calls -- so the two paths stay
        # independent and this cannot double-emit on the non-streaming one.
        pending = self._harem_pending_stream
        if pending:
            self._harem_pending_stream = []
            if delta is None:
                delta = DeltaMessage(tool_calls=list(pending))
            elif delta.tool_calls:
                delta.tool_calls.extend(pending)
            else:
                delta.tool_calls = list(pending)
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
        self._harem_rejected.clear()          # HAREM-GLM47-RETRY
        self._harem_pending_stream.clear()    # HAREM-GLM47-RETRY
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
