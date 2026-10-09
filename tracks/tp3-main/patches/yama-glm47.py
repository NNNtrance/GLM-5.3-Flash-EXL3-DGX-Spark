#!/usr/bin/env python3
"""HAREM-GLM47-FAILCLOSED: the GLM-5.3-Flash tool-call parser fails closed.

The 21d93d0d8 port of the previous stack's patch-glm47-failclosed-tp3.py (11 September,
issue #7). Six anchors out of six hold unchanged in the new tree, BUT the previous patch
took ``DeltaMessage`` from ``vllm.entrypoints.openai.engine.protocol``, which has been
deleted: applied as it was, the parser would crash at import. Only A2 changed in this port
(the new path is ``vllm.entrypoints.generate.base.protocol``; the module already defines
``logger``, so the second definition was dropped). A1 and A3-A6 are BYTE FOR BYTE the
previous text; the behaviour is the same.

Behaviour (HAREM_GLM47_FAILCLOSED, default ON; =0 is exactly upstream): when a tool call
closes it is validated whole (the name is identifier-shaped and among the request's tools;
the keys are in the schema; with HAREM_GLM47_REQUIRED on, the required keys are present);
a rejected call falls to CONTENT as raw text and no tool call is made; the argument stream
is buffered (a half-finished call never reaches the client).

NOTE (this port keeps the previous behaviour; it is a design decision and was NOT changed
here): a proposal in this repository's PR #11 argues that dropping a rejection to content
ends an agent loop, and turns the rejection into a sentinel-keyed tool call instead. The
production behaviour was ported as it was.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harem_yama as Y  # noqa: E402

REL = "parser/glm47_moe.py"
MARK = "HAREM-GLM47-FAILCLOSED"

# --- anchor 1: stdlib imports ------------------------------------------------
A1_OLD = "import functools\nimport json\n"
A1_NEW = "import functools\nimport json\nimport os  # HAREM-GLM47-FAILCLOSED\n"

# --- anchor 2: vllm imports ---------------------------------------------------
# 21d93d0d8: DeltaMessage moved to vllm.entrypoints.generate.base.protocol (the
# old vllm.entrypoints.openai.engine.protocol module is gone, so production's
# import would crash the parser at import); the module already defines `logger`.
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

# HAREM-GLM47-FAILCLOSED.  Both already sit in this module's import graph
# (parser_engine imports them), so there is no new cycle.
from vllm.entrypoints.generate.base.protocol import DeltaMessage
from vllm.tool_parsers.utils import find_tool_name, find_tool_properties
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

D = Y.Duzen
YAMA = Y.Yama(
    ad="yama-glm47",
    aciklama="glm47 araç çağrısı ayrıştırıcısı fail-closed (issue #7)",
    duzenler=[
        D("A1-stdlib", REL, A1_OLD, A1_NEW),
        D("A2-vllm-import", REL, A2_OLD, A2_NEW),
        D("A3-kapi", REL, A3_OLD, A3_NEW),
        D("A4-config-imza", REL, A4_OLD, A4_NEW),
        D("A5-arg-akisi", REL, A5_OLD, A5_NEW),
        D("A6-ezmeler", REL, A6_OLD, A6_NEW),
    ],
    taban={REL: "3de35febe004c85fe5078123cbd25b3c57f7feafce1dde4e4b457f092e31af59"},
)

if __name__ == "__main__":
    sys.exit(Y.main(YAMA, __file__))
