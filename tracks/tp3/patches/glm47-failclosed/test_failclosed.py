#!/usr/bin/env python3
"""Unit tests for patch-glm47-failclosed-tp3.py (issue #7).

CPU only, no model, no engine.  Run inside the production image on a node:

    # copy patch-glm47-failclosed-tp3.py and test_failclosed.py to /tmp/issue7
    # on any node of the deployment, then:
    docker run --rm -v /tmp/issue7:/w --entrypoint bash \\
        exl3-zeus:754421f -c "
          python3 /w/patch-glm47-failclosed-tp3.py \\
              --root /usr/local/lib/python3.12/dist-packages --in-place &&
          python3 /w/test_failclosed.py"

The container is throwaway (--rm), so the in-place patch dies with it.

The corruption vectors are the ones captured live and quoted in issue #7.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import types

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionToolsParam


def _tool(name: str, props: dict) -> ChatCompletionToolsParam:
    return ChatCompletionToolsParam(
        type="function",
        function={
            "name": name,
            "description": name,
            "parameters": {"type": "object", "properties": props},
        },
    )


TOOLS = [
    _tool(
        "Bash",
        {
            "command": {"type": "string"},
            "description": {"type": "string"},
            "timeout": {"type": "integer"},
        },
    ),
    _tool("Read", {"file_path": {"type": "string"}}),
]

REQUEST = types.SimpleNamespace(tools=TOOLS, tool_choice="auto")


class FakeTokenizer:
    """Enough tokenizer for the parser engine: text lexing only."""

    def get_vocab(self) -> dict[str, int]:
        return {}


# ── vectors ───────────────────────────────────────────────────────────────────

VALID = (
    "<think>plan</think>"
    "<tool_call>Bash"
    "<arg_key>command</arg_key><arg_value>ls -la /mnt/depo</arg_value>"
    "<arg_key>description</arg_key><arg_value>List the depot</arg_value>"
    "</tool_call>"
)

# issue #7, captured: </arg_value> arrives where </arg_key> belongs, so the
# regex swallows a key that is really key+tag+key.
GARBAGE_KEY_1 = (
    "<think>x</think>"
    "<tool_call>Bash"
    "<arg_key>S1_latest.json</arg_value>"
    "<arg_key>description</arg_key><arg_value>print it</arg_value>"
    "</tool_call>"
)

# issue #7, captured: the model falls back to Anthropic-style syntax mid-call.
GARBAGE_KEY_2 = (
    "<tool_call>Bash"
    "<arg_key>print_code_snapshot</arg_value><arg_key>description</arg_key>"
    "<arg_value>Print full source of multi_zones.py</arg_value>"
    "<parameter name=\"command\">echo hi</invoke>"
    "</tool_call>"
)

# issue #7, captured: the turn that came back empty -- the call never closes.
UNCLOSED = (
    "<think>y</think>"
    "<tool_call>Bash"
    "<arg_key>command</arg_key><arg_value>for $y in [29000, 29600]"
)

# issue #7, captured: the name itself is prose, no <arg_key> ever arrives.
HALLUCINATED_NAME = (
    "<tool_call>Bfor $y in [29000, 29600] # sanity check band summary\n"
    "Bash(command=...</tool_call>"
)

PLAIN_TEXT = "<think>thinking</think>Here is the answer: 42."

NO_ARG_CALL = "<tool_call>Read</tool_call>"

OFF_SCHEMA_KEY = (
    "<tool_call>Bash"
    "<arg_key>commnad</arg_key><arg_value>ls</arg_value>"
    "</tool_call>"
)


# ── harness ───────────────────────────────────────────────────────────────────

FAILS: list[str] = []
CHECKS = [0]


def check(cond: bool, label: str) -> None:
    CHECKS[0] += 1
    if cond:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label}")
        FAILS.append(label)


def load_parser_module(failclosed: str | None):
    """(Re)import vllm.parser.glm47_moe with the gate set to *failclosed*."""
    if failclosed is None:
        os.environ.pop("HAREM_GLM47_FAILCLOSED", None)
    else:
        os.environ["HAREM_GLM47_FAILCLOSED"] = failclosed
    mod = importlib.import_module("vllm.parser.glm47_moe")
    mod = importlib.reload(mod)
    if hasattr(mod.glm47_moe_config, "cache_clear"):
        mod.glm47_moe_config.cache_clear()
    return mod


def new_parser(mod):
    return mod.Glm47MoeParser(FakeTokenizer(), list(TOOLS))


def nonstream(mod, text):
    """extract_tool_calls: returns (tool_calls, content)."""
    res = new_parser(mod).extract_tool_calls(text, REQUEST)
    calls = [(tc.function.name, tc.function.arguments) for tc in res.tool_calls]
    return calls, (res.content or "")


def stream(mod, text, chunk=7):
    """Feed *text* in chunks; returns (tool_call_deltas, content, n_partial).

    ``n_partial`` counts deltas that carry a tool-call fragment without the
    closing arguments -- i.e. anything the client could act on too early.
    """
    p = new_parser(mod)
    deltas = []
    prev = ""
    for i in range(0, len(text), chunk):
        piece = text[i : i + chunk]
        d = p.extract_tool_calls_streaming(
            previous_text=prev,
            current_text=prev + piece,
            delta_text=piece,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=REQUEST,
        )
        prev += piece
        if d is not None:
            deltas.append(d)
    d = p.finish_streaming()
    if d is not None:
        deltas.append(d)
    tool_deltas = [d for d in deltas if getattr(d, "tool_calls", None)]
    content = "".join(d.content for d in deltas if getattr(d, "content", None))
    return tool_deltas, content, deltas


# ── tests ─────────────────────────────────────────────────────────────────────


def test_failclosed_on(mod) -> None:
    print("\n[1] non-streaming, gate ON")

    calls, content = nonstream(mod, VALID)
    check(len(calls) == 1, "valid call -> exactly one tool call")
    if calls:
        name, args = calls[0]
        check(name == "Bash", "valid call -> name Bash")
        try:
            parsed = json.loads(args)
        except ValueError:
            parsed = None
        check(
            parsed == {"command": "ls -la /mnt/depo", "description": "List the depot"},
            "valid call -> schema-correct JSON arguments",
        )
    check("<tool_call>" not in content, "valid call -> no raw XML in content")

    for label, vector in (
        ("garbage key #1", GARBAGE_KEY_1),
        ("garbage key #2", GARBAGE_KEY_2),
        ("unclosed call", UNCLOSED),
        ("hallucinated name", HALLUCINATED_NAME),
        ("off-schema key", OFF_SCHEMA_KEY),
    ):
        calls, content = nonstream(mod, vector)
        check(not calls, f"{label} -> no tool call emitted")
        check("<tool_call>" in content, f"{label} -> raw text surfaced as content")

    calls, content = nonstream(mod, PLAIN_TEXT)
    check(not calls, "plain text -> no tool call")
    check(content.strip() == "Here is the answer: 42.", "plain text -> content intact")

    calls, content = nonstream(mod, NO_ARG_CALL)
    check(
        calls == [("Read", "{}")],
        "no-argument call -> passes with empty arguments",
    )


def test_streaming_on(mod) -> None:
    print("\n[2] streaming, gate ON")

    tool_deltas, content, _ = stream(mod, VALID)
    check(len(tool_deltas) == 1, "valid call -> exactly one tool-call delta")
    if tool_deltas:
        tc = tool_deltas[0].tool_calls[0]
        check(tc.function.name == "Bash", "valid call -> name in that delta")
        args = tc.function.arguments or ""
        check(
            json.loads(args)
            == {"command": "ls -la /mnt/depo", "description": "List the depot"},
            "valid call -> complete arguments in that single delta",
        )

    for label, vector in (
        ("garbage key #1", GARBAGE_KEY_1),
        ("garbage key #2", GARBAGE_KEY_2),
        ("unclosed call", UNCLOSED),
        ("hallucinated name", HALLUCINATED_NAME),
    ):
        tool_deltas, content, _ = stream(mod, vector)
        check(not tool_deltas, f"{label} -> streaming emits nothing partial")
        check("<tool_call>" in content, f"{label} -> streamed back as content")


def test_never_closed_sweep(mod) -> None:
    """White-box: the defensive sweep for a slot that never got a TOOL_CALL_END.

    In practice StreamingParserEngine.finish() synthesises a TOOL_CALL_END for
    an open call (streaming_parser_engine.py:262-275) and validation catches it
    as truncated, so this branch is depth rather than the usual route.  Test it
    anyway so it is not unverified code.
    """
    print("\n[2b] streaming, gate ON -- never-closed sweep")
    p = new_parser(mod)
    p.extract_tool_calls_streaming(
        previous_text="",
        current_text=UNCLOSED,
        delta_text=UNCLOSED,
        previous_token_ids=[],
        current_token_ids=[],
        delta_token_ids=[],
        request=REQUEST,
    )
    check(
        bool(p._tool_slots) and 0 not in p._harem_closed,
        "slot is open and unclosed before the sweep",
    )
    delta = p._harem_finalize(None, True)
    check(delta is not None and "<tool_call>" in (delta.content or ""),
          "sweep surfaces the never-closed call as content")
    check(
        not p._tool_slots[0].name and not p._tool_slots[0].args,
        "sweep clears the slot so _build_extracted_result skips it",
    )


def test_failclosed_off(mod_off) -> None:
    print("\n[3] gate off (HAREM_GLM47_FAILCLOSED=0) -> upstream behaviour")

    check(
        mod_off.glm47_moe_config(thinking=True).stream_arg_deltas is True,
        "stream_arg_deltas back to True",
    )

    calls, _ = nonstream(mod_off, VALID)
    check(len(calls) == 1 and calls[0][0] == "Bash", "valid call still works")

    # The bug, reproduced: upstream salvages the garbage key into a real
    # argument name instead of refusing the call.
    calls, _ = nonstream(mod_off, GARBAGE_KEY_1)
    keys = sorted(json.loads(calls[0][1])) if calls else []
    check(
        len(calls) == 1 and any("</arg_value>" in k or "arg_key" in k for k in keys),
        "garbage key is salvaged again (upstream bug reproduced)",
    )

    calls, _ = nonstream(mod_off, UNCLOSED)
    check(len(calls) == 1, "unclosed call is emitted again (upstream bug reproduced)")

    tool_deltas, _, _ = stream(mod_off, VALID)
    check(len(tool_deltas) > 1, "arguments stream as several deltas again")


def main() -> int:
    mod_on = load_parser_module(None)  # unset -> default ON
    check(
        mod_on.glm47_moe_config(thinking=True).stream_arg_deltas is False,
        "default (unset env) is fail-closed",
    )
    test_failclosed_on(mod_on)
    test_streaming_on(mod_on)
    test_never_closed_sweep(mod_on)

    mod_off = load_parser_module("0")
    test_failclosed_off(mod_off)

    print(f"\n{CHECKS[0] - len(FAILS)}/{CHECKS[0]} checks passed")
    if FAILS:
        print("FAILED:")
        for f in FAILS:
            print(f"  - {f}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
