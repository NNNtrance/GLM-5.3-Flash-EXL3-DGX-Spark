#!/usr/bin/env python3
"""Model-free gate for `patch-toolcall-autostrict-tp3.py`.

Proves the gate does exactly one thing: widen `tool_choice="auto"` to arm the
structural-tag grammar.  Everything else keeps upstream behaviour.

    gate unset / =0   -> upstream byte for byte: the grammar is discarded
    gate =1           -> the structural tag is BUILT for tool_choice="auto"
    tool_choice="none"-> still discarded with the gate ON (we only widen "auto")
    strict:true tool  -> still armed with the gate OFF (no regression)

Runs on CPU in the serving image; no weights, no engine, no GPU.

    docker run --rm --entrypoint bash \
      -v "$PWD":/work harem/exl3-zeus:754421f -lc '
        D=/usr/local/lib/python3.12/dist-packages
        python3 /work/patch-toolcall-autostrict-tp3.py --root $D --in-place
        python3 /work/test_toolcall_autostrict.py'

Exit 0 = all pass.  Run it against a THROWAWAY container: it patches the tree
in place.
"""
import os
import sys

sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionToolsParam
from vllm.tool_parsers.structural_tag_registry import get_model_structural_tag

MODEL = "glm_4_7"


def tool(strict: bool = False) -> ChatCompletionToolsParam:
    fn = {
        "name": "bash",
        "description": "run a shell command",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    }
    if strict:
        fn["strict"] = True
    return ChatCompletionToolsParam.model_validate({"type": "function", "function": fn})


PLAIN = [tool()]
STRICT = [tool(strict=True)]


def check(label, tools, tool_choice, want_built):
    built = get_model_structural_tag(MODEL, tools, tool_choice, False) is not None
    ok = built == want_built
    print(f"  {'PASS' if ok else 'FAIL'} {label:<48} tag={'BUILT' if built else 'None'}")
    return ok


def main() -> int:
    results = []

    os.environ.pop("HAREM_TOOLCALL_AUTOSTRICT", None)
    results.append(check("gate unset -> upstream (grammar discarded)", PLAIN, "auto", False))

    os.environ["HAREM_TOOLCALL_AUTOSTRICT"] = "0"
    results.append(check("gate =0 -> upstream (grammar discarded)", PLAIN, "auto", False))

    os.environ["HAREM_TOOLCALL_AUTOSTRICT"] = "1"
    results.append(check("gate =1 -> GRAMMAR ARMED", PLAIN, "auto", True))
    results.append(check("gate =1, tool_choice=none -> discarded", PLAIN, "none", False))

    os.environ["HAREM_TOOLCALL_AUTOSTRICT"] = "0"
    results.append(check("strict:true tool, gate off -> armed (no regression)", STRICT, "auto", True))

    print("ALL-PASS" if all(results) else "SOME-FAILED")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
