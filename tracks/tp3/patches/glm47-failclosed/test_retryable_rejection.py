#!/usr/bin/env python3
"""HAREM_GLM47_RETRY gate: a rejected tool call must come back RETRYABLE, not as prose.

Model-free. Run it inside a THROWAWAY container built from the EXL3 image, with
the model sidecar IDENTITY-MOUNTED (it is a relative-symlink tree; mounting it
anywhere else dangles every link and the tokenizer will not load):

  docker run --rm --entrypoint bash \
    -v $MODEL_HOST_PATH:$MODEL_HOST_PATH:ro \
    -v $MODEL_LINK_TARGET:$MODEL_LINK_TARGET:ro \
    -v $PWD:/work <image> -lc '
      D=/usr/local/lib/python3.12/dist-packages
      python3 /work/patch-glm47-failclosed-tp3.py --root $D --in-place
      python3 /work/test_retryable_rejection.py'

NEVER run it in the live container -- importing vllm there is a known hazard.

WHY THIS EXISTS. Upstream surfaces a rejected call as CONTENT, which ends the
agent's turn: every agent loop reads "text, no tool call" as "finished". Case 1
below is the production failure of 2026-09-12 verbatim -- name='bash' with an
argument key 'review_expr' that is not in bash's schema.

Cases 1/4/5 must return a CALL carrying __harem_invalid_tool_call__ (the client's
schema rejects it, the model is told why, the loop continues). Case 3 must fall
back to content: a hallucinated NAME has no call to make. Cases 2/6 prove the
happy paths are untouched, and the last case proves HAREM_GLM47_RETRY=0 restores
upstream byte for byte.
"""
import json, sys
from types import SimpleNamespace
from transformers import AutoTokenizer
from vllm.parser.glm47_moe import Glm47MoeParser
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionToolsParam

TOK = AutoTokenizer.from_pretrained("$MODEL_HOST_PATH", trust_remote_code=True)
TOOLS = [ChatCompletionToolsParam.model_validate({
  "type":"function","function":{"name":"bash","description":"run",
   "parameters":{"type":"object",
     "properties":{"command":{"type":"string"},"description":{"type":"string"}},
     "required":["command"]}}})]

def run(text):
    p = Glm47MoeParser(TOK, tools=TOOLS)
    req = SimpleNamespace(tools=TOOLS, tool_choice="auto", chat_template_kwargs={}, messages=[])
    return p.extract_tool_calls(text, req)

def show(label, text, want_call, want_sentinel):
    try:
        r = run(text)
    except Exception as e:
        print(f"  ERROR {label}: {type(e).__name__}: {e}"); return False
    names = [t.function.name for t in (r.tool_calls or [])]
    args  = [t.function.arguments for t in (r.tool_calls or [])]
    sent  = any("__harem_invalid_tool_call__" in a for a in args)
    ok = (bool(names) == want_call) and (sent == want_sentinel)
    print(f"  {'PASS' if ok else 'FAIL'} {label}")
    for a in args:
        # closing tags too -- '</arg_value>' does not contain '<arg_value>',
        # which is how a real leak passed this check the first time.
        for bad in ("<arg_key", "<arg_value", "<tool_call", "</arg", "</tool"):
            if bad in a:
                print(f"       !! MARKUP LEAKED INTO ARGS: {bad}")
                ok = False
    print(f"       tool_calls={names}")
    for a in args:
        print(f"       args={a}")
    print(f"       content={repr((r.content or '')[:70])}")
    return ok

TS, TE = "<tool_call>", "</tool_call>"
res = []
res.append(show("PROD FAILURE: bad arg key -> retryable sentinel",
  f"{TS}bash<arg_key>review_expr</arg_key><arg_value>none</arg_value>{TE}", True, True))
res.append(show("valid call -> real call, untouched",
  f"{TS}bash<arg_key>command</arg_key><arg_value>ls -la</arg_value>{TE}", True, False))
# DELIBERATE CHANGE 2026-09-12: a name that merely STARTS with a real tool
# ('bashhhh') now resolves to that tool instead of falling to content. The
# sentinel guarantees the call is refused either way, so the real choice is
# between a retryable error and a dead turn -- and a dead turn is the thing this
# whole arm exists to prevent. Mislabelling costs one wasted round trip.
res.append(show("near-miss name ('bashhhh') -> retryable, NOT a dead turn",
  f"{TS}bashhhh<arg_key>command</arg_key><arg_value>ls</arg_value>{TE}", True, True))
res.append(show("missing required key -> retryable sentinel",
  f"{TS}bash<arg_key>description</arg_key><arg_value>x</arg_value>{TE}", True, True))
res.append(show("never-closed (truncated) -> retryable sentinel",
  f"{TS}bash<arg_key>command</arg_key><arg_value>python -c \"print(1)\"", True, True))
res.append(show("plain prose -> untouched", "The answer is 42.", False, False))

# Captured in production 2026-09-12: at depth the model collapses the WHOLE call
# into the name slot, losing <arg_key> entirely. The leading token is still the
# tool it meant, so this must be retryable, not a dead turn.
res.append(show("PROD FAILURE 2: garbage welded onto a valid name -> retryable",
  f"{TS}bash generating-docs-animations-properly,119-133{TE}", True, True))
res.append(show("leading token valid, trailing junk + args -> retryable",
  f"{TS}bash -n '500,565p' foo.py<arg_key>command</arg_key><arg_value>ls</arg_value>{TE}",
  True, True))
# The counter-case: no VALID leading token is a real hallucination, and there is
# no call to make. Must stay content -- salvaging here would invent a tool.
res.append(show("no valid leading token -> still content",
  f"{TS}zzzznotatool something{TE}", False, False))

# Captured 2026-09-12 16:10:58 -- this one KILLED A TURN on the first salvage
# rule. A sed range welded on with NO separator: the longest identifier-shaped
# run is 'bash1635' (digits are identifier chars), which is not a tool. Matching
# against the tool LIST instead resolves it to 'bash'.
res.append(show("PROD FAILURE 3: digits welded on, no separator -> retryable",
  f"{TS}bash1635,1676p src/app/modules/admin/intake-comp{TE}", True, True))
res.append(show("punctuation-separated junk (already worked) -> retryable",
  f"{TS}bash@1</arg_value>{TE}", True, True))
# Longest-first must win: with both 'bash' and 'bash_profile' offered, a name
# beginning 'bash_profile...' must resolve to 'bash_profile', not to 'bash'.
TOOLS.append(ChatCompletionToolsParam.model_validate({
  "type":"function","function":{"name":"bash_profile","description":"x",
   "parameters":{"type":"object","properties":{"p":{"type":"string"}}}}}))
def resolved_name(text):
    p = Glm47MoeParser(TOK, tools=TOOLS)
    req = SimpleNamespace(tools=TOOLS, tool_choice="auto", chat_template_kwargs={}, messages=[])
    r = p.extract_tool_calls(text, req)
    return [t.function.name for t in (r.tool_calls or [])]
got = resolved_name(f"{TS}bash_profile,1676p junk{TE}")
ok = got == ["bash_profile"]
print(f"  {'PASS' if ok else 'FAIL'} longest-first: 'bash_profile,1676p junk' -> {got} (want ['bash_profile'])")
res.append(ok)

import os
print("  --- gate off (HAREM_GLM47_RETRY=0) must restore upstream content-only ---")
os.environ["HAREM_GLM47_RETRY"]="0"
import importlib, vllm.parser.glm47_moe as g
importlib.reload(g)
from vllm.parser.glm47_moe import Glm47MoeParser as P2
p = P2(TOK, tools=TOOLS)
req = SimpleNamespace(tools=TOOLS, tool_choice="auto", chat_template_kwargs={}, messages=[])
r = p.extract_tool_calls(f"{TS}bash<arg_key>review_expr</arg_key><arg_value>none</arg_value>{TE}", req)
off_ok = not r.tool_calls
print(f"  {'PASS' if off_ok else 'FAIL'} gate=0 -> tool_calls={[t.function.name for t in (r.tool_calls or [])]} content={repr((r.content or '')[:60])}")
res.append(off_ok)

print("ALL-PASS" if all(res) else "SOME-FAILED")
sys.exit(0 if all(res) else 1)
