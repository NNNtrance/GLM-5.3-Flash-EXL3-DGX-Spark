#!/usr/bin/env python3
"""Identify the chat template you are serving, by fingerprint.

docs/14 section 9.11 records that the served chat_template.jinja "matches
neither checkpoint on disk, and has never been verified against a named source"
-- the last open provenance item in the retraction audit. This settles the
operational half of it: it fingerprints a file against the known
zai-org/GLM-5.3-Flash revisions and tells you which one you have, what it
implies for multi-turn behaviour, and whether it carries the tool-calling fixes.

WHY A FINGERPRINT IS ENOUGH. Since the initial upload, zai-org/GLM-5.3-Flash has
only ever changed chat_template.jinja (plus README, LICENSE and eval results):
the 62 safetensors are bit-identical throughout and config.json,
generation_config.json, tokenizer.json, tokenizer_config.json and the weight
index have never moved. So template provenance can be settled by size and sha256
alone, with no tokenizer-compatibility question to answer.
Verified against the HF commit list and per-revision file hashes, 8 September
2026, by a second cluster [reported].

WHAT THE REVISIONS DIFFER ON, and it is not cosmetic:
  26 Aug c5b82b63 -- clear_thinking CLEARS by default (safe in multi-turn).
      This is the file turboderp/GLM-5.3-Flash-exl3 ships, so a stack that took
      its template from that checkpoint has the safe default by accident of date.
  27 Aug 04c4e9e9 -- clear_thinking RETAINS by default; adds the vision macros.
      local-inference-lab/GLM-5.3-Flash-NVFP4 ships this one verbatim.
  04 Sep 690b7052 -- still RETAINS; adds four tool-calling correctness fixes,
      the load-bearing one being `elif content is not none`, without which a
      literal "None" is emitted before every tool call on an assistant turn
      whose content is null -- most turns, in agentic use.

Usage:  python3 verify-chat-template.py [PATH]
        default: $CHAT_TEMPLATE_HOST, else ./chat_template.jinja
Exit 0 if the file matches a known revision, 1 if it does not (local edits are
normal -- the report says which base it is closest to), 2 if it is unreadable.
Written for this recipe by a second cluster; use freely (Apache-2.0).
"""
import hashlib
import os
import sys

KNOWN = {
    "41cff9af7b3a86c96751b107a8444f245fbda0bd5320b636a5bb1f7f4ba1a5c3": (
        "c5b82b63", "26 Aug 2026", 8617, "CLEARS", False),
    "34d5ee66b12fa6446cdae131c352b8f68cd85369e0e6fda115583805fada3891": (
        "04c4e9e9", "27 Aug 2026", 10644, "RETAINS", False),
    "0c4099f3382d6c92700dfb99725025360966fd73032f0ecf32377c0d9e6309c5": (
        "690b7052", "04 Sep 2026", 10950, "RETAINS", True),
}
# Sizes are a weaker but still useful signal when a file carries local edits.
SIZES = {8617: "c5b82b63 (26 Aug)", 10644: "04c4e9e9 (27 Aug)",
         10950: "690b7052 (04 Sep, current)"}


def main() -> int:
    path = (sys.argv[1] if len(sys.argv) > 1
            else os.environ.get("CHAT_TEMPLATE_HOST") or "chat_template.jinja")
    try:
        raw = open(path, "rb").read()
    except OSError as e:
        print(f"verify-chat-template: cannot read {path}: {e}")
        return 2
    sha = hashlib.sha256(raw).hexdigest()
    print(f"file   : {path}")
    print(f"bytes  : {len(raw)}")
    print(f"sha256 : {sha}")

    if sha in KNOWN:
        rev, date, _, retention, fixes = KNOWN[sha]
        print(f"\nEXACT MATCH: zai-org/GLM-5.3-Flash @ {rev} ({date}), unmodified.")
        print(f"  clear_thinking default : {retention}"
              + ("" if retention == "CLEARS" else
                 "  <-- pass clear_thinking:true in multi-turn"))
        print(f"  tool-calling fixes     : {'yes' if fixes else 'NO'}"
              + ("" if fixes else "  <-- emits a literal 'None' before tool calls"))
        return 0

    print("\nNo exact match: this file is not any published z.ai revision "
          "(local edits are normal).")
    near = SIZES.get(len(raw))
    if near:
        print(f"  size matches {near} exactly -- likely that base, unmodified in length.")
    else:
        closest = min(SIZES, key=lambda n: abs(n - len(raw)))
        print(f"  closest published size is {closest} B ({SIZES[closest]}), "
              f"delta {len(raw) - closest:+d} B.")

    # Behavioural read, which is what actually matters and does not need a hash.
    txt = raw.decode("utf-8", "replace")
    if "clear_thinking" not in txt:
        print("  clear_thinking : ABSENT -- this template has no retention control")
    elif "clear_thinking is defined else true" in txt:
        print("  clear_thinking : defaults TRUE (clears) -- safe in multi-turn")
    elif "clear_thinking is defined and not clear_thinking" in txt:
        print("  clear_thinking : 26-Aug form -- clears by default, safe in multi-turn")
    elif "clear_thinking is defined else false" in txt:
        print("  clear_thinking : defaults FALSE (RETAINS) -- pass clear_thinking:true, "
              "and note a per-request chat_template_kwargs block REPLACES engine "
              "defaults rather than merging")
    if "elif content is not none" in txt:
        print("  tool-call fix  : present (no 'None' leak)")
    elif "{%- else -%}" in txt and "tool_call" in txt:
        print("  tool-call fix  : LIKELY MISSING -- expect a literal 'None' before "
              "tool calls on null-content assistant turns")
    return 1


if __name__ == "__main__":
    sys.exit(main())
