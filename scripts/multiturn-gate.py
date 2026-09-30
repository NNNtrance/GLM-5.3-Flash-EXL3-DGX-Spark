#!/usr/bin/env python3
"""Multi-turn agentic gate — the symptom class a single-turn gate cannot see.

Contributed by @jdecker76 (see results/community/jdecker76/multiturn-gate/).
Written for this recipe by a second cluster; use freely (Apache-2.0).

WHY THIS EXISTS. Every gate in this repository is single-turn: the correctness
probe, the code exam, the tool-call gate. We ran a GLM-5.3-Flash stack in
production on real multi-user agentic traffic that passed all of them — 10/10,
12/12, 8/8, cold and warm — while users hit looping, premature end-of-turn and
degraded long-session quality badly enough that we rolled the engine back. The
whole symptom family lives in WARM MULTI-TURN sessions, so no single-turn gate
could ever have seen it. This is the gate we wish had existed.

It checks three things a single-turn gate structurally cannot:

  A. REASONING RETENTION (differential).  Agent clients echo the previous
     assistant turn back verbatim, `<think>...</think>` included, inside
     `content`.  The GLM chat template EXTRACTS that back out into
     reasoning_content and re-renders it into every later prompt unless
     `clear_thinking:true` is passed -- and z.ai's official template has
     defaulted to RETAIN since 2026-08-27.  Retention compounds every turn and
     is a self-priming loop attractor.
     The test: run the SAME conversation twice, once echoing assistant content
     verbatim (what real clients send) and once with <think> blocks stripped by
     us.  If the server retains, arm A's prompt_tokens grow far faster than
     arm B's.  Engine-agnostic: it measures rendered prompt size, not config.

  B. LOOPING / degenerate repetition inside a single response.

  C. EARLY STOP: finish_reason == "stop" at an implausibly short length on a
     prompt that explicitly asks for a long answer.

Exit code 0 = pass, 1 = failure, 2 = could not run.  Stdlib only: this has to
run on a Spark node with no venv.

  python3 multiturn-gate.py [API_BASE]      default http://192.0.2.10:8001, or $API
  python3 multiturn-gate.py --base http://192.0.2.10:8001 --think high
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request

# Turns are ordinary agentic work: each asks for something substantive, so a
# short "stop" is a real signal rather than a terse-but-correct answer.
TURNS = [
    "Write a Python function `parse_duration(s)` that turns '1h30m', '45s' and "
    "'2d4h' into seconds. Include the code and a one-line explanation.",
    "Now add validation: raise ValueError with a clear message on malformed input. "
    "Show the updated function.",
    "Write five pytest test cases for it, including two failure cases.",
    "Refactor it to support fractional units like '1.5h'. Show the diff-relevant parts.",
    "Summarise the final design in 3 bullet points, then list any edge cases still unhandled.",
]

THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def post(base: str, body: dict, timeout: int) -> dict:
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def degenerate_repeat(text: str) -> tuple[int, str]:
    """Longest run of a repeated non-trivial line, and the line."""
    lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) > 12]
    best, best_line, run, prev = 1, "", 1, None
    for ln in lines:
        if ln == prev:
            run += 1
            if run > best:
                best, best_line = run, ln
        else:
            run, prev = 1, ln
    # also catch a phrase repeated many times without newlines
    for m in re.finditer(r"(.{25,120}?)\1{3,}", text, re.S):
        n = len(m.group(0)) // max(len(m.group(1)), 1)
        if n > best:
            best, best_line = n, m.group(1).strip()[:80]
    return best, best_line


def run_arm(base, model, timeout, max_tokens, kwargs, strip_think):
    """One conversation. Returns per-turn records; echoes assistant turns back
    the way a real agent client does."""
    history, recs = [], []
    for i, prompt in enumerate(TURNS):
        history.append({"role": "user", "content": prompt})
        body = {"model": model, "messages": history, "max_tokens": max_tokens}
        if kwargs is not None:
            body["chat_template_kwargs"] = kwargs
        d = post(base, body, timeout)
        ch = d["choices"][0]
        msg = ch.get("message", {})
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        usage = d.get("usage", {})
        # Reconstruct what a real client echoes back: the model's own output,
        # thinking included, in `content`.
        echoed = (f"<think>{reasoning}</think>{content}" if reasoning else content)
        if strip_think:
            echoed = THINK_RE.sub("", echoed)
        history.append({"role": "assistant", "content": echoed})
        recs.append({
            "turn": i + 1,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "finish": ch.get("finish_reason"),
            "content": content,
        })
    return recs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base", nargs="?", default=None,
                    help="API base, positional (their other scripts take it this way)")
    ap.add_argument("--base", dest="base_opt", default=None)
    ap.add_argument("--model", default="glm-5.3-flash")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--max-tokens", type=int, default=1400)
    ap.add_argument("--think", default="low",
                    help="reasoning_effort to send, or 'none' to send no kwargs")
    ap.add_argument("--retention-ratio", type=float, default=1.35,
                    help="FAIL if verbatim-echo prompt growth exceeds stripped-echo "
                         "growth by more than this factor")
    ap.add_argument("--min-stop-tokens", type=int, default=120,
                    help="a 'stop' finish shorter than this is a suspected early stop")
    ap.add_argument("--max-repeat", type=int, default=4,
                    help="FAIL if any response repeats a line/phrase more than this")
    a = ap.parse_args()
    a.base = a.base_opt or a.base or os.environ.get("API", "http://192.0.2.10:8001")

    kwargs = None if a.think == "none" else {"reasoning_effort": a.think}

    print(f"multiturn gate: {a.model} @ {a.base}  turns={len(TURNS)}  effort={a.think}")
    try:
        print("  arm A: echoing assistant turns VERBATIM (real client behaviour)")
        arm_a = run_arm(a.base, a.model, a.timeout, a.max_tokens, kwargs, strip_think=False)
        print("  arm B: echoing with <think> stripped by us (control)")
        arm_b = run_arm(a.base, a.model, a.timeout, a.max_tokens, kwargs, strip_think=True)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, KeyError) as e:
        print(f"  COULD NOT RUN: {type(e).__name__}: {e}")
        return 2

    failures: list[str] = []

    # ---- A. retention -----------------------------------------------------
    # Compare growth from the first to the last turn. Arm A carries the model's
    # own thinking back in; arm B does not. If the server re-renders retained
    # reasoning, A's prompt grows disproportionately.
    ga = arm_a[-1]["prompt_tokens"] - arm_a[0]["prompt_tokens"]
    gb = arm_b[-1]["prompt_tokens"] - arm_b[0]["prompt_tokens"]
    ratio = (ga / gb) if gb > 0 else float("inf") if ga > 0 else 1.0
    print("\n  prompt_tokens by turn")
    print("    verbatim :", [r["prompt_tokens"] for r in arm_a])
    print("    stripped :", [r["prompt_tokens"] for r in arm_b])
    print(f"    growth   : verbatim +{ga}, stripped +{gb}, ratio {ratio:.2f} "
          f"(fail > {a.retention_ratio})")
    if ratio > a.retention_ratio:
        failures.append(
            f"RETENTION: verbatim-echo prompt growth is {ratio:.2f}x the stripped "
            f"control (+{ga} vs +{gb} tokens). Prior reasoning is being re-rendered "
            f"into later prompts — pass clear_thinking:true in EVERY variant "
            f"(per-request chat_template_kwargs REPLACE engine defaults)."
        )

    # ---- B. looping + C. early stop --------------------------------------
    print("\n  per-turn responses")
    for arm_name, arm in (("verbatim", arm_a), ("stripped", arm_b)):
        for r in arm:
            n, line = degenerate_repeat(r["content"])
            flag = ""
            if n > a.max_repeat:
                flag += " LOOP"
                failures.append(
                    f"LOOPING: {arm_name} turn {r['turn']} repeats a line/phrase "
                    f"{n}x: {line[:60]!r}"
                )
            if r["finish"] == "stop" and r["completion_tokens"] < a.min_stop_tokens:
                flag += " EARLY-STOP"
                failures.append(
                    f"EARLY STOP: {arm_name} turn {r['turn']} finished 'stop' after only "
                    f"{r['completion_tokens']} completion tokens on a prompt that asks "
                    f"for substantive output."
                )
            if r["finish"] not in ("stop", "length"):
                failures.append(
                    f"FINISH: {arm_name} turn {r['turn']} finish_reason={r['finish']!r}")
            print(f"    {arm_name:8s} t{r['turn']}  in={r['prompt_tokens']:6d} "
                  f"out={r['completion_tokens']:5d} finish={str(r['finish']):6s}"
                  f" maxrepeat={n}{flag}")

    print()
    if failures:
        print(f"MULTITURN GATE: FAIL ({len(failures)})")
        for f in failures:
            print("  - " + f)
        return 1
    print("MULTITURN GATE: PASS  (retention bounded, no degenerate repetition, "
          "no early stops)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
