#!/usr/bin/env python3
"""longctx-copy-fidelity.py — synthetic long-context tool-argument copy-fidelity probe.

Reproduces the shape of the defect in docs/14-troubleshooting.md §9.15 and
results/gates/index-topk-8192-12sep.md without any private data: past roughly 32k
tokens of agentic context, a sparse-attention engine can mis-copy long near-duplicate
path strings inside tool-call arguments — a single character slips
(`/srv/.../scripts/player_controller.gd` -> `/srv/.../scripts/playr_controller.gd`),
prose gets spliced onto the end of a path that started out correct, or the model
reaches for a look-alike path that only ever appeared in an error message.

Method. A deterministic, seeded generator builds a fake Godot-style project under a
fictional root (default `/srv/work/projects/alpha-widget`) with ~40 real, near-duplicate
file paths (a `_controller.gd` and a `_controller_test.gd` per module) and about 15
decoy siblings that exist only inside simulated "file not found" tool errors. It then
fabricates a plausible agent transcript — read_file / search_files / write_file /
terminal calls and their results — until the prompt reaches a target token count
(chars / 3.6, `--tokens`, default 35000). From there it drives `--turns` (default 30)
real turns against a live OpenAI-compatible chat endpoint: each turn asks for a specific
next step that can only be done correctly by copying one exact path out of the context
byte-for-byte, without restating that path in the prompt. The model's real response
(including its tool_calls) is appended to the conversation, plus a synthetic tool
result, so the context keeps growing like a real session. Every tool call's path-like
arguments are then classified as EXACT, SLIP (Levenshtein distance 1-3 from a real
path), SPLICE (a real or decoy path with extra text glued on), DECOY (a path that only
ever appeared in an error message), or OTHER, plus JSON-invalid arguments and
finish_reason == "length" runaways.

This is a measurement, not a gate: it always exits 0 unless the endpoint cannot be
reached at all for the first request.

Stdlib only (urllib, json, argparse, random, difflib, plus the usual re/os/time/
statistics/dataclasses/uuid). No third-party dependencies. Python 3.10+.

Usage (defaults match the production engine's own API shape):
    ./longctx-copy-fidelity.py --url http://localhost:8001/v1 --turns 30 --tokens 35000

Usage (the two arms this repository's §9.15 entry compares):
    ./longctx-copy-fidelity.py --turns 30 --tokens 35000 --strict --effort low \\
        --cache-salt fresh --extra-body '{"chat_template_kwargs":{"clear_thinking":true}}' \\
        --out fresh-run.json
    ./longctx-copy-fidelity.py --turns 30 --tokens 35000 --strict --effort low \\
        --cache-salt session --extra-body '{"chat_template_kwargs":{"clear_thinking":true}}' \\
        --out session-run.json

Written by us for this recipe; use freely (Apache-2.0).
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import random
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field

# --------------------------------------------------------------------------------------
# 1. The fake project: a fictional root, ~40 real near-duplicate paths, ~15 decoys
# --------------------------------------------------------------------------------------

ROOT = "/srv/work/projects/alpha-widget"

MODULES = [
    "player", "enemy", "inventory", "camera", "dialogue", "save", "audio",
    "physics", "ui_hud", "level_loader", "pathfinding", "particle_fx",
    "score_tracker", "input_map", "quest_log", "achievement",
    "settings_menu", "combo_system", "status_effect", "counter",
]

# Hand-written decoys: each is one character or one path segment away from a real
# sibling (projects/project, scripts/script, controller/controler, hyphen/underscore,
# an added letter, a stray extension). None of these ever exist; they appear only
# inside simulated "not found" tool errors in the fabricated transcript.
DECOYS = [
    "/srv/work/project/alpha-widget/scripts/player_controller.gd",
    "/srv/work/projects/alpha_widget/scripts/enemy_controller.gd",
    "/srv/work/projects/alpha-widget/script/inventory_controller.gd",
    "/srv/work/projects/alpha-widget/scripts/camera_controler.gd",
    "/srv/work/projects/alpha-widget/scripts/dialogue_controllers.gd",
    "/srv/work/projects/alpha-widget/scripts/save_contoller.gd",
    "/srv/work/projects/alpha-widget/scripts/physics_controller_tets.gd",
    "/srv/work/projects/alpha-widget/scripts/ui_hud_controller_test.gd.bak",
    "/srv/work/projects/alpha-widget/scripts/level_loader_ctrl.gd",
    "/srv/work/projects/alpha-widget/scripts/pathfinding_controler_test.gd",
    "/srv/work/projects/alpha-widget/Scripts/particle_fx_controller.gd",
    "/srv/work/projects/alpha-widget/scripts/score_tracker_controller_tst.gd",
    "/srv/work/projects/alpha-widget/scripts/input_map_controllre.gd",
    "/srv/work/projects/alpha-widget/scripts/quest_log_controller.gd.tmp",
    "/srv/work/projects/alpha-widget/scripts/achievement_controller_rest.gd",
]


def real_path(module: str, kind: str) -> str:
    """kind is 'controller' or 'test'."""
    if kind == "controller":
        return f"{ROOT}/scripts/{module}_controller.gd"
    return f"{ROOT}/scripts/{module}_controller_test.gd"


@dataclass
class Corpus:
    real: list[str]
    decoys: list[str]
    real_set: set[str] = field(default_factory=set)
    decoy_set: set[str] = field(default_factory=set)
    all_paths: list[str] = field(default_factory=list)
    real_lookup: dict[str, tuple[str, str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.real_set = set(self.real)
        self.decoy_set = set(self.decoys)
        self.all_paths = self.real + self.decoys
        # Every ancestor directory of a real path (the project root, "scripts/", ...).
        # A call whose workdir/path names one of these is copying a real location,
        # not mis-copying a file, so it scores EXACT rather than OTHER.
        self.dir_set: set[str] = set()
        for path in self.real:
            parts = path.split("/")
            for i in range(2, len(parts)):
                self.dir_set.add("/".join(parts[:i]))
        for m in MODULES:
            self.real_lookup[real_path(m, "controller")] = (m, "controller")
            self.real_lookup[real_path(m, "test")] = (m, "test")


def build_corpus() -> Corpus:
    real: list[str] = []
    for m in MODULES:
        real.append(real_path(m, "controller"))
        real.append(real_path(m, "test"))
    return Corpus(real=real, decoys=list(DECOYS))


def nearest_real(path: str, corpus: Corpus) -> str:
    close = difflib.get_close_matches(path, corpus.real, n=1, cutoff=0.3)
    return close[0] if close else corpus.real[0]


# --------------------------------------------------------------------------------------
# 2. Deterministic fake file content
# --------------------------------------------------------------------------------------


def _camel(name: str) -> str:
    return "".join(p.capitalize() for p in name.split("_"))


def _numbered(text: str) -> str:
    return "\n".join(f"{i:5d}\t{line}" for i, line in enumerate(text.splitlines(), 1))


def gd_controller_source(module: str, idx: int) -> str:
    cls = _camel(module) + "Controller"
    lines = [
        "extends Node",
        f"class_name {cls}",
        "",
        f"signal {module}_state_changed(old_state, new_state)",
        "",
        f"@export var {module}_speed: float = {1.0 + (idx % 5) * 0.25:.2f}",
        f"@export var {module}_max_value: int = {10 + idx}",
        'var _state: String = "idle"',
        "",
        "func _ready() -> void:",
        f'\tprint("{module} controller ready")',
        "",
        "func _physics_process(delta: float) -> void:",
        f"\t_update_{module}(delta)",
        "",
        f"func _update_{module}(delta: float) -> void:",
        '\tif _state == "idle":',
        '\t\t_state = "moving"',
        f'\t\temit_signal("{module}_state_changed", "idle", _state)',
        "",
        f"func reset_{module}() -> void:",
        '\t_state = "idle"',
    ]
    return "\n".join(lines)


def gd_test_source(module: str) -> str:
    cls = _camel(module) + "Controller"
    lines = [
        'extends "res://addons/gdunit4/src/GdUnitTestSuite.gd"',
        "",
        f"func test_{module}_starts_idle() -> void:",
        f"\tvar c := {cls}.new()",
        '\tassert_that(c._state).is_equal("idle")',
        "",
        f"func test_{module}_transitions_on_physics_process() -> void:",
        f"\tvar c := {cls}.new()",
        "\tc._physics_process(0.016)",
        '\tassert_that(c._state).is_equal("moving")',
        "",
        f"func test_{module}_reset_returns_to_idle() -> void:",
        f"\tvar c := {cls}.new()",
        "\tc._physics_process(0.016)",
        f"\tc.reset_{module}()",
        '\tassert_that(c._state).is_equal("idle")',
    ]
    return "\n".join(lines)


def read_result(module: str, kind: str, idx: int) -> str:
    src = gd_controller_source(module, idx) if kind == "controller" else gd_test_source(module)
    return _numbered(src)


def enoent_result(bad_path: str, suggestion: str) -> str:
    return f"<tool_error>ENOENT: no such file '{bad_path}'. Did you mean '{suggestion}'?</tool_error>"


def write_result(path: str, nbytes: int) -> str:
    return f"Wrote {nbytes} bytes to '{path}'."


def test_run_result(module: str, all_pass: bool) -> str:
    header = "Running gdunit4 ...\n"
    a = f"  test_{module}_starts_idle .......................... OK\n"
    c = f"  test_{module}_reset_returns_to_idle ................ OK\n"
    if all_pass:
        b = f"  test_{module}_transitions_on_physics_process ...... OK\n"
        return header + a + b + c + "3 passed, 0 failed in 0.31s"
    b = (f"  test_{module}_transitions_on_physics_process ...... FAIL\n"
         '    expected: "moving", actual: "idle"\n')
    return header + a + b + c + "2 passed, 1 failed in 0.29s"


def search_result(hits: list[tuple[str, int, str]]) -> str:
    if not hits:
        return "No matches."
    return "\n".join(f"{p}:{ln}: {snip}" for p, ln, snip in hits)


# --------------------------------------------------------------------------------------
# 3. Tool schemas (OpenAI function-calling format)
# --------------------------------------------------------------------------------------


def _tool(name: str, desc: str, props: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": props,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


TOOLS: list[dict] = [
    _tool("read_file", "Read a file and return its contents with line numbers.",
          {"path": {"type": "string", "description": "Absolute path to the file."}},
          ["path"]),
    _tool("write_file", "Create a file or overwrite it completely with new content.",
          {"path": {"type": "string", "description": "Absolute path to the file."},
           "content": {"type": "string", "description": "Complete file content."}},
          ["path", "content"]),
    _tool("search_files", "Search for a pattern under an absolute path — a single file to "
          "search just that file, or a directory to search every file under it.",
          {"path": {"type": "string", "description": "Absolute file or directory path."},
           "pattern": {"type": "string", "description": "Literal text or a simple regex."}},
          ["path", "pattern"]),
    _tool("terminal", "Run one shell command in a working directory.",
          {"command": {"type": "string", "description": "The exact shell command to run."},
           "workdir": {"type": "string", "description": "Absolute working directory."}},
          ["command", "workdir"]),
]


def apply_strict(tools: list[dict]) -> list[dict]:
    out = []
    for t in tools:
        t2 = json.loads(json.dumps(t))
        t2["function"]["strict"] = True
        out.append(t2)
    return out


SYSTEM_PROMPT = f"""You are an autonomous coding agent maintaining the Godot project rooted at
`{ROOT}`. You act only through the four tools available to you; there is no other way to read,
search, change, or run anything in the repository, and no human is watching the conversation to
correct a wrong path for you.

Rules of engagement:
- Every turn, decide the next concrete action and call exactly one tool for it. Do not describe
  what you would do in prose — call the tool. Only answer in plain text, one short sentence, if
  the user asks a question that needs no file operation at all.
- Copy every file path you use byte-for-byte from a path that already appeared earlier in this
  conversation — in something you already read, searched, or wrote. Never retype a path from
  memory of its spelling, never abbreviate it, and never invent one that has not been shown to
  you. Near-duplicate paths in this repository differ by a single character or a single path
  segment, and only one of them is real.
- If a tool call comes back with an error, read the error and correct the exact path it names
  before trying again. Do not repeat a call that just failed.
- Keep any prose you do write under one sentence. The tool calls are the work."""

TURN_TEMPLATES: list[tuple[str, str]] = [
    ("controller", "Open the {module} controller script again and paste back its signal "
                    "declaration line, exactly as it is written."),
    ("controller", "Re-read the {module} controller from the top. I want to confirm the "
                    "exported speed default before I touch it."),
    ("controller", "Search just the {module} controller script itself — not the whole "
                    "directory — for the word 'idle' and give me the line numbers."),
    ("controller", "Use the terminal tool to `cat` the {module} controller script's absolute "
                    "path directly (no `cd` first) and show me the output."),
    ("test", "Open the {module} controller's test file and show me its second test function."),
    ("test", "Read the {module} controller's test file from the top. Which suite class does it "
             "extend?"),
    ("test", "Run the {module} controller's test file through `gdunit_runner.sh` from the "
             "terminal tool, the same way we've been running it, and show me the output."),
]


def pick_turn(turn_idx: int, rng: random.Random) -> tuple[str, str, str]:
    """Returns (prompt_text, expected_path, module). Module order is fixed so every module
    is exercised before any repeats; template choice is the only thing the seed varies."""
    module = MODULES[(turn_idx - 1) % len(MODULES)]
    kind, template = rng.choice(TURN_TEMPLATES)
    return template.format(module=module), real_path(module, kind), module


# --------------------------------------------------------------------------------------
# 4. Building the long-context prefix
# --------------------------------------------------------------------------------------

ACTIONS = ("read_controller", "read_test", "search", "run_tests", "fix_or_detour")


def build_prefix(rng: random.Random, target_chars: int, corpus: Corpus) -> list[dict]:
    msgs: list[dict] = [{
        "role": "user",
        "content": (
            f"We're picking this back up. Walk the scripts directory under `{ROOT}/scripts` "
            "module by module: read each controller and its test, run the suite, and fix "
            f"anything that fails. Start with `{MODULES[0]}`."
        ),
    }]
    order = list(range(len(MODULES)))
    rng.shuffle(order)
    call_id = 0
    total_chars = sum(len(m["content"]) for m in msgs)
    laps = 0
    while total_chars < target_chars and laps < 60:
        for mi in order:
            module = MODULES[mi]
            for action in ACTIONS:
                call_id += 1
                cid = f"call_{call_id}"
                if action == "read_controller":
                    name, path = "read_file", real_path(module, "controller")
                    args = {"path": path}
                    result = read_result(module, "controller", mi)
                elif action == "read_test":
                    name, path = "read_file", real_path(module, "test")
                    args = {"path": path}
                    result = read_result(module, "test", mi)
                elif action == "search":
                    name = "search_files"
                    args = {"path": f"{ROOT}/scripts", "pattern": f"{module}_state_changed"}
                    hits = [
                        (real_path(module, "controller"), 4,
                         f"signal {module}_state_changed(old_state, new_state)"),
                        (real_path(module, "controller"), 18,
                         f'emit_signal("{module}_state_changed", "idle", _state)'),
                    ]
                    result = search_result(hits)
                elif action == "run_tests":
                    name = "terminal"
                    # The fixture's test runner takes the suite's absolute path directly, so
                    # every fabricated terminal call below models that same convention — the
                    # probe is specifically about copying absolute paths, so nothing in the
                    # fixture should teach the model a relative-path habit instead.
                    args = {
                        "command": f"gdunit_runner.sh {real_path(module, 'test')}",
                        "workdir": ROOT,
                    }
                    result = test_run_result(module, rng.random() > 0.2)
                else:  # fix_or_detour
                    if rng.random() < 0.3 and DECOYS:
                        decoy = rng.choice(DECOYS)
                        suggestion = nearest_real(decoy, corpus)
                        name = "read_file"
                        args = {"path": decoy}
                        result = enoent_result(decoy, suggestion)
                    else:
                        name = "write_file"
                        path = real_path(module, "controller")
                        content = gd_controller_source(module, mi)
                        args = {"path": path, "content": content}
                        result = write_result(path, len(content))
                msgs.append({
                    "role": "assistant", "content": "",
                    "tool_calls": [{"id": cid, "type": "function",
                                    "function": {"name": name, "arguments": json.dumps(args)}}],
                })
                msgs.append({"role": "tool", "tool_call_id": cid, "content": result})
                total_chars += len(result) + len(json.dumps(args))
                if total_chars >= target_chars:
                    break
            if total_chars >= target_chars:
                break
        laps += 1
    msgs.append({"role": "user", "content": "Good. Let's keep going with the rest of the modules."})
    return msgs


# --------------------------------------------------------------------------------------
# 5. Classification: EXACT / SLIP / SPLICE / DECOY / OTHER
# --------------------------------------------------------------------------------------

PATHLIKE_KEYS = ("path", "workdir", "cwd", "file", "directory")
ABS_PATH_RE = re.compile(r"/(?:[A-Za-z0-9_.+-]+/)+[A-Za-z0-9_.+-]+")


def extract_candidates(args: dict) -> list[tuple[str, str]]:
    """Returns (source_key, candidate_string) for every path-like value in a tool call's
    parsed arguments: the dedicated keys, plus absolute-path-shaped substrings inside a
    'command' string."""
    out: list[tuple[str, str]] = []
    for k in PATHLIKE_KEYS:
        v = args.get(k)
        if isinstance(v, str) and v.strip():
            out.append((k, v.strip()))
    cmd = args.get("command")
    if isinstance(cmd, str):
        for m in ABS_PATH_RE.finditer(cmd):
            s = m.group(0)
            if len(s) >= 12 and s.count("/") >= 3:
                out.append(("command", s))
    return out


def levenshtein(a: str, b: str, cutoff: int = 64) -> int:
    if a == b:
        return 0
    if abs(len(a) - len(b)) >= cutoff:
        return cutoff
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        row_min = cur[0]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            v = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            cur[j] = v
            row_min = min(row_min, v)
        if row_min >= cutoff:
            return cutoff
        prev = cur
    return prev[-1]


def classify_candidate(candidate: str, corpus: Corpus) -> dict:
    if candidate in corpus.real_set:
        return {"label": "EXACT", "nearest": candidate, "distance": 0}
    if candidate.rstrip("/") in corpus.dir_set:
        return {"label": "EXACT", "nearest": candidate.rstrip("/"), "distance": 0,
                "note": "directory of the fixture"}
    if candidate in corpus.decoy_set:
        return {"label": "DECOY", "nearest": candidate, "distance": None}
    longest_prefix = None
    for p in corpus.all_paths:
        if candidate.startswith(p) and len(candidate) > len(p):
            if longest_prefix is None or len(p) > len(longest_prefix):
                longest_prefix = p
    if longest_prefix is not None:
        return {"label": "SPLICE", "nearest": longest_prefix, "distance": None}
    close = difflib.get_close_matches(candidate, corpus.all_paths, n=1, cutoff=0.5)
    if close:
        d = levenshtein(candidate, close[0], cutoff=6)
        if 1 <= d <= 3:
            return {"label": "SLIP", "nearest": close[0], "distance": d}
    return {"label": "OTHER", "nearest": close[0] if close else None, "distance": None}


def has_degenerate_repetition(text: str, n: int = 6, min_count: int = 3) -> bool:
    words = text.split()
    if len(words) < n:
        return False
    seen: dict[str, int] = {}
    for i in range(len(words) - n + 1):
        gram = " ".join(words[i:i + n])
        c = seen.get(gram, 0) + 1
        seen[gram] = c
        if c >= min_count:
            return True
    return False


# --------------------------------------------------------------------------------------
# 6. Synthetic tool results for the real, scored turns
# --------------------------------------------------------------------------------------


def synth_tool_result(name: str, args: dict, corpus: Corpus) -> str:
    if name == "read_file":
        p = args.get("path") or ""
        if p in corpus.real_set:
            module, kind = corpus.real_lookup[p]
            idx = MODULES.index(module)
            return read_result(module, kind, idx)
        if p in corpus.decoy_set:
            return enoent_result(p, nearest_real(p, corpus))
        return f"<tool_error>ENOENT: no such file '{p}'.</tool_error>"
    if name == "search_files":
        p = args.get("path") or ""
        pattern = args.get("pattern") or ""
        if p in corpus.real_set:
            return f"{p}:3: match for '{pattern}'"
        if p in corpus.decoy_set:
            return enoent_result(p, nearest_real(p, corpus))
        base = os.path.dirname(p) if p else ""
        hits = [rp for rp in corpus.real if base and os.path.dirname(rp) == base][:2]
        if not hits:
            return f"No matches for '{pattern}' under '{p}'."
        return "\n".join(f"{h}:1: match for '{pattern}'" for h in hits)
    if name == "write_file":
        p = args.get("path") or ""
        content = args.get("content") or ""
        return write_result(p, len(content))
    if name == "terminal":
        cmd = args.get("command") or ""
        wd = args.get("workdir") or ""
        return f"$ {cmd}\n(workdir: {wd})\n(exit 0)"
    return f"<tool_error>unknown tool '{name}'</tool_error>"


# --------------------------------------------------------------------------------------
# 7. HTTP
# --------------------------------------------------------------------------------------


def describe_error(e: Exception) -> str:
    if isinstance(e, urllib.error.HTTPError):
        try:
            detail = e.read()[:300].decode("utf-8", "replace")
        except Exception:
            detail = ""
        return f"HTTP {e.code}: {detail}" if detail else f"HTTP {e.code}"
    return f"{type(e).__name__}: {e}"


def post_chat(url: str, body: dict, api_key: str, timeout: int) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key and api_key.lower() != "none":
        headers["Authorization"] = f"Bearer {api_key}"
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def deep_merge(base: dict, extra: dict) -> None:
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_merge(base[k], v)
        else:
            base[k] = v


def build_body(args: argparse.Namespace, messages: list[dict], tools: list[dict],
                cache_salt_val: str | None) -> dict:
    body: dict = {
        "model": args.model,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    if args.effort:
        body["reasoning_effort"] = args.effort
    if cache_salt_val is not None:
        body["cache_salt"] = cache_salt_val
    if args.extra_body:
        deep_merge(body, args.extra_body)
    return body


# --------------------------------------------------------------------------------------
# 8. Scoring one turn
# --------------------------------------------------------------------------------------


def score_and_append(turn_idx: int, messages: list[dict], resp: dict, secs: float,
                      expected_path: str, corpus: Corpus) -> dict:
    choice = (resp.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    tool_calls = msg.get("tool_calls") or []
    finish = choice.get("finish_reason")
    usage = resp.get("usage") or {}
    ptk = int(usage.get("prompt_tokens") or 0)
    ctk = int(usage.get("completion_tokens") or 0)

    candidates: list[dict] = []
    json_invalid = 0
    parsed_per_call: list[tuple[str, dict]] = []
    for tc in tool_calls:
        fn = tc.get("function") or {}
        name = fn.get("name") or ""
        raw_args = fn.get("arguments") or ""
        try:
            parsed = json.loads(raw_args) if raw_args else {}
            if not isinstance(parsed, dict):
                raise ValueError("arguments is not a JSON object")
        except Exception:
            json_invalid += 1
            parsed = {}
        parsed_per_call.append((name, parsed))
        for key, cand in extract_candidates(parsed):
            cls = classify_candidate(cand, corpus)
            candidates.append({"tool": name, "key": key, "raw": cand, **cls})

    expected_hit = any(c["raw"] == expected_path for c in candidates)
    runaway = finish == "length"
    degenerate = has_degenerate_repetition(content) or has_degenerate_repetition(reasoning)
    for tc in tool_calls:
        fn = tc.get("function") or {}
        if has_degenerate_repetition(fn.get("arguments") or ""):
            degenerate = True

    if expected_hit:
        defect = "EXACT"
    elif candidates:
        best = min(candidates, key=lambda c: levenshtein(c["raw"], expected_path, cutoff=9999))
        defect = best["label"]
    elif not tool_calls:
        defect = "NO_TOOL_CALL"
    else:
        defect = "OTHER"

    bad = (not expected_hit) or json_invalid > 0 or runaway or degenerate

    row = {
        "turn": turn_idx, "prompt_tokens": ptk, "completion_tokens": ctk,
        "latency_s": round(secs, 2), "finish_reason": finish, "n_tool_calls": len(tool_calls),
        "expected_path": expected_path, "expected_hit": expected_hit, "defect": defect,
        "json_invalid": json_invalid, "runaway": runaway, "degenerate_repetition": degenerate,
        "candidates": candidates, "bad": bool(bad),
    }

    echo: dict = {"role": "assistant", "content": content}
    if reasoning:
        echo["reasoning_content"] = reasoning
    if tool_calls:
        echo["tool_calls"] = tool_calls
    messages.append(echo)
    for tc, (name, parsed) in zip(tool_calls, parsed_per_call):
        result = synth_tool_result(name, parsed, corpus)
        messages.append({"role": "tool", "tool_call_id": tc.get("id") or f"call_t{turn_idx}",
                          "content": result[:4000]})
    return row


def print_turn_line(row: dict, quiet: bool) -> None:
    if quiet:
        return
    print(f"[t{row['turn']:02d}] ptk={row['prompt_tokens']:6d} ctk={row['completion_tokens']:4d} "
          f"{row['latency_s']:5.1f}s calls={row['n_tool_calls']} defect={row['defect']:<12s} "
          f"hit={'Y' if row['expected_hit'] else 'n'} json_bad={row['json_invalid']} "
          f"runaway={'Y' if row['runaway'] else 'n'} rep={'Y' if row['degenerate_repetition'] else 'n'}",
          flush=True)


# --------------------------------------------------------------------------------------
# 9. Summary
# --------------------------------------------------------------------------------------


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    turn_defect_counts = {"EXACT": 0, "SLIP": 0, "SPLICE": 0, "DECOY": 0, "OTHER": 0,
                           "NO_TOOL_CALL": 0}
    cand_label_counts = {"EXACT": 0, "SLIP": 0, "SPLICE": 0, "DECOY": 0, "OTHER": 0}
    json_invalid_total = runaway_total = degenerate_total = bad_turns = 0
    first_bad = None
    latencies = []
    for r in rows:
        turn_defect_counts[r["defect"]] = turn_defect_counts.get(r["defect"], 0) + 1
        for c in r["candidates"]:
            cand_label_counts[c["label"]] = cand_label_counts.get(c["label"], 0) + 1
        json_invalid_total += r["json_invalid"]
        runaway_total += int(r["runaway"])
        degenerate_total += int(r["degenerate_repetition"])
        if r["bad"]:
            bad_turns += 1
            if first_bad is None:
                first_bad = r["turn"]
        latencies.append(r["latency_s"])
    return {
        "n_turns": n,
        "turn_defect_counts": turn_defect_counts,
        "candidate_label_counts": cand_label_counts,
        "json_invalid_total": json_invalid_total,
        "runaway_total": runaway_total,
        "degenerate_repetition_total": degenerate_total,
        "bad_turns": bad_turns,
        "bad_rate": (bad_turns / n) if n else 0.0,
        "first_bad_turn": first_bad,
        "prompt_tokens_first": rows[0]["prompt_tokens"] if rows else None,
        "prompt_tokens_last": rows[-1]["prompt_tokens"] if rows else None,
        "median_latency_s": statistics.median(latencies) if latencies else None,
    }


def print_summary(summary: dict) -> None:
    print("\n=== summary ===")
    print(f"turns: {summary['n_turns']}   bad turns: {summary['bad_turns']} "
          f"({summary['bad_rate']:.0%})   first bad turn: {summary['first_bad_turn']}")
    print(f"prompt_tokens: first={summary['prompt_tokens_first']} last={summary['prompt_tokens_last']}")
    med = summary["median_latency_s"]
    print(f"median latency: {med:.1f}s" if med is not None else "median latency: n/a")
    print("\nper-turn result (what the turn's required path came back as):")
    for k in ("EXACT", "SLIP", "SPLICE", "DECOY", "OTHER", "NO_TOOL_CALL"):
        print(f"  {k:<14s} {summary['turn_defect_counts'].get(k, 0)}")
    print("\nevery classified path-like argument (all tool calls, all turns):")
    for k in ("EXACT", "SLIP", "SPLICE", "DECOY", "OTHER"):
        print(f"  {k:<14s} {summary['candidate_label_counts'].get(k, 0)}")
    print(f"\njson-invalid tool-call arguments: {summary['json_invalid_total']}")
    print(f"runaway turns (finish_reason=length): {summary['runaway_total']}")
    print(f"turns with degenerate repetition: {summary['degenerate_repetition_total']}")


def write_json_out(path: str, args: argparse.Namespace, rows: list[dict], summary: dict) -> None:
    out = {
        "settings": {
            "url": args.url, "model": args.model, "tokens_target": args.tokens,
            "turns": args.turns, "seed": args.seed, "temperature": args.temperature,
            "max_tokens": args.max_tokens, "strict": args.strict, "effort": args.effort,
            "cache_salt": args.cache_salt, "extra_body": args.extra_body,
        },
        "summary": summary,
        "turns": rows,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------------------
# 10. CLI and main
# --------------------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Synthetic long-context tool-argument copy-fidelity probe "
                     "(docs/14-troubleshooting.md §9.15).")
    ap.add_argument("--url", default="http://localhost:8001/v1")
    ap.add_argument("--model", default="glm-5.3-flash")
    ap.add_argument("--api-key", default="none")
    ap.add_argument("--tokens", type=int, default=35000, help="target prompt size, turn 1")
    ap.add_argument("--turns", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--strict", action="store_true", help='add "strict": true to every tool')
    ap.add_argument("--effort", choices=["low", "high"], default=None,
                     help="sent as reasoning_effort")
    ap.add_argument("--cache-salt", choices=["fresh", "session"], default=None,
                     help="fresh: unique cache_salt per turn. session: one for the whole run. "
                          "omitted entirely unless this flag is given")
    ap.add_argument("--extra-body", default=None,
                     help="JSON object deep-merged into every request body, e.g. "
                          '\'{"chat_template_kwargs":{"clear_thinking":true}}\'')
    ap.add_argument("--out", default=None, help="write full per-turn JSON results here")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--rescore", default=None, metavar="RESULTS_JSON",
                    help="re-classify the candidates of an earlier --out file with the "
                         "current classifier (no requests) and print its summary")
    a = ap.parse_args(argv)
    if a.extra_body:
        try:
            a.extra_body = json.loads(a.extra_body)
        except json.JSONDecodeError as e:
            ap.error(f"--extra-body is not valid JSON: {e}")
    return a


def rescore(path: str, corpus: Corpus, quiet: bool) -> int:
    with open(path) as f:
        data = json.load(f)
    rows = data["turns"]
    for r in rows:
        for c in r["candidates"]:
            c.update(classify_candidate(c["raw"], corpus))
        if not quiet:
            print_turn_line(r, quiet)
    data["summary"] = summarize(rows)
    data["rescored_with"] = "classifier of " + os.path.basename(__file__)
    print_summary(data["summary"])
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"rescored {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    corpus = build_corpus()
    if args.rescore:
        return rescore(args.rescore, corpus, args.quiet)
    tools = apply_strict(TOOLS) if args.strict else TOOLS
    run_id = uuid.uuid4().hex[:12]
    session_salt = run_id if args.cache_salt == "session" else None

    rng_turns = random.Random(args.seed + 9973)
    target_tokens = args.tokens
    rows: list[dict] = []
    messages: list[dict] | None = None
    max_calibration = 3

    for attempt in range(max_calibration):
        target_chars = int(target_tokens * 3.6)
        prefix_rng = random.Random(args.seed)
        prefix = build_prefix(prefix_rng, target_chars, corpus)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + prefix
        rng_turns = random.Random(args.seed + 9973)
        text, expected_path, _module = pick_turn(1, rng_turns)
        messages.append({"role": "user", "content": text})
        cache_salt_val = (f"{run_id}-t1" if args.cache_salt == "fresh" else session_salt)
        body = build_body(args, messages, tools, cache_salt_val)
        try:
            t0 = time.time()
            resp = post_chat(args.url, body, args.api_key, args.timeout)
            secs = time.time() - t0
        except Exception as e:  # noqa: BLE001 — first request failing means unreachable
            print(f"ERROR: could not reach {args.url}: {describe_error(e)}", file=sys.stderr)
            return 1
        usage = resp.get("usage") or {}
        actual = int(usage.get("prompt_tokens") or 0)
        if actual <= 0:
            if not args.quiet:
                print("WARNING: endpoint returned no prompt_tokens usage; skipping calibration",
                      file=sys.stderr)
            break
        off = abs(actual - target_tokens) / max(1, target_tokens)
        if off <= 0.15 or attempt == max_calibration - 1:
            if off > 0.15 and not args.quiet:
                print(f"WARNING: prompt_tokens {actual} still {off:.0%} off target "
                      f"{target_tokens} after {attempt + 1} attempt(s)", file=sys.stderr)
            break
        scale = target_tokens / actual
        new_target = max(1000, int(target_tokens * scale))
        if not args.quiet:
            print(f"[calib] target={target_tokens} -> actual prompt_tokens={actual} "
                  f"({off:.0%} off) -> rescaling to {new_target}", file=sys.stderr)
        target_tokens = new_target

    row1 = score_and_append(1, messages, resp, secs, expected_path, corpus)
    rows.append(row1)
    print_turn_line(row1, args.quiet)

    for turn_idx in range(2, args.turns + 1):
        text, expected_path, _module = pick_turn(turn_idx, rng_turns)
        messages.append({"role": "user", "content": text})
        cache_salt_val = (f"{run_id}-t{turn_idx}" if args.cache_salt == "fresh" else session_salt)
        body = build_body(args, messages, tools, cache_salt_val)
        try:
            t0 = time.time()
            resp = post_chat(args.url, body, args.api_key, args.timeout)
            secs = time.time() - t0
        except Exception as e:  # noqa: BLE001 — mid-run failure: log and stop, still report
            print(f"ERROR on turn {turn_idx}: {describe_error(e)}", file=sys.stderr)
            break
        row = score_and_append(turn_idx, messages, resp, secs, expected_path, corpus)
        rows.append(row)
        print_turn_line(row, args.quiet)

    summary = summarize(rows)
    print_summary(summary)
    if args.out:
        write_json_out(args.out, args, rows, summary)
        if not args.quiet:
            print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
