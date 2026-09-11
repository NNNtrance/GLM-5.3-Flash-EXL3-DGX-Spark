#!/usr/bin/env python3
"""toolcall-gate.py — issue #7: first-corruption-rate measurement for tool-call XML
under a long agentic context.

Sets up a synthetic but realistic agentic coding session (system prompt + 30 tool
definitions + a fabricated repository), drives the model turn by turn, and counts/
classifies the tool calls on every turn:

  a) well-formed tool call          (JSON parses, matches the schema)
  b) fail-closed rejection          (raw <tool_call> text appears in the content)
  c) parsed but out-of-schema call  (unknown / missing key)  -> seed of a cascade
  d) empty turn                     (empty content, no tool_calls, finish_reason=stop)
  e) repeated call                  (byte-identical to the previous assistant turn)
  f) finish_reason=length

Output: one JSONL line per turn + a summary table bucketed by 10k tokens of context.
Exit code is ALWAYS 0 (this is a measurement, not a pass/fail gate).

Stdlib only. Python 3.12.

Usage (smoke-test against the mock engine):
    ./toolcall-gate.py --mock --rounds 14 --sessions 1 --out /tmp/gate-mock

Usage (real run, recommended settings):
    ./toolcall-gate.py --api http://127.0.0.1:8001 --rounds 30 --sessions 4 \
        --concurrency 2 --max-prompt-tokens 120000 --out ./out/run-1
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import pathlib
import re
import sys
import threading
import time
import urllib.error
import urllib.request

# --------------------------------------------------------------------------------------
# 1. Tool definitions (OpenAI `tools` format) — 30 tools, realistic schemas
# --------------------------------------------------------------------------------------


def _t(name: str, desc: str, props: dict, required: list[str]) -> dict:
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


_STR = {"type": "string"}
_INT = {"type": "integer"}
_BOOL = {"type": "boolean"}

TOOLS: list[dict] = [
    _t("bash", "Execute a shell command in the repository root. Long commands and heredocs "
       "are allowed. Output is truncated to 30000 characters.",
       {"command": {"type": "string", "description": "The exact shell command to run. May be "
                    "multi-line; quote carefully. Do not use interactive flags."},
        "description": {"type": "string", "description": "5-10 word description of what the "
                        "command does, in active voice."},
        "timeout_ms": {"type": "integer", "description": "Timeout in milliseconds, max 600000."},
        "run_in_background": _BOOL},
       ["command"]),
    _t("read", "Read a file from the repository. Returns the content with cat -n style line numbers.",
       {"file_path": {"type": "string", "description": "Absolute or repo-relative path."},
        "offset": {"type": "integer", "description": "1-based first line to read."},
        "limit": {"type": "integer", "description": "Maximum number of lines to return."}},
       ["file_path"]),
    _t("write", "Create a new file or fully overwrite an existing one.",
       {"file_path": _STR,
        "content": {"type": "string", "description": "Complete file content. No placeholders, "
                    "no elisions; write the whole file."}},
       ["file_path", "content"]),
    _t("edit", "Exact string replacement in a file. old_string must be unique in the file.",
       {"file_path": _STR,
        "old_string": {"type": "string", "description": "Text to replace, including indentation."},
        "new_string": {"type": "string", "description": "Replacement text."},
        "replace_all": _BOOL},
       ["file_path", "old_string", "new_string"]),
    _t("multi_edit", "Apply several exact replacements to one file atomically.",
       {"file_path": _STR,
        "edits": {"type": "array", "description": "Ordered list of replacements.",
                  "items": {"type": "object",
                            "properties": {"old_string": _STR, "new_string": _STR,
                                           "replace_all": _BOOL},
                            "required": ["old_string", "new_string"]}}},
       ["file_path", "edits"]),
    _t("glob", "Find files by glob pattern, newest first.",
       {"pattern": {"type": "string", "description": "Glob such as src/**/*.py"},
        "path": _STR},
       ["pattern"]),
    _t("grep", "Search file contents with a regular expression (ripgrep semantics).",
       {"pattern": {"type": "string", "description": "Regular expression."},
        "path": _STR,
        "glob": _STR,
        "output_mode": {"type": "string", "enum": ["content", "files_with_matches", "count"]},
        "context_lines": _INT,
        "case_insensitive": _BOOL},
       ["pattern"]),
    _t("ls", "List a directory.", {"path": _STR, "recursive": _BOOL}, ["path"]),
    _t("git_status", "Show the working tree status.", {"short": _BOOL}, []),
    _t("git_diff", "Show unstaged or staged changes.",
       {"staged": _BOOL, "path": _STR, "stat_only": _BOOL}, []),
    _t("git_log", "Show recent commits.",
       {"max_count": _INT, "path": _STR, "oneline": _BOOL}, []),
    _t("git_commit", "Create a commit from the staged changes.",
       {"message": {"type": "string", "description": "Full commit message, subject plus body."},
        "add_all": _BOOL},
       ["message"]),
    _t("run_tests", "Run the pytest suite or a subset of it.",
       {"target": {"type": "string", "description": "Path or node id, e.g. tests/test_money.py::test_round"},
        "verbose": _BOOL, "maxfail": _INT, "keyword": _STR},
       []),
    _t("lint", "Run ruff over the given paths.", {"paths": {"type": "array", "items": _STR},
                                                 "fix": _BOOL}, []),
    _t("typecheck", "Run mypy over the given package.", {"package": _STR, "strict": _BOOL}, []),
    _t("format_code", "Run the formatter over the given paths.",
       {"paths": {"type": "array", "items": _STR}, "check_only": _BOOL}, ["paths"]),
    _t("postgres_readonly_execute_query",
       "Run a single read-only SQL statement against the analytics replica. SELECT only; "
       "statements that write are rejected by the connection role.",
       {"sql": {"type": "string", "description": "One SQL statement, no trailing semicolon "
                "required. Multi-line formatting is fine."},
        "database": _STR,
        "max_rows": _INT,
        "timeout_s": _INT},
       ["sql"]),
    _t("postgres_list_tables", "List tables and row estimates in a schema.",
       {"schema": _STR, "include_views": _BOOL}, []),
    _t("postgres_explain", "Return the query plan for a statement without executing it.",
       {"sql": _STR, "analyze": _BOOL}, ["sql"]),
    _t("redis_get", "Read one key from the cache cluster.", {"key": _STR, "db": _INT}, ["key"]),
    _t("http_fetch", "Fetch a URL and return the body as text.",
       {"url": _STR, "method": {"type": "string", "enum": ["GET", "HEAD"]},
        "headers": {"type": "object", "additionalProperties": _STR}},
       ["url"]),
    _t("docker_ps", "List running containers.", {"all": _BOOL}, []),
    _t("docker_logs", "Tail the logs of a container.",
       {"container": _STR, "tail": _INT, "since": _STR}, ["container"]),
    _t("k8s_get_pods", "List pods in a namespace.", {"namespace": _STR, "selector": _STR}, []),
    _t("metrics_query", "Query the metrics backend with PromQL.",
       {"query": _STR, "lookback": _STR, "step": _STR}, ["query"]),
    _t("jira_create_issue", "Create a tracker issue.",
       {"project": _STR, "summary": _STR,
        "description": {"type": "string", "description": "Full issue body in Markdown, "
                        "including reproduction steps and expected behaviour."},
        "issue_type": {"type": "string", "enum": ["Bug", "Task", "Story"]},
        "labels": {"type": "array", "items": _STR}},
       ["project", "summary", "description"]),
    _t("slack_post_message", "Post a message to a channel.",
       {"channel": _STR,
        "text": {"type": "string", "description": "Message body; Slack mrkdwn is supported."},
        "thread_ts": _STR},
       ["channel", "text"]),
    _t("notebook_edit", "Replace the source of one notebook cell.",
       {"notebook_path": _STR, "cell_id": _STR, "new_source": _STR,
        "cell_type": {"type": "string", "enum": ["code", "markdown"]}},
       ["notebook_path", "new_source"]),
    _t("todo_write", "Replace the task list for the current piece of work.",
       {"todos": {"type": "array",
                  "items": {"type": "object",
                            "properties": {"content": _STR, "status": {"type": "string",
                                           "enum": ["pending", "in_progress", "completed"]}},
                            "required": ["content", "status"]}}},
       ["todos"]),
    _t("task_spawn", "Delegate a self-contained sub-task to a sub-agent.",
       {"description": _STR,
        "prompt": {"type": "string", "description": "Complete, standalone instructions for the "
                   "sub-agent, including file paths."},
        "subagent_type": _STR},
       ["description", "prompt"]),
]

TOOL_SCHEMA = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS}

SYSTEM_PROMPT = """You are a senior software engineer working inside a real repository through
tools. The repository is a double-entry accounting service called `ledger-svc` (Python 3.12,
FastAPI, PostgreSQL, pytest).

Rules of engagement:
- Always inspect before you change. Read the file, or grep for the symbol, before editing it.
- Prefer the dedicated tools (read/edit/grep/glob/run_tests) over shelling out with bash.
- One tool call per distinct fact you need; batch independent calls in the same turn.
- After any edit, run the relevant tests with run_tests before claiming the work is done.
- Never invent file paths or symbol names. If a read fails, grep for the real path.
- Keep prose short. Report what you did, what it showed, and the next step.
- When the task is finished, say so in one sentence and stop calling tools.

Repository layout is under src/ledger/ with tests under tests/. The analytics replica is
read-only and reachable through postgres_readonly_execute_query."""

# --------------------------------------------------------------------------------------
# 2. Fake repository — fully deterministic generation
# --------------------------------------------------------------------------------------

_VERBS = ["normalize", "resolve", "apply", "collect", "validate", "merge", "split", "round",
          "reconcile", "project", "summarize", "expand", "rebuild", "flush", "index", "cache"]
_NOUNS = ["postings", "accounts", "entries", "balances", "currencies", "periods", "journals",
          "batches", "statements", "transfers", "allocations", "reversals", "fx_rates", "buckets"]


def _func_src(mod: str, i: int) -> str:
    v = _VERBS[(i * 5 + len(mod)) % len(_VERBS)]
    n = _NOUNS[(i * 3 + len(mod) * 2) % len(_NOUNS)]
    name = f"{v}_{n}"
    k = 2 + (i % 5)
    todo = "\n    # TODO(ledger-412): this path predates the Decimal migration.\n" if i % 7 == 3 else ""
    return f'''

def {name}(rows: list[dict], *, period: str, strict: bool = False) -> dict[str, Decimal]:
    """{v.capitalize()} the {n.replace('_', ' ')} for one accounting period.

    Args:
        rows: raw {n} as returned by the importer, newest first.
        period: period key in ``YYYY-MM`` form.
        strict: when true, a row that fails validation raises instead of being skipped.

    Returns:
        Mapping of account code to the {v}ed amount, as Decimal.

    Raises:
        LedgerError: if ``strict`` and a row is malformed.
    """{todo}
    out: dict[str, Decimal] = defaultdict(Decimal)
    skipped = 0
    for idx, row in enumerate(rows):
        code = str(row.get("account") or "").strip()
        if not code:
            if strict:
                raise LedgerError(f"row {{idx}} has no account code")
            skipped += 1
            continue
        amount = to_decimal(row.get("amount", 0))
        if row.get("period") not in (None, period):
            skipped += 1
            continue
        scale = Decimal({k}) if row.get("kind") == "accrual" else Decimal(1)
        out[code] += (amount * scale).quantize(CENT, rounding=ROUND_HALF_EVEN)
    if skipped:
        LOG.debug("{name}: skipped %d of %d rows for %s", skipped, len(rows), period)
    return dict(out)
'''


def _module_src(path: str, n_funcs: int) -> str:
    mod = path.rsplit("/", 1)[-1].removesuffix(".py")
    head = f'''"""{mod} — part of ledger-svc.

Generated module for the {mod.replace('_', ' ')} layer. Everything in here operates on
``Decimal`` amounts; floats are never allowed past the importer boundary.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_EVEN, ROUND_HALF_UP
from typing import Iterable, Iterator, Sequence

LOG = logging.getLogger("ledger.{mod}")
CENT = Decimal("0.01")
DEFAULT_CURRENCY = "EUR"


class LedgerError(RuntimeError):
    """Raised when {mod} input cannot be trusted."""


@dataclass(slots=True)
class {mod.title().replace('_', '')}Row:
    account: str
    amount: Decimal
    period: str
    kind: str = "cash"
    memo: str = ""
    tags: list[str] = field(default_factory=list)

    def key(self) -> tuple[str, str]:
        return (self.account, self.period)


def to_decimal(value: object) -> Decimal:
    """Coerce importer output to Decimal without going through float."""
    if isinstance(value, Decimal):
        return value
    if isinstance(value, int):
        return Decimal(value)
    text = str(value).strip().replace(",", "")
    if not text:
        return Decimal(0)
    try:
        return Decimal(text)
    except Exception as exc:  # noqa: BLE001
        raise LedgerError(f"cannot coerce {{value!r}} to Decimal") from exc


def round_money(value: Decimal, *, currency: str = DEFAULT_CURRENCY,
                mode: str = "half_even") -> Decimal:
    """Round to the minor unit of ``currency``.

    NOTE: the ``half_up`` branch is what the statement importer asks for; the rest of the
    codebase assumes banker's rounding. Mixing them is the source of ledger-412.
    """
    rounding = ROUND_HALF_UP if mode == "half_up" else ROUND_HALF_EVEN
    return value.quantize(CENT, rounding=rounding)
'''
    body = "".join(_func_src(mod, i) for i in range(n_funcs))
    tail = f'''

def iter_{mod}(rows: Iterable[dict]) -> Iterator[{mod.title().replace('_', '')}Row]:
    """Adapt raw dicts to typed rows, dropping anything unusable."""
    for row in rows:
        try:
            yield {mod.title().replace('_', '')}Row(
                account=str(row["account"]),
                amount=to_decimal(row["amount"]),
                period=str(row.get("period", "")),
                kind=str(row.get("kind", "cash")),
                memo=str(row.get("memo", ""))[:240],
                tags=list(row.get("tags") or []),
            )
        except (KeyError, LedgerError):
            LOG.warning("dropping unusable {mod} row: %r", row)
            continue


__all__ = ["LedgerError", "round_money", "to_decimal", "iter_{mod}"]
'''
    return head + body + tail


def _test_src(name: str, target: str, n_cases: int) -> str:
    cases = "".join(f'''

def test_{_VERBS[i % len(_VERBS)]}_{_NOUNS[i % len(_NOUNS)]}_{i}() -> None:
    rows = [{{"account": "4{i:03d}", "amount": "{i * 7}.{(i * 13) % 100:02d}", "period": "2026-0{1 + i % 9}"}}]
    got = round_money(to_decimal(rows[0]["amount"]))
    assert got == Decimal("{i * 7}.{(i * 13) % 100:02d}")
''' for i in range(n_cases))
    return f'''"""Tests for {target}."""
from __future__ import annotations

from decimal import Decimal

import pytest

from {target.replace('/', '.').removesuffix('.py').removeprefix('src.')} import (
    LedgerError,
    round_money,
    to_decimal,
)
{cases}

def test_round_money_rejects_float() -> None:
    with pytest.raises(LedgerError):
        to_decimal("not a number")


@pytest.mark.parametrize("mode,expected", [("half_even", "2.66"), ("half_up", "2.67")])
def test_round_money_modes(mode: str, expected: str) -> None:
    assert round_money(Decimal("2.665"), mode=mode) == Decimal(expected)
'''


README_SRC = """# ledger-svc

Double-entry accounting service. Python 3.12, FastAPI, PostgreSQL 16, pytest.

## Layout

    src/ledger/            core domain
      accounts.py          chart of accounts
      ledger.py            journal + posting engine
      postings.py          posting validation
      currency.py          currency metadata and minor units
      reconcile.py         bank statement reconciliation  <-- ledger-412 lives here
      util/money.py        Decimal helpers, round_money()
      util/dates.py        period arithmetic
      importers/           csv_import.py, ofx_import.py, camt_import.py
      reports/             balance.py, cashflow.py, trial_balance.py
      db/                  models.py, migrations.py, session.py
      api/                 routes.py, schemas.py, deps.py
    tests/                 one test module per source module

## Invariants

1. Amounts are `Decimal` from the importer boundary inward. No floats, ever.
2. Every journal entry balances to zero per currency, checked in `ledger.post_entry`.
3Rounding is banker's rounding (`ROUND_HALF_EVEN`) everywhere except the statement
   importer, which the bank requires to use `ROUND_HALF_UP`. This asymmetry is deliberate
   and is the reason `round_money` takes a `mode`.
4. The analytics replica is read-only. Never write from a report path.

## Known issues

- ledger-412: reconciliation drift of one cent on statements that mix accrual and cash
  rows in the same period. Reproduction is in `tests/test_reconcile.py::test_rounding_drift`.
- ledger-455: `camt_import` is slow on files over 50 MB (single-threaded XML walk).

## Running

    make dev            # install + pre-commit
    pytest -q           # full suite, ~40 s
    make migrate        # alembic upgrade head
"""

CI_LOG = """============================= test session starts ==============================
platform linux -- Python 3.12.4, pytest-8.2.2, pluggy-1.5.0
rootdir: /build/ledger-svc
configfile: pyproject.toml
plugins: anyio-4.4.0, cov-5.0.0
collected 412 items

tests/test_accounts.py ................................                  [  7%]
tests/test_currency.py ........................                          [ 13%]
tests/test_ledger.py .............................................       [ 24%]
tests/test_money.py ..........................                           [ 30%]
tests/test_postings.py ...................................               [ 39%]
tests/test_reconcile.py .........F.........                              [ 44%]
tests/test_reports_balance.py .......................                    [ 50%]
tests/test_reports_cashflow.py ..................                        [ 55%]

=================================== FAILURES ===================================
____________________________ test_rounding_drift _______________________________

    def test_rounding_drift() -> None:
        rows = load_fixture("statement_2026_03_mixed.json")
        drift = reconcile_period(rows, period="2026-03")
>       assert drift["4010"] == Decimal("0.00")
E       AssertionError: assert Decimal('0.01') == Decimal('0.00')
E         +Decimal('0.01')
E         -Decimal('0.00')

tests/test_reconcile.py:118: AssertionError
----------------------------- Captured log call --------------------------------
WARNING  ledger.reconcile:reconcile.py:204 mixed kind rows in 2026-03: 318 cash, 44 accrual
DEBUG    ledger.reconcile:reconcile.py:211 round_money mode=half_up for 44 accrual rows
=========================== short test summary info ============================
FAILED tests/test_reconcile.py::test_rounding_drift - AssertionError: assert De...
1 failed, 411 passed, 0 skipped in 38.91s
"""

TRACEBACK_2 = """Traceback (most recent call last):
  File "/srv/ledger/src/ledger/api/routes.py", line 212, in post_statement
    drift = reconcile_period(payload.rows, period=payload.period, strict=True)
  File "/srv/ledger/src/ledger/reconcile.py", line 241, in reconcile_period
    totals = collect_postings(rows, period=period, strict=strict)
  File "/srv/ledger/src/ledger/reconcile.py", line 96, in collect_postings
    raise LedgerError(f"row {idx} has no account code")
ledger.reconcile.LedgerError: row 187 has no account code

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/usr/local/lib/python3.12/site-packages/fastapi/routing.py", line 299, in app
    raw_response = await run_endpoint_function(...)
  File "/srv/ledger/src/ledger/api/routes.py", line 220, in post_statement
    raise HTTPException(status_code=422, detail=str(exc)) from exc
fastapi.exceptions.HTTPException: 422: row 187 has no account code
"""


def build_repo() -> dict[str, str]:
    """Deterministic in-memory fake repository. Same bytes on every run."""
    spec = [
        ("src/ledger/accounts.py", 11),
        ("src/ledger/ledger.py", 14),
        ("src/ledger/postings.py", 12),
        ("src/ledger/currency.py", 9),
        ("src/ledger/reconcile.py", 13),
        ("src/ledger/util/money.py", 8),
        ("src/ledger/util/dates.py", 8),
        ("src/ledger/importers/csv_import.py", 10),
        ("src/ledger/importers/ofx_import.py", 10),
        ("src/ledger/importers/camt_import.py", 11),
        ("src/ledger/reports/balance.py", 10),
        ("src/ledger/reports/cashflow.py", 10),
        ("src/ledger/reports/trial_balance.py", 9),
        ("src/ledger/db/models.py", 12),
        ("src/ledger/db/migrations.py", 8),
        ("src/ledger/db/session.py", 6),
        ("src/ledger/api/routes.py", 13),
        ("src/ledger/api/schemas.py", 9),
        ("src/ledger/api/deps.py", 6),
    ]
    repo: dict[str, str] = {p: _module_src(p, n) for p, n in spec}
    for p, _ in spec:
        base = p.rsplit("/", 1)[-1]
        repo[f"tests/test_{base}"] = _test_src(base, p, 7)
    repo["README.md"] = README_SRC
    repo["pyproject.toml"] = (
        '[project]\nname = "ledger-svc"\nversion = "3.4.1"\nrequires-python = ">=3.12"\n'
        'dependencies = ["fastapi>=0.111", "sqlalchemy>=2.0", "alembic>=1.13", "httpx>=0.27"]\n\n'
        '[tool.pytest.ini_options]\naddopts = "-q --strict-markers"\ntestpaths = ["tests"]\n\n'
        '[tool.ruff]\nline-length = 100\nselect = ["E", "F", "I", "UP", "B"]\n\n'
        '[tool.mypy]\nstrict = true\nplugins = []\n'
    )
    repo["ci/last-failure.log"] = CI_LOG
    return repo


# --------------------------------------------------------------------------------------
# 3. Tool executor — against the fake repo, including realistic errors
# --------------------------------------------------------------------------------------


class FakeRepo:
    def __init__(self) -> None:
        self.files = build_repo()
        self.edits = 0
        self.commits = 0
        self.fixed = False  # changes the run_tests outcome

    # -- helpers ----------------------------------------------------------------------
    def _norm(self, p: str) -> str:
        p = (p or "").strip()
        for pre in ("/srv/ledger/", "/build/ledger-svc/", "./"):
            if p.startswith(pre):
                p = p[len(pre):]
        return p.lstrip("/")

    def _numbered(self, text: str, offset: int = 1, limit: int | None = None) -> str:
        lines = text.splitlines()
        start = max(1, offset)
        end = len(lines) if limit is None else min(len(lines), start - 1 + limit)
        return "\n".join(f"{i:6d}\t{lines[i - 1]}" for i in range(start, end + 1))

    # -- tools ------------------------------------------------------------------------
    def call(self, name: str, args: dict) -> str:
        fn = getattr(self, f"_tool_{name}", None)
        if fn is None:
            return (f"<tool_use_error>Unknown tool `{name}`. Available tools are listed in the "
                    f"system prompt.</tool_use_error>")
        try:
            return fn(args)
        except KeyError as exc:
            return f"<tool_use_error>InputValidationError: missing required argument {exc}</tool_use_error>"
        except Exception as exc:  # noqa: BLE001
            return f"<tool_use_error>{type(exc).__name__}: {exc}</tool_use_error>"

    def _tool_read(self, a: dict) -> str:
        p = self._norm(a["file_path"])
        if p not in self.files:
            near = [k for k in self.files if p.rsplit("/", 1)[-1] in k][:4]
            hint = f" Did you mean: {', '.join(near)}?" if near else ""
            return f"<tool_use_error>ENOENT: no such file '{p}'.{hint}</tool_use_error>"
        return self._numbered(self.files[p], int(a.get("offset") or 1),
                              int(a["limit"]) if a.get("limit") else None)

    def _tool_write(self, a: dict) -> str:
        p = self._norm(a["file_path"])
        new = p not in self.files
        self.files[p] = a["content"]
        self.edits += 1
        return f"{'Created' if new else 'Overwrote'} {p} ({len(a['content'].splitlines())} lines)."

    def _tool_edit(self, a: dict) -> str:
        p = self._norm(a["file_path"])
        if p not in self.files:
            return f"<tool_use_error>ENOENT: no such file '{p}'</tool_use_error>"
        old, new = a["old_string"], a["new_string"]
        body = self.files[p]
        n = body.count(old)
        if n == 0:
            return ("<tool_use_error>String to replace not found in file. The file may have "
                    "changed since you read it; read it again.</tool_use_error>")
        if n > 1 and not a.get("replace_all"):
            return (f"<tool_use_error>Found {n} matches for old_string; it must be unique or "
                    f"replace_all must be true.</tool_use_error>")
        self.files[p] = body.replace(old, new)
        self.edits += 1
        if "round_money" in old or "ROUND_HALF" in old or "mode" in old:
            self.fixed = True
        return f"Applied 1 edit to {p}."

    def _tool_multi_edit(self, a: dict) -> str:
        out = [self._tool_edit({"file_path": a["file_path"], **e}) for e in a["edits"]]
        return "\n".join(out)

    def _tool_glob(self, a: dict) -> str:
        pat = a["pattern"].replace("**/", "").replace("*", "")
        root = self._norm(a.get("path") or "")
        hits = [k for k in sorted(self.files) if k.startswith(root) and pat.strip("./") in k]
        return "\n".join(hits[:200]) or "No files matched."

    def _tool_grep(self, a: dict) -> str:
        try:
            rx = re.compile(a["pattern"], re.I if a.get("case_insensitive") else 0)
        except re.error as exc:
            return f"<tool_use_error>invalid regex: {exc}</tool_use_error>"
        root = self._norm(a.get("path") or "")
        mode = a.get("output_mode") or "content"
        hits, files = [], []
        for k in sorted(self.files):
            if not k.startswith(root):
                continue
            if a.get("glob"):
                suffix = a["glob"].rsplit("*", 1)[-1]
                if suffix and not k.endswith(suffix):
                    continue
            found = False
            for i, line in enumerate(self.files[k].splitlines(), 1):
                if rx.search(line):
                    found = True
                    hits.append(f"{k}:{i}:{line}")
            if found:
                files.append(k)
        if mode == "files_with_matches":
            return "\n".join(files) or "No matches."
        if mode == "count":
            return "\n".join(f"{f}:{sum(1 for h in hits if h.startswith(f + ':'))}" for f in files) or "0"
        return "\n".join(hits[:400]) or "No matches found."

    def _tool_ls(self, a: dict) -> str:
        root = self._norm(a["path"])
        names = sorted({k[len(root):].lstrip("/").split("/")[0]
                        for k in self.files if k.startswith(root)})
        if not names:
            return f"<tool_use_error>ENOENT: no such directory '{root}'</tool_use_error>"
        return "\n".join(names)

    def _tool_bash(self, a: dict) -> str:
        cmd = a["command"]
        if re.search(r"\bpytest\b", cmd):
            return self._tool_run_tests({})
        if cmd.strip().startswith(("ls", "find")):
            return self._tool_ls({"path": "src/ledger"})
        if cmd.strip().startswith("cat "):
            return self._tool_read({"file_path": cmd.split(maxsplit=1)[1].split()[0]})
        if "git" in cmd:
            return self._tool_git_status({})
        if re.search(r"\brm\b|\bsudo\b", cmd):
            return "<tool_use_error>Command blocked by policy (destructive).</tool_use_error>"
        return (f"$ {cmd}\n(exit 0)\n"
                f"ledger-svc 3.4.1 — 412 tests, 19 modules, {len(self.files)} tracked files")

    def _tool_run_tests(self, a: dict) -> str:
        if self.fixed:
            return ("============================= test session starts ====================\n"
                    "collected 412 items\n\n412 passed in 37.44s\n")
        return CI_LOG

    def _tool_lint(self, a: dict) -> str:
        return ("src/ledger/reconcile.py:204:9: B008 function call in default argument\n"
                "src/ledger/importers/camt_import.py:88:1: E501 line too long (112 > 100)\n"
                "Found 2 errors.")

    def _tool_typecheck(self, a: dict) -> str:
        return ("src/ledger/reconcile.py:211: error: Argument \"mode\" to \"round_money\" has "
                "incompatible type \"str | None\"; expected \"str\"  [arg-type]\n"
                "Found 1 error in 1 file (checked 19 source files)")

    def _tool_format_code(self, a: dict) -> str:
        return "reformatted 0 files, 19 files left unchanged"

    def _tool_git_status(self, a: dict) -> str:
        mods = "\n".join(f" M {p}" for p in sorted(self.files)[:self.edits])
        return f"On branch fix/ledger-412\nChanges not staged for commit:\n{mods or ' (clean)'}"

    def _tool_git_diff(self, a: dict) -> str:
        if not self.edits:
            return "(no changes)"
        return ("diff --git a/src/ledger/reconcile.py b/src/ledger/reconcile.py\n"
                "--- a/src/ledger/reconcile.py\n+++ b/src/ledger/reconcile.py\n"
                "@@ -208,7 +208,7 @@ def reconcile_period(rows, *, period, strict=False):\n"
                "-        amount = round_money(raw, mode=\"half_up\")\n"
                "+        amount = round_money(raw, mode=\"half_even\")\n")

    def _tool_git_log(self, a: dict) -> str:
        return ("a91c4f2 fix(reconcile): keep banker's rounding for accrual rows\n"
                "7d0e118 test: pin rounding drift fixture\n"
                "3b55a90 chore: bump sqlalchemy to 2.0.31\n"
                "1f2ab07 feat(camt): stream large files")

    def _tool_git_commit(self, a: dict) -> str:
        self.commits += 1
        return f"[fix/ledger-412 {self.commits:07x}] {a['message'].splitlines()[0]}\n 1 file changed"

    def _tool_postgres_readonly_execute_query(self, a: dict) -> str:
        sql = a["sql"].strip().rstrip(";")
        if re.match(r"(?i)^\s*(insert|update|delete|drop|alter|create)", sql):
            return ("<tool_use_error>permission denied: role \"analytics_ro\" cannot write "
                    "(SQLSTATE 42501)</tool_use_error>")
        if "postingz" in sql or "posting_lines" in sql:
            return ('<tool_use_error>relation "posting_lines" does not exist (SQLSTATE 42P01). '
                    'Use postgres_list_tables first.</tool_use_error>')
        return ("account | period  | kind    | n   | total\n"
                "--------+---------+---------+-----+----------\n"
                " 4010   | 2026-03 | cash    | 318 |  91244.55\n"
                " 4010   | 2026-03 | accrual |  44 |   7310.01\n"
                " 4020   | 2026-03 | cash    | 201 |  44190.00\n"
                " 4020   | 2026-03 | accrual |  12 |   1820.50\n"
                "(4 rows)")

    def _tool_postgres_list_tables(self, a: dict) -> str:
        return ("schema | table        | est_rows\n"
                "-------+--------------+---------\n"
                " public| accounts     |     1842\n"
                " public| journals     |   219044\n"
                " public| postings     |  4120887\n"
                " public| statements   |    20114\n"
                " public| fx_rates     |    88120\n"
                "(5 rows)")

    def _tool_postgres_explain(self, a: dict) -> str:
        return ("Seq Scan on postings  (cost=0.00..91244.55 rows=4120887 width=48)\n"
                "  Filter: (period = '2026-03'::text)\n"
                "Planning Time: 0.204 ms")

    def _tool_redis_get(self, a: dict) -> str:
        return "(nil)"

    def _tool_http_fetch(self, a: dict) -> str:
        return "<tool_use_error>network egress is blocked in this environment</tool_use_error>"

    def _tool_docker_ps(self, a: dict) -> str:
        return ("CONTAINER ID   IMAGE                 STATUS\n"
                "9a1c0f2b41de   ledger-svc:3.4.1      Up 4 hours\n"
                "41b7c9e0a512   postgres:16.3         Up 4 hours (healthy)")

    def _tool_docker_logs(self, a: dict) -> str:
        return TRACEBACK_2

    def _tool_k8s_get_pods(self, a: dict) -> str:
        return ("NAME                          READY   STATUS    RESTARTS\n"
                "ledger-api-7f9c8d4b6-2xqzr    1/1     Running   0\n"
                "ledger-worker-5d7b9c8-jk4lm   1/1     Running   3")

    def _tool_metrics_query(self, a: dict) -> str:
        return "ledger_reconcile_drift_cents{env=\"prod\"} 1 @1757... \n(1 series)"

    def _tool_jira_create_issue(self, a: dict) -> str:
        return f"Created LEDGER-{461 + self.commits}: {a['summary']}"

    def _tool_slack_post_message(self, a: dict) -> str:
        return f"Posted to {a['channel']} (ts 1757.{len(a['text']):04d})"

    def _tool_notebook_edit(self, a: dict) -> str:
        return "Updated 1 cell."

    def _tool_todo_write(self, a: dict) -> str:
        return f"Task list updated ({len(a['todos'])} items)."

    def _tool_task_spawn(self, a: dict) -> str:
        return "Sub-agent finished: no additional findings."


# --------------------------------------------------------------------------------------
# 4. Scenario — user turns (each one includes pasted content that grows the context)
# --------------------------------------------------------------------------------------

TASKS: list[str] = [
    "CI is red on fix/ledger-412. Here is the full failing log, pasted:\n\n```\n" + CI_LOG +
    "```\n\nFind out why test_rounding_drift fails. Start by reading the module the log points at.",

    "Now read src/ledger/util/money.py in full and explain exactly which rounding mode each "
    "caller ends up with. I want the call chain, not a guess.",

    "Find every caller of round_money across the whole repository, including tests, and tell me "
    "which ones pass mode explicitly.",

    "Read src/ledger/importers/csv_import.py and src/ledger/importers/ofx_import.py and say "
    "whether either of them relies on half_up.",

    "Apply the minimal fix in src/ledger/reconcile.py so that accrual rows use banker's "
    "rounding like everything else, then run the test suite.",

    "Check the data side too. The analytics replica has the real row mix. List the tables, then "
    "query the per-kind totals for account 4010 in period 2026-03.",

    "A second failure came in from production. Here is the traceback, pasted verbatim:\n\n```\n"
    + TRACEBACK_2 + "```\n\nFind the guard that raised it and decide whether strict=True is "
    "right for the API path.",

    "Read src/ledger/api/routes.py and src/ledger/api/schemas.py and tell me what the 422 body "
    "looks like to a client.",

    "Write a new module src/ledger/reports/drift.py that reports per-account rounding drift for "
    "a period, with full docstrings and type hints, then write its test module.",

    "Run ruff and mypy over the changed files and fix whatever they report.",

    "Show me the diff, then commit it with a proper message referencing ledger-412.",

    "Read src/ledger/db/models.py and src/ledger/reports/balance.py and tell me whether the "
    "balance report would have caught this drift.",

    "Grep for every TODO in src/ and group them by module. Then read the two modules with the "
    "most TODOs in full.",

    "The camt importer is slow (ledger-455). Read src/ledger/importers/camt_import.py in full "
    "and propose a streaming design. Do not change it yet.",

    "Open a tracker issue for ledger-455 with the findings, then post a one-paragraph summary "
    "to #ledger-dev.",

    "Read tests/test_reconcile.py and tests/test_money.py in full and tell me which assertions "
    "would break if we switched the whole codebase to half_up.",
]


# --------------------------------------------------------------------------------------
# 5. Classification
# --------------------------------------------------------------------------------------

RAW_MARKERS = ("<tool_call>", "<arg_key>", "<arg_value>", "</invoke>", "<parameter name=",
               "<tool_use_error>")


def classify_raw_content(content: str) -> list[str]:
    """Signature of a fail-closed rejection: raw tool-call syntax inside the content."""
    return [m for m in RAW_MARKERS if m in (content or "")]


def check_args(name: str, raw_args: str) -> dict:
    """Validate one tool call against its schema."""
    res: dict = {"tool": name, "json_ok": True, "known_tool": name in TOOL_SCHEMA,
                 "unknown_keys": [], "missing_required": [], "wellformed": False,
                 "raw_markers_in_args": [m for m in RAW_MARKERS if m in (raw_args or "")]}
    try:
        args = json.loads(raw_args or "{}")
        if not isinstance(args, dict):
            raise ValueError("arguments is not an object")
    except Exception as exc:  # noqa: BLE001
        res["json_ok"] = False
        res["json_error"] = f"{type(exc).__name__}: {exc}"[:160]
        return res
    schema = TOOL_SCHEMA.get(name)
    if schema is None:
        return res
    props = set(schema.get("properties", {}))
    req = set(schema.get("required", []))
    keys = set(args)
    res["unknown_keys"] = sorted(keys - props)
    res["missing_required"] = sorted(req - keys)
    res["wellformed"] = not res["unknown_keys"] and not res["missing_required"]
    res["args"] = args
    return res


# --------------------------------------------------------------------------------------
# 6. HTTP
# --------------------------------------------------------------------------------------


def post_chat(api: str, body: dict, timeout: int) -> dict:
    req = urllib.request.Request(api.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(body, ensure_ascii=False).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


# --------------------------------------------------------------------------------------
# 7. Session driver
# --------------------------------------------------------------------------------------


def run_session(sid: int, args, out_lock: threading.Lock, jsonl: pathlib.Path) -> list[dict]:
    repo = FakeRepo()
    # task order shifted per session — so sessions are independent
    order = [TASKS[(i + sid * 3) % len(TASKS)] for i in range(len(TASKS))]
    msgs: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": order[0]}]
    next_task = 1
    ctk: dict = {"clear_thinking": not args.retain, "reasoning_effort": args.effort}
    rows: list[dict] = []
    prev_calls: list[tuple[str, str]] = []
    first_corruption: dict | None = None
    pasted_tokens = 0
    ramp_files = ([p for p in sorted(repo.files) if p.startswith("src/") and p.endswith(".py")]
                  + [p for p in sorted(repo.files) if p.startswith("tests/")])
    shift = (sid * 4) % max(1, len(ramp_files))
    ramp_files = ramp_files[shift:] + ramp_files[:shift]

    for turn in range(1, args.rounds + 1):
        body: dict = {"model": args.model, "messages": msgs, "tools": TOOLS,
                      "tool_choice": "auto", "max_tokens": args.max_tokens,
                      "chat_template_kwargs": ctk}
        if args.temperature is not None:
            body["temperature"] = args.temperature
        if args.seed is not None:
            body["seed"] = args.seed + sid
        t0 = time.time()
        try:
            d = post_chat(args.api, body, args.timeout)
        except (urllib.error.HTTPError, Exception) as e:  # noqa: B014
            if isinstance(e, urllib.error.HTTPError):
                msg = f"HTTP {e.code}: {e.read()[:400].decode('utf-8', 'replace')}"
            else:
                msg = f"{type(e).__name__}: {e}"[:200]
            err = {"session": sid, "turn": turn, "error": msg}
            rows.append(err)
            with out_lock:
                with jsonl.open("a") as f:
                    f.write(json.dumps(err, ensure_ascii=False) + "\n")
                print(f"[s{sid} t{turn:02d}] HATA {msg}", flush=True)
            break
        secs = round(time.time() - t0, 2)

        ch = d["choices"][0]
        m = ch.get("message") or {}
        content = m.get("content") or ""
        reasoning = m.get("reasoning_content") or ""
        tcs = m.get("tool_calls") or []
        usage = d.get("usage") or {}
        ptk = int(usage.get("prompt_tokens") or 0)
        ctk_out = int(usage.get("completion_tokens") or 0)
        finish = ch.get("finish_reason")
        stop_reason = ch.get("stop_reason")

        # --- classification ----------------------------------------------------------
        checks = [check_args((tc.get("function") or {}).get("name") or "",
                             (tc.get("function") or {}).get("arguments") or "")
                  for tc in tcs]
        wellformed = sum(1 for c in checks if c["wellformed"] and c["known_tool"])
        schema_viol = [c for c in checks
                       if c["json_ok"] and (c["unknown_keys"] or c["missing_required"]
                                            or not c["known_tool"])]
        json_err = [c for c in checks if not c["json_ok"]]
        raw_hits = classify_raw_content(content)
        rejected = bool(raw_hits) and ("<tool_call>" in raw_hits or "<arg_key>" in raw_hits)
        empty_turn = (not content.strip()) and (not tcs) and finish == "stop"
        cur_calls = [((tc.get("function") or {}).get("name") or "",
                      (tc.get("function") or {}).get("arguments") or "") for tc in tcs]
        repeat = bool(cur_calls) and cur_calls == prev_calls
        corrupt = rejected or bool(schema_viol) or bool(json_err) or empty_turn

        row = {
            "session": sid, "turn": turn, "secs": secs,
            "prompt_tokens": ptk, "completion_tokens": ctk_out,
            "bucket": (ptk // 10000) * 10000,
            "finish_reason": finish, "stop_reason": stop_reason,
            "n_messages": len(msgs), "n_tools": len(TOOLS),
            "content_chars": len(content), "reasoning_chars": len(reasoning),
            "n_tool_calls": len(tcs), "wellformed": wellformed,
            "rejected_raw": 1 if rejected else 0, "raw_markers": raw_hits,
            "schema_violations": [{"tool": c["tool"], "unknown": c["unknown_keys"],
                                   "missing": c["missing_required"],
                                   "known_tool": c["known_tool"]} for c in schema_viol],
            "json_errors": [{"tool": c["tool"], "err": c.get("json_error")} for c in json_err],
            "empty_turn": 1 if empty_turn else 0,
            "repeat_of_previous": 1 if repeat else 0,
            "finish_length": 1 if finish == "length" else 0,
            "corrupt": 1 if corrupt else 0,
            "tools_called": [c["tool"] for c in checks],
        }
        if rejected or json_err:
            row["snippet"] = (content or (json_err[0].get("tool") if json_err else ""))[:300]
        if corrupt and first_corruption is None:
            first_corruption = {"turn": turn, "prompt_tokens": ptk,
                                "bucket": row["bucket"],
                                "kind": ("rejected" if rejected else
                                         "json_error" if json_err else
                                         "schema_violation" if schema_viol else "empty_turn")}
        # --- append the assistant message to history EXACTLY as the CLIENT would ---
        echo: dict = {"role": "assistant", "content": content}
        if reasoning and not args.drop_reasoning:
            echo["reasoning_content"] = reasoning
        if tcs:
            echo["tool_calls"] = tcs  # JSON returned by the server, unmodified
        msgs.append(echo)

        # --- tool results -------------------------------------------------------------
        if tcs:
            for tc, c in zip(tcs, checks):
                fnc = tc.get("function") or {}
                if c["json_ok"] and c["known_tool"]:
                    result = repo.call(c["tool"], c.get("args") or {})
                elif c["json_ok"]:
                    result = (f"<tool_use_error>InputValidationError: unknown tool "
                              f"'{c['tool']}'</tool_use_error>")
                else:
                    result = ("<tool_use_error>InputValidationError: arguments are not valid "
                              "JSON</tool_use_error>")
                if c["json_ok"] and c["known_tool"] and (c["unknown_keys"] or c["missing_required"]):
                    result = (f"<tool_use_error>InputValidationError: unexpected keys "
                              f"{c['unknown_keys']}, missing {c['missing_required']} for tool "
                              f"'{c['tool']}'</tool_use_error>")
                msgs.append({"role": "tool", "tool_call_id": tc.get("id") or f"call_{turn}",
                             "content": result[:30000]})
        else:
            # no tool call -> the agentic client hands over the next task
            if next_task < len(order):
                text = order[next_task]
                next_task += 1
            else:
                text = ("Good. Next: re-read the two files you changed in full and "
                        "double-check the rounding mode on every call site.")
            # Context ramp: so the measurement still reaches the 60k+ region even if
            # the model stays quiet — do what the client would do, paste a file into the message.
            if not args.no_ramp:
                target = min(args.max_prompt_tokens,
                             int(4000 + turn * args.max_prompt_tokens / max(1, args.rounds)))
                while ptk + pasted_tokens < target and ramp_files:
                    path = ramp_files.pop(0)
                    blob = repo.files[path]
                    pasted_tokens += len(blob) // 4
                    text += (f"\n\nFor context, here is the current content of `{path}` "
                             f"as my editor shows it:\n\n```python\n{blob}```\n")
            msgs.append({"role": "user", "content": text})
            row["user_task_delivered"] = True
            row["ramp_pasted_tokens"] = pasted_tokens

        rows.append(row)
        with out_lock:
            with jsonl.open("a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[s{sid} t{turn:02d}] ptk={ptk} ctk={ctk_out} fin={finish} "
                  f"calls={len(tcs)} ok={wellformed} rej={row['rejected_raw']} "
                  f"schema={len(schema_viol)} empty={row['empty_turn']} "
                  f"rep={row['repeat_of_previous']} {secs}s", flush=True)

        prev_calls = cur_calls
        if ptk > args.max_prompt_tokens:
            print(f"[s{sid}] prompt_tokens {ptk} > {args.max_prompt_tokens}, session ended",
                  flush=True)
            break

    for r in rows:
        r["first_corruption"] = first_corruption
    return rows


# --------------------------------------------------------------------------------------
# 8. Summary
# --------------------------------------------------------------------------------------

COLS = [("turns", None), ("tool_calls", "n_tool_calls"), ("wellformed", "wellformed"),
        ("rejected(b)", "rejected_raw"), ("schema_violation(c)", None), ("json_error", None),
        ("empty_turn(d)", "empty_turn"), ("repeat(e)", "repeat_of_previous"),
        ("length(f)", "finish_length")]


def summarize(rows: list[dict], args) -> tuple[dict, str]:
    ok = [r for r in rows if "error" not in r]
    errs = [r for r in rows if "error" in r]
    buckets: dict[int, list[dict]] = {}
    for r in ok:
        buckets.setdefault(r["bucket"], []).append(r)

    def agg(rs: list[dict]) -> list:
        return [len(rs),
                sum(r["n_tool_calls"] for r in rs),
                sum(r["wellformed"] for r in rs),
                sum(r["rejected_raw"] for r in rs),
                sum(len(r["schema_violations"]) for r in rs),
                sum(len(r["json_errors"]) for r in rs),
                sum(r["empty_turn"] for r in rs),
                sum(r["repeat_of_previous"] for r in rs),
                sum(r["finish_length"] for r in rs)]

    head = ["context_bucket"] + [c[0] for c in COLS]
    lines = ["| " + " | ".join(head) + " |",
             "|" + "|".join("---" for _ in head) + "|"]
    for b in sorted(buckets):
        lines.append("| " + " | ".join([f"{b//1000}–{b//1000+10}k"] +
                                       [str(x) for x in agg(buckets[b])]) + " |")
    lines.append("| " + " | ".join(["TOTAL"] + [str(x) for x in agg(ok)]) + " |")
    table = "\n".join(lines)

    firsts = {}
    for r in ok:
        fc = r.get("first_corruption")
        if fc:
            firsts[r["session"]] = fc
    sessions = sorted({r["session"] for r in ok})
    summ = {
        "api": args.api, "model": args.model, "effort": args.effort,
        "clear_thinking": not args.retain,
        "temperature": args.temperature, "seed": args.seed,
        "n_tools": len(TOOLS), "rounds_requested": args.rounds,
        "sessions": len(sessions), "turns": len(ok), "errors": len(errs),
        "max_prompt_tokens_seen": max((r["prompt_tokens"] for r in ok), default=0),
        "totals": dict(zip([c[0] for c in COLS], agg(ok))),
        "corrupt_turns": sum(r["corrupt"] for r in ok),
        "sessions_with_corruption": len(firsts),
        "first_corruption": {str(s): firsts.get(s) for s in sessions},
        "first_corruption_rate_per_session": round(len(firsts) / max(1, len(sessions)), 3),
        "corrupt_turn_ratio": round(sum(r["corrupt"] for r in ok) / max(1, len(ok)), 4),
        "total_secs": round(sum(r["secs"] for r in ok), 1),
    }
    md = [f"# issue #7 — tool-call corruption measurement ({time.strftime('%Y-%m-%d %H:%M')})", "",
          f"- API: `{args.api}` · model `{args.model}` · effort `{args.effort}` · "
          f"clear_thinking `{not args.retain}` · tool count {len(TOOLS)}",
          f"- sessions {len(sessions)} · turns {len(ok)} · errors {len(errs)} · "
          f"highest prompt_tokens {summ['max_prompt_tokens_seen']}",
          f"- **sessions with first corruption: {len(firsts)}/{len(sessions)}** · "
          f"corrupt turn ratio {summ['corrupt_turn_ratio']}", "",
          table, "", "## First corruption (per session)", ""]
    for s in sessions:
        fc = firsts.get(s)
        md.append(f"- session {s}: " + (f"turn {fc['turn']}, {fc['prompt_tokens']} tokens, "
                                       f"kind `{fc['kind']}`" if fc else "no corruption"))
    if errs:
        md += ["", "## Errors", ""] + [f"- session {e['session']} turn {e['turn']}: {e['error']}"
                                        for e in errs]
    return summ, "\n".join(md) + "\n"


# --------------------------------------------------------------------------------------
# 9. main
# --------------------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="issue #7 tool-call corruption measurement")
    ap.add_argument("--api", default="http://127.0.0.1:8001")
    ap.add_argument("--model", default="glm-5.3-flash")
    ap.add_argument("--rounds", type=int, default=30, help="maximum model turns per session")
    ap.add_argument("--sessions", type=int, default=2, help="number of independent sessions")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="number of sessions driven concurrently (each session is inherently sequential)")
    ap.add_argument("--max-prompt-tokens", type=int, default=120000)
    ap.add_argument("--max-tokens", type=int, default=2048, help="maximum generation per turn")
    ap.add_argument("--effort", choices=["low", "high"], default="low")
    ap.add_argument("--temperature", type=float, default=None,
                    help="not sent if omitted (server uses its own generation_config)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--retain", action="store_true",
                    help="clear_thinking=false (reasoning is retained in history)")
    ap.add_argument("--drop-reasoning", action="store_true",
                    help="do not echo reasoning_content back in the assistant message")
    ap.add_argument("--no-ramp", action="store_true",
                    help="disable the context ramp (if the model stays quiet, context may not reach 60k)")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--out", default=".")
    ap.add_argument("--mock", action="store_true",
                    help="start a local mock engine and connect to it (does not touch production)")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jsonl = out / "turns.jsonl"

    mock_srv = None
    if args.mock:
        import importlib.util
        mock_path = pathlib.Path(__file__).resolve().parent / "toolcall-gate-mock.py"
        spec = importlib.util.spec_from_file_location("toolcall_gate_mock", mock_path)
        mock_engine = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mock_engine)
        mock_srv, port = mock_engine.start()
        args.api = f"http://127.0.0.1:{port}"
        print(f"[mock] mock engine on 127.0.0.1:{port}", flush=True)

    if not args.mock and "127.0.0.1" not in args.api:
        print("[notice] targeting a non-local engine: " + args.api, flush=True)

    lock = threading.Lock()
    rows: list[dict] = []
    t0 = time.time()
    try:
        if args.concurrency <= 1:
            for sid in range(args.sessions):
                rows += run_session(sid, args, lock, jsonl)
        else:
            with cf.ThreadPoolExecutor(args.concurrency) as ex:
                for rs in ex.map(lambda s: run_session(s, args, lock, jsonl),
                                 range(args.sessions)):
                    rows += rs
    except KeyboardInterrupt:
        print("[interrupted] summarizing the turns collected so far", flush=True)
    finally:
        if mock_srv is not None:
            mock_srv.shutdown()

    summ, md = summarize(rows, args)
    summ["wall_secs"] = round(time.time() - t0, 1)
    (out / "summary.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False))
    (out / "summary.md").write_text(md)
    print("\n" + md)
    print(f"outputs: {jsonl}, {out/'summary.json'}, {out/'summary.md'}")
    return 0  # measurement — always 0


if __name__ == "__main__":
    sys.exit(main())
