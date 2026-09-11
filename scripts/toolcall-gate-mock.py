#!/usr/bin/env python3
"""toolcall-gate-mock.py — local mock engine for toolcall-gate.py (OpenAI-compatible,
stdlib only).

DOES NOT TOUCH THE PRODUCTION ENGINE. It only speaks `/v1/chat/completions` and returns
scripted replies: well-formed tool calls, a schema-violating call, a malformed JSON
argument, two raw `<tool_call>` text blobs captured from issue #7 (the shape a
fail-closed rejection takes on the client side), an empty turn, a repeated call, and a
`finish_reason=length`. A 12-step cycle, so every counter toolcall-gate.py tracks gets
triggered at least once.

Usage:
    import importlib; m = importlib.import_module("toolcall-gate-mock")
    srv, port = m.start()                   # in a thread, picks a free port
    python3 toolcall-gate-mock.py --port 8099    # standalone
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Real corruption vectors captured in issue #7 (the fail-closed patch surfaces these
# as plain text; this is exactly how they appear on the client side).
RAW_1 = ("<tool_call>Bash<arg_key>S1_latest.json</arg_value><arg_key>description</arg_key>"
         "Print full source of multi_zones.py b</arg_value></tool_call>")
RAW_2 = ("<tool_call>Bash<arg_key>Simpsons<parameter name=\"command\">cat "
         "/srv/ledger/src/ledger/reconcile.py</invoke><tool_use_error>"
         "InputValidationError: ...")


def _tc(i: int, name: str, args_obj_or_str) -> dict:
    raw = args_obj_or_str if isinstance(args_obj_or_str, str) else json.dumps(args_obj_or_str)
    return {"id": f"call_mock_{i}", "type": "function",
            "function": {"name": name, "arguments": raw}}


def script(idx: int) -> dict:
    """idx = number of assistant messages in the request. A 12-step cycle triggers every counter once."""
    k = idx % 12
    if k == 0:  # (a) one well-formed call
        return {"content": "", "reasoning": "The log points at reconcile.py; read it first.",
                "tool_calls": [_tc(idx, "read", {"file_path": "src/ledger/reconcile.py"})],
                "finish": "tool_calls"}
    if k == 1:  # (a) two well-formed calls
        return {"content": "", "reasoning": "Need money.py and all call sites.",
                "tool_calls": [_tc(idx, "read", {"file_path": "src/ledger/util/money.py"}),
                               _tc(idx + 100, "grep", {"pattern": "round_money", "path": "src",
                                                       "output_mode": "content"})],
                "finish": "tool_calls"}
    if k == 2:  # (a) well-formed but nonexistent file -> realistic tool error
        return {"content": "", "reasoning": "",
                "tool_calls": [_tc(idx, "read", {"file_path": "src/ledger/rounding_rules.py"})],
                "finish": "tool_calls"}
    if k == 3:  # (c) parsed but out-of-schema key
        return {"content": "", "reasoning": "",
                "tool_calls": [_tc(idx, "read", {"file_path": "src/ledger/util/money.py",
                                                 "recursive": True, "max_depth": 3})],
                "finish": "tool_calls"}
    if k == 4:  # (b) fail-closed rejection — raw text as content
        return {"content": RAW_1, "reasoning": "", "tool_calls": [], "finish": "stop"}
    if k == 5:  # (d) empty turn
        return {"content": "", "reasoning": "", "tool_calls": [], "finish": "stop",
                "stop_reason": 154829}
    if k == 6:  # (a) well-formed
        return {"content": "", "reasoning": "",
                "tool_calls": [_tc(idx, "read", {"file_path": "src/ledger/ledger.py"})],
                "finish": "tool_calls"}
    if k == 7:  # (e) identical to the previous call
        return {"content": "", "reasoning": "",
                "tool_calls": [_tc(idx, "read", {"file_path": "src/ledger/ledger.py"})],
                "finish": "tool_calls"}
    if k == 8:  # (b) second rejection vector — </invoke> / <parameter name= fragments
        return {"content": RAW_2, "reasoning": "", "tool_calls": [], "finish": "stop"}
    if k == 9:  # (f) finish_reason=length
        return {"content": "I will now walk the call chain in detail, starting from the importer "
                           "boundary and following every round_money call site until the " * 6,
                "reasoning": "", "tool_calls": [], "finish": "length"}
    if k == 10:  # malformed JSON argument (an upstream parser would have salvaged this)
        return {"content": "", "reasoning": "",
                "tool_calls": [_tc(idx, "edit", '{"file_path": "src/ledger/reconcile.py", '
                                                '"old_string": "mode=\\"half_up\\""')],
                "finish": "tool_calls"}
    return {"content": "Done: accrual rows now use banker's rounding and the suite is green.",
            "reasoning": "", "tool_calls": [], "finish": "stop"}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    checked = False

    def log_message(self, fmt: str, *a) -> None:  # silent
        pass

    def _json(self, code: int, obj: dict) -> None:
        data = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.startswith("/v1/models"):
            self._json(200, {"object": "list", "data": [{"id": "glm-5.3-flash"}]})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if not self.path.startswith("/v1/chat/completions"):
            self._json(404, {"error": "not found"})
            return
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n)
        try:
            body = json.loads(raw)
        except Exception as exc:  # noqa: BLE001
            self._json(400, {"error": f"bad json: {exc}"})
            return

        msgs = body.get("messages") or []
        tools = body.get("tools") or []
        ctk = body.get("chat_template_kwargs") or {}
        if not tools:
            self._json(400, {"error": "mock: request carries no `tools` — harness bug"})
            return
        if "clear_thinking" not in ctk:
            self._json(400, {"error": "mock: chat_template_kwargs.clear_thinking missing"})
            return
        if not Handler.checked:
            Handler.checked = True
            print(f"[mock] first request validated: tools={len(tools)}, "
                  f"chat_template_kwargs={ctk}, temperature="
                  f"{body.get('temperature', '(not sent)')}", file=sys.stderr, flush=True)

        idx = sum(1 for m in msgs if m.get("role") == "assistant")
        s = script(idx)
        msg: dict = {"role": "assistant", "content": s["content"]}
        if s.get("reasoning"):
            msg["reasoning_content"] = s["reasoning"]
        if s.get("tool_calls"):
            msg["tool_calls"] = s["tool_calls"]

        ptk = max(1, len(raw) // 4)
        gen = s["content"] + json.dumps(s.get("tool_calls") or []) + s.get("reasoning", "")
        choice: dict = {"index": 0, "message": msg, "finish_reason": s["finish"]}
        if s.get("stop_reason") is not None:
            choice["stop_reason"] = s["stop_reason"]
        self._json(200, {
            "id": f"chatcmpl-mock-{idx}", "object": "chat.completion",
            "created": int(time.time()), "model": body.get("model") or "glm-5.3-flash",
            "choices": [choice],
            "usage": {"prompt_tokens": ptk, "completion_tokens": max(1, len(gen) // 4) + 40,
                      "total_tokens": ptk + max(1, len(gen) // 4) + 40},
        })


def start(port: int = 0) -> tuple[ThreadingHTTPServer, int]:
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8099)
    a = ap.parse_args()
    srv, port = start(a.port)
    print(f"mock engine http://127.0.0.1:{port} (Ctrl+C to stop)", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        srv.shutdown()
