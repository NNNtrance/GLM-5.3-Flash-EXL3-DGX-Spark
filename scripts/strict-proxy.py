#!/usr/bin/env python3
"""strict-proxy.py — a read-only-to-the-engine shim that marks every function tool
`strict: true` on the way to an OpenAI-compatible vLLM server.

Why: in vLLM, `get_model_structural_tag()` (vllm/tool_parsers/structural_tag_registry.py)
returns None for `tool_choice="auto"` unless `_any_tool_strict(tools)` is true, so the
xgrammar structural-tag constraint — the sampling-layer guard that makes malformed
tool-call syntax unsamplable — never activates for clients that do not set the field.
`VLLM_ENFORCE_STRICT_TOOL_CALLING` defaults to True but does not lift that gate.

This proxy lets an unmodified client (e.g. scripts/toolcall-gate.py, Hermes) be A/B'd
against the same engine with the grammar on, with no engine-side and no client-side
change: point the client at the proxy instead of the engine.

Two modes:
  * default          — set `tools[i].function.strict = true` and nothing else.
  * --schema-strict  — additionally rewrite each `parameters` schema into OpenAI
                       strict-schema style: recursive `additionalProperties: false`
                       and every declared property listed in `required`. Use only if
                       the flag-only mode misbehaves; it changes what the model is
                       told the tools look like.

Everything that is not `POST /v1/chat/completions` is forwarded byte for byte
(`/v1/models`, `/tokenize`, `/metrics`, …). Streaming (SSE) and non-streaming both
pass through; one JSON line per request is appended to the log with the sizes and
timings needed to price grammar compilation.

Stdlib only. Python 3.12.

Usage:
    ./strict-proxy.py --upstream http://127.0.0.1:8001 --listen 127.0.0.1:8011 \
        --log /tmp/strict-proxy.jsonl
    ./strict-proxy.py --passthrough        # same plumbing, no injection (control arm)
"""
from __future__ import annotations

import argparse
import http.server
import json
import sys
import threading
import time
import urllib.error
import urllib.request

CFG: dict = {}
_LOCK = threading.Lock()
_SEQ = [0]


# --------------------------------------------------------------------------------------
# schema surgery
# --------------------------------------------------------------------------------------


def strictify_schema(node) -> None:
    """Recursively force OpenAI strict-schema style on a JSON Schema fragment."""
    if isinstance(node, list):
        for item in node:
            strictify_schema(item)
        return
    if not isinstance(node, dict):
        return
    props = node.get("properties")
    if isinstance(props, dict):
        node["additionalProperties"] = False
        node["required"] = list(props.keys())
        for sub in props.values():
            strictify_schema(sub)
    for key in ("items", "additionalItems", "contains", "not"):
        if key in node:
            strictify_schema(node[key])
    for key in ("anyOf", "oneOf", "allOf", "prefixItems"):
        if isinstance(node.get(key), list):
            strictify_schema(node[key])
    for key in ("$defs", "definitions", "patternProperties"):
        if isinstance(node.get(key), dict):
            for sub in node[key].values():
                strictify_schema(sub)


def inject(body: dict) -> int:
    """Mark every function tool strict. Returns how many tools were touched."""
    tools = body.get("tools")
    if not isinstance(tools, list):
        return 0
    n = 0
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function")
        if not isinstance(fn, dict):
            continue
        fn["strict"] = True
        n += 1
        if CFG["schema_strict"]:
            params = fn.get("parameters")
            if isinstance(params, dict):
                strictify_schema(params)
    return n


# --------------------------------------------------------------------------------------
# logging
# --------------------------------------------------------------------------------------


def log(rec: dict) -> None:
    rec["ts"] = round(time.time(), 3)
    line = json.dumps(rec, ensure_ascii=False)
    with _LOCK:
        with open(CFG["log"], "a", encoding="utf-8") as f:
            f.write(line + "\n")
    if CFG["verbose"]:
        print(line, flush=True)


# --------------------------------------------------------------------------------------
# handler
# --------------------------------------------------------------------------------------


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "strict-proxy/1.0"

    def log_message(self, *a) -> None:  # silence the default stderr chatter
        pass

    # ---- plumbing --------------------------------------------------------------------

    def _read_body(self) -> bytes:
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    def _upstream_headers(self) -> dict:
        drop = {"host", "content-length", "connection", "accept-encoding",
                "transfer-encoding"}
        return {k: v for k, v in self.headers.items() if k.lower() not in drop}

    def _forward(self, body: bytes, headers: dict, meta: dict) -> None:
        url = CFG["upstream"].rstrip("/") + self.path
        req = urllib.request.Request(url, data=body or None, method=self.command,
                                     headers=headers)
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=CFG["timeout"]) as r:
                t_hdr = time.time()
                ctype = (r.headers.get("Content-Type") or "").lower()
                streaming = "text/event-stream" in ctype
                self.send_response(r.status)
                for k, v in r.headers.items():
                    if k.lower() in ("transfer-encoding", "content-length", "connection"):
                        continue
                    self.send_header(k, v)
                self.send_header("Connection", "close")
                self.end_headers()

                if streaming:
                    nbytes = 0
                    nchunks = 0
                    t_first = None
                    tail = b""
                    while True:
                        chunk = r.read(4096)
                        if not chunk:
                            break
                        if t_first is None:
                            t_first = time.time()
                        nbytes += len(chunk)
                        nchunks += 1
                        tail = (tail + chunk)[-8192:]
                        self.wfile.write(chunk)
                        self.wfile.flush()
                    meta.update({
                        "stream": True, "status": r.status, "bytes": nbytes,
                        "sse_chunks": nchunks,
                        "ttfb_ms": None if t_first is None else round((t_first - t0) * 1000),
                        "hdr_ms": round((t_hdr - t0) * 1000),
                        "total_ms": round((time.time() - t0) * 1000),
                    })
                    meta.update(_scan_sse_tail(tail))
                    log(meta)
                    return

                raw = r.read()
                self.wfile.write(raw)
                meta.update({
                    "stream": False, "status": r.status, "bytes": len(raw),
                    "hdr_ms": round((t_hdr - t0) * 1000),
                    "total_ms": round((time.time() - t0) * 1000),
                })
                meta.update(_scan_json_response(raw))
                log(meta)
        except urllib.error.HTTPError as e:
            raw = e.read()
            self.send_response(e.code)
            self.send_header("Content-Type", e.headers.get("Content-Type",
                                                           "application/json"))
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(raw)
            meta.update({"status": e.code, "error": raw[:400].decode("utf-8", "replace"),
                         "total_ms": round((time.time() - t0) * 1000)})
            log(meta)
        except Exception as e:  # upstream unreachable / timeout
            msg = f"{type(e).__name__}: {e}"
            raw = json.dumps({"error": {"message": "strict-proxy upstream failure: "
                                                   + msg}}).encode()
            try:
                self.send_response(502)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(raw)
            except Exception:
                pass
            meta.update({"status": 502, "error": msg,
                         "total_ms": round((time.time() - t0) * 1000)})
            log(meta)

    # ---- verbs -----------------------------------------------------------------------

    def _handle(self) -> None:
        body = self._read_body()
        _SEQ[0] += 1
        meta = {"seq": _SEQ[0], "path": self.path, "method": self.command}
        is_chat = self.command == "POST" and self.path.rstrip("/").endswith(
            "/v1/chat/completions")
        if is_chat and body:
            try:
                d = json.loads(body)
            except Exception as e:
                meta["parse_error"] = str(e)
                self._forward(body, self._upstream_headers(), meta)
                return
            n_tools = len(d.get("tools") or [])
            n_inj = 0 if CFG["passthrough"] else inject(d)
            meta.update({
                "n_tools": n_tools, "strict_injected": n_inj,
                "schema_strict": CFG["schema_strict"] and not CFG["passthrough"],
                "tool_choice": d.get("tool_choice"),
                "n_messages": len(d.get("messages") or []),
                "req_stream": bool(d.get("stream")),
                "max_tokens": d.get("max_tokens"),
                "chat_template_kwargs": d.get("chat_template_kwargs"),
            })
            body = json.dumps(d, ensure_ascii=False).encode()
        headers = self._upstream_headers()
        if body:
            headers["Content-Length"] = str(len(body))
        self._forward(body, headers, meta)

    do_GET = _handle
    do_POST = _handle
    do_PUT = _handle
    do_DELETE = _handle


# --------------------------------------------------------------------------------------
# response scanners (cheap, best-effort — never raise)
# --------------------------------------------------------------------------------------


def _scan_json_response(raw: bytes) -> dict:
    out: dict = {}
    try:
        d = json.loads(raw)
    except Exception:
        return out
    usage = d.get("usage") or {}
    if usage:
        out["prompt_tokens"] = usage.get("prompt_tokens")
        out["completion_tokens"] = usage.get("completion_tokens")
    ch = (d.get("choices") or [{}])[0]
    if isinstance(ch, dict):
        out["finish_reason"] = ch.get("finish_reason")
        msg = ch.get("message") or {}
        if isinstance(msg, dict):
            out["n_tool_calls"] = len(msg.get("tool_calls") or [])
            out["content_chars"] = len(msg.get("content") or "")
    return out


def _scan_sse_tail(tail: bytes) -> dict:
    out: dict = {}
    try:
        text = tail.decode("utf-8", "replace")
    except Exception:
        return out
    for line in reversed(text.splitlines()):
        if not line.startswith("data: ") or line.strip() == "data: [DONE]":
            continue
        try:
            d = json.loads(line[6:])
        except Exception:
            continue
        usage = d.get("usage") or {}
        if usage:
            out["prompt_tokens"] = usage.get("prompt_tokens")
            out["completion_tokens"] = usage.get("completion_tokens")
        ch = (d.get("choices") or [{}])[0]
        if isinstance(ch, dict) and ch.get("finish_reason"):
            out["finish_reason"] = ch.get("finish_reason")
        if out:
            break
    return out


# --------------------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="inject strict:true into OpenAI function tools")
    ap.add_argument("--upstream", default="http://127.0.0.1:8001")
    ap.add_argument("--listen", default="127.0.0.1:8011")
    ap.add_argument("--log", default="/tmp/strict-proxy.jsonl")
    ap.add_argument("--schema-strict", action="store_true",
                    help="also rewrite parameters to OpenAI strict-schema style "
                         "(recursive additionalProperties:false, all fields required)")
    ap.add_argument("--passthrough", action="store_true",
                    help="control arm: identical plumbing and logging, no injection")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("-v", "--verbose", action="store_true", help="echo the log to stdout")
    a = ap.parse_args()

    host, _, port = a.listen.rpartition(":")
    CFG.update({"upstream": a.upstream, "log": a.log, "schema_strict": a.schema_strict,
                "passthrough": a.passthrough, "timeout": a.timeout, "verbose": a.verbose})
    mode = ("passthrough" if a.passthrough
            else "strict+schema" if a.schema_strict else "strict-flag-only")
    print(f"strict-proxy {mode}: http://{host}:{port} -> {a.upstream}  log={a.log}",
          flush=True)
    srv = http.server.ThreadingHTTPServer((host or "127.0.0.1", int(port)), Handler)
    srv.daemon_threads = True
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
