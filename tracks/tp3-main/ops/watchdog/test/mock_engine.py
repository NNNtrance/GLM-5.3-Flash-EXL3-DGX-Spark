#!/usr/bin/env python3
"""A MOCK vLLM front door for the watchdog tests (local only). Its behaviour is read from the config file
on every request:
{"health":200, "chat_delay":0, "chat_code":200, "metrics_move":false}
"""
import json, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CONFIG = sys.argv[2]
PORT = int(sys.argv[1])
COUNTER = [1000]


def config():
    try:
        return json.load(open(CONFIG))
    except Exception:
        return {}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _write(self, code, body, ctype="application/json"):
        b = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        try:
            self.wfile.write(b)
        except Exception:
            pass

    def do_GET(self):
        c = config()
        if self.path == "/health":
            self._write(c.get("health", 200), "")
        elif self.path == "/metrics":
            if c.get("metrics_move"):
                COUNTER[0] += 500
            self._write(200, f'vllm:generation_tokens_total{{engine="0"}} {COUNTER[0]}.0\n'
                             f'vllm:prompt_tokens_total{{engine="0"}} 7.0e+03\n'
                             f'vllm:generation_tokens_created{{engine="0"}} 1.7e9\n', "text/plain")
        else:
            self._write(404, "")

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        self.rfile.read(n)
        c = config()
        time.sleep(float(c.get("chat_delay", 0)))
        code = c.get("chat_code", 200)
        self._write(code, json.dumps({"choices": [{"message": {"content": "2"}, "finish_reason": "length"}]})
                    if code == 200 else json.dumps({"error": "x"}))


ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
