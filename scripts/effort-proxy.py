#!/usr/bin/env python3
"""Reasoning-effort proxy: raises reasoning_effort for a benchmark run without touching the server.

Every quality number in this repository is measured at reasoning_effort `low`, which is what the
three nodes serve. To measure the same engine at a different effort we do not restart it and we do
not change a flag on it: we put this reverse proxy between the harness and the API and point the
harness's --base-url at the proxy.

It forwards everything byte-for-byte, except a POST whose path ends in /chat/completions: in that one
case it parses the JSON body, sets chat_template_kwargs.reasoning_effort, re-serialises, and
forwards. Streaming responses pass through unchanged. GET /effort-status returns the injection
counters so the run can be verified afterwards rather than assumed.

Usage:
    python3 effort-proxy.py <low|high|max> [listen_port=8011] [target=http://192.0.2.10:8001]

Then run the harness against http://127.0.0.1:<listen_port> and stop the proxy when it finishes.

Requires: aiohttp. Written by us for this recipe; use freely (Apache-2.0).
"""
import sys
import json
import asyncio  # noqa: F401  (aiohttp needs a loop policy on some distributions)
from aiohttp import web, ClientSession, ClientTimeout

EFFORT = sys.argv[1] if len(sys.argv) > 1 else "high"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8011
TARGET = sys.argv[3] if len(sys.argv) > 3 else "http://192.0.2.10:8001"
assert EFFORT in ("low", "high", "max")

COUNTERS = {"requests": 0, "injected": 0}

HOP_BY_HOP = ("host", "content-length", "transfer-encoding")


async def forward(req: web.Request):
    url = TARGET + req.rel_url.path_qs
    body = await req.read()
    headers = {k: v for k, v in req.headers.items() if k.lower() not in HOP_BY_HOP}
    COUNTERS["requests"] += 1

    if req.method == "POST" and req.path.endswith("/chat/completions") and body:
        try:
            payload = json.loads(body)
            kwargs = dict(payload.get("chat_template_kwargs") or {})
            kwargs["reasoning_effort"] = EFFORT
            payload["chat_template_kwargs"] = kwargs
            body = json.dumps(payload).encode()
            headers["Content-Type"] = "application/json"
            COUNTERS["injected"] += 1
        except Exception as exc:  # a body we cannot parse is forwarded untouched, and said so
            print("could not inject:", exc, file=sys.stderr, flush=True)

    async with ClientSession(timeout=ClientTimeout(total=None)) as session:
        async with session.request(req.method, url, data=body, headers=headers) as upstream:
            out = web.StreamResponse(
                status=upstream.status,
                headers={
                    k: v for k, v in upstream.headers.items()
                    if k.lower() not in ("content-length", "transfer-encoding", "content-encoding")
                },
            )
            await out.prepare(req)
            async for chunk in upstream.content.iter_any():
                await out.write(chunk)
            await out.write_eof()
            return out


async def status(req):
    return web.json_response({"effort": EFFORT, "target": TARGET, **COUNTERS})


app = web.Application(client_max_size=1024 ** 3)
app.router.add_get("/effort-status", status)
app.router.add_route("*", "/{path:.*}", forward)
print(f"effort proxy: :{PORT} -> {TARGET}  reasoning_effort={EFFORT}", flush=True)
web.run_app(app, host="127.0.0.1", port=PORT, print=None)
