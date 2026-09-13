#!/usr/bin/env python3
"""prefill-interference.py — do decoding streams slow down while another stream's long prompt is read?

Runs D prose decode streams (short prompt, long answer) against an OpenAI-compatible endpoint and
reports each stream's delivered chunks per window. At t = --at seconds one long prompt (--ctx tokens,
max_tokens 2) is fired at the same engine: chunked-prefill interference shows up as a dip in the
decoders' windows. Sends requests only; changes nothing on the engine.

  BASE=http://192.0.2.10:8001 python3 prefill-interference.py --dec 3 --ctx 100000 --at 20 \
      --max-tokens 1500 --corpus /path/to/some/text/files

The long prompt is built from text files under --corpus (a fresh, non-repeating prompt is the point:
a repeated prompt would be served from the prefix cache and nothing would be read). Without --corpus a
seeded synthetic filler is used; it is a valid load but a less realistic one.
"""
import argparse, json, os, random, sys, threading, time, urllib.request

BASE = os.environ.get("BASE", "http://127.0.0.1:8000").rstrip("/")
MODEL = os.environ.get("MODEL", "glm-5.3-flash")
HDR = {"Content-Type": "application/json"}
DIRECTIVE = "/nothink Answer directly, no preamble."


def tokenize_count(text):
    req = urllib.request.Request(BASE + "/tokenize", data=json.dumps({"model": MODEL, "prompt": text}).encode(),
                                 headers=HDR)
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["count"]


def corpus_files(root):
    out = []
    for base, _, files in os.walk(root):
        if "/.git" in base or "/node_modules" in base:
            continue
        for f in files:
            p = os.path.join(base, f)
            try:
                if 300 < os.path.getsize(p) < 120_000 and f.rsplit(".", 1)[-1] in (
                        "md", "py", "gd", "txt", "rs", "ts", "js", "c", "h", "cpp", "yaml", "toml", "cfg"):
                    out.append(p)
            except OSError:
                pass
    return sorted(out)


def build_prompt(target_tokens, seed, files):
    rng = random.Random(seed)
    parts = [f"Session {seed}: you are reviewing a software project. Files follow.\n"]
    chars, target_chars = 0, int(target_tokens * 3.4)
    if files:
        files = list(files); rng.shuffle(files)
        for p in files:
            if chars >= target_chars:
                break
            try:
                t = open(p, errors="replace").read()
            except OSError:
                continue
            blk = f"\n### {p}\n```\n{t[:60000]}\n```\n"; parts.append(blk); chars += len(blk)
    else:
        words = ["alpha", "beta", "gamma", "delta", "sigma", "omega", "vector", "matrix", "buffer", "stream"]
        while chars < target_chars:
            line = " ".join(rng.choice(words) + str(rng.randrange(1000)) for _ in range(12)) + "\n"
            parts.append(line); chars += len(line)
    task = ("\nTask: in about 250 words of plain prose, describe the architecture you see and name three "
            "concrete risks. End with 'END OF REVIEW'.")
    text = "".join(parts) + task
    n = tokenize_count(text)
    if n > target_tokens * 1.08:
        text = text[: int(len(text) * target_tokens / n)] + "\n```\n" + task
        n = tokenize_count(text)
    return text, n


def stream(prompt, max_tokens, effort, on_chunk, tag):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "stream": True,
            "max_tokens": max_tokens, "temperature": 0.7,
            "chat_template_kwargs": {"clear_thinking": True, "reasoning_effort": effort}}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(), headers=HDR)
    t0 = time.time(); first = None; n = 0
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            try:
                ch = json.loads(line[5:])
            except Exception:
                continue
            d = (ch.get("choices") or [{}])[0].get("delta") or {}
            if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
                if first is None:
                    first = time.time()
                n += 1
                on_chunk(tag, time.time())
    return t0, first, time.time(), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dec", type=int, default=3)
    ap.add_argument("--ctx", type=int, default=100000)
    ap.add_argument("--at", type=float, default=20.0)
    ap.add_argument("--max-tokens", type=int, default=1500)
    ap.add_argument("--win", type=float, default=5.0)
    ap.add_argument("--effort", default="low")
    ap.add_argument("--corpus", default="")
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--out", default="prefill-interference.json")
    a = ap.parse_args()

    files = corpus_files(a.corpus) if a.corpus else []
    print(f"corpus {len(files)} files; building {a.ctx}-token prompt ...", flush=True)
    big, big_n = build_prompt(a.ctx, a.seed, files)
    print(f"long prompt = {big_n} tokens", flush=True)

    dec_prompt = (DIRECTIVE + "\nWrite a 900-word essay in plain prose (no lists, no headings) about how a small "
                  "software team should plan its first product: audience, scope, pacing, testing discipline, "
                  "and launch. Stream {k} of the team notes.")
    T0 = time.time(); lock = threading.Lock(); hits = {}

    def on_chunk(tag, ts):
        with lock:
            hits.setdefault(tag, []).append(ts - T0)

    results = {}

    def run_dec(k):
        results[f"dec{k}"] = stream(dec_prompt.format(k=k), a.max_tokens, a.effort, on_chunk, f"dec{k}")

    def run_big():
        time.sleep(a.at)
        print(f"[{time.time()-T0:6.1f}s] long prompt fired ({big_n} tok)", flush=True)
        results["big"] = stream(big, 2, a.effort, on_chunk, "big")
        t0b, firstb, t1b, _ = results["big"]
        print(f"[{time.time()-T0:6.1f}s] long prompt done (prefill {(firstb or t1b) - t0b:.1f}s)", flush=True)

    th = [threading.Thread(target=run_dec, args=(k,)) for k in range(1, a.dec + 1)] + [threading.Thread(target=run_big)]
    for t in th: t.start()
    for t in th: t.join()
    end = time.time() - T0

    big_start, big_end = a.at, results["big"][2] - T0
    print("\nwindow(s)   " + "  ".join(f"dec{k:<4d}" for k in range(1, a.dec + 1)) + "   phase")
    w = 0.0
    while w < end:
        row = [f"{sum(1 for t in hits.get(f'dec{k}', []) if w <= t < w + a.win) / a.win:6.1f}  " for k in range(1, a.dec + 1)]
        phase = "PREFILL" if big_start <= w < big_end else ("before" if w < big_start else "after")
        print(f"{w:5.0f}-{w+a.win:<4.0f}  " + "".join(row) + f"  {phase}")
        w += a.win
    print("\nper-stream totals (chunks/s over own decode phase):")
    for k in range(1, a.dec + 1):
        t0, first, t1, n = results[f"dec{k}"]
        print(f"  dec{k}: {n} chunks, TTFT {first - t0:.1f}s, {n / max(t1 - first, 1e-6):.1f} chunks/s")
    t0, first, t1, n = results["big"]
    dt = (first or t1) - t0
    print(f"  long: prefill {big_n} tok in {dt:.1f}s = {big_n / max(dt, 1e-6):.0f} tok/s")
    json.dump({"hits": hits, "long": [big_n, dt], "args": vars(a)}, open(a.out, "w"))


if __name__ == "__main__":
    main()
