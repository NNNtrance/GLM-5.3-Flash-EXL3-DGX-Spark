#!/usr/bin/env python3
"""needle-lite, six depths, production settings.

Six needles at six depths of one ~48k-token haystack, one request each,
temperature 0, thinking LEFT ON (production runs reasoning_effort=low and this
model sometimes puts the whole answer in `reasoning_content` with `content`
empty -- so, like correctness-probe.py, this scores `content` and reports
`either` separately).  `enable_thinking=false` is never sent: it is banned on
this stack.

    needle-lite6.py [API]        default http://192.0.2.10:8001
Prints "NEEDLE-LITE: n/6" as its last line; exit 0 only on 6/6.
"""

import json
import random
import sys
import urllib.request

API = sys.argv[1] if len(sys.argv) > 1 else "http://192.0.2.10:8001"
# "concurrent" as argv[2]: fire all six needles at once instead of one at a
# time.  Concurrency is the axis that matters for anything sharing a per-request
# scratch buffer: a bug that a single sequential request hides by finding its own
# leftovers shows up the moment two requests are in flight.
CONCURRENT = len(sys.argv) > 2 and sys.argv[2] == "concurrent"
URL = API + "/v1/chat/completions"

WORDS = (
    "harbour lantern gravel meadow cinder ledger anchor thistle furnace pebble "
    "willow cavern marble drifting quarry beacon thicket ember lattice hollow "
).split()
CODES = ["QX7-4412", "ZM3-9087", "KR8-2251", "TV5-6630", "BN2-3374", "LP9-8105"]
# fractional depths through the haystack
DEPTHS = [0.02, 0.2, 0.4, 0.6, 0.8, 0.97]


def filler(n_words: int, seed: int) -> list:
    rnd = random.Random(seed)
    return [rnd.choice(WORDS) for _ in range(n_words)]


def call(body, timeout=600):
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def main() -> None:
    base = filler(42000, 4242)

    def build(i, depth, code):
        words = list(base)
        at = int(len(words) * depth)
        words[at:at] = f"The vault code for room {i + 1} is {code} .".split()
        hay = " ".join(words)
        return (
            "You are given a long document. Somewhere inside it there is a line of the "
            f"form 'The vault code for room {i + 1} is XXX-NNNN'. Find it.\n\n"
            "=== DOCUMENT ===\n" + hay + "\n=== END DOCUMENT ===\n\n"
            f"Reply with the vault code for room {i + 1} and nothing else."
        )

    def run(i):
        depth, code = DEPTHS[i], CODES[i]
        try:
            d = call(
                {
                    "model": "glm-5.3-flash",
                    "messages": [{"role": "user", "content": build(i, depth, code)}],
                    "temperature": 0,
                    "max_tokens": 64,
                }
            )
            msg = d["choices"][0]["message"]
            content = msg.get("content") or ""
            reasoning = msg.get("reasoning_content") or ""
            n = d["usage"]["prompt_tokens"]
            in_c = code in content
            return (i, depth, n, in_c, in_c or (code in reasoning), content.strip()[:48], None)
        except Exception as e:  # noqa: BLE001
            return (i, depth, 0, False, False, "", str(e))

    if CONCURRENT:
        import concurrent.futures as cf

        with cf.ThreadPoolExecutor(max_workers=6) as ex:
            results = sorted(ex.map(run, range(6)))
    else:
        results = [run(i) for i in range(6)]

    ok_content = sum(1 for r in results if r[3])
    ok_either = sum(1 for r in results if r[4])
    for i, depth, n, in_c, in_e, txt, err in results:
        if err:
            print(f"  depth {depth:.2f}: ERROR {err}")
        else:
            print(
                f"  depth {depth:.2f} (n={n:,}): "
                f"{'PASS' if in_c else ('EITHER-ONLY' if in_e else 'FAIL')} -> {txt!r}"
            )
    tag = "concurrent" if CONCURRENT else "sequential"
    print(f"NEEDLE-LITE ({tag}): {ok_content}/6  (either {ok_either}/6)")
    sys.exit(0 if ok_content == 6 else 1)


if __name__ == "__main__":
    main()
