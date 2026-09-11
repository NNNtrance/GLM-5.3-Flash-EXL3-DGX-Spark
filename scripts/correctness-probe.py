#!/usr/bin/env python3
"""Output-correctness probe: 10 short questions with machine-checkable answers,
plus one long paragraph checked for repetition locks.

A broken MoE path (experts multiplied by zero, a mis-shaped decode kernel)
produces repetition locks or fluent nonsense here; a healthy engine passes
10/10 clean. This is the cheapest gate we have -- run it after every engine
start and after every configuration change.

Usage: python3 correctness-probe.py [API_BASE]   default http://192.0.2.10:8001
Exit code 0 on a full pass, 1 otherwise.
Written by us for this recipe; use freely (Apache-2.0).

Two counts are reported on purpose. With thinking ON the model sometimes puts
the whole answer in the reasoning field and leaves content EMPTY -- same
prompt, same settings, even at temperature 0. So:
  content  : what a plain client sees
  both     : what the model actually knows"""
import json,os,sys,urllib.request,re,collections
BASE=sys.argv[1] if len(sys.argv)>1 else os.environ.get("API","http://192.0.2.10:8001")
def ask(p,mx=400,temp=0.0):
    body=json.dumps({"model":"glm-5.3-flash","messages":[{"role":"user","content":p}],
                     "max_tokens":mx,"temperature":temp,
                     "chat_template_kwargs":{"enable_thinking":True,"reasoning_effort":"low"},}).encode()
    r=urllib.request.Request(BASE+"/v1/chat/completions",body,{"Content-Type":"application/json"})
    with urllib.request.urlopen(r,timeout=300) as h:
        m=json.load(h)["choices"][0]["message"]
    # With thinking ON the model sometimes puts the whole answer in the
    # 'reasoning' field and leaves 'content' EMPTY -- same prompt, same
    # settings, and it varies even at temperature 0. Hence the two counts
    # described in the module docstring.
    return (m.get("content") or ""), (m.get("reasoning_content") or m.get("reasoning") or "")

TESTS=[
 ("arith-1","What is 17 multiplied by 23? Reply with only the number.", lambda s: "391" in s),
 ("arith-2","What is 2 to the power of 10? Reply with only the number.", lambda s: "1024" in s),
 ("arith-3","What is 144 divided by 12? Reply with only the number.", lambda s: "12" in s),
 ("fact-1","What is the capital city of Japan? Reply with only the city name.", lambda s: "tokyo" in s.lower()),
 ("fact-2","What is the chemical symbol for gold? Reply with only the symbol.", lambda s: re.search(r'\bAu\b',s) is not None),
 ("fact-3","Who wrote the play Hamlet? Reply with only the name.", lambda s: "shakespeare" in s.lower()),
 ("code-1","Write a Python function named rev that returns its string argument reversed. Output only code, no explanation.",
   lambda s: "def rev" in s and "[::-1]" in s.replace(" ","")),
 ("code-2","Write a Python one-liner that sums the integers 1 to 100 using sum() and range(). Output only code.",
   lambda s: "sum(" in s and "range(" in s),
 ("logic-1","Alice is taller than Bob. Bob is taller than Carol. Who is the shortest? Reply with only the name.",
   lambda s: "carol" in s.lower()),
]
def looks_coherent(t):
    w=re.findall(r"[A-Za-z']+",t)
    if len(w)<25: return False,f"too short ({len(w)} words)"
    c=collections.Counter(x.lower() for x in w)
    top,n=c.most_common(1)[0]
    if n/len(w)>0.25: return False,f"repetition lock: '{top}' {100*n/len(w):.0f}%"
    if len(set(c))/len(w)<0.30: return False,"word variety too low"
    if sum(1 for ch in t if ord(ch)==0xFFFD): return False,"U+FFFD replacement characters"
    return True,f"{len(w)} words, variety {100*len(set(c))/len(w):.0f}%"

passed=0; passed_content=0; total=0; errors=[]; empty_content=0
for name,question,check in TESTS:
    total+=1
    try:
        c,reasoning=ask(question,400)
        ok_i = bool(c.strip()) and check(c)      # what a plain client sees
        ok   = ok_i or check(c+" "+reasoning)           # what the model actually knows
        if not c.strip(): empty_content+=1
    except Exception as e:
        ok=ok_i=False; c=f"ERROR: {e}"; reasoning=""
    if ok: passed+=1
    if ok_i: passed_content+=1
    if not ok: errors.append((name,(c or reasoning)[:110].replace("\n"," ")))
    where = "" if ok_i else ("  <- answer only in reasoning" if ok else "")
    print(f"  {'PASS' if ok else 'FAIL'}  {name:12s} "
          f"{(c or '(empty)')[:52].strip().replace(chr(10),' ')}{where}")

print("  --- long text (repetition-lock check) ---")
long_txt,long_rz=ask("Explain in one paragraph how a hash table works and why lookups are fast.",600)
if not long_txt.strip(): long_txt=long_rz; empty_content+=1
ok,why=looks_coherent(long_txt)
total+=1; passed+= 1 if ok else 0
if not ok: errors.append(("long-text",why+" || "+long_txt[:110].replace("\n"," ")))
print(f"  {'PASS' if ok else 'FAIL'}  long-text    {why}")
print(f"  first 150 chars: {long_txt[:150].strip()}")

print(f"\n  RESULT (model knowledge, both fields): {passed}/{total}")
print(f"  RESULT (content only, what a client sees): {passed_content}/{total-1}")
print(f"  requests with EMPTY content: {empty_content}")
if empty_content:
    print("  NOTE: an empty content field means the model put its whole answer in")
    print("        the reasoning field. The model knows the answer; a client that")
    print("        reads only content sees nothing. The fix is a mandatory system")
    print("        line telling the model to always write a final answer outside")
    print("        its reasoning -- see the configuration docs.")
if errors:
    print("  FAILURES:")
    for a,d in errors: print(f"    {a}: {d}")
sys.exit(0 if passed==total else 1)
