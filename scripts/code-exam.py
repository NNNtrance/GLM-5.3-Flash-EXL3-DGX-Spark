#!/usr/bin/env python3
"""Code exam: the model writes a function, we EXECUTE it and check the result.

Far sharper than a knowledge quiz -- wrong code fails the assertions. Twelve
tasks, each with its own assertion suite. A healthy production engine scores
12/12; arms with a mis-shaped decode kernel scored 0/12 to 10/12 here while
still looking fluent, which is exactly why this gate exists.

The generated code is executed with the SAME python interpreter that runs this
script, in a temporary file, with a 25-second timeout and no sandbox. Run it
against your own engine only, and read what it is about to execute if that
matters to you.

Usage: python3 code-exam.py [API_BASE] [OUT_JSON] [REPEATS]
       default API http://192.0.2.10:8001, no JSON output, 1 repeat
Written by us for this recipe; use freely (Apache-2.0)."""
import json,sys,re,subprocess,tempfile,os,urllib.request
BASE=sys.argv[1] if len(sys.argv)>1 else os.environ.get("API","http://192.0.2.10:8001")
OUT=sys.argv[2] if len(sys.argv)>2 else None
REPEATS=int(sys.argv[3]) if len(sys.argv)>3 else 1

P=[
 ("binary_search","Write a Python function `bsearch(a, x)` that returns the index of x in the sorted list a, or -1 if absent. Use binary search. Output only the function code.",
  "assert bsearch([1,3,5,7,9],7)==3; assert bsearch([1,3,5,7,9],4)==-1; assert bsearch([],1)==-1; assert bsearch([2],2)==0"),
 ("roman","Write a Python function `to_roman(n)` converting an integer 1..3999 to a Roman numeral string. Output only the function code.",
  "assert to_roman(1)=='I'; assert to_roman(4)=='IV'; assert to_roman(1994)=='MCMXCIV'; assert to_roman(3999)=='MMMCMXCIX'"),
 ("parens","Write a Python function `balanced(s)` returning True if the brackets in s (round, square, curly) are correctly balanced and nested. Output only the function code.",
  "assert balanced('([]{})'); assert not balanced('([)]'); assert balanced(''); assert not balanced('(')"),
 ("anagram","Write a Python function `group_anagrams(words)` returning a list of lists grouping anagrams together. Output only the function code.",
  "r=group_anagrams(['eat','tea','tan','ate','nat','bat']); s=sorted(sorted(g) for g in r); assert s==[['ate','eat','tea'],['bat'],['nat','tan']], s"),
 ("lru","Write a Python function `first_unique(s)` returning the first non-repeating character of string s, or None. Output only the function code.",
  "assert first_unique('swiss')=='w'; assert first_unique('aabb') is None; assert first_unique('x')=='x'"),
 ("matrix","Write a Python function `spiral(m)` returning the elements of a 2-D list m in clockwise spiral order as a flat list. Output only the function code.",
  "assert spiral([[1,2,3],[4,5,6],[7,8,9]])==[1,2,3,6,9,8,7,4,5]; assert spiral([[1]])==[1]; assert spiral([])==[]"),
 ("intervals","Write a Python function `merge_intervals(iv)` merging overlapping intervals in a list of [start,end] pairs, returning them sorted. Output only the function code.",
  "assert merge_intervals([[1,3],[2,6],[8,10],[15,18]])==[[1,6],[8,10],[15,18]]; assert merge_intervals([])==[]"),
 ("dp","Write a Python function `lis(a)` returning the length of the longest strictly increasing subsequence of list a. Output only the function code.",
  "assert lis([10,9,2,5,3,7,101,18])==4; assert lis([])==0; assert lis([7,7,7])==1"),
 ("graph","Write a Python function `has_cycle(n, edges)` returning True if the undirected graph with n nodes (0..n-1) and the given edge list contains a cycle. Output only the function code.",
  "assert has_cycle(3,[[0,1],[1,2],[2,0]]); assert not has_cycle(3,[[0,1],[1,2]]); assert not has_cycle(1,[])"),
 ("text","Write a Python function `word_freq(text)` returning a dict of lowercase word -> count, where words are maximal runs of letters. Output only the function code.",
  "assert word_freq('The cat, the CAT!')=={'the':2,'cat':2}; assert word_freq('')=={}"),
 ("number","Write a Python function `is_palindrome_num(n)` returning True if the integer n reads the same forwards and backwards. Negative numbers are never palindromes. Output only the function code.",
  "assert is_palindrome_num(121); assert not is_palindrome_num(-121); assert is_palindrome_num(0); assert not is_palindrome_num(10)"),
 ("date","Write a Python function `days_between(a, b)` taking two 'YYYY-MM-DD' strings and returning the absolute number of days between them. Output only the function code.",
  "assert days_between('2020-01-01','2020-12-31')==365; assert days_between('2021-03-01','2021-03-01')==0; assert days_between('2020-03-01','2020-02-28')==2"),
]

def ask(p):
    body=json.dumps({"model":"glm-5.3-flash","messages":[{"role":"user","content":p}],
                     "max_tokens":1600,"temperature":0.0,
                     "chat_template_kwargs":{"enable_thinking":True,"reasoning_effort":"low"},}).encode()
    r=urllib.request.Request(BASE+"/v1/chat/completions",body,{"Content-Type":"application/json"})
    with urllib.request.urlopen(r,timeout=600) as h:
        m=json.load(h)["choices"][0]["message"]
    # On short answers the model sometimes puts the whole output in
    # 'reasoning' and leaves 'content' empty (with thinking ON, intermittently
    # even at temperature 0). Empty content would score zero here even though
    # the model did write the code, so we fall back to the reasoning field.
    c=(m.get("content") or "").strip()
    return c if c else (m.get("reasoning_content") or m.get("reasoning") or "")

def extract_code(t):
    b=re.findall(r"```(?:python)?\s*\n(.*?)```", t, re.S)
    return b[0] if b else t

def try_code(code, test):
    with tempfile.NamedTemporaryFile("w",suffix=".py",delete=False) as f:
        f.write(code+"\n\n"+test+"\nprint('DONE')\n"); path=f.name
    try:
        r=subprocess.run([sys.executable,path],capture_output=True,text=True,timeout=25)
        return ("DONE" in r.stdout), (r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "")
    except subprocess.TimeoutExpired:
        return False,"timeout"
    finally:
        os.unlink(path)

results={}; passed=0; total=0
for name,prompt,test in P:
    wins=0
    for t in range(REPEATS):
        try:
            c=ask(prompt); k=extract_code(c)
            ok,error=try_code(k,test)
        except Exception as e:
            ok,error=False,f"request error: {str(e)[:70]}"
        if ok: wins+=1
        else: last_error=error
    total+=REPEATS; passed+=wins
    results[name]=wins
    verdict="PASS" if wins==REPEATS else ("PARTIAL" if wins else "FAIL")
    extra="" if wins==REPEATS else f"   ({last_error[:70]})"
    print(f"  {verdict:5s} {name:14s} {wins}/{REPEATS}{extra}", flush=True)

print(f"\n  CODE EXAM: {passed}/{total}  ({100*passed/total:.1f}%)")
if OUT: json.dump({"passed":passed,"total":total,"detail":results},open(OUT,"w"),indent=1)
