#!/usr/bin/env python3
"""Behaviour test for patch-xgrammar-backports-tp3.py (vllm#52805 + #53046).

RUNS ONLY INSIDE THE IMAGE, against an ALREADY-PATCHED tree:

    sudo docker run --rm -v $PWD/test_xgrammar_backports.py:/tmp/t.py:ro \\
      --entrypoint bash exl3-zeus:754421f -c \\
      'python3 /tmp/p.py --root /usr/local/lib/python3.12/dist-packages --in-place \\
       && python3 /tmp/t.py'

It drives XgrammarGrammar with a fake xgrammar matcher, so no GPU, no model and
no real grammar compilation.  The #52805 hunks are exercised behaviourally in
both gate positions.  The #53046 hunk lives inside
``StructuredOutputManager.grammar_bitmask``, which needs a scheduler, a
VllmConfig and spec-decode state to call; it is checked STRUCTURALLY here
(validate-before-accept under ``post_reasoning_end_in_window``) and
behaviourally in production by the FSM-ERROR count in a strict window.

Exit 0 = every check passed.  Any failure prints FAIL and exits 1.
"""

import importlib
import os
import sys

R = "/usr/local/lib/python3.12/dist-packages"

_fail = 0
_pass = 0


def check(cond, what):
    global _fail, _pass
    if cond:
        _pass += 1
        print(f"  ok   {what}")
    else:
        _fail += 1
        print(f"  FAIL {what}")


class FakeMatcher:
    """Minimal xgr.GrammarMatcher stand-in.

    Accepts tokens while they are in ``good``; terminates once ``term_after``
    tokens have been accepted.  Counts every accept_token call so a test can
    prove the patch stopped feeding it.
    """

    def __init__(self, good=(1, 2, 3, 4, 5, 6, 7, 8), term_after=3):
        self.good = set(good)
        self.term_after = term_after
        self.n = 0
        self.calls = 0
        self.rollbacks = []
        self.resets = 0

    def accept_token(self, token):
        self.calls += 1
        if token not in self.good:
            return False
        self.n += 1
        return True

    def is_terminated(self):
        return self.n >= self.term_after

    def rollback(self, k):
        self.rollbacks.append(k)
        self.n -= k

    def reset(self):
        self.resets += 1
        self.n = 0

    def fill_next_token_bitmask(self, bitmask, idx):
        pass


def load(gate):
    """(Re)import the patched module with HAREM_XGRAMMAR_BACKPORT=gate."""
    if gate is None:
        os.environ.pop("HAREM_XGRAMMAR_BACKPORT", None)
    else:
        os.environ["HAREM_XGRAMMAR_BACKPORT"] = gate
    import vllm.v1.structured_output.backend_xgrammar as m

    m = importlib.reload(m)
    return m


def grammar(mod, matcher):
    return mod.XgrammarGrammar(vocab_size=32, matcher=matcher, ctx=None)


def suite_52805_on():
    print("#52805, gate ON (default)")
    mod = load(None)
    check(mod._HAREM_XG_BACKPORT is True, "gate defaults to ON when unset")

    # A batch that crosses termination: 5 drafts, grammar terminates at 3.
    fm = FakeMatcher(term_after=3)
    g = grammar(mod, fm)
    ok = g.accept_tokens("r1", [1, 2, 3, 4, 5])
    check(ok is True, "batch crossing termination returns True")
    check(g.is_terminated() is True, "grammar is terminated after the batch")
    check(fm.calls == 3, f"matcher fed only up to the terminal token (calls={fm.calls})")
    check(
        g.num_processed_tokens == 3,
        f"num_processed_tokens counts accepted only ({g.num_processed_tokens})",
    )

    # Already terminated -> True, and the matcher is not touched at all.
    before = fm.calls
    check(g.accept_tokens("r1", [4]) is True, "terminated grammar accepts as True")
    check(fm.calls == before, "terminated grammar does not call the matcher")

    # validate_tokens on a terminated grammar.
    check(g.validate_tokens([4, 5]) == [], "validate_tokens returns [] once terminated")
    check(fm.calls == before, "validate_tokens does not call the matcher either")

    # validate_tokens stops at the terminal token and rolls back what it probed.
    fm2 = FakeMatcher(term_after=2)
    g2 = grammar(mod, fm2)
    got = g2.validate_tokens([1, 2, 3, 4])
    check(got == [1, 2], f"validate_tokens stops at the terminal token ({got})")
    check(fm2.rollbacks == [2], f"probe is rolled back exactly once ({fm2.rollbacks})")
    check(fm2.n == 0, "matcher state restored after the probe")
    check(g2.is_terminated() is False, "validate_tokens does not advance _is_terminated")

    # A genuinely bad token still fails loudly.
    fm3 = FakeMatcher(good=(1,), term_after=99)
    g3 = grammar(mod, fm3)
    check(g3.accept_tokens("r3", [1, 99]) is False, "a truly invalid token returns False")

    # reset clears termination.
    fm4 = FakeMatcher(term_after=1)
    g4 = grammar(mod, fm4)
    g4.accept_tokens("r4", [1])
    check(g4.is_terminated() is True, "terminated before reset")
    g4.reset()
    check(g4.is_terminated() is False, "reset clears _is_terminated")
    check(g4.num_processed_tokens == 0, "reset clears num_processed_tokens")
    check(fm4.resets == 1, "reset resets the matcher once")

    # rollback still recomputes termination from the matcher.
    fm5 = FakeMatcher(term_after=2)
    g5 = grammar(mod, fm5)
    g5.accept_tokens("r5", [1, 2])
    g5.rollback(1)
    check(g5.is_terminated() is False, "rollback un-terminates when the matcher does")


def suite_52805_off():
    print("#52805, gate OFF (HAREM_XGRAMMAR_BACKPORT=0 -> pre-backport upstream)")
    mod = load("0")
    check(mod._HAREM_XG_BACKPORT is False, "gate reads OFF")

    fm = FakeMatcher(term_after=3)
    g = grammar(mod, fm)
    ok = g.accept_tokens("r1", [1, 2, 3, 4, 5])
    check(ok is True, "pre-backport: whole batch accepted")
    check(fm.calls == 5, f"pre-backport: matcher fed past termination (calls={fm.calls})")
    check(
        g.num_processed_tokens == 5,
        f"pre-backport: counts the whole batch ({g.num_processed_tokens})",
    )
    check(
        g.accept_tokens("r1", [4]) is False,
        "pre-backport: terminated grammar returns False (the bug)",
    )

    fm2 = FakeMatcher(term_after=2)
    g2 = grammar(mod, fm2)
    got = g2.validate_tokens([1, 2, 3, 4])
    check(got == [1, 2, 3, 4], f"pre-backport: validate_tokens probes past ({got})")

    fm3 = FakeMatcher(term_after=1)
    g3 = grammar(mod, fm3)
    g3.accept_tokens("r3", [1])
    g3.reset()
    check(
        g3.is_terminated() is True,
        "pre-backport: reset leaves _is_terminated set (the bug)",
    )


def suite_53046():
    print("#53046, structured_output/__init__.py (structural)")
    src = open(os.path.join(R, "vllm/v1/structured_output/__init__.py")).read()
    check("HAREM-XGRAMMAR-BACKPORT" in src, "file carries the mark")
    check(
        "if _HAREM_XG_BACKPORT and post_reasoning_end_in_window:" in src,
        "post-reasoning-end branch is gated",
    )
    i_branch = src.find("if _HAREM_XG_BACKPORT and post_reasoning_end_in_window:")
    tail = src[i_branch : i_branch + 700]
    i_val = tail.find("grammar.validate_tokens([token])")
    i_acc = tail.find("grammar.accept_tokens(req_id, [token])")
    check(i_val != -1, "validate_tokens is called in the branch")
    check(i_acc != -1 and i_val < i_acc, "validate_tokens runs BEFORE accept_tokens")
    check(
        "else:\n                            accepted = grammar.accept_tokens(req_id, [token])"
        in src,
        "else-arm keeps the pre-backport single accept_tokens call",
    )
    check(
        src.count("accepted = grammar.accept_tokens(req_id, [token])") == 2,
        "exactly two accept_tokens call sites (gated + else)",
    )
    check("import os" in src.split("logger = init_logger")[0], "os imported")

    print("both files marked")
    bsrc = open(os.path.join(R, "vllm/v1/structured_output/backend_xgrammar.py")).read()
    check("HAREM-XGRAMMAR-BACKPORT" in bsrc, "backend_xgrammar carries the mark")
    check(
        bsrc.count("_HAREM_XG_BACKPORT = os.environ.get") == 1,
        "exactly one gate definition in backend_xgrammar",
    )
    check(
        src.count("_HAREM_XG_BACKPORT = os.environ.get") == 1,
        "exactly one gate definition in structured_output",
    )


def main():
    suite_52805_on()
    suite_52805_off()
    suite_53046()
    print(f"\n{_pass} passed, {_fail} failed")
    sys.exit(1 if _fail else 0)


if __name__ == "__main__":
    main()
