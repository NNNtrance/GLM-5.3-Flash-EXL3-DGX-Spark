#!/usr/bin/env python3
"""HAREM-TP3 backport of two upstream vLLM structured-output bugfixes.

Backports, against our pinned vLLM (``0.1.dev20051+g487ecf187``, image
``exl3-zeus:754421f``), which predates both merges:

* vllm-project/vllm#52805 -- "[Bugfix][Structured Output] Stop XGrammar token
  batches at termination"  (merge commit 12f64b39d29282437e35be9aa5db432fb2a1a6e6,
  merged 2026-08-18).  Touches ``vllm/v1/structured_output/backend_xgrammar.py``.
* vllm-project/vllm#53046 -- "[Bugfix][Structured Output] Avoid spurious FSM
  errors after speculative reasoning end"  (merge commit
  c6e19b3be24338759a443e03c8325d76da9ee202, merged 2026-08-21).  Touches
  ``vllm/v1/structured_output/__init__.py``.

PROVENANCE / LICENCE
--------------------
Both hunks were taken from the **upstream vllm-project/vllm** pull-request
diffs (Apache-2.0), via ``gh api repos/vllm-project/vllm/pulls/<N>/files``.
No third-party (AGPL) fork was used as a source.  The upstream tests
(``tests/v1/spec_decode/test_mtp_structured_output.py``) are NOT backported --
they need the upstream test tree, which our image does not ship.

WHY WE NEED THEM
----------------
Production runs GLM-5.3-Flash with speculative decoding (DFlash2, k=7) and a
reasoning parser (``deepseek_r1``).  With structured output active -- Hindsight
runs ``strict_schema``, and the 11 Sep strict tool-call A/B put grammar on the
tool-call path too -- every strict window logged a burst of

    ERROR ... Failed to advance FSM for request <id> for tokens <t>.
              Please file an issue.

plus ``grammar_matcher`` warnings (measured 11 Sep: 200 ERROR over 26 requests
and 245 warnings in one 120-turn strict window; every turn still completed
correctly).  Two upstream causes, both fixed by the PRs above:

1. #52805.  ``accept_tokens`` advanced the FSM through a *whole* draft batch and
   only then checked termination, so draft tokens that follow the grammar's
   terminal token were fed to the matcher and rejected.  A terminated grammar
   also answered ``accept_tokens`` with ``False``, which the caller reads as a
   failure rather than "nothing left to do".  ``reset()`` never cleared
   ``_is_terminated``, so a reused grammar object stayed terminated forever.
2. #53046.  When reasoning ends in the middle of a speculative window, the
   drafts after the ``</think>`` marker predate the bitmask and are not
   guaranteed grammar-valid.  Upstream *advanced* the matcher with them and
   swallowed the rejection -- but ``accept_token`` already logged the ERROR by
   then.  Validating first (``validate_tokens``, which rolls back) and only
   advancing on success keeps the state machine correct and silent.

Both are pure log/state hygiene: they do not change which tokens are emitted on
a healthy path.  Neither touches a CUDA kernel, so speed and KV are expected
unchanged.

ENV GATE -- DEFAULT ON
----------------------
  HAREM_XGRAMMAR_BACKPORT unset or 1 -> backported behaviour (recommended).
  HAREM_XGRAMMAR_BACKPORT=0          -> upstream-487ecf187 behaviour, exactly.

Default is ON, like patch-glm47-failclosed-tp3.py and for the same reason: the
pre-backport default is the bug, so an unset knob must be the safe one.  The
gate means a rollback needs no re-patch and no image change -- drop
``HAREM_XGRAMMAR_BACKPORT=0`` into EXTRA_ENV.  (Note the sidecar identity is
keyed on the patch-*.py set and the prelude text, not on EXTRA_ENV, so flipping
the gate does NOT invalidate the fastload sidecar.)

LOGGING
-------
One line per patched module at import::

    [HAREM-XGRAMMAR-BACKPORT] backend_xgrammar enabled=True (HAREM_XGRAMMAR_BACKPORT='')
    [HAREM-XGRAMMAR-BACKPORT] structured_output enabled=True (HAREM_XGRAMMAR_BACKPORT='')

A boot log without both lines is a boot where this patch did not run.

HOW TO INVOKE IT
----------------
    run python3 "$TP3_DIR/patch-xgrammar-backports-tp3.py" \\
        --root "$(dirname "$VLLM_PY")" --in-place

``--root`` is the DIST-PACKAGES root (the ``REL`` paths below start with
``vllm/``), same convention as patch-flashkda-tp3.py and
patch-glm47-failclosed-tp3.py.  Without ``--in-place`` this is a DRY RUN: it
validates every anchor, prints "dry run OK", exits 0 and writes nothing.

ORDER IN THE PRELUDE
--------------------
After patch-glm47-failclosed-tp3.py.  No ordering constraint in fact -- no
other HAREM patch touches ``vllm/v1/structured_output/`` -- but that is where
the upstream reporter put it and where ours lives, so the two recipes match.

FAIL-CLOSED PATCHING
--------------------
Six anchors across two files, each required exactly once in the installed file.
A missing or duplicated anchor exits non-zero instead of guessing.  Each file is
marked independently, so re-running is a no-op per file.
"""

import argparse
import os
import sys

MARK = "HAREM-XGRAMMAR-BACKPORT"

REL_BACKEND = "vllm/v1/structured_output/backend_xgrammar.py"
REL_INIT = "vllm/v1/structured_output/__init__.py"

_GATE_SRC = '''

# {mark}.  Backport of vllm-project/vllm#52805 + #53046 into 487ecf187.
# "0" restores the pre-backport upstream behaviour exactly; anything else (or
# unset) takes the fix, because the pre-backport default is the bug.
_HAREM_XG_BACKPORT = os.environ.get("HAREM_XGRAMMAR_BACKPORT", "1") != "0"
logger.info(
    "[{mark}] {module} enabled=%s (HAREM_XGRAMMAR_BACKPORT=%r)",
    _HAREM_XG_BACKPORT,
    os.environ.get("HAREM_XGRAMMAR_BACKPORT", ""),
)
'''

# --- backend_xgrammar.py -----------------------------------------------------

B1_OLD = "import json\nfrom dataclasses import dataclass, field\n"
B1_NEW = "import json\nimport os\nfrom dataclasses import dataclass, field\n"

B2_OLD = "logger = init_logger(__name__)\n"
B2_NEW = "logger = init_logger(__name__)\n" + _GATE_SRC.format(
    mark=MARK, module="backend_xgrammar"
)

# vllm-project/vllm#52805, hunk 1.
B3_OLD = '''        """Accepts a list of tokens and advances the FSM.

        Returns True if the FSM was advanced successfully.
        Returns False if the FSM failed to advance.
        """
        if self._is_terminated:
            return False
        for token in tokens:
            if not self.matcher.accept_token(token):
                logger.error(
                    "Failed to advance FSM for request %s "
                    "for tokens %s. Please file an issue.",
                    request_id,
                    token,
                )
                return False
            self.num_processed_tokens += 1
        self._is_terminated = self.matcher.is_terminated()
        return True
'''
B3_NEW = '''        """Accepts a list of tokens and advances the FSM.

        Returns True if all grammar-constrained tokens were accepted.
        Tokens after termination are ignored.  Returns False if the FSM
        failed to advance.  (HAREM-XGRAMMAR-BACKPORT, vllm#52805.)
        """
        if self._is_terminated:
            # vllm#52805: a terminated grammar has nothing left to accept;
            # reporting that as a failure is what produced the spurious
            # "Failed to advance FSM" bursts under speculative decoding.
            return _HAREM_XG_BACKPORT
        for token in tokens:
            if not self.matcher.accept_token(token):
                logger.error(
                    "Failed to advance FSM for request %s "
                    "for tokens %s. Please file an issue.",
                    request_id,
                    token,
                )
                return False
            self.num_processed_tokens += 1
            if _HAREM_XG_BACKPORT:
                # vllm#52805: stop the batch at termination instead of
                # feeding the matcher the drafts that follow it.
                self._is_terminated = self.matcher.is_terminated()
                if self._is_terminated:
                    break
        if not _HAREM_XG_BACKPORT:
            self._is_terminated = self.matcher.is_terminated()
        return True
'''

# vllm-project/vllm#52805, hunk 2.
B4_OLD = '''        Returns the prefix list of tokens that are accepted by the FSM.
        """
        accepted_tokens = []
        for token in tokens:
            if self.matcher.accept_token(token):
                accepted_tokens.append(token)
            else:
                break
'''
B4_NEW = '''        Returns the prefix list of tokens that are accepted by the FSM.
        """
        if _HAREM_XG_BACKPORT and self._is_terminated:
            # vllm#52805: nothing after termination is grammar-constrained.
            return []

        accepted_tokens = []
        for token in tokens:
            if self.matcher.accept_token(token):
                accepted_tokens.append(token)
                if _HAREM_XG_BACKPORT and self.matcher.is_terminated():
                    # vllm#52805: stop probing past the terminal token.
                    break
            else:
                break
'''

# vllm-project/vllm#52805, hunk 3.
B5_OLD = '''    def reset(self):
        self.num_processed_tokens = 0
        self.matcher.reset()
'''
B5_NEW = '''    def reset(self):
        self.matcher.reset()
        self.num_processed_tokens = 0
        if _HAREM_XG_BACKPORT:
            # vllm#52805: without this a reused grammar stays terminated.
            self._is_terminated = False
'''

# --- structured_output/__init__.py -------------------------------------------

C1_OLD = "import itertools\nimport multiprocessing\n"
C1_NEW = "import itertools\nimport multiprocessing\nimport os\n"

C2_OLD = "logger = init_logger(__name__)\n"
C2_NEW = "logger = init_logger(__name__)\n" + _GATE_SRC.format(
    mark=MARK, module="structured_output"
)

# vllm-project/vllm#53046.
C3_OLD = """                    if advance_grammar and not grammar.is_terminated():
                        accepted = grammar.accept_tokens(req_id, [token])
                        if accepted:
"""
C3_NEW = """                    if advance_grammar and not grammar.is_terminated():
                        if _HAREM_XG_BACKPORT and post_reasoning_end_in_window:
                            # vllm#53046: drafts after the reasoning-end
                            # marker predate the bitmask, so probe with a
                            # rolled-back validate first.  Advancing and
                            # swallowing the rejection still logged the
                            # "Failed to advance FSM" ERROR.
                            accepted = bool(grammar.validate_tokens([token]))
                            if accepted:
                                accepted = grammar.accept_tokens(req_id, [token])
                        else:
                            accepted = grammar.accept_tokens(req_id, [token])
                        if accepted:
"""

FILES = [
    (
        REL_BACKEND,
        [
            ("B1-stdlib-imports", B1_OLD, B1_NEW),
            ("B2-gate", B2_OLD, B2_NEW),
            ("B3-accept-tokens", B3_OLD, B3_NEW),
            ("B4-validate-tokens", B4_OLD, B4_NEW),
            ("B5-reset", B5_OLD, B5_NEW),
        ],
    ),
    (
        REL_INIT,
        [
            ("C1-stdlib-imports", C1_OLD, C1_NEW),
            ("C2-gate", C2_OLD, C2_NEW),
            ("C3-grammar-bitmask", C3_OLD, C3_NEW),
        ],
    ),
]

NAME = "patch-xgrammar-backports"


def apply(src: str, where: str, anchors) -> str:
    if MARK in src:
        print(f"{NAME}: already applied ({where})")
        return src
    for name, old, _new in anchors:
        n = src.count(old)
        if n != 1:
            print(
                f"{NAME}: {name} count={n} (expected 1) in {where} -- refusing",
                file=sys.stderr,
            )
            sys.exit(3)
    for _name, old, new in anchors:
        src = src.replace(old, new, 1)
    return src


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dist-packages root")
    ap.add_argument("--in-place", action="store_true")
    ap.add_argument("--out-dir", default="", help="write here instead of in place")
    a = ap.parse_args()

    wrote = 0
    for rel, anchors in FILES:
        p = os.path.join(a.root, rel)
        src = open(p).read()
        out = apply(src, rel, anchors)
        if out is src:
            continue
        if a.out_dir:
            dst = os.path.join(a.out_dir, os.path.basename(rel))
        elif a.in_place:
            dst = p
        else:
            print(f"{NAME}: {rel} anchors OK")
            continue
        open(dst, "w").write(out)
        print(f"{NAME}: applied to {dst} (vllm#52805/#53046, gate default ON)")
        wrote += 1

    if not (a.in_place or a.out_dir):
        print(f"{NAME}: dry run OK (pass --in-place or --out-dir to write)")


if __name__ == "__main__":
    main()
