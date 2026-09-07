# tracks — one repository, two node counts, kept apart

This repository serves the same model on **two** working arrangements: three nodes at TP=3 with
expert parallelism, and two nodes at TP=2. Most of what is here belongs to both. A small number of
files belong to exactly one, and mixing those two sets is how a reader ends up running a three-node
patch tree at two ranks.

**This directory is the small number.** One folder per track, holding only the files that differ:
the environment templates, the patch tree, the autostart unit and its preflight, and a results
summary. Everything else stays where it is, shared, because it genuinely is shared.

If you do not know which track is yours, [docs/00-start-here.md](../docs/00-start-here.md) asks one
question and answers it.

| | |
|---|---|
| [**tp3/**](tp3/) | Three nodes, TP=3 + expert parallelism. The production recipe |
| [**tp2/**](tp2/) | Two nodes, TP=2, expert parallelism off |

---

## The layout

```text
tracks/
  tp3/
    README.md                    what this track is, its settings and its numbers
    env.tp3.example              the routed-experts-only template
    env.tp3-full.example         the full-scope template -- this is production
    harem-exl3.service           the autostart unit
    motor-onkosul-exl3.sh        its preflight
    patches/                     the in-container patch tree, 23 files
    patches-optional/sm12/       measured, free, NOT in production -- the one sm_12x fix
                                 that did not ride with production 11
    patches/indexer-workspace/
                                 +10.25 % KV pool -- the sparse-indexer K-gather
                                 workspace bound, IN production since configuration 12
  tp2/
    README.md                    what this track is, its settings and its numbers
    env.tp2-full.example         the two-node production candidate
    harem-exl3-tp2.service       the autostart unit
    motor-onkosul-exl3-tp2.sh    its preflight
    patches/                     the in-container patch tree, 18 files
    patches/vision/              a POINTER, not a copy: the tower at two ranks
    patches/prefix-hit-and-kpool-tail/
                                 a POINTER, not a copy: the two backports at two ranks
```

**Four files are shared between the trees by copying, and the two `patches/` subdirectories above
say so rather than duplicating them.** `patch-vllm-tp3.py`, `patch-vision-tp3.py`,
`patch-prefixhit-tp3.py` and `patch-kpooltail-tp3.py` are used at two ranks byte for byte — none of
them reads the rank count — and the only rank-dependent thing in any of them is a *number you
measure*, the prefix-cache block granularity (3,328 at three ranks, **4,608** at two:
[docs/15](../docs/15-tp2-track.md) §5.10).

Nothing was rewritten in the move. `tracks/tp3/patches/` is the tree this repository used to publish
as `patches/tp3full/` and `tracks/tp2/patches/` is `patches/tp2full/`, file for file. One file did
disappear, and it was a duplicate: `patches/tp2/patch-fullscope-tp2.py` was **byte-identical** to the
copy inside the two-node tree, and this repository's own rule is that two copies of a file are a coin
flip unless something checks ([docs/08](../docs/08-fast-boot.md) §12). The surviving copy is
[`tracks/tp2/patches/patch-fullscope-tp2.py`](tp2/patches/patch-fullscope-tp2.py) and every reference
now points at it.

## The one thing to read before you copy anything

**The directory name in this repository is not the directory name on your nodes.** On a node the
three-node tree lives at `~/exl3-zeus/tp3full/` and that name is load-bearing in three places: the
launcher mounts `$TP3_DIR/tp3-prelude.sh` at `/start.sh`, `tp3-prelude.sh` inside the directory is a
**hard link** to `tp3full-prelude.sh` rather than a copy, and the directory's file list and the full
text of the prelude are hashed into the fast-load sidecar's identity
([docs/08](../docs/08-fast-boot.md) §4).

**The launcher directory and the patch-tree directory are two things.** The examples on this page use
`tp3full/` for both, for brevity; the unit's `WorkingDirectory`/`ExecStart` point at wherever
`start-tp3.sh` lives, and `TP3_DIR` and `OVERLAY_DIR` point at wherever the patches, the prelude and
the overlay live. On our nodes those are separate directories and the second is named after the
configuration, which is why [`tp3/env.tp3-full.example`](tp3/env.tp3-full.example) shows a
configuration-named path. Any name works: set `TP3_DIR` to wherever you put the patch tree.

So the copy is still:

```
cp -r tracks/tp3/patches/. ~/exl3-zeus/tp3full/
```

and the hard link is still:

```
ln ~/exl3-zeus/tp3full/tp3full-prelude.sh ~/exl3-zeus/tp3full/tp3-prelude.sh
```

Renaming the tree **on the node** invalidates every sidecar on every node and costs a dump boot each.
Renaming it **in this repository**, which is what happened here, costs nothing: the manifest hashes
the directory you serve from, not the one you cloned from.

One more consequence, and it has bitten us twice: **adding a file to the node-side tree — even a file
that is never called — refuses the next boot on every node.** Once it was the TP=2 patch dropped into
the TP=3 tree that did it ([docs/13](../docs/13-full-scope-checkpoint.md) §6.4). Separate directories
are the answer, and they are the reason this one exists.

## What is shared, and stays outside this directory

Nothing below is duplicated per track. Where a shared file needs a per-track edit, the track's page
says which lines.

| Shared | Why it is shared |
|---|---|
| [`docs/`](../docs/) | Nineteen pages, each carrying an **Applies to** badge on its first line. [docs/00-start-here.md](../docs/00-start-here.md) §5 is the index |
| [`scripts/`](../scripts/) | **Both launchers and both preludes** — `start-tp3.sh` + `tp3-prelude.sh` and `start-tp2full.sh` + `tp2-prelude.sh`. They are per track but they live together, because the benchmark harness, the probes and the prompt sets beside them are identical at either rank count and a launcher is read next to them |
| [`patches/tp3/`](../patches/tp3/) | **Both tracks draw from it.** It is the routed-experts-only tree — production configurations 1 to 8 and the rollback at three ranks — and it is also where TP=2 gets `patch-swblock-tp3.py`, `patch-kvdiag-tp3.py`, `patch-draftkv-tp3.py`, `patch-epfilter-tp3.py` and `patch-fastload-tp3.py`, all of which are `tp`-agnostic and gated on their own environment knobs |
| [`patches/kernel/`](../patches/kernel/) | The NCCL mesh plugin patches. `0005` is a no-op with one cable per pair, and that is a cabling property rather than a rank-count one |
| [`patches/dflash2-port/`](../patches/dflash2-port/) · [`patches/indexer-overlay/`](../patches/indexer-overlay/) | The drafter port and the GB10 top-k overlay. Both are properties of the model and the part |
| [`bench/`](../bench/) | Model-free, engine down. Most of it runs on one node |
| [`results/`](../results/README.md) · [`charts/`](../charts/) | Each file's header names its arm and its rank count |
| [`audit/`](../audit/README.md) | The post-install self-check. Its bands are the three-node ones and it says so |
| [`envs/README.md`](../envs/README.md) | The rule that governs both tracks: derive each node's file with `sed`, on that node, and **never copy a finished env file between nodes** |
| [`systemd/README.md`](../systemd/README.md) | The autostart hazard, the install commands and the three whole-cluster reboot tests we have run. The units themselves are per track and live here |

## Which track a file belongs to, when it is not obvious

Three questions, in order:

1. **Does it name a rank count, a pad, or expert parallelism as mandatory?** Then it is TP=3's. The
   whole of [docs/03](../docs/03-tp3-padding-and-sidecars.md) and
   [docs/13](../docs/13-full-scope-checkpoint.md) §7 are in this class.
2. **Does it name a fabric, a kernel, a checkpoint or a measurement rule?** Then it is shared. None
   of those depend on how many ranks are reading them.
3. **Is it a number?** Then it belongs to the arm it was measured on, and that arm's node count is in
   the file's own header. A three-node figure is not a two-node figure with a discount applied, and
   [docs/15](../docs/15-tp2-track.md) §4 is the measured comparison rather than a projection.
