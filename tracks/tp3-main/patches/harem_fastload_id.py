#!/usr/bin/env python3
"""HAREM-TP3 fastload: the identity of a pre-sliced per-rank weight sidecar.

Kept free of torch/vllm imports on purpose: both the in-engine module
(``harem_fastload.py``) and the cheap prelude preflight (``preflight-fastload.py``)
compute the identity with this same code, so "the sidecar is stale" is one
definition, not two that can drift apart.

Everything in here is either cheap file metadata or the hash of a small file.
The 163 GiB of weights are pinned by hashing the checkpoint's own SHA256SUMS
manifest plus the (name, size) list of the shards -- never by reading them.
"""

import glob
import hashlib
import json
import os

SCHEMA = 2
# Bump ONLY when the on-disk sidecar layout or the dump semantics change.
# (The dumper's own file hash is deliberately NOT part of the identity: it is the
# tool, not the recipe, and hashing it made every edit to the tool invalidate a
# perfectly good sidecar -- and, worse, invalidate it *mid-boot*, because
# /opt/harem-tp3 is a live read-only mount of the working directory.)
SIDECAR_FORMAT = 1


def sha_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _hf_revision(model_dir: str) -> str:
    """The HF commit the checkpoint was materialised from, if it left a trace."""
    meta = sorted(glob.glob(os.path.join(model_dir, ".cache/huggingface/download/*.metadata")))
    for m in meta:
        try:
            with open(m) as f:
                first = f.readline().strip()
            if len(first) == 40 and all(c in "0123456789abcdef" for c in first):
                return first
        except OSError:
            continue
    return ""


def checkpoint_identity(model_dir: str) -> dict:
    d: dict = {"path": model_dir, "realpath": os.path.realpath(model_dir)}
    if not os.path.isdir(model_dir):
        d["missing"] = True
        return d
    for f in ("config.json", "model.safetensors.index.json", "quantization_config.json",
              "SHA256SUMS", "MANIFEST.json", "exl3-mcg-storage-abi.json"):
        p = os.path.join(model_dir, f)
        if os.path.isfile(p):
            d[f] = sha_file(p)
    d["hf_revision"] = _hf_revision(os.path.realpath(model_dir))
    shards = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    lines = []
    total = 0
    for s in shards:
        try:
            n = os.path.getsize(s)
        except OSError:
            n = -1
        total += max(n, 0)
        lines.append(f"{os.path.basename(s)}:{n}")
    d["n_weight_files"] = len(shards)
    d["weight_bytes"] = total
    d["weight_list_sha256"] = _sha_text("\n".join(lines))
    return d


def file_identity(model_dir: str, draft_dir: str, tp3_dir: str = "/opt/harem-tp3") -> dict:
    """Everything that decides what a rank's loaded weights look like, minus the
    things only the engine knows (vllm/cuda-exl3 versions, hf overrides). Those
    are added by harem_fastload.engine_identity()."""
    ident = {
        "schema": SCHEMA,
        "sidecar_format": SIDECAR_FORMAT,
        "image_tag": os.environ.get("HAREM_IMAGE_TAG", ""),
        "tp_size": os.environ.get("TP_SIZE", ""),
        "enable_ep": os.environ.get("ENABLE_EP", ""),
        "node_rank": os.environ.get("NODE_RANK", ""),
        "ep_filter_suffixes": os.environ.get("HAREM_EP_FILTER_SUFFIXES", ".trellis"),
        "checkpoint": {"model": checkpoint_identity(model_dir)},
        "patches": {},
    }
    if draft_dir:
        ident["checkpoint"]["draft"] = checkpoint_identity(draft_dir)
    # Only the things that decide what the loaded weights ARE. The fastload
    # modules themselves are excluded on purpose (see SIDECAR_FORMAT above).
    # The upstream-port patch set is yama-*.py + its engine and
    # prelude; the TP3 config hook and the EP/mixed loaders now live in the
    # cuda_exl3 plugin, so its Python sources are part of the identity too.
    for p in sorted(glob.glob(os.path.join(tp3_dir, "patch-*.py"))
                    + glob.glob(os.path.join(tp3_dir, "yama-*.py"))) + [
        os.path.join(tp3_dir, "tp3-prelude.sh"),
        os.path.join(tp3_dir, "harem-prelude.sh"),
        os.path.join(tp3_dir, "harem_yama.py"),
        os.path.join(tp3_dir, "overlay/cuda_exl3/_harem_ep.py"),
    ]:
        if os.path.isfile(p):
            ident["patches"][os.path.relpath(p, tp3_dir)] = sha_file(p)
    import importlib.util

    spec = importlib.util.find_spec("cuda_exl3")
    if spec is not None and spec.submodule_search_locations:
        pkg = list(spec.submodule_search_locations)[0]
        ident["plugin"] = {f: sha_file(os.path.join(pkg, f))
                           for f in sorted(os.listdir(pkg)) if f.endswith(".py")}
    return ident


def diff(expected: dict, found: dict, path: str = "") -> list[str]:
    """Human-readable list of identity mismatches (empty == identical)."""
    out: list[str] = []
    keys = sorted(set(expected) | set(found))
    for k in keys:
        pk = f"{path}.{k}" if path else k
        a, b = expected.get(k, "<yok>"), found.get(k, "<yok>")
        if isinstance(a, dict) and isinstance(b, dict):
            out += diff(a, b, pk)
        elif a != b:
            out.append(f"{pk}: kayitli={a!r} simdiki={b!r}")
    return out


if __name__ == "__main__":  # pragma: no cover - debugging aid
    import sys

    m = sys.argv[1] if len(sys.argv) > 1 else "/var/tmp/glm-5.3-flash-tr3-4bpw-tp3"
    dr = sys.argv[2] if len(sys.argv) > 2 else "/var/tmp/dflash2-draft-tp3"
    print(json.dumps(file_identity(m, dr), indent=2, sort_keys=True))
