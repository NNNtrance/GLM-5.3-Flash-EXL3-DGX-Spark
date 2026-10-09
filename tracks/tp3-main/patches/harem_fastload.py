#!/usr/bin/env python3
"""HAREM-TP3 fastload: dump / restore the weights a rank actually holds.

Why
---
At TP=3 + EP every rank opens the *whole* 163.58 GiB EXL3 checkpoint and keeps
54.86 GiB of it, because the slicing (heads 64->66, shared expert 2048->2112,
vocab padding, EP expert map 96/288, per-projection ``narrow``) happens while
the tensors stream past.  That is ~150 000 small, strided host->device copies.

This module records the *result* of that work once -- the exact tensors the rank
holds after ``load_weights`` -- and on later boots copies them straight back.
It deliberately does NOT re-derive the slicing: the sidecar is produced by a
normal full-checkpoint load, so "what the loader would have produced" and "what
is on disk" cannot drift by construction.  The MANIFEST records a sha256 per
tensor, so the claim is checkable rather than believed.

Modes (env HAREM_FASTLOAD_MODE)
    unset / ""  upstream behaviour, this module is inert
    dump        load normally from the checkpoint, then write the sidecar
    load        restore from the sidecar; refuse if identity or names disagree

Other env
    HAREM_FASTLOAD_DIR         base path; the rank dir is "<base>-r<tp_rank>"
    HAREM_FASTLOAD_SHARD_BYTES sidecar shard size (default 2 GiB)
    HAREM_FASTLOAD_READ        "buffered" (default, one f.read per shard) or
                               "mmap" (safe_open); an A/B that needs no re-dump
    HAREM_FASTLOAD_VERIFY      tensors re-hashed after a restore.
                               int (default 64), "all", or 0 to skip
    HAREM_DROP_CKPT_CACHE      "1" (default) fadvise(DONTNEED) the checkpoint
                               shards after loading; on unified memory their
                               page cache is charged against the KV pool
    HAREM_MALLOC_TRIM          "1" (default) returns the loader's freed host
                               arenas to the OS -- on unified memory they would
                               otherwise be charged to "non-torch" and shrink KV
    HAREM_FASTLOAD_POSTHASH_N  how many tensors the post-process hash covers:
                               "all" (default) or an int (stratified sample)
    HAREM_FASTLOAD_POSTHASH    label; if set, hash every parameter/buffer AFTER
                               process_weights_after_loading into
                               /cache/harem-poststate-<tag>-r<rank>-<label>.json

Written-parts ledger
    cuda_exl3's mixed fused linear (EXL3 + bf16 shards in one layer) records, in
    ``layer.exl3_mixed_loaded``, every (part, shard) its weight loaders wrote, and
    refuses in ``process_weights_after_loading`` a layer with an unwritten part.
    A restore bypasses those loaders, so the dump stores each such module's
    ledger in the manifest (``models[tag]["yazildi_defteri"]``, version 1) and the
    restore marks ONLY the recorded parts. A sidecar without the ledger (written before
    the ledger existed) is refused -- re-dump; there is no silent fall back to a
    normal load. The plugin's own check is left untouched and still decides.
    Dump and restore both run inside load_weights_hook, i.e. BEFORE
    process_weights_after_loading (base_loader F1 vs F2), so the restored
    tensors are the raw loaded ones and post-processing runs exactly once.

Everything fails closed: any missing name, extra name, shape/dtype/size
mismatch, identity mismatch or hash mismatch raises instead of serving a model
whose weights nobody checked.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import time

import torch

from vllm.logger import init_logger

try:
    from vllm.model_executor.model_loader import harem_fastload_id as hid
except ImportError:  # installed side by side rather than into the package
    import harem_fastload_id as hid  # type: ignore

logger = init_logger(__name__)

MANIFEST = "MANIFEST.json"
_GiB = 1 << 30


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _mode() -> str:
    return os.environ.get("HAREM_FASTLOAD_MODE", "").strip().lower()


def _shard_bytes() -> int:
    return int(os.environ.get("HAREM_FASTLOAD_SHARD_BYTES", str(2 * _GiB)))


def _rank() -> int:
    from vllm.distributed import get_tensor_model_parallel_rank

    r = int(get_tensor_model_parallel_rank())
    env = os.environ.get("NODE_RANK", "")
    if env not in ("", None) and int(env) != r:
        raise RuntimeError(
            f"harem-fastload: NODE_RANK={env} but the tensor-parallel rank is {r}. "
            "The sidecar is per tensor-parallel rank; refusing rather than "
            "restoring another rank's shard."
        )
    return r


def _rank_dir() -> str:
    base = os.environ.get("HAREM_FASTLOAD_DIR", "").strip()
    if not base:
        raise RuntimeError(
            "harem-fastload: HAREM_FASTLOAD_MODE is set but HAREM_FASTLOAD_DIR is empty"
        )
    return f"{base}-r{_rank()}"


def _tag(model_config) -> str:
    return os.path.basename(str(model_config.model).rstrip("/")) or "model"


def _nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


DEFTER_ATTR = "exl3_mixed_loaded"      # cuda_exl3 linear.py Exl3MixedLinearMethod
DEFTER_ALAN = "yazildi_defteri"        # manifest models[tag] field
DEFTER_SURUM = 1


def _defter_topla(model) -> dict[str, list[list]]:
    """Every module that carries the plugin's written-parts ledger, as
    {module name: sorted [[part, shard], ...]} (JSON-safe, order-stable)."""
    out: dict[str, list[list]] = {}
    for name, mod in model.named_modules():
        led = getattr(mod, DEFTER_ATTR, None)
        if led is None:
            continue
        if not isinstance(led, set):
            raise RuntimeError(
                f"harem-fastload: {name}.{DEFTER_ATTR} is a {type(led).__name__}, not a set"
            )
        out[name] = sorted([str(p), int(i)] for p, i in led)
    return out


def _defter_uygula(model, defter: dict) -> tuple[int, int]:
    """Mark ONLY the parts the dump recorded. A ledger entry for a module the
    model lacks is refused; a ledgered module missing from the entry gets
    nothing, so the plugin's own check refuses it."""
    mods = {name: mod for name, mod in model.named_modules() if hasattr(mod, DEFTER_ATTR)}
    extra = sorted(set(defter) - set(mods))
    if extra:
        raise RuntimeError(
            f"harem-fastload: the ledger names {len(extra)} module(s) this model does "
            f"not have: {extra[:5]}"
        )
    parts = 0
    for name, mod in mods.items():
        rec = defter.get(name, [])
        getattr(mod, DEFTER_ATTR).update((str(p), int(i)) for p, i in rec)
        parts += len(rec)
    return len(mods), parts


def _entries(model) -> list[tuple[str, torch.Tensor]]:
    """Every parameter and buffer the rank holds, in a stable order.

    ``remove_duplicate=False`` so tied tensors appear under every name they have;
    the dump stores the bytes once and records the alias.
    """
    out: list[tuple[str, torch.Tensor]] = []
    seen: dict[str, torch.Tensor] = {}
    for name, t in list(model.named_parameters(remove_duplicate=False)) + list(
        model.named_buffers(remove_duplicate=False)
    ):
        if name in seen:
            if seen[name] is not t:
                raise RuntimeError(
                    f"harem-fastload: name {name!r} maps to two different tensors"
                )
            continue
        seen[name] = t
        out.append((name, t))
    return out


def _canon_u8(t: torch.Tensor) -> torch.Tensor:
    """A 1-D uint8 view of a tensor's values in CANONICAL (row-major) order.

    A few buffers are stored transposed (the drafter keeps a (22,512,256) cache
    with stride (262144,1,512)). What has to survive a dump/restore round trip
    is the *values*, not the stride, so the sidecar always holds the canonical
    layout and ``copy_`` puts it back into whatever layout the model wants.
    ``.contiguous()`` is a no-op for every real weight tensor.
    """
    x = t.detach()
    if not x.is_contiguous():
        x = x.contiguous()
    x = x.reshape(-1)
    if x.numel() == 0:
        return x.to(torch.uint8) if x.dtype != torch.uint8 else x
    return x.view(torch.uint8)


def _flat_u8(t: torch.Tensor) -> torch.Tensor:
    """Same as _canon_u8 but refuses to copy -- used only where a *view* of the
    destination is needed so the restore writes in place."""
    x = t.detach()
    if not x.is_contiguous():
        raise RuntimeError(
            f"harem-fastload: destination is not contiguous (shape={tuple(x.shape)}, "
            f"stride={x.stride()})"
        )
    x = x.reshape(-1)
    if x.numel() == 0:
        return x.to(torch.uint8) if x.dtype != torch.uint8 else x
    return x.view(torch.uint8)


def _cpu_bytes(t: torch.Tensor) -> torch.Tensor:
    return _canon_u8(t).to("cpu", copy=True)


def _write_back(dst: torch.Tensor, src_u8: torch.Tensor) -> None:
    """Copy the canonical bytes in *src_u8* (CPU, 1-D uint8) into *dst*."""
    if dst.numel() == 0:
        return
    if dst.is_contiguous():
        _flat_u8(dst).copy_(src_u8)
        return
    # strided destination: rebuild the logical tensor, let copy_ do the layout
    dst.copy_(src_u8.view(dst.dtype).reshape(dst.shape))


def _sha(cpu_u8: torch.Tensor) -> str:
    if cpu_u8.numel() == 0:
        return hashlib.sha256(b"").hexdigest()
    return hashlib.sha256(memoryview(cpu_u8.numpy())).hexdigest()


def _paths():
    """The MAIN model and draft directories -- never the model currently being
    loaded, so the drafter's own dump lands under the same identity as the
    target's instead of looking like a different build."""
    model_dir = os.environ.get("HAREM_FASTLOAD_MODEL_PATH", "")
    draft = os.environ.get("HAREM_FASTLOAD_DRAFT_PATH", "")
    if not model_dir:
        raise RuntimeError("harem-fastload: HAREM_FASTLOAD_MODEL_PATH is not set")
    return model_dir, draft


_ADDR = __import__("re").compile(r" at 0x[0-9a-fA-F]+")


def _scrub(v):
    """Strip CPython object addresses out of a repr.

    ``hf_overrides`` is a *callable* for the DFlash drafter
    (``SpeculativeConfig.hf_config_override``), so its repr carries a heap
    address that differs on every boot. An address is not identity; without
    this the drafter's sidecar is "stale" the instant it is written.
    """
    return _ADDR.sub("", v) if isinstance(v, str) else v


def model_identity(model_config) -> dict:
    """The per-model half of the identity: what this particular sub-model was
    loaded as. Kept out of the shared identity because the target and the
    drafter legitimately differ here."""
    return {
        "model_path": str(model_config.model),
        "dtype": str(model_config.dtype),
        "quantization": str(model_config.quantization),
        "hf_overrides": _scrub(
            json.dumps(
                getattr(model_config, "hf_overrides", None), sort_keys=True, default=str
            )
        ),
        "revision": str(model_config.revision),
    }


def engine_identity() -> dict:
    """File identity plus the things only the running engine knows. Identical
    for every sub-model in one boot."""
    from vllm.config import get_current_vllm_config

    vc = get_current_vllm_config()
    pc = vc.parallel_config
    model_dir, draft_dir = _paths()
    ident = hid.file_identity(model_dir, draft_dir, os.environ.get("TP3_DIR", "/opt/harem-tp3"))
    try:
        import vllm

        vllm_ver = vllm.__version__
    except Exception:  # pragma: no cover
        vllm_ver = "?"
    try:
        import cuda_exl3

        exl3_ver = getattr(cuda_exl3, "__version__", "?")
    except Exception:  # pragma: no cover
        exl3_ver = "?"
    ident["engine"] = {
        "vllm_version": vllm_ver,
        "cuda_exl3_version": exl3_ver,
        "tensor_parallel_size": pc.tensor_parallel_size,
        "pipeline_parallel_size": pc.pipeline_parallel_size,
        "data_parallel_size": pc.data_parallel_size,
        "enable_expert_parallel": bool(pc.enable_expert_parallel),
        "expert_placement_strategy": str(pc.expert_placement_strategy),
    }
    return ident



def _drop_ckpt_cache(where: str) -> None:
    """Tell the kernel it can forget the checkpoint files we just streamed.

    The GB10 GPU pool is host memory, and vLLM sizes the KV pool from the
    *free* memory left after loading. Page cache is not free memory. Reading
    163.58 GiB through the page cache therefore taxes the KV pool even though
    nothing needs those pages again. POSIX_FADV_DONTNEED on the read-only
    checkpoint shards is the exact undo, and costs milliseconds.
    """
    if os.environ.get("HAREM_DROP_CKPT_CACHE", "1").strip() not in ("1", "true", "yes"):
        return
    import glob as _glob

    # deliberately NOT _paths(): this runs on the plain upstream path too, where
    # the fastload env may be absent. No paths -> nothing to do, never an error.
    dirs = [os.environ.get("HAREM_FASTLOAD_MODEL_PATH", ""),
            os.environ.get("HAREM_FASTLOAD_DRAFT_PATH", "")]
    n = 0
    for d in {os.path.realpath(x) for x in dirs if x and os.path.isdir(x)}:
        for f in _glob.glob(os.path.join(d, "*.safetensors")):
            try:
                fd = os.open(f, os.O_RDONLY)
            except OSError:
                continue
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                n += 1
            except OSError:
                pass
            finally:
                os.close(fd)
    if n:
        logger.info(
            "[harem-fastload] page cache dropped for %d checkpoint shards after %s", n, where
        )



def _malloc_trim(where: str) -> None:
    """Give the glibc arenas back to the OS after a load.

    On GB10 the GPU pool IS host memory, so vLLM's "non-torch memory" term --
    measured after weight loading, subtracted from the KV budget -- includes
    whatever the loader's transient host buffers left mapped. The eager
    safetensors strategy reads a whole shard into bytes and deserialises it,
    so ~2.7 GiB of arena stays resident and comes straight out of the KV pool
    (measured 5 Eyl: KV 4,413,404 -> 4,231,404, -4.1%). malloc_trim costs
    milliseconds and is numerically inert.
    """
    if os.environ.get("HAREM_MALLOC_TRIM", "1").strip() not in ("1", "true", "yes"):
        return
    try:
        import ctypes

        before = _rss_gib()
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        after = _rss_gib()
        logger.info(
            "[harem-fastload] malloc_trim after %s: RSS %.2f -> %.2f GiB",
            where, before, after,
        )
    except Exception as e:  # pragma: no cover - never fail a boot over this
        logger.warning("[harem-fastload] malloc_trim failed: %s", e)


def _rss_gib() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / (1 << 20)
    except OSError:
        pass
    return float("nan")


# --------------------------------------------------------------------------- #
# dump
# --------------------------------------------------------------------------- #
def _dump(model, model_config, d: str, tag: str) -> None:
    from safetensors.torch import save_file

    os.makedirs(d, exist_ok=True)
    entries = _entries(model)
    budget = _shard_bytes()
    t0 = time.perf_counter()
    logger.info(
        "[harem-fastload] dump start tag=%s dir=%s tensors=%d", tag, d, len(entries)
    )

    tensors: dict[str, dict] = {}
    shards: list[str] = []
    by_ptr: dict[tuple[int, int], str] = {}
    buf: dict[str, torch.Tensor] = {}
    buf_bytes = 0
    total_bytes = 0

    def flush():
        nonlocal buf, buf_bytes
        if not buf:
            return
        name = f"{tag}-{len(shards):05d}.safetensors"
        p = os.path.join(d, name)
        save_file(buf, p)
        fd = os.open(p, os.O_RDONLY)
        try:
            os.fsync(fd)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
        for n in buf:
            tensors[n]["shard"] = name
        shards.append(name)
        logger.info(
            "[harem-fastload] shard %s: %d tensors, %.2f GiB", name, len(buf),
            buf_bytes / _GiB,
        )
        buf = {}
        buf_bytes = 0

    for name, t in entries:
        if t.is_meta:
            raise RuntimeError(
                f"harem-fastload: {name} is still on the meta device after "
                "load_weights; refusing to dump an unmaterialised model"
            )
        nb = _nbytes(t)
        meta = {
            "shape": list(t.shape),
            "dtype": str(t.dtype).replace("torch.", ""),
            "nbytes": nb,
            "contiguous": bool(t.is_contiguous()),
        }
        key = (t.data_ptr(), nb)
        if nb and key in by_ptr:
            meta["alias_of"] = by_ptr[key]
            tensors[name] = meta
            continue
        cpu = _cpu_bytes(t)
        meta["sha256"] = _sha(cpu)
        meta["shard"] = None
        tensors[name] = meta
        by_ptr[key] = name
        buf[name] = cpu
        buf_bytes += nb
        total_bytes += nb
        if buf_bytes >= budget:
            flush()
    flush()

    man_path = os.path.join(d, MANIFEST)
    man = {}
    if os.path.isfile(man_path):
        try:
            man = json.load(open(man_path))
        except Exception:
            man = {}
    if man.get("identity") and hid.diff(man["identity"], engine_identity()):
        # a second model (the drafter) written into the same rank dir must agree
        # with the first, otherwise the dir is a mix of two builds.
        raise RuntimeError(
            "harem-fastload: the rank directory already holds a sidecar from a "
            "different build; remove it before dumping"
        )
    man.setdefault("schema", hid.SCHEMA)
    man["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    man["host"] = socket.gethostname()
    man["rank"] = _rank()
    man["identity"] = engine_identity()
    man.setdefault("models", {})
    defter = _defter_topla(model)
    man["models"][tag] = {
        **model_identity(model_config),
        "n_tensors": len(tensors),
        "bytes": total_bytes,
        "shards": shards,
        "tensors": tensors,
        DEFTER_ALAN: {"surum": DEFTER_SURUM, "moduller": defter},
    }
    tmp = man_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(man, f, sort_keys=True)
    os.replace(tmp, man_path)
    dt = time.perf_counter() - t0
    logger.info(
        "[harem-fastload] dump done tag=%s %.2f GiB in %d shards, %.1f s (%.0f MB/s)",
        tag, total_bytes / _GiB, len(shards), dt, total_bytes / max(dt, 1e-9) / 1e6,
    )
    logger.info(
        "[harem-fastload] ledger dumped tag=%s: %d module(s), %d part(s)",
        tag, len(defter), sum(len(v) for v in defter.values()),
    )


# --------------------------------------------------------------------------- #
# restore
# --------------------------------------------------------------------------- #
def _pick_sample(names: list[str], n: int) -> list[str]:
    """A stratified, deterministic sample: experts, attention, embeddings,
    norms, everything else -- so "we checked 200 tensors" is not "we checked
    200 layer norms"."""
    order = ["expert", "attn", "embed", "norm", "other"]
    buckets: dict[str, list[str]] = {k: [] for k in order}
    for x in sorted(names):
        low = x.lower()
        if "expert" in low or "w13" in low or "w2_" in low:
            buckets["expert"].append(x)
        elif "attn" in low or "_proj" in low or "kv_" in low or "q_" in low:
            buckets["attn"].append(x)
        elif "embed" in low or "lm_head" in low:
            buckets["embed"].append(x)
        elif "norm" in low:
            buckets["norm"].append(x)
        else:
            buckets["other"].append(x)
    live = [k for k in order if buckets[k]]
    if not live:
        return []
    out: list[str] = []
    per = max(1, n // len(live))
    for k in live:
        b = buckets[k]
        take = min(per, len(b))
        step = max(1, len(b) // take)
        out.extend(b[::step][:take])
    if len(out) < min(n, len(names)):
        seen = set(out)
        for x in sorted(names):
            if x not in seen:
                out.append(x)
                seen.add(x)
                if len(out) >= min(n, len(names)):
                    break
    return out[:n]


def _restore(model, model_config, d: str, tag: str) -> None:
    from safetensors.torch import load as st_load

    t0 = time.perf_counter()
    man_path = os.path.join(d, MANIFEST)
    if not os.path.isfile(man_path):
        raise RuntimeError(f"harem-fastload: no {MANIFEST} in {d}")
    man = json.load(open(man_path))
    logger.info(
        "[harem-fastload] restore start tag=%s dir=%s manifest=%s",
        tag, d, man.get("created_utc"),
    )
    if int(man.get("rank", -1)) != _rank():
        raise RuntimeError(
            f"harem-fastload: {d} was written by rank {man.get('rank')}, this is rank {_rank()}"
        )
    problems = hid.diff(man.get("identity", {}), engine_identity())
    if problems:
        raise RuntimeError(
            "harem-fastload: the sidecar is stale - refusing to boot.\n  "
            + "\n  ".join(problems[:20])
            + f"\n  ({len(problems)} fark). Yeniden uretmek icin: HAREM_FASTLOAD_MODE=dump"
        )
    if tag not in man.get("models", {}):
        raise RuntimeError(
            f"harem-fastload: {d} holds {sorted(man.get('models', {}))}, not {tag!r}"
        )
    m = man["models"][tag]
    led = m.get(DEFTER_ALAN)
    if not isinstance(led, dict) or led.get("surum") != DEFTER_SURUM or not isinstance(led.get("moduller"), dict):
        raise RuntimeError(
            f"harem-fastload: {tag} sidecar in {d} carries no written-parts ledger "
            f"(version {DEFTER_SURUM}); it predates the ledger. Refusing -- re-dump "
            "with HAREM_FASTLOAD_MODE=dump (no fall back to a normal load)."
        )
    mine = model_identity(model_config)
    # scrub the stored side too: a sidecar written before the address scrubbing
    # existed is not stale, it just recorded a heap address.
    mprob = hid.diff({k: _scrub(m[k]) for k in mine if k in m}, mine)
    if mprob:
        raise RuntimeError(
            f"harem-fastload: {tag} sidecar was written for a different model "
            "configuration - refusing to boot.\n  " + "\n  ".join(mprob)
        )
    entries = _entries(model)
    tmap = dict(entries)
    want = set(m["tensors"])
    have = set(tmap)
    if want != have:
        missing = sorted(want - have)[:10]
        extra = sorted(have - want)[:10]
        raise RuntimeError(
            f"harem-fastload: model/sidecar tensor names differ "
            f"(sidecar-only {len(want - have)}: {missing}; model-only {len(have - want)}: {extra})"
        )

    by_shard: dict[str, list[str]] = {}
    aliases: list[tuple[str, str]] = []
    for n, meta in m["tensors"].items():
        if meta.get("alias_of"):
            aliases.append((n, meta["alias_of"]))
            continue
        by_shard.setdefault(meta["shard"], []).append(n)

    done = 0
    bytes_done = 0
    read_mode = os.environ.get("HAREM_FASTLOAD_READ", "buffered").strip().lower()
    if read_mode not in ("buffered", "mmap"):
        raise RuntimeError(f"harem-fastload: HAREM_FASTLOAD_READ={read_mode!r} (buffered|mmap)")
    for shard in m["shards"]:
        p = os.path.join(d, shard)
        if read_mode == "mmap":
            from safetensors import safe_open

            fh = safe_open(p, framework="pt")
            sd = None
        else:
            # one sequential read per shard, then a contiguous copy per tensor
            with open(p, "rb") as f:
                blob = f.read()
            sd = st_load(blob)
            del blob
            fh = None
        for n in by_shard.get(shard, []):
            dst = tmap[n]
            meta = m["tensors"][n]
            if list(dst.shape) != meta["shape"] or str(dst.dtype).replace("torch.", "") != meta["dtype"]:
                raise RuntimeError(
                    f"harem-fastload: {n} shape/dtype changed: sidecar "
                    f"{meta['shape']}/{meta['dtype']} vs model {list(dst.shape)}/{dst.dtype}"
                )
            src = sd[n] if sd is not None else fh.get_tensor(n)
            if src.numel() != meta["nbytes"]:
                raise RuntimeError(f"harem-fastload: {n} byte count changed in the sidecar")
            if dst.is_meta:
                raise RuntimeError(f"harem-fastload: {n} is on the meta device")
            _write_back(dst, src)
            done += 1
            bytes_done += meta["nbytes"]
        del sd, fh
    for alias, canon in aliases:
        a, c = tmap[alias], tmap[canon]
        if a.data_ptr() != c.data_ptr():
            _write_back(a, _canon_u8(c))
        done += 1
    n_mod, n_part = _defter_uygula(model, led["moduller"])
    logger.info(
        "[harem-fastload] ledger restored tag=%s: %d module(s), %d part(s) marked", tag, n_mod, n_part
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    logger.info(
        "Loading weights took %.2f seconds", dt
    )
    logger.info(
        "[harem-fastload] restored %d tensors, %.2f GiB from %d shards in %.1f s "
        "(%.0f MB/s, read=%s)",
        done, bytes_done / _GiB, len(m["shards"]), dt,
        bytes_done / max(dt, 1e-9) / 1e6, read_mode,
    )
    _verify(model, m, tmap, tag)


def _verify(model, m: dict, tmap: dict, tag: str) -> None:
    spec = os.environ.get("HAREM_FASTLOAD_VERIFY", "64").strip().lower()
    if spec in ("0", "no", "off", ""):
        logger.warning("[harem-fastload] verification skipped (HAREM_FASTLOAD_VERIFY=%s)", spec)
        return
    hashed = [n for n, meta in m["tensors"].items() if meta.get("sha256")]
    if spec in ("all", "-1"):
        sample = sorted(hashed)
    else:
        sample = _pick_sample(hashed, int(spec))
    t0 = time.perf_counter()
    bad = []
    nbytes = 0
    for n in sample:
        cpu = _cpu_bytes(tmap[n])
        nbytes += cpu.numel()
        if _sha(cpu) != m["tensors"][n]["sha256"]:
            bad.append(n)
    dt = time.perf_counter() - t0
    if bad:
        raise RuntimeError(
            f"harem-fastload: {len(bad)}/{len(sample)} tensors do NOT match the "
            f"manifest hash: {bad[:10]}"
        )
    logger.info(
        "[harem-fastload] verify OK tag=%s %d/%d tensors re-hashed (%.2f GiB, %.1f s)",
        tag, len(sample), len(hashed), nbytes / _GiB, dt,
    )


# --------------------------------------------------------------------------- #
# hooks called from base_loader.load_model
# --------------------------------------------------------------------------- #
def load_weights_hook(loader, model, model_config) -> None:
    mode = _mode()
    if mode not in ("dump", "load"):
        loader.load_weights(model, model_config)
        _drop_ckpt_cache("upstream-load")
        _malloc_trim("upstream-load")
        return
    d = _rank_dir()
    tag = _tag(model_config)
    only = os.environ.get("HAREM_FASTLOAD_ONLY", "").strip()
    if only and tag not in [x.strip() for x in only.split(",")]:
        logger.info("[harem-fastload] tag=%s not in HAREM_FASTLOAD_ONLY; normal load", tag)
        loader.load_weights(model, model_config)
        _drop_ckpt_cache(f"normal/{tag}")
        _malloc_trim(f"normal/{tag}")
        return
    if mode == "load":
        _restore(model, model_config, d, tag)
        _drop_ckpt_cache(f"restore/{tag}")
        _malloc_trim(f"restore/{tag}")
        return
    loader.load_weights(model, model_config)
    _dump(model, model_config, d, tag)
    _drop_ckpt_cache(f"dump/{tag}")
    _malloc_trim(f"dump/{tag}")


def after_process_hook(model, model_config) -> None:
    label = os.environ.get("HAREM_FASTLOAD_POSTHASH", "").strip()
    if not label:
        return
    tag = _tag(model_config)
    rank = _rank()
    out = f"/cache/harem-poststate-{tag}-r{rank}-{label}.json"
    t0 = time.perf_counter()
    names = [n for n, _ in _entries(model)]
    spec = os.environ.get("HAREM_FASTLOAD_POSTHASH_N", "all").strip().lower()
    want = set(names) if spec in ("all", "-1", "") else set(_pick_sample(names, int(spec)))
    h: dict[str, str] = {}
    for name, t in _entries(model):
        if name not in want:
            continue
        if t.is_meta:
            h[name] = "meta"
            continue
        h[name] = _sha(_cpu_bytes(t))
    with open(out, "w") as f:
        json.dump(
            {"tag": tag, "rank": rank, "label": label, "n": len(h),
             "n_total": len(names), "sample": spec, "sha256": h},
            f, sort_keys=True,
        )
    logger.info(
        "[harem-fastload] post-process hashes: %d tensors -> %s (%.1f s)",
        len(h), out, time.perf_counter() - t0,
    )
