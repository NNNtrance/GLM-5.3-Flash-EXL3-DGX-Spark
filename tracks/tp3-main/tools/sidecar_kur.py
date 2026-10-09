#!/usr/bin/env python3
"""HAREM offline sidecar builder (runs on a node, inside the image, before the first boot).

Does NOT touch the original checkpoint (it only reads). It builds a sidecar directory next to it:

  hedef   ("target") for the GLM-5.3-Flash turboderp EXL3 checkpoint:
          * The fused ``self_attn.qkv_proj`` EXL3 tensor of the KDA layers is split into
            q/k/v_proj: the trellis [K/16, N/16, 16*b] is sliced on the 512-wide (8,192
            column) TILE boundaries, suh is copied three times, svh is sliced, mul1 is
            copied. There is NO requantization; the 128-wide Hadamard blocks and the 16x16
            tiles are not cut, so every part is an independent EXL3 tensor. (vLLM's generic
            splitter at TP3 would use the layer's PADDED sizes and shift q/k/v.)
          * ``self_attn.conv1d.weight`` [3P, 1, k] -> q/k/v_conv1d (a third each, dim 0).
          * The dead fused ``attn.qkv.{weight,bias}`` tensors of the vision tower (in the
            blocks that have EXL3 q/k/v_proj) are dropped.
          The split tensors are written to ``harem-bolme-0000N.safetensors``; every other
          file is a SYMLINK RELATIVE TO THE ORIGINAL (except config.json: the
          packed_modules_mapping is added to the inline quantization_config, because vLLM
          reads the inline config before quantization_config_file).
          ``model.safetensors.index.json`` is rewritten and carries
          ``metadata.harem_sidecar``: the index is AUTHORITATIVE
          (patches/yama-sidecar-dizin.py), the dropped names are listed in ``dusurulen``.
          ``quantization_config.json``: the qkv_proj entries are split into q/k/v_proj and
          ``packed_modules_mapping`` is added (in_proj_qkvbfg_a mixed, MLA
          fused_qkv_a_proj, vision qkv_proj).
  taslak  ("draft") the DFlash2 draft: config.json is padded with
          ``cuda_exl3._harem_tp3.draft_plan`` / ``apply_draft_plan`` and marked with the
          ``harem_tp_pad`` tag (TP3: 32/8 -> 36/9); the weights are symlinks.
  dogrula ("verify") checks an existing target sidecar against the original, byte for byte.
  kesik   ("truncated"; a single-node TEST mode, not used by the recipe) builds a TEST
          sidecar of the first N layers from a TARGET sidecar: config has num_hidden_layers=N,
          layer_types / mlp_layer_types / indexer_types cut to the first N,
          linear_attn_config.kda_layers / full_attn_layers < N. The index carries only the
          tensors of layers < N and the layerless ones (embedding, norm, lm_head, vision
          tower); EVERY other tensor in the referenced shards (layers >= N, and what the
          target dropped) is written to ``dusurulen`` so that the index filter
          (yama-sidecar-dizin) counts them as "superseded", skips them and checks the count.
          The weight files are symlinks relative to the files of the target sidecar (no copy).
  kesik-dogrula  checks an existing truncated sidecar against its source (the target sidecar).

Usage:
  sidecar_kur.py hedef  --kaynak DIR --cikti DIR
  sidecar_kur.py taslak --kaynak DIR --cikti DIR --tp 3 [--kaynak-config F] [--harem-tp3 PATH]
  sidecar_kur.py dogrula --kaynak DIR --cikti DIR
  sidecar_kur.py kesik  --kaynak TARGET_SIDECAR --cikti DIR --katman N
  sidecar_kur.py kesik-dogrula --kaynak TARGET_SIDECAR --cikti DIR
Exit: 0 ok; 2 input/format error; 3 the output directory exists and is not ours; 4 verification.
Idempotent: if a sidecar built from the same source with the same tool version exists, it is
only verified, not rewritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import sys
import time
from pathlib import Path

SURUM = 1
INDEX = "model.safetensors.index.json"
QCFG = "quantization_config.json"
BOLME_DOSYA = "harem-bolme-{:05d}.safetensors"
BOLME_SINIRI = 2 << 30  # 2 GiB / file
QKV_RE = re.compile(r"^(?P<pfx>.+\.layers\.\d+\.self_attn)\.qkv_proj\.(?P<leaf>trellis|suh|svh|mul1)$")
CONV_RE = re.compile(r"^(?P<pfx>.+\.layers\.\d+\.self_attn)\.conv1d\.weight$")
VIS_RE = re.compile(r"^(?P<pfx>.*visual\.blocks\.\d+\.attn)\.qkv\.(weight|bias)$")
PACKED = {
    "in_proj_qkvbfg_a": ["q_proj", "k_proj", "v_proj", "b_proj", "f_a_proj", "g_a_proj"],
    "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
}
DT_BAYT = {"I16": 2, "F16": 2, "BF16": 2, "I32": 4, "F32": 4, "U8": 1, "I8": 1, "F64": 8, "I64": 8}


class Hata(Exception):
    def __init__(self, kod, mesaj):
        super().__init__(mesaj)
        self.kod = kod


def arac_sha() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def baslik_oku(p: Path) -> tuple[dict, int]:
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    return h, 8 + n


def tensor_oku(p: Path, veri_bas: int, ofs: tuple) -> bytes:
    a, b = ofs
    with open(p, "rb") as f:
        f.seek(veri_bas + a)
        v = f.read(b - a)
    if len(v) != b - a:
        raise Hata(2, f"{p}: kısa okuma")
    return v


def safetensors_yaz(p: Path, tensorler: list) -> dict:
    """tensorler ("tensors"): [(name, dtype, shape, bytes)] -> header (name -> meta)."""
    hdr, ofs = {"__metadata__": {"format": "pt", "harem": f"sidecar_kur v{SURUM}"}}, 0
    for ad, dt, shape, veri in tensorler:
        bekl = DT_BAYT[dt]
        for s in shape:
            bekl *= s
        if bekl != len(veri):
            raise Hata(2, f"{ad}: {len(veri)} bayt, şekil {shape} {dt} için {bekl}")
        hdr[ad] = {"dtype": dt, "shape": list(shape), "data_offsets": [ofs, ofs + len(veri)]}
        ofs += len(veri)
    hb = json.dumps(hdr, separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for _, _, _, veri in tensorler:
            f.write(veri)
    os.replace(tmp, p)
    return {k: v for k, v in hdr.items() if k != "__metadata__"}


# ---------------------------------------------------------------- splitting (pure)

def trellis_bol(veri: bytes, shape: list) -> list:
    """[K16, N16, W] int16 -> three parts, cut along dim 1 on the tile boundaries."""
    import numpy as np

    k16, n16, w = shape
    if n16 % 3:
        raise Hata(2, f"trellis {shape}: N/16 üçe bölünmüyor")
    p = n16 // 3
    if (p * 16) % 128:
        raise Hata(2, f"trellis {shape}: parça {p * 16} sütun 128'in katı değil (Hadamard bloğu kesilir)")
    a = np.frombuffer(veri, dtype="<i2").reshape(k16, n16, w)
    return [np.ascontiguousarray(a[:, i * p:(i + 1) * p, :]).tobytes() for i in range(3)], [k16, p, w]


def dim0_bol(veri: bytes, shape: list, dt: str) -> tuple:
    """Into three along dim 0 (svh [N], conv1d [N,1,k]) -> contiguous byte slices."""
    if shape[0] % 3:
        raise Hata(2, f"{shape}: dim 0 üçe bölünmüyor")
    satir = DT_BAYT[dt]
    for s in shape[1:]:
        satir *= s
    p = shape[0] // 3
    return [veri[i * p * satir:(i + 1) * p * satir] for i in range(3)], [p] + list(shape[1:])


# ---------------------------------------------------------------- hedef (target)

def kaynak_tara(kaynak: Path) -> dict:
    idx = json.loads((kaynak / INDEX).read_text())
    wm = idx["weight_map"]
    basliklar = {}
    for f in sorted(set(wm.values())):
        basliklar[f] = baslik_oku(kaynak / f)
    for ad, f in wm.items():
        if ad not in basliklar[f][0]:
            raise Hata(2, f"dizin {ad} -> {f} diyor ama başlıkta yok")
    return {"index": idx, "wm": wm, "basliklar": basliklar}


def plan_cikar(t: dict) -> dict:
    wm = t["wm"]
    qkv, conv, vis_eski = {}, {}, []
    for ad in wm:
        m = QKV_RE.match(ad)
        if m:
            qkv.setdefault(m["pfx"], {})[m["leaf"]] = ad
            continue
        m = CONV_RE.match(ad)
        if m:
            conv[m["pfx"]] = ad
            continue
        m = VIS_RE.match(ad)
        if m and f"{m['pfx']}.q_proj.trellis" in wm:
            vis_eski.append(ad)
    for pfx, d in qkv.items():
        if set(d) != {"trellis", "suh", "svh", "mul1"}:
            raise Hata(2, f"{pfx}.qkv_proj eksik tensör: {sorted(d)}")
        for leaf in ("q_proj", "k_proj", "v_proj", "q_conv1d", "k_conv1d", "v_conv1d"):
            if any(n.startswith(f"{pfx}.{leaf}.") for n in wm):
                raise Hata(2, f"{pfx}.{leaf} zaten var; checkpoint zaten bölünmüş görünüyor")
    if set(qkv) != set(conv):
        raise Hata(2, f"qkv_proj ve conv1d katmanları farklı: {sorted(set(qkv) ^ set(conv))[:4]}")
    if not qkv:
        raise Hata(2, "bölünecek KDA qkv_proj bulunamadı")
    return {"qkv": dict(sorted(qkv.items(), key=lambda kv: _katman(kv[0]))),
            "conv": conv, "vis_eski": sorted(vis_eski)}


def _katman(pfx: str) -> int:
    return int(re.search(r"\.layers\.(\d+)\.", pfx + ".").group(1))


def bolunmus_tensorler(kaynak: Path, t: dict, plan: dict):
    """Generator of (name, dtype, shape, bytes), in KDA layer order."""
    wm, bas = t["wm"], t["basliklar"]

    def oku(ad):
        f = wm[ad]
        h, vb = bas[f]
        m = h[ad]
        return m["dtype"], m["shape"], tensor_oku(kaynak / f, vb, m["data_offsets"])

    for pfx, d in plan["qkv"].items():
        dt, sh, v = oku(d["trellis"])
        if dt != "I16" or len(sh) != 3:
            raise Hata(2, f"{d['trellis']}: beklenmeyen {dt} {sh}")
        parcalar, psh = trellis_bol(v, sh)
        n = sh[1] * 16
        dt_s, sh_s, v_s = oku(d["suh"])
        dt_v, sh_v, v_v = oku(d["svh"])
        dt_m, sh_m, v_m = oku(d["mul1"])
        if sh_s != [sh[0] * 16] or sh_v != [n] or sh_m != []:
            raise Hata(2, f"{pfx}: suh {sh_s} / svh {sh_v} / mul1 {sh_m} trellis {sh} ile uyuşmuyor")
        svh_p, svh_sh = dim0_bol(v_v, sh_v, dt_v)
        dtc, shc, vc = oku(plan["conv"][pfx])
        if shc[0] != n:
            raise Hata(2, f"{plan['conv'][pfx]}: {shc} qkv genişliği {n} ile uyuşmuyor")
        conv_p, conv_sh = dim0_bol(vc, shc, dtc)
        for i, adp in enumerate(("q_proj", "k_proj", "v_proj")):
            yield f"{pfx}.{adp}.trellis", "I16", psh, parcalar[i]
            yield f"{pfx}.{adp}.suh", dt_s, sh_s, v_s
            yield f"{pfx}.{adp}.svh", dt_v, svh_sh, svh_p[i]
            yield f"{pfx}.{adp}.mul1", dt_m, sh_m, v_m
        for i, adc in enumerate(("q_conv1d", "k_conv1d", "v_conv1d")):
            yield f"{pfx}.{adc}.weight", dtc, conv_sh, conv_p[i]


def qcfg_yeni(kaynak: Path, plan: dict, yeni_bas: dict) -> dict:
    q = json.loads((kaynak / QCFG).read_text())
    ts = q["tensor_storage"]
    for pfx in plan["qkv"]:
        eski = ts.pop(f"{pfx}.qkv_proj", None)
        if eski is None or eski.get("quant_format") != "exl3":
            raise Hata(2, f"{QCFG}: {pfx}.qkv_proj exl3 girdisi yok")
        for adp in ("q_proj", "k_proj", "v_proj"):
            g = {k: v for k, v in eski.items() if k != "stored_tensors"}
            st = {}
            for leaf in ("suh", "svh", "mul1", "trellis"):
                ad = f"{pfx}.{adp}.{leaf}"
                m = yeni_bas[ad]
                st[ad] = {"shape": m["shape"],
                          "n_bytes": m["data_offsets"][1] - m["data_offsets"][0],
                          "dtype": eski["stored_tensors"][f"{pfx}.qkv_proj.{leaf}"]["dtype"]}
            g["stored_tensors"] = st
            ts[f"{pfx}.{adp}"] = g
    eski_pm = q.get("packed_modules_mapping") or {}
    for k, v in PACKED.items():
        if k in eski_pm and eski_pm[k] != v:
            raise Hata(2, f"{QCFG}: packed_modules_mapping[{k}] zaten farklı: {eski_pm[k]}")
    q["packed_modules_mapping"] = {**eski_pm, **PACKED}
    return q


def config_yeni(kaynak: Path) -> dict:
    """config.json + packed_modules_mapping inside the inline quantization_config.

    vLLM 21d93d0d8 reads the inline quantization_config in config.json BEFORE
    ``--hf-overrides quantization_config_file`` (weight_utils.get_quant_config), and
    cuda-exl3 takes the mapping from the inline dictionary before it merges tensor_storage
    from the file: if the mapping stayed only in quantization_config.json it would be lost
    silently."""
    c = json.loads((kaynak / "config.json").read_text())
    q = c.get("quantization_config")
    if not isinstance(q, dict) or q.get("quant_method") != "exl3":
        raise Hata(2, "config.json'da satır içi exl3 quantization_config yok")
    eski = q.get("packed_modules_mapping") or {}
    for k, v in PACKED.items():
        if k in eski and eski[k] != v:
            raise Hata(2, f"config.json: packed_modules_mapping[{k}] zaten farklı: {eski[k]}")
    q["packed_modules_mapping"] = {**eski, **PACKED}
    return c


def hedef_kur(kaynak: Path, cikti: Path) -> dict:
    kaynak, cikti = kaynak.resolve(), cikti.absolute()
    if not (kaynak / INDEX).is_file() or not (kaynak / QCFG).is_file():
        raise Hata(2, f"{kaynak}: {INDEX} ya da {QCFG} yok")
    if cikti.exists():
        idx = cikti / INDEX
        meta = json.loads(idx.read_text()).get("metadata", {}).get("harem_sidecar") if idx.is_file() else None
        if meta and meta.get("kaynak") == str(kaynak) and meta.get("arac_sha256") == arac_sha():
            r = dogrula(kaynak, cikti)
            r["durum"] = "zaten-var (dogrulandi)"
            return r
        if any(cikti.iterdir()):
            raise Hata(3, f"{cikti} var ve bu araçla bu kaynaktan kurulmuş bir sidecar değil; dokunulmadı")
    t = kaynak_tara(kaynak)
    plan = plan_cikar(t)
    gecici = cikti.with_name(cikti.name + ".kuruluyor")
    if gecici.exists():
        raise Hata(3, f"{gecici} var (yarım kurulum?); elle bakın")
    gecici.mkdir(parents=True)
    t0 = time.time()
    yeni_wm, yeni_bas, dosya_no, tampon, boy = {}, {}, 1, [], 0

    def bosalt():
        nonlocal dosya_no, tampon, boy
        if not tampon:
            return
        ad = BOLME_DOSYA.format(dosya_no)
        h = safetensors_yaz(gecici / ad, tampon)
        for k, m in h.items():
            yeni_wm[k] = ad
            yeni_bas[k] = m
        dosya_no, tampon, boy = dosya_no + 1, [], 0

    for x in bolunmus_tensorler(kaynak, t, plan):
        if boy + len(x[3]) > BOLME_SINIRI:
            bosalt()
        tampon.append(x)
        boy += len(x[3])
    bosalt()
    dusen = sorted([a for d in plan["qkv"].values() for a in d.values()]
                   + list(plan["conv"].values()) + plan["vis_eski"])
    wm = {k: v for k, v in t["wm"].items() if k not in set(dusen)}
    cakisan = set(wm) & set(yeni_wm)
    if cakisan:
        raise Hata(2, f"yeni adlar orijinalle çakışıyor: {sorted(cakisan)[:3]}")
    wm.update(yeni_wm)
    for f in sorted(os.listdir(kaynak)):
        if f in (INDEX, QCFG, "config.json") or f.startswith("."):
            continue
        if (gecici / f).exists():
            raise Hata(2, f"{f}: kaynakta da bizim dosya adımızla aynı ad var")
        os.symlink(os.path.relpath(kaynak / f, cikti), gecici / f)
    (gecici / "config.json").write_text(json.dumps(config_yeni(kaynak), indent=2) + "\n")
    meta = dict(t["index"].get("metadata") or {})
    bayt = {**{a: m for f, (h, _) in t["basliklar"].items() for a, m in h.items() if a != "__metadata__"},
            **yeni_bas}
    meta["total_size"] = sum(bayt[a]["data_offsets"][1] - bayt[a]["data_offsets"][0] for a in wm)
    meta["harem_sidecar"] = {
        "surum": SURUM, "arac_sha256": arac_sha(), "kaynak": str(kaynak),
        "kaynak_index_sha256": hashlib.sha256((kaynak / INDEX).read_bytes()).hexdigest(),
        "kaynak_qcfg_sha256": hashlib.sha256((kaynak / QCFG).read_bytes()).hexdigest(),
        "bolunen_kda": len(plan["qkv"]), "eklenen": len(yeni_wm), "dusurulen": dusen,
        "tarih": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }
    (gecici / INDEX).write_text(json.dumps({"metadata": meta, "weight_map": wm}, indent=2))
    (gecici / QCFG).write_text(json.dumps(qcfg_yeni(kaynak, plan, yeni_bas)))
    if cikti.exists():
        cikti.rmdir()  # it was checked above that it is empty
    os.replace(gecici, cikti)
    r = dogrula(kaynak, cikti)
    r.update(durum="kuruldu", sure_sn=round(time.time() - t0, 1))
    return r


def dogrula(kaynak: Path, cikti: Path) -> dict:
    """At byte level: every remaining tensor is in the original file with the same header,
    every split tensor is identical to the slice of the original fused tensor, and the
    dropped ones are declared."""
    kaynak, cikti = kaynak.resolve(), cikti.absolute()
    t = kaynak_tara(kaynak)
    plan = plan_cikar(t)
    idx = json.loads((cikti / INDEX).read_text())
    meta = idx["metadata"]["harem_sidecar"]
    wm = idx["weight_map"]
    sorun = []
    dusen = set(meta["dusurulen"])
    beklenen_dusen = set(a for d in plan["qkv"].values() for a in d.values()) \
        | set(plan["conv"].values()) | set(plan["vis_eski"])
    if dusen != beklenen_dusen:
        sorun.append(f"dusurulen listesi plana uymuyor ({len(dusen)} vs {len(beklenen_dusen)})")
    for ad, f in t["wm"].items():
        if ad in dusen:
            if ad in wm:
                sorun.append(f"{ad} düşürülmüş ama dizinde")
            continue
        if wm.get(ad) != f:
            sorun.append(f"{ad}: dizin {wm.get(ad)} diyor, orijinal {f}")
    bas_c = {}
    for f in sorted(set(wm.values())):
        p = cikti / f
        if f.startswith("harem-bolme-"):
            if p.is_symlink():
                sorun.append(f"{f} symlink olmamalı")
        elif not (p.is_symlink() and p.resolve() == (kaynak / f).resolve()):
            sorun.append(f"{f} orijinale symlink değil")
        bas_c[f] = baslik_oku(p)
    for ad, f in wm.items():
        if ad not in bas_c[f][0]:
            sorun.append(f"{ad} {f} başlığında yok")
    # split tensors: byte for byte against the slices of the original fused tensor
    yeni = {x[0]: x for x in bolunmus_tensorler(kaynak, t, plan)}
    if set(yeni) != {a for a, f in wm.items() if f.startswith("harem-bolme-")}:
        sorun.append("bölme dosyalarındaki ad kümesi beklenen değil")
    for ad, (_, dt, sh, veri) in yeni.items():
        f = wm.get(ad)
        if f is None:
            continue
        h, vb = bas_c[f]
        m = h[ad]
        if m["dtype"] != dt or m["shape"] != list(sh):
            sorun.append(f"{ad}: başlık {m['dtype']} {m['shape']} != {dt} {sh}")
        elif hashlib.sha256(tensor_oku(cikti / f, vb, m["data_offsets"])).digest() \
                != hashlib.sha256(veri).digest():
            sorun.append(f"{ad}: baytlar orijinal dilimden farklı")
    if (cikti / "config.json").is_symlink() or \
            json.loads((cikti / "config.json").read_text()) != config_yeni(kaynak):
        sorun.append("config.json beklenen (satır içi eşlemeli) kopya değil")
    q = json.loads((cikti / QCFG).read_text())
    for pfx in plan["qkv"]:
        if f"{pfx}.qkv_proj" in q["tensor_storage"]:
            sorun.append(f"{QCFG}: {pfx}.qkv_proj hâlâ var")
        for adp in ("q_proj", "k_proj", "v_proj"):
            g = q["tensor_storage"].get(f"{pfx}.{adp}")
            tr = f"{pfx}.{adp}.trellis"
            if not g or g["stored_tensors"][tr]["shape"] != bas_c[wm[tr]][0][tr]["shape"]:
                sorun.append(f"{QCFG}: {pfx}.{adp} girdisi başlıkla uyuşmuyor")
    if q.get("packed_modules_mapping", {}).get("in_proj_qkvbfg_a") != PACKED["in_proj_qkvbfg_a"]:
        sorun.append(f"{QCFG}: packed_modules_mapping eksik")
    if sorun:
        raise Hata(4, "doğrulama: " + " | ".join(sorun[:8]) + (f" (+{len(sorun) - 8})" if len(sorun) > 8 else ""))
    return {"kda_katman": len(plan["qkv"]), "bolunen_tensor": len(yeni), "dusurulen": len(dusen),
            "gorsel_olu": len(plan["vis_eski"]), "dizin_tensor": len(wm),
            "bolme_dosya": sorted({f for f in wm.values() if f.startswith("harem-bolme-")})}


# ---------------------------------------------------------------- taslak (draft)

def harem_tp3_yukle(yol: str | None):
    if yol:
        import importlib.util

        spec = importlib.util.spec_from_file_location("_harem_tp3_arac", yol)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    from cuda_exl3 import _harem_tp3  # the installed plugin

    return _harem_tp3


def taslak_satirlar(kaynak: Path) -> tuple[int, int]:
    """The draft checkpoint's q_proj / k_proj row counts (from the header; the same in every layer)."""
    dosyalar = sorted(kaynak.glob("*.safetensors"))
    if (kaynak / INDEX).is_file():
        dosyalar = [kaynak / f for f in sorted(set(json.loads((kaynak / INDEX).read_text())["weight_map"].values()))]
    q, kv = set(), set()
    for f in dosyalar:
        for ad, m in baslik_oku(f)[0].items():
            if ad.endswith("self_attn.q_proj.weight"):
                q.add(m["shape"][0])
            elif ad.endswith("self_attn.k_proj.weight"):
                kv.add(m["shape"][0])
    if len(q) != 1 or len(kv) != 1:
        raise Hata(2, f"{kaynak}: q_proj/k_proj satır sayıları tek değil ({sorted(q)}, {sorted(kv)})")
    return q.pop(), kv.pop()


def taslak_kur(kaynak: Path, cikti: Path, tp: int, yol: str | None,
               kaynak_config: str | None = None) -> dict:
    from types import SimpleNamespace

    kaynak, cikti = kaynak.resolve(), cikti.absolute()
    h = harem_tp3_yukle(yol)
    cfg_yol = Path(kaynak_config) if kaynak_config else kaynak / "config.json"
    cfg = json.loads(cfg_yol.read_text())
    if "harem_tp_pad" in cfg:
        raise Hata(2, f"{cfg_yol} zaten etiketli; kaynak dolgusuz taslak olmalı")
    q_rows, kv_rows = taslak_satirlar(kaynak)
    hd = int(cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"])
    if (cfg["num_attention_heads"] * hd, cfg["num_key_value_heads"] * hd) != (q_rows, kv_rows):
        raise Hata(2, f"{cfg_yol}: config {cfg['num_attention_heads']}/{cfg['num_key_value_heads']} "
                      f"baş diyor, checkpoint {q_rows // hd}/{kv_rows // hd} taşıyor. Kaynak config "
                      "dolgulu görünüyor; orijinali --kaynak-config ile verin.")
    ns = SimpleNamespace(**cfg)
    plan = h.draft_plan(ns, tp)
    h.apply_draft_plan(ns, plan, tp)
    yeni = dict(cfg)
    for k in list(plan) + ["harem_tp_pad"]:
        yeni[k] = getattr(ns, k)
    metin = json.dumps(yeni, indent=2) + "\n"
    if cikti.exists():
        mevcut = (cikti / "config.json").read_text() if (cikti / "config.json").is_file() else None
        if mevcut == metin:
            return {"durum": "zaten-var", "plan": plan, "etiket": yeni["harem_tp_pad"]}
        raise Hata(3, f"{cikti} var ve içeriği bu planla kurulmuş değil; dokunulmadı")
    gecici = cikti.with_name(cikti.name + ".kuruluyor")
    gecici.mkdir(parents=True)
    for f in sorted(os.listdir(kaynak)):
        if f == "config.json" or f.startswith("."):
            continue
        os.symlink(os.path.relpath(kaynak / f, cikti), gecici / f)
    (gecici / "config.json").write_text(metin)
    if taslak_satirlar(gecici) != (q_rows, kv_rows):
        raise Hata(4, "taslak sidecar ağırlıkları kaynaktan farklı görünüyor")
    os.replace(gecici, cikti)
    return {"durum": "kuruldu", "plan": plan, "etiket": yeni["harem_tp_pad"]}


# ---------------------------------------------------------------- kesik (truncated; single-node test mode)

KATMAN_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.")
KESIK_LISTE = ("layer_types", "mlp_layer_types", "indexer_types")


def _katman_no(ad: str):
    m = KATMAN_RE.match(ad)
    return int(m.group(1)) if m else None


def kesik_config(c: dict, n: int, kaynak: str) -> dict:
    """config.json cut down to the first n layers (the inline quantization_config unchanged)."""
    c = json.loads(json.dumps(c))
    t = c["text_config"]
    eski_n = int(t["num_hidden_layers"])
    if not 1 <= n < eski_n:
        raise Hata(2, f"--katman {n}: 1..{eski_n - 1} olmalı (kaynak {eski_n} katman)")
    t["num_hidden_layers"] = n
    for k in KESIK_LISTE:
        if k in t:
            if len(t[k]) != eski_n:
                raise Hata(2, f"text_config.{k} uzunluğu {len(t[k])} != {eski_n}")
            t[k] = t[k][:n]
    la = t.get("linear_attn_config") or {}
    for k in ("kda_layers", "full_attn_layers"):
        if k in la:
            la[k] = [i for i in la[k] if i < n]
    c["harem_kesik"] = {"katman": n, "kaynak_katman": eski_n, "kaynak": kaynak,
                        "not": "TEST MODELI: gercek agirliklar, ilk N katman"}
    return c


def kesik_qcfg(q: dict, tut: set) -> dict:
    """Reduce tensor_storage to the modules of the kept tensors only (the mapping unchanged)."""
    q = json.loads(json.dumps(q))
    ts = {}
    for mod, g in q["tensor_storage"].items():
        st = g.get("stored_tensors") or {}
        if st:
            v = [a in tut for a in st]
            if all(v):
                ts[mod] = g
            elif any(v):
                raise Hata(2, f"{QCFG}: {mod} tensörleri kısmen tutuluyor")
        elif any(a.startswith(mod + ".") for a in tut):
            ts[mod] = g
    q["tensor_storage"] = ts
    return q


def _kesik_plan(kaynak: Path, n: int) -> dict:
    idx = json.loads((kaynak / INDEX).read_text())
    meta = (idx.get("metadata") or {}).get("harem_sidecar")
    if not meta or meta.get("kip", "hedef") != "hedef":
        raise Hata(2, f"{kaynak}: HAREM hedef sidecar'ı değil (metadata.harem_sidecar yok ya da kesik)")
    wm = idx["weight_map"]
    # A language-model layer whose prefix is not recognised must not be kept wholesale as
    # "layerless": every name carrying '.layers.<i>.' must match KATMAN_RE.
    garip = [a for a in wm if _katman_no(a) is None and re.search(r"(^|\.)layers\.\d+\.", a)]
    if garip:
        raise Hata(2, f"katman öneki tanınmayan tensör(ler): {garip[:3]}")
    tut = {a for a in wm if (_katman_no(a) is None or _katman_no(a) < n)}
    katmanlar = sorted({_katman_no(a) for a in tut if _katman_no(a) is not None})
    if not katmanlar:
        raise Hata(2, f"katman < {n} tensörü yok")
    dosyalar = sorted({wm[a] for a in tut})
    basliklar = {f: baslik_oku(kaynak / f) for f in dosyalar}
    dosyadaki = {}
    for f, (h, _) in basliklar.items():
        for a in h:
            if a != "__metadata__":
                if a in dosyadaki:
                    raise Hata(2, f"{a} iki parçada birden: {dosyadaki[a]}, {f}")
                dosyadaki[a] = f
    for a in tut:
        if dosyadaki.get(a) != wm[a]:
            raise Hata(2, f"{a}: dizin {wm[a]} diyor, başlıklarda {dosyadaki.get(a)}")
    dusen = sorted(set(dosyadaki) - tut)
    return {"idx": idx, "meta": meta, "wm": wm, "tut": tut, "dosyalar": dosyalar,
            "basliklar": basliklar, "dusen": dusen, "katmanlar": katmanlar}


def kesik_kur(kaynak: Path, cikti: Path, n: int) -> dict:
    kaynak, cikti = kaynak.resolve(), cikti.absolute()
    p = _kesik_plan(kaynak, n)
    cfg = kesik_config(json.loads((kaynak / "config.json").read_text()), n, str(kaynak))
    q = kesik_qcfg(json.loads((kaynak / QCFG).read_text()), p["tut"])
    if cikti.exists():
        idx = cikti / INDEX
        m = json.loads(idx.read_text()).get("metadata", {}).get("harem_sidecar") if idx.is_file() else None
        if m and m.get("kip") == "kesik" and m.get("kaynak") == str(kaynak) \
                and m.get("arac_sha256") == arac_sha() and m.get("katman") == n:
            r = kesik_dogrula(kaynak, cikti)
            r["durum"] = "zaten-var (dogrulandi)"
            return r
        if any(cikti.iterdir()):
            raise Hata(3, f"{cikti} var ve bu araçla bu kaynaktan kurulmuş bir kesik sidecar değil; dokunulmadı")
    gecici = cikti.with_name(cikti.name + ".kuruluyor")
    if gecici.exists():
        raise Hata(3, f"{gecici} var (yarım kurulum?); elle bakın")
    gecici.mkdir(parents=True)
    t0 = time.time()
    agirlik = set(p["dosyalar"])
    tum_agirlik = set(p["wm"].values())
    for f in sorted(os.listdir(kaynak)):
        if f in (INDEX, QCFG, "config.json") or f.startswith("."):
            continue
        if f in tum_agirlik and f not in agirlik:
            continue  # an unreferenced weight shard: not linked (the loader does not see it)
        if f.endswith(".safetensors") and f not in agirlik:
            continue  # a shard outside the index (e.g. mtp.safetensors): not linked
        os.symlink(os.path.relpath(kaynak / f, cikti), gecici / f)
    (gecici / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    (gecici / QCFG).write_text(json.dumps(q))
    bayt = {a: m for f, (h, _) in p["basliklar"].items() for a, m in h.items() if a != "__metadata__"}
    meta = dict(p["idx"].get("metadata") or {})
    meta.pop("harem_sidecar", None)
    meta["total_size"] = sum(bayt[a]["data_offsets"][1] - bayt[a]["data_offsets"][0] for a in p["tut"])
    katman_dusen = sum(1 for a in p["dusen"] if _katman_no(a) is not None and _katman_no(a) >= n)
    meta["harem_sidecar"] = {
        "surum": SURUM, "kip": "kesik", "katman": n, "arac_sha256": arac_sha(), "kaynak": str(kaynak),
        "kaynak_index_sha256": hashlib.sha256((kaynak / INDEX).read_bytes()).hexdigest(),
        "kaynak_kok": p["meta"].get("kaynak"),
        # harem-prelude.sh reads this at boot ("bolunen_kda"): the split KDA layers kept in the truncated sidecar
        "bolunen_kda": len({_katman_no(a) for a in p["tut"]
                            if a.endswith(".self_attn.q_proj.trellis") and p["wm"][a].startswith("harem-bolme-")}),
        "dusurulen": p["dusen"], "dusurulen_katman_ge_n": katman_dusen,
        "dusurulen_hedefin": len(p["dusen"]) - katman_dusen,
        "tarih": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }
    wm = {a: p["wm"][a] for a in sorted(p["tut"])}
    (gecici / INDEX).write_text(json.dumps({"metadata": meta, "weight_map": wm}, indent=2))
    if cikti.exists():
        cikti.rmdir()
    os.replace(gecici, cikti)
    r = kesik_dogrula(kaynak, cikti)
    r.update(durum="kuruldu", sure_sn=round(time.time() - t0, 1))
    return r


def kesik_dogrula(kaynak: Path, cikti: Path) -> dict:
    """Truncated sidecar: the index = the source's tensors of layer < N plus its layerless tensors
    (same file, same header); dropped = EVERY other tensor in the referenced shards; the
    symlinks point at the source; config/qcfg are the truncated copies; no unreferenced
    weight shard is linked."""
    kaynak, cikti = kaynak.resolve(), cikti.absolute()
    idx = json.loads((cikti / INDEX).read_text())
    meta = idx["metadata"]["harem_sidecar"]
    if meta.get("kip") != "kesik":
        raise Hata(4, "kesik sidecar değil")
    n = int(meta["katman"])
    p = _kesik_plan(kaynak, n)
    sorun = []
    wm = idx["weight_map"]
    if set(wm) != p["tut"]:
        sorun.append(f"dizin tensör kümesi beklenen değil ({len(wm)} vs {len(p['tut'])})")
    for a, f in wm.items():
        if p["wm"].get(a) != f:
            sorun.append(f"{a}: dizin {f}, kaynak {p['wm'].get(a)}")
        k = _katman_no(a)
        if k is not None and k >= n:
            sorun.append(f"{a}: katman {k} >= {n} dizinde")
    if sorted(meta["dusurulen"]) != p["dusen"]:
        sorun.append(f"dusurulen listesi beklenen değil ({len(meta['dusurulen'])} vs {len(p['dusen'])})")
    if set(meta["dusurulen"]) & set(wm):
        sorun.append("dusurulen ile dizin kesişiyor")
    goruldu = set()
    for f in sorted(set(wm.values())):
        pth = cikti / f
        if not (pth.is_symlink() and pth.resolve() == (kaynak / f).resolve()):
            sorun.append(f"{f} kaynağa symlink değil")
            continue
        h, _ = baslik_oku(pth)
        hk, _ = p["basliklar"][f]
        if h != hk:
            sorun.append(f"{f}: başlık kaynaktakinden farklı")
        goruldu |= {a for a in h if a != "__metadata__"}
    if goruldu != set(wm) | set(meta["dusurulen"]):
        sorun.append("parçalardaki tensörler = dizin + düşürülen değil")
    fazla = [f for f in os.listdir(cikti) if f.endswith(".safetensors") and f not in set(wm.values())]
    if fazla:
        sorun.append(f"başvurulmayan ağırlık parçası bağlı: {fazla[:3]}")
    cfg_bek = kesik_config(json.loads((kaynak / "config.json").read_text()), n, str(kaynak))
    cfg = json.loads((cikti / "config.json").read_text())
    if (cikti / "config.json").is_symlink() or cfg != cfg_bek:
        sorun.append("config.json beklenen kesik kopya değil")
    if json.loads((cikti / QCFG).read_text()) != kesik_qcfg(json.loads((kaynak / QCFG).read_text()), p["tut"]):
        sorun.append(f"{QCFG} beklenen kesik kopya değil")
    t = cfg["text_config"]
    if t["num_hidden_layers"] != n or any(len(t[k]) != n for k in KESIK_LISTE if k in t):
        sorun.append("config katman alanları N ile tutarsız")
    if sorun:
        raise Hata(4, "kesik doğrulama: " + " | ".join(sorun[:8]) + (f" (+{len(sorun) - 8})" if len(sorun) > 8 else ""))
    say = {}
    for a in wm:
        k = _katman_no(a)
        say[str(k) if k is not None else "katmansiz"] = say.get(str(k) if k is not None else "katmansiz", 0) + 1
    if meta.get("bolunen_kda") != len({_katman_no(a) for a in wm if a.endswith(".self_attn.q_proj.trellis")
                                       and wm[a].startswith("harem-bolme-")}):
        raise Hata(4, "kesik doğrulama: bolunen_kda sayısı tutarsız")
    return {"katman": n, "tutulan_katmanlar": p["katmanlar"], "bolunen_kda": meta["bolunen_kda"],
            "dizin_tensor": len(wm),
            "dusurulen": len(meta["dusurulen"]),
            "dusurulen_katman_ge_n": meta["dusurulen_katman_ge_n"],
            "dusurulen_hedefin": meta["dusurulen_hedefin"], "parcalar": sorted(set(wm.values())),
            "katman_basina_tensor": say,
            "layer_types": t.get("layer_types"), "mlp_layer_types": t.get("mlp_layer_types"),
            "kda_layers": t["linear_attn_config"].get("kda_layers"),
            "full_attn_layers": t["linear_attn_config"].get("full_attn_layers"),
            "toplam_bayt": idx["metadata"]["total_size"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("kip", choices=["hedef", "taslak", "dogrula", "kesik", "kesik-dogrula"])
    ap.add_argument("--kaynak", required=True)
    ap.add_argument("--cikti", required=True)
    ap.add_argument("--tp", type=int, default=3)
    ap.add_argument("--katman", type=int, default=None, help="kesik: tutulacak katman sayısı")
    ap.add_argument("--harem-tp3", default=None, help="cuda_exl3/_harem_tp3.py yolu (kurulu değilse)")
    ap.add_argument("--kaynak-config", default=None, help="taslak: dolgusuz config.json (varsayılan kaynak/config.json)")
    a = ap.parse_args()
    try:
        if a.kip == "hedef":
            r = hedef_kur(Path(a.kaynak), Path(a.cikti))
        elif a.kip == "dogrula":
            r = dogrula(Path(a.kaynak), Path(a.cikti))
            r["durum"] = "dogrulandi"
        elif a.kip == "kesik":
            if not a.katman:
                raise Hata(2, "kesik: --katman N gerekli")
            r = kesik_kur(Path(a.kaynak), Path(a.cikti), a.katman)
        elif a.kip == "kesik-dogrula":
            r = kesik_dogrula(Path(a.kaynak), Path(a.cikti))
            r["durum"] = "dogrulandi"
        else:
            r = taslak_kur(Path(a.kaynak), Path(a.cikti), a.tp, a.harem_tp3, a.kaynak_config)
    except Hata as e:
        print(f"[sidecar_kur] HATA ({e.kod}): {e}", file=sys.stderr)
        return e.kod
    print(json.dumps(r, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
