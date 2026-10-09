#!/usr/bin/env python3
"""HAREM anchored-patch engine (vLLM 21d93d0d8).

Every ``yama-*.py`` script defines a :class:`Yama` and calls ``main(YAMA)``:

    yama-X.py --root <vllm package root> [--check | --apply | --revert | --durum]

``--check`` is the default (it writes nothing); ``--durum`` prints the state as JSON.
Rules (fail-closed):

* Every edit (``Duzen``) is defined by an exact-text anchor. The anchor must occur
  EXACTLY ONCE in the pristine file, the new text must occur EXACTLY ONCE in an
  applied file, and the new text must carry the "HAREM" marker.
* The base sha256 of every file is pinned (21d93d0d8). When the known edits of ALL
  patches that touch a file are undone in memory, the base sha must come out; if it
  does not, the file was changed in a way we do not know -> exit 2, nothing is
  written. This lets several patches touch one file (model.py: full scope + vision)
  and be applied and reverted independently of each other.
* ``--apply`` is idempotent: an applied edit is skipped, a second run changes not a
  single byte. A half-applied patch is completed.
* ``--revert`` undoes only this patch's edits; the result must again equal the base
  sha (with the other patches taken out).
* Writes are atomic (temporary file + os.replace, permissions kept); a Python file is
  compiled before it is written.

Exit codes: 0 ok - 2 file missing / base differs - 3 anchor missing or more than one -
4 partial or inconsistent state - 5 does not compile - 6 final check - 7 definition error.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

ISARET = "HAREM"
SURUM = "1.0"


@dataclass(frozen=True)
class Duzen:
    etiket: str
    dosya: str  # relative to the vllm package root
    capa: str
    yeni: str


@dataclass(frozen=True)
class Kopya:
    kaynak: str  # relative to the patches directory (a new file; not in vLLM)
    hedef: str  # relative to the vllm package root


@dataclass
class Yama:
    ad: str
    aciklama: str
    duzenler: list
    taban: dict  # file -> sha256 (pristine 21d93d0d8)
    kopyalar: list = field(default_factory=list)


class YamaHatasi(Exception):
    def __init__(self, kod: int, mesaj: str):
        super().__init__(mesaj)
        self.kod = kod


def sha256(metin: str) -> str:
    return hashlib.sha256(metin.encode("utf-8")).hexdigest()


def _oku(p: Path) -> str:
    with open(p, encoding="utf-8", newline="") as f:
        return f.read()


def _yaz_atomik(p: Path, metin: str) -> None:
    mod = os.stat(p).st_mode & 0o7777 if p.exists() else 0o644
    tmp = p.with_name(f".{p.name}.harem-tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(metin)
    os.chmod(tmp, mod)
    os.replace(tmp, p)


def _derle(metin: str, ad: str) -> None:
    try:
        compile(metin, ad, "exec")
    except SyntaxError as e:
        raise YamaHatasi(5, f"{ad}: yamalı kaynak derlenmiyor: {e}") from None


def kardes_yamalar(dizin: Path) -> dict:
    """All yama-*.py definitions in the same directory (name -> Yama)."""
    out = {}
    for p in sorted(dizin.glob("yama-*.py")):
        spec = importlib.util.spec_from_file_location(f"_harem_yama_{p.stem.replace('-', '_')}", p)
        mod = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(dizin))
        try:
            spec.loader.exec_module(mod)
        finally:
            sys.path.pop(0)
        y = getattr(mod, "YAMA", None)
        if not isinstance(y, Yama):
            raise YamaHatasi(7, f"{p.name}: YAMA tanımı yok")
        if y.ad in out:
            raise YamaHatasi(7, f"aynı ad iki kez: {y.ad}")
        out[y.ad] = y
    return out


def tanim_denetle(yamalar: dict) -> None:
    """The definitions' own consistency (without looking at any file)."""
    taban: dict = {}
    for y in yamalar.values():
        for d in y.duzenler:
            if ISARET not in d.yeni:
                raise YamaHatasi(7, f"{y.ad}/{d.etiket}: yeni metinde {ISARET} işareti yok")
            if d.capa == d.yeni or not d.capa:
                raise YamaHatasi(7, f"{y.ad}/{d.etiket}: boş ya da değişmeyen düzenleme")
            if d.dosya not in y.taban:
                raise YamaHatasi(7, f"{y.ad}/{d.etiket}: {d.dosya} için taban sha yok")
        for dosya, sha in y.taban.items():
            if taban.setdefault(dosya, sha) != sha:
                raise YamaHatasi(7, f"{dosya}: yamalar farklı taban sha bildiriyor")


def _hepsi(yamalar: dict, dosya: str) -> list:
    return [d for y in yamalar.values() for d in y.duzenler if d.dosya == dosya]


def _sade(metin: str, duzenler: list) -> str:
    """Undo all known edits (back to the base text)."""
    for d in duzenler:
        n = metin.count(d.yeni)
        if n == 1:
            metin = metin.replace(d.yeni, d.capa, 1)
        elif n > 1:
            raise YamaHatasi(4, f"{d.dosya}/{d.etiket}: yeni metin {n} kez geçiyor")
    return metin


def _durumlar(metin: str, duzenler: list) -> dict:
    out = {}
    for d in duzenler:
        if metin.count(d.yeni) == 1:
            out[d.etiket] = "uygulanmis"
        else:
            n = metin.count(d.capa)
            out[d.etiket] = "hazir" if n == 1 else ("capa-yok" if n == 0 else f"capa-x{n}")
    return out


def calistir(yama: Yama, kok: Path, mod: str, dizin: Path) -> dict:
    yamalar = kardes_yamalar(dizin)
    if yamalar.get(yama.ad) is None:
        yamalar[yama.ad] = yama
    tanim_denetle(yamalar)
    if not kok.is_dir():
        raise YamaHatasi(2, f"vllm paketi yok: {kok}")
    rapor = {"yama": yama.ad, "mod": mod, "dosyalar": {}, "kopyalar": {}, "yazilan": []}

    dosyalar = sorted({d.dosya for d in yama.duzenler})
    metinler, durum = {}, {}
    for f in dosyalar:
        p = kok / f
        if not p.is_file():
            raise YamaHatasi(2, f"dosya yok: {p}")
        metin = _oku(p)
        saf = _sade(metin, _hepsi(yamalar, f))
        if sha256(saf) != yama.taban[f]:
            raise YamaHatasi(2, f"{f}: taban sha256 beklenen değil (bilinen yamalar çıkarılınca "
                                f"{sha256(saf)[:16]}, beklenen {yama.taban[f][:16]}). Ağaç 21d93d0d8 değil "
                                "ya da dosya bizim bilmediğimiz biçimde değişmiş; yazılmadı.")
        bunlar = [d for d in yama.duzenler if d.dosya == f]
        for d in bunlar:  # the anchor occurs exactly once in the pristine text
            n = saf.count(d.capa)
            if n != 1:
                raise YamaHatasi(3, f"{f}/{d.etiket}: çapa saf dosyada {n} kez (beklenen 1)")
        metinler[f] = metin
        durum[f] = _durumlar(metin, bunlar)
        bozuk = {k: v for k, v in durum[f].items() if v not in ("hazir", "uygulanmis")}
        if bozuk:
            raise YamaHatasi(3, f"{f}: çapa sorunu {bozuk}")
        rapor["dosyalar"][f] = dict(durum[f])

    for k in yama.kopyalar:
        src, dst = dizin / k.kaynak, kok / k.hedef
        if not src.is_file():
            raise YamaHatasi(2, f"kopya kaynağı yok: {src}")
        if not dst.exists():
            rapor["kopyalar"][k.hedef] = "hazir"
        elif sha256(_oku(dst)) == sha256(_oku(src)):
            rapor["kopyalar"][k.hedef] = "uygulanmis"
        else:
            raise YamaHatasi(4, f"{dst} var ama bizim kopyamız değil; üzerine yazılmadı")

    hepsi = [v for d in durum.values() for v in d.values()] + list(rapor["kopyalar"].values())
    if all(v == "uygulanmis" for v in hepsi):
        rapor["ozet"] = "uygulanmis"
    elif all(v == "hazir" for v in hepsi):
        rapor["ozet"] = "hazir"
    else:
        rapor["ozet"] = "kismi"

    if mod in ("check", "durum"):
        if rapor["ozet"] == "kismi" and mod == "check":
            raise YamaHatasi(4, f"{yama.ad}: yarım uygulanmış ({json.dumps(rapor['dosyalar'])}); "
                                "--apply tamamlar, --revert geri alır")
        return rapor

    for f in dosyalar:
        metin = metinler[f]
        bunlar = [d for d in yama.duzenler if d.dosya == f]
        yeni = metin
        for d in bunlar:
            if mod == "apply" and durum[f][d.etiket] == "hazir":
                if yeni.count(d.capa) != 1:
                    raise YamaHatasi(3, f"{f}/{d.etiket}: çapa uygulama sırasında tek değil")
                yeni = yeni.replace(d.capa, d.yeni, 1)
            elif mod == "revert" and durum[f][d.etiket] == "uygulanmis":
                yeni = yeni.replace(d.yeni, d.capa, 1)
        if yeni == metin:
            continue
        hedef_durum = "uygulanmis" if mod == "apply" else "hazir"
        son = _durumlar(yeni, bunlar)
        if any(v != hedef_durum for v in son.values()):
            raise YamaHatasi(6, f"{f}: son-kontrol {son} (beklenen hepsi {hedef_durum})")
        if sha256(_sade(yeni, _hepsi(yamalar, f))) != yama.taban[f]:
            raise YamaHatasi(6, f"{f}: son-kontrol: sonuç taban sha'ya geri döndürülemiyor")
        if f.endswith(".py"):
            _derle(yeni, f)
        _yaz_atomik(kok / f, yeni)
        rapor["yazilan"].append(f)
        rapor["dosyalar"][f] = son

    for k in yama.kopyalar:
        src, dst = dizin / k.kaynak, kok / k.hedef
        if mod == "apply" and rapor["kopyalar"][k.hedef] == "hazir":
            metin = _oku(src)
            if k.hedef.endswith(".py"):
                _derle(metin, k.hedef)
            dst.parent.mkdir(parents=True, exist_ok=True)
            with open(dst, "w", encoding="utf-8", newline="") as fh:
                fh.write(metin)
            os.chmod(dst, 0o644)
            rapor["yazilan"].append(k.hedef)
            rapor["kopyalar"][k.hedef] = "uygulanmis"
        elif mod == "revert" and rapor["kopyalar"][k.hedef] == "uygulanmis":
            dst.unlink()
            rapor["yazilan"].append(k.hedef)
            rapor["kopyalar"][k.hedef] = "hazir"
    rapor["ozet"] = "uygulanmis" if mod == "apply" else "hazir"
    return rapor


def main(yama: Yama, dosya: str, argv=None) -> int:
    """``dosya``: the calling patch script's ``__file__`` (sibling patches and copy
    sources are looked up in that directory)."""
    ap = argparse.ArgumentParser(description=f"{yama.ad}: {yama.aciklama}")
    ap.add_argument("--root", required=True, help="vllm paket kökü (.../dist-packages/vllm)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--check", action="store_const", dest="mod", const="check",
                   help="çapaları ve tabanı doğrula, yazma (varsayılan)")
    g.add_argument("--apply", action="store_const", dest="mod", const="apply")
    g.add_argument("--revert", action="store_const", dest="mod", const="revert")
    g.add_argument("--durum", action="store_const", dest="mod", const="durum",
                   help="durumu JSON yaz (sınav için), yazma")
    a = ap.parse_args(argv)
    mod = a.mod or "check"
    dizin = Path(dosya).resolve().parent
    try:
        r = calistir(yama, Path(a.root), mod, dizin)
    except YamaHatasi as e:
        print(f"[{yama.ad}] HATA ({e.kod}): {e}", file=sys.stderr)
        return e.kod
    if mod == "durum":
        print(json.dumps(r, ensure_ascii=False, sort_keys=True))
        return 0
    n = sum(len(v) for v in r["dosyalar"].values()) + len(r["kopyalar"])
    if mod == "check":
        print(f"[{yama.ad}] --check: {n} parça {r['ozet']}; taban sha tamam ({len(r['dosyalar'])} dosya)")
    elif r["yazilan"]:
        print(f"[{yama.ad}] --{mod}: {', '.join(r['yazilan'])} yazıldı -> {r['ozet']}")
    else:
        print(f"[{yama.ad}] --{mod}: değişiklik yok (zaten {r['ozet']})")
    return 0
