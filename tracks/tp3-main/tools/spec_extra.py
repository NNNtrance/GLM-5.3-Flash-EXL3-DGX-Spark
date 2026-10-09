#!/usr/bin/env python3
"""HAREM draft extras -- a generic extension of --speculative-config.

TASLAK_EK_B64 = base64(JSON object). The launcher merges it into the SPEC dictionary:
  * on a clashing key the EXTRA wins (the overridden keys are logged);
  * ``method`` and ``model`` CANNOT be in the extra (even with the same value) -> REFUSE
    (fail-closed);
  * if the knob is empty or absent SPEC is not touched at all (byte for byte the same; the
    launcher does not call this script then).
Why base64: an ssh "..." line carries only [A-Za-z0-9+/=]; quotes, brackets, commas and spaces
do not survive.
Users: A3 (num_speculative_tokens_per_batch_size), B1 (disable_eagle_block_drop, upstream
#53388), later ones.

Subcommands (exit 0 = ok, 3 = REFUSE, 2 = usage):
  kodla '<JSON object>'            stdout: base64 (compact JSON, key order kept)
  coz <b64>                        stdout: JSON (checked)
  denetle <b64> [--nst N]          checks the EXTRA on its own (the launcher's early gate; N = the
                                   launcher's num_speculative_tokens, default 7 = the SPEC line of
                                   the launcher); stdout: JSON
  birlestir '<SPEC JSON>' <b64>    stdout: the merged SPEC (compact JSON); stderr: one note line

Known-key checks (an unknown key passes unchanged; vLLM validates it itself at boot):
  num_speculative_tokens_per_batch_size : the vLLM 21d93d0d8 rules (v1/spec_decode/dynamic/utils.py:7-74:
      a non-empty list, triples, start/end >= 1, start <= end, K >= 0, non-overlapping, the first
      range starts at 1) + HAREM: integers only (no bool or fraction; vLLM silently rounds with
      int()), K <= num_speculative_tokens (vLLM silently clips, utils.py:113-126 -> REFUSE here).
      K=0: passes but WARNS (#53426: at a K=0 step the draft still runs and its cost is paid).
  num_speculative_tokens                : integer >= 1 (WARNS: the block the draft sees changes; it is not lowered in the production schedule)
  disable_eagle_block_drop              : bool
"""
import base64
import binascii
import json
import sys

YASAK = ("method", "model")
B64_ALFABE = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=")


class Red(Exception):
    pass


def _tamsayi(x):
    return isinstance(x, int) and not isinstance(x, bool)


def cizelge_denetle(cz, nst):
    """vLLM rules + HAREM rules. Returns: (normalised schedule, warnings)."""
    uyar = []
    if not isinstance(cz, list) or not cz:
        raise Red("num_speculative_tokens_per_batch_size bos olmayan bir liste olmali")
    girdiler = []
    for g in cz:
        if not isinstance(g, list) or len(g) != 3:
            raise Red(f"cizelge girdisi [bas, son, K] olmali: {g!r}")
        if not all(_tamsayi(v) for v in g):
            raise Red(f"cizelge girdisi yalniz tamsayi tasir (bool/kesir yok): {g!r}")
        bas, son, k = g
        if bas <= 0 or son <= 0:
            raise Red(f"aralik pozitif olmali: {g!r}")
        if bas > son:
            raise Red(f"aralik baslangici bitisten buyuk: {g!r}")
        if k < 0:
            raise Red(f"K >= 0 olmali: {g!r}")
        if k > nst:
            raise Red(f"K={k} > num_speculative_tokens={nst}: vLLM sessizce kirpardi -> RED: {g!r}")
        if k == 0:
            uyar.append(f"K=0 araligi {g!r}: taslak yine kosar (#53426), dogrulama yok")
        girdiler.append((bas, son, k))
    girdiler.sort(key=lambda e: e[0])
    onceki = 0
    for bas, son, _ in girdiler:
        if bas <= onceki:
            raise Red(f"araliklar cakismasiz olmali: {cz!r}")
        onceki = son
    if girdiler[0][0] != 1:
        raise Red(f"ilk aralik 1'den baslamali: {cz!r}")
    return [list(e) for e in girdiler], uyar


def ek_coz(b64):
    if not b64 or set(b64) - B64_ALFABE:
        raise Red("TASLAK_EK_B64 base64 alfabesi disinda karakter tasiyor ya da bos")
    try:
        ham = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError) as e:
        raise Red(f"base64 cozulemedi: {e}") from None
    try:
        ek = json.loads(ham.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise Red(f"EK gecerli JSON degil: {e}") from None
    if not isinstance(ek, dict) or not ek:
        raise Red(f"EK bos olmayan bir JSON NESNESI olmali: {ek!r}")
    yasak = [k for k in YASAK if k in ek]
    if yasak:
        raise Red(f"EK {yasak} anahtar(lar)ini tasiyamaz (method/model degistirilemez)")
    return ek


def anahtar_denetle(spec):
    """Known keys on the merged (or default) SPEC. Returns: warnings."""
    uyar = []
    nst = spec.get("num_speculative_tokens")
    if not _tamsayi(nst) or nst < 1:
        raise Red(f"num_speculative_tokens tamsayi >= 1 olmali: {nst!r}")
    if "disable_eagle_block_drop" in spec and not isinstance(spec["disable_eagle_block_drop"], bool):
        raise Red(f"disable_eagle_block_drop bool olmali: {spec['disable_eagle_block_drop']!r}")
    cz = spec.get("num_speculative_tokens_per_batch_size")
    if cz is not None:
        normal, u = cizelge_denetle(cz, nst)
        spec["num_speculative_tokens_per_batch_size"] = normal
        uyar += u
    return uyar


def kodla(json_metin):
    try:
        ek = json.loads(json_metin)
    except json.JSONDecodeError as e:
        raise Red(f"gecerli JSON degil: {e}") from None
    b64 = base64.b64encode(json.dumps(ek, separators=(",", ":")).encode()).decode()
    ek_coz(b64)  # the same rules
    return b64


def denetle(b64, nst=7):
    ek = ek_coz(b64)
    spec = {"num_speculative_tokens": nst}
    spec.update(ek)
    uyar = anahtar_denetle(spec)
    return ek, uyar


def birlestir(spec_metin, b64):
    spec = json.loads(spec_metin)
    if not isinstance(spec, dict):
        raise Red("SPEC bir JSON nesnesi degil")
    ek = ek_coz(b64)
    ezilen = [k for k in ek if k in spec and spec[k] != ek[k]]
    spec.update(ek)
    uyar = anahtar_denetle(spec)
    not_ = (f"[harem-taslak-ek] anahtarlar={list(ek)} ezilen={ezilen}"
            + (f" UYARI: {' | '.join(uyar)}" if uyar else ""))
    return json.dumps(spec, separators=(",", ":")), not_


def main(argv):
    if len(argv) < 2 or argv[0] in ("-h", "--help"):
        print(__doc__, file=sys.stderr)
        return 2
    try:
        if argv[0] == "kodla" and len(argv) == 2:
            print(kodla(argv[1]))
        elif argv[0] == "coz" and len(argv) == 2:
            print(json.dumps(ek_coz(argv[1]), separators=(",", ":")))
        elif argv[0] == "denetle" and len(argv) in (2, 4):
            nst = 7
            if len(argv) == 4:
                if argv[2] != "--nst":
                    print(__doc__, file=sys.stderr)
                    return 2
                nst = int(argv[3])
            ek, uyar = denetle(argv[1], nst)
            print(json.dumps(ek, separators=(",", ":")))
            for u in uyar:
                print(f"UYARI: {u}", file=sys.stderr)
        elif argv[0] == "birlestir" and len(argv) == 3:
            spec, not_ = birlestir(argv[1], argv[2])
            print(spec)
            print(not_, file=sys.stderr)
        else:
            print(__doc__, file=sys.stderr)
            return 2
    except Red as e:
        print(f"[harem-taslak-ek] RED: {e}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
