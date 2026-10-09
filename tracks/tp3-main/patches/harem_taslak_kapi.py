#!/usr/bin/env python3
"""HAREM draft-model tag gate (a boot gate; maintainers' decision of 7 October 2026).

With TP > 1 the speculative draft model's config.json must carry the ``harem_tp_pad``
tag, and the tag's ``tp`` must equal the TP the draft will be served at; otherwise the
boot is REFUSED (fail-closed). Reason: given an untagged but already padded draft (an
older sidecar, 36/9), the loader still pads with zeros, but ``yama-dflash-tp3``'s
zero-row proof and its log line are silently skipped; the previous production watchdog
checked this without a tag as well.

Why in the prelude (and not in yama-dflash-tp3): a ``HAREM_FASTLOAD_MODE=load`` boot
never calls ``load_weights`` (harem_fastload._restore), so a check inside the engine
would not run in that mode. The prelude runs at every boot, before the engine starts
and without touching the GPU.

The arguments are read with ``vllm serve``'s OWN parser: abbreviations
(``--tensor-parallel 3``), ``-tp``/``-sc``, the dotted form
(``--speculative-config.model X``), underscores and a ``--config`` yaml all resolve as
they do in vLLM. If parsing fails the gate refuses.

  * No draft (no speculation, or MTP without a separate model) -> passes.
  * Draft TP is 1 -> the tag is not required (upstream); if present, its tp must be 1.
  * Draft TP > 1 -> the tag is required and its tp must be equal; if the tag carries
    head counts, the head counts in the config must equal the padded values in the tag.

Use (harem-prelude.sh): python3 harem_taslak_kapi.py <vllm serve arguments...>
Exit 0 = passed, 3 = refused, 4 = the arguments could not be parsed (that refuses too).
"""

import json
import os
import sys

ETIKET = "[harem-taslak-kapi]"


def karar(tp, spec):
    """(passed, message). tp: the target TP; spec: the --speculative-config dict or None."""
    if not spec:
        return True, f"tp={tp} spekulatif taslak yok -> GECTI"
    if not isinstance(spec, dict):
        return False, f"speculative-config sozluk degil: {spec!r}"
    model = spec.get("model")
    if not model:
        return True, f"tp={tp} method={spec.get('method')} ayri taslak modeli yok -> GECTI"
    dtp_ham = spec.get("draft_tensor_parallel_size") or tp
    try:
        dtp = int(dtp_ham)
    except (TypeError, ValueError):
        return False, f"draft_tensor_parallel_size okunamadi: {dtp_ham!r}"
    cfgp = os.path.join(str(model), "config.json")
    if not os.path.isfile(cfgp):
        if dtp > 1:
            return False, f"taslak tp={dtp} ama yerel config.json yok: {cfgp} (etiket dogrulanamaz)"
        return True, f"taslak tp=1, yerel config yok ({model}) -> GECTI (upstream)"
    try:
        with open(cfgp) as f:
            cfg = json.load(f)
    except Exception as e:  # noqa: BLE001 -- a broken config = refuse
        return False, f"taslak config.json okunamadi: {cfgp}: {e!r}"
    if not isinstance(cfg, dict):
        return False, f"taslak config.json sozluk degil: {cfgp}"
    tag = cfg.get("harem_tp_pad")
    if tag is None:
        if dtp > 1:
            return False, (f"taslak tp={dtp} ama config harem_tp_pad etiketi TASIMIYOR: {cfgp} "
                           "(etiketli sidecar kullan: araclar/sidecar_kur.py taslak --tp N)")
        return True, f"taslak tp=1 etiketsiz ({model}) -> GECTI (upstream)"
    if not isinstance(tag, dict) or "tp" not in tag:
        return False, f"harem_tp_pad bicimi bozuk: {tag!r} ({cfgp})"
    try:
        ttp = int(tag["tp"])
    except (TypeError, ValueError):
        return False, f"harem_tp_pad.tp okunamadi: {tag!r} ({cfgp})"
    if ttp != dtp:
        return False, f"taslak tp={ttp} icin etiketli, sunulan taslak tp={dtp}: {cfgp}"
    for anahtar in ("num_attention_heads", "num_key_value_heads"):
        if anahtar in tag:
            deger = tag[anahtar]
            if not (isinstance(deger, list) and len(deger) == 2):
                return False, f"harem_tp_pad.{anahtar} bicimi bozuk: {deger!r} ({cfgp})"
            if cfg.get(anahtar) != deger[1]:
                return False, (f"config {anahtar}={cfg.get(anahtar)} etiketteki dolgulu "
                               f"degerle ({deger[1]}) tutmuyor: {cfgp}")
    return True, (f"taslak tp={dtp} {model} harem_tp_pad="
                  f"{json.dumps(tag, sort_keys=True, separators=(',', ':'))} -> GECTI")


def tp_ve_spec(argv):
    """Reads ``vllm serve <argv>`` with vLLM's own parser: (tp, speculative_config)."""
    from vllm import platforms

    if platforms.current_platform.is_unspecified():
        # Only in a GPU-less test container: switch to the CPU platform, as vLLM's own
        # `vllm bench` path does; parsing does not depend on the platform (needed for the
        # DeviceConfig default).
        from vllm.platforms.cpu import CpuPlatform

        platforms.current_platform = CpuPlatform()
    import vllm.entrypoints.cli.serve as serve
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    p = FlexibleArgumentParser(prog="vllm")
    sp = p.add_subparsers(dest="subparser")
    for c in serve.cmd_init():
        c.subparser_init(sp)
    a = p.parse_args(["serve", *argv])
    return int(a.tensor_parallel_size), a.speculative_config


def main(argv):
    try:
        tp, spec = tp_ve_spec(argv)
    except SystemExit as e:
        print(f"{ETIKET} RED: vllm serve argumanlari ayristirilamadi (cikis {e.code})", flush=True)
        return 4
    except Exception as e:  # noqa: BLE001
        print(f"{ETIKET} RED: vllm serve argumanlari ayristirilamadi: {e!r}", flush=True)
        return 4
    gecti, mesaj = karar(tp, spec)
    print(f"{ETIKET} {'' if gecti else 'RED: '}{mesaj}", flush=True)
    return 0 if gecti else 3


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
