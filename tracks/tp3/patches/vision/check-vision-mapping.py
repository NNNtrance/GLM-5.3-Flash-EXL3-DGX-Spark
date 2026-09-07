#!/usr/bin/env python3
"""HAREM vision arm, gate 1 (model-free): does the packed mapping resolve the
whole EXL3 vision tower?

Runs on CPU, reads only safetensors HEADERS, never tensor data, and never
touches the engine.  It is the "motorsuz dogrulama" step of
02-donanim-model/VISION-TP3-PLAN.md section 8.1 and must pass before a rank is
started with .env.tp3.vision.

WHAT IT PROVES
  * The checkpoint really carries the vision tower as EXL3 (172 modules with a
    `.trellis`), not as bf16 dense.
  * cuda_exl3's `_augment_from_checkpoint()` recovers all of them even though
    `quantization_config.json`'s `tensor_storage` table names none of them.
  * `Exl3Config.resolve()` maps every one of vLLM's 99 vision linear prefixes
    onto its checkpoint tensors -- but ONLY with the UNION of the two packed
    mappings:

      upstream Glm4vForConditionalGeneration : qkv_proj -> q/k/v_proj   OK
                                               gate_up_proj -> gate_up_proj  X
      HAREM full-scope (patch-fullscope-tp3) : gate_up_proj -> gate/up   OK
                                               qkv_proj              (absent) X

    The full-scope mapping SHADOWS the inherited one (it is set on the same
    class attribute), so the missing half has to come back through
    CUDA_EXL3_PACKED_MAPPING, which cuda_exl3 merges UNDER the class mapping
    (config.py:112-125, :381-386) and therefore cannot clobber it.

WHY IT MATTERS.  `Exl3Config.get_quant_method` (config.py:458-470) FAILS OPEN:
a prefix that does not resolve silently becomes an UnquantizedLinearMethod.
The load then dies with "missing weight" -- loud, but four minutes in and with
a misleading message.  This gate says the same thing in three seconds.

Usage:
    check-vision-mapping.py --model /var/tmp/glm-5.3-flash-turboderp-4.05bpw-tp3
    check-vision-mapping.py --model ... --env-mapping '{"qkv_proj": [...]}'
"""

import argparse
import json
import os
import re
import struct
import sys

# The mapping Glm4vForConditionalGeneration declares upstream
# (glm4_1v.py:1762-1769).  `gate_up_proj -> ["gate_up_proj"]` is a self-map and
# resolves nothing.
UPSTREAM_MAPPING = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_up_proj"],
}

# What patch-fullscope-tp3.py's A2 puts on Glm5NextForConditionalGeneration
# when HAREM_EXL3_FULLSCOPE=1 (_HAREM_FS_PACKED_MAPPING).
FULLSCOPE_MAPPING = {
    "gate_up_proj": ["gate_proj", "up_proj"],
    "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
    "in_proj_qkv": ["qkv_proj"],
}

# The env supplement that turns the shadowed mapping back into the union.
# Kept here so the .env file and this gate cannot drift apart.
ENV_MAPPING = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}

DEPTH = 24  # vision_config.depth; re-read from config.json below


def vllm_vision_prefixes(depth: int) -> list[str]:
    """Every vision module vLLM builds with a quant_config.

    The attention fusion is named `qkv_proj` in the PREFIX (multimodal.py:150,
    `prefix=f"{prefix}.qkv_proj" if quant_config else f"{prefix}.qkv"`) even
    though the attribute stays `qkv` -- the prefix is what get_quant_method
    resolves, so that is the name this gate must use.
    """
    out = []
    for b in range(depth):
        out += [
            f"visual.blocks.{b}.attn.qkv_proj",
            f"visual.blocks.{b}.attn.proj",
            f"visual.blocks.{b}.mlp.gate_up_proj",
            f"visual.blocks.{b}.mlp.down_proj",
        ]
    out += [
        "visual.merger.proj",
        "visual.merger.gate_up_proj",
        "visual.merger.down_proj",
    ]
    return out


def header_vision_modules(model_dir: str) -> set[str]:
    """Checkpoint modules under model.visual.* that own a `.trellis`.

    Headers only: an 8-byte length then that many bytes of JSON.
    """
    index = os.path.join(model_dir, "model.safetensors.index.json")
    with open(index) as f:
        weight_map = json.load(f)["weight_map"]
    shards = sorted({v for k, v in weight_map.items() if ".visual." in k})
    mods = set()
    for shard in shards:
        path = os.path.join(model_dir, shard)
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            head = json.loads(f.read(n))
        for k in head:
            if k == "__metadata__" or ".visual." not in k:
                continue
            mod, _, leaf = k.rpartition(".")
            if leaf == "trellis":
                mods.add(mod)
    return mods


def build_config(model_dir: str, class_mapping: dict, env_mapping: dict | None):
    """Construct an Exl3Config exactly as the engine would.

    `_model_path_hint` is the class attribute cuda_exl3 sets from
    `hf_config._name_or_path` in override_quantization_method; setting it here
    is what arms `_augment_from_checkpoint()`.
    """
    from cuda_exl3.config import Exl3Config

    if env_mapping is None:
        os.environ.pop("CUDA_EXL3_PACKED_MAPPING", None)
    else:
        os.environ["CUDA_EXL3_PACKED_MAPPING"] = json.dumps(
            env_mapping, separators=(",", ":")
        )

    with open(os.path.join(model_dir, "quantization_config.json")) as f:
        full = json.load(f)

    Exl3Config._model_path_hint = model_dir
    cfg = Exl3Config(full)
    cfg.packed_modules_mapping = dict(class_mapping)
    return cfg


def run_arm(name: str, model_dir: str, class_mapping, env_mapping, prefixes):
    cfg = build_config(model_dir, class_mapping, env_mapping)
    resolved, missing, covered = 0, [], set()
    for p in prefixes:
        infos = cfg.resolve(p)
        if infos is None:
            missing.append(p)
        else:
            resolved += 1
            covered.update(i.name for i in infos)
    print(
        f"  {name:<28s} resolved {resolved:3d}/{len(prefixes)}  "
        f"checkpoint modules covered {len(covered):3d}"
    )
    if missing:
        shown = sorted({re.sub(r"\.blocks\.\d+\.", ".blocks.N.", m) for m in missing})
        print(f"      unresolved (deduped): {', '.join(shown)}")
    return resolved, covered, missing


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model dir (sidecar is fine)")
    ap.add_argument(
        "--env-mapping",
        default=json.dumps(ENV_MAPPING, separators=(",", ":")),
        help=(
            "CUDA_EXL3_PACKED_MAPPING value under test. The prelude passes the "
            "env file's own value so a typo there is caught here rather than "
            "four minutes into a load. An empty value is tested as {} and will "
            "fail -- which is the point."
        ),
    )
    a = ap.parse_args()

    if not os.path.isdir(a.model):
        print(f"[vision-map] FAIL: no such model dir {a.model}", file=sys.stderr)
        return 3

    with open(os.path.join(a.model, "config.json")) as f:
        depth = json.load(f)["vision_config"]["depth"]
    prefixes = vllm_vision_prefixes(depth)

    ck = header_vision_modules(a.model)
    print(f"[vision-map] vision_config.depth={depth} -> {len(prefixes)} vLLM prefixes")
    print(f"[vision-map] checkpoint EXL3 vision modules (headers): {len(ck)}")
    if len(ck) != 172:
        print(
            f"[vision-map] FAIL: expected 172 EXL3 vision modules, found {len(ck)}. "
            "This is not the turboderp full-scope checkpoint the arm was designed "
            "for; stop and re-measure before patching anything.",
            file=sys.stderr,
        )
        return 3

    env_mapping = json.loads(a.env_mapping) if a.env_mapping.strip() else {}
    print("[vision-map] resolve() by packed mapping:")
    run_arm("upstream Glm4v only", a.model, UPSTREAM_MAPPING, None, prefixes)
    run_arm("full-scope only", a.model, FULLSCOPE_MAPPING, None, prefixes)
    n, covered, missing = run_arm(
        "UNION (full-scope + env)", a.model, FULLSCOPE_MAPPING, env_mapping, prefixes
    )

    if missing:
        print(
            f"[vision-map] FAIL: {len(missing)} vision prefixes do not resolve "
            "under the union. get_quant_method fails OPEN, so these would "
            "silently become bf16 linears and die later on a missing weight.",
            file=sys.stderr,
        )
        return 3
    if covered != ck:
        only_ck = sorted(ck - covered)[:6]
        only_cov = sorted(covered - ck)[:6]
        print(
            f"[vision-map] FAIL: union covers {len(covered)} modules, headers "
            f"hold {len(ck)}. Only in checkpoint: {only_ck}. "
            f"Only in resolve: {only_cov}",
            file=sys.stderr,
        )
        return 3

    print(
        f"[vision-map] PASS: union resolves {n}/{len(prefixes)} vLLM vision "
        f"prefixes onto all {len(covered)} EXL3 vision modules "
        f"(env supplement {a.env_mapping})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
