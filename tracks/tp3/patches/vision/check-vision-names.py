#!/usr/bin/env python3
"""HAREM vision arm, gate 2 (model-free): does the PATCHED weight mapper land
every checkpoint tensor on the right vLLM parameter, with the right shard?

Runs on CPU after patch-vision-tp3.py has been applied to the image, reads only
safetensors HEADERS, builds no model and touches no GPU.  Together with
check-vision-mapping.py this is the "motorsuz dogrulama" of
02-donanim-model/VISION-TP3-PLAN.md section 8.1.

WHAT IT PROVES.  `WeightsMapper._map_name_with_shard` (utils.py:97-133) mutates
the key inside the substring loop, so a badly ordered entry can fire twice and
rewrite a name that an earlier entry already rewrote -- silently, into a
parameter path that does not exist, or worse, one that does.  Rather than
reason about it, replay all 1,007 real `model.visual.*` names from the
checkpoint through the mapper the patched image actually carries and compare
every destination against the expected vLLM parameter path.

Expected, from the tower's own module tree (multimodal.py):
    blocks.N.attn.{q,k,v}_proj.X -> blocks.N.attn.qkv.X        shard q/k/v
    blocks.N.attn.qkv.{weight,bias}                            DROPPED (VS3)
    blocks.N.attn.proj.X         -> unchanged
    blocks.N.mlp.{gate,up}_proj.X-> blocks.N.mlp.gate_up_proj.X shard 0/1
    blocks.N.mlp.down_proj.X     -> unchanged
    merger.{gate,up}_proj.X      -> merger.gate_up_proj.X       shard 0/1
    everything else              -> unchanged

Usage:
    check-vision-names.py --model /var/tmp/glm-5.3-flash-turboderp-4.05bpw-tp3
"""

import argparse
import json
import os
import re
import struct
import sys

VIS_PREFIX = "model.visual."


def checkpoint_vision_names(model_dir: str) -> list[str]:
    index = os.path.join(model_dir, "model.safetensors.index.json")
    with open(index) as f:
        weight_map = json.load(f)["weight_map"]
    shards = sorted({v for k, v in weight_map.items() if k.startswith(VIS_PREFIX)})
    names = []
    for shard in shards:
        with open(os.path.join(model_dir, shard), "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            head = json.loads(f.read(n))
        names += [k for k in head if k.startswith(VIS_PREFIX)]
    return sorted(names)


def expected(name: str):
    """(destination, shard_id) the tower's module tree implies, or None to drop."""
    # The top-level mapper (glm4_1v.py:1772, "model.visual." -> "visual.") has
    # already run and AutoWeightsLoader has stripped "visual.", so the tower's
    # own mapper sees the name relative to itself.
    rel = name[len(VIS_PREFIX) :]
    if re.fullmatch(r"blocks\.\d+\.attn\.qkv\.(weight|bias)", rel):
        return None  # VS3 drops the dead pre-quantization fused tensor
    m = re.match(r"(blocks\.\d+\.attn\.)([qkv])_proj\.(.+)", rel)
    if m:
        return f"{m.group(1)}qkv.{m.group(3)}", m.group(2)
    m = re.match(r"((?:blocks\.\d+\.mlp|merger)\.)(gate|up)_proj\.(.+)", rel)
    if m:
        return (
            f"{m.group(1)}gate_up_proj.{m.group(3)}",
            0 if m.group(2) == "gate" else 1,
        )
    return rel, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    a = ap.parse_args()

    from vllm.models.glm5next.nvidia.multimodal import (
        Glm5NextVisionTransformer,
        _harem_vision_env,
    )

    if not _harem_vision_env():
        print(
            "[vision-names] FAIL: HAREM_VISION is not 1 in this environment, so "
            "the class carries the upstream mapper and this gate would test "
            "nothing. Set HAREM_VISION=1.",
            file=sys.stderr,
        )
        return 3

    mapper = Glm5NextVisionTransformer.hf_to_vllm_mapper
    names = checkpoint_vision_names(a.model)
    print(f"[vision-names] {len(names)} model.visual.* tensors from the headers")

    bad, dropped, remapped, passthrough = [], 0, 0, 0
    dests = set()
    for name in names:
        rel = name[len(VIS_PREFIX) :]
        got = mapper._map_name_with_shard(rel)
        want = expected(name)
        if want is None:
            # The mapper itself does not drop these -- VS3's load_weights filter
            # does, before the mapper ever sees them.  Assert that separately.
            dropped += 1
            continue
        if got is None:
            bad.append((rel, "DROPPED", want))
            continue
        if got != want:
            bad.append((rel, got, want))
            continue
        dests.add(got[0])
        if got[0] != rel:
            remapped += 1
        else:
            passthrough += 1

    # VS3's filter, applied to the same names.
    from vllm.models.glm5next.nvidia.multimodal import _HAREM_VISION_DEAD_FUSED

    filtered = [n for n in names if n.endswith(_HAREM_VISION_DEAD_FUSED)]
    print(
        f"[vision-names] VS3 would drop {len(filtered)} dead fused tensors "
        f"(expected {dropped})"
    )
    print(
        f"[vision-names] mapper: {remapped} remapped, {passthrough} passed "
        f"through, {len(dests)} distinct destinations"
    )

    if len(filtered) != dropped:
        print(
            f"[vision-names] FAIL: VS3's filter drops {len(filtered)} tensors "
            f"but {dropped} are dead fused ones.",
            file=sys.stderr,
        )
        return 3
    if bad:
        for rel, got, want in bad[:20]:
            print(f"    {rel}\n      got  {got}\n      want {want}", file=sys.stderr)
        print(
            f"[vision-names] FAIL: {len(bad)} of {len(names)} names map wrongly.",
            file=sys.stderr,
        )
        return 3

    print(
        f"[vision-names] PASS: all {len(names) - dropped} live vision tensors map "
        f"onto the expected parameter and shard; {dropped} dead fused tensors "
        "are dropped by VS3"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
