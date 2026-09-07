#!/usr/bin/env python3
"""HAREM vision arm -- model-free VIDEO GEOMETRY gate (7 September 2026, round 3).

WHY THIS GATE EXISTS.  Rounds 1 and 2 of the vision arm died on the first video
request, on all three ranks, with

    ValueError: Attempted to assign 1656 = 1656 multimodal tokens to
                1728 placeholders          -> worker dead -> EngineDeadError

and the fault was catchable on a CPU in under a second: a video's PLACEHOLDER
count and its ENCODER ROW count come from two different frame samplers
(glm4_1v.py:1265 vs glm5next.py:597), and nothing checked that they agree.  The
install-day gates (check-vision-mapping.py, check-vision-names.py) looked only
at WEIGHT LOADING.  Geometry had no gate at all.  This is that gate.

WHAT IT ASSERTS, per video:
  1. `Glm5NextVideoProcessor.sample_frames` returns a NON-EMPTY frame list.
     (A clip shorter than ~0.5 s makes extract_t = int(duration*fps_interval)
     collapse to 0 and the sampler return [] -- measured on a 2-frame fixture;
     that request 400s in the frontend, which is survivable, but a boot-time
     gate should say so out loud.)
  2. len(timestamps) == grid_t, where
       - grid_t comes from the REAL pixel path (the processor's own
         _preprocess, i.e. the tensor the encoder will actually see), and
       - timestamps come from `Glm5NextProcessingInfo._get_video_second_idx_glm46v`
         -- the PATCHED method when HAREM_VISION=1.  Unpatched, this gate fails,
         which is the point: it reproduces the boot, it does not simulate it.
  3. placeholders == encoder rows, i.e. len(timestamps)*H*W//merge^2 ==
     T*H*W//merge^2 -- the exact equation that killed the engine.
  4. the braked token count stays under --max-video-tokens.

It builds NO model, loads NO weights, needs NO GPU: only the checkpoint's
processor config plus the fixture MP4s.

The 1080p / 60 s case is ANALYTIC on purpose.  vLLM's own loader caps a video at
VideoMediaIO.num_frames (32) before the processor ever sees it, so a real 60 s
1080p decode would move ~11 GB of frames through a container that runs during
boot -- for geometry that is fully determined by (frame count, h, w, budget).
The analytic arm therefore drives the SAME functions (glm_sample_frame_indices,
smart_resize, _pixel_budget) with 1080p metadata, in BOTH loader branches
(do_sample_frames True and False), and asserts the same equality.

Usage:
  check-video-geometry.py --model /path/to/checkpoint [--fixtures DIR]
                          [--max-video-tokens 8000]
"""

import argparse
import os
import sys

import numpy as np


def fail(msg: str) -> int:
    print(f"[video-geom] FAIL: {msg}", file=sys.stderr)
    return 4


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--fixtures", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "fixtures"))
    ap.add_argument("--max-video-tokens", type=int, default=8000,
                    help="the per-item brake the .env passes as "
                         "--mm-processor-kwargs max_image_tokens")
    a = ap.parse_args()

    from vllm.transformers_utils.processors.glm5next import (
        Glm5NextProcessor, _pixel_budget, glm_sample_frame_indices, smart_resize)
    from vllm.models.glm5next.nvidia.multimodal import Glm5NextProcessingInfo

    gated = os.environ.get("HAREM_VISION") == "1"
    own = "_get_video_second_idx_glm46v" in vars(Glm5NextProcessingInfo)
    if gated and not own:
        return fail(
            "HAREM_VISION=1 but Glm5NextProcessingInfo does not define "
            "_get_video_second_idx_glm46v -- VS4 is not applied to this tree. "
            "Booting now reproduces the round-1/round-2 engine death.")

    proc = Glm5NextProcessor.from_pretrained(a.model)
    vp, ip = proc.video_processor, proc.image_processor
    tps = int(vp.temporal_patch_size)
    m2 = int(ip.merge_size) ** 2
    factor = vp.patch_size * vp.merge_size * vp.patch_expand_factor

    class _Info(Glm5NextProcessingInfo):
        """A real subclass, not a duck: the patched methods call zero-arg
        `super()`, which needs `self` to be an instance of the class.  No
        ModelConfig is built -- only the two accessors the video-geometry path
        touches are supplied."""
        def __init__(self):
            pass

        def get_video_processor(self, **k):
            return vp

        def get_hf_processor(self, **k):
            return proc

    try:
        info = _Info()
    except TypeError:
        info = object.__new__(_Info)
    ts_fn = Glm5NextProcessingInfo._get_video_second_idx_glm46v

    # --- real fixtures, full pixel path -----------------------------------
    from vllm.model_executor.models.glm4_1v import _to_video_metadata
    from vllm.multimodal.media.image import ImageMediaIO
    from vllm.multimodal.media.video import VideoMediaIO
    vio = VideoMediaIO(ImageMediaIO())

    fixtures = sorted(f for f in os.listdir(a.fixtures)
                      if f.endswith(".mp4") and "2frame" not in f)
    if not fixtures:
        return fail(f"no fixture .mp4 under {a.fixtures}")

    checked = 0
    for name in fixtures:
        path = os.path.join(a.fixtures, name)
        with open(path, "rb") as fh:
            loaded = vio.load_bytes(fh.read())
        frames, md = loaded.media if hasattr(loaded, "media") else loaded

        meta = _to_video_metadata(md)
        if md.get("do_sample_frames", True):
            sampled = vp.sample_frames(meta)
            if sampled is None or len(sampled) == 0:
                return fail(f"{name}: sample_frames() returned no frames "
                            f"(fps={md.get('fps')}, "
                            f"duration={md.get('duration')})")
        else:
            sampled = range(len(frames))

        out = proc(text="<|begin_of_video|><|video|><|end_of_video|>",
                   videos=[[frames]], video_metadata=[[meta]],
                   do_sample_frames=md.get("do_sample_frames", True),
                   max_image_tokens=a.max_video_tokens)
        t, h, w = [int(x) for x in out["video_grid_thw"].tolist()[0]]
        per_frame = h * w // m2
        rows = t * per_frame

        ts = ts_fn(info, md, len(frames))
        ph = len(ts) * per_frame
        ok = (len(ts) == t) and (ph == rows)
        print(f"[video-geom] {name}: frames={len(frames)} sampled={len(sampled)} "
              f"grid=({t},{h},{w}) rows={rows} timestamps={len(ts)} "
              f"placeholders={ph} tokens={rows} {'OK' if ok else 'MISMATCH'}")
        if not ok:
            return fail(f"{name}: {ph} placeholders for {rows} encoder rows "
                        f"(timestamps={len(ts)}, grid_t={t}). Booting with this "
                        "would kill the engine core on the first video request.")
        if rows > a.max_video_tokens:
            return fail(f"{name}: {rows} vision tokens exceeds the brake "
                        f"({a.max_video_tokens})")
        checked += 1

    # --- analytic: 1080p, both loader branches ----------------------------
    for label, n_total, fps, dur, delivered, do_sample in (
            ("1080p/60s@30fps loader-capped", 1800, 30.0, 60.0, 32, False),
            ("1080p/60s@0.5fps whole-clip", 30, 0.5, 60.0, 30, True),
            ("1080p/8s@8fps whole-clip", 64, 8.0, 8.0, 64, True)):
        if do_sample:
            idx = [int(i) for i in glm_sample_frame_indices(
                n_total, fps, dur, target_fps=vp.fps_interval,
                max_frame_count=vp.max_frame_count_dynamic,
                temporal_patch_size=tps)]
            nf = len(idx)
        else:
            idx = np.linspace(0, n_total - 1, delivered, dtype=int).tolist()
            nf = delivered
        if nf == 0:
            return fail(f"{label}: the sampler selected no frames")
        md = {"fps": fps, "duration": dur, "total_num_frames": n_total,
              "frames_indices": idx, "do_sample_frames": do_sample}
        mn, mx = _pixel_budget(vp.min_image_tokens, a.max_video_tokens,
                               vp.patch_size, vp.merge_size, tps)
        hb, wb = smart_resize(t=nf, h=1080, w=1920, t_factor=tps,
                              h_factor=factor, w_factor=factor,
                              min_pixels=mn, max_pixels=mx)
        grid_t = max(nf + (-nf % tps), tps) // tps
        gh, gw = hb // vp.patch_size, wb // vp.patch_size
        tokens = grid_t * gh * gw // m2
        ts = ts_fn(info, md, nf)
        ok = len(ts) == grid_t and tokens <= a.max_video_tokens
        print(f"[video-geom] {label}: frames={nf} canvas={hb}x{wb} "
              f"grid=({grid_t},{gh},{gw}) tokens={tokens} timestamps={len(ts)} "
              f"{'OK' if ok else 'MISMATCH'}")
        if len(ts) != grid_t:
            return fail(f"{label}: {len(ts)} timestamps for grid_t={grid_t}")
        if tokens > a.max_video_tokens:
            return fail(f"{label}: {tokens} vision tokens exceeds the brake "
                        f"({a.max_video_tokens})")
        checked += 1

    # --- the per-prompt video limit VS7 declares --------------------------
    limit = None
    if gated:
        try:
            limit = Glm5NextProcessingInfo.get_supported_mm_limits(info)["video"]
        except Exception as e:
            return fail(f"get_supported_mm_limits raised {type(e).__name__}: {e}")

    print(f"[video-geom] PASS: {checked} video geometries agree "
          f"(placeholders == encoder rows, brake {a.max_video_tokens} tok/item)"
          + (f"; declared per-prompt video limit = {limit}" if limit else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
