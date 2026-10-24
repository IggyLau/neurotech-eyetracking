"""Find cameras and measure what they really deliver on this machine.

The vendor spec (U20CAM-9281M: 120 fps MJPG at 1280x800, 1280x720, 800x600 and
640x480) is only an upper bound: the USB link, codec and the OS camera stack (AVFoundation on macOS)
decide the real numbers. Run this before trusting any timing-based metric.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from .camera import open_capture, default_backend

# Modes worth trying on the OV9281 and GC0308. 640x400 is not a native U20CAM-9281M mode;
# AVFoundation delivers it by scaling (likely from the 16:10 1280x800 mode).
CANDIDATE_MODES = [
    (1280, 800, 120), (1280, 720, 120), (800, 600, 120),
    (640, 400, 120), (640, 480, 120), (640, 480, 60), (640, 480, 30),
]


def list_cameras(max_index: int = 8) -> list[dict]:
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, default_backend())
        if cap.isOpened():
            ok, img = cap.read()
            found.append({
                "index": i,
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps_reported": cap.get(cv2.CAP_PROP_FPS),
                "grabbed": bool(ok),
                "mono_like": bool(ok and _is_mono(img)),
            })
        cap.release()
    return found


def _is_mono(img: np.ndarray) -> bool:
    """OV9281 is monochrome; its frames arrive as 3 identical channels."""
    if img is None or img.ndim < 3:
        return True
    small = img[::8, ::8].astype(np.int16)
    return bool(np.abs(small[..., 0] - small[..., 1]).mean() < 2 and
                np.abs(small[..., 1] - small[..., 2]).mean() < 2)


def measure_mode(index: int, width: int, height: int, fps: float, fourcc: str = "MJPG",
                 seconds: float = 3.0) -> dict:
    cap = open_capture(index, width, height, fps, fourcc)
    try:
        for _ in range(10):  # warm-up: auto-exposure and first-frame latency
            cap.read()
        stamps = []
        t_end = time.perf_counter() + seconds
        while time.perf_counter() < t_end:
            ok, _ = cap.read()
            if ok:
                stamps.append(time.perf_counter())
        dt = np.diff(stamps) if len(stamps) > 1 else np.array([np.nan])
        median_dt = float(np.median(dt))
        return {
            "requested": f"{width}x{height}@{fps:g}",
            "actual_size": f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}",
            "fps_reported": round(cap.get(cv2.CAP_PROP_FPS), 1),
            "fps_measured": round(len(stamps) / seconds, 1),
            "jitter_ms": round(float(np.std(dt)) * 1000, 2),
            # a gap > 1.5x the median frame interval counts as a dropped frame
            "gaps": int(np.sum(dt > 1.5 * median_dt)) if len(stamps) > 1 else 0,
        }
    finally:
        cap.release()


def probe(index: int | None = None, seconds: float = 3.0, modes=None) -> None:
    cams = list_cameras()
    if not cams:
        print("No cameras found. On macOS, allow camera access for your terminal in "
              "System Settings > Privacy & Security > Camera.")
        return
    print("Cameras found:")
    for c in cams:
        tag = "  <- looks monochrome (OV9281?)" if c["mono_like"] else ""
        print(f"  [{c['index']}] default {c['width']}x{c['height']} @ {c['fps_reported']:.0f} fps{tag}")
    if index is None:
        print("\nRe-run with --index N to measure frame rates for one camera.")
        return
    print(f"\nMeasuring camera {index} ({seconds:g}s per mode)...")
    print(f"{'requested':>16} {'actual':>10} {'reported':>9} {'measured':>9} {'jitter ms':>10} {'gaps':>5}")
    for w, h, f in modes or CANDIDATE_MODES:
        try:
            r = measure_mode(index, w, h, f, seconds=seconds)
        except RuntimeError as e:
            print(f"{w}x{h}@{f}: {e}")
            continue
        print(f"{r['requested']:>16} {r['actual_size']:>10} {r['fps_reported']:>9} "
              f"{r['fps_measured']:>9} {r['jitter_ms']:>10} {r['gaps']:>5}")
    print("\nPut the best mode into config/local.yaml (eye_camera.width/height/fps).")
