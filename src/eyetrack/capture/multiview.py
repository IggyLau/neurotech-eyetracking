"""Live view of every camera in a mode, side by side, with pupil detection on the eye cameras.

For checking which index is which camera, aiming the cameras, and seeing that every stream runs at its frame
rate before recording. Nothing is saved. With a calibration.json, each eye also gets its eyeball outline and the
gaze ray from the eyeball centre through the pupil, from the eye model that calibration was fitted with.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from ..pupil.detector import DetectorParams, detect_pupil, draw_result
from .camera import Camera
from .recorder import MODE_NAMES, mode_roles

TILE_W = 480


def _tile(img: np.ndarray, label: str) -> np.ndarray:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    t = cv2.resize(img, (TILE_W, int(TILE_W * img.shape[0] / img.shape[1])))
    cv2.putText(t, label, (8, t.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    return t


def eye_models_from_calibration(path) -> dict:
    """role -> EyeModel from each eye's jeo mapping in a calibration.json (eyes without one are left out)."""
    from ..calibration.scene import Calibration
    cal = Calibration.load(path)
    return {role: maps["jeo"].eye_model for role, maps in cal.eyes.items() if maps.get("jeo") is not None}


def draw_eye(img: np.ndarray, r, eye_model=None, min_confidence: float = 0.5) -> np.ndarray:
    """Pupil ellipse, and the gaze ray when there's an eye model and the pupil is trusted."""
    vis = draw_result(img, r)
    if eye_model is None:
        return vis
    from ..viz.plots import draw_gaze_ray
    ok = r.found and r.confidence >= min_confidence
    return draw_gaze_ray(vis, eye_model, r.cx if ok else np.nan, r.cy if ok else np.nan)


def preview(cfg: dict, mode: str = "3", calibration=None):
    roles = mode_roles(mode)
    models = eye_models_from_calibration(calibration) if calibration else {}
    for role in roles:
        if role.startswith("eye") and calibration and role not in models:
            print(f"  {role}: no eye model in {calibration}, showing the pupil only")
    cams = {}
    try:
        for role in roles:
            cams[role] = Camera.from_config(cfg[f"{role}_camera"], role).start()
    except Exception:
        for c in cams.values():
            c.stop()
        raise
    params = DetectorParams.from_config(cfg["pupil"])
    print(f"Preview, {MODE_NAMES[mode]}. q = quit")
    last = {r: (-1, None) for r in roles}
    counts = {r: (time.perf_counter(), 0, 0.0) for r in roles}
    try:
        while True:
            tiles = []
            for role, cam in cams.items():
                f = cam.latest()
                t0, n0, fps = counts[role]
                now = time.perf_counter()
                if now - t0 >= 1:
                    counts[role] = (now, cam.frame_count, (cam.frame_count - n0) / (now - t0))
                    fps = counts[role][2]
                label = f"{role} (index {cfg[f'{role}_camera']['index']})  {fps:.0f} fps"
                if f is None:
                    tiles.append(_tile(np.zeros((300, 400, 3), np.uint8), label + "  no frames"))
                    continue
                if role.startswith("eye"):
                    idx, vis = last[role]
                    if f.index != idx:
                        img = cv2.flip(f.image, 1) if cfg[f"{role}_camera"].get("mirror_preview") else f.image
                        r = detect_pupil(img, params)
                        vis = draw_eye(img, r, models.get(role), cfg["pupil"]["min_confidence"])
                        last[role] = (f.index, vis)
                    tiles.append(_tile(vis, label))
                else:
                    tiles.append(_tile(f.image, label))
            if tiles:
                h = max(t.shape[0] for t in tiles)
                tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
                cv2.imshow("eyetrack preview", np.hstack(tiles))
            if cv2.waitKey(15) & 0xFF == ord("q"):
                break
    finally:
        for c in cams.values():
            c.stop()
        cv2.destroyAllWindows()
