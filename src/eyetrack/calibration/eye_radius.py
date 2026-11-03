"""Measure the eye model's radius from fixations on targets at known visual angles.

The `targets` part of `eyetrack validate` shows a dot at each of the 9 calibration positions in turn. For each
one, the pupil position over the last part of the fixation is compared with where the dot was: the radius is
chosen so the model's angles between fixations match the true angles between the dots (see
`pupil.eye_model.fit_radius_to_targets`). The result goes in config/local.yaml as `eye_model.radius_px`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..pupil.eye_model import EyeModel, fit_radius_to_targets
from .targets import CALIBRATION_POINTS

TARGET_POINTS = CALIBRATION_POINTS
SETTLE_S = 1.2          # use the last 1.2 s of each target, after the eye has landed on it
MIN_SAMPLES = 30        # trusted frames needed in that window (a quarter of it at 120 Hz)


def target_directions(points: np.ndarray, screen_px: tuple[int, int], width_cm: float,
                      distance_cm: float) -> np.ndarray:
    """Unit vectors from the eye to normalised screen positions, with the eye facing the screen centre."""
    points = np.asarray(points, float)
    cm_per_px = width_cm / screen_px[0]
    x = (points[:, 0] - 0.5) * screen_px[0] * cm_per_px
    y = (points[:, 1] - 0.5) * screen_px[1] * cm_per_px
    v = np.stack([x, y, np.full(len(x), float(distance_cm))], axis=1)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def measure_radius(session: str | Path, cfg: dict) -> dict | None:
    """Radius from a session's target fixations, or None if it has no targets part.

    Needs screen.width_cm in the config; without it the result says so instead of guessing.
    """
    session = Path(session)
    tpath = session / "targets.csv"
    if not tpath.exists():
        return None
    targets = pd.read_csv(tpath)
    markers = pd.read_csv(session / "markers_s.csv").sort_values("t_s").reset_index(drop=True)
    pupil = pd.read_csv(session / "pupil.csv")
    em = json.loads((session / "metrics_summary.json").read_text()).get("eye_model")
    scr = cfg.get("screen", {})
    out: dict = {"targets": len(targets), "radius_px": None}
    if not em:
        out["note"] = "no eye model in this session"
        return out
    if not scr.get("width_cm"):
        out["note"] = "set screen.width_cm (and distance_cm) in config/local.yaml to measure the radius"
        return out

    ends = list(markers.t_s[1:]) + [float(pupil.t_s.max())]
    good = pupil.valid.to_numpy(bool) & ~pupil.in_blink.to_numpy(bool)
    t = pupil.t_s.to_numpy()
    xy, pts = [], []
    for (ts, label), te in zip(markers[["t_s", "label"]].itertuples(index=False), ends):
        if not str(label).startswith("target_"):
            continue
        row = targets[targets.label == label]
        sel = good & (t >= te - SETTLE_S) & (t < te)
        if not len(row) or sel.sum() < MIN_SAMPLES:
            continue
        xy.append((np.median(pupil.cx[sel]), np.median(pupil.cy[sel])))
        pts.append((float(row.x.iloc[0]), float(row.y.iloc[0])))
    out["targets_used"] = len(xy)
    if len(xy) < 4:
        out["note"] = f"only {len(xy)} targets had a steady pupil; need at least 4"
        return out

    screen_px = (int(targets.screen_w.iloc[0]), int(targets.screen_h.iloc[0]))
    dirs = target_directions(np.array(pts), screen_px, scr["width_cm"], scr.get("distance_cm", 57))
    radius, rms = fit_radius_to_targets(em["cx"], em["cy"], np.array(xy), dirs)
    model = EyeModel(em["cx"], em["cy"], radius)
    cx, cy = np.median(np.array(xy), axis=0)
    yaw0, _ = model.angles_deg(np.array([cx]), np.array([cy]))
    yaw1, _ = model.angles_deg(np.array([cx + 1]), np.array([cy]))
    out.update({"radius_px": round(radius, 1), "rms_error_deg": round(rms, 2),
                "deg_per_px": round(float(yaw1[0] - yaw0[0]), 3),
                "session_radius_px": round(float(em["radius"]), 1),
                "session_radius_source": em.get("radius_source", "fitted")})
    return out
