import json

import numpy as np
import pandas as pd
import pytest

from eyetrack.calibration.eye_radius import TARGET_POINTS, measure_radius, target_directions
from eyetrack.config import load_config
from eyetrack.pupil.eye_model import fit_radius_to_targets

SCREEN, WIDTH_CM, DIST_CM = (1440, 900), 30.0, 57.0


def _rotation(yaw_deg, pitch_deg):
    a, b = np.radians(yaw_deg), np.radians(pitch_deg)
    ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    return rx @ ry


def _pupil_positions(center, radius, noise_px=0.0, seed=0):
    """Where the pupil appears for each target, seen by a camera tilted 25 deg up and 10 deg sideways."""
    d = target_directions(np.array(TARGET_POINTS), SCREEN, WIDTH_CM, DIST_CM)
    v = (_rotation(10, 25) @ (d * [1, 1, -1]).T).T          # model convention: looking at the camera is -z
    rng = np.random.default_rng(seed)
    return np.asarray(center) + radius * v[:, :2] + rng.normal(0, noise_px, (len(d), 2)), d


def test_radius_from_targets_recovers_the_true_radius():
    center, radius = (330.0, 210.0), 140.0
    xy, dirs = _pupil_positions(center, radius, noise_px=0.3)
    r, rms = fit_radius_to_targets(*center, xy, dirs)
    assert r == pytest.approx(radius, rel=0.03)
    assert rms < 0.5


def test_measure_radius_from_a_session(tmp_path):
    center, radius = (330.0, 210.0), 140.0
    xy, _ = _pupil_positions(center, radius)
    fs, rows, markers, labels = 120.0, [], [(0.0, "targets_intro")], []
    t = 1.0
    for k, (x, y) in enumerate(xy):
        markers.append((t, f"target_{k}"))
        labels.append({"label": f"target_{k}", "x": TARGET_POINTS[k][0], "y": TARGET_POINTS[k][1],
                       "screen_w": SCREEN[0], "screen_h": SCREEN[1]})
        for i in range(int(2.5 * fs)):
            rows.append({"t_s": t + i / fs, "cx": x, "cy": y, "valid": True, "in_blink": False})
        t += 2.5
    markers.append((t, "blink_intro"))
    pd.DataFrame(rows).to_csv(tmp_path / "pupil.csv", index=False)
    pd.DataFrame(markers, columns=["t_s", "label"]).to_csv(tmp_path / "markers_s.csv", index=False)
    pd.DataFrame(labels).to_csv(tmp_path / "targets.csv", index=False)
    (tmp_path / "metrics_summary.json").write_text(json.dumps(
        {"eye_model": {"cx": center[0], "cy": center[1], "radius": 53.0, "radius_source": "fitted"}}))
    cfg = load_config()
    assert measure_radius(tmp_path, cfg)["radius_px"] is None      # no screen width yet: says so, no guess
    cfg["screen"]["width_cm"], cfg["screen"]["distance_cm"] = WIDTH_CM, DIST_CM
    r = measure_radius(tmp_path, cfg)
    assert r["targets_used"] == 9
    assert r["radius_px"] == pytest.approx(radius, rel=0.02)
    assert r["session_radius_px"] == 53.0
