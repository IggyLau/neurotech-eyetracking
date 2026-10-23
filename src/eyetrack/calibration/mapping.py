"""Map pupil position (eye-camera px) to gaze position (scene-camera px).

Standard 2D regression approach for head-mounted trackers (as in Pupil Capture's
2D mode): a 2nd-order polynomial fitted on fixations to known targets.
Validation error is reported in scene px and in degrees using the scene
camera's horizontal field of view.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .targets import get_dictionary


def poly_features(x: np.ndarray, y: np.ndarray, degree: int = 2) -> np.ndarray:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    cols = [np.ones_like(x)]
    for d in range(1, degree + 1):
        for i in range(d + 1):
            cols.append(x ** (d - i) * y ** i)
    return np.stack(cols, axis=-1)


@dataclass
class PolyMapper:
    coef_x: np.ndarray
    coef_y: np.ndarray
    degree: int
    norm: tuple[float, float, float, float]  # mean_x, mean_y, scale_x, scale_y of pupil input

    @classmethod
    def fit(cls, pupil_xy: np.ndarray, scene_xy: np.ndarray, degree: int = 2) -> "PolyMapper":
        pupil_xy = np.asarray(pupil_xy, float)
        scene_xy = np.asarray(scene_xy, float)
        n_terms = poly_features(np.zeros(1), np.zeros(1), degree).shape[1]
        if len(pupil_xy) < n_terms:
            raise ValueError(f"Need at least {n_terms} calibration points for degree {degree}, got {len(pupil_xy)}")
        mx, my = pupil_xy.mean(axis=0)
        sx, sy = pupil_xy.std(axis=0) + 1e-9
        A = poly_features((pupil_xy[:, 0] - mx) / sx, (pupil_xy[:, 1] - my) / sy, degree)
        cx = np.linalg.lstsq(A, scene_xy[:, 0], rcond=None)[0]
        cy = np.linalg.lstsq(A, scene_xy[:, 1], rcond=None)[0]
        return cls(cx, cy, degree, (mx, my, sx, sy))

    def predict(self, px, py) -> tuple[np.ndarray, np.ndarray]:
        mx, my, sx, sy = self.norm
        A = poly_features((np.asarray(px, float) - mx) / sx, (np.asarray(py, float) - my) / sy, self.degree)
        return A @ self.coef_x, A @ self.coef_y

    def save(self, path: str | Path):
        Path(path).write_text(json.dumps({"coef_x": self.coef_x.tolist(), "coef_y": self.coef_y.tolist(),
                                          "degree": self.degree, "norm": list(self.norm)}, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "PolyMapper":
        d = json.loads(Path(path).read_text())
        return cls(np.array(d["coef_x"]), np.array(d["coef_y"]), d["degree"], tuple(d["norm"]))


def detect_marker_track(scene_video: str | Path, dict_name: str = "DICT_4X4_50", marker_id: int = 0) -> pd.DataFrame:
    """Center of the calibration marker in every scene frame (NaN where not found)."""
    detector = cv2.aruco.ArucoDetector(get_dictionary(dict_name), cv2.aruco.DetectorParameters())
    cap = cv2.VideoCapture(str(scene_video))
    rows, i = [], 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        corners, ids, _ = detector.detectMarkers(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        x = y = np.nan
        if ids is not None:
            for c, mid in zip(corners, ids.ravel()):
                if mid == marker_id:
                    x, y = c[0].mean(axis=0)
        rows.append({"frame_index": i, "marker_x": x, "marker_y": y})
        i += 1
    cap.release()
    return pd.DataFrame(rows)


def target_points(markers: pd.DataFrame, pupil: pd.DataFrame, scene: pd.DataFrame, prefix: str,
                  settle_s: float = 0.6, end_trim_s: float = 0.2) -> pd.DataFrame:
    """One robust (pupil, scene) pair per target: medians over the settled part of each dwell.

    markers: t_s, label     pupil: t_s, cx, cy, valid     scene: t_s, marker_x, marker_y
    """
    m = markers[markers.label.str.startswith(prefix + "_")].sort_values("t_s").reset_index(drop=True)
    ends = list(m.t_s.iloc[1:]) + [m.t_s.iloc[-1] + (m.t_s.diff().median() if len(m) > 1 else 2.0)] if len(m) else []
    rows = []
    for (t0, label), t1 in zip(m[["t_s", "label"]].itertuples(index=False), ends):
        a, b = t0 + settle_s, t1 - end_trim_s
        p = pupil[(pupil.t_s >= a) & (pupil.t_s <= b) & pupil.valid]
        s = scene[(scene.t_s >= a) & (scene.t_s <= b)].dropna()
        if len(p) < 5 or len(s) < 3:
            continue
        rows.append({"label": label, "pupil_x": p.cx.median(), "pupil_y": p.cy.median(),
                     "scene_x": s.marker_x.median(), "scene_y": s.marker_y.median(),
                     "pupil_sd": float(np.hypot(p.cx.std(), p.cy.std()))})
    return pd.DataFrame(rows)


def validation_error(mapper: PolyMapper, pts: pd.DataFrame, scene_width: int, hfov_deg: float) -> pd.DataFrame:
    gx, gy = mapper.predict(pts.pupil_x, pts.pupil_y)
    out = pts.copy()
    out["gaze_x"], out["gaze_y"] = gx, gy
    out["error_px"] = np.hypot(gx - pts.scene_x, gy - pts.scene_y)
    out["error_deg"] = out.error_px * (hfov_deg / scene_width)  # small-angle approximation
    return out
