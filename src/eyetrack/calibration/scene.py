"""Gaze -> scene-camera pixels, fitted per eye, and two eyes combined into one gaze point.

The "jeo" mapping is ported from JEOresearch/EyeTracker (https://github.com/JEOresearch/EyeTracker),
Copyright (c) 2024 JEOresearch, MIT License (see NOTICE):
  - project_gaze_to_tangent, build_camera_calibration, camera_pixel_ray, camera_angular_error_degrees:
    FrontCameraTracker/Orlosky3DEyeTrackerFrontCamera.py
  - averaging the two eyes' gaze points: VREyeTracker/EyeTrackerStereo.cs
JEO's versions work on one live sample at a time with five fixed targets on a 640x480 image. Here they take
any number of targets, any image size and whole recordings. docs/design-history/jeoresearch-changes.md lists
every change.

How it works (JEO): the eye model gives a 3D gaze direction per frame. The direction while looking at the
centre target becomes "forward", the one at the target above it fixes "up". Each direction is projected onto
the plane one unit in front of the eye (tangent-plane coordinates), and a 2x2 matrix maps those to pixel
offsets from the centre target. Each eye gets its own basis and matrix, so the two eye cameras don't have to
be mounted symmetrically.

The "poly" mapping (calibration/mapping.py, this repo's older method) maps pupil position straight to scene
pixels and doesn't use the eye model. Both are fitted and reported; config calibration.method picks one.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..pupil.eye_model import EyeModel
from .mapping import PolyMapper

JEO_DIAGONAL_FOV_DEGREES = 60.0   # JEO's FRONT_DIAGONAL_FOV_DEGREES, used when the camera's isn't configured


# ------------------------------------------------------------------ ported from JEO (FrontCameraTracker)

def project_gaze_to_tangent(direction, basis):
    """Verbatim from JEO: one gaze direction -> (right, up) on the plane one unit along `forward`."""
    direction = np.asarray(direction, dtype=float)
    if direction.shape != (3,) or not np.isfinite(direction).all():
        raise ValueError("No valid gaze direction.")
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        raise ValueError("Gaze direction has zero length.")
    direction = direction / norm
    forward, right, up = basis
    depth = np.dot(direction, forward)
    if depth <= 1e-4:
        raise ValueError("Gaze points away from the calibration plane.")
    return np.array([np.dot(direction, right), np.dot(direction, up)]) / depth


def project_many(directions: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """project_gaze_to_tangent for an (N, 3) array; rows JEO would reject come back as NaN."""
    d = np.asarray(directions, float)
    norm = np.linalg.norm(d, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = d / norm
        forward, right, up = basis
        depth = d @ forward
        out = np.stack([d @ right, d @ up], axis=-1) / depth[..., None]
    bad = ~np.isfinite(out).all(axis=-1) | ~(depth > 1e-4) | ~(norm[..., 0] >= 1e-8)
    out[bad] = np.nan
    return out


def build_camera_calibration(directions, targets, centre_index: int, top_index: int):
    """JEO's build_camera_calibration, generalised from exactly five targets to any number.

    directions: (N, 3) gaze directions, one per target. targets: (N, 2) where each target was in the scene
    image (px). centre_index / top_index: the centre target and the one above it, which set the basis.
    JEO's targets are fixed at the image centre and 15% in from each edge; here they're wherever the
    target was seen, and offsets are taken from the centre target rather than the image centre.
    Returns (basis, matrix, rms_px).
    """
    directions = np.asarray(directions, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if len(directions) < 3 or directions.shape != (len(targets), 3) or not np.isfinite(directions).all():
        raise ValueError("At least three valid gaze samples are required.")
    norms = np.linalg.norm(directions, axis=1)
    if np.any(norms < 1e-8):
        raise ValueError("A gaze sample has zero length.")
    directions = directions / norms[:, None]
    forward = directions[centre_index]
    up = directions[top_index] - np.dot(directions[top_index], forward) * forward
    if np.linalg.norm(up) < 1e-3:
        raise ValueError("CENTER and TOP rays are too similar; repeat calibration.")
    up /= np.linalg.norm(up)
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    basis = np.array([forward, right, up])
    others = [i for i in range(len(directions)) if i != centre_index]
    measured = np.array([project_gaze_to_tangent(directions[i], basis) for i in others])
    singular_values = np.linalg.svd(measured, compute_uv=False)
    if singular_values[-1] < 1e-3 or singular_values[0] / singular_values[-1] > 100:
        raise ValueError("Outer gaze samples do not span two reliable axes; repeat calibration.")
    offsets = targets[others] - targets[centre_index]
    coefficients, _, _, _ = np.linalg.lstsq(measured, offsets, rcond=None)
    matrix = coefficients.T
    if np.linalg.matrix_rank(matrix) < 2 or np.linalg.cond(matrix) > 100:
        raise ValueError("Target mapping is degenerate; repeat calibration.")
    errors = np.linalg.norm(measured @ coefficients - offsets, axis=1)
    return basis, matrix, float(np.sqrt(np.mean(errors ** 2)))


def camera_pixel_ray(point, size, diagonal_fov_deg: float | None = None) -> np.ndarray:
    """JEO's camera_pixel_ray for any image size: square pixels, centred principal point, pinhole."""
    w, h = size
    fov = diagonal_fov_deg or JEO_DIAGONAL_FOV_DEGREES
    focal_pixels = math.hypot(w, h) / (2.0 * math.tan(math.radians(fov) / 2.0))
    ray = np.array([(point[0] - w / 2) / focal_pixels, (h / 2 - point[1]) / focal_pixels, 1.0], dtype=float)
    return ray / np.linalg.norm(ray)


def camera_angular_error_degrees(target, gaze, size, diagonal_fov_deg: float | None = None) -> float:
    """JEO's camera_angular_error_degrees: angle between the scene-camera rays through two pixels."""
    target_ray = camera_pixel_ray(target, size, diagonal_fov_deg)
    gaze_ray = camera_pixel_ray(gaze, size, diagonal_fov_deg)
    # atan2 stays accurate for very small errors and identical samples.
    return math.degrees(math.atan2(np.linalg.norm(np.cross(target_ray, gaze_ray)), np.dot(target_ray, gaze_ray)))


# ------------------------------------------------------------------ choosing JEO's centre and top targets

def centre_and_top(targets: np.ndarray) -> tuple[int, int]:
    """The target nearest the middle of the set, and the one most directly above it (image y points down)."""
    t = np.asarray(targets, float)
    centre = int(np.argmin(np.linalg.norm(t - np.median(t, axis=0), axis=1)))
    d = t - t[centre]
    score = -d[:, 1] - 2 * np.abs(d[:, 0])     # far up, little sideways
    score[centre] = -np.inf
    score[d[:, 1] >= 0] = -np.inf
    if not np.isfinite(score).any():
        raise ValueError("No calibration target above the centre target.")
    return centre, int(np.argmax(score))


# ------------------------------------------------------------------ per-eye mappers

@dataclass
class JeoMapper:
    """Eye-model gaze direction -> scene px (JEO). Keeps the eye model it was fitted with."""
    basis: np.ndarray           # rows: forward, right, up (eye-camera frame)
    matrix: np.ndarray          # 2x2, tangent coordinates -> px offset from the centre target
    centre_px: np.ndarray       # where the centre target was in the scene image
    eye_model: EyeModel
    rms_px: float = float("nan")

    @classmethod
    def fit(cls, eye_model: EyeModel, pupil_xy: np.ndarray, scene_xy: np.ndarray) -> "JeoMapper":
        pupil_xy, scene_xy = np.asarray(pupil_xy, float), np.asarray(scene_xy, float)
        dirs = eye_model.gaze_vectors(pupil_xy[:, 0], pupil_xy[:, 1])
        ok = np.isfinite(dirs).all(axis=1)
        if ok.sum() < 3:
            raise ValueError(f"Only {ok.sum()} calibration pupils fall on the fitted eyeball; need 3+.")
        dirs, scene_xy = dirs[ok], scene_xy[ok]
        c, top = centre_and_top(scene_xy)
        basis, matrix, rms = build_camera_calibration(dirs, scene_xy, c, top)
        return cls(basis, matrix, scene_xy[c].copy(), eye_model, rms)

    def predict(self, px, py) -> tuple[np.ndarray, np.ndarray]:
        """JEO's current_gaze_camera_point for a whole recording (unclipped; NaN where there's no direction)."""
        dirs = self.eye_model.gaze_vectors(np.asarray(px, float), np.asarray(py, float))
        m = project_many(dirs, self.basis)
        pts = self.centre_px + m @ self.matrix.T
        return pts[..., 0], pts[..., 1]

    def to_dict(self) -> dict:
        return {"basis": self.basis.tolist(), "matrix": self.matrix.tolist(), "centre_px": self.centre_px.tolist(),
                "eye_model": self.eye_model.to_dict(), "rms_px": self.rms_px}

    @classmethod
    def from_dict(cls, d: dict) -> "JeoMapper":
        em = d["eye_model"]
        model = EyeModel(em["cx"], em["cy"], em["radius"], radius_source=em.get("radius_source", "fitted"))
        return cls(np.array(d["basis"]), np.array(d["matrix"]), np.array(d["centre_px"]), model,
                   d.get("rms_px", float("nan")))


def _poly_to_dict(m: PolyMapper) -> dict:
    return {"coef_x": m.coef_x.tolist(), "coef_y": m.coef_y.tolist(), "degree": m.degree, "norm": list(m.norm)}


def _poly_from_dict(d: dict) -> PolyMapper:
    return PolyMapper(np.array(d["coef_x"]), np.array(d["coef_y"]), d["degree"], tuple(d["norm"]))


@dataclass
class Calibration:
    """Per-eye gaze -> scene mappings from one calibration session (saved as calibration.json)."""
    eyes: dict = field(default_factory=dict)     # role -> {"poly": PolyMapper | None, "jeo": JeoMapper | None}
    method: str = "jeo"
    scene_size: tuple[int, int] | None = None
    diagonal_fov_deg: float | None = None

    def mapper(self, role: str, method: str | None = None):
        """The mapper for this eye: the requested method, else whichever was fitted."""
        maps = self.eyes.get(role) or {}
        method = method or self.method
        return maps.get(method) or maps.get("poly") or maps.get("jeo")

    def predict(self, role: str, px, py, method: str | None = None) -> tuple[np.ndarray, np.ndarray]:
        m = self.mapper(role, method)
        if m is None:
            n = len(np.atleast_1d(px))
            return np.full(n, np.nan), np.full(n, np.nan)
        return m.predict(px, py)

    def save(self, path: str | Path):
        eyes = {role: {k: (_poly_to_dict(m) if k == "poly" else m.to_dict()) for k, m in maps.items() if m is not None}
                for role, maps in self.eyes.items()}
        Path(path).write_text(json.dumps({"version": 2, "method": self.method, "scene_size": self.scene_size,
                                          "diagonal_fov_deg": self.diagonal_fov_deg, "eyes": eyes}, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Calibration":
        d = json.loads(Path(path).read_text())
        if "coef_x" in d:          # calibration.json from before two-eye support: one eye, polynomial
            return cls({"eye": {"poly": _poly_from_dict(d), "jeo": None}}, method="poly")
        eyes = {}
        for role, maps in d["eyes"].items():
            eyes[role] = {"poly": _poly_from_dict(maps["poly"]) if "poly" in maps else None,
                          "jeo": JeoMapper.from_dict(maps["jeo"]) if "jeo" in maps else None}
        size = tuple(d["scene_size"]) if d.get("scene_size") else None
        return cls(eyes, d.get("method", "jeo"), size, d.get("diagonal_fov_deg"))


# ------------------------------------------------------------------ two eyes

def combine_eyes(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Average two eyes' gaze points (EyeTrackerStereo.cs: combined = (left + right) / 2).

    Unlike JEO, a frame where only one eye has a gaze point keeps that eye's point instead of being dropped.
    a, b: (N, 2) scene px with NaN where the eye had none.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok_a, ok_b = np.isfinite(a).all(axis=1), np.isfinite(b).all(axis=1)
    out = np.full_like(a, np.nan)
    both = ok_a & ok_b
    out[both] = (a[both] + b[both]) / 2
    out[ok_a & ~ok_b] = a[ok_a & ~ok_b]
    out[ok_b & ~ok_a] = b[ok_b & ~ok_a]
    return out


def align_samples(t_src: np.ndarray, values: np.ndarray, t_dst: np.ndarray, max_dt: float) -> np.ndarray:
    """Nearest sample of values (at t_src) for each t_dst; NaN when none is within max_dt seconds."""
    t_src, values = np.asarray(t_src, float), np.asarray(values, float)
    out = np.full((len(t_dst),) + values.shape[1:], np.nan)
    if not len(t_src):
        return out
    j = np.clip(np.searchsorted(t_src, t_dst), 1, len(t_src) - 1)
    left_closer = np.abs(t_dst - t_src[j - 1]) <= np.abs(t_src[j] - t_dst)
    j = np.where(left_closer, j - 1, j)
    if len(t_src) == 1:
        j = np.zeros(len(t_dst), int)
    ok = np.abs(t_src[j] - t_dst) <= max_dt
    out[ok] = values[j[ok]]
    return out
