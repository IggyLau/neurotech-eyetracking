"""Eyeball model: turns 2D pupil ellipses into eye-rotation angles in degrees.

Idea from JEOresearch's 3DTracker (MIT): the pupil is a circle on a sphere, so
each ellipse's minor axis points along the line through the projected eyeball
center. JEO intersects random pairs of these lines frame by frame; here the fit
is done once over a whole recording (least-squares intersection of all lines,
with outlier trimming), which is steadier for offline analysis.

With eyeball center c and radius R (pixels), a pupil at p gives the unit gaze
vector v = ((px-cx)/R, (py-cy)/R, -sqrt(1 - ...)), from which yaw/pitch follow.
These are eye-in-head angles: they let saccade velocities be reported in deg/s
without a scene calibration. Accuracy depends on the camera seeing the eye
from a fixed position, so re-fit after the glasses move.

The radius sets the scale (degrees per pixel) and is the weak part of the fit: with gaze confined to a
screen the minor-axis lines are nearly parallel, and fitted radii ranged 53-218 px across sessions on one
build. A radius measured once from targets at known angles (`fit_radius_to_targets`, the `targets` part of
`eyetrack validate`) can be passed in instead; `radius_warning` flags fits that can't be physical.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def minor_axis_direction(axis_w: float, axis_h: float, angle_deg: float) -> np.ndarray:
    """Unit vector along the ellipse's minor axis (OpenCV RotatedRect convention)."""
    theta = np.radians(angle_deg)
    if axis_w > axis_h:  # the first axis (along theta) is the major one
        theta += np.pi / 2
    return np.array([np.cos(theta), np.sin(theta)])


def intersect_lines(points: np.ndarray, dirs: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Least-squares point closest to all lines p_i + t*d_i."""
    if weights is None:
        weights = np.ones(len(points))
    A = np.zeros((2, 2))
    b = np.zeros(2)
    for p, d, w in zip(points, dirs, weights):
        P = np.eye(2) - np.outer(d, d)
        A += w * P
        b += w * P @ p
    return np.linalg.lstsq(A, b, rcond=None)[0]


def point_line_distance(c: np.ndarray, points: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    v = c - points
    return np.abs(v[:, 0] * dirs[:, 1] - v[:, 1] * dirs[:, 0])


@dataclass
class EyeModel:
    cx: float
    cy: float
    radius: float
    n_used: int = 0
    residual_px: float = np.nan
    radius_source: str = "fitted"     # "fitted" from the ellipses, "fixed" (passed in, e.g. from config), or
    #                                   "fallback" (the fit was implausible; set from the pupil size in pipeline.py)

    @classmethod
    def fit(cls, ellipses: list[tuple], confidences: np.ndarray | None = None,
            min_confidence: float = 0.85, min_eccentricity: float = 0.03,
            trim_iterations: int = 3, radius: float | None = None) -> "EyeModel":
        """Fit from cv2 RotatedRects ((cx, cy), (w, h), angle).

        Only clearly elliptical pupils (eye rotated away from the camera axis) constrain
        the center, so near-circular ellipses are skipped. radius: use this (px) instead of fitting it;
        the center is still fitted.
        """
        conf = np.ones(len(ellipses)) if confidences is None else np.asarray(confidences)
        pts, dirs, ratios = [], [], []
        for e, c in zip(ellipses, conf):
            if e is None or c < min_confidence:
                continue
            (x, y), (w, h), a = e
            ratio = min(w, h) / max(w, h)
            if 1 - ratio < min_eccentricity:
                continue
            pts.append((x, y))
            dirs.append(minor_axis_direction(w, h, a))
            ratios.append(ratio)
        if len(pts) < 10:
            raise ValueError(f"Only {len(pts)} usable ellipses; look around in all directions "
                             "for a few seconds so the eyeball model can be fit.")
        pts, dirs, ratios = np.array(pts), np.array(dirs), np.array(ratios)

        keep = np.ones(len(pts), bool)
        for _ in range(trim_iterations):
            center = intersect_lines(pts[keep], dirs[keep])
            dist = point_line_distance(center, pts, dirs)
            cutoff = np.percentile(dist[keep], 80)
            keep = dist <= max(cutoff, 1.0)
        center = intersect_lines(pts[keep], dirs[keep])
        residual = float(np.median(point_line_distance(center, pts[keep], dirs[keep])))

        if radius is not None:
            return cls(float(center[0]), float(center[1]), float(radius), int(keep.sum()), residual, "fixed")
        # Pupil center sits at R*sin(phi) from the eyeball center, where cos(phi) = minor/major
        d = np.hypot(pts[keep, 0] - center[0], pts[keep, 1] - center[1])
        sin_phi = np.sqrt(np.clip(1 - ratios[keep] ** 2, 1e-6, 1))
        radius = float(np.median(d / sin_phi))
        return cls(float(center[0]), float(center[1]), radius, int(keep.sum()), residual)

    def gaze_vectors(self, px: np.ndarray, py: np.ndarray) -> np.ndarray:
        """Unit gaze vectors (x right, y down, z toward the camera is negative)."""
        x = (np.asarray(px, float) - self.cx) / self.radius
        y = (np.asarray(py, float) - self.cy) / self.radius
        r2 = x * x + y * y
        z = -np.sqrt(np.clip(1 - r2, 0, 1))
        v = np.stack([x, y, z], axis=-1)
        v[r2 > 1] = np.nan  # pupil outside the fitted sphere: not physically valid
        return v

    def angles_deg(self, px: np.ndarray, py: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(yaw, pitch) in degrees; 0,0 = looking straight into the eye camera."""
        v = self.gaze_vectors(px, py)
        yaw = np.degrees(np.arctan2(v[..., 0], -v[..., 2]))
        pitch = np.degrees(np.arctan2(v[..., 1], -v[..., 2]))
        return yaw, pitch

    def to_dict(self) -> dict:
        return {"cx": self.cx, "cy": self.cy, "radius": self.radius,
                "n_used": self.n_used, "residual_px": self.residual_px, "radius_source": self.radius_source}


def radius_warning(radius: float, pupil_diameter_px: float, bounds=(1.5, 6.0)) -> str | None:
    """A message if the radius can't be physical for this pupil size, else None.

    The pupil turns about a point ~10-12 mm behind it, and pupils are ~2-8 mm across, so the radius is
    roughly 1.5-6 pupil diameters. Below that, each pixel of jitter becomes too many degrees and noise
    passes as saccades.
    """
    if not (np.isfinite(radius) and np.isfinite(pupil_diameter_px) and pupil_diameter_px > 0):
        return None
    ratio = radius / pupil_diameter_px
    if bounds[0] <= ratio <= bounds[1]:
        return None
    side = "small" if ratio < bounds[0] else "large"
    return (f"eye model radius {radius:.0f} px is {ratio:.1f}x the pupil diameter ({pupil_diameter_px:.0f} px), "
            f"too {side} to be physical (expected {bounds[0]}-{bounds[1]}x). Angles, saccade sizes and speeds are "
            "scaled wrongly; set eye_model.radius_px from a `targets` measurement.")


def fallback_model(model: EyeModel, pupil_diameter_px: float, ratio: float, px: np.ndarray, py: np.ndarray) -> EyeModel:
    """The same center with radius = ratio x pupil diameter, for when the fitted radius can't be physical.

    The radius is kept above the 99th percentile of the pupil's distance from the center (x 1.05), so almost every
    pupil position still lands on the sphere and gets an angle.
    """
    reach = float(np.nanpercentile(np.hypot(np.asarray(px, float) - model.cx, np.asarray(py, float) - model.cy), 99))
    radius = max(ratio * pupil_diameter_px, 1.05 * reach)
    return EyeModel(model.cx, model.cy, radius, model.n_used, model.residual_px, "fallback")


def angle_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle in degrees between unit vectors (rows)."""
    return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=-1), -1, 1)))


def fit_radius_to_targets(cx: float, cy: float, pupil_xy: np.ndarray, target_dirs: np.ndarray,
                          radius_range=(30.0, 600.0)) -> tuple[float, float]:
    """Radius (px) that makes the model's angles between fixations match the true angles between targets.

    pupil_xy: (n, 2) pupil position while looking at each target; target_dirs: (n, 3) unit vectors from the
    eye to each target. Only angles between pairs are compared, so the eye's resting direction and the
    camera's tilt don't matter. The center stays fixed (the session's fit), because that is the center the
    radius will be used with. Returns (radius, RMS error of the pairwise angles in degrees).
    """
    pupil_xy, target_dirs = np.asarray(pupil_xy, float), np.asarray(target_dirs, float)
    i, j = np.triu_indices(len(pupil_xy), 1)
    true = angle_between(target_dirs[i], target_dirs[j])
    reach = np.max(np.hypot(pupil_xy[:, 0] - cx, pupil_xy[:, 1] - cy))

    def err(r):
        v = EyeModel(cx, cy, r).gaze_vectors(pupil_xy[:, 0], pupil_xy[:, 1])
        return float(np.sqrt(np.mean((angle_between(v[i], v[j]) - true) ** 2)))

    lo = max(radius_range[0], reach * 1.001)    # every fixation must land on the sphere
    grid = np.geomspace(lo, radius_range[1], 400)
    errs = [err(r) for r in grid]
    k = int(np.argmin(errs))
    fine = np.linspace(grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)], 200)
    best = min(fine, key=err)
    return float(best), err(best)
