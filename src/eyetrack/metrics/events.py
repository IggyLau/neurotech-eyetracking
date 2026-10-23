"""Fixation / saccade detection with a velocity threshold (I-VT, Salvucci & Goldberg 2000).

Positions can be eye-rotation angles in degrees (from EyeModel), calibrated
scene gaze in degrees, or raw pupil pixels; the threshold must match the unit.

Sampling-rate limits for the OV9281 at ~120 Hz: saccades (20-200 ms) span
3-25 samples, so detection and amplitude are fine, but peak velocity is
underestimated for small saccades. Microsaccades (<1 deg, ~10-20 ms) need
>=200 Hz plus a lot less pupil-center noise than a DIY rig usually gives.
For a second opinion, run pymovements or REMoDNaV on gaze.csv (`pip install .[compare]`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

from .blinks import _runs


def velocity(t_s: np.ndarray, x: np.ndarray, y: np.ndarray, smoothing_window: int = 3) -> np.ndarray:
    """Point-to-point speed (units/s) after a median filter, using central differences."""
    t_s = np.asarray(t_s, float)
    if smoothing_window > 1:
        x = median_filter(np.asarray(x, float), size=smoothing_window, mode="nearest")
        y = median_filter(np.asarray(y, float), size=smoothing_window, mode="nearest")
    vx = np.gradient(x, t_s)
    vy = np.gradient(y, t_s)
    return np.hypot(vx, vy)


def detect_events(t_s, x, y, valid=None, threshold: float = 30.0, min_saccade_ms: float = 10,
                  min_fixation_ms: float = 60, smoothing_window: int = 3) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Return (saccades, fixations, per-sample label array: 'fix'|'sac'|'none')."""
    t_s = np.asarray(t_s, float)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    valid = np.isfinite(x) & np.isfinite(y) if valid is None else (np.asarray(valid, bool) & np.isfinite(x) & np.isfinite(y))

    # interpolate tiny gaps so the median filter / gradient don't smear NaNs; masked out again below
    xi = pd.Series(np.where(valid, x, np.nan)).interpolate(limit_direction="both").to_numpy()
    yi = pd.Series(np.where(valid, y, np.nan)).interpolate(limit_direction="both").to_numpy()
    speed = velocity(t_s, xi, yi, smoothing_window)
    speed[~valid] = np.nan

    labels = np.full(len(t_s), "none", dtype=object)
    fast = valid & (speed > threshold)
    saccades = []
    for s, e in _runs(fast):
        dur_ms = (t_s[e - 1] - t_s[s]) * 1000 + _dt_ms(t_s)
        if dur_ms < min_saccade_ms:
            continue
        # amplitude from the samples just outside the fast run (start/end of the movement)
        a, b = max(s - 1, 0), min(e, len(t_s) - 1)
        if not (valid[a] and valid[b]):
            continue  # saccade touching a blink/dropout: amplitude unreliable
        labels[s:e] = "sac"
        saccades.append({
            "onset_s": t_s[s], "offset_s": t_s[e - 1], "duration_ms": dur_ms,
            "start_x": xi[a], "start_y": yi[a], "end_x": xi[b], "end_y": yi[b],
            "amplitude": float(np.hypot(xi[b] - xi[a], yi[b] - yi[a])),
            "peak_velocity": float(np.nanmax(speed[s:e])),
        })

    fixations = []
    for s, e in _runs(valid & (labels != "sac")):
        dur_ms = (t_s[e - 1] - t_s[s]) * 1000 + _dt_ms(t_s)
        if dur_ms < min_fixation_ms:
            continue
        labels[s:e] = "fix"
        fx, fy = x[s:e], y[s:e]
        fixations.append({
            "onset_s": t_s[s], "offset_s": t_s[e - 1], "duration_ms": dur_ms,
            "x": float(np.nanmean(fx)), "y": float(np.nanmean(fy)),
            "dispersion": float((np.nanmax(fx) - np.nanmin(fx)) + (np.nanmax(fy) - np.nanmin(fy))),
        })

    sac_cols = ["onset_s", "offset_s", "duration_ms", "start_x", "start_y", "end_x", "end_y", "amplitude", "peak_velocity"]
    fix_cols = ["onset_s", "offset_s", "duration_ms", "x", "y", "dispersion"]
    return pd.DataFrame(saccades, columns=sac_cols), pd.DataFrame(fixations, columns=fix_cols), labels


def _dt_ms(t_s: np.ndarray) -> float:
    return float(np.median(np.diff(t_s)) * 1000) if len(t_s) > 1 else 0.0


def summary(saccades: pd.DataFrame, fixations: pd.DataFrame, duration_s: float, unit: str) -> dict:
    out = {
        "unit": unit,
        "saccade_count": int(len(saccades)),
        "saccade_rate_per_s": len(saccades) / duration_s if duration_s else None,
        "fixation_count": int(len(fixations)),
    }
    if len(saccades):
        out.update({
            f"saccade_amplitude_mean_{unit}": float(saccades.amplitude.mean()),
            "saccade_duration_mean_ms": float(saccades.duration_ms.mean()),
            f"saccade_peak_velocity_mean_{unit}_per_s": float(saccades.peak_velocity.mean()),
        })
    if len(fixations):
        out.update({
            "fixation_duration_mean_ms": float(fixations.duration_ms.mean()),
            "fixation_duration_median_ms": float(fixations.duration_ms.median()),
        })
    return out
