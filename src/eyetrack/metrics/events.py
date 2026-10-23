"""Fixation / saccade detection with a velocity threshold (I-VT, Salvucci & Goldberg 2000).

Positions can be eye-rotation angles in degrees (from EyeModel), calibrated
scene gaze in degrees, or raw pupil pixels; the threshold must match the unit.

Sampling-rate limits for the OV9281 at ~120 Hz: saccades (20-200 ms) span
3-25 samples, so detection and amplitude are fine, but peak velocity is
underestimated for small saccades. Microsaccades (<1 deg, ~10-20 ms) need
>=200 Hz plus a lot less pupil-center noise than a DIY rig usually gives.
For a second opinion, run pymovements or REMoDNaV on gaze.csv (`pip install .[compare]`).

Threshold: a fixed 30 deg/s works on clean lab data, but velocity noise grows with
sampling rate x position noise, so on a noisy DIY tracker a fixed threshold turns
noise into "saccades". threshold="adaptive" uses the data-driven peak-velocity
threshold of Nyström & Holmqvist (2010): iterate PT = mean + 6*SD of all velocity
samples below PT until it converges, never going below `min_threshold`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from .blinks import _runs


def velocity(t_s: np.ndarray, x: np.ndarray, y: np.ndarray, smoothing_ms: float = 20.0) -> np.ndarray:
    """Speed (units/s) from a Savitzky-Golay smoothed position signal (Nyström & Holmqvist 2010)."""
    t_s = np.asarray(t_s, float)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    fs = 1 / np.median(np.diff(t_s)) if len(t_s) > 1 else 1.0
    win = int(round(smoothing_ms / 1000 * fs))
    win = win + 1 if win % 2 == 0 else win
    if win >= 5 and len(x) > win:
        x = savgol_filter(x, win, 2, mode="nearest")
        y = savgol_filter(y, win, 2, mode="nearest")
    return np.hypot(np.gradient(x, t_s), np.gradient(y, t_s))


def adaptive_threshold(speed: np.ndarray, start: float | None = None, k: float = 6.0,
                       min_threshold: float = 0.0, max_iter: int = 50) -> float:
    v = speed[np.isfinite(speed)]
    if len(v) < 10:
        return max(start or 0.0, min_threshold)
    pt = start if start is not None else float(np.percentile(v, 99))
    for _ in range(max_iter):
        below = v[v < pt]
        if len(below) < 10:
            break
        new = float(below.mean() + k * below.std())
        if abs(new - pt) < 0.01 * pt:
            pt = new
            break
        pt = new
    return max(pt, min_threshold)


def rms_s2s(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    """Precision as RMS of sample-to-sample distance over `mask` samples (standard eye-tracker precision)."""
    d = np.hypot(np.diff(x), np.diff(y))
    m = mask[1:] & mask[:-1]
    d = d[m & np.isfinite(d)]
    return float(np.sqrt(np.mean(d ** 2))) if len(d) else 0.0


def detect_events(t_s, x, y, valid=None, threshold: float | str = "adaptive", min_saccade_ms: float = 10,
                  min_fixation_ms: float = 60, smoothing_ms: float = 20.0, min_threshold: float = 30.0,
                  merge_gap_ms: float = 0.0, adaptive_k: float = 6.0, min_amplitude: float = 0.0,
                  amplitude_noise_factor: float = 0.0) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Return (saccades, fixations, per-sample label array: 'fix'|'sac'|'none').

    threshold: a number (units/s) or "adaptive" (Nyström & Holmqvist, k*SD, floored at min_threshold).
    merge_gap_ms: join saccades separated by a shorter gap (post-saccadic wobble split off).
    min_amplitude / amplitude_noise_factor: drop candidates smaller than
        max(min_amplitude, amplitude_noise_factor * RMS-S2S noise) - noise bursts are
        fast but tiny, real saccades move the eye.
    """
    t_s = np.asarray(t_s, float)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    valid = np.isfinite(x) & np.isfinite(y) if valid is None else (np.asarray(valid, bool) & np.isfinite(x) & np.isfinite(y))

    # interpolate tiny gaps so the filter / gradient don't smear NaNs; masked out again below
    xi = pd.Series(np.where(valid, x, np.nan)).interpolate(limit_direction="both").to_numpy()
    yi = pd.Series(np.where(valid, y, np.nan)).interpolate(limit_direction="both").to_numpy()
    speed = velocity(t_s, xi, yi, smoothing_ms)
    speed[~valid] = np.nan
    if threshold == "adaptive":
        threshold = adaptive_threshold(speed, k=adaptive_k, min_threshold=min_threshold)
    elif isinstance(threshold, str):
        raise ValueError(f"threshold must be a number or 'adaptive', got {threshold!r}")

    labels = np.full(len(t_s), "none", dtype=object)
    fast = valid & (speed > threshold)
    if merge_gap_ms > 0:
        gap = max(1, int(round(merge_gap_ms / _dt_ms(t_s))))
        for s, e in _runs(~fast):
            if 0 < s and e < len(fast) and e - s <= gap and valid[s:e].all():
                fast[s:e] = True
    noise = rms_s2s(x, y, valid & ~fast)
    amp_floor = max(min_amplitude, amplitude_noise_factor * noise)
    saccades = []
    for s, e in _runs(fast):
        dur_ms = (t_s[e - 1] - t_s[s]) * 1000 + _dt_ms(t_s)
        if dur_ms < min_saccade_ms:
            continue
        # amplitude from the samples just outside the fast run (start/end of the movement)
        a, b = max(s - 1, 0), min(e, len(t_s) - 1)
        if not (valid[a] and valid[b]):
            continue  # saccade touching a blink/dropout: amplitude unreliable
        amplitude = float(np.hypot(xi[b] - xi[a], yi[b] - yi[a]))
        if amplitude < amp_floor:
            continue
        labels[s:e] = "sac"
        saccades.append({
            "onset_s": t_s[s], "offset_s": t_s[e - 1], "duration_ms": dur_ms,
            "start_x": xi[a], "start_y": yi[a], "end_x": xi[b], "end_y": yi[b],
            "amplitude": amplitude,
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
    sacc_df = pd.DataFrame(saccades, columns=sac_cols)
    sacc_df.attrs["velocity_threshold"] = float(threshold)
    sacc_df.attrs["noise_rms_s2s"] = noise
    sacc_df.attrs["amplitude_floor"] = amp_floor
    return sacc_df, pd.DataFrame(fixations, columns=fix_cols), labels


def event_kwargs(cfg_events: dict) -> dict:
    """detect_events() keyword arguments from the `events` config section (degree units)."""
    keys = ["min_threshold", "adaptive_k", "amplitude_noise_factor", "min_amplitude",
            "min_saccade_ms", "min_fixation_ms", "smoothing_ms", "merge_gap_ms"]
    return {"threshold": cfg_events["velocity_threshold"], **{k: cfg_events[k] for k in keys}}


def _dt_ms(t_s: np.ndarray) -> float:
    return float(np.median(np.diff(t_s)) * 1000) if len(t_s) > 1 else 0.0


def summary(saccades: pd.DataFrame, fixations: pd.DataFrame, duration_s: float, unit: str) -> dict:
    out = {
        "unit": unit,
        "saccade_count": int(len(saccades)),
        "saccade_rate_per_s": len(saccades) / duration_s if duration_s else None,
        "fixation_count": int(len(fixations)),
        # precision of the eye signal (RMS sample-to-sample); compare with docs/benchmarks.md noise levels
        f"noise_rms_s2s_{unit}": saccades.attrs.get("noise_rms_s2s"),
        f"velocity_threshold_{unit}_per_s": saccades.attrs.get("velocity_threshold"),
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
