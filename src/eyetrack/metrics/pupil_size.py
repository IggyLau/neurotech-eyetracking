"""Pupil-diameter preprocessing: blink removal, interpolation, smoothing, baseline correction.

Follows common pupillometry practice (e.g. Mathôt et al. 2018): remove blinks and
their edges, linearly interpolate short gaps, low-pass, then express changes
relative to a pre-event baseline. Diameter is in eye-camera pixels; it is only
comparable within one recording with the glasses unmoved and lighting constant.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def clean_diameter(t_s: np.ndarray, diameter: np.ndarray, bad: np.ndarray,
                   smoothing_ms: float = 50, max_gap_ms: float = 500) -> np.ndarray:
    t_s = np.asarray(t_s, float)
    d = np.asarray(diameter, float).copy()
    d[np.asarray(bad, bool)] = np.nan

    # reject implausibly fast diameter changes: speed > median + 16*MAD (Kret & Sjak-Shie 2019)
    speed = np.abs(np.diff(d, prepend=np.nan)) / np.diff(t_s, prepend=np.nan)
    if np.isfinite(speed).sum() > 10:
        med = np.nanmedian(speed)
        mad = np.nanmedian(np.abs(speed - med))
        if mad > 0:
            d[speed > med + 16 * mad] = np.nan

    s = pd.Series(d, index=pd.Index(t_s))
    fs = 1 / np.median(np.diff(t_s)) if len(t_s) > 1 else 1
    limit = max(1, int(max_gap_ms / 1000 * fs))
    s = s.interpolate(method="index", limit=limit, limit_area="inside")
    win = max(1, int(round(smoothing_ms / 1000 * fs)))
    if win > 1:
        s = s.rolling(win, center=True, min_periods=1).mean()
    return s.to_numpy()


def baseline_correct(t_s: np.ndarray, diameter: np.ndarray, events_s: list[float],
                     baseline_ms: float = 500, window_s: float = 5.0) -> pd.DataFrame:
    """Epoch around each event: subtractive baseline correction (diameter - mean of pre-event baseline)."""
    t_s = np.asarray(t_s, float)
    rows = []
    for k, ev in enumerate(events_s):
        base = (t_s >= ev - baseline_ms / 1000) & (t_s < ev)
        b = np.nanmean(diameter[base]) if base.any() else np.nan
        sel = (t_s >= ev - baseline_ms / 1000) & (t_s <= ev + window_s)
        for t, d in zip(t_s[sel], diameter[sel]):
            rows.append({"event": k, "t_rel_s": t - ev, "diameter": d, "change": d - b,
                         "change_pct": 100 * (d - b) / b if b else np.nan})
    return pd.DataFrame(rows)


def marker_epochs(t_s: np.ndarray, diameter: np.ndarray, markers: pd.DataFrame, labels: list[str],
                  baseline_ms: float = 500, max_window_s: float = 30.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Baseline-corrected epochs after each marker whose label is in `labels`.

    Each epoch runs until the next marker (of any label) or max_window_s, so a
    task_start epoch covers the task and a lights_on epoch covers the light period.
    Returns (long table of samples, one summary row per epoch).
    """
    t_s = np.asarray(t_s, float)
    diameter = np.asarray(diameter, float)
    m = markers.sort_values("t_s").reset_index(drop=True)
    samples, rows = [], []
    for i, mk in m.iterrows():
        if mk.label not in labels or not (t_s[0] < mk.t_s < t_s[-1]):
            continue
        end = m.t_s[i + 1] if i + 1 < len(m) else t_s[-1]
        window = float(min(max_window_s, end - mk.t_s))
        ep = baseline_correct(t_s, diameter, [mk.t_s], baseline_ms, window)
        k = len(rows)
        ep["event"], ep["label"] = k, mk.label
        samples.append(ep)
        post = ep[ep.t_rel_s >= 0]
        pct = post.change_pct.to_numpy()
        ok = np.isfinite(pct)
        base = ep[ep.t_rel_s < 0].diameter
        rows.append({
            "event": k, "label": mk.label, "onset_s": float(mk.t_s), "window_s": window,
            "baseline_px": float(base.mean()) if base.notna().any() else np.nan,
            "mean_change_pct": float(pct[ok].mean()) if ok.any() else np.nan,
            "min_change_pct": float(pct[ok].min()) if ok.any() else np.nan,
            "t_min_s": float(post.t_rel_s.to_numpy()[ok][pct[ok].argmin()]) if ok.any() else np.nan,
            "max_change_pct": float(pct[ok].max()) if ok.any() else np.nan,
            "t_max_s": float(post.t_rel_s.to_numpy()[ok][pct[ok].argmax()]) if ok.any() else np.nan,
            "valid_fraction": float(ok.mean()) if len(ok) else 0.0,
        })
    samples = pd.concat(samples, ignore_index=True) if samples else pd.DataFrame(
        columns=["event", "label", "t_rel_s", "diameter", "change", "change_pct"])
    return samples, pd.DataFrame(rows)


def summary(diameter: np.ndarray) -> dict:
    d = np.asarray(diameter, float)
    return {"pupil_diameter_mean_px": float(np.nanmean(d)) if np.isfinite(d).any() else None,
            "pupil_diameter_sd_px": float(np.nanstd(d)) if np.isfinite(d).any() else None,
            "pupil_valid_fraction": float(np.isfinite(d).mean()) if len(d) else 0.0}
