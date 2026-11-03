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
                  amplitude_noise_factor: float = 0.0, fill_gap_ms: float = 0.0,
                  duration_bias_samples: float = 0.0, persist_ratio: float = 0.0, persist_ms: float = 50.0,
                  bridge_gap_ms: float = 0.0, exclude=None) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Return (saccades, fixations, per-sample label array: 'fix'|'sac'|'none').

    threshold: a number (units/s) or "adaptive" (Nyström & Holmqvist, k*SD, floored at min_threshold).
    merge_gap_ms: join saccades separated by a shorter gap (post-saccadic wobble split off).
    min_amplitude / amplitude_noise_factor: drop candidates smaller than
        max(min_amplitude, amplitude_noise_factor * RMS-S2S noise) - noise bursts are
        fast but tiny, real saccades move the eye.
    fill_gap_ms: treat runs of missing samples this short (valid samples on both sides) as valid,
        linearly interpolated. Velocity needs unbroken samples, so a single lost frame otherwise hides
        the saccade around it.
    duration_bias_samples: subtract this many sample intervals from saccade durations. Velocity
        smoothing keeps the speed above threshold after the eye has landed, so durations run long.
    persist_ratio: drop a candidate unless the eye stays moved. The net displacement between the median
        position over persist_ms before it and persist_ms after it must be at least this share of its
        amplitude (and above the amplitude floor). A saccade moves the eye to a new place; a detector glitch
        (a lash or glare taken for the pupil for a frame or two) jumps out and back. 0 = off.
    bridge_gap_ms: a run of missing samples longer than fill_gap_ms but at most this long counts as a
        saccade when the eye is somewhere else afterwards, far enough that it must have moved at least at
        the velocity threshold on average (and by more than the amplitude floor). On a low-cost tracker,
        glare often hides large saccades. These rows have `inferred` = True, no duration and no peak
        velocity. 0 = off.
    exclude: samples that are never bridged, such as blinks: the eye can't be followed through a closed lid.
    """
    t_s = np.asarray(t_s, float)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    valid = np.isfinite(x) & np.isfinite(y) if valid is None else (np.asarray(valid, bool) & np.isfinite(x) & np.isfinite(y))
    measured = valid.copy()  # real samples only: the noise estimate ignores filled gaps
    if fill_gap_ms > 0 and len(t_s) > 1:
        max_gap = int(np.floor(fill_gap_ms / _dt_ms(t_s) + 1e-6))
        x, y = x.copy(), y.copy()
        for s, e in _runs(~valid):
            if 0 < s and e < len(valid) and e - s <= max_gap:
                for arr in (x, y):
                    arr[s:e] = np.interp(t_s[s:e], [t_s[s - 1], t_s[e]], [arr[s - 1], arr[e]])
                valid[s:e] = True

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
    noise = rms_s2s(x, y, measured & ~fast)
    amp_floor = max(min_amplitude, amplitude_noise_factor * noise)
    excl = np.zeros(len(t_s), bool) if exclude is None else np.asarray(exclude, bool)
    pw = persist_ms / 1000

    def settled(i0: int, i1: int) -> np.ndarray | None:
        """Median measured position over samples i0..i1-1 (None if fewer than 2)."""
        k = np.arange(max(i0, 0), min(i1, len(t_s)))
        k = k[measured[k]]
        return np.array([np.median(x[k]), np.median(y[k])]) if len(k) >= 2 else None

    saccades = []
    n_spikes = 0
    for s, e in _runs(fast):
        dur_ms = (t_s[e - 1] - t_s[s]) * 1000 + _dt_ms(t_s)
        if dur_ms < min_saccade_ms:
            continue
        dur_ms -= duration_bias_samples * _dt_ms(t_s)
        # amplitude from the samples just outside the fast run (start/end of the movement)
        a, b = max(s - 1, 0), min(e, len(t_s) - 1)
        if not (valid[a] and valid[b]):
            continue  # saccade touching a blink/dropout: amplitude unreliable
        amplitude = float(np.hypot(xi[b] - xi[a], yi[b] - yi[a]))
        if amplitude < amp_floor:
            continue
        if persist_ratio > 0:
            before = settled(int(np.searchsorted(t_s, t_s[a] - pw, "left")), a + 1)
            after = settled(b, int(np.searchsorted(t_s, t_s[b] + pw, "right")))
            if before is not None and after is not None:
                net = float(np.hypot(*(after - before)))
                if net < max(amp_floor, persist_ratio * amplitude):
                    n_spikes += 1
                    continue
        labels[s:e] = "sac"
        saccades.append({
            "onset_s": t_s[s], "offset_s": t_s[e - 1], "duration_ms": dur_ms,
            "start_x": xi[a], "start_y": yi[a], "end_x": xi[b], "end_y": yi[b],
            "amplitude": amplitude,
            "peak_velocity": float(np.nanmax(speed[s:e])), "inferred": False,
        })

    n_bridged = 0
    if bridge_gap_ms > 0 and len(t_s) > 1:
        for s, e in _runs(~valid):
            if s == 0 or e >= len(t_s) or excl[s:e].any():
                continue
            gap_s = t_s[e] - t_s[s - 1]
            if gap_s * 1000 > bridge_gap_ms or (labels[max(s - 2, 0):min(e + 2, len(t_s))] == "sac").any():
                continue
            before = settled(int(np.searchsorted(t_s, t_s[s - 1] - pw, "left")), s)
            after = settled(e, int(np.searchsorted(t_s, t_s[e] + pw, "right")))
            if before is None or after is None:
                continue
            amplitude = float(np.hypot(*(after - before)))
            if amplitude < max(amp_floor, threshold * gap_s):
                continue
            n_bridged += 1
            labels[s:e] = "sac"
            saccades.append({
                "onset_s": t_s[s - 1], "offset_s": t_s[e], "duration_ms": np.nan,
                "start_x": before[0], "start_y": before[1], "end_x": after[0], "end_y": after[1],
                "amplitude": amplitude, "peak_velocity": np.nan, "inferred": True,
            })
        saccades.sort(key=lambda r: r["onset_s"])

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

    sac_cols = ["onset_s", "offset_s", "duration_ms", "start_x", "start_y", "end_x", "end_y", "amplitude",
                "peak_velocity", "inferred"]
    fix_cols = ["onset_s", "offset_s", "duration_ms", "x", "y", "dispersion"]
    sacc_df = pd.DataFrame(saccades, columns=sac_cols)
    sacc_df["inferred"] = sacc_df["inferred"].astype(bool)
    sacc_df.attrs["velocity_threshold"] = float(threshold)
    sacc_df.attrs["noise_rms_s2s"] = noise
    sacc_df.attrs["amplitude_floor"] = amp_floor
    sacc_df.attrs["spikes_rejected"] = n_spikes
    sacc_df.attrs["bridged"] = n_bridged
    return sacc_df, pd.DataFrame(fixations, columns=fix_cols), labels


def step_signal(t_s, x, y, valid=None, half_window_ms: float = 60.0, noise_factor: float = 8.0,
                min_amplitude: float = 1.0) -> dict:
    """The edge-filter signal behind detect_steps(), also used by the viewer.

    step[i] = |median of the next half_window_ms - median of the previous half_window_ms| (trusted samples).
    threshold = max(min_amplitude, noise_factor x noise of one window median), where the noise of a window
    median is 1.25 x the per-sample noise SD / sqrt(samples per window), and the per-sample noise SD is
    estimated robustly from sample-to-sample jumps (median absolute deviation, so saccades don't inflate it).
    """
    t_s = np.asarray(t_s, float)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y) & (np.ones(len(t_s), bool) if valid is None else np.asarray(valid, bool))
    h = half_window_ms / 1000
    dt = float(np.median(np.diff(t_s))) if len(t_s) > 1 else 0.0
    idx = np.flatnonzero(ok)
    tv, xv, yv = t_s[idx], x[idx], y[idx]
    near = np.diff(tv) < 1.5 * dt if dt else np.zeros(max(len(tv) - 1, 0), bool)
    d = np.hypot(np.diff(xv), np.diff(yv))[near]
    sample_sd = 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2) if len(d) else 0.0
    n_win = max(1.0, h / dt) if dt else 1.0
    median_noise = 1.25 * sample_sd / np.sqrt(n_win)
    threshold = max(min_amplitude, noise_factor * median_noise)
    step = np.full(len(t_s), np.nan)
    dx = np.zeros(len(t_s))
    dy = np.zeros(len(t_s))
    lo_b = np.searchsorted(tv, t_s - h, side="left")
    hi_b = np.searchsorted(tv, t_s, side="left")
    lo_a = np.searchsorted(tv, t_s, side="right")
    hi_a = np.searchsorted(tv, t_s + h, side="right")
    for i in range(len(t_s)):
        if hi_b[i] - lo_b[i] >= 2 and hi_a[i] - lo_a[i] >= 2:
            dx[i] = np.median(xv[lo_a[i]:hi_a[i]]) - np.median(xv[lo_b[i]:hi_b[i]])
            dy[i] = np.median(yv[lo_a[i]:hi_a[i]]) - np.median(yv[lo_b[i]:hi_b[i]])
            step[i] = np.hypot(dx[i], dy[i])
    return {"ok": ok, "dt": dt, "step": step, "dx": dx, "dy": dy, "threshold": float(threshold),
            "sample_sd": float(sample_sd), "median_noise": float(median_noise), "samples_per_window": float(n_win),
            "lo_b": lo_b, "hi_b": hi_b}


def detect_steps(t_s, x, y, valid=None, exclude=None, half_window_ms: float = 60.0, noise_factor: float = 8.0,
                 min_amplitude: float = 1.0, min_separation_ms: float = 120.0, min_fixation_ms: float = 60.0
                 ) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Saccades as steps in a staircase: an edge filter built from medians.

    At every sample, step = median position over the next half_window_ms minus median over the previous
    half_window_ms, using trusted samples only. Fixations are flat, so the step is ~0; a saccade moves the
    eye from one level to another, so the step peaks at the saccade and its size is the amplitude. The
    medians ride over short dropouts and ignore spikes, unlike frame-to-frame velocity, which needs
    unbroken samples. A peak counts if its amplitude exceeds max(min_amplitude, noise_factor x the noise of
    a window median); onset / offset are then refined at full frame rate (10% / 90% of the step).

    exclude: samples (e.g. blinks) a step window must not touch, since the eyelid fakes eye movement.
    Trade-off: saccades closer together than ~2 x half_window_ms merge into one, and peak velocity is only
    approximate. Same outputs as detect_events().
    """
    t_s = np.asarray(t_s, float)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    excl = np.zeros(len(t_s), bool) if exclude is None else np.asarray(exclude, bool)
    h = half_window_ms / 1000
    sig = step_signal(t_s, x, y, valid, half_window_ms, noise_factor, min_amplitude)
    ok, dt, step, dx, dy, threshold = sig["ok"], sig["dt"], sig["step"], sig["dx"], sig["dy"], sig["threshold"]
    lo_b, hi_b = sig["lo_b"], sig["hi_b"]
    idx = np.flatnonzero(ok)
    tv, xv, yv = t_s[idx], x[idx], y[idx]

    # peaks: largest first, at least min_separation_ms apart, windows clear of excluded samples
    order = np.argsort(-np.nan_to_num(step))
    taken: list[int] = []
    sep = min_separation_ms / 1000
    for i in order:
        if not np.isfinite(step[i]) or step[i] < threshold:
            break
        if any(abs(t_s[i] - t_s[j]) < sep for j in taken):
            continue
        w = (t_s >= t_s[i] - h) & (t_s <= t_s[i] + h)
        if excl[w].any():
            continue
        taken.append(i)

    labels = np.full(len(t_s), "none", dtype=object)
    saccades = []
    for i in sorted(taken):
        u = np.array([dx[i], dy[i]]) / step[i]
        b0 = (lo_b[i], hi_b[i])
        start = np.array([np.median(xv[slice(*b0)]), np.median(yv[slice(*b0)])])
        end = start + np.array([dx[i], dy[i]])
        w = np.flatnonzero((tv >= t_s[i] - h) & (tv <= t_s[i] + h))
        prog = ((np.column_stack([xv[w], yv[w]]) - start) @ u) / step[i]   # 0 = start level, 1 = end level
        moving = np.flatnonzero((prog > 0.1) & (prog < 0.9))
        if len(moving):
            on_k, off_k = w[moving[0]], w[moving[-1]]
            onset = tv[max(on_k - 1, 0)] if on_k > 0 and tv[on_k] - tv[on_k - 1] < 1.5 * dt else tv[on_k]
            offset = tv[min(off_k + 1, len(tv) - 1)]
        else:   # the jump happened inside a gap or between two frames: centre it on the peak
            onset, offset = t_s[i] - dt / 2, t_s[i] + dt / 2
        seg = (tv >= onset - dt) & (tv <= offset + dt)
        vt, vx, vy = tv[seg], xv[seg], yv[seg]
        steps_ok = np.diff(vt) < 1.5 * dt if len(vt) > 1 else np.zeros(0, bool)
        speeds = (np.hypot(np.diff(vx), np.diff(vy)) / np.maximum(np.diff(vt), 1e-9))[steps_ok]
        labels[(t_s >= onset) & (t_s <= offset)] = "sac"
        saccades.append({"onset_s": onset, "offset_s": offset, "duration_ms": max((offset - onset) * 1000, dt * 1000),
                         "start_x": start[0], "start_y": start[1], "end_x": end[0], "end_y": end[1],
                         "amplitude": float(step[i]),
                         "peak_velocity": float(speeds.max()) if len(speeds) else float(step[i] / max(offset - onset, dt))})

    # fixations: the stretches between saccades (and blinks / exclusions) with enough trusted samples
    fixations = []
    bounds = [t_s[0]] + [v for sc in saccades for v in (sc["onset_s"], sc["offset_s"])] + [t_s[-1] + dt]
    for a, b in zip(bounds[0::2], bounds[1::2]):
        seg = (t_s > a) & (t_s < b) & ok & ~excl
        if not seg.any():
            continue
        dur_ms = (t_s[seg][-1] - t_s[seg][0]) * 1000 + dt * 1000
        if dur_ms < min_fixation_ms or seg.sum() < 0.3 * dur_ms / 1000 / max(dt, 1e-9):
            continue
        labels[seg & (labels != "sac")] = "fix"
        fx, fy = x[seg], y[seg]
        fixations.append({"onset_s": t_s[seg][0], "offset_s": t_s[seg][-1], "duration_ms": dur_ms,
                          "x": float(np.median(fx)), "y": float(np.median(fy)),
                          "dispersion": float((fx.max() - fx.min()) + (fy.max() - fy.min()))})

    sac_cols = ["onset_s", "offset_s", "duration_ms", "start_x", "start_y", "end_x", "end_y", "amplitude", "peak_velocity"]
    fix_cols = ["onset_s", "offset_s", "duration_ms", "x", "y", "dispersion"]
    sacc_df = pd.DataFrame(saccades, columns=sac_cols)
    sacc_df.attrs["velocity_threshold"] = float("nan")
    sacc_df.attrs["amplitude_floor"] = float(threshold)
    sacc_df.attrs["noise_rms_s2s"] = rms_s2s(x, y, ok & (labels != "sac"))
    sacc_df.attrs["method"] = "step"
    return sacc_df, pd.DataFrame(fixations, columns=fix_cols), labels


def event_kwargs(cfg_events: dict) -> dict:
    """detect_events() keyword arguments from the `events` config section (degree units)."""
    keys = ["min_threshold", "adaptive_k", "amplitude_noise_factor", "min_amplitude",
            "min_saccade_ms", "min_fixation_ms", "smoothing_ms", "merge_gap_ms"]
    optional = ["fill_gap_ms", "duration_bias_samples", "persist_ratio", "persist_ms", "bridge_gap_ms"]
    return {"threshold": cfg_events["velocity_threshold"], **{k: cfg_events[k] for k in keys},
            **{k: cfg_events[k] for k in optional if k in cfg_events}}


def step_kwargs(cfg_events: dict) -> dict:
    """detect_steps() keyword arguments from the `events.step` config section."""
    st = cfg_events.get("step", {})
    keys = ["half_window_ms", "noise_factor", "min_amplitude", "min_separation_ms"]
    return {**{k: st[k] for k in keys if k in st}, "min_fixation_ms": cfg_events.get("min_fixation_ms", 60)}


def _dt_ms(t_s: np.ndarray) -> float:
    return float(np.median(np.diff(t_s)) * 1000) if len(t_s) > 1 else 0.0


def summary(saccades: pd.DataFrame, fixations: pd.DataFrame, duration_s: float, unit: str) -> dict:
    out = {
        "unit": unit,
        "saccade_count": int(len(saccades)),
        "saccade_rate_per_s": len(saccades) / duration_s if duration_s else None,
        # saccades inferred across tracking gaps (no duration or peak velocity), and glitches rejected
        "saccades_inferred": int(saccades["inferred"].sum()) if "inferred" in saccades else 0,
        "spikes_rejected": saccades.attrs.get("spikes_rejected", 0),
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
