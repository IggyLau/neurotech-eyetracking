"""Offline processing: raw session folder (or any eye video) -> measurement CSVs + summary."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .calibration.mapping import (PolyMapper, detect_marker_track, target_points, validation_error)
from .metrics import aoi as aoi_mod
from .metrics import blinks as blinks_mod
from .metrics import events as events_mod
from .metrics import pupil_size
from .pupil.detector import DetectorParams, detect_pupil, draw_result
from .pupil.eye_model import EyeModel


# ---------------------------------------------------------------- inputs

def resolve_inputs(path: str | Path) -> dict:
    """Accept a session folder (eye.avi + eye_ts.csv ...) or a bare video file."""
    path = Path(path)
    if path.is_dir():
        return {"dir": path, "eye_video": path / "eye.avi", "eye_ts": path / "eye_ts.csv",
                "scene_video": path / "scene.avi", "scene_ts": path / "scene_ts.csv",
                "markers": path / "markers.csv"}
    out_dir = path.parent / f"{path.stem}_processed"
    out_dir.mkdir(exist_ok=True)
    return {"dir": out_dir, "eye_video": path, "eye_ts": None, "scene_video": None,
            "scene_ts": None, "markers": None}


def load_timestamps(ts_path: Path | None, n_frames: int, fps: float) -> tuple[np.ndarray, int]:
    """Seconds since the first frame, plus the absolute t_ns of frame 0 (0 if unknown)."""
    if ts_path and Path(ts_path).exists():
        ts = pd.read_csv(ts_path)
        t_ns = ts.t_ns.to_numpy(np.int64)[:n_frames]
        return (t_ns - t_ns[0]) / 1e9, int(t_ns[0])
    return np.arange(n_frames) / (fps or 30.0), 0


# ---------------------------------------------------------------- pupil

def detect_video(video: Path, params: DetectorParams, annotate_to: Path | None = None,
                 max_frames: int | None = None, progress: bool = True) -> tuple[pd.DataFrame, list, float]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    writer = None
    rows, ellipses = [], []
    i = 0
    while True:
        ok, img = cap.read()
        if not ok or (max_frames and i >= max_frames):
            break
        r = detect_pupil(img, params)
        rows.append({"frame_index": i, "found": r.found, "cx": r.cx, "cy": r.cy,
                     "axis_minor": r.axis_minor, "axis_major": r.axis_major, "angle": r.angle,
                     "diameter_px": r.diameter_px, "confidence": r.confidence})
        ellipses.append(r.ellipse)
        if annotate_to is not None:
            vis = draw_result(img, r)
            if writer is None:
                writer = cv2.VideoWriter(str(annotate_to), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                         (vis.shape[1], vis.shape[0]))
            writer.write(vis)
        i += 1
        if progress and i % 500 == 0:
            print(f"  pupil detection: {i}/{total or '?'} frames")
    cap.release()
    if writer is not None:
        writer.release()
    return pd.DataFrame(rows), ellipses, fps


def run_pupil_stage(inp: dict, cfg: dict, annotate: bool = False, max_frames: int | None = None) -> tuple[pd.DataFrame, list, int]:
    params = DetectorParams.from_config(cfg["pupil"])
    annotate_to = inp["dir"] / "eye_annotated.mp4" if annotate else None
    df, ellipses, fps = detect_video(inp["eye_video"], params, annotate_to, max_frames)
    t_s, t0_ns = load_timestamps(inp["eye_ts"], len(df), fps)
    df.insert(1, "t_s", t_s[:len(df)])
    df["valid"] = df.found & (df.confidence >= cfg["pupil"]["min_confidence"])
    df["size_outlier"] = size_outliers(df.diameter_px.to_numpy(), df.valid.to_numpy(), fps,
                                       cfg["pupil"].get("max_size_ratio", 0))
    df["valid"] &= ~df.size_outlier
    return df, ellipses, t0_ns


def size_outliers(diameter: np.ndarray, valid: np.ndarray, fps: float, max_ratio: float,
                  window_s: float = 2.0) -> np.ndarray:
    """Frames whose diameter is > max_ratio x (or < 1/max_ratio x) the rolling median of valid frames.

    The pupil cannot change size several-fold between frames, so these are usually the
    iris, an eyelash shadow or an eyelid crease picked instead of the pupil.
    """
    if not max_ratio:
        return np.zeros(len(diameter), bool)
    d = pd.Series(np.where(valid, diameter, np.nan))
    win = max(3, int(window_s * (fps or 30)))
    ref = d.rolling(win, center=True, min_periods=max(3, win // 10)).median().bfill().ffill()
    ratio = np.asarray(diameter) / ref.to_numpy()
    return np.isfinite(ratio) & ((ratio > max_ratio) | (ratio < 1 / max_ratio))


def load_markers(inp: dict, t0_ns: int) -> pd.DataFrame:
    p = inp.get("markers")
    if not p or not Path(p).exists():
        return pd.DataFrame(columns=["t_s", "label"])
    m = pd.read_csv(p)
    m["t_s"] = (m.t_ns - t0_ns) / 1e9
    return m[["t_s", "label"]]


def load_scene_times(inp: dict, t0_ns: int) -> np.ndarray | None:
    p = inp.get("scene_ts")
    if not p or not Path(p).exists():
        return None
    return (pd.read_csv(p).t_ns.to_numpy(np.int64) - t0_ns) / 1e9


# ---------------------------------------------------------------- main entry

def process(path: str | Path, cfg: dict, calibration: str | Path | None = None,
            aois: dict | None = None, annotate: bool = False, overlay: bool = False,
            max_frames: int | None = None) -> dict:
    inp = resolve_inputs(path)
    out = inp["dir"]
    print(f"Processing {inp['eye_video']}")

    pupil, ellipses, t0_ns = run_pupil_stage(inp, cfg, annotate, max_frames)
    t = pupil.t_s.to_numpy()
    duration = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    fs = 1 / np.median(np.diff(t)) if len(t) > 1 else 0.0

    # blinks
    bcfg = cfg["blinks"]
    blinks = blinks_mod.detect_blinks(t, pupil.valid.to_numpy(), bcfg["min_duration_ms"], bcfg["max_duration_ms"],
                                      bcfg.get("merge_gap_ms", 0), pupil.confidence.to_numpy(),
                                      bcfg.get("max_blink_confidence", 1.0))
    in_blink = blinks_mod.blink_mask(t, blinks, bcfg["pad_ms"])
    pupil["in_blink"] = in_blink
    good = pupil.valid.to_numpy() & ~in_blink

    # pupil size
    pupil["diameter_clean_px"] = pupil_size.clean_diameter(
        t, pupil.diameter_px.to_numpy(), ~good, cfg["pupil_size"]["smoothing_ms"])

    # eyeball model -> rotation angles in degrees
    eye_model = None
    try:
        eye_model = EyeModel.fit(ellipses, pupil.confidence.to_numpy())
        yaw, pitch = eye_model.angles_deg(pupil.cx.to_numpy(), pupil.cy.to_numpy())
        pupil["yaw_deg"], pupil["pitch_deg"] = yaw, pitch
    except ValueError as e:
        print(f"  eye model not fitted: {e}")

    # fixations / saccades (degrees if we have the eye model, else eye-image pixels)
    ev_kwargs = events_mod.event_kwargs(cfg["events"])
    if eye_model is not None:
        ex, ey, unit = pupil.yaw_deg, pupil.pitch_deg, "deg"
    else:
        ex, ey, unit = pupil.cx, pupil.cy, "px"
        ev_kwargs.update(threshold=cfg["events"]["velocity_threshold_px"], min_amplitude=0)
    saccades, fixations, labels = events_mod.detect_events(t, ex.to_numpy(), ey.to_numpy(), good, **ev_kwargs)
    pupil["event"] = labels

    markers = load_markers(inp, t0_ns)

    # pupil change after each marker (light reflex, task-evoked dilation)
    pcfg = cfg["pupil_size"]
    epochs, epoch_summary = pupil_size.marker_epochs(
        t, pupil.diameter_clean_px.to_numpy(), markers, pcfg.get("epoch_labels", []),
        pcfg["baseline_ms"], pcfg.get("epoch_max_s", 30))

    # gaze in scene coordinates
    gaze = None
    scene_t = load_scene_times(inp, t0_ns)
    if calibration:
        mapper = PolyMapper.load(calibration)
        gx, gy = mapper.predict(pupil.cx, pupil.cy)
        gaze = pd.DataFrame({"t_s": t, "gaze_x": np.where(good, gx, np.nan), "gaze_y": np.where(good, gy, np.nan)})
        if scene_t is not None:
            gaze["scene_frame"] = np.clip(np.searchsorted(scene_t, t), 0, len(scene_t) - 1)
        gaze.to_csv(out / "gaze.csv", index=False)
        # fixation positions in scene px for AOIs / heatmap
        fx_x, fx_y = mapper.predict(*_fixation_pupil_xy(pupil, fixations))
        fixations["scene_x"], fixations["scene_y"] = fx_x, fx_y

    summary = {
        "source": str(inp["eye_video"]),
        "frames": int(len(pupil)),
        "duration_s": duration,
        "sample_rate_hz": float(fs),
        "detection_rate": float(pupil.found.mean()) if len(pupil) else 0.0,
        "valid_rate": float(pupil.valid.mean()) if len(pupil) else 0.0,
        "mean_confidence": float(pupil.confidence.mean()) if len(pupil) else 0.0,
        "blink_count": int((blinks.kind == "blink").sum()) if len(blinks) else 0,
        "blink_rate_per_min": float((blinks.kind == "blink").sum() / duration * 60) if duration and len(blinks) else 0.0,
        "eye_model": eye_model.to_dict() if eye_model else None,
        **pupil_size.summary(pupil.diameter_clean_px.to_numpy()),
        **events_mod.summary(saccades, fixations, duration, unit),
        "pupil_epochs": epoch_summary.to_dict(orient="records"),
    }

    if aois and calibration:
        fx = fixations.rename(columns={"x": "eye_x", "y": "eye_y", "scene_x": "x", "scene_y": "y"})
        aoi_table = aoi_mod.aoi_metrics(fx, aois)
        aoi_table.to_csv(out / "aoi.csv", index=False)
        summary["aoi"] = aoi_table.to_dict(orient="records")

    pupil.to_csv(out / "pupil.csv", index=False)
    blinks.to_csv(out / "blinks.csv", index=False)
    saccades.to_csv(out / "saccades.csv", index=False)
    fixations.to_csv(out / "fixations.csv", index=False)
    if len(markers):
        markers.to_csv(out / "markers_s.csv", index=False)
    if len(epoch_summary):
        epochs.to_csv(out / "pupil_epochs.csv", index=False)
        epoch_summary.to_csv(out / "pupil_epochs_summary.csv", index=False)
    (out / "metrics_summary.json").write_text(json.dumps(summary, indent=2, default=_json_default))

    from .viz import plots
    plots.timeseries_report(pupil, saccades, blinks, markers, out / "report.png", unit)
    plots.epochs_report(epochs, out / "pupil_epochs.png")
    if gaze is not None and inp["scene_video"] and Path(inp["scene_video"]).exists():
        plots.scene_heatmap(inp["scene_video"], fixations, out / "heatmap.png")
        if overlay:
            plots.gaze_overlay_video(inp["scene_video"], gaze, out / "overlay.mp4")

    print(f"  wrote results to {out}")
    _print_summary(summary)
    return summary


def _fixation_pupil_xy(pupil: pd.DataFrame, fixations: pd.DataFrame):
    xs, ys = [], []
    for f in fixations.itertuples():
        sel = (pupil.t_s >= f.onset_s) & (pupil.t_s <= f.offset_s) & pupil.valid
        xs.append(pupil.cx[sel].mean())
        ys.append(pupil.cy[sel].mean())
    return np.array(xs), np.array(ys)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    return str(o)


def _print_summary(s: dict):
    keys = ["frames", "duration_s", "sample_rate_hz", "detection_rate", "valid_rate", "blink_count",
            "pupil_diameter_mean_px", "saccade_count", "fixation_count", "fixation_duration_mean_ms"]
    for k in keys:
        v = s.get(k)
        if v is not None:
            print(f"    {k:28s} {v:.3f}" if isinstance(v, float) else f"    {k:28s} {v}")
    for e in s.get("pupil_epochs", []):
        print(f"    epoch {e['event']} {e['label']:<11s} at {e['onset_s']:6.1f}s  baseline {e['baseline_px']:.1f}px  "
              f"mean {e['mean_change_pct']:+.1f}%  min {e['min_change_pct']:+.1f}%  max {e['max_change_pct']:+.1f}%")


# ---------------------------------------------------------------- calibration

def calibrate_from_session(session: str | Path, cfg: dict) -> dict:
    """Fit pupil->scene mapping from a session recorded with `eyetrack calibrate`."""
    inp = resolve_inputs(session)
    out = inp["dir"]
    pupil, _, t0_ns = run_pupil_stage(inp, cfg)
    markers = load_markers(inp, t0_ns)
    scene_t = load_scene_times(inp, t0_ns)
    track = detect_marker_track(inp["scene_video"], cfg["calibration"]["aruco_dict"])
    track["t_s"] = scene_t[:len(track)]

    cal = target_points(markers, pupil, track, "cal")
    val = target_points(markers, pupil, track, "val")
    cal.to_csv(out / "calibration_points.csv", index=False)
    if len(cal) < 6:
        raise RuntimeError(f"Only {len(cal)} usable calibration targets (need 6+). Check that the scene "
                           "camera sees the marker and the pupil is detected; see calibration_points.csv.")
    mapper = PolyMapper.fit(cal[["pupil_x", "pupil_y"]].to_numpy(), cal[["scene_x", "scene_y"]].to_numpy(),
                            cfg["calibration"]["poly_degree"])
    mapper.save(out / "calibration.json")

    scene_w = cfg["scene_camera"]["width"]
    hfov = cfg["calibration"]["scene_hfov_deg"]
    result = {"calibration_targets": len(cal), "calibration_file": str(out / "calibration.json")}
    fit_err = validation_error(mapper, cal, scene_w, hfov)
    result["fit_error_deg_mean"] = float(fit_err.error_deg.mean())
    if len(val):
        v = validation_error(mapper, val, scene_w, hfov)
        v.to_csv(out / "validation.csv", index=False)
        result.update({"validation_targets": len(v), "validation_error_deg_mean": float(v.error_deg.mean()),
                       "validation_error_px_mean": float(v.error_px.mean())})
    (out / "calibration_summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result
