"""Offline processing: raw session folder (or any eye video) -> measurement CSVs + summary."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .calibration.mapping import PolyMapper, detect_all_markers, poly_features, target_points
from .calibration.scene import (Calibration, JeoMapper, align_samples, camera_angular_error_degrees,
                                combine_eyes)
from .metrics import aoi as aoi_mod
from .metrics import blinks as blinks_mod
from .metrics import events as events_mod
from .metrics import pupil_size
from .pupil.detector import DetectorParams, detect_pupil, draw_result
from .pupil.eye_model import EyeModel, fallback_model, radius_warning


# ---------------------------------------------------------------- inputs

def resolve_inputs(path: str | Path, eye: str = "eye") -> dict:
    """Accept a session folder (eye.avi + eye_ts.csv ...) or a bare video file.

    eye="eye2" points the eye video at the second eye camera (eye2.avi) and writes results to <session>/eye2/.
    """
    path = Path(path)
    if path.is_dir():
        inp = {"dir": path, "session": path, "role": eye,
               "eye_video": path / "eye.avi", "eye_ts": path / "eye_ts.csv",
               "eye2_video": path / "eye2.avi", "eye2_ts": path / "eye2_ts.csv",
               "scene_video": path / "scene.avi", "scene_ts": path / "scene_ts.csv",
               "markers": path / "markers.csv"}
        if eye == "eye2":
            inp.update(dir=path / "eye2", eye_video=path / "eye2.avi", eye_ts=path / "eye2_ts.csv")
            inp["dir"].mkdir(exist_ok=True)
        return inp
    out_dir = path.parent / f"{path.stem}_processed"
    out_dir.mkdir(exist_ok=True)
    return {"dir": out_dir, "session": out_dir, "role": "eye", "eye_video": path, "eye_ts": None,
            "eye2_video": None, "eye2_ts": None, "scene_video": None, "scene_ts": None, "markers": None}


def load_timestamps(ts_path: Path | None, n_frames: int, fps: float) -> tuple[np.ndarray, int]:
    """Seconds since the first frame, plus the absolute t_ns of frame 0 (0 if unknown)."""
    if ts_path and Path(ts_path).exists():
        ts = pd.read_csv(ts_path)
        t_ns = ts.t_ns.to_numpy(np.int64)[:n_frames]
        return (t_ns - t_ns[0]) / 1e9, int(t_ns[0])
    return np.arange(n_frames) / (fps or 30.0), 0


# ---------------------------------------------------------------- pupil

def brightness_features(img: np.ndarray, r, eye_xy: tuple[int, int] | None) -> tuple[float, float, tuple | None]:
    """(frame_brightness, eye_brightness, updated eye_xy) for one frame; shared with the real-time path.

    Brightness is independent of the detector: a blink brightens the eye area (lid skin covers the dark
    pupil and iris); a lighting problem changes the whole frame. eye_xy is the last confident pupil
    centre, where the eye-area brightness is measured.
    """
    gray = img[:, :, 0] if img.ndim == 3 else img
    if r.found and r.confidence >= 0.5:
        eye_xy = (int(r.cx), int(r.cy))
    ex, ey = eye_xy or (gray.shape[1] // 2, gray.shape[0] // 2)
    return (float(gray[::4, ::4].mean()),
            float(gray[max(0, ey - 50):ey + 50, max(0, ex - 80):ex + 80].mean()), eye_xy)


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
    eye_xy = None   # last confident pupil centre: where the eye-area brightness is measured
    while True:
        ok, img = cap.read()
        if not ok or (max_frames and i >= max_frames):
            break
        r = detect_pupil(img, params)
        frame_b, eye_b, eye_xy = brightness_features(img, r, eye_xy)
        rows.append({"frame_index": i, "found": r.found, "cx": r.cx, "cy": r.cy,
                     "axis_minor": r.axis_minor, "axis_major": r.axis_major, "angle": r.angle,
                     "diameter_px": r.diameter_px, "confidence": r.confidence,
                     "frame_brightness": frame_b, "eye_brightness": eye_b})
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


def fit_eye_model(ellipses: list, pupil: pd.DataFrame, good: np.ndarray, em_cfg: dict) -> tuple[EyeModel, str | None]:
    """Eyeball model for one eye, with the configured fixed radius or the implausible-radius fallback.

    Raises ValueError when there are too few usable ellipses.
    """
    eye_model = EyeModel.fit(ellipses, pupil.confidence.to_numpy(), radius=em_cfg.get("radius_px"))
    pupil_d = float(pupil.axis_major[good].median())
    warning = radius_warning(eye_model.radius, pupil_d, tuple(em_cfg.get("plausible_radius", (1.5, 6.0))))
    if warning and eye_model.radius_source == "fitted" and em_cfg.get("fallback_ratio"):
        eye_model = fallback_model(eye_model, pupil_d, em_cfg["fallback_ratio"],
                                   pupil.cx[good].to_numpy(), pupil.cy[good].to_numpy())
        warning += (f" Using {eye_model.radius:.0f} px instead ({em_cfg['fallback_ratio']} x the pupil "
                    "diameter); sizes and speeds in degrees are approximate.")
    return eye_model, warning


def first_t_ns(ts_path) -> int | None:
    if ts_path and Path(ts_path).exists():
        ts = pd.read_csv(ts_path, nrows=1)
        return int(ts.t_ns.iloc[0]) if len(ts) else None
    return None


# ---------------------------------------------------------------- main entry

def process(path: str | Path, cfg: dict, calibration: str | Path | None = None,
            aois: dict | None = None, annotate: bool = False, overlay: bool = False,
            max_frames: int | None = None, eye: str = "eye") -> dict:
    """Process one session. With a second eye camera (eye2.avi) that eye is processed first, into
    <session>/eye2/, and its gaze is combined with the first eye's when a calibration is given."""
    inp = resolve_inputs(path, eye)
    out = inp["dir"]
    if eye == "eye" and inp["eye2_video"] and Path(inp["eye2_video"]).exists():
        print("Second eye camera found: processing eye2.avi first")
        process(path, cfg, None, None, annotate, False, max_frames, eye="eye2")
    print(f"Processing {inp['eye_video']}")

    pupil, ellipses, t0_ns = run_pupil_stage(inp, cfg, annotate, max_frames)
    t = pupil.t_s.to_numpy()
    duration = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    fs = 1 / np.median(np.diff(t)) if len(t) > 1 else 0.0

    # blinks
    bcfg = cfg["blinks"]
    if bcfg.get("method", "brightness") == "brightness" and "eye_brightness" in pupil:
        # blink vs dropout from image brightness (independent of the pupil detector)
        blinks = blinks_mod.classify_dropouts(
            t, pupil.valid.to_numpy(), pupil.frame_brightness.to_numpy(), pupil.eye_brightness.to_numpy(),
            bcfg["min_duration_ms"], bcfg["max_duration_ms"], bcfg.get("merge_gap_ms", 0),
            bcfg.get("lighting_drop", 0.08), bcfg.get("eyelid_rise", 3.0))
    else:
        blinks = blinks_mod.detect_blinks(t, pupil.valid.to_numpy(), bcfg["min_duration_ms"], bcfg["max_duration_ms"],
                                          bcfg.get("merge_gap_ms", 0), pupil.confidence.to_numpy(),
                                          bcfg.get("max_blink_confidence", 1.0))
    in_blink = blinks_mod.blink_mask(t, blinks, bcfg["pad_ms"])
    pupil["in_blink"] = in_blink
    # per frame: why there is no trusted pupil (blank when there is one)
    kind = np.full(len(t), "", dtype=object)
    for ev in blinks.itertuples():
        if ev.kind != "camera_gap":
            kind[ev.start_index:ev.end_index] = ev.kind
    kind[pupil.valid.to_numpy()] = ""
    pupil["dropout_kind"] = kind
    good = pupil.valid.to_numpy() & ~in_blink

    # pupil size
    pupil["diameter_clean_px"] = pupil_size.clean_diameter(
        t, pupil.diameter_px.to_numpy(), ~good, cfg["pupil_size"]["smoothing_ms"])

    # eyeball model -> rotation angles in degrees
    eye_model, eye_model_warning = None, None
    em_cfg = cfg.get("eye_model") or {}
    try:
        eye_model, eye_model_warning = fit_eye_model(ellipses, pupil, good, em_cfg)
        if eye_model_warning:
            print(f"  WARNING: {eye_model_warning}")
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
    if cfg["events"].get("method", "ivt") == "step":
        st_kwargs = events_mod.step_kwargs(cfg["events"])
        if unit == "px":
            st_kwargs["min_amplitude"] = cfg["events"].get("step", {}).get("min_amplitude_px", 2.0)
        saccades, fixations, labels = events_mod.detect_steps(t, ex.to_numpy(), ey.to_numpy(), pupil.valid.to_numpy(),
                                                              exclude=in_blink, **st_kwargs)
    else:
        saccades, fixations, labels = events_mod.detect_events(t, ex.to_numpy(), ey.to_numpy(), good,
                                                               exclude=in_blink, **ev_kwargs)
    pupil["event"] = labels

    markers = load_markers(inp, t0_ns)

    # pupil change after each marker (light reflex, task-evoked dilation)
    pcfg = cfg["pupil_size"]
    epochs, epoch_summary = pupil_size.marker_epochs(
        t, pupil.diameter_clean_px.to_numpy(), markers, pcfg.get("epoch_labels", []),
        pcfg["baseline_ms"], pcfg.get("epoch_max_s", 30), tuple(pcfg.get("epoch_end_ignore", [])))

    # gaze in scene coordinates (both eyes combined when there are two)
    gaze = None
    scene_t = load_scene_times(inp, t0_ns)
    if calibration and eye == "eye":
        cal = Calibration.load(calibration)
        gaze = scene_gaze(cal, inp, pupil, good, t0_ns, cfg["calibration"].get("method"))
        if scene_t is not None:
            gaze["scene_frame"] = np.clip(np.searchsorted(scene_t, t), 0, len(scene_t) - 1)
        gaze.to_csv(out / "gaze.csv", index=False)
        # fixation positions in scene px for AOIs / heatmap
        fixations["scene_x"], fixations["scene_y"] = _fixation_scene_xy(gaze, fixations)

    summary = {
        "source": str(inp["eye_video"]),
        "eye": eye,
        "frames": int(len(pupil)),
        "duration_s": duration,
        "sample_rate_hz": float(fs),
        "detection_rate": float(pupil.found.mean()) if len(pupil) else 0.0,
        "valid_rate": float(pupil.valid.mean()) if len(pupil) else 0.0,
        "mean_confidence": float(pupil.confidence.mean()) if len(pupil) else 0.0,
        "blink_count": int((blinks.kind == "blink").sum()) if len(blinks) else 0,
        "blink_rate_per_min": float((blinks.kind == "blink").sum() / duration * 60) if duration and len(blinks) else 0.0,
        # why frames lacked a trusted pupil, as a fraction of all frames (signal-quality diagnosis)
        "missing_frames": {k: float((pupil.dropout_kind == k).mean()) for k in ("blink", "closed", "lighting", "lost",
                                                                              "dropout", "low_confidence", "closed_or_lost")
                           if (pupil.dropout_kind == k).any()},
        "camera_gaps": int((blinks.kind == "camera_gap").sum()) if len(blinks) else 0,
        "eye_model": eye_model.to_dict() if eye_model else None,
        "eye_model_warning": eye_model_warning,
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
        eyes = overlay_eyes(inp, pupil, eye_model, cal, t0_ns)
        snap = plots.gaze_overlay_snapshot(inp["scene_video"], gaze, out / "overlay_snapshot.png", eyes)
        if snap:
            summary["overlay_snapshot"] = str(snap)
        if overlay:
            plots.gaze_overlay_video(inp["scene_video"], gaze, out / "overlay.mp4", eyes)

    print(f"  wrote results to {out}")
    _print_summary(summary)
    return summary


def _fixation_scene_xy(gaze: pd.DataFrame, fixations: pd.DataFrame):
    xs, ys = [], []
    for f in fixations.itertuples():
        sel = (gaze.t_s >= f.onset_s) & (gaze.t_s <= f.offset_s)
        xs.append(gaze.gaze_x[sel].median() if sel.any() else np.nan)
        ys.append(gaze.gaze_y[sel].median() if sel.any() else np.nan)
    return np.array(xs, float), np.array(ys, float)


def _second_eye(inp: dict, t0_ns: int) -> pd.DataFrame | None:
    """The second eye's processed pupil.csv, with t_s moved onto the first eye's clock."""
    p = Path(inp["session"]) / "eye2" / "pupil.csv"
    t0_2 = first_t_ns(inp.get("eye2_ts"))
    if not p.exists() or t0_2 is None:
        return None
    p2 = pd.read_csv(p)
    p2["t_s"] = p2.t_s + (t0_2 - t0_ns) / 1e9
    return p2


def scene_gaze(cal: Calibration, inp: dict, pupil: pd.DataFrame, good: np.ndarray, t0_ns: int,
               method: str | None = None) -> pd.DataFrame:
    """Gaze in scene-camera px per frame of the first eye: each eye's point, and the two combined."""
    t = pupil.t_s.to_numpy()
    g1 = np.stack(cal.predict("eye", pupil.cx.to_numpy(), pupil.cy.to_numpy(), method), axis=1)
    g1[~good] = np.nan
    gaze = pd.DataFrame({"t_s": t})
    combined = g1
    p2 = _second_eye(inp, t0_ns) if "eye2" in cal.eyes else None
    if p2 is not None:
        good2 = (p2.valid & ~p2.in_blink).to_numpy()
        g2 = np.stack(cal.predict("eye2", p2.cx.to_numpy(), p2.cy.to_numpy(), method), axis=1)
        g2[~good2] = np.nan
        dt = float(np.median(np.diff(p2.t_s))) if len(p2) > 1 else 0.01
        g2 = align_samples(p2.t_s.to_numpy(), g2, t, max_dt=1.5 * dt)
        combined = combine_eyes(g1, g2)
        gaze["gaze_eye2_x"], gaze["gaze_eye2_y"] = g2[:, 0], g2[:, 1]
    gaze.insert(1, "gaze_x", combined[:, 0])
    gaze.insert(2, "gaze_y", combined[:, 1])
    gaze.insert(3, "gaze_eye_x", g1[:, 0])
    gaze.insert(4, "gaze_eye_y", g1[:, 1])
    return gaze


def overlay_eyes(inp: dict, pupil: pd.DataFrame, eye_model, cal: Calibration, t0_ns: int) -> list[dict]:
    """Eye videos, pupil tracks and eyeball models for the eye insets (gaze rays) on the scene overlay."""
    eyes = []
    for role in ("eye", "eye2"):
        video = inp.get(f"{role}_video")
        if not video or not Path(video).exists():
            continue
        jeo = (cal.eyes.get(role) or {}).get("jeo") if cal else None
        if role == "eye":
            track, model = pupil, eye_model
        else:
            track = _second_eye(inp, t0_ns)
            model = None
            summary = Path(inp["session"]) / "eye2" / "metrics_summary.json"
            if summary.exists():
                em = json.loads(summary.read_text()).get("eye_model")
                if em:
                    model = EyeModel(em["cx"], em["cy"], em["radius"], radius_source=em.get("radius_source", "fitted"))
            if track is None:
                continue
        eyes.append({"role": role, "video": video, "pupil": track, "eye_model": jeo.eye_model if jeo else model})
    return eyes


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

def _target_errors(pts: pd.DataFrame, gx, gy, size, dfov) -> pd.DataFrame:
    out = pts[["label", "scene_x", "scene_y"]].copy()
    out["gaze_x"], out["gaze_y"] = gx, gy
    out["error_px"] = np.hypot(out.gaze_x - out.scene_x, out.gaze_y - out.scene_y)
    out["error_deg"] = [camera_angular_error_degrees((a, b), (c, d), size, dfov) if np.isfinite([c, d]).all()
                        else np.nan for a, b, c, d in out[["scene_x", "scene_y", "gaze_x", "gaze_y"]].to_numpy()]
    return out


def calibrate_from_session(session: str | Path, cfg: dict, check_with: str | Path | None = None) -> dict:
    """Fit gaze -> scene mappings for every eye camera in a session recorded with `eyetrack calibrate`.

    Works for on-screen targets (one marker that moves) and the printed board (one marker per target; the
    session's meta.json says which). Both mappings (jeo, poly) are fitted per eye and their errors reported,
    for the calibration targets and the held-out accuracy targets, per eye and with both eyes combined.
    check_with: an existing calibration.json to score on this session instead of fitting a new one (e.g. the
    half-size board at half the distance, to measure parallax). Every target then counts as held out.
    """
    inp = resolve_inputs(session)
    out = inp["dir"]
    meta_path = out / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    marker_for_label = (meta.get("calibration") or {}).get("marker_for_label")
    ccfg = cfg["calibration"]
    settle = ccfg.get("settle_s", 0.6)

    cap = cv2.VideoCapture(str(inp["scene_video"]))
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    dfov = cfg["scene_camera"].get("diagonal_fov_deg")
    track = detect_all_markers(inp["scene_video"], ccfg["aruco_dict"])
    existing = Calibration.load(check_with) if check_with else None
    cal = Calibration(method=ccfg.get("method", "jeo"), scene_size=size, diagonal_fov_deg=dfov)

    roles = [r for r in ("eye", "eye2") if inp.get(f"{r}_video") and Path(inp[f"{r}_video"]).exists()]
    points, errors, result = {}, [], {"scene_size": list(size), "diagonal_fov_deg": dfov, "eyes": {}}
    n_terms = poly_features(np.zeros(1), np.zeros(1), ccfg["poly_degree"]).shape[1]
    for role in roles:
        rinp = resolve_inputs(session, role)
        pupil, ellipses, t0_ns = run_pupil_stage(rinp, cfg)
        markers = load_markers(rinp, t0_ns)
        scene_t = load_scene_times(rinp, t0_ns)
        tr = track.copy()
        tr["t_s"] = scene_t[np.clip(tr.frame_index.to_numpy(), 0, len(scene_t) - 1)] if len(tr) else []
        cal_pts = target_points(markers, pupil, tr, "cal", settle, marker_for_label=marker_for_label)
        val_pts = target_points(markers, pupil, tr, "val", settle, marker_for_label=marker_for_label)
        cal_pts.to_csv(rinp["dir"] / "calibration_points.csv", index=False)
        points[role] = (cal_pts, val_pts)
        info = {"calibration_targets": len(cal_pts), "accuracy_targets": len(val_pts)}

        if existing is not None:
            maps = existing.eyes.get(role) or {}
        else:
            maps = {"poly": None, "jeo": None}
            pxy, sxy = cal_pts[["pupil_x", "pupil_y"]].to_numpy(), cal_pts[["scene_x", "scene_y"]].to_numpy()
            if len(cal_pts) >= n_terms:
                maps["poly"] = PolyMapper.fit(pxy, sxy, ccfg["poly_degree"])
            else:
                info["poly_error"] = f"only {len(cal_pts)} targets, need {n_terms}"
            try:
                good = pupil.valid.to_numpy()
                model, warning = fit_eye_model(ellipses, pupil, good, cfg.get("eye_model") or {})
                if warning:
                    info["eye_model_warning"] = warning
                maps["jeo"] = JeoMapper.fit(model, pxy, sxy)
                info["jeo_fit_rms_px"] = maps["jeo"].rms_px
            except ValueError as e:
                info["jeo_error"] = str(e)
            cal.eyes[role] = maps
        for method, m in maps.items():
            if m is None:
                continue
            for kind, pts in (("calibration", cal_pts), ("accuracy", val_pts)):
                if existing is not None:
                    kind, pts = "accuracy", pd.concat([cal_pts, val_pts], ignore_index=True)
                if not len(pts):
                    continue
                e = _target_errors(pts, *m.predict(pts.pupil_x.to_numpy(), pts.pupil_y.to_numpy()), size, dfov)
                e.insert(0, "role", role)
                e.insert(1, "method", method)
                e.insert(2, "kind", kind)
                errors.append(e)
                if existing is not None:
                    break
        result["eyes"][role] = info

    if not cal.eyes and existing is None:
        raise RuntimeError("No eye video in this session.")
    err = pd.concat(errors, ignore_index=True) if errors else pd.DataFrame()
    # both eyes combined: average the two eyes' predicted points per target (EyeTrackerStereo.cs)
    if len(roles) == 2 and len(err):
        for (method, kind), g in err.groupby(["method", "kind"]):
            a, b = g[g.role == "eye"].set_index("label"), g[g.role == "eye2"].set_index("label")
            both = a.index.intersection(b.index)
            if not len(both):
                continue
            pts = a.loc[both, ["scene_x", "scene_y"]].reset_index()
            c = combine_eyes(a.loc[both, ["gaze_x", "gaze_y"]].to_numpy(), b.loc[both, ["gaze_x", "gaze_y"]].to_numpy())
            e = _target_errors(pts, c[:, 0], c[:, 1], size, dfov)
            e.insert(0, "role", "both")
            e.insert(1, "method", method)
            e.insert(2, "kind", kind)
            err = pd.concat([err, e], ignore_index=True)
    if len(err):
        err.to_csv(out / ("check_errors.csv" if existing is not None else "calibration_errors.csv"), index=False)
        table = (err.groupby(["role", "method", "kind"])[["error_deg", "error_px"]].mean().round(2)
                 .reset_index().to_dict(orient="records"))
        result["mean_errors"] = table
    if existing is None:
        cal.save(out / "calibration.json")
        result["calibration_file"] = str(out / "calibration.json")
        result["method_used_by_process"] = cal.method
    else:
        result["checked_calibration"] = str(check_with)
    (out / ("check_summary.json" if existing is not None else "calibration_summary.json")).write_text(
        json.dumps(result, indent=2, default=_json_default))
    _print_calibration(result)
    return result


def _print_calibration(r: dict):
    for role, info in r["eyes"].items():
        print(f"  {role}: {info['calibration_targets']} calibration targets, {info['accuracy_targets']} accuracy targets"
              + "".join(f"; {k}: {v}" for k, v in info.items() if k.endswith(("error", "warning"))))
    if r.get("mean_errors"):
        print(f"  mean error (deg uses a {r['diagonal_fov_deg'] or 60:g} deg diagonal field of view):")
        for row in r["mean_errors"]:
            print(f"    {row['role']:5s} {row['method']:5s} {row['kind']:12s} {row['error_deg']:6.2f} deg "
                  f"{row['error_px']:7.1f} px")
    if r.get("calibration_file"):
        print(f"  saved {r['calibration_file']} (process uses: {r['method_used_by_process']})")
