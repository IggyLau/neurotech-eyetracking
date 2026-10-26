"""Causal (online) per-frame metrics: each frame uses only itself and earlier frames.

These mirror the offline stages in pipeline.py so the live numbers mean the same thing, with the
differences that causality forces:

  validity        confidence >= min_confidence, and diameter within max_size_ratio of the median of
                  the past 2 s (offline uses a centred 2 s median). Latency: 0 frames. Until 2 s x 10%
                  of history exists nothing is called an outlier (offline back-fills instead).
  blink_state     per-frame state of the current dropout run from the same brightness rule as
                  blinks.classify_dropouts (eye area brighter -> eyelid; whole frame darker -> lighting).
                  Provisional while the run lasts (it is judged on the frames seen so far): 0 frames
                  latency. The final event kind is known on the first valid frame after the run, and
                  after merge_gap_ms more for eyelid events (merging, as offline): written to
                  live_blinks.csv with the time it became known.
  saccade_flag    I-VT on a central-difference speed |p[n] - p[n-2]| / (t[n] - t[n-2]), i.e. the speed
                  the offline I-VT computes for frame n-1 (np.gradient; its Savitzky-Golay smoothing is a
                  no-op at 120 Hz). Group delay: 1 frame (8.3 ms at 120 Hz), plus 0-2 frames until the
                  movement exceeds the causal amplitude-noise floor (see OnlineSaccades). No minimum
                  saccade/fixation durations; offline remains the reference.
  yaw/pitch       only with a saved eye model (EyeModel parameters from a previous `eyetrack process`);
                  otherwise NaN and speed is in eye-image px/s.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np

from ..pipeline import brightness_features
from ..pupil.detector import DetectorParams, PupilResult, detect_pupil
from ..pupil.eye_model import EyeModel

COLUMNS = ["frame_index", "t_ns", "found", "cx", "cy", "axis_minor", "axis_major", "angle", "diameter_px",
           "confidence", "frame_brightness", "eye_brightness", "size_outlier", "valid", "blink_state",
           "yaw_deg", "pitch_deg", "speed", "saccade_flag", "queue_ms", "proc_ms", "latency_ms"]
BLINK_COLUMNS = ["onset_s", "offset_s", "duration_ms", "start_index", "end_index", "kind", "known_at_s"]
EYELID_KINDS = ("blink", "closed")


def load_eye_model(path: str | Path | None) -> EyeModel | None:
    """EyeModel from a session folder, a metrics_summary.json (key "eye_model") or a plain {cx, cy, radius} JSON."""
    if not path:
        return None
    p = Path(path)
    if p.is_dir():
        p = p / "metrics_summary.json"
    d = json.loads(p.read_text())
    d = d.get("eye_model", d)
    if not d or "radius" not in d:
        raise ValueError(f"No eye model in {p}")
    return EyeModel(float(d["cx"]), float(d["cy"]), float(d["radius"]), int(d.get("n_used", 0)),
                    float(d.get("residual_px", np.nan)))


class OnlineSizeCheck:
    """Causal version of pipeline.size_outliers: ratio to the median of the last window_s of valid diameters."""

    def __init__(self, fps: float, max_ratio: float, window_s: float = 2.0):
        self.max_ratio = max_ratio
        self.win = max(3, int(window_s * (fps or 30)))
        self.min_periods = max(3, self.win // 10)
        self.hist: deque[float] = deque(maxlen=self.win)   # nan for frames that weren't confidence-valid
        self.n_finite = 0

    def update(self, diameter: float, conf_valid: bool) -> bool:
        """True if this frame is a size outlier. The reference excludes the frame itself."""
        out = False
        if self.max_ratio and conf_valid and self.n_finite >= self.min_periods:
            vals = np.array(self.hist)
            ref = np.median(vals[np.isfinite(vals)])
            ratio = diameter / ref
            out = bool(np.isfinite(ratio) and (ratio > self.max_ratio or ratio < 1 / self.max_ratio))
        if len(self.hist) == self.hist.maxlen and np.isfinite(self.hist[0]):
            self.n_finite -= 1
        v = diameter if conf_valid else np.nan
        self.hist.append(v)
        self.n_finite += int(np.isfinite(v))
        return out


class OnlineBlinks:
    """Streaming blinks.classify_dropouts. update() per frame returns the provisional state ("" = trusted pupil)
    and appends finished events to self.events (dicts with BLINK_COLUMNS)."""

    def __init__(self, fps: float, min_ms=30, max_ms=500, merge_gap_ms=80, lighting_drop=0.08,
                 eyelid_rise=3.0, context_ms=100):
        self.dt = 1 / (fps or 120)
        self.min_ms, self.max_ms, self.merge_gap_ms = min_ms, max_ms, merge_gap_ms
        self.lighting_drop, self.eyelid_rise = lighting_drop, eyelid_rise
        self.ctx: deque[tuple[bool, float, float]] = deque(maxlen=max(1, int(round(context_ms / 1000 * fps))))
        self.t0 = None
        self.prev_t = None
        self.prev_index = None
        self.run = None        # current dropout run
        self.pending = None    # last finished event, held for merge_gap_ms in case the next one merges
        self.events: list[dict] = []
        self._hist = np.zeros(2000, np.int64)           # frame intervals 0-100 ms: camera-gap test
        self._dt_meas = self.dt
        self._n_diffs = 0

    def _kind(self, run, dur_ms: float) -> str:
        if run["fb0"] is not None:
            darker = (run["fb0"] - run["fb_min"]) / max(run["fb0"], 1e-6) >= self.lighting_drop
            lid = np.median(run["eb"]) - run["eb0"] >= self.eyelid_rise
        else:
            darker = lid = False
        if lid and not darker:
            return "dropout" if dur_ms < self.min_ms else "blink" if dur_ms <= self.max_ms else "closed"
        if darker:
            return "lighting"
        return "dropout" if dur_ms < self.min_ms else "lost"

    def _emit(self, ev: dict, t_s: float):
        ev = dict(ev, duration_ms=(ev["offset_s"] - ev["onset_s"]) * 1000, known_at_s=t_s)
        self.events.append({k: ev[k] for k in BLINK_COLUMNS})

    def _finish(self, ev: dict, t_s: float):
        p = self.pending
        if p is not None and self.merge_gap_ms > 0:
            kinds = {p["kind"], ev["kind"]}
            lidlike = kinds <= {"blink", "closed", "dropout"} and bool(kinds & {"blink", "closed"})
            if lidlike and (ev["onset_s"] - p["offset_s"]) * 1000 <= self.merge_gap_ms:
                p["offset_s"], p["end_index"] = ev["offset_s"], ev["end_index"]
                p["kind"] = "blink" if (p["offset_s"] - p["onset_s"]) * 1000 <= self.max_ms else "closed"
                return
        if p is not None:
            self._emit(p, t_s)
        self.pending = ev

    def update(self, index: int, t_s: float, valid: bool, fb: float, eb: float) -> str:
        if self.t0 is None:
            self.t0 = t_s
        ts = t_s - self.t0
        if self.prev_t is not None:
            # session-so-far median frame interval (offline uses the whole session's median), from a
            # 0.05 ms histogram; the nominal rate is used until 30 intervals have been seen
            self._hist[min(int((t_s - self.prev_t) * 20000), len(self._hist) - 1)] += 1
            self._n_diffs += 1
            if self._n_diffs >= 30 and self._n_diffs % 30 == 0:
                c = np.cumsum(self._hist)
                self._dt_meas = (np.searchsorted(c, self._n_diffs / 2) + 0.5) / 20000
        if self.prev_t is not None and t_s - self.prev_t > 2.5 * self._dt_meas:   # frames that never arrived
            self.events.append({"onset_s": self.prev_t - self.t0, "offset_s": ts,
                                "duration_ms": (t_s - self.prev_t) * 1000, "start_index": index,
                                "end_index": index, "kind": "camera_gap", "known_at_s": ts})
        state = ""
        if valid:
            if self.run is not None:
                r = self.run
                r["offset_s"], r["end_index"] = ts, index
                r["kind"] = self._kind(r, (ts - r["onset_s"]) * 1000)
                self._finish({k: r[k] for k in ("onset_s", "offset_s", "start_index", "end_index", "kind")}, ts)
                self.run = None
            if self.pending is not None and (ts - self.pending["offset_s"]) * 1000 > self.merge_gap_ms:
                self._emit(self.pending, ts)
                self.pending = None
        else:
            if self.run is None:
                before = [(f, e) for ok, f, e in self.ctx if ok]
                self.run = {"onset_s": (self.prev_t - self.t0) if self.prev_t is not None else ts,
                            "start_index": index, "fb0": float(np.median([b[0] for b in before])) if before else None,
                            "eb0": float(np.median([b[1] for b in before])) if before else None,
                            "fb_min": fb, "eb": []}
            r = self.run
            r["fb_min"] = min(r["fb_min"], fb)
            r["eb"].append(eb)
            state = self._kind(r, (ts + self.dt - r["onset_s"]) * 1000)
        self.ctx.append((valid, fb, eb))
        self.prev_t, self.prev_index = t_s, index
        return state

    def close(self, t_s: float | None = None):
        """End of stream: finish an open run (as offline does at the last frame) and the pending event."""
        ts = (t_s - self.t0) if (t_s is not None and self.t0 is not None) else (
            (self.prev_t - self.t0) if self.prev_t is not None else 0.0)
        if self.run is not None:
            r = self.run
            r["offset_s"], r["end_index"] = ts + self.dt, (self.prev_index or 0) + 1
            r["kind"] = self._kind(r, (r["offset_s"] - r["onset_s"]) * 1000)
            self._finish({k: r[k] for k in ("onset_s", "offset_s", "start_index", "end_index", "kind")}, ts)
            self.run = None
        if self.pending is not None:
            self._emit(self.pending, ts)
            self.pending = None


class OnlineSaccades:
    """Causal I-VT from the central-difference speed of the previous frame (1-frame delay).

    As offline, a fast run only counts once the eye has moved more than
    max(min_amplitude, amplitude_noise_factor x RMS-S2S noise) from where the run started; the noise is
    the RMS sample-to-sample distance of the last noise_window non-saccade samples. That check can add
    a frame or two to the onset of small saccades; it removes most fast-but-tiny noise bursts.
    """

    def __init__(self, threshold: float, amplitude_noise_factor: float = 0.0, min_amplitude: float = 0.0,
                 noise_window: int = 240):
        self.threshold = threshold
        self.factor, self.min_amplitude = amplitude_noise_factor, min_amplitude
        self.hist: deque[tuple[float, float, float, bool]] = deque(maxlen=3)
        self.s2s: deque[float] = deque(maxlen=noise_window)
        self.s2s_sum = 0.0
        self.run_start = None      # (x, y) before the current fast run
        self.confirmed = False

    @property
    def noise(self) -> float:
        return float(np.sqrt(self.s2s_sum / len(self.s2s))) if len(self.s2s) >= 10 else 0.0

    def update(self, t_s: float, x: float, y: float, ok: bool) -> tuple[float, bool]:
        ok = bool(ok and np.isfinite(x) and np.isfinite(y))
        self.hist.append((t_s, x, y, ok))
        if len(self.hist) < 3 or not all(h[3] for h in self.hist):
            self.run_start, self.confirmed = None, False
            return np.nan, False
        (t0, x0, y0, _), (_, x1, y1, _), (t2, x2, y2, _) = self.hist
        speed = float(np.hypot(x2 - x0, y2 - y0) / (t2 - t0)) if t2 > t0 else np.nan
        fast = bool(speed > self.threshold)
        if not fast:
            d2 = (x2 - x1) ** 2 + (y2 - y1) ** 2
            if len(self.s2s) == self.s2s.maxlen:
                self.s2s_sum -= self.s2s[0]
            self.s2s.append(d2)
            self.s2s_sum += d2
            self.run_start, self.confirmed = None, False
            return speed, False
        if self.run_start is None:
            self.run_start = (x0, y0)       # the sample before frame n-1, whose speed this is
        if not self.confirmed:
            if self.factor and len(self.s2s) < 10:
                return speed, False             # noise floor not known yet (first ~10 frames)
            floor = max(self.min_amplitude, self.factor * self.noise)
            self.confirmed = bool(np.hypot(x2 - self.run_start[0], y2 - self.run_start[1]) >= floor)
        return speed, self.confirmed


class OnlineProcessor:
    """Per-frame detection + causal metrics. process() returns one row dict (COLUMNS minus timing fields)."""

    def __init__(self, cfg: dict, eye_model: EyeModel | None = None, fps: float | None = None):
        self.params = DetectorParams.from_config(cfg["pupil"])
        self.min_conf = cfg["pupil"]["min_confidence"]
        fps = fps or cfg["eye_camera"].get("fps") or 120
        self.size = OnlineSizeCheck(fps, cfg["pupil"].get("max_size_ratio", 0))
        b = cfg["blinks"]
        self.blinks = OnlineBlinks(fps, b["min_duration_ms"], b["max_duration_ms"], b.get("merge_gap_ms", 0),
                                   b.get("lighting_drop", 0.08), b.get("eyelid_rise", 3.0))
        self.eye_model = eye_model
        ev = cfg["events"]
        if eye_model is not None:
            thr = ev["velocity_threshold"]
            thr = ev.get("min_threshold", 30) if isinstance(thr, str) else thr   # "adaptive" needs the whole session
            self.unit = "deg"
        else:
            thr, self.unit = ev["velocity_threshold_px"], "px"
        self.saccades = OnlineSaccades(float(thr), ev.get("amplitude_noise_factor", 0.0),
                                       ev.get("min_amplitude", 0.0) if eye_model is not None else 0.0)
        self.eye_xy = None
        self.last_t = None

    def process(self, img: np.ndarray, t_ns: int, index: int) -> dict:
        r: PupilResult = detect_pupil(img, self.params)
        fb, eb, self.eye_xy = brightness_features(img, r, self.eye_xy)
        t_s = t_ns / 1e9
        conf_valid = bool(r.found and r.confidence >= self.min_conf)
        outlier = self.size.update(r.diameter_px, conf_valid)
        valid = conf_valid and not outlier
        state = self.blinks.update(index, t_s, valid, fb, eb)
        yaw = pitch = np.nan
        if self.eye_model is not None and r.found:
            y_, p_ = self.eye_model.angles_deg(np.array([r.cx]), np.array([r.cy]))
            yaw, pitch = float(y_[0]), float(p_[0])
        x, y = (yaw, pitch) if self.eye_model is not None else (r.cx, r.cy)
        speed, flag = self.saccades.update(t_s, x, y, valid)
        self.last_t = t_s
        return {"frame_index": index, "t_ns": t_ns, "found": r.found, "cx": r.cx, "cy": r.cy,
                "axis_minor": r.axis_minor, "axis_major": r.axis_major, "angle": r.angle,
                "diameter_px": r.diameter_px, "confidence": r.confidence, "frame_brightness": fb,
                "eye_brightness": eb, "size_outlier": outlier, "valid": valid, "blink_state": state,
                "yaw_deg": yaw, "pitch_deg": pitch, "speed": speed, "saccade_flag": flag,
                "ellipse": r.ellipse}

    def close(self):
        self.blinks.close()
