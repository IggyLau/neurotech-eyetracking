"""Replay a recorded eye video as if it were the camera, to test and benchmark the real-time path.

ReplayCamera has the same interface as capture.camera.Camera (start/stop/latest/on_frame/frame_count/
actual), so recorder.record() and the live view run unchanged on it. Frames are decoded, then held
until their original time (from eye_ts.csv, scaled by 1/speed) and stamped with perf_counter_ns at
release, like the camera grab thread stamps a frame when read() returns (latency is measured from
that stamp). Each frame also carries t_media_ns, its original capture time, which the live metrics use
so blink durations and velocities are right at any replay speed; live_pupil.csv's t_ns is that time.
speed=0 releases frames as fast as the consumer takes them.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ..capture.camera import Frame


def resolve_video(path: str | Path) -> tuple[Path, Path | None]:
    """Session folder or video file -> (video, timestamps csv or None)."""
    p = Path(path)
    if p.is_dir():
        p = p / "eye.avi"
    ts = p.with_name(p.stem + "_ts.csv")
    return p, ts if ts.exists() else None


class ReplayCamera:
    def __init__(self, video: str | Path, ts_path: str | Path | None = None, speed: float = 1.0,
                 max_frames: int | None = None, start_frame: int = 0, name: str = "eye"):
        self.name = name
        self.video = str(video)
        self.cap = cv2.VideoCapture(self.video)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video {video}")
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 120.0
        n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if ts_path is not None:
            t = pd.read_csv(ts_path).t_ns.to_numpy(np.int64)
            self.rel_s = (t - t[0]) / 1e9
            self.t0_ns = int(t[0])
        else:
            self.rel_s = np.arange(max(n, 1)) / fps
            self.t0_ns = 0
        self.fps = fps
        self.speed = speed
        self.max_frames = max_frames
        self.start_frame = start_frame
        if start_frame:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        self.on_frame = None
        self._latest: Frame | None = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._count = 0
        self.finished = False
        self.decode_s = 0.0

    @property
    def actual(self) -> dict:
        return {"width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps_reported": float(self.fps), "replay_of": self.video, "replay_speed": self.speed}

    @property
    def frame_count(self) -> int:
        return self._count

    def start(self) -> "ReplayCamera":
        self._running = True
        self._thread = threading.Thread(target=self._loop, name=f"{self.name}-replay", daemon=True)
        self._thread.start()
        return self

    def _loop(self):
        t_wall0 = time.perf_counter()
        i = self.start_frame
        base = self.rel_s[min(i, len(self.rel_s) - 1)]
        while self._running:
            if self.max_frames and self._count >= self.max_frames:
                break
            t0 = time.perf_counter()
            ok, img = self.cap.read()
            self.decode_s += time.perf_counter() - t0
            if not ok:
                break
            rel = (self.rel_s[i] - base) if i < len(self.rel_s) else (i - self.start_frame) / self.fps
            if self.speed:
                due = t_wall0 + rel / self.speed
                while True:
                    wait = due - time.perf_counter()
                    if wait <= 0:
                        break
                    time.sleep(wait - 0.0005 if wait > 0.001 else 0)   # coarse sleep, then a short spin
            frame = Frame(img, time.perf_counter_ns(), self._count)
            frame.t_media_ns = self.t0_ns + int(round((base + rel) * 1e9))
            self._count += 1
            i += 1
            if self.on_frame is not None:
                self.on_frame(frame)
            with self._lock:
                self._latest = frame
        self.finished = True

    def latest(self) -> Frame | None:
        with self._lock:
            return self._latest

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5)
        self.cap.release()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()


def draw_live(img: np.ndarray, row: dict | None, mirror: bool = False) -> np.ndarray:
    """Preview image with the live result drawn on it (drawn before mirroring, text after)."""
    out = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    found = row is not None and row["found"]
    if found:
        colour = (20, 255, 255) if row["valid"] else (0, 140, 255)
        cv2.ellipse(out, row["ellipse"], colour, 2)     # raw cv2 RotatedRect (not written to the CSV)
        cv2.circle(out, (int(row["cx"]), int(row["cy"])), 3, (255, 255, 0), -1)
    if mirror:
        out = cv2.flip(out, 1)
    if found:
        cv2.putText(out, f"conf {row['confidence']:.2f}  d {row['diameter_px']:.1f}px", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(out, "no pupil", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    if row is not None:
        tags = [t for t, on in ((row["blink_state"] or "", bool(row["blink_state"])),
                                ("SACCADE", bool(row["saccade_flag"]))) if on]
        if tags:
            cv2.putText(out, "  ".join(tags), (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 255), 2)
    return out


def live_stats_text(proc, cam) -> str:
    _, row = proc.latest()
    lat = f"{row['latency_ms']:.1f}ms" if row else "--"
    return f"cam {cam.frame_count}  proc lag {lat}  dropped {proc.dropped}"


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    d = np.diff(np.r_[0, mask.astype(int), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def compare_with_offline(live_csv: str | Path, offline_csv: str | Path, start_frame: int = 0) -> dict:
    """Agreement of live (causal) results with an offline pupil.csv of the same video, frame by frame."""
    live = pd.read_csv(live_csv)
    off = pd.read_csv(offline_csv)
    live = live.assign(src_index=live.frame_index + start_frame)
    m = live.merge(off, left_on="src_index", right_on="frame_index", suffixes=("_live", "_off"))
    if not len(m):
        return {"frames_compared": 0}
    both = m.found_live.astype(bool) & m.found_off.astype(bool)
    dxy = np.hypot(m.cx_live - m.cx_off, m.cy_live - m.cy_off)[both]
    out = {"frames_compared": int(len(m)),
           "centre_diff_px_median": round(float(dxy.median()), 4) if both.any() else None,
           "centre_diff_px_max": round(float(dxy.max()), 4) if both.any() else None,
           "valid_agreement": round(float((m.valid_live.astype(bool) == m.valid_off.astype(bool)).mean()), 4)}
    if "dropout_kind" in m:
        off_lid = m.dropout_kind.fillna("").isin(["blink", "closed"]).to_numpy()
        live_lid = m.blink_state.fillna("").isin(["blink", "closed"]).to_numpy()
        out["eyelid_frames_offline"] = int(off_lid.sum())
        out["eyelid_frames_live"] = int(live_lid.sum())
        out["eyelid_frame_agreement_recall"] = round(float((live_lid & off_lid).sum() / max(off_lid.sum(), 1)), 4)
        out["eyelid_frame_agreement_precision"] = round(float((live_lid & off_lid).sum() / max(live_lid.sum(), 1)), 4)
    if "event" in m:
        # event level: an offline saccade counts as found if a live flag run overlaps it (+2 frames: the live
        # flag at frame n describes frame n-1, and the amplitude check can confirm a frame or two later)
        ro = _runs((m.event == "sac").to_numpy())
        rl = _runs(m.saccade_flag.astype(bool).to_numpy())
        delays = []
        for s_, e_ in ro:
            hit = [a for a, b in rl if a <= e_ + 2 and b >= s_]
            if hit:
                delays.append(hit[0] - s_)
        live_only = sum(not any(a <= e_ + 2 and b >= s_ for s_, e_ in ro) for a, b in rl)
        out["saccades_offline"] = len(ro)
        out["saccade_runs_live"] = len(rl)
        out["saccades_found_live"] = round(len(delays) / max(len(ro), 1), 4)
        out["live_runs_without_offline_saccade"] = round(live_only / max(len(rl), 1), 4)
        out["saccade_onset_delay_frames_median"] = float(np.median(delays)) if delays else None
    blinks_off = Path(offline_csv).with_name("blinks.csv")
    blinks_live = Path(live_csv).with_name("live_blinks.csv")
    if blinks_off.exists() and blinks_live.exists():
        bo, bl = pd.read_csv(blinks_off), pd.read_csv(blinks_live)
        lo, hi = live.src_index.min(), live.src_index.max()
        bo = bo[(bo.start_index >= lo) & (bo.end_index <= hi)]
        out["blink_events_offline"] = bo.kind.value_counts().to_dict()
        out["blink_events_live"] = bl.kind.value_counts().to_dict()
        if len(bl):
            out["blink_event_known_after_offset_ms_median"] = round(float((bl.known_at_s - bl.offset_s).median() * 1000), 1)
    return out
