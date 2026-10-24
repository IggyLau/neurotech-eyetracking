"""Scrolling pupil-diameter strip for the live and record previews.

Shows the last few seconds of diameter so dilation and constriction are visible
while recording. After a marker key it also shows the change from the 0.5 s
before that marker, the same baseline `eyetrack process` uses for its epochs.
Raw per-frame diameter only: cleaning and smoothing happen offline.
"""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from ..pupil.detector import PupilResult

BG = (30, 30, 30)
LINE = (255, 200, 0)
MARK = (255, 80, 200)
TEXT = (230, 230, 230)


class PupilTrace:
    def __init__(self, window_s: float = 10.0, min_confidence: float = 0.5, height: int = 120,
                 baseline_s: float = 0.5):
        self.window_s = window_s
        self.min_confidence = min_confidence
        self.height = height
        self.baseline_s = baseline_s
        self.samples: deque[tuple[float, float]] = deque()   # (t_s, diameter or nan)
        self.marks: deque[tuple[float, str, float]] = deque()  # (t_s, label, baseline diameter)

    def add(self, t_s: float, r: PupilResult):
        ok = r.found and r.confidence >= self.min_confidence
        self.samples.append((t_s, r.diameter_px if ok else np.nan))
        while self.samples and self.samples[0][0] < t_s - self.window_s:
            self.samples.popleft()
        while self.marks and self.marks[0][0] < t_s - self.window_s:
            self.marks.popleft()

    def mark(self, t_s: float, label: str):
        self.marks.append((t_s, label, self._median_between(t_s - self.baseline_s, t_s)))

    def current(self, span_s: float = 0.25) -> float:
        """Median diameter over the last span_s, which steadies the per-frame jitter."""
        if not self.samples:
            return np.nan
        t_now = self.samples[-1][0]
        return self._median_between(t_now - span_s, t_now + 1e-9)

    def _median_between(self, t0: float, t1: float) -> float:
        d = [d for t, d in self.samples if t0 <= t < t1 and np.isfinite(d)]
        return float(np.median(d)) if d else np.nan

    def draw(self, width: int) -> np.ndarray:
        h = self.height
        img = np.full((h, width, 3), BG, np.uint8)
        if not self.samples:
            return img
        t = np.array([s[0] for s in self.samples])
        d = np.array([s[1] for s in self.samples])
        t_end = t[-1]
        valid = np.isfinite(d)

        top, bottom = 26, h - 8
        if valid.any():
            lo, hi = np.percentile(d[valid], [2, 98])
            mid, span = (lo + hi) / 2, max(hi - lo, 4.0) * 1.2  # at least +-2.4 px so noise isn't magnified
            lo, hi = mid - span / 2, mid + span / 2
            xs = ((t - (t_end - self.window_s)) / self.window_s * (width - 1)).astype(int)
            ys = (bottom - (d - lo) / (hi - lo) * (bottom - top)).round()
            # draw each run of valid samples as its own line, so blinks show as gaps
            run = []
            for x, y, ok in zip(xs, ys, valid):
                if ok:
                    run.append((x, int(y)))
                elif run:
                    _polyline(img, run)
                    run = []
            _polyline(img, run)
            cv2.putText(img, f"{hi:.0f}", (width - 34, top + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT, 1)
            cv2.putText(img, f"{lo:.0f}", (width - 34, bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT, 1)

        for tm, label, _ in self.marks:
            x = int((tm - (t_end - self.window_s)) / self.window_s * (width - 1))
            cv2.line(img, (x, top), (x, bottom), MARK, 1)
            cv2.putText(img, label, (x + 3, top + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, MARK, 1)

        cur = self.current()
        text = f"pupil {cur:.1f}px" if np.isfinite(cur) else "pupil --"
        text += f"  valid {valid.mean():.0%}"
        if self.marks:
            _, label, base = self.marks[-1]
            if np.isfinite(base) and np.isfinite(cur):
                text += f"  since {label}: {100 * (cur - base) / base:+.1f}%"
        cv2.putText(img, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT, 1)
        return img


def _polyline(img: np.ndarray, pts: list[tuple[int, int]]):
    if len(pts) > 1:
        cv2.polylines(img, [np.array(pts, np.int32)], False, LINE, 1, cv2.LINE_AA)
    elif pts:
        cv2.circle(img, pts[0], 1, LINE, -1)


def with_trace(frame_img: np.ndarray, trace: PupilTrace) -> np.ndarray:
    """Stack the trace strip under a preview image of the same width."""
    return np.vstack([frame_img, trace.draw(frame_img.shape[1])])
