"""Tracker status for the on-screen indicator: is the tracker still working, and does it pass the checks?

Judged over the last `window_s` of live results (realtime/worker.py rows):
    no frames      no new result for `stale_s`: the camera stopped or was unplugged
    camera dark    median frame brightness below `dark_brightness`: the lens is covered or the LEDs are off
    video          frames no longer match the video file's size, so they aren't being saved
    lost           fewer than `lost_fraction` of frames valid: the pupil is out of frame or not found
    weak           fewer than `ok_fraction` valid, the median pupil outside the frame's central zone,
                   or the frame rate below `min_fps`
    ok             everything else

The thresholds follow the alignment checks in protocol.py, relaxed for a session in progress, where the eye
moves and blinks.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

COLOURS = {"ok": (60, 180, 60), "weak": (0, 170, 230), "lost": (40, 40, 220)}   # BGR


@dataclass
class Status:
    level: str      # ok | weak | lost
    text: str


class TrackerStatus:
    def __init__(self, shape: tuple[int, int], window_s: float = 2.0, stale_s: float = 0.5,
                 ok_fraction: float = 0.8, lost_fraction: float = 0.3, min_fps: float = 100,
                 dark_brightness: float = 10, zone=(0.1, 0.1, 0.9, 0.9)):
        self.h, self.w = shape
        self.window_s, self.stale_s = window_s, stale_s
        self.ok_fraction, self.lost_fraction, self.min_fps = ok_fraction, lost_fraction, min_fps
        self.dark_brightness, self.zone = dark_brightness, zone
        self.rows: deque = deque()
        self.last_seen = None

    def add(self, rows: list[dict], now: float | None = None):
        now = time.perf_counter() if now is None else now
        for r in rows:
            self.rows.append((now, r))
            self.last_seen = now
        while self.rows and now - self.rows[0][0] > self.window_s:
            self.rows.popleft()

    def status(self, now: float | None = None, video_ok: bool = True) -> Status:
        now = time.perf_counter() if now is None else now
        if self.last_seen is None or now - self.last_seen > self.stale_s:
            return Status("lost", "NO FRAMES - check the camera cable")
        rows = [r for _, r in self.rows]
        if not video_ok:
            return Status("lost", "VIDEO NOT SAVING - frame size changed")
        if np.median([r["frame_brightness"] for r in rows]) < self.dark_brightness:
            return Status("lost", "CAMERA DARK - lens covered or LEDs off")
        valid = np.array([bool(r["valid"]) for r in rows])
        frac = float(valid.mean())
        if frac < self.lost_fraction:
            return Status("lost", f"PUPIL LOST - {frac:.0%} tracked")
        t = np.array([r["t_ns"] for r in rows]) / 1e9
        fps = (len(t) - 1) / (t[-1] - t[0]) if len(t) > 1 and t[-1] > t[0] else 0.0
        cx = np.median([r["cx"] for r in rows if r["valid"]]) / self.w
        cy = np.median([r["cy"] for r in rows if r["valid"]]) / self.h
        x0, y0, x1, y1 = self.zone
        if not (x0 <= cx <= x1 and y0 <= cy <= y1):
            return Status("weak", "PUPIL NEAR EDGE - re-aim camera")
        if frac < self.ok_fraction:
            return Status("weak", f"WEAK - {frac:.0%} tracked")
        if fps < self.min_fps:
            return Status("weak", f"LOW FPS - {fps:.0f}")
        return Status("ok", f"TRACKING {frac:.0%}")


def draw_status(img, st: Status, scale: float = 0.5):
    """A small dot and label in the bottom-right corner. Kept small so it barely changes screen brightness."""
    h, w = img.shape[:2]
    (tw, th), _ = cv2.getTextSize(st.text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = w - tw - 40, h - 16
    colour = COLOURS[st.level] if img.ndim == 3 else int(np.mean(COLOURS[st.level]))
    cv2.circle(img, (x - 14, y - th // 2), 7, colour, -1, cv2.LINE_AA)
    cv2.putText(img, st.text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)
    return img
