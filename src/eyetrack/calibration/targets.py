"""Calibration / validation target sequences shown full-screen on the laptop.

Each target is an ArUco marker the scene camera can find, with a red dot in
its center for the wearer to look at. Sit ~50-60 cm from the screen and keep
the head still while the dot moves.
"""

from __future__ import annotations

import cv2
import numpy as np

# normalised (x, y) screen positions
CALIBRATION_POINTS = [(0.5, 0.5), (0.1, 0.1), (0.5, 0.1), (0.9, 0.1), (0.9, 0.5),
                      (0.9, 0.9), (0.5, 0.9), (0.1, 0.9), (0.1, 0.5)]
VALIDATION_POINTS = [(0.3, 0.3), (0.7, 0.3), (0.7, 0.7), (0.3, 0.7), (0.5, 0.5)]


def get_dictionary(name: str = "DICT_4X4_50"):
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def marker_image(size: int = 220, marker_id: int = 0, dict_name: str = "DICT_4X4_50") -> np.ndarray:
    m = cv2.aruco.generateImageMarker(get_dictionary(dict_name), marker_id, size)
    m = cv2.copyMakeBorder(m, size // 8, size // 8, size // 8, size // 8, cv2.BORDER_CONSTANT, value=255)
    m = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
    c = m.shape[0] // 2
    cv2.circle(m, (c, c), max(4, size // 25), (0, 0, 255), -1)
    return m


class TargetSequence:
    """Callable display for recorder.record(): returns (image, label) per tick, or None when done."""

    def __init__(self, points, labels=None, screen=(1440, 900), dwell_s=2.0, dict_name="DICT_4X4_50",
                 lead_in_s=2.0):
        self.points = points
        self.labels = labels or [f"cal_{i}" for i in range(len(points))]
        self.w, self.h = screen
        self.dwell = dwell_s
        self.lead_in = lead_in_s
        self.marker = marker_image(dict_name=dict_name)

    def frame(self, elapsed: float):
        canvas = np.full((self.h, self.w, 3), 128, np.uint8)
        if elapsed < self.lead_in:
            cv2.putText(canvas, "Keep your head still and look at the red dot", (40, self.h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            return canvas, None
        i = int((elapsed - self.lead_in) // self.dwell)
        if i >= len(self.points):
            return None
        nx, ny = self.points[i]
        mh, mw = self.marker.shape[:2]
        x = int(np.clip(nx * self.w - mw / 2, 0, self.w - mw))
        y = int(np.clip(ny * self.h - mh / 2, 0, self.h - mh))
        canvas[y:y + mh, x:x + mw] = self.marker
        return canvas, self.labels[i]
