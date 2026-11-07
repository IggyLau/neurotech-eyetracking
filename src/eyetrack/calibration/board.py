"""Printed calibration board: ArUco markers on a wall at a known distance, seen by the scene camera.

Layout: marker IDs 0-8 form a 3x3 calibration grid (row by row from the top left, so 4 is the centre and 1
the one above it); IDs 9-12 sit halfway between the centre and each corner and are used only to measure
accuracy. Outer targets are `spread_deg` from the centre as seen from the eye at `distance_cm`.

During `eyetrack calibrate --board` the laptop shows which marker to look at and says its number out loud
(macOS `say`), so the wearer can keep looking at the wall. The scene camera finds the markers in every frame,
so where each target is in the image never depends on the head staying still.

The calibration order follows JEOresearch's FrontCameraTracker (centre, top, bottom, left, right) and then
adds the four corners, which JEO doesn't use.
"""

from __future__ import annotations

import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .targets import marker_image

CAL_IDS = list(range(9))
VAL_IDS = [9, 10, 11, 12]
CAL_ORDER = [4, 1, 7, 3, 5, 0, 2, 6, 8]     # centre, top, bottom, left, right (JEO), then corners
GRID = {0: (-1, -1), 1: (0, -1), 2: (1, -1), 3: (-1, 0), 4: (0, 0), 5: (1, 0), 6: (-1, 1), 7: (0, 1), 8: (1, 1),
        9: (-0.5, -0.5), 10: (0.5, -0.5), 11: (-0.5, 0.5), 12: (0.5, 0.5)}   # (x, y) in units of spread, y down


def layout_cm(distance_cm: float, spread_deg: float) -> dict[int, tuple[float, float]]:
    """Marker centre positions on the wall in cm from the centre marker (x right, y down)."""
    return {mid: (distance_cm * math.tan(math.radians(gx * spread_deg)),
                  distance_cm * math.tan(math.radians(gy * spread_deg))) for mid, (gx, gy) in GRID.items()}


def target_labels() -> tuple[list[str], dict[str, int]]:
    """Recording order of target labels, and which marker each label is."""
    labels = [f"cal_{m}" for m in CAL_ORDER] + [f"val_{m}" for m in VAL_IDS]
    return labels, {lab: int(lab.split("_")[1]) for lab in labels}


def write_board_pdf(path: str | Path, distance_cm: float = 100, spread_deg: float = 20, marker_cm: float = 8,
                    dict_name: str = "DICT_4X4_50") -> Path:
    """A PDF with a layout page, then one marker per page printed at marker_cm wide (print at 100% scale)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    path = Path(path)
    pos = layout_cm(distance_cm, spread_deg)
    page_w, page_h = 8.5, 11.0          # US letter, inches
    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(page_w, page_h))
        fig.text(0.08, 0.95, "Eye tracker calibration board", fontsize=16, weight="bold")
        lines = [f"Wearer's eyes {distance_cm:g} cm from the wall, level with marker 4.",
                 f"Outer markers are {spread_deg:g} deg from the centre. Print the marker pages at 100% scale;",
                 f"each black square should measure {marker_cm:g} cm. Tape each marker with its centre at the",
                 "position below (cm from the centre of marker 4; + is right and down).",
                 "Markers 0-8 calibrate, 9-12 measure accuracy. For the parallax check, print the same layout",
                 "with half the distance (eyetrack board --distance-cm 50) and record a check session."]
        for i, line in enumerate(lines):
            fig.text(0.08, 0.91 - i * 0.022, line, fontsize=9.5)
        ax = fig.add_axes([0.1, 0.33, 0.8, 0.42])
        for mid, (x, y) in pos.items():
            colour = "black" if mid in CAL_IDS else "tab:blue"
            ax.add_patch(plt.Rectangle((x - marker_cm / 2, y - marker_cm / 2), marker_cm, marker_cm,
                                       fill=False, ec=colour, lw=1.5))
            ax.text(x, y, str(mid), ha="center", va="center", fontsize=12, color=colour, weight="bold")
        lim = max(abs(v) for p in pos.values() for v in p) + marker_cm
        ax.set_xlim(-lim, lim)
        ax.set_ylim(lim, -lim)
        ax.set_aspect("equal")
        ax.set_xlabel("cm (right)")
        ax.set_ylabel("cm (down)")
        ax.grid(alpha=0.3)
        rows = [f"{mid:>2}   x {x:+6.1f} cm   y {y:+6.1f} cm   {'calibration' if mid in CAL_IDS else 'accuracy'}"
                for mid, (x, y) in sorted(pos.items())]
        fig.text(0.1, 0.03, "\n".join(rows), fontsize=8.5, family="monospace")
        pdf.savefig(fig)
        plt.close(fig)

        side_in = marker_cm / 2.54
        for mid in sorted(pos):
            img = marker_image(size=800, marker_id=mid, dict_name=dict_name)   # 800 px marker + white border
            scale = side_in * img.shape[0] / 800        # page inches for the whole image incl. border
            fig = plt.figure(figsize=(page_w, page_h))
            ax = fig.add_axes([(page_w - scale) / 2 / page_w, (page_h - scale) / 2 / page_h,
                               scale / page_w, scale / page_h])
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), interpolation="nearest")
            ax.axis("off")
            x, y = pos[mid]
            fig.text(0.5, 0.9, f"Marker {mid}", ha="center", fontsize=28, weight="bold")
            fig.text(0.5, 0.86, f"x {x:+.1f} cm, y {y:+.1f} cm from marker 4   "
                                f"({'calibration' if mid in CAL_IDS else 'accuracy'})", ha="center", fontsize=12)
            fig.text(0.5, 0.1, f"Black square = {marker_cm:g} cm. Look at the red dot.", ha="center", fontsize=11)
            pdf.savefig(fig)
            plt.close(fig)
    return path


def say(text: str):
    """Speak a short cue without blocking (macOS `say`); a terminal bell elsewhere."""
    if shutil.which("say"):
        subprocess.Popen(["say", text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        print("\a", end="", flush=True)


@dataclass
class BoardSequence:
    """Display for recorder.record(): tells the wearer which board marker to look at.

    Returns (image, label) per tick and None when done; record() writes a marker each time the label changes.
    """
    dwell_s: float = 2.5
    lead_in_s: float = 4.0
    screen: tuple[int, int] = (1440, 900)
    speak: bool = True
    distance_cm: float = 100
    spread_deg: float = 20

    def __post_init__(self):
        self.labels, self.marker_for_label = target_labels()
        self._spoken = -1
        self.meta = {"type": "board", "marker_for_label": self.marker_for_label,
                     "distance_cm": self.distance_cm, "spread_deg": self.spread_deg}

    def frame(self, elapsed: float):
        w, h = self.screen
        canvas = np.full((h, w, 3), 40, np.uint8)
        if elapsed < self.lead_in_s:
            if self.speak and self._spoken == -1:
                self._spoken = -2
                say("Look at each marker when you hear its number")
            for k, line in enumerate(["Calibration board", f"Sit {self.distance_cm:g} cm from the board, eyes level with marker 4.",
                                      "Look at each marker's red dot when you hear its number.",
                                      "Keep your head still if you can; small movements are fine."]):
                cv2.putText(canvas, line, (60, h // 3 + k * 60), cv2.FONT_HERSHEY_SIMPLEX, 1.1 if k else 1.6,
                            (255, 255, 255), 2)
            return canvas, None
        i = int((elapsed - self.lead_in_s) // self.dwell_s)
        if i >= len(self.labels):
            return None
        label = self.labels[i]
        mid = self.marker_for_label[label]
        if self.speak and i != self._spoken:
            self._spoken = i
            say(str(mid))
        # small map of the board with the current marker highlighted, plus the number in large type
        cx, cy, unit = w // 2, h // 2 + 60, min(w, h) // 6
        for m, (gx, gy) in GRID.items():
            p = (int(cx + gx * unit), int(cy + gy * unit))
            on = m == mid
            cv2.rectangle(canvas, (p[0] - 28, p[1] - 28), (p[0] + 28, p[1] + 28),
                          (0, 0, 255) if on else (120, 120, 120), -1 if on else 2)
            cv2.putText(canvas, str(m), (p[0] - 14, p[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        kind = "calibration" if label.startswith("cal") else "accuracy check"
        cv2.putText(canvas, f"Look at marker {mid}", (60, 110), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 255, 255), 4)
        cv2.putText(canvas, f"{i + 1}/{len(self.labels)}  ({kind})", (60, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (200, 200, 200), 2)
        return canvas, label
