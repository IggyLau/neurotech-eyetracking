"""Figures and videos for a processed session."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ..metrics.aoi import heatmap, overlay_heatmap  # noqa: E402


def timeseries_report(pupil: pd.DataFrame, saccades: pd.DataFrame, blinks: pd.DataFrame,
                      markers: pd.DataFrame, path: Path, unit: str):
    t = pupil.t_s
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    ax = axes[0]
    ax.plot(t, pupil.diameter_px, color="0.75", lw=0.6, label="raw")
    ax.plot(t, pupil.diameter_clean_px, color="C0", lw=1.2, label="cleaned")
    ax.set_ylabel("pupil diameter (px)")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1]
    if "yaw_deg" in pupil:
        ax.plot(t, pupil.yaw_deg, lw=0.8, label="horizontal (yaw)")
        ax.plot(t, pupil.pitch_deg, lw=0.8, label="vertical (pitch)")
    else:
        ax.plot(t, pupil.cx, lw=0.8, label="pupil x")
        ax.plot(t, pupil.cy, lw=0.8, label="pupil y")
    for s in saccades.itertuples():
        ax.axvspan(s.onset_s, s.offset_s, color="C3", alpha=0.25, lw=0)
    ax.set_ylabel(f"eye position ({unit})")
    ax.legend(loc="upper right", fontsize=8, title="red = saccades", title_fontsize=8)

    ax = axes[2]
    ax.plot(t, pupil.confidence, lw=0.6, color="C2")
    for b in blinks[blinks.kind == "blink"].itertuples() if len(blinks) else []:
        ax.axvspan(b.onset_s, b.offset_s, color="k", alpha=0.2, lw=0)
    ax.set_ylabel("detection confidence")
    ax.set_xlabel("time (s)   grey = blinks")
    ax.set_ylim(0, 1.05)

    for m in markers.itertuples() if len(markers) else []:
        for a in axes:
            a.axvline(m.t_s, color="purple", lw=0.8, ls="--")
        axes[0].annotate(m.label, (m.t_s, 1), xycoords=("data", "axes fraction"), fontsize=7,
                         rotation=90, va="top", ha="right", color="purple")

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _middle_frame(video: Path) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(n // 2, 0))
    ok, img = cap.read()
    cap.release()
    return img if ok else None


def scene_heatmap(scene_video: Path, fixations: pd.DataFrame, path: Path):
    bg = _middle_frame(scene_video)
    if bg is None or "scene_x" not in fixations or fixations.empty:
        return
    h, w = bg.shape[:2]
    dens = heatmap(fixations.scene_x.to_numpy(), fixations.scene_y.to_numpy(), (w, h),
                   fixations.duration_ms.to_numpy())
    cv2.imwrite(str(path), overlay_heatmap(bg, dens))


def gaze_overlay_video(scene_video: Path, gaze: pd.DataFrame, path: Path):
    """Scene video with the gaze point drawn on each frame (uses gaze samples mapped to scene frames)."""
    if "scene_frame" not in gaze:
        return
    per_frame = gaze.dropna().groupby("scene_frame")[["gaze_x", "gaze_y"]].mean()
    cap = cv2.VideoCapture(str(scene_video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    writer = None
    i = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if writer is None:
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (img.shape[1], img.shape[0]))
        if i in per_frame.index:
            gx, gy = per_frame.loc[i]
            cv2.circle(img, (int(gx), int(gy)), 14, (0, 0, 255), 3)
        writer.write(img)
        i += 1
    cap.release()
    if writer is not None:
        writer.release()
