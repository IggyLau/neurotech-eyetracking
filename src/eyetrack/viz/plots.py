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
    clean = pupil.diameter_clean_px.dropna()
    if len(clean):      # scale to the cleaned trace: a few raw misdetections can be 100x the pupil
        lo, hi = clean.min(), clean.max()
        ax.set_ylim(lo - 0.15 * (hi - lo), hi + 0.15 * (hi - lo))
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
    # a tick per saccade along the top: over a whole session a saccade is narrower than a pixel, so the
    # bands alone vanish wherever saccades are short (reading)
    if len(saccades):
        ax.vlines(saccades.onset_s, 0.93, 1.0, transform=ax.get_xaxis_transform(), color="C3", lw=0.5)
    ax.set_ylabel(f"eye position ({unit})")
    ax.legend(loc="upper right", fontsize=8, title="red = saccades (tick at onset)", title_fontsize=8)

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


def epochs_report(epochs: pd.DataFrame, path: Path):
    """Baseline-corrected pupil change after each marker, one panel per marker label."""
    if epochs.empty:
        return
    labels = list(dict.fromkeys(epochs.label))
    fig, axes = plt.subplots(len(labels), 1, figsize=(10, 2.6 * len(labels)), squeeze=False)
    for ax, label in zip(axes[:, 0], labels):
        sel = epochs[epochs.label == label]
        for k, ep in sel.groupby("event"):
            ax.plot(ep.t_rel_s, ep.change_pct, lw=1, label=f"#{k}")
        ax.axvline(0, color="purple", lw=0.8, ls="--")
        ax.axhline(0, color="0.6", lw=0.6)
        ax.set_title(label, fontsize=9, loc="left")
        ax.set_ylabel("change from baseline (%)")
        ax.legend(loc="upper right", fontsize=7)
    axes[-1, 0].set_xlabel("time from marker (s)")
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


EYE_COLOURS = {"eye": (255, 140, 0), "eye2": (60, 200, 60)}     # BGR: first eye blue, second eye green
COMBINED_COLOUR = (0, 0, 255)


def draw_gaze_ray(img: np.ndarray, eye_model, px: float, py: float, colour=(0, 255, 255)) -> np.ndarray:
    """Eye image with the eyeball outline, its centre and the gaze ray from the centre through the pupil.

    The same picture JEOresearch's 3D tracker draws on the eye camera (sphere circle plus the line from the
    eyeball centre through the pupil). Does nothing without an eye model.
    """
    if eye_model is None:
        return img
    c = (int(round(eye_model.cx)), int(round(eye_model.cy)))
    cv2.circle(img, c, int(round(eye_model.radius)), colour, 1, cv2.LINE_AA)
    cv2.circle(img, c, 4, colour, -1)
    if np.isfinite(px) and np.isfinite(py):
        d = np.array([px - eye_model.cx, py - eye_model.cy])
        end = np.array(c) + 2.0 * d
        cv2.line(img, c, (int(end[0]), int(end[1])), colour, 2, cv2.LINE_AA)
        cv2.circle(img, (int(px), int(py)), 4, (0, 0, 255), -1)
    return img


def _eye_inset(eye: dict, t_s: float, cap: cv2.VideoCapture, state: dict, width: int) -> np.ndarray | None:
    """Eye frame nearest t_s (on the first eye's clock) with its gaze ray, resized to width."""
    p = eye["pupil"]
    if not len(p):
        return None
    k = int(np.clip(np.searchsorted(p.t_s.to_numpy(), t_s), 0, len(p) - 1))
    fi = int(p.frame_index.iloc[k])
    if fi < state["next"] or fi - state["next"] > 300:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        state["next"] = fi
    while state["next"] < fi:
        cap.grab()
        state["next"] += 1
    ok, img = cap.read()
    if not ok:
        return None
    state["next"] += 1
    row = p.iloc[k]
    ok_pupil = bool(row.get("valid", True))
    draw_gaze_ray(img, eye["eye_model"], row.cx if ok_pupil else np.nan, row.cy if ok_pupil else np.nan,
                  EYE_COLOURS.get(eye["role"], (0, 255, 255)))
    img = cv2.resize(img, (width, int(width * img.shape[0] / img.shape[1])))
    cv2.putText(img, eye["role"], (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, EYE_COLOURS.get(eye["role"], (0, 255, 255)), 2)
    return img


def draw_gaze_frame(img: np.ndarray, row: pd.Series | None, insets: list[np.ndarray] = ()) -> np.ndarray:
    """Scene frame with each eye's gaze point (small dots) and the combined gaze (red ring), plus eye insets."""
    h, w = img.shape[:2]
    if row is not None:
        for role, col in EYE_COLOURS.items():
            x, y = row.get(f"gaze_{role}_x", np.nan), row.get(f"gaze_{role}_y", np.nan)
            if np.isfinite(x) and np.isfinite(y):
                cv2.circle(img, (int(x), int(y)), 7, col, -1, cv2.LINE_AA)
        x, y = row.get("gaze_x", np.nan), row.get("gaze_y", np.nan)
        if np.isfinite(x) and np.isfinite(y):
            inside = 0 <= x < w and 0 <= y < h
            cv2.circle(img, (int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))), 18, COMBINED_COLOUR, 3,
                       cv2.LINE_AA)
            if not inside:
                cv2.putText(img, "gaze outside scene", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COMBINED_COLOUR, 2)
    x0 = 10
    for ins in insets:
        if ins is None:
            continue
        ih, iw = ins.shape[:2]
        if ih + 10 > h or x0 + iw > w:
            break
        img[h - ih - 10:h - 10, x0:x0 + iw] = ins
        x0 += iw + 10
    return img


def _per_scene_frame(gaze: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in gaze.columns if c.startswith("gaze_")]
    return gaze.groupby("scene_frame")[cols + ["t_s"]].median()


def gaze_overlay_snapshot(scene_video: Path, gaze: pd.DataFrame, path: Path, eyes: list[dict] = (),
                          inset_width: int = 240) -> Path | None:
    """One scene frame with the gaze drawn on it (the frame in the middle of the frames with a gaze point)."""
    if "scene_frame" not in gaze:
        return None
    per = _per_scene_frame(gaze).dropna(subset=["gaze_x", "gaze_y"])
    if not len(per):
        return None
    fi = int(per.index[len(per) // 2])
    cap = cv2.VideoCapture(str(scene_video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ok, img = cap.read()
    cap.release()
    if not ok:
        return None
    row = per.loc[fi]
    insets = []
    for eye in eyes:
        ecap = cv2.VideoCapture(str(eye["video"]))
        insets.append(_eye_inset(eye, row.t_s, ecap, {"next": 10**9}, inset_width))
        ecap.release()
    cv2.imwrite(str(path), draw_gaze_frame(img, row, insets))
    return path


def gaze_overlay_video(scene_video: Path, gaze: pd.DataFrame, path: Path, eyes: list[dict] = (),
                       inset_width: int = 240):
    """Scene video with the gaze drawn on each frame (gaze samples mapped to scene frames), with eye insets."""
    if "scene_frame" not in gaze:
        return
    per = _per_scene_frame(gaze)
    cap = cv2.VideoCapture(str(scene_video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    ecaps = [(eye, cv2.VideoCapture(str(eye["video"])), {"next": 0}) for eye in eyes]
    writer = None
    i = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if writer is None:
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (img.shape[1], img.shape[0]))
        row = per.loc[i] if i in per.index else None
        insets = [_eye_inset(e, row.t_s, c, st, inset_width) for e, c, st in ecaps] if row is not None else []
        writer.write(draw_gaze_frame(img, row, insets))
        i += 1
    cap.release()
    for _, c, _ in ecaps:
        c.release()
    if writer is not None:
        writer.release()
