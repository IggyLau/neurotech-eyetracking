"""Area-of-interest metrics and gaze heatmaps on the scene image.

AOIs are rectangles in scene-camera pixels, e.g. {"left_page": [0, 0, 320, 480]}.
Because the scene camera moves with the head, fixed-pixel AOIs only make sense
when the head is still (chin rest, or a screen task). For moving heads, mark
objects per frame (or use gazeMapper with ArUco-tagged planes) first.
"""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd


def assign_aoi(x: np.ndarray, y: np.ndarray, aois: dict[str, list[float]]) -> np.ndarray:
    labels = np.full(len(x), None, dtype=object)
    for name, (ax, ay, aw, ah) in aois.items():
        inside = (x >= ax) & (x < ax + aw) & (y >= ay) & (y < ay + ah)
        labels[inside & (labels == None)] = name  # noqa: E711  first AOI listed wins on overlap
    return labels


def aoi_metrics(fixations: pd.DataFrame, aois: dict[str, list[float]]) -> pd.DataFrame:
    """Per AOI: dwell time, fixation count, time to first fixation, entries and revisits."""
    if fixations.empty:
        return pd.DataFrame(columns=["aoi", "dwell_ms", "fixations", "first_fixation_s", "entries", "revisits"])
    fx = fixations.sort_values("onset_s").reset_index(drop=True)
    fx["aoi"] = assign_aoi(fx.x.to_numpy(), fx.y.to_numpy(), aois)
    # an "entry" is a fixation in an AOI whose previous fixation was elsewhere
    entered = fx.aoi.ne(fx.aoi.shift())
    t0 = fx.onset_s.iloc[0]
    rows = []
    for name in aois:
        sel = fx.aoi == name
        entries = int((sel & entered).sum())
        rows.append({
            "aoi": name,
            "dwell_ms": float(fx.loc[sel, "duration_ms"].sum()),
            "fixations": int(sel.sum()),
            "first_fixation_s": float(fx.loc[sel, "onset_s"].min() - t0) if sel.any() else np.nan,
            "entries": entries,
            "revisits": max(entries - 1, 0),
        })
    return pd.DataFrame(rows)


def heatmap(x: np.ndarray, y: np.ndarray, size: tuple[int, int], weights: np.ndarray | None = None,
            sigma_px: float = 25) -> np.ndarray:
    """Duration-weighted gaze density, normalised to 0..1. size = (width, height)."""
    w, h = size
    acc = np.zeros((h, w), np.float32)
    weights = np.ones(len(x)) if weights is None else np.asarray(weights, float)
    for xi, yi, wi in zip(x, y, weights):
        if np.isfinite(xi) and np.isfinite(yi) and 0 <= xi < w and 0 <= yi < h:
            acc[int(yi), int(xi)] += wi
    acc = cv2.GaussianBlur(acc, (0, 0), sigma_px)
    return acc / acc.max() if acc.max() > 0 else acc


def overlay_heatmap(background: np.ndarray, density: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    bg = background if background.ndim == 3 else cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
    colored = cv2.applyColorMap((density * 255).astype(np.uint8), cv2.COLORMAP_JET)
    mask = (density > 0.05)[..., None]
    blended = cv2.addWeighted(bg, 1 - alpha, colored, alpha, 0)
    return np.where(mask, blended, bg)
