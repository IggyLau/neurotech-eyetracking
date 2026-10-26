"""Edge-based pupil detector, following the method published for Pupil Labs' 2D detector.

Written from the published description (Świrski et al. 2012; Kassner et al. 2014), not from the
Pupil Labs source, so it stays under this repository's licence:
  1. coarse search: the box with the largest "dark centre, bright surround" contrast (Haar-like
     feature on an integral image) gives the pupil's rough position and size
  2. edges (Canny) near that box, kept only where the pupil-dark side is next to them, and dropped
     next to bright pixels (glints, the screen reflected on the cornea)
  3. RANSAC ellipse fit to the remaining edge pixels: the ellipse that most edge pixels agree on,
     so an arc hidden by a reflection or the eyelid is simply ignored instead of bending the fit
  4. confidence = share of the ellipse's visible perimeter backed by an edge pixel

The threshold detector in detector.py fits the outline of a dark blob, so anything that takes a bite
out of the blob moves the fit. This one fits the pupil's edge directly.
"""

from __future__ import annotations

import cv2
import numpy as np

from .detector import DetectorParams, PupilResult, to_gray


def coarse_pupil(gray: np.ndarray, radii: tuple[int, ...], stride: int = 4,
                 ignore_border: int = 10) -> tuple[int, int, int]:
    """(x, y, r) of the box with the strongest dark-centre / bright-surround contrast."""
    h, w = gray.shape
    ii = cv2.integral(gray).astype(np.float64)
    best = (w // 2, h // 2, radii[0], -np.inf)
    for r in radii:
        R = 2 * r                              # surround box half-size
        lo = ignore_border + R
        ys = np.arange(lo, h - lo, stride)
        xs = np.arange(lo, w - lo, stride)
        if not len(xs) or not len(ys):
            continue
        Y, X = np.meshgrid(ys, xs, indexing="ij")

        def box_sum(half):
            return (ii[Y + half, X + half] - ii[Y - half, X + half]
                    - ii[Y + half, X - half] + ii[Y - half, X - half])

        inner, outer = box_sum(r), box_sum(R)
        n_in, n_out = (2 * r) ** 2, (2 * R) ** 2
        resp = (outer - inner) / (n_out - n_in) - inner / n_in
        k = np.unravel_index(np.argmax(resp), resp.shape)
        if resp[k] > best[3]:
            best = (int(X[k]), int(Y[k]), r, float(resp[k]))
    return best[0], best[1], best[2]


def _ellipse_distance(pts: np.ndarray, ellipse) -> np.ndarray:
    """Approximate distance (px) from points to an OpenCV RotatedRect ellipse."""
    (cx, cy), (w, h), ang = ellipse
    a, b = max(w / 2, 1e-3), max(h / 2, 1e-3)
    t = np.radians(ang)
    dx, dy = pts[:, 0] - cx, pts[:, 1] - cy
    x = dx * np.cos(t) + dy * np.sin(t)
    y = -dx * np.sin(t) + dy * np.cos(t)
    r = np.sqrt((x / a) ** 2 + (y / b) ** 2)
    return np.abs(r - 1) * np.hypot(x, y) / np.maximum(r, 1e-6)


def _perimeter(ellipse, n: int = 72) -> np.ndarray:
    (cx, cy), (w, h), ang = ellipse
    t = np.radians(ang)
    s = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x, y = w / 2 * np.cos(s), h / 2 * np.sin(s)
    return np.stack([cx + x * np.cos(t) - y * np.sin(t), cy + x * np.sin(t) + y * np.cos(t)], axis=1)


def detect_pupil_edges(image: np.ndarray, params: DetectorParams | None = None,
                       rng: np.random.Generator | None = None) -> PupilResult:
    p = params or DetectorParams()
    rng = rng or np.random.default_rng(0)
    gray_full = to_gray(image)
    ox, oy = 0, 0
    gray = gray_full
    if p.roi:
        x, y, w, h = p.roi
        gray, ox, oy = gray_full[y:y + h, x:x + w], x, y
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 1. coarse position and size
    cx0, cy0, r0 = coarse_pupil(blur, p.edge_radii, ignore_border=p.ignore_border)
    result = PupilResult(found=False, darkest=(cx0 + ox, cy0 + oy))
    # a reflection over part of the pupil makes the coarse box too small, so never search a tiny area
    half = int(max(p.edge_roi_scale * r0, p.edge_roi_min))
    x0, y0 = max(0, cx0 - half), max(0, cy0 - half)
    x1, y1 = min(gray.shape[1], cx0 + half), min(gray.shape[0], cy0 + half)
    roi = blur[y0:y1, x0:x1]
    centre = blur[max(0, cy0 - r0):cy0 + r0, max(0, cx0 - r0):cx0 + r0]
    if roi.size == 0 or centre.size == 0:
        return result

    # 2. pupil-dark edges, away from bright reflections
    pupil_level = float(np.percentile(centre, 25))
    surround_level = float(np.median(roi))
    dark = (roi <= pupil_level + p.edge_intensity_range).astype(np.uint8)
    near_dark = cv2.dilate(dark, np.ones((7, 7), np.uint8))
    bright = (roi >= max(surround_level, pupil_level + p.edge_intensity_range) + p.edge_bright_margin)
    near_bright = cv2.dilate(bright.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    edges = cv2.Canny(roi, p.edge_canny_low, p.edge_canny_high, L2gradient=True)
    edges[(near_dark == 0) | near_bright] = 0
    ys, xs = np.nonzero(edges)
    if len(xs) < 12:
        return result
    pts = np.stack([xs, ys], axis=1).astype(np.float32)

    # 3. RANSAC: the ellipse most edge pixels lie on
    best, best_n = None, 0
    min_ax, max_ax = 0.4 * r0, 1.8 * half
    for _ in range(p.edge_iterations):
        sample = pts[rng.choice(len(pts), 5, replace=False)]
        try:
            e = cv2.fitEllipse(sample)
        except cv2.error:
            continue
        w_, h_ = e[1]
        if not (np.isfinite(w_) and np.isfinite(h_)) or min(w_, h_) < min_ax or max(w_, h_) > max_ax \
                or max(w_, h_) > p.max_aspect * max(min(w_, h_), 1e-3):
            continue
        n = int(np.count_nonzero(_ellipse_distance(pts, e) < p.edge_inlier_px))
        if n > best_n:
            best, best_n = e, n
    if best is None or best_n < 8:
        return result
    for _ in range(2):  # refit on the inliers
        inl = pts[_ellipse_distance(pts, best) < p.edge_inlier_px]
        if len(inl) < 6:
            break
        best = cv2.fitEllipse(inl)

    (ecx, ecy), (w_, h_), ang = best
    if not np.isfinite([ecx, ecy, w_, h_]).all() or min(w_, h_) <= 0 or max(w_, h_) > p.max_aspect * min(w_, h_):
        return result
    # the inside must be pupil-dark (ignoring reflections)
    mask = np.zeros(roi.shape, np.uint8)
    cv2.ellipse(mask, best, 1, -1)
    inside = mask.astype(bool) & ~bright
    if not inside.any() or np.median(roi[inside]) > pupil_level + 2 * p.edge_intensity_range:
        return result

    # 4. confidence: edge support around the perimeter that isn't hidden by a reflection
    dist = cv2.distanceTransform((edges == 0).astype(np.uint8), cv2.DIST_L2, 3)
    per = _perimeter(best)
    px = np.clip(per[:, 0].round().astype(int), 0, roi.shape[1] - 1)
    py = np.clip(per[:, 1].round().astype(int), 0, roi.shape[0] - 1)
    inside_roi = (per[:, 0] >= 0) & (per[:, 0] < roi.shape[1]) & (per[:, 1] >= 0) & (per[:, 1] < roi.shape[0])
    visible = inside_roi & ~near_bright[py, px]
    supported = visible & (dist[py, px] <= 2.0)
    confidence = float(supported.sum() / max(visible.sum(), 0.5 * len(per)))

    result.found = True
    result.cx, result.cy = ecx + x0 + ox, ecy + y0 + oy
    result.axis_minor, result.axis_major = (w_, h_) if w_ <= h_ else (h_, w_)
    result.angle = ang
    result.confidence = confidence
    result.ellipse = ((result.cx, result.cy), (w_, h_), ang)
    return result
