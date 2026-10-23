"""Dark-pupil detector adapted from JEOresearch/EyeTracker (OrloskyPupilDetector.py, MIT).

Pipeline per frame (same idea as the original):
  1. find the darkest block in the image (the pupil under off-axis IR light)
  2. threshold at darkest+5 / +15 / +25 inside a box around it
  3. keep the largest roughly-round contour from each threshold
  4. score an ellipse fit on each and keep the best
  5. drop contour points that bend away from the centroid (eyelid/glint edges), refit

Changes from the original: works on any resolution (no forced 640x480 crop),
optional ROI, vectorised darkest-block search, morphological closing instead of
dilation (unbiased diameter), no GUI/globals, and it returns a
confidence score (fraction of contour pixels lying on the fitted ellipse).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

_KERNEL = np.ones((5, 5), np.uint8)


@dataclass
class PupilResult:
    found: bool
    cx: float = np.nan
    cy: float = np.nan
    axis_minor: float = np.nan
    axis_major: float = np.nan
    angle: float = np.nan          # OpenCV RotatedRect angle of the first axis, degrees
    confidence: float = 0.0        # 0..1, fraction of contour border on the fitted ellipse
    darkest: tuple[int, int] | None = None
    ellipse: tuple | None = field(default=None, repr=False)  # raw cv2 RotatedRect in full-frame coords

    @property
    def diameter_px(self) -> float:
        # Under (near-)orthographic projection the major axis is not foreshortened,
        # so it tracks true pupil diameter better than the minor axis or area.
        return self.axis_major


@dataclass
class DetectorParams:
    threshold_offsets: tuple[int, ...] = (5, 15, 25)
    search_box: int = 250
    min_area: float = 300
    max_aspect: float = 3.0
    block: int = 20               # darkest-block size, px
    ignore_border: int = 20
    roi: tuple[int, int, int, int] | None = None   # x, y, w, h

    @classmethod
    def from_config(cls, pcfg: dict) -> "DetectorParams":
        return cls(threshold_offsets=tuple(pcfg.get("threshold_offsets", (5, 15, 25))),
                   search_box=pcfg.get("search_box", 250),
                   min_area=pcfg.get("min_area", 300),
                   max_aspect=pcfg.get("max_aspect", 3.0),
                   roi=tuple(pcfg["roi"]) if pcfg.get("roi") else None)


def to_gray(image: np.ndarray) -> np.ndarray:
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def get_darkest_area(gray: np.ndarray, block: int = 20, ignore_border: int = 20) -> tuple[int, int]:
    """Center of the darkest block x block window (box-filter version of JEO's sparse search)."""
    mean = cv2.blur(gray, (block, block)).astype(np.float32)
    b = ignore_border + block // 2
    if mean.shape[0] > 2 * b and mean.shape[1] > 2 * b:
        mean[:b, :] = np.inf
        mean[-b:, :] = np.inf
        mean[:, :b] = np.inf
        mean[:, -b:] = np.inf
    _, _, min_loc, _ = cv2.minMaxLoc(mean)
    return int(min_loc[0]), int(min_loc[1])


def apply_binary_threshold(gray: np.ndarray, darkest_value: int, added: int) -> np.ndarray:
    _, out = cv2.threshold(gray, int(darkest_value) + added, 255, cv2.THRESH_BINARY_INV)
    return out


def mask_outside_square(image: np.ndarray, center: tuple[int, int], size: int) -> np.ndarray:
    x, y = center
    half = size // 2
    mask = np.zeros_like(image)
    mask[max(0, y - half):min(image.shape[0], y + half), max(0, x - half):min(image.shape[1], x + half)] = 255
    return cv2.bitwise_and(image, mask)


def filter_contours_by_area_and_return_largest(contours, min_area: float, max_aspect: float):
    best, best_area = None, 0.0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        _, _, w, h = cv2.boundingRect(contour)
        if max(w / h, h / w) <= max_aspect and area > best_area:
            best, best_area = contour, area
    return best


def check_ellipse_goodness(binary: np.ndarray, contour: np.ndarray) -> float:
    """Fraction of the fitted ellipse's area that is filled in the binary image."""
    ellipse = cv2.fitEllipse(contour)
    mask = np.zeros_like(binary)
    cv2.ellipse(mask, ellipse, 255, -1)
    area = np.count_nonzero(mask)
    if area == 0:
        return 0.0
    return float(np.count_nonzero((binary == 255) & (mask == 255)) / area)


def check_contour_pixels(contour: np.ndarray, shape) -> tuple[int, float]:
    """(contour pixels under a thick ellipse band, fraction of contour pixels under a thin band)."""
    contour_mask = np.zeros(shape, np.uint8)
    cv2.drawContours(contour_mask, [contour], -1, 255, 1)
    ellipse = cv2.fitEllipse(contour)
    thick = np.zeros(shape, np.uint8)
    thin = np.zeros(shape, np.uint8)
    cv2.ellipse(thick, ellipse, 255, 10)
    cv2.ellipse(thin, ellipse, 255, 4)
    border = np.count_nonzero(contour_mask)
    n_thick = int(np.count_nonzero(contour_mask & thick))
    ratio = float(np.count_nonzero(contour_mask & thin) / border) if border else 0.0
    return n_thick, ratio


def optimize_contours_by_angle(contour: np.ndarray) -> np.ndarray:
    """Keep contour points whose local curvature points toward the centroid (convex pupil edge)."""
    pts = contour.reshape(-1, 2).astype(np.float64)
    n = len(pts)
    spacing = max(1, n // 25)
    centroid = pts.mean(axis=0)
    prev_pts = np.roll(pts, spacing, axis=0)
    next_pts = np.roll(pts, -spacing, axis=0)
    bisector = ((prev_pts - pts) + (next_pts - pts)) / 2
    to_centroid = centroid - pts
    keep = np.einsum("ij,ij->i", to_centroid, bisector) >= np.cos(np.radians(60))
    return pts[keep].astype(np.int32).reshape(-1, 1, 2)


def detect_pupil(image: np.ndarray, params: DetectorParams | None = None) -> PupilResult:
    params = params or DetectorParams()
    gray_full = to_gray(image)
    ox, oy = 0, 0
    gray = gray_full
    if params.roi:
        x, y, w, h = params.roi
        gray = gray_full[y:y + h, x:x + w]
        ox, oy = x, y

    darkest = get_darkest_area(gray, params.block, params.ignore_border)
    darkest_value = int(cv2.blur(gray, (5, 5))[darkest[1], darkest[0]])

    best_score, best_contour, best_ratio = 0.0, None, 0.0
    for offset in sorted(params.threshold_offsets, reverse=True):  # relaxed -> strict, as in JEO
        binary = mask_outside_square(apply_binary_threshold(gray, darkest_value, offset),
                                     darkest, params.search_box)
        # JEO dilates here to fill glints/eyelash gaps; a closing (dilate then erode) fills
        # them the same way without inflating the pupil, so diameters stay unbiased
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, _KERNEL, iterations=2)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = filter_contours_by_area_and_return_largest(contours, params.min_area, params.max_aspect)
        if contour is None or len(contour) <= 5:
            continue
        filled = check_ellipse_goodness(closed, contour)
        n_thick, ratio = check_contour_pixels(contour, closed.shape)
        score = filled * n_thick * n_thick * ratio
        if score > best_score:
            best_score, best_contour, best_ratio = score, contour, ratio

    result = PupilResult(found=False, darkest=(darkest[0] + ox, darkest[1] + oy))
    if best_contour is None:
        return result
    refined = optimize_contours_by_angle(best_contour)
    if len(refined) < 5:
        refined = best_contour
    (cx, cy), (w, h), angle = cv2.fitEllipse(refined)
    if not np.isfinite([cx, cy, w, h]).all() or min(w, h) <= 0:
        return result
    result.found = True
    result.cx, result.cy = cx + ox, cy + oy
    result.axis_minor, result.axis_major = (w, h) if w <= h else (h, w)
    result.angle = angle
    result.confidence = best_ratio
    result.ellipse = ((cx + ox, cy + oy), (w, h), angle)
    return result


def draw_result(image: np.ndarray, r: PupilResult) -> np.ndarray:
    out = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if r.found:
        cv2.ellipse(out, r.ellipse, (20, 255, 255), 2)
        cv2.circle(out, (int(r.cx), int(r.cy)), 3, (255, 255, 0), -1)
        cv2.putText(out, f"conf {r.confidence:.2f}  d {r.diameter_px:.1f}px", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(out, "no pupil", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return out
