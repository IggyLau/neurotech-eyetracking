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
    # Soft bias against dark regions near the image border (glasses frame, hair, mount): grey levels added
    # to the darkest-block search at the edge, fading linearly to 0 at edge_margin x width/height inward
    edge_penalty: float = 0.0
    edge_margin: float = 0.2
    # The pupil centre must lie in this box, as fractions of the full image [x0, y0, x1, y1]. Unlike roi, the
    # image isn't cropped, so a pupil near the box edge keeps its whole outline. None = anywhere
    search_zone: tuple[float, float, float, float] | None = None
    # Surround check: a pupil is darker than the iris on every side; hair or a frame edge isn't. The ring just
    # outside the candidate is split into surround_sectors directions, and each counts if it is at least
    # surround_margin grey levels brighter than the inside (off the image counts as not brighter). The candidate
    # score is multiplied by (fraction of sectors that count) ** surround; 0 = off
    surround: float = 0.0
    surround_margin: int = 10
    surround_sectors: int = 8
    roi: tuple[int, int, int, int] | None = None   # x, y, w, h
    glint_kernel: int = 0         # >0: grey-level opening that erases bright glints/reflections smaller than this
    median_blur: int = 0          # >0: median filter (odd px) against sensor noise / eyelashes
    close_kernel: int = 5         # morphological closing kernel that fills holes in the dark blob
    close_iterations: int = 2
    refine: str = "angle"         # contour clean-up before the final fit: "angle" (JEO), "hull", "none"
    n_seeds: int = 1              # dark candidate regions to try (1 = JEO: darkest only)
    score: str = "jeo"            # candidate ranking: "jeo" (favours big blobs) or "shape" (ellipse fit x contrast)
    bright_margin: int = 60       # refine "visible": contour points next to pixels this much brighter than the pupil are dropped
    method: str = "threshold"     # "threshold" (JEO-based, this file), "edge" (edge_detector.py) or
                                  # "pupil_labs" (pupil_labs.py, needs the pupil-detectors package)
    pl_pupil_size_min: int = 10   # pupil_labs only: pupil diameter range, px
    pl_pupil_size_max: int = 150
    # edge method only
    edge_radii: tuple[int, ...] = (8, 12, 17, 24, 34, 48)
    edge_roi_scale: float = 3.0
    edge_roi_min: int = 60
    edge_intensity_range: int = 20
    edge_bright_margin: int = 40
    edge_canny_low: int = 30
    edge_canny_high: int = 70
    edge_inlier_px: float = 1.5
    edge_iterations: int = 150

    @classmethod
    def from_config(cls, pcfg: dict) -> "DetectorParams":
        keys = ["search_box", "min_area", "max_aspect", "glint_kernel", "median_blur",
                "close_kernel", "close_iterations", "refine", "n_seeds", "score", "block", "bright_margin",
                "edge_penalty", "edge_margin", "surround", "surround_margin", "surround_sectors",
                "method", "edge_roi_scale", "edge_roi_min", "edge_intensity_range", "edge_bright_margin", "edge_canny_low",
                "edge_canny_high", "edge_inlier_px", "edge_iterations", "pl_pupil_size_min", "pl_pupil_size_max"]
        kw = {k: pcfg[k] for k in keys if k in pcfg}
        if "edge_radii" in pcfg:
            kw["edge_radii"] = tuple(pcfg["edge_radii"])
        return cls(threshold_offsets=tuple(pcfg.get("threshold_offsets", (5, 15, 25))),
                   roi=tuple(pcfg["roi"]) if pcfg.get("roi") else None,
                   search_zone=tuple(pcfg["search_zone"]) if pcfg.get("search_zone") else None, **kw)


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


def edge_weight(shape: tuple[int, int], margin: float) -> np.ndarray:
    """1 at the image border, falling linearly to 0 at margin x width (x) / height (y) inward."""
    h, w = shape
    x = np.minimum(np.arange(w), np.arange(w)[::-1]) / max(margin * w, 1)
    y = np.minimum(np.arange(h), np.arange(h)[::-1]) / max(margin * h, 1)
    return np.clip(1 - np.minimum.outer(y, x), 0, 1).astype(np.float32)


def get_dark_seeds(gray: np.ndarray, n: int = 1, block: int = 20, ignore_border: int = 20,
                   min_dist: int = 100, edge_penalty: float = 0.0, edge_margin: float = 0.2,
                   zone: tuple[int, int, int, int] | None = None) -> list[tuple[int, int]]:
    """Up to n darkest block centers, at least min_dist apart (n=1 is JEO's single darkest point).

    Several seeds let the detector try the pupil *and* e.g. a mascara clump or a glasses
    frame, then keep whichever looks most like a pupil.
    """
    mean = cv2.blur(gray, (block, block)).astype(np.float32)
    if edge_penalty:
        mean += edge_penalty * edge_weight(mean.shape, edge_margin)
    if zone is not None:                    # x0, y0, x1, y1 in px: seeds only inside
        x0, y0, x1, y1 = zone
        inside = np.zeros(mean.shape, bool)
        inside[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)] = True
        mean[~inside] = np.inf
    b = ignore_border + block // 2
    if mean.shape[0] > 2 * b and mean.shape[1] > 2 * b:
        mean[:b, :] = np.inf
        mean[-b:, :] = np.inf
        mean[:, :b] = np.inf
        mean[:, -b:] = np.inf
    seeds = []
    for _ in range(max(1, n)):
        _, _, loc, _ = cv2.minMaxLoc(mean)
        if not np.isfinite(mean[loc[1], loc[0]]):
            break
        seeds.append((int(loc[0]), int(loc[1])))
        cv2.circle(mean, loc, min_dist, float("inf"), -1)
    return seeds


def ellipse_contrast(gray: np.ndarray, contour: np.ndarray, ring_px: int = 12) -> float:
    """(mean grey just outside the fitted ellipse - mean inside) / 255. Pupils are much darker than the iris."""
    mask = np.zeros(gray.shape, np.uint8)
    cv2.ellipse(mask, cv2.fitEllipse(contour), 255, -1)
    ring = cv2.dilate(mask, np.ones((2 * ring_px + 1, 2 * ring_px + 1), np.uint8)) & ~mask
    if not mask.any() or not ring.any():
        return 0.0
    return float(cv2.mean(gray, ring)[0] - cv2.mean(gray, mask)[0]) / 255.0


def surround_fraction(gray: np.ndarray, contour: np.ndarray, margin: int = 10, sectors: int = 8,
                      ring_px: int = 12, n_angles: int = 64) -> float:
    """Share of directions around the fitted ellipse where the band just outside it is at least `margin` grey
    levels brighter than the inside. A pupil scores ~1; a dark blob at the image edge or against hair scores
    lower. Sampled at n_angles points per ring (cheap enough for every candidate at 120 fps); directions
    whose samples fall off the image count as not brighter."""
    (cx, cy), (w, h), ang = cv2.fitEllipse(contour)
    H, W = gray.shape
    if not np.isfinite([cx, cy, w, h]).all() or min(w, h) <= 0 or max(w, h) > max(H, W):
        return 0.0                          # degenerate fits can be thousands of px across: not a pupil
    t = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    c, s_ = np.cos(np.radians(ang)), np.sin(np.radians(ang))

    def sample(da: float, scale: float = 1.0):
        x = (w / 2 * scale + da) * np.cos(t)
        y = (h / 2 * scale + da) * np.sin(t)
        px = np.rint(cx + x * c - y * s_).astype(int)
        py = np.rint(cy + x * s_ + y * c).astype(int)
        ok = (px >= 0) & (px < W) & (py >= 0) & (py < H)
        val = np.full(n_angles, np.nan, np.float32)
        val[ok] = gray[py[ok], px[ok]]
        return val

    inside = np.concatenate([sample(0, k) for k in (0.25, 0.5, 0.7)])
    if np.isnan(inside).all():
        return 0.0
    inside_mean = float(np.nanmean(inside))
    outside = np.stack([sample(d) for d in np.linspace(ring_px / 3, ring_px, 3)])   # 3 x n_angles
    per_sector = outside.reshape(3, sectors, n_angles // sectors)
    good = 0
    for k in range(sectors):
        vals = per_sector[:, k]
        if np.isnan(vals).mean() > 0.5:      # mostly off the image
            continue
        good += float(np.nanmean(vals)) - inside_mean >= margin
    return good / sectors


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


def visible_contour(contour: np.ndarray, gray: np.ndarray, pupil_value: int, bright_margin: int) -> np.ndarray:
    """Contour points that border the iris, not a reflection.

    Where a bright reflection covers part of the pupil, the dark blob's outline follows the reflection
    instead of the pupil edge. Those points sit next to pixels far brighter than the pupil, so drop them
    and fit the ellipse to the rest of the outline.
    """
    bright = (gray >= pupil_value + bright_margin).astype(np.uint8)
    near = cv2.dilate(bright, np.ones((5, 5), np.uint8))
    pts = contour.reshape(-1, 2)
    keep = near[pts[:, 1], pts[:, 0]] == 0
    return contour[keep] if keep.sum() >= max(6, 0.4 * len(pts)) else contour


def detect_pupil(image: np.ndarray, params: DetectorParams | None = None) -> PupilResult:
    params = params or DetectorParams()
    if params.method == "edge":
        from .edge_detector import detect_pupil_edges
        return detect_pupil_edges(image, params)
    if params.method == "pupil_labs":
        from .pupil_labs import detect_pupil_pl
        return detect_pupil_pl(image, params)
    gray_full = to_gray(image)
    ox, oy = 0, 0
    gray = gray_full
    if params.roi:
        x, y, w, h = params.roi
        gray = gray_full[y:y + h, x:x + w]
        ox, oy = x, y

    if params.median_blur:
        gray = cv2.medianBlur(gray, params.median_blur)
    if params.glint_kernel:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (params.glint_kernel, params.glint_kernel))
        gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, k)
    close_k = np.ones((params.close_kernel, params.close_kernel), np.uint8)

    zone = None
    if params.search_zone:
        H, W = gray_full.shape
        fx0, fy0, fx1, fy1 = params.search_zone
        zone = (int(fx0 * W) - ox, int(fy0 * H) - oy, int(fx1 * W) - ox, int(fy1 * H) - oy)
    # With a zone, a shadow reaching just inside it can be the darkest spot; its blob is centred outside and
    # rejected. Then the next darkest spots are tried in order and the first that passes is kept, so the
    # darkest valid spot still wins (unlike n_seeds > 1, which keeps the best-scoring of all).
    first_pass = zone is not None and params.n_seeds == 1
    seeds = get_dark_seeds(gray, 3 if first_pass else params.n_seeds, params.block, params.ignore_border,
                           params.search_box // 2, params.edge_penalty, params.edge_margin, zone)
    if not seeds:
        return PupilResult(found=False)
    darkest = seeds[0]
    blurred = cv2.blur(gray, (5, 5))

    best_score, best_contour, best_ratio, best_value = 0.0, None, 0.0, 0
    for seed in seeds:
        seed_value = int(blurred[seed[1], seed[0]])
        for offset in sorted(params.threshold_offsets, reverse=True):  # relaxed -> strict, as in JEO
            binary = mask_outside_square(apply_binary_threshold(gray, seed_value, offset),
                                         seed, params.search_box)
            # JEO dilates here to fill glints/eyelash gaps; a closing (dilate then erode) fills
            # them the same way without inflating the pupil, so diameters stay unbiased
            closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_k, iterations=params.close_iterations)
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour = filter_contours_by_area_and_return_largest(contours, params.min_area, params.max_aspect)
            if contour is None or len(contour) <= 5:
                continue
            if zone is not None:            # the blob's centre has to be in the zone, not just its seed
                mo = cv2.moments(contour)
                if mo["m00"] <= 0 or not (zone[0] <= mo["m10"] / mo["m00"] < zone[2]
                                          and zone[1] <= mo["m01"] / mo["m00"] < zone[3]):
                    continue
            filled = check_ellipse_goodness(closed, contour)
            n_thick, ratio = check_contour_pixels(contour, closed.shape)
            if params.score == "jeo":
                score = filled * n_thick * n_thick * ratio
            else:  # "shape": elliptical, solid and darker than its surroundings - size-independent
                score = filled * ratio * max(ellipse_contrast(gray, contour), 0.0)
            if params.surround:
                score *= surround_fraction(gray, contour, params.surround_margin,
                                           params.surround_sectors) ** params.surround
            if score > best_score:
                best_score, best_contour, best_ratio, best_value = score, contour, ratio, seed_value
        if first_pass and best_contour is not None:
            break

    result = PupilResult(found=False, darkest=(darkest[0] + ox, darkest[1] + oy))
    if best_contour is None:
        return result
    if params.refine == "angle":
        refined = optimize_contours_by_angle(best_contour)
    elif params.refine == "hull":  # convex hull bridges bites taken out by reflections / eyelids
        refined = cv2.convexHull(best_contour)
    elif params.refine == "visible":  # fit only the outline that borders the iris
        refined = optimize_contours_by_angle(visible_contour(best_contour, gray, best_value, params.bright_margin))
    else:
        refined = best_contour
    if len(refined) < 5:
        refined = best_contour
    (cx, cy), (w, h), angle = cv2.fitEllipse(refined)
    if not np.isfinite([cx, cy, w, h]).all() or min(w, h) <= 0:
        return result
    if zone is not None and not (zone[0] <= cx < zone[2] and zone[1] <= cy < zone[3]):
        return result                       # the fitted ellipse ran off the zone (e.g. merged with a shadow)
    result.found = True
    result.cx, result.cy = cx + ox, cy + oy
    result.axis_minor, result.axis_major = (w, h) if w <= h else (h, w)
    result.angle = angle
    result.confidence = best_ratio
    if params.refine == "visible":  # share of the visible outline on the final ellipse
        band = np.zeros(gray.shape, np.uint8)
        cv2.ellipse(band, ((cx, cy), (w, h), angle), 255, 4)
        pts = refined.reshape(-1, 2)
        result.confidence = float(np.mean(band[pts[:, 1], pts[:, 0]] > 0))
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
