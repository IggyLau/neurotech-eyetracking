"""Pupil Labs' own 2D pupil detector (pupil-detectors, LGPL-3.0), as an optional backend.

Install on macOS (it compiles from source and needs OpenCV 4; OpenCV 5 builds only with patches and then
misdetects):
    brew install eigen opencv@4
    SKBUILD_CONFIGURE_OPTIONS="-DOpenCV_DIR=$(brew --prefix opencv@4)/lib/cmake/opencv4" \
    CXXFLAGS="-I$(brew --prefix)/include/eigen3" pip install pupil-detectors
The library is used as an installed dependency, not copied into this repository.
The detector keeps the previous frame's ellipse as a prior, so feed it consecutive frames.

Its defaults are tuned for Pupil Labs' 192x192 / 400x400 eye cameras. The only change needed at
640x400 is the pupil size range (diameter in px), set from DetectorParams.
"""

from __future__ import annotations

import numpy as np

from .detector import DetectorParams, PupilResult, to_gray

_detectors: dict = {}


def _detector(params: DetectorParams):
    key = (params.pl_pupil_size_min, params.pl_pupil_size_max)
    if key not in _detectors:
        from pupil_detectors import Detector2D
        _detectors[key] = Detector2D({"pupil_size_min": params.pl_pupil_size_min,
                                      "pupil_size_max": params.pl_pupil_size_max})
    return _detectors[key]


def detect_pupil_pl(image: np.ndarray, params: DetectorParams | None = None) -> PupilResult:
    params = params or DetectorParams()
    gray = np.ascontiguousarray(to_gray(image))
    ox, oy = 0, 0
    if params.roi:
        x, y, w, h = params.roi
        gray, ox, oy = np.ascontiguousarray(gray[y:y + h, x:x + w]), x, y
    r = _detector(params).detect(gray)
    e = r["ellipse"]
    (cx, cy), (w, h), ang = e["center"], e["axes"], e["angle"]
    if r["confidence"] <= 0 or not np.isfinite([cx, cy, w, h]).all() or min(w, h) <= 0:
        return PupilResult(found=False)
    out = PupilResult(found=True, cx=cx + ox, cy=cy + oy, angle=ang, confidence=float(r["confidence"]))
    out.axis_minor, out.axis_major = (w, h) if w <= h else (h, w)
    out.ellipse = ((out.cx, out.cy), (w, h), ang)
    return out
