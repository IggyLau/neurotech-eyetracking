import numpy as np
import pytest

from eyetrack.pupil.detector import DetectorParams, detect_pupil

from synthetic import eye_image


@pytest.mark.parametrize("cx,cy,axes,angle", [
    (320, 200, (40, 40), 0),
    (250, 180, (30, 44), 30),
    (420, 230, (50, 56), 75),
    (300, 150, (24, 26), 0),
])
def test_finds_pupil_center_and_size(cx, cy, axes, angle):
    r = detect_pupil(eye_image(cx, cy, axes, angle), DetectorParams())
    assert r.found
    assert abs(r.cx - cx) < 1.0 and abs(r.cy - cy) < 1.0
    assert r.diameter_px == pytest.approx(max(axes), rel=0.08)
    assert r.confidence > 0.6


def test_closed_eye_is_not_a_confident_pupil():
    img = np.full((400, 640, 3), 150, np.uint8)
    img[195:205, 100:540] = 70  # eyelash line of a closed eye
    r = detect_pupil(img, DetectorParams())
    assert not r.found or r.confidence < 0.6


def test_glint_removal_and_hull_handle_big_reflection():
    img = eye_image(320, 200, (50, 50), glint=False)
    import cv2
    cv2.rectangle(img, (325, 170), (360, 190), (240, 240, 240), -1)  # window reflection biting the pupil edge
    r = detect_pupil(img, DetectorParams(glint_kernel=0, refine="angle"))
    r2 = detect_pupil(img, DetectorParams(glint_kernel=41, refine="hull"))
    assert r2.found and abs(r2.cx - 320) < 2 and abs(r2.cy - 200) < 2
    assert np.hypot(r2.cx - 320, r2.cy - 200) <= np.hypot(r.cx - 320, r.cy - 200) + 0.5


def test_size_outlier_flags_iris_frames():
    from eyetrack.pipeline import size_outliers
    d = np.full(200, 40.0)
    d[100:103] = 120.0  # iris picked for 3 frames
    out = size_outliers(d, np.ones(200, bool), fps=120, max_ratio=1.6)
    assert out[100:103].all() and out.sum() == 3


def test_roi_offsets_are_mapped_back_to_full_frame():
    img = eye_image(450, 220, (36, 36))
    r = detect_pupil(img, DetectorParams(roi=(300, 100, 300, 250)))
    assert r.found
    assert abs(r.cx - 450) < 1.0 and abs(r.cy - 220) < 1.0
