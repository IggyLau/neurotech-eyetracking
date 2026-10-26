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


def _screen_reflection(img, x0, y0, x1, y1, grey=130):
    import cv2
    cv2.rectangle(img, (x0, y0), (x1, y1), (grey, grey, grey), -1)  # a lit screen reflected on the cornea
    return img


def test_visible_outline_scores_a_pupil_partly_covered_by_a_reflection():
    img = _screen_reflection(eye_image(320, 200, (50, 50), glint=False), 330, 195, 380, 235)
    r = detect_pupil(img, DetectorParams())
    rv = detect_pupil(img, DetectorParams(refine="visible"))
    assert rv.found and np.hypot(rv.cx - 320, rv.cy - 200) < 2.5
    assert rv.confidence > r.confidence and rv.confidence > 0.8


@pytest.mark.parametrize("cx,cy,axes,angle", [(320, 200, (40, 40), 0), (250, 180, (30, 44), 30)])
def test_edge_method_finds_pupil(cx, cy, axes, angle):
    r = detect_pupil(eye_image(cx, cy, axes, angle), DetectorParams(method="edge"))
    assert r.found and abs(r.cx - cx) < 1.5 and abs(r.cy - cy) < 1.5
    assert r.diameter_px == pytest.approx(max(axes), rel=0.1)
    assert r.confidence > 0.6


def test_edge_method_ignores_arc_hidden_by_reflection():
    img = _screen_reflection(eye_image(320, 200, (50, 50), glint=False), 330, 195, 380, 235)
    r = detect_pupil(img, DetectorParams(method="edge"))
    assert r.found and np.hypot(r.cx - 320, r.cy - 200) < 2.5
    assert r.diameter_px == pytest.approx(50, rel=0.1)
