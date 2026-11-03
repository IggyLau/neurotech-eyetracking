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


def test_edge_penalty_prefers_a_central_pupil_over_a_dark_border():
    import numpy as np
    from eyetrack.pupil.detector import edge_weight, get_dark_seeds
    img = np.full((400, 640), 120, np.uint8)
    img[30:90, 30:90] = 5                       # dark corner (glasses frame / hair)
    img[180:220, 300:340] = 15                  # pupil, a little less dark
    assert get_dark_seeds(img)[0][0] < 100      # no penalty: the corner wins
    seed = get_dark_seeds(img, edge_penalty=30, edge_margin=0.2)[0]
    assert abs(seed[0] - 320) < 15 and abs(seed[1] - 200) < 15
    w = edge_weight((400, 640), 0.2)
    assert w[0, 320] == 1 and w[200, 320] == 0 and 0 < w[40, 320] < 1


def test_search_zone_ignores_a_darker_blob_outside_it():
    import numpy as np
    import cv2
    from eyetrack.pupil.detector import DetectorParams, detect_pupil
    img = np.full((400, 640), 140, np.uint8)
    cv2.circle(img, (60, 80), 45, 0, -1)         # very dark blob at the edge (hair / frame)
    cv2.circle(img, (440, 150), 35, 20, -1)      # pupil near the zone's right edge, outline crosses it
    off = detect_pupil(img, DetectorParams())
    assert off.found and off.cx < 200            # without a zone the edge blob wins
    r = detect_pupil(img, DetectorParams(search_zone=(0.25, 0.20, 0.75, 0.80)))
    assert r.found and abs(r.cx - 440) < 3 and abs(r.cy - 150) < 3
    assert 66 < r.diameter_px < 74               # whole outline kept, not clipped at the zone edge (x 480)


def test_search_zone_with_nothing_inside_finds_nothing():
    import numpy as np
    import cv2
    from eyetrack.pupil.detector import DetectorParams, detect_pupil
    img = np.full((400, 640), 140, np.uint8)
    cv2.circle(img, (60, 80), 45, 0, -1)
    assert not detect_pupil(img, DetectorParams(search_zone=(0.25, 0.20, 0.75, 0.80))).found


def test_surround_fraction_pupil_vs_edge_blob():
    import numpy as np
    import cv2
    from eyetrack.pupil.detector import surround_fraction
    img = np.full((400, 640), 140, np.uint8)
    cv2.circle(img, (320, 200), 35, 20, -1)                     # pupil: brighter all round
    cv2.rectangle(img, (0, 0), (60, 120), 10, -1)               # dark blob running off the top-left corner
    find = lambda x, y: max(cv2.findContours(((img < 60) * 255).astype(np.uint8), cv2.RETR_EXTERNAL,
                                             cv2.CHAIN_APPROX_NONE)[0],
                            key=lambda c: -np.hypot(*(c.reshape(-1, 2).mean(0) - (x, y))))
    assert surround_fraction(img, find(320, 200)) == 1.0
    assert surround_fraction(img, find(30, 60)) <= 0.5


def test_surround_picks_the_pupil_over_a_darker_edge_blob():
    import numpy as np
    import cv2
    from eyetrack.pupil.detector import DetectorParams, detect_pupil
    img = np.full((400, 640), 140, np.uint8)
    cv2.circle(img, (40, 200), 50, 0, -1)          # darker, bigger blob cut off by the left edge
    cv2.circle(img, (380, 210), 35, 20, -1)
    assert detect_pupil(img, DetectorParams(n_seeds=3)).cx < 150
    r = detect_pupil(img, DetectorParams(n_seeds=3, surround=2))
    assert abs(r.cx - 380) < 3


def test_zone_skips_a_shadow_reaching_into_it_and_finds_the_pupil():
    import numpy as np
    import cv2
    from eyetrack.pupil.detector import DetectorParams, detect_pupil
    img = np.full((400, 640), 120, np.uint8)
    cv2.rectangle(img, (0, 40), (175, 300), 8, -1)     # shadow from the left edge, just into the zone (x >= 160)
    cv2.circle(img, (360, 185), 26, 20, -1)            # pupil, less dark than the shadow
    cv2.circle(img, (352, 176), 5, 255, -1)            # glint
    r = detect_pupil(img, DetectorParams(search_zone=(0.25, 0.20, 0.75, 0.80)))
    assert r.found and abs(r.cx - 360) < 4 and abs(r.cy - 185) < 4
