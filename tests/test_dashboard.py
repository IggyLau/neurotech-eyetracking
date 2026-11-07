"""Camera dashboard logic and drawing, on synthetic frames (no cameras)."""

import numpy as np

from eyetrack.capture.camera import Frame
from eyetrack.capture.dashboard import (CamStats, assign_roles, camera_status, classify, compose, render_tile)
from eyetrack.config import load_config
from eyetrack.pupil.detector import DetectorParams, detect_pupil

from synthetic import eye_image


def test_classify_tells_nir_from_colour_and_black():
    nir = eye_image(320, 200)                                   # grey, 3 identical channels
    colour = np.random.default_rng(0).integers(0, 255, (400, 640, 3), dtype=np.uint8)
    black = np.zeros((800, 1280, 3), np.uint8)
    assert classify(nir, (640, 400)) == "nir"
    assert classify(colour, (640, 480)) == "colour"
    assert classify(black, (1280, 800)) == "nir_black"         # OV9281 native size, exposure reset
    assert classify(black[:720], (1920, 1080)) == "black"
    assert classify(None, (0, 0)) == "none"


def test_assign_roles_takes_nir_cameras_as_the_eyes():
    found = [{"index": 0, "kind": "nir"}, {"index": 1, "kind": "colour"}, {"index": 2, "kind": "colour"},
             {"index": 3, "kind": "nir_black"}]
    assert assign_roles(found, scene_index=1) == {"eye": 0, "eye2": 3, "scene": 1}
    assert assign_roles(found[:2], scene_index=0) == {"eye": 0}     # configured scene index is an eye camera


def _feed(st, fps, seconds, drop_every=0, t0=100.0):
    n = int(fps * seconds)
    k = 0
    for i in range(n):
        if drop_every and i % drop_every == 0 and i:
            continue
        st.on_frame(Frame(None, int((t0 + i / fps) * 1e9), k))
        k += 1


def test_stats_measure_fps_and_dropped_frames():
    st = CamStats(120)
    _feed(st, 120, 3, drop_every=50)
    now = st.stamps[-1]
    assert abs(st.measured_fps(now) - 120) < 4
    assert st.gaps == len(range(50, 360, 50))
    level, _ = camera_status(st, now, brightness=80, nir=False)
    assert level == "ok"
    slow = CamStats(120)
    _feed(slow, 30, 3)
    assert camera_status(slow, slow.stamps[-1], 80, False)[0] == "bad"
    assert camera_status(st, now + 2, 80, False)[1] == "NO FRAMES"
    assert camera_status(st, now, 0.0, False)[1] == "BLACK FRAMES"


def test_pupil_stats_and_tile_rendering():
    cfg = load_config()
    params = DetectorParams.from_config(cfg["pupil"])
    st = CamStats(120, cfg["pupil"]["min_confidence"])
    _feed(st, 120, 2)
    t_end = st.stamps[-1]
    rng = np.random.default_rng(1)
    for k in range(40):
        img = eye_image(320 + rng.normal(0, 0.5), 200 + rng.normal(0, 0.5))
        r = detect_pupil(img, params)
        st.add_pupil(t_end - 1.0 + k * 0.025, r)
    assert st.valid_fraction(t_end) > 0.9
    assert st.precision_px(t_end) < 3
    tile = render_tile("eye", 0, img, st, t_end, True, r, params.search_zone, mirror=True)
    scene = render_tile("scene", 1, np.full((720, 1280, 3), 120, np.uint8), CamStats(30), t_end, False)
    board = compose([tile, tile, scene])
    assert board.shape[0] == 2 * (tile.shape[0] + 8) and board.shape[1] == 2 * (tile.shape[1] + 8)
    assert tile[5, 5].tolist() != [0, 0, 0]          # coloured status header
