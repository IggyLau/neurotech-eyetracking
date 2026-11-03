import numpy as np

from eyetrack.realtime.status import TrackerStatus, draw_status


def rows(n=240, fps=120, valid=1.0, cx=320, cy=200, brightness=120, t0=0.0):
    return [{"t_ns": int((t0 + k / fps) * 1e9), "valid": k < valid * n, "cx": cx, "cy": cy,
             "frame_brightness": brightness} for k in range(n)]


def status_of(r, **kw):
    s = TrackerStatus((400, 640))
    s.add(r, now=1.0)
    return s.status(now=1.0, **kw)


def test_good_tracking_is_ok():
    assert status_of(rows()).level == "ok"


def test_each_failure_is_flagged():
    assert status_of(rows(valid=0.1)).text.startswith("PUPIL LOST")
    assert status_of(rows(brightness=0)).text.startswith("CAMERA DARK")
    assert status_of(rows(cx=630)).text.startswith("PUPIL NEAR EDGE")
    assert status_of(rows(valid=0.6)).level == "weak"
    assert status_of(rows(fps=60, n=120)).text.startswith("LOW FPS")
    assert status_of(rows(), video_ok=False).text.startswith("VIDEO NOT SAVING")


def test_no_new_frames_is_lost():
    s = TrackerStatus((400, 640))
    assert s.status(now=0.0).text.startswith("NO FRAMES")
    s.add(rows(), now=1.0)
    assert s.status(now=2.0).text.startswith("NO FRAMES")


def test_draw_status_marks_bottom_right_only():
    img = np.full((900, 1440, 3), 128, np.uint8)
    out = draw_status(img.copy(), status_of(rows()))
    changed = np.argwhere((out != 128).any(axis=2))
    assert changed[:, 0].min() > 850 and changed[:, 1].min() > 1100
