"""Frame-rate watchdog in Camera, with a stand-in capture (no hardware needed)."""

import numpy as np

from eyetrack.capture import camera as camera_mod
from eyetrack.capture.camera import Camera


class FakeCapture:
    opened = 0

    def __init__(self, *args):
        FakeCapture.opened += 1
        self.released = False

    def read(self):
        return True, np.zeros((4, 4, 3), np.uint8)

    def release(self):
        self.released = True


def _camera(monkeypatch, uvc=True):
    FakeCapture.opened = 0
    monkeypatch.setattr(camera_mod, "open_capture", FakeCapture)
    cam = Camera(0, fps=120, uvc={"exposure_ms": 8} if uvc else None, min_fps=100)
    cam.resent = 0
    monkeypatch.setattr(cam, "_apply_uvc", lambda: setattr(cam, "resent", cam.resent + 1))
    return cam


def _feed(cam, fps, seconds, start_s=0.0):
    for i in range(int(fps * seconds)):
        cam._watchdog(int((start_s + i / fps) * 1e9), True)


def test_full_rate_does_nothing(monkeypatch):
    cam = _camera(monkeypatch)
    _feed(cam, 120, 10)
    assert cam.recoveries == [] and FakeCapture.opened == 1


def test_low_rate_resends_uvc_then_reopens(monkeypatch):
    cam = _camera(monkeypatch)
    first = cam.cap
    _feed(cam, 30, 5)
    actions = [r["action"] for r in cam.recoveries]
    assert actions[:2] == ["resend_uvc", "reopen"]
    assert cam.resent == 1 and first.released and FakeCapture.opened == 2
    assert cam._need_uvc               # controls go out again after the reopened stream's first frame


def test_recovery_resets_the_steps(monkeypatch):
    cam = _camera(monkeypatch)
    _feed(cam, 30, 2.1)                # resend_uvc at 2 s
    _feed(cam, 120, 3, start_s=2.1)    # back to full rate
    _feed(cam, 30, 1.5, start_s=5.1)   # drops again: resend first, not reopen
    assert [r["action"] for r in cam.recoveries] == ["resend_uvc", "resend_uvc"]


def test_without_uvc_reopens_straight_away(monkeypatch):
    cam = _camera(monkeypatch, uvc=False)
    _feed(cam, 30, 2.5)
    assert [r["action"] for r in cam.recoveries] == ["reopen"]


def test_off_for_video_files(monkeypatch):
    monkeypatch.setattr(camera_mod, "open_capture", FakeCapture)
    assert Camera("eye.avi", min_fps=100).min_fps is None


def test_reopens_back_off_while_the_camera_stays_down(monkeypatch):
    cam = _camera(monkeypatch, uvc=False)
    for i in range(int(60 / 0.01)):        # no frames at all for 60 s (camera gone from USB)
        cam._watchdog(int(i * 0.01 * 1e9), False)
    times = [r["t_ns"] / 1e9 for r in cam.recoveries]
    gaps = [b - a for a, b in zip(times, times[1:])]
    assert gaps == sorted(gaps) and gaps[-1] >= 16   # 2, 4, 8, 16, then capped at 30 s
    assert len(cam.recoveries) <= 6
