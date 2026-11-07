"""Recorder smoke test using video files in place of cameras (no hardware needed)."""

import json

import cv2
import numpy as np
import pandas as pd

from eyetrack.capture.recorder import record
from eyetrack.config import load_config

from synthetic import eye_image


def _video(path, n=60, size=(640, 400)):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, size)
    for i in range(n):
        w.write(eye_image(200 + i * 3, 200, size=size))
    w.release()
    return str(path)


def test_record_writes_videos_timestamps_and_meta(tmp_path):
    cfg = load_config()
    cfg["eye_camera"]["index"] = _video(tmp_path / "fake_eye.avi")
    cfg["scene_camera"]["index"] = _video(tmp_path / "fake_scene.avi", size=(640, 480))
    cfg["recording"]["output_dir"] = str(tmp_path / "data")
    session = record(cfg, name="smoke", duration=1.0, preview=False)

    for f in ["eye.avi", "scene.avi", "eye_ts.csv", "scene_ts.csv", "markers.csv", "meta.json"]:
        assert (session / f).exists(), f
    ts = pd.read_csv(session / "eye_ts.csv")
    assert len(ts) > 10
    assert np.all(np.diff(ts.t_ns) > 0)
    meta = json.loads((session / "meta.json").read_text())
    assert meta["frames"]["eye"] == len(ts)


def test_draw_inset_puts_the_eye_view_bottom_right():
    from eyetrack.capture.recorder import draw_inset
    screen = np.full((900, 1440), 128, np.uint8)
    eye = np.zeros((400, 640, 3), np.uint8)
    out = draw_inset(screen, eye, (0.25, 0.2, 0.75, 0.8), width=320, margin=20)
    assert out.shape == (900, 1440, 3)
    assert out[880 - 10, 1420 - 10].max() == 0          # inside the inset (black eye image)
    assert out[100, 100].tolist() == [128, 128, 128]    # rest of the screen unchanged


def test_three_camera_mode_records_both_eyes_and_scene(tmp_path):
    cfg = load_config()
    cfg["eye_camera"]["index"] = _video(tmp_path / "fake_eye.avi")
    cfg["eye2_camera"]["index"] = _video(tmp_path / "fake_eye2.avi")
    cfg["scene_camera"]["index"] = _video(tmp_path / "fake_scene.avi", size=(640, 480))
    cfg["recording"]["output_dir"] = str(tmp_path / "data")
    session = record(cfg, name="three", duration=1.0, preview=False, realtime=False, mode="3")
    for f in ["eye.avi", "eye2.avi", "scene.avi", "eye2_ts.csv"]:
        assert (session / f).exists(), f
    meta = json.loads((session / "meta.json").read_text())
    assert meta["mode"] == "3" and set(meta["frames"]) == {"eye", "eye2", "scene"}


def test_mode_roles():
    import pytest
    from eyetrack.capture.recorder import mode_roles
    assert mode_roles("1") == ("eye",)
    assert mode_roles("2eyes") == ("eye", "eye2")
    assert mode_roles(None, use_scene=False) == ("eye",)
    with pytest.raises(ValueError):
        mode_roles("4")
