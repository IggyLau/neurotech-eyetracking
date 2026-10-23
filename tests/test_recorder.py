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
