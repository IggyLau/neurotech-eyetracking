"""End-to-end: synthetic eye video with known saccades and blinks -> process()."""

import json

import cv2
import numpy as np
import pandas as pd

from eyetrack.config import load_config
from eyetrack.pipeline import process

from synthetic import sphere_ellipse


def _write_session(tmp_path, fs=120, seconds=6):
    session = tmp_path / "session"
    session.mkdir()
    n = int(fs * seconds)
    t = np.arange(n) / fs
    # alternate gaze between yaw -25 and +25 deg every 0.5 s, with small pitch wander; blinks at 2 s and 4 s
    yaw = np.where((t // 0.5) % 2 == 0, -25.0, 25.0)
    pitch = 8 * np.sin(2 * np.pi * t / 3)
    blink = ((t > 2.0) & (t < 2.15)) | ((t > 4.0) & (t < 4.15))
    writer = cv2.VideoWriter(str(session / "eye.avi"), cv2.VideoWriter_fourcc(*"MJPG"), fs, (640, 400))
    for i in range(n):
        img = np.full((400, 640), 150, np.uint8)
        if not blink[i]:
            (cx, cy), axes, ang = sphere_ellipse((320, 200), 110, yaw[i], pitch[i], pupil_radius=22)
            cv2.circle(img, (int(cx), int(cy)), 70, 95, -1)
            cv2.ellipse(img, ((cx, cy), axes, ang), 15, -1)
        else:
            img[195:205, 120:520] = 90
        writer.write(cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 1.0), cv2.COLOR_GRAY2BGR))
    writer.release()
    pd.DataFrame({"frame_index": np.arange(n), "t_ns": (t * 1e9).astype(np.int64) + 10**12}).to_csv(
        session / "eye_ts.csv", index=False)
    pd.DataFrame({"t_ns": [int(1e12 + 3e9)], "label": ["task_start"]}).to_csv(session / "markers.csv", index=False)
    return session


def test_process_session_end_to_end(tmp_path):
    session = _write_session(tmp_path)
    summary = process(session, load_config())
    for f in ["pupil.csv", "blinks.csv", "saccades.csv", "fixations.csv", "metrics_summary.json", "report.png"]:
        assert (session / f).exists(), f
    assert summary["detection_rate"] > 0.9
    assert summary["blink_count"] == 2
    assert summary["eye_model"] is not None
    assert summary["unit"] == "deg"
    # 11 gaze switches in 6 s, minus any that coincide with a blink
    assert 9 <= summary["saccade_count"] <= 12
    sacc = pd.read_csv(session / "saccades.csv")
    assert 40 < np.median(sacc.amplitude) < 60  # true horizontal amplitude 50 deg
    assert json.loads((session / "metrics_summary.json").read_text())["frames"] == 720
