"""The UI's command building (the window itself is checked by hand)."""

import pytest

from eyetrack.config import load_config
from eyetrack.ui import build_config, command, latest_calibration, sessions


def test_build_config_sets_camera_indices():
    cfg = build_config(load_config(), {"eye": 3, "eye2": 4, "scene": 0})
    assert (cfg["eye_camera"]["index"], cfg["eye2_camera"]["index"], cfg["scene_camera"]["index"]) == (3, 4, 0)


def test_commands_follow_the_mode(tmp_path):
    c = command("record", "3", "x.yaml", name="t1", duration="10")
    assert c[-6:] == ["record", "--mode", "3", "--name", "t1", "--duration", "10"][-6:]
    c = command("calibrate", "3", "x.yaml", board=True, distance_cm="100")
    assert c[c.index("calibrate"):] == ["calibrate", "--mode", "3", "--board", "--distance-cm", "100"]
    assert "2" == command("calibrate", "2", "x.yaml")[-1]
    with pytest.raises(ValueError):
        command("calibrate", "2eyes", "x.yaml")       # no scene camera
    with pytest.raises(ValueError):
        command("check", "2", "x.yaml")               # no calibration chosen
    c = command("process", "1", "x.yaml", session="s", calibration="c.json", overlay=True)
    assert c[c.index("process"):] == ["process", "s", "--calibration", "c.json", "--overlay"]


def test_sessions_newest_first_and_latest_calibration(tmp_path):
    for name in ["20261101_100000", "20261102_100000_calibration", "notes"]:
        (tmp_path / name).mkdir()
    (tmp_path / "20261102_100000_calibration" / "calibration.json").write_text("{}")
    assert [p.name for p in sessions(tmp_path)] == ["20261102_100000_calibration", "20261101_100000"]
    assert latest_calibration(tmp_path).parent.name == "20261102_100000_calibration"
