"""Scene-camera calibration: JEO's tangent-plane mapping, two-eye combining, the board and the full pipeline."""

import json

import cv2

import numpy as np
import pandas as pd
import pytest

from eyetrack.calibration.board import CAL_IDS, VAL_IDS, layout_cm, target_labels, write_board_pdf
from eyetrack.calibration.scene import (Calibration, align_samples, build_camera_calibration,
                                        camera_angular_error_degrees, centre_and_top, combine_eyes,
                                        project_gaze_to_tangent, project_many)
from eyetrack.config import load_config

from synthetic import write_board_session


def _dirs(angles_deg):
    a = np.radians(np.asarray(angles_deg, float))
    return np.stack([np.tan(a[:, 0]), np.tan(a[:, 1]), np.ones(len(a))], axis=1)


def test_jeo_five_point_fit_recovers_a_linear_map():
    # JEO's five targets: centre, top, bottom, left, right on a 640x480 image, 15% margin
    targets = np.array([(320, 240), (320, 72), (320, 407), (96, 240), (543, 240)], float)
    # directions whose tangent coordinates map linearly onto the targets (z forward, y down)
    dirs = np.array([[(x - 320) / 600, (y - 240) / 600, 1.0] for x, y in targets])
    basis, matrix, rms = build_camera_calibration(dirs, targets, centre_index=0, top_index=1)
    assert rms < 1e-6
    pred = targets[0] + np.array([project_gaze_to_tangent(d, basis) for d in dirs]) @ matrix.T
    assert np.allclose(pred, targets, atol=1e-6)


def test_project_many_matches_the_single_sample_version_and_rejects_backward_gaze():
    basis = np.eye(3)[[2, 0, 1]]      # forward z, right x, up y
    d = np.array([[0.1, -0.2, 1.0], [0.0, 0.0, -1.0], [np.nan, 0, 1]])
    out = project_many(d, basis)
    assert np.allclose(out[0], project_gaze_to_tangent(d[0], basis))
    assert np.isnan(out[1]).all() and np.isnan(out[2]).all()


def test_centre_and_top_pick_the_middle_and_the_target_above_it():
    t = np.array([(100, 100), (300, 100), (500, 100), (100, 300), (300, 300), (500, 300), (300, 500)], float)
    assert centre_and_top(t) == (4, 1)


def test_angular_error_uses_the_pinhole_model():
    # 60 deg diagonal on 640x480: focal = 400 / tan(30 deg) = 692.8 px
    err = camera_angular_error_degrees((320, 240), (320 + 692.8 * np.tan(np.radians(5)), 240), (640, 480), 60)
    assert abs(err - 5) < 0.01


def test_combine_eyes_averages_and_keeps_a_lone_eye():
    a = np.array([[10, 10], [np.nan, np.nan], [0, 0]], float)
    b = np.array([[20, 30], [5, 6], [np.nan, np.nan]], float)
    assert np.allclose(combine_eyes(a, b), [[15, 20], [5, 6], [0, 0]])


def test_align_samples_takes_the_nearest_within_tolerance():
    out = align_samples(np.array([0.0, 0.1, 0.2]), np.array([[1.0], [2.0], [3.0]]), np.array([0.04, 0.16, 0.5]), 0.05)
    assert out[:2, 0].tolist() == [1.0, 3.0] and np.isnan(out[2, 0])


def test_old_single_eye_calibration_files_still_load(tmp_path):
    from eyetrack.calibration.mapping import PolyMapper
    rng = np.random.default_rng(1)
    p = rng.uniform(100, 500, (9, 2))
    PolyMapper.fit(p, p * 2 + 5).save(tmp_path / "calibration.json")
    cal = Calibration.load(tmp_path / "calibration.json")
    gx, gy = cal.predict("eye", [200.0], [300.0])
    assert abs(gx[0] - 405) < 1e-6 and abs(gy[0] - 605) < 1e-6
    assert np.isnan(cal.predict("eye2", [1.0], [1.0])[0][0])


def test_board_layout_and_pdf(tmp_path):
    pos = layout_cm(100, 20)
    assert pos[4] == (0.0, 0.0)
    assert abs(pos[5][0] - 36.4) < 0.1 and abs(pos[1][1] + 36.4) < 0.1
    labels, mfl = target_labels()
    assert labels[0] == "cal_4" and labels[1] == "cal_1"             # JEO's order: centre, then top
    assert sorted(mfl[x] for x in labels if x.startswith("cal")) == CAL_IDS
    assert [mfl[x] for x in labels if x.startswith("val")] == VAL_IDS
    pdf = write_board_pdf(tmp_path / "board.pdf", 100, 20, 8)
    assert pdf.read_bytes().count(b"/Type /Page\n") + pdf.read_bytes().count(b"/Type /Page ") >= 14 or \
        pdf.stat().st_size > 10_000


@pytest.fixture(scope="module")
def board_session(tmp_path_factory):
    session = tmp_path_factory.mktemp("board") / "20261107_120000_calibration"
    truth = write_board_session(session)
    return session, truth


def test_two_eye_board_calibration_with_asymmetric_cameras(board_session):
    from eyetrack.pipeline import calibrate_from_session
    session, _ = board_session
    cfg = load_config()
    res = calibrate_from_session(session, cfg)
    assert res["eyes"]["eye"]["calibration_targets"] == 9 and res["eyes"]["eye"]["accuracy_targets"] == 4
    assert res["eyes"]["eye2"]["calibration_targets"] == 9
    err = pd.read_csv(session / "calibration_errors.csv")
    acc = err[err.kind == "accuracy"].groupby(["role", "method"]).error_deg.mean()
    # the two eye cameras sit differently (20 vs 30 deg pitch, 8 deg yaw, other centre and radius); each eye
    # gets its own mapping, so both reach the held-out targets
    for role in ("eye", "eye2", "both"):
        for method in ("jeo", "poly"):
            assert acc[(role, method)] < 1.0, (role, method, acc)
    cal = Calibration.load(session / "calibration.json")
    assert set(cal.eyes) == {"eye", "eye2"} and cal.method == "jeo"


def test_check_with_scores_an_existing_calibration(board_session):
    from eyetrack.pipeline import calibrate_from_session
    session, _ = board_session
    res = calibrate_from_session(session, load_config(), check_with=session / "calibration.json")
    assert (session / "check_errors.csv").exists()
    assert {r["kind"] for r in res["mean_errors"]} == {"accuracy"}


def test_process_draws_combined_gaze_on_the_scene(board_session):
    from eyetrack.pipeline import calibrate_from_session, process
    session, truth = board_session
    cfg = load_config()
    if not (session / "calibration.json").exists():
        calibrate_from_session(session, cfg)
    summary = process(session, cfg, calibration=session / "calibration.json")
    assert (session / "eye2" / "pupil.csv").exists()
    assert (session / "overlay_snapshot.png").exists() and summary["overlay_snapshot"]
    gaze = pd.read_csv(session / "gaze.csv")
    for c in ["gaze_x", "gaze_y", "gaze_eye_x", "gaze_eye2_x"]:
        assert c in gaze
    # mid-dwell samples (eyes settled on a target): combined gaze within ~1 deg (12 px at f=700) of the truth
    t = gaze.t_s.to_numpy()
    phase = (t - 1.0) % 1.2
    sel = (t > 1.0) & (phase > 0.5) & (phase < 1.0) & gaze.gaze_x.notna().to_numpy()
    err = np.hypot(*(gaze.loc[sel, ["gaze_x", "gaze_y"]].to_numpy() - truth(t[sel])).T)
    assert sel.sum() > 200
    assert np.median(err) < 12, np.median(err)
    meta = json.loads((session / "metrics_summary.json").read_text())
    assert meta["eye"] == "eye"


def test_screen_target_dot_is_bigger_and_marker_still_found():
    from eyetrack.calibration.targets import SCREEN_DOT_RADIUS, TargetSequence, CALIBRATION_POINTS
    seq = TargetSequence(CALIBRATION_POINTS, screen=(1470, 956))
    img, label = seq.frame(2.5)
    red = (img[:, :, 2] == 255) & (img[:, :, 0] == 0)
    assert label == "cal_0" and abs(red.sum() - np.pi * SCREEN_DOT_RADIUS**2) < 0.15 * np.pi * SCREEN_DOT_RADIUS**2
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
    assert det.detectMarkers(img)[1] is not None


def test_preview_draws_gaze_ray_from_calibration_eye_models(board_session):
    from eyetrack.capture.multiview import draw_eye, eye_models_from_calibration
    from eyetrack.pipeline import calibrate_from_session
    from eyetrack.pupil.detector import DetectorParams, detect_pupil
    from synthetic import gaze_to_eye_image
    session, _ = board_session
    cfg = load_config()
    if not (session / "calibration.json").exists():
        calibrate_from_session(session, cfg)
    models = eye_models_from_calibration(session / "calibration.json")
    assert set(models) == {"eye", "eye2"}
    img = gaze_to_eye_image(10, 0, {"center": (320, 200), "radius": 110, "pitch": 20, "yaw": 0})
    r = detect_pupil(img, DetectorParams.from_config(cfg["pupil"]))
    plain = draw_eye(img, r)
    with_ray = draw_eye(img, r, models["eye"])
    assert (with_ray != plain).any()                               # eyeball and ray drawn
    assert (draw_eye(img, r, models["eye"], min_confidence=1.1) != plain).any()   # outline even when untrusted
