"""Real-time path: causal metrics match their offline counterparts, the worker processes every frame in order."""

import json

import cv2
import numpy as np
import pandas as pd
import pytest

from eyetrack.capture.camera import Frame
from eyetrack.capture.recorder import record
from eyetrack.config import load_config
from eyetrack.metrics.blinks import classify_dropouts
from eyetrack.realtime.online import (OnlineBlinks, OnlineProcessor, OnlineSaccades, OnlineSizeCheck,
                                      load_eye_model)
from eyetrack.realtime.replay import ReplayCamera
from eyetrack.realtime.worker import RealtimeProcessor

from synthetic import eye_image, step_trace

FS = 120.0


def test_size_check_uses_only_past_frames():
    sc = OnlineSizeCheck(FS, max_ratio=1.5)
    assert not sc.update(40.0, True)                 # no history yet: never an outlier
    for _ in range(100):
        assert not sc.update(40.0 + np.random.default_rng(1).normal(0, 0.5), True)
    assert sc.update(70.0, True)                     # iris picked: 1.75x the past median
    assert not sc.update(41.0, True)
    assert not sc.update(70.0, False)                # not confidence-valid: not judged


def _stream():
    """valid / brightness arrays with a blink, a lighting dropout, a short dropout, a lid flicker to merge."""
    n = 1200
    valid = np.ones(n, bool)
    fb = np.full(n, 100.0) + np.random.default_rng(0).normal(0, 0.3, n)
    eb = np.full(n, 60.0) + np.random.default_rng(1).normal(0, 0.3, n)
    valid[200:220] = False
    eb[200:220] = 75                                  # blink: eyelid brightens the eye area (167 ms)
    valid[400:415] = False
    fb[400:415] = 80                                  # lighting: whole frame 20% darker
    valid[600:602] = False                            # 17 ms dropout
    valid[800:812] = False
    eb[800:812] = 75                                  # blink, then a 5-frame (42 ms) half-open flicker,
    valid[817:830] = False
    eb[817:830] = 75                                  # then more blink: merged into one event
    valid[1000:1150] = False
    eb[1000:1150] = 75                                # 1.25 s closed
    return valid, fb, eb


def test_online_blinks_match_offline_classifier():
    valid, fb, eb = _stream()
    t = np.arange(len(valid)) / FS
    off = classify_dropouts(t, valid, fb, eb, 30, 500, 80, 0.08, 3.0)
    ob = OnlineBlinks(FS, 30, 500, 80, 0.08, 3.0)
    states = [ob.update(i, t[i], valid[i], fb[i], eb[i]) for i in range(len(t))]
    ob.close()
    live = pd.DataFrame(ob.events)
    assert list(live.kind) == list(off.kind) == ["blink", "lighting", "dropout", "blink", "closed"]
    assert list(live.start_index) == list(off.start_index)
    assert list(live.end_index) == list(off.end_index)
    np.testing.assert_allclose(live.duration_ms, off.duration_ms)
    # provisional per-frame state: "dropout" until min_duration has passed, then the eyelid kind
    assert states[199] == "" and states[200] == "dropout" and states[210] == "blink"
    assert states[405] == "lighting" and states[1100] == "closed"
    # final kinds are known after the run ends (+ merge gap for eyelid events), never before
    assert (live.known_at_s >= live.offset_s - 1e-9).all()


def test_online_blinks_report_camera_gaps():
    ob = OnlineBlinks(FS)
    t = list(np.arange(100) / FS) + list(0.9 + np.arange(50) / FS)      # ~75 ms with no frames
    for i, ti in enumerate(t):
        ob.update(i, ti, True, 100.0, 60.0)
    assert [e["kind"] for e in ob.events] == ["camera_gap"]


def test_online_saccade_speed_is_offline_gradient_delayed_one_frame():
    t, x, y, onsets = step_trace(fs=FS, n_steps=10, step_deg=10, noise=0.05)
    sac = OnlineSaccades(threshold=30, amplitude_noise_factor=3)
    out = [sac.update(ti, xi, yi, True) for ti, xi, yi in zip(t, x, y)]
    speed = np.array([o[0] for o in out])
    flag = np.array([o[1] for o in out])
    ref = np.hypot(np.gradient(x, t), np.gradient(y, t))
    np.testing.assert_allclose(speed[2:], ref[1:-1], rtol=1e-9)
    # every target jump is flagged within 3 frames of its onset, and nothing else is
    starts = np.flatnonzero(np.diff(flag.astype(int)) == 1) + 1
    assert len(starts) == len(onsets)
    for s, on in zip(starts, onsets):
        assert 0 <= t[s] - on <= 3 / FS + 1e-9


def test_online_saccade_ignores_noise_bursts_and_dropouts():
    sac = OnlineSaccades(threshold=30, amplitude_noise_factor=3)
    rng = np.random.default_rng(3)
    out = [sac.update(i / FS, rng.normal(0, 0.15), rng.normal(0, 0.15), True) for i in range(600)]
    fast = sum(o[0] > 30 for o in out)
    flags = sum(o[1] for o in out)
    assert fast > 30 and flags <= 3                  # fast-but-tiny noise is mostly rejected
    speed, flag = sac.update(5.0, 0, 0, False)
    assert np.isnan(speed) and not flag


def _frames(n=90, start_x=200, step=2.0):
    return [eye_image(start_x + i * step, 200, axes=(36, 40)) for i in range(n)]


def test_online_processor_on_synthetic_frames():
    cfg = load_config()
    proc = OnlineProcessor(cfg)
    rows = [proc.process(img, int(i / FS * 1e9), i) for i, img in enumerate(_frames())]
    assert all(r["valid"] for r in rows)
    assert abs(rows[-1]["cx"] - (200 + 89 * 2)) < 2
    assert rows[0]["blink_state"] == ""
    assert np.isnan(rows[0]["yaw_deg"])              # no eye model: pixel coordinates only
    assert proc.saccades.threshold == cfg["events"]["velocity_threshold_px"]


def _submit_all(rp, imgs, block):
    t0 = 1_000_000_000
    for i, img in enumerate(imgs):
        rp.submit(Frame(img, t0 + int(i / FS * 1e9), i), block=block)


@pytest.mark.parametrize("backend", ["thread", "process"])
def test_worker_processes_every_frame_in_order(tmp_path, backend):
    cfg = load_config()
    rp = RealtimeProcessor(cfg, tmp_path, backend=backend).start()
    imgs = _frames(120)
    _submit_all(rp, imgs, block=True)
    s = rp.stop()
    assert s["dropped"] == 0 and s["processed"] == 120
    live = pd.read_csv(tmp_path / "live_pupil.csv")
    assert list(live.frame_index) == list(range(120))
    assert (live.latency_ms > 0).all() and (live.proc_ms > 0).all()
    assert json.loads((tmp_path / "live_summary.json").read_text())["backend"] == backend
    assert (tmp_path / "live_blinks.csv").exists()


def test_worker_counts_dropped_frames_when_full(tmp_path):
    cfg = load_config()
    cfg["realtime"]["queue_frames"] = 2
    rp = RealtimeProcessor(cfg, tmp_path, backend="thread").start()
    _submit_all(rp, _frames(60, step=0), block=False)          # far faster than detection can run
    s = rp.stop()
    assert s["dropped"] > 0
    assert s["processed"] + s["dropped"] == s["submitted"] == 60
    live = pd.read_csv(tmp_path / "live_pupil.csv")
    assert live.frame_index.is_monotonic_increasing            # in order, with gaps where frames were skipped


def test_record_replay_writes_live_results_next_to_raw_video(tmp_path):
    video = tmp_path / "src" / "eye.avi"
    video.parent.mkdir()
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), FS, (640, 400))
    for img in _frames(80):
        w.write(img)
    w.release()
    t_ns = 5_000_000_000 + (np.arange(80) / FS * 1e9).astype(np.int64)
    pd.DataFrame({"frame_index": range(80), "t_ns": t_ns}).to_csv(video.with_name("eye_ts.csv"), index=False)

    cfg = load_config()
    cfg["recording"]["output_dir"] = str(tmp_path / "data")
    cam = ReplayCamera(video, video.with_name("eye_ts.csv"), speed=0)
    session = record(cfg, name="replay", cameras={"eye": cam}, preview=False, realtime=True, block_when_full=True)
    live = pd.read_csv(session / "live_pupil.csv")
    ts = pd.read_csv(session / "eye_ts.csv")
    assert len(live) == len(ts) == 80
    np.testing.assert_array_equal(live.t_ns, t_ns)             # metrics use the original capture times
    meta = json.loads((session / "meta.json").read_text())
    assert meta["realtime"]["dropped"] == 0 and meta["realtime"]["processed"] == 80


def test_load_eye_model_from_summary(tmp_path):
    (tmp_path / "metrics_summary.json").write_text(json.dumps(
        {"eye_model": {"cx": 320, "cy": 200, "radius": 150, "n_used": 50, "residual_px": 1.0}}))
    em = load_eye_model(tmp_path)
    assert (em.cx, em.cy, em.radius) == (320, 200, 150)
    cfg = load_config()
    proc = OnlineProcessor(cfg, em)
    r = proc.process(eye_image(320, 200), 0, 0)
    assert abs(r["yaw_deg"]) < 2 and abs(r["pitch_deg"]) < 2
    assert proc.unit == "deg"
    assert load_eye_model(None) is None
