import numpy as np
import pandas as pd
import pytest

from eyetrack.calibration.mapping import PolyMapper
from eyetrack.metrics.aoi import aoi_metrics
from eyetrack.metrics.blinks import detect_blinks
from eyetrack.metrics.events import detect_events
from eyetrack.metrics.pupil_size import baseline_correct, clean_diameter, marker_epochs

from synthetic import step_trace


def test_ivt_finds_every_saccade_with_correct_amplitude():
    t, x, y, onsets = step_trace(fs=120, n_steps=10, step_deg=10)
    sacc, fix, labels = detect_events(t, x, y, threshold=30)
    assert len(sacc) == len(onsets)
    assert np.allclose(sacc.onset_s.to_numpy(), onsets, atol=0.03)
    assert sacc.amplitude.mean() == pytest.approx(10, abs=1.0)
    assert len(fix) == len(onsets) + 1
    assert set(labels) <= {"fix", "sac", "none"}


def test_amplitude_noise_filter_rejects_noise_saccades():
    t, x, y, onsets = step_trace(fs=120, n_steps=10, step_deg=10, noise=0.5)
    raw, _, _ = detect_events(t, x, y, threshold=30)
    filtered, _, _ = detect_events(t, x, y, threshold=30, amplitude_noise_factor=3)
    assert len(raw) > 2 * len(onsets)            # noise at 120 Hz exceeds 30 deg/s often
    assert len(filtered) < len(raw) / 2          # most noise bursts removed...
    assert len(onsets) <= len(filtered) <= len(onsets) + 5  # ...all real ones kept (a few big bursts survive)
    assert filtered.attrs["noise_rms_s2s"] > 0.5


def test_adaptive_threshold_rises_with_noise():
    t, x, y, _ = step_trace(fs=120, noise=0.05)
    clean, _, _ = detect_events(t, x, y, threshold="adaptive", min_threshold=0)
    t, x, y, _ = step_trace(fs=120, noise=0.5)
    noisy, _, _ = detect_events(t, x, y, threshold="adaptive", min_threshold=0)
    assert noisy.attrs["velocity_threshold"] > 3 * clean.attrs["velocity_threshold"]


def test_blink_merge_and_low_confidence_dropouts():
    fs = 120
    t = np.arange(fs * 4) / fs
    valid = np.ones_like(t, bool)
    valid[(t >= 1.0) & (t < 1.08)] = False   # blink interrupted by a 30 ms flicker
    valid[(t >= 1.11) & (t < 1.2)] = False
    valid[(t >= 3.0) & (t < 3.2)] = False    # glare: detector still half-sure
    conf = np.where(valid, 0.9, 0.1)
    conf[(t >= 3.0) & (t < 3.2)] = 0.5
    b = detect_blinks(t, valid, 30, 500, merge_gap_ms=80, confidence=conf, max_blink_confidence=0.35)
    assert (b.kind == "blink").sum() == 1
    assert (b.kind == "low_confidence").sum() == 1


def test_blinks_vs_dropouts_vs_closed():
    fs = 120
    t = np.arange(fs * 10) / fs
    valid = np.ones_like(t, bool)
    for start in (1.0, 3.0, 5.0):          # three 150 ms blinks
        valid[(t >= start) & (t < start + 0.15)] = False
    valid[(t >= 7.0) & (t < 7.02)] = False  # 20 ms dropout
    valid[(t >= 8.0) & (t < 9.0)] = False   # 1 s closed
    b = detect_blinks(t, valid)
    assert (b.kind == "blink").sum() == 3
    assert (b.kind == "dropout").sum() == 1
    assert (b.kind == "closed_or_lost").sum() == 1


def test_pupil_cleaning_interpolates_blinks_and_baseline_shows_dilation():
    fs = 120
    t = np.arange(fs * 10) / fs
    d = 40 + 6 * (t > 5)  # dilation after an event at 5 s
    bad = (t > 2) & (t < 2.15)
    clean = clean_diameter(t, d.astype(float), bad, smoothing_ms=50)
    assert np.isfinite(clean[(t > 1) & (t < 3)]).all()
    ep = baseline_correct(t, clean, [5.0], baseline_ms=500, window_s=3)
    late = ep[ep.t_rel_s > 1].change.mean()
    assert late == pytest.approx(6, abs=0.5)


def test_poly_mapper_recovers_quadratic_mapping():
    rng = np.random.default_rng(2)
    p = rng.uniform(100, 500, size=(9, 2))
    sx = 0.002 * p[:, 0] ** 2 + 1.1 * p[:, 0] + 0.3 * p[:, 1] + 5
    sy = 0.9 * p[:, 1] + 0.001 * p[:, 0] * p[:, 1] - 20
    m = PolyMapper.fit(p, np.stack([sx, sy], 1))
    gx, gy = m.predict(p[:, 0], p[:, 1])
    assert np.allclose(gx, sx, atol=1e-6) and np.allclose(gy, sy, atol=1e-6)


def test_aoi_dwell_and_revisits():
    fix = pd.DataFrame({"onset_s": [0, 1, 2, 3], "offset_s": [0.5, 1.5, 2.5, 3.5],
                        "duration_ms": [500, 500, 500, 500], "x": [10, 300, 20, 30], "y": [10, 10, 10, 10]})
    out = aoi_metrics(fix, {"left": [0, 0, 100, 100], "right": [200, 0, 200, 100]}).set_index("aoi")
    assert out.loc["left", "dwell_ms"] == 1500
    assert out.loc["left", "entries"] == 2 and out.loc["left", "revisits"] == 1
    assert out.loc["right", "fixations"] == 1


def test_marker_epochs_end_at_next_marker_and_measure_change():
    fs = 120
    t = np.arange(fs * 12) / fs
    d = np.full_like(t, 40.0)
    d[(t >= 3) & (t < 6)] = 34.0     # constriction while a light is on
    d[(t >= 6) & (t < 9)] = 44.0     # dilation during a task
    markers = pd.DataFrame({"t_s": [3.0, 6.0, 9.0, 10.0], "label": ["lights_on", "task_start", "task_end", "blink_cue"]})
    samples, summ = marker_epochs(t, d, markers, ["lights_on", "task_start", "task_end"], baseline_ms=500)
    assert list(summ.label) == ["lights_on", "task_start", "task_end"]   # blink_cue is not an epoch label
    assert summ.window_s.tolist() == pytest.approx([3.0, 3.0, 1.0])     # each epoch stops at the next marker
    on, task = summ.iloc[0], summ.iloc[1]
    assert on.baseline_px == pytest.approx(40)
    assert on.min_change_pct == pytest.approx(-15, abs=0.1)
    assert task.baseline_px == pytest.approx(34)
    assert task.max_change_pct == pytest.approx(100 * 10 / 34, abs=0.1)
    assert set(samples.event) == {0, 1, 2}


def test_step_detector_finds_steps_across_dropouts():
    from eyetrack.metrics.events import detect_steps
    t, x, y, onsets = step_trace(fs=120, n_steps=10, step_deg=5)
    rng = np.random.default_rng(3)
    x = x + rng.normal(0, 0.2, len(x))
    valid = np.ones(len(t), bool)
    valid[(np.arange(len(t)) % 10) >= 6] = False       # 4 of every 10 frames missing, like screen flicker
    sacc, fix, _ = detect_steps(t, x, y, valid)
    assert len(sacc) == len(onsets)
    assert np.allclose(np.sort(sacc.onset_s.to_numpy()), onsets, atol=0.05)
    assert sacc.amplitude.median() == pytest.approx(5, abs=0.5)
    assert len(fix) >= len(onsets)


def test_dropouts_classified_by_brightness():
    from eyetrack.metrics.blinks import blink_mask, classify_dropouts
    fs = 120
    t = np.arange(fs * 6) / fs
    valid = np.ones(len(t), bool)
    frame_b = np.full(len(t), 150.0)
    eye_b = np.full(len(t), 120.0)
    blink = (t >= 1.0) & (t < 1.2)                          # eyelid: eye area brighter
    valid[blink] = False
    eye_b[blink] = 135
    for k in range(30):                                    # flicker: 4 dark frames every 10, for 2.5 s
        a = int((3.0 + k * 10 / fs) * fs)
        valid[a:a + 4] = False
        frame_b[a:a + 4] = 100
    ev = classify_dropouts(t, valid, frame_b, eye_b, merge_gap_ms=80)
    assert (ev.kind == "blink").sum() == 1
    assert (ev.kind == "lighting").sum() == 30              # never merged into one long "closed" event
    assert "closed" not in set(ev.kind)
    masked = blink_mask(t, ev)
    assert masked[blink].all() and not masked[(t > 3) & valid].any()


def test_gap_fill_recovers_saccades_next_to_lost_frames():
    t, x, y, onsets = step_trace(fs=120, n_steps=10, step_deg=10)
    valid = np.ones(len(t), bool)
    for t0 in onsets:  # lose 2 frames just before each saccade, as screen flicker does
        i = int(np.searchsorted(t, t0))
        valid[i - 2:i] = False
    kw = dict(threshold=30, amplitude_noise_factor=3, min_fixation_ms=60)
    plain, _, _ = detect_events(t, x, y, valid, **kw)
    filled, _, _ = detect_events(t, x, y, valid, fill_gap_ms=35, **kw)
    assert len(plain) < len(onsets)          # the amplitude check needs a valid sample before the saccade
    assert len(filled) == len(onsets)
    assert filled.amplitude.to_numpy() == pytest.approx(10, abs=0.5)
    # filled samples don't count as measurements: same noise estimate as without filling
    assert filled.attrs["noise_rms_s2s"] == pytest.approx(plain.attrs["noise_rms_s2s"], rel=0.05)


def test_long_gaps_are_not_filled():
    t, x, y, _ = step_trace(fs=120, n_steps=4)
    valid = np.ones(len(t), bool)
    valid[200:230] = False                    # 250 ms: a blink, not a lost frame
    _, _, labels = detect_events(t, x, y, valid, threshold=30, fill_gap_ms=35)
    assert (labels[200:230] == "none").all()


def test_duration_bias_correction_subtracts_one_sample():
    t, x, y, _ = step_trace(fs=120, n_steps=6, sacc_ms=40)
    a, _, _ = detect_events(t, x, y, threshold=30)
    b, _, _ = detect_events(t, x, y, threshold=30, duration_bias_samples=1)
    assert len(a) == len(b)
    assert (a.duration_ms - b.duration_ms).to_numpy() == pytest.approx(1000 / 120, abs=0.01)
