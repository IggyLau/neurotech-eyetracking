"""Guided validation protocol: timeline, alignment checks and the report, without a camera or screen."""

import numpy as np
import pytest
import pandas as pd

from eyetrack.config import load_config
from eyetrack.pipeline import process
from eyetrack.protocol import (EPOCH_LABELS, LEVELS, SPAN_LEVELS, ProtocolDisplay, analyze, arithmetic_trials,
                               assess_alignment, build_steps, clean_pupil_trace, report_lines, span_responses,
                               span_trials)
from eyetrack.pupil.detector import PupilResult

from test_pipeline import _write_session


def test_timeline_has_every_part_and_ends():
    d = ProtocolDisplay(build_steps((720, 450)))
    labels = [s.label for s in d.steps]
    assert labels.count("blink_cue") == 10
    assert labels.count("lights_on") == 3
    assert all(labels.count(lvl) == 4 for lvl in SPAN_LEVELS)     # digit span is the default effort task
    assert not set(LEVELS) & set(labels)                         # arithmetic is opt-in
    assert "reading" in labels and labels[2] == "baseline"
    assert 5 < d.total_s / 60 < 7
    img, label = d.frame(0.1)
    assert img.shape == (450, 720, 3) and label == "intro"
    assert d.frame(d.total_s + 1) is None
    # light reflex screens really are black and white
    on = next(i for i, s in enumerate(d.steps) if s.label == "lights_on")
    off = next(i for i, s in enumerate(d.steps) if s.label == "lights_off")
    assert d.frame(d.starts[on] + 0.5)[0].mean() > 240
    assert d.frame(d.starts[off] + 0.5)[0].mean() < 15


def test_alignment_checks_position_size_and_rate():
    good = PupilResult(found=True, cx=320, cy=200, axis_minor=50, axis_major=60, confidence=0.9)
    st = assess_alignment(good, (400, 640), fps=120, recent_valid=[True] * 30)
    assert st.ready
    off_centre = PupilResult(found=True, cx=600, cy=200, axis_minor=50, axis_major=60, confidence=0.9)
    st = assess_alignment(off_centre, (400, 640), 120, [True] * 30)
    assert not st.ready and "right" in st.hint
    tiny = PupilResult(found=True, cx=320, cy=200, axis_minor=10, axis_major=12, confidence=0.9)
    assert "closer" in assess_alignment(tiny, (400, 640), 120, [True] * 30).hint
    assert not assess_alignment(good, (400, 640), 60, [True] * 30).ready


def test_report_from_a_processed_session(tmp_path):
    session = _write_session(tmp_path)     # synthetic eye with blinks at 2.0 s and 4.0 s
    t0 = 10**12
    markers = [(0.1, "baseline"), (1.8, "blink_cue"), (2.4, "blink_wait"), (3.8, "blink_cue"),
               (4.4, "lights_on"), (5.0, "lights_off")]
    pd.DataFrame({"t_ns": [int(t0 + t * 1e9) for t, _ in markers], "label": [l for _, l in markers]}).to_csv(
        session / "markers.csv", index=False)
    cfg = load_config()
    cfg["pupil_size"]["epoch_labels"] = EPOCH_LABELS
    process(session, cfg)
    r = analyze(session)
    assert r["blinks"] == {"cued": 2, "detected_after_cue": 2, "hit_rate": 1.0}
    assert r["light_reflex"]["flashes"] == 1
    assert (session / "validation_report.json").exists()
    assert len(report_lines(r)) == 5      # no effort part in this session, so no effort lines


def test_arithmetic_trials_three_levels_interleaved():
    trials = arithmetic_trials(5, seed=1)
    assert len(trials) == 15 and len({t.problem for t in trials}) == 15
    for k in range(5):                      # every group of three has one of each level
        assert {t.level for t in trials[3 * k:3 * k + 3]} == set(LEVELS)
    for t in trials:
        small, big = sorted((t.a, t.b))
        if t.level == "task_easy":
            assert 2 <= small <= 4 and big < 10
        elif t.level == "task_medium":
            assert small < 10 <= big and big % 10
        else:
            assert 12 <= small < big < 20
    assert [t.problem for t in arithmetic_trials(5, seed=1)] == [t.problem for t in trials]


def test_medium_problems_get_harder():
    for seed in range(20):
        medium = [t.a * t.b for t in arithmetic_trials(5, seed=seed) if t.level == "task_medium"]
        assert medium == sorted(medium) and len(set(medium)) == 5


def test_only_arithmetic_has_just_that_part():
    d = ProtocolDisplay(build_steps((720, 450), parts=("arithmetic",), trials=arithmetic_trials(2, seed=0)))
    labels = {s.label for s in d.steps}
    assert {"baseline", "blink_cue", "lights_on", "reading"}.isdisjoint(labels)
    assert set(LEVELS) <= labels


def test_arithmetic_report_per_level(tmp_path):
    session = _write_session(tmp_path)
    t0 = 10**12
    markers = [(0.5, "task_easy"), (1.5, "answer"), (2.5, "task_medium"), (3.2, "answer"),
               (3.5, "task_hard"), (5.0, "answer")]
    pd.DataFrame({"t_ns": [int(t0 + t * 1e9) for t, _ in markers], "label": [l for _, l in markers]}).to_csv(
        session / "markers.csv", index=False)
    pd.DataFrame({"trial": [1, 2, 3], "level": list(LEVELS), "problem": ["3 × 4", "13 × 6", "12 × 17"],
                  "answer": [12, 78, 204]}).to_csv(session / "trials.csv", index=False)
    cfg = load_config()
    cfg["pupil_size"]["epoch_labels"] = EPOCH_LABELS
    process(session, cfg)
    r = analyze(session)
    assert [r["arithmetic"][lvl]["trials"] for lvl in LEVELS] == [1, 1, 1]
    per_trial = pd.read_csv(session / "arithmetic_trials.csv")
    assert list(per_trial.problem) == ["3 × 4", "13 × 6", "12 × 17"]
    assert "medium" in report_lines(r)[3]
    assert report_lines(r)[3].startswith("Mental arithmetic")


def test_score_arithmetic_solve_time_answers_and_early_talking(tmp_path, monkeypatch):
    import json
    from eyetrack import answers
    from eyetrack.answers import SAMPLE_RATE, save_wav
    from eyetrack.protocol import score_arithmetic

    sr, t0 = SAMPLE_RATE, 5 * 10**9
    rng = np.random.default_rng(0)
    audio = rng.normal(0, 0.0005, sr * 32).astype(np.float32)

    def speak(at_s, dur=0.4):
        t = np.arange(int(dur * sr)) / sr
        audio[int(at_s * sr):int(at_s * sr) + len(t)] += 0.05 * np.sin(2 * np.pi * 200 * t)

    # trial 1: problem at 1 s, SPACE at 4 s, answer prompt at 11 s, speaks at 11.4 s
    # trial 2: problem at 16 s, no SPACE, talks at 19 s (during solving), answer prompt at 26 s, speaks at 26.5 s
    speak(11.4); speak(19.0); speak(26.5)
    marks = [(1, "task_easy"), (4, "solved"), (11, "answer"), (15, "arith_fix"),
             (16, "task_hard"), (26, "answer"), (30, "done")]
    pd.DataFrame({"t_ns": [t0 + int(s * 1e9) for s, _ in marks], "label": [l for _, l in marks]}).to_csv(
        tmp_path / "markers.csv", index=False)
    pd.DataFrame({"trial": [1, 2], "level": ["task_easy", "task_hard"], "problem": ["6 × 7", "17 × 13"],
                  "answer": [42, 221]}).to_csv(tmp_path / "trials.csv", index=False)
    save_wav(tmp_path / "audio.wav", audio)
    (tmp_path / "audio_meta.json").write_text(json.dumps({"t0_ns": t0}))
    monkeypatch.setattr(answers, "transcribe_clips", lambda clips, sr: ["42.", "two hundred and twelve"])

    df = score_arithmetic(tmp_path)
    one, two = df.iloc[0], df.iloc[1]
    assert one.rt_s == pytest.approx(3.0) and np.isnan(two.rt_s)
    assert one.correct and one.heard == 42
    assert two.heard == 212 and not two.correct
    assert not one.spoke_during_solve and two.spoke_during_solve
    assert one.voice_onset_after_prompt_s == pytest.approx(0.4, abs=0.05)


def test_space_skips_to_the_answer_prompt():
    from eyetrack.protocol import Step
    blank = lambda t: np.zeros((10, 10, 3), np.uint8)   # noqa: E731
    d = ProtocolDisplay([Step("arith_fix", 4, blank), Step("task_hard", 10, blank, static=False, skip_on="solved"),
                         Step("answer", 4, blank), Step("done", 1, blank)])
    assert d.frame(5.0)[1] == "task_hard"
    d.on_key("solved", 4.1)                 # too soon after the problem appeared: ignored
    assert d.frame(5.0)[1] == "task_hard"
    d.on_key("solved", 6.5)                 # solved 2.5 s in: straight to the answer prompt
    assert d.frame(6.6)[1] == "answer"
    assert d.frame(6.5 + 4.1)[1] == "done"
    d.on_key("solved", 7.0)                 # SPACE during the answer step does nothing
    assert d.frame(7.1)[1] == "answer"


def test_span_trials_three_lengths_interleaved():
    trials = span_trials(4, seed=1)
    assert len(trials) == 12
    for k in range(0, 12, 3):
        assert sorted(len(t.digits) for t in trials[k:k + 3]) == [3, 5, 7]
    for t in trials:
        assert "0" not in t.digits and len(set(t.digits)) == len(t.digits)
    assert [t.digits for t in span_trials(4, seed=1)] == [t.digits for t in trials]


def test_arithmetic_is_opt_in_with_two_per_level():
    d = ProtocolDisplay(build_steps((720, 450), parts=("arithmetic",)))
    labels = [s.label for s in d.steps]
    assert all(labels.count(lvl) == 2 for lvl in LEVELS)


def test_digit_span_steps_without_a_voice_show_the_digits():
    spans = span_trials(1, seed=0)
    d = ProtocolDisplay(build_steps((720, 450), parts=("digit_span",), spans=spans, voice=None))
    labels = [s.label for s in d.steps]
    assert labels.count("recall") == 3 and all(lvl in labels for lvl in SPAN_LEVELS)
    i = labels.index("span_3")
    blank = d.frame(d.starts[i] + 0.8)[0]          # between digits: just the cross
    digit = d.frame(d.starts[i] + 0.2)[0]          # a digit is drawn above the cross
    assert (digit != blank).any()
    assert d.steps[i].duration == 3 + 2            # 3 digits at 1/s + 2 s hold


def _pupil(t, d, valid=None, cx=320.0):
    n = len(t)
    return pd.DataFrame({"t_s": t, "diameter_clean_px": d, "valid": np.ones(n, bool) if valid is None else valid,
                         "size_outlier": np.zeros(n, bool), "cx": np.full(n, cx), "cy": np.full(n, 200.0)})


def test_clean_trace_smooths_jitter_and_fills_only_short_gaps():
    t = np.arange(0, 6, 1 / 120)
    rng = np.random.default_rng(0)
    d = 50 + rng.normal(0, 2, len(t))                       # frame-to-frame jitter
    valid = np.ones(len(t), bool)
    valid[(t > 1.0) & (t < 1.1)] = False                    # 100 ms gap: filled
    valid[(t > 3.0) & (t < 3.5)] = False                    # 500 ms gap: left empty
    clean, usable = clean_pupil_trace(_pupil(t, d, valid))
    assert np.isfinite(clean[(t > 1.0) & (t < 1.1)]).all()
    assert np.isnan(clean[(t > 3.1) & (t < 3.4)]).all()
    seg = clean[(t > 0.2) & (t < 0.9)]
    assert np.nanstd(seg) < 0.5                             # jitter of sd 2 px mostly removed
    assert not usable[(t > 1.0) & (t < 1.1)].any()


def test_clean_trace_drops_off_track_frames():
    t = np.arange(0, 3, 1 / 120)
    d = np.full(len(t), 50.0)
    cx = np.full(len(t), 320.0)
    bad = (t > 1.5) & (t < 1.55)
    cx[bad] = 200.0                                          # a lash 120 px away, still "valid"
    d[bad] = 90.0
    p = _pupil(t, d)
    p["cx"] = cx
    clean, usable = clean_pupil_trace(p)
    assert not usable[bad].any()
    assert np.nanmax(clean) < 51


def test_span_response_is_subtractive_from_a_one_second_median_baseline():
    t = np.arange(0, 30, 1 / 120)
    d = np.full(len(t), 50.0)
    d[(t >= 12) & (t < 15)] = 53.0                           # span_3 at 10 s: window 12-15 s, +3 px
    d[(t >= 26) & (t < 29)] = 50.5                           # span_3 at 24 s: +0.5 px
    valid = np.ones(len(t), bool)
    valid[(t >= 25) & (t < 25.5)] = False                    # second trial's window 26-29 s stays usable
    markers = pd.DataFrame({"t_s": [10.0, 24.0], "label": ["span_3", "span_3"]})
    df = span_responses(_pupil(t, d, valid), markers, {"lowpass_hz": 0})
    assert df.change_px.round(2).tolist() == [3.0, 0.5]
    assert df.change_pct.round(1).tolist() == [6.0, 1.0]     # against the median baseline (50 px)
    assert not df.rejected.any()


def test_span_trial_with_an_unusable_window_is_dropped():
    t = np.arange(0, 20, 1 / 120)
    d = np.full(len(t), 50.0)
    valid = np.ones(len(t), bool)
    valid[(t >= 12) & (t < 14)] = False                      # most of the 12-15 s window lost
    df = span_responses(_pupil(t, d, valid), pd.DataFrame({"t_s": [10.0], "label": ["span_3"]}))
    assert df.rejected.iloc[0] and df.reason.iloc[0] == "window mostly unusable"


def test_parse_digits():
    from eyetrack.answers import parse_digits
    assert parse_digits("4 7 2") == "472"
    assert parse_digits("472.") == "472"
    assert parse_digits("Four, seven, two, nine.") == "4729"
    assert parse_digits("um") is None


def test_score_span_recall_and_talking_during_hold(tmp_path, monkeypatch):
    import json
    from eyetrack import answers
    from eyetrack.answers import SAMPLE_RATE, save_wav
    from eyetrack.protocol import score_span

    sr, t0 = SAMPLE_RATE, 5 * 10**9
    rng = np.random.default_rng(0)
    audio = rng.normal(0, 0.0005, sr * 30).astype(np.float32)

    def speak(at_s, dur=0.4):
        t = np.arange(int(dur * sr)) / sr
        audio[int(at_s * sr):int(at_s * sr) + len(t)] += 0.05 * np.sin(2 * np.pi * 200 * t)

    # trial 1: 3 digits from 1 s, hold 4-6 s, recall at 6 s, says them at 6.5 s
    # trial 2: 5 digits from 14 s, talks in the hold at 19.5 s, recall at 21 s, says them at 21.5 s
    speak(6.5, 1.5); speak(19.5); speak(21.5, 2.0)
    marks = [(1, "span_3"), (6, "recall"), (10, "span_fix"), (14, "span_5"), (21, "recall"), (27, "done")]
    pd.DataFrame({"t_ns": [t0 + int(s * 1e9) for s, _ in marks], "label": [l for _, l in marks]}).to_csv(
        tmp_path / "markers.csv", index=False)
    pd.DataFrame({"trial": [1, 2], "level": ["span_3", "span_5"], "digits": ["472", "61839"]}).to_csv(
        tmp_path / "span_trials.csv", index=False)
    save_wav(tmp_path / "audio.wav", audio)
    (tmp_path / "audio_meta.json").write_text(json.dumps({"t0_ns": t0}))
    monkeypatch.setattr(answers, "transcribe_clips", lambda clips, sr, prompt=None: ["4, 7, 2.", "6 1 8 9 3"])

    df = score_span(tmp_path)
    one, two = df.iloc[0], df.iloc[1]
    assert one.correct and one.heard == "472" and one.positions_correct == 3
    assert not two.correct and two.positions_correct == 3       # 6 1 8 right, 3 and 9 swapped
    assert not one.spoke_during_hold and two.spoke_during_hold


def test_targets_part_shows_each_dot_once_where_targets_csv_says():
    from eyetrack.calibration.eye_radius import TARGET_POINTS
    d = ProtocolDisplay(build_steps((720, 450), parts=("targets",)))
    shown = [s.label for s in d.steps if s.label.startswith("target_")]
    assert sorted(shown) == [f"target_{k}" for k in range(9)]
    k = int(shown[0][7:])
    i = next(i for i, s in enumerate(d.steps) if s.label == shown[0])
    img = d.frame(d.starts[i] + 1.0)[0]
    x, y = int(TARGET_POINTS[k][0] * 720), int(TARGET_POINTS[k][1] * 450)
    assert img[y, x + 8].max() < 40 and img[10, 10].max() > 100      # dark dot on grey
