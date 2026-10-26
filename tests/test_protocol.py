"""Guided validation protocol: timeline, alignment checks and the report, without a camera or screen."""

import numpy as np
import pytest
import pandas as pd

from eyetrack.config import load_config
from eyetrack.pipeline import process
from eyetrack.protocol import (EPOCH_LABELS, LEVELS, ProtocolDisplay, analyze, arithmetic_trials, assess_alignment,
                               build_steps, report_lines)
from eyetrack.pupil.detector import PupilResult

from test_pipeline import _write_session


def test_timeline_has_every_part_and_ends():
    d = ProtocolDisplay(build_steps((720, 450)))
    labels = [s.label for s in d.steps]
    assert labels.count("blink_cue") == 10
    assert labels.count("lights_on") == 3
    assert all(labels.count(lvl) == 5 for lvl in LEVELS)
    assert "reading" in labels and labels[2] == "baseline" and "break" in labels
    assert 6 < d.total_s / 60 < 8
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
    assert len(report_lines(r)) == 7


def test_arithmetic_trials_three_levels_interleaved():
    trials = arithmetic_trials(5, seed=1)
    assert len(trials) == 15 and len({t.problem for t in trials}) == 15
    for k in range(5):                      # every group of three has one of each level
        assert {t.level for t in trials[3 * k:3 * k + 3]} == set(LEVELS)
    for t in trials:
        small, big = sorted((t.a, t.b))
        if t.level == "task_easy":
            assert big < 10
        elif t.level == "task_medium":
            assert small < 10 <= big and big % 10
        else:
            assert 12 <= small < big < 20
    assert [t.problem for t in arithmetic_trials(5, seed=1)] == [t.problem for t in trials]


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
