import numpy as np
import pytest

from eyetrack.pupil.detector import PupilResult
from eyetrack.viz.live import PupilTrace, with_trace


def _r(d, conf=0.9):
    return PupilResult(found=True, axis_minor=d, axis_major=d, confidence=conf)


def test_trace_keeps_window_and_measures_change_since_marker():
    tr = PupilTrace(window_s=10, min_confidence=0.5)
    for i in range(120 * 20):
        t = i / 120
        tr.add(t, _r(40.0 if t < 15 else 44.0))
        if i == 120 * 15:
            tr.mark(t, "task_start")
    assert tr.samples[0][0] >= tr.samples[-1][0] - 10    # older samples dropped
    assert tr.marks[-1][2] == pytest.approx(40)          # baseline = 0.5 s before the marker
    assert tr.current() == pytest.approx(44)
    img = with_trace(np.zeros((300, 480, 3), np.uint8), tr)
    assert img.shape == (300 + tr.height, 480, 3)


def test_trace_ignores_low_confidence_and_draws_gaps():
    tr = PupilTrace(min_confidence=0.5)
    tr.add(0.0, _r(40.0))
    tr.add(0.01, _r(80.0, conf=0.2))                    # rejected: shown as a gap
    tr.add(0.02, PupilResult(found=False))
    assert np.isnan(tr.samples[1][1]) and np.isnan(tr.samples[2][1])
    assert tr.draw(320).shape == (tr.height, 320, 3)
    assert PupilTrace().draw(200).shape[1] == 200       # empty trace still draws
