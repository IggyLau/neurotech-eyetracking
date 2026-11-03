import importlib.util

import numpy as np
import pytest

from eyetrack.benchmark import compare, lund
from eyetrack.config import load_config

from synthetic import step_trace


def _recording(noise=0.05):
    t, x, y, onsets = step_trace(fs=120, n_steps=10, step_deg=10, noise=noise)
    label = np.full(len(t), lund.FIX)
    for t0 in onsets:
        label[(t >= t0) & (t < t0 + 0.04)] = lund.SAC
    rec = lund.Recording("synthetic", "img", t, x, y, np.ones(len(t), bool), label, 120.0)
    return rec, onsets


@pytest.mark.parametrize("name", ["eyetrack I-VT (default)", "eyetrack step",
                                  "pymovements Engbert-Kliegl", "REMoDNaV"])
def test_each_detector_finds_the_synthetic_saccades(name):
    module = {"pymovements Engbert-Kliegl": "pymovements", "REMoDNaV": "remodnav"}.get(name)
    if module and importlib.util.find_spec(module) is None:
        pytest.skip(f"{module} not installed (pip install -e '.[compare]')")
    rec, onsets = _recording()
    pred, amps = compare.DETECTORS[name](rec, load_config())
    n_gt, n_pred, n_match, amp_err, _ = compare._score(rec, pred, amps, 120)
    assert n_gt == len(onsets)
    assert n_match >= len(onsets) - 1
    assert np.median(amp_err) < 1.5
