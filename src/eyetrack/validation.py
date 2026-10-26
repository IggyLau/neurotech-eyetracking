"""Pass/fail checks for the validation protocol (docs/validation_protocol.md).

Run after `eyetrack process` on a session recorded for one protocol test. Every session gets the
signal-quality checks; the test named in the folder (e.g. ..._t3_blinks) adds its own. Targets come
from docs/benchmarks.md where a benchmark exists, and are marked as starting points where it doesn't.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TESTS = ("t1_fixation", "t2_ninepoint", "t3_blinks", "t4_saccades", "t5_light", "t6_arithmetic")


@dataclass
class Check:
    name: str
    value: float | None
    good: str        # target for "pass"
    verdict: str     # pass | usable | fail | n/a
    note: str = ""


def _grade(value, good, usable=None, higher_is_better=True) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    ok = value >= good if higher_is_better else value <= good
    if ok:
        return "pass"
    if usable is not None and (value >= usable if higher_is_better else value <= usable):
        return "usable"
    return "fail"


def _load(session: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_path = session / "metrics_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"{summary_path} not found; run `eyetrack process {session}` first")
    summary = json.loads(summary_path.read_text())
    pupil = pd.read_csv(session / "pupil.csv")
    markers = pd.read_csv(session / "markers_s.csv") if (session / "markers_s.csv").exists() else pd.DataFrame(
        columns=["t_s", "label"])
    epochs = (pd.read_csv(session / "pupil_epochs_summary.csv")
              if (session / "pupil_epochs_summary.csv").exists() else pd.DataFrame())
    return summary, pupil, markers, epochs


def dropped_frames(session: Path, nominal_fps: float) -> float | None:
    """Fraction of frames missing, from the recorded timestamps (total time x nominal rate vs frames got)."""
    ts_path = session / "eye_ts.csv"
    if not ts_path.exists():
        return None
    t = pd.read_csv(ts_path).t_ns.to_numpy(np.int64) / 1e9
    if len(t) < 2:
        return None
    expected = (t[-1] - t[0]) * nominal_fps + 1
    return max(0.0, 1 - len(t) / expected)


def signal_checks(session: Path, summary: dict, nominal_fps: float) -> list[Check]:
    noise = summary.get("noise_rms_s2s_deg")
    drop = dropped_frames(session, nominal_fps)
    model = summary.get("eye_model") or {}
    return [
        Check("sample rate (Hz)", summary.get("sample_rate_hz"), ">= 110",
              _grade(summary.get("sample_rate_hz"), 110, 90), "saccade metrics need ~120 Hz (benchmarks)"),
        Check("dropped frames (%)", None if drop is None else 100 * drop, "<= 2",
              _grade(None if drop is None else 100 * drop, 2, 5, higher_is_better=False)),
        Check("valid rate", summary.get("valid_rate"), ">= 0.95",
              _grade(summary.get("valid_rate"), 0.95, 0.85), "frames with a trusted pupil"),
        Check("mean confidence", summary.get("mean_confidence"), ">= 0.80",
              _grade(summary.get("mean_confidence"), 0.80, 0.60)),
        Check("eye model fit residual (px)", model.get("residual_px"), "<= 5",
              _grade(model.get("residual_px"), 5, 10, higher_is_better=False),
              "needed for degrees; refit after the camera moves"),
        Check("noise RMS-S2S (deg)", noise, "<= 0.28",
              _grade(noise, 0.28, 0.70, higher_is_better=False),
              "0.28 ~ 0.2 deg noise: saccade F1 0.85; 0.7 ~ 0.5 deg: F1 0.76 (benchmarks)"),
    ]


def fixation_checks(pupil: pd.DataFrame) -> list[Check]:
    good = pupil[pupil.valid & ~pupil.in_blink]
    if len(good) < 60:
        return [Check("fixation checks", None, "", "n/a", "too few valid samples")]
    d = good.diameter_clean_px.dropna()
    cv = 100 * d.std() / d.mean() if len(d) else None
    # drift: change in median eye position between the first and last 5 s
    if "yaw_deg" in good:
        x, y, unit = good.yaw_deg, good.pitch_deg, "deg"
    else:
        x, y, unit = good.cx, good.cy, "px"
    t = good.t_s
    first, last = t <= t.min() + 5, t >= t.max() - 5
    drift = float(np.hypot(x[last].median() - x[first].median(), y[last].median() - y[first].median()))
    drift_good, drift_ok = (0.5, 1.5) if unit == "deg" else (3, 8)
    return [
        Check("pupil size variation, steady (CV %)", cv, "<= 3",
              _grade(cv, 3, 8, higher_is_better=False),
              "starting point: <3% can show mental-effort dilation (5-15%); <8% only the light reflex (20-40%)"),
        Check(f"drift over the recording ({unit})", drift, f"<= {drift_good}",
              _grade(drift, drift_good, drift_ok, higher_is_better=False), "camera or glasses slipping"),
    ]


def blink_checks(summary: dict, markers: pd.DataFrame) -> list[Check]:
    cued = int((markers.label == "blink_cue").sum())
    found = summary.get("blink_count", 0)
    if not cued:
        return [Check("blinks found vs cued", found, "press b before each blink", "n/a", "no blink_cue markers")]
    err = abs(found - cued)
    return [Check(f"blinks found vs cued ({cued})", found, f"{cued} +- 1",
                  "pass" if err <= 1 else "usable" if err <= 2 else "fail",
                  "the glare threshold is not validated on labelled blinks (benchmarks)")]


def saccade_checks(summary: dict, session: Path, expected: int) -> list[Check]:
    count = summary.get("saccade_count", 0)
    sacc = pd.read_csv(session / "saccades.csv") if (session / "saccades.csv").exists() else pd.DataFrame()
    big = sacc[sacc.amplitude > 2] if len(sacc) else sacc     # the target jumps, not small corrections
    cv = 100 * big.amplitude.std() / big.amplitude.mean() if len(big) > 2 else None
    ratio = len(big) / expected if expected else None
    return [
        Check(f"target saccades found (expected {expected})", len(big), f"{int(0.85 * expected)}-{int(1.15 * expected)}",
              "pass" if ratio and 0.85 <= ratio <= 1.15 else "usable" if ratio and 0.7 <= ratio <= 1.3 else "fail",
              f"all saccades incl. small ones: {count}"),
        Check("amplitude consistency (CV %)", cv, "<= 25", _grade(cv, 25, 40, higher_is_better=False),
              "same two targets every time, so amplitudes should match"),
    ]


def epoch_checks(epochs: pd.DataFrame, label: str, column: str, good: float, usable: float,
                 name: str, below: bool) -> list[Check]:
    sel = epochs[epochs.label == label] if len(epochs) else epochs
    if not len(sel):
        return [Check(name, None, "", "n/a", f"no '{label}' marker epochs")]
    v = float(sel[column].mean())
    return [Check(name, v, f"{'<=' if below else '>='} {good}",
                  _grade(v, good, usable, higher_is_better=not below), f"{len(sel)} epoch(s), mean of {column}")]


def check(session: str | Path, test: str | None = None, nominal_fps: float = 120.0,
          expected_saccades: int = 30) -> tuple[str | None, list[Check]]:
    session = Path(session)
    if test is None:
        test = next((t for t in TESTS if session.name.endswith(t)), None)
    summary, pupil, markers, epochs = _load(session)
    checks = signal_checks(session, summary, nominal_fps)
    if test == "t1_fixation":
        checks += fixation_checks(pupil)
    elif test == "t3_blinks":
        checks += blink_checks(summary, markers)
    elif test == "t4_saccades":
        checks += saccade_checks(summary, session, expected_saccades)
    elif test == "t5_light":
        checks += epoch_checks(epochs, "lights_on", "min_change_pct", -10, -5, "constriction when light on (%)", True)
        checks += epoch_checks(epochs, "lights_off", "max_change_pct", 5, 2, "recovery when light off (%)", False)
    elif test == "t6_arithmetic":
        checks += epoch_checks(epochs, "task_start", "mean_change_pct", 5, 2, "dilation during the task (%)", False)
    return test, checks


def format_report(session: str | Path, test: str | None, checks: list[Check]) -> str:
    lines = [f"Validation: {Path(session).name}  (test: {test or 'signal quality only'})", ""]
    for c in checks:
        v = "-" if c.value is None or (isinstance(c.value, float) and np.isnan(c.value)) else (
            f"{c.value:.3g}" if isinstance(c.value, (int, float)) else str(c.value))
        lines.append(f"  {c.verdict.upper():7s} {c.name:42s} {v:>8s}   target {c.good:12s} {c.note}")
    verdicts = [c.verdict for c in checks if c.verdict != "n/a"]
    lines += ["", f"  {verdicts.count('pass')} pass, {verdicts.count('usable')} usable, {verdicts.count('fail')} fail"]
    return "\n".join(lines)
