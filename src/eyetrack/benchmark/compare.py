"""Compare this package's saccade detectors with pymovements and REMoDNaV on Lund2013.

All detectors get the same resampled, noise-added Lund2013 signal (see benchmark.lund) and are
scored against the same expert labels with the same event matching. Requires the optional
comparison packages: pip install -e ".[compare]"

    eyetrack benchmark compare
"""

from __future__ import annotations

import logging
import warnings
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from ..metrics.events import detect_events, detect_steps, event_kwargs, step_kwargs
from . import lund
from .scoring import match_events, prf


def _ours_ivt(sim, cfg):
    s, _, _ = detect_events(sim.t_s, sim.x_deg, sim.y_deg, sim.valid, **event_kwargs(cfg["events"]))
    return list(zip(s.onset_s, s.offset_s + 1 / sim.fs)), list(s.amplitude)


def _ours_step(sim, cfg):
    s, _, _ = detect_steps(sim.t_s, sim.x_deg, sim.y_deg, sim.valid, **step_kwargs(cfg["events"]))
    return list(zip(s.onset_s, s.offset_s + 1 / sim.fs)), list(s.amplitude)


def _pymovements(sim, cfg, threshold_factor: float = 6.0):
    """Engbert & Kliegl (2003/2015): per-axis median-based velocity SD x lambda, elliptic threshold."""
    import pymovements as pm
    pos = np.stack([sim.x_deg, sim.y_deg], 1)
    pos[~sim.valid] = np.nan
    vel = pm.transforms.numpy.pos2vel(pos, sampling_rate=sim.fs, method="smooth")
    t_ms = np.round(sim.t_s * 1000).astype(np.int64)
    ev = pm.events.microsaccades(vel, timesteps=t_ms, minimum_duration=6,
                                 threshold_factor=threshold_factor).frame
    on = ev["onset"].to_numpy() / 1000
    off = ev["offset"].to_numpy() / 1000 + 1 / sim.fs
    amps = []
    for a, b in zip(on, off):
        i = max(np.searchsorted(sim.t_s, a) - 1, 0)
        j = min(np.searchsorted(sim.t_s, b), len(sim.t_s) - 1)
        amps.append(float(np.hypot(sim.x_deg[j] - sim.x_deg[i], sim.y_deg[j] - sim.y_deg[i])))
    return list(zip(on, off)), amps


def _remodnav(sim, cfg):
    """Dar et al. (2021): adaptive thresholds in local context windows, with PSO and pursuit classes."""
    from remodnav.clf import EyegazeClassifier
    data = np.rec.fromarrays([np.where(sim.valid, sim.x_deg, np.nan), np.where(sim.valid, sim.y_deg, np.nan)],
                             names="x,y")
    clf = EyegazeClassifier(px2deg=1.0, sampling_rate=sim.fs, min_saccade_duration=1 / sim.fs)
    # default filter lengths are below one sample at 120 Hz; use the shortest valid ones
    events = clf(clf.preproc(data, savgol_length=3 / sim.fs, median_filter_length=5 / sim.fs))
    sac = [e for e in events if e["label"] in ("SACC", "ISAC")]
    return ([(e["start_time"], e["end_time"]) for e in sac],
            [float(np.hypot(e["end_x"] - e["start_x"], e["end_y"] - e["start_y"])) for e in sac])


DETECTORS = {
    "eyetrack I-VT (default)": _ours_ivt,
    "eyetrack step": _ours_step,
    "pymovements Engbert-Kliegl": _pymovements,
    "REMoDNaV": _remodnav,
}


def _score(gt_rec, pred, amps, fs):
    dt = 1 / gt_rec.fs
    gt_idx = lund.gt_events(gt_rec, lund.SAC)
    gt = [(gt_rec.t_s[s], gt_rec.t_s[e - 1] + dt) for s, e in gt_idx]
    gt_amp = [lund.gt_saccade_amplitude(gt_rec, s, e) for s, e in gt_idx]
    m = match_events(gt, pred, tol_s=1 / fs)
    amp_err = [abs(amps[p] - gt_amp[g]) for g, p in m if np.isfinite(amps[p])]
    onset_err = [abs(pred[p][0] - gt[g][0]) * 1000 for g, p in m]
    return len(gt), len(pred), len(m), amp_err, onset_err


def run(root: str | Path, cfg: dict, rate: float = 120, noises=(0.0, 0.2, 0.5), seed: int = 0,
        detectors: dict | None = None) -> pd.DataFrame:
    """Pooled saccade precision / recall / F1 per detector and noise level.

    A detector that raises on a recording is counted in `files_failed` and that recording is
    left out of its scores (REMoDNaV, for example, cannot filter signals containing gaps).
    """
    warnings.filterwarnings("ignore")
    logging.getLogger("remodnav").setLevel(logging.ERROR)
    detectors = detectors or DETECTORS
    recs = [lund.load_recording(f) for f in lund.ground_truth_files(Path(root))]
    rows = []
    for noise in noises:
        sims = [(r, lund.resample(r, rate, noise, seed=seed + zlib.crc32(r.name.encode())))
                for r in recs]
        for name, fn in detectors.items():
            n_gt = n_pr = n_m = failed = 0
            amp_errs, onset_errs = [], []
            for r, sim in sims:
                try:
                    pred, amps = fn(sim, cfg)
                except Exception:
                    failed += 1
                    continue
                g, p, m, ae, oe = _score(r, pred, amps, rate)
                n_gt, n_pr, n_m = n_gt + g, n_pr + p, n_m + m
                amp_errs += ae
                onset_errs += oe
            s = prf(n_gt, n_pr, n_m)
            rows.append({"noise_deg": noise, "detector": name, "precision": s["precision"],
                         "recall": s["recall"], "f1": s["f1"],
                         "amp_err_deg": float(np.median(amp_errs)) if amp_errs else np.nan,
                         "onset_err_ms": float(np.median(onset_errs)) if onset_errs else np.nan,
                         "files_failed": failed, "files": len(sims)})
    return pd.DataFrame(rows)
