"""Benchmark saccade / fixation / blink detection on Lund2013 (Andersson et al. 2017).

Lund2013: SMI HiSpeed 1250 tower tracker, 500 Hz (a few files are really 200 Hz),
hand-labelled sample by sample by two experts (MN, RA):
    1 fixation, 2 saccade, 3 post-saccadic oscillation, 4 smooth pursuit, 5 blink, 6 undefined

To simulate our OV9281 we resample the 500 Hz gaze to a lower rate (default 120 Hz)
and optionally add white noise, run the same metrics code the pipeline uses, and
score the result against the expert labels. Expert-vs-expert agreement (MN vs RA)
is reported as the ceiling no detector can be expected to beat.

Get the data:  git clone https://github.com/richardandersson/EyeMovementDetectorEvaluation data/datasets/lund2013
(GPL-3.0; kept out of this repo)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io

from ..metrics.blinks import detect_blinks
from ..metrics.events import detect_events, event_kwargs
from .scoring import cohen_kappa, match_events, prf, runs

FIX, SAC, PSO, PUR, BLINK, UNDEF = 1, 2, 3, 4, 5, 6
ARTICLE_DIR = Path("annotated_data") / "data used in the article"
ZEMBLYS_FIX = Path("annotated_data") / "fix_by_Zemblys2018" / "UH29_img_Europe_labelled_FIX_MN.mat"


@dataclass
class Recording:
    name: str
    stimulus: str
    t_s: np.ndarray
    x_deg: np.ndarray
    y_deg: np.ndarray
    valid: np.ndarray
    label: np.ndarray
    fs: float


def px_to_deg(px: np.ndarray, res: float, dim_m: float, dist_m: float) -> np.ndarray:
    """Screen pixels -> visual degrees from screen center (exact, not small-angle)."""
    metres = (np.asarray(px, float) - res / 2) * dim_m / res
    return np.degrees(np.arctan2(metres, dist_m))


def load_recording(path: Path) -> Recording:
    et = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)["ETdata"]
    pos = et.pos
    t_us = pos[:, 0]
    # 'dots' files have no timestamps; some files claim 500 Hz but are 200 Hz (Zemblys 2019)
    if np.isfinite(t_us).mean() > 0.9:
        fs = 1e6 / np.nanmedian(np.diff(t_us))
    else:
        fs = float(et.sampFreq)
    t = np.arange(len(pos)) / fs
    valid = (pos[:, 3] != 0) | (pos[:, 4] != 0)
    x = px_to_deg(pos[:, 3], et.screenRes[0], et.screenDim[0], et.viewDist)
    y = px_to_deg(pos[:, 4], et.screenRes[1], et.screenDim[1], et.viewDist)
    x[~valid] = np.nan
    y[~valid] = np.nan
    return Recording(path.stem, path.parent.name, t, x, y, valid, pos[:, 5].astype(int), fs)


def resample(rec: Recording, fs_out: float, noise_deg: float = 0.0, seed: int = 0) -> Recording:
    """Point-sample at fs_out (like a slower camera would), labels/validity from the nearest sample."""
    if fs_out >= rec.fs:
        t = rec.t_s
        idx = np.arange(len(t))
    else:
        t = np.arange(0, rec.t_s[-1], 1 / fs_out)
        idx = np.clip(np.round(t * rec.fs).astype(int), 0, len(rec.t_s) - 1)
    x = rec.x_deg[idx].copy()
    y = rec.y_deg[idx].copy()
    if noise_deg > 0:
        rng = np.random.default_rng(seed)
        x += rng.normal(0, noise_deg, len(x))
        y += rng.normal(0, noise_deg, len(y))
    return Recording(rec.name, rec.stimulus, t, x, y, rec.valid[idx], rec.label[idx], min(fs_out, rec.fs))


def gt_events(rec: Recording, code: int) -> list[tuple[int, int]]:
    return runs(rec.label == code)


def gt_saccade_amplitude(rec: Recording, s: int, e: int) -> float:
    a, b = max(s - 1, 0), min(e, len(rec.t_s) - 1)
    return float(np.hypot(rec.x_deg[b] - rec.x_deg[a], rec.y_deg[b] - rec.y_deg[a]))


def evaluate(gt_rec: Recording, sim: Recording, ev_kwargs: dict, cfg_blinks: dict) -> dict:
    sacc, fix, labels = detect_events(sim.t_s, sim.x_deg, sim.y_deg, sim.valid, **ev_kwargs)
    dt = 1 / gt_rec.fs
    # ground truth from the full-rate labels, in seconds
    gt_sac = [(gt_rec.t_s[s], gt_rec.t_s[e - 1] + dt) for s, e in gt_events(gt_rec, SAC)]
    gt_amp = [gt_saccade_amplitude(gt_rec, s, e) for s, e in gt_events(gt_rec, SAC)]
    pr_sac = list(zip(sacc.onset_s, sacc.offset_s + 1 / sim.fs))
    m = match_events(gt_sac, pr_sac, tol_s=1 / sim.fs)
    out = {f"sac_{k}": v for k, v in prf(len(gt_sac), len(pr_sac), len(m)).items()}
    if m:
        out["sac_onset_err_ms"] = float(np.median([abs(pr_sac[p][0] - gt_sac[g][0]) * 1000 for g, p in m]))
        amp_err = [sacc.amplitude.iloc[p] - gt_amp[g] for g, p in m]
        out["sac_amp_err_deg"] = float(np.median(np.abs(amp_err)))
        out["sac_amp_bias_deg"] = float(np.median(amp_err))
        out["sac_dur_gt_ms"] = float(np.median([(gt_sac[g][1] - gt_sac[g][0]) * 1000 for g, _ in m]))
        out["sac_dur_pred_ms"] = float(np.median([sacc.duration_ms.iloc[p] for _, p in m]))

    # sample-level saccade agreement on the simulated time base
    out["sac_kappa"] = cohen_kappa(sim.label == SAC, labels == "sac")

    # fixations: I-VT has no smooth-pursuit class, so score against fixation labels only
    # on image-viewing data (no pursuit); elsewhere pursuit would count as false fixations
    gt_fix = [(gt_rec.t_s[s], gt_rec.t_s[e - 1] + dt) for s, e in gt_events(gt_rec, FIX)
              if (e - s) * dt * 1000 >= 40]
    pr_fix = list(zip(fix.onset_s, fix.offset_s + 1 / sim.fs))
    mf = match_events(gt_fix, pr_fix, tol_s=1 / sim.fs)
    out.update({f"fix_{k}": v for k, v in prf(len(gt_fix), len(pr_fix), len(mf)).items()})
    if mf:
        out["fix_dur_err_ms"] = float(np.median([abs((pr_fix[p][1] - pr_fix[p][0]) - (gt_fix[g][1] - gt_fix[g][0])) * 1000
                                                 for g, p in mf]))

    blinks = detect_blinks(sim.t_s, sim.valid, cfg_blinks["min_duration_ms"], cfg_blinks["max_duration_ms"],
                           cfg_blinks.get("merge_gap_ms", 0))
    pr_bl = [(b.onset_s, b.offset_s) for b in blinks.itertuples() if b.kind in ("blink", "closed_or_lost")]
    gt_bl = [(gt_rec.t_s[s], gt_rec.t_s[e - 1] + dt) for s, e in gt_events(gt_rec, BLINK)]
    mb = match_events(gt_bl, pr_bl, tol_s=1 / sim.fs)
    out.update({f"blink_{k}": v for k, v in prf(len(gt_bl), len(pr_bl), len(mb)).items()})
    return out


def human_ceiling(root: Path) -> dict:
    """Saccade agreement between the two expert coders (MN as truth, RA as 'detector')."""
    n_gt = n_pr = n_m = 0
    for mn in sorted((root / ARTICLE_DIR).glob("*/*_MN.mat")):
        ra = mn.with_name(mn.name.replace("_MN.mat", "_RA.mat"))
        if not ra.exists():
            continue
        a, b = load_recording(mn), load_recording(ra)
        dt = 1 / a.fs
        ga = [(a.t_s[s], a.t_s[e - 1] + dt) for s, e in gt_events(a, SAC)]
        gb = [(b.t_s[s], b.t_s[e - 1] + dt) for s, e in gt_events(b, SAC)]
        n_gt, n_pr, n_m = n_gt + len(ga), n_pr + len(gb), n_m + len(match_events(ga, gb, dt))
    return prf(n_gt, n_pr, n_m)


def ground_truth_files(root: Path) -> list[Path]:
    files = sorted((root / ARTICLE_DIR).glob("*/*_MN.mat"))
    fix = root / ZEMBLYS_FIX
    if fix.exists():  # Zemblys 2018 corrected an annotator error in this file
        files = [fix if f.name == "UH29_img_Europe_labelled_MN.mat" else f for f in files]
    return files


def run(root: str | Path, cfg: dict, rates=(500, 120, 60, 24), noises=(0.0, 0.2, 0.5),
        ev_kwargs: dict | None = None) -> pd.DataFrame:
    root = Path(root)
    ev_kwargs = ev_kwargs or event_kwargs(cfg["events"])
    files = ground_truth_files(root)
    if not files:
        raise FileNotFoundError(f"No Lund2013 files under {root}. See the module docstring to download.")
    recs = [load_recording(f) for f in files]
    for r, f in zip(recs, files):
        r.stimulus = "img" if "img" in f.name else f.parent.name
    rows = []
    for fs in rates:
        for noise in noises:
            for rec in recs:
                sim = resample(rec, fs, noise, seed=hash(rec.name) % 2**32)
                rows.append({"rate_hz": fs, "noise_deg": noise, "file": rec.name, "stimulus": rec.stimulus,
                             **evaluate(rec, sim, ev_kwargs, cfg["blinks"])})
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Pool counts across files (micro-average) per rate x noise x stimulus."""
    def pool(g):
        n_gt, n_pred = g.sac_n_gt.sum(), g.sac_n_pred.sum()
        n_m = np.nansum(g.sac_recall * g.sac_n_gt)
        s = prf(int(n_gt), int(n_pred), int(round(n_m)))
        f_gt, f_pred = g.fix_n_gt.sum(), g.fix_n_pred.sum()
        f = prf(int(f_gt), int(f_pred), int(round(np.nansum(g.fix_recall * g.fix_n_gt))))
        b_gt, b_pred = g.blink_n_gt.sum(), g.blink_n_pred.sum()
        b_m = np.nansum(g.blink_recall.fillna(0) * g.blink_n_gt)
        b = prf(int(b_gt), int(b_pred), int(round(b_m)))
        return pd.Series({
            "saccades": int(n_gt), "sac_precision": s["precision"], "sac_recall": s["recall"], "sac_f1": s["f1"],
            "sac_kappa": g.sac_kappa.mean(), "amp_err_deg": g.sac_amp_err_deg.median(),
            "onset_err_ms": g.sac_onset_err_ms.median(),
            "dur_gt_ms": g.sac_dur_gt_ms.median(), "dur_pred_ms": g.sac_dur_pred_ms.median(),
            "fix_precision": f["precision"], "fix_recall": f["recall"], "fix_f1": f["f1"],
            "fix_dur_err_ms": g.fix_dur_err_ms.median(),
            "blinks": int(b_gt), "blink_recall": b["recall"], "blink_precision": b["precision"],
            "blink_f1": b["f1"],
        })
    stim = df.groupby(["rate_hz", "noise_deg", "stimulus"]).apply(pool, include_groups=False)
    allr = df.groupby(["rate_hz", "noise_deg"]).apply(pool, include_groups=False)
    allr["stimulus"] = "ALL"
    allr = allr.set_index("stimulus", append=True)
    return pd.concat([allr, stim]).sort_index()
