"""Event matching and agreement scores shared by the benchmarks."""

from __future__ import annotations

import numpy as np


def match_events(gt: list[tuple[float, float]], pred: list[tuple[float, float]], tol_s: float = 0.0):
    """One-to-one matching of (onset, offset) intervals by largest overlap (Hooge et al. 2018 style).

    Returns list of (gt_index, pred_index) pairs. Intervals match if they overlap,
    after widening each by tol_s (to allow for one-sample timing differences).
    """
    pairs = []
    for gi, (gs, ge) in enumerate(gt):
        for pi, (ps, pe) in enumerate(pred):
            ov = min(ge, pe) + tol_s - max(gs, ps)
            if ov > 0:
                pairs.append((ov, gi, pi))
    pairs.sort(reverse=True)
    used_g, used_p, out = set(), set(), []
    for _, gi, pi in pairs:
        if gi not in used_g and pi not in used_p:
            used_g.add(gi)
            used_p.add(pi)
            out.append((gi, pi))
    return out


def prf(n_gt: int, n_pred: int, n_match: int) -> dict:
    p = n_match / n_pred if n_pred else np.nan
    r = n_match / n_gt if n_gt else np.nan
    f = 2 * p * r / (p + r) if p and r and np.isfinite(p) and np.isfinite(r) else 0.0
    return {"precision": p, "recall": r, "f1": f, "n_gt": n_gt, "n_pred": n_pred}


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, bool)
    b = np.asarray(b, bool)
    po = np.mean(a == b)
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate([[False], np.asarray(mask, bool), [False]])
    edges = np.flatnonzero(np.diff(padded.astype(int)))
    return list(zip(edges[::2], edges[1::2]))
