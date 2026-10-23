"""Blink detection from pupil-detection dropouts.

A blink is a run of missing / low-confidence samples lasting 50-500 ms (typical
blink duration ~100-300 ms). Shorter gaps are detection noise; longer ones are
eyes closed or tracking loss and are reported separately.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index pairs of True runs."""
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(np.diff(padded.astype(int)))
    return list(zip(edges[::2], edges[1::2]))


def detect_blinks(t_s: np.ndarray, valid: np.ndarray, min_ms: float = 50, max_ms: float = 500,
                  merge_gap_ms: float = 0, confidence: np.ndarray | None = None,
                  max_blink_confidence: float = 1.0) -> pd.DataFrame:
    """merge_gap_ms: treat brief valid flickers inside a blink (eyelid half-open) as part of it.

    confidence / max_blink_confidence: in a real blink the pupil disappears, so detection
    confidence collapses. A dropout whose median confidence stays above the limit
    (glare or a reflection cutting the pupil) is reported as kind "low_confidence"
    instead of "blink".
    """
    t_s = np.asarray(t_s, float)
    dt = np.median(np.diff(t_s)) if len(t_s) > 1 else 0.0
    missing = ~np.asarray(valid, bool)
    if merge_gap_ms > 0 and dt > 0:
        gap = int(round(merge_gap_ms / 1000 / dt))
        for s, e in _runs(~missing):
            if s > 0 and e < len(missing) and e - s <= gap:
                missing[s:e] = True
    rows = []
    for start, end in _runs(missing):
        onset = t_s[start - 1] if start > 0 else t_s[start]
        offset = t_s[end] if end < len(t_s) else t_s[end - 1] + dt
        dur_ms = (offset - onset) * 1000
        if dur_ms < min_ms:
            kind = "dropout"
        elif confidence is not None and np.nanmedian(np.asarray(confidence)[start:end]) > max_blink_confidence:
            kind = "low_confidence"
        elif dur_ms <= max_ms:
            kind = "blink"
        else:
            kind = "closed_or_lost"
        rows.append({"onset_s": onset, "offset_s": offset, "duration_ms": dur_ms,
                     "start_index": int(start), "end_index": int(end), "kind": kind})
    return pd.DataFrame(rows, columns=["onset_s", "offset_s", "duration_ms", "start_index", "end_index", "kind"])


def blink_mask(t_s: np.ndarray, blinks: pd.DataFrame, pad_ms: float = 25) -> np.ndarray:
    """True for samples inside a blink or closed period, padded on both sides (eyelid partly closed)."""
    t_s = np.asarray(t_s, float)
    mask = np.zeros(len(t_s), bool)
    pad = pad_ms / 1000
    for _, b in blinks[blinks["kind"].isin(["blink", "closed_or_lost"])].iterrows():
        mask |= (t_s >= b.onset_s - pad) & (t_s <= b.offset_s + pad)
    return mask
