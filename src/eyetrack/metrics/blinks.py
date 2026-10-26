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


# Event kinds that mean the eye itself was (partly) closed. Only these hide the surrounding frames from
# later stages; lighting dropouts, camera gaps and lost tracking only lose their own missing frames.
EYELID_KINDS = ("blink", "closed", "closed_or_lost")


def classify_dropouts(t_s: np.ndarray, valid: np.ndarray, frame_brightness: np.ndarray, eye_brightness: np.ndarray,
                      min_ms: float = 50, max_ms: float = 500, merge_gap_ms: float = 0,
                      lighting_drop: float = 0.08, eyelid_rise: float = 3.0, context_ms: float = 100) -> pd.DataFrame:
    """Blink vs dropout, from image brightness, which doesn't depend on the pupil detector.

    For each run of frames without a trusted pupil, compare the frames inside it with the trusted frames
    just before (context_ms):
      eye area brighter by >= eyelid_rise grey levels  -> the eyelid covered the eye: "blink" (min_ms to
                                                          max_ms), "closed" (longer) or "dropout" (shorter)
      whole frame darker by >= lighting_drop (fraction) -> "lighting": exposure / screen flicker, eye open
      neither                                          -> "lost" (glare, reflection, pupil out of view), or
                                                          "dropout" if shorter than min_ms
    Missing camera frames (timestamp gaps > 2.5 x the usual interval) are added as "camera_gap".
    merge_gap_ms joins eyelid events separated by a brief half-open flicker; other kinds are never merged,
    so a run of lighting dropouts can't grow into one long "closed" event.
    """
    t_s = np.asarray(t_s, float)
    valid = np.asarray(valid, bool)
    fb = np.asarray(frame_brightness, float)
    eb = np.asarray(eye_brightness, float)
    dt = float(np.median(np.diff(t_s))) if len(t_s) > 1 else 0.0
    ctx = max(1, int(round(context_ms / 1000 / dt))) if dt else 1
    events = []
    for start, end in _runs(~valid):
        before = np.arange(max(0, start - ctx), start)
        before = before[valid[before]] if len(before) else before
        onset = t_s[start - 1] if start > 0 else t_s[start]
        offset = t_s[end] if end < len(t_s) else t_s[end - 1] + dt
        dur_ms = (offset - onset) * 1000
        if len(before):
            fb0, eb0 = np.median(fb[before]), np.median(eb[before])
            darker = (fb0 - np.min(fb[start:end])) / max(fb0, 1e-6) >= lighting_drop
            lid = np.median(eb[start:end]) - eb0 >= eyelid_rise
        else:
            darker = lid = False
        if lid and not darker:
            kind = "dropout" if dur_ms < min_ms else "blink" if dur_ms <= max_ms else "closed"
        elif darker:
            kind = "lighting"
        else:
            kind = "dropout" if dur_ms < min_ms else "lost"
        events.append([onset, offset, start, end, kind])
    # join eyelid events (and the short dropouts at their edges) separated by a brief valid flicker
    if merge_gap_ms > 0 and events:
        merged = [events[0]]
        for ev in events[1:]:
            prev = merged[-1]
            kinds = {prev[4], ev[4]}
            lidlike = kinds <= {"blink", "closed", "dropout"} and bool(kinds & {"blink", "closed"})
            if lidlike and (ev[0] - prev[1]) * 1000 <= merge_gap_ms:
                prev[1], prev[3] = ev[1], ev[3]
                d = (prev[1] - prev[0]) * 1000
                prev[4] = "blink" if d <= max_ms else "closed"
            else:
                merged.append(ev)
        events = merged
    rows = [{"onset_s": a, "offset_s": b, "duration_ms": (b - a) * 1000, "start_index": int(s_), "end_index": int(e_),
             "kind": k} for a, b, s_, e_, k in events]
    # camera gaps: frames that never arrived
    for i in np.flatnonzero(np.diff(t_s) > 2.5 * dt) if dt else []:
        rows.append({"onset_s": t_s[i], "offset_s": t_s[i + 1], "duration_ms": (t_s[i + 1] - t_s[i]) * 1000,
                     "start_index": int(i + 1), "end_index": int(i + 1), "kind": "camera_gap"})
    out = pd.DataFrame(rows, columns=["onset_s", "offset_s", "duration_ms", "start_index", "end_index", "kind"])
    return out.sort_values("onset_s").reset_index(drop=True)


def blink_mask(t_s: np.ndarray, blinks: pd.DataFrame, pad_ms: float = 25) -> np.ndarray:
    """True for samples inside a blink or eyes-closed period, padded on both sides (eyelid partly closed).
    Lighting dropouts, camera gaps and lost tracking are not included: only their own missing frames are lost."""
    t_s = np.asarray(t_s, float)
    mask = np.zeros(len(t_s), bool)
    pad = pad_ms / 1000
    for _, b in blinks[blinks["kind"].isin(EYELID_KINDS)].iterrows():
        mask |= (t_s >= b.onset_s - pad) & (t_s <= b.offset_s + pad)
    return mask
