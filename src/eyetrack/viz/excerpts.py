"""Visual excerpts of a processed validation session: what the detector saw at the moments that matter.

    excerpts.png           four panels, each with eye-video frames (detected pupil ellipse drawn on) and
                           the signal around them:
                             light reflex   dark -> white flash -> recovery
                             arithmetic     baseline vs peak dilation on the clearest hard problem
                             reading        horizontal eye position with saccades shaded, frames
                                            before / after the largest saccade (a line return)
                             blink          before / during / after a cued blink
                             failures       frames where detection went wrong (wrong object, pupil
                                            lost outside a blink, low confidence), next to the
                                            detection rate in each part of the protocol
    clip_<name>.mp4        a few seconds of annotated eye video around each moment, at normal speed
"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


class Video:
    """Random access to eye.avi by session time (s from the first frame)."""

    def __init__(self, session: Path):
        self.cap = cv2.VideoCapture(str(session / "eye.avi"))
        ts = pd.read_csv(session / "eye_ts.csv").t_ns.to_numpy(np.int64)
        self.t = (ts - ts[0]) / 1e9

    def index(self, t_s: float) -> int:
        return int(np.clip(np.searchsorted(self.t, t_s), 0, len(self.t) - 1))

    def frame(self, i: int) -> np.ndarray | None:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, img = self.cap.read()
        return img if ok else None


def annotate(img: np.ndarray, row: pd.Series, caption: str) -> np.ndarray:
    out = img.copy()
    if bool(row.found) and np.isfinite(row.cx):
        ellipse = ((row.cx, row.cy), (row.axis_minor, row.axis_major), row.angle)
        color = (40, 230, 230) if bool(row.valid) else (60, 60, 230)
        cv2.ellipse(out, ellipse, color, 2, cv2.LINE_AA)
        cv2.circle(out, (int(row.cx), int(row.cy)), 3, (255, 255, 0), -1)
        label = f"d {row.diameter_px:.0f}px  conf {row.confidence:.2f}"
    else:
        label = "no pupil"
    for text, y, scale in ((caption, 28, 0.75), (label, out.shape[0] - 14, 0.6)):
        cv2.putText(out, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _row(pupil: pd.DataFrame, i: int) -> pd.Series:
    return pupil.iloc[min(i, len(pupil) - 1)]


def write_clip(video: Video, pupil: pd.DataFrame, t0: float, t1: float, path: Path, caption: str, fps: float = 120):
    i0, i1 = video.index(t0), video.index(t1)
    writer = None
    video.cap.set(cv2.CAP_PROP_POS_FRAMES, i0)
    for i in range(i0, i1):
        ok, img = video.cap.read()
        if not ok:
            break
        img = annotate(img, _row(pupil, i), f"{caption}  t={video.t[i]:.2f}s")
        if writer is None:
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (img.shape[1], img.shape[0]))
        writer.write(img)
    if writer is not None:
        writer.release()


def make_excerpts(session: str | Path, clips: bool = True) -> Path:
    session = Path(session)
    pupil = pd.read_csv(session / "pupil.csv")
    markers = pd.read_csv(session / "markers_s.csv").sort_values("t_s").reset_index(drop=True)
    sacc = pd.read_csv(session / "saccades.csv")
    blinks = pd.read_csv(session / "blinks.csv")
    video = Video(session)
    t = pupil.t_s.to_numpy()
    d = pupil.diameter_clean_px.to_numpy()

    def next_marker(t_s, labels=None):
        later = markers[(markers.t_s > t_s) & (markers.label.isin(labels) if labels else True)]
        return float(later.t_s.iloc[0]) if len(later) else float(t[-1])

    panels = []   # (title, [(frame, caption)], plot function)

    # 1. light reflex: the first white flash
    on = markers[markers.label == "lights_on"]
    if len(on):
        t_on = float(on.t_s.iloc[0])
        t_off = next_marker(t_on)
        moments = [(t_on - 0.3, "dark: pupil large"), (t_on + 1.5, "white flash: constricted"),
                   (t_off + 4.0, "dark again: re-dilating")]

        def plot_light(ax, t_on=t_on, t_off=t_off, moments=moments):
            sel = (t > t_on - 3) & (t < t_off + 6)
            ax.plot(t[sel] - t_on, d[sel], color="C0")
            ax.axvspan(0, t_off - t_on, color="gold", alpha=0.3, label="white screen")
            for m, _ in moments:
                ax.axvline(m - t_on, color="k", lw=0.6, ls=":")
            ax.set_xlabel("s from flash"); ax.set_ylabel("pupil diameter (px)"); ax.legend(fontsize=8)
        panels.append(("Light reflex", moments, plot_light, (t_on - 1, t_off + 2)))

    # 2. arithmetic: the hard problem with the largest dilation
    trials_path = session / "arithmetic_trials.csv"
    if trials_path.exists():
        tr = pd.read_csv(trials_path)
        hard = tr[(tr.label == "task_hard") & tr.mean_change_pct.notna()]
        if len(hard):
            best = hard.loc[hard.mean_change_pct.idxmax()]
            t_task = float(best.onset_s)
            t_end = next_marker(t_task, ["answer"])
            t_peak = t_task + float(best.t_max_s) if pd.notna(best.t_max_s) else t_task + 3
            prob = best.problem if isinstance(best.get("problem"), str) else "hard problem"
            moments = [(t_task - 0.3, "before the problem"), (t_peak, f"solving {prob}: peak")]

            def plot_task(ax, t_task=t_task, t_end=t_end, best=best, moments=moments):
                sel = (t > t_task - 3) & (t < t_end + 3)
                base = np.nanmean(d[(t >= t_task - 0.5) & (t < t_task)])
                ax.plot(t[sel] - t_task, 100 * (d[sel] - base) / base, color="C1")
                ax.axvspan(0, t_end - t_task, color="C1", alpha=0.12, label="solving")
                ax.axhline(0, color="0.5", lw=0.6)
                for m, _ in moments:
                    ax.axvline(m - t_task, color="k", lw=0.6, ls=":")
                ax.set_xlabel("s from problem"); ax.set_ylabel("change from baseline (%)")
                ax.set_title(f"{best.get('problem', '')}: mean {best.mean_change_pct:+.1f}%", fontsize=9)
                ax.legend(fontsize=8)
            panels.append(("Mental arithmetic (hard)", moments, plot_task, (t_task - 1, min(t_end, t_task + 8))))

    # 3. saccades while reading: the largest saccade (usually a line return) in the reading window
    rd = markers[markers.label == "reading"]
    if len(rd) and len(sacc):
        r0 = float(rd.t_s.iloc[0])
        r1 = next_marker(r0)
        rs = sacc[(sacc.onset_s > r0 + 2) & (sacc.offset_s < r1 - 2)]
        if len(rs):
            big = rs.loc[rs.amplitude.idxmax()]
            moments = [(float(big.onset_s) - 0.05, "before the saccade"), (float(big.offset_s) + 0.05, "after")]
            xcol = "yaw_deg" if "yaw_deg" in pupil else "cx"

            def plot_read(ax, big=big, r0=r0, xcol=xcol, rs=rs):
                a = float(big.onset_s) - 3
                sel = (t > a) & (t < a + 6)
                x = pupil[xcol].where(pupil.valid).to_numpy()
                ax.plot(t[sel] - a, x[sel], color="C2", lw=0.9)
                for s_ in rs[(rs.onset_s > a) & (rs.onset_s < a + 6)].itertuples():
                    ax.axvspan(s_.onset_s - a, s_.offset_s - a, color="C3", alpha=0.3)
                ax.set_xlabel("s"); ax.set_ylabel("horizontal eye position (deg)" if xcol == "yaw_deg" else "pupil x (px)")
                ax.set_title(f"red = detected saccades; largest {big.amplitude:.1f} deg", fontsize=9)
            panels.append(("Reading saccades", moments, plot_read, (float(big.onset_s) - 1.5, float(big.onset_s) + 1.5)))

    # 4. a cued blink
    cues = markers[markers.label == "blink_cue"].t_s.to_numpy()
    real = blinks[blinks.kind == "blink"] if len(blinks) else blinks
    for c in cues:
        hit = real[(real.onset_s >= c) & (real.onset_s <= c + 1.5)]
        if len(hit):
            b = hit.iloc[0]
            mid = (b.onset_s + b.offset_s) / 2
            moments = [(b.onset_s - 0.15, "before"), (mid, f"blink ({b.duration_ms:.0f} ms)"), (b.offset_s + 0.15, "after")]

            def plot_blink(ax, b=b):
                a = b.onset_s - 1
                sel = (t > a) & (t < a + 2.5)
                ax.plot(t[sel] - a, pupil.confidence.to_numpy()[sel], color="C4")
                ax.axvspan(b.onset_s - a, b.offset_s - a, color="0.3", alpha=0.25, label="blink")
                ax.set_ylim(0, 1.05); ax.set_xlabel("s"); ax.set_ylabel("detection confidence"); ax.legend(fontsize=8)
            panels.append(("Cued blink", moments, plot_blink, (b.onset_s - 0.6, b.offset_s + 0.6)))
            break

    # 5. failures: worst examples outside blinks, spaced at least 1 s apart
    phases = markers.copy()
    phases["end"] = list(phases.t_s[1:]) + [float(t[-1])]
    part_of = {"baseline": "baseline", "blink_cue": "blinks", "blink_wait": "blinks", "lights_on": "light (white)",
               "lights_off": "light (dark)", "reading": "reading", "arith_fix": "arithmetic", "answer": "arithmetic",
               "task_easy": "arithmetic", "task_medium": "arithmetic", "task_hard": "arithmetic",
               "span_fix": "digit span", "span_3": "digit span", "span_5": "digit span", "span_7": "digit span",
               "recall": "digit span"}
    part = np.full(len(t), "", dtype=object)
    for ph in phases.itertuples():
        if ph.label in part_of:
            part[(t >= ph.t_s) & (t < ph.end)] = part_of[ph.label]
    pupil["part"] = part
    outside_blink = ~pupil.in_blink.astype(bool) & (pupil.part != "")
    kinds = [
        ("wrong object (size jump)", outside_blink & pupil.size_outlier.astype(bool) & pupil.found.astype(bool),
         lambda df: (df.diameter_px / np.nanmedian(pupil.diameter_px[pupil.valid])).sub(1).abs()),
        ("pupil lost (not a blink)", outside_blink & ~pupil.found.astype(bool), lambda df: -df.t_s * 0),
        ("low confidence", outside_blink & pupil.found.astype(bool) & ~pupil.valid.astype(bool)
         & ~pupil.size_outlier.astype(bool), lambda df: 1 - df.confidence),
    ]
    fails, used = [], []
    for name, mask, badness in kinds:
        cand = pupil[mask]
        if not len(cand):
            continue
        cand = cand.assign(bad=badness(cand)).sort_values("bad", ascending=False)
        for row in cand.itertuples():
            if all(abs(row.t_s - u) > 1.0 for u in used):
                fails.append((row.t_s, f"{name} [{row.part}]"))
                used.append(row.t_s)
                break

    def plot_rates(ax):
        order = ["baseline", "blinks", "light (dark)", "light (white)", "reading", "digit span", "arithmetic"]
        rates = [pupil[pupil.part == p_].valid.mean() * 100 if (pupil.part == p_).any() else np.nan for p_ in order]
        ax.barh(order, rates, color=["C0" if r >= 90 else "C1" if r >= 75 else "C3" for r in rates])
        ax.axvline(95, color="k", lw=0.6, ls=":")
        ax.set_xlim(0, 100); ax.invert_yaxis(); ax.set_xlabel("frames with a trusted pupil (%)")
        ax.set_title("detection rate by part (blinks included)", fontsize=9)
    if fails:
        panels.append(("Detection failing", fails[:3], plot_rates, None))

    # figure: one row per panel, frames then the plot
    n = len(panels)
    fig = plt.figure(figsize=(17, 3.6 * n))
    for r, (title, moments, plot_fn, _) in enumerate(panels):
        for c, (tm, caption) in enumerate(moments):
            i = video.index(tm)
            img = video.frame(i)
            ax = fig.add_subplot(n, 4, 4 * r + c + 1)
            if img is not None:
                ax.imshow(cv2.cvtColor(annotate(img, _row(pupil, i), caption), cv2.COLOR_BGR2RGB))
            ax.set_axis_off()
            if c == 0:
                ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
        plot_fn(fig.add_subplot(n, 4, 4 * r + 4))
    fig.tight_layout()
    out = session / "excerpts.png"
    fig.savefig(out, dpi=90)
    plt.close(fig)

    if clips:
        for title, moments, _, span in panels:
            if span is None:      # failures: a short clip around each example
                for k, (tm, cap) in enumerate(moments):
                    write_clip(video, pupil, tm - 0.75, tm + 0.75, session / f"clip_failure_{k + 1}.mp4", cap)
            else:
                write_clip(video, pupil, span[0], span[1], session / f"clip_{title.split()[0].lower()}.mp4", title)
    return out
