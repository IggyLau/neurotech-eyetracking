"""Frame-by-frame viewer for a processed session: `eyetrack viewer <session>` (the whole recording) or
`eyetrack viewer <session> --part reading` (one part).

Writes, in the session folder:
    viewer_<part>.mp4    the eye video, H.264 at the recorded frame rate, with the detected pupil (yellow =
                         trusted, red = rejected), time, frame number, the current protocol step, and a
                         SACCADE / BLINK banner on the frames inside one
    viewer_<part>.html   the video on top; below it a readout of the current frame, a 3 s close-up of eye
                         position with the saccade detector's step score, whole-recording traces of eye
                         position (saccade ticks), pupil size and confidence (click to jump), and a
                         clickable saccade list
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PART_LABELS = {"baseline": ["baseline"], "targets": [f"target_{k}" for k in range(9)], "blinks": ["blink_cue", "blink_wait"], "light": ["lights_on", "lights_off"],
               "reading": ["reading"], "digit_span": ["span_fix", "span_3", "span_5", "span_7", "recall"],
               "arithmetic": ["arith_fix", "task_easy", "task_medium", "task_hard", "answer"]}
PARTS = ("all", *PART_LABELS)


def part_window(markers: pd.DataFrame, part: str, end_s: float) -> tuple[float, float]:
    """First and last moment of a protocol part, from its markers ("all" = the whole recording)."""
    if part == "all":
        return 0.0, end_s
    m = markers.sort_values("t_s").reset_index(drop=True)
    m["end"] = list(m.t_s[1:]) + [end_s]
    sel = m[m.label.isin(PART_LABELS[part])]
    if not len(sel):
        raise ValueError(f"no '{part}' markers in this session")
    return float(sel.t_s.min()), float(sel.end.max())


def step_labels(markers: pd.DataFrame, times) -> list[str]:
    """The most recent protocol marker at each time ("" before the first)."""
    m = markers.sort_values("t_s")
    idx = np.searchsorted(m.t_s.to_numpy(), np.asarray(times, float), side="right")
    labels = m.label.to_numpy()
    return [labels[i - 1] if i else "" for i in idx]


def make_viewer(session: str | Path, part: str = "all", pad_s: float = 1.0, fps: float = 120.0,
                cfg: dict | None = None) -> Path:
    session = Path(session)
    pupil = pd.read_csv(session / "pupil.csv")
    markers = pd.read_csv(session / "markers_s.csv")
    sacc = pd.read_csv(session / "saccades.csv")
    blinks = pd.read_csv(session / "blinks.csv")
    t = pupil.t_s.to_numpy()
    a, b = part_window(markers, part, float(t[-1]))
    a, b = max(0.0, a - pad_s), min(float(t[-1]), b + pad_s)
    i0, i1 = int(np.searchsorted(t, a)), int(np.searchsorted(t, b, side="right"))
    win = pupil.iloc[i0:i1].reset_index(drop=True)
    sw = sacc[(sacc.onset_s >= a) & (sacc.onset_s < b)].reset_index(drop=True)
    bw = blinks[(blinks.onset_s < b) & (blinks.offset_s > a)] if len(blinks) else blinks
    real_blinks = bw[bw.kind == "blink"] if len(bw) else bw
    steps = step_labels(markers, win.t_s)

    def inside(tt, df):
        return bool(len(df)) and bool(np.any((df.onset_s.to_numpy() <= tt) & (df.offset_s.to_numpy() >= tt)))

    def label(img, txt, org):
        cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    cap = cv2.VideoCapture(str(session / "eye.avi"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, i0)
    video_path = session / f"viewer_{part}.mp4"
    writer = None
    for k in range(len(win)):
        ok, img = cap.read()
        if not ok:
            break
        row = win.iloc[k]
        if bool(row.found) and np.isfinite(row.cx):
            color = (40, 230, 230) if bool(row.valid) else (60, 60, 230)
            cv2.ellipse(img, ((row.cx, row.cy), (row.axis_minor, row.axis_major), row.angle), color, 2, cv2.LINE_AA)
            cv2.circle(img, (int(row.cx), int(row.cy)), 3, (255, 255, 0), -1)
        label(img, f"t {row.t_s:7.3f}s  frame {int(row.frame_index)}  conf {row.confidence:.2f}", (10, 24))
        label(img, steps[k], (10, img.shape[0] - 12))
        banner = "SACCADE" if inside(row.t_s, sw) else "BLINK" if inside(row.t_s, real_blinks) else None
        if banner:
            cv2.rectangle(img, (img.shape[1] - 150, 6), (img.shape[1] - 8, 40), (40, 40, 220) if banner == "SACCADE" else (90, 90, 90), -1)
            cv2.putText(img, banner, (img.shape[1] - 142, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        if writer is None:
            writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"avc1"), fps, (img.shape[1], img.shape[0]))
        writer.write(img)
    if writer is not None:
        writer.release()

    x = win.yaw_deg if "yaw_deg" in win else win.cx
    y = win.pitch_deg if "pitch_deg" in win else win.cy
    # the edge-filter signal the step detector uses, so the viewer can show why a step did or didn't count
    from ..metrics.events import step_signal
    st = (cfg or {}).get("events", {}).get("step", {})
    half, nf, amp = st.get("half_window_ms", 60.0), st.get("noise_factor", 8.0), st.get("min_amplitude", 1.0)
    sig = step_signal(win.t_s.to_numpy(), x.to_numpy(), y.to_numpy(), win.valid.to_numpy(bool), half, nf, amp)
    xm = pd.Series(np.where(win.valid, x, np.nan)).rolling(int(round(half / 1000 * fps)) | 1, center=True,
                                                             min_periods=2).median()
    on = np.sort(sw.onset_s.to_numpy())
    gaps_ms = np.diff(on) * 1000 if len(on) > 1 else np.array([])

    def nums(values, nd):
        return [None if not np.isfinite(v) else round(float(v), nd) for v in values]

    data = {
        "part": part, "fps": fps, "unit": "deg" if "yaw_deg" in win else "px",
        "t": win.t_s.round(4).tolist(),
        "x": [None if not v or not np.isfinite(xx) else round(float(xx), 3) for xx, v in zip(x, win.valid)],
        "y": [None if not v or not np.isfinite(yy) else round(float(yy), 3) for yy, v in zip(y, win.valid)],
        "valid": win.valid.astype(int).tolist(),
        "d": nums(win.diameter_clean_px, 2),
        "conf": nums(win.confidence, 2),
        "xm": nums(xm, 3),
        "score": nums(sig["step"], 3),
        "step": {"threshold": round(sig["threshold"], 3), "sample_sd": round(sig["sample_sd"], 3),
                 "median_noise": round(sig["median_noise"], 3), "samples_per_window": round(sig["samples_per_window"], 1),
                 "half_window_ms": half, "noise_factor": nf, "min_amplitude": amp,
                 "min_separation_ms": st.get("min_separation_ms", 120.0)},
        "gap_ms": [round(float(g), 1) for g in gaps_ms],
        "saccades": sw[["onset_s", "offset_s", "duration_ms", "amplitude", "peak_velocity"]].round(3)
                    .assign(step=step_labels(markers, sw.onset_s)).to_dict("records"),
        "blinks": real_blinks[["onset_s", "offset_s"]].round(3).to_dict("records") if len(real_blinks) else [],
        "markers": markers[(markers.t_s >= a) & (markers.t_s < b)].round(3).to_dict("records"),
    }
    title = "whole session" if part == "all" else part
    html = HTML.replace("__DATA__", json.dumps(data)).replace("__VIDEO__", video_path.name).replace("__PART__", title)
    out = session / f"viewer_{part}.html"
    out.write_text(html)
    return out


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Eye viewer: __PART__</title>
<style>
  :root { --bg:#f6f6f4; --fg:#1d1d1f; --muted:#6b6b70; --card:#fff; --line:#e3e3e0; --sac:#d6453d; --gap:#c9c9c4; --acc:#2f6fb0; --vert:#9a7b2f; --pup:#2f8f5b; --mark:#8a4fbf; }
  @media (prefers-color-scheme: dark) { :root { --bg:#161617; --fg:#ececec; --muted:#9a9aa0; --card:#212123; --line:#333336; --gap:#4a4a4e; --mark:#b58ae0; } }
  body { margin:0; background:var(--bg); color:var(--fg); font:14px/1.45 -apple-system, system-ui, sans-serif; }
  main { max-width:1000px; margin:0 auto; padding:16px; display:flex; flex-direction:column; gap:16px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px; }
  h1 { font-size:18px; margin:0 0 4px; } h2 { font-size:15px; margin:0 0 6px; } .muted { color:var(--muted); }
  video { width:100%; max-height:62vh; object-fit:contain; border-radius:6px; background:#000; display:block; margin-top:8px; }
  .controls { display:flex; flex-wrap:wrap; gap:6px; align-items:center; margin-top:8px; }
  button, select { font:inherit; padding:5px 10px; border-radius:6px; border:1px solid var(--line); background:var(--bg); color:var(--fg); cursor:pointer; }
  button:hover { border-color:var(--acc); }
  input[type=range] { width:100%; margin-top:8px; }
  #loading { padding:40px 0; text-align:center; }
  .stats { display:grid; grid-template-columns:repeat(auto-fill, minmax(140px, 1fr)); gap:8px; font-variant-numeric:tabular-nums; }
  .stat { border:1px solid var(--line); border-radius:8px; padding:6px 8px; }
  .stat b { display:block; font-size:17px; font-weight:600; } .stat span { color:var(--muted); font-size:12px; }
  .stat.hot { border-color:var(--sac); background:color-mix(in srgb, var(--sac) 12%, transparent); }
  canvas { width:100%; display:block; margin-top:6px; cursor:crosshair; }
  table { width:100%; border-collapse:collapse; font-variant-numeric: tabular-nums; font-size:13px; }
  th, td { text-align:right; padding:4px 6px; border-bottom:1px solid var(--line); } th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) { text-align:left; }
  thead th { position:sticky; top:0; background:var(--card); }
  tbody tr { cursor:pointer; } tbody tr:hover { background:color-mix(in srgb, var(--acc) 12%, transparent); }
  tr.now { background:color-mix(in srgb, var(--sac) 18%, transparent); }
  .list { max-height:420px; overflow:auto; }
  .legend span { display:inline-block; margin-right:12px; } .sw { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:4px; vertical-align:-1px; }
  kbd { border:1px solid var(--line); border-radius:4px; padding:0 4px; font-size:12px; }
</style></head>
<body><main>
<section class="card">
  <h1>Eye video: __PART__</h1>
  <div class="muted">Yellow ellipse = trusted pupil, red = rejected. <kbd>←</kbd>/<kbd>→</kbd> one frame, <kbd>Shift</kbd>+arrow 10 frames, <kbd>Space</kbd> play/pause. Click any trace below to jump there.</div>
  <div id="loading" class="muted">Loading the video…</div>
  <video id="v" preload="auto" muted playsinline hidden></video>
  <input id="seek" type="range" min="0" max="1" step="1" value="0">
  <div class="controls">
    <button id="back10">« 10</button><button id="back1">‹ 1</button><button id="play">Play</button>
    <button id="fwd1">1 ›</button><button id="fwd10">10 »</button>
    <select id="rate"><option value="0.1">0.1×</option><option value="0.25" selected>0.25×</option><option value="0.5">0.5×</option><option value="1">1×</option><option value="2">2×</option></select>
    <select id="jump"><option value="">Jump to step…</option></select>
  </div>
</section>

<section class="card">
  <h2>This frame</h2>
  <div class="stats" id="stats"></div>
</section>

<section class="card">
  <h2>Close-up: 3 s around this frame</h2>
  <canvas id="live" style="height:280px"></canvas>
  <div class="legend muted"><span><i class="sw" style="background:#888"></i>raw horizontal position (trusted frames)</span><span><i class="sw" style="background:var(--acc)"></i>median-filtered</span><span><i class="sw" style="background:var(--sac)"></i>detected saccade</span><span><i class="sw" style="background:#e08a1e"></i>step score</span><span><i class="sw" style="background:var(--fg)"></i>threshold</span></div>
  <div id="explain" class="muted" style="margin-top:6px"></div>
</section>

<section class="card">
  <h2>Whole recording</h2>
  <canvas id="plot" style="height:420px"></canvas>
  <div class="legend muted"><span><i class="sw" style="background:var(--acc)"></i>horizontal</span><span><i class="sw" style="background:var(--vert)"></i>vertical</span><span><i class="sw" style="background:var(--sac)"></i>saccade (tick at onset)</span><span><i class="sw" style="background:var(--pup)"></i>pupil diameter</span><span><i class="sw" style="background:var(--muted)"></i>confidence</span><span><i class="sw" style="background:var(--gap)"></i>no trusted pupil</span><span><i class="sw" style="background:var(--mark)"></i>protocol step</span></div>
</section>

<section class="card">
  <h2>Saccades per step</h2>
  <div class="list" style="max-height:none"><table><thead><tr><th>step</th><th>seconds</th><th>saccades</th><th>per s</th><th>median size</th><th>trusted</th></tr></thead><tbody id="steps"></tbody></table></div>
</section>

<section class="card">
  <div id="gaps" class="muted" style="margin-bottom:10px"></div>
  <h2>Saccades (<span id="n"></span>)</h2>
  <div class="muted" style="margin-bottom:6px">Click a row to jump to it.</div>
  <div class="list"><table><thead><tr><th>#</th><th>step</th><th>time s</th><th>ms</th><th>size</th><th>peak/s</th></tr></thead><tbody id="rows"></tbody></table></div>
</section>
</main>
<script>
const D = __DATA__;
const v = document.getElementById('v'), seek = document.getElementById('seek'), plot = document.getElementById('plot');
const N = D.t.length, fps = D.fps, t0 = D.t[0], t1 = D.t[N - 1];
seek.max = N - 1;
const frameAt = () => Math.min(N - 1, Math.max(0, Math.floor(v.currentTime * fps + 1e-3)));
const goFrame = k => { k = Math.min(N - 1, Math.max(0, k)); v.pause(); v.currentTime = (k + 0.5) / fps; };
const idxAt = ts => { let lo = 0, hi = N - 1; while (lo < hi) { const m = (lo + hi) >> 1; if (D.t[m] < ts) lo = m + 1; else hi = m; } return lo; };
const goTime = ts => goFrame(idxAt(ts));
document.getElementById('back1').onclick = () => goFrame(frameAt() - 1);
document.getElementById('fwd1').onclick = () => goFrame(frameAt() + 1);
document.getElementById('back10').onclick = () => goFrame(frameAt() - 10);
document.getElementById('fwd10').onclick = () => goFrame(frameAt() + 10);
const playBtn = document.getElementById('play');
playBtn.onclick = () => v.paused ? v.play() : v.pause();
v.onplay = () => playBtn.textContent = 'Pause'; v.onpause = () => playBtn.textContent = 'Play';
const rate = document.getElementById('rate');
rate.onchange = e => v.playbackRate = +e.target.value;
v.addEventListener('loadedmetadata', () => { document.getElementById('loading').hidden = true; v.hidden = false; v.playbackRate = +rate.value; goFrame(0); });
// Load the whole video into memory so every frame is seekable (simple local web servers don't support
// the range requests seeking needs). Opened straight from disk, fetch is blocked and the file is used as is.
fetch('__VIDEO__').then(r => r.blob()).then(b => { v.src = URL.createObjectURL(b); }).catch(() => { v.src = '__VIDEO__'; });
seek.oninput = () => goFrame(+seek.value);
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'SELECT') return;
  if (e.key === 'ArrowRight') { goFrame(frameAt() + (e.shiftKey ? 10 : 1)); e.preventDefault(); }
  if (e.key === 'ArrowLeft') { goFrame(frameAt() - (e.shiftKey ? 10 : 1)); e.preventDefault(); }
  if (e.key === ' ') { playBtn.click(); e.preventDefault(); }
});

// protocol steps: marker label -> [start, end)
const steps = D.markers.map((m, i) => ({ label: m.label, a: m.t_s, b: i + 1 < D.markers.length ? D.markers[i + 1].t_s : t1 }));
const jump = document.getElementById('jump');
steps.forEach(s => { const o = document.createElement('option'); o.value = s.a; o.textContent = `${(s.a - t0).toFixed(1)} s  ${s.label}`; jump.appendChild(o); });
jump.onchange = () => { if (jump.value !== '') goTime(+jump.value); jump.value = ''; };
const stepAt = ts => { let r = ''; for (const s of steps) { if (s.a <= ts) r = s.label; else break; } return r; };

// saccades per step, merged by label (e.g. all span_7 trials together)
const median = a => { if (!a.length) return null; const s = [...a].sort((p, q) => p - q); return s[Math.floor(s.length / 2)]; };
const byLabel = new Map();
steps.forEach(s => {
  const e = byLabel.get(s.label) || { label: s.label, a: s.a, secs: 0, sizes: [], valid: 0, n: 0 };
  e.secs += s.b - s.a;
  D.saccades.forEach(q => { if (q.onset_s >= s.a && q.onset_s < s.b) e.sizes.push(q.amplitude); });
  for (let i = idxAt(s.a); i < N && D.t[i] < s.b; i++) { e.n++; e.valid += D.valid[i]; }
  byLabel.set(s.label, e);
});
const stepRows = document.getElementById('steps');
[...byLabel.values()].forEach(e => {
  const tr = document.createElement('tr'), m = median(e.sizes);
  tr.innerHTML = `<td>${e.label}</td><td>${e.secs.toFixed(1)}</td><td>${e.sizes.length}</td><td>${(e.sizes.length / (e.secs || 1)).toFixed(1)}</td>` +
    `<td>${m === null ? '–' : m.toFixed(1) + (D.unit === 'deg' ? '°' : ' px')}</td><td>${e.n ? Math.round(100 * e.valid / e.n) + '%' : '–'}</td>`;
  tr.onclick = () => goTime(e.a);
  stepRows.appendChild(tr);
});

const rows = document.getElementById('rows');
document.getElementById('n').textContent = D.saccades.length;
const deg = D.unit === 'deg' ? '°' : ' px';
D.saccades.forEach((s, i) => {
  const tr = document.createElement('tr');
  tr.innerHTML = `<td>${i + 1}</td><td>${s.step}</td><td>${(s.onset_s - t0).toFixed(2)}</td><td>${s.duration_ms.toFixed(0)}</td><td>${s.amplitude.toFixed(1)}${deg}</td><td>${s.peak_velocity.toFixed(0)}${deg}</td>`;
  tr.onclick = () => goTime(s.onset_s - 0.08);
  rows.appendChild(tr);
});

const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const fmt = (z, nd, u = '') => z === null || z === undefined ? '–' : z.toFixed(nd) + u;
const stats = document.getElementById('stats');
function readout(k) {
  const now = D.t[k], inS = D.saccades.findIndex(s => s.onset_s <= now && now <= s.offset_s);
  const inB = D.blinks.some(b => b.onset_s <= now && now <= b.offset_s);
  const items = [
    ['time', fmt(now - t0, 3, ' s')], ['frame', `${k + 1} / ${N}`], ['step', stepAt(now) || '–'],
    ['pupil diameter', fmt(D.d[k], 1, ' px')], ['confidence', fmt(D.conf[k], 2)],
    ['horizontal', fmt(D.x[k], 1, deg)], ['vertical', fmt(D.y[k], 1, deg)],
    ['step score', fmt(D.score[k], 2) + ` / ${D.step.threshold.toFixed(2)}`],
    ['trusted pupil', D.valid[k] ? 'yes' : 'no', !D.valid[k]],
    ['event', inS >= 0 ? `saccade #${inS + 1}` : inB ? 'blink' : '–', inS >= 0],
  ];
  if (inS >= 0) { const s = D.saccades[inS]; items.push(['saccade size', s.amplitude.toFixed(1) + deg, true], ['saccade duration', s.duration_ms.toFixed(0) + ' ms', true]); }
  stats.innerHTML = items.map(([n, val, hot]) => `<div class="stat${hot ? ' hot' : ''}"><b>${val}</b><span>${n}</span></div>`).join('');
  [...rows.children].forEach((r, i) => r.classList.toggle('now', i === inS));
}

// The whole-recording plot has tens of thousands of points, so it is drawn once into an off-screen
// canvas and only the cursor is redrawn each frame.
let bg = null, bgW = 0;
const lanes = H => { const g = 10; const h = (H - 2 * g) / 4; return { pos: [0, h * 2], pup: [h * 2 + g, h], conf: [h * 3 + 2 * g, h] }; };
function renderBg(W, H) {
  const dpr = devicePixelRatio || 1;
  bg = document.createElement('canvas'); bg.width = W * dpr; bg.height = H * dpr; bgW = W;
  const c = bg.getContext('2d'); c.setTransform(dpr, 0, 0, dpr, 0, 0);
  const X = ts => (ts - t0) / (t1 - t0) * W, L = lanes(H);
  c.font = '11px system-ui';
  const line = (arr, col, [top, h], lo, hi) => {
    const Y = z => top + 4 + (1 - (z - lo) / (hi - lo || 1)) * (h - 8);
    // one min/max pair per pixel column keeps the drawing fast and every spike visible
    c.strokeStyle = col; c.lineWidth = 1; c.beginPath();
    let col0 = -1, mn = null, mx = null;
    const flush = px => { if (mn !== null) { c.moveTo(px + 0.5, Y(mx)); c.lineTo(px + 0.5, Y(mn) + 0.5); } };
    for (let i = 0; i < N; i++) {
      const px = Math.floor(X(D.t[i])), z = arr[i];
      if (px !== col0) { flush(col0); col0 = px; mn = mx = null; }
      if (z === null) continue;
      mn = mn === null ? z : Math.min(mn, z); mx = mx === null ? z : Math.max(mx, z);
    }
    flush(col0); c.stroke();
  };
  const range = arrs => { const vals = arrs.flat().filter(z => z !== null).sort((p, q) => p - q);
    return vals.length ? [vals[Math.floor(vals.length * 0.005)], vals[Math.floor(vals.length * 0.995)]] : [0, 1]; };
  // steps
  c.strokeStyle = css('--mark'); c.globalAlpha = 0.35; c.beginPath();
  steps.forEach(s => { c.moveTo(Math.round(X(s.a)) + 0.5, 0); c.lineTo(Math.round(X(s.a)) + 0.5, H); }); c.stroke(); c.globalAlpha = 1;
  // gaps in tracking under the position lane
  c.fillStyle = css('--gap');
  for (let i = 0; i < N; i++) if (!D.valid[i]) c.fillRect(X(D.t[i]), L.pos[0] + L.pos[1] - 5, Math.max(1, W / N), 5);
  const [plo, phi] = range([D.x, D.y]);
  line(D.x, css('--acc'), L.pos, plo, phi); line(D.y, css('--vert'), L.pos, plo, phi);
  c.fillStyle = css('--sac'); D.saccades.forEach(s => c.fillRect(Math.floor(X(s.onset_s)), L.pos[0], 1, 10));
  const [dlo, dhi] = range([D.d]); line(D.d, css('--pup'), L.pup, dlo, dhi);
  line(D.conf, css('--muted'), L.conf, 0, 1);
  c.fillStyle = css('--muted');
  c.fillText(`position ${phi.toFixed(0)}…${plo.toFixed(0)} ${D.unit}`, 4, L.pos[0] + 22);
  c.fillText(`pupil ${dlo.toFixed(0)}…${dhi.toFixed(0)} px`, 4, L.pup[0] + 12);
  c.fillText('confidence 0…1', 4, L.conf[0] + 12);
  // step names, skipping any that would overlap the previous one
  c.fillStyle = css('--mark'); let lastX = -1e9;
  steps.forEach(s => { const px = X(s.a); if (px - lastX > 60) { c.fillText(s.label, px + 3, H - 2); lastX = px; } });
}
function draw() {
  const k = frameAt(), now = D.t[k];
  seek.value = k;
  readout(k);
  const dpr = devicePixelRatio || 1, W = plot.clientWidth, H = plot.clientHeight;
  if (!W || !H) return;    // not laid out yet
  if (!bg || bgW !== W) renderBg(W, H);
  plot.width = W * dpr; plot.height = H * dpr;
  const c = plot.getContext('2d'); c.drawImage(bg, 0, 0); c.setTransform(dpr, 0, 0, dpr, 0, 0);
  const x = (now - t0) / (t1 - t0) * W;
  c.strokeStyle = css('--fg'); c.lineWidth = 1; c.beginPath(); c.moveTo(x, 0); c.lineTo(x, H); c.stroke();
  drawLive(k);
}
let lastK = -1, lastW = 0;
function tick() {
  requestAnimationFrame(tick);
  const k = frameAt();
  if (k !== lastK || plot.clientWidth !== lastW) { lastK = k; lastW = plot.clientWidth; draw(); }
}

const live = document.getElementById('live'), S = D.step;
document.getElementById('explain').innerHTML =
  `A step counts as a saccade when the score crosses the threshold line: <b>${S.threshold.toFixed(2)} ${D.unit}</b> = the larger of ` +
  `the minimum size (${S.min_amplitude} ${D.unit}) and ${S.noise_factor} × the noise of a ${S.half_window_ms} ms median ` +
  `(per-frame noise ${S.sample_sd.toFixed(2)} ${D.unit} ÷ √${S.samples_per_window.toFixed(0)} frames × 1.25 = ${S.median_noise.toFixed(2)} ${D.unit}). ` +
  `Two peaks closer than ${S.min_separation_ms} ms keep only the larger. The saccade list itself comes from the detector set in the config.`;
const g = D.gap_ms, near = g.filter(v => v < S.min_separation_ms + 30).length;
document.getElementById('gaps').innerHTML = g.length ? `Time between saccade onsets: median <b>${median(g).toFixed(0)} ms</b>; ` +
  `${near} of ${g.length} are within 30 ms of the ${S.min_separation_ms} ms limit (a pile-up there would mean the limit is merging real saccades).` : '';
function drawLive(k) {
  const dpr = devicePixelRatio || 1, W = live.clientWidth, H = live.clientHeight;
  live.width = W * dpr; live.height = H * dpr;
  const c = live.getContext('2d'); c.setTransform(dpr, 0, 0, dpr, 0, 0); c.clearRect(0, 0, W, H);
  const now = D.t[k], a = now - 1.5, b = now + 1.5, X = ts => (ts - a) / (b - a) * W;
  const i0 = idxAt(a), i1 = Math.min(N - 1, idxAt(b));
  const top = H * 0.62, xs = [];
  for (let i = i0; i <= i1; i++) { if (D.x[i] !== null) xs.push(D.x[i]); }
  const lo = xs.length ? Math.min(...xs) - 1 : 0, hi = xs.length ? Math.max(...xs) + 1 : 1, Y = z => 6 + (1 - (z - lo) / (hi - lo || 1)) * (top - 12);
  c.fillStyle = css('--sac'); c.globalAlpha = 0.28;
  D.saccades.forEach(s => { if (s.offset_s > a && s.onset_s < b) c.fillRect(X(s.onset_s), 0, Math.max(3, X(s.offset_s) - X(s.onset_s)), H); });
  c.globalAlpha = 1; c.fillStyle = '#888';
  for (let i = i0; i <= i1; i++) if (D.x[i] !== null) c.fillRect(X(D.t[i]) - 1.5, Y(D.x[i]) - 1.5, 3, 3);
  c.strokeStyle = css('--acc'); c.lineWidth = 2; c.beginPath(); let pen = false;
  for (let i = i0; i <= i1; i++) { const z = D.xm[i]; if (z === null) { pen = false; continue; } pen ? c.lineTo(X(D.t[i]), Y(z)) : c.moveTo(X(D.t[i]), Y(z)); pen = true; }
  c.stroke();
  c.strokeStyle = css('--line'); c.beginPath(); c.moveTo(0, top); c.lineTo(W, top); c.stroke();
  const smax = Math.max(S.threshold * 2.5, ...D.score.slice(i0, i1 + 1).filter(v => v !== null));
  const SY = v => H - 4 - v / smax * (H - top - 10);
  c.strokeStyle = '#e08a1e'; c.lineWidth = 1.6; c.beginPath(); pen = false;
  for (let i = i0; i <= i1; i++) { const v = D.score[i]; if (v === null) { pen = false; continue; } pen ? c.lineTo(X(D.t[i]), SY(v)) : c.moveTo(X(D.t[i]), SY(v)); pen = true; }
  c.stroke();
  c.setLineDash([5, 4]); c.strokeStyle = css('--fg'); c.lineWidth = 1; c.beginPath(); c.moveTo(0, SY(S.threshold)); c.lineTo(W, SY(S.threshold)); c.stroke(); c.setLineDash([]);
  c.strokeStyle = css('--fg'); c.beginPath(); c.moveTo(X(now), 0); c.lineTo(X(now), H); c.stroke();
  c.fillStyle = css('--muted'); c.font = '11px system-ui';
  c.fillText(`position (${D.unit})`, 4, 14); c.fillText('step score', 4, top + 14); c.fillText(`threshold ${S.threshold.toFixed(2)}`, W - 100, SY(S.threshold) - 4);
}
plot.onclick = e => { const r = plot.getBoundingClientRect(); goTime(t0 + (e.clientX - r.left) / r.width * (t1 - t0)); };
live.onclick = e => { const r = live.getBoundingClientRect(); goTime(D.t[frameAt()] - 1.5 + (e.clientX - r.left) / r.width * 3); };
requestAnimationFrame(tick);
</script></body></html>
"""
