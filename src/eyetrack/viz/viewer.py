"""Frame-by-frame viewer for one part of a processed session: `eyetrack viewer <session> --part reading`.

Writes, in the session folder:
    viewer_<part>.mp4    the eye video for that part, H.264 at the recorded frame rate, with the detected
                         pupil (yellow = trusted, red = rejected), time, frame number, and a SACCADE / BLINK
                         banner on the frames inside one
    viewer_<part>.html   the video with frame stepping (arrow keys), slow motion, an eye-position timeline
                         with saccades and tracking gaps marked (click to jump), and a clickable saccade list
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PART_LABELS = {"baseline": ["baseline"], "blinks": ["blink_cue", "blink_wait"], "light": ["lights_on", "lights_off"],
               "reading": ["reading"], "arithmetic": ["arith_fix", "task_easy", "task_medium", "task_hard", "answer"]}


def part_window(markers: pd.DataFrame, part: str, end_s: float) -> tuple[float, float]:
    """First and last moment of a protocol part, from its markers."""
    m = markers.sort_values("t_s").reset_index(drop=True)
    m["end"] = list(m.t_s[1:]) + [end_s]
    sel = m[m.label.isin(PART_LABELS[part])]
    if not len(sel):
        raise ValueError(f"no '{part}' markers in this session")
    return float(sel.t_s.min()), float(sel.end.max())


def make_viewer(session: str | Path, part: str = "reading", pad_s: float = 1.0, fps: float = 120.0,
                cfg: dict | None = None) -> Path:
    session = Path(session)
    pupil = pd.read_csv(session / "pupil.csv")
    markers = pd.read_csv(session / "markers_s.csv")
    sacc = pd.read_csv(session / "saccades.csv")
    blinks = pd.read_csv(session / "blinks.csv")
    t = pupil.t_s.to_numpy()
    a, b = part_window(markers, part, float(t[-1]))
    a, b = max(0.0, a - pad_s), min(float(t[-1]), b + pad_s)
    i0, i1 = int(np.searchsorted(t, a)), int(np.searchsorted(t, b))
    win = pupil.iloc[i0:i1].reset_index(drop=True)
    sw = sacc[(sacc.onset_s >= a) & (sacc.onset_s < b)].reset_index(drop=True)
    bw = blinks[(blinks.onset_s < b) & (blinks.offset_s > a)] if len(blinks) else blinks

    def inside(tt, df):
        return bool(len(df)) and bool(np.any((df.onset_s.to_numpy() <= tt) & (df.offset_s.to_numpy() >= tt)))

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
        info = f"t {row.t_s:7.3f}s  frame {int(row.frame_index)}  conf {row.confidence:.2f}"
        for txt, y in ((info, 24),):
            cv2.putText(img, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        real_blinks = bw[bw.kind == "blink"] if len(bw) else bw
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
    data = {
        "part": part, "fps": fps, "unit": "deg" if "yaw_deg" in win else "px",
        "t": win.t_s.round(4).tolist(),
        "x": [None if not v or not np.isfinite(xx) else round(float(xx), 3) for xx, v in zip(x, win.valid)],
        "y": [None if not v or not np.isfinite(yy) else round(float(yy), 3) for yy, v in zip(y, win.valid)],
        "valid": win.valid.astype(int).tolist(),
        "xm": [None if not np.isfinite(v) else round(float(v), 3) for v in xm],
        "score": [None if not np.isfinite(v) else round(float(v), 3) for v in sig["step"]],
        "step": {"threshold": round(sig["threshold"], 3), "sample_sd": round(sig["sample_sd"], 3),
                 "median_noise": round(sig["median_noise"], 3), "samples_per_window": round(sig["samples_per_window"], 1),
                 "half_window_ms": half, "noise_factor": nf, "min_amplitude": amp,
                 "min_separation_ms": st.get("min_separation_ms", 120.0)},
        "gap_ms": [round(float(g), 1) for g in gaps_ms],
        "saccades": sw[["onset_s", "offset_s", "duration_ms", "amplitude", "peak_velocity"]].round(3).to_dict("records"),
        "blinks": bw[bw.kind == "blink"][["onset_s", "offset_s"]].round(3).to_dict("records") if len(bw) else [],
        "markers": markers[(markers.t_s >= a) & (markers.t_s < b)].round(3).to_dict("records"),
    }
    html = HTML.replace("__DATA__", json.dumps(data)).replace("__VIDEO__", video_path.name).replace("__PART__", part)
    out = session / f"viewer_{part}.html"
    out.write_text(html)
    return out


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Eye viewer: __PART__</title>
<style>
  :root { --bg:#f6f6f4; --fg:#1d1d1f; --muted:#6b6b70; --card:#fff; --line:#e3e3e0; --sac:#d6453d; --gap:#c9c9c4; --acc:#2f6fb0; }
  @media (prefers-color-scheme: dark) { :root { --bg:#161617; --fg:#ececec; --muted:#9a9aa0; --card:#212123; --line:#333336; --gap:#4a4a4e; } }
  body { margin:0; background:var(--bg); color:var(--fg); font:14px/1.45 -apple-system, system-ui, sans-serif; }
  main { max-width:1180px; margin:0 auto; padding:16px; display:grid; grid-template-columns: minmax(0,1fr) 330px; gap:16px; }
  @media (max-width: 900px) { main { grid-template-columns: 1fr; } }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px; }
  h1 { font-size:18px; margin:0 0 4px; } .muted { color:var(--muted); }
  video { width:100%; border-radius:6px; background:#000; display:block; }
  .controls { display:flex; flex-wrap:wrap; gap:6px; align-items:center; margin-top:8px; }
  button, select { font:inherit; padding:5px 10px; border-radius:6px; border:1px solid var(--line); background:var(--bg); color:var(--fg); cursor:pointer; }
  button:hover { border-color:var(--acc); }
  input[type=range] { width:100%; margin-top:8px; }
  #readout { font-variant-numeric: tabular-nums; }
  canvas { width:100%; height:190px; display:block; margin-top:10px; cursor:crosshair; }
  table { width:100%; border-collapse:collapse; font-variant-numeric: tabular-nums; font-size:13px; }
  th, td { text-align:right; padding:4px 6px; border-bottom:1px solid var(--line); } th:first-child, td:first-child { text-align:left; }
  tbody tr { cursor:pointer; } tbody tr:hover { background:color-mix(in srgb, var(--acc) 12%, transparent); }
  tr.now { background:color-mix(in srgb, var(--sac) 18%, transparent); }
  .list { max-height:640px; overflow:auto; }
  .legend span { display:inline-block; margin-right:12px; } .sw { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:4px; vertical-align:-1px; }
  kbd { border:1px solid var(--line); border-radius:4px; padding:0 4px; font-size:12px; }
</style></head>
<body><main>
<section>
  <div class="card">
    <h1>Eye video: __PART__</h1>
    <div class="muted">Yellow ellipse = trusted pupil, red = rejected. <kbd>←</kbd>/<kbd>→</kbd> one frame, <kbd>Shift</kbd>+arrow 10 frames, <kbd>Space</kbd> play/pause.</div>
    <video id="v" preload="auto" muted playsinline></video>
    <input id="seek" type="range" min="0" max="1" step="1" value="0">
    <div class="controls">
      <button id="back10">« 10</button><button id="back1">‹ 1</button><button id="play">Play</button>
      <button id="fwd1">1 ›</button><button id="fwd10">10 »</button>
      <select id="rate"><option value="0.1">0.1×</option><option value="0.25" selected>0.25×</option><option value="0.5">0.5×</option><option value="1">1×</option></select>
      <span id="readout" class="muted"></span>
    </div>
    <h1 style="margin-top:14px">Live: 3 s around the current frame</h1>
    <canvas id="live" style="height:300px"></canvas>
    <div class="legend muted"><span><i class="sw" style="background:#888"></i>raw position (trusted frames)</span><span><i class="sw" style="background:var(--acc)"></i>median-filtered</span><span><i class="sw" style="background:var(--sac)"></i>detected saccade</span><span><i class="sw" style="background:#e08a1e"></i>step score</span><span><i class="sw" style="background:var(--fg)"></i>threshold</span></div>
    <div id="explain" class="muted" style="margin-top:6px"></div>
    <h1 style="margin-top:14px">Whole section</h1>
    <canvas id="plot"></canvas>
    <div class="legend muted"><span><i class="sw" style="background:var(--acc)"></i>horizontal</span><span><i class="sw" style="background:#9a7b2f"></i>vertical</span><span><i class="sw" style="background:var(--sac)"></i>saccade (tick at onset)</span><span><i class="sw" style="background:var(--gap)"></i>no trusted pupil</span></div>
  </div>
</section>
<aside class="card">
  <div id="gaps" class="muted" style="margin-bottom:10px"></div>
  <h1>Saccades (<span id="n"></span>)</h1>
  <div class="muted" style="margin-bottom:6px">Click a row to jump to it.</div>
  <div class="list"><table><thead><tr><th>#</th><th>time s</th><th>ms</th><th>size</th><th>peak/s</th></tr></thead><tbody id="rows"></tbody></table></div>
</aside>
</main>
<script>
const D = __DATA__;
const v = document.getElementById('v'), seek = document.getElementById('seek'), plot = document.getElementById('plot');
const N = D.t.length, fps = D.fps, t0 = D.t[0], t1 = D.t[N - 1];
seek.max = N - 1;
const frameAt = () => Math.min(N - 1, Math.max(0, Math.floor(v.currentTime * fps + 1e-3)));
const goFrame = k => { k = Math.min(N - 1, Math.max(0, k)); v.pause(); v.currentTime = (k + 0.5) / fps; };
const goTime = ts => { let k = 0; while (k < N - 1 && D.t[k] < ts) k++; goFrame(k); };
document.getElementById('back1').onclick = () => goFrame(frameAt() - 1);
document.getElementById('fwd1').onclick = () => goFrame(frameAt() + 1);
document.getElementById('back10').onclick = () => goFrame(frameAt() - 10);
document.getElementById('fwd10').onclick = () => goFrame(frameAt() + 10);
const playBtn = document.getElementById('play');
playBtn.onclick = () => v.paused ? v.play() : v.pause();
v.onplay = () => playBtn.textContent = 'Pause'; v.onpause = () => playBtn.textContent = 'Play';
document.getElementById('rate').onchange = e => v.playbackRate = +e.target.value;
v.addEventListener('loadedmetadata', () => { v.playbackRate = 0.25; goFrame(0); });
// Load the whole clip into memory so every frame is seekable (simple local web servers don't support
// the range requests seeking needs). Opened straight from disk, fetch is blocked and the file is used as is.
fetch('__VIDEO__').then(r => r.blob()).then(b => { v.src = URL.createObjectURL(b); }).catch(() => { v.src = '__VIDEO__'; });
seek.oninput = () => goFrame(+seek.value);
document.addEventListener('keydown', e => {
  if (e.key === 'ArrowRight') { goFrame(frameAt() + (e.shiftKey ? 10 : 1)); e.preventDefault(); }
  if (e.key === 'ArrowLeft') { goFrame(frameAt() - (e.shiftKey ? 10 : 1)); e.preventDefault(); }
  if (e.key === ' ') { playBtn.click(); e.preventDefault(); }
});
const rows = document.getElementById('rows');
document.getElementById('n').textContent = D.saccades.length;
D.saccades.forEach((s, i) => {
  const tr = document.createElement('tr');
  tr.innerHTML = `<td>${i + 1}</td><td>${(s.onset_s - t0).toFixed(2)}</td><td>${s.duration_ms.toFixed(0)}</td><td>${s.amplitude.toFixed(1)}°</td><td>${s.peak_velocity.toFixed(0)}°</td>`;
  tr.onclick = () => goTime(s.onset_s - 0.08);
  rows.appendChild(tr);
});
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
function draw() {
  const k = frameAt(), now = D.t[k];
  seek.value = k;
  const inS = D.saccades.findIndex(s => s.onset_s <= now && now <= s.offset_s);
  document.getElementById('readout').textContent = `frame ${k + 1}/${N}   t = ${(now - t0).toFixed(3)} s` + (D.valid[k] ? '' : '   (no trusted pupil)') + (inS >= 0 ? `   SACCADE #${inS + 1}` : '');
  [...rows.children].forEach((r, i) => r.classList.toggle('now', i === inS));
  const dpr = devicePixelRatio || 1, W = plot.clientWidth, H = plot.clientHeight;
  plot.width = W * dpr; plot.height = H * dpr;
  const c = plot.getContext('2d'); c.setTransform(dpr, 0, 0, dpr, 0, 0); c.clearRect(0, 0, W, H);
  const X = ts => (ts - t0) / (t1 - t0) * W;
  c.fillStyle = css('--gap');
  for (let i = 0; i < N; i++) if (!D.valid[i]) c.fillRect(X(D.t[i]), H - 8, Math.max(1, W / N), 8);
  // saccades as short ticks along the top: at this zoom one pixel is tens of ms, so full-height bands
  // would paint most of the plot red even though saccades are a small fraction of the time
  c.fillStyle = css('--sac');
  D.saccades.forEach(s => c.fillRect(X(s.onset_s), 0, 1, 9));
  const vals = D.x.concat(D.y).filter(z => z !== null); const lo = Math.min(...vals), hi = Math.max(...vals);
  const Y = z => 14 + (1 - (z - lo) / (hi - lo || 1)) * (H - 32);
  [[D.x, css('--acc')], [D.y, '#9a7b2f']].forEach(([arr, col]) => {
    c.strokeStyle = col; c.lineWidth = 1.2; c.beginPath(); let pen = false;
    for (let i = 0; i < N; i++) { const z = arr[i]; if (z === null) { pen = false; continue; }
      if (pen) c.lineTo(X(D.t[i]), Y(z)); else c.moveTo(X(D.t[i]), Y(z)); pen = true; }
    c.stroke();
  });
  c.strokeStyle = css('--fg'); c.lineWidth = 1; c.beginPath(); c.moveTo(X(now), 0); c.lineTo(X(now), H); c.stroke();
  c.fillStyle = css('--muted'); c.font = '11px system-ui';
  c.fillText(`${hi.toFixed(0)} ${D.unit}`, 4, 14); c.fillText(`${lo.toFixed(0)} ${D.unit}`, 4, H - 14);
  drawLive(k);
  requestAnimationFrame(draw);
}
const live = document.getElementById('live'), S = D.step;
document.getElementById('explain').innerHTML =
  `A step counts as a saccade when the score crosses the threshold line: <b>${S.threshold.toFixed(2)} ${D.unit}</b> = the larger of ` +
  `the minimum size (${S.min_amplitude} ${D.unit}) and ${S.noise_factor} × the noise of a ${S.half_window_ms} ms median ` +
  `(per-frame noise ${S.sample_sd.toFixed(2)} ${D.unit} ÷ √${S.samples_per_window.toFixed(0)} frames × 1.25 = ${S.median_noise.toFixed(2)} ${D.unit}). ` +
  `Two peaks closer than ${S.min_separation_ms} ms keep only the larger.`;
const g = D.gap_ms, near = g.filter(v => v < S.min_separation_ms + 30).length;
document.getElementById('gaps').innerHTML = g.length ? `Time between saccade onsets: median <b>${[...g].sort((a, b) => a - b)[Math.floor(g.length / 2)].toFixed(0)} ms</b>; ` +
  `${near} of ${g.length} are within 30 ms of the ${S.min_separation_ms} ms limit (a pile-up there would mean the limit is merging real saccades).` : '';
function drawLive(k) {
  const dpr = devicePixelRatio || 1, W = live.clientWidth, H = live.clientHeight;
  live.width = W * dpr; live.height = H * dpr;
  const c = live.getContext('2d'); c.setTransform(dpr, 0, 0, dpr, 0, 0); c.clearRect(0, 0, W, H);
  const now = D.t[k], a = now - 1.5, b = now + 1.5, X = ts => (ts - a) / (b - a) * W;
  let i0 = k, i1 = k; while (i0 > 0 && D.t[i0] > a) i0--; while (i1 < N - 1 && D.t[i1] < b) i1++;
  const top = H * 0.62, xs = [];
  for (let i = i0; i <= i1; i++) { if (D.x[i] !== null) xs.push(D.x[i]); }
  const lo = Math.min(...xs) - 1, hi = Math.max(...xs) + 1, Y = z => 6 + (1 - (z - lo) / (hi - lo || 1)) * (top - 12);
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
requestAnimationFrame(draw);
</script></body></html>
"""
