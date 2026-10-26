# Real-time processing

**Current choice:** `src/eyetrack/realtime/`.
- A worker thread runs pupil detection on **every** eye frame, in order, decoupled from the preview window.
- It computes causal (past-frames-only) validity, blink and saccade flags.
- It saves `live_pupil.csv`, `live_blinks.csv` and `live_summary.json` next to the raw video, so
  `eyetrack process` can still re-analyse everything offline.
- `eyetrack realtime-bench` replays a recording through the same path to measure it.

User docs: [realtime.md](../realtime.md).

## Iterations

### 1. Detection inside the preview loop (2026-10-24)
- The live and record previews ran the detector on the **latest** frame each time the window redrew (~50 Hz).
  Frames in between were skipped.
- Good enough for a live pupil-size trace, but not for live saccades or blinks.

### 2. Is real-time feasible? (2026-10-26)
At 120 fps each frame has 8.3 ms. Measured on a MacBook Air M3:

| Step | Time per frame |
|---|---|
| Decoding | 0.45 ms |
| Pupil detection | 1.2 ms |
| Blink and I-VT stages | microseconds |

So the limit was the program's structure, not compute.

### 3. Worker plus causal metrics
- The camera thread hands frames to a bounded queue (240 frames, 2 s) without ever waiting. If the queue fills,
  the frame is skipped by the live path only and counted; the recording keeps it.
- **Thread vs process:** both keep up. The process backend had fewer delay spikes (4 vs 24 frames over 25 ms in
  12,000), but takes ~1 s to start. Thread stays the default.
- Python's garbage collector was ruled out as the cause of the spikes (pauses under 3 ms). Probably the operating
  system's scheduling.
- **Replay of the full 49,412-frame session:** 0 dropped frames, delay 2.5 ms typical and 4.3 ms for the slowest 5%,
  about 30% of one CPU core, and about 4.5× headroom. A separate 30 s check gave 120.6 fps, 0 dropped, 3 frames late.

### 4. Live vs offline agreement
- Pupil centres are identical, and validity agrees on 99%+ of frames.
- Blinks: the same events. Each is confirmed ~80 ms after it ends.
- **Saccades: unreliable live.** 84% agreement over the whole session, but only 14% on a noisy reading segment,
  where offline used the step method. Treat the live flag as a monitor.

## Known limits
- **Camera delivery is the real bottleneck:** the first session averaged 107 fps, with 1,230 gaps over 20 ms.
- **Delay excludes USB and decoding:** it's measured from when the decoded frame reaches the program.
- **Live angles need a saved eye model** (`--eye-model`), and the current model is wrong (see
  [eye-model-and-gaze.md](eye-model-and-gaze.md)). Without it, live output is in pixels.
- **Interrupted recordings:** a recording cut off part-way leaves a timestamp file longer than the video, and
  offline processing then fails (a pre-existing recorder bug).
- **Not yet tested** with the live camera, or with the preview window's display cost.
