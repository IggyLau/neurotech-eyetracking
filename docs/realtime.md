# Real-time processing (120 fps)

`eyetrack record` and `eyetrack live` run pupil detection and causal per-frame metrics on **every** eye
frame as it arrives, on a worker separate from the preview window. The preview redraws at about 50 Hz
and shows the worker's newest result. `eyetrack record` also saves the live results next to the raw
video. Any session can still be re-analysed with the full, non-causal offline methods
(`eyetrack process`), which read only `eye.avi` / `eye_ts.csv` / `markers.csv`.

```
camera grab thread ──► eye.avi + eye_ts.csv          (every frame, as before)
        │ submit (never blocks)
        ▼
bounded queue (240 frames) ──► worker: detect_pupil + causal metrics ──► live_pupil.csv, live_blinks.csv
                                         │
                                         └──► newest result ──► preview window (~50 Hz)
```

If the worker falls behind and the queue fills, frames are skipped by the live path only. They are
counted as `dropped`, and the recording still has them. Settings are in the `realtime:` section of
`config/default.yaml`. Use `--no-realtime` to turn the worker off, and `--backend process` to run it
in a separate process.

## Files written by `eyetrack record`

| File | Contents |
|---|---|
| `live_pupil.csv` | Per frame: `frame_index` (matches `eye_ts.csv`), `t_ns`, ellipse (`cx`, `cy`, axes, `angle`), `diameter_px`, `confidence`, `frame_brightness`, `eye_brightness`, `size_outlier`, `valid`, `blink_state`, `yaw_deg`/`pitch_deg` (only with an eye model), `speed`, `saccade_flag`, `queue_ms`, `proc_ms`, `latency_ms` |
| `live_blinks.csv` | Blink / dropout / lighting / camera-gap events, with `known_at_s`: the time the live path knew the final kind |
| `live_summary.json` | Frames processed and dropped, processing fps, latency percentiles, worker CPU (also copied into `meta.json` under `realtime`) |
| `tracker_status.csv` | `t_ns`, `level`, `text` each time the on-screen tracker indicator changes |

`live_pupil.csv` is flushed every 60 rows (about 0.5 s), and `live_blinks.csv` after every event, so a
crash loses less than a second of results.

## Tracker indicator

While recording, a small dot and label in the bottom-right corner of the screen (the protocol screen in
`eyetrack validate`, and the eye tile in `eyetrack record`) say whether tracking is still working, judged
over the last 2 s (`realtime/status.py`):

| Indicator | Meaning |
|---|---|
| green `TRACKING 95%` | at least 80% of frames valid, pupil inside the central 80% of the image, frame rate at least 80% of the set rate |
| amber `WEAK`, `PUPIL NEAR EDGE`, `LOW FPS` | still tracking, but one of those checks fails: re-aim the camera, or check the cable |
| red `PUPIL LOST`, `CAMERA DARK`, `NO FRAMES`, `VIDEO NOT SAVING` | under 30% valid, a black image, no frames for 0.5 s, or frames that no longer match the video file's size (the camera reconnected in another mode): stop and fix the setup |

Changes are also printed in the terminal. The indicator is small so it barely changes screen brightness, but it
does change colour; it needs the real-time worker (`realtime.enabled`).

## Live metrics and their latency

| Metric | How (causal) | Latency |
|---|---|---|
| Pupil ellipse, confidence | `detect_pupil`, the same code as offline (identical results) | processing only (below) |
| `valid` | confidence ≥ `min_confidence` and diameter within `max_size_ratio` of the median of the **past** 2 s (offline uses a centred window) | 0 frames; no outlier check for the first ~0.2 s |
| `blink_state` | per-frame state of the current dropout, from the same brightness rule as offline `classify_dropouts` | 0 frames, but provisional: the first 30 ms (`min_duration_ms`) of every blink reads `dropout` |
| Blink events (`live_blinks.csv`) | the same rule, including merging of eyelid flickers | the final kind is known on the first valid frame after the event, plus up to `merge_gap_ms` (median 64-80 ms after the event ends) |
| `saccade_flag` | I-VT on the central-difference speed of frame n-1 (the speed offline computes), at the configured threshold, plus a causal version of the amplitude-vs-noise check (noise = RMS sample-to-sample over the last 2 s) | 1 frame (8.3 ms); median 1 frame after the offline onset |
| `yaw_deg`, `pitch_deg` | only with `--eye-model <session>` (an eye model fitted by an earlier `eyetrack process` of the same camera placement). Without it, positions and `speed` are eye-image pixels | 0 frames |

Live results compared with offline `eyetrack process` on the same recording (49,412 frames, 120 fps,
eye model from that session):

- Pupil centres are identical (largest difference 0.0001 px). `valid` agrees on 99.0% of frames.
- Blink events: live 112 blinks / 15 closed / 725 lighting / 1238 camera gaps; offline 106 / 13 / 753 / 1230.
  The differences come from the past-only size check.
- Saccades: 84% of offline saccades are flagged live. 29% of live flag runs have no offline saccade,
  because offline also applies minimum durations and a whole-session noise estimate. Treat the live
  flag as a monitor and the offline `saccades.csv` as the measurement.

## Measured throughput (MacBook Air M3, 640×400 MJPG)

Run with `eyetrack realtime-bench data/<session>`, which replays the recorded video with its original
frame timing through the same `record()` code path (including re-encoding eye.avi):

| Run | Frames | Processing fps | Dropped | Latency p50 / p95 / p99 | Over 25 ms | Worker core use |
|---|---|---|---|---|---|---|
| thread, real time, whole session | 49,412 | 107.2 (= source rate) | 0 | 2.5 / 4.3 / 6.9 ms | 25 (0.05%) | 28% |
| thread, real time | 12,000 | 120.6 | 0 | 2.6 / 4.0 / 9.8 ms | 24 (0.2%) | 32% |
| process, real time | 12,000 | 120.6 | 0 | 2.6 / 3.6 / 6.6 ms | 4 | 33% |
| thread, as fast as possible | 6,000 | 562 | 0 | 1.4 / 1.7 ms | 0 | 79% |

- Latency runs from the grab thread's timestamp (taken when `read()` returns the decoded frame) to the
  result being ready. USB transfer and AVFoundation's decode happen before that timestamp and are not
  included.
- Detection takes about 1.4 ms per frame when the worker is busy, but about 2.6 ms at 120 fps. That is
  probably macOS running a mostly idle thread on efficiency cores at low clock, which is fine for
  latency.
- The worker can run about 4.5× faster than the camera, so it uses about a third of one core at 120 fps.
- The thread backend keeps up. Its rare 20-70 ms spikes occur inside detection itself; garbage
  collection pauses were measured under 3 ms, so OS scheduling is the likely cause. The process
  backend has fewer spikes but costs a spawn at start (about 1 s) and a copy into shared memory.
- The replayed session itself averages 107 fps, with 1230 camera gaps over 20 ms. Frame delivery from
  USB / AVFoundation, not processing, is what limits the effective rate.

## Commands

```bash
eyetrack record --name task1                     # records + live processing (default on)
eyetrack record --name task1 --eye-model data/<earlier-session>   # live yaw/pitch in degrees
eyetrack live                                    # preview only; every frame processed, nothing saved
eyetrack live --replay data/<session>            # same preview on a recorded session
eyetrack realtime-bench data/<session>           # real-time replay: fps, dropped, latency (+ agreement with pupil.csv)
eyetrack realtime-bench data/<session> --fast    # as fast as possible: maximum throughput
eyetrack realtime-bench data/<session> --backend process --max-frames 12000
```

`realtime-bench` writes a new session folder (`<output_dir>/<stamp>_realtime_bench`, or under `--out`)
containing `realtime_bench.json`. In replays, the `t_ns` column of `live_pupil.csv` holds the source's
original capture times, so blink durations and velocities are correct at any replay speed.
