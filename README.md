# neurotech-eyetracking

Recording and analysis toolkit for the Neurotech@USC DIY eye-tracking glasses:
**OV9281** (mono, global shutter, no IR filter) as the eye camera and **GC0308** as the
forward scene camera.

The approach is **record raw, process offline**. `record` saves synced eye and scene video
with per-frame timestamps, and `process` computes all the metrics afterwards. Because the raw
video is kept, you can tune the detector or try other methods without re-recording anyone.

Pupil detection is adapted from [JEOresearch/EyeTracker](https://github.com/JEOresearch/EyeTracker)
(MIT, see `NOTICE`).

## Setup (macOS)

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest -q
```

The first time a camera is opened, macOS asks for camera access for your terminal app.
If no cameras are found, allow it under **System Settings → Privacy & Security → Camera**.

## Workflow

| Step | Command | What it's for |
|---|---|---|
| 1 | `eyetrack probe` then `eyetrack probe --index N` | Find camera indices and measure the **real** fps and dropped frames per mode |
| 2 | `eyetrack live` | Live pupil overlay for aiming the camera and tuning the LEDs (aim for confidence > 0.8) |
| 3 | `eyetrack calibrate` | Full-screen ArUco targets: 9 calibration + 5 validation points; saves `calibration.json` |
| 4 | `eyetrack record --name lightsoff` | Record a test. Keys: `1` lights off, `2` lights on, `3`/`4` task start/end, `b` blink cue, `m` marker, `q` stop |
| 5 | `eyetrack process data/<session> --calibration data/<cal>/calibration.json --overlay` | Compute all metrics |

Put your own settings (camera indices, best fps mode, eye ROI) in `config/local.yaml`
and pass `--config config/local.yaml` before the subcommand. Only the keys you change are needed.

`eyetrack process` also accepts a bare eye video, e.g. `eyetrack process data/samples/eye_test.mp4 --annotate`.

## What gets measured

| Measurement | Output | How | Limits with this hardware |
|---|---|---|---|
| Pupil center | `pupil.csv` (`cx, cy, confidence`) | Dark-pupil threshold + ellipse fit | Needs off-axis IR (no red-eye); eyelashes/glints lower confidence |
| Pupil diameter | `pupil.csv` (`diameter_px`, `diameter_clean_px`) | Ellipse major axis, blink removal, interpolation, smoothing | Pixels, not mm; only compare within a recording, with the glasses unmoved and lighting constant |
| Eye rotation | `pupil.csv` (`yaw_deg, pitch_deg`) | Eyeball sphere fitted from ellipse shapes (needs ~10 s of looking around) | Re-fit whenever the glasses shift |
| Blinks | `blinks.csv` | 50–500 ms runs of lost/low-confidence pupil | Partial blinks may be missed |
| Saccades | `saccades.csv` (amplitude, duration, peak velocity) | I-VT velocity threshold (30 °/s) | At ~120 Hz: reliable counts and amplitudes, underestimated peak velocity, no microsaccades |
| Fixations | `fixations.csv` | Gaps between saccades ≥ 60 ms | |
| Gaze on scene | `gaze.csv`, `overlay.mp4`, `heatmap.png` | 2nd-order polynomial from calibration | Parallax error when looking at distances other than the calibration distance |
| AOIs: dwell, revisits | `aoi.csv` (`--aois aois.yaml`) | Fixations inside scene-pixel rectangles | Only meaningful with a still head (screen tasks) |

Each session also gets `metrics_summary.json` (detection rate, sample rate, blink rate,
saccade/fixation stats) and `report.png` (pupil size, eye position with saccades, confidence and markers).

To cross-check event detection, run `pip install -e ".[compare]"` and feed `pupil.csv` to
pymovements or REMoDNaV.

**Validated on labelled data:** see [docs/benchmarks.md](docs/benchmarks.md).
- Saccades and fixations: Lund2013, resampled to 120 Hz. Saccade F1 is 0.91 with no noise and 0.85 with 0.2° noise (experts agree at 0.97).
- Blinks: F1 0.96.
- Pupil center: LPW, real IR eye video at 640×400. 80–86% of frames within 5 px, median error 1.6 px.

Re-run with `eyetrack benchmark lund|lpw` after downloading the datasets.

## Weekend test protocol

Record each test as its own session and process it. Fill in the results table in `docs/hardware.md`.

1. **Detection:** look at 9 points around the room for 30 s. Target: `valid_rate` > 0.95.
2. **Pupil dilation:** steady light for 10 s, press `1` and turn the lights off, wait 10 s, press `2`
   and turn them back on. Then do 30 s rest, `3`, 30 s of mental arithmetic (count back from 1000
   in 7s), `4`. You should see the diameter jump at the light changes and a smaller rise during the task.
3. **Saccades:** alternate between two dots ~10° apart (about 10 cm apart at 57 cm), once per second for
   30 s. Target: about 30 saccades with consistent amplitude.
4. **Blinks:** press `b` and blink, 10 times. Target: `blink_count` = 10.
5. **Gaze accuracy:** `eyetrack calibrate`. Target: validation error < 2°.

## Layout

```
src/eyetrack/
  capture/      camera.py (threaded grab + timestamps), probe.py, recorder.py
  pupil/        detector.py (adapted JEO), eye_model.py (eyeball fit -> degrees)
  calibration/  targets.py (ArUco target display), mapping.py (pupil -> scene polynomial)
  metrics/      blinks.py, pupil_size.py, events.py (I-VT), aoi.py
  viz/          plots.py
  pipeline.py   offline processing entry point
  cli.py        `eyetrack` command
tests/          synthetic eye images/traces with known ground truth
docs/hardware.md  parts, LED wiring and eye safety, mounting, results log
```
