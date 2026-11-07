# neurotech-eyetracking

Recording and analysis toolkit for low-cost, head-mounted eye trackers built from USB cameras.

It records synchronized video from one or two eye cameras and a scene camera with per-frame timestamps,
then computes pupil position and size, blinks, saccades, fixations, gaze on the scene, and area-of-interest
metrics offline. The reference hardware is one or two **InnoMaker U20CAM-9281M** (OV9281) global-shutter IR
cameras for the eyes and a 1080p UVC webcam for the scene, but any UVC camera supported by OpenCV works.

Raw video is always kept, so recordings can be re-processed with different settings later.

## Features

- **Capture:** threaded recording from up to three USB cameras (1, 2 or 3 camera modes: eye, both eyes, scene), with per-frame timestamps on a shared clock and keyboard event markers. Manual exposure is set over USB, and a watchdog reopens the camera if its frame rate drops
- **Pupil detection:** dark-pupil ellipse fitting with a per-frame confidence score. Optional search zone and edge penalty keep it off hair and the glasses frame
- **Eye model:** eyeball sphere fit, giving eye-rotation angles in degrees without calibration. The eyeball radius can be measured once with on-screen targets and fixed in the config
- **Guided validation:** `eyetrack validate` runs an on-screen protocol (fixation, targets, blinks, light flashes, reading, digit span) and reports what the build can measure
- **Pupillometry:** blink removal, interpolation, smoothing and baseline correction
- **Event detection:** velocity-threshold (I-VT) saccades and fixations, with noise-adaptive rejection of spurious saccades
- **Blinks:** detected from pupil dropouts and told apart from glare and tracking loss
- **Scene gaze:** calibration with a printed ArUco board or on-screen targets, fitted per eye, with held-out accuracy in degrees. Both eyes are combined into one gaze point. Gaze overlay image and video, heatmaps
- **Window:** `eyetrack ui` to pick the camera mode and run preview, record, calibrate and process
- **Areas of interest:** dwell time, time to first fixation, entries and revisits
- **Benchmarks:** scripts that score the pipeline against public hand-labelled datasets

## Installation

Requires Python 3.10+.

```bash
git clone https://github.com/IggyLau/neurotech-eyetracking.git
cd neurotech-eyetracking
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

On macOS, the terminal app needs camera access:
**System Settings → Privacy & Security → Camera**.

## Quick start

```bash
eyetrack ui                          # window: pick 1, 2 or 3 camera mode, then preview, record, calibrate, process
eyetrack probe                       # list connected cameras
eyetrack probe --index 0             # measure real frame rates for each mode of camera 0
eyetrack live                        # live pupil detection and a 10 s pupil-size trace
eyetrack preview --mode 3            # live view of both eyes + scene camera
eyetrack board                       # printable calibration board (PDF)
eyetrack calibrate --mode 3 --board  # calibrate with the board; writes data/<session>/calibration.json
eyetrack record --mode 3 --name task1    # record both eyes + scene until q is pressed
eyetrack realtime-bench data/<session>   # replay a recording through the live 120 fps path: fps, drops, latency
eyetrack process data/<session> --calibration data/<cal-session>/calibration.json --overlay
eyetrack check data/<session>          # grade a recording against docs/validation_protocol.md
eyetrack validate                    # guided: aim the camera, run the on-screen protocol, get a report
eyetrack validate --only targets     # measure the eyeball radius (needs screen.width_cm in your config)
eyetrack answer-test                 # check SPACE timing, microphone and spoken-number transcription
```

Camera modes, the calibration board, parallax and the gaze outputs are described in
[docs/scene-camera.md](docs/scene-camera.md).

`eyetrack process` also accepts a single eye video file:

```bash
eyetrack process path/to/eye_video.mp4 --annotate
```

### Recording markers

While recording, these keys write timestamped markers to `markers.csv`:

| Key | Marker |
|---|---|
| `1` / `2` | `lights_off` / `lights_on` |
| `3` / `4` | `task_start` / `task_end` |
| `b` | `blink_cue` |
| `m` | `marker` |
| `q` | stop recording |

### Real-time processing

`eyetrack record` and `eyetrack live` process every eye frame at the camera rate (120 fps) on a worker
separate from the preview window. `record` saves the live results (`live_pupil.csv`, `live_blinks.csv`,
`live_summary.json`) next to the raw video, and `eyetrack process` can still re-analyse the session
offline. On an M3, latency from capture to result is about 2.5 ms (p50) and 4 ms (p95), with no frames
dropped. Details, latencies of each live metric and benchmarks: [docs/realtime.md](docs/realtime.md).

## Configuration

Defaults are in [`config/default.yaml`](config/default.yaml). To override them, create a YAML file
containing only the keys you want to change and pass it before the subcommand:

```yaml
# config/local.yaml
eye_camera:
  index: 1
  fps: 120
scene_camera:
  index: 0
pupil:
  roi: [100, 50, 440, 300]   # x, y, w, h: restrict the pupil search to the eye region
```

```bash
eyetrack --config config/local.yaml record
```

## Outputs

`eyetrack process` writes these files into the session folder:

| File | Contents |
|---|---|
| `pupil.csv` | Per frame: pupil center, ellipse, diameter (raw and cleaned), confidence, validity, eye angles (`yaw_deg`, `pitch_deg`), event label |
| `blinks.csv` | Onset, offset, duration and type (`blink`, `low_confidence`, `closed_or_lost`, `dropout`) |
| `saccades.csv` | Onset, offset, duration, amplitude, peak velocity, and `inferred` for saccades bridged across a tracking gap |
| `fixations.csv` | Onset, offset, duration, position, dispersion |
| `gaze.csv` | Gaze in scene-camera pixels, combined and per eye (with `--calibration`) |
| `overlay_snapshot.png` | A scene frame with the gaze and the eye images with their gaze rays (with `--calibration`) |
| `eye2/` | The same per-eye files for the second eye camera, when there is one |
| `aoi.csv` | Area-of-interest metrics (with `--calibration` and `--aois`) |
| `metrics_summary.json` | Session summary: sample rate, detection rate, signal noise, blink rate, saccade and fixation statistics |
| `report.png` | Pupil size, eye position with saccades, confidence and markers over time |
| `pupil_epochs.csv`, `pupil_epochs_summary.csv`, `pupil_epochs.png` | Pupil change (%) after each marker (`lights_off`, `lights_on`, `task_start`, `task_end`, `marker`), relative to the 0.5 s before it; each epoch runs to the next marker |
| `heatmap.png`, `overlay.mp4` | Fixation heatmap and gaze overlay on the scene video (with `--calibration`; overlay needs `--overlay`) |
| `eye_annotated.mp4` | Eye video with the detected pupil drawn on it (`--annotate`) |

Areas of interest are rectangles in scene-camera pixels:

```yaml
# aois.yaml
left_page:  [0, 0, 320, 480]
right_page: [320, 0, 320, 480]
```

## Accuracy

The pipeline has been scored against two public hand-labelled datasets. Methods and full results are in
[docs/benchmarks.md](docs/benchmarks.md).

| Measurement | Dataset | Result at 120 Hz |
|---|---|---|
| Saccades | Lund2013, resampled to 120 Hz | F1 0.91 on clean signal, 0.85 with 0.2° noise (expert-vs-expert agreement: 0.97) |
| Fixations | Lund2013 | F1 0.96 |
| Blinks | Lund2013 | F1 0.96 |
| Pupil center | LPW, cropped to 640×400 | 80–86% of frames within 5 px; median error 1.6 px |

To reproduce these results, download the datasets as described in the docs, then run:

```bash
eyetrack benchmark lund
eyetrack benchmark lpw
```

## Limitations

- **Pupil diameter is in eye-camera pixels.** It can only be compared within a recording, with the headset unmoved and the lighting constant.
- **Saccade detection needs a high frame rate.** About 120 Hz is needed for reliable saccade counts. Peak velocity is underestimated and microsaccades are not resolved at this rate.
- **Gaze calibration is distance-specific.** The scene camera sits a few cm from the eyes, so expect parallax error at viewing distances other than the calibration distance. `eyetrack calibrate --check-with` measures it.
- **Area-of-interest metrics need a still head.** They use fixed scene-pixel rectangles, so they only make sense in screen tasks or with a stable head position.
- **Refit after any headset movement.** The eyeball model and calibration assume the eye camera doesn't move relative to the head.

## Known issues

Found on the reference build during validation sessions in late October and early November 2026.

- **Glare and screen reflections cause tracking loss.** Direct IR LEDs leave glare on the iris and dark edges that can be taken for the pupil, and a bright screen reflects on the cornea. Valid frames range from about 57% to 98% depending on the task. An 850 nm IR-pass filter and diffused LEDs are the planned fix ([docs/hardware.md](docs/hardware.md)).
- **Tracking drops out during large saccades.** Glare often hides the pupil during the jump back to the start of a line. Gaps of up to 150 ms are now bridged and the saccade recorded as `inferred` (no duration or peak velocity); longer gaps are still lost.
- **A fitted eyeball radius can be badly off.** Fits on the same hardware ranged from 46 to 218 px. A radius that is too small inflates gaze angles, saccade sizes and speeds, and the saccade count. When a fit is implausible the pipeline now falls back to 2.7× the pupil diameter, so sizes in degrees are approximate. Measure the radius with `eyetrack validate --only targets` and set `eye_model.radius_px`.
- **Live saccade flags miss most small saccades.** Without an eye model, the real-time path uses a pixel threshold (`events.velocity_threshold_px`, 400 px/s), which is roughly 140–190 °/s at typical radii. In one session it flagged 159 saccades where offline processing found 1,962. Set `realtime.eye_model` to an earlier session folder to run the live path in degrees. Offline processing remains the reference.
- **The live SACCADE label is hard to see.** It shows only on frames inside a saccade, about 20 ms.
- **Reading fixations were short with a bad eye-model fit.** Medians of 111–198 ms came from noise splitting fixations; with the radius fallback and glitch rejection, a re-processed session gives 200 ms, at the low end of the typical 200–250 ms.
- **The blink confidence threshold isn't validated on labelled blinks** from this camera. It was set from one cued-blink test.
- **Mental-effort pupil responses aren't reliable yet.** Digit-span dilation goes in the expected direction but with too few trials, and lid, lashes and speech add noise.
- **USB dropouts.** The camera can drop off the bus or fall back to auto exposure mid-recording. The watchdog recovers it, but frames in between are lost.

## Hardware

See [docs/hardware.md](docs/hardware.md) for camera specs, IR illumination, eye safety and mounting.

## Project layout

```
src/eyetrack/
  capture/       camera access, USB camera controls, frame-rate probe, recorder
  pupil/         pupil detector and eyeball model
  calibration/   calibration targets and board, gaze-to-scene mapping per eye, eyeball radius from targets
  metrics/       blinks, pupil size, saccades/fixations, areas of interest
  realtime/      live per-frame processing and tracker status
  benchmark/     scoring against labelled datasets and other saccade detectors
  viz/           plots, overlay video and session viewer
  pipeline.py    offline processing
  protocol.py    guided validation protocol and report
  cli.py         `eyetrack` command-line interface
  ui.py          `eyetrack ui` window
config/          default settings
docs/            hardware guide, benchmarks, validation protocol; design-history/ explains past choices
tests/           unit and integration tests on synthetic data
```

## Acknowledgements

Developed by the Neurotech@USC eye-tracking project. The pupil detector, the eyeball-model approach, and the
scene-camera calibration (tangent-plane mapping, two-eye averaging, angular error) are adapted from
[JEOresearch/EyeTracker](https://github.com/JEOresearch/EyeTracker) (MIT); see [NOTICE](NOTICE).
Benchmarks use [Lund2013](https://github.com/richardandersson/EyeMovementDetectorEvaluation)
(Andersson et al., 2017) and [LPW](https://perceptualui.org/research/datasets/LPW/) (Tonsen et al., 2016).
Neither dataset is redistributed here.

## License

MIT. See [LICENSE](LICENSE).
