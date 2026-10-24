# neurotech-eyetracking

Recording and analysis toolkit for low-cost, head-mounted eye trackers built from USB cameras.

It records synchronized eye- and scene-camera video with per-frame timestamps, then computes
pupil position and size, blinks, saccades, fixations, gaze on the scene, and area-of-interest
metrics offline. The reference hardware is an **OV9281** global-shutter IR camera for the eye and a
**GC0308** camera for the scene, but any UVC camera supported by OpenCV works.

Raw video is always kept, so recordings can be re-processed with different settings later.

## Features

- **Capture:** threaded recording from one or two USB cameras, with per-frame timestamps on a shared clock and keyboard event markers
- **Pupil detection:** dark-pupil ellipse fitting with a per-frame confidence score
- **Eye model:** eyeball sphere fit, giving eye-rotation angles in degrees without calibration
- **Pupillometry:** blink removal, interpolation, smoothing and baseline correction
- **Event detection:** velocity-threshold (I-VT) saccades and fixations, with noise-adaptive rejection of spurious saccades
- **Blinks:** detected from pupil dropouts and told apart from glare and tracking loss
- **Scene gaze:** ArUco-based screen calibration with validation error in degrees; gaze overlay video and heatmaps
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
eyetrack probe                       # list connected cameras
eyetrack probe --index 0             # measure real frame rates for each mode of camera 0
eyetrack live                        # live pupil detection on the eye camera
eyetrack calibrate                   # screen calibration; writes data/<session>/calibration.json
eyetrack record --name task1         # record eye + scene video until q is pressed
eyetrack process data/<session> --calibration data/<cal-session>/calibration.json --overlay
```

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
| `saccades.csv` | Onset, offset, duration, amplitude, peak velocity |
| `fixations.csv` | Onset, offset, duration, position, dispersion |
| `gaze.csv` | Gaze in scene-camera pixels (with `--calibration`) |
| `aoi.csv` | Area-of-interest metrics (with `--calibration` and `--aois`) |
| `metrics_summary.json` | Session summary: sample rate, detection rate, signal noise, blink rate, saccade and fixation statistics |
| `report.png` | Pupil size, eye position with saccades, confidence and markers over time |
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
- **Gaze calibration is distance-specific.** It's a 2D mapping, so expect parallax error at viewing distances other than the calibration distance.
- **Area-of-interest metrics need a still head.** They use fixed scene-pixel rectangles, so they only make sense in screen tasks or with a stable head position.
- **Refit after any headset movement.** The eyeball model and calibration assume the eye camera doesn't move relative to the head.

## Hardware

See [docs/hardware.md](docs/hardware.md) for camera specs, IR illumination, eye safety and mounting.

## Project layout

```
src/eyetrack/
  capture/       camera access, frame-rate probe, recorder
  pupil/         pupil detector and eyeball model
  calibration/   calibration targets and pupil-to-scene mapping
  metrics/       blinks, pupil size, saccades/fixations, areas of interest
  benchmark/     scoring against labelled datasets
  viz/           plots and overlay video
  pipeline.py    offline processing
  cli.py         `eyetrack` command-line interface
config/          default settings
docs/            hardware guide and benchmark results
tests/           unit and integration tests on synthetic data
```

## Acknowledgements

Developed by the Neurotech@USC eye-tracking project. The pupil detector and eyeball-model approach are
adapted from [JEOresearch/EyeTracker](https://github.com/JEOresearch/EyeTracker) (MIT); see [NOTICE](NOTICE).
Benchmarks use [Lund2013](https://github.com/richardandersson/EyeMovementDetectorEvaluation)
(Andersson et al., 2017) and [LPW](https://perceptualui.org/research/datasets/LPW/) (Tonsen et al., 2016).
Neither dataset is redistributed here.

## License

MIT. See [LICENSE](LICENSE).
