# Scene camera, two eyes and calibration

This page covers recording with one or two eye cameras and a forward-facing (scene) camera, calibrating gaze
onto the scene image, and processing the result.

## Camera modes

| Mode | Cameras | Use |
|---|---|---|
| `1` | eye | pupil size, blinks, saccades; no gaze on the scene |
| `2` | eye + scene | gaze on the scene from one eye |
| `2eyes` | both eyes | pupil and eye movements for each eye |
| `3` | both eyes + scene | gaze on the scene from both eyes, averaged |

`eye_camera` is the left eye (or the only eye), `eye2_camera` the right eye, `scene_camera` the
forward-facing camera. Every frame from every camera is stamped on the same clock.

Two eye cameras of the same model have the same USB vendor and product ID. The manual-exposure setting
(`uvc` block) is sent to every camera with that ID, so both stay at 120 fps. macOS has no stable way to tell
two identical cameras apart by index, so after plugging them in, check which index is which eye with
**Scan cameras** in `eyetrack ui` or with `eyetrack preview`.

USB bandwidth: two 640x400 MJPG streams at 120 fps plus a 1280x720 MJPG scene stream at 30 fps fit on
USB 2.0, but if a camera drops frames, put it on a different port or hub.

## Camera dashboard

```bash
eyetrack dashboard
```

A live check that every camera works before recording. It opens each camera index briefly and takes the
monochrome ones (the OV9281 sends three identical colour channels) as the NIR eye cameras, plus the configured
scene camera. `--no-scan` uses the configured indices instead.

Each panel has a status bar (green OK, amber weak, red failing) and shows:

- frame rate against the requested rate, dropped frames, time since the last frame, image brightness
  (all-black frames are flagged, as happens when the eye camera's exposure resets)
- on the NIR cameras: the detected pupil, the search zone, a trail of recent pupil centres, the current
  confidence, a 10 s confidence trace with the `pupil.min_confidence` line, the share of valid frames over the
  last 2 s, and precision (RMS frame-to-frame movement of the pupil centre over 1 s, in eye-image pixels; hold
  your gaze still to read it)

Keys: `q` quit, `s` swap the two eye cameras, `r` reset the counters, `p` save a snapshot. A summary is printed
on exit. Accuracy in degrees needs a calibration; the dashboard shows tracking quality (confidence, valid frames,
precision), not gaze accuracy.

## The window

```bash
eyetrack ui
```

1. Pick the camera mode.
2. Set each camera's index. **Scan cameras** shows a picture from every index.
3. **Live preview** shows every camera in the mode, with pupil detection on the eyes. **Camera dashboard**
   opens the dashboard above.
4. **Record**, **Calibrate** and **Process** run the commands below, with output in the log.
   After processing, the window shows `overlay_snapshot.png`: a scene frame with the gaze drawn on it.

On macOS, start it from Terminal.app so it can use the cameras.

## Command line

```bash
eyetrack preview --mode 3                       # live view of all three cameras
eyetrack record --mode 3 --name task1           # record both eyes + scene
eyetrack board                                  # printable calibration board (PDF)
eyetrack calibrate --mode 3 --board             # calibrate with the board on the wall
eyetrack calibrate --mode 3                     # or with targets on the laptop screen
eyetrack process data/<session> --calibration data/<cal-session>/calibration.json --overlay
```

## Calibration

The wearer looks at known targets while the scene camera sees them. The scene camera finds the target
(an ArUco marker) in every frame, so where a target is in the image is measured at the moment it's looked at,
and small head movements during calibration don't matter.

**Printed board.** `eyetrack board` writes a PDF: a layout page, then one marker per page. Print at 100%
scale and tape the markers to a wall at the positions on the layout page. Markers 0-8 are a 3x3 calibration
grid, 9-12 are accuracy targets that aren't used in the fit. The default layout puts the outer markers 20°
from the centre at 100 cm (`calibration.board`). During `calibrate --board` the laptop says each marker
number out loud, so the wearer can keep looking at the wall.

**Screen targets.** Without `--board`, a marker moves across the laptop screen (9 calibration and 5 accuracy
positions). The calibration distance is then the screen distance, about 50-60 cm.

**What's fitted.** For each eye camera, two mappings are fitted and both are reported:

- `jeo` (default): from [JEOresearch's FrontCameraTracker](https://github.com/JEOresearch/EyeTracker/tree/main/FrontCameraTracker).
  The eye model's 3D gaze direction is projected onto a plane in front of the eye, using the direction at the
  centre target as "forward" and the one at the target above it as "up". A 2x2 matrix maps that to pixel offsets
  from the centre target.
- `poly`: a 2nd-order polynomial from pupil position (eye-image pixels) to scene pixels. It doesn't use the
  eye model.

`calibration.method` picks which one `eyetrack process` uses. Each eye gets its own mapping, so the two eye
cameras don't need to be mounted symmetrically.

`calibration_summary.json` and `calibration_errors.csv` give the error for each eye, each mapping and both eyes
combined, on the calibration targets and on the held-out accuracy targets. Errors in degrees use a pinhole model
with `scene_camera.diagonal_fov_deg` (60 by default, as in JEO's code). Set it to your camera's field of view;
the accuracy numbers in degrees are only as good as that value.

## Parallax and calibration distance

The scene camera isn't where the eyes are, so a calibration is exact only at the distance it was made at.
The error is roughly `offset x |1/distance - 1/calibration distance|` radians. With 3 cm between one eye and
the camera and a calibration at 1 m, that's about 1.7° at 50 cm and at infinity. With both eyes and the camera
between them, the sideways part mostly cancels.

To measure it on your hardware, calibrate at one distance, then print the board for another distance and score
the existing calibration there without refitting:

```bash
eyetrack board --distance-cm 50
eyetrack calibrate --mode 3 --board --distance-cm 50 --check-with data/<cal-session>/calibration.json
```

The result is in `check_summary.json` and `check_errors.csv`.

## Outputs

`eyetrack calibrate` writes to the calibration session folder:

| File | Contents |
|---|---|
| `calibration.json` | per-eye mappings (`jeo` with the eye model it used, `poly`) |
| `calibration_summary.json`, `calibration_errors.csv` | errors per eye, mapping and target |
| `calibration_points.csv` (and `eye2/calibration_points.csv`) | pupil and scene position per target |

With a second eye camera, `eyetrack process` writes that eye's results to `<session>/eye2/` (same files as
the first eye), and with `--calibration` adds to the session folder:

| File | Contents |
|---|---|
| `gaze.csv` | per frame of the first eye: combined gaze (`gaze_x`, `gaze_y`), each eye's gaze, scene frame |
| `overlay_snapshot.png` | one scene frame with each eye's gaze (dots), the combined gaze (red ring) and the eye images with their gaze rays |
| `overlay.mp4` | the same for the whole video (`--overlay`) |

## Limitations

- The scene lens isn't undistorted. With a wide-angle lens, gaze near the image edges is less accurate,
  and calibration targets near the edges help less.
- The `jeo` mapping depends on the eye model fitted on the calibration session. If the eyeball radius fit is
  poor, it partly absorbs that within the calibrated range but not outside it.
- Neither mapping corrects for parallax or uses vergence to estimate depth.
- Recalibrate after the glasses move on the face.
