# Changes from JEOresearch/EyeTracker

The pupil detector and the eyeball model started from [JEOresearch/EyeTracker](https://github.com/JEOresearch/EyeTracker)
(MIT, see [NOTICE](../../NOTICE)). This file records how our versions differ from the originals. Everything else in the
repo (capture, blinks, saccades, pupil size, validation, real-time processing, benchmarks) is new and has no
counterpart in JEO. Since 2026-11-07 the scene-camera calibration is also ported from JEO (section below).

When `pupil/detector.py` or `pupil/eye_model.py` changes, add a dated line under the matching section.

## Pupil detector

Original: `OrloskyPupilDetector.py`. Ours: `src/eyetrack/pupil/detector.py`.

### Kept from JEO
- Find the darkest block, threshold at darkest + 5, 15 and 25 inside a box around it.
- Keep the largest roughly round contour at each threshold, fit an ellipse, keep the best of the three.
- Drop contour points that bend away from the centroid, then refit (`refine: angle`).
- Rank candidates with JEO's goodness score (`score: jeo`), which favours big blobs.

### Changed
| | JEO | Ours | Why |
|---|---|---|---|
| Image size | Crops and resizes every frame to 640×480 | Any resolution, no crop | Our camera runs at 640×400, so the crop would cut off the sides |
| Darkest-block search | Python loops, sampling every 10th pixel, 20 px border skipped | Box filter over the whole image | Same idea, faster, no sampling gaps |
| Filling the dark blob | Dilation | Closing (dilate, then erode) | Dilation made the pupil look bigger |
| Minimum blob area | 1000 px² | 300 px² | Our pupils are smaller in the image |
| Search box | Fixed in code | `search_box`, 250 px | Configurable |
| Confidence | Computed internally, not shown | Fraction of the outline on the fitted ellipse, saved per frame | Needed to decide which frames to trust |
| Structure | One script with a GUI, globals and a file dialog | A function with a parameters object, no GUI | Used by the offline pipeline, the live view and the benchmarks |

### Added (not in JEO)
- Frame filtering: frames with confidence below 0.5, or whose diameter jumps more than 1.5× against the 2 s
  rolling median, count as missing (2026-10-23, tuned on LPW).
- `roi`: crop the search to part of the image.
- Edge penalty: dark areas near the image border lose priority, so the glasses frame or hair isn't taken
  for the pupil (2026-11-01).
- Search zone: the pupil centre must lie inside the alignment box; up to 3 seeds are tried in order of
  darkness when a zone is set (2026-11-01, fallback seeds 2026-11-02).
- Options, off by default: multi-seed search with a shape score, surround check, glint removal, median blur,
  `refine: hull` and `visible`.
- Two other detectors to compare against: an edge + RANSAC version of the Pupil Labs method (`method: edge`)
  and Pupil Labs' own package (`method: pupil_labs`). Neither beat the JEO-based one on LPW.

Benchmark numbers for each change are in [pupil-detection.md](pupil-detection.md).

## Eyeball model

Original: `3DTracker/Orlosky3DEyeTracker.py`. Ours: `src/eyetrack/pupil/eye_model.py`.

### Kept from JEO
- The pupil is a circle on a sphere, so each ellipse's minor axis points along a line through the eyeball centre
  in the image. The centre is where those lines meet.
- Gaze direction comes from where the pupil sits on that sphere.

### Changed
| | JEO | Ours | Why |
|---|---|---|---|
| Centre | Intersects random pairs of lines frame by frame, averages recent centres | One least-squares fit over the whole recording, with outlier trimming | Steadier for offline analysis |
| Radius | Grows to the largest distance seen from the centre to the pupil's outer edge (confidence ≥ 0.65) | Fitted from the ellipse shapes, or fixed from the config | JEO's grows with any outlier and never shrinks |
| Output | 3D gaze ray for display and Unity | Yaw and pitch in degrees per frame | Saccade sizes and speeds in °/s |

### Added (not in JEO)
- Fixed radius from the config, `eye_model.radius_px` (2026-11-02).
- Radius measured from on-screen targets at known angles, the `targets` part of `eyetrack validate` (2026-11-02).
- Warning when the radius is implausible for the pupil size (2026-11-02).
- Fallback to 2.7 × the pupil diameter when the fitted radius is implausible (2026-11-03).

Why each was added: [eye-model-and-gaze.md](eye-model-and-gaze.md).

## Scene-camera calibration (2026-11-07)

Original: `FrontCameraTracker/Orlosky3DEyeTrackerFrontCamera.py` (one eye plus a front camera) and
`VREyeTracker/EyeTrackerStereo.cs` (two eyes in Unity). Ours: `src/eyetrack/calibration/scene.py` and
`calibration/board.py`.

### Kept from JEO
- `project_gaze_to_tangent`, verbatim.
- `build_camera_calibration`: forward from the centre target's gaze direction, up from the target above it,
  tangent-plane coordinates for the other targets, a 2x2 least-squares matrix with no offset term, and JEO's
  checks that refuse degenerate calibrations (singular values, matrix condition, centre and top rays too close).
- `camera_pixel_ray` and `camera_angular_error_degrees`: pinhole camera from the diagonal field of view, error as
  the angle between two rays. JEO's 60° default is kept when the camera's field of view isn't configured.
- Averaging the two eyes' gaze points (`EyeTrackerStereo.cs`).
- Target order: centre, top, bottom, left, right.

### Changed
| | JEO | Ours | Why |
|---|---|---|---|
| Targets | 5 dots on the front-camera image; the wearer holds a finger on each and presses C | 9 printed ArUco markers on a wall (or a marker on the laptop screen), found in every scene frame | A fixed distance, more targets, and no lining up a finger by eye |
| Number of targets | Exactly 5, centre first | Any number; centre and top picked from the target positions | 3x3 grid |
| Target position | Fixed pixels (centre, 15% from each edge) on a 640x480 image | Where the marker was seen during that fixation, any image size | Head movement during calibration doesn't matter |
| Offsets | From the image centre | From the centre target | The centre target need not be at the image centre |
| Samples | One live gaze direction per target, at the key press | Median over the settled part of each 2.5 s fixation | Less noise |
| Eyes | One eye (FrontCameraTracker); two eyes only in the Unity code | Each eye camera fitted separately, then averaged | Binocular frame |
| Missing eye | Stereo code drops the frame | The eye that has a gaze point is used alone | Blinks and lost tracking in one eye |
| Front image | Resized to 640x480 without keeping the aspect ratio | Kept at the camera's resolution | No distortion of 16:9 cameras |
| Accuracy check | 5x5 grid of dots, finger and key press | 4 held-out board markers, every session; `--check-with` scores an old calibration at another distance | Accuracy and parallax on every calibration |
| Saving | Calibration lost on exit | `calibration.json` with the eye model it used | Reprocessing |

### Added (not in JEO)
- The `poly` mapping (pupil position to scene pixels) is fitted alongside, and both are reported, because the
  `jeo` mapping depends on our eye model, whose radius is unreliable ([eye-model-and-gaze.md](eye-model-and-gaze.md)).
- Two eye cameras of the same model: manual exposure is set on every camera with that USB ID.

## Not taken from JEO
- The head-tracking, webcam and VR (Unity) trackers.
- The finger-pointing calibration and the live single-sample calibration loop.
- The Raspberry Pi and Lite detector variants.
