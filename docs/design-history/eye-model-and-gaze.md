# Eye model and gaze angles

**Current choice:** `pupil/eye_model.py`. A 2D orthographic eyeball model ("orthographic" means it ignores
perspective), fitted once per session from where the pupil ellipses' minor axes intersect. It converts pupil
position into yaw and pitch. **It is known to be wrong for our camera geometry**; the fix below is planned, not done.

## Iterations

### 1. Orthographic fit (2026-10-23)
- **Idea:** as the eye rotates away from the camera, the pupil looks like an ellipse whose minor axis points at
  the eyeball's centre. Intersecting many minor axes gives the centre; the ellipse shapes give the radius.
- **Tests:** passes on synthetic eyes. It needs varied gaze directions, so it's fitted on the whole session.

### 2. Scene-camera calibration path (not used yet)
- `calibration/mapping.py` maps pupil position to scene-camera pixels with a polynomial fitted on ArUco marker
  targets.
- We have no scene camera yet, so gaze accuracy has never been measured.

### 3. Finding: the model magnifies jitter about 5× (2026-10-26)
- At the baseline fixation, pupil-centre jitter was ~1.0 px. Converted to degrees it became 1.6°, because the
  model works at ~1.1° per pixel.
- **Why:** the fitted centre and radius put the pupil near the rim of the modelled eyeball, ~73° off-axis. The
  ellipse shape says the real angle is ~43°. 19–22% of baseline frames fell outside the sphere and got no angle.
- **Check:** with the radius constrained so the angle matches the ellipse shape (R ≈ 273 px instead of 198),
  baseline jitter drops from **1.59° to 0.32°** and frames outside the sphere from 19% to 0%. That scale is still
  unvalidated; a geometry estimate gives 0.45–0.55° per pixel.
- A robust joint refit of centre and radius still left 13–24 px residuals. The orthographic assumption breaks down
  at 3 cm, where perspective and the cornea's refraction matter.

### 4. Camera placement
- Viewing from below at ~43° is normal; Pupil Core does the same.
- The problem was framing: the pupil sat in the top fifth of the image (y ≈ 80 of 400), under the upper eyelid.
  Re-aim so the pupil is centred while looking at the screen.

### 5. Fitted radius varies between sessions; fixed, measured radius (2026-11-02)
- The fitted radius ranged **53–218 px** across sessions on the same hardware. On 2026-11-02 it came out at
  53 px, about the pupil's own diameter. At the baseline gaze that is 1.32° per pixel, against ~0.5° from
  the geometry estimate.
- **Effect:** noise read as 0.36°; the fixed 30 °/s I-VT threshold fired on jitter. There were 3.2 saccades/s
  during the baseline fixation, 5.3/s overall, and 33% of frames were labelled saccade. Big reading
  saccades fell outside the 53 px sphere, lost their angle, and were dropped.
- **Check:** re-running detection with the radius fixed and nothing else changed:

  | Radius | Noise | Baseline fixation | Reading | Frames labelled saccade |
  |---|---|---|---|---|
  | 53 (fitted) | 0.36° | 3.2/s | 3.8/s | 33% |
  | 130 | 0.24° | 1.0/s | 2.9/s | 10% |
  | 160 | 0.21° | 0.9/s | 2.8/s | 7% |
  | 200 | 0.18° | 0.6/s | 2.6/s | 5% |

- **Cause:** the radius comes from how far pupils sit from the fitted centre. With gaze confined to a
  screen, the minor-axis lines are nearly parallel, so the centre and with it the radius are poorly
  constrained.
- **Changed:**
  - `eye_model.radius_px` in the config fixes the radius; the centre is still fitted each session.
  - A **dots** part in `eyetrack validate` (9 points, 2.5 s each) measures it. The radius is chosen so the
    model's angles between fixations match the true angles between the dots, which needs
    `screen.width_cm` and `distance_cm`. Only angles between pairs are compared, so the camera's tilt and
    the resting gaze direction drop out. The centre is held at the session's fit, because that's the
    centre the radius will be used with.
  - `eyetrack process` warns when a radius is outside 1.5–6× the pupil diameter. The pupil turns about a
    point ~10–12 mm behind it and is 2–8 mm across.
- Not yet measured on real data. The dots step covers the "look around" item below only partly (9 directions).

### 6. Fallback when the fit is implausible (2026-11-03)
- A warning alone didn't help: the 53 px radius was still used, and every downstream number was wrong.
- **Changed:** when the fitted radius is outside `plausible_radius` and no `radius_px` is set, the pipeline
  uses `fallback_ratio` × the median pupil diameter (2.7), kept above the 99th percentile of the pupil's
  distance from the centre so pupils still land on the sphere. `metrics_summary.json` records
  `radius_source: fallback` and the warning says so.
- **Why 2.7:** fits that passed on this build were 2.4–3.3× the pupil diameter (5 sessions, Oct–Nov 2026).
  The failed 2026-11-02 fit was 0.9×, and a re-run on 2026-11-03 fitted 46 px (0.8×). Re-running that session
  at 120–160 px gave reading 2.8–3.0 saccades/s and 12–13% saccade frames either way, so the exact ratio matters
  less than avoiding the bad fit.
- This is a stopgap. Pupil size changes with light, so the fallback radius shifts between sessions; a measured
  `radius_px` is still the fix.

### 7. Scene camera, two eyes and JEO's calibration (2026-11-07)
- **Why now:** the next frame has two eye cameras (both U20CAM-9281M) and a scene camera. The goal for this step
  is a scene-camera image with the gaze drawn on it.
- **Two eyes:** recording takes a camera mode (1 = eye, 2 = eye + scene, 2eyes, 3 = both eyes + scene). The second
  eye is processed like the first, into `<session>/eye2/`, and the two gaze points are averaged per frame.
  Asymmetric camera placement is handled by fitting each eye on its own; nothing assumes the two cameras mirror
  each other.
- **Mapping:** ported JEO's FrontCameraTracker calibration (eye-model gaze direction -> tangent plane -> 2x2 matrix
  -> pixel) instead of writing a new one, and kept the older polynomial (pupil px -> scene px) next to it. Both
  are fitted and reported on every calibration. `jeo` is the default because it's JEO's tested approach; the
  polynomial doesn't depend on the eye model, so it may do better while the radius is unreliable (iteration 5).
  Which is better on real data is open.
- **Targets:** a printed board of ArUco markers at a fixed distance (100 cm, outer markers at ±20°), instead of
  JEO's finger held over dots on the image. The markers are found in every scene frame, so head movement during
  calibration doesn't shift the targets. Four markers are held out to measure accuracy.
- **Parallax:** the scene camera sits a few cm from each eye, so accuracy depends on distance. `--check-with`
  scores an existing calibration on a session at another distance (the half-size board at 50 cm).
- **Tests:** synthetic two-eye session with the eye cameras at 20° and 30° pitch, 8° yaw apart, different eyeball
  centres and radii, and the head turning ±2°. Held-out error is under 0.3° for each eye, both mappings and
  combined, and the combined gaze lands within 12 px of the true target. Synthetic eyes match the eye model
  exactly, so these numbers say the code is right, not how accurate real data will be.
- **Not measured yet:** no recording with three mounted cameras exists. The scene camera's field of view isn't
  measured (errors in degrees use JEO's 60° default), and its lens isn't undistorted.

## Planned
1. **9-point screen calibration**, with 4–5 extra points held out to measure accuracy, repeated at the end for
   drift. Map pupil position to screen degrees with the existing polynomial mapper. This gives real degrees and
   the first accuracy number.
2. **"Look around" step** at the start of a session, so the eye model is fitted on a wide range of directions.
3. **Longer term:** a pye3d-style 3D model with refraction correction, refitted over time. Use the LED glint
   (pupil-minus-glint vector) to compensate for the glasses slipping.

Targets: accuracy ≤ 1.5° at the centre and ≤ 2.5° at the edges; precision ≤ 0.3° raw.
