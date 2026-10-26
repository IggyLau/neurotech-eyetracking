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

## Planned
1. **9-point screen calibration**, with 4–5 extra points held out to measure accuracy, repeated at the end for
   drift. Map pupil position to screen degrees with the existing polynomial mapper. This gives real degrees and
   the first accuracy number.
2. **"Look around" step** at the start of a session, so the eye model is fitted on a wide range of directions.
3. **Longer term:** a pye3d-style 3D model with refraction correction, refitted over time. Use the LED glint
   (pupil-minus-glint vector) to compensate for the glasses slipping.

Targets: accuracy ≤ 1.5° at the centre and ≤ 2.5° at the edges; precision ≤ 0.3° raw.
