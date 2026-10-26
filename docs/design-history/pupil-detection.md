# Pupil detection

**Current choice:** the JEOresearch-based threshold detector (`pupil/detector.py`, `method: threshold`,
`refine: angle`), keeping frames with confidence ≥ 0.5 that pass a size-jump filter. Three alternatives exist as
options but are off by default.

## Iterations

### 1. Port of the JEOresearch detector (2026-10-23)
- **What:** darkest block → threshold at darkest + 5 / 15 / 25 → largest roundish contour → ellipse fit, keeping
  the best of the three thresholds. Ported from JEOresearch/EyeTracker (MIT), made resolution-independent, with a
  morphological closing instead of dilation so diameters aren't inflated.
- **Why this base:** the team's repo review picked JEO as the closest open-source match to a head-mounted IR
  setup. It is simple and fast (1.3 ms/frame).

### 2. Tuning on LPW (2026-10-23)
- **Data:** LPW (Tonsen 2016) records with a Pupil Labs camera, the closest public match to our OV9281.
  It was center-cropped to 640×400. Tuning used participants 1–11; results are reported on 12–22.
- **Tried:** glint removal, median blur, other threshold offsets, search box sizes, closing kernels and contour
  refinement, 60+ configurations in all. **None beat the default** on normal footage (79.7% within 5 px, median
  error 1.6 px).
- **Kept as options:** multi-candidate search (`n_seeds: 5, score: shape`). It rescued heavy mascara (14% → 72%)
  but was slightly worse on normal footage and 10× slower.

### 3. Which frames to trust
- With no filtering, 11.4% of kept frames were wrong (>10 px), usually the iris or a shadow.
- Confidence ≥ 0.6 alone cut that to 3.1% but threw away 4.9% of correct frames.
- **Chosen:** confidence ≥ 0.5 plus dropping frames whose diameter jumps >1.5× against the 2 s rolling median.
  That gives the same 3.1% wrong while keeping 88% of frames.

### 4. Our own recordings: the screen reflection (2026-10-26)
- In the first validate session, confidence sat around 0.65 on grey and white screens but was 1.0 on a dark screen.
- **Cause:** the lit screen reflects on the cornea as a grey patch about as bright as the iris. Where it covers
  the pupil, the dark blob's outline runs along the reflection. Visual checks showed the ellipse was usually
  still right; the confidence score penalised the outline.
- The JEOresearch sample video scores the same through our detector (median 0.62), so JEO's footage is not
  "higher confidence"; JEO's tracker just doesn't display a score.

### 5. Alternatives tried against the reflection
All on LPW held-out frames, scored by centre error against the hand labels:

| Variant | Within 5 px | Median error | Outcome |
|---|---|---|---|
| Default | 79.7% | 1.64 px | **kept** |
| `refine: visible`: ignore outline points next to bright pixels | 79.9% | 1.65 px | Option. Same ellipses, higher confidence. Rejected as default because it raises the score, not the accuracy, and lets more wrong frames through at a given cut-off |
| `method: edge`: our edge + RANSAC version of the published Pupil Labs method | 53.2% | – | Option. Its coarse pupil search often fails, and it jittered more on our data |
| `method: pupil_labs`: Pupil Labs' own `pupil-detectors` 2.0.2 | 26.8% (on a consecutive-frame subset) vs 75.2% default | 173 px | Option. Finds the pupil region, then usually rejects it |

Building Pupil Labs' detector took three tries:
1. Against Homebrew's OpenCV 5 it didn't compile, because `fitEllipse` and friends had moved.
2. With an include patch it compiled but misdetected even clean synthetic pupils.
3. Against `opencv@4` (4.14) it compiled cleanly; the LPW result in the table above comes from this build.

Pupil Capture users normally set an eye region and tune darkness and size per camera; we only set the size
range, so this is an "out of the box" result. Its defaults target Pupil Labs' 192×192 and 400×400 cameras.

### Lesson: optimise accuracy, not the confidence score
We nearly made `refine: visible` the default because it raised confidence. Confidence is the detector's opinion
of itself. Choose detectors on error against labels, then pick the confidence cut-off that gives an acceptable
wrong-frame rate.

## Open questions
- **Outline accuracy:** LPW labels only centres, so we can't measure whether the ellipse outline is accurate,
  which is what pupil size depends on. We need an ellipse- or mask-labelled set: TEyeD, ElSe/ExCuSe or OpenEDS.
- **"Lighting" frames:** in reading, 42% of frames were lost as "lighting". The frames are darker (brightness
  97 vs 162), the fixed threshold grabs a blob 2.3× too big, and the size filter rejects it. Planned fixes: search
  near the previous pupil, set the threshold from local contrast, and accept only plausible sizes. The IR filter
  should remove most of the cause.
- **Jitter between frames:** about 0.7 px median at fixation. A median + 1€ temporal filter cut it ~55% in a
  quick test, but would need to run offline in both directions (forwards and backwards) to keep saccade timing.
