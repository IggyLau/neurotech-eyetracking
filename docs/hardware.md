# Hardware guide

How to build a head-mounted eye tracker for this toolkit from two USB cameras and a glasses frame.

## Cameras

| | OV9281 (eye) | GC0308 (scene) |
|---|---|---|
| Sensor | 1 MP mono, **global shutter** | 0.3 MP colour, rolling shutter |
| Vendor max | 120 fps MJPG @ 1280×800, 1280×720, 800×600, 640×480 ([U20CAM-9281M](https://github.com/INNO-MAKER/U20CAM-9281M)) | 30 fps @ 640×480 |
| IR filter | **None** (sees 850 nm) | Module sold for eye tracking, built-in IR LEDs |
| Role | Pupil, blinks, saccades, pupil size | What the wearer sees, gaze overlay |

Vendor frame rates are upper bounds. The USB link, codec and operating system decide the real rate,
so measure it with `eyetrack probe --index N` and put the best mode in your config.

Measured on an InnoMaker U20CAM-9281M, MJPG, MacBook Air M3 (macOS 15), 3 s per mode:

| Requested | Delivered | Measured fps | Jitter ms | Gaps |
|---|---|---|---|---|
| 1280×800 @ 120 | 1280×800 | 119.3 | 3.22 | 20 |
| 1280×720 @ 120 | 1280×720 | 120.3 | 2.12 | 16 |
| 1280×800 @ 60 | 1280×800 | 30.7 | 3.55 | 1 |
| 640×400 @ 210 | 640×400 | 120.7 | 2.01 | 17 |
| **640×400 @ 120** | 640×400 | **120.7** | **1.73** | 15 |
| 640×480 @ 120 | 640×480 | 121.0 | 1.57 | 15 |
| 640×480 @ 30 | 640×480 | 29.7 | 5.44 | 2 |

- **120 fps is the maximum**, matching the vendor spec. A 210 fps request (from generic OV9281 specs)
  still gets 120. 1280×800 has no 60 fps mode, so that request falls back to 30.
- 640×400 is not a native mode. macOS most likely scales it from 1280×800, since both are 16:10, so it
  should keep the full field of view.
- The full 120 fps arrives, so the gaps are frames delivered late and then early, not dropped frames.
  Timestamps of ±2 ms are fine for saccade metrics at 120 Hz.
- **Exposure must be manual.** On auto exposure, dim IR makes the camera lengthen the exposure past one
  frame (30 ms was read back after the 2026-10-27 sessions), and it silently drops from 120 to 30 fps.
  AVFoundation ignores OpenCV's exposure setting, so `eyetrack` sets it over USB (`eye_camera.uvc` in the
  config, `capture/uvc.py`, needs `brew install libusb`): manual, 8 ms. If the image is too dark, add LED light,
  not exposure. The camera accepts `uvc.gain` (0–64) but the image doesn't change with it.
- The camera can still get stuck at 30 fps with manual exposure, until it is reopened. `eyetrack` watches the
  frame rate: below `eye_camera.watchdog_min_fps` (100) for 2 s, it re-sends the USB controls, then reopens
  the camera if that didn't help. If the camera stays down, it waits longer between reopens (up to 30 s).
  Each recovery is printed and saved under `camera_recoveries` in `meta.json`. A reopen leaves a gap of
  about a second in the recording, and the image is briefly bright: the camera starts on its own auto
  exposure until the 8 ms setting is sent again.
- The camera's other mode, external trigger, is switched by the UVC "Auto Focus" control. Leave it off
  (`focus_auto` = 0), or frames only arrive on trigger pulses.
- **Use 640×400 @ 120**: it has the lowest jitter at full rate, and the pupil settings were tuned at this
  pixel scale (LPW, see [benchmarks.md](benchmarks.md)).
- On macOS, camera access must be granted to the app that launches `eyetrack`. Run it from Terminal.app
  if another app's shell gets `not authorized to capture video`.
- The stock U20CAM-9281M lens passes 850 nm IR: a TV remote shows as a bright flash in `eyetrack live`.
  **Don't add an IR-cut filter**, which would block the LED light. An 850 nm pass filter, which blocks
  visible light, is optional and can help against room light and screen reflections.

## Parts

- 850 nm IR LEDs, 3 mm or 5 mm, low power (2–4). Use 850 nm rather than 940 nm because the OV9281 is
  far more sensitive at 850. Pick a part whose datasheet gives **radiant intensity (mW/sr)**.
- Resistors, 220 Ω – 1 kΩ assortment
- Thin wire, heat-shrink, and a spare USB cable to cut for 5 V (or the board's 5V/GND pads; check the pinout first)
- Hot glue, matte black marker or paint
- Cheap glasses or sunglasses with the lenses popped out; pipe cleaners or a printed clip (MEYElens / Pupil Labs models)
- M12 lens with a short focus distance, if the stock OV9281 lens can't focus at ~3 cm

## IR LED circuit

```
USB 5V ──[ R ]──►|──►|── GND        two 850 nm LEDs in series, Vf ≈ 1.4–1.6 V each
                LED1 LED2           I ≈ (5 V − 3 V) / R
```

| R | Current | Use |
|---|---|---|
| 1 kΩ | ~2 mA | **Start here** |
| 470 Ω | ~4 mA | Only if the image is still too dark with the LEDs aimed and diffused |
| 220 Ω | ~9 mA | Upper limit for near-eye use; check the datasheet first |

### Eye safety rules

- Use the least light that gives a clear dark pupil. Aim the LEDs at the eye before adding current. (On the
  U20CAM-9281M, exposure is capped at 8 ms for 120 fps and gain has no effect, so the camera can't make up for it.)
- Never exceed ~20 mA, and never put a lens in front of the LEDs.
- Diffuse each LED with a dab of hot glue (the JEO fix). This also softens glints.
- Compare the LED's radiant intensity at ~3 cm against the IEC 62471 *exempt group* limit.
  This guide is not a safety certification.
- Stop if the wearer feels warmth, dryness or discomfort, and keep test sessions short.

### Avoiding red-eye (bright pupil)

Red-eye happens when the light source is close to the camera's optical axis: light bounces off the
retina straight back into the lens. Eye-tracking camera modules with LEDs right next to the lens,
such as the GC0308, are prone to it. The detector needs the **pupil to be the darkest blob**, so:

- Mount the LEDs **off-axis**: ≥ 10–15 mm from the lens center, angled toward the eye.
- If the pupil still glows at some gaze angles, move the LEDs further out.
- Put matte black on the area right around the lens to cut stray reflections.

Build on a breadboard first. Check the image in `eyetrack live`, then solder.

## Mounting

- **Eye camera:** 3–4 cm from the eye, **below and slightly to the side**, looking up. This keeps
  eyelashes out of the way and the pupil visible when the wearer looks down. Mount it rigidly: any movement breaks
  calibration and the eyeball model.
- **Scene camera:** on the bridge or temple, facing forward. **Cover or desolder its IR LEDs.**
- Run the cables along the temple arm and tie them down so a tug doesn't move the cameras.
- If the eye camera shows a large area of skin, restrict the search with `pupil.roi: [x, y, w, h]` in your config.

## Verifying a build

Record each check as its own session with `eyetrack record`, then run `eyetrack process` on it.
The full step-by-step version, with pass/fail targets graded by `eyetrack check`, is
[validation_protocol.md](validation_protocol.md).

| Check | Procedure | Expected result |
|---|---|---|
| Detection | Look at 9 points spread across the field of view for 30 s | `valid_rate` > 0.95 in `metrics_summary.json` |
| Signal noise | Any steady fixation | Low `noise_rms_s2s_deg`. Compare with the noise levels in [benchmarks.md](benchmarks.md). |
| Pupil response | Steady light 10 s → press `1`, lights off 10 s → press `2`, lights on | Clear diameter increase after `lights_off` in `pupil_epochs.png` |
| Saccades | Alternate between two targets ~10° apart (~10 cm apart at 57 cm), once per second for 30 s | ~30 saccades of consistent amplitude |
| Blinks | Press `b` and blink, 10 times | `blink_count` = 10 |
| Gaze accuracy | `eyetrack calibrate` | Validation error < 2° |
