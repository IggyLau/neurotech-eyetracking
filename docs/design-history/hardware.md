# Hardware

**Current setup:**
- **Camera:** InnoMaker U20CAM-9281M (OV9281, 1 MP mono global shutter), recording at 640×400 / 120 fps MJPG.
- **LEDs:** 850 nm Chanzon 3 mm IR LEDs, mounted off-axis.
- **Mount:** a printed frame, with the camera held on a bent wire and cables taped along the temple.
- **Filter:** an 850 nm IR-pass filter is being bought.

How to build it is in [hardware.md](../hardware.md); this file records why.

## Camera

### Choice
- Global shutter, so no rolling-shutter smear during fast eye movements.
- 120 fps, enough for saccades; 24–60 fps loses many (see [saccades.md](saccades.md)).
- No IR-cut filter, so it sees 850 nm. A TV remote shows as a bright flash in `eyetrack live`.

### Frame rate and resolution (2026-10-24)
- The generic OV9281 spec claims 210 fps; InnoMaker's spec, and our measurements, cap it at **120 fps**.
- **Probe results:** 640×400 @ 120 had the lowest frame-time jitter (1.7 ms). Pupil settings were tuned at this
  pixel scale on LPW. 1280×800 has no 60 fps mode and falls back to 30.
- **macOS:** camera and microphone access only works when `eyetrack` is launched from Terminal.app (or the app's
  terminal tabs), not from other shells.

### Frame rate drops to 30 fps (2026-10-27)
- Both sessions that evening switched from 120 to 30 fps part-way, and never recovered.
- Ruled out by reading the camera's settings over USB: trigger mode (off) and "Low Light Compensation"
  (auto-exposure priority, off).
- **Cause:** auto exposure had lengthened the exposure to 30 ms, more than one frame at 120 fps.
- **Fix:** manual exposure at 8 ms, set over USB at every camera start (`capture/uvc.py`), because
  AVFoundation ignores OpenCV's exposure setting.

### Stuck at 30 fps with manual exposure (2026-11-01)
- One `eyetrack live` session ran at 30 fps the whole time, although the camera read back manual 8 ms
  exposure. Tests right after it held 120 fps (no window, preview, preview + live processing,
  `live`), and closing and reopening the camera fixed it. Cause unknown.
- **Fix:** a frame-rate watchdog in `capture/camera.py`. Below 100 fps for 2 s, re-send the UVC controls;
  if still low 2 s later, reopen the camera. 100 fps matches the `validate` alignment check.
- The first `validate` run with the watchdog had the camera drop off USB (libusb: device not found, 0 fps).
  Reopening every 2–4 s didn't help and kept flipping the image between bright (camera's own auto exposure
  after each restart) and dark (8 ms). Reopens now back off: 2, 4, 8, 16, then 30 s.

### Gain does nothing (2026-11-01)
- At 8 ms, gain 15, 30, 45 and 64 were set over USB and read back correctly, but the image was identical
  (mean grey 28–29, same scene). The 30 fps limit rules out longer exposure, so more IR light is the only
  way to a brighter image at 120 fps.

### Lens (for the filter)
- M12 mount, 2.8 mm, F2.2, about 148° diagonal field of view, focus set by screwing the lens, per InnoMaker's
  manual.
- Front diameter Ø14 mm (manual drawing), so a front filter must be wider than 14 mm.

## Illumination

### Visible vs IR LEDs
- Starter-kit LEDs are visible light. Dark-pupil tracking needs **near-infrared**:
  - The iris looks lighter in IR, because melanin absorbs far less at 850 nm than in visible light. That gives
    strong pupil/iris contrast even for dark eyes.
  - IR doesn't make the pupil constrict or dazzle the wearer.

### 850 nm vs 940 nm
- 940 nm is popular because it's nearly invisible.
- The OV9281 is much more sensitive at 850 nm. A faint red glow is acceptable.

### 3 mm vs 5 mm
- Both Chanzon parts have the same 60° beam. 3 mm chosen for size near the eye.

### Current and placement
- Started at 1 kΩ in series (a few mA), which was dim. Rule: the least light that gives a clear dark pupil,
  raising exposure before current, and never above ~20 mA.
- Off-axis by ≥ 10–15 mm from the lens to avoid red-eye (a bright pupil).

## Screen reflection and flicker
- **First validate session (2026-10-26):**
  - The laptop screen reflected on the cornea as a grey patch over the pupil.
  - Frame brightness cycled every ~10 frames (~12 Hz) on white screens. Probably a beat between the screen's
    light and the camera exposure; the LEDs weren't flickering.
  - Together these caused most lost frames while reading.
- **Fix: an IR-pass filter.**
  - Screens give off essentially nothing at 850 nm, so a filter that only passes near-IR removes the reflection.
  - Avoid "850 nm longpass" filters: their cut-on is *at* 850 nm, so they block about half the LED light.
  - The lens is very wide, and coated bandpass filters shift to shorter wavelengths at steep angles, so a 40 nm
    wide bandpass is safer than a 30 nm one.
  - **Shortlist:** Commonlands CBP850 10 mm ($9, ships from San Diego, sits inside the lens holder); jsygorge
    850 nm 40 nm-wide disc over the front of the lens; Edmund RG-780 coloured glass ($37, doesn't shift with angle).
  - **Free stopgap:** 2–3 layers of exposed, developed colour film negative.

## Mount (2026-10-24 to 10-26)
1. **CadQuery models,** viewed in Fusion: a hinged clip (M2 / M2.5 hinge, filleted) and a bent-wire version,
   printed in PLA on a Bambu printer.
2. **The hinged clip failed in use:** not accurate, and it clipped on against gravity.
3. **Switched to the wire with tape,** following the JEOresearch build video.
4. **Problems:** the glasses felt front-heavy, and the wire bend sat behind the frame and felt insecure. A single
   bar like Pupil Labs' was considered.
5. **New glasses design:** a new printed pair was compared side by side with the old one in Fusion, with hinge
   joints. Cable clips along the temple were designed, then dropped for tape.
6. CAD sources and outputs are in `hardware/` (not committed yet).

## Power
- Run on mains power for recordings. Late in the first session, a low battery dropped the frame rate.

## Next
- Fit the filter, re-aim the camera so the pupil is centred while looking at the screen, and re-run
  `eyetrack validate`.
- Then add a forward-facing colour scene camera and a second eye camera. That needs all three on one clock, a
  sync check (a flash every camera sees), and a check that USB bandwidth can carry two 120 fps eye streams plus
  the scene camera.
