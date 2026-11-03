"""Cable clips for the outside of the V3 glasses temples (Sides V3 v8.step), after the Pupil Labs
DIY frame's cable hooks (pupil-labs/pupil-geometry, LGPL-3.0), with the opening twice as large so a
thick USB cable (~4.5 mm) plus the LED wires fit.

Each clip is a box-section hook on the temple's flat outer face. The temple itself forms the inner
wall of the channel; the cable presses in through a slot in the top wall next to the temple, and the
lip over the rest of the top holds it there.

Outputs (in the temple STEP's own coordinates, so they line up when imported into the temple component):
    out/cable_clips_R.step, out/cable_clips_L.step     clips only, to join onto the temples in Fusion
    out/print/temple_R_with_clips.stl, ..._L...        temple + clips, ready to slice

Needs CadQuery (`pip install cadquery`):
    python hardware/mount/temple_cable_clips.py "Sides V3 v8.step"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cadquery as cq

OPENING_OUT = 7.0     # channel depth away from the temple, mm (Pupil Labs ~3.5, doubled)
OPENING_TALL = 7.0    # channel height, mm
WALL = 1.6
CLIP_WIDTH = 4.0      # along the temple
SLOT = 3.8            # entry slot next to the temple; a 4.5 mm cable squeezes through and is held
EMBED = 0.8           # clip overlaps the temple by this much so the two join into one body
CHAMFER = 0.6         # outer corners
POSITIONS = (20.0, 45.0, 70.0)   # clip centres, mm behind the hinge axis (the ear bend starts ~75 mm)

# Right temple (second solid in the STEP): hinge axis at z = 38.9, temple runs toward -z,
# flat outer face at x = 50, outward is +x. The left temple is its mirror about x = 25.
HINGE_Z = 38.9
OUTER_X = 50.0
MIRROR_X = 25.0


def arm_section(temple: cq.Solid, z: float) -> cq.BoundBox:
    bb = temple.BoundingBox()
    cut = cq.Solid.makeBox(bb.xlen + 40, bb.ylen + 40, 0.3, cq.Vector(bb.xmin - 20, bb.ymin - 20, z - 0.15))
    return temple.intersect(cut).BoundingBox()


def clip_at(temple: cq.Solid, s: float) -> cq.Solid:
    """One clip centred s mm behind the hinge, on the right temple's outer face."""
    z = HINGE_Z - s
    sec = arm_section(temple, z)
    yc = (sec.ymin + sec.ymax) / 2
    x_in, x_out = OUTER_X - EMBED, OUTER_X + OPENING_OUT + WALL
    y_lo, y_hi = yc - OPENING_TALL / 2 - WALL, yc + OPENING_TALL / 2 + WALL
    ch = CHAMFER
    outline = [(x_in, y_lo), (x_out - ch, y_lo), (x_out, y_lo + ch), (x_out, y_hi - ch), (x_out - ch, y_hi), (x_in, y_hi)]
    body = cq.Workplane("XY").polyline(outline).close().extrude(CLIP_WIDTH).translate((0, 0, z - CLIP_WIDTH / 2))
    channel = cq.Workplane("XY").box(OPENING_OUT, OPENING_TALL, CLIP_WIDTH + 2).translate(
        (OUTER_X + OPENING_OUT / 2, yc, z))
    slot = cq.Workplane("XY").box(SLOT, WALL + 2, CLIP_WIDTH + 2).translate(
        (OUTER_X + SLOT / 2, yc + OPENING_TALL / 2 + WALL / 2, z))
    return body.cut(channel).cut(slot).val()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sides_step", help="Sides V3 v8.step (two temples)")
    p.add_argument("--out", default=str(Path(__file__).parent / "out"))
    args = p.parse_args()
    out = Path(args.out)
    (out / "print").mkdir(parents=True, exist_ok=True)

    temples = sorted(cq.importers.importStep(args.sides_step).solids().vals(), key=lambda s: s.BoundingBox().xmin)
    left, right = temples                      # left is the mirror image at lower x
    clips_r = [clip_at(right, s) for s in POSITIONS]
    clips_l = [c.mirror("YZ", (MIRROR_X, 0, 0)) for c in clips_r]

    for side, temple, clips in (("R", right, clips_r), ("L", left, clips_l)):
        cq.exporters.export(cq.Compound.makeCompound(clips), str(out / f"cable_clips_{side}.step"))
        joined = temple
        for c in clips:
            joined = joined.fuse(c)
        joined = joined.clean()
        cq.exporters.export(joined, str(out / "print" / f"temple_{side}_with_clips.stl"))
        print(f"{side}: {len(clips)} clips, joined temple valid={joined.isValid()}, "
              f"solids={len(joined.Solids())}, volume {joined.Volume() / 1000:.2f} cm3")


if __name__ == "__main__":
    main()
