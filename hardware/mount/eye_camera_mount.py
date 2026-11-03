"""Glasses-rim mount for the InnoMaker U20CAM-9281M eye camera, with two 3 mm IR LED holders.

Parts (print two clips and one cradle):

    clip.stl     snaps onto the lower rim of the glasses: ramped lips at the mouth guide it on,
                 then hook over the rim's top edge. A tab reaches back and down to the hinge.
    cradle.stl   holds the 32 x 32 mm camera board on four screws. Two side arms meet the
                 clips at a hinge level with the lens tip, so tilting the camera re-aims it
                 without moving the lens. Two angled LED holders on the cradle aim at the eye.

Screws (SCREW below, M2.5 by default; the board's holes are 3.0 mm):
    board   4 x M2.5 x 6 mm, self-tapping into the standoffs
    hinge   2 x M2.5 x 10 mm plus 2 x M2.5 nuts. The head sits on the clip tab's inner face, the
            nut drops into the hex pocket on the arm's outer face. Tighten until the camera holds
            its angle.

Coordinates (world, for the preview): x along the rim, y toward the face, z up. The rim's
front-bottom edge is y = z = 0. As in the hand-held test photos, the lens tip sits just behind
and below the rim and looks steeply up at the eye, past the rim's back edge, so the rim stays
out of the image. The board hangs below, roughly level, in front of the cheek.

Needs CadQuery (`pip install cadquery`), which is not a dependency of the eyetrack package:

    python hardware/mount/eye_camera_mount.py                      # writes out/*.stl and out/*.step
    python hardware/mount/eye_camera_mount.py --preview U20CAM-9281.stp   # also renders out/preview.png
    ... --preview U20CAM-9281.stp --glasses Glasses-full-assembly.stl     # with the printed frame in the scene

Measure your own rim with calipers or a ruler and set RIM_H / RIM_D before printing.
Board dimensions come from InnoMaker's U20CAM-9281.stp (mechanical/ in their GitHub repo).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cadquery as cq

# ---------------------------------------------------------------- glasses (measure yours)
RIM_H = 3.5          # lower rim height (top to bottom), mm; measured 3-4 mm, curving along its length
RIM_D = 4.0          # lower rim depth (front to back), mm; measured
CLEARANCE = 0.25     # per side, between rim and clip walls
WALL = 1.6
LIP = 0.55           # snap lip overhang per side; hooks LIP - CLEARANCE over the rim's top edge
LIP_H = 1.0          # lip height; its top is a ramp, so the clip slides on, its bottom is flat, so it stays
CHAMFER = 0.6        # outer bottom edges of the clip

# ---------------------------------------------------------------- camera (from the STEP file)
BOARD = 31.8         # square board
BOARD_T = 1.6
HOLE_PITCH = 28.0    # 3.0 mm holes (M2.5) at (+-14, +-14)
LENS_TIP = 20.74     # lens tip above the board's front face
LENS_R = 7.0
HOLDER_HALF_X = 10.95
# cable connector on the back, board coordinates: x -10.7..-2.4, y -14.4..-11.1, 4.5 mm deep
CONNECTOR = (-10.7, -2.4, -14.4, -11.1)

# ---------------------------------------------------------------- mount geometry
PLATE_T = 2.0
STANDOFF_H = 2.0     # gap behind the board for its back-side parts
# clearance hole, self-tapping pilot, hex nut across flats and thickness
SCREWS = {"M2": dict(clear=2.4, pilot=1.8, nut_af=4.0, nut_t=1.6),
          "M2.5": dict(clear=2.9, pilot=2.2, nut_af=5.0, nut_t=2.0)}
SCREW = "M2.5"
SCREW_PILOT = SCREWS[SCREW]["pilot"]
HINGE_HOLE = SCREWS[SCREW]["clear"]
NUT_AF = SCREWS[SCREW]["nut_af"] + 0.3       # printed hex pockets need a little room
NUT_T = SCREWS[SCREW]["nut_t"] + 0.2
HINGE_R = 4.0
ARM_T = 3.5          # thick enough for the nut pocket
ARM_GAP = 0.6        # between board edge and arm
CLIP_LEN = 5.0       # clip length along the rim
# hinge = lens tip position relative to the rim's front-bottom edge (world y, z): behind and below
# the rim, as in the hand-held test, so the lens looks up past the rim's back edge. The hand-held
# video (tip ~4 mm below the rim) framed the iris at 40-50% of the image width, sometimes cutting
# off the eye corners; 8 mm below backs off ~15%. Lower it further if the corners still get cut.
HINGE_Y = RIM_D + 3.0
HINGE_Z = -8.0
# typical eye from the same edge: cornea ~12 mm behind the rim's back face, pupil ~17 mm above its top
EYE_Y = RIM_D + 12.0
EYE_Z = RIM_H + 17.0
# LED holders: 3 mm LEDs, aimed at a point on the lens axis this far beyond the lens tip
LED_X = 21.0
LED_U = 9.0
LED_TOP = 18.0       # height of the holder's top above the plate's front face
LED_BORE = 3.1
LED_SEAT = 4.2       # 3 mm LED flange sits in this counterbore

# wire version: two tunnels on the back of the plate for bendable aluminium armature wire
WIRE_D = 2.0         # 2 mm armature wire (or 12 AWG copper); bore adds 0.4 mm
WIRE_X = 13.0        # tunnels at x = +-13, clear of the cable notch
WIRE_Y0 = -6.0       # tunnels run from here to the top edge of the plate

# printed frame (Printables 98375, Glasses-full-assembly.stl), used only for the preview scene.
# STL axes: x across the frame, y up, z toward the wearer.
GLASSES_EYE_X = 47.0        # STL x of the lens centre the camera sits under
GLASSES_OTHER_EYE_X = -26.0
GLASSES_RIM_BOTTOM = -20.1  # STL y of the lower rim's bottom edge there
GLASSES_FRONT = -6.77       # STL z of the frame's front face

# derived
AIM_DIST = math.hypot(EYE_Y - HINGE_Y, EYE_Z - HINGE_Z)   # lens tip to cornea
TIP_N = STANDOFF_H + BOARD_T + LENS_TIP          # lens tip height above the plate's front face
ARM_IN = BOARD / 2 + ARM_GAP
ARM_OUT = ARM_IN + ARM_T
PLATE_X = LED_X + 4.0
PLATE_U = BOARD / 2 + 1.2
CLIP_X0 = ARM_IN - 0.3 - CLIP_LEN                # clip sits just inside each arm


def camera_plate() -> cq.Workplane:
    """Board backing plate with standoffs and LED holders, shared by both mounts.

    Camera frame: x lateral, y = in-plane (up-front when worn), z = optical axis.
    """
    plate = cq.Workplane("XY").box(2 * PLATE_X, 2 * PLATE_U, PLATE_T).translate((0, 0, -PLATE_T / 2))
    h = HOLE_PITCH / 2
    pts = [(sx * h, sy * h) for sx in (-1, 1) for sy in (-1, 1)]
    standoffs = cq.Workplane("XY").pushPoints(pts).circle(2.6).extrude(STANDOFF_H)
    body = plate.union(standoffs)

    # LED holders: blocks on the plate, bored along the line to the aim point
    aim = cq.Vector(0, 0, TIP_N + AIM_DIST)
    for s in (-1, 1):
        block = (cq.Workplane("XY").box(6, 6, LED_TOP + PLATE_T)
                 .translate((s * LED_X, LED_U, (LED_TOP - PLATE_T) / 2)))
        body = body.union(block)
        top = cq.Vector(s * LED_X, LED_U, LED_TOP)
        d = (aim - top).normalized()
        bore = cq.Solid.makeCylinder(LED_BORE / 2, 40, top - d * 30, d)
        seat = cq.Solid.makeCylinder(LED_SEAT / 2, 10, top, d)
        body = body.cut(cq.Workplane().add(bore)).cut(cq.Workplane().add(seat))

    # screw holes, and a notch open to the edge so the cable plug and cable pass
    body = body.cut(cq.Workplane("XY").pushPoints(pts).circle(SCREW_PILOT / 2)
                    .extrude(PLATE_T + STANDOFF_H + 2).translate((0, 0, -PLATE_T - 1)))
    x0, x1, y0, y1 = CONNECTOR
    notch = cq.Workplane("XY").box(x1 - x0 + 1.6, PLATE_U + y1 + 2.0, 10).translate(
        ((x0 + x1) / 2, (-PLATE_U - 1 + y1 + 1.0) / 2, 0))
    return body.cut(notch)


def cradle() -> cq.Workplane:
    """Hinged version: camera plate plus side arms that meet the rim clips level with the lens tip."""
    body = camera_plate()
    for s in (-1, 1):
        x0 = ARM_IN if s > 0 else -ARM_OUT
        arm = (cq.Workplane("YZ")
               .polyline([(-8, -PLATE_T), (8, -PLATE_T), (8, 0), (HINGE_R, TIP_N),
                          (-HINGE_R, TIP_N), (-8, 0)]).close()
               .extrude(ARM_T).translate((x0, 0, 0)))
        boss = cq.Workplane("YZ").center(0, TIP_N).circle(HINGE_R).extrude(ARM_T).translate((x0, 0, 0))
        body = body.union(arm).union(boss)
    hinge = cq.Workplane("YZ").center(0, TIP_N).circle(HINGE_HOLE / 2).extrude(2 * PLATE_X).translate((-PLATE_X, 0, 0))
    body = body.cut(hinge)
    # hex nut pockets on each arm's outer face, so the nut can't spin while tightening
    across_corners = NUT_AF / math.cos(math.radians(30))
    for s in (-1, 1):
        x0 = ARM_OUT - NUT_T if s > 0 else -ARM_OUT
        pocket = (cq.Workplane("YZ").center(0, TIP_N).polygon(6, across_corners).extrude(NUT_T)
                  .translate((x0, 0, 0)))
        body = body.cut(pocket)
    return body


def wire_cradle() -> cq.Workplane:
    """Wire version: camera plate with two tunnels on the back for bendable armature wire.

    Thread one wire through both tunnels as a U, then bend and wrap the two ends around the
    rim's outer corner and temple joint (as in JEO's build). Two strands stop the 32 mm board
    twisting the way a single wire would.
    """
    body = camera_plate()
    y0, y1 = WIRE_Y0, PLATE_U
    for s in (-1, 1):
        x = s * WIRE_X
        tunnel = (cq.Workplane("XY").box(WIRE_D + 2 * WALL + 0.4, y1 - y0, WIRE_D + 2 * WALL)
                  .translate((x, (y0 + y1) / 2, -PLATE_T - (WIRE_D + 2 * WALL) / 2)))
        body = body.union(tunnel)
        z = -PLATE_T - (WIRE_D + 2 * WALL) / 2
        bore = cq.Solid.makeCylinder((WIRE_D + 0.4) / 2, y1 - y0 + 2, cq.Vector(x, y0 - 1, z), cq.Vector(0, 1, 0))
        body = body.cut(cq.Workplane().add(bore))
    return body


def clip() -> cq.Workplane:
    """One rim clip in world coordinates (x along the rim, y toward the face, z up); x from 0 to CLIP_LEN.

    Profile (y, z): a U channel around the rim. Each wall ends in a lip whose upper face is a ramp
    (it spreads the walls as the clip is pushed up onto the rim) and whose lower face is flat (it
    hooks over the rim's top edge once the rim reaches the channel floor).
    """
    c, w, ch = CLEARANCE, WALL, CHAMFER
    yf_out, yf_in, yb_in, yb_out = -c - w, -c, RIM_D + c, RIM_D + c + w
    z_bot, z_floor, z_hook, z_top = -c - w, -c, RIM_H + c, RIM_H + c + LIP_H
    profile = [
        (yf_out + ch, z_bot), (yb_out - ch, z_bot), (yb_out, z_bot + ch), (yb_out, z_top),
        (yb_in, z_top), (yb_in - LIP, z_hook), (yb_in, z_hook),        # back lip: ramp, then hook
        (yb_in, z_floor), (yf_in, z_floor),
        (yf_in, z_hook), (yf_in + LIP, z_hook), (yf_in, z_top),        # front lip: hook, then ramp
        (yf_out, z_top), (yf_out, z_bot + ch),
    ]
    u = cq.Workplane("YZ").polyline(profile).close().extrude(CLIP_LEN)
    # tab: convex hull of the clip's floor and the hinge boss, wherever the hinge is
    pts = [(yf_out, z_floor), (yb_out, z_floor), (yf_out, z_bot + ch), (yf_out + ch, z_bot),
           (yb_out - ch, z_bot), (yb_out, z_bot + ch)]
    pts += [(HINGE_Y + HINGE_R * math.cos(a), HINGE_Z + HINGE_R * math.sin(a))
            for a in (2 * math.pi * k / 32 for k in range(32))]
    tab = cq.Workplane("YZ").polyline(_hull(pts)).close().extrude(CLIP_LEN)
    hole = cq.Workplane("YZ").center(HINGE_Y, HINGE_Z).circle(HINGE_HOLE / 2).extrude(CLIP_LEN)
    return u.union(tab).cut(hole)


def _hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """2D convex hull (monotone chain), counter-clockwise."""
    pts = sorted(set(points))
    cross = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])  # noqa: E731
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def aim_angle_deg() -> float:
    """Tilt of the optical axis above horizontal that points the lens tip at a typical eye."""
    return math.degrees(math.atan2(EYE_Z - HINGE_Z, EYE_Y - HINGE_Y))


def camera_to_world(phi_deg: float):
    """Location placing the camera frame so its lens tip sits on the hinge, tilted phi above horizontal."""
    phi = math.radians(phi_deg)
    n = cq.Vector(0, math.cos(phi), math.sin(phi))      # optical axis: up and back
    u = cq.Vector(0, -math.sin(phi), math.cos(phi))     # camera "y": up and front
    origin = cq.Vector(0, HINGE_Y, HINGE_Z) - n * TIP_N
    return origin, u, n


def place(shape: cq.Shape, phi_deg: float, z_offset: float = 0.0) -> cq.Shape:
    origin, u, n = camera_to_world(phi_deg)
    m = cq.Matrix([[1, 0, 0, origin.x], [0, u.y, n.y, origin.y], [0, u.z, n.z, origin.z]])
    return shape.translate(cq.Vector(0, 0, z_offset)).transformGeometry(m)


def _rod(a: cq.Vector, b: cq.Vector, r: float) -> cq.Shape:
    return cq.Solid.makeCylinder(r, (b - a).Length, a, (b - a).normalized())


def wire_path(phi_deg: float) -> list[cq.Shape]:
    """Illustrative wire for the preview: a U through both tunnels, ends wrapped round the rim corners."""
    origin, u, n = camera_to_world(phi_deg)
    z = -PLATE_T - (WIRE_D + 2 * WALL) / 2
    to_world = lambda x, y: cq.Vector(x, 0, 0) + origin + u * y + n * z  # noqa: E731
    r = WIRE_D / 2
    shapes = [_rod(to_world(-WIRE_X, WIRE_Y0 - 2), to_world(WIRE_X, WIRE_Y0 - 2), r)]
    for s in (-1, 1):
        bottom, top = to_world(s * WIRE_X, WIRE_Y0 - 2), to_world(s * WIRE_X, PLATE_U + 2)
        corner = cq.Vector(s * 22, RIM_D / 2, -1.5)
        shapes += [_rod(bottom, top, r), _rod(top, corner, r)]
        for k in range(3):  # wraps around the rim, which rises toward the lens corners
            shapes.append(cq.Solid.makeTorus(max(RIM_H, RIM_D) / 2 + 1.2, r,
                                             cq.Vector(s * (22 - 2.5 * k), RIM_D / 2, RIM_H / 2 + 1.0), cq.Vector(1, 0, 0)))
    return shapes


def assemblies(step_path: str, phi_deg: float) -> dict[str, dict[str, tuple]]:
    board = place(cq.importers.importStep(step_path).val(), phi_deg, STANDOFF_H + BOARD_T)
    common = {
        "camera": (board, "#2f9e6e"),
        "rim": (cq.Workplane("XY").box(80, RIM_D, RIM_H).translate((0, RIM_D / 2, RIM_H / 2)).val(), "#888888"),
        "eye": (cq.Workplane("XY").sphere(12).translate((0, EYE_Y + 12, EYE_Z)).val(), "#d9c2a3"),
    }
    return {
        "A. hinged rim clip": {
            "cradle": (place(cradle().val(), phi_deg), "#4c8eda"),
            "clip L": (clip().val().translate((CLIP_X0, 0, 0)), "#e8833a"),
            "clip R": (clip().val().translate((-CLIP_X0 - CLIP_LEN, 0, 0)), "#e8833a"),
            **common,
        },
        "B. bendable wire": {
            "plate": (place(wire_cradle().val(), phi_deg), "#8e6bd8"),
            **{f"wire {i}": (w, "#b07a3c") for i, w in enumerate(wire_path(phi_deg))},
            **common,
        },
    }


def glasses_world(stl_path: str):
    """The printed frame as a trimesh in world coordinates, the camera's lens centre at x = 0."""
    import numpy as np
    import trimesh
    parts = trimesh.load(stl_path).split(only_watertight=False)
    front = max(parts, key=lambda b: len(b.faces))   # the STL also holds the two temples, laid flat
    g = front.vertices
    # (x, y, z) -> (-x, z, y) is a proper rotation, so face orientation is kept
    front.vertices = np.column_stack([-(g[:, 0] - GLASSES_EYE_X), g[:, 2] - GLASSES_FRONT, g[:, 1] - GLASSES_RIM_BOTTOM])
    return front


def export_scene_meshes(step_path: str, glasses_stl: str, out_dir: Path, phi_deg: float) -> list[dict]:
    """World-placed STLs for Fusion: version A under one eye, version B (mirrored) under the other."""
    out_dir.mkdir(parents=True, exist_ok=True)
    other = -(GLASSES_OTHER_EYE_X - GLASSES_EYE_X)
    manifest = []
    for (variant, parts), dx in zip(assemblies(step_path, phi_deg).items(), (0.0, other)):
        tag = "A_hinged" if variant.startswith("A") else "B_wire"
        for label, (shape, color) in parts.items():
            if label == "rim":
                continue
            name = f"{tag}_{label.replace(' ', '_')}"
            path = out_dir / f"{name}.stl"
            cq.exporters.export(shape.translate(cq.Vector(dx, 0, 0)), str(path), tolerance=0.05, angularTolerance=0.2)
            manifest.append({"name": name, "path": str(path), "color": color})
    g = glasses_world(glasses_stl)
    g.export(out_dir / "glasses.stl")
    manifest.append({"name": "glasses", "path": str(out_dir / "glasses.stl"), "color": "#3a3a3a"})
    return manifest


def export_assembly(step_path: str, path: Path, phi_deg: float, spacing: float = 110.0):
    """Both mounts side by side (A at -x, B at +x) as one coloured STEP assembly, for Fusion."""
    asm = cq.Assembly(name="eye_camera_mounts")
    for k, (variant, parts) in enumerate(assemblies(step_path, phi_deg).items()):
        dx = (k - 0.5) * spacing
        name = variant.split(".")[0] + "_" + variant.split(". ")[1].replace(" ", "_")
        sub = cq.Assembly(name=name, loc=cq.Location(cq.Vector(dx, 0, 0)))
        for label, (shape, color) in parts.items():
            sub.add(shape, name=label.replace(" ", "_"), color=cq.Color(color))
        asm.add(sub)
    asm.save(str(path))


def preview(step_path: str, path: Path, phi_deg: float, glasses_stl: str | None = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    views = [("three-quarter", 22, -55), ("side", 0, 0)]
    asm = assemblies(step_path, phi_deg)
    frame = None
    if glasses_stl:
        g = glasses_world(glasses_stl)
        tri = g.triangles
        frame = tri[np.abs(tri[:, :, 0]).max(axis=1) < 40]   # just the lens the camera sits under
    fig = plt.figure(figsize=(13, 12))
    for row, (variant, parts) in enumerate(asm.items()):
        meshes = []
        if frame is not None:
            meshes.append((frame, "#3a3a3a", 0.85))
        for label, (shape, color) in parts.items():
            if label == "rim" and frame is not None:
                continue
            verts, tris = shape.tessellate(0.3, 0.5)
            v = np.array([(p.x, p.y, p.z) for p in verts])
            meshes.append((v[np.array(tris)], color, 0.35 if label == "eye" else 0.92))
        for col, (vname, elev, azim) in enumerate(views):
            ax = fig.add_subplot(2, 2, row * 2 + col + 1, projection="3d")
            for tri, color, alpha in meshes:
                ax.add_collection3d(Poly3DCollection(tri, facecolor=color, edgecolor="none", alpha=alpha))
            ax.set_xlim(-40, 40); ax.set_ylim(-45, 50); ax.set_zlim(-60, 35)
            ax.set_box_aspect((80, 95, 95))
            ax.set_xlabel("x along rim"); ax.set_ylabel("y toward face"); ax.set_zlabel("z up (mm)")
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(f"{variant} ({vname})", fontsize=11)
    fig.suptitle(f"Camera tilted {phi_deg:.0f} deg up, lens tip {AIM_DIST:.0f} mm from the eye. "
                 f"{'Dark grey = printed frame' if frame is not None else 'Grey = lower rim'}, green = U20CAM-9281M, "
                 "beige = eye", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(Path(__file__).parent / "out"))
    p.add_argument("--preview", metavar="STEP", help="U20CAM-9281.stp, to render an assembly preview")
    p.add_argument("--glasses", metavar="STL", help="Glasses-full-assembly.stl, to show the frame in the preview")
    p.add_argument("--tilt", type=float, help="camera tilt above horizontal for the preview (default: aimed at the eye)")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # STL for slicing (cradles print plate-down, the clip on its side); STEP opens as solids in Fusion
    parts = {"hinged_cradle": cradle(), "hinged_clip": clip(), "wire_plate": wire_cradle()}
    for name, part in parts.items():
        cq.exporters.export(part, str(out / f"{name}.stl"))
        cq.exporters.export(part, str(out / f"{name}.step"))
    phi = args.tilt if args.tilt is not None else aim_angle_deg()
    print(f"wrote {', '.join(parts)} (.stl and .step) to {out}. Print 2 hinged clips. "
          f"Aim angle for a typical eye: {aim_angle_deg():.0f} deg, lens tip {AIM_DIST:.0f} mm from the cornea")
    if args.preview:
        preview(args.preview, out / "preview.png", phi, args.glasses)
        if args.glasses:
            export_scene_meshes(args.preview, args.glasses, out / "scene", phi)
        export_assembly(args.preview, out / "both_mounts_assembly.step", phi)
        print(f"wrote {out / 'preview.png'} and {out / 'both_mounts_assembly.step'}")


if __name__ == "__main__":
    main()
