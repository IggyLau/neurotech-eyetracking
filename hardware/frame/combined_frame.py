"""Glasses frame for the eye tracker: the flat Ericsko front with the longer V3 temple arms.

Sources (neither is redistributed here; download them and pass the paths):

    front, hinge   Ericsko, "Glasses", Printables model 98375, CC BY 4.0.
                   https://www.printables.com/model/98375  (glasses-model_files.zip)
                   The front prints flat. Each temple joins it with a fork-and-fin hinge held by a
                   1.75 mm filament pin.
    arms           Sides V3 v8.step (two temples, about 139 mm, modular hinge at the front end).

What the script does:

    temple = the Ericsko temple's fork, up to just behind the knuckles
           + a tapered root that blends it into the V3 arm and is thicker than either
           + the V3 arm, cut off behind its own hinge split
    The V3 hinge axis is put on the Ericsko pin axis, so anything placed on the V3 arm by distance
    from its hinge (the cable clips in hardware/mount/temple_cable_clips.py) keeps its position.

    front  = the Ericsko front, unchanged in the lens area, plus
             - a stop pad on the back of each end piece. The original hinge has no real stop: a
               temple swings about 21 deg outward before it hits the front, so the arms don't press
               on the head. The pad stops the arm TOE_IN_DEG inward of straight back instead, which
               gives the grip.
             - a rib on the back of each end piece, between lens and hinge fin, to stiffen it.

The frame is not made narrower at the hinges. The lens openings run out to x = +-63.6 mm and the pins
sit at +-71.1 mm, so the hinges can't move 20 mm inward without cutting into the lenses. Grip comes
from the stop and the toe-in instead (see `fit_report`).

Outputs (out/):
    front.stl/.step, temple_R.stl/.step, temple_L.stl/.step   print poses (flat, outer face down)
    assembly.stl/.step                                         worn pose, temples at the stop
    test_front_end.stl, test_temple_stub.stl                   small pieces to test the joint first
    diagram_*.png                                              with --diagram

The Ericsko model exists only as STL, so the joins are done on meshes (manifold3d) and the STEP files
are faceted solids. For a clean STEP, export the front from the Fusion 360 source
(https://a360.co/3MI9A6a) and redo the pad and rib there.

Needs CadQuery, trimesh and manifold3d (`pip install cadquery trimesh manifold3d shapely`):

    python hardware/frame/combined_frame.py glasses-model_files.zip "Sides V3 v8.step"
    ... --diagram            # also renders the check diagrams
    ... --clips              # adds the cable clips from hardware/mount/temple_cable_clips.py

Coordinates: the front STL's own (x across, wearer's right is +x, y up, z toward the wearer; the plate
is z = -1..2). Temple parts are built in the Ericsko right temple STL's coordinates (x along the arm,
y up, z inward, outer face at z = 0).
"""

from __future__ import annotations

import argparse
import io
import math
import sys
import zipfile
from pathlib import Path

import cadquery as cq
import numpy as np
import trimesh
from shapely.geometry import Polygon, box as shp_box
from shapely.ops import unary_union

# ---------------------------------------------------------------- Ericsko hinge (measured from the STLs)
FRONT_PIN = (71.07, 6.95)     # right pin hole centre in the front STL (x, z); left is mirrored. 1.75 mm bore
TEMPLE_PIN = (64.04, 3.71)    # pin hole centre in the right temple STL (x, z)
TEMPLE_DY = -4.56             # temple y + this = front y (front's fin centred in the temple's fork gap)
FORK_TABS = ((8.46, 10.76), (14.86, 17.96))   # temple y of the two fork tabs
FORK_MID = (FORK_TABS[0][0] + FORK_TABS[1][1]) / 2
FORK_END = 69.8               # temple x where the knuckle ramps and the fin pocket end
FORK_CUT = 70.8               # keep the Ericsko temple in front of this

# ---------------------------------------------------------------- V3 arm (Sides V3 v8.step)
# The first solid (lower x) is the wearer's right temple: outer face at x = 0, hinge end at +z,
# y up. Its hinge axis is at z = 38.9 and the knuckles span y = 70.5..80.6.
V3_HINGE_Z = 38.9
V3_HINGE_Y = 75.55            # middle of the V3 knuckles, put level with the middle of the Ericsko tabs
V3_CUT = 18.5                 # keep the arm from this far behind its hinge axis (its hinge split ends ~17 mm)
V3_MIRROR_X = 25.0            # the two V3 temples mirror about this plane

# ---------------------------------------------------------------- changes
TOE_IN_DEG = 2.0              # arm angle inward of straight back at the stop; ~1.6 mm per side at the ear per degree
ROOT_T = 4.0                  # arm thickness just behind the fork (Ericsko 2.6, V3 3.8)
STOP_X0 = 71.4                # stop pad on the front: from here to the end of the end piece
STOP_GAP = 0.0                # gap between pad and temple at TOE_IN_DEG; sand the pad to open the stop further
RIB_X = (65.2, 68.0)          # rib between lens opening and hinge fin, front x
RIB_Y = (2.0, 14.0)
RIB_H = 2.0                   # rib height above the plate's back face (plate is 3 mm)
PLATE_BACK = 2.0              # front STL z of the plate's back face

# ---------------------------------------------------------------- fit check
HEAD_WIDTH = 145.0            # head width just above the ears; measure yours with calipers
EAR_DIST = 90.0               # distance from the hinge pin back to where the arm rests above the ear
REQUESTED_INSET = 20.0        # per side, as first asked for; only drawn in the diagram

TEST_FRONT_X = 50.0           # test_front_end.stl keeps the front from here outward (right side)
TEST_ARM_LEN = 36.0           # test_temple_stub.stl keeps this much arm behind the pin

OUT = Path(__file__).parent / "out"


# ---------------------------------------------------------------- helpers
def mesh_from_shape(shape: cq.Shape, tol: float = 0.05) -> trimesh.Trimesh:
    verts, tris = shape.tessellate(tol, 0.35)
    return trimesh.Trimesh(np.array([v.toTuple() for v in verts]), np.array(tris))


def transformed(m: trimesh.Trimesh, fn) -> trimesh.Trimesh:
    """Apply a point map; flips the faces if the map is a reflection."""
    v = fn(m.vertices.copy())
    out = trimesh.Trimesh(v, m.faces.copy(), process=False)
    if out.volume < 0:
        out.invert()
    return out


def union(meshes: list[trimesh.Trimesh]) -> trimesh.Trimesh:
    return trimesh.boolean.union(meshes, engine="manifold")


def clip_box(m: trimesh.Trimesh, lo, hi) -> trimesh.Trimesh:
    b = trimesh.creation.box(bounds=[lo, hi])
    return trimesh.boolean.intersection([m, b], engine="manifold")


def prism(poly: Polygon, z0: float, z1: float) -> trimesh.Trimesh:
    return trimesh.creation.extrude_polygon(poly, z1 - z0).apply_translation((0, 0, z0))


def temple_to_world(v: np.ndarray, phi_deg: float) -> np.ndarray:
    """Right temple coordinates -> front coordinates, temple turned phi outward of straight back."""
    along = v[:, 0] - TEMPLE_PIN[0]
    inward = v[:, 2] - TEMPLE_PIN[1]
    a = math.radians(phi_deg)
    x = -inward * math.cos(a) + along * math.sin(a)
    z = inward * math.sin(a) + along * math.cos(a)
    return np.column_stack([FRONT_PIN[0] + x, v[:, 1] + TEMPLE_DY, FRONT_PIN[1] + z])


def posed(m: trimesh.Trimesh, phi_deg: float, side: str = "R") -> trimesh.Trimesh:
    w = transformed(m, lambda v: temple_to_world(v, phi_deg))
    return w if side == "R" else transformed(w, lambda v: v * (-1, 1, 1))


def collides(a: trimesh.Trimesh, b: trimesh.Trimesh) -> float:
    i = trimesh.boolean.intersection([a, b], engine="manifold")
    return float(i.volume) if len(i.faces) else 0.0


def stop_angle(front: trimesh.Trimesh, temple: trimesh.Trimesh, lo=-10.0, hi=40.0) -> float:
    """Opening angle (deg outward of straight back) at which the temple first touches the front."""
    for _ in range(14):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if collides(front, posed(temple, mid)) > 0.01 else (mid, hi)
    return (lo + hi) / 2


def fold_limit(front: trimesh.Trimesh, temple: trimesh.Trimesh) -> float:
    """How far the temple folds inward (deg from straight back) before it hits the front."""
    start = -int(TOE_IN_DEG) - 3
    for phi in range(start, -181, -5):
        if collides(front, posed(temple, phi)) > 0.05:
            return phi + 5
    return -180


# ---------------------------------------------------------------- sources
def load_ericsko(src: str) -> tuple[trimesh.Trimesh, trimesh.Trimesh]:
    names = ("glasses-front.stl", "glasses_temple-right_part.stl")
    p = Path(src)
    if p.suffix == ".zip":
        with zipfile.ZipFile(p) as z:
            data = [z.read(n) for n in names]
    else:
        data = [(p / n).read_bytes() for n in names]
    return tuple(trimesh.load(io.BytesIO(d), file_type="stl") for d in data)


def v3_arm(sides_step: str, clips: bool) -> trimesh.Trimesh:
    """The V3 right arm behind its hinge split, in Ericsko temple coordinates."""
    solids = sorted(cq.importers.importStep(sides_step).solids().vals(), key=lambda s: s.BoundingBox().xmin)
    right_v3, left_v3 = solids
    z_cut = V3_HINGE_Z - V3_CUT
    bb = right_v3.BoundingBox()
    keep = cq.Solid.makeBox(bb.xlen + 20, bb.ylen + 20, z_cut - bb.zmin + 10, cq.Vector(bb.xmin - 10, bb.ymin - 10, bb.zmin - 10))
    arm = right_v3.intersect(keep)
    if clips:
        sys.path.insert(0, str(Path(__file__).parents[1] / "mount"))
        import temple_cable_clips as tc       # builds clips on the other V3 solid; mirror them across
        for s in tc.POSITIONS:
            arm = arm.fuse(tc.clip_at(left_v3, s).mirror("YZ", (V3_MIRROR_X, 0, 0)))
        arm = arm.clean()
    m = mesh_from_shape(arm)
    return transformed(m, lambda v: np.column_stack([
        TEMPLE_PIN[0] + (V3_HINGE_Z - v[:, 2]),
        v[:, 1] - V3_HINGE_Y + FORK_MID,
        v[:, 0]]))


# ---------------------------------------------------------------- parts
def build_temple(ericsko_temple: trimesh.Trimesh, arm: trimesh.Trimesh) -> dict[str, trimesh.Trimesh]:
    fork = clip_box(ericsko_temple, (50, -50, -5), (FORK_CUT, 50, 20))
    x_arm = TEMPLE_PIN[0] + V3_CUT
    ys = ericsko_temple.vertices[np.abs(ericsko_temple.vertices[:, 0] - FORK_CUT) < 0.6, 1]
    slab = trimesh.creation.box(bounds=[(FORK_END, ys.min(), 0), (FORK_CUT, ys.max(), ROOT_T)])
    arm_start = clip_box(arm, (x_arm, -50, -5), (x_arm + 1.5, 50, 20))
    root = trimesh.convex.convex_hull(np.vstack([slab.vertices, arm_start.vertices]))
    return {"fork": fork, "root": root, "arm": arm}


def footprint(front: trimesh.Trimesh) -> Polygon:
    s = front.section(plane_origin=(0, 0, 0.5), plane_normal=(0, 0, 1))
    planar, _ = s.to_2D(to_2D=np.eye(4))
    return unary_union(planar.polygons_full)


def stop_pad(fp: Polygon, fork: trimesh.Trimesh, side_x: float) -> trimesh.Trimesh:
    """Pad on the back of the right end piece whose top meets the temple's end face at -TOE_IN_DEG."""
    pts, _ = trimesh.sample.sample_surface(posed(fork, -TOE_IN_DEG), 60000, seed=1)
    pads = []
    for y0, y1 in FORK_TABS:
        y0, y1 = y0 + TEMPLE_DY, y1 + TEMPLE_DY
        near = pts[(pts[:, 1] > y0) & (pts[:, 1] < y1) & (pts[:, 0] > STOP_X0 - 0.5)]
        xs = np.linspace(STOP_X0, side_x, 8)
        tops = [near[np.abs(near[:, 0] - x) < 0.4, 2].min() - STOP_GAP for x in xs]
        # profile in the x-z plane, extruded along y, then trimmed to the end piece outline
        prof = Polygon([(STOP_X0, PLATE_BACK - 0.1), (side_x, PLATE_BACK - 0.1)] + list(zip(xs[::-1], tops[::-1])))
        pad = trimesh.creation.extrude_polygon(prof, y1 - y0)
        pad.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, (1, 0, 0)))  # extrude dir -> -y
        pad.apply_translation((0, y1, 0))
        outline = prism(fp.intersection(shp_box(STOP_X0, y0, side_x + 1, y1)), -1, 10)
        pads.append(trimesh.boolean.intersection([pad, outline], engine="manifold"))
    return union(pads)


def build_front(front: trimesh.Trimesh, fork: trimesh.Trimesh) -> dict[str, trimesh.Trimesh]:
    fp = footprint(front)
    side_x = front.bounds[1][0]
    rib = prism(fp.intersection(shp_box(RIB_X[0], RIB_Y[0], RIB_X[1], RIB_Y[1])), PLATE_BACK - 0.1, PLATE_BACK + RIB_H)
    pad = stop_pad(fp, fork, side_x)
    adds_r = union([rib, pad])
    adds = union([adds_r, transformed(adds_r, lambda v: v * (-1, 1, 1))])
    return {"ericsko": front, "adds": adds, "front": union([front, adds])}


# ---------------------------------------------------------------- checks
def inner_spacing(temple_world: trimesh.Trimesh, dist: float, phi: float) -> float:
    """Gap between the two arms' inner faces, dist mm behind the pin, measured along the arm."""
    a = math.radians(phi)
    z = FRONT_PIN[1] + dist * math.cos(a)
    v = temple_world.vertices
    s = temple_world.section(plane_origin=(0, 0, z), plane_normal=(0, 0, 1))
    return 2 * float(s.vertices[:, 0].min()) if s is not None else float("nan")


def fit_report(old_front, front, old_temple, temple) -> dict:
    old_stop = stop_angle(old_front, old_temple)
    new_stop = stop_angle(front, temple)
    gap_old_open = inner_spacing(posed(old_temple, old_stop), EAR_DIST, old_stop)
    gap_old_straight = inner_spacing(posed(old_temple, 0), EAR_DIST, 0)
    gap_new = inner_spacing(posed(temple, new_stop), EAR_DIST, new_stop)
    pin_span = 2 * FRONT_PIN[0]
    req_pin_span = pin_span - 2 * REQUESTED_INSET
    # arm deflection for the requested narrower front, arms parallel at the hinge
    req_gap = gap_old_straight - 2 * REQUESTED_INSET
    r = dict(old_stop=old_stop, new_stop=new_stop, gap_old_open=gap_old_open, gap_old_straight=gap_old_straight,
             gap_new=gap_new, pin_span=pin_span, req_pin_span=req_pin_span, req_gap=req_gap,
             req_splay=math.degrees(math.atan((HEAD_WIDTH - req_gap) / 2 / EAR_DIST)),
             interference_new=(HEAD_WIDTH - gap_new) / 2, interference_req=(HEAD_WIDTH - req_gap) / 2,
             front_width=front.extents[0], lens_edge=63.6, fold=fold_limit(front, temple))
    return r


def faceted_step(m: trimesh.Trimesh, path: Path) -> None:
    """Write a mesh as a STEP solid (one face per planar patch)."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeSolid, BRepBuilderAPI_Sewing
    from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
    from OCP.StlAPI import StlAPI_Reader
    from OCP.TopoDS import TopoDS, TopoDS_Shape

    tmp = path.with_suffix(".tmp.stl")
    m.export(tmp)
    raw = TopoDS_Shape()
    StlAPI_Reader().Read(raw, str(tmp))
    tmp.unlink()
    sew = BRepBuilderAPI_Sewing(1e-4)
    sew.Add(raw)
    sew.Perform()
    shell = sew.SewedShape()
    exp = cq.Shape.cast(shell).Shells()
    mk = BRepBuilderAPI_MakeSolid()
    for sh in exp:
        mk.Add(TopoDS.Shell_s(sh.wrapped))
    unify = ShapeUpgrade_UnifySameDomain(mk.Solid(), True, True, True)
    unify.Build()
    cq.exporters.export(cq.Shape.cast(unify.Shape()), str(path))


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ericsko", help="glasses-model_files.zip from Printables 98375, or a folder of its STLs")
    p.add_argument("sides_step", help="Sides V3 v8.step")
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--clips", action="store_true", help="add the cable clips to the arms")
    p.add_argument("--diagram", action="store_true", help="render diagram_*.png")
    p.add_argument("--no-step", action="store_true", help="skip the (slow) STEP export")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    front_src, temple_src = load_ericsko(args.ericsko)
    arm = v3_arm(args.sides_step, args.clips)
    tparts = build_temple(temple_src, arm)
    temple_r = union(list(tparts.values()))
    fparts = build_front(front_src, tparts["fork"])
    front = fparts["front"]
    temple_l = transformed(temple_r, lambda v: v * (-1, 1, 1))

    world = [front, posed(temple_r, -TOE_IN_DEG), posed(temple_r, -TOE_IN_DEG, "L")]
    parts = {"front": front, "temple_R": temple_r, "temple_L": temple_l}
    for name, m in parts.items():
        m.export(out / f"{name}.stl")
        print(f"{name}: watertight={m.is_watertight}, {m.volume / 1000:.2f} cm3, "
              f"{' x '.join(f'{e:.1f}' for e in m.extents)} mm")
    trimesh.util.concatenate(world).export(out / "assembly.stl")
    # test pieces: one end piece of the front and the first part of an arm, to check the joint
    clip_box(front, (TEST_FRONT_X, -50, -5), (100, 50, 20)).export(out / "test_front_end.stl")
    clip_box(temple_r, (50, -50, -5), (TEMPLE_PIN[0] + TEST_ARM_LEN, 50, 30)).export(out / "test_temple_stub.stl")
    if not args.no_step:
        for name, m in parts.items():
            faceted_step(m, out / f"{name}.step")
        asm = cq.Assembly()
        for name, m in zip(("front", "temple_R", "temple_L"), world):
            faceted_step(m, out / "_part.step")
            asm.add(cq.importers.importStep(str(out / "_part.step")), name=name)
        (out / "_part.step").unlink()
        asm.save(str(out / "assembly.step"))

    r = fit_report(front_src, front, temple_src, temple_r)
    print(f"Ericsko hinge stop: {r['old_stop']:.1f} deg outward. New stop: {r['new_stop']:.1f} deg "
          f"(target {-TOE_IN_DEG:.1f}).")
    print(f"Arm gap {EAR_DIST:.0f} mm behind the pins: Ericsko {r['gap_old_straight']:.1f} straight, "
          f"{r['gap_old_open']:.1f} at its stop; new {r['gap_new']:.1f} "
          f"-> {r['interference_new']:.1f} mm per side on a {HEAD_WIDTH:.0f} mm head.")
    print(f"Requested 20 mm inset: pins {r['pin_span']:.1f} -> {r['req_pin_span']:.1f} mm, inside the lens openings "
          f"(edge at +-{r['lens_edge']}); arms would spread {r['interference_req']:.1f} mm per side "
          f"({r['req_splay']:.1f} deg).")
    print(f"Temples fold inward to {r['fold']} deg before hitting the front.")

    if args.diagram:
        import frame_diagram
        frame_diagram.render(out, front_src, temple_src, tparts, fparts, temple_r, r)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()
