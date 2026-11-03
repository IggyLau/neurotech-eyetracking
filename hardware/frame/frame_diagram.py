"""Check diagrams for combined_frame.py: top and side views of the worn frame, and the hinge joint.

Called by `combined_frame.py --diagram`; writes out/diagram_overview.png and out/diagram_joint.png.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.patches import Circle, Polygon as MplPolygon
from shapely.geometry import Polygon
from shapely.ops import unary_union

import combined_frame as cf

C_ERICSKO = "#4a78b8"   # Ericsko parts
C_V3 = "#3c9a5f"        # V3 arm
C_NEW = "#e08a2c"       # new material
C_OLD = "#9a9a9a"       # original frame, for comparison
C_BAD = "#c8423b"       # the requested 20 mm inset
C_HEAD = "#d9c9b6"

# schematic head, half width (mm) against distance behind the front plate (mm); 145 mm wide at the ears
HEAD = [(0, -6), (10, 0), (25, 12), (45, 18), (58, 26), (64, 40), (68, 60), (71, 80), (72.5, 100),
        (72, 125), (68, 155), (58, 180), (40, 198), (0, 206)]


def silhouette(m: trimesh.Trimesh, axes=(0, 2)) -> Polygon:
    tris = m.vertices[m.faces][:, :, list(axes)]
    a, b = tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]
    area = np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
    polys = [Polygon(t) for t in tris[area > 1e-6]]
    return unary_union(polys).buffer(0.01).buffer(-0.01)


def section(m: trimesh.Trimesh, y: float) -> list[Polygon]:
    s = m.section(plane_origin=(0, y, 0), plane_normal=(0, 1, 0))
    if s is None:
        return []
    out = []
    for loop in s.discrete:
        out.append(Polygon(loop[:, [0, 2]]))
    # nested loops are holes: keep the even-odd area
    res = None
    for p in sorted(out, key=lambda p: -p.area):
        res = p if res is None else res.symmetric_difference(p)
    return [res] if res is not None else []


def draw(ax, geom, fc=None, ec="k", lw=0.8, ls="-", alpha=1.0, z=2, label=None):
    geoms = getattr(geom, "geoms", [geom])
    for g in geoms:
        if g.is_empty:
            continue
        ax.add_patch(MplPolygon(np.array(g.exterior.coords), closed=True, fc=fc or "none", ec=ec, lw=lw, ls=ls,
                                alpha=alpha, zorder=z, label=label))
        label = None
        for h in g.interiors:
            ax.add_patch(MplPolygon(np.array(h.coords), closed=True, fc="white" if fc else "none", ec=ec, lw=lw * 0.7,
                                    zorder=z))


def dim(ax, p0, p1, text, off=(0, 0), color="k", fs=8):
    ax.annotate("", p1, p0, arrowprops=dict(arrowstyle="<->", color=color, lw=0.8, shrinkA=0, shrinkB=0))
    mid = ((p0[0] + p1[0]) / 2 + off[0], (p0[1] + p1[1]) / 2 + off[1])
    ax.text(*mid, text, ha="center", va="center", fontsize=fs, color=color,
            bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.85))


def overview(out: Path, front_src, temple_src, temple_r, front, r):
    phi = r["new_stop"]
    new_r = cf.posed(temple_r, phi)
    old_r = cf.posed(temple_src, r["old_stop"])
    old_straight = cf.posed(temple_src, 0)
    sil_front = silhouette(front)
    sil_new = silhouette(new_r)
    sil_old = silhouette(old_r)
    sil_old0 = silhouette(old_straight)
    mirror = lambda g: Polygon(np.array(g.exterior.coords) * (-1, 1)) if g.geom_type == "Polygon" else \
        unary_union([Polygon(np.array(p.exterior.coords) * (-1, 1)) for p in g.geoms])

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], height_ratios=[1, 0.62])
    ax = fig.add_subplot(gs[:, 0])
    head = [(x, z) for x, z in HEAD] + [(-x, z) for x, z in HEAD[::-1]]
    ax.add_patch(MplPolygon(head, closed=True, fc=C_HEAD, ec="#b8a48d", lw=0.8, zorder=0))
    ax.text(0, 160, f"schematic head\n{cf.HEAD_WIDTH:.0f} mm wide at the ears\n(measure yours)", ha="center",
            fontsize=8, color="#7a6a58")
    for g in (sil_old, mirror(sil_old)):
        draw(ax, g, ec=C_OLD, ls="--", lw=0.9, z=1)
    for g in (sil_old0, mirror(sil_old0)):
        draw(ax, g, ec=C_OLD, ls=":", lw=0.8, z=1)
    draw(ax, sil_front, fc="#cfd8e6", ec=C_ERICSKO, lw=0.8, z=3)
    for g in (sil_new, mirror(sil_new)):
        draw(ax, g, fc="#bfe0c9", ec=C_V3, lw=0.9, z=3)

    # the requested 20 mm inset: pins at +-51 mm, arms straight back, through the lens openings and face
    xi = cf.FRONT_PIN[0] - cf.REQUESTED_INSET
    for s in (-1, 1):
        ax.plot([s * xi, s * xi], [cf.FRONT_PIN[1], 150], color=C_BAD, ls="--", lw=1.4, zorder=4)
        ax.plot(s * xi, cf.FRONT_PIN[1], "o", color=C_BAD, ms=5, zorder=5)
    ax.text(xi + 2, 150, "requested: hinges 20 mm in\n(arm path at 102 mm,\ninside the lens openings)",
            color=C_BAD, fontsize=8, va="top")

    zd = cf.FRONT_PIN[1] + cf.EAR_DIST
    dim(ax, (-r["gap_new"] / 2, zd), (r["gap_new"] / 2, zd), f"new arm gap {r['gap_new']:.0f} mm (unflexed)",
        off=(0, -5), color=C_V3)
    dim(ax, (-cf.HEAD_WIDTH / 2, zd + 14), (cf.HEAD_WIDTH / 2, zd + 14), f"head {cf.HEAD_WIDTH:.0f}", color="#7a6a58")
    dim(ax, (-r["gap_old_open"] / 2, zd + 30), (r["gap_old_open"] / 2, zd + 30),
        f"Ericsko arms at their stop ({r['old_stop']:.0f}° out): {r['gap_old_open']:.0f} mm", color=C_OLD)
    dim(ax, (-r["front_width"] / 2, -14), (r["front_width"] / 2, -14), f"front {r['front_width']:.1f} mm (unchanged)")
    dim(ax, (-r["pin_span"] / 2, -24), (r["pin_span"] / 2, -24), f"pin to pin {r['pin_span']:.1f}")
    ax.plot([-r["pin_span"] / 2, r["pin_span"] / 2], [cf.FRONT_PIN[1]] * 2, "k+", ms=8)
    ax.annotate(f"{cf.EAR_DIST:.0f} mm\nbehind pin", (r["gap_new"] / 2 + 2, zd), (95, zd - 25), fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=0.6))
    ax.set_xlim(-115, 115)
    ax.set_ylim(215, -32)
    ax.set_aspect("equal")
    ax.set_xlabel("x, mm (wearer's right is +x)")
    ax.set_ylabel("distance behind the front, mm")
    ax.set_title("Top view, worn, temples at their hinge stop")
    ax.plot([], [], color=C_V3, lw=6, alpha=0.5, label=f"new: Ericsko front + V3 arms, stop {-phi:.1f}° inward")
    ax.plot([], [], color=C_OLD, ls="--", label=f"Ericsko temples at their stop, {r['old_stop']:.0f}° outward")
    ax.plot([], [], color=C_OLD, ls=":", label="Ericsko temples straight back")
    ax.plot([], [], color=C_BAD, ls="--", label="requested 40 mm narrower (not built)")
    ax.legend(loc="lower center", fontsize=8, framealpha=0.9)

    # side view
    ax2 = fig.add_subplot(gs[0, 1])
    sv_new = silhouette(new_r, axes=(2, 1))
    sv_front = silhouette(front, axes=(2, 1))
    sv_old = silhouette(old_straight, axes=(2, 1))
    draw(ax2, sv_old, ec=C_OLD, ls="--", lw=0.9, z=1)
    draw(ax2, sv_front, fc="#cfd8e6", ec=C_ERICSKO, z=3)
    draw(ax2, sv_new, fc="#bfe0c9", ec=C_V3, z=2)
    pz = cf.FRONT_PIN[1]
    for s in (20, 45, 70):
        ax2.axvline(pz + s * math.cos(math.radians(phi)), color="#555", lw=0.6, ls=":")
        ax2.text(pz + s, 33, f"clip\n{s}", ha="center", fontsize=7, color="#555")
    ax2.axvline(pz, color="k", lw=0.6)
    ax2.text(pz + 1, -38, "pin", fontsize=7)
    ax2.axvline(pz + cf.EAR_DIST, color=C_V3, lw=0.6, ls="--")
    ax2.text(pz + cf.EAR_DIST + 1, -38, "ear rest\n(assumed)", fontsize=7, color=C_V3)
    ax2.set_aspect("equal")
    ax2.set_xlim(-12, 150)
    ax2.set_ylim(-42, 42)
    ax2.set_xlabel("distance behind the front, mm")
    ax2.set_ylabel("y, mm (up)")
    ax2.set_title("Side view (right temple). Dashed: Ericsko temple")

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis("off")
    lines = [
        f"Front: Ericsko, {r['front_width']:.1f} mm wide, lens area unchanged; GLASSES_* mount positions still hold.",
        f"Hinges stay at ±{cf.FRONT_PIN[0]:.1f} mm. The lens openings reach ±{r['lens_edge']} mm, so a 20 mm",
        f"inset would put the pins at ±{r['req_pin_span'] / 2:.0f} mm, inside the lenses, and the arms would have to",
        f"spread {r['interference_req']:.0f} mm per side ({r['req_splay']:.0f}°) to fit a {cf.HEAD_WIDTH:.0f} mm head.",
        "",
        f"Why it slips now: the Ericsko hinge only stops at {r['old_stop']:.0f}° outward, so the arms never press.",
        f"New: pad on each end piece stops the arm {-phi:.1f}° inward -> {r['interference_new']:.1f} mm of grip",
        f"per side at {cf.EAR_DIST:.0f} mm on a {cf.HEAD_WIDTH:.0f} mm head (each 1° adds ~{cf.EAR_DIST * math.pi / 180:.1f} mm).",
        "",
        f"Arm: V3, {cf.V3_CUT} mm of its front cut off, grafted onto the Ericsko fork with a 4 mm root.",
        f"V3 hinge axis = Ericsko pin axis, so clips at 20/45/70 mm behind the hinge line up.",
        f"Arms fold to about {-r['fold']}° before the V3 ear hooks hit the front (not fully flat).",
    ]
    ax3.text(0, 1, "\n".join(lines), va="top", fontsize=9.5, family="monospace", transform=ax3.transAxes)
    fig.suptitle("Combined frame: Ericsko front + V3 arms (check before printing)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "diagram_overview.png", dpi=130)
    plt.close(fig)


def joint(out: Path, tparts, fparts, r):
    phi = r["new_stop"]
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.8, 1])

    # temple by source, in print pose (side view)
    ax = fig.add_subplot(gs[0, :])
    colors = {"fork": C_ERICSKO, "root": C_NEW, "arm": C_V3}
    names = {"fork": "Ericsko temple: fork + pin knuckles", "root": "new root (4 mm thick)",
             "arm": "V3 arm (Sides V3 v8)"}
    for k in ("arm", "root", "fork"):
        draw(ax, silhouette(tparts[k], axes=(0, 1)), fc=colors[k], ec="k", lw=0.5, alpha=0.75, label=names[k],
             z={"arm": 1, "root": 2, "fork": 3}[k])
    px = cf.TEMPLE_PIN[0]
    ax.axvline(px, color="k", lw=0.7)
    ax.text(px, 25, "pin axis\n(1.75 mm)", ha="center", fontsize=8)
    for x, t in ((cf.FORK_CUT, f"Ericsko cut\n{cf.FORK_CUT - px:.1f} mm"), (px + cf.V3_CUT, f"V3 cut\n{cf.V3_CUT} mm")):
        ax.axvline(x, color="#444", lw=0.6, ls="--")
        ax.text(x + 0.8, 24, t, fontsize=8)
    for s in (20, 45, 70):
        ax.plot([px + s] * 2, [19.5, 21.5], color="#555", lw=2)
        if s > 20:
            ax.text(px + s, 22.5, f"clip {s}", ha="center", fontsize=7, color="#555")
    ax.set_aspect("equal")
    ax.set_xlim(55, 205)
    ax.set_ylim(-30, 30)
    ax.set_title("Right temple, outer face down as printed, coloured by source (distances behind the pin)")
    ax.legend(loc="lower left", fontsize=8)

    # joint sections at the lower fork tab and at the front's fin, worn pose at the stop
    front_add = fparts["adds"]
    ericsko = fparts["ericsko"]
    posed_parts = {k: cf.posed(v, phi) for k, v in tparts.items()}
    sec_axes = []
    for i, (y, title) in enumerate(((cf.FORK_TABS[1][0] + cf.TEMPLE_DY + 1.5, "through the upper fork tab"),
                                    (8.25, "through the front's hinge fin"))):
        ax = fig.add_subplot(gs[1, i])
        sec_axes.append(ax)
        for g in section(ericsko, y):
            draw(ax, g, fc="#cfd8e6", ec=C_ERICSKO, lw=0.7)
        for g in section(front_add, y):
            draw(ax, g, fc=C_NEW, ec="k", lw=0.6, alpha=0.85, z=3)
        for k in ("arm", "root", "fork"):
            for g in section(posed_parts[k], y):
                draw(ax, g, fc=colors[k], ec="k", lw=0.5, alpha=0.7, z=2)
        ax.add_patch(Circle(cf.FRONT_PIN, 0.875, fc="k", zorder=5))
        ax.set_xlim(58, 82)
        ax.set_ylim(30, -3)
        ax.set_aspect("equal")
        ax.grid(alpha=0.25)
        ax.set_xlabel("x, mm")
        ax.set_ylabel("distance behind the front, mm")
        ax.set_title(f"Top section {title} (y = {y:.1f})", fontsize=10)
    ax = sec_axes[0]
    ax.annotate("stop pad (new):\narm stops here", (72.6, 4.4), (74.5, -1.5), fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=0.6))
    ax.annotate("rib (new)", (66.5, 3.2), (59, 9), fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.6))
    ax.annotate("Ericsko front", (62, 0.5), (59, 15), fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.6))
    ax.annotate("lens opening\nstarts here", (63.8, -1), (59, 24), fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.6))
    ax = sec_axes[1]
    ax.annotate("fin sits in the\nfork gap", (71, 8.5), (74.5, 16), fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.6))

    # end piece seen from behind: where pad and rib go
    ax = fig.add_subplot(gs[1, 2])
    fp = cf.footprint(ericsko)
    draw(ax, fp, fc="#cfd8e6", ec=C_ERICSKO, lw=0.7)
    draw(ax, silhouette(front_add, axes=(0, 1)), fc=C_NEW, ec="k", lw=0.6, z=3)
    fin = cf.clip_box(ericsko, (66, -50, cf.PLATE_BACK + 2.5), (80, 50, 20))
    draw(ax, silhouette(fin, axes=(0, 1)), fc="#7d93b5", ec="k", lw=0.6, z=4)
    for y0, y1 in cf.FORK_TABS:
        ax.axhspan(y0 + cf.TEMPLE_DY, y1 + cf.TEMPLE_DY, xmin=0.62, xmax=1, color=C_ERICSKO, alpha=0.12)
    ax.text(74.3, 5, "temple\nfork tabs", fontsize=7, color=C_ERICSKO)
    ax.text(68.4, 15, "hinge fin", fontsize=7)
    ax.text(65.4, 0.5, "rib", fontsize=7)
    ax.text(71.6, 14.4, "pads", fontsize=7)
    ax.set_xlim(55, 78)
    ax.set_ylim(-8, 20)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.set_xlabel("x, mm")
    ax.set_ylabel("y, mm")
    ax.set_title("Right end piece from behind (new parts orange)", fontsize=10)
    fig.suptitle("Hinge joint: which part of which model goes where", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "diagram_joint.png", dpi=130)
    plt.close(fig)


def render(out: Path, front_src, temple_src, tparts, fparts, temple_r, r):
    overview(out, front_src, temple_src, temple_r, fparts["front"], r)
    joint(out, tparts, fparts, r)
    print(f"wrote {out / 'diagram_overview.png'} and {out / 'diagram_joint.png'}")
