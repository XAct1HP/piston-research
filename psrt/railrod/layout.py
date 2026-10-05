"""The rail rod in section: every part's profile in the plane of rotation.

Everything about this concept that matters happens in one plane -- the notch,
the tongue that hooks into it, the receiver socket it seats in, the clamp arm
that wraps the crankpin. So the design is laid out here in 2D first, and the
3D solids in :mod:`psrt.railrod.cad` are these profiles extruded, plus the few
features that are genuinely three-dimensional (the sleeve's seat, the rail
slots, the swing channels, the bolt hole).

Local frame, millimetres:

    x   across the rod, in the plane of rotation. The RIGHT rail and the
        RIGHT clamp sit at +x; the left ones are exact mirrors
    v   along the rod axis, measured UP from the crankpin centre toward the
        small end. The rod frame used everywhere else in the tool has z
        pointing the other way, from the small end (z = 0) to the big end
        (z = rod length), so z = L - v

How the conforming surfaces are made conform
--------------------------------------------
Two surfaces that must fit each other are not drawn twice. The rail notch is
constructed exactly -- a radiused top, a straight ramp, a blend back to full
width -- and the clamp tongue's nose and flank are then made by SUBTRACTING
the rail, grown by the tip clearance, from the clamp blank. The male side is therefore
the notch offset by the clearance, to the precision of the geometry kernel,
whatever the notch dimensions are changed to. The same is done between the
clamp and the receiver flank. There is no second definition to drift.

The swing
---------
The clamp does not hinge on anything. It runs in a curved channel cut through
the receiver's outboard wall, and a body held to a circular channel turns
about the centre of that circle -- so the pivot is wherever the channel's
radius puts it, which is out in open air above the rail, on none of the
parts. Nothing else defines it, and nothing else may: ``swing_radius`` and
``swing_pivot_lean`` place it, and every other swinging surface is derived
from it.

That gives the order everything else is built in:

* the pivot P is ``swing_radius`` from the seated tip, straight up and leaning
  in toward the rod axis by ``swing_pivot_lean``, so the tip's travel at the
  closed position is nearly horizontal and the clamp swings DOWN onto the
  crankpin as it closes rather than sideways;
* the clamp's tongue is an annular sector about P -- constant radius, so it
  slides through the channel without binding at any angle;
* the channel IS that tongue swept through ``swing_open_angle``, subtracted
  from the receiver. It is not drawn a second time, so it cannot drift;
* the tongue's tip is then the rail notch grown by the tip clearance,
  subtracted, exactly as before. The notch sits INSIDE the receiver -- below
  the floor of the sleeve's seat -- so the clamp reaches it through the wall
  rather than over the top, and the notch's width along the crank axis, the
  channel's width and the rail's depth are all one dimension.

The arc climbs as it runs outward, so the receiver has to be tall enough for
the channel mouth to land on the outboard face with a rim above it. That is a
real constraint, not a detail, and :func:`build_layout` raises with the
arithmetic when it is not met.

:func:`swing_check` then verifies the whole clamp numerically against the
rail, the receiver, the crankpin and the other clamp.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

MM = 1000.0
ARC_STEP = 0.05          # mm, maximum chord on any sampled curve
_RES = 128               # shapely quad_segs for small radii


class LayoutError(ValueError):
    """A combination of dimensions that cannot be built, with the reason."""


# --- small geometry helpers ---------------------------------------------------

def _shapely():
    try:
        import shapely  # noqa: F401
        from shapely import geometry, affinity, ops
    except ImportError as exc:             # pragma: no cover - install issue
        raise LayoutError(
            "the rail rod layout needs shapely: python -m pip install shapely"
        ) from exc
    return geometry, affinity, ops


def arc_points(centre, radius, a0, a1, step=ARC_STEP):
    """Points on a circular arc from angle a0 to a1 (radians), inclusive."""
    span = a1 - a0
    n = max(2, int(math.ceil(abs(span) * radius / step)) + 1)
    t = np.linspace(a0, a1, n)
    return np.column_stack([centre[0] + radius * np.cos(t),
                            centre[1] + radius * np.sin(t)])


def disc(centre, radius, step=ARC_STEP):
    geometry, _, _ = _shapely()
    pts = arc_points(centre, radius, 0.0, 2.0 * math.pi, step)[:-1]
    return geometry.Polygon(pts)


def bezier(p0, p1, p2, p3, step=ARC_STEP):
    p0, p1, p2, p3 = (np.asarray(p, dtype=float) for p in (p0, p1, p2, p3))
    approx = (np.linalg.norm(p1 - p0) + np.linalg.norm(p2 - p1)
              + np.linalg.norm(p3 - p2))
    n = max(4, int(math.ceil(approx / step)) + 1)
    t = np.linspace(0.0, 1.0, n)[:, None]
    return ((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1
            + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3)


def _unit(angle):
    return np.array([math.cos(angle), math.sin(angle)])


def _largest(geom):
    """The largest polygon of a (multi)polygon result."""
    if geom.geom_type == "Polygon":
        return geom
    parts = [g for g in getattr(geom, "geoms", []) if g.geom_type == "Polygon"]
    if not parts:
        raise LayoutError("a profile boolean produced nothing")
    return max(parts, key=lambda g: g.area)


def _round_convex(geom, radius):
    """Break every convex corner to ``radius`` (a morphological opening)."""
    if radius <= 0:
        return geom
    return geom.buffer(-radius, quad_segs=_RES).buffer(radius, quad_segs=_RES)


def _fill_concave(geom, radius):
    """Blend every concave corner with ``radius`` (a morphological closing)."""
    if radius <= 0:
        return geom
    return geom.buffer(radius, quad_segs=_RES).buffer(-radius, quad_segs=_RES)


def mirror(geom):
    _, affinity, _ = _shapely()
    return affinity.scale(geom, xfact=-1.0, yfact=1.0, origin=(0.0, 0.0))


# --- the layout ---------------------------------------------------------------

@dataclass
class Notch:
    """The rail-end locking notch on the RIGHT rail, exactly.

    Travelling down the rail (decreasing v): full face, then the top radius
    from A curving in to the deepest point D, round to T where it is tangent
    to the ramp, the straight ramp out to F1, the lower blend F1 -> F2 back to
    full width, the land, and the flat bottom face at v_e.
    """
    depth: float
    alpha: float                 # ramp angle from the rail axis, radians
    r_top: float
    r_low: float
    centre: np.ndarray           # C, centre of the top radius; NOT the pivot
    A: np.ndarray                # top radius meets the rail face
    D: np.ndarray                # deepest point
    T: np.ndarray                # top radius tangent to ramp
    F1: np.ndarray               # ramp tangent to lower blend
    F2: np.ndarray               # lower blend meets the rail face
    blend_centre: np.ndarray
    normal: np.ndarray           # outward normal of the ramp surface

    @property
    def length(self) -> float:
        return float(self.A[1] - self.F2[1])

    @property
    def ramp_length(self) -> float:
        return float(np.linalg.norm(self.T - self.F1))

    def surface_points(self, step=ARC_STEP) -> np.ndarray:
        """The notch surface from A down to F2."""
        c, cf = self.centre, self.blend_centre
        a_a = math.atan2(self.A[1] - c[1], self.A[0] - c[0])
        a_t = math.pi + self.alpha
        top = arc_points(c, self.r_top, a_a, a_t, step)
        ramp = np.array([self.T, self.F1])
        low = arc_points(cf, self.r_low, self.alpha, 0.0, step)
        return np.vstack([top, ramp[1:], low[1:]])


@dataclass
class Layout:
    """Every dimension and profile of the rail rod, in local mm."""

    # scalars
    L: float                     # rod centre distance
    Rb: float                    # crankpin (housing) bore radius
    width: float                 # big-end width along y
    eye_r_out: float
    eye_r_in: float
    eye_width: float
    x_o: float                   # rail outside face
    x_i: float                   # rail inside face
    rail_depth: float
    v_e: float                   # rail flat bottom face
    v_rt: float                  # receiver top face
    theta_r: float               # receiver half angle, radians
    R_o: float                   # clamp outer radius
    r_knob: float
    r_swing: float
    v_bolt: float
    r_hole: float
    x_lug: float
    v_lug_bottom: float
    lug_gap: float
    tip_clearance: float
    guide_clearance: float
    sleeve_x: float              # half width of the sleeve
    sleeve_y: float              # half depth of the sleeve
    sleeve_v0: float             # sleeve bottom
    sleeve_v1: float             # sleeve top (at the rails)
    notch: Notch = None
    # the swing
    pivot_pt: object = None      # centre of the channel arc: on no part
    open_angle: float = 0.0      # radians the clamp swings back from seated
    channel_r_in: float = 0.0    # channel inner radius about the pivot
    channel_r_out: float = 0.0   # channel outer radius about the pivot
    channel_clearance: float = 0.0
    # the sleeve
    sleeve_skin: float = 0.0     # solid skin round each rail channel
    sleeve_face: float = 0.0     # solid shell on each big face
    sleeve_core_kind: str = "solid"
    lattice_core: tuple = ()     # (x_half, y_half, v0, v1) of the core, mm
    lattice_cell: float = 0.0
    lattice_rho_mid: float = 0.0
    lattice_rho_end: float = 0.0
    lattice_exponent: float = 1.0
    lattice_ports: bool = False
    lattice_port_diameter: float = 0.0
    port_stations: tuple = ()    # (x, v) of each powder port, both faces
    port_depth: float = 0.0      # how far each port reaches through a shell
    lattice_print: dict = None   # printability verdict
    lattice_cells: dict = None   # cells across the core
    lattice_evacuation: dict = None
    # the socket
    v_seat: float = 0.0          # floor of the sleeve's seat
    x_rc: float = 0.0            # receiver half width at the top
    slot_clearance: float = 0.0
    seat_clearance: float = 0.0
    slot_corner_radius: float = 0.0
    v_mouth: tuple = (0.0, 0.0)  # v range of the channel mouth on the face
    # profiles (shapely), local mm
    rail_right: object = None    # lower right rail incl. notch
    rails_upper: object = None   # eye + both rails + blends, above v_joint
    v_joint: float = 0.0
    envelope: object = None      # receiver before any internal cut
    receiver: object = None      # envelope less the two swing channels
    receiver_section: object = None   # less the seat and rail slots, at y = 0
    channel_right: object = None # the right swing channel on its own
    tongue_right: object = None  # the right clamp's male end, untrimmed
    seat_rect: object = None     # the sleeve's seat
    slots: object = None         # the rail slots
    clamp_right: object = None
    clamp_left: object = None
    sleeve_cut: object = None    # region the sleeve top must stay out of
    notes: list = field(default_factory=list)

    # -- convenience -----------------------------------------------------
    def z(self, v):
        """Local v to the rod frame's z."""
        return self.L - v

    @property
    def pivot(self) -> np.ndarray:
        """The swing centre. It is on none of the parts -- it is wherever the
        channel's radius puts it, which is the whole point of the mechanism."""
        return self.pivot_pt

    def rail_left(self):
        return mirror(self.rail_right)


def build_layout(state) -> Layout:
    """Lay out every part from the design state. Raises LayoutError."""
    geometry, affinity, ops = _shapely()
    g = lambda p: state[p] * MM                                  # noqa: E731

    L = g("engine.rod_length")
    Rb = g("rod.big_end_bore") / 2.0
    width = g("rod.big_end_width")
    eye_r_in = g("pin.outer_diameter") / 2.0
    eye_r_out = g("railrod.eye_outer_diameter") / 2.0
    eye_width = g("small_end.bushing_width")

    x_o = g("railrod.rail_outer_span") / 2.0
    b = g("railrod.rail_width")
    x_i = x_o - b
    h = g("railrod.rail_depth")
    notes: list[str] = []

    if eye_r_out <= eye_r_in + 1.0:
        raise LayoutError(
            f"the eye outside diameter {2 * eye_r_out:.1f} mm leaves less "
            f"than 1 mm of wall around the {2 * eye_r_in:.2f} mm pin")
    if x_i <= 0.5:
        raise LayoutError(
            f"{b:.2f} mm rails do not fit inside a {2 * x_o:.1f} mm outside "
            "span: they meet on the rod axis")
    if h > width + 1e-9:
        notes.append(
            f"rail depth {h:.1f} mm exceeds the {width:.1f} mm big-end width; "
            "the receiver's rail slots break through its faces")
    if h > eye_width + 1e-9:
        raise LayoutError(
            f"rail depth {h:.1f} mm exceeds the {eye_width:.1f} mm eye width")

    # --- the notch, exactly --------------------------------------------------
    d = g("railrod.notch_depth")
    alpha = math.radians(state["railrod.notch_ramp_angle"])
    r_n = g("railrod.notch_top_radius")
    r_l = g("railrod.notch_lower_radius")
    land = g("railrod.rail_land_length")
    t_f = g("railrod.receiver_floor_thickness")

    if d >= b:
        raise LayoutError(f"a {d:.2f} mm notch cuts through a {b:.2f} mm rail")
    if r_n < d - 1e-9:
        raise LayoutError(
            f"notch top radius {r_n:.2f} mm is smaller than the {d:.2f} mm "
            "depth; it cannot return to the rail face without a step")

    # Rail foot sits on a floor t_f above the bore, measured where the bore
    # comes closest -- under the rail's inside edge.
    if x_i < Rb:
        v_e = math.sqrt(Rb ** 2 - x_i ** 2) + t_f
    else:
        v_e = t_f
    v_1 = v_e + land                         # ramp would meet the face here

    n = np.array([math.cos(alpha), math.sin(alpha)])       # ramp normal
    x_c = x_o - d + r_n
    reach = d - r_n * (1.0 - math.cos(alpha))
    if reach <= 1e-6:
        raise LayoutError(
            f"with a {r_n:.2f} mm top radius and a {math.degrees(alpha):.0f} "
            f"deg ramp, a {d:.2f} mm notch is all radius and has no ramp. "
            "Deepen the notch, shrink the radius, or steepen the ramp")
    s_r = reach / math.sin(alpha)            # ramp length before the blend
    v_c = v_1 + s_r * math.cos(alpha) + r_n * math.sin(alpha)
    C = np.array([x_c, v_c])
    T = C - r_n * n
    D = np.array([x_c - r_n, v_c])
    v_i = v_c + math.sqrt(max(r_n ** 2 - (r_n - d) ** 2, 0.0))
    A = np.array([x_o, v_i])

    # Lower blend: convex, tangent to the ramp line and to the face x = x_o.
    cf_x = x_o - r_l
    cf_v = T[1] + (-r_l - (cf_x - T[0]) * math.cos(alpha)) / math.sin(alpha)
    Cf = np.array([cf_x, cf_v])
    F1 = Cf + r_l * n
    F2 = np.array([x_o, cf_v])
    if F1[1] >= T[1] - 1e-6:
        raise LayoutError(
            f"the {r_l:.2f} mm lower blend is longer than the "
            f"{s_r:.2f} mm ramp it blends; reduce notch_lower_radius")
    if F2[1] <= v_e + 0.2:
        raise LayoutError(
            f"the lower blend runs into the rail's bottom face; the land "
            f"({land:.2f} mm) is too short for a {r_l:.2f} mm blend")
    notch = Notch(depth=d, alpha=alpha, r_top=r_n, r_low=r_l, centre=C, A=A,
                  D=D, T=T, F1=F1, F2=F2, blend_centre=Cf, normal=n)

    # --- rails --------------------------------------------------------------
    v_joint = L - eye_r_out - 12.0
    top_local = v_joint + 2.0
    surface = notch.surface_points()
    ring = np.vstack([
        [[x_i, v_e], [x_o, v_e]],
        surface[::-1],                        # F2 up to A
        [[x_o, top_local], [x_i, top_local]],
    ])
    rail_right = geometry.Polygon(ring)
    if not rail_right.is_valid:
        rail_right = rail_right.buffer(0)

    # eye + the rails' upper ends, blended
    eye = disc((0.0, L), eye_r_out)
    r_t = g("railrod.eye_transition_radius")
    bars = geometry.box(x_i, v_joint - 3.0, x_o, L).union(
        geometry.box(-x_o, v_joint - 3.0, -x_i, L))
    upper = _fill_concave(eye.union(bars), r_t)
    # The blend only changes the rails near the eye. Below the lowest point
    # it can reach, rebuild them as exact boxes: a buffer round trip leaves
    # micron-level wobble on a straight face, and that wobble would meet the
    # exact lower rails as a sliver face the mesher chokes on.
    reach = eye_r_out + r_t
    v_clip = L - max(
        math.sqrt(max(reach ** 2 - (x_o + r_t) ** 2, 0.0)),
        math.sqrt(max(reach ** 2 - max(x_i - r_t, 0.0) ** 2, 0.0))) - 0.5
    if v_clip <= v_joint + 1.0:
        raise LayoutError("the eye blend reaches the rail joint; the eye is "
                          "too close to the rails' straight run")
    exact = geometry.box(x_i, v_joint, x_o, v_clip).union(
        geometry.box(-x_o, v_joint, -x_i, v_clip))
    upper = _largest(upper.intersection(
        geometry.box(-2 * L, v_clip, 2 * L, 2 * L)).union(exact))

    # --- the sleeve's footprint, needed before the seat is cut ----------------
    t_sw = g("railrod.sleeve_wall")
    c_s = g("railrod.sleeve_clearance")
    gap_s = g("railrod.sleeve_end_gap")
    sleeve_x = x_o + c_s + t_sw
    sleeve_y = h / 2.0 + c_s + t_sw

    # --- receiver: a deep socket, not a shallow pocket -------------------------
    theta_r = math.radians(state["railrod.receiver_arc_half_angle"])
    gamma = math.radians(state["railrod.receiver_tip_angle"])
    c_t = g("railrod.tip_clearance")
    c_g = g("railrod.guide_clearance")
    c_sl = g("railrod.slot_clearance")
    r_slot = g("railrod.slot_corner_radius")
    c_ch = g("railrod.channel_clearance")
    wall = g("railrod.receiver_wall")
    rim = g("railrod.receiver_rim")
    eng = g("railrod.rail_engagement")
    seat_d = g("railrod.seat_depth")
    c_seat = g("railrod.seat_clearance")
    edge = g("railrod.edge_radius")

    v_seat = v_e + eng                  # floor of the seat the sleeve sits in
    v_rt = v_seat + seat_d              # receiver top face
    x_rc = x_o + c_sl + wall            # receiver half width at the top

    if v_rt <= Rb:
        raise LayoutError("the receiver top face is below the top of the bore")
    if A[1] >= v_seat - 0.5:
        raise LayoutError(
            f"the notch reaches up to v = {A[1]:.1f} mm but the floor of the "
            f"sleeve seat is at {v_seat:.1f} mm. The notch has to finish "
            "inside the receiver so the clamp can reach it through the wall: "
            "lengthen rail_engagement, or shorten the notch")
    if x_rc < sleeve_x + c_seat + 0.8:
        notes.append(
            f"only {x_rc - sleeve_x - c_seat:.2f} mm of rim is left between "
            "the sleeve seat and the outside of the receiver")

    # --- the swing: an arc about a pivot that lies on no part -----------------
    R_sw = g("railrod.swing_radius")
    lean = math.radians(state["railrod.swing_pivot_lean"])
    t_ch = g("railrod.channel_thickness")
    open_a = math.radians(state["railrod.swing_open_angle"])
    r_in = R_sw - t_ch / 2.0
    r_out = R_sw + t_ch / 2.0
    if r_in <= 0.5:
        raise LayoutError(
            f"a {t_ch:.1f} mm channel in a {R_sw:.1f} mm swing radius leaves "
            "no inner wall; raise swing_radius or thin the channel")

    # The pivot. It is on no part -- it is the centre of the channel's arc,
    # which is the only thing that defines it. Measured from the seated tip:
    # R_sw away, straight up, leaning OUT away from the rod axis by `lean`,
    # so the clamp swings DOWN onto the crankpin as it closes rather than
    # straight sideways. Out, not in: everything directly above the notch is
    # rail, and above that is sleeve, so a pivot leaning inboard is inside a
    # part, and then that part is the hinge and the channel is not.
    P = C + R_sw * np.array([math.sin(lean), math.cos(lean)])

    def _theta_at_x(radius, x):
        """Where the arc of this radius about P crosses the vertical line x.

        Angles are measured at the pivot from straight down, positive toward
        +x. The pivot leans OUT from the tip, so the seated tip sits at
        ``-lean`` and the clamp opens by turning to larger angles, which is
        what carries the tongue outward through the wall. Returns None if
        the arc never gets that far out.
        """
        ratio = (x - P[0]) / radius
        return math.asin(ratio) if abs(ratio) <= 1.0 else None

    th_break = _theta_at_x(r_in - c_ch, x_rc)
    th_lip = _theta_at_x(r_out + c_ch, x_rc)
    if th_break is None or th_lip is None:
        raise LayoutError(
            f"a {R_sw:.1f} mm swing radius leaning "
            f"{math.degrees(lean):.0f} deg puts the pivot at x = "
            f"{P[0]:.1f} mm, and the channel never reaches the receiver's "
            f"outer face at x = {x_rc:.1f} mm. The clamp could not be got "
            "in. Raise swing_radius, reduce swing_pivot_lean, or thin "
            "receiver_wall")
    v_mouth_top = P[1] - (r_in - c_ch) * math.cos(th_break)
    v_mouth_bot = P[1] - (r_out + c_ch) * math.cos(th_lip)
    if v_mouth_top + rim > v_rt:
        raise LayoutError(
            f"the channel mouth reaches v = {v_mouth_top:.1f} mm on the "
            f"receiver's outer face, and with a {rim:.1f} mm rim above it "
            f"the top face would have to be at {v_mouth_top + rim:.1f} mm, "
            f"not {v_rt:.1f}. The arc climbs as it goes out: lengthen "
            "rail_engagement, flatten it with a larger swing_radius, or thin "
            "receiver_wall")

    # Where the pivot lands is a hard requirement, not a preference.
    free_air = [
        ("rail", rail_right),
        ("rails above the notch", geometry.box(x_i, v_e, x_o, L)),
        ("sleeve", geometry.box(-sleeve_x, v_seat, sleeve_x, L)),
        ("crankpin bore", disc((0.0, 0.0), Rb)),
    ]
    P_pt = geometry.Point(float(P[0]), float(P[1]))
    for what, shape in free_air:
        if shape.contains(P_pt):
            raise LayoutError(
                f"the swing pivot lands at ({P[0]:.1f}, {P[1]:.1f}) mm, "
                f"inside the {what}. The pivot has to be in open air -- if "
                "it is on a part, that part is the hinge and the channel is "
                "not what sets the motion. Lean it further out "
                "(swing_pivot_lean) or shorten swing_radius")

    def sector(rho0, rho1, a0, a1):
        """An annular sector about the pivot, in the angle convention above."""
        outer = arc_points(P, rho1, a0 - math.pi / 2.0, a1 - math.pi / 2.0)
        inner = arc_points(P, rho0, a1 - math.pi / 2.0, a0 - math.pi / 2.0)
        ring = geometry.Polygon(np.vstack([outer, inner]))
        return ring if ring.is_valid else ring.buffer(0)

    # The tongue starts behind the seated angle and is trimmed to shape by the
    # rail itself, so the male form IS the notch offset by the tip clearance,
    # whatever the notch is changed to. It runs out past the receiver's face
    # so that it meets the clamp arm in open air.
    th_seat = -lean                       # where the seated tip sits
    th_tip = th_seat - 0.20
    th_tail = min(math.pi / 2.0 - 0.05, th_break + math.radians(20.0))
    if th_tail <= th_break:
        raise LayoutError("the channel has no length outside the receiver")
    tongue = sector(r_in, r_out, th_tip, th_tail)
    channel = sector(r_in - c_ch, r_out + c_ch, th_tip, th_tail + open_a)

    # --- receiver envelope ----------------------------------------------------
    E_r = Rb * np.array([math.sin(theta_r), math.cos(theta_r)])
    start_angle = math.pi - theta_r - gamma          # flank leaves the bore
    if not (0.05 < start_angle < math.pi - 0.05):
        raise LayoutError(
            "receiver_arc_half_angle plus receiver_tip_angle leaves the flank "
            "pointing back into the bore")
    P3 = np.array([x_rc, v_rt])
    if P3[0] >= E_r[0] + 30 or P3[1] <= E_r[1]:
        raise LayoutError("the receiver flank has nowhere to go: its top "
                          "corner sits below its bottom corner")
    handle = np.linalg.norm(P3 - E_r) / 3.0
    flank = bezier(E_r, E_r + handle * _unit(start_angle),
                   P3 - handle * np.array([0.0, 1.0]), P3)
    bore_arc = arc_points((0.0, 0.0), Rb, math.pi / 2.0,
                          math.pi / 2.0 - theta_r)       # top -> E_r
    right = np.vstack([bore_arc, flank[1:], [[0.0, v_rt]]])
    left = right[::-1].copy()
    left[:, 0] *= -1.0
    envelope = geometry.Polygon(np.vstack([right[:-1], left[1:-1]]))
    if not envelope.is_valid:
        envelope = envelope.buffer(0)
    envelope = _round_convex(envelope, edge)

    channels = channel.union(mirror(channel))
    cut = envelope.difference(channels)
    receiver = _largest(cut)
    if receiver.area < 0.5 * envelope.area:
        raise LayoutError(
            "the swing channels cut the receiver in half. The arc is passing "
            "through the wall almost lengthways: raise swing_radius or "
            "reduce swing_pivot_lean")

    # What is left of the wall outboard of the rail slot, below the notch --
    # this is what locates the rail foot sideways.
    below = receiver.intersection(geometry.box(x_o + c_sl, v_e, x_rc,
                                               F2[1]))
    if below.is_empty or below.area < 0.2:
        notes.append(
            "the swing channel removes the wall outboard of the rail slot; "
            "the rail foot is located outward by the clamp tongue alone")

    # the seat the sleeve drops into, and the slots the rails carry on down
    seat_rect = geometry.box(-(sleeve_x + c_seat), v_seat,
                             sleeve_x + c_seat, v_rt + 5.0)
    slots = geometry.box(x_i - c_sl, v_e, x_o + c_sl, v_rt + 5.0).union(
        geometry.box(-x_o - c_sl, v_e, -x_i + c_sl, v_rt + 5.0))
    receiver_section = receiver.difference(seat_rect).difference(slots)

    # --- clamps ---------------------------------------------------------------
    t_c = g("railrod.clamp_thickness")
    t_a = g("railrod.arm_thickness")
    R_o = Rb + t_c
    gap = g("railrod.lug_gap")
    d_b = g("railrod.bolt_diameter")
    r_hole = d_b / 2.0 + 0.15
    v_bolt = -(Rb + g("railrod.bolt_offset"))
    x_lug = g("railrod.lug_length")
    v_lb = v_bolt - r_hole - g("railrod.lug_wall")
    r_k = r_n - c_t

    if v_bolt + r_hole > -Rb - 0.8:
        raise LayoutError(
            f"the {2 * r_hole:.1f} mm bolt hole comes within 0.8 mm of the "
            "crankpin bore; raise bolt_offset")
    if x_lug < gap / 2.0 + 1.2 * d_b:
        raise LayoutError("the lugs are too short to engage the bolt thread")

    top_clip = v_rt
    band = envelope.buffer(c_g + t_a, quad_segs=_RES).difference(
        envelope.buffer(c_g, quad_segs=_RES))
    band = band.intersection(geometry.box(x_rc - 0.5, -2 * L, 2 * L,
                                          top_clip))
    ring_pts = np.vstack([
        [[0.0, 0.0]],
        arc_points((0.0, 0.0), R_o + 0.5, math.pi / 2.0 - theta_r
                   + math.radians(10.0), -math.pi / 2.0 - 0.01),
    ])
    sector_o = geometry.Polygon(ring_pts)
    annulus = disc((0.0, 0.0), R_o).difference(disc((0.0, 0.0), Rb))
    ring_body = annulus.intersection(sector_o)
    lug = geometry.box(gap / 2.0, v_lb, x_lug, -Rb * 0.75)

    blank = ops.unary_union([band, ring_body, lug])
    blank = _fill_concave(blank, max(2.0, t_a * 0.5))
    blank = blank.intersection(geometry.box(gap / 2.0, -2 * L, 2 * L,
                                            top_clip))

    keep_out = ops.unary_union([
        rail_right.buffer(c_t, quad_segs=_RES),
        envelope.buffer(c_g, quad_segs=_RES),
        disc((0.0, 0.0), Rb),
        geometry.box(-2 * L, -2 * L, gap / 2.0, 2 * L),
    ])
    arm = blank.difference(keep_out)
    tip = tongue.difference(rail_right.buffer(c_t, quad_segs=_RES))
    tip = _largest(tip) if not tip.is_empty else tip
    if tip.is_empty or tip.area < 0.2:
        raise LayoutError(
            "the rail cuts the whole tongue away: the channel arc is not "
            "aimed at the notch. Check swing_radius and swing_pivot_lean")

    # The tongue is NOT trimmed by the receiver -- it lives in the channel,
    # which is the tongue's own swept path, so it is already clear of it by
    # the channel clearance.
    tongue_safe = receiver.buffer(min(c_g, c_ch), quad_segs=_RES)
    final_keep_out = ops.unary_union([
        rail_right.buffer(c_t, quad_segs=_RES),
        tongue_safe,
        disc((0.0, 0.0), Rb),
        geometry.box(-2 * L, -2 * L, gap / 2.0, 2 * L),
    ])
    clamp = _round_convex(arm.union(tip), edge).difference(final_keep_out)
    clamp = _largest(clamp)
    if clamp.intersection(tip).area < 0.7 * tip.area:
        raise LayoutError(
            "the clamp tongue came out detached from the clamp arm; the "
            "channel exits the receiver too far from the arm")

    # --- sleeve -----------------------------------------------------------------
    sleeve_v0 = v_seat                    # the sleeve BOTTOMS in the seat
    eye_region = _fill_concave(eye.union(bars), r_t)
    sleeve_cut = eye_region.buffer(gap_s, quad_segs=_RES)
    column = geometry.box(-sleeve_x, sleeve_v0, sleeve_x, L)
    body = column.difference(sleeve_cut)
    sleeve_v1 = body.bounds[3]
    if sleeve_v1 - sleeve_v0 < 10.0:
        raise LayoutError("there is less than 10 mm of rail between the eye "
                          "and the big end for the sleeve")
    if sleeve_y > eye_width / 2.0 + 5.0:
        notes.append("the sleeve is deeper than the eye; check it clears "
                     "the piston bosses")

    # --- the sleeve's own architecture ------------------------------------
    # Two solid skins, one round each rail channel, joined across the web by
    # a graded sheet-gyroid lattice, with a thin shell on each big face and a
    # solid pad at the bottom where the sleeve bears on the receiver's seat.
    from . import lattice as lat

    core_kind = str(state["railrod.sleeve_core"])
    t_skin = g("railrod.sleeve_skin")
    t_face = g("railrod.sleeve_face_shell")
    allow = g("railrod.machining_allowance")
    core = ()
    port_stations: list = []
    port_depth = 0.0
    l_print = l_cells = l_evac = None
    cell = rho_mid = rho_end = grad_n = 0.0
    ports = False
    port_d = 0.0

    if core_kind == "sheet-gyroid":
        cell = g("railrod.lattice_cell")
        rho_mid = float(state["railrod.lattice_density"])
        rho_end = float(state["railrod.lattice_end_density"])
        grad_n = float(state["railrod.lattice_grading_exponent"])
        min_wall = g("railrod.lattice_min_wall")
        ports = bool(state["railrod.lattice_ports"]) and t_face > 0.0
        port_d = g("railrod.lattice_port_diameter")

        x_core = x_i - c_s - t_skin
        y_core = sleeve_y - t_face
        v_core0 = sleeve_v0 + max(t_face, 0.8)      # bearing pad on the seat
        # The top of the core is where the sleeve stops on the ROD AXIS.
        # The relief for the eye is a circle centred there, so the sleeve's
        # top face is lowest at x = 0 and climbs outward; taking the highest
        # point over the core's width would leave lattice hanging in air
        # above the middle of the part.
        if x_core > 0.0:
            axis = geometry.box(-0.25, v_core0, 0.25, L).difference(sleeve_cut)
            v_core1 = axis.bounds[3] if not axis.is_empty else v_core0
        else:
            v_core1 = v_core0
        if x_core <= 1.0:
            raise LayoutError(
                f"a {t_skin:.2f} mm skin on each rail channel leaves "
                f"{2 * max(x_core, 0.0):.2f} mm of web between them; there is "
                "no core left to put a lattice in. Thin sleeve_skin or widen "
                "rail_outer_span")
        if y_core <= 1.0:
            raise LayoutError(
                f"a {t_face:.2f} mm face shell leaves no core through the "
                "sleeve's depth")
        if v_core1 - v_core0 < 10.0:
            raise LayoutError("the lattice core is less than 10 mm long")
        core = (x_core, y_core, v_core0, v_core1)

        printed_skin = t_skin + allow
        if printed_skin >= x_i - c_s - 0.5:
            raise LayoutError(
                f"the rail skin printed at {printed_skin:.2f} mm (including "
                f"the {allow:.2f} mm machining allowance) closes the web "
                "before the lattice is even in it")

        l_print = lat.printability(rho_mid, rho_end, grad_n, cell, min_wall)
        if not l_print["printable"]:
            raise LayoutError(
                f"at {min(rho_mid, rho_end):.2f} relative density a "
                f"{cell:.1f} mm gyroid cell comes out "
                f"{l_print['nominal_sheet_mm']:.3f} mm nominal but only "
                f"{l_print['min_wall_mm']:.3f} mm at its thinnest, below the "
                f"{min_wall:.2f} mm the machine will print. Two things bind "
                "here at once: the lightest station, not the average, and "
                "the thinnest place on the wall, not its nominal. Raise "
                "lattice_density, enlarge lattice_cell, or buy a finer "
                "machine")

        l_cells = lat.cells_across(2 * x_core, 2 * y_core, cell)
        if not l_cells["homogenisation_valid"]:
            notes.append(
                f"the lattice core is only {l_cells['cells_across']:.1f} x "
                f"{l_cells['cells_through']:.1f} cells; below about four "
                "across, a lattice stops behaving like the continuum its "
                "effective properties describe, so the sleeve's stiffness "
                "here is indicative rather than predictive")

        # Powder ports: one staggered column per face, on twice the cell
        # pitch. Placed here rather than in the CAD so the solid model and
        # the mass model count the same holes.
        if ports and port_d > 0.0:
            pitch = 2.0 * cell
            v = v_core0 + pitch
            i = 0
            while v <= v_core1 - 0.5 * pitch:
                port_stations.append(
                    (0.5 * x_core * (1.0 if i % 2 else -1.0), v))
                v += pitch
                i += 1
        port_depth = sleeve_y - y_core + 1.0

        l_evac = lat.evacuation(2 * x_core, 2 * y_core, v_core1 - v_core0,
                                cell, min(rho_mid, rho_end), ports,
                                cell, port_d)
        if not l_evac["aperture_ok"]:
            raise LayoutError(
                f"at {max(rho_mid, rho_end):.2f} relative density a "
                f"{cell:.1f} mm cell leaves a {l_evac['aperture_mm']:.2f} mm "
                "channel through the lattice. Powder bridges below about "
                "0.5 mm and never comes out, and a sealed cell full of "
                "unfused metal is dead weight nothing will find. Enlarge "
                "lattice_cell or lighten the lattice")
        if not l_evac["path_ok"]:
            notes.append(
                f"powder has to travel {l_evac['path_mm']:.0f} mm through a "
                f"{l_evac['aperture_mm']:.2f} mm channel to get out "
                f"({l_evac['route']}); that is a long way to shake it. Turn "
                "lattice_ports on, or drop sleeve_face_shell to zero and "
                "leave the core open on both faces")
    elif core_kind != "solid":
        raise LayoutError(
            f"unknown sleeve_core {core_kind!r}; use 'sheet-gyroid' or 'solid'")

    return Layout(
        L=L, Rb=Rb, width=width, eye_r_out=eye_r_out, eye_r_in=eye_r_in,
        eye_width=eye_width, x_o=x_o, x_i=x_i, rail_depth=h, v_e=v_e,
        v_rt=v_rt, theta_r=theta_r, R_o=R_o, r_knob=r_k, r_swing=R_sw,
        v_bolt=v_bolt, r_hole=r_hole, x_lug=x_lug, v_lug_bottom=v_lb,
        lug_gap=gap, tip_clearance=c_t, guide_clearance=c_g,
        sleeve_x=sleeve_x, sleeve_y=sleeve_y, sleeve_v0=sleeve_v0,
        sleeve_v1=sleeve_v1, notch=notch, rail_right=rail_right,
        rails_upper=upper, v_joint=v_joint, envelope=envelope,
        receiver=receiver, receiver_section=receiver_section,
        channel_right=channel, tongue_right=tongue, seat_rect=seat_rect,
        slots=slots, clamp_right=clamp, clamp_left=mirror(clamp),
        sleeve_cut=sleeve_cut, pivot_pt=P, open_angle=open_a,
        channel_r_in=r_in, channel_r_out=r_out, channel_clearance=c_ch,
        v_seat=v_seat, x_rc=x_rc, slot_clearance=c_sl, seat_clearance=c_seat,
        slot_corner_radius=r_slot,
        sleeve_skin=t_skin, sleeve_face=t_face, sleeve_core_kind=core_kind,
        lattice_core=core, lattice_cell=cell, lattice_rho_mid=rho_mid,
        lattice_rho_end=rho_end, lattice_exponent=grad_n,
        lattice_ports=ports, lattice_port_diameter=port_d,
        port_stations=tuple(port_stations), port_depth=port_depth,
        lattice_print=l_print, lattice_cells=l_cells,
        lattice_evacuation=l_evac,
        v_mouth=(float(v_mouth_bot), float(v_mouth_top)), notes=notes)


# --- checks -----------------------------------------------------------------

def conformity(layout: Layout) -> dict:
    """How closely the clamp's male side follows the rail notch.

    Samples the notch surface from the deepest point down the ramp, and for
    each sample measures the distance to the clamp. A conforming fit reads
    the tip clearance everywhere along the tongue's nose and flank.
    """
    geometry, _, _ = _shapely()
    notch = layout.notch
    c = notch.centre
    samples = []
    for ang in np.linspace(math.pi, math.pi + notch.alpha, 12):
        samples.append(c + notch.r_top * _unit(ang))
    for t in np.linspace(0.05, 0.95, 10):
        samples.append(notch.T + t * (notch.F1 - notch.T))
    boundary = layout.clamp_right.exterior
    gaps = np.array([boundary.distance(geometry.Point(p)) for p in samples])
    return {
        "samples": len(samples),
        "gap_min_mm": float(gaps.min()),
        "gap_max_mm": float(gaps.max()),
        "target_mm": layout.tip_clearance,
        "conforms": bool(np.all(np.abs(gaps - layout.tip_clearance) < 0.02)),
    }


def guide_gap(layout: Layout) -> dict:
    """Clearance between each clamp and the receiver, closed."""
    gap = layout.clamp_right.distance(layout.receiver)
    return {"min_gap_mm": float(gap), "target_mm": layout.guide_clearance}


def _obstacles(layout: Layout):
    geometry, _, ops = _shapely()
    rail = layout.rail_right
    pin = disc((0.0, 0.0), layout.Rb)
    sleeve = geometry.box(-layout.sleeve_x, layout.sleeve_v0,
                          layout.sleeve_x, layout.L)
    if layout.sleeve_cut is not None:
        sleeve = sleeve.difference(layout.sleeve_cut)
    return {
        "rail": rail,
        "receiver": layout.receiver,
        "sleeve": sleeve,
        "crankpin": pin,
        "other clamp": layout.clamp_left,
    }


def swing_check(layout: Layout, max_angle_deg: float = 75.0,
                step_deg: float = 0.5, tolerance_mm2: float = 1e-3) -> dict:
    """Swing the right clamp open about the channel's centre and look for
    collisions.

    The pivot is the centre of the channel arc and lies on no part, so this
    is not a hinge that can be checked by inspection -- the clamp is held to
    the arc by the channel alone, and whether it can actually travel it is a
    question about every other solid. Opening is anticlockwise in the local
    frame (the tongue retracts out through the receiver wall and the lug
    swings out and away from the rod axis). The left clamp mirrors it.

    Returns the largest interference-free opening, what stops it, the
    opening at which the tongue leaves the rail notch (``release_deg``), the
    opening at which the tongue is clear of the receiver's outer face and
    the clamp can be threaded into its channel (``hook_in_angle_deg``), and
    whether that hook-in angle is inside the free swing.
    """
    _, affinity, _ = _shapely()
    pivot = tuple(layout.pivot)
    obstacles = _obstacles(layout)
    clamp = layout.clamp_right

    def hits(shape):
        return {name: shape.intersection(obs).area
                for name, obs in obstacles.items()
                if shape.intersects(obs)
                and shape.intersection(obs).area > tolerance_mm2}

    free = max_angle_deg
    blocker = None
    for ang in np.arange(step_deg, max_angle_deg + 1e-9, step_deg):
        rotated = affinity.rotate(clamp, ang, origin=pivot)
        collision = hits(rotated)
        if collision:
            free = ang - step_deg
            blocker = max(collision, key=collision.get)
            break

    # How far back the tongue has to come. Its innermost point travels
    # outward as the clamp opens, so both questions are the same one asked
    # at two stations: past the rail's outside face it is out of the notch,
    # past the receiver's outside face it is out of the channel, which is
    # the position the clamp is threaded in at.
    tongue = layout.tongue_right
    release = hook = None
    if tongue is not None and not tongue.is_empty:
        for ang in np.arange(0.0, max_angle_deg + 1e-9, step_deg):
            x_min = affinity.rotate(tongue, ang, origin=pivot).bounds[0]
            if release is None and x_min >= layout.x_o:
                release = float(ang)
            if x_min >= layout.x_rc:
                hook = float(ang)
                break

    closed = hits(clamp)
    return {
        "free_opening_deg": float(free),
        "limited_by": blocker,
        "release_deg": release,
        "hook_in_angle_deg": hook,
        "design_opening_deg": float(math.degrees(layout.open_angle)),
        "assemblable": (hook is not None and hook <= free and not closed),
        "closed_interference": {k: float(v) for k, v in closed.items()},
        "pivot_mm": [float(pivot[0]), float(pivot[1])],
    }


def section_payload(layout: Layout, swing_deg: float = 0.0) -> dict:
    """Every profile as point lists, for the 2D section view."""
    _, affinity, _ = _shapely()

    def rings(geom):
        if geom.is_empty:
            return []
        polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
        out = []
        for p in polys:
            out.append({"outer": np.round(np.asarray(p.exterior.coords), 4)
                        .tolist(),
                        "holes": [np.round(np.asarray(r.coords), 4).tolist()
                                  for r in p.interiors]})
        return out

    geometry, _, _ = _shapely()
    pivot = tuple(layout.pivot)
    right = affinity.rotate(layout.clamp_right, swing_deg, origin=pivot)
    left = affinity.rotate(layout.clamp_left, -swing_deg,
                           origin=(-pivot[0], pivot[1]))
    lower_rails = layout.rail_right.union(layout.rail_left())
    rails = lower_rails.union(layout.rails_upper)
    bolt = geometry.box(-layout.x_lug, layout.v_bolt - layout.r_hole + 0.15,
                        layout.x_lug, layout.v_bolt + layout.r_hole - 0.15)
    sleeve = geometry.box(-layout.sleeve_x, layout.sleeve_v0, layout.sleeve_x,
                          layout.L).difference(layout.sleeve_cut).difference(
        rails.buffer(0.02))
    n = layout.notch
    channels = rings(layout.channel_right.union(
        mirror(layout.channel_right))) if layout.channel_right else []
    return {
        "frame": "local mm: x across the rod, v up from the crankpin centre",
        "rod_length_mm": layout.L,
        "parts": {
            "rails": rings(rails),
            "sleeve": rings(sleeve),
            "receiver": rings(layout.receiver_section),
            "clamp_right": rings(right),
            "clamp_left": rings(left),
            "bolt": rings(bolt),
            "crankpin": rings(disc((0.0, 0.0), layout.Rb, step=0.5)),
            "eye_bore": rings(disc((0.0, layout.L), layout.eye_r_in, step=0.5)),
        },
        "notch": {k: np.round(getattr(n, k), 4).tolist()
                  for k in ("centre", "A", "D", "T", "F1", "F2")},
        "notch_dims": {
            "depth_mm": n.depth, "ramp_angle_deg": math.degrees(n.alpha),
            "top_radius_mm": n.r_top, "lower_radius_mm": n.r_low,
            "length_mm": n.length, "ramp_length_mm": n.ramp_length,
            "land_mm": float(n.F2[1] - layout.v_e),
        },
        "pivot": [float(pivot[0]), float(pivot[1])],
        "swing_deg": swing_deg,
        "bolt_axis_v_mm": layout.v_bolt,
        "channel": channels,
        "swing": {
            "pivot_mm": [float(pivot[0]), float(pivot[1])],
            "radius_mm": (layout.channel_r_in + layout.channel_r_out) / 2.0,
            "channel_thickness_mm": layout.channel_r_out - layout.channel_r_in,
            "channel_width_mm": layout.rail_depth,
            "open_angle_deg": math.degrees(layout.open_angle),
            "mouth_v_mm": [float(layout.v_mouth[0]), float(layout.v_mouth[1])],
        },
        "socket": {
            "seat_floor_v_mm": layout.v_seat,
            "top_face_v_mm": layout.v_rt,
            "rail_foot_v_mm": layout.v_e,
            "rail_engagement_mm": layout.v_seat - layout.v_e,
            "seat_depth_mm": layout.v_rt - layout.v_seat,
            "half_width_mm": layout.x_rc,
        },
    }
