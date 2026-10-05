"""Finite element analysis of each rail-rod part, through the whole cycle.

Every part is meshed and solved ON ITS OWN, loaded by the contact forces the
network in :mod:`psrt.railrod.network` says its neighbours put on it at each
crank angle. That is what "analyse each part independently through all crank
angles" means in practice, and it is how the parts' stresses can be read
separately even though the load paths between them switch around the cycle.

One solve per load pattern, not per crank angle
-----------------------------------------------
The load on a part is never one pattern scaled up and down -- the rail foot
carries firing, the notch carries overlap, the preload sits under both -- so
the conventional cases' trick of scaling a single field does not apply. What
does apply is superposition. Each part gets a set of UNIT load cases, one per
contact patch and direction (a 1 N bearing on the ramp, a 1 N bearing on the
floor, a 1 m/s^2 body force...), solved once each against a single
factorisation of the stiffness matrix. The stress at any crank angle is then

    sigma(theta) = sum_k c_k(theta) * sigma_k

with the coefficients c_k(theta) read straight off the network's contact
forces. That is exact for linear elasticity, costs one factorisation and a
dozen back-substitutions, and gives the full tensor at every angle -- so the
cycle peak and a Goodman fatigue factor come out for every element, not just
the von Mises at one angle.

Support without invented stress
-------------------------------
The loads the network hands a part are in equilibrium, so the part does not
need holding -- only its rigid-body motion needs removing. Each part is
restrained at three nodes in the classic 3-2-1 pattern, which is statically
determinate: for balanced loads those nodes react nothing. What they DO react
is reported, per crank angle, as the equilibrium residual; it measures how far
the FEA's distributed patches sit from the network's point contacts, and the
elements touching those nodes are left out of the headline numbers.

The exception is rod whip -- the rails' and sleeve's own transverse inertia --
which is antisymmetric and has no place in the symmetric network. It is
solved as its own case with the rails pinned at the eye and held at the feet,
and added on top.

What this does not model: contact. Each patch is a cosine-distributed
pressure delivering the network's force, not a contact solution, so peak
pressures right at the edge of a patch are indicative. Thread engagement in
the right clamp is a uniform shear on the tapped hole. Temperature is taken
from the thermal map for the rod.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from .. import materials as materials_mod
from ..fatigue import endurance_limit
from .analysis import analyse
from .cad import PART_LABELS, part_material

PARTS_WITH_FEA = ("rr_rails", "rr_sleeve", "rr_receiver", "rr_clamp_right",
                  "rr_clamp_left", "rr_bolt")
TOL = 0.03          # mm: how close a facet midpoint must be to a surface


@dataclass
class UnitCase:
    name: str
    kind: str                       # bearing | traction | body
    parts: list                     # [(selector_local, axis3)] for surface
    weight: object = None           # body: fn(x_mm, y_mm, v_mm) -> weight
    direction: tuple = (0, 0, 0)    # body direction, rod frame
    bc: str = "free"                # which restraint set
    label: str = ""
    coeff: object = None            # (n_theta,) multiplier on this case
    contact: int = -1               # network contact this case stands for


# --- selectors in local mm --------------------------------------------------------

def _local(layout, x):
    """(3, n) metres in the rod frame -> x, y, v in local mm."""
    xm, ym, zm = x[0] * 1e3, x[1] * 1e3, x[2] * 1e3
    return xm, ym, layout.L - zm


def sleeve_material_field(state, layout, material):
    """Properties of the sleeve at a point, alloy or lattice.

    The sleeve is one body of two materials: solid alloy in the rail skins,
    the face shells and the bearing pad, and the homogenised sheet gyroid
    inside the core box, whose density is graded along the sleeve's length.
    Returns a callable of local (x, y, v) in mm giving the mask, Young's
    modulus, Poisson ratio, density and the fraction of the alloy's strength
    the material has there. Returns None for a sleeve that is solid through.
    """
    if (layout.sleeve_core_kind != "sheet-gyroid"
            or not layout.lattice_core):
        return None
    from . import lattice as lattice_mod

    xc, yc, v0, v1 = layout.lattice_core
    mid = 0.5 * (v0 + v1)
    half = max(0.5 * (v1 - v0), 1e-9)
    c_e = state["railrod.lattice_modulus_coefficient"]
    n_e = state["railrod.lattice_modulus_exponent"]
    c_s = state["railrod.lattice_strength_coefficient"]
    n_s = state["railrod.lattice_strength_exponent"]

    def field(xm, ym, vm):
        inside = ((np.abs(xm) <= xc) & (np.abs(ym) <= yc)
                  & (vm >= v0) & (vm <= v1))
        rho = np.clip(lattice_mod.grading(
            (vm - mid) / half, layout.lattice_rho_mid,
            layout.lattice_rho_end, layout.lattice_exponent), 1e-3, 1.0)
        nu_l = 0.20 + (material.poisson - 0.20) * rho
        return (
            inside,
            np.where(inside, material.youngs_modulus * c_e * rho ** n_e,
                     material.youngs_modulus),
            np.where(inside, nu_l, material.poisson),
            np.where(inside, material.density * rho, material.density),
            np.where(inside, c_s * rho ** n_s, 1.0),
        )
    return field


def _lame_from(e, nu):
    return (e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)), e / (2.0 * (1.0 + nu)))


def _seg_dist(px, pv, a, b):
    ax, av = a
    bx, bv = b
    dx, dv = bx - ax, bv - av
    t = np.clip(((px - ax) * dx + (pv - av) * dv) / (dx * dx + dv * dv), 0, 1)
    return np.hypot(px - (ax + t * dx), pv - (av + t * dv))


def _mirror_sel(sel):
    return lambda x, y, v: sel(-x, y, v)


def _wrap(layout, sel):
    def f(x):
        return sel(*_local(layout, x))
    return f


# --- the unit cases per part ---------------------------------------------------
#
# Every contact in the network becomes its own unit case, on a patch of the
# real surface centred on the contact's point and loaded along the contact's
# normal. The first version grouped them -- one case for the whole clamp bore,
# one for the whole scoop -- and the moment of a force spread over a whole
# surface is not the moment of the same force where the network put it. The
# three restraint nodes then had to react the difference, and did so with
# gigapascals. Patch-per-contact keeps every line of action where the network
# resolved it, so what the restraints see is only the small offset between a
# point contact and a cosine pressure centred on it.

def _surfaces(state, lay) -> dict:
    """Named surface selectors, local mm, RIGHT side (mirror for the left)."""
    n = lay.notch
    C = n.centre
    c_t = lay.tip_clearance
    # The rail slots are cut at the SLOT clearance, not the tip clearance:
    # look for the slot walls anywhere else and the selector finds nothing.
    c_p = max(lay.slot_clearance, 0.02)
    off = c_t * n.normal
    T_k, F1_k = n.T + off, n.F1 + off
    a_lo = math.atan2(n.A[1] - C[1], n.A[0] - C[0]) - 0.05
    a_hi = math.pi + n.alpha + 0.05
    import shapely
    # The clamp's guide surfaces are now the walls of the swing channel as
    # well as the receiver's outer flank, and the channel is cut OUT of the
    # receiver -- so this has to be taken from the envelope, the receiver
    # before its own channels were cut, or the tongue's flanks select
    # nothing at all.
    near_receiver = lay.envelope.buffer(lay.guide_clearance + TOL)

    def angle(x, v):
        ang = np.arctan2(v - C[1], x - C[0])
        return np.where(ang < 0, ang + 2 * math.pi, ang)

    s = {}
    # rails
    s["rail_feet"] = lambda x, y, v: ((v < lay.v_e + 0.02)
                                      & (x >= lay.x_i - 0.01)
                                      & (x <= lay.x_o + 0.01))
    s["rail_flank"] = lambda x, y, v: _seg_dist(x, v, n.T, n.F1) < TOL
    s["rail_socket"] = lambda x, y, v: (
        (np.abs(np.hypot(x - C[0], v - C[1]) - n.r_top) < TOL)
        & (angle(x, v) >= a_lo) & (angle(x, v) <= a_hi))
    s["rail_wall"] = lambda x, y, v: ((np.abs(x - lay.x_i) < TOL)
                                      & (v >= lay.v_e - 0.01)
                                      & (v <= lay.v_seat + 0.01))
    # receiver
    h = lay.rail_depth / 2.0 + c_p
    s["rec_floor"] = lambda x, y, v: ((np.abs(v - lay.v_e) < 0.02)
                                      & (np.abs(y) <= h + 0.01)
                                      & (x >= lay.x_i - c_p - 0.01)
                                      & (x <= lay.x_o + c_p + 0.01))
    s["rec_web"] = lambda x, y, v: ((np.abs(x - (lay.x_i - c_p)) < TOL)
                                    & (v >= lay.v_e - 0.01)
                                    & (v <= lay.v_seat + 0.01)
                                    & (np.abs(y) <= h + 0.01))
    s["rec_bore"] = lambda x, y, v: np.hypot(x, v) < lay.Rb + 0.05
    s["rec_outer"] = lambda x, y, v: ((x > lay.x_o + c_p + 0.02)
                                      & (np.hypot(x, v) > lay.Rb + 0.1))
    # clamp (right)
    s["cl_flank"] = lambda x, y, v: _seg_dist(x, v, T_k, F1_k) < TOL
    s["cl_knob"] = lambda x, y, v: (
        (np.abs(np.hypot(x - C[0], v - C[1]) - lay.r_knob) < TOL)
        & (x < C[0] + 0.2))
    s["cl_guide"] = lambda x, y, v: (
        shapely.contains_xy(near_receiver, x, v)
        & (np.abs(x) > lay.x_o + c_p)
        & (np.hypot(x, v) > lay.Rb + 0.1))
    s["cl_pin"] = lambda x, y, v: np.hypot(x, v) < lay.Rb + 0.05
    s["cl_lug"] = lambda x, y, v: np.abs(x - lay.lug_gap / 2.0) < 0.01
    return s


# which surface of which part each network contact group loads
_GROUP_SURFACE = {
    ("rr_rails", "web"): "rail_wall",
    ("rr_receiver", "web"): "rec_web",
    ("rr_rails", "floor"): "rail_feet",
    ("rr_rails", "flank"): "rail_flank",
    ("rr_rails", "knob"): "rail_socket",
    ("rr_receiver", "floor"): "rec_floor",
    ("rr_receiver", "guide"): "rec_outer",
    ("rr_receiver", "receiver_pin"): "rec_bore",
    ("rr_clamp", "flank"): "cl_flank",
    ("rr_clamp", "knob"): "cl_knob",
    ("rr_clamp", "guide"): "cl_guide",
    ("rr_clamp", "clamp_pin"): "cl_pin",
    ("rr_clamp", "lug"): "cl_lug",
}

_BODY = {"rr_rails": "rail", "rr_receiver": "receiver",
         "rr_clamp_right": "clamp", "rr_clamp_left": "clamp"}


def _window(sel, point, radius):
    px, pv = float(point[0]), float(point[1])

    def f(x, y, v):
        return sel(x, y, v) & (np.hypot(x - px, v - pv) <= radius)
    return f


def _to3(vec2, mirror=False):
    """Local (x, v) direction -> rod-frame 3D (x, y, z); z = -v."""
    x = -vec2[0] if mirror else vec2[0]
    return (float(x), 0.0, float(-vec2[1]))


def _contact_cases(state, analysis, part) -> list:
    lay = analysis.layout
    cyc = analysis.cycle
    net = cyc.network
    surf = _surfaces(state, lay)
    body = _BODY[part]
    key = "rr_clamp" if part.startswith("rr_clamp") else part
    mirror_part = part == "rr_clamp_left"
    symmetric = part in ("rr_rails", "rr_receiver")

    by_group: dict = {}
    for i, c in enumerate(net.contacts):
        if body not in (c.a, c.b):
            continue
        by_group.setdefault(c.group, []).append(i)

    cases = []
    for group, idx in by_group.items():
        sname = _GROUP_SURFACE.get((key, group))
        if sname is None:
            continue
        sel = surf[sname]
        pts = np.array([net.contacts[i].point for i in idx])
        if len(idx) > 1:
            gaps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            spacing = np.concatenate([[gaps[0]], np.minimum(gaps[:-1],
                                                            gaps[1:]),
                                      [gaps[-1]]]) if len(gaps) > 1 else \
                np.array([gaps[0], gaps[0]])
        else:
            spacing = np.array([1e3])
        for k, i in enumerate(idx):
            c = net.contacts[i]
            sign = 1.0 if c.b == body else -1.0
            direction = sign * c.normal
            radius = max(0.6 * spacing[k], 0.5)
            win = _window(sel, c.point, radius)
            coeff = cyc.forces[:, i]
            if group in ("floor", "web"):
                win = sel                       # the whole face
            if symmetric:
                patches = [(win, _to3(direction)),
                           (_mirror_sel(win), _to3(direction, mirror=True))]
                # a contact ON the symmetry line appears once, not twice
                if abs(c.point[0]) < 1e-6:
                    patches = [(win, _to3(direction))]
                    coeff = 2.0 * coeff
            elif mirror_part:
                patches = [(_mirror_sel(win), _to3(direction, mirror=True))]
            else:
                patches = [(win, _to3(direction))]
            cases.append(UnitCase(
                f"{group}_{k}", "bearing", patches, coeff=coeff, contact=i,
                label=f"{group} contact {k + 1} of {len(idx)}"))
    return cases


def _web_from_balance(cases, analysis, wall):
    """In the rigid network the rails cannot move sideways, so the web
    contact never closes and the inward push of the clamp tips on the rail
    feet has no recorded reaction. Read it off the rails' sideways balance
    instead. The coupled network resolves it properly and needs no help."""
    if getattr(analysis, "coupled", False):
        return cases
    for case in cases:
        if case.name.startswith("web_"):
            case.coeff = np.maximum(np.asarray(case.coeff) , 0.0) + np.maximum(
                wall, 0.0)
    return cases


def part_cases(state, analysis, part) -> list:
    """Every unit case for one part, each carrying its coefficient series."""
    lay = analysis.layout
    cyc = analysis.cycle
    n = lay.notch
    f = cyc.f_rod
    pos = lambda a: np.maximum(a, 0.0)                     # noqa: E731
    from .network import rod_kinematics
    kin = rod_kinematics(state, cyc.theta)

    def w_small(x, y, v):
        return 1.0 - (lay.L - v) / lay.L

    def w_big(x, y, v):
        return (lay.L - v) / lay.L

    def one(x, y, v):
        return 1.0 + 0.0 * x

    if part == "rr_rails":
        r_in, ew = lay.eye_r_in, lay.eye_width / 2.0

        def eye(side):
            return lambda x, y, v: ((np.hypot(x, v - lay.L) < r_in + 0.3)
                                    & (np.abs(y) <= ew)
                                    & ((v < lay.L) if side < 0
                                       else (v > lay.L)))
        # The clamp contacts push each rail foot inward; the pocket web
        # between the feet pushes back. The network holds the rails rigid
        # sideways, so the reaction is read from the contact forces.
        net = cyc.network
        wall = np.zeros(len(cyc.theta))
        for i, c in enumerate(net.contacts):
            if "rail" in (c.a, c.b) and c.group in ("flank", "knob"):
                wall += cyc.forces[:, i] * c.normal[0]
        cases = [
            UnitCase("eye_comp", "bearing", [(eye(-1), (0, 0, 1))],
                     coeff=pos(f), label="pin driving the eye toward the crank"),
            UnitCase("eye_tens", "bearing", [(eye(+1), (0, 0, -1))],
                     coeff=pos(-f), label="pin pulling the eye toward the piston"),
        ]
        cases += _web_from_balance(_contact_cases(state, analysis, part),
                                   analysis, wall)
        cases += [
            UnitCase("whip_small", "body", [], weight=w_small,
                     direction=(1, 0, 0), bc="whip", coeff=-cyc.accel_small_x,
                     label="transverse inertia, small-end share"),
            UnitCase("whip_big", "body", [], weight=w_big,
                     direction=(1, 0, 0), bc="whip", coeff=-cyc.accel_bigend_x,
                     label="transverse inertia, big-end share"),
        ]
        return cases

    if part == "rr_receiver":
        net = cyc.network
        wall = np.zeros(len(cyc.theta))
        for i, c in enumerate(net.contacts):
            if "rail" in (c.a, c.b) and c.group in ("flank", "knob"):
                wall += cyc.forces[:, i] * c.normal[0]
        cases = _web_from_balance(_contact_cases(state, analysis, part),
                                  analysis, wall)
        cases.append(UnitCase("inertia", "body", [], weight=one,
                              direction=(0, 0, 1), coeff=cyc.accel_bigend_v,
                              label="receiver inertia along the rod"))
        return cases

    if part in ("rr_clamp_right", "rr_clamp_left"):
        side = "right" if part.endswith("right") else "left"
        d_b = state["railrod.bolt_diameter"] * 1e3
        pitch = state["railrod.bolt_pitch"] * 1e3
        r_minor = (d_b - 1.0825 * pitch) / 2.0
        cases = _contact_cases(state, analysis, part)
        if side == "right":
            def hole(x, y, v):
                return ((np.hypot(v - lay.v_bolt, y) < r_minor + 0.05)
                        & (x > lay.lug_gap / 2.0 - 0.01)
                        & (x < lay.x_lug + 0.01))
            cases.append(UnitCase("bolt", "traction", [(hole, (-1, 0, 0))],
                                  coeff=cyc.bolt,
                                  label="bolt thread pulling the lug inward"))
        else:
            def head(x, y, v):
                rr = np.hypot(v - lay.v_bolt, y)
                return ((np.abs(x + lay.x_lug) < 0.01) & (rr < 0.75 * d_b)
                        & (rr > lay.r_hole - 0.01))
            cases.append(UnitCase("bolt", "bearing", [(head, (1, 0, 0))],
                                  coeff=cyc.bolt,
                                  label="bolt head pulling the lug inward"))
        cases.append(UnitCase("inertia", "body", [], weight=one,
                              direction=(0, 0, 1), coeff=cyc.accel_bigend_v,
                              label="clamp inertia along the rod"))
        return cases

    if part == "rr_sleeve":
        c_s = state["railrod.sleeve_clearance"] * 1e3
        h = lay.rail_depth / 2.0 + c_s
        v0, v1 = lay.sleeve_v0, lay.sleeve_v1
        span = v1 - v0

        def middle(x, y, v):
            return ((np.abs(y) <= h + 0.01)
                    & (np.abs(np.abs(x) - (lay.x_o + c_s)) < TOL)
                    & (np.abs(v - (v0 + v1) / 2) < 0.15 * span))
        beta = state["railrod.bracing_fraction"]
        return [
            UnitCase("whip_small", "body", [], weight=w_small,
                     direction=(1, 0, 0), coeff=-cyc.accel_small_x,
                     label="transverse inertia, small-end share"),
            UnitCase("whip_big", "body", [], weight=w_big,
                     direction=(1, 0, 0), coeff=-cyc.accel_bigend_x,
                     label="transverse inertia, big-end share"),
            UnitCase("axial_small", "body", [], weight=w_small,
                     direction=(0, 0, 1), coeff=kin["a_small_v"],
                     label="axial inertia, small-end share"),
            UnitCase("axial_big", "body", [], weight=w_big,
                     direction=(0, 0, 1), coeff=cyc.accel_bigend_v,
                     label="axial inertia, big-end share"),
            UnitCase("bracing", "bearing", [(middle, (1, 0, 0))],
                     coeff=beta * pos(f),
                     label="rails bowing sideways against the sleeve"),
        ]

    if part == "rr_bolt":
        d_b = state["railrod.bolt_diameter"] * 1e3
        r = d_b / 2.0

        def thread(x, y, v):
            return ((np.hypot(v - lay.v_bolt, y) > r - 0.05) & (x > 0.2)
                    & (x < lay.x_lug - 0.6))

        def head(x, y, v):
            rr = np.hypot(v - lay.v_bolt, y)
            return (np.abs(x + lay.x_lug) < 0.01) & (rr > r + 0.01)
        return [UnitCase("head", "bearing", [(head, (-1, 0, 0))],
                         coeff=cyc.bolt, label="head seated on the left lug"),
                UnitCase("thread", "traction", [(thread, (1, 0, 0))],
                         coeff=cyc.bolt,
                         label="thread engaged in the right lug")]
    raise KeyError(part)


def _sleeve_channel(state, lay):
    c_s = state["railrod.sleeve_clearance"] * 1e3
    h = lay.rail_depth / 2.0 + c_s

    def channel(x, y, v):
        return ((np.abs(y) <= h + 0.01)
                & ((np.abs(np.abs(x) - (lay.x_o + c_s)) < TOL)
                   | (np.abs(np.abs(x) - (lay.x_i - c_s)) < TOL)))
    return channel


# --- assembly and solve --------------------------------------------------------------

@dataclass
class PartResult:
    part: str
    mesh: object
    material: object
    cases: list
    tensors: np.ndarray            # (n_cases, n_elements, 6) Pa per unit
    resultants: np.ndarray         # (n_cases, 3) N per unit, what was applied
    moments: np.ndarray            # (n_cases, 3) N m per unit, about centroid
    coeffs: np.ndarray             # (n_theta, n_cases)
    theta: np.ndarray
    excluded: np.ndarray           # element mask near restraint nodes
    notes: list = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    # For a part that is not one material throughout: which elements are the
    # softer region, and what fraction of the parent alloy's strength they
    # have. 1.0 everywhere for an ordinary part.
    region: np.ndarray | None = None
    strength_ratio: np.ndarray | None = None
    region_label: str = ""

    # -- cycle statistics ---------------------------------------------------
    def stress_at(self, i: int) -> np.ndarray:
        """(n_elements, 6) tensor at crank index i."""
        return np.tensordot(self.coeffs[i], self.tensors.astype(float),
                            axes=1)

    def cycle_stats(self, chunk: int = 48) -> dict:
        n_t = len(self.theta)
        n_e = self.tensors.shape[1]
        peak = np.zeros(n_e)
        peak_at = np.zeros(n_e, dtype=int)
        s_max = np.full(n_e, -np.inf)
        s_min = np.full(n_e, np.inf)
        for a in range(0, n_t, chunk):
            c = self.coeffs[a:a + chunk]
            s = np.einsum("tk,kec->tec", c, self.tensors,
                          dtype=np.float64)
            vm = _von_mises(s)
            signed = vm * _sign(s)
            i = vm.argmax(axis=0)
            better = vm[i, np.arange(n_e)] > peak
            peak = np.where(better, vm[i, np.arange(n_e)], peak)
            peak_at = np.where(better, a + i, peak_at)
            s_max = np.maximum(s_max, signed.max(axis=0))
            s_min = np.minimum(s_min, signed.min(axis=0))
        return {"peak": peak, "peak_at": peak_at, "s_max": s_max,
                "s_min": s_min}


def _von_mises(s):
    xx, yy, zz, xy, yz, zx = (s[..., i] for i in range(6))
    return np.sqrt(0.5 * ((xx - yy) ** 2 + (yy - zz) ** 2 + (zz - xx) ** 2)
                   + 3.0 * (xy ** 2 + yz ** 2 + zx ** 2))


def _sign(s):
    """Sign for a 'signed von Mises': that of the hydrostatic stress, so a
    tensile-dominated state reads positive and a compressive one negative."""
    tr = s[..., 0] + s[..., 1] + s[..., 2]
    return np.where(tr >= 0, 1.0, -1.0)


def _voigt(t):
    """(n, 3, 3) -> (n, 6): xx yy zz xy yz zx."""
    return np.stack([t[:, 0, 0], t[:, 1, 1], t[:, 2, 2], t[:, 0, 1],
                     t[:, 1, 2], t[:, 2, 0]], axis=1)


def _nodal_volume(mesh) -> np.ndarray:
    tets = mesh.elements.T
    c = mesh.points.T[tets]
    vol = np.abs(np.einsum("ij,ij->i", c[:, 1] - c[:, 0],
                           np.cross(c[:, 2] - c[:, 0], c[:, 3] - c[:, 0]))) / 6
    out = np.zeros(mesh.n_nodes)
    for k in range(4):
        np.add.at(out, tets[:, k], vol / 4.0)
    return out


def _rigid_modes(points) -> np.ndarray:
    """(3N, 6): three translations and three rotations about the centroid,
    node-major like the solver's degrees of freedom."""
    r = (points - points.mean(axis=1, keepdims=True)).T         # (N, 3)
    n = r.shape[0]
    phi = np.zeros((n, 3, 6))
    for k in range(3):
        phi[:, k, k] = 1.0
    for k, e in enumerate(np.eye(3)):
        phi[:, :, 3 + k] = np.cross(e, r)
    return phi.reshape(3 * n, 6)


def _three_two_one(points):
    """Three node indices for a statically determinate restraint."""
    centre = points.mean(axis=1)
    span = points.max(axis=1) - points.min(axis=1)

    def nearest(p):
        return int(np.argmin(np.linalg.norm(points - p[:, None], axis=0)))

    n1 = nearest(centre)
    n2 = nearest(centre + np.array([0.3 * span[0], 0, 0]))
    n3 = nearest(centre + np.array([0, 0, 0.3 * span[2]]))
    if len({n1, n2, n3}) < 3:
        n2 = nearest(centre + np.array([0.45 * span[0], 0, 0.1 * span[2]]))
        n3 = nearest(centre + np.array([-0.1 * span[0], 0, 0.45 * span[2]]))
    return n1, n2, n3


@dataclass
class UnitSolution:
    """One part meshed and solved for every unit case. Independent of the
    crank angle and of which network supplies the coefficients."""
    part: str
    mesh: object
    material: object
    cases: list
    loads: list                    # applied load vectors, before relief (N)
    relieved: list                 # the same after inertia relief
    displacements: np.ndarray      # (n_cases, 3N) metres per unit
    tensors: np.ndarray            # (n_cases, n_elements, 6) Pa per unit
    resultants: np.ndarray
    moments: np.ndarray
    excluded: np.ndarray
    notes: list
    timings: dict
    region: object = None          # element mask of the softer material
    strength_ratio: object = None  # its share of the alloy's strength
    region_label: str = ""


_UNIT_CACHE: dict = {}
# Each entry holds a mesh, every unit case's displacement field and its
# stress tensors -- hundreds of megabytes for a part meshed finely. Eight of
# those will exhaust a small machine before anything says why, and the
# sleeve's mesh grew when its lattice core joined the model, so this is
# deliberately short. Re-solving is slow; running out of memory is worse.
_UNIT_CACHE_LIMIT = 3


def unit_solve(state, part: str, target_elements: int = 25_000,
               analysis=None, use_cache: bool = True) -> UnitSolution:
    """Mesh one rail-rod part and solve all of its unit load cases."""
    from scipy.sparse.linalg import splu
    from skfem import (Basis, ElementTetP1, FacetBasis, LinearForm, MeshTet,
                       asm)
    from skfem.models.elasticity import linear_elasticity
    try:
        from skfem import ElementVector
    except ImportError:                                   # pragma: no cover
        from skfem import ElementVectorH1 as ElementVector

    from ..fea.mesh import tet_mesh
    from ..fea.solve import Bearing, _bearing_load, _lame, stress_tensors

    if part not in PARTS_WITH_FEA:
        raise KeyError(f"no rail-rod FEA for {part!r}")
    from ..geometry import fingerprint as geometry_fingerprint
    # Keyed on the GEOMETRY, not the whole design: a unit solution does not
    # depend on preload, fit or operating point, only on shape and material,
    # so a preload study reuses it.
    key = (geometry_fingerprint(state), part,
           int(target_elements))
    if use_cache and key in _UNIT_CACHE:
        return _UNIT_CACHE[key]

    timings = {}
    t0 = time.time()
    analysis = analysis or analyse(state)
    lay = analysis.layout

    material = materials_mod.get(part_material(state, part))
    field_of = (sleeve_material_field(state, lay, material)
                if part == "rr_sleeve" else None)
    whole = None
    if field_of is not None:
        from .cad import build_sleeve_fea
        whole = build_sleeve_fea(lay, state["railrod.sleeve_clearance"] * 1e3)
    mesh = tet_mesh(state, part, target_elements=target_elements, solid=whole)
    timings["mesh_s"] = time.time() - t0

    notes: list[str] = []
    cases = part_cases(state, analysis, part)
    extra = _sleeve_channel(state, lay) if part == "rr_sleeve" else None

    t1 = time.time()
    m = MeshTet(mesh.points, mesh.elements)
    basis = Basis(m, ElementVector(ElementTetP1()))
    centroids = mesh.points.T[mesh.elements.T].mean(axis=1).T
    region = strength_ratio = lame_e = None
    if field_of is None:
        lam, mu = _lame(material)
    else:
        # Lame parameters at every quadrature point, so the interface between
        # skin and core falls where the geometry puts it rather than on the
        # nearest element boundary.
        gx = np.asarray(basis.global_coordinates())
        shape = gx.shape[1:]
        inside, e_q, nu_q, _, _ = field_of(*_local(lay, gx.reshape(3, -1)))
        lam, mu = (v.reshape(shape) for v in _lame_from(e_q, nu_q))
        # and once per element, for recovering stress from displacement
        region, e_c, nu_c, _, strength_ratio = field_of(
            *_local(lay, centroids))
        lame_e = _lame_from(e_c, nu_c)
        notes.append(
            f"{int(region.sum())} of {mesh.n_elements} elements are the "
            "lattice core, carried as a homogenised continuum")
    K = asm(linear_elasticity(lam, mu), basis).tocsr()
    timings["assemble_s"] = time.time() - t1

    rho = material.density
    rho_field = None
    if field_of is not None:
        def rho_field(xm, ym, vm):
            return field_of(xm, ym, vm)[3]
    if part == "rr_rails":
        # the sleeve rides on the rails, so its transverse inertia reaches
        # them: smeared over the rails' length as extra density
        rho_whip = rho * (1.0 + analysis.masses["rr_sleeve"]
                          / max(analysis.masses["rr_rails"], 1e-9))
    else:
        rho_whip = rho

    def assemble(case):
        if case.kind == "body":
            d = np.asarray(case.direction, dtype=float)
            dens = rho_whip if case.name.startswith("whip") else rho
            weight = case.weight

            @LinearForm
            def body(v, w):
                xm, ym, vm = _local(lay, w.x)
                local_rho = dens if rho_field is None else rho_field(
                    xm, ym, vm)
                return local_rho * weight(xm, ym, vm) * sum(
                    d[i] * v[i] for i in range(3))

            return asm(body, basis)
        vec = np.zeros(basis.N)
        for sel, axis in case.parts:
            selector = _wrap(lay, sel)
            if case.kind == "bearing":
                vec = vec + _bearing_load(
                    m, basis, Bearing(selector, axis, 1.0, case.name), notes)
                continue
            facets = m.facets_satisfying(selector, boundaries_only=True)
            if facets.size == 0:
                raise ValueError(f"the {case.name} patch matched no facets")
            fb = FacetBasis(m, basis.elem, facets=facets)
            ax = np.asarray(axis, dtype=float)

            @LinearForm
            def trac(v, w):
                return sum(ax[i] * v[i] for i in range(3))

            part_vec = asm(trac, fb)
            total = float(part_vec.reshape(-1, 3).sum(axis=0) @ ax)
            vec = vec + part_vec / total
        return vec

    coeff_map = {c.name: np.asarray(c.coeff, dtype=float) for c in cases}
    load_scale = max(float(np.max(np.abs(v))) for v in coeff_map.values())
    loads, resultants, moments, kept = [], [], [], []
    centroid = mesh.points.mean(axis=1)
    arm = (mesh.points - centroid[:, None]).T
    for case in cases:
        try:
            vec = assemble(case)
        except ValueError as exc:
            # A patch that cannot take load in this direction is fine as
            # long as the network never asks it to.
            peak = float(np.max(np.abs(coeff_map[case.name])))
            if peak <= 1e-6 * max(load_scale, 1.0):
                notes.append(f"{case.name}: this surface cannot take load in "
                             "that direction and the network never asks it "
                             "to; skipped")
                continue
            raise ValueError(f"{part}, {case.name}: {exc} (the network asks "
                             f"this patch for up to {peak:.0f})") from exc
        loads.append(vec)
        nodal = vec.reshape(-1, 3)
        resultants.append(nodal.sum(axis=0))
        moments.append(np.cross(arm, nodal).sum(axis=0))
        kept.append(case)
    cases = kept
    notes = [n for n in notes if "kN spread over" not in n]

    # --- restraints -----------------------------------------------------------
    pts = mesh.points
    n1, n2, n3 = _three_two_one(pts)
    free_bc = np.array(sorted({3 * n1, 3 * n1 + 1, 3 * n1 + 2,
                               3 * n2 + 1, 3 * n2 + 2, 3 * n3 + 1}))
    bcs = {"free": free_bc}
    excluded_nodes = [n1, n2, n3]
    relieve = True
    if part == "rr_rails":
        xm, ym, vm = _local(lay, pts)
        eye_nodes = np.flatnonzero(np.hypot(xm, vm - lay.L)
                                   < lay.eye_r_in + 0.05)
        foot = np.flatnonzero((vm < lay.v_rt + 0.01)
                              & ((np.abs(np.abs(xm) - lay.x_i) < 0.02)
                                 | (np.abs(np.abs(xm) - lay.x_o) < 0.02)))
        bottom = np.flatnonzero(vm < lay.v_e + 0.02)
        bcs["whip"] = np.unique(np.concatenate([
            3 * eye_nodes, 3 * foot, 3 * bottom + 2, [3 * n1 + 1]]))
    if part == "rr_sleeve":
        # The sleeve is carried by the rails at its ends; held there.
        xm, ym, vm = _local(lay, pts)
        span = lay.sleeve_v1 - lay.sleeve_v0
        on_channel = np.asarray(extra(xm, ym, vm), dtype=bool)
        ends = on_channel & ((vm < lay.sleeve_v0 + 0.15 * span)
                             | (vm > lay.sleeve_v1 - 0.15 * span))
        nodes = np.flatnonzero(ends)
        bcs = {"free": np.unique(np.concatenate([
            3 * nodes, 3 * nodes + 2, [3 * n1 + 1]]))}
        for case in cases:
            case.bc = "free"
        excluded_nodes = []
        relieve = False

    # Inertia relief. A point contact and the cosine pressure standing in for
    # it do not have exactly the same line of action, so each unit case is
    # left slightly out of moment balance, and three restraint nodes reacting
    # that would read as a local singularity. Removing each case's rigid-body
    # resultant as a distributed inertia load spreads any imbalance over the
    # part instead. It is linear, so relieving every unit case is exactly the
    # same as relieving their sum at each crank angle -- where, the loads
    # being balanced, there is almost nothing left to relieve.
    vol = _nodal_volume(mesh)
    phi = _rigid_modes(mesh.points)
    mphi = phi * np.repeat(vol, 3)[:, None]
    gram = phi.T @ mphi
    relieved = []
    for i, case in enumerate(cases):
        vec = loads[i]
        if relieve and case.bc == "free":
            vec = vec - mphi @ np.linalg.solve(gram, phi.T @ vec)
        relieved.append(vec)

    t2 = time.time()
    displacements = np.zeros((len(cases), basis.N))
    for bc_name, dofs in bcs.items():
        members = [i for i, c in enumerate(cases) if c.bc == bc_name]
        if not members:
            continue
        free = np.setdiff1d(np.arange(basis.N), dofs)
        A = K[free][:, free].tocsc()
        lu = splu(A)
        for i in members:
            u = np.zeros(basis.N)
            u[free] = lu.solve(relieved[i][free])
            residual = np.linalg.norm(A @ u[free] - relieved[i][free]) / max(
                np.linalg.norm(relieved[i][free]), 1e-30)
            if residual > 1e-6:
                notes.append(f"{cases[i].name}: solve residual {residual:.1e}")
            displacements[i] = u
    timings["solve_s"] = time.time() - t2

    tensors = np.empty((len(cases), mesh.n_elements, 6), dtype=np.float32)
    for i in range(len(cases)):
        stress, _ = stress_tensors(mesh, displacements[i].reshape(-1, 3).T,
                                   material, lame=lame_e)
        tensors[i] = _voigt(stress)

    excluded = np.zeros(mesh.n_elements, dtype=bool)
    for node in excluded_nodes:
        excluded |= (mesh.elements == node).any(axis=0)
    timings["total_s"] = time.time() - t0

    sol = UnitSolution(part=part, mesh=mesh, material=material, cases=cases,
                       loads=loads, relieved=relieved,
                       displacements=displacements,
                       tensors=tensors, resultants=np.array(resultants),
                       moments=np.array(moments), excluded=excluded,
                       notes=notes, timings=timings, region=region,
                       strength_ratio=strength_ratio,
                       region_label=("sheet-gyroid core"
                                     if region is not None else ""))
    if use_cache:
        _UNIT_CACHE[key] = sol
        while len(_UNIT_CACHE) > _UNIT_CACHE_LIMIT:
            _UNIT_CACHE.pop(next(iter(_UNIT_CACHE)))
    return sol


def solve_part(state, part: str, target_elements: int = 25_000,
               coupled: bool = True, analysis=None) -> PartResult:
    """One part's stress through the whole cycle.

    ``coupled`` (the default) takes its loads from the contact network with
    the parts' own flexibility coupled in (:mod:`psrt.railrod.coupled`):
    how preload and firing load divide between the receiver seat, the notch
    and the crankpin depends on how the clamp bends, and a rigid-body network
    gets that badly wrong. ``coupled=False`` uses the rigid network, which is
    what the fast analytical layer runs on.
    """
    analysis = analysis or analyse(state)
    if coupled:
        from .coupled import coupled_analysis
        analysis = coupled_analysis(state, target_elements=target_elements)
    sol = unit_solve(state, part, target_elements, analyse(state))
    # coefficients from whichever network is in force, matched by name
    fresh = {c.name: np.asarray(c.coeff, dtype=float)
             for c in part_cases(state, analysis, part)}
    coeffs = np.column_stack([fresh[c.name] for c in sol.cases])
    notes = list(sol.notes)
    notes.append("loads from the " + ("flexibility-coupled" if coupled
                                      else "rigid") + " contact network")
    return PartResult(part=part, mesh=sol.mesh, material=sol.material,
                      cases=sol.cases, tensors=sol.tensors,
                      resultants=sol.resultants, moments=sol.moments,
                      coeffs=coeffs, theta=analysis.cycle.theta,
                      excluded=sol.excluded, notes=notes,
                      timings=dict(sol.timings), region=sol.region,
                      strength_ratio=sol.strength_ratio,
                      region_label=sol.region_label)


# --- reporting -----------------------------------------------------------------

def equilibrium_residual(result: PartResult) -> dict:
    """How far the applied loads are from balancing, around the cycle.

    Force AND moment, relative to the loads' own scale. The moment is the
    one that matters: a patch's cosine pressure does not act exactly on the
    network's contact point, and whatever that leaves unbalanced is reacted
    by the three restraint nodes. Cases held by physical restraints (rod
    whip) are left out -- their supports are meant to react.
    """
    free = np.array([c.bc == "free" for c in result.cases])
    c = result.coeffs[:, free]
    net_f = c @ result.resultants[free]
    net_m = c @ result.moments[free]
    scale_f = np.abs(c) @ np.linalg.norm(result.resultants[free], axis=1)
    size = float(np.ptp(result.mesh.points, axis=1).max())
    rel_f = np.linalg.norm(net_f, axis=1) / np.maximum(scale_f, 1e-9)
    rel_m = np.linalg.norm(net_m, axis=1) / np.maximum(scale_f * size, 1e-12)
    worst = int(np.argmax(rel_m))
    return {"max_relative_force": float(rel_f.max()),
            "max_relative_moment": float(rel_m.max()),
            "at_theta_deg": float(np.degrees(result.theta[worst])),
            "max_net_force_n": float(np.linalg.norm(net_f, axis=1).max()),
            "max_net_moment_nm": float(np.linalg.norm(net_m, axis=1).max())}


def fatigue_field(result: PartResult, stats: dict, state, temperature_k):
    """Goodman safety factor per element from the signed von Mises range."""
    mat = result.material
    finish = "machined"
    limit = endurance_limit(mat, temperature_k, 0.01, finish, "axial",
                            state["fatigue.reliability"])
    s_e, s_ut = limit.value, limit.ultimate
    if result.strength_ratio is not None:
        # The lattice is not the alloy. Its strength falls with relative
        # density like its stiffness does, and its surfaces are as-built --
        # a sheet a third of a millimetre thick cannot be machined -- which
        # roughly halves what an as-printed surface will take in fatigue.
        ratio = np.asarray(result.strength_ratio, dtype=float)
        rough = np.where(result.region,
                         state["railrod.lattice_surface_factor"], 1.0)
        s_e = s_e * ratio * rough
        s_ut = s_ut * ratio
    alt = 0.5 * (stats["s_max"] - stats["s_min"])
    mean = 0.5 * (stats["s_max"] + stats["s_min"])
    with np.errstate(divide="ignore", invalid="ignore"):
        tensile = alt / s_e + np.maximum(mean, 0.0) / s_ut
        sf = np.where(tensile > 0, 1.0 / tensile, np.inf)
    return sf, limit


def _interface_elements(result: PartResult) -> np.ndarray:
    """Elements with a node shared with the other material.

    One layer either side of the boundary, which is where a continuum with a
    step change in stiffness puts a stress it cannot resolve.
    """
    if result.region is None:
        return np.zeros(result.mesh.n_elements, dtype=bool)
    elements = result.mesh.elements
    n_nodes = result.mesh.points.shape[1]
    core_node = np.zeros(n_nodes, dtype=bool)
    skin_node = np.zeros(n_nodes, dtype=bool)
    core_node[elements[:, result.region].ravel()] = True
    skin_node[elements[:, ~result.region].ravel()] = True
    shared = core_node & skin_node
    return shared[elements].any(axis=0)


def summarise(result: PartResult, state, temperature_k=None) -> dict:
    from ..evaluate import evaluate
    from ..margins import ComponentContext

    metrics = evaluate(state)
    if temperature_k is None:
        temperature_k = metrics.thermal.rod
    stats = result.cycle_stats()
    mask = ~result.excluded
    peak = np.where(mask, stats["peak"], 0.0)
    allowable, basis = ComponentContext.strength(
        ComponentContext, result.material, temperature_k)
    sf, limit = fatigue_field(result, stats, state, temperature_k)
    sf_m = np.where(mask, sf, np.inf)
    worst = int(np.argmax(peak))
    worst_f = int(np.argmin(sf_m))
    centroids = result.mesh.points.T[result.mesh.elements.T].mean(axis=1)
    p995 = float(np.percentile(peak[mask], 99.5)) if mask.any() else 0.0

    # A part of two materials cannot be judged by one allowable: the skin's
    # 99.5th percentile against the alloy says nothing about a lattice at a
    # seventh of its strength. Each region is scored against its own.
    regions = None
    static_sf = allowable / max(p995, 1.0)
    if result.region is not None:
        ratio = np.asarray(result.strength_ratio, dtype=float)
        # Elements that touch the other material. A step change in modulus
        # is a singularity in a continuum exactly as a re-entrant corner is:
        # the stress there refines without settling, so it is reported but
        # not treated as the part's answer. What the real interface will
        # take depends on the fillet where each gyroid sheet meets the
        # shell, which is a sub-model of a few cells, not this mesh.
        interface = _interface_elements(result)
        regions = {}
        for name, sel in (("skin and shells", mask & ~result.region),
                          (result.region_label, mask & result.region)):
            if not sel.any():
                continue
            bulk = sel & ~interface
            if not bulk.any():
                bulk = sel
            here = stats["peak"][bulk]
            allow_here = allowable * float(np.median(ratio[sel]))
            q = float(np.percentile(here, 99.5))
            regions[name] = {
                "elements": int(sel.sum()),
                "interface_elements": int((sel & interface).sum()),
                "allowable_pa": allow_here,
                "peak_von_mises_pa": float(here.max()),
                "p99_5_von_mises_pa": q,
                "static_safety_factor": float(allow_here / max(q, 1.0)),
                "min_fatigue_safety_factor": float(np.min(sf_m[bulk])),
                "min_fatigue_at_interface": (
                    float(np.min(sf_m[sel & interface]))
                    if (sel & interface).any() else None),
            }
        static_sf = min(r["static_safety_factor"] for r in regions.values())
        sf_m = np.where(interface, np.inf, sf_m)
    return {
        "part": result.part,
        "label": PART_LABELS.get(result.part, result.part),
        "material": result.material.key,
        "elements": int(result.mesh.n_elements),
        "cases": [{"name": c.name, "label": c.label} for c in result.cases],
        "peak_von_mises_pa": float(peak.max()),
        "p99_5_von_mises_pa": p995,
        "peak_theta_deg": float(np.degrees(
            result.theta[stats["peak_at"][worst]])),
        "peak_location_mm": (centroids[worst] * 1e3).tolist(),
        "allowable_pa": float(allowable),
        "allowable_basis": basis,
        "static_safety_factor": float(static_sf),
        "regions": regions,
        "min_fatigue_safety_factor": float(sf_m.min()),
        "fatigue_location_mm": (centroids[int(np.argmin(sf_m))] * 1e3).tolist(),
        "endurance_limit_pa": float(limit.value),
        "temperature_c": float(temperature_k - 273.15),
        "equilibrium": equilibrium_residual(result),
        "timings": result.timings,
        "notes": result.notes,
    }


def field_payload(result: PartResult, state, stride: int = 2) -> dict:
    """Surface field for the viewport: per-vertex unit tensors so the page
    can recombine them at any crank angle without a round trip."""
    from ..fea.field import boundary_surface
    from ..evaluate import evaluate

    mesh = result.mesh
    tris = boundary_surface(mesh.elements)
    used = np.unique(tris)
    remap = np.full(mesh.points.shape[1], -1, dtype=np.int64)
    remap[used] = np.arange(len(used))

    tets = mesh.elements.T
    corners = mesh.points.T[tets]
    vol = np.abs(np.einsum("ij,ij->i", corners[:, 1] - corners[:, 0],
                           np.cross(corners[:, 2] - corners[:, 0],
                                    corners[:, 3] - corners[:, 0]))) / 6.0
    weight_sum = np.zeros(mesh.n_nodes)
    for c in range(4):
        np.add.at(weight_sum, tets[:, c], vol)
    weight_sum = np.maximum(weight_sum, 1e-300)

    def nodal(values):                         # (n_el, k) -> (n_used, k)
        out = np.zeros((mesh.n_nodes, values.shape[1]))
        for c in range(4):
            np.add.at(out, tets[:, c], values * vol[:, None])
        return (out / weight_sum[:, None])[used]

    n_cases = len(result.cases)
    unit = np.stack([nodal(result.tensors[k]) for k in range(n_cases)])
    stats = result.cycle_stats()
    metrics = evaluate(state)
    sf, _ = fatigue_field(result, stats, state, metrics.thermal.rod)
    sf_capped = np.minimum(np.where(np.isfinite(sf), sf, 50.0), 50.0)
    nodal_peak = nodal(stats["peak"][:, None])[:, 0]
    nodal_sf = nodal(sf_capped[:, None])[:, 0]

    keep = np.arange(0, len(result.theta), stride)
    return {
        "vertices": np.round(mesh.points[:, used].T * 1e3, 4).tolist(),
        "triangles": remap[tris].tolist(),
        "unit_mpa": np.round(unit / 1e6, 6).tolist(),   # (cases, verts, 6)
        "coeffs": np.round(result.coeffs[keep], 6).tolist(),
        "theta_deg": np.degrees(result.theta[keep]).tolist(),
        "peak_mpa": np.round(nodal_peak / 1e6, 3).tolist(),
        "fatigue_sf": np.round(nodal_sf, 3).tolist(),
        "case_names": [c.name for c in result.cases],
    }
