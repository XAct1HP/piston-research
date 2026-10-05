"""The contact network: who pushes on whom, at every crank angle.

The rail rod carries load through CONTACTS rather than through one continuous
forging, and which contacts carry it changes around the cycle. Under firing
the rails push down on the receiver floor; under overlap tension they pull up
on the clamp knobs and flanks; the bolt preload squeezes the whole big end all
the time, and how much of it survives at the rail notches is the question the
concept stands or falls on. None of that can be read off a single free-body
diagram -- the clamp alone has more contacts than a rigid body has equations
(knob, flank, receiver flank and scoop, crankpin, the other clamp's lug, the
bolt), so the split between them depends on stiffness. This module resolves
it.

The model
---------
A planar, symmetric half-model of rigid bodies joined by compression-only
contact springs:

    rail half        1 DOF   moves along the rod axis (the eye couples the
                             two rails, so by symmetry neither moves sideways)
    receiver half    1 DOF   along the rod axis
    right clamp      3 DOF   x, v and rotation in the plane
    crankpin         ground
    mirror plane     ground in x, standing in for the left clamp

Each contact is a point, a normal, an initial gap and a stiffness
``E_eff * patch_area / contact_depth``. A contact carries force only while
closed. The bolt is a spring with a preload. Solving is an active-set
iteration: guess which contacts are closed, solve the linear system, open the
ones in tension, close the ones that interpenetrate, repeat. With five degrees
of freedom it is instantaneous, which is what lets it run at every one of 720
crank angles.

What it deliberately leaves out, so the numbers are read correctly
------------------------------------------------------------------
* The parts are RIGID. Everything elastic is lumped into the contact
  stiffness through ``railrod.contact_depth``. Statically determinate
  quantities (the total rail load, what the floor carries after separation)
  do not depend on it; the SHARE between redundant contacts does. The FEA
  compliance of each part is the next refinement and is not in yet.
* No friction. The concept says the tensile load path must not rely on it,
  so leaving it out is the conservative reading for retention -- and the
  unconservative one for how much the bolt has to supply against pry.
* Symmetric loading only. Rod whip (the transverse inertia of the rod body)
  is antisymmetric, so it cannot pass through a symmetric half-model; the
  FEA layer carries it as its own load case.
* Inertia follows the tool's two-mass convention: the rod force at the small
  end already includes the reciprocating share of the rod, so only the
  big-end parts' own inertia is added here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .. import kinematics as kin
from .. import materials as materials_mod
from .layout import Layout, _shapely

NDOF = 5
RAIL, RECEIVER, CX, CV, CT = range(NDOF)
GROUND, RAIL_B, RECEIVER_B, CLAMP_B = "ground", "rail", "receiver", "clamp"


@dataclass
class Contact:
    group: str
    a: str                      # body on the -n side
    b: str                      # body on the +n side
    point: np.ndarray           # mm, local
    normal: np.ndarray          # from a toward b
    gap0: float                 # mm
    k: float                    # N/mm
    area: float                 # mm^2


def _jac(body: str, p) -> np.ndarray:
    """2 x NDOF: displacement of body point p per generalised coordinate."""
    j = np.zeros((2, NDOF))
    if body == RAIL_B:
        j[1, RAIL] = 1.0
    elif body == RECEIVER_B:
        j[1, RECEIVER] = 1.0
    elif body == CLAMP_B:
        j[0, CX] = 1.0
        j[1, CV] = 1.0
        j[0, CT] = -p[1]
        j[1, CT] = p[0]
    return j


def _unit(a):
    return np.array([math.cos(a), math.sin(a)])


@dataclass
class Network:
    layout: Layout
    contacts: list
    bolt_point: np.ndarray
    bolt_k: float               # N/mm
    preload: float              # N
    masses: dict                # kg, half receiver / clamp
    notes: list = field(default_factory=list)

    def G(self, c: Contact) -> np.ndarray:
        return (_jac(c.b, c.point) - _jac(c.a, c.point)).T @ c.normal

    def arrays(self):
        if getattr(self, "_arrays", None) is None:
            self._arrays = (np.array([self.G(c) for c in self.contacts]),
                            np.array([c.k for c in self.contacts]),
                            np.array([c.gap0 for c in self.contacts]))
        return self._arrays

    @property
    def G_bolt(self) -> np.ndarray:
        return _jac(CLAMP_B, self.bolt_point).T @ np.array([1.0, 0.0])

    def solve(self, eye_force: float, bigend_accel_v: float = 0.0,
              active=None, assembly: dict | None = None,
              max_iter: int = 80) -> dict:
        """Equilibrium for one load state.

        ``eye_force`` is the rod force at the small end, newtons, compression
        positive (the whole rod, not the half). ``bigend_accel_v`` is the
        big end's acceleration along the rod axis toward the small end, in
        m/s^2.

        With ``assembly=None`` this IS the assembly: the bolt is tightened to
        its preload as a constant force, and the parts move until the
        contacts react it -- which is what turning a nut does. Pass that
        result back as ``assembly`` for every operating load: from then on
        the bolt is a spring of fixed length, so its tension rises and falls
        only as the lugs it joins move apart or together.
        """
        F0 = self.preload
        Gm, kv, gap0 = self.arrays()
        gb = self.G_bolt
        k_scale = float(kv.max())
        q0_b = 0.0 if assembly is None else float(gb @ assembly["q"])

        load = np.zeros(NDOF)
        load[RAIL] -= eye_force / 2.0
        load[RECEIVER] -= self.masses["receiver_half"] * bigend_accel_v
        load[CV] -= self.masses["clamp"] * bigend_accel_v

        closed = (np.array(active, dtype=bool) if active is not None
                  else gap0 <= 0)
        bolt_on = True
        q = np.zeros(NDOF)
        settled = False
        reg = np.eye(NDOF) * k_scale * 1e-9
        for _ in range(max_iter):
            kc = np.where(closed, kv, 0.0)
            K = reg + (Gm.T * kc) @ Gm
            rhs = load - Gm.T @ (kc * gap0)
            if assembly is None:
                rhs += -F0 * gb
            elif bolt_on:
                K += 2.0 * self.bolt_k * np.outer(gb, gb)
                rhs += -(F0 - 2.0 * self.bolt_k * q0_b) * gb
            q = np.linalg.solve(K, rhs)
            gaps = gap0 + Gm @ q
            new_closed = gaps < 0.0
            if assembly is None:
                new_bolt = True
            else:
                new_bolt = (F0 + 2.0 * self.bolt_k * (gb @ q - q0_b)) > 0.0
            if np.array_equal(new_closed, closed) and new_bolt == bolt_on:
                settled = True
                break
            closed, bolt_on = new_closed, new_bolt
        forces = np.where(closed, -kv * gaps, 0.0)
        forces = np.maximum(forces, 0.0)
        if assembly is None:
            bolt_force = F0
        else:
            bolt_force = (max(F0 + 2.0 * self.bolt_k * (gb @ q - q0_b), 0.0)
                          if bolt_on else 0.0)
        out = self._summarise(q, forces, bolt_force, closed)
        out["settled"] = settled
        # A mode nothing resists shows up as a displacement that only the
        # regularising spring is holding: millimetres rather than microns.
        out["unrestrained"] = bool(np.max(np.abs(q[:4])) > 0.5)
        return out

    def _summarise(self, q, forces, bolt_force, closed) -> dict:
        if getattr(self, "_groups", None) is None:
            names = sorted({c.group for c in self.contacts})
            self._groups = {g: np.array([c.group == g for c in self.contacts])
                            for g in names}
            self._normals = np.array([c.normal for c in self.contacts])
            self._points = np.array([c.point for c in self.contacts])
        vec = forces[:, None] * self._normals
        sums = {g: vec[m].sum(axis=0) for g, m in self._groups.items()}
        zero = np.zeros(2)
        return {
            "q": q, "closed": closed, "forces": forces,
            "bolt_force": bolt_force,
            "group_force": sums,
            "floor": float(sums.get("floor", zero)[1]),
            "flank": float(np.linalg.norm(sums.get("flank", zero))),
            "knob": sums.get("knob", zero).copy(),
            "guide": sums.get("guide", zero).copy(),
            "clamp_pin": sums.get("clamp_pin", zero).copy(),
            "receiver_pin": float(sums.get("receiver_pin", zero)[1]),
            "lug": float(sums.get("lug", zero)[0]),
        }

    def group_detail(self, forces) -> dict:
        """Per-group list of (point, normal, force) for one solution."""
        out: dict = {}
        for c, f in zip(self.contacts, forces):
            out.setdefault(c.group, []).append((c.point, c.normal, float(f)))
        return out


def _material(key):
    return materials_mod.get(key)


def build_network(state, layout: Layout, part_masses: dict | None = None
                  ) -> Network:
    """Place every contact from the layout."""
    geometry, _, _ = _shapely()
    depth = state["railrod.contact_depth"] * 1e3                 # mm
    steel_rail = _material(state["materials.rod"])
    steel_big = _material(state["railrod.bigend_material"])
    bolt_mat = _material(state["railrod.bolt_material"])
    E_r = steel_rail.youngs_modulus / 1e6                        # N/mm^2
    E_b = steel_big.youngs_modulus / 1e6
    E_pin = 205e3

    def eff(e1, e2):
        return 1.0 / (1.0 / e1 + 1.0 / e2)

    lay = layout
    n = lay.notch
    h = lay.rail_depth
    w = lay.width
    b = lay.x_o - lay.x_i
    contacts: list[Contact] = []

    def add(group, a, b_, p, normal, gap0, area, e):
        normal = np.asarray(normal, dtype=float)
        normal = normal / np.linalg.norm(normal)
        contacts.append(Contact(group, a, b_, np.asarray(p, dtype=float),
                                normal, gap0, e * area / depth, area))

    # 1. rail foot on the receiver floor
    add("floor", RECEIVER_B, RAIL_B, ((lay.x_o + lay.x_i) / 2.0, lay.v_e),
        (0.0, 1.0), 0.0, b * h, eff(E_r, E_b))

    # 1b. rail foot against the slot web. The rigid half-model holds the
    #     rails still sideways, so this never closes there; with the parts'
    #     flexibility coupled in (psrt.railrod.coupled) it takes the inward
    #     push the clamp tips put on the feet.
    c_p = max(lay.slot_clearance, 0.02)
    add("web", RECEIVER_B, RAIL_B, (lay.x_i, 0.5 * (lay.v_e + lay.v_seat)),
        (1.0, 0.0), c_p, (lay.v_seat - lay.v_e) * h, eff(E_r, E_b))

    # 2. clamp flank on the ramp: two points so the flank can carry moment
    ramp = n.ramp_length
    fit = -state["railrod.seat_interference"] * 1e3
    for t in (0.2, 0.8):
        add("flank", RAIL_B, CLAMP_B, n.T + t * (n.F1 - n.T), n.normal, fit,
            ramp * h / 2.0, eff(E_r, E_b))

    # 3. knob in its socket: the top radius, from just below A to T
    c = n.centre
    a_a = math.atan2(n.A[1] - c[1], n.A[0] - c[0]) + math.radians(8.0)
    a_t = math.pi + n.alpha
    angles = np.linspace(a_a, a_t, 6)
    arc = n.r_top * (a_t - a_a) / len(angles)
    for ang in angles:
        p = c + n.r_top * _unit(ang)
        add("knob", RAIL_B, CLAMP_B, p, -_unit(ang), fit, arc * h,
            eff(E_r, E_b))

    # 4. clamp against the receiver: flank, top corner and swing channel.
    #    Every receiver boundary segment within reach of the clamp becomes a
    #    candidate, spaced out so no one stretch dominates by point count.
    rec = lay.receiver.exterior
    coords = np.asarray(rec.coords)
    ccw = rec.is_ccw
    cand = []
    for p0, p1 in zip(coords[:-1], coords[1:]):
        mid = 0.5 * (p0 + p1)
        if mid[0] <= lay.x_o + 0.2:
            continue
        seg = p1 - p0
        length = float(np.linalg.norm(seg))
        if length < 1e-9:
            continue
        outward = (np.array([seg[1], -seg[0]]) if ccw
                   else np.array([-seg[1], seg[0]])) / length
        probe = geometry.Point(mid + outward * (lay.guide_clearance + 0.05))
        if lay.clamp_right.distance(probe) < 0.08:
            cand.append((mid, outward, length))
    if cand:
        total = sum(cl for _, _, cl in cand)
        pieces = 14
        step = total / pieces
        def flush(bucket):
            """One lumped guide contact for a stretch of receiver boundary.

            A stretch inboard of the receiver's outer face is a wall of the
            swing channel, and the clamp meets it only over the tongue's
            width -- the rail depth -- not over the width of the big end.
            """
            length = sum(cl for _, _, cl in bucket)
            pm = sum(m * cl for m, _, cl in bucket) / length
            nm = sum(nn * cl for _, nn, cl in bucket)
            span = h if abs(pm[0]) < lay.x_rc - 1e-6 else w
            add("guide", RECEIVER_B, CLAMP_B, pm, nm, lay.guide_clearance,
                length * span, eff(E_b, E_b))

        acc, bucket = 0.0, []
        for mid, normal, length in cand:
            bucket.append((mid, normal, length))
            acc += length
            if acc >= step:
                flush(bucket)
                acc, bucket = 0.0, []
        if bucket:
            flush(bucket)

    # 5. clamp bore on the crankpin (through the bearing shell)
    start = lay.theta_r + math.radians(3.0)
    phis = np.linspace(start, math.pi - 0.02, 10)
    arc = lay.Rb * (math.pi - start) / len(phis)
    for phi in phis:
        u = np.array([math.sin(phi), math.cos(phi)])
        add("clamp_pin", GROUND, CLAMP_B, lay.Rb * u, u, 0.0, arc * w,
            eff(E_b, E_pin))

    # 6. receiver bore on the crankpin (half model: 0..theta_r)
    phis = np.linspace(0.0, lay.theta_r - math.radians(2.0), 6)
    arc = lay.Rb * lay.theta_r / len(phis)
    for i, phi in enumerate(phis):
        u = np.array([math.sin(phi), math.cos(phi)])
        area = arc * w * (0.5 if i == 0 else 1.0)
        add("receiver_pin", GROUND, RECEIVER_B, lay.Rb * u, u, 0.0, area,
            eff(E_b, E_pin))

    # 7. lug face against the other clamp's lug (the mirror plane)
    top = -math.sqrt(max(lay.Rb ** 2 - (lay.lug_gap / 2) ** 2, 0.0)) - 0.5
    for v in np.linspace(top, lay.v_lug_bottom + 0.5, 4):
        area = (top - lay.v_lug_bottom) / 4.0 * w
        add("lug", GROUND, CLAMP_B, (lay.lug_gap / 2.0, v), (1.0, 0.0),
            lay.lug_gap / 2.0, area, E_b)

    # the bolt: through both lugs, so its grip is both lug lengths
    d_b = state["railrod.bolt_diameter"] * 1e3
    pitch = state["railrod.bolt_pitch"] * 1e3
    area_s = math.pi / 4.0 * (d_b - 0.9382 * pitch) ** 2
    grip = 2.0 * lay.x_lug
    k_bolt = bolt_mat.youngs_modulus / 1e6 * area_s / grip
    bolt_point = np.array([lay.x_lug / 2.0, lay.v_bolt])

    masses = {"receiver_half": 0.0, "clamp": 0.0}
    if part_masses:
        masses["receiver_half"] = part_masses.get("rr_receiver", 0.0) / 2.0
        masses["clamp"] = part_masses.get("rr_clamp_right", 0.0)

    return Network(layout=lay, contacts=contacts, bolt_point=bolt_point,
                   bolt_k=k_bolt, preload=state["railrod.bolt_preload"],
                   masses=masses)


# --- the cycle --------------------------------------------------------------------

@dataclass
class CycleResult:
    theta: np.ndarray
    f_rod: np.ndarray                # N, + compression
    accel_bigend_v: np.ndarray       # m/s^2 along the rod toward small end
    accel_bigend_x: np.ndarray       # m/s^2 across the rod (local x)
    accel_small_x: np.ndarray        # m/s^2 across the rod at the small end
    floor: np.ndarray
    flank: np.ndarray
    knob: np.ndarray                 # (n, 2) on the clamp
    guide: np.ndarray                # (n, 2) on the clamp
    clamp_pin: np.ndarray            # (n, 2) on the clamp
    receiver_pin: np.ndarray
    lug: np.ndarray
    bolt: np.ndarray
    forces: np.ndarray               # (n_angles, n_contacts), newtons
    preload_state: dict
    network: Network

    def as_dict(self, stride: int = 1) -> dict:
        s = slice(None, None, stride)
        return {
            "theta_deg": np.degrees(self.theta[s]).tolist(),
            "f_rod_n": self.f_rod[s].tolist(),
            "floor_n": self.floor[s].tolist(),
            "flank_n": self.flank[s].tolist(),
            "knob_n": np.linalg.norm(self.knob[s], axis=1).tolist(),
            "guide_n": np.linalg.norm(self.guide[s], axis=1).tolist(),
            "clamp_pin_n": np.linalg.norm(self.clamp_pin[s], axis=1).tolist(),
            "receiver_pin_n": self.receiver_pin[s].tolist(),
            "lug_n": self.lug[s].tolist(),
            "bolt_n": self.bolt[s].tolist(),
        }


def rod_kinematics(state, theta: np.ndarray) -> dict:
    """Big-end and small-end acceleration in the ROD's frame.

    Cylinder frame: Z up the bore, crank at the origin, crankpin at
    r (sin t, cos t). The rod frame has v along the rod toward the small
    end and x across it; this matches how the viewport turns the rod mesh.
    """
    geom = kin.CrankGeometry.from_state(state)
    r = geom.crank_radius
    omega = state["operating.speed"]
    L = geom.rod_length
    pin_z = kin.pin_distance(geom, theta)
    crank = np.column_stack([r * np.sin(theta), r * np.cos(theta)])
    small = np.column_stack([np.zeros_like(theta), pin_z])
    axis = small - crank
    axis /= np.linalg.norm(axis, axis=1)[:, None]            # toward small end
    # local +x: the viewport turns the rod mesh by (pi - phi) about Y, which
    # carries the rod frame's +x to (-cos phi, -sin phi) in (X, Z). With the
    # axis written as (-sin phi, cos phi) that is (-axis_Z, axis_X).
    across = np.column_stack([-axis[:, 1], axis[:, 0]])

    a_big = -omega ** 2 * crank
    # piston acceleration by differentiating twice at constant speed
    dth = np.gradient(theta)
    dz = np.gradient(pin_z) / dth
    d2z = np.gradient(dz) / dth
    a_small = np.column_stack([np.zeros_like(theta), d2z * omega ** 2])
    return {
        "axis": axis, "across": across,
        "a_big_v": np.einsum("ij,ij->i", a_big, axis),
        "a_big_x": np.einsum("ij,ij->i", a_big, across),
        "a_small_v": np.einsum("ij,ij->i", a_small, axis),
        "a_small_x": np.einsum("ij,ij->i", a_small, across),
    }


def run_cycle(state, layout: Layout, sweep, part_masses: dict | None = None
              ) -> CycleResult:
    """Solve the network at every crank angle of the load sweep."""
    net = build_network(state, layout, part_masses)
    theta = np.asarray(sweep.theta, dtype=float)
    f_rod = np.asarray(sweep.f_rod, dtype=float)
    kinem = rod_kinematics(state, theta)

    pre = net.solve(0.0, 0.0)
    active = pre["closed"]
    rows = []
    for f, a in zip(f_rod, kinem["a_big_v"]):
        res = net.solve(float(f), float(a), active=active, assembly=pre)
        active = res["closed"]
        rows.append(res)

    def col(key):
        return np.array([r[key] for r in rows], dtype=float)

    return CycleResult(
        theta=theta, f_rod=f_rod,
        accel_bigend_v=kinem["a_big_v"], accel_bigend_x=kinem["a_big_x"],
        accel_small_x=kinem["a_small_x"],
        floor=col("floor"), flank=col("flank"),
        knob=np.array([r["knob"] for r in rows]),
        guide=np.array([r["guide"] for r in rows]),
        clamp_pin=np.array([r["clamp_pin"] for r in rows]),
        receiver_pin=col("receiver_pin"), lug=col("lug"),
        bolt=col("bolt_force"),
        forces=np.array([r["forces"] for r in rows]),
        preload_state=pre, network=net)
