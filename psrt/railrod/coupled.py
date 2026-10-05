"""The contact network with the parts' own flexibility coupled in.

The rigid network in :mod:`psrt.railrod.network` gets statically determinate
things right and load SHARING wrong, and on this concept the sharing is most of
the story. A rigid clamp tightened by its bolt pivots on the crankpin and
lands its whole preload on one corner of the receiver; a real clamp bends, and
the load spreads round the bore toward the seat. Which contacts carry the
preload, how much of it reaches the rail notch, and how much of the rod's
tension goes back into the bolt, all depend on that bending.

So this module takes each part's compliance from its own FEA -- the unit load
solutions :func:`psrt.railrod.fea.unit_solve` already computes -- and solves
the network with it:

    W_ij = h * P_i . U_j

is the generalised displacement at contact i per newton at contact j, on each
part that both touch, where P_i is contact i's unit patch load and U_j the
displacement it causes (h = 1/2 for the symmetric parts, which are solved with
both sides loaded). P_i here is the load AFTER inertia relief, which makes W
the free-free flexibility -- symmetric and positive semi-definite, measured in
the part's mean axes -- rather than a flexibility relative to three arbitrary
restraint nodes, which is neither. (The first version used the raw loads and
got negative self-compliances.) The gap at contact i is then

    gap_i = gap0_i + G_i . q + sum_j W_ij lambda_j + lambda_i / k_i + e_i

rigid-body motion, elastic deformation of both parts, the local contact
spring, and the deformation the external loads cause on their own (the pin
force on the eye, the big end's inertia). A contact carries force only while
its gap is closed. The bolt is one more interaction: tightened to its preload
at assembly, a spring of fixed length afterwards, with the clamp's own
deformation at the thread in its stretch. The active-set solve is the same as
the rigid one's, over forty-odd unknowns instead of five.

Cost: three part FEAs (rails, receiver, right clamp) the first time a design
is asked for, cached after that, then milliseconds per crank angle.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from .analysis import _scaled_factor, analyse
from .network import CLAMP_B, CycleResult, NDOF, RAIL, RECEIVER, CV

_CACHE: dict = {}

BODY_PARTS = (("rail", "rr_rails", 0.5), ("receiver", "rr_receiver", 0.5),
              ("clamp", "rr_clamp_right", 1.0))


@dataclass
class Compliance:
    W: np.ndarray            # (nc+1, nc+1) mm per N, bolt last
    e_comp: np.ndarray       # (nc+1,) mm per N of rod compression
    e_tens: np.ndarray       # (nc+1,) mm per N of rod tension
    e_accel: np.ndarray      # (nc+1,) mm per m/s^2 of big-end acceleration


def build_compliance(net, solutions: dict) -> Compliance:
    nc = len(net.contacts)
    W = np.zeros((nc + 1, nc + 1))
    e_comp = np.zeros(nc + 1)
    e_tens = np.zeros(nc + 1)
    e_accel = np.zeros(nc + 1)
    for body, part, half in BODY_PARTS:
        sol = solutions[part]
        idx, P, U = [], [], []
        ext = {}
        for k, case in enumerate(sol.cases):
            if case.contact >= 0:
                c = net.contacts[case.contact]
                # a contact on the symmetry line is applied once at twice the
                # per-side force; per side, its unit load is twice the patch
                scale = 2.0 if (half < 1.0 and abs(c.point[0]) < 1e-6) else 1.0
                idx.append(case.contact)
                P.append(sol.relieved[k] * scale)
                U.append(sol.displacements[k] * scale)
            elif case.name == "bolt":
                idx.append(nc)
                P.append(sol.relieved[k])
                U.append(sol.displacements[k])
            elif case.name in ("eye_comp", "eye_tens", "inertia"):
                ext[case.name] = sol.displacements[k]
        if not idx:
            continue
        P = np.array(P)
        U = np.array(U)
        Wx = half * (P @ U.T) * 1e3                       # m/N -> mm/N
        Wx = 0.5 * (Wx + Wx.T)                            # Maxwell, exactly
        W[np.ix_(idx, idx)] += Wx
        for name, u in ext.items():
            col = half * (P @ u) * 1e3
            target = {"eye_comp": e_comp, "eye_tens": e_tens,
                      "inertia": e_accel}[name]
            target[idx] += col
    return Compliance(W=W, e_comp=e_comp, e_tens=e_tens, e_accel=e_accel)


class CoupledNetwork:
    """Same interface as :class:`psrt.railrod.network.Network`, flexible."""

    def __init__(self, rigid, compliance: Compliance):
        self.rigid = rigid
        self.contacts = rigid.contacts
        self.C = compliance
        self.preload = rigid.preload
        self.bolt_k = rigid.bolt_k
        self.masses = rigid.masses
        self.notes = []

    def solve(self, eye_force: float, bigend_accel_v: float = 0.0,
              active=None, assembly: dict | None = None,
              max_iter: int = 400) -> dict:
        net = self.rigid
        Gm, kv, gap0 = net.arrays()
        nc = len(kv)
        gb = net.G_bolt
        W = self.C.W
        e = (self.C.e_comp * max(eye_force, 0.0)
             + self.C.e_tens * max(-eye_force, 0.0)
             + self.C.e_accel * bigend_accel_v)
        F0 = self.preload
        kb = self.bolt_k

        load = np.zeros(NDOF)
        load[RAIL] -= eye_force / 2.0
        load[RECEIVER] -= self.masses["receiver_half"] * bigend_accel_v
        load[CV] -= self.masses["clamp"] * bigend_accel_v

        closed = (np.array(active, dtype=bool) if active is not None
                  else gap0 <= 0)
        bolt_on = True
        n = NDOF + nc + 1
        reg = 1e-9 * float(kv.max())
        s_asm = None if assembly is None else assembly["half_stretch"]
        settled = False
        for _ in range(max_iter):
            M = np.zeros((n, n))
            r = np.zeros(n)
            M[:NDOF, :NDOF] = -reg * np.eye(NDOF)
            M[:NDOF, NDOF:NDOF + nc] = Gm.T
            M[:NDOF, -1] = -gb
            r[:NDOF] = -load
            for i in range(nc):
                row = NDOF + i
                if closed[i]:
                    M[row, :NDOF] = Gm[i]
                    M[row, NDOF:] = W[i]
                    M[row, row] += 1.0 / kv[i]
                    r[row] = -gap0[i] - e[i]
                else:
                    M[row, row] = 1.0
            if assembly is None:
                M[-1, -1] = 1.0
                r[-1] = F0
            elif bolt_on:
                # F_b = F0 + 2 kb (S - S_asm),  S = gb.q - (W_b . lam + e_b)
                M[-1, :NDOF] = -2.0 * kb * gb
                M[-1, NDOF:] = 2.0 * kb * W[-1]
                M[-1, -1] += 1.0
                r[-1] = F0 - 2.0 * kb * s_asm - 2.0 * kb * e[-1]
            else:
                M[-1, -1] = 1.0
                r[-1] = 0.0
            z = np.linalg.solve(M, r)
            q, lam, fb = z[:NDOF], z[NDOF:NDOF + nc], z[-1]
            gaps_open = gap0 + Gm @ q + W[:nc] @ np.concatenate(
                [np.where(closed, lam, 0.0), [fb]]) + e[:nc]
            # One change at a time -- the most negative contact force opened,
            # or failing that the deepest interpenetration closed. Changing
            # every violated contact at once is what the rigid solve does,
            # and with forty coupled contacts it cycles.
            neg = np.where(closed, lam, np.inf)
            pen = np.where(~closed, gaps_open, np.inf)
            bolt_change = None
            if assembly is not None:
                if bolt_on and fb < -1e-9:
                    bolt_change = False
                elif not bolt_on:
                    stretch = gb @ q - (W[-1] @ np.concatenate(
                        [np.where(closed, lam, 0.0), [0.0]]) + e[-1])
                    if F0 + 2.0 * kb * (stretch - s_asm) > 1e-9:
                        bolt_change = True
            tol_f = 1e-9 * max(F0, 1.0)
            if neg.min() < -tol_f:
                closed = closed.copy()
                closed[int(np.argmin(neg))] = False
            elif bolt_change is not None:
                bolt_on = bolt_change
            elif pen.min() < -1e-9:
                closed = closed.copy()
                closed[int(np.argmin(pen))] = True
            else:
                settled = True
                break
        lam = np.where(closed, np.maximum(lam, 0.0), 0.0)
        fb = max(fb, 0.0) if (assembly is None or bolt_on) else 0.0
        half_stretch = float(gb @ q - (W[-1] @ np.concatenate([lam, [fb]])
                                       + e[-1]))
        out = net._summarise(q, lam, fb, closed)
        out["settled"] = settled
        out["unrestrained"] = bool(np.max(np.abs(q[:4])) > 0.5)
        out["half_stretch"] = half_stretch
        return out


def _cycle(state, base, coupled_net) -> CycleResult:
    cyc = base.cycle
    pre = coupled_net.solve(0.0, 0.0)
    active = pre["closed"]
    rows = []
    for f, a in zip(cyc.f_rod, cyc.accel_bigend_v):
        res = coupled_net.solve(float(f), float(a), active=active,
                                assembly=pre)
        active = res["closed"]
        rows.append(res)

    def col(key):
        return np.array([r[key] for r in rows], dtype=float)

    out = copy.copy(cyc)
    out.floor = col("floor")
    out.flank = col("flank")
    out.knob = np.array([r["knob"] for r in rows])
    out.guide = np.array([r["guide"] for r in rows])
    out.clamp_pin = np.array([r["clamp_pin"] for r in rows])
    out.receiver_pin = col("receiver_pin")
    out.lug = col("lug")
    out.bolt = col("bolt_force")
    out.forces = np.array([r["forces"] for r in rows])
    out.preload_state = pre
    out.network = coupled_net
    out.unsettled = int(sum(not r["settled"] for r in rows))
    return out


def coupled_analysis(state, target_elements: int = 25_000):
    """The rail-rod analysis with the flexibility-coupled network in force."""
    from .fea import unit_solve

    key = (state.fingerprint(), int(target_elements))
    if key in _CACHE:
        return _CACHE[key]
    base = analyse(state)
    solutions = {part: unit_solve(state, part, target_elements, base)
                 for _, part, _ in BODY_PARTS}
    comp = build_compliance(base.cycle.network, solutions)
    net = CoupledNetwork(base.cycle.network, comp)
    cycle = _cycle(state, base, net)

    result = copy.copy(base)
    result.cycle = cycle
    result.coupled = True
    pre = cycle.preload_state
    i_t = int(np.argmin(cycle.f_rod))
    i_c = int(np.argmax(cycle.f_rod))
    result.separation_factor = (
        _scaled_factor(net, pre, float(cycle.f_rod[i_t]),
                       float(cycle.accel_bigend_v[i_t]), "floor", -1)
        if cycle.f_rod[i_t] < 0 else float("inf"))
    result.unseat_factor = _scaled_factor(
        net, pre, float(cycle.f_rod[i_c]), float(cycle.accel_bigend_v[i_c]),
        "flank", 1)
    result.notes = list(base.notes)
    if cycle.unsettled:
        result.notes.append(f"the coupled contact solve did not settle at "
                            f"{cycle.unsettled} crank angles")
    _CACHE.clear()
    _CACHE[key] = result
    return result
