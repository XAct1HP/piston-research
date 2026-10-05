"""One call that answers "does the rail rod work at this operating point".

:func:`analyse` lays the concept out, weighs it, runs the contact network
around the whole cycle and derives the handful of numbers every other layer
reads: margins, the UI, the assistant, the FEA load cases. It is a pure
function of the design state and the load sweep, and it is cached on the
state's fingerprint, so the margin layer, the server and the FEA can all ask
for it without paying twice.

Masses here come from the 2D layout -- profile areas times widths, with the
3D features (the seat, the rail slots, the swing channels, the bolt hole)
subtracted by hand, each at its own width along the crank axis.
That is thousands of times cheaper than rebuilding the solids and lands
within a percent of them -- ``tests/test_railrod.py`` holds it to that.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

from .. import materials as materials_mod
from . import layout as layout_mod
from .network import CycleResult, run_cycle

_CACHE: "OrderedDict[str, RailRodAnalysis]" = OrderedDict()
_CACHE_LIMIT = 16
DECIMATE = 4                     # network solved on every 4th sweep angle


def enabled(state) -> bool:
    return bool(state.has("railrod.enabled") and state["railrod.enabled"])


@dataclass
class RailRodAnalysis:
    layout: layout_mod.Layout
    masses: dict                     # kg per part
    cycle: CycleResult
    swing: dict
    conformity: dict
    separation_factor: float         # tension multiple that opens the floor
    unseat_factor: float             # compression multiple that unloads the flank
    notes: list = field(default_factory=list)

    @property
    def total_mass(self) -> float:
        return float(sum(self.masses.values()))

    def summary(self) -> dict:
        c = self.cycle
        n = self.layout.notch
        i_t = int(np.argmin(c.f_rod))
        i_c = int(np.argmax(c.f_rod))
        pre = c.preload_state
        return {
            "masses_kg": self.masses,
            "total_mass_kg": self.total_mass,
            "notch": {
                "depth_mm": n.depth, "ramp_angle_deg": math.degrees(n.alpha),
                "top_radius_mm": n.r_top, "lower_radius_mm": n.r_low,
                "length_mm": n.length, "ramp_length_mm": n.ramp_length,
                "land_mm": float(n.F2[1] - self.layout.v_e),
                "pivot_mm": n.centre.tolist(),
            },
            "self_seating": self_seating(self.layout),
            "assembly": {
                "bolt_n": float(pre["bolt_force"]),
                "rail_seat_preload_n": float(pre["floor"]),
                "flank_n": float(pre["flank"]),
                "knob_n": float(np.linalg.norm(pre["knob"])),
                "receiver_seat_n": float(np.linalg.norm(pre["guide"])),
                "lug_faces_n": float(pre["lug"]),
            },
            "peak_tension": {
                "theta_deg": float(np.degrees(c.theta[i_t])),
                "rod_force_n": float(c.f_rod[i_t]),
                "rail_seat_n": float(c.floor[i_t]),
                "flank_n": float(c.flank[i_t]),
                "knob_n": float(np.linalg.norm(c.knob[i_t])),
                "receiver_seat_n": float(np.linalg.norm(c.guide[i_t])),
                "bolt_n": float(c.bolt[i_t]),
            },
            "peak_compression": {
                "theta_deg": float(np.degrees(c.theta[i_c])),
                "rod_force_n": float(c.f_rod[i_c]),
                "rail_seat_n": float(c.floor[i_c]),
                "flank_n": float(c.flank[i_c]),
                "bolt_n": float(c.bolt[i_c]),
            },
            "bolt_range_n": [float(c.bolt.min()), float(c.bolt.max())],
            "separation_factor": self.separation_factor,
            "unseat_factor": self.unseat_factor,
            "swing": self.swing,
            "conformity": self.conformity,
            "sleeve": self.sleeve_summary(),
            "notes": self.notes + self.layout.notes,
        }

    def sleeve_summary(self) -> dict:
        """The sleeve's architecture and what it costs to print."""
        lay = self.layout
        out = {"core": lay.sleeve_core_kind,
               "skin_mm": lay.sleeve_skin,
               "face_shell_mm": lay.sleeve_face,
               "mass_kg": self.masses.get("rr_sleeve")}
        if lay.sleeve_core_kind != "sheet-gyroid" or not lay.lattice_core:
            return out
        from . import lattice as lattice_mod
        xc, yc, v0, v1 = lay.lattice_core
        rel = lattice_mod.mean_density(lay.lattice_rho_mid,
                                       lay.lattice_rho_end,
                                       lay.lattice_exponent)
        out.update({
            "cell_mm": lay.lattice_cell,
            "relative_density_mid": lay.lattice_rho_mid,
            "relative_density_end": lay.lattice_rho_end,
            "relative_density_mean": rel,
            "grading_exponent": lay.lattice_exponent,
            "core_mm": [2 * xc, 2 * yc, v1 - v0],
            "core_volume_mm3": 4.0 * xc * yc * (v1 - v0),
            "ports": len(lay.port_stations) * 2,
            "printability": lay.lattice_print,
            "cells": lay.lattice_cells,
            "evacuation": lay.lattice_evacuation,
        })
        return out


def self_seating(layout) -> dict:
    """Can bolt preload push the clamp tip INTO the notch at all?

    The clamp rides the crankpin on a circular bore, so about the crankpin
    centre it is a free hinge. The bolt turns it one way; a contact at the
    notch can only resist that if its line of action passes on the far side
    of the crankpin centre, which for a ramp at angle alpha through the
    notch at (x, v) means tan(alpha) > v / x. Below that angle, tightening
    the bolt swings the tip out of the notch, and preload has to be carried
    by the receiver seat and the lug faces instead.

    It is a RIGID-BODY criterion, so read it as necessary rather than
    sufficient: once the clamp is allowed to bend (the coupled network) a
    steeper ramp still has to overcome the clamp's own flexibility and the
    lug faces closing, and on the default proportions it does not.
    """
    n = layout.notch
    p = 0.5 * (n.T + n.F1)
    needed = math.degrees(math.atan2(p[1], p[0]))
    return {
        "ramp_angle_deg": math.degrees(n.alpha),
        "angle_for_self_seating_deg": needed,
        "self_seating": math.degrees(n.alpha) > needed,
        "explanation": (
            "the ramp's contact force only resists the bolt's turning moment "
            f"about the crankpin centre if the ramp is steeper than "
            f"{needed:.0f} deg at this location; shallower, the preload that "
            "reaches the notch comes from the seat interference alone"),
    }


def layout_masses(state, lay) -> dict:
    """Part masses from the 2D profiles, in kg."""
    from shapely import geometry

    rho = lambda key: materials_mod.get(key).density * 1e-9   # kg/mm^3 # noqa
    h = lay.rail_depth
    w = lay.width
    pi = math.pi

    lower = lay.rail_right.intersection(
        geometry.box(-1e4, lay.v_e - 1, 1e4, lay.v_joint))
    rails_mm3 = (lay.rails_upper.area * h + 2.0 * lower.area * h
                 + pi * (lay.eye_r_out ** 2 - lay.eye_r_in ** 2)
                 * max(lay.eye_width - h, 0.0)
                 - pi * lay.eye_r_in ** 2 * h)

    c_s = state["railrod.sleeve_clearance"] * 1e3
    b = lay.x_o - lay.x_i
    length = lay.L - lay.sleeve_v0
    column = geometry.box(-lay.sleeve_x, lay.sleeve_v0, lay.sleeve_x, lay.L)
    cut = column.intersection(lay.sleeve_cut)
    channel = 2 * (b + 2 * c_s) * (h + 2 * c_s)
    sleeve_mm3 = ((2 * lay.sleeve_x) * (2 * lay.sleeve_y) * length
                  - channel * length
                  - cut.area * 2 * lay.sleeve_y
                  + cut.intersection(geometry.box(
                      -lay.x_o - c_s, -1e4, -lay.x_i + c_s, 1e4)).area
                  * (h + 2 * c_s)
                  + cut.intersection(geometry.box(
                      lay.x_i - c_s, -1e4, lay.x_o + c_s, 1e4)).area
                  * (h + 2 * c_s))

    # The sleeve is not solid. What is left after the lattice core and the
    # powder ports are taken out is skin, face shell and the bearing pad that
    # sits in the receiver's seat; the core itself weighs its relative
    # density, averaged along the grading.
    core_mm3 = ports_mm3 = 0.0
    lattice_rel = 1.0
    if lay.sleeve_core_kind == "sheet-gyroid" and lay.lattice_core:
        from . import lattice as lattice_mod
        xc, yc, v0, v1 = lay.lattice_core
        core_mm3 = 4.0 * xc * yc * (v1 - v0)
        ports_mm3 = (2 * len(lay.port_stations) * pi
                     * (lay.lattice_port_diameter / 2.0) ** 2
                     * max(lay.sleeve_y - yc, 0.0))
        lattice_rel = lattice_mod.mean_density(
            lay.lattice_rho_mid, lay.lattice_rho_end, lay.lattice_exponent)
    sleeve_solid_mm3 = sleeve_mm3 - core_mm3 - ports_mm3
    sleeve_mm3 = sleeve_solid_mm3 + core_mm3 * lattice_rel

    # The receiver is the full big-end width, less three families of cut,
    # each of which has its OWN width along the crank axis. Taking the cuts
    # in order and removing each region only once keeps the overlaps (the
    # slots run up into the seat, the channels run into the slots) from
    # being subtracted twice.
    from .layout import mirror
    w_seat = min(w, 2.0 * (lay.sleeve_y + lay.seat_clearance))
    w_slot = min(w, h + 2.0 * lay.slot_clearance)
    w_chan = min(w, h + 2.0 * lay.channel_clearance)
    channels = lay.channel_right.union(mirror(lay.channel_right))
    seat2d = lay.envelope.intersection(lay.seat_rect)
    slot2d = lay.envelope.intersection(lay.slots).difference(lay.seat_rect)
    chan2d = lay.envelope.intersection(channels).difference(
        lay.seat_rect).difference(lay.slots)
    receiver_mm3 = (lay.envelope.area * w - seat2d.area * w_seat
                    - slot2d.area * w_slot - chan2d.area * w_chan)

    d_b = state["railrod.bolt_diameter"] * 1e3
    pitch = state["railrod.bolt_pitch"] * 1e3
    r_minor = (d_b - 1.0825 * pitch) / 2.0

    # Each clamp is two widths: the full big end where it is the bearing cap,
    # the channel's width where its tongue runs inside the receiver.
    def _clamp_mm3(poly, hole_r):
        inner = poly.intersection(lay.envelope).area
        outer = poly.area - inner
        return outer * w + inner * w_chan - pi * hole_r ** 2 * lay.x_lug

    clamp_r = _clamp_mm3(lay.clamp_right, r_minor)
    clamp_l = _clamp_mm3(lay.clamp_left, lay.r_hole)

    head_h = 0.7 * d_b
    bolt_mm3 = (pi * (d_b / 2) ** 2 * (2 * lay.x_lug - 0.5)
                + pi * (0.75 * d_b) ** 2 * head_h)

    big = state["railrod.bigend_material"]
    return {
        "rr_rails": rails_mm3 * rho(state["materials.rod"]),
        "rr_sleeve": sleeve_mm3 * rho(state["railrod.sleeve_material"]),
        "rr_receiver": receiver_mm3 * rho(big),
        "rr_clamp_right": clamp_r * rho(big),
        "rr_clamp_left": clamp_l * rho(big),
        "rr_bolt": bolt_mm3 * rho(state["railrod.bolt_material"]),
    }


def _scaled_factor(net, assembly, force, accel, key, target_sign, limit=12.0):
    """Multiple of ``force`` at which contact group ``key`` first reaches zero.

    Bisection on the full nonlinear network, because the answer is exactly
    the point where the network changes state and a linear extrapolation
    through it would be meaningless.
    """
    def value(scale):
        r = net.solve(force * scale, accel * scale, assembly=assembly,
                      active=assembly["closed"])
        return r[key] if key != "flank" else r["flank"]

    if value(1.0) <= 1e-6:
        lo, hi = 0.0, 1.0
    else:
        lo, hi = 1.0, limit
        if value(hi) > 1e-6:
            return float("inf")
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if value(mid) > 1e-6:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def fingerprint(state) -> str:
    return state.fingerprint()


def analyse(state, sweep=None, use_cache: bool = True) -> RailRodAnalysis:
    """Lay out, weigh and load the rail rod. Raises LayoutError if the
    dimensions cannot be built."""
    key = state.fingerprint()
    if use_cache and key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]

    if sweep is None:
        from ..evaluate import evaluate
        sweep = evaluate(state).sweep

    lay = layout_mod.build_layout(state)
    masses = layout_masses(state, lay)

    class _Decimated:
        pass

    dec = _Decimated()
    idx = np.arange(0, len(sweep.theta), DECIMATE)
    dec.theta = np.asarray(sweep.theta)[idx]
    dec.f_rod = np.asarray(sweep.f_rod)[idx]
    cycle = run_cycle(state, lay, dec, masses)

    net = cycle.network
    pre = cycle.preload_state
    i_t = int(np.argmin(cycle.f_rod))
    i_c = int(np.argmax(cycle.f_rod))
    notes: list[str] = []
    sep = (_scaled_factor(net, pre, float(cycle.f_rod[i_t]),
                          float(cycle.accel_bigend_v[i_t]), "floor", -1)
           if cycle.f_rod[i_t] < 0 else float("inf"))
    unseat = _scaled_factor(net, pre, float(cycle.f_rod[i_c]),
                            float(cycle.accel_bigend_v[i_c]), "flank", 1)
    if pre["floor"] <= 1e-6:
        notes.append(
            "the rail feet carry no preload at assembly: the clamp tips do "
            "not clamp the rails onto the receiver floor, so the rails lift "
            "off it on every tensile stroke")

    result = RailRodAnalysis(
        layout=lay, masses=masses, cycle=cycle,
        swing=layout_mod.swing_check(lay),
        conformity=layout_mod.conformity(lay),
        separation_factor=sep, unseat_factor=unseat, notes=notes)
    if use_cache:
        _CACHE[key] = result
        while len(_CACHE) > _CACHE_LIMIT:
            _CACHE.popitem(last=False)
    return result
