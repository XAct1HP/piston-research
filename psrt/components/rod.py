"""Connecting rod: tension, fatigue, and buckling about both axes.

Three checks, three different governing conditions.

**Tension** at overlap TDC. No gas pressure opposes the inertia of the
reciprocating mass, so the whole of it goes into the shank in tension. This is
how connecting rods actually break, and it gets worse as the square of engine
speed while the engine is making its least torque.

**Fatigue** across the whole cycle. The shank swings from heavy compression at
peak firing to heavy tension at overlap TDC, once per revolution. The mean
stress usually comes out compressive, which Goodman does not penalise -- so the
tensile excursion governs.

**Buckling** at peak firing, checked about both axes because they have
different end restraint::

    lambda = K L / r,   lambda_c = sqrt(2 pi^2 E / S_y)
    Euler   (lambda > lambda_c):  sigma_cr = pi^2 E / lambda^2
    Johnson (lambda <= lambda_c): sigma_cr = S_y - (S_y lambda / 2 pi)^2 / E

In the plane of rotation the rod is pinned at both ends, K = 1. Out of plane
the pin bosses and the big end restrain rotation, K is nearer 0.6. The section
carries more material about the in-plane axis to match -- which is exactly why
rods are I-sections rather than round bars.
"""

from __future__ import annotations

import math

from ..margins import ComponentContext, Margin, fatigue_margin


def critical_stress(slenderness: float, yield_strength: float,
                    modulus: float) -> tuple:
    """Euler above the transition slenderness, Johnson parabolic below."""
    if slenderness <= 0.0:
        return yield_strength, "Johnson (stocky)"
    transition = math.sqrt(2.0 * math.pi ** 2 * modulus / yield_strength)
    if slenderness > transition:
        return (math.pi ** 2 * modulus / slenderness ** 2, "Euler (slender)")
    johnson = yield_strength - (yield_strength * slenderness
                                / (2.0 * math.pi)) ** 2 / modulus
    return johnson, "Johnson (intermediate)"


def evaluate(ctx: ComponentContext) -> list:
    state, sweep, thermal = ctx.state, ctx.sweep, ctx.thermal

    area = state["rod.shank_area"]
    material = ctx.material("rod")
    temperature = thermal.rod
    allowable, basis = ctx.strength(material, temperature)
    yield_strength = allowable

    tension = abs(sweep.peak_rod_tension.value)
    compression = sweep.peak_rod_compression.value

    sigma_tension = tension / area
    sigma_compression = compression / area
    kf = state["fatigue.kf_rod"]
    finish = state["rod.surface_finish"]

    margins = [
        Margin(
            component="rod", mode="shank tension at overlap TDC",
            applied=sigma_tension, allowable=allowable, unit="Pa",
            equation="sigma = F_tension / A_shank",
            reference="direct tension on the minimum shank section",
            condition=ctx.at(
                f"overlap TDC, {sweep.peak_rod_tension.angle_deg:.0f} deg"),
            inputs={
                "tensile_force_n": tension,
                "shank_area_m2": area,
                "reciprocating_mass_kg": sweep.reciprocating_mass,
                "temperature_c": temperature - 273.15,
                "material": material.key,
                "allowable_basis": basis,
            },
            notes=["this load grows as the square of engine speed while the "
                   "engine makes its least torque",
                   "no notch factor on the static check; Kf is applied to the "
                   "fatigue margin"],
        ),
    ]

    for axis, slenderness_key, k_key in (
            ("in the plane of rotation", "rod.slenderness_in_plane",
             "rod.k_in_plane"),
            ("out of the plane of rotation", "rod.slenderness_out_of_plane",
             "rod.k_out_of_plane")):
        slenderness = state[slenderness_key]
        sigma_cr, regime = critical_stress(
            slenderness, yield_strength, material.youngs_modulus)
        transition = math.sqrt(2.0 * math.pi ** 2 * material.youngs_modulus
                               / yield_strength)
        margins.append(Margin(
            component="rod", mode=f"buckling {axis}",
            applied=sigma_compression, allowable=sigma_cr, unit="Pa",
            equation=("sigma_cr = pi^2 E / lambda^2" if "Euler" in regime
                      else "sigma_cr = S_y - (S_y lambda / 2 pi)^2 / E"),
            reference=f"{regime} column formula, lambda = K L / r",
            condition=ctx.at(
                f"peak firing, {sweep.peak_rod_compression.angle_deg:.0f} deg"),
            inputs={
                "compressive_force_n": compression,
                "compressive_stress_pa": sigma_compression,
                "slenderness": slenderness,
                "transition_slenderness": transition,
                "effective_length_factor": state[k_key],
                "rod_length_m": state["engine.rod_length"],
                "critical_stress_pa": sigma_cr,
                "regime": regime,
            },
            notes=[f"slenderness {slenderness:.1f} against a transition at "
                   f"{transition:.1f}, so {regime.split()[0]} governs",
                   "the shank is treated as prismatic; real rods taper toward "
                   "the small end, which this ignores"]))

    margins.append(fatigue_margin(
        ctx, "rod", "shank fatigue over the full cycle",
        alternating=kf * (sigma_tension + sigma_compression) / 2.0,
        mean=kf * (sigma_tension - sigma_compression) / 2.0,
        material=material, temperature=temperature,
        diameter=math.sqrt(4.0 * area / math.pi), finish=finish, load="axial",
        condition=ctx.at("full cycle, firing compression to overlap tension"),
        extra_inputs={
            "notch_factor": kf,
            "tensile_stress_pa": sigma_tension,
            "compressive_stress_pa": -sigma_compression,
        },
        notes=["mean stress comes out compressive on most rods, and Goodman "
               "does not penalise that, so the tensile excursion governs",
               f"surface finish '{finish}' sets the largest single Marin "
               "factor here"]))
    return margins
