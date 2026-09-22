"""Ring lands and grooves.

The top land is the strip of piston between the crown face and the top ring
groove. Gas gets behind the top ring, pressurises the groove, and tries to peel
the land outward. It is modelled as an annular cantilever built in at the
groove root, carrying the cylinder pressure over its radial depth::

    M = p h^2 / 2   per unit circumference
    Z = b^2 / 6     per unit circumference
    sigma = 3 p h^2 / b^2

with h the groove depth and b the land height. The groove root is a sharp
internal corner, so the notch factor here is the largest in the model.

Groove flank pressure is reported for completeness. For a flank in full
contact it works out close to cylinder pressure, which is why grooves rarely
fail statically -- they fail by pounding, a dynamic impact process this model
does not capture at all.
"""

from __future__ import annotations

import math

from ..margins import ComponentContext, Margin, fatigue_margin


def evaluate(ctx: ComponentContext) -> list:
    state, thermal = ctx.state, ctx.thermal

    p = ctx.sweep.trace.peak_pressure
    h = state["piston.ring_groove_depth"]
    b = state["piston.top_land_height"]
    kf = state["fatigue.kf_land"]

    bending = 3.0 * p * h * h / (b * b)

    material = ctx.material("piston")
    temperature = thermal.top_land
    allowable, basis = ctx.strength(material, temperature)

    bore = state["engine.bore"]
    root = state["piston.ring_groove_root_diameter"]
    ring_annulus = math.pi * (bore ** 2 - root ** 2) / 4.0
    flank_force = p * ring_annulus
    flank_pressure = flank_force / ring_annulus if ring_annulus > 0 else math.inf

    return [
        Margin(
            component="ring lands", mode="top land bending at the groove root",
            applied=bending, allowable=allowable, unit="Pa",
            equation="sigma = 3 p h^2 / b^2",
            reference="annular cantilever built in at the groove root",
            condition=ctx.at("peak firing pressure"),
            inputs={
                "peak_pressure_pa": p,
                "groove_depth_m": h,
                "land_height_m": b,
                "notch_factor_used_in_fatigue_only": kf,
                "land_temperature_c": temperature - 273.15,
                "material": material.key,
                "allowable_basis": basis,
            },
            notes=["no notch factor on the static check: aluminium "
                   "redistributes by yielding at the groove root, so Kf is "
                   "applied to the fatigue margin instead",
                   "the land is stiffer than a plain cantilever because it is "
                   "an annulus, so this is conservative",
                   "top land height and groove depth trade directly: deeper "
                   "grooves and shorter lands both raise this stress as the "
                   "square"],
        ),
        Margin(
            component="ring lands", mode="groove flank pressure",
            applied=flank_pressure,
            allowable=state["piston.allowable_groove_pressure"], unit="Pa",
            equation="p_flank = F_axial / A_contact, F_axial = p * A_ring",
            reference="ring seated on the lower groove flank by gas pressure",
            condition=ctx.at("peak firing pressure"),
            inputs={
                "ring_annulus_area_m2": ring_annulus,
                "axial_force_n": flank_force,
                "groove_root_diameter_m": root,
            },
            notes=["for a flank in full contact this reduces to cylinder "
                   "pressure, so a comfortable factor here means little",
                   "grooves fail by pounding, a dynamic impact process that is "
                   "not modelled; treat this as a lower bound on severity"],
        ),
        fatigue_margin(
            ctx, "ring lands", "top land fatigue",
            alternating=kf * bending / 2.0, mean=kf * bending / 2.0,
            material=material, temperature=temperature,
            diameter=b, finish="machined", load="bending",
            condition=ctx.at("gas load cycling once per engine cycle"),
            extra_inputs={"notch_factor": kf}),
    ]
