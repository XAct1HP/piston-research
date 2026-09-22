"""Cylinder sleeve: hoop stress and how much wall is left.

The sleeve is the one component that cannot be designed freely. Material comes
off the block and never goes back on, so every millimetre of overbore is a
millimetre off the wall that has to hold peak firing pressure and conduct heat
to the coolant.

Model: Lame's thick-walled cylinder under internal pressure. For internal
pressure alone, the hoop stress is largest at the bore surface::

    sigma_theta = p (b^2 + a^2) / (b^2 - a^2)
    sigma_r     = -p

with a the bore radius and b the outside radius of the wall.

What this does NOT capture: bore distortion under head-bolt clamping, which is
often what actually limits how thin a wall can go, because a distorted bore
stops the rings sealing long before the wall fails in hoop. That needs the FEA
layer in phase 6.
"""

from __future__ import annotations

import math

from ..margins import ComponentContext, Margin


def evaluate(ctx: ComponentContext) -> list:
    state, thermal = ctx.state, ctx.thermal

    a = state["engine.bore"] / 2.0
    wall = state["sleeve.wall_thickness"]
    b = a + wall
    p = ctx.sweep.trace.peak_pressure

    hoop = p * (b * b + a * a) / (b * b - a * a)
    radial = -p
    von_mises = math.sqrt(hoop * hoop - hoop * radial + radial * radial)

    material = ctx.material("sleeve")
    temperature = thermal.liner_at_tdc
    allowable, basis = ctx.strength(material, temperature)

    measured = state.get("sleeve.wall_thickness_measured", None)
    wall_note = ("wall thickness is measured" if measured else
                 "wall thickness is a bore-spacing estimate, not a "
                 "measurement: sonic-test the block and set "
                 "sleeve.wall_thickness_measured before acting on this")

    margins = [
        Margin(
            component="sleeve", mode="bore wall hoop stress",
            applied=von_mises, allowable=allowable, unit="Pa",
            equation="Lame: sigma_theta = p (b^2 + a^2)/(b^2 - a^2); "
                     "von Mises with sigma_r = -p",
            reference="Lame thick-walled cylinder, internal pressure",
            condition=ctx.at("peak firing pressure"),
            inputs={
                "peak_pressure_pa": p,
                "bore_radius_m": a,
                "outer_radius_m": b,
                "wall_thickness_m": wall,
                "hoop_stress_pa": hoop,
                "radial_stress_pa": radial,
                "material": material.key,
                "temperature_c": temperature - 273.15,
                "allowable_basis": basis,
            },
            notes=[wall_note,
                   "bore distortion under head-bolt clamping is not modelled; "
                   "it often limits wall thickness before hoop stress does",
                   ("grey iron carries far more in compression than tension, "
                    "so this is conservative") if material.yield_strength is None
                   else ""],
        ),
        Margin(
            component="sleeve", mode="wall thickness remaining",
            applied=state["block.min_wall_to_coolant"], allowable=wall,
            unit="m",
            equation="wall = (bore spacing - bore)/2, or the measured value",
            reference="subtractive manufacturing envelope",
            condition="geometric, independent of operating point",
            operating_dependent=False,
            inputs={
                "wall_thickness_m": wall,
                "minimum_acceptable_m": state["block.min_wall_to_coolant"],
                "bore_m": state["engine.bore"],
                "bore_spacing_m": state["block.bore_spacing"],
                "overbore_from_as_built_m":
                    state["engine.bore"] - state["block.as_built_bore"],
            },
            notes=[wall_note],
        ),
    ]
    for m in margins:
        m.notes = [n for n in m.notes if n]
    return margins
