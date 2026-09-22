"""Piston skirt: side thrust, bearing pressure and scuffing.

Rod obliquity turns axial force into side thrust, and the skirt is what reacts
it. The failure mode is not fracture, it is scuffing: the oil film breaks down,
aluminium welds to iron, and the bore is ruined in seconds.

Scuffing is a lubrication problem, so the honest model would be
elastohydrodynamic. What is used instead is the industry proxy -- projected
specific pressure against an allowable that comes from operating practice::

    p = |F_side| / (skirt length x bearing arc x bore)

The pV factor is reported alongside it, because pressure at low sliding speed
is much less dangerous than the same pressure at 20 m/s. There is no allowable
attached to pV here: inventing one would be worse than leaving it as context.
"""

from __future__ import annotations

import numpy as np

from ..margins import ComponentContext, Margin


def evaluate(ctx: ComponentContext) -> list:
    state, sweep = ctx.state, ctx.sweep

    area = state["piston.skirt_bearing_area"]
    index = int(np.argmax(np.abs(sweep.f_side)))
    side_force = abs(float(sweep.f_side[index]))
    angle = float(np.degrees(sweep.theta[index]))
    sliding_speed = abs(float(sweep.velocity[index]))

    pressure = side_force / area
    pv_factor = pressure * sliding_speed

    # The worst pV in the cycle is not always at peak side force.
    pv_series = np.abs(sweep.f_side) / area * np.abs(sweep.velocity)
    pv_index = int(np.argmax(pv_series))

    return [
        Margin(
            component="skirt", mode="specific pressure (scuffing)",
            applied=pressure,
            allowable=state["piston.allowable_skirt_pressure"], unit="Pa",
            equation="p = |F_side| / (L_skirt * bore * bearing arc)",
            reference="projected bearing pressure; allowable is operating "
                      "practice for a coated aluminium skirt on iron",
            condition=ctx.at(f"peak side thrust, {angle:.0f} deg"),
            inputs={
                "side_force_n": side_force,
                "bearing_area_m2": area,
                "skirt_length_m": state["piston.skirt_length"],
                "bearing_arc_fraction": state["piston.skirt_bearing_arc"],
                "sliding_speed_m_s": sliding_speed,
                "pv_factor_pa_m_s": pv_factor,
                "worst_pv_pa_m_s": float(pv_series[pv_index]),
                "worst_pv_angle_deg": float(np.degrees(sweep.theta[pv_index])),
                "skirt_temperature_c": ctx.thermal.skirt - 273.15,
            },
            notes=["scuffing is a lubrication failure; this specific-pressure "
                   "proxy is industry practice, not physics, and no oil film "
                   "is modelled",
                   "the bearing arc fraction is an estimate and the result "
                   "scales directly with it",
                   "peak side thrust and peak pV occur at different crank "
                   "angles; both are reported above"],
        ),
    ]
