"""Small end: projected bearing pressure on the bushing, both ways.

Two loads, two very different problems. At peak firing the small end is pushed
down onto the pin and the load spreads over the lower half of the bore -- lots
of material behind it. At overlap TDC the load reverses and pulls up against
the top of the small-end eye, which is the thinnest section of the whole rod.

The compressive case usually shows the higher pressure; the tensile case is
usually the one that cracks rods. Both are reported.
"""

from __future__ import annotations

from ..margins import ComponentContext, Margin


def evaluate(ctx: ComponentContext) -> list:
    state, sweep, thermal = ctx.state, ctx.sweep, ctx.thermal

    projected = state["pin.outer_diameter"] * state["small_end.bushing_width"]
    material = ctx.material_key(state["small_end.material"])
    temperature = thermal.pin
    retained = material.strength_factor_at(temperature)
    allowable = state["small_end.allowable_pressure"] * retained

    compression = sweep.peak_pin_compression
    tension = sweep.peak_pin_tension

    margins = []
    for label, extremum, share in (
            ("compression at peak firing", compression, 1.0),
            ("tension at overlap TDC", tension, 1.0)):
        pressure = abs(extremum.value) * share / projected
        margins.append(Margin(
            component="small end", mode=f"projected bearing pressure, {label}",
            applied=pressure, allowable=allowable, unit="Pa",
            equation="p = F / (d_pin * L_bushing)",
            reference="projected bearing area; allowable is bushing operating "
                      "practice derated for temperature",
            condition=ctx.at(f"{extremum.angle_deg:.0f} deg"),
            inputs={
                "force_n": extremum.value,
                "projected_area_m2": projected,
                "pin_diameter_m": state["pin.outer_diameter"],
                "bushing_width_m": state["small_end.bushing_width"],
                "bushing_material": material.key,
                "temperature_c": temperature - 273.15,
                "strength_retained_at_temperature": retained,
                "allowable_at_20c_pa": state["small_end.allowable_pressure"],
            },
            notes=(["the tensile case loads the thinnest section of the rod eye; "
                    "the eye's own hoop stress is not modelled and needs FEA"]
                   if "tension" in label else
                   ["load is assumed uniform over the projected area; real "
                    "contact concentrates it"]),
        ))
    return margins
