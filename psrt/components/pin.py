"""Gudgeon pin: bending between the bosses, and ovalisation.

The pin is a short, thick tube loaded in three-point bending: the rod small
end pushes in the middle, the two pin bosses react at the ends. Because the
load reverses -- compression at peak firing, tension at overlap TDC -- the
bending stress is fully reversed, which is the worst case for fatigue and the
reason the pin is usually a hardened steel tube rather than anything lighter.

Bending. For a simply supported beam of span S carrying a load W spread
uniformly over a central length c (the small-end width)::

    M_max = W (2 S - c) / 8

The span is taken between the centroids of the two boss bearing areas.

Ovalisation. A pin also flattens under diametral load, and a pin that ovalises
too much fatigues its own boss bores rather than itself. The Kolbenschmidt
relation for the change in horizontal diameter::

    dD = 0.09 (F / (l E)) ((Da + Di) / (Da - Di))^3

That cubed term is why pin bore diameter is such a sharp trade: boring the pin
out to save reciprocating mass costs ovality stiffness very quickly.
"""

from __future__ import annotations

import numpy as np

from ..margins import ComponentContext, Margin, fatigue_margin

KOLBENSCHMIDT_COEFFICIENT = 0.09


def _bending_stress(force: float, span: float, width: float,
                    modulus: float) -> float:
    return (force * (2.0 * span - width) / 8.0) / modulus


def evaluate(ctx: ComponentContext) -> list:
    state, sweep, thermal = ctx.state, ctx.sweep, ctx.thermal

    span = state["pin.bending_span"]
    width = state["small_end.bushing_width"]
    modulus = state["pin.section_modulus"]
    d_out = state["pin.outer_diameter"]
    d_in = state["pin.inner_diameter"]

    compression = sweep.peak_pin_compression
    tension = sweep.peak_pin_tension
    governing = (compression if abs(compression.value) >= abs(tension.value)
                 else tension)

    sigma_max = _bending_stress(compression.value, span, width, modulus)
    sigma_min = _bending_stress(tension.value, span, width, modulus)
    sigma_governing = max(abs(sigma_max), abs(sigma_min))

    material = ctx.material("pin")
    temperature = thermal.pin
    allowable, basis = ctx.strength(material, temperature)

    force = abs(governing.value)
    ratio = (d_out + d_in) / (d_out - d_in)
    ovality = (KOLBENSCHMIDT_COEFFICIENT * force
               / (state["pin.length"] * material.youngs_modulus) * ratio ** 3)

    kf = state["fatigue.kf_pin"]

    return [
        Margin(
            component="pin", mode="bending between the bosses",
            applied=sigma_governing, allowable=allowable, unit="Pa",
            equation="sigma = M / Z,  M = W (2S - c) / 8",
            reference="simply supported beam, central load distributed over "
                      "the small-end width",
            condition=ctx.at(f"{'peak firing' if governing is compression else 'overlap TDC'}, "
                             f"{governing.angle_deg:.0f} deg"),
            inputs={
                "governing_force_n": governing.value,
                "peak_compression_n": compression.value,
                "peak_tension_n": tension.value,
                "span_m": span,
                "small_end_width_m": width,
                "section_modulus_m3": modulus,
                "outer_diameter_m": d_out,
                "inner_diameter_m": d_in,
                "stress_at_peak_compression_pa": sigma_max,
                "stress_at_peak_tension_pa": sigma_min,
                "temperature_c": temperature - 273.15,
                "material": material.key,
                "allowable_basis": basis,
            },
            notes=["the boss reaction is assumed uniform; a real boss loads "
                   "its inner edge harder, which raises this",
                   "no notch factor on the static check; Kf is applied to the "
                   "fatigue margin"],
        ),
        Margin(
            component="pin", mode="ovalisation",
            applied=ovality, allowable=state["pin.max_ovality"], unit="m",
            equation="dD = 0.09 (F / (l E)) ((Da + Di)/(Da - Di))^3",
            reference="Kolbenschmidt pin ovality relation",
            condition=ctx.at(f"peak pin load, {governing.angle_deg:.0f} deg"),
            inputs={
                "force_n": force,
                "pin_length_m": state["pin.length"],
                "diameter_ratio_term": ratio,
                "ovality_m": ovality,
                "wall_thickness_m": (d_out - d_in) / 2.0,
            },
            notes=["excessive ovality fatigues the pin BOSS bores rather than "
                   "the pin itself, and that failure is not modelled here",
                   "the diameter ratio is cubed, so boring the pin out to save "
                   "mass costs ovality stiffness very fast"],
        ),
        fatigue_margin(
            ctx, "pin", "fully reversed bending fatigue",
            alternating=kf * (sigma_max - sigma_min) / 2.0,
            mean=kf * (sigma_max + sigma_min) / 2.0,
            material=material, temperature=temperature,
            diameter=d_out, finish="ground", load="bending",
            condition=ctx.at("load reversing between firing and overlap TDC"),
            extra_inputs={"notch_factor": kf},
            notes=["the pin sees fully reversed bending once per revolution, "
                   "which is the harshest fatigue case in the system"]),
    ]
