"""Piston crown: pressure bending, thermal gradient, and thermal fatigue.

The crown is the only part of the system loaded directly by combustion, and it
does that job at nearly 300 C, where 2618-T61 keeps under a third of its
room-temperature yield strength. Both halves of that sentence matter, and the
second one is why crown temperature is modelled at all.

Mechanical model: a clamped circular plate of radius a and thickness t under
uniform pressure q::

    sigma_edge   = 3 q a^2 / (4 t^2)
    sigma_centre = 3 q a^2 (1 + nu) / (8 t^2)
    w_max        = q a^4 / (64 D),   D = E t^3 / (12 (1 - nu^2))

Two locations, two temperatures. The edge is at the crown periphery, close to
the ring belt, and runs at roughly the top-land temperature. The centre is the
hottest point on the piston. The edge carries about twice the stress; the
centre sits at a much lower allowable. Which one governs is not obvious in
advance, so both are computed.

**The effective radius is the weak point of this model.** A real crown does
not span the whole bore: it is braced underneath by the pin bosses and around
its rim by the ring belt and skirt structure. ``crown_support_radius_fraction``
captures that, and the stress goes as its square, so the result is only as good
as that one estimate. Phase 6 FEA is what settles it.

Static checks compare nominal stress against yield -- no notch factor, because
a ductile aluminium crown redistributes stress by yielding locally at a
fillet. The notch factor belongs in the fatigue check, where it is applied.
"""

from __future__ import annotations

import math

from ..margins import ComponentContext, Margin, fatigue_margin

# Coffin-Manson constants for wrought aluminium. Generic literature values.
EPSILON_F = 0.35
EXPONENT_C = -0.60
START_STOP_CYCLES_REQUIRED = 20_000.0   # ~25 years at two starts a day


def evaluate(ctx: ComponentContext) -> list:
    state, thermal = ctx.state, ctx.thermal

    a_full = state["engine.bore"] / 2.0
    a = a_full * state["piston.crown_support_radius_fraction"]
    t = state["piston.crown_thickness"]
    q = ctx.sweep.trace.peak_pressure - state["operating.crankcase_pressure"]

    material = ctx.material("piston")
    nu, e_mod = material.poisson, material.youngs_modulus

    sigma_edge = 3.0 * q * a * a / (4.0 * t * t)
    sigma_centre = 3.0 * q * a * a * (1.0 + nu) / (8.0 * t * t)
    rigidity = e_mod * t ** 3 / (12.0 * (1.0 - nu * nu))
    deflection = q * a ** 4 / (64.0 * rigidity)

    gradient = thermal.crown_gradient
    sigma_thermal = e_mod * material.cte * gradient / (2.0 * (1.0 - nu))

    radius_note = (
        f"effective crown radius is {a * 1e3:.1f} mm, "
        f"{state['piston.crown_support_radius_fraction']:.2f} of the bore "
        "radius. Stress scales with its square, so this single estimate "
        "dominates the result")

    common = {
        "pressure_pa": q,
        "effective_radius_m": a,
        "bore_radius_m": a_full,
        "support_fraction": state["piston.crown_support_radius_fraction"],
        "crown_thickness_m": t,
        "thermal_stress_pa": sigma_thermal,
        "gradient_k": gradient,
        "material": material.key,
    }

    margins = []
    for label, stress, temperature in (
            ("edge (crown-to-wall junction)", sigma_edge, thermal.top_land),
            ("centre (hottest point)", sigma_centre, thermal.crown_top)):
        allowable, basis = ctx.strength(material, temperature)
        retained = material.strength_factor_at(temperature)
        inputs = dict(common)
        inputs.update({
            "bending_stress_pa": stress,
            "total_stress_pa": stress + sigma_thermal,
            "temperature_c": temperature - 273.15,
            "strength_retained_at_temperature": retained,
            "allowable_basis": basis,
        })
        margins.append(Margin(
            component="crown", mode=f"pressure plus thermal stress, {label}",
            applied=stress + sigma_thermal, allowable=allowable, unit="Pa",
            equation=("sigma = 3 q a^2 / (4 t^2) + E alpha dT / (2(1-nu))"
                      if "edge" in label else
                      "sigma = 3 q a^2 (1+nu) / (8 t^2) + E alpha dT / (2(1-nu))"),
            reference="clamped circular plate under uniform pressure; "
                      "through-thickness thermal gradient",
            condition=ctx.at("peak firing pressure"),
            inputs=inputs,
            notes=[radius_note,
                   f"this location runs at {temperature - 273.15:.0f} C and "
                   f"keeps {retained * 100:.0f}% of room-temperature strength",
                   "no notch factor: a ductile crown redistributes by yielding "
                   "locally, so Kf belongs in the fatigue check, not here",
                   "mechanical and thermal stress are superposed, which is "
                   "conservative"]))

    margins.append(Margin(
        component="crown", mode="deflection under peak pressure",
        applied=deflection, allowable=t * 0.10, unit="m",
        equation="w = q a^4 / (64 D),  D = E t^3 / (12(1-nu^2))",
        reference="clamped circular plate, small-deflection theory",
        condition=ctx.at("peak firing pressure"),
        inputs={"deflection_m": deflection, "crown_thickness_m": t,
                "flexural_rigidity": rigidity, "effective_radius_m": a},
        notes=["the limit is a working rule of 10% of crown thickness: past "
               "that the small-deflection solution stops being valid, so the "
               "number above would itself be wrong"]))

    kf = state["fatigue.kf_crown"]
    margins.append(fatigue_margin(
        ctx, "crown", "high-cycle fatigue at the crown-to-wall junction",
        alternating=kf * sigma_edge / 2.0,
        mean=kf * sigma_edge / 2.0 + sigma_thermal,
        material=material, temperature=thermal.top_land,
        diameter=t, finish="machined", load="bending",
        condition=ctx.at("gas load cycling once per engine cycle"),
        extra_inputs={"notch_factor": kf, "effective_radius_m": a},
        notes=[radius_note,
               "thermal stress is carried as a steady mean stress, which it is "
               "at constant load but is not during warm-up"]))

    margins.append(_thermal_fatigue(ctx, material, thermal))
    return margins


def _thermal_fatigue(ctx, material, thermal) -> Margin:
    """Coffin-Manson screening life for the start-stop thermal cycle."""
    state = ctx.state
    t_hot = thermal.crown_top
    t_cold = state["thermal.coolant_temperature"] - 70.0      # cold soak
    delta_t = max(t_hot - t_cold, 0.0)
    constraint = state["piston.thermal_constraint_factor"]

    allowable, _ = ctx.strength(material, t_hot)
    strain_total = material.cte * delta_t * constraint
    strain_elastic = 2.0 * allowable / material.youngs_modulus
    strain_plastic = max(strain_total - strain_elastic, 0.0)

    if strain_plastic <= 0.0:
        cycles = math.inf
    else:
        cycles = 0.5 * (strain_plastic / (2.0 * EPSILON_F)) ** (1.0 / EXPONENT_C)

    factor = (math.inf if math.isinf(cycles)
              else cycles / START_STOP_CYCLES_REQUIRED)

    return Margin(
        component="crown", mode="thermal-mechanical fatigue (start-stop)",
        kind="factor", applied=factor, allowable=1.0, unit="-",
        equation="d_eps_p / 2 = eps_f' (2N)^c, with d_eps_p from constrained "
                 "thermal expansion less the elastic range",
        reference="Coffin-Manson low-cycle fatigue; generic wrought aluminium "
                  "constants (eps_f' = 0.35, c = -0.60)",
        condition="cold soak to full load and back",
        inputs={
            "delta_t_k": delta_t,
            "constraint_factor": constraint,
            "total_strain": strain_total,
            "elastic_strain": strain_elastic,
            "plastic_strain": strain_plastic,
            "cycles_to_initiation": None if math.isinf(cycles) else cycles,
            "cycles_required": START_STOP_CYCLES_REQUIRED,
        },
        notes=["SCREENING CHECK ONLY. The answer is dominated by the "
               f"constraint factor ({constraint:.2f}), which is an estimate: "
               "the plastic strain is a small difference between two larger "
               "numbers, so a 20% change there moves the predicted life by an "
               "order of magnitude",
               "real thermal-mechanical fatigue needs transient thermal FEA "
               "and a cyclic plasticity model",
               "these are engine start-stop cycles, not crank revolutions",
               "it also assumes every cycle reaches full load at the current "
               "speed, which no road engine does; for a duty cycle that spends "
               "most of its life at part load the real figure is far better"],
        life_cycles=cycles)
