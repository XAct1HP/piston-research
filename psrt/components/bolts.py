"""Rod bolts: preload, separation, and fatigue.

Rod bolts are the most highly stressed parts in the engine and the least
forgiving. They sit at 60 to 80 percent of proof load before the engine has
even turned over, and then take a tensile pulse once per revolution forever.

Three checks, and they fail in a specific order.

**Separation** goes first. While the joint stays clamped the bolt only sees a
fraction C of the external load -- typically a quarter. The moment the cap
lifts, the bolt takes all of it, and every fatigue number below becomes
meaningless. This is why a rod bolt failure is usually sudden.

**Fatigue** is next, at the thread root, using Shigley's factor for a
preloaded joint::

    n_f = S_e (S_ut - sigma_i) / (sigma_a (S_ut + S_e))

**Yielding** at the proof strength is last, and rarely governs.

The load on the bolts is the rod's tensile load at overlap TDC plus the
centrifugal force of the cap itself, which at TDC points away from the crank
centre -- the same direction the rod is already pulling.
"""

from __future__ import annotations

from ..margins import ComponentContext, Margin
from ..fatigue import endurance_limit, preloaded_bolt


def evaluate(ctx: ComponentContext) -> list:
    state, sweep, thermal = ctx.state, ctx.sweep, ctx.thermal

    rod_tension = abs(sweep.at_overlap_tdc()["f_rod"])
    crank_radius = state["engine.crank_radius"]
    cap_centrifugal = (state["bolts.cap_mass"] * sweep.speed ** 2 * crank_radius)
    total = rod_tension + cap_centrifugal

    count = int(state["bolts.count"])
    per_bolt = total / count

    area = state["bolts.stress_area"]
    material = ctx.material_key(state["bolts.material"])
    temperature = thermal.rod
    limit = ctx.endurance(material, temperature,
                          state["bolts.thread_diameter"],
                          state["bolts.surface_finish"], "axial")

    result = preloaded_bolt(
        preload=state["bolts.preload"], external_load=per_bolt,
        stress_area=area,
        stiffness_factor=state["bolts.joint_stiffness_factor"],
        s_e=limit.value, s_ut=limit.ultimate,
        proof_strength=state["bolts.proof_strength"])

    shared = {
        "rod_tension_at_overlap_tdc_n": rod_tension,
        "cap_centrifugal_n": cap_centrifugal,
        "total_external_load_n": total,
        "load_per_bolt_n": per_bolt,
        "bolt_count": count,
        "stress_area_m2": area,
        "preload_n": state["bolts.preload"],
        "preload_stress_pa": result.preload_stress,
        "preload_fraction_of_proof":
            result.preload_stress / state["bolts.proof_strength"],
        "joint_stiffness_factor": state["bolts.joint_stiffness_factor"],
        "alternating_stress_pa": result.alternating_stress,
        "mean_stress_pa": result.mean_stress,
        "material": material.key,
        "temperature_c": temperature - 273.15,
    }

    def make(mode, factor, equation, reference, notes):
        inputs = dict(shared)
        inputs.update(limit.as_dict())
        return Margin(
            component="rod bolts", mode=mode, kind="factor",
            applied=factor, allowable=1.0, unit="-",
            equation=equation, reference=reference,
            condition=ctx.at("overlap TDC, peak tensile load"),
            inputs=inputs, notes=notes)

    return [
        make("joint separation", result.separation_factor,
             "n = F_preload / (F_external (1 - C))",
             "Shigley, preloaded bolted joint",
             ["once the cap separates the bolt takes the whole external load "
              "and the fatigue margin below no longer applies",
              "the joint stiffness factor C is an estimate; a softer joint "
              "raises the bolt's share of the load"]),
        make("thread root fatigue", result.fatigue_factor,
             "n_f = S_e (S_ut - sigma_i) / (sigma_a (S_ut + S_e))",
             "Shigley, fatigue of a preloaded bolt, Goodman criterion",
             ["rolled threads carry compressive residual stress that improves "
              "this considerably; the benefit is not credited here",
              "assumes the external load fluctuates between zero and its peak, "
              "which is what a rod bolt sees once per revolution"]),
        make("yielding at proof load", result.yield_factor,
             "n_p = (S_proof A_t - F_preload) / (C F_external)",
             "Shigley, static load factor for a preloaded bolt",
             [f"preload is "
              f"{shared['preload_fraction_of_proof'] * 100:.0f}% of proof; "
              "common practice is 70 to 85% for a reused bolt"]),
    ]
