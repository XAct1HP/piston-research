"""``evaluate(state) -> Metrics``: the pure function the whole tool rests on.

No hidden state, no side effects, deterministic. Everything else -- the AI
tool layer, the optimiser, undo, A/B comparison -- is downstream of that one
property. If evaluation is pure, a change can be proposed, its full
consequences seen, and the whole thing thrown away cleanly.

The result also carries its own caveats. A metric without its provenance is a
number waiting to be misused, so :class:`Metrics` reports warnings alongside
results: mean piston speed into race-engine territory, an energy closure error
that suggests something is wrong, components not yet modelled.

Phase 1 covers geometry, masses, the load environment and indicated
performance. Structural margins arrive in phase 2 -- until then the tool tells
you what the loads *are*, not whether the parts survive them, and says so.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

from . import combustion, kinematics as kin, loads as loads_mod
from . import materials as materials_mod
from . import thermal as thermal_mod
from .components import evaluate_all
from .margins import MarginReport
from .state import DesignState
from .units import G_0, rad_s_to_rpm

_CACHE: "OrderedDict[str, Metrics]" = OrderedDict()
_CACHE_LIMIT = 256


@dataclass
class Metrics:
    """Everything computed from a design state, plus what to distrust."""

    fingerprint: str
    geometry: dict = field(default_factory=dict)
    masses: dict = field(default_factory=dict)
    combustion: dict = field(default_factory=dict)
    loads: dict = field(default_factory=dict)
    performance: dict = field(default_factory=dict)
    checks: dict = field(default_factory=dict)
    structural: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    sweep: loads_mod.LoadSweep | None = None
    report: MarginReport | None = None
    thermal: object = None

    def to_dict(self, include_sweep: bool = False) -> dict:
        """Compact, JSON-safe form. This becomes the ``get_metrics()`` payload
        the AI layer reads in phase 5, so it stays small: scalars and short
        lists, never a megabyte of crank-angle arrays."""
        out = {
            "fingerprint": self.fingerprint,
            "geometry": self.geometry,
            "masses": self.masses,
            "combustion": self.combustion,
            "loads": self.loads,
            "performance": self.performance,
            "checks": self.checks,
            "structural": self.structural,
            "warnings": self.warnings,
            "notes": self.notes,
        }
        if include_sweep and self.sweep is not None:
            s = self.sweep
            out["sweep"] = {
                "theta_deg": np.degrees(s.theta).tolist(),
                "pressure": s.pressure.tolist(),
                "f_pin": s.f_pin.tolist(),
                "f_side": s.f_side.tolist(),
                "torque": s.torque.tolist(),
            }
        return out


def evaluate(state: DesignState, use_cache: bool = True) -> Metrics:
    """Evaluate a design. Never mutates the state it is given."""
    key = state.fingerprint()
    if use_cache and key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]

    issues = state.validate()
    if issues:
        raise ValueError("design state is not valid:\n  " + "\n  ".join(issues))

    geom = kin.CrankGeometry.from_state(state)
    trace = combustion.simulate(state)
    sweep = loads_mod.compute(state, trace)
    thermal_map = thermal_mod.compute(state, sweep)
    report = evaluate_all(state, sweep, thermal_map)

    omega = state["operating.speed"]
    rpm = rad_s_to_rpm(omega)

    geometry = {
        "bore_m": state["engine.bore"],
        "stroke_m": state["engine.stroke"],
        "rod_length_m": state["engine.rod_length"],
        "rod_ratio": state["engine.rod_ratio"],
        "bore_stroke_ratio": state["engine.bore"] / state["engine.stroke"],
        "compression_ratio": state["engine.compression_ratio"],
        "displacement_cyl_m3": state["engine.displacement_cyl"],
        "displacement_total_m3": state["engine.displacement_total"],
        "clearance_volume_m3": state["engine.clearance_volume"],
        "n_cylinders": int(state["engine.n_cylinders"]),
        "max_rod_angle_deg": math.degrees(kin.max_rod_angle(geom)),
        "mean_piston_speed_m_s": state["engine.mean_piston_speed"],
    }

    masses = {
        "piston_kg": state["masses.piston"],
        "piston_assembly_kg": state["masses.piston_assembly"],
        "rod_total_kg": state["masses.rod_total"],
        "rod_reciprocating_kg": state["masses.rod_reciprocating"],
        "rod_rotating_kg": state["masses.rod_rotating"],
        "reciprocating_kg": state["masses.reciprocating"],
        "reciprocating_source": "specified (geometry kernel arrives in phase 3)",
    }

    comb = {
        "peak_pressure_pa": trace.peak_pressure,
        "peak_pressure_angle_deg": math.degrees(trace.peak_pressure_angle),
        "max_pressure_rise_bar_per_deg":
            trace.max_pressure_rise_rate / 1.0e5 * math.pi / 180.0,
        "trapped_mass_kg": trace.trapped_mass,
        "fuel_mass_kg": trace.fuel_mass,
        "heat_released_j": trace.heat_released,
        "imep_gross_pa": sweep.imep_gross(),
        "imep_net_pa": sweep.imep_net(),
        "pmep_pa": trace.pmep(sweep.displacement),
        "source": trace.source,
    }

    peak_c = sweep.peak_pin_compression
    peak_t = sweep.peak_pin_tension
    side = sweep.peak_side_thrust
    accel = sweep.peak_acceleration
    overlap = sweep.at_overlap_tdc()

    load_summary = {
        "peak_pin_compression_n": peak_c.value,
        "peak_pin_compression_deg": peak_c.angle_deg,
        "peak_pin_tension_n": peak_t.value,
        "peak_pin_tension_deg": peak_t.angle_deg,
        "peak_rod_compression_n": sweep.peak_rod_compression.value,
        "peak_rod_tension_n": sweep.peak_rod_tension.value,
        "peak_side_thrust_n": side.value,
        "peak_side_thrust_deg": side.angle_deg,
        "peak_acceleration_m_s2": accel.value,
        "peak_acceleration_g": accel.value / G_0,
        "overlap_tdc_rod_tension_n": overlap["f_rod"],
        "gas_force_at_peak_pressure_n": sweep.at_peak_pressure()["f_gas"],
    }

    performance = {
        "speed_rpm": rpm,
        "indicated_torque_nm": sweep.indicated_torque(),
        "indicated_power_w": sweep.indicated_power(),
        "indicated_torque_per_litre_nm_l":
            sweep.indicated_torque() / (state["engine.displacement_total"] * 1e3),
        "fmep_pa": sweep.fmep(),
        "bmep_pa": sweep.bmep(),
        "brake_torque_nm": sweep.brake_torque(),
        "brake_power_w": sweep.brake_power(),
        "brake_torque_per_litre_nm_l":
            sweep.brake_torque() / (state["engine.displacement_total"] * 1e3),
        "mechanical_efficiency": sweep.mechanical_efficiency(),
        "note": ("brake figures subtract an empirical Chen-Flynn friction "
                 "correlation, not a friction model; treat them as an "
                 "estimate calibrated by the fmep_* coefficients"),
    }

    work_torque = sweep.cycle_work()
    work_imep = sweep.imep_net() * sweep.displacement
    closure_error = (abs(work_torque - work_imep) / abs(work_imep)
                     if work_imep else float("nan"))

    checks = {
        "energy_closure_error": closure_error,
        "energy_closure_pass": bool(closure_error < 1.0e-3),
        "cycle_work_from_torque_j": work_torque,
        "cycle_work_from_imep_j": work_imep,
    }

    warnings: list[str] = []
    notes: list[str] = []

    eta_mech = performance["mechanical_efficiency"]
    if not 0.55 < eta_mech < 0.97:
        warnings.append(
            f"mechanical efficiency {eta_mech:.2f} is outside the plausible "
            "0.55 to 0.97 band; check the fmep_* coefficients")

    mps = geometry["mean_piston_speed_m_s"]
    if mps > 25.0:
        warnings.append(
            f"mean piston speed {mps:.1f} m/s is race-engine territory; "
            "production engines rarely exceed 20 m/s at redline")
    elif mps > 20.0:
        notes.append(f"mean piston speed {mps:.1f} m/s is at the high end of "
                     "production practice")

    rr = geometry["rod_ratio"]
    if rr < 3.0:
        warnings.append(
            f"rod ratio {rr:.2f} is short: expect high side thrust, a strong "
            "secondary inertia term and elevated skirt loading")
    elif rr > 4.2:
        notes.append(f"rod ratio {rr:.2f} is long; side thrust is low but the "
                     "rod is more slender for a given weight")

    prr = comb["max_pressure_rise_bar_per_deg"]
    if prr > 10.0:
        warnings.append(
            f"peak pressure rise rate {prr:.1f} bar/deg is high; in a real "
            "engine this would sound rough and sits near the knock boundary")

    if not checks["energy_closure_pass"]:
        warnings.append(
            f"energy closure error {closure_error * 100:.3f}% exceeds the 0.1% "
            "tolerance: the force resolution and the pressure trace disagree")

    cr_param = state.param("engine.compression_ratio")
    if cr_param.maximum is not None:
        if state["engine.compression_ratio"] >= cr_param.maximum - 1e-9:
            notes.append("compression ratio is at its upper bound: "
                         + (cr_param.why_max or "bound reached"))

    bore_param = state.param("engine.bore")
    if bore_param.maximum is not None:
        headroom = bore_param.maximum - state["engine.bore"]
        notes.append(
            f"bore has {headroom * 1000:.2f} mm of headroom before its upper "
            f"bound ({bore_param.why_max or 'no reason recorded'})")

    for role in ("piston", "pin", "rod", "sleeve"):
        key_m = state.get(f"materials.{role}", None)
        if key_m:
            try:
                materials_mod.get(key_m)
            except KeyError as exc:
                warnings.append(str(exc))

    if state.meta.get("provenance"):
        notes.append("input provenance: " + state.meta["provenance"])

    by_component: dict = {}
    for margin in report.margins:
        current = by_component.get(margin.component)
        factor = margin.safety_factor
        if current is None or factor < current:
            by_component[margin.component] = factor

    binding = report.binding
    operating_binding = report.binding_operating
    structural = {
        "minimum_safety_factor": (None if math.isinf(report.minimum)
                                  else report.minimum),
        "binding_component": binding.component if binding else None,
        "binding_mode": binding.mode if binding else None,
        "binding_operating_component":
            operating_binding.component if operating_binding else None,
        "binding_operating_mode":
            operating_binding.mode if operating_binding else None,
        "minimum_operating_safety_factor":
            (operating_binding.safety_factor if operating_binding
             and math.isfinite(operating_binding.safety_factor) else None),
        "failing": [f"{m.component}: {m.mode}" for m in report.failing],
        "by_component": {k: (None if math.isinf(v) else v)
                         for k, v in by_component.items()},
        "margin_count": len(report.margins),
        "thermal": thermal_map.as_dict(),
    }

    required = state["constraints.min_safety_factor"]
    for margin in report.failing:
        warnings.append(
            f"FAILS: {margin.component} {margin.mode} at safety factor "
            f"{margin.safety_factor:.2f}")
    if binding and not math.isinf(report.minimum) and report.minimum < required:
        if not report.failing:
            warnings.append(
                f"minimum safety factor {report.minimum:.2f} is below the "
                f"required {required:.2f}: {binding.component}, {binding.mode}")

    notes.append(
        "structural margins use closed-form models; novel geometry falls "
        "outside what they capture, and phase 6 FEA is what resolves that")

    metrics = Metrics(
        fingerprint=key, geometry=geometry, masses=masses, combustion=comb,
        loads=load_summary, performance=performance, checks=checks,
        structural=structural, warnings=warnings, notes=notes, sweep=sweep,
        report=report, thermal=thermal_map)

    if use_cache:
        _CACHE[key] = metrics
        while len(_CACHE) > _CACHE_LIMIT:
            _CACHE.popitem(last=False)
    return metrics


def torque_curve(state: DesignState, rpm_values) -> dict:
    """Evaluate across engine speed. Returns arrays plus the headline peaks.

    Each point is a full evaluation at that speed, so the inertia relief on
    the gas load and the rising tensile load at overlap TDC are both captured
    properly rather than scaled from one operating point.
    """
    rpms, torque, power, pin_c, pin_t, peak_p = [], [], [], [], [], []
    b_torque, b_power, min_sf, binding = [], [], [], []
    for rpm in rpm_values:
        probe = state.copy()
        probe.set("operating.speed", rpm * 2.0 * math.pi / 60.0,
                  actor="sweep", rationale=f"torque curve point at {rpm:.0f} rpm")
        m = evaluate(probe)
        rpms.append(float(rpm))
        torque.append(m.performance["indicated_torque_nm"])
        power.append(m.performance["indicated_power_w"])
        b_torque.append(m.performance["brake_torque_nm"])
        b_power.append(m.performance["brake_power_w"])
        pin_c.append(m.loads["peak_pin_compression_n"])
        pin_t.append(m.loads["peak_pin_tension_n"])
        peak_p.append(m.combustion["peak_pressure_pa"])
        min_sf.append(m.structural["minimum_safety_factor"])
        binding.append(m.structural["binding_component"])

    i_t = int(np.argmax(torque))
    i_p = int(np.argmax(power))
    i_bt = int(np.argmax(b_torque))
    i_bp = int(np.argmax(b_power))
    return {
        "rpm": rpms,
        "indicated_torque_nm": torque,
        "indicated_power_w": power,
        "brake_torque_nm": b_torque,
        "brake_power_w": b_power,
        "peak_brake_torque_nm": b_torque[i_bt],
        "peak_brake_torque_rpm": rpms[i_bt],
        "peak_brake_power_w": b_power[i_bp],
        "peak_brake_power_rpm": rpms[i_bp],
        "peak_pin_compression_n": pin_c,
        "peak_pin_tension_n": pin_t,
        "peak_pressure_pa": peak_p,
        "minimum_safety_factor": min_sf,
        "binding_component": binding,
        "peak_torque_nm": torque[i_t],
        "peak_torque_rpm": rpms[i_t],
        "peak_power_w": power[i_p],
        "peak_power_rpm": rpms[i_p],
    }


def compare(before: Metrics, after: Metrics) -> dict:
    """Deltas between two evaluations.

    Reports what got worse as well as what got better -- a tool that only
    surfaces the improvement teaches you to trust it exactly when you should
    not.
    """
    out: dict[str, dict] = {}
    for section in ("geometry", "masses", "combustion", "loads", "performance"):
        a, b = getattr(before, section), getattr(after, section)
        for k in a:
            if not isinstance(a[k], (int, float)) or isinstance(a[k], bool):
                continue
            if k not in b or not isinstance(b[k], (int, float)):
                continue
            old, new = float(a[k]), float(b[k])
            if old == new:
                continue
            out[f"{section}.{k}"] = {
                "before": old, "after": new, "delta": new - old,
                "percent": ((new - old) / abs(old) * 100.0) if old else None,
            }
    return out
