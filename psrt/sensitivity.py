"""Ranked levers: what you can actually play with, and how far.

Looking at a torque curve and asking "what are my options" is a sensitivity
analysis, and it deserves to be a first-class feature rather than something
the AI improvises. Perturb every unlocked parameter, measure the change in the
target metric, then -- and this is the part that makes it useful -- work out
how far that parameter can actually move before something stops it.

Three things can stop it, and the answer says which:

* a **parameter bound** -- the bore runs into the coolant jacket
* a **safety factor** -- the crown goes below its fatigue limit first
* the **search cap** -- nothing stopped it inside the range explored, so the
  reported gain is a floor rather than a ceiling

A lever with a steep gradient and no headroom is worth less than a shallow one
with plenty, which is exactly what a bare derivative hides. The output is
sorted by what each lever can actually deliver, not by how sharply it responds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .evaluate import evaluate
from .state import ConstraintViolation, DesignState, Mutability

# Shorthand for the objectives people actually ask about.
OBJECTIVES = {
    "torque": ("performance.brake_torque_nm", "maximise", "N.m"),
    "indicated_torque": ("performance.indicated_torque_nm", "maximise", "N.m"),
    "power": ("performance.brake_power_w", "maximise", "W"),
    "peak_pressure": ("combustion.peak_pressure_pa", "maximise", "Pa"),
    "reciprocating_mass": ("masses.reciprocating_kg", "minimise", "kg"),
    "piston_mass": ("masses.piston_assembly_kg", "minimise", "kg"),
    "safety_factor": ("structural.minimum_operating_safety_factor",
                      "maximise", "-"),
    "rod_tension": ("loads.peak_rod_tension_n", "minimise", "N"),
    "side_thrust": ("loads.peak_side_thrust_n", "minimise", "N"),
    "displacement": ("geometry.displacement_total_m3", "maximise", "m^3"),
}

# The parameters that are genuinely design levers. Ranking all 145 would bury
# the answer in operating conditions and provenance; these are the ones a
# person can actually change about the part.
DESIGN_LEVERS = (
    "engine.bore", "engine.stroke", "engine.rod_length",
    "engine.compression_ratio",
    "piston.crown_thickness", "piston.skirt_length", "piston.top_land_height",
    "piston.ring_groove_depth", "piston.total_height",
    "piston.compression_height", "piston.wall_thickness",
    "piston.boss_inner_span", "piston.boss_width_factor",
    "pin.outer_diameter", "pin.inner_diameter", "pin.length",
    "small_end.bushing_width",
    "rod.shank_height", "rod.shank_width", "rod.shank_web",
    "rod.shank_flange", "rod.shank_taper",
    "rod.big_end_bore", "rod.big_end_width",
    "bolts.preload", "bolts.thread_diameter",
    "operating.soc_angle", "operating.burn_duration",
    # Masses are levers in their own right here, because the analytical layer
    # reads them from the design state rather than from the solids. Changing
    # crown thickness does NOT move them until the geometry is adopted -- see
    # the note rank_levers attaches for mass objectives.
    "masses.piston", "masses.rings", "masses.pin", "masses.rod_total",
    "masses.rod_cg_from_small_end",
)

SEARCH_CAP = 0.15          # explore +/-15% when a parameter has no bound
BISECTION_STEPS = 7


@dataclass
class Lever:
    """One parameter, and what moving it is actually worth."""

    path: str
    description: str
    unit: str
    value: float
    direction: int                       # +1 raise it, -1 lower it
    gradient: float                      # d(objective)/d(parameter), SI
    elasticity: float                    # % objective per % parameter
    reachable_value: float
    reachable_delta: float               # objective change at that value
    reachable_percent: float
    limited_by: str                      # "parameter bound" | "safety factor"
                                         # | "search cap"
    limit_reason: str
    side_effects: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "path": self.path, "description": self.description,
            "unit": self.unit, "current_value": self.value,
            "direction": "increase" if self.direction > 0 else "decrease",
            "gradient_per_si_unit": _safe(self.gradient),
            "percent_per_percent": _safe(self.elasticity),
            "reachable_value": _safe(self.reachable_value),
            "reachable_gain": _safe(self.reachable_delta),
            "reachable_gain_percent": _safe(self.reachable_percent),
            "limited_by": self.limited_by,
            "limit_reason": self.limit_reason,
            "side_effects": self.side_effects,
            "notes": self.notes,
        }


def _safe(v):
    return v if isinstance(v, (int, float)) and math.isfinite(v) else None


def _dig(metrics: dict, path: str):
    node = metrics
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def resolve_objective(objective: str) -> tuple:
    """Accept either a shorthand ('torque') or a full metric path."""
    if objective in OBJECTIVES:
        return OBJECTIVES[objective]
    return (objective, "maximise", "")


def measure(state: DesignState, objective_path: str):
    return _dig(evaluate(state).to_dict(), objective_path)


def _try(state: DesignState, path: str, value: float):
    """Evaluate a candidate. Returns None if the design state refuses it."""
    try:
        candidate = state.with_changes({path: value}, actor="sensitivity",
                                       rationale="probing a lever")
    except ConstraintViolation:
        return None
    from .schema import refresh_bounds
    refresh_bounds(candidate)
    try:
        return evaluate(candidate)
    except (ValueError, ZeroDivisionError):
        return None


def rank_levers(state: DesignState, objective: str = "torque",
                paths=None, find_limits: bool = True,
                min_safety_factor: float | None = None) -> dict:
    """Rank the levers that move an objective, by what they can deliver."""
    objective_path, sense, unit = resolve_objective(objective)
    want = 1.0 if sense == "maximise" else -1.0

    base_metrics = evaluate(state)
    base = _dig(base_metrics.to_dict(), objective_path)
    if base is None:
        raise ValueError(
            f"no metric at {objective_path!r}; try one of: "
            + ", ".join(sorted(OBJECTIVES)))

    floor = (min_safety_factor if min_safety_factor is not None
             else max(1.0, 0.0))
    candidates = [p for p in (paths or DESIGN_LEVERS) if state.has(p)]

    levers = []
    for path in candidates:
        param = state.param(path)
        if param.mutability in (Mutability.LOCKED, Mutability.DERIVED):
            continue
        value = state.get(path)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue

        # The probe step has to fit inside the parameter's own bounds. A flat
        # half-percent step is larger than the LS3 bore's entire remaining
        # headroom, so both probes get refused and the most important lever
        # in the engine silently disappears from the ranking.
        base_step = abs(value) * 0.005 or 1e-6
        up_room = ((param.maximum - value) if param.maximum is not None
                   else base_step)
        down_room = ((value - param.minimum) if param.minimum is not None
                     else base_step)

        probe, step = None, 0.0
        for room, sign in ((up_room, 1.0), (down_room, -1.0)):
            if room <= 0:
                continue
            trial = sign * min(base_step, room * 0.4)
            if trial == 0.0:
                continue
            probe = _try(state, path, value + trial)
            if probe is not None:
                step = trial
                break
        if probe is None or step == 0.0:
            continue

        probed = _dig(probe.to_dict(), objective_path)
        if probed is None:
            continue
        gradient = (probed - base) / step
        if gradient == 0.0 or not math.isfinite(gradient):
            continue

        direction = 1 if (gradient * want) > 0 else -1
        elasticity = (gradient * value / base * 100.0 / 100.0
                      if base else float("nan"))

        lever = Lever(
            path=path, description=param.description, unit=param.unit,
            value=value, direction=direction, gradient=gradient,
            elasticity=elasticity, reachable_value=value,
            reachable_delta=0.0, reachable_percent=0.0,
            limited_by="not explored", limit_reason="")
        if find_limits:
            _find_limit(state, lever, objective_path, base, floor)
        levers.append(lever)

    levers.sort(key=lambda l: -abs(l.reachable_percent or 0.0))

    extra = []
    if "mass" in objective_path or any(
            l.path.startswith("masses.") for l in levers[:3]):
        extra.append(
            "mass levers act directly on the design state, because the fast "
            "analytical layer reads masses from it rather than rebuilding the "
            "solids. A geometry change -- a thinner crown, a bored-out pin -- "
            "does not move these until you run the geometry check and adopt "
            "its masses")

    return {
        "objective": objective,
        "objective_path": objective_path,
        "sense": sense,
        "unit": unit,
        "baseline": base,
        "min_safety_factor_required": floor,
        "levers": [l.to_dict() for l in levers],
        "notes": [
            "each lever is reported at the furthest value it can actually "
            "reach, not at a unit step, so a steep lever with no headroom "
            "ranks below a shallow one with plenty",
            "limits are found by bisection at "
            f"{BISECTION_STEPS} steps, so a reported edge is approximate",
            "a lever being unlocked does not make it free: changing stroke "
            "means a different crankshaft, changing rod length means "
            "different rods. Lock what you are not prepared to replace and "
            "run this again",
        ] + extra,
    }


def _find_limit(state, lever: Lever, objective_path: str, base: float,
                floor: float) -> None:
    """How far this lever can go before a bound or a margin stops it."""
    param = state.param(lever.path)
    value = lever.value

    if lever.direction > 0:
        hard = param.maximum
        capped = value * (1.0 + SEARCH_CAP) if value > 0 else value + abs(value or 1.0) * SEARCH_CAP
        reason = param.why_max
    else:
        hard = param.minimum
        capped = value * (1.0 - SEARCH_CAP) if value > 0 else value - abs(value or 1.0) * SEARCH_CAP
        reason = param.why_min

    if hard is not None and ((lever.direction > 0 and hard < capped)
                             or (lever.direction < 0 and hard > capped)):
        target, limited_by = hard, "parameter bound"
        limit_reason = reason or "declared bound on this parameter"
    else:
        target, limited_by = capped, "search cap"
        limit_reason = (f"nothing stopped it within {SEARCH_CAP * 100:.0f}% of "
                        "its current value, so this gain is a floor")

    # Walk out to the target; if a margin fails on the way, bisect back.
    best_value, best_metrics = value, None
    lo, hi = value, target
    probe = _try(state, lever.path, hi)
    if probe is not None and _feasible(probe, floor):
        best_value, best_metrics = hi, probe
    else:
        limited_by = "safety factor"
        blocker = _blocker(probe, floor) if probe is not None else None
        limit_reason = (f"{blocker} falls below {floor:.2f} first"
                        if blocker else
                        "the design becomes invalid before the bound")
        for _ in range(BISECTION_STEPS):
            mid = 0.5 * (lo + hi)
            trial = _try(state, lever.path, mid)
            if trial is not None and _feasible(trial, floor):
                lo, best_value, best_metrics = mid, mid, trial
            else:
                hi = mid

    if best_metrics is None:
        lever.limited_by = limited_by
        lever.limit_reason = limit_reason or "no feasible movement found"
        lever.notes.append("this lever cannot move at all without breaking "
                           "something")
        return

    reached = _dig(best_metrics.to_dict(), objective_path)
    lever.reachable_value = best_value
    lever.reachable_delta = (reached - base) if reached is not None else 0.0
    lever.reachable_percent = (lever.reachable_delta / abs(base) * 100.0
                               if base else float("nan"))
    lever.limited_by = limited_by
    lever.limit_reason = limit_reason
    lever.side_effects = _side_effects(evaluate(state), best_metrics)


def _feasible(metrics, floor: float) -> bool:
    worst = metrics.structural.get("minimum_safety_factor")
    return worst is None or worst >= floor


def _blocker(metrics, floor: float):
    report = metrics.report
    if report is None:
        return None
    worst = report.binding
    return f"{worst.component} {worst.mode}" if worst else None


def _side_effects(before, after) -> list:
    """What else moved, so a gain never reads as free."""
    out = []
    watch = (("loads.peak_side_thrust_n", "side thrust", 1),
             ("loads.peak_pin_compression_n", "peak pin load", 1),
             ("loads.peak_rod_tension_n", "rod tension", -1),
             ("masses.reciprocating_kg", "reciprocating mass", 1),
             ("combustion.peak_pressure_pa", "peak cylinder pressure", 1))
    b, a = before.to_dict(), after.to_dict()
    for path, label, worse_when_rising in watch:
        old, new = _dig(b, path), _dig(a, path)
        if not isinstance(old, (int, float)) or not old or old == new:
            continue
        pct = (new - old) / abs(old) * 100.0
        if abs(pct) < 0.5:
            continue
        rose = pct > 0
        out.append({
            "metric": label, "percent": pct,
            "worse": rose if worse_when_rising > 0 else not rose,
        })

    sf_before = before.structural.get("minimum_operating_safety_factor")
    sf_after = after.structural.get("minimum_operating_safety_factor")
    if sf_before and sf_after and abs(sf_after - sf_before) > 0.01:
        out.append({
            "metric": "minimum safety factor",
            "from": sf_before, "to": sf_after,
            "worse": sf_after < sf_before,
            "binding": after.structural.get("binding_operating_mode"),
        })
    return out
