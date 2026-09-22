"""Search the design for a stated target, without breaking anything.

The optimiser is the last thing built on purpose. It is only as trustworthy
as the model underneath it, and an optimiser is the single best tool ever
devised for finding a model's weak spots and driving straight into them. So
this module is built to be suspicious of its own answer.

Three things follow from that.

**It cannot cheat the constraints.** Every candidate is written through
``DesignState.set``, so a parameter bound -- including the block envelope's
overbore ceiling -- refuses it the same way it refuses a person. Safety
factors ride along as a nonlinear constraint, and the returned design is
re-validated from scratch rather than trusted because the search said so.

**It re-measures what it optimised.** ``masses.piston`` is a stored number,
not an expression over the geometry, so the fast analytical layer does not
see a thinner crown get lighter. An optimiser let loose on that seam will
happily buy stress margin with mass that never actually leaves the part. So
after the search the solids are rebuilt, the masses re-measured, and the
design re-evaluated. If the verified answer differs from the one the search
believed, the result says so rather than quietly reporting the optimistic
number.

**It says what stopped it.** An optimum is a statement about which
constraint is binding, and the binding constraint is more informative than
the objective value. If that constraint happens to be one of the phase 2
idealisations that are still calibrated guesses rather than FEA-fitted
values, the result flags it -- because an optimum sitting on a guess is worth
exactly as much as the guess.

The search is SLSQP with finite-difference gradients, started from the
existing design. That is deliberate rather than lazy: the starting point is a
real engine, the run is deterministic and reproducible, and a few hundred
evaluations at 36 ms each stays interactive. A stochastic global search would
give a different answer every run, which is a poor property for a tool whose
whole claim is that its numbers are traceable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .sensitivity import OBJECTIVES, rank_levers
from .state import ConstraintViolation, Mutability

__all__ = ["Move", "OptimiseResult", "optimise", "frontier",
           "Frontier", "DEFAULT_MAX_LEVERS"]

DEFAULT_MAX_LEVERS = 8
SEARCH_CAP = 0.15          # explore +/-15% when a parameter has no bound

# Parameters whose phase 2 model still rests on a calibrated constant rather
# than a fitted one. An optimum that lands on one of these is reported with a
# warning, because the answer inherits the guess.
CALIBRATED = {
    "crown": "piston.crown_support_radius_fraction",
    "pin": "pin.support_span_factor",
    "skirt": "piston.skirt_bearing_arc",
}

# Bookkeeping, not design. The analytical layer reads masses from the state
# rather than from the solids, so letting the search write them directly buys
# mass reductions that no machining would produce.
EXCLUDED_PREFIXES = ("masses.",)

# Levers grouped by WHAT YOU WOULD HAVE TO DO to realise them. The grouping
# is not cosmetic. Turned loose on everything at once, the search answers
# "how do I get more torque?" with a longer stroke and a 55% longer burn --
# which is true, and is also a different engine and a remap rather than a
# redesigned piston system. Worse, the burn duration is the single easiest
# way to exploit the edges of a single-zone Wiebe model, so the biggest
# number in the answer would be the least trustworthy part of it.
#
# The default is what you can do to THIS block with THESE parts remachined.
# Everything else is opt-in and says what it costs.
LEVER_CLASSES: dict[str, tuple] = {
    "machining": (
        # Comes off the existing block, inside the envelope's ceiling.
        "engine.bore",
    ),
    "piston": (
        "piston.crown_thickness", "piston.skirt_length",
        "piston.top_land_height", "piston.ring_groove_depth",
        "piston.total_height", "piston.compression_height",
        "piston.wall_thickness", "piston.boss_inner_span",
        "piston.boss_width_factor",
        "pin.outer_diameter", "pin.inner_diameter", "pin.length",
        "small_end.bushing_width",
        # New pistons anyway, so the chamber volume they leave is a choice.
        "engine.compression_ratio",
    ),
    "rod": (
        "rod.shank_height", "rod.shank_width", "rod.shank_web",
        "rod.shank_flange", "rod.shank_taper",
        "rod.big_end_bore", "rod.big_end_width",
        "bolts.preload", "bolts.thread_diameter",
    ),
    "rotating": (
        # A different crank and different rods: a new bottom end, not a
        # machining operation.
        "engine.stroke", "engine.rod_length",
    ),
    "tuning": (
        # Calibration, not hardware. Also where the combustion model is
        # weakest, so gains here deserve the most suspicion.
        "operating.soc_angle", "operating.burn_duration",
    ),
}

DEFAULT_CLASSES = ("machining", "piston", "rod")

CLASS_COST = {
    "machining": "machining the existing block",
    "piston": "new pistons and pins",
    "rod": "new connecting rods",
    "rotating": "a new crankshaft and rods -- a different bottom end",
    "tuning": "a calibration change, not hardware",
}


def _class_of(path: str) -> str:
    for name, paths in LEVER_CLASSES.items():
        if path in paths:
            return name
    return "other"


@dataclass
class Move:
    """One lever, and how far the search actually took it."""

    path: str
    before: float
    after: float
    minimum: float | None
    maximum: float | None
    unit: str = ""

    @property
    def delta(self) -> float:
        return self.after - self.before

    @property
    def percent(self) -> float | None:
        if not self.before:
            return None
        return self.delta / abs(self.before) * 100.0

    @property
    def at_bound(self) -> str:
        """Whether the search pushed this lever all the way to a limit."""
        span = (self.maximum or self.after) - (self.minimum or self.after)
        tolerance = max(abs(span) * 1e-6, 1e-12)
        if self.maximum is not None and self.after >= self.maximum - tolerance:
            return "maximum"
        if self.minimum is not None and self.after <= self.minimum + tolerance:
            return "minimum"
        return ""

    def as_dict(self) -> dict:
        return {"path": self.path, "before": self.before, "after": self.after,
                "delta": self.delta, "percent": self.percent,
                "unit": self.unit, "at_bound": self.at_bound}


@dataclass
class OptimiseResult:
    """A searched design, and every reason to doubt it."""

    objective: str
    direction: str
    unit: str
    before: float
    after: float
    verified: float
    state: object = None
    moves: list = field(default_factory=list)
    binding_before: str = ""
    binding_after: str = ""
    safety_before: float = 0.0
    safety_after: float = 0.0
    floor: float = 0.0
    evaluations: int = 0
    converged: bool = False
    message: str = ""
    warnings: list = field(default_factory=list)
    worse: list = field(default_factory=list)

    @property
    def gain(self) -> float:
        return self.after - self.before

    @property
    def gain_percent(self) -> float | None:
        if not self.before:
            return None
        return self.gain / abs(self.before) * 100.0

    @property
    def verification_error(self) -> float | None:
        """How far the honest answer is from the one the search believed."""
        if not self.after:
            return None
        return (self.verified - self.after) / abs(self.after) * 100.0

    @property
    def moved(self) -> list:
        return [m for m in self.moves if abs(m.percent or 0.0) > 0.01]

    def as_dict(self) -> dict:
        return {
            "objective": self.objective, "direction": self.direction,
            "unit": self.unit, "before": self.before, "after": self.after,
            "verified": self.verified, "gain": self.gain,
            "gain_percent": self.gain_percent,
            "verification_error_percent": self.verification_error,
            "moves": [m.as_dict() for m in self.moved],
            "binding_before": self.binding_before,
            "binding_after": self.binding_after,
            "safety_before": self.safety_before,
            "safety_after": self.safety_after,
            "safety_floor": self.floor,
            "evaluations": self.evaluations, "converged": self.converged,
            "message": self.message, "warnings": list(self.warnings),
            "worse": list(self.worse),
        }

    def summary(self) -> str:
        arrow = "up" if self.direction == "maximise" else "down"
        lines = [
            f"{self.objective}: {self.before:.4g} -> {self.after:.4g} "
            f"{self.unit} ({self.gain_percent:+.2f}%, wanted it {arrow})"]
        if self.verification_error is not None \
                and abs(self.verification_error) > 0.05:
            lines.append(
                f"  VERIFIED against rebuilt geometry: {self.verified:.4g} "
                f"{self.unit} ({self.verification_error:+.2f}% from what the "
                "search believed)")
        lines.append(f"  worst safety factor {self.safety_before:.3f} -> "
                     f"{self.safety_after:.3f} (floor {self.floor:.2f})")
        if self.binding_before != self.binding_after:
            lines.append(f"  binding constraint moved: {self.binding_before} "
                         f"-> {self.binding_after}")
        else:
            lines.append(f"  still binding on {self.binding_after}")
        lines.append("")
        for move in sorted(self.moved,
                           key=lambda m: -abs(m.percent or 0.0)):
            flag = f"  [at {move.at_bound}]" if move.at_bound else ""
            lines.append(f"  {move.path:38s} {move.before:12.6g} -> "
                         f"{move.after:12.6g}  {move.percent:+7.2f}%{flag}")
        for warning in self.warnings:
            lines.append(f"\n  ! {warning}")
        return "\n".join(lines)


# --------------------------------------------------------------------------

def _objective_value(metrics, path: str) -> float:
    section, key = path.split(".", 1)
    return float(getattr(metrics, section)[key])


def _candidate_levers(state, objective: str, levers, max_levers: int,
                      classes=DEFAULT_CLASSES) -> list:
    """The levers to search, best first.

    Ranking ORDERS the candidates; it does not select them. That distinction
    matters for mass objectives: ``masses.piston`` is a stored number, so a
    geometry lever like crown thickness shows no effect on mass in the fast
    layer and never gets ranked at all. Selecting by rank alone therefore
    returned an empty set and the optimiser refused to run -- when the real
    answer was that those levers work fine once the masses are measured off
    the solids.
    """
    if levers:
        chosen = list(levers)
    else:
        allowed = []
        for name in classes:
            allowed.extend(LEVER_CLASSES.get(name, ()))

        order = {}
        try:
            ranked = rank_levers(state, objective)
            order = {row["path"]: i
                     for i, row in enumerate(ranked["levers"])}
        except Exception:                                     # noqa: BLE001
            pass

        # Ranked levers first in rank order, then the rest in declared order.
        chosen = sorted(allowed,
                        key=lambda path: (order.get(path, 10_000),
                                          allowed.index(path)))[:max_levers]

    usable = []
    for path in chosen:
        if any(path.startswith(prefix) for prefix in EXCLUDED_PREFIXES):
            continue
        if not state.has(path):
            continue
        param = state.param(path)
        if param.mutability in (Mutability.LOCKED, Mutability.DERIVED):
            continue
        if not isinstance(param.value, (int, float)) \
                or isinstance(param.value, bool):
            continue
        usable.append(path)
    return usable


def _box(state, paths: list) -> tuple:
    """Search bounds per lever, in the parameter's own units."""
    lows, highs = [], []
    for path in paths:
        param = state.param(path)
        value = float(param.value)
        span = abs(value) * SEARCH_CAP or 1e-6
        low = param.minimum if param.minimum is not None else value - span
        high = param.maximum if param.maximum is not None else value + span
        # Never let the box exclude where we started.
        lows.append(min(low, value))
        highs.append(max(high, value))
    return np.array(lows), np.array(highs)


def optimise(state, objective: str = "torque", levers=None,
             max_levers: int = DEFAULT_MAX_LEVERS,
             classes=DEFAULT_CLASSES,
             min_safety_factor: float | None = None,
             max_iterations: int = 60, verify: bool = True,
             geometry_masses: bool | None = None) -> OptimiseResult:
    """Search toward ``objective`` without breaking any constraint.

    ``levers`` names the parameters to move. Left out, the top-ranked levers
    are used, drawn from ``classes`` -- by default what you can do to this
    block with remachined parts, NOT a new crank or a remap. Pass
    ``classes=("machining", "piston", "rod", "rotating", "tuning")`` to let
    it change the engine instead of the piston system.

    ``min_safety_factor`` overrides the floor. The default floor is the
    design's configured minimum OR its current worst margin, whichever is
    lower: demanding an improvement the design does not already meet turns
    "find me more torque" into "fix the safety factor first", which is a
    different question and produces a confusing answer.

    ``verify`` rebuilds the solids afterwards and re-evaluates -- leave it on
    unless you know why you are turning it off.

    ``geometry_masses`` rebuilds the solids on EVERY candidate rather than
    only at the end. It is required for any objective that depends on mass,
    because the analytical layer reads masses from the state: without it a
    thinner crown weighs exactly the same as a thick one and the search has
    nothing to follow. It costs about half a second per evaluation, so it
    defaults on only when the objective needs it.
    """
    from scipy.optimize import minimize

    from . import geometry as geo
    from .evaluate import evaluate

    if objective not in OBJECTIVES:
        raise ValueError(
            f"unknown objective {objective!r}; available: "
            + ", ".join(sorted(OBJECTIVES)))

    metric_path, direction, unit = OBJECTIVES[objective]
    sign = -1.0 if direction == "maximise" else 1.0

    if geometry_masses is None:
        geometry_masses = metric_path.startswith("masses.")

    paths = _candidate_levers(state, objective, levers, max_levers, classes)
    if not paths:
        raise ValueError(
            "no usable levers: every candidate is locked, derived, or "
            "excluded. Name the parameters explicitly with levers=[...] if "
            "you meant something the ranking did not pick.")

    start = np.array([float(state[path]) for path in paths])
    lows, highs = _box(state, paths)

    base_metrics = evaluate(state)
    before = _objective_value(base_metrics, metric_path)
    safety_before = float(
        base_metrics.structural["minimum_operating_safety_factor"])

    configured = float(state.get("constraints.min_safety_factor", 1.5))
    if min_safety_factor is not None:
        floor = min_safety_factor
    else:
        # A real engine may already sit below the tool's configured floor --
        # the LS3 does, at 1.24 against a default of 1.50. Holding the search
        # to 1.50 makes it spend the whole budget recovering margin and
        # reports a torque number that came from fixing something else.
        floor = min(configured, safety_before)
    binding_before = (
        f"{base_metrics.structural['binding_operating_component']}: "
        f"{base_metrics.structural['binding_operating_mode']}")

    # Scale the search variables so SLSQP's single step size suits parameters
    # that differ by six orders of magnitude (bolt preload in newtons, ring
    # height in metres). Without this it stalls on the small ones.
    scale = np.where(np.abs(start) > 0, np.abs(start), 1.0)

    counter = {"n": 0}
    cache: dict = {}

    def build(x):
        key = tuple(np.round(x, 12))
        if key in cache:
            return cache[key]
        changes = {path: float(value)
                   for path, value in zip(paths, x * scale)}
        try:
            candidate = state.with_changes(changes, actor="optimiser",
                                           rationale=f"search: {objective}")
            if geometry_masses:
                candidate = geo.adopt_masses(candidate, actor="optimiser")
            metrics = evaluate(candidate)
        except (ConstraintViolation, ValueError, ZeroDivisionError):
            cache[key] = None
            return None
        counter["n"] += 1
        cache[key] = (candidate, metrics)
        return cache[key]

    def cost(x):
        built = build(x)
        if built is None:
            # Refused or unbuildable: a large finite penalty, not an
            # exception. SLSQP cannot recover from a NaN.
            return 1e6
        return sign * _objective_value(built[1], metric_path) / max(
            abs(before), 1e-12)

    def safety(x):
        built = build(x)
        if built is None:
            return -1.0
        value = built[1].structural["minimum_operating_safety_factor"]
        if value is None or not np.isfinite(value):
            return -1.0
        return float(value) - floor

    # Screen the levers before spending the budget on them. Two evaluations
    # each buys two things: levers that cannot move the objective are dropped
    # so the search works in a smaller space, and a lever that does NOTHING is
    # reported -- because in this model that usually means a gap rather than a
    # genuine insensitivity. piston.skirt_length was exactly that: the
    # structural model uses it for skirt bearing area, the geometry builder
    # never reads it, so it moved no mass at all.
    inert: list[str] = []
    if len(paths) > 1:
        reference = cost(start / scale)
        keep_index = []
        for i in range(len(paths)):
            moved = False
            for step in (1.005, 0.995):
                probe = (start / scale).copy()
                probe[i] *= step
                probe = np.clip(probe, lows / scale, highs / scale)
                value = cost(probe)
                if value < 1e5 and abs(value - reference) > 1e-9:
                    moved = True
                    break
            (keep_index if moved else inert).append(
                i if moved else paths[i])
        if keep_index:
            paths = [paths[i] for i in keep_index]
            start = np.array([float(state[path]) for path in paths])
            lows, highs = _box(state, paths)
            scale = np.where(np.abs(start) > 0, np.abs(start), 1.0)
            cache.clear()

    result = minimize(
        cost, start / scale, method="SLSQP",
        bounds=list(zip(lows / scale, highs / scale)),
        constraints=[{"type": "ineq", "fun": safety}],
        options={"maxiter": max_iterations, "ftol": 1e-9, "eps": 1e-4})

    built = build(result.x)
    if built is None or safety(result.x) < -1e-9:
        # The search ended somewhere the design will not accept. Report the
        # starting point rather than an illegal one.
        best_state, best_metrics = state, base_metrics
        message = ("the search could not find a feasible design better than "
                   "the one it started from")
        converged = False
    else:
        best_state, best_metrics = built
        message = str(result.message)
        converged = bool(result.success)

    after = _objective_value(best_metrics, metric_path)
    safety_after = float(
        best_metrics.structural["minimum_operating_safety_factor"])
    binding_after = (
        f"{best_metrics.structural['binding_operating_component']}: "
        f"{best_metrics.structural['binding_operating_mode']}")

    moves = [
        Move(path, float(state[path]), float(best_state[path]),
             state.param(path).minimum, state.param(path).maximum,
             state.param(path).unit)
        for path in paths]

    warnings: list[str] = []
    verified = after

    if verify:
        verified, verify_state, notes = _verify(best_state, metric_path)
        warnings.extend(notes)
        if verify_state is not None:
            best_state = verify_state

    result_obj = OptimiseResult(
        objective=objective, direction=direction, unit=unit,
        before=before, after=after, verified=verified, state=best_state,
        moves=moves, binding_before=binding_before,
        binding_after=binding_after, safety_before=safety_before,
        safety_after=safety_after, floor=floor,
        evaluations=counter["n"], converged=converged, message=message,
        warnings=warnings)

    if inert:
        result_obj.warnings.append(
            "these levers were dropped because they move this objective not "
            "at all in this model: " + ", ".join(inert) + ". That is usually "
            "a gap between two models rather than real insensitivity -- worth "
            "checking before trusting their absence from the answer.")

    if not result_obj.moved and result_obj.safety_before <= floor + 1e-9:
        result_obj.warnings.append(
            f"nothing moved because the design already sits ON its safety "
            f"floor ({floor:.3f}) and every improving direction costs "
            "margin. That is a well-posed refusal, not a failure: decide how "
            "much margin the gain is worth and pass min_safety_factor, or "
            "relieve the binding constraint first.")

    if geometry_masses:
        result_obj.warnings.append(
            "masses were measured off rebuilt solids on every candidate, so "
            "the mass figures here are real rather than the stored ones.")

    if min_safety_factor is None and floor < configured:
        result_obj.warnings.append(
            f"the safety floor was held at this design's current worst "
            f"margin ({floor:.3f}) rather than its configured minimum "
            f"({configured:.2f}), which it does not currently meet. The "
            "search was told not to make it worse, not to make it good.")

    result_obj.worse = _regressions(base_metrics, best_metrics)
    result_obj.warnings.extend(_suspicion(best_state, moves, binding_after,
                                          result_obj))
    return result_obj


def _verify(candidate, metric_path: str):
    """Rebuild the solids, re-measure the masses, re-evaluate.

    The search runs on an analytical layer whose masses are stored numbers.
    This is the step that checks the optimum survives contact with the parts
    that would actually be made.
    """
    from . import geometry as geo
    from .evaluate import evaluate

    notes: list[str] = []
    try:
        adopted = geo.adopt_masses(candidate, actor="optimiser-verify")
    except Exception as exc:                                  # noqa: BLE001
        notes.append(
            f"could not rebuild the solids to verify this design ({exc}). "
            "The numbers above are the analytical layer's, unchecked.")
        return _objective_value(evaluate(candidate), metric_path), None, notes

    metrics = evaluate(adopted)
    return _objective_value(metrics, metric_path), adopted, notes


def _regressions(before, after) -> list:
    """Anything that got materially worse, in plain words."""
    watched = (
        ("performance.brake_torque_nm", "brake torque", +1),
        ("performance.brake_power_w", "brake power", +1),
        ("geometry.displacement_total_m3", "displacement", +1),
        ("geometry.compression_ratio", "compression ratio", +1),
        ("masses.reciprocating_kg", "reciprocating mass", -1),
        ("masses.piston_assembly_kg", "piston assembly mass", -1),
        ("loads.peak_side_thrust_n", "peak side thrust", -1),
        ("loads.peak_pin_compression_n", "peak pin force", -1),
        ("loads.peak_rod_compression_n", "peak rod force", -1),
        ("performance.mechanical_efficiency", "mechanical efficiency", +1),
        ("combustion.peak_pressure_pa", "peak cylinder pressure", -1),
    )
    out = []
    for path, label, good in watched:
        section, key = path.split(".", 1)
        old = getattr(before, section).get(key)
        new = getattr(after, section).get(key)
        if old is None or new is None or old == new or not old:
            continue
        change = (new - old) / abs(old) * 100.0
        if (change > 0) == (good > 0) or abs(change) < 0.05:
            continue
        out.append({"label": label, "before": float(old), "after": float(new),
                    "percent": change})
    return sorted(out, key=lambda row: row["percent"], reverse=True)


def _suspicion(state, moves, binding_after: str, result) -> list:
    """Reasons to distrust the answer, stated with the answer."""
    notes = []

    used = {_class_of(m.path) for m in result.moved}
    for name in sorted(used - set(DEFAULT_CLASSES)):
        notes.append(
            f"this answer needs {CLASS_COST.get(name, name)}, which is not "
            "a change to the piston system you already have.")
    if "tuning" in used:
        notes.append(
            "part of this gain comes from the combustion timing, which is "
            "a single-zone Wiebe fit, not a combustion simulation. It is "
            "the easiest thing in this model to over-exploit -- treat a "
            "large burn-duration move as a modelling artefact until a "
            "pressure trace says otherwise.")

    component = binding_after.split(":")[0].strip()
    if component in CALIBRATED:
        notes.append(
            f"the optimum sits on {binding_after}, whose model rests on "
            f"{CALIBRATED[component]} -- a CALIBRATED constant, not one "
            "fitted from FEA. This answer is worth exactly as much as that "
            "constant is.")

    pinned = [m for m in moves if m.at_bound]
    if pinned:
        names = ", ".join(f"{m.path} at its {m.at_bound}" for m in pinned[:4])
        notes.append(
            f"{len(pinned)} lever(s) finished hard against a limit ({names}). "
            "The search wanted to go further, so the answer is set by the "
            "bound rather than by the physics -- check the bound is real.")

    if result.verification_error is not None \
            and abs(result.verification_error) > 1.0:
        notes.append(
            f"rebuilding the solids moved the objective by "
            f"{result.verification_error:+.2f}%. The analytical layer's "
            "stored masses had drifted from the geometry during the search; "
            "trust the verified number.")

    if not result.converged:
        notes.append(
            "the search did not converge, so this is the best point it "
            "reached rather than an optimum.")

    return notes


# --------------------------------------------------------------------------
# the trade-off frontier
# --------------------------------------------------------------------------

@dataclass
class Frontier:
    """A set of designs, each the best possible at one level of a constraint.

    Read it as a price list. Every point says what the objective is worth at
    a given safety factor, so the question stops being "what is the optimum"
    -- which has no answer without knowing what you are willing to spend --
    and becomes "how much torque does the next 0.1 of margin cost", which
    does.
    """

    objective: str
    against: str
    unit: str
    points: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    @property
    def knee(self) -> dict | None:
        """Where the exchange rate turns worst: the last cheap point.

        Not a recommendation. It is the point past which each further unit of
        constraint costs more objective than the one before, which is a fact
        about the curve rather than a judgement about the engine.
        """
        usable = [p for p in self.points if p["feasible"]]
        if len(usable) < 3:
            return None
        best = None
        for i in range(1, len(usable) - 1):
            before = usable[i]["value"] - usable[i - 1]["value"]
            after = usable[i + 1]["value"] - usable[i]["value"]
            span = usable[i + 1]["level"] - usable[i]["level"]
            if span <= 0:
                continue
            bend = abs(before - after)
            if best is None or bend > best[0]:
                best = (bend, usable[i])
        return best[1] if best else None

    def as_dict(self) -> dict:
        return {"objective": self.objective, "against": self.against,
                "unit": self.unit, "points": list(self.points),
                "knee": self.knee, "notes": list(self.notes)}

    def summary(self) -> str:
        lines = [f"{self.objective} against {self.against}", ""]
        lines.append(f"  {self.against:>14}  {self.objective:>14}  "
                     f"{'change':>9}  binding")
        base = next((p["value"] for p in self.points if p["feasible"]), None)
        for point in self.points:
            if not point["feasible"]:
                lines.append(f"  {point['level']:14.3f}  "
                             f"{'infeasible':>14}  {'':>9}  "
                             f"{point.get('reason', '')}")
                continue
            change = ((point["value"] - base) / abs(base) * 100.0
                      if base else 0.0)
            lines.append(f"  {point['level']:14.3f}  {point['value']:14.4g}  "
                         f"{change:+8.2f}%  {point['binding']}")
        knee = self.knee
        if knee:
            lines.append("")
            lines.append(f"  the curve bends hardest at "
                         f"{self.against} = {knee['level']:.3f}")
        for note in self.notes:
            lines.append(f"\n  ! {note}")
        return "\n".join(lines)


def frontier(state, objective: str = "torque",
             against: str = "safety_factor", points: int = 6,
             low: float | None = None, high: float | None = None,
             **kwargs) -> Frontier:
    """Trace the trade between ``objective`` and ``against``.

    Uses the epsilon-constraint method: hold the second quantity at a series
    of levels and optimise the first at each. That is slower than a
    population method but every point is a real design that satisfies every
    constraint, arrived at deterministically -- which matters more here than
    speed, because a front whose points are not individually buildable is a
    picture rather than an answer.
    """
    from .evaluate import evaluate

    if against != "safety_factor":
        raise ValueError(
            f"can only trade against safety_factor for now, not {against!r}. "
            "A mass or displacement ceiling would need its own constraint in "
            "the search rather than a bound on a parameter.")

    metric_path, _, unit = OBJECTIVES[objective]
    base = evaluate(state)
    current = float(base.structural["minimum_operating_safety_factor"])

    low = current if low is None else low
    high = max(float(state.get("constraints.min_safety_factor", 1.5)),
               current * 1.4) if high is None else high
    if high <= low:
        high = low * 1.4

    levels = np.linspace(low, high, max(points, 2))
    rows, notes = [], []

    for level in levels:
        try:
            result = optimise(state, objective,
                              min_safety_factor=float(level), **kwargs)
        except Exception as exc:                              # noqa: BLE001
            rows.append({"level": float(level), "feasible": False,
                         "reason": f"{type(exc).__name__}: {exc}"})
            continue

        achieved = result.safety_after
        if achieved < level - 1e-6:
            rows.append({"level": float(level), "feasible": False,
                         "reason": "no design reaches this safety factor "
                                   "with these levers"})
            continue

        rows.append({
            "level": float(level),
            "value": float(result.after),
            "verified": float(result.verified),
            "safety": float(achieved),
            "binding": result.binding_after,
            "feasible": True,
            "moves": [m.as_dict() for m in result.moved],
        })

    if not any(row["feasible"] for row in rows):
        notes.append("no level on this range produced a feasible design")
    notes.append(
        "each point is a separate constrained search from the SAME starting "
        "design, so the front is a set of alternatives, not a path")

    return Frontier(objective=objective, against=against, unit=unit,
                    points=rows, notes=notes)
