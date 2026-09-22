"""What else moves when the bore moves.

Enlarging a bore is never a single-parameter change, and a tool that lets you
type a bigger number and answers "more torque" is lying to you by omission.
The displacement goes up, but so does the piston, and the compression ratio
shifts because the swept volume grew while the chamber did not. The heavier
piston raises the inertia load on the pin and the rod and pushes harder on the
thinner wall it now runs in. That wall has less material behind it, distorts
more under head bolt clamping, and has to reject more heat through less metal.

So this module answers a bore change with everything that followed from it,
sorted by how much each thing moved, and with the things that got WORSE
listed as prominently as the things that got better. It also re-checks every
structural margin, because the number that matters is not how much torque you
gained but which component you just moved closer to its limit.

The bore is enforced elsewhere -- :func:`psrt.schema.refresh_bounds` puts the
envelope ceiling on the parameter itself, so a bore past the block's limit is
refused before it ever gets here. This module is for bores that ARE allowed,
and its job is to make sure "allowed" is not mistaken for "free".
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["Consequence", "Cascade", "overbore_cascade"]


# What to watch, and which direction is bad. The wording is what the user
# reads, so each one says what the number means rather than naming a field.
WATCHED: tuple[tuple[str, str, int, str], ...] = (
    ("geometry.displacement_total_m3", "displacement", +1, "m^3"),
    ("geometry.compression_ratio", "compression ratio", +1, ":1"),
    ("geometry.bore_stroke_ratio", "bore/stroke ratio", 0, "-"),
    ("performance.brake_torque_nm", "brake torque", +1, "N m"),
    ("performance.brake_power_w", "brake power", +1, "W"),
    ("performance.bmep_pa", "BMEP", +1, "Pa"),
    ("performance.brake_torque_per_litre_nm_l", "torque per litre", +1,
     "N m/L"),
    ("performance.mechanical_efficiency", "mechanical efficiency", +1, "-"),
    ("masses.piston_kg", "piston mass", -1, "kg"),
    ("masses.piston_assembly_kg", "piston assembly mass", -1, "kg"),
    ("masses.reciprocating_kg", "reciprocating mass", -1, "kg"),
    ("combustion.peak_pressure_pa", "peak cylinder pressure", 0, "Pa"),
    ("combustion.trapped_mass_kg", "trapped mass", +1, "kg"),
    ("loads.peak_pin_compression_n", "peak pin force", -1, "N"),
    ("loads.peak_rod_compression_n", "peak rod force", -1, "N"),
    ("loads.peak_side_thrust_n", "peak side thrust", -1, "N"),
    ("loads.peak_acceleration_g", "peak piston acceleration", -1, "g"),
    ("loads.gas_force_at_peak_pressure_n", "gas force on the crown", 0, "N"),
)


@dataclass
class Consequence:
    """One thing that moved, and whether moving was good or bad."""

    key: str
    label: str
    before: float
    after: float
    unit: str = ""
    good_direction: int = 0        # +1 up is good, -1 down is good, 0 neutral

    @property
    def delta(self) -> float:
        return self.after - self.before

    @property
    def percent(self) -> float | None:
        if not self.before:
            return None
        return self.delta / abs(self.before) * 100.0

    @property
    def verdict(self) -> str:
        if not self.delta or not self.good_direction:
            return "neutral"
        improving = (self.delta > 0) == (self.good_direction > 0)
        return "better" if improving else "worse"

    def as_dict(self) -> dict:
        return {"key": self.key, "label": self.label, "before": self.before,
                "after": self.after, "delta": self.delta,
                "percent": self.percent, "unit": self.unit,
                "verdict": self.verdict}


@dataclass
class Cascade:
    """Everything that followed from one bore change."""

    before_bore: float
    after_bore: float
    consequences: list
    margins_before: dict = field(default_factory=dict)
    margins_after: dict = field(default_factory=dict)
    binding_before: str = ""
    binding_after: str = ""
    headroom_after: float | None = None
    notes: list = field(default_factory=list)

    @property
    def overbore(self) -> float:
        return self.after_bore - self.before_bore

    @property
    def worse(self) -> list:
        return [c for c in self.consequences if c.verdict == "worse"]

    @property
    def better(self) -> list:
        return [c for c in self.consequences if c.verdict == "better"]

    @property
    def margin_losses(self) -> list:
        """Components whose safety factor fell, worst first."""
        out = []
        for name, before in self.margins_before.items():
            after = self.margins_after.get(name)
            if after is None or after >= before:
                continue
            out.append({"component": name, "before": before, "after": after,
                        "delta": after - before,
                        "percent": (after - before) / abs(before) * 100.0
                                   if before else None})
        return sorted(out, key=lambda row: row["delta"])

    @property
    def newly_failing(self) -> list:
        return [name for name, after in self.margins_after.items()
                if after < 1.0 and self.margins_before.get(name, 0.0) >= 1.0]

    def as_dict(self) -> dict:
        return {
            "before_bore_m": self.before_bore,
            "after_bore_m": self.after_bore,
            "overbore_m": self.overbore,
            "consequences": [c.as_dict() for c in self.consequences],
            "worse": [c.as_dict() for c in self.worse],
            "better": [c.as_dict() for c in self.better],
            "margin_losses": self.margin_losses,
            "newly_failing": self.newly_failing,
            "binding_before": self.binding_before,
            "binding_after": self.binding_after,
            "headroom_after_m": self.headroom_after,
            "notes": list(self.notes),
        }

    def summary(self) -> str:
        lines = [f"bore {self.before_bore * 1000:.3f} -> "
                 f"{self.after_bore * 1000:.3f} mm "
                 f"({self.overbore * 1000:+.3f} mm)"]
        for c in sorted(self.consequences,
                        key=lambda c: -abs(c.percent or 0.0)):
            if c.percent is None or abs(c.percent) < 0.005:
                continue
            mark = {"better": "+", "worse": "!", "neutral": " "}[c.verdict]
            lines.append(f"  {mark} {c.label:28s} {c.percent:+7.2f}%")
        if self.binding_before != self.binding_after:
            lines.append(f"  ! binding constraint moved: "
                         f"{self.binding_before} -> {self.binding_after}")
        for row in self.margin_losses[:3]:
            lines.append(f"  ! {row['component']:28s} safety factor "
                         f"{row['before']:.2f} -> {row['after']:.2f}")
        for note in self.notes:
            lines.append(f"  * {note}")
        return "\n".join(lines)


def _flatten(metrics) -> dict:
    out = {}
    for section in ("geometry", "masses", "combustion", "loads",
                    "performance"):
        for key, value in getattr(metrics, section).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            out[f"{section}.{key}"] = float(value)
    return out


def _margins(metrics) -> tuple[dict, str]:
    """Safety factor per component and mode, and the binding one's name.

    Taken off the Metrics the evaluation already produced rather than
    recomputed, so the margins and the performance figures being compared
    describe the same evaluation.
    """
    rep = getattr(metrics, "report", None)
    if rep is None:
        return {}, ""
    factors = {}
    for margin in rep.margins:
        name = f"{margin.component}: {margin.mode}"
        factor = margin.safety_factor
        if factor is None:
            continue
        # Keep the worst mode per component rather than the last one seen.
        if name not in factors or factor < factors[name]:
            factors[name] = float(factor)
    binding = rep.binding
    return factors, (f"{binding.component}: {binding.mode}" if binding else "")


def overbore_cascade(state, new_bore: float, remeasure: bool = True) -> Cascade:
    """Everything that follows from taking this design's bore to ``new_bore``.

    Raises :class:`psrt.state.ConstraintViolation` if the block will not
    allow it -- the envelope bound is on the parameter, so this cannot be
    used to sneak past it.

    ``remeasure`` rebuilds the solids and re-measures the masses on the
    enlarged bore. It is on by default and it matters more than it looks:
    ``masses.piston`` is a stored number, not an expression, so without this
    a bigger bore reports more torque and an UNCHANGED piston -- hiding the
    inertia penalty that is half the reason overboring is a trade at all.
    Both states are measured the same way, so the comparison stays fair.
    It costs a geometry rebuild; pass False in a hot path and the note says
    the masses are stale.
    """
    from .envelope import envelope as compute_envelope
    from .evaluate import evaluate

    before_bore = state["engine.bore"]

    after_state = state.with_changes(
        {"engine.bore": new_bore}, actor="cascade",
        rationale=f"overbore to {new_bore * 1000:.3f} mm")

    stale_masses = False
    if remeasure:
        from . import geometry as geo
        try:
            # Measure BOTH sides the same way. Adopting on the after-state
            # alone would compare a measured piston against a typed one and
            # attribute the difference to the bore.
            state = geo.adopt_masses(state, actor="cascade")
            after_state = geo.adopt_masses(after_state, actor="cascade")
        except Exception:                               # noqa: BLE001
            stale_masses = True
    else:
        stale_masses = True

    before_metrics = evaluate(state)
    before_margins, before_binding = _margins(before_metrics)
    after_metrics = evaluate(after_state)
    after_margins, after_binding = _margins(after_metrics)

    old, new = _flatten(before_metrics), _flatten(after_metrics)
    consequences = []
    for key, label, direction, unit in WATCHED:
        if key not in old or key not in new:
            continue
        if old[key] == new[key]:
            continue
        consequences.append(Consequence(key, label, old[key], new[key],
                                        unit, direction))

    env = compute_envelope(after_state)
    notes = []

    if stale_masses:
        notes.append(
            "masses were NOT re-measured, so the piston mass shown is the "
            "one stored for the old bore. The inertia penalty of this "
            "overbore is missing from every load below.")

    # The wall is the thing the user cannot see and the thing that kills
    # blocks, so it gets said in millimetres rather than left as a margin.
    if state.has("block.bore_spacing"):
        spacing = state["block.bore_spacing"]
        wall_before = (spacing - before_bore) / 2.0
        wall_after = (spacing - new_bore) / 2.0
        notes.append(
            f"bore-to-bore wall {wall_before * 1000:.2f} -> "
            f"{wall_after * 1000:.2f} mm, and every bit of that comes off "
            "the material carrying head bolt clamping load")

    if env.headroom is not None:
        notes.append(
            f"{env.headroom * 1000:.3f} mm of overbore left after this, "
            f"against the {env.binding.name} limit"
            if env.binding else "")

    notes.append(
        "heat rejection per unit wall thickness rises with bore area over "
        "wall: this model has no transient thermal FEA, so that is a "
        "direction, not a number")
    notes.append(
        "ring size and tension have to be respecified for a new bore; the "
        "ring model here does not do that for you")

    return Cascade(
        before_bore=before_bore, after_bore=new_bore,
        consequences=consequences,
        margins_before=before_margins, margins_after=after_margins,
        binding_before=before_binding, binding_after=after_binding,
        headroom_after=env.headroom,
        notes=[n for n in notes if n])
