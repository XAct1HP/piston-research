"""Margins: what each component can take, what it is being asked to take.

A margin is not just a number. It carries the equation that produced it, where
that equation comes from, the operating condition it was evaluated at, and
every input that went into it. Three reasons that matters:

1. You can check the work. A safety factor you cannot trace is a rumour.
2. The tool can teach. ``explain_margin("rod")`` in phase 5 reads these fields
   and has something real to say.
3. When phase 6 fits notch factors from FEA, the inputs dict shows exactly
   which assumption moved.

A margin's ``kind`` decides how the safety factor is formed. Stress and
pressure margins divide allowable by applied. Deflection margins divide the
limit by the deflection. Life margins divide achieved cycles by required
cycles -- so 'safety factor 2.4' on a life margin means twice the design life,
not twice the strength.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import fatigue as fatigue_mod
from . import materials as materials_mod
from .materials import Material
from .thermal import ThermalMap


@dataclass
class Margin:
    component: str
    mode: str                  # what fails, in plain words
    applied: float
    allowable: float
    unit: str
    equation: str
    reference: str
    condition: str             # the operating point that governs
    inputs: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    kind: str = "stress"       # stress | pressure | deflection | life | factor
    life_cycles: float | None = None
    life_hours: float | None = None
    operating_dependent: bool = True
    """False for a purely geometric check that does not move with speed or
    load. Those still matter -- a bore wall too thin is a bore wall too thin --
    but they mask the speed-dependent behaviour when you are asking what
    limits the engine, so they are reported separately."""

    @property
    def safety_factor(self) -> float:
        if self.kind == "factor":
            return self.applied
        if self.applied == 0.0:
            return math.inf
        return self.allowable / self.applied

    @property
    def passes(self) -> bool:
        return self.safety_factor >= 1.0

    def to_dict(self) -> dict:
        sf = self.safety_factor
        return {
            "component": self.component,
            "mode": self.mode,
            "kind": self.kind,
            "applied": self.applied,
            "allowable": self.allowable,
            "unit": self.unit,
            "safety_factor": None if math.isinf(sf) else sf,
            "passes": self.passes,
            "equation": self.equation,
            "reference": self.reference,
            "condition": self.condition,
            "operating_dependent": self.operating_dependent,
            "life_cycles": (None if self.life_cycles is None
                            or math.isinf(self.life_cycles)
                            else self.life_cycles),
            "life_hours": (None if self.life_hours is None
                           or math.isinf(self.life_hours)
                           else self.life_hours),
            "inputs": self.inputs,
            "notes": self.notes,
        }


@dataclass
class MarginReport:
    margins: list = field(default_factory=list)
    thermal: ThermalMap | None = None

    def sorted(self) -> list:
        return sorted(self.margins, key=lambda m: m.safety_factor)

    @property
    def minimum(self) -> float:
        return min((m.safety_factor for m in self.margins), default=math.inf)

    @property
    def binding(self) -> "Margin | None":
        """The margin closest to failing. This is the answer to 'what is
        actually stopping me', and it is the single most useful output of the
        whole structural layer."""
        if not self.margins:
            return None
        return min(self.margins, key=lambda m: m.safety_factor)

    @property
    def binding_operating(self) -> "Margin | None":
        """The worst margin that actually moves with the operating point.

        This is usually the more useful answer to 'what limits this engine',
        because a fixed geometric check can sit lowest at every speed and tell
        you nothing about where the engine is working hardest.
        """
        candidates = [m for m in self.margins if m.operating_dependent]
        if not candidates:
            return None
        return min(candidates, key=lambda m: m.safety_factor)

    @property
    def failing(self) -> list:
        return [m for m in self.margins if not m.passes]

    def for_component(self, component: str) -> list:
        return [m for m in self.margins if m.component == component]

    def to_dict(self) -> dict:
        binding = self.binding
        operating = self.binding_operating
        return {
            "minimum_safety_factor": (None if math.isinf(self.minimum)
                                      else self.minimum),
            "binding": (f"{binding.component}: {binding.mode}"
                        if binding else None),
            "binding_operating": (f"{operating.component}: {operating.mode}"
                                  if operating else None),
            "failing_count": len(self.failing),
            "margins": [m.to_dict() for m in self.sorted()],
            "thermal": self.thermal.as_dict() if self.thermal else None,
        }


@dataclass
class ComponentContext:
    """Everything a component model needs, assembled once."""

    state: object
    sweep: object
    thermal: ThermalMap

    # -- convenience --------------------------------------------------------

    def p(self, path: str):
        return self.state[path]

    def material(self, role: str) -> Material:
        return materials_mod.get(self.state[f"materials.{role}"])

    def material_key(self, key: str) -> Material:
        return materials_mod.get(key)

    def strength(self, material: Material, temperature: float) -> tuple:
        """Allowable static strength at temperature, and what it is based on.

        Grey iron has no yield point and is far stronger in compression than
        tension, so sizing falls back to ultimate strength -- which the
        returned basis string says out loud.
        """
        y = material.yield_at(temperature)
        if y is None:
            return material.ultimate_at(temperature), "ultimate (no yield point)"
        return y, "yield at temperature"

    def endurance(self, material: Material, temperature: float,
                  diameter: float, finish: str = "machined",
                  load: str = "bending"):
        return fatigue_mod.endurance_limit(
            material, temperature, diameter, finish, load,
            self.state["fatigue.reliability"])

    def rpm(self) -> float:
        return self.sweep.speed * 60.0 / (2.0 * math.pi)

    def at(self, label: str) -> str:
        return f"{label} at {self.rpm():.0f} rpm"


def fatigue_margin(ctx: ComponentContext, component: str, mode: str,
                   alternating: float, mean: float, material: Material,
                   temperature: float, diameter: float, finish: str,
                   load: str, condition: str, extra_inputs: dict | None = None,
                   notes: list | None = None) -> Margin:
    """Assemble a fatigue margin, with every Marin factor recorded."""
    limit = ctx.endurance(material, temperature, diameter, finish, load)
    n = fatigue_mod.goodman_factor(alternating, mean, limit.value, limit.ultimate)
    cycles = fatigue_mod.life_cycles(alternating, mean, limit.value,
                                     limit.ultimate)
    hours = fatigue_mod.hours_at_speed(cycles, ctx.sweep.speed)

    inputs = {
        "alternating_stress_pa": alternating,
        "mean_stress_pa": mean,
        "material": material.key,
        "temperature_c": temperature - 273.15,
        "surface_finish": finish,
        "loading": load,
    }
    limit_inputs = limit.as_dict()
    limit_inputs.pop("notes", None)      # they are reported in notes below
    inputs.update(limit_inputs)
    if extra_inputs:
        inputs.update(extra_inputs)

    all_notes = list(limit.notes) + list(notes or [])
    required = ctx.state["fatigue.design_life_cycles"]
    if math.isfinite(cycles) and cycles < required:
        all_notes.append(
            f"predicted life {cycles:.3g} cycles is short of the "
            f"{required:.3g} cycle design life")

    return Margin(
        component=component, mode=mode, kind="factor",
        applied=n, allowable=1.0, unit="-",
        equation="Goodman: sigma_a/S_e + sigma_m/S_ut = 1/n",
        reference="Shigley, Mechanical Engineering Design, ch. 6 (Marin factors)",
        condition=condition, inputs=inputs, notes=all_notes,
        life_cycles=cycles, life_hours=hours)
