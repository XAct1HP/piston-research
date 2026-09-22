"""The design state: parameters, mutability classes, and the change log.

The design is one object. Every parameter in it carries a *mutability class*
that decides who is allowed to write it:

    free      an open design variable
    bounded   open, but only inside [minimum, maximum]
    locked    pinned by the user; writes are refused until explicitly unlocked
    derived   computed from other parameters; never written directly

This is what makes "these specs are not allowed to change" enforceable rather
than a polite request. When the AI layer arrives in phase 5, a write to a
locked parameter comes back as a structured tool error explaining the
constraint -- so the model has to reason around a real limit instead of being
asked nicely to respect one.

Every mutation is recorded in a change log with an actor and a rationale.
That log is the undo stack, the audit trail, and the record you read back when
you want to know why the crown ended up 0.6 mm thicker.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterator


class Mutability(str, Enum):
    FREE = "free"
    BOUNDED = "bounded"
    LOCKED = "locked"
    DERIVED = "derived"


class ConstraintViolation(Exception):
    """A write was refused. Carries enough detail to explain itself."""

    def __init__(self, path: str, attempted: Any, reason: str,
                 bound: Any = None, unit: str = "") -> None:
        self.path = path
        self.attempted = attempted
        self.reason = reason
        self.bound = bound
        self.unit = unit
        super().__init__(f"{path}: {reason} (attempted {attempted!r})")

    def as_dict(self) -> dict:
        """Structured form, for returning to a tool caller."""
        return {
            "error": "constraint_violation",
            "path": self.path,
            "attempted": self.attempted,
            "reason": self.reason,
            "bound": self.bound,
            "unit": self.unit,
        }


class UnknownParameter(KeyError):
    pass


@dataclass
class Parameter:
    """One design variable, with its bounds and the reasons behind them."""

    value: Any = None
    mutability: Mutability = Mutability.FREE
    unit: str = ""
    minimum: float | None = None
    maximum: float | None = None
    why: str = ""            # why it is locked, or a general note
    why_min: str = ""        # what physical fact sets the lower bound
    why_max: str = ""        # what physical fact sets the upper bound
    description: str = ""
    source: str = "user"     # user | published | estimated | derived | measured
    optional: bool = False   # True if leaving this unset is legitimate

    def to_dict(self) -> dict:
        out: dict[str, Any] = {"v": self.value, "class": self.mutability.value}
        if self.unit:
            out["unit"] = self.unit
        if self.minimum is not None:
            out["min"] = self.minimum
        if self.maximum is not None:
            out["max"] = self.maximum
        for key, val in (("why", self.why), ("why_min", self.why_min),
                         ("why_max", self.why_max),
                         ("description", self.description)):
            if val:
                out[key] = val
        if self.source != "user":
            out["source"] = self.source
        if self.optional:
            out["optional"] = True
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "Parameter":
        return cls(
            value=d.get("v"),
            mutability=Mutability(d.get("class", "free")),
            unit=d.get("unit", ""),
            minimum=d.get("min"),
            maximum=d.get("max"),
            why=d.get("why", ""),
            why_min=d.get("why_min", ""),
            why_max=d.get("why_max", ""),
            description=d.get("description", ""),
            source=d.get("source", "user"),
            optional=bool(d.get("optional", False)),
        )


@dataclass
class Change:
    """One recorded mutation."""

    path: str
    old: Any
    new: Any
    actor: str               # "user" | "ai" | "optimiser" | "preset"
    rationale: str
    unit: str = ""
    at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "path": self.path, "old": self.old, "new": self.new,
            "actor": self.actor, "rationale": self.rationale,
            "unit": self.unit, "at": self.at,
        }


# --- Derived parameter registry -------------------------------------------
# A derived parameter is computed, never stored. Registering a function here
# rather than eval()-ing an expression string keeps derivations testable and
# keeps arbitrary code out of a JSON file that an AI can write to.

DerivedFn = Callable[["DesignState"], Any]
_DERIVED: dict[str, DerivedFn] = {}


def derived(path: str) -> Callable[[DerivedFn], DerivedFn]:
    """Decorator registering the function that computes a derived parameter."""

    def wrap(fn: DerivedFn) -> DerivedFn:
        _DERIVED[path] = fn
        return fn

    return wrap


class DesignState:
    """A complete piston-system design.

    Sections are the top level ("engine", "piston", "rod", ...), parameters
    sit inside them, and everything is addressed by dotted path:
    ``state["engine.bore"]``.
    """

    def __init__(self, sections: dict[str, dict[str, Parameter]] | None = None,
                 meta: dict | None = None) -> None:
        self.sections: dict[str, dict[str, Parameter]] = sections or {}
        self.meta: dict = meta or {}
        self.log: list[Change] = []

    # -- access -------------------------------------------------------------

    def param(self, path: str) -> Parameter:
        section, _, name = path.partition(".")
        try:
            return self.sections[section][name]
        except KeyError:
            raise UnknownParameter(f"no such parameter: {path}") from None

    def has(self, path: str) -> bool:
        try:
            self.param(path)
            return True
        except UnknownParameter:
            return False

    def get(self, path: str, default: Any = "__raise__") -> Any:
        """Read a value. Derived parameters are computed on access."""
        try:
            p = self.param(path)
        except UnknownParameter:
            if default == "__raise__":
                raise
            return default
        if p.mutability is Mutability.DERIVED:
            fn = _DERIVED.get(path)
            if fn is None:
                raise UnknownParameter(
                    f"{path} is derived but has no registered derivation")
            return fn(self)
        return p.value

    def __getitem__(self, path: str) -> Any:
        return self.get(path)

    def unit(self, path: str) -> str:
        return self.param(path).unit

    def iter_params(self) -> Iterator[tuple[str, Parameter]]:
        for section, params in self.sections.items():
            for name, p in params.items():
                yield f"{section}.{name}", p

    # -- mutation -----------------------------------------------------------

    def set(self, path: str, value: Any, actor: str = "user",
            rationale: str = "") -> None:
        """Write a parameter in place, enforcing its mutability class.

        Raises ConstraintViolation if the write is not permitted.
        """
        p = self.param(path)

        if p.mutability is Mutability.LOCKED:
            raise ConstraintViolation(
                path, value,
                p.why or "parameter is locked and cannot be changed",
                unit=p.unit)

        if p.mutability is Mutability.DERIVED:
            raise ConstraintViolation(
                path, value,
                "parameter is derived from others and cannot be set directly",
                unit=p.unit)

        if isinstance(p.value, (int, float)) and not isinstance(p.value, bool):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ConstraintViolation(
                    path, value, "expected a numeric value", unit=p.unit)

        if p.mutability is Mutability.BOUNDED:
            if p.minimum is not None and value < p.minimum:
                raise ConstraintViolation(
                    path, value,
                    p.why_min or f"below the lower bound of {p.minimum}",
                    bound=p.minimum, unit=p.unit)
            if p.maximum is not None and value > p.maximum:
                raise ConstraintViolation(
                    path, value,
                    p.why_max or f"above the upper bound of {p.maximum}",
                    bound=p.maximum, unit=p.unit)

        old = p.value
        p.value = value
        self.log.append(Change(path, old, value, actor, rationale, p.unit))

    def with_changes(self, changes: dict[str, Any], actor: str = "ai",
                     rationale: str = "") -> "DesignState":
        """Return a NEW state with these changes applied atomically.

        This is what ``propose()`` will use in phase 5: evaluate the
        consequences of a change without touching the live design. If any
        single change is refused, nothing is applied and the violation
        propagates.
        """
        candidate = self.copy()
        for path, value in changes.items():
            candidate.set(path, value, actor=actor, rationale=rationale)
        return candidate

    def lock(self, path: str, why: str = "") -> None:
        p = self.param(path)
        if p.mutability is Mutability.DERIVED:
            raise ConstraintViolation(path, None, "derived parameters are "
                                      "already not directly writable")
        p.mutability = Mutability.LOCKED
        p.why = why or "locked by the user"

    def unlock(self, path: str, mutability: Mutability = Mutability.FREE) -> None:
        p = self.param(path)
        if p.mutability is not Mutability.LOCKED:
            return
        p.mutability = mutability
        p.why = ""

    # -- housekeeping -------------------------------------------------------

    def copy(self) -> "DesignState":
        clone = DesignState(copy.deepcopy(self.sections),
                            copy.deepcopy(self.meta))
        clone.log = list(self.log)
        return clone

    def fingerprint(self) -> str:
        """Stable hash over stored values. Keys the evaluation cache."""
        payload = {path: p.value for path, p in sorted(self.iter_params())
                   if p.mutability is not Mutability.DERIVED}
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def validate(self) -> list[str]:
        """Check the state is internally consistent. Returns a list of issues."""
        issues: list[str] = []
        for path, p in self.iter_params():
            if p.mutability is Mutability.DERIVED:
                if path not in _DERIVED:
                    issues.append(f"{path}: derived but no derivation registered")
                continue
            if p.value is None:
                if not p.optional:
                    issues.append(f"{path}: no value set")
                continue
            if p.mutability is Mutability.BOUNDED:
                if p.minimum is not None and p.value < p.minimum:
                    issues.append(
                        f"{path}: {p.value} is below its lower bound {p.minimum}")
                if p.maximum is not None and p.value > p.maximum:
                    issues.append(
                        f"{path}: {p.value} is above its upper bound {p.maximum}")
        return issues

    # -- serialisation ------------------------------------------------------

    def to_dict(self, include_derived: bool = False) -> dict:
        out: dict[str, Any] = {
            "schema_version": 1,
            "units": "SI",
            "meta": self.meta,
        }
        for section, params in self.sections.items():
            out[section] = {n: p.to_dict() for n, p in params.items()}
            if include_derived:
                for name, p in params.items():
                    if p.mutability is Mutability.DERIVED:
                        out[section][name]["v"] = self.get(f"{section}.{name}")
        return out

    def to_json(self, indent: int = 2, include_derived: bool = False) -> str:
        return json.dumps(self.to_dict(include_derived), indent=indent)

    @classmethod
    def from_dict(cls, d: dict) -> "DesignState":
        reserved = {"schema_version", "units", "meta", "log"}
        sections: dict[str, dict[str, Parameter]] = {}
        for key, body in d.items():
            if key in reserved or not isinstance(body, dict):
                continue
            sections[key] = {n: Parameter.from_dict(v)
                             for n, v in body.items() if isinstance(v, dict)}
        return cls(sections, d.get("meta", {}))

    @classmethod
    def from_json(cls, text: str) -> "DesignState":
        return cls.from_dict(json.loads(text))

    @classmethod
    def load(cls, path: str) -> "DesignState":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_json(fh.read())

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.to_json())

    def __repr__(self) -> str:
        n = sum(len(v) for v in self.sections.values())
        name = self.meta.get("name", "unnamed")
        return f"<DesignState {name!r} {n} parameters {self.fingerprint()}>"
