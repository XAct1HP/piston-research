"""One editing session: the live design, its history, and staged proposals.

The propose / commit / revert cycle is here rather than in phase 5 because it
belongs to the design, not to the AI. ``propose()`` evaluates a change against
a copy and returns what would happen without touching the live state. Nothing
is applied until ``commit()``.

That single property is what makes it safe to let an AI drive later: the worst
case is a wasted evaluation. It is also just a good way for a person to work --
try it, see the whole consequence including what got worse, keep it or throw it
away.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from ..evaluate import compare, evaluate
from ..schema import load_state, refresh_bounds
from ..state import ConstraintViolation, DesignState


@dataclass
class Proposal:
    id: str
    changes: dict
    rationale: str
    actor: str
    state: DesignState
    deltas: dict


@dataclass
class Session:
    """The design being edited, plus undo and staged proposals."""

    state: DesignState
    path: str | None = None
    proposals: dict = field(default_factory=dict)
    undo_stack: list = field(default_factory=list)

    @classmethod
    def from_file(cls, path: str) -> "Session":
        state, _ = load_state(path)
        return cls(state=state, path=path)

    # -- reading ------------------------------------------------------------

    def metrics(self):
        return evaluate(self.state)

    # -- writing ------------------------------------------------------------

    def set(self, changes: dict, actor: str = "user",
            rationale: str = "") -> dict:
        """Apply changes to the live design. Refusals come back structured."""
        before = evaluate(self.state)
        snapshot = self.state.copy()
        try:
            for path, value in changes.items():
                self.state.set(path, value, actor=actor, rationale=rationale)
        except ConstraintViolation as exc:
            self.state = snapshot
            return {"ok": False, "violation": exc.as_dict()}

        refresh_bounds(self.state)

        # A value can be inside its own bounds and still make the design
        # impossible -- a rod flange inside its range but too thick for the
        # shank height it sits in. That only shows up when something DERIVED
        # from it is computed, which used to happen after the write had
        # already landed, so the failure reached the browser as a 500 and the
        # design was left holding a value it could not evaluate.
        try:
            after = evaluate(self.state)
        except (ValueError, ZeroDivisionError, ArithmeticError) as exc:
            self.state = snapshot
            return {"ok": False, "violation": {
                "error": "impossible_combination",
                "path": ", ".join(changes),
                "reason": str(exc),
                "attempted": {k: v for k, v in changes.items()},
            }}

        self.undo_stack.append(snapshot)
        return {"ok": True, "deltas": compare(before, after)}

    def propose(self, changes: dict, rationale: str = "",
                actor: str = "user") -> dict:
        """Evaluate a change without applying it.

        Returns the resulting metrics and the deltas, including what got
        worse. A tool that only surfaces the improvement teaches you to trust
        it exactly when you should not.
        """
        before = evaluate(self.state)
        try:
            candidate = self.state.with_changes(changes, actor=actor,
                                                rationale=rationale)
            evaluate(candidate)
        except ConstraintViolation as exc:
            return {"ok": False, "violation": exc.as_dict()}
        except (ValueError, ZeroDivisionError, ArithmeticError) as exc:
            return {"ok": False, "violation": {
                "error": "impossible_combination",
                "path": ", ".join(changes),
                "reason": str(exc),
                "attempted": {k: v for k, v in changes.items()},
            }}

        refresh_bounds(candidate)
        after = evaluate(candidate)
        proposal = Proposal(
            id=uuid.uuid4().hex[:12], changes=changes, rationale=rationale,
            actor=actor, state=candidate, deltas=compare(before, after))
        self.proposals[proposal.id] = proposal
        return {
            "ok": True,
            "proposal_id": proposal.id,
            "changes": changes,
            "rationale": rationale,
            "deltas": proposal.deltas,
            "metrics": after.to_dict(),
            "regressions": self._regressions(proposal.deltas),
        }

    def commit(self, proposal_id: str) -> dict:
        proposal = self.proposals.pop(proposal_id, None)
        if proposal is None:
            return {"ok": False, "error": f"no proposal {proposal_id}"}
        self.undo_stack.append(self.state.copy())
        self.state = proposal.state
        return {"ok": True, "committed": proposal.changes}

    def revert(self) -> dict:
        if not self.undo_stack:
            return {"ok": False, "error": "nothing to undo"}
        self.state = self.undo_stack.pop()
        return {"ok": True}

    def discard(self, proposal_id: str) -> dict:
        self.proposals.pop(proposal_id, None)
        return {"ok": True}

    def save(self) -> dict:
        if not self.path:
            return {"ok": False, "error": "this session has no file"}
        self.state.save(self.path)
        return {"ok": True, "path": self.path}

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _regressions(deltas: dict) -> list:
        """Name what got worse. Higher load, higher stress, more mass, and any
        safety factor that fell."""
        worse = []
        for key, delta in deltas.items():
            rising_is_bad = any(t in key for t in (
                "peak_pin", "peak_rod", "side_thrust", "reciprocating",
                "peak_pressure", "acceleration", "fmep"))
            if rising_is_bad and delta["delta"] > 0:
                worse.append({"metric": key, "percent": delta["percent"]})
        return sorted(worse, key=lambda d: -(d["percent"] or 0.0))
