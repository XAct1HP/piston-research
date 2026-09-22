"""The design state: constraint enforcement, atomicity, and round-tripping."""

import json

import pytest

from psrt.schema import default_state, refresh_bounds
from psrt.state import (ConstraintViolation, DesignState, Mutability,
                        Parameter, UnknownParameter)


def test_locked_parameter_refuses_writes(state):
    with pytest.raises(ConstraintViolation) as exc:
        state.set("block.as_built_bore", 0.090)
    assert "physical fact" in exc.value.reason


def test_derived_parameter_refuses_writes(state):
    with pytest.raises(ConstraintViolation) as exc:
        state.set("engine.rod_ratio", 3.8)
    assert "derived" in exc.value.reason


def test_bounded_parameter_enforces_both_ends(state):
    lo = state.param("engine.bore").minimum
    hi = state.param("engine.bore").maximum

    with pytest.raises(ConstraintViolation) as under:
        state.set("engine.bore", lo - 0.0005)
    assert under.value.bound == lo
    assert "subtractive" in under.value.reason

    with pytest.raises(ConstraintViolation):
        state.set("engine.bore", hi + 0.0005)

    state.set("engine.bore", (lo + hi) / 2.0)


def test_violation_serialises_for_a_tool_caller(state):
    try:
        state.set("engine.compression_ratio", 30.0)
    except ConstraintViolation as exc:
        payload = exc.as_dict()
    assert payload["error"] == "constraint_violation"
    assert payload["path"] == "engine.compression_ratio"
    assert payload["bound"] == 14.0
    json.dumps(payload)


def test_non_numeric_write_to_a_numeric_parameter_is_refused(state):
    with pytest.raises(ConstraintViolation):
        state.set("engine.bore", "bigger")


def test_unknown_parameter_raises(state):
    with pytest.raises(UnknownParameter):
        state.set("engine.turbo_boost", 2.0)


def test_unlocking_permits_the_write(state):
    state.lock("engine.stroke", "crankshaft is not being remanufactured")
    with pytest.raises(ConstraintViolation) as exc:
        state.set("engine.stroke", 0.090)
    assert "remanufactured" in exc.value.reason

    state.unlock("engine.stroke")
    state.set("engine.stroke", 0.090, rationale="longer stroke crank sourced")
    assert state["engine.stroke"] == 0.090


def test_with_changes_does_not_touch_the_original(state):
    before = state.fingerprint()
    candidate = state.with_changes({"engine.bore": 0.0875}, "ai", "test")
    assert state.fingerprint() == before
    assert state["engine.bore"] != candidate["engine.bore"]


def test_with_changes_is_atomic(state):
    before = state.fingerprint()
    with pytest.raises(ConstraintViolation):
        state.with_changes(
            {"engine.bore": 0.0875, "engine.stroke": 0.088,
             "block.as_built_bore": 0.100},
            "ai", "one of these is locked")
    assert state.fingerprint() == before


def test_every_change_is_logged_with_an_actor_and_a_rationale(state):
    state.set("engine.bore", 0.0875, actor="ai", rationale="chasing torque")
    entry = state.log[-1]
    assert entry.path == "engine.bore"
    assert entry.actor == "ai"
    assert entry.rationale == "chasing torque"
    assert entry.old != entry.new


def test_fingerprint_is_stable_and_sensitive(state):
    a = state.fingerprint()
    assert state.copy().fingerprint() == a
    state.set("masses.piston", state["masses.piston"] + 0.001)
    assert state.fingerprint() != a


def test_json_round_trip_preserves_everything(state):
    restored = DesignState.from_json(state.to_json())
    assert restored.fingerprint() == state.fingerprint()
    assert restored.param("block.as_built_bore").mutability is Mutability.LOCKED
    assert restored.param("engine.bore").why_min == state.param("engine.bore").why_min
    assert restored["engine.rod_ratio"] == pytest.approx(state["engine.rod_ratio"])


def test_derived_values_follow_their_inputs(state):
    before = state["engine.displacement_cyl"]
    state.set("engine.bore", 0.0880)
    assert state["engine.displacement_cyl"] > before
    assert state["engine.displacement_cyl"] == pytest.approx(
        3.141592653589793 / 4 * 0.0880 ** 2 * state["engine.stroke"], rel=1e-12)


def test_rod_mass_split_conserves_total_mass(state):
    total = state["masses.rod_reciprocating"] + state["masses.rod_rotating"]
    assert total == pytest.approx(state["masses.rod_total"], rel=1e-12)


def test_refresh_bounds_ties_the_bore_to_the_block(state):
    bore = state.param("engine.bore")
    assert bore.minimum == state["block.as_built_bore"]
    assert "subtractive" in bore.why_min


def test_resleeving_releases_the_lower_bound(state):
    state.set("block.resleeving_allowed", True)
    refresh_bounds(state)
    assert state.param("engine.bore").minimum is None
    assert "resleeving" in state.param("engine.bore").why_min


def test_siamesed_bores_are_limited_by_their_web_not_by_coolant(state):
    """A siamesed block has no coolant between the cylinders, so the old
    behaviour -- drop the bore-to-bore limit entirely -- was wrong twice
    over: it left the bore unbounded, and it ignored that the web still
    carries head clamping load. It now has a limit, a LARGER one."""
    refresh_bounds(state)
    before = state.param("engine.bore").maximum

    state.set("block.siamesed", True)
    refresh_bounds(state)
    after = state.param("engine.bore").maximum

    assert after is not None
    assert after > before
    assert "web" in state.param("engine.bore").why_max


def test_validate_catches_an_out_of_range_value():
    s = DesignState({"x": {"a": Parameter(5.0, Mutability.BOUNDED, "-",
                                          minimum=0.0, maximum=1.0)}})
    assert any("above its upper bound" in i for i in s.validate())
