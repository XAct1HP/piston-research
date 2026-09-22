"""The optimiser, and every way it could lie.

An optimiser is the best tool ever built for finding a model's weak spots and
driving into them, so most of what is tested here is not "does it find a
better design" but "does it refuse to cheat, and does it say what it doubts".
"""

import warnings

import numpy as np
import pytest

from psrt.evaluate import evaluate
from psrt.optimise import (DEFAULT_CLASSES, LEVER_CLASSES, frontier,
                           optimise)
from psrt.schema import load_state
from psrt.state import ConstraintViolation

warnings.filterwarnings("ignore", message="Values in x were outside bounds")


@pytest.fixture(scope="module")
def ls3():
    return load_state("examples/ls3.json")[0]


# Each search costs seconds, and a mass objective rebuilds the solids on every
# candidate. Compute the handful of runs these tests share exactly once.
@pytest.fixture(scope="module")
def torque_run(ls3):
    return optimise(ls3, "torque", max_levers=5, verify=False)


@pytest.fixture(scope="module")
def mass_run(ls3):
    """At the design's own safety floor there is no margin to spend, so the
    right answer is that nothing moves."""
    return optimise(ls3, "reciprocating_mass", max_levers=4, verify=False)


@pytest.fixture(scope="module")
def mass_run_with_room(ls3):
    """Named levers rather than a ranked search: this fixture rebuilds the
    solids on every candidate, so it is the most expensive thing in the file
    and there is no reason to let it wander. piston.skirt_length is in the
    list on purpose -- it is the lever the geometry builder ignores."""
    return optimise(ls3, "reciprocating_mass",
                    levers=["piston.total_height", "piston.skirt_length",
                            "piston.ring_groove_depth"],
                    min_safety_factor=1.20, max_iterations=12, verify=False)


@pytest.fixture(scope="module")
def torque_frontier(ls3):
    return frontier(ls3, "torque", points=4, max_levers=4, verify=False)


# --- it cannot cheat -------------------------------------------------------

def test_the_result_respects_every_parameter_bound(ls3, torque_run):
    """Candidates are written through DesignState.set, so a bound refuses the
    optimiser exactly as it refuses a person."""
    result = torque_run
    for move in result.moves:
        param = ls3.param(move.path)
        if param.minimum is not None:
            assert move.after >= param.minimum - 1e-12
        if param.maximum is not None:
            assert move.after <= param.maximum + 1e-12


def test_the_result_cannot_exceed_the_block_envelope(ls3, torque_run):
    """The overbore ceiling is the constraint most worth trying to sneak
    past, because the bore is the strongest torque lever there is."""
    assert torque_run.state["engine.bore"] <= ls3.param("engine.bore").maximum


def test_the_returned_design_is_valid_on_its_own_terms(torque_run):
    assert not torque_run.state.validate()


def test_the_safety_floor_actually_holds(ls3):
    result = optimise(ls3, "torque", max_levers=6, min_safety_factor=1.35,
                      verify=False)
    assert result.safety_after >= 1.35 - 1e-6
    fresh = evaluate(result.state)
    assert fresh.structural["minimum_operating_safety_factor"] >= 1.35 - 1e-6


def test_mass_bookkeeping_is_not_a_lever(mass_run_with_room):
    """masses.piston is a stored number, not a design choice. Letting the
    search write it buys mass reductions that no machining would produce."""
    assert not any(m.path.startswith("masses.")
                   for m in mass_run_with_room.moves)


# --- it finds the right answer ---------------------------------------------

def test_it_matches_a_brute_force_search(ls3):
    """Two levers, small enough to check exhaustively. The optimiser must be
    at least as good as a grid -- it is allowed to be better, because the
    grid has finite resolution."""
    paths = ["engine.bore", "engine.compression_ratio"]
    result = optimise(ls3, "torque", levers=paths, verify=False)

    floor = evaluate(ls3).structural["minimum_operating_safety_factor"]
    bore = ls3.param("engine.bore")
    best = -np.inf
    for b in np.linspace(bore.minimum, bore.maximum, 9):
        for cr in np.linspace(9.5, 12.5, 21):
            try:
                candidate = ls3.with_changes(
                    {"engine.bore": float(b),
                     "engine.compression_ratio": float(cr)},
                    actor="grid", rationale="brute force")
                metrics = evaluate(candidate)
            except (ConstraintViolation, ValueError):
                continue
            safety = metrics.structural["minimum_operating_safety_factor"]
            if safety is None or safety < floor - 1e-9:
                continue
            best = max(best, metrics.performance["brake_torque_nm"])

    assert result.after >= best * 0.999


def test_it_is_deterministic(ls3, torque_run):
    """A tool whose whole claim is traceable numbers cannot give a different
    answer each run."""
    again = optimise(ls3, "torque", max_levers=5, verify=False)
    assert again.after == pytest.approx(torque_run.after, rel=1e-12)


# --- it says what it doubts -------------------------------------------------

def test_it_flags_an_optimum_that_rests_on_a_calibrated_guess(torque_run):
    """If the binding constraint's model is a calibrated constant rather than
    an FEA-fitted one, the answer inherits the guess and must say so."""
    assert any("CALIBRATED" in w for w in torque_run.warnings)


def test_it_flags_a_lever_pinned_against_its_bound(torque_run):
    """Then the answer is set by the bound, not by the physics."""
    assert any("against a limit" in w for w in torque_run.warnings)


def test_it_reports_what_got_worse(ls3):
    result = optimise(ls3, "safety_factor", max_levers=6, verify=False)
    labels = {row["label"] for row in result.worse}
    assert "brake torque" in labels, "buying margin with torque must be shown"


def test_it_explains_a_refusal_instead_of_returning_nothing(mass_run):
    """The LS3 sits on its own safety floor, so with no margin to spend there
    is no mass to save. That is a well-posed answer and has to read like
    one."""
    assert not mass_run.moved
    assert any("safety floor" in w for w in mass_run.warnings)


def test_a_lever_with_no_effect_is_reported_not_silently_dropped(
        mass_run_with_room):
    """piston.skirt_length is read by the structural model and ignored by the
    geometry builder, so it moves no mass at all. A lever that does nothing
    is a gap between two models, and the tool should say so rather than
    quietly leaving it out of the answer."""
    assert any("not at all in this model" in w
               for w in mass_run_with_room.warnings)


# --- lever classes ---------------------------------------------------------

def test_the_default_search_does_not_buy_a_new_crankshaft(torque_run):
    """Turned loose on everything, the search answers 'more torque' with a
    longer stroke and a longer burn -- true, and a different engine plus a
    remap rather than a redesigned piston system."""
    moved = {m.path for m in torque_run.moved}
    assert not moved & set(LEVER_CLASSES["rotating"])
    assert not moved & set(LEVER_CLASSES["tuning"])


def test_opting_into_a_new_bottom_end_says_what_it_costs(ls3):
    result = optimise(ls3, "torque", max_levers=8,
                      classes=DEFAULT_CLASSES + ("rotating", "tuning"),
                      verify=False)
    if {m.path for m in result.moved} & set(LEVER_CLASSES["rotating"]):
        assert any("crankshaft" in w for w in result.warnings)
    if {m.path for m in result.moved} & set(LEVER_CLASSES["tuning"]):
        assert any("Wiebe" in w for w in result.warnings)


def test_an_unknown_objective_is_refused_with_the_list(ls3):
    with pytest.raises(ValueError, match="unknown objective"):
        optimise(ls3, "vibes", verify=False)


# --- the frontier ----------------------------------------------------------

def test_the_frontier_is_monotonic(torque_frontier):
    """More safety factor cannot buy more torque. If it does, the search is
    finding different local optima at different levels and the curve is not
    a frontier at all."""
    usable = [p for p in torque_frontier.points if p["feasible"]]
    assert len(usable) >= 3
    values = [p["value"] for p in usable]
    assert all(b <= a + 1e-6 for a, b in zip(values, values[1:])), values


def test_every_frontier_point_actually_meets_its_constraint(torque_frontier):
    """A front whose points are not individually buildable is a picture, not
    an answer."""
    for point in torque_frontier.points:
        if point["feasible"]:
            assert point["safety"] >= point["level"] - 1e-6
