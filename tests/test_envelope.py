"""The manufacturing envelope: what the block will actually allow.

The block is the one part of this design that already exists. Everything here
is about not pretending otherwise -- and about being honest that the numbers
which decide whether you break into an oil gallery are the ones nobody
publishes.
"""

import pytest

from psrt.cascade import overbore_cascade
from psrt.envelope import envelope
from psrt.schema import default_state, load_state, refresh_bounds
from psrt.state import ConstraintViolation


@pytest.fixture(scope="module")
def ls3():
    return load_state("examples/ls3.json")[0]


# --- the floor -------------------------------------------------------------

def test_the_bore_cannot_go_below_as_built(ls3):
    """Machining is subtractive. This is the whole premise."""
    state = ls3.copy()
    with pytest.raises(ConstraintViolation, match="subtractive"):
        state.set("engine.bore", state["block.as_built_bore"] - 1e-5,
                  actor="test", rationale="should be refused")


def test_the_refusal_says_why_not_just_no(ls3):
    state = ls3.copy()
    try:
        state.set("engine.bore", 0.05, actor="test", rationale="far too small")
    except ConstraintViolation as exc:
        assert "as-built" in str(exc)
    else:
        pytest.fail("a 50 mm bore in a 103 mm block was accepted")


# --- the ceiling -----------------------------------------------------------

def test_the_ceiling_is_enforced_on_the_parameter_not_checked_afterwards(ls3):
    """The bound lives on engine.bore, so every write path in the tool -- CLI,
    API, optimiser, assistant -- refuses an over-bore rather than computing
    one and mentioning the problem later."""
    state = ls3.copy()
    env = envelope(state)
    with pytest.raises(ConstraintViolation):
        state.set("engine.bore", env.max_bore + 1e-5, actor="test",
                  rationale="one micron past the block's limit")


def test_the_tightest_limit_wins_not_the_last_one_computed(ls3):
    env = envelope(ls3)
    assert env.binding.max_bore == min(l.max_bore for l in env.known_limits)


def test_a_limit_that_cannot_be_evaluated_is_reported_not_skipped(ls3):
    """A skipped limit silently RAISES the ceiling, and the limit nobody
    could evaluate is exactly the one likely to bite."""
    env = envelope(ls3)
    assert env.unknown_limits
    assert all(limit.missing for limit in env.unknown_limits)
    assert any("could not be evaluated" in note for note in env.notes)


def test_no_evaluable_limit_pins_the_bore_rather_than_freeing_it():
    """The dangerous failure mode is an unbounded bore, not a stuck one."""
    state = default_state()
    for path in ("block.bore_spacing", "block.catalogue_max_bore",
                 "block.liner_thickness"):
        if state.has(path):
            state.param(path).value = None
    refresh_bounds(state)
    bore = state.param("engine.bore")
    assert bore.maximum == pytest.approx(state["engine.bore"])
    assert "pinned" in bore.why_max or "no limit" in bore.why_max


# --- data tiers ------------------------------------------------------------

def test_a_catalogue_bore_outranks_a_computed_one(ls3):
    """Tier 2 is the trick worth building around: if a piston maker tools up
    for a bore, the industry has already established it survives. Here it is
    both TIGHTER and more trustworthy than the computed bore-to-bore limit."""
    env = envelope(ls3)
    assert env.binding.name == "aftermarket piston availability"
    assert env.binding.tier == 2

    computed = next(l for l in env.known_limits
                    if l.name.startswith("bore-to-bore"))
    assert computed.tier == 3
    assert env.binding.max_bore < computed.max_bore


def test_a_limit_is_only_as_good_as_its_weakest_input(ls3):
    """Published spacing plus an assumed wall thickness is a tier 3 number."""
    env = envelope(ls3)
    computed = next(l for l in env.known_limits
                    if l.name.startswith("bore-to-bore"))
    assert ls3.param("block.bore_spacing").source == "published"
    assert computed.tier == 3


def test_an_assumed_ceiling_says_so_on_the_parameter():
    state = default_state()
    refresh_bounds(state)
    assert "ASSUMED" in state.param("engine.bore").why_max


# --- liner type ------------------------------------------------------------

def test_a_plated_bore_cannot_be_overbored_at_all(ls3):
    """Nikasil and friends are microns thick: the first cut removes the
    running surface and exposes aluminium no ring can seal against."""
    state = ls3.with_changes({"block.liner_type": "coated-aluminium"},
                             actor="test", rationale="plated bore")
    limit = next(l for l in envelope(state).limits if l.name == "plated bore")
    assert limit.max_bore == pytest.approx(state["block.as_built_bore"])
    assert "replating" in limit.reason


# --- resleeving ------------------------------------------------------------

def test_resleeving_removes_the_floor(ls3):
    state = ls3.with_changes({"block.resleeving_allowed": True},
                             actor="test", rationale="sleeves")
    refresh_bounds(state)
    assert envelope(state).min_bore is None
    assert state.param("engine.bore").minimum is None


def test_resleeving_must_not_make_the_block_worse(ls3):
    """The first cut of this model subtracted two sleeve walls from the
    unsleeved ceiling and made resleeving come out SMALLER than stock -- which
    would mean nobody would ever pay for it. Interlocking sleeves become the
    structure between the bores, which is where the gain comes from."""
    stock = envelope(ls3).max_bore
    sleeved_state = ls3.with_changes({"block.resleeving_allowed": True},
                                     actor="test", rationale="sleeves")
    sleeved = envelope(sleeved_state).max_bore
    assert sleeved > stock


def test_resleeving_does_not_move_an_oil_gallery(ls3):
    """A sleeve helps with the bore. It does not relocate the block's
    plumbing, and a sleeve big enough to reach a gallery still scraps it."""
    state = ls3.with_changes(
        {"block.resleeving_allowed": True, "block.gallery_offset": 0.0500},
        actor="test", rationale="sleeves and a known gallery")
    limit = next(l for l in envelope(state).limits if l.name == "oil gallery")
    assert limit.known
    assert limit.max_bore < envelope(state).as_built + 0.010


# --- the cascade -----------------------------------------------------------

def test_an_overbore_reports_what_got_worse_as_well_as_better(ls3):
    """A tool that answers a bigger bore with only the torque gain is lying
    to you by omission."""
    cascade = overbore_cascade(ls3, 0.103500, remeasure=False)
    assert cascade.better and cascade.worse
    labels = {c.label for c in cascade.worse}
    assert "peak side thrust" in labels


def test_the_cascade_remeasures_mass_because_the_stored_one_is_stale(ls3):
    """masses.piston is a stored number, not an expression, so without a
    re-measure a bigger bore reports more torque and an UNCHANGED piston --
    hiding the inertia penalty that makes overboring a trade at all."""
    fast = overbore_cascade(ls3, 0.103500, remeasure=False)
    full = overbore_cascade(ls3, 0.103500, remeasure=True)

    assert not any(c.label == "piston mass" for c in fast.consequences)
    grew = next(c for c in full.consequences if c.label == "piston mass")
    assert grew.delta > 0
    assert grew.verdict == "worse"
    assert any("NOT re-measured" in note for note in fast.notes)


def test_the_cascade_cannot_sneak_past_the_envelope(ls3):
    with pytest.raises(ConstraintViolation):
        overbore_cascade(ls3, 0.110, remeasure=False)


def test_the_cascade_reports_the_wall_in_millimetres(ls3):
    """The wall is what kills blocks and it is the thing the user cannot
    see, so it gets said in millimetres rather than left as a ratio."""
    cascade = overbore_cascade(ls3, 0.103500, remeasure=False)
    assert any("bore-to-bore wall" in note for note in cascade.notes)
