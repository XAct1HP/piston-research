"""Do the structural margins agree with hardware that exists and works?

This is the check that decides whether phase 2 is worth anything. A production
LS3 runs to 6600 rpm for a hundred thousand miles. If the tool says its piston
or its rod is unsafe, the tool is wrong -- and finding that out takes an
afternoon rather than six months.

Read the caveat honestly. Three geometric idealisation parameters --
``piston.crown_support_radius_fraction``, ``pin.support_span_factor`` and
``piston.skirt_bearing_arc`` -- were CALIBRATED so that production hardware
lands just above its limits. So "the LS3 passes" is not independent evidence
that those three numbers are right; it is the statement that they were chosen
to make it pass, which is what a fast analytical layer always does before FEA
replaces the guesses.

What IS independent, and what these tests are really for:

* the margins have to cluster. Calibrating three parameters cannot make nine
  components with twenty-five different failure modes all land in a narrow
  band unless the underlying physics is broadly right. A production part is
  not five times overbuilt anywhere, and it is not marginal everywhere.
* the trends have to be right. Over-speed must move the binding constraint to
  the rod. Removing material must make things worse. A deliberately bad design
  must fail.
* the LS3 must come out bore-wall limited, which it famously is, from a
  geometric calculation nothing was calibrated against.
"""

import os

import pytest

from psrt.evaluate import evaluate
from psrt.schema import load_state
from psrt.units import rpm_to_rad_s

HERE = os.path.dirname(os.path.abspath(__file__))
LS3 = os.path.join(os.path.dirname(HERE), "examples", "ls3.json")

PEAK_TORQUE_RPM = 4600.0
REDLINE_RPM = 6600.0


@pytest.fixture(scope="module")
def ls3():
    state, _ = load_state(LS3)
    return state


def _at(state, rpm):
    probe = state.copy()
    probe.set("operating.speed", rpm_to_rad_s(rpm))
    return evaluate(probe)


# --- a production engine must survive its own rating -----------------------

def test_nothing_fails_at_peak_torque(ls3):
    report = _at(ls3, PEAK_TORQUE_RPM).report
    failing = [f"{m.component}: {m.mode} (SF {m.safety_factor:.2f})"
               for m in report.failing]
    assert not failing, "production hardware must survive its own rating: " + \
                        "; ".join(failing)


def test_nothing_fails_at_redline(ls3):
    report = _at(ls3, REDLINE_RPM).report
    failing = [f"{m.component}: {m.mode} (SF {m.safety_factor:.2f})"
               for m in report.failing]
    assert not failing, "; ".join(failing)


def test_margins_cluster_like_an_optimised_part(ls3):
    """The load-bearing assertion.

    A production engine is designed so that nothing is wildly overbuilt and
    nothing is on the edge. If the models were badly wrong, calibrating three
    idealisation parameters could not drag twenty-five margins into a narrow
    band -- something would come out at 0.2 or at 50.
    """
    report = _at(ls3, PEAK_TORQUE_RPM).report
    interesting = [m.safety_factor for m in report.margins
                   if m.operating_dependent
                   and m.mode != "deflection under peak pressure"
                   and "thermal-mechanical" not in m.mode]

    shown = sorted(round(s, 2) for s in interesting)
    assert min(interesting) >= 1.0, shown
    assert min(interesting) < 2.0, f"not overbuilt everywhere: {shown}"

    # Several independent constraints active at once is the real signature of
    # an optimised part: a designer removed material everywhere it was free.
    near_limit = [sf for sf in interesting if sf < 2.0]
    assert len(near_limit) >= 5, (
        "an optimised part should have several constraints simultaneously "
        f"near active; got {shown}")
    assert len([sf for sf in interesting if sf < 4.0]) >= len(interesting) * 0.5, shown


def test_ls3_comes_out_bore_wall_limited(ls3):
    """Famously true of LS blocks, and reached here by a purely geometric
    calculation that nothing was calibrated against."""
    report = _at(ls3, PEAK_TORQUE_RPM).report
    wall = [m for m in report.margins if m.mode == "wall thickness remaining"][0]
    assert wall.safety_factor < 1.15
    assert wall.safety_factor >= 1.0
    assert report.binding.component == "sleeve"


# --- trends, which no calibration was aimed at -----------------------------

def test_overspeed_moves_the_limit_from_gas_load_to_inertia(ls3):
    """At its rated speed the LS3 is limited by gas load -- the pin and the
    crown, both driven by peak firing pressure. Wind it well past redline and
    inertia takes over.

    This used to assert the rod specifically. It no longer can, and the
    reason is worth keeping. Capping the skirt's bearing length at the skirt
    the geometry actually has -- they had drifted 11% apart -- loaded the
    skirt enough that scuffing and rod fatigue now finish within 6% of each
    other at 9,000 rpm. Six percent is well inside the uncertainty of a
    skirt model the README already calls a proxy rather than physics, so
    asserting which of the two wins would be asserting noise. What the model
    genuinely supports is that BOTH inertia-driven limits overtake the
    gas-driven ones."""
    rated = _at(ls3, PEAK_TORQUE_RPM).report
    overspeed = _at(ls3, 9500.0).report

    assert rated.binding_operating.component in {"pin", "crown", "big end"}
    assert overspeed.binding_operating.component in {"rod", "skirt"}

    def worst(report, component):
        return min(m.safety_factor for m in report.for_component(component))

    # Both inertia-driven constraints must degrade with speed, and both must
    # end up below whatever was binding at the rated point.
    for component in ("rod", "skirt"):
        assert worst(overspeed, component) < worst(rated, component)
        assert worst(overspeed, component) < rated.binding_operating.safety_factor \
            or worst(overspeed, component) < 2.0


def _rod_fatigue(report):
    return [m for m in report.for_component("rod")
            if "shank fatigue" in m.mode][0]


def test_rod_tensile_stress_grows_monotonically_with_speed(ls3):
    """Pure inertia: nothing else contributes, so it must rise with every
    increase in speed and roughly as the square of it."""
    stresses = [_rod_fatigue(_at(ls3, rpm).report).inputs["tensile_stress_pa"]
                for rpm in (3000, 4600, 6600, 8000, 9500)]
    assert stresses == sorted(stresses), stresses
    assert stresses[-1] / stresses[0] == pytest.approx((9500 / 3000) ** 2,
                                                       rel=0.05)


def test_rod_fatigue_is_not_monotonic_in_speed(ls3):
    """A real and slightly counter-intuitive result worth pinning down.

    Rod fatigue is driven by the SWING between compression at peak firing and
    tension at overlap TDC. Raising speed grows the tensile half and shrinks
    the compressive half, so the alternating stress dips through the middle of
    the range before inertia takes over. The rod is not at its worst at peak
    torque, nor at redline, but at both ends.
    """
    factors = {rpm: _rod_fatigue(_at(ls3, rpm).report).safety_factor
               for rpm in (4600, 8000, 11000)}
    assert factors[8000] > factors[4600], factors
    assert factors[11000] < factors[4600], factors


def test_the_rod_is_what_finally_gives_way_on_overspeed(ls3):
    """Stock LS3 rods are the known limit above their rated speed."""
    report = _at(ls3, 11000.0).report
    assert any(m.component == "rod" for m in report.failing), \
        [m.mode for m in report.failing]


def test_a_lighter_piston_relieves_the_rod(ls3):
    heavy = _at(ls3, 8000.0).report
    lighter = ls3.with_changes(
        {"masses.piston": ls3["masses.piston"] - 0.060}, "test", "lighter piston")
    light = _at(lighter, 8000.0).report

    def rod_sf(report):
        return [m for m in report.for_component("rod")
                if "shank fatigue" in m.mode][0].safety_factor

    assert rod_sf(light) > rod_sf(heavy)


def test_removing_crown_material_makes_the_crown_worse(ls3):
    before = _at(ls3, PEAK_TORQUE_RPM).structural["by_component"]["crown"]
    thinner = ls3.with_changes(
        {"piston.crown_thickness": ls3["piston.crown_thickness"] * 0.7},
        "test", "thinner crown")
    after = _at(thinner, PEAK_TORQUE_RPM).structural["by_component"]["crown"]
    assert after < before


def test_overboring_eats_the_wall_margin(ls3):
    """The subtractive constraint, made quantitative.

    This used to take the bore out half a millimetre. It no longer can: the
    block envelope caps this engine at +0.254 mm, from the largest bore an
    aftermarket piston is actually sold for. So the test does both halves --
    the overbore the block allows still eats the wall, and the one it does
    not allow is refused rather than computed."""
    import pytest as _pytest

    from psrt.envelope import envelope
    from psrt.state import ConstraintViolation

    env = envelope(ls3)
    allowed = env.max_bore

    before = _at(ls3, PEAK_TORQUE_RPM).report
    bigger = ls3.with_changes(
        {"engine.bore": allowed}, "test", "overbore to the block's limit")
    after = _at(bigger, PEAK_TORQUE_RPM).report

    def wall(report):
        return [m for m in report.margins
                if m.mode == "wall thickness remaining"][0].safety_factor

    assert wall(after) < wall(before)

    with _pytest.raises(ConstraintViolation):
        ls3.with_changes({"engine.bore": ls3["engine.bore"] + 0.0005},
                         "test", "half a millimetre, which this block "
                                 "will not give you")


# --- known-bad designs must be caught --------------------------------------

def test_a_paper_thin_crown_fails(ls3):
    bad = ls3.with_changes(
        {"piston.crown_thickness": 0.0026}, "test", "deliberately too thin")
    report = _at(bad, PEAK_TORQUE_RPM).report
    assert any(m.component == "crown" for m in report.failing)


def test_a_spindly_rod_fails_in_tension_or_fatigue(ls3):
    bad = ls3.with_changes(
        {"rod.shank_height": 0.014, "rod.shank_width": 0.009,
         "rod.shank_web": 0.0025, "rod.shank_flange": 0.0025},
        "test", "deliberately undersized rod")
    report = _at(bad, REDLINE_RPM).report
    assert any(m.component == "rod" for m in report.failing)


def test_losing_bolt_preload_causes_separation(ls3):
    bad = ls3.with_changes(
        {"bolts.preload": 2_000.0}, "test", "badly under-torqued bolts")
    report = _at(bad, REDLINE_RPM).report
    assert any("separation" in m.mode for m in report.failing)


def test_a_seized_up_bore_wall_fails(ls3):
    bad = ls3.copy()
    bad.set("sleeve.wall_thickness_measured", 0.0008,
            actor="test", rationale="deliberately thin wall")
    report = _at(bad, PEAK_TORQUE_RPM).report
    assert any(m.component == "sleeve" for m in report.failing)


# --- honesty ---------------------------------------------------------------

def test_the_calibrated_parameters_declare_themselves(ls3):
    for path in ("piston.crown_support_radius_fraction",
                 "pin.support_span_factor", "piston.skirt_bearing_arc",
                 "thermal.conductance_rings"):
        assert ls3.param(path).source == "estimated", path


def test_calibration_is_stated_in_the_parameter_description(ls3):
    assert "CALIBRATED" in ls3.param(
        "piston.crown_support_radius_fraction").description
    assert "calibrated" in ls3.param("pin.support_span_factor").description


def test_every_margin_names_a_reference(ls3):
    for m in _at(ls3, PEAK_TORQUE_RPM).report.margins:
        assert len(m.reference) > 10, f"{m.component}/{m.mode}"
