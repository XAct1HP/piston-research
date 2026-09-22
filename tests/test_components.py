"""Every closed-form component model against a hand calculation."""

import math

import numpy as np
import pytest

from psrt.components import (big_end, bolts, crown, evaluate_all, pin, rings,
                             rod, skirt, sleeve, small_end)
from psrt.units import rpm_to_rad_s


def _find(margins, fragment):
    for m in margins:
        if fragment.lower() in m.mode.lower():
            return m
    raise AssertionError(f"no margin matching {fragment!r} in "
                         f"{[m.mode for m in margins]}")


# --- sleeve ----------------------------------------------------------------

def test_sleeve_hoop_stress_matches_lame(ctx, state):
    m = _find(sleeve.evaluate(ctx), "hoop")
    a = state["engine.bore"] / 2.0
    b = a + state["sleeve.wall_thickness"]
    p = ctx.sweep.trace.peak_pressure
    expected = p * (b * b + a * a) / (b * b - a * a)
    assert m.inputs["hoop_stress_pa"] == pytest.approx(expected, rel=1e-12)


def test_thicker_wall_lowers_hoop_stress(ctx, state):
    before = _find(sleeve.evaluate(ctx), "hoop").applied
    state.set("sleeve.wall_thickness_measured", 0.012)
    assert _find(sleeve.evaluate(ctx), "hoop").applied < before


def test_wall_check_is_flagged_as_not_operating_dependent(ctx):
    assert _find(sleeve.evaluate(ctx), "wall thickness").operating_dependent is False


def test_estimated_wall_thickness_says_so(ctx):
    assert any("sonic-test" in n for n in _find(sleeve.evaluate(ctx), "hoop").notes)


# --- crown -----------------------------------------------------------------

def test_crown_edge_stress_matches_the_plate_formula(ctx, state):
    m = _find(crown.evaluate(ctx), "edge")
    a = (state["engine.bore"] / 2.0
         * state["piston.crown_support_radius_fraction"])
    t = state["piston.crown_thickness"]
    q = ctx.sweep.trace.peak_pressure - state["operating.crankcase_pressure"]
    assert m.inputs["bending_stress_pa"] == pytest.approx(
        3.0 * q * a * a / (4.0 * t * t), rel=1e-12)


def test_crown_edge_carries_more_stress_than_the_centre(ctx):
    margins = crown.evaluate(ctx)
    assert (_find(margins, "edge").inputs["bending_stress_pa"]
            > _find(margins, "centre").inputs["bending_stress_pa"])


def test_crown_centre_is_checked_at_a_higher_temperature(ctx):
    margins = crown.evaluate(ctx)
    assert (_find(margins, "centre").inputs["temperature_c"]
            > _find(margins, "edge").inputs["temperature_c"])


def test_crown_stress_scales_with_the_square_of_thickness(ctx, state):
    before = _find(crown.evaluate(ctx), "edge").inputs["bending_stress_pa"]
    state.set("piston.crown_thickness", state["piston.crown_thickness"] * 2.0)
    after = _find(crown.evaluate(ctx), "edge").inputs["bending_stress_pa"]
    assert after == pytest.approx(before / 4.0, rel=1e-9)


def test_crown_static_check_carries_no_notch_factor(ctx):
    assert any("no notch factor" in n
               for n in _find(crown.evaluate(ctx), "edge").notes)


def test_thermal_fatigue_declares_its_sensitivity(ctx):
    m = _find(crown.evaluate(ctx), "thermal-mechanical")
    assert any("SCREENING CHECK ONLY" in n for n in m.notes)
    assert any("constraint factor" in n for n in m.notes)


def test_crown_support_fraction_is_flagged_as_the_dominant_estimate(ctx):
    assert any("dominates the result" in n
               for n in _find(crown.evaluate(ctx), "edge").notes)


# --- ring lands ------------------------------------------------------------

def test_land_bending_matches_the_cantilever_formula(ctx, state):
    m = _find(rings.evaluate(ctx), "top land bending")
    p = ctx.sweep.trace.peak_pressure
    h = state["piston.ring_groove_depth"]
    b = state["piston.top_land_height"]
    assert m.applied == pytest.approx(3.0 * p * h * h / (b * b), rel=1e-12)


def test_a_shorter_land_is_worse_as_the_square(ctx, state):
    before = _find(rings.evaluate(ctx), "top land bending").applied
    state.set("piston.top_land_height", state["piston.top_land_height"] / 2.0)
    after = _find(rings.evaluate(ctx), "top land bending").applied
    assert after == pytest.approx(4.0 * before, rel=1e-9)


def test_groove_pressure_admits_it_misses_pounding(ctx):
    assert any("pounding" in n
               for n in _find(rings.evaluate(ctx), "flank pressure").notes)


# --- skirt -----------------------------------------------------------------

def test_skirt_pressure_uses_the_projected_area(ctx, state):
    m = _find(skirt.evaluate(ctx), "specific pressure")
    expected = abs(ctx.sweep.peak_side_thrust.value) / state["piston.skirt_bearing_area"]
    assert m.applied == pytest.approx(expected, rel=1e-12)


def test_a_longer_skirt_lowers_pressure(ctx, state):
    """Stays inside the skirt panel the geometry actually has: the bearing
    length is now capped at total height less the ring belt, because the
    structural model and the geometry builder had drifted apart on what a
    skirt is."""
    before = _find(skirt.evaluate(ctx), "specific pressure").applied
    panel = state["piston.total_height"] - state["piston.ring_belt_height"]
    longer = min(state["piston.skirt_length"] * 1.5, panel)
    ratio = longer / state["piston.skirt_length"]
    assert ratio > 1.05, "no headroom left in the panel to lengthen into"

    state.set("piston.skirt_length", longer)
    after = _find(skirt.evaluate(ctx), "specific pressure").applied
    assert after == pytest.approx(before / ratio, rel=1e-9)


def test_the_bearing_length_cannot_exceed_the_skirt_that_exists(state):
    """The gap the optimiser found: the structural model read skirt_length
    for bearing area and the geometry builder never read it at all, so the
    two had drifted 11% apart on the LS3 and 20% on the default."""
    from psrt.state import ConstraintViolation

    panel = state["piston.total_height"] - state["piston.ring_belt_height"]
    with pytest.raises(ConstraintViolation, match="skirt panel"):
        state.set("piston.skirt_length", panel * 1.05)


def test_a_longer_rod_lowers_side_thrust(ctx, state):
    """Less obliquity, less side thrust. The rod ratio trade, made visible."""
    from psrt import loads as loads_mod
    from psrt.margins import ComponentContext
    before = _find(skirt.evaluate(ctx), "specific pressure").applied
    state.set("engine.rod_length", state["engine.rod_length"] * 1.2)
    longer = ComponentContext(state, loads_mod.compute(state), ctx.thermal)
    assert _find(skirt.evaluate(longer), "specific pressure").applied < before


def test_skirt_admits_no_oil_film_is_modelled(ctx):
    assert any("lubrication failure" in n
               for n in _find(skirt.evaluate(ctx), "specific pressure").notes)


# --- pin -------------------------------------------------------------------

def test_pin_bending_matches_the_beam_formula(ctx, state):
    m = _find(pin.evaluate(ctx), "bending between")
    span = state["pin.bending_span"]
    width = state["small_end.bushing_width"]
    z = state["pin.section_modulus"]
    force = m.inputs["governing_force_n"]
    expected = abs(force * (2.0 * span - width) / 8.0 / z)
    assert m.applied == pytest.approx(expected, rel=1e-12)


def test_pin_bending_reverses_sign_with_the_load(ctx):
    m = _find(pin.evaluate(ctx), "bending between")
    assert m.inputs["stress_at_peak_compression_pa"] > 0
    assert m.inputs["stress_at_peak_tension_pa"] < 0


def test_boring_the_pin_out_costs_ovality_fast(ctx, state):
    before = _find(pin.evaluate(ctx), "ovalisation").applied
    state.set("pin.inner_diameter", state["pin.inner_diameter"] * 1.25)
    after = _find(pin.evaluate(ctx), "ovalisation").applied
    assert after > 1.5 * before, "the diameter ratio is cubed"


def test_support_span_factor_moves_the_moment(ctx, state):
    wide = _find(pin.evaluate(ctx), "bending between").applied
    state.set("pin.support_span_factor", 0.0)
    narrow = _find(pin.evaluate(ctx), "bending between").applied
    assert narrow < wide


# --- small end -------------------------------------------------------------

def test_small_end_pressure_uses_projected_area(ctx, state):
    margins = small_end.evaluate(ctx)
    m = _find(margins, "compression")
    projected = state["pin.outer_diameter"] * state["small_end.bushing_width"]
    expected = abs(ctx.sweep.peak_pin_compression.value) / projected
    assert m.applied == pytest.approx(expected, rel=1e-12)


def test_small_end_reports_both_load_directions(ctx):
    modes = [m.mode for m in small_end.evaluate(ctx)]
    assert any("compression" in m for m in modes)
    assert any("tension" in m for m in modes)


# --- rod -------------------------------------------------------------------

def test_rod_tension_uses_the_shank_area(ctx, state):
    m = _find(rod.evaluate(ctx), "shank tension")
    expected = abs(ctx.sweep.peak_rod_tension.value) / state["rod.shank_area"]
    assert m.applied == pytest.approx(expected, rel=1e-12)


def test_euler_and_johnson_agree_at_the_transition():
    """Both curves must pass through S_y/2 at the transition slenderness.
    If they do not, the column model has a discontinuity in it."""
    yield_strength, modulus = 900e6, 205e9
    transition = math.sqrt(2.0 * math.pi ** 2 * modulus / yield_strength)
    below, regime_lo = rod.critical_stress(transition * 0.999, yield_strength, modulus)
    above, regime_hi = rod.critical_stress(transition * 1.001, yield_strength, modulus)
    assert below == pytest.approx(yield_strength / 2.0, rel=5e-3)
    assert above == pytest.approx(yield_strength / 2.0, rel=5e-3)
    assert abs(below - above) / yield_strength < 1e-2, "no jump at the transition"
    assert "Johnson" in regime_lo and "Euler" in regime_hi


def test_a_stocky_column_approaches_yield_strength():
    value, regime = rod.critical_stress(1e-6, 900e6, 205e9)
    assert value == pytest.approx(900e6, rel=1e-6)
    assert "Johnson" in regime


def test_a_slender_column_follows_euler():
    value, regime = rod.critical_stress(300.0, 900e6, 205e9)
    assert value == pytest.approx(math.pi ** 2 * 205e9 / 300.0 ** 2, rel=1e-12)
    assert "Euler" in regime


def test_rod_is_checked_about_both_axes(ctx):
    modes = [m.mode for m in rod.evaluate(ctx)]
    assert any("in the plane" in m for m in modes)
    assert any("out of the plane" in m for m in modes)


def test_out_of_plane_slenderness_is_higher_but_better_restrained(ctx, state):
    assert (state["rod.slenderness_out_of_plane"]
            > state["rod.slenderness_in_plane"])
    assert state["rod.k_out_of_plane"] < state["rod.k_in_plane"]


def test_rod_fatigue_mean_stress_is_compressive(ctx):
    """Which Goodman does not penalise, so the tensile excursion governs."""
    m = _find(rod.evaluate(ctx), "shank fatigue")
    assert m.inputs["mean_stress_pa"] < 0


# --- bolts -----------------------------------------------------------------

def test_bolt_load_includes_the_cap_centrifugal_force(ctx, state):
    m = _find(bolts.evaluate(ctx), "separation")
    expected = (state["bolts.cap_mass"] * ctx.sweep.speed ** 2
                * state["engine.crank_radius"])
    assert m.inputs["cap_centrifugal_n"] == pytest.approx(expected, rel=1e-12)
    assert m.inputs["total_external_load_n"] > m.inputs["rod_tension_at_overlap_tdc_n"]


def test_bolt_load_is_shared_between_bolts(ctx, state):
    m = _find(bolts.evaluate(ctx), "separation")
    assert m.inputs["load_per_bolt_n"] == pytest.approx(
        m.inputs["total_external_load_n"] / state["bolts.count"], rel=1e-12)


def test_all_three_bolt_checks_are_reported(ctx):
    modes = [m.mode for m in bolts.evaluate(ctx)]
    assert any("separation" in m for m in modes)
    assert any("fatigue" in m for m in modes)
    assert any("yielding" in m for m in modes)


def test_separation_warns_that_it_invalidates_the_fatigue_check(ctx):
    assert any("no longer applies" in n
               for n in _find(bolts.evaluate(ctx), "separation").notes)


# --- big end ---------------------------------------------------------------

def test_big_end_pressure_uses_projected_area(ctx, state):
    m = _find(big_end.evaluate(ctx), "specific pressure")
    projected = state["rod.big_end_bore"] * state["rod.big_end_width"]
    assert m.applied == pytest.approx(
        abs(m.inputs["peak_rod_force_n"]) / projected, rel=1e-12)


def test_ocvirk_solution_round_trips(ctx):
    """The eccentricity the solver returns must carry the load it was given."""
    m = _find(big_end.evaluate(ctx), "oil film")
    eps = m.inputs["eccentricity_ratio"]
    load = big_end._ocvirk_load(
        eps, viscosity=m.inputs["viscosity_pa_s"], omega=ctx.sweep.speed,
        radius=ctx.state["rod.big_end_bore"] / 2.0,
        length=ctx.state["rod.big_end_width"],
        clearance=m.inputs["radial_clearance_m"])
    assert load == pytest.approx(m.inputs["peak_rod_force_n"], rel=1e-4)


def test_film_thickness_follows_eccentricity(ctx):
    m = _find(big_end.evaluate(ctx), "oil film")
    assert m.inputs["minimum_film_m"] == pytest.approx(
        m.inputs["radial_clearance_m"] * (1.0 - m.inputs["eccentricity_ratio"]),
        rel=1e-12)


def test_oil_film_declares_itself_a_lower_bound(ctx):
    assert any("LOWER BOUND ONLY" in n
               for n in _find(big_end.evaluate(ctx), "oil film").notes)


# --- the report ------------------------------------------------------------

def test_every_component_contributes_margins(ctx, state):
    report = evaluate_all(state, ctx.sweep, ctx.thermal)
    components = {m.component for m in report.margins}
    assert components == {"sleeve", "crown", "ring lands", "skirt", "pin",
                          "small end", "rod", "rod bolts", "big end"}


def test_every_margin_carries_its_provenance(ctx, state):
    for m in evaluate_all(state, ctx.sweep, ctx.thermal).margins:
        assert m.equation, f"{m.component}/{m.mode} has no equation"
        assert m.reference, f"{m.component}/{m.mode} has no reference"
        assert m.condition, f"{m.component}/{m.mode} has no condition"
        assert m.inputs, f"{m.component}/{m.mode} has no inputs"


def test_binding_is_the_worst_margin(ctx, state):
    report = evaluate_all(state, ctx.sweep, ctx.thermal)
    assert report.binding.safety_factor == report.minimum


def test_binding_operating_ignores_geometric_checks(ctx, state):
    report = evaluate_all(state, ctx.sweep, ctx.thermal)
    assert report.binding_operating.operating_dependent is True
