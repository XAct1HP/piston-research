"""Fatigue: Marin factors, mean-stress corrections and bolt checks."""

import math

import pytest

from psrt import fatigue as fa
from psrt import materials as mt
from psrt.units import c_to_k

STEEL = mt.get("4340")
ALUMINIUM = mt.get("2618-T61")


def test_surface_factor_against_shigley_table():
    """k_a = a S_ut^b. For 4340 at 1100 MPa, machined gives 4.51*1100^-0.265."""
    assert fa.surface_factor(STEEL, "machined") == pytest.approx(
        4.51 * 1100.0 ** -0.265, rel=1e-12)
    assert fa.surface_factor(STEEL, "polished") == 1.0
    assert (fa.surface_factor(STEEL, "as-forged")
            < fa.surface_factor(STEEL, "machined")
            < fa.surface_factor(STEEL, "ground"))


def test_unknown_surface_finish_is_refused():
    with pytest.raises(KeyError):
        fa.surface_factor(STEEL, "sandblasted")


def test_size_factor():
    assert fa.size_factor(0.020) == pytest.approx(1.24 * 20.0 ** -0.107, rel=1e-12)
    assert fa.size_factor(0.020, "axial") == 1.0, "axial loading has no size effect"
    assert fa.size_factor(0.100) == pytest.approx(1.51 * 100.0 ** -0.157, rel=1e-12)
    assert fa.size_factor(0.002) == 1.0


def test_load_and_reliability_factors():
    assert fa.load_factor("bending") == 1.0
    assert fa.load_factor("axial") == 0.85
    assert fa.reliability_factor(0.99) == 0.814
    assert fa.reliability_factor(0.50) > fa.reliability_factor(0.999)


def test_steel_endurance_limit_is_half_ultimate_capped():
    raw, note = fa.uncorrected_endurance_limit(STEEL)
    assert raw == pytest.approx(0.5 * STEEL.ultimate_strength)
    assert "0.5 S_ut" in note
    strong = mt.get("300M")
    assert fa.uncorrected_endurance_limit(strong)[0] == 700e6


def test_aluminium_is_flagged_as_having_no_true_endurance_limit():
    raw, note = fa.uncorrected_endurance_limit(ALUMINIUM)
    assert raw == ALUMINIUM.endurance_limit
    assert "no true endurance limit" in note


def test_every_marin_factor_is_reported(): 
    limit = fa.endurance_limit(STEEL, c_to_k(120), 0.020, "machined", "axial", 0.99)
    product = (limit.uncorrected * limit.k_surface * limit.k_size
               * limit.k_load * limit.k_temperature * limit.k_reliability)
    assert limit.value == pytest.approx(product, rel=1e-12)
    assert limit.value < limit.uncorrected


def test_temperature_derating_reaches_the_endurance_limit():
    cool = fa.endurance_limit(ALUMINIUM, c_to_k(20), 0.010)
    hot = fa.endurance_limit(ALUMINIUM, c_to_k(300), 0.010)
    assert hot.value < 0.4 * cool.value


def test_service_temperature_overrun_is_flagged():
    limit = fa.endurance_limit(ALUMINIUM, c_to_k(420), 0.010)
    assert any("service limit" in n for n in limit.notes)


def test_goodman_does_not_penalise_compressive_mean_stress():
    s_e, s_ut = 300e6, 1100e6
    assert fa.goodman_factor(100e6, -200e6, s_e, s_ut) == pytest.approx(3.0)
    assert fa.goodman_factor(100e6, 0.0, s_e, s_ut) == pytest.approx(3.0)
    assert fa.goodman_factor(100e6, 200e6, s_e, s_ut) < 3.0


def test_gerber_is_less_conservative_than_goodman():
    s_e, s_ut = 300e6, 1100e6
    assert (fa.gerber_factor(100e6, 300e6, s_e, s_ut)
            > fa.goodman_factor(100e6, 300e6, s_e, s_ut))


def test_zero_alternating_stress_gives_infinite_factor():
    assert math.isinf(fa.goodman_factor(0.0, 500e6, 300e6, 1100e6))


def test_life_is_infinite_below_the_endurance_limit():
    assert math.isinf(fa.life_cycles(100e6, 0.0, 300e6, 1100e6))


def test_life_falls_as_stress_rises():
    s_e, s_ut = 300e6, 1100e6
    mild = fa.life_cycles(400e6, 0.0, s_e, s_ut)
    harsh = fa.life_cycles(600e6, 0.0, s_e, s_ut)
    assert 1e3 < harsh < mild < 1e7


def test_mean_stress_shortens_life():
    s_e, s_ut = 300e6, 1100e6
    assert (fa.life_cycles(400e6, 300e6, s_e, s_ut)
            < fa.life_cycles(400e6, 0.0, s_e, s_ut))


def test_hours_at_speed_converts_correctly():
    """10^8 cycles at 6000 rpm, once per revolution, is 10^8/360000 hours."""
    from psrt.units import rpm_to_rad_s
    hours = fa.hours_at_speed(1e8, rpm_to_rad_s(6000))
    assert hours == pytest.approx(1e8 / (6000 * 60), rel=1e-9)


# --- preloaded bolts -------------------------------------------------------

def test_separation_factor_matches_the_closed_form():
    r = fa.preloaded_bolt(45_000, 9_500, 58e-6, 0.25, 300e6, 1930e6, 1300e6)
    assert r.separation_factor == pytest.approx(45_000 / (9_500 * 0.75), rel=1e-12)


def test_preload_sets_the_mean_stress():
    r = fa.preloaded_bolt(45_000, 9_500, 58e-6, 0.25, 300e6, 1930e6, 1300e6)
    assert r.preload_stress == pytest.approx(45_000 / 58e-6, rel=1e-12)
    assert r.mean_stress == pytest.approx(
        r.preload_stress + r.alternating_stress, rel=1e-12)


def test_a_softer_joint_puts_more_load_into_the_bolt():
    stiff = fa.preloaded_bolt(45_000, 9_500, 58e-6, 0.15, 300e6, 1930e6, 1300e6)
    soft = fa.preloaded_bolt(45_000, 9_500, 58e-6, 0.45, 300e6, 1930e6, 1300e6)
    assert soft.alternating_stress > stiff.alternating_stress
    assert soft.fatigue_factor < stiff.fatigue_factor


def test_more_preload_helps_separation_and_hurts_yielding():
    low = fa.preloaded_bolt(30_000, 9_500, 58e-6, 0.25, 300e6, 1930e6, 1300e6)
    high = fa.preloaded_bolt(60_000, 9_500, 58e-6, 0.25, 300e6, 1930e6, 1300e6)
    assert high.separation_factor > low.separation_factor
    assert high.yield_factor < low.yield_factor
    assert high.fatigue_factor < low.fatigue_factor
