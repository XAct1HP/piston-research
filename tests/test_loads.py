"""The load chain: energy conservation, closed-form checks, and sanity bands."""

import math

import numpy as np
import pytest

from psrt import kinematics as kin, loads
from psrt.units import rpm_to_rad_s


def test_energy_closes_between_torque_and_imep(state):
    """Integrated instantaneous torque must equal IMEP_net x displacement.

    This is the check that ties the thermodynamics and the mechanics together.
    If the force resolution, the rod obliquity or the torque expression were
    wrong, this would not close.
    """
    sweep = loads.compute(state)
    from_torque = sweep.cycle_work()
    from_imep = sweep.imep_net() * sweep.displacement
    assert from_torque == pytest.approx(from_imep, rel=1e-3)


@pytest.mark.parametrize("rpm", [1200, 3000, 5500, 7800])
def test_energy_closes_at_every_speed(state, rpm):
    state.set("operating.speed", rpm_to_rad_s(rpm))
    sweep = loads.compute(state)
    assert sweep.cycle_work() == pytest.approx(
        sweep.imep_net() * sweep.displacement, rel=1e-3)


def test_torque_is_zero_at_both_dead_centres(state):
    sweep = loads.compute(state)
    scale = np.max(np.abs(sweep.torque))
    for deg in (-360.0, -180.0, 0.0, 180.0, 360.0):
        assert abs(sweep.at_angle(deg)["torque"]) < scale * 1e-6


def test_side_thrust_is_zero_at_both_dead_centres(state):
    sweep = loads.compute(state)
    scale = np.max(np.abs(sweep.f_side))
    for deg in (-180.0, 0.0, 180.0, 360.0):
        assert abs(sweep.at_angle(deg)["f_side"]) < scale * 1e-6


def test_overlap_tdc_load_matches_the_closed_form(state):
    """At overlap TDC the inertia term is exactly -m_recip * omega^2 * r * (1 + r/l).

    The pin load equals that plus whatever small gas force remains. With the
    crankcase held at exhaust pressure the residual is only the tail of the
    blowdown decay -- a fraction of a newton against nearly ten kilonewtons --
    so the test checks the inertia term exactly and the total loosely.
    """
    state.set("operating.crankcase_pressure", state["operating.exhaust_pressure"])
    sweep = loads.compute(state)
    geom = kin.CrankGeometry.from_state(state)
    at_tdc = sweep.at_overlap_tdc()

    expected_inertia = -(state["masses.reciprocating"]
                         * kin.acceleration_at_tdc(geom, state["operating.speed"]))
    assert at_tdc["f_inertia"] == pytest.approx(expected_inertia, rel=1e-12)
    assert at_tdc["f_pin"] == pytest.approx(
        at_tdc["f_gas"] + at_tdc["f_inertia"], rel=1e-12)
    assert at_tdc["f_pin"] == pytest.approx(expected_inertia, rel=1e-3)
    assert at_tdc["f_pin"] < 0, "the rod is in tension at overlap TDC"


def test_peak_tension_occurs_at_overlap_tdc(state):
    state.set("operating.speed", rpm_to_rad_s(7000))
    sweep = loads.compute(state)
    assert abs(sweep.peak_pin_tension.angle_deg) == pytest.approx(360.0, abs=1.0)


def test_inertia_force_scales_with_speed_squared(state):
    state.set("operating.speed", rpm_to_rad_s(3000))
    low = loads.compute(state).at_overlap_tdc()["f_inertia"]
    state.set("operating.speed", rpm_to_rad_s(6000))
    high = loads.compute(state).at_overlap_tdc()["f_inertia"]
    assert high == pytest.approx(4.0 * low, rel=1e-9)


def test_tensile_load_grows_while_compressive_load_shrinks_with_speed(state):
    """The effect that decides how an engine is stressed at high rpm."""
    state.set("operating.speed", rpm_to_rad_s(3000))
    low = loads.compute(state)
    state.set("operating.speed", rpm_to_rad_s(7500))
    high = loads.compute(state)

    assert abs(high.peak_pin_tension.value) > abs(low.peak_pin_tension.value)
    assert high.peak_pin_compression.value < low.peak_pin_compression.value


def test_rod_force_never_below_pin_force_in_magnitude(state):
    """F_rod = F_pin / cos(phi), and cos(phi) <= 1."""
    sweep = loads.compute(state)
    assert np.all(np.abs(sweep.f_rod) >= np.abs(sweep.f_pin) - 1e-9)


def test_lighter_piston_relieves_the_rod(state):
    """The coupling that makes reciprocating mass the most productive lever."""
    state.set("operating.speed", rpm_to_rad_s(7000))
    heavy = abs(loads.compute(state).peak_pin_tension.value)
    state.set("masses.piston", state["masses.piston"] - 0.040)
    light = abs(loads.compute(state).peak_pin_tension.value)
    assert light < heavy
    assert (heavy - light) > 1000.0, "40 g should be worth more than a kN"


def test_gas_force_uses_the_pressure_differential(state):
    sweep = loads.compute(state)
    at_peak = sweep.at_peak_pressure()
    area = state["engine.bore_area"]
    expected = (at_peak["pressure"] - state["operating.crankcase_pressure"]) * area
    assert at_peak["f_gas"] == pytest.approx(expected, rel=1e-12)


def test_indicated_output_lands_in_a_plausible_band(state):
    """Coarse hardware sanity: a naturally aspirated SI engine makes roughly
    90 to 130 N.m per litre of indicated torque near its peak."""
    state.set("operating.speed", rpm_to_rad_s(4500))
    sweep = loads.compute(state)
    per_litre = sweep.indicated_torque() / (
        state["engine.displacement_total"] * 1e3)
    assert 80.0 < per_litre < 140.0, f"{per_litre:.1f} N.m/L is not plausible"
