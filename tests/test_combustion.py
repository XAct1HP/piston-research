"""Combustion: polytropic limit, Wiebe behaviour, and the measured-trace path."""

import math

import numpy as np
import pytest

from psrt import combustion, kinematics as kin
from psrt.schema import default_state


def test_motored_cylinder_is_polytropic(state):
    """With no fuel the first law reduces to p V^gamma = constant.

    This is the single strongest check on the RK4 integration: any error in
    the ODE, the volume derivative or the step loop breaks it immediately.
    """
    state.set("operating.fuel_lhv", 0.0)
    tr = combustion.simulate(state)
    gamma = state["operating.gamma"]
    ivc, evo = state["operating.ivc_angle"], state["operating.evo_angle"]

    closed = (tr.theta >= ivc) & (tr.theta <= evo)
    invariant = tr.pressure[closed] * tr.volume[closed] ** gamma
    assert np.std(invariant) / np.mean(invariant) < 1e-6


def test_motored_tdc_pressure_matches_hand_calculation(state):
    state.set("operating.fuel_lhv", 0.0)
    tr = combustion.simulate(state)
    geom = kin.CrankGeometry.from_state(state)
    ivc = state["operating.ivc_angle"]

    v_ivc = float(kin.cylinder_volume(geom, ivc))
    p_ivc = state["operating.trapped_pressure"]
    expected = p_ivc * (v_ivc / state["engine.clearance_volume"]) ** state["operating.gamma"]

    at_tdc = tr.pressure[int(np.argmin(np.abs(tr.theta)))]
    assert at_tdc == pytest.approx(expected, rel=2e-4)


def test_wiebe_runs_from_zero_to_essentially_one():
    soc, dur = math.radians(-15.0), math.radians(50.0)
    assert combustion.wiebe_mfb(soc - 0.1, soc, dur, 6.908, 2.0) == pytest.approx(0.0)
    assert combustion.wiebe_mfb(soc, soc, dur, 6.908, 2.0) == pytest.approx(0.0)
    assert combustion.wiebe_mfb(soc + dur, soc, dur, 6.908, 2.0) == pytest.approx(
        0.999, abs=1e-3)


def test_wiebe_burn_rate_integrates_to_the_mass_fraction():
    soc, dur = math.radians(-15.0), math.radians(50.0)
    th = np.linspace(soc, soc + dur, 20001)
    rate = combustion.wiebe_burn_rate(th, soc, dur, 6.908, 2.0)
    assert np.trapezoid(rate, th) == pytest.approx(
        float(combustion.wiebe_mfb(soc + dur, soc, dur, 6.908, 2.0)), rel=1e-6)


def test_peak_pressure_lands_in_a_plausible_window(state):
    tr = combustion.simulate(state)
    deg = math.degrees(tr.peak_pressure_angle)
    assert 5.0 < deg < 30.0, "healthy SI combustion peaks 12-18 deg ATDC"
    assert 2.0e6 < tr.peak_pressure < 1.5e7


def test_more_advance_raises_peak_pressure_and_moves_it_earlier(state):
    late = combustion.simulate(state)
    state.set("operating.soc_angle", math.radians(-28.0))
    early = combustion.simulate(state)
    assert early.peak_pressure > late.peak_pressure
    assert early.peak_pressure_angle < late.peak_pressure_angle


def test_trapped_mass_scales_with_volumetric_efficiency(state):
    base = state["operating.trapped_mass"]
    state.set("operating.ve_peak", state["operating.ve_peak"] * 1.2)
    assert state["operating.trapped_mass"] == pytest.approx(base * 1.2, rel=1e-12)


def test_higher_compression_raises_peak_pressure_and_imep(state):
    lo = combustion.simulate(state)
    vd = state["engine.displacement_cyl"]
    imep_lo = lo.imep_gross(vd)

    state.set("engine.compression_ratio", 12.5)
    hi = combustion.simulate(state)
    assert hi.peak_pressure > lo.peak_pressure
    assert hi.imep_gross(state["engine.displacement_cyl"]) > imep_lo


def test_measured_trace_round_trips_through_csv(state, tmp_path):
    original = combustion.simulate(state)
    path = tmp_path / "trace.csv"
    np.savetxt(path, np.column_stack([np.degrees(original.theta),
                                      original.pressure / 1e5]),
               delimiter=",")

    loaded = combustion.from_csv(state, str(path))
    assert loaded.source == "measured"
    assert np.allclose(loaded.pressure, original.pressure, rtol=1e-6, atol=10.0)


def test_measured_trace_drives_the_same_load_chain(state, tmp_path):
    from psrt import loads

    original = combustion.simulate(state)
    path = tmp_path / "trace.csv"
    np.savetxt(path, np.column_stack([np.degrees(original.theta),
                                      original.pressure / 1e5]), delimiter=",")

    a = loads.compute(state, original)
    b = loads.compute(state, combustion.from_csv(state, str(path)))
    assert b.peak_pin_compression.value == pytest.approx(
        a.peak_pin_compression.value, rel=1e-4)
