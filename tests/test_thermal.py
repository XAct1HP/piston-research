"""The thermal map: right bands, and the right trends."""

import pytest

from psrt import loads as loads_mod, thermal as thermal_mod
from psrt.units import k_to_c, rpm_to_rad_s


def _map(state):
    return thermal_mod.compute(state, loads_mod.compute(state))


def test_temperatures_land_where_measurements_put_them(thermal_map):
    """Published piston measurements: crown 280-330 C, ring belt 180-220 C,
    skirt 120-170 C on a naturally aspirated engine at full load."""
    assert 250.0 < k_to_c(thermal_map.crown_top) < 340.0
    assert 160.0 < k_to_c(thermal_map.ring_belt) < 240.0
    assert 110.0 < k_to_c(thermal_map.skirt) < 190.0


def test_temperatures_fall_along_the_heat_path(thermal_map):
    assert (thermal_map.crown_top > thermal_map.crown_underside
            > thermal_map.top_land > thermal_map.ring_belt
            > thermal_map.skirt)


def test_pin_and_rod_run_cooler_than_the_crown(thermal_map, state):
    assert thermal_map.rod < thermal_map.pin < thermal_map.crown_underside
    assert thermal_map.rod > state["thermal.oil_temperature"]


def test_more_speed_means_more_heat(state):
    state.set("operating.speed", rpm_to_rad_s(2000))
    low = _map(state)
    state.set("operating.speed", rpm_to_rad_s(6000))
    high = _map(state)
    assert high.heat_to_piston > low.heat_to_piston
    assert high.crown_top > low.crown_top


def test_a_thicker_crown_runs_hotter_on_its_face(state):
    thin = _map(state)
    state.set("piston.crown_thickness", state["piston.crown_thickness"] * 2.0)
    thick = _map(state)
    assert thick.crown_gradient == pytest.approx(2.0 * thin.crown_gradient,
                                                 rel=1e-9)
    assert thick.crown_top > thin.crown_top


def test_better_oil_cooling_lowers_the_crown(state):
    before = _map(state)
    state.set("thermal.conductance_oil", state["thermal.conductance_oil"] * 3.0)
    after = _map(state)
    assert after.crown_top < before.crown_top


def test_gradient_follows_fourier_conduction(state, thermal_map):
    """dT = q t / (k A), straight from Fourier's law."""
    from psrt import materials
    material = materials.get(state["materials.piston"])
    expected = (thermal_map.crown_heat_flux * state["piston.crown_thickness"]
                / material.thermal_conductivity)
    assert thermal_map.crown_gradient == pytest.approx(expected, rel=1e-12)


def test_the_estimate_declares_itself(thermal_map):
    assert "not a thermal analysis" in thermal_map.note
