"""Hardware validation: does the model agree with a real engine?

The cheapest and most informative check available without a dyno. If the tool
cannot reproduce the published output of a well-documented production engine,
nothing else it says is worth much.

Caveat, stated plainly: the LS3 example's volumetric efficiency and friction
coefficients were calibrated against the same two published points these tests
check, so the performance assertions confirm the calibration held, not that
the breathing model is independently right. The assertions that are NOT
circular are the ones on loads and kinematics -- peak cylinder pressure,
reciprocating mass, the tensile case at overlap TDC -- none of which the
calibration touches, and which are what this tool actually exists to compute.
"""

import os

import pytest

from psrt.evaluate import evaluate, torque_curve
from psrt.schema import load_state
from psrt.units import rpm_to_rad_s, w_to_hp

HERE = os.path.dirname(os.path.abspath(__file__))
LS3 = os.path.join(os.path.dirname(HERE), "examples", "ls3.json")

# Published GM figures for the LS3 in the C6 Corvette.
PUBLISHED_TORQUE_NM = 575.0       # 424 lb-ft
PUBLISHED_TORQUE_RPM = 4600.0
PUBLISHED_POWER_HP = 430.0
PUBLISHED_POWER_RPM = 5900.0


@pytest.fixture(scope="module")
def ls3():
    state, _ = load_state(LS3)
    return state


@pytest.fixture(scope="module")
def ls3_curve(ls3):
    return torque_curve(ls3, range(2000, 6601, 100))


def test_example_file_exists_and_is_valid(ls3):
    assert ls3.validate() == []
    assert ls3["engine.n_cylinders"] == 8


def test_geometry_matches_published_ls3(ls3):
    assert ls3["engine.bore"] == pytest.approx(0.103251, abs=1e-6)
    assert ls3["engine.stroke"] == pytest.approx(0.092, abs=1e-6)
    assert ls3["engine.displacement_total"] * 1e6 == pytest.approx(6162.0, abs=5.0)
    assert ls3["engine.rod_ratio"] == pytest.approx(3.349, abs=0.01)


def test_peak_brake_torque_matches_published(ls3_curve):
    err = (ls3_curve["peak_brake_torque_nm"] - PUBLISHED_TORQUE_NM) / PUBLISHED_TORQUE_NM
    assert abs(err) < 0.05, (
        f"model {ls3_curve['peak_brake_torque_nm']:.0f} N.m vs published "
        f"{PUBLISHED_TORQUE_NM:.0f} N.m ({err * 100:+.1f}%)")


def test_peak_torque_occurs_near_the_published_speed(ls3_curve):
    assert abs(ls3_curve["peak_brake_torque_rpm"] - PUBLISHED_TORQUE_RPM) < 600.0


def test_brake_power_matches_published_at_rated_speed(ls3_curve):
    i = ls3_curve["rpm"].index(PUBLISHED_POWER_RPM)
    hp = w_to_hp(ls3_curve["brake_power_w"][i])
    err = (hp - PUBLISHED_POWER_HP) / PUBLISHED_POWER_HP
    assert abs(err) < 0.05, (
        f"model {hp:.0f} hp vs published {PUBLISHED_POWER_HP:.0f} hp "
        f"({err * 100:+.1f}%)")


# --- The non-circular checks ----------------------------------------------

def test_peak_cylinder_pressure_is_plausible_for_a_na_engine(ls3):
    """Not touched by the VE or FMEP calibration. A naturally aspirated SI
    engine at 10.7:1 runs somewhere around 50 to 80 bar peak."""
    m = evaluate(ls3)
    bar = m.combustion["peak_pressure_pa"] / 1e5
    assert 45.0 < bar < 85.0, f"{bar:.1f} bar is not plausible"


def test_mean_piston_speed_at_redline_is_production_typical(ls3):
    probe = ls3.copy()
    probe.set("operating.speed", ls3["engine.redline"])
    mps = probe["engine.mean_piston_speed"]
    assert 17.0 < mps < 22.0, (
        f"{mps:.1f} m/s; production engines land near 20 m/s at redline")


def test_rod_tension_at_redline_is_survivable_for_a_stock_rod(ls3):
    """The LS3's powdered-metal rod is known to be the weak link above roughly
    6600 rpm. The computed tensile load should be large but not absurd: tens
    of kilonewtons, not hundreds.
    """
    probe = ls3.copy()
    probe.set("operating.speed", ls3["engine.redline"])
    tension = abs(evaluate(probe).loads["overlap_tdc_rod_tension_n"])
    assert 15_000.0 < tension < 40_000.0, f"{tension / 1000:.1f} kN"


def test_energy_closes_on_the_real_engine(ls3):
    assert evaluate(ls3).checks["energy_closure_pass"]


def test_estimated_inputs_are_declared_not_hidden(ls3):
    """Masses are estimates. The tool must say so rather than imply they are
    measurements -- every margin computed in phase 2 will rest on them."""
    assert ls3.param("masses.piston").source == "estimated"
    assert ls3.param("engine.bore").source == "published"
    assert "estimates" in ls3.meta["provenance"]


def test_bore_headroom_is_tight_on_this_block(ls3):
    """LS blocks are famously bore-limited: 4.400 in spacing around a 4.065 in
    bore leaves very little wall. The envelope model should reflect that."""
    headroom = ls3.param("engine.bore").maximum - ls3["engine.bore"]
    assert 0.0 < headroom < 0.0015, f"{headroom * 1000:.2f} mm of overbore"
