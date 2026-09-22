"""Kinematics: analytic derivatives against numerical, and against closed forms."""

import math

import numpy as np
import pytest

from psrt import kinematics as kin
from tests.conftest import d1, d2

ANGLES = [-3.0, -1.7, -0.9, -0.3, 0.0, 0.4, 1.1, 2.2, math.pi, 4.0, 5.5]


def test_rod_must_be_longer_than_crank():
    with pytest.raises(ValueError):
        kin.CrankGeometry(crank_radius=0.05, rod_length=0.04,
                          bore_area=1e-3, clearance_volume=1e-5)


def test_displacement_is_zero_at_tdc_and_stroke_at_bdc(geom):
    assert kin.displacement(geom, 0.0) == pytest.approx(0.0, abs=1e-15)
    assert kin.displacement(geom, math.pi) == pytest.approx(geom.stroke, rel=1e-14)


def test_first_derivative_matches_numerical(geom):
    for th in ANGLES:
        analytic = kin.d_displacement_d_theta(geom, th)
        numeric = d1(lambda x: float(kin.displacement(geom, x)), th)
        assert analytic == pytest.approx(numeric, rel=1e-8, abs=1e-12)


def test_second_derivative_matches_numerical(geom):
    for th in ANGLES:
        analytic = kin.d2_displacement_d_theta2(geom, th)
        numeric = d2(lambda x: float(kin.displacement(geom, x)), th)
        assert analytic == pytest.approx(numeric, rel=1e-6, abs=1e-9)


def test_velocity_is_zero_at_both_reversals(geom):
    omega = 600.0
    assert kin.velocity(geom, 0.0, omega) == pytest.approx(0.0, abs=1e-12)
    assert kin.velocity(geom, math.pi, omega) == pytest.approx(0.0, abs=1e-12)
    assert kin.velocity(geom, 2 * math.pi, omega) == pytest.approx(0.0, abs=1e-12)


def test_acceleration_at_dead_centres_matches_closed_form(geom):
    omega = 733.0
    assert kin.acceleration(geom, 0.0, omega) == pytest.approx(
        kin.acceleration_at_tdc(geom, omega), rel=1e-13)
    assert kin.acceleration(geom, math.pi, omega) == pytest.approx(
        kin.acceleration_at_bdc(geom, omega), rel=1e-13)


def test_tdc_acceleration_exceeds_bdc_because_rod_is_finite(geom):
    """The asymmetry that makes overlap TDC the governing tensile case."""
    omega = 733.0
    a_tdc = abs(kin.acceleration(geom, 0.0, omega))
    a_bdc = abs(kin.acceleration(geom, math.pi, omega))
    assert a_tdc > a_bdc
    assert a_tdc / a_bdc == pytest.approx(
        (1 + geom.lam) / (1 - geom.lam), rel=1e-12)


def test_not_using_the_simple_harmonic_approximation(geom):
    """Guard against someone 'simplifying' this back to a sine wave.

    The second-order approximation a = -w^2 r (cos th + lam cos 2th) is a good
    approximation, so the test asserts the exact solution differs from it by a
    small but unmistakable amount away from the dead centres.
    """
    omega = 733.0
    r, lam = geom.crank_radius, geom.lam
    th = 1.2
    approx = omega ** 2 * r * (math.cos(th) + lam * math.cos(2 * th))
    exact = kin.acceleration(geom, th, omega)
    rel = abs(exact - approx) / abs(approx)
    assert 1e-4 < rel < 5e-2


def test_rod_angle(geom):
    assert kin.rod_angle(geom, 0.0) == pytest.approx(0.0, abs=1e-15)
    assert kin.rod_angle(geom, math.pi) == pytest.approx(0.0, abs=1e-15)
    assert kin.rod_angle(geom, math.pi / 2) == pytest.approx(
        kin.max_rod_angle(geom), rel=1e-14)


def test_volume_reproduces_the_compression_ratio(geom, state):
    v_bdc = kin.cylinder_volume(geom, math.pi)
    v_tdc = kin.cylinder_volume(geom, 0.0)
    assert v_bdc / v_tdc == pytest.approx(
        state["engine.compression_ratio"], rel=1e-12)


def test_volume_derivative_matches_numerical(geom):
    for th in ANGLES:
        analytic = kin.d_volume_d_theta(geom, th)
        numeric = d1(lambda x: float(kin.cylinder_volume(geom, x)), th)
        assert analytic == pytest.approx(numeric, rel=1e-8, abs=1e-14)


def test_sweep_endpoints_and_spacing():
    th = kin.sweep(-360.0, 360.0, 0.25)
    assert len(th) == 2881
    assert math.degrees(th[0]) == pytest.approx(-360.0)
    assert math.degrees(th[-1]) == pytest.approx(360.0)
    assert np.allclose(np.diff(np.degrees(th)), 0.25)
