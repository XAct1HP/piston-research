"""Section properties against hand calculations."""

import math

import pytest

from psrt import sections


def test_i_beam_against_hand_calculation():
    """30 x 18 mm section, 4.8 mm web, 5.0 mm flanges.

        A    = 18*30 - (18-4.8)(30-10)        = 276 mm^2
        I_xx = (18*30^3 - 13.2*20^3)/12       = 31700 mm^4
        I_yy = (2*5*18^3 + 20*4.8^3)/12       = 5044.3 mm^4
    """
    s = sections.i_beam(0.030, 0.018, 0.0048, 0.0050)
    assert s.area * 1e6 == pytest.approx(276.0, rel=1e-12)
    assert s.i_xx * 1e12 == pytest.approx(31700.0, rel=1e-12)
    assert s.i_yy * 1e12 == pytest.approx(5044.32, rel=1e-9)


def test_i_beam_is_stiffer_in_plane_than_out():
    """The reason rods are I-sections: the axis with less end restraint gets
    more material."""
    s = sections.i_beam(0.030, 0.018, 0.0048, 0.0050)
    assert s.i_xx > s.i_yy
    assert s.r_xx > s.r_yy
    assert s.r_min == s.r_yy


def test_i_beam_degenerates_to_a_rectangle():
    solid = sections.i_beam(0.030, 0.018, 0.018, 0.0001)
    rect = sections.rectangle(0.030, 0.018)
    assert solid.area == pytest.approx(rect.area, rel=1e-12)
    assert solid.i_xx == pytest.approx(rect.i_xx, rel=1e-12)


@pytest.mark.parametrize("height,width,web,flange", [
    (0.030, 0.018, 0.0048, 0.020),   # flanges overlap
    (0.030, 0.018, 0.0250, 0.005),   # web wider than the flange
])
def test_impossible_i_beams_are_refused(height, width, web, flange):
    with pytest.raises(ValueError):
        sections.i_beam(height, width, web, flange)


def test_tube_against_hand_calculation():
    s = sections.tube(0.022, 0.013)
    assert s.area * 1e6 == pytest.approx(
        math.pi * (22.0 ** 2 - 13.0 ** 2) / 4.0, rel=1e-12)
    assert s.i_xx * 1e12 == pytest.approx(
        math.pi * (22.0 ** 4 - 13.0 ** 4) / 64.0, rel=1e-12)
    assert s.z_xx == pytest.approx(s.i_xx / 0.011, rel=1e-12)


def test_solid_bar_is_the_limit_of_a_tube():
    s = sections.tube(0.022, 0.0)
    assert s.i_xx * 1e12 == pytest.approx(math.pi * 22.0 ** 4 / 64.0, rel=1e-12)


def test_tube_with_bore_larger_than_outside_is_refused():
    with pytest.raises(ValueError):
        sections.tube(0.013, 0.022)


def test_bolt_stress_area_matches_the_iso_table():
    """ISO 898: M10x1.5 has a tensile stress area of 58.0 mm^2."""
    assert sections.bolt_stress_area(0.010, 0.0015) * 1e6 == pytest.approx(
        58.0, abs=0.1)
    # M8x1.25 -> 36.6 mm^2
    assert sections.bolt_stress_area(0.008, 0.00125) * 1e6 == pytest.approx(
        36.6, abs=0.1)
    # M12x1.75 -> 84.3 mm^2
    assert sections.bolt_stress_area(0.012, 0.00175) * 1e6 == pytest.approx(
        84.3, abs=0.2)
