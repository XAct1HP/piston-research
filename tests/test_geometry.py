"""The geometry kernel: closed forms, conventions, and the exactness check.

Geometry rebuilds cost a few hundred milliseconds each, so fixtures here are
module-scoped and the cache does most of the work.
"""

import math
import os

import cadquery as cq
import pytest

from psrt import geometry as geo
from psrt import materials as mt
from psrt import sections
from psrt.geometry import build as build_mod
from psrt.geometry import profiles
from psrt.geometry.properties import combine, measure
from psrt.schema import default_state, load_state

HERE = os.path.dirname(os.path.abspath(__file__))
LS3 = os.path.join(os.path.dirname(HERE), "examples", "ls3.json")
STEEL = mt.get("4340")


# --- the inertia convention, which this module got wrong first -------------

def test_occ_inertia_is_centroidal_not_about_the_origin():
    """Pinning down the convention. OpenCascade's MatrixOfInertia does not
    move when the solid moves, so it is centroidal; the tensor about the
    origin has to be built with the parallel axis theorem."""
    here = measure(cq.Workplane("XY").box(20, 10, 5), STEEL)
    there = measure(cq.Workplane("XY").box(20, 10, 5).translate((50, 0, 0)), STEEL)

    assert here.inertia_centroidal == pytest.approx(there.inertia_centroidal)
    assert here.inertia_origin[1][1] != pytest.approx(there.inertia_origin[1][1])


def test_box_mass_properties_against_closed_forms():
    p = measure(cq.Workplane("XY").box(20, 10, 5).translate((50, 0, 0)), STEEL)
    assert p.volume == pytest.approx(1000e-9, rel=1e-12)
    assert p.mass == pytest.approx(1000e-9 * STEEL.density, rel=1e-12)
    assert p.centre_of_mass[0] == pytest.approx(0.050, rel=1e-9)

    assert p.inertia_centroidal[0][0] == pytest.approx(
        p.mass * (0.010 ** 2 + 0.005 ** 2) / 12.0, rel=1e-9)
    assert p.inertia_centroidal[1][1] == pytest.approx(
        p.mass * (0.020 ** 2 + 0.005 ** 2) / 12.0, rel=1e-9)


def test_parallel_axis_shift_is_applied_to_the_origin_tensor():
    p = measure(cq.Workplane("XY").box(20, 10, 5).translate((50, 0, 0)), STEEL)
    assert p.inertia_origin[1][1] == pytest.approx(
        p.inertia_centroidal[1][1] + p.mass * 0.050 ** 2, rel=1e-9)


def test_combining_two_parts_conserves_mass_and_places_the_centroid():
    a = measure(cq.Workplane("XY").box(10, 10, 10), STEEL)
    b = measure(cq.Workplane("XY").box(10, 10, 10).translate((20, 0, 0)), STEEL)
    both = combine([a, b])
    assert both.mass == pytest.approx(a.mass + b.mass, rel=1e-12)
    assert both.centre_of_mass[0] == pytest.approx(0.010, rel=1e-9)


# --- the parts -------------------------------------------------------------

@pytest.fixture(scope="module")
def state():
    return default_state("geometry test")


@pytest.fixture(scope="module")
def parts(state):
    return geo.build_all(state)


def test_pin_volume_matches_the_tube_formula(state, parts):
    expected = sections.tube(state["pin.outer_diameter"],
                             state["pin.inner_diameter"]).area * state["pin.length"]
    assert parts["pin"]["properties"].volume == pytest.approx(expected, rel=1e-9)


def test_pin_is_balanced_on_its_own_centre(parts):
    for axis in parts["pin"]["properties"].centre_of_mass:
        assert abs(axis) < 1e-9


def test_every_part_is_a_solid_with_positive_volume(parts):
    for name, entry in parts.items():
        assert entry["properties"].volume > 0.0, name
        assert entry["properties"].mass > 0.0, name


def test_rod_centre_of_mass_reads_from_the_small_end(state, parts):
    """The rod is built with its small end at z = 0, so the centre of mass
    reads straight off as the distance the two-mass split needs."""
    cg = parts["rod"]["properties"].centre_of_mass[2]
    assert 0.5 < cg / state["engine.rod_length"] < 0.85


def test_i_outline_encloses_the_same_area_as_the_section_formula():
    """The lofted shank profile and the closed-form section properties have to
    describe the same shape, or the mass model and the stress model disagree."""
    outline = build_mod._i_outline(30.0, 18.0, 4.8, 5.0)
    wire = cq.Workplane("XY").polyline(outline).close()
    area = wire.extrude(1.0).val().Volume()          # unit depth -> area
    assert area == pytest.approx(
        sections.i_beam(0.030, 0.018, 0.0048, 0.0050).area * 1e6, rel=1e-9)


def test_the_taper_puts_more_section_at_the_big_end():
    """Tests the taper geometry directly. Whether the ROD's centre of mass
    moves up or down depends on where the shank's centroid sits relative to
    it, so this checks the shank on its own."""
    small = build_mod._i_outline(20.0, 14.0, 4.0, 4.0)
    big = build_mod._i_outline(30.0, 21.0, 6.0, 6.0)
    shank = (cq.Workplane("XY").polyline(small).close()
             .workplane(offset=100.0).polyline(big).close().loft(ruled=True))
    centroid = measure(shank, STEEL).centre_of_mass[2]
    assert centroid > 0.050, "a tapered shank balances past its own mid-span"


def test_taper_adds_mass(state):
    straight = geo.masses_from_geometry(
        state.with_changes({"rod.shank_taper": 1.0}, "test", "prismatic"))
    tapered = geo.masses_from_geometry(
        state.with_changes({"rod.shank_taper": 1.5}, "test", "tapered"))
    assert tapered["rod_total_kg"] > straight["rod_total_kg"]


def test_two_mass_split_conserves_rod_mass(state):
    measured = geo.masses_from_geometry(state)
    fraction = measured["rod_cg_fraction_of_length"]
    total = measured["rod_total_kg"]
    assert total * fraction + total * (1.0 - fraction) == pytest.approx(
        total, rel=1e-12)


def test_a_thicker_crown_makes_a_heavier_piston(state):
    light = geo.masses_from_geometry(state)["piston_kg"]
    heavy = geo.masses_from_geometry(state.with_changes(
        {"piston.crown_thickness": state["piston.crown_thickness"] * 1.6},
        "test", "thicker crown"))["piston_kg"]
    assert heavy > light


def test_dishing_the_crown_removes_the_volume_asked_for(state):
    plain = geo.build_all(state)["piston"]["properties"].volume
    dish = 6.0e-6          # 6 cc
    dished = geo.build_all(state.with_changes(
        {"piston.crown_dish_volume": dish}, "test", "6 cc dish")
    )["piston"]["properties"].volume
    assert plain - dished == pytest.approx(dish, rel=1e-6)


def test_boring_the_pin_out_lightens_it(state):
    heavy = geo.masses_from_geometry(state)["pin_kg"]
    light = geo.masses_from_geometry(state.with_changes(
        {"pin.inner_diameter": state["pin.inner_diameter"] * 1.3},
        "test", "bigger pin bore"))["pin_kg"]
    assert light < heavy


# --- impossible geometry is refused, not silently built --------------------

def test_a_dish_deeper_than_the_crown_is_refused(state):
    with pytest.raises(ValueError, match="dish"):
        build_mod.build_piston(state.with_changes(
            {"piston.crown_dish_volume": 200.0e-6}, "test", "absurd dish"))


def test_a_crown_thicker_than_the_piston_is_refused(state):
    bad = state.copy()
    bad.param("piston.crown_thickness").value = bad["piston.total_height"] * 1.2
    with pytest.raises(ValueError, match="crown thickness"):
        build_mod.build_piston(bad)


def test_a_pin_bore_below_the_piston_is_refused(state):
    with pytest.raises(ValueError, match="pin bore"):
        build_mod.build_piston(state.with_changes(
            {"piston.compression_height": state["piston.total_height"]},
            "test", "pin below the skirt"))


def test_a_small_end_bore_larger_than_its_outside_is_refused(state):
    with pytest.raises(ValueError, match="small-end bore"):
        build_mod.build_rod(state.with_changes(
            {"rod.small_end_outer_diameter": state["pin.outer_diameter"] * 0.9},
            "test", "impossible small end"))


# --- the cache -------------------------------------------------------------

def test_cache_key_ignores_non_geometric_parameters(state):
    before = geo.fingerprint(state)
    moved = state.with_changes({"operating.speed": 123.0}, "test", "new speed")
    assert geo.fingerprint(moved) == before


def test_cache_key_follows_geometric_parameters(state):
    before = geo.fingerprint(state)
    thicker = state.with_changes(
        {"piston.crown_thickness": state["piston.crown_thickness"] + 1e-4},
        "test", "thicker")
    assert geo.fingerprint(thicker) != before


# --- the exactness check ---------------------------------------------------

def test_the_bundled_examples_are_geometrically_self_consistent():
    for path in (LS3, os.path.join(os.path.dirname(HERE), "examples",
                                   "generic-i4.json")):
        state, _ = load_state(path)
        check = geo.check_masses(state)
        assert check["passes"], (path, check["rows"])


def test_the_check_catches_a_mass_that_has_drifted(state):
    drifted = state.with_changes(
        {"masses.piston": state["masses.piston"] * 0.6}, "test", "wrong mass")
    check = geo.check_masses(drifted)
    assert not check["passes"]
    assert any(not row["within_tolerance"] for row in check["rows"])


def test_the_check_flags_reciprocating_mass_specifically(state):
    drifted = state.with_changes(
        {"masses.rod_cg_from_small_end": state["engine.rod_length"] * 0.4},
        "test", "wrong balance point")
    check = geo.check_masses(drifted)
    assert abs(check["reciprocating_relative_error"]) > 0.03
    assert any("reciprocating" in n for n in check["notes"])


def test_adopting_the_geometry_makes_the_check_pass(state):
    drifted = state.with_changes(
        {"masses.piston": state["masses.piston"] * 0.6}, "test", "wrong mass")
    assert geo.check_masses(geo.adopt_masses(drifted))["passes"]


def test_adopting_records_who_changed_what(state):
    drifted = state.with_changes(
        {"masses.pin": 0.05}, "test", "wrong mass")
    adopted = geo.adopt_masses(drifted)
    assert adopted.log[-1].actor == "geometry"
    assert "measured" in adopted.log[-1].rationale


def test_the_check_admits_the_solids_run_heavy(state):
    assert any("runs slightly heavy" in n
               for n in geo.check_masses(state)["notes"])


# --- bore profiles: the hexagon question -----------------------------------

def test_profile_area_factors_match_closed_forms():
    assert profiles.circle().area_factor == pytest.approx(math.pi)
    assert profiles.polygon(6).area_factor == pytest.approx(
        3.0 * math.sqrt(3.0) / 2.0, rel=1e-12)
    assert profiles.polygon(8).area_factor == pytest.approx(
        2.0 * math.sqrt(2.0), rel=1e-12)
    assert profiles.polygon(4).area_factor == pytest.approx(2.0, rel=1e-12)


def test_the_circle_encloses_the_most_area_for_a_given_envelope():
    """The whole answer to the hexagonal-bore question, in one assertion."""
    rows = profiles.compare(0.05)
    assert rows[0]["key"] == "circle"
    assert all(r["fraction_of_circle"] <= 1.0 for r in rows)


def test_a_hexagon_loses_about_a_sixth_of_the_area():
    rows = {r["key"]: r for r in profiles.compare(0.05)}
    assert rows["polygon-6"]["fraction_of_circle"] == pytest.approx(0.827, abs=1e-3)


def test_polygons_approach_the_circle_as_sides_increase():
    factors = [profiles.polygon(n).area_factor for n in (4, 6, 8, 12, 24)]
    assert factors == sorted(factors)
    assert factors[-1] < math.pi


def test_only_the_circle_is_marked_sealable():
    for row in profiles.compare(0.05):
        assert row["sealable"] is (row["key"] == "circle")


def test_unknown_profile_is_refused():
    with pytest.raises(KeyError):
        profiles.get("trapezoid")


def test_profile_study_uses_the_block_envelope(state):
    study = geo.profile_study(state)
    assert study["envelope_diameter_m"] == state.param("engine.bore").maximum
    circle = next(r for r in study["rows"] if r["key"] == "circle")
    assert circle["displacement_total_m3"] == pytest.approx(
        circle["area_m2"] * state["engine.stroke"] * state["engine.n_cylinders"],
        rel=1e-12)


def test_a_hexagonal_bore_actually_builds(state):
    """The tool should let you try it and get real geometry, not a refusal."""
    hexed = state.with_changes({"block.bore_profile": "polygon-6"},
                               "test", "hexagonal bore")
    piston = geo.build_all(hexed)["piston"]["properties"]
    round_piston = geo.build_all(state)["piston"]["properties"]
    assert piston.volume < round_piston.volume


# --- export ----------------------------------------------------------------

def test_step_export_writes_readable_files(state, tmp_path):
    written = geo.export(state, str(tmp_path), fmt="step")
    assert len(written) == 4
    for path in written:
        assert os.path.getsize(path) > 1000
        with open(path, encoding="utf-8", errors="ignore") as fh:
            assert "ISO-10303" in fh.read(200)


def test_stl_export_writes_files(state, tmp_path):
    written = geo.export(state, str(tmp_path), fmt="stl")
    assert all(os.path.getsize(p) > 1000 for p in written)


def test_tessellation_returns_usable_triangles(state):
    mesh = geo.tessellate(state, tolerance=0.5)
    assert set(mesh) == {"piston", "pin", "rod", "sleeve"}
    for part, data in mesh.items():
        assert data["triangle_count"] > 0, part
        assert len(data["vertices"][0]) == 3
        assert max(max(t) for t in data["triangles"]) < data["vertex_count"]


# --- invalid geometry must fail where the mistake is -----------------------

def _ls3():
    from psrt.schema import load_state
    return load_state("examples/ls3.json")[0]


@pytest.mark.parametrize("path,value,word", [
    ("rod.shank_flange", 0.020, "flanges"),
    ("rod.shank_height", 0.010, "flanges"),
    ("rod.shank_web", 0.020, "web"),
])
def test_an_impossible_i_section_is_refused_by_name(path, value, word):
    """An impossible section still produces an outline -- one that crosses
    itself. OpenCascade extrudes that into a shape of zero or NEGATIVE
    volume, which nothing rejected: it became a zero mass, then a meaningless
    inertia tensor, and finally 'Eigenvalues did not converge' thrown from
    the principal-moment decomposition, which names neither the part nor the
    parameter. The refusal has to say which dimension is wrong."""
    from psrt.geometry import build_all

    state = _ls3().with_changes({path: value}, actor="test",
                                rationale="impossible section")
    with pytest.raises(ValueError, match=word):
        build_all(state)


def test_a_solid_with_no_volume_is_refused_at_the_measurement():
    """The guard above catches the rod. This one catches every other part,
    because a boolean that goes wrong returns a shape rather than raising."""
    import numpy as np

    from psrt import materials as materials_mod
    from psrt.geometry.properties import MassProperties, measure

    class Empty:
        pass

    # Measured directly: a degenerate tensor must not reach eigvalsh.
    bad = MassProperties(
        volume=0.0, mass=0.0, density=7800.0, centre_of_mass=(0.0, 0.0, 0.0),
        inertia_origin=np.full((3, 3), np.nan),
        inertia_centroidal=np.full((3, 3), np.nan))
    with pytest.raises(ValueError, match="not finite"):
        bad.principal_moments
