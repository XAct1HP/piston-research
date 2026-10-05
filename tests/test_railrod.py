"""The rail connecting rod: geometry, swing, contact network, FEA coupling.

The checks that matter most are the geometric ones, because the concept's
whole promise rests on them: the notch is radiused and not stepped, the clamp
knob and flank are the notch offset by the clearance, the clamp can actually
be hooked in and swung shut, and each part comes out as its own valid solid.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from psrt.schema import load_state, refresh_bounds
from psrt.state import ConstraintViolation

pytest.importorskip("shapely")


@pytest.fixture()
def state():
    s, _ = load_state("examples/ls3.json")
    s.set("railrod.enabled", True)
    refresh_bounds(s)
    return s


@pytest.fixture()
def layout(state):
    from psrt.railrod.layout import build_layout
    return build_layout(state)


# --- the notch ----------------------------------------------------------------

def test_notch_is_deepest_at_the_top_and_returns_to_full_width(layout):
    n = layout.notch
    x_o = layout.x_o
    # A is on the rail face, D is the deepest point, directly beside the
    # top radius's centre
    assert n.A[0] == pytest.approx(x_o)
    assert n.D[0] == pytest.approx(x_o - n.depth)
    assert n.D[1] == pytest.approx(n.centre[1])
    # travelling down: A above D above T above F1 above F2 above the foot
    assert n.A[1] > n.D[1] > n.T[1] > n.F1[1] > n.F2[1] > layout.v_e
    # F2 is back on the full-width face, leaving a land
    assert n.F2[0] == pytest.approx(x_o)
    assert n.F2[1] - layout.v_e > 0.2


def test_the_top_transition_is_a_tangent_radius_not_a_corner(layout):
    n = layout.notch
    assert np.linalg.norm(n.T - n.centre) == pytest.approx(n.r_top)
    ramp = n.F1 - n.T
    # the ramp is tangent to the top radius at T ...
    assert abs(ramp @ (n.T - n.centre)) < 1e-9 * np.linalg.norm(ramp)
    # ... and to the lower blend at F1
    assert abs(ramp @ (n.F1 - n.blend_centre)) < 1e-9 * np.linalg.norm(ramp)
    # the ramp runs at the specified angle from the rail axis
    angle = math.degrees(math.atan2(abs(ramp[0]), abs(ramp[1])))
    assert angle == pytest.approx(math.degrees(n.alpha))


def test_a_radius_smaller_than_the_depth_is_refused(state):
    depth = state["railrod.notch_depth"]
    with pytest.raises(ConstraintViolation):
        state.set("railrod.notch_top_radius", depth * 0.8)


def test_notch_never_protrudes_outside_the_rail_envelope(layout):
    minx, _, maxx, _ = layout.rail_right.bounds
    assert maxx <= layout.x_o + 1e-9
    assert minx >= layout.x_i - 1e-9


# --- the clamp fits the notch -------------------------------------------------

def test_knob_and_flank_are_the_notch_offset_by_the_clearance(layout):
    from psrt.railrod.layout import conformity
    c = conformity(layout)
    assert c["conforms"], c
    assert c["gap_min_mm"] == pytest.approx(layout.tip_clearance, abs=5e-3)


def test_clamp_clears_the_receiver_by_the_guide_clearance(layout):
    from psrt.railrod.layout import guide_gap
    gap = guide_gap(layout)
    assert gap["min_gap_mm"] == pytest.approx(layout.guide_clearance,
                                              abs=0.01)


def test_the_clamps_are_mirror_images(layout):
    from psrt.railrod.layout import mirror
    assert mirror(layout.clamp_right).symmetric_difference(
        layout.clamp_left).area < 1e-6


# --- the swing ----------------------------------------------------------------

def test_clamp_can_be_hooked_in_and_swung_shut(layout):
    from psrt.railrod.layout import swing_check
    swing = swing_check(layout)
    assert swing["assemblable"], swing
    assert swing["hook_in_angle_deg"] <= swing["free_opening_deg"]
    assert not swing["closed_interference"]


def test_the_pivot_is_on_none_of_the_parts(layout):
    """The swing centre is defined by the channel's radius and by nothing
    else, so it must not land on any solid. If it ever does, some part has
    quietly become the hinge and the channel has stopped being the thing
    that sets the motion."""
    from shapely.geometry import Point
    from psrt.railrod.layout import disc
    p = Point(*layout.pivot)
    for name, part in (("rail", layout.rail_right),
                       ("rails above", layout.rails_upper),
                       ("receiver", layout.receiver),
                       ("clamp", layout.clamp_right),
                       ("other clamp", layout.clamp_left),
                       ("crankpin", disc((0.0, 0.0), layout.Rb))):
        assert not part.contains(p), f"the pivot landed inside the {name}"


def test_the_clamp_swings_out_of_the_notch_without_cutting_the_rail(layout):
    """Opening has to free the tongue monotonically. The notch is not an arc
    about the pivot, so this is a real question, not a tautology."""
    from shapely import affinity
    pivot = tuple(layout.pivot)
    last = layout.tongue_right.intersection(layout.rail_right).area
    for angle in (1, 2, 5, 10, 20, 30, 45):
        turned = affinity.rotate(layout.clamp_right, angle, origin=pivot)
        assert turned.intersection(layout.rail_right).area < 1e-3, angle
        tongue = affinity.rotate(layout.tongue_right, angle, origin=pivot)
        here = tongue.intersection(layout.rail_right).area
        assert here <= last + 1e-6, f"the tongue dug back in at {angle} deg"
        last = here


def test_the_channel_the_notch_and_the_rail_are_one_width(state, layout):
    """Front to back, the clamp tongue fills the notch and the channel takes
    the tongue: one dimension, not three."""
    h = layout.rail_depth
    assert layout.rail_depth == pytest.approx(
        state["railrod.rail_depth"] * 1e3)
    payload = __import__("psrt.railrod.layout", fromlist=["x"]).section_payload(
        layout)
    assert payload["swing"]["channel_width_mm"] == pytest.approx(h)


def test_the_notch_sits_inside_the_receiver(layout):
    """The clamp reaches the notch through the receiver's wall, not over the
    top of it, so the whole notch has to finish below the seat floor."""
    assert layout.notch.A[1] < layout.v_seat
    assert layout.v_e < layout.notch.F2[1]
    # and the channel mouth has to land on the outboard face with a rim above
    assert layout.v_mouth[1] < layout.v_rt


# --- the solids -----------------------------------------------------------------

def test_six_separate_valid_solids_whose_masses_match_the_layout(state, layout):
    pytest.importorskip("cadquery")
    from psrt import materials
    from psrt.geometry.properties import measure
    from psrt.railrod import cad
    from psrt.railrod.analysis import layout_masses

    from psrt.railrod import lattice as lattice_mod

    parts = cad.build_parts(state, layout)
    assert set(parts) == set(cad.PART_NAMES)
    expected = layout_masses(state, layout)
    for name, solid in parts.items():
        shape = solid.val()
        assert shape.isValid(), name
        assert len(solid.solids().vals()) == 1, f"{name} is not one solid"
        material = materials.get(cad.part_material(state, name))
        mass = measure(solid, material).mass
        if name == "rr_sleeve" and layout.sleeve_core_kind == "sheet-gyroid":
            # The sleeve's solid is deliberately only its skins, face shells
            # and bearing pad -- the lattice core is a mesh, not a B-rep.
            # Add the core back at its length-averaged relative density and
            # the two models have to agree just as they do for every other
            # part.
            core = cad.sleeve_core_box(layout)
            rel = lattice_mod.mean_density(layout.lattice_rho_mid,
                                           layout.lattice_rho_end,
                                           layout.lattice_exponent)
            mass += core.val().Volume() * 1e-9 * material.density * rel
        assert mass == pytest.approx(expected[name], rel=0.01), name


def test_the_sleeve_is_meshed_whole_and_solved_as_two_materials(state):
    """The printed sleeve has a hole where its lattice is. Meshing that hole
    would drop the core out of the model and leave the skins carrying loads
    they share with it, so the FEA meshes the block whole and gives the core
    elements the homogenised properties."""
    pytest.importorskip("cadquery")
    pytest.importorskip("skfem")
    pytest.importorskip("tetgen")
    from psrt.railrod import cad, fea

    lay_clear = state["railrod.sleeve_clearance"] * 1e3
    from psrt.railrod.layout import build_layout
    lay = build_layout(state)
    printed = cad.build_sleeve(lay, lay_clear).val().Volume()
    whole = cad.build_sleeve_fea(lay, lay_clear).val().Volume()
    assert whole > printed * 1.5, "the FEA solid should include the core"

    result = fea.solve_part(state, "rr_sleeve", target_elements=6000)
    assert result.region is not None
    assert result.region.any() and not result.region.all()
    summary = fea.summarise(result, state)
    regions = summary["regions"]
    assert set(regions) == {"skin and shells", "sheet-gyroid core"}
    # the core must be allowed far less stress than the alloy around it
    assert (regions["sheet-gyroid core"]["allowable_pa"]
            < 0.5 * regions["skin and shells"]["allowable_pa"])


def test_a_contact_patch_standing_edge_on_still_gets_its_load(state):
    """A cosine pressure about the surface normal delivers nothing along an
    axis tangential to that surface. The clamp tongue in its channel is
    exactly that -- a key, not a bearing -- and it must not fail the solve.
    """
    pytest.importorskip("skfem")
    import numpy as np
    from skfem import Basis, ElementTetP1, ElementVector, MeshTet

    from psrt.fea.solve import Bearing, _bearing_load

    m = MeshTet().refined(2)
    basis = Basis(m, ElementVector(ElementTetP1()))
    # a face of the unit cube, loaded IN its own plane
    face = lambda x: np.abs(x[2]) < 1e-9        # noqa: E731
    notes = []
    load = _bearing_load(m, basis, Bearing(face, (1, 0, 0), 250.0, "shear"),
                         notes)
    delivered = float(load.reshape(-1, 3)[:, 0].sum())
    assert delivered == pytest.approx(250.0, rel=1e-6)
    assert any("uniform traction" in n for n in notes)


def test_both_mass_paths_agree_on_the_whole_rod(state):
    """``layout_masses`` works from the 2D profiles and
    ``masses_from_geometry`` from the solids. They are independent routes to
    the same number and they have to land on it, or one of them is quietly
    wrong -- which is how the sleeve's lattice core went missing from the
    geometry path the first time.
    """
    pytest.importorskip("cadquery")
    from psrt import geometry as geo
    from psrt.railrod.analysis import layout_masses
    from psrt.railrod.layout import build_layout

    lay = build_layout(state)
    from_profiles = sum(layout_masses(state, lay).values())
    from_solids = geo.masses_from_geometry(state)["rod_body_kg"]
    assert from_solids == pytest.approx(from_profiles, rel=0.01)


def test_the_lattice_mesh_weighs_what_the_density_model_says(state, layout):
    """The generated gyroid is the check on the density model, not the other
    way round: if the mesh does not weigh what the analytic relative density
    claims, the mass of every lattice sleeve in the tool is wrong."""
    pytest.importorskip("skimage")
    import numpy as np
    from psrt.railrod import cad
    from psrt.railrod import lattice as lattice_mod
    if layout.sleeve_core_kind != "sheet-gyroid":
        pytest.skip("this sleeve has no lattice")

    verts, faces = cad.build_sleeve_lattice(layout, resolution=0.18)
    tri = verts[faces]
    volume = abs(np.einsum("ij,ij->i", tri[:, 0],
                           np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0)
    box = cad.sleeve_core_box(layout).val().Volume()
    target = lattice_mod.mean_density(layout.lattice_rho_mid,
                                      layout.lattice_rho_end,
                                      layout.lattice_exponent)
    # Marching cubes on a wall a couple of voxels thick loses a little, and
    # loses it from below, so this is one-sided on purpose.
    assert 0.94 * target <= volume / box <= 1.01 * target


def test_the_lattice_is_printable_and_can_be_emptied(layout):
    if layout.sleeve_core_kind != "sheet-gyroid":
        pytest.skip("this sleeve has no lattice")
    assert layout.lattice_print["printable"]
    # The thinnest place on the wall, not its nominal thickness, is what the
    # machine has to hold.
    assert (layout.lattice_print["min_wall_mm"]
            < layout.lattice_print["nominal_sheet_mm"])
    assert layout.lattice_evacuation["aperture_ok"]
    assert layout.lattice_evacuation["path_ok"]


def test_a_lattice_too_fine_to_print_is_refused(state):
    """The lightest station binds, not the average."""
    from psrt.railrod.layout import LayoutError, build_layout
    state.set("railrod.lattice_density", 0.10)
    with pytest.raises(LayoutError) as exc:
        build_layout(state)
    assert "thinnest" in str(exc.value)


def test_end_grading_buys_more_shear_than_the_same_mass_spread_flat(state):
    """The point of grading toward the ends, in one assertion.

    A shear web's demand peaks at the ends, so the same average density put
    where the shear is has to come out stiffer than the same average spread
    evenly. If this ever fails, the grading is pointing the wrong way.
    """
    from psrt import materials
    from psrt.railrod import lattice as lattice_mod
    mat = materials.get("AlSi10Mg-T6")
    mid, end, n = 0.30, 0.50, 2.0
    flat = lattice_mod.mean_density(mid, end, n)
    graded = lattice_mod.effective_shear(mat, mid, end, n, 0.80, 1.35)
    uniform = lattice_mod.effective_shear(mat, flat, flat, n, 0.80, 1.35)
    assert graded > uniform


def test_geometry_kernel_swaps_the_rod_for_the_rail_parts(state):
    pytest.importorskip("cadquery")
    from psrt import geometry as geo
    parts = geo.build_all(state)
    assert "rod" not in parts
    assert {"rr_rails", "rr_receiver", "rr_clamp_right"} <= set(parts)
    masses = geo.masses_from_geometry(state)
    assert 0.3 < masses["rod_total_kg"] < 1.0


# --- the contact network ------------------------------------------------------

def test_network_assembly_holds_the_bolt_at_its_preload(state, layout):
    from psrt.railrod.network import build_network
    net = build_network(state, layout, {"rr_receiver": 0.03,
                                        "rr_clamp_right": 0.09})
    pre = net.solve(0.0, 0.0)
    assert pre["settled"]
    assert pre["bolt_force"] == pytest.approx(state["railrod.bolt_preload"])


def test_network_forces_balance_every_body(state, layout):
    from psrt.railrod.network import NDOF, build_network
    net = build_network(state, layout, {"rr_receiver": 0.0,
                                        "rr_clamp_right": 0.0})
    pre = net.solve(0.0, 0.0)
    for force in (40_000.0, -10_000.0):
        r = net.solve(force, 0.0, assembly=pre, active=pre["closed"])
        Gm, _, _ = net.arrays()
        generalised = Gm.T @ r["forces"] - net.G_bolt * r["bolt_force"]
        generalised[0] -= force / 2.0
        scale = max(abs(force), state["railrod.bolt_preload"])
        assert np.all(np.abs(generalised[:NDOF]) < 1e-4 * scale)


def test_firing_load_reaches_the_crankpin_through_the_rail_feet(state):
    from psrt.railrod.analysis import analyse
    a = analyse(state)
    i = int(np.argmax(a.cycle.f_rod))
    per_rail = a.cycle.f_rod[i] / 2.0
    # the floor carries at least the firing load (plus whatever preload)
    assert a.cycle.floor[i] >= 0.9 * per_rail


def test_coupled_network_reduces_to_rigid_when_parts_are_stiff(state):
    from psrt.railrod.analysis import analyse
    from psrt.railrod.coupled import Compliance, CoupledNetwork
    a = analyse(state)
    rigid = a.cycle.network
    n = len(rigid.contacts) + 1
    stiff = CoupledNetwork(rigid, Compliance(np.zeros((n, n)), np.zeros(n),
                                             np.zeros(n), np.zeros(n)))
    for f in (0.0, 30_000.0, -9_000.0):
        pre_r = rigid.solve(0.0, 0.0)
        pre_c = stiff.solve(0.0, 0.0)
        r = rigid.solve(f, 0.0, assembly=pre_r, active=pre_r["closed"])
        c = stiff.solve(f, 0.0, assembly=pre_c, active=pre_c["closed"])
        assert c["floor"] == pytest.approx(r["floor"], rel=1e-3, abs=5.0)
        assert c["bolt_force"] == pytest.approx(r["bolt_force"], rel=1e-3)


# --- margins and interfaces ------------------------------------------------------

def test_margins_replace_the_conventional_rod_and_bolts(state):
    from psrt.evaluate import evaluate
    m = evaluate(state)
    components = {mg.component for mg in m.report.margins}
    assert "rod" not in components and "rod bolts" not in components
    assert {"rails", "rail notch", "rail joint", "rail-rod bolt",
            "swing clamps"} <= components


def test_an_unbuildable_rail_rod_is_refused_on_write(state):
    from psrt.server.session import Session
    session = Session(state=state)
    result = session.set({"railrod.rail_land_length": 0.0010,
                          "railrod.notch_lower_radius": 0.0030})
    assert not result["ok"]


def test_disabled_design_is_untouched():
    s, _ = load_state("examples/ls3.json")
    from psrt.evaluate import evaluate
    m = evaluate(s)
    assert "rod" in {mg.component for mg in m.report.margins}


def test_assistant_tool_reports_the_rail_rod(state):
    from psrt.ai.tools import dispatch
    from psrt.server.session import Session
    out = dispatch(Session(state=state), "railrod_report", {})
    assert "error" not in out, out
    assert out["swing"]["assemblable"]
    assert out["network"].startswith("rigid")


# --- FEA ------------------------------------------------------------------------

@pytest.mark.slow
def test_receiver_fea_balances_around_the_whole_cycle(state):
    pytest.importorskip("skfem")
    pytest.importorskip("tetgen")
    from psrt.railrod import fea
    result = fea.solve_part(state, "rr_receiver", target_elements=6000,
                            coupled=False)
    eq = fea.equilibrium_residual(result)
    assert eq["max_relative_force"] < 1e-3
    assert eq["max_relative_moment"] < 5e-3
    su = fea.summarise(result, state)
    assert su["p99_5_von_mises_pa"] > 0
    assert len(result.theta) == result.coeffs.shape[0]


@pytest.mark.slow
def test_coupled_compliance_is_symmetric_positive(state):
    pytest.importorskip("skfem")
    from psrt.railrod.coupled import coupled_analysis
    a = coupled_analysis(state, target_elements=6000)
    W = a.cycle.network.C.W
    assert np.allclose(W, W.T)
    assert np.linalg.eigvalsh(W).min() > -1e-9 * np.abs(W).max()
    assert a.cycle.unsettled == 0
