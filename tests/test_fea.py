"""The FEA layer: what is verified, and what is deliberately refused.

Meshing and solving both work. The solver is checked against three closed
forms -- a patch test, beam theory and Lame -- and the meshing is checked
against the element count it was asked for, on all four real parts.

The one honest limitation is pinned down here too: constant-strain tetrahedra
are stiff in bending, so displacements converge from below. A test asserts
that rather than leaving it to be discovered.
"""

import numpy as np
import pytest

from psrt import materials as mt
from psrt.fea import (audit_surface, element_stress, solve_linear_elastic,
                      surface_of, tet_mesh, unconstrained_rigid_modes, weld)
from psrt.fea.mesh import MeshResult
from psrt.geometry import build as build_mod
from psrt.schema import default_state

STEEL = mt.get("4340")
TOL = 1e-9
LENGTH, WIDTH = 0.040, 0.010

CUBE_FACES = np.array([
    [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
    [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])


def bar_mesh(max_volume_mm3: float) -> MeshResult:
    import tetgen

    w, l = WIDTH, LENGTH
    corners = np.array([[0, 0, 0], [w, 0, 0], [w, w, 0], [0, w, 0],
                        [0, 0, l], [w, 0, l], [w, w, l], [0, w, l]], dtype=float)
    engine = tetgen.TetGen(corners * 1000.0, CUBE_FACES)
    nodes, elements = engine.tetrahedralize(
        order=1, mindihedral=15.0, minratio=1.4,
        maxvolume=max_volume_mm3, fixedvolume=True)[:2]
    return MeshResult(points=np.ascontiguousarray(nodes.T) * 1e-3,
                      elements=np.ascontiguousarray(elements.T),
                      tolerance=0.0, surface_triangles=12)


# 3-2-1 restraint: enough to remove rigid-body motion, and nothing more.
# Clamping the whole base instead would block Poisson contraction and invent
# a stress concentration that is a boundary condition, not a load path.
RESTRAINT = {
    "z": lambda x: x[2] < TOL,
    "y": lambda x: (x[2] < TOL) & (x[1] < TOL),
    "x": lambda x: (x[2] < TOL) & (x[1] < TOL) & (x[0] < TOL),
}


# --- the assumptions the solver rests on -----------------------------------

def test_scikit_fem_preserves_node_order_and_is_node_major():
    """Stress recovery indexes the original element table against the solved
    displacement vector. Both of these have to hold for that to be valid, and
    a permutation would pass the patch test while corrupting every real
    result -- so they are asserted rather than assumed."""
    from skfem import Basis, ElementTetP1, ElementVector, MeshTet

    mesh = bar_mesh(30.0)
    m = MeshTet(mesh.points, mesh.elements)
    assert np.allclose(m.p, mesh.points), "node order preserved"
    assert np.array_equal(np.asarray(m.t), mesh.elements), "elements preserved"

    basis = Basis(m, ElementVector(ElementTetP1()))
    assert basis.N == 3 * mesh.n_nodes

    base = np.flatnonzero(mesh.points[2] < TOL)
    z_dofs = np.sort(np.asarray(basis.get_dofs(lambda x: x[2] < TOL)
                                .nodal["u^3"]))
    assert np.array_equal(z_dofs, np.sort(3 * base + 2)), "dof = 3*node + comp"


# --- the patch test --------------------------------------------------------

@pytest.mark.parametrize("max_volume", [300.0, 30.0, 4.0])
def test_uniform_tension_is_recovered_exactly(max_volume):
    """A bar under uniform traction must return exactly that stress, at any
    mesh density. If this is not exact, nothing else from the solver means
    anything."""
    mesh = bar_mesh(max_volume)
    applied = 50e6
    result = solve_linear_elastic(
        mesh, STEEL, fixed=RESTRAINT,
        traction=lambda x: x[2] > LENGTH - TOL,
        traction_vector=(0.0, 0.0, applied), load_case="patch test")

    assert result.von_mises.max() == pytest.approx(applied, rel=1e-9)
    assert result.von_mises.min() == pytest.approx(applied, rel=1e-9)
    assert not result.free_rigid_modes


@pytest.mark.parametrize("max_volume", [300.0, 30.0, 4.0])
def test_axial_extension_matches_hookes_law(max_volume):
    mesh = bar_mesh(max_volume)
    applied = 50e6
    result = solve_linear_elastic(
        mesh, STEEL, fixed=RESTRAINT,
        traction=lambda x: x[2] > LENGTH - TOL,
        traction_vector=(0.0, 0.0, applied), load_case="patch test")
    expected = applied * LENGTH / STEEL.youngs_modulus
    assert result.displacement[2].max() == pytest.approx(expected, rel=1e-9)


def test_stress_scales_linearly_with_load():
    mesh = bar_mesh(30.0)
    peaks = []
    for applied in (25e6, 50e6):
        result = solve_linear_elastic(
            mesh, STEEL, fixed=RESTRAINT,
            traction=lambda x: x[2] > LENGTH - TOL,
            traction_vector=(0.0, 0.0, applied), load_case="linearity")
        peaks.append(result.peak_stress)
    assert peaks[1] == pytest.approx(2.0 * peaks[0], rel=1e-9)


def test_equilibrium_residual_is_negligible():
    result = solve_linear_elastic(
        bar_mesh(30.0), STEEL, fixed=RESTRAINT,
        traction=lambda x: x[2] > LENGTH - TOL,
        traction_vector=(0.0, 0.0, 50e6), load_case="residual")
    assert result.residual < 1e-10


# --- rigid-body modes ------------------------------------------------------

def test_an_under_constrained_part_is_detected():
    """An equilibrium residual cannot find this: K times a rigid motion is
    zero, so a free mode satisfies K u = f exactly. The stresses can even come
    back right while the displacements are meaningless."""
    mesh = bar_mesh(30.0)
    result = solve_linear_elastic(
        mesh, STEEL, fixed={"z": lambda x: x[2] < TOL},
        traction=lambda x: x[2] > LENGTH - TOL,
        traction_vector=(0.0, 0.0, 50e6), load_case="under-constrained")

    assert set(result.free_rigid_modes) == {
        "translation x", "translation y", "rotation about z"}
    assert any("UNDER-CONSTRAINED" in n for n in result.notes)
    assert result.residual < 1e-10, "the residual is blind to this"


def test_a_fully_fixed_part_has_no_free_modes():
    mesh = bar_mesh(30.0)
    dofs = np.arange(3 * mesh.n_nodes)
    assert unconstrained_rigid_modes(mesh, dofs) == []


def test_an_unrestrained_part_has_all_six_modes_free():
    mesh = bar_mesh(30.0)
    assert len(unconstrained_rigid_modes(mesh, np.array([], dtype=int))) == 6


def test_no_restraint_at_all_is_refused():
    with pytest.raises(ValueError, match="singular"):
        solve_linear_elastic(
            bar_mesh(300.0), STEEL, fixed=lambda x: x[2] < -1.0,
            traction=lambda x: x[2] > LENGTH - TOL,
            traction_vector=(0.0, 0.0, 1e6), load_case="free body")


def test_a_traction_that_lands_nowhere_is_refused():
    with pytest.raises(ValueError, match="matched no boundary facets"):
        solve_linear_elastic(
            bar_mesh(300.0), STEEL, fixed=RESTRAINT,
            traction=lambda x: x[2] > 10.0,
            traction_vector=(0.0, 0.0, 1e6), load_case="nowhere")


def test_an_unknown_axis_is_refused():
    with pytest.raises(ValueError, match="unknown axis"):
        solve_linear_elastic(
            bar_mesh(300.0), STEEL, fixed={"w": lambda x: x[2] < TOL},
            load_case="bad axis")


# --- stress recovery -------------------------------------------------------

def test_rigid_translation_produces_no_stress():
    mesh = bar_mesh(30.0)
    field = np.ones((3, mesh.n_nodes)) * 1e-3
    von_mises, _ = element_stress(mesh, field, STEEL)
    assert np.allclose(von_mises, 0.0, atol=1e-6)


def test_hydrostatic_strain_has_no_von_mises_stress():
    """Von Mises is deviatoric: uniform expansion raises pressure, not it."""
    mesh = bar_mesh(30.0)
    field = mesh.points * 1e-4
    von_mises, principal = element_stress(mesh, field, STEEL)
    assert np.allclose(von_mises, 0.0, atol=1.0)
    assert np.all(principal[0] > 0), "but the principal stresses are not zero"


# --- the mesh pipeline -----------------------------------------------------

def test_cadquery_surfaces_are_not_watertight_until_welded():
    """OpenCascade triangulates each face independently and never merges the
    seams, so what comes out is a pile of patches, not a shell. tetgen cannot
    tell that from a surface with holes and responds by segfaulting."""
    state = default_state()
    for part in ("pin", "rod", "piston", "sleeve"):
        vertices, triangles = surface_of(state, part, 0.4, 0.35)
        before = audit_surface(vertices, triangles)
        assert before["boundary_edges"] > 0, f"{part} unexpectedly closed"

        after = audit_surface(*weld(vertices, triangles))
        assert after["watertight"], part
        assert after["triangles"] == before["triangles"]


def test_welding_does_not_lose_triangles():
    state = default_state()
    vertices, triangles = surface_of(state, "pin", 0.4, 0.35)
    welded_v, welded_f = weld(vertices, triangles)
    assert len(welded_f) == len(triangles)
    assert len(welded_v) < len(vertices), "duplicates were merged"


def test_tessellation_tolerance_is_actually_honoured():
    """It was not, for three phases. OpenCascade caches a triangulation on the
    shape and reuses it whatever tolerance is asked for, so every call after
    the first returned the same mesh."""
    state = default_state()
    counts = [len(surface_of(state, "pin", tol, ang)[1])
              for tol, ang in ((0.1, 0.15), (0.6, 0.4), (2.0, 0.8))]
    assert counts[0] > counts[1] > counts[2], counts


@pytest.mark.parametrize("part", ["pin", "sleeve", "rod", "piston"])
def test_every_part_meshes_close_to_the_requested_density(part):
    """The regression this guards is a real one, and it cost a phase.

    This module used to subdivide the surface before meshing. tetgen preserves
    the boundary it is handed, so feeding it subdivided slivers forced it to
    flood the volume with Steiner points -- the pin came out at 390,000
    elements against a 25,000 target, and the conclusion drawn was that tetgen
    could not be controlled. The subdivision was the cause. Without it every
    part lands close to its target, which is what this asserts."""
    mesh = tet_mesh(default_state(), part, target_elements=25_000)
    assert 25_000 / 1.6 < mesh.n_elements < 25_000 * 1.6, mesh.n_elements
    assert mesh.volume_drift < 0.02


def test_a_coarser_target_really_gives_a_coarser_mesh():
    """The knob has to work in both directions, monotonically."""
    counts = [tet_mesh(default_state(), "pin", target_elements=n).n_elements
              for n in (8_000, 25_000)]
    assert counts[0] < counts[1] * 0.6, counts


def test_the_package_states_what_works():
    import psrt.fea as fea
    assert "validated" in fea.STATUS
    assert "constant-strain" in fea.STATUS


# --- against closed forms --------------------------------------------------

LONG, SIDE = 0.100, 0.010

LONG_FACES = np.array([
    [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
    [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])


def cantilever_mesh(max_volume_mm3: float) -> MeshResult:
    import tetgen

    s, l = SIDE, LONG
    corners = np.array([[0, 0, 0], [s, 0, 0], [s, s, 0], [0, s, 0],
                        [0, 0, l], [s, 0, l], [s, s, l], [0, s, l]],
                       dtype=float)
    nodes, elements = tetgen.TetGen(
        corners * 1000.0, LONG_FACES).tetrahedralize(
            order=1, mindihedral=15.0, minratio=1.4,
            maxvolume=max_volume_mm3, fixedvolume=True)[:2]
    return MeshResult(points=np.ascontiguousarray(nodes.T) * 1e-3,
                      elements=np.ascontiguousarray(elements.T),
                      tolerance=0.0, surface_triangles=12)


def _cantilever(max_volume_mm3, applied=1.0e6):
    mesh = cantilever_mesh(max_volume_mm3)
    result = solve_linear_elastic(
        mesh, STEEL, fixed=lambda x: x[2] < TOL,
        traction=lambda x: x[2] > LONG - TOL,
        traction_vector=(applied, 0.0, 0.0), load_case="cantilever")
    return mesh, result, applied * SIDE * SIDE


def test_bending_stress_matches_beam_theory():
    """Stress is what this layer exists to produce, so stress is what has to
    match. Sampled on the outer fibres at mid-span, away from both the clamp
    and the loaded end, where Saint-Venant says beam theory applies."""
    mesh, result, load = _cantilever(0.8)
    second_moment = SIDE ** 4 / 12.0

    centroid = mesh.points.T[mesh.elements.T].mean(axis=1)
    offset = centroid[:, 0] - SIDE / 2.0
    outer = (np.abs(centroid[:, 2] - LONG / 2) < 0.006) \
        & (np.abs(offset) > SIDE / 2 * 0.65)
    assert outer.sum() > 200

    theory = load * (LONG - centroid[:, 2]) * np.abs(offset) / second_moment
    ratio = result.von_mises[outer] / theory[outer]
    assert np.median(ratio) == pytest.approx(1.0, abs=0.03)


def test_constant_strain_tetrahedra_are_stiff_in_bending():
    """Not a defect to fix -- a property of the element, asserted so that the
    deflections are never mistaken for precise. It converges from BELOW, so
    the test is that refinement closes the gap and never overshoots."""
    poisson = STEEL.poisson
    shear = STEEL.youngs_modulus / (2.0 * (1.0 + poisson))
    second_moment = SIDE ** 4 / 12.0
    area = SIDE * SIDE

    tips = []
    for max_volume in (12.0, 0.8):
        _, result, load = _cantilever(max_volume)
        timoshenko = (load * LONG ** 3 / (3 * STEEL.youngs_modulus * second_moment)
                      + load * LONG / ((5.0 / 6.0) * shear * area))
        tips.append(float(np.abs(result.displacement[0]).max()) / timoshenko)

    assert tips[0] < tips[1] < 1.0
    assert tips[1] > 0.90


def test_a_pressurised_sleeve_matches_lame():
    """The one closed form available on real CAD geometry rather than a block,
    so it checks the mesher and the solver together."""
    from psrt import materials as materials_mod
    from psrt.schema import load_state

    state, _ = load_state("examples/ls3.json")
    inner = state["engine.bore"] / 2.0
    outer = inner + state["sleeve.wall_thickness"]
    applied = 10e6
    hoop = applied * (outer ** 2 + inner ** 2) / (outer ** 2 - inner ** 2)
    # Plane stress at the bore: hoop tension against radial compression.
    expected = np.sqrt(hoop ** 2 + hoop * applied + applied ** 2)

    readings = []
    for target in (25_000, 60_000):
        mesh = tet_mesh(state, "sleeve", target_elements=target)
        bottom = mesh.points[2].min()
        middle = 0.5 * (bottom + mesh.points[2].max())
        span = 0.25 * (mesh.points[2].max() - bottom)
        result = solve_linear_elastic(
            mesh, materials_mod.get("grey-iron"),
            fixed={"z": lambda x: x[2] < bottom + 1e-6,
                   "x": lambda x: (x[2] < bottom + 1e-6) & (np.abs(x[1]) < 2e-3),
                   "y": lambda x: (x[2] < bottom + 1e-6) & (np.abs(x[0]) < 2e-3)},
            traction=lambda x: np.hypot(x[0], x[1]) < (inner + outer) / 2.0,
            pressure=applied, load_case="lame")

        centroid = mesh.points.T[mesh.elements.T].mean(axis=1)
        radius = np.hypot(centroid[:, 0], centroid[:, 1])
        at_bore = (np.abs(centroid[:, 2] - middle) < span) \
            & (radius < inner + 0.15 * (outer - inner))
        readings.append(float(np.median(result.von_mises[at_bore])))
        assert result.residual < 1e-9

    assert readings[0] == pytest.approx(expected, rel=0.03)
    # And the answer must not move when the mesh is refined.
    assert readings[1] == pytest.approx(readings[0], rel=0.02)


def test_pressure_and_a_fixed_traction_cannot_both_be_given():
    mesh = bar_mesh(30.0)
    with pytest.raises(ValueError, match="not both"):
        solve_linear_elastic(
            mesh, STEEL, fixed=RESTRAINT,
            traction=lambda x: x[2] > LENGTH - TOL,
            traction_vector=(0.0, 0.0, 1e6), pressure=1e6,
            load_case="contradictory")


# --- the field the viewport draws -----------------------------------------

def test_the_drawn_surface_is_closed():
    """Every edge of the boundary surface must be shared by exactly two
    triangles. An open edge means a face was miscounted as interior, and the
    part would render with holes in it."""
    from collections import Counter

    from psrt.fea.field import boundary_surface

    mesh = bar_mesh(30.0)
    triangles = boundary_surface(mesh.elements)
    edges = Counter()
    for a, b, c in triangles:
        for edge in ((a, b), (b, c), (c, a)):
            edges[tuple(sorted(edge))] += 1
    assert set(edges.values()) == {2}


def test_the_drawn_surface_encloses_the_right_volume():
    """Closed is not enough -- the winding has to be outward too, or the
    signed volume comes back negative and the lighting is inside out."""
    from psrt.fea.field import boundary_surface

    mesh = bar_mesh(30.0)
    points = mesh.points.T
    triangles = points[boundary_surface(mesh.elements)]
    signed = np.einsum("ij,ij->i", triangles[:, 0],
                       np.cross(triangles[:, 1], triangles[:, 2])).sum() / 6.0
    assert signed == pytest.approx(LENGTH * WIDTH * WIDTH, rel=1e-9)


def test_a_uniform_field_averages_to_itself_at_the_nodes():
    from psrt.fea.field import nodal_field

    mesh = bar_mesh(30.0)
    values = np.full(mesh.n_elements, 42.0)
    nodal = nodal_field(mesh.points, mesh.elements, values)
    assert nodal == pytest.approx(42.0, rel=1e-12)


def test_the_colour_scale_is_clipped_not_maxed():
    """A point restraint can produce one element reading several times the
    real peak. Scaled against that, every genuine feature washes out, so the
    payload carries a clip and the legend says so."""
    mesh = bar_mesh(30.0)
    result = solve_linear_elastic(
        mesh, STEEL, fixed=RESTRAINT,
        traction=lambda x: x[2] > LENGTH - TOL,
        traction_vector=(0.0, 0.0, 50e6), load_case="clip")

    from psrt.fea.field import field_payload
    payload = field_payload(result)
    span = payload["range_mpa"]
    assert span["clip"] <= span["max"]
    assert len(payload["stress_mpa"]) == len(payload["vertices"])
    assert max(max(t) for t in payload["triangles"]) < len(payload["vertices"])


# --- the component load cases ---------------------------------------------

CASE_PARTS = ["pin", "piston", "rod", "sleeve"]


@pytest.fixture(scope="module")
def ls3():
    from psrt.schema import load_state
    return load_state("examples/ls3.json")[0]


@pytest.mark.parametrize("part", CASE_PARTS)
def test_every_case_is_properly_restrained_and_in_equilibrium(part, ls3):
    """The two ways a load case silently lies: a free rigid-body mode, or a
    residual that says the applied load never reached the supports."""
    from psrt.fea.cases import CASES

    case = CASES[part](ls3, target_elements=12_000)
    assert case.solve.free_rigid_modes == []
    assert case.solve.residual < 1e-9


@pytest.mark.parametrize("part", CASE_PARTS)
def test_every_case_samples_where_its_formula_applies(part, ls3):
    """A ratio taken over the whole part compares a peak against a formula
    that does not describe it. The rod is the clear case: its highest stress
    is in the small-end ring and `F / A_shank` is about the shank."""
    from psrt.fea.cases import CASES

    case = CASES[part](ls3, target_elements=12_000)
    assert case.sample is not None
    assert case.sample.sum() > 100, f"{part}: sample region caught almost nothing"
    assert np.isfinite(case.sampled_stress)
    assert case.sampled_stress <= case.solve.peak_stress


def test_the_sleeve_case_reproduces_lame(ls3):
    """The one case with an exact closed form on its own geometry, so its
    ratio has a known right answer -- 1.0. It is the end-to-end check that
    the mesher, the pressure load and the solver agree."""
    from psrt.fea.cases import CASES

    case = CASES["sleeve"](ls3, target_elements=20_000)
    assert case.ratio == pytest.approx(1.0, abs=0.10)


@pytest.mark.parametrize("part", CASE_PARTS)
def test_the_reported_ratio_does_not_move_with_mesh_density(part, ls3):
    """The test that matters most here. A ratio that climbs as the mesh
    refines is tracking a boundary-condition singularity, not the part -- the
    pin sampled up to its restraint edge gave 2.01, 2.39, 2.38 at three
    densities before the sample was pulled clear of it."""
    from psrt.fea.cases import CASES

    coarse = CASES[part](ls3, target_elements=12_000).ratio
    fine = CASES[part](ls3, target_elements=30_000).ratio
    assert fine == pytest.approx(coarse, rel=0.15), (part, coarse, fine)


# --- slivers ---------------------------------------------------------------

def _sliver_mesh(flatness: float) -> MeshResult:
    """A two-element mesh whose second tetrahedron is almost flat.

    Built by hand rather than hunted for in a real part. The crash it
    reproduces -- "Eigenvalues did not converge" out of the principal-stress
    decomposition -- depends on mesher luck, so waiting for a geometry that
    happens to produce one is not a test.
    """
    # Nodes 1, 2 and 3 span the plane x + y + z = 1e-2. Node 4 sits on that
    # same plane to within `flatness`, so the second tetrahedron is flat.
    points = np.array([
        [0.0, 0.0, 0.0],
        [1e-2, 0.0, 0.0],
        [0.0, 1e-2, 0.0],
        [0.0, 0.0, 1e-2],
        [0.5e-2, 0.5e-2, flatness],
    ]).T
    elements = np.array([[0, 1, 2, 3], [1, 2, 3, 4]]).T
    return MeshResult(points=np.ascontiguousarray(points),
                      elements=np.ascontiguousarray(elements),
                      tolerance=0.0, surface_triangles=0)


def test_a_sliver_element_does_not_crash_the_stress_recovery():
    """This is the reported bug. A tetrahedron with four almost-coplanar
    corners has a singular Jacobian; np.linalg.solve returns infinities, and
    eigvalsh raises LinAlgError on them -- a long way from the cause, and it
    kills the whole solve."""
    mesh = _sliver_mesh(1e-18)
    field = np.zeros_like(mesh.points)
    field[2] = np.linspace(0.0, 1e-6, mesh.n_nodes)

    von_mises, principal, quality = element_stress(mesh, field, STEEL,
                                                   report=True)
    assert np.isfinite(von_mises).all()
    assert np.isfinite(principal).all()
    assert quality["degenerate_elements"] == 1


def test_a_healthy_element_beside_a_sliver_is_still_solved():
    """Discarding the bad element must not quietly discard the good one."""
    mesh = _sliver_mesh(1e-18)
    field = np.zeros_like(mesh.points)
    field[2] = np.linspace(0.0, 1e-6, mesh.n_nodes)

    von_mises, _, quality = element_stress(mesh, field, STEEL, report=True)
    assert quality["degenerate_elements"] == 1
    assert von_mises[0] > 0.0
    assert von_mises[1] == 0.0


def test_slivers_are_counted_in_the_result_notes():
    """Silently zeroing elements would be worse than crashing: a handful is
    normal, thousands mean the answer is not usable."""
    mesh = bar_mesh(30.0)
    result = solve_linear_elastic(
        mesh, STEEL, fixed=RESTRAINT,
        traction=lambda x: x[2] > LENGTH - TOL,
        traction_vector=(0.0, 0.0, 50e6), load_case="sliver reporting")
    assert not any("slivers" in note for note in result.notes), \
        "a clean mesh must not report slivers"


# --- the whole cycle from one solve ----------------------------------------

def test_stress_scales_exactly_with_load(ls3):
    """The claim the crank-angle scrubbing rests on. In a linear analysis
    with a single load pattern the stress everywhere is proportional to the
    load, so the field at any other crank angle is this field times a number.
    If that is only approximately true, the scrubbing is a lie."""
    from psrt import materials as materials_mod
    from psrt.fea import solve_linear_elastic, tet_mesh

    mesh = tet_mesh(ls3, "sleeve", target_elements=10_000)
    inner = ls3["engine.bore"] / 2.0
    outer = inner + ls3["sleeve.wall_thickness"]
    bottom = mesh.points[2].min()
    wall = outer - inner

    def run(pressure):
        return solve_linear_elastic(
            mesh, materials_mod.get("grey-iron"),
            fixed={"z": lambda x: x[2] < bottom + 1e-5,
                   "y": lambda x: (x[2] < bottom + 1e-5) & (np.abs(x[0]) < wall),
                   "x": lambda x: (x[2] < bottom + 1e-5) & (np.abs(x[0]) < wall)
                                  & (x[0] > 0)},
            traction=lambda x: np.hypot(x[0], x[1]) < (inner + outer) / 2.0,
            pressure=pressure, load_case="linearity")

    reference = run(7.0e6)
    for factor in (0.1, 0.5, 1.7):
        scaled = run(7.0e6 * factor)
        assert scaled.von_mises == pytest.approx(
            reference.von_mises * factor, rel=1e-9)


@pytest.mark.parametrize("part", CASE_PARTS)
def test_every_case_carries_its_whole_load_history(part, ls3):
    from psrt.fea.cases import CASES

    case = CASES[part](ls3, target_elements=10_000)
    assert case.load_series is not None
    scale = np.abs(case.load_series) / case.load_reference
    # The solve is done at the governing angle, so nothing in the cycle may
    # exceed it -- a scale above 1 would mean the case solved the wrong point.
    assert scale.max() == pytest.approx(1.0, abs=1e-6)
    assert scale.min() >= 0.0


def test_the_crown_reference_is_a_pressure_not_a_force(ls3):
    """`force` is the headline number a person reads and for the crown it is
    the total gas force; the load series is a pressure. Dividing by the wrong
    one scales the whole cycle by the bore area -- a factor of 119 here."""
    from psrt.fea.cases import CASES

    case = CASES["piston"](ls3, target_elements=10_000)
    assert case.load_reference != pytest.approx(abs(case.force))
    assert np.max(np.abs(case.load_series)) == pytest.approx(
        case.load_reference, rel=1e-9)


# --- what the colours are measured against ---------------------------------

def test_the_allowable_is_taken_at_temperature_not_cold(ls3):
    """The derating tables are keyed in KELVIN while the thermal map reports
    Celsius. Mixing them returns the room-temperature strength and silently
    overstates an aluminium crown's allowable by about half."""
    from psrt.fea.cases import allowable_for

    piston = allowable_for(ls3, "piston")
    assert piston["temperature_c"] > 150
    assert piston["retained_fraction"] < 0.75
    assert piston["allowable_pa"] < piston["cold_pa"] * 0.75


def test_the_allowable_matches_what_the_margins_use(ls3):
    """The colour scale and the safety factors must not disagree about what
    a part is allowed to carry."""
    from psrt import materials as materials_mod
    from psrt.fea.cases import SERVICE_TEMPERATURE, allowable_for
    from psrt.margins import ComponentContext

    for part, key in (("piston", "materials.piston"), ("pin", "materials.pin"),
                      ("rod", "materials.rod"),
                      ("sleeve", "materials.sleeve")):
        info = allowable_for(ls3, part)
        material = materials_mod.get(ls3[key])
        expected, basis = ComponentContext.strength(
            ComponentContext, material, info["temperature_c"] + 273.15)
        assert info["allowable_pa"] == pytest.approx(expected, rel=1e-12)
        assert info["basis"] == basis


def test_a_brittle_material_falls_back_to_ultimate(ls3):
    """Grey iron has no yield point, so a yield-referenced colour scale has
    to say what it IS referenced to."""
    from psrt.fea.cases import allowable_for

    assert "ultimate" in allowable_for(ls3, "sleeve")["basis"]


# --- calibration: what it found, and why it stopped ------------------------

def test_a_self_equilibrated_load_set_leaves_the_restraint_idle():
    """The premise of the whole calibration attempt. If the loads and the
    reactions are both applied as contact pressures, the restraint only has
    to remove rigid-body motion -- so it carries no force and invents no
    stress concentration. That is the difference between a stress field you
    can look at and one you can calibrate against."""
    from psrt.fea.calibrate import _pin_case
    from psrt.schema import load_state

    state, _ = load_state("examples/ls3.json")
    _, result, _ = _pin_case(state, 12_000)

    assert result.free_rigid_modes == []
    assert result.residual < 1e-9
    # With rigid supports this case reported 733 MPa at the 99.5th
    # percentile, almost all of it restraint artefact.
    assert result.percentile_stress() < 600e6


def test_the_pin_span_cannot_be_fitted_and_says_so():
    """A linear analysis takes the contact pressure distribution as an INPUT,
    so the resultant sits at the centroid of whatever the analyst prescribed.
    Spreading the boss reaction over a third of the boss returns one answer
    and over the whole boss returns another -- the method reproduces its own
    assumption. It has to report a bracket, not a fit."""
    from psrt.fea.calibrate import pin_span_bracket
    from psrt.schema import load_state

    state, _ = load_state("examples/ls3.json")
    bracket = pin_span_bracket(state, target_elements=12_000)

    low, high = bracket["bracket"]
    assert high > low * 2, "the assumption dominates; that is the point"
    assert low <= bracket["current"] <= high, \
        "the phase 2 guess should at least sit inside the bracket"
    assert "cannot determine" in bracket["verdict"]


def test_the_crown_does_not_deflect_like_a_plate():
    """The clamped-plate model says the crown dishes down at the centre and
    is held at the rim. It does the opposite: the pin boss pad is over half
    the bore wide and supports the centre, so the rim falls away around it.
    A constant fitted to a model with the wrong shape is a fudge factor, not
    a calibration -- which is why this phase ships no new constants."""
    from psrt.fea.calibrate import crown_deflection_profile
    from psrt.schema import load_state

    state, _ = load_state("examples/ls3.json")
    profile = crown_deflection_profile(state, target_elements=25_000)

    assert profile["dishes_like_a_plate"] is False
    assert profile["centre_deflection_m"] > profile["rim_deflection_m"]
    assert profile["boss_pad_fraction_of_radius"] > 0.4


def test_the_calibrated_constants_were_left_alone():
    """Whatever else this phase did, it must not have quietly written a
    fitted number into the design state."""
    from psrt.schema import default_state

    state = default_state()
    assert state["piston.crown_support_radius_fraction"] == pytest.approx(0.30)
    assert state["pin.support_span_factor"] == pytest.approx(0.50)
    assert state.param("piston.crown_support_radius_fraction").source \
        == "estimated"


# --- a load that lands nowhere ---------------------------------------------

@pytest.mark.parametrize("part", CASE_PARTS)
def test_every_case_applies_the_load_it_says_it_does(part, ls3):
    """A traction spread over a curved surface and sized by PROJECTED area
    delivers pi/2 times the intended force, because the arc is longer than
    its projection. That is how the pin came to apply 77 kN for a 48.5 kN
    load and report stresses 59% high. Bearings are rescaled to their
    resultant after assembly, so this now holds whatever the mesh does."""
    from psrt.fea.cases import CASES

    case = CASES[part](ls3, target_elements=12_000)
    note = next(n for n in case.solve.notes if "applied load" in n)
    total = float(note.split(",")[-1].strip().split()[0])

    if case.load_unit == "N":
        # A single contact patch: total force is the resultant.
        assert total == pytest.approx(abs(case.force) / 1e3, rel=0.02)
    else:
        # Pressure on a closed surface: the resultant is zero by symmetry,
        # so only the total magnitude means anything.
        assert total > 0


def test_a_load_that_reaches_nothing_is_refused(ls3):
    """The worst failure this tool can have is a wrong answer that looks
    like a good one. A load landing entirely on restrained nodes solves
    perfectly and returns zeros, which paints a uniformly green part and
    reads as 'this component is fine'."""
    from psrt import materials as materials_mod
    from psrt.fea import solve_linear_elastic, tet_mesh

    mesh = tet_mesh(ls3, "pin", target_elements=8_000)
    everywhere = lambda x: np.ones(x.shape[1], dtype=bool)   # noqa: E731

    with pytest.raises(ValueError, match="restrained nodes|no displacement"):
        solve_linear_elastic(
            mesh, materials_mod.get(ls3["materials.pin"]),
            fixed=everywhere,
            traction=lambda x: x[2] > 0,
            traction_vector=(0.0, 0.0, -1e6), load_case="goes nowhere")


def test_a_pressurised_tube_is_not_mistaken_for_no_load(ls3):
    """The first version of that guard tested the RESULTANT, which for a
    pressurised cylinder is exactly zero by symmetry -- it rejected the one
    case in this tool with an exact analytical answer."""
    from psrt.fea.cases import CASES

    case = CASES["sleeve"](ls3, target_elements=12_000)
    note = next(n for n in case.solve.notes if "applied load" in n)
    assert "0.00, +0.00" in note or "resultant" in note
    assert case.solve.peak_stress > 0


# --- meshes that make a stiffness matrix singular --------------------------

def _strip(n_elements: int, origin=(0.0, 0.0, 0.0)):
    """A face-connected strip of real tetrahedra, as a healthy starting point.

    The points run along a helix rather than a straight line: four points in
    a row have to be genuinely non-coplanar, or every element has zero volume
    and the flat-element repair correctly deletes the entire strip.
    """
    ox, oy, oz = origin
    points = [[ox + i, oy + np.cos(i), oz + np.sin(i)]
              for i in range(n_elements + 3)]
    elements = [[i, i + 1, i + 2, i + 3] for i in range(n_elements)]
    return np.array(points, dtype=float), np.array(elements)


def test_the_healthy_strip_really_is_healthy():
    """Every test below starts from this, so it has to be a mesh with real
    volume and no faults -- otherwise a repair counter reads 1 for the wrong
    reason and the test passes without testing anything."""
    from psrt.fea.mesh import _face_components, _make_solvable

    points, elements = _strip(50)
    nodes, elems, report = _make_solvable(points, elements, "test")
    assert report == {"orphan_nodes": 0, "islands_dropped": 0, "reoriented": 0,
                      "welded_nodes": 0, "flat_elements": 0,
                      "pinched_elements": 0}
    assert len(elems) == 50
    assert _face_components(elems)[0] == 1


def test_an_orphan_node_is_removed_before_it_can_go_singular():
    """A point no element references contributes nothing to the stiffness
    matrix, so its rows and columns are exactly zero and the whole system is
    singular -- however well the part is restrained. The symptom is non-finite
    displacements, which looks like a restraint problem and is not."""
    from psrt.fea.mesh import _make_solvable

    points, elements = _strip(50)
    points = np.vstack([points, [[99.0, 99.0, 99.0]]])      # referenced by none

    nodes, elems, report = _make_solvable(points, elements, "test")
    assert report["orphan_nodes"] == 1
    assert report["islands_dropped"] == 0
    assert len(nodes) == len(points) - 1
    assert elems.max() < len(nodes), "elements must be reindexed, not just cut"


def test_a_speck_of_detached_mesh_is_discarded():
    from psrt.fea.mesh import _make_solvable

    points, elements = _strip(200)
    base = len(points)
    points = np.vstack([points, [[50.0, 50.0, 50.0], [51.0, 50.0, 50.0],
                                 [50.0, 51.0, 50.0], [50.0, 50.0, 51.0]]])
    elements = np.vstack([elements, [[base, base + 1, base + 2, base + 3]]])

    nodes, elems, report = _make_solvable(points, elements, "test")
    assert report["islands_dropped"] == 1
    assert len(elems) == 200
    assert elems.max() < len(nodes)


def test_a_large_detached_piece_is_refused_not_deleted():
    """Debris is one thing; half the part is another. At that size it is more
    likely a real piece of the geometry, and quietly deleting it would be
    worse than failing."""
    from psrt.fea.mesh import _make_solvable

    points, elements = _strip(10)
    base = len(points)
    extra_points, extra_elements = _strip(10, origin=(100.0, 0.0, 0.0))
    points = np.vstack([points, extra_points])
    elements = np.vstack([elements, extra_elements + base])

    with pytest.raises(ValueError, match="disconnected pieces"):
        _make_solvable(points, elements, "test")


def test_coincident_nodes_are_welded_so_load_can_cross_the_crack():
    """Two nodes at the same coordinates leave a crack: the elements either
    side share no index, so nothing carries load between them and each piece
    moves independently."""
    from psrt.fea.mesh import _face_components, _make_solvable

    left_points, left_elements = _strip(20)
    right_points, right_elements = _strip(20)
    base = len(left_points)
    # The right-hand strip is a duplicate of the left, node for node, so
    # every one of its points is coincident with one of the left's.
    points = np.vstack([left_points, right_points])
    elements = np.vstack([left_elements, right_elements + base])
    assert _face_components(elements)[0] == 2, "two pieces before welding"

    nodes, elems, report = _make_solvable(points, elements, "test")
    assert report["welded_nodes"] == base
    assert _face_components(elems)[0] == 1, "one piece after welding"


def test_a_flat_element_is_dropped():
    from psrt.fea.mesh import _make_solvable

    points, elements = _strip(20)
    base = len(points)
    # Four coplanar points, hung off an existing node so it is not an island.
    points = np.vstack([points, [[0.0, 5.0, 0.0], [1.0, 5.0, 0.0],
                                 [0.0, 6.0, 0.0]]])
    elements = np.vstack([elements, [[0, base, base + 1, base + 2]]])
    for row in points[-3:]:
        row[2] = points[0][2]                     # coplanar with node 0

    nodes, elems, report = _make_solvable(points, elements, "test")
    assert report["flat_elements"] == 1
    assert len(elems) == 20


def test_an_inverted_element_is_turned_the_right_way():
    """A negatively-oriented tetrahedron contributes NEGATIVE stiffness,
    which destroys the positive-definiteness the solver relies on. Different
    tetgen builds order nodes differently, so this can be clean on one
    platform and not on another."""
    from psrt.fea.mesh import _make_solvable

    points, elements = _strip(20)
    flipped = elements.copy()
    flipped[5] = flipped[5][[0, 2, 1, 3]]

    _, before, _ = _make_solvable(points, elements, "test")
    _, after, report = _make_solvable(points, flipped, "test")
    assert report["reoriented"] == 1
    assert sorted(after[5]) == sorted(before[5]), "same element, turned over"


# --- the fault every other repair misses -----------------------------------

def _hinge(shared: int):
    """A healthy strip plus one tetrahedron attached by ``shared`` nodes.

    One shared node is a vertex hinge, two is an edge hinge; three would be a
    whole face, which is the only amount that actually ties two tetrahedra
    together. The extra element has real volume, so nothing else in
    ``_make_solvable`` has any reason to remove it.
    """
    points, elements = _strip(200)
    base = len(points)
    hinge = [25, 26, 27][:shared]               # interior nodes of the strip
    fresh = [[25.0, 20.0, 0.0], [26.0, 20.0, 0.0],
             [25.0, 21.0, 0.0], [25.0, 20.0, 1.0]][:4 - shared]
    points = np.vstack([points, fresh])
    chunk = hinge + [base + i for i in range(len(fresh))]
    return points, np.vstack([elements, [chunk]])


def test_a_chunk_hinged_on_one_node_is_found_and_discarded():
    """The island test joins any two elements sharing a single NODE, so a
    chunk hinged on one vertex passes it and still leaves the stiffness
    matrix singular: the chunk rotates about its hinge at zero strain, and
    therefore at zero energy, however the part is restrained. This is the
    fault that reports 0 orphan nodes and 0 islands dropped and returns NaN
    anyway, and only face-connectivity finds it."""
    import scipy.sparse as sp

    from psrt.fea.mesh import _face_components, _make_solvable

    points, elements = _hinge(1)

    # The node-connectivity test that runs first sees nothing wrong.
    rows = np.repeat(np.arange(len(elements)), 4)
    incidence = sp.csr_matrix(
        (np.ones(rows.size, dtype=np.int8), (rows, elements.reshape(-1))),
        shape=(len(elements), len(points)))
    assert sp.csgraph.connected_components(
        incidence @ incidence.T, directed=False)[0] == 1

    # Face-connectivity does not.
    assert _face_components(elements)[0] == 2

    nodes, elems, report = _make_solvable(points, elements, "test")
    assert report["islands_dropped"] == 0
    assert report["flat_elements"] == 0
    assert report["pinched_elements"] == 1
    assert _face_components(elems)[0] == 1


def test_a_chunk_hinged_on_one_edge_is_also_found():
    """Two shared nodes is a hinge as well: the chunk still turns about that
    edge at zero strain. Only three shared nodes -- a whole face -- actually
    ties two tetrahedra together."""
    from psrt.fea.mesh import _face_components, _make_solvable

    points, elements = _hinge(2)
    assert _face_components(elements)[0] == 2

    nodes, elems, report = _make_solvable(points, elements, "test")
    assert report["pinched_elements"] == 1
    assert _face_components(elems)[0] == 1


def test_a_chunk_sharing_a_whole_face_is_left_alone():
    """The other side of the same test: three shared nodes is a real bond, so
    the repair must not touch it. A pinch detector that eats healthy elements
    would be worse than none."""
    from psrt.fea.mesh import _face_components, _make_solvable

    points, elements = _hinge(3)
    assert _face_components(elements)[0] == 1

    nodes, elems, report = _make_solvable(points, elements, "test")
    assert report["pinched_elements"] == 0
    assert len(elems) == 201


def test_a_large_pinched_piece_is_refused_not_deleted():
    """Half a part hinged on one node is more likely real geometry than
    meshing debris, so this refuses rather than quietly deleting it."""
    from psrt.fea.mesh import _make_solvable

    points, elements = _strip(10)
    base = len(points)
    chunk_points, chunk_elements = _strip(10, origin=(0.0, 20.0, 0.0))
    chunk_points[0] = points[5]                 # coincident: welds to a hinge
    points = np.vstack([points, chunk_points])
    elements = np.vstack([elements, chunk_elements + base])

    with pytest.raises(ValueError, match="vertices or edges"):
        _make_solvable(points, elements, "test")


def test_face_connectivity_accepts_a_real_mesh():
    """A conforming tetrahedral mesh of one solid is face-connected by
    construction, so this must not fire on the real parts -- otherwise the
    repair above would start eating geometry."""
    from psrt.fea.mesh import _face_components

    mesh = bar_mesh(30.0)
    assert _face_components(mesh.elements.T)[0] == 1


# --- the solver does not trust its first answer ----------------------------

def test_the_residual_check_accepts_a_healthy_solve():
    """The fast path has to stay the fast path: a clean mesh must be solved
    by the direct factorisation and leave no note about anything else."""
    mesh = bar_mesh(30.0)
    result = solve_linear_elastic(
        mesh, STEEL, fixed=RESTRAINT,
        traction=lambda x: x[2] > LENGTH - TOL,
        traction_vector=(0.0, 0.0, 50e6), load_case="residual check")
    assert not any("conjugate gradients" in n or "MINRES" in n
                   or "equilibration made this system worse" in n
                   for n in result.notes)


def test_equilibration_does_not_change_the_answer():
    """Symmetric diagonal scaling is an exact substitution, not an
    approximation: it changes the condition number and nothing else. If the
    two solves disagreed, the primary solver would be introducing error to
    avoid a platform bug, which is not a trade worth making."""
    from skfem import solve as skfem_solve

    from psrt.fea.solve import _equilibrated_direct, _restrained_dofs, _lame

    from skfem import (Basis, ElementTetP1, ElementVector, FacetBasis,
                       LinearForm, MeshTet, asm, condense)
    from skfem.models.elasticity import linear_elasticity

    mesh = bar_mesh(30.0)
    m = MeshTet(mesh.points, mesh.elements)
    basis = Basis(m, ElementVector(ElementTetP1()))
    stiffness = asm(linear_elasticity(*_lame(STEEL)), basis)
    facets = FacetBasis(m, basis.elem,
                        facets=m.facets_satisfying(lambda x: x[2] > LENGTH - TOL))

    @LinearForm
    def applied(v, w):
        return 50e6 * v.value[2]

    system = condense(asm(linear_elasticity(*_lame(STEEL)), basis),
                      asm(applied, facets),
                      D=_restrained_dofs(mesh, RESTRAINT))
    plain = skfem_solve(*system)
    scaled = skfem_solve(*system, solver=_equilibrated_direct)
    assert np.allclose(plain, scaled, rtol=1e-9,
                       atol=1e-12 * float(np.abs(plain).max()))


def test_equilibration_flattens_the_diagonal_it_is_there_to_flatten():
    """The mechanism, measured rather than asserted. The real connecting rod
    meshes with its smallest element about a millionth of the median volume,
    and element stiffness follows element size, so the stiffness matrix
    inherits that spread. This is the step that removes it."""
    import scipy.sparse as sp

    from psrt.fea.solve import _lame, _restrained_dofs

    from skfem import (Basis, ElementTetP1, ElementVector, MeshTet, asm)
    from skfem.models.elasticity import linear_elasticity

    mesh = bar_mesh(30.0)
    m = MeshTet(mesh.points, mesh.elements)
    basis = Basis(m, ElementVector(ElementTetP1()))
    stiffness = asm(linear_elasticity(*_lame(STEEL)), basis)

    diagonal = stiffness.diagonal()
    root = np.sqrt(diagonal)
    scaling = sp.diags(1.0 / root)
    flattened = (scaling @ stiffness @ scaling).diagonal()
    assert np.allclose(flattened, 1.0), "the point is a unit diagonal"


def test_a_singular_system_is_solved_anyway_and_the_stress_is_right():
    """The point of the MINRES rung. A stiffness matrix's null space is made
    of rigid-body motions, and a rigid-body motion has zero strain -- so a
    system that is genuinely singular still has a correct stress field, and
    the displacements are wrong only by an amount that carries no stress.

    This is what stands between a user and a part painted uniformly green
    because the direct solver returned NaN.
    """
    from psrt.fea.solve import _minres_solver

    mesh = bar_mesh(30.0)
    loaded = lambda x: x[2] > LENGTH - TOL                    # noqa: E731
    healthy = solve_linear_elastic(
        mesh, STEEL, fixed=RESTRAINT, traction=loaded,
        traction_vector=(0.0, 0.0, 50e6), load_case="reference")

    from skfem import (Basis, ElementTetP1, ElementVector, FacetBasis,
                       LinearForm, MeshTet, asm, condense)
    from skfem.models.elasticity import linear_elasticity

    from psrt.fea.solve import _lame, _restrained_dofs

    m = MeshTet(mesh.points, mesh.elements)
    basis = Basis(m, ElementVector(ElementTetP1()))
    stiffness = asm(linear_elasticity(*_lame(STEEL)), basis)
    facets = FacetBasis(m, basis.elem, facets=m.facets_satisfying(loaded))

    @LinearForm
    def applied(v, w):
        return 50e6 * v.value[2]

    load = asm(applied, facets)
    dirichlet = _restrained_dofs(mesh, RESTRAINT)
    matrix, vector, x, include = condense(stiffness, load, D=dirichlet)

    # Solved with MINRES rather than a factorisation, the stresses have to
    # agree with the healthy solve to the last decimal that matters.
    from skfem import solve as skfem_solve

    displacement = skfem_solve(matrix, vector, x, include,
                               solver=_minres_solver)
    stress, _ = element_stress(mesh, displacement.reshape((-1, 3)).T, STEEL)
    assert np.allclose(stress, healthy.von_mises, rtol=1e-6,
                       atol=1e-3 * float(np.max(healthy.von_mises)))


@pytest.mark.parametrize("part", CASE_PARTS)
def test_real_meshes_come_back_solvable(part, ls3):
    from psrt.fea import tet_mesh

    mesh = tet_mesh(ls3, part, target_elements=12_000)
    assert np.unique(mesh.elements).size == mesh.n_nodes, "orphan nodes left"
    assert mesh.elements.max() < mesh.n_nodes
