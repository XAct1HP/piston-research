"""Linear elastostatics on a tetrahedral mesh.

scikit-fem assembles the stiffness matrix and scipy solves it. Stress recovery
is done here in numpy rather than through the library, for a reason worth
stating: a four-node tetrahedron has a linear displacement field, so its
strain is CONSTANT over the element and follows in closed form from the nodal
displacements::

    u(x) = u0 + G (x - p0)        so    G = U J^-1

with ``J`` the matrix of edge vectors from node 0 and ``U`` the matching
displacement differences. From there::

    eps = (G + G^T) / 2
    sig = 2 mu eps + lambda tr(eps) I
    von Mises = sqrt(3/2 s:s),     s = sig - tr(sig)/3 I

That is exact for this element, it vectorises over the whole mesh in a few
lines, and -- more to the point -- it is short enough to check by eye. The
patch test in the test suite proves it: a bar under uniform tension comes back
at exactly F/A.

Constant-strain tetrahedra are stiff in bending, so a coarse mesh
*underestimates* peak stress. That is the wrong direction to be wrong in, which
is why every result carries a convergence check rather than a single number.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..materials import Material
from .mesh import MeshResult


@dataclass
class SolveResult:
    """A solved load case."""

    mesh: MeshResult
    displacement: np.ndarray          # (3, n_nodes), metres
    von_mises: np.ndarray             # (n_elements,), Pa
    principal: np.ndarray             # (3, n_elements), Pa, ascending
    material: str
    load_case: str
    residual: float = 0.0
    free_rigid_modes: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    @property
    def peak_stress(self) -> float:
        return float(self.von_mises.max())

    @property
    def peak_element(self) -> int:
        return int(np.argmax(self.von_mises))

    @property
    def peak_location(self) -> tuple:
        nodes = self.mesh.elements[:, self.peak_element]
        return tuple(self.mesh.points[:, nodes].mean(axis=1))

    @property
    def max_displacement(self) -> float:
        return float(np.linalg.norm(self.displacement, axis=0).max())

    def percentile_stress(self, q: float = 99.5) -> float:
        """A high percentile rather than the outright maximum.

        A single element at a re-entrant corner can report an arbitrarily
        large stress that refines to infinity and means nothing physical. The
        99.5th percentile is what to compare between meshes.
        """
        return float(np.percentile(self.von_mises, q))

    def nodal_von_mises(self) -> np.ndarray:
        """Element stresses averaged onto nodes, for display."""
        totals = np.zeros(self.mesh.n_nodes)
        counts = np.zeros(self.mesh.n_nodes)
        for corner in range(4):
            np.add.at(totals, self.mesh.elements[corner], self.von_mises)
            np.add.at(counts, self.mesh.elements[corner], 1.0)
        return totals / np.maximum(counts, 1.0)

    def as_dict(self) -> dict:
        return {
            "load_case": self.load_case,
            "material": self.material,
            "peak_von_mises_pa": self.peak_stress,
            "p99_5_von_mises_pa": self.percentile_stress(),
            "peak_location_m": list(self.peak_location),
            "max_displacement_m": self.max_displacement,
            "equilibrium_residual": self.residual,
            "free_rigid_modes": self.free_rigid_modes,
            "well_posed": not self.free_rigid_modes,
            "mesh": self.mesh.as_dict(),
            "notes": self.notes,
        }


def _lame(material: Material) -> tuple:
    e, nu = material.youngs_modulus, material.poisson
    return (e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)), e / (2.0 * (1.0 + nu)))


# A tetrahedron this much smaller than the mesh's typical element is a
# numerical artefact rather than a piece of the part. Relative, because the
# absolute size depends entirely on whether the part is a wrist pin or a
# cylinder block.
DEGENERATE_RATIO = 1e-9


def element_stress(mesh: MeshResult, displacement: np.ndarray,
                   material: Material, report: bool = False):
    """Constant strain per tetrahedron, in closed form. See the module docs.

    Tetrahedral meshers occasionally emit a sliver: an element with four
    almost-coplanar corners and a volume near zero. Its Jacobian is singular,
    ``np.linalg.solve`` returns infinities, and the infinities propagate into
    the stress tensor. The failure then surfaces a long way from its cause --
    ``numpy.linalg.LinAlgError: Eigenvalues did not converge``, raised by the
    principal-stress decomposition, which says nothing about a bad element
    and kills the whole solve.

    So slivers are found before the solve rather than after. They carry no
    stress, they are counted, and the count is reported: a handful is normal
    and means nothing, while thousands mean the mesh is unusable and the
    answer should not be believed.
    """
    stress, quality = stress_tensors(mesh, displacement, material)

    deviator = stress - (np.einsum("nii->n", stress) / 3.0)[:, None, None] * np.eye(3)
    von_mises = np.sqrt(1.5 * np.einsum("nij,nij->n", deviator, deviator))
    principal = np.sort(np.linalg.eigvalsh(stress), axis=1).T

    if report:
        return von_mises, principal, quality
    return von_mises, principal


def stress_tensors(mesh: MeshResult, displacement: np.ndarray,
                   material: Material) -> tuple:
    """The full Cauchy stress tensor per element, and a mesh-quality report.

    Separate from :func:`element_stress` because calibration needs the
    components, not just the invariants: a bending moment is the integral of
    the AXIAL stress over a section, and von Mises throws the sign away.
    """
    lam, mu = _lame(material)
    p, elements = mesh.points, mesh.elements

    p0 = p[:, elements[0]]
    j = np.stack([p[:, elements[i]] - p0 for i in (1, 2, 3)], axis=1)   # (3,3,n)
    u0 = displacement[:, elements[0]]
    u = np.stack([displacement[:, elements[i]] - u0 for i in (1, 2, 3)], axis=1)

    j_t = np.moveaxis(j, 2, 0)          # (n, 3, 3) rows = edge components
    u_t = np.moveaxis(u, 2, 0)

    # |det J| is six times the element volume. Compare against the mesh's own
    # median so the test means the same thing on any part at any scale.
    volume = np.abs(np.linalg.det(j_t))
    reference = np.median(volume)
    healthy = volume > max(reference * DEGENERATE_RATIO, 0.0)

    grad = np.zeros((len(volume), 3, 3))
    if healthy.any():
        solved = np.linalg.solve(np.swapaxes(j_t[healthy], 1, 2),
                                 np.swapaxes(u_t[healthy], 1, 2))
        grad[healthy] = np.swapaxes(solved, 1, 2)

    strain = 0.5 * (grad + np.swapaxes(grad, 1, 2))
    trace = np.einsum("nii->n", strain)
    stress = 2.0 * mu * strain + lam * trace[:, None, None] * np.eye(3)

    # Belt and braces: a healthy Jacobian can still be ill-conditioned enough
    # to produce a non-finite gradient. eigvalsh cannot decompose one of
    # those either, so they are zeroed here rather than raising later.
    finite = np.isfinite(stress).all(axis=(1, 2))
    stress[~finite] = 0.0

    discarded = int((~healthy).sum() + (healthy & ~finite).sum())
    return stress, {
        "degenerate_elements": discarded,
        "element_count": int(len(volume)),
        "min_volume_m3": float(volume.min() / 6.0),
        "median_volume_m3": float(reference / 6.0),
    }


AXIS_COMPONENT = {"x": 0, "y": 1, "z": 2}
RIGID_MODES = ("translation x", "translation y", "translation z",
               "rotation about x", "rotation about y", "rotation about z")


def _restrained_dofs(mesh: MeshResult, fixed) -> np.ndarray:
    """Degrees of freedom to hold, selected by NODE position.

    scikit-fem's ``get_dofs`` evaluates its predicate at boundary FACET
    midpoints, not at nodes. That is right for a traction and wrong for a
    restraint: a selector naming one corner node matches no facet midpoint at
    all and silently returns nothing, leaving the part free to move. This
    module selects nodes directly instead.

    The mapping is explicit. scikit-fem preserves the node order it is handed
    and lays its vector degrees of freedom out node-major, so the component
    ``c`` of node ``n`` is degree of freedom ``3n + c``. Both of those facts
    are asserted in the test suite rather than assumed here.
    """
    points = mesh.points

    def nodes_where(selector) -> np.ndarray:
        return np.flatnonzero(np.asarray(selector(points), dtype=bool))

    if callable(fixed):
        nodes = nodes_where(fixed)
        if nodes.size == 0:
            return np.array([], dtype=int)
        return np.sort(np.concatenate([3 * nodes + c for c in (0, 1, 2)]))

    collected = []
    for axis, selector in fixed.items():
        if axis not in AXIS_COMPONENT:
            raise ValueError(f"unknown axis {axis!r}; use x, y or z")
        nodes = nodes_where(selector)
        if nodes.size:
            collected.append(3 * nodes + AXIS_COMPONENT[axis])
    if not collected:
        return np.array([], dtype=int)
    return np.unique(np.concatenate(collected))


def unconstrained_rigid_modes(mesh: MeshResult, dirichlet: np.ndarray) -> list:
    """Which rigid-body motions the restraints fail to remove.

    An equilibrium residual cannot find these: a rigid-body mode satisfies
    K u = f exactly, because K times a rigid motion is zero. The system is
    still singular, the sparse solver still returns something, and the
    stresses can even come back perfectly right while the displacements are
    quietly meaningless -- which is exactly what happened the first time this
    was written.

    So test it directly. Build the six rigid motions as nodal fields and ask,
    for each, whether the restrained degrees of freedom touch it at all. A
    mode the restraints never see is a mode the solver is free to add.
    """
    points = mesh.points
    centre = points.mean(axis=1, keepdims=True)
    offset = points - centre
    n = points.shape[1]

    modes = []
    for axis in range(3):
        field = np.zeros((3, n))
        field[axis] = 1.0
        modes.append(field)
    for axis in range(3):
        omega = np.zeros(3)
        omega[axis] = 1.0
        modes.append(np.cross(np.broadcast_to(omega, offset.T.shape),
                              offset.T).T)

    free = []
    for name, field in zip(RIGID_MODES, modes):
        flat = field.T.reshape(-1)              # node-major, matching the dofs
        scale = float(np.abs(flat).max()) or 1.0
        touched = float(np.abs(flat[dirichlet]).max()) if dirichlet.size else 0.0
        if touched / scale < 1e-9:
            free.append(name)
    return free


@dataclass
class Bearing:
    """A contact pressure spread over an arc, delivering a known resultant.

    This is what replaces a rigid restraint at a reaction. A pinned node
    reacts a load through a single point, which in a continuum means an
    infinite stress that refines toward infinity: fine for holding a part
    still, useless for measuring anything near it. A real boss or bush
    presses over an arc, and the pressure falls off from the centre of
    contact roughly as the cosine of the angle -- the classical Hertzian
    line-contact distribution, close enough for a conforming journal.

    ``selector`` picks the facets. ``axis`` is the direction the load is
    delivered in. ``force`` is the total resultant wanted, in newtons; the
    distribution is assembled with a unit amplitude and then scaled so the
    resultant is exactly that, which makes the answer independent of how the
    mesher happened to facet the arc.
    """

    selector: object
    axis: tuple
    force: float
    name: str = "bearing"


def _bearing_load(m, basis, bearing: Bearing, notes: list):
    """Assemble one cosine-distributed contact pressure, scaled to resultant."""
    # skfem is imported lazily inside the solver so the package stays usable
    # without it; this helper sits outside that scope and needs its own.
    from skfem import FacetBasis, LinearForm, asm

    facets = m.facets_satisfying(bearing.selector, boundaries_only=True)
    if facets.size == 0:
        raise ValueError(
            f"the {bearing.name} selector matched no boundary facets; that "
            "load would be applied nowhere")

    facet_basis = FacetBasis(m, basis.elem, facets=facets)
    direction = np.asarray(bearing.axis, dtype=float)
    direction = direction / np.linalg.norm(direction)

    # cos(angle between the inward normal and the load direction), clipped at
    # zero so only the half of the bore that is actually being pressed
    # carries any load. w.n is the OUTWARD normal, hence the sign.
    @LinearForm
    def distributed(v, w):
        cosine = -sum(direction[i] * w.n[i] for i in range(3))
        weight = np.maximum(cosine, 0.0)
        return weight * sum(direction[i] * v[i] for i in range(3))

    load = asm(distributed, facet_basis)

    # Scale to the resultant asked for. Summing the assembled nodal forces
    # along the load direction gives what a unit amplitude delivers.
    nodal = load.reshape((-1, 3))
    delivered = float(np.sum(nodal @ direction))
    if abs(delivered) < 1e-30:
        raise ValueError(
            f"the {bearing.name} arc delivers no net force along its own "
            "axis; the selector probably spans both sides of the bore")

    scale = bearing.force / delivered
    notes.append(
        f"{bearing.name}: {bearing.force / 1e3:.2f} kN spread over "
        f"{facets.size} facets as a cosine contact pressure")
    return load * scale


RESIDUAL_TOLERANCE = 1e-8      # ||K u - f|| / ||f|| we will accept


def _solve_condensed(system, notes: list) -> np.ndarray:
    """Solve the condensed system, and do not trust the first answer.

    The direct sparse factorisation is right almost always and silently
    wrong occasionally, and "occasionally" turned out to mean "on one of the
    four parts, on one platform". The connecting rod solved on Linux and
    returned NaN on Windows, from the same source, on the same geometry.

    The cause is conditioning, and the rod earns it honestly. It is the one
    part whose mesh spans a huge range of element sizes -- the I-beam web
    meets the small-end boss at an acute angle, tetgen fills the corner with
    slivers, and the smallest element comes out around a MILLIONTH of the
    median volume. Element stiffness scales with size, so the stiffness
    matrix inherits that spread: on the rod its diagonal runs over four
    orders of magnitude, against two on the piston and one on the pin. At
    that spread the condition number is high enough that whether the
    factorisation survives depends on the pivoting choices of whichever
    SuperLU the local scipy was built against. Hence one machine, one part.

    So the first thing done here is not to solve but to RESCALE. Symmetric
    diagonal equilibration -- the substitution u = D^-1/2 y with D the
    diagonal of K -- gives an equivalent system whose diagonal is all ones.
    It is exact, not an approximation: the solution transforms straight
    back. It costs one sparse product. On the rod it takes the condition
    number to about 1.5e6, which any factorisation on any platform handles
    without noticing, and it is why the size spread stops mattering.

    Then the answer gets CHECKED rather than assumed: the relative
    equilibrium residual of the condensed system is a few flops and it
    separates a real solution from a plausible-looking one. NaN is not the
    only way a bad solve presents -- it can also come back finite and wrong,
    which no ``isfinite`` test catches and this one does. Only if the check
    fails does this climb the ladder, each rung costing more than the last:

    1. equilibrated direct factorisation -- fast, and what runs every time;
    2. the plain direct solve, in case the rescaling itself was degenerate;
    3. preconditioned conjugate gradients, which never factorises, so bad
       conditioning costs it iterations rather than accuracy;
    4. MINRES, which does not require the matrix to be positive DEFINITE,
       only symmetric. That is the one that survives a genuinely singular
       system, and it is worth having for a reason specific to elasticity:
       the null space of a stiffness matrix is made of rigid-body motions,
       and a rigid-body motion has zero strain. Whatever arbitrary amount of
       it MINRES leaves in the displacements, THE STRESSES ARE UNAFFECTED.
       A free-floating chunk makes the displacement field partly arbitrary
       and the stress field still correct.

    Which rung was needed is recorded in the notes, because it is a
    statement about mesh quality and the user should see it.
    """
    from skfem import solve as skfem_solve

    matrix, vector = system[0], system[1]
    include = system[3] if len(system) > 3 else None
    reference = float(np.linalg.norm(vector)) or 1.0

    def residual(full) -> float:
        if full is None or not np.isfinite(full).all():
            return float("inf")
        reduced = full[include] if include is not None else full
        return float(np.linalg.norm(matrix @ reduced - vector)) / reference

    def attempt(**kwargs):
        try:
            answer = skfem_solve(*system, **kwargs)
        except Exception:                                     # noqa: BLE001
            return None, float("inf")
        return answer, residual(answer)

    displacement, worst = attempt(solver=_equilibrated_direct)
    if worst <= RESIDUAL_TOLERANCE:
        return displacement

    tried = [f"equilibrated direct (residual {worst:.2e})"]

    candidate, error = attempt()
    if error <= RESIDUAL_TOLERANCE:
        notes.append(
            "diagonal equilibration made this system worse rather than "
            "better, which should not happen and is worth reporting; the "
            "plain factorisation solved it.")
        return candidate
    tried.append(f"direct factorisation (residual {error:.2e})")
    if error < worst:
        displacement, worst = candidate, error

    try:
        from skfem.utils import solver_iter_pcg

        candidate, error = attempt(
            solver=solver_iter_pcg(rtol=1e-12, maxiter=50_000))
    except TypeError:                          # older scikit-fem wants tol=
        candidate, error = attempt(
            solver=solver_iter_pcg(tol=1e-12, maxiter=50_000))
    except ImportError:
        candidate, error = None, float("inf")

    if error <= RESIDUAL_TOLERANCE:
        notes.append(
            "the direct solver did not converge on this mesh, so it was "
            "solved by preconditioned conjugate gradients instead. That "
            "points at an ill-conditioned stiffness matrix rather than a "
            "singular one: the answer is sound, but a different element "
            "count would probably mesh the part better.")
        return candidate
    tried.append(f"conjugate gradients (residual {error:.2e})")
    if error < worst:
        displacement, worst = candidate, error

    candidate, error = attempt(solver=_minres_solver)
    if error <= RESIDUAL_TOLERANCE:
        notes.append(
            "NEITHER the direct solver NOR conjugate gradients could solve "
            "this mesh, so it was solved by MINRES, which tolerates a "
            "singular stiffness matrix. Something in this mesh can move at "
            "zero strain -- most likely a chunk of elements attached to the "
            "rest by a single edge. The STRESSES here are still right, "
            "because a zero-strain motion produces no stress; the "
            "DISPLACEMENTS are right only up to that motion, so read the "
            "deflection numbers with suspicion and re-mesh at a different "
            "element count before trusting them.")
        return candidate
    tried.append(f"MINRES (residual {error:.2e})")
    if error < worst:
        displacement, worst = candidate, error

    notes.append("solvers tried: " + "; ".join(tried))
    return displacement


def _equilibrated_direct(matrix, vector, **_):
    """A direct solve of the diagonally equilibrated system.

    K u = f becomes (S K S) y = S f with S = diag(K)^-1/2 and u = S y. The
    two systems have the same solution and very different condition numbers
    whenever the mesh spans a wide range of element sizes, which is the
    situation this whole ladder exists for. Nothing is approximated.
    """
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve

    diagonal = matrix.diagonal().copy()
    bad = ~np.isfinite(diagonal) | (diagonal <= 0.0)
    if bad.all():
        raise ValueError("the stiffness matrix has no usable diagonal")
    diagonal[bad] = 1.0
    root = np.sqrt(diagonal)
    scaling = sp.diags(1.0 / root)
    scaled = (scaling @ matrix @ scaling).tocsc()
    return spsolve(scaled, vector / root) / root


def _minres_solver(matrix, vector, **_):
    """MINRES with Jacobi preconditioning, as a scikit-fem solver callable."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import minres

    diagonal = matrix.diagonal().copy()
    diagonal[~np.isfinite(diagonal) | (diagonal <= 0.0)] = 1.0
    preconditioner = sp.diags(1.0 / diagonal)
    try:
        answer, _ = minres(matrix, vector, rtol=1e-13, maxiter=100_000,
                           M=preconditioner)
    except TypeError:                              # scipy < 1.12 spells it tol
        answer, _ = minres(matrix, vector, tol=1e-13, maxiter=100_000,
                           M=preconditioner)
    return answer


def solve_linear_elastic(mesh: MeshResult, material: Material,
                         fixed, traction=None, traction_vector=None,
                         pressure=None, bearings=(),
                         load_case: str = "unnamed") -> SolveResult:
    """Solve one static load case.

    ``fixed`` restrains nodes. It is either a single predicate over an
    (3, n) array of coordinates in metres -- which pins all three directions --
    or a dict mapping "x", "y", "z" to separate predicates, which pins only
    those directions.

    The dict form matters more than it looks. Clamping all three components of
    a loaded face blocks Poisson contraction there, which invents a stress
    concentration that is a boundary condition rather than a load path. A bar
    in tension should be held in z on one face and restrained only enough
    elsewhere to remove rigid-body motion; done that way the patch test comes
    back exact, and done the obvious way it is 37% out.

    ``traction`` selects the facets to load. ``traction_vector`` is the force
    per unit area on them, in pascals, in a fixed direction; ``pressure`` is
    a scalar in pascals acting along the inward surface normal instead, which
    is what a gas load on a crown or a bore actually is. Give one or the
    other, never both.

    ``bearings`` is a sequence of :class:`Bearing`, each a cosine contact
    pressure delivering an exact resultant. Give every load AND every
    reaction as a bearing and the load set is self-equilibrated: ``fixed``
    then only has to remove rigid-body motion, carries almost no force, and
    invents no stress concentration. That is the difference between a stress
    field you can look at and one you can calibrate against.
    """
    from skfem import Basis, ElementTetP1, FacetBasis, LinearForm, MeshTet, asm
    from skfem import condense
    from skfem.models.elasticity import linear_elasticity

    try:
        from skfem import ElementVector
    except ImportError:                                   # older scikit-fem
        from skfem import ElementVectorH1 as ElementVector

    m = MeshTet(mesh.points, mesh.elements)
    basis = Basis(m, ElementVector(ElementTetP1()))
    lam, mu = _lame(material)
    stiffness = asm(linear_elasticity(lam, mu), basis)

    load = np.zeros(basis.N)
    notes = []
    if traction is not None and (traction_vector is not None
                                 or pressure is not None):
        if traction_vector is not None and pressure is not None:
            raise ValueError(
                "give either traction_vector or pressure, not both: one is a "
                "fixed direction and the other follows the surface normal")
        facets = m.facets_satisfying(traction, boundaries_only=True)
        if facets.size == 0:
            raise ValueError(
                "the traction selector matched no boundary facets; the load "
                "would be applied nowhere")
        facet_basis = FacetBasis(m, basis.elem, facets=facets)

        if pressure is not None:
            # A pressure acts along the inward normal, so its direction varies
            # facet by facet. skfem hands the outward normal to the form as
            # w.n, which is what makes this a one-line change rather than a
            # per-facet assembly.
            magnitude = float(pressure)

            @LinearForm
            def applied(v, w):
                return -magnitude * sum(w.n[i] * v[i] for i in range(3))

            notes.append(f"pressure of {magnitude / 1e6:.3f} MPa applied over "
                         f"{facets.size} boundary facets, along their normals")
        else:
            vector = np.asarray(traction_vector, dtype=float)

            @LinearForm
            def applied(v, w):
                return sum(vector[i] * v[i] for i in range(3))

            notes.append(f"traction applied over {facets.size} boundary facets")

        load = asm(applied, facet_basis)

    for bearing in bearings:
        load = load + _bearing_load(m, basis, bearing, notes)

    dirichlet = _restrained_dofs(mesh, fixed)
    if dirichlet.size == 0:
        raise ValueError(
            "the fixed selector matched no nodes, so the part is free to "
            "translate and the stiffness matrix is singular")

    # What actually got applied, before solving anything. A load that lands
    # nowhere, or lands entirely on restrained nodes, produces a perfectly
    # valid solve full of zeros -- which renders as a uniformly green part
    # and reads as "this component is fine". That is the worst failure mode
    # this tool has: a wrong answer that looks like a good one.
    nodal = load.reshape((-1, 3))
    applied = nodal.sum(axis=0)                     # net resultant
    # Total force magnitude, NOT the resultant. A pressurised tube carries a
    # large load and a resultant of exactly zero, by symmetry; testing the
    # resultant would reject the one case in this tool with an exact
    # analytical answer.
    applied_total = float(np.abs(nodal).sum())
    if applied_total < 1e-9:
        raise ValueError(
            "no load was applied: every traction and bearing matched no "
            "facets. Nothing would be stressed, so there is no result.")

    free_dofs = np.setdiff1d(np.arange(basis.N), dirichlet)
    if float(np.abs(load[free_dofs]).max() if free_dofs.size else 0.0) < 1e-12:
        raise ValueError(
            "the whole applied load sits on restrained nodes, so it is "
            "reacted before it can stress anything. The load selector and "
            "the restraint selector are picking the same region.")

    system = condense(stiffness, load, D=dirichlet)
    displacement = _solve_condensed(system, notes)

    # NaN slips through every other check in this function, which is exactly
    # why it gets its own. A singular or near-singular system returns NaN
    # displacements; NaN fails every comparison, so the "no displacement"
    # test below is False, the equilibrium residual is NaN and so never
    # exceeds its tolerance, and the stress recovery zeroes non-finite
    # tensors by design. The result is a part reported at 0 MPa everywhere
    # and painted uniformly green -- a wrong answer wearing the face of a
    # healthy one.
    if not np.isfinite(displacement).all():
        counts = ", ".join(
            f"{axis}: {int(np.asarray(sel(mesh.points), dtype=bool).sum())} nodes"
            for axis, sel in (fixed.items() if isinstance(fixed, dict)
                              else [("all", fixed)]))
        raise ValueError(
            "the solve returned non-finite displacements, so the stiffness "
            f"matrix is singular. The restraints hold {counts}. If those "
            "numbers are healthy the restraint is not the problem and the "
            "MESH is: a node no element references has an all-zero row, and "
            "a disconnected island floats free with its own rigid-body "
            "modes, and a chunk joined to the rest by a single edge can "
            "rotate about it at zero strain, any of which makes the matrix "
            "singular however well the part is held. tet_mesh removes all "
            f"three (this mesh: {mesh.orphan_nodes} orphan nodes, "
            f"{mesh.islands_dropped} islands dropped, "
            f"{mesh.pinched_elements} pinched elements, "
            f"{mesh.welded_nodes} welded nodes, {mesh.flat_elements} flat "
            f"elements, {mesh.reoriented} reoriented), and three solvers "
            "were tried on it. Re-mesh at a different element count and "
            "report these numbers.")

    field = displacement.reshape((-1, 3)).T

    repairs = [(mesh.welded_nodes, "coincident node(s) welded"),
               (mesh.flat_elements, "flat element(s) removed"),
               (mesh.reoriented, "inverted element(s) reoriented"),
               (mesh.orphan_nodes, "orphan node(s) removed"),
               (mesh.islands_dropped, "disconnected island(s) discarded"),
               (mesh.pinched_elements, "edge-pinched element(s) discarded")]
    done = [f"{count} {what}" for count, what in repairs if count]
    if done:
        notes.append("mesh repaired before solving: " + ", ".join(done)
                     + ". Any of those would have made the matrix singular.")

    notes.append(
        f"applied load: resultant ({applied[0] / 1e3:+.2f}, "
        f"{applied[1] / 1e3:+.2f}, {applied[2] / 1e3:+.2f}) kN, "
        f"{applied_total / 1e3:.2f} kN total")

    if float(np.abs(field).max()) < 1e-15:
        raise ValueError(
            f"a load of {applied_total / 1e3:.2f} kN produced no "
            "displacement anywhere. The part is restrained everywhere the "
            "load could move it, so the solve is arithmetically fine and "
            "physically meaningless.")

    # Equilibrium check. An under-constrained part leaves a rigid-body mode in
    # the system; the sparse solver will happily return SOMETHING, the stress
    # can even come back correct, and only the displacements are quietly
    # nonsense. This catches it rather than letting it through.
    loose = unconstrained_rigid_modes(mesh, dirichlet)
    if loose:
        notes.append(
            "UNDER-CONSTRAINED: the restraints leave " + ", ".join(loose)
            + " free. The stiffness matrix is singular; stresses may still "
            "look right but the displacements are meaningless. Add restraints.")

    free = free_dofs
    residual = 0.0
    if free.size:
        out_of_balance = (stiffness @ displacement - load)[free]
        scale = max(float(np.abs(load).max()), 1e-30)
        residual = float(np.abs(out_of_balance).max() / scale)
        if residual > 1e-6:
            notes.append(
                f"equilibrium residual {residual:.2e} is large. The part is "
                "probably under-constrained -- a free rigid-body mode leaves "
                "the displacements meaningless even when the stresses look "
                "right. Restrain it properly.")

    von_mises, principal, quality = element_stress(mesh, field, material,
                                                   report=True)

    # A finite, non-zero displacement field that recovers to zero stress
    # everywhere is not a result, it is a bug -- and it renders as a
    # uniformly green part, which reads as "this component is fine". The
    # guards above cover a load that lands nowhere and a solve that returns
    # NaN; this covers the remaining way to get an all-zero field.
    if float(np.nanmax(von_mises)) <= 0.0:
        raise ValueError(
            f"the solve moved the part (peak displacement "
            f"{float(np.abs(field).max()) * 1e6:.2f} um under "
            f"{applied_total / 1e3:.2f} kN) but every element recovered zero "
            f"stress. {quality['degenerate_elements']:,} of "
            f"{quality['element_count']:,} elements were rejected as "
            f"slivers; the smallest element is "
            f"{quality['min_volume_m3']:.3e} m^3 against a median of "
            f"{quality['median_volume_m3']:.3e}. Report this with those "
            "numbers -- it means the stress recovery, not the solve, is "
            "wrong.")
    bad = quality["degenerate_elements"]
    if bad:
        share = bad / max(quality["element_count"], 1)
        notes.append(
            f"{bad:,} of {quality['element_count']:,} elements were slivers "
            "with no usable Jacobian and carry no stress")
        if share > 0.01:
            notes.append(
                f"that is {share * 100:.1f}% of the mesh -- too many to "
                "ignore. Re-mesh at a different element count before "
                "believing this result.")
    return SolveResult(mesh=mesh, displacement=field, von_mises=von_mises,
                       principal=principal, material=material.key,
                       load_case=load_case, residual=residual,
                       free_rigid_modes=loose, notes=notes)
