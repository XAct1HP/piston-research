"""Finite element analysis: meshing, solving, and calibrating the fast layer.

Phase 2 sized every component with closed-form models carrying geometric
idealisations that were CALIBRATED rather than derived -- the crown's
effective support radius, the pin's support span, a notch factor per
component. They were labelled as guesses everywhere they appeared. This is
where they stop being guesses.

A note on the stack. The build plan said gmsh and CalculiX. Neither survived
contact with the target machine: CalculiX is a binary you have to go and find
per platform, and the gmsh wheel needs OpenGL libraries that a headless box
does not have. What is used instead installs with pip on Windows and needs
nothing else:

    CadQuery solid -> surface triangulation -> tetgen -> scikit-fem

The triangulation is already there -- the viewport has been drawing it since
phase 3 -- so the thing analysed is exactly the thing displayed, which was the
point of having one geometry kernel in the first place.
"""

from .mesh import MeshResult, audit_surface, surface_of, tet_mesh, weld
from .solve import (SolveResult, element_stress, solve_linear_elastic,
                    unconstrained_rigid_modes)

__all__ = ["MeshResult", "tet_mesh", "surface_of", "weld", "audit_surface",
           "SolveResult", "solve_linear_elastic", "element_stress",
           "unconstrained_rigid_modes"]

STATUS = """Meshing and solving both work, and both are validated.

  * all four parts mesh within a factor of 1.5 of a requested element count
    -- usually within 10% -- in under a second, from the raw CadQuery surface
  * bending stress matches beam theory to 1.3% on a refined mesh
  * a pressurised sleeve matches the Lame thick-cylinder solution to 1.5%,
    and the answer does not move when the mesh is refined 2.6x
  * a patch test recovers uniform tension exactly, at every refinement
  * rigid-body modes are detected, which an equilibrium residual cannot do
  * every mesh is repaired before it is solved: coincident nodes welded,
    flat elements dropped, inverted elements reoriented, orphan nodes and
    disconnected islands removed, and chunks hinged on a single vertex or
    edge discarded. That last one is the fault no other check finds: it is
    NODE-connected, so the island test passes it, and it still leaves the
    stiffness matrix singular
  * the solve is checked rather than trusted. The relative equilibrium
    residual of the condensed system decides whether the direct
    factorisation actually worked, and if it did not the solver falls back
    to conjugate gradients and then to MINRES, which tolerates a singular
    matrix. A stiffness matrix's null space is made of rigid-body motions
    and those carry no strain, so the stresses survive even when the
    displacements are only defined up to one. Which rung was used is
    reported in the result notes, because it is a statement about the mesh

Known limitation: these are constant-strain tetrahedra, so DISPLACEMENT is
stiff -- a cantilever tip reaches only 95% of the Timoshenko value at 27,000
elements, converging from below. Stress is the quantity this layer exists to
produce and stress is accurate; do not read the deflections as precise.

NOT delivered: calibrated replacements for the phase 2 idealisations. The
attempt is in calibrate.py and it failed twice, for two specific reasons
worth more than the numbers would have been. See that module, and the
"Calibration: what it found" section of the README.
"""
