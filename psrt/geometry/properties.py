"""Exact mass properties from an OpenCascade solid.

This is the point of having a geometry kernel at all. Up to phase 2 every mass
in the design state was a number somebody typed, and every structural margin
was computed from those numbers. Here they stop being estimates: volume, centre
of mass and the full inertia tensor come out of the same solid that gets
exported to STEP and meshed for FEA, so the thing analysed is the thing drawn.

Reciprocating mass is the parameter that couples the whole system together --
it sets the inertia force, which sets rod tension at overlap TDC, which is what
actually breaks connecting rods. Getting it from geometry rather than from a
guess is the single biggest accuracy improvement in this phase.

A note on the inertia tensor, because the convention is easy to get wrong and
this module got it wrong first. OpenCascade's ``MatrixOfInertia`` is
**centroidal**: translate a solid and the matrix does not change. The tensor
about the coordinate origin, which is what you need when combining parts that
sit at different places, follows from the parallel axis theorem and is added
here. ``tests/test_geometry.py`` pins the convention down by translating a box
and asserting which of the two moves.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps

from ..materials import Material


@dataclass(frozen=True)
class MassProperties:
    """Everything OpenCascade knows about a solid's distribution of matter."""

    volume: float                  # m^3
    mass: float                    # kg
    density: float                 # kg/m^3
    centre_of_mass: tuple          # (x, y, z) in m
    inertia_origin: np.ndarray     # 3x3, kg m^2, about the coordinate origin
    inertia_centroidal: np.ndarray  # 3x3, kg m^2, about the centre of mass
    material: str = ""

    @property
    def principal_moments(self) -> np.ndarray:
        """Eigenvalues of the centroidal tensor, ascending.

        Guarded because this is where a bad solid used to surface. LAPACK
        answers a tensor full of NaN with "Eigenvalues did not converge",
        which names neither the part nor the parameter that caused it.
        """
        if not np.isfinite(self.inertia_centroidal).all():
            raise ValueError(
                "this solid's inertia tensor is not finite, so it has no "
                "principal moments. The geometry is invalid at these "
                "parameters -- check the section dimensions.")
        return np.linalg.eigvalsh(self.inertia_centroidal)

    def as_dict(self) -> dict:
        return {
            "volume_m3": self.volume,
            "mass_kg": self.mass,
            "density_kg_m3": self.density,
            "centre_of_mass_m": list(self.centre_of_mass),
            "inertia_centroidal_kg_m2": self.inertia_centroidal.tolist(),
            "principal_moments_kg_m2": self.principal_moments.tolist(),
            "material": self.material,
        }


def _matrix(gprops) -> np.ndarray:
    m = gprops.MatrixOfInertia()
    return np.array([[m.Value(i, j) for j in (1, 2, 3)] for i in (1, 2, 3)])


def measure(shape, material: Material, scale: float = 1.0e-3) -> MassProperties:
    """Measure a CadQuery/OCC shape.

    Geometry is modelled in millimetres because that is what CAD kernels are
    comfortable with and what a machinist reads; ``scale`` converts to metres
    so that everything leaving this module is SI, like the rest of the tool.
    """
    solid = shape.wrapped if hasattr(shape, "wrapped") else shape
    if hasattr(solid, "val"):
        solid = solid.val().wrapped

    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(solid, props)

    volume = props.Mass() * scale ** 3          # unit density -> volume

    # A boolean that went wrong does not raise -- it returns a shape whose
    # measured volume is zero or negative. Every number derived from it is
    # meaningless, and left alone it propagates into a zero mass and an
    # inertia tensor that finally fails somewhere unrelated. Catch it here,
    # where the shape is, rather than wherever the NaN lands.
    if not np.isfinite(volume) or volume <= 0.0:
        raise ValueError(
            f"this solid measures a volume of {volume:.4g} m^3, so it is not "
            "a solid. A boolean has produced an empty or self-intersecting "
            "shape -- the parameters are outside what this geometry can be "
            "built from.")

    mass = volume * material.density

    centre = props.CentreOfMass()
    com = (centre.X() * scale, centre.Y() * scale, centre.Z() * scale)

    # OCC returns the CENTROIDAL tensor, for unit density, in model units.
    inertia_centroidal = _matrix(props) * material.density * scale ** 5

    # Parallel axis, taking the tensor out to the coordinate origin.
    r = np.array(com)
    shift = mass * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
    inertia_origin = inertia_centroidal + shift

    return MassProperties(
        volume=volume, mass=mass, density=material.density,
        centre_of_mass=com, inertia_origin=inertia_origin,
        inertia_centroidal=inertia_centroidal, material=material.key)


def combine(parts: list) -> MassProperties:
    """Combine several measured parts into one equivalent body.

    Used for the piston assembly: piston, rings, pin and retainers move
    together, so what the load chain needs is their combined mass and their
    combined centre of mass.
    """
    total_mass = sum(p.mass for p in parts)
    total_volume = sum(p.volume for p in parts)
    if total_mass <= 0.0:
        raise ValueError("cannot combine parts with no mass")

    com = tuple(
        sum(p.mass * p.centre_of_mass[i] for p in parts) / total_mass
        for i in range(3))

    inertia = np.zeros((3, 3))
    for p in parts:
        r = np.array(p.centre_of_mass) - np.array(com)
        inertia += p.inertia_centroidal + p.mass * (
            np.dot(r, r) * np.eye(3) - np.outer(r, r))

    return MassProperties(
        volume=total_volume, mass=total_mass,
        density=total_mass / total_volume,
        centre_of_mass=com, inertia_origin=inertia,
        inertia_centroidal=inertia, material="assembly")
