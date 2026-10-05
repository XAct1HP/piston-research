"""The stabilising sleeve's sheet-gyroid core: density, stiffness, printability.

The sleeve is the part of this rod concept that earns its keep, so it is the
one part that is not a solid. It is two thin solid skins -- one wrapped round
each rail channel, machined, the only surfaces that touch anything -- joined
across the web between the rails by a graded sheet-gyroid lattice, with a thin
shell on each of the two big faces. Printed in AlSi10Mg, heat treated, then
the two rail channels are machined in ONE fixture so they stay parallel, and
the sleeve slides up the rails from the big-end side and is captured when the
receiver goes on.

Why a sheet gyroid
------------------
A gyroid is a triply periodic minimal surface. Take the surface itself and
give it a thickness -- the SHEET form -- and you get a solid that is
continuous, has no flat faces to slump, is self-supporting at every angle a
powder-bed machine cares about, and is very nearly isotropic, which matters
because this web is asked for shear, not for axial stiffness along one nice
convenient direction. The network (solid) form is easier to argue about for
powder but is bending-dominated and pays 20-30% of its stiffness for that.

The one thing that can kill a printed lattice
---------------------------------------------
Trapped powder. A sealed cell full of unfused metal is dead weight that no
amount of vibration gets out and no inspection sees. The sheet gyroid's own
void is TWO interpenetrating continuous networks, neither of which is ever
sealed BY THE LATTICE -- but the skins and face shells around it can seal it,
and here they nearly do: the core is a long thin duct closed on four sides.
So :func:`evacuation` is a real check with a real verdict, not a comment, and
the face shells carry powder ports on the cell pitch by default.

Homogenisation
--------------
Meshing a 4 mm gyroid through a 110 mm sleeve is millions of elements for a
part whose job is one number -- the shear stiffness of a web. So the core is
carried as an effective continuum with Gibson-Ashby scaling,

    E*/Es = C rho^n

and the shear modulus taken from that through the effective Poisson ratio,
which keeps E*, G* and nu* consistent with each other instead of fitting three
independent power laws that quietly disagree. C and n are DESIGN STATE, not
constants, and they are marked estimated: they are the coefficients you
replace the moment you have compression coupons off your own machine, and
until you do, every lattice stiffness in this tool rests on them.
"""

from __future__ import annotations

import math

import numpy as np

# Everything below is written in terms of the level-set OFFSET c -- the
# sheet is the region |phi| < c -- rather than a thickness, because c is what
# the geometry is actually built from and thickness is a consequence of it.
#
# Relative density against offset, measured by voxel counting one unit cell
# at 320^3 and fitted. The first coefficient is the thin-sheet slope; the
# cubic term is the sheet's two faces starting to see each other through the
# cell. Good to better than half a percent from rho = 0.03 to 0.80, which is
# the whole usable range, so there is no thin-sheet caveat to carry.
#
# A first version of this module used the textbook thin-sheet area instead
# (rho = 3.091 t / a) and came out 9% heavy against the mesh it was
# generating. Measuring beats quoting.
RHO_LINEAR = 0.6435
RHO_CUBIC = 0.0159

# The gradient of the gyroid function on its own zero surface is not
# constant: |grad phi| runs from sqrt(2) k at the flattest points to
# sqrt(3) k at the saddles, with a mean near 1.529 k (same measurement).
# Since a wall is 2c / |grad phi| thick, ONE offset gives three different
# thicknesses, and which one you mean matters:
#
#   MEAN  -- what the sheet weighs, and what a drawing would call it;
#   MIN   -- the thinnest place on the wall, which is what decides whether
#            the machine can print it at all.
#
# Checking printability against the mean thickness passes lattices whose
# saddles come out 15% thinner than the machine can hold. This tool checks
# the minimum.
GRAD_MIN = math.sqrt(2.0)
GRAD_MEAN = 1.529
GRAD_MAX = math.sqrt(3.0)


# numpy renamed trapz to trapezoid in 2.0 and this tool has to open on both.
_trapz = getattr(np, "trapezoid", None) or np.trapz


class LatticeError(ValueError):
    """A lattice that cannot be printed or cannot be emptied of powder."""


# --- density and thickness ---------------------------------------------------

def density_at_offset(c):
    """Relative density of the sheet ``|phi| < c``."""
    c = np.asarray(c, dtype=float)
    return c * (RHO_LINEAR + RHO_CUBIC * c ** 2)


def offset_for_density(rho):
    """Level-set offset that gives a target relative density.

    Newton on the cubic above; three steps is already at machine precision
    over the usable range.
    """
    rho = np.asarray(rho, dtype=float)
    c = rho / RHO_LINEAR
    for _ in range(4):
        f = c * (RHO_LINEAR + RHO_CUBIC * c ** 2) - rho
        df = RHO_LINEAR + 3.0 * RHO_CUBIC * c ** 2
        c = c - f / df
    return c


def sheet_thickness(relative_density: float, cell: float, where="mean"):
    """Wall thickness at a target relative density, same units as ``cell``.

    ``where`` is "mean" for what the sheet nominally measures, or "min" for
    the thinnest place on it -- the one the machine has to be able to hold.
    """
    grad = {"mean": GRAD_MEAN, "min": GRAD_MAX, "max": GRAD_MIN}[where]
    return offset_for_density(relative_density) * cell / (grad * math.pi)


def density_for_thickness(thickness: float, cell: float, where="mean"):
    """The inverse of :func:`sheet_thickness`."""
    grad = {"mean": GRAD_MEAN, "min": GRAD_MAX, "max": GRAD_MIN}[where]
    return density_at_offset(thickness * grad * math.pi / cell)


def grading(s, rho_mid: float, rho_end: float, exponent: float):
    """Relative density at normalised half-span ``s`` in [-1, 1].

    ``s`` is measured from the sleeve's mid-length, so s = 0 is the middle and
    |s| = 1 is either end. Density runs from ``rho_mid`` to ``rho_end`` as
    |s|**exponent.

    Denser at the ENDS is not a typo. The sleeve is the shear web of a
    built-up column: the rails are the chords and the web carries the shear
    V = dM/dz of the buckled shape. For a first mode M ~ sin(pi z / L), so
    V ~ cos(pi z / L) -- largest at the ends, zero at mid-length, exactly
    opposite to where the bending moment peaks. Putting material where the
    bending is worst is the intuitive move and the wrong one for this part.
    """
    s = np.abs(np.asarray(s, dtype=float))
    return rho_mid + (rho_end - rho_mid) * np.clip(s, 0.0, 1.0) ** exponent


def mean_density(rho_mid: float, rho_end: float, exponent: float) -> float:
    """Length-averaged relative density, for mass."""
    return rho_mid + (rho_end - rho_mid) / (exponent + 1.0)


# --- the level set -----------------------------------------------------------

def level_set(x, y, z, cell):
    """The gyroid function. Zero on the minimal surface itself."""
    k = 2.0 * math.pi / cell
    return (np.sin(k * x) * np.cos(k * y)
            + np.sin(k * y) * np.cos(k * z)
            + np.sin(k * z) * np.cos(k * x))


def sheet_field(x, y, z, cell, offset):
    """Signed field whose negative region is the sheet at level-set ``offset``.

    Takes the offset, not a thickness, because the offset is what the
    geometry is; :func:`offset_for_density` is how a density becomes one.
    """
    return np.abs(level_set(x, y, z, cell)) - np.asarray(offset, dtype=float)


# --- homogenised properties --------------------------------------------------

def homogenised(material, rho: float, coefficient: float, exponent: float,
                strength_coefficient: float, strength_exponent: float,
                temperature: float | None = None) -> dict:
    """Effective properties of the sheet gyroid at relative density ``rho``.

    Returns SI. ``E`` and ``G`` are consistent with one another through the
    effective Poisson ratio rather than being fitted separately.
    """
    rho = float(np.clip(rho, 1e-4, 1.0))
    e_rel = coefficient * rho ** exponent
    # The effective Poisson ratio of a sheet gyroid drifts down from the
    # parent alloy's toward about 0.2 as it gets lighter. Small effect on the
    # shear modulus, but free to carry.
    nu = 0.20 + (material.poisson - 0.20) * rho
    e_eff = material.youngs_modulus * e_rel
    y = material.yield_at(temperature) if temperature is not None \
        else material.yield_strength
    return {
        "relative_density": rho,
        "modulus_ratio": e_rel,
        "youngs_modulus": e_eff,
        "poisson": nu,
        "shear_modulus": e_eff / (2.0 * (1.0 + nu)),
        "density": material.density * rho,
        "yield_strength": ((y or 0.0) * strength_coefficient
                           * rho ** strength_exponent) or None,
    }


def effective_shear(material, rho_mid, rho_end, exponent, coefficient,
                    modulus_exponent, span=None, rod_length=None,
                    samples: int = 201) -> float:
    """One shear modulus for an axially graded web, energy-consistent.

    The built-up column formula wants a single shear stiffness, but the web's
    density varies along its length and so does the shear it carries. The
    honest reduction is to match the stored shear energy, not to average the
    modulus:

        1/G_eff = integral( V^2 / G(z) ) / integral( V^2 )

    a compliance average weighted by where the shear actually is. This is the
    whole reason end-grading pays at all: a flat average cannot see it, and
    the design would look no better for putting material where it works.

    ``span`` is (z0, z1), the sleeve's ends measured along the ROD, and
    ``rod_length`` its pin-to-pin length, in the same units. Given those, the
    weight is the real V(z)^2 = cos(pi z / L)^2 of the first buckling mode
    over the piece of the rod this sleeve actually covers. Without them it
    falls back to treating the sleeve as if it spanned the whole rod.

    One asymmetry worth knowing about: the sleeve does not sit centred on the
    rod -- it runs from the big end up to the eye -- so the shear at its two
    ends is not equal, while ``grading`` is symmetric about its mid-length.
    An asymmetric grading would do better still, and nothing here stops one
    being fitted later.
    """
    s = np.linspace(-1.0, 1.0, samples)
    rho = grading(s, rho_mid, rho_end, exponent)
    if span is not None and rod_length:
        z0, z1 = span
        z = 0.5 * (z0 + z1) + 0.5 * (z1 - z0) * s
        weight = np.cos(math.pi * np.asarray(z, dtype=float)
                        / float(rod_length)) ** 2
    else:
        # Sleeve treated as spanning the rod: V ~ cos(pi z / L) in rod
        # coordinates is sin(pi s / 2) in half-span coordinates -- zero at
        # mid-length, largest at the ends.
        weight = np.sin(0.5 * math.pi * s) ** 2
    nu = 0.20 + (material.poisson - 0.20) * rho
    g = material.youngs_modulus * coefficient * rho ** modulus_exponent \
        / (2.0 * (1.0 + nu))
    return float(_trapz(weight, s) / _trapz(weight / g, s))


# --- manufacture -------------------------------------------------------------

def printability(rho_mid, rho_end, exponent, cell, min_wall) -> dict:
    """Is the thinnest sheet in the graded core actually printable?

    The binding station is wherever the density is LOWEST, which with
    end-grading is mid-length. Checked there, not at the average.
    """
    rho_low = min(rho_mid, rho_end)
    rho_high = max(rho_mid, rho_end)
    return {
        "min_relative_density": rho_low,
        "max_relative_density": rho_high,
        # the thinnest place on the thinnest sheet: what the machine must hold
        "min_wall_mm": float(sheet_thickness(rho_low, cell, "min")),
        # what the same sheet nominally measures
        "nominal_sheet_mm": float(sheet_thickness(rho_low, cell, "mean")),
        "heaviest_sheet_mm": float(sheet_thickness(rho_high, cell, "mean")),
        "machine_min_wall_mm": min_wall,
        "printable": float(sheet_thickness(rho_low, cell, "min")) >= min_wall,
        "cell_mm": cell,
    }


def cells_across(core_width, core_depth, cell) -> dict:
    """How many unit cells the core is wide, which bounds homogenisation.

    Below about four cells across, a lattice stops behaving like the
    continuum the effective properties describe and starts behaving like the
    handful of struts it actually is, stiffer in some places and softer in
    others than any average says.
    """
    nx = core_width / cell
    ny = core_depth / cell
    return {"across_mm": core_width, "through_mm": core_depth,
            "cells_across": nx, "cells_through": ny,
            "homogenisation_valid": min(nx, ny) >= 4.0}


def evacuation(core_width, core_depth, core_length, cell, rho,
               ports: bool, port_pitch: float, port_diameter: float) -> dict:
    """Can the unfused powder get out, and how far does it have to travel?

    The sheet gyroid's void is two interpenetrating continuous networks, so
    nothing is ever sealed by the lattice itself. What can seal it is the
    solid around it. With the two rail-channel skins on the sides and a shell
    on each face, the core is closed on four sides and drains only along its
    own length -- for this sleeve, over a hundred millimetres. Ports through
    the face shells on the cell pitch cut that to half the shell spacing.

    The aperture reported is the narrowest gap the powder has to pass, taken
    as the gyroid's own channel width at this density; below roughly ten times
    the powder's particle size (so about 0.5 mm for a 30-50 micron aluminium
    cut) it stops flowing and starts bridging.
    """
    # Channel width of the void between sheets, from the cell pitch less the
    # two sheet walls it passes between.
    aperture = cell * (1.0 - rho) / 2.0
    if ports:
        path = max(port_pitch, core_depth / 2.0) if port_diameter > 0 else \
            core_length / 2.0
        path = min(core_depth / 2.0 + port_pitch, core_length / 2.0)
        route = (f"out through the face-shell ports on a {port_pitch:.1f} mm "
                 "pitch")
    else:
        path = core_length / 2.0
        route = ("along the core to the sleeve's open ends -- the face shells "
                 "close it on both faces")
    return {
        "aperture_mm": aperture,
        "path_mm": path,
        "path_over_aperture": path / max(aperture, 1e-6),
        "route": route,
        "ports": ports,
        "port_diameter_mm": port_diameter if ports else 0.0,
        # Two independent criteria: the powder has to fit through, and it has
        # to be shaken out over a path that is not absurd relative to that gap.
        "aperture_ok": aperture >= 0.5,
        "path_ok": path / max(aperture, 1e-6) <= 60.0,
    }


def mesh(core, cell, rho_mid, rho_end, exponent, resolution=0.25):
    """A triangle mesh of the graded sheet gyroid, clipped to the core box.

    ``core`` is (x0, x1, y0, y1, z0, z1) in mm. Returns (vertices, faces).

    This is what actually gets printed, and it is a mesh rather than a B-rep
    on purpose: a gyroid has no analytic faces worth handing to a solid
    kernel, and every powder-bed machine takes a mesh anyway. The skins and
    shells stay exact solids and are exported separately.
    """
    try:
        from skimage.measure import marching_cubes
    except ImportError as exc:                 # pragma: no cover - install
        raise LatticeError(
            "the lattice mesh needs scikit-image: "
            "python -m pip install scikit-image") from exc

    x0, x1, y0, y1, z0, z1 = core
    nx = max(8, int(math.ceil((x1 - x0) / resolution)))
    ny = max(8, int(math.ceil((y1 - y0) / resolution)))
    nz = max(8, int(math.ceil((z1 - z0) / resolution)))
    xs = np.linspace(x0, x1, nx)
    ys = np.linspace(y0, y1, ny)
    zs = np.linspace(z0, z1, nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

    mid = 0.5 * (z0 + z1)
    half = max(0.5 * (z1 - z0), 1e-9)
    rho = grading((Z - mid) / half, rho_mid, rho_end, exponent)
    field = sheet_field(X, Y, Z, cell, offset_for_density(rho))

    # Close the field off at the box faces so marching cubes returns a
    # watertight surface rather than one that runs out of the grid.
    field[0, :, :] = field[-1, :, :] = 1.0
    field[:, 0, :] = field[:, -1, :] = 1.0
    field[:, :, 0] = field[:, :, -1] = 1.0

    verts, faces, _, _ = marching_cubes(field, level=0.0,
                                        spacing=(xs[1] - xs[0],
                                                 ys[1] - ys[0],
                                                 zs[1] - zs[0]))
    verts = verts + np.array([x0, y0, z0])
    return verts, faces
