"""Turning FEA runs into the constants the fast layer uses.

This is the part of the FEA layer that was always the point. Phase 2 sized
every component with closed forms carrying geometric idealisations that were
CALIBRATED rather than derived -- an effective support radius for the crown,
an effective support span for the pin -- chosen so that production hardware
landed just above its limits. They were labelled as guesses everywhere they
appeared, and the optimiser now warns whenever an optimum rests on one, which
on a real engine it usually does. This module replaces the guesses.

The method matters more than the numbers, because the obvious method does not
work. Comparing a whole-part FEA peak against a closed form gives a ratio
dominated by whatever boundary condition was applied: the peak element sits
under the restraint, where a rigid support invents a singularity that refines
toward infinity. Calibrating against that is calibrating against the analyst.

So two things change. Loads and REACTIONS are both applied as distributed
contact pressures, leaving the restraint carrying almost nothing (see
:class:`psrt.fea.solve.Bearing`). And the comparison is made on an integral
rather than a point: the bending moment the part actually develops, obtained
by integrating axial stress over cross-sections. A moment is insensitive to
local stress concentrations in a way a peak stress is not, which is exactly
the property a calibration needs.

Each fit is run across a range of geometries, not just the one engine, and
what comes back is a value AND the envelope it was fitted over. The fast
layer can then carry FEA-grade accuracy where it was fitted and say "this is
extrapolation, run the FEA" where it was not.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .solve import stress_tensors

__all__ = ["MomentDistribution", "bending_moment", "Fit",
           "pin_span_bracket", "fit_crown_radius", "crown_plate_moment"]

AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


@dataclass
class MomentDistribution:
    """Bending moment along a part's length, from the stress field."""

    station: np.ndarray        # m, position along the beam axis
    moment: np.ndarray         # N m, about the transverse axis
    axial_force: np.ndarray    # N, net axial force on each section
    axis: str = "y"

    @property
    def peak(self) -> float:
        return float(np.max(np.abs(self.moment)))

    @property
    def peak_station(self) -> float:
        return float(self.station[int(np.argmax(np.abs(self.moment)))])

    def as_dict(self) -> dict:
        return {"axis": self.axis, "station_m": self.station.tolist(),
                "moment_nm": self.moment.tolist(),
                "peak_moment_nm": self.peak,
                "peak_station_m": self.peak_station}


def bending_moment(mesh, displacement, material, axis: str = "y",
                   bending: str = "z", stations: int = 41
                   ) -> MomentDistribution:
    """Integrate axial stress over cross-sections to get M along ``axis``.

    For a section at station s, M = integral of sigma_axial * r dA, where r is
    the distance from the section's own centroid along ``bending``. Slicing a
    tetrahedral mesh exactly would mean clipping every element; instead each
    station takes a slab of elements and divides by the slab thickness, which
    converges to the same integral and is far simpler to get right.

    The section centroid is recomputed per station rather than assumed at the
    origin: a part is not obliged to be symmetric about the axis it bends
    about, and getting that wrong puts a spurious P*e term into every moment.
    """
    a = AXIS_INDEX[axis]
    b = AXIS_INDEX[bending]

    stress, _ = stress_tensors(mesh, displacement, material)
    corners = mesh.points.T[mesh.elements.T]
    centroid = corners.mean(axis=1)
    volume = np.abs(np.einsum(
        "ij,ij->i", corners[:, 1] - corners[:, 0],
        np.cross(corners[:, 2] - corners[:, 0],
                 corners[:, 3] - corners[:, 0]))) / 6.0

    axial = stress[:, a, a]
    along = centroid[:, a]
    lo, hi = along.min(), along.max()

    edges = np.linspace(lo, hi, stations + 1)
    width = (hi - lo) / stations
    station = 0.5 * (edges[:-1] + edges[1:])

    moment = np.zeros(stations)
    force = np.zeros(stations)
    for i in range(stations):
        inside = (along >= edges[i]) & (along < edges[i + 1])
        if not inside.any():
            continue
        v = volume[inside]
        area = v / width                      # dA for this slab
        offset = centroid[inside, b]
        # Area-weighted centroid of THIS section, not the part's.
        neutral = float(np.sum(offset * area) / max(np.sum(area), 1e-30))
        moment[i] = float(np.sum(axial[inside] * (offset - neutral) * area))
        force[i] = float(np.sum(axial[inside] * area))

    return MomentDistribution(station=station, moment=moment,
                              axial_force=force, axis=axis)


@dataclass
class Fit:
    """One calibrated constant, with what it was fitted over."""

    name: str
    value: float
    previous: float
    samples: list = field(default_factory=list)
    envelope: dict = field(default_factory=dict)
    spread: float = 0.0
    notes: list = field(default_factory=list)

    @property
    def change_percent(self) -> float:
        if not self.previous:
            return float("nan")
        return (self.value - self.previous) / abs(self.previous) * 100.0

    def as_dict(self) -> dict:
        return {"name": self.name, "value": self.value,
                "previous": self.previous,
                "change_percent": self.change_percent,
                "samples": list(self.samples), "envelope": dict(self.envelope),
                "spread_percent": self.spread, "notes": list(self.notes)}

    def summary(self) -> str:
        lines = [f"{self.name}: {self.previous:.4g} -> {self.value:.4g} "
                 f"({self.change_percent:+.1f}%)",
                 f"  fitted over {len(self.samples)} geometries, "
                 f"spread {self.spread:.1f}%"]
        for key, (low, high) in self.envelope.items():
            lines.append(f"  valid for {key} in [{low:.4g}, {high:.4g}]")
        for note in self.notes:
            lines.append(f"  ! {note}")
        return "\n".join(lines)


def _pin_case(state, target_elements: int):
    """The pin, loaded and reacted entirely through contact pressures."""
    from .. import materials as materials_mod
    from ..evaluate import evaluate
    from .mesh import tet_mesh
    from .solve import Bearing, solve_linear_elastic

    metrics = evaluate(state)
    sweep = metrics.sweep
    force = abs(float(sweep.peak_pin_compression.value))

    radius = state["pin.outer_diameter"] / 2.0
    half_contact = state["small_end.bushing_width"] / 2.0
    half_gap = state["piston.boss_inner_span"] / 2.0
    half_length = state["pin.length"] / 2.0
    surface = radius * 0.90

    mesh = tet_mesh(state, "pin", target_elements=target_elements)

    def loaded(x):
        return ((np.abs(x[1]) <= half_contact) & (x[2] > 0)
                & (np.hypot(x[0], x[2]) > surface))

    def boss(sign):
        def selector(x):
            side = x[1] * sign
            return ((side >= half_gap) & (side <= half_length) & (x[2] < 0)
                    & (np.hypot(x[0], x[2]) > surface))
        return selector

    # Only enough restraint to remove rigid-body motion. The load set already
    # sums to zero, so these carry almost nothing.
    def anchor(x):
        return (np.abs(x[1]) < half_length * 0.08) & (x[2] < -radius * 0.98)

    result = solve_linear_elastic(
        mesh, materials_mod.get(state["materials.pin"]),
        fixed={"z": anchor,
               "y": lambda x: anchor(x) & (x[0] > 0),
               "x": lambda x: anchor(x) & (x[0] > 0)},
        bearings=[
            Bearing(loaded, (0.0, 0.0, -1.0), force, "small end"),
            Bearing(boss(-1.0), (0.0, 0.0, 1.0), force / 2.0, "left boss"),
            Bearing(boss(1.0), (0.0, 0.0, 1.0), force / 2.0, "right boss"),
        ],
        load_case=f"pin, self-equilibrated, {force / 1e3:.1f} kN")
    return mesh, result, force


def pin_span_bracket(state, target_elements: int = 18_000,
                     fractions=(1 / 3, 2 / 3, 1.0)) -> dict:
    """What ``pin.support_span_factor`` could be, and why FEA cannot settle it.

    This started out as a fit and became a bracket, which is the more honest
    thing for it to be.

    The fast layer models the pin as a simply supported beam, M = W(2S - c)/8,
    where S is an effective span set by where the boss reaction acts. The
    plan was to solve the pin with distributed contact pressures instead of
    rigid restraints, extract the bending moment, and invert for S.

    That works, and the answer is meaningless. A linear analysis takes the
    contact pressure distribution as an INPUT, and the resultant of a
    prescribed distribution sits at its own centroid -- so the "fitted" span
    comes back as wherever the analyst chose to put the pressure. Spreading
    the reaction over the inner third of the boss returns a factor of 0.32;
    over the whole boss width, 1.04. The method reproduces its own
    assumption to three figures.

    Settling it needs a contact solution: the pin bends away from the boss
    outboard of its inner edge, so the contact patch is part of the ANSWER
    and shrinks under load. That is a Signorini problem, not a linear solve,
    and it is not in this tool.

    So this returns the bracket instead -- the span, and the bending stress,
    for a reaction concentrated near the boss inner edge through to one
    spread over the whole boss. The truth is inside it. A number with an
    honest range beats a false constant.
    """
    from .. import materials as materials_mod
    from ..evaluate import evaluate
    from .mesh import tet_mesh
    from .solve import Bearing, solve_linear_elastic

    metrics = evaluate(state)
    force = abs(float(metrics.sweep.peak_pin_compression.value))

    radius = state["pin.outer_diameter"] / 2.0
    half_contact = state["small_end.bushing_width"] / 2.0
    half_gap = state["piston.boss_inner_span"] / 2.0
    half_length = state["pin.length"] / 2.0
    boss_width = half_length - half_gap
    surface = radius * 0.90

    mesh = tet_mesh(state, "pin", target_elements=target_elements)
    material = materials_mod.get(state["materials.pin"])

    def loaded(x):
        return ((np.abs(x[1]) <= half_contact) & (x[2] > 0)
                & (np.hypot(x[0], x[2]) > surface))

    def anchor(x):
        return (np.abs(x[1]) < half_length * 0.08) & (x[2] < -radius * 0.98)

    rows = []
    for fraction in fractions:
        outer = half_gap + boss_width * fraction

        def boss(sign, outer=outer):
            def selector(x):
                side = x[1] * sign
                return ((side >= half_gap) & (side <= outer) & (x[2] < 0)
                        & (np.hypot(x[0], x[2]) > surface))
            return selector

        result = solve_linear_elastic(
            mesh, material,
            fixed={"z": anchor,
                   "y": lambda x: anchor(x) & (x[0] > 0),
                   "x": lambda x: anchor(x) & (x[0] > 0)},
            bearings=[
                Bearing(loaded, (0.0, 0.0, -1.0), force, "small end"),
                Bearing(boss(-1.0), (0.0, 0.0, 1.0), force / 2.0, "left"),
                Bearing(boss(1.0), (0.0, 0.0, 1.0), force / 2.0, "right"),
            ],
            load_case=f"pin, reaction over {fraction:.0%} of the boss")

        distribution = bending_moment(mesh, result.displacement, material,
                                      axis="y", bending="z")
        span = (8.0 * distribution.peak / force + 2.0 * half_contact) / 2.0
        rows.append({
            "reaction_over": fraction,
            "moment_nm": distribution.peak,
            "span_m": span,
            # Matches the schema: bending_span = boss_gap + factor * boss_width,
            # where boss_width = (pin.length - boss_gap) / 2, one boss.
            "factor": (span - 2.0 * half_gap) / max(boss_width, 1e-12),
        })

    factors = [r["factor"] for r in rows]
    moments = [r["moment_nm"] for r in rows]
    return {
        "parameter": "pin.support_span_factor",
        "current": float(state["pin.support_span_factor"]),
        "bracket": (float(min(factors)), float(max(factors))),
        "moment_bracket_nm": (float(min(moments)), float(max(moments))),
        "stress_spread_percent": float(
            (max(moments) - min(moments)) / min(moments) * 100.0),
        "rows": rows,
        "verdict": (
            "not fitted: a linear analysis returns the contact distribution "
            "it was given, so this method cannot determine where the boss "
            "reaction acts. The bracket is real; the midpoint is not a "
            "measurement."),
    }


def _pin_material(state):
    from .. import materials as materials_mod
    return materials_mod.get(state["materials.pin"])


# --------------------------------------------------------------------------
# the crown
# --------------------------------------------------------------------------

def crown_plate_moment(mesh, displacement, material, thickness: float,
                       patch_fraction: float = 0.18) -> dict:
    """Radial bending moment per unit width at the middle of the crown.

    Measured at the CENTRE on purpose. The crown meets the ring-belt wall at
    a sharp re-entrant corner with no fillet, which is a stress singularity:
    the peak there grows without limit as the mesh refines, so anything
    calibrated against it is calibrated against the element size. The centre
    of a plate has no such feature, and plate theory relates the two
    perfectly well.

    Returns the moment per unit width, and the patch it was measured over so
    the caller can check it was big enough to be meaningful and small enough
    to still be "the centre".
    """
    stress, _ = stress_tensors(mesh, displacement, material)
    corners = mesh.points.T[mesh.elements.T]
    centroid = corners.mean(axis=1)
    volume = np.abs(np.einsum(
        "ij,ij->i", corners[:, 1] - corners[:, 0],
        np.cross(corners[:, 2] - corners[:, 0],
                 corners[:, 3] - corners[:, 0]))) / 6.0

    top = float(mesh.points[2].max())
    radius = np.hypot(centroid[:, 0], centroid[:, 1])
    patch = float(np.max(radius) * patch_fraction)

    inside = (radius <= patch) & (centroid[:, 2] >= top - thickness)
    if inside.sum() < 20:
        raise ValueError(
            f"only {int(inside.sum())} elements in the crown centre patch; "
            "mesh finer or widen patch_fraction before trusting a moment "
            "measured from it")

    # At the axis the section is axisymmetric, so sigma_xx is the radial
    # bending stress. Both in-plane normal components are averaged, which
    # halves the mesh noise for free.
    axial = 0.5 * (stress[inside, 0, 0] + stress[inside, 1, 1])
    z = centroid[inside, 2]
    v = volume[inside]

    neutral = float(np.sum(z * v) / np.sum(v))
    area = float(np.pi * patch ** 2)
    moment = float(np.sum(axial * (z - neutral) * v) / area)

    return {"moment_per_width_nm_m": moment, "patch_radius_m": patch,
            "elements": int(inside.sum()), "neutral_z_m": neutral}


def _crown_case(state, target_elements: int):
    """The crown under gas pressure, reacted through the pin bore."""
    from .. import materials as materials_mod
    from ..evaluate import evaluate
    from .mesh import tet_mesh
    from .solve import Bearing, solve_linear_elastic

    metrics = evaluate(state)
    pressure = float(np.max(metrics.sweep.pressure))

    bore_radius = state["engine.bore"] / 2.0
    comp_height = state["piston.compression_height"]
    pin_radius = state["pin.outer_diameter"] / 2.0
    half_pin = state["pin.length"] / 2.0
    force = pressure * np.pi * bore_radius ** 2

    mesh = tet_mesh(state, "piston", target_elements=target_elements)
    top = float(mesh.points[2].max())

    def crown(x):
        return x[2] > top - 1e-4

    def bore(x):
        radial = np.hypot(x[0], x[2] + comp_height)
        return ((radial < pin_radius * 1.12) & (np.abs(x[1]) <= half_pin)
                & (x[2] > -comp_height))

    # Anchor on a BOSS, not at the middle of the bore: the space between the
    # bosses is cut away for the rod's small end to swing through, so there
    # is no material there to hold on to.
    #
    # The band is sized against the mesh rather than guessed. A fixed 2 mm
    # strip caught zero nodes at 30,000 elements and the solve failed with
    # "the fixed selector matched no nodes", which is a true statement about
    # a mistake made three lines earlier.
    half_gap = state["piston.boss_inner_span"] / 2.0

    def band(fraction):
        outer = half_gap + (half_pin - half_gap) * fraction

        def selector(x):
            return bore(x) & (x[1] > half_gap) & (x[1] < outer)
        return selector

    anchor = None
    for fraction in (0.15, 0.25, 0.40, 0.60, 1.0):
        candidate = band(fraction)
        if int(candidate(mesh.points).sum()) >= 12:
            anchor = candidate
            break
    if anchor is None:
        raise ValueError(
            "could not find enough nodes on a pin boss to anchor the crown "
            "solve; mesh finer")

    result = solve_linear_elastic(
        mesh, materials_mod.get(state["materials.piston"]),
        fixed={"z": anchor,
               "y": lambda x: anchor(x) & (x[0] > 0),
               "x": lambda x: anchor(x) & (x[0] > 0)},
        traction=crown, pressure=pressure,
        bearings=[Bearing(bore, (0.0, 0.0, 1.0), force, "pin bore")],
        load_case=f"crown, {pressure / 1e6:.1f} MPa")
    return mesh, result, pressure


def fit_crown_radius(state, geometries=None, target_elements: int = 30_000,
                     verbose: bool = False) -> Fit:
    """Fit ``piston.crown_support_radius_fraction`` from the centre moment.

    Unlike the pin, this one is genuinely measurable. The crown's support is
    its own structure -- the ring belt and the wall below it, all present in
    the mesh -- rather than a contact whose distribution has to be assumed.
    So the FEA determines the effective support radius instead of returning
    whatever was prescribed.

    For a clamped circular plate of radius a under uniform pressure q, the
    radial moment per unit width at the centre is

        M_c = q a^2 (1 + nu) / 16

    which inverts for a. What that a absorbs, besides the real support
    radius, is the difference between a truly clamped edge and the partial
    restraint a ring belt actually gives -- which is exactly what a
    calibrated idealisation is for, provided it is said out loud.
    """
    from .. import materials as materials_mod

    samples = []
    for changes in (geometries or [{}]):
        trial = (state.with_changes(changes, actor="calibration",
                                    rationale="crown fit")
                 if changes else state)
        material = materials_mod.get(trial["materials.piston"])
        thickness = trial["piston.crown_thickness"]
        try:
            mesh, result, pressure = _crown_case(trial, target_elements)
            measured = crown_plate_moment(mesh, result.displacement,
                                          material, thickness)
        except Exception as exc:                              # noqa: BLE001
            samples.append({"changes": changes, "error": str(exc)[:120]})
            continue

        moment = measured["moment_per_width_nm_m"]
        if moment <= 0:
            samples.append({"changes": changes,
                            "error": f"centre moment came out {moment:.3g}"})
            continue

        radius = float(np.sqrt(16.0 * moment
                               / (pressure * (1.0 + material.poisson))))
        fraction = radius / (trial["engine.bore"] / 2.0)
        samples.append({
            "changes": changes, "moment_per_width_nm_m": moment,
            "effective_radius_m": radius, "fraction": fraction,
            "crown_thickness_m": thickness,
            "elements": int(mesh.n_elements),
        })
        if verbose:
            print(f"  {changes or 'base'}: M_c={moment:8.2f} N  "
                  f"a={radius * 1000:6.2f} mm  fraction={fraction:.4f}")

    good = [s for s in samples if "fraction" in s]
    if not good:
        raise ValueError("no crown geometry produced a usable centre moment")

    fractions = np.array([s["fraction"] for s in good])
    value = float(np.median(fractions))
    spread = float((fractions.max() - fractions.min())
                   / max(abs(value), 1e-12) * 100.0)

    notes = [
        "fitted from the centre moment, not the edge stress: the crown-to-"
        "wall corner has no fillet in this geometry and is a singularity, so "
        "an edge-stress fit would be a fit to the element size",
        "the fitted radius absorbs the difference between a clamped edge and "
        "the partial restraint a ring belt really gives, as well as the "
        "support radius itself",
    ]
    if spread > 20.0:
        notes.append(
            f"varies {spread:.0f}% across the geometries tried, so one "
            "constant describes it poorly")

    envelope = {}
    for key in ("piston.crown_thickness", "engine.bore",
                "piston.wall_thickness"):
        values = []
        for s in good:
            trial = (state.with_changes(s["changes"], actor="calibration",
                                        rationale="envelope")
                     if s["changes"] else state)
            values.append(trial[key])
        envelope[key] = (float(min(values)), float(max(values)))

    return Fit(name="piston.crown_support_radius_fraction", value=value,
               previous=float(state["piston.crown_support_radius_fraction"]),
               samples=good, envelope=envelope, spread=spread, notes=notes)


def crown_deflection_profile(state, target_elements: int = 45_000) -> dict:
    """The crown's deflected shape, which is what settles the plate question.

    The clamped-plate idealisation says the crown dishes DOWN at the centre
    and is held at its rim. This measures which way it actually goes.

    On the LS3 geometry the answer is the other way round: the centre rises
    while the rim falls. The pin boss pad, which the geometry builder carries
    up to the crown underside, is over half the bore wide, so the crown
    centre is SUPPORTED rather than spanning. A clamped plate of some
    effective radius is not a mildly inaccurate description of that; it is
    the wrong shape, and a constant fitted to it is holding together a model
    whose deflection is inverted relative to the part.
    """
    from .mesh import tet_mesh  # noqa: F401  (documents the dependency)

    mesh, result, pressure = _crown_case(state, target_elements)
    points = mesh.points
    top = float(points[2].max())
    face = np.abs(points[2] - top) < 5e-4

    radius = np.hypot(points[0][face], points[1][face])
    deflection = result.displacement[2][face]
    bore_radius = state["engine.bore"] / 2.0

    edges = np.linspace(0.0, float(radius.max()), 13)
    rows = []
    for i in range(len(edges) - 1):
        inside = (radius >= edges[i]) & (radius < edges[i + 1])
        if inside.sum() < 3:
            continue
        rows.append({
            "r_over_R": float((edges[i] + edges[i + 1]) / 2.0 / bore_radius),
            "deflection_m": float(deflection[inside].mean()),
            "nodes": int(inside.sum()),
        })

    centre = rows[0]["deflection_m"] if rows else 0.0
    rim = rows[-1]["deflection_m"] if rows else 0.0
    pad = (state["pin.outer_diameter"]
           * state["piston.boss_width_factor"] / 2.0)

    return {
        "pressure_pa": pressure,
        "profile": rows,
        "centre_deflection_m": centre,
        "rim_deflection_m": rim,
        "boss_pad_fraction_of_radius": float(pad / bore_radius),
        "dishes_like_a_plate": bool(centre < rim),
        "verdict": (
            "the crown centre deflects the OPPOSITE way to a plate under "
            "pressure: the boss pad supports it and the rim falls away "
            "around it. The clamped-plate model has the wrong shape for this "
            "geometry, so no effective radius fitted to it means anything."
            if centre > rim else
            "the crown dishes at the centre as a plate does, so an effective "
            "support radius is a meaningful thing to fit"),
    }
