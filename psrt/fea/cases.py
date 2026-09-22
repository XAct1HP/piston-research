"""Load cases: real boundary conditions for real components.

Each case answers the same question the analytical layer answers, on the same
geometry, under the same load from the same crank-angle sweep -- so the two
numbers are directly comparable and their ratio means something.

The pin is first because it is the cleanest. It is a tube in three-point
bending: the rod's small end pushes on the middle, the two bosses react at the
ends. Both the load and the supports act on the outside surface over known
widths, so there is nothing to invent. It is also the component currently
binding on the LS3, at a safety factor of 1.24.

**The whole cycle from one solve.** Each case solves at the crank angle that
governs it, and then carries the load's whole crank-angle history. Because
the analysis is LINEAR and each case has a single load pattern -- one
pressure, or one force in one direction -- the stress everywhere scales
exactly with the load. So the field at any other crank angle is the solved
field times load(theta)/load(reference). That is not an approximation: for
linear elasticity with a fixed load distribution it is the answer, and it
costs one solve instead of three hundred.

It does carry one real assumption. When the load REVERSES -- the pin goes
into tension at overlap TDC, the rod likewise -- the part bears on the
opposite side of its contact, so the true boundary conditions are mirrored
rather than negated. The magnitude is right and the distribution is mirrored
from what is actually solved. For the tensile half of the cycle, which is an
order of magnitude lighter than firing, that is a reasonable place to stop;
for a case where tension governed it would not be.

A caution that applies to every case here. The ratio each one reports is NOT
a stress concentration factor yet. The peak element usually sits directly
under an applied restraint, where a rigid boundary condition invents a
singularity that refines toward infinity, and even the 99.5th percentile
carries whatever contact idealisation the case assumed. Reading these ratios
straight into the fast layer would be calibrating against the boundary
conditions rather than against the part.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import materials as materials_mod
from ..evaluate import evaluate
from .mesh import MeshResult, tet_mesh
from .solve import Bearing, SolveResult, solve_linear_elastic


# Where each component actually runs, so its allowable is the hot one rather
# than the room-temperature catalogue figure. An aluminium crown at 300 C has
# lost a third of its yield strength, and a stress map drawn against the cold
# number would look comfortable when it is not.
SERVICE_TEMPERATURE = {
    "piston": "crown_underside_c",
    "pin": "pin_c",
    "rod": "rod_c",
    "sleeve": "liner_at_tdc_c",
}


def allowable_for(state, part: str, metrics=None) -> dict:
    """The material's yield strength where this part actually runs."""
    from ..evaluate import evaluate

    metrics = metrics if metrics is not None else evaluate(state)
    key = {"piston": "materials.piston", "pin": "materials.pin",
           "rod": "materials.rod", "sleeve": "materials.sleeve"}[part]
    material = materials_mod.get(state[key])

    # as_dict() reports Celsius for display; the derating tables and every
    # strength_at() call work in KELVIN. Mixing them silently returns the
    # room-temperature strength, which for an aluminium crown at 232 C
    # overstates the allowable by about 50%.
    celsius = float(metrics.thermal.as_dict().get(
        SERVICE_TEMPERATURE[part], 20.0))
    kelvin = celsius + 273.15

    # Reuse the margin layer's own rule so the colour scale and the safety
    # factors cannot disagree about what this part is allowed to carry.
    # Grey iron has no yield point, so it falls back to ultimate and says so.
    from ..margins import ComponentContext

    allowable, basis = ComponentContext.strength(
        ComponentContext, material, kelvin)

    return {
        "material": material.name,
        "temperature_c": celsius,
        "allowable_pa": float(allowable),
        "basis": basis,
        "retained_fraction": float(material.strength_factor_at(kelvin)),
        "cold_pa": float(material.yield_strength
                         if material.yield_strength is not None
                         else material.ultimate_strength),
    }


def _centroids(mesh: MeshResult) -> np.ndarray:
    """Element centroids, (n_elements, 3), in metres."""
    return mesh.points.T[mesh.elements.T].mean(axis=1)


@dataclass
class CaseResult:
    """One solved component, next to what the fast layer said about it."""

    component: str
    solve: SolveResult
    force: float
    analytical_stress: float
    analytical_equation: str
    condition: str
    sample: np.ndarray | None = None
    sample_region: str = "the whole part"
    load_series: np.ndarray | None = None   # the driving load vs crank angle
    load_unit: str = "N"
    # The value of load_series at which the solve was done. Kept separate
    # from `force`, which is the headline number a person reads: for the
    # crown those differ by the bore area, and dividing by the wrong one
    # scales the whole cycle by a factor of 119.
    reference_load: float | None = None

    @property
    def load_reference(self) -> float:
        if self.reference_load is not None:
            return abs(self.reference_load)
        return abs(self.force)

    @property
    def sampled_stress(self) -> float:
        """The FEA stress to compare, over the region the formula describes.

        Comparing a whole-part peak against a closed form is how you get a
        meaningless number. The rod is the clearest case: its highest stress
        is in the small-end ring, while ``F / A_shank`` describes the SHANK,
        so the unsampled ratio came out at 5.6 and meant nothing. Each case
        says which elements its formula is about, and the comparison is taken
        there.
        """
        if self.sample is None:
            return self.solve.percentile_stress()
        values = self.solve.von_mises[self.sample]
        if values.size == 0:
            return float("nan")
        return float(np.percentile(values, 99.5))

    @property
    def ratio(self) -> float:
        """Sampled FEA stress over analytical nominal.

        NOT a stress concentration factor. See the module docstring.
        """
        if self.analytical_stress <= 0:
            return float("nan")
        return self.sampled_stress / self.analytical_stress

    def as_dict(self) -> dict:
        out = self.solve.as_dict()
        out.update({
            "component": self.component,
            "force_n": self.force,
            "analytical_stress_pa": self.analytical_stress,
            "analytical_equation": self.analytical_equation,
            "condition": self.condition,
            "load_series": (self.load_series.tolist()
                            if self.load_series is not None else None),
            "load_unit": self.load_unit,
            "reference_load": self.load_reference,
            "sampled_stress_pa": self.sampled_stress,
            "sample_region": self.sample_region,
            "sample_elements": (int(self.sample.sum())
                                if self.sample is not None
                                else int(self.solve.mesh.n_elements)),
            "fea_over_analytical": self.ratio,
        })
        return out


def pin_bending(state, tolerance: float = 0.5,
                mesh: MeshResult | None = None,
                target_elements: int = 25_000) -> CaseResult:
    """The gudgeon pin in three-point bending at its governing load.

    Geometry frame (from :mod:`psrt.geometry.build`): the pin is a tube along
    Y, centred on the origin. The load acts in -Z.

    Load: the small end bears on the middle of the pin over its own width. A
    uniform traction on that part of the outer surface has a resultant of
    pressure times PROJECTED area, so the pressure that delivers force F is
    F / (pin diameter x small-end width).

    Support: the bosses react on the underside, outboard of the gap between
    them. Two extra nodal restraints remove the remaining rigid-body motion
    without blocking anything the load path needs.
    """
    metrics = evaluate(state)
    sweep = metrics.sweep
    compression = sweep.peak_pin_compression
    tension = sweep.peak_pin_tension
    governing = (compression if abs(compression.value) >= abs(tension.value)
                 else tension)
    force = abs(governing.value)

    outer_r = state["pin.outer_diameter"] / 2.0
    half_contact = state["small_end.bushing_width"] / 2.0
    half_gap = state["piston.boss_inner_span"] / 2.0
    half_length = state["pin.length"] / 2.0
    material = materials_mod.get(state["materials.pin"])

    if mesh is None:
        mesh = tet_mesh(state, "pin", tolerance=tolerance,
                        target_elements=target_elements)

    surface = outer_r * 0.90        # "on the outer surface", with a tolerance

    def loaded(x):
        return (np.abs(x[1]) <= half_contact) & (x[2] > 0) & (
            np.hypot(x[0], x[2]) > surface)

    def supported(x):
        return ((np.abs(x[1]) >= half_gap) & (np.abs(x[1]) <= half_length)
                & (x[2] < 0) & (np.hypot(x[0], x[2]) > surface))

    def support_end(x):
        return supported(x) & (x[1] > 0)

    def support_corner(x):
        return supported(x) & (x[1] > 0) & (x[0] > 0)

    result = solve_linear_elastic(
        mesh, material,
        fixed={"z": supported, "y": support_end, "x": support_corner},
        # A Bearing rather than a hand-computed traction. Spreading a uniform
        # traction over a curved surface and sizing it by PROJECTED area
        # delivers pi/2 times the force intended -- the arc is longer than
        # its projection -- so this case was applying 77 kN for a 48.5 kN
        # load and reporting stresses about 59% too high. A Bearing is scaled
        # to its resultant after assembly, so the mesh and the curvature
        # cannot get it wrong.
        bearings=[Bearing(loaded, (0.0, 0.0, -1.0), force, "small end")],
        load_case=f"pin bending, {force / 1e3:.1f} kN")

    # What the fast layer says, on the same load: M / Z with the moment from a
    # simply supported beam carrying a central distributed load.
    span = state["pin.bending_span"]
    modulus = state["pin.section_modulus"]
    nominal = (force * (2.0 * span - 2.0 * half_contact) / 8.0) / modulus

    # Compare on the tension fibre over the loaded span. There is no clear
    # strip between the load and the supports to sample -- on a real engine
    # the small end very nearly fills the gap between the bosses -- so the
    # sample is taken on the far side of the section from the load, where the
    # bending stress is what M/Z describes and the contact patch is not.
    centre = _centroids(mesh)
    # 0.6 of the gap, chosen by convergence rather than by eye. The restraint
    # edge at |y| = half_gap is a singularity: sampling right up to it gives
    # ratios of 2.01, 2.39, 2.38 at 15k, 30k and 60k elements -- a number that
    # moves with mesh density is not a property of the part. Backing off to
    # 0.6 gives 1.36, 1.31, 1.33, which holds still.
    sample = ((np.abs(centre[:, 1]) < half_gap * 0.6)
              & (centre[:, 2] < -outer_r * 0.35))

    result.notes.append(
        "the bosses are modelled as reacting on the underside of the pin over "
        "their own width; a real boss loads its inner edge harder")
    return CaseResult(
        component="pin", solve=result, force=force,
        analytical_stress=nominal,
        analytical_equation="sigma = M / Z,  M = W (2S - c) / 8",
        condition=f"{governing.angle_deg:.0f} deg at "
                  f"{metrics.performance['speed_rpm']:.0f} rpm",
        sample=sample,
        sample_region="the tension fibre over the loaded span",
        load_series=np.asarray(sweep.f_pin, dtype=float), load_unit="N")


def crown_pressure(state, tolerance: float = 0.4,
                   mesh: MeshResult | None = None,
                   target_elements: int = 25_000) -> CaseResult:
    """The piston crown under peak cylinder pressure.

    Geometry frame (from :mod:`psrt.geometry.build`): the crown face is the
    flat top at z = 0 and the piston hangs below it to z = -total_height. The
    pin bore runs along Y at z = -compression_height.

    Load: peak cylinder pressure over the crown, applied along the inward
    normal so it stays a pressure rather than a fixed direction.

    Support: the pin bore. During firing the gas drives the piston toward the
    crank and the pin holds it back, so the load lands on the UPPER half of
    the bore. Restraining the lower half instead would put the crown in
    tension over a support that is not there.

    This is the case that ``piston.crown_support_radius_fraction`` exists to
    approximate, so the ratio it reports is the one that matters most.
    """
    metrics = evaluate(state)
    sweep = metrics.sweep
    peak = float(np.max(sweep.pressure))

    bore_r = state["engine.bore"] / 2.0
    comp_height = state["piston.compression_height"]
    pin_r = state["pin.outer_diameter"] / 2.0
    pin_half_length = state["pin.length"] / 2.0
    material = materials_mod.get(state["materials.piston"])

    if mesh is None:
        mesh = tet_mesh(state, "piston", tolerance=tolerance,
                        target_elements=target_elements)

    crown_z = float(mesh.points[2].max())

    def loaded(x):
        return x[2] > crown_z - 1e-4

    def bore_surface(x):
        radial = np.hypot(x[0], x[2] + comp_height)
        return ((radial < pin_r * 1.12) & (np.abs(x[1]) <= pin_half_length)
                & (x[2] > -comp_height))

    def bore_end(x):
        return bore_surface(x) & (x[1] > 0)

    def bore_corner(x):
        return bore_surface(x) & (x[1] > 0) & (x[0] > 0)

    result = solve_linear_elastic(
        mesh, material,
        fixed={"z": bore_surface, "y": bore_end, "x": bore_corner},
        traction=loaded, pressure=peak,
        load_case=f"crown, {peak / 1e6:.1f} MPa")

    # What the fast layer says: a clamped circular plate over an effective
    # support radius that is a calibrated fraction of the bore.
    support = bore_r * state["piston.crown_support_radius_fraction"]
    thickness = state["piston.crown_thickness"]
    thickness_for_sample = thickness * 1.05
    nominal = 3.0 * peak * support ** 2 / (4.0 * thickness ** 2)

    # Compare within the crown disc: the plate formula describes that and
    # nothing below it.
    centre = _centroids(mesh)
    sample = ((centre[:, 2] > crown_z - thickness_for_sample)
              & (np.hypot(centre[:, 0], centre[:, 1]) < support))

    result.notes.append(
        "mechanical pressure only: the thermal term in the analytical crown "
        "check is not in this solve, so the two are comparable on bending "
        "alone and not on the total")
    return CaseResult(
        component="piston", solve=result, force=peak * np.pi * bore_r ** 2,
        analytical_stress=nominal,
        analytical_equation="sigma = 3 q a^2 / (4 t^2),  a = f x bore/2",
        condition=f"peak pressure, {metrics.performance['speed_rpm']:.0f} rpm",
        sample=sample,
        sample_region="the crown disc inside the effective support radius",
        load_series=np.asarray(sweep.pressure, dtype=float),
        load_unit="Pa", reference_load=peak)


def rod_compression(state, tolerance: float = 0.4,
                    mesh: MeshResult | None = None,
                    target_elements: int = 25_000) -> CaseResult:
    """The connecting rod under its peak compressive load.

    Geometry frame: the small end is at z = 0, the big end at z = rod length,
    and both bores run along Y.

    Load: the pin bears on the wall of the small-end bore that faces the big
    end, and drives the rod at the crank. Applied as a bearing, so the force
    delivered is exactly the force asked for whatever the mesh does.

    Support: the near half of the big-end bore, where the crankpin reacts.
    """
    metrics = evaluate(state)
    sweep = metrics.sweep
    governing = sweep.peak_rod_compression
    force = abs(governing.value)

    length = state["engine.rod_length"]
    small_r = state["pin.outer_diameter"] / 2.0
    small_w = state["small_end.bushing_width"]
    big_r = state["rod.big_end_bore"] / 2.0
    big_w = state["rod.big_end_width"]
    material = materials_mod.get(state["materials.rod"])

    if mesh is None:
        mesh = tet_mesh(state, "rod", tolerance=tolerance,
                        target_elements=target_elements)


    def loaded(x):
        # The FAR wall of the small-end bore, toward the big end. Under
        # compression the pin drives the rod at the crank, so it bears on the
        # side of the bore that faces the big end -- not the near side. A
        # traction vector ignores surface normals and so applied happily to
        # the wrong wall; a bearing follows the normal and refused, which is
        # how this was found.
        #
        # Half the bushing width either side of the centreline, too: the
        # whole width either side doubled the contact band.
        radial = np.hypot(x[0], x[2])
        return ((radial < small_r * 1.12) & (np.abs(x[1]) <= small_w / 2.0)
                & (x[2] > 0))

    def big_end_bore(x):
        radial = np.hypot(x[0], x[2] - length)
        return ((radial < big_r * 1.10) & (np.abs(x[1]) <= big_w)
                & (x[2] < length))

    def big_end_side(x):
        return big_end_bore(x) & (x[1] > 0)

    def big_end_corner(x):
        return big_end_bore(x) & (x[1] > 0) & (x[0] > 0)

    result = solve_linear_elastic(
        mesh, material,
        fixed={"z": big_end_bore, "y": big_end_side, "x": big_end_corner},
        bearings=[Bearing(loaded, (0.0, 0.0, 1.0), force, "small end bore")],
        load_case=f"rod compression, {force / 1e3:.1f} kN")

    area = state["rod.shank_area"]
    nominal = force / area

    # Compare in the shank. The rod's highest stress is in the small-end
    # ring, which F / A_shank says nothing about -- sampling the whole part
    # gave a ratio of 5.6 and meant nothing.
    centre = _centroids(mesh)
    small_od = state["rod.small_end_outer_diameter"]
    big_od = state["rod.big_end_outer_diameter"]
    sample = ((centre[:, 2] > small_od * 0.75)
              & (centre[:, 2] < length - big_od * 0.75))

    result.notes.append(
        "a column check, not a buckling one: this is a linear solve, so it "
        "reports the stress at the load and says nothing about stability")
    return CaseResult(
        component="rod", solve=result, force=force,
        analytical_stress=nominal,
        analytical_equation="sigma = F / A_shank",
        condition=f"{governing.angle_deg:.0f} deg at "
                  f"{metrics.performance['speed_rpm']:.0f} rpm",
        sample=sample,
        sample_region="the shank, between the two end rings",
        load_series=np.asarray(sweep.f_rod, dtype=float), load_unit="N")


def sleeve_pressure(state, tolerance: float = 0.4,
                    mesh: MeshResult | None = None,
                    target_elements: int = 25_000) -> CaseResult:
    """The cylinder wall under peak pressure.

    The one component with an exact closed form on its real geometry, which
    is why it doubles as the validation case for the whole layer: Lame's
    thick-cylinder solution, and the solver lands within 1.5% of it.

    Geometry frame: an annulus about Z, running from z = -length to z = 0.
    """
    metrics = evaluate(state)
    peak = float(np.max(metrics.sweep.pressure))

    inner = state["engine.bore"] / 2.0
    outer = inner + state["sleeve.wall_thickness"]
    material = materials_mod.get(state["materials.sleeve"])

    if mesh is None:
        mesh = tet_mesh(state, "sleeve", tolerance=tolerance,
                        target_elements=target_elements)

    bottom = float(mesh.points[2].min())

    def loaded(x):
        return np.hypot(x[0], x[1]) < (inner + outer) / 2.0

    def base(x):
        return x[2] < bottom + 1e-5

    def base_line(x):
        return base(x) & (np.abs(x[1]) < (outer - inner))

    def base_point(x):
        return base_line(x) & (x[0] > 0)

    result = solve_linear_elastic(
        mesh, material,
        fixed={"z": base, "y": base_line, "x": base_point},
        traction=loaded, pressure=peak,
        load_case=f"bore pressure, {peak / 1e6:.1f} MPa")

    hoop = peak * (outer ** 2 + inner ** 2) / (outer ** 2 - inner ** 2)
    nominal = float(np.sqrt(hoop ** 2 + hoop * peak + peak ** 2))

    # Compare at the bore, at mid-height, away from the restrained base.
    centre = _centroids(mesh)
    top = float(mesh.points[2].max())
    middle = 0.5 * (bottom + top)
    # A third of the wall, not a fifteenth: at a coarse mesh a thinner band
    # than one element caught no centroids at all and the comparison came
    # back as nan.
    sample = ((np.abs(centre[:, 2] - middle) < 0.25 * (top - bottom))
              & (np.hypot(centre[:, 0], centre[:, 1])
                 < inner + 0.35 * (outer - inner)))

    result.notes.append(
        "gas pressure only: no thermal gradient, no ring side load, and the "
        "liner is unsupported rather than sitting in a block")
    return CaseResult(
        component="sleeve", solve=result, force=peak,
        analytical_stress=nominal,
        analytical_equation="Lame: sigma_theta = p (b^2 + a^2) / (b^2 - a^2)",
        condition=f"peak pressure, {metrics.performance['speed_rpm']:.0f} rpm",
        sample=sample,
        sample_region="the bore wall at mid-height",
        load_series=np.asarray(metrics.sweep.pressure, dtype=float),
        load_unit="Pa", reference_load=peak)


CASES = {"pin": pin_bending, "piston": crown_pressure,
         "rod": rod_compression, "sleeve": sleeve_pressure}
