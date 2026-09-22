"""The default design state, and the derivations that hang off it.

Crank angle convention used everywhere in this tool:

    theta = 0        firing TDC (top dead centre, start of the power stroke)
    theta < 0        compression stroke
    theta > 0        expansion stroke
    theta = +/-180   BDC
    theta = 360      exhaust/intake TDC ("overlap TDC")

Angles are stored in radians, speeds in rad/s, everything in SI. The CLI
displays degrees and rpm.

Phase note: compression ratio is an *input* here and clearance volume is
derived from it. Once the geometry kernel lands in phase 3 this flips --
the crown dish volume becomes the input and compression ratio becomes
derived from real geometry.
"""

from __future__ import annotations

import math

from .schema_structural import link_pin_to_boss, structural_sections
from .state import DesignState, Mutability, Parameter, derived

FREE = Mutability.FREE
BOUNDED = Mutability.BOUNDED
LOCKED = Mutability.LOCKED
DERIVED = Mutability.DERIVED


# --- Derivations -----------------------------------------------------------

@derived("engine.crank_radius")
def _crank_radius(s: DesignState) -> float:
    return s["engine.stroke"] / 2.0


@derived("engine.bore_area")
def _bore_area(s: DesignState) -> float:
    return math.pi * s["engine.bore"] ** 2 / 4.0


@derived("engine.displacement_cyl")
def _disp_cyl(s: DesignState) -> float:
    return s["engine.bore_area"] * s["engine.stroke"]


@derived("engine.displacement_total")
def _disp_total(s: DesignState) -> float:
    return s["engine.displacement_cyl"] * s["engine.n_cylinders"]


@derived("engine.clearance_volume")
def _clearance_volume(s: DesignState) -> float:
    """V_c = V_d / (CR - 1), from the definition CR = (V_d + V_c) / V_c."""
    return s["engine.displacement_cyl"] / (s["engine.compression_ratio"] - 1.0)


@derived("engine.rod_ratio")
def _rod_ratio(s: DesignState) -> float:
    """Rod length over crank radius. Typically 3.0 to 4.0; higher means less
    side thrust and a gentler secondary inertia term."""
    return s["engine.rod_length"] / s["engine.crank_radius"]


@derived("engine.mean_piston_speed")
def _mean_piston_speed(s: DesignState) -> float:
    """2 * stroke * rev/s. The classic durability yardstick: production
    engines sit near 20 m/s at redline, race engines above 25 m/s."""
    rev_per_s = s["operating.speed"] / (2.0 * math.pi)
    return 2.0 * s["engine.stroke"] * rev_per_s


@derived("masses.piston_assembly")
def _piston_assembly(s: DesignState) -> float:
    return (s["masses.piston"] + s["masses.rings"]
            + s["masses.pin"] + s["masses.retainers"])


@derived("masses.rod_reciprocating")
def _rod_recip(s: DesignState) -> float:
    """Reciprocating share of the rod, split at its true centre of mass.

    Two-mass equivalent: the mass carried at the small end is
    m_rod * (L - x_cg) / L, with x_cg measured from the small-end centre.
    For a typical rod (cg about 0.72 L from the small end) this lands near
    0.28 m_rod, which is where the familiar "one third" rule of thumb comes
    from -- but the rule of thumb is not used here.
    """
    length = s["engine.rod_length"]
    x_cg = s["masses.rod_cg_from_small_end"]
    return s["masses.rod_total"] * (length - x_cg) / length


@derived("masses.rod_rotating")
def _rod_rotating(s: DesignState) -> float:
    length = s["engine.rod_length"]
    x_cg = s["masses.rod_cg_from_small_end"]
    return s["masses.rod_total"] * x_cg / length


@derived("masses.reciprocating")
def _reciprocating(s: DesignState) -> float:
    """Total reciprocating mass per cylinder.

    This is the number that couples the whole system together: it sets the
    inertia force, which sets rod tension at overlap TDC, which is what
    actually breaks connecting rods.
    """
    return s["masses.piston_assembly"] + s["masses.rod_reciprocating"]


@derived("operating.volumetric_efficiency")
def _ve(s: DesignState) -> float:
    """Volumetric efficiency at the current speed.

    A parabolic calibration curve, not derived physics. It exists so that the
    torque curve has a real shape rather than being flat, and so that engine
    speed actually changes the charge. Fit it to a dyno sheet, or bypass it
    entirely by importing a measured pressure trace.
    """
    n = s["operating.speed"]
    n_peak = s["operating.ve_peak_speed"]
    frac = (n - n_peak) / n_peak
    ve = s["operating.ve_peak"] * (1.0 - s["operating.ve_breadth"] * frac * frac)
    return max(ve, 0.15)


@derived("operating.trapped_mass")
def _trapped_mass(s: DesignState) -> float:
    """Charge mass sealed in the cylinder at IVC, in kg.

    m = VE * rho_manifold * V_displaced. Taking volumetric efficiency against
    swept volume is the usual convention.
    """
    from .units import R_AIR
    rho = s["operating.intake_pressure"] / (
        R_AIR * s["operating.intake_temperature"])
    return (s["operating.volumetric_efficiency"] * rho
            * s["engine.displacement_cyl"])


@derived("operating.trapped_pressure")
def _trapped_pressure(s: DesignState) -> float:
    """Cylinder pressure at IVC that follows from the trapped charge."""
    from . import kinematics as _kin
    from .units import R_AIR
    geom = _kin.CrankGeometry.from_state(s)
    v_ivc = float(_kin.cylinder_volume(geom, s["operating.ivc_angle"]))
    return (s["operating.trapped_mass"] * R_AIR
            * s["operating.intake_temperature"] / v_ivc)


# --- The default state -----------------------------------------------------

def default_state(name: str = "untitled") -> DesignState:
    """A complete, internally consistent design state with generic values.

    The numbers are a plausible mid-size naturally aspirated four-cylinder,
    present so that every code path has something to run on. Replace them
    with a real engine before believing any result.
    """
    engine = {
        "bore": Parameter(
            0.0860, BOUNDED, "m", minimum=0.0860, maximum=0.0895,
            why_min="subtractive manufacturing only: the bore cannot be made "
                    "smaller than the as-built bore without resleeving",
            why_max="placeholder envelope limit; set it properly from the "
                    "block model before trusting it",
            description="Cylinder bore diameter", source="estimated"),
        "stroke": Parameter(
            0.0860, FREE, "m",
            description="Crankshaft stroke", source="estimated"),
        "rod_length": Parameter(
            0.1430, FREE, "m",
            description="Connecting rod centre-to-centre length",
            source="estimated"),
        "n_cylinders": Parameter(
            4, FREE, "-", description="Number of cylinders"),
        "compression_ratio": Parameter(
            10.5, BOUNDED, "-", minimum=6.0, maximum=14.0,
            why_max="knock limit on pump fuel; raise only with the fuel and "
                    "boost level accounted for",
            description="Geometric compression ratio"),
        "redline": Parameter(
            7000 * 2 * math.pi / 60, FREE, "rad/s",
            description="Maximum engine speed"),
        # derived
        "crank_radius": Parameter(None, DERIVED, "m",
                                  description="Half the stroke"),
        "bore_area": Parameter(None, DERIVED, "m^2",
                               description="Piston crown area exposed to gas"),
        "displacement_cyl": Parameter(None, DERIVED, "m^3",
                                      description="Swept volume per cylinder"),
        "displacement_total": Parameter(None, DERIVED, "m^3",
                                        description="Total swept volume"),
        "clearance_volume": Parameter(None, DERIVED, "m^3",
                                      description="Volume above the piston at TDC"),
        "rod_ratio": Parameter(None, DERIVED, "-",
                               description="Rod length / crank radius"),
        "mean_piston_speed": Parameter(None, DERIVED, "m/s",
                                       description="Mean piston speed at the "
                                                   "current operating point"),
    }

    operating = {
        "speed": Parameter(
            5000 * 2 * math.pi / 60, FREE, "rad/s",
            description="Engine speed at the operating point"),
        "intake_pressure": Parameter(
            95_000.0, FREE, "Pa",
            description="Manifold pressure at IVC (in-cylinder)"),
        "intake_temperature": Parameter(
            330.0, FREE, "K",
            description="Charge temperature at IVC"),
        "crankcase_pressure": Parameter(
            101_325.0, FREE, "Pa",
            description="Pressure under the piston; opposes gas force"),
        "exhaust_pressure": Parameter(
            110_000.0, FREE, "Pa",
            description="Cylinder pressure during the exhaust stroke"),
        "afr": Parameter(
            14.7, FREE, "-",
            description="Air-fuel ratio by mass"),
        "fuel_lhv": Parameter(
            44.0e6, FREE, "J/kg",
            description="Lower heating value of the fuel (gasoline ~44 MJ/kg)"),
        "combustion_efficiency": Parameter(
            0.97, BOUNDED, "-", minimum=0.5, maximum=1.0,
            description="Fraction of fuel energy actually released"),
        "heat_loss_fraction": Parameter(
            0.14, BOUNDED, "-", minimum=0.0, maximum=0.5,
            description="Share of released heat lost to the walls during the "
                        "closed period. A lumped stand-in for a real heat "
                        "transfer model -- see the limits section",
            source="estimated"),
        "ve_peak": Parameter(
            0.92, BOUNDED, "-", minimum=0.20, maximum=1.40,
            description="Peak volumetric efficiency. Above 1.0 means the "
                        "intake is tuned or boosted",
            source="estimated"),
        "ve_peak_speed": Parameter(
            4500 * 2 * math.pi / 60, FREE, "rad/s",
            description="Engine speed at which volumetric efficiency peaks",
            source="estimated"),
        "ve_breadth": Parameter(
            0.45, BOUNDED, "-", minimum=0.0, maximum=3.0,
            description="How sharply VE falls away from its peak. Small means "
                        "a broad flat curve, large means a peaky one",
            source="estimated"),
        "ivc_angle": Parameter(
            math.radians(-140.0), FREE, "rad",
            description="Intake valve closing; start of the closed period"),
        "evo_angle": Parameter(
            math.radians(130.0), FREE, "rad",
            description="Exhaust valve opening; end of the closed period"),
        "soc_angle": Parameter(
            math.radians(-15.0), FREE, "rad",
            description="Start of combustion (start of heat release)"),
        "burn_duration": Parameter(
            math.radians(50.0), BOUNDED, "rad", minimum=math.radians(10.0),
            maximum=math.radians(120.0),
            description="Crank angle for 0 to ~100% mass fraction burned"),
        "wiebe_a": Parameter(
            6.908, FREE, "-",
            description="Wiebe efficiency parameter; 6.908 gives 99.9% burned "
                        "at the end of the stated duration"),
        "wiebe_m": Parameter(
            2.0, FREE, "-",
            description="Wiebe form factor; 2 is typical for SI engines"),
        "fmep_constant": Parameter(
            30_000.0, FREE, "Pa",
            description="Chen-Flynn constant term: accessory and valvetrain "
                        "drag that does not scale with load or speed",
            source="estimated"),
        "fmep_load_coeff": Parameter(
            0.0080, FREE, "-",
            description="Chen-Flynn load term, multiplying peak cylinder "
                        "pressure. Covers ring and bearing friction rising "
                        "with gas load",
            source="estimated"),
        "fmep_speed_coeff": Parameter(
            800.0, FREE, "Pa.s/m",
            description="Chen-Flynn linear speed term (hydrodynamic friction)",
            source="estimated"),
        "fmep_speed2_coeff": Parameter(
            90.0, FREE, "Pa.s^2/m^2",
            description="Chen-Flynn quadratic speed term (windage, pumping)",
            source="estimated"),
        # derived
        "volumetric_efficiency": Parameter(
            None, DERIVED, "-",
            description="VE at the current speed, from the calibration curve"),
        "trapped_mass": Parameter(
            None, DERIVED, "kg", description="Charge mass sealed at IVC"),
        "trapped_pressure": Parameter(
            None, DERIVED, "Pa",
            description="Cylinder pressure at IVC implied by the trapped mass"),
        "gamma": Parameter(
            1.32, BOUNDED, "-", minimum=1.20, maximum=1.40,
            description="Ratio of specific heats used over the closed period. "
                        "Below the cold-air 1.4 to account for hot burned gas",
            source="estimated"),
    }

    masses = {
        "piston": Parameter(
            0.426391, FREE, "kg", source="derived",
            description="Bare piston. Measured from the CadQuery solid, so "
                        "the default state is self-consistent with its own "
                        "geometry"),
        "rings": Parameter(
            0.030, FREE, "kg", description="Full ring pack"),
        "pin": Parameter(
            0.120410, FREE, "kg", source="derived",
            description="Gudgeon pin, measured from the solid"),
        "retainers": Parameter(
            0.004, FREE, "kg", description="Circlips or other pin retainers"),
        "rod_total": Parameter(
            0.693863, FREE, "kg", source="derived",
            description="Connecting rod complete with cap, bolts and bearings, "
                        "measured from the solid plus rod.hardware_mass"),
        "rod_cg_from_small_end": Parameter(
            0.100102, FREE, "m", source="derived",
            description="Rod centre of mass from the small-end centre, "
                        "measured from the solid rather than assumed. It "
                        "lands at 0.700 of rod length here, which is where "
                        "the usual rule of thumb puts it -- but only once the "
                        "shank taper and the bolt bosses are modelled. "
                        "Without them the solid says 0.62, and that error "
                        "would land straight in reciprocating mass"),
        # derived
        "piston_assembly": Parameter(None, DERIVED, "kg",
                                     description="Piston + rings + pin + retainers"),
        "rod_reciprocating": Parameter(None, DERIVED, "kg",
                                       description="Rod mass acting at the small end"),
        "rod_rotating": Parameter(None, DERIVED, "kg",
                                  description="Rod mass acting at the crankpin"),
        "reciprocating": Parameter(None, DERIVED, "kg",
                                   description="Total reciprocating mass per cylinder"),
    }

    materials = {
        "piston": Parameter("2618-T61", FREE, "",
                            description="Piston material key"),
        "pin": Parameter("4340", FREE, "", description="Gudgeon pin material"),
        "rod": Parameter("4340", FREE, "", description="Connecting rod material"),
        "sleeve": Parameter("grey-iron", FREE, "",
                            description="Cylinder liner material"),
    }

    block = {
        "as_built_bore": Parameter(
            0.0860, LOCKED, "m",
            why="a physical fact about the existing block, not a design choice",
            description="Bore diameter as the block was manufactured"),
        "bore_spacing": Parameter(
            0.0960, LOCKED, "m",
            why="fixed by the block casting",
            description="Cylinder centre-to-centre distance",
            source="estimated"),
        "min_wall_to_coolant": Parameter(
            0.0040, BOUNDED, "m", minimum=0.0025,
            why_min="below about 2.5 mm the wall distorts badly under head "
                    "bolt clamping and cooling becomes unreliable",
            description="Minimum acceptable bore wall thickness",
            source="estimated"),
        "siamesed": Parameter(
            False, FREE, "",
            description="True if adjacent bores share a wall with no coolant "
                        "passage between them"),
        "liner_type": Parameter(
            "cast-in-iron", FREE, "",
            description="cast-in-iron | coated-aluminium | wet-sleeve | dry-sleeve"),
        "bore_profile": Parameter(
            "circle", FREE, "",
            description="Bore cross-section: 'circle', 'polygon-N' or "
                        "'oval-R'. Only a circle can be sealed by piston "
                        "rings; the others exist so the comparison can be "
                        "computed rather than asserted"),
        "resleeving_allowed": Parameter(
            False, FREE, "",
            description="If true, the bore may exceed the subtractive envelope "
                        "because a thicker sleeve will be pressed in"),

        # --- the rest of the envelope -------------------------------------
        # Everything below feeds psrt.envelope, which turns them into an
        # overbore ceiling. Each one is optional: a limit whose inputs are
        # missing is reported as unknown rather than silently skipped, which
        # matters because an unknown limit could be the binding one.
        "deck_height": Parameter(
            None, LOCKED, "m", optional=True,
            why="fixed by the block casting",
            description="Crank centreline to deck face",
            source="published"),
        "liner_thickness": Parameter(
            None, FREE, "m", optional=True,
            description="Wall thickness of a cast-in or pressed-in liner. "
                        "Boring eats into this before it reaches the parent "
                        "block material",
            source="estimated"),
        "min_liner_wall": Parameter(
            0.0015, BOUNDED, "m", minimum=0.0008,
            why_min="below about 0.8 mm a cast-in iron liner will not stay "
                    "round under thermal load and can collapse into the "
                    "coolant jacket",
            description="Minimum liner wall thickness to leave after boring",
            source="estimated"),
        "min_siamesed_web": Parameter(
            0.0025, BOUNDED, "m", minimum=0.0015,
            why_min="a siamesed web carries head clamping load with no "
                    "coolant behind it; thinner than about 1.5 mm and it "
                    "cracks between the bores",
            description="Minimum web between siamesed bores",
            source="estimated"),
        "gallery_offset": Parameter(
            None, FREE, "m", optional=True,
            description="Bore centre to the near wall of the main oil "
                        "gallery or a drainback. Not published -- sonic "
                        "testing or a sectioned block",
            source="estimated"),
        "min_wall_to_gallery": Parameter(
            0.0030, BOUNDED, "m", minimum=0.0020,
            why_min="breaking into an oil gallery scraps the block",
            description="Minimum material between bore and oil gallery",
            source="estimated"),
        "head_bolt_offset": Parameter(
            None, FREE, "m", optional=True,
            description="Bore centre to the near edge of a head bolt boss",
            source="estimated"),
        "min_wall_to_head_bolt": Parameter(
            0.0030, BOUNDED, "m", minimum=0.0020,
            why_min="the boss carries clamping load straight into the deck; "
                    "too little material and the bore distorts under torque",
            description="Minimum material between bore and head bolt boss",
            source="estimated"),
        "catalogue_max_bore": Parameter(
            None, FREE, "m", optional=True,
            description="Largest bore an aftermarket piston is sold for. "
                        "This is the industry's own verdict on what the "
                        "block tolerates and is worth more than anything "
                        "this tool could infer from first principles",
            source="catalogue"),
        "min_sleeve_to_sleeve": Parameter(
            0.0005, BOUNDED, "m", minimum=0.0,
            why_min="interlocking sleeves may touch: at that point the "
                    "sleeves themselves are the structure between the bores "
                    "and the parent metal between them is gone",
            description="Minimum gap between adjacent sleeve outside "
                        "diameters when resleeved",
            source="estimated"),
        "min_wall_when_sleeved": Parameter(
            0.0015, BOUNDED, "m", minimum=0.0008,
            why_min="a pressed-in ductile iron sleeve carries the load the "
                    "parent wall used to, but the parent metal still has to "
                    "hold the sleeve",
            description="Minimum parent-block wall to retain around a sleeve. "
                        "Smaller than the unsleeved minimum, which is the "
                        "point of sleeving",
            source="estimated"),
        "replacement_sleeve_wall": Parameter(
            0.0030, BOUNDED, "m", minimum=0.0020,
            why_min="a pressed-in ductile iron sleeve thinner than about "
                    "2 mm distorts when it is pressed and will not hold "
                    "round",
            description="Wall thickness of the sleeve used when resleeving",
            source="estimated"),
    }

    constraints = {
        "mass_ceiling_reciprocating": Parameter(
            None, FREE, "kg", optional=True,
            description="Optional ceiling on reciprocating mass per cylinder"),
        "min_safety_factor": Parameter(
            1.5, FREE, "-",
            description="Minimum acceptable safety factor on any component"),
    }

    sections_map = {
        "engine": engine,
        "operating": operating,
        "masses": masses,
        "materials": materials,
        "block": block,
        "constraints": constraints,
    }
    sections_map.update(structural_sections())

    state = DesignState(
        sections_map,
        meta={
            "name": name,
            "notes": "",
            "provenance": "generic placeholder values, not a real engine",
        },
    )
    refresh_bounds(state)
    return state


def refresh_bounds(state: DesignState) -> list[str]:
    """Recompute cross-parameter bounds and return a note for each one set.

    The bore is the parameter this matters for. Its floor is the as-built
    bore, because machining is subtractive; its ceiling comes from
    :mod:`psrt.envelope`, which weighs every limit the block model can
    evaluate and picks the tightest.

    Setting the bound here rather than checking it at the point of use is
    deliberate. A bound on the parameter is enforced by every write path in
    the tool -- the CLI, the API, the optimiser and the assistant all go
    through ``DesignState.set`` -- so an overbore past the envelope is
    REFUSED with its reason rather than computed and reported afterwards.
    """
    notes: list[str] = []
    if not (state.has("block.as_built_bore") and state.has("engine.bore")):
        return notes

    from .envelope import envelope as compute_envelope

    bore = state.param("engine.bore")
    env = compute_envelope(state)

    if env.min_bore is None:
        bore.minimum = None
        bore.why_min = ("resleeving allowed: a pressed-in sleeve adds material "
                        "back, so the bore is no longer bounded below")
        notes.append("bore lower bound removed (resleeving allowed)")
    else:
        bore.minimum = env.min_bore
        bore.why_min = (
            "subtractive manufacturing only: material comes off the block, "
            f"never on, so the bore cannot go below the as-built "
            f"{env.min_bore * 1000:.2f} mm")
        notes.append(f"bore lower bound set to as-built "
                     f"{env.min_bore * 1000:.2f} mm")

    # The skirt's bearing length cannot exceed the skirt the piston has.
    # These two came from different models -- the structural layer reads
    # skirt_length for bearing area, the geometry builder derives the panel
    # from total height less the ring belt -- and nothing tied them together,
    # so they had drifted 11% apart on the LS3 and 20% on the default. They
    # are still separate quantities: a real skirt is barrelled and relieved,
    # so the CONTACT length is legitimately shorter than the panel. What is
    # not legitimate is contact longer than the panel.
    if all(state.has(p) for p in ("piston.skirt_length", "piston.total_height",
                                  "piston.ring_belt_height")):
        panel = state["piston.total_height"] - state["piston.ring_belt_height"]
        if panel and panel > 0:
            skirt = state.param("piston.skirt_length")
            skirt.maximum = panel
            skirt.why_max = (
                f"the skirt panel is only {panel * 1000:.2f} mm long "
                f"({state['piston.total_height'] * 1000:.1f} mm total height "
                f"less the {state['piston.ring_belt_height'] * 1000:.1f} mm "
                "ring belt); the bearing length cannot exceed it")
            if skirt.value is not None and skirt.value > panel:
                notes.append(
                    f"skirt length clamped from {skirt.value * 1000:.2f} to "
                    f"{panel * 1000:.2f} mm: it was longer than the skirt "
                    "the geometry actually has")
                skirt.value = panel

    binding = env.binding
    if binding is None:
        # Nothing could be evaluated. Leaving the bore unbounded above would
        # let the tool cheerfully machine through a coolant jacket, so the
        # bore is pinned where it is and the reason says what to supply.
        bore.maximum = state["engine.bore"]
        bore.why_max = (
            "no limit in the block model could be evaluated, so the bore is "
            "pinned at its current size. Supply the bore spacing, the liner "
            "thickness or a catalogue bore to open it up.")
        notes.append("bore upper bound pinned: the envelope has no usable "
                     "inputs")
        return notes

    bore.maximum = binding.max_bore
    qualifier = "" if binding.tier <= 2 else " [ASSUMED input -- sonic-test]"
    bore.why_max = f"{binding.name}: {binding.reason}{qualifier}"
    notes.append(
        f"bore upper bound set to {binding.max_bore * 1000:.2f} mm by "
        f"{binding.name} (tier {binding.tier})")

    if env.unknown_limits:
        notes.append(
            "not every limit could be evaluated ("
            + ", ".join(limit.name for limit in env.unknown_limits)
            + "); the ceiling may be optimistic")

    return notes


def migrate(state: DesignState) -> list[str]:
    """Add any parameters the current schema defines but this state lacks.

    The design state grows with every phase: phase 2 brings structural
    parameters, phase 3 replaces specified masses with geometry. A file saved
    today must still open tomorrow, so loading fills in what is missing from
    the defaults and leaves every existing value alone. Returns one note per
    parameter added.
    """
    template = default_state()
    added: list[str] = []
    for section, params in template.sections.items():
        target = state.sections.setdefault(section, {})
        for name, param in params.items():
            if name not in target:
                import copy as _copy
                target[name] = _copy.deepcopy(param)
                added.append(f"{section}.{name}")
    return added


def load_state(path: str) -> tuple[DesignState, list[str]]:
    """Load a design state from disk, migrate it, and refresh its bounds.

    This is the front door for every command. Returns the state and any notes
    worth showing the user.
    """
    state = DesignState.load(path)
    notes: list[str] = []
    added = migrate(state)
    if added:
        notes.append(
            f"added {len(added)} parameter(s) introduced since this file was "
            f"saved: {', '.join(added[:8])}"
            + (" ..." if len(added) > 8 else ""))
    notes.extend(refresh_bounds(state))
    notes.extend(link_pin_to_boss(state))
    return state, notes
