"""Phase 2 additions to the design state: component geometry and allowables.

These are the parameters the structural models in :mod:`psrt.components` read.
They are specified directly here because there is no geometry kernel yet --
phase 3 replaces most of them with values measured off a real CadQuery solid,
at which point several become derived rather than free.

Where an allowable stress or pressure appears as a parameter rather than a
material property, it is because the limit is an operating-practice number
(scuffing, groove pounding, bearing pressure) rather than something you can
look up in a materials handbook. Those carry ``source="estimated"`` and the
value is typical industry practice, not a guarantee.
"""

from __future__ import annotations

import math

from . import sections
from .state import DesignState, Mutability, Parameter, derived

FREE = Mutability.FREE
BOUNDED = Mutability.BOUNDED
LOCKED = Mutability.LOCKED
DERIVED = Mutability.DERIVED


# --- Derivations -----------------------------------------------------------

@derived("sleeve.wall_thickness")
def _sleeve_wall(s: DesignState) -> float:
    """Bore wall thickness.

    Uses the measured value when one has been entered. Otherwise it falls back
    to half the material between adjacent bores, which is a *bore-to-bore*
    figure: the wall on the coolant side is usually different and can be
    thinner. Sonic-test the real block and set ``wall_thickness_measured``
    before any sleeve margin here is worth acting on.
    """
    measured = s.get("sleeve.wall_thickness_measured", None)
    if measured:
        return measured
    return max((s["block.bore_spacing"] - s["engine.bore"]) / 2.0, 1e-4)


@derived("sleeve.outer_diameter")
def _sleeve_od(s: DesignState) -> float:
    return s["engine.bore"] + 2.0 * s["sleeve.wall_thickness"]


@derived("piston.ring_belt_height")
def _ring_belt(s: DesignState) -> float:
    """Crown face down to the bottom of the oil ring groove.

    Three grooves: top compression, second compression, oil control, with a
    land between each. Everything below this is skirt.
    """
    return (s["piston.top_land_height"]
            + 2.0 * s["piston.ring_axial_height"]
            + 2.0 * s["piston.second_land_height"]
            + s["piston.oil_ring_height"])


@derived("piston.ring_groove_root_diameter")
def _groove_root(s: DesignState) -> float:
    return s["engine.bore"] - 2.0 * s["piston.ring_groove_depth"]


@derived("piston.skirt_bearing_area")
def _skirt_area(s: DesignState) -> float:
    """Projected bearing area of one skirt face.

    The skirt does not contact over its whole circumference; it bears over an
    arc on the thrust side. ``skirt_bearing_arc`` is that arc expressed as a
    fraction of the bore diameter.
    """
    return s["piston.skirt_length"] * s["engine.bore"] * s["piston.skirt_bearing_arc"]


@derived("pin.area")
def _pin_area(s: DesignState) -> float:
    return sections.tube(s["pin.outer_diameter"], s["pin.inner_diameter"]).area


@derived("pin.second_moment")
def _pin_i(s: DesignState) -> float:
    return sections.tube(s["pin.outer_diameter"], s["pin.inner_diameter"]).i_xx


@derived("pin.section_modulus")
def _pin_z(s: DesignState) -> float:
    return sections.tube(s["pin.outer_diameter"], s["pin.inner_diameter"]).z_xx


@derived("pin.bending_span")
def _pin_span(s: DesignState) -> float:
    """Centre-to-centre distance between the two boss bearing areas.

    The reaction sits somewhere between the inner edge of each boss and its
    centroid; ``pin.support_span_factor`` says where. The moment, and so the
    bending stress, is directly proportional to this span.
    """
    boss_width = (s["pin.length"] - s["piston.boss_inner_span"]) / 2.0
    return (s["piston.boss_inner_span"]
            + s["pin.support_span_factor"] * boss_width)


@derived("rod.shank_area")
def _rod_area(s: DesignState) -> float:
    return _rod_section(s).area


@derived("rod.shank_i_in_plane")
def _rod_ixx(s: DesignState) -> float:
    return _rod_section(s).i_xx


@derived("rod.shank_i_out_of_plane")
def _rod_iyy(s: DesignState) -> float:
    return _rod_section(s).i_yy


@derived("rod.slenderness_in_plane")
def _rod_lambda_x(s: DesignState) -> float:
    sec = _rod_section(s)
    return s["rod.k_in_plane"] * s["engine.rod_length"] / sec.r_xx


@derived("rod.slenderness_out_of_plane")
def _rod_lambda_y(s: DesignState) -> float:
    sec = _rod_section(s)
    return s["rod.k_out_of_plane"] * s["engine.rod_length"] / sec.r_yy


@derived("bolts.stress_area")
def _bolt_area(s: DesignState) -> float:
    return sections.bolt_stress_area(s["bolts.thread_diameter"],
                                     s["bolts.thread_pitch"])


def _rod_section(s: DesignState) -> sections.SectionProperties:
    return sections.i_beam(s["rod.shank_height"], s["rod.shank_width"],
                           s["rod.shank_web"], s["rod.shank_flange"])


# --- Sections --------------------------------------------------------------

def structural_sections() -> dict:
    """The phase 2 parameter set, keyed by section name."""

    sleeve = {
        "wall_thickness_measured": Parameter(
            None, FREE, "m", optional=True, source="measured",
            description="Measured bore wall thickness at its thinnest point. "
                        "Set this from a sonic test; until then the tool "
                        "falls back to a bore-spacing estimate"),
        "wall_thickness": Parameter(
            None, DERIVED, "m",
            description="Bore wall thickness used for the hoop stress model"),
        "outer_diameter": Parameter(
            None, DERIVED, "m", description="Outside diameter of the liner"),
    }

    piston = {
        "crown_thickness": Parameter(
            0.0070, BOUNDED, "m", minimum=0.0025,
            why_min="below about 2.5 mm a crown cannot carry peak firing "
                    "pressure or conduct its own heat away",
            description="Crown thickness at its thinnest section"),
        "crown_support_radius_fraction": Parameter(
            0.30, BOUNDED, "-", minimum=0.20, maximum=1.00, source="estimated",
            description="Largest unsupported crown radius, as a fraction of "
                        "the bore radius. A real crown is braced underneath "
                        "by the pin bosses and around its rim by the ring "
                        "belt, so it spans far less than the full bore. THIS "
                        "IS THE BIGGEST APPROXIMATION IN THE CROWN MODEL: the "
                        "stress scales with its square, and the default was "
                        "CALIBRATED so that a production piston sits just "
                        "above its fatigue limit at redline, not derived from "
                        "anything. Only FEA on real geometry settles it"),
        "thermal_constraint_factor": Parameter(
            0.60, BOUNDED, "-", minimum=0.0, maximum=1.0, source="estimated",
            description="Fraction of free thermal expansion the crown is "
                        "actually restrained from making. Drives the "
                        "thermal-fatigue estimate almost single-handedly"),
        "skirt_length": Parameter(
            0.0280, BOUNDED, "m",
            description="Axial length of the skirt that BEARS on the bore. "
                        "Not the same as the skirt panel the geometry draws: "
                        "a real skirt is barrelled and relieved, so the "
                        "contact length is shorter. It cannot be longer, "
                        "though, and refresh_bounds caps it at the panel "
                        "length -- these two came from different models and "
                        "had drifted 11% apart before that cap existed",
            source="estimated"),
        "skirt_bearing_arc": Parameter(
            1.00, BOUNDED, "-", minimum=0.10, maximum=1.00, source="estimated",
            description="Fraction of the full projected area that bears. The "
                        "standard convention for skirt specific pressure uses "
                        "the whole projected area, skirt length x bore, the "
                        "same way journal bearing pressure is defined -- so "
                        "1.0 matches published allowables. Reduce it only if "
                        "you also reduce the allowable to match"),
        "top_land_height": Parameter(
            0.0060, BOUNDED, "m", minimum=0.0025,
            why_min="a shorter top land cracks under peak pressure and runs "
                    "too hot",
            description="Axial height from the crown face to the top ring groove"),
        "ring_groove_depth": Parameter(
            0.0040, FREE, "m", description="Radial depth of the top ring groove"),
        "ring_axial_height": Parameter(
            0.0012, FREE, "m", description="Axial height of the top ring"),
        "boss_inner_span": Parameter(
            0.0240, FREE, "m",
            description="Clear gap between the two pin bosses; the rod small "
                        "end runs in it"),
        "allowable_skirt_pressure": Parameter(
            2.5e6, FREE, "Pa", source="estimated",
            description="Specific pressure at which the skirt starts to scuff, "
                        "on the full-projected-area convention. Published "
                        "practice spans roughly 1.5 to 3 MPa depending on "
                        "skirt coating, oil and bore finish: a band, not a "
                        "sharp limit"),
        "allowable_groove_pressure": Parameter(
            35.0e6, FREE, "Pa", source="estimated",
            description="Groove flank pressure an aluminium piston tolerates "
                        "before the groove starts to pound out"),
        "compression_height": Parameter(
            0.0300, FREE, "m",
            description="Crown face to pin centre. With rod length and crank "
                        "radius this fixes where the crown sits at TDC"),
        "total_height": Parameter(
            0.0520, FREE, "m",
            description="Crown face to the bottom of the skirt"),
        "wall_thickness": Parameter(
            0.0030, BOUNDED, "m", minimum=0.0012,
            why_min="below about 1.2 mm the ring belt wall cannot be cast or "
                    "machined reliably",
            description="Radial wall from the ring groove root inward to the "
                        "under-crown cavity"),
        "second_land_height": Parameter(
            0.0035, FREE, "m",
            description="Height of the lands between the ring grooves"),
        "oil_ring_height": Parameter(
            0.0030, FREE, "m", description="Axial height of the oil ring groove"),
        "oil_groove_depth": Parameter(
            0.0040, FREE, "m", description="Radial depth of the oil ring groove"),
        "crown_dish_volume": Parameter(
            0.0, FREE, "m^3",
            description="Volume cut out of the crown face. Positive dishes the "
                        "crown and lowers compression ratio; zero is flat"),
        "boss_width_factor": Parameter(
            2.40, BOUNDED, "-", minimum=1.30, maximum=6.00,
            description="Width of the pin boss pad across the thrust axis, as "
                        "a multiple of pin diameter. Wider spreads the boss "
                        "bore pressure and costs reciprocating mass"),
        "skirt_width_fraction": Parameter(
            0.78, BOUNDED, "-", minimum=0.30, maximum=1.00,
            description="Width of the skirt across the thrust faces, as a "
                        "fraction of the bore. Material outside it is relieved "
                        "at the pin-axis ends. Distinct from "
                        "skirt_bearing_arc, which is a pressure convention"),
        "cold_clearance": Parameter(
            6.0e-5, FREE, "m",
            description="Diametral piston-to-bore clearance when cold"),
        "ring_belt_height": Parameter(
            None, DERIVED, "m",
            description="Crown face to the bottom of the oil ring groove"),
        "ring_groove_root_diameter": Parameter(
            None, DERIVED, "m", description="Diameter at the bottom of the groove"),
        "skirt_bearing_area": Parameter(
            None, DERIVED, "m^2", description="Projected area of one skirt face"),
    }

    pin = {
        "outer_diameter": Parameter(
            0.0220, FREE, "m", description="Gudgeon pin outside diameter"),
        "inner_diameter": Parameter(
            0.0130, BOUNDED, "m", minimum=0.0,
            description="Pin bore. Bigger saves reciprocating mass and costs "
                        "bending stiffness -- the classic piston trade"),
        "length": Parameter(
            0.0620, FREE, "m", description="Overall pin length"),
        "support_span_factor": Parameter(
            0.50, BOUNDED, "-", minimum=0.0, maximum=1.0, source="estimated",
            description="Where the boss reaction acts, between the boss inner "
                        "edge (0.0) and the boss centroid (1.0). Real boss "
                        "pressure peaks near the inner edge but is not "
                        "concentrated there, so the truth is in between. Like "
                        "the crown support radius, this was calibrated so that "
                        "a production pin passes, and only FEA settles it"),
        "max_ovality": Parameter(
            7.0e-5, FREE, "m", source="estimated",
            description="Allowable increase in horizontal diameter under load. "
                        "Beyond roughly 0.07 mm the boss bore fatigues"),
        "area": Parameter(None, DERIVED, "m^2"),
        "second_moment": Parameter(None, DERIVED, "m^4"),
        "section_modulus": Parameter(None, DERIVED, "m^3"),
        "bending_span": Parameter(
            None, DERIVED, "m",
            description="Effective support span between the pin bosses"),
    }

    small_end = {
        "bushing_width": Parameter(
            0.0220, FREE, "m", description="Axial width of the small-end bore"),
        "material": Parameter(
            "bronze-c93200", FREE, "",
            description="Bushing material, or the rod material for a "
                        "bushingless small end"),
        "allowable_pressure": Parameter(
            80.0e6, FREE, "Pa", source="estimated",
            description="Projected specific pressure the bushing tolerates in "
                        "compression"),
    }

    rod = {
        "shank_height": Parameter(
            0.0300, FREE, "m",
            description="Section height in the plane of rotation"),
        "shank_width": Parameter(
            0.0180, FREE, "m",
            description="Flange width, measured along the crank axis"),
        "shank_web": Parameter(
            0.0048, BOUNDED, "m", minimum=0.0020,
            why_min="a thinner web cannot be forged or machined reliably",
            description="Web thickness of the I or H section"),
        "shank_flange": Parameter(
            0.0050, BOUNDED, "m", minimum=0.0020,
            description="Flange thickness"),
        "shank_taper": Parameter(
            1.25, BOUNDED, "-", minimum=1.00, maximum=2.00,
            description="How much larger the shank section is at the big end "
                        "than at the small end. The shank_* dimensions "
                        "describe the MINIMUM section, which is what the "
                        "tension and buckling checks use"),
        "k_in_plane": Parameter(
            1.00, FREE, "-", source="estimated",
            description="Effective-length factor for buckling in the plane of "
                        "rotation. 1.0 = pinned at both ends, which is what "
                        "the pin and crankpin actually provide"),
        "k_out_of_plane": Parameter(
            0.60, FREE, "-", source="estimated",
            description="Effective-length factor out of plane, where the pin "
                        "bosses and big end restrain rotation"),
        "small_end_outer_diameter": Parameter(
            0.0300, FREE, "m", description="Outside diameter of the small end"),
        "big_end_outer_diameter": Parameter(
            0.0680, FREE, "m", description="Outside diameter of the big end"),
        "big_end_bore": Parameter(
            0.0550, FREE, "m", description="Big-end bore diameter"),
        "big_end_width": Parameter(
            0.0220, FREE, "m", description="Big-end bearing width"),
        "bearing_clearance": Parameter(
            4.0e-5, FREE, "m", source="estimated",
            description="Radial running clearance of the big-end bearing"),
        "allowable_bearing_pressure": Parameter(
            60.0e6, FREE, "Pa", source="estimated",
            description="Projected specific pressure for a trimetal shell"),
        "oil_viscosity": Parameter(
            0.0080, FREE, "Pa.s", source="estimated",
            description="Dynamic viscosity at bearing temperature"),
        "surface_finish": Parameter(
            "machined", FREE, "",
            description="polished | ground | machined | hot-rolled | as-forged"),
        "hardware_mass": Parameter(
            0.060, FREE, "kg", source="estimated",
            description="Rod bolts, bearing shells and cap features the solid "
                        "model does not carry. Treated as lumped at the big "
                        "end centre, which is where they sit"),
        "shank_area": Parameter(None, DERIVED, "m^2"),
        "shank_i_in_plane": Parameter(None, DERIVED, "m^4"),
        "shank_i_out_of_plane": Parameter(None, DERIVED, "m^4"),
        "slenderness_in_plane": Parameter(None, DERIVED, "-"),
        "slenderness_out_of_plane": Parameter(None, DERIVED, "-"),
    }

    bolts = {
        "count": Parameter(2, FREE, "-", description="Rod bolts per rod"),
        "thread_diameter": Parameter(0.0100, FREE, "m"),
        "thread_pitch": Parameter(0.00150, FREE, "m"),
        "preload": Parameter(
            45_000.0, FREE, "N",
            description="Assembly preload per bolt"),
        "joint_stiffness_factor": Parameter(
            0.25, BOUNDED, "-", minimum=0.05, maximum=0.8, source="estimated",
            description="Share of the external load the bolt takes. 0.2 to 0.3 "
                        "for a stiff rod joint"),
        "proof_strength": Parameter(
            1300.0e6, FREE, "Pa",
            description="Bolt proof strength"),
        "material": Parameter("300M", FREE, ""),
        "surface_finish": Parameter(
            "ground", FREE, "",
            description="Rolled threads behave better than this suggests; the "
                        "benefit is not credited"),
        "cap_mass": Parameter(
            0.180, FREE, "kg", source="estimated",
            description="Rod cap plus its bearing shell. Its centrifugal force "
                        "adds to the bolt load at TDC"),
        "stress_area": Parameter(None, DERIVED, "m^2"),
    }

    thermal = {
        "coolant_temperature": Parameter(
            368.15, FREE, "K", description="Coolant temperature (95 C)"),
        "oil_temperature": Parameter(
            383.15, FREE, "K", description="Oil temperature (110 C)"),
        "crown_heat_share": Parameter(
            0.30, BOUNDED, "-", minimum=0.05, maximum=0.6, source="estimated",
            description="Share of wall heat rejection that goes into the "
                        "piston rather than the head or the liner"),
        "conductance_rings": Parameter(
            9.0, FREE, "W/K", source="estimated",
            description="Piston to liner through the ring pack; the dominant "
                        "path out of a piston"),
        "conductance_skirt": Parameter(
            3.0, FREE, "W/K", source="estimated",
            description="Piston to liner through the skirt"),
        "conductance_oil": Parameter(
            4.0, FREE, "W/K", source="estimated",
            description="Underside to oil. Raise this for a cooling jet or a "
                        "gallery-cooled piston"),
    }

    fatigue = {
        "reliability": Parameter(
            0.99, FREE, "-",
            description="Reliability the endurance limit is corrected for"),
        "design_life_cycles": Parameter(
            1.0e8, FREE, "-",
            description="Target life. 10^8 cycles is about 480 hours at "
                        "3500 rpm for a part loaded once per revolution"),
        "kf_crown": Parameter(
            1.8, FREE, "-", source="estimated",
            description="Fatigue notch factor at the crown-to-wall junction"),
        "kf_land": Parameter(
            2.2, FREE, "-", source="estimated",
            description="Notch factor at the ring groove root, a sharp corner"),
        "kf_pin": Parameter(
            1.3, FREE, "-", source="estimated",
            description="Notch factor for the pin; a plain tube, so mild"),
        "kf_rod": Parameter(
            1.6, FREE, "-", source="estimated",
            description="Notch factor at the shank-to-end transitions"),
        "kf_bolt": Parameter(
            3.0, FREE, "-", source="estimated",
            description="Notch factor at the thread root. Already implicit in "
                        "the bolt fatigue method, so it is reported rather "
                        "than applied twice"),
    }

    return {
        "sleeve": sleeve, "piston": piston, "pin": pin,
        "small_end": small_end, "rod": rod, "bolts": bolts,
        "thermal": thermal, "fatigue": fatigue,
    }


def link_pin_to_boss(state: DesignState) -> list[str]:
    """Keep the pin bore in the piston equal to the pin's outside diameter.

    They are the same hole. Letting them drift apart silently is exactly the
    kind of inconsistency this tool exists to prevent, so the piston's bore is
    not a separate parameter at all -- this just checks the pin fits between
    its own bosses.
    """
    notes: list[str] = []
    span = state["piston.boss_inner_span"]
    length = state["pin.length"]
    if length <= span:
        notes.append(
            f"pin length {length * 1e3:.1f} mm does not span the {span * 1e3:.1f} mm "
            "gap between the bosses; it has nothing to bear on")
    if state["small_end.bushing_width"] > span:
        notes.append(
            f"small-end width {state['small_end.bushing_width'] * 1e3:.1f} mm "
            f"exceeds the {span * 1e3:.1f} mm gap between the bosses")
    return notes
