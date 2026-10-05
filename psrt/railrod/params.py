"""Design-state parameters for the rail connecting rod.

Everything here lives in one section, ``railrod``, so a design file that has
never heard of the concept still opens (``schema.migrate`` fills the section
in from these defaults) and a conventional-rod design is untouched until
``railrod.enabled`` is switched on.

Dimensions are SI like the rest of the state. Angles are in DEGREES, because
that is how a notch or a receiver is dimensioned on a drawing and nobody reads
0.5236 rad as thirty degrees.

The defaults are sized around the LS3 example (53.34 mm big-end bore, 23.95 mm
pin, 154 mm centres). They are a starting point for the optimiser and for
you, not a validated design -- the concept document says in so many words that
every dimension is still open, and the tool treats them that way.
"""

from __future__ import annotations

from ..state import Mutability, Parameter

FREE = Mutability.FREE
BOUNDED = Mutability.BOUNDED


def railrod_section() -> dict:
    """The ``railrod`` parameter section."""
    return {
        "enabled": Parameter(
            False, FREE, "",
            description="Use the rail connecting rod (monolithic small end + "
                        "twin steel rails, stabilising sleeve, big-end "
                        "receiver, two swing clamps, one tangential bolt) in "
                        "place of the conventional I-beam rod"),

        # --- small end and rails ----------------------------------------
        "eye_outer_diameter": Parameter(
            0.0320, BOUNDED, "m", minimum=0.020,
            description="Outside diameter of the small-end eye. Its bore is "
                        "the gudgeon pin diameter and its width is "
                        "small_end.bushing_width"),
        "rail_outer_span": Parameter(
            0.0280, BOUNDED, "m", minimum=0.010,
            description="Distance between the OUTSIDE faces of the two rails. "
                        "This is the strict envelope the sleeve slides over, "
                        "so nothing below the eye may exceed it"),
        "rail_width": Parameter(
            0.0060, BOUNDED, "m", minimum=0.0025,
            why_min="a rail thinner than about 2.5 mm has nothing left under "
                    "its locking notch",
            description="Rail thickness in the plane of rotation (x). The "
                        "notch is cut into this dimension"),
        "rail_depth": Parameter(
            0.0160, BOUNDED, "m", minimum=0.006,
            description="Rail depth along the crank axis (y). Out-of-plane "
                        "buckling rests almost entirely on this"),
        "eye_transition_radius": Parameter(
            0.0040, BOUNDED, "m", minimum=0.0005,
            description="Blend radius where the rails run into the eye"),

        # --- locking notch ------------------------------------------------
        "notch_depth": Parameter(
            0.0012, BOUNDED, "m", minimum=0.0003,
            description="Depth of the locking notch at its deepest point, "
                        "measured in from the rail's outside face"),
        "notch_ramp_angle": Parameter(
            45.0, BOUNDED, "deg", minimum=8.0, maximum=75.0,
            description="Angle of the tapered ramp from the rail axis. "
                        "Steeper holds the rail with less lateral pry per "
                        "newton of tension but makes a shorter, more "
                        "abrupt recess"),
        "notch_top_radius": Parameter(
            0.0018, BOUNDED, "m", minimum=0.0003,
            description="Radius of the deepest/top transition of the notch. "
                        "It is also the socket the clamp tongue's nose sits "
                        "in -- the nose, NOT the pivot: the pivot is the "
                        "centre of the swing channel's arc and is on no part "
                        "at all. Bounded below by the notch depth: a "
                        "smaller radius cannot reach back to the rail face "
                        "without a step or an undercut, which the concept "
                        "rules out"),
        "notch_lower_radius": Parameter(
            0.0010, BOUNDED, "m", minimum=0.0002,
            description="Blend radius where the ramp returns to full rail "
                        "width"),
        "rail_land_length": Parameter(
            0.0030, BOUNDED, "m", minimum=0.0010,
            description="Length from the rail's flat bottom face up to where "
                        "the ramp would meet the full-width face. The "
                        "full-width land below the notch is this less the "
                        "lower blend"),

        # --- big-end receiver --------------------------------------------
        "receiver_arc_half_angle": Parameter(
            50.0, BOUNDED, "deg", minimum=25.0, maximum=80.0,
            description="Half the angle of crankpin bore the receiver wraps, "
                        "measured from the rod axis. The swing clamps wrap "
                        "the rest"),
        "receiver_floor_thickness": Parameter(
            0.0040, BOUNDED, "m", minimum=0.0015,
            description="Thinnest material between the crankpin bore and the "
                        "floor a rail seats on"),
        "rail_engagement": Parameter(
            0.0110, BOUNDED, "m", minimum=0.004,
            description="How deep each rail sits in the receiver, from its "
                        "flat foot up to the floor of the sleeve seat. The "
                        "locking notch lives inside this depth, so the swing "
                        "clamp reaches it through the receiver's own wall "
                        "rather than over the top"),
        "receiver_wall": Parameter(
            0.0040, BOUNDED, "m", minimum=0.0020,
            description="Material outboard of each rail slot at the top of "
                        "the receiver. This is the wall the curved swing "
                        "channel is cut through, so it has to be thicker "
                        "than the channel is deep"),
        "receiver_rim": Parameter(
            0.0020, BOUNDED, "m", minimum=0.0008,
            description="Material left above the channel mouth, between it "
                        "and the receiver's top face"),
        "slot_corner_radius": Parameter(
            0.0010, BOUNDED, "m", minimum=0.0002,
            description="Radius in the vertical corners of the rail slots "
                        "and the sleeve's seat. A sharp internal corner is "
                        "not a thing that can be machined -- the cutter has "
                        "a radius -- and modelling one puts a singularity "
                        "where the slot meets the swing channel, which is "
                        "exactly where this receiver is most loaded"),
        "slot_clearance": Parameter(
            1.0e-4, BOUNDED, "m", minimum=0.0,
            description="Clearance between a rail and the slot it drops "
                        "into in the receiver"),
        "seat_depth": Parameter(
            0.0030, BOUNDED, "m", minimum=0.0008,
            description="Depth of the inset seat machined into the "
                        "receiver's top face. The bottom of the stabilising "
                        "sleeve lands in it; the rails carry on down through "
                        "its floor into their slots"),
        "seat_clearance": Parameter(
            1.0e-4, BOUNDED, "m", minimum=0.0,
            description="Clearance between the sleeve and the walls of the "
                        "seat it sits in"),
        "receiver_tip_angle": Parameter(
            30.0, BOUNDED, "deg", minimum=10.0, maximum=80.0,
            description="Included angle at the receiver's lower corners, "
                        "between the bore and the flank. Small is a knife "
                        "edge; large pushes the clamp arm outward"),
        "edge_radius": Parameter(
            0.0006, BOUNDED, "m", minimum=0.0001,
            description="Radius every convex corner of the receiver and the "
                        "clamps is broken to"),

        # --- swing clamps -------------------------------------------------
        "clamp_thickness": Parameter(
            0.0065, BOUNDED, "m", minimum=0.0025,
            description="Radial thickness of each clamp around the crankpin"),
        "arm_thickness": Parameter(
            0.0050, BOUNDED, "m", minimum=0.0020,
            description="Thickness of the clamp arm where it runs up the "
                        "receiver flank to the rail"),
        "swing_radius": Parameter(
            0.0140, BOUNDED, "m", minimum=0.0040,
            description="Radius of the swing channel's centreline. This is "
                        "the dimension that DEFINES the swing: the pivot is "
                        "the centre of this arc, and it lies on no part -- "
                        "it floats above the rail, outside every solid. A "
                        "larger radius makes the clamp enter flatter and "
                        "drive more squarely into the notch; a smaller one "
                        "makes it drop in more steeply"),
        "swing_pivot_lean": Parameter(
            15.0, BOUNDED, "deg", minimum=-20.0, maximum=70.0,
            description="Where the pivot sits, as an angle from straight up, "
                        "leaning OUT away from the rod axis, seen from the "
                        "seated clamp tip. It has to lean out: lean it in "
                        "and the pivot lands inside the rail or the sleeve, "
                        "and the layout refuses it, because a pivot on a "
                        "part is a part acting as the hinge. Leaning it out "
                        "also tips the tip's travel, so the clamp swings "
                        "down onto the crankpin as it closes rather than "
                        "sliding straight sideways into the notch"),
        "swing_open_angle": Parameter(
            40.0, BOUNDED, "deg", minimum=5.0, maximum=75.0,
            description="How far back the clamp swings from seated. The "
                        "channel through the receiver is the clamp's tongue "
                        "swept through this angle, so it is what makes the "
                        "channel a channel rather than a slot"),
        "channel_thickness": Parameter(
            0.0050, BOUNDED, "m", minimum=0.0020,
            description="Radial thickness of the swing channel, and so of "
                        "the clamp tongue that runs in it. Its width along "
                        "the crank axis is not a free parameter: it is the "
                        "rail depth, because the tongue has to fill the "
                        "notch front to back"),
        "channel_clearance": Parameter(
            7.5e-5, BOUNDED, "m", minimum=0.0,
            description="Clearance between the clamp tongue and the walls "
                        "of the channel it swings in"),
        "tip_clearance": Parameter(
            5.0e-5, BOUNDED, "m", minimum=0.0,
            description="Gap between the knob/flank and the rail notch they "
                        "conform to. Geometric: the contact network treats "
                        "the seated joint as closed"),
        "seat_interference": Parameter(
            5.0e-5, BOUNDED, "m", minimum=-2e-4, maximum=1e-3,
            description="Designed overlap between the clamp's knob and flank "
                        "and the rail notch in the nominal closed position. "
                        "Positive means the tip has to be forced into the "
                        "notch, so the bolt preload drives the rail foot down "
                        "onto the receiver floor -- the only way the joint "
                        "carries preload at the notch at all. Zero is "
                        "line-to-line; negative is a running gap"),
        "guide_clearance": Parameter(
            0.0, BOUNDED, "m", minimum=0.0,
            description="Gap between the clamp and the receiver flank and "
                        "scoop in the closed position. The contact network "
                        "closes it only if the joint deflects that far"),
        "lug_gap": Parameter(
            1.0e-3, BOUNDED, "m", minimum=0.0,
            description="Gap between the two lug faces at assembly, before "
                        "preload. With a gap the bolt clamps the crankpin "
                        "rather than the lugs"),
        "lug_length": Parameter(
            0.0120, BOUNDED, "m", minimum=0.005,
            description="Length of each lug along the bolt axis"),
        "lug_wall": Parameter(
            0.0025, BOUNDED, "m", minimum=0.0010,
            description="Material below the bolt hole at the bottom of the "
                        "lug"),
        "shell_mass": Parameter(
            0.030, FREE, "kg", source="estimated",
            description="Big-end bearing shells, the one piece of rod "
                        "hardware not modelled as a solid. Lumped at the "
                        "crankpin centre"),
        "bigend_material": Parameter(
            "4340", FREE, "",
            description="Material of the receiver and both clamps"),

        # --- tangential fastener -----------------------------------------
        "bolt_diameter": Parameter(0.0080, BOUNDED, "m", minimum=0.004),
        "bolt_pitch": Parameter(0.00125, FREE, "m"),
        "bolt_offset": Parameter(
            0.0065, BOUNDED, "m", minimum=0.002,
            description="Distance from the crankpin bore surface to the bolt "
                        "axis, straight down"),
        "bolt_preload": Parameter(
            25_000.0, BOUNDED, "N", minimum=0.0,
            description="Assembly preload in the single tangential bolt"),
        "bolt_material": Parameter("300M", FREE, ""),
        "bolt_proof_strength": Parameter(1300.0e6, FREE, "Pa"),
        "bolt_finish": Parameter(
            "ground", FREE, "",
            description="Rolled threads are better than this credits"),

        # --- stabilising sleeve ------------------------------------------
        "sleeve_material": Parameter(
            "AlSi10Mg-T6", FREE, "",
            description="Parent alloy of BOTH the sleeve's solid skins and "
                        "its lattice core. AlSi10Mg-T6 is the standard "
                        "powder-bed aluminium and the sleeve is designed to "
                        "be printed. Other candidates in the database: "
                        "Scalmalloy-class alloys are not in it yet; "
                        "CFRP-quasi-iso, PEEK-CF30, AZ80-T5, 7075-T6 and "
                        "Ti-6Al-4V remain if you want to price the sleeve as "
                        "a solid part instead (set sleeve_core to solid)"),
        "sleeve_core": Parameter(
            "sheet-gyroid", FREE, "",
            description="What fills the web between the two rail skins: "
                        "'sheet-gyroid' for the printed lattice, 'solid' for "
                        "the original one-piece sleeve. Everything else "
                        "about the sleeve is the same either way"),
        "sleeve_skin": Parameter(
            0.0009, BOUNDED, "m", minimum=0.0005,
            why_min="a skin thinner than half a millimetre cannot be held "
                    "flat through machining, let alone bear on a rail",
            description="Thickness of the solid skin wrapped round each rail "
                        "channel, AFTER machining. These are the only "
                        "surfaces on the sleeve that touch anything, and "
                        "both channels are machined in one fixture so they "
                        "stay parallel. Thicker skins eat the lattice core "
                        "from both sides, and the core has to stay at least "
                        "four cells wide for its effective properties to "
                        "mean anything"),
        "sleeve_face_shell": Parameter(
            0.0008, BOUNDED, "m", minimum=0.0,
            description="Thickness of the thin solid shell on each of the "
                        "sleeve's two big faces. Ties the two rail skins "
                        "together and makes the part handleable; set it to "
                        "zero to leave the lattice open on both faces, which "
                        "is the easiest thing in the world to get powder out "
                        "of and the least stiff"),
        "machining_allowance": Parameter(
            0.0004, BOUNDED, "m", minimum=0.0,
            description="Stock left on the printed rail channels for the "
                        "single machining fixture. The skin is PRINTED at "
                        "sleeve_skin plus this and machined back"),

        # --- the lattice core ---------------------------------------------
        "lattice_cell": Parameter(
            0.0035, BOUNDED, "m", minimum=0.0015,
            description="Gyroid unit cell. It fights itself from both ends: "
                        "a smaller cell fits more cells across the core "
                        "(which homogenisation needs) but thins the sheet "
                        "below what the machine can print at a given "
                        "density, and a larger cell does the reverse"),
        "lattice_density": Parameter(
            0.36, BOUNDED, "-", minimum=0.05, maximum=0.75,
            description="Relative density of the lattice at the sleeve's "
                        "MID-LENGTH, where the shear it carries is least. "
                        "This is the number to sweep: the web's shear "
                        "stiffness is what turns two rails into one column, "
                        "so it lands straight on the buckling margins"),
        "lattice_end_density": Parameter(
            0.50, BOUNDED, "-", minimum=0.05, maximum=0.75,
            description="Relative density at each END of the sleeve. Denser "
                        "than mid-length on purpose: a shear web's demand is "
                        "V = dM/dz, which peaks at the ends and vanishes at "
                        "mid-length, the opposite of the bending moment"),
        "lattice_grading_exponent": Parameter(
            2.0, BOUNDED, "-", minimum=0.5, maximum=6.0,
            description="How the density runs from mid-length to the ends, "
                        "as |s|^exponent. 1 is linear; higher keeps the "
                        "light middle for longer and puts the transition "
                        "nearer the ends"),
        "lattice_min_wall": Parameter(
            0.00035, BOUNDED, "m", minimum=0.0001, source="estimated",
            description="Thinnest sheet the machine will print. The binding "
                        "station is wherever the graded density is LOWEST, "
                        "not the average. Replace with your own machine's "
                        "figure"),
        "lattice_ports": Parameter(
            True, FREE, "",
            description="Powder ports through the face shells, on the cell "
                        "pitch. With shells on both faces and a rail skin on "
                        "each side the core is closed on four sides and "
                        "drains only along its own length; the ports cut "
                        "that path by an order of magnitude. Irrelevant if "
                        "sleeve_face_shell is zero"),
        "lattice_port_diameter": Parameter(
            0.0025, BOUNDED, "m", minimum=0.001,
            description="Diameter of each powder port through the face "
                        "shells"),
        "lattice_modulus_coefficient": Parameter(
            0.80, BOUNDED, "-", minimum=0.1, maximum=2.0, source="estimated",
            description="C in E*/Es = C rho^n for the sheet gyroid. A "
                        "CALIBRATED GUESS from published sheet-TPMS fits, "
                        "not a measurement. Every lattice stiffness in this "
                        "tool rests on it, so it is the first thing to "
                        "replace with compression coupons off your own "
                        "machine in your own build orientation"),
        "lattice_modulus_exponent": Parameter(
            1.35, BOUNDED, "-", minimum=1.0, maximum=2.5, source="estimated",
            description="n in E*/Es = C rho^n. Near 1 is stretch-dominated, "
                        "near 2 is bending-dominated; sheet gyroids sit in "
                        "between and closer to stretch, which is why they "
                        "beat the network form. Also a calibrated guess"),
        "lattice_strength_coefficient": Parameter(
            0.65, BOUNDED, "-", minimum=0.1, maximum=2.0, source="estimated",
            description="C in sigma*/sigma_s = C rho^n for the lattice. "
                        "Calibrated guess. Printed lattice strength is hit "
                        "harder by surface roughness than stiffness is, "
                        "because a rough as-built sheet is all notch"),
        "lattice_strength_exponent": Parameter(
            1.45, BOUNDED, "-", minimum=1.0, maximum=2.5, source="estimated",
            description="n in sigma*/sigma_s = C rho^n. Calibrated guess"),
        "lattice_surface_factor": Parameter(
            0.5, BOUNDED, "-", minimum=0.1, maximum=1.0, source="estimated",
            description="What an as-built powder-bed surface does to the "
                        "lattice's ENDURANCE limit. The material database's "
                        "figures assume machined surfaces, which the skins "
                        "are and a third-of-a-millimetre sheet can never be. "
                        "Set it to 1.0 if your lattice strength "
                        "coefficients already came from fatigue tests on "
                        "as-built specimens, or this knocks the surface down "
                        "twice"),
        "sleeve_wall": Parameter(
            0.0020, BOUNDED, "m", minimum=0.0005,
            description="Sleeve wall outside the rails, in x and in y"),
        "sleeve_clearance": Parameter(
            5.0e-5, BOUNDED, "m", minimum=0.0,
            description="Clearance between each rail and its sleeve channel"),
        "sleeve_end_gap": Parameter(
            0.0010, BOUNDED, "m", minimum=0.0002,
            description="Gap between the sleeve and the eye at the top, and "
                        "between the sleeve and the clamp tips at the bottom"),
        "bracing_fraction": Parameter(
            0.02, BOUNDED, "-", minimum=0.0, maximum=0.2, source="estimated",
            description="Lateral force the sleeve must supply to keep the "
                        "rails straight, as a fraction of rod compression. "
                        "The classical bracing rule is 1-2%"),

        # --- the contact network -----------------------------------------
        "contact_depth": Parameter(
            0.0020, BOUNDED, "m", minimum=1e-4, source="estimated",
            description="Effective compliant depth behind each contact patch, "
                        "setting contact stiffness as E * area / depth. The "
                        "network treats the parts as rigid, so this is where "
                        "all of their elasticity is lumped. Load SHARING "
                        "between redundant contacts depends on it; statically "
                        "determinate paths do not"),
    }


def refresh_railrod_bounds(state) -> list[str]:
    """Bounds that depend on other parameters.

    The notch top radius has to reach back to the rail face from the deepest
    point without undercutting, which needs radius >= depth. Enforced as a
    bound so every write path refuses a violating value with the reason.
    """
    notes: list[str] = []
    if not state.has("railrod.notch_top_radius"):
        return notes
    radius = state.param("railrod.notch_top_radius")
    depth = state["railrod.notch_depth"]
    radius.minimum = depth
    radius.why_min = (
        f"the top radius must be at least the {depth * 1e3:.2f} mm notch "
        "depth, or the recess needs a step or an undercut to return to the "
        "rail face -- the concept requires a radiused transition, not a "
        "stepped pocket or a hook")
    if radius.value is not None and radius.value < depth:
        notes.append(
            f"notch top radius raised from {radius.value * 1e3:.2f} to "
            f"{depth * 1e3:.2f} mm to match the notch depth")
        radius.value = depth

    width = state.param("railrod.rail_width")
    span = state["railrod.rail_outer_span"]
    width.maximum = span / 2.0 - 0.0005
    width.why_max = "the two rails would meet on the rod axis"

    depth_p = state.param("railrod.notch_depth")
    depth_p.maximum = state["railrod.rail_width"] * 0.6
    depth_p.why_max = ("a notch deeper than 60% of the rail width leaves too "
                       "little net section to carry tension")

    if state.has("railrod.rail_engagement"):
        eng = state.param("railrod.rail_engagement")
        eng.minimum = max(0.004, state["railrod.rail_land_length"]
                          + state["railrod.notch_top_radius"] * 2.0
                          + 0.0015)
        eng.why_min = (
            "the notch has to finish below the floor of the sleeve seat, or "
            "the clamp is reaching over the top of the receiver instead of "
            "through its wall")
    return notes
