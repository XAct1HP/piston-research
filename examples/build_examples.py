"""Generate the bundled example design states.

Run:  python examples/build_examples.py

Each example records, per parameter, whether its value is published, measured
or estimated. That matters: a result is only as trustworthy as its weakest
input, and estimated piston masses are the weakest inputs here.
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from psrt.schema import default_state, refresh_bounds
from psrt.units import rpm_to_rad_s

HERE = os.path.dirname(os.path.abspath(__file__))


def _set(state, path, value, source=None, why=None):
    p = state.param(path)
    was_locked = p.mutability.value == "locked"
    if was_locked:
        state.unlock(path)
    state.set(path, value, actor="preset", rationale="example engine data")
    if was_locked:
        state.lock(path, why or p.why)
    if source:
        p.source = source


def build_ls3():
    """GM LS3, 6.2 L pushrod V8 (2008-2017 Corvette / Camaro SS).

    Chosen as the validation engine because its architecture is unusually well
    documented: 4.065 in bore, 3.622 in stroke and the 4.400 in bore spacing
    shared across the whole LS family.
    """
    s = default_state("GM LS3 6.2L V8")

    _set(s, "block.as_built_bore", 0.103251, "published",
         "the block exists; the bore cannot be made smaller without resleeving")
    _set(s, "block.bore_spacing", 0.111760, "published",
         "4.400 in bore spacing, fixed by the LS block casting")
    _set(s, "block.min_wall_to_coolant", 0.0040, "estimated")
    _set(s, "block.liner_type", "cast-in-iron", "published")

    # The block description has to land before the bore, because it is what
    # sets the bore's bounds. Setting the bore first is refused -- which is
    # the constraint layer behaving correctly.
    refresh_bounds(s)

    _set(s, "engine.bore", 0.103251, "published")
    _set(s, "engine.stroke", 0.092000, "published")
    _set(s, "engine.rod_length", 0.154051, "published")
    _set(s, "engine.n_cylinders", 8, "published")
    _set(s, "engine.compression_ratio", 10.70, "published")
    _set(s, "engine.redline", rpm_to_rad_s(6600), "published")

    # Reciprocating masses. These are the weakest numbers in the file: weigh
    # the real parts before trusting any margin computed from them.
    _set(s, "masses.piston", 0.431, "estimated")
    _set(s, "masses.rings", 0.036, "estimated")
    _set(s, "masses.pin", 0.118, "estimated")
    _set(s, "masses.retainers", 0.000, "estimated")
    _set(s, "masses.rod_total", 0.464, "estimated")
    _set(s, "masses.rod_cg_from_small_end", 0.1110, "estimated")

    _set(s, "materials.piston", "A390-T6", "estimated")
    _set(s, "materials.pin", "4340", "estimated")
    _set(s, "materials.rod", "4340", "estimated")
    _set(s, "materials.sleeve", "grey-iron", "published")

    # Breathing and friction, calibrated against two published points:
    # 424 lb-ft (575 N.m) at 4600 rpm and 430 hp at 5900 rpm. Two points do
    # not validate a model -- they calibrate two curves that were always
    # going to be calibrated. The structural load numbers this tool exists to
    # produce do not depend on that fit; they depend on peak cylinder
    # pressure and reciprocating mass, which come from elsewhere.
    _set(s, "operating.ve_peak", 1.00, "estimated")
    _set(s, "operating.ve_peak_speed", rpm_to_rad_s(4600), "estimated")
    _set(s, "operating.ve_breadth", 0.75, "estimated")
    _set(s, "operating.speed", rpm_to_rad_s(4600))
    _set(s, "operating.intake_pressure", 99_000.0, "estimated")
    _set(s, "operating.intake_temperature", 325.0, "estimated")
    _set(s, "operating.soc_angle", math.radians(-18.0), "estimated")
    _set(s, "operating.burn_duration", math.radians(48.0), "estimated")
    _set(s, "operating.heat_loss_fraction", 0.12, "estimated")
    _set(s, "operating.fmep_speed2_coeff", 120.0, "estimated")

    # --- Component geometry (phase 2) -------------------------------------
    # Pin and rod dimensions are reverse-engineered from published part
    # weights and the crank journal size, not measured. They are consistent
    # with the masses above, which is the most that can be said for them.
    _set(s, "piston.crown_thickness", 0.0065, "estimated")
    _set(s, "piston.crown_support_radius_fraction", 0.30, "estimated")
    _set(s, "piston.skirt_length", 0.0300, "estimated")
    _set(s, "piston.skirt_bearing_arc", 1.00, "estimated")
    _set(s, "piston.top_land_height", 0.0050, "estimated")
    _set(s, "piston.ring_groove_depth", 0.0035, "estimated")
    _set(s, "piston.boss_inner_span", 0.0240, "estimated")
    _set(s, "piston.compression_height", 0.028194, "published")
    _set(s, "piston.total_height", 0.0440, "estimated")
    _set(s, "piston.wall_thickness", 0.0030, "estimated")
    _set(s, "piston.second_land_height", 0.0035, "estimated")
    _set(s, "piston.oil_ring_height", 0.0030, "estimated")
    _set(s, "piston.oil_groove_depth", 0.0040, "estimated")
    _set(s, "piston.skirt_width_fraction", 0.60, "estimated")

    _set(s, "pin.outer_diameter", 0.023950, "published")
    _set(s, "pin.inner_diameter", 0.016600, "estimated")
    _set(s, "pin.length", 0.063500, "estimated")
    _set(s, "pin.support_span_factor", 0.50, "estimated")

    _set(s, "small_end.bushing_width", 0.023500, "estimated")
    _set(s, "small_end.material", "4340", "estimated")
    _set(s, "small_end.allowable_pressure", 120.0e6, "estimated")

    _set(s, "rod.shank_height", 0.0280, "estimated")
    _set(s, "rod.shank_width", 0.0170, "estimated")
    _set(s, "rod.shank_web", 0.0050, "estimated")
    _set(s, "rod.shank_flange", 0.0055, "estimated")
    _set(s, "rod.small_end_outer_diameter", 0.0310, "estimated")
    _set(s, "rod.big_end_outer_diameter", 0.0650, "estimated")
    _set(s, "rod.hardware_mass", 0.045, "estimated")
    _set(s, "rod.shank_taper", 1.25, "estimated")
    _set(s, "rod.big_end_bore", 0.053340, "published")
    _set(s, "rod.big_end_width", 0.0200, "estimated")
    # Production rods are shot-peened, which Shigley's table does not credit.
    # "machined" is the conservative stand-in; "as-forged" would model a rough
    # scaled forging surface and is far too harsh for a finished rod.
    _set(s, "rod.surface_finish", "machined", "estimated")

    _set(s, "bolts.count", 2, "published")
    _set(s, "bolts.thread_diameter", 0.0090, "published")
    _set(s, "bolts.thread_pitch", 0.0010, "estimated")
    _set(s, "bolts.preload", 35_000.0, "estimated")
    _set(s, "bolts.proof_strength", 1100.0e6, "estimated")
    _set(s, "bolts.cap_mass", 0.150, "estimated")

    # Conductances scale roughly with crown area, which is 44% larger than
    # the 86 mm default these were calibrated on.
    _set(s, "thermal.conductance_rings", 13.0, "estimated")
    _set(s, "thermal.conductance_skirt", 4.3, "estimated")
    _set(s, "thermal.conductance_oil", 5.8, "estimated")

    # The piston, pin and rod solids were sized so their measured masses land
    # within a few percent of the published part weights. That is the geometry
    # being calibrated to the masses, not the reverse -- the masses are the
    # better-grounded data. What the solids then add is the rod's centre of
    # mass, which was previously a rule of thumb.
    s.meta["notes"] = (
        "Published output: 430 hp at 5900 rpm, 424 lb-ft (575 N.m) at 4600 rpm. "
        "This tool reports INDICATED figures, which should sit roughly 10-15% "
        "above those brake numbers.")
    s.meta["provenance"] = (
        "bore, stroke, rod length, bore spacing and compression ratio are "
        "published LS3 figures; all masses and the breathing calibration are "
        "estimates and must be replaced with measurements before any margin "
        "computed from them is trusted. Pin and rod section dimensions are "
        "reverse-engineered from published part weights and the journal size: "
        "they are self-consistent, not measured. All masses are now MEASURED "
        "from those solids rather than entered, so the design is internally "
        "consistent; the rod mass in particular disagrees with a commonly "
        "quoted figure and the geometry is believed over it")
    refresh_bounds(s)

    # Adopt the masses the solids actually weigh. The piston and pin land
    # within about 1% of their published weights; the rod does not, and the
    # geometry is the more trustworthy of the two here -- a 464 g figure is
    # widely quoted for an LS-family rod but appears to describe the LS7
    # titanium part rather than the LS3 powdered-metal one, and the solid
    # comes out near 640 g, which matches what PM steel rods actually weigh.
    from psrt import geometry as _geometry
    s = _geometry.adopt_masses(s)
    refresh_bounds(s)
    return s


def build_generic_i4():
    s = default_state("Generic 2.0L I4")
    s.meta["notes"] = ("Placeholder values throughout. Useful for exercising "
                       "the tool, useless for any real conclusion.")
    refresh_bounds(s)
    return s


def main():
    for filename, builder in (("ls3.json", build_ls3),
                              ("generic-i4.json", build_generic_i4)):
        state = builder()
        issues = state.validate()
        if issues:
            raise SystemExit(f"{filename} is not valid: {issues}")
        path = os.path.join(HERE, filename)
        state.save(path)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
