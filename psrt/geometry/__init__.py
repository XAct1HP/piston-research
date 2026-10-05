"""The geometry kernel: exact mass properties, the exactness check, export.

Phase 1 and 2 ran on masses somebody typed in. Every structural margin was
computed from them, and the tool said so. This module is where that stops.

The architecture point worth keeping: **the analytical layer never calls the
geometry kernel.** A CadQuery piston rebuild takes a couple of hundred
milliseconds, and the optimiser in phase 7 needs thousands of evaluations a
second. So the fast path stays closed-form over the same parameters, and the
kernel is invoked for three things only -- display, meshing, and the
*exactness check*: rebuild the solids, compare their true mass and centre of
mass against what the design state claims, and report the error loudly.

If the error drifts past a few percent, the analytical model needs fixing, and
the tool should say so rather than quietly carrying on.
"""

from __future__ import annotations

import math
from collections import OrderedDict

from .. import materials as materials_mod
from ..state import DesignState
from . import build as build_mod
from . import profiles
from .properties import MassProperties, combine, measure

_CACHE: "OrderedDict[str, dict]" = OrderedDict()
_CACHE_LIMIT = 32

# Which design-state parameters actually change the solids. The cache key is
# built from these alone, so changing engine speed never triggers a rebuild.
GEOMETRY_PATHS = (
    "engine.bore", "engine.stroke", "engine.rod_length",
    "block.bore_profile",
    "piston.crown_thickness", "piston.total_height", "piston.compression_height",
    "piston.wall_thickness", "piston.top_land_height", "piston.second_land_height",
    "piston.ring_axial_height", "piston.ring_groove_depth",
    "piston.oil_ring_height", "piston.oil_groove_depth",
    "piston.crown_dish_volume", "piston.skirt_width_fraction",
    "piston.cold_clearance", "piston.boss_inner_span",
    "pin.outer_diameter", "pin.inner_diameter", "pin.length",
    "small_end.bushing_width",
    "rod.shank_height", "rod.shank_width", "rod.shank_web", "rod.shank_flange",
    "rod.shank_taper",
    "rod.small_end_outer_diameter", "rod.big_end_outer_diameter",
    "rod.big_end_bore", "rod.big_end_width", "bolts.thread_diameter",
    "rod.hardware_mass",
    "sleeve.wall_thickness_measured", "block.bore_spacing",
    "materials.piston", "materials.pin", "materials.rod", "materials.sleeve",
)


# Rail-rod parameters that change how the parts are LOADED but not their
# shape. Leaving them out of the key means a preload or fit study re-uses the
# solids, the meshes and the unit FEA solutions instead of rebuilding them.
RAILROD_LOAD_ONLY = frozenset({
    "bolt_preload", "seat_interference", "contact_depth", "bracing_fraction",
    "bolt_proof_strength", "bolt_finish", "shell_mass",
})


def _geometry_payload(state: DesignState) -> dict:
    payload = {p: state.get(p, None) for p in GEOMETRY_PATHS}
    # Every rail-rod dimension shapes a solid, so when the concept is on the
    # whole section is part of the key. Off, it contributes nothing and the
    # conventional cache is untouched.
    if state.has("railrod.enabled") and state["railrod.enabled"]:
        for name in state.sections.get("railrod", {}):
            if name in RAILROD_LOAD_ONLY:
                continue
            payload[f"railrod.{name}"] = state.get(f"railrod.{name}", None)
    return payload


def fingerprint(state: DesignState) -> str:
    import hashlib
    import json
    payload = _geometry_payload(state)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def build_all(state: DesignState, use_cache: bool = True) -> dict:
    """Build every part and measure it. Returns {name: (solid, properties)}."""
    key = fingerprint(state)
    if use_cache and key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]

    out = {}
    railrod_on = state.has("railrod.enabled") and state["railrod.enabled"]
    for name, (builder, role) in build_mod.BUILDS.items():
        if railrod_on and name == "rod":
            continue
        solid = builder(state)
        material = materials_mod.get(state[f"materials.{role}"])
        out[name] = {"solid": solid, "properties": measure(solid, material)}

    if railrod_on:
        # The rail rod replaces the conventional rod with six separate
        # solids, each measured in its own material.
        from ..railrod import cad as rr_cad
        for name, solid in rr_cad.build_parts(state).items():
            material = materials_mod.get(rr_cad.part_material(state, name))
            out[name] = {"solid": solid, "properties": measure(solid, material)}

    if use_cache:
        _CACHE[key] = out
        while len(_CACHE) > _CACHE_LIMIT:
            _CACHE.popitem(last=False)
    return out


# --- derived masses --------------------------------------------------------

def masses_from_geometry(state: DesignState) -> dict:
    """What the solids say the parts weigh, and where the rod balances.

    The rod's centre of mass is the valuable one. The two-mass split that sets
    reciprocating mass depends on it directly, and up to now it was a guess
    with the familiar "about 0.72 of rod length" behind it. Here it is measured.

    Rod hardware -- bolts, bearing shells, cap features the solid does not
    carry -- is added as a lump at the big-end centre, which is where it sits.
    """
    parts = build_all(state)
    piston = parts["piston"]["properties"]
    pin = parts["pin"]["properties"]
    if "rod" in parts:
        rod = parts["rod"]["properties"]
        hardware = state["rod.hardware_mass"]
    else:
        # Rail rod: every part is a real solid, the bolt included, so the
        # only hardware left to lump at the big end is the bearing shells.
        #
        # One part is not ALL solid. The sleeve's B-rep is its skins, face
        # shells and bearing pad; its lattice core is a mesh, so measuring
        # the solids alone loses the core and under-reports the rod by the
        # weight of it. Add the core back as its own body at the density the
        # grading averages to -- which is exactly what
        # psrt.railrod.analysis.layout_masses does, and the two paths are
        # checked against each other in the test suite.
        import dataclasses

        from ..railrod import lattice as lattice_mod
        from ..railrod.cad import PART_NAMES, sleeve_core_box
        from ..railrod.layout import build_layout

        measured = [parts[n]["properties"] for n in PART_NAMES]
        layout = build_layout(state)
        core = sleeve_core_box(layout)
        if core is not None:
            alloy = materials_mod.get(state["railrod.sleeve_material"])
            fraction = lattice_mod.mean_density(
                layout.lattice_rho_mid, layout.lattice_rho_end,
                layout.lattice_exponent)
            measured.append(measure(core, dataclasses.replace(
                alloy, density=alloy.density * fraction)))
        rod = combine(measured)
        hardware = state["railrod.shell_mass"]

    length = state["engine.rod_length"]
    body_mass = rod.mass
    body_cg = rod.centre_of_mass[2]

    total_mass = body_mass + hardware
    cg = (body_mass * body_cg + hardware * length) / total_mass

    return {
        "piston_kg": piston.mass,
        "pin_kg": pin.mass,
        "rod_body_kg": body_mass,
        "rod_hardware_kg": hardware,
        "rod_total_kg": total_mass,
        "rod_cg_from_small_end_m": cg,
        "rod_cg_fraction_of_length": cg / length,
        "rod_body_cg_from_small_end_m": body_cg,
        "sleeve_kg": parts["sleeve"]["properties"].mass,
        "piston_inertia_kg_m2": piston.inertia_centroidal.tolist(),
    }


def check_masses(state: DesignState, tolerance: float = 0.03) -> dict:
    """Compare what the design state claims against what the solids weigh.

    This is the check that keeps the fast analytical layer honest. It runs on
    demand rather than on every evaluation, because a rebuild costs roughly a
    thousand times what an analytical evaluation does.
    """
    measured = masses_from_geometry(state)
    rows = []
    for label, path, value in (
            ("piston", "masses.piston", measured["piston_kg"]),
            ("gudgeon pin", "masses.pin", measured["pin_kg"]),
            ("connecting rod", "masses.rod_total", measured["rod_total_kg"]),
            ("rod CG from small end", "masses.rod_cg_from_small_end",
             measured["rod_cg_from_small_end_m"])):
        specified = state[path]
        error = ((value - specified) / specified) if specified else float("nan")
        rows.append({
            "quantity": label, "path": path,
            "specified": specified, "from_geometry": value,
            "relative_error": error,
            "within_tolerance": abs(error) <= tolerance,
        })

    # Reciprocating mass is what actually matters: it sets the inertia force,
    # which sets rod tension, which is what breaks rods.
    spec_recip = state["masses.reciprocating"]
    geo_recip = (measured["piston_kg"] + state["masses.rings"]
                 + measured["pin_kg"] + state["masses.retainers"]
                 + measured["rod_total_kg"]
                 * (1.0 - measured["rod_cg_fraction_of_length"]))
    recip_error = (geo_recip - spec_recip) / spec_recip

    worst = max(abs(r["relative_error"]) for r in rows
                if not math.isnan(r["relative_error"]))

    notes = [
        "the solid models carry no fillets, ring-groove chamfers, valve "
        "reliefs, oil drain holes or forging draft. Every one of those removes "
        "material, so a geometry mass runs slightly heavy",
    ]
    if abs(recip_error) > tolerance:
        notes.append(
            f"reciprocating mass differs by {recip_error * 100:+.1f}%. That "
            "propagates straight into inertia force, rod tension and every "
            "margin downstream of them")

    return {
        "rows": rows,
        "reciprocating_specified_kg": spec_recip,
        "reciprocating_from_geometry_kg": geo_recip,
        "reciprocating_relative_error": recip_error,
        "worst_relative_error": worst,
        "tolerance": tolerance,
        "passes": worst <= tolerance and abs(recip_error) <= tolerance,
        "notes": notes,
    }


def adopt_masses(state: DesignState, actor: str = "geometry") -> DesignState:
    """Return a new state whose masses come from the solids.

    After this the design is self-consistent: the masses driving the load
    chain are the masses of the parts that would actually be made.
    """
    measured = masses_from_geometry(state)
    return state.with_changes(
        {
            "masses.piston": measured["piston_kg"],
            "masses.pin": measured["pin_kg"],
            "masses.rod_total": measured["rod_total_kg"],
            "masses.rod_cg_from_small_end": measured["rod_cg_from_small_end_m"],
        },
        actor=actor,
        rationale="measured from the CadQuery solids rather than estimated")


# --- export ----------------------------------------------------------------

def export(state: DesignState, directory: str, fmt: str = "step") -> list:
    """Write every part to STEP or STL. Returns the paths written."""
    import os

    from cadquery import exporters

    os.makedirs(directory, exist_ok=True)
    parts = build_all(state)
    name = state.meta.get("name", "design").replace(" ", "-").lower()
    written = []
    for part, entry in parts.items():
        path = os.path.join(directory, f"{name}-{part}.{fmt}")
        exporters.export(entry["solid"], path)
        written.append(path)
    return written


def tessellate(state: DesignState, tolerance: float = 0.1) -> dict:
    """Triangles and normals per part, ready for three.js in phase 4."""
    parts = build_all(state)
    out = {}
    for part, entry in parts.items():
        solid = entry["solid"].val() if hasattr(entry["solid"], "val") \
            else entry["solid"]
        vertices, triangles = build_mod.tessellate_solid(solid, tolerance)
        out[part] = {
            "vertices": [[v.x, v.y, v.z] for v in vertices],
            "triangles": [list(t) for t in triangles],
            "vertex_count": len(vertices),
            "triangle_count": len(triangles),
        }
    return out


def profile_study(state: DesignState) -> dict:
    """What each bore cross-section could displace inside this block.

    The envelope radius is whatever the block's subtractive limit allows, so
    the comparison is against what you could actually machine.
    """
    envelope = state.param("engine.bore").maximum or state["engine.bore"]
    radius = envelope / 2.0
    rows = profiles.compare(radius)
    stroke = state["engine.stroke"]
    cylinders = state["engine.n_cylinders"]
    for row in rows:
        row["displacement_total_m3"] = row["area_m2"] * stroke * cylinders
    return {
        "envelope_radius_m": radius,
        "envelope_diameter_m": envelope,
        "current_bore_m": state["engine.bore"],
        "rows": rows,
    }
