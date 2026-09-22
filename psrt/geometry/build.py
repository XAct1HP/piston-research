"""Parametric solid builds, driven by the design state.

Each build is a feature sequence that mirrors how the part is actually made,
so changing a parameter rebuilds a real solid rather than scaling a mesh. One
solid then feeds four consumers: exact mass properties, STEP export for a
machinist, a watertight body for phase 6 to mesh, and a tessellation for the
phase 4 viewport.

Geometry is modelled in MILLIMETRES, because that is what CAD kernels are
comfortable with and what a drawing reads. Everything leaving this module goes
through :mod:`psrt.geometry.properties`, which converts to SI.

Frames, chosen so each part is measured in the way the load chain wants it:

* piston -- axis Z, crown face at z = 0, body running down to negative z, pin
  bore along Y
* pin -- axis Y, centred on the origin
* rod -- axis Z, small-end centre at z = 0, big-end centre at z = rod length,
  both bores along Y. Centre of mass then reads directly as the distance from
  the small end, which is exactly what the two-mass split needs
* sleeve -- axis Z

These are single-piece idealisations. No fillets, no ring-groove chamfers, no
valve reliefs, no oil drain holes, no forging draft. Every one of those omitted
features removes material, so a mass from this module runs slightly heavy --
which the exactness check in :mod:`psrt.geometry` reports rather than hides.
"""

from __future__ import annotations

import math

import cadquery as cq

from . import profiles

MM = 1000.0          # metres -> millimetres


def _mm(state, path: str) -> float:
    return state[path] * MM


def _ring_cutter(outer_d: float, depth: float, z_top: float, height: float):
    """An annular groove cutter: removes ``depth`` radially over ``height``."""
    pad = max(outer_d * 0.1, 2.0)
    outer = (cq.Workplane("XY", origin=(0, 0, z_top - height))
             .circle(outer_d / 2.0 + pad).extrude(height))
    inner = (cq.Workplane("XY", origin=(0, 0, z_top - height - pad))
             .circle(outer_d / 2.0 - depth).extrude(height + 2 * pad))
    return outer.cut(inner)


def _cylinder_along_y(radius: float, width: float, z_centre: float):
    return cq.Workplane(obj=cq.Solid.makeCylinder(
        radius, width, cq.Vector(0.0, -width / 2.0, z_centre),
        cq.Vector(0.0, 1.0, 0.0)))


# --- piston ----------------------------------------------------------------

def build_piston(state):
    """Crown, ring belt, under-crown cavity, pin bosses, skirt."""
    profile = profiles.get(state["block.bore_profile"])

    bore = _mm(state, "engine.bore")
    outer_d = bore - _mm(state, "piston.cold_clearance")
    radius = outer_d / 2.0

    height = _mm(state, "piston.total_height")
    crown_t = _mm(state, "piston.crown_thickness")
    groove_d = _mm(state, "piston.ring_groove_depth")
    ring_h = _mm(state, "piston.ring_axial_height")
    top_land = _mm(state, "piston.top_land_height")
    mid_land = _mm(state, "piston.second_land_height")
    oil_h = _mm(state, "piston.oil_ring_height")
    oil_d = _mm(state, "piston.oil_groove_depth")
    wall = _mm(state, "piston.wall_thickness")
    comp_height = _mm(state, "piston.compression_height")
    pin_d = _mm(state, "pin.outer_diameter")
    pin_len = _mm(state, "pin.length")
    boss_gap = _mm(state, "piston.boss_inner_span")
    skirt_w = outer_d * state["piston.skirt_width_fraction"]

    if crown_t >= height:
        raise ValueError("crown thickness exceeds the whole piston height")
    if comp_height + pin_d / 2.0 > height:
        raise ValueError(
            "the pin bore falls below the bottom of the piston: compression "
            "height plus half the pin diameter exceeds total height")

    # 1. The blank, crown face at z = 0.
    body = profile.build(cq.Workplane("XY"), radius).extrude(-height)

    # 2. Ring grooves: top compression, second compression, oil control.
    z = -top_land
    for depth, groove_height in ((groove_d, ring_h), (groove_d, ring_h),
                                 (oil_d, oil_h)):
        body = body.cut(_ring_cutter(outer_d, depth, z, groove_height))
        z -= groove_height + mid_land

    # 3. Under-crown cavity, leaving the crown and the ring-belt wall.
    cavity_r = radius - groove_d - wall
    if cavity_r <= pin_d:
        raise ValueError(
            "the under-crown cavity is smaller than the pin: the ring belt "
            "wall and groove depth leave nothing to hollow out")
    body = body.cut(
        cq.Workplane("XY", origin=(0, 0, -height)).circle(cavity_r)
        .extrude(height - crown_t))

    # 4. Pin bosses. A pad around the pin bore, carried up to the crown
    #    underside so it is structurally connected rather than floating. Not
    #    a slab across the whole bore -- that was an early version of this
    #    build and it made the piston 40% heavy.
    boss_width = pin_d * state["piston.boss_width_factor"]
    boss_bottom = -(comp_height + pin_d * 0.75)
    boss_top = -crown_t
    bridge = (cq.Workplane("XY", origin=(0, 0, boss_bottom))
              .box(boss_width, pin_len, boss_top - boss_bottom,
                   centered=(True, True, False)))
    envelope = profile.build(cq.Workplane("XY"), radius).extrude(-height)
    body = body.union(bridge.intersect(envelope))

    # 5. Open the space between the bosses so the rod small end can swing,
    #    from the bottom of the piston up to just above the pin bore.
    slot_top = -(comp_height - pin_d * 0.60)
    body = body.cut(
        cq.Workplane("XY", origin=(0, 0, -height))
        .box(outer_d * 1.5, boss_gap, slot_top + height,
             centered=(True, True, False)))

    # 6. Pin bore.
    body = body.cut(_cylinder_along_y(pin_d / 2.0, pin_len * 1.5, -comp_height))

    # 7. Skirt relief: take the pin-axis ends away below the ring belt.
    belt = _mm(state, "piston.ring_belt_height")
    for sign in (1.0, -1.0):
        body = body.cut(
            cq.Workplane("XY", origin=(0, sign * (skirt_w / 2.0 + outer_d / 2.0),
                                       -height))
            .box(outer_d * 2.0, outer_d, height - belt,
                 centered=(True, True, False)))

    # 8. Crown dish.
    dish_volume = state["piston.crown_dish_volume"] * MM ** 3
    if dish_volume > 0.0:
        dish_r = radius * 0.80
        depth = dish_volume / (math.pi * dish_r ** 2)
        if depth >= crown_t:
            raise ValueError(
                f"a {state['piston.crown_dish_volume'] * 1e6:.1f} cc dish needs "
                f"{depth:.2f} mm of depth but the crown is only "
                f"{crown_t:.2f} mm thick")
        body = body.cut(
            cq.Workplane("XY", origin=(0, 0, -depth)).circle(dish_r)
            .extrude(depth + 1.0))

    return body


# --- pin -------------------------------------------------------------------

def build_pin(state):
    """A plain tube. The simplest part in the system and the hardest worked."""
    outer = _mm(state, "pin.outer_diameter")
    inner = _mm(state, "pin.inner_diameter")
    length = _mm(state, "pin.length")
    pin = _cylinder_along_y(outer / 2.0, length, 0.0)
    if inner > 0.0:
        pin = pin.cut(_cylinder_along_y(inner / 2.0, length * 1.5, 0.0))
    return pin


# --- connecting rod --------------------------------------------------------

def build_rod(state):
    """Small-end ring, I-section shank, big-end ring.

    The section orientation follows :mod:`psrt.sections`: the shank height
    lies in the plane of rotation (X here), the flange width runs along the
    crank axis (Y), and the web is thin along Y.
    """
    length = _mm(state, "engine.rod_length")
    small_od = _mm(state, "rod.small_end_outer_diameter")
    small_id = _mm(state, "pin.outer_diameter")
    small_w = _mm(state, "small_end.bushing_width")
    big_od = _mm(state, "rod.big_end_outer_diameter")
    big_id = _mm(state, "rod.big_end_bore")
    big_w = _mm(state, "rod.big_end_width")

    height = _mm(state, "rod.shank_height")
    width = _mm(state, "rod.shank_width")
    web = _mm(state, "rod.shank_web")
    flange = _mm(state, "rod.shank_flange")

    if small_id >= small_od:
        raise ValueError("the small-end bore is larger than its outside diameter")
    if big_id >= big_od:
        raise ValueError("the big-end bore is larger than its outside diameter")

    small = _cylinder_along_y(small_od / 2.0, small_w, 0.0).cut(
        _cylinder_along_y(small_id / 2.0, small_w * 1.5, 0.0))
    big = _cylinder_along_y(big_od / 2.0, big_w, length).cut(
        _cylinder_along_y(big_id / 2.0, big_w * 1.5, length))

    # Bolt bosses. The rod bolts run parallel to the rod axis through pads on
    # either side of the big end, and those pads carry real mass -- enough to
    # move the rod's centre of mass by more than a tenth of its length. An
    # earlier version of this build left them out and put the centre of mass
    # at 0.59 L instead of the 0.7-0.75 a real rod shows.
    bolt_d = _mm(state, "bolts.thread_diameter")
    boss_d = bolt_d * 1.9          # pad diameter around the bolt
    boss_z = bolt_d * 1.7          # pad length along the rod axis
    x_bolt = big_id / 2.0 + boss_d / 2.0
    for sign in (1.0, -1.0):
        big = big.union(
            cq.Workplane("XY", origin=(sign * x_bolt, 0.0, length - boss_z / 2.0))
            .box(boss_d, big_w, boss_z, centered=(True, True, False)))
        big = big.cut(_cylinder_along_y(
            big_id / 2.0, big_w * 1.5, length))

    # The shank runs between the two ends, overlapping each so the union is
    # one solid rather than three touching ones.
    z0 = small_od / 2.0 * 0.6
    z1 = length - big_od / 2.0 * 0.6
    span = z1 - z0
    if span <= 0.0:
        raise ValueError(
            "the small and big ends overlap: there is no room for a shank")

    # The shank tapers: narrow at the small end, wide at the big end, as real
    # rods are. This is not cosmetic -- it redistributes shank mass toward the
    # big end and changes the rod's centre of mass, which feeds straight into
    # the reciprocating/rotating split and therefore into rod tension.
    #
    # Which WAY the rod's centre of mass moves depends on the rod. The taper
    # adds material around the shank's own centroid, so if that sits below the
    # whole rod's centre of mass the net effect is to pull the centre of mass
    # DOWN, even though the shank's share of it went up. Both directions show
    # up across the bundled examples. Do not assume the sign.
    taper = state["rod.shank_taper"]
    shank = (cq.Workplane("XY", origin=(0.0, 0.0, z0))
             .polyline(_i_outline(height, width, web, flange)).close()
             .workplane(offset=span)
             .polyline(_i_outline(height * taper, width * taper,
                                  web * taper, flange * taper)).close()
             .loft(ruled=True))

    return small.union(shank).union(big)


def _i_outline(height: float, width: float, web: float, flange: float) -> list:
    """Closed outline of an I-section, as points in the section plane.

    X runs along the section height (the plane of rotation), Y along the
    flange width (the crank axis), matching :mod:`psrt.sections`.

    The two guards below are not pedantry. Without them an impossible section
    -- flanges thicker than half the height, or a web wider than the flange
    -- still produces an outline, but one that crosses itself. OpenCascade
    extrudes it into a solid of zero or NEGATIVE volume, which is not
    rejected anywhere: it becomes a zero mass, a meaningless inertia tensor,
    and finally `numpy.linalg.LinAlgError: Eigenvalues did not converge` out
    of the principal-moment decomposition, thrown from a module that has
    nothing to do with the mistake. Better to say which dimension is wrong.
    """
    if flange * 2.0 >= height:
        raise ValueError(
            f"the rod's I-section is impossible: two {flange:.2f} mm "
            f"flanges do not fit inside a {height:.2f} mm shank height. "
            "Reduce rod.shank_flange or raise rod.shank_height")
    if web >= width:
        raise ValueError(
            f"the rod's I-section is impossible: a {web:.2f} mm web is not "
            f"narrower than the {width:.2f} mm flange width. Reduce "
            "rod.shank_web or raise rod.shank_width")

    h, b = height / 2.0, width / 2.0
    inner, w = h - flange, web / 2.0
    return [
        (h, b), (h, -b), (inner, -b), (inner, -w),
        (-inner, -w), (-inner, -b), (-h, -b), (-h, b),
        (-inner, b), (-inner, w), (inner, w), (inner, b),
    ]


# --- sleeve ----------------------------------------------------------------

def build_sleeve(state):
    """The liner, as a plain annular tube over the piston's travel."""
    profile = profiles.get(state["block.bore_profile"])
    bore_r = _mm(state, "engine.bore") / 2.0
    wall = _mm(state, "sleeve.wall_thickness")
    length = (_mm(state, "engine.stroke") + _mm(state, "piston.total_height")
              + 10.0)

    outer = (cq.Workplane("XY", origin=(0, 0, -length))
             .circle(bore_r + wall).extrude(length))
    inner = profile.build(
        cq.Workplane("XY", origin=(0, 0, -length - 5.0)), bore_r
    ).extrude(length + 10.0)
    return outer.cut(inner)


BUILDS = {
    "piston": (build_piston, "piston"),
    "pin": (build_pin, "pin"),
    "rod": (build_rod, "rod"),
    "sleeve": (build_sleeve, "sleeve"),
}


def tessellate_solid(solid, tolerance: float, angular: float = 0.35) -> tuple:
    """Triangulate a solid at a tolerance that is actually honoured.

    OpenCascade caches a triangulation on the shape once one has been
    computed, and ``tessellate`` silently reuses it -- so every call after the
    first returns the SAME mesh no matter what tolerance is asked for. The
    gudgeon pin came back as 1,008 triangles at every tolerance from 0.05 to
    2.5 mm. Clearing the cached triangulation first is what makes the
    parameter mean anything.
    """
    from OCP.BRepTools import BRepTools

    shape = solid.wrapped if hasattr(solid, "wrapped") else solid
    BRepTools.Clean_s(shape)
    vertices, triangles = solid.tessellate(tolerance, angular)
    return vertices, triangles
