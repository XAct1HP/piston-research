"""Solid models of the rail rod's parts, one solid per part.

Each part is its own solid so that it can be meshed, loaded and analysed on
its own, which is what the concept needs: the load moves from one path to
another around the cycle (compression through the rail feet, tension through
the notches), and each part has to be checked for the loads IT sees.

Frame: the rod frame the rest of the tool uses -- small-end centre at the
origin, big-end centre at z = rod length, both bores along Y, millimetres. So
the parts drop into the viewport exactly where the conventional rod does.

The critical surfaces are built exactly. The rail notch is cut with true
circular arcs and a true line, from the same numbers
:class:`psrt.railrod.layout.Notch` holds. The receiver and the clamps are
derived by boolean offsets in :mod:`psrt.railrod.layout` so that they conform
to their neighbours; their profiles are brought in here as lines where the
profile is straight and interpolating splines through densely sampled points
where it is curved, joined tangent-continuously wherever the profile itself is
smooth. The spline error is well under a micron on these radii.
"""

from __future__ import annotations

import math

import numpy as np

from .layout import Layout, arc_points

LINE_MIN = 0.6           # mm: a profile segment this long is a straight run
CORNER_DEG = 6.0         # a turn sharper than this is a real corner
SHOULDER_RELIEF = 0.02   # mm: keeps coincident faces out of the booleans


def _cq():
    import cadquery as cq
    return cq


# --- profile to solid -----------------------------------------------------------

def _clean_ring(coords) -> np.ndarray:
    pts = np.asarray(coords, dtype=float)
    if np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    keep = [0]
    for i in range(1, len(pts)):
        if np.linalg.norm(pts[i] - pts[keep[-1]]) > 1e-5:
            keep.append(i)
    pts = pts[keep]
    if np.linalg.norm(pts[0] - pts[-1]) < 1e-5:
        pts = pts[:-1]
    return pts


RESAMPLE = 0.05          # mm: spline interpolation points on curved runs
MAX_EDGE = 6.0           # mm: longest single spline edge
MIN_EDGE = 0.08          # mm: shortest edge worth keeping


def _resample(chunk: np.ndarray, spacing: float) -> np.ndarray:
    """Even arc-length spacing along a polyline, ends kept exactly.

    Boolean offsets leave curved runs with wildly uneven vertex spacing -- a
    0.6 mm edge break arrives as dozens of points a few microns apart next
    to quarter-millimetre chords on the bore. An interpolating spline through
    that oscillates between the clusters; the first version of this module
    put 2.4% of extra area into a clamp that way. Evenly spaced points fix
    it without moving the curve by more than the original chord sagitta.
    """
    seg = np.linalg.norm(np.diff(chunk, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    n = max(3, int(math.ceil(total / spacing)) + 1)
    t = np.linspace(0.0, total, n)
    return np.column_stack([np.interp(t, cum, chunk[:, 0]),
                            np.interp(t, cum, chunk[:, 1])])


def _fit_spline(points, t0=None, t1=None):
    """A smooth edge through ``points``, with few poles.

    An interpolating spline through a thousand points is exact on the curve
    but carries a thousand poles, and OpenCascade's mass-property integration
    over the extruded surface then goes quietly wrong -- a clamp came out 3%
    heavy that way while its outline was correct to 0.2 microns. A
    least-squares approximation to a micron has a few dozen poles and
    integrates correctly. Pinning end tangents on top of that was tried and
    made things worse (the tangent magnitude has to match OpenCascade's
    parametrisation, which it does not expose); at a smooth join the
    approximation's own end direction is within a fraction of a degree of
    the neighbour's, which is below anything the mesher resolves.
    """
    cq = _cq()
    return cq.Edge.makeSplineApprox(points, tol=5e-4, minDeg=3, maxDeg=8)


def ring_edges(coords, to3d) -> list:
    """Edges for one closed profile ring: lines on straight runs, splines on
    curved runs, split at genuine corners."""
    cq = _cq()
    pts = _clean_ring(coords)
    n = len(pts)
    seg = np.roll(pts, -1, axis=0) - pts                  # seg i: i -> i+1
    length = np.linalg.norm(seg, axis=1)
    straight = length > LINE_MIN
    unit = seg / length[:, None]

    breaks = []
    smooth_at = {}
    for i in range(n):
        prev = unit[i - 1]
        turn = math.degrees(math.acos(float(np.clip(prev @ unit[i], -1, 1))))
        if turn > CORNER_DEG or straight[i] or straight[i - 1]:
            breaks.append(i)
            smooth_at[i] = turn <= CORNER_DEG
    if len(breaks) < 2:
        breaks = sorted(set(breaks) | {0, n // 2})
        for b in breaks:
            smooth_at[b] = True

    # Drop breaks that would leave a sliver edge. Boolean offsets leave the
    # odd step a few microns long where two arcs meet; as its own edge it
    # becomes a face a few microns wide, and tetgen, which must honour every
    # boundary face, floods the part with elements to conform to it.
    cum_b = np.concatenate([[0.0], np.cumsum(length)])

    def arc_between(a, b):
        return (cum_b[b] - cum_b[a]) if b > a else (cum_b[n] - cum_b[a]
                                                     + cum_b[b])
    changed = True
    while changed and len(breaks) > 3:
        changed = False
        for k in range(len(breaks)):
            a, b = breaks[k], breaks[(k + 1) % len(breaks)]
            if arc_between(a, b) < MIN_EDGE:
                drop = b if not straight[b] else a
                breaks.remove(drop)
                smooth_at[breaks[k % len(breaks)]] = False
                changed = True
                break

    cum = np.concatenate([[0.0], np.cumsum(length)])
    perimeter = cum[-1]

    def point_at(arc):
        arc = arc % perimeter
        j = int(np.searchsorted(cum, arc, side="right") - 1)
        j = min(j, n - 1)
        f = (arc - cum[j]) / max(length[j], 1e-12)
        return pts[j] + f * seg[j]

    def tangent_at(i, forward):
        """Tangent at break vertex i, measured over a baseline rather than
        from one raw segment: boolean offsets leave micron-scale zigzags, and
        a tangent read off one of those points the spline the wrong way."""
        if straight[i] and (forward or not smooth_at.get(i, False)):
            if forward:
                return unit[i]
        if straight[i - 1] and (not forward or not smooth_at.get(i, False)):
            if not forward:
                return unit[i - 1]
        if smooth_at.get(i, False):
            if straight[i]:
                return unit[i]
            if straight[i - 1]:
                return unit[i - 1]
            a, b = point_at(cum[i] - 0.1), point_at(cum[i] + 0.1)
        elif forward:
            a, b = pts[i], point_at(cum[i] + 0.1)
        else:
            a, b = point_at(cum[i] - 0.1), pts[i]
        t = b - a
        return t / np.linalg.norm(t)

    edges = []
    for k, start in enumerate(breaks):
        end = breaks[(k + 1) % len(breaks)]
        idx = [(start + j) % n for j in range(((end - start) % n) + 1)]
        chunk = pts[idx]
        p0, p1 = to3d(chunk[0]), to3d(chunk[-1])
        if len(chunk) == 2:
            edges.append(cq.Edge.makeLine(p0, p1))
            continue
        t0 = tangent_at(start, True)
        t1 = tangent_at(end, False)
        chunk = _resample(chunk, RESAMPLE)
        # Long faces are split: OpenCascade integrates area and volume over
        # a face with a fixed number of Gauss points, and over an 80 mm
        # B-spline face that undercounts by 3% -- the clamp's mass was off by
        # exactly that while its outline was right to half a micron.
        pieces = max(1, int(math.ceil((len(chunk) - 1) * RESAMPLE
                                      / MAX_EDGE)))
        cuts = np.linspace(0, len(chunk) - 1, pieces + 1).round().astype(int)
        for a, b in zip(cuts[:-1], cuts[1:]):
            edges.append(_fit_spline([to3d(p) for p in chunk[a:b + 1]]))
    return edges


def profile_solid(polygon, layout: Layout, width: float, y0: float | None = None):
    """Extrude a local (x, v) profile along Y, centred on y = 0."""
    cq = _cq()

    def to3d(p, direction=False):
        if direction:
            return cq.Vector(float(p[0]), 0.0, -float(p[1]))
        return cq.Vector(float(p[0]), 0.0, layout.L - float(p[1]))

    outer = cq.Wire.assembleEdges(ring_edges(polygon.exterior.coords, to3d))
    inner = [cq.Wire.assembleEdges(ring_edges(r.coords, to3d))
             for r in polygon.interiors]
    face = cq.Face.makeFromWires(outer, inner)
    solid = cq.Solid.extrudeLinear(face, cq.Vector(0.0, width, 0.0))
    start = -width / 2.0 if y0 is None else y0
    return cq.Workplane("XY").add(solid.translate(cq.Vector(0.0, start, 0.0)))


def _box(x0, x1, y0, y1, v0, v1, layout):
    """Axis-aligned box in local (x, y, v)."""
    cq = _cq()
    z0, z1 = layout.L - v1, layout.L - v0
    return cq.Workplane("XY").add(cq.Solid.makeBox(
        x1 - x0, y1 - y0, z1 - z0, cq.Vector(x0, y0, z0)))


def _slot(x0, x1, y0, y1, v0, v1, radius, layout):
    """A rectangular pocket with radiused vertical corners.

    The corners run along the rod axis, which is the direction the cutter
    goes in, so this is the shape a machined slot actually has.
    """
    cq = _cq()
    z0, z1 = layout.L - v1, layout.L - v0
    body = cq.Workplane("XY").add(cq.Solid.makeBox(
        x1 - x0, y1 - y0, z1 - z0, cq.Vector(x0, y0, z0)))
    limit = 0.49 * min(x1 - x0, y1 - y0)
    radius = min(radius, limit)
    if radius <= 1e-6:
        return body
    try:
        filleted = body.val().fillet(radius, [
            e for e in body.val().Edges()
            if e.geomType() == "LINE"
            and abs(e.startPoint().z - e.endPoint().z) > 1e-6
            and abs(e.startPoint().x - e.endPoint().x) < 1e-9
            and abs(e.startPoint().y - e.endPoint().y) < 1e-9])
        if filleted.isValid():
            return cq.Workplane("XY").add(filleted)
    except Exception:                                      # noqa: BLE001
        pass
    return body


def _cyl_x(radius, x0, x1, v, layout):
    cq = _cq()
    return cq.Workplane("XY").add(cq.Solid.makeCylinder(
        radius, x1 - x0, cq.Vector(x0, 0.0, layout.L - v),
        cq.Vector(1.0, 0.0, 0.0)))


def _cyl_y(radius, width, v, layout):
    cq = _cq()
    return cq.Workplane("XY").add(cq.Solid.makeCylinder(
        radius, width, cq.Vector(0.0, -width / 2.0, layout.L - v),
        cq.Vector(0.0, 1.0, 0.0)))


# --- the parts -------------------------------------------------------------------

def _notch_cutter(layout: Layout, sign: float):
    """The void of one notch, with exact arcs, extruded through the rail."""
    cq = _cq()
    n = layout.notch
    L = layout.L

    def p(pt):
        return cq.Vector(sign * float(pt[0]), 0.0, L - float(pt[1]))

    c, cf = n.centre, n.blend_centre
    a_a = math.atan2(n.A[1] - c[1], n.A[0] - c[0])
    a_t = math.pi + n.alpha
    mid_top = c + n.r_top * np.array([math.cos((a_a + a_t) / 2),
                                      math.sin((a_a + a_t) / 2)])
    mid_low = cf + n.r_low * np.array([math.cos(n.alpha / 2),
                                       math.sin(n.alpha / 2)])
    out = layout.x_o + 2.0
    edges = [
        cq.Edge.makeThreePointArc(p(n.A), p(mid_top), p(n.T)),
        cq.Edge.makeLine(p(n.T), p(n.F1)),
        cq.Edge.makeThreePointArc(p(n.F1), p(mid_low), p(n.F2)),
        cq.Edge.makeLine(p(n.F2), p((out, n.F2[1]))),
        cq.Edge.makeLine(p((out, n.F2[1])), p((out, n.A[1] + 0.5))),
        cq.Edge.makeLine(p((out, n.A[1] + 0.5)), p((n.A[0], n.A[1] + 0.5))),
        cq.Edge.makeLine(p((n.A[0], n.A[1] + 0.5)), p(n.A)),
    ]
    face = cq.Face.makeFromWires(cq.Wire.assembleEdges(edges))
    depth = layout.rail_depth + 4.0
    solid = cq.Solid.extrudeLinear(face, cq.Vector(0.0, depth, 0.0))
    return cq.Workplane("XY").add(solid.translate(cq.Vector(0.0, -depth / 2, 0)))


def build_rails(layout: Layout, blend: float):
    """The monolithic small end and twin rails, notches cut exactly.

    Built from exact primitives -- a cylinder for the eye, one straight box
    per rail running from the foot into the eye -- rather than from the
    sampled 2D profile. Every face is then a true plane or cylinder, the
    rails have no joint along their length, and nothing is left for the
    mesher to trip over. The rail-to-eye blends are real fillets on the
    edges where the rail faces run into the eye.
    """
    cq = _cq()
    h = layout.rail_depth
    L = layout.L
    body = _cyl_y(layout.eye_r_out, layout.eye_width, L, layout)
    for sign in (1.0, -1.0):
        x0, x1 = sorted((sign * layout.x_i, sign * layout.x_o))
        body = body.union(_box(x0, x1, -h / 2.0, h / 2.0, layout.v_e,
                               L, layout))
    # The concave edges between the rail side faces (x = +-x_o, +-x_i) and
    # the eye: straight lines along Y, just below the eye.
    targets = {round(abs(v), 4) for v in (layout.x_o, layout.x_i)}

    def is_junction(edge):
        if edge.geomType() != "LINE":
            return False
        a, b = edge.startPoint(), edge.endPoint()
        if abs(a.x - b.x) > 1e-6 or abs(a.z - b.z) > 1e-6:
            return False
        if round(abs(a.x), 4) not in targets:
            return False
        return 0.0 < a.z < layout.eye_r_out + 1e-6 and abs(a.y - b.y) > 1.0

    solid = body.val()
    junction = [e for e in solid.Edges() if is_junction(e)]
    if junction and blend > 0:
        try:
            filleted = solid.fillet(blend, junction)
            if filleted.isValid():
                body = cq.Workplane("XY").add(filleted)
        except Exception:                                  # noqa: BLE001
            pass
    for sign in (1.0, -1.0):
        body = body.cut(_notch_cutter(layout, sign))
    body = body.cut(_cyl_y(layout.eye_r_in, layout.eye_width * 2, L,
                           layout))
    return body


def build_sleeve(layout: Layout, clearance: float):
    """The stabilising sleeve's SOLID part: two rail skins, two face shells.

    What comes back is the printed sleeve minus its lattice: a solid skin
    wrapped round each rail channel (the machined surfaces, the only ones
    that touch anything), a thin shell on each of the two big faces, a
    bearing pad at the bottom where the sleeve sits in the receiver's seat,
    and powder ports through the shells. The core between the skins is left
    empty here because a gyroid is not a thing to hand a solid kernel --
    :func:`build_sleeve_lattice` returns it as a mesh, which is what the
    machine wants anyway.

    With ``railrod.sleeve_core`` set to ``solid`` none of that happens and
    the sleeve comes back as the original one-piece block.
    """
    body = _sleeve_block(layout, clearance)
    if layout.sleeve_core_kind != "sheet-gyroid" or not layout.lattice_core:
        return body

    xc, yc, v0, v1 = layout.lattice_core
    body = body.cut(_box(-xc, xc, -yc, yc, v0, v1, layout))
    ports = _powder_ports(layout)
    if ports is not None:
        body = body.cut(ports)
    return body


def _sleeve_block(layout: Layout, clearance: float):
    """The sleeve's outside shape: a block round both rails, relieved to the
    eye. Nothing hollowed out of it yet."""
    sx, sy = layout.sleeve_x, layout.sleeve_y
    body = _box(-sx, sx, -sy, sy, layout.sleeve_v0, layout.L, layout)
    h = layout.rail_depth / 2.0 + clearance
    for sign in (1.0, -1.0):
        x0, x1 = sorted((sign * (layout.x_i - clearance),
                         sign * (layout.x_o + clearance)))
        body = body.cut(_box(x0, x1, -h, h, layout.sleeve_v0 - 1.0,
                             layout.L + 1.0, layout))
    top = layout.sleeve_cut.intersection(
        __import__("shapely").geometry.box(-sx - 2, layout.sleeve_v0 + 5,
                                           sx + 2, layout.L + 40))
    from .layout import _largest
    cutter = profile_solid(_largest(top), layout, 2 * sy + 4.0)
    return body.cut(cutter)


def build_sleeve_fea(layout: Layout, clearance: float):
    """The sleeve as ONE connected body, for meshing.

    The printed sleeve has a hole where its lattice is, which is honest about
    what the solid kernel can carry and useless to mesh: the core would fall
    out of the model and the skins would be left carrying loads they share
    with it. So the FEA meshes the block whole, ports and all, and the
    elements inside the core box are given the lattice's homogenised
    properties instead of the alloy's. One mesh, two materials, no tied
    contact to go wrong.
    """
    body = _sleeve_block(layout, clearance)
    ports = _powder_ports(layout)
    if ports is not None:
        body = body.cut(ports)
    return body


def _powder_ports(layout: Layout):
    """Ports through both face shells, so the core does not drain 100 mm.

    One staggered column per face on twice the cell pitch. That is enough:
    what matters is the distance from any cell to the nearest opening, and
    with a column down the middle that is never more than the core's own
    half width.
    """
    if not layout.port_stations or layout.lattice_port_diameter <= 0:
        return None
    cq = _cq()
    _, yc, _, _ = layout.lattice_core
    r = layout.lattice_port_diameter / 2.0
    solids = []
    for x, v in layout.port_stations:
        for sign in (1.0, -1.0):
            solids.append(cq.Solid.makeCylinder(
                r, layout.port_depth, cq.Vector(x, sign * yc, layout.L - v),
                cq.Vector(0.0, sign, 0.0)))
    return cq.Workplane("XY").add(cq.Compound.makeCompound(solids))


def build_sleeve_lattice(layout: Layout, resolution: float = 0.22):
    """The graded sheet gyroid as a triangle mesh: (vertices, faces).

    Raises :class:`psrt.railrod.lattice.LatticeError` if the sleeve is not a
    lattice one or scikit-image is not installed.
    """
    from . import lattice as lat
    if layout.sleeve_core_kind != "sheet-gyroid" or not layout.lattice_core:
        raise lat.LatticeError("this sleeve has no lattice core")
    xc, yc, v0, v1 = layout.lattice_core
    verts, faces = lat.mesh(
        (-xc, xc, -yc, yc, v0, v1), layout.lattice_cell,
        layout.lattice_rho_mid, layout.lattice_rho_end,
        layout.lattice_exponent, resolution=resolution)
    # local (x, y, v) to the rod frame (x, y, z = L - v)
    out = verts.copy()
    out[:, 2] = layout.L - verts[:, 2]
    return out, faces


def export_lattice_stl(layout: Layout, path: str,
                       resolution: float = 0.18) -> str:
    """Write the sleeve's lattice core as a binary STL.

    This is the half of the sleeve the solid kernel cannot carry, so it is
    exported on its own: the printer gets the skins and shells as a STEP and
    the core as a mesh, and unites them on the build plate. Writing the STL
    here rather than going through a mesh library keeps the export free of
    another dependency.
    """
    import struct

    verts, faces = build_sleeve_lattice(layout, resolution=resolution)
    tri = verts[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(n, axis=1)
    n = n / np.where(ln[:, None] == 0.0, 1.0, ln[:, None])
    with open(path, "wb") as fh:
        fh.write(b"psrt rail-rod sleeve lattice".ljust(80, b" "))
        fh.write(struct.pack("<I", len(tri)))
        block = np.zeros((len(tri), 12), dtype="<f4")
        block[:, 0:3] = n
        block[:, 3:6] = tri[:, 0]
        block[:, 6:9] = tri[:, 1]
        block[:, 9:12] = tri[:, 2]
        raw = np.zeros((len(tri), 50), dtype=np.uint8)
        raw[:, :48] = block.view(np.uint8).reshape(len(tri), 48)
        fh.write(raw.tobytes())
    return path


def sleeve_core_box(layout: Layout):
    """The core region as a solid, for the homogenised continuum the FEA
    stands in for the lattice with."""
    if layout.sleeve_core_kind != "sheet-gyroid" or not layout.lattice_core:
        return None
    xc, yc, v0, v1 = layout.lattice_core
    return _box(-xc, xc, -yc, yc, v0, v1, layout)


def build_receiver(layout: Layout):
    """The big-end receiver: a deep socket.

    Three families of cut, in the order a machinist would take them:

    * the seat in the top face, a shallow rectangular inset the bottom of the
      stabilising sleeve lands in;
    * the two rail slots through the floor of that seat, which carry on down
      to the face the rail feet sit on;
    * the two swing channels through the outboard walls. These are cut only
      across the rail's own depth, not the full width of the big end, because
      that is the width the clamp tongue has to be to fill the notch -- the
      channel, the notch and the rail are one dimension front to back.
    """
    from .layout import _largest, mirror

    body = profile_solid(layout.envelope, layout, layout.width)

    # the seat
    r_c = layout.slot_corner_radius
    sx = layout.sleeve_x + layout.seat_clearance
    sy = layout.sleeve_y + layout.seat_clearance
    body = body.cut(_slot(-sx, sx, -sy, sy, layout.v_seat, layout.v_rt + 1.0,
                          r_c, layout))

    # the rail slots, with the corner radius the cutter leaves
    c_sl = layout.slot_clearance
    hs = layout.rail_depth / 2.0 + c_sl
    for sign in (1.0, -1.0):
        x0, x1 = sorted((sign * (layout.x_i - c_sl),
                         sign * (layout.x_o + c_sl)))
        body = body.cut(_slot(x0, x1, -hs, hs, layout.v_e, layout.v_rt + 1.0,
                              r_c, layout))

    # the swing channels
    h_ch = layout.rail_depth / 2.0 + layout.channel_clearance
    for prof in (layout.channel_right, mirror(layout.channel_right)):
        cutter = profile_solid(_largest(prof), layout, 2.0 * h_ch, y0=-h_ch)
        body = body.cut(cutter)
    return body


def thread_minor_radius(diameter_mm: float, pitch_mm: float) -> float:
    return (diameter_mm - 1.0825 * pitch_mm) / 2.0


def build_clamp(layout: Layout, side: str, bolt_d: float, pitch: float):
    """One swing clamp. The right clamp carries the thread; the left clamp
    takes the bolt head on its lug's outer face.

    The clamp is two widths, not one. Outside the receiver it is the full
    width of the big end, because there it is the bearing cap. Where it
    passes INTO the receiver it narrows to the rail depth, because there it
    is the tongue running in the channel. The step is made by cutting the
    receiver's own envelope out of the clamp everywhere except the channel's
    band along the crank axis, so the two can never disagree about where the
    step falls.
    """
    polygon = layout.clamp_right if side == "right" else layout.clamp_left
    body = profile_solid(polygon, layout, layout.width)
    h_ch = layout.rail_depth / 2.0 + layout.channel_clearance
    outer = layout.width / 2.0 + 1.0
    if outer > h_ch:
        # The clamp's inner face was made by subtracting this very envelope
        # in 2D, so with guide_clearance at zero the two surfaces are
        # coincident, and a boolean between coincident faces is where
        # OpenCascade quietly returns nothing at all. Grow the cutter by a
        # hair: the only material it takes that the exact one would not is
        # SHOULDER_RELIEF off the step's shoulder, outside the channel band.
        env = layout.envelope.buffer(SHOULDER_RELIEF, quad_segs=64)
        for y0, y1 in ((-outer, -h_ch), (h_ch, outer)):
            body = body.cut(profile_solid(env, layout, y1 - y0, y0=y0))
    if side == "right":
        hole = _cyl_x(thread_minor_radius(bolt_d, pitch), 0.0,
                      layout.x_lug + 2.0, layout.v_bolt, layout)
    else:
        hole = _cyl_x(layout.r_hole, -layout.x_lug - 2.0, 0.0,
                      layout.v_bolt, layout)
    return body.cut(hole)


def build_bolt(layout: Layout, bolt_d: float):
    """The single tangential fastener: head on the left lug, thread in the
    right lug."""
    shank = _cyl_x(bolt_d / 2.0, -layout.x_lug, layout.x_lug - 0.5,
                   layout.v_bolt, layout)
    head_h = 0.7 * bolt_d
    head = _cyl_x(0.75 * bolt_d, -layout.x_lug - head_h, -layout.x_lug,
                  layout.v_bolt, layout)
    return shank.union(head)


PART_NAMES = ("rr_rails", "rr_sleeve", "rr_receiver", "rr_clamp_right",
              "rr_clamp_left", "rr_bolt")

PART_LABELS = {
    "rr_rails": "small end + rails",
    "rr_sleeve": "stabilising sleeve",
    "rr_receiver": "big-end receiver",
    "rr_clamp_right": "swing clamp (right, threaded)",
    "rr_clamp_left": "swing clamp (left, bolt head)",
    "rr_bolt": "tangential bolt",
}


def build_parts(state, layout: Layout | None = None) -> dict:
    """Every part as a CadQuery solid, keyed by part name."""
    from .layout import build_layout

    layout = layout or build_layout(state)
    d_b = state["railrod.bolt_diameter"] * 1e3
    pitch = state["railrod.bolt_pitch"] * 1e3
    c_t = state["railrod.tip_clearance"] * 1e3
    c_s = state["railrod.sleeve_clearance"] * 1e3
    return {
        "rr_rails": build_rails(
            layout, state["railrod.eye_transition_radius"] * 1e3),
        "rr_sleeve": build_sleeve(layout, c_s),
        "rr_receiver": build_receiver(layout),
        "rr_clamp_right": build_clamp(layout, "right", d_b, pitch),
        "rr_clamp_left": build_clamp(layout, "left", d_b, pitch),
        "rr_bolt": build_bolt(layout, d_b),
    }


def part_material(state, part: str) -> str:
    return {
        "rr_rails": state["materials.rod"],
        "rr_sleeve": state["railrod.sleeve_material"],
        "rr_receiver": state["railrod.bigend_material"],
        "rr_clamp_right": state["railrod.bigend_material"],
        "rr_clamp_left": state["railrod.bigend_material"],
        "rr_bolt": state["railrod.bolt_material"],
    }[part]
