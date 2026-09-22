"""Volume meshing: CadQuery solid to tetrahedra.

The surface triangulation comes straight off the same OpenCascade solid the
viewport draws and the mass properties are measured from, so there is no
second geometry to drift out of step. tetgen fills it with tetrahedra.

Mesh density is set by a TARGET ELEMENT COUNT, not by the surface tolerance.
That was the first thing tried and it does not work: a gudgeon pin is a plain
tube, so CadQuery resolves it in about a thousand triangles at any tolerance
you ask for, and the knob does nothing.

A volume ceiling per element is the honest knob. It is predictable, it gives
the convergence check two refinement levels that really differ, and it keeps
the direct solve inside the memory of an ordinary machine.

One hard-won negative result, recorded because it cost a phase. This module
used to subdivide the surface before meshing, on the theory that CadQuery
hands over slivers -- the pin arrives as triangles 1.4 mm around the
circumference and 63.5 mm along the axis -- and that tetgen could not build
good tetrahedra against a boundary like that. The pin then meshed to 390,000
elements against a 25,000 target, and the conclusion drawn was that tetgen's
quality criterion could not be reined in.

That conclusion was wrong. THE SUBDIVISION WAS CAUSING IT. Splitting an
over-long edge at its midpoint never touches the short edge opposite, so the
median aspect ratio CLIMBS under refinement -- 16 to 28 on the pin -- and
tetgen, which preserves the boundary it is handed, had no choice but to flood
the volume with Steiner points to conform to it. Removing the subdivision
entirely fixes every part at once: all four now land within 6% of the
requested element count in under half a second, from the raw CadQuery surface.

One measurement recorded and NOT acted on, so it does not get re-measured.
The rod meshes with its smallest element about a millionth of the median
volume, which is what made its stiffness matrix conditioning-sensitive (see
``solve._solve_condensed``). Sweeping tetgen's quality knobs, ``min_ratio=3``
with ``mindihedral=10`` cuts that spread six-fold AND lands closer to the
requested element count than the ``min_ratio=2`` in use. Tightening the ratio
instead makes it far worse -- 1.2 gives a spread of 3.7e10 -- because a
radius-edge ratio cannot see a sliver at all, which is a known limitation of
the criterion rather than of this mesher. The looser setting was left alone
anyway: the solver now removes the conditioning problem exactly, the
validation tolerances against Lame and beam theory are pinned to the current
element quality, and trading proven stress accuracy for a problem already
solved is not a trade worth making. Revisit only with a validation run.

A full isotropic remesher (split, collapse, flip, relax) was written before
this was understood. It worked on the pin and the sleeve and introduced
self-intersections on the rod and the piston, and once the real cause was
found it had nothing left to do. It is not in the tree. Do not add surface
refinement back here without a measurement showing it helps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry import build as build_mod


@dataclass
class MeshResult:
    """A tetrahedral mesh, in metres."""

    points: np.ndarray            # (3, n_nodes)
    elements: np.ndarray          # (4, n_elements)
    tolerance: float
    surface_triangles: int
    volume_drift: float = 0.0     # fraction lost to chordal remeshing
    orphan_nodes: int = 0         # points no element referenced
    islands_dropped: int = 0      # disconnected pieces discarded
    reoriented: int = 0           # inverted elements turned the right way
    welded_nodes: int = 0         # coincident points merged
    flat_elements: int = 0        # zero-volume elements removed
    pinched_elements: int = 0     # chunks attached by only a vertex or edge

    @property
    def n_nodes(self) -> int:
        return self.points.shape[1]

    @property
    def n_elements(self) -> int:
        return self.elements.shape[1]

    def volume(self) -> float:
        """Summed tet volumes. Compared against the solid's true volume as a
        check that the mesh actually filled the part."""
        p = self.points
        a, b, c, d = (p[:, self.elements[i]] for i in range(4))
        return float(np.abs(np.einsum(
            "ij,ij->j", b - a, np.cross((c - a).T, (d - a).T).T)).sum() / 6.0)

    def as_dict(self) -> dict:
        return {"nodes": self.n_nodes, "elements": self.n_elements,
                "tolerance_mm": self.tolerance,
                "surface_triangles": self.surface_triangles,
                "volume_m3": self.volume()}


def surface_of(state, part: str, tolerance: float = 0.6,
               angular: float = 0.35):
    """Triangulate a part. Returns (vertices, triangles) in millimetres."""
    builder, _ = build_mod.BUILDS[part]
    vertices, triangles = build_mod.tessellate_solid(
        builder(state).val(), tolerance, angular)
    return (np.array([[v.x, v.y, v.z] for v in vertices], dtype=float),
            np.array(triangles, dtype=np.int32))


ELEMENT_CEILING = 250_000


def _surface_volume(v: np.ndarray, f: np.ndarray) -> float:
    """Volume enclosed by a closed triangulation, by the divergence theorem."""
    p = v[f]
    return float(np.einsum("ij,ij->i",
                           p[:, 0], np.cross(p[:, 1], p[:, 2])).sum() / 6.0)


def weld(vertices: np.ndarray, triangles: np.ndarray,
         tolerance: float = 1e-6) -> tuple:
    """Merge coincident vertices so the surface is actually closed.

    This is the one that mattered. OpenCascade triangulates each FACE
    independently and does not merge the vertices along the seams between
    them, so what comes out of ``tessellate`` is not a watertight shell -- it
    is a pile of patches whose edges coincide in space but not in the index
    table. The gudgeon pin arrives with 148 duplicated vertex positions and
    292 edges belonging to a single triangle.

    tetgen cannot tell a shell like that from a surface with holes in it. It
    responds by flooding the volume with Steiner points, or by segfaulting.
    Both happened here before this function existed.

    Welding on a rounded coordinate key makes the shell closed, after which a
    plain tube meshes into tens of thousands of tetrahedra instead of
    hundreds of thousands.
    """
    vertices = np.asarray(vertices, dtype=float)
    decimals = max(int(round(-np.log10(tolerance))), 0)
    keys = np.round(vertices, decimals)

    _, first, inverse = np.unique(keys, axis=0, return_index=True,
                                  return_inverse=True)
    welded = vertices[np.sort(first)]
    # np.unique sorts its output, so rebuild the mapping in that order.
    order = np.argsort(np.sort(first))
    lookup = np.empty(len(first), dtype=np.int64)
    lookup[np.argsort(np.sort(first))] = np.arange(len(first))
    remap = np.searchsorted(np.sort(first), first)[inverse]

    faces = remap[np.asarray(triangles, dtype=np.int64)]
    keep = ((faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2])
            & (faces[:, 2] != faces[:, 0]))
    return welded, faces[keep]


def audit_surface(vertices: np.ndarray, triangles: np.ndarray) -> dict:
    """Is this shell closed? Every edge of a watertight surface is shared by
    exactly two triangles."""
    from collections import Counter

    edges: Counter = Counter()
    for tri in triangles:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edges[(a, b) if a < b else (b, a)] += 1
    sharing = Counter(edges.values())
    return {
        "triangles": int(len(triangles)),
        "vertices": int(len(vertices)),
        "boundary_edges": int(sharing.get(1, 0)),
        "non_manifold_edges": int(sum(n for k, n in sharing.items() if k > 2)),
        "watertight": sharing.get(1, 0) == 0 and not any(
            k > 2 for k in sharing),
    }


VOLUME_DRIFT_LIMIT = 0.02          # 2%, generous: the parts are mostly flat


def tet_mesh(state, part: str, target_elements: int = 25_000,
             tolerance: float = 0.4, min_ratio: float = 2.0) -> MeshResult:
    """Mesh one part to roughly ``target_elements`` tetrahedra.

    ``tolerance`` controls the surface triangulation, which sets how
    faithfully curvature is captured; ``target_elements`` sets the density,
    through a per-element volume ceiling. The surface itself is passed to
    tetgen untouched -- see the module docstring for why.
    """
    import tetgen

    from ..geometry import build_all

    volume_mm3 = build_all(state)[part]["properties"].volume * 1e9
    # Edge length for roughly the element count asked for. A tetrahedron of
    # edge h occupies about h^3/8, so this is the cube root of the share each
    # element gets, with a little slack.
    target_edge = (volume_mm3 / max(target_elements, 100) * 6.0) ** (1.0 / 3.0)

    vertices, triangles = surface_of(state, part,
                                     min(tolerance, target_edge / 3.0))
    vertices, triangles = weld(vertices, triangles)

    report = audit_surface(vertices, triangles)
    if not report["watertight"]:
        raise ValueError(
            f"the {part} surface is not watertight after welding: "
            f"{report['boundary_edges']} open edges, "
            f"{report['non_manifold_edges']} non-manifold. tetgen cannot mesh "
            "it and may crash the process rather than refuse.")

    # The surface goes to tetgen exactly as CadQuery drew it. See the module
    # docstring before adding any refinement here.
    meshed_mm3 = abs(_surface_volume(vertices, triangles))
    drift = abs(meshed_mm3 - volume_mm3) / max(volume_mm3, 1e-30)
    if drift > VOLUME_DRIFT_LIMIT:
        raise ValueError(
            f"the {part} triangulation encloses {meshed_mm3:.1f} mm^3 against "
            f"the solid's {volume_mm3:.1f} mm^3, a {drift * 100:.1f}% gap past "
            f"the {VOLUME_DRIFT_LIMIT * 100:.0f}% limit. The surface tolerance "
            "is too coarse to represent this part.")

    max_volume = max(volume_mm3 / max(target_elements, 100) * 2.0, 1e-9)
    engine = tetgen.TetGen(np.ascontiguousarray(vertices),
                           np.ascontiguousarray(triangles))
    result = engine.tetrahedralize(
        order=1, mindihedral=10.0, minratio=min_ratio,
        maxvolume=max_volume, fixedvolume=True, steinerleft=ELEMENT_CEILING)
    nodes, elements = result[0], result[1]

    if len(elements) > ELEMENT_CEILING:
        raise ValueError(
            f"the {part} meshed to {len(elements):,} elements, past the "
            f"{ELEMENT_CEILING:,} ceiling. Lower target_elements: a direct "
            "solve at that size will exhaust memory rather than fail cleanly.")

    if len(elements) == 0:
        raise ValueError(
            f"tetgen produced no elements for the {part}. The surface "
            "triangulation is probably not watertight at this tolerance; try "
            "a smaller one.")

    nodes, elements, repair = _make_solvable(nodes, elements, part)

    # Model units are millimetres; everything downstream is SI.
    return MeshResult(points=np.ascontiguousarray(nodes.T) * 1e-3,
                      elements=np.ascontiguousarray(elements.T),
                      tolerance=tolerance,
                      surface_triangles=len(triangles),
                      volume_drift=drift,
                      orphan_nodes=repair["orphan_nodes"],
                      islands_dropped=repair["islands_dropped"],
                      reoriented=repair["reoriented"],
                      welded_nodes=repair["welded_nodes"],
                      flat_elements=repair["flat_elements"],
                      pinched_elements=repair["pinched_elements"])


ISLAND_TOLERANCE = 0.01        # discard debris, refuse to hide a real piece


def _make_solvable(nodes: np.ndarray, elements: np.ndarray, part: str):
    """Repair every mesh fault that makes a stiffness matrix singular.

    All of these are invisible until the solve fails, and the failure blames
    the wrong thing: the symptom is non-finite displacements, which reads as
    an under-restrained part. A mesh can be restrained at 666 nodes and still
    produce a singular matrix.

    WELD coincident points first. Two nodes at the same coordinates leave a
    crack through the part: the elements either side share no node, so load
    crosses nowhere, and either piece can move independently.

    DROP FLAT elements. A tetrahedron with four coplanar corners has no
    volume, so it contributes nothing and its own matrix is singular.

    REORIENT inverted elements. A negatively-oriented tetrahedron contributes
    NEGATIVE stiffness, which destroys the positive-definiteness the solver
    relies on. Swapping two of its nodes turns it the right way without
    moving anything. Different tetgen builds order nodes differently, so this
    can be clean on one platform and not on another.

    REMOVE ORPHAN nodes -- points no element references. Their rows and
    columns are exactly zero, so the matrix is singular however well the part
    is held.

    DISCARD DISCONNECTED islands, which float free with their own six
    rigid-body modes that restraining the part does not touch. Debris under
    ``ISLAND_TOLERANCE`` goes; anything larger is refused, because at that
    size it is more likely a real piece of the geometry than an artefact, and
    deleting part of the component silently would be worse than failing.

    DISCARD PINCHED chunks last, and see ``_face_components`` for why the
    island test above cannot find them: two elements sharing one node count
    as connected there, so a chunk hinged on a single vertex or edge passes
    that test and still leaves the matrix singular.
    """
    import scipy.sparse as sp

    report = {"orphan_nodes": 0, "islands_dropped": 0, "reoriented": 0,
              "welded_nodes": 0, "flat_elements": 0, "pinched_elements": 0}

    # --- weld coincident points ------------------------------------------
    scale = float(np.abs(nodes).max()) or 1.0
    keys = np.round(nodes / (scale * 1e-10)).astype(np.int64)
    _, first, inverse = np.unique(keys, axis=0, return_index=True,
                                  return_inverse=True)
    if first.size != len(nodes):
        report["welded_nodes"] = int(len(nodes) - first.size)
        nodes = nodes[np.sort(first)]
        # re-index through the sorted survivors
        order = np.argsort(first)
        rank = np.empty_like(order)
        rank[order] = np.arange(order.size)
        elements = rank[inverse][elements]

    def signed_volumes(pts, els):
        corners = pts[els]
        return np.einsum(
            "ij,ij->i", corners[:, 1] - corners[:, 0],
            np.cross(corners[:, 2] - corners[:, 0],
                     corners[:, 3] - corners[:, 0])) / 6.0

    # --- drop flat elements, then turn inverted ones the right way -------
    volume = signed_volumes(nodes, elements)
    reference = float(np.median(np.abs(volume))) or 1.0
    solid = np.abs(volume) > reference * 1e-12
    if not solid.all():
        report["flat_elements"] = int((~solid).sum())
        elements = elements[solid]
        volume = volume[solid]

    inverted = volume < 0.0
    if inverted.any():
        report["reoriented"] = int(inverted.sum())
        elements[inverted] = elements[inverted][:, [0, 2, 1, 3]]

    # --- orphan nodes -----------------------------------------------------
    used, inverse = np.unique(elements, return_inverse=True)
    if used.size != len(nodes):
        report["orphan_nodes"] += int(len(nodes) - used.size)
        nodes = nodes[used]
        elements = inverse.reshape(elements.shape)

    # --- disconnected pieces ---------------------------------------------
    rows = np.repeat(np.arange(len(elements)), elements.shape[1])
    cols = elements.reshape(-1)
    incidence = sp.csr_matrix(
        (np.ones(rows.size, dtype=np.int8), (rows, cols)),
        shape=(len(elements), len(nodes)))
    count, label = sp.csgraph.connected_components(
        incidence @ incidence.T, directed=False)

    if count > 1:
        sizes = np.bincount(label)
        keep = int(np.argmax(sizes))
        stray = int(len(elements) - sizes[keep])
        share = stray / max(len(elements), 1)
        if share > ISLAND_TOLERANCE:
            raise ValueError(
                f"the {part} meshed into {count} disconnected pieces, and "
                f"{share * 100:.1f}% of the elements are not attached to the "
                "largest one. That is too much to be meshing debris, so it "
                "is more likely a real piece of the geometry -- discarding "
                "it silently would be worse than refusing. Re-mesh at a "
                "different element count.")
        report["islands_dropped"] = count - 1
        elements = elements[label == keep]
        used, inverse = np.unique(elements, return_inverse=True)
        report["orphan_nodes"] += int(len(nodes) - used.size)
        nodes = nodes[used]
        elements = inverse.reshape(elements.shape)

    # --- pinch points ----------------------------------------------------
    # The island test above joins any two elements sharing a single NODE, so
    # it calls a mesh connected when it is really two chunks meeting at a
    # point. That is the fault that survives every other repair here, and it
    # is the reason a part can be restrained at six hundred nodes and still
    # return NaN: a chunk hanging off one vertex -- or off one edge -- can
    # rotate about it at zero strain and therefore zero energy, so the
    # stiffness matrix has a null space no restraint on the part can reach,
    # while the island counter reports nothing wrong.
    #
    # A real solid mesh is FACE-connected: you can walk from any tetrahedron
    # to any other through shared triangles. Build the graph on shared faces
    # instead of shared nodes and anything outside the main component is
    # attached by an edge at most. Dropping a pinched chunk can pinch off
    # what it was holding, so this repeats until it is clean.
    for _ in range(8):
        count, label = _face_components(elements)
        if count <= 1:
            break
        sizes = np.bincount(label)
        keep = int(np.argmax(sizes))
        stray = int(len(elements) - sizes[keep])
        share = stray / max(len(elements), 1)
        if share > ISLAND_TOLERANCE:
            raise ValueError(
                f"the {part} meshed into {count} chunks joined only at "
                f"vertices or edges, and {share * 100:.1f}% of the elements "
                "are in the smaller ones. Each of those can rotate about its "
                "pinch point at zero energy, so the stiffness matrix is "
                "singular whatever the restraints do -- but that is too much "
                "of the part to discard silently. Re-mesh at a different "
                "element count.")
        report["pinched_elements"] += stray
        elements = elements[label == keep]
        used, inverse = np.unique(elements, return_inverse=True)
        report["orphan_nodes"] += int(len(nodes) - used.size)
        nodes = nodes[used]
        elements = inverse.reshape(elements.shape)

    return nodes, elements, report


def _face_components(elements: np.ndarray) -> tuple:
    """Connected components of the element graph, joined by shared FACES.

    Two tetrahedra are adjacent here only if they share three nodes. Sorting
    every face's node triple and looking for duplicates finds those pairs in
    one pass, which is why this is cheap enough to run on every mesh.
    """
    import scipy.sparse as sp

    n = len(elements)
    if n == 0:
        return 0, np.zeros(0, dtype=int)

    faces = np.sort(np.stack([elements[:, [0, 1, 2]], elements[:, [0, 1, 3]],
                              elements[:, [0, 2, 3]], elements[:, [1, 2, 3]]],
                             axis=1), axis=2).reshape(-1, 3)
    owner = np.repeat(np.arange(n), 4)
    order = np.lexsort((faces[:, 2], faces[:, 1], faces[:, 0]))
    faces, owner = faces[order], owner[order]
    twin = np.flatnonzero(np.all(faces[1:] == faces[:-1], axis=1))
    graph = sp.coo_matrix(
        (np.ones(twin.size, dtype=np.int8), (owner[twin], owner[twin + 1])),
        shape=(n, n))
    return sp.csgraph.connected_components(graph, directed=False)
