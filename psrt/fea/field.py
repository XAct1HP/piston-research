"""Turning a solved volume into something the viewport can draw.

The solver produces one stress per tetrahedron. The viewport draws triangles.
This module bridges the two: it pulls the boundary surface off the tet mesh
and carries the element stresses out to its vertices, so three.js can shade a
continuous field instead of a mosaic of flat facets.

Two decisions worth stating, because both affect what the picture claims.

Element stresses are averaged to each vertex by VOLUME, not by count. A
sliver next to a well-shaped element should not pull the value at a shared
node as hard, and with constant-strain tetrahedra the stress is a cell
average anyway, so weighting by the volume each cell represents is the
consistent choice.

The colour scale is clipped at a high percentile rather than at the outright
maximum. A single element at a re-entrant corner or under a point restraint
can report an arbitrarily large stress that refines toward infinity and means
nothing physical; scaled against that, every real feature washes out to the
same colour. The clip is reported alongside the field so the legend can say
so rather than quietly lying about the range.
"""

from __future__ import annotations

import numpy as np

__all__ = ["boundary_surface", "nodal_field", "field_payload"]

# The four triangular faces of a tetrahedron, each wound outward when the
# element itself is positively oriented.
_TET_FACES = ((0, 2, 1), (0, 1, 3), (1, 2, 3), (0, 3, 2))


def boundary_surface(elements: np.ndarray) -> np.ndarray:
    """The outward-facing triangles of a tetrahedral mesh.

    A face shared by two tetrahedra is interior; a face appearing once is on
    the boundary. Sorting each face's node ids identifies shared faces without
    caring about winding, while the unsorted copy keeps the winding needed to
    draw it.
    """
    tets = np.asarray(elements).T
    faces = np.concatenate([tets[:, list(f)] for f in _TET_FACES], axis=0)
    keys = np.sort(faces, axis=1)
    _, first, counts = np.unique(keys, axis=0, return_index=True,
                                 return_counts=True)
    return faces[first[counts == 1]]


def nodal_field(points: np.ndarray, elements: np.ndarray,
                values: np.ndarray) -> np.ndarray:
    """Carry per-element values out to the nodes, weighted by element volume."""
    tets = np.asarray(elements).T
    corners = np.asarray(points).T[tets]
    volume = np.abs(np.einsum(
        "ij,ij->i",
        corners[:, 1] - corners[:, 0],
        np.cross(corners[:, 2] - corners[:, 0],
                 corners[:, 3] - corners[:, 0]))) / 6.0

    n_nodes = np.asarray(points).shape[1]
    weighted = np.zeros(n_nodes)
    total = np.zeros(n_nodes)
    for corner in range(4):
        np.add.at(weighted, tets[:, corner], values * volume)
        np.add.at(total, tets[:, corner], volume)
    return weighted / np.maximum(total, 1e-300)


def field_payload(result, clip_percentile: float = 99.0) -> dict:
    """A solved case as vertices, triangles and per-vertex stress, in MPa.

    Coordinates are converted to millimetres to match everything else the
    viewport draws. Only boundary vertices are sent -- the interior of the
    mesh is invisible and would roughly triple the payload.
    """
    mesh = result.mesh
    triangles = boundary_surface(mesh.elements)
    nodal = nodal_field(mesh.points, mesh.elements, result.von_mises)

    used = np.unique(triangles)
    remap = np.full(mesh.points.shape[1], -1, dtype=np.int64)
    remap[used] = np.arange(len(used))

    vertices = mesh.points[:, used].T * 1e3
    stress = nodal[used] / 1e6
    clip = float(np.percentile(stress, clip_percentile))

    displacement = np.linalg.norm(result.displacement[:, used], axis=0)

    return {
        "vertices": vertices.tolist(),
        "triangles": remap[triangles].tolist(),
        "stress_mpa": stress.tolist(),
        "displacement_mm": (displacement * 1e3).tolist(),
        "range_mpa": {
            "min": float(stress.min()),
            "max": float(stress.max()),
            "clip": clip,
            "clip_percentile": clip_percentile,
        },
        "element_count": int(mesh.n_elements),
        "node_count": int(mesh.n_nodes),
    }
