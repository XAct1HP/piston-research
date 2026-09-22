"""Why did the FEA fail on THIS machine?

The rod's stiffness matrix came back singular on one Windows install and
solved cleanly everywhere else, which is the hardest kind of bug to chase:
the code is right, the mesh looks right, and the only thing that differs is
the LAPACK/SuperLU build underneath scipy. This script prints everything
that could plausibly differ between two machines, for all four parts, so the
answer arrives in one paste instead of ten rounds of guessing.

    python examples/diagnose_fea.py

Everything here is read-only. It changes no files and no state.
"""

from __future__ import annotations

import os
import platform
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def banner(text: str) -> None:
    print()
    print(text)
    print("-" * len(text))


def main() -> int:
    banner("platform")
    print(f"python   {sys.version.split()[0]} on {platform.platform()}")
    for name in ("numpy", "scipy", "skfem", "tetgen", "cadquery"):
        try:
            module = __import__(name)
            print(f"{name:9s}{getattr(module, '__version__', '?')}")
        except Exception as exc:                              # noqa: BLE001
            print(f"{name:9s}MISSING ({exc})")

    try:
        import scipy.linalg as sla
        print("lapack   ", sla.get_lapack_funcs("gesdd").typecode,
              sla.lapack.__name__)
    except Exception:                                         # noqa: BLE001
        pass
    try:
        import numpy as _np
        cfg = _np.__config__.show(mode="dicts")               # numpy >= 1.25
        build = cfg.get("Build Dependencies", {}).get("blas", {})
        print("blas     ", build.get("name"), build.get("version"))
    except Exception:                                         # noqa: BLE001
        print("blas      (numpy would not say)")

    from psrt.fea import cases
    from psrt.fea.mesh import _face_components, tet_mesh
    from psrt.schema import default_state

    state = default_state()

    for part in ("pin", "piston", "rod", "sleeve"):
        banner(f"{part}")
        try:
            mesh = tet_mesh(state, part, target_elements=25_000)
        except Exception:                                     # noqa: BLE001
            traceback.print_exc()
            continue

        print(f"nodes {mesh.n_nodes}  elements {mesh.n_elements}  "
              f"volume drift {mesh.volume_drift * 100:.3f}%")
        print(f"repairs: welded {mesh.welded_nodes}  "
              f"flat {mesh.flat_elements}  reoriented {mesh.reoriented}  "
              f"orphans {mesh.orphan_nodes}  islands {mesh.islands_dropped}  "
              f"pinched {mesh.pinched_elements}")

        components, label = _face_components(mesh.elements.T)
        sizes = np.bincount(label) if components else np.array([0])
        print(f"face-connected components {components} "
              f"(largest {sizes.max()})")

        corners = mesh.points.T[mesh.elements.T]
        volume = np.abs(np.einsum(
            "ij,ij->i", corners[:, 1] - corners[:, 0],
            np.cross(corners[:, 2] - corners[:, 0],
                     corners[:, 3] - corners[:, 0]))) / 6.0
        print(f"element volume: min {volume.min():.3e}  "
              f"median {np.median(volume):.3e}  "
              f"ratio {np.median(volume) / max(volume.min(), 1e-300):.3e}")

        try:
            result = cases.CASES[part](state, mesh=mesh)
        except Exception:                                     # noqa: BLE001
            traceback.print_exc()
            continue

        solve = result.solve
        print(f"peak {np.max(solve.von_mises) / 1e6:.2f} MPa   "
              f"max displacement {solve.max_displacement * 1e6:.2f} um")
        for note in solve.notes:
            print(f"  - {note}")

    print()
    print("Paste this whole output. The lines that matter are the repair "
          "counters, the component count, and any note naming conjugate "
          "gradients or MINRES.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
