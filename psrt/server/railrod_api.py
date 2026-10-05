"""HTTP endpoints for the rail connecting rod.

Kept apart from ``app.py`` because the concept is optional: with
``railrod.enabled`` off every one of these answers ``enabled: false`` and the
rest of the tool behaves exactly as it did.

The FEA endpoint is the expensive one. The first call on a design runs the
three part solves the coupled contact network needs (rails, receiver, right
clamp) -- tens of seconds -- and every later call on the same geometry reuses
them. The stress field it returns is not a single picture: the per-vertex
unit-case tensors stay on the server, and ``/at`` recombines them for any
crank angle in a few milliseconds, which is what lets the viewport's crank
slider drive the colours.
"""

from __future__ import annotations

import math
import traceback

import numpy as np


def _finite(v):
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def register(app, session, clean):
    from ..railrod import analysis as rr_analysis
    from ..railrod.cad import PART_LABELS, PART_NAMES

    store: dict = {}          # part -> prepared field, for the current design

    def enabled():
        return rr_analysis.enabled(session.state)

    def margins_for(metrics):
        comps = {"rails", "rail notch", "rail joint", "rail-rod bolt",
                 "swing clamps", "stabilising sleeve", "big end", "small end"}
        return [m.to_dict() for m in metrics.report.sorted()
                if m.component in comps]

    @app.get("/api/railrod")
    def railrod_summary():
        if not enabled():
            return {"ok": True, "enabled": False}
        try:
            metrics = session.metrics()
            a = rr_analysis.analyse(session.state, metrics.sweep)
        except Exception as exc:                          # noqa: BLE001
            return {"ok": False, "enabled": True,
                    "error": f"{type(exc).__name__}: {exc}"}
        from ..railrod import coupled
        coupled_summary = None
        for key, value in coupled._CACHE.items():
            if key[0] == session.state.fingerprint():
                coupled_summary = value.summary()
        conventional = session.state["masses.rod_total"]
        return clean({
            "ok": True, "enabled": True,
            "summary": a.summary(),
            "coupled": coupled_summary,
            "margins": margins_for(metrics),
            "parts": [{"name": n, "label": PART_LABELS[n],
                       "mass_kg": a.masses[n]} for n in PART_NAMES],
            "rod_total_specified_kg": conventional,
            "shell_mass_kg": session.state["railrod.shell_mass"],
        })

    @app.get("/api/railrod/section")
    def railrod_section(swing: float = 0.0, theta_deg: float | None = None):
        if not enabled():
            return {"ok": True, "enabled": False}
        from ..railrod.layout import section_payload
        try:
            a = rr_analysis.analyse(session.state)
        except Exception as exc:                          # noqa: BLE001
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        out = section_payload(a.layout, swing)
        # contact forces at the requested crank angle, for the arrows
        use = a
        from ..railrod import coupled
        for key, value in coupled._CACHE.items():
            if key[0] == session.state.fingerprint():
                use = value
        cyc = use.cycle
        if theta_deg is None:
            forces = cyc.preload_state["forces"]
            label = "assembly (bolt preload only)"
            bolt = cyc.preload_state["bolt_force"]
        else:
            i = int(np.argmin(np.abs(np.degrees(cyc.theta) - theta_deg)))
            forces = cyc.forces[i]
            label = f"{np.degrees(cyc.theta[i]):.0f} deg"
            bolt = float(cyc.bolt[i])
        out["contacts"] = [
            {"group": c.group, "point": c.point.tolist(),
             "normal": c.normal.tolist(), "force_n": float(f),
             "on": c.b}
            for c, f in zip(cyc.network.contacts, forces) if f > 1.0]
        out["bolt_n"] = float(bolt)
        out["forces_at"] = label
        out["network"] = ("coupled" if getattr(use, "coupled", False)
                          else "rigid")
        out["ok"] = True
        return clean(out)

    @app.get("/api/railrod/coupled")
    def railrod_coupled(elements: int = 20_000):
        """Run (or fetch) the flexibility-coupled network for this design."""
        if not enabled():
            return {"ok": False, "error": "the rail rod is not enabled"}
        try:
            from ..railrod.coupled import coupled_analysis
            a = coupled_analysis(session.state, target_elements=elements)
        except Exception as exc:                          # noqa: BLE001
            traceback.print_exc()
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        _forget_margins()
        return clean({"ok": True, "summary": a.summary()})

    @app.get("/api/railrod/cycle")
    def railrod_cycle():
        if not enabled():
            return {"ok": True, "enabled": False}
        a = rr_analysis.analyse(session.state)
        from ..railrod import coupled
        out = {"ok": True, "rigid": a.cycle.as_dict(stride=2)}
        for key, value in coupled._CACHE.items():
            if key[0] == session.state.fingerprint():
                out["coupled"] = value.cycle.as_dict(stride=2)
        return clean(out)

    @app.get("/api/railrod/fea/{part}")
    def railrod_fea(part: str, elements: int = 20_000, coupled: bool = True):
        if not enabled():
            return {"ok": False, "error": "the rail rod is not enabled"}
        if part not in PART_NAMES:
            return {"ok": False, "error": f"no rail-rod part {part!r}"}
        try:
            from ..railrod import fea
            result = fea.solve_part(session.state, part,
                                    target_elements=elements, coupled=coupled)
            if coupled:
                _forget_margins()
            summary = fea.summarise(result, session.state)
            field = _prepare(result)
        except MemoryError:
            return {"ok": False, "error": "ran out of memory; try fewer "
                                          "elements"}
        except ImportError as exc:
            return {"ok": False, "error": f"the FEA libraries are not "
                    f"installed ({exc}). python -m pip install scikit-fem "
                    "tetgen scipy shapely"}
        except Exception as exc:                          # noqa: BLE001
            traceback.print_exc()
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        store.clear()
        store[part] = {"result": result, "field": field,
                       "fingerprint": session.state.fingerprint()}
        return clean({
            "ok": True, "part": part, "label": PART_LABELS[part],
            "summary": summary,
            "field": {
                "vertices": field["vertices"],
                "triangles": field["triangles"],
                "peak_mpa": field["peak_mpa"],
                "fatigue_sf": field["fatigue_sf"],
                "range_mpa": field["range"],
            },
            "theta_deg": np.degrees(result.theta).tolist(),
            "allowable_mpa": summary["allowable_pa"] / 1e6,
            "fingerprint": session.state.fingerprint(),
        })

    @app.get("/api/railrod/fea/{part}/at")
    def railrod_fea_at(part: str, theta_deg: float):
        entry = store.get(part)
        if entry is None or entry["fingerprint"] != session.state.fingerprint():
            return {"ok": False, "error": "run the part first"}
        result, field = entry["result"], entry["field"]
        i = int(np.argmin(np.abs(np.degrees(result.theta) - theta_deg)))
        s = np.tensordot(result.coeffs[i], field["unit"], axes=1)
        vm = _vm(s) / 1e6
        loads = [{"case": c.name, "label": c.label,
                  "value": float(result.coeffs[i, k])}
                 for k, c in enumerate(result.cases)
                 if abs(result.coeffs[i, k]) > 1e-9]
        return clean({"ok": True, "index": i,
                      "theta_deg": float(np.degrees(result.theta[i])),
                      "stress_mpa": np.round(vm, 2).tolist(),
                      "peak_mpa": float(vm.max()),
                      "loads": sorted(loads, key=lambda d: -abs(d["value"]))[:12]})

    def _forget_margins():
        """The margins were evaluated on the rigid network; now that the
        coupled one exists for this design, make them re-read it."""
        from ..evaluate import _CACHE
        _CACHE.pop(session.state.fingerprint(), None)

    def _prepare(result):
        from ..fea.field import boundary_surface
        from ..railrod.fea import fatigue_field
        from ..evaluate import evaluate

        mesh = result.mesh
        tris = boundary_surface(mesh.elements)
        used = np.unique(tris)
        remap = np.full(mesh.points.shape[1], -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        tets = mesh.elements.T
        c = mesh.points.T[tets]
        vol = np.abs(np.einsum("ij,ij->i", c[:, 1] - c[:, 0],
                               np.cross(c[:, 2] - c[:, 0],
                                        c[:, 3] - c[:, 0]))) / 6.0
        wsum = np.zeros(mesh.n_nodes)
        for k in range(4):
            np.add.at(wsum, tets[:, k], vol)
        wsum = np.maximum(wsum, 1e-300)

        def nodal(values):
            out = np.zeros((mesh.n_nodes,) + values.shape[1:])
            for k in range(4):
                np.add.at(out, tets[:, k], values * vol.reshape(
                    (-1,) + (1,) * (values.ndim - 1)))
            return (out / wsum.reshape((-1,) + (1,) * (values.ndim - 1)))[used]

        unit = np.stack([nodal(result.tensors[k].astype(float))
                         for k in range(len(result.cases))])
        stats = result.cycle_stats()
        metrics = evaluate(session.state)
        sf, _ = fatigue_field(result, stats, session.state,
                              metrics.thermal.rod)
        sf = np.minimum(np.where(np.isfinite(sf), sf, 20.0), 20.0)
        mask = ~result.excluded
        peak_e = np.where(mask, stats["peak"], 0.0)
        peak = nodal(peak_e) / 1e6
        return {
            "vertices": np.round(mesh.points[:, used].T * 1e3, 4).tolist(),
            "triangles": remap[tris].tolist(),
            "unit": unit,
            "peak_mpa": np.round(peak, 2).tolist(),
            "fatigue_sf": np.round(nodal(np.where(mask, sf, 20.0)), 3)
            .tolist(),
            "range": {"p99_mpa": float(np.percentile(peak, 99.0)),
                      "max_mpa": float(peak.max())},
        }

    def _vm(s):
        xx, yy, zz, xy, yz, zx = (s[..., i] for i in range(6))
        return np.sqrt(0.5 * ((xx - yy) ** 2 + (yy - zz) ** 2
                              + (zz - xx) ** 2)
                       + 3.0 * (xy ** 2 + yz ** 2 + zx ** 2))
