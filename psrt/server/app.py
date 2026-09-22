"""FastAPI backend.

Every endpoint is a thin wrapper over the same pure functions the CLI uses.
There is no separate "web" model of the design -- the browser reads and writes
exactly what `evaluate()` reads and writes, which is the whole reason the
purity rule from phase 1 was worth keeping.

Two things are deliberately shaped for phase 5. Parameter writes return a
structured constraint violation rather than a 500, so a caller that is refused
learns *why* and can reason around it. And propose / commit / revert already
exist, so the AI tool layer is wiring rather than new machinery.
"""

from __future__ import annotations

import json
import math
import os
import sys
import traceback

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..ai.agent import ENV_FILE, Agent, no_key_message
from ..ai.tools import TOOLS
from .. import kinematics as kin
from .. import geometry as geo
from ..evaluate import torque_curve
from ..schema import default_state
from ..state import Mutability
from ..units import rad_s_to_rpm, rpm_to_rad_s
from .session import Session

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


class ChangeRequest(BaseModel):
    changes: dict
    rationale: str = ""
    actor: str = "user"


class ChatRequest(BaseModel):
    message: str


def _clean(value):
    """JSON cannot hold NaN or infinity; the browser should get null."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return _clean(float(value))
    return value


def create_app(state_path: str | None = None) -> FastAPI:
    app = FastAPI(title="Piston System Research Tool", version="0.4.0")
    session = (Session.from_file(state_path) if state_path
               else Session(state=default_state("untitled")))
    app.state.session = session

    # -- the design state ---------------------------------------------------

    @app.get("/api/state")
    def get_state():
        """Every parameter with its value, bounds, lock state and the reason
        behind each bound. The left rail is a direct rendering of this."""
        out = []
        for section, params in session.state.sections.items():
            entries = []
            for name, p in params.items():
                path = f"{section}.{name}"
                try:
                    value = session.state.get(path)
                except Exception:
                    value = None
                entries.append({
                    "path": path, "name": name, "value": _clean(value),
                    "unit": p.unit, "class": p.mutability.value,
                    "min": _clean(p.minimum), "max": _clean(p.maximum),
                    "why": p.why, "why_min": p.why_min, "why_max": p.why_max,
                    "description": p.description, "source": p.source,
                    "editable": p.mutability in (Mutability.FREE,
                                                 Mutability.BOUNDED),
                })
            out.append({"section": section, "parameters": entries})
        return {
            "name": session.state.meta.get("name", "untitled"),
            "notes": session.state.meta.get("notes", ""),
            "provenance": session.state.meta.get("provenance", ""),
            "fingerprint": session.state.fingerprint(),
            "sections": out,
            "issues": session.state.validate(),
            "can_undo": bool(session.undo_stack),
        }

    @app.post("/api/state")
    def set_state(request: ChangeRequest):
        result = session.set(request.changes, actor=request.actor,
                             rationale=request.rationale)
        return _clean(result)

    @app.post("/api/propose")
    def propose(request: ChangeRequest):
        return _clean(session.propose(request.changes, request.rationale,
                                      request.actor))

    @app.post("/api/commit/{proposal_id}")
    def commit(proposal_id: str):
        return _clean(session.commit(proposal_id))

    @app.post("/api/discard/{proposal_id}")
    def discard(proposal_id: str):
        return session.discard(proposal_id)

    @app.post("/api/revert")
    def revert():
        return session.revert()

    @app.post("/api/save")
    def save():
        return session.save()

    # -- results ------------------------------------------------------------

    @app.get("/api/metrics")
    def metrics():
        return _clean(session.metrics().to_dict())

    @app.get("/api/margins")
    def margins():
        return _clean(session.metrics().report.to_dict())

    @app.get("/api/sweep")
    def sweep(points: int = 361):
        """Crank-angle arrays, decimated, plus the assembly kinematics the
        viewport needs to place each part at a given angle.

        Positions are in millimetres to match the tessellation, and the frame
        is: crank axis at the origin, cylinder axis along +Z.
        """
        m = session.metrics()
        s = m.sweep
        state = session.state
        geom = kin.CrankGeometry.from_state(state)

        step = max(1, len(s.theta) // points)
        idx = np.arange(0, len(s.theta), step)
        theta = s.theta[idx]

        mm = 1000.0
        r = geom.crank_radius * mm
        comp_height = state["piston.compression_height"] * mm

        pin_z = kin.pin_distance(geom, theta) * mm
        crank_x = r * np.sin(theta)
        crank_z = r * np.cos(theta)

        # The rod points down from the pin to the crankpin, tilted by the
        # obliquity. Writing it as pi - phi keeps it continuous through TDC;
        # atan2 on the rod vector is the same angle but wraps at +/-180, which
        # would glitch any interpolation the viewport does between samples.
        rod_ry = np.pi - kin.rod_angle(geom, theta)

        return _clean({
            "theta_deg": np.degrees(theta).tolist(),
            "pressure_bar": (s.pressure[idx] / 1e5).tolist(),
            "f_gas_kn": (s.f_gas[idx] / 1e3).tolist(),
            "f_pin_kn": (s.f_pin[idx] / 1e3).tolist(),
            "f_side_kn": (s.f_side[idx] / 1e3).tolist(),
            "torque_nm": s.torque[idx].tolist(),
            "piston_translate_mm": (pin_z + comp_height).tolist(),
            "pin_z_mm": pin_z.tolist(),
            "rod_rotation_y": rod_ry.tolist(),
            "crankpin_x_mm": crank_x.tolist(),
            "crankpin_z_mm": crank_z.tolist(),
            "sleeve_translate_mm": float(
                (geom.crank_radius + geom.rod_length) * mm + comp_height),
            "crank_radius_mm": float(r),
            "rpm": rad_s_to_rpm(s.speed),
            "peak_f_pin_kn": float(np.max(np.abs(s.f_pin)) / 1e3),
            "peak_f_side_kn": float(np.max(np.abs(s.f_side)) / 1e3),
        })

    @app.get("/api/curve")
    def curve(start: int = 1000, stop: int = 7000, step: int = 250):
        if stop <= start or step <= 0:
            raise HTTPException(400, "stop must exceed start and step positive")
        return _clean(torque_curve(session.state, range(start, stop + 1, step)))

    @app.get("/api/geometry")
    def geometry(tolerance: float = 0.3):
        try:
            mesh = geo.tessellate(session.state, tolerance=tolerance)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        parts = geo.build_all(session.state)
        return _clean({
            "ok": True,
            "meshes": mesh,
            "properties": {k: v["properties"].as_dict()
                           for k, v in parts.items()},
            "pin_axis": "y",
        })

    @app.get("/api/optimise")
    def run_optimise(objective: str = "torque", levers: int = 8,
                     classes: str = "", min_sf: float | None = None):
        from ..optimise import DEFAULT_CLASSES, optimise

        chosen = tuple(c for c in classes.split(",") if c) or DEFAULT_CLASSES
        try:
            result = optimise(session.state, objective, classes=chosen,
                              max_levers=levers, min_safety_factor=min_sf)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        except Exception as exc:                              # noqa: BLE001
            traceback.print_exc()
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        # The searched design is NOT committed. It comes back as a proposal
        # the user accepts or throws away, like every other write in the tool.
        return _clean({"ok": True, **result.as_dict(),
                       "summary": result.summary()})

    @app.get("/api/frontier")
    def run_frontier(objective: str = "torque", points: int = 6,
                     levers: int = 6):
        from ..optimise import frontier

        try:
            front = frontier(session.state, objective, points=points,
                             max_levers=levers)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        except Exception as exc:                              # noqa: BLE001
            traceback.print_exc()
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        return _clean({"ok": True, **front.as_dict()})

    @app.get("/api/envelope")
    def block_envelope():
        from ..envelope import envelope as compute_envelope
        return _clean({"ok": True, **compute_envelope(session.state).as_dict()})

    @app.get("/api/overbore")
    def overbore(bore_mm: float, fast: bool = False):
        from ..cascade import overbore_cascade
        from ..state import ConstraintViolation

        try:
            result = overbore_cascade(session.state, bore_mm / 1000.0,
                                      remeasure=not fast)
        except ConstraintViolation as exc:
            # The block refusing is an answer with a reason in it, not a
            # server error.
            return {"ok": False, "refused": True, "error": str(exc)}
        except Exception as exc:                          # noqa: BLE001
            traceback.print_exc()
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        return _clean({"ok": True, **result.as_dict(),
                       "summary": result.summary()})

    @app.get("/api/fea/cases")
    def fea_cases():
        """Declared BEFORE /api/fea/{part} on purpose: FastAPI matches routes
        in declaration order, so the parameterised route would otherwise
        swallow this one and answer "no load case for 'cases'"."""
        try:
            from ..fea.cases import CASES
        except ImportError:
            return {"ok": True, "parts": []}
        return {"ok": True, "parts": sorted(CASES)}

    @app.get("/api/fea/{part}")
    def fea(part: str, elements: int = 25_000):
        """Run a component load case and return a shadeable stress field.

        This is deliberately synchronous. A 25,000 element pin meshes and
        solves in about two seconds, which is slow enough to warrant the
        spinner the front end shows and fast enough not to need a job queue.
        """
        # The FEA libraries are the only dependencies added after the tool was
        # first installed, so on a machine where everything else works these
        # are what will be missing. skfem is imported lazily INSIDE the solver,
        # so the failure surfaces mid-solve rather than here -- both paths have
        # to say what to type, or it reads as a bare crash.
        def missing(exc):
            return {"ok": False,
                    "error": f"the FEA libraries are not installed ({exc}). "
                             "Install them with:  python -m pip install "
                             "scikit-fem tetgen scipy   then restart the tool."}

        try:
            from ..fea.cases import CASES
            from ..fea.field import field_payload
        except ImportError as exc:
            return missing(exc)

        if part not in CASES:
            return {"ok": False,
                    "error": f"no load case for {part!r} yet; "
                             f"available: {', '.join(sorted(CASES))}"}
        try:
            case = CASES[part](session.state, target_elements=elements)
        except MemoryError:
            return {"ok": False,
                    "error": f"ran out of memory solving the {part} at "
                             f"{elements:,} elements. Try fewer."}
        except ImportError as exc:
            return missing(exc)
        except Exception as exc:                      # noqa: BLE001
            # Deliberately broad. A solver failure is a result to report, not
            # a reason to hand the browser a 500 it can only show as "failed".
            traceback.print_exc()
            return {"ok": False,
                    "error": f"{type(exc).__name__}: {exc}"}

        field = field_payload(case.solve)

        # The solver guards its own answer, but what the viewport paints is
        # the SURFACE field built afterwards -- boundary triangles, nodal
        # averaging, remapped indices. A bug in any of that turns a healthy
        # solve into a uniformly green part, and the solver would never know.
        # So the thing actually being drawn gets checked too.
        surface_peak = max(field["stress_mpa"]) if field["stress_mpa"] else 0.0
        if surface_peak <= 0.0:
            return {
                "ok": False,
                "error": (
                    f"the {part} solved to a peak of "
                    f"{case.solve.peak_stress / 1e6:.1f} MPa, but the surface "
                    "field drawn from it is zero everywhere. The solve is "
                    "fine and the surface extraction is wrong: "
                    f"{len(field['vertices']):,} vertices, "
                    f"{len(field['triangles']):,} triangles from "
                    f"{case.solve.mesh.n_elements:,} elements. Report those "
                    "numbers."),
            }

        # The whole cycle from the one solve. Linear elasticity with a single
        # load pattern means the stress everywhere scales exactly with the
        # load, so the field at any crank angle is this field times
        # load(theta)/load(reference). Decimated on the SAME stride as
        # /api/sweep so the viewport's slider index addresses both.
        cycle = None
        if case.load_series is not None:
            from ..fea.cases import allowable_for

            series = np.asarray(case.load_series, dtype=float)
            metrics = session.metrics()
            theta = metrics.sweep.theta
            stride = max(1, len(theta) // 361)
            keep = np.arange(0, len(theta), stride)
            reference = case.load_reference or 1.0
            cycle = {
                "theta_deg": np.degrees(theta[keep]).tolist(),
                "scale": (np.abs(series[keep]) / reference).tolist(),
                "signed": (series[keep] / reference).tolist(),
                "load": series[keep].tolist(),
                "unit": case.load_unit,
                "reference": float(reference),
            }

        return _clean({
            "ok": True,
            "part": part,
            "field": field,
            "cycle": cycle,
            "allowable": allowable_for(session.state, part),
            "force_n": case.force,
            "condition": case.condition,
            "analytical_stress_pa": case.analytical_stress,
            "analytical_equation": case.analytical_equation,
            "fea_over_analytical": case.ratio,
            "sampled_stress_pa": case.sampled_stress,
            "sample_region": case.sample_region,
            "peak_stress_pa": case.solve.peak_stress,
            "surface_peak_mpa": surface_peak,
            "percentile_stress_pa": case.solve.percentile_stress(),
            "residual": case.solve.residual,
            "free_rigid_modes": case.solve.free_rigid_modes,
            "notes": case.solve.notes,
        })

    @app.get("/api/geometry/check")
    def geometry_check(tolerance: float = 0.03):
        return _clean(geo.check_masses(session.state, tolerance=tolerance))

    @app.get("/api/profiles")
    def profiles():
        return _clean(geo.profile_study(session.state))

    @app.post("/api/rpm/{rpm}")
    def set_rpm(rpm: float):
        return _clean(session.set({"operating.speed": rpm_to_rad_s(rpm)},
                                  actor="user",
                                  rationale="operating point moved in the UI"))

    # -- the assistant ------------------------------------------------------

    app.state.agent = None

    def agent() -> Agent:
        if app.state.agent is None:
            app.state.agent = Agent(session)
        return app.state.agent

    @app.get("/api/ai/status")
    def ai_status():
        """Whether the assistant can run, and why not if it cannot."""
        current = agent()
        return {
            "available": current.available,
            "model": current.model,
            "tools": [t["name"] for t in TOOLS],
            "reason": None if current.available else no_key_message(),
            "env_file": str(ENV_FILE),
            "env_file_found": ENV_FILE.is_file(),
            "key_source": (
                "environment" if os.environ.get("ANTHROPIC_API_KEY")
                and not ENV_FILE.is_file() else
                ".env file" if current.available else None),
            "turns": len([m for m in current.messages if m["role"] == "user"]),
        }

    @app.post("/api/chat")
    def chat(request: ChatRequest):
        """Stream one assistant turn as server-sent events.

        Each event is one JSON object: text as it arrives, a chip per tool
        call, and a card for any staged proposal. The stream stays open for
        the whole turn, which can include several rounds of tool calls.
        """
        def events():
            try:
                for event in agent().run(request.message):
                    yield "data: " + json.dumps(_clean(event)) + "\n\n"
            except Exception as exc:                            # noqa: BLE001
                yield "data: " + json.dumps({
                    "type": "error",
                    "message": f"{type(exc).__name__}: {exc}"}) + "\n\n"
                yield "data: " + json.dumps({"type": "done"}) + "\n\n"

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    @app.post("/api/chat/reset")
    def chat_reset():
        app.state.agent = None
        return {"ok": True}

    # -- the page -----------------------------------------------------------

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.middleware("http")
    async def never_cache_the_front_end(request, call_next):
        """Stop the browser serving a stale page after the tool is updated.

        FastAPI sends ETag and Last-Modified for static files, which is
        correct and not enough: Chrome will reuse a script from its memory
        cache for the rest of a session without revalidating. The symptom is
        vicious -- the Python restarts and picks up every change, the files on
        disk are right, and the page keeps running last week's JavaScript, so
        the evidence says the folder was never updated when it was.

        This is a local tool whose assets are a few hundred kilobytes off
        localhost, so there is nothing to gain by caching them at all.
        """
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static"):
            response.headers["Cache-Control"] = "no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

    @app.get("/")
    def index():
        return FileResponse(os.path.join(STATIC, "index.html"))

    @app.get("/api/version")
    def version():
        """What the running server actually has, so "is it stale?" is a
        question with an answer rather than a guess."""
        import hashlib

        files = {}
        for name in ("index.html", "app.js", "styles.css"):
            path = os.path.join(STATIC, name)
            try:
                raw = open(path, "rb").read()
            except OSError:
                continue
            files[name] = {
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest()[:12],
                "modified": os.path.getmtime(path),
            }
        # Capabilities read off the LOADED modules, not off the files. The
        # first version of this endpoint checked the source on disk, which
        # answers the wrong question entirely: the disk is always current
        # after an edit, while the running process keeps whatever it
        # imported at startup. "Your files are new" is not "your server is
        # new", and confusing the two costs a debugging session.
        from .. import fea as fea_pkg
        from ..fea import cases as fea_cases
        from ..fea import mesh as fea_mesh
        from ..fea import solve as fea_solve

        def has_const(func, needle):
            try:
                return any(isinstance(c, str) and needle in c
                           for c in func.__code__.co_consts)
            except AttributeError:
                return False

        loaded = {
            "distributed_bearings": hasattr(fea_solve, "Bearing"),
            "stress_tensors": hasattr(fea_solve, "stress_tensors"),
            "sliver_guard": hasattr(fea_solve, "DEGENERATE_RATIO"),
            "nan_guard": has_const(fea_solve.solve_linear_elastic,
                                   "non-finite displacements"),
            "zero_stress_guard": has_const(fea_solve.solve_linear_elastic,
                                           "recovered zero"),
            "load_reporting": has_const(fea_solve.solve_linear_elastic,
                                        "applied load"),
            "crank_cycle": hasattr(fea_cases, "allowable_for"),
            "calibration": hasattr(fea_pkg, "STATUS"),
            "solver_ladder": hasattr(fea_solve, "RESIDUAL_TOLERANCE"),
            "pinch_repair": hasattr(fea_mesh, "_face_components"),
        }

        return {
            "ok": True,
            "python": sys.version.split()[0],
            "package_dir": os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))),
            "static": files,
            "loaded": loaded,
            "stale": [name for name, present in loaded.items() if not present],
            "up_to_date": all(loaded.values()),
        }

    return app


def serve(state_path: str | None = None, host: str = "127.0.0.1",
          port: int = 8000, open_browser: bool = True) -> None:
    import uvicorn

    if open_browser:
        import threading
        import webbrowser
        threading.Timer(
            1.2, lambda: webbrowser.open(f"http://{host}:{port}/")).start()
    uvicorn.run(create_app(state_path), host=host, port=port, log_level="warning")
