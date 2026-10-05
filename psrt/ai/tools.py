"""The toolbox the model is given, and the dispatcher behind it.

Every tool is a thin wrapper over the same pure functions the CLI and the
browser use. Two properties matter more than the rest:

**Refusals are answers.** A write to a locked parameter, or outside a bounded
range, comes back as a structured explanation of the constraint -- not an
exception, not a silent clamp. The model has to reason around a real limit
instead of being politely asked to respect one.

**Proposals are staged.** ``propose`` evaluates against a copy and reports
what would happen, including what got worse. Nothing touches the live design
until ``commit``. The worst case of letting a model drive is a wasted
evaluation.

Units are SI throughout, because that is what the design state holds: metres,
kilograms, pascals, newtons, radians, rad/s. Every tool that returns a
parameter says its unit, and a write that looks like it arrived in
millimetres is refused with that spelled out rather than being quietly
accepted as a 100-metre bore.
"""

from __future__ import annotations

import math

from .. import geometry as geo
from ..evaluate import torque_curve
from ..sensitivity import OBJECTIVES, rank_levers
from ..state import Mutability
from ..units import rad_s_to_rpm, rpm_to_rad_s

# --- schemas ---------------------------------------------------------------

TOOLS = [
    {
        "name": "get_design_state",
        "description": (
            "Read the design. Returns every parameter with its SI value, its "
            "unit, its mutability class (free / bounded / locked / derived), "
            "its bounds and the physical reason behind each bound. Start here "
            "when you need to know what you are allowed to change."),
        "input_schema": {
            "type": "object",
            "properties": {
                "section": {
                    "type": "string",
                    "description": ("Limit to one section: engine, operating, "
                                    "masses, materials, block, constraints, "
                                    "sleeve, piston, pin, small_end, rod, "
                                    "bolts, thermal, fatigue. Omit for all."),
                },
                "editable_only": {
                    "type": "boolean",
                    "description": "Only parameters you can actually write.",
                },
            },
        },
    },
    {
        "name": "get_metrics",
        "description": (
            "Everything computed from the current design at the current "
            "operating point: geometry, masses, combustion, peak loads, "
            "indicated and brake output, the energy-closure check, and the "
            "structural summary with the binding constraint named."),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_margins",
        "description": (
            "All twenty-five structural margins, worst first, each with its "
            "safety factor, the condition it was evaluated at, and whether it "
            "moves with the operating point."),
        "input_schema": {
            "type": "object",
            "properties": {
                "component": {
                    "type": "string",
                    "description": ("Filter to one component: sleeve, crown, "
                                    "ring lands, skirt, pin, small end, rod, "
                                    "rod bolts, big end."),
                },
            },
        },
    },
    {
        "name": "explain_margin",
        "description": (
            "The full working behind a margin: the equation used, where that "
            "equation comes from, every input that went into it, the Marin "
            "factors for a fatigue check, the predicted life, and what the "
            "model does NOT capture. Use this before claiming anything about "
            "why a component is limiting."),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": ("Component name or words from the failure "
                                    "mode, e.g. 'crown' or 'buckling'."),
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_curve",
        "description": (
            "Torque, power and peak loads against engine speed. Each point is "
            "a full evaluation, so inertia relief on the gas load and the "
            "rising tensile load at overlap TDC are both captured."),
        "input_schema": {
            "type": "object",
            "properties": {
                "start_rpm": {"type": "number"},
                "stop_rpm": {"type": "number"},
                "step_rpm": {"type": "number"},
            },
        },
    },
    {
        "name": "rank_levers",
        "description": (
            "Answer 'what are my options'. Perturbs every unlocked design "
            "parameter, measures its effect on an objective, then finds how "
            "far it can actually move before a bound or a safety factor stops "
            "it -- and says which. Results are ranked by what each lever can "
            "deliver, not by how sharply it responds. Slow (a second or two): "
            "it runs hundreds of evaluations."),
        "input_schema": {
            "type": "object",
            "properties": {
                "objective": {
                    "type": "string",
                    "description": "One of: " + ", ".join(sorted(OBJECTIVES)),
                },
                "paths": {
                    "type": "array", "items": {"type": "string"},
                    "description": ("Restrict to these parameters. Omit for "
                                    "the standard design-lever set."),
                },
            },
            "required": ["objective"],
        },
    },
    {
        "name": "propose",
        "description": (
            "Evaluate a change WITHOUT applying it. Returns the resulting "
            "metrics, the deltas, and explicitly what got worse. The live "
            "design is untouched until commit. Always propose before you "
            "commit, and always report the regressions to the user."),
        "input_schema": {
            "type": "object",
            "properties": {
                "changes": {
                    "type": "object",
                    "description": ("Dotted parameter paths to new SI values, "
                                    "e.g. {\"engine.bore\": 0.1035}. Metres, "
                                    "not millimetres."),
                },
                "rationale": {
                    "type": "string",
                    "description": ("Why, in one line. This is recorded in the "
                                    "design's change log and is what the user "
                                    "reads back later."),
                },
            },
            "required": ["changes", "rationale"],
        },
    },
    {
        "name": "commit",
        "description": "Apply a staged proposal to the live design.",
        "input_schema": {
            "type": "object",
            "properties": {"proposal_id": {"type": "string"}},
            "required": ["proposal_id"],
        },
    },
    {
        "name": "discard",
        "description": "Throw a staged proposal away.",
        "input_schema": {
            "type": "object",
            "properties": {"proposal_id": {"type": "string"}},
            "required": ["proposal_id"],
        },
    },
    {
        "name": "revert",
        "description": "Undo the last committed change to the live design.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "set_speed",
        "description": (
            "Move the operating point. Many constraints migrate with engine "
            "speed: gas load dominates in the mid range, inertia at the top."),
        "input_schema": {
            "type": "object",
            "properties": {"rpm": {"type": "number"}},
            "required": ["rpm"],
        },
    },
    {
        "name": "check_geometry",
        "description": (
            "Rebuild the solids and compare their true masses and the rod's "
            "centre of mass against what the design state claims. Slow (about "
            "a second). Use it after changing geometry, because the fast "
            "analytical layer does NOT rebuild solids and its masses will "
            "drift."),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "optimise",
        "description": (
            "Search the design toward an objective while respecting every "
            "parameter bound and every safety factor. Returns the levers "
            "that moved, what got worse, which constraint is binding at the "
            "end, and a list of reasons to distrust the answer -- read those "
            "out, do not hide them. It does NOT commit anything. By default "
            "it only moves what you could change on this block with "
            "remachined parts; pass classes to include a new crank "
            "('rotating') or the combustion timing ('tuning'), both of which "
            "mean something other than a redesigned piston system. Slow "
            "(seconds to a minute)."),
        "input_schema": {
            "type": "object",
            "properties": {
                "objective": {
                    "type": "string",
                    "description": ("torque, power, safety_factor, "
                                    "reciprocating_mass, piston_mass, "
                                    "side_thrust, displacement"),
                },
                "classes": {
                    "type": "array", "items": {"type": "string"},
                    "description": ("machining, piston, rod, rotating, "
                                    "tuning. Default: machining, piston, rod"),
                },
                "min_safety_factor": {"type": "number"},
                "levers": {"type": "integer"},
            },
            "required": ["objective"],
        },
    },
    {
        "name": "frontier",
        "description": (
            "What the objective is worth at each of several safety factors. "
            "Use this when asked what something COSTS, or when the honest "
            "answer is a trade rather than a number -- it turns 'what is the "
            "optimum' into 'how much torque does the next 0.1 of margin "
            "cost'. Slow (tens of seconds)."),
        "input_schema": {
            "type": "object",
            "properties": {
                "objective": {"type": "string"},
                "points": {"type": "integer"},
            },
            "required": ["objective"],
        },
    },
    {
        "name": "block_envelope",
        "description": (
            "What the BLOCK will allow. Machining is subtractive -- material "
            "comes off, never on -- so the bore has a hard floor at its "
            "as-built size and a ceiling set by whichever internal feature "
            "the wall reaches first. Returns every limit that could be "
            "evaluated, which one binds, which could NOT be evaluated, and a "
            "confidence tier: 1 published or measured, 2 aftermarket piston "
            "availability, 3 assumed and needing a sonic test. Read this "
            "before suggesting any bore change, and quote the binding limit "
            "and its tier when you do."),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "overbore_cascade",
        "description": (
            "What ELSE moves if the bore goes to a given size. Enlarging a "
            "bore is never a single-parameter change: displacement and "
            "compression ratio move, the piston gets heavier, inertia and "
            "side thrust rise, the wall thins and every structural margin "
            "shifts. Returns what got better AND what got worse, plus any "
            "margin that fell. Slow (rebuilds the solids to re-measure the "
            "masses). Never recommend an overbore without running this -- "
            "reporting only the torque gain is the single most misleading "
            "thing this tool could do."),
        "input_schema": {
            "type": "object",
            "properties": {
                "bore_mm": {
                    "type": "number",
                    "description": "Target bore in millimetres.",
                },
            },
            "required": ["bore_mm"],
        },
    },
    {
        "name": "compare_bore_profiles",
        "description": (
            "What each bore cross-section -- circle, polygon, oval -- could "
            "displace inside this block's own envelope, and whether a piston "
            "ring can seal it."),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "railrod_report",
        "description": (
            "The rail connecting rod concept (monolithic small end and twin "
            "steel rails, stabilising sleeve, big-end receiver, two swing "
            "clamps hooked into radiused rail notches, one tangential bolt). "
            "Returns part masses, notch geometry, the swing/assembly check, "
            "whether the ramp is steep enough to self-seat, and the contact "
            "forces at assembly, peak tension and peak firing. Its "
            "parameters live in the 'railrod' section; set "
            "railrod.enabled to true (via propose/commit) to use it. With "
            "coupled=true the parts' FEA flexibility is coupled into the "
            "contact network (tens of seconds the first time); the rigid "
            "network is optimistic about load sharing and says so."),
        "input_schema": {
            "type": "object",
            "properties": {
                "coupled": {"type": "boolean",
                            "description": "use the FEA-coupled network"},
            },
        },
    },
    {
        "name": "railrod_fea",
        "description": (
            "Finite element analysis of ONE rail-rod part through the whole "
            "crank cycle: rr_rails, rr_sleeve, rr_receiver, rr_clamp_right, "
            "rr_clamp_left or rr_bolt. Each part is solved separately, "
            "loaded by the contact forces the coupled network gives it at "
            "every crank angle. Returns the cycle-peak von Mises, where and "
            "at what angle it occurs, a static and a Goodman fatigue safety "
            "factor, and the equilibrium check. Slow: up to a minute."),
        "input_schema": {
            "type": "object",
            "properties": {
                "part": {"type": "string",
                         "enum": ["rr_rails", "rr_sleeve", "rr_receiver",
                                  "rr_clamp_right", "rr_clamp_left",
                                  "rr_bolt"]},
                "elements": {"type": "integer",
                             "description": "target mesh size, default "
                                            "20000"},
            },
            "required": ["part"],
        },
    },
]


def tool_names() -> list:
    return [t["name"] for t in TOOLS]


# --- dispatch --------------------------------------------------------------

class ToolError(Exception):
    pass


def _clean(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def _unit_hint(session, changes: dict) -> str | None:
    """Catch the millimetre mistake before the bounds check does.

    A model that writes 103.25 for a bore in metres is not confused about
    engines, it is confused about units, and saying so is more useful than
    'above the upper bound of 0.10376'.
    """
    for path, value in changes.items():
        if not session.state.has(path) or not isinstance(value, (int, float)):
            continue
        param = session.state.param(path)
        if param.unit != "m" or not isinstance(param.value, (int, float)):
            continue
        current = param.value
        if current and value / current > 100.0 and abs(value / 1000.0 - current) < current:
            return (f"{path} takes METRES. You passed {value}, which is "
                    f"{value / 1000.0} m if you meant millimetres; the current "
                    f"value is {current} m ({current * 1000:.3f} mm).")
    return None


def dispatch(session, name: str, arguments: dict) -> dict:
    """Run one tool against a live Session. Always returns JSON-safe output."""
    arguments = arguments or {}
    try:
        return _clean(_dispatch(session, name, arguments))
    except ToolError as exc:
        return {"error": str(exc)}
    except Exception as exc:                                  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def _dispatch(session, name: str, args: dict) -> dict:
    state = session.state

    if name == "get_design_state":
        section = args.get("section")
        editable_only = bool(args.get("editable_only"))
        out = {}
        for sec, params in state.sections.items():
            if section and sec != section:
                continue
            rows = {}
            for key, p in params.items():
                editable = p.mutability in (Mutability.FREE, Mutability.BOUNDED)
                if editable_only and not editable:
                    continue
                path = f"{sec}.{key}"
                try:
                    value = state.get(path)
                except Exception:                              # noqa: BLE001
                    value = None
                entry = {"value": value, "unit": p.unit,
                         "class": p.mutability.value}
                if p.minimum is not None:
                    entry["min"] = p.minimum
                if p.maximum is not None:
                    entry["max"] = p.maximum
                for k in ("why", "why_min", "why_max", "description"):
                    if getattr(p, k):
                        entry[k] = getattr(p, k)
                if p.source != "user":
                    entry["source"] = p.source
                rows[key] = entry
            if rows:
                out[sec] = rows
        if section and not out:
            raise ToolError(f"no section named {section!r}; sections are: "
                            + ", ".join(state.sections))
        return {"units": "SI", "parameters": out,
                "provenance": state.meta.get("provenance", "")}

    if name == "get_metrics":
        return session.metrics().to_dict()

    if name == "get_margins":
        report = session.metrics().report
        margins = report.margins
        if args.get("component"):
            wanted = args["component"].lower()
            margins = [m for m in margins if wanted in m.component.lower()]
            if not margins:
                raise ToolError(
                    f"no component matching {args['component']!r}; try: "
                    + ", ".join(sorted({m.component for m in report.margins})))
        return {
            "binding": report.to_dict()["binding"],
            "binding_operating": report.to_dict()["binding_operating"],
            "margins": [
                {k: v for k, v in m.to_dict().items()
                 if k not in ("inputs",)}
                for m in sorted(margins, key=lambda m: m.safety_factor)],
        }

    if name == "explain_margin":
        query = str(args["query"]).lower()
        report = session.metrics().report
        hits = [m for m in report.margins
                if query in m.component.lower() or query in m.mode.lower()]
        if not hits:
            raise ToolError(
                f"nothing matching {query!r}. Components: "
                + ", ".join(sorted({m.component for m in report.margins})))
        return {"matches": [m.to_dict()
                            for m in sorted(hits, key=lambda m: m.safety_factor)]}

    if name == "get_curve":
        start = int(args.get("start_rpm", 1500))
        stop = int(args.get("stop_rpm", 7000))
        step = int(args.get("step_rpm", 500))
        if stop <= start or step <= 0:
            raise ToolError("stop_rpm must exceed start_rpm and step_rpm > 0")
        if (stop - start) / step > 80:
            raise ToolError("that is more than 80 points; use a bigger step")
        return torque_curve(state, range(start, stop + 1, step))

    if name == "rank_levers":
        objective = args.get("objective", "torque")
        return rank_levers(state, objective=objective, paths=args.get("paths"))

    if name == "propose":
        changes = args.get("changes") or {}
        if not isinstance(changes, dict) or not changes:
            raise ToolError("changes must be a non-empty object of "
                            "path -> SI value")
        hint = _unit_hint(session, changes)
        if hint:
            return {"ok": False, "violation": {"error": "unit_mistake",
                                               "reason": hint}}
        result = session.propose(changes, args.get("rationale", ""), actor="ai")
        if result.get("ok"):
            result["metrics"] = {
                k: v for k, v in result["metrics"].items()
                if k in ("performance", "loads", "structural", "masses",
                         "combustion", "warnings")}
        return result

    if name == "commit":
        return session.commit(str(args["proposal_id"]))

    if name == "discard":
        return session.discard(str(args["proposal_id"]))

    if name == "revert":
        return session.revert()

    if name == "set_speed":
        rpm = float(args["rpm"])
        if not 200 <= rpm <= 20000:
            raise ToolError("rpm must be between 200 and 20000")
        result = session.set({"operating.speed": rpm_to_rad_s(rpm)},
                             actor="ai", rationale=f"operating point {rpm:.0f} rpm")
        if result.get("ok"):
            metrics = session.metrics()
            result["now"] = {
                "rpm": rad_s_to_rpm(metrics.sweep.speed),
                "brake_torque_nm": metrics.performance["brake_torque_nm"],
                "minimum_safety_factor":
                    metrics.structural["minimum_safety_factor"],
                "binding": metrics.structural["binding_operating_mode"],
            }
        return result

    if name == "check_geometry":
        return geo.check_masses(state)

    if name == "optimise":
        from ..optimise import DEFAULT_CLASSES, optimise

        classes = tuple(args.get("classes") or ()) or DEFAULT_CLASSES
        result = optimise(state, args.get("objective", "torque"),
                          classes=classes,
                          max_levers=int(args.get("levers", 8)),
                          min_safety_factor=args.get("min_safety_factor"))
        out = result.as_dict()
        out.pop("state", None)
        return out

    if name == "frontier":
        from ..optimise import frontier

        front = frontier(state, args.get("objective", "torque"),
                         points=int(args.get("points", 6)))
        return front.as_dict()

    if name == "block_envelope":
        from ..envelope import envelope as compute_envelope
        return compute_envelope(state).as_dict()

    if name == "overbore_cascade":
        from ..cascade import overbore_cascade
        from ..state import ConstraintViolation

        bore_mm = args.get("bore_mm")
        if bore_mm is None:
            raise ToolError("overbore_cascade needs bore_mm")
        try:
            result = overbore_cascade(state, float(bore_mm) / 1000.0)
        except ConstraintViolation as exc:
            # A refusal is the answer, not an error: it carries the physical
            # reason, which is exactly what the model should relay.
            return {"refused": True, "reason": str(exc)}
        return result.as_dict()

    if name == "compare_bore_profiles":
        return geo.profile_study(state)

    if name in ("railrod_report", "railrod_fea"):
        from ..railrod import analysis as rr
        if not rr.enabled(state):
            raise ToolError(
                "the rail rod is not enabled for this design; propose "
                "railrod.enabled = true first")
        if name == "railrod_report":
            if args.get("coupled"):
                from ..railrod.coupled import coupled_analysis
                a = coupled_analysis(state)
            else:
                a = rr.analyse(state)
            out = a.summary()
            out["network"] = ("coupled" if args.get("coupled") else
                              "rigid -- load sharing is optimistic")
            return out
        from ..railrod import fea
        part = args.get("part")
        if part not in fea.PARTS_WITH_FEA:
            raise ToolError(f"part must be one of {', '.join(fea.PARTS_WITH_FEA)}")
        result = fea.solve_part(state, part,
                                target_elements=int(args.get("elements",
                                                             20_000)))
        return fea.summarise(result, state)

    raise ToolError(f"unknown tool {name!r}; available: "
                    + ", ".join(tool_names()))
