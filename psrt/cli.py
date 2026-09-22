"""Command-line interface for the piston system research tool.

    python -m psrt new my-engine.json
    python -m psrt show my-engine.json
    python -m psrt loads my-engine.json --rpm 6500
    python -m psrt curve my-engine.json
    python -m psrt export my-engine.json --out sweep.csv --rpm 6500
    python -m psrt compare before.json after.json
    python -m psrt materials

Phase 4 puts a browser front end on top of exactly these calls. Until then
this is the whole interface, and it is enough to do real work with.
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np

from . import materials as materials_mod
from . import units as u
from .evaluate import compare as compare_metrics, evaluate, torque_curve
from .schema import default_state, load_state, refresh_bounds
from .state import ConstraintViolation, DesignState, Mutability

BAR = "-" * 74


def _hdr(title: str) -> None:
    print(f"\n{title}\n{BAR}")


def _row(label: str, value: str, extra: str = "") -> None:
    print(f"  {label:<34} {value:>18}  {extra}")


# --- commands --------------------------------------------------------------

def cmd_new(args) -> int:
    state = default_state(args.name)
    state.save(args.out)
    print(f"Wrote a default design state to {args.out}")
    print("It holds generic placeholder values. Replace them with a real "
          "engine before trusting any result.")
    return 0


def cmd_show(args) -> int:
    state, notes = load_state(args.state)

    print(f"\n{state.meta.get('name', 'unnamed')}")
    if state.meta.get("notes"):
        print(f"  {state.meta['notes']}")
    print(f"  fingerprint {state.fingerprint()}")

    for section in state.sections:
        _hdr(section.upper())
        for name, p in state.sections[section].items():
            path = f"{section}.{name}"
            try:
                value = state.get(path)
            except Exception as exc:                      # pragma: no cover
                value = f"<{exc}>"
            shown = u.fmt(value, p.unit)
            flag = {
                Mutability.FREE: "",
                Mutability.BOUNDED: "[bounded]",
                Mutability.LOCKED: "[LOCKED]",
                Mutability.DERIVED: "[derived]",
            }[p.mutability]
            _row(name, shown, flag)
            if args.verbose:
                if p.description:
                    print(f"      {p.description}")
                if p.mutability is Mutability.BOUNDED:
                    lo = u.fmt(p.minimum, p.unit) if p.minimum is not None else "-"
                    hi = u.fmt(p.maximum, p.unit) if p.maximum is not None else "-"
                    print(f"      range {lo} .. {hi}")
                    if p.why_min:
                        print(f"      min: {p.why_min}")
                    if p.why_max:
                        print(f"      max: {p.why_max}")
                if p.mutability is Mutability.LOCKED and p.why:
                    print(f"      locked: {p.why}")

    if notes:
        _hdr("BOUNDS REFRESHED")
        for n in notes:
            print(f"  - {n}")

    issues = state.validate()
    _hdr("VALIDATION")
    if issues:
        for i in issues:
            print(f"  ISSUE  {i}")
        return 1
    print("  state is internally consistent")
    return 0


def _apply_rpm(state: DesignState, rpm: float | None) -> None:
    if rpm is not None:
        state.set("operating.speed", u.rpm_to_rad_s(rpm), actor="cli",
                  rationale="operating point set on the command line")


def cmd_loads(args) -> int:
    state, _ = load_state(args.state)
    _apply_rpm(state, args.rpm)
    m = evaluate(state)
    sweep = m.sweep

    print(f"\n{state.meta.get('name', 'unnamed')} at "
          f"{m.performance['speed_rpm']:.0f} rpm")

    _hdr("GEOMETRY")
    g = m.geometry
    _row("bore x stroke", f"{g['bore_m'] * 1e3:.2f} x {g['stroke_m'] * 1e3:.2f} mm")
    _row("displacement", u.fmt(g["displacement_total_m3"], "m^3"),
         f"({g['n_cylinders']} cyl)")
    _row("compression ratio", f"{g['compression_ratio']:.2f} : 1")
    _row("rod ratio (l/r)", f"{g['rod_ratio']:.3f}",
         f"max obliquity {g['max_rod_angle_deg']:.1f} deg")
    _row("mean piston speed", f"{g['mean_piston_speed_m_s']:.2f} m/s")

    _hdr("MASSES")
    ms = m.masses
    _row("piston assembly", u.fmt(ms["piston_assembly_kg"], "kg"))
    _row("rod, reciprocating share", u.fmt(ms["rod_reciprocating_kg"], "kg"))
    _row("RECIPROCATING TOTAL", u.fmt(ms["reciprocating_kg"], "kg"))
    _row("rod, rotating share", u.fmt(ms["rod_rotating_kg"], "kg"))

    _hdr("COMBUSTION")
    c = m.combustion
    _row("peak cylinder pressure", u.fmt(c["peak_pressure_pa"], "Pa"),
         f"at {c['peak_pressure_angle_deg']:.1f} deg ATDC")
    _row("max pressure rise rate",
         f"{c['max_pressure_rise_bar_per_deg']:.2f} bar/deg")
    _row("IMEP gross", u.fmt(c["imep_gross_pa"], "Pa"))
    _row("IMEP net", u.fmt(c["imep_net_pa"], "Pa"))
    _row("trapped charge", f"{c['trapped_mass_kg'] * 1e6:.1f} mg",
         f"source: {c['source']}")

    _hdr("LOADS")
    ld = m.loads
    _row("peak pin force, COMPRESSION", u.fmt(ld["peak_pin_compression_n"], "N"),
         f"at {ld['peak_pin_compression_deg']:.1f} deg")
    _row("peak pin force, TENSION", u.fmt(ld["peak_pin_tension_n"], "N"),
         f"at {ld['peak_pin_tension_deg']:.1f} deg")
    _row("peak side thrust", u.fmt(ld["peak_side_thrust_n"], "N"),
         f"at {ld['peak_side_thrust_deg']:.1f} deg")
    _row("peak piston acceleration", f"{ld['peak_acceleration_g']:.0f} g")
    _row("rod tension at overlap TDC",
         u.fmt(ld["overlap_tdc_rod_tension_n"], "N"),
         "inertia only, no gas force")

    _hdr("INDICATED PERFORMANCE")
    pf = m.performance
    _row("indicated torque", f"{pf['indicated_torque_nm']:.1f} N.m")
    _row("indicated power", u.fmt(pf["indicated_power_w"], "W"),
         f"= {u.w_to_hp(pf['indicated_power_w']):.1f} hp")
    print(f"  {pf['note']}")

    _hdr("CHECKS")
    ck = m.checks
    verdict = "PASS" if ck["energy_closure_pass"] else "FAIL"
    _row("energy closure", f"{ck['energy_closure_error'] * 100:.5f} %",
         f"{verdict} (integrated torque vs IMEP x Vd)")

    _hdr("STRUCTURAL MARGINS")
    st = m.structural
    for comp, sf in sorted(st["by_component"].items(),
                           key=lambda kv: (kv[1] is None, kv[1])):
        flag = "" if sf is None or sf >= 1.0 else "  FAIL"
        shown = "inf" if sf is None else f"{sf:.2f}"
        _row(comp, shown, flag)
    print(f"\n  binding: {st['binding_component']} - {st['binding_mode']}")
    print(f"  run 'psrt margins {args.state}' for the full report")

    if m.warnings:
        _hdr("WARNINGS")
        for w in m.warnings:
            print(f"  ! {w}")
    if m.notes and args.verbose:
        _hdr("NOTES")
        for n in m.notes:
            print(f"  - {n}")

    if args.angles:
        _hdr("AT SPECIFIC CRANK ANGLES")
        print(f"  {'deg':>7} {'p (bar)':>10} {'F_gas (kN)':>12} "
              f"{'F_pin (kN)':>12} {'F_side (kN)':>12} {'T (N.m)':>10}")
        for a in args.angles:
            d = sweep.at_angle(a)
            print(f"  {d['angle_deg']:7.1f} {d['pressure'] / 1e5:10.2f} "
                  f"{d['f_gas'] / 1e3:12.2f} {d['f_pin'] / 1e3:12.2f} "
                  f"{d['f_side'] / 1e3:12.2f} {d['torque']:10.1f}")
    return 0


def cmd_curve(args) -> int:
    state, _ = load_state(args.state)
    rpms = list(range(args.start, args.stop + 1, args.step))
    tc = torque_curve(state, rpms)

    wide = "-" * 88
    print(f"\n{state.meta.get('name', 'unnamed')}: output and peak loads "
          f"against engine speed\n{wide}")
    print(f"  {'rpm':>6} {'ind Tq':>9} {'brake Tq':>9} {'brake':>8} "
          f"{'brake':>8} {'pin comp':>10} {'pin tens':>10} {'p max':>8}")
    print(f"  {'':>6} {'N.m':>9} {'N.m':>9} {'kW':>8} "
          f"{'hp':>8} {'kN':>10} {'kN':>10} {'bar':>8}")
    for i, rpm in enumerate(tc["rpm"]):
        print(f"  {rpm:6.0f} {tc['indicated_torque_nm'][i]:9.1f} "
              f"{tc['brake_torque_nm'][i]:9.1f} "
              f"{tc['brake_power_w'][i] / 1e3:8.1f} "
              f"{u.w_to_hp(tc['brake_power_w'][i]):8.1f} "
              f"{tc['peak_pin_compression_n'][i] / 1e3:10.2f} "
              f"{tc['peak_pin_tension_n'][i] / 1e3:10.2f} "
              f"{tc['peak_pressure_pa'][i] / 1e5:8.1f}")

    _hdr("PEAKS")
    _row("peak BRAKE torque", f"{tc['peak_brake_torque_nm']:.1f} N.m",
         f"at {tc['peak_brake_torque_rpm']:.0f} rpm")
    _row("peak BRAKE power", f"{tc['peak_brake_power_w'] / 1e3:.1f} kW",
         f"= {u.w_to_hp(tc['peak_brake_power_w']):.1f} hp at "
         f"{tc['peak_brake_power_rpm']:.0f} rpm")
    _row("peak indicated torque", f"{tc['peak_torque_nm']:.1f} N.m",
         f"at {tc['peak_torque_rpm']:.0f} rpm")
    print("  brake figures subtract an empirical friction correlation; "
          "indicated is\n  what the model computes directly")

    i_hi = int(np.argmax(np.abs(tc["peak_pin_tension_n"])))
    print(f"\n  Note: peak tensile load rises with the square of engine speed "
          f"and\n  reaches {abs(tc['peak_pin_tension_n'][i_hi]) / 1e3:.1f} kN "
          f"at {tc['rpm'][i_hi]:.0f} rpm. Compare that with the compressive\n"
          f"  peak, which falls at high rpm as inertia relieves the gas load. "
          f"The\n  tensile case is what sizes the rod, and it is worst where "
          f"the engine\n  makes the least torque.")
    return 0


def cmd_export(args) -> int:
    state, _ = load_state(args.state)
    _apply_rpm(state, args.rpm)
    s = evaluate(state).sweep

    cols = {
        "crank_angle_deg": np.degrees(s.theta),
        "cylinder_pressure_pa": s.pressure,
        "cylinder_volume_m3": s.volume,
        "piston_position_m": s.position,
        "piston_velocity_m_s": s.velocity,
        "piston_acceleration_m_s2": s.acceleration,
        "rod_angle_rad": s.rod_angle,
        "f_gas_n": s.f_gas,
        "f_inertia_n": s.f_inertia,
        "f_pin_n": s.f_pin,
        "f_rod_n": s.f_rod,
        "f_side_n": s.f_side,
        "torque_nm": s.torque,
    }
    data = np.column_stack(list(cols.values()))
    header = ("# " + state.meta.get("name", "unnamed")
              + f" at {u.rad_s_to_rpm(s.speed):.0f} rpm; SI units; "
                f"f_pin and f_rod positive in compression\n"
              + ",".join(cols))
    np.savetxt(args.out, data, delimiter=",", header=header, comments="")
    print(f"Wrote {data.shape[0]} rows x {data.shape[1]} columns to {args.out}")
    return 0


def cmd_compare(args) -> int:
    a, _ = load_state(args.before)
    b, _ = load_state(args.after)
    deltas = compare_metrics(evaluate(a), evaluate(b))

    print(f"\n{args.before}  ->  {args.after}\n{BAR}")
    if not deltas:
        print("  no differences in any computed metric")
        return 0
    print(f"  {'metric':<42} {'before':>12} {'after':>12} {'change':>9}")
    for key in sorted(deltas):
        d = deltas[key]
        pct = f"{d['percent']:+.2f}%" if d["percent"] is not None else "-"
        print(f"  {key:<42} {d['before']:12.4g} {d['after']:12.4g} {pct:>9}")

    worse = [k for k in deltas
             if ("peak_pin" in k or "side_thrust" in k or "reciprocating" in k
                 or "peak_pressure" in k) and deltas[k]["delta"] > 0]
    if worse:
        print(f"\n  Got worse: {', '.join(sorted(worse))}")
    return 0


def cmd_margins(args) -> int:
    state, _ = load_state(args.state)
    _apply_rpm(state, args.rpm)
    m = evaluate(state)
    report = m.report

    print(f"\n{state.meta.get('name', 'unnamed')} structural margins at "
          f"{m.performance['speed_rpm']:.0f} rpm")

    _hdr("TEMPERATURES")
    th = m.thermal.as_dict()
    for label, key in (("crown, top face", "crown_top_c"),
                       ("crown, underside", "crown_underside_c"),
                       ("top land", "top_land_c"),
                       ("ring belt", "ring_belt_c"),
                       ("skirt", "skirt_c"),
                       ("pin", "pin_c"), ("connecting rod", "rod_c")):
        _row(label, f"{th[key]:.0f} C")
    _row("heat into the piston", f"{th['heat_to_piston_w']:.0f} W",
         f"flux {th['crown_heat_flux_w_m2'] / 1e3:.0f} kW/m2")
    print(f"  {m.thermal.note}")

    _hdr("MARGINS, WORST FIRST")
    print(f"  {'SF':>7}  {'component':<12} {'mode':<46} {'condition'}")
    for margin in report.sorted():
        sf = margin.safety_factor
        shown = "  inf" if math.isinf(sf) else f"{sf:7.2f}"
        flag = "" if margin.passes else "  <<< FAILS"
        print(f"  {shown}  {margin.component:<12} {margin.mode:<46} "
              f"{margin.condition}{flag}")

    binding = report.binding
    _hdr("BINDING CONSTRAINT")
    print(f"  {binding.component}: {binding.mode}")
    print(f"  safety factor {binding.safety_factor:.2f} at "
          f"{binding.condition}")
    print(f"  {binding.equation}")
    operating = report.binding_operating
    if operating is not None and operating is not binding:
        print(f"\n  Worst constraint that moves with the operating point:")
        print(f"  {operating.component}: {operating.mode} "
              f"(SF {operating.safety_factor:.2f})")
    if binding.life_cycles is not None and math.isfinite(binding.life_cycles) \
            and binding.life_hours is not None:
        print(f"  predicted life {binding.life_cycles:.3g} cycles "
              f"= {binding.life_hours:.0f} hours at this speed")

    if args.explain:
        for margin in report.margins:
            if (args.explain.lower() not in margin.component.lower()
                    and args.explain.lower() not in margin.mode.lower()):
                continue
            _hdr(f"{margin.component.upper()}: {margin.mode}")
            sf = margin.safety_factor
            _row("safety factor", "inf" if math.isinf(sf) else f"{sf:.3f}",
                 "PASS" if margin.passes else "FAIL")
            if margin.kind != "factor":
                _row("applied", u.fmt(margin.applied, margin.unit))
                _row("allowable", u.fmt(margin.allowable, margin.unit))
            print(f"\n  equation   {margin.equation}")
            print(f"  reference  {margin.reference}")
            print(f"  condition  {margin.condition}")
            if margin.life_cycles is not None:
                life = ("beyond the endurance limit"
                        if math.isinf(margin.life_cycles)
                        else f"{margin.life_cycles:.3g} cycles")
                print(f"  life       {life}")
            print("\n  inputs")
            for key, value in margin.inputs.items():
                if isinstance(value, float):
                    print(f"    {key:<42} {value:.6g}")
                elif value is not None:
                    print(f"    {key:<42} {value}")
            if margin.notes:
                print("\n  notes")
                for note in margin.notes:
                    print(f"    - {note}")

    if report.failing:
        _hdr("FAILING")
        for margin in report.failing:
            print(f"  {margin.component}: {margin.mode} "
                  f"(SF {margin.safety_factor:.2f})")
        return 1

    required = state["constraints.min_safety_factor"]
    if report.minimum < required:
        print(f"\n  All margins above 1.0, but the minimum "
              f"({report.minimum:.2f}) is below the required {required:.2f}.")
    return 0


def cmd_levers(args) -> int:
    from .sensitivity import OBJECTIVES, rank_levers

    state, _ = load_state(args.state)
    _apply_rpm(state, args.rpm)
    for path in args.lock or []:
        if state.has(path):
            state.lock(path, "locked on the command line")

    if args.objective not in OBJECTIVES and "." not in args.objective:
        print(f"Unknown objective {args.objective!r}. Try one of:")
        for key, (path, sense, unit) in sorted(OBJECTIVES.items()):
            print(f"  {key:20s} {sense:9s} {path}")
        return 2

    result = rank_levers(state, args.objective)
    sense = result["sense"]
    print(f"\n{state.meta.get('name', 'unnamed')}: what moves "
          f"{args.objective} ({sense})")
    print(f"{BAR}")
    print(f"  baseline {u.fmt(result['baseline'], result['unit'])} at "
          f"{u.rad_s_to_rpm(state['operating.speed']):.0f} rpm\n")

    if not result["levers"]:
        print("  Nothing in the design-lever set moves this objective.")
        for note in result["notes"]:
            print(f"  - {note}")
        return 0

    print(f"  {'lever':<32} {'move':>8} {'gain':>9}   stopped by")
    for lever in result["levers"][:12]:
        gain = lever["reachable_gain_percent"]
        print(f"  {lever['path']:<32} {lever['direction']:>8} "
              f"{gain:+8.2f}%   {lever['limited_by']}")
        print(f"      {lever['limit_reason'][:88]}")
        for effect in lever["side_effects"][:3]:
            if "percent" in effect:
                mark = "worse" if effect["worse"] else "better"
                print(f"        {effect['metric']}: "
                      f"{effect['percent']:+.1f}% ({mark})")
            elif "to" in effect:
                print(f"        min safety factor {effect['from']:.2f} -> "
                      f"{effect['to']:.2f} ({effect.get('binding') or ''})")

    _hdr("NOTES")
    for note in result["notes"]:
        print(f"  - {note}")
    return 0


def cmd_serve(args) -> int:
    from .server import serve

    print(f"\nPiston System Research Tool")
    print(f"  http://{args.host}:{args.port}/")
    print(f"  design: {args.state}")
    print(f"  Ctrl-C to stop\n")
    serve(args.state, host=args.host, port=args.port,
          open_browser=not args.no_browser)
    return 0


def cmd_geometry(args) -> int:
    state, _ = load_state(args.state)
    from . import geometry as geo

    if args.profiles:
        return _profile_study(state, geo)

    print(f"\n{state.meta.get('name', 'unnamed')} geometry")
    parts = geo.build_all(state)

    _hdr("MASS PROPERTIES")
    print(f"  {'part':<10} {'volume':>12} {'mass':>10} {'material':<14} "
          f"{'centre of mass (mm)'}")
    for name, entry in parts.items():
        p = entry["properties"]
        com = ", ".join(f"{v * 1e3:7.2f}" for v in p.centre_of_mass)
        print(f"  {name:<10} {p.volume * 1e6:9.2f} cc {p.mass * 1e3:8.1f} g "
              f"{p.material:<14} {com}")

    measured = geo.masses_from_geometry(state)
    _hdr("RECIPROCATING MASS")
    _row("rod centre of mass", f"{measured['rod_cg_from_small_end_m'] * 1e3:.2f} mm",
         f"= {measured['rod_cg_fraction_of_length']:.3f} of rod length")
    _row("rod reciprocating share",
         f"{(1 - measured['rod_cg_fraction_of_length']) * 100:.1f} %",
         "measured, not the one-third rule")
    _row("rod hardware (not modelled)",
         u.fmt(measured["rod_hardware_kg"], "kg"), "bolts, shells, at the big end")

    if args.check:
        check = geo.check_masses(state, tolerance=args.tolerance)
        _hdr("EXACTNESS CHECK")
        print(f"  {'quantity':<24} {'in the state':>14} {'from geometry':>14} "
              f"{'error':>8}")
        for row in check["rows"]:
            flag = "" if row["within_tolerance"] else "   DRIFT"
            print(f"  {row['quantity']:<24} {row['specified']:14.5f} "
                  f"{row['from_geometry']:14.5f} "
                  f"{row['relative_error'] * 100:+7.2f}%{flag}")
        print(f"  {'reciprocating mass':<24} "
              f"{check['reciprocating_specified_kg']:14.5f} "
              f"{check['reciprocating_from_geometry_kg']:14.5f} "
              f"{check['reciprocating_relative_error'] * 100:+7.2f}%")
        print(f"\n  {'PASS' if check['passes'] else 'DRIFT'} at a "
              f"{args.tolerance * 100:.0f}% tolerance")
        for note in check["notes"]:
            print(f"  - {note}")
        if not check["passes"]:
            print("\n  Run with --adopt to take the geometry's word for it.")

    if args.adopt:
        adopted = geo.adopt_masses(state)
        adopted.save(args.state)
        _hdr("ADOPTED")
        for change in adopted.log[-4:]:
            print(f"  {change.path:<34} {change.old:.5f} -> {change.new:.5f}")
        print(f"\n  Written back to {args.state}")

    if args.export:
        written = geo.export(state, args.export, fmt=args.format)
        _hdr("EXPORTED")
        for path in written:
            print(f"  {path}")
        print(f"\n  STEP opens in any CAD package; hand it to a machinist.")
    return 0


def _profile_study(state, geo) -> int:
    study = geo.profile_study(state)
    print(f"\n{state.meta.get('name', 'unnamed')}: what each bore "
          f"cross-section could displace")
    print(f"{BAR}")
    print(f"  Envelope radius {study['envelope_radius_m'] * 1e3:.2f} mm "
          f"(the most this block allows), current bore "
          f"{study['current_bore_m'] * 1e3:.2f} mm\n")
    print(f"  {'profile':<22} {'area':>10} {'vs circle':>10} "
          f"{'equiv bore':>11} {'displacement':>13}  seals?")
    for row in study["rows"]:
        print(f"  {row['name']:<22} {row['area_m2'] * 1e6:8.1f} mm2 "
              f"{row['fraction_of_circle'] * 100:9.1f}% "
              f"{row['equivalent_diameter_m'] * 1e3:9.2f} mm "
              f"{row['displacement_total_m3'] * 1e6:10.0f} cc  "
              f"{'yes' if row['sealable'] else 'NO'}")

    circle = next(r for r in study["rows"] if r["key"] == "circle")
    hexagon = next((r for r in study["rows"] if r["key"] == "polygon-6"), None)
    print(f"\n{BAR}")
    if hexagon:
        loss = (1.0 - hexagon["fraction_of_circle"]) * 100.0
        print(f"  A hexagonal bore inside the same envelope displaces "
              f"{loss:.1f}% LESS than\n  the circle, not more: "
              f"{hexagon['area_m2'] * 1e6:.0f} mm2 against "
              f"{circle['area_m2'] * 1e6:.0f} mm2. In bore terms that is "
              f"{hexagon['equivalent_diameter_m'] * 1e3:.1f} mm\n"
              f"  against {circle['equivalent_diameter_m'] * 1e3:.1f} mm.")
        print(f"\n  {hexagon['note']}")
    print(f"\n  A circle encloses the most area for a given maximum width, and "
          f"the\n  envelope IS a maximum width. Any other shape inside it "
          f"displaces less;\n  to gain area its corners have to push past the "
          f"envelope, into exactly\n  the material the block does not have.")
    return 0


def cmd_optimise(args) -> int:
    """Search toward a target without breaking anything."""
    state, _ = load_state(args.state)
    _apply_rpm(state, args.rpm)
    from .optimise import DEFAULT_CLASSES, LEVER_CLASSES, optimise

    classes = tuple(args.classes.split(",")) if args.classes \
        else DEFAULT_CLASSES
    unknown = [c for c in classes if c not in LEVER_CLASSES]
    if unknown:
        print(f"Error: unknown lever class(es) {', '.join(unknown)}; "
              f"available: {', '.join(LEVER_CLASSES)}", file=sys.stderr)
        return 2

    print(f"\nsearching ({', '.join(classes)})...")
    result = optimise(state, args.objective, classes=classes,
                      max_levers=args.levers,
                      min_safety_factor=args.min_sf)

    print(f"\n{state.meta.get('name', 'unnamed')}: {args.objective}\n")
    print(result.summary())

    if result.worse:
        _hdr("WHAT GOT WORSE")
        for row in result.worse:
            print(f"  {row['label']:30s} {row['before']:12.5g} -> "
                  f"{row['after']:12.5g}  {row['percent']:+7.2f}%")

    print(f"\n  ({result.evaluations} evaluations)")
    if args.save:
        result.state.save(args.save)
        print(f"  written to {args.save}")
    print()
    return 0


def cmd_frontier(args) -> int:
    """What the next unit of safety factor costs in objective."""
    state, _ = load_state(args.state)
    _apply_rpm(state, args.rpm)
    from .optimise import frontier

    print(f"\nsearching {args.points} points...")
    front = frontier(state, args.objective, points=args.points,
                     max_levers=args.levers)
    print(f"\n{state.meta.get('name', 'unnamed')}\n")
    print(front.summary())
    print()
    return 0


def cmd_envelope(args) -> int:
    """What the block will allow, and how much of that is actually known."""
    state, _ = load_state(args.state)
    from .envelope import envelope

    env = envelope(state)
    print(f"\n{state.meta.get('name', 'unnamed')} block envelope")
    _hdr("LIMITS ON THE BORE")
    print(f"  {'max bore':>10}  {'tier':>4}  {'limit':<32} why")
    for limit in env.limits:
        shown = (f"{limit.max_bore * 1000:9.2f} " if limit.known
                 else "        - ")
        print(f"  {shown}  {limit.tier:>4}  {limit.name:<32} {limit.reason}")

    _hdr("VERDICT")
    _row("as built", f"{env.as_built * 1000:.3f} mm")
    _row("current", f"{env.current * 1000:.3f} mm")
    if env.max_bore is None:
        _row("maximum", "unknown -- no limit could be evaluated")
    else:
        _row("maximum", f"{env.max_bore * 1000:.3f} mm",
             f"{env.overbore_available * 1000:+.3f} mm over as-built")
        _row("headroom left", f"{env.headroom * 1000:.3f} mm")
        _row("set by", f"{env.binding.name} (tier {env.binding.tier})")
        _row("confidence", f"tier {env.confidence}: "
                           f"{env.binding.provenance}")

    if env.notes:
        _hdr("READ THIS BEFORE CUTTING")
        for note in env.notes:
            print(f"  - {note}")
    print()
    return 0


def cmd_overbore(args) -> int:
    """Everything that follows from a bore change, not just the torque."""
    state, _ = load_state(args.state)
    from .cascade import overbore_cascade
    from .state import ConstraintViolation

    try:
        cascade = overbore_cascade(state, args.bore_mm / 1000.0,
                                   remeasure=not args.fast)
    except ConstraintViolation as exc:
        print(f"\nRefused: {exc}\n", file=sys.stderr)
        return 2

    print(f"\n{state.meta.get('name', 'unnamed')}: overbore consequences")
    print(f"\n{cascade.summary()}\n")
    if cascade.newly_failing:
        print("  COMPONENTS THAT NOW FAIL: "
              + ", ".join(cascade.newly_failing) + "\n")
    return 0


def cmd_fea(args) -> int:
    """Run a component load case and report it against the fast layer."""
    state, _ = load_state(args.state)
    from .fea.cases import CASES

    if args.part not in CASES:
        print(f"Error: no load case for {args.part!r} yet; "
              f"available: {', '.join(sorted(CASES))}", file=sys.stderr)
        return 2

    print(f"\nmeshing and solving the {args.part}...")
    case = CASES[args.part](state, target_elements=args.elements)
    solve = case.solve
    mesh = solve.mesh

    print(f"\n{state.meta.get('name', 'unnamed')}: {args.part}")
    print(f"  mesh          {mesh.n_elements:,} elements, "
          f"{mesh.n_nodes:,} nodes")
    print(f"  load          {case.force / 1e3:.1f} kN at {case.condition}")
    print(f"  equilibrium   residual {solve.residual:.2e}")
    if solve.free_rigid_modes:
        print(f"  UNRESTRAINED  {', '.join(solve.free_rigid_modes)}")

    sampled = (int(case.sample.sum()) if case.sample is not None
               else mesh.n_elements)
    print(f"\n  peak              {solve.peak_stress / 1e6:9.1f} MPa"
          f"   whole part")
    print(f"  compared          {case.sampled_stress / 1e6:9.1f} MPa"
          f"   {case.sample_region} ({sampled:,} elements)")
    print(f"  analytical        {case.analytical_stress / 1e6:9.1f} MPa"
          f"   {case.analytical_equation}")
    print(f"  FEA / analytical  {case.ratio:9.3f}")

    print("\n  The peak sits under an applied restraint and is a boundary")
    print("  condition, not a material stress. The compared figure is taken")
    print("  where the analytical formula actually applies -- but it is still")
    print("  not a stress concentration factor: it carries this case's contact")
    print("  idealisation with it.")
    for note in solve.notes:
        print(f"  - {note}")
    print()
    return 0


def cmd_materials(args) -> int:
    print(f"\nMaterial database\n{BAR}")
    for mat in materials_mod.listing():
        y = ("-" if mat.yield_strength is None
             else f"{mat.yield_strength / 1e6:.0f}")
        print(f"  {mat.key:<16} rho {mat.density:>5.0f} kg/m3   "
              f"E {mat.youngs_modulus / 1e9:>5.0f} GPa   "
              f"yield {y:>5} MPa   UTS {mat.ultimate_strength / 1e6:>5.0f} MPa")
        if args.verbose:
            print(f"      {mat.name}")
            print(f"      {mat.source}")
            if mat.derating:
                pts = "  ".join(
                    f"{u.k_to_c(t):.0f}C:{f * 100:.0f}%" for t, f in mat.derating)
                print(f"      strength retained: {pts}")
    print(f"\n  All values are nominal handbook figures. Replace them with "
          f"your\n  material certificate before they inform a part you intend "
          f"to make.")
    return 0


def cmd_validate(args) -> int:
    state, _ = load_state(args.state)
    issues = state.validate()
    if issues:
        print("Design state has problems:")
        for i in issues:
            print(f"  {i}")
        return 1
    m = evaluate(state)
    ok = m.checks["energy_closure_pass"]
    print(f"State valid. Energy closure "
          f"{m.checks['energy_closure_error'] * 100:.5f} % "
          f"({'PASS' if ok else 'FAIL'})")
    for w in m.warnings:
        print(f"  ! {w}")
    return 0 if ok else 1


# --- wiring ----------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="psrt",
        description="Piston system research tool. Phase 1: the load chain.")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="show descriptions, bounds and reasons")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("new", help="write a default design state")
    s.add_argument("out")
    s.add_argument("--name", default="untitled")
    s.set_defaults(func=cmd_new)

    s = sub.add_parser("show", help="print the design state")
    s.add_argument("state")
    s.set_defaults(func=cmd_show)

    s = sub.add_parser("loads", help="load environment at one operating point")
    s.add_argument("state")
    s.add_argument("--rpm", type=float, default=None)
    s.add_argument("--angles", type=float, nargs="*", default=None,
                   metavar="DEG", help="also report these crank angles")
    s.set_defaults(func=cmd_loads)

    s = sub.add_parser("curve", help="torque and peak loads against rpm")
    s.add_argument("state")
    s.add_argument("--start", type=int, default=1000)
    s.add_argument("--stop", type=int, default=7000)
    s.add_argument("--step", type=int, default=500)
    s.set_defaults(func=cmd_curve)

    s = sub.add_parser("export", help="write the crank-angle sweep as CSV")
    s.add_argument("state")
    s.add_argument("--out", required=True)
    s.add_argument("--rpm", type=float, default=None)
    s.set_defaults(func=cmd_export)

    s = sub.add_parser("compare", help="deltas between two design states")
    s.add_argument("before")
    s.add_argument("after")
    s.set_defaults(func=cmd_compare)

    s = sub.add_parser("margins", help="structural margins for every component")
    s.add_argument("state")
    s.add_argument("--rpm", type=float, default=None)
    s.add_argument("--explain", default=None, metavar="COMPONENT",
                   help="show the equation, inputs and caveats behind every "
                        "margin matching this component or mode")
    s.set_defaults(func=cmd_margins)

    s = sub.add_parser("levers", help="what can I change, and how far")
    s.add_argument("state")
    s.add_argument("--objective", default="torque",
                   help="torque, power, reciprocating_mass, safety_factor, ...")
    s.add_argument("--rpm", type=float, default=None)
    s.add_argument("--lock", action="append", metavar="PATH",
                   help="treat this parameter as fixed; repeatable")
    s.set_defaults(func=cmd_levers)

    s = sub.add_parser("serve", help="start the browser front end")
    s.add_argument("state")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--no-browser", action="store_true")
    s.set_defaults(func=cmd_serve)

    s = sub.add_parser("geometry", help="build the solids and measure them")
    s.add_argument("state")
    s.add_argument("--check", action="store_true",
                   help="compare the state's masses against the solids")
    s.add_argument("--adopt", action="store_true",
                   help="write the geometry's masses back into the state file")
    s.add_argument("--export", metavar="DIR", default=None,
                   help="write every part to this directory")
    s.add_argument("--format", default="step", choices=["step", "stl"])
    s.add_argument("--profiles", action="store_true",
                   help="compare what each bore cross-section could displace")
    s.add_argument("--tolerance", type=float, default=0.03)
    s.set_defaults(func=cmd_geometry)

    s = sub.add_parser("optimise", help="search toward a target without "
                                        "breaking anything")
    s.add_argument("state")
    s.add_argument("objective", nargs="?", default="torque")
    s.add_argument("--rpm", type=float)
    s.add_argument("--levers", type=int, default=8,
                   help="how many levers to search (default 8)")
    s.add_argument("--classes", default=None,
                   help="comma-separated lever classes: machining, piston, "
                        "rod, rotating, tuning. Default is machining,piston,"
                        "rod -- this block with remachined parts")
    s.add_argument("--min-sf", dest="min_sf", type=float, default=None,
                   help="safety factor floor for the search")
    s.add_argument("--save", default=None,
                   help="write the optimised design to this file")
    s.set_defaults(func=cmd_optimise)

    s = sub.add_parser("frontier", help="what the next unit of safety factor "
                                        "costs in objective")
    s.add_argument("state")
    s.add_argument("objective", nargs="?", default="torque")
    s.add_argument("--rpm", type=float)
    s.add_argument("--points", type=int, default=6)
    s.add_argument("--levers", type=int, default=6)
    s.set_defaults(func=cmd_frontier)

    s = sub.add_parser("envelope", help="what the block will allow, and how "
                                       "much of that is actually known")
    s.add_argument("state")
    s.set_defaults(func=cmd_envelope)

    s = sub.add_parser("overbore", help="everything that follows from a bore "
                                        "change, not just the torque")
    s.add_argument("state")
    s.add_argument("bore_mm", type=float)
    s.add_argument("--fast", action="store_true",
                   help="skip re-measuring the masses from geometry "
                        "(quicker, and the piston mass will be stale)")
    s.set_defaults(func=cmd_overbore)

    s = sub.add_parser("fea", help="solve one component and compare it to "
                                   "the analytical model")
    s.add_argument("state")
    s.add_argument("part", nargs="?", default="pin")
    s.add_argument("--elements", type=int, default=25_000,
                   help="target element count (default 25000)")
    s.set_defaults(func=cmd_fea)

    s = sub.add_parser("materials", help="list the material database")
    s.set_defaults(func=cmd_materials)

    s = sub.add_parser("validate", help="check a state and its energy closure")
    s.add_argument("state")
    s.set_defaults(func=cmd_validate)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ConstraintViolation as exc:
        print(f"Refused: {exc}", file=sys.stderr)
        return 2
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
