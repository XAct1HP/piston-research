"""Rail connecting rod: the fast analytical checks.

Runs in place of the conventional rod and rod-bolt models when
``railrod.enabled`` is set. Two kinds of number come out of here, and the
margins say which is which:

* Statically determinate ones -- the rail's axial force is the rod force,
  wherever the contacts put it, so rail compression, buckling and the notch's
  net section are as trustworthy as the load sweep.
* Load-sharing ones -- how much preload reaches the notch, whether the rail
  feet stay seated, what the bolt sees. These come from the contact network.
  When the flexibility-coupled network has been run for this design (by the
  FEA, or ``python -m psrt railrod --coupled``) its answer is used; otherwise
  the rigid network's, and the margin says so, because on this concept the
  rigid network is optimistic about exactly these things.

Buckling uses the built-up column formula for two chords joined by a shear
web, because that is what two rails and a stabilising sleeve are::

    P_cr = P_e / (1 + P_e / S),   P_e = pi^2 E I / (K L)^2,   S = G_s A_web

(Timoshenko & Gere, *Theory of Elastic Stability*, 2.18). Without the sleeve
S goes to zero and each rail buckles on its own; with an infinitely stiff one
the pair acts as a single section with the rails' parallel-axis inertia.
"""

from __future__ import annotations

import math

import numpy as np

from .. import materials as materials_mod
from ..margins import ComponentContext, Margin, fatigue_margin
from .rod import critical_stress


def _analysis(ctx):
    from ..railrod.analysis import analyse
    from ..railrod import coupled

    base = analyse(ctx.state, ctx.sweep)
    key_prefix = ctx.state.fingerprint()
    for key, value in coupled._CACHE.items():
        if key[0] == key_prefix:
            return value, True
    return base, False


def _peterson_q(material, radius_m: float) -> float:
    """Notch sensitivity, Peterson: q = 1 / (1 + a / r)."""
    s_ut_mpa = material.ultimate_strength / 1e6
    a_mm = 0.0254 * (2070.0 / max(s_ut_mpa, 100.0)) ** 1.8
    return 1.0 / (1.0 + a_mm / (radius_m * 1e3))


def notch_kt(depth: float, radius: float, width: float) -> float:
    """Stress concentration at the notch root, net-section basis.

    A shallow U-notch on one edge of a bar in tension: Inglis gives
    1 + 2 sqrt(d / r) for a notch in a semi-infinite plate, and a finite
    remaining ligament relieves it. The relief used here is Peterson's
    ligament form, blended so Kt -> 1 as the notch vanishes. It is a
    closed-form estimate for a SHARP-ENOUGH radius; the FEA of the rails part
    is what to believe at the actual geometry.
    """
    if depth <= 0.0:
        return 1.0
    inglis = 1.0 + 2.0 * math.sqrt(depth / radius)
    ligament = max(width - depth, 1e-9)
    relief = 1.0 / (1.0 + 0.5 * depth / ligament)
    return max(1.0, 1.0 + (inglis - 1.0) * relief)


def evaluate(ctx: ComponentContext) -> list:
    state, sweep, thermal = ctx.state, ctx.sweep, ctx.thermal
    # A LayoutError (a ValueError) propagates on purpose: every write path
    # refuses a change that evaluates to an unbuildable rod, with the reason.
    analysis, coupled = _analysis(ctx)

    lay = analysis.layout
    cyc = analysis.cycle
    n = lay.notch
    source = ("flexibility-coupled contact network" if coupled else
              "RIGID contact network -- run the coupled analysis before "
              "trusting any load-sharing number")

    mat_rail = ctx.material("rod")
    temp = thermal.rod
    s_y, basis = ctx.strength(mat_rail, temp)
    E = mat_rail.youngs_modulus

    b = (lay.x_o - lay.x_i) * 1e-3
    h = lay.rail_depth * 1e-3
    d = n.depth * 1e-3
    area = b * h
    area_net = (b - d) * h

    f_comp = float(np.max(sweep.f_rod))
    f_tens = float(max(-np.min(sweep.f_rod), 0.0))
    per_rail_c = f_comp / 2.0
    per_rail_t = f_tens / 2.0
    margins = []

    # --- rail axial --------------------------------------------------------
    margins.append(Margin(
        component="rails", mode="rail compression at peak firing",
        applied=per_rail_c / area, allowable=s_y, unit="Pa",
        equation="sigma = (F_rod / 2) / (b h)",
        reference="direct compression, each rail carries half the rod force",
        condition=ctx.at(f"peak firing, {sweep.peak_rod_compression.angle_deg:.0f} deg"),
        inputs={"rod_force_n": f_comp, "rail_width_m": b, "rail_depth_m": h,
                "rail_area_m2": area, "allowable_basis": basis}))

    # --- buckling, built-up column ---------------------------------------
    sleeve_mat = materials_mod.get(state["railrod.sleeve_material"])
    G_s = sleeve_mat.youngs_modulus / (2.0 * (1.0 + sleeve_mat.poisson))
    L = state["engine.rod_length"]
    s_half = (lay.x_o + lay.x_i) / 2.0 * 1e-3              # centreline, m
    sleeve_len = (lay.sleeve_v1 - lay.sleeve_v0) * 1e-3
    web_depth = 2.0 * lay.sleeve_y * 1e-3
    web_width = 2.0 * (lay.x_i - state["railrod.sleeve_clearance"] * 1e3) \
        * 1e-3
    A_web = max(web_depth * web_width, 0.0)
    I_web_ip = web_depth * web_width ** 3 / 12.0
    I_web_op = web_width * web_depth ** 3 / 12.0
    E_s = sleeve_mat.youngs_modulus

    # The web is not one material. Down its middle runs the lattice core; the
    # rail skins and the face shells around it are solid. Add the two as a
    # composite section rather than pretending either one is the whole thing.
    lattice_inputs = {}
    if lay.sleeve_core_kind == "sheet-gyroid" and lay.lattice_core:
        from ..railrod import lattice as lattice_mod
        xc, yc, _, _ = lay.lattice_core
        core_w, core_d = 2.0 * xc * 1e-3, 2.0 * yc * 1e-3
        A_core = core_w * core_d
        I_core_ip = core_d * core_w ** 3 / 12.0
        I_core_op = core_w * core_d ** 3 / 12.0
        c_E = state["railrod.lattice_modulus_coefficient"]
        n_E = state["railrod.lattice_modulus_exponent"]
        # Shear gets the energy-weighted average along the grading, because
        # shear is what this web is for and it is not uniform along it.
        # Bending gets the mid-length density, because that is where the
        # moment the sleeve helps carry actually peaks.
        G_lat = lattice_mod.effective_shear(
            sleeve_mat, lay.lattice_rho_mid, lay.lattice_rho_end,
            lay.lattice_exponent, c_E, n_E,
            span=((lay.L - lay.sleeve_v1) * 1e-3,
                  (lay.L - lay.sleeve_v0) * 1e-3), rod_length=L)
        E_lat = lattice_mod.homogenised(
            sleeve_mat, lay.lattice_rho_mid, c_E, n_E,
            state["railrod.lattice_strength_coefficient"],
            state["railrod.lattice_strength_exponent"])["youngs_modulus"]
        GA_web = G_s * max(A_web - A_core, 0.0) + G_lat * A_core
        EI_ip = E_s * max(I_web_ip - I_core_ip, 0.0) + E_lat * I_core_ip
        EI_op = E_s * max(I_web_op - I_core_op, 0.0) + E_lat * I_core_op
        lattice_inputs = {
            "lattice_relative_density_mid": lay.lattice_rho_mid,
            "lattice_relative_density_end": lay.lattice_rho_end,
            "lattice_shear_modulus_pa": G_lat,
            "lattice_youngs_modulus_pa": E_lat,
            "solid_shear_modulus_pa": G_s,
            "core_area_fraction": (A_core / A_web) if A_web else 0.0,
        }
    else:
        GA_web = G_s * A_web
        EI_ip = E_s * I_web_ip
        EI_op = E_s * I_web_op

    S_shear = GA_web / 1.2 * (sleeve_len / L)
    # Equivalent second moments, already converted into rail-steel terms, so
    # the modular ratio below is 1 and the composite is not double counted.
    I_sleeve_ip = EI_ip / E
    I_sleeve_op = EI_op / E
    e_ratio = 1.0

    for axis, I_rails, I_sl, k_path, builtup in (
            ("in the plane of rotation",
             2.0 * (h * b ** 3 / 12.0 + area * s_half ** 2),
             I_sleeve_ip, "rod.k_in_plane", True),
            ("out of the plane of rotation",
             2.0 * (b * h ** 3 / 12.0), I_sleeve_op, "rod.k_out_of_plane",
             False)):
        K = state[k_path]
        I_eff = I_rails + e_ratio * I_sl * (sleeve_len / L)
        P_e = math.pi ** 2 * E * I_eff / (K * L) ** 2
        if builtup:
            P_cr = P_e / (1.0 + P_e / max(S_shear, 1e-9))
            P_single = 2.0 * math.pi ** 2 * E * (h * b ** 3 / 12.0) / (K * L) ** 2
            P_cr = max(P_cr, P_single)
        else:
            P_cr = P_e
        sigma_e = P_cr / (2.0 * area)
        # inelastic: Johnson through the equivalent slenderness
        lam = math.pi * math.sqrt(E / max(sigma_e, 1.0))
        sigma_cr, regime = critical_stress(lam, s_y, E)
        margins.append(Margin(
            component="rails", mode=f"buckling {axis}",
            applied=per_rail_c / area, allowable=sigma_cr, unit="Pa",
            equation=("P_cr = P_e / (1 + P_e / S), built-up column"
                      if builtup else "P_cr = pi^2 E I / (K L)^2"),
            reference=("Timoshenko & Gere 2.18: chords joined by a shear web "
                       "(the sleeve)" if builtup else
                       "Euler/Johnson on the two rails together"),
            condition=ctx.at("peak firing"),
            inputs={"effective_length_factor": K, "rod_length_m": L,
                    "rails_second_moment_m4": I_rails,
                    "sleeve_equivalent_second_moment_m4": I_sl,
                    "sleeve_shear_stiffness_n": S_shear if builtup else None,
                    "euler_load_n": P_e, "critical_load_n": P_cr,
                    "equivalent_slenderness": lam, "regime": regime,
                    "sleeve_material": sleeve_mat.key,
                    "sleeve_core": lay.sleeve_core_kind, **lattice_inputs},
            notes=["the sleeve is credited with its shear stiffness (in "
                   "plane) and its own bending stiffness, both scaled by the "
                   "fraction of the rod it covers",
                   "out of plane the rails sit side by side with no spacing, "
                   "so there is no built-up action to gain -- their depth is "
                   "what resists"] if builtup else
                  ["out-of-plane buckling rests on the rail DEPTH; widening "
                   "the rail spacing does nothing for it"]))

    # --- notch net section ------------------------------------------------
    kt = notch_kt(d, n.r_top * 1e-3, b)
    q = _peterson_q(mat_rail, n.r_top * 1e-3)
    kf = 1.0 + q * (kt - 1.0)
    ecc = d / 2.0                                  # net-section centroid shift
    z_net = h * (b - d) ** 2 / 6.0

    def net_stress(force):
        return force / area_net + force * ecc / z_net

    sig_t = net_stress(per_rail_t)
    sig_c = net_stress(per_rail_c)
    margins.append(Margin(
        component="rail notch",
        mode="net section at the notch, peak firing (compression passes "
             "through it to the flat foot)",
        applied=sig_c, allowable=s_y, unit="Pa",
        equation="sigma = F/A_net + F e / Z_net,  e = d/2",
        reference="axial plus the eccentricity of the notched section",
        condition=ctx.at("peak firing"),
        inputs={"force_per_rail_n": per_rail_c, "net_area_m2": area_net,
                "eccentricity_m": ecc, "notch_depth_m": d},
        notes=["the notch is ABOVE the foot, so all of the firing load "
               "crosses the notched section on its way to the floor. That is "
               "the concept as specified; it makes the notch a compression "
               "detail as well as a tensile one",
               "static check without Kt; the fatigue check below applies it"]))

    alt = kf * (sig_c + sig_t) / 2.0
    mean = kf * (sig_t - sig_c) / 2.0
    margins.append(fatigue_margin(
        ctx, "rail notch", "notch-root fatigue over the full cycle",
        alternating=alt, mean=mean, material=mat_rail, temperature=temp,
        diameter=math.sqrt(4.0 * area_net / math.pi),
        finish=state["rod.surface_finish"], load="axial",
        condition=ctx.at("firing compression to overlap tension"),
        extra_inputs={"kt": kt, "notch_sensitivity_q": q, "kf": kf,
                      "notch_top_radius_m": n.r_top * 1e-3,
                      "net_stress_tension_pa": sig_t,
                      "net_stress_compression_pa": -sig_c},
        notes=["Kt is a closed-form estimate for a radiused edge notch; the "
               "rails FEA gives the real concentration"]))

    # --- contact and joint (load sharing) ---------------------------------
    flank_area = n.ramp_length * 1e-3 * h
    flank_peak = float(np.max(cyc.flank))
    if flank_area > 0 and flank_peak > 0:
        margins.append(Margin(
            component="rail notch", mode="ramp bearing pressure",
            applied=flank_peak / flank_area, allowable=s_y, unit="Pa",
            equation="p = N_flank / (ramp length x rail depth)",
            reference="projected bearing pressure against yield",
            condition=ctx.at("worst angle in the cycle"),
            inputs={"flank_force_n": flank_peak, "ramp_length_m":
                    n.ramp_length * 1e-3, "source": source},
            notes=[f"from the {source}"]))

    pre = cyc.preload_state
    sep = analysis.separation_factor
    margins.append(Margin(
        component="rail joint", mode="rail feet stay seated at overlap TDC",
        applied=min(sep, 10.0) if math.isfinite(sep) else 10.0,
        allowable=1.0, kind="factor", unit="-",
        equation="tension multiple at which the floor contact opens",
        reference="bisection on the contact network",
        condition=ctx.at("peak tension"),
        inputs={"assembly_seat_preload_n": pre["floor"],
                "source": source},
        notes=[f"from the {source}",
               "below 1 the rails lift off the receiver floor every "
               "revolution and slam back at firing: fretting and impact at "
               "the foot and the knob"]))

    bolt = cyc.bolt
    d_b = state["railrod.bolt_diameter"]
    pitch = state["railrod.bolt_pitch"]
    a_s = math.pi / 4.0 * (d_b - 0.9382 * pitch) ** 2
    bolt_mat = materials_mod.get(state["railrod.bolt_material"])
    sigma_a = (bolt.max() - bolt.min()) / 2.0 / a_s
    sigma_m = (bolt.max() + bolt.min()) / 2.0 / a_s
    kf_bolt = state["fatigue.kf_bolt"]
    margins.append(fatigue_margin(
        ctx, "rail-rod bolt", "thread-root fatigue",
        alternating=kf_bolt * sigma_a, mean=sigma_m, material=bolt_mat,
        temperature=temp, diameter=d_b, finish=state["railrod.bolt_finish"],
        load="axial", condition=ctx.at("full cycle"),
        extra_inputs={"bolt_min_n": float(bolt.min()),
                      "bolt_max_n": float(bolt.max()), "stress_area_m2": a_s,
                      "kf_thread": kf_bolt, "source": source},
        notes=[f"bolt force history from the {source}",
               "the single bolt is tangential: the rod's tension reaches it "
               "only through the clamps' prying, and that amplification is "
               "what this checks"]))
    margins.append(Margin(
        component="rail-rod bolt", mode="peak load against proof",
        applied=float(bolt.max()), unit="N",
        allowable=state["railrod.bolt_proof_strength"] * a_s,
        equation="F_max <= S_proof A_t",
        reference="proof load of the fastener", condition=ctx.at("full cycle"),
        inputs={"bolt_max_n": float(bolt.max()), "preload_n":
                state["railrod.bolt_preload"], "source": source}))

    # --- assembly -----------------------------------------------------------
    swing = analysis.swing
    hook = swing["hook_in_angle_deg"]
    margins.append(Margin(
        component="swing clamps", mode="clamp can be hooked in and swung shut",
        applied=(swing["free_opening_deg"] / hook) if hook else 0.0,
        allowable=1.0, kind="factor", unit="-",
        equation="free swing angle / angle needed to hook the knob in",
        reference="swept-area interference check about the knob pivot",
        condition="assembly", operating_dependent=False,
        inputs=dict(swing)))

    # --- sleeve -------------------------------------------------------------
    s_temp = sleeve_mat.max_service_temp
    margins.append(Margin(
        component="stabilising sleeve", mode="core material temperature",
        applied=temp - 273.15, allowable=s_temp - 273.15, unit="C",
        kind="stress",
        equation="rod temperature <= material service limit (Celsius ratio)",
        reference="material database service temperature",
        condition=ctx.at("steady running"),
        inputs={"rod_temperature_c": temp - 273.15,
                "service_limit_c": s_temp - 273.15,
                "material": sleeve_mat.key}))

    if lay.lattice_print is not None:
        pr = lay.lattice_print
        margins.append(Margin(
            component="stabilising sleeve",
            mode="lattice sheet against the machine's minimum wall",
            applied=pr["min_wall_mm"] / max(pr["machine_min_wall_mm"], 1e-9),
            allowable=1.0, unit="-", kind="factor", operating_dependent=False,
            equation="thinnest place on the lightest sheet / min printable wall",
            reference=("gyroid wall thickness is 2c/|grad phi| and |grad phi| "
                       "runs from sqrt(2) k to sqrt(3) k over the cell, so "
                       "the thinnest place is 18% under the nominal sheet"),
            condition="as printed, at the lightest station of the grading",
            inputs=dict(pr),
            notes=["the binding station is wherever the graded density is "
                   "LOWEST, which with end-grading is mid-length, not the "
                   "average"]))
    if lay.lattice_evacuation is not None:
        ev = lay.lattice_evacuation
        margins.append(Margin(
            component="stabilising sleeve", mode="powder evacuation",
            applied=min(ev["aperture_mm"] / 0.5,
                        60.0 / max(ev["path_over_aperture"], 1e-9)),
            allowable=1.0, unit="-",
            kind="factor", operating_dependent=False,
            equation=("worse of: channel width / 0.5 mm, and 60 / (escape "
                      "path / channel width). Powder has to fit through the "
                      "gap AND be shakeable along the route"),
            reference=("a sealed cell of unfused powder is dead weight that "
                       "no inspection finds; aluminium stops flowing below "
                       "roughly ten particle diameters of gap"),
            condition=ev["route"],
            inputs=dict(ev),
            notes=[] if ev["aperture_ok"] else
                  [f"the {ev['aperture_mm']:.2f} mm channel is below the "
                   "0.5 mm powder will reliably flow through at all; the "
                   "path length is the lesser problem"]))
    if lay.lattice_cells is not None and not \
            lay.lattice_cells["homogenisation_valid"]:
        margins.append(Margin(
            component="stabilising sleeve",
            mode="cells across the core, for the effective properties to mean "
                 "anything",
            applied=min(lay.lattice_cells["cells_across"],
                        lay.lattice_cells["cells_through"]) / 4.0,
            allowable=1.0, unit="-", kind="factor",
            operating_dependent=False,
            equation="cells across the narrowest direction / 4",
            reference=("below about four cells a lattice behaves like the "
                       "few struts it is, not like the continuum its "
                       "effective properties describe"),
            condition="model validity, not a physical failure",
            inputs=dict(lay.lattice_cells)))
    return margins
