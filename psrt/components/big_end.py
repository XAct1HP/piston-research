"""Big-end bearing: specific pressure, and a lower bound on oil film.

Specific pressure over the projected area is the practical design criterion
and the one to trust::

    p = F_max / (d_bore * L_bearing)

Minimum oil film thickness is computed from the Ocvirk short-bearing solution::

    W = (mu omega R L^3 eps) / (4 c^2 (1 - eps^2)^2)
        * sqrt(pi^2 (1 - eps^2) + 16 eps^2)

solved for the eccentricity ratio by bisection, giving h_min = c (1 - eps).

Read that number as a LOWER BOUND and nothing more. It is a steady-load
solution, and a rod bearing is nothing like steady: the load rotates relative
to the shell and reverses twice per cycle, and the squeeze-film effect that
produces carries several times more load than this predicts. A real answer
needs a dynamically loaded bearing solution over the whole cycle. What the
number is good for is comparison -- if a change halves it, that is real.
"""

from __future__ import annotations

import math

import numpy as np

from ..margins import ComponentContext, Margin

MIN_FILM_ALLOWABLE = 1.0e-6     # ~1 micron, composite surface roughness


def _ocvirk_load(eps: float, viscosity: float, omega: float, radius: float,
                 length: float, clearance: float) -> float:
    if eps <= 0.0:
        return 0.0
    denom = 4.0 * clearance ** 2 * (1.0 - eps ** 2) ** 2
    shape = math.sqrt(math.pi ** 2 * (1.0 - eps ** 2) + 16.0 * eps ** 2)
    return viscosity * omega * radius * length ** 3 * eps * shape / denom


def _solve_eccentricity(load: float, **kwargs) -> float:
    """Bisect for the eccentricity ratio that carries this load."""
    lo, hi = 1.0e-6, 0.999999
    if _ocvirk_load(hi, **kwargs) < load:
        return hi
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _ocvirk_load(mid, **kwargs) < load:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def evaluate(ctx: ComponentContext) -> list:
    state, sweep = ctx.state, ctx.sweep

    bore = state["rod.big_end_bore"]
    width = state["rod.big_end_width"]
    projected = bore * width

    index = int(np.argmax(np.abs(sweep.f_rod)))
    peak_force = abs(float(sweep.f_rod[index]))
    angle = float(np.degrees(sweep.theta[index]))
    pressure = peak_force / projected

    clearance = state["rod.bearing_clearance"]
    geometry = {
        "viscosity": state["rod.oil_viscosity"],
        "omega": sweep.speed,
        "radius": bore / 2.0,
        "length": width,
        "clearance": clearance,
    }
    eccentricity = _solve_eccentricity(peak_force, **geometry)
    film = clearance * (1.0 - eccentricity)

    return [
        Margin(
            component="big end", mode="projected specific pressure",
            applied=pressure,
            allowable=state["rod.allowable_bearing_pressure"], unit="Pa",
            equation="p = F_max / (d_bore * L_bearing)",
            reference="projected bearing pressure; allowable is trimetal "
                      "shell operating practice",
            condition=ctx.at(f"peak rod force, {angle:.0f} deg"),
            inputs={
                "peak_rod_force_n": peak_force,
                "projected_area_m2": projected,
                "big_end_bore_m": bore,
                "big_end_width_m": width,
                "length_to_diameter_ratio": width / bore,
            },
            notes=["this is the practical design criterion for a rod bearing "
                   "and the number to act on"],
        ),
        Margin(
            component="big end", mode="minimum oil film thickness",
            applied=MIN_FILM_ALLOWABLE, allowable=film, unit="m",
            equation="Ocvirk short bearing, solved for eccentricity ratio; "
                     "h_min = c (1 - eps)",
            reference="Ocvirk short-bearing approximation, steady load",
            condition=ctx.at(f"peak rod force, {angle:.0f} deg"),
            inputs={
                "peak_rod_force_n": peak_force,
                "eccentricity_ratio": eccentricity,
                "radial_clearance_m": clearance,
                "minimum_film_m": film,
                "viscosity_pa_s": state["rod.oil_viscosity"],
                "allowable_film_m": MIN_FILM_ALLOWABLE,
            },
            notes=["LOWER BOUND ONLY: this is a steady-load solution and a rod "
                   "bearing is dynamically loaded. Squeeze-film effects carry "
                   "several times more load than this predicts",
                   "useful for comparing designs, not for predicting failure"],
        ),
    ]
