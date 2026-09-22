"""Fatigue: corrected endurance limit, mean-stress correction, finite life.

A piston system fails by fatigue, not by yielding. A connecting rod that sees
19 kN of tension once per revolution at 7000 rpm accumulates 10^8 cycles in
under 500 hours, so "the stress is below yield" says almost nothing useful.
What matters is the alternating stress, the mean stress it sits on, and how
many cycles the part has to survive.

The method is the standard Marin approach (Shigley, *Mechanical Engineering
Design*, ch. 6): take the rotary-beam endurance limit for the material and
knock it down for surface finish, size, loading mode, temperature and the
reliability you want, then apply a mean-stress correction and read a life off
the S-N line.

    S_e = k_a k_b k_c k_d k_e S_e'

Every factor is reported alongside the result, because a safety factor of 1.8
means one thing if k_a is 0.9 and another entirely if it is 0.45.

Stress concentration is handled separately, by a fatigue notch factor Kf the
caller supplies per component. In phase 2 those are engineering estimates in
the design state. In phase 6 they get fitted from FEA runs, which is what
turns the fast layer from indicative into trustworthy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .materials import Material

# Shigley table 6-2: k_a = a * S_ut^b with S_ut in MPa.
SURFACE_FACTORS = {
    "polished": (1.00, 0.0),
    "ground": (1.58, -0.085),
    "machined": (4.51, -0.265),
    "cold-drawn": (4.51, -0.265),
    "hot-rolled": (57.7, -0.718),
    "as-forged": (272.0, -0.995),
    "shot-peened": (1.58, -0.085),   # treated as ground; the compressive
                                     # residual stress benefit is not credited
}

# Shigley table 6-5.
RELIABILITY_FACTORS = {
    0.50: 1.000, 0.90: 0.897, 0.95: 0.868,
    0.99: 0.814, 0.999: 0.753, 0.9999: 0.702,
}


@dataclass
class EnduranceLimit:
    """A corrected endurance limit and every factor that produced it."""

    value: float                  # Pa
    uncorrected: float            # Pa
    k_surface: float
    k_size: float
    k_load: float
    k_temperature: float
    k_reliability: float
    ultimate: float               # Pa, at temperature
    notes: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "endurance_limit_pa": self.value,
            "uncorrected_pa": self.uncorrected,
            "k_surface": self.k_surface, "k_size": self.k_size,
            "k_load": self.k_load, "k_temperature": self.k_temperature,
            "k_reliability": self.k_reliability,
            "ultimate_at_temperature_pa": self.ultimate,
            "notes": self.notes,
        }


def surface_factor(material: Material, finish: str) -> float:
    try:
        a, b = SURFACE_FACTORS[finish]
    except KeyError:
        known = ", ".join(sorted(SURFACE_FACTORS))
        raise KeyError(f"unknown surface finish {finish!r}; known: {known}") from None
    if a == 1.0 and b == 0.0:
        return 1.0
    return min(1.0, a * (material.ultimate_strength / 1e6) ** b)


def size_factor(diameter: float, load: str = "bending") -> float:
    """Shigley eq. 6-20. ``diameter`` in metres; axial loading gets k_b = 1."""
    if load == "axial":
        return 1.0
    d_mm = diameter * 1e3
    if d_mm < 2.79:
        return 1.0
    if d_mm <= 51.0:
        return 1.24 * d_mm ** -0.107
    if d_mm <= 254.0:
        return 1.51 * d_mm ** -0.157
    return 0.6


def load_factor(load: str) -> float:
    return {"bending": 1.0, "axial": 0.85, "torsion": 0.59}[load]


def reliability_factor(reliability: float) -> float:
    if reliability in RELIABILITY_FACTORS:
        return RELIABILITY_FACTORS[reliability]
    keys = sorted(RELIABILITY_FACTORS)
    nearest = min(keys, key=lambda k: abs(k - reliability))
    return RELIABILITY_FACTORS[nearest]


def uncorrected_endurance_limit(material: Material) -> tuple[float, str]:
    """S_e', the rotary-beam limit before any correction.

    Steels have a genuine endurance limit at roughly 0.5 S_ut, capped at
    700 MPa. Aluminium does not have one at all -- it keeps losing strength
    with cycles -- so the table value is a fatigue strength quoted at 5x10^8
    cycles, and calling it a limit is a convenient fiction that the returned
    note makes explicit.
    """
    is_steel = material.key in {"4340", "300M"} or "steel" in material.name.lower()
    if is_steel:
        s_ut = material.ultimate_strength
        return (min(0.5 * s_ut, 700e6),
                "steel: S_e' = 0.5 S_ut capped at 700 MPa")
    return (material.endurance_limit,
            "non-ferrous: no true endurance limit; this is the fatigue "
            "strength at 5x10^8 cycles and life keeps falling beyond it")


def endurance_limit(material: Material, temperature: float,
                    diameter: float, finish: str = "machined",
                    load: str = "bending",
                    reliability: float = 0.99) -> EnduranceLimit:
    """The corrected endurance limit S_e, with its provenance attached."""
    raw, note = uncorrected_endurance_limit(material)
    k_a = surface_factor(material, finish)
    k_b = size_factor(diameter, load)
    k_c = load_factor(load)
    k_d = material.strength_factor_at(temperature)
    k_e = reliability_factor(reliability)

    notes = [note]
    if temperature > material.max_service_temp:
        notes.append(
            f"temperature {temperature - 273.15:.0f} C is above the material's "
            f"{material.max_service_temp - 273.15:.0f} C service limit; the "
            "derating curve is being extrapolated and should not be trusted")

    return EnduranceLimit(
        value=raw * k_a * k_b * k_c * k_d * k_e, uncorrected=raw,
        k_surface=k_a, k_size=k_b, k_load=k_c, k_temperature=k_d,
        k_reliability=k_e, ultimate=material.ultimate_at(temperature),
        notes=notes)


# --- Mean stress correction -----------------------------------------------

def goodman_factor(alternating: float, mean: float, s_e: float,
                   s_ut: float) -> float:
    """Modified Goodman safety factor.

    A compressive mean stress is beneficial rather than harmful, so it is not
    penalised: the factor reduces to S_e / sigma_a. This is standard practice
    and it matters here, because the crown and the sleeve both live under
    compressive mean stress.
    """
    if alternating <= 0.0:
        return math.inf
    if mean <= 0.0:
        return s_e / alternating
    denom = alternating / s_e + mean / s_ut
    return 1.0 / denom if denom > 0 else math.inf


def gerber_factor(alternating: float, mean: float, s_e: float,
                  s_ut: float) -> float:
    """Gerber parabolic criterion. Less conservative than Goodman and usually
    closer to test data for ductile materials."""
    if alternating <= 0.0:
        return math.inf
    if mean <= 0.0:
        return s_e / alternating
    ratio = 2.0 * mean * s_e / (s_ut * alternating)
    return 0.5 * (s_ut / mean) ** 2 * (alternating / s_e) * (
        -1.0 + math.sqrt(1.0 + ratio ** 2))


def _strength_fraction(s_ut: float) -> float:
    """Shigley's f, the fraction of S_ut reached at 10^3 cycles."""
    mpa = s_ut / 1e6
    if mpa < 490.0:
        return 0.9
    if mpa > 1400.0:
        return 0.77
    return 1.06 - 4.1e-4 * mpa + 1.5e-7 * mpa ** 2


def equivalent_reversed_stress(alternating: float, mean: float,
                               s_ut: float) -> float:
    """Goodman-equivalent fully reversed stress, for reading a life off S-N."""
    if mean <= 0.0:
        return alternating
    if mean >= s_ut:
        return math.inf
    return alternating / (1.0 - mean / s_ut)


def life_cycles(alternating: float, mean: float, s_e: float,
                s_ut: float) -> float:
    """Cycles to failure from the S-N line, with mean stress corrected out.

        S_f = a N^b,  a = (f S_ut)^2 / S_e,  b = -(1/3) log10(f S_ut / S_e)

    Returns infinity when the equivalent reversed stress sits below S_e. For
    aluminium that answer is optimistic: the material has no true endurance
    limit, and the note on the endurance limit says so.
    """
    reversed_stress = equivalent_reversed_stress(alternating, mean, s_ut)
    if not math.isfinite(reversed_stress):
        return 0.0
    if reversed_stress <= s_e:
        return math.inf
    f = _strength_fraction(s_ut)
    a = (f * s_ut) ** 2 / s_e
    b = -math.log10(f * s_ut / s_e) / 3.0
    if reversed_stress >= f * s_ut:
        return 1.0e3
    return (reversed_stress / a) ** (1.0 / b)


def hours_at_speed(cycles: float, speed_rad_s: float,
                   cycles_per_revolution: float = 1.0) -> float:
    """Convert a cycle count to running hours. 'Failed at 40 hours' is a very
    different piece of information from 'safety factor 1.8'."""
    if not math.isfinite(cycles):
        return math.inf
    rev_per_hour = speed_rad_s / (2.0 * math.pi) * 3600.0
    return cycles / (rev_per_hour * cycles_per_revolution)


# --- Preloaded bolts -------------------------------------------------------

@dataclass
class BoltResult:
    fatigue_factor: float
    yield_factor: float
    separation_factor: float
    preload_stress: float
    alternating_stress: float
    mean_stress: float


def preloaded_bolt(preload: float, external_load: float, stress_area: float,
                   stiffness_factor: float, s_e: float, s_ut: float,
                   proof_strength: float) -> BoltResult:
    """Shigley's three checks for a preloaded bolt under a fluctuating load.

    ``external_load`` is the tensile load per bolt, taken as fluctuating
    between zero and its peak -- which is what a rod bolt actually sees, once
    per revolution, at overlap TDC.

    The separation check is the one that bites first in practice: once a rod
    cap separates even momentarily, the bolt sees the full load instead of a
    fraction of it, and the fatigue calculation above no longer applies.
    """
    sigma_i = preload / stress_area
    sigma_a = stiffness_factor * external_load / (2.0 * stress_area)
    sigma_m = sigma_a + sigma_i

    if sigma_a <= 0.0:
        n_f = math.inf
    elif sigma_i >= s_ut:
        n_f = 0.0
    else:
        n_f = (s_e * (s_ut - sigma_i)) / (sigma_a * (s_ut + s_e))

    load_term = stiffness_factor * external_load
    n_p = ((proof_strength * stress_area - preload) / load_term
           if load_term > 0 else math.inf)

    sep_term = external_load * (1.0 - stiffness_factor)
    n_0 = preload / sep_term if sep_term > 0 else math.inf

    return BoltResult(fatigue_factor=n_f, yield_factor=n_p,
                      separation_factor=n_0, preload_stress=sigma_i,
                      alternating_stress=sigma_a, mean_stress=sigma_m)
