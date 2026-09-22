"""SI units and display conversions.

The entire tool computes in strict SI. Mixed units are the most common source
of silent wrongness in engineering code, so there is exactly one rule: if a
number is inside the model, it is SI. Conversions live here and are applied
only when formatting output for a human.
"""

from __future__ import annotations

import math

# --- Physical constants ----------------------------------------------------

R_UNIVERSAL = 8.314462618        # J/(mol K)
R_AIR = 287.05                   # J/(kg K), specific gas constant for dry air
GAMMA_AIR = 1.400                # ratio of specific heats, cold air
P_ATM = 101_325.0                # Pa
T_STD = 288.15                   # K
G_0 = 9.80665                    # m/s^2

# --- Length ----------------------------------------------------------------

MM = 1.0e-3                      # multiply millimetres by this to get metres
INCH = 0.0254

def m_to_mm(x: float) -> float:
    return x / MM

def mm_to_m(x: float) -> float:
    return x * MM

def m_to_in(x: float) -> float:
    return x / INCH

def in_to_m(x: float) -> float:
    return x * INCH

# --- Volume ----------------------------------------------------------------

CC = 1.0e-6                      # cubic centimetre in m^3
LITRE = 1.0e-3

def m3_to_cc(x: float) -> float:
    return x / CC

def cc_to_m3(x: float) -> float:
    return x * CC

def m3_to_litre(x: float) -> float:
    return x / LITRE

# --- Pressure --------------------------------------------------------------

BAR = 1.0e5
KPA = 1.0e3
MPA = 1.0e6
PSI = 6894.757293168361

def pa_to_bar(x: float) -> float:
    return x / BAR

def bar_to_pa(x: float) -> float:
    return x * BAR

def pa_to_mpa(x: float) -> float:
    return x / MPA

def pa_to_psi(x: float) -> float:
    return x / PSI

def psi_to_pa(x: float) -> float:
    return x * PSI

# --- Force and mass --------------------------------------------------------

KN = 1.0e3
GRAM = 1.0e-3

def n_to_kn(x: float) -> float:
    return x / KN

def kg_to_g(x: float) -> float:
    return x / GRAM

def g_to_kg(x: float) -> float:
    return x * GRAM

# --- Rotation --------------------------------------------------------------

def rpm_to_rad_s(rpm: float) -> float:
    """Revolutions per minute to radians per second."""
    return rpm * 2.0 * math.pi / 60.0

def rad_s_to_rpm(w: float) -> float:
    return w * 60.0 / (2.0 * math.pi)

def deg_to_rad(x: float) -> float:
    return math.radians(x)

def rad_to_deg(x: float) -> float:
    return math.degrees(x)

# --- Power -----------------------------------------------------------------

KW = 1.0e3
HP_MECHANICAL = 745.699871582    # imperial horsepower
PS_METRIC = 735.49875            # metric horsepower / PferdestArke

def w_to_kw(x: float) -> float:
    return x / KW

def w_to_hp(x: float) -> float:
    return x / HP_MECHANICAL

def w_to_ps(x: float) -> float:
    return x / PS_METRIC

# --- Temperature -----------------------------------------------------------

T_ZERO_C = 273.15

def c_to_k(x: float) -> float:
    return x + T_ZERO_C

def k_to_c(x: float) -> float:
    return x - T_ZERO_C

# --- Display registry ------------------------------------------------------
# Maps an SI unit string to (display unit, conversion factor from SI).
# Used by the CLI and, later, by the UI so that a parameter knows how it
# should be shown without every call site re-deciding.

DISPLAY = {
    "m": ("mm", 1.0 / MM),
    "m^2": ("mm^2", 1.0e6),
    "m^3": ("cc", 1.0 / CC),
    "m^4": ("mm^4", 1.0e12),
    "Pa": ("bar", 1.0 / BAR),
    "kg": ("g", 1.0 / GRAM),
    "N": ("kN", 1.0 / KN),
    "N.m": ("N.m", 1.0),
    "W": ("kW", 1.0 / KW),
    "K": ("C", 1.0),           # offset handled specially in fmt()
    "rad": ("deg", 180.0 / math.pi),
    "rad/s": ("rpm", 60.0 / (2.0 * math.pi)),
    "m/s": ("m/s", 1.0),
    "m/s^2": ("m/s^2", 1.0),
    "J": ("J", 1.0),
    "J/kg": ("MJ/kg", 1.0e-6),
    "-": ("", 1.0),
    "": ("", 1.0),
}

def fmt(value, si_unit: str = "", places: int = 3) -> str:
    """Format an SI value in its conventional display unit."""
    if value is None:
        return "-"
    if isinstance(value, str) or isinstance(value, bool):
        return str(value)
    disp, factor = DISPLAY.get(si_unit, (si_unit, 1.0))
    if si_unit == "K":
        shown = k_to_c(value)
    else:
        shown = value * factor
    text = f"{shown:.{places}f}".rstrip("0").rstrip(".")
    if text in ("", "-"):
        text = "0"
    return f"{text} {disp}".strip()
