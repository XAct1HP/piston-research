"""Single-zone combustion model: cylinder pressure against crank angle.

Every structural load in the piston system is downstream of p(theta), so this
is the foundation. The model is a first-law energy balance on the cylinder
contents over the closed period, with heat release shaped by a Wiebe function:

    dp/dtheta = -gamma * p/V * dV/dtheta + (gamma - 1)/V * dQ/dtheta

integrated by fourth-order Runge-Kutta from intake valve closing to exhaust
valve opening. Outside the closed period the cylinder is at manifold or
exhaust pressure, with an exponential blowdown between EVO and BDC.

What this model is good for: the shape and magnitude of the pressure trace,
peak pressure and where it falls, pressure rise rate, and IMEP. That is
everything the load chain needs.

What it is not good for: predicting knock, resolving how a bore change alters
flame travel, or capturing heat transfer properly. Wall heat loss appears here
as a single lumped fraction of released heat, which is a stand-in, not physics.
When an answer has to be right, import a measured pressure trace instead --
:func:`from_csv` exists for exactly that.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import kinematics as kin
from .units import R_AIR


@dataclass
class PressureTrace:
    """Cylinder pressure and everything computed alongside it."""

    theta: np.ndarray            # rad, crank angle (0 = firing TDC)
    pressure: np.ndarray         # Pa
    volume: np.ndarray           # m^3
    mfb: np.ndarray              # mass fraction burned, 0..1
    trapped_mass: float          # kg
    fuel_mass: float             # kg
    heat_released: float         # J, after efficiency and wall losses
    source: str = "wiebe"        # "wiebe" or "measured"

    # -- headline numbers ---------------------------------------------------

    @property
    def peak_pressure(self) -> float:
        return float(self.pressure.max())

    @property
    def peak_pressure_angle(self) -> float:
        """Crank angle of peak pressure, in radians. Healthy SI combustion
        puts this around 12 to 18 degrees after TDC."""
        return float(self.theta[int(np.argmax(self.pressure))])

    @property
    def max_pressure_rise_rate(self) -> float:
        """Max dp/dtheta in Pa/rad. Above roughly 10 bar/deg an SI engine is
        usually audibly rough; it is also a knock proxy."""
        return float(np.max(np.gradient(self.pressure, self.theta)))

    def imep(self, displacement: float, lo_deg: float, hi_deg: float) -> float:
        """Mean effective pressure over a crank-angle window: (1/Vd) * int p dV."""
        mask = ((self.theta >= np.radians(lo_deg))
                & (self.theta <= np.radians(hi_deg)))
        work = np.trapezoid(self.pressure[mask], self.volume[mask])
        return work / displacement

    def imep_gross(self, displacement: float) -> float:
        """Compression and expansion strokes only."""
        return self.imep(displacement, -180.0, 180.0)

    def imep_net(self, displacement: float) -> float:
        """All four strokes, so pumping work is included."""
        return self.imep(displacement, -360.0, 360.0)

    def pmep(self, displacement: float) -> float:
        """Pumping mean effective pressure. Negative on a throttled engine."""
        return self.imep_net(displacement) - self.imep_gross(displacement)


# --- Wiebe heat release ----------------------------------------------------

def wiebe_mfb(theta, soc: float, duration: float, a: float, m: float):
    """Mass fraction burned, 0 before SOC, asymptotically 1 after.

        x_b = 1 - exp(-a * ((theta - soc)/duration) ** (m + 1))

    a = 6.908 puts 99.9% burned at the end of the stated duration; m = 2 is a
    typical SI form factor.
    """
    tau = (np.asarray(theta, dtype=float) - soc) / duration
    tau = np.clip(tau, 0.0, None)
    return 1.0 - np.exp(-a * tau ** (m + 1.0))


def wiebe_burn_rate(theta, soc: float, duration: float, a: float, m: float):
    """dx_b/dtheta, in 1/rad."""
    tau = (np.asarray(theta, dtype=float) - soc) / duration
    tau = np.clip(tau, 0.0, None)
    return (a * (m + 1.0) / duration) * tau ** m * np.exp(-a * tau ** (m + 1.0))


# --- The model -------------------------------------------------------------

def simulate(state, theta: np.ndarray | None = None) -> PressureTrace:
    """Compute the pressure trace for a design state over a full cycle."""
    geom = kin.CrankGeometry.from_state(state)
    if theta is None:
        theta = kin.sweep(-360.0, 360.0, 0.25)

    ivc = state["operating.ivc_angle"]
    evo = state["operating.evo_angle"]
    p_int = state["operating.intake_pressure"]  # manifold, for reference
    t_int = state["operating.intake_temperature"]
    p_exh = state["operating.exhaust_pressure"]
    gamma = state["operating.gamma"]
    soc = state["operating.soc_angle"]
    duration = state["operating.burn_duration"]
    a_w = state["operating.wiebe_a"]
    m_w = state["operating.wiebe_m"]

    volume = kin.cylinder_volume(geom, theta)

    # Trapped charge at IVC. Mass comes from volumetric efficiency against
    # swept volume, so engine speed genuinely changes the charge; the pressure
    # at IVC then follows from the ideal gas law rather than being assumed.
    # The mixture is air plus fuel, so fuel mass follows from the AFR.
    v_ivc = float(kin.cylinder_volume(geom, ivc))
    trapped_mass = state["operating.trapped_mass"]
    p_ivc = trapped_mass * R_AIR * t_int / v_ivc
    fuel_mass = trapped_mass / (1.0 + state["operating.afr"])
    heat_released = (fuel_mass
                     * state["operating.fuel_lhv"]
                     * state["operating.combustion_efficiency"]
                     * (1.0 - state["operating.heat_loss_fraction"]))

    def dp_dtheta(th: float, p: float) -> float:
        v = float(kin.cylinder_volume(geom, th))
        dv = float(kin.d_volume_d_theta(geom, th))
        dq = heat_released * float(wiebe_burn_rate(th, soc, duration, a_w, m_w))
        return -gamma * p * dv / v + (gamma - 1.0) * dq / v

    pressure = np.empty_like(theta)

    closed = (theta >= ivc) & (theta <= evo)
    idx = np.flatnonzero(closed)

    # Gas exchange. The intake portion is held at the trapped pressure rather
    # than at manifold pressure: it keeps the PV loop self-consistent and
    # avoids inventing a pumping artefact. The cost is that PMEP from this
    # model is approximate -- it is not a breathing model.
    pressure[theta < ivc] = p_ivc
    pressure[theta > evo] = p_exh

    if idx.size:
        p = p_ivc
        pressure[idx[0]] = p
        for j in range(idx.size - 1):
            th0 = float(theta[idx[j]])
            h = float(theta[idx[j + 1]] - th0)
            k1 = dp_dtheta(th0, p)
            k2 = dp_dtheta(th0 + h / 2.0, p + h * k1 / 2.0)
            k3 = dp_dtheta(th0 + h / 2.0, p + h * k2 / 2.0)
            k4 = dp_dtheta(th0 + h, p + h * k3)
            p = p + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            p = max(p, 1.0e3)          # keep the integrator physical
            pressure[idx[j + 1]] = p

        # Blowdown: exponential decay from the pressure at EVO toward the
        # exhaust back pressure. Cosmetic for peak loads, but it avoids a
        # step discontinuity in the force trace during the exhaust stroke.
        p_evo = pressure[idx[-1]]
        tail = theta > evo
        if p_evo > p_exh:
            decay = np.radians(25.0)   # blowdown time constant, crank angle
            pressure[tail] = (p_exh + (p_evo - p_exh)
                              * np.exp(-(theta[tail] - evo) / decay))

    mfb = wiebe_mfb(theta, soc, duration, a_w, m_w)
    mfb = np.where(theta < ivc, 0.0, mfb)

    return PressureTrace(
        theta=theta, pressure=pressure, volume=volume, mfb=mfb,
        trapped_mass=float(trapped_mass), fuel_mass=float(fuel_mass),
        heat_released=float(heat_released), source="wiebe")


def from_csv(state, path: str, angle_column: int = 0, pressure_column: int = 1,
             angle_unit: str = "deg", pressure_unit: str = "bar",
             theta: np.ndarray | None = None) -> PressureTrace:
    """Load a measured pressure trace and resample it onto the crank grid.

    The file needs two columns: crank angle and cylinder pressure, with 0
    degrees at firing TDC. A measured trace makes every downstream number
    defensible in a way a Wiebe fit cannot.
    """
    raw = np.genfromtxt(path, delimiter=",", skip_header=0, comments="#")
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    ang = raw[:, angle_column].astype(float)
    prs = raw[:, pressure_column].astype(float)

    ang = np.radians(ang) if angle_unit.lower().startswith("deg") else ang
    scale = {"bar": 1.0e5, "pa": 1.0, "kpa": 1.0e3,
             "mpa": 1.0e6, "psi": 6894.757293168361}[pressure_unit.lower()]
    prs = prs * scale

    order = np.argsort(ang)
    ang, prs = ang[order], prs[order]

    geom = kin.CrankGeometry.from_state(state)
    if theta is None:
        theta = kin.sweep(-360.0, 360.0, 0.25)

    # Resample onto the requested grid. Angles the file actually covers are
    # interpolated directly; only angles outside its range are wrapped by the
    # 720 degree cycle period. Wrapping unconditionally would alias the last
    # sample of a full-cycle trace back onto the first.
    span = 4.0 * np.pi
    lo, hi = float(ang.min()), float(ang.max())
    query = np.asarray(theta, dtype=float).copy()
    outside = (query < lo) | (query > hi)
    if outside.any():
        query[outside] = lo + np.mod(query[outside] - lo, span)
    pressure = np.interp(query, ang, prs)

    coverage = np.degrees(hi - lo)
    if coverage < 359.0:
        import warnings as _warnings
        _warnings.warn(
            f"measured trace covers only {coverage:.0f} crank degrees; "
            "angles outside that range are clamped to the nearest sample, "
            "so loads during the uncovered strokes are not trustworthy",
            stacklevel=2)

    return PressureTrace(
        theta=theta, pressure=pressure,
        volume=kin.cylinder_volume(geom, theta),
        mfb=np.full_like(theta, np.nan),
        trapped_mass=float("nan"), fuel_mass=float("nan"),
        heat_released=float("nan"), source="measured")
