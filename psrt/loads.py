"""The load chain: from cylinder pressure to the force in every joint.

Gas pressure and inertia combine at the gudgeon pin, resolve through the rod
obliquity into an axial rod force and a side thrust on the liner, and that
axial force times the effective crank arm is the torque curve.

Sign conventions (fixed here, used everywhere downstream)
---------------------------------------------------------
* ``acceleration`` is positive pointing away from the cylinder head, matching
  :mod:`psrt.kinematics`.
* ``f_gas`` is positive when gas pushes the piston down the bore.
* ``f_pin``, ``f_rod``: **positive is compression**, negative is tension.
* ``f_side`` is positive toward the major thrust side (the side the piston is
  pressed against during the power stroke).
* ``torque`` is positive when the cylinder drives the crankshaft.

The governing relations::

    F_gas  = (p_cyl - p_crankcase) * A_bore
    F_pin  = F_gas - m_recip * a
    phi    = arcsin((r/l) sin(theta))
    F_rod  = F_pin / cos(phi)
    F_side = F_pin * tan(phi)
    T      = F_rod * r * sin(theta + phi)

Two crank angles govern almost everything
-----------------------------------------
Peak firing pressure, a little after TDC, gives the largest **compressive**
load: it sizes crown thickness, pin bending, rod buckling and bearing pressure.

Overlap TDC at redline gives the largest **tensile** load: there is no
cylinder pressure to oppose the inertia of the reciprocating mass, so the
whole of it goes into the rod in tension. That is the condition people forget,
and it is how connecting rods actually break.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import combustion, kinematics as kin


@dataclass
class Extremum:
    """A peak value and where in the cycle it happens."""

    value: float
    angle: float          # rad
    unit: str = "N"

    @property
    def angle_deg(self) -> float:
        return float(np.degrees(self.angle))


@dataclass
class LoadSweep:
    """Every force in the piston system, resolved over a full cycle."""

    theta: np.ndarray            # rad
    speed: float                 # rad/s
    pressure: np.ndarray         # Pa
    volume: np.ndarray           # m^3
    position: np.ndarray         # m, down from TDC
    velocity: np.ndarray         # m/s
    acceleration: np.ndarray     # m/s^2
    rod_angle: np.ndarray        # rad
    f_gas: np.ndarray            # N
    f_inertia: np.ndarray        # N
    f_pin: np.ndarray            # N, + compression
    f_rod: np.ndarray            # N, + compression
    f_side: np.ndarray           # N
    torque: np.ndarray           # N.m, per cylinder, instantaneous
    reciprocating_mass: float    # kg
    displacement: float          # m^3, per cylinder
    n_cylinders: int
    trace: combustion.PressureTrace
    _mean_piston_speed: float = 0.0
    _fmep_a: float = 0.0
    _fmep_b: float = 0.0
    _fmep_c: float = 0.0
    _fmep_d: float = 0.0

    # -- peaks --------------------------------------------------------------

    def _peak(self, arr: np.ndarray, mode: str = "max") -> Extremum:
        i = int(np.argmax(arr)) if mode == "max" else int(np.argmin(arr))
        return Extremum(float(arr[i]), float(self.theta[i]))

    @property
    def peak_pin_compression(self) -> Extremum:
        return self._peak(self.f_pin, "max")

    @property
    def peak_pin_tension(self) -> Extremum:
        """Most negative pin force. Reported as a negative number."""
        return self._peak(self.f_pin, "min")

    @property
    def peak_rod_compression(self) -> Extremum:
        return self._peak(self.f_rod, "max")

    @property
    def peak_rod_tension(self) -> Extremum:
        return self._peak(self.f_rod, "min")

    @property
    def peak_side_thrust(self) -> Extremum:
        i = int(np.argmax(np.abs(self.f_side)))
        return Extremum(float(self.f_side[i]), float(self.theta[i]))

    @property
    def peak_acceleration(self) -> Extremum:
        i = int(np.argmax(np.abs(self.acceleration)))
        return Extremum(float(self.acceleration[i]), float(self.theta[i]),
                        unit="m/s^2")

    # -- work and torque ----------------------------------------------------

    def cycle_work(self) -> float:
        """Indicated work per cylinder per cycle, by integrating torque.

        int T dtheta over the full 720 degrees. This must equal
        IMEP_net * displacement -- the check lives in tests/test_loads.py and
        is the single best evidence that the force resolution is right.
        """
        return float(np.trapezoid(self.torque, self.theta))

    def mean_torque_per_cylinder(self) -> float:
        """Indicated torque contributed by one cylinder, averaged over a
        four-stroke cycle (720 degrees = 4 pi radians)."""
        return self.cycle_work() / (4.0 * np.pi)

    def indicated_torque(self) -> float:
        """Indicated torque for the whole engine, in N.m.

        Indicated, not brake: friction and accessory drive are not modelled,
        so expect real crank torque to land roughly 10 to 15 percent below
        this on a healthy engine.
        """
        return self.mean_torque_per_cylinder() * self.n_cylinders

    def indicated_power(self) -> float:
        return self.indicated_torque() * self.speed

    def imep_net(self) -> float:
        return self.trace.imep_net(self.displacement)

    def imep_gross(self) -> float:
        return self.trace.imep_gross(self.displacement)

    # -- friction and brake output ------------------------------------------

    def fmep(self) -> float:
        """Friction mean effective pressure, Chen-Flynn form::

            FMEP = A + B * p_max + C * S_p + D * S_p^2

        with S_p the mean piston speed. It is an empirical correlation, not
        physics: the coefficients are a calibration and are labelled as such
        in the design state. It exists so the tool can report BRAKE output,
        which is the only thing a published dyno figure can be compared with.
        """
        return (self._fmep_a
                + self._fmep_b * self.trace.peak_pressure
                + self._fmep_c * self._mean_piston_speed
                + self._fmep_d * self._mean_piston_speed ** 2)

    def bmep(self) -> float:
        """Brake mean effective pressure: IMEP net less friction."""
        return self.imep_net() - self.fmep()

    def brake_torque(self) -> float:
        """Torque at the crankshaft, in N.m. Comparable with a dyno sheet."""
        return (self.bmep() * self.displacement * self.n_cylinders
                / (4.0 * np.pi))

    def brake_power(self) -> float:
        return self.brake_torque() * self.speed

    def mechanical_efficiency(self) -> float:
        """BMEP / IMEP_net. Healthy engines sit around 0.80 to 0.90 at peak
        torque, falling toward 0.75 at redline."""
        imep = self.imep_net()
        return self.bmep() / imep if imep else float("nan")

    # -- named conditions ---------------------------------------------------

    def at_angle(self, angle_deg: float) -> dict:
        """Everything about the state of the system at one crank angle."""
        i = int(np.argmin(np.abs(self.theta - np.radians(angle_deg))))
        return {
            "angle_deg": float(np.degrees(self.theta[i])),
            "pressure": float(self.pressure[i]),
            "acceleration": float(self.acceleration[i]),
            "f_gas": float(self.f_gas[i]),
            "f_inertia": float(self.f_inertia[i]),
            "f_pin": float(self.f_pin[i]),
            "f_rod": float(self.f_rod[i]),
            "f_side": float(self.f_side[i]),
            "torque": float(self.torque[i]),
        }

    def at_overlap_tdc(self) -> dict:
        """The inertia-only condition: exhaust TDC, no gas force to oppose it.

        This is the tensile case that sizes the rod shank, the rod bolts and
        the pin bosses.
        """
        return self.at_angle(360.0)

    def at_peak_pressure(self) -> dict:
        return self.at_angle(float(np.degrees(self.trace.peak_pressure_angle)))


def compute(state, trace: combustion.PressureTrace | None = None,
            theta: np.ndarray | None = None) -> LoadSweep:
    """Resolve the full load chain for a design state. Pure function."""
    geom = kin.CrankGeometry.from_state(state)
    omega = state["operating.speed"]

    if trace is None:
        trace = combustion.simulate(state, theta)
    theta = trace.theta

    area = geom.bore_area
    m_recip = state["masses.reciprocating"]
    p_case = state["operating.crankcase_pressure"]

    position = kin.displacement(geom, theta)
    velocity = kin.velocity(geom, theta, omega)
    accel = kin.acceleration(geom, theta, omega)
    phi = kin.rod_angle(geom, theta)

    f_gas = (trace.pressure - p_case) * area
    f_inertia = -m_recip * accel
    f_pin = f_gas + f_inertia

    cos_phi = np.cos(phi)
    f_rod = f_pin / cos_phi
    f_side = f_pin * np.tan(phi)
    torque = f_rod * geom.crank_radius * np.sin(theta + phi)

    return LoadSweep(
        theta=theta, speed=omega,
        pressure=trace.pressure, volume=trace.volume,
        position=position, velocity=velocity, acceleration=accel,
        rod_angle=phi,
        f_gas=f_gas, f_inertia=f_inertia, f_pin=f_pin, f_rod=f_rod,
        f_side=f_side, torque=torque,
        reciprocating_mass=m_recip,
        displacement=state["engine.displacement_cyl"],
        n_cylinders=int(state["engine.n_cylinders"]),
        trace=trace,
        _mean_piston_speed=state["engine.mean_piston_speed"],
        _fmep_a=state["operating.fmep_constant"],
        _fmep_b=state["operating.fmep_load_coeff"],
        _fmep_c=state["operating.fmep_speed_coeff"],
        _fmep_d=state["operating.fmep_speed2_coeff"])
