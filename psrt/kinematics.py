"""Exact crank-slider kinematics with finite rod length.

No simple-harmonic shortcut. The second-order term is what makes the inertia
load asymmetric between TDC and BDC, and that asymmetry is precisely what the
connecting rod has to survive -- at overlap TDC the full inertia load goes
into the rod in tension with no cylinder pressure to oppose it.

Geometry and sign conventions
-----------------------------
``r`` is the crank radius (half the stroke) and ``l`` the rod centre-to-centre
length. Let ``x`` be the distance from the crank axis to the pin axis::

    x(theta) = r cos(theta) + sqrt(l^2 - r^2 sin^2(theta))

and let ``s`` be piston displacement measured *down from TDC*::

    s(theta) = (r + l) - x(theta)

Everything downstream uses ``s``: it is zero at TDC, equals the stroke at BDC,
and its second derivative is positive when the piston accelerates away from
the cylinder head. That is the convention the load chain in :mod:`psrt.loads`
expects, and it is the one in which the familiar textbook form

    F_pin = F_gas - m_recip * a

comes out with positive meaning compression in the rod.

Derivatives are analytic, not finite-differenced. With
``R = sqrt(l^2 - r^2 sin^2(theta))``::

    dx/dtheta   = -r sin(theta) - (r^2/2) sin(2 theta) / R
    d2x/dtheta2 = -r cos(theta) - r^2 cos(2 theta) / R
                  - (r^4/4) sin^2(2 theta) / R^3

``tests/test_kinematics.py`` checks these against high-order numerical
differentiation and against the closed-form values at both dead centres.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CrankGeometry:
    """The two lengths that define a slider-crank, plus the bore area."""

    crank_radius: float      # m
    rod_length: float        # m
    bore_area: float         # m^2
    clearance_volume: float  # m^3

    def __post_init__(self) -> None:
        if self.rod_length <= self.crank_radius:
            raise ValueError(
                f"rod length ({self.rod_length} m) must exceed crank radius "
                f"({self.crank_radius} m) or the mechanism cannot rotate")

    @property
    def stroke(self) -> float:
        return 2.0 * self.crank_radius

    @property
    def rod_ratio(self) -> float:
        return self.rod_length / self.crank_radius

    @property
    def lam(self) -> float:
        """Crank-to-rod ratio r/l, the usual symbol lambda."""
        return self.crank_radius / self.rod_length

    @classmethod
    def from_state(cls, state) -> "CrankGeometry":
        return cls(
            crank_radius=state["engine.crank_radius"],
            rod_length=state["engine.rod_length"],
            bore_area=state["engine.bore_area"],
            clearance_volume=state["engine.clearance_volume"],
        )


def _root(geom: CrankGeometry, theta):
    """R = sqrt(l^2 - r^2 sin^2 theta). Always real for l > r."""
    r, l = geom.crank_radius, geom.rod_length
    return np.sqrt(l * l - (r * np.sin(theta)) ** 2)


def pin_distance(geom: CrankGeometry, theta):
    """Distance from crank axis to gudgeon pin axis, in metres."""
    r = geom.crank_radius
    return r * np.cos(theta) + _root(geom, theta)


def displacement(geom: CrankGeometry, theta):
    """Piston displacement measured down from TDC, in metres.

    Zero at TDC, equal to the stroke at BDC.
    """
    r, l = geom.crank_radius, geom.rod_length
    return (r + l) - pin_distance(geom, theta)


def d_displacement_d_theta(geom: CrankGeometry, theta):
    """ds/dtheta, in m/rad. Positive while the piston moves away from TDC."""
    r = geom.crank_radius
    R = _root(geom, theta)
    dx = -r * np.sin(theta) - (r * r / 2.0) * np.sin(2.0 * theta) / R
    return -dx


def d2_displacement_d_theta2(geom: CrankGeometry, theta):
    """d2s/dtheta2, in m/rad^2."""
    r = geom.crank_radius
    R = _root(geom, theta)
    d2x = (-r * np.cos(theta)
           - r * r * np.cos(2.0 * theta) / R
           - (r ** 4 / 4.0) * np.sin(2.0 * theta) ** 2 / R ** 3)
    return -d2x


def velocity(geom: CrankGeometry, theta, omega: float):
    """Piston velocity in m/s, positive moving away from the cylinder head."""
    return omega * d_displacement_d_theta(geom, theta)


def acceleration(geom: CrankGeometry, theta, omega: float):
    """Piston acceleration in m/s^2, positive pointing away from the head.

    Assumes constant angular velocity, which is the standard assumption for
    load analysis. At TDC this returns +omega^2 * r * (1 + r/l); at BDC,
    -omega^2 * r * (1 - r/l). The asymmetry between those two is the whole
    reason finite rod length matters.
    """
    return omega ** 2 * d2_displacement_d_theta2(geom, theta)


def rod_angle(geom: CrankGeometry, theta):
    """Rod obliquity phi, in radians, from sin(phi) = (r/l) sin(theta).

    Zero at both dead centres, maximum near mid-stroke. This angle is what
    turns axial rod force into side thrust on the cylinder wall.
    """
    return np.arcsin(geom.lam * np.sin(theta))


def cylinder_volume(geom: CrankGeometry, theta):
    """Instantaneous cylinder volume in m^3."""
    return geom.clearance_volume + geom.bore_area * displacement(geom, theta)


def d_volume_d_theta(geom: CrankGeometry, theta):
    """dV/dtheta in m^3/rad. Needed for the first-law pressure integration."""
    return geom.bore_area * d_displacement_d_theta(geom, theta)


def sweep(start_deg: float = -360.0, end_deg: float = 360.0,
          step_deg: float = 0.25) -> np.ndarray:
    """A crank-angle array in radians, inclusive of both endpoints.

    Default is a full four-stroke cycle at quarter-degree resolution: fine
    enough that peak pressure and peak force are not missed between samples,
    cheap enough to run thousands of times in an optimiser.
    """
    n = int(round((end_deg - start_deg) / step_deg)) + 1
    return np.radians(np.linspace(start_deg, end_deg, n))


# --- Closed-form reference values, used by the tests -----------------------

def acceleration_at_tdc(geom: CrankGeometry, omega: float) -> float:
    """omega^2 * r * (1 + r/l). Positive: piston accelerates away from head."""
    return omega ** 2 * geom.crank_radius * (1.0 + geom.lam)


def acceleration_at_bdc(geom: CrankGeometry, omega: float) -> float:
    """-omega^2 * r * (1 - r/l). Negative: piston accelerates toward the head."""
    return -omega ** 2 * geom.crank_radius * (1.0 - geom.lam)


def max_rod_angle(geom: CrankGeometry) -> float:
    """Peak obliquity, arcsin(r/l), reached where sin(theta) = 1."""
    return math.asin(geom.lam)
