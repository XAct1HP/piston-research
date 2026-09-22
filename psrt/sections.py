"""Cross-section properties.

Closed-form area, second moment and radius of gyration for the sections this
tool needs. Phase 3 replaces these with exact values from the CadQuery solid;
until then they are the design variables themselves, and they are the reason
the buckling and bending models have anything to work with.

Orientation convention for the connecting rod
---------------------------------------------
The rod's I-section has its web lying in the plane of rotation, flanges
perpendicular to the crank axis::

        <--- B --->        B  flange width, along the CRANK AXIS
      +-----------+  ^     H  section height, in the PLANE OF ROTATION
      |___________|  |     tf flange thickness
          |   |      H     tw web thickness, measured along the crank axis
          |tw |      |
       ___|___|___   |
      |           |  |
      +-----------+  v

``i_xx`` resists bending in the plane of rotation (in-plane buckling, where
the rod is effectively pinned at both ends). ``i_yy`` resists bending along
the crank axis (out-of-plane buckling, where the pin and big end provide much
more restraint). I_xx is normally the larger of the two, which is exactly why
rods are built this way: the axis with less end restraint gets more material.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SectionProperties:
    area: float          # m^2
    i_xx: float          # m^4, bending in the plane of rotation
    i_yy: float          # m^4, bending along the crank axis
    z_xx: float          # m^3, section modulus
    z_yy: float          # m^3

    @property
    def r_xx(self) -> float:
        """Radius of gyration about xx, in metres."""
        return math.sqrt(self.i_xx / self.area)

    @property
    def r_yy(self) -> float:
        return math.sqrt(self.i_yy / self.area)

    @property
    def r_min(self) -> float:
        return min(self.r_xx, self.r_yy)


def i_beam(height: float, width: float, web: float,
           flange: float) -> SectionProperties:
    """Symmetric I or H section. See the module docstring for orientation.

    Raises ValueError if the flanges overlap or the web is wider than the
    flange, which are the two ways these four numbers can describe nothing.
    """
    if flange * 2.0 >= height:
        raise ValueError(
            f"flanges ({flange * 1e3:.2f} mm each) do not fit in a section "
            f"{height * 1e3:.2f} mm tall")
    if web > width:
        raise ValueError(
            f"web ({web * 1e3:.2f} mm) cannot be wider than the flange "
            f"({width * 1e3:.2f} mm)")

    inner_h = height - 2.0 * flange
    area = width * height - (width - web) * inner_h

    i_xx = (width * height ** 3 - (width - web) * inner_h ** 3) / 12.0
    i_yy = (2.0 * flange * width ** 3 + inner_h * web ** 3) / 12.0

    return SectionProperties(
        area=area, i_xx=i_xx, i_yy=i_yy,
        z_xx=i_xx / (height / 2.0), z_yy=i_yy / (width / 2.0))


def tube(outer_dia: float, inner_dia: float) -> SectionProperties:
    """Hollow circular section, as used for the gudgeon pin."""
    if inner_dia >= outer_dia:
        raise ValueError(
            f"pin bore ({inner_dia * 1e3:.2f} mm) must be smaller than its "
            f"outside diameter ({outer_dia * 1e3:.2f} mm)")
    area = math.pi * (outer_dia ** 2 - inner_dia ** 2) / 4.0
    i = math.pi * (outer_dia ** 4 - inner_dia ** 4) / 64.0
    return SectionProperties(area=area, i_xx=i, i_yy=i,
                             z_xx=i / (outer_dia / 2.0),
                             z_yy=i / (outer_dia / 2.0))


def rectangle(height: float, width: float) -> SectionProperties:
    area = height * width
    i_xx = width * height ** 3 / 12.0
    i_yy = height * width ** 3 / 12.0
    return SectionProperties(area=area, i_xx=i_xx, i_yy=i_yy,
                             z_xx=i_xx / (height / 2.0),
                             z_yy=i_yy / (width / 2.0))


def annulus_area(outer_dia: float, inner_dia: float) -> float:
    return math.pi * (outer_dia ** 2 - inner_dia ** 2) / 4.0


def bolt_stress_area(thread_dia: float, pitch: float) -> float:
    """ISO 898 tensile stress area::

        A_t = (pi/4) (d - 0.938194 p)^2

    Uses the mean of pitch and minor diameter, which is the standard basis for
    quoting bolt strength.
    """
    return math.pi / 4.0 * (thread_dia - 0.938194 * pitch) ** 2
