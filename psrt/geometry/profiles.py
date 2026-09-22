"""Bore cross-section profiles.

The bore does not have to be a circle. It is one in every production engine
ever built, and this module exists so that claim can be tested rather than
asserted -- you can ask for a hexagonal bore, get real geometry, and read off
what it actually costs.

The envelope constraint is what makes the comparison meaningful. A block gives
you a maximum radius from the bore centre before the wall reaches a coolant
passage or the neighbouring cylinder. Every profile here is generated to fit
inside that same radius, so the areas are directly comparable.

For a fixed envelope radius R:

    circle              pi R^2            = 3.1416 R^2
    regular hexagon     (3 sqrt 3 / 2) R^2 = 2.5981 R^2   (82.7% of the circle)
    regular octagon     2 sqrt 2 R^2       = 2.8284 R^2   (90.0%)
    oval, ratio b       pi b R^2

The circle wins, and not by a little. That is the whole answer to the
hexagonal-bore question, and it comes from geometry rather than from tradition:
a circle encloses the most area for a given maximum width, so any other shape
inside the same envelope displaces less. To gain area, a polygon's corners have
to push past the envelope radius -- into exactly the material that is not there.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import cadquery as cq


@dataclass(frozen=True)
class BoreProfile:
    """One cross-section shape, with its area and a builder."""

    key: str
    name: str
    area_factor: float                  # area = area_factor * R^2
    build: Callable                     # (workplane, radius_mm) -> workplane
    sealable: bool
    note: str = ""

    def area(self, radius: float) -> float:
        return self.area_factor * radius ** 2

    def equivalent_diameter(self, radius: float) -> float:
        """Diameter of the circle with the same area, for comparison."""
        return 2.0 * math.sqrt(self.area(radius) / math.pi)


def _circle(wp, radius):
    return wp.circle(radius)


def _polygon(sides):
    def build(wp, radius):
        # CadQuery's polygon() takes the circumscribed diameter, which is
        # exactly the envelope constraint: every vertex sits on radius.
        return wp.polygon(sides, 2.0 * radius)
    return build


def _oval(ratio):
    def build(wp, radius):
        return wp.ellipse(radius, radius * ratio)
    return build


def circle() -> BoreProfile:
    return BoreProfile(
        key="circle", name="circular bore", area_factor=math.pi,
        build=_circle, sealable=True,
        note="the only shape a piston ring can seal: a circular ring under "
             "radial tension conforms to a circular bore by itself")


def polygon(sides: int) -> BoreProfile:
    if sides < 3:
        raise ValueError("a polygon needs at least three sides")
    factor = 0.5 * sides * math.sin(2.0 * math.pi / sides)
    names = {3: "triangular", 4: "square", 5: "pentagonal", 6: "hexagonal",
             8: "octagonal", 12: "dodecagonal"}
    return BoreProfile(
        key=f"polygon-{sides}", name=f"{names.get(sides, str(sides) + '-sided')} bore",
        area_factor=factor, build=_polygon(sides), sealable=False,
        note="no piston ring can seal this. The corners need a conformable "
             "seal surviving flame temperature and 100+ bar with essentially "
             "no lubrication, and they are stress risers and heat traps "
             "besides")


def oval(ratio: float = 0.9) -> BoreProfile:
    return BoreProfile(
        key=f"oval-{ratio:g}", name=f"oval bore, {ratio:g}:1",
        area_factor=math.pi * ratio, build=_oval(ratio), sealable=False,
        note="a ring can be made to follow a mild oval, and production bores "
             "are deliberately machined slightly oval to run round when hot, "
             "but a ring cannot maintain tension against a large one")


def get(key: str) -> BoreProfile:
    if key == "circle":
        return circle()
    if key.startswith("polygon-"):
        return polygon(int(key.split("-", 1)[1]))
    if key.startswith("oval-"):
        return oval(float(key.split("-", 1)[1]))
    raise KeyError(
        f"unknown bore profile {key!r}; use 'circle', 'polygon-N' or 'oval-R'")


def compare(envelope_radius: float, keys=None) -> list:
    """Area of every profile inside one envelope radius. Pure function.

    This is the study behind the hexagon question: fix what the block allows,
    then ask what each shape can displace inside it.
    """
    keys = keys or ["circle", "polygon-12", "polygon-8", "polygon-6",
                    "polygon-4", "oval-0.9"]
    reference = circle().area(envelope_radius)
    rows = []
    for key in keys:
        profile = get(key)
        area = profile.area(envelope_radius)
        rows.append({
            "key": key,
            "name": profile.name,
            "area_m2": area,
            "fraction_of_circle": area / reference,
            "equivalent_diameter_m": profile.equivalent_diameter(envelope_radius),
            "sealable": profile.sealable,
            "note": profile.note,
        })
    return sorted(rows, key=lambda r: -r["area_m2"])
