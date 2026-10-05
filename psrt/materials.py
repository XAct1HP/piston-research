"""Material properties, with temperature derating.

IMPORTANT: every number here is a nominal handbook value for a generic
condition of the alloy. Real properties depend on heat treatment, section
size, casting quality and supplier. Before any result from this tool informs a
part you intend to make, replace these with values from the actual material
certificate. Each entry carries a ``source`` string saying as much, and the
tool surfaces it rather than hiding it.

Temperature matters more than people expect for pistons. 2618-T61 loses
roughly half its room-temperature yield strength by 250 C, and a piston crown
runs hotter than that. Ignoring derating makes every crown look about twice as
strong as it is.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .units import c_to_k


@dataclass(frozen=True)
class Material:
    key: str
    name: str
    density: float                  # kg/m^3
    youngs_modulus: float           # Pa
    poisson: float                  # -
    yield_strength: float | None    # Pa at 20 C; None for brittle materials
    ultimate_strength: float        # Pa at 20 C
    endurance_limit: float          # Pa, fully reversed, polished, 10^7 cycles
    thermal_conductivity: float     # W/(m K)
    cte: float                      # 1/K
    max_service_temp: float         # K, beyond which the model refuses
    derating: tuple = ()            # ((T_kelvin, factor), ...) on strength
    source: str = ("nominal handbook value for a generic condition; verify "
                   "against your material certificate")

    def strength_factor_at(self, temperature: float) -> float:
        """Fraction of room-temperature strength retained at a temperature."""
        if not self.derating:
            return 1.0
        temps = np.array([t for t, _ in self.derating])
        facs = np.array([f for _, f in self.derating])
        return float(np.interp(temperature, temps, facs))

    def yield_at(self, temperature: float) -> float | None:
        if self.yield_strength is None:
            return None
        return self.yield_strength * self.strength_factor_at(temperature)

    def ultimate_at(self, temperature: float) -> float:
        return self.ultimate_strength * self.strength_factor_at(temperature)

    def endurance_at(self, temperature: float) -> float:
        return self.endurance_limit * self.strength_factor_at(temperature)


def _d(*pairs):
    """Build a derating table from (degrees C, retained fraction) pairs."""
    return tuple((c_to_k(t), f) for t, f in pairs)


MATERIALS: dict[str, Material] = {
    "2618-T61": Material(
        key="2618-T61", name="Aluminium 2618-T61 (forged piston alloy)",
        density=2760.0, youngs_modulus=74.0e9, poisson=0.33,
        yield_strength=372.0e6, ultimate_strength=441.0e6,
        endurance_limit=124.0e6, thermal_conductivity=146.0, cte=22.3e-6,
        max_service_temp=c_to_k(380.0),
        derating=_d((20, 1.00), (100, 0.95), (150, 0.88), (200, 0.70),
                    (250, 0.45), (300, 0.28), (350, 0.15), (400, 0.07))),

    "4032-T6": Material(
        key="4032-T6", name="Aluminium 4032-T6 (low-expansion piston alloy)",
        density=2680.0, youngs_modulus=79.0e9, poisson=0.33,
        yield_strength=315.0e6, ultimate_strength=380.0e6,
        endurance_limit=110.0e6, thermal_conductivity=141.0, cte=19.4e-6,
        max_service_temp=c_to_k(350.0),
        derating=_d((20, 1.00), (100, 0.93), (150, 0.84), (200, 0.65),
                    (250, 0.40), (300, 0.22), (350, 0.10))),

    "A390-T6": Material(
        key="A390-T6", name="Aluminium A390-T6 (hypereutectic cast)",
        density=2730.0, youngs_modulus=81.0e9, poisson=0.33,
        yield_strength=310.0e6, ultimate_strength=330.0e6,
        endurance_limit=95.0e6, thermal_conductivity=134.0, cte=18.0e-6,
        max_service_temp=c_to_k(330.0),
        derating=_d((20, 1.00), (100, 0.94), (150, 0.85), (200, 0.66),
                    (250, 0.42), (300, 0.24))),

    "4340": Material(
        key="4340", name="Steel 4340, quenched and tempered",
        density=7850.0, youngs_modulus=205.0e9, poisson=0.29,
        yield_strength=950.0e6, ultimate_strength=1100.0e6,
        endurance_limit=500.0e6, thermal_conductivity=44.5, cte=12.3e-6,
        max_service_temp=c_to_k(400.0),
        derating=_d((20, 1.00), (200, 0.94), (300, 0.88), (400, 0.78),
                    (500, 0.62)),
        source=("nominal for a common Q&T condition; 4340 spans roughly 860 "
                "to 1500 MPa yield depending on temper, so this must be set "
                "from your actual heat treatment")),

    "300M": Material(
        key="300M", name="Steel 300M (high-strength rod bolt / rod alloy)",
        density=7870.0, youngs_modulus=205.0e9, poisson=0.29,
        yield_strength=1590.0e6, ultimate_strength=1930.0e6,
        endurance_limit=690.0e6, thermal_conductivity=38.0, cte=12.0e-6,
        max_service_temp=c_to_k(400.0),
        derating=_d((20, 1.00), (200, 0.93), (300, 0.86), (400, 0.75))),

    "Ti-6Al-4V": Material(
        key="Ti-6Al-4V", name="Titanium Ti-6Al-4V, annealed",
        density=4430.0, youngs_modulus=114.0e9, poisson=0.342,
        yield_strength=880.0e6, ultimate_strength=950.0e6,
        endurance_limit=500.0e6, thermal_conductivity=6.7, cte=8.6e-6,
        max_service_temp=c_to_k(400.0),
        derating=_d((20, 1.00), (200, 0.82), (300, 0.74), (400, 0.66))),

    "grey-iron": Material(
        key="grey-iron", name="Grey cast iron, class 40 (cylinder liner)",
        density=7200.0, youngs_modulus=110.0e9, poisson=0.26,
        yield_strength=None, ultimate_strength=276.0e6,
        endurance_limit=110.0e6, thermal_conductivity=48.0, cte=11.0e-6,
        max_service_temp=c_to_k(400.0),
        derating=_d((20, 1.00), (200, 0.95), (300, 0.90), (400, 0.80)),
        source=("grey iron has no distinct yield point and is far stronger in "
                "compression than tension; sizing uses ultimate strength")),

    "ductile-iron": Material(
        key="ductile-iron", name="Ductile iron 65-45-12 (sleeve)",
        density=7100.0, youngs_modulus=169.0e9, poisson=0.29,
        yield_strength=310.0e6, ultimate_strength=448.0e6,
        endurance_limit=185.0e6, thermal_conductivity=33.0, cte=11.6e-6,
        max_service_temp=c_to_k(400.0),
        derating=_d((20, 1.00), (200, 0.93), (300, 0.87), (400, 0.76))),

    "bronze-c93200": Material(
        key="bronze-c93200", name="Bearing bronze C93200 (small-end bushing)",
        density=8930.0, youngs_modulus=100.0e9, poisson=0.34,
        yield_strength=124.0e6, ultimate_strength=240.0e6,
        endurance_limit=70.0e6, thermal_conductivity=59.0, cte=18.0e-6,
        max_service_temp=c_to_k(230.0),
        derating=_d((20, 1.00), (100, 0.88), (150, 0.76), (200, 0.60))),

    # --- candidate cores for the rail rod's stabilising sleeve -------------
    # The concept leaves the sleeve material open, so these are here to be
    # compared, not recommended. The composites are entered as ISOTROPIC
    # equivalents -- a quasi-isotropic layup's in-plane figures -- because
    # the solver is isotropic. A real laminate is far weaker through its
    # thickness and its strength depends on the layup; treat these as a
    # first cut and replace them with coupon data.
    "CFRP-quasi-iso": Material(
        key="CFRP-quasi-iso",
        name="Carbon/epoxy, quasi-isotropic layup (isotropic equivalent)",
        density=1550.0, youngs_modulus=55.0e9, poisson=0.30,
        yield_strength=None, ultimate_strength=600.0e6,
        endurance_limit=240.0e6, thermal_conductivity=5.0, cte=2.0e-6,
        max_service_temp=c_to_k(150.0),
        derating=_d((20, 1.00), (100, 0.90), (130, 0.75), (150, 0.55)),
        source=("in-plane quasi-isotropic figures for a 60% fibre epoxy; "
                "through-thickness and interlaminar strength are an order of "
                "magnitude lower and are NOT represented. Epoxy matrices "
                "soften near 130 C, which is close to rod temperature")),

    "PEEK-CF30": Material(
        key="PEEK-CF30", name="PEEK, 30% short carbon fibre (injection moulded)",
        density=1410.0, youngs_modulus=22.0e9, poisson=0.35,
        yield_strength=None, ultimate_strength=220.0e6,
        endurance_limit=80.0e6, thermal_conductivity=0.9, cte=15.0e-6,
        max_service_temp=c_to_k(250.0),
        derating=_d((20, 1.00), (100, 0.85), (150, 0.62), (200, 0.45),
                    (250, 0.35)),
        source="flow-direction values; cross-flow is roughly 60% of these"),

    "AZ80-T5": Material(
        key="AZ80-T5", name="Magnesium AZ80-T5 (forged)",
        density=1800.0, youngs_modulus=45.0e9, poisson=0.35,
        yield_strength=275.0e6, ultimate_strength=380.0e6,
        endurance_limit=100.0e6, thermal_conductivity=76.0, cte=26.0e-6,
        max_service_temp=c_to_k(150.0),
        derating=_d((20, 1.00), (100, 0.88), (150, 0.72), (200, 0.52))),

    "7075-T6": Material(
        key="7075-T6", name="Aluminium 7075-T6",
        density=2810.0, youngs_modulus=71.7e9, poisson=0.33,
        yield_strength=503.0e6, ultimate_strength=572.0e6,
        endurance_limit=160.0e6, thermal_conductivity=130.0, cte=23.4e-6,
        max_service_temp=c_to_k(200.0),
        derating=_d((20, 1.00), (100, 0.92), (150, 0.75), (200, 0.50))),

    "AlSi10Mg-T6": Material(
        key="AlSi10Mg-T6",
        name="Aluminium AlSi10Mg, laser powder-bed fused, stress-relieved + T6",
        density=2670.0, youngs_modulus=70.0e9, poisson=0.33,
        yield_strength=230.0e6, ultimate_strength=340.0e6,
        endurance_limit=95.0e6, thermal_conductivity=140.0, cte=21.0e-6,
        max_service_temp=c_to_k(200.0),
        derating=_d((20, 1.00), (100, 0.88), (150, 0.70), (200, 0.46)),
        source=("nominal LPBF values for a stress-relieved + T6 condition, "
                "machined surfaces. Printed aluminium is more supplier- and "
                "orientation-dependent than wrought: strength varies with "
                "build direction, and the endurance limit here assumes the "
                "surface has been MACHINED. As-built LPBF surfaces roughly "
                "halve it, which matters because a lattice cannot be "
                "machined -- see psrt.railrod.lattice")),
}


def get(key: str) -> Material:
    try:
        return MATERIALS[key]
    except KeyError:
        known = ", ".join(sorted(MATERIALS))
        raise KeyError(f"unknown material {key!r}; known: {known}") from None


def listing() -> list[Material]:
    return [MATERIALS[k] for k in sorted(MATERIALS)]
