"""A coarse temperature map for the piston.

Why this exists: 2618-T61 keeps 88% of its room-temperature yield strength at
150 C and 28% at 300 C. A piston crown runs hotter than 300 C. Sizing a crown
against room-temperature properties overstates its strength by roughly a
factor of three, so temperature cannot simply be ignored.

Why it is coarse: doing this properly needs conjugate heat transfer and a
transient thermal FEA, which is well outside phase 2. What happens instead is
a lumped-conductance network -- one node per region, a handful of named
conductances to the coolant and the oil, driven by the heat rejection the
cycle model already computes.

    q_crown = crown_heat_share x heat_loss_fraction x fuel energy rate

    T_underside = T_sink + q_crown / (G_rings + G_skirt + G_oil)
    T_crown_top = T_underside + q_crown x t_crown / (k x A_bore)

The conductances are calibration constants, labelled ``estimated`` in the
design state. They are set so a naturally aspirated engine at full load lands
near 300 C at the crown, 200 C at the ring belt and 150 C at the skirt, which
is where measurements on real pistons put them. Treat the absolute numbers as
indicative and the *trends* as meaningful: a thicker crown does run hotter on
its face, a bigger bore does reject more heat, and both of those move the
right way here.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import materials as materials_mod
from .units import k_to_c


@dataclass(frozen=True)
class ThermalMap:
    """Steady-state temperatures around the piston, in kelvin."""

    crown_top: float
    crown_underside: float
    top_land: float
    ring_belt: float
    skirt: float
    liner_at_tdc: float
    pin: float
    rod: float
    heat_to_piston: float        # W
    crown_heat_flux: float       # W/m^2
    crown_gradient: float        # K across the crown thickness
    note: str = ("lumped-conductance estimate, not a thermal analysis; "
                 "absolute values are indicative, trends are meaningful")

    def as_dict(self) -> dict:
        return {
            "crown_top_c": k_to_c(self.crown_top),
            "crown_underside_c": k_to_c(self.crown_underside),
            "top_land_c": k_to_c(self.top_land),
            "ring_belt_c": k_to_c(self.ring_belt),
            "skirt_c": k_to_c(self.skirt),
            "liner_at_tdc_c": k_to_c(self.liner_at_tdc),
            "pin_c": k_to_c(self.pin),
            "rod_c": k_to_c(self.rod),
            "heat_to_piston_w": self.heat_to_piston,
            "crown_heat_flux_w_m2": self.crown_heat_flux,
            "crown_gradient_k": self.crown_gradient,
            "note": self.note,
        }


def compute(state, sweep) -> ThermalMap:
    """Temperatures for the current operating point. Pure function."""
    trace = sweep.trace

    # Fuel energy rate per cylinder. A four-stroke fires once every two
    # revolutions, so cycles per second is speed / (4 pi).
    cycles_per_second = sweep.speed / (4.0 * 3.141592653589793)
    fuel_energy_per_cycle = (
        trace.fuel_mass * state["operating.fuel_lhv"]
        * state["operating.combustion_efficiency"])
    heat_to_walls = (fuel_energy_per_cycle
                     * state["operating.heat_loss_fraction"]
                     * cycles_per_second)
    q_piston = heat_to_walls * state["thermal.crown_heat_share"]

    conductance = (state["thermal.conductance_rings"]
                   + state["thermal.conductance_skirt"]
                   + state["thermal.conductance_oil"])

    # Sink temperature, weighted by which path carries the heat.
    t_coolant = state["thermal.coolant_temperature"]
    t_oil = state["thermal.oil_temperature"]
    g_cool = (state["thermal.conductance_rings"]
              + state["thermal.conductance_skirt"])
    g_oil = state["thermal.conductance_oil"]
    t_sink = (g_cool * t_coolant + g_oil * t_oil) / (g_cool + g_oil)

    underside = t_sink + q_piston / conductance

    area = state["engine.bore_area"]
    flux = q_piston / area
    crown_material = materials_mod.get(state["materials.piston"])
    gradient = flux * state["piston.crown_thickness"] / crown_material.thermal_conductivity
    crown_top = underside + gradient

    # Regional temperatures fall off along the heat paths out of the crown.
    # These fractions are calibration, not derivation.
    top_land = t_sink + (crown_top - t_sink) * 0.72
    ring_belt = t_sink + (crown_top - t_sink) * 0.52
    skirt = t_sink + (crown_top - t_sink) * 0.30

    # The liner at the top ring reversal point runs close to the ring belt.
    # The pin and rod sit in the oil and stay much cooler than the crown.
    liner_at_tdc = ring_belt
    pin_temp = t_oil + 0.30 * (underside - t_oil)
    rod_temp = t_oil + 0.15 * (underside - t_oil)

    return ThermalMap(
        crown_top=crown_top, crown_underside=underside, top_land=top_land,
        ring_belt=ring_belt, skirt=skirt, liner_at_tdc=liner_at_tdc,
        pin=pin_temp, rod=rod_temp, heat_to_piston=q_piston,
        crown_heat_flux=flux, crown_gradient=gradient)
