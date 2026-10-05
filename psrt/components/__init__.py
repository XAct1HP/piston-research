"""Component structural models.

Each module exposes ``evaluate(ctx) -> list[Margin]`` and is a pure function of
the design state, the load sweep and the thermal map. Adding a component means
writing one module and adding it to ``COMPONENTS`` below; nothing else in the
tool has to change.

Every model here is closed-form and runs in microseconds, which is what lets
the optimiser in phase 7 evaluate thousands of designs. The cost is that novel
geometry -- exactly where a redesign is most likely to fail -- falls outside
what a textbook formula can capture. Phase 6 addresses that by fitting notch
factors from real FEA runs and folding them back into these same models.
"""

from __future__ import annotations

from . import (big_end, bolts, crown, pin, railrod, rings, rod, skirt, sleeve,
               small_end)
from ..margins import ComponentContext, MarginReport

COMPONENTS = (sleeve, crown, rings, skirt, pin, small_end, rod, bolts, big_end)


def evaluate_all(state, sweep, thermal_map) -> MarginReport:
    """Run every component model. Pure function."""
    ctx = ComponentContext(state, sweep, thermal_map)
    margins = []
    rail = state.has("railrod.enabled") and state["railrod.enabled"]
    for module in COMPONENTS:
        # The rail rod replaces the I-beam shank and its two cap bolts; the
        # bearing and the small end stay, since the bore and the bush do.
        if rail and module in (rod, bolts):
            continue
        margins.extend(module.evaluate(ctx))
    if rail:
        margins.extend(railrod.evaluate(ctx))
    return MarginReport(margins=margins, thermal=thermal_map)
