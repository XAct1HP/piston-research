"""Piston System Research Tool.

A parametric design and analysis tool for the reciprocating load chain of a
combustion engine: cylinder sleeve, piston, rings, gudgeon pin, small end and
connecting rod.

Design principle: ``evaluate(design_state) -> metrics`` is a pure function.
No hidden state, no side effects, fully deterministic. Everything else in the
tool -- the optimizer, the AI tool layer, undo, A/B comparison -- depends on
that property holding.

All internal quantities are strict SI: metres, kilograms, seconds, pascals,
newtons, joules, kelvin, radians. Conversion happens only at the display edge
(see :mod:`psrt.units`).
"""

__version__ = "0.1.0"
