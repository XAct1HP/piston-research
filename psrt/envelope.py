"""The manufacturing envelope: what the block will actually allow.

Every other component in this tool can be designed freely. The block cannot.
It is an existing object, and machining is subtractive -- material comes off,
never on -- so the bore has a hard floor at its as-built size and a ceiling
set by whichever internal feature the wall reaches first.

This module computes that ceiling. It does not return a number on its own:
it returns every limit it could evaluate, which one binds, and *how much the
answer should be trusted*, because the inputs are not equally knowable.

    Tier 1  bore, stroke, deck height, liner type
            Manufacturer specs and service data. Findable, reliable.

    Tier 2  the largest bore an aftermarket piston is sold for
            If a piston maker tools up for a +1.0 mm piston, the industry has
            already established that the overbore survives in the field. This
            is second-hand evidence and it is better than anything this tool
            could derive from first principles.

    Tier 3  coolant jacket geometry, oil gallery positions, the real wall
            thickness at the thin spot
            Not published by anybody. Either sonic-test the block, section
            one, or accept an assumption -- and an assumption that decides
            whether you break into an oil gallery deserves to be shouted
            about rather than buried in a tooltip.

A limit whose inputs are missing is reported as UNKNOWN, never skipped. The
difference matters: a skipped limit silently raises the ceiling, and the
limit nobody could evaluate is exactly the one likely to bite.

One operation escapes the subtractive rule. Resleeving -- boring the block
oversize and pressing in a thicker ductile-iron sleeve -- adds material back
and resets the envelope completely. It is a different class of work with its
own cost and risk, so it is a mode the user switches on deliberately, never
something an optimiser is allowed to discover on its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["Limit", "Envelope", "envelope", "TIERS", "tier_of"]


# How much to trust a number, by where it came from. The wording is what the
# user sees next to the limit, so it says what to DO about it.
TIERS: dict[str, tuple[int, str]] = {
    "measured": (1, "measured on this block"),
    "published": (1, "manufacturer specification or service data"),
    "catalogue": (2, "aftermarket piston availability -- the industry's own "
                     "verdict on what this block tolerates"),
    "derived": (1, "computed from other parameters"),
    "estimated": (3, "ASSUMED -- sonic-test the block before cutting"),
    "user": (3, "entered by hand, unverified"),
}


def tier_of(state, *paths: str) -> tuple[int, str]:
    """The tier of the least trustworthy input among ``paths``.

    A limit is only as good as its weakest input, so a bore-to-bore limit
    built from a published spacing and an assumed wall thickness is a tier 3
    number, not a tier 1 one.
    """
    worst = (0, "")
    for path in paths:
        try:
            source = state.param(path).source
        except Exception:                       # unknown parameter
            continue
        tier, why = TIERS.get(source, (3, "unknown provenance"))
        if tier > worst[0]:
            worst = (tier, why)
    return worst if worst[0] else (3, "unknown provenance")


@dataclass
class Limit:
    """One ceiling on the bore, and how much to believe it."""

    name: str
    max_bore: float | None          # None means it could not be evaluated
    reason: str
    tier: int = 3
    provenance: str = ""
    inputs: dict = field(default_factory=dict)
    missing: tuple = ()              # parameters that would let it be computed

    @property
    def known(self) -> bool:
        return self.max_bore is not None

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "max_bore_m": self.max_bore,
            "reason": self.reason,
            "tier": self.tier,
            "provenance": self.provenance,
            "inputs": dict(self.inputs),
            "missing": list(self.missing),
            "known": self.known,
        }


@dataclass
class Envelope:
    """Every limit on the bore, which one binds, and what is still unknown."""

    as_built: float
    current: float
    limits: list
    resleeved: bool = False
    notes: list = field(default_factory=list)

    @property
    def known_limits(self) -> list:
        return [limit for limit in self.limits if limit.known]

    @property
    def unknown_limits(self) -> list:
        return [limit for limit in self.limits if not limit.known]

    @property
    def binding(self):
        """The limit that governs, or None if nothing could be evaluated."""
        known = self.known_limits
        if not known:
            return None
        return min(known, key=lambda limit: limit.max_bore)

    @property
    def max_bore(self) -> float | None:
        binding = self.binding
        return binding.max_bore if binding else None

    @property
    def min_bore(self) -> float | None:
        """Resleeving adds material back, so the floor disappears with it."""
        return None if self.resleeved else self.as_built

    @property
    def headroom(self) -> float | None:
        """How much further this bore can go, in metres of diameter."""
        if self.max_bore is None:
            return None
        return self.max_bore - self.current

    @property
    def overbore_available(self) -> float | None:
        """Total overbore the block allows, from as-built."""
        if self.max_bore is None:
            return None
        return self.max_bore - self.as_built

    @property
    def confidence(self) -> int:
        """The tier of the binding limit: how much the ceiling is worth."""
        binding = self.binding
        return binding.tier if binding else 3

    @property
    def trustworthy(self) -> bool:
        return self.confidence <= 2

    def summary(self) -> str:
        if self.max_bore is None:
            return ("no limit could be evaluated: the block model is missing "
                    "every input that would set one")
        binding = self.binding
        head = (f"{self.max_bore * 1000:.2f} mm maximum bore "
                f"({self.overbore_available * 1000:+.2f} mm over as-built), "
                f"set by {binding.name}")
        if self.confidence == 3:
            head += " -- from an ASSUMED input, not a measured one"
        return head

    def as_dict(self) -> dict:
        return {
            "as_built_bore_m": self.as_built,
            "current_bore_m": self.current,
            "max_bore_m": self.max_bore,
            "min_bore_m": self.min_bore,
            "headroom_m": self.headroom,
            "overbore_available_m": self.overbore_available,
            "binding": self.binding.as_dict() if self.binding else None,
            "confidence_tier": self.confidence,
            "resleeving": self.resleeved,
            "limits": [limit.as_dict() for limit in self.limits],
            "unknown": [limit.name for limit in self.unknown_limits],
            "notes": list(self.notes),
            "summary": self.summary(),
        }


# --------------------------------------------------------------------------
# the individual limits
# --------------------------------------------------------------------------

def _have(state, *paths: str) -> bool:
    return all(state.has(path) and state.get(path) is not None
               for path in paths)


def _bore_to_bore(state) -> Limit:
    """The neighbouring cylinder, through whatever is between them."""
    siamesed = bool(state.get("block.siamesed", False))
    wall_path = ("block.min_siamesed_web" if siamesed
                 else "block.min_wall_to_coolant")
    name = "bore-to-bore web" if siamesed else "bore-to-bore coolant wall"

    if not _have(state, "block.bore_spacing", wall_path):
        return Limit(name, None,
                     "needs the bore spacing and the minimum wall",
                     missing=("block.bore_spacing", wall_path))

    spacing = state["block.bore_spacing"]
    wall = state[wall_path]
    tier, why = tier_of(state, "block.bore_spacing", wall_path)
    detail = ("no coolant between the bores, so the web is structural"
              if siamesed else "coolant passage between the bores")
    return Limit(
        name, spacing - 2.0 * wall,
        f"{spacing * 1000:.1f} mm centres less two {wall * 1000:.1f} mm "
        f"walls ({detail})",
        tier, why,
        {"bore_spacing_m": spacing, "min_wall_m": wall, "siamesed": siamesed})


def _liner(state) -> Limit:
    """What is left of the liner once the boring bar has been through it."""
    liner_type = state.get("block.liner_type", "cast-in-iron")

    if liner_type == "coated-aluminium":
        # Nikasil, Alusil and PTWA are microns thick. The first cut removes
        # the running surface entirely and exposes aluminium that cannot run
        # against a ring. There is no overbore without replating or sleeving.
        as_built = state["block.as_built_bore"]
        tier, why = tier_of(state, "block.as_built_bore")
        return Limit(
            "plated bore", as_built,
            "a plated aluminium bore has a running surface microns thick: "
            "boring removes it entirely and exposes aluminium no ring can "
            "run on. Overbore requires replating or sleeving, not cutting",
            tier, why, {"liner_type": liner_type})

    if not _have(state, "block.liner_thickness", "block.min_liner_wall"):
        return Limit(
            "liner wall", None,
            "needs the liner thickness, which is not usually published",
            missing=("block.liner_thickness",))

    as_built = state["block.as_built_bore"]
    thickness = state["block.liner_thickness"]
    minimum = state["block.min_liner_wall"]
    tier, why = tier_of(state, "block.as_built_bore", "block.liner_thickness",
                        "block.min_liner_wall")
    return Limit(
        "liner wall", as_built + 2.0 * (thickness - minimum),
        f"{thickness * 1000:.1f} mm liner, leaving {minimum * 1000:.1f} mm "
        "after boring; past this the bar is into the parent block",
        tier, why,
        {"liner_thickness_m": thickness, "min_liner_wall_m": minimum})


def _clearance(state, offset_path: str, min_path: str, name: str,
               consequence: str) -> Limit:
    """A generic 'do not cut into that' limit at a known radial offset."""
    if not _have(state, offset_path, min_path):
        return Limit(name, None,
                     f"needs {offset_path.split('.')[-1].replace('_', ' ')}, "
                     "which is not published -- sonic-test or section a block",
                     missing=(offset_path,))

    offset = state[offset_path]
    minimum = state[min_path]
    tier, why = tier_of(state, offset_path, min_path)
    return Limit(
        name, 2.0 * (offset - minimum),
        f"{offset * 1000:.1f} mm from bore centre, leaving "
        f"{minimum * 1000:.1f} mm of material: {consequence}",
        tier, why, {"offset_m": offset, "min_wall_m": minimum})


def _catalogue(state) -> Limit:
    """What pistons are actually sold for this block.

    Worth its own limit because it is evidence of a different kind: not a
    calculation but a record of what has already survived in service.
    """
    if not _have(state, "block.catalogue_max_bore"):
        return Limit(
            "aftermarket piston availability", None,
            "no catalogue bore recorded; the largest piston sold for this "
            "block is the most reliable ceiling available and is worth "
            "looking up",
            missing=("block.catalogue_max_bore",))

    value = state["block.catalogue_max_bore"]
    tier, why = tier_of(state, "block.catalogue_max_bore")
    as_built = state["block.as_built_bore"]
    return Limit(
        "aftermarket piston availability", value,
        f"pistons are sold up to {value * 1000:.2f} mm "
        f"({(value - as_built) * 1000:+.2f} mm over), so that overbore is "
        "established practice on this block",
        tier, why, {"catalogue_max_bore_m": value})


def _resleeved(state) -> list:
    """With sleeves going in, a different set of things limit the bore.

    The naive model -- take the unsleeved ceiling and subtract two sleeve
    walls -- makes resleeving come out WORSE than the stock block, which is
    obviously wrong or nobody would pay for it. Two things were missing.

    First, interlocking sleeves become the structure between the bores. They
    can be pressed until they very nearly touch, so the coolant-wall rule
    between cylinders is replaced by a sleeve-to-sleeve gap that is close to
    zero. That is what turns a non-siamesed block into a siamesed one and it
    is where most of the gain comes from.

    Second, the parent metal that remains does less work once a ductile iron
    sleeve is carrying the load, so the wall it has to keep is thinner than
    the unsleeved minimum.

    What resleeving does NOT do is help with anything outside the bore: an
    oil gallery is where it is, and a sleeve big enough to reach it still
    scraps the block.
    """
    wall = state.get("block.replacement_sleeve_wall", 0.003)
    limits = []

    if _have(state, "block.bore_spacing", "block.min_sleeve_to_sleeve"):
        spacing = state["block.bore_spacing"]
        gap = state["block.min_sleeve_to_sleeve"]
        tier, why = tier_of(state, "block.bore_spacing",
                            "block.min_sleeve_to_sleeve",
                            "block.replacement_sleeve_wall")
        limits.append(Limit(
            "sleeve-to-sleeve", spacing - gap - 2.0 * wall,
            f"{spacing * 1000:.1f} mm centres less a {gap * 1000:.1f} mm gap "
            f"between sleeves and two {wall * 1000:.1f} mm sleeve walls; the "
            "sleeves themselves become the structure between the bores",
            tier, why,
            {"bore_spacing_m": spacing, "sleeve_gap_m": gap,
             "sleeve_wall_m": wall}))
    else:
        limits.append(Limit(
            "sleeve-to-sleeve", None,
            "needs the bore spacing and the minimum sleeve-to-sleeve gap",
            missing=("block.bore_spacing", "block.min_sleeve_to_sleeve")))

    min_wall = state.get("block.min_wall_when_sleeved", 0.0015)
    for offset_path, name, consequence in (
            ("block.gallery_offset", "oil gallery",
             "breaking into it scraps the block, sleeve or no sleeve"),
            ("block.head_bolt_offset", "head bolt boss",
             "the boss still carries clamping load into the deck")):
        if not _have(state, offset_path):
            limits.append(Limit(
                name, None,
                f"needs {offset_path.split('.')[-1].replace('_', ' ')}, "
                "which is not published -- sonic-test or section a block",
                missing=(offset_path,)))
            continue
        offset = state[offset_path]
        tier, why = tier_of(state, offset_path, "block.min_wall_when_sleeved")
        limits.append(Limit(
            name, 2.0 * (offset - min_wall) - 2.0 * wall,
            f"{offset * 1000:.1f} mm from bore centre, leaving "
            f"{min_wall * 1000:.1f} mm of parent metal around a "
            f"{wall * 1000:.1f} mm sleeve: {consequence}",
            tier, why,
            {"offset_m": offset, "min_wall_m": min_wall,
             "sleeve_wall_m": wall}))

    return limits


def envelope(state) -> Envelope:
    """Everything the block allows, and how much of it is actually known."""
    as_built = state["block.as_built_bore"]
    current = state["engine.bore"]
    resleeving = bool(state.get("block.resleeving_allowed", False))
    notes: list[str] = []

    if resleeving:
        limits = _resleeved(state)
        notes.append(
            "resleeving mode: the bore may go BELOW as-built as well as "
            "above, because a pressed-in sleeve adds material back. This is "
            "a different class of work -- machining the block oversize, "
            "pressing and re-honing -- with its own cost and its own risk of "
            "distortion, and it is on only because it was switched on.")
    else:
        limits = [
            _bore_to_bore(state),
            _liner(state),
            _clearance(state, "block.gallery_offset",
                       "block.min_wall_to_gallery", "oil gallery",
                       "breaking into it scraps the block"),
            _clearance(state, "block.head_bolt_offset",
                       "block.min_wall_to_head_bolt", "head bolt boss",
                       "the bore distorts under clamping load"),
            _catalogue(state),
        ]

    result = Envelope(as_built=as_built, current=current, limits=limits,
                      resleeved=resleeving, notes=notes)

    unknown = result.unknown_limits
    if unknown:
        notes.append(
            f"{len(unknown)} limit(s) could not be evaluated "
            f"({', '.join(limit.name for limit in unknown)}). The ceiling "
            "below is therefore an upper bound on an upper bound: one of "
            "these could be tighter than anything that was computed.")

    binding = result.binding
    if binding is not None and binding.tier == 3:
        notes.append(
            f"the binding limit rests on an assumed number "
            f"({binding.name}). Sonic-test the block before cutting to it.")

    if result.max_bore is not None and result.max_bore < current:
        notes.append(
            f"the current bore of {current * 1000:.2f} mm is ALREADY past "
            f"this ceiling by {(current - result.max_bore) * 1000:.2f} mm.")

    return result
