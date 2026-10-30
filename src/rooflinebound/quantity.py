"""Quantities that carry how much they can be trusted.

Every number this package reports is a ``Quantity``: a value with a unit, a
*level* that says how it was obtained, an optional bound *direction*, and the
provenance of every input it was computed from.

Two rules are enforced here rather than left to callers:

1. **A result is never more certain than its weakest input.** Arithmetic
   returns the weakest level of its operands; there is no way to raise it.
2. **A bound keeps its direction honestly.** ``lower + lower`` is a lower
   bound, ``exact / lower`` is an upper bound, and combinations whose
   direction is unknowable (``lower - lower``, ``lower + upper``) raise
   instead of returning a misleading number.

All arithmetic assumes non-negative values, which holds for every physical
quantity the package deals with (bytes, FLOPs, seconds, counts).
"""

from __future__ import annotations

import enum
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal

__all__ = [
    "Level",
    "Trust",
    "Provenance",
    "Unit",
    "Quantity",
    "IndeterminateBound",
    "qmax",
    "qsum",
]


class IndeterminateBound(ValueError):
    """Raised when an operation would produce a bound of unknown direction."""


class Level(enum.IntEnum):
    """How a value was obtained, strongest first.

    The integer order is the trust order: a larger value is weaker, so the
    level of a derived quantity is simply ``max`` of its inputs' levels.
    """

    EXACT = 0  # arithmetic on model config and hardware spec, no assumption
    EXPECTED = 1  # exact expectation under a stated statistical assumption
    BOUND = 2  # provable inequality under stated assumptions
    ASSUMED = 3  # a modelling input the user must supply
    ESTIMATE = 4  # a bound scaled by user-supplied efficiency factors

    @property
    def tag(self) -> str:
        return self.name


class Trust(enum.IntEnum):
    """Where an *input* value came from, strongest first."""

    DEFINITION = 0  # a mathematical or unit constant
    PUBLIC_CONFIG = 1  # a published model config / checkpoint header
    OFFICIAL = 2  # a vendor-owned specification
    THIRD_PARTY = 3  # a public but not vendor-owned source
    USER = 4  # supplied by the user, unverified by this package


@dataclass(frozen=True)
class Provenance:
    """One input a quantity depends on."""

    ref: str  # e.g. "spec:h20-3e#mem_bw", "user:--mem-reserve-gb"
    trust: Trust
    source: str | None = None  # URL or file the value was read from

    def to_dict(self) -> dict:
        d = {"ref": self.ref, "trust": self.trust.name}
        if self.source:
            d["source"] = self.source
        return d


# --------------------------------------------------------------------- units


@dataclass(frozen=True)
class Unit:
    """A product of base units with integer exponents, e.g. ``FLOP/s``."""

    dims: tuple[tuple[str, int], ...] = ()

    @staticmethod
    def parse(text: str) -> Unit:
        """Parse ``"B"``, ``"FLOP/s"``, ``"B*token"``, ``"1"`` or ``""``."""
        text = text.strip()
        if text in ("", "1"):
            return Unit()
        num, _, den = text.partition("/")
        exps: dict = {}
        for part, sign in ((num, 1), (den, -1)):
            for name in filter(None, (p.strip() for p in part.split("*"))):
                if name == "1":
                    continue
                exps[name] = exps.get(name, 0) + sign
        return Unit._from(exps)

    @staticmethod
    def _from(exps: Mapping[str, int]) -> Unit:
        return Unit(tuple(sorted((k, v) for k, v in exps.items() if v)))

    def __mul__(self, other: Unit) -> Unit:
        exps = dict(self.dims)
        for k, v in other.dims:
            exps[k] = exps.get(k, 0) + v
        return Unit._from(exps)

    def __truediv__(self, other: Unit) -> Unit:
        return self * Unit(tuple((k, -v) for k, v in other.dims))

    def __str__(self) -> str:
        def side(items):
            return "*".join(k if abs(v) == 1 else f"{k}^{abs(v)}" for k, v in items)

        num = side([d for d in self.dims if d[1] > 0])
        den = side([d for d in self.dims if d[1] < 0])
        if not den:
            return num or "1"
        return f"{num or '1'}/{den}"


UnitLike = str | Unit


def _unit(u: UnitLike) -> Unit:
    return u if isinstance(u, Unit) else Unit.parse(u)


# ------------------------------------------------------------------ quantity

Direction = Literal["lower", "upper"] | None

_FLIP = {None: None, "lower": "upper", "upper": "lower"}

# Significant digits shown per level: no estimate looks more precise than it is.
_SIG_DIGITS = {
    Level.EXACT: None,  # full precision
    Level.EXPECTED: 4,
    Level.BOUND: 3,
    Level.ASSUMED: 3,
    Level.ESTIMATE: 2,
}


def _combine_dir(a: Direction, b: Direction) -> Direction:
    if a is None:
        return b
    if b is None or a == b:
        return a
    raise IndeterminateBound(
        f"combining a {a} bound with an {b} bound has no defined direction"
    )


@dataclass(frozen=True)
class Quantity:
    """A non-negative value with unit, level, bound direction and provenance.

    ``direction`` says which side the true value lies on: ``"lower"`` means
    the truth is >= ``value``; ``"upper"`` means it is <= ``value``.

    ``interval`` (EXPECTED values) is a guaranteed range around an expected
    ``value``, e.g. the fewest and most experts a batch can touch.
    """

    value: float
    unit: Unit = field(default_factory=Unit)
    level: Level = Level.EXACT
    direction: Direction = None
    interval: tuple[float, float] | None = None
    provenance: tuple[Provenance, ...] = ()
    derivation: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.unit, Unit):
            object.__setattr__(self, "unit", _unit(self.unit))
        if not math.isfinite(self.value) or self.value < 0:
            raise ValueError(f"quantities are finite and non-negative, got {self.value}")
        if self.direction is not None and self.level < Level.BOUND:
            raise ValueError(f"a {self.level.tag} value cannot carry a bound direction")
        if self.level == Level.BOUND and self.direction is None:
            raise ValueError("a BOUND needs a direction ('lower' or 'upper')")
        if self.interval is not None:
            lo, hi = self.interval
            if not lo <= self.value <= hi:
                raise ValueError(f"value {self.value} outside its interval {self.interval}")

    # ------------------------------------------------------------ constructors

    @classmethod
    def exact(cls, value: float, unit: UnitLike = "", *prov: Provenance,
              derivation: str | None = None) -> Quantity:
        return cls(float(value), _unit(unit), Level.EXACT, provenance=tuple(prov),
                   derivation=derivation)

    @classmethod
    def constant(cls, value: float, unit: UnitLike = "", name: str = "") -> Quantity:
        """A definitional constant (bytes per fp16 element, 2 FLOPs per MAC ...)."""
        prov = (Provenance(f"def:{name}", Trust.DEFINITION),) if name else ()
        return cls(float(value), _unit(unit), Level.EXACT, provenance=prov)

    @classmethod
    def expected(cls, value: float, unit: UnitLike, low: float, high: float,
                 *prov: Provenance, derivation: str | None = None) -> Quantity:
        return cls(float(value), _unit(unit), Level.EXPECTED,
                   interval=(float(low), float(high)), provenance=tuple(prov),
                   derivation=derivation)

    @classmethod
    def assumed(cls, value: float, unit: UnitLike, ref: str,
                trust: Trust = Trust.USER, source: str | None = None) -> Quantity:
        """A modelling input that theory cannot supply; always attributed."""
        return cls(float(value), _unit(unit), Level.ASSUMED,
                   provenance=(Provenance(ref, trust, source),))

    @classmethod
    def efficiency(cls, value: float, ref: str, source: str | None = None) -> Quantity:
        """A user-supplied efficiency factor in (0, 1]; taints results as ESTIMATE."""
        if not 0 < value <= 1:
            raise ValueError(f"efficiency factor {ref} must be in (0, 1], got {value}")
        return cls(float(value), Unit(), Level.ESTIMATE,
                   provenance=(Provenance(ref, Trust.USER, source),))

    # ---------------------------------------------------------------- helpers

    @property
    def trust(self) -> Trust | None:
        """Weakest trust among the inputs, ``None`` if there are none."""
        return max((p.trust for p in self.provenance), default=None)

    @property
    def bounds(self) -> tuple[float, float]:
        return self.interval if self.interval is not None else (self.value, self.value)

    def as_bound(self, direction: Literal["lower", "upper"],
                 derivation: str | None = None) -> Quantity:
        """Declare this value a bound on some true quantity.

        Used where a model is a provable bound, e.g. ``max(t_compute, t_mem)``
        at peak rates is a lower bound on step time. The level drops to BOUND
        unless an input already made it weaker.
        """
        if self.direction not in (None, direction):
            raise IndeterminateBound(f"a {self.direction} bound cannot be redeclared {direction}")
        return replace(self, level=max(self.level, Level.BOUND), direction=direction,
                       derivation=derivation or self.derivation)

    def to(self, unit: UnitLike, scale: float) -> Quantity:
        """Rescale into another unit, e.g. ``q.to("GB", 1e-9)``."""
        return self._derive(self.value * scale, _unit(unit), self.direction,
                            self._scaled_interval(scale), (self,))

    def with_derivation(self, text: str) -> Quantity:
        return replace(self, derivation=text)

    def _scaled_interval(self, k: float) -> tuple[float, float] | None:
        if self.interval is None:
            return None
        return (self.interval[0] * k, self.interval[1] * k)

    @staticmethod
    def _derive(value: float, unit: Unit, direction: Direction,
                interval: tuple[float, float] | None,
                operands: Iterable[Quantity]) -> Quantity:
        ops = list(operands)
        level = max(q.level for q in ops)
        if level == Level.ESTIMATE:
            # Scaling a bound by a fitted efficiency gives a guess, not a bound.
            direction = None
        elif direction is not None:
            level = max(level, Level.BOUND)
        if level < Level.EXPECTED or (interval is not None and interval[0] == interval[1]):
            interval = None
        prov: dict = {}
        for q in ops:
            for p in q.provenance:
                prov.setdefault(p.ref, p)
        return Quantity(value, unit, level, direction, interval, tuple(prov.values()))

    @staticmethod
    def _coerce(other: Any) -> Quantity:
        if isinstance(other, Quantity):
            return other
        if isinstance(other, (int, float)):
            # A bare number in an expression is a pure scale such as 2 FLOPs/MAC;
            # anything with physical meaning must be a Quantity with provenance.
            return Quantity.constant(other)
        return NotImplemented

    # -------------------------------------------------------------- arithmetic

    def __add__(self, other: Any) -> Quantity:
        o = self._coerce(other)
        if o is NotImplemented:
            return NotImplemented
        if self.unit != o.unit:
            raise ValueError(f"cannot add {self.unit} and {o.unit}")
        (al, ah), (bl, bh) = self.bounds, o.bounds
        return self._derive(self.value + o.value, self.unit,
                            _combine_dir(self.direction, o.direction),
                            (al + bl, ah + bh), (self, o))

    __radd__ = __add__

    def __sub__(self, other: Any) -> Quantity:
        o = self._coerce(other)
        if o is NotImplemented:
            return NotImplemented
        if self.unit != o.unit:
            raise ValueError(f"cannot subtract {o.unit} from {self.unit}")
        (al, ah), (bl, bh) = self.bounds, o.bounds
        return self._derive(self.value - o.value, self.unit,
                            _combine_dir(self.direction, _FLIP[o.direction]),
                            (max(al - bh, 0.0), max(ah - bl, 0.0)), (self, o))

    def __rsub__(self, other: Any) -> Quantity:
        o = self._coerce(other)
        return NotImplemented if o is NotImplemented else o.__sub__(self)

    def __mul__(self, other: Any) -> Quantity:
        o = self._coerce(other)
        if o is NotImplemented:
            return NotImplemented
        (al, ah), (bl, bh) = self.bounds, o.bounds
        return self._derive(self.value * o.value, self.unit * o.unit,
                            _combine_dir(self.direction, o.direction),
                            (al * bl, ah * bh), (self, o))

    __rmul__ = __mul__

    def __truediv__(self, other: Any) -> Quantity:
        o = self._coerce(other)
        if o is NotImplemented:
            return NotImplemented
        (al, ah), (bl, bh) = self.bounds, o.bounds
        if bl <= 0:
            raise ZeroDivisionError("division by a quantity that may be zero")
        return self._derive(self.value / o.value, self.unit / o.unit,
                            _combine_dir(self.direction, _FLIP[o.direction]),
                            (al / bh, ah / bl), (self, o))

    def __rtruediv__(self, other: Any) -> Quantity:
        o = self._coerce(other)
        return NotImplemented if o is NotImplemented else o.__truediv__(self)

    # ------------------------------------------------------------------ output

    def format(self, precision: int | None = None) -> str:
        """Human text that cannot be mistaken for a stronger claim.

        ``≥ 1.23 s [BOUND]``, ``≈ 18 ms [ESTIMATE]``, ``223 [EXPECTED 64..256]``.
        """
        sig = precision if precision is not None else _SIG_DIGITS[self.level]
        prefix = {"lower": "≥ ", "upper": "≤ "}.get(self.direction or "", "")
        if self.level == Level.ESTIMATE:
            prefix = "≈ " + prefix
        unit = "" if str(self.unit) == "1" else f" {self.unit}"
        tag = self.level.tag
        if self.interval is not None:
            tag += f" {_fmt(self.interval[0], sig)}..{_fmt(self.interval[1], sig)}"
        return f"{prefix}{_fmt(self.value, sig)}{unit} [{tag}]"

    def __str__(self) -> str:
        return self.format()

    def to_dict(self) -> dict:
        """JSON form. There is deliberately no way to emit a bare number."""
        d: dict = {
            "value": self.value,
            "unit": str(self.unit),
            "level": self.level.tag,
            "direction": self.direction,
            "provenance": [p.to_dict() for p in self.provenance],
        }
        if self.interval is not None:
            d["interval"] = list(self.interval)
        if self.derivation:
            d["derivation"] = self.derivation
        return d


def _fmt(x: float, sig: int | None) -> str:
    if sig is None:
        x = float(f"{x:.12g}")  # drop binary noise from unit scaling, keep real digits
        return f"{int(x):,}" if x.is_integer() else repr(x)
    if x == 0:
        return "0"
    return f"{x:.{sig}g}" if abs(x) < 10 ** sig else f"{round(x, sig - len(str(int(abs(x))))):,.0f}"


def qmax(*qs: Quantity) -> Quantity:
    """Max of quantities with the same unit.

    The max of lower bounds is a lower bound of the max of the true values
    (and likewise for upper bounds), so the direction rule is the same as for
    addition.
    """
    if not qs:
        raise ValueError("qmax() needs at least one quantity")
    unit = qs[0].unit
    if any(q.unit != unit for q in qs):
        raise ValueError("qmax() needs quantities of one unit")
    direction: Direction = None
    for q in qs:
        direction = _combine_dir(direction, q.direction)
    lo = max(q.bounds[0] for q in qs)
    hi = max(q.bounds[1] for q in qs)
    return Quantity._derive(max(q.value for q in qs), unit, direction, (lo, hi), qs)


def qsum(qs: Iterable[Quantity], unit: UnitLike = "") -> Quantity:
    """Sum of quantities; an empty sum is an exact zero of ``unit``."""
    total = Quantity.exact(0, unit)
    for q in qs:
        total = total + q
    return total
