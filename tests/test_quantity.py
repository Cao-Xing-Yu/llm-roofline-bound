import json

import pytest

from rooflinebound.quantity import (
    IndeterminateBound,
    Level,
    Provenance,
    Quantity,
    Trust,
    Unit,
    qmax,
    qsum,
)

SPEC = Provenance("spec:test#bw", Trust.OFFICIAL, "https://example.com/datasheet")


def lower(v, unit="s"):
    return Quantity.exact(v, unit).as_bound("lower")


def upper(v, unit="s"):
    return Quantity.exact(v, unit).as_bound("upper")


# ------------------------------------------------------------------- units


def test_unit_algebra():
    assert str(Unit.parse("FLOP") / Unit.parse("B")) == "FLOP/B"
    assert str(Unit.parse("B") / Unit.parse("B/s")) == "s"
    assert str(Unit.parse("B*token") * Unit.parse("1/token")) == "B"
    assert str(Unit.parse("") ) == "1"
    assert Unit.parse("B*s/s") == Unit.parse("B")


def test_add_requires_same_unit():
    with pytest.raises(ValueError):
        Quantity.exact(1, "B") + Quantity.exact(1, "s")


# ------------------------------------------------------- level propagation


@pytest.mark.parametrize("a_level", list(Level))
@pytest.mark.parametrize("b_level", list(Level))
def test_result_is_never_stronger_than_weakest_input(a_level, b_level):
    def make(level):
        if level == Level.EXACT:
            return Quantity.exact(0.5)
        if level == Level.EXPECTED:
            return Quantity.expected(0.5, "", 0.25, 0.75)
        if level == Level.BOUND:
            return Quantity.exact(0.5).as_bound("lower")
        if level == Level.ASSUMED:
            return Quantity.assumed(0.5, "", "user:--x")
        return Quantity.efficiency(0.5, "user:--eff")

    a, b = make(a_level), make(b_level)
    for r in (a * b, a + b, qmax(a, b)):
        assert r.level == max(a_level, b_level)


def test_level_cannot_be_raised_by_as_bound():
    est = Quantity.exact(10, "s") / Quantity.efficiency(0.5, "user:--mfu")
    assert est.level == Level.ESTIMATE
    assert est.as_bound("lower").level == Level.ESTIMATE


def test_estimate_drops_bound_direction():
    est = lower(1.0) / Quantity.efficiency(0.6, "user:--mbu")
    assert est.level == Level.ESTIMATE
    assert est.direction is None
    assert est.format().startswith("≈ ")


def test_exact_has_no_direction():
    with pytest.raises(ValueError):
        Quantity(1.0, Unit(), Level.EXACT, direction="lower")
    with pytest.raises(ValueError):
        Quantity(1.0, Unit(), Level.BOUND)


# ------------------------------------------------------ bound directions


def test_bound_directions():
    assert (lower(1) + lower(2)).direction == "lower"
    assert (lower(1) + Quantity.exact(2, "s")).direction == "lower"
    assert (lower(3) - Quantity.exact(1, "s")).direction == "lower"
    assert (Quantity.exact(3, "s") - lower(1)).direction == "upper"
    assert (lower(1) - upper(0.5)).direction == "lower"
    assert (Quantity.exact(1, "token") / lower(0.01)).direction == "upper"
    assert qmax(lower(1), lower(2)).direction == "lower"


@pytest.mark.parametrize("expr", [
    lambda: lower(1) + upper(2),
    lambda: lower(2) - lower(1),
    lambda: lower(2) / lower(1),
    lambda: qmax(lower(1), upper(2)),
])
def test_indeterminate_bounds_raise(expr):
    with pytest.raises(IndeterminateBound):
        expr()


# --------------------------------------------------------------- intervals


def test_expected_interval_propagates():
    experts = Quantity.expected(223.0, "expert", 64, 256)
    per_expert = Quantity.exact(1e6, "B/expert")
    b = experts * per_expert
    assert b.level == Level.EXPECTED
    assert b.value == 223e6 and b.interval == (64e6, 256e6)
    assert str(b.unit) == "B"


def test_interval_dropped_when_exact():
    q = Quantity.exact(2, "B") * 3
    assert q.interval is None and q.level == Level.EXACT


def test_value_must_lie_in_interval():
    with pytest.raises(ValueError):
        Quantity.expected(300, "", 64, 256)


def test_negative_rejected():
    with pytest.raises(ValueError):
        Quantity.exact(-1, "B")


# -------------------------------------------------------------- provenance


def test_provenance_is_collected_and_deduplicated():
    bw = Quantity.exact(4e12, "B/s", SPEC)
    t = Quantity.exact(1e9, "B") / bw + Quantity.exact(1e9, "B") / bw
    assert [p.ref for p in t.provenance] == ["spec:test#bw"]
    assert t.trust == Trust.OFFICIAL


def test_trust_is_weakest_input():
    a = Quantity.exact(1, "B", SPEC)
    b = Quantity.assumed(1, "B", "user:--reserve")
    assert (a + b).trust == Trust.USER


def test_efficiency_range_checked():
    with pytest.raises(ValueError):
        Quantity.efficiency(1.2, "user:--mbu")
    with pytest.raises(ValueError):
        Quantity.efficiency(0.0, "user:--mbu")


# ------------------------------------------------------------------ output


def test_format_marks_level_and_direction():
    assert lower(1.23456).format() == "≥ 1.23 s [BOUND]"
    assert upper(850.4, "token/s").format() == "≤ 850 token/s [BOUND]"
    assert Quantity.exact(753_330_000, "param").format() == "753,330,000 param [EXACT]"
    assert Quantity.expected(223.4, "expert", 64, 256).format() == "223.4 expert [EXPECTED 64..256]"
    est = Quantity.exact(18.37, "ms") / Quantity.efficiency(1.0, "user:--mbu")
    assert est.format() == "≈ 18 ms [ESTIMATE]"


def test_large_values_rounded_to_significant_digits():
    assert upper(123_456, "token").format() == "≤ 123,000 token [BOUND]"


def test_json_never_emits_bare_numbers():
    q = Quantity.exact(1e9, "B", SPEC) / Quantity.efficiency(0.5, "user:--mbu")
    d = json.loads(json.dumps(q.to_dict()))
    assert set(d) >= {"value", "unit", "level", "direction", "provenance"}
    assert d["level"] == "ESTIMATE"
    assert {p["ref"] for p in d["provenance"]} == {"spec:test#bw", "user:--mbu"}


def test_qsum_empty_is_exact_zero():
    z = qsum([], "B")
    assert z.value == 0 and z.level == Level.EXACT and str(z.unit) == "B"
