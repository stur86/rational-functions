import numpy as np
from numpy.polynomial import Polynomial
import pytest
from dataclasses import dataclass
from rational_functions.integral import (
    RationalFunctionIntegral,
    _int_cconj_pair,
    _find_cconj_pairs,
)
from rational_functions.terms import (
    RationalIntegralLogTerm,
    RationalIntegralGeneralTerm,
    RationalTerm,
    RationalIntegralArctanTerm,
    RationalIntegralLogPairTerm,
)
from .conftest import ComparisonMethods


@dataclass(frozen=True)
class TermPair:
    f: tuple[RationalTerm]
    intf: tuple[RationalIntegralGeneralTerm]


# Test pairs of rational terms and their corresponding integrals
_term_int_pairs: list[TermPair] = [
    TermPair((RationalTerm(2.0, 1.5, 1),), (RationalIntegralLogTerm(1.5, 2),)),
    TermPair((RationalTerm(2.0, 1.5j, 1),), (RationalIntegralLogTerm(1.5j, 2),)),
    TermPair((RationalTerm(2.0j, 1.5, 1),), (RationalIntegralLogTerm(1.5, 2.0j),)),
    TermPair((RationalTerm(2.0, 1.5, 2),), (RationalTerm(2.0, -1.5, 1),)),
    TermPair(
        (RationalTerm(1.0 + 0.5j, 1.0), RationalTerm(1.0 - 0.5j, 1.0)),
        (RationalIntegralLogPairTerm(2.0, 1.0 + 0.5j),),
    ),
    TermPair(
        (RationalTerm(1.0 + 0.5j, 1.0j), RationalTerm(1.0 - 0.5j, -1.0j)),
        (RationalIntegralArctanTerm(-1.0, 1.0 + 0.5j),),
    ),
    TermPair(
        (RationalTerm(1.0 + 0.5j, 1.0), RationalTerm(1.0 - 0.5j, 0.5)),
        (
            RationalIntegralLogPairTerm(1.5, 1.0 + 0.5j),
            RationalIntegralArctanTerm(0.25j, 1.0 + 0.5j),
        ),
    ),
]


@pytest.mark.filterwarnings("ignore:All terms are RationalTerms")
@pytest.mark.parametrize("int_pair", _term_int_pairs)
def test_ratint_basic(int_pair: TermPair) -> None:
    """Test basic functionality of RationalFunctionIntegral."""
    int_terms = int_pair.intf
    ratint = RationalFunctionIntegral(int_terms)

    assert isinstance(ratint, RationalFunctionIntegral)
    assert len(ratint._terms) == len(int_terms)

    x = np.linspace(-1, 1, 100)

    y1 = ratint(x)
    y2 = np.sum([term(x) for term in int_terms], axis=0)
    assert np.allclose(y1, y2)

    # Now try with a polynomial part
    testp = Polynomial([1.0, 2.0, -0.5])

    ratint += testp

    assert ratint._poly == testp
    y1 = ratint(x)
    y2 = y2 + testp(x)

    assert np.allclose(y1, y2)


@pytest.mark.filterwarnings("ignore:All terms are RationalTerms")
@pytest.mark.parametrize("int_pair", _term_int_pairs)
def test_ratint_from_rterms(
    int_pair: TermPair, comparison_methods: ComparisonMethods
) -> None:
    """Test creation of RationalFunctionIntegral from RationalIntegralGeneralTerm."""
    terms, int_terms = int_pair.f, int_pair.intf
    ratint = RationalFunctionIntegral.from_rational_terms(terms)

    assert isinstance(ratint, RationalFunctionIntegral)
    assert comparison_methods.compare_ratint_seqs(ratint._terms, int_terms)

    x = np.linspace(-1, 1, 100)

    y1 = ratint(x)
    y2 = np.sum([iterm(x) for iterm in int_terms], axis=0)
    assert np.allclose(y1, y2)

    # Try it with a polynomial part
    tpoly = Polynomial([1.0, 2.0, -0.5])

    ratint = RationalFunctionIntegral.from_rational_terms(terms, tpoly)
    assert isinstance(ratint, RationalFunctionIntegral)

    y1 = ratint(x)
    y2 += tpoly.integ()(x)

    assert np.allclose(y1, y2)


@pytest.mark.filterwarnings("ignore:All terms are RationalTerms")
def test_int_real_line() -> None:
    """Test the real_line method of RationalFunctionIntegral."""

    r1 = RationalFunctionIntegral([], [1.0, 1.0])
    assert np.isnan(r1.real_line())

    r2 = RationalFunctionIntegral([RationalIntegralLogTerm(1.0, 1.0)])
    assert np.isnan(r2.real_line())

    r3 = RationalFunctionIntegral([RationalIntegralLogPairTerm(1.0, 1.0)])
    assert np.isnan(r3.real_line())
    assert r3.real_line(as_cauchy_pv=True) == 0.0

    r4 = RationalFunctionIntegral(
        [
            RationalIntegralArctanTerm(1.0, 1.0 + 1.0j),
            RationalIntegralArctanTerm(2.0, 1.0 - 1.5j),
        ]
    )

    assert np.isclose(r4.real_line(), -np.pi / 3)


def test_cconj_pairs():
    """Test the _int_cconj_pair function."""

    # Case 1: the terms are purely imaginary and conjugate
    out_terms = _int_cconj_pair(1.0j, -1.0j, 2.0j)

    assert len(out_terms) == 1
    assert isinstance(out_terms[0], RationalIntegralArctanTerm)
    assert out_terms[0]._a == -4.0
    assert out_terms[0]._x0 == 0.0
    assert out_terms[0]._w == 2.0

    # Case 2: the terms are purely real and equal
    out_terms = _int_cconj_pair(1.0, 1.0, 2.0j)

    assert len(out_terms) == 1
    assert isinstance(out_terms[0], RationalIntegralLogPairTerm)
    assert out_terms[0]._a == 2.0
    assert out_terms[0]._x0 == 0.0
    assert out_terms[0]._w == 2.0

    # Case 3: arbitrary terms
    out_terms = _int_cconj_pair(1.0 + 1.0j, 2.0 - 0.5j, 3.0 + 4.0j)

    assert len(out_terms) == 2
    assert isinstance(out_terms[0], RationalIntegralLogPairTerm)
    assert out_terms[0]._a == 3.0 + 0.5j
    assert out_terms[0]._x0 == 3.0
    assert out_terms[0]._w == 4.0
    assert isinstance(out_terms[1], RationalIntegralArctanTerm)
    assert out_terms[1]._a == -6 - 4.0j
    assert out_terms[1]._x0 == 3.0
    assert out_terms[1]._w == 4.0


def test_find_conj_pairs() -> None:
    # Test the _find_cconj_pairs function

    terms = [
        RationalTerm(1.0, 2.0),
        RationalTerm(1 + 0.5j, 1.0, 1),
        RationalTerm(1 + 2.0j, 1.0, 2),
        RationalTerm(1 - 2.0j, 1.0, 2),
        RationalTerm(1 - 0.5j, 1.0, 1),
        RationalTerm(1 + 0.5j, 1.0, 1),
        RationalTerm(1 - 0.6j, 1.0, 1),
    ]

    conj_pairs, remaining_terms = _find_cconj_pairs(terms)

    assert conj_pairs == [(terms[1], terms[4])]
    assert remaining_terms == [terms[0], terms[2], terms[3], terms[5], terms[6]]
