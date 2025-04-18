import numpy as np
from numpy.polynomial import Polynomial
import pytest
from dataclasses import dataclass
from rational_functions.integral import RationalFunctionIntegral
from rational_functions.terms import (
    RationalIntegralLogTerm,
    RationalIntegralGeneralTerm,
    RationalTerm,
)


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
def test_ratint_from_iterms(int_pair: TermPair) -> None:
    """Test creation of RationalFunctionIntegral from RationalIntegralGeneralTerm."""
    terms, int_terms = int_pair.f, int_pair.intf
    ratint = RationalFunctionIntegral.from_rational_terms(terms)

    assert isinstance(ratint, RationalFunctionIntegral)

    x = np.linspace(-1, 1, 100)

    y1 = ratint(x)
    y2 = np.sum([iterm(x) for iterm in int_terms], axis=0)
    assert np.allclose(y1, y2)
