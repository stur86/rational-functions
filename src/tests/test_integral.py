import numpy as np
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
    TermPair((RationalTerm(2, 1, 1),), (RationalIntegralLogTerm(1, 2),)),
]


@pytest.mark.parametrize("int_pair", _term_int_pairs)
def test_ratint_basic(int_pair: TermPair) -> None:
    """Test basic functionality of RationalFunctionIntegral."""
    _, int_terms = int_pair.f, int_pair.intf
    ratint = RationalFunctionIntegral(int_terms)

    assert isinstance(ratint, RationalFunctionIntegral)
    assert len(ratint._terms) == len(int_terms)

    x = np.linspace(-1, 1, 100)

    y1 = ratint(x)
    y2 = np.sum([term(x) for term in int_terms], axis=0)
    assert np.allclose(y1, y2)
