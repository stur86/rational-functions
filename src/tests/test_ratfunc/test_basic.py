import numpy as np
import pytest
from numpy.polynomial import Polynomial
from rational_functions import RationalFunction
from rational_functions.terms import RationalTerm


@pytest.mark.parametrize(
    "terms,poly",
    [
        ([RationalTerm(3.0, 1.0)], None),
        ([], Polynomial([1.0, 2.0])),
        (
            [RationalTerm(2.0 + 1.0j, 1.0), RationalTerm(2.0 - 1.0j, 1.0)],
            Polynomial([1.0, 2.0]),
        ),
    ],
)
def test_ratfunc_eval(terms: list[RationalTerm], poly: Polynomial | None):
    """Test the evaluation of the rational function."""
    rat_func = RationalFunction(terms, poly)

    x = np.linspace(-1, 1, 100)
    y1 = rat_func(x)
    y2 = np.sum([term(x) for term in terms], axis=0)
    if poly is not None:
        y2 += poly(x)
    assert np.allclose(y1, y2)
