import numpy as np
import pytest
from numpy.polynomial import Polynomial
from rational_functions import RationalFunction
from rational_functions.terms import RationalTerm
from rational_functions.roots import PolynomialRoot


@pytest.mark.parametrize(
    "terms,poly",
    [
        (
            [RationalTerm(PolynomialRoot(3.0), [1.0])],
            None
        ),
        (
            [],
            Polynomial([1.0, 2.0])
        ),
        (
            [RationalTerm(PolynomialRoot(2.0+1.0j, is_complex_pair=True), [1.0])],
            Polynomial([1.0, 2.0])
        )
    ]
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


@pytest.mark.parametrize(
    "terms,poles",
    [
        (
            [RationalTerm(PolynomialRoot(3.0), [1.0])],
            [PolynomialRoot(3.0)]
        ),
        (
            [RationalTerm(PolynomialRoot(3.0), [1.0]), RationalTerm(PolynomialRoot(3.0, 2), [1.0])],
            [PolynomialRoot(3.0, 2)]
        ),
        (
            [RationalTerm(PolynomialRoot(3.0), [1.0]), RationalTerm(PolynomialRoot(3.0+1.0j, is_complex_pair=True), [1.0, 2.0]), 
             RationalTerm(PolynomialRoot(3.0-1.0j, multiplicity=2, is_complex_pair=True), [1.0, 2.0])],
            [PolynomialRoot(3.0), PolynomialRoot(3.0+1.0j, 2, is_complex_pair=True)]
        )
    ]
)
def test_ratfunc_poles(terms: list[RationalTerm], poles: list[PolynomialRoot]):
    """Test the poles of the rational function."""
    rat_func = RationalFunction(terms, None)
    
    # Hashes are computed so that they are equal for equivalent roots
    assert set(map(hash, rat_func.poles)) == set(map(hash, poles))

    
