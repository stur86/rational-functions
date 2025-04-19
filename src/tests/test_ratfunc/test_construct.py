import pytest
import numpy as np
from numpy.polynomial import Polynomial
from rational_functions import RationalFunction
from rational_functions.roots import PolynomialRoot
from rational_functions.decomp import catalogue_roots


@pytest.mark.parametrize(
    "num, den",
    [
        (Polynomial([1.0, 2.0]), Polynomial.fromroots([2.0, 4.0])),
        (Polynomial([1.0, 2.0]), Polynomial.fromroots([2.0, 4.0, 5.0])),
        (Polynomial([1.0, 2.0]), Polynomial([1.0, 2.0])),
        (Polynomial([1.0]), Polynomial.fromroots([-2.0, 4.0])),
        (Polynomial([1.0]), Polynomial.fromroots([0.5 + 0.1j, 0.5 - 0.1j])),
        (Polynomial([-1.0, 2.0, 3.0, 0.5]), Polynomial.fromroots([2.0, 4.0]) * 2),
    ],
)
def test_ratfunc_from_frac(num: Polynomial, den: Polynomial):
    """Test the creation of a rational function from a fraction."""
    ratfunc = RationalFunction.from_fraction(num, den)

    x = np.linspace(-1, 1, 100)

    y1 = ratfunc(x)
    y2 = num(x) / den(x)

    assert np.allclose(y1, y2)
    assert isinstance(ratfunc, RationalFunction)
    assert ratfunc._poly == num // den


@pytest.mark.parametrize(
    "num, poles",
    [
        (Polynomial([1.0, 2.0]), [PolynomialRoot(2.0), PolynomialRoot(4.0)]),
        (Polynomial([1.0, 2.0]), [PolynomialRoot(2.0), PolynomialRoot(4.0, 2)]),
        (
            Polynomial([1.0, 2.0]),
            [PolynomialRoot(2.0 + 1.0j), PolynomialRoot(2.0 - 1.0j)],
        ),
    ],
)
def test_ratfunc_from_poles(num: Polynomial, poles: list[PolynomialRoot]):
    rf = RationalFunction.from_poles(num, poles)
    assert isinstance(rf, RationalFunction)
    assert np.allclose(rf.numerator.coef, num.coef)

    droots = catalogue_roots(rf.denominator)
    droots = sorted(droots, key=lambda r: (r.real, r.imag))

    for i, p in enumerate(sorted(poles, key=lambda r: (r.real, r.imag))):
        assert np.isclose(droots[i].value, p.value)
        assert droots[i].multiplicity == p.multiplicity


@pytest.mark.parametrize(
    "roots, poles",
    [
        ([PolynomialRoot(3.0)], [PolynomialRoot(2.0), PolynomialRoot(4.0)]),
        (
            [PolynomialRoot(3.0), PolynomialRoot(1.0)],
            [PolynomialRoot(2.0), PolynomialRoot(4.0, 2)],
        ),
        (
            [PolynomialRoot(1.0 + 1.0j)],
            [PolynomialRoot(2.0 + 1.0j), PolynomialRoot(2.0 - 1.0j)],
        ),
    ],
)
def test_ratfunc_from_roots_and_poles(
    roots: list[PolynomialRoot], poles: list[PolynomialRoot]
):
    rf = RationalFunction.from_roots_and_poles(roots, poles)

    assert isinstance(rf, RationalFunction)

    nroots = catalogue_roots(rf.numerator)
    nroots = sorted(nroots, key=lambda r: (r.real, r.imag))

    droots = catalogue_roots(rf.denominator)
    droots = sorted(droots, key=lambda r: (r.real, r.imag))

    for i, r in enumerate(sorted(roots, key=lambda r: (r.real, r.imag))):
        assert np.isclose(nroots[i].value, r.value)
        assert nroots[i].multiplicity == r.multiplicity

    for i, p in enumerate(sorted(poles, key=lambda r: (r.real, r.imag))):
        assert np.isclose(droots[i].value, p.value)
        assert droots[i].multiplicity == p.multiplicity


@pytest.mark.parametrize(
    "x0, w",
    [
        (0.0, 2.0),
        (-1.0, 1.0),
        (3.0, 4.0),
    ],
)
def test_ratfunc_cauchy(x0: float, w: float):
    """Test construction of a Cauchy distribution."""

    rf = RationalFunction.cauchy(x0, w)
    assert isinstance(rf, RationalFunction)

    x = np.linspace(x0 - 5 * w, x0 + 5 * w, 100)
    y = rf(x)

    assert np.allclose(y, 1.0 / (np.pi * w * (w**2 + (x - x0) ** 2)))
