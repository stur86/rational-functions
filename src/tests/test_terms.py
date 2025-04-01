import numpy as np
from numpy.polynomial import Polynomial
import pytest
from rational_functions.roots import PolynomialRoot
from rational_functions.terms import (
    RationalTerm,
    RationalIntegralArctanTerm,
    RationalIntegralLogTerm,
    RationalIntegralLogPairTerm,
)
from pytest_snapshot.plugin import Snapshot


def test_arctan_term():
    a = 2.0
    r_r = 3.0
    r_i = 1.5
    atant = RationalIntegralArctanTerm(a, r_r + r_i * 1j)

    x = np.linspace(-1, 1, 100)

    assert np.allclose(atant(x), a / r_i * np.arctan((x - r_r) / r_i))


def test_log_term():
    a = 2.0
    r = 3.0 + 1.5j
    logt = RationalIntegralLogTerm(a, r)

    x = np.linspace(-1, 1, 100, dtype=np.complex128)

    assert np.allclose(logt(x), a * np.log(x - r))


def test_log_pair_term():
    a = 2.0
    r_r = 3.0
    r_i = 1.5
    logpt = RationalIntegralLogPairTerm(a, r_r + r_i * 1j)

    x = np.linspace(-1, 1, 100, dtype=np.complex128)

    assert np.allclose(logpt(x), 0.5 * a * np.log((x - r_r) ** 2 + r_i**2))


@pytest.mark.parametrize(
    "root",
    [
        PolynomialRoot(value=3.0, multiplicity=1),
        PolynomialRoot(value=3.0, multiplicity=2),
        PolynomialRoot(value=3.0 + 1.0j, multiplicity=1),
        PolynomialRoot(value=3.0 + 1.0j, multiplicity=2),
    ],
)
def test_rational_term_num_den(root: PolynomialRoot):
    term = RationalTerm(root, 1.0)

    assert term._coef == 1.0
    assert term.numerator == np.polynomial.Polynomial([1.0])
    assert (
        term.denominator
        == np.polynomial.Polynomial([-root.value, 1.0]) ** root.multiplicity
    )


@pytest.mark.parametrize(
    "root,a",
    [
        (PolynomialRoot(value=3.0), 1.0),
        (PolynomialRoot(value=3.0, multiplicity=2), -1.0),
        (PolynomialRoot(value=3.0, multiplicity=3), -1.0),
        (PolynomialRoot(value=3.0 + 1.0j), 1.0),
        (PolynomialRoot(value=3.0 + 1.0j, multiplicity=2), -1.0),
        (PolynomialRoot(value=3.0 - 1.0j), 1.0+0.5j),
        (PolynomialRoot(value=3.0 - 1.0j, multiplicity=3), -1.0+0.5j),
    ],
)
def test_rational_term_eval(root: PolynomialRoot, a: complex):
    pterm = RationalTerm(root, a)

    x = np.linspace(-1, 1, 10000)
    y = a / (x - root.value) ** root.multiplicity

    assert np.allclose(pterm(x), y)

    # Evaluate derivatives
    dterm = pterm.deriv()
    assert isinstance(dterm, RationalTerm)
    pderiv = dterm(x)

    dy_dx = np.gradient(y, x)

    assert np.allclose(pderiv, dy_dx, rtol=1e-3)

    int_y = np.cumsum(y) * (x[1] - x[0])
    pterm_int = pterm.integ()
    pint = np.sum([term(x) for term in pterm_int], axis=0)

    assert np.allclose(pint - pint[0], int_y - int_y[0], rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    "rterm",
    [
        RationalTerm(PolynomialRoot(value=3.0), [1.0]),
        RationalTerm(PolynomialRoot(value=3.0, multiplicity=2), [-1.0]),
        RationalTerm(PolynomialRoot(value=3.0 + 1.0j), [1.0]),
        RationalTerm(PolynomialRoot(value=3.0 + 1.0j, multiplicity=2), [-1.0]),
    ],
)
def test_rational_term_str(rterm: RationalTerm, snapshot: Snapshot) -> None:
    """Test the __str__ method of the RationalTerm class."""
    assert isinstance(rterm.__str__(), str)

    snapshot.assert_match(rterm.__str__(), "rterm_str")


@pytest.mark.parametrize(
    "rterm1,rterm2",
    [
        (
            RationalTerm(PolynomialRoot(value=3.0), [1.0]),
            RationalTerm(PolynomialRoot(value=4.0), [-1.0]),
        ),
        (
            RationalTerm(PolynomialRoot(value=3.0), [1.0]),
            RationalTerm(
                PolynomialRoot(value=2.0 + 1.0j), [-1.0, 0.5]
            ),
        ),
        (
            RationalTerm(PolynomialRoot(value=3.0), [1.0]),
            RationalTerm(PolynomialRoot(value=3.0, multiplicity=2), [-1.0]),
        ),
    ],
)
def test_rational_term_product(rterm1: RationalTerm, rterm2: RationalTerm) -> None:

    rtermprod = RationalTerm.product(rterm1, rterm2)

    for term in rtermprod:
        assert isinstance(term, RationalTerm)

    x = np.linspace(-1, 1, 50)

    y1 = rterm1(x) * rterm2(x)
    y2 = np.sum([term(x) for term in rtermprod], axis=0)
    assert np.allclose(y1, y2)


@pytest.mark.parametrize(
    "rterm,poly",
    [
        (RationalTerm(PolynomialRoot(value=3.0), [1.0]), Polynomial([1.0, 2.0])),
        (
            RationalTerm(PolynomialRoot(value=3.0, multiplicity=3), [1.0]),
            Polynomial([1.0, 2.0, 3.0]),
        ),
        (RationalTerm(PolynomialRoot(value=3.0), [1.0]), Polynomial([-1.0])),
        (
            RationalTerm(PolynomialRoot(value=3.0 + 1.0j), [1.0]),
            Polynomial([-1.0, 2.0]),
        ),
    ],
)
def test_rational_term_polynomial_product(
    rterm: RationalTerm, poly: Polynomial
) -> None:
    """Test the product of a RationalTerm with a Polynomial."""
    out_terms, out_poly = RationalTerm.product_w_polynomial(rterm, poly)

    for term in out_terms:
        assert isinstance(term, RationalTerm)

    x = np.linspace(-1, 1, 50)

    y1 = rterm(x) * poly(x)
    y2 = np.sum([term(x) for term in out_terms], axis=0) + out_poly(x)
    assert np.allclose(y1, y2)


def test_rational_term_neg() -> None:

    rterm = RationalTerm(
        PolynomialRoot(value=3.0 + 1.0j), 1.0
    )

    x = np.linspace(-1, 1, 50)
    y1 = rterm(x) * (-1)
    y2 = (-rterm)(x)

    assert np.allclose(y1, y2)
    assert isinstance((-rterm), rterm.__class__)
    assert (-rterm).root == rterm.root
