import numpy as np
from numpy.polynomial import Polynomial
import pytest
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
    assert np.isclose(atant.real_line, a / r_i * np.pi)


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


def test_rational_term_invalid() -> None:
    with pytest.raises(ValueError):
        RationalTerm(1, 1, -1)

    with pytest.raises(ValueError):
        RationalTerm(1, 1, 0)


@pytest.mark.parametrize(
    "pole,coef,order",
    [
        (3.0, 1.0, 1),
        (3.0, 1.0, 2),
        (3.0 + 1.0j, 2.5, 1),
        (3.0 + 1.0j, 2.5, 2),
        (3.0 - 1.0j, 2.5j, 1),
        (3.0 - 1.0j, 2.5j, 2),
    ],
)
def test_rational_term_num_den(pole: complex, coef: complex, order: int) -> None:
    term = RationalTerm(pole, coef, order)

    assert term.coef == coef
    assert term.denominator == np.polynomial.Polynomial([-pole, 1.0]) ** order


@pytest.mark.parametrize(
    "pole,coef,order",
    [
        (3.0, 1.0, 1),
        (3.0, -1.0, 2),
        (3.0, -1.0, 3),
        (3.0 + 1.0j, 1.0, 1),
        (3.0 + 1.0j, -1.0, 2),
        (3.0 - 1.0j, 1.0 + 0.5j, 1),
        (3.0 - 1.0j, -1.0 + 0.5j, 3),
    ],
)
def test_rational_term_eval(pole: complex, coef: complex, order: int) -> None:
    pterm = RationalTerm(pole, coef, order)

    x = np.linspace(-1, 1, 10000)
    y = coef / (x - pole) ** order

    assert np.allclose(pterm(x), y)

    # Evaluate derivatives
    dterm = pterm.deriv()
    assert isinstance(dterm, RationalTerm)
    pderiv = dterm(x)

    dy_dx = np.gradient(y, x)

    assert np.allclose(pderiv, dy_dx, rtol=1e-3)

    int_y = np.cumsum(y) * (x[1] - x[0])
    pterm_int = pterm.integ()
    pint = pterm_int(x)

    assert np.allclose(pint - pint[0], int_y - int_y[0], rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    "rterm",
    [
        RationalTerm(3.0, 1.0),
        RationalTerm(3.0, -1.0, 2),
        RationalTerm(3.0 + 1.0j, 1.0),
        RationalTerm(3.0 + 1.0j, -1.0, 2),
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
            RationalTerm(3.0, 1.0),
            RationalTerm(4.0, -1.0),
        ),
        (
            RationalTerm(3.0, 1.0),
            RationalTerm(2.0 + 1.0j, -1.0),
        ),
        (
            RationalTerm(3.0, 2.0),
            RationalTerm(3.0, -1.0 + 2.0j, 2),
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
        (RationalTerm(3.0, 1.0), Polynomial([1.0, 2.0])),
        (
            RationalTerm(3.0, 1.0, 3),
            Polynomial([1.0, 2.0, 3.0]),
        ),
        (RationalTerm(3.0, 1.0), Polynomial([-1.0])),
        (
            RationalTerm(3.0 + 1.0j, 1.0),
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
    rterm = RationalTerm(3.0 + 1.0j, 1.0)

    x = np.linspace(-1, 1, 50)
    y1 = rterm(x) * (-1)
    y2 = (-rterm)(x)

    assert np.allclose(y1, y2)
    assert isinstance((-rterm), rterm.__class__)
    assert (-rterm).pole == rterm.pole
    assert (-rterm).order == rterm.order


def test_simplify() -> None:
    """Test the simplify method of the RationalTerm class."""
    rterm1 = RationalTerm(3.0, 1.0)
    rterm2 = RationalTerm(3.0, 2.0)
    rterm3 = RationalTerm(4.0, -1.0)

    terms = [rterm1, rterm2, rterm3]
    simplified_terms = RationalTerm.simplify(terms)
    # Sort them to ensure the order is consistent
    simplified_terms.sort(key=lambda x: x.pole)
    # Check that the simplified terms are correct
    assert len(simplified_terms) == 2
    assert simplified_terms[0].pole == rterm1.pole
    assert simplified_terms[0].order == rterm2.order
    assert simplified_terms[1].pole == rterm3.pole
    assert simplified_terms[1].order == rterm3.order
    assert np.isclose(simplified_terms[0].coef, 3.0)
    assert np.isclose(simplified_terms[1].coef, -1.0)

    # Testing simplification with tolerances
    rterm1 = RationalTerm(2.999, 1.0)
    rterm2 = RationalTerm(3.001, 2.0)

    simplified_terms = RationalTerm.simplify([rterm1, rterm2], atol=0.01)
    # Check that the simplified terms are correct
    assert len(simplified_terms) == 1
    assert simplified_terms[0].pole == 3.0
    assert simplified_terms[0].order == 1
    assert simplified_terms[0].coef == 3.0

    rterm1 = RationalTerm(3.0, 1.0 + 0.001j)
    rterm2 = RationalTerm(3.0 + 0.001j, 2.0)
    simplified_terms = RationalTerm.simplify([rterm1, rterm2], imtol=0.01)

    # Check that the simplified terms are correct
    assert len(simplified_terms) == 1
    assert simplified_terms[0].pole == 3.0
    assert simplified_terms[0].order == 1
    assert simplified_terms[0].coef == 3.0
