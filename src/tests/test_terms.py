import numpy as np
import pytest
from rational_functions.rtypes import PolynomialRoot
from rational_functions.terms import (
    RationalTermSingle,
    RationalTermComplexPair,
    RationalTerm,
    RationalIntegralArctanTerm,
    RationalIntegralLogTerm,
    RationalIntegralLogPairTerm,
)

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
        PolynomialRoot(value=3.0 - 1.0j, multiplicity=1, is_complex_pair=True),
        PolynomialRoot(value=3.0 - 1.0j, multiplicity=2, is_complex_pair=True),
    ],
)
def test_rational_term(root: PolynomialRoot):
    coefs = [1.0]
    term = RationalTerm(root, coefs)

    if root.is_complex_pair:
        assert isinstance(term, RationalTermComplexPair)
        assert len(term._coefs) == 2
        assert np.all(term._coefs == coefs + [0.0])
        assert term.numerator == np.polynomial.Polynomial(coefs)
        assert (
            term.denominator
            == np.polynomial.Polynomial(
                [root.real**2 + root.imag**2, -2 * root.real, 1.0]
            )
            ** root.multiplicity
        )
    else:
        assert isinstance(term, RationalTermSingle)
        assert len(term._coefs) == 1
        assert np.all(term._coefs == coefs)
        assert term.numerator == np.polynomial.Polynomial(coefs)
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
        (PolynomialRoot(value=3.0 - 1.0j), 1.0),
        (PolynomialRoot(value=3.0 - 1.0j, multiplicity=3), -1.0),
        (PolynomialRoot(value=3.0 - 1.0j, multiplicity=1, is_complex_pair=True), 1.0),
        (PolynomialRoot(value=3.0 - 1.0j, multiplicity=2, is_complex_pair=True), -1.0),
    ],
)
def test_rational_term_eval(root: PolynomialRoot, a: float):
    pterm = RationalTerm(root, [a])

    x = np.linspace(-1, 1, 10000)
    if root.is_complex_pair:
        y = a / ((x - root.real) ** 2 + root.imag**2) ** root.multiplicity
    else:
        y = a / (x - root.value) ** root.multiplicity

    assert np.allclose(pterm(x), y)

    # Evaluate derivatives
    assert len(pterm.deriv()) == 1 + root.is_complex_pair
    pderiv = np.sum([term(x) for term in pterm.deriv()], axis=0)

    dy_dx = np.gradient(y, x)

    assert np.allclose(pderiv, dy_dx, rtol=1e-3)

    int_y = np.cumsum(y) * (x[1] - x[0])
    pterm_int = pterm.integ()
    pint = np.sum([term(x) for term in pterm_int], axis=0)

    assert np.allclose(pint - pint[0], int_y - int_y[0], rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    "rterm",
    [
        RationalTermSingle(PolynomialRoot(value=3.0), [1.0]),
        RationalTermSingle(PolynomialRoot(value=3.0, multiplicity=2), [-1.0]),
        RationalTermSingle(PolynomialRoot(value=3.0 + 1.0j), [1.0]),
        RationalTermSingle(PolynomialRoot(value=3.0 + 1.0j, multiplicity=2), [-1.0]),
        RationalTermComplexPair(
            PolynomialRoot(value=3.0 - 1.0j, multiplicity=1, is_complex_pair=True), [1.0]
        ),
        RationalTermComplexPair(
            PolynomialRoot(value=3.0 - 1.0j, multiplicity=2, is_complex_pair=True), [-1.0]
        ),
    ],
)
def test_rational_term_repr(rterm: RationalTerm, snapshot) -> None:
    """Test the __repr__ method of the RationalTerm class."""
    assert isinstance(rterm.__repr__(), str)
    assert isinstance(rterm.__str__(), str)
    
    snapshot.assert_match(rterm.__repr__(), "rterm_repr")
    
    print(rterm)