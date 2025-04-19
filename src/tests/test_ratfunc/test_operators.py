import pytest
import numpy as np
from typing import TypeVar
from itertools import chain, product
from numpy.polynomial import Polynomial
from pytest_snapshot.plugin import Snapshot
from rational_functions import RationalFunction
from rational_functions.terms import RationalTerm
from rational_functions.integral import RationalFunctionIntegral


T = TypeVar("T")


def self_product_w_o_duplicates(a: list[T]) -> list[tuple[T, T]]:
    """Return all unique pairs of elements from a list.
    Args:
        a (list[T]): Input list.
    Returns:
        list[tuple[T, T]]: List of unique pairs.
    """
    prod = []

    for i, a1 in enumerate(a):
        for a2 in a[i + 1 :]:
            prod.append((a1, a2))

    return prod


_test_terms = {
    "r_s": RationalTerm(3.0, 2.0),  # Real root, real coefficient
    "r_m": RationalTerm(3.0, 2.0, 2),  # Real root, real coefficient with multiplicity
    "c_s": RationalTerm(3.0 + 1.0j, 2.0),  # Complex root, real coefficient
    "c_m": RationalTerm(
        3.0 + 1.0j, 2.0, 2
    ),  # Complex root, real coefficient with multiplicity
    "c*_s": RationalTerm(3.0 - 1.0j, -2.0),  # Complex conjugate root, real coefficient
    "c*_m": RationalTerm(
        3.0 - 1.0j, -2.0, 2
    ),  # Complex conjugate root, real coefficient with multiplicity
}

_test_ratfuncs = chain(
    *[
        [
            RationalFunction([term], None),
            RationalFunction([term], Polynomial([2.0])),
            RationalFunction([term], Polynomial([2.0, 3.0])),
        ]
        for term in _test_terms.values()
    ]
)

# Plus a series of rational functions with term pairs
_test_ratfuncs = chain(
    _test_ratfuncs,
    *[
        [
            RationalFunction([term1, term2], None),
            RationalFunction([term1, term2], Polynomial([2.0])),
            RationalFunction([term1, term2], Polynomial([2.0, 3.0])),
        ]
        for term1, term2 in self_product_w_o_duplicates(list(_test_terms.values()))
    ],
)
_test_ratfuncs = list(_test_ratfuncs)

_test_ratfunc_pairs = self_product_w_o_duplicates(_test_ratfuncs)

_test_polynomials = [
    Polynomial([1.0]),
    Polynomial([2.0, 3.0]),
    Polynomial([3.0, 4.0, 5.0]),
]
_test_scalars = [1.0, 2.0, -1.0, -0.35, 2.2j, 3.0 + 1.0j]


@pytest.mark.parametrize(
    "rf",
    _test_ratfuncs,
)
def test_rfunc_neg(rf: RationalFunction) -> None:
    """Test negation of a rational function."""
    x = np.linspace(-1, 1, 50)
    y1 = rf(x) * (-1)
    y2 = (-rf)(x)

    assert np.allclose(y1, y2)
    assert isinstance((-rf), rf.__class__)


@pytest.mark.parametrize("rf_l, rf_r", _test_ratfunc_pairs)
def test_rfunc_add(rf_l: RationalFunction, rf_r: RationalFunction) -> None:
    """Test addition of two rational functions."""
    x = np.linspace(-1, 1, 50)

    rf_s = rf_l + rf_r

    assert isinstance(rf_s, RationalFunction)

    y1 = rf_l(x) + rf_r(x)
    y2 = (rf_l + rf_r)(x)

    assert np.allclose(y1, y2)


@pytest.mark.parametrize(
    "rf,p",
    product(_test_ratfuncs, _test_polynomials),
)
def test_rfunc_poly_add(rf: RationalFunction, p: Polynomial) -> None:
    """Test addition of a rational function and a polynomial."""
    x = np.linspace(-1, 1, 50)

    s1 = rf + p
    s2 = p + rf

    assert isinstance(s1, rf.__class__)
    assert isinstance(s2, rf.__class__)

    y1 = rf(x) + p(x)
    y2 = s1(x)
    y3 = s2(x)

    assert np.allclose(y1, y2)
    assert np.allclose(y1, y3)
    assert np.allclose(y2, y3)


@pytest.mark.parametrize(
    "rf,s",
    product(_test_ratfuncs, _test_scalars),
)
def test_rfunc_scalar_add(rf: RationalFunction, s: complex) -> None:
    """Test addition of a rational function and a scalar."""

    x = np.linspace(-1, 1, 50)

    s1 = rf + s
    s2 = s + rf

    assert isinstance(s1, rf.__class__)
    assert isinstance(s2, rf.__class__)

    y1 = rf(x) + s
    y2 = s1(x)
    y3 = s2(x)

    assert np.allclose(y1, y2)
    assert np.allclose(y1, y3)


@pytest.mark.parametrize("rf_l, rf_r", _test_ratfunc_pairs)
def test_rfunc_sub(rf_l: RationalFunction, rf_r: RationalFunction) -> None:
    """Test subtraction of two rational functions."""
    x = np.linspace(-1, 1, 50)

    s1 = rf_l - rf_r
    s2 = rf_r - rf_l
    assert isinstance(s1, RationalFunction)
    assert isinstance(s2, RationalFunction)

    y1 = rf_l(x) - rf_r(x)
    y2 = s1(x)
    y3 = s2(x)

    assert np.allclose(y1, y2)
    assert np.allclose(-y1, y3)


@pytest.mark.parametrize("rf_l, rf_r", _test_ratfunc_pairs)
def test_rfunc_mul(rf_l: RationalFunction, rf_r: RationalFunction) -> None:
    """Test multiplication of two rational functions."""
    x = np.linspace(-1, 1, 50)

    m1 = rf_l * rf_r
    m2 = rf_r * rf_l

    assert isinstance(m1, RationalFunction)
    assert isinstance(m2, RationalFunction)

    y0 = rf_l(x) * rf_r(x)
    y1 = m1(x)
    y2 = m2(x)

    assert np.allclose(y0, y1)
    assert np.allclose(y0, y2)


@pytest.mark.parametrize(
    "rf,p",
    product(_test_ratfuncs, _test_polynomials),
)
def test_rfunc_poly_mul(rf: RationalFunction, p: Polynomial) -> None:
    """Test multiplication of a rational function and a polynomial."""
    x = np.linspace(-1, 1, 50)

    m1 = rf * p
    m2 = p * rf

    assert isinstance(m1, RationalFunction)
    assert isinstance(m2, RationalFunction)

    y0 = rf(x) * p(x)
    y1 = m1(x)
    y2 = m2(x)

    assert np.allclose(y0, y1)
    assert np.allclose(y0, y2)


@pytest.mark.parametrize(
    "rf,s",
    product(_test_ratfuncs, _test_scalars),
)
def test_rfunc_scalar_mul(rf: RationalFunction, s: complex) -> None:
    """Test multiplication of a rational function and a scalar."""
    x = np.linspace(-1, 1, 50)

    m1 = rf * s
    m2 = s * rf

    assert isinstance(m1, RationalFunction)
    assert isinstance(m2, RationalFunction)

    y0 = rf(x) * s
    y1 = m1(x)
    y2 = m2(x)
    assert np.allclose(y0, y1)
    assert np.allclose(y0, y2)


@pytest.mark.parametrize(
    "rf",
    _test_ratfuncs,
)
def test_rfunc_recip(rf: RationalFunction) -> None:
    """Test inverse of a rational function."""
    x = np.linspace(-1, 1, 50)

    recip = rf.reciprocal()

    assert isinstance(recip, RationalFunction)

    y0 = 1.0 / rf(x)
    y1 = recip(x)

    assert np.allclose(y1, y0)


@pytest.mark.parametrize("rf_l, rf_r", _test_ratfunc_pairs)
def test_rfunc_div(rf_l: RationalFunction, rf_r: RationalFunction) -> None:
    """Test division of two rational functions."""
    x = np.linspace(-1, 1, 50)

    div = rf_l / rf_r

    assert isinstance(div, RationalFunction)

    y0 = rf_l(x) / rf_r(x)
    y1 = div(x)

    assert np.allclose(y1, y0)


@pytest.mark.parametrize(
    "rf,p",
    list(product(_test_ratfuncs, _test_polynomials)),
)
def test_rfunc_poly_div(rf: RationalFunction, p: Polynomial) -> None:
    """Test division of a rational function by a polynomial."""
    x = np.linspace(-1, 1, 50)

    div = rf / p

    assert isinstance(div, RationalFunction)

    y0 = rf(x) / p(x)
    y1 = div(x)

    assert np.allclose(y1, y0)


@pytest.mark.parametrize(
    "rf,s",
    list(product(_test_ratfuncs, _test_scalars)),
)
def test_rfunc_scalar_div(rf: RationalFunction, s: complex) -> None:
    """Test division of a rational function by a scalar."""
    x = np.linspace(-1, 1, 50)

    div = rf / s

    assert isinstance(div, RationalFunction)

    y0 = rf(x) / s
    y1 = div(x)

    assert np.allclose(y1, y0)


@pytest.mark.parametrize(
    "rf,m",
    product(_test_ratfuncs, [-2, -1, 1, 2, 3]),
)
def test_rfunc_pow(rf: RationalFunction, m: int) -> None:
    """Test power of a rational function."""
    x = np.linspace(-1, 1, 50)

    pow1 = rf**m
    assert isinstance(pow1, RationalFunction)

    y0 = rf(x) ** m
    y1 = pow1(x)

    assert np.allclose(y0, y1)


@pytest.mark.parametrize("rf,m", product(_test_ratfuncs, [1, 2, 3]))
def test_rfunc_deriv(rf: RationalFunction, m: int) -> None:
    """Test derivative of a rational function."""
    x = np.linspace(-1, 1, 1000)

    d1 = rf.deriv(m)

    assert isinstance(d1, RationalFunction)

    y0 = rf(x)
    for _ in range(m):
        y0 = np.gradient(y0, x, edge_order=2)

    y1 = d1(x)

    # Exclude edges from comparison, and use a relaxed tolerance
    # due to flaws in the numerical derivative
    assert np.allclose(y0[m:-m], y1[m:-m], rtol=5e-3)


@pytest.mark.parametrize("rf", _test_ratfuncs)
def test_rfunc_integ(rf: RationalFunction) -> None:
    """Test integral of a rational function."""
    x = np.linspace(-1, 1, 10000)

    i1 = rf.integ()

    # Are all the terms higher than order 1?
    high_ord = all(term.order > 1 for term in rf._terms)
    if high_ord:
        assert isinstance(i1, RationalFunction)

        # Try forcing the integral to be a RationalFunctionIntegral
        with pytest.warns(UserWarning, match="All terms are RationalTerms"):
            iforce = rf.integ(force_iobj=True)
        assert isinstance(iforce, RationalFunctionIntegral)
    else:
        assert isinstance(i1, RationalFunctionIntegral)

    assert i1._poly == rf._poly.integ()

    y0 = rf(x)
    y1 = i1(x) - i1(x[0])
    y2 = np.cumsum(y0) * (x[1] - x[0])
    y2 -= y2[0]

    assert np.allclose(y1, y2, atol=1e-3)


@pytest.mark.parametrize(
    "rf",
    _test_ratfuncs,
)
def test_rfunc_str(rf: RationalFunction, snapshot: Snapshot) -> None:
    """Test the __str__ method of the RationalFunction class."""
    assert isinstance(rf.__str__(), str)

    snapshot.assert_match(rf.__str__(), "ratfunc_str")


def test_rfunc_notimpl():
    # Test not implemented operations

    rf = RationalFunction.from_fraction([1.0], [-1.0, 1.0])

    with pytest.raises(TypeError):
        rf + None

    with pytest.raises(TypeError):
        rf - None

    with pytest.raises(TypeError):
        rf * None

    with pytest.raises(TypeError):
        rf / None

    with pytest.raises(TypeError):
        rf**None

    with pytest.raises(ValueError):
        rf**3.5
