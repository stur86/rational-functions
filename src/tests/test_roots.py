import pytest
from rational_functions.roots import PolynomialRoot
from dataclasses import FrozenInstanceError


@pytest.mark.parametrize(
    "v, m",
    [
        (3.0, 1),
        (3.0, 2),
        (3.0 + 1.0j, 1),
        (3.0 + 1.0j, 2),
        (3.0 - 1.0j, 1),
        (3.0 - 1.0j, 2),
        (2.0, 1),
    ],
)
def test_polynomial_root(v: complex, m: int):
    p_root = PolynomialRoot(value=v, multiplicity=m)

    assert p_root.is_real == (v.imag == 0)
    assert p_root.real == v.real
    assert p_root.imag == v.imag

    m_poly = p_root.monic_polynomial()
    assert m_poly.coef[-1] == 1.0  # Is monic
    assert m_poly(v) == 0.0  # Is a root

    # Degree should follow multiplicity
    assert m_poly.degree() == m

    p_mul_root = p_root.with_multiplicity(4)

    assert p_mul_root.value == v
    assert p_mul_root.multiplicity == 4

    # Try changing a field
    with pytest.raises(FrozenInstanceError):
        p_root.multiplicity = 4


@pytest.mark.parametrize(
    "r1, r2, is_equivalent",
    [
        (
            PolynomialRoot(value=3.0, multiplicity=2),
            PolynomialRoot(value=2.0, multiplicity=3),
            False,
        ),
        (
            PolynomialRoot(value=3.0, multiplicity=2),
            PolynomialRoot(value=3.0, multiplicity=3),
            True,
        ),
    ],
)
def test_polyroot_equivalence(
    r1: PolynomialRoot, r2: PolynomialRoot, is_equivalent: bool
):
    assert r1.is_equivalent(r2) == is_equivalent


def test_polyroot_highest():
    r1 = PolynomialRoot(value=3.0, multiplicity=2)
    r2 = PolynomialRoot(value=3.0, multiplicity=3)

    r3 = r1.highest(r2)
    r4 = r2.highest(r1)

    assert r3 == r4
    assert r3.value == 3.0
    assert r3.multiplicity == 3
