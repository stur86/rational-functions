import pytest
from rational_functions.types import PolynomialRoot
from dataclasses import FrozenInstanceError


@pytest.mark.parametrize(
    "v, m, is_pair, valid",
    [
        (3.0, 1, False, True),
        (3.0, 2, False, True),
        (3.0 + 1.0j, 1, False, True),
        (3.0 + 1.0j, 2, False, True),
        (3.0 - 1.0j, 1, True, True),
        (3.0 - 1.0j, 2, True, True),
        (2.0, 1, True, False),
    ],
)
def test_polynomial_root(v: complex, m: int, is_pair: bool, valid: bool):

    if not valid:
        with pytest.raises(AssertionError):
            PolynomialRoot(value=v, multiplicity=m, is_complex_pair=is_pair)
        return

    p_root = PolynomialRoot(value=v, multiplicity=m, is_complex_pair=is_pair)

    assert p_root.is_real == (v.imag == 0)
    assert p_root.real == v.real
    assert p_root.imag == v.imag

    m_poly = p_root.monic_polynomial()
    assert m_poly.coef[-1] == 1.0  # Is monic
    assert m_poly(v) == 0.0  # Is a root
    if is_pair:
        assert m_poly(v.conjugate()) == 0.0  # Is also a root

    # Degree should follow multiplicity
    assert m_poly.degree() == (1 + is_pair) * m

    p_mul_root = p_root.with_multiplicity(4)

    assert p_mul_root.value == v
    assert p_mul_root.multiplicity == 4
    assert p_mul_root.is_complex_pair == is_pair

    # Try changing a field
    with pytest.raises(FrozenInstanceError):
        p_root.multiplicity = 4
