import pytest
from numpy.polynomial import Polynomial
from rational_functions.roots import PolynomialRoot
from rational_functions.lcm import RootLCM


@pytest.mark.parametrize(
    "roots",
    [
        [
            PolynomialRoot(2.0, 2),
            PolynomialRoot(3.0 + 1.0j, 1),
            PolynomialRoot(3.0 - 1.0j, 1),
        ],
        [PolynomialRoot(4.0, 3)],
        [PolynomialRoot(2.0 + 1.0j, 3), PolynomialRoot(-1.0, 1)],
    ],
)
def test_lcm(roots: list[PolynomialRoot]):
    lcm = RootLCM(roots)

    assert sorted(lcm.roots, key=lambda r: hash(r)) == sorted(
        roots, key=lambda r: hash(r)
    )

    poly = Polynomial([1.0])
    for root in roots:
        poly *= root.monic_polynomial()

    assert lcm.polynomial == poly

    # Residuals
    for i, r in enumerate(roots):
        other_roots = roots[:i] + roots[i + 1 :]
        for m in range(1, r.multiplicity + 1):
            lcm_res = lcm.residual(r.value, m)
            other_poly = Polynomial([-r.value, 1.0]) ** (r.multiplicity - m)
            for other_root in other_roots:
                other_poly *= other_root.monic_polynomial()
            assert lcm_res == other_poly

    # Find the root with the highest multiplicity
    max_root = max(roots, key=lambda r: r.multiplicity)

    with pytest.raises(ValueError):
        lcm.residual(max_root.value, max_root.multiplicity + 1)

    # Find the root with the max absolute value
    max_abs_root = max(roots, key=lambda r: abs(r.value))

    with pytest.raises(ValueError):
        lcm.residual(max_abs_root.value * 2, 1)
