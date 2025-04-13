import pytest
import numpy as np
from numpy.polynomial import Polynomial
from rational_functions.decomp import catalogue_roots, partial_frac_decomposition


@pytest.mark.parametrize(
    "roots",
    [
        [1.0],
        [2.0, -1.0],
        [1.0 + 1.0j, 1.0 - 1.0j],
        [1.0, 2.0j],
        [2.0, 2.0, 3**0.5 - 1.0j, 3**0.5 + 1.0j],
    ],
)
def test_cat_roots(roots):
    p = Polynomial.fromroots(roots)
    found_roots = catalogue_roots(p)

    # Reconstruct the polynomial
    p_rec = Polynomial([1.0])
    for r in found_roots:
        p_rec *= r.monic_polynomial()

    assert np.allclose(p.coef, p_rec.coef)
    assert np.allclose(p_rec(roots), 0.0)


@pytest.mark.parametrize(
    "numerator,denominator",
    [
        (Polynomial([0.0, 1.0]), Polynomial.fromroots([2.0, 3.0, -1.5])),
        (
            Polynomial([0.5, 1.0, -1.0]),
            Polynomial.fromroots([2.0, 3.0, 0.8 + 0.2j, 0.8 - 0.2j]),
        ),
        (Polynomial([0.5, 1.0, -1.0]), Polynomial.fromroots([2.0, 2.0, 3.0])),
        (Polynomial([0.5]), Polynomial.fromroots([2.0j])),
        (Polynomial([0.5, 1.0, -1.0]), Polynomial.fromroots([2.0, 2.0j, 3.0])),
    ],
)
def test_partial_frac_decomp(numerator: Polynomial, denominator: Polynomial) -> None:
    poles = catalogue_roots(denominator)
    pfracs = partial_frac_decomposition(numerator.coef, poles)

    x = np.linspace(-1, 1, 50)

    y1 = numerator(x) / denominator(x)
    y2 = np.sum([pf(x) for pf in pfracs], axis=0)

    assert np.allclose(y1, y2)
