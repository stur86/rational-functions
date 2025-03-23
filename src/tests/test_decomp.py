import pytest
import numpy as np
from numpy.polynomial import Polynomial
from rational_functions.decomp import catalogue_roots


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
