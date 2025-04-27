import numpy as np
from numpy.polynomial import Polynomial
from rational_functions.fit import fit_ratfun_leastsq


def test_fit_ratfun_leastsq():
    x = np.linspace(-2, 2, 30)

    def func(x):
        return np.exp(-(x**2) / 2)

    y = func(x)

    m = 2
    n = 4
    num, den = fit_ratfun_leastsq(x, y, m, n)

    assert num.degree() == m
    assert den.degree() == n

    x_dense = np.linspace(-2, 2, 1000)
    y_dense = func(x_dense)
    y_fit = num(x_dense) / den(x_dense)

    assert np.allclose(y_fit, y_dense, atol=1e-3)

    # Try fitting an actual rational function
    targ_num = Polynomial([0, 1])
    targ_den = Polynomial([1, 0, 1])

    y_rational = targ_num(x) / targ_den(x)
    num, den = fit_ratfun_leastsq(x, y_rational, targ_num.degree(), targ_den.degree())
    num_coef = num.convert().coef
    den_coef = den.convert().coef

    # Normalize to make denominator monic
    num_coef /= den_coef[-1]
    den_coef /= den_coef[-1]

    assert np.allclose(num_coef, targ_num.coef, atol=1e-12, rtol=0)
    assert np.allclose(den_coef, targ_den.coef, atol=1e-12, rtol=0)
