import pytest
import numpy as np
from numpy.polynomial import Polynomial
from rational_functions import RationalFunction


@pytest.mark.parametrize(
    "num, den",
    [
        (Polynomial([1.0, 2.0]), Polynomial.fromroots([2.0, 4.0])),
        (Polynomial([1.0, 2.0]), Polynomial.fromroots([2.0, 4.0, 5.0])),
        (Polynomial([1.0, 2.0]), Polynomial([1.0, 2.0])),
        (Polynomial([1.0]), Polynomial.fromroots([-2.0, 4.0])),
        (Polynomial([1.0]), Polynomial.fromroots([0.5+0.1j, 0.5-0.1j])),
        (Polynomial([-1.0, 2.0, 3.0, 0.5]), Polynomial.fromroots([2.0, 4.0])*2),
    ]
)
def test_ratfunc_from_frac(num: Polynomial, den: Polynomial):
    """Test the creation of a rational function from a fraction."""
    ratfunc = RationalFunction.from_fraction(num, den)
    
    x = np.linspace(-1, 1, 100)
    
    y1 = ratfunc(x)
    y2 = num(x) / den(x)
    
    assert np.allclose(y1, y2)
    assert isinstance(ratfunc, RationalFunction)
    assert ratfunc._poly == num // den
    