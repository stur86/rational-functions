from numpy.polynomial import Polynomial
from rational_functions.utils import as_polynomial


def test_as_polynomial():
    
    p = Polynomial([1,2,3])
    c = [4,5,6]
    
    p_p = as_polynomial(p)
    c_p = as_polynomial(c)
    
    assert isinstance(p_p, Polynomial)
    assert isinstance(c_p, Polynomial)
    
    assert p_p == p
    assert (c_p.coef == c).all()