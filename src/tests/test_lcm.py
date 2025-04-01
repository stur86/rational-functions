import pytest
from numpy.polynomial import Polynomial
from rational_functions.roots import PolynomialRoot
from rational_functions.lcm import RootLCM


def test_lcm():
    
    roots = [PolynomialRoot(2.0, 2), PolynomialRoot(3.0+1.0j, 1), 
             PolynomialRoot(3.0-1.0j, 1)]
    
    lcm = RootLCM(roots)
    
    assert lcm.roots == roots
    assert lcm.polynomial == Polynomial.fromroots([2.0, 2.0, 3.0+1.0j, 3.0-1.0j])
    assert lcm.residual(2.0, 2) == Polynomial.fromroots([3.0+1.0j, 3.0-1.0j])
    
    with pytest.raises(ValueError):
        lcm.residual(2.0, 3)
        
    with pytest.raises(ValueError):
        lcm.residual(-1.0, 1)
    
