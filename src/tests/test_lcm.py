from rational_functions.roots import PolynomialRoot
from rational_functions.lcm import RootLCM


def test_lcm():
    
    roots = [PolynomialRoot(2.0, 2), PolynomialRoot(3.0+1.0j, 1), 
             PolynomialRoot(3.0-1.0j, 1)]
    
    lcm = RootLCM(roots)
    
    print(lcm)