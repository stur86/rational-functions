from numpy.polynomial import Polynomial
from collections import defaultdict
from itertools import chain
from typing import Mapping
from .roots import PolynomialRoot

class RootLCM:
    """Least Common Multiple of a product of roots"""
    
    _factors: Mapping[complex, int]
    
    def __init__(self, roots: list[PolynomialRoot]) -> None:
        """Initialize a Root Least Common Multiple object.
        
        Args:
            roots (list[PolynomialRoot]): roots to reduce to the least
                common multiple
        """
        
        self._factors = defaultdict(int)
        
        for root in roots:
            self._factors[root.value] = max(self._factors[root.value], root.multiplicity)
        
        print(self._factors)
        
    @property
    def polynomial(self) -> Polynomial:
        """Return the polynomial whose roots are the least common multiple of the roots."""
        return Polynomial.fromroots(list(chain(*[[k] * v for k, v in self._factors.items()])))
