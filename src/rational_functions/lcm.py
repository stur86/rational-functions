from itertools import chain

from .roots import PolynomialRoot

class RootLCM:
    """Least Common Multiple of a product of roots"""
    
    _factors: dict[complex, int]
    
    def __init__(self, roots: list[PolynomialRoot]) -> None:
        """Initialize a Root Least Common Multiple object.
        
        Args:
            roots (list[PolynomialRoot]): roots to reduce to the least
                common multiple
        """
        
        # First, separate complex pairs from singles
        single_roots = filter(lambda r: not r.is_complex_pair, roots)
        pair_roots = filter(lambda r: r.is_complex_pair, roots)
        single_roots = chain(single_roots, *map(lambda r: r.split(), pair_roots))
        
        print(list(single_roots))