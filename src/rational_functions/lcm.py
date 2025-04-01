
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
        
        print(list(roots))