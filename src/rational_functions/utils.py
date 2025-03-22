"""Utility functions for the package."""
import numpy as np
from numpy.typing import ArrayLike

PolynomialDef = np.polynomial.Polynomial | ArrayLike

def as_polynomial(p: PolynomialDef) -> np.polynomial.Polynomial:
    """Coerce input to a numpy polynomial.
    
    Args:
        p: Input polynomial, or series of coefficients.
        
    Returns:
        np.polynomial.Polynomial: Coerced polynomial.
    """
    
    if isinstance(p, np.polynomial.Polynomial):
        return p
    return np.polynomial.Polynomial(p)

