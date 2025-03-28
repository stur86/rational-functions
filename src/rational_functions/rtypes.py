"""Type definitions for rational functions."""

import numpy as np
from numpy.polynomial import Polynomial
from numpy.typing import ArrayLike
from dataclasses import dataclass

PolyDefType = ArrayLike | Polynomial


@dataclass(frozen=True)
class PolynomialRoot:
    """Defines a polynomial root with value, multiplicity,
    and supports complex conjugate pairs of roots.

    Note:
        For complex pair roots, the multiplicity refers to the
        multiplicity of each individual root. For example, the roots of the
        polynomial $x^2+1$ would be described by a single `PolynomialRoot`
        object with value $i$ (or equivalently, $-i$) and multiplicity 1.
    """

    value: complex
    multiplicity: int = 1
    is_complex_pair: bool = False

    def __post_init__(self):
        if self.is_complex_pair:
            assert not self.is_real, "Complex pair roots must be complex."

    @property
    def is_real(self) -> bool:
        """Check if the root is real."""
        return np.isreal(self.value)

    @property
    def real(self) -> float:
        """Return the real part of the root."""
        return np.real(self.value)

    @property
    def imag(self) -> float:
        """Return the imaginary part of the root."""
        return np.imag(self.value)

    def monic_polynomial(self) -> Polynomial:
        """Return the monic polynomial for the root."""
        if self.is_complex_pair:
            p = Polynomial([self.real**2 + self.imag**2, -2 * self.real, 1.0])
        else:
            p = Polynomial([-self.value, 1.0])
        return p**self.multiplicity

    def with_multiplicity(self, multiplicity: int) -> "PolynomialRoot":
        """Return a new PolynomialRoot with a different multiplicity."""
        return PolynomialRoot(
            value=self.value,
            multiplicity=multiplicity,
            is_complex_pair=self.is_complex_pair,
        )
    
    def is_equivalent(self, root: "PolynomialRoot") -> bool:
        """Check if two roots are equivalent, multiplicity aside."""
        
        if self.is_complex_pair != root.is_complex_pair:
            return False
        
        if self.is_complex_pair:
            return (self.value == root.value) or (self.value == root.value.conjugate())
        else:
            return self.value == root.value
    