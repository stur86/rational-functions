"""Type definitions for rational functions."""

import numpy as np
from numpy.polynomial import Polynomial
from numpy.typing import ArrayLike
from dataclasses import dataclass

PolyDefType = ArrayLike | Polynomial


@dataclass(frozen=True)
class PolynomialRoot:
    """Defines a polynomial root with value and multiplicity."""

    value: complex
    multiplicity: int = 1

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
        return Polynomial([-self.value, 1.0]) ** self.multiplicity

    def with_multiplicity(self, multiplicity: int) -> "PolynomialRoot":
        """Return a new PolynomialRoot with a different multiplicity."""
        return PolynomialRoot(
            value=self.value,
            multiplicity=multiplicity,
        )

    def is_equivalent(self, root: "PolynomialRoot") -> bool:
        """Check if two roots are equivalent, multiplicity aside."""

        return self.value == root.value

    def highest(self, root: "PolynomialRoot") -> "PolynomialRoot":
        """Return the root with the highest multiplicity between
        two equivalent roots.

        Args:
            root (PolynomialRoot): Other root to compare with.

        Returns:
            PolynomialRoot: Root with the highest multiplicity.

        Raises:
            AssertionError: If the roots are not equivalent.
        """

        assert self.is_equivalent(root), "Roots are not equivalent."

        return PolynomialRoot(
            value=self.value,
            multiplicity=max(self.multiplicity, root.multiplicity),
        )

    def __hash__(self) -> int:
        """Return a hash of the root."""
        return hash((self.value, self.multiplicity))
