from numpy.polynomial import Polynomial
from collections import defaultdict
from itertools import chain
from typing import Mapping
from .roots import PolynomialRoot


class RootLCM:
    """Least Common Multiple of a product of roots"""

    _factors: defaultdict[complex, int]

    def __init__(self, roots: list[PolynomialRoot]) -> None:
        """Initialize a Root Least Common Multiple object.

        Args:
            roots (list[PolynomialRoot]): roots to reduce to the least
                common multiple
        """

        self._factors = defaultdict(int)

        for root in roots:
            self._factors[root.value] = max(
                self._factors[root.value], root.multiplicity
            )

    @staticmethod
    def _poly_from_roots(roots: Mapping[complex, int]) -> Polynomial:
        """Return the polynomial whose roots are the given roots."""
        roots = list(chain(*[[k] * v for k, v in roots.items()]))
        if len(roots) == 0:
            return Polynomial([1.0])
        return Polynomial.fromroots(roots)

    @property
    def roots(self) -> list[PolynomialRoot]:
        """Return the roots of the polynomial."""
        return [PolynomialRoot(k, v) for k, v in self._factors.items()]

    @property
    def polynomial(self) -> Polynomial:
        """Return the polynomial whose roots are the least common multiple of the roots."""
        return self._poly_from_roots(self._factors)

    def residual(self, root: complex, multiplicity: int) -> Polynomial:
        """Return the residual of the polynomial when divided by (x - root) ** multiplicity."""

        m = self._factors[root]
        if m < multiplicity:
            raise ValueError(
                f"Root {root} has multiplicity {m}, but {multiplicity} is required"
            )

        residual_factors = self._factors.copy()
        residual_factors[root] -= multiplicity
        residual_poly = self._poly_from_roots(residual_factors)

        return residual_poly

    def __str__(self) -> str:
        """Return a string representation of the LCM."""
        return (
            "RootLCM{"
            + ", ".join(f"(r={k}, m={v})" for k, v in self._factors.items())
            + "}"
        )
