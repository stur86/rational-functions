import warnings
from .terms import RationalIntegralGeneralTerm, RationalIntegralLogTerm, RationalTerm
import numpy as np
from numpy.typing import ArrayLike
from numpy.polynomial import Polynomial
from rational_functions.utils import as_polynomial, PolynomialDef


class RationalFunctionIntegral:
    """Class that encapsulates the integral of a rational function.
    It includes different kinds of terms, including logarithmic and arctangent terms.
    """

    _poly: Polynomial
    _terms: list[RationalIntegralGeneralTerm]

    def __init__(
        self, terms: list[RationalIntegralGeneralTerm], poly: PolynomialDef
    ) -> None:
        if all(isinstance(term, RationalTerm) for term in terms):
            warnings.warn(
                "All terms are RationalTerms. Consider using a regular RationalFunction instead."
            )

        self._poly = as_polynomial(poly)
        self._terms = list(terms)

    def __call__(self, x: ArrayLike) -> ArrayLike:
        """Evaluate the integral at given points."""

        x = np.asarray(x)
        result = self._poly(x)
        for term in self._terms:
            result += term(x)
        return result

    def __add__(self, other: "RationalFunctionIntegral") -> "RationalFunctionIntegral":
        return RationalFunctionIntegral(
            self._terms + other._terms,
            self._poly + other._poly,
        )

    def __array__(self) -> np.ndarray:
        raise RuntimeError("Cannot convert RationalFunctionIntegral to numpy array.")

    @classmethod
    def from_polynomial(cls, poly: PolynomialDef) -> "RationalFunctionIntegral":
        """Create a RationalFunctionIntegral from a polynomial part of a RationalFunction."""
        return cls([], poly.integ())

    @classmethod
    def from_rational_terms(
        cls, terms: list[RationalTerm]
    ) -> "RationalFunctionIntegral":
        """Create a RationalFunctionIntegral from a list of RationalTerms."""
        int_terms: list[RationalIntegralGeneralTerm] = []

        for term in terms:
            if term.order == 1:
                int_terms.append(RationalIntegralLogTerm(term.coef, term.pole))
            elif term.order > 1:
                m = term.order - 1
                int_terms.append(RationalTerm(term.pole, -1 / m * term.coef, m))

        return cls(int_terms, Polynomial([0.0]))
