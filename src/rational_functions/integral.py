import warnings
from typing import Union
from .terms import (
    RationalIntegralGeneralTerm,
    RationalIntegralLogTerm,
    RationalTerm,
    RationalIntegralArctanTerm,
    RationalIntegralLogPairTerm,
)
import numpy as np
from numpy.typing import ArrayLike
from numpy.polynomial import Polynomial
from rational_functions.utils import as_polynomial, PolynomialDef


def _find_cconj_pairs(
    terms: list[RationalTerm],
    atol: float = 0.0,
    rtol: float = 0.0,
) -> tuple[list[tuple[RationalTerm, RationalTerm]], list[RationalTerm]]:
    """Identify complex conjugate pairs in a list of rational terms,
    and separate them from the rest. Tolerances can be set to control
    the comparison.

    Args:
        terms (list[RationalTerm]): List of RationalTerms to check for conjugate pairs.
        atol (float, optional): Absolute tolerance for comparison. Defaults to 0.0.
        rtol (float, optional): Relative tolerance for comparison. Defaults to 0.0.

    Returns:
        list[tuple[RationalTerm, RationalTerm]]: List of tuples containing conjugate pairs.
        list[RationalTerm]: List of remaining terms that are not part of a conjugate pair.
    """

    conj_pairs: list[tuple[RationalTerm, RationalTerm]] = []
    remaining_terms: list[RationalTerm] = []

    # Filter for terms that are guaranteed to be excluded
    # from conjugate pairing
    # (e.g., order > 1 or real pole)
    def _exclude_term(t: RationalTerm) -> bool:
        return (t.order > 1) or np.isclose(t.pole.imag, 0.0, atol=atol, rtol=rtol)

    excluded_idx = [i for i, t in enumerate(terms) if _exclude_term(t)]
    remaining_terms += [terms[i] for i in excluded_idx]
    terms = [t for i, t in enumerate(terms) if i not in excluded_idx]

    evaluated = [False] * len(terms)

    for i, t1 in enumerate(terms):
        if evaluated[i]:
            # Already evaluated
            continue

        for j, t2 in enumerate(terms[i + 1 :], start=i + 1):
            if evaluated[j]:
                continue

            if np.isclose(t1.pole, np.conj(t2.pole), atol=atol, rtol=rtol):
                # Found a conjugate pair
                conj_pairs.append((t1, t2))
                evaluated[i] = True
                evaluated[j] = True
                break

        if not evaluated[i]:
            # No conjugate pair found
            remaining_terms.append(t1)
            evaluated[i] = True
            continue

    assert all(evaluated), "Not all terms were evaluated."
    return conj_pairs, remaining_terms


def _int_cconj_pair(
    a1: complex, a2: complex, r: complex
) -> list[RationalIntegralGeneralTerm]:
    """Calculate the integral of a conjugate pair of complex poles.

    Args:
        a1 (complex): Coefficient for the first term.
        a2 (complex): Coefficient for the second term.
        r (complex): Pole for the first term; the second term is its conjugate.

    Returns:
        tuple[RationalIntegralLogPairTerm, RationalIntegralArctanTerm]: Integral terms for the conjugate pair.
    """

    out_terms: list[RationalIntegralGeneralTerm] = []

    c1 = a1 + a2
    if c1 != 0.0:
        out_terms.append(RationalIntegralLogPairTerm(c1, r))
    c2 = 1.0j * (a1 - a2) * np.imag(r)
    if c2 != 0.0:
        out_terms.append(RationalIntegralArctanTerm(c2, r))

    return out_terms


def _integrate_terms(
    terms: list[RationalTerm], atol: float = 0.0, rtol: float = 0.0
) -> list[RationalIntegralGeneralTerm]:
    """Integrate a list of rational terms and return the resulting integral terms.
    This function handles complex conjugate pairs of terms, identifying them with
    tolerances and integrating them appropriately.

    Args:
        terms (list[RationalTerm]): List of RationalTerms to integrate.
        atol (float): Absolute tolerance for comparing terms.
        rtol (float): Relative tolerance for comparing terms.

    Returns:
        list[RationalIntegralGeneralTerm]: List of integrated terms.
    """
    int_terms: list[RationalIntegralGeneralTerm] = []

    # First, split out the complex conjugate pairs
    cconj_terms, terms = _find_cconj_pairs(terms, atol=atol, rtol=rtol)

    # Now, first the regular terms
    for term in terms:
        if term.order == 1:
            int_terms.append(RationalIntegralLogTerm(term.coef, term.pole))
        elif term.order > 1:
            m = term.order - 1
            int_terms.append(RationalTerm(term.pole, -1 / m * term.coef, m))

    for t1, t2 in cconj_terms:
        int_terms += _int_cconj_pair(t1.coef, t2.coef, t1.pole)

    return int_terms


class RationalFunctionIntegral:
    """Class that encapsulates the integral of a rational function.
    It includes different kinds of terms, including logarithmic and arctangent terms.
    """

    _poly: Polynomial
    _terms: list[RationalIntegralGeneralTerm]

    def __init__(
        self,
        terms: list[RationalIntegralGeneralTerm],
        poly: PolynomialDef | None = None,
    ) -> None:
        """Initialize the RationalFunctionIntegral with a list of terms and an optional polynomial.

        Args:
            terms (list[RationalIntegralGeneralTerm]): List of terms to include in the integral.
            poly (PolynomialDef | None, optional): Polynomial part of the integral.
                If None, defaults to a zero polynomial.
        """

        if all(isinstance(term, RationalTerm) for term in terms):
            warnings.warn(
                "All terms are RationalTerms. Consider using a regular RationalFunction instead."
            )

        if poly is None:
            poly = Polynomial([0.0])

        self._poly = as_polynomial(poly)
        self._terms = list(terms)

    def definite(self, a: float, b: float) -> complex:
        """Evaluate the definite integral from a to b.

        Args:
            a (float): Lower limit of integration.
            b (float): Upper limit of integration.

        Returns:
            complex: Value of the definite integral.
        """
        assert np.real(a) == a, "Lower limit must be real."
        assert np.real(b) == b, "Upper limit must be real."

        return self(b) - self(a)

    def real_line(self, as_cauchy_pv: bool = False) -> complex:
        """Evaluate the integral over the real line,
        from negative infinity to positive infinity.

        Args:
            as_cauchy_pv (bool): If True, the integral should be interpreted
                as a Cauchy principal value. Otherwise, it is a regular integral
                and will always be NaN if the integral diverges.

        Returns:
            complex: Value of the integral over the real line.
        """

        ans = 0.0
        if self._poly.degree() > 0:
            # Polynomial part diverges
            return np.nan

        for t in self._terms:
            if isinstance(t, RationalTerm):
                # Contribution is zero
                continue
            elif isinstance(t, RationalIntegralLogTerm):
                ans += np.nan
            elif isinstance(t, RationalIntegralLogPairTerm):
                if as_cauchy_pv:
                    # Cauchy principal value is zero,
                    # as the term goes to infinity but is symmetric
                    continue
                else:
                    ans += np.nan
            elif isinstance(t, RationalIntegralArctanTerm):
                ans += t.real_line

        return ans

    def __call__(self, x: ArrayLike) -> ArrayLike:
        """Evaluate the integral at given points."""

        x = np.asarray(x)
        result = self._poly(x).astype(np.complex128)
        for term in self._terms:
            result += term(x)
        return result

    def __add__(
        self, other: Union[Polynomial, "RationalFunctionIntegral"]
    ) -> "RationalFunctionIntegral":
        other_terms: list[RationalIntegralGeneralTerm]
        other_poly: Polynomial

        if isinstance(other, RationalFunctionIntegral):
            other_terms = other._terms
            other_poly = other._poly
        elif isinstance(other, Polynomial):
            other_terms = []
            other_poly = as_polynomial(other)
        else:
            return NotImplemented

        return RationalFunctionIntegral(
            self._terms + other_terms,
            self._poly + other_poly,
        )

    def __array__(self) -> np.ndarray:
        raise RuntimeError("Cannot convert RationalFunctionIntegral to numpy array.")

    @classmethod
    def from_polynomial(cls, poly: PolynomialDef) -> "RationalFunctionIntegral":
        """Create a RationalFunctionIntegral from a polynomial part of a RationalFunction."""
        return cls([], poly.integ())

    @classmethod
    def from_rational_terms(
        cls,
        terms: list[RationalTerm],
        poly: PolynomialDef | None = None,
        atol: float = 0.0,
        rtol: float = 0.0,
    ) -> "RationalFunctionIntegral":
        """Create a RationalFunctionIntegral from a list of RationalTerms. Terms
        will be compared to detect conjugate pairs so they can be combined into
        appropriate integral terms; tolerances can be set to control this.

        Args:
            terms (list[RationalTerm]): List of RationalTerms to include in the integral.
            poly (PolynomialDef | None): Polynomial part of the rational function to
                integrate. If None, defaults to a zero polynomial.
            atol (float): Absolute tolerance for comparing terms.
            rtol (float): Relative tolerance for comparing terms.

        """
        int_terms: list[RationalIntegralGeneralTerm] = _integrate_terms(
            terms, atol=atol, rtol=rtol
        )

        int_poly: PolynomialDef = None
        if poly is not None:
            int_poly = as_polynomial(poly).integ()

        return cls(int_terms, int_poly)
