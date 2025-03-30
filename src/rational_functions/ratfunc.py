import numpy as np
from numpy.typing import ArrayLike
from numpy.polynomial import Polynomial
from rational_functions.terms import RationalTerm
from rational_functions.decomp import catalogue_roots, partial_frac_decomposition


class RationalFunction:
    """Rational function represented by
    sum of terms with a single pole for the proper part,
    plus a residual polynomial for the improper part.
    """

    _poly: Polynomial
    _terms: list[RationalTerm]

    def __init__(self, terms: list[RationalTerm], poly: Polynomial | None = None):
        """Initialize the rational function.

        Args:
            terms (list[RationalTerm]): List of terms.
            poly (Polynomial, optional): Residual polynomial. Defaults to None.
        """

        self._terms = terms
        self._poly = poly if poly is not None else Polynomial([0.0])
        self._poly = self._poly.convert().trim()

    def __call__(self, x: ArrayLike) -> ArrayLike:
        """Evaluate the rational function at given points.

        Args:
            x (ArrayLike): Points to evaluate the function at.

        Returns:
            ArrayLike: Evaluated values.
        """
        p_y = self._poly(x)
        t_y = np.sum([term(x) for term in self._terms], axis=0)

        return p_y + t_y

    @classmethod
    def from_fraction(
        cls,
        numerator: Polynomial,
        denominator: Polynomial,
        atol: float = 1e-8,
        rtol: float = 1e-5,
        imtol: float = 1e-13,
    ) -> "RationalFunction":
        """Construct a RationalFunction from a fraction of two
        polynomials.

        Warning:
            This method requires finding the roots of the denominator polynomial.
            This will cause numerical inaccuracy, especially for high degree
            or ill-conditioned polynomials. Also, pay attention to the tolerances
            used to identify equivalent roots. In general, it is recommended to build the
            RationalFunction from its terms whenever possible.

        Args:
            numerator (Polynomial): Numerator polynomial.
            denominator (Polynomial): Denominator polynomial.
            atol (float, optional): Absolute tolerance for root equivalence. Defaults to 1e-8.
            rtol (float, optional): Relative tolerance for root equivalence. Defaults to 1e-5.
            imtol (float, optional): Tolerance for imaginary part of roots to be considered
                zero. Defaults to 1e-13.

        Returns:
            RationalFunction: Rational function object.
        """

        # Polynomial part
        poly_quot = numerator // denominator
        # Residual numerator
        poly_rem = numerator % denominator

        # Find roots
        den_roots = catalogue_roots(denominator)
        rterms = partial_frac_decomposition(
            poly_rem, den_roots, atol=atol, rtol=rtol, imtol=imtol
        )

        return cls(rterms, poly_quot)
