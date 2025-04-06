import numpy as np
from functools import cached_property
from typing import Union
from numpy.typing import ArrayLike
from numpy.polynomial import Polynomial
from rational_functions.terms import RationalTerm
from rational_functions.decomp import catalogue_roots, partial_frac_decomposition
from rational_functions.roots import PolynomialRoot
from rational_functions.lcm import RootLCM
from rational_functions.utils import as_polynomial

_RFuncOpCompatibleType = Union["RationalFunction", Polynomial, np.number]


class RationalFunction:
    r"""Rational function represented by
    sum of terms with a single pole for the proper part,
    plus a residual polynomial for the improper part.

    The rational function, given a polynomial
    $p(x)$ and a list of terms with roots $r_i$, coefficients $c_i$
    and order $k_i$, is defined as:

    $$
    R(x) = p(x)+\sum_{i=1}^n \frac{c_i}{(x - r_i)^{k_i}}
    $$



    """

    _poly: Polynomial
    _terms: list[RationalTerm]
    _lcm: RootLCM

    def __init__(self, terms: list[RationalTerm], poly: Polynomial | None = None):
        """Initialize the rational function.

        Args:
            terms (list[RationalTerm]): List of terms.
            poly (Polynomial, optional): Residual polynomial. Defaults to None.
        """

        self._terms = RationalTerm.simplify(terms)
        self._lcm = RootLCM([term.denominator_root for term in self._terms])
        self._poly = poly if poly is not None else Polynomial([0.0])
        self._poly = self._poly.trim()

    @property
    def poles(self) -> list[PolynomialRoot]:
        """Get the poles of the rational function.

        Returns:
            list[PolynomialRoot]: List of poles.
        """

        return self._lcm.roots

    @cached_property
    def numerator(self) -> Polynomial:
        """Get the numerator polynomial of the rational function.

        Returns:
            Polynomial: Numerator polynomial.
        """
        num = Polynomial([0.0])
        for term in self._terms:
            num += term.coef * self._lcm.residual(term.pole, term.order)

        return num

    @cached_property
    def denominator(self) -> Polynomial:
        """Get the denominator polynomial of the rational function.

        Returns:
            Polynomial: Denominator polynomial.
        """
        return self._lcm.polynomial

    def __neg__(self) -> "RationalFunction":
        """Negate the rational function.

        Returns:
            RationalFunction: Negated rational function.
        """
        neg_terms = [term.__neg__() for term in self._terms]
        return RationalFunction(neg_terms, -self._poly)

    def __add__(self, other: _RFuncOpCompatibleType) -> "RationalFunction":
        """Add two rational functions.

        Args:
            other (RationalFunction): Other rational function to add.

        Returns:
            RationalFunction: Resulting rational function.
        """
        other_terms = []
        other_poly: Polynomial

        is_other_ratfunc = isinstance(other, RationalFunction)

        if is_other_ratfunc:
            other_terms = list(other._terms)
            other_poly = other._poly
        elif isinstance(other, Polynomial):
            other_poly = other
        elif np.isscalar(other):
            # If other is a scalar, convert it to a polynomial
            other_poly = Polynomial([other])
        else:
            return NotImplemented

        sum_terms = RationalTerm.simplify(self._terms + other_terms)

        ans = RationalFunction(
            sum_terms,
            self._poly + other_poly,
        )

        return ans

    def __radd__(
        self,
        other: _RFuncOpCompatibleType,
    ) -> "RationalFunction":
        """Right add operator for RationalFunction.

        Args:
            other (RationalFunction): Other rational function to add.

        Returns:
            RationalFunction: Resulting rational function.
        """
        return self + other

    def __sub__(self, other: _RFuncOpCompatibleType) -> "RationalFunction":
        """Subtract two rational functions.

        Args:
            other (RationalFunction): Other rational function to subtract.

        Returns:
            RationalFunction: Resulting rational function.
        """
        return self + (-other)

    def __rsub__(self, other: _RFuncOpCompatibleType) -> "RationalFunction":
        """Right subtract operator for RationalFunction.

        Args:
            other (RationalFunction): Other rational function to subtract.

        Returns:
            RationalFunction: Resulting rational function.
        """

        return (-self) + other

    def __mul__(self, other: _RFuncOpCompatibleType) -> "RationalFunction":
        """Multiply two rational functions.

        Args:
            other (RationalFunction): Other rational function to multiply.

        Returns:
            RationalFunction: Resulting rational function.
        """

        other_poly: Polynomial
        other_terms: list[RationalTerm] = []
        is_other_ratfunc = isinstance(other, RationalFunction)

        if is_other_ratfunc:
            other_poly = other._poly
            other_terms = list(other._terms)
        elif isinstance(other, Polynomial):
            other_poly = other
        elif np.isscalar(other):
            # If other is a scalar, convert it to a polynomial
            other_poly = Polynomial([other])
        else:
            return NotImplemented

        mul_poly = self._poly * other_poly
        mul_terms: list[RationalTerm] = []

        for term in self._terms:
            # Multiply the term by the other polynomial
            new_terms, new_poly = RationalTerm.product_w_polynomial(term, other_poly)
            mul_terms.extend(new_terms)
            mul_poly += new_poly

        for term in other_terms:
            # Multiply the term by the other polynomial
            new_terms, new_poly = RationalTerm.product_w_polynomial(term, self._poly)
            mul_terms.extend(new_terms)
            mul_poly += new_poly

        # Term-term multiplication
        for term1 in self._terms:
            for term2 in other_terms:
                new_terms = RationalTerm.product(term1, term2)
                mul_terms.extend(new_terms)

        return RationalFunction(mul_terms, mul_poly)

    def __rmul__(self, other: _RFuncOpCompatibleType) -> "RationalFunction":
        """Right multiply operator for RationalFunction.

        Args:
            other (RationalFunction): Other rational function to multiply.

        Returns:
            RationalFunction: Resulting rational function.
        """

        return self * other

    def __truediv__(self, other: _RFuncOpCompatibleType) -> "RationalFunction":
        """Divide two rational functions.

        Args:
            other (RationalFunction): Other rational function to divide.

        Returns:
            RationalFunction: Resulting rational function.
        """
        if isinstance(other, RationalFunction):
            return self * other.reciprocal()
        elif isinstance(other, Polynomial):
            numerator = self.numerator
            denominator = self.denominator * other
            return RationalFunction.from_fraction(
                numerator,
                denominator,
            )
        elif np.isscalar(other):
            # Divide each term by the scalar
            new_poly = self._poly / other
            new_terms = map(lambda t: RationalTerm(t.pole, t.coef / other), self._terms)
            return RationalFunction(list(new_terms), new_poly)

        return NotImplemented

    def reciprocal(self) -> "RationalFunction":
        r"""Get the reciprocal of the rational function.

        $$
        R(x) = \frac{P(x)}{Q(x)} \implies R^{-1}(x) = \frac{Q(x)}{P(x)}
        $$

        Returns:
            RationalFunction: Inverse of the rational function.
        """
        numerator = self.denominator
        denominator = self.numerator + self._poly * numerator

        return RationalFunction.from_fraction(
            numerator,
            denominator,
        )

    def deriv(self, m: int = 1) -> "RationalFunction":
        r"""Differentiate the rational function in x.
        
        $$
        \begin{align*}
            \frac{d^m R(x)}{dx^m} &= \frac{d^{m-1}R'(x)}{dx^{m-1}} = \\
            &= \frac{d^{m-1}}{dx^{m-1}} \frac{P'(x)Q(x)-P(x)Q'(x)}{Q^2(x)}
        \end{align*}
        $$

        Args:
            m (int, optional): Order of the derivative. Defaults to 1.

        Returns:
            RationalFunction: Derivative of the rational function.
        """

        diff_poly = self._poly.deriv(m)
        diff_terms = [term.deriv(m) for term in self._terms]

        return RationalFunction(diff_terms, diff_poly)

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
        numerator: Polynomial | ArrayLike,
        denominator: Polynomial | ArrayLike,
        atol: float = 0.0,
        rtol: float = 0.0,
        imtol: float = 0.0,
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
            numerator (Polynomial | ArrayLike): Numerator polynomial, or series of coefficients in increasing order.
            denominator (Polynomial | ArrayLike): Denominator polynomial, or series of coefficients in increasing order.
            atol (float, optional): Absolute tolerance for root equivalence. Defaults to 0.
            rtol (float, optional): Relative tolerance for root equivalence. Defaults to 0.
            imtol (float, optional): Tolerance for imaginary part of roots to be considered
                zero. Defaults to 0.

        Returns:
            RationalFunction: Rational function object.
        """
        
        # Cast to polynomial
        numerator = as_polynomial(numerator)
        denominator = as_polynomial(denominator)

        # Make denominator monic
        c = denominator.coef[-1]
        denominator /= c
        numerator /= c

        # Polynomial part
        poly_quot = numerator // denominator
        # Residual numerator
        poly_rem = numerator % denominator

        # Find roots
        if denominator.degree() == 0:
            # No roots, return polynomial part
            return cls([], poly_quot)
        den_roots = catalogue_roots(denominator, atol=atol, rtol=rtol, imtol=imtol)
        rterms = partial_frac_decomposition(
            poly_rem.coef,
            den_roots,
        )

        return cls(rterms, poly_quot)

    @classmethod
    def from_poles(
        cls,
        numerator: Polynomial,
        poles: list[PolynomialRoot],
    ) -> None:
        """Construct a RationalFunction from a list of poles
        and a numerator polynomial. This method is more efficient
        and numerically stable than using the from_fraction
        method, especially for high degree or ill-conditioned
        denominators.

        Note:
            The numerator polynomial should be scaled for a
            denominator polynomial with leading coefficient 1.

        Args:
            numerator (Polynomial): Numerator polynomial.
            poles (list[PolynomialRoot]): Poles of the rational function.

        Returns:
            RationalFunction: Rational function object.
        """

        lcm = RootLCM(poles)
        denominator = lcm.polynomial
        poly = numerator // denominator
        poly_rem = numerator % denominator
        rterms = partial_frac_decomposition(
            poly_rem.coef,
            poles,
        )
        return cls(rterms, poly)

    def __array__(self) -> None:
        # This method is necessary to avoid 
        # the Polynomial __add__ and __mul__ methods
        # being called when adding or multiplying
        # Polynomial with RationalFunctions.
        
        # It works by raising an error in the 
        # polyutils.as_series function,        
        # which is a necessary step in all
        # the Polynomial operations.
        raise RuntimeError(
            "Cannot convert RationalFunction to array. Use __call__ instead."
        )
