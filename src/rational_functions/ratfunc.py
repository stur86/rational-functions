import numpy as np
from functools import cached_property
from typing import Union
from numpy.typing import ArrayLike
from numpy.polynomial import Polynomial
from rational_functions.terms import RationalTerm
from rational_functions.decomp import catalogue_roots, partial_frac_decomposition
from rational_functions.roots import PolynomialRoot
from rational_functions.lcm import RootLCM
from rational_functions.utils import as_polynomial, PolynomialDef
from rational_functions.integral import RationalFunctionIntegral, _integrate_terms
from dataclasses import dataclass

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

    # Approximation options
    @dataclass
    class ApproximationOptions:
        atol: float = 0.0
        rtol: float = 0.0
        ztol: float = 0.0

    __approx_opts: ApproximationOptions = ApproximationOptions()

    def __init__(
        self,
        terms: list[RationalTerm],
        poly: PolynomialDef | None = None,
        atol: float | None = None,
        rtol: float | None = None,
        ztol: float | None = None,
    ) -> None:
        """Initialize the rational function. Terms with very close poles or
        near-zero coefficients will be grouped together and simplified
        according to the tolerances passed to the constructor or the global
        approximation options (see set_approximation_options).

        Args:
            terms (list[RationalTerm]): List of terms.
            poly (Polynomial, optional): Residual polynomial. Defaults to None.
            atol (float, optional): Absolute tolerance for root equivalence. Defaults to None.
            rtol (float, optional): Relative tolerance for root equivalence. Defaults to None.
            ztol (float, optional): Absolute tolerance below which imaginary or real part of
                roots will be approximated to zero. Defaults to None.
        """

        atol = atol if atol is not None else self.__approx_opts.atol
        rtol = rtol if rtol is not None else self.__approx_opts.rtol
        ztol = ztol if ztol is not None else self.__approx_opts.ztol

        self._terms = RationalTerm.simplify(terms, atol, rtol)
        self._lcm = RootLCM([term.denominator_root for term in self._terms])
        self._poly = as_polynomial(poly) if poly is not None else Polynomial([0.0])
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

        sum_terms = self._terms + other_terms

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

        ztol = self.__approx_opts.ztol

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
            new_terms, new_poly = RationalTerm.product_w_polynomial(
                term, other_poly, ztol=ztol
            )
            mul_terms.extend(new_terms)
            mul_poly += new_poly

        for term in other_terms:
            # Multiply the term by the other polynomial
            new_terms, new_poly = RationalTerm.product_w_polynomial(
                term, self._poly, ztol=ztol
            )
            mul_terms.extend(new_terms)
            mul_poly += new_poly

        # Term-term multiplication
        for term1 in self._terms:
            for term2 in other_terms:
                new_terms = RationalTerm.product(term1, term2, ztol=ztol)
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
            # Find extended numerator for both
            ext_numerator = self.numerator + self._poly * self.denominator
            other_ext_numerator = other.numerator + other._poly * other.denominator
            numerator = ext_numerator * other.denominator
            denominator = other_ext_numerator * self.denominator
            return RationalFunction.from_fraction(
                numerator,
                denominator,
                atol=self.__approx_opts.atol,
                rtol=self.__approx_opts.rtol,
                ztol=self.__approx_opts.ztol,
            )
        elif isinstance(other, Polynomial):
            # We must divide by the highest order coefficience since the
            # denominator will always be monic
            numerator = (self.numerator + self._poly * self.denominator) / other.coef[
                -1
            ]
            den_poles = self.poles + catalogue_roots(
                other,
                atol=self.__approx_opts.atol,
                rtol=self.__approx_opts.rtol,
                ztol=self.__approx_opts.ztol,
            )
            return RationalFunction.from_poles(
                numerator, den_poles, ztol=self.__approx_opts.ztol
            )
        elif np.isscalar(other):
            # Divide each term by the scalar
            new_poly = self._poly / other
            new_terms = map(
                lambda t: RationalTerm(t.pole, t.coef / other, order=t.order),
                self._terms,
            )
            return RationalFunction(new_terms, new_poly)

        return NotImplemented

    def __pow__(self, exponent: int) -> "RationalFunction":
        """Raise the rational function to a power.

        Args:
            exponent (int): Exponent to raise the rational function to.

        Returns:
            RationalFunction: Resulting rational function.
        """
        if not exponent % 1 == 0:
            # Only integer powers are allowed
            raise ValueError("Power must be an integer.")
        exponent = int(exponent)

        if exponent == 0:
            return RationalFunction([], Polynomial([1.0]))
        elif exponent < 0:
            return self.reciprocal() ** -exponent

        ans = RationalFunction(self._terms, self._poly)
        for _ in range(exponent - 1):
            ans = ans * self
        return ans

    def reciprocal(
        self,
        atol: float | None = None,
        rtol: float | None = None,
        ztol: float | None = None,
    ) -> "RationalFunction":
        r"""Get the reciprocal of the rational function.

        $$
        R(x) = \frac{P(x)}{Q(x)} \implies R^{-1}(x) = \frac{Q(x)}{P(x)}
        $$

        Args:
            atol (float | None, optional): Absolute tolerance for root equivalence. Defaults to None.
            rtol (float | None, optional): Relative tolerance for root equivalence. Defaults to None.
            ztol (float | None, optional): Absolute tolerance below which imaginary or real part of
                roots will be approximated to zero. Defaults to None.

        Returns:
            RationalFunction: Inverse of the rational function.
        """
        numerator = self.denominator
        denominator = self.numerator + self._poly * numerator

        atol = atol if atol is not None else self.__approx_opts.atol
        rtol = rtol if rtol is not None else self.__approx_opts.rtol
        ztol = ztol if ztol is not None else self.__approx_opts.ztol

        return RationalFunction.from_fraction(
            numerator, denominator, atol=atol, rtol=rtol, ztol=ztol
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

    def integ(
        self, force_iobj: bool = False
    ) -> Union[RationalFunctionIntegral, "RationalFunction"]:
        """Integrate the rational function.

        Args:
            force_iobj (bool, optional): Force the integral to be a RationalFunctionIntegral object.

        Returns:
            Union[RationalFunctionIntegral, RationalFunction]: Integral of the rational function.
        """

        atol = self.__approx_opts.atol
        rtol = self.__approx_opts.rtol

        int_terms = _integrate_terms(self._terms, atol=atol, rtol=rtol)
        int_poly = self._poly.integ()
        # Check if all terms are RationalTerms
        if (not force_iobj) and all(
            isinstance(term, RationalTerm) for term in int_terms
        ):
            # If all terms are RationalTerms, return a RationalFunction
            return RationalFunction(int_terms, int_poly)

        return RationalFunctionIntegral(int_terms, int_poly)

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

    def __str__(self) -> str:
        """String representation of the rational function.

        Returns:
            str: String representation.
        """
        ans = f"({self.numerator})/({self.denominator})"
        if self._poly != Polynomial([0.0]):
            ans = f"{self._poly} + {ans}"

        return ans

    def __array__(self) -> None:
        # This method is necessary to avoid
        # the Polynomial __add__ and __mul__ methods
        # being called when adding or multiplying
        # Polynomial with RationalFunctions.

        # It works by raising an error in the
        # polyutils.as_series function,
        # which is a necessary step in all
        # the Polynomial operations.
        raise RuntimeError("Cannot convert RationalFunction to array.")

    @classmethod
    def from_fraction(
        cls,
        numerator: PolynomialDef,
        denominator: PolynomialDef,
        atol: float | None = None,
        rtol: float | None = None,
        ztol: float | None = None,
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
            atol (float, optional): Absolute tolerance for root equivalence. Defaults to None (use global setting).
            rtol (float, optional): Relative tolerance for root equivalence. Defaults to None (use global setting).
            ztol (float, optional): Absolute tolerance below which imaginary or real part of
                roots will be approximated to zero. Defaults to None (use global setting).

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

        atol = atol if atol is not None else cls.__approx_opts.atol
        rtol = rtol if rtol is not None else cls.__approx_opts.rtol
        ztol = ztol if ztol is not None else cls.__approx_opts.ztol

        # Polynomial part
        poly_quot = numerator // denominator
        # Residual numerator
        poly_rem = numerator % denominator

        # Find roots
        if denominator.degree() == 0:
            # No roots, return polynomial part
            return cls([], poly_quot)
        den_roots = catalogue_roots(denominator, atol=atol, rtol=rtol, ztol=ztol)
        rterms = partial_frac_decomposition(poly_rem.coef, den_roots, ztol=ztol)

        return cls(rterms, poly_quot)

    @classmethod
    def from_poles(
        cls,
        numerator: PolynomialDef,
        poles: list[PolynomialRoot],
        ztol: float | None = None,
    ) -> "RationalFunction":
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
            ztol (float, optional): Absolute tolerance below which imaginary or real part of
                roots will be approximated to zero. Defaults to None (use global setting).

        Returns:
            RationalFunction: Rational function object.
        """

        numerator = as_polynomial(numerator)

        ztol = ztol if ztol is not None else cls.__approx_opts.ztol

        lcm = RootLCM(poles)
        denominator = lcm.polynomial
        poly = numerator // denominator
        poly_rem = numerator % denominator
        rterms = partial_frac_decomposition(poly_rem.coef, lcm.roots, ztol=ztol)

        return cls(rterms, poly)

    @classmethod
    def from_roots_and_poles(
        cls,
        roots: list[PolynomialRoot],
        poles: list[PolynomialRoot],
        ztol: float | None = None,
    ) -> "RationalFunction":
        """Construct a RationalFunction from a list of roots
        for the numerator and a list of poles for the denominator.

        Args:
            roots (list[PolynomialRoot]): Roots of the numerator.
            poles (list[PolynomialRoot]): Poles of the denominator.
            ztol (float, optional): Absolute tolerance below which imaginary or real part of
                roots will be approximated to zero. Defaults to None (use global setting).

        Returns:
            RationalFunction: Rational function object.
        """

        # Build the list of roots
        root_list: list[complex] = np.concatenate(
            [[r.value] * r.multiplicity for r in roots]
        )

        return cls.from_poles(Polynomial.fromroots(root_list), poles, ztol=ztol)

    @classmethod
    def cauchy(cls, x0: float, w: float) -> "RationalFunction":
        r"""Construct a Cauchy distribution rational function,
        normalized to 1:

        $$
            R(x) = \frac{1}{\pi w} \frac{1}{(x-x_0)^2 + w^2}
        $$

        Args:
            x0 (float): Location parameter.
            w (float): Scale parameter.

        Returns:
            RationalFunction: Cauchy distribution rational function.
        """

        assert np.isreal(x0), "x0 must be real"
        assert np.isreal(w), "w must be real"
        assert w > 0, "w must be positive"

        r = x0 + 1.0j * w
        a = -0.5j / (np.pi * w**2)

        return RationalFunction([RationalTerm(r, a), RationalTerm(r.conjugate(), -a)])

    @classmethod
    def set_approximation_options(
        cls,
        atol: float | None = None,
        rtol: float | None = None,
        ztol: float | None = None,
    ) -> None:
        """Set the approximation options for the rational function.
        They control how the poles of a rational function are grouped
        together and approximated. These are used as defaults in from_fraction,
        and whenever they can't be set by the user directly like in division.

        Any value that is not passed gets left to its pre-existing value.

        Args:
            atol (float | None, optional): Absolute tolerance to consider two poles identical. Defaults to None.
            rtol (float | None, optional): Relative tolerance to consider two poles identical. Defaults to None.
            ztol (float | None, optional): Absolute tolerance below which imaginary or real part of
                values will be approximated to zero. Defaults to None.
        """

        if atol is not None:
            cls.__approx_opts.atol = atol
        if rtol is not None:
            cls.__approx_opts.rtol = rtol
        if ztol is not None:
            cls.__approx_opts.ztol = ztol


__all__ = ["RationalFunction"]
