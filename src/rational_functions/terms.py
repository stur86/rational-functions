"""Individual terms in a proper rational function."""

from abc import ABC, abstractmethod
from typing import Union
import numpy as np
from numpy.typing import ArrayLike
from numpy.polynomial import Polynomial
from .roots import PolynomialRoot
from .decomp import partial_frac_decomposition

RationalIntegralGeneralTerm = Union["RationalIntegralTermBase", "RationalTerm"]


class RationalTerm:
    """A single term in a proper rational function,
    corresponding to a single real or complex pole r of order k in the
    denominator, taking the form

    $$
    R(x) = \\frac{c}{(x-r)^k}
    $$

    """

    _r: complex
    _c: complex
    _k: int

    def __init__(self, pole: complex, coef: complex, order: int = 1):
        """Create a new RationalTermSingle instance.

        Args:
            pole (complex): Value of the pole for the denominator
            coef (complex): Coefficient for the numerator
            order (int): Order of the term
                Defaults to 1.

        Raises:
            ValueError: If the order is less than 1.
        """

        if order < 1:
            raise ValueError("Order must be greater than or equal to 1")

        self._r = pole
        self._c = coef
        self._k = order

    def __neg__(self) -> "RationalTerm":
        """Negate the term."""
        return self.__class__(self._r, -self._c, self._k)

    def __eq__(self, other: object) -> bool:
        """Check if two terms are equal."""
        if not isinstance(other, RationalTerm):
            return False
        return self._r == other._r and self._c == other._c and self._k == other._k

    def __hash__(self) -> int:
        """Hash the term."""
        return hash((self._r, self._c, self._k))

    @property
    def pole(self) -> complex:
        """Return the pole of the term."""
        return self._r

    @property
    def coef(self) -> complex:
        """Return the coefficient of the term."""
        return self._c

    @property
    def order(self) -> int:
        """Return the order of the term."""
        return self._k

    @property
    def denominator_root(self) -> PolynomialRoot:
        """Return the root of the denominator."""
        return PolynomialRoot(self._r, self._k)

    @property
    def denominator(self) -> Polynomial:
        """Return the denominator of the term."""
        return Polynomial([-self._r, 1.0]) ** self._k

    @staticmethod
    def product(term1: "RationalTerm", term2: "RationalTerm") -> list["RationalTerm"]:
        """Compute and decompose the product of two rational terms. This method
        is not implemented as an overloaded operator __mul__ since the output is
        not a RationalTerm but a list of RationalTerms.

        Args:
            term1 (RationalTerm): First term
            term2 (RationalTerm | Polynomial): Second term
        Returns:
            list[RationalTerm]: List of terms in the product
        """

        r1 = term1._r
        r2 = term2._r

        if r1 == r2:
            return [
                RationalTerm(
                    r1,
                    term1._c * term2._c,
                    order=term1._k + term2._k,
                )
            ]

        roots = [PolynomialRoot(r1, term1._k), PolynomialRoot(r2, term2._k)]
        return partial_frac_decomposition([term1.coef * term2.coef], roots)

    @staticmethod
    def product_w_polynomial(
        term: "RationalTerm", poly: Polynomial
    ) -> tuple[list["RationalTerm"], Polynomial]:
        """Compute and decompose the product of a rational term and a polynomial.
        This method is not implemented as an overloaded operator __mul__ since the output is
        not a RationalTerm but a list of RationalTerms and a Polynomial.

        Args:
            term (RationalTerm): Rational term
            poly (Polynomial): Polynomial to multiply with

        Returns:
            tuple[list[RationalTerm], Polynomial]: List of terms in the product and the remaining polynomial
        """

        num = poly * term._c

        den = term.denominator
        poly_out = num // den
        poly_rem = num % den

        terms = partial_frac_decomposition(
            poly_rem.coef, [PolynomialRoot(term.pole, term.order)]
        )

        return terms, poly_out

    @staticmethod
    def simplify(terms: list["RationalTerm"]) -> list["RationalTerm"]:
        """Simplify a list of RationalTerms by combining terms with the same pole
        and order.

        Args:
            terms (list[RationalTerm]): List of RationalTerms to simplify

        Returns:
            list[RationalTerm]: Simplified list of RationalTerms
        """

        simplified_terms: dict[tuple[complex, int], complex] = {}
        for term in terms:
            key = (term.pole, term.order)
            if key in simplified_terms:
                simplified_terms[key] += term.coef
            else:
                simplified_terms[key] = term.coef

        return list(
            [
                RationalTerm(r, c, k)
                for (r, k), c in simplified_terms.items()
                if c != 0.0
            ]
        )

    def __call__(self, x: ArrayLike) -> ArrayLike:
        """Evaluate the term at the given x values.

        Args:
            x (ArrayLike): values at which to evaluate the term

        Returns:
            ArrayLike: evaluated values
        """

        den = x - self.pole
        return self._c / den**self.order

    def deriv(self, m: int = 1) -> "RationalTerm":
        """Compute the derivative of the term.

        Args:
            m (int): Order of the derivative
                Defaults to 1.

        Returns:
            RationalTerm: Term of the derivative
        """

        assert m >= 1, "Derivative order must be greater than or equal to 1"

        c = self._c * np.prod(self.order + np.arange(m)) * (-1) ** m
        return RationalTerm(self.pole, c, self.order + m)

    def integ(self) -> RationalIntegralGeneralTerm:
        """Compute the integral of the term.

        Returns:
            RationalIntegralGeneralTerm: Integral
        """

        if self.order == 1:
            return RationalIntegralLogTerm(self._c, self.pole)
        else:
            c = -self._c / (self.order - 1)
            return RationalTerm(self.pole, c, self.order - 1)

    def __str__(self) -> str:
        """Print the term."""
        num = self._c
        den = Polynomial([-self.pole, 1.0])
        mul_str = ""
        if self.order > 1:
            mul_str = str(self.order).translate(Polynomial._superscript_mapping)

        return f"{num:.2G} / ({den}){mul_str}"


class RationalIntegralTermBase(ABC):
    """Base class for a single term in the integral of a proper rational function."""

    @abstractmethod
    def __call__(self, x: ArrayLike) -> ArrayLike:
        """Evaluate the term at the given x values."""
        ...


class RationalIntegralLogTerm(RationalIntegralTermBase):
    r"""Integral term of the form

    $$
    I(x) = a \log(x-r)
    $$

    """

    def __init__(self, a: float, r: complex):
        """Create a new RationalIntegralLogTerm instance.

        Args:
            a (float): Coefficient
            r (complex): Corresponding root
        """

        self._a = a
        self._r = r

    def __call__(self, x: ArrayLike) -> ArrayLike:
        """Evaluate the term at the given x values.

        Args:
            x (ArrayLike): values at which to evaluate the term

        Returns:
            ArrayLike: evaluated values
        """
        return self._a * np.log(x - self._r + 0.0j)


class RationalIntegralLogPairTerm(RationalIntegralTermBase):
    r"""Integral term of the form

    $$
    I(x) = a \log((x-\mathrm{Re}(r))^2+\mathrm{Im}(r)^2)
    $$
    """

    def __init__(self, a: float, r: complex):
        """Create a new RationalIntegralLogPairTerm instance.

        Args:
            a (float): Coefficient
            r (complex): Corresponding root
        """

        self._a = a
        self._x0 = r.real
        self._m = r.imag

    def __call__(self, x: ArrayLike) -> ArrayLike:
        """Evaluate the term at the given x values.

        Args:
            x (ArrayLike): values at which to evaluate the term

        Returns:
            ArrayLike: evaluated values
        """
        return 0.5 * self._a * np.log((x - self._x0) ** 2 + self._m**2 + 0.0j)


class RationalIntegralArctanTerm(RationalIntegralTermBase):
    r"""Integral term of the form

    $$
    I(x) = \frac{a}{\mathrm{Im}(r)} \arctan((x-\mathrm{Re}(r))/\mathrm{Im}(r))
    $$
    """

    def __init__(self, a: float, r: complex):
        """Create a new RationalIntegralArctanTerm instance.

        Args:
            a (float): Coefficient
            r (complex): Corresponding root
        """

        self._a = a
        self._x0 = r.real
        self._m = r.imag

    def __call__(self, x: ArrayLike) -> ArrayLike:
        """Evaluate the term at the given x values.

        Args:
            x (ArrayLike): values at which to evaluate the term

        Returns:
            ArrayLike: evaluated values
        """

        return self._a / self._m * np.arctan((x - self._x0) / self._m)
