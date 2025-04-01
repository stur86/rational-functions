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
    corresponding to a single real or complex pole r of multiplicity k in the
    denominator, taking the form

    $$
    R(x) = \\frac{a}{(x-r)^k}
    $$

    """

    _root: PolynomialRoot
    _coef: complex

    def __init__(self, root: PolynomialRoot, coef: complex):
        """Create a new RationalTermSingle instance.

        Args:
            root (PolynomialRoot): Polynomial root for the denominator
            coef (complex): Coefficient for the numerator
        """

        self._root = root
        self._coef = coef

    def __neg__(self) -> "RationalTerm":
        """Negate the term."""
        return self.__class__(self._root, -self._coef)

    @property
    def root(self) -> PolynomialRoot:
        """Return the root of the term."""
        return self._root

    @property
    def numerator(self) -> Polynomial:
        """Return the numerator of the term."""
        return Polynomial([self._coef])

    @property
    def denominator(self) -> Polynomial:
        """Return the denominator of the term."""
        return self._root.monic_polynomial()

    @classmethod
    def product(
        cls, term1: "RationalTerm", term2: "RationalTerm"
    ) -> list["RationalTerm"]:
        """Compute and decompose the product of two rational terms. This method
        is not implemented as an overloaded operator __mul__ since the output is
        not a RationalTerm but a list of RationalTerms.

        Args:
            term1 (RationalTerm): First term
            term2 (RationalTerm | Polynomial): Second term
        Returns:
            list[RationalTerm]: List of terms in the product
        """


        r1 = term1._root
        r2 = term2._root

        if r1.is_equivalent(r2):
            return [RationalTerm(
                r1.with_multiplicity(r1.multiplicity+r2.multiplicity),
                term1._coef*term2._coef
            )]

        roots = [r1, r2]
        num = Polynomial([term1._coef*term2._coef])
        return partial_frac_decomposition(num, roots)

    @classmethod
    def product_w_polynomial(
        cls, term: "RationalTerm", poly: Polynomial
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

        num = poly * term._coef
        r = term._root

        den = r.monic_polynomial()
        poly_out = num // den
        poly_rem = num % den

        terms = partial_frac_decomposition(poly_rem, [r])

        return terms, poly_out

    def __call__(self, x: ArrayLike) -> ArrayLike:
        """Evaluate the term at the given x values.

        Args:
            x (ArrayLike): values at which to evaluate the term

        Returns:
            ArrayLike: evaluated values
        """

        r = self._root
        den = x - r.value
        return self._coef / den**r.multiplicity

    def deriv(self) -> "RationalTerm":
        """Compute the derivative of the term.

        Returns:
            RationalTerm: Term of the derivative
        """

        r = self._root
        r_d = PolynomialRoot(value=r.value, multiplicity=r.multiplicity + 1)
        a = -self._coef * r.multiplicity
        return RationalTerm(r_d, a)

    def integ(self) -> tuple[RationalIntegralGeneralTerm]:
        """Compute the integral of the term.

        Returns:
            tuple[RationalIntegralGeneralTerm]: Terms of the integral
        """

        r = self._root
        if r.multiplicity == 1:
            return (RationalIntegralLogTerm(self._coef, r.value),)
        else:
            r_i = r.with_multiplicity(r.multiplicity - 1)
            a = -self._coef / r_i.multiplicity
            return (RationalTerm(r_i, a),)

    def __str__(self) -> str:
        """Print the term."""
        num = self._coef
        den = Polynomial([-self._root.value, 1.0])
        mul_str = ""
        if self._root.multiplicity > 1:
            mul_str = str(self._root.multiplicity).translate(
                Polynomial._superscript_mapping
            )

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
