from abc import ABC, abstractmethod
from typing import Union
import numpy as np
from numpy.typing import ArrayLike
from numpy.polynomial import Polynomial
from .types import PolynomialRoot

RationalIntegralGeneralTerm = Union["RationalIntegralTermBase", "RationalTermBase"]


class RationalTermBase(ABC):
    """Base class for a single term in a proper rational function."""

    _root: PolynomialRoot
    _coefs: ArrayLike

    def __init__(self, root: PolynomialRoot, coefs: ArrayLike):
        self._root = root
        self._coefs = np.asarray(coefs)

    @abstractmethod
    def __call__(self, x: ArrayLike) -> ArrayLike:
        """Evaluate the term at the given x values.

        Args:
            x (ArrayLike): values at which to evaluate the term

        Returns:
            ArrayLike: evaluated values
        """
        ...

    @abstractmethod
    def deriv(self) -> tuple["RationalTermBase", ...]:
        """Return the derivative of the term."""
        ...

    @abstractmethod
    def integ(self) -> tuple[RationalIntegralGeneralTerm, ...]:
        """Return the integral of the term."""
        ...

    @property
    def numerator(self) -> Polynomial:
        """Return the numerator of the term."""
        return Polynomial(np.trim_zeros(self._coefs, trim="b"))

    @property
    def denominator(self) -> Polynomial:
        """Return the denominator of the term."""
        raise NotImplementedError("Denominator property not implemented in base class")


class RationalTermSingle(RationalTermBase):
    """A single term in a proper rational function,
    corresponding to a single real or complex pole r of multiplicity k in the
    denominator, taking the form

    $$
    R(x) = \\frac{a}{(x-r)^k}
    $$

    """

    def __init__(self, root: PolynomialRoot, coefs: ArrayLike):
        """Create a new RationalTermSingle instance.

        Args:
            root (PolynomialRoot): Polynomial root for the denominator
            coefs (ArrayLike): Coefficients for the numerator
        """

        assert not root.is_complex_pair, "Complex pair roots not allowed"
        assert len(coefs) == 1, "Only one coefficient allowed for real root"

        super().__init__(root, coefs)

    def __call__(self, x: ArrayLike) -> ArrayLike:
        """Evaluate the term at the given x values.

        Args:
            x (ArrayLike): values at which to evaluate the term

        Returns:
            ArrayLike: evaluated values
        """

        r = self._root
        den = x - r.value
        num = self._coefs[0]
        return num / den**r.multiplicity

    def deriv(self) -> tuple["RationalTermSingle"]:
        """Compute the derivative of the term.

        Returns:
            tuple[RationalTermSingle]: Terms of the derivative
        """

        r = self._root
        r_d = PolynomialRoot(value=r.value, multiplicity=r.multiplicity + 1)
        a = -self._coefs[0] * r.multiplicity
        return (RationalTermSingle(r_d, [a]),)

    def integ(self) -> tuple[RationalIntegralGeneralTerm]:
        """Compute the integral of the term.

        Returns:
            tuple[RationalIntegralGeneralTerm]: Terms of the integral
        """

        r = self._root
        if r.multiplicity == 1:
            return (RationalIntegralLogTerm(self._coefs[0], r.value),)
        else:
            r_i = r.with_multiplicity(r.multiplicity - 1)
            a = -self._coefs[0] / r_i.multiplicity
            return (RationalTermSingle(r_i, [a]),)

    @property
    def denominator(self) -> Polynomial:
        """Return the denominator of the term."""
        return Polynomial([-self._root.value, 1]) ** self._root.multiplicity


class RationalTermComplexPair(RationalTermBase):
    """A single term in a proper rational function,
    corresponding to a complex conjugate pair of poles in the denominator,
    of the form

    $$
    R(x) = \\frac{a+bx}{((x-\\mathrm{Re}(r))^2+\\mathrm{Im}(r)^2)^k}
    $$

    """

    def __init__(self, root: PolynomialRoot, coefs: ArrayLike):
        """Create a new RationalTermComplexPair instance.

        Args:
            root (PolynomialRoot): Polynomial root for the denominator
            coefs (ArrayLike): Coefficients for the numerator
        """

        assert root.is_complex_pair, "Complex pair roots required"
        assert not root.is_real, "Complex pair roots must be complex"
        assert len(coefs) <= 2, "Two coefficients allowed at most for complex pair"
        coefs = np.pad(coefs, (0, 2 - len(coefs)), mode="constant", constant_values=0)
        super().__init__(root, coefs)

    def __call__(self, x: ArrayLike) -> ArrayLike:
        """Evaluate the term at the given x values.

        Args:
            x (ArrayLike): values at which to evaluate the term

        Returns:
            ArrayLike: evaluated values
        """

        r = self._root
        den = (x - r.real) ** 2 + r.imag**2
        num = self._coefs[0] + self._coefs[1] * x
        return num / den**r.multiplicity

    def deriv(self) -> tuple["RationalTermComplexPair", "RationalTermComplexPair"]:
        """Compute the derivative of the term.

        Returns:
            tuple[RationalTermComplexPair]: Terms of the derivative
        """

        r = self._root
        k = r.multiplicity
        pnum = Polynomial(self._coefs)
        px = Polynomial([-r.real, 1])
        pden = px**2 + r.imag**2

        pnum_t2 = -2 * k * pnum * px

        k_term_num = self._coefs[1] + pnum_t2 // pden
        kp_term_num = pnum_t2 % pden

        assert len(k_term_num) == 1
        assert len(kp_term_num) == 2

        return (
            RationalTermComplexPair(r, k_term_num.coef),
            RationalTermComplexPair(r.with_multiplicity(k + 1), kp_term_num.coef),
        )

    def integ(self) -> tuple[RationalIntegralGeneralTerm, ...]:
        """Compute the integral of the term.

        Returns:
            tuple[RationalIntegralGeneralTerm]: Terms of the integral
        """
        r = self._root
        # First, write numerator and denominator as functions of t = x - r.real
        x = Polynomial([r.real, 1.0])
        t = Polynomial([-r.real, 1.0])
        num = Polynomial(self._coefs)(x)

        k = r.multiplicity
        m = r.imag
        int_terms = []

        # Start with the linear term, if present
        if len(num) == 2 and num.coef[1] != 0:
            if k == 1:
                int_terms.append(RationalIntegralLogPairTerm(num.coef[1], r.value))
            else:
                r1 = r.with_multiplicity(k - 1)
                int_terms.append(
                    RationalTermComplexPair(
                        r1, Polynomial([num.coef[1] / (2 * (1 - k))])(t).coef
                    )
                )

        a = num.coef[0]

        while k > 0:
            if k == 1:
                # Split out logarithmic and arctangent term
                int_terms.append(RationalIntegralArctanTerm(a, r.value))
                break
            else:
                k -= 1
                int_terms.append(
                    RationalTermComplexPair(
                        r.with_multiplicity(k),
                        Polynomial([0.0, a / (2 * k * m**2)])(t).coef,
                    )
                )
                a *= (2 * k - 1) / (2 * m**2 * k)

        return tuple(int_terms)

    @property
    def denominator(self) -> Polynomial:
        """Return the denominator of the term."""
        r = self._root
        return (Polynomial([-r.real, 1]) ** 2 + r.imag**2) ** r.multiplicity


class RationalTerm(RationalTermBase):
    """A single term in a proper rational function,
    corresponding to a single root or complex
    conjugate pair of roots in the denominator.
    """

    def __new__(cls, root: PolynomialRoot, coefs: ArrayLike):
        if root.is_complex_pair:
            return RationalTermComplexPair(root, coefs)
        else:
            return RationalTermSingle(root, coefs)


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
