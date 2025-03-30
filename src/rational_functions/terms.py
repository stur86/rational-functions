"""Individual terms in a proper rational function."""

from abc import ABC, abstractmethod
from typing import Union
import numpy as np
from numpy.typing import ArrayLike
from numpy.polynomial import Polynomial
from .roots import PolynomialRoot
from .decomp import partial_frac_decomposition

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
        
    def __neg__(self) -> "RationalTermBase":
        """Negate the term."""
        return self.__class__(self._root, -self._coefs)

    @property
    def root(self) -> PolynomialRoot:
        """Return the root of the term."""
        return self._root

    @property
    def numerator(self) -> Polynomial:
        """Return the numerator of the term."""
        return Polynomial(self._coefs).trim()

    @property
    def denominator(self) -> Polynomial:
        """Return the denominator of the term."""
        return self._root.monic_polynomial()
    
    @classmethod
    def product(cls, term1: "RationalTermBase", term2: "RationalTermBase") -> list["RationalTermBase"]:
        """Compute and decompose the product of two rational terms. This method
        is not implemented as an overloaded operator __mul__ since the output is
        not a RationalTerm but a list of RationalTerms.
        
        Args:
            term1 (RationalTermBase): First term
            term2 (RationalTermBase | Polynomial): Second term
        Returns:
            list[RationalTermBase]: List of terms in the product
        """
        
        num1 = Polynomial(term1._coefs)
        num2 = Polynomial(term2._coefs)
        
        r1 = term1._root
        r2 = term2._root
        
        if r1.is_equivalent(r2):
            roots = [r1.with_multiplicity(r1.multiplicity + r2.multiplicity)]
        else:
            roots = [r1, r2]

        return partial_frac_decomposition(num1 * num2, roots)
    
    @classmethod
    def product_w_polynomial(cls, term: "RationalTermBase", poly: Polynomial) -> tuple[list["RationalTermBase"], Polynomial]:
        """Compute and decompose the product of a rational term and a polynomial.
        This method is not implemented as an overloaded operator __mul__ since the output is        
        not a RationalTerm but a list of RationalTerms and a Polynomial.
        
        Args:
            term (RationalTermBase): Rational term
            poly (Polynomial): Polynomial to multiply with
            
        Returns:
            tuple[list[RationalTermBase], Polynomial]: List of terms in the product and the remaining polynomial
        """
        
        num = poly*Polynomial(term._coefs)
        r = term._root
        
        den = r.monic_polynomial()
        poly_out = num // den
        poly_rem = num % den
        
        terms = partial_frac_decomposition(poly_rem, [r])

        return terms, poly_out
        
        

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
        
    def __str__(self) -> str:
        """Print the term."""
        num = Polynomial(self._coefs)
        den = Polynomial([-self._root.value, 1.0])
        mul_str = ""
        if self._root.multiplicity > 1:
            mul_str = str(self._root.multiplicity).translate(Polynomial._superscript_mapping)

        return f"{num} / ({den}){mul_str}"


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

    def __str__(self):
        num = Polynomial(self._coefs)
        den = Polynomial([-self._root.real, 1.0])**2 + self._root.imag**2
        mul_str = ""
        if self._root.multiplicity > 1:
            mul_str = str(self._root.multiplicity).translate(Polynomial._superscript_mapping)

        return f"({num}) / ({den}){mul_str}"


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
