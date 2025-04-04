import numpy as np
import sys
from functools import cached_property
from types import FrameType
from typing import Union
from numpy.typing import ArrayLike
from numpy.polynomial import Polynomial
from numpy.polynomial.polyutils import _add as np_poly_add
from rational_functions.terms import RationalTerm
from rational_functions.decomp import catalogue_roots, partial_frac_decomposition
from rational_functions.roots import PolynomialRoot
from rational_functions.lcm import RootLCM

_RFuncOpCompatibleType = Union["RationalFunction", Polynomial, np.number]

def _get_callercode_safe(depth: int) -> FrameType | None:
    try:
        return sys._getframe(depth).f_code
    except ValueError:
        # If the stack frame is not available, return None
        return None
    

class RationalFunction:
    """Rational function represented by
    sum of terms with a single pole for the proper part,
    plus a residual polynomial for the improper part.
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

        self._terms = list(terms)
        self._lcm = RootLCM([term.root for term in self._terms])
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
            num += term.coef*self._lcm.residual(term.root.value, term.root.multiplicity)
        
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
        sum_terms = list(self._terms)
        other_terms = []
        other_poly: Polynomial
        
        is_other_ratfunc = isinstance(other, RationalFunction)

        if not is_other_ratfunc:
            # This check is necessary to avoid additions Polynomial+RationalFunction
            # being delegated to the Polynomial.__add__ method,
            # which returns a Polynomial object with RationalFunction coefficients.
            for i in range(1, 4):
                if _get_callercode_safe(i) == np_poly_add.__code__:                
                    # This means it's being called from inside a Polynomial.__add__ method
                    # which is not what we want
                    raise ValueError("Can not add a polynomial to a rational function.")

        if is_other_ratfunc:
            other_terms = list(other._terms)
            other_poly = other._poly
        elif isinstance(other, Polynomial):
            other_poly = other
        elif np.isscalar(other):
            # If other is a scalar, convert it to a polynomial
            other_poly = Polynomial([other])
        else:
            raise TypeError("Unsupported type for addition: {}".format(type(other)))

        for term in other_terms:
            was_added = False
            for i, existing_term in enumerate(sum_terms):
                if existing_term.root == term.root:
                    sum_terms[i] = RationalTerm(term.root, existing_term._coef+term._coef)
                    was_added = True
                    break
            if not was_added:
                sum_terms.append(term)
        
        # Remove terms with zero coefficients
        sum_terms = [term for term in sum_terms if term._coef != 0.0]

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
    
    def __sub__(
        self, other: _RFuncOpCompatibleType
    ) -> "RationalFunction":
        """Subtract two rational functions.

        Args:
            other (RationalFunction): Other rational function to subtract.

        Returns:
            RationalFunction: Resulting rational function.
        """
        return self + (-other)
    
    def __rsub__(
        self, other: _RFuncOpCompatibleType
    ) -> "RationalFunction":
        """Right subtract operator for RationalFunction.

        Args:
            other (RationalFunction): Other rational function to subtract.

        Returns:
            RationalFunction: Resulting rational function.
        """
        
        return (- self) + other
    
    def __mul__(self, other: _RFuncOpCompatibleType) -> "RationalFunction":
        """Multiply two rational functions.

        Args:
            other (RationalFunction): Other rational function to multiply.

        Returns:
            RationalFunction: Resulting rational function.
        """
                        
        return None
        


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
            numerator (Polynomial): Numerator polynomial.
            denominator (Polynomial): Denominator polynomial.
            atol (float, optional): Absolute tolerance for root equivalence. Defaults to 0.
            rtol (float, optional): Relative tolerance for root equivalence. Defaults to 0.
            imtol (float, optional): Tolerance for imaginary part of roots to be considered
                zero. Defaults to 0.

        Returns:
            RationalFunction: Rational function object.
        """

        # Polynomial part
        poly_quot = numerator // denominator
        # Residual numerator
        poly_rem = numerator % denominator
        # Make denominator monic
        c = denominator.coef[-1]
        denominator /= c
        numerator /= c

        # Find roots
        if denominator.degree() == 0:
            # No roots, return polynomial part
            return cls([], poly_quot)
        den_roots = catalogue_roots(denominator, atol=atol, rtol=rtol, imtol=imtol)
        rterms = partial_frac_decomposition(
            poly_rem,
            den_roots,
        )

        return cls(rterms, poly_quot)
