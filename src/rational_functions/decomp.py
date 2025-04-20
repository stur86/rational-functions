"""Decomposition utilities for rational functions."""

import numpy as np
from numpy.polynomial import Polynomial
from numpy.typing import NDArray
from .roots import PolynomialRoot
from .utils import round_to_zero
import typing

if typing.TYPE_CHECKING:
    from rational_functions.terms import RationalTerm


def catalogue_roots(
    p: Polynomial, atol: float = 1e-8, rtol: float = 1e-5, ztol: float = 1e-13
) -> list[PolynomialRoot]:
    """Extract the roots of a polynomial and group them
    into PolynomialRoot objects.

    Args:
        p (Polynomial): Input polynomial.
        atol (float): Absolute tolerance for root comparison. Default is 1e-8.
        rtol (float): Relative tolerance for root comparison. Default is 1e-5.
        ztol (float): Absolute tolerance under which real or
            imaginary parts are considered zero. Default is 1e-13.

    Returns:
        list[PolynomialRoot]: List of PolynomialRoot objects.

    Warning:
        The process of root-finding is very sensitive to numerical noise; this function uses
        tolerances to group together roots that are close to each other. It's important
        to pay attention to the absolute tolerances used, as if the roots are expected
        to be close enough to zero, they may be grouped together.
        Similarly, roots that are supposed to have high multiplicity might be split
        into multiple roots if the tolerances are too low, and real roots might display
        a small imaginary part due to numerical errors. See
        [the NumPy documentation](https://numpy.org/doc/stable/reference/generated/numpy.isclose.html)
        for more information.
    """

    roots = p.roots()
    # Filter out complex roots with small imaginary parts
    roots = np.where(np.abs(np.imag(roots)) < ztol, np.real(roots), roots)

    # Find unique roots
    roots_mults: list[tuple[complex, int]] = []

    # Group roots by multiplicity
    extracted = np.zeros_like(roots, dtype=bool)
    for i, r in enumerate(roots):
        if extracted[i]:
            continue
        r_map = np.isclose(roots, r, atol=atol, rtol=rtol)
        mult = int(np.sum(r_map))
        # Remove all occurrences of the root
        extracted[r_map] = True
        r_val = np.mean(roots[r_map]).astype(np.complex128)
        # Round to zero if close enough
        r_val = round_to_zero(r_val, ztol)

        roots_mults.append((r_val, mult))

    return list([PolynomialRoot(r, m) for r, m in roots_mults])


def partial_frac_decomposition(
    num_coef: NDArray[np.number],
    denominator_roots: list[PolynomialRoot],
    ztol: float = 1e-13,
) -> list["RationalTerm"]:
    """Perform a partial fraction decomposition of a rational function.

    Args:
        num_coef (NDArray[np.number]): Coefficients of the numerator of the rational function.
        denominator_roots (list[PolynomialRoot]): Roots of the denominator polynomial.
        ztol (float): Absolute tolerance under which real or
            imaginary parts of the solution are considered zero. Default is 1e-13.

    Returns:
        list[tuple[PolynomialRoot, ArrayLike]]: List of partial fraction terms as root corresponding to the term
            (such that its monic polynomial is the denominator of the term) and coefficients of its numerator.
    """

    # Imported inside to avoid a circular import
    from rational_functions.terms import RationalTerm

    # Total degree of the denominator
    deg = sum([r.multiplicity for r in denominator_roots])

    # We construct a linear system
    M = np.zeros((deg, deg), dtype=np.complex128)
    m_i = 0

    for i, r in enumerate(denominator_roots):
        # Build the polynomial of all other roots
        residual_p = Polynomial([1.0])
        for j, r2 in enumerate(denominator_roots):
            if i == j:
                continue
            residual_p *= r2.monic_polynomial()

        for k in range(1, r.multiplicity + 1):
            residual_root_p = r.with_multiplicity(r.multiplicity - k).monic_polynomial()
            c = (residual_p * residual_root_p).coef
            # Column corresponding to constant term
            M[: len(c), m_i] = c
            m_i += 1

    y = np.zeros(deg, dtype=np.complex128)
    # We build the right-hand side of the system
    y[: len(num_coef)] = num_coef

    # Solving gives us the corresponding coefficients of the partial fractions
    x = np.linalg.solve(M, y)
    x = round_to_zero(x, ztol)

    m_i = 0
    terms: list[RationalTerm] = []
    # We collect the coefficients and build the terms
    for r in denominator_roots:
        for i in range(r.multiplicity):
            coef = x[m_i]
            m_i += 1
            terms.append(RationalTerm(r.value, coef, i + 1))

    return terms
