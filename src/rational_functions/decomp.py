"""Decomposition utilities for rational functions."""

import numpy as np
from numpy.polynomial import Polynomial
from .rtypes import PolynomialRoot
import typing

if typing.TYPE_CHECKING:
    from rational_functions.terms import RationalTerm


def catalogue_roots(
    p: Polynomial, atol: float = 1e-8, rtol: float = 1e-5, imtol: float = 1e-13
) -> list[PolynomialRoot]:
    """Extract the roots of a polynomial and group them
    into PolynomialRoot objects.

    Args:
        p (Polynomial): Input polynomial.
        atol (float): Absolute tolerance for root comparison.
        rtol (float): Relative tolerance for root comparison.
        imtol (float): Absolute tolerance under which imaginary parts are considered zero.

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
    roots = np.where(np.abs(np.imag(roots)) < imtol, np.real(roots), roots)

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
        r_val = np.mean(roots[r_map])
        if abs(r_val.imag) < imtol:
            r_val = float(r_val.real)
        else:
            r_val = complex(r_val)
        roots_mults.append((r_val, mult))

    # Group together complex conjugate pairs
    root_pairs_mults: list[tuple[complex, int]] = []
    root_single_mults: list[tuple[complex, int]] = []
    extracted[:] = False
    roots_vals = np.array(roots_mults)[:, 0]

    for i, (r, m) in enumerate(roots_mults):
        if extracted[i]:
            continue

        conj_i = None
        if np.iscomplex(r):
            # Look for a conjugate
            found_i = np.where(
                np.isclose(roots_vals[i + 1 :], r.conjugate(), atol=atol, rtol=rtol)
            )[0]
            if len(found_i) == 1:
                conj_i = int(found_i[0] + i + 1)

        if conj_i is not None:
            conj_m = roots_mults[conj_i][1]
            pair_m = min(m, conj_m)
            m -= pair_m
            conj_m -= pair_m

            root_pairs_mults.append((r, pair_m))
            if conj_m > 0:
                root_single_mults.append(
                    ((r + np.conj(roots_vals[conj_i])) / 2.0, conj_m)
                )
            extracted[conj_i] = True

        if m > 0:
            root_single_mults.append((r, m))
        extracted[i] = True

    root_objs = [PolynomialRoot(r, m, False) for r, m in root_single_mults]
    root_objs += [PolynomialRoot(r, m, True) for r, m in root_pairs_mults]

    return root_objs


def partial_frac_decomposition(
    numerator: Polynomial, denominator_roots: list[PolynomialRoot]
) -> list["RationalTerm"]:
    """Perform a partial fraction decomposition of a rational function.

    Args:
        numerator (Polynomial): Numerator of the rational function.
        denominator_roots (list[PolynomialRoot]): Roots of the denominator polynomial.

    Returns:
        list[tuple[PolynomialRoot, ArrayLike]]: List of partial fraction terms as root corresponding to the term
            (such that its monic polynomial is the denominator of the term) and coefficients of its numerator.
    """

    # Imported inside to avoid a circular import
    from rational_functions.terms import RationalTerm

    # Total degree of the denominator
    deg = sum([r.multiplicity * (1 + r.is_complex_pair) for r in denominator_roots])

    # We construct a linear system
    M = np.zeros((deg, deg))
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
            if r.is_complex_pair:
                # Column corresponding to linear term (only for complex pairs)
                M[1 : len(c) + 1, m_i] = c
                m_i += 1

    y = np.zeros(deg)
    y[: len(numerator.coef)] = numerator.coef

    # Solving gives us the corresponding coefficients of the partial fractions
    x = np.linalg.solve(M, y)

    m_i = 0
    terms: list[RationalTerm] = []
    # We collect the coefficients and build the terms
    for r in denominator_roots:
        for i in range(r.multiplicity):
            if r.is_complex_pair:
                coef = x[m_i : m_i + 2]
                m_i += 2
            else:
                coef = x[m_i : m_i + 1]
                m_i += 1
            terms.append(RationalTerm(r.with_multiplicity(i + 1), coef))

    return terms
