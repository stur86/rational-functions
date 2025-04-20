"""Utility functions for the package."""

import numpy as np
from typing import Iterable, Callable, TypeVar
from numpy.typing import ArrayLike

PolynomialDef = np.polynomial.Polynomial | ArrayLike


def as_polynomial(p: PolynomialDef) -> np.polynomial.Polynomial:
    """Coerce input to a numpy polynomial.

    Args:
        p: Input polynomial, or series of coefficients.

    Returns:
        np.polynomial.Polynomial: Coerced polynomial.
    """

    if isinstance(p, np.polynomial.Polynomial):
        return p
    return np.polynomial.Polynomial(p)


T = TypeVar("T")


def group_by_closeness(
    data: Iterable[T],
    key: Callable[[T], complex] = lambda x: x,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> dict[complex, list[T]]:
    """Group elements by closeness of their keys. Keys must be numeric (float or complex).

    Args:
        data: Iterable of elements to group.
        key: Function to extract the key from each element.
        atol: Absolute tolerance for closeness.
        rtol: Relative tolerance for closeness.

    Returns:
        dict: Dictionary with keys as unique values and values as lists of elements.
    """

    # Start by extracting keys from the data
    key_arr = np.array([key(d) for d in data])
    sort_idx = np.argsort(key_arr, stable=True)
    key_sorted = key_arr[sort_idx]

    # Cluster by similarity
    edges = (
        np.where(~np.isclose(key_sorted[:-1], key_sorted[1:], atol=atol, rtol=rtol))[0]
        + 1
    )
    # Groups
    groups = np.split(sort_idx, edges)

    # Create a dictionary to hold the grouped elements
    grouped_dict: dict[complex, list[T]] = {}
    for group in groups:
        # Use the mean of the group as the key
        key_value = np.mean(key_arr[group])
        grouped_dict[key_value] = list([data[i] for i in group])

    return grouped_dict


def round_to_zero(x: ArrayLike, tol: float) -> ArrayLike:
    x = np.where(np.abs(x.real) < tol, 0.0, x.real) + np.where(
        np.abs(x.imag) < tol, 0.0, x.imag * 1j
    )
    return x
