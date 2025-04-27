import numpy as np
from numpy.polynomial import Polynomial
from numpy.typing import NDArray


def fit_ratfun_leastsq(
    x: NDArray[np.number], y: NDArray[np.number], m: int, n: int
) -> tuple[Polynomial, Polynomial]:
    r"""Fit a rational function to data using least squares.
    The fit is found by minimizing the sum of squares of the residuals:

    $$
        (N\alpha- YD\beta)^2
    $$

    where $\alpha$ and $\beta$ are the coefficients of the numerator and denominator polynomials, respectively,
    $N$ is the Vandermonde matrix of the numerator polynomial, $D$ is the Vandermonde matrix of the denominator polynomial,
    and $Y$ is a diagonal matrix with the y-coordinates of the data points on the diagonal.

    Args:
        x (NDArray[np.number]): x coordinates of data points
        y (NDArray[np.number]): y coordinates of data points
        m (int): Degree of the numerator polynomial
        n (int): Degree of the denominator polynomial

    Returns:
        tuple[Polynomial, Polynomial]: Numerator and denominator polynomials
    """

    if m < 0 or n < 0:
        raise ValueError(
            "Degrees of numerator and denominator polynomials must be non-negative"
        )

    if len(x) != len(y):
        raise ValueError("x and y must have the same length")

    # Rescale x to (-1, 1)
    xl, xr = np.min(x), np.max(x)
    c = (xl + xr) / 2
    d = (xr - xl) / 2
    x = (x - c) / d

    # Construct Vandermonde matrices
    N = np.vander(x, m + 1, increasing=True)
    D = np.vander(x, n + 1, increasing=True)
    YD = y[:, None] * D

    # SVD of N
    U, s, Vh = np.linalg.svd(N, full_matrices=False)

    # \alpha = AB\beta
    AB = Vh.T @ (1 / s[:, None] * U.T) @ YD

    # \beta solution
    M = YD.T @ U @ U.T @ YD - YD.T @ YD
    # We constrain beta to be monic
    beta = np.linalg.lstsq(M[:, :-1], -M[:, -1], rcond=None)[0]
    beta = np.concatenate((beta, [1]))  # Add the last coefficient of beta
    # \alpha solution
    alpha = AB @ beta

    num = Polynomial(alpha, domain=(xl, xr), window=(-1, 1))
    den = Polynomial(beta, domain=(xl, xr), window=(-1, 1))

    return num, den


def smooth_data_tv(
    x: NDArray[np.number], y: NDArray[np.number], lmb: float
) -> NDArray[np.number]:
    r"""Smooth data using a total variation denoising method.

    Args:
        x (NDArray[np.number]): x coordinates of data points
        y (NDArray[np.number]): y coordinates of data points
        lmb (float): Regularization parameter

    Returns:
        NDArray[np.number]: Smoothed y coordinates
    """

    N = len(x)
    # Sort x and y
    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]

    # Average step size
    h = (x[-1] - x[0]) / (N - 1)

    # Build the finite difference matrix
    D2 = np.zeros((N - 2, N))
    for i in range(1, N - 1):
        # x points
        x_loc = x[i - 1 : i + 2] - x[i]
        # Vandermonde matrix
        V = np.vander(x_loc, 3, increasing=True)
        # Finite difference vector
        D2[i - 1, i - 1 : i + 2] = np.linalg.solve(V.T, [0, 0, 2]) * h**2

    # Solve the linear system
    return np.linalg.solve(np.eye(N) + lmb * D2.T @ D2, y)
