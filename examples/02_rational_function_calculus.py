# %%
import numpy as np
from numpy.polynomial import Polynomial
from rational_functions import RationalFunction
import matplotlib.pyplot as plt

# %%
numerator = Polynomial([1.0, 3.0, 1.0, -1.0])
denominator = Polynomial.fromroots([2.0, 4.0, -2.0, -3.0, 1.0j, -1.0j])

# %%

ratfunc = RationalFunction.from_fraction(numerator, denominator)
x = np.linspace(-1, 1, 100)
# %%
fig, ax = plt.subplots()
ax.plot(x, ratfunc(x), label="Rational function", c="r", lw=0.8, ls="-")

# %%
# Derivative
ratfunc_derivative = ratfunc.deriv()
ratfunc_num_gradient = np.gradient(ratfunc(x), x)

fig, ax = plt.subplots()
ax.plot(x, ratfunc_derivative(x), label="RationalFunction.diff", c="b", lw=0.8, ls="-")
ax.plot(x, ratfunc_num_gradient, label="np.gradient", c="g", lw=0.8, ls="--")
ax.legend()
ax.set_title("Derivative of the rational function")
ax.set_xlabel("x")
ax.set_ylabel("y")

# %%
# Second derivative

ratfunc_second_derivative = ratfunc.deriv(2)
ratfunc_num_second_gradient = np.gradient(ratfunc_num_gradient, x)

fig, ax = plt.subplots()
ax.plot(
    x,
    ratfunc_second_derivative(x),
    label="RationalFunction.diff",
    c="b",
    lw=0.8,
    ls="-",
)
ax.plot(x, ratfunc_num_second_gradient, label="np.gradient", c="g", lw=0.8, ls="--")
ax.legend()
ax.set_title("Second derivative of the rational function")
ax.set_xlabel("x")
ax.set_ylabel("y")

# %%
# Integral

ratfunc_integral = ratfunc.integ()
ratfunc_num_integral = np.cumsum(ratfunc(x)) * (x[1] - x[0])

fig, ax = plt.subplots()
ax.plot(
    x,
    ratfunc_integral(x) - ratfunc_integral(x)[0],
    label="RationalFunction.integ",
    c="b",
    lw=0.8,
    ls="-",
)
ax.plot(x, ratfunc_num_integral, label="np.cumsum", c="g", lw=0.8, ls="--")
ax.legend()
ax.set_title("Integral of the rational function")
ax.set_xlabel("x")
ax.set_ylabel("y")
