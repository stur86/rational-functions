# %%
import numpy as np
from numpy.polynomial import Polynomial
from rational_functions import RationalFunction, RationalTerm
import matplotlib.pyplot as plt

# %%
numerator = Polynomial([1.0, 3.0, 1.0, -1.0])
denominator = Polynomial.fromroots([2.0, 4.0])

# %%

ratfunc = RationalFunction.from_fraction(numerator, denominator)

x = np.linspace(-1, 1, 100)
y1 = numerator(x) / denominator(x)
y2 = ratfunc(x)

fig, ax = plt.subplots()

ax.plot(x, y1, label="Fraction of polynomials", c="k", lw=1.5)
ax.plot(x, y2, label="Rational function", c="r", lw=0.8, ls="--")
ax.set_title("Rational function from fraction of polynomials")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
# %%

# Example of loss of numerical precision in the polynomial division
# form

deg = 70
rng = np.random.default_rng(0)

coefs_hi = np.random.normal(size=deg)
# Poles span multiple orders of magnitude
roots_hi = 10 ** (3 * rng.random(deg))

ratfunc_hi = RationalFunction(
    [RationalTerm(r, c) for (c, r) in zip(coefs_hi, roots_hi)]
)

numerator_hi = ratfunc_hi.numerator
denominator_hi = ratfunc_hi.denominator
# %%

fig, ax = plt.subplots()
x_long = np.logspace(0, 1, 400)

y_hi = numerator_hi(x_long) / denominator_hi(x_long)
y_r_hi = ratfunc_hi(x_long)

l1 = ax.plot(x_long, y_hi, label="np.Polynomial/np.Polynomial", c="k", lw=0.5)
l2 = ax.plot(x_long, y_r_hi, label="RationalFunction", c="b", lw=1, ls="--")
ax.set_title(f"Rational function from fraction of polynomials (deg={deg})")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_xscale("log")
ax.set_ylim(-100, 100)
ax.set_xlim(x_long[0], x_long[-1])

# Plot the poles of the rational function
for root in roots_hi:
    if root > x_long[-1]:
        continue
    last_pl = ax.axvline(root, color="r", lw=1, label="Poles")

ax.legend(handles=[*l1, *l2, last_pl])