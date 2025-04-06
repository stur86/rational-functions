# %%
import numpy as np
from numpy.polynomial import Polynomial
from rational_functions import RationalFunction
import matplotlib.pyplot as plt
# %%

P = Polynomial.fromroots([2.0, 4.0])
Q = Polynomial.fromroots([1.0+0.4j, 1.0-0.4j, -1.5, 2.2])

R = RationalFunction.from_fraction(P, Q)
# %%
x = np.linspace(-1, 1, 1000)

y = np.real(R(x))

fig, ax = plt.subplots()
ax.plot(x, y, label="Rational function", c="r", lw=0.8, ls="-")
ax.set_title("Rational function from fraction of polynomials")
ax.set_xlabel("x")
ax.set_ylabel("y")
# %%

# Reciprocal
R_inv = R.reciprocal()
y_inv = np.real(R_inv(x))

fig, ax = plt.subplots()
ax.plot(x, R.reciprocal()(x), label="Reciprocal", c="b", lw=0.8, ls="--")
ax.legend()
ax.set_title("Reciprocal of the rational function")
ax.set_xlabel("x")
ax.set_ylabel("y")
# %%
