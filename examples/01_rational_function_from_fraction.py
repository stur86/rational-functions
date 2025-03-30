# %%
import numpy as np
from numpy.polynomial import Polynomial
from rational_functions import RationalFunction
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
