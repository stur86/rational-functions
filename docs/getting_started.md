# Getting started

## Installation

Simply install `rational-functions` from the PyPI index:

```bash
pip install rational-functions
```

## Usage

Import the `RationalFunction` class.

```py
from rational_functions import RationalFunction
```

The most intuitive way to create a rational function is from a ratio of polynomials. You can use Numpy polynomials or simple arrays:

```py
import numpy as np
from rational_functions import RationalFunction


P = np.polynomial.Polynomial([-1.0, 1.0])
Q = np.polynomial.Polynomial([1.0, -3.0, 2.0])

R = RationalFunction.from_fraction(P, Q)
# Alternatively:
R = RationalFunction.from_fraction([-1.0, 1.0], [1.0, -3.0, 2.0])

print(R)
```

While this is possible and fine when working with small degree polynomials, it quickly suffers from the exact same accuracy problems that causes us to avoid the ratio-of-polynomials representation in the first place, because the polynomials need to be decomposed to retrieve the partial fractions. In other words, ideally you only want to do this for functions of very low degree, and then build bigger functions only by summing or multiplying smaller ones.

The natural way to build a `RationalFunction` object is to specify its partial fraction terms directly:

```py
from rational_functions import RationalFunction, RationalTerm

terms = [RationalTerm(2.0, 1.0, 1), RationalTerm(-1.0j, 2.0, 2)]

R = RationalFunction(terms, [1, -2, 1])
```
