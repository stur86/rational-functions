# Getting started

## Installation

Simply install `rational-functions` from the PyPI index:

```bash
pip install rational-functions
```

## Usage

### Creating rational functions

Import the `RationalFunction` class.

```py
from rational_functions import RationalFunction
```

The most intuitive way to create a rational function is from a ratio of polynomials. You can use Numpy polynomials or simple arrays:

```py
import numpy as np
from rational_functions import RationalFunction


P = np.polynomial.Polynomial([-1.0, 1.0])
Q = np.polynomial.Polynomial([0.5, -2.0, 1.0])

R = RationalFunction.from_fraction(P, Q)
# Alternatively:
R = RationalFunction.from_fraction([-1.0, 1.0], [0.5, -2.0, 1.0])

print(R)
```
```
((-1+0j) + (1+0j)·x)/((0.5+0j) - (2-0j)·x + (1+0j)·x²)
```

While this is possible and fine when working with small degree polynomials, it quickly suffers from the exact same accuracy problems that causes us to avoid the ratio-of-polynomials representation in the first place, because the polynomials need to be decomposed to retrieve the partial fractions. In other words, ideally you only want to do this for functions of very low degree, and then build bigger functions only by summing or multiplying smaller ones.

The natural way to build a `RationalFunction` object is to specify its partial fraction terms directly:

```py
from rational_functions import RationalFunction, RationalTerm

terms = [RationalTerm(2.0, 1.0, 1), RationalTerm(-1.0j, 2.0, 2)]

R = RationalFunction(terms, [1, -2, 1])

print(R)
```
```
1.0 - 2.0·x + 1.0·x² + ((-5+0j) + (2+2j)·x + (1+0j)·x²)/((2+0j) - (1+4j)·x - (2-2j)·x² + (1+0j)·x³)
```

The arguments to the `RationalTerm` are the value of the pole, the coefficient at the numerator, and the order.

For more constructors, see [the class's API][rational_functions.ratfunc.RationalFunction]

### Using rational functions

`RationalFunction` objects can be used as callables to compute their values, same as Numpy's `Polynomial` class:

```py

terms = [RationalTerm(2.0, 1.0, 1), RationalTerm(-1.0j, 2.0, 2)]
R = RationalFunction(terms, [1, -2, 1])

x = np.linspace(-1, 1, 100)
y = R(x)
```

They also support the following operators:

* `+`, addition
* `-`, subtraction
* `*`, multiplication
* `/`, true division

with other rational functions, polynomials, or scalars.

```py
R1 = RationalFunction([RationalTerm(1.0, 1.0, 1)])
R2 = RationalFunction([RationalTerm(2.0, -0.5, 1)], [1, -2, 1])

Rsum = R1+R2
Rdiff = R1-R2
Rprod = R1*R2
Rdiv = R1/R2
```

!!! warning
    Division by another rational function or a polynomial requires recomputing the partial fraction decomposition.
    This is vulnerable to numerical instabilities due to limits of NumPy's linear system solving and root finding
    algorithms.

They support the power operator `**` too, but only with scalar integers.

```py
R = RationalFunction([RationalTerm(2.0, -0.5, 1)], [1, -2, 1])

Rsquare = R**2
Rcube = R**3
Rinv = R**(-1)
```

### Derivatives and integrals

Like NumPy's `Polynomial` class, `RationalFunction` implements `.deriv()` and `.integ()` methods. While `.deriv()` works exactly as in the `Polynomial` class, `.integ()` is more problematic because there is no guarantee that the integral of a rational function is, itself, a rational function.

Instead, generally speaking `.integ()` can return either a `RationalFunction` or a `RationalFunctionIntegral` object. The latter can still be evaluated by calling it, but it does not support further integration, differentiation, or other operations. When using `.integ()`, you should be ready to check for the return type, unless you use the dedicated argument to force the return type to be a `RationalFunctionIntegral`. See also [the method's documentation][rational_functions.ratfunc.RationalFunction.integ].
