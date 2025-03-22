import numpy as np
from numpy.polynomial import Polynomial
from numpy.typing import ArrayLike
from dataclasses import dataclass

PolyDefType = ArrayLike | Polynomial


@dataclass
class PolynomialRoot:
    value: complex
    multiplicity: int = 1
    is_complex_pair: bool = False

    @property
    def is_real(self) -> bool:
        return np.isreal(self.value)

    @property
    def real(self) -> float:
        return np.real(self.value)

    @property
    def imag(self) -> float:
        return np.imag(self.value)

    def with_multiplicity(self, multiplicity: int) -> "PolynomialRoot":
        return PolynomialRoot(
            value=self.value,
            multiplicity=multiplicity,
            is_complex_pair=self.is_complex_pair,
        )


def coerce_to_polynomial(arg: PolyDefType) -> Polynomial:
    if isinstance(arg, Polynomial):
        return arg.copy()
    else:
        return Polynomial(arg)
