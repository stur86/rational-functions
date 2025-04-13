import numpy as np
from numpy.polynomial import Polynomial
from rational_functions.utils import as_polynomial, group_by_closeness


def test_as_polynomial():
    p = Polynomial([1, 2, 3])
    c = [4, 5, 6]

    p_p = as_polynomial(p)
    c_p = as_polynomial(c)

    assert isinstance(p_p, Polynomial)
    assert isinstance(c_p, Polynomial)

    assert p_p == p
    assert (c_p.coef == c).all()


def test_group_by_closeness():
    # Test a simple group
    data = [1, 3.00000002, 3, 2, 1.00000001]
    grouped = group_by_closeness(data, atol=1e-8, rtol=1e-5)
    assert len(grouped) == 3

    assert grouped == {
        1.000000005: [1, 1.00000001],
        2.0: [2],
        3.00000001: [3, 3.00000002],
    }

    # Now with tuples and a key
    data = [
        ("e", 3.00000002),
        ("a", 1),
        ("b", 1.00000001),
        ("c", 2),
        ("d", 3),
        ("f", 3),
        ("g", 3.00000002),
    ]
    grouped = group_by_closeness(data, key=lambda x: x[1], atol=1e-8, rtol=1e-5)

    assert len(grouped) == 3
    assert grouped == {
        1.000000005: [("a", 1), ("b", 1.00000001)],
        2.0: [("c", 2)],
        3.00000001: [("d", 3), ("f", 3), ("e", 3.00000002), ("g", 3.00000002)],
    }

    # Test different tolerances
    data = [1, 1.0001, 2, 3, 3.0001]

    strict_groups = group_by_closeness(data, atol=0.00001)
    lenient_groups = group_by_closeness(data, atol=0.001)

    assert len(strict_groups) == 5
    assert len(lenient_groups) == 3

    assert np.allclose(list(strict_groups.keys()), data)
    assert np.allclose(list(sorted(lenient_groups.keys())), [1.00005, 2.0, 3.00005])
