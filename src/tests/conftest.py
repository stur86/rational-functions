import pytest
import numpy as np
from itertools import groupby
from rational_functions.terms import RationalIntegralGeneralTerm


class ComparisonMethods:
    @staticmethod
    def compare_ratint_terms(
        tgt: RationalIntegralGeneralTerm,
        ref: RationalIntegralGeneralTerm,
        atol: float = 1e-8,
        rtol: float = 1e-5,
    ) -> bool:
        if not isinstance(tgt, type(ref)):
            return False

        d1 = tgt.__dict__
        d2 = ref.__dict__

        if set(d1.keys()) != set(d2.keys()):
            return False

        for k1, v1 in d1.items():
            v2 = d2[k1]
            if not np.isclose(v1, v2, atol=atol, rtol=rtol):
                return False

        return True

    @staticmethod
    def compare_ratint_seqs(
        tgt: list[RationalIntegralGeneralTerm],
        ref: list[RationalIntegralGeneralTerm],
        atol: float = 1e-8,
        rtol: float = 1e-5,
    ) -> bool:
        if len(tgt) != len(ref):
            return False

        # Group them by type
        tgt_groups = groupby(
            sorted(tgt, key=lambda x: type(x).__name__), key=lambda x: type(x).__name__
        )
        ref_groups = groupby(
            sorted(ref, key=lambda x: type(x).__name__), key=lambda x: type(x).__name__
        )

        for tgt_name, tgt_vals in tgt_groups:
            ref_name, ref_vals = next(ref_groups, (None, None))
            if tgt_name != ref_name:
                return False

            # Compare the lists of values
            tgt_vals = list(tgt_vals)
            ref_vals = list(ref_vals)

            if len(tgt_vals) != len(ref_vals):
                return False

            passed = [False for _ in range(len(tgt_vals))]
            used = [False for _ in range(len(ref_vals))]
            for i, t_val in enumerate(tgt_vals):
                for j, r_val in enumerate(ref_vals):
                    if used[j]:
                        continue
                    if ComparisonMethods.compare_ratint_terms(
                        t_val, r_val, atol=atol, rtol=rtol
                    ):
                        passed[i] = True
                        used[j] = True
                        break

            if not all(passed):
                return False

        return True


@pytest.fixture
def comparison_methods():
    """Fixture to provide the ComparisonMethods class."""
    return ComparisonMethods
