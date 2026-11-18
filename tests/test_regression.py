"""Regression test: current local-radiation path must reproduce the baseline.

Run:  python -m pytest tests -q      (or:  python tests/test_regression.py)

Tolerance: rtol 1e-6. The liner/shade root-solves use brentq(xtol=1 K), so a
refactor that changes the order of floating-point operations can shift a
converged liner temperature within that tolerance. If a pure refactor fails
only at the ~1e-4 level, check whether that is solver tolerance before
loosening RTOL, and say so in the commit.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from regression_cases import run_all, BASELINE  # noqa: E402

RTOL, ATOL = 1e-6, 1e-9


def test_local_radiation_regression():
    ref = np.load(BASELINE)
    new = run_all()
    assert set(ref.files) == set(new), "case/key set changed"
    bad = []
    for k in ref.files:
        a, b = ref[k], new[k]
        if a.shape != b.shape:
            bad.append(f"{k}: shape {a.shape} -> {b.shape}")
        elif not np.allclose(a, b, rtol=RTOL, atol=ATOL, equal_nan=True):
            rel = np.nanmax(np.abs(b - a) / np.maximum(np.abs(a), ATOL))
            bad.append(f"{k}: max rel diff {rel:.3e}")
    assert not bad, "regression mismatch:\n  " + "\n  ".join(bad)


if __name__ == '__main__':
    test_local_radiation_regression()
    print("regression OK")
