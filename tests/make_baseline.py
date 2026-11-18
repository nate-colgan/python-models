"""Regenerate tests/baseline_local_radiation.npz.

Only run this when a change to the local radiation path is INTENDED to move
results. Record why in the commit message.
"""
import numpy as np
from regression_cases import run_all, BASELINE

if __name__ == '__main__':
    data = run_all()
    np.savez_compressed(BASELINE, **data)
    print(f"wrote {len(data)} arrays to {BASELINE}")
