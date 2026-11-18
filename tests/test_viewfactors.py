"""View-factor tests (brief §3.4 and §6.5). Accuracy target 2% (speed prioritised).

Geometry: brief defaults (L_ps 3.048 m, H_block 6.096 m) with YAML radii.
Run:  python -m pytest tests/test_viewfactors.py -q
"""
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from thermal.viewfactors import (cavity_geometry, cavity_view_factors,  # noqa: E402
                                 coax_F_outer_to_inner, coax_F_outer_self)

R_PS = 2 * 0.26 / np.pi
R_IN = 5 * R_PS
R_L = 1 / np.pi
R_SH = (1 + np.sqrt(2) / 2) * R_L + 1e-4
G_LOW = cavity_geometry(R_IN, R_SH, R_L, 3.048, 6.096, 0.005, 0.025, n_ax=12)   # 6 lower + 6 active
G_FULL = cavity_geometry(R_IN, R_SH, R_L, 3.048, 3.048, 0.005, 0.025, n_ax=6)   # H == L
N_MC = 65_536          # default ray count (Sobol); accuracy target 2%
TOL = 0.02


def _rows_and_recip(enc, tol_row=1e-9, tol_rec=1e-9):
    F, A = enc.F, enc.A
    assert np.all(F >= -1e-12)
    assert np.max(np.abs(F.sum(1) - 1)) < tol_row, np.max(np.abs(F.sum(1) - 1))
    X = A[:, None] * F
    assert np.max(np.abs(X - X.T)) / X.max() < tol_rec


def test_analytic_limits():
    # long cylinders: F_oi -> r1/r2, F_oo -> 1 - r1/r2
    assert abs(coax_F_outer_to_inner(R_SH, R_IN, 1e4) - R_SH / R_IN) < 1e-4
    assert abs(coax_F_outer_self(R_SH, R_IN, 1e4) - (1 - R_SH / R_IN)) < 1e-4
    assert coax_F_outer_to_inner(R_SH, R_IN, 0.0) == 0.0


def test_outer_analytic_rows_reciprocity_and_vs_mc():
    an = cavity_view_factors(G_LOW, shaded=True, method='analytic')['outer']
    _rows_and_recip(an)
    mc = cavity_view_factors(G_LOW, shaded=True, method='montecarlo', n_rays=N_MC)['outer']
    assert an.names == mc.names
    big = an.F > 0.05
    rel = np.abs(mc.F - an.F)[big] / an.F[big]
    assert rel.max() < TOL, f"analytic vs MC rel diff {rel.max():.3%}"
    assert np.abs(mc.F - an.F).max() < 0.003


def test_inner_lower_zone_mc_rows_reciprocity_convergence():
    ref = cavity_view_factors(G_LOW, shaded=True, n_rays=2 ** 19, seed=7)['inner']
    errs = []
    for n in (N_MC // 16, N_MC // 4, N_MC):
        e = cavity_view_factors(G_LOW, shaded=True, n_rays=n, seed=1)['inner']
        assert e.method == 'montecarlo'
        _rows_and_recip(e, tol_row=1e-9, tol_rec=1e-9)
        big = ref.F > 0.05
        errs.append((np.abs(e.F - ref.F)[big] / ref.F[big]).max())
    for nm in ('R', 'E', 'floor_in', 'ceil_in'):
        assert nm in ref.idx
    # ray-count convergence vs a 2**19-ray reference
    assert errs[2] < errs[0], errs
    assert errs[2] < TOL, errs


def test_inner_full_height_analytic_matches_mc():
    an = cavity_view_factors(G_FULL, shaded=True)['inner']
    assert an.method == 'analytic'
    _rows_and_recip(an)
    mc = cavity_view_factors(G_FULL, shaded=True, method='montecarlo', n_rays=N_MC)['inner']
    assert an.names == mc.names, (an.names, mc.names)
    big = an.F > 0.05
    assert (np.abs(mc.F - an.F)[big] / an.F[big]).max() < TOL


def test_unshaded_single_enclosure():
    s = cavity_view_factors(G_LOW, shaded=False, n_rays=N_MC // 2)['single']
    _rows_and_recip(s)
    assert s.names[0] == 'B0' and 'R' in s.idx
    s2 = cavity_view_factors(G_FULL, shaded=False)['single']
    assert s2.method == 'analytic'
    _rows_and_recip(s2)


def test_cache_roundtrip(tmp_path):
    t = time.perf_counter()
    a = cavity_view_factors(G_LOW, shaded=True, n_rays=N_MC // 4, cache_dir=str(tmp_path))
    t1 = time.perf_counter() - t
    t = time.perf_counter()
    b = cavity_view_factors(G_LOW, shaded=True, n_rays=N_MC // 4, cache_dir=str(tmp_path))
    t2 = time.perf_counter() - t
    for k in a:
        assert np.array_equal(a[k].F, b[k].F) and a[k].names == b[k].names
    assert t2 < 0.2 * t1


def _gap_transport(F, A, n, zc, H, T, eps):
    """Net upward radiative transfer across mid-plane, isolated outer gap."""
    sig = 5.670374e-8
    N = len(A)
    rerad = np.zeros(N, bool); rerad[2 * n:] = True
    E = sig * T ** 4
    M = np.eye(N); b = np.zeros(N)
    for i in range(N):
        if rerad[i]:
            M[i] -= F[i]
        else:
            M[i] -= (1 - eps[i]) * F[i]; b[i] = eps[i] * E[i]
    J = np.linalg.solve(M, b)
    ring = np.r_[np.arange(n), np.arange(n), -1, -1]
    lo = [i for i in range(N) if ring[i] >= 0 and zc[ring[i]] < H / 2]
    hi = [i for i in range(N) if ring[i] >= 0 and zc[ring[i]] > H / 2]
    return sum(A[i] * F[i, j] * (J[i] - J[j]) for i in lo for j in hi)


def test_axial_transport_sanity_and_truncation():
    """§6.5: linear wall T (lower end hotter, 10 K/m, ~1800 C) on block and shade
    outer faces; effective axial conductivity of order 1e3 W/m-K; truncating the
    matrix to ±3 rings must lose a large share of it."""
    enc = cavity_view_factors(G_LOW, shaded=True)['outer']
    e = G_LOW['edges']; n = len(e) - 1; zc = 0.5 * (e[1:] + e[:-1]); H = e[-1]
    Tz = 2073.0 - 10.0 * (zc - H / 2)
    T = np.r_[Tz, Tz, 2073.0, 2073.0]
    eps = np.r_[np.full(n, 0.85), np.full(n, 0.9), 0.85, 0.85]
    Q = _gap_transport(enc.F, enc.A, n, zc, H, T, eps)
    k_eff = Q / (np.pi * (R_IN ** 2 - R_SH ** 2) * 10.0)
    assert 500 < k_eff < 5000, k_eff
    Ft = enc.F.copy()
    ring = np.r_[np.arange(n), np.arange(n), -1, -1]
    for i in range(len(ring)):
        for j in range(len(ring)):
            if ring[i] >= 0 and ring[j] >= 0 and abs(ring[i] - ring[j]) > 3:
                Ft[i, i] += Ft[i, j]; Ft[i, j] = 0.0
    Qt = _gap_transport(Ft, enc.A, n, zc, H, T, eps)
    assert Qt < 0.85 * Q, (Q, Qt)
