"""Optimizer speed-ups (2026-11-25): population seeding, sizing warm start, per-pattern bests."""
import json
import os

import numpy as np

from thermal import optimize as O


def _row(x_phys, cost, feasible=True, layout=None, by_layout=None, swing=450.0, vf=1.02):
    return dict(design=x_phys, lifetime_cost=cost, feasible=feasible, capex0=cost / 2, RTE=0.27, N_ps=7, N_h=12,
                energy_mismatch=0.01, layout=layout, by_layout=by_layout, sizing_converged=True, swing=swing,
                volume_factor=vf)


def _design(opt, u):
    lo, hi = np.array(O.var_bounds(opt)).T
    return O.decode(lo + np.asarray(u) * (hi - lo), opt)


def test_seed_population_uses_best_feasible_designs_and_bounds():
    opt = O.build_opt({})
    n = len(O.var_bounds(opt))
    rows = [_row(_design(opt, np.full(n, f)), cost) for f, cost in ((0.2, 3.0), (0.8, 1.0), (0.5, 2.0))]
    rows.append(_row(_design(opt, np.full(n, 0.3)), 0.5, feasible=False))
    X = O.seed_population(opt, popsize=2, seed_rows=rows, seed_frac=0.3)
    assert X.shape == (2 * n, n)
    lo, hi = np.array(O.var_bounds(opt)).T
    assert np.all(X >= lo - 1e-9) and np.all(X <= hi + 1e-9)
    integ = O.var_integrality(opt)
    assert np.allclose(X[:, integ], np.round(X[:, integ]))
    # the cheapest feasible design comes first; the infeasible one is not seeded
    assert np.allclose(X[0], O.encode(rows[1]['design'], opt))
    assert not any(np.allclose(x, O.encode(rows[3]['design'], opt)) for x in X)


def test_warm_start_uses_the_nearest_sized_design(tmp_path):
    opt = O.build_opt({})
    n = len(O.var_bounds(opt))
    rows = [_row(_design(opt, np.full(n, 0.5)), 1.0, swing=420.0, vf=1.05),
            _row(_design(opt, np.full(n, 0.9)), 1.0, swing=600.0, vf=0.97)]
    assert O.write_warm_seed(opt, str(tmp_path), rows) == 2
    obj = O.Objective({}, opt, log_dir=str(tmp_path))
    lo, hi = np.array(O.var_bounds(opt)).T
    x = lo + np.full(n, 0.52) * (hi - lo)
    d = O.decode(x, opt)
    w = obj._warm_start(x, d)
    assert w is not None and abs(w['swing'] - 420.0) < 1e-9
    assert abs(d['T_min_guess_C'] - (d['T_max_C'] - 420.0)) < 1e-9 and d['volume_factor_guess'] == 1.05
    far = lo + np.full(n, 0.05) * (hi - lo)                 # nothing within warm_dist: cold start
    assert obj._warm_start(far, O.decode(far, opt)) is None


def test_layout_bests_from_multi_pattern_records():
    bl = lambda c1, c2: {'checkerboard': dict(feasible=True, lifetime_cost=c1, capex0=1, RTE=0.2, N_ps=9, N_h=16),
                         'hex6': dict(feasible=c2 is not None, lifetime_cost=c2, capex0=1, RTE=0.2, N_ps=7, N_h=24)}
    rows = [dict(feasible=True, layout='checkerboard', lifetime_cost=5.0, by_layout=bl(5.0, 6.0)),
            dict(feasible=True, layout='hex6', lifetime_cost=4.0, by_layout=bl(7.0, 4.0)),
            dict(feasible=True, layout='checkerboard', lifetime_cost=4.5, by_layout=bl(4.5, None))]
    b = O.layout_bests(rows)
    assert list(b) == ['hex6', 'checkerboard']
    assert b['hex6']['lifetime_cost'] == 4.0 and b['checkerboard']['lifetime_cost'] == 4.5


def test_layout_patterns_option():
    assert O.layout_patterns({'layout': {'pattern': 'all'}}) == ['checkerboard', 'hex6', 'honeycomb', 'hex3']
    assert O.layout_patterns({'layout': {'pattern': ['hex6', 'HEX3']}}) == ['hex6', 'hex3']
    assert O.layout_patterns({}) == ['checkerboard']
