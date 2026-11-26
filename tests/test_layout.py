"""Hot-zone layouts (thermal/layout.py): feasible stick counts, perimeter heaters, spacing, walls."""
import math

import numpy as np
import pytest

from thermal import layout as L

TIER_A_BELOW_20 = {        # Nate, 2026-11-25: symmetric (>= 3 mirror axes), compact, every stick surrounded
    'checkerboard': {1: 4, 4: 9, 5: 12, 9: 16, 12: 21, 13: 24, 16: 25},
    'hex6': {1: 6, 3: 13, 4: 18, 6: 22, 7: 24, 10: 33, 12: 37, 13: 42, 15: 46, 16: 48, 18: 52, 19: 54},
    'honeycomb': {1: 3, 3: 6, 6: 10, 7: 12, 10: 15, 12: 18, 13: 21, 15: 21, 18: 25, 19: 27},
    'hex3': {1: 3, 4: 6, 6: 7, 9: 10, 12: 13, 13: 12, 16: 15, 18: 16},
}


def test_feasible_counts_below_20():
    for name, want in TIER_A_BELOW_20.items():
        assert L.feasible_counts(name) == want, name


@pytest.mark.parametrize('name', list(L.PATTERNS))
def test_every_stick_fully_surrounded(name):
    """Every Tier A cluster and a round cluster: each stick has its full ring of heaters at unit
    spacing, and every heater is next to a stick."""
    ring = L.PATTERNS[name]['around_stick']
    for n in list(L.feasible_counts(name)) + [25]:
        lay = L.plan(name, n)
        st, he = np.array(lay['sticks']), np.array(lay['heaters'])
        D = np.hypot(*(st[:, None, :] - he[None, :, :]).transpose(2, 0, 1))
        assert np.all((np.abs(D - 1) < 1e-6).sum(axis=1) == ring), (name, n)
        assert np.all((np.abs(D - 1) < 1e-6).any(axis=0)), (name, n)
        assert lay['N'] == len(st) and lay['N_h'] == len(he)


def test_plan_snaps_up_below_20_and_any_count_above():
    lay = L.plan('checkerboard', 7)
    assert (lay['N'], lay['N_h'], lay['tier']) == (9, 16, 'A')
    assert L.plan('hex3', 7)['N'] == 9 and L.plan('hex6', 7)['N'] == 7
    big = L.plan('checkerboard', 23)
    assert big['N'] == 23 and big['tier'] == 'round'


@pytest.mark.parametrize('name', list(L.PATTERNS))
def test_large_plants_approach_the_bulk_ratio(name):
    lay = L.plan(name, 600)
    assert abs(lay['N_h'] / lay['N'] / L.PATTERNS[name]['rho'] - 1) < 0.12


def test_checkerboard_spacing_and_walls_match_the_old_pitch_model():
    """Checkerboard: d = sqrt((pi r_out^2 + pi r_hc^2) / 2) and the rotated-square walls as before."""
    from thermal import tank as tk
    r_in, r_out, r_hc = 0.963, 1.684, 0.344
    p_old = math.sqrt((math.pi * r_out ** 2 + math.pi * r_hc ** 2) / 2)
    S = math.pi * r_in / 2
    d, w = tk.cavity_webs(r_in, r_out, r_hc)
    assert abs(d - p_old) < 1e-12
    assert abs(w['stick-heater'] - (p_old - S / math.sqrt(2) - r_hc)) < 1e-9
    assert abs(w['stick-stick'] - (math.sqrt(2) * p_old - S)) < 1e-9
    assert abs(w['heater-heater'] - (math.sqrt(2) * p_old - 2 * r_hc)) < 1e-9


def test_spacing_from_plan_area_per_stick():
    """k d^2 = pi r_out^2 + rho pi r_hc^2 for every pattern."""
    for name, p in L.PATTERNS.items():
        d = L.spacing(name, 1.5, 0.4)
        assert abs(p['k'] * d * d - (math.pi * 1.5 ** 2 + p['rho'] * math.pi * 0.4 ** 2)) < 1e-9


def test_heater_cell_holds_its_share_of_graphite():
    """build_heater: the heater cell holds heater.cell_graphite_ratio x the stick cell's graphite."""
    import os
    from thermal.build import load_config, build_tinless, build_heater
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_config(os.path.join(root, 'tinless_model_input.yaml'))
    m = build_tinless(cfg, geometry_only=True)
    V1 = build_heater(m).V
    m.cfg['heater']['cell_graphite_ratio'] = 9 / 16
    assert abs(build_heater(m).V / V1 - 9 / 16) < 1e-12
