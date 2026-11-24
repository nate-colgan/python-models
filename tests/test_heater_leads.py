"""Heater lead model (thermal/heater_leads.py): conduction, Joule heat, energy balance."""
import math
import os
import warnings

import numpy as np
from scipy.integrate import quad

from thermal.build import load_config
from thermal import heater_leads as HL

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = load_config(os.path.join(ROOT, 'tinless_model_input.yaml'))
NO_EXCHANGE = {'gap': 10.0, 'eps_wall': 1e-9, 'k_gap': 1e-14, 'r_far': None}


def _ctx(**kw):
    return HL.lead_context(CFG, {'charge_hr': 30.0}, dict({'P_heater': 576e3}, **kw))


def test_pure_conduction_matches_kirchhoff_integral():
    ctx = _ctx(zones=['copper'], **NO_EXCHANGE)
    lead = dict(mats=['copper'], lens=[ctx['L']], d=[0.05], cooled=False)
    s = HL.solve_state(ctx, lead, dict(I=0.0, T_hot=500.0, T_ins_hot=500.0))
    A = math.pi * 0.05 ** 2 / 4
    exact = quad(HL.MATERIALS['copper']['k'], ctx['li']['T_term'], 500.0)[0] * A / ctx['L']
    assert abs(s['Q_hot'] / exact - 1) < 1e-3 and abs(s['Q_end'] / exact - 1) < 1e-3


def test_joule_heat_splits_evenly_with_equal_end_temperatures():
    ctx = _ctx(zones=['copper'], **NO_EXCHANGE)
    lead = dict(mats=['copper'], lens=[ctx['L']], d=[0.05], cooled=False)
    T0 = ctx['li']['T_term']
    s = HL.solve_state(ctx, lead, dict(I=2000.0, T_hot=T0, T_ins_hot=T0))
    assert abs(-s['Q_hot'] / s['Q_J'] - 0.5) < 1e-3 and abs(s['Q_end'] / s['Q_J'] - 0.5) < 1e-3


def test_energy_balance_closes_both_modes():
    ctx = _ctx()
    for cooled in (False, True):
        r = HL.evaluate(ctx, HL.make_lead(ctx, [0.45, 0.12, 0.15], [0.3, 0.4], cooled))
        assert r['ok']
        for st in ('charge', 'idle'):
            s = r[st]
            assert abs(s['balance']) < 1e-3 * max(1.0, abs(s['Q_J']) + abs(s['Q_hot']))


def test_element_section_graphite_lead_runs_away():
    """The element's own cross-section continued through the top insulation cannot carry the
    current: 9.6 kA in a 158 mm graphite bar has no steady state below the runaway limit."""
    ctx = _ctx()
    r = HL.evaluate(ctx, HL.baseline_lead(ctx))
    assert not r['ok']


def test_zone_lengths_respect_minimum():
    ctx = _ctx()
    mz = ctx['li']['min_zone_len']
    for fr in ([0.0, 0.0], [1.0, 1.0], [0.3, 0.6]):
        lead = HL.make_lead(ctx, [0.3, 0.1, 0.2], fr, False)
        assert min(lead['lens']) >= mz - 1e-12 and abs(sum(lead['lens']) - ctx['L']) < 1e-12
    lead = HL.make_lead(ctx, [0.3, 0.2], [0.5], False, zones=['graphite', 'copper'])
    assert lead['mats'] == ['graphite', 'copper'] and abs(sum(lead['lens']) - ctx['L']) < 1e-12


def test_joint_above_colder_material_limit_is_a_violation():
    """A passive graphite / copper lead with a short graphite zone puts the joint far above the
    copper limit: the joint check must report it."""
    ctx = _ctx()
    r = HL.evaluate(ctx, HL.make_lead(ctx, [0.4, 0.3], [0.1], False, zones=['graphite', 'copper']))
    j = r['charge']['T_joint'][0]
    assert j['between'] == ('graphite', 'copper') and j['limit'] == ctx['mats']['copper']['T_max']
    assert j['T'] > j['limit'] and r['viol_K'] >= j['T'] - j['limit'] - 1e-9 and not r['feasible']


def test_thermax_profile_carries_uniform_flux():
    """Undisturbed Thermax: steady 1-D, so k(T_w) dT_w/dx is the same at every height and the
    hot face is at the block T_max."""
    from thermal.properties import thermax_tcx
    ctx = _ctx()
    T_hot = ctx['charge']['T_ins_hot']
    assert abs(T_hot - ctx['T_max']) < 1e-9 and ctx['idle']['T_ins_hot'] == T_hot
    x = np.linspace(0, ctx['L'], 2001)
    Tw, w = HL._ins(ctx, T_hot, x)
    q = -thermax_tcx(0.5 * (Tw[1:] + Tw[:-1])) * np.diff(Tw) / np.diff(x)
    assert abs(Tw[0] - T_hot) < 1e-6 and abs(Tw[-1] - ctx['li']['T_ins_cold']) < 1e-6
    assert q.std() / q.mean() < 1e-3
    assert w[0] == 0 and abs(w[-1] - 1) < 1e-12


def test_idle_open_joint_cuts_idle_heat_and_balances():
    """Opening the graphite / tungsten joint while idle: the energy balance still closes and the
    heat rejected at the cooled copper falls; charging is unchanged."""
    lead = None
    out = {}
    for after in (None, 'graphite'):
        ctx = _ctx(idle_break_after=after)
        lead = HL.make_lead(ctx, [0.45, 0.12, 0.12], [0.6, 0.1], True)
        out[after] = HL.evaluate(ctx, lead)
    c, b = out[None], out['graphite']
    assert b['idle']['open_joint'] is not None and c['idle']['open_joint'] is None
    assert abs(b['idle']['balance']) < 1e-3 * max(1.0, abs(b['idle']['Q_hot']))
    assert b['idle']['Q_rej'] < 0.5 * c['idle']['Q_rej']
    assert abs(b['charge']['Q_rej'] - c['charge']['Q_rej']) < 1e-9


def test_lead_table_interpolation_per_ampere():
    """LeadTable: values stored per ampere, linear in t_top / T_max and in I, clamped outside."""
    grid = dict(I_kA=[4.0, 10.0], t_top=[1.0, 2.0], T_max_C=[1800.0, 2400.0])
    rows = []
    for I in grid['I_kA']:
        for t in grid['t_top']:
            for T in grid['T_max_C']:
                per_A = 10.0 + 2.0 * t + 0.01 * (T - 1800.0) + 0.1 * (I - 4.0)   # $ per A
                rows.append(dict(I_kA=I, t_top=t, T_max_C=T, ok=True, feasible=True,
                                 **{k: per_A * I * 1e3 for k in HL.TABLE_KEYS}))
    lt = HL.LeadTable(dict(key='test', grid=grid, rows=rows, scenario=None, elec_price=75.0))
    r = lt(7e3, 1.5, 2100.0)
    exact = (10.0 + 3.0 + 3.0 + 0.3) * 7e3
    assert abs(r['total'] - exact) < 1e-6 * exact and not r['clamped']
    r = lt(20e3, 5.0, 2100.0)               # current and t_top above the grid: per-A values at the edge
    assert r['clamped'] and abs(r['total'] - (10.0 + 4.0 + 3.0 + 0.6) * 20e3) < 1e-6 * r['total']
    assert 'I = 4 kA' in lt.summary()


def test_optimizer_loads_lead_table(tmp_path):
    """optimize.lead_table: None without a table; with opt['leads']['table'] a LeadTable (cached)."""
    import json
    from thermal import optimize as O
    assert O.lead_table(O.OPT_DEFAULTS) is None
    grid = dict(I_kA=[4.0, 10.0], t_top=[1.0, 2.0], T_max_C=[1800.0, 2400.0])
    rows = [dict(I_kA=I, t_top=t, T_max_C=T, ok=True, feasible=True, **{k: 5.0 * I * 1e3 for k in HL.TABLE_KEYS})
            for I in grid['I_kA'] for t in grid['t_top'] for T in grid['T_max_C']]
    path = tmp_path / 'lt.json'
    path.write_text(json.dumps(dict(key='t', grid=grid, rows=rows, scenario=None, elec_price=75.0)))
    opt = O.merge(O.OPT_DEFAULTS, {'leads': {'table': str(path)}})
    lt = O.lead_table(opt)
    assert lt is O.lead_table(opt) and abs(lt(6e3, 1.5, 2000.0)['total'] - 5.0 * 6e3) < 1e-6


def test_profile_joints_match_and_taper_is_limited():
    """Both sides of a joint share its contact diameter; a zone widens from a joint at no more than
    taper_deg (half-angle) up to its body diameter; volumes integrate the profile."""
    ctx = _ctx(taper_deg=60.0)
    lead = HL.make_lead(ctx, [0.40, 0.10, 0.20], [0.5, 0.5], True, fj=[0.5, 1.0])
    edges = np.concatenate([[0.0], np.cumsum(lead['lens'])])
    assert abs(lead['d_joint'][0] - 0.05) < 1e-12 and abs(lead['d_joint'][1] - 0.10) < 1e-12
    for j, xj in enumerate(edges[1:-1]):
        lo = HL.lead_profile(lead, np.array([xj - 1e-9]))[1][0]
        hi = HL.lead_profile(lead, np.array([xj + 1e-9]))[1][0]
        assert abs(lo - lead['d_joint'][j]) < 1e-6 and abs(hi - lead['d_joint'][j]) < 1e-6
    xs = np.linspace(1e-9, ctx['L'] - 1e-9, 20001)
    D = HL.lead_profile(lead, xs)[1]
    slope = np.abs(np.diff(D) / np.diff(xs)) / 2                     # dr/dx
    assert slope.max() <= math.tan(math.radians(60.0)) * (1 + 1e-6)
    # graphite: body 0.40 m narrowing to the 0.05 m joint over (0.40 - 0.05) / (2 tan 60) m
    t2 = 2 * math.tan(math.radians(60.0))
    Lg, lt = lead['lens'][0], (0.40 - 0.05) / t2
    cone = math.pi / 12 * lt * (0.40 ** 2 + 0.40 * 0.05 + 0.05 ** 2)
    exact = math.pi * 0.40 ** 2 / 4 * (Lg - lt) + cone
    assert abs(HL.zone_volumes(lead)[0] / exact - 1) < 1e-3
