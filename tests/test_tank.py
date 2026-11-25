"""System-scale tank model (thermal/tank.py), brief §7.
Run:  python -m pytest tests/test_tank.py -q
"""
import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import quad

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from thermal import tank as T  # noqa: E402

C = T.merge_defaults({'columns': {'f_3D': 0.54}})


def test_kirchhoff_integrals_exact():
    for a, b in ((20.0, 2400.0), (85.0, 1100.0), (1100.0, 2400.0), (1500.0, 2300.0)):
        assert abs(T.phi_thermax(a, b) / quad(lambda t: float(T.k_thermax(t)), a, b)[0] - 1) < 1e-9
        assert abs(T.psi_oxide(a, b) / quad(lambda t: float(T.k_oxide(t)), a, b)[0] - 1) < 1e-9
        ref = quad(lambda t: float(T.k_graphite(t)), a, b, points=list(T._KG_T[(T._KG_T > a) & (T._KG_T < b)]),
                   limit=200)[0]
        assert abs(T.psi_graphite(a, b) / ref - 1) < 1e-9


def test_cylinder_tends_to_planar():
    """Side wall -> planar slab as t_side / r_i -> 0 (same faces)."""
    c = T.merge_defaults({'t_side': 0.5})
    hz = SimpleNamespace(H_block=10.0)
    for r_i, tol in ((50.0, 0.01), (500.0, 0.001)):
        geo = SimpleNamespace(r_i=r_i, r_o=r_i + 0.5)
        Q, Ts = T.side_wall(2400.0, geo, hz, c)
        planar = 2 * math.pi * r_i * hz.H_block * T.phi_thermax(Ts, 2400.0) / 0.5
        assert abs(Q / planar - 1) < tol


def test_creep_ratio():
    col = C['columns']
    assert abs(T.sigma_allow(2200.0, col) / T.sigma_allow(2400.0, col) - 1.22) < 0.01
    assert T.sigma_allow(20.0, col) == col['sigma_cap']


def _stick(P=120e3, E=6.0e6 * 3600):
    return T.StickInputs(P_stick=P, E_stick_e=E, E_stick_th=13.9e6 * 3600, V_block=50.0, r_out=1.8, r_in=1.3,
                         H_block=6.0, L_ps=3.0, T_min=1600.0, T_max=2400.0, rho_g=1450.0, t_discharge=E / P,
                         t=np.array([0.0, 1e5]), T_avg=np.array([2400.0, 1600.0]),
                         T_top=np.array([2380.0, 1580.0]), T_bot=np.array([2420.0, 1640.0]), A_hcav_default=2.0)


def test_checkerboard_and_energy_mismatch():
    st = _stick(E=5.5e6 * 3600)          # 209 sticks x 5.5 MWh vs 1.25 GWh -> -8%
    hz = T.size_hot_zone(st, 25e6, 50, A_hcav=2.0)
    assert hz.N_ps == math.ceil(25e6 / st.P_stick) == hz.N_h
    assert abs(2 * hz.N_ps * hz.pitch ** 2 / (hz.N_ps * hz.A_c) - 1) < 1e-12
    assert abs(hz.energy_mismatch - (hz.N_ps * st.E_stick_e / (25e6 * 50 * 3600) - 1)) < 1e-12
    res = T.tank_system(st, {'columns': {'f_3D': 0.54}})
    assert abs(res.hz.A_hcav - 2.0) < 1e-12
    assert any('energy mismatch' in w for w in res.warnings)


def test_floor_convergence_and_air_balance():
    res = T.tank_system_reference({'columns': {'f_3D': 0.54}})
    cd, geo = res.cols, res.geo
    b, fl, it = T.bottom_and_floor(cd, geo.A_hot, geo.D_tank, 2400.0, 40.0, res.cfg)
    assert it['converged']
    assert abs(b['Q_bot'] / (T.AIR['rho'] * fl['Vdot'] * T.AIR['cp'] * fl['dT_air']) - 1) < 1e-12
    # built column at its design point reproduces the joint at T1
    Q1, Tj = T.column_heat(cd, cd.T_design, cd.T_floor_design)
    assert abs(Tj - res.cfg['columns']['T1']) < 0.5


def test_fixed_graphite_length_option():
    c = T.merge_defaults({'columns': {'L_graphite': 1.0, 'T1': 2500.0}})
    cd = T.design_columns(1000.0, 115e3, 2400.0, 80.0, 2.0, c['columns'])
    assert abs(cd.L_g - 1.0) < 1e-12 and abs(cd.L_ox - 1.0) < 1e-12
    Q1, Tj = T.column_heat(cd, 2400.0, 80.0)
    assert 80 < Tj < 2400


def test_reference_design_point():
    """Brief §7.2 v3 point, within ~±30% (sanity values, not targets). Uses the brief's
    own column basis (pitch 1.5 m, sigma_ref 1.62 MPa, 0.15 m square minimum section)."""
    res = T.tank_system_reference({'columns': {'f_3D': 0.54, 'T1': 1150.0, 'pitch': 1.5, 'sigma_ref': 1.62e6,
                                               'w_min': 0.0225 / 0.23},
                                   't_side': 1.03, 't_top': 1.03})
    near = lambda v, ref, tol=0.30: abs(v / ref - 1) <= tol
    assert near(res.cols.N_col, 437, 0.02)
    assert near(math.sqrt(res.cols.A_top), 0.40, 0.05) and near(math.sqrt(res.cols.A_joint), 0.15, 0.05)
    assert near(res.cost['weights']['column_graphite_t'], 144)
    assert near(res.floor_design['Vdot'], 40) and near(res.floor_design['P_fan'], 10e3)
    assert res.floor_design['T_floor_max'] <= 93.0
    assert near(res.design['q_bot'], 1090)
    assert near(res.loss_cycle['Q_total'] * 86400 / res.hz.E_th * 100, 1.5)
    g = res.cost['groups']
    assert near(g['insulation'][1] + g['supports'][1], 8.2e6)
    assert 15e6 * 0.7 <= res.cost['total'][1] <= 24e6 * 1.3


def _storage(nA=6, nR=4, r_in=1.0, r_out=2.0, L=3.0, T0=2300.0 + 273.15):
    from thermal.components import GraphiteBlock, Pipe, StorageModule
    blk = GraphiteBlock(np.full((nA, nR), T0), inner_radius=r_in, outer_radius=r_out, length=L, th_ins=0.0)
    pipe = Pipe(OD=0.02, length=L, wall_thickness=0.001, roughness=1e-5, th_ins=0.0)
    return StorageModule(N_pipes=1, Pipe=pipe, Block=blk, num_ax_segments=nA, num_rad_segments=nR,
                         bypass_range=[0, 1])


def test_cell_sink_per_cell_loss():
    """Uniform block: sink total = (q_top + q_bot) x graphite plan area / fill, all on the
    top / bottom rings; q matches the plant's roof and bottom losses per m2."""
    res = T.tank_system_reference({'columns': {'f_3D': 0.54}}, N_ps=200)
    sk = res.cell_sink
    st = _storage()
    Q = sk.for_storage(st)(st.block.temp_dist)
    assert np.all(Q[1:-1] == 0)
    qt, qb = sk.q(2300.0)
    A_g = math.pi * (2.0 ** 2 - 1.0 ** 2)
    assert abs(Q[-1].sum() / (qt * A_g * sk.f_open) - 1) < 1e-9 and abs(Q[0].sum() / (qb * A_g * sk.f_open) - 1) < 1e-9
    L = T.losses(2300.0, 2300.0, 2300.0, res.geo, res.hz, res.cols, res.cfg)
    assert abs(qt * res.geo.A_hot / L['Q_top'] - 1) < 2e-3 and abs(qb * res.geo.A_hot / L['Q_bot'] - 1) < 2e-3


def test_update_temps_node_sink_energy():
    """A per-node sink removes exactly its energy from the block (adiabatic otherwise)."""
    from thermal.properties import storage_rho, storage_cp
    st = _storage()
    T0 = st.block.temp_dist.copy()
    Qn = np.zeros_like(T0)
    Qn[-1], Qn[0, 2] = 150.0, 400.0
    dt = 3600.0
    Tn, _, lost = st.update_temps(T0, np.zeros(T0.shape[0]), dt, 10, 300.0, False, Q_out=Qn)
    nR = T0.shape[1]
    r = 1.0 + np.arange(nR + 1) / nR
    vol = math.pi * np.diff(r ** 2)[None, :] * st.axial_dz()[:, None]
    dE = float(np.sum(storage_rho(2000.0) * storage_cp(2000.0) * vol * (Tn - T0)))
    assert abs(-dE / (Qn.sum() * dt) - 1) < 1e-9 and abs(lost / (Qn.sum() * dt) - 1) < 1e-12


def test_pitch_from_capacity():
    """Tributary area x areal load = column capacity at the hot face; A_top = w_max x thickness."""
    col = T.merge_defaults({})['columns']
    A_max = col['w_max'] * col['thickness']
    p_bear = 111e3
    p = T.column_pitch(p_bear, 2400.0, col)
    assert abs(p_bear * p ** 2 / (float(T.sigma_allow(2400.0, col)) * A_max) - 1) < 1e-12
    cd = T.design_columns(1000.0, p_bear, 2400.0, 80.0, 2.0, col)
    assert abs(cd.pitch - p) < 1e-12 and abs(cd.A_top / A_max - 1) < 1e-9
    assert T.column_pitch(p_bear, 2400.0, dict(col, pitch=1.7)) == 1.7
    # defaults reproduce the v3 slide column: ~333 kN capacity at the 2400 C hot face, and the
    # slide's stresses (333 kN / (width x 230 mm)) at its tabulated widths
    assert abs(float(T.sigma_allow(2400.0, col)) * A_max / 333e3 - 1) < 0.01
    for w, sig in ((0.136, 10.6), (0.312, 4.6), (0.513, 2.8)):
        assert abs(333e3 / (w * col['thickness']) / 1e6 / sig - 1) < 0.02
    # creep allowable matches the slide's 30-yr column (1237, 1945, 2351 C -> 19.0, 4.7, 2.9 MPa)
    for Tc, a in ((1237.0, 19.0), (1945.0, 4.7), (2351.0, 2.9)):
        assert abs(float(T.sigma_allow(Tc, col)) / 1e6 / a - 1) < 0.03


def test_heater_cell_matches_stick_and_closes_energy():
    """build_heater: same graphite volume / height / mesh as the stick cell around the
    heater cavity; the charge stops at the volume-weighted T_max and conserves energy."""
    from thermal.build import build_tinless, build_heater, load_config
    from thermal.properties import storage_rho, storage_cp
    cfg = load_config(os.path.join(ROOT, 'tinless_model_input.yaml'))
    m = build_tinless(cfg, {'storage.num_ax_segments': 10, 'storage.num_rad_segments': 4}, geometry_only=True)
    hm = build_heater(m, area_ratio=4.0)
    st = m.storage
    assert abs(hm.V / (math.pi * (st.block.rad_out ** 2 - st.block.rad_in ** 2) * m.H_block) - 1) < 1e-12
    assert np.allclose(hm.storage.axial_dz(), st.axial_dz())
    from thermal.heater import element
    assert abs(hm.r_in - 4.0 * element(cfg['heater'], m.H_block, m.T_min)['r_h']) < 1e-12
    r = hm.sim.run_heating_simulation(hm.heater, end_time=200 * 3600, **hm.run_kwargs)
    assert r['T_avg'][-1] >= hm.T_max and r['T_avg'][-2] < hm.T_max
    nR = hm.storage.num_rad_segments
    rr = hm.r_in + np.arange(nR + 1) * (hm.r_out - hm.r_in) / nR
    W = np.diff(rr ** 2)[None, :] * hm.storage.axial_dz()[:, None]
    Tf = r['temp_snapshots'][-1]
    assert abs(float((Tf * W).sum() / W.sum()) - r['T_avg'][-1]) < 1e-9
    # heater energy (step-held power) = block gain; constant cp / rho in properties
    E_in = float(np.sum(r['Q_heater'][1:] * np.diff(r['time'])))
    dE = storage_rho(2400.0) * storage_cp(2400.0) * hm.V * (r['T_avg'][-1] - r['T_avg'][0])
    assert abs(E_in / dE - 1) < 1e-6


def test_tube_heater_element_sizing():
    """Tube element: V^2 / R = P, R = rho_e legs^2 H / A, A = pi D^2 w (1 - w); rod unchanged."""
    from thermal.heater import element
    from thermal.properties import heater_graphite_resistivity
    h = dict(type='tube', voltage=60.0, wall_ratio=0.1, legs=2, density=1850.0, max_temperature=2873.0,
             max_power_per_m=150e3)
    H, P = 10.0, 600e3
    el = element(h, H, 2000.0, P=P)
    rho = float(np.max(heater_graphite_resistivity(np.linspace(2000.0, 2873.0, 41))))   # highest over the range
    assert abs(float(heater_graphite_resistivity(18 + 273.15)) - 1.6486e-5) < 1e-12        # Mersen 2020 data
    assert abs(float(heater_graphite_resistivity(2002 + 273.15)) - 1.1357e-5) < 1e-12
    assert abs(el['rho_e'] - rho) < 1e-15
    assert abs(el['V'] ** 2 / el['R'] / P - 1) < 1e-12
    assert abs(rho * 2 ** 2 * H / el['A_cross'] / el['R'] - 1) < 1e-12
    assert abs(math.pi * el['D'] ** 2 * 0.1 * 0.9 / el['A_cross'] - 1) < 1e-12
    assert abs(el['t'] / el['D'] - 0.1) < 1e-12 and abs(el['I'] - P / 60.0) < 1e-9
    assert element(h, H, 2000.0)['P'] == 150e3 * H                       # default: the power cap
    assert element(h, H, 2000.0, P=P / 4)['D'] < el['D']                 # element follows the power
    rod = element(dict(type='rod', side_len=0.15), H, 2000.0, P=P)
    assert abs(rod['r_h'] - 2 * 0.15 / math.pi) < 1e-12 and abs(rod['A_cross'] - 0.0225) < 1e-12


def test_optimizer_decode_valid_and_roundtrip():
    """Every point in the design box decodes to a valid geometry; encode inverts decode."""
    from thermal import optimize as O
    opt = O.OPT_DEFAULTS
    g = opt['geometry']
    rng = np.random.default_rng(0)
    lo, hi = np.array([b[0] for b in O.BOUNDS]), np.array([b[1] for b in O.BOUNDS])
    for _ in range(500):
        x = lo + rng.random(len(lo)) * (hi - lo)
        x[O.INTEGRALITY] = np.round(x[O.INTEGRALITY])
        d = O.decode(x, opt)
        assert 0.10 - 1e-12 <= d['ps_side'] <= d['liner_side'] - 2 * g['ps_liner_gap'] + 1e-12
        assert d['L'] + g['min_lower_zone'] - 1e-12 <= d['H_block'] <= g['H_max'] + 1e-12
        assert 1600 - 1e-9 <= d['T_max_C'] <= g['T_cap_C'] + 1e-9
        assert 5.0 <= d['avg_flux'] <= 20.0
        assert 2.0 <= d['L_bot'] <= 4.0
        n8 =d['shade_thickness'] / (0.0254 / 8)                 # whole 1/8 in, 1/4 - 6 in
        assert abs(n8 - round(n8)) < 1e-9 and 2 <= round(n8) <= 48
        x2 = O.encode(d, opt)
        d2 = O.decode(x2, opt)
        for k in ('liner_side', 'ps_side', 'L', 'avg_flux', 'shade_thickness', 'T_max_C', 't_side', 't_top'):
            assert abs(d2[k] - d[k]) < 1e-9, k
        if d['H_block'] > d['L']:
            assert abs(d2['H_block'] - d['H_block']) < 1e-9


def test_degradation_keeps_geometry():
    """cfg['degradation'] changes the cell efficiency and emissivities, not the block:
    cell D -> efficiency and inactive emissivity, bare D -> bare emissivity."""
    from thermal.build import build_tinless, load_config
    cfg = load_config(os.path.join(ROOT, 'tinless_model_input.yaml'))
    ov = {'storage.num_ax_segments': 10, 'storage.num_rad_segments': 4}
    m0 = build_tinless(cfg, ov, geometry_only=True)
    m1 = build_tinless(dict(cfg, degradation={'cell': 0.2, 'bare': 0.1, 'emis_end': 0.8}), ov, geometry_only=True)
    assert m1.r_out == m0.r_out and m1.r_in == m0.r_in and m1.V_block == m0.V_block
    assert abs(m1.sim.cell_eff_factor - 0.8) < 1e-12 and m0.sim.cell_eff_factor == 1.0
    for (b0, a0, eb0, ei0), (b1, a1, eb1, ei1) in zip(m0.powerstick.emissivity_table, m1.powerstick.emissivity_table):
        assert a1 == a0
        assert abs(eb1 - (eb0 + 0.1 * (0.8 - eb0))) < 1e-12
        assert abs(ei1 - (ei0 + 0.2 * (0.8 - ei0))) < 1e-12


def test_maintenance_event_counts():
    """Replacements and cleanings strictly before end of life; cleanings exclude the
    ones that coincide with a replacement."""
    from thermal.optimize import _n_events
    assert _n_events(0.0, 2.0) == 0
    assert _n_events(1.9, 2.0) == 0
    assert _n_events(2.0, 2.0) == 0          # reaches the threshold exactly at end of life
    assert _n_events(5.0, 2.0) == 2
    assert _n_events(5.0, 1.0) == 4          # resets at 1, 2, 3, 4 nm; 2 and 4 are replacements


def test_pareto_front():
    from thermal.optimize import pareto
    rows = [dict(capex_total=c, RTE=e, violations=v) for c, e, v in
            [(100, .30, {}), (90, .25, {'rte': .1}), (110, .29, {}), (120, .35, {}), (80, .40, {'energy': .1})]]
    front = pareto(rows)
    assert [(r['capex_total'], r['RTE']) for r in front] == [(90, .25), (100, .30), (120, .35)]


def test_shade_open_point():
    """Opening placed where the falling flux, extrapolated back, meets the target."""
    from types import SimpleNamespace as NS
    from thermal.sizing import shade_open_point
    ps = NS(width=0.25, height=1.0, f_bare=0.0, f_active=1.0)
    A = 4 * 0.25 * 1.0 * 1e4
    m = NS(powerstick=ps, f_shade_min=0.0, f_shade_max=0.9, avg_flux=10.0)
    t = np.array([0., 1., 2., 3., 4., 5.]) * 3600
    q = np.array([12., 10., 10., 9.5, 9.0, 8.5])          # opens between 2 h and 3 h, at 2 h
    r = dict(time=t, f_shade=np.array([.9, .5, .2, 0., 0., 0.]), ps_active=np.ones(6, bool),
             T_avg=np.array([2500., 2450., 2400., 2350., 2300., 2250.]), Q_cell=q * A)
    pt = shade_open_point(m, r)
    assert pt['seen'] and abs(pt['t_open'] - 2 * 3600) < 1e-6 and abs(pt['T_open'] - 2400.) < 1e-9
    # still regulating when the stick comes out: f linear in T reaches 0 at 2200 K, 6 h
    r['f_shade'] = np.array([.9, .5, .4, .3, .2, .1])
    pt = shade_open_point(m, r)
    assert not pt['seen'] and abs(pt['T_open'] - 2200.) < 1e-6 and abs(pt['t_open'] - 6 * 3600) < 1e-3
    r['f_shade'] = np.full(6, .9)                           # closed to the end
    assert shade_open_point(m, r) is None


def test_cavity_webs():
    """Walls between cavities on the checkerboard, stick squares at 45 deg and aligned."""
    import math
    from thermal.tank import cavity_webs
    r_in, r_out, r_h = 0.661, 0.992, 0.4295
    p, w = cavity_webs(r_in, r_out, r_h, rotated=True)
    S = math.pi * r_in / 2
    assert abs(p - math.sqrt((math.pi * r_out**2 + math.pi * r_h**2) / 2)) < 1e-12
    assert abs(w['stick-heater'] - (p - S / math.sqrt(2) - r_h)) < 1e-12
    assert abs(w['stick-stick'] - (math.sqrt(2) * p - S)) < 1e-12
    _, w0 = cavity_webs(r_in, r_out, r_h, rotated=False)
    assert abs(w0['stick-stick'] - math.sqrt(2) * (p - S)) < 1e-12
    assert w['stick-stick'] > w0['stick-stick'] and w['stick-heater'] < w0['stick-heater']


def test_lifetime_mean_D():
    """Cycle-average deposit over the life: saw-tooth with a partial last interval."""
    from thermal.optimize import lifetime_mean_D
    dc, db = lifetime_mean_D(0.4, 1.0, 0.1)                  # never replaced: mean = half the end
    assert abs(dc - 0.2) < 1e-6 and abs(db - 0.05) < 1e-4
    dc, db = lifetime_mean_D(1.85, 1.0, 0.1)                 # one full interval + 0.85 of the next
    assert abs(dc - (1 * 0.5 + 0.85 * 0.425) / 1.85) < 1e-5
    assert abs(db - 0.05) < 1e-3
    assert lifetime_mean_D(0.0, 1.0, 0.1) == (0.0, 0.0)


def test_cost_scenario_heater_fixed():
    """Scenario heater_fixed per heater (Nate, 2026-11-24): Pilot 7500, FOAK 6000, NOAK 4500 $."""
    from thermal import optimize as O
    for sc, want in (('Pilot', 7500.0), ('FOAK', 6000.0), ('NOAK', 4500.0)):
        assert O.build_opt({'economics': {'cost_scenario': sc}})['costs']['heater_fixed'] == want


def test_cost_scenario_cell_efficiency():
    """Cell efficiency per scenario (Nate, 2026-11-25): Pilot and FOAK the quadratic fit (None), NOAK a flat
    55%; design_cfg carries it to powerstick.cell_efficiency, build_tinless to the simulation, and
    properties.cell_efficiency returns the flat value (or the fit with None)."""
    from thermal import optimize as O
    from thermal.build import load_config, build_tinless
    from thermal.properties import cell_efficiency, panel_efficiency
    base = load_config(os.path.join(ROOT, 'tinless_model_input.yaml'))
    d = O.design_from_cfg(base)
    for sc, want in (('Pilot', None), ('FOAK', None), ('NOAK', 0.55), (None, None)):
        opt = O.build_opt({'economics': {'cost_scenario': sc}})
        assert opt['cells']['efficiency'] == want
        cfg = O.design_cfg(base, d, opt)
        assert cfg['powerstick']['cell_efficiency'] == want
        assert build_tinless(cfg, geometry_only=True).sim.cell_eff_fixed == want
    T = np.array([1900.0, 2300.0])
    assert np.allclose(cell_efficiency(1.24, T, flux=10.0, fixed=0.55), 0.55)
    assert np.allclose(cell_efficiency(1.24, T, flux=10.0), panel_efficiency(1.24, T, flux=10.0))


def test_cost_scenario_layers_and_none_is_unchanged():
    """No scenario: build_opt = OPT_DEFAULTS + inputs, and tank cost inputs equal to the YAML centrals leave the
    tank costs unchanged. A scenario overrides the inputs it lists and is recorded in _sources."""
    from thermal import optimize as O
    from thermal.build import load_config
    base = load_config(os.path.join(ROOT, 'tinless_model_input.yaml'))
    d = O.design_from_cfg(base)
    plain = O.build_opt({})
    assert {k: v for k, v in plain.items() if k != '_sources'} == O.OPT_DEFAULTS
    ref = O.design_cfg(base, d, plain)['tank']['costs']
    yc = base['tank']['costs']
    cell = O.build_opt({'tank_costs': {k: O._central(yc[k]) for k in ('storage_graphite_per_kg', 'steel_per_t',
                                                                       'thermax_per_kg', 'fans_per_m3s')},
                        'costs': {'iso_graphite_per_kg': O._central(yc['column_graphite_per_kg'])}})
    assert O.design_cfg(base, d, cell)['tank']['costs'] == ref
    noak = O.build_opt({'economics': {'cost_scenario': 'NOAK'}, 'costs': {'cell_per_cm2': 99.0, 'shade_motor': 7.0}})
    assert noak['costs']['cell_per_cm2'] == 1.872 and noak['costs']['shade_motor'] == 7.0
    assert noak['_sources']['costs.cell_per_cm2'] == 'scenario NOAK' and noak['_sources']['costs.shade_motor'] == 'inputs'
    tc = O.design_cfg(base, d, noak)['tank']['costs']
    assert O._central(tc['column_graphite_per_kg']) == 2.0 and O._central(tc['storage_graphite_per_kg']) == 0.5
    assert O.apply_scenario(noak)['costs'] == noak['costs']                  # idempotent
    with pytest.raises(ValueError):
        O.build_opt({'economics': {'cost_scenario': 'demo'}})


def test_design_vars_from_inputs():
    """design_vars from the inputs: any order, new bounds; decode / encode follow them."""
    from thermal import optimize as O
    dv = [list(v) for v in reversed(O.DESIGN_VARS)]
    dv[[v[0] for v in dv].index('L')][2] = 4.0
    opt = O.build_opt({'design_vars': dv})
    b = dict(zip([v[0] for v in O.design_vars(opt)], O.var_bounds(opt)))
    assert b['L'] == (3.0, 4.0)
    x = np.array([(lo + hi) / 2 for lo, hi in O.var_bounds(opt)])
    x[O.var_integrality(opt)] = np.round(x[O.var_integrality(opt)])
    d = O.decode(x, opt)
    assert abs(d['L'] - 3.5) < 1e-12
    assert np.allclose(O.encode(d, opt), x)
    with pytest.raises(ValueError):
        O.build_opt({'design_vars': dv[:-1]})


def test_heater_gap_from_voltage():
    """Minimum heater -> cavity gap: arc clearance V / 7.5 V/in when geometry.heater_min_gap is None
    (60 V -> 8 in, 180 V -> 24 in); a number fixes it."""
    from thermal import optimize as O
    opt = O.build_opt({})
    assert opt['geometry']['heater_min_gap'] is None
    for V, inch in ((60.0, 8.0), (180.0, 24.0)):
        assert abs(O.heater_gap(opt, {'heater': {'voltage': V}}) - inch * 0.0254) < 1e-12
    fixed = O.merge(opt, {'geometry': {'heater_min_gap': 0.25}})
    assert O.heater_gap(fixed, {'heater': {'voltage': 180.0}}) == 0.25
    with pytest.raises(ValueError):
        O.heater_gap(opt, {'heater': {}})
