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
    assert abs(hm.r_in - 4.0 * 2 * cfg['heater']['side_len'] / math.pi) < 1e-12
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
