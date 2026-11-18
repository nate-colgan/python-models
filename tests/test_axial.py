"""Phase 3 tests: coupled full-height radiation solve (thermal/axial.py).

Uses tinless_model_input.yaml with fixed overrides for geometry/mesh so the
checks do not depend on the user's current sweep values.
Run:  python -m pytest tests/test_axial.py -q
"""
import os
import sys

import numpy as np
import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from thermal.build import build_tinless                                    # noqa: E402
from thermal.axial import Kirchhoff, layered_conduction_flux, SIGMA       # noqa: E402
from thermal.properties import ins_tcx                                    # noqa: E402

from thermal.build import load_config  # noqa: E402
CFG = load_config(os.path.join(ROOT, 'tinless_model_input.yaml'))
FIX = {'storage.L': 3.0, 'radiation.vf_rays': 65536, 'liner_bottom.enabled': False,
       'shade.shade_bool': True, 'liner_bool': True, 'powerstick.tungsten_foil': False,
       'shade.outer_foil': False, 'shade.model': 'porous'}


def _build(H=None, nax=10, shaded=True, axial=True):
    ov = dict(FIX, **{'storage.block_height': H, 'storage.num_ax_segments': nax,
                      'radiation.axial': axial, 'shade.shade_bool': shaded})
    if not shaded:
        # same cavity as the shaded build (shaded cavities are derived from the liner)
        ov['powerstick.area_ratio'] = build_tinless(CFG, dict(ov, **{'shade.shade_bool': True}),
                                                    geometry_only=True).area_ratio
    return build_tinless(CFG, ov)


@pytest.fixture(scope='module')
def tall():
    return _build(H=6.0, nax=20)


@pytest.fixture(scope='module')
def tall_unshaded():
    return _build(H=6.0, nax=20, shaded=False)


def _profile(m, T_top=2000.0, grad=-30.0):
    e = m.mesh.edges
    z = 0.5 * (e[1:] + e[:-1])
    return T_top + grad * (z - e[-1])          # hotter at the bottom when grad < 0


def test_energy_balance_and_shade_rings(tall):
    ax = tall.sim.axial_model
    for f in (0.0, 0.3, 0.9, 1.0):
        s = ax.solve(_profile(tall), f=f)
        assert s['balance_err'] < 1e-6, (f, s['balance_err'])
        # every shade ring passes what it receives: net into both slat faces = 0
        J, G, A = s['J'], s['G'], ax.area(f)
        net = A[ax.iSo] * (G[ax.iSo] - J[ax.iSo]) + A[ax.iSi] * (G[ax.iSi] - J[ax.iSi])
        assert np.max(np.abs(net)) < 1e-6 * s['Q_block'].sum()
        for nm, T in s['T_rerad'].items():
            assert 300 < T < 3000, (nm, T)


def test_tau1_matches_unshaded_single_enclosure(tall, tall_unshaded):
    """§6.4: shade fully open (f = 0, every ring) = unshaded enclosure (straight-
    through transmission; differences are ray-tracing noise only)."""
    T = _profile(tall)
    a = tall.sim.axial_model.solve(T, f=0.0)
    b = tall_unshaded.sim.axial_model.solve(T)
    assert abs(a['Q_cell'].sum() / b['Q_cell'].sum() - 1) < 0.01


def test_tau1_ring_distribution_matches_unshaded(tall, tall_unshaded):
    """Long-range axial transport survives an open shade (it did not with the
    brief's diffuse-transmission model: lower-ring flux was ~27% low)."""
    T = _profile(tall)
    a = tall.sim.axial_model.solve(T, f=0.0)
    b = tall_unshaded.sim.axial_model.solve(T)
    assert np.max(np.abs(a['Q_block'] - b['Q_block'])) < 0.03 * np.abs(b['Q_block']).max()


def test_tau0_only_conduction(tall):
    """§6.4: f = 1 -> nothing transmitted; liner input equals shade conduction."""
    s = tall.sim.axial_model.solve(_profile(tall), f=1.0)
    assert abs(s['Q_through_openings']) < 1e-9 * s['Q_ps'].sum()
    assert abs(s['Q_shade_cond'].sum() / s['Q_ps'].sum() - 1) < 1e-6


def test_cell_flux_control(tall):
    ax = tall.sim.axial_model
    ax._x = ax._f = None
    Q_tgt = tall.sim.nominal_cell_flux * ax.A_cell_cm2
    for T_top in (2300.0, 2100.0, 1900.0, 1700.0):
        s = ax.solve(_profile(tall, T_top), control=True)
        f_lo, f_hi = tall.sim.f_shade_min, tall.sim.f_shade_max
        Qc = s['Q_cell'].sum()
        if f_lo < s['f'] < f_hi:
            assert abs(Qc / Q_tgt - 1) < 1e-6
        elif s['f'] == f_lo:
            assert Qc <= Q_tgt * (1 + 1e-9)
        else:
            assert Qc >= Q_tgt * (1 - 1e-9)


def test_axial_H_equals_L_close_to_local_model():
    """§6.2 (single step): axial-on at block_height == L vs the local model,
    unshaded (no shade-transmission modelling difference): ends only, < 1%."""
    leg = _build(H=None, nax=10, shaded=False, axial=False)
    ax = _build(H=None, nax=10, shaded=False, axial=True)
    T = np.full_like(leg.storage.block.temp_dist, 2000.0)
    ql = leg.sim._step_calc(T, 0.0)
    _, qa = ax.sim._radiation_step(T, False)
    assert abs(qa[2].sum() / ql[2].sum() - 1) < 0.01
    assert abs(qa[1].sum() / ql[1].sum() - 1) < 0.01


def test_mesh_independent_for_uniform_wall():
    a = _build(H=None, nax=10).sim.axial_model.solve(np.full(10, 2000.0), f=0.5)
    b = _build(H=None, nax=30).sim.axial_model.solve(np.full(30, 2000.0), f=0.5)
    assert abs(a['Q_cell'].sum() / b['Q_cell'].sum() - 1) < 0.01      # MC noise, 2% target


def test_vertical_transport_direction(tall):
    ax = tall.sim.axial_model
    hot_bottom = ax.solve(_profile(tall, 1900.0, -60.0), f=0.9)
    vt = ax.vertical_transport(hot_bottom)
    assert vt['outer'] > 0                  # block-shade gap carries heat upward


def test_kirchhoff_exact_and_layered_limit():
    K = Kirchhoff()
    t = 0.04
    for To, Ti in ((2300.0, 1900.0), (1500.0, 800.0)):
        ex = (K.I_of(To) - K.I_of(Ti)) / t
        assert abs(layered_conduction_flux(To, Ti, t, 40) / ex - 1) < 2e-3
    # k(T) is linear above 1000 C -> one mean-k layer is exact there
    To, Ti = 2400.0, 1700.0
    one = ins_tcx(0.5 * (To + Ti)) * (To - Ti) / t
    assert abs(one / ((K.I_of(To) - K.I_of(Ti)) / t) - 1) < 1e-4


def test_transient_stable_tall_block():
    """Explicit wall flux is unstable with ring-to-ring exchange; the implicit
    (linearised) wall coupling in update_temps must keep a tall-block run smooth."""
    m = _build(H=6.0, nax=20)
    kw = dict(m.run_kwargs, end_time=6 * 3600, stop_at_cutoff=True)
    r = m.sim.run_simulation(**kw)
    T = m.storage.block.temp_dist
    assert np.all(np.isfinite(T)) and T.min() > 1500 and T.max() <= m.T_max + 1
    assert np.all(np.diff(r['T_avg']) <= 1e-6)                # monotone discharge
    Tw = T[:, 0]
    assert np.max(np.abs(np.diff(Tw, 2))) < 20.0               # no ring-to-ring checkerboard


# --------------------------------------------------------------------------
# Rotating-vane shade: closed/open limits blended to hold the setpoint
# --------------------------------------------------------------------------
@pytest.fixture(scope='module')
def vanes():
    ov = dict(FIX, **{'storage.block_height': 6.0, 'storage.num_ax_segments': 10,
                      'radiation.axial': True, 'shade.model': 'vanes', 'radiation.vane_rays': 2 ** 12})
    return build_tinless(CFG, ov)


def test_vane_limits_and_setpoint(vanes):
    ax = vanes.sim.axial_model
    Q_tgt = vanes.sim.nominal_cell_flux * ax.A_cell_cm2
    for Tw in (2600.0, 2100.0, 1900.0, 1700.0):
        s = ax.solve(np.full(ax.n, Tw), control=True)
        assert s['Q_cell_min'] < s['Q_cell_max']
        assert s['balance_err'] < 1e-6
        Qc = s['Q_cell'].sum()
        if s['Q_cell_min'] < Q_tgt < s['Q_cell_max']:
            assert abs(Qc / Q_tgt - 1) < 1e-6 and 0 < s['w'] < 1
        elif Q_tgt <= s['Q_cell_min']:
            assert s['w'] == 0.0 and abs(Qc - s['Q_cell_min']) < 1e-6 * Qc
        else:
            assert s['w'] == 1.0 and abs(Qc - s['Q_cell_max']) < 1e-6 * Qc


def test_vane_transient_stable(vanes):
    m = vanes
    r = m.sim.run_simulation(**dict(m.run_kwargs, end_time=4 * 3600, stop_at_cutoff=True))
    T = m.storage.block.temp_dist
    assert np.all(np.isfinite(T)) and T.min() > 1500 and T.max() <= m.T_max + 1
    assert np.all(np.diff(r['T_avg']) <= 1e-6)
    assert np.all(r['Q_cell_min'][np.isfinite(r['Q_cell_min'])] <= r['Q_cell'][np.isfinite(r['Q_cell_min'])] * (1 + 1e-9))


# --------------------------------------------------------------------------
# Phase 4: liner bottom + RGF pad
# --------------------------------------------------------------------------
def _pad_model(t_rgf, model='porous'):
    ov = dict(FIX, **{'storage.block_height': 6.0, 'storage.num_ax_segments': 10, 'radiation.axial': True,
                      'shade.model': model, 'liner_bottom.enabled': True, 'liner_bottom.rgf_thickness': t_rgf,
                      'radiation.vane_rays': 2 ** 12})
    return build_tinless(CFG, ov)


def test_pad_limits_and_balance():
    """§6.4: very thick pad -> Q_pad -> 0; pad -> 0 -> T_LB ~ T_R; energy closes with the pad."""
    out = {}
    for t in (1e-4, 0.025, 2.0):
        ax = _pad_model(t).sim.axial_model
        s = ax.solve(np.full(ax.n, 2000.0), f=0.95)
        assert s['balance_err'] < 1e-6
        assert s['Q_cap'] > 0 and s['q_cap'] > 0
        out[t] = s
    assert abs(out[2.0]['Q_pad']) < 0.02 * abs(out[1e-4]['Q_pad'])
    assert abs(out[1e-4]['T_R'] - out[1e-4]['T_LB']) < 15.0          # only the 5 mm graphite plate left
    assert out[1e-4]['Q_pad'] > out[0.025]['Q_pad'] > out[2.0]['Q_pad']


def test_pad_with_vanes_and_transient():
    m = _pad_model(0.025, 'vanes')
    ax = m.sim.axial_model
    s = ax.solve(np.full(ax.n, 2300.0), control=True)
    assert s['balance_err'] < 1e-6 and np.isfinite(s['T_LB'])
    r = m.sim.run_simulation(**dict(m.run_kwargs, end_time=3 * 3600, stop_at_cutoff=True))
    assert np.all(np.isfinite(r['T_LB'])) and np.all(r['q_cap'] > 0)
    assert np.all(np.diff(r['T_avg']) <= 1e-6)
