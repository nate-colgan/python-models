"""
Regression cases for the tinless model.

Parameters are hard-coded (NOT read from tinless_model_input.yaml) so that editing
the YAML never moves the baseline. Each case runs a short simulation with the
current per-segment ("local") radiation path, which must stay reproducible once
the decoupled-block-height / axial-radiation code lands:
    block_height == L, radiation.axial == False, liner_bottom.enabled == False.

Baseline created 2026-11-16, after switching spectral integration to
Gauss-Legendre per emissivity band (tinless.py `_spectral_grid`).
"""
import os
import sys
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from thermal.components import GraphiteBlock, Pipe, StorageModule, PowerStick  # noqa: E402
from thermal.tinless import TinlessSimulation, HeatingElement                  # noqa: E402

BASELINE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'baseline_local_radiation.npz')

EMIS = [((0.1, 1.24), 0.9, 0.02, 0.02),
        ((1.24, 20.0), 0.015, 0.02, 0.02)]
# r_out / num_rad chosen so the explicit radial conduction is stable at the
# 240 s sub-step (Fourier number ~0.32 < 0.5).
P = dict(L=3.048, num_ax=4, num_rad=10, side=0.26, area_ratio=5.0, T_ps=323.0,
         f_bare=0.05, f_active=0.9, liner_side=0.5, liner_th=0.005,
         shade_ratio=1 + np.sqrt(2) / 2 + 1e-6, shade_th=0.0391, shade_eps=0.9, corner_gap=0.003175,
         T_min=1873.0, T_range=800.0, r_out=1.55, flux=7.5)


def _build(liner_bool=True, liner_material='graphite', shade_bool=True, T0=None):
    p = P
    r_ps = 2 * p['side'] / np.pi
    r_in = r_ps * p['area_ratio']
    T0 = p['T_min'] + p['T_range'] if T0 is None else T0
    blk = GraphiteBlock(np.full((p['num_ax'], p['num_rad']), float(T0)), r_in, p['r_out'], p['L'], 0.0)
    st = StorageModule(1, Pipe(0.02, p['L'], 0.001, 1e-5, 0.0), blk, p['num_ax'], p['num_rad'], [0, 1])
    ps = PowerStick(p['num_ax'], p['side'], p['L'], p['T_ps'], p['f_bare'], p['f_active'],
                    liner_bool, p['liner_side'] if liner_bool else None, EMIS)
    shade_side = p['shade_ratio'] * p['liner_side']
    per = 4 * shade_side
    return TinlessSimulation(
        st, ps, T_sur=300.0, block_heat_loss=False,
        shade_bool=shade_bool, shade_side_len=shade_side if shade_bool else None,
        shade_thickness=p['shade_th'], shade_surface_emissivity=p['shade_eps'],
        f_shade_min=8 * p['shade_th'] / per, f_shade_max=1 - 8 * p['corner_gap'] / per,
        nominal_cell_flux=p['flux'] if shade_bool else None,
        liner_material=liner_material, liner_thickness=p['liner_th'] if liner_bool else 0.0)


DISCHARGE_KEYS = ('time', 'Q_rad', 'Q_cell', 'P_elec', 'T_avg', 'T_inner', 'T_outer',
                  'T_liner', 'f_shade')
HEATING_KEYS = ('time', 'Q_heater', 'T_heater', 'T_avg', 'T_inner', 'T_outer', 'T_liner')


def _discharge(**kw):
    T0 = kw.pop('T0')
    sim = _build(T0=T0, **kw)
    # Start part-way through discharge so the run crosses the shade-control
    # phase and the shades-fully-open phase.
    r = sim.run_simulation(end_time=10 * 3600, dt_short=60, dt_medium=600, dt_long=1200,
                           sub_steps=5, dt_short_duration=1800,
                           T_ps_cutoff=P['T_min'], stop_at_cutoff=True)
    return {k: np.asarray(r[k], dtype=float) for k in DISCHARGE_KEYS}


def _heating():
    sim = _build(liner_material='graphite', shade_bool=False, T0=P['T_min'])
    heater = HeatingElement(diameter=4 * 0.15 / np.pi, max_temperature=2873.0,
                            max_power=400e3, emissivity=0.85)
    r = sim.run_heating_simulation(heater, T_block_max=P['T_min'] + P['T_range'],
                                   end_time=4 * 3600, dt=300, sub_steps=5)
    return {k: np.asarray(r[k], dtype=float) for k in HEATING_KEYS}


CASES = {
    'shaded_graphite': lambda: _discharge(liner_material='graphite', shade_bool=True, T0=P['T_min'] + 250),
    'shaded_tac':      lambda: _discharge(liner_material='tac', shade_bool=True, T0=P['T_min'] + 250),
    'unshaded_tac':    lambda: _discharge(liner_material='tac', shade_bool=False, T0=P['T_min'] + 150),
    'no_liner':        lambda: _discharge(liner_bool=False, shade_bool=False, T0=P['T_min'] + 150),
    'heating':         _heating,
}


def run_all():
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for name, fn in CASES.items():
            res = fn()
            for k in ('T_avg', 'T_inner', 'T_outer'):
                if not np.all((res[k] > 300.0) & (res[k] < 4000.0)):
                    raise RuntimeError(f"{name}: non-physical {k} (conduction instability?) "
                                       f"range {res[k].min():.0f}..{res[k].max():.0f} K")
            for k, v in res.items():
                out[f'{name}/{k}'] = v
    return out
