"""System design optimizer for the tinless plant.

One design = one call of `evaluate(x, base_cfg, opt)`:

  1. heater cavity   smallest heater.area_ratio that charges the heater cell within the
                     design charge time (build_heater, roof + floor losses on)
  2. discharge       stick run with clean cells (roof + floor losses on, deposition on),
                     block sized by shade-open sizing (thermal/sizing.py): T_min = the
                     block average at which the shades are fully open, block volume so that
                     happens at the target duration (T_max is the design variable).
                     Deposition happens only while the stick is in: during charge and hold
                     the stick is retracted and a door closes the hole (door loss not modelled).
  3. tank            sized on the clean stick, AC basis: N_ps = ceil(P_rated / P_stick_AC),
                     energy target enforced on clean performance
  4. degradation     stick run at the end of a replacement interval: cells replaced at
                     D_cell = cell_replace_D, bare areas cleaned (reset) at D_bare =
                     bare_clean_D. Performance = average of the clean and end runs
                     (linear in D -> exact time average of the saw-tooth).
  5. metrics         lifetime capex (initial + cell replacements + bare cleanings, all
                     x (1 + soft costs)), net round-trip efficiency = (AC out - floor and
                     heat-rejection fans) / AC in (incl. hold losses), and the lifetime cost
                     of the energy lost = price x (AC in - net AC out) x cycles

Optimized variables (DESIGN_VARS) are reparameterised so every point in the box is a
valid geometry: powerstick side as a fraction of what fits in the liner, block height as
a fraction between L and H_max. T_min and the swing are outputs of the sizing; T_min >=
T_min_floor_C and swing_min <= swing <= swing_max are constraints.

Objective: minimise lifetime cost = lifetime capex + lifetime cost of the energy lost,
with one electricity price for charge and discharge (so efficiency is weighted in $).
Optional floor RTE >= objective.rte_min. `differential_evolution` minimises
f = lifetime_cost / 1e6 for feasible designs and f = INFEASIBLE + 1e3 x (sum of
normalised constraint violations) otherwise.
"""
import contextlib
import copy
import dataclasses
import io
import json
import math
import os
import time
import traceback
from types import SimpleNamespace

import numpy as np

from .build import build_tinless, build_heater, heater_max_power
from .sizing import run_discharge, shade_open_resize, FluxUnreachable
from .properties import storage_rho, storage_cp
from . import tank as tk

C2K = 273.15
INFEASIBLE = 1e4
FAILED = 1e5
INCH = 0.0254

# --------------------------------------------------------------------------
# Inputs that are not optimized (economics, costs, fidelity). Physics inputs come
# from the base YAML (tinless_model_input.yaml).
# --------------------------------------------------------------------------
OPT_DEFAULTS = {
    'P_rated': 25.0e6,            # W AC export
    'duration_hr': 50.0,          # h at P_rated (energy target, enforced)
    'charge_hr': 30.0,            # h design charge time (heater cavity sized to meet it)
    'economics': {
        'eta_out': 0.98,          # DC -> AC inverter
        'eta_in': 0.98,           # AC -> heater supply
        'cycles_per_year': 65.7,  # incl. hold at temperature (hold losses charged)
        'lifetime_yr': 30.0,
        'soft_cost_frac': 0.30,   # tax, freight, soft costs on all capex (incl. replacements)
        'elec_price': 75.0,       # $/MWh, same for charge and discharge: prices the energy lost
    },
    'objective': {'rte_min': 0.0},     # optional floor on net RTE (0 = none)
    'costs': {
        'cell_per_cm2': 1.0,      # $/cm2 of cell-covered powerstick face (incl. at replacement)
        'stick_per_m': 100.0,     # $/m of stick length
        'stick_fixed': 2000.0,    # $/stick
        'heater_fixed': 5000.0,   # $/heater element (PLACEHOLDER)
        'charge_electrics_per_kW': 75.0,   # $/kW heater supply (PLACEHOLDER)
        'inverter_per_kW': 100.0,          # $/kW_AC peak (NREL ~$75/kW ex-factory + install)
        'heat_rejection_per_kWth': 125.0,  # $/kW_th dry cooler (IEA SHC Task 48: $61-134, 2014)
        'bare_clean_per_stick': 0.0,       # $/stick per bare-area cleaning (tracked; cost TBD)
    },
    'parasitics': {
        # dry cooler fans + coolant pumps, kW_e per kW_th rejected (DIN V 18599 seasonal
        # default for dry heat rejection 0.045; energy class E ~0.033). Floor-cooling fans
        # come from the tank model (pressure drop x flow / fan efficiency), run all cycle.
        'heat_rejection_kWe_per_kWth': 0.03,
    },
    'deposition': {
        'd_dead_nm': 10.0,        # linear: D = deposit / d_dead (cells dead at 10 nm)
        'emis_end': 0.8,          # bare / inactive emissivity at D = 1
        'cell_replace_D': 0.1,    # replace cells at this D
        'bare_clean_D': 0.01,     # clean bare areas (reset their emissivity) at this D
    },
    'flux_band_tol': 0.005,       # |flux / target - 1| within this counts as "at target"
    # shade-open sizing: converged when the shades open within tol_T [K] of T_min and
    # tol_t (fraction) of the duration; first guess T_min = T_max - swing_guess. The energy
    # target counts as met within tol_t.
    'sizing': {'tol_T': 3.0, 'tol_t': 0.01, 'max_iter': 6, 'swing_guess': 500.0},
    'geometry': {
        'ps_liner_gap': 0.02,     # m min clearance powerstick -> liner, each side
        'heater_min_gap': 0.25,   # m min radial clearance heater -> cavity wall
        'H_max': 12.0,            # m block height upper bound
        'T_cap_C': 2500.0,        # C block T_max cap (heater max 2600 C)
        'T_min_floor_C': 1400.0,  # C lowest allowed T_min (constraint; T_min is a sizing output)
        'swing_min': 200.0, 'swing_max': 1000.0,   # C allowed swing (constraint)
    },
    'limits': {'D_tank_max': 80.0, 'H_tank_max': 30.0},
    'fidelity': {'dz': 0.5, 'nrad': 5, 'dt_scale': 2.0, 'vane_rays': 2 ** 12, 'heater_ar_tol': 0.03},
}

# name, lower, upper, integer, description
DESIGN_VARS = [
    ('liner_side', 0.20, 0.60, False, 'liner side length [m]'),
    ('ps_frac', 0.0, 1.0, False, 'powerstick side: 0.10 m + frac x (room in liner - 0.10)'),
    ('L', 1.0, 5.0, False, 'stick length [m]'),
    ('H_frac', 0.0, 1.0, False, 'block height: L + frac x (H_max - L)'),
    ('flux_idx', 0, 6, True, 'cell flux target 5 + 2.5 i [W/cm2]'),
    ('shade_qin', 1, 8, True, 'shade thickness i x 1/4 in'),
    ('t_side', 0.5, 2.0, False, 'tank side insulation [m]'),
    ('t_top', 0.5, 2.0, False, 'tank top insulation [m]'),
    ('T_max_C', 1600.0, 2500.0, False, 'block T_max [C] (capped at geometry.T_cap_C); T_min from shade-open sizing'),
]
VAR_NAMES = [v[0] for v in DESIGN_VARS]
BOUNDS = [(v[1], v[2]) for v in DESIGN_VARS]
INTEGRALITY = np.array([v[3] for v in DESIGN_VARS])


def merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = v
    return out


# --------------------------------------------------------------------------
# Design vector <-> physical design
# --------------------------------------------------------------------------
def decode(x, opt):
    """x (DESIGN_VARS order) -> physical design dict."""
    g = opt['geometry']
    v = dict(zip(VAR_NAMES, np.asarray(x, float)))
    liner = v['liner_side']
    ps_hi = min(0.40, liner - 2 * g['ps_liner_gap'])
    ps = 0.10 + v['ps_frac'] * max(ps_hi - 0.10, 0.0)
    L = v['L']
    H = L + v['H_frac'] * max(g['H_max'] - L, 0.0)
    if H - L < 0.25:                       # a lower zone needs room for the liner bottom
        H = L
    flux = 5.0 + 2.5 * int(round(v['flux_idx']))
    shade_t = int(round(v['shade_qin'])) * 0.25 * INCH
    return dict(liner_side=liner, ps_side=ps, L=L, H_block=H, avg_flux=flux, shade_thickness=shade_t,
                t_side=v['t_side'], t_top=v['t_top'], T_max_C=min(v['T_max_C'], g['T_cap_C']))


def encode(design, opt):
    """Physical design dict -> x (inverse of decode, clipped to the bounds)."""
    g = opt['geometry']
    liner, L = design['liner_side'], design['L']
    ps_hi = min(0.40, liner - 2 * g['ps_liner_gap'])
    x = [liner, (design['ps_side'] - 0.10) / max(ps_hi - 0.10, 1e-9), L,
         (design['H_block'] - L) / max(g['H_max'] - L, 1e-9), round((design['avg_flux'] - 5.0) / 2.5),
         round(design['shade_thickness'] / (0.25 * INCH)), design['t_side'], design['t_top'], design['T_max_C']]
    return np.clip(np.array(x, float), [b[0] for b in BOUNDS], [b[1] for b in BOUNDS])


def design_from_cfg(cfg):
    """The design currently in a model YAML (for a baseline / starting point)."""
    s, p, sh, t = cfg['storage'], cfg['powerstick'], cfg.get('shade', {}), cfg.get('tank', {})
    L = float(s['L'])
    return dict(liner_side=p['liner_side_len'], ps_side=p['side_len'], L=L,
                H_block=float(s.get('block_height') or L), avg_flux=float(s['avg_flux']),
                shade_thickness=float(sh.get('thickness', 0.0254)), t_side=float(t.get('t_side', 1.0)),
                t_top=float(t.get('t_top', 1.0)),
                T_max_C=float(s['T_min']) - C2K + float(s['block_temp_range']),
                T_min_guess_C=float(s['T_min']) - C2K)


def design_cfg(base_cfg, d, opt):
    """Model config for a physical design (inputs from base_cfg and opt)."""
    f = opt['fidelity']
    cfg = copy.deepcopy(base_cfg)
    cfg['end_time'] = max(float(cfg['end_time']), 4 * opt['duration_hr'] * 3600)   # run to the T_min cutoff
    s, p = cfg['storage'], cfg['powerstick']
    s['L'], s['block_height'] = d['L'], d['H_block']
    if cfg['heater'].get('max_power_per_m') is None:      # cap = max_power_per_m x block height
        raise KeyError("YAML heater.max_power_per_m is missing: run the YAML cell of tinless system model.ipynb")
    cfg['heater'].pop('max_power', None)
    s['avg_flux'], s['duration_hr'] = d['avg_flux'], opt['duration_hr']
    g = opt['geometry']
    T_lo = d.get('T_min_guess_C', d['T_max_C'] - opt['sizing']['swing_guess'])   # sizing start
    T_lo = min(max(T_lo, d['T_max_C'] - g['swing_max']), d['T_max_C'] - g['swing_min'])
    s['T_min'], s['block_temp_range'] = T_lo + C2K, d['T_max_C'] - T_lo
    s['volume_factor'] = 1.0
    s['num_ax_segments'] = max(10, int(round(d['H_block'] / f['dz'])))
    s['num_rad_segments'] = f['nrad']
    p['liner_side_len'], p['side_len'] = d['liner_side'], d['ps_side']
    cfg.setdefault('shade', {})['thickness'] = d['shade_thickness']
    rad = cfg.setdefault('radiation', {})
    rad['vane_rays'] = f['vane_rays']
    rad['vf_cache_dir'] = None                 # every design is a new geometry
    t = cfg.setdefault('tank', {})
    t['t_side'], t['t_top'] = d['t_side'], d['t_top']
    t['P_rated'], t['duration_hr'] = opt['P_rated'], opt['duration_hr']
    t['block_axial_losses'] = True
    return cfg


# --------------------------------------------------------------------------
# Pipeline pieces
# --------------------------------------------------------------------------
def _with_heater_ar(m, cfg, ar, P=None):
    c = copy.deepcopy(cfg)
    c['heater']['area_ratio'] = float(ar)
    if P is not None:
        c['heater']['max_power'] = float(P)
    m2 = SimpleNamespace(**vars(m))
    m2.cfg = c
    return m2, c


def _charge(m, cfg, ar, end_hr, dt_scale, P=None):
    m2, c = _with_heater_ar(m, cfg, ar, P)
    hm = build_heater(m2, area_ratio=ar)
    sink = tk.cell_loss_sink(m2, c['tank']).for_storage(hm.storage)
    kw = dict(hm.run_kwargs, dt=hm.run_kwargs['dt'] * dt_scale)
    r = hm.sim.run_heating_simulation(hm.heater, end_time=end_hr * 3600, Q_sink_fn=sink, **kw)
    Ta, t = r['T_avg'], r['time']
    done = Ta[-1] >= hm.T_max
    t_c = float(np.interp(hm.T_max, Ta[-2:], t[-2:])) / 3600 if done else np.inf
    dt = np.diff(t)
    return dict(ar=float(ar), t_charge=t_c, E_heater=float(np.sum(r['Q_heater'][1:] * dt)),
                E_sink=float(np.sum(r['Q_sink'][1:] * dt)), T_heater_max=float(r['T_heater'].max()),
                r_cav=hm.r_in, P_heater=hm.heater.max_power)


def heater_search(m, cfg, opt):
    """Heater power and cavity for t_charge <= charge_hr. The power cap is YAML
    heater.max_power_per_m x heater length (= block height).
      - cavity at its minimum (heater_min_gap) meets the target at the cap: the heater
        power is cut to the smallest that still meets it (electrics costed on that);
      - otherwise the cavity grows at the cap: smallest area_ratio that meets it.
    -> dict(ok, violation, ar, t_charge, P_heater, ...)."""
    h = cfg['heater']
    target = opt['charge_hr']
    dts = opt['fidelity']['dt_scale']
    tol = opt['fidelity']['heater_ar_tol']
    r_h = 2 * h['side_len'] / math.pi
    if h['max_temperature'] <= m.T_max + 1.0:
        return dict(ok=False, violation=(m.T_max + 1.0 - h['max_temperature']) / 100.0,
                    reason='heater max temperature <= block T_max')
    st = m.storage
    V = math.pi * (st.block.rad_out ** 2 - st.block.rad_in ** 2) * m.H_block
    E_blk = storage_rho(m.T_max) * storage_cp(m.T_max) * V * (m.T_max - m.T_min)
    P_cap = heater_max_power(h, m.H_block)
    t_floor = E_blk / P_cap / 3600
    if t_floor > target:
        return dict(ok=False, violation=t_floor / target - 1, reason='charge power-limited above target')
    end_hr = 1.5 * target
    ar_lo = (r_h + opt['geometry']['heater_min_gap']) / r_h
    c_lo = _charge(m, cfg, ar_lo, end_hr, dts)
    if c_lo['t_charge'] <= target:
        # minimum cavity is enough at the cap: smallest heater power that still meets the target
        P_lo, P_hi, c_hi = E_blk / (target * 3600), P_cap, c_lo
        while P_hi / P_lo > 1 + tol:
            P_mid = math.sqrt(P_lo * P_hi)
            c_mid = _charge(m, cfg, ar_lo, end_hr, dts, P=P_mid)
            if c_mid['t_charge'] <= target:
                P_hi, c_hi = P_mid, c_mid
            else:
                P_lo = P_mid
        return dict(ok=True, violation=0.0, **c_hi)
    ar_hi, c_hi = ar_lo, c_lo
    for _ in range(8):
        ar_hi *= 1.6
        c_hi = _charge(m, cfg, ar_hi, end_hr, dts)
        if c_hi['t_charge'] <= target:
            break
    else:
        tc = c_hi['t_charge']
        return dict(ok=False, violation=(tc / target - 1) if np.isfinite(tc) else 1.0,
                    reason='charge time not met at any cavity size')
    while ar_hi / ar_lo > 1 + tol:
        ar_mid = math.sqrt(ar_lo * ar_hi)
        c_mid = _charge(m, cfg, ar_mid, end_hr, dts)
        if c_mid['t_charge'] <= target:
            ar_hi, c_hi = ar_mid, c_mid
        else:
            ar_lo = ar_mid
    return dict(ok=True, violation=0.0, **c_hi)


def _run_kw(cfg, opt):
    """Screening run settings for sizing.run_discharge."""
    dts = opt['fidelity']['dt_scale']
    return dict(dt_scale=dts, dt_at_shade_limit=float(cfg['dt_long']) * dts, stop_at_cutoff=True)


def _discharge(cfg, opt, degradation=None, deposition=True):
    c = copy.deepcopy(cfg)
    if degradation:
        c['degradation'] = degradation
    m, r = run_discharge(c, deposition=deposition, **_run_kw(c, opt))
    return m, r, tk.stick_from_run(m, r), discharge_metrics(m, r, opt)


def discharge_metrics(m, r, opt):
    """Heat rejection, peak power and cell-flux band statistics over the stick-in period
    (time-weighted). Flux = Q_cell / active cell area, as the shade controller uses."""
    t = np.asarray(r['time'], float)
    on = np.asarray(r['ps_active'][1:], bool)
    w = np.diff(t) * on
    T_on = float(w.sum())
    P_el = np.asarray(r['P_elec'][1:], float)
    Q_rej = np.asarray(r['Q_rad'][1:], float) - P_el      # heat into the stick not converted
    ps = m.powerstick
    A_act = 4 * ps.width * ps.height * (1 - ps.f_bare) * ps.f_active * 1e4
    q = np.asarray(r['Q_cell'][1:], float) / A_act
    q_t = float(m.avg_flux)
    E_rad = float((w * np.asarray(r['Q_rad'][1:], float)).sum())       # heat into the stick
    E_sink = float((w * np.asarray(r['Q_sink'][1:], float)).sum())     # roof + floor losses, stick in
    if T_on <= 0:
        return dict(Q_rej_peak=0.0, E_rej=0.0, P_elec_peak=0.0, t_on_hr=0.0, E_rad=0.0, E_sink=0.0,
                    E_released=0.0)
    tol = opt.get('flux_band_tol', 0.005)
    above, below = q > q_t * (1 + tol), q < q_t * (1 - tol)
    band = ~above & ~below
    E_el = float((w * P_el).sum())
    mean = float((w * q).sum() / T_on)
    return dict(
        Q_rej_peak=float(Q_rej[on].max()), E_rej=float((w * Q_rej).sum()), P_elec_peak=float(P_el[on].max()),
        t_on_hr=T_on / 3600, flux_target=q_t, E_rad=E_rad, E_sink=E_sink, E_released=E_rad + E_sink,
        frac_above=float((w * above).sum() / T_on), frac_band=float((w * band).sum() / T_on),
        frac_below=float((w * below).sum() / T_on),
        E_frac_above=float((w * P_el * above).sum() / E_el), E_frac_below=float((w * P_el * below).sum() / E_el),
        flux_mean=mean, flux_std=float(np.sqrt((w * (q - mean) ** 2).sum() / T_on)),
        flux_rms_dev=float(np.sqrt((w * (q - q_t) ** 2).sum() / T_on)),   # about the target
        flux_max=float(q[on].max()), flux_end=float(q[on][-1]), P_elec_end=float(P_el[on][-1]),
    )


def _n_events(total, step):
    """Resets strictly before end of life when a quantity grows by `total` and is reset
    every `step`."""
    if total <= 0 or not np.isfinite(step) or step <= 0:
        return 0
    return max(int(math.ceil(total / step - 1e-9)) - 1, 0)


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def evaluate_design(d, base_cfg, opt):
    """Full evaluation of a physical design -> metrics dict (capex_total, RTE, violations, ...)."""
    eco, cost, dep, par = opt['economics'], opt['costs'], opt['deposition'], opt['parasitics']
    cfg = design_cfg(base_cfg, d, opt)
    out = dict(design=d)
    viol = {}
    g, sz = opt['geometry'], opt['sizing']
    # 1. shade-open sizing on the clean discharge: T_min and block volume (deposition per
    #    cycle from the final run; the stick is in only during discharge)
    try:
        so = shade_open_resize(cfg, tol_T=sz['tol_T'], tol_t=sz['tol_t'], max_iter=sz['max_iter'],
                               T_min_floor=g['T_min_floor_C'] + C2K, verbose=False, **_run_kw(cfg, opt))
    except FluxUnreachable:
        out.update(violations={'flux': 1.0}, feasible=False,
                   heater={'reason': 'target flux not reached even with the shades fully open'})
        return out
    cfg = so.cfg
    T_min_C = so.T_min - C2K
    out.update(T_min_C=T_min_C, swing=so.swing, volume_factor=so.volume_factor, sizing_runs=len(so.history),
               sizing_converged=so.converged, sizing_history=so.history)
    if so.below_floor:
        T_last = np.nanmin([so.T_open, so.T_min - 100.0]) - C2K
        viol['T_min'] = max(g['T_min_floor_C'] - T_last, 1.0) / 100.0
    if so.swing < g['swing_min']:
        viol['swing'] = (g['swing_min'] - so.swing) / 100.0
    if so.swing > g['swing_max']:
        viol['swing'] = (so.swing - g['swing_max']) / 100.0
    if viol:
        out.update(violations=viol, feasible=False, heater={'reason': 'sizing outside the T_min / swing limits'})
        return out
    m, r_f = so.m, so.r
    st_f, mf = tk.stick_from_run(m, r_f), discharge_metrics(m, r_f, opt)
    # 2. heater cavity for the final block
    m0 = build_tinless(cfg, geometry_only=True)
    hs = heater_search(m0, cfg, opt)
    out['heater'] = {k: v for k, v in hs.items()}
    if not hs['ok']:
        viol['charge'] = hs['violation']
        out.update(violations=viol, feasible=False)
        return out
    cfg['heater']['area_ratio'] = hs['ar']
    dep_nm = float(r_f.get('dep_total_nm', 0.0))
    cyc_life = eco['cycles_per_year'] * eco['lifetime_yr']
    dep_life = dep_nm * cyc_life                               # nm on one stick slot over the life
    d_rep = dep['cell_replace_D'] * dep['d_dead_nm']
    d_cln = dep['bare_clean_D'] * dep['d_dead_nm']
    n_rep = _n_events(dep_life, d_rep)
    n_reset = _n_events(dep_life, d_cln)                       # bare resets, incl. at replacements
    n_clean = sum(1 for k in range(1, n_reset + 1)
                  if abs(k * d_cln / d_rep - round(k * d_cln / d_rep)) > 1e-6)   # cleanings alone
    D_c = min(d_rep, dep_life) / dep['d_dead_nm']              # end-of-interval states
    D_b = min(d_cln, dep_life) / dep['d_dead_nm']
    # 3. tank: pass 1 (clean) for the side + edge share, then sized on the clean AC stick
    t1 = tk.tank_system(st_f, cfg['tank'])
    pf = t1.stick_correction['penalty_frac']
    eo = eco['eta_out']
    E_f_ac = st_f.E_stick_e * (1 - pf) * eo
    st_size = dataclasses.replace(st_f, P_stick=E_f_ac / st_f.t_discharge, E_stick_e=E_f_ac)
    tank = tk.tank_system(st_size, cfg['tank'])
    hz, geo, c = tank.hz, tank.geo, tank.cfg
    N = hz.N_ps
    # 4. end of a replacement interval (no new deposition in this run)
    if max(D_c, D_b) > 1e-4:
        _, r_d, st_d, md = _discharge(cfg, opt, degradation=dict(cell=D_c, bare=D_b, emis_end=dep['emis_end']),
                                      deposition=False)
    else:
        r_d, st_d, md = r_f, st_f, mf
    E_d_ac = st_d.E_stick_e * (1 - pf) * eo
    # constraints
    if hz.energy_mismatch < -sz['tol_t']:                      # within the sizing tolerance
        viol['energy'] = -hz.energy_mismatch
    lim = opt['limits']
    if geo.D_tank > lim['D_tank_max']:
        viol['D_tank'] = geo.D_tank / lim['D_tank_max'] - 1
    if geo.H_tank > lim['H_tank_max']:
        viol['H_tank'] = geo.H_tank / lim['H_tank_max'] - 1
    fl = tank.floor_design
    T_lim = c['floor']['T_limit_annexM']
    if max(fl['T_floor_max'], fl['T_floor_col']) > T_lim:
        viol['floor_T'] = max(fl['T_floor_max'], fl['T_floor_col']) / T_lim - 1
    if tank.foundation_pressure > c['soil_allowable']:
        viol['foundation'] = tank.foundation_pressure / c['soil_allowable'] - 1
    cycle_hr = 8760.0 / eco['cycles_per_year']
    t_dis_f, t_dis_d = st_f.t_discharge / 3600, st_d.t_discharge / 3600
    t_dis = 0.5 * (t_dis_f + t_dis_d)
    t_hold = cycle_hr - hs['t_charge'] - t_dis
    if cycle_hr - hs['t_charge'] - max(t_dis_f, t_dis_d) < 0:
        viol['cycle_time'] = (hs['t_charge'] + max(t_dis_f, t_dis_d)) / cycle_hr - 1
    # 5. energy per cycle [J], interval-average cells
    T_max_C = st_f.T_max
    T_mid = 0.5 * (st_f.T_min + st_f.T_max)
    L_mid = tk.losses(T_mid, T_mid, T_mid, geo, hz, tank.cols, c)
    # steady cycle: the heaters put back the heat the discharge released (the block is not
    # emptied to a uniform T_min, so the uniform-start charge run would over-count it) plus
    # the roof + floor losses during the charge; side + edge losses on top
    E_released = 0.5 * (mf['E_released'] + md['E_released'])
    E_charge_heat = (N * (E_released + hs['E_sink'])
                     + (L_mid['Q_side'] + L_mid['Q_edge']) * hs['t_charge'] * 3600)
    E_hold_heat = tank.loss_fn(T_max_C) * max(t_hold, 0.0) * 3600       # stick out, door closed
    E_in = (E_charge_heat + E_hold_heat) / eco['eta_in']
    E_out = N * 0.5 * (E_f_ac + E_d_ac)
    P_floor_fan = tank.loss_peak['P_fan']
    E_floor_fan = P_floor_fan * cycle_hr * 3600                          # runs all cycle
    E_rej_fan = par['heat_rejection_kWe_per_kWth'] * N * 0.5 * (mf['E_rej'] + md['E_rej'])
    E_net = E_out - E_floor_fan - E_rej_fan
    RTE, RTE_gross = E_net / E_in, E_out / E_in
    rte_min = (opt.get('objective') or {}).get('rte_min', 0.0) or 0.0
    if RTE < rte_min:
        viol['rte'] = 1 - RTE / rte_min
    # capex
    ps = m.powerstick
    A_cell_cm2 = 4 * ps.width * m.L * (1 - ps.f_bare) * 1e4
    cell_cost = cost['cell_per_cm2'] * A_cell_cm2
    stick_cost = cell_cost + cost['stick_per_m'] * m.L + cost['stick_fixed']
    P_heat_kW = hs['P_heater'] / 1e3
    P_ac_peak = N * max(mf['P_elec_peak'], md['P_elec_peak']) * eo
    inv_kW = max(opt['P_rated'], P_ac_peak) / 1e3
    Q_rej_peak = max(mf['Q_rej_peak'], md['Q_rej_peak'])
    lines = {
        'tank (containment, insulation, supports, foundation)': tank.cost['total'][1],
        'storage graphite': tank.cost['storage_graphite'][1],
        'powersticks': N * stick_cost,
        'heaters': N * (cost['heater_fixed'] + cost['charge_electrics_per_kW'] * P_heat_kW),
        'inverters': cost['inverter_per_kW'] * inv_kW,
        'heat rejection': cost['heat_rejection_per_kWth'] * N * Q_rej_peak / 1e3,
    }
    soft = 1 + eco['soft_cost_frac']
    capex0 = soft * sum(lines.values())
    rep_cost = soft * N * cell_cost
    clean_cost = soft * N * cost['bare_clean_per_stick']
    capex_total = capex0 + n_rep * rep_cost + n_clean * clean_cost
    loss_cost = eco['elec_price'] * (E_in - E_net) / 3.6e9 * cyc_life        # $ over the life

    def per_yr(step):
        return step / dep_nm / eco['cycles_per_year'] if dep_nm > 0 else np.inf

    MWh = 1 / 3.6e9
    out.update(
        feasible=not viol, violations=viol, lifetime_cost=capex_total + loss_cost, loss_cost=loss_cost,
        capex_total=capex_total, capex0=capex0, capex_lines=lines,
        rep_cost=rep_cost, n_rep=n_rep, rep_interval_yr=per_yr(d_rep), n_clean=n_clean,
        clean_interval_yr=per_yr(d_cln), clean_cost=clean_cost, D_cell_end=D_c, D_bare_end=D_b,
        RTE=RTE, RTE_gross=RTE_gross, E_out_MWh=E_out * MWh, E_net_MWh=E_net * MWh, E_in_MWh=E_in * MWh,
        E_hold_MWh=E_hold_heat * MWh / eco['eta_in'], E_floor_fan_MWh=E_floor_fan * MWh,
        E_rej_fan_MWh=E_rej_fan * MWh, P_floor_fan_kW=P_floor_fan / 1e3,
        N_ps=N, D_hot=hz.D_hot, D_tank=geo.D_tank, H_tank=geo.H_tank,
        energy_mismatch=hz.energy_mismatch, dep_nm_per_cycle=dep_nm,
        E_stick_fresh_kWh=st_f.E_stick_e / 3.6e6, E_stick_deg_kWh=st_d.E_stick_e / 3.6e6,
        P_stick_fresh_kW=st_f.P_stick / 1e3, t_discharge_hr=t_dis_f, t_discharge_deg_hr=t_dis_d,
        t_charge_hr=hs['t_charge'], t_hold_hr=t_hold, heater_area_ratio=hs['ar'], heater_cavity_D=2 * hs['r_cav'],
        side_edge_frac=pf, loss_pct_day=tank.loss_cycle['Q_total'] * 86400 / hz.E_th * 100,
        Q_rej_peak_kW=Q_rej_peak / 1e3, inverter_kW=inv_kW, r_in=m.r_in, r_out=m.r_out,
        flux_fresh=mf, flux_deg=md, tank_warnings=tank.warnings,
    )
    return out


def objective_from(res):
    if res.get('error'):
        return FAILED
    if not res.get('feasible'):
        return INFEASIBLE + 1e3 * float(sum(res['violations'].values()))
    return res['lifetime_cost'] / 1e6


class Objective:
    """Picklable objective for differential_evolution (workers). Logs every evaluation
    to <log_dir>/evals_<pid>.jsonl."""

    def __init__(self, base_cfg, opt, log_dir=None, quiet=True):
        self.base_cfg, self.opt, self.log_dir, self.quiet = base_cfg, opt, log_dir, quiet

    def __call__(self, x):
        t0 = time.perf_counter()
        d = decode(x, self.opt)
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf) if self.quiet else contextlib.nullcontext():
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    res = evaluate_design(d, self.base_cfg, self.opt)
        except Exception as e:                              # infeasible geometry, solver failure
            res = dict(design=d, error=f"{type(e).__name__}: {e}", trace=traceback.format_exc(limit=3))
        f = objective_from(res)
        res.update(f=f, x=list(map(float, x)), wall_s=time.perf_counter() - t0)
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            with open(os.path.join(self.log_dir, f'evals_{os.getpid()}.jsonl'), 'a') as fh:
                fh.write(json.dumps(_jsonable(res)) + '\n')
        return f


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and not np.isfinite(o):
        return str(o)
    return o


def load_log(log_dir):
    """All logged evaluations (list of dicts), oldest first."""
    rows = []
    if not os.path.isdir(log_dir):
        return rows
    for fn in sorted(os.listdir(log_dir)):
        if fn.startswith('evals_') and fn.endswith('.jsonl'):
            with open(os.path.join(log_dir, fn)) as fh:
                rows += [json.loads(l) for l in fh if l.strip()]
    return rows


def run_de(base_cfg, opt, log_dir, maxiter=30, popsize=10, workers=20, seed=0, x0=None, tol=1e-3,
           callback=None):
    """differential_evolution over DESIGN_VARS in a process pool (one thread per worker)."""
    import multiprocessing as mp
    from scipy.optimize import differential_evolution
    env = {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')}
    for k in env:
        os.environ[k] = '1'
    obj = Objective(base_cfg, opt, log_dir)
    try:
        with mp.get_context('spawn').Pool(workers) as pool:
            res = differential_evolution(obj, BOUNDS, integrality=INTEGRALITY, maxiter=maxiter, popsize=popsize,
                                         seed=seed, x0=x0, tol=tol, polish=False, updating='deferred',
                                         workers=pool.map, init='sobol', callback=callback)
    finally:
        for k, v in env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return res


DETAILED = {'dz': 1 / 3, 'nrad': 5, 'dt_scale': 1.0, 'vane_rays': 2 ** 14, 'heater_ar_tol': 0.01}


def summary(res):
    """Printable one-design report (multi-line str)."""
    d = res['design']
    L = [f"Design: liner {d['liner_side']*1e3:.0f} mm, powerstick {d['ps_side']*1e3:.0f} mm, L {d['L']:.2f} m, "
         f"block H {d['H_block']:.2f} m, flux {d['avg_flux']:.1f} W/cm2, shade {d['shade_thickness']/INCH:.2f} in, "
         f"insulation side {d['t_side']:.2f} / top {d['t_top']:.2f} m, T_max {d['T_max_C']:.0f} C"]
    if res.get('error'):
        return '\n'.join(L + [f"FAILED: {res['error']}"])
    h = res.get('heater', {})
    if 'capex_total' not in res:
        return '\n'.join(L + [f"INFEASIBLE before sizing: {res.get('violations')} ({h.get('reason', '')})"])
    ff = res['flux_fresh']
    L += [
        f"Plant: {res['N_ps']} sticks + heaters, hot zone {res['D_hot']:.1f} m, tank {res['D_tank']:.1f} x {res['H_tank']:.1f} m; "
        f"block r {res['r_in']:.3f}-{res['r_out']:.3f} m, T {res['T_min_C']:.0f}-{d['T_max_C']:.0f} C (shade-open sizing: "
        f"volume x{res['volume_factor']:.3f}, {res['sizing_runs']} runs{'' if res['sizing_converged'] else ', NOT converged'}); heater {res['heater'].get('P_heater', 0)/1e3:,.0f} kW, cavity {res['heater_cavity_D']*1e3:.0f} mm "
        f"(area_ratio {res['heater_area_ratio']:.2f})",
        f"Cycle: charge {res['t_charge_hr']:.1f} h, hold {res['t_hold_hr']:.1f} h, discharge {res['t_discharge_hr']:.1f} h clean / "
        f"{res['t_discharge_deg_hr']:.1f} h end of interval; energy vs target {res['energy_mismatch']*100:+.1f}% (clean); "
        f"losses {res['loss_pct_day']:.2f} %/day",
        f"Per stick: {res['E_stick_fresh_kWh']:,.0f} kWh clean -> {res['E_stick_deg_kWh']:,.0f} kWh at D_cell "
        f"{res['D_cell_end']:.2f} / D_bare {res['D_bare_end']:.2f}; deposition {res['dep_nm_per_cycle']*1e3:.2f} pm/cycle",
        f"Maintenance: {res['n_rep']} cell replacements (every {res['rep_interval_yr']:.1f} yr), "
        f"{res['n_clean']} bare-area cleanings besides (every {res['clean_interval_yr']:.1f} yr)",
        f"Flux (clean): {ff.get('frac_above', 0)*100:.0f}% of stick-in time above target, {ff.get('frac_band', 0)*100:.0f}% at "
        f"target, {ff.get('frac_below', 0)*100:.0f}% below; std {ff.get('flux_std', 0):.2f} W/cm2, rms about target "
        f"{ff.get('flux_rms_dev', 0):.2f}, max {ff.get('flux_max', 0):.1f}, end {ff.get('flux_end', 0):.1f}",
        f"Per cycle: out {res['E_out_MWh']:,.0f} MWh AC - fans {res['E_floor_fan_MWh']:,.1f} floor / "
        f"{res['E_rej_fan_MWh']:,.1f} heat rejection = {res['E_net_MWh']:,.0f} MWh net; in {res['E_in_MWh']:,.0f} MWh "
        f"(hold {res['E_hold_MWh']:,.0f})",
        f"Round trip: {res['RTE']*100:.1f}% net ({res['RTE_gross']*100:.1f}% before fans); energy lost over the "
        f"life ${res['loss_cost']/1e6:,.1f}M",
        f"Capex ${res['capex0']/1e6:.1f}M initial (incl. soft costs): " + ', '.join(
            f"{k.split(' (')[0]} ${v/1e6:.1f}M" for k, v in res['capex_lines'].items())
        + f"; inverters sized {res['inverter_kW']/1e3:.1f} MW; cell replacement ${res['rep_cost']/1e6:.1f}M each",
        f"LIFETIME CAPEX ${res['capex_total']/1e6:,.1f}M; objective (capex + energy lost) "
        f"${res['lifetime_cost']/1e6:,.1f}M" + ('' if res['feasible'] else f"  INFEASIBLE {res['violations']}"),
    ]
    return '\n'.join(L)


def pareto(rows, allow=('rte',)):
    """Non-dominated designs (min capex_total, max RTE) among rows whose only violations
    are in `allow` (the RTE floor does not disqualify a design from the front)."""
    ok = [r for r in rows if 'capex_total' in r and set(r.get('violations') or {}) <= set(allow)]
    ok.sort(key=lambda r: (r['capex_total'], -r['RTE']))
    front, best = [], -np.inf
    for r in ok:
        if r['RTE'] > best:
            front.append(r)
            best = r['RTE']
    return front


def evaluate(design, base_cfg, opt, fidelity=None, quiet=True):
    """evaluate_design with optional fidelity override (e.g. DETAILED), stdout silenced."""
    o = merge(opt, {'fidelity': fidelity}) if fidelity else opt
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf) if quiet else contextlib.nullcontext():
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = evaluate_design(design, base_cfg, o)
    res['f'] = objective_from(res)
    return res
