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
  4. degradation     cells replaced at D_cell = cell_replace_D, bare areas cleaned (reset)
                     at D_bare = bare_clean_D (and at replacements). Performance = average
                     over every cycle of the life: the clean run averaged with a run at twice
                     the lifetime-mean D's (exact for performance linear in D, incl. a
                     partial last interval and designs never replaced).
  5. metrics         lifetime capex (initial + cell replacements + bare cleanings, all
                     x (1 + soft costs)), net round-trip efficiency = (AC out - floor and
                     heat-rejection fans) / AC in (incl. hold losses), and the lifetime cost
                     of the energy lost = price x (AC in - net AC out) x cycles

Optimized variables (DESIGN_VARS, or opt['design_vars'] from the notebook inputs cell) are reparameterised so every point in the box is a
valid geometry: powerstick side as a fraction of what fits in the liner, block height as
a fraction between L + min_lower_zone and H_max. T_min and the swing are outputs of the sizing; T_min >=
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

from .build import build_tinless, build_heater
from .heater import heater_max_power, element as heater_element, element_mass, describe as describe_heater
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
    'charge_hr': 50.0,            # h design charge time (heater power, then cavity, sized to meet it)
    'economics': {
        'eta_out': 0.98,          # DC -> AC inverter
        'eta_in': 0.98,           # AC -> heater supply
        'cycles_per_year': 8760.0 / 168.0,   # 1-week cycle: charge + hold + discharge (hold losses charged)
        'lifetime_yr': 30.0,
        'soft_cost_frac': 0.30,   # tax, freight, soft costs on all capex (incl. replacements)
        'elec_price': 75.0,       # $/MWh, same for charge and discharge: prices the energy lost
        # 'Pilot' | 'FOAK' | 'NOAK' (thermal/cost_scenarios.py) overrides the cost inputs it lists, on top
        # of these and the notebook inputs; None = the inputs as written
        'cost_scenario': None,
    },
    'objective': {'rte_min': 0.0},     # optional floor on net RTE (0 = none)
    'costs': {
        'cell_per_cm2': 1.0,      # $/cm2 of cell-covered powerstick face (incl. at replacement)
        'stick_per_m': 100.0,     # $/m of stick length
        'stick_fixed': 2000.0,    # $/stick
        # iso-molded graphite $/kg: the tank support-column graphite and the heater graphite; None ->
        # the YAML tank.costs column_graphite_per_kg (central)
        'iso_graphite_per_kg': None,
        # heater: heater_fixed (element machining, terminals, assembly, install) + element graphite (cross-
        # section x (H_block + t_top) x heater.density) x heater_graphite_per_kg, per heater
        'heater_fixed': 0.0,               # $/heater
        'heater_graphite_per_kg': None,    # $/kg; None -> iso_graphite_per_kg
        'charge_electrics_per_kW': 75.0,   # $/kW heater supply (PLACEHOLDER)
        'inverter_per_kW': 100.0,          # $/kW_AC peak (NREL ~$75/kW ex-factory + install)
        'heat_rejection_per_kWth': 125.0,  # $/kW_th dry cooler (IEA SHC Task 48: $61-134, 2014)
        'bare_clean_per_stick': 0.0,       # $/stick per bare-area cleaning (tracked; cost TBD)
        'interconnect_per_kW': 150.0,      # $/kW grid connection, sized on max(peak charge, peak output) AC
                                           #   (LBNL: completed storage projects avg ~$151/kW) PLACEHOLDER
        # liner: grafoil walls (4 x side x L) + CFC floor (side^2 x floor_area_factor: the extra
        # area covers CFC upper features)
        'liner_wall_per_m2': 55.0,         # $/m2 grafoil
        'liner_floor_per_m2': 1000.0,      # $/m2 CFC
        'liner_floor_area_factor': 2.0,
        # shades: 8 RGF vanes per stick, full block height; one graphite / CFC axle per vane up through
        # the top insulation (t_top + shade_axle_extra) to the drive
        'shade_rgf_per_m3': 15000.0,       # $/m3 rigid graphite felt, machined (PLACEHOLDER)
        'shade_axle_per_m': 300.0,         # $/m graphite / CFC axle (PLACEHOLDER)
        'shade_axle_extra': 0.5,           # m axle above the top insulation to the drive
        'shade_drive': 'linkage',          # 'linkage': 1 motor + linkage per stick | 'per_vane': 1 motor per vane
        'shade_motor': 1500.0,             # $/motor incl. driver (PLACEHOLDER)
        'shade_linkage': 2000.0,           # $/stick linkage, 'linkage' drive (PLACEHOLDER)
    },
    # central values replacing the YAML tank.costs rates for every design (scalar $ per unit;
    # keys of tank.TANK_DEFAULTS['costs']); empty = the YAML values
    'tank_costs': {},
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
        'H_max': 14.0,            # m block height upper bound
        'min_lower_zone': 0.25,   # m block always this much taller than the stick: a floor below the stick,
                                  #   liner bottom + RGF pad and radiation through the lower zone are always modelled
        'T_cap_C': 2500.0,        # C block T_max cap (heater max 2600 C)
        'T_min_floor_C': 1400.0,  # C lowest allowed T_min (constraint; T_min is a sizing output)
        'min_wall': 0.30,         # m min graphite wall between neighbouring cavities (constraint);
                                  #   stick cavities turned 45 deg to the checkerboard grid
        'swing_min': 200.0, 'swing_max': 1000.0,   # C allowed swing (constraint)
    },
    'limits': {'D_tank_max': 80.0, 'H_tank_max': 30.0},
    # heater current leads through the top insulation (thermal/heater_leads.py). table: path of a lead
    # table (HL.build_lead_table) -> every design gets its leads' cost and losses interpolated from it
    # at its heater current, t_top and block T_max, and the element graphite covers the block height
    # only (the leads replace the length through the roof). None -> leads not modelled (the element
    # runs on through the top insulation, as before).
    'leads': {'table': None, 'n_per_heater': 2},
    'fidelity': {'dz': 0.5, 'nrad': 5, 'dt_scale': 2.0, 'vane_rays': 2 ** 12, 'heater_ar_tol': 0.03},
}

# name, lower, upper, integer, description
DESIGN_VARS = [
    ('liner_side', 0.20, 0.60, False, 'liner side length [m]'),
    ('ps_frac', 0.0, 1.0, False, 'powerstick side: 0.10 m + frac x (room in liner - 0.10)'),
    ('L', 3.0, 6.0, False, 'stick length [m]'),
    ('H_frac', 0.0, 1.0, False, 'block height: L + dz_min + frac x (H_max - L - dz_min)'),
    ('avg_flux', 5.0, 20.0, False, 'cell flux target [W/cm2] (continuous)'),
    ('shade_8th', 2, 48, True, 'shade thickness i x 1/8 in (1/4 - 6 in)'),
    ('t_side', 0.5, 3.0, False, 'tank side insulation [m]'),
    ('t_top', 0.5, 3.0, False, 'tank top insulation [m]'),
    ('L_bot', 2.0, 4.0, False, 'bottom stack under the hot zone (support columns + Thermax) [m]'),
    ('T_max_C', 1800.0, 2500.0, False, 'block T_max [C] (capped at geometry.T_cap_C); T_min from shade-open sizing'),
]
VAR_NAMES = [v[0] for v in DESIGN_VARS]
BOUNDS = [(v[1], v[2]) for v in DESIGN_VARS]
INTEGRALITY = np.array([v[3] for v in DESIGN_VARS])
OPT_DEFAULTS['design_vars'] = [list(v) for v in DESIGN_VARS]


def design_vars(opt):
    """The run's optimized variables: opt['design_vars'] (the notebook inputs cell), else
    DESIGN_VARS. Names must be exactly VAR_NAMES (any order); bounds can change.
    -> list of (name, lo, hi, integer, description)."""
    dv = [tuple(v) for v in ((opt or {}).get('design_vars') or DESIGN_VARS)]
    names = [v[0] for v in dv]
    if sorted(names) != sorted(VAR_NAMES):
        raise ValueError(f"design_vars must name each of {VAR_NAMES} once, got {names}")
    for name, lo, hi, *_ in dv:
        if not lo < hi:
            raise ValueError(f"design_vars {name}: lower {lo} must be below upper {hi}")
    return dv


def var_bounds(opt):
    return [(float(v[1]), float(v[2])) for v in design_vars(opt)]


def var_integrality(opt):
    return np.array([bool(v[3]) for v in design_vars(opt)])


def merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = v
    return out


def _flat(d, pre=''):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict) and v:
            out.update(_flat(v, f"{pre}{k}."))
        else:
            out[f"{pre}{k}"] = v
    return out


def scenario_name(opt):
    """Active cost scenario ('pilot' / 'foak' / 'noak') or None."""
    from .cost_scenarios import SCENARIOS
    sc = ((opt or {}).get('economics') or {}).get('cost_scenario')
    if sc in (None, '', False) or str(sc).lower() == 'none':
        return None
    sc = str(sc).lower()
    if sc not in SCENARIOS:
        raise ValueError(f"economics.cost_scenario must be one of {sorted(SCENARIOS)} (any case) or None, not {sc!r}")
    return sc


def apply_scenario(opt, given=None):
    """opt with the active cost scenario's values on top (idempotent). opt['_sources']
    records, per dotted key, 'inputs' (given, e.g. the notebook inputs cell) or 'scenario X'."""
    from .cost_scenarios import SCENARIOS
    sc = scenario_name(opt)
    src = dict(opt.get('_sources') or {})
    if given is not None:
        src.update({k: 'inputs' for k in _flat(given)})
    out = merge(opt, SCENARIOS[sc]) if sc else copy.deepcopy(opt)
    if sc:
        src.update({k: f"scenario {sc.upper()}" for k in _flat(SCENARIOS[sc])})
    out['_sources'] = src
    return out


def build_opt(inputs):
    """Optimizer inputs for a run: OPT_DEFAULTS < `inputs` (the notebook inputs cell) < the
    cost scenario named in inputs['economics']['cost_scenario']. Checks design_vars and the
    tank_costs keys."""
    opt = apply_scenario(merge(OPT_DEFAULTS, inputs), given=inputs)
    design_vars(opt)
    bad = set(opt.get('tank_costs') or {}) - set(tk.TANK_DEFAULTS['costs'])
    if bad:
        raise KeyError(f"tank_costs: unknown keys {sorted(bad)} (see tank.TANK_DEFAULTS['costs'])")
    return opt


def _central(v):
    """Central value of a YAML [low, central, high] cost triple (or a scalar)."""
    return float(np.broadcast_to(np.asarray(v, float), (3,))[1])


def _with_central(old, v):
    """Triple `old` with its central value set to v (low / high widened to include it)."""
    if isinstance(old, (list, tuple)) and len(old) == 3:
        return [min(float(old[0]), v), float(v), max(float(old[2]), v)]
    return float(v)


def apply_tank_costs(cfg_tank, opt):
    """YAML tank section with the optimizer's cost inputs: tank_costs (central values),
    costs.iso_graphite_per_kg -> column_graphite_per_kg, and the economics the tank model
    reports lost heat with (elec_price, heater_eff = eta_in)."""
    c = cfg_tank.setdefault('costs', {})
    over = dict(opt.get('tank_costs') or {})
    iso = (opt.get('costs') or {}).get('iso_graphite_per_kg')
    if iso is not None:
        over['column_graphite_per_kg'] = iso
    for k, v in over.items():
        if v is None:
            continue
        if k not in tk.TANK_DEFAULTS['costs']:
            raise KeyError(f"tank_costs: unknown key {k!r}")
        c[k] = _with_central(c.get(k, tk.TANK_DEFAULTS['costs'][k]), float(v))
    eco = opt.get('economics') or {}
    cfg_tank['economics'] = dict(cfg_tank.get('economics') or {}, elec_price=eco.get('elec_price', 87.0),
                                 heater_eff=eco.get('eta_in', 0.9))
    return cfg_tank


# --------------------------------------------------------------------------
# Design vector <-> physical design
# --------------------------------------------------------------------------
def decode(x, opt):
    """x (DESIGN_VARS order) -> physical design dict."""
    g = opt['geometry']
    v = dict(zip([dv[0] for dv in design_vars(opt)], np.asarray(x, float)))
    liner = v['liner_side']
    ps_hi = min(0.40, liner - 2 * g['ps_liner_gap'])
    ps = 0.10 + v['ps_frac'] * max(ps_hi - 0.10, 0.0)
    L = v['L']
    dz0 = g.get('min_lower_zone', 0.25)           # always a lower zone (floor, liner bottom, pad)
    H = L + dz0 + v['H_frac'] * max(g['H_max'] - L - dz0, 0.0)
    flux = float(v['avg_flux'])
    shade_t = int(round(v['shade_8th'])) * INCH / 8
    return dict(liner_side=liner, ps_side=ps, L=L, H_block=H, avg_flux=flux, shade_thickness=shade_t,
                t_side=v['t_side'], t_top=v['t_top'], L_bot=v['L_bot'], T_max_C=min(v['T_max_C'], g['T_cap_C']))


def encode(design, opt):
    """Physical design dict -> x (inverse of decode, clipped to the bounds)."""
    g = opt['geometry']
    liner, L = design['liner_side'], design['L']
    ps_hi = min(0.40, liner - 2 * g['ps_liner_gap'])
    v = dict(liner_side=liner, ps_frac=(design['ps_side'] - 0.10) / max(ps_hi - 0.10, 1e-9), L=L,
             H_frac=(design['H_block'] - L - g.get('min_lower_zone', 0.25))
             / max(g['H_max'] - L - g.get('min_lower_zone', 0.25), 1e-9), avg_flux=design['avg_flux'],
             shade_8th=round(design['shade_thickness'] / (INCH / 8)), t_side=design['t_side'],
             t_top=design['t_top'], L_bot=design.get('L_bot', 2.0), T_max_C=design['T_max_C'])
    dv = design_vars(opt)
    return np.clip(np.array([v[d[0]] for d in dv], float), [d[1] for d in dv], [d[2] for d in dv])


def design_from_cfg(cfg):
    """The design currently in a model YAML (for a baseline / starting point)."""
    s, p, sh, t = cfg['storage'], cfg['powerstick'], cfg.get('shade', {}), cfg.get('tank', {})
    L = float(s['L'])
    return dict(liner_side=p['liner_side_len'], ps_side=p['side_len'], L=L,
                H_block=float(s.get('block_height') or L), avg_flux=float(s['avg_flux']),
                shade_thickness=float(sh.get('thickness', 0.0254)), t_side=float(t.get('t_side', 1.0)),
                t_top=float(t.get('t_top', 1.0)), L_bot=float(t.get('L_bot', 2.0)),
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
    if 'L_bot' in d:                           # older logged designs have no L_bot: YAML value
        t['L_bot'] = d['L_bot']
    t['P_rated'], t['duration_hr'] = opt['P_rated'], opt['duration_hr']
    t['block_axial_losses'] = True
    apply_tank_costs(t, opt)                   # cost rates from the optimizer inputs / scenario
    return cfg


def lifetime_mean_D(dep_life, d_rep, d_cln, n=200001):
    """Cycle-averaged deposit [nm] on the cells and on the bare area over the life.
    dep_life: deposit over the life with no resets; cells reset every d_rep, bare area every
    d_cln and at each cell replacement. Deposit is linear in cycles, so a uniform grid in
    deposit is a uniform grid in cycles."""
    if dep_life <= 0:
        return 0.0, 0.0
    x = np.linspace(0.0, dep_life, n)
    dc = np.mod(x, d_rep) if d_rep > 0 else x
    db = np.mod(dc, d_cln) if d_cln > 0 else dc
    return float(dc.mean()), float(db.mean())


def liner_shade_costs(m, t_top, cost):
    """Per-stick liner and shade hardware cost [$] for a built model m (build_tinless).
    Liner: grafoil walls 4 x side x L at liner_wall_per_m2, CFC floor side^2 x
    liner_floor_area_factor at liner_floor_per_m2. Shade: 8 RGF vanes (width = vane length
    l, thickness t) over the full block height, an axle per vane through the top
    insulation, and the drive (one motor per vane, or one motor + linkage per stick)."""
    s, L, H = m.liner_side_len, m.L, m.H_block
    liner = (cost['liner_wall_per_m2'] * 4 * s * L
             + cost['liner_floor_per_m2'] * cost['liner_floor_area_factor'] * s ** 2) if m.liner_bool else 0.0
    out = dict(liner=liner, shade=0.0, shade_rgf_m3=0.0, shade_parts={})
    if not m.shade_bool:
        return out
    lay = m.vane_lay
    l = lay['l'] if lay is not None else m.shade_side_len / (1 + np.sqrt(2))   # octagon panel width
    t = lay['t'] if lay is not None else m.shade_thickness
    n = 8
    V = n * l * t * H
    drive = cost.get('shade_drive', 'linkage')
    if drive == 'per_vane':
        c_drive = n * cost['shade_motor']
    elif drive == 'linkage':
        c_drive = cost['shade_motor'] + cost['shade_linkage']
    else:
        raise ValueError(f"costs.shade_drive must be 'linkage' or 'per_vane', not {drive!r}")
    parts = dict(rgf=cost['shade_rgf_per_m3'] * V,
                 axles=n * cost['shade_axle_per_m'] * (t_top + cost['shade_axle_extra']),
                 drive=c_drive)
    out.update(shade=sum(parts.values()), shade_rgf_m3=V, shade_parts=parts)
    return out


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
                r_cav=hm.r_in, P_heater=hm.heater.max_power,
                element={k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in hm.element.items()})


def heater_search(m, cfg, opt):
    """Heater power and cavity for t_charge <= charge_hr. The power cap is YAML
    heater.max_power_per_m x heater length (= block height).
      - cavity at its minimum (heater_min_gap) meets the target at the cap: the heater
        power is cut to the smallest that still meets it (electrics costed on that);
      - otherwise the cavity grows at the cap: smallest area_ratio that meets it.
    The element is sized for each power tried (thermal.heater.element: a tube from the
    voltage, wall ratio and resistivity), and the minimum cavity follows its radius.
    -> dict(ok, violation, ar, t_charge, P_heater, element, ...)."""
    h = cfg['heater']
    target = opt['charge_hr']
    dts = opt['fidelity']['dt_scale']
    tol = opt['fidelity']['heater_ar_tol']
    T_lo = float(m.T_min)

    def ar_min(P):                       # cavity at the minimum heater gap, element sized for P
        r = heater_element(h, m.H_block, T_lo, P=P)['r_h']
        return (r + opt['geometry']['heater_min_gap']) / r

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
    ar_lo = ar_min(P_cap)
    c_lo = _charge(m, cfg, ar_lo, end_hr, dts)
    if c_lo['t_charge'] <= target:
        # minimum cavity is enough at the cap: smallest heater power that still meets the target
        P_lo, P_hi, c_hi = E_blk / (target * 3600), P_cap, c_lo
        while P_hi / P_lo > 1 + tol:
            P_mid = math.sqrt(P_lo * P_hi)
            c_mid = _charge(m, cfg, ar_min(P_mid), end_hr, dts, P=P_mid)
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
                    E_released=0.0, E_cell=0.0, E_el=0.0)
    tol = opt.get('flux_band_tol', 0.005)
    above, below = q > q_t * (1 + tol), q < q_t * (1 - tol)
    band = ~above & ~below
    E_el = float((w * P_el).sum())                                      # DC
    E_cell = float((w * np.asarray(r['Q_cell'][1:], float)).sum())     # absorbed by the active cells
    mean = float((w * q).sum() / T_on)
    return dict(
        Q_rej_peak=float(Q_rej[on].max()), E_rej=float((w * Q_rej).sum()), P_elec_peak=float(P_el[on].max()),
        t_on_hr=T_on / 3600, flux_target=q_t, E_rad=E_rad, E_sink=E_sink, E_released=E_rad + E_sink,
        E_cell=E_cell, E_el=E_el,
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
    opt = apply_scenario(opt)                  # no-op when build_opt already applied it
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
    # graphite walls between neighbouring cavities on the checkerboard
    pitch, webs = tk.cavity_webs(m.r_in, m.r_out, hs['r_cav'], rotated=True)
    out.update(pitch=pitch, webs=webs, min_wall=min(webs.values()))
    if out['min_wall'] < g['min_wall']:
        viol['wall'] = (g['min_wall'] - out['min_wall']) / g['min_wall']
        out.update(violations=viol, feasible=False)
        return out
    dep_nm = float(r_f.get('dep_total_nm', 0.0))
    cyc_life = eco['cycles_per_year'] * eco['lifetime_yr']
    dep_life = dep_nm * cyc_life                               # nm on one stick slot over the life
    d_rep = dep['cell_replace_D'] * dep['d_dead_nm']
    d_cln = dep['bare_clean_D'] * dep['d_dead_nm']
    n_rep = _n_events(dep_life, d_rep)
    n_reset = _n_events(dep_life, d_cln)                       # bare resets, incl. at replacements
    n_clean = sum(1 for k in range(1, n_reset + 1)
                  if abs(k * d_cln / d_rep - round(k * d_cln / d_rep)) > 1e-6)   # cleanings alone
    # Performance over the life = the average over every cycle. Deposition grows linearly with
    # cycles; cell D saw-tooths (reset at replacements), bare D too (reset at cleanings and at
    # replacements), with a partial last interval. Performance ~linear in D, so the clean run
    # averaged with one run at TWICE the lifetime-mean D's gives the lifetime average exactly.
    Dc_mean, Db_mean = lifetime_mean_D(dep_life, d_rep, d_cln)
    D_c = 2 * Dc_mean / dep['d_dead_nm']
    D_b = 2 * Db_mean / dep['d_dead_nm']
    # 3. tank: pass 1 (clean) for the side + edge share, then sized on the clean AC stick
    st_f = dataclasses.replace(st_f, A_hcav_default=math.pi * hs['r_cav'] ** 2)   # searched cavity -> tank layout
    t1 = tk.tank_system(st_f, cfg['tank'])
    pf = t1.stick_correction['penalty_frac']
    eo = eco['eta_out']
    E_f_ac = st_f.E_stick_e * (1 - pf) * eo
    st_size = dataclasses.replace(st_f, P_stick=E_f_ac / st_f.t_discharge, E_stick_e=E_f_ac)
    tank = tk.tank_system(st_size, cfg['tank'])
    hz, geo, c = tank.hz, tank.geo, tank.cfg
    N = hz.N_ps
    # 4. lifetime-average equivalent state (no new deposition in this run)
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
    T_lim = tk.floor_limit(c)                    # floor design limit (else API 650 Annex M)
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
    # 5. energy per cycle [J], lifetime-average cells
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
    # heater leads (lead table): per lead at this design's current, t_top and block T_max. Joule heat
    # while charging is electricity bought; the heat they drain from the storage (charging: net, < 0
    # when Joule heat flows down into the block; idle = hold + discharge, graphite joint open) is put
    # back by the heaters; the heat they reject (cooled copper) loads the heat rejection
    ld = None
    E_lead_joule = E_lead_drain = E_lead_rej = 0.0
    if lt:
        el = hs['element']
        V = float(el.get('V') or cfg['heater'].get('voltage') or 0.0)
        if V <= 0:
            raise ValueError("heater leads need the heater voltage (heater.voltage)")
        n_ld = N * int(opt['leads'].get('n_per_heater', 2))
        ld = lt(hs['P_heater'] / V, c['t_top'], T_max_C)
        t_ch_s, t_idle_s = hs['t_charge'] * 3600, max(cycle_hr - hs['t_charge'], 0.0) * 3600
        E_lead_joule = n_ld * ld['Q_J_charge'] * t_ch_s
        E_lead_drain = n_ld * (ld['drain_charge'] * t_ch_s + ld['drain_idle'] * t_idle_s)
        E_lead_rej = n_ld * (max(ld['Q_rej_charge'], 0.0) * t_ch_s + max(ld['Q_rej_idle'], 0.0) * t_idle_s)
        E_charge_heat += E_lead_drain
    E_in = (E_charge_heat + E_hold_heat + E_lead_joule) / eco['eta_in']
    E_out = N * 0.5 * (E_f_ac + E_d_ac)
    P_floor_fan = tank.loss_peak['P_fan']
    E_floor_fan = P_floor_fan * cycle_hr * 3600                          # runs all cycle
    E_rej_fan = par['heat_rejection_kWe_per_kWth'] * (N * 0.5 * (mf['E_rej'] + md['E_rej']) + E_lead_rej)
    E_net = E_out - E_floor_fan - E_rej_fan
    RTE, RTE_gross = E_net / E_in, E_out / E_in
    # average cell efficiency = DC / heat absorbed by the active cells, life average; the
    # absorbed heat is scaled to the plant like the output (side + edge share)
    E_abs = mf['E_cell'] + md['E_cell']
    eta_cell = (mf['E_el'] + md['E_el']) / E_abs if E_abs > 0 else np.nan
    E_cell_abs = N * (1 - pf) * 0.5 * E_abs
    rte_min = (opt.get('objective') or {}).get('rte_min', 0.0) or 0.0
    if RTE < rte_min:
        viol['rte'] = 1 - RTE / rte_min
    # capex
    ps = m.powerstick
    A_cell_cm2 = 4 * ps.width * m.L * (1 - ps.f_bare) * 1e4
    cell_cost = cost['cell_per_cm2'] * A_cell_cm2
    stick_cost = cell_cost + cost['stick_per_m'] * m.L + cost['stick_fixed']
    ls = liner_shade_costs(m, cfg['tank']['t_top'], cost)
    P_heat_kW = hs['P_heater'] / 1e3
    lt = lead_table(opt)
    # element graphite: block height only when the leads are modelled (they replace the length
    # through the roof insulation); else the element runs on through the top insulation
    m_heater = element_mass(hs['element'], hz.H_block + (0.0 if lt else c['t_top']))   # kg
    heater_per_kg = cost.get('heater_graphite_per_kg')
    if heater_per_kg is None:                                # iso-molded graphite, as the support columns
        heater_per_kg = _central(c['costs']['column_graphite_per_kg'])
    hc_parts = {'fixed': float(cost.get('heater_fixed', 0.0) or 0.0), 'graphite': m_heater * heater_per_kg}
    hc = dict(total=sum(hc_parts.values()), parts=hc_parts)
    P_ac_peak = N * max(mf['P_elec_peak'], md['P_elec_peak']) * eo
    inv_kW = max(opt['P_rated'], P_ac_peak) / 1e3
    Q_rej_peak = max(mf['Q_rej_peak'], md['Q_rej_peak'])
    P_lead_J = 0.0                                           # W lead Joule per heater at full power
    if ld:
        n_ph = int(opt['leads'].get('n_per_heater', 2))
        P_lead_J = n_ph * ld['Q_J_charge']
        # cooling per stick slot: discharge peak + the idle leads, or the leads while charging
        Q_rej_peak = max(Q_rej_peak + n_ph * max(ld['Q_rej_idle'], 0.0), n_ph * max(ld['Q_rej_charge'], 0.0))
    P_chg_ac = N * (hs['P_heater'] + P_lead_J) / eco['eta_in']   # plant peak draw: all heaters at full power
    ic_kW = max(P_chg_ac, P_ac_peak) / 1e3                  # grid connection: the larger of charge and output
    lines = {
        'tank (containment, insulation, supports, foundation)': tank.cost['total'][1],
        'storage graphite': tank.cost['storage_graphite'][1],
        'powersticks': N * stick_cost,
        'liners (grafoil walls, CFC floor)': N * ls['liner'],
        'shades (RGF vanes, axles, drives)': N * ls['shade'],
        'heaters': N * (hc['total'] + cost['charge_electrics_per_kW'] * (P_heat_kW + P_lead_J / 1e3)),
        'inverters': cost['inverter_per_kW'] * inv_kW,
        'grid interconnect': cost['interconnect_per_kW'] * ic_kW,
        'heat rejection': cost['heat_rejection_per_kWth'] * N * Q_rej_peak / 1e3,
    }
    if ld:
        lines['heater leads (graphite / W / water-cooled Cu)'] = (
            N * int(opt['leads'].get('n_per_heater', 2)) * (ld['material'] + ld['joints']))
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
        RTE=RTE, RTE_gross=RTE_gross, eta_cell=eta_cell, RTE_frac_cell=RTE / eta_cell,
        E_cell_abs_MWh=E_cell_abs * MWh, E_dc_MWh=E_out / eo * MWh, E_out_MWh=E_out * MWh, E_net_MWh=E_net * MWh, E_in_MWh=E_in * MWh,
        E_hold_MWh=E_hold_heat * MWh / eco['eta_in'], E_floor_fan_MWh=E_floor_fan * MWh,
        E_rej_fan_MWh=E_rej_fan * MWh, P_floor_fan_kW=P_floor_fan / 1e3,
        N_ps=N, D_hot=hz.D_hot, D_tank=geo.D_tank, H_tank=geo.H_tank,
        energy_mismatch=hz.energy_mismatch, dep_nm_per_cycle=dep_nm,
        E_stick_fresh_kWh=st_f.E_stick_e / 3.6e6, E_stick_deg_kWh=st_d.E_stick_e / 3.6e6,
        P_stick_fresh_kW=st_f.P_stick / 1e3, t_discharge_hr=t_dis_f, t_discharge_deg_hr=t_dis_d,
        m_heater_kg=m_heater, heater_per_kg=heater_per_kg, heater_cost_each=hc['total'],
        heater_cost_parts=hc['parts'], cost_scenario=scenario_name(opt),
        t_charge_hr=hs['t_charge'], t_hold_hr=t_hold, heater_area_ratio=hs['ar'], heater_cavity_D=2 * hs['r_cav'],
        side_edge_frac=pf, loss_pct_day=tank.loss_cycle['Q_total'] * 86400 / hz.E_th * 100,
        Q_rej_peak_kW=Q_rej_peak / 1e3, inverter_kW=inv_kW, interconnect_kW=ic_kW, P_charge_ac_kW=P_chg_ac / 1e3,
        T_floor_max_C=fl['T_floor_max'], T_floor_col_C=fl['T_floor_col'], v_air=c['floor']['v_air'],
        foundation_kPa=tank.foundation_pressure / 1e3,
        r_in=m.r_in, r_out=m.r_out, liner_cost=ls['liner'], shade_cost=ls['shade'],
        shade_rgf_m3=ls['shade_rgf_m3'], shade_parts=ls['shade_parts'],
        flux_fresh=mf, flux_deg=md, tank_warnings=tank.warnings,
        leads=ld, E_lead_joule_MWh=E_lead_joule * MWh, E_lead_drain_MWh=E_lead_drain * MWh,
        E_lead_lost_MWh=(E_lead_joule + E_lead_drain) * MWh,
    )
    return out


_LEAD_TABLES = {}


def lead_table(opt):
    """The run's lead table (heater_leads.LeadTable) from opt['leads']['table'] (loaded once per
    process), or None."""
    path = ((opt or {}).get('leads') or {}).get('table')
    if not path:
        return None
    if path not in _LEAD_TABLES:
        from .heater_leads import LeadTable
        with open(path) as fh:
            _LEAD_TABLES[path] = LeadTable(json.load(fh), path)
    return _LEAD_TABLES[path]


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
        res.update(f=f, x=list(map(float, x)), x_names=[v[0] for v in design_vars(self.opt)],
                   wall_s=time.perf_counter() - t0)
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
    """differential_evolution over the run's design_vars (opt['design_vars'], else DESIGN_VARS)
    in a process pool (one thread per worker)."""
    import multiprocessing as mp
    from scipy.optimize import differential_evolution
    env = {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')}
    for k in env:
        os.environ[k] = '1'
    opt = apply_scenario(opt)
    obj = Objective(base_cfg, opt, log_dir)
    try:
        with mp.get_context('spawn').Pool(workers) as pool:
            res = differential_evolution(obj, var_bounds(opt), integrality=var_integrality(opt), maxiter=maxiter,
                                         popsize=popsize,
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
         f"insulation side {d['t_side']:.2f} / top {d['t_top']:.2f} m, bottom stack {d.get('L_bot', float('nan')):.2f} m, "
         f"T_max {d['T_max_C']:.0f} C"]
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
        f"(area_ratio {res['heater_area_ratio']:.2f})"
        + (f"; element {describe_heater(h['element'])}, {res.get('m_heater_kg', np.nan):,.0f} kg" if h.get('element') else ''),
        f"Grid: pitch {res['pitch']:.3f} m, graphite walls " + ', '.join(f"{k} {v*1e3:.0f}" for k, v in res['webs'].items())
        + " mm (stick cavities at 45 deg)",
        *([f"Floor (design-day air {res.get('v_air', float('nan')):.0f} m/s): max {res['T_floor_max_C']:.0f} C, "
           f"column hot spot {res['T_floor_col_C']:.0f} C; foundation {res['foundation_kPa']:.0f} kPa"]
          if 'T_floor_col_C' in res else []),
        f"Cycle: charge {res['t_charge_hr']:.1f} h, hold {res['t_hold_hr']:.1f} h, discharge {res['t_discharge_hr']:.1f} h clean / "
        f"{res['t_discharge_deg_hr']:.1f} h at 2x life-mean wear; energy vs target {res['energy_mismatch']*100:+.1f}% (clean); "
        f"losses {res['loss_pct_day']:.2f} %/day",
        f"Per stick: {res['E_stick_fresh_kWh']:,.0f} kWh clean, {0.5*(res['E_stick_fresh_kWh']+res['E_stick_deg_kWh']):,.0f} kWh "
        f"life average ({res['E_stick_deg_kWh']:,.0f} kWh at D_cell {res['D_cell_end']:.3f} / D_bare "
        f"{res['D_bare_end']:.3f} = 2x life-mean wear); deposition {res['dep_nm_per_cycle']*1e3:.2f} pm/cycle",
        f"Maintenance: {res['n_rep']} cell replacements (every {res['rep_interval_yr']:.1f} yr), "
        f"{res['n_clean']} bare-area cleanings besides (every {res['clean_interval_yr']:.1f} yr)",
        f"Flux (clean): {ff.get('frac_above', 0)*100:.0f}% of stick-in time above target, {ff.get('frac_band', 0)*100:.0f}% at "
        f"target, {ff.get('frac_below', 0)*100:.0f}% below; std {ff.get('flux_std', 0):.2f} W/cm2, rms about target "
        f"{ff.get('flux_rms_dev', 0):.2f}, max {ff.get('flux_max', 0):.1f}, end {ff.get('flux_end', 0):.1f}",
        f"Per cycle: out {res['E_out_MWh']:,.0f} MWh AC - fans {res['E_floor_fan_MWh']:,.1f} floor / "
        f"{res['E_rej_fan_MWh']:,.1f} heat rejection = {res['E_net_MWh']:,.0f} MWh net; in {res['E_in_MWh']:,.0f} MWh "
        f"(hold {res['E_hold_MWh']:,.0f})",
        f"Round trip: {res['RTE']*100:.1f}% net ({res['RTE_gross']*100:.1f}% before parasitic loads); energy lost over the "
        f"life ${res['loss_cost']/1e6:,.1f}M",
        f"Capex ${res['capex0']/1e6:.1f}M initial (incl. soft costs): " + ', '.join(
            f"{k.split(' (')[0]} ${v/1e6:.1f}M" for k, v in res['capex_lines'].items())
        + f"; inverters {res['inverter_kW']/1e3:.1f} MW, interconnect {res['interconnect_kW']/1e3:.1f} MW "
        f"(charge {res['P_charge_ac_kW']/1e3:.1f} MW); cell replacement ${res['rep_cost']/1e6:.1f}M each"
        + (f"; heater element ${res['heater_cost_each']:,.0f} each" if 'heater_cost_each' in res else '')
        + (f"; cost scenario {res['cost_scenario'].upper()}" if res.get('cost_scenario') else ''),
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
