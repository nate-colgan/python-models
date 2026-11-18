"""Phase 7: confirmation of the recommended numerical settings + the brief's §7 first runs.

  confirm  the two recommended setting combinations at the YAML baseline (H = 10 m),
           compared with the finest phase-6 run (nax = 60):
             detailed : 0.33 m rings, 5 radial, dt x1, long step at the shade limit, 2^14 rays
             screening: 0.50 m rings, 5 radial, dt x2, long step at the shade limit, 2^12 rays
  height   H_block in {L, 1.5L, 2L} (+ the YAML 10 m), shaded (vanes) and unshaded,
           screening settings; charge time per height (central heater, full block height)
  pad      liner_bottom.rgf_thickness in {0, 12.5, 25, 50} mm at H = 2L and H = 10 m

Unshaded cases use the same cavity as the shaded ones (area_ratio taken from the
shaded build), so only the shade differs. Results -> studies/phase7_results.jsonl.

    python studies/phase7_runs.py                 # everything
    python studies/phase7_runs.py height pad      # some groups
    python studies/phase7_runs.py --report
"""
import json
import os
import sys
import time
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'studies'))
from thermal.build import build_tinless, load_config      # noqa: E402
from thermal.tinless import HeatingElement                # noqa: E402
from phase6_sensitivity import metrics as base_metrics    # noqa: E402

OUT = os.path.join(ROOT, 'studies', 'phase7_results.jsonl')
P6 = os.path.join(ROOT, 'studies', 'phase6_results.jsonl')
BASE = {'radiation.axial': True, 'radiation.vf_cache_dir': os.path.join(ROOT, '.vf_cache')}

DETAILED = dict(dz=1 / 3, nrad=5, dt_scale=1.0, lim='long', rays=2 ** 14)
SCREEN = dict(dz=0.5, nrad=5, dt_scale=2.0, lim='long', rays=2 ** 12)


def n_rings(H, dz):
    return max(10, int(round(H / dz)))


def overrides(H, num, extra=None):
    ov = dict(BASE, **{'storage.block_height': H, 'storage.num_ax_segments': n_rings(H, num['dz']),
                       'storage.num_rad_segments': num['nrad'], 'radiation.vane_rays': num['rays']})
    ov.update(extra or {})
    return ov


def metrics(m, r, wall):
    res = base_metrics(m, r, wall)
    t, dt = r['time'], np.diff(r['time'])
    on = r['ps_active'][1:]
    E_cell = float(np.sum((r['Q_cell'][1:] * dt)[on]))
    E_rad = float(np.sum((r['Q_rad'][1:] * dt)[on]))
    E_el = res['E_elec_kWh'] * 3.6e6
    T0 = m.T_max
    res.update(eff_cell=E_el / E_cell if E_cell else np.nan,     # electricity / heat into cells
               eff_block=E_el / E_rad if E_rad else np.nan,      # electricity / heat leaving the block
               q_cap_avg=float(np.nanmean(r['q_cap'][1:][on])) if np.any(np.isfinite(r['q_cap'])) else np.nan,
               T_LB_min=float(np.nanmin(r['T_LB'])) if np.any(np.isfinite(r['T_LB'])) else np.nan)
    for k in ('Q_vert_rad', 'Q_vert_cond', 'Q_rad_low'):
        v = r.get(k)
        if v is not None and np.any(np.isfinite(v)):
            res[f'{k}_kWh'] = float(np.nansum((np.asarray(v)[1:] * dt)[on])) / 3.6e6
            res[f'{k}_peak_kW'] = float(np.nanmax(np.asarray(v))) / 1e3
    return res


def charge(m, cfg, num):
    """Charge from T_min to T_max with the YAML heater spanning the full block height."""
    h = cfg['heater']
    st = m.storage
    st.block.temp_dist = np.full_like(st.block.temp_dist, float(m.T_min))
    d = 2 * (2 * h['side_len'] / np.pi)
    if d / 2 >= st.block.rad_in:
        return dict(charge_h=np.nan, charge_note='heater does not fit the cavity')
    heater = HeatingElement(diameter=d, max_temperature=h['max_temperature'],
                            max_power=h['max_power'], emissivity=h['emissivity'])
    kw = m.run_kwargs
    r = m.sim.run_heating_simulation(heater=heater, T_block_max=m.T_max, end_time=400 * 3600,
                                     dt=kw['dt_short'] * num['dt_scale'], sub_steps=kw['sub_steps'])
    return dict(charge_h=float(r['time'][-1] / 3600), charge_Pavg_kW=float(np.mean(r['Q_heater'][1:]) / 1e3),
                charge_T_heater_max=float(r['T_heater'].max()))


def run(cfg, group, label, ov, num, do_charge=False, extra=None):
    m = build_tinless(cfg, ov)
    kw = dict(m.run_kwargs, stop_at_cutoff=True)
    for k in ('dt_short', 'dt_medium', 'dt_long'):
        kw[k] = kw[k] * num['dt_scale']
    kw['dt_at_shade_limit'] = kw[f"dt_{num['lim']}"]
    t0 = time.perf_counter()
    r = m.sim.run_simulation(**kw)
    res = metrics(m, r, time.perf_counter() - t0)
    res.update(group=group, label=label, n_ax=int(m.mesh.n_ax), n_rad=int(m.storage.num_rad_segments),
               H=float(m.H_block), L=float(m.L), shaded=bool(m.shade_bool), area_ratio=float(m.area_ratio),
               t_rgf=float(m.liner_bottom['rgf_thickness'] or 0.0), **(extra or {}))
    if do_charge:
        res.update(charge(m, cfg, num))
    with open(OUT, 'a', encoding='utf-8') as f:
        f.write(json.dumps(res) + '\n')
    print(f"[{group}] {label}: t_end {res['t_end_h']:.2f} h, E_elec {res['E_elec_kWh']:.0f} kWh, "
          f"hold {res['hold_h']:.2f} h, P_avg {res['P_avg_kW']:.1f} / peak {res['P_peak_kW']:.1f} kW, "
          f"spread {res['spread_max_K']:.0f} K, q_cap {res['q_cap_peak']:.2f}"
          + (f", charge {res.get('charge_h', np.nan):.2f} h" if do_charge else '')
          + f"  [{res['wall_s']:.0f} s]", flush=True)
    return m


def g_confirm(cfg):
    H = float(cfg['storage']['block_height'] or cfg['storage']['L'])
    for name, num in (('detailed', DETAILED), ('screening', SCREEN)):
        run(cfg, 'confirm', name, overrides(H, num), num)


def g_height(cfg):
    L = float(cfg['storage']['L'])
    for H in (L, 1.5 * L, 2 * L, float(cfg['storage']['block_height'] or L)):
        ov = overrides(H, SCREEN, {'shade.shade_bool': True})
        ar = build_tinless(cfg, ov, geometry_only=True).area_ratio
        run(cfg, 'height', f'H={H:g} shaded', ov, SCREEN, do_charge=True)
        run(cfg, 'height', f'H={H:g} unshaded',
            overrides(H, SCREEN, {'shade.shade_bool': False, 'powerstick.area_ratio': ar}), SCREEN)


def g_pad(cfg):
    L = float(cfg['storage']['L'])
    for H in (2 * L, float(cfg['storage']['block_height'] or L)):
        for t in (0.0, 0.0125, 0.025, 0.05):
            ov = overrides(H, SCREEN, {'liner_bottom.enabled': True, 'liner_bottom.rgf_thickness': t})
            try:
                run(cfg, 'pad', f'H={H:g} rgf={t * 1e3:g}mm', ov, SCREEN)
            except Exception as e:        # a zero-thickness pad may not be solvable: use 0.1 mm
                if t > 0:
                    raise
                print(f"  rgf=0 failed ({type(e).__name__}: {e}); using 0.1 mm", flush=True)
                ov['liner_bottom.rgf_thickness'] = 1e-4
                run(cfg, 'pad', f'H={H:g} rgf=0.1mm', ov, SCREEN)


GROUPS = {'confirm': g_confirm, 'height': g_height, 'pad': g_pad}


def report():
    rows = [json.loads(l) for l in open(OUT, encoding='utf-8')]
    conf = [r for r in rows if r['group'] == 'confirm']
    if conf and os.path.exists(P6):
        ref = [json.loads(l) for l in open(P6, encoding='utf-8')]
        ref = next(r for r in ref if r['label'] == 'nax=60')
        keys = ('t_end_h', 'E_elec_kWh', 'hold_h', 'P_peak_kW', 'low_share', 'spread_max_K', 'q_cap_peak')
        print(f"\n== confirm (vs phase-6 nax=60: " + ', '.join(f"{k} {ref[k]:.3g}" for k in keys) + ')')
        for r in conf:
            print(f"{r['label']:10s} nax {r['n_ax']:3d}  " + '  '.join(
                f"{k} {(r[k] / ref[k] - 1) * 100:+.1f}%" if k not in ('hold_h', 'spread_max_K')
                else f"{k} {r[k] - ref[k]:+.2f}" for k in keys) + f"  [{r['wall_s']:.0f} s]")
    cols = [('t_end_h', 'disch h', '.2f'), ('hold_h', 'hold h', '.2f'), ('over_h', 'over h', '.2f'),
            ('P_avg_kW', 'P avg', '.1f'), ('P_peak_kW', 'P peak', '.1f'), ('E_elec_kWh', 'E_el kWh', '.0f'),
            ('eff_cell', 'eff cell', '.3f'), ('eff_block', 'eff blk', '.3f'), ('spread_max_K', 'spread', '.0f'),
            ('low_share', 'low shr', '.3f'), ('Q_vert_rad_kWh', 'vrad kWh', '.0f'),
            ('Q_vert_cond_kWh', 'vcond kWh', '.0f'), ('q_cap_avg', 'qcap avg', '.2f'),
            ('q_cap_peak', 'qcap pk', '.2f'), ('T_LB_min', 'T_LB min', '.0f'), ('T_LB_max', 'T_LB max', '.0f'),
            ('charge_h', 'charge h', '.2f'), ('wall_s', 'run s', '.0f')]
    for g in ('height', 'pad'):
        rs = [r for r in rows if r['group'] == g]
        if not rs:
            continue
        print(f"\n== {g}")
        print(f"{'case':20s}" + ''.join(f"{c[1]:>10s}" for c in cols))
        for r in rs:
            cells = []
            for k, _, fmt in cols:
                v = r.get(k)
                cells.append(f"{v:{fmt}}" if isinstance(v, (int, float)) and np.isfinite(v) else '-')
            print(f"{r['label']:20s}" + ''.join(f"{c:>10s}" for c in cells))


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    sys.stdout.reconfigure(encoding='utf-8')
    if '--report' in sys.argv:
        report()
        sys.exit()
    groups = [a for a in sys.argv[1:] if a in GROUPS] or list(GROUPS)
    try:
        import re
        import yaml
        nb = json.load(open(os.path.join(ROOT, 'tinless system model.ipynb'), encoding='utf-8'))
        src = ''.join(next(c for c in nb['cells'] if c.get('id') == 'a8c3d501')['source'])
        cfg = yaml.safe_load(re.search(r'yaml_content = """(.*?)"""', src, re.S).group(1))
    except Exception:
        cfg = load_config(os.path.join(ROOT, 'tinless_model_input.yaml'))
    print(f"Baseline: L {cfg['storage']['L']} m, block_height {cfg['storage']['block_height']} m, "
          f"pad {cfg['liner_bottom']['enabled']} ({cfg['liner_bottom'].get('rgf_thickness')} m); groups {groups}",
          flush=True)
    for g in groups:
        try:
            GROUPS[g](cfg)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[{g}] FAILED {type(e).__name__}: {e}", flush=True)
    report()
