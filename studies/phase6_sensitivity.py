"""Phase 6: numerical sensitivity of the tinless axial model.

One-at-a-time variations around the YAML baseline (radiation.axial forced true):
  axial    storage.num_ax_segments (total block rings over block_height)
  radial   storage.num_rad_segments
  dt       all time steps scaled together (dt_short, dt_medium, dt_long)
  dtlim    time step while the vanes sit fully closed (default: dt_short)
  rays     radiation.vane_rays (vane view-factor ray count)
Each case is a full discharge to T_avg = T_min. Metrics are appended to
studies/phase6_results.jsonl as each case finishes; run with --report to print
the convergence tables (relative change vs the finest setting).

    python studies/phase6_sensitivity.py            # run everything (hours)
    python studies/phase6_sensitivity.py axial dt   # only some groups
    python studies/phase6_sensitivity.py --report
"""
import json
import os
import sys
import time
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from thermal.build import build_tinless, load_config  # noqa: E402

OUT = os.path.join(ROOT, 'studies', 'phase6_results.jsonl')
BASE = {'radiation.axial': True, 'radiation.vf_cache_dir': os.path.join(ROOT, '.vf_cache')}

GROUPS = {
    # name: (list of (label, overrides, run-kwarg overrides)), ordered cheap -> expensive
    'radial': [(f'nrad={n}', {'storage.num_rad_segments': n}, {}) for n in (5, 10, 20, 40)],
    'dtlim':  [(f'dt_at_limit={k}', {}, {'dt_at_shade_limit': k}) for k in ('short', 'medium', 'long')],
    'dt':     [(f'dt x{s}', {}, {'dt_scale': s}) for s in (2.0, 1.0, 0.5, 0.25)],
    'rays':   [(f'rays=2^{e}', {'radiation.vane_rays': 2 ** e}, {}) for e in (12, 14, 16)],
    'axial':  [(f'nax={n}', {'storage.num_ax_segments': n}, {}) for n in (10, 20, 30, 45, 60)],
}


def metrics(m, r, wall):
    t, dt = r['time'], np.diff(r['time'])
    on = r['ps_active'][1:]
    t_end = r['t_ps_removed'] if r['t_ps_removed'] is not None else t[-1]
    ax = m.sim.axial_model
    tgt = m.sim.nominal_cell_flux * ax.A_cell_cm2
    P = r['P_elec'][1:]
    E_el = float(np.sum((P * dt)[on]))
    held = np.abs(r['Q_cell'][1:] - tgt) < 1e-3 * tgt
    over = r['Q_cell'][1:] > tgt * 1.001
    E_rad = float(np.sum((r['Q_rad'][1:] * dt)[on]))
    E_low = float(np.nansum((r['Q_rad_low'][1:] * dt)[on]))
    snap = m.storage.block.temp_dist
    return dict(t_end_h=t_end / 3600, E_elec_kWh=E_el / 3.6e6, P_avg_kW=E_el / t_end / 1e3,
                P_peak_kW=float(np.max(r['P_elec'])) / 1e3,
                hold_h=float(np.sum((dt * held)[on])) / 3600, over_h=float(np.sum((dt * over)[on])) / 3600,
                low_share=E_low / E_rad if E_rad else np.nan,
                spread_max_K=float(np.nanmax(r['T_wall_spread'])), spread_end_K=float(snap[:, 0].max() - snap[:, 0].min()),
                q_cap_peak=float(np.nanmax(r['q_cap'])) if np.any(np.isfinite(r['q_cap'])) else np.nan,
                T_LB_max=float(np.nanmax(r['T_LB'])) if np.any(np.isfinite(r['T_LB'])) else np.nan,
                n_steps=int(len(t)), wall_s=wall)


def run_case(cfg, group, label, ov, rk):
    rk = dict(rk)
    m = build_tinless(cfg, dict(BASE, **ov))
    kw = dict(m.run_kwargs, stop_at_cutoff=True)
    s = rk.pop('dt_scale', 1.0)
    for k in ('dt_short', 'dt_medium', 'dt_long'):
        kw[k] = kw[k] * s
    lim = rk.pop('dt_at_shade_limit', None)
    if lim is not None:
        kw['dt_at_shade_limit'] = kw[f'dt_{lim}']
    t0 = time.perf_counter()
    r = m.sim.run_simulation(**kw)
    res = metrics(m, r, time.perf_counter() - t0)
    res.update(group=group, label=label, n_ax=int(m.mesh.n_ax), n_rad=int(m.storage.num_rad_segments),
               H=float(m.H_block), L=float(m.L))
    with open(OUT, 'a', encoding='utf-8') as f:
        f.write(json.dumps(res) + '\n')
    print(f"[{group}] {label}: t_end {res['t_end_h']:.3f} h, E_elec {res['E_elec_kWh']:.1f} kWh, "
          f"hold {res['hold_h']:.2f} h, over {res['over_h']:.2f} h, P_peak {res['P_peak_kW']:.1f} kW, "
          f"low share {res['low_share']:.3f}, spread max {res['spread_max_K']:.1f} K, "
          f"q_cap peak {res['q_cap_peak']:.2f}  [{res['wall_s']:.0f} s, {res['n_steps']} steps]", flush=True)


def report():
    rows = [json.loads(l) for l in open(OUT, encoding='utf-8')]
    keys = ('t_end_h', 'E_elec_kWh', 'hold_h', 'over_h', 'P_peak_kW', 'low_share', 'spread_max_K', 'q_cap_peak')
    for g in GROUPS:
        rs = [r for r in rows if r['group'] == g]
        if not rs:
            continue
        ref = rs[-1]          # last = finest setting in GROUPS order
        print(f"\n== {g}  (relative change vs {ref['label']})")
        print(f"{'case':16s}" + ''.join(f"{k:>14s}" for k in keys) + f"{'run s':>8s}")
        for r in rs:
            cells = []
            for k in keys:
                v, v0 = r[k], ref[k]
                if k in ('hold_h', 'over_h', 'spread_max_K'):
                    cells.append(f"{v:9.2f} ({v - v0:+.2f})")       # absolute diff for small quantities
                else:
                    cells.append(f"{v:8.2f} ({(v / v0 - 1) * 100:+.1f}%)" if v0 else f"{v:14.3f}")
            print(f"{r['label']:16s}" + ''.join(f"{c:>14s}" for c in cells) + f"{r['wall_s']:8.0f}")


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    sys.stdout.reconfigure(encoding='utf-8')
    if '--report' in sys.argv:
        report()
        sys.exit()
    groups = [a for a in sys.argv[1:] if a in GROUPS] or list(GROUPS)
    # baseline = the YAML in the notebook's input cell (a8c3d501); falls back to the file
    try:
        import re
        import yaml
        nb = json.load(open(os.path.join(ROOT, 'tinless system model.ipynb'), encoding='utf-8'))
        src = ''.join(next(c for c in nb['cells'] if c.get('id') == 'a8c3d501')['source'])
        cfg = yaml.safe_load(re.search(r'yaml_content = """(.*?)"""', src, re.S).group(1))
    except Exception:
        cfg = load_config(os.path.join(ROOT, 'tinless_model_input.yaml'))
    print(f"Baseline YAML: L {cfg['storage']['L']} m, block_height {cfg['storage']['block_height']} m, "
          f"num_ax {cfg['storage']['num_ax_segments']}, num_rad {cfg['storage']['num_rad_segments']}, "
          f"shade {cfg['shade'].get('model')}, pad {cfg['liner_bottom']['enabled']}; groups {groups}", flush=True)
    for g in groups:
        for label, ov, rk in GROUPS[g]:
            try:
                run_case(cfg, g, label, ov, rk)
            except Exception as e:                       # keep going overnight
                print(f"[{g}] {label}: FAILED {type(e).__name__}: {e}", flush=True)
    report()
