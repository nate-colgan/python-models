"""
Batch of optimizer cases: the whole optimizer-notebook pipeline for each case, in a loop.

For every case (cost scenario x electricity price, on top of the notebook's OPT_INPUTS):
  1. optimizer inputs (O.build_opt)
  2. heater lead table for the case's costs and price (heater_leads.build_lead_table; cached in
     opt_runs/_lead_tables by an input hash, copied into the run folder)
  3. differential evolution (O.run_de), logged to opt_runs/<time>_p<price>_<scenario>[_layout]/;
     optionally seeded from earlier runs and from the batch's earlier cases of the same scenario
  4. results: best designs, Pareto / capex figure, variable scatter, cell-flux band of the optimum
  5. detailed-settings check of the top n_verify designs
  6. full charge-hold-discharge cycle of the optimum at detailed settings (+ layout, time-series and
     loss figures)
  7. full lead optimization of the optimum's heater leads (heater_leads.lead_for_design)
  8. run report (report.write_report) in the run folder
Each step after the optimizer run is guarded: a failure is logged and the case goes on to the
report with what it has; the batch goes on to the next case.

Use from the notebook (cell opt_batch: run_cases(...)), or as a separate process that does not
depend on the notebook kernel:
    python -m thermal.batch <batch_inputs.json>
where the json holds dict(cases, base_cfg, opt_inputs, lead_inputs, lead_table, settings).
"""
import copy
import json
import os
import sys
import time
import traceback

C2K = 273.15

SETTINGS_DEFAULTS = dict(
    workers=20, popsize=10, maxiter=25, seed=0,
    seed_runs=[],                 # run folders whose best designs seed every case (and the sizing warm start)
    seed_previous=True,           # also seed each case from the batch's earlier cases of the same scenario
    use_lead_table=True,          # heater leads modelled (lead table per case); False: not modelled
    rebuild_lead_table=False,     # False: reuse a saved lead table built with exactly the case's inputs
    n_verify=3,                   # top designs re-checked at detailed settings
    cycle_hr=168.0,               # full-cycle check: cycle length [h]
    detailed_fidelity=None,       # fidelity of the verify and full-cycle steps; None = O.DETAILED
    run_cycle=True, run_lead_detail=True, run_verify=True,
    yaml_path='tinless_model_input.yaml', notebook='tinless system optimizer.ipynb',
)


def case_name(scenario, price):
    return f"{scenario or 'inputs'} ${price:g}/MWh"


def _say(log, *a):
    msg = ' '.join(str(x) for x in a)
    print(msg, flush=True)
    if log:
        with open(log, 'a', encoding='utf-8') as fh:
            fh.write(time.strftime('%H:%M:%S ') + msg + '\n')


def _run_dir(opt):
    from . import optimize as O
    p = opt['layout']['pattern']
    return os.path.join('opt_runs', time.strftime('%Y%m%d_%H%M%S') + f"_p{opt['economics']['elec_price']:.0f}"
                        + (f"_{O.scenario_name(opt)}" if O.scenario_name(opt) else '')
                        + ('' if p == 'checkerboard' else '_' + (p if isinstance(p, str) else '-'.join(p))))


def _figures(figs, opt, rows, best, base_cfg, say):
    """The results-cell figures: Pareto / capex breakdown, variable scatter, cell-flux band."""
    import numpy as np
    import matplotlib.pyplot as plt
    from . import optimize as O
    ok = [r for r in rows if r.get('feasible')]
    pool = [r for r in rows if 'capex_total' in r and set(r.get('violations') or {}) <= {'rte'}]
    front = O.pareto(rows)
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
    if pool:
        ax[0].scatter([r['RTE'] * 100 for r in pool], [r['capex_total'] / 1e6 for r in pool], s=8, alpha=.35,
                      label='feasible except RTE floor')
        ax[0].plot([r['RTE'] * 100 for r in front], [r['capex_total'] / 1e6 for r in front], 'o-', c='C3', ms=4,
                   label='Pareto front')
        ax[0].axvline(opt['objective']['rte_min'] * 100, c='k', ls=':', lw=.8, label='this run RTE floor')
    ax[0].set_xlabel('net round-trip efficiency [%]'); ax[0].set_ylabel('lifetime capex [$M]')
    ax[0].grid(alpha=.3); ax[0].legend(fontsize=8); ax[0].set_title('Capex vs efficiency')
    if best:
        b = best[0]
        parts = {k.split(' (')[0]: v * (1 + opt['economics']['soft_cost_frac']) for k, v in b['capex_lines'].items()}
        parts['cell replacements'] = b['n_rep'] * b['rep_cost']
        parts['bare cleanings'] = b['n_clean'] * b['clean_cost']
        ax[1].barh(list(parts), np.array(list(parts.values())) / 1e6)
        ax[1].set_xlabel('$M incl. soft costs'); ax[1].invert_yaxis(); ax[1].grid(alpha=.3, axis='x')
        ax[1].set_title(f"Best: capex \\${b['capex_total']/1e6:,.1f}M, RTE {b['RTE']*100:.1f}% "
                        f"(+ energy lost \\${b['loss_cost']/1e6:,.1f}M)")
    plt.tight_layout()
    figs.keep('pareto_and_capex', 'Left: lifetime capex vs net round-trip efficiency of every feasible design, with '
                                  'the Pareto front. Right: capex breakdown of the optimum (incl. soft costs).', fig=fig)
    plt.close(fig)
    if ok:
        keys = ['liner_side', 'ps_side', 'L', 'H_block', 'avg_flux', 'shade_thickness', 't_side', 't_top',
                'L_bot', 'T_max_C', 'T_min_C', 'swing']
        fig, axs = plt.subplots(3, 5, figsize=(19, 10))
        v = np.array([r['lifetime_cost'] for r in ok]) / 1e6
        e = np.array([r['RTE'] for r in ok]) * 100
        sc = None
        for a, k in zip(axs.flat, keys):
            sc = a.scatter([r['design'].get(k, r.get(k)) for r in ok], v, c=e, s=8, alpha=.6, cmap='viridis')
            a.set_xlabel(k); a.grid(alpha=.3)
        for a in axs.flat[len(keys):]:
            a.axis('off')
        fig.colorbar(sc, ax=axs.flat[-1], label='net RTE [%]')
        for a in axs[:, 0]:
            a.set_ylabel('capex + energy lost [$M]')
        plt.tight_layout()
        figs.keep('variable_scatter', 'Objective (lifetime capex + energy lost) of every feasible design vs each '
                                      'optimized variable and the sizing outputs T_min and swing; colour = net RTE. A '
                                      'tight lower edge means the variable matters; a flat one means it does not.', fig=fig)
        plt.close(fig)
    if best:
        r = best[0]
        cfg = O.design_cfg(base_cfg, r['design'], opt)
        cfg['heater']['area_ratio'] = r.get('heater_area_ratio', cfg['heater'].get('area_ratio'))
        cfg['storage']['T_min'] = r['T_min_C'] + C2K
        cfg['storage']['block_temp_range'] = r['swing']
        cfg['storage']['volume_factor'] = r['volume_factor']
        m, run, _, fm = O._discharge(cfg, opt, deposition=False)
        ps = m.powerstick
        A = 4 * ps.width * ps.height * (1 - ps.f_bare) * ps.f_active * 1e4
        t = run['time'] / 3600
        fig, ax = plt.subplots(1, 2, figsize=(14, 4))
        ax[0].plot(t, run['Q_cell'] / A, label='cell flux')
        ax[0].axhline(m.avg_flux, c='k', ls=':', label='target')
        ax[0].set_xlabel('time [h]'); ax[0].set_ylabel('W/cm2'); ax[0].grid(alpha=.3); ax[0].legend()
        ax[1].plot(t, run['f_shade'], c='C1'); ax[1].set_xlabel('time [h]'); ax[1].set_ylabel('shade closed fraction')
        ax[1].grid(alpha=.3)
        plt.tight_layout()
        figs.keep('flux_band', 'Clean discharge of the optimum (screening settings): cell flux vs the target, and the '
                               'shade closed fraction (1 = closed, 0 = fully open). The shades open fully at the end of '
                               'the discharge by construction (shade-open sizing).', fig=fig)
        plt.close(fig)


def run_case(scenario, price, base_cfg, opt_inputs, lead_inputs, lead_table, settings=None, seed_rows=None,
             log=None):
    """One case of the batch (see the module docstring). -> dict(log_dir, report, best, error per step)"""
    import matplotlib
    import matplotlib.pyplot as plt
    from . import optimize as O
    from . import heater_leads as HL
    from . import report as R
    st = dict(SETTINGS_DEFAULTS, **(settings or {}))
    name = case_name(scenario, price)
    out = dict(case=name, errors={})
    inputs = O.merge(opt_inputs, {'economics': {'cost_scenario': scenario, 'elec_price': float(price)}})
    t_case = time.perf_counter()

    def step(label, fn):
        t0 = time.perf_counter()
        _say(log, f"[{name}] {label} ...")
        try:
            v = fn()
            _say(log, f"[{name}] {label}: done ({(time.perf_counter() - t0) / 60:.1f} min)")
            return v
        except Exception as e:                       # noqa: BLE001
            out['errors'][label] = f"{type(e).__name__}: {e}"
            _say(log, f"[{name}] {label}: FAILED {type(e).__name__}: {e}\n{traceback.format_exc(limit=4)}")
            return None
        finally:
            plt.close('all')

    # 1-2. inputs and heater lead table
    opt = O.build_opt(inputs)
    if st['use_lead_table']:
        lt = step('lead table', lambda: HL.build_lead_table(base_cfg, inputs, lead_inputs, lead_table,
                                                            force=st['rebuild_lead_table']))
        if lt is None:
            return out
    else:
        lt = None
        opt = O.merge(opt, {'leads': {'table': None}})
    log_dir = _run_dir(opt)
    os.makedirs(log_dir, exist_ok=True)
    out['log_dir'] = log_dir
    if lt is not None:
        lt_path = os.path.join(log_dir, 'lead_table.json')
        with open(lt_path, 'w') as fh:
            json.dump(lt.data, fh, indent=1, default=float)
        opt = O.merge(opt, {'leads': {'table': lt_path, 'n_per_heater': lead_inputs.get('n_leads', 2)}})
    _say(log, f"[{name}] run folder {log_dir}")

    # 3. differential evolution
    gen = [0]
    t_run = time.perf_counter()

    def cb(xk, convergence):
        gen[0] += 1
        rows = O.load_log(log_dir)
        feas = [r for r in rows if r.get('feasible')]
        b = min(feas, key=lambda r: r['lifetime_cost']) if feas else None
        _say(log, f"[{name}] gen {gen[0]:3d} {(time.perf_counter() - t_run) / 60:6.1f} min {len(rows)} designs  best "
             + (f"${b['lifetime_cost'] / 1e6:,.2f}M (RTE {b['RTE'] * 100:.1f}%, {b.get('layout')})" if b else 'none feasible')
             + f"  convergence {convergence:.3f}")

    x0 = O.encode(O.design_from_cfg(base_cfg), opt)
    de = step('optimizer', lambda: O.run_de(base_cfg, opt, log_dir, maxiter=st['maxiter'], popsize=st['popsize'],
                                             workers=st['workers'], seed=st['seed'], x0=x0, callback=cb,
                                             seed_rows=seed_rows or None))
    with open(os.path.join(log_dir, 'inputs.json'), 'w') as fh:
        json.dump(dict(opt=opt, base_yaml=st['yaml_path'], batch_case=name,
                       de=dict(fun=None if de is None else float(de.fun),
                               x=[] if de is None else [float(v) for v in de.x])), fh, indent=1, default=str)

    # 4. results and figures
    rows = O.load_log(log_dir)
    ok = [r for r in rows if r.get('feasible')]
    best = sorted(ok, key=lambda r: r['lifetime_cost'])[:10]
    _say(log, f"[{name}] {len(rows)} designs, {len(ok)} feasible, {sum('error' in r for r in rows)} failed")
    if not best:
        out['errors']['results'] = 'no feasible design'
        return out
    for p, v in O.layout_bests(rows).items():
        _say(log, f"[{name}]   {p:13s} ${v['lifetime_cost'] / 1e6:8.2f}M  RTE {v['RTE'] * 100:5.2f}%  "
                  f"{v['N_ps']} sticks + {v['N_h']} heaters")
    figs = R.FigureStore()
    step('results figures', lambda: _figures(figs, opt, rows, best, base_cfg, lambda *a: _say(log, *a)))

    # 5. detailed check of the top designs
    verified = []
    if st['run_verify']:
        def verify():
            for r in best[:st['n_verify']]:
                verified.append((r, O.evaluate(r['design'], base_cfg, opt, fidelity=st['detailed_fidelity'] or O.DETAILED)))
        step('detailed check of the top designs', verify)

    # 6. full cycle of the optimum
    cy = None
    if st['run_cycle']:
        def cycle():
            from .cycle import run_cycle, plot_layout, plot_timeseries, plot_losses
            top = best[0]
            opt_det = O.merge(opt, {'fidelity': st['detailed_fidelity'] or O.DETAILED})
            cfg_top = O.design_cfg(base_cfg, dict(top['design'], T_min_guess_C=top['T_min_C']), opt_det)
            sz = opt_det['sizing']
            c = run_cycle(cfg_top, opt=opt_det, cycle_hr=st['cycle_hr'], charge_hr=opt_det['charge_hr'],
                          resize_block=True, size_heater_cavity=True, verbose=False,
                          resize_kw=dict(tol_T=sz['tol_T'], tol_t=sz['tol_t'], max_iter=sz['max_iter'],
                                         T_min_floor=opt_det['geometry']['T_min_floor_C'] + C2K))
            for fn, key, cap in (
                    (plot_layout, 'cycle_layout', 'Optimum to scale: plan (sticks in square cavities with liner and '
                     'shade vanes, heaters in round cavities, in the run\'s layout pattern) and a vertical section '
                     'through a stick and its nearest heater; dashed circles = the annular cells the model analyses.'),
                    (plot_timeseries, 'cycle_timeseries', 'One week cycle of the optimum (detailed settings, clean '
                     'cells): temperatures, powers per stick / heater, and cell flux with the shade position during '
                     'discharge.'),
                    (plot_losses, 'cycle_losses', 'Where the electricity bought in one cycle goes, as % of AC in '
                     '(detailed settings, clean cells).')):
                fig, _ = fn(c)
                figs.keep(key, cap, fig=fig)
                plt.close(fig)
            return c
        cy = step('full cycle of the optimum (detailed)', cycle)

    # 7. heater leads of the optimum
    lead_detail = None
    if st['run_lead_detail'] and O.lead_table(opt) is not None:
        def leads():
            dd = HL.lead_for_design(base_cfg, inputs, best[0], lead_inputs, lead_table, lt=O.lead_table(opt),
                                    verbose=False)
            dd['rank'] = 0
            return dd
        lead_detail = step('lead design of the optimum', leads)

    # 8. report
    def report():
        return R.write_report(log_dir, dict(
            opt=opt, opt_inputs=inputs, base_cfg=base_cfg, rows=rows, best=best, figs=figs,
            run=dict(RUN_DE=True, WORKERS=st['workers'], POPSIZE=st['popsize'], MAXITER=st['maxiter'], SEED=st['seed'],
                     BATCH_CASE=name),
            de=de, verified=verified or None, cycle=cy, lead_detail=lead_detail, lead_inputs=lead_inputs,
            lead_table_settings=lead_table, yaml_path=st['yaml_path'], notebook=st['notebook']))
    out['report'] = step('report', report)
    b = best[0]
    out['best'] = dict(lifetime_cost=b['lifetime_cost'], capex0=b['capex0'], RTE=b['RTE'], N_ps=b['N_ps'],
                       N_h=b.get('N_h'), layout=b.get('layout'), RTE_cycle=getattr(cy, 'RTE', None))
    _say(log, f"[{name}] case done in {(time.perf_counter() - t_case) / 3600:.2f} h: best ${b['lifetime_cost'] / 1e6:,.2f}M, "
              f"RTE {b['RTE'] * 100:.1f}%, {b['N_ps']} sticks + {b.get('N_h')} heaters ({b.get('layout')}); report "
              f"{out.get('report')}")
    return out


def run_cases(cases, base_cfg, opt_inputs, lead_inputs, lead_table, settings=None, log=None):
    """Run every case [(scenario, price), ...] in order (see the module docstring).
    -> list of run_case results; a summary table is printed and logged at the end."""
    import matplotlib
    if 'ipykernel' not in sys.modules:             # separate process: no display needed
        matplotlib.use('Agg', force=False)
    from . import optimize as O
    st = dict(SETTINGS_DEFAULTS, **(settings or {}))
    log = log or os.path.join('opt_runs', time.strftime('batch_%Y%m%d_%H%M%S.log'))
    os.makedirs(os.path.dirname(log) or '.', exist_ok=True)
    _say(log, f"batch of {len(cases)} cases: " + ', '.join(case_name(s, p) for s, p in cases) + f"; log {log}")
    base_seed = [r for run in st['seed_runs'] for r in O.load_log(run)]
    results = []
    t0 = time.perf_counter()
    for k, (scenario, price) in enumerate(cases, 1):
        _say(log, f"=== case {k}/{len(cases)}: {case_name(scenario, price)} ===")
        seed_rows = list(base_seed)
        if st['seed_previous']:
            for prev in results:
                if prev.get('log_dir') and prev['case'].split(' ')[0] == str(scenario or 'inputs'):
                    seed_rows += O.load_log(prev['log_dir'])
        try:
            results.append(run_case(scenario, price, base_cfg, opt_inputs, lead_inputs, lead_table, st,
                                    seed_rows=seed_rows, log=log))
        except Exception as e:                       # noqa: BLE001
            _say(log, f"case {case_name(scenario, price)} FAILED: {type(e).__name__}: {e}\n{traceback.format_exc()}")
            results.append(dict(case=case_name(scenario, price), errors={'case': repr(e)}))
    _say(log, f"\nbatch done in {(time.perf_counter() - t0) / 3600:.1f} h")
    _say(log, f"{'case':22s} {'objective':>11s} {'capex':>9s} {'RTE':>7s} {'cycle RTE':>10s}  plant / run folder")
    for r in results:
        b = r.get('best')
        if b:
            _say(log, f"{r['case']:22s} ${b['lifetime_cost'] / 1e6:9.2f}M ${b['capex0'] / 1e6:7.2f}M {b['RTE'] * 100:6.2f}% "
                      + (f"{b['RTE_cycle'] * 100:9.2f}%" if b.get('RTE_cycle') else f"{'-':>10s}")
                      + f"  {b['N_ps']} + {b['N_h']} {b['layout']}  {r.get('log_dir')}"
                      + (f"  (failed steps: {', '.join(r['errors'])})" if r.get('errors') else ''))
        else:
            _say(log, f"{r['case']:22s} no result ({r.get('errors')})")
    return results


def launch(cases, base_cfg, opt_inputs, lead_inputs, lead_table, settings=None, path=None):
    """Start run_cases in a separate process (survives the notebook kernel); inputs saved to `path`.
    -> (subprocess.Popen, log path)"""
    import subprocess
    stamp = time.strftime('%Y%m%d_%H%M%S')
    path = path or os.path.join('opt_runs', f'batch_{stamp}_inputs.json')
    log = os.path.join('opt_runs', f'batch_{stamp}.log')
    os.makedirs('opt_runs', exist_ok=True)
    with open(path, 'w') as fh:
        json.dump(dict(cases=[list(c) for c in cases], base_cfg=base_cfg, opt_inputs=opt_inputs,
                       lead_inputs=lead_inputs, lead_table=lead_table, settings=settings or {}, log=log),
                  fh, indent=1, default=str)
    flags = 0
    if os.name == 'nt':
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | 0x00000008          # DETACHED_PROCESS
    out = open(log.replace('.log', '_console.txt'), 'w')
    p = subprocess.Popen([sys.executable, '-u', '-m', 'thermal.batch', path], stdout=out, stderr=subprocess.STDOUT,
                         cwd=os.getcwd(), creationflags=flags)
    return p, log


if __name__ == '__main__':
    import warnings
    warnings.simplefilter('ignore')
    with open(sys.argv[1]) as fh:
        a = json.load(fh)
    run_cases([tuple(c) for c in a['cases']], a['base_cfg'], a['opt_inputs'], a['lead_inputs'], a['lead_table'],
              a.get('settings'), log=a.get('log'))
