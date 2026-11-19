"""One full charge -> hold -> discharge cycle of the plant, with an energy ledger.

Uses the YAML design as it stands: no block resizing, no heater-cavity search.

  charge     heater cell (build_heater, heater.area_ratio from the YAML) from a uniform
             T_min until the volume-weighted block average reaches T_max; roof + floor
             losses on every cell (CellLossSink), side + edge losses of the plant at the
             mid temperature for the charge time (bought on top of the heater energy)
  hold       heaters hold the block at T_max: every loss path at T_max for
             t_hold = 8760 / cycles_per_year - t_charge - t_discharge
  discharge  the model notebook's main run (sizing.run_discharge: stick in, roof + floor
             losses on, from a uniform T_max until the block average reaches T_min);
             side + edge losses come out of the heat the stick would have absorbed
  plant      tank sized as the optimizer does (AC stick, N = ceil(P_rated / P_stick_AC)),
             inverter efficiencies, floor-cooling fans all cycle, heat-rejection fans on
             the heat the sticks reject

The ledger splits the AC bought per cycle into where it ends up. Steady cycle: the heaters
put back the heat the discharge released plus the losses (the stick comes out when the
volume-weighted block average reaches T_min, so this is ~rho cp V (T_max - T_min) less the
cutoff overshoot). The uniform-start charge run is used for the charge time and the
charge-phase roof + floor losses; `left_in_block` = full-swing heat - heat released.
"""
import copy
import dataclasses
from types import SimpleNamespace

import numpy as np

from .build import build_heater
from .sizing import run_discharge
from . import tank as tk

C2K = 273.15
J_MWh = 1 / 3.6e9


def _charge(m, cfg, end_hr):
    hm = build_heater(m)
    sink = tk.cell_loss_sink(m, cfg['tank']).for_storage(hm.storage)
    r = hm.sim.run_heating_simulation(hm.heater, end_time=end_hr * 3600, Q_sink_fn=sink, **hm.run_kwargs)
    t, Ta = np.asarray(r['time'], float), np.asarray(r['T_avg'], float)
    done = Ta[-1] >= hm.T_max
    t_c = float(np.interp(hm.T_max, Ta[-2:], t[-2:])) if done else float(t[-1])
    dt = np.diff(t)
    return hm, r, dict(t_charge=t_c, done=bool(done), E_heater=float(np.sum(r['Q_heater'][1:] * dt)),
                       E_sink=float(np.sum(r['Q_sink'][1:] * dt)))


def run_cycle(cfg, opt=None, degradation=None, verbose=True):
    """Full cycle for the design in `cfg` (a loaded YAML). `opt` overrides
    thermal.optimize.OPT_DEFAULTS (efficiencies, cycles, parasitics); `degradation`
    e.g. {'cell': 0.05, 'bare': 0.005} for a part-worn stick (default clean).
    -> SimpleNamespace(RTE, ledger, groups, N, times, runs, tank, ...)"""
    from . import optimize as O
    opt = O.merge(O.OPT_DEFAULTS, opt or {})
    eco, par = opt['economics'], opt['parasitics']
    c = copy.deepcopy(cfg)
    if degradation:
        c['degradation'] = degradation
    cycle_hr = 8760.0 / eco['cycles_per_year']
    say = print if verbose else (lambda *a, **k: None)

    # discharge (stick cell)
    say("discharge ...", flush=True)
    m, rd = run_discharge(c)
    st = tk.stick_from_run(m, rd)
    md = O.discharge_metrics(m, rd, opt)
    t_dis = st.t_discharge
    on = np.asarray(rd['ps_active'][1:], bool)
    dt_d = np.diff(rd['time'])
    E_rad = float(np.sum((rd['Q_rad'][1:] * dt_d)[on]))      # heat into the stick
    E_cell = float(np.sum((rd['Q_cell'][1:] * dt_d)[on]))    # absorbed by the active cells
    E_e = st.E_stick_e                                        # DC
    # charge (heater cell)
    say("charge ...", flush=True)
    hm, rc, ch = _charge(m, c, cycle_hr - t_dis / 3600)
    t_ch = ch['t_charge']
    if not ch['done']:
        say(f"WARNING: charge did not reach T_max within {cycle_hr - t_dis/3600:.0f} h "
            f"(heater.area_ratio {hm.area_ratio}); hold set to 0")
    # plant: tank sized on the AC stick (as the optimizer)
    say("tank ...", flush=True)
    eo, ei = eco['eta_out'], eco['eta_in']
    t1 = tk.tank_system(st, c['tank'])
    pf1 = t1.stick_correction['penalty_frac']
    E_ac1 = E_e * (1 - pf1) * eo
    tank = tk.tank_system(dataclasses.replace(st, P_stick=E_ac1 / t_dis, E_stick_e=E_ac1), c['tank'])
    hz, geo, ct = tank.hz, tank.geo, tank.cfg
    N = hz.N_ps
    E_side_dis = tank.stick_correction['E_loss_per_stick_kWh'] * 3.6e6     # per stick, side + edges
    t_hold = max(cycle_hr * 3600 - t_ch - t_dis, 0.0) if ch['done'] else 0.0
    T_max_C, T_min_C = st.T_max, st.T_min
    T_mid = 0.5 * (T_max_C + T_min_C)
    L_mid = tk.losses(T_mid, T_mid, T_mid, geo, hz, tank.cols, ct)
    L_hot = tk.losses(T_max_C, T_max_C, T_max_C, geo, hz, tank.cols, ct)

    # ---- ledger [J per cycle, plant] --------------------------------------------
    E_ch_side = (L_mid['Q_side'] + L_mid['Q_edge']) * t_ch
    E_hold = {k: L_hot[k] * t_hold for k in ('Q_side', 'Q_edge', 'Q_top', 'Q_bot')}
    E_released = md['E_released']                                            # per stick, discharge
    heat_in = N * (E_released + ch['E_sink']) + E_ch_side + sum(E_hold.values())   # heaters -> block
    AC_in = heat_in / ei
    k = 1 - E_side_dis / E_rad                       # side + edge heat comes out of the stick's share
    DC = N * k * E_e
    E_rej = par['heat_rejection_kWe_per_kWth'] * N * md['E_rej']
    E_floor = L_hot['P_fan'] * cycle_hr * 3600
    net = DC * eo - E_floor - E_rej
    rows = [
        ('power electronics', 'heater supply (1 - eta_in)', AC_in - heat_in),
        ('insulation: charge', 'roof + floor (cell losses)', N * ch['E_sink']),
        ('insulation: charge', 'side + edges', E_ch_side),
        ('insulation: hold', 'side', E_hold['Q_side']),
        ('insulation: hold', 'edges', E_hold['Q_edge']),
        ('insulation: hold', 'roof', E_hold['Q_top']),
        ('insulation: hold', 'floor', E_hold['Q_bot']),
        ('insulation: discharge', 'roof + floor (cell losses)', N * st.E_sink),
        ('insulation: discharge', 'side + edges', N * E_side_dis),
        ('stick', 'absorbed on bare + inactive area', N * k * (E_rad - E_cell)),
        ('stick', 'cell conversion loss', N * k * (E_cell - E_e)),
        ('power electronics', 'inverter (1 - eta_out)', DC * (1 - eo)),
        ('parasitics', 'floor-cooling fans (all cycle)', E_floor),
        ('parasitics', 'heat-rejection fans + pumps', E_rej),
        ('output', 'net AC out', net),
    ]
    resid = AC_in - sum(r[2] for r in rows)
    assert abs(resid) < 1e-6 * AC_in, f"ledger does not close: {resid * J_MWh:.3f} MWh"
    groups = {}
    for g, _, v in rows:
        groups[g] = groups.get(g, 0.0) + v
    RTE = net / AC_in
    res = SimpleNamespace(
        RTE=RTE, RTE_gross=DC * eo / AC_in, AC_in=AC_in, net=net, ledger=rows, groups=groups, N=N,
        t_charge_hr=t_ch / 3600, t_hold_hr=t_hold / 3600, t_discharge_hr=t_dis / 3600, cycle_hr=cycle_hr,
        charge_done=ch['done'], m=m, hm=hm, r_dis=rd, r_chg=rc, stick=st, tank=tank, flux=md,
        T_min_C=T_min_C, T_max_C=T_max_C, hold_loss_W=L_hot['Q_total'], P_floor_fan=L_hot['P_fan'], opt=opt,
        left_in_block=st.E_stick_th - E_released, E_released=E_released,
        T_end_vol=st.T_avg[-1] if len(st.T_avg) else np.nan)
    if verbose:
        print(report(res))
    return res


def report(res):
    """Printable ledger (str)."""
    A = res.AC_in
    L = [f"Plant: {res.N} sticks + {res.N} heaters; block {res.T_min_C:.0f}-{res.T_max_C:.0f} C; cycle "
         f"{res.cycle_hr:.1f} h = charge {res.t_charge_hr:.1f} + hold {res.t_hold_hr:.1f} + discharge "
         f"{res.t_discharge_hr:.1f} h" + ('' if res.charge_done else '  (CHARGE INCOMPLETE)'),
         f"Round trip: {res.RTE*100:.1f}% net ({res.RTE_gross*100:.1f}% before fans);  "
         f"AC in {A*J_MWh:,.0f} MWh, net AC out {res.net*J_MWh:,.0f} MWh per cycle",
         "",
         f"{'':24s}{'':38s}{'MWh/cycle':>11s}{'kWh/stick':>11s}{'% of AC in':>11s}"]
    last = None
    for g, lab, v in res.ledger:
        L.append(f"{(g if g != last else ''):24s}{lab:38s}{v*J_MWh:11,.1f}{v/res.N/3.6e6:11,.0f}{v/A*100:10.1f}%")
        last = g
    L += ["", f"Heat released per stick on discharge {res.E_released/3.6e6:,.0f} kWh; full swing "
          f"{res.T_max_C:.0f} -> {res.T_min_C:.0f} C; block average at stick removal {res.T_end_vol:.0f} C; "
          f"{res.left_in_block/3.6e6:,.0f} kWh per stick of the full swing not released."]
    L += ["", "Totals by group:"]
    for g, v in sorted(res.groups.items(), key=lambda kv: -kv[1]):
        L.append(f"  {g:24s}{v*J_MWh:11,.1f} MWh  {v/A*100:5.1f}%")
    return '\n'.join(L)
