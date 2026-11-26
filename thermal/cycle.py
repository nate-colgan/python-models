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
  plant      tank sized as the optimizer does (AC stick, N_ps = the layout's feasible count >=
             ceil(P_rated / P_stick_AC), N_h heaters of the layout, thermal/layout.py; the heater cell
             holds N_ps / N_h of a stick cell's graphite),
             inverter efficiencies, floor-cooling fans all cycle, heat-rejection fans on
             the heat the sticks reject
  leads      with opt['leads']['table'] (as the optimizer): heater current leads from the lead
             table at the heater power as run: Joule heat while charging (bought), net heat
             drawn from the storage (put back by the heaters), rejected heat on the fans

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
from .heater import describe as describe_heater
from .sizing import run_discharge
from . import tank as tk

C2K = 273.15
J_MWh = 1 / 3.6e9


def _charge(m, cfg, end_hr, P=None):
    """Heater-cell charge; P [W] overrides the heater power (else YAML max_power_per_m x H)."""
    if P is not None:
        from .optimize import _with_heater_ar
        m, cfg = _with_heater_ar(m, cfg, cfg['heater']['area_ratio'], P)
    hm = build_heater(m)
    sink = tk.cell_loss_sink(m, cfg['tank']).for_storage(hm.storage)
    r = hm.sim.run_heating_simulation(hm.heater, end_time=end_hr * 3600, Q_sink_fn=sink, **hm.run_kwargs)
    t, Ta = np.asarray(r['time'], float), np.asarray(r['T_avg'], float)
    done = Ta[-1] >= hm.T_max
    t_c = float(np.interp(hm.T_max, Ta[-2:], t[-2:])) if done else float(t[-1])
    dt = np.diff(t)
    return hm, r, dict(t_charge=t_c, done=bool(done), E_heater=float(np.sum(r['Q_heater'][1:] * dt)),
                       E_sink=float(np.sum(r['Q_sink'][1:] * dt)))


def run_cycle(cfg, opt=None, degradation=None, verbose=True, cycle_hr=None, charge_hr=None, P_tol=0.01,
              resize_block=False, size_heater_cavity=False, resize_kw=None):
    """Full cycle for the design in `cfg` (a loaded YAML). `opt` overrides
    thermal.optimize.OPT_DEFAULTS (efficiencies, cycles, parasitics); `degradation`
    e.g. {'cell': 0.05, 'bare': 0.005} for a part-worn stick (default clean).
    cycle_hr: cycle length [h] (default 8760 / cycles_per_year); hold = cycle - charge -
    discharge. charge_hr: target charge time [h]: the heater power is cut (bisection, at
    the YAML heater cavity, up to the max_power_per_m x H cap) to the smallest that
    charges within it; None = heater at the cap.
    resize_block: shade-open sizing first (thermal.sizing.shade_open_resize, T_max fixed):
    T_min = block average at which the shades are fully open, block volume so that is at
    the duration; resize_kw go to it (tol_T, tol_t, max_iter).
    size_heater_cavity: the optimizer's heater search (needs charge_hr): smallest heater
    power with the cavity at its minimum gap, the cavity growing only if the power cap
    is not enough.
    -> SimpleNamespace(RTE, ledger, groups, N, times, runs, tank, cfg (as run), ...)"""
    from . import optimize as O
    opt = O.apply_scenario(O.merge(O.OPT_DEFAULTS, opt or {}))     # cost scenario: efficiencies, parasitics
    eco, par = opt['economics'], opt['parasitics']
    c = copy.deepcopy(cfg)
    if degradation:
        c['degradation'] = degradation
    cycle_hr = float(cycle_hr) if cycle_hr else 8760.0 / eco['cycles_per_year']
    say = print if verbose else (lambda *a, **k: None)

    # discharge (stick cell), optionally after shade-open sizing of T_min and the block volume
    so = hs = None
    if resize_block:
        from .sizing import shade_open_resize
        say("block: shade-open sizing (T_min and volume, T_max fixed) ...", flush=True)
        so = shade_open_resize(c, verbose=verbose, **(resize_kw or {}))
        if not so.converged:
            say("WARNING: shade-open sizing did not converge; the last pass is used")
        c, m, rd = so.cfg, so.m, so.r
    else:
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
    # stick / heater counts of the layout: the heater cell holds N_ps / N_h of a stick cell's graphite
    o_cnt = O.merge(opt, {'charge_hr': charge_hr or opt['charge_hr']})
    counts = O.plant_counts(st, m, c, o_cnt)
    c['heater']['cell_graphite_ratio'] = counts[0] / counts[1]
    m.cfg = c
    # charge (heater cell)
    say(f"layout {c['tank'].get('layout', 'checkerboard')}: {counts[0]} sticks + {counts[1]} heaters", flush=True)
    say("charge ...", flush=True)
    end_hr = cycle_hr - t_dis / 3600
    if size_heater_cavity:
        if not charge_hr:
            raise ValueError("size_heater_cavity needs charge_hr")
        say("heater: power + cavity search ...", flush=True)
        o2 = O.merge(opt, {'charge_hr': charge_hr, 'fidelity': {'dt_scale': 1.0, 'heater_ar_tol': 0.01}})
        hs = O.heater_search(m, c, o2)
        if hs['ok']:
            c['heater']['area_ratio'] = hs['ar']
        else:
            say(f"WARNING: heater search failed ({hs.get('reason')}); YAML heater cavity used")
    if hs is not None and hs['ok']:
        hm, rc, ch = _charge(m, c, end_hr, P=hs['P_heater'])
        say(f"heater sized: {hs['P_heater']/1e3:,.0f} kW, cavity Ø{2*hm.r_in*1e3:.0f} mm "
            f"(area_ratio {hs['ar']:.2f}), charge {ch['t_charge']/3600:.1f} h; element {describe_heater(hm.element)}")
    else:
        hm, rc, ch = _charge(m, c, end_hr)                   # at the power cap
    if (hs is None or not hs['ok']) and charge_hr and ch['done'] and ch['t_charge'] < charge_hr * 3600:
        # smallest heater power that still charges within charge_hr
        from .properties import storage_rho, storage_cp
        E_blk = (storage_rho(m.T_max) * storage_cp(m.T_max) * m.V_block * (m.T_max - m.T_min)
                 * c['heater']['cell_graphite_ratio'])
        P_cap = hm.heater.max_power
        P_lo, P_hi, best = E_blk / (charge_hr * 3600), P_cap, (hm, rc, ch)
        while P_hi / P_lo > 1 + P_tol:
            P_mid = (P_lo * P_hi) ** 0.5
            trial = _charge(m, c, end_hr, P=P_mid)
            if trial[2]['done'] and trial[2]['t_charge'] <= charge_hr * 3600:
                P_hi, best = P_mid, trial
            else:
                P_lo = P_mid
        hm, rc, ch = best
        say(f"heater power sized to {hm.heater.max_power/1e3:,.0f} kW (cap {P_cap/1e3:,.0f} kW) "
            f"for a {charge_hr:g} h charge")
    elif (hs is None or not hs['ok']) and charge_hr and ch['done']:
        say(f"WARNING: the heater at its cap ({hm.heater.max_power/1e3:,.0f} kW) needs "
            f"{ch['t_charge']/3600:.1f} h, more than charge_hr = {charge_hr:g} h")
    t_ch = ch['t_charge']
    st = dataclasses.replace(st, A_hcav_default=np.pi * hm.r_in ** 2)    # heater cavity as run -> tank layout
    pitch, webs = tk.cavity_webs(m.r_in, m.r_out, hm.r_in, rotated=True, layout=c['tank'].get('layout', 'checkerboard'))
    if min(webs.values()) < opt['geometry']['min_wall']:
        say(f"WARNING: graphite wall {min(webs.values())*1e3:.0f} mm < {opt['geometry']['min_wall']*1e3:.0f} mm minimum "
            f"({', '.join(f'{k} {v*1e3:.0f}' for k, v in webs.items())} mm)")
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
    N, N_h = hz.N_ps, hz.N_h
    if (N, N_h) != counts:
        say(f"WARNING: the plant with the heater as run has {N} sticks + {N_h} heaters; the heater cell was "
            f"sized for {counts[0]} + {counts[1]}")
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
    # heater leads (opt['leads']['table'], as the optimizer): Joule heat while charging is bought; the
    # net heat they drain from the storage is put back by the heaters; their rejected heat loads the fans
    lt = O.lead_table(opt)
    lead = None
    E_lj = E_ld = E_lr = 0.0
    if lt:
        lead = O.lead_energies(lt, opt, hm.heater.max_power, O.heater_voltage(hm.element, c), float(ct['t_top']),
                               T_max_C, t_ch, max(cycle_hr * 3600 - t_ch, 0.0), N_h)
        E_lj, E_ld, E_lr = lead['E_joule'], lead['E_drain'], lead['E_rej']
    heat_in = (N * E_released + N_h * ch['E_sink'] + E_ch_side + sum(E_hold.values())
               + E_ld)                                                       # heaters -> block
    AC_in = (heat_in + E_lj) / ei
    k = 1 - E_side_dis / E_rad                       # side + edge heat comes out of the stick's share
    DC = N * k * E_e
    E_rej = par['heat_rejection_kWe_per_kWth'] * (N * md['E_rej'] + E_lr)
    E_floor = L_hot['P_fan'] * cycle_hr * 3600
    net = DC * eo - E_floor - E_rej
    rows = [
        ('power electronics', 'heater supply (1 - eta_in)', AC_in - heat_in - E_lj),
    ] + ([
        ('heater leads', 'Joule heat (charging)', E_lj),
        ('heater leads', 'heat drawn from the storage (net)', E_ld),
    ] if lead else []) + [
        ('insulation: charge', 'roof + floor (cell losses)', N_h * ch['E_sink']),
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
    eta_cell = E_e / E_cell                          # cell efficiency, DC / absorbed by the active cells
    res = SimpleNamespace(
        RTE=RTE, RTE_gross=DC * eo / AC_in, AC_in=AC_in, net=net, ledger=rows, groups=groups, N=N, N_h=N_h,
        layout=hz.layout, cluster=hz.cluster,
        DC=DC, E_cell_abs=N * k * E_cell, eta_cell=eta_cell, RTE_frac_cell=RTE / eta_cell,
        t_charge_hr=t_ch / 3600, t_hold_hr=t_hold / 3600, t_discharge_hr=t_dis / 3600, cycle_hr=cycle_hr,
        charge_done=ch['done'], m=m, hm=hm, r_dis=rd, r_chg=rc, stick=st, tank=tank, flux=md,
        T_min_C=T_min_C, T_max_C=T_max_C, hold_loss_W=L_hot['Q_total'], P_floor_fan=L_hot['P_fan'], opt=opt,
        left_in_block=st.E_stick_th - E_released, E_released=E_released, cfg=c, sizing=so, heater_sizing=hs,
        pitch=pitch, webs=webs, lead=lead,
        T_end_vol=st.T_avg[-1] if len(st.T_avg) else np.nan)
    if verbose:
        print(report(res))
    return res


def report(res):
    """Printable ledger (str)."""
    A = res.AC_in
    s_ = res.cfg['storage']
    L = [f"Design as run: block r {res.m.r_in:.3f}-{res.m.r_out:.3f} m x {res.m.H_block:.2f} m "
         f"(volume_factor {float(s_.get('volume_factor', 1.0) or 1.0):.4f}"
         f"{', shade-open sized' if res.sizing is not None else ''}); heater {res.hm.heater.max_power/1e3:,.0f} kW, "
         f"cavity Ø{2*res.hm.r_in*1e3:.0f} mm (area_ratio {res.hm.area_ratio:.2f}"
         f"{', searched' if res.heater_sizing is not None and res.heater_sizing['ok'] else ''})",
         f"Grid: pitch {res.pitch:.3f} m, graphite walls " + ', '.join(f"{k} {v*1e3:.0f}" for k, v in res.webs.items())
         + " mm (stick cavities at 45 deg)",
         f"Plant: {res.N} sticks + {getattr(res, 'N_h', res.N)} heaters ({getattr(res, 'layout', 'checkerboard')}); block {res.T_min_C:.0f}-{res.T_max_C:.0f} C; cycle "
         f"{res.cycle_hr:.1f} h = charge {res.t_charge_hr:.1f} + hold {res.t_hold_hr:.1f} + discharge "
         f"{res.t_discharge_hr:.1f} h" + ('' if res.charge_done else '  (CHARGE INCOMPLETE)'),
         f"Round trip: {res.RTE*100:.1f}% net ({res.RTE_gross*100:.1f}% before parasitic loads);  "
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


def _square(a, rot_deg=0.0):
    """Corners (4, 2) of a square of side a centred at 0, rotated by rot_deg."""
    P = 0.5 * a * np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]], float)
    return _rot(P, rot_deg)


def _rot(P, deg):
    t = np.radians(deg)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    return np.asarray(P, float) @ R.T


def _cut(P, y=0.0):
    """x-extent (lo, hi) where the line at height y crosses convex polygon P, or None."""
    P = np.asarray(P, float)
    xs = []
    for a, b in zip(P, np.roll(P, -1, axis=0)):
        if (a[1] - y) * (b[1] - y) < 0:
            xs.append(a[0] + (y - a[1]) * (b[0] - a[0]) / (b[1] - a[1]))
        elif a[1] == y:
            xs.append(a[0])
    return (min(xs), max(xs)) if len(xs) >= 2 else None


def plot_layout(res, axes=None, n=4, rot=None, full_max=80):
    """To-scale plan and profile of the hot zone as built: stick and heater cavities of the layout
    (thermal/layout.py: checkerboard, hex6, honeycomb or hex3; stick-heater spacing d from the plan
    area per stick, as tank.size_hot_zone) in one continuous graphite matrix. Small plants
    (N_ps + N_h <= full_max) are drawn whole, with the hot-zone circle; larger ones as a window about
    the centre about n spacings wide. Stick cavity square (side S = pi r_in / 2), turned `rot` deg
    (None: the layout's: 45 deg on the checkerboard, the thickest-wall rotation on the hexagonal
    patterns) with its liner, powerstick and the 8 octagonal shade vanes (closed solid, open dashed,
    full block height); heater cavity round. The profile is the vertical section through a central
    stick and its nearest heater (dotted in the plan). The analysis cells are overlaid dashed: the
    discharge model gives a stick its graphite as an annulus out to an adiabatic r_out, the charge
    model gives each heater N_ps / N_h of that. -> (fig, axes)"""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle, Polygon, Patch
    from matplotlib.lines import Line2D
    from . import layout as LY

    m, hm = res.m, res.hm
    cfg = m.cfg
    p_, h, sh = cfg['powerstick'], cfg['heater'], cfg.get('shade', {}) or {}
    H, L = float(m.H_block), float(m.L)
    ps = float(m.powerstick.width)
    liner = float(p_['liner_side_len']) if m.liner_bool else None
    t_l = float(p_.get('liner_thickness', 0.0) or 0.0)
    el = hm.element
    tube = el['type'] == 'tube'
    hs = float(el['D']) if tube else float(el['A_cross']) ** 0.5     # tube OD or rod side
    S = np.pi * m.r_in / 2                                    # square stick cavity (r_in = 2 S / pi)
    r_hc = float(hm.r_in)
    lb = m.liner_bottom
    t_lb, t_rgf = (float(lb['thickness']), float(lb['rgf_thickness'])) if H > L + 1e-9 else (0.0, 0.0)
    pat = getattr(res, 'layout', None) or cfg.get('tank', {}).get('layout', 'checkerboard')
    cl = getattr(res, 'cluster', None)
    N_ps, N_h = res.N, getattr(res, 'N_h', res.N)
    d = LY.spacing(pat, m.r_out, r_hc)
    rot, webs = LY.webs(pat, S, r_hc, d, rot=rot)
    col = dict(graphite='#a6a6a6', cavity='white', liner='#c8a165', shade='#e8901c', stick='#3a6fc4',
               heater='#d43c2c', pad='#555555', cell_s='#1f3f8f', cell_h='#8f1f1f')

    # sites (m): the whole cluster for a small plant, else a window of the lattice about the centre
    full = cl is not None and N_ps + N_h <= full_max
    if full:
        St, Ht = np.array(cl['sticks']) * d, np.array(cl['heaters']) * d
    else:
        S0, H0, _, _ = LY.lattice(pat, R=n + 3)
        St, Ht = S0 * d, H0 * d
    # section: through the stick nearest the centre and its nearest heater; turn everything so it is +x
    s0 = St[np.argmin(np.hypot(*St.T))]
    h0 = Ht[np.argmin(np.hypot(*(Ht - s0).T))]
    phi = np.degrees(np.arctan2(*(h0 - s0)[::-1]))
    St, Ht = _rot(St - s0, -phi), _rot(Ht - s0, -phi)
    rot_t = rot - phi
    if full:
        c_hz = _rot(-s0[None, :], -phi)[0]                     # hot-zone centre in the turned frame
        R_hz = 0.5 * res.tank.hz.D_hot
        lo_x, hi_x, lo_y, hi_y = c_hz[0] - R_hz - 0.3, c_hz[0] + R_hz + 0.3, c_hz[1] - R_hz - 0.3, c_hz[1] + R_hz + 0.3
    else:
        half = 0.5 * n * d
        lo_x, hi_x, lo_y, hi_y = -half, half + d, -half, half
        keep = lambda P, r: (P[:, 0] > lo_x - r) & (P[:, 0] < hi_x + r) & (P[:, 1] > lo_y - r) & (P[:, 1] < hi_y + r)
        St, Ht = St[keep(St, S)], Ht[keep(Ht, r_hc)]

    # stick-cavity contents as polygons about the stick centre, turned with the cavity
    shapes = dict(cavity=_square(S, rot_t), stick=_square(ps, rot_t))
    if liner:
        shapes.update(liner_o=_square(liner, rot_t), liner_i=_square(liner - 2 * t_l, rot_t))
    vanes, t_vane = None, None
    if m.shade_bool and liner and str(sh.get('model', 'vanes')).lower() == 'vanes':
        from .correlations import vane_layout_for_liner, _vane_boxes, _vane_box_corners
        try:
            lay, _ = vane_layout_for_liner(liner, float(sh.get('thickness', 0.05)), float(sh.get('corner_gap', 0.003175)),
                                           float(sh.get('vane_clearance', 0.005)), S=S)
            t_vane = lay['t']
            vanes = {th: [_rot(np.array(_vane_box_corners(b, lay['t']))[[0, 1, 3, 2]], rot_t) for b in _vane_boxes(lay, th)]
                     for th in (0.0, 90.0)}
        except ValueError:
            vanes = None

    if axes is None:
        w, hgt = hi_x - lo_x, hi_y - lo_y
        wr = [w / hgt, w / (H + 0.5)]                     # equal heights at a common scale per panel
        fw = min(17.0, 6.0 * sum(wr))
        fig, axes = plt.subplots(1, 2, figsize=(fw, fw / sum(wr) + 1.8), gridspec_kw={'width_ratios': wr})
    else:
        fig = axes[0].figure
    ap, ax = axes

    # ---- plan --------------------------------------------------------------------------
    if full:
        ap.add_patch(Circle(c_hz, R_hz, fc=col['graphite'], ec='k', lw=.8, zorder=0))
    else:
        ap.add_patch(Rectangle((lo_x, lo_y), hi_x - lo_x, hi_y - lo_y, fc=col['graphite'], ec='none', zorder=0))
    for xy in St:
        ap.add_patch(Polygon(shapes['cavity'] + xy, fc=col['cavity'], ec='k', lw=.6, zorder=1))
        if vanes:
            for P in vanes[0.0]:
                ap.add_patch(Polygon(P + xy, fc=col['shade'], ec='none', zorder=2))
            for P in vanes[90.0]:
                ap.add_patch(Polygon(P + xy, fc='none', ec=col['shade'], ls='--', lw=.8, zorder=2))
        if liner:
            ap.add_patch(Polygon(shapes['liner_o'] + xy, fc=col['liner'], ec='k', lw=.5, zorder=3))
            ap.add_patch(Polygon(shapes['liner_i'] + xy, fc=col['cavity'], ec='none', zorder=3))
        ap.add_patch(Polygon(shapes['stick'] + xy, fc=col['stick'], ec='k', lw=.5, zorder=4))
    for xy in Ht:
        ap.add_patch(Circle(xy, r_hc, fc=col['cavity'], ec='k', lw=.6, zorder=1))
        ap.add_patch(Circle(xy, hs / 2, fc=col['heater'], ec='k', lw=.5, zorder=4) if tube else
                     Rectangle(xy - hs / 2, hs, hs, fc=col['heater'], ec='k', lw=.5, zorder=4))
    ap.add_patch(Circle((0, 0), m.r_out, fc='none', ec=col['cell_s'], ls='--', lw=1.6, zorder=5))
    ap.add_patch(Circle((d, 0), hm.r_out, fc='none', ec=col['cell_h'], ls='--', lw=1.6, zorder=5))
    ap.plot([lo_x, hi_x], [0, 0], c='k', lw=.6, ls=':', zorder=5)
    ap.set_title(f"Plan ({pat}): stick-heater spacing {d:.3f} m; min graphite wall {min(webs.values())*1e3:.0f} mm\n"
                 + ', '.join(f"{k} {v*1e3:.0f}" for k, v in webs.items()) + " mm (dotted = profile section)", fontsize=10)

    # ---- profile: vertical section along the dotted line (y = 0) ----------------------------
    ax.add_patch(Rectangle((lo_x, 0), hi_x - lo_x, H, fc=col['graphite'], ec='k', lw=.6, zorder=0))
    z_act = H - L
    on = lambda P: P[np.abs(P[:, 1]) < 1e-6 * max(d, 1.0)]
    for x, _ in on(St):
        a, b = _cut(shapes['cavity'])
        ax.add_patch(Rectangle((x + a, 0), b - a, H, fc=col['cavity'], ec='k', lw=.6, zorder=1))
        if vanes:                                          # vanes cut by the section, full height
            for th, kw in ((0.0, dict(fc=col['shade'], ec='none')),
                           (90.0, dict(fc='none', ec=col['shade'], ls='--', lw=.8))):
                for P in vanes[th]:
                    c_ = _cut(P)
                    if c_:
                        ax.add_patch(Rectangle((x + c_[0], 0), c_[1] - c_[0], H, zorder=2, **kw))
        if liner:
            (oa, ob), (ia, ib) = _cut(shapes['liner_o']), _cut(shapes['liner_i'])
            z_lb = z_act - t_lb
            z_lw = z_lb - t_rgf                            # liner walls run down past the pad (model: H_low)
            for a, b in ((oa, ia), (ib, ob)):
                ax.add_patch(Rectangle((x + a, z_lw), b - a, H - z_lw, fc=col['liner'], ec='k', lw=.4, zorder=3))
            if t_lb > 0:
                ax.add_patch(Rectangle((x + oa, z_lb), ob - oa, t_lb, fc=col['liner'], ec='k', lw=.4, zorder=3))
            if t_rgf > 0:
                ax.add_patch(Rectangle((x + ia, z_lw), ib - ia, t_rgf, fc=col['pad'], ec='k', lw=.4, zorder=3))
        a, b = _cut(shapes['stick'])
        ax.add_patch(Rectangle((x + a, z_act), b - a, L, fc=col['stick'], ec='k', lw=.5, zorder=4))
    for x, _ in on(Ht):
        ax.add_patch(Rectangle((x - r_hc, 0), 2 * r_hc, H, fc=col['cavity'], ec='k', lw=.6, zorder=1))
        ax.add_patch(Rectangle((x - hs / 2, 0), hs, H, fc=col['heater'], ec='k', lw=.5, zorder=4))
    for xc, ro, c in ((0.0, m.r_out, col['cell_s']), (d, hm.r_out, col['cell_h'])):
        for s in (-1, 1):
            ax.plot([xc + s * ro] * 2, [0, H], c=c, ls='--', lw=1.4, zorder=5)
    ax.set_title('Profile: vertical section through a stick and its nearest heater', fontsize=10)
    ax.set_ylabel('height [m]')

    ap.set_aspect('equal')
    ap.set_xlim(lo_x, hi_x)
    ap.set_ylim(lo_y, hi_y)
    ax.set_aspect('equal')
    ax.set_xlim(lo_x, hi_x)
    ax.set_ylim(-0.2, H + 0.3)
    for a in (ap, ax):
        a.set_xlabel('[m]')

    leg = [Patch(fc=col['graphite'], ec='k', label=f"graphite matrix, block height {H:.2f} m"),
           Patch(fc=col['stick'], ec='k', label=f"powerstick {ps*1e3:.0f} mm square, L {L:.2f} m (top of the block)"),
           Patch(fc='white', ec='k', label=f"stick cavity {S*1e3:.0f} mm square; heater cavity Ø{2*r_hc*1e3:.0f} mm")]
    if liner:
        leg.append(Patch(fc=col['liner'], ec='k', label=f"liner {liner*1e3:.0f} mm square, {t_l*1e3:.0f} mm wall"
                                                         + (f"; bottom {t_lb*1e3:.0f} mm + RGF pad {t_rgf*1e3:.0f} mm" if t_rgf else '')))
    if vanes:
        leg += [Patch(fc=col['shade'], ec='none', label=f"shade vanes closed (octagon, {t_vane*1e3:.0f} mm thick, full block height)"),
                Patch(fc='none', ec=col['shade'], ls='--', label='shade vanes open')]
    leg += [Patch(fc=col['heater'], ec='k', label=f"heater {describe_heater(el)}, full height, {hm.heater.max_power/1e3:,.0f} kW"),
            Line2D([0], [0], c=col['cell_s'], ls='--', label=f"discharge model: stick cell, adiabatic r_out {m.r_out:.3f} m"),
            Line2D([0], [0], c=col['cell_h'], ls='--', label=f"charge model: heater cell, adiabatic r_out {hm.r_out:.3f} m")]
    fig.legend(handles=leg, loc='lower center', ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"Hot zone to scale ({pat}): {N_ps} sticks + {N_h} heaters in one graphite matrix"
                 + ('' if full else ' (window about the centre)') + "\n"
                 f"dashed circles = analysis cells: a stick's graphite in one annulus; each heater charges "
                 f"{N_ps}/{N_h} of that, adiabatic r_out in place of the layout's symmetry lines", fontsize=10)
    fig.tight_layout(rect=(0, 0.13, 1, 0.95))
    return fig, axes


def plot_timeseries(cy):
    """Temperatures, powers and discharge flux / shade over the cycle, one stick + one
    heater cell (charge: heater cell; discharge: stick cell). -> (fig, axes)"""
    import matplotlib.pyplot as plt
    from .sizing import cell_flux
    rc, rd, m = cy.r_chg, cy.r_dis, cy.m
    tc = np.asarray(rc['time']) / 3600
    kc = tc <= cy.t_charge_hr + 1e-9
    t1 = cy.t_charge_hr + cy.t_hold_hr                       # discharge starts here
    on = np.asarray(rd['ps_active'], bool)
    td = np.asarray(rd['time'])[on] / 3600 + t1
    t_end = t1 + cy.t_discharge_hr

    def C(T):
        return np.asarray(T, float) - C2K

    fig, ax = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    a = ax[0]
    a.plot(tc[kc], C(rc['T_heater'])[kc], c='C3', label='heater')
    a.plot(tc[kc], C(rc['T_inner'])[kc], c='C1', label='cavity wall')
    a.plot(tc[kc], C(rc['T_avg'])[kc], c='k', label='block average')
    a.plot(tc[kc], C(rc['T_outer'])[kc], c='C0', label='block outer')
    a.plot([cy.t_charge_hr, t1], [cy.T_max_C] * 2, c='k')
    a.plot(td, C(rd['T_inner'])[on], c='C1')
    a.plot(td, C(rd['T_avg'])[on], c='k')
    a.plot(td, C(rd['T_outer'])[on], c='C0')
    if m.liner_bool:
        a.plot(td, C(rd['T_liner'])[on], c='C2', label='liner')
    a.set_ylabel('°C'); a.set_title('Temperatures (charge: heater cell; discharge: stick cell)')

    a = ax[1]
    a.plot(tc[kc], np.asarray(rc['Q_heater'])[kc] / 1e3, c='C3', label='heater power')
    a.plot(tc[kc], np.asarray(rc['Q_sink'])[kc] / 1e3, c='C7', label='roof + floor loss (per cell)')
    a.plot([cy.t_charge_hr, t1], [cy.hold_loss_W / cy.N / 1e3] * 2, c='C3', ls='--',
           label='hold: heater makes up all losses')
    a.plot(td, np.asarray(rd['Q_rad'])[on] / 1e3, c='C1', label='heat into stick')
    a.plot(td, np.asarray(rd['Q_cell'])[on] / 1e3, c='C4', label='absorbed by active cells')
    a.plot(td, np.asarray(rd['P_elec'])[on] / 1e3, c='C2', label='electricity (DC)')
    a.plot(td, np.asarray(rd['Q_sink'])[on] / 1e3, c='C7')
    a.set_ylabel('kW per stick / heater'); a.set_yscale('symlog', linthresh=10); a.set_title('Power')

    a = ax[2]
    a.plot(td, cell_flux(m, rd)[on], c='C0', label='cell flux')
    a.axhline(m.avg_flux, c='C0', ls=':', lw=1, label='target flux')
    a.set_ylabel('W/cm2'); a.set_title('Discharge: cell flux and shade')
    a2 = a.twinx()
    a2.plot(td, np.asarray(rd['f_shade'])[on], c='C1', label='shade closed fraction')
    a2.set_ylabel('shade closed fraction'); a2.set_ylim(0, 1.05)
    a2.legend(loc='upper right', fontsize=8)
    for a in ax:
        for x in (cy.t_charge_hr, t1, t_end):
            a.axvline(x, c='0.6', lw=.8, ls='--')
        a.grid(alpha=.3); a.legend(loc='upper left', fontsize=8)
    ax[-1].set_xlabel('time in cycle [h]  (charge | hold | discharge)')
    fig.tight_layout()
    return fig, ax


LEDGER_COLOURS = {'power electronics': 'C3', 'insulation: charge': 'C0', 'insulation: hold': 'C9',
                  'insulation: discharge': 'C5', 'stick': 'C1', 'parasitics': 'C4', 'residual': '0.6'}


def plot_losses(cy):
    """Ledger as % of AC bought (bars) and by group (donut). -> (fig, axes)"""
    import matplotlib.pyplot as plt
    cols = LEDGER_COLOURS
    rows = [r for r in cy.ledger if r[0] != 'output']
    fig, ax = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [3, 2]})
    lab = [f"{g}: {l}" for g, l, _ in rows]
    pct = np.array([v for _, _, v in rows]) / cy.AC_in * 100
    ax[0].barh(lab, pct, color=[cols.get(g, 'k') for g, _, _ in rows])
    for i, p in enumerate(pct):
        ax[0].text(max(p, 0) + 0.3, i, f"{p:.1f}%", va='center', fontsize=8)
    ax[0].invert_yaxis(); ax[0].set_xlabel('% of AC bought'); ax[0].grid(alpha=.3, axis='x')
    ax[0].set_title(f"Where the electricity goes: net out {cy.RTE*100:.1f}%")
    g = sorted(cy.groups.items(), key=lambda kv: -kv[1])
    ax[1].pie([max(v, 0) for _, v in g], labels=[f"{k}\n{v/cy.AC_in*100:.1f}%" for k, v in g],
              colors=[cols.get(k, 'C2') for k, _ in g], startangle=90, counterclock=False,
              wedgeprops=dict(width=.45), textprops=dict(fontsize=8))
    ax[1].set_title(f"AC in {cy.AC_in*J_MWh:,.0f} MWh / cycle")
    fig.tight_layout()
    return fig, ax
