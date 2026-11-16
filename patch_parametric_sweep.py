import json

with open('tinless system model.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

changed = []

for cell in nb['cells']:
    cid = cell.get('id', '')
    src = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])

    # ── Sweep cell ff640d4b ───────────────────────────────────────────────
    if cid.startswith('ff640d4b'):
        new_src = '''\
import thermal.components as _tc
from scipy.integrate import cumulative_trapezoid

# ── Fixed design parameters ──────────────────────────────────────────────
_avg_flux        = 20          # W/cm^2
_duration_hr     = 50          # hr  (fixed discharge duration)
_side_len        = 0.13        # m
_L               = 3.0         # m
_th_ins          = 0.0         # m
_num_ax          = 6
_num_rad         = 12
_T_ps            = 323         # K
_panel_eff       = 0.5
_f_bare          = 0.2
_f_active        = 0.83
_T_sur            = 300         # K
_dt              = 1200        # s
_sub_steps       = 5
_pwr_tol         = 0.01        # bisection tolerance (fraction of avg_power)
_T_min           = 2173        # K  (block empty/minimum temperature)

_r_ps_eff      = 2 * _side_len / np.pi
_avg_power = _avg_flux * 1e4 * 4 * _side_len * _L * (1.0 - _f_bare) * _f_active  # cell-weighted [W]

# ── Sweep parameters ─────────────────────────────────────────────────────
liner_bool_vals = [False, True]           # tungsten liner on/off
temp_range_vals = [250, 375, 500]         # K  (block operating temperature swing)
bandgap_vals    = [1.24, 1.38, 1.55]     # microns

# ── Helpers ───────────────────────────────────────────────────────────────
def _build_and_run(T_max, dT_range, area_ratio, emis_table, liner_bool_):
    """
    Size a graphite storage block and run a full transient discharge simulation.

    T_min = T_max - dT_range = _T_min (since T_max = _T_min + dT_range).
    Block volume is sized from the required energy; inner radius follows from
    area_ratio = r_in / r_ps_eff; outer radius derived from V_block and r_in.
    """
    T_min_    = T_max - dT_range
    T_avg_mid = T_min_ + dT_range / 2
    e_dens_   = storage_rho(T_avg_mid) * storage_cp(T_avg_mid) * dT_range
    f_cell_      = (1.0 - _f_bare) * _f_active
    f_inact_     = (1.0 - _f_bare) * (1.0 - _f_active)
    ps_tmp_      = PowerStick(axial_steps=1, width=_side_len, height=_L,
                             cooling_temp=_T_ps, panel_efficiency=_panel_eff,
                             f_bare=_f_bare, f_active=_f_active,
                             liner_bool=liner_bool_, emissivity_table=emis_table)
    inc_bins_, _, _ = ps_tmp_.incident_energy(T_min_, 1.0)
    inc_tot_     = sum(inc_bins_.values())
    f_Q_         = sum(
        inc_bins_[row[0]] / inc_tot_ * f_cell_ * row[1]
        / (_f_bare * row[2] + f_cell_ * row[1] + f_inact_ * row[3])
        for row in emis_table)
    tot_power_   = _avg_flux * 1e4 * 4 * _side_len * _L * f_cell_ / f_Q_
    V_blk_    = tot_power_ * _duration_hr * 3600 / e_dens_
    r_in_     = _r_ps_eff * area_ratio
    r_out_    = np.sqrt(V_blk_ / (np.pi * _L) + r_in_**2)

    init_dist_ = np.full((_num_ax, _num_rad), float(T_max))
    block_  = GraphiteBlock(init_dist_, inner_radius=r_in_, outer_radius=r_out_,
                            length=_L, th_ins=_th_ins)
    pipe_   = Pipe(OD=0.02, length=_L, wall_thickness=0.001, roughness=1e-5, th_ins=0.0)
    stor_   = StorageModule(N_pipes=1, Pipe=pipe_, Block=block_,
                            num_ax_segments=_num_ax, num_rad_segments=_num_rad,
                            bypass_range=[0, 1])
    ps_     = PowerStick(axial_steps=_num_ax, width=_side_len, height=_L,
                         cooling_temp=_T_ps, panel_efficiency=_panel_eff,
                         f_bare=_f_bare, f_active=_f_active,
                         liner_bool=liner_bool_, emissivity_table=emis_table)
    sim_    = TinlessSimulation(stor_, ps_, T_sur=_T_sur, block_heat_loss=False)
    res_    = sim_.run_simulation(end_time=_duration_hr * 3600, dt=_dt, sub_steps=_sub_steps)
    return res_, r_out_

def _pss_avg_q(res_, dT_range, T_max):
    """
    Compute average cell heat flux over the pseudo-steady-state window.
    PSS onset fixed at t=0; window ends when SOC drops to zero.
    """
    T_min_   = T_max - dT_range
    soc      = (res_['T_avg'] - T_min_) / dT_range
    idx_pss  = int(0)
    soc0     = np.where(soc <= 0)[0]
    idx_soc0 = int(soc0[0]) if len(soc0) else len(soc) - 1
    if idx_pss >= idx_soc0:
        return float(res_['Q_cell'][idx_pss])
    return float(res_['Q_cell'][idx_pss:idx_soc0 + 1].mean())

# ── Main sweep ───────────────────────────────────────────────────────────
sweep_results = {}
total = len(liner_bool_vals) * len(bandgap_vals) * len(temp_range_vals)
n = 0

for liner_bool_ in liner_bool_vals:
    for bg_val in bandgap_vals:
        emis_table_bg = [
            ((0.1,    bg_val), 0.9,  0.1, 0.1),
            ((bg_val, 20.0),   0.05, 0.1, 0.1),
        ]
        for dT_range in temp_range_vals:
            T_max = _T_min + dT_range
            n += 1
            print(f\'[{n}/{total}] liner={liner_bool_}  dT={dT_range}K  T_max={T_max}K  bg={bg_val}um  -- bisecting area_ratio\')

            def _f(ar, _Tmax=T_max, _dT=dT_range, _emis=emis_table_bg, _lb=liner_bool_):
                res_, _ = _build_and_run(_Tmax, _dT, ar, _emis, _lb)
                q = _pss_avg_q(res_, _dT, _Tmax)
                print(f\'    ar={ar:.3f}  avg_Q={q/1e3:.1f} kW  (target {_avg_power/1e3:.0f} kW)\')
                return q - _avg_power

            f0 = _f(10.0)
            if abs(f0) / _avg_power <= _pwr_tol:
                ar_opt = 10.0
            else:
                if f0 > 0:
                    ar_hi, f_hi = 10.0, f0
                    ar_lo = 10.0
                    while True:
                        ar_lo /= 2
                        f_lo = _f(ar_lo)
                        if f_lo <= 0:
                            break
                        ar_hi, f_hi = ar_lo, f_lo
                else:
                    ar_lo, f_lo = 10.0, f0
                    ar_hi = 10.0
                    while True:
                        ar_hi *= 2
                        f_hi = _f(ar_hi)
                        if f_hi >= 0:
                            break
                        ar_lo, f_lo = ar_hi, f_hi

                for _ in range(20):
                    ar_mid = (ar_lo + ar_hi) / 2
                    f_mid  = _f(ar_mid)
                    if abs(f_mid) / _avg_power <= _pwr_tol:
                        break
                    if f_mid > 0:
                        ar_hi = ar_mid
                    else:
                        ar_lo = ar_mid
                ar_opt = (ar_lo + ar_hi) / 2

            res_opt, r_out_opt = _build_and_run(T_max, dT_range, ar_opt, emis_table_bg, liner_bool_)
            Q  = res_opt[\'Q_rad\']
            Pe = res_opt[\'P_elec\']
            Ti = res_opt[\'T_inner\']
            Ta = res_opt[\'T_avg\']
            t  = res_opt[\'time\']
            cum_Q = cumulative_trapezoid(Q,  t, initial=0)
            cum_P = cumulative_trapezoid(Pe, t, initial=0)
            eff   = cum_P[-1] / cum_Q[-1] * 100 if cum_Q[-1] > 0 else 0.0

            sweep_results[(dT_range, bg_val, liner_bool_)] = dict(
                area_ratio = ar_opt,
                r_out      = r_out_opt,
                peak_dT    = float(np.max(Ta - Ti)),
                peak_power = float(np.max(Q)) / 1e3,
                efficiency = eff,
            )
            print(f\'    -> ar_opt={ar_opt:.3f}  r_out={r_out_opt:.3f}m  \'
                  f\'peak_Q={np.max(Q)/1e3:.0f}kW  eff={eff:.1f}%\')

print(\'Sweep complete.\')
'''
        cell['source'] = new_src
        cell['outputs'] = []
        cell['execution_count'] = None
        changed.append(cid)

    # ── Plot cell 52d87076 ────────────────────────────────────────────────
    elif cid.startswith('52d87076'):
        new_src = '''\
colors = plt.cm.plasma(np.linspace(0.2, 0.8, 2))
liner_labels = {False: \'No liner\', True: \'W liner\'}

for dT_range in temp_range_vals:
    T_max = _T_min + dT_range
    fig, axes = plt.subplots(4, 2, figsize=(12, 18))

    ax_ar,  ax_rout = axes[0]
    ax_dT,  ax_pwr  = axes[1]
    ax_eff, ax_bl1  = axes[2]
    ax_bl2, ax_bl3  = axes[3]

    for ci, liner_bool_ in enumerate(liner_bool_vals):
        lbl    = liner_labels[liner_bool_]
        kwargs = dict(color=colors[ci], marker=\'o\', label=lbl)

        ax_ar.plot(  bandgap_vals, [sweep_results[(dT_range, bg, liner_bool_)][\'area_ratio\'] for bg in bandgap_vals], **kwargs)
        ax_rout.plot(bandgap_vals, [sweep_results[(dT_range, bg, liner_bool_)][\'r_out\']      for bg in bandgap_vals], **kwargs)
        ax_dT.plot(  bandgap_vals, [sweep_results[(dT_range, bg, liner_bool_)][\'peak_dT\']    for bg in bandgap_vals], **kwargs)
        ax_pwr.plot( bandgap_vals, [sweep_results[(dT_range, bg, liner_bool_)][\'peak_power\'] / (4 * _side_len * _L * (1 - _f_bare) * _f_active * 10) for bg in bandgap_vals], **kwargs)
        ax_eff.plot( bandgap_vals, [sweep_results[(dT_range, bg, liner_bool_)][\'efficiency\'] for bg in bandgap_vals], **kwargs)

    ax_ar.set_ylabel(\'Optimal area ratio\')
    ax_ar.set_title(\'Optimal area ratio\')
    ax_rout.set_ylabel(\'Outer radius [m]\')
    ax_rout.set_title(\'Block outer radius\')
    ax_dT.set_ylabel(\'Peak ΔT (inner − avg) [K]\')
    ax_dT.set_title(\'Peak inner wall dT\')
    ax_pwr.set_ylabel(\'Peak thermal power [W/cm²]\')
    ax_pwr.set_title(\'Peak power (start of discharge)\')
    ax_eff.set_ylabel(\'Overall efficiency [%]\')
    ax_eff.set_title(\'Cumulative elec / thermal\')

    for ax in [ax_ar, ax_rout, ax_dT, ax_pwr, ax_eff]:
        ax.set_xlabel(\'Bandgap [μm]\')
        ax.grid(True)

    for ax in [ax_bl1, ax_bl2, ax_bl3]:
        ax.set_visible(False)

    ax_ar.legend(fontsize=9)
    fig.suptitle(
        f\'Parametric sweep  |  ΔT={dT_range} K  T_min={_T_min} K  T_max={T_max} K  \'
        f\'dur={_duration_hr} hr  avg_flux={_avg_flux} W/cm²\',
        fontsize=10,
    )
    fig.tight_layout()

plt.show()
'''
        cell['source'] = new_src
        cell['outputs'] = []
        cell['execution_count'] = None
        changed.append(cid)

print(f'Changed: {changed}')
with open('tinless system model.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Saved.')
