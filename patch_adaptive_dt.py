"""
1. YAML: replace dt with dt_short / dt_medium / dt_long + dt_short_duration
2. run_simulation: adaptive while-loop replacing fixed n_steps for-loop
3. All cells reading cfg['dt']: update to read new keys
4. Plot cell 35eeb2ae: add log x-axis
"""
import json

NB = 'tinless system model.ipynb'
with open(NB, encoding='utf-8') as f:
    nb = json.load(f)

# ── 1. YAML cell a8c3d501 ─────────────────────────────────────────────────────
for cell in nb['cells']:
    if not cell.get('id','').startswith('a8c3d501'):
        continue
    src = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])

    old_dt = 'dt: 150                    # time step [s]\ndt_short: 150                    # time step [s]\ndt_medium: 600                   # time step [s]\ndt_long: 3600'
    # Simpler: replace the existing dt line
    for old in [
        'dt: 900                    # time step [s]',
        'dt: 150                    # time step [s]',
        'dt: 600                    # time step [s]',
    ]:
        if old in src:
            src = src.replace(old,
                'dt_short: 150              # timestep [s] — used for the first dt_short_duration seconds\n'
                'dt_medium: 900             # timestep [s] — used when shield is active and not clamped\n'
                'dt_long: 3600              # timestep [s] — used when shield is clamped at f_shield_max\n'
                'dt_short_duration: 3600    # seconds of real time to use dt_short at the start',
                1)
            print(f'Replaced {old!r} in YAML')
            break

    cell['source'] = src
    print(f'  dt_short in YAML: {src.count("dt_short")}')
    break

# ── 2. TinlessSimulation.run_simulation in cell 03dd9a23 ─────────────────────
NEW_RUN_SIM = '''\
    def run_simulation(self, end_time, dt_short, dt_medium, dt_long, sub_steps,
                       dt_short_duration=3600, snap_every=None, print_interval=0):
        storage = self.storage
        _use_shield = self.shield_bool and self.nominal_cell_flux is not None

        # ── Initial state ─────────────────────────────────────────────────────
        T = storage.block.temp_dist
        f_shield = self._compute_f_shield(T) if _use_shield else 0.0
        Q0, Q_elec0, Q_cell0, _, T_liner0 = self._step_calc(T, f_shield=f_shield)

        time_arr     = [0.0]
        Q_arr        = [Q0.sum()]
        Q_cell_arr   = [Q_cell0.sum()]
        Q_elec_arr   = [Q_elec0.sum()]
        T_avg_arr    = [T.mean()]
        T_inner_arr  = [T[:, 0].mean()]
        T_outer_arr  = [T[:, -1].mean()]
        T_liner_arr  = [T_liner0.mean()]
        f_shield_arr = [f_shield]
        snap_times   = [0.0]
        temp_snaps   = [T.copy()]

        t            = 0.0
        next_snap    = snap_every if snap_every else float('inf')
        last_print_t = -float('inf')

        # ── Time loop ─────────────────────────────────────────────────────────
        while t < end_time:
            T        = storage.block.temp_dist
            f_shield = self._compute_f_shield(T) if _use_shield else 0.0

            # Adaptive timestep
            if t < dt_short_duration:
                dt = dt_short
            elif self.shield_bool and f_shield >= self.f_shield_max - 1e-4:
                dt = dt_long
            else:
                dt = dt_medium if self.shield_bool else dt_long
            dt = min(dt, end_time - t)

            Q_rad, Q_elec, Q_cell, Q_waste, T_liner = self._step_calc(T, f_shield=f_shield)
            T_new, _, _ = storage.update_temps(
                T, -Q_rad, dt, sub_steps, self.T_sur, self.block_heat_loss
            )
            storage.block.temp_dist = T_new
            t += dt

            time_arr.append(t)
            Q_arr.append(Q_rad.sum())
            Q_cell_arr.append(Q_cell.sum())
            Q_elec_arr.append(Q_elec.sum())
            T_avg_arr.append(T_new.mean())
            T_inner_arr.append(T_new[:, 0].mean())
            T_outer_arr.append(T_new[:, -1].mean())
            T_liner_arr.append(T_liner.mean())
            f_shield_arr.append(f_shield)

            if snap_every and t >= next_snap:
                snap_times.append(t)
                temp_snaps.append(T_new.copy())
                next_snap += snap_every

            if print_interval and (t - last_print_t >= print_interval):
                print(f"t = {t/3600:.2f} hr  "
                      f"T_avg = {T_new.mean():.0f} K  "
                      f"T_inner = {T_new[:,0].mean():.0f} K  "
                      f"T_liner = {T_liner.mean():.0f} K  "
                      f"Q = {Q_rad.sum()/1e3:.1f} kW  "
                      f"Q_cell = {Q_cell.sum()/1e3:.1f} kW  "
                      f"P_elec = {Q_elec.sum()/1e3:.1f} kW  "
                      f"f_sh = {f_shield:.3f}  dt = {dt:.0f} s")
                last_print_t = t

        if snap_times[-1] != t:
            snap_times.append(t)
            temp_snaps.append(storage.block.temp_dist.copy())

        return {
            'time':           np.array(time_arr),
            'Q_rad':          np.array(Q_arr),
            'Q_cell':         np.array(Q_cell_arr),
            'P_elec':         np.array(Q_elec_arr),
            'T_avg':          np.array(T_avg_arr),
            'T_inner':        np.array(T_inner_arr),
            'T_outer':        np.array(T_outer_arr),
            'T_liner':        np.array(T_liner_arr),
            'f_shield':       np.array(f_shield_arr),
            'snap_times':     np.array(snap_times),
            'temp_snapshots': np.array(temp_snaps),
        }

'''

for cell in nb['cells']:
    if not cell.get('id','').startswith('03dd9a23'):
        continue
    src = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])
    lines = src.splitlines(keepends=True)

    # Find run_simulation method start and end
    start = next(i for i,l in enumerate(lines) if '    def run_simulation(' in l)
    end   = next((i for i in range(start+1, len(lines))
                  if lines[i].startswith('    def ')), len(lines))

    new_lines = lines[:start] + [NEW_RUN_SIM] + lines[end:]
    src = ''.join(new_lines)
    cell['source'] = src
    cell['outputs'] = []
    cell['execution_count'] = None
    print(f'Replaced run_simulation. Cell now {len(src.splitlines())} lines.')
    break

# ── 3. Run cell b2e4f810 ──────────────────────────────────────────────────────
for cell in nb['cells']:
    if not cell.get('id','').startswith('b2e4f810'):
        continue
    src = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])

    # Replace dt read
    src = src.replace("dt                   = cfg['dt']",
                      "dt_short             = cfg['dt_short']\n"
                      "dt_medium            = cfg['dt_medium']\n"
                      "dt_long              = cfg['dt_long']\n"
                      "dt_short_duration    = cfg.get('dt_short_duration', 3600)")

    # Replace run_simulation call
    src = src.replace(
        'results = sim.run_simulation(\n'
        '    end_time      = end_time,\n'
        '    dt            = dt,\n'
        '    sub_steps     = sub_steps,\n'
        '    snap_every    = 3600,       # save full temp distribution every 1 hr\n'
        '    print_interval = int(3600 / dt),  # print progress every 1 hr of sim time\n'
        ')',
        'results = sim.run_simulation(\n'
        '    end_time          = end_time,\n'
        '    dt_short          = dt_short,\n'
        '    dt_medium         = dt_medium,\n'
        '    dt_long           = dt_long,\n'
        '    sub_steps         = sub_steps,\n'
        '    dt_short_duration = dt_short_duration,\n'
        '    snap_every        = 3600,\n'
        '    print_interval    = 3600,\n'
        ')'
    )

    cell['source'] = src
    cell['outputs'] = []
    cell['execution_count'] = None
    print(f'Updated b2e4f810')
    # Verify no stale dt reference
    leftovers = [l.rstrip() for l in src.splitlines()
                 if "cfg['dt']" in l or "= dt," in l or "= dt\n" in l]
    if leftovers:
        print(f'  WARNING stale dt refs: {leftovers}')
    break

# ── 4. Run cell c7d9e412 ──────────────────────────────────────────────────────
for cell in nb['cells']:
    if not cell.get('id','').startswith('c7d9e412'):
        continue
    src = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])

    src = src.replace("_dt          = _cfg['dt']",
                      "_dt_short    = _cfg['dt_short']\n"
                      "_dt_medium   = _cfg['dt_medium']\n"
                      "_dt_long     = _cfg['dt_long']")

    src = src.replace(
        "res_      = sim_.run_simulation(end_time=_end_time, dt=_dt, sub_steps=_sub_steps)",
        "res_      = sim_.run_simulation(end_time=_end_time,\n"
        "                                dt_short=_dt_short, dt_medium=_dt_medium, dt_long=_dt_long,\n"
        "                                sub_steps=_sub_steps, dt_short_duration=0)"
    )

    cell['source'] = src
    cell['outputs'] = []
    cell['execution_count'] = None
    print(f'Updated c7d9e412')
    break

# ── 5. Heating cells: read dt_short as their single dt ───────────────────────
for cell in nb['cells']:
    cid = cell.get('id','')
    if not (cid.startswith('heat_sim_01') or cid.startswith('heat_bisect_')):
        continue
    src = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])
    if "_dt        = _cfg['dt']" in src:
        src = src.replace("_dt        = _cfg['dt']",
                          "_dt        = _cfg['dt_short']  # heating uses the short timestep")
        cell['source'] = src
        cell['outputs'] = []
        cell['execution_count'] = None
        print(f'Updated {cid[:16]} to use dt_short')

# ── 6. Plot cell 35eeb2ae — log x-axis ───────────────────────────────────────
for cell in nb['cells']:
    if not cell.get('id','').startswith('35eeb2ae'):
        continue
    src = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])

    # After each ax.set_xlabel('Time [hr]') add set_xscale('log')
    # There are 3 figures; add log scale and sensible xlim after xlabel
    src = src.replace(
        "ax1.set_xlabel('Time [hr]')\n"
        "ax1.set_ylabel('Power [kW]')",
        "ax1.set_xlabel('Time [hr]')\n"
        "ax1.set_xscale('log')\n"
        "ax1.set_ylabel('Power [kW]')"
    )
    src = src.replace(
        "ax2.set_xlabel('Time [hr]')\n"
        "ax2.set_ylabel('Temperature [C]')",
        "ax2.set_xlabel('Time [hr]')\n"
        "ax2.set_xscale('log')\n"
        "ax2.set_ylabel('Temperature [C]')"
    )
    src = src.replace(
        "ax3.set_xlabel('Time [hr]')\n"
        "ax3.set_ylabel('Cell flux [W/cm²]')",
        "ax3.set_xlabel('Time [hr]')\n"
        "ax3.set_xscale('log')\n"
        "ax3.set_ylabel('Cell flux [W/cm²]')"
    )
    # Also handle plain ASCII superscript variant
    src = src.replace(
        "ax3.set_xlabel('Time [hr]')\n"
        "ax3.set_ylabel('Cell flux [W/cm2]')",
        "ax3.set_xlabel('Time [hr]')\n"
        "ax3.set_xscale('log')\n"
        "ax3.set_ylabel('Cell flux [W/cm2]')"
    )
    src = src.replace(
        "ax3.set_xlabel('Time [hr]')\n"
        "ax3.set_ylabel('Cell flux [W/cm\xb2]')",
        "ax3.set_xlabel('Time [hr]')\n"
        "ax3.set_xscale('log')\n"
        "ax3.set_ylabel('Cell flux [W/cm\xb2]')"
    )

    cell['source'] = src
    cell['outputs'] = []
    cell['execution_count'] = None
    count = src.count("set_xscale('log')")
    print(f'Added {count} log x-axis calls to plot cell 35eeb2ae')
    break

with open(NB, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Saved.')
