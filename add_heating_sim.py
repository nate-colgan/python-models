import json

with open('tinless system model.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

# ── HeatingElement class ─────────────────────────────────────────────────────
heating_element_cls = '''\
class HeatingElement:
    """
    Cylindrical resistive heating element at the centre of the graphite annulus.

    Modelled as a gray-body radiator with uniform surface temperature.  Heat is
    transferred from the heater to the graphite inner wall via thermal radiation
    in a two-surface coaxial enclosure.  No thermal mass is assumed: the heater
    equilibrium temperature is solved algebraically each timestep from the
    electrical power supplied and the current graphite inner-wall temperature.

    Control phases
    --------------
    Phase 1 (power-limited):  supply max_power at the lowest heater temperature
        that drives that flux into the graphite.  T_heater rises as the graphite
        warms and the temperature difference shrinks.
    Phase 2 (temperature-limited):  once T_heater would exceed max_temperature,
        clamp T_heater = max_temperature and supply whatever heat that drives.
        Power delivered decreases as the block approaches max_temperature.

    The simulation stops when T_avg_block >= T_block_max.

    Attributes
    ----------
    diameter        : float -- heater rod outer diameter [m]
    max_temperature : float -- maximum allowed heater surface temperature [K]
    max_power       : float -- maximum electrical power input [W]
    emissivity      : float -- gray-body emissivity of the heater surface
    """

    def __init__(self, diameter, max_temperature, max_power, emissivity):
        """
        Inputs
        ------
        diameter        : float -- heater rod outer diameter [m]
        max_temperature : float -- maximum allowed heater surface temperature [K]
        max_power       : float -- maximum electrical power input [W]
        emissivity      : float -- gray-body emissivity of the heater surface
        """
        self.diameter        = float(diameter)
        self.max_temperature = float(max_temperature)
        self.max_power       = float(max_power)
        self.emissivity      = float(emissivity)

'''

# ── run_heating_simulation method ────────────────────────────────────────────
run_heating_method = '''
    def run_heating_simulation(self, heater, T_block_max, dt, sub_steps,
                                snap_every=None, print_interval=0):
        """
        Run a transient charging simulation with a central resistive heating element.

        The heater radiates outward to the graphite inner wall (opposite direction to
        discharge).  At each timestep the heater temperature is solved from a two-surface
        coaxial radiation enclosure model; the resulting per-segment heat flux is passed
        to StorageModule.update_temps as a positive source at j=0 (inner wall).

        The simulation runs until T_avg_block >= T_block_max or a hard cap of 1000
        timesteps is reached.

        Inputs
        ------
        heater      : HeatingElement -- heater object (diameter, max_temperature,
                                        max_power, emissivity)
        T_block_max : float          -- target charged block temperature [K];
                                        heating stops when T_avg >= T_block_max
        dt          : float          -- timestep [s]
        sub_steps   : int            -- finite-difference sub-steps per dt
        snap_every  : float|None     -- interval between full-field temperature
                                        snapshots [s]; None records start/end only
        print_interval : int         -- print status every N timesteps; 0 = silent

        Returns
        -------
        dict with keys:
          'time'           : ndarray -- time [s]
          'Q_heater'       : ndarray -- total heater power delivered [W]
          'T_heater'       : ndarray -- heater surface temperature [K]
          'T_avg'          : ndarray -- volume-average block temperature [K]
          'T_inner'        : ndarray -- axial-average inner wall temperature [K]
          'T_outer'        : ndarray -- axial-average outer wall temperature [K]
          'snap_times'     : ndarray -- times of full-field snapshots [s]
          'temp_snapshots' : ndarray -- (n_ax, n_rad) temperature field at each snap
        """
        from scipy.optimize import brentq

        sigma   = 5.67036713e-8
        storage = self.storage
        r_in    = storage.block.rad_in
        L_blk   = storage.block.length
        n_ax    = storage.num_ax_segments
        seg_len = L_blk / n_ax

        A_h_seg  = np.pi * heater.diameter * seg_len          # heater area / segment
        A_gr_seg = np.pi * 2.0 * r_in * seg_len               # graphite inner-wall area / seg
        eps_h    = heater.emissivity

        def _q_segs(T_h, T_inner_arr):
            """Per-segment radiation [W] from heater at T_h to graphite inner wall."""
            Q_seg = np.zeros(n_ax)
            for i in range(n_ax):
                eps_gr  = gr_emis(T_inner_arr[i])
                R_enc   = (1.0 - eps_h) / eps_h + 1.0 + (A_h_seg / A_gr_seg) * (1.0 - eps_gr) / eps_gr
                Q_seg[i] = sigma * (T_h**4 - T_inner_arr[i]**4) * A_h_seg / R_enc
            return Q_seg

        def _total_q(T_h, T_inner_arr):
            return float(_q_segs(T_h, T_inner_arr).sum())

        def _solve_T_heater(T_inner_arr):
            """Return (T_h, Q_segs) for the current timestep."""
            T_inner_max   = float(T_inner_arr.max())
            q_at_max_temp = _total_q(heater.max_temperature, T_inner_arr)

            if q_at_max_temp >= heater.max_power:
                # Phase 1: find T_h < max_temperature that delivers max_power
                T_lo = T_inner_max + 0.1
                T_hi = heater.max_temperature
                try:
                    T_h = brentq(
                        lambda Th: _total_q(Th, T_inner_arr) - heater.max_power,
                        T_lo, T_hi, xtol=1.0, maxiter=50
                    )
                except ValueError:
                    T_h = heater.max_temperature
            else:
                # Phase 2: heater clamped at max_temperature, power falls below max
                T_h = heater.max_temperature

            return T_h, _q_segs(T_h, T_inner_arr)

        # ── Initial state ────────────────────────────────────────────────────
        T         = storage.block.temp_dist.copy()
        T_h_0, _  = _solve_T_heater(T[:, 0])

        snap_steps    = max(1, round(snap_every / dt)) if snap_every else int(1e9)
        time_list     = [0.0]
        Q_list        = [0.0]
        T_h_list      = [T_h_0]
        T_avg_list    = [float(T.mean())]
        T_inner_list  = [float(T[:, 0].mean())]
        T_outer_list  = [float(T[:, -1].mean())]
        snap_times    = [0.0]
        temp_snaps    = [T.copy()]

        # ── Time-stepping ────────────────────────────────────────────────────
        for step in range(10000):
            T_inner_arr = T[:, 0]
            T_h, Q_segs = _solve_T_heater(T_inner_arr)

            T_new, _, _ = storage.update_temps(
                T, Q_segs, dt, sub_steps, self.T_sur, self.block_heat_loss
            )
            storage.block.temp_dist = T_new
            T = T_new

            idx = step + 1
            t   = idx * dt
            time_list.append(t)
            Q_list.append(float(Q_segs.sum()))
            T_h_list.append(T_h)
            T_avg_list.append(float(T_new.mean()))
            T_inner_list.append(float(T_new[:, 0].mean()))
            T_outer_list.append(float(T_new[:, -1].mean()))

            if idx % snap_steps == 0:
                snap_times.append(t)
                temp_snaps.append(T_new.copy())

            if print_interval and (idx % print_interval == 0):
                phase = 1 if T_h < heater.max_temperature - 1.0 else 2
                print(
                    f"t={t/3600:.2f} hr  T_avg={T_new.mean():.0f} K  "
                    f"T_heater={T_h:.0f} K  Q={Q_segs.sum()/1e3:.1f} kW  "
                    f"[phase {phase}]"
                )

            if T_new.mean() >= T_block_max:
                print(f"Block fully charged: T_avg={T_new.mean():.1f} K >= {T_block_max} K "
                      f"at t={t/3600:.2f} hr")
                break

        if snap_times[-1] != time_list[-1]:
            snap_times.append(time_list[-1])
            temp_snaps.append(storage.block.temp_dist.copy())

        return {
            'time':           np.array(time_list),
            'Q_heater':       np.array(Q_list),
            'T_heater':       np.array(T_h_list),
            'T_avg':          np.array(T_avg_list),
            'T_inner':        np.array(T_inner_list),
            'T_outer':        np.array(T_outer_list),
            'snap_times':     np.array(snap_times),
            'temp_snapshots': np.array(temp_snaps),
        }
'''

# ── Patch cell 03dd9a23 ───────────────────────────────────────────────────────
for cell in nb['cells']:
    if cell.get('id', '').startswith('03dd9a23'):
        src = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])

        # 1. Prepend HeatingElement before TinlessSimulation
        if 'class HeatingElement' not in src:
            src = heating_element_cls + src

        # 2. Append run_heating_simulation inside TinlessSimulation (at end of cell)
        if 'run_heating_simulation' not in src:
            src = src.rstrip('\n') + '\n' + run_heating_method

        cell['source'] = src
        cell['outputs'] = []
        cell['execution_count'] = None
        print('Patched cell 03dd9a23')
        break

with open('tinless system model.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Notebook saved.')
