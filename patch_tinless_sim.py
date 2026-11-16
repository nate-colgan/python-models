import json

with open('tinless system model.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if not cell.get('id', '').startswith('03dd9a23'):
        continue
    src = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])
    orig = src

    # ── 1. Extend __init__ signature ─────────────────────────────────────────
    src = src.replace(
        '    def __init__(self, storage, powerstick, T_sur=300, block_heat_loss=False):\n',
        '    def __init__(self, storage, powerstick, T_sur=300, block_heat_loss=False,\n'
        '                 shield_bool=False, shield_side_len=None,\n'
        '                 f_shield_min=0.0, f_shield_max=0.9, nominal_cell_flux=None):\n',
        1,
    )

    # ── 2. Append shield geometry block after liner block in __init__ ─────────
    old_after_liner = (
        '            self.A_liner_seg = 2 * np.pi * r_liner * powerstick.height_step\n'
        '        else:\n'
        '            self.VF_gr_liner = None\n'
        '            self.VF_liner_ps = None\n'
        '            self.A_liner_seg = None\n'
        '\n'
        '    def _step_calc(self, T_dist):\n'
    )
    new_after_liner = (
        '            self.A_liner_seg = 2 * np.pi * r_liner * powerstick.height_step\n'
        '        else:\n'
        '            self.VF_gr_liner = None\n'
        '            self.VF_liner_ps = None\n'
        '            self.A_liner_seg = None\n'
        '\n'
        '        self.shield_bool       = shield_bool\n'
        '        self.f_shield_min      = f_shield_min\n'
        '        self.f_shield_max      = f_shield_max\n'
        '        self.nominal_cell_flux = nominal_cell_flux\n'
        '        if shield_bool:\n'
        '            if not powerstick.liner_bool:\n'
        '                raise ValueError("shield_bool=True requires liner_bool=True")\n'
        '            if shield_side_len is None:\n'
        '                raise ValueError("shield_side_len must be set when shield_bool=True")\n'
        '            r_shield           = 2 * shield_side_len / np.pi\n'
        '            self.VF_gr_shield  = r_shield / r_in\n'
        '            self.VF_shield_lin = r_liner / r_shield\n'
        '            self.A_shield_seg  = 2 * np.pi * r_shield * powerstick.height_step\n'
        '        else:\n'
        '            self.VF_gr_shield  = None\n'
        '            self.VF_shield_lin = None\n'
        '            self.A_shield_seg  = None\n'
        '\n'
        '    def _step_calc(self, T_dist):\n'
    )
    if old_after_liner not in src:
        print('WARNING: after_liner pattern not found')
    else:
        src = src.replace(old_after_liner, new_after_liner, 1)

    # ── 3. Add f_shield param to _step_calc + pass to spectral_net_radiation ──
    old_step_sig = (
        '    def _step_calc(self, T_dist):\n'
        '        ps   = self.powerstick\n'
    )
    new_step_sig = (
        '    def _step_calc(self, T_dist, f_shield=0.0):\n'
        '        ps   = self.powerstick\n'
    )
    if old_step_sig not in src:
        print('WARNING: _step_calc signature pattern not found')
    else:
        src = src.replace(old_step_sig, new_step_sig, 1)

    old_snr_call = (
        '            Q_rad[i], T_liner_i = spectral_net_radiation(\n'
        '                T_wall_i, ps.emissivity_table, self.VF,\n'
        '                self.A_ps_seg, ps.f_bare, ps.f_active, ps.liner_bool,\n'
        '                VF_gr_liner=self.VF_gr_liner, VF_liner_ps=self.VF_liner_ps,\n'
        '                A_liner_seg=self.A_liner_seg\n'
        '            )\n'
    )
    new_snr_call = (
        '            Q_rad[i], T_liner_i = spectral_net_radiation(\n'
        '                T_wall_i, ps.emissivity_table, self.VF,\n'
        '                self.A_ps_seg, ps.f_bare, ps.f_active, ps.liner_bool,\n'
        '                VF_gr_liner=self.VF_gr_liner, VF_liner_ps=self.VF_liner_ps,\n'
        '                A_liner_seg=self.A_liner_seg,\n'
        '                shield_bool=self.shield_bool, f_shield=f_shield,\n'
        '                VF_gr_shield=self.VF_gr_shield, VF_shield_lin=self.VF_shield_lin,\n'
        '                A_shield_seg=self.A_shield_seg,\n'
        '            )\n'
    )
    if old_snr_call not in src:
        print('WARNING: spectral_net_radiation call pattern not found')
    else:
        src = src.replace(old_snr_call, new_snr_call, 1)

    # ── 4. Insert _compute_f_shield between _step_calc and run_simulation ─────
    old_step_end = (
        '        return Q_rad, Q_elec, Q_cell, Q_waste, T_liner\n'
        '\n'
        '    def run_simulation(self, end_time, dt, sub_steps, snap_every=None, print_interval=0):\n'
    )
    new_step_end = (
        '        return Q_rad, Q_elec, Q_cell, Q_waste, T_liner\n'
        '\n'
        '    def _compute_f_shield(self, T_dist):\n'
        '        """\n'
        '        Bisect f_shield in [f_shield_min, f_shield_max] to match nominal_cell_flux\n'
        '        [W/cm²].  Returns the nearest bound if the target lies outside the\n'
        '        achievable range.\n'
        '        """\n'
        '        from scipy.optimize import brentq\n'
        '        ps = self.powerstick\n'
        '        A_cell_cm2 = 4 * ps.width * ps.height * (1 - ps.f_bare) * ps.f_active * 1e4\n'
        '\n'
        '        def _flux(fs):\n'
        '            Q_rad, _, _, _, _ = self._step_calc(T_dist, f_shield=fs)\n'
        '            return Q_rad.sum() / A_cell_cm2\n'
        '\n'
        '        f_lo, f_hi = self.f_shield_min, self.f_shield_max\n'
        '        flux_lo = _flux(f_lo)\n'
        '        flux_hi = _flux(f_hi)\n'
        '\n'
        '        if flux_lo <= self.nominal_cell_flux:\n'
        '            return f_lo   # min shield already below nominal; open to minimum\n'
        '        if flux_hi >= self.nominal_cell_flux:\n'
        '            return f_hi   # max shield still above nominal; clamp to max\n'
        '        return brentq(lambda fs: _flux(fs) - self.nominal_cell_flux,\n'
        '                      f_lo, f_hi, xtol=1e-4, maxiter=30)\n'
        '\n'
        '    def run_simulation(self, end_time, dt, sub_steps, snap_every=None, print_interval=0):\n'
    )
    if old_step_end not in src:
        print('WARNING: step_end/run_simulation boundary pattern not found')
    else:
        src = src.replace(old_step_end, new_step_end, 1)

    # ── 5. Add f_shield_arr to run_simulation arrays block ────────────────────
    src = src.replace(
        '        T_liner_arr = np.zeros(n_steps + 1)\n'
        '        snap_times  = []\n',
        '        T_liner_arr  = np.zeros(n_steps + 1)\n'
        '        f_shield_arr = np.zeros(n_steps + 1)\n'
        '        snap_times   = []\n',
        1,
    )

    # ── 6. Compute initial f_shield and pass to first _step_calc ─────────────
    old_init_step = (
        '        T = storage.block.temp_dist\n'
        '        Q0, Q_elec0, Q_cell0, _, T_liner0 = self._step_calc(T)\n'
        '        Q_arr[0]       = Q0.sum()\n'
        '        Q_cell_arr[0]  = Q_cell0.sum()\n'
        '        Q_elec_arr[0]  = Q_elec0.sum()\n'
        '        T_avg_arr[0]   = T.mean()\n'
        '        T_inner_arr[0] = T[:, 0].mean()\n'
        '        T_outer_arr[0] = T[:, -1].mean()\n'
        '        T_liner_arr[0] = T_liner0.mean()\n'
    )
    new_init_step = (
        '        T = storage.block.temp_dist\n'
        '        _use_shield = self.shield_bool and self.nominal_cell_flux is not None\n'
        '        f_shield_0 = self._compute_f_shield(T) if _use_shield else 0.0\n'
        '        Q0, Q_elec0, Q_cell0, _, T_liner0 = self._step_calc(T, f_shield=f_shield_0)\n'
        '        Q_arr[0]       = Q0.sum()\n'
        '        Q_cell_arr[0]  = Q_cell0.sum()\n'
        '        Q_elec_arr[0]  = Q_elec0.sum()\n'
        '        T_avg_arr[0]   = T.mean()\n'
        '        T_inner_arr[0] = T[:, 0].mean()\n'
        '        T_outer_arr[0] = T[:, -1].mean()\n'
        '        T_liner_arr[0] = T_liner0.mean()\n'
        '        f_shield_arr[0] = f_shield_0\n'
    )
    if old_init_step not in src:
        print('WARNING: init_step pattern not found')
    else:
        src = src.replace(old_init_step, new_init_step, 1)

    # ── 7. Compute f_shield each loop step and pass to _step_calc ─────────────
    old_loop_step = (
        '        for step in range(n_steps):\n'
        '            T = storage.block.temp_dist\n'
        '            Q_rad, Q_elec, Q_cell, Q_waste, T_liner = self._step_calc(T)\n'
    )
    new_loop_step = (
        '        for step in range(n_steps):\n'
        '            T = storage.block.temp_dist\n'
        '            f_shield = self._compute_f_shield(T) if _use_shield else 0.0\n'
        '            Q_rad, Q_elec, Q_cell, Q_waste, T_liner = self._step_calc(T, f_shield=f_shield)\n'
    )
    if old_loop_step not in src:
        print('WARNING: loop_step pattern not found')
    else:
        src = src.replace(old_loop_step, new_loop_step, 1)

    # ── 8. Store f_shield each loop step ─────────────────────────────────────
    old_liner_store = (
        '            T_liner_arr[idx] = T_liner.mean()\n'
        '\n'
        '            if idx % snap_steps == 0:\n'
    )
    new_liner_store = (
        '            T_liner_arr[idx]  = T_liner.mean()\n'
        '            f_shield_arr[idx] = f_shield\n'
        '\n'
        '            if idx % snap_steps == 0:\n'
    )
    if old_liner_store not in src:
        print('WARNING: liner_store pattern not found')
    else:
        src = src.replace(old_liner_store, new_liner_store, 1)

    # ── 9. Add f_shield to print statement ────────────────────────────────────
    old_print = (
        '                      f"Q_cell = {Q_cell_arr[idx]/1e3:.1f} kW "\n'
        '                      f"P_elec = {Q_elec_arr[idx]/1e3:.1f} kW")\n'
    )
    new_print = (
        '                      f"Q_cell = {Q_cell_arr[idx]/1e3:.1f} kW  "\n'
        '                      f"P_elec = {Q_elec_arr[idx]/1e3:.1f} kW  "\n'
        '                      f"f_sh = {f_shield_arr[idx]:.3f}")\n'
    )
    if old_print not in src:
        print('WARNING: print pattern not found')
    else:
        src = src.replace(old_print, new_print, 1)

    # ── 10. Add f_shield_arr to return dict ───────────────────────────────────
    old_return = (
        "            'T_liner':        T_liner_arr,\n"
        "            'snap_times':     np.array(snap_times),\n"
    )
    new_return = (
        "            'T_liner':        T_liner_arr,\n"
        "            'f_shield':       f_shield_arr,\n"
        "            'snap_times':     np.array(snap_times),\n"
    )
    if old_return not in src:
        print('WARNING: return dict pattern not found')
    else:
        src = src.replace(old_return, new_return, 1)

    if src != orig:
        cell['source'] = src
        cell['outputs'] = []
        cell['execution_count'] = None
        print('Patched 03dd9a23')
    else:
        print('WARNING: no changes made to 03dd9a23')
    break

with open('tinless system model.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Saved.')
