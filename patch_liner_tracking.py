import json

with open('tinless system model.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

changed = []

for cell in nb['cells']:
    if not cell.get('id', '').startswith('03dd9a23'):
        continue
    src = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])

    # 1. _step_calc: add T_liner array, store per-segment, return 4 values
    old_step = (
        '        Q_waste = np.zeros(n_ax)\n'
        '        for i in range(n_ax):\n'
        '            T_wall_i = T_dist[i, 0]\n'
        '            Q_rad[i], T_liner_i = spectral_net_radiation(\n'
        '                T_wall_i, ps.emissivity_table, self.VF,\n'
        '                self.A_ps_seg, ps.f_bare, ps.f_active, ps.liner_bool,\n'
        '                VF_gr_liner=self.VF_gr_liner, VF_liner_ps=self.VF_liner_ps,\n'
        '                A_liner_seg=self.A_liner_seg\n'
        '            )\n'
        '            T_emit_i = T_liner_i if ps.liner_bool else T_wall_i\n'
        '            VF_emit  = self.VF_liner_ps if ps.liner_bool else self.VF\n'
        '            _, _, Q_cell[i] = ps.generate_electricity_step(\n'
        '                T_emit_i, VF_emit, ps.panel_efficiency\n'
        '            )\n'
        '            Q_waste[i] = Q_rad[i] - Q_cell[i] * ps.panel_efficiency\n'
        '        return Q_rad, Q_cell, Q_waste'
    )
    new_step = (
        '        Q_waste = np.zeros(n_ax)\n'
        '        T_liner = np.zeros(n_ax)\n'
        '        for i in range(n_ax):\n'
        '            T_wall_i = T_dist[i, 0]\n'
        '            Q_rad[i], T_liner_i = spectral_net_radiation(\n'
        '                T_wall_i, ps.emissivity_table, self.VF,\n'
        '                self.A_ps_seg, ps.f_bare, ps.f_active, ps.liner_bool,\n'
        '                VF_gr_liner=self.VF_gr_liner, VF_liner_ps=self.VF_liner_ps,\n'
        '                A_liner_seg=self.A_liner_seg\n'
        '            )\n'
        '            T_liner[i]  = T_liner_i if ps.liner_bool else 0.0\n'
        '            T_emit_i = T_liner_i if ps.liner_bool else T_wall_i\n'
        '            VF_emit  = self.VF_liner_ps if ps.liner_bool else self.VF\n'
        '            _, _, Q_cell[i] = ps.generate_electricity_step(\n'
        '                T_emit_i, VF_emit, ps.panel_efficiency\n'
        '            )\n'
        '            Q_waste[i] = Q_rad[i] - Q_cell[i] * ps.panel_efficiency\n'
        '        return Q_rad, Q_cell, Q_waste, T_liner'
    )

    # 2. run_simulation: add T_liner_arr declaration
    old_arr = (
        '        T_outer_arr = np.zeros(n_steps + 1)\n'
        '        snap_times  = []\n'
        '        temp_snaps  = []'
    )
    new_arr = (
        '        T_outer_arr = np.zeros(n_steps + 1)\n'
        '        T_liner_arr = np.zeros(n_steps + 1)\n'
        '        snap_times  = []\n'
        '        temp_snaps  = []'
    )

    # 3. Initial _step_calc call + T_liner_arr[0]
    old_init = (
        '        Q0, Q_cell0, _ = self._step_calc(T)\n'
        '        Q_arr[0]       = Q0.sum()\n'
        '        Q_cell_arr[0]  = Q_cell0.sum()\n'
        '        P_elec_arr[0]  = Q_cell0.sum() * self.powerstick.panel_efficiency\n'
        '        T_avg_arr[0]   = T.mean()\n'
        '        T_inner_arr[0] = T[:, 0].mean()\n'
        '        T_outer_arr[0] = T[:, -1].mean()'
    )
    new_init = (
        '        Q0, Q_cell0, _, T_liner0 = self._step_calc(T)\n'
        '        Q_arr[0]       = Q0.sum()\n'
        '        Q_cell_arr[0]  = Q_cell0.sum()\n'
        '        P_elec_arr[0]  = Q_cell0.sum() * self.powerstick.panel_efficiency\n'
        '        T_avg_arr[0]   = T.mean()\n'
        '        T_inner_arr[0] = T[:, 0].mean()\n'
        '        T_outer_arr[0] = T[:, -1].mean()\n'
        '        T_liner_arr[0] = T_liner0.mean()'
    )

    # 4. Loop _step_calc call
    old_loop_call = '            Q_rad, Q_cell, Q_waste = self._step_calc(T)'
    new_loop_call = '            Q_rad, Q_cell, Q_waste, T_liner = self._step_calc(T)'

    # 5. Loop index assignment: add T_liner_arr[idx]
    old_idx = (
        '            T_avg_arr[idx]   = T_new.mean()\n'
        '            T_inner_arr[idx] = T_new[:, 0].mean()\n'
        '            T_outer_arr[idx] = T_new[:, -1].mean()'
    )
    new_idx = (
        '            T_avg_arr[idx]   = T_new.mean()\n'
        '            T_inner_arr[idx] = T_new[:, 0].mean()\n'
        '            T_outer_arr[idx] = T_new[:, -1].mean()\n'
        '            T_liner_arr[idx] = T_liner.mean()'
    )

    # 6. Return dict: insert T_liner after T_outer
    old_ret = (
        "            'T_outer':        T_outer_arr,\n"
        "            'snap_times':     np.array(snap_times),"
    )
    new_ret = (
        "            'T_outer':        T_outer_arr,\n"
        "            'T_liner':        T_liner_arr,\n"
        "            'snap_times':     np.array(snap_times),"
    )

    ok = True
    for label, old in [('_step_calc body', old_step), ('T_liner_arr decl', old_arr),
                       ('init call', old_init), ('loop call', old_loop_call),
                       ('loop idx', old_idx), ('return dict', old_ret)]:
        if old not in src:
            print(f'WARNING: pattern "{label}" not found')
            ok = False

    if ok:
        src = src.replace(old_step,      new_step,      1)
        src = src.replace(old_arr,       new_arr,       1)
        src = src.replace(old_init,      new_init,      1)
        src = src.replace(old_loop_call, new_loop_call, 1)
        src = src.replace(old_idx,       new_idx,       1)
        src = src.replace(old_ret,       new_ret,       1)
        cell['source'] = src
        cell['outputs'] = []
        cell['execution_count'] = None
        changed.append(cell.get('id',''))

print(f'Changed: {changed}')
with open('tinless system model.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Saved.')
