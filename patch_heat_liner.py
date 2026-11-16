import json

with open('tinless system model.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if not cell.get('id', '').startswith('03dd9a23'):
        continue
    src = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])

    # 1. Insert _liner_avg helper before the Initial state block
    old_init = (
        '        # ── Initial state ────────────────────────────────────────────────────\n'
        '        T         = storage.block.temp_dist.copy()\n'
        '        T_h_0, _  = _solve_T_heater(T[:, 0])'
    )
    new_init = (
        '        # ── Liner temperature helper (monitoring only) ───────────────────────\n'
        '        ps_heat   = self.powerstick\n'
        '        n_ax_heat = storage.num_ax_segments\n'
        '\n'
        '        def _liner_avg(T_d):\n'
        '            if not ps_heat.liner_bool:\n'
        '                return 0.0\n'
        '            tl = np.zeros(n_ax_heat)\n'
        '            for i in range(n_ax_heat):\n'
        '                _, tli = spectral_net_radiation(\n'
        '                    float(T_d[i, 0]), ps_heat.emissivity_table, self.VF,\n'
        '                    self.A_ps_seg, ps_heat.f_bare, ps_heat.f_active, ps_heat.liner_bool,\n'
        '                    VF_gr_liner=self.VF_gr_liner, VF_liner_ps=self.VF_liner_ps,\n'
        '                    A_liner_seg=self.A_liner_seg\n'
        '                )\n'
        '                tl[i] = tli\n'
        '            return float(tl.mean())\n'
        '\n'
        '        # ── Initial state ────────────────────────────────────────────────────\n'
        '        T         = storage.block.temp_dist.copy()\n'
        '        T_h_0, _  = _solve_T_heater(T[:, 0])'
    )

    # 2. Add T_liner_list to initial list declarations
    old_lists = (
        '        T_outer_list  = [float(T[:, -1].mean())]\n'
        '        snap_times    = [0.0]'
    )
    new_lists = (
        '        T_outer_list  = [float(T[:, -1].mean())]\n'
        '        T_liner_list  = [_liner_avg(T)]\n'
        '        snap_times    = [0.0]'
    )

    # 3. Append T_liner in the loop
    old_append = (
        '            T_inner_list.append(float(T_new[:, 0].mean()))\n'
        '            T_outer_list.append(float(T_new[:, -1].mean()))'
    )
    new_append = (
        '            T_inner_list.append(float(T_new[:, 0].mean()))\n'
        '            T_outer_list.append(float(T_new[:, -1].mean()))\n'
        '            T_liner_list.append(_liner_avg(T_new))'
    )

    # 4. Print statement: add T_liner between T_avg and T_heater
    old_print = (
        '                print(\n'
        '                    f"t={t/3600:.2f} hr  T_inner={T_new[:, 0].mean():.0f} K  T_avg={T_new.mean():.0f} K  "\n'
        '                    f"T_heater={T_h:.0f} K  Q={Q_segs.sum()/1e3:.1f} kW  "\n'
        '                    f"[phase {phase}]"\n'
        '                )'
    )
    new_print = (
        '                print(\n'
        '                    f"t={t/3600:.2f} hr  T_inner={T_new[:, 0].mean():.0f} K  T_avg={T_new.mean():.0f} K  "\n'
        '                    f"T_liner={T_liner_list[-1]:.0f} K  T_heater={T_h:.0f} K  Q={Q_segs.sum()/1e3:.1f} kW  "\n'
        '                    f"[phase {phase}]"\n'
        '                )'
    )

    # 5. Return dict: add T_liner after T_outer
    old_ret = (
        "            'T_outer':        np.array(T_outer_list),\n"
        "            'snap_times':     np.array(snap_times),"
    )
    new_ret = (
        "            'T_outer':        np.array(T_outer_list),\n"
        "            'T_liner':        np.array(T_liner_list),\n"
        "            'snap_times':     np.array(snap_times),"
    )

    ok = True
    for label, old in [
        ('init comment', old_init),
        ('list decls',   old_lists),
        ('append',       old_append),
        ('print',        old_print),
        ('return dict',  old_ret),
    ]:
        if old not in src:
            print(f'WARNING: pattern "{label}" not found')
            ok = False

    if ok:
        src = src.replace(old_init,   new_init,   1)
        src = src.replace(old_lists,  new_lists,  1)
        src = src.replace(old_append, new_append, 1)
        src = src.replace(old_print,  new_print,  1)
        src = src.replace(old_ret,    new_ret,    1)
        cell['source'] = src
        cell['outputs'] = []
        cell['execution_count'] = None
        print('Patched 03dd9a23')
    break

with open('tinless system model.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Saved.')
