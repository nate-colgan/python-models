import json

with open('tinless system model.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

patched = []

for cell in nb['cells']:
    cid = cell.get('id', '')
    src = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])
    orig = src

    # ── ff640d4b: add Qc extraction + avg_cell_flux to sweep_results ─────────
    if cid.startswith('ff640d4b'):
        # 1. Add Qc = res_opt['Q_cell'] alongside the other extractions
        src = src.replace(
            "                Q  = res_opt['Q_rad']\n"
            "                Pe = res_opt['P_elec']\n",
            "                Q  = res_opt['Q_rad']\n"
            "                Qc = res_opt['Q_cell']\n"
            "                Pe = res_opt['P_elec']\n",
            1,
        )
        # 2. Add avg_cell_flux to the results dict
        src = src.replace(
            "                    min_power  = float(np.min(Q)) / 1e3,\n"
            "                    efficiency = eff,\n"
            "                )",
            "                    min_power      = float(np.min(Q)) / 1e3,\n"
            "                    efficiency     = eff,\n"
            "                    avg_cell_flux  = float(Qc.mean()) / (4 * _side_len * _L * (1.0 - _f_bare) * _f_active * 1e4),\n"
            "                )",
            1,
        )

    # ── 52d87076: add avg_cell_flux subplot ───────────────────────────────────
    elif cid.startswith('52d87076'):
        # 1. Add ax_bl2 plot inside the try block (after ax_eff_c)
        src = src.replace(
            "            ax_eff_c.plot(   bandgap_vals, [sweep_results[(dT_range, bg, liner_bool_)]['efficiency_cell'] for bg in bandgap_vals], **kwargs)\n",
            "            ax_eff_c.plot(   bandgap_vals, [sweep_results[(dT_range, bg, liner_bool_)]['efficiency_cell'] for bg in bandgap_vals], **kwargs)\n"
            "            ax_bl2.plot(     bandgap_vals, [sweep_results[(dT_range, bg, liner_bool_)]['avg_cell_flux']    for bg in bandgap_vals], **kwargs)\n",
            1,
        )
        # 2. Add labels for ax_bl2
        src = src.replace(
            "ax_eff_c.set_ylabel('Cell efficiency [%]')\n"
            "ax_eff_c.set_title('Cumulative elec / cell thermal')\n",
            "ax_eff_c.set_ylabel('Cell efficiency [%]')\n"
            "ax_eff_c.set_title('Cumulative elec / cell thermal')\n"
            "ax_bl2.set_ylabel('Avg cell flux [W/cm²]')\n"
            "ax_bl2.set_title('Average cell power density')\n",
            1,
        )
        # 3. Add ax_bl2 to the xlabel/grid loop
        src = src.replace(
            'for ax in [ax_ar, ax_rout, ax_pwr, ax_min_pwr, ax_eff, ax_eff_c]:',
            'for ax in [ax_ar, ax_rout, ax_pwr, ax_min_pwr, ax_eff, ax_eff_c, ax_bl2]:',
            1,
        )
        # 4. Remove ax_bl2 from the hidden-axes loop
        src = src.replace(
            'for ax in [ax_bl2, ax_bl3]:',
            'for ax in [ax_bl3]:',
            1,
        )

    if src != orig:
        cell['source'] = src
        cell['outputs'] = []
        cell['execution_count'] = None
        patched.append(cid[:12])

print(f'Patched cells: {patched}')

with open('tinless system model.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Saved.')
