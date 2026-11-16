import json

with open('tinless system model.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

# ── 1. Update heating sim cell to derive diameter from area_ratio ─────────────
for cell in nb['cells']:
    if cell.get('id', '').startswith('heat_sim_01'):
        src = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])

        src = src.replace(
            "_h = _cfg['heater']\n"
            "_he_diam  = _h['diameter']\n"
            "_he_T_max = _h['max_temperature']\n"
            "_he_P_max = _h['max_power']\n"
            "_he_emis  = _h['emissivity']\n",

            "_h = _cfg['heater']\n"
            "_he_T_max = _h['max_temperature']\n"
            "_he_P_max = _h['max_power']\n"
            "_he_emis  = _h['emissivity']\n"
            # derive diameter after r_in is known (done below after geometry block)
        )

        # After r_in is computed, derive heater diameter
        src = src.replace(
            "# -- Derived geometry ---------------------------------------------------------\n",
            "# -- Derived geometry ---------------------------------------------------------\n"
        )
        src = src.replace(
            "_r_in        = _r_ps_eff * _area_ratio\n"
            "_r_out       = np.sqrt(_V_block / (np.pi * _L) + _r_in**2)\n"
            "\n"
            "assert",

            "_r_in        = _r_ps_eff * _area_ratio\n"
            "_r_out       = np.sqrt(_V_block / (np.pi * _L) + _r_in**2)\n"
            "\n"
            "# Derive heater diameter: area_ratio = D_in / D_heater if provided\n"
            "_he_ar  = _h.get('area_ratio')\n"
            "_he_diam = (2 * _r_in / _he_ar) if _he_ar else _h['diameter']\n"
            "\n"
            "assert"
        )

        cell['source'] = src
        cell['outputs'] = []
        cell['execution_count'] = None
        print('Updated heating sim cell heat_sim_01')
        break

# ── 2. Build plot cell source ─────────────────────────────────────────────────
plot_src = (
    "fig, (ax_T, ax_Q) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)\n"
    "\n"
    "_t_hr = _heat_res['time'] / 3600\n"
    "\n"
    "# -- Temperature plot ---------------------------------------------------------\n"
    "ax_T.plot(_t_hr, _heat_res['T_inner'], label='T inner wall', color='firebrick')\n"
    "ax_T.plot(_t_hr, _heat_res['T_avg'],   label='T avg block',  color='steelblue')\n"
    "ax_T.plot(_t_hr, _heat_res['T_outer'], label='T outer wall', color='seagreen')\n"
    "ax_T.plot(_t_hr, _heat_res['T_heater'],label='T heater',     color='darkorange', linestyle='--')\n"
    "ax_T.axhline(_T_block_max, color='grey', linestyle=':', linewidth=1, label=f'T_max = {_T_block_max} K')\n"
    "ax_T.axhline(_T_min,       color='grey', linestyle='-.', linewidth=1, label=f'T_min = {_T_min} K')\n"
    "ax_T.set_ylabel('Temperature [K]')\n"
    "ax_T.set_title('Heating simulation — block and heater temperatures')\n"
    "ax_T.legend(fontsize=9)\n"
    "ax_T.grid(True)\n"
    "\n"
    "# -- Power plot ---------------------------------------------------------------\n"
    "ax_Q.plot(_t_hr, _heat_res['Q_heater'] / 1e3, color='darkorange', label='Q heater')\n"
    "ax_Q.axhline(_he_P_max / 1e3, color='grey', linestyle=':', linewidth=1,\n"
    "             label=f'P_max = {_he_P_max/1e3:.0f} kW')\n"
    "ax_Q.set_xlabel('Time [hr]')\n"
    "ax_Q.set_ylabel('Heater power [kW]')\n"
    "ax_Q.set_title('Heater power delivered')\n"
    "ax_Q.legend(fontsize=9)\n"
    "ax_Q.grid(True)\n"
    "\n"
    "fig.tight_layout()\n"
    "plt.show()\n"
)

new_cell = {
    'cell_type': 'code',
    'execution_count': None,
    'id': 'heat_plot_01',
    'metadata': {},
    'outputs': [],
    'source': plot_src,
}

# Insert immediately after heat_sim_01
insert_idx = None
for i, cell in enumerate(nb['cells']):
    if cell.get('id', '').startswith('heat_sim_01'):
        insert_idx = i + 1
        break

if insert_idx is None:
    print('ERROR: heat_sim_01 not found')
else:
    nb['cells'].insert(insert_idx, new_cell)
    print(f'Inserted plot cell heat_plot_01 at index {insert_idx}')

with open('tinless system model.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Saved.')
