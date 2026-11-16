import json

with open('tinless system model.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

src = (
    "# -- Heating simulation ------------------------------------------------------\n"
    "# Loads all parameters from YAML. Block starts at T_min (discharged state) and\n"
    "# is heated to T_min + block_temp_range. Heater diameter must fit inside r_in.\n"
    "\n"
    "with open('tinless_model_input.yaml') as _f:\n"
    "    _cfg = yaml.safe_load(_f)\n"
    "\n"
    "_end_time  = _cfg['end_time']\n"
    "_dt        = _cfg['dt']\n"
    "_sub_steps = _cfg['sub_steps']\n"
    "_T_sur     = _cfg['T_sur']\n"
    "_blk_loss  = _cfg['block_heat_loss_bool']\n"
    "_liner     = _cfg['liner_bool']\n"
    "\n"
    "_s = _cfg['storage']\n"
    "_T_min          = _s['T_min']\n"
    "_block_temp_range = _s['block_temp_range']\n"
    "_L              = _s['L']\n"
    "_th_ins         = _s['th_ins']\n"
    "_num_ax         = _s['num_ax_segments']\n"
    "_num_rad        = _s['num_rad_segments']\n"
    "_avg_flux       = _s['avg_flux']\n"
    "_duration_hr    = _s['duration_hr']\n"
    "\n"
    "_p = _cfg['powerstick']\n"
    "_side_len   = _p['side_len']\n"
    "_area_ratio = _p['area_ratio']\n"
    "_T_ps       = _p['cooling_temp']\n"
    "_panel_eff  = _p['panel_efficiency']\n"
    "_f_bare     = _p['f_bare']\n"
    "_f_active   = _p['f_active']\n"
    "\n"
    "_h = _cfg['heater']\n"
    "_he_diam  = _h['diameter']\n"
    "_he_T_max = _h['max_temperature']\n"
    "_he_P_max = _h['max_power']\n"
    "_he_emis  = _h['emissivity']\n"
    "\n"
    "# -- Derived geometry ---------------------------------------------------------\n"
    "_T_block_max = _T_min + _block_temp_range\n"
    "_r_ps_eff    = 2 * _side_len / np.pi\n"
    "_A_ps        = 4 * _side_len * _L\n"
    "_T_avg_mid   = _T_min + _block_temp_range / 2\n"
    "_e_dens      = storage_rho(_T_avg_mid) * storage_cp(_T_avg_mid) * _block_temp_range\n"
    "_V_block     = _avg_flux * _A_ps * 1e4 * _duration_hr * 3600 / _e_dens\n"
    "_r_in        = _r_ps_eff * _area_ratio\n"
    "_r_out       = np.sqrt(_V_block / (np.pi * _L) + _r_in**2)\n"
    "\n"
    "assert _he_diam / 2 < _r_in, (\n"
    "    f'Heater radius {_he_diam/2:.3f} m >= block inner radius {_r_in:.3f} m -- '\n"
    "    f'reduce heater diameter or area_ratio'\n"
    ")\n"
    "\n"
    "print(f'Block:  r_in={_r_in:.3f} m  r_out={_r_out:.3f} m  L={_L} m')\n"
    "print(f'Temps:  T_init={_T_min} K  T_target={_T_block_max} K')\n"
    "print(f'Heater: D={_he_diam} m  T_max={_he_T_max} K  P_max={_he_P_max/1e3:.0f} kW  emis={_he_emis}')\n"
    "print()\n"
    "\n"
    "# -- Build components ---------------------------------------------------------\n"
    "_init_dist = np.full((_num_ax, _num_rad), float(_T_min))\n"
    "_block   = GraphiteBlock(_init_dist, inner_radius=_r_in, outer_radius=_r_out,\n"
    "                          length=_L, th_ins=_th_ins)\n"
    "_pipe    = Pipe(OD=0.02, length=_L, wall_thickness=0.001, roughness=1e-5, th_ins=0.0)\n"
    "_storage = StorageModule(N_pipes=1, Pipe=_pipe, Block=_block,\n"
    "                          num_ax_segments=_num_ax, num_rad_segments=_num_rad,\n"
    "                          bypass_range=[0, 1])\n"
    "\n"
    "# PowerStick needed for TinlessSimulation.__init__; not used during heating\n"
    "_raw_emis   = _cfg['emissivity_tables'][0]['value']\n"
    "_emis_table = [(tuple(row[0]), float(row[1]), float(row[2]), float(row[3]))\n"
    "               for row in _raw_emis]\n"
    "_ps = PowerStick(axial_steps=_num_ax, width=_side_len, height=_L,\n"
    "                 cooling_temp=_T_ps, panel_efficiency=_panel_eff,\n"
    "                 f_bare=_f_bare, f_active=_f_active,\n"
    "                 liner_bool=_liner, emissivity_table=_emis_table)\n"
    "\n"
    "_heater = HeatingElement(diameter=_he_diam, max_temperature=_he_T_max,\n"
    "                          max_power=_he_P_max, emissivity=_he_emis)\n"
    "\n"
    "# -- Run ----------------------------------------------------------------------\n"
    "_sim = TinlessSimulation(_storage, _ps, T_sur=_T_sur, block_heat_loss=_blk_loss)\n"
    "\n"
    "_heat_res = _sim.run_heating_simulation(\n"
    "    heater      = _heater,\n"
    "    T_block_max = _T_block_max,\n"
    "    end_time    = _end_time,\n"
    "    dt          = _dt,\n"
    "    sub_steps   = _sub_steps,\n"
    "    snap_every  = 3600,\n"
    "    print_interval = int(3600 / _dt),\n"
    ")\n"
    "\n"
    "# -- Summary ------------------------------------------------------------------\n"
    "_t_hr = _heat_res['time'] / 3600\n"
    "print(f'\\nCharge time:    {_t_hr[-1]:.2f} hr')\n"
    "print(f'T_avg final:    {_heat_res[\"T_avg\"][-1]:.1f} K  (target {_T_block_max} K)')\n"
    "print(f'T_heater range: {_heat_res[\"T_heater\"].min():.0f} -- {_heat_res[\"T_heater\"].max():.0f} K')\n"
    "print(f'Q_heater range: {_heat_res[\"Q_heater\"][1:].min()/1e3:.1f} -- {_heat_res[\"Q_heater\"].max()/1e3:.1f} kW')\n"
)

new_cell = {
    'cell_type': 'code',
    'execution_count': None,
    'id': 'heat_sim_01',
    'metadata': {},
    'outputs': [],
    'source': src,
}

# Insert after c7d9e412 (standalone bisection, index 10)
insert_idx = None
for i, cell in enumerate(nb['cells']):
    if cell.get('id', '').startswith('c7d9e412'):
        insert_idx = i + 1
        break

if insert_idx is None:
    print('ERROR: c7d9e412 not found')
else:
    nb['cells'].insert(insert_idx, new_cell)
    print(f'Inserted heating sim cell at index {insert_idx}')

with open('tinless system model.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Saved.')
