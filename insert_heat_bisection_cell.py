import json

with open('tinless system model.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

# Check heat_bisect_01 doesn't already exist
existing = [c.get('id','') for c in nb['cells'] if c.get('id','').startswith('heat_bisect')]
if existing:
    print(f'Already exists: {existing}. Remove them first.')
    exit()

src = (
    "# -- Find heater area_ratio such that block reaches T_block_max at end_time ---\n"
    "# Objective: T_avg(end_time) - T_block_max = 0.\n"
    "# Larger area_ratio -> larger r_in_heat -> smaller A_h/A_gr ratio\n"
    "#   -> smaller enclosure resistance -> more heat -> higher f.\n"
    "# So: f > 0 (overcharged) -> smaller area_ratio; f < 0 -> larger area_ratio.\n"
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
    "_T_min            = _s['T_min']\n"
    "_block_temp_range = _s['block_temp_range']\n"
    "_L                = _s['L']\n"
    "_th_ins           = _s['th_ins']\n"
    "_num_ax           = _s['num_ax_segments']\n"
    "_num_rad          = _s['num_rad_segments']\n"
    "_avg_flux         = _s['avg_flux']\n"
    "_duration_hr      = _s['duration_hr']\n"
    "\n"
    "_p = _cfg['powerstick']\n"
    "_side_len  = _p['side_len']\n"
    "_T_ps      = _p['cooling_temp']\n"
    "_panel_eff = _p['panel_efficiency']\n"
    "_f_bare    = _p['f_bare']\n"
    "_f_active  = _p['f_active']\n"
    "\n"
    "_raw_emis   = _cfg['emissivity_tables'][0]['value']\n"
    "_emis_table = [(tuple(row[0]), float(row[1]), float(row[2]), float(row[3]))\n"
    "               for row in _raw_emis]\n"
    "\n"
    "_h = _cfg['heater']\n"
    "_he_side_len = _h['side_len']\n"
    "_he_T_max    = _h['max_temperature']\n"
    "_he_P_max    = _h['max_power']\n"
    "_he_emis     = _h['emissivity']\n"
    "\n"
    "# -- Derived fixed quantities -------------------------------------------------\n"
    "_T_block_max   = _T_min + _block_temp_range\n"
    "_r_heater_eff  = 2 * _he_side_len / np.pi    # effective heater radius (circular equiv)\n"
    "_he_diam       = 2 * _r_heater_eff            # heater diameter passed to HeatingElement\n"
    "_A_ps          = 4 * _side_len * _L\n"
    "_T_avg_mid     = _T_min + _block_temp_range / 2\n"
    "_e_dens        = storage_rho(_T_avg_mid) * storage_cp(_T_avg_mid) * _block_temp_range\n"
    "_V_block       = _avg_flux * _A_ps * 1e4 * _duration_hr * 3600 / _e_dens\n"
    "\n"
    "print(f'V_block = {_V_block:.3f} m^3')\n"
    "print(f'T_min = {_T_min} K   T_block_max = {_T_block_max} K   end_time = {_end_time/3600:.1f} hr')\n"
    "print(f'Heater: side_len={_he_side_len} m  D_eff={_he_diam:.4f} m  '\n"
    "      f'T_max={_he_T_max} K  P_max={_he_P_max/1e3:.0f} kW')\n"
    "print()\n"
    "\n"
    "# -- Bisection objective ------------------------------------------------------\n"
    "def simulate_charge_final_T(area_ratio):\n"
    "    r_in_      = _r_heater_eff * area_ratio\n"
    "    r_out_     = np.sqrt(_V_block / (np.pi * _L) + r_in_**2)\n"
    "    init_dist_ = np.full((_num_ax, _num_rad), float(_T_min))\n"
    "    block_     = GraphiteBlock(init_dist_, inner_radius=r_in_, outer_radius=r_out_,\n"
    "                               length=_L, th_ins=_th_ins)\n"
    "    pipe_      = Pipe(OD=0.02, length=_L, wall_thickness=0.001, roughness=1e-5, th_ins=0.0)\n"
    "    storage_   = StorageModule(N_pipes=1, Pipe=pipe_, Block=block_,\n"
    "                               num_ax_segments=_num_ax, num_rad_segments=_num_rad,\n"
    "                               bypass_range=[0, 1])\n"
    "    ps_        = PowerStick(axial_steps=_num_ax, width=_side_len, height=_L,\n"
    "                            cooling_temp=_T_ps, panel_efficiency=_panel_eff,\n"
    "                            f_bare=_f_bare, f_active=_f_active,\n"
    "                            liner_bool=_liner, emissivity_table=_emis_table)\n"
    "    heater_    = HeatingElement(diameter=_he_diam, max_temperature=_he_T_max,\n"
    "                                max_power=_he_P_max, emissivity=_he_emis)\n"
    "    sim_       = TinlessSimulation(storage_, ps_, T_sur=_T_sur, block_heat_loss=_blk_loss)\n"
    "    res_       = sim_.run_heating_simulation(\n"
    "        heater=heater_, T_block_max=float('inf'),\n"
    "        end_time=_end_time, dt=_dt, sub_steps=_sub_steps,\n"
    "    )\n"
    "    T_final = res_['T_avg'][-1]\n"
    "    print(f'  area_ratio={area_ratio:6.3f}  r_in={r_in_:.4f} m  '\n"
    "          f'T_avg_final={T_final:.1f} K  (target {_T_block_max} K)')\n"
    "    return T_final - _T_block_max\n"
    "\n"
    "# -- Bisection ----------------------------------------------------------------\n"
    "_T_tol    = 0.01 * _block_temp_range\n"
    "_ar_guess = _h['area_ratio']\n"
    "\n"
    "print('Searching for area_ratio where T_avg(end_time) = T_block_max...')\n"
    "f0 = simulate_charge_final_T(_ar_guess)\n"
    "\n"
    "if abs(f0) <= _T_tol:\n"
    "    ar_opt = _ar_guess\n"
    "else:\n"
    "    if f0 > 0:          # overcharged -> need smaller area_ratio\n"
    "        ar_hi, f_hi = _ar_guess, f0\n"
    "        ar_lo = _ar_guess\n"
    "        while True:\n"
    "            ar_lo /= 2\n"
    "            f_lo = simulate_charge_final_T(ar_lo)\n"
    "            if f_lo <= 0:\n"
    "                break\n"
    "            ar_hi, f_hi = ar_lo, f_lo\n"
    "    else:               # undercharged -> need larger area_ratio\n"
    "        ar_lo, f_lo = _ar_guess, f0\n"
    "        ar_hi = _ar_guess\n"
    "        while True:\n"
    "            ar_hi *= 2\n"
    "            f_hi = simulate_charge_final_T(ar_hi)\n"
    "            if f_hi >= 0:\n"
    "                break\n"
    "            ar_lo, f_lo = ar_hi, f_hi\n"
    "\n"
    "    print(f'Bracket: [{ar_lo:.3f}, {ar_hi:.3f}]')\n"
    "\n"
    "    for _ in range(20):\n"
    "        ar_mid = (ar_lo + ar_hi) / 2\n"
    "        f_mid  = simulate_charge_final_T(ar_mid)\n"
    "        if abs(f_mid) <= _T_tol:\n"
    "            break\n"
    "        if f_mid > 0:\n"
    "            ar_hi = ar_mid\n"
    "        else:\n"
    "            ar_lo = ar_mid\n"
    "\n"
    "    ar_opt = (ar_lo + ar_hi) / 2\n"
    "\n"
    "_r_in_opt  = _r_heater_eff * ar_opt\n"
    "_r_out_opt = np.sqrt(_V_block / (np.pi * _L) + _r_in_opt**2)\n"
    "print()\n"
    "print(f'Optimal heater area_ratio = {ar_opt:.3f}')\n"
    "print(f'  r_in  = {_r_in_opt:.4f} m   D_in  = {2*_r_in_opt:.4f} m')\n"
    "print(f'  r_out = {_r_out_opt:.4f} m   D_out = {2*_r_out_opt:.4f} m')\n"
    "print(f'  D_in / D_heater = {2*_r_in_opt / _he_diam:.3f}')\n"
)

new_cell = {
    'cell_type': 'code',
    'execution_count': None,
    'id': 'heat_bisect_01',
    'metadata': {},
    'outputs': [],
    'source': src,
}

# Insert after heat_plot_01
insert_idx = None
for i, cell in enumerate(nb['cells']):
    if cell.get('id', '').startswith('heat_plot_01'):
        insert_idx = i + 1
        break

if insert_idx is None:
    print('ERROR: heat_plot_01 not found')
else:
    nb['cells'].insert(insert_idx, new_cell)
    print(f'Inserted heat_bisect_01 at index {insert_idx}')

with open('tinless system model.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Saved.')
