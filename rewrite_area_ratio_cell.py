import json

with open('tinless system model.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

src = [
    "# -- Find area_ratio such that avg Q in pseudo-steady-state = avg_power ------\n",
    "# Pseudo-steady-state (PSS) begins when T_inner - T_avg stops becoming more\n",
    "# negative, i.e. at the minimum of delta = T_inner - T_avg (peak gradient).\n",
    "# We average Q_rad from that point until SOC = 0, and bisect on area_ratio\n",
    "# until that average equals the design value avg_power.\n",
    "\n",
    "def simulate_pss_avg_power(area_ratio):\n",
    "    r_ps_eff_ = 2 * side_len / np.pi\n",
    "    r_in_     = r_ps_eff_ * area_ratio\n",
    "    r_out_    = np.sqrt(V_block / (np.pi * L) + r_in_**2)\n",
    "    init_dist = np.full((num_ax, num_rad), float(T_init))\n",
    "    block_    = GraphiteBlock(init_dist, inner_radius=r_in_, outer_radius=r_out_,\n",
    "                              length=L, th_ins=th_ins)\n",
    "    pipe_     = Pipe(OD=0.02, length=L, wall_thickness=0.001, roughness=1e-5, th_ins=0.0)\n",
    "    storage_  = StorageModule(N_pipes=1, Pipe=pipe_, Block=block_,\n",
    "                              num_ax_segments=num_ax, num_rad_segments=num_rad,\n",
    "                              bypass_range=[0, 1])\n",
    "    ps_       = PowerStick(axial_steps=num_ax, width=side_len, height=L,\n",
    "                           cooling_temp=T_ps, panel_efficiency=panel_eff,\n",
    "                           f_bare=f_bare, f_active=f_active,\n",
    "                           liner_bool=liner_bool, emissivity_table=emissivity_table)\n",
    "    sim_      = TinlessSimulation(storage_, ps_, T_sur=T_sur,\n",
    "                                  block_heat_loss=block_heat_loss_bool)\n",
    "    res_      = sim_.run_simulation(end_time=end_time, dt=dt, sub_steps=sub_steps)\n",
    "\n",
    "    delta = res_['T_inner'] - res_['T_avg']\n",
    "    soc   = (res_['T_avg'] - T_min) / (T_max - T_min)\n",
    "\n",
    "    # PSS start: index of minimum delta (most negative = largest gradient)\n",
    "    idx_pss  = int(np.argmin(delta))\n",
    "    t_pss_hr = res_['time'][idx_pss] / 3600\n",
    "\n",
    "    # SOC = 0: first timestep where T_avg <= T_min\n",
    "    soc0_mask = np.where(soc <= 0)[0]\n",
    "    if len(soc0_mask) > 0:\n",
    "        idx_soc0  = int(soc0_mask[0])\n",
    "        t_soc0_hr = res_['time'][idx_soc0] / 3600\n",
    "        soc0_str  = f'SOC=0 @ {t_soc0_hr:.1f} hr'\n",
    "    else:\n",
    "        idx_soc0  = len(soc) - 1\n",
    "        soc0_str  = f'SOC={soc[-1]:.2f} at end (never reached 0)'\n",
    "\n",
    "    # Average Q over [PSS start, SOC=0]\n",
    "    if idx_pss >= idx_soc0:\n",
    "        # PSS occurs after (or at) depletion -- use single-point value\n",
    "        avg_Q = float(res_['Q_rad'][idx_pss])\n",
    "    else:\n",
    "        avg_Q = float(res_['Q_rad'][idx_pss:idx_soc0 + 1].mean())\n",
    "\n",
    "    print(f'  area_ratio = {area_ratio:6.3f}  '\n",
    "          f'PSS @ {t_pss_hr:.1f} hr  {soc0_str}  '\n",
    "          f'avg_Q = {avg_Q/1e3:.2f} kW  (target {avg_power/1e3:.2f} kW)')\n",
    "    return avg_Q - avg_power\n",
    "\n",
    "pwr_tol  = 0.02   # accept within 2% of avg_power\n",
    "ar_guess = 10.0\n",
    "\n",
    "print('Searching for area_ratio where avg PSS power = avg_power...')\n",
    "f0 = simulate_pss_avg_power(ar_guess)\n",
    "\n",
    "if abs(f0) / avg_power <= pwr_tol:\n",
    "    ar_opt = ar_guess\n",
    "else:\n",
    "    # Expand bracket from initial guess\n",
    "    if f0 > 0:          # too much heat -> need larger area_ratio (smaller VF)\n",
    "        ar_lo, f_lo = ar_guess, f0\n",
    "        ar_hi = ar_guess\n",
    "        while True:\n",
    "            ar_hi *= 2\n",
    "            f_hi = simulate_pss_avg_power(ar_hi)\n",
    "            if f_hi <= 0:\n",
    "                break\n",
    "            ar_lo, f_lo = ar_hi, f_hi\n",
    "    else:               # too little heat -> need smaller area_ratio (larger VF)\n",
    "        ar_hi, f_hi = ar_guess, f0\n",
    "        ar_lo = ar_guess\n",
    "        while True:\n",
    "            ar_lo /= 2\n",
    "            f_lo = simulate_pss_avg_power(ar_lo)\n",
    "            if f_lo >= 0:\n",
    "                break\n",
    "            ar_hi, f_hi = ar_lo, f_lo\n",
    "\n",
    "    print(f'Bracket: [{ar_lo:.3f}, {ar_hi:.3f}]')\n",
    "\n",
    "    for _ in range(20):\n",
    "        ar_mid = (ar_lo + ar_hi) / 2\n",
    "        f_mid  = simulate_pss_avg_power(ar_mid)\n",
    "        if abs(f_mid) / avg_power <= pwr_tol:\n",
    "            break\n",
    "        if f_mid > 0:   # still too much heat\n",
    "            ar_lo = ar_mid\n",
    "        else:\n",
    "            ar_hi = ar_mid\n",
    "\n",
    "    ar_opt = (ar_lo + ar_hi) / 2\n",
    "\n",
    "r_in_opt  = 2 * side_len / np.pi * ar_opt\n",
    "r_out_opt = np.sqrt(V_block / (np.pi * L) + r_in_opt**2)\n",
    "print()\n",
    "print(f'Optimal area_ratio = {ar_opt:.3f}')\n",
    "print(f'  r_in  = {r_in_opt:.4f} m   D_in  = {2*r_in_opt:.4f} m')\n",
    "print(f'  r_out = {r_out_opt:.4f} m   D_out = {2*r_out_opt:.4f} m')\n",
    "print(f'  VF    = {1/ar_opt:.4f}')\n",
]

# Replace cell c7d9e412
replaced = False
for cell in nb['cells']:
    if cell.get('id', '').startswith('c7d9e412'):
        cell['source'] = src
        cell['outputs'] = []
        cell['execution_count'] = None
        replaced = True
        print('Replaced cell c7d9e412')
        break

if not replaced:
    print('ERROR: cell c7d9e412 not found')

with open('tinless system model.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Done.')
