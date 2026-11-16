import json

with open('tinless system model.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

# ── Cell 3 (d8ee6fe0): block setup ────────────────────────────────────────────
cell3_src = [
    "# -- Energy requirements -------------------------------------------------------\n",
    "avg_power        =  2.4e5    # average power to extract [W]\n",
    "duration_hr      =  50       # storage duration [hr]\n",
    "T_min            =  2173     # minimum block temperature — defines 'empty' state [K]\n",
    "block_temp_range =  500      # operating temperature swing [K]\n",
    "\n",
    "T_max  = T_min + block_temp_range\n",
    "T_init = T_max\n",
    "T_avg  = T_min + block_temp_range / 2\n",
    "\n",
    "E_required = avg_power * duration_hr * 3600\n",
    "e_density  = storage_rho(T_avg) * storage_cp(T_avg) * block_temp_range\n",
    "V_block    = E_required / e_density\n",
    "\n",
    'print(f"Required energy:           {E_required/3.6e9:.3f} MWhr")\n',
    'print(f"Volumetric energy density: {e_density/3.6e9:.4f} MWhr/m^3")\n',
    'print(f"Required block volume:     {V_block:.3f} m^3")\n',
    "\n",
    "# -- Geometry ------------------------------------------------------------------\n",
    "side_len   =  0.05    # side length of square powerstick cross-section [m]\n",
    "area_ratio =  10      # D_in / D_ps_eff  (controls inner-wall-to-stick gap)\n",
    "L          =  3       # axial length [m]\n",
    "th_ins     =  0.0     # outer-wall insulation thickness [m]\n",
    "\n",
    "# Effective cylinder radius: same lateral surface area as the square cross-section\n",
    "#   square perimeter = 4*s  =>  2*pi*r_eff = 4*s  =>  r_eff = 2*s/pi\n",
    "r_ps_eff = 2 * side_len / np.pi\n",
    "r_in     = r_ps_eff * area_ratio\n",
    "r_out    = np.sqrt(V_block / (np.pi * L) + r_in**2)\n",
    "\n",
    'print(f"\\nSquare side:   {side_len:.4f} m  =>  r_ps_eff = {r_ps_eff:.4f} m")\n',
    'print(f"r_in  = {r_in:.4f} m   D_in  = {2*r_in:.4f} m")\n',
    'print(f"r_out = {r_out:.4f} m   D_out = {2*r_out:.4f} m")\n',
    'print(f"View factor (r_ps_eff/r_in) = {r_ps_eff/r_in:.4f}")\n',
    'print(f"Actual annulus volume       = {np.pi*(r_out**2 - r_in**2)*L:.4f} m^3")\n',
    "\n",
    "# -- Grid ----------------------------------------------------------------------\n",
    "num_ax  = 10\n",
    "num_rad = 10\n",
    "\n",
    "# -- Build objects -------------------------------------------------------------\n",
    "initial_temp_dist = np.full((num_ax, num_rad), float(T_init))\n",
    "\n",
    "block = GraphiteBlock(\n",
    "    initial_temp_distribution=initial_temp_dist,\n",
    "    inner_radius=r_in,\n",
    "    outer_radius=r_out,\n",
    "    length=L,\n",
    "    th_ins=th_ins,\n",
    ")\n",
    "\n",
    "dummy_pipe = Pipe(OD=0.02, length=L, wall_thickness=0.001, roughness=1e-5, th_ins=0.0)\n",
    "storage = StorageModule(\n",
    "    N_pipes=1,\n",
    "    Pipe=dummy_pipe,\n",
    "    Block=block,\n",
    "    num_ax_segments=num_ax,\n",
    "    num_rad_segments=num_rad,\n",
    "    bypass_range=[0, 1],\n",
    ")\n",
    "\n",
    'print(f"\\nGrid: {num_ax} ax x {num_rad} rad  |  segment_length = {storage.segment_length:.4f} m")\n',
    'print(f"T_init = {T_init} K  ({T_init - 273.15:.0f} C)")\n',
]

# ── Cell 4 (95616a8d): powerstick definition ───────────────────────────────────
cell4_src = [
    "# -- Powerstick ---------------------------------------------------------------\n",
    "panel_efficiency = 0.7    # fraction of below-bandgap absorbed power converted to electricity\n",
    "liner_bool       = False  # True = tungsten liner emissivity, False = graphite\n",
    "\n",
    "# width=side_len so A_step = 4*side_len*seg_len = 2*pi*r_ps_eff*seg_len (cylindrical equiv)\n",
    "powerstick = PowerStick(\n",
    "    axial_steps=num_ax,\n",
    "    width=side_len,\n",
    "    height=L,\n",
    "    cooling_temp=T_ps,\n",
    "    panel_efficiency=panel_efficiency,\n",
    "    f_bare=f_bare,\n",
    "    f_active=f_active,\n",
    "    liner_bool=liner_bool,\n",
    "    emissivity_table=emissivity_table,\n",
    ")\n",
    "\n",
    'print(f"PowerStick: {powerstick.num_ax_steps} axial steps  |  cooling_temp = {powerstick.cooling_temp} K")\n',
    'print(f"  side_len               = {side_len:.4f} m")\n',
    'print(f"  r_ps_eff (same area)   = {r_ps_eff:.4f} m")\n',
    'print(f"  A_step (per segment)   = {powerstick.A_step:.6f} m^2")\n',
    'print(f"  Total area             = {powerstick.A_step * num_ax:.6f} m^2  (= 4*side_len*L)")\n',
]

# ── Cell 5 (f3a91c20): heat transfer calc ─────────────────────────────────────
cell5_src = [
    "# -- Instantaneous heat transfer and power generation -------------------------\n",
    "VF      = r_ps_eff / r_in   # view factor: graphite inner wall -> powerstick\n",
    "seg_len = L / num_ax\n",
    "# Cylindrical equivalent area per segment: 2*pi*r_eff*seg_len = 4*side_len*seg_len = A_step\n",
    "A_ps_seg = 4 * side_len * seg_len\n",
    "\n",
    "Q_rad   = np.zeros(num_ax)\n",
    "P_elec  = np.zeros(num_ax)\n",
    "Q_waste = np.zeros(num_ax)\n",
    "\n",
    "for i in range(num_ax):\n",
    "    T_wall_i = storage.block.temp_dist[i, 0]\n",
    "\n",
    "    Q_rad[i] = spectral_net_radiation(\n",
    "        T_wall_i, emissivity_table, VF, A_ps_seg, f_bare, f_active, liner_bool\n",
    "    )\n",
    "    P_elec[i], Q_waste[i] = powerstick.generate_electricity_step(\n",
    "        T_wall_i, VF, panel_efficiency\n",
    "    )\n",
    "\n",
    "Q_total  = Q_rad.sum()\n",
    "P_total  = P_elec.sum()\n",
    "QW_total = Q_waste.sum()\n",
    "\n",
    "T_wall = storage.block.temp_dist[:, 0]\n",
    'print(f"Inner wall temperature: {T_wall[0]:.1f} K  ({T_wall[0]-273.15:.0f} C)")\n',
    'print(f"View factor:            {VF:.4f}")\n',
    'print(f"A_ps_seg per step:      {A_ps_seg:.6f} m^2  (= powerstick.A_step = {powerstick.A_step:.6f} m^2)")\n',
    'print()\n',
    'print(f"Total Q to powerstick:  {Q_total/1e3:.2f} kW")\n',
    'print(f"Total electrical power: {P_total/1e3:.2f} kW")\n',
    'print(f"Total waste heat:       {QW_total/1e3:.2f} kW")\n',
    'print(f"Effective efficiency:   {P_total/(P_total+QW_total)*100:.1f}%")\n',
]

target_sources = {
    'd8ee6fe0': cell3_src,
    '95616a8d': cell4_src,
    'f3a91c20': cell5_src,
}

for cell in nb['cells']:
    cid = cell.get('id', '')
    for prefix, src in target_sources.items():
        if cid.startswith(prefix):
            cell['source'] = src
            cell['outputs'] = []
            cell['execution_count'] = None
            print(f'Updated cell {cid[:8]}')

with open('tinless system model.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Done.')
