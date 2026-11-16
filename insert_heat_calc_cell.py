import json

with open('tinless system model.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

lines = [
    "# -- Instantaneous heat transfer and power generation ------------------------\n",
    "# Inner wall = storage.Block.temp_distribution[:, 0]  (innermost radial node)\n",
    "VF       = r_ps / r_in        # view factor: graphite inner wall -> powerstick (D_ps/D_in)\n",
    "seg_len  = L / num_ax         # axial segment length [m]\n",
    "A_ps_seg = np.pi * D_ps * seg_len   # cylindrical powerstick area per segment [m^2]\n",
    "\n",
    "Q_rad   = np.zeros(num_ax)    # heat leaving inner wall per segment [W]  (cylindrical area)\n",
    "P_elec  = np.zeros(num_ax)    # electrical power per segment [W]\n",
    "Q_waste = np.zeros(num_ax)    # waste heat per segment [W]\n",
    "\n",
    "for i in range(num_ax):\n",
    "    T_wall_i = storage.Block.temp_distribution[i, 0]   # innermost node of segment i\n",
    "\n",
    "    Q_rad[i] = spectral_net_radiation(\n",
    "        T_wall_i, emissivity_table, VF, A_ps_seg, f_bare, f_active, liner_bool\n",
    "    )\n",
    "    # generate_electricity_step uses A_step = 4*D_ps*seg_len (rectangular, ~27% > cylindrical)\n",
    "    P_elec[i], Q_waste[i] = powerstick.generate_electricity_step(\n",
    "        T_wall_i, VF, panel_efficiency\n",
    "    )\n",
    "\n",
    "Q_total  = Q_rad.sum()\n",
    "P_total  = P_elec.sum()\n",
    "QW_total = Q_waste.sum()\n",
    "\n",
    "T_wall = storage.Block.temp_distribution[:, 0]\n",
    'print(f"Inner wall temperature:     {T_wall[0]:.1f} K  ({T_wall[0]-273.15:.0f} C)")\n',
    'print(f"View factor (D_ps/D_in):    {VF:.4f}")\n',
    'print(f"A_ps total (cylindrical):   {np.pi*D_ps*L:.4f} m^2")\n',
    'print(f"A_step total (rectangular): {powerstick.A_step*num_ax:.4f} m^2")\n',
    'print()\n',
    'print(f"Total Q to powerstick:      {Q_total/1e3:.2f} kW  (cylindrical)")\n',
    'print(f"Total electrical power:     {P_total/1e3:.2f} kW")\n',
    'print(f"Total waste heat:           {QW_total/1e3:.2f} kW")\n',
    'print(f"Effective efficiency:       {P_total/(P_total+QW_total)*100:.1f}%")\n',
]

new_cell = {
    'cell_type': 'code',
    'execution_count': None,
    'id': 'f3a91c20',
    'metadata': {},
    'outputs': [],
    'source': lines,
}

nb['cells'].insert(5, new_cell)

with open('tinless system model.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f'Inserted cell at index 5. Total cells: {len(nb["cells"])}')
