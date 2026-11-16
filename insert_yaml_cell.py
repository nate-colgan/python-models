import json

with open('tinless system model.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

src = [
    "yaml_content = \"\"\"\n",
    "# Tinless System Model - Simulation Input\n",
    "# Graphite storage block radiating directly to a central powerstick (no tin piping)\n",
    "\n",
    "# -- General Simulation Parameters --------------------------------------------\n",
    "end_time: 180000          # total simulation time [s]  (50 hr discharge)\n",
    "dt: 60                    # time step [s]\n",
    "sub_steps: 5              # FD sub-steps per time step (stability)\n",
    "T_sur: 300                # surroundings temperature [K]\n",
    "block_heat_loss_bool: False\n",
    "liner_bool: False         # True = tungsten liner emissivity, False = bare graphite\n",
    "print_outputs_bool: True\n",
    "\n",
    "# -- Emissivity Table ---------------------------------------------------------\n",
    "# Format per band: [[wl_min, wl_max], emis_active, emis_bare, emis_inactive]\n",
    "# Use range format to allow time-varying tables (e.g. panels offline during charging)\n",
    "emissivity_tables:\n",
    "  - range: [0, 179999]\n",
    "    value:\n",
    "      - [[0.1, 1.24], 0.95, 0.05, 0.2]   # below bandgap\n",
    "      - [[1.24, 20.0], 0.05, 0.05, 0.2]  # above bandgap\n",
    "\n",
    "# -- Storage Block ------------------------------------------------------------\n",
    "storage:\n",
    "  avg_power: 2.4e5         # average power to extract [W]\n",
    "  duration_hr: 50          # discharge duration [hr]\n",
    "  T_min: 2173              # minimum (empty) block temperature [K]\n",
    "  block_temp_range: 500    # operating temperature swing [K]\n",
    "  L: 3.0                   # axial length [m]\n",
    "  th_ins: 0.0              # outer-wall insulation thickness [m]\n",
    "  num_ax_segments: 10\n",
    "  num_rad_segments: 10\n",
    "\n",
    "# -- Powerstick ---------------------------------------------------------------\n",
    "powerstick:\n",
    "  side_len: 0.05           # square cross-section side length [m]\n",
    "  area_ratio: 10           # D_in / D_ps_eff\n",
    "  cooling_temp: 323        # powerstick cooling temperature [K]\n",
    "  panel_efficiency: 0.7\n",
    "  f_bare: 0.2\n",
    "  f_active: 0.85\n",
    "  bandgap: 1.24            # PV bandgap wavelength [microns]\n",
    "\"\"\"\n",
    "\n",
    "with open('tinless_model_input.yaml', 'w') as f:\n",
    "    f.write(yaml_content)\n",
    "\n",
    "print('Wrote tinless_model_input.yaml')\n",
]

yaml_cell = {
    'cell_type': 'code',
    'execution_count': None,
    'id': 'a8c3d501',
    'metadata': {},
    'outputs': [],
    'source': src,
}

# Insert before the run cell (id b2e4f810), which is currently at index 7
nb['cells'].insert(7, yaml_cell)

with open('tinless system model.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f'Inserted YAML cell at index 7. Total cells: {len(nb["cells"])}')
