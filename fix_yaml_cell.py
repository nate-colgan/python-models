import json

with open('tinless system model.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

old_snippet = (
    "# -- Powerstick ---------------------------------------------------------------\n"
    "powerstick:\n"
    "  side_len: 0.13           # square cross-section side length [m]\n"
    "  area_ratio: 15           # D_in / D_ps_eff\n"
    "  cooling_temp: 323        # powerstick cooling temperature [K]\n"
    "  panel_efficiency: 0.5\n"
    "  f_bare: 0.2\n"
    "  f_active: .83\n"
    "  bandgap: 1.55            # PV bandgap wavelength [microns]\n"
    '"""\n'
)

new_snippet = (
    "# -- Powerstick ---------------------------------------------------------------\n"
    "powerstick:\n"
    "  side_len: 0.13           # square cross-section side length [m]\n"
    "  area_ratio: 15           # D_in / D_ps_eff\n"
    "  cooling_temp: 323        # powerstick cooling temperature [K]\n"
    "  panel_efficiency: 0.5\n"
    "  f_bare: 0.2\n"
    "  f_active: .83\n"
    "  bandgap: 1.55            # PV bandgap wavelength [microns]\n"
    "\n"
    "# -- Heater ------------------------------------------------------------------\n"
    "# Central cylindrical resistive heating element used during charging.\n"
    "# Phase 1: max_power supplied; T_heater rises as graphite warms.\n"
    "# Phase 2: T_heater reaches max_temperature; power reduces to hold that temp.\n"
    "# Heating stops when block T_avg reaches T_min + block_temp_range.\n"
    "heater:\n"
    "  diameter: 0.05            # m   (heater rod outer diameter)\n"
    "  max_temperature: 2800     # K   (maximum heater surface temperature)\n"
    "  max_power: 500000         # W   (maximum electrical power input)\n"
    "  emissivity: 0.9           # gray-body emissivity of heater surface\n"
    '"""\n'
)

patched = False
for cell in nb['cells']:
    if cell.get('id', '').startswith('a8c3d501'):
        src = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])
        if old_snippet in src:
            src = src.replace(old_snippet, new_snippet, 1)
            cell['source'] = src
            cell['outputs'] = []
            cell['execution_count'] = None
            patched = True
            print('Patched YAML template cell a8c3d501')
        else:
            print('WARNING: expected snippet not found in a8c3d501')
        break

if not patched:
    print('ERROR: cell not found or snippet mismatch')

with open('tinless system model.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Saved notebook.')
