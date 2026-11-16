import json

with open('tinless system model.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

src = (
    '# ── Energy requirements ───────────────────────────────────────────────────────\n'
    'avg_power       =        # average power to extract [W]\n'
    'duration_hr     =        # storage duration [hr]\n'
    'T_min           =        # minimum block temperature — defines "empty" state [K]\n'
    'block_temp_range =       # operating temperature swing [K]  (T_max = T_min + block_temp_range)\n'
    '\n'
    'T_max  = T_min + block_temp_range\n'
    'T_init = T_max            # start fully charged\n'
    'T_avg  = T_min + block_temp_range / 2\n'
    '\n'
    '# Required thermal energy\n'
    'E_required = avg_power * duration_hr * 3600  # J\n'
    '\n'
    '# Volumetric energy density: rho * cp * dT  (evaluated at average temperature,\n'
    '# consistent with the storage_rho / storage_cp used in update_temps)\n'
    'e_density = storage_rho(T_avg) * storage_cp(T_avg) * block_temp_range  # J/m^3\n'
    '\n'
    'V_block = E_required / e_density\n'
    '\n'
    'print(f"Required energy:          {E_required/3.6e6:.3f} MWhr")\n'
    'print(f"Volumetric energy density: {e_density/3.6e6:.4f} MWhr/m^3")\n'
    'print(f"Required block volume:     {V_block:.3f} m^3")\n'
    '\n'
    '# ── Geometry ──────────────────────────────────────────────────────────────────\n'
    'D_ps        =        # powerstick outer diameter [m]\n'
    'area_ratio  =        # A_inner_wall / A_ps = D_in / D_ps  (controls view factor and gap)\n'
    'L           =        # axial length [m]\n'
    'th_ins      = 0.0    # outer-wall insulation thickness [m]\n'
    '\n'
    'r_ps  = D_ps / 2\n'
    'r_in  = r_ps * area_ratio                             # inner graphite radius from area ratio\n'
    'r_out = np.sqrt(V_block / (np.pi * L) + r_in**2)     # outer radius from volume + geometry\n'
    '\n'
    'print(f"\\nr_in  = {r_in:.4f} m   D_in  = {2*r_in:.4f} m")\n'
    'print(f"r_out = {r_out:.4f} m   D_out = {2*r_out:.4f} m")\n'
    'print(f"View factor (D_ps/D_in) = {r_ps/r_in:.4f}")\n'
    'print(f"Actual annulus volume   = {np.pi*(r_out**2 - r_in**2)*L:.4f} m^3")\n'
    '\n'
    '# ── Grid ──────────────────────────────────────────────────────────────────────\n'
    'num_ax  =       # number of axial segments\n'
    'num_rad =       # number of radial segments\n'
    '\n'
    '# ── Build objects ─────────────────────────────────────────────────────────────\n'
    'initial_temp_dist = np.full((num_ax, num_rad), float(T_init))\n'
    '\n'
    'block = GraphiteBlock(\n'
    '    initial_temp_distribution=initial_temp_dist,\n'
    '    inner_radius=r_in,\n'
    '    outer_radius=r_out,\n'
    '    length=L,\n'
    '    th_ins=th_ins,\n'
    ')\n'
    '\n'
    '# StorageModule supplies update_temps; dummy Pipe only sets segment_length via pipe.length\n'
    'dummy_pipe = Pipe(OD=0.02, length=L, wall_thickness=0.001, roughness=1e-5, th_ins=0.0)\n'
    'storage = StorageModule(\n'
    '    N_pipes=1,\n'
    '    Pipe=dummy_pipe,\n'
    '    Block=block,\n'
    '    num_ax_segments=num_ax,\n'
    '    num_rad_segments=num_rad,\n'
    '    bypass_range=[0, 1],\n'
    ')\n'
    '\n'
    'print(f"\\nGrid: {num_ax} ax x {num_rad} rad  |  segment_length = {storage.segment_length:.4f} m")\n'
    'print(f"T_init = {T_init} K  ({T_init - 273.15:.0f} C)")\n'
)

# Replace cell at index 3 (the one we inserted previously)
nb['cells'][3]['source'] = src
nb['cells'][3]['outputs'] = []
nb['cells'][3]['execution_count'] = None

with open('tinless system model.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print('Done.')
