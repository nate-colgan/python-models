import json

with open('tinless system model.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

changed = []

for cell in nb['cells']:
    cid = cell.get('id', '')
    src = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])

    # ── Cell b2e4f810: main discharge sim ────────────────────────────────
    if cid.startswith('b2e4f810'):
        old = (
            "E_req   = avg_flux * A_ps * 1e4 * duration_hr * 3600\n"
            "e_dens  = storage_rho(T_avg_) * storage_cp(T_avg_) * block_temp_range\n"
            "V_block = E_req / e_dens"
        )
        new = (
            "f_cell_     = (1.0 - f_bare) * f_active\n"
            "f_inact_    = (1.0 - f_bare) * (1.0 - f_active)\n"
            "emis_cell_  = np.mean([row[1] for row in emissivity_table])\n"
            "emis_bare_  = np.mean([row[2] for row in emissivity_table])\n"
            "emis_inact_ = np.mean([row[3] for row in emissivity_table])\n"
            "emis_avg_   = f_cell_ * emis_cell_ + f_bare * emis_bare_ + f_inact_ * emis_inact_\n"
            "E_req   = avg_flux * 1e4 * A_ps * (emis_avg_ / emis_cell_) * duration_hr * 3600\n"
            "e_dens  = storage_rho(T_avg_) * storage_cp(T_avg_) * block_temp_range\n"
            "V_block = E_req / e_dens"
        )
        if old in src:
            cell['source'] = src.replace(old, new, 1)
            cell['outputs'] = []
            cell['execution_count'] = None
            changed.append(cid)
        else:
            print(f'WARNING: old string not found in {cid}')

    # ── Cell c7d9e412: discharge bisection ───────────────────────────────
    elif cid.startswith('c7d9e412'):
        old = (
            "_A_ps       = 4 * _side_len * _L           # total powerstick surface area [m^2]\n"
            "_e_dens     = storage_rho(_T_min + _dT_range / 2) * storage_cp(_T_min + _dT_range / 2) * _dT_range\n"
            "_V_block    = _avg_flux * _A_ps * 1e4 * _duration_hr * 3600 / _e_dens\n"
            "_avg_power  = _avg_flux * _A_ps * 1e4 * (1 - _f_bare) * _f_active  # cell-weighted target [W]"
        )
        new = (
            "_A_ps       = 4 * _side_len * _L           # total powerstick surface area [m^2]\n"
            "_f_cell     = (1.0 - _f_bare) * _f_active\n"
            "_f_inact    = (1.0 - _f_bare) * (1.0 - _f_active)\n"
            "_emis_cell  = np.mean([row[1] for row in _emis_table])\n"
            "_emis_bare  = np.mean([row[2] for row in _emis_table])\n"
            "_emis_inact = np.mean([row[3] for row in _emis_table])\n"
            "_emis_avg   = _f_cell * _emis_cell + _f_bare * _emis_bare + _f_inact * _emis_inact\n"
            "_e_dens     = storage_rho(_T_min + _dT_range / 2) * storage_cp(_T_min + _dT_range / 2) * _dT_range\n"
            "_V_block    = _avg_flux * 1e4 * _A_ps * (_emis_avg / _emis_cell) * _duration_hr * 3600 / _e_dens\n"
            "_avg_power  = _avg_flux * 1e4 * _A_ps * _f_cell  # cell-weighted target [W]"
        )
        if old in src:
            cell['source'] = src.replace(old, new, 1)
            cell['outputs'] = []
            cell['execution_count'] = None
            changed.append(cid)
        else:
            print(f'WARNING: old string not found in {cid}')

    # ── Cell ff640d4b: parametric sweep ──────────────────────────────────
    elif cid.startswith('ff640d4b'):
        # 1. Remove _avg_power_tot from top-level; recompute _avg_power directly
        old1 = (
            "_avg_power_tot = _avg_flux * 4 * _side_len * _L * 1e4   # total thermal [W]\n"
            "_avg_power     = _avg_power_tot * (1 - _f_bare) * _f_active  # cell-weighted [W]"
        )
        new1 = (
            "_avg_power = _avg_flux * 1e4 * 4 * _side_len * _L * (1.0 - _f_bare) * _f_active  # cell-weighted [W]"
        )
        # 2. Inside _build_and_run, replace _avg_power_tot with emis-weighted formula
        old2 = (
            "    T_min_    = T_max - dT_range\n"
            "    T_avg_mid = T_min_ + dT_range / 2\n"
            "    e_dens_   = storage_rho(T_avg_mid) * storage_cp(T_avg_mid) * dT_range\n"
            "    V_blk_    = _avg_power_tot * _duration_hr * 3600 / e_dens_"
        )
        new2 = (
            "    T_min_    = T_max - dT_range\n"
            "    T_avg_mid = T_min_ + dT_range / 2\n"
            "    e_dens_   = storage_rho(T_avg_mid) * storage_cp(T_avg_mid) * dT_range\n"
            "    f_cell_      = (1.0 - _f_bare) * _f_active\n"
            "    f_inact_     = (1.0 - _f_bare) * (1.0 - _f_active)\n"
            "    emis_cell_   = np.mean([row[1] for row in emis_table])\n"
            "    emis_bare_   = np.mean([row[2] for row in emis_table])\n"
            "    emis_inact_  = np.mean([row[3] for row in emis_table])\n"
            "    emis_avg_    = f_cell_ * emis_cell_ + _f_bare * emis_bare_ + f_inact_ * emis_inact_\n"
            "    tot_power_   = _avg_flux * 1e4 * 4 * _side_len * _L * (emis_avg_ / emis_cell_)\n"
            "    V_blk_    = tot_power_ * _duration_hr * 3600 / e_dens_"
        )
        if old1 not in src:
            print(f'WARNING: old1 string not found in {cid}')
        elif old2 not in src:
            print(f'WARNING: old2 string not found in {cid}')
        else:
            src = src.replace(old1, new1, 1)
            src = src.replace(old2, new2, 1)
            cell['source'] = src
            cell['outputs'] = []
            cell['execution_count'] = None
            changed.append(cid)

print(f'Changed: {changed}')

with open('tinless system model.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Saved.')
