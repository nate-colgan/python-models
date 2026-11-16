import json

with open('tinless system model.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

changed = []

for cell in nb['cells']:
    cid = cell.get('id', '')
    src = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])

    # ── Cell b2e4f810 ────────────────────────────────────────────────────
    if cid.startswith('b2e4f810'):
        old = (
            "emis_cell_  = sum(inc_bins_[row[0]] / inc_tot_ * row[1] for row in emissivity_table)\n"
            "emis_bare_  = sum(inc_bins_[row[0]] / inc_tot_ * row[2] for row in emissivity_table)\n"
            "emis_inact_ = sum(inc_bins_[row[0]] / inc_tot_ * row[3] for row in emissivity_table)\n"
            "emis_avg_   = f_cell_ * emis_cell_ + f_bare * emis_bare_ + f_inact_ * emis_inact_\n"
            "print(emis_avg_, emis_cell_, emis_bare_, emis_inact_, inc_bins_)\n"
            "E_req   = avg_flux * 1e4 * A_ps * (emis_avg_ / emis_cell_) * duration_hr * 3600\n"
            "print(f\"Required power: {avg_flux * 1e4 * A_ps * (emis_avg_ / emis_cell_):.3f}\")"
        )
        new = (
            "f_Q_        = sum(\n"
            "    inc_bins_[row[0]] / inc_tot_ * f_cell_ * row[1]\n"
            "    / (f_bare * row[2] + f_cell_ * row[1] + f_inact_ * row[3])\n"
            "    for row in emissivity_table)\n"
            "E_req   = avg_flux * 1e4 * A_ps * f_cell_ / f_Q_ * duration_hr * 3600"
        )
        if old in src:
            cell['source'] = src.replace(old, new, 1)
            cell['outputs'] = []
            cell['execution_count'] = None
            changed.append(cid)
        else:
            print(f'WARNING: pattern not found in {cid}')

    # ── Cell c7d9e412 ────────────────────────────────────────────────────
    elif cid.startswith('c7d9e412'):
        old = (
            "_emis_cell  = sum(_inc_bins_[row[0]] / _inc_tot_ * row[1] for row in _emis_table)\n"
            "_emis_bare  = sum(_inc_bins_[row[0]] / _inc_tot_ * row[2] for row in _emis_table)\n"
            "_emis_inact = sum(_inc_bins_[row[0]] / _inc_tot_ * row[3] for row in _emis_table)\n"
            "_emis_avg   = _f_cell * _emis_cell + _f_bare * _emis_bare + _f_inact * _emis_inact\n"
            "_V_block    = _avg_flux * 1e4 * _A_ps * (_emis_avg / _emis_cell) * _duration_hr * 3600 / _e_dens"
        )
        new = (
            "_f_Q        = sum(\n"
            "    _inc_bins_[row[0]] / _inc_tot_ * _f_cell * row[1]\n"
            "    / (_f_bare * row[2] + _f_cell * row[1] + _f_inact * row[3])\n"
            "    for row in _emis_table)\n"
            "_V_block    = _avg_flux * 1e4 * _A_ps * _f_cell / _f_Q * _duration_hr * 3600 / _e_dens"
        )
        if old in src:
            cell['source'] = src.replace(old, new, 1)
            cell['outputs'] = []
            cell['execution_count'] = None
            changed.append(cid)
        else:
            print(f'WARNING: pattern not found in {cid}')

    # ── Cell ff640d4b ────────────────────────────────────────────────────
    elif cid.startswith('ff640d4b'):
        old = (
            "    emis_cell_   = sum(inc_bins_[row[0]] / inc_tot_ * row[1] for row in emis_table)\n"
            "    emis_bare_   = sum(inc_bins_[row[0]] / inc_tot_ * row[2] for row in emis_table)\n"
            "    emis_inact_  = sum(inc_bins_[row[0]] / inc_tot_ * row[3] for row in emis_table)\n"
            "    emis_avg_    = f_cell_ * emis_cell_ + _f_bare * emis_bare_ + f_inact_ * emis_inact_\n"
            "    tot_power_   = _avg_flux * 1e4 * 4 * _side_len * _L * (emis_avg_ / emis_cell_)"
        )
        new = (
            "    f_Q_         = sum(\n"
            "        inc_bins_[row[0]] / inc_tot_ * f_cell_ * row[1]\n"
            "        / (_f_bare * row[2] + f_cell_ * row[1] + f_inact_ * row[3])\n"
            "        for row in emis_table)\n"
            "    tot_power_   = _avg_flux * 1e4 * 4 * _side_len * _L * f_cell_ / f_Q_"
        )
        if old in src:
            cell['source'] = src.replace(old, new, 1)
            cell['outputs'] = []
            cell['execution_count'] = None
            changed.append(cid)
        else:
            print(f'WARNING: pattern not found in {cid}')

print(f'Changed: {changed}')

with open('tinless system model.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Saved.')
