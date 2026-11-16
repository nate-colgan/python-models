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
            "    emis_cell_  = np.mean([row[1] for row in emissivity_table])\n"
            "    emis_bare_  = np.mean([row[2] for row in emissivity_table])\n"
            "    emis_inact_ = np.mean([row[3] for row in emissivity_table])\n"
        )
        new = (
            "    ps_tmp_     = PowerStick(axial_steps=1, width=side_len, height=L,\n"
            "                            cooling_temp=T_ps, panel_efficiency=panel_eff,\n"
            "                            f_bare=f_bare, f_active=f_active,\n"
            "                            liner_bool=liner_bool, emissivity_table=emissivity_table)\n"
            "    inc_bins_, _, _ = ps_tmp_.incident_energy(T_min, 1.0)\n"
            "    inc_tot_    = sum(inc_bins_.values())\n"
            "    emis_cell_  = sum(inc_bins_[row[0]] / inc_tot_ * row[1] for row in emissivity_table)\n"
            "    emis_bare_  = sum(inc_bins_[row[0]] / inc_tot_ * row[2] for row in emissivity_table)\n"
            "    emis_inact_ = sum(inc_bins_[row[0]] / inc_tot_ * row[3] for row in emissivity_table)\n"
        )
        if old not in src:
            # try without leading spaces (cell might not be indented)
            old = old.replace("    ", "")
            new = new.replace("    ", "")
        if old in src:
            cell['source'] = src.replace(old, new, 1)
            cell['outputs'] = []
            cell['execution_count'] = None
            changed.append(cid)
        else:
            print(f'WARNING: pattern not found in {cid}')

    # ── Cell c7d9e412: discharge bisection ───────────────────────────────
    elif cid.startswith('c7d9e412'):
        old = (
            "_emis_cell  = np.mean([row[1] for row in _emis_table])\n"
            "_emis_bare  = np.mean([row[2] for row in _emis_table])\n"
            "_emis_inact = np.mean([row[3] for row in _emis_table])\n"
        )
        new = (
            "_ps_tmp_    = PowerStick(axial_steps=1, width=_side_len, height=_L,\n"
            "                        cooling_temp=_T_ps, panel_efficiency=_panel_eff,\n"
            "                        f_bare=_f_bare, f_active=_f_active,\n"
            "                        liner_bool=_liner_bool, emissivity_table=_emis_table)\n"
            "_inc_bins_, _, _ = _ps_tmp_.incident_energy(_T_min, 1.0)\n"
            "_inc_tot_   = sum(_inc_bins_.values())\n"
            "_emis_cell  = sum(_inc_bins_[row[0]] / _inc_tot_ * row[1] for row in _emis_table)\n"
            "_emis_bare  = sum(_inc_bins_[row[0]] / _inc_tot_ * row[2] for row in _emis_table)\n"
            "_emis_inact = sum(_inc_bins_[row[0]] / _inc_tot_ * row[3] for row in _emis_table)\n"
        )
        if old in src:
            cell['source'] = src.replace(old, new, 1)
            cell['outputs'] = []
            cell['execution_count'] = None
            changed.append(cid)
        else:
            print(f'WARNING: pattern not found in {cid}')

    # ── Cell ff640d4b: parametric sweep (_build_and_run) ─────────────────
    elif cid.startswith('ff640d4b'):
        old = (
            "    emis_cell_   = np.mean([row[1] for row in emis_table])\n"
            "    emis_bare_   = np.mean([row[2] for row in emis_table])\n"
            "    emis_inact_  = np.mean([row[3] for row in emis_table])\n"
        )
        new = (
            "    ps_tmp_      = PowerStick(axial_steps=1, width=_side_len, height=_L,\n"
            "                             cooling_temp=_T_ps, panel_efficiency=_panel_eff,\n"
            "                             f_bare=_f_bare, f_active=_f_active,\n"
            "                             liner_bool=_liner_bool, emissivity_table=emis_table)\n"
            "    inc_bins_, _, _ = ps_tmp_.incident_energy(T_min_, 1.0)\n"
            "    inc_tot_     = sum(inc_bins_.values())\n"
            "    emis_cell_   = sum(inc_bins_[row[0]] / inc_tot_ * row[1] for row in emis_table)\n"
            "    emis_bare_   = sum(inc_bins_[row[0]] / inc_tot_ * row[2] for row in emis_table)\n"
            "    emis_inact_  = sum(inc_bins_[row[0]] / inc_tot_ * row[3] for row in emis_table)\n"
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
