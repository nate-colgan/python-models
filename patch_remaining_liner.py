import json

with open('tinless system model.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

patched = []

for cell in nb['cells']:
    cid = cell.get('id', '')
    src = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])
    orig = src

    # ── b2e4f810 ─────────────────────────────────────────────────────────────
    if cid.startswith('b2e4f810'):
        # 1. Read liner_side_len from YAML (after T_ps)
        src = src.replace(
            "T_ps           = p['cooling_temp']\n",
            "T_ps           = p['cooling_temp']\n"
            "liner_side_len = p['liner_side_len']\n",
            1,
        )
        # 2. ps_tmp_ — add liner_width (note: continuation lines have zero indent)
        src = src.replace(
            'ps_tmp_ = PowerStick(axial_steps=1, width=side_len, height=L,\n'
            'cooling_temp=T_ps,\n'
            'f_bare=f_bare, f_active=f_active,\n'
            'liner_bool=liner_bool, emissivity_table=emissivity_table)',
            'ps_tmp_ = PowerStick(axial_steps=1, width=side_len, height=L,\n'
            'cooling_temp=T_ps,\n'
            'f_bare=f_bare, f_active=f_active,\n'
            'liner_bool=liner_bool, liner_width=liner_side_len,\n'
            'emissivity_table=emissivity_table)',
            1,
        )
        # 3. powerstick — add liner_width
        src = src.replace(
            '    liner_bool=liner_bool, emissivity_table=emissivity_table,\n)',
            '    liner_bool=liner_bool, liner_width=liner_side_len,\n'
            '    emissivity_table=emissivity_table,\n)',
            1,
        )

    # ── c7d9e412 ─────────────────────────────────────────────────────────────
    elif cid.startswith('c7d9e412'):
        # 1. Read liner_side_len from YAML (after _f_active)
        src = src.replace(
            "_f_active    = _p['f_active']\n",
            "_f_active       = _p['f_active']\n"
            "_liner_side_len = _p['liner_side_len']\n",
            1,
        )
        # 2. _ps_tmp_ — add liner_width
        src = src.replace(
            '                        liner_bool=_liner_bool, emissivity_table=_emis_table)\n',
            '                        liner_bool=_liner_bool, liner_width=_liner_side_len,\n'
            '                        emissivity_table=_emis_table)\n',
            1,
        )
        # 3. ps_ inside simulate_pss_avg_power — add liner_width
        src = src.replace(
            '                           liner_bool=_liner_bool, emissivity_table=_emis_table)\n',
            '                           liner_bool=_liner_bool, liner_width=_liner_side_len,\n'
            '                           emissivity_table=_emis_table)\n',
            1,
        )

    # ── heat_sim_01 ───────────────────────────────────────────────────────────
    elif cid.startswith('heat_sim_01'):
        # 1. Read liner_side_len from YAML (after _f_active)
        src = src.replace(
            "_f_active        = _p['f_active']\n",
            "_f_active        = _p['f_active']\n"
            "_liner_side_len  = _p['liner_side_len']\n",
            1,
        )
        # 2. _ps — add liner_width
        src = src.replace(
            '                 liner_bool=_liner, emissivity_table=_emis_table)\n',
            '                 liner_bool=_liner, liner_width=_liner_side_len,\n'
            '                 emissivity_table=_emis_table)\n',
            1,
        )

    # ── heat_bisect_01 ────────────────────────────────────────────────────────
    elif cid.startswith('heat_bisect_01'):
        # 1. Read liner_side_len from YAML (after _f_active)
        src = src.replace(
            "_f_active  = _p['f_active']\n",
            "_f_active       = _p['f_active']\n"
            "_liner_side_len = _p['liner_side_len']\n",
            1,
        )
        # 2. ps_ inside simulate_charge_final_T — add liner_width
        src = src.replace(
            '                            liner_bool=_liner, emissivity_table=_emis_table)\n',
            '                            liner_bool=_liner, liner_width=_liner_side_len,\n'
            '                            emissivity_table=_emis_table)\n',
            1,
        )

    if src != orig:
        cell['source'] = src
        cell['outputs'] = []
        cell['execution_count'] = None
        patched.append(cid[:12])

print(f'Patched cells: {patched}')

with open('tinless system model.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Saved.')
