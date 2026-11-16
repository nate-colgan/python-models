import json

with open('tinless system model.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if not cell.get('id', '').startswith('ff640d4b'):
        continue
    src = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])

    fixes = [
        # 1. Add _liner_side_len to fixed parameters (after _T_max)
        (
            '_T_max           = 2373        # K  (block full/maximum temperature)\n',
            '_T_max           = 2373        # K  (block full/maximum temperature)\n'
            '_liner_side_len  = 0.20        # m  (square side length of tungsten liner)\n',
        ),
        # 2. ps_tmp_ inside _build_and_run — add liner_width
        (
            '    ps_tmp_      = PowerStick(axial_steps=1, width=_side_len, height=_L,\n'
            '                             cooling_temp=_T_ps,\n'
            '                             f_bare=_f_bare, f_active=_f_active,\n'
            '                             liner_bool=liner_bool_, emissivity_table=emis_table)\n',
            '    ps_tmp_      = PowerStick(axial_steps=1, width=_side_len, height=_L,\n'
            '                             cooling_temp=_T_ps,\n'
            '                             f_bare=_f_bare, f_active=_f_active,\n'
            '                             liner_bool=liner_bool_, liner_width=_liner_side_len,\n'
            '                             emissivity_table=emis_table)\n',
        ),
        # 3. ps_ inside _build_and_run — add liner_width
        (
            '    ps_     = PowerStick(axial_steps=_num_ax, width=_side_len, height=_L,\n'
            '                         cooling_temp=_T_ps,\n'
            '                         f_bare=_f_bare, f_active=_f_active,\n'
            '                         liner_bool=liner_bool_, emissivity_table=emis_table)\n',
            '    ps_     = PowerStick(axial_steps=_num_ax, width=_side_len, height=_L,\n'
            '                         cooling_temp=_T_ps,\n'
            '                         f_bare=_f_bare, f_active=_f_active,\n'
            '                         liner_bool=liner_bool_, liner_width=_liner_side_len,\n'
            '                         emissivity_table=emis_table)\n',
        ),
    ]

    ok = True
    for i, (old, new) in enumerate(fixes):
        if old not in src:
            print(f'WARNING: fix #{i+1} pattern not found')
            ok = False

    if ok:
        for old, new in fixes:
            src = src.replace(old, new, 1)
        cell['source'] = src
        cell['outputs'] = []
        cell['execution_count'] = None
        print('Patched ff640d4b')
    break

with open('tinless system model.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Saved.')
