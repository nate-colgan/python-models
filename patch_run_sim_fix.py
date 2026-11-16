import json

with open('tinless system model.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if not cell.get('id', '').startswith('03dd9a23'):
        continue
    src = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])

    fixes = [
        # 1. Initial _step_calc unpack: add Q_elec0
        (
            '        Q0, Q_cell0, _, T_liner0 = self._step_calc(T)\n'
            '        Q_arr[0]       = Q0.sum()\n'
            '        Q_cell_arr[0]  = Q_cell0.sum()\n'
            '        Q_elec_arr[0]  = Q_elec0.sum()',

            '        Q0, Q_elec0, Q_cell0, _, T_liner0 = self._step_calc(T)\n'
            '        Q_arr[0]       = Q0.sum()\n'
            '        Q_cell_arr[0]  = Q_cell0.sum()\n'
            '        Q_elec_arr[0]  = Q_elec0.sum()',
        ),
        # 2. Loop _step_calc unpack: add Q_elec, correct order
        (
            '            Q_rad, Q_cell, Q_waste, T_liner = self._step_calc(T)',
            '            Q_rad, Q_elec, Q_cell, Q_waste, T_liner = self._step_calc(T)',
        ),
        # 3. Loop assignment: P_elec_arr -> Q_elec_arr, Q_cell -> Q_elec
        (
            '            P_elec_arr[idx]  = Q_cell.sum()',
            '            Q_elec_arr[idx]  = Q_elec.sum()',
        ),
        # 4. Print statement: P_elec_arr -> Q_elec_arr
        (
            '                      f"P = {P_elec_arr[idx]/1e3:.1f} kW")',
            '                      f"P_elec = {Q_elec_arr[idx]/1e3:.1f} kW")',
        ),
        # 5. Return dict: P_elec_arr -> Q_elec_arr
        (
            "            'P_elec':         P_elec_arr,",
            "            'P_elec':         Q_elec_arr,",
        ),
    ]

    ok = True
    for label, (old, new) in enumerate(fixes):
        if old not in src:
            print(f'WARNING: fix #{label+1} pattern not found')
            ok = False

    if ok:
        for old, new in fixes:
            src = src.replace(old, new, 1)
        cell['source'] = src
        cell['outputs'] = []
        cell['execution_count'] = None
        print('Patched 03dd9a23')
    break

with open('tinless system model.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Saved.')
