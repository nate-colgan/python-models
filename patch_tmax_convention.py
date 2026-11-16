import json

with open('tinless system model.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if not cell.get('id', '').startswith('ff640d4b'):
        continue
    src = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])

    fixes = [
        # 1. Fixed parameter: _T_min -> _T_max
        (
            '_T_min           = 1873        # K  (block empty/minimum temperature)',
            '_T_max           = 2373        # K  (block full/maximum temperature)',
        ),
        # 2. _build_and_run: signature + docstring + first line
        (
            'def _build_and_run(T_min, dT_range, area_ratio, emis_table, liner_bool_):\n'
            '    """\n'
            '    Size a graphite storage block and run a full transient discharge simulation.\n'
            '\n'
            '    T_max = T_min + dT_range.\n'
            '    Block volume is sized from the required energy; inner radius follows from\n'
            '    area_ratio = r_in / r_ps_eff; outer radius derived from V_block and r_in.\n'
            '    """\n'
            '    T_max     = T_min + dT_range\n'
            '    T_avg_mid = T_min + dT_range / 2',

            'def _build_and_run(T_max, dT_range, area_ratio, emis_table, liner_bool_):\n'
            '    """\n'
            '    Size a graphite storage block and run a full transient discharge simulation.\n'
            '\n'
            '    T_min = T_max - dT_range.\n'
            '    Block volume is sized from the required energy; inner radius follows from\n'
            '    area_ratio = r_in / r_ps_eff; outer radius derived from V_block and r_in.\n'
            '    """\n'
            '    T_min     = T_max - dT_range\n'
            '    T_avg_mid = T_min + dT_range / 2',
        ),
        # 3. _pss_avg_q: signature + first line inside
        (
            'def _pss_avg_q(res_, dT_range, T_min):\n'
            '    """\n'
            '    Compute average cell heat flux over the pseudo-steady-state window.\n'
            '    PSS onset fixed at t=0; window ends when SOC drops to zero.\n'
            '    """\n'
            '    soc      = (res_[\'T_avg\'] - T_min) / dT_range',

            'def _pss_avg_q(res_, dT_range, T_max):\n'
            '    """\n'
            '    Compute average cell heat flux over the pseudo-steady-state window.\n'
            '    PSS onset fixed at t=0; window ends when SOC drops to zero.\n'
            '    """\n'
            '    T_min    = T_max - dT_range\n'
            '    soc      = (res_[\'T_avg\'] - T_min) / dT_range',
        ),
        # 4. Print statement in sweep loop
        (
            "print(f'[{n}/{total}] liner={liner_bool_}  dT={dT_range}K  T_max={_T_min + dT_range}K  bg={bg_val}um  -- bisecting area_ratio')",
            "print(f'[{n}/{total}] liner={liner_bool_}  dT={dT_range}K  T_min={_T_max - dT_range}K  bg={bg_val}um  -- bisecting area_ratio')",
        ),
        # 5. _f closure: default arg + two calls inside
        (
            'def _f(ar, _Tmin=_T_min, _dT=dT_range, _emis=emis_table_bg, _lb=liner_bool_):\n'
            '                    res_, _ = _build_and_run(_Tmin, _dT, ar, _emis, _lb)\n'
            '                    q = _pss_avg_q(res_, _dT, _Tmin)',

            'def _f(ar, _Tmax=_T_max, _dT=dT_range, _emis=emis_table_bg, _lb=liner_bool_):\n'
            '                    res_, _ = _build_and_run(_Tmax, _dT, ar, _emis, _lb)\n'
            '                    q = _pss_avg_q(res_, _dT, _Tmax)',
        ),
        # 6. Final _build_and_run call after bisection
        (
            'res_opt, r_out_opt = _build_and_run(_T_min, dT_range, ar_opt, emis_table_bg, liner_bool_)',
            'res_opt, r_out_opt = _build_and_run(_T_max, dT_range, ar_opt, emis_table_bg, liner_bool_)',
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
