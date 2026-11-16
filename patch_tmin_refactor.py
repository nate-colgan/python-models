import json

with open('tinless system model.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if not cell.get('id', '').startswith('ff640d4b'):
        continue
    src = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])

    # 1. _build_and_run: signature + docstring + first two body lines
    old1 = (
        'def _build_and_run(T_max, dT_range, area_ratio, emis_table, liner_bool_):\n'
        '    """\n'
        '    Size a graphite storage block and run a full transient discharge simulation.\n'
        '\n'
        '    T_min = T_max - dT_range = _T_min (since T_max = _T_min + dT_range).\n'
        '    Block volume is sized from the required energy; inner radius follows from\n'
        '    area_ratio = r_in / r_ps_eff; outer radius derived from V_block and r_in.\n'
        '    """\n'
        '    T_min_    = T_max - dT_range\n'
        '    T_avg_mid = T_min_ + dT_range / 2\n'
    )
    new1 = (
        'def _build_and_run(T_min, dT_range, area_ratio, emis_table, liner_bool_):\n'
        '    """\n'
        '    Size a graphite storage block and run a full transient discharge simulation.\n'
        '\n'
        '    T_max = T_min + dT_range.\n'
        '    Block volume is sized from the required energy; inner radius follows from\n'
        '    area_ratio = r_in / r_ps_eff; outer radius derived from V_block and r_in.\n'
        '    """\n'
        '    T_max     = T_min + dT_range\n'
        '    T_avg_mid = T_min + dT_range / 2\n'
    )

    # 2. _pss_avg_q: signature + body (remove T_min_ = T_max - dT_range)
    old2 = (
        'def _pss_avg_q(res_, dT_range, T_max):\n'
        '    """\n'
        '    Compute average cell heat flux over the pseudo-steady-state window.\n'
        '    PSS onset fixed at t=0; window ends when SOC drops to zero.\n'
        '    """\n'
        '    T_min_   = T_max - dT_range\n'
        '    soc      = (res_[\'T_avg\'] - T_min_) / dT_range\n'
    )
    new2 = (
        'def _pss_avg_q(res_, dT_range, T_min):\n'
        '    """\n'
        '    Compute average cell heat flux over the pseudo-steady-state window.\n'
        '    PSS onset fixed at t=0; window ends when SOC drops to zero.\n'
        '    """\n'
        '    soc      = (res_[\'T_avg\'] - T_min) / dT_range\n'
    )

    # 3. Main loop: remove T_max = ... line, update print and _f closure
    old3 = (
        '        for dT_range in temp_range_vals:\n'
        '            T_max = _T_min + dT_range\n'
        '            n += 1\n'
        '            print(f\'[{n}/{total}] liner={liner_bool_}  dT={dT_range}K  T_max={T_max}K  bg={bg_val}um  -- bisecting area_ratio\')\n'
        '\n'
        '            def _f(ar, _Tmax=T_max, _dT=dT_range, _emis=emis_table_bg, _lb=liner_bool_):\n'
        '                res_, _ = _build_and_run(_Tmax, _dT, ar, _emis, _lb)\n'
        '                q = _pss_avg_q(res_, _dT, _Tmax)\n'
    )
    new3 = (
        '        for dT_range in temp_range_vals:\n'
        '            n += 1\n'
        '            print(f\'[{n}/{total}] liner={liner_bool_}  dT={dT_range}K  T_max={_T_min + dT_range}K  bg={bg_val}um  -- bisecting area_ratio\')\n'
        '\n'
        '            def _f(ar, _Tmin=_T_min, _dT=dT_range, _emis=emis_table_bg, _lb=liner_bool_):\n'
        '                res_, _ = _build_and_run(_Tmin, _dT, ar, _emis, _lb)\n'
        '                q = _pss_avg_q(res_, _dT, _Tmin)\n'
    )

    # 4. Final _build_and_run call
    old4 = '            res_opt, r_out_opt = _build_and_run(T_max, dT_range, ar_opt, emis_table_bg, liner_bool_)\n'
    new4 = '            res_opt, r_out_opt = _build_and_run(_T_min, dT_range, ar_opt, emis_table_bg, liner_bool_)\n'

    ok = True
    for label, old in [('build sig+body', old1), ('pss_avg_q sig+body', old2),
                        ('main loop', old3), ('final call', old4)]:
        if old not in src:
            print(f'WARNING: pattern "{label}" not found')
            ok = False

    if ok:
        src = src.replace(old1, new1, 1)
        src = src.replace(old2, new2, 1)
        src = src.replace(old3, new3, 1)
        src = src.replace(old4, new4, 1)
        cell['source'] = src
        cell['outputs'] = []
        cell['execution_count'] = None
        print('Patched ff640d4b')
    break

with open('tinless system model.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Saved.')
