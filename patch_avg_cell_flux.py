import json

with open('tinless system model.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

patched = []

for cell in nb['cells']:
    cid = cell.get('id', '')
    if not cid.startswith('ff640d4b'):
        continue
    src = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])
    orig = src

    old = (
        '                    efficiency_cell = eff_cell,\n'
        '                )'
    )
    new = (
        '                    efficiency_cell = eff_cell,\n'
        '                    avg_cell_flux  = float(Q_cell.mean()) / (4 * _side_len * _L * (1.0 - _f_bare) * _f_active * 1e4),\n'
        '                )'
    )

    if old not in src:
        # Try alternative indentation (in case stored differently)
        print('Primary pattern not found, trying content-only match...')
        import re
        m = re.search(r'( +)efficiency_cell = eff_cell,\n(\s*)\)', src)
        if m:
            indent = m.group(1)
            close_indent = m.group(2)
            old = f'{indent}efficiency_cell = eff_cell,\n{close_indent})'
            new = (f'{indent}efficiency_cell = eff_cell,\n'
                   f'{indent}avg_cell_flux  = float(Q_cell.mean()) / (4 * _side_len * _L * (1.0 - _f_bare) * _f_active * 1e4),\n'
                   f'{close_indent})')
            print(f'  found with indent={repr(indent)}, close_indent={repr(close_indent)}')
        else:
            print('ERROR: Could not find pattern')
            break

    src = src.replace(old, new, 1)

    if src != orig:
        cell['source'] = src
        cell['outputs'] = []
        cell['execution_count'] = None
        patched.append(cid[:12])
    else:
        print('No change made')
    break

print(f'Patched cells: {patched}')

with open('tinless system model.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Saved.')
