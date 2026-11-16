import json

with open('tinless system model.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if not cell.get('id', '').startswith('03dd9a23'):
        continue
    src = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])

    old = (
        '            _, _, Q_cell[i] = ps.generate_electricity_step(\n'
        '                T_emit_i, VF_emit, ps.panel_efficiency\n'
        '            )'
    )
    new = (
        '            Q_cell[i], _, _ = ps.generate_electricity_step(\n'
        '                T_emit_i, VF_emit\n'
        '            )'
    )

    if old in src:
        cell['source'] = src.replace(old, new, 1)
        cell['outputs'] = []
        cell['execution_count'] = None
        print('Patched 03dd9a23')
    else:
        print(f'WARNING: pattern not found')
    break

with open('tinless system model.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Saved.')
