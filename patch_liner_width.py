import json

with open('tinless system model.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if not cell.get('id', '').startswith('03dd9a23'):
        continue
    src = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])

    old = '        if powerstick.liner_bool:\n            r_liner          = (r_in + r_ps_eff) / 2\n'
    new = '        if powerstick.liner_bool:\n            r_liner          = 2 * powerstick.liner_width / np.pi\n'

    if old not in src:
        print('WARNING: pattern not found')
    else:
        cell['source'] = src.replace(old, new, 1)
        cell['outputs'] = []
        cell['execution_count'] = None
        print('Patched 03dd9a23')
    break

with open('tinless system model.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Saved.')
