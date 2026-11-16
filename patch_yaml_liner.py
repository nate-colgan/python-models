import json

with open('tinless system model.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if not cell.get('id', '').startswith('a8c3d501'):
        continue
    src = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])

    old = '  f_bare: 0.2\n  f_active: 0.83\n'
    new = '  f_bare: 0.2\n  f_active: 0.83\n  liner_side_len: 0.20     # square side length of tungsten liner [m]; r_liner = 2*side_len/pi\n'

    if old not in src:
        print('WARNING: pattern not found')
    else:
        cell['source'] = src.replace(old, new, 1)
        cell['outputs'] = []
        cell['execution_count'] = None
        print('Patched a8c3d501')
    break

with open('tinless system model.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Saved.')
