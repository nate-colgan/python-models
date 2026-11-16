import json

with open('tinless system model.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if not cell.get('id', '').startswith('a8c3d501'):
        continue
    src = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])

    shield_section = (
        '\n'
        '# -- Radiation Shield --------------------------------------------------------\n'
        '# Two negligibly thin W sheets (VF=1 between them) positioned between the\n'
        '# graphite inner wall and the liner.  Covers f_shield fraction of the liner\n'
        '# area; f_shield is found by bisection to hold cell flux at the nominal value.\n'
        '# shield_bool=False disables the shield; f_shield is clamped to [min, max].\n'
        'shield:\n'
        '  shield_bool: False\n'
        '  shield_side_len: 0.18    # square side length of W shield [m]; r_shield = 2*side_len/pi\n'
        '  f_shield_min: 0.0        # minimum shielded area fraction (lower bisection bound)\n'
        '  f_shield_max: 0.9        # maximum shielded area fraction (upper bisection bound)\n'
    )

    old = '\n# -- Heater'
    new = shield_section + '\n# -- Heater'

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
