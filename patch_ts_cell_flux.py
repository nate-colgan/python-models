import json

with open('tinless system model.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if not cell.get('id', '').startswith('35eeb2ae'):
        continue
    src = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])

    new_fig = (
        '\n'
        '# -- Cell flux vs time ------------------------------------------------------\n'
        '_cell_area_cm2 = 4 * side_len * L * (1 - f_bare) * f_active * 1e4  # cm²\n'
        'fig3, ax3 = plt.subplots(figsize=(10, 4))\n'
        'ax3.plot(time_hr, results[\'Q_cell\'] / _cell_area_cm2, color=\'C5\')\n'
        'ax3.set_xlabel(\'Time [hr]\')\n'
        'ax3.set_ylabel(\'Cell flux [W/cm²]\')\n'
        'ax3.set_title(\'Cell power density vs Time\')\n'
        'ax3.grid(True)\n'
        'fig3.tight_layout()\n'
    )

    old = '\nplt.show()\n'
    new = new_fig + '\nplt.show()\n'

    if old not in src:
        print('WARNING: pattern not found')
    else:
        cell['source'] = src.replace(old, new, 1)
        cell['outputs'] = []
        cell['execution_count'] = None
        print('Patched 35eeb2ae')
    break

with open('tinless system model.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Saved.')
