import json, re

with open('tinless system model.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

changed = []

for cell in nb['cells']:
    cid = cell.get('id', '')
    src = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])
    orig = src

    # ── Cell 03dd9a23: _step_calc + run_simulation ────────────────────────
    if cid.startswith('03dd9a23'):
        # 1. _step_calc: swap return slot (use electricity, not cell_flux)
        src = src.replace(
            '        _, _, Q_cell[i] = ps.generate_electricity_step(\n'
            '            T_emit_i, VF_emit, ps.panel_efficiency\n'
            '        )',
            '        Q_cell[i], _, _ = ps.generate_electricity_step(\n'
            '            T_emit_i, VF_emit\n'
            '        )',
        )
        # 2. _step_calc: Q_waste no longer needs panel_efficiency factor
        src = src.replace(
            '        Q_waste[i] = Q_rad[i] - Q_cell[i] * ps.panel_efficiency',
            '        Q_waste[i] = Q_rad[i] - Q_cell[i]',
        )
        # 3. run_simulation: P_elec_arr — initial step
        src = src.replace(
            '        P_elec_arr[0]  = Q_cell0.sum() * self.powerstick.panel_efficiency',
            '        P_elec_arr[0]  = Q_cell0.sum()',
        )
        # 4. run_simulation: P_elec_arr — loop step
        src = src.replace(
            '            P_elec_arr[idx]  = Q_cell.sum() * self.powerstick.panel_efficiency',
            '            P_elec_arr[idx]  = Q_cell.sum()',
        )

    # ── All cells: strip panel_efficiency= kwarg from PowerStick(...) ─────
    # Handles both single-line and multi-line instantiations
    src = re.sub(r',?\s*panel_efficiency\s*=\s*[^\s,\n)]+', '', src)

    if src != orig:
        cell['source'] = src
        cell['outputs'] = []
        cell['execution_count'] = None
        changed.append(cid)

print(f'Changed cells: {changed}')
with open('tinless system model.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Saved.')
