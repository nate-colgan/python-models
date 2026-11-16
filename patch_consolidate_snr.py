"""
Consolidate spectral_net_radiation + graphite_to_powerstick_heat_transfer into
the TinlessSimulation cell (03dd9a23) and expose them as class members.
"""
import json

SCRATCHPAD = r'C:\Users\NATECO~1\AppData\Local\Temp\claude\c--Users-NateColgan-Documents-thermal-modelling\c47ec152-e9e6-46d7-8142-ad36070a1585\scratchpad'

with open('tinless system model.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

# ── Pull full source of the two functions from cell 9494082a ──────────────────
snr_src = None
for cell in nb['cells']:
    if cell.get('id', '').startswith('9494082a'):
        snr_src = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])
        break
assert snr_src is not None, 'cell 9494082a not found'

# ── Patch cell 9494082a → stub comment ───────────────────────────────────────
for cell in nb['cells']:
    if cell.get('id', '').startswith('9494082a'):
        cell['source'] = (
            '# spectral_net_radiation and graphite_to_powerstick_heat_transfer\n'
            '# are defined at the top of the TinlessSimulation cell (03dd9a23).\n'
            '# Re-run that cell if these names are not in scope.\n'
        )
        cell['outputs'] = []
        cell['execution_count'] = None
        print('Stubbed cell 9494082a')
        break

# ── Patch cell 03dd9a23 ───────────────────────────────────────────────────────
for cell in nb['cells']:
    if not cell.get('id', '').startswith('03dd9a23'):
        continue
    src = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])
    orig = src

    # 1. Prepend both functions (from 9494082a) before HeatingElement class
    old_he = 'class HeatingElement:\n'
    new_he = snr_src + '\n\n' + old_he
    if old_he not in src:
        print('WARNING: HeatingElement class not found')
    else:
        src = src.replace(old_he, new_he, 1)
        print('Prepended SNR functions before HeatingElement')

    # 2. Add graphite_to_powerstick_heat_transfer as a staticmethod class attribute
    #    immediately before __init__ (after the class docstring)
    old_init = (
        '    def __init__(self, storage, powerstick, T_sur=300, block_heat_loss=False,\n'
        '                 shield_bool=False, shield_side_len=None,\n'
        '                 f_shield_min=0.0, f_shield_max=0.9, nominal_cell_flux=None):\n'
    )
    new_init = (
        '    graphite_to_powerstick_heat_transfer = staticmethod(\n'
        '        graphite_to_powerstick_heat_transfer\n'
        '    )\n'
        '\n'
        + old_init
    )
    if old_init not in src:
        print('WARNING: __init__ signature not found')
    else:
        src = src.replace(old_init, new_init, 1)
        print('Added graphite_to_powerstick_heat_transfer staticmethod')

    # 3. Add _spectral_net_radiation instance method between __init__ and _step_calc
    old_step = '    def _step_calc(self, T_dist, f_shield=0.0):\n'
    new_step = (
        '    def _spectral_net_radiation(self, T_hot, f_shield=0.0):\n'
        '        """Spectral radiation to the powerstick using this simulation\'s stored geometry."""\n'
        '        return spectral_net_radiation(\n'
        '            T_hot, self.powerstick.emissivity_table, self.VF, self.A_ps_seg,\n'
        '            self.powerstick.f_bare, self.powerstick.f_active, self.powerstick.liner_bool,\n'
        '            VF_gr_liner=self.VF_gr_liner, VF_liner_ps=self.VF_liner_ps,\n'
        '            A_liner_seg=self.A_liner_seg,\n'
        '            shield_bool=self.shield_bool, f_shield=f_shield,\n'
        '            VF_gr_shield=self.VF_gr_shield, VF_shield_lin=self.VF_shield_lin,\n'
        '            A_shield_seg=self.A_shield_seg,\n'
        '        )\n'
        '\n'
        + old_step
    )
    if old_step not in src:
        print('WARNING: _step_calc signature not found')
    else:
        src = src.replace(old_step, new_step, 1)
        print('Added _spectral_net_radiation method')

    # 4. Collapse the 9-line SNR call in _step_calc to the 1-line instance method call
    old_snr_call = (
        '            Q_rad[i], T_liner_i = spectral_net_radiation(\n'
        '                T_wall_i, ps.emissivity_table, self.VF,\n'
        '                self.A_ps_seg, ps.f_bare, ps.f_active, ps.liner_bool,\n'
        '                VF_gr_liner=self.VF_gr_liner, VF_liner_ps=self.VF_liner_ps,\n'
        '                A_liner_seg=self.A_liner_seg,\n'
        '                shield_bool=self.shield_bool, f_shield=f_shield,\n'
        '                VF_gr_shield=self.VF_gr_shield, VF_shield_lin=self.VF_shield_lin,\n'
        '                A_shield_seg=self.A_shield_seg,\n'
        '            )\n'
    )
    new_snr_call = (
        '            Q_rad[i], T_liner_i = self._spectral_net_radiation(T_wall_i, f_shield=f_shield)\n'
    )
    if old_snr_call not in src:
        print('WARNING: _step_calc SNR call not found')
    else:
        src = src.replace(old_snr_call, new_snr_call, 1)
        print('Collapsed _step_calc SNR call')

    # 5. Collapse the 6-line SNR call in _liner_avg (run_heating_simulation) to 1 line
    old_liner_snr = (
        '                _, tli = spectral_net_radiation(\n'
        '                    float(T_d[i, 0]), ps_heat.emissivity_table, self.VF,\n'
        '                    self.A_ps_seg, ps_heat.f_bare, ps_heat.f_active, ps_heat.liner_bool,\n'
        '                    VF_gr_liner=self.VF_gr_liner, VF_liner_ps=self.VF_liner_ps,\n'
        '                    A_liner_seg=self.A_liner_seg\n'
        '                )\n'
    )
    new_liner_snr = (
        '                _, tli = self._spectral_net_radiation(float(T_d[i, 0]))\n'
    )
    if old_liner_snr not in src:
        print('WARNING: _liner_avg SNR call not found')
    else:
        src = src.replace(old_liner_snr, new_liner_snr, 1)
        print('Collapsed _liner_avg SNR call')

    if src != orig:
        cell['source'] = src
        cell['outputs'] = []
        cell['execution_count'] = None
        print('Patched 03dd9a23')
    else:
        print('WARNING: no changes made to 03dd9a23')
    break

with open('tinless system model.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Saved.')
