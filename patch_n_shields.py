import json

NB = 'tinless system model.ipynb'
with open(NB, encoding='utf-8') as f:
    nb = json.load(f)

# ── 1. YAML cell a8c3d501 ─────────────────────────────────────────────────────
for cell in nb['cells']:
    if not cell.get('id','').startswith('a8c3d501'):
        continue
    src = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])

    # Remove nominal_cell_flux line (handle encoding variation in comment)
    for old in [
        '  nominal_cell_flux: 20    # target incident cell flux [W/cm²]; shield adjusts f_shield to maintain this\n',
        '  nominal_cell_flux: 20    # target incident cell flux [W/cm^2]; shield adjusts f_shield to maintain this\n',
        '  nominal_cell_flux: 20   # target incident cell flux [W/cm²]; shield adjusts f_shield to maintain this\n',
        '  nominal_cell_flux: 20   # target incident cell flux [W/cm^2]; shield adjusts f_shield to maintain this\n',
    ]:
        if old in src:
            src = src.replace(old, '')
            print(f'Removed nominal_cell_flux line from YAML')
            break
    else:
        # Find and remove any line containing nominal_cell_flux in the YAML block
        lines = src.splitlines(keepends=True)
        new_lines = [l for l in lines if 'nominal_cell_flux' not in l]
        if len(new_lines) < len(lines):
            src = ''.join(new_lines)
            print(f'Removed nominal_cell_flux line(s) from YAML (fallback)')

    # Add N_shields after shield_side_len line
    old_ssl = '  shield_side_len: 0.32    # square side length of W shield [m]; r_shield = 2*side_len/pi\n'
    new_ssl = old_ssl + '  N_shields: 2             # number of W shield layers\n'
    if old_ssl in src:
        src = src.replace(old_ssl, new_ssl, 1)
        print('Added N_shields to YAML')

    # Update shield section comment
    old_comment = (
        '# Two negligibly thin W sheets (VF=1 between them) positioned between the\n'
        '# graphite inner wall and the liner.  Covers f_shield fraction of the liner\n'
        '# area; f_shield is found by bisection to hold cell flux at the nominal value.\n'
        '# shield_bool=False disables the shield; f_shield is clamped to [min, max].'
    )
    new_comment = (
        '# N_shields negligibly thin W sheets stacked between the graphite inner wall\n'
        '# and the liner (VF=1 between consecutive sheets).  Covers f_shield fraction\n'
        '# of the liner area; f_shield is found by bisection to hold cell flux at\n'
        '# storage.avg_flux.  shield_bool=False disables the shield.'
    )
    if old_comment in src:
        src = src.replace(old_comment, new_comment, 1)
        print('Updated YAML shield comment')

    cell['source'] = src
    print(f'  nominal_cell_flux in YAML: {src.count("nominal_cell_flux")}')
    print(f'  N_shields in YAML: {src.count("N_shields")}')
    break

# ── 2. TinlessSimulation cell 03dd9a23 ────────────────────────────────────────
for cell in nb['cells']:
    if not cell.get('id','').startswith('03dd9a23'):
        continue
    src = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])

    # a) __init__ signature: add N_shields param
    old_sig = (
        '    def __init__(self, storage, powerstick, T_sur=300, block_heat_loss=False,\n'
        '                 shield_bool=False, shield_side_len=None,\n'
        '                 f_shield_min=0.0, f_shield_max=0.9, nominal_cell_flux=None):'
    )
    new_sig = (
        '    def __init__(self, storage, powerstick, T_sur=300, block_heat_loss=False,\n'
        '                 shield_bool=False, shield_side_len=None, N_shields=2,\n'
        '                 f_shield_min=0.0, f_shield_max=0.9, nominal_cell_flux=None):'
    )
    if old_sig in src:
        src = src.replace(old_sig, new_sig, 1)
        print('Updated __init__ signature')

    # b) Store N_shields on self, inserted after shield_bool assignment
    old_assign = (
        '        self.shield_bool       = shield_bool\n'
        '        self.f_shield_min      = f_shield_min\n'
    )
    new_assign = (
        '        self.shield_bool       = shield_bool\n'
        '        self.N_shields         = N_shields\n'
        '        self.f_shield_min      = f_shield_min\n'
    )
    if old_assign in src:
        src = src.replace(old_assign, new_assign, 1)
        print('Added self.N_shields assignment')

    # c) spectral_net_radiation: generalize R_shield for N shields
    old_R = '            R_shield = R1 + R2 + R3'
    new_R = '            R_shield = R1 + (self.N_shields - 1) * R2 + R3'
    if old_R in src:
        src = src.replace(old_R, new_R, 1)
        print('Generalized R_shield for N_shields')

    cell['source'] = src
    cell['outputs'] = []
    cell['execution_count'] = None

    lines = src.splitlines()
    for kw in ['N_shields', 'R_shield = R1']:
        hits = [f"L{i+1}" for i,l in enumerate(lines) if kw in l]
        print(f'  {kw}: {hits}')
    break

# ── 3. Run cell b2e4f810 ──────────────────────────────────────────────────────
for cell in nb['cells']:
    if not cell.get('id','').startswith('b2e4f810'):
        continue
    src = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])

    src = src.replace(
        "nominal_cell_flux = sh.get('nominal_cell_flux', None)",
        "N_shields         = sh.get('N_shields', 2)"
    )
    old_ctor = (
        'sim = TinlessSimulation(storage, powerstick, T_sur=T_sur, block_heat_loss=block_heat_loss_bool,\n'
        '                        shield_bool=shield_bool, shield_side_len=shield_side_len,\n'
        '                        f_shield_min=f_shield_min, f_shield_max=f_shield_max,\n'
        '                        nominal_cell_flux=nominal_cell_flux)'
    )
    new_ctor = (
        'sim = TinlessSimulation(storage, powerstick, T_sur=T_sur, block_heat_loss=block_heat_loss_bool,\n'
        '                        shield_bool=shield_bool, shield_side_len=shield_side_len,\n'
        '                        N_shields=N_shields,\n'
        '                        f_shield_min=f_shield_min, f_shield_max=f_shield_max,\n'
        '                        nominal_cell_flux=avg_flux)'
    )
    if old_ctor in src:
        src = src.replace(old_ctor, new_ctor, 1)
        print('Updated b2e4f810 TinlessSimulation constructor')
    else:
        print('WARNING: b2e4f810 constructor not matched')

    cell['source'] = src
    cell['outputs'] = []
    cell['execution_count'] = None
    print(f'  b2e4f810 nominal_cell_flux remaining: {src.count("nominal_cell_flux")}')
    break

# ── 4. Run cell c7d9e412 ──────────────────────────────────────────────────────
for cell in nb['cells']:
    if not cell.get('id','').startswith('c7d9e412'):
        continue
    src = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])

    src = src.replace(
        "_nominal_cell_flux = _sh.get('nominal_cell_flux', None)",
        "_N_shields         = _sh.get('N_shields', 2)"
    )
    old_ctor2 = (
        '                                  f_shield_min=_f_shield_min,\n'
        '                                  f_shield_max=_f_shield_max,\n'
        '                                  nominal_cell_flux=_nominal_cell_flux)'
    )
    new_ctor2 = (
        '                                  N_shields=_N_shields,\n'
        '                                  f_shield_min=_f_shield_min,\n'
        '                                  f_shield_max=_f_shield_max,\n'
        '                                  nominal_cell_flux=_avg_flux)'
    )
    if old_ctor2 in src:
        src = src.replace(old_ctor2, new_ctor2, 1)
        print('Updated c7d9e412 TinlessSimulation constructor')
    else:
        print('WARNING: c7d9e412 constructor not matched')

    cell['source'] = src
    cell['outputs'] = []
    cell['execution_count'] = None
    print(f'  c7d9e412 nominal_cell_flux remaining: {src.count("nominal_cell_flux")}')
    break

with open(NB, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Saved.')
