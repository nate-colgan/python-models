import json

with open('tinless system model.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

changes = []

for cell in nb['cells']:
    cid = cell.get('id', '')
    src = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])
    orig = src

    # ── YAML template cell a8c3d501: add nominal_cell_flux to shield section ──
    if cid.startswith('a8c3d501'):
        old = ('  f_shield_max: 0.95        # maximum shielded area fraction (upper bisection bound)\n'
               '\n'
               '# -- Heater')
        new = ('  f_shield_max: 0.95        # maximum shielded area fraction (upper bisection bound)\n'
               '  nominal_cell_flux: 2.5    # target incident cell flux [W/cm²]; shield adjusts f_shield to maintain this\n'
               '\n'
               '# -- Heater')
        if old not in src:
            print(f'WARNING a8c3d501: nominal_cell_flux pattern not found')
        else:
            src = src.replace(old, new, 1)

    # ── Main run cell b2e4f810: add shield parsing + update sim call ──────────
    if cid.startswith('b2e4f810'):
        # 1. Add shield section parsing after liner_side_len
        old_liner = "liner_side_len = p['liner_side_len']\n"
        new_liner  = ("liner_side_len = p['liner_side_len']\n"
                      "\n"
                      "sh                = cfg.get('shield', {})\n"
                      "shield_bool       = sh.get('shield_bool', cfg.get('shield_bool', False))\n"
                      "shield_side_len   = sh.get('shield_side_len', None)\n"
                      "f_shield_min      = sh.get('f_shield_min', 0.0)\n"
                      "f_shield_max      = sh.get('f_shield_max', 0.9)\n"
                      "nominal_cell_flux = sh.get('nominal_cell_flux', None)\n")
        if old_liner not in src:
            print('WARNING b2e4f810: liner_side_len pattern not found')
        else:
            src = src.replace(old_liner, new_liner, 1)

        # 2. Update TinlessSimulation instantiation
        old_sim = ('sim = TinlessSimulation(storage, powerstick, T_sur=T_sur, block_heat_loss=block_heat_loss_bool)\n')
        new_sim  = ('sim = TinlessSimulation(storage, powerstick, T_sur=T_sur, block_heat_loss=block_heat_loss_bool,\n'
                    '                        shield_bool=shield_bool, shield_side_len=shield_side_len,\n'
                    '                        f_shield_min=f_shield_min, f_shield_max=f_shield_max,\n'
                    '                        nominal_cell_flux=nominal_cell_flux)\n')
        if old_sim not in src:
            print('WARNING b2e4f810: TinlessSimulation call pattern not found')
        else:
            src = src.replace(old_sim, new_sim, 1)

    # ── PSS bisection cell c7d9e412: add shield parsing + update sim call ──────
    if cid.startswith('c7d9e412'):
        # 1. Add shield section parsing after _liner_side_len is read
        old_liner_ = "_liner_side_len = _p['liner_side_len']\n"
        new_liner_  = ("_liner_side_len    = _p['liner_side_len']\n"
                       "\n"
                       "_sh                = _cfg.get('shield', {})\n"
                       "_shield_bool       = _sh.get('shield_bool', _cfg.get('shield_bool', False))\n"
                       "_shield_side_len   = _sh.get('shield_side_len', None)\n"
                       "_f_shield_min      = _sh.get('f_shield_min', 0.0)\n"
                       "_f_shield_max      = _sh.get('f_shield_max', 0.9)\n"
                       "_nominal_cell_flux = _sh.get('nominal_cell_flux', None)\n")
        if old_liner_ not in src:
            print('WARNING c7d9e412: _liner_side_len pattern not found')
        else:
            src = src.replace(old_liner_, new_liner_, 1)

        # 2. Update TinlessSimulation call inside the bisection function
        old_sim_ = ('    sim_      = TinlessSimulation(storage_, ps_, T_sur=_T_sur,\n'
                    '                                  block_heat_loss=_blk_loss)\n')
        new_sim_  = ('    sim_      = TinlessSimulation(storage_, ps_, T_sur=_T_sur,\n'
                     '                                  block_heat_loss=_blk_loss,\n'
                     '                                  shield_bool=_shield_bool,\n'
                     '                                  shield_side_len=_shield_side_len,\n'
                     '                                  f_shield_min=_f_shield_min,\n'
                     '                                  f_shield_max=_f_shield_max,\n'
                     '                                  nominal_cell_flux=_nominal_cell_flux)\n')
        if old_sim_ not in src:
            print('WARNING c7d9e412: TinlessSimulation call pattern not found')
        else:
            src = src.replace(old_sim_, new_sim_, 1)

    # ── Scratch cell: fix r_in to use actual formula ───────────────────────────
    if cid == 'scratch_shield_01':
        old_rin = ("r_in            = 0.20    # graphite inner-wall effective radius [m]  (= storage.block.rad_in)\n")
        new_rin  = ("area_ratio      = 6       # r_in / r_ps_eff  (from YAML powerstick.area_ratio)\n"
                    "# r_in is derived in the simulation as r_ps_eff * area_ratio:\n"
                    "# (= storage.block.rad_in once the block is sized)\n")
        if old_rin not in src:
            print('WARNING scratch: r_in pattern not found')
        else:
            src = src.replace(old_rin, new_rin, 1)
        # Also update derived geometry to compute r_in from area_ratio
        old_rps = ("r_ps_eff = 2 * side_len / np.pi\n"
                   "r_liner  = 2 * liner_side_len / np.pi\n"
                   "r_shield = 2 * shield_side_len / np.pi\n")
        new_rps  = ("r_ps_eff = 2 * side_len / np.pi\n"
                    "r_in     = r_ps_eff * area_ratio\n"
                    "r_liner  = 2 * liner_side_len / np.pi\n"
                    "r_shield = 2 * shield_side_len / np.pi\n")
        if old_rps not in src:
            print('WARNING scratch: r_ps_eff block pattern not found')
        else:
            src = src.replace(old_rps, new_rps, 1)

    if src != orig:
        cell['source'] = src
        cell['outputs'] = []
        cell['execution_count'] = None
        changes.append(cid[:16])

print(f'Patched cells: {changes}')
with open('tinless system model.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Saved.')
