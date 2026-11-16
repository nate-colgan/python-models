import json

with open('tinless system model.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if not cell.get('id', '').startswith('9494082a'):
        continue
    src = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])
    orig = src

    # ── 1. spectral_net_radiation: extend signature ───────────────────────────
    src = src.replace(
        'def spectral_net_radiation(T_hot, emissivity_table, VF, A_ps, f_bare, f_active,\n'
        '                            liner_bool=False, VF_gr_liner=None, VF_liner_ps=None, A_liner_seg=None):\n',
        'def spectral_net_radiation(T_hot, emissivity_table, VF, A_ps, f_bare, f_active,\n'
        '                            liner_bool=False, VF_gr_liner=None, VF_liner_ps=None, A_liner_seg=None,\n'
        '                            shield_bool=False, f_shield=0.0,\n'
        '                            VF_gr_shield=None, VF_shield_lin=None, A_shield_seg=None):\n',
        1,
    )

    # ── 2. spectral_net_radiation: update docstring liner notes + add shield params ──
    src = src.replace(
        '    liner_bool     : bool   True = tungsten liner\n'
        '    VF_gr_liner    : float  r_liner / r_in              (liner_bool=True only)\n'
        '    VF_liner_ps    : float  r_ps_eff / r_liner          (liner_bool=True only)\n'
        '    A_liner_seg    : float  Liner area per segment [m^2] (liner_bool=True only)\n'
        '    """\n',
        '    liner_bool     : bool   True = tungsten liner\n'
        '    VF_gr_liner    : float  r_liner / r_in              (liner_bool=True only)\n'
        '    VF_liner_ps    : float  r_ps_eff / r_liner          (liner_bool=True only)\n'
        '    A_liner_seg    : float  Liner area per segment [m^2] (liner_bool=True only)\n'
        '    shield_bool    : bool   True = double-W radiation shield (requires liner_bool=True)\n'
        '    f_shield       : float  Fraction of liner area covered by shield [0, 1]\n'
        '    VF_gr_shield   : float  r_shield / r_in  (view factor ratio, graphite -> shield)\n'
        '    VF_shield_lin  : float  r_liner / r_shield  (view factor ratio, shield -> liner)\n'
        '    A_shield_seg   : float  Shield area per segment [m^2]\n'
        '    """\n',
        1,
    )

    # ── 3. Replace the liner model block (from sigma/eps_gr setup through return) ─
    old_liner = (
        '    # ── Liner model: graphite -> tungsten liner -> powerstick ──────────\n'
        '    eps_gr = gr_emis(T_hot)\n'
        '    B_hot  = _B(T_hot)\n'
        '\n'
        '    def _eps_liner(T_liner):\n'
        '        #return np.array([w_emis_spectral(T_liner, wl) for wl in wavelengths])\n'
        '        return 0.55, 0.8\n'
        '\n'
        '    def _balance(T_liner):\n'
        '        eps_L_in, eps_L_out     = _eps_liner(T_liner)\n'
        '        B_lin     = _B(T_liner)\n'
        '        denom_in  = VF_gr_liner * (1 - eps_gr) / eps_gr + 1 + (1 - eps_L_out) / eps_L_out\n'
        '        Q_in      = float(np.sum((B_hot - B_lin) * bw / denom_in)) * A_liner_seg\n'
        '        denom_out = VF_liner_ps * (1 - eps_L_in) / eps_L_in + 1 + (1 - avg_ps_emis) / avg_ps_emis\n'
        '        Q_out     = float(np.sum(B_lin * bw / denom_out)) * A_ps\n'
        '        return Q_in - Q_out\n'
        '\n'
        '    from scipy.optimize import brentq\n'
        '    T_liner = brentq(_balance, 400.0, T_hot - 1.0, xtol=1.0, maxiter=50)\n'
        '\n'
        '    eps_L_in, _ = _eps_liner(T_liner)\n'
        '    B_lin     = _B(T_liner)\n'
        '    denom_out = VF_liner_ps * (1 - eps_L_in) / eps_L_in + 1 + (1 - avg_ps_emis) / avg_ps_emis\n'
        '    Q_ps      = float(np.sum(B_lin * bw / denom_out)) * A_ps\n'
        '    return Q_ps, T_liner\n'
    )
    new_liner = (
        '    # ── Liner model: graphite -> tungsten liner -> powerstick ──────────\n'
        '    sigma  = 5.67036713e-8\n'
        '    eps_gr = gr_emis(T_hot)\n'
        '    B_hot  = _B(T_hot)\n'
        '\n'
        '    # Shield: precompute gray-body series resistance (gr -> W_out -> W_in -> liner)\n'
        '    _shield_active = shield_bool and f_shield > 0 and A_shield_seg is not None\n'
        '    _f_direct      = 1.0 - f_shield if _shield_active else 1.0\n'
        '    if _shield_active:\n'
        '        eps_w       = w_emis_grey(T_hot)\n'
        '        A_shield_sh = f_shield * A_shield_seg\n'
        '        A_liner_sh  = f_shield * A_liner_seg\n'
        '        R1       = (VF_gr_shield  * (1 - eps_gr) / eps_gr + 1 + (1 - eps_w) / eps_w) / A_shield_sh\n'
        '        R2       = (2 / eps_w - 1) / A_shield_sh\n'
        '        R3       = (VF_shield_lin * (1 - eps_w)  / eps_w  + 1 + (1 - eps_w) / eps_w) / A_liner_sh\n'
        '        R_shield = R1 + R2 + R3\n'
        '\n'
        '    def _eps_liner(T_liner):\n'
        '        #return np.array([w_emis_spectral(T_liner, wl) for wl in wavelengths])\n'
        '        return 0.55, 0.8\n'
        '\n'
        '    def _balance(T_liner):\n'
        '        eps_L_in, eps_L_out = _eps_liner(T_liner)\n'
        '        B_lin     = _B(T_liner)\n'
        '        # Unshielded direct path: (1 - f_shield) of liner area\n'
        '        denom_in  = VF_gr_liner * (1 - eps_gr) / eps_gr + 1 + (1 - eps_L_out) / eps_L_out\n'
        '        Q_in      = _f_direct * float(np.sum((B_hot - B_lin) * bw / denom_in)) * A_liner_seg\n'
        '        # Shielded path: gray-body series resistance\n'
        '        if _shield_active:\n'
        '            Q_in += sigma * (T_hot**4 - T_liner**4) / R_shield\n'
        '        denom_out = VF_liner_ps * (1 - eps_L_in) / eps_L_in + 1 + (1 - avg_ps_emis) / avg_ps_emis\n'
        '        Q_out     = float(np.sum(B_lin * bw / denom_out)) * A_ps\n'
        '        return Q_in - Q_out\n'
        '\n'
        '    from scipy.optimize import brentq\n'
        '    T_liner = brentq(_balance, 400.0, T_hot - 1.0, xtol=1.0, maxiter=50)\n'
        '\n'
        '    eps_L_in, _ = _eps_liner(T_liner)\n'
        '    B_lin     = _B(T_liner)\n'
        '    denom_out = VF_liner_ps * (1 - eps_L_in) / eps_L_in + 1 + (1 - avg_ps_emis) / avg_ps_emis\n'
        '    Q_ps      = float(np.sum(B_lin * bw / denom_out)) * A_ps\n'
        '    return Q_ps, T_liner\n'
    )

    if old_liner not in src:
        print('WARNING: liner block pattern not found')
    else:
        src = src.replace(old_liner, new_liner, 1)

    # ── 4. graphite_to_powerstick_heat_transfer: extend signature ─────────────
    src = src.replace(
        'def graphite_to_powerstick_heat_transfer(T_outer, T_ps, k_gr,\n'
        '                                          emissivity_table, f_bare, f_active,\n'
        '                                          D_in, D_out, D_ps, L,\n'
        '                                          liner_bool=False, tol=1e-2, max_iter=50):\n',
        'def graphite_to_powerstick_heat_transfer(T_outer, T_ps, k_gr,\n'
        '                                          emissivity_table, f_bare, f_active,\n'
        '                                          D_in, D_out, D_ps, L,\n'
        '                                          liner_bool=False,\n'
        '                                          shield_bool=False, f_shield=0.0,\n'
        '                                          VF_gr_shield=None, VF_shield_lin=None, A_shield_seg=None,\n'
        '                                          tol=1e-2, max_iter=50):\n',
        1,
    )

    # ── 5. graphite_to_powerstick: pass shield kwargs through to spectral_net_radiation ─
    src = src.replace(
        '    def residual(T_inner):\n'
        '        Q_rad, _ = spectral_net_radiation(T_inner, emissivity_table, VF_gr_ps,\n'
        '                                          A_ps, f_bare, f_active, liner_bool)\n'
        '        return T_inner - (T_outer - Q_rad * R_cond)\n'
        '\n'
        '    T_inner = brentq(residual, T_ps + 1e-3, T_outer - 1e-3, xtol=tol, maxiter=max_iter)\n'
        '    Q, _ = spectral_net_radiation(T_inner, emissivity_table, VF_gr_ps,\n'
        '                                  A_ps, f_bare, f_active, liner_bool)\n',
        '    _shield_kw = dict(shield_bool=shield_bool, f_shield=f_shield,\n'
        '                      VF_gr_shield=VF_gr_shield, VF_shield_lin=VF_shield_lin,\n'
        '                      A_shield_seg=A_shield_seg)\n'
        '\n'
        '    def residual(T_inner):\n'
        '        Q_rad, _ = spectral_net_radiation(T_inner, emissivity_table, VF_gr_ps,\n'
        '                                          A_ps, f_bare, f_active, liner_bool,\n'
        '                                          **_shield_kw)\n'
        '        return T_inner - (T_outer - Q_rad * R_cond)\n'
        '\n'
        '    T_inner = brentq(residual, T_ps + 1e-3, T_outer - 1e-3, xtol=tol, maxiter=max_iter)\n'
        '    Q, _ = spectral_net_radiation(T_inner, emissivity_table, VF_gr_ps,\n'
        '                                  A_ps, f_bare, f_active, liner_bool, **_shield_kw)\n',
        1,
    )

    if src != orig:
        cell['source'] = src
        cell['outputs'] = []
        cell['execution_count'] = None
        print(f'Patched 9494082a')
    else:
        print('WARNING: no changes made')
    break

with open('tinless system model.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Saved.')
