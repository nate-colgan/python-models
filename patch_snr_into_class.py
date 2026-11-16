"""
Move spectral_net_radiation and graphite_to_powerstick_heat_transfer fully inside
the TinlessSimulation class as @staticmethod methods.  Remove module-level definitions.
"""
import json, re, textwrap

with open('tinless system model.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if not cell.get('id', '').startswith('03dd9a23'):
        continue
    src = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])
    orig = src

    # ── 1. Strip the two module-level function defs from the top of the cell ──
    # They occupy everything before "class HeatingElement:"
    boundary = 'class HeatingElement:\n'
    idx = src.find(boundary)
    assert idx >= 0, 'HeatingElement class not found'
    src = src[idx:]          # keep from HeatingElement onwards
    print('Removed module-level function defs')

    # ── 2. Build @staticmethod text for spectral_net_radiation ────────────────
    snr_static = textwrap.indent('''\
@staticmethod
def spectral_net_radiation(T_hot, emissivity_table, VF, A_ps, f_bare, f_active,
                            liner_bool=False, VF_gr_liner=None, VF_liner_ps=None, A_liner_seg=None,
                            shield_bool=False, f_shield=0.0,
                            VF_gr_shield=None, VF_shield_lin=None, A_shield_seg=None):
    """
    Incident radiative power [W] on the powerstick from a hot surface.

    liner_bool=False: graphite inner wall -> powerstick directly.
                      Returns (Q_total, None).
    liner_bool=True:  graphite -> tungsten liner -> powerstick.
                      Returns (Q_ps, T_liner).
    """
    h   = 6.62607015e-34
    c   = 2.99792458e8
    k_B = 1.380649e-23

    wavelength_min = emissivity_table[0][0][0]
    wavelength_max = emissivity_table[-1][0][1]
    band_gap       = emissivity_table[0][0][1]
    n_bins = 300

    wavelengths   = np.logspace(np.log10(wavelength_min), np.log10(wavelength_max), n_bins)
    wavelengths_m = wavelengths * 1e-6

    def _B(T):
        return np.pi * (2*h*c**2) / wavelengths_m**5 / np.expm1(h*c / (wavelengths_m * k_B * T))

    # Band widths: forward diff below bandgap, backward diff above
    bw_fwd = np.concatenate([wavelengths_m[1:] - wavelengths_m[:-1],
                             [wavelengths_m[-1] - wavelengths_m[-2]]])
    bw_bwd = np.concatenate([[wavelengths_m[1] - wavelengths_m[0]],
                             wavelengths_m[1:] - wavelengths_m[:-1]])
    bw = np.where(wavelengths < band_gap, bw_fwd, bw_bwd)

    # Per-wavelength area-weighted powerstick emissivity
    avg_ps_emis = np.zeros(n_bins)
    for (wl_range, emis_active, emis_bare, emis_inactive) in emissivity_table:
        mask = (wavelengths >= wl_range[0]) & (wavelengths < wl_range[1])
        ep = (f_bare * emis_bare
              + (1 - f_bare) * f_active       * emis_active
              + (1 - f_bare) * (1 - f_active) * emis_inactive)
        avg_ps_emis[mask] = ep

    if not liner_bool:
        eps_gr = gr_emis(T_hot)
        B_hot  = _B(T_hot)
        denom  = VF * (1 - eps_gr) / eps_gr + 1 + (1 - avg_ps_emis) / avg_ps_emis
        return float(np.sum(B_hot * bw / denom)) * A_ps, None

    # ── Liner model: graphite -> tungsten liner -> powerstick ──────────
    sigma  = 5.67036713e-8
    eps_gr = gr_emis(T_hot)
    B_hot  = _B(T_hot)

    # Shield: precompute gray-body series resistance (gr -> W_out -> W_in -> liner)
    _shield_active = shield_bool and f_shield > 0 and A_shield_seg is not None
    _f_direct      = 1.0 - f_shield if _shield_active else 1.0
    if _shield_active:
        eps_w       = w_emis_grey(T_hot)
        A_shield_sh = f_shield * A_shield_seg
        A_liner_sh  = f_shield * A_liner_seg
        R1       = (VF_gr_shield  * (1 - eps_gr) / eps_gr + 1 + (1 - eps_w) / eps_w) / A_shield_sh
        R2       = (2 / eps_w - 1) / A_shield_sh
        R3       = (VF_shield_lin * (1 - eps_w)  / eps_w  + 1 + (1 - eps_w) / eps_w) / A_liner_sh
        R_shield = R1 + R2 + R3

    def _eps_liner(T_liner):
        return 0.55, 0.8

    def _balance(T_liner):
        eps_L_in, eps_L_out = _eps_liner(T_liner)
        B_lin     = _B(T_liner)
        denom_in  = VF_gr_liner * (1 - eps_gr) / eps_gr + 1 + (1 - eps_L_out) / eps_L_out
        Q_in      = _f_direct * float(np.sum((B_hot - B_lin) * bw / denom_in)) * A_liner_seg
        if _shield_active:
            Q_in += sigma * (T_hot**4 - T_liner**4) / R_shield
        denom_out = VF_liner_ps * (1 - eps_L_in) / eps_L_in + 1 + (1 - avg_ps_emis) / avg_ps_emis
        Q_out     = float(np.sum(B_lin * bw / denom_out)) * A_ps
        return Q_in - Q_out

    from scipy.optimize import brentq
    T_liner = brentq(_balance, 400.0, T_hot - 1.0, xtol=1.0, maxiter=50)

    eps_L_in, _ = _eps_liner(T_liner)
    B_lin     = _B(T_liner)
    denom_out = VF_liner_ps * (1 - eps_L_in) / eps_L_in + 1 + (1 - avg_ps_emis) / avg_ps_emis
    Q_ps      = float(np.sum(B_lin * bw / denom_out)) * A_ps
    return Q_ps, T_liner

''', '    ')

    # ── 3. Build @staticmethod text for graphite_to_powerstick_heat_transfer ──
    gtps_static = textwrap.indent('''\
@staticmethod
def graphite_to_powerstick_heat_transfer(T_outer, T_ps, k_gr,
                                          emissivity_table, f_bare, f_active,
                                          D_in, D_out, D_ps, L,
                                          liner_bool=False,
                                          shield_bool=False, f_shield=0.0,
                                          VF_gr_shield=None, VF_shield_lin=None, A_shield_seg=None,
                                          tol=1e-2, max_iter=50):
    """
    Heat transfer from an annular graphite region to a central powerstick via
    series conduction (outer->inner wall) and spectral radiation (inner->PS).
    """
    from scipy.optimize import brentq

    A_ps     = np.pi * D_ps * L
    VF_gr_ps = D_ps / D_in
    R_cond   = RprimeCondCyl(D_in, D_out, k_gr) / L

    _shield_kw = dict(shield_bool=shield_bool, f_shield=f_shield,
                      VF_gr_shield=VF_gr_shield, VF_shield_lin=VF_shield_lin,
                      A_shield_seg=A_shield_seg)

    def residual(T_inner):
        Q_rad, _ = TinlessSimulation.spectral_net_radiation(
            T_inner, emissivity_table, VF_gr_ps,
            A_ps, f_bare, f_active, liner_bool, **_shield_kw)
        return T_inner - (T_outer - Q_rad * R_cond)

    T_inner = brentq(residual, T_ps + 1e-3, T_outer - 1e-3, xtol=tol, maxiter=max_iter)
    Q, _ = TinlessSimulation.spectral_net_radiation(
        T_inner, emissivity_table, VF_gr_ps,
        A_ps, f_bare, f_active, liner_bool, **_shield_kw)
    return Q, T_inner

''', '    ')

    # ── 4. Replace the old staticmethod() assignment with the full @staticmethod defs ─
    old_static_assign = (
        '    graphite_to_powerstick_heat_transfer = staticmethod(\n'
        '        graphite_to_powerstick_heat_transfer\n'
        '    )\n'
        '\n'
        '    def __init__(self, storage, powerstick, T_sur=300, block_heat_loss=False,\n'
    )
    new_static_assign = (
        snr_static
        + gtps_static
        + '    def __init__(self, storage, powerstick, T_sur=300, block_heat_loss=False,\n'
    )
    if old_static_assign not in src:
        print('WARNING: old staticmethod assignment not found')
    else:
        src = src.replace(old_static_assign, new_static_assign, 1)
        print('Replaced staticmethod assignment with full @staticmethod defs')

    # ── 5. Update _spectral_net_radiation wrapper to call the class staticmethod ─
    old_wrapper_call = '        return spectral_net_radiation(\n'
    new_wrapper_call = '        return TinlessSimulation.spectral_net_radiation(\n'
    if old_wrapper_call not in src:
        print('WARNING: _spectral_net_radiation wrapper call not found')
    else:
        src = src.replace(old_wrapper_call, new_wrapper_call, 1)
        print('Updated _spectral_net_radiation to call TinlessSimulation.spectral_net_radiation')

    if src != orig:
        cell['source'] = src
        cell['outputs'] = []
        cell['execution_count'] = None
        print('Patched 03dd9a23')
    else:
        print('WARNING: no changes made')
    break

with open('tinless system model.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Saved.')
