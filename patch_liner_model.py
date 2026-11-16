import json

with open('tinless system model.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

changed = []

for cell in nb['cells']:
    cid = cell.get('id', '')
    src = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])

    # ── Cell 9494082a: spectral_net_radiation + graphite_to_powerstick ────
    if cid.startswith('9494082a'):
        # 1. Replace the entire spectral_net_radiation function
        old_snr = (
            'def spectral_net_radiation(T_hot, emissivity_table, VF, A_ps, f_bare, f_active, liner_bool=False):\n'
            '    """\n'
            '    Incident radiative power [W] on the powerstick from a hot surface.\n'
            '    Mirrors PowerStick.incident_energy: integrates spectral_radiance(T_hot) over\n'
            '    wavelength bands, then multiplies by powerstick area A_ps.\n'
            '\n'
            '    Parameters\n'
            '    ----------\n'
            '    T_hot          : float  Hot surface temperature [K]\n'
            '    emissivity_table : list  [(wavelength_range, emis_active, emis_bare, emis_inactive), ...]\n'
            '                             wavelength ranges in microns\n'
            '    VF             : float  View factor hot->powerstick  (A_ps / A_hot, e.g. D_ps/D_in)\n'
            '    A_ps           : float  Powerstick surface area [m^2]  (= pi * D_ps * L)\n'
            '    f_bare         : float  Fraction of powerstick with no PV cell\n'
            '    f_active       : float  Fraction of non-bare area that is active PV\n'
            '    liner_bool     : bool   True = tungsten liner (w_emis_spectral), False = graphite (gr_emis)\n'
            '    """\n'
            '    h   = 6.62607015e-34\n'
            '    c   = 2.99792458e8\n'
            '    k_B = 1.380649e-23\n'
            '\n'
            '    wavelength_min = emissivity_table[0][0][0]\n'
            '    wavelength_max = emissivity_table[-1][0][1]\n'
            '    band_gap       = emissivity_table[0][0][1]\n'
            '    n_bins = 500\n'
            '\n'
            '    wavelengths   = np.logspace(np.log10(wavelength_min), np.log10(wavelength_max), n_bins)\n'
            '    wavelengths_m = wavelengths * 1e-6\n'
            '    spectral_radiance = np.pi * (2*h*c**2) / wavelengths_m**5 / np.expm1(h*c / (wavelengths_m * k_B * T_hot))\n'
            '\n'
            '    incident_bins = {tuple(wl_range): 0.0 for wl_range, *_ in emissivity_table}\n'
            '\n'
            '    for i, wavelength in enumerate(wavelengths):\n'
            '        if liner_bool:\n'
            '            emitter_emis = w_emis_spectral(T_hot, wavelength)\n'
            '        else:\n'
            '            emitter_emis = gr_emis(T_hot)\n'
            '        if wavelength < band_gap:\n'
            '            band_width = wavelengths_m[i+1] - wavelengths_m[i]\n'
            '        else:\n'
            '            band_width = wavelengths_m[i] - wavelengths_m[i-1]\n'
            '\n'
            '        for (wavelength_range, emis_active, emis_bare, emis_inactive) in emissivity_table:\n'
            '            if wavelength_range[0] <= wavelength < wavelength_range[1]:\n'
            '                avg_ps_emis = (f_bare * emis_bare\n'
            '                               + (1 - f_bare) * f_active       * emis_active\n'
            '                               + (1 - f_bare) * (1 - f_active) * emis_inactive)\n'
            '                incident_bins[tuple(wavelength_range)] += (\n'
            '                    spectral_radiance[i] * band_width\n'
            '                    / (VF * (1 - emitter_emis) / emitter_emis + 1 + (1 - avg_ps_emis) / avg_ps_emis)\n'
            '                )\n'
            '                break\n'
            '\n'
            '    # Absorbed power: mirrors generate_electricity_step — each component fraction sums to avg_ps_emis/avg_ps_emis = 1\n'
            '    Q_total = sum(incident_bins.values()) * A_ps\n'
            '    return Q_total'
        )
        new_snr = (
            'def spectral_net_radiation(T_hot, emissivity_table, VF, A_ps, f_bare, f_active,\n'
            '                            liner_bool=False, VF_gr_liner=None, VF_liner_ps=None, A_liner_seg=None):\n'
            '    """\n'
            '    Incident radiative power [W] on the powerstick from a hot surface.\n'
            '\n'
            '    liner_bool=False: graphite inner wall -> powerstick directly.\n'
            '                      Returns (Q_total, None).\n'
            '    liner_bool=True:  graphite -> tungsten liner (T_liner found from energy\n'
            '                      balance) -> powerstick.  Liner at r_liner = (r_in+r_ps)/2;\n'
            '                      emissivity from w_emis_spectral.\n'
            '                      Returns (Q_ps, T_liner).\n'
            '\n'
            '    Parameters\n'
            '    ----------\n'
            '    T_hot          : float  Graphite inner-wall temperature [K]\n'
            '    emissivity_table : list  [(wl_range, emis_active, emis_bare, emis_inactive), ...]\n'
            '    VF             : float  r_ps_eff / r_in  (used when liner_bool=False)\n'
            '    A_ps           : float  Powerstick area per axial segment [m^2]\n'
            '    f_bare         : float  Bare fraction of powerstick\n'
            '    f_active       : float  Active PV fraction of non-bare area\n'
            '    liner_bool     : bool   True = tungsten liner\n'
            '    VF_gr_liner    : float  r_liner / r_in              (liner_bool=True only)\n'
            '    VF_liner_ps    : float  r_ps_eff / r_liner          (liner_bool=True only)\n'
            '    A_liner_seg    : float  Liner area per segment [m^2] (liner_bool=True only)\n'
            '    """\n'
            '    h   = 6.62607015e-34\n'
            '    c   = 2.99792458e8\n'
            '    k_B = 1.380649e-23\n'
            '\n'
            '    wavelength_min = emissivity_table[0][0][0]\n'
            '    wavelength_max = emissivity_table[-1][0][1]\n'
            '    band_gap       = emissivity_table[0][0][1]\n'
            '    n_bins = 500\n'
            '\n'
            '    wavelengths   = np.logspace(np.log10(wavelength_min), np.log10(wavelength_max), n_bins)\n'
            '    wavelengths_m = wavelengths * 1e-6\n'
            '\n'
            '    def _B(T):\n'
            '        return np.pi * (2*h*c**2) / wavelengths_m**5 / np.expm1(h*c / (wavelengths_m * k_B * T))\n'
            '\n'
            '    # Band widths: forward diff below bandgap, backward diff above\n'
            '    bw_fwd = np.concatenate([wavelengths_m[1:] - wavelengths_m[:-1],\n'
            '                             [wavelengths_m[-1] - wavelengths_m[-2]]])\n'
            '    bw_bwd = np.concatenate([[wavelengths_m[1] - wavelengths_m[0]],\n'
            '                             wavelengths_m[1:] - wavelengths_m[:-1]])\n'
            '    bw = np.where(wavelengths < band_gap, bw_fwd, bw_bwd)\n'
            '\n'
            '    # Per-wavelength area-weighted powerstick emissivity\n'
            '    avg_ps_emis = np.zeros(n_bins)\n'
            '    for (wl_range, emis_active, emis_bare, emis_inactive) in emissivity_table:\n'
            '        mask = (wavelengths >= wl_range[0]) & (wavelengths < wl_range[1])\n'
            '        ep = (f_bare * emis_bare\n'
            '              + (1 - f_bare) * f_active       * emis_active\n'
            '              + (1 - f_bare) * (1 - f_active) * emis_inactive)\n'
            '        avg_ps_emis[mask] = ep\n'
            '\n'
            '    if not liner_bool:\n'
            '        eps_gr = gr_emis(T_hot)\n'
            '        B_hot  = _B(T_hot)\n'
            '        denom  = VF * (1 - eps_gr) / eps_gr + 1 + (1 - avg_ps_emis) / avg_ps_emis\n'
            '        return float(np.sum(B_hot * bw / denom)) * A_ps, None\n'
            '\n'
            '    # ── Liner model: graphite -> tungsten liner -> powerstick ──────────\n'
            '    eps_gr = gr_emis(T_hot)\n'
            '    B_hot  = _B(T_hot)\n'
            '\n'
            '    def _eps_liner(T_liner):\n'
            '        return np.array([w_emis_spectral(T_liner, wl) for wl in wavelengths])\n'
            '\n'
            '    def _balance(T_liner):\n'
            '        eps_L    = _eps_liner(T_liner)\n'
            '        denom_in = VF_gr_liner * (1 - eps_gr) / eps_gr + 1 + (1 - eps_L) / eps_L\n'
            '        Q_in     = float(np.sum(B_hot * bw / denom_in)) * A_liner_seg\n'
            '        B_lin    = _B(T_liner)\n'
            '        denom_out = VF_liner_ps * (1 - eps_L) / eps_L + 1 + (1 - avg_ps_emis) / avg_ps_emis\n'
            '        Q_out     = float(np.sum(B_lin * bw / denom_out)) * A_ps\n'
            '        return Q_in - Q_out\n'
            '\n'
            '    from scipy.optimize import brentq\n'
            '    T_liner = brentq(_balance, 300.0, T_hot - 1.0, xtol=1.0, maxiter=50)\n'
            '\n'
            '    eps_L     = _eps_liner(T_liner)\n'
            '    B_lin     = _B(T_liner)\n'
            '    denom_out = VF_liner_ps * (1 - eps_L) / eps_L + 1 + (1 - avg_ps_emis) / avg_ps_emis\n'
            '    Q_ps      = float(np.sum(B_lin * bw / denom_out)) * A_ps\n'
            '    return Q_ps, T_liner'
        )

        # 2. Update graphite_to_powerstick_heat_transfer to unpack the tuple
        old_gtp_1 = (
            '        Q_rad = spectral_net_radiation(T_inner, emissivity_table, VF_gr_ps,\n'
            '                                       A_ps, f_bare, f_active, liner_bool)\n'
            '        return T_inner - (T_outer - Q_rad * R_cond)'
        )
        new_gtp_1 = (
            '        Q_rad, _ = spectral_net_radiation(T_inner, emissivity_table, VF_gr_ps,\n'
            '                                          A_ps, f_bare, f_active, liner_bool)\n'
            '        return T_inner - (T_outer - Q_rad * R_cond)'
        )
        old_gtp_2 = (
            '    Q = spectral_net_radiation(T_inner, emissivity_table, VF_gr_ps,\n'
            '                               A_ps, f_bare, f_active, liner_bool)\n'
            '    return Q, T_inner'
        )
        new_gtp_2 = (
            '    Q, _ = spectral_net_radiation(T_inner, emissivity_table, VF_gr_ps,\n'
            '                                  A_ps, f_bare, f_active, liner_bool)\n'
            '    return Q, T_inner'
        )

        ok = True
        if old_snr not in src:
            print(f'WARNING: spectral_net_radiation pattern not found in {cid}')
            ok = False
        if old_gtp_1 not in src:
            print(f'WARNING: graphite_to_powerstick pattern 1 not found in {cid}')
            ok = False
        if old_gtp_2 not in src:
            print(f'WARNING: graphite_to_powerstick pattern 2 not found in {cid}')
            ok = False

        if ok:
            src = src.replace(old_snr, new_snr, 1)
            src = src.replace(old_gtp_1, new_gtp_1, 1)
            src = src.replace(old_gtp_2, new_gtp_2, 1)
            cell['source'] = src
            cell['outputs'] = []
            cell['execution_count'] = None
            changed.append(cid)

    # ── Cell 03dd9a23: TinlessSimulation.__init__ and _step_calc ──────────
    elif cid.startswith('03dd9a23'):
        # 1. __init__: add liner geometry
        old_init = (
            '        r_ps_eff      = 2 * powerstick.width / np.pi\n'
            '        self.VF       = r_ps_eff / storage.block.rad_in\n'
            '        self.A_ps_seg = powerstick.A_step'
        )
        new_init = (
            '        r_ps_eff      = 2 * powerstick.width / np.pi\n'
            '        r_in          = storage.block.rad_in\n'
            '        self.VF       = r_ps_eff / r_in\n'
            '        self.A_ps_seg = powerstick.A_step\n'
            '\n'
            '        if powerstick.liner_bool:\n'
            '            r_liner          = (r_in + r_ps_eff) / 2\n'
            '            self.VF_gr_liner = r_liner / r_in\n'
            '            self.VF_liner_ps = r_ps_eff / r_liner\n'
            '            self.A_liner_seg = 2 * np.pi * r_liner * powerstick.height_step\n'
            '        else:\n'
            '            self.VF_gr_liner = None\n'
            '            self.VF_liner_ps = None\n'
            '            self.A_liner_seg = None'
        )

        # 2. _step_calc: pass liner geometry, use T_liner for generate_electricity_step
        old_step = (
            '            T_wall_i = T_dist[i, 0]\n'
            '            Q_rad[i] = spectral_net_radiation(\n'
            '                T_wall_i, ps.emissivity_table, self.VF,\n'
            '                self.A_ps_seg, ps.f_bare, ps.f_active, ps.liner_bool\n'
            '            )\n'
            '            _, _, Q_cell[i] = ps.generate_electricity_step(\n'
            '                T_wall_i, self.VF, ps.panel_efficiency\n'
            '            )'
        )
        new_step = (
            '            T_wall_i = T_dist[i, 0]\n'
            '            Q_rad[i], T_liner_i = spectral_net_radiation(\n'
            '                T_wall_i, ps.emissivity_table, self.VF,\n'
            '                self.A_ps_seg, ps.f_bare, ps.f_active, ps.liner_bool,\n'
            '                VF_gr_liner=self.VF_gr_liner, VF_liner_ps=self.VF_liner_ps,\n'
            '                A_liner_seg=self.A_liner_seg\n'
            '            )\n'
            '            T_emit_i = T_liner_i if ps.liner_bool else T_wall_i\n'
            '            VF_emit  = self.VF_liner_ps if ps.liner_bool else self.VF\n'
            '            _, _, Q_cell[i] = ps.generate_electricity_step(\n'
            '                T_emit_i, VF_emit, ps.panel_efficiency\n'
            '            )'
        )

        ok = True
        if old_init not in src:
            print(f'WARNING: __init__ pattern not found in {cid}')
            ok = False
        if old_step not in src:
            print(f'WARNING: _step_calc pattern not found in {cid}')
            ok = False

        if ok:
            src = src.replace(old_init, new_init, 1)
            src = src.replace(old_step, new_step, 1)
            cell['source'] = src
            cell['outputs'] = []
            cell['execution_count'] = None
            changed.append(cid)

print(f'Changed: {changed}')

with open('tinless system model.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Saved.')
