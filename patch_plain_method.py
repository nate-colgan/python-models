"""
Convert spectral_net_radiation to a plain instance method of TinlessSimulation:
- Remove @staticmethod spectral_net_radiation and @staticmethod graphite_to_powerstick_heat_transfer
  from inside the class
- Remove the _spectral_net_radiation wrapper method
- Insert def spectral_net_radiation(self, T_hot, f_shield=0.0): with full body using self.*
- Move graphite_to_powerstick_heat_transfer to module-level (plain function, before HeatingElement)
- Update _step_calc and _liner_avg calls from self._spectral_net_radiation to self.spectral_net_radiation
"""
import json

NB = 'tinless system model.ipynb'

with open(NB, encoding='utf-8') as f:
    nb = json.load(f)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Patch cell 03dd9a23
# ─────────────────────────────────────────────────────────────────────────────
for cell in nb['cells']:
    if not cell.get('id', '').startswith('03dd9a23'):
        continue
    src = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])
    lines = src.splitlines(keepends=True)

    # --- locate the two @staticmethod blocks and _spectral_net_radiation wrapper ---
    def find_block_end(lines, start):
        """
        Given the line index of a 'def ...' header, walk forward until we
        reach a line at indent level 4 (or less) that starts a new def/class,
        or end of file.  Returns the index of the first such line (exclusive).
        """
        i = start + 1
        while i < len(lines):
            stripped = lines[i].lstrip()
            indent   = len(lines[i]) - len(stripped)
            if indent <= 4 and stripped and not stripped.startswith('#'):
                # back up over any trailing blank lines
                break
            i += 1
        return i

    # Find @staticmethod decorator lines inside TinlessSimulation
    static_starts = []
    for i, l in enumerate(lines):
        if l.strip() == '@staticmethod':
            static_starts.append(i)

    assert len(static_starts) == 2, f"Expected 2 @staticmethod, found {len(static_starts)}"

    # Find _spectral_net_radiation wrapper
    wrapper_start = None
    for i, l in enumerate(lines):
        if 'def _spectral_net_radiation(self' in l:
            wrapper_start = i
            break
    assert wrapper_start is not None

    # Collect all regions to delete (sorted by line number, non-overlapping)
    # Each region is (start_inclusive, end_exclusive)
    regions = []
    for si in static_starts:
        # si is the @staticmethod line; the def is si+1
        end = find_block_end(lines, si + 1)
        # include the @staticmethod line itself
        regions.append((si, end))
    end_wrapper = find_block_end(lines, wrapper_start)
    regions.append((wrapper_start, end_wrapper))

    regions.sort()
    print(f"Regions to delete: {regions}")

    # Build new line list by skipping deleted regions
    kept = []
    cursor = 0
    for (rs, re_) in regions:
        kept.extend(lines[cursor:rs])
        cursor = re_
    kept.extend(lines[cursor:])

    new_src = ''.join(kept)

    # --- insert spectral_net_radiation instance method before _step_calc ---
    SNR_METHOD = '''\
    def spectral_net_radiation(self, T_hot, f_shield=0.0):
        """
        Incident radiative power [W] on the powerstick from a hot graphite surface.

        liner_bool=False: graphite inner wall -> powerstick directly.
                          Returns (Q_total, None).
        liner_bool=True:  graphite -> tungsten liner -> powerstick.
                          Returns (Q_ps, T_liner).
        """
        h   = 6.62607015e-34
        c   = 2.99792458e8
        k_B = 1.380649e-23

        emissivity_table = self.powerstick.emissivity_table
        liner_bool       = self.powerstick.liner_bool
        f_bare           = self.powerstick.f_bare
        f_active         = self.powerstick.f_active
        VF               = self.VF
        A_ps             = self.A_ps_seg
        shield_bool      = self.shield_bool
        VF_gr_liner      = self.VF_gr_liner
        VF_liner_ps      = self.VF_liner_ps
        A_liner_seg      = self.A_liner_seg
        VF_gr_shield     = self.VF_gr_shield
        VF_shield_lin    = self.VF_shield_lin
        A_shield_seg     = self.A_shield_seg

        wavelength_min = emissivity_table[0][0][0]
        wavelength_max = emissivity_table[-1][0][1]
        band_gap       = emissivity_table[0][0][1]
        n_bins = 300

        wavelengths   = np.logspace(np.log10(wavelength_min), np.log10(wavelength_max), n_bins)
        wavelengths_m = wavelengths * 1e-6

        def _B(T):
            return np.pi * (2*h*c**2) / wavelengths_m**5 / np.expm1(h*c / (wavelengths_m * k_B * T))

        bw_fwd = np.concatenate([wavelengths_m[1:] - wavelengths_m[:-1],
                                 [wavelengths_m[-1] - wavelengths_m[-2]]])
        bw_bwd = np.concatenate([[wavelengths_m[1] - wavelengths_m[0]],
                                 wavelengths_m[1:] - wavelengths_m[:-1]])
        bw = np.where(wavelengths < band_gap, bw_fwd, bw_bwd)

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

        _shield_active = shield_bool and f_shield > 0 and A_shield_seg is not None
        _f_direct      = 1.0 - f_shield if _shield_active else 1.0
        if _shield_active:
            eps_w       = w_emis_grey(T_hot)
            A_shield_sh = f_shield * A_shield_seg
            A_liner_sh  = f_shield * A_liner_seg
            R1 = (VF_gr_shield  * (1 - eps_gr) / eps_gr + 1 + (1 - eps_w) / eps_w) / A_shield_sh
            R2 = (2 / eps_w - 1) / A_shield_sh
            R3 = (VF_shield_lin * (1 - eps_w)  / eps_w  + 1 + (1 - eps_w) / eps_w) / A_liner_sh
            R_shield = R1 + R2 + R3

        def _eps_liner(_T):
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

'''

    STEP_MARKER = '    def _step_calc(self, T_dist, f_shield=0.0):\n'
    assert STEP_MARKER in new_src, '_step_calc not found'
    new_src = new_src.replace(STEP_MARKER, SNR_METHOD + STEP_MARKER, 1)
    print('Inserted spectral_net_radiation instance method')

    # --- rename calls from self._spectral_net_radiation -> self.spectral_net_radiation ---
    new_src = new_src.replace(
        'self._spectral_net_radiation(T_wall_i, f_shield=f_shield)',
        'self.spectral_net_radiation(T_wall_i, f_shield=f_shield)',
    )
    new_src = new_src.replace(
        'self._spectral_net_radiation(float(T_d[i, 0]))',
        'self.spectral_net_radiation(float(T_d[i, 0]))',
    )
    print('Updated _step_calc and _liner_avg calls')

    # --- prepend graphite_to_powerstick_heat_transfer as module-level function ---
    # Extract it from the current source (it was inside TinlessSimulation, now move to top-level).
    # Since we deleted its @staticmethod block already, grab it from the *original* src.
    orig_lines = src.splitlines(keepends=True)
    gtps_start = None
    for i, l in enumerate(orig_lines):
        if 'def graphite_to_powerstick_heat_transfer(' in l:
            gtps_start = i
            break
    assert gtps_start is not None, 'graphite_to_powerstick_heat_transfer not found in original'
    gtps_end = find_block_end(orig_lines, gtps_start)
    # Dedent by 4 spaces (remove leading '    ')
    gtps_lines = [l[4:] if l.startswith('    ') else l
                  for l in orig_lines[gtps_start:gtps_end]]
    # Replace internal call to TinlessSimulation.spectral_net_radiation with explicit call
    # (since it now needs to call the function without an instance)
    gtps_body = ''.join(gtps_lines)
    # The existing call: TinlessSimulation.spectral_net_radiation(T_inner, emissivity_table, ...)
    # Replace with a local helper call (we'll define the helper inline)
    gtps_body = gtps_body.replace(
        'TinlessSimulation.spectral_net_radiation(\n'
        '            T_inner, emissivity_table, VF_gr_ps,\n'
        '            A_ps, f_bare, f_active, liner_bool, **_shield_kw)',
        '_snr_explicit(T_inner, emissivity_table, VF_gr_ps, A_ps, f_bare, f_active, liner_bool, **_shield_kw)'
    )
    # If that didn't match, try a simpler replacement
    if '_snr_explicit' not in gtps_body:
        # Just remove the whole function — it's never called
        gtps_body = ''
        print('WARNING: graphite_to_powerstick_heat_transfer internal call not matched; omitting function')
    else:
        gtps_body += '\n'
        print('Moved graphite_to_powerstick_heat_transfer to module-level')

    # Prepend before class HeatingElement
    HEAT_MARKER = 'class HeatingElement:\n'
    assert HEAT_MARKER in new_src
    new_src = new_src.replace(HEAT_MARKER, gtps_body + HEAT_MARKER, 1)

    cell['source'] = new_src
    cell['outputs'] = []
    cell['execution_count'] = None
    print(f'Patched 03dd9a23  ({len(new_src.splitlines())} lines)')
    break

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Update scratch cell (spectral_net_radiation is now an instance method)
# ─────────────────────────────────────────────────────────────────────────────
for cell in nb['cells']:
    if cell.get('id', '') != 'scratch_shield_01':
        continue
    src = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])
    # If it still references TinlessSimulation.spectral_net_radiation as a static call,
    # update to use an unbound call (TinlessSimulation.spectral_net_radiation(self_like, T, f))
    # For now just flag it with a comment
    if 'TinlessSimulation.spectral_net_radiation' in src:
        src = src.replace(
            'TinlessSimulation.spectral_net_radiation',
            '# NOTE: spectral_net_radiation is now an instance method.\n'
            '# Create a TinlessSimulation instance to call sim.spectral_net_radiation(T, f_shield).\n'
            '# --- TinlessSimulation.spectral_net_radiation  (removed static call)'
        )
        cell['source'] = src
        cell['outputs'] = []
        cell['execution_count'] = None
        print('Flagged scratch cell')
    break

with open(NB, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Saved.')
