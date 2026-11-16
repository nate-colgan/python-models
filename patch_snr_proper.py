"""
Make spectral_net_radiation a plain instance method of TinlessSimulation with the
full body using self.* — no static method, no wrapper.
graphite_to_powerstick_heat_transfer stays module-level (custom geometry, not self.*).
"""
import json, textwrap

with open('tinless system model.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if not cell.get('id', '').startswith('03dd9a23'):
        continue
    src = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])
    orig = src

    # ── 1. Remove both @staticmethod blocks from inside TinlessSimulation ──────
    # They occupy everything between the class docstring and __init__
    old_statics = (
        '    @staticmethod\n'
        '    def spectral_net_radiation(T_hot, emissivity_table, VF, A_ps, f_bare, f_active,\n'
        '                                liner_bool=False, VF_gr_liner=None, VF_liner_ps=None, A_liner_seg=None,\n'
        '                                shield_bool=False, f_shield=0.0,\n'
        '                                VF_gr_shield=None, VF_shield_lin=None, A_shield_seg=None):\n'
    )
    # Find start and end of the two staticmethod blocks
    start = src.find(old_statics)
    assert start >= 0, 'spectral_net_radiation @staticmethod not found'
    # End is just before '    def __init__'
    end_marker = '    def __init__(self, storage, powerstick, T_sur=300, block_heat_loss=False,\n'
    end = src.find(end_marker, start)
    assert end >= 0, '__init__ not found after staticmethod blocks'
    src = src[:start] + end_marker + src[end + len(end_marker):]
    print('Removed @staticmethod blocks')

    # ── 2. Remove _spectral_net_radiation wrapper ─────────────────────────────
    old_wrapper = (
        '    def _spectral_net_radiation(self, T_hot, f_shield=0.0):\n'
        '        """Spectral radiation to the powerstick using this simulation\'s stored geometry."""\n'
        '        return TinlessSimulation.spectral_net_radiation(\n'
        '            T_hot, self.powerstick.emissivity_table, self.VF, self.A_ps_seg,\n'
        '            self.powerstick.f_bare, self.powerstick.f_active, self.powerstick.liner_bool,\n'
        '            VF_gr_liner=self.VF_gr_liner, VF_liner_ps=self.VF_liner_ps,\n'
        '            A_liner_seg=self.A_liner_seg,\n'
        '            shield_bool=self.shield_bool, f_shield=f_shield,\n'
        '            VF_gr_shield=self.VF_gr_shield, VF_shield_lin=self.VF_shield_lin,\n'
        '            A_shield_seg=self.A_shield_seg,\n'
        '        )\n'
        '\n'
    )
    if old_wrapper not in src:
        print('WARNING: _spectral_net_radiation wrapper not found')
    else:
        src = src.replace(old_wrapper, '', 1)
        print('Removed _spectral_net_radiation wrapper')

    # ── 3. Insert full instance method spectral_net_radiation before _step_calc ─
    snr_method = textwrap.indent('''\
def spectral_net_radiation(self, T_hot, f_shield=0.0):
    """
    Incident radiative power [W] on the powerstick from a hot surface.

    liner_bool=False: graphite inner wall -> powerstick directly. Returns (Q_total, None).
    liner_bool=True:  graphite -> tungsten liner -> powerstick.   Returns (Q_ps, T_liner).
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

    old_step = '    def _step_calc(self, T_dist, f_shield=0.0):\n'
    if old_step not in src:
        print('WARNING: _step_calc not found')
    else:
        src = src.replace(old_step, snr_method + old_step, 1)
        print('Inserted spectral_net_radiation instance method')

    # ── 4. Update _step_calc call (was self._spectral_net_radiation) ──────────
    src = src.replace(
        '            Q_rad[i], T_liner_i = self._spectral_net_radiation(T_wall_i, f_shield=f_shield)\n',
        '            Q_rad[i], T_liner_i = self.spectral_net_radiation(T_wall_i, f_shield=f_shield)\n',
        1,
    )

    # ── 5. Update _liner_avg call ─────────────────────────────────────────────
    src = src.replace(
        '                _, tli = self._spectral_net_radiation(float(T_d[i, 0]))\n',
        '                _, tli = self.spectral_net_radiation(float(T_d[i, 0]))\n',
        1,
    )
    print('Updated _step_calc and _liner_avg calls')

    if src != orig:
        cell['source'] = src
        cell['outputs'] = []
        cell['execution_count'] = None
        print('Patched 03dd9a23')
    else:
        print('WARNING: no changes made')
    break

# ── 6. Update scratch cell ────────────────────────────────────────────────────
# spectral_net_radiation is now an instance method; scratch cell needs its own
# local function or to build a sim instance.  Simplest: define a local wrapper.
for cell in nb['cells']:
    if cell.get('id', '') != 'scratch_shield_01':
        continue
    src = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])
    old_call = '        Q_ps, _ = TinlessSimulation.spectral_net_radiation(\n'
    new_call = '        Q_ps, _ = _snr(\n'
    # Add a local function definition after the derived geometry block
    old_anchor = '# ── Sweep ──'
    new_anchor = (
        '# ── Local radiation function (mirrors TinlessSimulation.spectral_net_radiation) ──\n'
        'def _snr(T_hot, f_shield=0.0):\n'
        '    class _Mock:\n'
        '        pass\n'
        '    m = _Mock()\n'
        '    m.powerstick = _Mock()\n'
        '    m.powerstick.emissivity_table = emissivity_table\n'
        '    m.powerstick.liner_bool = True\n'
        '    m.powerstick.f_bare = f_bare\n'
        '    m.powerstick.f_active = f_active\n'
        '    m.VF = VF_gr_liner\n'
        '    m.A_ps_seg = A_ps\n'
        '    m.shield_bool = True\n'
        '    m.VF_gr_liner = VF_gr_liner\n'
        '    m.VF_liner_ps = VF_liner_ps\n'
        '    m.A_liner_seg = A_liner_seg\n'
        '    m.VF_gr_shield = VF_gr_shield\n'
        '    m.VF_shield_lin = VF_shield_lin\n'
        '    m.A_shield_seg = A_shield_seg\n'
        '    from types import MethodType\n'
        '    return TinlessSimulation.spectral_net_radiation(m, T_hot, f_shield)\n'
        '\n'
        '# ── Sweep ──'
    )
    if old_anchor not in src or old_call not in src:
        print('WARNING: scratch cell patterns not found')
    else:
        src = src.replace(old_anchor, new_anchor, 1)
        src = src.replace(old_call, new_call, 1)
        # Also fix the closing paren line count (the old call had 17 lines, new has 1)
        cell['source'] = src
        cell['outputs'] = []
        cell['execution_count'] = None
        print('Updated scratch cell')
    break

with open('tinless system model.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Saved.')
