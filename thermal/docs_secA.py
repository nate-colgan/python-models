"""Model documentation, part A: inputs (2), geometry and discretisation (3), storage block
conduction (4.1), powerstick and cells (4.5), material properties (7) and the matching
alternative options (appendix A).

Numbers are computed live from the YAML / code where practical (a geometry-only build of the
YAML base case, property functions evaluated on grids) and file:line references are found by
searching the source, so the text follows the code. Each sec_* function returns rows
[area, setting, value, where] for the numerical-settings summary (section 10).
"""
import inspect
import os
import re

import numpy as np

C2K = 273.15
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = {}
DIM = '777777'


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _src(fname):
    if fname not in _SRC:
        try:
            _SRC[fname] = open(os.path.join(_HERE, fname), encoding='utf-8').read().splitlines()
        except OSError:
            _SRC[fname] = []
    return _SRC[fname]


from .heater import describe as _heater_describe

def ref(fname, pattern, nth=1, span=0):
    """'file.py:N' (or 'file.py:N-M' with span) for the nth source line containing `pattern`."""
    n = 0
    for i, ln in enumerate(_src(fname)):
        if pattern in ln:
            n += 1
            if n == nth:
                return f"{fname}:{i + 1}" + (f"-{i + 1 + span}" if span else '')
    print(f"[docs_secA] reference not found: {fname!r} {pattern!r}")
    return fname


def _f(x, nd=3):
    """Compact number."""
    if x is None:
        return '-'
    if isinstance(x, (bool, np.bool_)):
        return str(bool(x))
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}"
    x = float(x)
    if x == 0:
        return '0'
    a = abs(x)
    if a >= 1e5 or a < 1e-3:
        return f"{x:.{nd}g}"
    if a >= 100:
        return f"{x:,.1f}" if a < 1e4 else f"{x:,.0f}"
    return f"{x:.{nd}g}" if a < 1 else f"{x:.{nd + 1}g}"


def _cache(ctx):
    """Geometry-only build of the YAML base case + its heater cell (cached in ctx)."""
    if '_secA' in ctx:
        return ctx['_secA']
    from .build import build_tinless, build_heater
    m = build_tinless(ctx['base_cfg'], geometry_only=True)
    hm = build_heater(m)
    ctx['_secA'] = (m, hm)
    return m, hm


def _plt():
    import matplotlib
    matplotlib.use('Agg', force=False)
    import matplotlib.pyplot as plt
    return plt


def _nsub(storage, dt, sub_steps):
    T = storage.block.temp_dist.copy()
    storage.update_temps(T, np.zeros(storage.num_ax_segments), dt, sub_steps, 300.0, False)
    return storage.last_n_sub


def _dt_crit(storage):
    """min C/sum(G) of the explicit update, as in update_temps (start-of-step properties)."""
    from .properties import storage_tcx, storage_cp, storage_rho
    from .components import _vprop
    T = storage.block.temp_dist
    nA, nR = storage.num_ax_segments, storage.num_rad_segments
    ri, ro = storage.block.rad_in, storage.block.rad_out
    dr = (ro - ri) / nR
    dz = storage.axial_dz()
    dzc = 0.5 * (dz[1:] + dz[:-1])
    r_lo = ri + np.arange(nR) * dr
    ring = np.pi * ((r_lo + dr) ** 2 - r_lo ** 2)
    ln_out = np.log((r_lo + 1.5 * dr) / (r_lo + 0.5 * dr))
    ln_in = np.log((r_lo + 0.5 * dr) / (r_lo - 0.5 * dr))
    k = _vprop(storage_tcx, T)
    C = _vprop(storage_cp, T) * _vprop(storage_rho, T) * ring[None, :] * dz[:, None]
    G = np.zeros_like(T)
    G[:, :-1] += k[:, :-1] * 2 * np.pi * dz[:, None] / ln_out[None, :-1]
    G[:, 1:] += k[:, 1:] * 2 * np.pi * dz[:, None] / ln_in[None, 1:]
    if nA > 1:
        G[:-1] += k[:-1] / dzc[:, None] * ring[None, :]
        G[1:] += k[1:] / dzc[:, None] * ring[None, :]
    return float(np.min(C / G))


def _refcount(word):
    """Number of non-docs source files in thermal/ that mention `word`."""
    n = 0
    for fn in os.listdir(_HERE):
        if fn.endswith('.py') and not fn.startswith('docs'):
            if word in open(os.path.join(_HERE, fn), encoding='utf-8').read():
                n += 1
    return n


# ==========================================================================
# 2. Inputs, 3. Geometry, 4.1 Block conduction
# ==========================================================================
def sec_inputs_geometry(doc, ctx):
    from .docs import yaml_table
    from .properties import storage_rho, storage_cp, storage_tcx
    cfg = ctx['base_cfg']
    m, hm = _cache(ctx)
    s, p, sh, h = cfg['storage'], cfg['powerstick'], cfg.get('shade', {}), cfg['heater']
    rows = []

    # ---------------------------------------------------------------- 2
    doc.h1('2. Inputs')
    doc.p("The cell model is driven by one YAML file, ", (ctx['yaml_path'], {'mono': True}), ". ",
          ("build.load_config", {'mono': True}), f" ({ref('build.py', 'def load_config')}) reads it. It tries "
          "UTF-8, then cp1252, then latin-1, because the notebook cell that writes the file uses the platform "
          "encoding. ", ("build.build_tinless(cfg, overrides)", {'mono': True}),
          " applies dotted-key overrides to a deep copy, for example {'storage.L': 4.0} "
          f"({ref('build.py', 'def _apply_overrides')}), so the input dict is never modified. The optimizer "
          f"overwrites some keys for every design ({ref('optimize.py', 'def design_cfg')}; section 9). Those keys "
          "are flagged in the table below, and their YAML values only matter when the model is run on its own "
          "from the notebook.")
    doc.h2('2.1 YAML inputs')
    doc.p("Every leaf of the YAML is listed below with its value and the comment from the file. Temperatures in "
          "the storage, powerstick and heater blocks are in K; tank temperatures are in C. Lengths are in m.")
    yaml_table(doc, ctx['yaml_path'])
    T_min, T_max = float(s['T_min']), float(s['T_min']) + float(s['block_temp_range'])
    doc.p(f"Derived at build time: T_max = T_min + block_temp_range = {T_max:.0f} K "
          f"({T_max - C2K:.0f} C); T_avg = {T_min + s['block_temp_range'] / 2:.0f} K "
          f"({ref('build.py', 'T_max = T_min + block_temp_range')}).")

    doc.h2('2.2 Inputs that are ignored, overridden or unused')
    doc.p("Several keys are kept for other model options or for history. In the configuration documented here "
          "(axial radiation, vane shades, graphite liner, liner bottom enabled, block taller than the stick), "
          "they are not used. The last column is checked against the code each time the document is generated.")
    unused = [
        ['powerstick.area_ratio', 'ignored', 'With shade + liner the cavity is the smallest that fits the vanes, and '
         'area_ratio is recomputed as r_in / r_ps (section 3.3).', ref('build.py', 'area_ratio = r_in / r_ps_eff')],
        ['shade.liner_to_shade_perimeter_ratio', 'overridden',
         'Sizes the legacy porous shade. For vanes, shade_side_len = l(1+sqrt2), the octagon perimeter / 4.',
         ref('build.py', "shade_side_len = vane_lay['l']")],
        ['shade.num_conduction_layers', 'unused',
         'Stored on the simulation object and never read. The vanes use the exact Kirchhoff integral (section 4.3).',
         ref('tinless.py', 'self.shade_num_conduction_layers =')],
        ['cavity_ends.emissivity', 'unused', 'Copied into the build result only. The floor and ceiling are '
         'reradiating, so their emissivity does not enter.', ref('build.py', "ce_cfg = cfg.get('cavity_ends'")],
        ['cavity_ends.ceiling_gap', 'unused', 'Not read anywhere (the gap is treated as closed). Referenced in '
         f"{_refcount('ceiling_gap')} source files.", 'viewfactors.py (comment)'],
        ['radiation.vf_method, vf_rays', 'unused here', 'Only for the unshaded or porous-shade enclosures. The vane '
         'view factors use vane_rays.', ref('build.py', "n_rays = int(rad_cfg.get('vf_rays'")],
        ['emissivity_tables[0].range', 'ignored', "YAML reads 'end_time-1' as a string. Only the first table's "
         "value is used, so time-varying tables are not supported.", ref('build.py', "raw_emis = cfg['emissivity_tables']")],
        ['print_outputs_bool', 'unused', 'Read only by the legacy tin-loop model (simulation.py).', 'simulation.py'],
        ['T_sur, storage.th_ins', 'inactive', 'Used only by the outer-wall loss, which is off '
         '(block_heat_loss_bool: False).', ref('components.py', 'if block_heat_loss:')],
        ['powerstick.cooling_temp', 'partly used', 'Only sets the default bottom-cap temperature. It does not enter '
         'the cell efficiency or the stick emission (section 4.5).', ref('build.py', 'T_cap=float(T_ps if T_cap')],
        ['heater.charge_hr', 'notebook only', 'Read by the model notebook cycle cell. The optimizer has its own '
         'charge_hr (section 9).', 'notebook'],
        ['tank.block_axial_losses', 'discharge only', 'Gates the roof/floor sink in the discharge. The charge '
         'always applies the sink.', ref('sizing.py', "t_cfg.get('block_axial_losses'")],
    ]
    doc.table(['key', 'status', 'why', 'where'], unused, widths=[1.7, 0.8, 3.3, 1.3], size=8)
    doc.h3('YAML comments that disagree with the code')
    end_h = float(cfg['end_time']) / 3600
    doc.table(['key', 'YAML comment says', 'code does', 'where'], [
        ['end_time', '"(50 hr discharge)"', f"{cfg['end_time']} s = {end_h:.1f} h. The run normally stops "
         "earlier, at the T_min cutoff. The optimizer uses max(end_time, 4 x duration).", ref('optimize.py', "cfg['end_time'] = max(")],
        ['dt_long', '"used when shade is clamped at f_shade_max"', 'dt_long is used when the shade is fully open '
         '(f = f_min) or absent, and in the conduction-only tail. At f_max the step is dt_short, or '
         'dt_at_shade_limit (the optimizer sets dt_long).', ref('tinless.py', 'if t < dt_short_duration:', span=7)],
        ['storage.avg_flux', '"average power density to extract"', 'Absorbed flux in the active cells per active-cell '
         'area. It is the sizing basis and the shade setpoint; it is not electric output.', ref('build.py', 'E_req = avg_flux')],
        ['emissivity_tables.range', '"time-varying tables"', 'Ignored (see above).', ref('build.py', "raw_emis = cfg['emissivity_tables']")],
    ], widths=[1.2, 1.7, 3.0, 1.2], size=8)
    doc.review(f"YAML comments for end_time (says 50 h, is {end_h:.1f} h) and dt_long (the code uses it with the "
               "shade fully open, not clamped closed) disagree with the code, and several keys are unused or "
               "ignored (table 2.2). Confidence high; no effect on results, but misleading for users "
               f"({ref('tinless.py', 'if t < dt_short_duration:')}).")
    doc.h2('2.3 Defaults used when a key is missing')
    doc.p("build_tinless falls back to these values when a key is absent. None of them are active with the current "
          "YAML, except volume_factor (absent, so 1.0, unless the sizing loop sets it).")
    dflt = [
        ['liner_material', "'tac'", ref('build.py', "liner_material = cfg.get('liner_material'")],
        ['powerstick.liner_thickness', '0.0 m', ref('build.py', "liner_thickness = p.get('liner_thickness'")],
        ['powerstick.tungsten_foil', 'False', ref('build.py', "p.get('tungsten_foil', False)")],
        ['shade_bool', 'False', ref('build.py', "shade_bool = sh.get('shade_bool'")],
        ['shade.thickness', '0.05 m', ref('build.py', "shade_thickness = sh.get('thickness'")],
        ['shade.surface_emissivity', '0.8', ref('build.py', "shade_eps = sh.get('surface_emissivity'")],
        ['shade.num_conduction_layers', '5 (unused)', ref('build.py', "sh.get('num_conduction_layers'")],
        ['shade.corner_gap', '0.003175 m', ref('build.py', "corner_gap = sh.get('corner_gap'")],
        ['shade.model', "'vanes'", ref('build.py', "shade_model = str(sh.get('model'")],
        ['shade.vane_clearance', '0.005 m', ref('build.py', "vane_clear = float(sh.get('vane_clearance'")],
        ['storage.block_height', 'L', ref('build.py', "H_block = s.get('block_height'")],
        ['storage.volume_factor', '1.0', ref('build.py', 'V_block = E_req / e_dens')],
        ['radiation.axial', 'False', ref('build.py', "axial = bool(rad_cfg.get('axial'")],
        ['radiation.vane_rays', '2^14', ref('build.py', "rad_cfg.get('vane_rays'")],
        ['liner_bottom.enabled', 'False', ref('build.py', "lb_enabled = bool(lb_cfg.get('enabled'")],
        ['liner_bottom.rgf_thickness', '0.025 m (YAML 0.05)', ref('build.py', "t_rgf = float(lb_cfg.get('rgf_thickness'")],
        ['liner_bottom.rgf_outer_emissivity', 'shade.surface_emissivity', ref('build.py', "eps_R = lb_cfg.get('rgf_outer_emissivity'")],
        ['bottom_cap.temperature', 'powerstick.cooling_temp', ref('build.py', "T_cap = (cfg.get('bottom_cap'")],
        ['dt_short_duration', '3600 s', ref('build.py', "dt_short_duration=cfg.get('dt_short_duration'")],
        ['degradation.emis_end', '0.8', ref('build.py', "e_end = float(deg.get('emis_end'")],
        ['powerstick.tungsten_foil_thickness', '25 um (deposition only)', ref('build.py', "tungsten_foil_thickness")],
        ['heater.max_power', 'max_power_per_m x H (KeyError if both missing)', ref('build.py', 'def heater_max_power')],
    ]
    doc.table(['key', 'default', 'where'], dflt, widths=[2.4, 2.8, 1.9], size=8)

    # ---------------------------------------------------------------- 3
    lay = m.vane_lay
    me = m.mesh
    st = m.storage
    nR = st.num_rad_segments
    dr = (m.r_out - m.r_in) / nR
    dr_h = (hm.r_out - hm.r_in) / nR
    doc.h1('3. Geometry and discretisation')
    doc.p("Geometry is built by build.build_tinless (stick cell) and build.build_heater (heater cell). All "
          "numbers in this section are for the YAML base case and are computed when the document is generated. "
          "The optimizer's designs change L, H, the liner and stick sides, the vane thickness, the flux and the "
          "temperatures (section 9). The example design is in section 8.")
    fig = _build_flowchart()
    doc.fig(fig, 'Steps of build.build_tinless and build.build_heater.', width=4.8, name='Figure 3.1')

    doc.h2('3.1 The stick cell')
    doc.p("One powerstick and its share of the graphite form an axisymmetric cell. A square, water-cooled stick of "
          f"side s_ps = {p['side_len'] * 1e3:.0f} mm and length L = {s['L']} m hangs top-aligned inside a square "
          f"graphite liner of side s_l = {p['liner_side_len'] * 1e3:.0f} mm and wall {p['liner_thickness'] * 1e3:.1f} mm. "
          "A ring of eight rotating graphite-felt (RGF) vanes surrounds the liner. All of it sits in a square cavity "
          f"in a graphite block of height H = {m.H_block:g} m. The outer radius of the cell is a symmetry line, so "
          "it is adiabatic. The region below the stick (the lower zone) is part of the same cavity: its walls "
          "radiate to everything in the cavity, and a liner-bottom plate with an RGF pad closes the liner at its "
          "lower end.")
    fig = _plan_figure(m)
    if fig is not None:
        doc.fig(fig, 'Plan section of the stick cell (YAML base case) to scale: square cavity, vanes closed '
                     '(solid) and open (dashed), liner, stick, and the perimeter-equivalent circles used by the '
                     'model (dotted). The outer dashed circle is the adiabatic cell boundary r_out.',
                width=4.6, name='Figure 3.2')

    doc.h2('3.2 Equivalent circles')
    doc.p("Every square section is replaced by the circle with the same perimeter, so surface areas (and "
          "radiation areas) are exact:")
    doc.eq(r'r = \frac{2s}{\pi}\qquad (2\pi r = 4s)',
           where='s: side of the square (stick, liner, cavity, heater; the vane octagon uses its perimeter / 4).')
    doc.table(['surface', 'square side', 'equivalent radius', 'where'], [
        ['powerstick', f"{p['side_len'] * 1e3:.0f} mm", f"r_ps = {m.r_ps_eff * 1e3:.1f} mm", ref('build.py', 'r_ps_eff = 2 * side_len / np.pi')],
        ['liner', f"{p['liner_side_len'] * 1e3:.0f} mm", f"r_l = {m.r_liner_eff * 1e3:.1f} mm", ref('build.py', 'r_liner_eff = 2 * liner_side_len / np.pi')],
        ['vane octagon (closed)', f"perimeter/4 = {m.shade_side_len * 1e3:.0f} mm",
         f"r_sh = {2 * m.shade_side_len / np.pi * 1e3:.1f} mm", ref('build.py', "shade_side_len = vane_lay['l']")],
        ['cavity', f"S = {np.pi * m.r_in / 2 * 1e3:.0f} mm", f"r_in = {m.r_in * 1e3:.1f} mm", ref('build.py', 'r_in = 2.0 * S_min / np.pi')],
        ['heater element', _heater_describe(hm.element), f"r_h = {hm.r_heater * 1e3:.1f} mm", ref('heater.py', 'def element')],
    ], widths=[1.5, 1.7, 1.8, 2.1], size=8.5)
    doc.p("Cross-sections are not preserved: a circle with the perimeter of a square has 4/pi = 1.27 times its "
          "area. The block volume is set directly from the energy requirement (3.4), so the graphite mass is "
          "right. The radial conduction path, however, is a cylindrical annulus, not a square bore inside a "
          "square checkerboard cell.")
    doc.review(f"Equivalent-perimeter circles keep areas but not the conduction geometry. The cavity of side "
               f"{np.pi * m.r_in / 2:.2f} m becomes r_in = {m.r_in:.3f} m inside a {m.r_out - m.r_in:.2f} m-thick "
               "annulus with an adiabatic circular boundary. The real cell has a square bore and square symmetry "
               "lines with corners. Effect on the discharge-end temperature field not quantified (medium "
               "confidence that it matters at the % level). A 2-D check of one plan section would bound it.")

    doc.h2('3.3 Cavity sizing for the vanes')
    doc.p("With a shade and a liner, powerstick.area_ratio is ignored. The cavity is the smallest square that "
          "lets the eight vanes swing fully open without touching the liner or the wall "
          f"({ref('correlations.py', 'def vane_layout_for_liner')}; layout in {ref('correlations.py', 'def _vane_boxes')}). "
          "Face vanes have length l and pivot at 1.5 l on the axes. Corner vanes have length l sqrt2 and pivot at "
          "l sqrt2 on the diagonals. Closed, they form an octagon (a 3l square with l-corners cut). Open (90 deg), "
          "a face vane spans l..2l radially and a corner vane spans l sqrt2/2..3l sqrt2/2 along the diagonal. "
          "With clearance c:")
    doc.eq(r'l = s_l + \sqrt{2}\,c,\qquad S_{min} = 4\,l + 2\,c,\qquad r_{in} = \frac{2\,S_{min}}{\pi},\qquad '
           r'AR = \frac{r_{in}}{r_{ps}}',
           where='first: open corner vanes clear the liner corners; second: open face vanes clear the wall.')
    doc.p(f"For the base case: c = {cfg['shade'].get('vane_clearance', 0.005) * 1e3:.0f} mm, l = {lay['l'] * 1e3:.1f} mm, "
          f"S_min = {2 * lay['A'] * 1e3:.0f} mm, r_in = {m.r_in * 1e3:.1f} mm, output AR = {m.area_ratio:.2f} "
          f"(the YAML area_ratio {p['area_ratio']} is ignored). Closed vane ends are shortened by "
          "t tan(pi/8)/2 + gap/2 so adjacent inner corners are `gap` apart "
          f"({ref('correlations.py', 'shorten = 0.5 * t * np.tan')}), with vane thickness "
          f"t = {lay['t'] * 1e3:.1f} mm and gap = {lay['gap'] * 1e3:.3f} mm. Vane physics is in section 4.3.")
    doc.p(f"build.py also re-checks the cavity against S_min after setting the vane layout "
          f"({ref('build.py', 'if S_cav < S_min - 1e-9:')}). The cavity has already been set to S_min, so "
          "this enlargement branch cannot trigger. The legacy porous-shade feasibility checks in "
          f"TinlessSimulation.__init__ ({ref('tinless.py', 'shade geometry infeasible: r_shade=')}) still run, "
          "on the octagon-equivalent radius r_sh. They always pass for the vane layout.")
    rows += [['geometry', 'vane clearance c', f"{cfg['shade'].get('vane_clearance', 0.005) * 1e3:.0f} mm",
              ref('build.py', "vane_clear = float(sh.get('vane_clearance'")],
             ['geometry', 'cavity side', 'S = 4 l + 2c (minimum), l = s_l + sqrt2 c', ref('correlations.py', 'S_min = 4.0 * l + 2.0 * clearance')]]

    doc.h2('3.4 Block volume and radii from the energy requirement')
    doc.p("The block holds the heat the stick absorbs over the design discharge at the target cell flux. "
          "Fractions of the stick surface:")
    doc.eq(r'f_{cell} = (1-f_{bare})\,f_{active},\qquad f_{inact} = (1-f_{bare})\,(1-f_{active})',
           where=f"f_bare = {p['f_bare']}, f_active = {p['f_active']}: f_cell = {(1 - p['f_bare']) * p['f_active']:.3f}, "
                 f"f_inact = {(1 - p['f_bare']) * (1 - p['f_active']):.3f}")
    doc.p("f_Q is the fraction of the heat absorbed by the stick that lands on active cells. It is band-weighted "
          "by the two-surface net flux q_b into the stick in each emissivity-table band b "
          f"({ref('build.py', 'f_Q = sum(')}):")
    doc.eq(r'f_Q = \sum_b w_b\,\frac{f_{cell}\,\varepsilon_{a,b}}{f_{bare}\,\varepsilon_{bare,b} + f_{cell}\,'
           r'\varepsilon_{a,b} + f_{inact}\,\varepsilon_{in,b}},\qquad w_b = q_b \,/\, \sum q_b')
    doc.eq(r'q_b = \int_b \frac{E_{b\lambda}(T_e)}{R_\lambda}\,d\lambda,\qquad R_\lambda = \frac{r_{ps}}{r_{in}}\,'
           r'\frac{1-\varepsilon_e}{\varepsilon_e} + 1 + \frac{1-\bar{\varepsilon}_{ps,b}}{\bar{\varepsilon}_{ps,b}},'
           r'\qquad T_e = T_{min} - 100\,\mathrm{K},\quad \varepsilon_e = \varepsilon_{TaC}(T_e)',
           where=f"PowerStick.incident_energy ({ref('components.py', 'def incident_energy')}): 300 log-spaced "
                 "wavelength points over the table range; the emitter emissivity is tac_emis_temp whenever a liner "
                 "is present.")
    doc.eq(r'E_{req} = q_{set}\,A_{ps}\,f_{cell}\,\frac{t_{dis}}{f_Q},\qquad A_{ps} = 4\,s_{ps}\,L',
           where='q_set = storage.avg_flux (W/cm2 x 1e4), t_dis = storage.duration_hr x 3600.')
    doc.eq(r'V_{block} = \phi_V\,\frac{E_{req}}{\rho(\bar{T})\,c_p(\bar{T})\,\Delta T},\qquad '
           r'r_{out} = \sqrt{\frac{V_{block}}{\pi H} + r_{in}^2}',
           where='phi_V = storage.volume_factor (sizing correction, default 1); T-bar = T_min + Delta T / 2.')
    rho_, cp_ = float(storage_rho(T_min + s['block_temp_range'] / 2)), float(storage_cp(T_min + s['block_temp_range'] / 2))
    doc.table(['quantity', 'value', 'where'], [
        ['A_ps (stick area)', f"{m.A_ps:.3f} m2", ref('build.py', 'A_ps = 4 * side_len * L')],
        ['f_Q', f"{m.f_Q:.4f}", ref('build.py', 'f_Q = sum(')],
        ['cell power at target flux', f"{s['avg_flux'] * 1e4 * m.A_ps * (1 - p['f_bare']) * p['f_active'] / 1e3:.1f} kW", ''],
        ['E_req', f"{m.E_req / 3.6e9:.3f} MWh(th)", ref('build.py', 'E_req = avg_flux')],
        ['rho c_p Delta T', f"{rho_:.0f} x {cp_:.0f} x {s['block_temp_range']} = {rho_ * cp_ * s['block_temp_range'] / 1e9:.3f} GJ/m3",
         ref('build.py', 'e_dens = storage_rho')],
        ['V_block (volume_factor 1)', f"{m.V_block:.3f} m3 ({m.V_block * rho_ / 1e3:.1f} t)", ref('build.py', 'V_block = E_req / e_dens')],
        ['r_in / r_out', f"{m.r_in:.4f} / {m.r_out:.4f} m", ref('build.py', 'r_out = np.sqrt(V_block / (np.pi * H_block)')],
    ], widths=[2.2, 3.0, 1.9], size=8.5)
    doc.p("The annulus spans the full block height H, including the lower zone below the stick. The volume "
          "assumes the whole block swings uniformly through Delta T, while the stick is withdrawn when the "
          "volume-averaged temperature reaches T_min. Roof and floor losses, and the time the shades spend fully "
          "open with the flux below target, are not in E_req. The shade-open sizing loop (sizing.py, section 6) "
          "corrects for them through volume_factor and T_min.")
    doc.review(f"f_Q, which feeds E_req and V_block directly, uses a different spectral model from the transient "
               f"({ref('build.py', 'inc_bins, _, _ = ps_tmp.incident_energy')}). It uses the legacy 300-bin grid "
               "(about 2 % error noted in tinless.py), the TaC emissivity for any liner (here graphite), the cavity "
               "view factor r_ps/r_in instead of the liner's, and an assumed emitter temperature T_min - 100 K. "
               f"f_Q = {m.f_Q:.3f} is a band ratio, so the error is probably around 1 % in V_block (medium "
               "confidence). It disappears if the sizing loop always runs, but build_tinless on its own is "
               "biased.")
    doc.review("build_tinless alone undersizes the block: E_req excludes roof/floor losses and the shade-open "
               "shortfall. Reported designs must come from the volume_factor / T_min iteration (sizing.py). "
               "Confidence high that this is by design; check that no output path uses an un-iterated build.")
    rows += [['sizing', 'f_Q emitter temperature', 'T_min - 100 K, TaC emissivity', ref('build.py', 'inc_bins, _, _ = ps_tmp.incident_energy')],
             ['sizing', 'f_Q spectral grid', '300 log-spaced points, 0.1-20 um', ref('components.py', 'n_bins = 300')],
             ['sizing', 'volume_factor', '1.0 unless set by sizing loop', ref('build.py', 'V_block = E_req / e_dens')]]

    doc.h2('3.5 Lower zone, liner bottom and RGF pad')
    doc.p(f"z = 0 is the cavity floor. The stick and liner (length L) hang top-aligned, so their lower ends are at "
          f"z_act0 = H - L = {me.z_act0:.3f} m. The liner bottom is a graphite plate of thickness "
          f"t_lb = liner_thickness = {me.t_lb * 1e3:.1f} mm ({ref('build.py', 't_lb = liner_thickness')}). Under it is "
          f"an RGF pad of thickness t_RGF = {me.t_rgf * 1e3:.0f} mm. The pad underside is at")
    doc.eq(r'H_{low} = H - L - t_{lb} - t_{RGF}', where=f"= {me.H_low:.3f} m ({ref('build.py', 'H_low = H_block - L - t_lb - t_rgf')}); "
                                                         "an error is raised if negative.")
    doc.p("The space below H_low is the lower zone: block wall, cavity floor and the pad underside, all inside "
          "the same radiation enclosure as the active zone (section 4.2). The plate and pad are cavity "
          "surfaces, not block rings. The pad underside emissivity defaults to the vane emissivity "
          f"({_f(m.liner_bottom.get('rgf_outer_emissivity') or sh.get('surface_emissivity', 0.8))}), and the "
          "bottom-cap temperature to the stick cooling temperature. The liner-bottom physics is in section 4.4. "
          "If block_height == L there is no lower zone, and the liner-bottom model is switched off with a message "
          f"({ref('build.py', 'block_height == L: no space below the liner')}). Block_height > L needs "
          f"radiation.axial: true ({ref('build.py', 'needs radiation.axial: true')}).")

    doc.h2('3.6 Axial and radial mesh')
    doc.p(f"storage.num_ax_segments = n_ax is the TOTAL number of block rings over H "
          f"({ref('viewfactors.py', 'def axial_mesh')}). It is split between the lower and active zones in "
          "proportion to height, with a ring edge exactly at z_act0:")
    doc.eq(r'n_{act} = \min\left(\max\left(\mathrm{round}\left(n_{ax}\,\frac{L}{H}\right),\,1\right),\,n_{ax}-1\right),'
           r'\qquad n_{low} = n_{ax} - n_{act}')
    doc.eq(r'\Delta z_{low} = \frac{z_{act0}}{n_{low}},\qquad \Delta z_{act} = \frac{L}{n_{act}}',
           where='ring heights differ slightly between the zones unless the split divides exactly.')
    doc.p(f"Base case: n_ax = {s['num_ax_segments']} gives {me.n_low} lower + {me.n_act} active rings, "
          f"dz_low = {me.z_act0 / max(me.n_low, 1):.3f} m and dz_act = {s['L'] / me.n_act:.3f} m. The powerstick gets one axial "
          f"step per active ring (A_step = 4 s_ps dz_act = {m.powerstick.A_step:.3f} m2). The optimizer sets "
          f"n_ax = max(10, round(H / dz)) with dz = {ctx['opt']['fidelity']['dz']} m for screening "
          "(" + ref('optimize.py', "s['num_ax_segments'] = max(10") + "). If H > L, n_ax must be at least 2.")
    doc.p(f"Radially the annulus r_in..r_out is cut into n_R = {nR} rings of EQUAL THICKNESS "
          f"dr = (r_out - r_in)/n_R = {dr * 1e3:.1f} mm, not equal area ({ref('components.py', 'dr = (r_outer - r_inner) / nR')}). "
          "Nodes sit at the arithmetic ring mid-radius r_c,j = r_in + (j + 1/2) dr. Ring volumes are the exact "
          "annulus volumes, so outer rings carry more mass. With non-uniform ring heights, the edges are "
          f"installed with StorageModule.set_axial_mesh ({ref('components.py', 'def set_axial_mesh')}); they must "
          "be strictly increasing, with n_ax + 1 entries.")
    fig = _rz_figure(m, hm)
    if fig is not None:
        doc.fig(fig, 'Vertical (r-z) sections with the conduction mesh, stick cell (left) and heater cell (right), '
                     'YAML base case, radial scale exaggerated. Grey lines: ring boundaries; dots: nodes. Blue: '
                     'stick; black: liner, liner-bottom plate; orange: RGF pad; green dashed: vane ring (octagon '
                     'equivalent); red: heater rod.', name='Figure 3.3')
    rows += [['block mesh', 'axial rings (total over H)', f"{s['num_ax_segments']} (YAML); optimizer max(10, round(H/dz))",
              ref('viewfactors.py', 'def axial_mesh')],
             ['block mesh', 'lower / active split', f"{me.n_low} / {me.n_act}, edge at z = H - L",
              ref('viewfactors.py', 'n_act = min(max(int(round(')],
             ['block mesh', 'radial rings', f"{nR}, uniform in r (dr = {dr * 1e3:.1f} mm stick, {dr_h * 1e3:.1f} mm heater)",
              ref('components.py', 'dr = (r_outer - r_inner) / nR')],
             ['block mesh', 'node position', 'ring mid-radius r_lo + dr/2; ring mid-height', ref('components.py', 'ln_out = np.log')]]

    doc.h2('3.7 Heater cell')
    doc.p(f"Each stick has one heater (checkerboard, N_heaters = N_sticks). The charge is modelled in a separate "
          f"cell ({ref('build.py', 'def build_heater')}). It has the same graphite volume, height, axial edges and "
          "radial ring count as the stick cell, around a round heater cavity:")
    doc.eq(r'r_h = \frac{2\,s_h}{\pi},\qquad r_{hc} = AR_h\,r_h,\qquad r_{out,h} = '
           r'\sqrt{\frac{V_{block}}{\pi H} + r_{hc}^2},\qquad P_{max} = p^\prime_{max}\,H',
           where='AR_h = heater.area_ratio (D_cavity / D_heater); p\'_max = heater.max_power_per_m, unless an absolute '
                 'heater.max_power is given (the optimizer sizes it).')
    doc.table(['quantity', 'value', 'where'], [
        ['heater radiating radius r_h', f"{hm.r_heater * 1e3:.1f} mm", ref('heater.py', 'def element')],
        ['cavity radius r_hc', f"{hm.r_in * 1e3:.1f} mm (AR {hm.area_ratio:g})", ref('build.py', 'r_in = ar * r_h')],
        ['outer radius r_out,h', f"{hm.r_out * 1e3:.1f} mm", ref('build.py', 'r_out = float(np.sqrt(V / (np.pi * H)')],
        ['graphite volume', f"{hm.V:.3f} m3 (same as stick cell)", ref('build.py', 'V = np.pi * (st.block.rad_out')],
        ['heater power cap', f"{hm.heater.max_power / 1e6:.3f} MW", ref('build.py', 'def heater_max_power')],
        ['heater max surface T / emissivity', f"{h['max_temperature']} K / {h['emissivity']}", ref('build.py', 'heater = HeatingElement(')],
        ['initial block T', f"uniform T_min = {hm.T_min:.0f} K", ref('build.py', 'initial_temp_distribution=np.full((nA, nR), float(m.T_min))')],
        ['charge stops at', f"volume-average T >= T_max = {hm.T_max:.0f} K", ref('build.py', 'kw = dict(T_block_max')],
        ['time step', f"dt_short = {cfg['dt_short']} s (x dt_scale in the optimizer)", ref('build.py', 'kw = dict(T_block_max')],
    ], widths=[2.6, 2.6, 1.9], size=8.5)
    doc.p("A ValueError is raised if r_h >= r_hc. The heater cell has no liner, stick or vanes; a dummy "
          "PowerStick with liner_bool = False satisfies the simulation constructor. The heater radiation and the "
          "power/temperature control are described in section 4.6.")
    doc.review(f"The charge and the discharge are modelled in two different adiabatic cells, each starting from a "
               f"uniform temperature: T_min for the heater cell ({ref('build.py', 'initial_temp_distribution=np.full((nA, nR), float(m.T_min))')}) "
               f"and T_max for the stick cell ({ref('build.py', 'initial_temp_distribution=np.full((n_ax_tot')}). In "
               "the real checkerboard, heat from a heater must cross the pitch to reach a stick cavity, and the "
               "discharge starts from the field the charge left. Neither is represented. Medium confidence that "
               "this affects cycle efficiency and peak graphite temperature at the few-% / tens-of-K level.")
    rows += [['initial conditions', 'discharge block T', 'uniform T_max', ref('build.py', 'initial_temp_distribution=np.full((n_ax_tot')],
             ['initial conditions', 'charge block T', 'uniform T_min', ref('build.py', 'initial_temp_distribution=np.full((nA, nR), float(m.T_min))')]]

    doc.h2('3.8 Derived geometry summary')
    cy = ctx.get('cy')
    cols = [('YAML base case', m, hm)]
    if cy is not None and getattr(cy, 'm', None) is not None:
        cols.append(('example design (sec. 8)', cy.m, getattr(cy, 'hm', None)))
    names = ['stick length L', 'block height H', 'rings lower + active', 'r_ps / r_liner', 'vane length l / thickness t',
             'cavity side S / r_in', 'r_out', 'block volume', 'E_req (thermal)', 'f_Q', 'heater r_hc / r_out,h',
             'heater power cap']

    def _col(mm, hh):
        try:
            lv = mm.vane_lay or {}
            out = [f"{mm.L:.2f} m", f"{mm.H_block:.2f} m", f"{mm.mesh.n_low} + {mm.mesh.n_act}",
                   f"{mm.r_ps_eff * 1e3:.1f} / {mm.r_liner_eff * 1e3:.1f} mm",
                   f"{lv.get('l', np.nan) * 1e3:.1f} / {lv.get('t', np.nan) * 1e3:.1f} mm",
                   f"{np.pi * mm.r_in / 2 * 1e3:.0f} / {mm.r_in * 1e3:.1f} mm", f"{mm.r_out * 1e3:.1f} mm",
                   f"{mm.V_block:.2f} m3", f"{mm.E_req / 3.6e9:.2f} MWh", f"{mm.f_Q:.4f}"]
            out += ([f"{hh.r_in * 1e3:.0f} / {hh.r_out * 1e3:.0f} mm", f"{hh.heater.max_power / 1e3:,.0f} kW"]
                    if hh is not None else ['-', '-'])
            return out
        except Exception as e:                      # example object without the expected fields
            return [f'({type(e).__name__})'] * len(names)
    data = [_col(mm, hh) for _, mm, hh in cols]
    doc.table(['quantity'] + [c[0] for c in cols], [[n] + [d[i] for d in data] for i, n in enumerate(names)],
              widths=[2.3] + [4.8 / len(cols)] * len(cols), size=8.5)

    # ---------------------------------------------------------------- 4 / 4.1
    rows += _sec_block(doc, ctx, m, hm)
    return rows


def _sec_block(doc, ctx, m, hm):
    from .properties import storage_rho, storage_cp, storage_tcx
    from .components import StorageModule
    cfg = ctx['base_cfg']
    rows = []
    st = m.storage
    nR, nA = st.num_rad_segments, st.num_ax_segments
    dr = (m.r_out - m.r_in) / nR
    Tm = float(m.T_max)
    k0, cp0, rho0 = float(storage_tcx(Tm)), float(storage_cp(Tm)), float(storage_rho(Tm))
    alpha = k0 / (rho0 * cp0)
    sig = inspect.signature(StorageModule.update_temps).parameters
    safety = sig['safety'].default
    dtc_s, dtc_h = _dt_crit(st), _dt_crit(hm.storage)
    ns = {dt: _nsub(st, dt, cfg['sub_steps']) for dt in (cfg['dt_short'], cfg['dt_medium'], cfg['dt_long'])}
    ns_h = _nsub(hm.storage, cfg['dt_short'], cfg['sub_steps'])

    doc.h1('4. Component physics')
    doc.p("This section describes the physics of each component of the cell model. 4.1 covers the graphite block "
          "conduction shared by the stick and heater cells. 4.2-4.4 cover the cavity radiation, vanes and liner; "
          "4.5 the stick and cells; 4.6 the heater; 4.7 deposition. Time stepping and the transient driver are in "
          "section 6.")
    doc.h2('4.1 Storage block and conduction')
    doc.p(f"The block temperature field T[i, j] (axial ring i = 0 at the bottom, radial ring j = 0 at the cavity "
          f"wall) is advanced by StorageModule.update_temps ({ref('components.py', 'def update_temps')}). The routine "
          "is called once per transient time step, with the inner-wall heat input from the radiation model "
          "(discharge) or the heater (charge), and an optional prescribed sink.")
    doc.h3('Governing equation')
    doc.eq(r'\rho c_p\,\frac{\partial T}{\partial t} = \frac{1}{r}\,\frac{\partial}{\partial r}\left(k\,r\,'
           r'\frac{\partial T}{\partial r}\right) + \frac{\partial}{\partial z}\left(k\,\frac{\partial T}{\partial z}\right),'
           r'\qquad r_{in} \leq r \leq r_{out},\;\; 0 \leq z \leq H',
           where='axisymmetric, no internal generation; rho, c_p and k are functions of T through the property '
                 'functions (constants at present, section 7.1).')
    doc.h3('Finite-volume discretisation')
    doc.p("Each node is a ring of rectangular r-z section. Its capacity and the conductances to its neighbours are:")
    doc.eq(r'C_{ij} = \rho(T_{ij})\,c_p(T_{ij})\,\pi\,(r_{hi,j}^2 - r_{lo,j}^2)\,\Delta z_i,\qquad '
           r'r_{lo,j} = r_{in} + j\,\Delta r,\;\; r_{hi,j} = r_{lo,j} + \Delta r',
           where=ref('components.py', 'C = _vprop(storage_cp, T)'))
    doc.eq(r'G^{r}_{i,j\rightarrow j+1} = \frac{2\pi\,k\,\Delta z_i}{\ln(r_{c,j+1}/r_{c,j})},\qquad '
           r'G^{z}_{i\rightarrow i+1,\,j} = \frac{k\,\pi\,(r_{hi,j}^2 - r_{lo,j}^2)}{(\Delta z_i + \Delta z_{i+1})/2}',
           where=f"radial: exact steady cylindrical conduction between node radii "
                 f"({ref('components.py', 'hout_r[:, :-1] =')}); axial: centre-to-centre distance, valid for "
                 f"non-uniform ring heights ({ref('components.py', 'hout_a[:-1] =')}).")
    doc.p("The k used for each flux is the node's own k(T): the flux leaving node j uses k_j, and the flux "
          "entering node j+1 from j uses k_(j+1). The same applies axially. The two are equal only when k is "
          "uniform. The net heat into a node is")
    doc.eq(r'N_{ij} = \sum_{nb} G_{ij,nb}\,(T_{nb} - T_{ij}) + Q_{w,i}\,\delta_{j0} - Q_{sink,ij}')
    doc.review(f"The conduction scheme is not conservative when k varies with T. Each node uses its own k for "
               f"both its in- and out-fluxes ({ref('components.py', 'hout_r[:, :-1] =')}, "
               f"{ref('components.py', 'hin_r[:, 1:] =')}), so the heat leaving one node differs from the heat "
               "entering its neighbour. This is latent today, because storage_tcx is a constant, but it will "
               "create or destroy energy if the k(T) table is re-enabled. Fix: a face conductivity (mean of the two "
               "nodes, or k at the face T), as _axial_diag already does "
               f"({ref('tinless.py', 'k = np.asarray(storage_tcx(0.5')}). Confidence high.")
    doc.h3('Explicit sub-stepping and stability')
    doc.p("Conduction is integrated with forward Euler in n_sub sub-steps of dt_s = dt / n_sub, with all "
          "properties evaluated at the start of each sub-step:")
    doc.eq(r'T^{m+1}_{ij} = T^{m}_{ij} + \frac{\Delta t_s}{C_{ij}(T^m)}\;N_{ij}(T^m)',
           where=ref('components.py', 'T = T + net * sub_dt / C'))
    doc.p("n_sub is raised (never lowered) from the YAML sub_steps so that every node satisfies the explicit "
          "positivity limit. The limit is evaluated once per call, at the start-of-step properties, with the "
          "radiation coupling excluded because it is implicit (below):")
    doc.eq(r'n_{sub} = \max\left(n_{sub,YAML},\;\left\lceil \frac{\Delta t}{s\,\Delta t_{crit}} \right\rceil\right),'
           r'\qquad \Delta t_{crit} = \min_{ij}\,\frac{C_{ij}}{\sum_{nb} G_{ij,nb}}',
           where=f"s = safety = {safety} ({ref('components.py', 'n_sub = max(n_sub')}); the count used is stored in "
                 "last_n_sub.")
    doc.table(['', 'stick cell', 'heater cell'], [
        ['dr / dz', f"{dr * 1e3:.1f} mm / {st.axial_dz().min():.3f} m",
         f"{(hm.r_out - hm.r_in) / nR * 1e3:.1f} mm / {hm.storage.axial_dz().min():.3f} m"],
        ['alpha = k / (rho c_p)', f"{alpha:.3g} m2/s", f"{alpha:.3g} m2/s"],
        ['dt_crit', f"{dtc_s:.0f} s", f"{dtc_h:.0f} s"],
        ['n_sub used', ', '.join(f"{ns[d]} at {d} s" for d in ns), f"{ns_h} at {cfg['dt_short']} s"],
        ['radial diffusion time (r_out - r_in)^2 / alpha', f"{(m.r_out - m.r_in) ** 2 / alpha / 3600:.1f} h",
         f"{(hm.r_out - hm.r_in) ** 2 / alpha / 3600:.1f} h"],
        ['axial diffusion time H^2 / alpha', f"{m.H_block ** 2 / alpha / 86400:.0f} days", ''],
    ], widths=[2.6, 2.4, 2.1], size=8.5)
    doc.p("Axial conduction is very slow compared with a discharge. Axial redistribution of heat in the block comes "
          "almost entirely from radiation across the cavity (section 4.2), not from conduction through the "
          "graphite.")
    rows += [['block conduction', 'time integration', 'forward Euler (explicit), properties at sub-step start',
              ref('components.py', 'T = T + net * sub_dt / C')],
             ['block conduction', 'minimum sub-steps', f"{cfg['sub_steps']} (YAML sub_steps)", ref('build.py', "sub_steps=cfg['sub_steps']")],
             ['block conduction', 'stability safety factor', f"{safety}; dt_crit = min C/sum G (stick {dtc_s:.0f} s)",
              ref('components.py', 'dt_crit = float(np.min')],
             ['block conduction', 'sub-steps used (base case)', ', '.join(f"{ns[d]} @ {d} s" for d in ns),
              ref('components.py', 'n_sub = max(n_sub')]]

    doc.h3('Implicit inner-wall radiation (discharge)')
    doc.p("The ring-to-ring radiation exchange in the cavity is equivalent to an effective conductivity of order "
          "1e3 W/m-K, far too stiff for an explicit wall flux. The radiation model therefore returns, at the "
          "start-of-step wall temperatures T_w0, both the heat leaving each block ring Q_rad (n_ax values) and its "
          "sensitivity D = dQ_in/dT_w (an n_ax x n_ax matrix, obtained from the Newton system, section 4.2). In "
          "every sub-step, the wall column (j = 0) is solved with the wall input linearised and treated "
          "implicitly, while conduction stays explicit (IMEX):")
    doc.eq(r'\left(\frac{C_w}{\Delta t_s}\,I - D\right)\,T_w^{m+1} = \frac{C_w}{\Delta t_s}\,T_w^m + N^{cond}_w + Q_0 '
           r'- D\,T_{w0},\qquad Q_w = Q_0 + D\,(T_w^{m+1} - T_{w0})',
           where=f"Q_0 = -Q_rad (into the block), D = -dQ_rad/dT_w ({ref('tinless.py', 'dQ_dTw=-_ax.last')}); "
                 f"solve: {ref('components.py', 'Tw_new = np.linalg.solve')}. Q_w is added to the wall node's net, "
                 "so the explicit formula reproduces T_w^(m+1).")
    doc.p("The radiation solution, the vane position and the sink are evaluated once per macro time step "
          f"(dt = {cfg['dt_short']}-{cfg['dt_long']} s, doubled in optimizer screening) and held or linearised over "
          "its sub-steps. In the charge, the heater flux per ring is computed once per step at the start-of-step "
          "wall temperatures and held constant (explicit; section 4.6).")
    rows += [['block conduction', 'wall radiation coupling', 'linearised, backward Euler per sub-step (IMEX); '
              'radiation evaluated once per macro step', ref('components.py', 'Tw_new = np.linalg.solve')]]

    doc.h3('Boundary conditions')
    doc.table(['boundary', 'condition', 'where'], [
        ['inner wall r = r_in', 'Heat input Q_w,i [W] per ring, added to node (i, 0). The radiating surface '
         'temperature handed to the radiation / heater models is the node-centre temperature T[i, 0].',
         ref('components.py', 'hin_r[:, 0] = 0.0 if implicit else Q_rad')],
        ['outer wall r = r_out', 'Adiabatic (symmetry line between cells). Optional loss through insulation (off, '
         'see below) or a prescribed per-ring sink on the outer node.', ref('components.py', 'Q_node[:, -1] = Q_out')],
        ['top z = H and bottom z = 0', 'Adiabatic, except for the prescribed per-node sink Q_sink: roof and floor '
         'insulation losses q(T)·A_ring·f_open on every node of the top and bottom rings (tank.CellLossSink, section '
         '5), evaluated at the start of each step.', ref('tank.py', 'def for_storage')],
        ['after stick removal', 'Q_rad = 0 for every ring, with no implicit coupling. Conduction only.',
         ref('tinless.py', 'Q_rad   = np.zeros(n_ax)')],
    ], widths=[1.4, 4.3, 1.4], size=8.5)
    fac_s = m.r_in * np.log((m.r_in + dr / 2) / m.r_in) / k0
    drh = (hm.r_out - hm.r_in) / nR
    fac_h = hm.r_in * np.log((hm.r_in + drh / 2) / hm.r_in) / k0
    q_hmax = hm.heater.max_power / (2 * np.pi * hm.r_in * hm.H)
    ch_hr = cfg['heater'].get('charge_hr') or 30
    q_h30 = m.E_req / (ch_hr * 3600) / (2 * np.pi * hm.r_in * hm.H)
    doc.p("There is no conduction resistance between the wall surface and the first node centre, half a ring "
          "(dr/2) inside the graphite. For a quasi-steady wall flux q_s the surface differs from the node by about")
    doc.eq(r'\Delta T_s \approx \frac{q_s\,r_{in}}{k}\,\ln\frac{r_{in} + \Delta r/2}{r_{in}}',
           where=f"stick cell: {fac_s * 1e3:.2f} K per kW/m2; heater cell: {fac_h * 1e3:.2f} K per kW/m2 "
                 f"(k = {k0:g} W/m-K)")
    doc.review(f"Inner-wall half cell missing: the radiation and heater models see the node-centre temperature "
               f"(dr/2 = {dr / 2 * 1e3:.0f} mm into the graphite for the stick cell, {drh / 2 * 1e3:.0f} mm for the "
               f"heater cell) as the wall surface temperature ({ref('components.py', 'hin_r[:, 0] = 0.0 if implicit else Q_rad')}; "
               f"{ref('tinless.py', 'T_h, Q_segs = _solve_T_heater(T[:, 0])')}). Estimated surface-node error: "
               f"{fac_h * q_hmax:.0f} K in the heater cell at the YAML heater cap ({q_hmax / 1e3:.0f} kW/m2) and "
               f"{fac_h * q_h30:.0f} K at the power of a {ch_hr:g} h charge ({q_h30 / 1e3:.1f} kW/m2); about "
               f"{fac_s * 5e3:.0f}-{fac_s * 15e3:.0f} K in discharge (5-15 kW/m2). In the charge this means "
               "the heater temperature is under-predicted while power-limited, the heat transfer is "
               "over-predicted while temperature-limited, and the peak graphite surface temperature is "
               "under-reported. Confidence high; first order in dr, so a num_rad_segments study would show it. "
               "A surface node or half-cell resistance would fix it.")
    doc.p("Outer-wall loss when block_heat_loss_bool is true (off in this configuration), "
          f"{ref('components.py', 'R_conv = 1 / (100 * np.pi')}:")
    doc.eq(r'Q_{out,i} = \frac{T_{i,n_R-1} - T_{sur}}{R_{ins} + R_{conv}},\quad R_{ins} = \frac{\ln(D_{ins}/2r_{c,n_R-1})}'
           r'{2\pi\,k_{ins}(\bar{T})\,\Delta z_i},\quad R_{conv} = \frac{1}{h\,\pi D_{ins}\,\Delta z_i}',
           where='D_ins = 2 (r_out + th_ins), h = 100 W/m2K (hard-coded), k_ins = ins_tcx at the mean of T_sur and '
                 'the node T. The resistance starts at the outer node centre.')
    doc.review(f"The outer-wall loss option hard-codes h = 100 W/m2K and treats the outer graphite half-cell as "
               f"insulation ({ref('components.py', 'r_c_last = r_lo[-1] + 0.5 * dr')}). It is inactive "
               "(block_heat_loss_bool False); flagged only in case it is switched on. Low priority.")
    doc.review(f"After the stick is removed, run_simulation sets Q_rad = 0 for every ring "
               f"({ref('tinless.py', 'Q_rad   = np.zeros(n_ax)')}), which also switches off ring-to-ring radiation "
               "inside the cavity. That radiation is the main axial redistribution mechanism (axial conduction "
               "takes months). This only affects the post-cutoff tail; the optimizer stops at the cutoff. Medium "
               "confidence.")
    doc.h3('Material properties in the conduction')
    doc.p(f"k = storage_tcx(T), c_p = storage_cp(T), rho = storage_rho(T) (section 7.1), currently the constants "
          f"{k0:g} W/m-K, {cp0:g} J/kg-K and {rho0:g} kg/m3. They are evaluated per node through _vprop "
          f"({ref('components.py', 'def _vprop')}), which broadcasts scalar results and falls back to element-wise "
          "evaluation for scalar-only functions. With constant rho, mass is conserved exactly; with rho(T) on a "
          "fixed geometry, it would not be.")
    doc.h3('Energy tally, averages and initial conditions')
    doc.eq(r'E_{step} = \sum_{subs}\sum_{ij} N_{ij}\,\Delta t_s,\qquad '
           r'\bar{T} = \frac{\sum_{ij}(r_{hi,j}^2 - r_{lo,j}^2)\,\Delta z_i\,T_{ij}}{\sum_{ij}(r_{hi,j}^2 - r_{lo,j}^2)\,\Delta z_i}',
           where=f"E_step: {ref('components.py', 'int_energy_tot += float')}; volume average used for the stick cutoff "
                 f"and the charge end: {ref('tinless.py', '_W = np.diff(_r ** 2)')}")
    doc.p("update_temps returns the energy added in the step (including the wall input and the sink) and the "
          "outer-wall loss, and stores the mean applied wall input in last_Q_wall. Both transient drivers discard "
          "these values: the results keep the start-of-step radiation output Q_rad (discharge) or the explicit "
          "heater input (charge), plus the applied sink. T_inner and T_outer are the dz-weighted means of the "
          "first and last radial node columns (node centres, not surfaces). Initial conditions: discharge uniform "
          f"at T_max = {m.T_max:.0f} K; charge uniform at T_min = {m.T_min:.0f} K (section 3.7).")
    doc.review(f"The discharge records Q_rad evaluated at start-of-step wall temperatures "
               f"({ref('tinless.py', 'Q_arr.append(Q_rad.sum())')}), but the block actually loses the "
               f"linearised implicit flux, whose mean is stored in last_Q_wall "
               f"({ref('components.py', 'self.last_Q_wall =')}) and never read. The update's own energy total is "
               "also discarded. With 1800-3600 s steps, the integral of Q_rad over time can drift from the block "
               "enthalpy change. Check that the cycle ledger uses the temperature field; add a closure check "
               "(sum of rho c_p V dT versus the integral of Q_rad + Q_sink). Confidence high that the two differ; "
               "magnitude not measured.")
    doc.p("Other conduction routines in components.py are not called: _update_temps_reference (the original "
          "loop version with uniform dz, kept to verify the vectorised one), update_temps_matrix (an unfinished "
          "implicit attempt: the axial off-diagonals have no matching diagonal terms) and radial_conduction "
          "(references a non-existent attribute).")
    fig = _update_flowchart()
    doc.fig(fig, 'One call of StorageModule.update_temps (one transient time step).', width=4.6, name='Figure 4.1')
    rows += [['block conduction', 'outer-wall h (option, off)', '100 W/m2K', ref('components.py', 'R_conv = 1 / (100 * np.pi')],
             ['block conduction', 'conductance floor in dt_crit', '1e-300', ref('components.py', 'np.maximum(G, 1e-300)')]]
    return rows


# --------------------------------------------------------------------------
# figures for sections 3 / 4.1
# --------------------------------------------------------------------------
def _build_flowchart():
    from .docs_base import flowchart
    d = 1.4
    y = [9.0 - k * d for k in range(7)]
    return flowchart({
        'y': (3.4, y[0], 'YAML dict + dotted overrides\n(load_config, _apply_overrides)', 'io'),
        'c': (3.4, y[1], 'Cavity and vanes around the liner: l = s_l + sqrt2 c,\nS = 4l + 2c, r_in = 2S/pi; '
                         'octagon r_sh; legacy shade checks', 'proc'),
        'e': (3.4, y[2], 'Energy: f_cell, f_Q (incident_energy at T_min - 100 K)\nE_req = q A f_cell t / f_Q', 'proc'),
        'v': (3.4, y[3], 'Block: V = phi_V E_req / (rho c_p dT)\nr_out = sqrt(V / (pi H) + r_in^2)', 'proc'),
        'm': (3.4, y[4], 'Axial mesh: n_low + n_act rings,\nedge at H - L; H_low = H - L - t_lb - t_RGF', 'proc'),
        'k': (3.4, y[5], 'Components: GraphiteBlock (uniform T_max), StorageModule,\nPowerStick, TinlessSimulation, '
                         'VaneHybridRadiation', 'proc'),
        'o': (3.4, y[6], 'Model namespace m: sim, run_kwargs,\nmesh, r_in, r_out, E_req, V_block, ...', 'out'),
        'h': (6.55, y[6], 'build_heater(m):\nsame V, H, mesh;\nr_hc = AR_h r_h;\nuniform T_min', 'proc'),
        'x': (0.65, y[4], 'errors:\nH < L,\nH_low < 0,\nH > L w/o\naxial', 'dec'),
    }, [('y', 'c'), ('c', 'e'), ('e', 'v'), ('v', 'm'), ('m', 'k'), ('k', 'o'), ('o', 'h'), ('m', 'x')],
        size=(7.4, 9.8))


def _update_flowchart():
    from .docs_base import flowchart
    d = 1.4
    y = [9.0 - k * d for k in range(7)]
    return flowchart({
        'i': (3.2, y[0], 'T (nA x nR), Q_w (+ dQ/dT_w), dt, Q_sink', 'io'),
        'n': (3.2, y[1], 'dt_crit = min C / sum G at start-of-step k, rho c_p\nn_sub = max(sub_steps, '
                         'ceil(dt / (0.9 dt_crit)))', 'proc'),
        'l': (3.2, y[2], 'sub-step: k(T), radial + axial fluxes\n(each node with its own k), net N', 'loop'),
        's': (3.2, y[3], 'N -= Q_sink (top / bottom rings)', 'proc'),
        'w': (3.2, y[4], 'wall column: (C/dt_s - D) T_w = C/dt_s T_w + N_w + Q_0 - D T_w0\n'
                         '(discharge) or N_w += Q_w (charge)', 'proc'),
        'u': (3.2, y[5], 'T += N dt_s / C ;  E += sum N dt_s', 'proc'),
        'o': (3.2, y[6], 'T_new, E (discarded by callers), last_Q_wall', 'out'),
        'rp': (6.4, y[3], 'repeat\nn_sub\ntimes', 'loop'),
    }, [('i', 'n'), ('n', 'l'), ('l', 's'), ('s', 'w'), ('w', 'u'), ('u', 'o'),
        ('u', 'rp'), ('rp', 'l')], size=(7.0, 9.8))


def _plan_figure(m):
    try:
        from .correlations import _vane_boxes, _vane_box_corners
    except Exception:
        return None
    plt = _plt()
    from matplotlib.patches import Rectangle, Circle, Polygon
    lay = m.vane_lay
    if not lay:
        return None
    A = lay['A']
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.add_patch(Circle((0, 0), m.r_out, fc='#EDE7DD', ec='#888888', ls='--', lw=1))
    ax.add_patch(Rectangle((-A, -A), 2 * A, 2 * A, fc='white', ec='#444444', lw=1.4))
    sl = m.liner_side_len / 2
    tl = max(m.liner_thickness, 0.003)
    ax.add_patch(Rectangle((-sl, -sl), 2 * sl, 2 * sl, fc='#555555', ec='#222222', lw=1))
    ax.add_patch(Rectangle((-sl + tl, -sl + tl), 2 * (sl - tl), 2 * (sl - tl), fc='white', ec='none'))
    sp = m.powerstick.width / 2
    ax.add_patch(Rectangle((-sp, -sp), 2 * sp, 2 * sp, fc='#3B6FB6', ec='#1F3F7A'))
    for th, ls, a in ((0, '-', 1.0), (90, '--', 0.55)):
        for bx in _vane_boxes(lay, th):
            P = _vane_box_corners(bx, lay['t'])
            ax.add_patch(Polygon([P[0], P[1], P[3], P[2]], closed=True, fc='#7FB77E' if th == 0 else 'none',
                                 ec='#2E7D32', ls=ls, lw=1, alpha=a))
    for r, lab in ((m.r_in, 'r_in'), (m.r_liner_eff, 'r_l'), (m.r_ps_eff, 'r_ps'),
                   (2 * m.shade_side_len / np.pi, 'r_sh')):
        ax.add_patch(Circle((0, 0), r, fc='none', ec='#B5520A', ls=':', lw=1))
    ax.annotate('', xy=(m.r_out, 0), xytext=(0, 0), arrowprops=dict(arrowstyle='->', color='#666666', lw=0.8))
    ax.text(m.r_out * 0.72, 0.02, f"r_out = {m.r_out:.3f} m", fontsize=7.5, color='#444444')
    ax.text(-A + 0.02, A - 0.06, f"cavity S = {2 * A:.3f} m", fontsize=7.5)
    ax.text(0.03, -m.r_in - 0.06, f"r_in = {m.r_in:.3f} m (dotted circles: r_ps, r_l, r_sh, r_in)", fontsize=7,
            color='#B5520A', ha='center')
    R = m.r_out * 1.04
    ax.set_xlim(-R, R)
    ax.set_ylim(-R, R)
    ax.set_aspect('equal')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.tick_params(labelsize=7)
    return fig


def _rz_figure(m, hm):
    plt = _plt()
    from matplotlib.patches import Rectangle
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 6.0), gridspec_kw=dict(width_ratios=[1.1, 1]))
    me = m.mesh
    H = m.H_block
    for ax, (ri, ro, edges, title) in zip(axs, ((m.r_in, m.r_out, m.storage.axial_dz(), 'stick cell'),
                                                 (hm.r_in, hm.r_out, hm.storage.axial_dz(), 'heater cell'))):
        nR = m.storage.num_rad_segments
        z = np.concatenate([[0], np.cumsum(edges)])
        r = ri + np.arange(nR + 1) * (ro - ri) / nR
        ax.add_patch(Rectangle((ri, 0), ro - ri, H, fc='#EDE7DD', ec='none'))
        for zz in z:
            ax.plot([ri, ro], [zz, zz], color='#999999', lw=0.5)
        for rr in r:
            ax.plot([rr, rr], [0, H], color='#999999', lw=0.5)
        rc, zc = 0.5 * (r[1:] + r[:-1]), 0.5 * (z[1:] + z[:-1])
        RC, ZC = np.meshgrid(rc, zc)
        ax.plot(RC.ravel(), ZC.ravel(), '.', ms=2.5, color='#555555')
        ax.set_title(title, fontsize=9)
        ax.set_xlim(0, ro * 1.03)
        ax.set_ylim(-0.2, H + 0.2)
        ax.set_xlabel('r [m]', fontsize=8)
        ax.tick_params(labelsize=7)
    ax = axs[0]
    z0 = me.z_act0
    ax.add_patch(Rectangle((0, z0), m.r_ps_eff, m.L, fc='#3B6FB6', ec='none'))
    ax.plot([m.r_liner_eff] * 2, [z0, H], color='k', lw=1.6)
    ax.add_patch(Rectangle((0, z0 - me.t_lb), m.r_liner_eff, me.t_lb, fc='k', ec='none'))
    ax.add_patch(Rectangle((0, me.H_low), m.r_liner_eff, me.t_rgf, fc='#E39B3A', ec='none'))
    if m.shade_bool:
        rsh = 2 * m.shade_side_len / np.pi
        ax.plot([rsh] * 2, [0, H], color='#2E7D32', ls='--', lw=1.2)
    ax.axhline(z0, color='#B5520A', ls=':', lw=0.8)
    ax.text(0.02, z0 - 0.35, f"z = H - L = {z0:.2f} m\n(liner bottom)", fontsize=7, color='#B5520A')
    ax.text(m.r_liner_eff + 0.02, z0 + m.L * 0.75, 'stick (blue)\n+ liner', fontsize=7)
    ax.text(m.r_in + 0.01, H * 0.25, f"{me.n_low} lower\nrings", fontsize=7)
    ax.text(m.r_in + 0.01, z0 + m.L * 0.5, f"{me.n_act} active\nrings", fontsize=7)
    ax.set_ylabel('z [m] (0 = cavity floor)', fontsize=8)
    ax = axs[1]
    ax.add_patch(Rectangle((0, 0), hm.r_heater, H, fc='#C0392B', ec='none'))
    ax.text(hm.r_heater + 0.02, H * 0.5, 'heater', fontsize=7, color='#C0392B')
    fig.tight_layout()
    return fig


# ==========================================================================
# 4.5 Powerstick and cells
# ==========================================================================
def sec_stick(doc, ctx):
    from .properties import panel_efficiency, _PANEL_EFF_COEFFS
    cfg = ctx['base_cfg']
    m, _ = _cache(ctx)
    p = cfg['powerstick']
    ps = m.powerstick
    tab = m.emissivity_table
    bg = tab[0][0][1]
    f_b, f_a = p['f_bare'], p['f_active']
    f_c, f_i = (1 - f_b) * f_a, (1 - f_b) * (1 - f_a)
    rows = []
    doc.h2('4.5 Powerstick and cells')
    doc.p(f"The powerstick is a square, water-cooled tube of side s_ps = {ps.width * 1e3:.0f} mm and length "
          f"L = {ps.height} m, covered with PV cells. The model splits its surface into three parts with different "
          "spectral absorptance:")
    doc.table(['part', 'area fraction', 'value', 'meaning'], [
        ['bare', 'f_bare', f"{f_b:g}", 'not covered by the cell assembly (SNGC); absorbs heat only'],
        ['active cells', '(1 - f_bare) f_active', f"{f_c:.3f}", 'covered and generating'],
        ['inactive', '(1 - f_bare)(1 - f_active)', f"{f_i:.3f}", 'covered but not generating (gaps, busbars); heat only'],
    ], widths=[1.1, 1.7, 0.6, 3.7], size=8.5)
    doc.p(f"The stick is cut into n_act = {ps.num_ax_steps} axial steps, one per active block ring, each of area "
          f"A_step = 4 s_ps dz = {ps.A_step:.3f} m2 ({ref('components.py', 'self.A_step = 4 * self.height_step')}). "
          "In the in-use path the PowerStick object only carries geometry, the area fractions and the emissivity "
          "table. The axial model reads them, and the exchange between liner and stick is computed in "
          f"axial.LinerTable ({ref('axial.py', 'class LinerTable')}; physics in section 4.4).")
    doc.h3('Spectral emissivity table')
    doc.p("emissivity_tables[0].value gives, for each wavelength band, the absorptance (= emissivity) of the "
          "active, bare and inactive parts. The first band edge above the shortest wavelength is taken as the "
          f"cell bandgap wavelength, lambda_g = {bg} um ({ref('axial.py', 'self.band_gap = tab[0][0][1]')}).")
    doc.table(['band [um]', 'eps active', 'eps bare', 'eps inactive', 'area-average eps_ps', 'active share a'],
              [[f"{b[0]:g} - {b[1]:g}", f"{ea:g}", f"{eb:g}", f"{ei:g}",
                f"{f_b * eb + f_c * ea + f_i * ei:.4f}", f"{f_c * ea / (f_b * eb + f_c * ea + f_i * ei):.3f}"]
               for b, ea, eb, ei in tab], widths=[1.1, 0.9, 0.9, 1.0, 1.5, 1.2], size=8.5)
    doc.eq(r'\bar{\varepsilon}_{ps}(\lambda) = f_{bare}\,\varepsilon_{bare} + f_{cell}\,\varepsilon_{a} + '
           r'f_{inact}\,\varepsilon_{in},\qquad a(\lambda) = \frac{f_{cell}\,\varepsilon_a}{\bar{\varepsilon}_{ps}}',
           where=f"{ref('axial.py', 'act[m] = (1 - f_bare)')}; a is the share of the stick's absorbed power that the "
                 "active cells absorb at that wavelength.")
    doc.p("Spectral integrals are taken with 10 Gauss-Legendre nodes per band in log(lambda) "
          f"({ref('tinless.py', 'def _spectral_grid')}). Band edges, including the bandgap, are never straddled, so the "
          "piecewise-constant emissivities are integrated exactly. Radiation outside the table range "
          f"({tab[0][0][0]:g}-{tab[-1][0][1]:g} um) is not counted.")
    fig = _emis_table_figure(m)
    doc.fig(fig, 'Stick emissivity table (area fractions applied) with the blackbody spectrum at the block T_max and '
                 'T_min for scale.', name='Figure 4.5')
    doc.review(f"The spectral integration covers only {tab[0][0][0]:g}-{tab[-1][0][1]:g} um. About "
               f"{_planck_tail(tab[-1][0][1], m.T_max) * 100:.1f} % of blackbody power at T_max lies beyond "
               f"{tab[-1][0][1]:g} um and is dropped from the liner-to-stick exchange. Low priority; it goes into waste heat.")
    doc.h3('Cell power')
    doc.p("For each active ring k, the liner model (section 4.4) gives the heat absorbed by the stick Q_ps,k and the "
          "part absorbed by active cells Q_cell,k:")
    doc.eq(r'Q_{cell,k} = A_{step}\sum_n a(\lambda_n)\,\frac{E_{b\lambda}(T_{Li,k},\lambda_n)\,w_n}{R_n},\qquad '
           r'R_n = \frac{r_{ps}}{r_l}\,\frac{1-\varepsilon_{Li}}{\varepsilon_{Li}} + 1 + '
           r'\frac{1-\bar{\varepsilon}_{ps}}{\bar{\varepsilon}_{ps}}')
    doc.eq(r'q_{cell,k} = \frac{Q_{cell,k}}{10^4\,A_{step}\,f_{cell}}\;\;[\mathrm{W/cm^2}]',
           where=f"T_Li: liner inner-face temperature; w_n: quadrature weights; the stick's own emission at its "
                 f"cooling temperature is neglected. {ref('axial.py', 'denom = (r_ps / r_l)')}, "
                 f"{ref('axial.py', 'self.A_cell_cm2_seg =')}")
    doc.p("Electrical output, using the panel-efficiency fit (section 7.5) with the liner inner face as the "
          "emitter:")
    doc.eq(r'\eta_0 = a\,T_{Li}^2 + b\,T_{Li} + c,\qquad \eta = \eta_0\;0.998^{\,2\,q_{cell}\,(1-\eta_0)}')
    doc.eq(r'P_{el,k} = (1 - D_c)\,\eta\,Q_{cell,k},\qquad Q_{waste,k} = Q_{ps,k} - P_{el,k}',
           where=f"{ref('axial.py', 'eff = panel_efficiency(self.liner.band_gap')}; D_c = cell degradation "
                 f"(sim.cell_eff_factor = 1 - D_c, {ref('build.py', 'sim.cell_eff_factor = 1.0 - D_deg')}).")
    a_, b_, c_ = _PANEL_EFF_COEFFS[bg] if bg in _PANEL_EFF_COEFFS else (np.nan,) * 3
    Tpk = -b_ / (2 * a_)
    qset = cfg['storage']['avg_flux']
    doc.p(f"For lambda_g = {bg} um, eta_0 peaks at {panel_efficiency(bg, Tpk):.3f} at T = {Tpk:.0f} K. At the "
          f"target flux of {qset} W/cm2, the flux factor is {0.998 ** (2 * qset * (1 - panel_efficiency(bg, Tpk))):.3f}. "
          "Q_cell includes the out-of-band light absorbed by active cells (eps_active = "
          f"{tab[-1][1]:g} beyond lambda_g). The efficiency multiplies that whole absorbed power. All of the stick's "
          "absorbed heat other than P_el is waste heat to the cooling water.")
    doc.review(f"panel_efficiency: the docstrings ({ref('properties.py', 'η = η_0(T) · 0.998^(2 · flux · η_0)')}, "
               f"{ref('components.py', 'η = η_0 · 0.998^(2 · flux · η_0)')}) say 0.998^(2 flux eta_0), but the code "
               f"uses (1 - eta_0) ({ref('properties.py', 'return eta_0 * (0.998')}). Confirm which is intended. The "
               f"ratio of the two is 0.998^(2 q (1 - 2 eta_0)): {_dfac(bg, m.T_min, qset):.2f} % at T_min and "
               f"{_dfac(bg, m.T_max, qset):.2f} % at T_max for q = {qset} W/cm2, so the numerical effect is small "
               "today but grows with flux and with lower eta_0. Confidence high that they disagree.")
    doc.review("The efficiency fit depends only on the emitter temperature. It ignores the cell temperature "
               "(powerstick.cooling_temp is not used) and the spectrum actually reaching the cells (vanes, liner "
               "emissivity), and it multiplies Q_cell including out-of-band absorption. The basis of the fit "
               "(electric power per absorbed power or per in-band power; which cell, which emitter) is not "
               "documented in the code. It is not clamped and goes negative below about "
               f"{_eta_root(a_, b_, c_)[0]:.0f} K or above {_eta_root(a_, b_, c_)[1]:.0f} K. Medium confidence "
               "that this matters; it sets every electrical output in the model.")
    doc.h3('Cooling temperature')
    doc.p(f"powerstick.cooling_temp = {p['cooling_temp']} K is only used as the default temperature of the bottom cap "
          f"under the liner bottom (section 4.4, {ref('build.py', 'T_cap=float(T_ps if T_cap')}). Stick emission is "
          "neglected (the stick is a cold absorber), and the cell temperature does not enter the efficiency.")
    doc.h3('Degradation')
    doc.p(f"For lifetime studies, cfg['degradation'] = {{cell: D_c, bare: D_b, emis_end}} "
          "(" + ref('build.py', "deg = cfg.get('degradation')") + ") changes the run's table:")
    doc.eq(r'\varepsilon_{bare} \rightarrow \varepsilon_{bare} + D_b\,(\varepsilon_{end} - \varepsilon_{bare}),\qquad '
           r'\varepsilon_{in} \rightarrow \varepsilon_{in} + D_c\,(\varepsilon_{end} - \varepsilon_{in}),\qquad '
           r'\eta \rightarrow (1-D_c)\,\eta',
           where='eps_end = 0.8 default; the active-cell absorptance is unchanged; sizing (f_Q, E_req) always uses '
                 'the fresh table, so geometry is identical. How D is obtained from deposition: sections 4.7 and 9.4.')
    doc.h3('Legacy cell routines')
    doc.p(f"PowerStick.generate_electricity_step ({ref('components.py', 'def generate_electricity_step')}) belongs to "
          "the tin-loop model. It assumes exactly two bands, computes an unused photon-current estimate with "
          "placeholder constants (V_panel = 1, P_collect = 1) and the same panel_efficiency. "
          "PowerStick.incident_energy is still used once, by the block sizing (f_Q, section 3.4).")
    rows += [['stick / cells', 'spectral quadrature', '10 Gauss-Legendre nodes per band in log(lambda)', ref('tinless.py', "n_gauss_per_band', 10")],
             ['stick / cells', 'wavelength range', f"{tab[0][0][0]:g}-{tab[-1][0][1]:g} um ({len(tab)} bands)", ref('build.py', "raw_emis = cfg['emissivity_tables']")],
             ['stick / cells', 'bandgap', f"{bg} um (must be 1.24 / 1.38 / 1.55)", ref('properties.py', '_PANEL_EFF_COEFFS = {')],
             ['stick / cells', 'flux degradation base', '0.998 per (2 flux (1 - eta_0))', ref('properties.py', 'return eta_0 * (0.998')],
             ['stick / cells', 'liner table grid', 'T_Li 350-3600 K, 2 K steps', ref('axial.py', 'T = np.arange(350.0, 3600.0, 2.0)')],
             ['stick / cells', 'liner wall fixed point', '50 iterations, tol 1e-9 K', ref('axial.py', 'T_new = T + q_out * t / liner_k')],
             ['stick / cells', 'emissivity floor', '1e-9', ref('axial.py', 'eps_in = np.maximum(eps_in, 1e-9)')]]
    return rows


def _planck_tail(lam_um, T):
    """Fraction of blackbody power beyond lam_um at T."""
    from scipy.integrate import quad
    h, c, k = 6.62607015e-34, 2.99792458e8, 1.380649e-23
    x0 = h * c / (lam_um * 1e-6 * k * T)
    val, _ = quad(lambda x: x ** 3 / np.expm1(x), 1e-9, x0)
    return val * 15 / np.pi ** 4


def _dfac(bg, T, q):
    """Percent difference in eta between the docstring (eta_0) and code (1 - eta_0) flux exponents."""
    from .properties import panel_efficiency
    e0 = panel_efficiency(bg, float(T))
    return (1 - 0.998 ** (2 * q * (1 - 2 * e0))) * 100


def _eta_root(a, b, c):
    d = np.sqrt(max(b * b - 4 * a * c, 0))
    r = sorted([(-b + d) / (2 * a), (-b - d) / (2 * a)])
    return r


def _emis_table_figure(m):
    plt = _plt()
    tab = m.emissivity_table
    p = m.cfg['powerstick']
    f_b, f_a = p['f_bare'], p['f_active']
    f_c, f_i = (1 - f_b) * f_a, (1 - f_b) * (1 - f_a)
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    for key, idx, col in (('active', 1, '#2F5F9E'), ('bare', 2, '#B5520A'), ('inactive', 3, '#777777')):
        xs, ys = [], []
        for r in tab:
            xs += [r[0][0], r[0][1]]
            ys += [r[idx], r[idx]]
        ax.step(xs, ys, where='post', color=col, lw=1.6, label=f"eps {key}")
    xs, ys = [], []
    for b, ea, eb, ei in tab:
        xs += [b[0], b[1]]
        e = f_b * eb + f_c * ea + f_i * ei
        ys += [e, e]
    ax.plot(xs, ys, color='k', ls='--', lw=1, label='area average')
    ax.set_xscale('log')
    ax.set_xlabel('wavelength [um]')
    ax.set_ylabel('emissivity / absorptance')
    ax.set_ylim(0, 1.0)
    ax2 = ax.twinx()
    lam = np.logspace(np.log10(tab[0][0][0]), np.log10(tab[-1][0][1]), 400)
    h, c, k = 6.62607015e-34, 2.99792458e8, 1.380649e-23
    for T, ls in ((m.T_max, '-'), (m.T_min, ':')):
        E = 2 * np.pi * h * c ** 2 / (lam * 1e-6) ** 5 / np.expm1(h * c / (lam * 1e-6 * k * T)) * 1e-6
        ax2.plot(lam, E / 1e6, color='#C0392B', ls=ls, lw=0.9, alpha=0.7, label=f"E_b,lambda at {T:.0f} K")
    ax2.set_ylabel('E_b,lambda [MW/m2/um]', color='#C0392B')
    ax2.tick_params(colors='#C0392B', labelsize=7)
    ax.axvline(tab[0][0][1], color='#999999', lw=0.7)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc='upper right')
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    return fig


# ==========================================================================
# 7. Material properties
# ==========================================================================
def sec_properties(doc, ctx):
    from . import properties as P
    plt = _plt()
    m, _ = _cache(ctx)
    rows = []
    T_lo, T_hi = float(m.T_min), float(m.T_max)
    doc.h1('7. Material properties')
    doc.p(f"All material properties are in thermal/properties.py. The table below lists every function and "
          f"constant, its form, and whether the configuration documented here (axial radiation, vanes, graphite "
          f"liner) calls it. Values at the block operating limits T_min = {T_lo:.0f} K and T_max = {T_hi:.0f} K are "
          "evaluated live. 'No source' means the code gives no reference.")
    use = 'IN USE'

    def v(fn, *a):
        try:
            return _f(float(np.asarray(fn(*a)).ravel()[0]))
        except Exception as e:
            return f'({type(e).__name__})'
    inv = [
        ['storage_rho', 'block density', 'constant', f"{v(P.storage_rho, T_lo)} kg/m3", use + ' (block)', ref('properties.py', 'def storage_rho')],
        ['storage_cp', 'block specific heat', 'constant (table unused)', f"{v(P.storage_cp, T_lo)} J/kg-K", use + ' (block)', ref('properties.py', 'def storage_cp')],
        ['storage_tcx', 'block conductivity', 'constant (table unused)', f"{v(P.storage_tcx, T_lo)} W/m-K", use + ' (block)', ref('properties.py', 'def storage_tcx')],
        ['gr_emis', 'graphite emissivity', 'linear in T', f"{v(P.gr_emis, T_lo)} / {v(P.gr_emis, T_hi)}", use + ' (block wall, liner, heater cavity)', ref('properties.py', 'def gr_emis')],
        ['gr_tcx', 'graphite conductivity', 'table, np.interp', f"{v(P.gr_tcx, T_lo)} / {v(P.gr_tcx, T_hi)} W/m-K", use + ' (liner wall, liner-bottom plate)', ref('properties.py', 'def gr_tcx')],
        ['gr_cp', 'graphite c_p (Entegris)', 'polynomial', f"{v(P.gr_cp, T_lo)} / {v(P.gr_cp, T_hi)} J/kg-K", 'not used', ref('properties.py', 'def gr_cp')],
        ['gr_rho', 'graphite density', 'constant', f"{v(P.gr_rho, T_lo)} kg/m3", 'not used', ref('properties.py', 'def gr_rho')],
        ['liner_eps_out / liner_k', 'liner selector', 'grafoil constants, else graphite', '-', use, ref('properties.py', 'def liner_eps_out')],
        ['ins_tcx', 'RGF / carbon felt in argon', 'table (C), np.interp', f"{v(P.ins_tcx, T_lo)} / {v(P.ins_tcx, T_hi)} W/m-K", use + ' (vanes, RGF pad)', ref('properties.py', 'def ins_tcx')],
        ['thermax_tcx (+ integral)', 'Thermax board', 'exponential in T(C)', f"{v(P.thermax_tcx, T_lo)} / {v(P.thermax_tcx, T_hi)} W/m-K", use + ' (tank insulation, sec. 5)', ref('properties.py', 'def thermax_tcx')],
        ['panel_efficiency', 'TPV efficiency', 'quadratic in T_emitter x flux factor', f"{v(lambda T: P.panel_efficiency(1.24, T), T_lo)} / {v(lambda T: P.panel_efficiency(1.24, T), T_hi)} (1.24 um, no flux)", use, ref('properties.py', 'def panel_efficiency')],
        ['tac_emis_temp', 'TaC total emissivity', 'linear', f"{v(P.tac_emis_temp, T_lo)} / {v(P.tac_emis_temp, T_hi)}", 'sizing only (f_Q)', ref('properties.py', 'def tac_emis_temp')],
        ['tac_emis_wavelength_temp', 'TaC spectral', 'linear + 1/lambda', '-', 'not called (inlined for tac liner)', ref('properties.py', 'def tac_emis_wavelength_temp')],
        ['tac_emis_binned_1eV', 'TaC 2-band', 'linear', '-', 'not used', ref('properties.py', 'def tac_emis_binned_1eV')],
        ['zrc_emis_temp', 'ZrC total', 'linear, clipped', f"{v(P.zrc_emis_temp, T_lo)} / {v(P.zrc_emis_temp, T_hi)}", 'not used', ref('properties.py', 'def zrc_emis_temp')],
        ['zrc_emis_wavelength_temp', 'ZrC spectral', 'linear, clipped', '-', 'zrc liner option (inlined)', ref('properties.py', 'def zrc_emis_wavelength_temp')],
        ['wc_emis_grey', '"tungsten" grey', 'constant', v(P.wc_emis_grey, T_lo), 'not used', ref('properties.py', 'def wc_emis_grey')],
        ['w_emis_total', 'W total hemispherical', 'degree-5 polynomial', f"{v(P.w_emis_total, T_lo)} / {v(P.w_emis_total, T_hi)}", 'W-foil options only', ref('properties.py', 'def w_emis_total')],
        ['w_emis_spectral(_vec)', 'W spectral', '2-D table, bilinear', '-', 'W-foil options only', ref('properties.py', 'def w_emis_spectral_vec')],
        ['w_tcx', 'W conductivity', 'piecewise linear', f"{v(P.w_tcx, T_lo)} / {v(P.w_tcx, T_hi)} W/m-K", 'not used', ref('properties.py', 'def w_tcx')],
        ['Sn_rho, Sn_tcx, Sn_mu, Sn_cp, Sn_Pr', 'liquid tin', 'correlations', '-', 'legacy tin model only', ref('properties.py', 'def Sn_rho')],
    ]
    doc.table(['function', 'property', 'form', f"value at {T_lo:.0f} / {T_hi:.0f} K", 'use', 'where'], inv,
              widths=[1.3, 1.15, 1.05, 1.3, 1.15, 1.15], size=7.5)

    # 7.1 storage graphite
    doc.h2('7.1 Storage graphite (block)')
    tC = np.array([300, 750, 1200, 1995, 2400.])
    cp_tab = np.array([1.27, 1.92, 1.99, 2.13, 2.13]) * 1e3
    k_tab = np.array([31.5, 28.7, 24.8, 23.1, 23.1 + (23.1 - 24.8) / (1995 - 1200) * (2400 - 1995)])
    Tavg_C = (T_lo + T_hi) / 2 - C2K
    doc.p(f"The block uses three constants: rho = {_f(P.storage_rho(T_lo))} kg/m3, c_p = {_f(P.storage_cp(T_lo))} "
          f"J/kg-K and k = {_f(P.storage_tcx(T_lo))} W/m-K ({ref('properties.py', 'def storage_rho')}, "
          f"{ref('properties.py', 'def storage_cp')}, {ref('properties.py', 'def storage_tcx')}). The c_p and k "
          "functions build an interpolation table and then return the constant without using it. The comment "
          "says c_p was 'replaced with fixed cp until better enthalpy handling implemented'. An alternative density "
          "of 1743 kg/m3 is commented out. No source is given for any of them. The unused tables are:")
    doc.table(['T [C]'] + [f"{t:.0f}" for t in tC], [
        ['c_p [J/kg-K]'] + [f"{x:.0f}" for x in cp_tab],
        ['k [W/m-K]'] + [f"{x:.2f}" for x in k_tab],
    ], widths=[1.3] + [1.1] * 5, size=8.5)
    doc.p(f"The 2400 C entries are extrapolated: c_p held flat, k continued with the 1200-1995 C slope. At the "
          f"operating mean ({Tavg_C:.0f} C) the tables give c_p = {np.interp(Tavg_C, tC, cp_tab):.0f} J/kg-K and "
          f"k = {np.interp(Tavg_C, tC, k_tab):.1f} W/m-K. For comparison, the YAML column graphite is "
          f"{_f(ctx['base_cfg'].get('tank', {}).get('columns', {}).get('rho_graphite', np.nan))} kg/m3.")
    doc.review(f"Block properties are unsourced constants that disagree with the tables in the same functions: "
               f"rho = {_f(P.storage_rho(T_lo))} kg/m3 (1743 commented out), c_p = {_f(P.storage_cp(T_lo))} "
               f"(table {np.interp(Tavg_C, tC, cp_tab):.0f} at {Tavg_C:.0f} C, "
               f"{(np.interp(Tavg_C, tC, cp_tab) / P.storage_cp(T_lo) - 1) * 100:+.0f} %), k = {_f(P.storage_tcx(T_lo))} "
               f"(table {np.interp(Tavg_C, tC, k_tab):.1f}). rho c_p sets the block volume and graphite cost "
               f"linearly (1450 vs 1743 kg/m3 is a 20 % volume difference) "
               f"({ref('build.py', 'e_dens = storage_rho')}). k = 20 is conservative for discharge. A graphite grade "
               "and source should be chosen. Confidence high.")

    # 7.2 graphite
    doc.h2('7.2 Graphite (cavity surfaces, liner, liner-bottom plate)')
    doc.eq(r'\varepsilon_{gr}(T) = 0.679 + 6\times10^{-5}\,T\;\;[T\ \mathrm{in\ K}]',
           where=f"gr_emis, {ref('properties.py', 'def gr_emis')}; no source, no clamp (reaches 1 at 5350 K); used for the "
                 "block wall, liner faces (graphite liner), liner-bottom plate and heater-cavity wall.")
    Tg = np.array([20, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400.])
    doc.p(f"gr_tcx ({ref('properties.py', 'def gr_tcx')}) interpolates the table below with np.interp and holds the "
          "end values flat outside it. It is used for the graphite liner wall and the liner-bottom plate.")
    doc.table(['T (as coded)'] + [f"{t:.0f}" for t in Tg], [['k [W/m-K]'] + [f"{float(P.gr_tcx(t)):.0f}" for t in Tg]],
              widths=[1.0] + [0.47] * 13, size=7.5)
    doc.review(f"gr_tcx's docstring says T is in K, but the table starts at 20 (room temperature in C?). If the "
               f"table is in C, the liner conductivity at T_max is about "
               f"{float(P.gr_tcx(T_hi - C2K)):.1f} rather than {float(P.gr_tcx(T_hi)):.1f} W/m-K. Small effect "
               f"(liner wall dT only). gr_emis has no source or validity range ({ref('properties.py', 'def gr_tcx')}). "
               "Also note the liner graphite (k about 34-40) and the block graphite (k = 20) are different "
               "materials in the model. Medium confidence.")
    doc.eq(r'c_{p,gr} = 4184\,\left(0.44391 + 3.0795\times10^{-5}\,T - \frac{6.1257\times10^{4}}{T^2} + '
           r'\frac{1.0795\times10^{7}}{T^3}\right)\;\;[\mathrm{J/kg\,K}]',
           where=f"gr_cp 'from Entegris' ({ref('properties.py', 'def gr_cp')}), not used by the tinless path; "
                 f"gr_rho = {_f(P.gr_rho(T_lo))} kg/m3 (comment 'fix'), not used.")

    # 7.3 liner selector
    doc.h2('7.3 Liner material selector')
    doc.p(f"liner_eps_out(material, T) and liner_k(material, T) ({ref('properties.py', 'def liner_eps_out')}) return "
          f"constants for the materials in LINER_GRAY = {P.LINER_GRAY} ({ref('properties.py', 'LINER_GRAY = {')}), "
          "and graphite values (gr_emis, gr_tcx) for every other material. The TaC and ZrC liners therefore have a "
          "graphite outer face and graphite wall conductivity; only their inner-face spectral emissivity differs "
          "(7.6). For the graphite liner used here, both faces have emissivity gr_emis(T) and k = gr_tcx(T).")

    # 7.4 insulation
    doc.h2('7.4 Insulation: RGF (carbon felt) and Thermax')
    tI = np.array([0, 400, 800, 1000, 2000, 2500.])
    doc.p(f"ins_tcx ({ref('properties.py', 'def ins_tcx')}), 'carbon insulation thermal conductivity in argon', "
          "converts T to C and interpolates this table, holding the end values flat outside it. No source is given.")
    doc.table(['T [C]'] + [f"{t:.0f}" for t in tI], [['k [W/m-K]'] + [f"{float(P.ins_tcx(t + C2K)):.3f}" for t in tI]],
              widths=[1.3] + [0.95] * 6, size=8.5)
    doc.p("The 2500 C point is an extrapolation of the 1000-2000 C slope (0.94 + 0.59 x 0.5). The RGF vanes and pad "
          "use the exact Kirchhoff integral I(T) = integral of k dT, tabulated from 150 to 6000 K in 0.25 K steps "
          f"({ref('axial.py', 'def __init__(self, k_fn=ins_tcx, T_lo=150.0')}), so steady 1-D conduction through a "
          "layer of thickness t is q = (I(T1) - I(T2)) / t exactly.")
    doc.eq(r'k_{Thermax} = %.5f\;\exp(%.4f\;T_C)\;\;[\mathrm{W/m\,K}],\qquad \int k\,dT = \frac{%.5f}{%.4f}\,'
           r'\left(e^{%.4f\,T_{C,hi}} - e^{%.4f\,T_{C,lo}}\right)'
           % (P.THERMAX_K0, P.THERMAX_B, P.THERMAX_K0, P.THERMAX_B, P.THERMAX_B, P.THERMAX_B),
           where=f"thermax_tcx, 'measured', 640 kg/m3 ({ref('properties.py', 'THERMAX_K0 =')}); range of the "
                 "measurement not stated; used by the tank model (section 5).")
    doc.review(f"The RGF conductivity ins_tcx is unsourced, and above 2000 C it is a linear extrapolation that is "
               f"then held flat above 2500 C ({ref('properties.py', 'def ins_tcx')}). The vanes run on their hot "
               "face at about the block temperature (1600-2100 C), and vane conduction sets the closed-shade "
               "leakage and the pad loss. The thermax_tcx fit is unbounded with T and has no stated validity "
               "range. Medium confidence that the results are sensitive to this.")

    # 7.5 panel efficiency
    doc.h2('7.5 TPV panel efficiency')
    doc.p(f"panel_efficiency(bandgap, T_emitter, flux) ({ref('properties.py', 'def panel_efficiency')}) is a quadratic "
          "fit in the emitter temperature for three cell bandgap wavelengths, with an optional flux factor:")
    doc.eq(r'\eta_0 = a\,T^2 + b\,T + c,\qquad \eta = \eta_0\;0.998^{\,2\,q\,(1-\eta_0)}\;\;(q\ \mathrm{in\ W/cm^2})')
    doc.table(['bandgap [um]', 'a [1/K2]', 'b [1/K]', 'c', 'peak eta_0', 'at T [K]', 'eta_0 < 0 outside [K]'],
              [[f"{k:g}", f"{a:.4e}", f"{b:.4e}", f"{c:.5f}", f"{c - b * b / (4 * a):.3f}", f"{-b / (2 * a):.0f}",
                f"{_eta_root(a, b, c)[0]:.0f} - {_eta_root(a, b, c)[1]:.0f}"]
               for k, (a, b, c) in sorted(P._PANEL_EFF_COEFFS.items())], widths=[0.9, 1.0, 1.0, 0.8, 0.9, 0.8, 1.7], size=8.5)
    doc.p("No source is given. There is no clamp. A bandgap other than the three keys raises ValueError "
          "(exact float lookup). The docstring states the exponent as 2 flux eta_0; the code uses 2 flux (1 - eta_0) "
          "(Review item in 4.5).")

    # 7.6 TaC, ZrC, W
    doc.h2('7.6 TaC, ZrC and tungsten (other liner / foil options)')
    doc.table(['function', 'formula / data', 'range, clamp', 'source', 'where'], [
        ['tac_emis_temp', 'eps = 0.0238 + 1.084e-4 T', 'none', 'no source', ref('properties.py', 'def tac_emis_temp')],
        ['tac spectral (inlined)', 'eps = -0.0351 + 8.15e-5 T + (0.3008 - 4.46e-5 T) / lambda[um]', 'clip [0, 1]', 'no source',
         ref('axial.py', 'eps_in = np.clip(-0.0351')],
        ['tac_emis_wavelength_temp', 'same, with lambda in m (1e6 lambda)', 'none', 'no source', ref('properties.py', 'def tac_emis_wavelength_temp')],
        ['tac_emis_binned_1eV', '[0.204 + 7.18e-5 T, 0.0685 + 7.58e-5 T] (below / above 1.24 um)', 'none', 'no source', ref('properties.py', 'def tac_emis_binned_1eV')],
        ['zrc_emis_temp', 'eps = 0.77 - 9e-5 T', 'clip [0.35, 1]', 'Shabalin, UHT Materials II, Table A.6', ref('properties.py', 'def zrc_emis_temp')],
        ['zrc spectral', 'eps = 0.70 - 1e-4 T + 0.02 (lambda[um] - 0.665)', 'clip [0, 1]', 'same', ref('properties.py', 'def zrc_emis_wavelength_temp')],
        ['wc_emis_grey', 'eps = 0.4 (docstring: tungsten)', '-', 'no source', ref('properties.py', 'def wc_emis_grey')],
        ['w_emis_total', 'degree-5 polynomial in T/1000, Planck-weighted fit to the spectral table, +-0.004', 'clip 200-3100 K', 'from table', ref('properties.py', 'def w_emis_total')],
        ['w_emis_spectral(_vec)', 'table: T in {0, 700, 1600, 2600, 3100} K x 25 wavelengths 0-1000 um; bilinear', 'vec: clamp; scalar: ValueError', 'no source', ref('properties.py', 'def w_emis_spectral_vec')],
        ['w_tcx', '173 (T<300); 173 - 0.03(T-300) (T<1000); 152 - 0.01(T-1000)', 'none', 'no source', ref('properties.py', 'def w_tcx')],
    ], widths=[1.3, 2.6, 0.9, 1.2, 1.1], size=7.5)
    doc.p("Liquid-tin correlations (Sn_rho from Stankus, Sn_tcx, Sn_mu, Sn_cp = 300 constant, Sn_Pr) are used only by "
          f"the legacy tin-loop model ({ref('properties.py', 'def Sn_rho')}).")

    # 7.7 plots
    doc.h2('7.7 Property plots')
    T = np.linspace(300, 3200, 400)
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.0))
    ax = axs[0]
    ax.plot(T, P.gr_tcx(T), label='gr_tcx (liner, plate)', color='#2F5F9E')
    ax.plot(T, np.full_like(T, float(P.storage_tcx(T_lo))), label='storage_tcx (block, used)', color='k')
    ax.plot(tC + C2K, k_tab, 'o--', ms=3, color='#999999', label='storage table (unused)')
    ax.set_xlabel('T [K]'); ax.set_ylabel('k [W/m-K]'); ax.set_title('graphite conductivity', fontsize=9)
    ax.legend(fontsize=6.5)
    ax = axs[1]
    ax.plot(T, np.full_like(T, float(P.storage_cp(T_lo))), color='k', label='storage_cp (block, used)')
    ax.plot(tC + C2K, cp_tab, 'o--', ms=3, color='#999999', label='storage table (unused)')
    ax.plot(T, P.gr_cp(T), color='#B5520A', label='gr_cp Entegris (unused)')
    ax.set_xlabel('T [K]'); ax.set_ylabel('c_p [J/kg-K]'); ax.set_title('graphite specific heat', fontsize=9)
    ax.legend(fontsize=6.5)
    for a in axs:
        a.axvspan(T_lo, T_hi, color='#F3E0C8', alpha=0.5, lw=0)
        a.tick_params(labelsize=7)
    fig.tight_layout()
    doc.fig(fig, 'Graphite conductivity and specific heat. Shaded: block operating range of the YAML case.', name='Figure 7.1')

    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.0))
    ax = axs[0]
    ax.plot(T, P.gr_emis(T), label='graphite gr_emis', color='k')
    ax.plot(T, P.tac_emis_temp(T), label='TaC total', color='#2F5F9E')
    ax.plot(T, [P.zrc_emis_temp(t) for t in T], label='ZrC total', color='#2E7D32')
    ax.plot(T, [P.w_emis_total(t) for t in T], label='W total', color='#777777')
    ax.axhline(P.LINER_GRAY['grafoil']['eps'], color='#B5520A', ls=':', label='grafoil')
    ax.set_xlabel('T [K]'); ax.set_ylabel('emissivity'); ax.set_title('total emissivities', fontsize=9)
    ax.set_ylim(0, 1); ax.legend(fontsize=6.5)
    ax.axvspan(T_lo, T_hi, color='#F3E0C8', alpha=0.5, lw=0)
    ax = axs[1]
    lam = np.logspace(np.log10(0.3), np.log10(20), 200)
    for Tt, ls in ((T_lo, ':'), (T_hi, '-')):
        ax.plot(lam, np.clip(-0.0351 + 8.15e-5 * Tt + (0.3008 - 4.46e-5 * Tt) / lam, 0, 1), color='#2F5F9E', ls=ls,
                label=f"TaC {Tt:.0f} K")
        ax.plot(lam, P.w_emis_spectral_vec(Tt, lam), color='#777777', ls=ls, label=f"W {Tt:.0f} K")
        ax.plot(lam, np.clip(0.70 - 1e-4 * Tt + 0.02 * (lam - 0.665), 0, 1), color='#2E7D32', ls=ls, label=f"ZrC {Tt:.0f} K")
    ax.axhline(float(P.gr_emis(T_hi)), color='k', lw=0.8, label=f"graphite {T_hi:.0f} K")
    ax.set_xscale('log'); ax.set_xlabel('wavelength [um]'); ax.set_ylabel('spectral emissivity')
    ax.set_title('spectral emissivities (liner inner-face options)', fontsize=9)
    ax.set_ylim(0, 1); ax.legend(fontsize=5.8, ncol=2)
    for a in axs:
        a.tick_params(labelsize=7)
    fig.tight_layout()
    doc.fig(fig, 'Emissivities. Left: total emissivity versus temperature. Right: spectral emissivity at the block '
                 'T_min (dotted) and T_max (solid).', name='Figure 7.2')

    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.0))
    ax = axs[0]
    ax.plot(T, P.ins_tcx(T), color='#B5520A', label='RGF ins_tcx')
    ax.plot(T, P.thermax_tcx(T), color='#2F5F9E', label='Thermax')
    ax.plot(tI + C2K, [float(P.ins_tcx(t + C2K)) for t in tI], 'o', ms=3, color='#B5520A')
    ax.axvline(2500 + C2K, color='#999999', ls=':', lw=0.8)
    ax.text(2500 + C2K + 20, 0.1, 'RGF table end\n(extrapolated)', fontsize=6.5, color='#666666')
    ax.set_xlabel('T [K]'); ax.set_ylabel('k [W/m-K]'); ax.set_title('insulation conductivity', fontsize=9)
    ax.legend(fontsize=7)
    ax.axvspan(T_lo, T_hi, color='#F3E0C8', alpha=0.5, lw=0)
    ax = axs[1]
    Te = np.linspace(1200, 3000, 300)
    for bg, col in zip(sorted(P._PANEL_EFF_COEFFS), ('#2F5F9E', '#2E7D32', '#B5520A')):
        ax.plot(Te, [P.panel_efficiency(bg, t) for t in Te], color=col, label=f"{bg} um, no flux")
    q = float(m.cfg['storage']['avg_flux'])
    ax.plot(Te, [P.panel_efficiency(1.24, t, flux=q) for t in Te], color='#2F5F9E', ls='--', label=f"1.24 um, {q:g} W/cm2")
    ax.plot(Te, [P.panel_efficiency(1.24, t, flux=2 * q) for t in Te], color='#2F5F9E', ls=':', label=f"1.24 um, {2 * q:g} W/cm2")
    ax.set_xlabel('emitter (liner inner face) T [K]'); ax.set_ylabel('efficiency'); ax.set_ylim(0, 0.55)
    ax.set_title('panel_efficiency', fontsize=9)
    ax.legend(fontsize=6.5)
    for a in axs:
        a.tick_params(labelsize=7)
    fig.tight_layout()
    doc.fig(fig, 'Left: insulation conductivity (dots: RGF table points). Right: TPV panel efficiency versus emitter '
                 'temperature for the three bandgaps, and the flux factor for the in-use 1.24 um cells.',
            name='Figure 7.3')
    rows += [['properties', 'block rho / c_p / k', f"{_f(P.storage_rho(T_lo))} / {_f(P.storage_cp(T_lo))} / "
              f"{_f(P.storage_tcx(T_lo))} (constants)", ref('properties.py', 'def storage_rho')],
             ['properties', 'gr_tcx table', '20-2400 (as coded), flat outside', ref('properties.py', 'def gr_tcx')],
             ['properties', 'ins_tcx table', '0-2500 C, flat outside (2500 C point extrapolated)', ref('properties.py', 'def ins_tcx')],
             ['properties', 'Kirchhoff integral of ins_tcx', 'trapezoid table 150-6000 K, dT 0.25 K',
              ref('axial.py', 'def __init__(self, k_fn=ins_tcx, T_lo=150.0')],
             ['properties', 'w_emis_total clip', '200-3100 K', ref('properties.py', 'def w_emis_total')],
             ['properties', 'panel_efficiency', 'no clamp; bandgap must be 1.24 / 1.38 / 1.55', ref('properties.py', 'def panel_efficiency')]]
    return rows


# ==========================================================================
# Appendix A
# ==========================================================================
def alternatives(doc, ctx):
    doc.h2('Geometry, conduction and property options (sections 3, 4.1, 4.5, 7)')
    doc.h3('Local per-segment radiation (radiation.axial: false)')
    doc.p("The original model: each block ring exchanges radiation only with the stick step beside it, with infinite "
          "coaxial-cylinder formulas, and the shade fraction is found by brentq on the total cell flux "
          f"({ref('tinless.py', 'def _step_calc')}, {ref('tinless.py', 'def _compute_f_shade')}). It needs "
          f"block_height == L (one ring per stick step; NotImplementedError otherwise, "
          f"{ref('tinless.py', 'local per-segment radiation needs one block ring')}), "
          "and the block is uniform rings of height L / n. The block conduction is the same update_temps, but with an "
          "explicit wall flux (no dQ/dT_w). In the legacy liner solve, the liner inner-face temperature is floored at "
          f"400 K ({ref('tinless.py', 'T_liner_in   = max(T_liner_out - Q_in * R_cond_liner, 400.0)')}).")
    doc.h3('Porous shade and other cavity sizes')
    doc.p("With shade.model: porous, the cavity is r_in = (1 + sqrt2) r_liner + 0.01 m "
          f"({ref('build.py', 'r_in = (1.0 + np.sqrt(2)) * (2 * liner_side_len / np.pi) + 0.01')}). The shade is an "
          "octagon sized by liner_to_shade_perimeter_ratio (at least 1 + sqrt2/2), with open-area limits "
          "f_min = 8 t / perimeter and f_max = 1 - 8 gap / perimeter. Without a shade, r_in = "
          "powerstick.area_ratio x r_ps.")
    doc.h3('Liner material options')
    doc.bl([
        "tac: inner face uses the TaC spectral fit (section 7.6, clipped to [0, 1]); outer face graphite; wall "
        "k = gr_tcx. A warning is raised if the bandgap is not 1.24 um.",
        "zrc: inner face uses the ZrC spectral fit (Shabalin); otherwise as tac.",
        f"grafoil: gray, eps = {_grafoil()['eps']} on both faces, k = {_grafoil()['k']} W/m-K through the wall "
        "(typical 0.5 mm walls, CFC floor). Constant properties, from LINER_GRAY.",
        "tungsten_foil: a W foil between the TaC face and the stick (spectral W emissivity table). Supported only "
        "by the local model; the axial model raises NotImplementedError.",
        "liner_bool: false: no liner. The axial model requires a liner.",
    ])
    doc.h3('Outer-wall heat loss (block_heat_loss_bool: true)')
    doc.p("Radial loss from the outer node through th_ins of carbon insulation (ins_tcx) and h = 100 W/m2K to T_sur "
          "(section 4.1). It is off because the cell's outer radius is a symmetry line in the checkerboard; plant "
          "side losses are handled by the tank model (section 5).")
    doc.h3('Temperature-dependent block properties')
    doc.p("storage_cp and storage_tcx contain (unused) tables. Re-enabling them would need the conservative face "
          "conductivity noted in 4.1, and an enthalpy formulation for consistent energy bookkeeping with c_p(T).")


def _grafoil():
    from .properties import LINER_GRAY
    return LINER_GRAY.get('grafoil', {'eps': '?', 'k': '?'})
