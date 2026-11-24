"""Model documentation: section 4.6 (heater and the charge cell), 4.7 (deposition and
degradation), 6 (transient simulation) and the matching appendix-A alternatives.

Numbers are read from the YAML (ctx['base_cfg']), the optimizer settings (ctx['opt']) and
the code (function defaults, constants) when the document is built. file:line references
are found by searching the source, so they follow the code.
"""
import inspect
import os

import numpy as np

from .docs_base import flowchart as _flowchart, DIM

C2K = 273.15
_SRC = {}
_FIG = {'n': 0}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _ref(fname, pat, nth=1):
    """'fname:line' of the nth line of thermal/fname containing pat (live from the source)."""
    if fname not in _SRC:
        try:
            _SRC[fname] = open(os.path.join(os.path.dirname(__file__), fname), encoding='utf-8').read().splitlines()
        except OSError:
            _SRC[fname] = []
    k = 0
    for i, line in enumerate(_SRC[fname]):
        if pat in line:
            k += 1
            if k == nth:
                return f"{fname}:{i + 1}"
    return fname


def _dflt(fn, name):
    return inspect.signature(fn).parameters[name].default


def _figname(sec):
    _FIG['n'] += 1
    return f"Figure {sec}.{_FIG['n']}"


def _log10_fit(fn, T1=2000.0, T2=2600.0):
    """(A, B) with log10 fn(T) = A - B / T, recovered from the code by two evaluations."""
    y1, y2 = np.log10(fn(T1)), np.log10(fn(T2))
    B = (y1 - y2) / (1.0 / T2 - 1.0 / T1)
    return y1 + B / T1, B


def _C(T):
    return np.asarray(T, float) - C2K


def _ring_means(T, r):
    w = np.diff(np.asarray(r) ** 2)
    return (np.asarray(T) * w).sum(axis=-1) / w.sum()


def _geom_model(ctx):
    """Stick model geometry from the YAML (cheap: no radiation model). None on failure."""
    try:
        from .build import build_tinless
        return build_tinless(ctx['base_cfg'], geometry_only=True)
    except Exception as e:                       # pragma: no cover - documentation only
        print(f"[docs] secC: geometry build failed ({type(e).__name__}: {e})")
        return None


def flowchart(*a, **k):
    """docs_base.flowchart with the arrows drawn after the boxes. The arrows clip against the
    boxes' frames, which are only updated when the boxes are drawn; drawn first (zorder 2)
    they clip against the previous draw's frames, wrong for a savefig at another dpi or
    with bbox_inches='tight'."""
    from matplotlib.text import Annotation
    fig = _flowchart(*a, **k)
    for ax in fig.axes:
        for t in ax.texts:
            if isinstance(t, Annotation):
                t.set_zorder(3.5)
    return fig


def _nl(doc, items):
    """Numbered list that restarts at 1 (Word's List Number style continues across lists)."""
    for i, it in enumerate(items, 1):
        para = doc.p((f"{i}.  ", {'bold': True}), it)
        para.paragraph_format.left_indent = doc.Inches(0.32)
        para.paragraph_format.first_line_indent = doc.Inches(-0.22)


def _A_act_cm2(m):
    ps = m.powerstick
    return 4 * ps.width * ps.height * (1 - ps.f_bare) * ps.f_active * 1e4


# ==========================================================================
# 4.6 Heater, 4.7 Deposition
# ==========================================================================
def sec_heater_deposition(doc, ctx):
    cfg, opt, cy = ctx['base_cfg'], ctx['opt'], ctx.get('cy')
    rows = []
    rows += _sec_heater(doc, cfg, opt, cy)
    rows += _sec_deposition(doc, ctx, cfg, opt, cy)
    return rows


def _sec_heater(doc, cfg, opt, cy):
    from .build import heater_max_power
    from .heater import element, describe
    h = cfg['heater']
    H = float(cfg['storage'].get('block_height') or cfg['storage']['L'])
    el = element(h, H, float(cfg['storage']['T_min']))
    r_h = el['r_h']
    P_cap = heater_max_power(h, H)
    _FIG['n'] = 0
    doc.h2('4.6 Heater and the charge cell')
    doc.p("Charging is modelled on a separate cell: one resistive heater in its own round cavity in the "
          "graphite. On the checkerboard every stick has one heater (N_h = N_ps). The heater cell holds the same "
          "graphite volume as the stick cell, so the charge and the discharge describe the same material from the "
          "point of view of the two kinds of cavity. The cell is built by ",
          ('build.build_heater', {'mono': True}), f" ({_ref('build.py', 'def build_heater')}) from a built stick "
          "model, and is run by ", ('TinlessSimulation.run_heating_simulation', {'mono': True}),
          f" ({_ref('tinless.py', 'def run_heating_simulation')}). The time loop is in section 6.6.")
    doc.h3('Geometry')
    doc.eq(r'r_h = \frac{2\,s_h}{\pi},\qquad r_{in,h} = a_h\,r_h,\qquad '
           r'r_{out,h} = \sqrt{\frac{V_{block}}{\pi H} + r_{in,h}^2},\qquad '
           r'V_{block} = \pi\,(r_{out}^2 - r_{in}^2)_{stick}\,H',
           where='s_h: heater side length (square rod; r_h is the radius of the circle with the same perimeter); '
                 'a_h: heater.area_ratio (cavity radius / heater radius); H: block height.')
    doc.bl([
        'The heater cell uses the same axial ring edges and the same number of radial rings as the stick cell '
        f"({_ref('build.py', 'storage.set_axial_mesh(st.axial_edges)')}).",
        'The heater spans the full block height. The cavity has no liner and no shade, and the placeholder '
        'powerstick in the heater model is never used.',
        'The outer radius is adiabatic: it stands for the symmetry line between a heater and its neighbouring '
        'sticks. Roof and floor losses are applied as a sink on the top and bottom rings (section 6.8).',
        f"Initial state: every node at T_min ({_ref('build.py', 'np.full((nA, nR), float(m.T_min))')}). This is "
        "not the field left by a discharge.",
    ])
    doc.table(['YAML heater input', 'value', 'use'], [
        ['heater.type', f"{h.get('type', 'rod')}", f"element at the power cap: {describe(el)}; r_h = {r_h:.4f} m "
                                                    f"({_ref('heater.py', 'def element')}). Tube: D = sqrt(rho_e legs^2 H P "
                                                    "/ (V^2 pi w (1 - w))), rho_e = the highest of "
                                                    "properties.heater_graphite_resistivity (Mersen 2020 data to 2002 C, extrapolated above) over "
                                                    "[T_min, max_temperature]; the element is resized for every heater "
                                                    "power the charge search tries"],
        ['heater.voltage / wall_ratio / legs', f"{h.get('voltage')} V / {h.get('wall_ratio')} / {h.get('legs', 1)}",
         'tube: DC supply voltage at full power, wall thickness / OD, strips in series'],
        ['heater.density', f"{h.get('density', 1850.0)} kg/m3", 'element mass = A_cross x (H_block + t_top) x density '
                                                               '(costed at heater_graphite_per_kg)'],
        ['heater.area_ratio', f"{h['area_ratio']}", f"cavity radius {h['area_ratio'] * r_h:.3f} m; the optimizer "
                                                   "searches it (section 9)"],
        ['heater.max_temperature', f"{h['max_temperature']} K", f"surface temperature limit "
                                                                f"({h['max_temperature'] - C2K:.0f} C)"],
        ['heater.max_power_per_m', f"{h['max_power_per_m']:,.0f} W/m", f"cap = max_power_per_m x H = "
                                                                       f"{P_cap / 1e6:.2f} MW at H = {H:g} m"],
        ['heater.max_power', str(h.get('max_power')), 'absolute override of the cap (used by the power searches '
                                                       'in cycle.py / optimize.py)'],
        ['heater.emissivity', f"{h['emissivity']}", 'gray, constant'],
        ['heater.charge_hr', f"{h.get('charge_hr')}", 'target charge time for the heater search'],
    ], widths=[1.7, 1.3, 4.1])

    doc.h3('Radiation from the heater')
    doc.p("Each axial ring is treated as two long concentric gray cylinders (heater and cavity wall), with view "
          f"factor 1 from heater to wall ({_ref('tinless.py', 'def _q_segs')}):")
    doc.eq(r'Q_i = \frac{\sigma\,(T_h^4 - T_{w,i}^4)\,A_{h,i}}'
           r'{\frac{1-\varepsilon_h}{\varepsilon_h} + 1 + \frac{A_{h,i}}{A_{w,i}}\,\frac{1-\varepsilon_{gr}(T_{w,i})}'
           r'{\varepsilon_{gr}(T_{w,i})}},\qquad A_{h,i} = \pi D_h\,\Delta z_i,\quad A_{w,i} = 2\pi r_{in,h}\,\Delta z_i',
           where='T_h: heater temperature (one value for the whole heater); T_w,i: inner-wall node of ring i; '
                 'eps_gr from properties.gr_emis.')
    doc.p("Rings do not exchange radiation with each other, and the cavity floor and ceiling are not modelled. "
          "The power each ring receives therefore depends only on its own wall temperature.")
    doc.h3('Heater temperature and power control')
    doc.p(f"Each step, _solve_T_heater ({_ref('tinless.py', 'def _solve_T_heater')}) sets T_h from the "
          "start-of-step wall temperatures:")
    _nl(doc, [
        'If the hottest wall node is at or above heater.max_temperature: no heat (Q = 0).',
        'Otherwise evaluate the total power at T_h = max_temperature. If it is at least the power cap P_max '
        '("phase 1", power-limited): solve sum_i Q_i(T_h) = P_max for T_h with Brent\'s method on '
        '[max T_w + 0.1 K, max_temperature], xtol 1 K, at most 50 iterations. If there is no sign change in that '
        'bracket, T_h = max_temperature is used.',
        'Otherwise ("phase 2", temperature-limited): T_h = max_temperature, and the power is whatever the wall '
        'accepts at that temperature (< P_max).',
    ])
    doc.p("All of P_max goes into the block as radiation: the element is lossless inside the transient. The heater "
          "efficiency (economics.heater_eff / eta_in) is applied only in the plant accounting (sections 8, 9).")
    fig = _fig_heater_curve(h, H, cfg)
    if fig is not None:
        dP = getattr(_fig_heater_curve, 'dP', np.nan)
        doc.fig(fig, "Heater power per metre and heater temperature against a uniform cavity-wall temperature, from "
                     "the equations above, for the YAML heater. Left of the knee the heater is at its power cap "
                     "(phase 1); right of it, at its temperature limit (phase 2). Charging slows sharply once the "
                     "wall approaches the heater limit. The cavity area ratio hardly enters the exchange (between "
                     f"area ratios 3 and 7 the power differs by at most {dP * 100:.1f}%). The cavity size matters "
                     "through the wall area the power spreads over (wall-to-bulk temperature difference) and the "
                     "graphite it displaces.", width=6.4, name=_figname('4.6'))
    if cy is not None:
        hm = cy.hm
        doc.p(f"Example design (section 8): heater {hm.heater.max_power / 1e3:,.0f} kW over {hm.H:.2f} m, cavity "
              f"radius {hm.r_in:.3f} m (area ratio {hm.area_ratio:.2f}), cell outer radius {hm.r_out:.3f} m, charge "
              f"{cy.t_charge_hr:.1f} h from {hm.T_min - C2K:.0f} C to a volume-average {hm.T_max - C2K:.0f} C. "
              "The run is shown in section 6.6.")
    doc.review(f"Uniform-temperature heater: one T_h for the whole element, so the power per ring follows the local "
               f"wall temperature ({_ref('tinless.py', 'def _q_segs')}). A series resistive element has roughly "
               "uniform power per length and a varying temperature. The model sends more power to cold rings, "
               "for example the top and bottom rings that carry the roof and floor losses. "
               "The cavity floor, ceiling and ring-to-ring exchange are also missing. "
               "Confidence high (code); effect on charge time low to moderate (not run).")
    doc.review(f"Power cap can be exceeded: if Brent's method finds no sign change, T_h falls back to max_temperature "
               f"with total power above P_max ({_ref('tinless.py', 'T_h = heater.max_temperature')}). This can only "
               "happen when one wall ring is much hotter than the rest. Rare, but it is not reported.")
    doc.review(f"The charge stops on the volume-average block temperature reaching T_max "
               f"({_ref('tinless.py', 'if T_avg_list[-1] >= T_block_max')}). The cavity wall can run far above "
               f"T_max, up to the heater limit ({h['max_temperature'] - C2K:.0f} C). Neither the optimizer's T_max cap "
               "nor any check limits the bore-wall temperature, and graphite loss at the heater bore is not "
               "modelled.")
    rows = [
        ['heater', 'heater temperature solve', 'Brent, xtol 1 K, maxiter 50', _ref('tinless.py', 'lambda Th: _total_q')],
        ['heater', 'heater solve bracket', '[max T_w + 0.1 K, T_h,max]', _ref('tinless.py', 'T_lo = T_inner_max + 0.1')],
        ['heater', 'heater fallback', 'T_h = T_h,max (power may exceed cap)', _ref('tinless.py', 'T_h = heater.max_temperature')],
        ['heater', 'heater power cap', f"max_power_per_m x H ({P_cap / 1e6:.2f} MW in the YAML)",
         _ref('build.py', 'def heater_max_power')],
        ['heater', 'heater-cell initial state', 'uniform T_min', _ref('build.py', 'np.full((nA, nR), float(m.T_min))')],
    ]
    return rows


def _fig_heater_curve(h, H, cfg):
    try:
        import matplotlib.pyplot as plt
        from scipy.optimize import brentq
        from .properties import gr_emis
    except Exception:
        return None
    sig = 5.67036713e-8
    from .heater import element
    r_h = element(h, H, float(cfg['storage']['T_min']))['r_h']
    D = 2 * r_h
    eps_h = float(h['emissivity'])
    T_hmax = float(h['max_temperature'])
    Pm = float(h['max_power_per_m'])
    T_max = float(cfg['storage']['T_min']) + float(cfg['storage']['block_temp_range'])
    Tw = np.linspace(1300.0, T_hmax - 5.0, 160)
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8))
    curves = {}
    for ar in (3.0, float(h['area_ratio']), 7.0):
        P, Th = [], []
        for T in Tw:
            eg = float(gr_emis(T))
            R = (1 - eps_h) / eps_h + 1 + (1 / ar) * (1 - eg) / eg
            q = lambda Tx: sig * (Tx ** 4 - T ** 4) * np.pi * D / R
            if q(T_hmax) >= Pm:
                t_ = brentq(lambda Tx: q(Tx) - Pm, T + 0.1, T_hmax, xtol=0.5)
                P.append(Pm); Th.append(t_)
            else:
                P.append(q(T_hmax)); Th.append(T_hmax)
        curves[ar] = (np.array(P), np.array(Th))
    P, Th = curves[float(h['area_ratio'])]
    ax[0].plot(_C(Tw), P / 1e3, c='C3', label=f"YAML heater, area ratio {h['area_ratio']}")
    ax[1].plot(_C(Tw), _C(Th), c='C3', label=f"YAML heater, area ratio {h['area_ratio']}")
    # sensitivity to the cavity size (used in the caption)
    P3, P7 = curves[3.0][0], curves[7.0][0]
    _fig_heater_curve.dP = float(np.max(np.abs(P3 - P7) / P7))
    for a in ax:
        a.axvline(T_max - C2K, c='0.5', lw=0.8, ls=':')
        a.text(T_max - C2K, 0.04, ' YAML T_max ', fontsize=7, color='0.4', ha='right',
               transform=a.get_xaxis_transform())
        a.set_xlabel('cavity wall temperature [C]')
        a.grid(alpha=0.3)
        a.legend(fontsize=7, loc='center left')
    ax[0].set_ylabel('heater power [kW per m of heater]')
    ax[0].set_title(f"power (cap {Pm / 1e3:.0f} kW/m)", fontsize=9)
    ax[1].set_ylabel('heater temperature [C]')
    ax[1].set_title(f"heater temperature (limit {T_hmax - C2K:.0f} C)", fontsize=9)
    fig.tight_layout()
    return fig


def _sec_deposition(doc, ctx, cfg, opt, cy):
    from . import deposition as dp
    _FIG['n'] = 0
    doc.h2('4.7 Deposition and degradation')
    doc.p("Graphite (and, for carbide or tungsten surfaces, the metal) evaporates from the hot surface facing the "
          "stick and condenses on the cold stick. The deposit darkens the bare and inactive areas and lowers the "
          "cell efficiency. The model has two parts:")
    doc.bl([
        ('a rate model, thermal/deposition.py, evaluated each step during a discharge and integrated to a deposit '
         'thickness per cycle; it is a diagnostic only and does not change the run it is computed in;'),
        ('a degradation state (D_c, D_b) applied to a whole run through the emissivity table and a cell-efficiency '
         'factor (build.py). The optimizer maps the deposit per cycle to a lifetime-average D (section 9.4).'),
    ])
    doc.h3('Transport model')
    doc.p("Steady, one-dimensional diffusion of vapour across a stagnant argon gap of width d from the emitter "
          "(saturated vapour at its surface) to the stick, which is a perfect sink (zero concentration). The flux at "
          "the emitter is scaled by the area ratio to a thickness rate on the stick "
          f"({_ref('deposition.py', 'def deposition_rate_tinless')}):")
    doc.eq(r'\bar{T} = \frac{T_e + T_{ps}}{2},\qquad h_C = \frac{D_{C\!-\!Ar}(\bar{T})}{d},\qquad '
           r'C_{sat} = \frac{P_C(T_e)}{R\,T_e},\qquad J_C = h_C\,C_{sat}\;\;[\mathrm{mol\,m^{-2}s^{-1}}]')
    doc.eq(r'\dot{\delta}_C = \frac{J_C\,(A_e/A_{ps})}{\rho_{m,gr}}\times 3.154\times10^{16}\;\;[\mathrm{nm/yr}],'
           r'\qquad \dot{\delta}_M = \frac{h_M\,P_M(T_e)/(R T_e)\,(A_e/A_{ps})}{\rho_{m,M}}\times 3.154\times10^{16}',
           where='T_e: emitter temperature; T_ps: stick (cooling) temperature; d: emitter-to-stick gap; '
                 'A_e/A_ps: area ratio; rho_m: molar density of the deposit; 3.154e16 = 1e9 nm/m x 3.154e7 s/yr.')
    A_C, B_C = _log10_fit(dp.vapor_pressure_graphite)
    A_T, B_T = _log10_fit(dp.vapor_pressure_tac_ta)
    A_Z, B_Z = _log10_fit(dp.vapor_pressure_zrc_zr)
    A_W, B_W = _log10_fit(dp.vapor_pressure_tungsten)
    DC0 = dp.diffusion_coefficient_carbon_argon(298.0)
    DW0 = dp.diffusion_coefficient_tungsten_argon(298.0)
    nC = np.log(dp.diffusion_coefficient_carbon_argon(2000.0) / DC0) / np.log(2000.0 / 298.0)
    doc.table(['quantity', 'expression / value (from the code)', 'source in code'], [
        ['carbon vapour pressure P_C', f"log10 P [Pa] = {A_C:.3f} - {B_C:,.0f} / T   ('Brewer et al.')",
         _ref('deposition.py', 'def vapor_pressure_graphite')],
        ['Ta over C-saturated TaC', f"log10 P [Pa] = {A_T:.3f} - {B_T:,.0f} / T", _ref('deposition.py', 'def vapor_pressure_tac_ta')],
        ['Zr over ZrC', f"log10 P [Pa] = {A_Z:.3f} - {B_Z:,.0f} / T   ('Pollock')", _ref('deposition.py', 'def vapor_pressure_zrc_zr')],
        ['W', f"log10 P [Pa] = {A_W:.3f} - {B_W:,.0f} / T   ('Plante & Sessoms')", _ref('deposition.py', 'def vapor_pressure_tungsten')],
        ['C in Ar diffusivity', f"{DC0:.2e} (T/298)^{nC:.2f} m2/s (includes x3.5 'from CFD studies')",
         _ref('deposition.py', 'def diffusion_coefficient_carbon_argon')],
        ['metal in Ar diffusivity (all metals)', f"{DW0:.2e} (T/298)^{nC:.2f} m2/s (includes x3.5)",
         _ref('deposition.py', 'def diffusion_coefficient_tungsten_argon')],
        ['C through W (solid)', '9.22e-7 exp(-169805 / (R T)) m2/s (Buhsmer & Crayton)',
         _ref('deposition.py', 'def diffusion_coefficient_carbon_tungsten')],
        ['molar densities', f"graphite {dp.gr_rho_molar(0):,.0f}, TaC {dp.tac_rho_molar(0):,.0f}, ZrC "
                            f"{dp.zrc_rho_molar(0):,.0f}, W {dp.w_rho_molar(0):,.0f} mol/m3", _ref('deposition.py', 'def gr_rho_molar')],
        ['C solubility in W', f"{_dflt(dp.deposition_rate_tinless, 'f_C_sol_W')} (mole fraction, at a_C = 1)",
         _ref('deposition.py', 'f_C_sol_W=0.003')],
        ['argon pressure', f"P_total = {_dflt(dp.deposition_rate_tinless, 'P_total')} Pa: accepted but not used",
         _ref('deposition.py', 'P_total=101325')],
    ], widths=[1.8, 3.7, 1.6], size=8)
    doc.h3('Surface materials')
    doc.bl([
        'graphite, TaC, ZrC: the carbon flux is that of saturated graphite at T_e. Carbides on graphite are taken '
        'as carbon-saturated (carbon activity 1), so a carbide coating does not reduce carbon deposition. TaC and '
        'ZrC add a metal term.',
        'tungsten foil (thickness th_w) in front of the liner: carbon must dissolve in and diffuse through the W. '
        'The solid and gas resistances are in series, with the partition coefficient k_s = C_W / C_sat:',
    ])
    doc.eq(r'J_C = \frac{h_C\,D_{C,W}\,C_{W}}{t_W\,h_C + D_{C,W}\,k_s},\qquad C_W = \rho_{m,W}\,x_{C,sol},\qquad '
           r'k_s = C_W / C_{sat}')
    doc.bl(['w_shell: the same series form, with the carbon loading set by the graphite-side face at T_graphite '
            '(not used by the tinless model).'])
    m0 = cy.m if cy is not None else _geom_model(ctx)
    if m0 is not None:
        dk = m0.dep_kwargs
        doc.p(f"In use (built in {_ref('build.py', 'dep_kwargs = dict(')}): ")
        doc.kv([
            ('surface_material', f"{dk['surface_material']} (liner {cfg.get('liner_material')}, tungsten foil "
                                 f"{cfg['powerstick'].get('tungsten_foil', False)})"),
            ('gap d', f"{dk['d'] * 1e3:.1f} mm  = r_liner,eff - r_ps,eff (equivalent-circle radii)"),
            ('area ratio A_e / A_ps', f"{dk['area_ratio']:.3f}  = r_liner,eff / r_ps,eff"),
            ('T_ps', f"{dk['T_ps']:.0f} K (powerstick cooling temperature)"),
            ('emitter temperature T_e', 'mean liner inner-face temperature over the active rings at the start of the '
                                        f"step ({_ref('tinless.py', 'def _dep_T_emitter')})"),
        ], w0=2.0)
        fig = _fig_dep_rate(dk, cy)
        if fig is not None:
            doc.fig(fig, "Deposition rate on the stick against emitter temperature, with the in-use gap and area "
                         "ratio, for the surface options in the code. The in-use liner is graphite, so only carbon "
                         "deposits. With a 25 um tungsten foil the carbon rate equals the bare-graphite rate (markers "
                         "on the black line): the solid-diffusion term is negligible against the gas resistance in "
                         "this series model, so the foil only adds tungsten. Shaded band: liner inner-face "
                         "temperatures in the example discharge (when built with it).", width=6.0,
                    name=_figname('4.7'))
    doc.h3('Accumulation over a run')
    doc.p(f"While the stick is in, the rate is evaluated once per step ({_ref('tinless.py', '_T_e = self._dep_T_emitter')}). "
          "After the stick is removed it is 0, which also applies to the step in which the cutoff is crossed. The deposit "
          f"per discharge is the trapezoid integral over the recorded times ({_ref('tinless.py', 'dep_C_nm     = np.trapz')}):")
    doc.eq(r'\delta_{run}\;[\mathrm{nm}] = \frac{1}{3.154\times10^{7}}\int_0^{t_{end}}\dot{\delta}(t)\,dt\qquad'
           r'(\dot{\delta}\ \mathrm{in\ nm/yr},\ t\ \mathrm{in\ s})')
    if cy is not None:
        r = cy.r_dis
        doc.p(f"Example discharge: {float(r['dep_total_nm']) * 1e3:.3f} pm per cycle "
              f"(carbon {float(r['dep_C_nm']) * 1e3:.3f} pm, metal {float(r['dep_metal_nm']) * 1e3:.3f} pm).")
    doc.h3('Degradation state')
    doc.p(f"cfg['degradation'] = {{'cell': D_c, 'bare': D_b, 'emis_end': e_end}} ({_ref('build.py', 'deg = cfg.get(')}); "
          "{'frac': D} sets both. Every emissivity band (lambda range, eps_active, eps_bare, eps_inactive) is changed "
          "to:")
    doc.eq(r"\varepsilon_{bare}' = \varepsilon_{bare} + D_b\,(e_{end} - \varepsilon_{bare}),\qquad "
           r"\varepsilon_{inact}' = \varepsilon_{inact} + D_c\,(e_{end} - \varepsilon_{inact}),\qquad "
           r"\varepsilon_{active}' = \varepsilon_{active},\qquad \eta' = (1 - D_c)\,\eta",
           where=f"e_end default {0.8}; eta multiplied through sim.cell_eff_factor in every radiation step "
                 f"({_ref('build.py', 'sim.cell_eff_factor = 1.0 - D_deg')}, {_ref('axial.py', 'cell_eff_factor', 2)}).")
    em = cfg['emissivity_tables'][0]['value']
    ex_D = float(opt.get('deposition', {}).get('cell_replace_D', 0.1))
    e_end = float(opt.get('deposition', {}).get('emis_end', 0.8))
    trows = []
    for (b, a, eb, ei) in em:
        trows.append([f"{b[0]}-{b[1]} um", f"{a}", f"{eb} -> {eb + ex_D * (e_end - eb):.3f}",
                      f"{ei} -> {ei + ex_D * (e_end - ei):.3f}"])
    doc.p(f"YAML emissivity table and its value at D_c = D_b = {ex_D} (the cell-replacement point), e_end = {e_end}:")
    doc.table(['band', 'eps_active (unchanged)', 'eps_bare', 'eps_inactive'], trows, widths=[1.4, 1.7, 2.0, 2.0])
    doc.p("Block sizing uses the fresh table, so a degraded model has exactly the fresh geometry. Deposit is mapped "
          f"to D linearly, D = deposit / d_dead with d_dead = {opt.get('deposition', {}).get('d_dead_nm')} nm. The "
          "life-average treatment (replacement and cleaning saw-tooth, one run at 2x the mean D) is in section 9.4.")
    doc.review("The rate is evaluated at the arithmetic mean liner temperature over the active rings "
               f"({_ref('tinless.py', '_T_e = self._dep_T_emitter')}). It is strongly Arrhenius (about x2 per "
               f"{np.log10(2) * 2100 ** 2 / B_C:.0f} K near 2100 K), so an axial liner gradient makes the rate at the "
               "mean temperature an underestimate of the mean rate. An area-weighted per-ring rate would fix it. "
               "Confidence high (code); effect depends on the liner gradient.")
    doc.review("Transport model: stagnant-film diffusion across the full liner-stick gap with a perfect sink, an "
               "unexplained x3.5 'CFD' enhancement on both diffusivities, and no pressure dependence. P_total is "
               f"unused ({_ref('deposition.py', 'P_total=101325')}) although D ~ 1/P, so a non-atmospheric argon "
               "fill scales the result directly. There is no Stefan flow and no buoyant convection beyond the "
               "factor. The source and validity range of the x3.5 factor should be documented.")
    doc.review(f"Carbon vapour pressure log10 P = {A_C:.3f} - {B_C:,.0f}/T Pa gives about "
               f"{dp.vapor_pressure_graphite(2840):.1f} Pa at 2840 K. As a sanity check, handbook total carbon "
               "vapour pressure is about 1 Pa near 2840 K (CRC), so the correlation looks about 3x high, which would "
               "be conservative. It is unclear whether it is C1 or total carbon; every mole is deposited as one C "
               "atom. Low to medium confidence in this comparison: please verify the source.")
    doc.review(f"The metal diffusivity {DW0:.1e} m2/s at 298 K is used for Ta, Zr and W. It looks 5-10x below a "
               "kinetic-theory estimate for a metal atom in argon (low confidence). It does not affect the in-use "
               "graphite liner, but it does affect the TaC and W options.")
    doc.review("Only the liner inner face is a source. The vanes and the block bore are hotter and larger, and sit "
               "in the same argon, but are not considered, and the liner gap is assumed to be the only path to the "
               "stick. The deposit is uniform over the stick, cells and bare area get the same thickness, the "
               f"D-thickness-emissivity/efficiency mapping is linear (d_dead "
               f"{opt.get('deposition', {}).get('d_dead_nm')} nm, e_end {e_end}) with no cited basis, and "
               "eps_active is unchanged by deposit.")
    rows = [
        ['deposition', 'deposition rate evaluation', 'once per step, start-of-step mean liner T',
         _ref('tinless.py', '_T_e = self._dep_T_emitter')],
        ['deposition', 'deposit integration', 'trapezoid over recorded times / 3.154e7 s/yr',
         _ref('tinless.py', 'dep_C_nm     = np.trapz')],
        ['deposition', 'gas diffusivity enhancement', 'x3.5 (C and metal)', _ref('deposition.py', 'conv_factor = 3.5')],
        ['deposition', 'degradation e_end (default)', '0.8', _ref('build.py', "e_end = float(deg.get('emis_end', 0.8))")],
    ]
    return rows


def _fig_dep_rate(dk, cy):
    try:
        import matplotlib.pyplot as plt
        from .deposition import deposition_rate_tinless as rate
    except Exception:
        return None
    T = np.linspace(1700.0, 2600.0, 120)
    base = dict(T_ps=dk['T_ps'], d=dk['d'], area_ratio=dk['area_ratio'])
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for lab, kw, c, sty in (('graphite or TaC: carbon (in use)', dict(surface_material='graphite'), 'k',
                             dict(lw=2.4)),
                            ('W foil 25 um: carbon', dict(surface_material='tungsten', th_w=25e-6), 'C0',
                             dict(lw=1.0, ls='none', marker='o', ms=3, markevery=8)),
                            ('TaC: tantalum', dict(surface_material='tac'), 'C1', dict(ls='--')),
                            ('W foil: tungsten', dict(surface_material='tungsten', th_w=25e-6), 'C9', dict(ls='--'))):
        key = 'metal' if lab.endswith('tantalum') or lab.endswith('tungsten') else 'C'
        y = np.array([rate(t, **base, **kw)[key] for t in T])
        ax.semilogy(_C(T), np.maximum(y, 1e-12), c=c, label=lab, **sty)
    if cy is not None:
        on = np.asarray(cy.r_dis['ps_active'], bool)
        tl = np.asarray(cy.r_dis['T_liner'], float)[on]
        tl = tl[np.isfinite(tl) & (tl > 0)]
        if tl.size:
            ax.axvspan(_C(tl.min()), _C(tl.max()), color='C2', alpha=0.15, label='example liner inner face')
    ax.set_xlabel('emitter temperature T_e [C]')
    ax.set_ylabel('deposition rate on the stick [nm/yr]')
    ax.set_title(f"gap {dk['d'] * 1e3:.0f} mm, area ratio {dk['area_ratio']:.2f}, T_ps {dk['T_ps']:.0f} K", fontsize=9)
    ax.grid(alpha=0.3, which='both')
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


# ==========================================================================
# 6. Transient simulation
# ==========================================================================
def sec_transient(doc, ctx):
    cfg, opt, cy = ctx['base_cfg'], ctx['opt'], ctx.get('cy')
    from . import sizing as S
    from .tinless import TinlessSimulation as TS
    from .components import StorageModule
    _FIG['n'] = 0
    rows = []
    dts = float(opt.get('fidelity', {}).get('dt_scale', 1.0))
    doc.h1('6. Transient simulation')
    doc.p("A cycle is simulated as two separate transients on two cells (section 1.2): the discharge of the stick "
          "cell with the stick in, and the charge of the heater cell. Both march the block temperature field T(z, r) "
          "in time. At every step a quasi-steady heat exchange at the cavity wall (radiation to the stick through "
          "the vanes and liner, or from the heater) is held fixed while the conduction solver (section 4.1) advances "
          "the block. A sizing loop around the discharge (6.7) sets T_min and the block volume. This section covers "
          "what is iterated, in what order, with which step sizes and stopping rules, and what is recorded.")
    fig = flowchart({
        'y': (3.7, 7.0, 'Inputs: YAML (dt_*, sub_steps, end_time, T_min, swing, flux target,\n'
                        'heater, tank) + optimizer fidelity (dt_scale, tolerances)', 'io'),
        'b': (3.7, 5.9, 'build_tinless: stick cell, mesh, vanes (ray-traced anchors),\nblock at uniform T_max', 'proc'),
        'd': (2.4, 4.6, 'Discharge transient (6.2)\nstick in until T_avg = T_min', 'loop'),
        's': (5.9, 4.6, 'Shade-open sizing (6.7)\nT_min, volume factor', 'loop'),
        'k': (0.95, 3.3, 'roof/floor sink\n(provisional\nplant, 5, 6.8)', 'proc'),
        'h': (3.1, 3.3, 'build_heater: same graphite\nvolume, block at uniform T_min', 'proc'),
        'c': (3.1, 2.05, 'Charge transient (6.6)\nheater until T_avg = T_max', 'loop'),
        'o': (5.3, 0.75, 'stick energy and power, deposition,\ncharge time, heater energy\n-> sections 5, 8, 9', 'out'),
    }, [('y', 'b'), ('b', 'd'), ('d', 's', 'run'), ('s', 'b', 'new T_min, V', 0.25), ('k', 'd'), ('k', 'c'),
        ('d', 'h', 'final design'), ('h', 'c'), ('c', 'o'), ('s', 'o', 'final run')], size=(7.4, 7.5))
    doc.fig(fig, 'How the transients fit together. Green: loops in time or iteration; blue: one-off computations.',
            width=5.6, name=_figname('6'))

    # ---------------------------------------------------------------- 6.1 inputs
    doc.h2('6.1 Inputs and settings')
    rk = dict(end_time=cfg['end_time'], dt_short=cfg['dt_short'], dt_medium=cfg['dt_medium'], dt_long=cfg['dt_long'],
              dt_short_duration=cfg.get('dt_short_duration', 3600), sub_steps=cfg['sub_steps'])
    doc.table(['setting', 'YAML', 'meaning in the code', 'optimizer screening'], [
        ['end_time', f"{rk['end_time']:,} s ({rk['end_time'] / 3600:.1f} h)",
         'maximum simulated discharge time; the optimizer sets max(end_time, 4 x duration)', 'same rule'],
        ['dt_short', f"{rk['dt_short']} s", 'first dt_short_duration seconds; while the vanes are fully closed '
                                             '(default); the whole charge', f"x{dts:g}"],
        ['dt_medium', f"{rk['dt_medium']} s", 'while the vanes regulate', f"x{dts:g}"],
        ['dt_long', f"{rk['dt_long']} s", 'while the vanes are fully open; after the stick is removed', f"x{dts:g}"],
        ['dt_short_duration', f"{rk['dt_short_duration']} s", 'initial period at dt_short (not scaled)', 'not scaled'],
        ['dt_at_shade_limit', '(argument)', 'dt while the vanes are fully closed; None -> dt_short',
         f"dt_long x {dts:g}"],
        ['sub_steps', f"{rk['sub_steps']}", 'minimum conduction sub-steps per step (raised automatically for '
                                            'stability, section 4.1)', 'same'],
        ['stop_at_cutoff', '(argument)', 'end the discharge when the stick is removed (no conduction-only tail)', 'True'],
        ['storage.T_min', f"{cfg['storage']['T_min']} K", 'stick-removal cutoff T_ps_cutoff; sizing start guess',
         'T_max - swing_guess'],
        ['storage.avg_flux', f"{cfg['storage']['avg_flux']} W/cm2", 'vane-control setpoint (cell flux)', 'design variable'],
        ['tank.block_axial_losses', f"{cfg.get('tank', {}).get('block_axial_losses', True)}",
         'roof/floor sink on the top/bottom rings in both runs', 'True'],
    ], widths=[1.35, 1.1, 3.3, 1.35], size=8)
    doc.p("Screening values come from optimize._run_kw "
          f"({_ref('optimize.py', 'return dict(dt_scale=dts, dt_at_shade_limit')}). The notebook, cycle.py and the "
          "example in section 8 use the YAML values with dt_scale = 1, dt_at_shade_limit = None, and keep the "
          "conduction-only tail.")

    # ---------------------------------------------------------------- 6.2 discharge
    doc.h2('6.2 Discharge loop')
    doc.p(("TinlessSimulation.run_simulation", {'mono': True}), f" ({_ref('tinless.py', 'def run_simulation')}), "
          "called through ", ('sizing.run_discharge', {'mono': True}), f" ({_ref('sizing.py', 'def run_discharge')}).")
    doc.h3('Initial state')
    doc.bl([
        f"Every node of the block at T_max = T_min + swing ({_ref('build.py', 'np.full((n_ax_tot, num_rad), float(T_max))')}). "
        "There is no charge-end profile and no hold-period relaxation.",
        'A radiation step at t = 0 fills record 0 (fluxes, vane position, liner temperature, deposition rate). '
        'ps_active[0] = True.',
        'Vane control is on when the shade exists and a flux target is set (both true in use).',
    ])
    doc.p("Block averages used by the cutoff and recorded (weights over ring cross-section and ring height; density "
          f"not included, {_ref('tinless.py', '_Tavg = lambda T_d')}):")
    doc.eq(r'\bar{T} = \frac{\sum_{j,k} (r_{k+1}^2 - r_k^2)\,\Delta z_j\,T_{jk}}{\sum_{j,k}(r_{k+1}^2 - r_k^2)\,\Delta z_j},'
           r'\qquad T_{inner} = \frac{\sum_j \Delta z_j\,T_{j,0}}{\sum_j \Delta z_j},\qquad '
           r'T_{outer} = \frac{\sum_j \Delta z_j\,T_{j,n_R-1}}{\sum_j \Delta z_j}')
    doc.h3('One step (stick in)')
    _nl(doc, [
        'Radiation and vane control at the start-of-step wall temperatures T_w = T[:, 0] (sections 4.2-4.4). The vane '
        'model solves the enclosure at the closed, 10 deg and open anchors, then picks the vane position s that holds '
        'the flux target (6.4). Returns per-ring heat leaving the block Q_rad (all rings, lower zone included), '
        f"per-ring cell heat Q_cell, liner temperatures and the linearisation dQ_rad/dT_w ({_ref('axial.py', 'def step(self, T_dist, control)', 2)}).",
        'Electrical power per active ring: P_i = Q_cell,i x eta(T_liner,in,i, q_i) x (1 - D_c) (section 4.5).',
        'Choose dt from the time and the vane state (6.3).',
        'Roof/floor sink Q_sink(T) at the start-of-step temperatures, held for the step (6.8).',
        f"Conduction update ({_ref('tinless.py', 'dQ_dTw=-_ax.last')}): storage.update_temps(T, -Q_rad, dt, sub_steps, "
        "dQ_dTw, Q_out = Q_sink). Inside each sub-step conduction is explicit, and the wall heat input is implicit "
        "and linearised about the start-of-step wall temperatures (IMEX, section 4.1; equation below).",
        't <- t + dt; test the cutoff (6.5).',
        'Record the step (6.9) and, while the stick is in, the deposition rate (4.7). Take a snapshot of the field '
        'when due (every 600 s, so in practice every step).',
    ])
    doc.p("Wall heat input to ring j within a step:")
    doc.eq(r'Q_{w,j}(\tau) =-Q_{rad,j} - \sum_{k} \frac{\partial Q_{rad,j}}{\partial T_{w,k}}\,'
           r'\left(T_{w,k}(\tau) - T_{w,k}(t)\right),\qquad t \leq \tau \leq t + \Delta t',
           where='Q_rad and its Jacobian come from the start-of-step radiation solve. In the vane model the Jacobian is '
                 'the s-blend of the three anchors\' fixed-geometry Jacobians (vanes frozen during the step).')
    fig = flowchart({
        'i': (3.7, 10.5, 'build: block T = T_max (uniform); sink = cell_loss_sink(m)\n'
                         'record 0: radiation step at t = 0', 'io'),
        'w': (3.7, 9.5, 't < end_time ?', 'dec'),
        'z': (6.4, 9.5, 'final snapshot,\ndeposit integral,\nreturn results', 'out'),
        'a': (3.7, 8.6, 'stick in ?', 'dec'),
        'c': (6.4, 8.6, 'conduction-only:\nQ_rad = 0, f = 0\ndt = dt_long', 'proc'),
        'r': (3.7, 7.35, 'Radiation + vane control at T_w(t)\n3 anchor solves -> Q_lo, Q_hi\n'
                         's = 0 | Brent on s | s = 2\nQ_rad (rings), Q_cell, T_liner, dQ/dT_w', 'loop'),
        'e': (3.7, 6.1, 'P_elec = sum Q_cell eta(T_L, q) (1 - D_c)\ndt rule (6.3)', 'proc'),
        'k': (3.7, 5.15, 'Q_sink(T) on top / bottom rings (held)', 'proc'),
        'u': (3.7, 4.2, 'update_temps: n_sub = max(sub_steps, stability)\nexplicit conduction + implicit '
                        'linearised wall flux - Q_sink', 'loop'),
        't': (3.7, 3.3, 't += dt;  T = T_new', 'proc'),
        'x': (3.7, 2.4, 'T_avg(T_new) <= T_min\nand stick in ?', 'dec'),
        'o': (6.4, 2.4, 'stick out;\nt_ps_removed = t\n(stop_at_cutoff -> exit)', 'out'),
        'g': (3.7, 1.15, 'record step k (+ deposition rate)', 'proc'),
        'n': (0.65, 5.8, 'next\nstep', 'loop'),
    }, [('i', 'w'), ('w', 'a', 'yes'), ('w', 'z', 'no'), ('a', 'r', 'yes'), ('a', 'c', 'no'), ('r', 'e'), ('e', 'k'),
        ('c', 'k', '', -0.3), ('k', 'u'), ('u', 't'), ('t', 'x'), ('x', 'o', 'yes'), ('x', 'g', 'no'), ('o', 'g'),
        ('g', 'n', '', -0.25), ('n', 'w', '', -0.25)], size=(7.4, 11.0))
    doc.fig(fig, 'Discharge time loop (TinlessSimulation.run_simulation).', width=5.7, name=_figname('6'))
    doc.review("Uniform initial temperatures: the discharge starts at a uniform T_max "
               f"({_ref('build.py', 'np.full((n_ax_tot, num_rad), float(T_max))')}) and the charge at a uniform "
               f"T_min ({_ref('build.py', 'np.full((nA, nR), float(m.T_min))')}). The cycle is not closed: the charge "
               "leaves the bore hotter than the outside and the discharge leaves it colder, and no hold is "
               "simulated. The volume average is right, but the late-discharge field sets the shade-open point, "
               "which is the sizing criterion (6.7), so T_min and the volume depend on this assumption. Confidence "
               "high (code); effect on T_min unknown, plausibly tens of K. Suggest one closed-cycle test.")
    doc.review("Quasi-steady radiation: the liner, vanes and liner-bottom pad have no heat capacity, and the vane "
               "position is frozen within a step. The wall flux inside the step follows the fixed-vane linearisation "
               f"({_ref('axial.py', 'dQ_dTw=self._pw(')}), not the controller, so the heat actually removed "
               "(storage.last_Q_wall) differs from the recorded start-of-step Q_rad. ΣQ_rad dt (used for E_rad, "
               "eta_block and heat rejection) is therefore not exactly the block's energy loss. Estimated below 1% "
               "at the YAML steps, growing with dt_scale. Neither last_Q_wall nor the update's energy residual is "
               "recorded; recording them would give an energy-closure check.")

    # ---------------------------------------------------------------- 6.3 dt
    doc.h2('6.3 Time-step rules')
    doc.p(f"dt is chosen after the step's radiation solve from the vane state just computed "
          f"({_ref('tinless.py', 'if t < dt_short_duration')}). The first matching rule wins:")
    dsl = 'dt_at_shade_limit if given, else dt_short'
    doc.table(['#', 'condition', 'dt', 'YAML / screening value'], [
        ['1', 't < dt_short_duration', 'dt_short', f"{rk['dt_short']} s / {rk['dt_short'] * dts:g} s"],
        ['2', 'vanes fully closed (f_shade == f_shade_max, s = 0)', dsl,
         f"{rk['dt_short']} s / {rk['dt_long'] * dts:g} s"],
        ['3', 'vanes regulating (f_shade > f_shade_min)', 'dt_medium', f"{rk['dt_medium']} s / {rk['dt_medium'] * dts:g} s"],
        ['4', 'vanes fully open, or no shade', 'dt_long', f"{rk['dt_long']} s / {rk['dt_long'] * dts:g} s"],
        ['-', 'stick removed (conduction-only)', 'dt_long', f"{rk['dt_long']} s / -"],
        ['-', 'always', 'min(dt, end_time - t)', ''],
    ], widths=[0.3, 3.0, 2.0, 1.8])
    doc.p("The equality test in rule 2 is exact. It works because the vane controller returns exactly s = 0 or "
          "s = 2 at its limits. There is no error control: dt depends only on the vane state. Inside each step "
          "the conduction solver uses n_sub = max(sub_steps, ceil(dt / (0.9 dt_crit))) sub-steps "
          f"({_ref('components.py', 'n_sub = max(n_sub, int(np.ceil(time_step / (safety * dt_crit))))')}).")
    doc.review(f"YAML comments vs code: tinless_model_input.yaml says dt_medium is 'used when shade is active and "
               f"not clamped' and dt_long is 'used when shade is clamped at f_shade_max'. In the code "
               f"({_ref('tinless.py', 'f_shade == self.f_shade_max')}), vanes clamped closed get dt_short (or "
               "dt_at_shade_limit), and dt_long is used when the vanes are fully OPEN and after stick removal. The "
               "comment matches only the optimizer screening call. Also dt_short_duration is not scaled by "
               "dt_scale, and the YAML end_time comment says '50 hr' but 240000 s is 66.7 h.")
    doc.review("There is no time-step error control. A dt convergence check of t_open, T_open, stick energy and "
               "deposit (e.g. halving all dt_*) has not been documented. The optimizer screens at "
               f"dt_scale = {dts:g}.")
    rows += [
        ['transient', 'dt_short / dt_medium / dt_long', f"{rk['dt_short']} / {rk['dt_medium']} / {rk['dt_long']} s "
                                                         f"(screening x{dts:g})", 'YAML; ' + _ref('tinless.py', 'if t < dt_short_duration')],
        ['transient', 'dt_short_duration', f"{rk['dt_short_duration']} s (not scaled)", 'YAML'],
        ['transient', 'dt at vanes-closed limit', f"dt_short (screening {rk['dt_long'] * dts:g} s)",
         _ref('tinless.py', 'f_shade == self.f_shade_max')],
        ['transient', 'conduction sub-steps', f"max({rk['sub_steps']}, ceil(dt / (0.9 dt_crit)))",
         _ref('components.py', 'n_sub = max(n_sub, int(np.ceil(time_step / (safety * dt_crit))))')],
        ['transient', 'snapshot interval', '600 s (every step once dt > 600 s)', _ref('sizing.py', 'snap_every=600')],
        ['transient', 'discharge end_time', f"{rk['end_time']:,} s (optimizer: max(that, 4 x duration))", 'YAML; '
         + _ref('optimize.py', "cfg['end_time'] = max(")],
    ]
    if cy is not None:
        fig = _fig_dt_shade(cy)
        if fig is not None:
            doc.fig(fig, "Example discharge (stick in). Top: cell flux with the vanes at their closed and open limits "
                         "(the reachable band), the flux actually delivered, and the target. Middle: vane position "
                         "(blend parameter s/2: 0 closed, 1 open) and the estimated vane angle. Bottom: time step "
                         "taken, coloured by the rule that chose it.", width=6.6, name=_figname('6'))

    # ---------------------------------------------------------------- 6.4 shade control
    doc.h2('6.4 Shade control over time')
    doc.p("The vanes are re-positioned from scratch at every step from the current wall temperatures. There are no "
          "controller dynamics: no actuator rate limit, deadband, lag or hysteresis. Within a step the position is "
          f"frozen. With the rotating-vane model ({_ref('axial.py', 'class VaneHybridRadiation')}):")
    doc.eq(r'Q_{tgt} = q_{tgt}\,A_{cell},\quad A_{cell} = 4\,w\,L\,(1-f_{bare})\,f_{active},\qquad '
           r's = \left\{0\ \mathrm{if}\ Q_{lo} \geq Q_{tgt};\ \ 2\ \mathrm{if}\ Q_{hi} \leq Q_{tgt};\ \ '
           r'\mathrm{root\ of}\ \sum Q_{cell}(s) = Q_{tgt}\ \mathrm{otherwise}\right\}',
           where='Q_lo, Q_hi: total cell heat with the vanes closed / open; s in [0, 2] blends the closed, 10 deg and '
                 'open anchors piecewise-linearly; reported f_shade = 1 - s/2 (1 closed, 0 open).')
    doc.bl([
        f"Root found with Brent's method on [0, 2], xtol 1e-9 ({_ref('axial.py', 's = brentq(lambda x: qcell_tot(x) - Q_tgt')}).",
        'Three phases of a discharge: (1) hot block, vanes closed and flux above target (the cells are over-driven); '
        '(2) vanes regulating, flux held at target, constant power; (3) vanes fully open, flux and power fall until '
        'the stick is removed. Shade-open sizing (6.7) puts the end of phase 2 at T_min, so phase 3 is short.',
        'build.py sets f_shade_min, f_shade_max = 0, 1 for the vane model '
        f"({_ref('build.py', 'sim.f_shade_min, sim.f_shade_max = f_shade_min, f_shade_max')}). The legacy "
        'perimeter-based limits apply only to the porous and local models.',
    ])
    doc.review("Vane control is ideal and instantaneous: any position in [closed, open] is reached each step, and "
               "nothing limits the actuator speed or resolution, the vane-angle accuracy (the angle is only a "
               "plotting estimate) or the temperature limits on the vanes. While the vanes are closed early on the "
               "cells see more than the target flux, and this phase is not constrained except through peak heat "
               "rejection and peak power.")
    rows += [['shade control', 'vane position solve', 'Brent on s in [0, 2], xtol 1e-9',
              _ref('axial.py', 's = brentq(lambda x: qcell_tot(x) - Q_tgt')],
             ['shade control', 'vane limits (f_shade)', '0 (open) .. 1 (closed)',
              _ref('build.py', 'sim.f_shade_min, sim.f_shade_max = f_shade_min, f_shade_max')]]

    # ---------------------------------------------------------------- 6.5 cutoff
    doc.h2('6.5 Stick removal and the conduction-only tail')
    doc.p(f"After each step ({_ref('tinless.py', '_Tavg(T_new) <= T_ps_cutoff')}): if the stick is in and the "
          "volume-average temperature of the new field is at or below T_min (T_ps_cutoff = storage.T_min, "
          f"{_ref('build.py', 'run_kwargs = dict(')}), the stick is removed and t_ps_removed = t. The test is at "
          "step ends only, so T_avg overshoots below T_min by up to one step's drop (a few K). With stop_at_cutoff the "
          "loop ends there. Otherwise it continues to end_time with no radiation exchange at all: Q_rad = 0, "
          "f_shade = 0, dt = dt_long, the liner is recorded at the wall temperature, and the roof/floor sink stays "
          "on. If T_min is never reached, the stick stays in until end_time.")
    doc.review("The step that crosses the cutoff is dropped from the stick energy. ps_active is set False BEFORE the "
               f"step is recorded ({_ref('tinless.py', 'ps_active_arr.append(_ps_active)')}), so that step is logged "
               "as stick-out even though it produced power. With stop_at_cutoff (all optimizer runs) the loop "
               f"breaks before recording it at all ({_ref('tinless.py', 'if stop_at_cutoff:')}). tank.stick_from_run "
               f"({_ref('tank.py', 't_end = float(r[')}) and optimize.discharge_metrics mask energy by ps_active "
               "but use t_end = t_ps_removed, which includes the step's duration. So P_stick = E_e/t_end is biased "
               "low by about dt_last/t_end: roughly 0.5-1% at YAML steps and up to about 2% at screening steps, "
               "the same order as the sizing tolerance. The step's deposition is also 0. The axial extras keep the "
               "step (_on = ... or t_ps_removed == t), which is inconsistent. Confidence high on the behaviour; "
               "magnitude estimated.")

    # ---------------------------------------------------------------- 6.6 charge
    doc.h2('6.6 Charge loop')
    doc.p(("TinlessSimulation.run_heating_simulation", {'mono': True}), f" ({_ref('tinless.py', 'def run_heating_simulation')}) "
          "on the heater cell (4.6). Callers: cycle._charge and optimize._charge / heater_search, both with the "
          f"roof/floor sink ({_ref('cycle.py', 'for_storage(hm.storage)')}).")
    doc.bl([
        f"Fixed dt = YAML dt_short ({rk['dt_short']} s; x dt_scale in optimizer screening), n_steps = "
        "int(end_time / dt). There are no dt_short/medium/long rules.",
        'Each step: solve T_h and the per-ring Q at the start-of-step wall temperatures, evaluate the sink, then '
        'update_temps(T, +Q, dt, sub_steps, Q_out = sink). Unlike the discharge, the heater flux is fully explicit '
        '(no Jacobian): it is held at its start-of-step value through all sub-steps.',
        f"Stop after the first step with volume-average T_avg >= T_max ({_ref('tinless.py', 'if T_avg_list[-1] >= T_block_max')}), "
        'or at n_steps. The callers take the charge time by linear interpolation of t at T_avg = T_max between the '
        f"last two records ({_ref('optimize.py', 't_c = float(np.interp(hm.T_max, Ta[-2:], t[-2:]))')}). If T_max is "
        'not reached: t_charge = inf (optimizer) or the end time (cycle).',
        'The optimizer runs the charge to 1.5 x the target charge time '
        f"({_ref('optimize.py', 'end_hr = 1.5 * target')}). The cycle runs it to the cycle length minus the discharge.",
        'Recorded: time, Q_heater (record 0 = 0), T_heater, T_avg, T_inner, T_outer, T_liner (always 0 for the '
        'heater cell: it has no liner), Q_sink, field snapshots.',
    ])
    fig = flowchart({
        'i': (3.7, 6.5, 'build_heater(m): same V and mesh, r_in = a_h r_h\nT = T_min (uniform); sink from the '
                        'stick plant', 'io'),
        'w': (3.7, 5.5, 'step < int(end_time / dt) ?', 'dec'),
        'z': (6.5, 5.5, 'end\n(not charged)', 'out'),
        'h': (3.7, 4.3, 'solve T_h at T_w(t):\nmax T_w >= T_h,max -> Q = 0\nsum Q(T_h,max) >= P_max -> Brent: '
                        'sum Q(T_h) = P_max\nelse T_h = T_h,max', 'loop'),
        'u': (3.7, 3.05, 'Q_sink(T);  update_temps(T, +Q_i explicit, dt, - Q_sink)', 'loop'),
        'r': (3.7, 2.15, 'record t, sum Q, T_h, T_avg, T_inner, T_outer, Q_sink', 'proc'),
        'x': (3.7, 1.2, 'T_avg >= T_max ?', 'dec'),
        'o': (6.5, 1.2, "'fully charged';\ncaller interpolates\nt_charge", 'out'),
        'n': (0.6, 3.4, 'next\nstep', 'loop'),
    }, [('i', 'w'), ('w', 'h', 'yes'), ('w', 'z', 'no'), ('h', 'u'), ('u', 'r'), ('r', 'x'), ('x', 'o', 'yes'),
        ('x', 'n', 'no', -0.25), ('n', 'w', '', -0.25)], size=(7.4, 7.0))
    doc.fig(fig, 'Charge time loop (TinlessSimulation.run_heating_simulation).', width=5.2, name=_figname('6'))
    if cy is not None:
        fig = _fig_charge(cy)
        if fig is not None:
            doc.fig(fig, "Example charge (heater cell). Left: heater, cavity-wall, block-average and outer "
                         "temperatures. Right: heater power with the cap, and the roof + floor loss. Phase 1 (power "
                         "cap) ends when the heater reaches its temperature limit; in phase 2 the power falls as the "
                         "wall warms.", width=6.8, name=_figname('6'))
    doc.review("The heater flux is explicit in the charge: computed once per step and held. A rough estimate of the "
               "linearised wall conductance near 2400 K is about 500 W/m2K per wall area at area ratio 4.5. Against "
               "a wall-node capacity of about 4e5 J/m2K that is a time constant of about 800 s, so dt = 600 s "
               f"(1200 s screening) is inside the explicit limit but not by a wide margin "
               f"({_ref('tinless.py', 'T_h, Q_segs = _solve_T_heater(T[:, 0])')}). It is first-order accurate near "
               "phase 2, and the charge time is interpolated only over the last step. Medium confidence (estimate).")
    rows += [
        ['charge', 'charge time step', f"dt_short = {rk['dt_short']} s (x dt_scale)", 'build.py; '
         + _ref('optimize.py', "kw = dict(hm.run_kwargs, dt=hm.run_kwargs['dt'] * dt_scale)")],
        ['charge', 'charge end time (optimizer)', '1.5 x charge_hr', _ref('optimize.py', 'end_hr = 1.5 * target')],
        ['charge', 'charge stop', 'T_avg >= T_max; t_charge by linear interpolation over the last step',
         _ref('tinless.py', 'if T_avg_list[-1] >= T_block_max')],
        ['discharge', 'stick-removal cutoff', 'T_avg <= T_min, tested at step ends', _ref('tinless.py', '_Tavg(T_new) <= T_ps_cutoff')],
        ['discharge', 'initial block state', 'uniform T_max', _ref('build.py', 'np.full((n_ax_tot, num_rad), float(T_max))')],
    ]

    # ---------------------------------------------------------------- 6.7 sizing
    doc.h2('6.7 Shade-open sizing loop')
    tolT, tolt = _dflt(S.shade_open_resize, 'tol_T'), _dflt(S.shade_open_resize, 'tol_t')
    mi, sd = _dflt(S.shade_open_resize, 'max_iter'), _dflt(S.shade_open_resize, 'step_down')
    nfit, tol_o = _dflt(S.shade_open_point, 'n_fit'), _dflt(S.shade_open_point, 'tol')
    so_ = opt.get('sizing', {})
    doc.p("Goal: with T_max fixed, choose T_min and the block volume so that the vanes reach fully open exactly when "
          "the volume-average block temperature reaches T_min, at t = duration. Power is constant while the vanes "
          "regulate, so the design duration is then delivered at the target flux, and T_min is as low as the "
          "target flux allows. ", ('sizing.shade_open_resize', {'mono': True}),
          f" ({_ref('sizing.py', 'def shade_open_resize')}) iterates full discharge runs:")
    doc.h3('Finding the opening point in a run')
    doc.p(('sizing.shade_open_point', {'mono': True}), f" ({_ref('sizing.py', 'def shade_open_point')}). Using the "
          f"stick-in records, 'open' means f_shade <= f_shade_min + {tol_o:g}:")
    doc.bl([
        f"No stick-in record with the vanes not fully open: raise FluxUnreachable ({_ref('sizing.py', 'raise FluxUnreachable')}). "
        'The target flux cannot be reached even fully open, and the optimizer marks the design infeasible.',
        'Seen (the vanes open during the run): i = first record after the last non-open one. t_open = t_i, refined by '
        'extrapolating the falling cell flux over the next record back to the target, and clipped to '
        '[t_(i-1), t_i]. T_open is interpolated in time:',
    ])
    doc.eq(r't_{open} = \mathrm{clip}\left(t_i - \frac{q_{tgt} - q_i}{(q_i - q_{i+1})/(t_{i+1} - t_i)},\; t_{i-1},\; t_i\right)')
    doc.bl([f"Not seen (still regulating at the cutoff, or at end_time): straight-line fits over the last "
            f"{nfit} regulating records ({_ref('sizing.py', 'b, a = np.polyfit')}), then f_shade = 0 solved for T:"])
    doc.eq(r'f = a + b\,\bar{T}\;\Rightarrow\; T_{open} = \frac{f_{min} - a}{b},\qquad '
           r'\bar{T} = c + d\,t\;\Rightarrow\; t_{open} = \frac{T_{open} - c}{d}')
    doc.bl(['Never regulated (vanes closed to the end, fewer than 2 regulating records): None.'])
    doc.h3('Update')
    doc.eq(r'T_{min}^{(n+1)} = T_{open}^{(n)},\qquad V^{(n+1)} = V^{(n)}\,\frac{t_{req}}{t_{open}^{(n)}},\qquad '
           r'\mathrm{volume\_factor} = V^{(n+1)}/V_{build}(T_{max} - T_{min}^{(n+1)})',
           where='V_build: the energy-basis volume build_tinless gives for the new swing at volume_factor 1 '
                 '(section 3); t_req = storage.duration_hr.')
    doc.bl([
        f"Converged when |T_open - T_min| <= tol_T and |t_open / t_req - 1| <= tol_t "
        f"({_ref('sizing.py', 'if abs(dT) <= tol_T and abs(dt_frac) <= tol_t')}).",
        f"Vanes never opened: T_min drops by step_down ({sd:g} K) and volume_factor resets to 1.",
        'T_min below T_min_floor: stop with below_floor = True. max_iter reached: stop with converged = False. In '
        'both cases the last run is returned.',
        'This is plain successive substitution: no damping, no secant or Newton step. The extrapolated part shrinks '
        'to zero as the loop converges.',
    ])
    doc.table(['setting', 'function default', 'optimizer'], [
        ['tol_T', f"{tolT} K", f"{so_.get('tol_T')} K"],
        ['tol_t', f"{tolt}", f"{so_.get('tol_t')}"],
        ['max_iter', f"{mi}", f"{so_.get('max_iter')}"],
        ['step_down', f"{sd} K", f"{sd} K"],
        ['T_min_floor', 'None', f"{opt.get('geometry', {}).get('T_min_floor_C')} C"],
        ['first guess T_min', 'cfg storage.T_min', f"T_max - {so_.get('swing_guess')} K (swing_guess)"],
        ['n_fit (extrapolation)', f"{nfit} records", f"{nfit}"],
        ['open tolerance on f_shade', f"{tol_o:g}", f"{tol_o:g}"],
        ['discharge runs', 'run_discharge defaults', f"dt_scale {dts:g}, dt_at_shade_limit 2 x dt_long, stop_at_cutoff"],
    ], widths=[2.0, 2.2, 2.9])
    fig = flowchart({
        'i': (3.5, 7.7, 'cfg: T_max = T_min0 + swing0 (fixed); volume_factor', 'io'),
        'r': (3.5, 6.8, 'run_discharge(cfg)  (full transient, 6.2)', 'loop'),
        'p': (3.5, 5.6, 'shade_open_point:\nseen -> flux back-extrapolation\nnot seen -> linear fits (last 8)\n'
                        'none regulating -> None', 'proc'),
        'f': (6.4, 5.6, 'FluxUnreachable\n(infeasible)', 'out'),
        'nn': (1.2, 3.1, 'None:\nT_next =\nT_min - 100 K', 'proc'),
        'c': (3.5, 4.3, '|T_open - T_min| <= tol_T and\n|t_open/t_req - 1| <= tol_t ?', 'dec'),
        'ok': (6.4, 4.3, 'converged:\nreturn last run', 'out'),
        'b': (3.5, 3.1, 'T_next < T_min_floor\nor it = max_iter ?', 'dec'),
        'st': (6.4, 3.1, 'stop (below_floor /\nnot converged)', 'out'),
        'u': (3.5, 1.75, 'T_min = T_next; swing = T_max - T_next\nV = V_block t_req / t_open\n'
                         'volume_factor = V / V_build(new swing)', 'proc'),
        'L': (0.35, 1.75, 'next\npass', 'loop'),
    }, [('i', 'r'), ('r', 'p'), ('p', 'f', 'all open'), ('p', 'nn'), ('p', 'c', 'point'),
        ('c', 'ok', 'yes'), ('c', 'b', 'no: T_next = T_open'), ('nn', 'b'), ('b', 'st', 'yes'),
        ('b', 'u', 'no'), ('u', 'L'), ('L', 'r', '', -0.3)], size=(7.4, 8.2))
    doc.fig(fig, 'Shade-open sizing loop (sizing.shade_open_resize).', width=5.4, name=_figname('6'))
    if cy is not None and getattr(cy, 'sizing', None) is not None and cy.sizing.history:
        hist = cy.sizing.history
        doc.p(f"Example design: {len(hist)} passes, converged = {cy.sizing.converged}.")
        doc.table(['pass', 'T_min [C]', 'swing [K]', 'volume factor', 'r_out [m]', 'T_open [C]', 't_open [h]',
                   'opening'], [[str(h['it']), f"{h['T_min'] - C2K:.1f}", f"{h['swing']:.1f}", f"{h['volume_factor']:.4f}",
                                f"{h['r_out']:.4f}", '-' if not np.isfinite(h['T_open']) else f"{h['T_open'] - C2K:.1f}",
                                '-' if not np.isfinite(h['t_open_hr']) else f"{h['t_open_hr']:.2f}",
                                {True: 'seen', False: 'extrapolated', None: 'none'}[h['seen']]] for h in hist],
                   widths=[0.5, 0.9, 0.85, 1.0, 0.9, 0.9, 0.85, 1.2], size=8)
        fig = _fig_sizing(cy)
        if fig is not None:
            doc.fig(fig, 'Sizing passes of the example design: T_min tried and the opening temperature found (left), '
                         'opening time against the target duration (right).', width=6.2, name=_figname('6'))
    doc.review("Indexing in shade_open_point: f_shade[i] and Q_cell[i] describe the state at time[i-1] (6.9) but are "
               f"paired with time[i] and T_avg[i] ({_ref('sizing.py', 'is_open = on &')}). The detected opening is "
               "therefore about one step late, and T_open about one step's drop too low: about 2.5-5 K at YAML "
               "steps against tol_T of 2-3 K. The loop converges on this biased definition, so the final design's "
               "vanes probably open slightly before T_min, with a brief power sag at the end. Confidence high on the "
               "indexing; magnitude estimated.")
    doc.review("Sizing loop robustness: the volume update assumes t_open ~ V (true while regulating), and the T_min "
               "update ignores how V (r_out, radial profile) shifts T_open. The polyfit slopes are not guarded "
               f"(b or d near 0), nor is the fit checked for non-monotone shade histories ({_ref('sizing.py', 'T_open = (m.f_shade_min - a) / b')}). "
               "The None branch steps a fixed 100 K and discards the volume correction. When max_iter is reached the "
               "unconverged last pass is used (the optimizer records sizing_converged but does not penalise it).")
    rows += [
        ['sizing', 'tol_T / tol_t (default)', f"{tolT} K / {tolt}", _ref('sizing.py', 'def shade_open_resize')],
        ['sizing', 'tol_T / tol_t (optimizer)', f"{so_.get('tol_T')} K / {so_.get('tol_t')}", 'optimize.OPT_DEFAULTS'],
        ['sizing', 'max_iter (default / optimizer)', f"{mi} / {so_.get('max_iter')}", _ref('sizing.py', 'def shade_open_resize')],
        ['sizing', 'step_down when vanes never open', f"{sd} K", _ref('sizing.py', 'T_next = T_min - step_down')],
        ['sizing', 'extrapolation window', f"last {nfit} regulating records, linear", _ref('sizing.py', 'b, a = np.polyfit')],
        ['sizing', 'first guess T_min (optimizer)', f"T_max - {so_.get('swing_guess')} K", _ref('optimize.py', "T_lo = d.get('T_min_guess_C'")],
    ]

    # ---------------------------------------------------------------- 6.8 sink
    doc.h2('6.8 Roof and floor losses in the transient')
    doc.p("The per-area roof and floor losses q_top(T), q_bot(T) of the plant are derived in section 5. "
          f"cell_loss_sink ({_ref('tank.py', 'def cell_loss_sink')}) builds them before the run from a provisional "
          "plant: N from the required energy at an assumed 45% heat-to-electricity ratio, tabulated at 40 hot-face "
          "temperatures from T_min - 150 C to T_max + 100 C. CellLossSink.for_storage "
          f"({_ref('tank.py', 'def for_storage')}) turns them into a sink on THIS block model:")
    doc.eq(r'Q_{n_A-1,k} = q_{top}(T_{n_A-1,k})\,a_k,\qquad Q_{0,k} = q_{bot}(T_{0,k})\,a_k,\qquad '
           r'a_k = \pi\,(r_{k+1}^2 - r_k^2)\,\frac{1}{\phi_{fill}}',
           where='k: radial node; top ring n_A-1 loses to the roof, bottom ring 0 to the floor; phi_fill = graphite '
                 'volume / (hot-zone plan area x block height), so the cavities\' and margin\'s share of roof/floor '
                 'area is drawn from the graphite. Each node uses its own temperature (np.interp, held flat outside '
                 'the table).')
    doc.bl([
        'Evaluated at the start-of-step field and held for the step (explicit). update_temps subtracts it from each '
        f"node ({_ref('components.py', 'if Q_node is not None:')}).",
        'The discharge uses it during the stick-in phase and the conduction-only tail. The charge uses the same object '
        'mapped onto the heater cell (same volume and radial rings, so the same total area and sink).',
        'Side-wall and edge losses are not applied in the transients. They reach only perimeter cells and are '
        'charged afterwards in the tank model (section 5).',
        "The recorded Q_sink is the total sink held over each step. The stick-in part of its time integral is "
        "E_sink, the discharge's roof + floor loss in the ledger.",
    ])
    if cy is not None:
        on = np.asarray(cy.r_dis['ps_active'][1:], bool)
        dtd = np.diff(cy.r_dis['time'])
        E_sd = float(np.sum((np.asarray(cy.r_dis['Q_sink'][1:]) * dtd)[on]))
        E_sc = float(np.sum(np.asarray(cy.r_chg['Q_sink'][1:]) * np.diff(cy.r_chg['time'])))
        E_rd = float(np.sum((np.asarray(cy.r_dis['Q_rad'][1:]) * dtd)[on]))
        doc.p(f"Example: roof + floor loss {E_sd / 3.6e6:,.0f} kWh per stick cell during the discharge "
              f"({E_sd / max(E_rd, 1e-9) * 100:.1f}% of the heat into the stick) and {E_sc / 3.6e6:,.0f} kWh per "
              "heater cell during the charge.")
    doc.review("Sink approximations: it comes from a 1-D plant model at a uniform hot face, applied node by node at "
               "start-of-step temperatures. It is built from a provisional plant (heat-to-electricity 0.45, N from "
               f"energy; {_ref('tank.py', 'N = int(math.ceil(E_req / (eta_e')}) and never updated to the final "
               "plant. The whole top-face loss comes out of one ring of height H/n_A. These are reasonable "
               "first-order choices, but the final plant's sink (TankResult.cell_sink) could be compared and "
               "iterated.")
    rows += [
        ['sink', 'loss table', '40 points, T_min - 150 C .. T_max + 100 C', _ref('tank.py', 'T = np.linspace(T_lo - 150.0')],
        ['sink', 'provisional plant ratio eta_e', f"{_dflt(__import__('thermal.tank', fromlist=['x']).cell_loss_sink, 'eta_e')}",
         _ref('tank.py', 'def cell_loss_sink')],
        ['sink', 'time treatment', 'start-of-step, held over the step', _ref('tinless.py', '_Qs = None if Q_sink_fn is None')],
    ]

    # ---------------------------------------------------------------- 6.9 outputs
    doc.h2('6.9 Recorded outputs and timing conventions')
    doc.p("Record k >= 1 is written at the END of step k (time t_k). Fluxes, vane position, liner and vane "
          "temperatures and the deposition rate were computed at the START of the step (state at t_(k-1)) and held "
          "through it. Block temperatures are the end-of-step field. So sum_k P_k (t_k - t_(k-1)) is a left-Riemann "
          "energy integral, and records 0 and 1 hold identical fluxes.")
    fig = _fig_timing()
    if fig is not None:
        doc.fig(fig, 'Timing of a record. The step from t_(k-1) to t_k uses fluxes evaluated at the start state; '
                     'record k stores those fluxes with the end-of-step temperatures and time.', width=6.2,
                name=_figname('6'))
    doc.table(['key (discharge)', 'meaning', 'timing'], [
        ['time', 's, step ends', 'end'],
        ['Q_rad', 'W, total heat leaving the block wall (all rings) = heat into liner + liner-bottom pad', 'start'],
        ['Q_cell', 'W, absorbed by the active cell area (all bands)', 'start'],
        ['P_elec', 'W, DC electricity (eta x (1 - D_c))', 'start'],
        ['f_shade, vane_angle', '1 - s/2 (1 closed, 0 open); approximate vane angle [deg]', 'start'],
        ['Q_cell_min, Q_cell_max', 'W, cell heat with the vanes closed / open (reachable band)', 'start'],
        ['T_avg, T_inner, T_outer', 'K, volume-average; height-averaged inner wall / outer node', 'end'],
        ['T_liner, T_liner_out', 'K, mean liner inner / outer face over the active rings', 'start'],
        ['T_sh_in, T_sh_out', 'K, mean vane outer / inner face', 'start'],
        ['Q_pad, T_R, T_LB, Q_cap, q_cap, Q_LB_cells', 'liner-bottom model outputs (section 4.4)', 'start'],
        ['T_wall_spread, Q_rad_low, Q_vert_rad, Q_vert_cond', 'axial diagnostics (wall spread, lower-zone heat, '
                                                              'vertical transport)', 'start'],
        ['Q_sink', 'W, roof + floor sink held over the step', 'held'],
        ['ps_active, t_ps_removed', 'stick in after this step\'s cutoff test; removal time', 'end'],
        ['dep_rate; dep_C_nm, dep_metal_nm, dep_total_nm', 'nm/yr per record; nm per run', 'start; integral'],
        ['snap_times, temp_snapshots, liner_out_snaps', 'full field (n_A x n_R) and liner outer faces at snapshots', 'end'],
    ], widths=[2.3, 3.9, 0.9], size=8)
    doc.p("Not recorded: the heat actually applied at the wall over the step (storage.last_Q_wall), the sub-step "
          "count (last_n_sub), the update's internal-energy change, and the per-step waste heat (computed downstream "
          "as Q_rad - P_elec, the stick heat rejection used for dry-cooler sizing and fan energy, section 9).")
    doc.review("Records 0 and 1 carry identical fluxes, both evaluated at t = 0. This is harmless for left-Riemann "
               "sums but slightly biases the trapezoid deposition integral and plots. The charge run's T_liner is "
               "always 0, and print_interval counts steps in the charge but seconds in the discharge (cosmetic).")

    # ---------------------------------------------------------------- 6.10 example fields
    doc.h2('6.10 Example runs: block temperature fields')
    if cy is None:
        doc.p('No example design was run for this build; the figures of this section are produced when the document '
              'is built with the example design (section 8).')
    else:
        fig = _fig_fields(cy)
        if fig is not None:
            doc.fig(fig, 'Block temperature fields (radius horizontal, height vertical, cavity wall on the left). '
                         'Charge end: heater cell. Discharge: stick cell at the start of regulation, half-way, and '
                         'at stick removal. Dashed line: liner-bottom plane (the stick and liner hang in the active '
                         'zone above it).', width=6.8, name=_figname('6'))
        fig = _fig_rings(cy)
        if fig is not None:
            doc.fig(fig, 'Example discharge, ring-mean temperatures (radially volume-weighted) of the bottom ring, the '
                         'ring just below the liner bottom, the ring just above it, and the top ring, with the '
                         'block average; vertical line: stick removal. The tail after it is the conduction-only phase '
                         'with roof + floor losses.', width=6.4, name=_figname('6'))
        r = cy.r_dis
        on = np.asarray(r['ps_active'], bool)
        doc.kv([
            ('discharge records', f"{len(r['time'])} ({int(on.sum())} stick-in)"),
            ('stick removed', f"{(r['t_ps_removed'] or np.nan) / 3600:.2f} h"),
            ('block average at removal', f"{np.asarray(r['T_avg'])[on][-1] - C2K:.1f} C (T_min "
                                         f"{cy.T_min_C:.1f} C)"),
            ('charge records', f"{len(cy.r_chg['time'])}, charge time {cy.t_charge_hr:.2f} h"),
        ], w0=2.4)
    return rows


# --------------------------------------------------------------------------
# example figures
# --------------------------------------------------------------------------
def _fig_dt_shade(cy):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    r, m = cy.r_dis, cy.m
    t = np.asarray(r['time'], float)
    on = np.asarray(r['ps_active'], bool)
    A = _A_act_cm2(m)
    k = np.where(on)[0]
    if k.size < 3:
        return None
    th = t[k] / 3600
    fig, ax = plt.subplots(3, 1, figsize=(8.5, 8.2), sharex=True)
    a = ax[0]
    qmin = np.asarray(r.get('Q_cell_min', np.full_like(t, np.nan)), float)[k] / A
    qmax = np.asarray(r.get('Q_cell_max', np.full_like(t, np.nan)), float)[k] / A
    a.fill_between(th, qmin, qmax, color='C0', alpha=0.15, label='reachable (closed .. open)')
    a.plot(th, qmin, c='C0', lw=0.8, ls='--', label='vanes closed')
    a.plot(th, qmax, c='C0', lw=0.8, ls=':', label='vanes open')
    a.plot(th, np.asarray(r['Q_cell'], float)[k] / A, c='k', lw=1.3, label='delivered')
    a.axhline(m.avg_flux, c='C3', lw=0.9, label='target')
    a.set_ylabel('cell flux [W/cm2]')
    a.set_ylim(0, min(np.nanmax(qmax) * 1.05, m.avg_flux * 4) if np.isfinite(np.nanmax(qmax)) else None)
    a.legend(fontsize=7, ncol=3)
    a = ax[1]
    a.plot(th, 1 - np.asarray(r['f_shade'], float)[k], c='C1', label='opening s/2 (0 closed, 1 open)')
    a.set_ylabel('vane opening s/2')
    a.set_ylim(-0.05, 1.05)
    if 'vane_angle' in r:
        a2 = a.twinx()
        a2.plot(th, np.asarray(r['vane_angle'], float)[k], c='C4', lw=0.9, ls='--', label='vane angle (estimate)')
        a2.set_ylabel('vane angle [deg]')
        a2.set_ylim(-4, 94)
        a2.legend(fontsize=7, loc='center right')
    a.legend(fontsize=7, loc='center left')
    a = ax[2]
    dt = np.diff(t)
    tk = t[1:]
    fs = np.asarray(r['f_shade'], float)[1:]
    onk = np.asarray(r['ps_active'], bool)[1:] | (t[1:] == (r['t_ps_removed'] or -1))
    sel = onk & (tk <= (r['t_ps_removed'] or tk[-1]) + 1)
    cat = np.where(tk - dt < float(cy.cfg.get('dt_short_duration', 3600)), 0,
                   np.where(fs >= m.f_shade_max, 1, np.where(fs > m.f_shade_min, 2, 3)))
    names = ['initial period (dt_short)', 'vanes closed', 'vanes regulating (dt_medium)', 'vanes open (dt_long)']
    for c in range(4):
        s = sel & (cat == c)
        if s.any():
            a.scatter(tk[s] / 3600, dt[s], s=6, c=f"C{c}", label=names[c])
    a.set_ylabel('time step [s]')
    a.set_xlabel('time in discharge [h]')
    a.legend(fontsize=7, ncol=2)
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def _fig_charge(cy):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    rc, hm = cy.r_chg, cy.hm
    t = np.asarray(rc['time'], float) / 3600
    Th = np.asarray(rc['T_heater'], float)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))
    a = ax[0]
    a.plot(t, _C(Th), c='C3', label='heater')
    a.plot(t, _C(rc['T_inner']), c='C1', label='cavity wall (height avg.)')
    a.plot(t, _C(rc['T_avg']), c='k', label='block average')
    a.plot(t, _C(rc['T_outer']), c='C0', label='block outer')
    a.axhline(hm.T_max - C2K, c='k', ls=':', lw=0.8)
    a.axhline(hm.heater.max_temperature - C2K, c='C3', ls=':', lw=0.8)
    a.set_ylabel('C')
    a = ax[1]
    a.plot(t, np.asarray(rc['Q_heater'], float) / 1e3, c='C3', label='heater power')
    a.axhline(hm.heater.max_power / 1e3, c='C3', ls=':', lw=0.8, label='power cap')
    a.plot(t, np.asarray(rc['Q_sink'], float) / 1e3, c='C7', label='roof + floor loss')
    a.set_ylabel('kW per heater cell')
    ph = np.where(Th >= hm.heater.max_temperature - 1.0)[0]
    for a in ax:
        if ph.size:
            a.axvline(t[ph[0]], c='0.5', ls='--', lw=0.8)
            a.text(t[ph[0]], a.get_ylim()[1], ' phase 2', fontsize=7, va='top', color='0.35')
        a.set_xlabel('time in charge [h]')
        a.grid(alpha=0.3)
        a.legend(fontsize=7)
    fig.tight_layout()
    return fig


def _fig_sizing(cy):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    h = cy.sizing.history
    it = [x['it'] for x in h]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    ax[0].plot(it, [x['T_min'] - C2K for x in h], 'o-', c='C0', label='T_min used')
    ax[0].plot(it, [x['T_open'] - C2K for x in h], 's--', c='C1', label='T_open found')
    ax[0].set_ylabel('C')
    ax[1].plot(it, [x['t_open_hr'] for x in h], 'o-', c='C2', label='t_open')
    dur = float(cy.cfg['storage']['duration_hr'])
    ax[1].axhline(dur, c='k', ls=':', lw=0.8, label='target duration')
    ax[1].set_ylabel('h')
    for a in ax:
        a.set_xlabel('pass')
        a.set_xticks(it)
        a.grid(alpha=0.3)
        a.legend(fontsize=7)
    fig.tight_layout()
    return fig


def _fig_timing():
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    fig, ax = plt.subplots(figsize=(8, 2.3))
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.1, 2.6)
    ax.axis('off')
    x0, x1 = 1.6, 8.0
    ax.annotate('', xy=(9.8, 0.5), xytext=(0.2, 0.5), arrowprops=dict(arrowstyle='-|>', lw=1))
    for x, lab in ((x0, 't_(k-1)'), (x1, 't_k')):
        ax.plot([x, x], [0.38, 0.62], c='k')
        ax.text(x, 0.08, lab, ha='center', fontsize=9)
    ax.add_patch(plt.Rectangle((x0, 0.8), x1 - x0, 0.5, fc='#E8EEF7', ec='#2F5F9E'))
    ax.text(0.5 * (x0 + x1), 1.05, 'step k: fluxes of state T(t_(k-1)) held;  conduction sub-steps',
            ha='center', va='center', fontsize=8)
    ax.text(x0, 2.05, 'evaluated here: radiation, vane\nposition, dt, Q_sink, deposition rate', ha='center',
            va='center', fontsize=8, color='#2F5F9E')
    ax.text(x1, 2.05, 'record k: time t_k, field T(t_k), T_avg(t_k)\n+ fluxes / f_shade of the start state',
            ha='center', va='center', fontsize=8, color='#8E2C6E')
    ax.annotate('', xy=(x0, 1.33), xytext=(x0, 1.75), arrowprops=dict(arrowstyle='-|>', lw=0.8, color='#2F5F9E'))
    ax.annotate('', xy=(x1, 1.75), xytext=(x1, 1.33), arrowprops=dict(arrowstyle='-|>', lw=0.8, color='#8E2C6E'))
    return fig


def _edges_z(st):
    dz = st.axial_dz() if hasattr(st, 'axial_dz') else st.dz
    return np.concatenate([[0.0], np.cumsum(dz)])


def _fig_fields(cy):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    r, m, hm = cy.r_dis, cy.m, cy.hm
    snaps = np.asarray(r['temp_snapshots'], float)
    ts = np.asarray(r['snap_times'], float)
    if snaps.ndim != 3 or len(ts) < 3:
        return None
    t = np.asarray(r['time'], float)
    fs = np.asarray(r['f_shade'], float)
    on = np.asarray(r['ps_active'], bool)
    t_rem = r['t_ps_removed'] or t[on][-1]
    reg = np.where(on & (fs < m.f_shade_max - 1e-9))[0]
    t_reg = t[reg[0]] if reg.size else 0.0
    picks = [('discharge: regulation starts', t_reg), ('discharge: half-way', 0.5 * (t_reg + t_rem)),
             ('discharge: stick removed', t_rem)]
    panels = []
    cs = np.asarray(cy.r_chg['temp_snapshots'], float)
    if cs.ndim == 3:
        panels.append(('charge end (heater cell)', cs[-1], hm.storage))
    for lab, tt in picks:
        i = int(np.argmin(np.abs(ts - tt)))
        panels.append((f"{lab}, {ts[i] / 3600:.1f} h", snaps[i], m.storage))
    vmin = min(float(p[1].min()) for p in panels) - C2K
    vmax = max(float(p[1].max()) for p in panels) - C2K
    fig, ax = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 4.6), sharey=True)
    ax = np.atleast_1d(ax)
    z_lb = None
    try:
        z_lb = float(m.mesh.z_act0) if m.mesh.n_low else None
    except Exception:
        pass
    pc = None
    for a, (lab, T, st) in zip(ax, panels):
        rr = st.block.rad_in + np.arange(st.num_rad_segments + 1) * (st.block.rad_out - st.block.rad_in) / st.num_rad_segments
        zz = _edges_z(st)
        pc = a.pcolormesh(rr, zz, T - C2K, cmap='inferno', vmin=vmin, vmax=vmax, shading='flat')
        if z_lb is not None:
            a.axhline(z_lb, c='w', ls='--', lw=0.8)
        a.set_title(lab, fontsize=8)
        a.set_xlabel('r [m]')
    ax[0].set_ylabel('z [m] (0 = cavity floor)')
    fig.colorbar(pc, ax=list(ax), label='C', shrink=0.9)
    return fig


def _fig_rings(cy):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    r, m = cy.r_dis, cy.m
    snaps = np.asarray(r['temp_snapshots'], float)
    ts = np.asarray(r['snap_times'], float) / 3600
    if snaps.ndim != 3 or len(ts) < 3:
        return None
    st = m.storage
    rr = st.block.rad_in + np.arange(st.num_rad_segments + 1) * (st.block.rad_out - st.block.rad_in) / st.num_rad_segments
    rm = _ring_means(snaps, rr)                     # (n_snap, nA)
    zc = 0.5 * (_edges_z(st)[1:] + _edges_z(st)[:-1])
    nA = rm.shape[1]
    n_low = int(getattr(m.mesh, 'n_low', 0) or 0)
    idx = [(0, 'bottom ring'), (nA - 1, 'top ring')]
    if 0 < n_low < nA:
        idx[1:1] = [(n_low - 1, 'below liner bottom'), (n_low, 'above liner bottom')]
    t_end = (r['t_ps_removed'] or ts[-1] * 3600) / 3600
    keep = ts <= t_end * 1.3 + 2
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    for c, (j, lab) in enumerate(idx):
        ax.plot(ts[keep], rm[keep, j] - C2K, c=f"C{c}", label=f"{lab} (z = {zc[j]:.2f} m)")
    t = np.asarray(r['time'], float) / 3600
    kk = t <= t_end * 1.3 + 2
    ax.plot(t[kk], np.asarray(r['T_avg'], float)[kk] - C2K, c='k', lw=1.4, label='block average')
    ax.axvline(t_end, c='0.5', ls='--', lw=0.8)
    ax.axhline(cy.T_min_C, c='k', ls=':', lw=0.8)
    ax.set_xlabel('time in discharge [h]')
    ax.set_ylabel('C')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


# ==========================================================================
# Appendix A
# ==========================================================================
def alternatives(doc, ctx):
    doc.h2('A.C Transient, heater and deposition options not used')
    doc.p(('Local per-segment radiation (radiation.axial: false). ', {'bold': True}),
          f"TinlessSimulation._step_calc / _compute_f_shade ({_ref('tinless.py', 'def _compute_f_shade')}): each ring "
          "exchanges only with its own stick segment, and the shade fraction is found by Brent's method (xtol 1e-4, 30 "
          "iterations) on the summed local flux. The wall flux is explicit (no Jacobian). It needs block height = stick "
          "length.")
    doc.p(('Porous shade (shade.model: porous). ', {'bold': True}),
          "The shade is a thin porous sheet with an open fraction, solved inside one AxialRadiation enclosure by "
          "Newton's method with the flux constraint. The Jacobian it gives the conduction step includes the "
          "controller response. Its time-step limits follow the same rules (6.3).")
    doc.p(('Outer-wall insulation loss (block_heat_loss_bool: true). ', {'bold': True}),
          "A conduction + convection loss from the outer node through th_ins of insulation to T_sur, in place of "
          "the adiabatic outer radius. Not used: the outer radius is a symmetry line in the checkerboard.")
    doc.p(('Discharge run options. ', {'bold': True}),
          "stop_at_cutoff (no conduction-only tail) and dt_at_shade_limit (longer steps while the vanes are closed) "
          "are used only in optimizer screening. dt_scale coarsens all three steps. Deposition can be switched off.")
    doc.p(('Deposition surfaces. ', {'bold': True}),
          "tac and zrc (carbon as graphite plus Ta or Zr vapour), tungsten (W foil in front of a TaC liner: series "
          "solid-gas carbon resistance plus W vapour) and w_shell (solid W liner). They are selected by "
          "powerstick.tungsten_foil and liner_material. The in-use graphite liner deposits carbon only.")
    doc.p(('Heater power. ', {'bold': True}),
          "heater.max_power (absolute) overrides max_power_per_m x H. The cycle and optimizer power searches use it "
          "to cut the heater to the smallest power that charges within charge_hr.")
