"""Model documentation, section 5: tank and plant (thermal/tank.py).

sec_tank(doc, ctx) writes section 5 and returns the numerical-settings rows for section 10;
alternatives(doc, ctx) writes the tank options for Appendix A. Numbers are computed live
from thermal.tank and the YAML `tank:` block (tank_system_reference = v3 design point)."""
import copy
import math

import numpy as np

from . import tank as tk
from .docs_base import flowchart
from .report import _fmt

C2K = 273.15
DIM = '666666'
OPT_SET = {'t_side', 't_top', 'L_bot', 'P_rated', 'duration_hr', 'block_axial_losses'}

# key -> (unit, meaning, used in)
DESC = {
    'P_rated': ('W_e', 'plant AC rating; N_ps = ceil(P_rated / P_stick)', 'size_hot_zone'),
    'duration_hr': ('h', 'rated duration; energy mismatch N E_stick,e / (P_rated t) - 1 is reported, not resized',
                    'size_hot_zone'),
    'T_amb': ('C', 'annual-mean ambient: shell / deck sink and floor-air inlet for every loss calculation',
              'losses, side_wall, roof'),
    'T_air_design': ('C', 'design-day floor-air inlet: floor temperature check and column design only',
                     '_design_bottom'),
    't_side': ('m', 'side Thermax thickness (cylindrical wall)', 'tank_geometry, side_wall'),
    't_top': ('m', 'roof Thermax thickness (planar); also loads the columns', 'tank_geometry, roof'),
    'L_bot': ('m', 'bottom stack height: columns in parallel with Thermax', 'design_columns, bottom_heat'),
    'H_wd': ('m', 'stick withdrawal headspace above the roof Thermax; null -> L_ps + t_top', 'tank_geometry'),
    'H_deck': ('m', 'deck / structure allowance at the top of the tank', 'tank_geometry'),
    'c_shell': ('m', 'radial clearance Thermax -> shell (geometry only, no thermal role)', 'tank_geometry'),
    'soil_allowable': ('Pa', 'allowable foundation pressure (warning; optimizer constraint)', '_system'),
    'thermax.rho': ('kg/m3', 'Thermax density (column load, mass, cost); k(T) in properties.thermax_tcx',
                    'tank_geometry, costs'),
    'columns.pitch': ('m', 'square column grid pitch; null -> from capacity / areal load', 'column_pitch'),
    'columns.thickness': ('m', 'column thickness (constant; the width varies with height)', 'design_columns'),
    'columns.w_max': ('m', 'largest graphite width (hot face); sets the pitch', 'column_pitch'),
    'columns.w_min': ('m', 'smallest graphite width (foot of the graded section)', 'design_columns'),
    'columns.sigma_ref': ('Pa', 'creep allowable at T_ref (1 % strain in 30 yr, no extra factor)', 'sigma_allow'),
    'columns.T_ref': ('C', 'reference temperature of sigma_ref', 'sigma_allow'),
    'columns.n': ('-', 'creep stress exponent', 'sigma_allow'),
    'columns.Q_kcal': ('kcal/mol', 'creep activation energy', 'sigma_allow'),
    'columns.sigma_cap': ('Pa', 'upper cap on the allowable (strength fraction)', 'sigma_allow'),
    'columns.T1': ('C', 'graphite -> oxide joint temperature (oxide max service T if L_graphite set)',
                   'design_columns'),
    'columns.L_graphite': ('m', 'fixed graphite segment length (joint T becomes an output); null -> joint at T1',
                           'design_columns'),
    'columns.sigma_ox': ('Pa', 'oxide post allowable stress (no temperature dependence)', 'design_columns'),
    'columns.rho_graphite': ('kg/m3', 'column graphite density (mass, cost)', 'costs'),
    'columns.rho_oxide': ('kg/m3', 'oxide post density', 'costs'),
    'columns.f_3D': ('-', 'multiplier on the 1-D column heat (calibrated to the v3 FEA)', 'bottom_heat'),
    'columns.plate_area': ('m2', 'bearing plate area per column (floor hot-spot check)', 'floor_air'),
    'floor.plate_t': ('m', 'floor plate thickness - NOT USED in the calculation', '-'),
    'floor.k_steel': ('W/m-K', 'fin conductivity', 'floor_air'),
    'floor.fin_h': ('m', 'fin height (under the plate)', 'floor_air'),
    'floor.fin_t': ('m', 'fin thickness', 'floor_air'),
    'floor.fin_pitch': ('m', 'fin pitch', 'floor_air'),
    'floor.channel_clearance': ('m', 'channel height = fin_h + clearance', 'floor_air'),
    'floor.v_air': ('m/s', 'channel air velocity; single pass, flow length = D_tank', 'floor_air'),
    'floor.K_minor': ('-', 'minor-loss coefficient (inlet, outlet, louvres)', 'floor_air'),
    'floor.eta_fan': ('-', 'fan efficiency', 'floor_air'),
    'floor.T_limit_normal': ('C', 'API 650 base floor limit - reported margin only', 'floor_air'),
    'floor.T_limit_annexM': ('C', 'API 650 Annex M limit: warning and optimizer constraint', '_system'),
    'shell.h_out': ('W/m2-K', 'shell and deck convection to ambient (radiation added)', 'side_wall, roof'),
    'shell.emissivity': ('-', 'shell and deck emissivity (outside, and deck in the headspace)', 'side_wall, roof'),
    'shell.h_headspace': ('W/m2-K', 'argon natural convection, roof Thermax -> deck', 'roof'),
    'shell.emis_insulation': ('-', 'Thermax top-face emissivity', 'roof'),
    'economics.elec_price': ('$/MWh', 'price of lost heat in the tank report (optimizer uses its own)', 'report'),
    'economics.heater_eff': ('-', 'heater efficiency in the tank report', 'report'),
    'block_axial_losses': ('-', 'roof + floor losses as a sink in the cell runs (read by sizing.run_discharge)',
                           'sizing.py'),
}


def _flat(d, pre=''):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict) and k != 'costs':
            out.update(_flat(v, pre + k + '.'))
        elif k != 'costs':
            out[pre + k] = v
    return out


def _v(x):
    if x is None:
        return 'null'
    if isinstance(x, bool):
        return 'true' if x else 'false'
    if isinstance(x, float) and (abs(x) >= 1e5 or 0 < abs(x) < 1e-3):
        return f"{x:.3g}"
    return _fmt(x)


def _cfg(ctx):
    return tk.merge_defaults(copy.deepcopy((ctx.get('base_cfg') or {}).get('tank')))


def _ref(c):
    return tk.tank_system_reference(copy.deepcopy(c))


def _plant_tank(ctx):
    cy = ctx.get('cy')
    return getattr(cy, 'tank', None) if cy is not None else None


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def _fig_plan(r_in, r_out, r_hc, label):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon, Rectangle
    p, webs = tk.cavity_webs(r_in, r_out, r_hc, rotated=True)
    S = math.pi * r_in / 2
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    n = 4
    ax.add_patch(Rectangle((-0.5 * p, -0.5 * p), n * p, n * p, fc='#b3b3b3', ec='none'))
    for i in range(n):
        for j in range(n):
            x, y = i * p, j * p
            if (i + j) % 2 == 0:
                h = S / math.sqrt(2)
                ax.add_patch(Polygon([(x + h, y), (x, y + h), (x - h, y), (x, y - h)], fc='white', ec='#1f3f8f', lw=1))
                g = 0.12 * S
                ax.add_patch(Polygon([(x + g, y), (x, y + g), (x - g, y), (x, y - g)], fc='#3a6fc4', ec='none'))
            else:
                ax.add_patch(Circle((x, y), r_hc, fc='white', ec='#8f1f1f', lw=1))
                ax.add_patch(Circle((x, y), 0.25 * r_hc, fc='#d43c2c', ec='none'))
    xs, ys = 1 * p, 1 * p                                    # a stick
    ax.add_patch(Circle((xs, ys), r_out, fc='none', ec='#1f3f8f', ls='--', lw=1))
    kw = dict(arrowstyle='<->', lw=1.1, color='#B5520A', shrinkA=0, shrinkB=0)
    tag = dict(fontsize=8, color='#B5520A', ha='center', va='center', fontweight='bold',
               bbox=dict(boxstyle='circle,pad=0.15', fc='white', ec='#B5520A', lw=0.6))
    # (1) pitch, below the stick-heater pair
    yb = ys - 0.38 * p
    ax.annotate('', xy=(xs + p, yb), xytext=(xs, yb), arrowprops=dict(arrowstyle='<->', lw=0.9))
    ax.plot([xs, xs], [yb - 0.05 * p, ys - S / math.sqrt(2) * 0.2], color='k', lw=0.4, ls=':')
    ax.plot([xs + p, xs + p], [yb - 0.05 * p, ys - r_hc], color='k', lw=0.4, ls=':')
    ax.text(xs + 0.5 * p, yb - 0.08 * p, 'p', fontsize=9, ha='center', va='top',
            bbox=dict(fc='white', ec='none', pad=0.5))
    # (2) stick-heater wall: square corner -> heater circle
    a, b = xs + S / math.sqrt(2), xs + p - r_hc
    ax.annotate('', xy=(b, ys), xytext=(a, ys), arrowprops=kw)
    ax.text(0.5 * (a + b), ys + 0.13 * p, '1', **tag)
    # (3) stick-stick wall: face midpoint -> diagonal neighbour's face midpoint
    d = S / 2 / math.sqrt(2)
    pa, pb = (xs + d, ys + d), (xs + p - d, ys + p - d)
    ax.annotate('', xy=pb, xytext=pa, arrowprops=kw)
    ax.text(0.5 * (pa[0] + pb[0]) - 0.12 * p, 0.5 * (pa[1] + pb[1]) + 0.12 * p, '2', **tag)
    # (4) heater-heater wall along the diagonal
    hx, hy = 2 * p, 1 * p
    e = r_hc / math.sqrt(2)
    qa, qb = (hx + e, hy + e), (hx + p - e, hy + p - e)
    ax.annotate('', xy=qb, xytext=qa, arrowprops=kw)
    ax.text(0.5 * (qa[0] + qb[0]) + 0.12 * p, 0.5 * (qa[1] + qb[1]) - 0.12 * p, '3', **tag)
    ax.set_xlim(-0.55 * p, (n - 0.45) * p)
    ax.set_ylim(-0.55 * p, (n - 0.45) * p)
    ax.set_aspect('equal')
    ax.set_xlabel('m')
    ax.set_ylabel('m')
    txt = (f"p = {p:.2f} m  (stick-heater centres)\n"
           f"S = {S:.2f} m  (stick cavity side)\n"
           f"r_hc = {r_hc:.2f} m,  r_out = {r_out:.2f} m\n\n"
           f"(1) stick-heater wall  {webs['stick-heater'] * 1e3:.0f} mm\n"
           f"(2) stick-stick wall   {webs['stick-stick'] * 1e3:.0f} mm\n"
           f"(3) heater-heater wall {webs['heater-heater'] * 1e3:.0f} mm\n\n"
           "dashed circle: stick-cell\nanalysis annulus r_out")
    ax.text(1.03, 0.98, txt, transform=ax.transAxes, fontsize=7.5, va='top', ha='left', family='monospace')
    ax.set_title(f'Checkerboard plan ({label})', fontsize=9)
    fig.subplots_adjust(right=0.62)
    return fig, p, webs


def _fig_section():
    """Schematic half section of the tank with the heat-flow paths (not to scale)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Polygon, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(7.0, 7.4))
    R = lambda x0, y0, x1, y1, fc, **k: ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc=fc,
                                                               ec=k.pop('ec', '#444444'), lw=k.pop('lw', 0.6), **k))
    thx = '#f1e3bf'
    R(0, 0, 9.6, 0.55, '#cfcfcf')
    ax.text(4.6, 0.27, 'structural concrete mat', ha='center', va='center', fontsize=7)
    R(0, 0.55, 9.6, 0.9, '#e8e2bd')
    ax.text(4.6, 0.72, 'cellular glass', ha='center', va='center', fontsize=7)
    R(0, 0.9, 9.6, 1.25, '#d9b99b')
    ax.text(4.6, 1.07, 'HT slab, rails, firebrick caps  (cost only: no heat path modelled)', ha='center',
            va='center', fontsize=7)
    R(0, 1.25, 9.6, 1.65, 'white')
    for x in np.arange(0.15, 9.3, 0.3):
        ax.plot([x, x], [1.65, 1.36], color='#555555', lw=0.8)
    R(0, 1.65, 9.25, 1.75, '#555555')
    # bottom stack
    R(0, 1.75, 8.0, 3.75, thx, hatch='...', ec='#999999')
    for xc in (0.9, 2.9, 4.9, 6.9):
        ax.add_patch(Rectangle((xc - 0.3, 1.75), 0.6, 0.06, fc='#333333', ec='none'))
        ax.add_patch(Rectangle((xc - 0.22, 1.81), 0.44, 0.4, fc='#b87d5b', ec='#444444', lw=0.5))
        ax.add_patch(Polygon([(xc - 0.09, 2.21), (xc + 0.09, 2.21), (xc + 0.09, 2.7), (xc + 0.26, 3.75),
                              (xc - 0.26, 3.75), (xc - 0.09, 2.7)], fc='#6e6e6e', ec='#333333', lw=0.5))
    ax.text(6.9, 2.0, 'oxide', fontsize=6, ha='left', va='center', color='#5a3010', rotation=0,
            bbox=dict(fc='white', ec='none', pad=0.5, alpha=0.7))
    ax.annotate('graded graphite column\n(width follows creep allowable)', xy=(6.95, 3.3), xytext=(8.5, 3.15),
                fontsize=6.5, arrowprops=dict(arrowstyle='-', lw=0.5), va='center',
                bbox=dict(fc='white', ec='none', pad=0.5, alpha=0.85))
    ax.text(5.9, 3.05, 'bottom Thermax', fontsize=7, ha='center', bbox=dict(fc=thx, ec='none', pad=0.5))
    # hot zone
    R(0, 3.75, 8.0, 7.75, '#9e9e9e')
    for k, xc in enumerate(np.arange(0.5, 7.8, 0.95)):
        if k % 2 == 0:
            R(xc - 0.22, 3.75, xc + 0.22, 7.75, 'white', ec='#1f3f8f', lw=0.5)
            R(xc - 0.07, 5.4, xc + 0.07, 7.75 + 1.2, '#3a6fc4', ec='none')
        else:
            R(xc - 0.16, 3.75, xc + 0.16, 7.75, 'white', ec='#8f1f1f', lw=0.5)
            R(xc - 0.04, 3.95, xc + 0.04, 7.75 + 1.2, '#d43c2c', ec='none')
    ax.text(4.0, 4.3, 'HOT ZONE: graphite cells (stick cavities blue, heaters red)', ha='center', fontsize=7.5,
            bbox=dict(fc='white', ec='none', pad=1, alpha=0.85))
    # side / top insulation, shell, headspace, deck
    R(8.0, 1.75, 9.0, 8.95, thx, hatch='...', ec='#999999')
    R(0, 7.75, 8.0, 8.95, thx, hatch='...', ec='#999999')
    ax.text(4.0, 8.35, 'roof Thermax (t_top)', ha='center', fontsize=7.5, bbox=dict(fc=thx, ec='none', pad=0.5))
    ax.text(8.5, 6.6, 'side Thermax\n(t_side)', ha='center', fontsize=7, rotation=90, va='center',
            bbox=dict(fc=thx, ec='none', pad=0.5))
    R(9.1, 1.65, 9.25, 12.0, '#555555')
    R(0, 11.85, 9.25, 12.0, '#555555')
    for x in np.arange(0, 9.0, 0.6):
        ax.plot([x, x + 0.3, x + 0.6], [10.95, 11.85, 10.95], color='#888888', lw=0.6)
    ax.plot([0, 9.1], [10.95, 10.95], color='#888888', lw=0.6)
    ax.text(4.6, 10.15, 'argon headspace: stick withdrawal space H_wd\n(withdrawn stick dashed)', ha='center',
            fontsize=7)
    ax.add_patch(Rectangle((1.4, 9.2), 0.14, 1.6, fc='none', ec='#3a6fc4', ls='--', lw=0.8))
    ax.text(6.9, 11.4, 'deck trusses (H_deck)', fontsize=6.5, ha='center',
            bbox=dict(fc='white', ec='none', pad=0.5, alpha=0.9))
    ax.text(4.6, 12.25, 'deck / roof plate (area A_tank) -> ambient: h_out + radiation to T_amb', ha='center',
            fontsize=7)
    ax.plot([0, 0], [-0.2, 12.6], color='k', lw=0.6, ls='-.')
    ax.text(0.05, 12.5, 'CL', fontsize=7)
    ax.text(9.35, 11.0, 'steel shell\n(c_shell gap)', fontsize=6.5, ha='left')

    def arrow(p0, p1, txt, tx, ty, col='#c0392b', ha='left'):
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=11, lw=1.8, color=col, zorder=5))
        ax.text(tx, ty, txt, fontsize=7.5, color=col, ha=ha, va='center', zorder=6,
                bbox=dict(fc='white', ec='none', pad=0.6, alpha=0.9))
    arrow((7.0, 5.7), (10.6, 5.7), 'Q_side\n(cylindrical,\nover H)', 9.6, 6.55)
    arrow((7.4, 7.5), (9.9, 9.4), 'Q_edge (top)', 9.95, 9.55)
    arrow((7.4, 3.95), (9.9, 2.2), 'Q_edge (bottom)', 9.95, 2.45)
    arrow((3.0, 7.0), (3.0, 12.8), 'Q_top', 3.12, 9.4)
    arrow((2.9, 3.6), (2.9, 1.45), 'Q_bot,col\n(x f_3D)', 2.75, 2.95, ha='right')
    arrow((4.4, 3.6), (4.4, 1.45), 'Q_bot,thx', 4.55, 2.4)
    arrow((0.3, 1.45), (10.8, 1.45), '', 0, 0, col='#2f5f9e')
    ax.text(10.0, 1.05, 'floor air: in at T_amb\n(annual) or T_air,design;\nout +dT_air -> fans,\nexhaust',
            fontsize=6.5, color='#2f5f9e', ha='left', va='top')
    ax.text(10.75, 5.0, 'ambient\nT_amb', fontsize=7, ha='left')
    # dimension brackets
    for y0, y1, t in ((1.75, 3.75, 'L_bot'), (3.75, 7.75, 'H'), (7.75, 8.95, 't_top'), (8.95, 10.95, 'H_wd'),
                      (10.95, 11.85, 'H_deck')):
        ax.annotate('', xy=(-0.35, y1), xytext=(-0.35, y0), arrowprops=dict(arrowstyle='<->', lw=0.6))
        ax.text(-0.45, 0.5 * (y0 + y1), t, rotation=90, ha='right', va='center', fontsize=7)
    ax.set_xlim(-1.1, 12.3)
    ax.set_ylim(-0.3, 12.9)
    ax.set_aspect('equal')
    ax.axis('off')
    return fig


def _fig_network():
    b = {
        'hz': (3.7, 7.3, 'Hot-zone graphite\nT_side = block mean,  T_top = top ring,  T_bot = bottom ring', 'io'),
        'side': (0.75, 5.75, 'Side Thermax\ncylindrical r_i -> r_o\nover H', 'proc'),
        'edge': (2.25, 5.75, 'Edges (top,\nbottom)\nS = 0.54 pi D_hot', 'proc'),
        'roof': (3.75, 5.75, 'Roof Thermax\nplanar, t_top\narea A_hot', 'proc'),
        'col': (5.35, 5.75, 'N_col columns\ngraphite -> joint\n-> oxide, x f_3D', 'proc'),
        'thx': (6.75, 5.75, 'Bottom Thermax\nA_hot - N_col A_mean\nthickness L_bot', 'proc'),
        'shell': (1.5, 4.2, 'Steel shell, T_shell\nh_out + eps sigma T^4', 'proc'),
        'hs': (3.75, 4.2, 'Argon headspace\nh_hs + eps_hs sigma T^4', 'proc'),
        'floor': (6.05, 4.2, 'Floor plate + fins\nT_floor (avg) = cold face', 'proc'),
        'deck': (3.75, 2.75, 'Deck / roof, A_tank\nh_out + eps sigma T^4', 'proc'),
        'air': (6.05, 2.75, 'Forced-air channels\nh_eff, dT_air = Q_bot/(rho V c_p)', 'loop'),
        'amb': (2.3, 1.2, 'Ambient air T_amb\n(annual mean; also on the design day)', 'out'),
        'exh': (6.05, 1.2, 'Exhaust air\nfan power dp V / eta_fan', 'out'),
    }
    a = [('hz', 'side'), ('hz', 'edge'), ('hz', 'roof'), ('hz', 'col'), ('hz', 'thx'),
         ('side', 'shell'), ('edge', 'shell'), ('roof', 'hs'), ('hs', 'deck'), ('deck', 'amb'),
         ('shell', 'amb'), ('col', 'floor'), ('thx', 'floor'), ('floor', 'air'), ('air', 'exh', 'T_in -> T_in + dT')]
    g = [(0.05, 5.12, 7.45, 6.6, 'insulation paths (1-D Kirchhoff conduction)'),
         (4.6, 0.6, 7.45, 4.72, 'floor cooling (fixed point on T_floor)')]
    return flowchart(b, a, size=(7.5, 8.0), groups=g, fontsize=7.5)


def _fig_losses(res, c):
    import matplotlib.pyplot as plt
    T = np.arange(1200.0, 2501.0, 50.0)
    L = [tk.losses(x, x, x, res.geo, res.hz, res.cols, c) for x in T]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for k, lab in (('Q_side', 'side'), ('Q_edge', 'edges'), ('Q_top', 'roof'), ('Q_bot_col', 'bottom: columns'),
                   ('Q_bot_thx', 'bottom: Thermax'), ('Q_total', 'total')):
        ax.plot(T, [d[k] / 1e3 for d in L], lw=2.2 if k == 'Q_total' else 1.3, label=lab,
                color='k' if k == 'Q_total' else None)
    ax.set_xlabel('uniform hot-face temperature [C]')
    ax.set_ylabel('loss [kW]')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, ncol=2)
    ax2 = ax.twinx()
    ax2.plot(T, [d['Q_total'] * 86400 / res.hz.E_th * 100 for d in L], ls=':', color='k')
    ax2.set_ylabel('total, % of stored swing per day (dotted)', fontsize=8)
    return fig


def _column_profile(cd, c, T_hot):
    col = c['columns']
    Q1, Tj = tk.column_heat(cd, T_hot, cd.T_floor_design)
    A_min = col['w_min'] * col['thickness']
    Tg = np.linspace(Tj, T_hot, 600)
    Ag = np.maximum(cd.F / tk.sigma_allow(Tg, col), A_min)
    kA = tk.k_graphite(Tg) * Ag
    zg = np.concatenate([[0.0], np.cumsum(0.5 * (kA[1:] + kA[:-1]) * np.diff(Tg))]) / Q1
    To = np.linspace(cd.T_floor_design, Tj, 100)
    zo = np.array([cd.A_ox * tk.psi_oxide(cd.T_floor_design, x) / Q1 for x in To])
    return dict(Tg=Tg, wg=Ag / col['thickness'], yg=cd.L_ox + zg, To=To, wo=np.full_like(To, cd.A_ox / col['thickness']),
                yo=zo, Tj=Tj, Q1=Q1)


def _fig_columns(res, c, T_hot):
    import matplotlib.pyplot as plt
    col = c['columns']
    cd = res.cols
    pr = _column_profile(cd, c, T_hot)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 3.6), gridspec_kw=dict(width_ratios=[1.15, 1]))
    T = np.linspace(800, 2550, 300)
    a1.semilogy(T, tk.sigma_allow(T, col) / 1e6, color='k', label='sigma_allow(T)')
    A_min = col['w_min'] * col['thickness']
    for v, lab, ls, f in ((cd.F / cd.A_top / 1e6, f'F / A_top (w_max) = {cd.F / cd.A_top / 1e6:.2f} MPa', '--', 0.9),
                          (cd.F / A_min / 1e6, f'F / A_min (w_min) = {cd.F / A_min / 1e6:.1f} MPa', ':', 1.06),
                          (col['sigma_ox'] / 1e6, f"sigma_ox = {col['sigma_ox'] / 1e6:.1f} MPa (oxide, any T)", '-.',
                           1.06)):
        a1.axhline(v, ls=ls, lw=0.9, color='#2f5f9e')
        a1.text(1290, v * f, lab, fontsize=6.5, color='#2f5f9e', va='bottom' if f > 1 else 'top')
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
    a1.set_ylim(1.5, 30)
    a1.yaxis.set_major_locator(FixedLocator([2, 3, 5, 10, 20]))
    a1.yaxis.set_major_formatter(FixedFormatter(['2', '3', '5', '10', '20']))
    a1.yaxis.set_minor_formatter(NullFormatter())
    a1.axvline(col['T1'], color='#B5520A', lw=0.8)
    a1.text(col['T1'] + 15, 1.7, f"T1 {col['T1']:.0f} C", fontsize=7, color='#B5520A', rotation=90)
    a1.set_xlabel('T [C]')
    a1.set_ylabel('stress [MPa]')
    a1.set_title('creep allowable and section stresses', fontsize=8.5)
    a1.grid(alpha=0.3, which='both')
    for y, w, fc in ((pr['yg'], pr['wg'], '#6e6e6e'), (pr['yo'], pr['wo'], '#b87d5b')):
        a2.fill_betweenx(y, -w * 500, w * 500, color=fc)
    a2.set_xlabel('column width [mm] (thickness %.0f mm)' % (col['thickness'] * 1e3))
    a2.set_ylabel('height above floor plate [m]')
    a2.set_xlim(-max(pr['wg']) * 650, max(pr['wg']) * 650)
    t2 = a2.twiny()
    t2.plot(np.concatenate([pr['To'], pr['Tg']]), np.concatenate([pr['yo'], pr['yg']]), color='#c0392b')
    t2.set_xlabel('temperature [C] (red)', fontsize=8, color='#c0392b')
    a2.axhline(cd.L_ox, color='#B5520A', lw=0.7, ls='--')
    a2.text(-max(pr['wg']) * 630, cd.L_ox + 0.03, f"joint {pr['Tj']:.0f} C", fontsize=7, color='#B5520A')
    fig.tight_layout()
    return fig, pr


def _fig_floor(res, c, T_hot):
    import matplotlib.pyplot as plt
    V = np.linspace(2.0, 12.0, 21)
    out = []
    for v in V:
        c2 = copy.deepcopy(c)
        c2['floor']['v_air'] = float(v)
        _, fl, _ = tk.bottom_and_floor(res.cols, res.geo.A_hot, res.geo.D_tank, T_hot, c['T_air_design'], c2)
        out.append(fl)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot(V, [f['T_floor_max'] for f in out], label='floor max (outlet end)')
    ax.plot(V, [f['T_floor_avg'] for f in out], label='floor average (bottom-stack cold face)')
    ax.plot(V, [f['T_floor_col'] for f in out], label='under a column (bearing plate)')
    ax.axhline(c['floor']['T_limit_annexM'], color='#B5520A', ls='--', lw=0.9)
    ax.text(V[0], c['floor']['T_limit_annexM'] + 3, f"T_limit_annexM {c['floor']['T_limit_annexM']:.0f} C",
            fontsize=7, color='#B5520A')
    ax.axvline(c['floor']['v_air'], color='k', lw=0.6, ls=':')
    ax.set_xlabel('channel air velocity v_air [m/s]')
    ax.set_ylabel('temperature [C], design-day air')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc='upper center')
    ax2 = ax.twinx()
    ax2.plot(V, [f['P_fan'] / 1e3 for f in out], color='k', ls='-.', lw=1)
    ax2.set_ylabel('fan power [kW] (dash-dot)', fontsize=8)
    return fig


def _fig_sink(sink, T_max, T_min):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    ax.plot(sink.T_C, sink.q_top, label='q_top (roof)')
    ax.plot(sink.T_C, sink.q_bot, label='q_bot (bottom stack + floor)')
    ax.axvspan(T_min, T_max, color='#dddddd', alpha=0.6, lw=0)
    ax.set_xlabel('hot-face (ring node) temperature [C]')
    ax.set_ylabel('W per m2 of hot-zone plan area')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5)
    return fig


def _fig_order():
    b = {
        'in': (3.4, 8.78, 'StickInputs (stick_from_run)  +  YAML tank: block (merge_defaults)', 'io'),
        'hz': (3.4, 7.98, 'size_hot_zone: N_ps = ceil(P_rated / P_stick), pitch, D_hot, fill, m_g, E_th, mismatch',
               'proc'),
        'geo': (3.4, 7.18, 'tank_geometry: D_tank, H_tank, H_wd, A_hot, A_tank, p_bear (+ range warnings)', 'proc'),
        'col': (2.6, 6.26, 'design_columns(T_max, T_f): pitch from capacity,\ngraded section, joint at T1 -> S_g, S_ox',
                'loop'),
        'fl': (2.6, 5.30, 'bottom_and_floor (design-day air): bottom_heat(T_f)\n<-> floor_air -> T_f,avg '
                          '(tol 1e-3 K, <= 60 it.)', 'loop'),
        'dec': (6.0, 5.78, 'T_f moved\n> 1e-3 K ?\n(<= 30 passes)', 'dec'),
        'des': (3.4, 4.34, 'design = losses(T_max, design-day air);  floor-limit warnings  (floor_design)', 'proc'),
        'pk': (3.4, 3.62, 'loss_peak = losses(T_max, annual air);  loss_fn(T) defined', 'proc'),
        'cyc': (3.4, 2.90, 'loss_cycle: losses() at <= 60 history points, trapezoid;  E_loss, E_side', 'proc'),
        'cost': (3.4, 2.18, 'costs (V_dot from design floor)  ->  weights  ->  foundation pressure', 'proc'),
        'cor': (3.4, 1.46, 'economics (report);  stick_correction: penalty_frac', 'proc'),
        'out': (3.4, 0.70, 'CellLossSink.from_system  ->  TankResult (+ mismatch / discharge-only warnings)', 'out'),
    }
    a = [('in', 'hz'), ('hz', 'geo'), ('geo', 'col'), ('col', 'fl'), ('fl', 'dec', '', 0.25),
         ('dec', 'col', 'yes', 0.25), ('dec', 'des', 'no', -0.2), ('des', 'pk'), ('pk', 'cyc'), ('cyc', 'cost'),
         ('cost', 'cor'), ('cor', 'out')]
    g = [(0.35, 4.85, 7.1, 6.85, '_design_bottom: outer fixed point on the floor temperature')]
    return flowchart(b, a, size=(7.2, 9.4), groups=g, fontsize=7.5)


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------
def _inputs_table(doc, c):
    d = _flat(tk.TANK_DEFAULTS)
    u = _flat(c)
    rows = []
    for k, dv in d.items():
        uv = u.get(k)
        unit, mean, where = DESC.get(k, ('', '', ''))
        diff = (dv != uv) and not (dv is None and uv is None)
        if k.split('.')[-1] in OPT_SET or k in OPT_SET:
            mean += ' [set per design by the optimizer]'
        rows.append([k, _v(dv), (_v(uv), {'bold': True}) if diff else _v(uv), unit, mean, where])
    doc.table(['key (tank.)', 'code default', 'in use (YAML)', 'unit', 'meaning', 'used in'], rows,
              widths=[1.35, 0.65, 0.7, 0.55, 2.75, 1.1], size=7.5)


def _constants_table(doc):
    A = tk.AIR
    kg = ', '.join(f"{t:.0f}:{k:.0f}" for t, k in zip(tk._KG_T, tk._KG_K))
    rows = [
        ['Stefan-Boltzmann sigma', f"{tk.SIGMA:.6g} W/m2-K4", 'all radiation terms'],
        ['g', f"{tk.G} m/s2", 'column load, foundation'],
        ['R (gas constant)', f"{tk.R_GAS} J/mol-K", 'creep Arrhenius term'],
        ['kcal', f"{tk.KCAL:.0f} J", 'Q_kcal -> J/mol'],
        ['steel density', f"{tk.RHO_STEEL:.0f} kg/m3", 'shell, bottom mass'],
        ['air properties (fixed, ~60 C)', f"rho {A['rho']}, mu {A['mu']:.4g}, k {A['k']}, cp {A['cp']:.0f}, "
                                          f"Pr {A['Pr']}", 'floor_air (all conditions)'],
        ['edge shape factor', '0.54 x edge length, 2 edges', 'losses (Q_edge)'],
        ['friction factor', 'Petukhov f = (0.79 ln Re - 1.64)^-2', 'floor_air'],
        ['Nusselt number', 'Gnielinski, valid 3e3 < Re < 5e6', 'floor_air'],
        ['fin efficiency', 'tanh(mH)/(mH), adiabatic tip', 'floor_air'],
        ['oxide k(T)', '2.8 + 1000 / T[K]  W/m-K', 'k_oxide, psi_oxide'],
        ['column graphite k(T)', f"table T[C]:k[W/m-K] = {kg}; held constant outside", 'k_graphite (Mersen 2020)'],
        ['Thermax k(T)', 'properties.thermax_tcx (section 7), exact Kirchhoff integral', 'k_thermax, phi_thermax'],
        ['storage graphite rho, cp', 'properties.storage_rho / storage_cp (section 7) via the stick model',
         'm_graphite, E_th'],
        ['provisional heat-to-electricity ratio', '0.45 (cell_loss_sink eta_e; stick_from_run fallback)',
         'cell_loss_sink, stick_from_run'],
        ['reference-mode storage density', '1450 kg/m3 (fill only)', 'tank_system_reference'],
        ['bearing warning', '225 kPa (hard-coded, independent of soil_allowable)', 'tank_geometry'],
        ['API 650 usual range (warning)', 'H_tank 10-30 m, D_tank 15-80 m', 'tank_geometry'],
    ]
    doc.table(['constant / correlation', 'value', 'where'], rows, widths=[1.9, 3.4, 1.8], size=7.5)


COST_LINES = [   # name, quantity basis, rate key, rate unit
    ('tank shell', 'pi D_tank H_tank t_shell rho_steel', 'steel_per_t', '$/t'),
    ('tank bottom', 'A_tank t_bottom rho_steel (1 + annular frac)', 'steel_per_t', '$/t'),
    ('roof, ring girder, wind girders', 'A_tank roof_kg_m2', 'steel_per_t', '$/t'),
    ('deck trusses', 'A_tank deck_kg_m2 (low/central/high)', 'deck_per_t', '$/t'),
    ('gas tightness, nozzles', 'fixed + per penetration x (N_ps + N_h)', None, '$ + $/pen.'),
    ('Thermax side', 'pi (r_o^2 - r_i^2)(L_bot + H + t_top) rho_thx', 'thermax_per_kg', '$/kg'),
    ('Thermax top', 'A_hot t_top rho_thx', 'thermax_per_kg', '$/kg'),
    ('Thermax bottom', '(A_hot L_bot - N_col (V_g + V_ox)) rho_thx', 'thermax_per_kg', '$/kg'),
    ('column graphite', 'N_col V_g rho_graphite', 'column_graphite_per_kg', '$/kg'),
    ('oxide posts', 'N_col V_ox rho_oxide', 'oxide_per_kg', '$/kg'),
    ('bearing plates', 'N_col plate_area bearing_plate_t rho_steel', 'structural_steel_per_kg', '$/kg'),
    ('column tie grid', 'N_col', 'tie_grid', '$/column'),
    ('floor plate, fins, welds', 'A_tank', 'floor_per_m2', '$/m2'),
    ('HT slab, rails, firebrick', 'A_tank', 'slab_per_m2', '$/m2'),
    ('cellular glass', 'A_tank', 'cellular_glass_per_m2', '$/m2'),
    ('structural mat', 'A_tank mat_thickness', 'mat_per_m3', '$/m3'),
    ('fans, louvres, exhaust, controls', 'V_dot (design floor)', 'fans_per_m3s', '$/(m3/s)'),
]


def _r(x):
    return f"{x:,.0f}" if abs(x) >= 100 else f"{x:g}"


def _tri(v):
    v = np.broadcast_to(np.asarray(v, float), (3,))
    return ' / '.join(_r(float(x)) for x in v)


def _cost_table(doc, c, res, res2):
    k = c['costs']
    rows = []
    for name, basis, rk, unit in COST_LINES:
        ln = res.cost['lines'][name]
        rate = (f"{_tri(k['gas_tight_fixed'])} + {_tri(k['per_penetration'])}" if rk is None else _tri(k[rk]))
        q = ln['qty'][1]
        r = [name, basis, rate, unit, f"{_fmt(q)} {ln['unit']}", f"{ln['cost'][1] / 1e6:.2f}"]
        if res2 is not None:
            r.append(f"{res2.cost['lines'][name]['cost'][1] / 1e6:.2f}")
        rows.append(r)
    tot = ['total (tank)', '', '', '', '', (f"{res.cost['total'][1] / 1e6:.2f}", {'bold': True})]
    sg = ['storage graphite (separate)', 'm_graphite', _tri(k['storage_graphite_per_kg']), '$/kg',
          f"{res.hz.m_graphite / 1e3:,.0f} t", f"{res.cost['storage_graphite'][1] / 1e6:.2f}"]
    if res2 is not None:
        tot.append((f"{res2.cost['total'][1] / 1e6:.2f}", {'bold': True}))
        sg.append(f"{res2.cost['storage_graphite'][1] / 1e6:.2f}")
    rows += [tot, sg]
    head = ['line', 'quantity', 'rate low / central / high', 'unit', 'ref. qty (central)', 'ref. $M']
    w = [1.25, 1.85, 1.35, 0.6, 0.95, 0.5]
    if res2 is not None:
        head.append('design $M')
        w = [1.2, 1.75, 1.3, 0.55, 0.85, 0.45, 0.5]
    doc.table(head, rows, widths=w, size=7)


def _summary_vals(res):
    hz, g, cd, c = res.hz, res.geo, res.cols, res.cfg
    pk, cyc, d, fl = res.loss_peak, res.loss_cycle, res.design, res.floor_design
    t = c['columns']['thickness']
    E = hz.E_th
    return [
        ('t_side / t_top', 'm', f"{c['t_side']:.2f} / {c['t_top']:.2f}"),
        ('sticks N_ps (= heaters N_h)', '-', f"{hz.N_ps}" if hz.N_ps else 'n/a (hot zone given)'),
        ('hot zone D_hot x H', 'm', f"{hz.D_hot:.1f} x {hz.H_block:.2f}"),
        ('fill (graphite / hot-zone volume)', '-', f"{hz.fill:.3f}"),
        ('storage graphite', 't', f"{hz.m_graphite / 1e3:,.0f}"),
        ('stored swing E_th', 'GWh_th', f"{E / 3.6e12:.2f}"),
        ('energy mismatch', '%', f"{hz.energy_mismatch * 100:+.1f}" if np.isfinite(hz.energy_mismatch) else 'n/a'),
        ('tank D x H (H_wd)', 'm', f"{g.D_tank:.1f} x {g.H_tank:.1f} ({g.H_wd:.2f})"),
        ('column-grid load p_bear', 'kPa', f"{g.p_bear / 1e3:.0f}"),
        ('columns: number, pitch, load', '-, m, kN', f"{cd.N_col:.0f}, {cd.pitch:.2f}, {cd.F / 1e3:.0f}"),
        ('graphite width top -> joint, length', 'mm, m',
         f"{cd.A_top / t * 1e3:.0f} -> {cd.A_joint / t * 1e3:.0f}, {cd.L_g:.2f}"),
        ('oxide width, length', 'mm, m', f"{cd.A_ox / t * 1e3:.0f}, {cd.L_ox:.2f}"),
        ('peak loss: side / edges / roof', 'kW', f"{pk['Q_side'] / 1e3:.0f} / {pk['Q_edge'] / 1e3:.0f} / "
                                                 f"{pk['Q_top'] / 1e3:.0f}"),
        ('peak loss: bottom (columns / Thermax)', 'kW', f"{pk['Q_bot'] / 1e3:.0f} ({pk['Q_bot_col'] / 1e3:.0f} / "
                                                        f"{pk['Q_bot_thx'] / 1e3:.0f})"),
        ('peak loss total, % per day', 'kW, %', f"{pk['Q_total'] / 1e3:.0f}, {pk['Q_total'] * 86400 / E * 100:.2f}"),
        ('cycle-average loss total, % per day', 'kW, %', f"{cyc['Q_total'] / 1e3:.0f}, "
                                                         f"{cyc['Q_total'] * 86400 / E * 100:.2f}"),
        ('shell / roof Thermax top / deck (peak)', 'C', f"{pk['T_shell']:.0f} / {pk['T_ins_top']:.0f} / "
                                                         f"{pk['T_deck']:.0f}"),
        ('joint T (design-day)', 'C', f"{d['T_joint']:.0f}"),
        ('floor avg / max / column spot (design day)', 'C', f"{fl['T_floor_avg']:.0f} / {fl['T_floor_max']:.0f} / "
                                                            f"{fl['T_floor_col']:.0f}"),
        ('floor air V_dot, dT_air, h_eff', 'm3/s, K, W/m2-K', f"{fl['Vdot']:.1f}, {fl['dT_air']:.1f}, "
                                                              f"{fl['h_eff']:.1f}"),
        ('floor-channel Re, fin efficiency', '-', f"{fl['Re']:,.0f}, {fl['eta_fin']:.2f}"),
        ('pressure drop, fan power', 'Pa, kW', f"{fl['dp']:.0f}, {fl['P_fan'] / 1e3:.1f}"),
        ('foundation pressure', 'kPa', f"{res.foundation_pressure / 1e3:.0f}"),
        ('tank capex low / central / high', '$M', ' / '.join(f"{x / 1e6:.1f}" for x in res.cost['total'])),
        ('sink f_open = 1/fill; q_top, q_bot at T_max', '-, W/m2',
         f"{res.cell_sink.f_open:.2f}; {res.cell_sink.q(res.cols.T_design)[0]:.0f}, "
         f"{res.cell_sink.q(res.cols.T_design)[1]:.0f}"),
    ]


# --------------------------------------------------------------------------
# Section 5
# --------------------------------------------------------------------------
def sec_tank(doc, ctx):
    c = _cfg(ctx)
    col, fl_c, sh = c['columns'], c['floor'], c['shell']
    ref = _ref(c)
    T_ref_max = ref.cols.T_design
    plant = _plant_tank(ctx)
    cy = ctx.get('cy')

    doc.h1('5. Tank and plant')
    doc.p("The per-stick results of the cell model are scaled to a whole plant by thermal/tank.py. The plant is a "
          "single hot zone: stick cells and heater cells on a checkerboard in one continuous graphite matrix. It sits "
          "inside an insulated, gas-tight API 650-type steel tank, on graded graphite-to-oxide support columns. The "
          "columns stand through bottom insulation on a forced-air-cooled, finned steel floor, which rests on a "
          "high-temperature slab, cellular glass and a concrete mat. The side and roof insulation are Thermax. "
          "Every calculation in the module is closed form or a 1-D integral (milliseconds per call), so the "
          "optimizer can call it for every design.")
    doc.p("Inside tank.py temperatures are in degC unless a name ends in _K. Entry points: ",
          ('stick_from_run', {'mono': True}), " (per-stick inputs from a cell run), ",
          ('tank_system', {'mono': True}), " (full plant), ", ('tank_system_reference', {'mono': True}),
          " (hot zone given directly; the v3 design point used as the reference case below), ",
          ('cell_loss_sink / CellLossSink', {'mono': True}), " (roof and floor losses as a sink on the cell model) "
          "and ", ('calibrate_f3d', {'mono': True}), ".")
    doc.review("tank.py cites a design brief 'steel_tank_system_brief.md' (sections 2-8) for its assumptions. That "
               "file is not in the repository, so the brief references cannot be checked. [confidence high; "
               "affects reviewability only] (tank.py:1)")

    doc.h3('Inputs of the tank model')
    doc.p("All inputs come from the YAML tank: block, deep-merged over tank.TANK_DEFAULTS (tank.py:40-69; "
          "merge_defaults also converts numeric strings such as '25.0e6' to floats). Bold values differ from the "
          "code default. Cost rates are listed in section 5.8. The optimizer overwrites t_side, t_top, P_rated, "
          "duration_hr and block_axial_losses for every design (optimize.design_cfg).")
    _inputs_table(doc, c)
    if tk.TANK_DEFAULTS['floor']['T_limit_annexM'] != fl_c['T_limit_annexM']:
        doc.review(f"The code default floor.T_limit_annexM is {tk.TANK_DEFAULTS['floor']['T_limit_annexM']:.0f} C, "
                   f"but the YAML uses {fl_c['T_limit_annexM']:.0f} C. If the YAML key were dropped, the default "
                   "would apply silently. (tank.py TANK_DEFAULTS, YAML tank.floor)")
    doc.h3('Fixed internal constants and correlations')
    _constants_table(doc)

    # ---------------- 5.1
    doc.h2('5.1 Hot-zone layout and sizing')
    doc.p("Per-stick inputs (StickInputs, from tank.stick_from_run on a discharge run):")
    doc.bl([
        "P_stick = E_e / t_end: the discharge-average electrical power. E_e is the integral of P_elec while the "
        "stick is in, and t_end is the stick-removal time (shade-open cutoff).",
        "E_stick,th = rho V int c_p dT from T_min to T_max: the stored swing of the stick's graphite annulus, with "
        "V_block = pi (r_out^2 - r_in^2) H.",
        "eta_block = E_e / E_rad: electricity out per unit of heat into the stick, used to convert a heat loss into "
        "an electricity loss (fallback 0.45 if E_rad = 0).",
        "Histories T_avg(t) (block volume mean), T_top(t) and T_bot(t) (radially area-weighted means of the top and "
        "bottom axial rings). These come from snapshots every 600 s, interpolated onto the solver steps.",
        "Heater cavity area A_hcav = pi r_hc^2 (from the heater element radius and area_ratio, or the optimizer's heater "
        "search). E_sink is the roof + floor sink energy removed during the run (section 5.7).",
    ])
    from . import layout as LY
    doc.p("Layout (thermal/layout.py, tank.layout; the optimizer input layout.pattern). Sticks and heaters sit in one "
          "graphite matrix in one of four patterns, and every stick must have its full ring of heater neighbours "
          "(perimeter heaters), so in general N_h != N_ps:")
    doc.table(['pattern', 'heaters around a stick', 'sticks around a heater', 'heaters per stick (bulk)',
               'k = plan area per stick / d^2'],
              [[nm, str(p['around_stick']), str(p['around_heater']), f"{p['rho']:g}", f"{p['k']:.3f}"]
               for nm, p in LY.PATTERNS.items()], widths=[1.3, 1.3, 1.3, 1.3, 1.6])
    doc.p("Stick count (size_hot_zone -> layout.plan): N_req = ceil(P_rated / P_stick). Below 20 sticks only "
          "'feasible' counts are allowed: the smallest Tier A count >= N_req. A Tier A cluster has every stick fully "
          "surrounded, no empty stick slot enclosed by its heaters, every slot inside its convex hull filled, and "
          "at least three mirror axes (D3 / D4 / D6). From 20 sticks N_ps = N_req, in a round cluster (the N_ps stick "
          "sites nearest the high-symmetry centre that needs the fewest heaters); N_h is every heater site next to "
          "a stick, so the heaters per stick approach the bulk ratio as the plant grows.")
    doc.code(LY.table())
    doc.fig(LY.plot_patterns(), "Feasible (Tier A) layouts below 20 sticks: sticks (blue squares) and heaters "
                                "(red circles) at unit stick-heater spacing; title N_ps + N_h.", width=6.4,
            name='Figure 5.1a')
    doc.eq(r'A_s = \pi r_{out}^2 + \rho\,A_{hcav} = k\,d^2,\qquad D_{hot} = 2\left(d \max_i |x_i| + d/2\right)',
           where='A_s: plan area per stick (its cell circle: graphite + cavity, and its bulk share rho of heater '
                 'cavities); d: stick-heater centre spacing; x_i: unit-spacing positions of every stick and heater '
                 'about the cluster centroid. Checkerboard: k = 2 and d = the old pitch p = sqrt(A_c / 2).')
    doc.eq(r'm_g = \rho_g H \left[(N_{ps} + N_h)\frac{k d^2}{1 + \rho} - N_{ps} S^2 - N_h A_{hcav}\right],\qquad '
           r'\phi_{fill} = \frac{N_{ps}\,\pi (r_{out}^2 - r_{in}^2) H}{A_{hot} H}',
           where='m_g: graphite (cost, column and foundation load): every cavity\'s lattice cell (k d^2 / (1 + rho) '
                 'per site) less the cavities, so the perimeter heaters carry their graphite; the rest of the '
                 'hot-zone circle is insulation, while the side losses use the full circle (conservative). The '
                 'stick cells\' graphite N_ps V_block is the part that swings (E_th; phi_fill spreads the roof / '
                 'floor loss over the cells, section 5.7).')
    doc.eq(r'E_{th} = N_{ps} E_{stick,th},\qquad \epsilon_E = \frac{N_{ps} E_{stick,e}}{P_{rated}\,t_{dur}} - 1',
           where='epsilon_E (energy_mismatch) is reported and warned above 1 %; rounding up to a feasible count '
                 'makes it positive. The optimizer makes epsilon_E < -1 % a constraint.')
    doc.p("Heaters: the N_h heaters charge the stick cells' graphite, so the heater cell holds N_ps / N_h of a "
          "stick cell's graphite (heater.cell_graphite_ratio, section 4.6) and the heater power, element, leads, "
          "heater capex and the charging draw are per heater x N_h. The optimizer fixes (N_ps, N_h) before the "
          "heater search with a provisional heater (optimize.plant_counts) and repeats the search once if the "
          "searched cavity changes them. Hung equipment for the foundation: hung_kg_per_pair x (N_ps + N_h) / 2.")
    doc.p("Graphite walls between cavities (tank.cavity_webs -> layout.webs; the optimizer constraint is min wall "
          ">= geometry.min_wall). The stick cavity is a square of side S, whose perimeter equals that of the r_in "
          "circle; the heater cavity a circle of radius r_hc. Checkerboard: squares turned 45 deg, corners to the "
          "heaters and faces to the diagonal stick neighbours:")
    doc.eq(r'S = \pi r_{in}/2,\quad w_{sh} = d - S/\sqrt{2} - r_{hc},\quad w_{ss} = \sqrt{2}\,d - S,'
           r'\quad w_{hh} = \sqrt{2}\,d - 2 r_{hc}')
    doc.p("Hexagonal patterns: the square's rotation (0-90 deg in 0.5 deg steps) that gives the thickest thinnest "
          "wall; walls from the exact point-to-square and square-to-square distances to the nearest neighbours of "
          "each kind.")
    if cy is not None:
        r_in, r_out, r_hc, lab = cy.m.r_in, cy.m.r_out, cy.hm.r_in, 'example design'
    elif ctx.get('top'):
        t = ctx['top']
        r_in, r_out, r_hc, lab = t['r_in'], t['r_out'], t['heater_cavity_D'] / 2, 'best optimizer design'
    else:
        r_in, r_out, r_hc, lab = 0.943, 1.512, 0.345, 'illustrative dimensions'
    fig, p_, webs = _fig_plan(r_in, r_out, r_hc, lab)
    doc.fig(fig, f"Checkerboard hot zone in plan, bulk ({lab}: r_in {r_in:.3f} m, r_out {r_out:.3f} m, r_hc {r_hc:.3f} m). "
                 "Stick cavities (blue sticks) and heater cavities (red heaters) alternate on pitch p. The dashed "
                 "circle is the stick cell's analysis annulus (area pi r_out^2), which overlaps its neighbours: it "
                 "is an equal-area model, not a geometric boundary.", width=5.6, name='Figure 5.1')
    sq = (math.pi - math.pi ** 2 / 4) * r_in ** 2 / (math.pi * (r_out ** 2 - r_in ** 2))
    doc.review(f"Two different cavity areas are used. V_block subtracts a circular stick cavity, pi r_in^2, but "
               f"cavity_webs and the layout use a square of equal perimeter, area pi^2 r_in^2/4 (21 % smaller). "
               f"The real graphite per stick is therefore about {sq * 100:.0f} % more than the swing graphite "
               f"modelled ({lab}). [confidence medium; effect a few % on the stored heat] (tank.size_hot_zone)")
    doc.review("Perimeter graphite and heaters: the graphite outside the stick cells (around the perimeter heaters and "
               "the half-spacing ring) is costed and loaded but taken as held hot, not swung; the heater cell is the "
               "average heater (N_ps / N_h of a stick cell's graphite), so perimeter heaters, which charge less "
               "graphite, would in practice run at lower power than the interior ones. The analysis cells stay "
               "annular for every pattern. [confidence medium; matters most for small plants and hex6]")

    # ---------------- 5.2
    doc.h2('5.2 Tank geometry')
    doc.eq(r'D_{tank} = D_{hot} + 2 t_{side} + 2 c_{shell},\qquad H_{tank} = L_{bot} + H + t_{top} + H_{wd} + H_{deck}')
    doc.eq(r'H_{wd} = L_{ps} + t_{top}\;\;(\mathrm{default}),\qquad r_i = D_{hot}/2,\quad r_o = r_i + t_{side},\qquad '
           r'p_{bear} = g\left(\frac{m_g}{A_{hot}} + \rho_{thx} t_{top}\right)')
    doc.p("p_bear is the areal load on the column grid: storage graphite plus the roof Thermax resting on it. "
          "Sticks, liners and heaters hang from the deck. Shade vanes (~1 % of the block mass) are neglected. The "
          "bottom Thermax is assumed to carry no load. Warnings: H_tank outside 10-30 m, D_tank outside 15-80 m (the "
          "usual API 650 range), and p_bear > 225 kPa (CSP hot-tank precedent). The optimizer limits are "
          f"D_tank <= {ctx['opt']['limits']['D_tank_max']:g} m and H_tank <= {ctx['opt']['limits']['H_tank_max']:g} m.")
    doc.review("The default withdrawal headspace H_wd = L_ps + t_top (marked 'assumption' in the YAML) adds t_top a "
               "second time on top of the roof Thermax already in H_tank. It has no allowance for hoist or crane "
               "clearance or for liner and shade hardware above the stick. It feeds the H_tank <= 30 m constraint "
               "and the shell cost. [confidence medium] (tank.py:333-334)")

    # ---------------- 5.3
    doc.h2('5.3 Insulation and heat-loss paths')
    doc.p("Every path is steady 1-D conduction with temperature-dependent conductivity, handled exactly with the "
          "Kirchhoff transform. The hot faces are given temperatures (no contact resistance and no gap between the "
          "graphite and the insulation). For Thermax, k = k_0 exp(B T[C]) (section 7), so:")
    doc.eq(r'\Phi(T_c, T_h) = \int_{T_c}^{T_h} k_{thx}(T)\,dT = \frac{k_0}{B}\left(e^{B T_h} - e^{B T_c}\right)')
    fig = _fig_section()
    doc.fig(fig, "Schematic half section of the tank (not to scale) with the heat-flow paths. Red: insulation "
                 "losses; blue: floor-cooling air. The slab, cellular glass and mat below the floor channels are "
                 "costed but carry no modelled heat.", width=6.6, name='Figure 5.2')
    doc.fig(_fig_network(), "Heat-flow network. Each box is a 1-D element solved in closed form or by a scalar root "
                            "find; the floor block is a fixed point on the floor temperature.", width=6.6,
            name='Figure 5.3')
    doc.h3('Side wall (side_wall)')
    doc.p("A cylindrical Thermax wall from r_i to r_o over the block height H only (the side insulation next to the "
          "bottom stack and the roof is covered by the edge term). The steel shell's own resistance is neglected. "
          "The shell surface temperature T_s is found by brentq on [T_amb, T_hot] (xtol 1e-6) from:")
    doc.eq(r'Q_{side} = \frac{2\pi H\,\Phi(T_s, T_{side})}{\ln(r_o/r_i)} = 2\pi r_o H\left[h_{out}(T_s - T_{amb}) + '
           r'\varepsilon\sigma(T_s^4 - T_{amb}^4)\right]',
           where=f"h_out = {sh['h_out']:g} W/m2K (fixed, no wind model), eps = {sh['emissivity']:g}; radiation to "
                 "T_amb (no sky temperature, no solar gain); temperatures in K in the T^4 terms.")
    doc.h3('Edges')
    doc.eq(r'Q_{edge} = 2 \times 0.54\,\pi D_{hot}\,\Phi(T_s, T_{side})',
           where='conduction shape factor of a two-wall edge (S = 0.54 x edge length), applied to the top and '
                 'bottom ring edges, edge length = hot-face circumference, from T_side to the shell temperature.')
    doc.review(f"The edge model applies the equal-thickness two-wall shape factor to walls of "
               f"{c['t_side']:g} / {c['t_top']:g} / {c['L_bot']:g} m (side / top / bottom). Both edges use T_side "
               f"and T_shell, but the bottom corner actually ends on the cooled floor. It is about "
               f"{ref.loss_peak['Q_edge'] / ref.loss_peak['Q_total'] * 100:.0f} % of the reference loss, so the error "
               "is bounded, but it has not been checked against FEA. [confidence medium-low] (tank.py:543)")
    doc.h3('Roof (roof)')
    doc.p("Planar Thermax over A_hot, then the argon headspace (natural convection plus radiation between two "
          "surfaces), then the steel deck (area A_tank), then ambient. Nested root finds: the inner one gives the "
          "Thermax top-face temperature T_c for a deck temperature T_d, and the outer one T_d in (T_amb, T_hot - 1).")
    doc.eq(r'\frac{\Phi(T_c, T_{top})}{t_{top}} = h_{hs}(T_c - T_d) + \varepsilon_{hs}\sigma(T_c^4 - T_d^4),\qquad '
           r'\varepsilon_{hs} = \left[\frac{1}{\varepsilon_{ins}} + \frac{A_{hot}}{A_{tank}}\left(\frac{1}{\varepsilon} '
           r'- 1\right)\right]^{-1}')
    doc.eq(r'Q_{top} = A_{hot}\,\frac{\Phi(T_c, T_{top})}{t_{top}} = A_{tank}\left[h_{out}(T_d - T_{amb}) + '
           r'\varepsilon\sigma(T_d^4 - T_{amb}^4)\right]',
           where=f"h_hs = {sh['h_headspace']:g} W/m2K, eps_ins = {sh['emis_insulation']:g}. The headspace height "
                 "does not enter, and the upper shell walls around it are not a loss path.")
    doc.review("Penetration and door losses are not modelled: N_ps stick penetrations, N_h heater leads, 8 shade axles "
               "per stick through t_top, and the stick-hole doors during charge and hold. With hundreds of sticks "
               "these could be comparable to the 1-D roof loss. [confidence medium; effect unknown, possibly tens "
               "of % of Q_top] (tank.py:370-387)")
    doc.h3('Bottom stack')
    doc.p("The columns (section 5.4) and the bottom Thermax are parallel 1-D paths over L_bot with no lateral "
          "exchange. The single factor f_3D corrects the column heat for 3-D effects. The cold face is the "
          "floor-plate average temperature T_f from the floor model (section 5.5).")
    doc.eq(r'Q_{bot} = N_{col} f_{3D} Q_{1D}(T_{bot}, T_f) + (A_{hot} - N_{col}\bar A_{col})\,'
           r'\frac{\Phi(T_f, T_{bot})}{L_{bot}},\qquad \bar A_{col} = \frac{V_g + V_{ox}}{L_{bot}}')
    doc.h3('Loss evaluations kept on the result')
    doc.p("losses(T_side, T_top, T_bot, ..., T_air) sums the four paths for the given hot faces. T_air is only the "
          "floor-air inlet (default T_amb); the shell and deck always see T_amb. tank_system keeps:")
    doc.bl([
        "design: uniform T_max, floor air at T_air_design. floor_design is the matching floor solution; the "
        "optimizer's floor-temperature constraint uses it.",
        "loss_peak: uniform T_max, annual-mean air. The optimizer takes the floor fan power from it.",
        "loss_fn(T_hot, T_top=None, T_bot=None): total loss for a uniform (or split) hot face with annual-mean air. "
        "The optimizer uses it for the hold, at T_max for the whole hold, which is an upper bound.",
        "loss_cycle: each path averaged in time over the discharge history, with T_side = T_avg(t), T_top = top-ring "
        "mean and T_bot = bottom-ring mean. Up to 60 evenly spaced history points, trapezoid rule. The integrals "
        "E_loss (all paths) and E_side (side + edges) over the discharge are kept for the stick correction.",
    ])
    doc.fig(_fig_losses(ref, c), "Loss by path against a uniform hot-face temperature, reference plant (v3 point, "
                                 "YAML insulation), annual-mean air.", width=6.0, name='Figure 5.4')
    doc.review(f"'Design-day' changes only the floor-air inlet ({c['T_air_design']:g} C). The shell, deck and roof "
               f"still see T_amb = {c['T_amb']:g} C, with no solar gain or sky temperature, so the design shell and "
               "deck temperatures are optimistic. The loss magnitude is hardly affected because the insulation "
               "dominates. [confidence medium] (tank.py:541-545, 710)")
    doc.review(f"Thermax conductivity is a single exponential, k = {0.04916:g} exp({0.0011:g} T[C]), used from ambient to "
               f"2500 C, with no source or fill gas noted. The side, roof and bottom-Thermax paths "
               f"({(ref.loss_peak['Q_side'] + ref.loss_peak['Q_top'] + ref.loss_peak['Q_bot_thx']) / ref.loss_peak['Q_total'] * 100:.0f} % "
               "of the reference loss) scale directly with it. [confidence medium] (properties.py:297-309)")
    doc.h3('Side and edge loss correction of the stick (stick_correction)')
    doc.p("Roof and floor losses act on every cell, and with block_axial_losses they are already a sink in the cell "
          "runs (section 5.7). Side and edge losses reach only the perimeter cells. They are spread over all sticks "
          "after the run and converted to electricity at the stick's average conversion ratio:")
    doc.eq(r'e_{loss} = \frac{1}{N_{ps}}\int_0^{t_{dis}}(Q_{side} + Q_{edge})\,dt,\qquad \Delta E_e = '
           r'\eta_{block}\,e_{loss},\qquad f_{pen} = \Delta E_e / E_{stick,e}',
           where='if the roof and floor were not in the run (E_sink = 0), e_loss uses all paths instead. '
                 'P_stick,net = (E_stick,e - Delta E_e) / t_dis.')
    doc.review("The side and edge penalty is spread evenly over all sticks. In reality the perimeter cells lose this "
               "heat, reach the shade-open T_min sooner and change the plant's discharge profile; the linear "
               "eta_block conversion is a first-order estimate. The optimizer also uses the pass-1 penalty for the "
               "pass-2 plant (larger N, so a slightly smaller true penalty), which is mildly conservative. "
               "[confidence medium-low; effect ~ penalty_frac, typically a few %] (tank.py:747-759, optimize.py)")
    doc.review("loss_cycle and the tank report's annual economics cover the discharge only and annualise it as if "
               "constant (a warning says so). The optimizer does its own charge and hold accounting, so this is "
               "report-only. [confidence high; report only] (tank.py:744-746, 777)")

    # ---------------- 5.4
    doc.h2('5.4 Support columns')
    doc.p("Square grid of columns, each a graded graphite member (hot end) on an oxide post (cold end) with a "
          "bearing plate on the floor. The section has a constant thickness t_c, and its width follows the creep "
          "allowable at the local temperature. The columns are sized at the peak hot face T_max with the design-day "
          "floor.")
    doc.eq(r'\sigma_{allow}(T) = \min\left\{\sigma_{ref}\exp\left[\frac{Q}{nR}\left(\frac{1}{T} - \frac{1}{T_{ref}}'
           r'\right)\right],\;\sigma_{cap}\right\}',
           where=f"sigma_ref {col['sigma_ref'] / 1e6:g} MPa at T_ref {col['T_ref']:g} C (1 % strain in 30 yr, no "
                 f"further factor); Q = {col['Q_kcal']:g} kcal/mol, n = {col['n']:g} -> Q/(nR) = "
                 f"{col['Q_kcal'] * tk.KCAL / (col['n'] * tk.R_GAS):.0f} K; cap {col['sigma_cap'] / 1e6:g} MPa; "
                 "T in K.")
    doc.eq(r'p_{col} = \sqrt{\sigma_{allow}(T_{max})\,w_{max} t_c / p_{bear}},\qquad N_{col} = A_{hot}/p_{col}^2,'
           r'\qquad F = p_{bear}\,p_{col}^2',
           where='the widest section (w_max, at the hot face) is loaded exactly to its creep allowable at T_max; '
                 'columns.pitch overrides p_col. N_col is not rounded.')
    doc.eq(r'A_g(T) = \max\left(F/\sigma_{allow}(T),\; w_{min} t_c\right),\qquad A_{ox} = F/\sigma_{ox}')
    doc.p("The column geometry along its height comes from 1-D heat continuity. With Q dz/A = -k dT (exact for a "
          "varying section), each segment is described by its shape integral S = int dz/A:")
    doc.eq(r'Q_{1D} = \frac{\Psi_g(T_j, T_h)}{S_g} = \frac{\Psi_{ox}(T_f, T_j)}{S_{ox}},\qquad '
           r'\Psi(a, b) = \int_a^b k\,dT')
    doc.p("Design with the joint at T1 (default, columns.L_graphite = null); the integrals use the trapezoid rule on "
          "4001 points from T_f to T_h:")
    doc.eq(r'Q_{1D} = \frac{I(T_1) + A_{ox}\Psi_{ox}(T_f, T_1)}{L_{bot}},\quad L_g = \frac{I(T_1)}{Q_{1D}},\quad '
           r'V_g = \frac{J(T_1)}{Q_{1D}},\quad I = \int_{T_1}^{T_h} k A_g\,dT,\quad J = \int_{T_1}^{T_h} k A_g^2\,dT')
    doc.p("Then L_ox = L_bot - L_g, V_ox = A_ox L_ox, S_g = Psi_g(T1, T_h)/Q_1D and S_ox = L_ox/A_ox. The geometry is "
          "then frozen. At any other hot and cold face (cycle history, sink table) column_heat finds the joint "
          "temperature by brentq from the continuity equation above. The design raises an error if T_max <= T1 "
          "and warns if L_ox <= 0. Graphite k(T) is the Mersen 2020 table (constant table, section 5 constants); "
          "oxide k = 2.8 + 1000/T[K].")
    figc, pr = _fig_columns(ref, c, T_ref_max)
    doc.fig(figc, f"Left: creep allowable against temperature with the section stresses. The graphite width shrinks "
                  f"from w_max at the hot face until it reaches w_min (where sigma_allow = F/A_min). Right: the "
                  f"reference-case column ({ref.cols.N_col:.0f} columns at {ref.cols.pitch:.2f} m, F = "
                  f"{ref.cols.F / 1e3:.0f} kN) drawn to scale in width, with its 1-D temperature profile at the "
                  f"design point.", width=6.6, name='Figure 5.5')
    cal = tk.calibrate_f3d(copy.deepcopy(c))
    doc.p("f_3D calibration (calibrate_f3d). A 3-D FEA of the v3 design gave 774 kW through 982 m2 (pitch 1.70 m, "
          "L_bot 2.0 m, 2400 C hot face, joint at 1100 C, floor pinned at 85 C). f_3D is the scalar that makes "
          "N_col f_3D Q_1D + Q_thx match it (brentq on [0.1, 20]). Re-run with the current YAML: ",
          (f"f_3D = {cal['f_3D']:.3f}", {'bold': True}),
          f" (1-D columns {cal['Q_col_1D'] / 1e3:.0f} kW + Thermax {cal['Q_thx'] / 1e3:.0f} kW = "
          f"{cal['Q_1D'] / 1e3:.0f} kW uncorrected). The YAML uses f_3D = {col['f_3D']:g}.")
    doc.review(f"f_3D: the YAML uses {col['f_3D']:g} (calibrated; re-run gives {cal['f_3D']:.3f}), the code default "
               f"is {tk.TANK_DEFAULTS['columns']['f_3D']:g}, tests/test_tank.py pins 0.54, and earlier project notes "
               "quote 0.88 for sigma_ref 2.81 MPa. The calibration point uses T1 = 1100 C, a fixed 1.70 m pitch and "
               f"an 85 C floor, but the design uses T1 = {col['T1']:.0f} C, a computed pitch and the model's own floor "
               "temperature. It is one scalar fitted to one FEA point and is only valid near the v3 column geometry. "
               "[confidence medium; the column path is ~2/3 of the bottom loss] (YAML tank.columns.f_3D; "
               "tank.py:869-888)")
    doc.review("Only creep at the hot face is checked. Not checked: buckling (the graded foot is "
               f"{col['w_min'] * 1e3:.0f} x {col['thickness'] * 1e3:.0f} mm over ~{ref.cols.L_g:.1f} m; weak-axis "
               f"L/r ~ {ref.cols.L_g / (col['w_min'] / math.sqrt(12)):.0f}), oxide creep and strength against "
               f"temperature (sigma_ox is a constant {col['sigma_ox'] / 1e6:g} MPa up to T1), joint and bearing "
               "contact stress, lateral and seismic load, the tie grid (costed only), and differential thermal "
               "expansion of the ~35 m hot zone (graphite grows ~0.3 m in diameter at 2400 C). Creep is designed "
               "for T_max held for 30 years, which is conservative. [confidence medium] (tank.py:413-471)")

    # ---------------- 5.5
    doc.h2('5.5 Floor cooling')
    doc.p("Steel fins hang under the floor plate. Air is blown once across the full tank diameter through the "
          "channels between the fins (height fin_h + clearance). All the bottom heat Q_bot goes into this air: no "
          "heat path down into the slab is modelled, and the plate's own conduction and spreading are ignored "
          "(floor.plate_t is unused). Air properties are fixed at ~60 C.")
    doc.eq(r's = p_f - t_f,\quad H_{ch} = h_f + c,\quad D_h = \frac{4 s H_{ch}}{2(s + H_{ch})},\quad '
           r'Re = \frac{\rho v D_h}{\mu}')
    doc.eq(r'f = (0.79\ln Re - 1.64)^{-2},\qquad Nu = \frac{(f/8)(Re - 1000)Pr}{1 + 12.7\sqrt{f/8}\,(Pr^{2/3} - 1)},'
           r'\qquad h = Nu\,k/D_h')
    doc.eq(r'\eta_{fin} = \frac{\tanh(m h_f)}{m h_f},\quad m = \sqrt{\frac{2h}{k_s t_f}},\qquad '
           r'h_{eff} = h\,\frac{(p_f - t_f) + 2\eta_{fin} h_f}{p_f}',
           where='h_eff is per m2 of plate: bare base between fins plus both fin faces (adiabatic tip).')
    doc.eq(r'\dot V = v\,D_{tank}\,\frac{s H_{ch}}{p_f},\qquad \Delta T_{air} = \frac{Q_{bot}}{\rho \dot V c_p},\qquad '
           r'\Delta p = \left(f\frac{D_{tank}}{D_h} + K\right)\frac{\rho v^2}{2},\qquad P_{fan} = \Delta p\,\dot V/\eta_{fan}')
    doc.eq(r'T_{f,avg} = T_{in} + \frac{\Delta T_{air}}{2} + \frac{q_{bot}}{h_{eff}},\quad T_{f,max} = T_{in} + '
           r'\Delta T_{air} + \frac{q_{bot}}{h_{eff}},\quad T_{f,col} = T_{f,max} + \frac{Q_{col}/A_{plate} - q_{bot}}{h_{eff}}')
    doc.p("T_f,avg is the cold face of the bottom stack. T_f,max is at the outlet end with the average flux. T_f,col "
          "puts one column's heat (f_3D Q_1D) through its bearing-plate footprint with no spreading in the plate. V_dot "
          "and P_fan depend only on the geometry and v_air: the fan runs at constant speed all cycle, and v_air is an "
          "input. There is no fan-sizing loop; the floor temperature is only checked.")
    doc.p("Iterations. bottom_and_floor is a fixed point on T_f: start at T_in + 30 K, then "
          "T_f <- T_f,avg(Q_bot(T_f)) until the change is below 1e-3 K (at most 60 iterations; non-convergence "
          "is warned). _design_bottom wraps it in an outer fixed point that re-designs the columns for the new T_f: "
          f"start at T_air_design + 40 = {c['T_air_design'] + 40:g} C, tolerance 1e-3 K, at most 30 passes. The "
          "pitch does not change because the governing section is at the hot face.")
    doc.p(f"Limits: warnings, and an optimizer constraint, if T_f,max or T_f,col exceed T_limit_annexM = "
          f"{fl_c['T_limit_annexM']:g} C (API 650 Annex M, the design basis). T_limit_normal = "
          f"{fl_c['T_limit_normal']:g} C is only reported as a margin.")
    doc.fig(_fig_floor(ref, c, T_ref_max), "Floor temperatures (design-day air, hot face at T_max) and fan power "
                                           "against channel velocity for the reference plant, with the columns "
                                           "held at their design geometry. Dotted line: YAML v_air.", width=6.0,
            name='Figure 5.6')
    q_now = ref.floor_design['T_floor_col']
    Q_col_now = col['f_3D'] * tk.column_heat(ref.cols, T_ref_max, ref.cols.T_floor_design)[0]
    dT_spot = (2.1e3 - Q_col_now) / col['plate_area'] / ref.floor_design['h_eff']
    doc.review(f"The column hot-spot check uses the f_3D-scaled column heat ({col['f_3D']:g} x 1-D). The FEA behind "
               "f_3D says a column carries ~1.2 kW at its top and ~2.1 kW at its foot (it picks up heat sideways "
               "from the Thermax), and f_3D < 1 only matches the total. With 2.1 kW at the foot, the reference "
               f"column-spot temperature ({q_now:.0f} C) would rise by about {dT_spot:.0f} K. Ignoring plate "
               "spreading works the other way. [confidence high that the foot heat is understated] "
               "(tank.py:487, 512, 516)")
    doc.review("Floor-air simplifications: properties are fixed at 60 C for both 15 C and 40 C inlets (~5-10 % on "
               "density, affecting dT_air and fan power). D_h treats the fin gap plus clearance as a closed duct. "
               "Every channel is D_tank long (chords are shorter, so the pressure drop is conservative). Manifolds and "
               "louvres are only K_minor. There is no conduction into the slab. floor.plate_t is unused. "
               "[confidence medium-low; effect a few K / %] (tank.py:497-517)")
    doc.review("A non-converged outer column/floor fixed point (_design_bottom, 30 passes) gives no warning; only "
               "the inner loop's flag is reported. [confidence high; low impact, it converges in a few passes "
               "in practice] (tank.py:695-701)")

    # ---------------- 5.6
    doc.h2('5.6 Foundation')
    doc.eq(r'p_{found} = g\,\frac{m_g + m_{thx} + m_{col,g} + m_{ox} + m_{steel} + m_{deck}}{A_{tank}}',
           where='m_steel = shell + bottom + roof; m_deck at its central value. Compared with soil_allowable '
                 f"({c['soil_allowable'] / 1e3:g} kPa): a warning, and a constraint in the optimizer.")
    doc.p("The load is taken as uniform over the tank footprint. The column-grid bearing p_bear (section 5.2) is "
          "checked separately against 225 kPa.")
    doc.review(f"The foundation pressure leaves out the {c['costs']['mat_thickness']:g} m concrete mat itself "
               f"(~{2400 * tk.G * c['costs']['mat_thickness'] / 1e3:.0f} kPa at 2400 kg/m3), the HT slab, cellular "
               "glass, floor-plate upgrade and fins, and the hung equipment (sticks, liners, heaters, shades, "
               "carried through the deck and shell). It also ignores ring-wall concentration under the shell. "
               f"Reference: {ref.foundation_pressure / 1e3:.0f} kPa computed. [confidence medium-high; matters if "
               "soil_allowable is a gross bearing value] (tank.py:738-743)")

    # ---------------- 5.7
    doc.h2('5.7 Per-cell roof and floor loss sink')
    doc.p("Every cell loses heat straight up through the roof and down through the bottom stack, so these two "
          "paths are applied inside the cell simulations as a heat sink (Q_sink_fn) on the top and bottom block "
          "rings of every stick and heater cell. Section 6 describes how the transient solver applies it. "
          "CellLossSink.from_system tabulates the plant's per-area fluxes against a uniform hot-face temperature, "
          "with annual-mean air:")
    doc.eq(r'q_{top}(T) = Q_{top}(T)/A_{hot},\qquad q_{bot}(T) = Q_{bot}(T)/A_{hot}\quad '
           r'(T \in [T_{lo} - 150,\; T_{hi} + 100],\;40\ \mathrm{points,\ clamped\ outside})')
    doc.p("For a block model with radial node edges r_0 = r_in ... r_n = r_out, the sink on radial node j of the top "
          "ring (index -1, rings run bottom-up) and of the bottom ring (index 0) is evaluated at that node's own "
          "temperature:")
    doc.eq(r'Q^{top}_j = q_{top}(T_{top,j})\,\pi(r_{j+1}^2 - r_j^2) f_{open},\qquad Q^{bot}_j = q_{bot}(T_{bot,j})\,'
           r'\pi(r_{j+1}^2 - r_j^2) f_{open},\qquad f_{open} = 1/\phi_{fill}')
    doc.eq(r'\sum_j \pi(r_{j+1}^2 - r_j^2) f_{open} = \frac{\pi(r_{out}^2 - r_{in}^2)}{\phi_{fill}} = '
           r'\frac{A_{hot}}{N_{ps}}',
           where="each cell carries its full share of the plant's roof and floor area, including the cavities and "
                 "the perimeter ring (cavity ceilings and floors re-radiate to the graphite). The heater block has "
                 "the same graphite volume, so the same f_open gives it A_hot/N_ps too.")
    doc.p("The cell runs happen before the plant is known, so cell_loss_sink builds a provisional plant: "
          "N_prov = ceil(P_rated t_dur / (0.45 E_stick,th)), P_stick = P_rated/N_prov, the model's geometry and heater "
          "cavity, and columns designed at T_max. Its sink is tabulated over [T_min, T_max]. The per-area fluxes "
          "barely depend on plant size (the floor-air temperature rise moves q_bot by ~1 %). The final plant's sink is "
          "kept as TankResult.cell_sink for comparison. Users: sizing.run_discharge (if block_axial_losses), "
          "optimize._charge and cycle.py (always). There is no double counting because the cell model's own block "
          "loss is off (block_heat_loss_bool: False).")
    doc.fig(_fig_sink(ref.cell_sink, T_ref_max, 1900.0), "Per-area roof and floor fluxes of the reference plant "
                                                         "(CellLossSink table). Shaded: the reference swing.",
            width=5.6, name='Figure 5.7')
    doc.review("The cell runs use the provisional plant's sink (N from a 0.45 conversion ratio, not the final N). "
               "The docstring estimates ~1 % on q_bot. The final-plant sink is computed but not fed back. "
               "[confidence high; effect ~1 % of the roof + floor loss] (tank.py:844-863)")

    # ---------------- 5.8
    doc.h2('5.8 Tank cost model')
    doc.p("Each line is quantity x rate, with both as [low, central, high] triples multiplied element by element "
          "(low quantity x low rate, and so on), so the low-high band is wider than a rate-only band. All rates are "
          "placeholders. Groups: containment (shell, bottom, roof, deck, gas tightness), insulation (three Thermax "
          "lines), supports (column graphite, oxide, bearing plates, tie grid) and floor & foundation (floor, slab, "
          "cellular glass, mat, fans). Storage graphite is reported separately as a pass-through.")
    doc.p(f"Steel parameters: t_shell {c['costs']['t_shell']:g} m (average), t_bottom {c['costs']['t_bottom']:g} m "
          f"+ {c['costs']['bottom_annular_frac'] * 100:.0f} % annular ring, roof {c['costs']['roof_kg_m2']:g} kg/m2, "
          f"mat thickness {c['costs']['mat_thickness']:g} m, steel {tk.RHO_STEEL:.0f} kg/m3. In the reference case "
          "the hot zone is given directly (no sticks), so the gas-tightness line has only its fixed part.")
    _cost_table(doc, c, ref, plant)
    doc.p("The optimizer uses the central values only (index [1]): tank.cost['total'][1] and "
          "tank.cost['storage_graphite'][1], then multiplies all capex by (1 + soft_cost_frac = "
          f"{ctx['opt']['economics']['soft_cost_frac']:g}). The tank report also prices lost heat at "
          f"economics.elec_price = {c['economics']['elec_price']:g} $/MWh / heater_eff; the optimizer ignores this "
          f"and uses its own price ({ctx['opt']['economics']['elec_price']:g} $/MWh).")
    doc.review("All tank cost rates are placeholders (tank.py:38, YAML). Not costed here: penetrations for shade axles "
               "(8 per stick), withdrawal doors or hatches, the stick-handling crane, the argon system, "
               "side-insulation support, and instrumentation. [confidence high that they are placeholders] "
               "(tank.py:556-610)")

    # ---------------- 5.9
    doc.h2('5.9 Calculation order and worked example')
    doc.p("tank_system(stick, cfg) runs in a fixed order (figure). There is no iteration between the hot zone and "
          "the tank; the only loops are the two floor/column fixed points. The side wall and roof are solved "
          "directly at every call.")
    doc.fig(_fig_order(), "tank_system calculation order (tank.py:704-781).", width=5.6, name='Figure 5.8')
    doc.p("How the optimizer uses it (optimize.evaluate_design):")
    doc.nl([
        "stick_from_run on the clean, shade-open-sized discharge; cavity_webs (rotated) -> minimum-wall constraint.",
        "Pass 1: tank_system with the searched heater cavity -> penalty_frac f_pen.",
        "AC stick: E_ac = E_stick,e (1 - f_pen) eta_out, P_stick = E_ac / t_dis. Pass 2: tank_system on it. This "
        "plant (N, geometry, columns, floor, costs, loss_fn) is used from here on.",
        "Constraints: energy mismatch < -tol, D_tank, H_tank, max(T_f,max, T_f,col) > T_limit_annexM, "
        "p_found > soil_allowable.",
        "Energy: side + edge at the mid temperature during the charge; loss_fn(T_max) during the hold; "
        "loss_peak P_fan for the whole cycle.",
        "Capex: tank cost total and storage graphite, central values.",
    ])
    doc.p("Worked reference case: tank_system_reference with the YAML tank block (v3 design point: D_hot 35.4 m, "
          "H 10 m, 10,490 t graphite, 3.0 GWh_th, uniform sweep 2400 -> 1900 C for the cycle average)"
          + ("; the right column is the example design's plant (section 8)." if plant is not None else "."))
    rv = _summary_vals(ref)
    if plant is not None:
        pv = _summary_vals(plant)
        rows = [[a, u, b, pv[i][2]] for i, (a, u, b) in enumerate(rv)]
        doc.table(['quantity', 'unit', 'reference (v3)', 'example design'], rows, widths=[2.6, 1.1, 1.7, 1.7],
                  size=8)
    else:
        doc.table(['quantity', 'unit', 'reference (v3)'], [list(r) for r in rv], widths=[3.0, 1.3, 2.8], size=8)
    if ref.warnings:
        doc.p(('Reference warnings: ', {'bold': True}), '; '.join(ref.warnings))
    if plant is not None and plant.warnings:
        doc.p(('Example design warnings: ', {'bold': True}), '; '.join(plant.warnings))

    return [
        ['tank', 'side-wall shell temperature', 'brentq on [T_amb, T_hot], xtol 1e-6 K', 'tank.side_wall'],
        ['tank', 'roof: Thermax top face / deck', 'nested brentq, xtol 1e-6 K', 'tank.roof'],
        ['tank', 'column design T grid', '4001 points T_f..T_max, trapezoid', 'tank.design_columns'],
        ['tank', 'column joint temperature', 'brentq on [T_f, T_hot], xtol 1e-6 K', 'tank.column_heat'],
        ['tank', 'floor / bottom-stack fixed point', 'start T_in + 30 K, tol 1e-3 K, max 60', 'tank.bottom_and_floor'],
        ['tank', 'column re-design / floor fixed point',
         f"start T_air_design + 40 = {c['T_air_design'] + 40:g} C, tol 1e-3 K, max 30 (unwarned)",
         'tank._design_bottom'],
        ['tank', 'cycle-average losses', '<= 60 evenly spaced history points, trapezoid', 'tank._system'],
        ['tank', 'cell sink table', '40 points on [T_lo - 150, T_hi + 100] C, clamped', 'CellLossSink.from_system'],
        ['tank', 'provisional sink plant', 'eta_e = 0.45 (N from energy)', 'tank.cell_loss_sink'],
        ['tank', 'f_3D calibration', 'brentq on [0.1, 20], xtol 1e-6', 'tank.calibrate_f3d'],
        ['tank', 'floor air properties', 'fixed at ~60 C', 'tank.AIR'],
        ['tank', 'f_3D (in use)', f"{col['f_3D']:g}", 'YAML tank.columns.f_3D'],
    ]


def alternatives(doc, ctx):
    c = _cfg(ctx)
    doc.h2('Tank and plant (section 5)')
    doc.bl([
        "block_axial_losses: false. sizing.run_discharge then runs the stick with no roof or floor sink, E_sink = 0, "
        "and tank_system's stick correction takes all paths post hoc (e_loss from E_loss instead of E_side). The "
        "optimizer forces true, and optimize._charge and cycle.py attach the sink regardless of the flag.",
        "columns.L_graphite = x (m): fixes the graphite segment length (e.g. an as-built design). The joint "
        "temperature becomes an output (brentq on the continuity equation), and T1 is then the oxide's maximum "
        "service temperature, with a warning if the joint exceeds it.",
        "columns.pitch = x (m): fixes the column grid instead of deriving it from w_max and the creep allowable. "
        "The hot-face section is then F/sigma_allow(T_max) and may exceed w_max without a warning.",
        f"H_wd = x (m): sets the withdrawal headspace directly (default L_ps + t_top; current YAML "
        f"{'null' if c['H_wd'] is None else c['H_wd']}).",
        "bottom_and_floor(..., T_floor=x) pins the floor temperature (used for calibration and comparison with FEA "
        "that has a fixed floor).",
        "losses(..., T_air=x) evaluates any loss state with another floor-air inlet; the design-day call uses "
        "T_air_design. The shell and roof always use T_amb.",
        "cavity_webs(rotated=False): stick squares aligned with the grid (faces to the heaters). The optimizer and "
        "cycle use rotated=True.",
        "tank_system_reference(cfg, D_hot, H_block, m_graphite, E_th, T_min, T_max): the plant with the hot zone "
        "given directly; the cycle average is a uniform linear sweep T_max -> T_min.",
        "calibrate_f3d(...): re-fits f_3D to an FEA bottom heat (defaults: the v3 point).",
    ])
