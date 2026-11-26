"""Model documentation, sections 4.2-4.4: radiation in the stick cavity, the rotating-vane
shades, and the liner / liner-bottom models (plus their alternatives for Appendix A).

Numbers are pulled live: the in-use model is built from the YAML (view factors from the
on-disk cache when present) and solved at a few wall temperatures; code line references
are located by searching the source, so they follow edits."""
import copy
import inspect
import os
import time

import numpy as np

SIGMA = 5.67036713e-8


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _src(mod):
    path = inspect.getsourcefile(mod)
    return os.path.basename(path), open(path, encoding='utf-8').read().splitlines()


def _ref(mod, pat=None, obj=None, nth=1):
    """'file.py:N' of the nth line containing pat, or 'file.py:a-b' of obj's source."""
    name, lines = _src(mod)
    if obj is not None:
        src, start = inspect.getsourcelines(obj)
        return f"{name}:{start}-{start + len(src) - 1}"
    k = 0
    for i, l in enumerate(lines, 1):
        if pat in l:
            k += 1
            if k == nth:
                return f"{name}:{i}"
    return name


def _f(x, d=3):
    return f"{x:.{d}g}"


def _model(ctx):
    """Build the in-use model once and run the probe solves; cached in ctx."""
    if '_secB' in ctx:
        return ctx['_secB']
    from .build import build_tinless
    from . import correlations as C
    t0 = time.time()
    cfg = copy.deepcopy(ctx['base_cfg'])
    m = build_tinless(cfg)
    ax = getattr(m.sim, 'axial_model', None)
    R = dict(cfg=cfg, m=m, ax=ax, C=C, ok=ax is not None and hasattr(ax, 'anchors'))
    if not R['ok']:
        ctx['_secB'] = R
        return R
    a = ax.anchors['open']
    R['a'] = a
    ax.ALL_ANCHORS = True            # the control-window figures and tables show all three anchors
    T_min = float(cfg['storage']['T_min'])
    T_max = T_min + float(cfg['storage']['block_temp_range'])
    R['T_min'], R['T_max'] = T_min, T_max
    # control window over uniform wall temperatures
    Ts = np.linspace(T_min - 50, T_max + 50, 12)
    win = []
    for T in Ts:
        s = ax.solve(np.full(ax.n, T), control=True)
        win.append((T, s['Q_cell_min'], s['Q_cell_mid'], s['Q_cell_max'], float(s['Q_cell'].sum()), s['s'],
                    s['vane_angle'], s['balance_err']))
    R['win'] = np.array(win)
    R['newton_per_call'] = np.mean([v.stats['newton_its'] / max(v.stats['calls'], 1) for v in ax.anchors.values()])
    # timing of one controlled hybrid solve at mid temperature (warm)
    t1 = time.time()
    s_mid = ax.solve(np.full(ax.n, 0.5 * (T_min + T_max)), control=True)
    R['t_solve'] = time.time() - t1
    R['s_mid'] = s_mid
    # blend vs direct ray trace at intermediate angles (wall linear hot bottom -> cool top)
    try:
        from .axial import AxialRadiation
        from types import SimpleNamespace
        g = ax.g
        zc = 0.5 * (g['edges'][1:] + g['edges'][:-1])
        Tb, Tt = T_max - 125.0, T_max - 225.0
        Tw = Tb + (Tt - Tb) * (zc - zc.min()) / (zc.max() - zc.min())
        rad = cfg.get('radiation', {}) or {}
        n_rays = int(rad.get('vane_rays', 2 ** 14) or 2 ** 14)
        cache = rad.get('vf_cache_dir', None)
        pad = None
        if ax.pad is not None:
            pad = dict(t_rgf=m.mesh.t_rgf, t_lb=m.mesh.t_lb, eps_R=ax.pad.eps_R, T_cap=ax.pad.T_cap,
                       include_cells=ax.pad.include_cells)
        from scipy.optimize import brentq
        comp = []
        for th in (2.0, 5.0, 30.0):
            enc = ax._enclosure(C, ax.lay, th, 2, n_rays, cache)
            vp = SimpleNamespace(C=enc.F[None], A_ring=enc.A, slat=enc.slat, names=enc.names, idx=enc.idx,
                                 pairs=enc.pairs)
            so = AxialRadiation(m.sim, g, {'porous': vp}, pad=pad).solve(Tw, f=1.0)
            Qc = float(so['Q_cell'].sum())
            s = brentq(lambda x: ax.solve(Tw, control=False, s=x)['Q_cell'].sum() - Qc, 0.0, 2.0, xtol=1e-9)
            hb = ax.solve(Tw, control=False, s=s)
            Qt, Qh = so['Q_block'], hb['Q_block']
            nl = ax.n_low
            comp.append(dict(theta=th, s=s, ang=hb['vane_angle'], Qt=Qt.copy(), Qh=Qh.copy(),
                             sum_t=Qt.sum(), sum_h=Qh.sum(),
                             low_t=Qt[:nl].sum() / Qt.sum(), low_h=Qh[:nl].sum() / Qh.sum(),
                             err=np.abs(Qh - Qt).max() / Qt.sum(),
                             dTLi=float(np.abs(hb['T_liner_in'] - so['T_liner_in']).max())))
        R['comp'], R['comp_Tw'] = comp, (Tb, Tt)
    except Exception as e:                                      # keep the document building
        print(f"[docs secB] blend comparison skipped: {type(e).__name__}: {e}")
        R['comp'] = None
    R['t_total'] = time.time() - t0
    ctx['_secB'] = R
    return R


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def _fig_vanes(R):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle
    C, lay, m = R['C'], R['ax'].lay, R['m']
    A, l, lh, t = lay['A'], lay['l'], lay['liner_half'], lay['t']
    ps = m.powerstick.width / 2
    angles = (0.0, R['ax'].THETA_MID, 30.0, 90.0)
    fig, axs = plt.subplots(1, 4, figsize=(7.4, 2.25))
    octo = np.array([(1.5 * l, -0.5 * l), (1.5 * l, 0.5 * l), (0.5 * l, 1.5 * l), (-0.5 * l, 1.5 * l),
                     (-1.5 * l, 0.5 * l), (-1.5 * l, -0.5 * l), (-0.5 * l, -1.5 * l), (0.5 * l, -1.5 * l)])
    for axp, th in zip(axs, angles):
        axp.add_patch(Rectangle((-A, -A), 2 * A, 2 * A, fc='#F4E9DD', ec='#8A5A2B', lw=1.2))
        axp.add_patch(Polygon(octo, closed=True, fc='none', ec='#999999', ls=':', lw=0.7))
        axp.add_patch(Rectangle((-lh, -lh), 2 * lh, 2 * lh, fc='#DDDDDD', ec='#333333', lw=1.0))
        axp.add_patch(Rectangle((-ps, -ps), 2 * ps, 2 * ps, fc='#3D6FB6', ec='none'))
        for k, b in enumerate(C._vane_boxes(lay, th)):
            P = np.array(C._vane_box_corners(b, t))[[0, 1, 3, 2]]
            axp.add_patch(Polygon(P, closed=True, fc='#2E7D32' if b[4] == 'f' else '#7CB342', ec='k', lw=0.4))
            axp.plot(*b[0], marker='o', ms=1.6, color='k')
        lim = A * 1.04
        axp.set_xlim(-lim, lim)
        axp.set_ylim(-lim, lim)
        axp.set_aspect('equal')
        axp.set_xticks([])
        axp.set_yticks([])
        tag = {0.0: 'closed anchor', R['ax'].THETA_MID: 'mid anchor', 90.0: 'open anchor'}.get(th, 'not an anchor')
        axp.set_title(f"θ = {th:g}°  ({tag})", fontsize=8)
    fig.tight_layout(pad=0.4)
    return fig


def _fig_section(R):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    m, ax = R['m'], R['ax']
    lay, me = ax.lay, m.mesh
    A, l, lh, t = lay['A'], lay['l'], lay['liner_half'], lay['t']
    H, z0, Hl = float(me.edges[-1]), float(me.z_act0), float(me.H_low)
    tL, trgf = float(m.liner_thickness), float(me.t_rgf)
    ps = m.powerstick.width / 2
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 6.2), gridspec_kw=dict(width_ratios=[1.35, 1]))
    wb = 0.18
    for axp in (a1, a2):
        for sgn in (-1, 1):
            axp.add_patch(Rectangle((sgn * A if sgn > 0 else -A - wb, 0), wb, H, fc='#F4E9DD', ec='#8A5A2B', lw=0.8))
            xv = sgn * 1.5 * l
            axp.add_patch(Rectangle((xv - t / 2, 0), t, H, fc='#2E7D32', ec='none'))
            xl = sgn * lh
            axp.add_patch(Rectangle((xl - (tL if sgn > 0 else 0), Hl), tL, H - Hl, fc='#555555', ec='none'))
        axp.add_patch(Rectangle((-ps, z0 + 0.06), 2 * ps, H + 0.4 - z0 - 0.06, fc='#3D6FB6', ec='none'))
        axp.add_patch(Rectangle((-lh, Hl), 2 * lh, trgf, fc='#C9B79C', ec='k', lw=0.4, hatch='///'))
        axp.add_patch(Rectangle((-lh, Hl + trgf), 2 * lh, tL, fc='#555555', ec='none'))
        axp.plot([-A - wb, A + wb], [0, 0], color='k', lw=2)
        axp.plot([-A - wb, -lh], [H, H], color='k', lw=2)
        axp.plot([lh, A + wb], [H, H], color='k', lw=2)
        for z in me.edges:
            axp.plot([A, A + wb * 0.45], [z, z], color='#8A5A2B', lw=0.6)
            axp.plot([-A - wb * 0.45, -A], [z, z], color='#8A5A2B', lw=0.6)
        for z in me.edges_act:
            axp.plot([lh, lh + 0.04], [z, z], color='#555555', lw=0.6)
        axp.set_xlim(-A - wb - 0.02, A + wb + 0.02)
        axp.set_xticks([])
    a1.set_ylim(-0.3, H + 0.6)
    a1.set_ylabel('z [m]  (0 = cavity floor)')
    kw = dict(fontsize=6.5, ha='left', va='center', arrowprops=dict(arrowstyle='-', lw=0.5, color='#333333'))
    xt = -1.5 * l + t
    a1.annotate('floor_in | floor_out (reradiating)', (0.25, 0.0), (xt, 0.45), **kw)
    a1.annotate(f'block rings B0..B{ax.n - 1}: gray, ε = gr_emis(T)', (A + wb / 2, 1.6), (xt, 1.4), **kw)
    a1.annotate('vanes: full height; per ring p/m\nface bins + end faces', (-1.5 * l, 2.9), (xt, 2.6), **kw)
    a1.annotate(f'liner rings L0..L{ax.n_act - 1} (outer face)', (-lh, 8.2), (xt, 4.0), **kw)
    a1.annotate('stick inside liner\n(LinerTable, section 4.4)', (0.0, 8.8), (xt, 5.0), **kw)
    a1.annotate('ceil_in | ceil_out (reradiating;\nliner gap treated closed)', (0.3, H), (xt, 6.1), **kw)
    a1.axhline(z0, color='#B5520A', lw=0.6, ls='--')
    a1.text(A - 0.01, z0 + 0.12, f'z_act0 = {z0:g} m', fontsize=6.3, color='#B5520A', ha='right')
    a1.set_title('Section through the vane pivots (closed); horizontal scale exaggerated', fontsize=8)
    a2.set_ylim(Hl - 0.25, z0 + 0.45)
    a2.set_xlim(-0.03, 1.5 * l + t)
    a2.set_title('Zoom: liner bottom', fontsize=8)
    x2 = lh + 0.03
    a2.annotate('L0 (liner ring)', (lh, z0 + 0.33), (x2, z0 + 0.38), **kw)
    a2.annotate('stick bottom cap (T_cap)', (0.03, z0 + 0.06), (x2, z0 + 0.2), **kw)
    a2.annotate(f'graphite plate {tL * 1e3:.0f} mm;\nLB = its top face', (0.08, Hl + trgf + tL), (x2, z0 + 0.06), **kw)
    a2.annotate(f'RGF pad {trgf * 1e3:.0f} mm', (0.08, Hl + trgf / 2), (x2, Hl - 0.03), **kw)
    a2.annotate('E: liner edge (reradiating)', (lh, Hl + 0.02), (x2, Hl - 0.11), **kw)
    a2.annotate('R: pad underside (gray ε_R)', (0.05, Hl), (x2, Hl - 0.19), **kw)
    a2.axhline(z0, color='#B5520A', lw=0.6, ls='--')
    fig.tight_layout(pad=0.5)
    return fig


def _fig_flow():
    from .docs_base import flowchart
    boxes = {
        'in': (3.6, 9.0, 'Block temperatures  T_w,k = T[k, 0]  (inner FD node, every ring)', 'io'),
        'nwt': (3.6, 7.55, 'Newton on x = [J (all surfaces), σT⁴ of vane faces, liner rings, pad R]\n'
                           'rows: radiosity · vane face (Kirchhoff conduction) · liner (LinerTable)\n'
                           '· pad (LinerBottom: brentq for T_LB);  warm start, tol 1e-9·max σT_w⁴', 'loop'),
        'fin': (3.6, 6.35, 'Per anchor: Q_block per ring, Q_ps / Q_cell per liner ring,\n'
                           'pad outputs, energy balance, dQ_block/dT_w (implicit function theorem)', 'proc'),
        'tgt': (3.6, 5.3, 'Q_lo = ΣQ_cell(0°),  Q_hi = ΣQ_cell(90°),  Q_tgt = q_nom × A_cells', 'proc'),
        'd1': (1.3, 4.25, 'Q_hi ≤ Q_tgt ?', 'dec'),
        'd2': (3.6, 4.25, 'Q_lo ≥ Q_tgt ?', 'dec'),
        'br': (5.9, 4.25, 'brentq on s ∈ [0, 2]:\nΣ q_cell(T_Lo(Q_ps(s))) = Q_tgt', 'proc'),
        'op': (1.3, 3.2, 's = 2 (open)\nunder-delivers', 'out'),
        'cl': (3.6, 3.2, 's = 0 (closed)\nover-delivers', 'out'),
        'bl': (3.6, 2.15, 'Blend Q_block, Q_ps, pad outputs, dQ/dT_w piecewise-linearly in s;\n'
                          'invert LinerTable per ring → T_Lo, T_Li, Q_cell', 'proc'),
        'ef': (3.6, 1.15, 'Cells (4.5): η(T_Li, flux) → Q_elec, Q_waste;  f_shade = 1 − s/2', 'proc'),
        'bk': (3.6, 0.3, 'Block update (4.1): heat −Q_block per ring, implicit wall with −dQ/dT_w', 'io'),
    }
    arrows = [('in', 'nwt', ''), ('nwt', 'fin', ''), ('fin', 'tgt', ''), ('tgt', 'd1', '', 0.0), ('d1', 'op', 'yes'),
              ('d1', 'd2', 'no'), ('d2', 'cl', 'yes'), ('d2', 'br', 'no'), ('op', 'bl', ''), ('cl', 'bl', ''),
              ('br', 'bl', ''), ('bl', 'ef', ''), ('ef', 'bk', '')]
    groups = [(0.25, 5.85, 6.95, 8.4, 'for each anchor θ ∈ {0°, 10°, 90°}: fixed ray-traced F, own warm start')]
    return flowchart(boxes, arrows, size=(7.2, 9.5), groups=groups, fontsize=7.6)


def _fig_window(R):
    import matplotlib.pyplot as plt
    w = R['win']
    tgt = R['Q_tgt']
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.7))
    a1.plot(w[:, 0], w[:, 3] / 1e3, '-', color='#2E7D32', label='open (90°)')
    a1.plot(w[:, 0], w[:, 2] / 1e3, '--', color='#7CB342', label=f"mid ({R['ax'].THETA_MID:g}°)")
    a1.plot(w[:, 0], w[:, 1] / 1e3, '-', color='#8A5A2B', label='closed (0°)')
    a1.plot(w[:, 0], w[:, 4] / 1e3, 'o', ms=3, color='k', label='controlled')
    a1.axhline(tgt / 1e3, color='#B5520A', lw=0.8, ls=':', label='setpoint')
    a1.set_xlabel('uniform block wall T [K]')
    a1.set_ylabel('ΣQ_cell [kW]')
    a1.legend(fontsize=6.5)
    a1.grid(alpha=0.3)
    a2.plot(w[:, 0], w[:, 5], 'o-', ms=3, color='k')
    a2.set_ylabel('s (0 closed, 2 open)')
    a2.set_xlabel('uniform block wall T [K]')
    a3 = a2.twinx()
    a3.plot(w[:, 0], w[:, 6], 's--', ms=2.5, color='#2F5F9E')
    a3.set_ylabel('vane angle estimate [°]', color='#2F5F9E')
    a2.grid(alpha=0.3)
    fig.tight_layout(pad=0.4)
    return fig


def _fig_comp(R):
    import matplotlib.pyplot as plt
    comp, ax = R['comp'], R['ax']
    zc = 0.5 * (ax.g['edges'][1:] + ax.g['edges'][:-1])
    fig, a = plt.subplots(figsize=(7.0, 2.8))
    cols = ('#8A5A2B', '#2F5F9E', '#2E7D32')
    for c, col in zip(comp, cols):
        a.plot(zc, c['Qt'] / 1e3, 'o', ms=3.5, color=col, label=f"traced θ = {c['theta']:g}°")
        a.plot(zc, c['Qh'] / 1e3, '-', lw=1.1, color=col, label=f"blend s = {c['s']:.3f}")
    a.axvline(ax.g['z_act0'], color='#B5520A', lw=0.6, ls='--')
    a.set_xlabel('ring centre z [m]')
    a.set_ylabel('Q_block per ring [kW]')
    a.legend(fontsize=6.3, ncol=3)
    a.grid(alpha=0.3)
    fig.tight_layout(pad=0.4)
    return fig


def _fig_liner(R):
    import matplotlib.pyplot as plt
    tb = R['ax'].liner
    m = (tb.T_out > 1500) & (tb.T_out < 2700)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.5))
    a1.plot(tb.T_out[m], tb.q_out[m] / 1e3, color='k', label='q_out (into stick)')
    a1.plot(tb.T_out[m], tb.q_cell[m] / 1e3, color='#2E7D32', label='q_cell (active cells)')
    a1.set_xlabel('liner outer-face T_Lo [K]')
    a1.set_ylabel('kW per m² liner outer area')
    a1.legend(fontsize=7)
    a1.grid(alpha=0.3)
    a2.plot(tb.T_out[m], (tb.T_out - tb.T_in)[m], color='#555555')
    a2.set_xlabel('liner outer-face T_Lo [K]')
    a2.set_ylabel('T_Lo − T_Li [K]')
    a3 = a2.twinx()
    a3.plot(tb.T_out[m], (tb.q_cell / tb.q_out)[m], color='#2F5F9E', ls='--')
    a3.set_ylabel('q_cell / q_out', color='#2F5F9E')
    a2.grid(alpha=0.3)
    fig.tight_layout(pad=0.4)
    return fig


# --------------------------------------------------------------------------
# 4.2 - 4.4
# --------------------------------------------------------------------------
def sec_radiation(doc, ctx):
    from . import axial as AX, viewfactors as VF, correlations as CO, build as BU, tinless as TL, components as CP
    from . import properties as PR
    R = _model(ctx)
    cfg = R['cfg']
    rad = cfg.get('radiation', {}) or {}
    sh = cfg.get('shade', {}) or {}
    lbc = cfg.get('liner_bottom', {}) or {}
    rows = []                                                       # numerical-settings summary
    if not R['ok']:
        doc.h2('4.2 Radiation in the stick cavity')
        doc.p('The YAML does not select the axial vane model (radiation.axial: true, shade.model: vanes); '
              'sections 4.2-4.4 describe that configuration. See Appendix A for the alternatives.')
        return rows
    m, ax, a = R['m'], R['ax'], R['a']
    lay, me, tb, pd = ax.lay, m.mesh, ax.liner, ax.pad
    n, na, nl = ax.n, ax.n_act, ax.n_low
    S = 2 * lay['A']
    shorten = 0.5 * lay['t'] * np.tan(np.pi / 8) + 0.5 * lay['gap']
    n_rays = int(rad.get('vane_rays', 2 ** 14) or 2 ** 14)
    ps = m.powerstick
    A_cell = ax.A_cell_cm2
    Q_tgt = m.sim.nominal_cell_flux * A_cell
    R['Q_tgt'] = Q_tgt
    names = a.names
    A_ = a.A
    cnt = lambda pred: sum(1 for x in names if pred(x))
    import re
    is_face = lambda x: re.match(r'^[fd]\d+[pm]\d+$', x) is not None
    is_end = lambda x: re.match(r'^[fd]\d+e[pm]$', x) is not None
    opt_rays = None
    try:
        from . import optimize as O
        opt_rays = (O.DEFAULTS if hasattr(O, 'DEFAULTS') else {}).get('fidelity', {}).get('vane_rays')
    except Exception:
        pass
    if opt_rays is None:
        try:
            opt_rays = ctx['opt']['fidelity']['vane_rays']
        except Exception:
            opt_rays = None
    kir = a.kir
    eps_s = ax.anchors['open'].eps_s

    # ======================================================================
    doc.h2('4.2 Radiation in the stick cavity')
    doc.p('The block bore, the vanes, the liner and the liner-bottom pad form one closed radiation '
          'enclosure over the full block height H = ', f"{float(me.edges[-1]):g} m",
          '. The stick and liner, of length L = ', f"{m.L:g} m", ', hang top-aligned in it; the ',
          f"{nl}", ' block rings below them form the lower zone (geometry and mesh: section 3). The model is:')
    doc.bl([
        'Gray, diffuse and opaque inside the cavity (total σT⁴ radiosity method). The only spectral exchange is '
        'between the liner inner face and the stick (section 4.4).',
        'View factors are ray-traced once per vane angle in the real square geometry: square bore, square '
        'liner, 8 rectangular vanes. No equivalent cylinders are used in the cavity.',
        f'Vanes are solved at three fixed angles (0°, {ax.THETA_MID:g}°, 90°). Each angle gets a complete '
        'coupled Newton solve, and the shade position is a blend of these three solutions (section 4.3).',
        'Each liner ring passes the heat it absorbs to the stick through a pre-tabulated local chain: wall '
        'conduction, then spectral two-surface exchange (section 4.4). The pad under the liner feeds a '
        'quasi-steady liner-bottom model.',
        'The result is the heat leaving every block ring, which drives the block conduction update '
        '(section 4.1), together with its sensitivity to the wall temperatures.',
    ])
    doc.fig(_fig_flow(), 'Radiation solve and vane control in one discharge time step. Every box is described '
                         'in sections 4.2-4.4. The cell model is in section 4.5 and the block update in 4.1.',
            name='Figure 4.2.1', width=6.6)

    # ---------------------------------------------------------------- surfaces
    doc.h3('Enclosure surfaces')
    doc.p(f"Each anchor enclosure has N = {a.N} surfaces (", (_ref(CO, 'names = [f\'B{k}', ), {'mono': True}),
          '). Areas below are for the current YAML. The cavity side is S = ', f"{S:.4f} m",
          ' (half-side A = S/2). The block cylinder used by the conduction model has r_in = 2S/π = ',
          f"{m.r_in:.4f} m", ', which has the same perimeter, so a block ring has the same area in both models.')
    get = lambda nm: A_[a.idx[nm]] if nm in a.idx else float('nan')
    face_f = [A_[i] for i, x in enumerate(names) if is_face(x) and x[0] == 'f']
    face_d = [A_[i] for i, x in enumerate(names) if is_face(x) and x[0] == 'd']
    ends = [A_[i] for i, x in enumerate(names) if is_end(x)]
    srows = [
        ['B0..B' + str(n - 1), f"{n}", 'cavity walls x, y = ±A, one per block ring',
         f"{get('B0'):.4f} (ring 0)", 'gray, ε = gr_emis(T_w), T_w given'],
        ['L0..L' + str(na - 1), f"{na}", 'liner outer faces ±S_L/2, active rings',
         f"{get('L0'):.4f}", 'gray ε = gr_emis(T_Lo); T_Lo unknown; sink = LinerTable'],
        ['E', f"{cnt(lambda x: x == 'E')}", 'liner outer faces beside pad + plate',
         f"{get('E'):.4f}", 'reradiating (adiabatic)'],
        ['R', f"{cnt(lambda x: x == 'R')}", 'pad underside at z = H_low, facing down',
         f"{get('R'):.4f}", 'gray ε_R; T_R unknown; sink = LinerBottom'],
        ['floor_in / floor_out', '2', 'z = 0, inside / outside the closed-octagon footprint, minus vane '
         'footprints', f"{get('floor_in'):.4f} / {get('floor_out'):.4f}", 'reradiating'],
        ['ceil_in / ceil_out', '2', 'z = H, same split; liner footprint removed',
         f"{get('ceil_in'):.4f} / {get('ceil_out'):.4f}", 'reradiating'],
        ['f{k}p{b}, f{k}m{b}', f"{len(face_f)}", 'face vanes (4 lumped): ± faces, ring k, length bin b',
         f"{face_f[0]:.4f}" if face_f else '-', f'gray ε_s = {eps_s:g}; T unknown; p↔m conduction'],
        ['d{k}p{b}, d{k}m{b}', f"{len(face_d)}", 'corner vanes (4 lumped): ± faces',
         f"{face_d[0]:.4f}" if face_d else '-', 'as face vanes'],
        ['f/d{k}ep, em', f"{len(ends)}", 'vane end faces (thickness t)', f"{ends[0]:.4f}" if ends else '-',
         'reradiating (adiabatic)'],
    ]
    doc.table(['surface', 'count', 'geometry', 'area [m²]', 'radiative model'], srows,
              widths=[1.15, 0.45, 2.35, 1.05, 2.1], size=8)
    doc.fig(_fig_section(R), 'Section through the cavity on a vane-pivot plane, vanes closed. Left: full height '
                             'with the block rings, full-height vanes, liner rings and stick. Right: the liner '
                             'bottom (pad underside R, RGF pad, graphite plate, liner edge E). Only half of the '
                             'vanes cut this plane; the corner vanes lie on the diagonals.',
            name='Figure 4.2.2', width=6.6)
    doc.bl([
        'Symmetry lumping: one cavity wall, one liner face and one representative vane per class (face, corner) '
        'are traced, and their areas multiplied by 4. Hits on any of the four copies go to the same lumped '
        'surface. This needs only the 90° rotational symmetry, which holds because all 8 vanes rotate the same '
        'way about their own centres. Mirror symmetry is not assumed.',
        'Vane faces are split at every block-ring edge, including the lower zone, and into 2 bins along the '
        'vane length. The bin count is fixed in code and not exposed in the YAML.',
        'The closed-octagon footprint (|x|, |y| ≤ 1.5 l and |x| + |y| ≤ 2 l) only labels the floor and ceiling '
        'as "in" or "out". It does not move with the vanes.',
        'Under the liner, floor_in includes the liner footprint, because the liner stops above the floor. At '
        'the ceiling the liner footprint is removed and the real liner/ceiling gap is treated as closed.',
        'Floor and ceiling areas are Monte Carlo estimates (accepted fraction × (2A)²). For example floor_out '
        f"and ceil_out should be identical but differ by "
        f"{abs(get('floor_out') - get('ceil_out')) / get('floor_out'):.1e} (relative).",
        'Not in the enclosure: the liner interior, the stick, and the top of the liner channel. These are '
        'handled by LinerTable and LinerBottom (section 4.4).',
    ])
    rows += [['radiation', 'cavity surfaces per vane anchor', f"{a.N} ({n} block, {na} liner, "
              f"{len(face_f) + len(face_d)} vane faces, {len(ends)} vane ends, E, R, 4 floor/ceiling)",
              _ref(CO, 'def _vane_enclosure')],
             ['radiation', 'vane face bins along length', '2 (not in YAML)', _ref(AX, 'n_bins=2')]]

    # ---------------------------------------------------------------- view factors
    doc.h3('View factors: ray tracing in the square geometry')
    doc.p('For each emitting surface i, N_i rays are launched from uniformly distributed points with '
          'cosine-weighted (diffuse) directions. Each ray is traced to its first hit, and')
    doc.eq(r'F_{ij} = \frac{N_{ij}}{N_i},\qquad \mathbf{d} = \cos\theta\,\mathbf{n} + \sin\theta\,(\cos\phi\,'
           r'\mathbf{t}_1 + \sin\phi\,\mathbf{t}_2),\quad \sin^2\theta = u_1,\ \phi = 2\pi u_2',
           where=f"u1, u2 and the two position coordinates come from a 4-D scrambled Sobol sequence "
                 f"(scipy.stats.qmc, seed 2024 + surface index), {_ref(CO, 'def cos_dirs')}, "
                 f"{_ref(CO, 'def wall_emitter')}.")
    doc.bl([
        f"Rays per surface: 2^ceil(log2 n) with n = radiation.vane_rays = {n_rays}, the same for every surface "
        f"and every anchor. The optimizer's screening fidelity uses {opt_rays if opt_rays else '2^12'} "
        "(section 9).",
        'Intersections: the 4 cavity walls, 4 liner faces and 4 faces of each of the 8 vanes are infinite '
        'vertical planes, clipped to the segment (0 ≤ s ≤ 1) and to their z range. The floor (z = 0), the '
        'ceiling (z = H) and the pad plane (z = H_low, upward rays inside the liner footprint) are horizontal '
        'planes.',
        f"Tolerances: hits need path length > 1e-9 m. Emission points are offset 1e-10 m along the normal. A "
        f"ray with no hit raises an error ({_ref(CO, 'ray escaped the cavity')}).",
        'Floor and ceiling emitters: rejection sampling on the full square. The Sobol sample is doubled until '
        'at least n points are accepted, up to 64× n.',
        'Seeds are fixed, so the Monte Carlo noise is deterministic: the same in every run, never averaged '
        'out.',
    ])
    doc.p('The raw matrix satisfies neither reciprocity nor summation exactly. A symmetric Sinkhorn scaling ',
          (_ref(VF, obj=VF.enforce_reciprocity), {'mono': True}),
          ' restores both, while zero entries stay zero:')
    doc.eq(r'X = \frac{1}{2}\left(AF + (AF)^T\right),\qquad d_i \leftarrow d_i\sqrt{\frac{A_i}{d_i\sum_j X_{ij}d_j}}'
           r',\qquad F^{\prime}_{ij} = \frac{d_i X_{ij} d_j}{A_i}',
           where='Iterated until max |row sum / A − 1| < 1e-12 (at most 500 iterations). Afterwards '
                 'A_i F_ij = A_j F_ji holds exactly and every row sums to 1.')
    Fm = a.C[0]
    rs = float(np.abs(Fm.sum(1) - 1).max())
    X = a.A[:, None] * Fm
    rec = float(np.abs(X - X.T).max() / X.max())
    doc.p(f"For the current model: max row-sum error {rs:.1e} and max reciprocity error {rec:.1e} (relative). "
          f"The matrices are cached on disk ({rad.get('vf_cache_dir')!s}, pickle) under a SHA-1 key of the "
          "vane layout, ring edges, z_act0, H_low, angle, bin count and ray count ",
          (_ref(AX, obj=AX.VaneHybridRadiation._enclosure), {'mono': True}),
          '. A fresh trace of one anchor takes a few seconds.')
    doc.p('Accuracy evidence. The vane study reports the heat fraction at 2^15 rays within 0.014% of 2^18 rays '
          '(S = 1.2 m layout, H = 6 m, 5° and 30°; ', (_ref(CO, 'def vane_ray_convergence'), {'mono': True}),
          '). The YAML comment quotes ~0.02% for 2^14.')
    doc.review('The square/vane ray tracer has no analytic validation test. The cylinder tracer is tested '
               'against Howell C-93 (tests/test_viewfactors.py); the square tracer is not, e.g. a no-vane '
               'square duct against rectangular-duct formulas. Its ray-convergence check used the S = 1.2 m '
               'study layout, not the in-use layout. Small surfaces (vane ends, R, E) get the same ray count as '
               'large ones and are then rescaled by Sinkhorn. Confidence in the totals: high. In individual '
               f"small-surface rows: moderate. {_ref(CO, 'def _vane_enclosure')}.")
    rows += [['radiation', 'rays per surface (vane anchors)',
              f"{n_rays} (YAML radiation.vane_rays); {opt_rays or '2^12'} in optimizer screening",
              _ref(BU, "vane_rays")],
             ['radiation', 'ray sampler', 'scrambled Sobol 4-D, seed 2024 + surface index (fixed)',
              _ref(CO, 'qmc.Sobol(4')],
             ['radiation', 'ray hit / offset tolerances', 'path > 1e-9 m; emission offset 1e-10 m',
              _ref(CO, 'tf > 1e-9')],
             ['radiation', 'reciprocity / summation', 'symmetric Sinkhorn, tol 1e-12, max 500 its',
              _ref(VF, 'def enforce_reciprocity')],
             ['radiation', 'view-factor cache', f"{rad.get('vf_cache_dir')} (pickle per anchor, SHA-1 geometry key)",
              _ref(AX, '"vane-v1"')]]

    # ---------------------------------------------------------------- radiosity
    doc.h3('Radiosity formulation')
    doc.p('With radiosity J and irradiation per unit area G = F J, every surface satisfies')
    doc.eq(r'J_i = (1-\epsilon_i)\,G_i + \epsilon_i E_i,\qquad G_i = \sum_j F_{ij} J_j,\qquad E_i = \sigma T_i^4',
           where=f"Reradiating surfaces (floor, ceiling, E, vane ends) use J_i = G_i (zero net flux). "
                 f"{_ref(AX, '# --- radiosity rows')}.")
    doc.p('Block rings have a given temperature. Every other gray surface has an energy balance that closes '
          'the system:')
    doc.eq(r'\epsilon_L\left(G_L-\sigma T_{Lo}^4\right) = q_{out}(T_{Lo})\qquad'
           r'\epsilon_R\left(G_R-\sigma T_R^4\right) = \frac{Q_{pad}(T_R,\,T_{Li,0})}{A_R}',
           where='Left: liner ring. q_out is the LinerTable flux into the stick per unit liner outer area (section '
                 '4.4). Right: pad underside; Q_pad comes from LinerBottom (section 4.4).')
    doc.eq(r'\epsilon_s\left(G_o-\sigma T_o^4\right) = \frac{I(T_o)-I(T_i)}{t} = '
           r'-\epsilon_s\left(G_i-\sigma T_i^4\right),\qquad I(T) = \int_{T_0}^{T} k_{RGF}(T^{\prime})\,dT^{\prime}',
           where='Vane faces: p face (o) and m face (i) of the same vane, ring and bin, joined by exact 1-D '
                 'conduction through the thickness t (section 4.3).')
    doc.p(f"Unknowns: x = [J (N = {a.N}), E = σT⁴ of {a.n_sl} p-faces, {a.n_sl} m-faces, {na} liner rings"
          f"{', 1 pad underside' if a.pad is not None else ''}]. That is {a.N + a.nE} unknowns per anchor. E is "
          "used rather than T, with T = (E/σ)^¼, and is clamped at ≥ 1e-3 W/m² inside the residual.")
    doc.p('Outputs of an anchor solve (', (_ref(AX, obj=AX.AxialRadiation._finish), {'mono': True}), '):')
    doc.eq(r'Q_{B,k} = A_{B,k}\left(J_{B,k}-G_{B,k}\right),\qquad Q_{ps,k} = A_{L}\,q_{out}(T_{Lo,k}),\qquad '
           r'Q_{cell,k} = A_L\,q_{cell}(T_{Lo,k})',
           where='Q_B is the net power leaving block ring k (positive cools the block). The check '
                 'balance_err = |ΣQ_B − ΣQ_ps − Q_pad| / ΣQ_B is reported.')

    # ---------------------------------------------------------------- Newton
    doc.h3('Newton solution')
    doc.bl([
        'Residual: all radiosity rows plus the vane, liner and pad balances above. The Jacobian is analytic '
        'and dense, and each step is one np.linalg.solve. It includes dε/dT of the liner outer face (central '
        'difference ±0.5 K), the Kirchhoff k(T) terms, and dQ_pad/dT_R (implicit-function theorem with ±0.05 '
        'K differences). The pad\'s dependence on the bottom liner ring\'s T_Li is lagged: it has no Jacobian '
        'entry.',
        f"Convergence: max |r| < {a.tol:g} · max σT_w⁴, with at least one update. At most {a.max_iter} "
        "iterations. After the last iteration the result is accepted if max |r| < 1e3 × that tolerance.",
        'Damping: the step is scaled so that no emissive power falls below half its current value. In the '
        'porous-shade mode only, the blocked fraction moves at most 0.25 per step.',
        'Initial guess (first call only): J = mean σT_w⁴, E = 0.5 × mean σT_w⁴, vane outer faces 0.9 × mean. '
        'After that each anchor warm-starts from its own previous converged state.',
        'Fallback: if the warm start fails, restart from the initial guess. If that also fails, raise '
        'RuntimeError, which stops the run.',
        f"Cost (current model): {R['newton_per_call']:.1f} Newton iterations per anchor call on average, and "
        f"{R['t_solve'] * 1e3:.0f} ms for one controlled 3-anchor solve.",
    ])
    doc.eq(r'\lambda = \min\left(1,\ \min_{\Delta E_j<0}\ \frac{0.5\,E_j}{|\Delta E_j|}\right),\qquad '
           r'x \leftarrow x + \lambda\,\Delta x,\qquad \mathbf{K}\,\Delta x = -r',
           where=f"K: Jacobian. {_ref(AX, obj=AX.AxialRadiation._newton)}; initial guess "
                 f"{_ref(AX, obj=AX.AxialRadiation._guess)}.")
    rows += [['radiation', 'Newton tolerance / max iterations',
              f"max|r| < {a.tol:g}·max σT_w⁴; {a.max_iter} its; accept < 1e3·tol", _ref(AX, 'tol=1e-9, max_iter=40')],
             ['radiation', 'Newton damping', 'no E more than halved per step; Δf ≤ 0.25 (porous only)',
              _ref(AX, 'keep E > 0')],
             ['radiation', 'Newton initial guess', 'J = mean σT_w⁴, E = 0.5·mean (vane outer 0.9·mean); then warm start',
              _ref(AX, 'def _guess')],
             ['radiation', 'emissive-power clamp', 'E ≥ 1e-3 W/m²', _ref(AX, '1e-3)', nth=1)],
             ['radiation', 'dε/dT (liner outer face, block)', 'central difference ±0.5 K', _ref(AX, 'def _dgr_emis')]]

    # ---------------------------------------------------------------- wall coupling
    doc.h3('Block wall temperature and coupling to the block')
    doc.p('The wall temperature of ring k is T[k, 0], the innermost radial cell of the block conduction grid '
          '(section 4.1). The wall emissivity is ε_B = gr_emis(T_w) (section 7). The radiation result is applied '
          'as heat input to that cell. Because ring-to-ring radiation is very stiff (an effective axial k of '
          'order 1e3 W/m·K), the block update treats the wall implicitly using the sensitivity matrix')
    doc.eq(r'\frac{dQ_B}{dT_w} = A_B\left(\frac{dJ_B}{dT_w} - F_B\,\frac{dJ}{dT_w}\right),\quad '
           r'\frac{dx}{dT_w} = -\mathbf{K}^{-1}\frac{\partial r}{\partial T_w},\quad '
           r'\frac{\partial r_{B}}{\partial T_w} = -\left[\epsilon^{\prime}(E_B-G_B)+4\epsilon\sigma T_w^3\right]',
           where=f"Implicit-function theorem at the converged state, {_ref(AX, obj=AX.AxialRadiation._dQ_dTw)}. "
                 "The vane model blends the three anchor matrices in s. The IMEX update is in "
                 f"{_ref(CP, 'implicit = dQ_dTw is not None')} (section 4.1).")
    dr = (m.r_out - m.r_in) / m.storage.num_rad_segments
    doc.review(f"The emitting temperature is the centre of the first radial cell (dr = {dr * 1e3:.0f} mm, so the "
               f"node sits {dr * 500:.0f} mm inside the surface). There is no half-cell resistance to the surface. "
               "At a wall flux of 7-17 kW/m² and k = 20 W/m·K the real surface is about 10-30 K cooler, so "
               "radiation, vane position and cutoff timing are driven by a slightly hot temperature. Estimated "
               "effect: percent level on flux at fixed vane position. Confidence: moderate. "
               f"{_ref(AX, 'T_dist[:, 0], control=control)')}, {_ref(CP, 'hin_r[:, 0] = 0.0 if implicit')}.")
    doc.review('dQ/dT_w is evaluated at a fixed vane position. While the vanes are controlling, the true response '
               'at constant cell power is much smaller. Within one 600-1800 s step the extracted heat therefore '
               'follows the wall temperature as if the vanes were frozen; radiation is re-solved only at the next '
               'step. Expected: a small systematic under-extraction during control. A run with half the time step '
               'would show its size. ' + _ref(AX, 'dQ_dTw=self._pw') + '.')

    # ---------------------------------------------------------------- gray vs spectral
    doc.h3('Gray cavity, spectral stick')
    doc.p('All cavity surfaces are gray with total emissivities. Graphite block and liner faces use '
          'gr_emis(T) = 0.679 + 6e-5 T (no clip). Vanes and pad use a constant ε. Spectral detail matters only '
          'where the stick\'s band-selective surface is involved, and that is handled per wavelength node in '
          'section 4.4. The cavity sees full-spectrum σT⁴; the stick exchange integrates over the emissivity '
          f"table's {ps.emissivity_table[0][0][0]:g}-{ps.emissivity_table[-1][0][1]:g} µm range, which holds "
          f"{float(tb.B(1800.0).sum()) / (SIGMA * 1800.0 ** 4) * 100:.2f}% of σT⁴ at 1800 K and "
          f"{float(tb.B(2400.0).sum()) / (SIGMA * 2400.0 ** 4) * 100:.2f}% at 2400 K.")
    doc.review('The cavity is treated as gray. Graphite is close to gray, so this is reasonable. gr_emis is a '
               'linear fit with no upper clip. Confidence that the gray assumption is adequate: moderate-high. '
               'Effect: likely < 1-2% on block-to-liner heat.')

    # ---------------------------------------------------------------- vertical transport
    doc.h3('Vertical transport and cavity ends')
    w = R['win']
    sm = R['s_mid']
    lowsh = [float(so['Q_block'][:nl].sum() / so['Q_block'].sum()) for so in sm['anchors']]
    doc.p('Exchange between rings is part of the enclosure: every surface sees every other through F. The '
          'lower-zone block rings reach the liner only through the cavity: via the vane openings and gaps to '
          'the liner rings, to E and R, and via the floor. The diagnostic net upward radiation across the '
          'liner-bottom plane is')
    doc.eq(r'Q_{vert} = \sum_{i\,\in\,\mathrm{below}}\ \sum_{j\,\in\,\mathrm{above}} A_i F_{ij}\,(J_i - J_j)',
           where=f"Floor, R and E count as below; the ceiling counts as above. Blended in s like the other outputs "
                 f"({_ref(AX, obj=AX.AxialRadiation.vertical_transport)}). Diagnostic only.")
    doc.p(f"At a uniform wall of {0.5 * (R['T_min'] + R['T_max']):.0f} K the lower zone supplies "
          f"{lowsh[0] * 100:.0f}% of the block heat with the vanes closed, {lowsh[1] * 100:.0f}% at "
          f"{ax.THETA_MID:g}° and {lowsh[2] * 100:.0f}% open.")
    if R.get('comp'):
        q = R['comp'][0]['Qt'] / 1e3
        b0_txt = (f"With a hotter bottom and a reradiating floor, the bottom ring B0 radiates far more than its "
                  f"neighbours. In the {R['comp'][0]['theta']:g}° comparison case of section 4.3, B0 gives "
                  f"{q[0]:.1f} kW against {q[2:min(9, nl)].min():.1f}-{q[2:min(9, nl)].max():.1f} kW for rings "
                  f"2-{min(8, nl - 1)}. This follows from the adiabatic-floor assumption.")
    else:
        b0_txt = ('With a hotter bottom and a reradiating floor, the bottom ring B0 radiates more than its '
                  'neighbours (adiabatic-floor assumption).')
    doc.bl([
        'Floor and ceiling (in and out) are adiabatic reradiating surfaces, so there is no radiative loss '
        'through the cavity ends. Insulation losses through the roof and floor are applied to the block rings '
        'by the tank model (section 5).',
        'The ~1 cm gap between the ceiling and the liner is treated as closed. The YAML keys '
        'cavity_ends.emissivity and cavity_ends.ceiling_gap are read but not used.',
        b0_txt,
    ])
    doc.review('The cavity ends are adiabatic and the ceiling gap is closed. If the floor has finite conductance '
               'to the foundation, or the ceiling gap leaks, the bottom and top rings lose heat that the model '
               'keeps, and the floor-reflection boost to ring B0 is overstated. cavity_ends.emissivity and '
               f"ceiling_gap look configurable but have no effect ({_ref(BU, 'ce_cfg = ')}). Confidence: high that "
               'this is a simplification. Magnitude unknown; probably small next to the stick heat.')
    doc.review('After the stick is withdrawn at cutoff, Q_rad = 0 for every ring. The cavity then no longer '
               'redistributes heat axially (effective k ~1e3 W/m·K), and only block conduction acts. This may '
               'matter for the temperature profile carried into the next charge. Confidence: high that it is '
               f"omitted. Magnitude not quantified. {_ref(TL, 'Conduction-only phase')} (section 6).")

    # ======================================================================
    doc.h2('4.3 Shades (rotating vanes)')
    doc.p('Eight vanes (modelled as rigid graphite felt, RGF) surround the liner on an octagon. Each runs the full block height and '
          'turns about its own vertical centre line. At 0° they form a closed octagonal ring with small corner '
          'gaps; at 90° they are edge-on. Turning them throttles the radiation from the block wall to the '
          'liner. This is how the cell flux is held at the setpoint while the block cools.')
    doc.h3('Geometry')
    doc.eq(r'l = s_L + \sqrt{2}\,c,\qquad S_{min} = 4l + 2c,\qquad \delta = \frac{t}{2}\tan\frac{\pi}{8} + '
           r'\frac{g}{2}',
           where=f"s_L: liner side; c: shade.vane_clearance; t: shade.thickness; g: shade.corner_gap. Cavity side "
                 f"S = S_min sets r_in = 2S/π, and powerstick.area_ratio is ignored "
                 f"({_ref(CO, obj=CO.vane_layout_for_liner)}, {_ref(BU, 'S_cav = np.pi * r_in / 2.0')}).")
    doc.bl([
        'Face vanes (4, on the axes): pivot at radius 1.5 l, length l − 2δ.',
        'Corner vanes (4, on the diagonals): pivot at l√2, length l√2 − 2δ.',
        'The shortening δ leaves the gap g between the inner corners of adjacent closed vanes.',
        'All 8 vanes turn by the same angle θ about their centres (0 = tangential/closed, 90° = radial/edge-on).',
        'When open, a face vane spans radius l..2l and clears the wall by c + δ. A corner vane clears the liner '
        'corner by c + δ along the diagonal.',
    ])
    doc.kv([('liner side s_L / stick side', f"{2 * lay['liner_half']:.4f} m / {ps.width:.4f} m"),
            ('clearance c, corner gap g, thickness t',
             f"{float(sh.get('vane_clearance', 0.005)) * 1e3:.2f} mm, {lay['gap'] * 1e3:.3f} mm, {lay['t'] * 1e3:.1f} mm"),
            ('l (face) / l√2 (corner)', f"{lay['l']:.4f} m / {lay['l'] * np.sqrt(2):.4f} m"),
            ('shortening δ per end', f"{shorten * 1e3:.2f} mm"),
            ('effective lengths face / corner', f"{lay['l'] - 2 * shorten:.4f} m / {lay['l'] * np.sqrt(2) - 2 * shorten:.4f} m"),
            ('cavity side S (r_in = 2S/π)', f"{S:.4f} m ({m.r_in:.4f} m)")], w0=2.6)
    doc.fig(_fig_vanes(R), f"Plan view of the vanes at the three anchor angles and at 30° (not an anchor). "
                           f"Brown: cavity wall; grey: liner; blue: stick; dark/light green: face/corner vanes; "
                           f"dotted: the closed-octagon footprint. Drawn from {_ref(CO, 'def _vane_boxes')}.",
            name='Figure 4.3.1')
    try:
        bad = [th for th in (0, 1, 2, 5, 10, 20, 30, 45, 60, 75, 90) if R['C'].vane_collisions(lay, th)]
    except Exception:
        bad = ['?']
    doc.p('Collision check: ', (_ref(CO, 'def vane_collisions'), {'mono': True}),
          ' tests vane/wall, vane/liner and vane/vane overlap. Build does not call it. Run at '
          '0, 1, 2, 5, 10, 20, 30, 45, 60, 75 and 90° for this geometry it reports ',
          ('no collisions' if not bad else f"problems at {bad}"), '.')

    doc.h3('Ray-traced anchor positions')
    doc.p(f"Three complete enclosures are traced and solved: closed (0°), mid ({ax.THETA_MID:g}°) and open (90°). "
          f"Each is an independent AxialRadiation solve at fixed geometry, with its own warm start. In each, the "
          f"vanes are opaque bodies with real thickness, real gaps and real shadowing "
          f"({_ref(AX, obj=AX.VaneHybridRadiation.__init__)}). The mid anchor is at {ax.THETA_MID:g}° because "
          f"heat versus angle bends most there: most of the throttling range lies between 0° and about 15°.")

    doc.h3('Blending between anchors')
    doc.p('The vane position is represented by a blend parameter s ∈ [0, 2] (0 closed, 1 mid anchor, 2 open). '
          'For each block ring k and liner ring j:')
    doc.eq(r'Q(s) = Q_{0^\circ} + s\,(Q_{mid} - Q_{0^\circ})\ \ (s\leq 1),\qquad '
           r'Q(s) = Q_{mid} + (s-1)\,(Q_{90^\circ} - Q_{mid})\ \ (s>1)',
           where=f"Applied to Q_B,k, the liner-ring heat Q_ps,j, the pad outputs, dQ_B/dT_w and the vane "
                 f"face temperatures ({_ref(AX, obj=AX.VaneHybridRadiation._pw)}).")
    doc.bl([
        'Liner temperatures are not blended. Each ring\'s blended Q_ps,j/A_L is mapped back through the '
        'LinerTable to T_Lo, which gives q_cell and T_Li (section 4.4).',
        'Each anchor conserves energy to round-off, and the blend is linear, so the blended state does too. '
        f"Current model: balance_err = {sm['balance_err']:.1e}.",
        'f_shade is reported as f = 1 − s/2. It is a blend parameter, NOT a blocked area fraction: f = 0.5 is '
        f"the {ax.THETA_MID:g}° anchor. The time-step rule (section 6) uses f = 1 (closed), 0 < f < 1 and "
        'f = 0 (open).',
        'Vane angle (plots only): within each s segment, s is mapped to angle using the normalised heat-fraction '
        'curve between the two anchor angles. The curve comes from the vane study table VANE_GRID, at this '
        f"vane's R'' = t/k_RGF(1900 K) = {float(R['C'].vane_R_cond(lay['t'])):.3g} m²K/W and H/L = "
        f"{float(me.edges[-1]) / m.L:.2f}.",
    ])
    doc.review('Shade position is a piecewise-linear blend of three ray-traced angles, not a trace at the actual '
               'angle. Totals are exact by construction. The axial split of block cooling is approximate: see '
               'the measured comparison below, which has per-ring errors ≤ ~1% of total heat and the lower-zone '
               'share low by ~0.01. Confidence: high for totals, moderate for the axial distribution. The '
               f"effect on the block temperature field over a cycle is not checked. {_ref(AX, 'def _pw')}.")
    doc.review('f_shade (= 1 − s/2) is easy to misread as a closed fraction; f = 0.5 means the 10° anchor. The '
               'vanes have no actuator dynamics or rate limit, so s can jump freely between steps. Low risk, but '
               f"plots of f_shade need this caveat. {_ref(AX, 'f=1.0 - s / 2.0')}.")

    doc.h3('Control law')
    doc.nl([
        'Solve the three anchors at the current wall temperatures (no control inside the anchors).',
        f"Target: Q_tgt = q_nom · A_cells = {m.sim.nominal_cell_flux:g} W/cm² × {A_cell:,.0f} cm² = "
        f"{Q_tgt / 1e3:.2f} kW, summed over all rings. There is no per-ring constraint. A_cells = "
        "n_act · A_step · (1 − f_bare) · f_active.",
        'If ΣQ_cell(open) ≤ Q_tgt: s = 2 (open, under-delivers). If ΣQ_cell(closed) ≥ Q_tgt: s = 0 (closed, '
        'over-delivers). No error is raised in either case.',
        'Otherwise: brentq on s ∈ [0, 2] (xtol 1e-9) for Σ_k A_L q_cell(T_Lo,k(s)) = Q_tgt. It includes the '
        'liner-bottom cell share only if liner_bottom.include_flux_to_cells_in_power is true.',
        'Discharge runs always use control (shade on and avg_flux set). Without control, s defaults to 2.',
    ])
    doc.eq(r'\sum_{k=1}^{n_{act}} A_L\,q_{cell}\left(T_{Lo,k}(s)\right) = q_{nom}\,A_{cells},\qquad '
           r'T_{Lo,k}(s) = q_{out}^{-1}\left(\frac{Q_{ps,k}(s)}{A_L}\right)',
           where=f"{_ref(AX, obj=AX.VaneHybridRadiation.solve)}")
    doc.fig(_fig_window(R), 'Control window at a uniform wall temperature. Left: total cell power at the three '
                            'anchors, the setpoint, and the controlled result. Right: the blend parameter s and '
                            'the estimated vane angle.', name='Figure 4.3.2')
    wt = [[f"{r[0]:.0f}", f"{r[1] / 1e3:.1f}", f"{r[2] / 1e3:.1f}", f"{r[3] / 1e3:.1f}", f"{r[4] / 1e3:.1f}",
           f"{r[5]:.3f}", f"{r[6]:.1f}"] for r in w[::2]]
    doc.table(['T_wall [K]', 'ΣQ_cell 0° [kW]', f"{ax.THETA_MID:g}° [kW]", '90° [kW]', 'controlled [kW]', 's',
               'angle est. [°]'], wt, widths=[0.85, 1.1, 0.9, 0.9, 1.15, 0.8, 1.0], size=8)
    hi = w[w[:, 1] > Q_tgt]
    if len(hi):
        doc.review(f"The closed vanes cannot hold the setpoint at high wall temperature. At a uniform "
                   f"{hi[0, 0]:.0f} K the closed limit already gives {hi[0, 1] / Q_tgt:.2f}× the target, and at "
                   f"{w[-1, 0]:.0f} K {w[-1, 1] / Q_tgt:.2f}×. The model then sits at s = 0 and over-delivers early "
                   f"in discharge (T_max = {R['T_max']:.0f} K). This depends on vane thickness and conductivity "
                   "and on the corner gaps. Check that it is intended, or whether thicker or tighter vanes are "
                   "needed.")

    doc.h3('Vane conduction and surface')
    doc.p(f"Each conducting pair is the p and m face of one vane class, ring and length bin "
          f"({a.n_sl} pairs). The flux through the thickness uses the exact Kirchhoff integral of the RGF "
          "conductivity k_RGF = ins_tcx(T) (section 7):")
    doc.eq(r"q_c = \frac{1}{t}\int_{T_i}^{T_o} k_{RGF}(T)\,dT = \frac{I(T_o)-I(T_i)}{t}",
           where=f"I(T) is tabulated by cumulative trapezoid from {kir.T[0]:.0f} to {kir.T[-1]:.0f} K in "
                 f"{kir.T[1] - kir.T[0]:g} K steps and linearly interpolated, clamped outside "
                 f"({_ref(AX, obj=AX.Kirchhoff)}). This is the n → ∞ limit of a layered model, so "
                 "shade.num_conduction_layers is not used.")
    doc.bl([
        f"Vane surfaces are gray with ε_s = shade.surface_emissivity = {eps_s:g} on both faces. End faces "
        "are reradiating.",
        'Not modelled: conduction along the vane (between length bins or rings), conduction to pivots or '
        'supports, and vane heat capacity (quasi-steady).',
    ])
    doc.review('Vane conduction uses RGF felt in argon (ins_tcx, 0.09-1.235 W/m·K, constant above 2500 °C). '
               'Confirm the vane material and its k(T). With solid graphite vanes (k ~ 30-40 W/m·K) the closed '
               'vanes would pass much more heat and the control window would shrink. The model is 1-D through '
               'the thickness with adiabatic ends and ε_s constant. The numerics are exact; the material inputs '
               f"dominate. {_ref(AX, 'self.kir = Kirchhoff()')}.")

    # blend accuracy
    doc.h3('Accuracy of the blend (measured for this document)')
    if R.get('comp'):
        Tb, Tt = R['comp_Tw']
        doc.p(f"Wall linear from {Tb:.0f} K (bottom) to {Tt:.0f} K (top). Each angle below was ray-traced "
              f"directly ({n_rays} rays, same solver) and compared with the blend at the s that gives the same "
              "total cell power:")
        ct = [[f"{c['theta']:g}", f"{c['s']:.4f}", f"{c['ang']:.2f}", f"{c['sum_t'] / 1e3:.2f} / {c['sum_h'] / 1e3:.2f}",
               f"{c['low_t']:.4f} / {c['low_h']:.4f}", f"{c['err'] * 100:.2f}", f"{c['dTLi']:.1f}"]
              for c in R['comp']]
        doc.table(['traced θ [°]', 's (blend)', 'angle est. [°]', 'ΣQ_B traced / blend [kW]',
                   'lower-zone share traced / blend', 'max |ΔQ_ring| / ΣQ_B [%]', 'max |ΔT_Li| [K]'], ct,
                  widths=[0.75, 0.75, 0.8, 1.35, 1.35, 1.15, 0.85], size=8)
        doc.fig(_fig_comp(R), 'Heat leaving each block ring: direct ray trace (markers) against the three-anchor '
                              'blend (lines). The dashed line is the liner-bottom plane.', name='Figure 4.3.3')
    else:
        doc.p('(Comparison not available in this build.)')

    # ======================================================================
    doc.h2('4.4 Liner and liner bottom')
    doc.h3('LinerTable: liner ring → stick')
    doc.p('Each liner ring is a quasi-steady chain. Heat absorbed on the outer face is conducted through the '
          'wall and radiated from the inner face to the stick segment opposite. The chain is tabulated once '
          'against the inner-face temperature T_Li and then read off by the solver (',
          (_ref(AX, obj=AX.LinerTable), {'mono': True}), ').')
    doc.p(('Wavelength quadrature. ', {'bold': True}),
          f"Gauss-Legendre in ln λ with {getattr(m.sim, 'n_gauss_per_band', 10)} nodes in each emissivity-table "
          f"band. With {len(ps.emissivity_table)} bands that gives {tb.wl.size} nodes "
          f"({tb.wl.min():.3f}-{tb.wl.max():.2f} µm). Nodes never straddle a band edge, so the piecewise-constant "
          f"stick emissivity is integrated exactly ({_ref(TL, 'def _spectral_grid')}). The code docstring quotes "
          "~1e-6 relative error against a 20,000-bin reference.")
    doc.eq(r'\int_{\lambda_a}^{\lambda_b} f\,d\lambda \approx \sum_i w_i\,\lambda_i\,\frac{\ln\lambda_b-\ln\lambda_a}{2}'
           r'\,f(\lambda_i),\qquad B_i(T) = \frac{2\pi h c^2}{\lambda_i^5\left(e^{hc/\lambda_i k_B T}-1\right)}',
           where='w_i, x_i are the Gauss-Legendre weights and nodes; λ_i = exp(mid + half-width · x_i) in ln λ.')
    doc.p(('Stick as one composite surface. ', {'bold': True}),
          'The bare, active-cell and inactive-cell areas are lumped per node into an area-weighted emissivity. '
          'The active cells\' share of the absorbed power is:')
    doc.eq(r'\bar{\epsilon} = f_b\epsilon_b + (1-f_b)f_a\epsilon_a + (1-f_b)(1-f_a)\epsilon_{ia},\qquad '
           r'\phi_{act} = \frac{(1-f_b)f_a\epsilon_a}{\bar{\epsilon}}',
           where=f"f_b = f_bare = {ps.f_bare:g}, f_a = f_active = {ps.f_active:g} (section 4.5). Stick emission is "
                 "neglected (cooled, ~323 K).")
    bt = []
    for (rng, e_a, e_b, e_i) in ps.emissivity_table:
        msk = (tb.wl >= rng[0]) & (tb.wl < rng[1])
        bt.append([f"{rng[0]:g}-{rng[1]:g}", f"{e_a:g} / {e_b:g} / {e_i:g}", f"{int(msk.sum())}",
                   f"{tb.eps_cells[msk][0]:.4f}", f"{tb.act[msk][0]:.4f}"])
    doc.table(['band [µm]', 'ε active / bare / inactive', 'nodes', 'ε̄ (stick)', 'φ_act'], bt,
              widths=[1.2, 1.9, 0.8, 1.4, 1.4], size=8.5)
    doc.p(('Two-surface exchange, per node. ', {'bold': True}),
          'Long enclosure with F(stick → liner) = 1, gray diffuse per node:')
    doc.eq(r'q_{ps} = \sum_i \frac{B_i(T_{Li})\,\Delta\lambda_i}{\frac{r_{ps}}{r_l}\,\frac{1-\epsilon_{in}}{\epsilon_{in}} + 1 + '
           r'\frac{1-\bar{\epsilon}_i}{\bar{\epsilon}_i}},\qquad q_{cell} = \sum_i \phi_{act,i}\,(\ldots)_i,\qquad '
           r'q_{out} = \frac{4 w_{ps}}{2\pi r_l}\,q_{ps}',
           where=f"q_ps is per unit stick area and q_out per unit liner OUTER area. r_ps = 2w_ps/π, r_l = 2s_L/π "
                 f"(equal-perimeter radii), r_ps/r_l = {tb.r_ratio:.4f}. ε_in = gr_emis(T_Li) for a graphite "
                 f"liner, floored at 1e-9.")
    doc.p(('Wall conduction. ', {'bold': True}),
          f"Planar, thickness t_L = {m.liner_thickness * 1e3:g} mm, k = liner_k(material, T_Lo) evaluated at "
          "the outer face (gr_tcx for graphite). It is solved by fixed point (≤ 50 iterations, 1e-9 K), and "
          "T_Lo(T_Li) is checked to be monotone:")
    doc.eq(r'T_{Lo} = T_{Li} + \frac{q_{out}\,t_L}{k(T_{Lo})}')
    doc.p(f"Table: T_Li = {tb.T_in[0]:.0f}, {tb.T_in[0] + (tb.T_in[1] - tb.T_in[0]):.0f}, …, {tb.T_in[-1]:.0f} K "
          f"({tb.T_in.size} points). The solver interpolates q_out, q_cell, their derivatives (np.gradient) and "
          "T_Li linearly in T_Lo, clamped at the ends. The vane blend uses the inverse, T_Lo(q_out).")
    doc.fig(_fig_liner(R), 'LinerTable for the current liner and stick. Left: flux into the stick and into the '
                           'active cells per unit liner outer area. Right: temperature drop across the wall and '
                           'the cell share.', name='Figure 4.4.1')
    doc.p('Cell power: Q_cell,k = A_L q_cell(T_Lo,k). It counts all absorbed wavelengths, including '
          'sub-band-gap absorption at ε_active of the long-wave band. Electrical power uses '
          'panel_efficiency(T_Li,k, flux_k) (section 4.5).')
    doc.review('The liner-to-stick exchange is local per ring: no axial crosstalk along the channel, no top-end '
               'model (the stick passes through the ceiling and the liner top is implicitly closed), and no '
               'conduction along the liner. The equal-perimeter radius ratio is exact for long square-in-square '
               'channels. r_l uses the liner OUTER side (r_ps/r_l = 0.52 against 0.54 for the inner face), a <1% '
               f"effect. Confidence: moderate. Effect: probably a few % near the ends. {_ref(AX, 'r_l = 2 * ps.liner_width')}.")
    doc.review('"Q_cell" is total absorbed power in the active-cell area, including long-wave absorption at '
               'ε_active = 0.015. It drives both the setpoint control and panel_efficiency. Check that '
               'panel_efficiency expects total absorbed flux rather than above-gap flux. The stick is lumped into '
               'one surface with area-weighted ε under uniform irradiation. '
               f"{_ref(AX, 'act[m] = (1 - f_bare)')}.")
    doc.review('Liner conduction is a planar slab with k at the outer-face T; the liner has no heat capacity. '
               'gr_tcx tabulates k against 20-2400 and is called with T in K; if the source data are in °C, k is '
               '~273 K off. The effect on the 5 mm liner ΔT is a few K. Low priority. '
               f"{_ref(PR, 'def gr_tcx')}.")

    # ---------------------------------------------------------------- liner bottom
    doc.h3('Liner bottom: RGF pad, graphite plate and bottom cap')
    if pd is None:
        doc.p('liner_bottom.enabled is false: the pad underside R is reradiating (adiabatic liner bottom).')
    else:
        doc.p('Below the liner, heat passes up through the pad and plate and is radiated from the plate\'s top face '
              'into the liner channel. The model is quasi-steady (', (_ref(AX, obj=AX.LinerBottom), {'mono': True}),
              '):')
        doc.p(('R (pad underside, cavity, ε_R) → RGF pad (Kirchhoff, t_rgf) → graphite plate (k = gr_tcx(T_LB), '
               't_lb) → LB (plate top face) → stick bottom cap, stick sides, liner walls', {'mono': True}))
        doc.kv([('t_rgf / t_lb (= liner thickness)', f"{pd.t_rgf * 1e3:g} mm / {pd.t_lb * 1e3:g} mm"),
                ('ε_R (pad underside)', f"{pd.eps_R:g} (liner_bottom.rgf_outer_emissivity, null → shade ε)"),
                ('T_cap (stick bottom cap), ε_cap', f"{pd.T_cap:g} K, ε_bare per band"),
                ('areas A_R / A_LB / A_cap [m²]', f"{pd.A_R:.4f} / {pd.A_LB:.4f} / {pd.A_cap:.4f} (square footprints)"),
                ('LB view factors: cap / stick sides / liner walls',
                 f"{pd.F[0]:.3f} / {pd.F[1]:.3f} / {pd.F[2]:.3f}"),
                ('include_flux_to_cells_in_power', str(pd.include_cells))], w0=2.8)
        doc.eq(r'F_1 = \frac{A_{cap}}{A_{LB}},\quad F_2 = (1-F_1)\frac{2\pi r_1 L\cdot\frac{1}{2}\left(1-F_{12}\right)}'
               r'{\pi(r_2^2-r_1^2)},\quad F_3 = 1-F_1-F_2',
               where='r_1 = 2 s_ps/π and r_2 = 2 s_in/π are equal-perimeter radii of the stick and the liner inner '
                     'face. F_12 (stick side → liner wall) is from Howell C-93 finite coaxial cylinders over the '
                     'liner length L, using reciprocity.')
        doc.p('Per node, with the cap reflecting LB\'s radiosity back, the stick sides at radiosity '
              'J_c = (1 − ε̄)/ε̄ · q_ps(T_Li,0) (reflected liner radiation), and the liner walls black at the bottom '
              'ring\'s T_Li,0:')
        doc.eq(r'J = \frac{\epsilon E + (1-\epsilon)\left(F_1\epsilon_c E_{cap} + F_2 J_c + F_3 J_w\right)}'
               r'{1-(1-\epsilon)F_1(1-\epsilon_c)},\qquad Q_{LB} = A_{LB}\sum_i (J_i - G_i)')
        doc.eq(r'I(T_{mid}) + \frac{Q\,t_{rgf}}{A_R} - I(T_R) = 0,\qquad T_{mid} = T_{LB} + \frac{Q\,t_{lb}}{A_R\,k(T_{LB})},'
               r'\qquad Q = Q_{LB}(T_{LB})',
               where='Solved for T_LB by brentq (xtol 1e-6 K) in a ±40 K bracket around the previous value, '
                     'else [250 K, max(T_R, T_Li) + 100 K]. dQ/dT_R = (dQ/dT_LB)·k_RGF(T_R)/res\'(T_LB) for the '
                     'Newton Jacobian (central differences ±0.05 K).')
        sm_ = R['s_mid']
        doc.p('Bookkeeping: Q_pad goes into the energy balance (Q_sink = ΣQ_ps + Q_pad). Q_cap = A_cap Σ ε_c J is '
              'the cap\'s absorption. The "net extra heat to the stick sides" is Q_pad − Q_cap, because liner-wall '
              'radiation absorbed by LB is already credited to the stick by the local table. It is added to the '
              f"bottom ring's Q_ps / Q_cell only if include_flux_to_cells_in_power is true. At a uniform "
              f"{0.5 * (R['T_min'] + R['T_max']):.0f} K wall: Q_pad = {sm_.get('Q_pad', float('nan')):.0f} W, "
              f"Q_cap = {sm_.get('Q_cap', float('nan')):.0f} W, T_LB = {sm_.get('T_LB', float('nan')):.0f} K, "
              f"T_R = {sm_.get('T_R', float('nan')):.0f} K.")
        doc.review('Liner-bottom simplifications: the liner walls seen from LB are black at the bottom ring\'s T_Li, '
                   'so the whole channel is lumped. The annulus split uses circular-annulus reciprocity. The pad '
                   'term is lagged in the Newton Jacobian. With include_flux_to_cells_in_power false, Q_pad is '
                   'taken from the block but appears in neither Q_ps, Q_waste nor P: an untracked sink of ~0.1 '
                   'kW, about 0.1% of the block heat. Low priority. '
                   f"{_ref(AX, obj=AX.LinerBottom.outputs)}.")

    # ---------------------------------------------------------------- constants table
    rows += [['shades', 'anchor angles', f"0°, {ax.THETA_MID:g}°, 90°; piecewise-linear blend in s ∈ [0, 2]",
              _ref(AX, 'THETA_MID = ')],
             ['shades', 'setpoint search', 'brentq on s, xtol 1e-9; clamp s = 0 / 2 if infeasible',
              _ref(AX, 'brentq(lambda x: qcell_tot(x) - Q_tgt')],
             ['shades', 'vane conduction', f"Kirchhoff I(T) of ins_tcx, {kir.T[0]:.0f}-{kir.T[-1]:.0f} K, step "
              f"{kir.T[1] - kir.T[0]:g} K", _ref(AX, 'def __init__(self, k_fn=ins_tcx')],
             ['shades', 'vane surface emissivity', f"{eps_s:g} (shade.surface_emissivity), constant",
              _ref(AX, 'self.eps_s = ')],
             ['liner', 'wavelength nodes', f"{tb.wl.size} ({getattr(m.sim, 'n_gauss_per_band', 10)} Gauss-Legendre "
              f"in ln λ per band)", _ref(TL, "n_gauss_per_band")],
             ['liner', 'LinerTable grid', f"T_Li {tb.T_in[0]:.0f}-{tb.T_in[-1]:.0f} K, step "
              f"{tb.T_in[1] - tb.T_in[0]:g} K ({tb.T_in.size} pts)", _ref(AX, 'np.arange(350.0, 3600.0, 2.0)')],
             ['liner', 'liner wall fixed point', '≤ 50 its, 1e-9 K', _ref(AX, 'for _ in range(50)')],
             ['liner', 'emissivity floor', '1e-9', _ref(AX, 'np.maximum(eps_in, 1e-9)')],
             ['liner bottom', 'T_LB solve', 'brentq xtol 1e-6 K; ±40 K warm bracket, else [250, max+100] K',
              _ref(AX, 'prev - 40.0')],
             ['liner bottom', 'pad sensitivity', 'central difference ±0.05 K', _ref(AX, 'def Q_and_dQ')]]
    doc.h3('Fixed internal values (sections 4.2-4.4)')
    doc.table(['area', 'setting', 'value', 'where'], [list(r) for r in rows], widths=[0.85, 1.9, 2.95, 1.4], size=7.5)

    # ---------------------------------------------------------------- remaining review items
    doc.review('Stale code and comments that may confuse readers. In correlations.py, the ψ machinery '
               '(VANE_PSI_GRID = None) claims to be "used by the vane hybrid" but is not. The YAML comments '
               '"f_shade is found by bisection", shade.num_conduction_layers and the dt_long note do not apply '
               'to the vane model. Q_through_openings is meaningless for vane surface names. '
               f"{_ref(CO, 'VANE_PSI_GRID = None')}.")
    print(f"[docs secB] model probes {R.get('t_total', 0):.0f} s")
    return rows


# --------------------------------------------------------------------------
# Appendix A
# --------------------------------------------------------------------------
def alternatives(doc, ctx):
    from . import tinless as TL, viewfactors as VF, axial as AX
    doc.h2('Radiation, shade and liner options')
    doc.h3('Local per-segment radiation (radiation.axial: false)')
    doc.p('Each block ring exchanges only with the liner segment opposite. The model uses the infinite '
          'coaxial-cylinder two-surface formula with equal-perimeter radii (view factor r_liner/r_in), '
          'spectrally on the same 20 nodes, scaled by the open fraction 1 − f. The shade adds a gray series path '
          'over area f·A_shade: block → shade outer face, one-layer RGF conduction at the mean shade T, shade '
          'inner face → liner. That path is solved for Q by brentq (xtol 0.1 W). The liner balance is solved for '
          'T_Lo by brentq (xtol 1 K). f is found by brentq on total cell flux (xtol 1e-4), clamped to '
          '[f_min, f_max]. There is no axial exchange, and it requires block_height = L '
          f"({_ref(TL, obj=TL.TinlessSimulation.spectral_net_radiation)}).")
    doc.h3('Porous-sheet shade (shade.model: porous)')
    doc.p('Equivalent coaxial cylinders (block r_in, shade r_sh = 2·shade_side_len/π, liner). The shade is a thin '
          'porous sheet with straight-through transmission τ = 1 − f: a ray crossing it hits a slat with '
          'probability f. It is traced once (Sobol, 2^16 rays), giving F(τ) = C0 + τC1 + τ²C2 + τ³C3. Each '
          'τ-power is symmetrised and slat rows divided by (1 − τ), so reciprocity and summation hold for every '
          'τ. Slat area is f·A_ring. Cell-flux control puts f into the Newton system as an extra unknown, '
          'tries the previous bound first, and falls back to brentq on f (xtol 1e-6). Here dQ/dT_w includes the '
          f"control response ({_ref(VF, obj=VF.porous_shade_view_factors)}, "
          f"{_ref(AX, obj=AX.AxialRadiation.solve)}).")
    doc.h3('Analytic vs Monte Carlo view factors (radiation.vf_method)')
    doc.p('This applies only to the unshaded cavity; shaded cavities are always ray-traced. "analytic" uses the '
          'Howell C-93 finite coaxial-cylinder relations, with ring-to-ring exchange by superposition, '
          'X = ½[G(|b−c|) + G(|a−d|) − G(|a−c|) − G(|b−d|)], ring-to-end by closure and floor-to-ceiling by row '
          'closure. It is exact only when both cylinders span the full height. With a lower zone it falls back to '
          'Monte Carlo (Sobol, radiation.vf_rays = 65,536, Sinkhorn); the stated accuracy is ~1% on F > 0.05. '
          '"montecarlo" forces Monte Carlo everywhere as a cross-check. Results are cached as .npz '
          f"({_ref(VF, obj=VF.cavity_view_factors)}).")
    doc.h3('Liner materials (liner_material)')
    doc.p('tac: inner face spectral ε = clip(−0.0351 + 8.15e-5 T + (0.3008 − 4.46e-5 T)/λ, 0, 1), outer face '
          'graphite, k = gr_tcx. zrc: inner ε = clip(0.70 − 1e-4 T + 0.02(λ − 0.665), 0, 1), outer graphite. '
          'grafoil: gray ε = 0.5 on both faces, k = 4 W/m·K, intended for 0.5 mm walls. graphite (in use): gray '
          f"gr_emis on both faces, k = gr_tcx ({_ref(AX, obj=AX.LinerTable.eps_in)}; properties.LINER_GRAY).")
    doc.h3('Tungsten foils')
    doc.p('powerstick.tungsten_foil (W foil between a TaC liner and the stick, spectral w_emis) and '
          'shade.outer_foil (W foil on the shade outer face, total emissivity, brentq on foil T) exist only in the '
          'local model. The axial model raises NotImplementedError for both '
          f"({_ref(AX, 'tungsten_foil not supported')}, {_ref(AX, 'outer_foil not supported')}).")
    doc.h3('Vane heat-fraction study')
    doc.p('This is an offline ray-traced study: S = 1.2 m; L = 3 m; H = 3, 6 and 10 m; 5 vane thicknesses '
          '(3-50.8 mm); 13 angles; 2^15 rays. It is stored as VANE_GRID (PCHIP in θ, linear in ln R\'\' and H/L) '
          'plus 9-coefficient fits. In the current model it only converts s to an approximate vane angle for '
          'plots; heat flows come from the three traced anchors.')
