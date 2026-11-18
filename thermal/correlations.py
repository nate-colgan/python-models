import numpy as np
import scipy as sp


def CircPipeFF(Re, D_h, rough):
    #return circular pipe friction factor (turbulent: Zigrang and Sylvester)
    if (Re < 2300):
        return 64 / Re
    else:
        return (-2 * np.log10(2 * rough / (7.54 * D_h) - 5.02 / Re * np.log10(2*rough / (7.54 * D_h) + 13 / Re))) ** (-2)

def RectPipeFF(Re, H, W, rough):
    #return recatngular pipe friction factor from Nellis and Klein, eq 5-60 and 5-65
    AR = H / W
    D_h = 2 * H * W / (H + W)
    if (Re < 2300):
        return 96 / Re * (1 - 1.3553 * AR + 1.9467 * AR ** 2 - 1.7012 * AR ** 3 + 0.9564 * AR ** 4 - 0.2537 * AR ** 5)
    else:
        return (-2 * np.log10(2 * rough / (7.54 * D_h) - 5.02 / Re * np.log10(2 * rough / (7.54 * D_h) + 13 / Re))) ** (-2)

def CircPipeNu(Re, ff, Pr):
    #return circular pipe Nusselt number from Nellis and Klein eq 5-76 and Taler 2018
    if (Re < 2300):
        return 4.36
    else:
        return 5.31 + 0.221 * (Re * Pr) ** 0.8174

def RectPipeNu(Re, H, W, ff, Pr):
    #return rectangular pipe Nusselt number
    AR = H / W
    if (Re < 3000):
        return 8.235 * (1 - 2.042 * AR + 3.085 * AR ** 2 -2.477 * AR ** 3 + 1.058 * AR ** 4 - 0.186 * AR ** 5)
    else:
        return 5.31 + 0.221 * (Re * Pr) ** 0.8174

def RprimeCondCyl(D_in, D_out, k_wall):
    #def cylindrical wall conduction thermal resistance [K/Wm]
    return np.log(D_out / D_in) / (2 * np.pi * k_wall)

def RCondWall(L, A_c, k_wall):
    #define planar conduction thermal resistance [K/W]
    return L / (k_wall * A_c)

def RprimeConvCyl(Nu, k_fluid, D):
    #define convection thermal resistance for a fluid in a pipe [K/Wm]
    h_bar = Nu * k_fluid / D
    per = np.pi * D
    return 1 / (h_bar * per)

def RConvWall(Nu, L, A_c, k_fluid):
    #def convection thermal resistance on a planar wall [K/W]
    #L is the characteristic length, hydraulic diameter for internal flows
    h_bar = Nu * k_fluid / L
    return 1 / (h_bar * A_c)

def Rrad(T_sur, T_s, A_s, VF, emis):
    #def exact radiation thermal resistance [K/W]
    return 1 / (A_s * emis * (T_s ** 2 + T_sur ** 2) * (T_sur + T_s) * sp.constants.sigma)

def Rrad_2surfaces(T_1, T_2, A_1, A_2, emis_1, emis_2, VF_12):
    #def radiation thermal resistance between 2 grey bodies [K/W]
    R_rad = ((1 - emis_1) / (A_1 * emis_1) + 1 / (A_1 * VF_12) + (1 - emis_2) /
             (A_2 * emis_2)) / ((T_1 ** 2 + T_2 ** 2) * (T_1 + T_2) * sp.constants.sigma)
    return R_rad


# =============================================================================
# Rotating-vane shade: heat fraction vs vane angle and block/stick height ratio
# =============================================================================
# Ray-traced study in the real square geometry (square cavity, square liner,
# octagonal ring of 8 centre-pivoted vanes). Produces the heat fraction
#     phi(theta, H/L) = heat into the stick with vanes at angle theta
#                       / heat into the stick with no vanes
# at a fixed block-wall temperature distribution, and fits a correlation.
#
# Run the whole study:   python -m thermal.correlations          (~15-20 min)
# Ray-count check:       python -m thermal.correlations --convergence
# Use the fit:           vane_heat_fraction(theta_deg, H_over_L)
#
# Geometry (all from the cavity side S, held fixed; "max size" layout):
#   face vanes   length l,    pivot at 1.5 l on the axes
#   corner vanes length l*√2, pivot at l*√2 on the diagonals
#   closed (theta = 0): the vanes form an octagon (square 3l with l-corners cut)
#   open (theta = 90): edge-on; a face vane spans l..2l radially, a corner vane
#   l√2/2..3l√2/2 along the diagonal
#   max vane size  : open face vane reaches the cavity wall  -> 2l = S/2 - clearance
#   max liner size : open corner vane reaches the liner corner -> liner half = l/2 - clearance/√2
#   vane ends shortened so the inner corners of adjacent closed vanes are `gap` apart
# Physics: gray diffuse cavity surfaces (cavity walls ε = gr_emis(T), vanes ε_s,
# floor/ceiling/pad reradiating); vane faces joined by exact RGF conduction
# through the thickness; liner -> stick two-band and local per segment (the
# same solver as the tinless axial model, thermal/axial.py).

VANE_DEFAULTS = dict(
    S=1.2,                 # cavity side [m] (fixed for the study)
    clearance=0.005,       # minimum clearance vane/wall and vane/liner [m]
    t_vane=0.00635,        # vane thickness [m] for single-thickness calls (vane_case)
    thicknesses=(0.003, 0.00635, 0.0127, 0.0254, 0.0508),   # run_vane_study sweeps these [m]
    gap=0.003175,          # gap between adjacent vane edges when closed [m] (1/8")
    L=3.0,                 # stick/liner length [m] (top-aligned)
    heights=(3.0, 6.0, 10.0),
    angles=(0, 2, 5, 10, 20, 30, 45, 60, 90),
    rings_per_L=10,        # axial rings over L (ring count scales with height)
    n_bins=2,              # bins across each vane face
    n_rays=2 ** 15,        # Sobol rays per surface; heat fraction within 0.014% of 2^18 rays
                           # (vane_ray_convergence, H = 6 m, 5 and 30 deg) -> well inside 1%
    T_bottom=1950.0, T_top=1850.0,   # linear block-wall temperature [K]
)


def _vane_layout(S, clearance, t_vane, gap):
    """Vane length l, liner half-width, and the closed-octagon description."""
    A = S / 2.0
    l = (A - clearance) / 2.0
    liner_half = l / 2.0 - clearance / np.sqrt(2.0)
    return dict(A=A, l=l, liner_half=liner_half, t=t_vane, gap=gap)


def vane_layout_for_liner(liner_side, t_vane, gap, clearance=0.005, S=None):
    """Vane layout around a given liner (tinless model use). Vane length is the
    minimum that clears the liner corners when open, l = liner_side + √2·clearance;
    the cavity side must be at least S_min = 4 l + 2·clearance (open face vanes
    clear the wall). Returns (layout dict, S_min); S defaults to S_min."""
    l = liner_side + np.sqrt(2.0) * clearance
    S_min = 4.0 * l + 2.0 * clearance
    S = S_min if S is None else S
    if S < S_min - 1e-9:
        raise ValueError(f"cavity side {S:.3f} m < {S_min:.3f} m needed for the vanes around a "
                         f"{liner_side:.3f} m liner")
    return dict(A=S / 2.0, l=l, liner_half=liner_side / 2.0, t=t_vane, gap=gap), S_min


def _vane_boxes(lay, theta_deg):
    """8 vane boxes (centre, long axis a, thickness axis b, half-length, class)."""
    l, t, gap = lay['l'], lay['t'], lay['gap']
    shorten = 0.5 * t * np.tan(np.pi / 8) + 0.5 * gap
    th = np.radians(theta_deg)
    c_, s_ = np.cos(th), np.sin(th)
    rot = lambda v: np.array([c_ * v[0] - s_ * v[1], s_ * v[0] + c_ * v[1]])
    boxes = []
    for k in range(8):
        ph = k * np.pi / 4
        rh = np.array([np.cos(ph), np.sin(ph)])
        tg = np.array([-np.sin(ph), np.cos(ph)])
        if k % 2 == 0:
            R, half, cls = 1.5 * l, 0.5 * l, 'f'
        else:
            R, half, cls = l * np.sqrt(2), 0.5 * l * np.sqrt(2), 'd'
        boxes.append((R * rh, rot(tg), rot(rh), half - shorten, cls))
    return boxes


def _vane_box_corners(box, t):
    c, a, b, h, _ = box
    return [c + su * h * a + sv * 0.5 * t * b for su in (-1, 1) for sv in (-1, 1)]


def _segments_cross(p1, p2, q1, q2):
    d = lambda a, b, c: (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    return (d(p1, p2, q1) * d(p1, p2, q2) < 0) and (d(q1, q2, p1) * d(q1, q2, p2) < 0)


def vane_collisions(lay, theta_deg):
    """List of problems at this angle: vane/vane overlap, vane vs wall or liner."""
    t = lay['t']
    boxes = _vane_boxes(lay, theta_deg)
    probs = []
    outlines = []
    for i, bx in enumerate(boxes):
        P = _vane_box_corners(bx, t)
        P = [P[0], P[1], P[3], P[2]]
        outlines.append(P)
        for q in P:
            if np.max(np.abs(q)) > lay['A'] - 1e-9:
                probs.append(f"vane {i} hits cavity wall")
            if np.max(np.abs(q)) < lay['liner_half'] + 1e-9:
                probs.append(f"vane {i} hits liner")
    for i in range(8):
        for j in range(i + 1, 8):
            for e1 in range(4):
                for e2 in range(4):
                    if _segments_cross(outlines[i][e1], outlines[i][(e1 + 1) % 4],
                                       outlines[j][e2], outlines[j][(e2 + 1) % 4]):
                        probs.append(f"vanes {i},{j} overlap")
    return sorted(set(probs))


def _vane_enclosure(lay, edges, edges_act, z_act0, H_low, theta_deg, n_bins, n_rays, seed=2024,
                    with_vanes=True):
    """Monte Carlo view factors for the square cavity with vanes at theta.
    Returns SimpleNamespace(F, A, names, idx, pairs)."""
    from types import SimpleNamespace
    from scipy.stats import qmc
    from .viewfactors import enforce_reciprocity
    A_, lh, t = lay['A'], lay['liner_half'], lay['t']
    l = lay['l']
    e, ea = np.asarray(edges, float), np.asarray(edges_act, float)
    n, na = len(e) - 1, len(ea) - 1
    H = e[-1]
    low = H_low > 1e-12
    has_E = low and (z_act0 - H_low > 1e-9)
    boxes = _vane_boxes(lay, theta_deg) if with_vanes else []
    INF = np.inf

    names = [f'B{k}' for k in range(n)] + [f'L{k}' for k in range(na)]
    if has_E:
        names.append('E')
    if low:
        names.append('R')
    names += ['floor_in', 'floor_out', 'ceil_in', 'ceil_out']
    if with_vanes:
        for k in range(n):
            for c in ('f', 'd'):
                names += [f'{c}{k}p{b}' for b in range(n_bins)] + [f'{c}{k}m{b}' for b in range(n_bins)]
                names += [f'{c}{k}ep', f'{c}{k}em']
    ix = {nm: i for i, nm in enumerate(names)}
    N = len(names)

    # vertical segments: (P0, P1, z0, z1, kind, extra)
    segs = []
    sq = [np.array(p, float) for p in ((1, -1), (1, 1), (-1, 1), (-1, -1))]
    for i in range(4):
        segs.append((A_ * sq[i], A_ * sq[(i + 1) % 4], 0.0, H, 'B', None))
        segs.append((lh * sq[i], lh * sq[(i + 1) % 4], H_low if low else 0.0, H, 'L', None))
    for bx in boxes:
        c, a, b, h, cls = bx
        P = lambda u, v: c + u * a + v * b
        segs.append((P(-h, t / 2), P(h, t / 2), 0.0, H, 'face', (cls, 'p')))
        segs.append((P(-h, -t / 2), P(h, -t / 2), 0.0, H, 'face', (cls, 'm')))
        segs.append((P(h, -t / 2), P(h, t / 2), 0.0, H, 'end', (cls, 'ep')))
        segs.append((P(-h, -t / 2), P(-h, t / 2), 0.0, H, 'end', (cls, 'em')))

    def in_octagon(x, y):
        ax_, ay_ = np.abs(x), np.abs(y)
        return (ax_ <= 1.5 * l) & (ay_ <= 1.5 * l) & (ax_ + ay_ <= 2 * l)

    def in_vane(x, y):
        hit = np.zeros(np.shape(x), bool)
        for c, a, b, h, _ in boxes:
            u = (x - c[0]) * a[0] + (y - c[1]) * a[1]
            v = (x - c[0]) * b[0] + (y - c[1]) * b[1]
            hit |= (np.abs(u) < h) & (np.abs(v) < t / 2)
        return hit

    ring = lambda z: np.clip(np.searchsorted(e, z, side='right') - 1, 0, n - 1)
    aring = lambda z: np.clip(np.searchsorted(ea, z, side='right') - 1, 0, na - 1)

    def trace(px, py, pz, dx, dy, dz):
        m = px.size
        tb = np.full(m, INF)
        tgt = np.full(m, -1)
        # horizontal planes
        with np.errstate(divide='ignore', invalid='ignore'):
            tf = np.where(dz < 0, -pz / dz, INF)
            tc = np.where(dz > 0, (H - pz) / dz, INF)
        tf = np.where(tf > 1e-9, tf, INF)
        tc = np.where(tc > 1e-9, tc, INF)
        xf, yf = px + np.where(np.isfinite(tf), tf, 0) * dx, py + np.where(np.isfinite(tf), tf, 0) * dy
        xc, yc = px + np.where(np.isfinite(tc), tc, 0) * dx, py + np.where(np.isfinite(tc), tc, 0) * dy
        better = tf < tb
        tb = np.where(better, tf, tb)
        tgt = np.where(better, np.where(in_octagon(xf, yf), ix['floor_in'], ix['floor_out']), tgt)
        better = tc < tb
        tb = np.where(better, tc, tb)
        tgt = np.where(better, np.where(in_octagon(xc, yc), ix['ceil_in'], ix['ceil_out']), tgt)
        if low:
            with np.errstate(divide='ignore', invalid='ignore'):
                tr = np.where((dz > 0) & (pz < H_low - 1e-12), (H_low - pz) / dz, INF)
            trf = np.where(np.isfinite(tr), tr, 0)
            xr, yr = px + trf * dx, py + trf * dy
            tr = np.where(np.isfinite(tr) & (np.abs(xr) <= lh) & (np.abs(yr) <= lh) & (tr > 1e-9), tr, INF)
            better = tr < tb
            tb = np.where(better, tr, tb)
            tgt = np.where(better, ix['R'], tgt)
        # vertical segments
        for (P0, P1, z0, z1, kind, extra) in segs:
            ex, ey = P1[0] - P0[0], P1[1] - P0[1]
            den = dx * ey - dy * ex
            ax_, ay_ = P0[0] - px, P0[1] - py
            with np.errstate(divide='ignore', invalid='ignore'):
                tt = (ax_ * ey - ay_ * ex) / den
                ss = (ax_ * dy - ay_ * dx) / den
            zz = pz + tt * dz
            ok = (np.abs(den) > 1e-15) & (tt > 1e-9) & (ss >= 0) & (ss <= 1) & (zz >= z0) & (zz <= z1) & (tt < tb)
            if not ok.any():
                continue
            if kind == 'B':
                tid = ring(zz)
            elif kind == 'L':
                tid = np.where(zz >= z_act0 - 1e-12, n + aring(zz), ix['E'] if has_E else n)
            else:
                cls, side = extra
                k = ring(zz)
                if kind == 'face':
                    b_ = np.clip((np.nan_to_num(ss) * n_bins).astype(int), 0, n_bins - 1)
                    base = np.array([ix[f'{cls}{kk}{side}0'] for kk in range(n)])
                    tid = base[k] + b_
                else:
                    base = np.array([ix[f'{cls}{kk}{side}'] for kk in range(n)])
                    tid = base[k]
            tb = np.where(ok, tt, tb)
            tgt = np.where(ok, tid, tgt)
        if np.any(tgt < 0):
            raise RuntimeError("ray escaped the cavity")
        return tgt

    def cos_dirs(u1, u2, nrm, t1, t2):
        st, ct = np.sqrt(u1), np.sqrt(1 - u1)
        ph = 2 * np.pi * u2
        a_, b_ = st * np.cos(ph), st * np.sin(ph)
        d = ct[:, None] * nrm[None, :] + a_[:, None] * t1[None, :] + b_[:, None] * t2[None, :]
        return d[:, 0], d[:, 1], d[:, 2]

    ez = np.array([0.0, 0.0, 1.0])
    m_exp = int(np.ceil(np.log2(n_rays)))
    F = np.zeros((N, N))
    Aarea = np.zeros(N)
    dz_ = np.diff(e)
    dza = np.diff(ea)

    def wall_emitter(P0, P1, z0, z1, nrm2, lo=0.0, hi=1.0, seed_i=0):
        U = qmc.Sobol(4, scramble=True, seed=seed + seed_i).random_base2(m_exp)
        s = lo + (hi - lo) * U[:, 0]
        z = z0 + (z1 - z0) * U[:, 1]
        px = P0[0] + s * (P1[0] - P0[0]) + 1e-10 * nrm2[0]
        py = P0[1] + s * (P1[1] - P0[1]) + 1e-10 * nrm2[1]
        tg = np.array([P1[0] - P0[0], P1[1] - P0[1], 0.0])
        tg /= np.linalg.norm(tg)
        dx, dy, dzz = cos_dirs(U[:, 2], U[:, 3], np.array([nrm2[0], nrm2[1], 0.0]), tg, ez)
        return trace(px, py, z, dx, dy, dzz)

    def plane_emitter(z, up, accept, area_full, seed_i):
        """Uniform points on the square |x|,|y| <= A at height z, filtered by accept."""
        m_need = 2 ** m_exp
        got = []
        mm = m_exp
        while True:
            U = qmc.Sobol(4, scramble=True, seed=seed + seed_i).random_base2(mm)
            x = (2 * U[:, 0] - 1) * A_
            y = (2 * U[:, 1] - 1) * A_
            keep = accept(x, y)
            if keep.sum() >= m_need or mm > m_exp + 6:
                break
            mm += 1
        frac = keep.mean()
        x, y, U = x[keep], y[keep], U[keep]
        nrm = np.array([0.0, 0.0, 1.0 if up else -1.0])
        dx, dy, dzz = cos_dirs(U[:, 2], U[:, 3], nrm, np.array([1.0, 0, 0]), np.array([0, 1.0, 0]))
        pz = np.full(x.size, z + (1e-10 if up else -1e-10))
        return trace(x, y, pz, dx, dy, dzz), frac * (2 * A_) ** 2

    wallB = (A_ * sq[0], A_ * sq[1])                 # x = +A, normal -x
    wallL = (lh * sq[0], lh * sq[1])                 # x = +lh, normal +x
    si = 0
    for k in range(n):
        tg_ = wall_emitter(*wallB, e[k], e[k + 1], np.array([-1.0, 0.0]), seed_i=si); si += 1
        F[ix[f'B{k}']] = np.bincount(tg_, minlength=N) / tg_.size
        Aarea[ix[f'B{k}']] = 8 * A_ * dz_[k]
    for k in range(na):
        tg_ = wall_emitter(*wallL, ea[k], ea[k + 1], np.array([1.0, 0.0]), seed_i=si); si += 1
        F[ix[f'L{k}']] = np.bincount(tg_, minlength=N) / tg_.size
        Aarea[ix[f'L{k}']] = 8 * lh * dza[k]
    if has_E:
        tg_ = wall_emitter(*wallL, H_low, z_act0, np.array([1.0, 0.0]), seed_i=si); si += 1
        F[ix['E']] = np.bincount(tg_, minlength=N) / tg_.size
        Aarea[ix['E']] = 8 * lh * (z_act0 - H_low)
    if low:
        U = qmc.Sobol(4, scramble=True, seed=seed + si).random_base2(m_exp); si += 1
        x, y = (2 * U[:, 0] - 1) * lh, (2 * U[:, 1] - 1) * lh
        dx, dy, dzz = cos_dirs(U[:, 2], U[:, 3], np.array([0, 0, -1.0]), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]))
        tg_ = trace(x, y, np.full(x.size, H_low - 1e-10), dx, dy, dzz)
        F[ix['R']] = np.bincount(tg_, minlength=N) / tg_.size
        Aarea[ix['R']] = (2 * lh) ** 2
    not_liner = lambda x, y: ~((np.abs(x) < lh) & (np.abs(y) < lh))
    for nm, z, up, inside in (('floor_in', 0.0, True, True), ('floor_out', 0.0, True, False),
                              ('ceil_in', H, False, True), ('ceil_out', H, False, False)):
        excl_liner = (nm.startswith('ceil') or not low)
        acc = (lambda ins, ex: (lambda x, y: (in_octagon(x, y) == ins) & ~in_vane(x, y)
                                & (not_liner(x, y) if ex else True)))(inside, excl_liner)
        tg_, area = plane_emitter(z, up, acc, None, si); si += 1
        F[ix[nm]] = np.bincount(tg_, minlength=N) / tg_.size
        Aarea[ix[nm]] = area
    if with_vanes:
        reps = {'f': boxes[0], 'd': boxes[1]}
        for cls, (c, a, b, h, _) in reps.items():
            P = lambda u, v: c + u * a + v * b
            fp = (P(-h, t / 2), P(h, t / 2), b)
            fm = (P(-h, -t / 2), P(h, -t / 2), -b)
            ep = (P(h, -t / 2), P(h, t / 2), a)
            em = (P(-h, -t / 2), P(-h, t / 2), -a)
            for k in range(n):
                for side, (P0, P1, nrm) in (('p', fp), ('m', fm)):
                    for bb in range(n_bins):
                        tg_ = wall_emitter(P0, P1, e[k], e[k + 1], nrm, bb / n_bins, (bb + 1) / n_bins, seed_i=si); si += 1
                        nm = f'{cls}{k}{side}{bb}'
                        F[ix[nm]] = np.bincount(tg_, minlength=N) / tg_.size
                        Aarea[ix[nm]] = 4 * (2 * h / n_bins) * dz_[k]
                for side, (P0, P1, nrm) in (('ep', ep), ('em', em)):
                    tg_ = wall_emitter(P0, P1, e[k], e[k + 1], nrm, seed_i=si); si += 1
                    nm = f'{cls}{k}{side}'
                    F[ix[nm]] = np.bincount(tg_, minlength=N) / tg_.size
                    Aarea[ix[nm]] = 4 * t * dz_[k]
    Fr = enforce_reciprocity(F, Aarea)
    pairs = [(f'{c}{k}p{bb}', f'{c}{k}m{bb}') for k in range(n) for c in ('f', 'd') for bb in range(n_bins)] \
        if with_vanes else None
    slat = np.array([nm in {p for pr in (pairs or []) for p in pr} for nm in names])
    return SimpleNamespace(F=Fr, A=Aarea, names=names, idx=ix, pairs=pairs, slat=slat)


def _vane_sim(cfg, lay, L, n_act, overrides=None):
    """TinlessSimulation carrying the liner/stick/vane parameters for the solver."""
    from .build import build_tinless
    p = cfg['powerstick']
    stick_ratio = p['side_len'] / p['liner_side_len']        # keep the YAML stick/liner ratio
    ov = {'storage.L': L, 'storage.block_height': None, 'storage.num_ax_segments': n_act,
          'radiation.axial': False, 'shade.shade_bool': False,
          'powerstick.liner_side_len': 2 * lay['liner_half'],
          'powerstick.side_len': 2 * lay['liner_half'] * stick_ratio,
          'shade.thickness': lay['t']}
    ov.update(overrides or {})
    return build_tinless(cfg, ov).sim


def vane_case(cfg, H, theta_deg, prm=None, with_vanes=True):
    """One ray-trace + coupled radiation solve. Returns dict of results."""
    from types import SimpleNamespace
    from .viewfactors import axial_mesh
    from .axial import AxialRadiation
    P = dict(VANE_DEFAULTS, **(prm or {}))
    lay = _vane_layout(P['S'], P['clearance'], P['t_vane'], P['gap'])
    L = P['L']
    n_ax = int(round(P['rings_per_L'] * H / L))
    n_act, n_low, edges, edges_act, z_act0 = axial_mesh(L, H, n_ax)
    sim = _vane_sim(cfg, lay, L, n_act)
    t_lb = sim.liner_thickness
    t_rgf = float((cfg.get('liner_bottom') or {}).get('rgf_thickness', 0.025) or 0.0)
    H_low = H - L - t_lb - t_rgf if n_low else 0.0
    enc = _vane_enclosure(lay, edges, edges_act, z_act0, H_low, theta_deg, P['n_bins'],
                          P['n_rays'], with_vanes=with_vanes)
    g = dict(edges=edges, edges_act=edges_act, n_act=n_act, n_low=n_low, z_act0=z_act0)
    if with_vanes:
        vfs = {'porous': SimpleNamespace(C=enc.F[None], A_ring=enc.A, slat=enc.slat, names=enc.names,
                                         idx=enc.idx, pairs=enc.pairs)}
    else:
        vfs = {'single': SimpleNamespace(F=enc.F, A=enc.A, names=enc.names, idx=enc.idx)}
    ax = AxialRadiation(sim, g, vfs)
    zc = 0.5 * (edges[1:] + edges[:-1])
    T = P['T_bottom'] + (P['T_top'] - P['T_bottom']) * zc / H
    s = ax.solve(T, f=1.0)
    return dict(H=H, theta=theta_deg, Q_ps=float(s['Q_ps'].sum()), Q_cell=float(s['Q_cell'].sum()),
                Q_low=float(s['Q_block'][:n_low].sum()), Q_block=float(s['Q_block'].sum()),
                T_liner_in=float(np.mean(s['T_liner_in'])), balance=float(s['balance_err']), Q_rings=s['Q_block'].copy(), n_low=n_low,
                n_surfaces=len(enc.names), layout=lay)


def _phi_model(theta_deg, h, c):
    """Heat-fraction fit (9 coefficients c):
        phi = p0 + (p90 - p0) * s^a / (s^a + k (1 - s)^b),   s = sin(theta)
        g   = 1 - exp(-(H/L - 1) / c8)            (0 at H/L = 1, saturates with height)
        p0  = c0 + c1 g,  ln k = c2 + c3 g,  a = c4 + c5 g,  p90 = c6,  b = c7"""
    s = np.sin(np.radians(np.asarray(theta_deg, float)))
    g = 1.0 - np.exp(-(np.asarray(h, float) - 1.0) / c[8])
    p0 = c[0] + c[1] * g
    k = np.exp(c[2] + c[3] * g)
    a = c[4] + c[5] * g
    p90, b = c[6], c[7]
    with np.errstate(divide='ignore', invalid='ignore'):
        w = np.where(s > 0, s ** a / (s ** a + k * (1 - s) ** b), 0.0)
    return p0 + (p90 - p0) * w


def fit_vane_heat_fraction(theta, h, phi):
    """Least-squares fit of _phi_model to data arrays. Returns (coeffs, max_abs_err)."""
    from scipy.optimize import least_squares
    theta, h, phi = map(lambda v: np.asarray(v, float), (theta, h, phi))
    x0 = np.array([phi[theta == theta.min()].min(), 0.05, -2.0, 0.0, 1.2, 0.0,
                   phi[theta == theta.max()].mean(), 0.4, 0.5])
    lo = [0, -1, -10, -5, 0.05, -2, 0.5, 0.05, 0.02]
    hi = [1.5, 1, 10, 5, 5, 2, 1.5, 5, 20]
    res = least_squares(lambda c: _phi_model(theta, h, c) - phi, x0, bounds=(lo, hi))
    return res.x, float(np.max(np.abs(res.fun)))


def vane_R_cond(t_vane, T_ref=None, k=None):
    """Vane through-thickness conduction resistance per unit area R'' = t / k [m^2 K/W],
    with k = RGF k(T_ref) (thermal.properties.ins_tcx) unless k is given.
    T_ref defaults to the study's mean wall temperature."""
    from .properties import ins_tcx
    if k is None:
        T_ref = 0.5 * (VANE_DEFAULTS['T_bottom'] + VANE_DEFAULTS['T_top']) if T_ref is None else T_ref
        k = float(ins_tcx(T_ref))
    return np.asarray(t_vane, float) / k


# Results of run_vane_study() with VANE_DEFAULTS, 2026-11-17 (S = 1.2 m, 1/8" closed
# gap, L = 3 m, H = 3/6/10 m, wall 1950 K bottom -> 1850 K top, 2^15 rays/surface).
#
# VANE_GRID: the traced heat fractions themselves. vane_heat_fraction() interpolates
#   them (default method='grid'): monotone PCHIP in theta, linear in ln(R''), linear
#   in H/L. Exact at the traced points. Angles 1, 3, 7.5, 15 deg were added later
#   (extend_vane_study); before them, interpolation missed those points by up to
#   0.004 (<= 12.7 mm), 0.008 (25.4 mm) and 0.023 (50.8 mm), so with the 13-angle
#   grid the interpolation error should be well under ~0.01 even for thick vanes.
#   Vanes run the full block height (floor to ceiling); stick/liner top-aligned.
# VANE_FIT_TABLE: one 9-coefficient _phi_model fit per R'' (method='fit').
#   Max |fit - data|: 0.003 (3 mm), 0.005 (6.35), 0.009 (12.7), 0.025 (25.4), 0.027 (50.8 mm)
#   -- the smooth form cannot follow the thick-vane kinks.
# Traced angles [deg]: 0, 1, 2, 3, 5, 7.5, 10, 15, 20, 30, 45, 60, 90
VANE_GRID = dict(
    R_cond=[0.00416758, 0.00882139, 0.0176428, 0.0352855, 0.0705711],   # m2K/W (t = 3, 6.35, 12.7, 25.4, 50.8 mm RGF at 1900 K)
    H_over_L=[1, 2, 3.33333],
    theta=[0, 1, 2, 3, 5, 7.5, 10, 15, 20, 30, 45, 60, 90],
    phi=[  # [R][H/L][theta]
        [[0.81454, 0.82092, 0.83305, 0.84455, 0.86377, 0.88179, 0.89553, 0.91534, 0.92974, 0.94992, 0.96964, 0.98366, 0.99579],
         [0.86425, 0.86841, 0.87628, 0.8843, 0.89736, 0.90957, 0.91915, 0.9331, 0.94353, 0.95836, 0.97355, 0.98498, 0.9951],
         [0.86309, 0.86725, 0.87507, 0.88287, 0.89594, 0.90838, 0.91781, 0.9322, 0.94264, 0.95789, 0.97351, 0.98517, 0.99548]],
        [[0.71104, 0.72152, 0.74245, 0.76803, 0.80856, 0.84303, 0.86683, 0.89777, 0.91807, 0.94409, 0.96712, 0.98233, 0.99484],
         [0.79052, 0.79793, 0.81231, 0.82976, 0.85789, 0.8813, 0.89793, 0.9198, 0.9345, 0.95371, 0.97145, 0.98379, 0.99417],
         [0.79205, 0.79885, 0.81278, 0.82924, 0.85655, 0.88011, 0.8964, 0.91869, 0.93343, 0.9531, 0.97136, 0.98397, 0.99455]],
        [[0.59351, 0.61652, 0.63644, 0.66681, 0.73843, 0.7987, 0.83659, 0.88112, 0.90758, 0.93888, 0.96458, 0.98055, 0.99292],
         [0.70192, 0.72004, 0.73458, 0.75641, 0.80724, 0.84924, 0.87562, 0.90702, 0.92625, 0.94944, 0.96927, 0.98217, 0.99235],
         [0.71054, 0.72652, 0.73944, 0.75942, 0.80709, 0.84811, 0.874, 0.90583, 0.92504, 0.94876, 0.96916, 0.98236, 0.99279]],
        [[0.47781, 0.52772, 0.5671, 0.58715, 0.63147, 0.72351, 0.78845, 0.85738, 0.89324, 0.93163, 0.96063, 0.97729, 0.98883],
         [0.60492, 0.64815, 0.68006, 0.69553, 0.72784, 0.79481, 0.84027, 0.88882, 0.91489, 0.94354, 0.96584, 0.97919, 0.9885],
         [0.62472, 0.66131, 0.68873, 0.70253, 0.73178, 0.79488, 0.83888, 0.88754, 0.91379, 0.94272, 0.96574, 0.97943, 0.989]],
        [[0.3881, 0.46926, 0.5404, 0.58291, 0.62299, 0.64008, 0.68024, 0.80012, 0.86162, 0.91679, 0.9525, 0.97033, 0.97946],
         [0.51933, 0.59362, 0.65322, 0.68635, 0.71744, 0.7308, 0.75944, 0.84567, 0.89003, 0.93121, 0.95858, 0.97276, 0.97978],
         [0.54728, 0.61132, 0.66258, 0.69232, 0.72055, 0.73237, 0.76012, 0.84397, 0.889, 0.93032, 0.9585, 0.97314, 0.98035]],
    ],
)

VANE_FIT_TABLE = {
    0.00416758: [0.812012, 0.049725, -1.484070, 0.181502, 1.030017, -0.029868, 0.995746, 0.560836, 0.029688],
    0.00882139: [0.705986, 0.081615, -2.221323, 0.188140, 1.194545, -0.041337, 0.994601, 0.365000, 0.031794],
    0.0176428: [0.590925, 0.115909, -2.862785, 0.187708, 1.421847, -0.057383, 0.992039, 0.202740, 0.314636],
    0.0352855: [0.490711, 0.138951, -2.494118, 0.181120, 1.340385, -0.091838, 0.985985, 0.934695, 0.395769],
    0.0705711: [0.386280, 0.153867, -0.420020, 0.018049, 0.496090, -0.040728, 0.971932, 3.369572, 0.426295],
}


def _grid_interp(theta_deg, H_over_L, R_cond, grid):
    from scipy.interpolate import PchipInterpolator
    Rg = np.asarray(grid['R_cond'], float)
    hg = np.asarray(grid['H_over_L'], float)
    tg = np.asarray(grid['theta'], float)
    P = np.asarray(grid['phi'], float)
    th = np.clip(np.asarray(theta_deg, float), tg[0], tg[-1])
    lnR = np.log(Rg)
    x = float(np.clip(np.log(R_cond), lnR[0], lnR[-1]))
    i = int(np.clip(np.searchsorted(lnR, x) - 1, 0, len(Rg) - 2))
    wr = (x - lnR[i]) / (lnR[i + 1] - lnR[i])
    h = float(np.clip(H_over_L, hg[0], hg[-1]))
    j = int(np.clip(np.searchsorted(hg, h) - 1, 0, len(hg) - 2))
    wh = (h - hg[j]) / (hg[j + 1] - hg[j])
    f = lambda a, b: PchipInterpolator(tg, P[a, b])(th)
    return ((1 - wr) * ((1 - wh) * f(i, j) + wh * f(i, j + 1))
            + wr * ((1 - wh) * f(i + 1, j) + wh * f(i + 1, j + 1)))


def vane_heat_fraction(theta_deg, H_over_L, t_vane=None, R_cond=None, k=None, method='grid',
                       grid=None, table=None):
    """Heat into the stick with vanes at theta (0 = closed, 90 = edge-on), as a
    fraction of the no-vane value, for block/stick height ratio H_over_L.

    Vane given by EITHER t_vane [m] (RGF felt, k(T_ref) from ins_tcx, or pass k
    [W/m-K] for another material) OR R_cond = t/k [m^2 K/W] directly. Default:
    VANE_DEFAULTS['t_vane']. theta may be an array; H_over_L and R_cond scalars.

    method='grid' (default): interpolate the traced data (VANE_GRID, or `grid`
    from load_vane_grid()). method='fit': per-R'' parametric fits (VANE_FIT_TABLE
    or `table`), interpolated linearly in ln(R''). Outside the studied R''/H/L
    range the nearest data are used and a warning is raised.
    Valid for the study's layout (max-size square-cavity octagon)."""
    import warnings
    if R_cond is None:
        R_cond = vane_R_cond(VANE_DEFAULTS['t_vane'] if t_vane is None else t_vane, k=k)
    R_cond = float(R_cond)
    if method == 'grid':
        gr = VANE_GRID if grid is None else grid
        Rg, hg = gr['R_cond'], gr['H_over_L']
        if not (min(Rg) * (1 - 1e-4) <= R_cond <= max(Rg) * (1 + 1e-4)) or not (min(hg) - 1e-4 <= H_over_L <= max(hg) + 1e-4):
            warnings.warn(f"R_cond = {R_cond:.3g} m2K/W or H/L = {H_over_L} outside the studied range "
                          f"(R'' {min(Rg):.3g}-{max(Rg):.3g}, H/L {min(hg):.2f}-{max(hg):.2f}); "
                          "using the nearest data", stacklevel=2)
        return _grid_interp(theta_deg, H_over_L, R_cond, gr)
    tab = VANE_FIT_TABLE if table is None else table
    Rs = np.array(sorted(tab))
    if len(Rs) == 1:
        return _phi_model(theta_deg, H_over_L, np.asarray(tab[Rs[0]]))
    lnRs = np.log(Rs)
    lnR = np.log(R_cond)
    if lnR < lnRs[0] - 1e-9 or lnR > lnRs[-1] + 1e-9:
        warnings.warn(f"R_cond = {R_cond:.3g} m2K/W outside the studied range "
                      f"[{Rs[0]:.3g}, {Rs[-1]:.3g}]; using the nearest fit", stacklevel=2)
    lnR = float(np.clip(lnR, lnRs[0], lnRs[-1]))
    j = int(np.clip(np.searchsorted(lnRs, lnR) - 1, 0, len(Rs) - 2))
    w = (lnR - lnRs[j]) / (lnRs[j + 1] - lnRs[j])
    lo = _phi_model(theta_deg, H_over_L, np.asarray(tab[Rs[j]]))
    hi = _phi_model(theta_deg, H_over_L, np.asarray(tab[Rs[j + 1]]))
    return (1 - w) * lo + w * hi


def run_vane_study(cfg=None, prm=None, out='vane_heat_fraction_study.npz', verbose=True):
    """Full study: every thickness x height x angle, plus the no-vane reference per
    height; fits _phi_model per thickness and saves data + fits to `out`.
    Returns (rows, table, errors), table = {R'': coeffs} for vane_heat_fraction(table=...).

    prm overrides VANE_DEFAULTS, e.g. prm={'thicknesses': (0.01, 0.02), 'S': 1.3}."""
    import time
    import yaml
    import os
    if cfg is None:
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        from .build import load_config
        cfg = load_config(os.path.join(here, 'tinless_model_input.yaml'))
    P = dict(VANE_DEFAULTS, **(prm or {}))
    T_ref = 0.5 * (P['T_bottom'] + P['T_top'])
    refs = {}
    rows, table, errs = [], {}, {}
    for H in P['heights']:
        t0 = time.perf_counter()
        refs[H] = vane_case(cfg, H, 0.0, P, with_vanes=False)
        if verbose:
            print(f"H = {H} m (H/L = {H/P['L']:.2f}): no vanes Q_stick {refs[H]['Q_ps']/1e3:.1f} kW, "
                  f"lower block {refs[H]['Q_low']/1e3:.1f} kW  [{time.perf_counter()-t0:.0f} s]", flush=True)
    for t_v in P['thicknesses']:
        Pt = dict(P, t_vane=t_v)
        lay = _vane_layout(P['S'], P['clearance'], t_v, P['gap'])
        R = float(vane_R_cond(t_v, T_ref))
        if verbose:
            print(f"\n=== vane t = {t_v*1e3:.2f} mm, R'' = {R:.4g} m2K/W (k at {T_ref:.0f} K) | cavity side "
                  f"{P['S']} m, vane l = {lay['l']*100:.2f} / {lay['l']*np.sqrt(2)*100:.2f} cm, liner side "
                  f"{2*lay['liner_half']*100:.2f} cm, closed gap {lay['gap']*1e3:.2f} mm, rays {P['n_rays']}")
            for th in P['angles']:
                pr = vane_collisions(lay, th)
                if pr:
                    print(f"  WARNING theta={th}: {pr}")
        sub = []
        for H in P['heights']:
            ref = refs[H]
            for th in P['angles']:
                t0 = time.perf_counter()
                r = vane_case(cfg, H, th, Pt)
                r.update(t_vane=t_v, R_cond=R, phi=r['Q_ps'] / ref['Q_ps'], phi_cell=r['Q_cell'] / ref['Q_cell'],
                         low_share=r['Q_low'] / r['Q_block'] if r['Q_block'] else np.nan,
                         ref_low_share=ref['Q_low'] / ref['Q_block'])
                sub.append(r)
                if verbose:
                    print(f"  H {H:4.1f} theta {th:5.1f}: heat fraction {r['phi']:.4f}  cell {r['phi_cell']:.4f}  "
                          f"lower-block share {r['low_share']:.3f} (no vanes {r['ref_low_share']:.3f})  "
                          f"T_liner_in {r['T_liner_in']:.0f} K  [{time.perf_counter()-t0:.0f} s]", flush=True)
        rows += sub
        th_ = np.array([r['theta'] for r in sub], float)
        hr_ = np.array([r['H'] for r in sub], float) / P['L']
        ph_ = np.array([r['phi'] for r in sub])
        coeffs, err = fit_vane_heat_fraction(th_, hr_, ph_)
        table[R], errs[R] = [float(c) for c in coeffs], err
        if verbose:
            print(f"  fit: max |fit - data| = {err:.4f}")
    if verbose:
        print("\nVANE_FIT_TABLE = {")
        for R in sorted(table):
            print(f"    {R:.6g}: [{', '.join(f'{c:.6f}' for c in table[R])}],")
        print("}")
    np.savez(out, theta=np.array([r['theta'] for r in rows], float),
             H_over_L=np.array([r['H'] for r in rows], float) / P['L'],
             t_vane=np.array([r['t_vane'] for r in rows]), R_cond=np.array([r['R_cond'] for r in rows]),
             phi=np.array([r['phi'] for r in rows]), phi_cell=np.array([r['phi_cell'] for r in rows]),
             low_share=np.array([r['low_share'] for r in rows]),
             table_R=np.array(sorted(table)), table_coeffs=np.array([table[R] for R in sorted(table)]),
             params=str(P))
    return rows, table, errs


def extend_vane_study(angles, path='vane_heat_fraction_study.npz', cfg=None, prm=None, verbose=True):
    """Trace only the extra `angles` for every thickness and height already in `path`,
    merge them into the file (existing points are kept, not rerun) and refit the
    per-thickness parametric fits. prm must match the original study's settings
    (defaults: VANE_DEFAULTS). Returns the merged grid (see load_vane_grid)."""
    import time
    import yaml
    import os
    if cfg is None:
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        from .build import load_config
        cfg = load_config(os.path.join(here, 'tinless_model_input.yaml'))
    P = dict(VANE_DEFAULTS, **(prm or {}))
    d = dict(np.load(path))
    T_ref = 0.5 * (P['T_bottom'] + P['T_top'])
    thicknesses = np.unique(d['t_vane'])
    Hs = np.unique(np.round(d['H_over_L'] * P['L'], 9))
    new = {k: [] for k in ('theta', 'H_over_L', 't_vane', 'R_cond', 'phi', 'phi_cell', 'low_share')}
    refs = {H: vane_case(cfg, H, 0.0, P, with_vanes=False) for H in Hs}
    for t_v in thicknesses:
        Pt = dict(P, t_vane=float(t_v))
        lay = _vane_layout(P['S'], P['clearance'], float(t_v), P['gap'])
        R = float(vane_R_cond(t_v, T_ref))
        for H in Hs:
            for th in angles:
                if np.any(np.isclose(d['t_vane'], t_v) & np.isclose(d['H_over_L'] * P['L'], H)
                          & np.isclose(d['theta'], th)):
                    continue
                if vane_collisions(lay, th):
                    print(f"  WARNING t={t_v*1e3:.2f} mm theta={th}: {vane_collisions(lay, th)}")
                t0 = time.perf_counter()
                r = vane_case(cfg, H, th, Pt)
                ref = refs[H]
                vals = dict(theta=th, H_over_L=H / P['L'], t_vane=float(t_v), R_cond=R,
                            phi=r['Q_ps'] / ref['Q_ps'], phi_cell=r['Q_cell'] / ref['Q_cell'],
                            low_share=r['Q_low'] / r['Q_block'] if r['Q_block'] else np.nan)
                for k, v in vals.items():
                    new[k].append(v)
                if verbose:
                    print(f"  t {t_v*1e3:5.2f} mm  H {H:4.1f}  theta {th:5.1f}: heat fraction {vals['phi']:.4f}  "
                          f"[{time.perf_counter()-t0:.0f} s]", flush=True)
    for k, v in new.items():
        d[k] = np.concatenate([d[k], np.asarray(v, float)])
    order = np.lexsort((d['theta'], d['H_over_L'], d['t_vane']))
    for k in new:
        d[k] = d[k][order]
    # refit per thickness
    table = {}
    for t_v in thicknesses:
        m = np.isclose(d['t_vane'], t_v)
        coeffs, err = fit_vane_heat_fraction(d['theta'][m], d['H_over_L'][m], d['phi'][m])
        table[float(d['R_cond'][m][0])] = [float(c) for c in coeffs]
        if verbose:
            print(f"  refit t = {t_v*1e3:.2f} mm: max |fit - data| = {err:.4f}")
    d['table_R'] = np.array(sorted(table))
    d['table_coeffs'] = np.array([table[R] for R in sorted(table)])
    np.savez(path, **d)
    return load_vane_grid(path)


def load_vane_table(path='vane_heat_fraction_study.npz'):
    """{R'': coeffs} from a saved run_vane_study() file, for vane_heat_fraction(method='fit', table=...)."""
    d = np.load(path)
    return {float(R): list(c) for R, c in zip(d['table_R'], d['table_coeffs'])}


def load_vane_grid(path='vane_heat_fraction_study.npz'):
    """Grid of traced heat fractions from a saved run_vane_study() file, for
    vane_heat_fraction(grid=...)."""
    d = np.load(path)
    Rs, hs, ths = np.unique(d['R_cond']), np.unique(d['H_over_L']), np.unique(d['theta'])
    P = np.full((len(Rs), len(hs), len(ths)), np.nan)
    for R, h, t, ph in zip(d['R_cond'], d['H_over_L'], d['theta'], d['phi']):
        P[np.searchsorted(Rs, R), np.searchsorted(hs, h), np.searchsorted(ths, t)] = ph
    return dict(R_cond=Rs, H_over_L=hs, theta=ths, phi=P)


def vane_ray_convergence(cfg=None, H=6.0, angles=(5, 30), ray_exps=(14, 15, 16, 17, 18)):
    """Heat fraction vs ray count (justifies VANE_DEFAULTS['n_rays'] for 1% accuracy)."""
    import yaml
    import os
    if cfg is None:
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        from .build import load_config
        cfg = load_config(os.path.join(here, 'tinless_model_input.yaml'))
    out = {}
    for m in ray_exps:
        prm = {'n_rays': 2 ** m}
        ref = vane_case(cfg, H, 0.0, prm, with_vanes=False)
        out[m] = [vane_case(cfg, H, th, prm)['Q_ps'] / ref['Q_ps'] for th in angles]
        print(f"2^{m} rays: " + "  ".join(f"theta {a}: {v:.4f}" for a, v in zip(angles, out[m])), flush=True)
    best = out[max(ray_exps)]
    for m in ray_exps:
        print(f"2^{m}: max rel diff vs 2^{max(ray_exps)} = "
              f"{max(abs(a / b - 1) for a, b in zip(out[m], best)):.3%}")
    return out


# -----------------------------------------------------------------------------
# Normalised vane position psi (used by the tinless model's vane hybrid)
# -----------------------------------------------------------------------------
#   psi(theta) = (Q(theta) - Q_closed_wall) / (Q(90 deg) - Q_closed_wall)
# with Q the heat into the stick and Q_closed_wall a solid closed vane ring (no
# corner gaps). psi is nearly independent of temperature level and gradient
# (+-0.02 over 1900-2650 K, 300 K gradient, H = 3 and 6 m; 2026-11-17 check), so
# the model computes the two anchors at runtime and blends them with psi.
CLOSED_WALL_GAP = -0.002      # overlapping vane ends -> solid wall


def build_vane_psi(path='vane_heat_fraction_study.npz', cfg=None, prm=None, verbose=True):
    """Trace the solid closed wall for every thickness x height in `path` and store
    its heat fraction (cw_*) in the file, so load_vane_psi_grid() can form psi."""
    import time
    import yaml
    import os
    if cfg is None:
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        from .build import load_config
        cfg = load_config(os.path.join(here, 'tinless_model_input.yaml'))
    P = dict(VANE_DEFAULTS, **(prm or {}))
    d = dict(np.load(path))
    Hs = np.unique(np.round(d['H_over_L'] * P['L'], 9))
    refs = {H: vane_case(cfg, H, 0.0, P, with_vanes=False) for H in Hs}
    cw_t, cw_h, cw_phi = [], [], []
    for t_v in np.unique(d['t_vane']):
        for H in Hs:
            t0 = time.perf_counter()
            r = vane_case(cfg, H, 0.0, dict(P, t_vane=float(t_v), gap=CLOSED_WALL_GAP))
            cw_t.append(float(t_v)); cw_h.append(H / P['L']); cw_phi.append(r['Q_ps'] / refs[H]['Q_ps'])
            if verbose:
                print(f"  closed wall t {t_v*1e3:5.2f} mm H {H:4.1f}: heat fraction {cw_phi[-1]:.4f} "
                      f"[{time.perf_counter()-t0:.0f} s]", flush=True)
    d.update(cw_t=np.array(cw_t), cw_H_over_L=np.array(cw_h), cw_phi=np.array(cw_phi))
    np.savez(path, **d)
    return load_vane_psi_grid(path)


def load_vane_psi_grid(path='vane_heat_fraction_study.npz'):
    """psi grid (same layout as load_vane_grid) from a study file with closed-wall data."""
    g = load_vane_grid(path)
    d = np.load(path)
    psi = np.empty_like(g['phi'])
    i90 = int(np.argmin(np.abs(g['theta'] - 90.0)))
    t_of_R = {float(R): float(t) for R, t in zip(d['R_cond'], d['t_vane'])}
    for i, R in enumerate(g['R_cond']):
        for j, h in enumerate(g['H_over_L']):
            m = np.isclose(d['cw_t'], t_of_R[float(R)]) & np.isclose(d['cw_H_over_L'], h)
            cw = float(d['cw_phi'][m][0])
            psi[i, j] = (g['phi'][i, j] - cw) / (g['phi'][i, j, i90] - cw)
    return dict(R_cond=g['R_cond'], H_over_L=g['H_over_L'], theta=g['theta'], phi=psi)


VANE_PSI_GRID = None          # filled in below from the study file (see build_vane_psi)


def vane_psi(theta_deg, H_over_L, R_cond, grid=None):
    """Normalised vane position psi(theta) for vane conduction resistance R'' [m2K/W]."""
    gr = VANE_PSI_GRID if grid is None else grid
    if gr is None:
        raise RuntimeError("VANE_PSI_GRID not built: run build_vane_psi()")
    return _grid_interp(theta_deg, H_over_L, float(R_cond), gr)


def vane_angle_from_psi(psi, H_over_L, R_cond, grid=None, n=2001):
    """Inverse of vane_psi (psi is monotone in theta); clamps to [psi(0), 1]."""
    th = np.linspace(0.0, 90.0, n)
    ps = vane_psi(th, H_over_L, R_cond, grid)
    ps = np.maximum.accumulate(ps)
    return float(np.interp(psi, ps, th))


if __name__ == '__main__':
    import sys
    if '--convergence' in sys.argv:
        vane_ray_convergence()
    elif '--psi' in sys.argv:
        build_vane_psi()
    else:
        run_vane_study()
