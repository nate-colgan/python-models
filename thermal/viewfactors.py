"""
View factors for the full-height emitter cavity (decoupled block height).

Geometry (z = 0 at cavity floor, bottom-up; equivalent coaxial cylinders):

Shaded case: two enclosures coupled through the shade's open fraction
  outer  (block-shade gap, 0..H):
      B_k        block wall rings          cylinder r_in, facing in
      So_k       shade outer-face rings    cylinder r_sh, facing out
      floor_out  annulus r_sh..r_in, z = 0
      ceil_out   annulus r_sh..r_in, z = H
  inner  (shade-liner gap + space under the liner, 0..H):
      Si_k       shade inner-face rings    cylinder r_sh, facing in
      L_k        liner outer-wall rings    cylinder r_liner, z_act0..H
      E          liner-bottom + pad edge   cylinder r_liner, H_low..z_act0
      R          pad underside             disk r < r_liner at z = H_low
      floor_in   disk r < r_sh at z = 0    (annulus r_liner..r_sh if no lower zone)
      ceil_in    annulus r_liner..r_sh at z = H  (1 cm ceiling-liner gap closed, lumped)

Unshaded case: one enclosure, same as `inner` but with the block wall rings
(B_k, r_in) as the outer cylinder.

Methods
-------
Analytic (exact; used whenever both cylinders span the full height, so the
ring-to-ring kernel depends only on axial separation):
  finite coaxial-cylinder relations (Howell catalog C-93) for outer->inner and
  outer->self, ring-to-ring by exchange superposition, ring->end annulus by
  closure, floor<->ceiling by row closure. Row sums and reciprocity hold to
  round-off.
Monte Carlo (inner/unshaded enclosure when the liner stops above the floor,
or on request for cross-checks): vectorised diffuse ray tracing with scrambled
Sobol sampling (fixed seed), then symmetric Sinkhorn scaling so reciprocity and
row sums hold exactly. Accuracy target: 2% on every view factor > 0.05.

Results are cached on disk by geometry (cache_dir).
"""
import hashlib
import os
from types import SimpleNamespace

import numpy as np

_INF = np.inf


# ============================================================================
# Analytic coaxial-cylinder relations (Howell catalog C-93)
# ============================================================================
def coax_F_outer_to_inner(r1, r2, h):
    """F from inner surface of outer cylinder (r2) to outer surface of inner
    cylinder (r1), both of length h, aligned. Vectorised in h; F(0) = 0."""
    h = np.asarray(h, dtype=float)
    hs = np.where(h > 0, h, 1.0)
    R = r2 / r1
    H = hs / r1
    A = H ** 2 + R ** 2 - 1.0
    B = H ** 2 - R ** 2 + 1.0
    term = (np.sqrt((A + 2.0) ** 2 - (2.0 * R) ** 2) * np.arccos(np.clip(B / (R * A), -1, 1))
            + B * np.arcsin(1.0 / R) - np.pi * A / 2.0)
    F = 1.0 / R - (1.0 / (np.pi * R)) * (np.arccos(np.clip(B / A, -1, 1)) - term / (2.0 * H))
    return np.where(h > 0, F, 0.0)


def coax_F_outer_self(r1, r2, h):
    """F from inner surface of outer cylinder (r2) to itself, with a coaxial
    inner cylinder (r1) of the same length h. Vectorised in h; F(0) = 0."""
    h = np.asarray(h, dtype=float)
    hs = np.where(h > 0, h, 1.0)
    R = r2 / r1
    H = hs / r1
    s = np.sqrt(4 * R ** 2 + H ** 2)
    t1 = 1 - 1 / R + 2 / (np.pi * R) * np.arctan(2 * np.sqrt(R ** 2 - 1) / H)
    br = (s / H * np.arcsin(np.clip((4 * (R ** 2 - 1) + (H ** 2 / R ** 2) * (R ** 2 - 2))
                                    / (H ** 2 + 4 * (R ** 2 - 1)), -1, 1))
          - np.arcsin((R ** 2 - 2) / R ** 2) + np.pi / 2 * (s / H - 1))
    F = t1 - H / (2 * np.pi * R) * br
    return np.where(h > 0, F, 0.0)


def _superpose(G, ea, eb):
    """Exchange A_i F_ij between rings [ea_i, ea_i+1] and [eb_j, eb_j+1] of two
    coaxial full-height cylinders, from G(x) = A(x) F(x) for aligned length-x
    segments:  X = ½[G(|b-c|) + G(|a-d|) - G(|a-c|) - G(|b-d|)]."""
    a, b = ea[:-1, None], ea[1:, None]
    c, d = eb[None, :-1], eb[None, 1:]
    return 0.5 * (G(np.abs(b - c)) + G(np.abs(a - d)) - G(np.abs(a - c)) - G(np.abs(b - d)))


def _analytic_annulus_enclosure(r1, r2, edges, outer_name, inner_name, floor_name, ceil_name):
    """Exact view factors for an annular enclosure between full-height coaxial
    cylinders r1 < r2 (z = edges[0]..edges[-1]) closed by end annuli."""
    e = np.asarray(edges, dtype=float) - edges[0]
    H = e[-1]
    n = len(e) - 1
    G_oo = lambda x: 2 * np.pi * r2 * x * coax_F_outer_self(r1, r2, x)
    G_oi = lambda x: 2 * np.pi * r2 * x * coax_F_outer_to_inner(r1, r2, x)
    # outer segment [0,x] -> one end annulus, and inner segment [0,x] -> one end annulus
    G_oe = lambda x: 2 * np.pi * r2 * x * 0.5 * (1 - coax_F_outer_self(r1, r2, x) - coax_F_outer_to_inner(r1, r2, x))
    G_ie = lambda x: 2 * np.pi * r1 * x * 0.5 * (1 - (r2 / r1) * coax_F_outer_to_inner(r1, r2, x))

    N = 2 * n + 2
    io, ii, ifl, ice = np.arange(n), n + np.arange(n), 2 * n, 2 * n + 1
    A = np.r_[2 * np.pi * r2 * np.diff(e), 2 * np.pi * r1 * np.diff(e),
              np.pi * (r2 ** 2 - r1 ** 2), np.pi * (r2 ** 2 - r1 ** 2)]
    X = np.zeros((N, N))
    X[np.ix_(io, io)] = _superpose(G_oo, e, e)
    Xoi = _superpose(G_oi, e, e)
    X[np.ix_(io, ii)] = Xoi
    X[np.ix_(ii, io)] = Xoi.T
    a, b = e[:-1], e[1:]
    X[io, ifl] = G_oe(b) - G_oe(a)
    X[io, ice] = G_oe(H - a) - G_oe(H - b)
    X[ii, ifl] = G_ie(b) - G_ie(a)
    X[ii, ice] = G_ie(H - a) - G_ie(H - b)
    X[ifl, :2 * n] = X[:2 * n, ifl]
    X[ice, :2 * n] = X[:2 * n, ice]
    X[ifl, ice] = X[ice, ifl] = A[ifl] - X[ifl, :2 * n].sum()
    names = ([f'{outer_name}{k}' for k in range(n)] + [f'{inner_name}{k}' for k in range(n)]
             + [floor_name, ceil_name])
    return X / A[:, None], A, names


# ============================================================================
# Monte Carlo ray tracer for a coaxial enclosure
# ============================================================================
def _trace(px, py, pz, dx, dy, dz, R_out, r_obs, z_obs0, z_obs1, H, obs_bottom):
    """First hit. kind: 0 outer cylinder, 1 obstruction cylinder, 2 floor,
    3 ceiling, 4 obstruction bottom disk."""
    n = px.size
    a = dx * dx + dy * dy
    b = 2.0 * (px * dx + py * dy)
    a_ok = a > 1e-14
    a_safe = np.where(a_ok, a, 1.0)

    c = px * px + py * py - R_out * R_out
    disc = np.maximum(b * b - 4.0 * a * c, 0.0)
    t_out = np.where(a_ok, (-b + np.sqrt(disc)) / (2.0 * a_safe), _INF)
    t_out = np.where(t_out > 1e-9, t_out, _INF)

    c2 = px * px + py * py - r_obs * r_obs
    disc2 = b * b - 4.0 * a * c2
    ok2 = a_ok & (disc2 > 0)
    t1 = np.where(ok2, (-b - np.sqrt(np.where(ok2, disc2, 0.0))) / (2.0 * a_safe), _INF)
    zt = pz + np.where(np.isfinite(t1), t1, 0.0) * dz
    t_obs = np.where(ok2 & (t1 > 1e-7) & (zt >= z_obs0) & (zt <= z_obs1), t1, _INF)

    t_floor = np.where(dz < 0, -pz / np.where(dz < 0, dz, -1.0), _INF)
    t_ceil = np.where(dz > 0, (H - pz) / np.where(dz > 0, dz, 1.0), _INF)
    t_floor = np.where(t_floor > 1e-9, t_floor, _INF)
    t_ceil = np.where(t_ceil > 1e-9, t_ceil, _INF)

    if obs_bottom:
        tb = np.where((dz > 0) & (pz < z_obs0 - 1e-12), (z_obs0 - pz) / np.where(dz > 0, dz, 1.0), _INF)
        tbf = np.where(np.isfinite(tb), tb, 0.0)
        xb, yb = px + tbf * dx, py + tbf * dy
        t_bot = np.where(np.isfinite(tb) & (xb * xb + yb * yb <= r_obs * r_obs), tb, _INF)
    else:
        t_bot = np.full(n, _INF)

    T = np.vstack([t_out, t_obs, t_floor, t_ceil, t_bot])
    kind = np.argmin(T, axis=0)
    t = T[kind, np.arange(n)]
    if not np.all(np.isfinite(t)):
        raise RuntimeError("ray escaped the enclosure — geometry error")
    return kind, px + t * dx, py + t * dy, pz + t * dz


def _diffuse_dirs(rng, n, normal, t1, t2):
    u1, u2 = rng.random(n), rng.random(n)
    sin_t, cos_t = np.sqrt(u1), np.sqrt(1.0 - u1)
    phi = 2.0 * np.pi * u2
    a, b = sin_t * np.cos(phi), sin_t * np.sin(phi)
    d = cos_t[:, None] * normal[None, :] + a[:, None] * t1[None, :] + b[:, None] * t2[None, :]
    return d[:, 0], d[:, 1], d[:, 2]


def _sample_surface(rng, n, surf):
    if surf['type'] == 'cyl':
        z = surf['z0'] + (surf['z1'] - surf['z0']) * rng.random(n)
        px, py, pz = np.full(n, surf['r']), np.zeros(n), z
        dx, dy, dz = _diffuse_dirs(rng, n, np.array([surf['sign'], 0.0, 0.0]),
                                   np.array([0, 1.0, 0]), np.array([0, 0, 1.0]))
    else:
        ri, ro = surf['ri'], surf['ro']
        r = np.sqrt(ri ** 2 + (ro ** 2 - ri ** 2) * rng.random(n))
        px, py, pz = r, np.zeros(n), np.full(n, surf['z'])
        dx, dy, dz = _diffuse_dirs(rng, n, np.array([0.0, 0.0, surf['sign']]),
                                   np.array([1.0, 0, 0]), np.array([0, 1.0, 0]))
    return px, py, pz, dx, dy, dz


def _area(surf):
    if surf['type'] == 'cyl':
        return 2.0 * np.pi * surf['r'] * (surf['z1'] - surf['z0'])
    return np.pi * (surf['ro'] ** 2 - surf['ri'] ** 2)


def _sample_from_uniform(u, surf):
    """Emission points + cosine-weighted directions from uniforms u (n, 3)."""
    n = u.shape[0]
    if surf['type'] == 'cyl':
        z = surf['z0'] + (surf['z1'] - surf['z0']) * u[:, 0]
        p = (np.full(n, surf['r']), np.zeros(n), z)
        normal, t1, t2 = np.array([surf['sign'], 0.0, 0.0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])
    else:
        ri, ro = surf['ri'], surf['ro']
        r = np.sqrt(ri ** 2 + (ro ** 2 - ri ** 2) * u[:, 0])
        p = (r, np.zeros(n), np.full(n, surf['z']))
        normal, t1, t2 = np.array([0.0, 0.0, surf['sign']]), np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
    sin_t, cos_t = np.sqrt(u[:, 1]), np.sqrt(1.0 - u[:, 1])
    phi = 2.0 * np.pi * u[:, 2]
    a, b = sin_t * np.cos(phi), sin_t * np.sin(phi)
    d = cos_t[:, None] * normal[None, :] + a[:, None] * t1[None, :] + b[:, None] * t2[None, :]
    return p + (d[:, 0], d[:, 1], d[:, 2])


def _mc_matrix(surfs, classify, enc, n_rays, seed, chunk=262_144, sampler='sobol'):
    """Row i = fraction of rays from surface i landing on each surface.

    sampler='sobol' (default): scrambled Sobol points (quasi-Monte Carlo); the
    ray count is rounded up to a power of two. Converges ~2-3x faster than
    pseudo-random: 65,536 rays/surface gives worst-entry error ~1% for F > 0.05.
    sampler='random': plain pseudo-random sampling.
    """
    N = len(surfs)
    F = np.zeros((N, N))
    if sampler == 'sobol':
        from scipy.stats import qmc
        m = int(np.ceil(np.log2(max(n_rays, 2))))
        n_used = 2 ** m
    else:
        rng = np.random.default_rng(seed)
        n_used = n_rays
    for i, s in enumerate(surfs):
        if sampler == 'sobol':
            U = qmc.Sobol(3, scramble=True, seed=seed * 100_003 + i).random_base2(m)
        counts = np.zeros(N)
        for c0 in range(0, n_used, chunk):
            m_c = min(chunk, n_used - c0)
            u = U[c0:c0 + m_c] if sampler == 'sobol' else rng.random((m_c, 3))
            kind, xh, yh, zh = _trace(*_sample_from_uniform(u, s), **enc)
            counts += np.bincount(classify(kind, xh, yh, zh), minlength=N)
        F[i] = counts / n_used
    return F, np.array([_area(s) for s in surfs])


def enforce_reciprocity(F, A, tol=1e-12, max_iter=500):
    """Symmetric Sinkhorn scaling: returns F' with A_i F'_ij = A_j F'_ji exactly
    and rows summing to 1 (to tol). Zero entries stay zero."""
    X = A[:, None] * F
    X = 0.5 * (X + X.T)
    d = np.ones(len(A))
    for _ in range(max_iter):
        s = d * (X @ d)
        if np.max(np.abs(s / A - 1.0)) < tol:
            break
        d = d * np.sqrt(A / np.maximum(s, 1e-300))
    Xs = d[:, None] * X * d[None, :]
    return Xs / A[:, None]


def _mc_liner_enclosure(R_out, r_liner, edges, edges_act, z_act0, H_low, n_rays, seed, outer_name,
                        sampler='sobol'):
    """Enclosure between an outer full-height cylinder (rings `edges`) and the
    liner (outer wall z_act0..H, edge H_low..z_act0, pad underside at H_low)."""
    e, ea = np.asarray(edges), np.asarray(edges_act)
    n, na = len(e) - 1, len(ea) - 1
    H = e[-1]
    low = H_low > 1e-12
    surfs, names = [], []
    for k in range(n):
        surfs.append(dict(type='cyl', r=R_out, z0=e[k], z1=e[k + 1], sign=-1.0)); names.append(f'{outer_name}{k}')
    for k in range(na):
        surfs.append(dict(type='cyl', r=r_liner, z0=ea[k], z1=ea[k + 1], sign=+1.0)); names.append(f'L{k}')
    has_edge = low and (z_act0 - H_low > 1e-9)
    if has_edge:
        surfs.append(dict(type='cyl', r=r_liner, z0=H_low, z1=z_act0, sign=+1.0)); names.append('E')
    if low:
        surfs.append(dict(type='ann', ri=0.0, ro=r_liner, z=H_low, sign=-1.0)); names.append('R')
    surfs.append(dict(type='ann', ri=0.0 if low else r_liner, ro=R_out, z=0.0, sign=+1.0)); names.append('floor')
    surfs.append(dict(type='ann', ri=r_liner, ro=R_out, z=H, sign=-1.0)); names.append('ceil')
    ix = {nm: i for i, nm in enumerate(names)}

    def classify(kind, x, y, z):
        ring = np.clip(np.searchsorted(e, z, side='right') - 1, 0, n - 1)
        lring = np.clip(np.searchsorted(ea, z, side='right') - 1, 0, na - 1)
        out = np.empty(kind.size, dtype=int)
        m0 = kind == 0
        out[m0] = ring[m0]
        m1 = kind == 1
        on_l = m1 & (z >= z_act0)
        out[on_l] = n + lring[on_l]
        if has_edge:
            out[m1 & (z < z_act0)] = ix['E']
        if low:
            out[kind == 4] = ix['R']
        out[kind == 2] = ix['floor']
        out[kind == 3] = ix['ceil']
        return out

    enc = dict(R_out=R_out, r_obs=r_liner, z_obs0=H_low if low else 0.0, z_obs1=H, H=H, obs_bottom=low)
    F, A = _mc_matrix(surfs, classify, enc, n_rays, seed, sampler=sampler)
    return F, A, names


# ============================================================================
# Public API
# ============================================================================
def axial_mesh(L_ps, H_block, n_ax):
    """Axial ring mesh, z = 0 at the cavity floor (bottom-up).

    n_ax (YAML storage.num_ax_segments) is the TOTAL number of block rings over
    H_block. When the block is taller than the powerstick, the rings are split
    between the lower zone and the active zone (L_ps, top-aligned) in proportion
    to height, with a ring edge exactly at the liner-bottom plane z_act0, so ring
    heights differ slightly between the zones. The powerstick gets one axial
    step per active ring.

    Returns (n_act, n_low, edges, edges_act, z_act0).
    """
    n_ax = int(n_ax)
    z_act0 = H_block - L_ps
    if z_act0 <= 1e-12:
        edges = np.linspace(0.0, H_block, n_ax + 1)
        return n_ax, 0, edges, edges, 0.0
    if n_ax < 2:
        raise ValueError("num_ax_segments must be >= 2 when block_height > L")
    n_act = min(max(int(round(n_ax * L_ps / H_block)), 1), n_ax - 1)
    n_low = n_ax - n_act
    edges_act = np.linspace(z_act0, H_block, n_act + 1)
    edges = np.concatenate([np.linspace(0.0, z_act0, n_low + 1)[:-1], edges_act])
    return n_act, n_low, edges, edges_act, z_act0


def cavity_geometry(r_in, r_sh, r_liner, L_ps, H_block, t_lb, t_rgf, n_ax):
    """Geometry dict for cavity_view_factors. n_ax = total block rings (see axial_mesh)."""
    has_low = H_block - L_ps > 1e-12
    H_low = H_block - L_ps - t_lb - t_rgf if has_low else 0.0
    if H_low < 0:
        raise ValueError(f"H_low = {H_low:.4f} m < 0: block too short for liner + bottom pad")
    n_act, n_low, edges, edges_act, z_act0 = axial_mesh(L_ps, H_block, n_ax)
    return dict(r_in=r_in, r_sh=r_sh, r_liner=r_liner, L_ps=L_ps, H=H_block, t_lb=t_lb, t_rgf=t_rgf,
                H_low=H_low, z_act0=z_act0, edges=edges, edges_act=edges_act, n_act=n_act, n_low=n_low)


def geometry_from_build(m):
    """Geometry dict from a thermal.build.build_tinless() result."""
    r_sh = 2 * m.shade_side_len / np.pi if (m.shade_bool and m.shade_side_len) else None
    me = m.mesh
    return dict(r_in=m.r_in, r_sh=r_sh, r_liner=m.r_liner_eff, L_ps=m.L, H=m.H_block,
                t_lb=me.t_lb, t_rgf=me.t_rgf, H_low=me.H_low, z_act0=me.z_act0,
                edges=me.edges, edges_act=me.edges_act, n_act=me.n_act, n_low=me.n_low)


def _enc(F, A, names, method):
    return SimpleNamespace(F=F, A=A, names=list(names), idx={n: i for i, n in enumerate(names)}, method=method)


def _cache_key(g, shaded, method, n_rays, seed, sampler='sobol'):
    parts = [f"{g['r_in']:.6f}", f"{g['r_sh'] or 0:.6f}", f"{g['r_liner']:.6f}", f"{g['H']:.6f}",
             f"{g['z_act0']:.6f}", f"{g['H_low']:.6f}", ",".join(f"{x:.6f}" for x in g['edges']),
             ",".join(f"{x:.6f}" for x in g['edges_act']), str(shaded), method, str(n_rays), str(seed), sampler, "v2"]
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


# Default ray count: accuracy target 2% on every view factor > 0.05 (speed
# prioritised for the optimizer). Scrambled Sobol at 2**16 rays/surface gives
# ~1% worst-entry error on a 62-surface enclosure (measured vs analytic).
DEFAULT_RAYS = 65_536


def cavity_view_factors(g, shaded=True, method='analytic', n_rays=DEFAULT_RAYS, seed=12345,
                        cache_dir=None, sampler='sobol'):
    """View factors for the emitter cavity.

    shaded=True  -> {'outer': enc, 'inner': enc}
    shaded=False -> {'single': enc}
    Each enc has F (rows sum to 1, A_i F_ij = A_j F_ji), A, names, idx, method.

    method='analytic': exact relations wherever both cylinders span the full
    height (the outer enclosure always; the inner/unshaded one when there is
    no lower zone); Monte Carlo otherwise. method='montecarlo': Monte Carlo
    everywhere (cross-check).
    """
    if cache_dir:
        key = _cache_key(g, shaded, method, n_rays, seed, sampler)
        path = os.path.join(cache_dir, f"vf_{key}.npz")
        if os.path.exists(path):
            d = np.load(path, allow_pickle=False)
            encs = {}
            for nm in ('outer', 'inner', 'single'):
                if f'{nm}_F' in d.files:
                    encs[nm] = _enc(d[f'{nm}_F'], d[f'{nm}_A'], [str(x) for x in d[f'{nm}_names']],
                                    str(d[f'{nm}_method']))
            return encs

    e, ea = g['edges'], g['edges_act']
    low = g['H_low'] > 1e-12
    mc = method == 'montecarlo'
    encs = {}

    def liner_side(R_out, outer_name, seed_off):
        if not low and not mc:
            F, A, nm = _analytic_annulus_enclosure(g['r_liner'], R_out, e, outer_name, 'L', 'floor', 'ceil')
            return _enc(F, A, nm, 'analytic')
        F, A, nm = _mc_liner_enclosure(R_out, g['r_liner'], e, ea, g['z_act0'], g['H_low'],
                                       n_rays, seed + seed_off, outer_name, sampler)
        return _enc(enforce_reciprocity(F, A), A, nm, 'montecarlo')

    if shaded:
        if g['r_sh'] is None:
            raise ValueError("shaded=True needs r_sh")
        if mc:
            F, A, nm = _mc_liner_enclosure(g['r_in'], g['r_sh'], e, e, 0.0, 0.0, n_rays, seed, 'B', sampler)
            nm = [x.replace('L', 'So') if x.startswith('L') else x for x in nm]
            nm = [{'floor': 'floor_out', 'ceil': 'ceil_out'}.get(x, x) for x in nm]
            encs['outer'] = _enc(enforce_reciprocity(F, A), A, nm, 'montecarlo')
        else:
            F, A, nm = _analytic_annulus_enclosure(g['r_sh'], g['r_in'], e, 'B', 'So', 'floor_out', 'ceil_out')
            encs['outer'] = _enc(F, A, nm, 'analytic')
        inner = liner_side(g['r_sh'], 'Si', 1)
        inner.names = [{'floor': 'floor_in', 'ceil': 'ceil_in'}.get(x, x) for x in inner.names]
        inner.idx = {n: i for i, n in enumerate(inner.names)}
        encs['inner'] = inner
    else:
        encs['single'] = liner_side(g['r_in'], 'B', 2)

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        out = {}
        for nm, en in encs.items():
            out[f'{nm}_F'], out[f'{nm}_A'] = en.F, en.A
            out[f'{nm}_names'] = np.array(en.names)
            out[f'{nm}_method'] = np.array(en.method)
        np.savez_compressed(path, **out)
    return encs


def reciprocity_error(F, A):
    """Max |A_i F_ij - A_j F_ji| / max(...) over pairs with F > 1e-3."""
    X = A[:, None] * F
    M = np.maximum(X, X.T)
    mask = (F > 1e-3) | (F.T > 1e-3)
    return float(np.max(np.abs(X - X.T)[mask] / M[mask])) if mask.any() else 0.0


# ============================================================================
# Shade as a porous sheet with straight-through transmission
# ============================================================================
def _porous_surfaces(g):
    """Opaque surfaces of the merged cavity with the shade as a porous sheet.
    Slat faces So_k / Si_k are defined per ring with the FULL ring area; their
    emission/absorption is per unit slat area (area f * A_ring at runtime)."""
    e, ea = np.asarray(g['edges']), np.asarray(g['edges_act'])
    n, na = len(e) - 1, len(ea) - 1
    H, r_in, r_sh, r_l = e[-1], g['r_in'], g['r_sh'], g['r_liner']
    H_low, z_act0 = g['H_low'], g['z_act0']
    low = H_low > 1e-12
    S, names, slat = [], [], []
    for k in range(n):
        S.append(dict(type='cyl', r=r_in, z0=e[k], z1=e[k + 1], sign=-1.0)); names.append(f'B{k}'); slat.append(0)
    for k in range(n):
        S.append(dict(type='cyl', r=r_sh, z0=e[k], z1=e[k + 1], sign=+1.0)); names.append(f'So{k}'); slat.append(1)
    for k in range(n):
        S.append(dict(type='cyl', r=r_sh, z0=e[k], z1=e[k + 1], sign=-1.0)); names.append(f'Si{k}'); slat.append(1)
    for k in range(na):
        S.append(dict(type='cyl', r=r_l, z0=ea[k], z1=ea[k + 1], sign=+1.0)); names.append(f'L{k}'); slat.append(0)
    has_edge = low and (z_act0 - H_low > 1e-9)
    if has_edge:
        S.append(dict(type='cyl', r=r_l, z0=H_low, z1=z_act0, sign=+1.0)); names.append('E'); slat.append(0)
    if low:
        S.append(dict(type='ann', ri=0.0, ro=r_l, z=H_low, sign=-1.0)); names.append('R'); slat.append(0)
    S.append(dict(type='ann', ri=r_sh, ro=r_in, z=0.0, sign=+1.0)); names.append('floor_out'); slat.append(0)
    S.append(dict(type='ann', ri=0.0 if low else r_l, ro=r_sh, z=0.0, sign=+1.0)); names.append('floor_in'); slat.append(0)
    S.append(dict(type='ann', ri=r_sh, ro=r_in, z=H, sign=-1.0)); names.append('ceil_out'); slat.append(0)
    S.append(dict(type='ann', ri=r_l, ro=r_sh, z=H, sign=-1.0)); names.append('ceil_in'); slat.append(0)
    return S, names, np.array(slat, bool), has_edge, low


def _mc_porous(g, n_rays, seed, sampler='sobol', chunk=262_144):
    """Ray-trace the merged cavity; the shade cylinder is crossed, not hit.
    Returns K (3, N, N): F(tau) = K0 + tau K1 + tau^2 K2 with tau = open fraction.
    A crossing hits a slat with prob f = 1 - tau (outer face when moving inward,
    inner face when moving outward); a line crosses the cylinder at most twice."""
    S, names, slat, has_edge, low = _porous_surfaces(g)
    e, ea = np.asarray(g['edges']), np.asarray(g['edges_act'])
    n, na = len(e) - 1, len(ea) - 1
    H, r_sh = e[-1], g['r_sh']
    ix = {nm: i for i, nm in enumerate(names)}
    N = len(S)
    enc = dict(R_out=g['r_in'], r_obs=g['r_liner'], z_obs0=g['H_low'] if low else 0.0,
               z_obs1=H, H=H, obs_bottom=low)
    iSo, iSi = ix['So0'], ix['Si0']

    def end_surface(kind, x, y, z):
        ring = np.clip(np.searchsorted(e, z, side='right') - 1, 0, n - 1)
        lring = np.clip(np.searchsorted(ea, z, side='right') - 1, 0, na - 1)
        rr = x * x + y * y
        out = np.empty(kind.size, dtype=int)
        out[kind == 0] = ring[kind == 0]
        m1 = kind == 1
        on_l = m1 & (z >= g['z_act0'] - 1e-12)
        out[on_l] = 3 * n + lring[on_l]
        if has_edge:
            out[m1 & ~on_l] = ix['E']
        if low:
            out[kind == 4] = ix['R']
        inside = rr < r_sh * r_sh
        out[(kind == 2) & ~inside] = ix['floor_out']
        out[(kind == 2) & inside] = ix['floor_in']
        out[(kind == 3) & ~inside] = ix['ceil_out']
        out[(kind == 3) & inside] = ix['ceil_in']
        return out

    if sampler == 'sobol':
        from scipy.stats import qmc
        m = int(np.ceil(np.log2(max(n_rays, 2))))
        n_used = 2 ** m
    else:
        rng = np.random.default_rng(seed)
        n_used = n_rays
    K = np.zeros((3, N, N))
    for i, s in enumerate(S):
        if sampler == 'sobol':
            U = qmc.Sobol(3, scramble=True, seed=seed * 100_003 + i).random_base2(m)
        acc = np.zeros((3, N))
        for c0 in range(0, n_used, chunk):
            mc_ = min(chunk, n_used - c0)
            u = U[c0:c0 + mc_] if sampler == 'sobol' else rng.random((mc_, 3))
            px, py, pz, dx, dy, dz = _sample_from_uniform(u, s)
            kind, xh, yh, zh = _trace(px, py, pz, dx, dy, dz, **enc)
            t_hit = np.sqrt((xh - px) ** 2 + (yh - py) ** 2 + (zh - pz) ** 2)   # |d| = 1
            # crossings of r = r_sh strictly between emission and hit
            a = dx * dx + dy * dy
            b = 2.0 * (px * dx + py * dy)
            c = px * px + py * py - r_sh * r_sh
            disc = b * b - 4 * a * c
            ok = (a > 1e-14) & (disc > 0)
            sq = np.sqrt(np.where(ok, disc, 0.0))
            a_s = np.where(ok, a, 1.0)
            roots = np.stack([(-b - sq) / (2 * a_s), (-b + sq) / (2 * a_s)])
            valid = ok[None, :] & (roots > 1e-7) & (roots < t_hit[None, :] - 1e-7)
            ncross = valid.sum(0)
            jend = end_surface(kind, xh, yh, zh)
            for p in range(3):
                acc[p] += np.bincount(jend[ncross == p], minlength=N)
            for r in range(2):                     # roots sorted: r = 0 comes first
                v = valid[r]
                if not v.any():
                    continue
                t = roots[r, v]
                zc = pz[v] + t * dz[v]
                xc, yc = px[v] + t * dx[v], py[v] + t * dy[v]
                inward = (xc * dx[v] + yc * dy[v]) < 0
                kr = np.clip(np.searchsorted(e, zc, side='right') - 1, 0, n - 1)
                tgt = np.where(inward, iSo + kr, iSi + kr)
                first = ~valid[0, v] if r == 1 else np.ones(int(v.sum()), bool)
                c1 = np.bincount(tgt[first], minlength=N)
                c2 = np.bincount(tgt[~first], minlength=N)
                acc[0] += c1; acc[1] -= c1           # (1 - tau) at the first crossing
                acc[1] += c2; acc[2] -= c2           # tau (1 - tau) at the second
        K[:, i, :] = acc / n_used
    A = np.array([_area(s) for s in S])
    return K, A, names, slat


def _porous_symmetrise(K, A_ring, slat):
    """Exchange-area form X(tau) = A(tau) F(tau), A = A_ring (1 - tau) for slats.
    Symmetrise each tau-power coefficient and close row sums on the diagonal,
    so reciprocity and energy conservation hold exactly for every tau. Returns
    C (4, N, N) with F(tau) = sum_p tau^p C_p (slat rows divided by 1 - tau)."""
    N = K.shape[1]
    X = np.zeros((4, N, N))
    ns = ~slat
    X[:3, ns] = A_ring[None, ns, None] * K[:, ns]
    Ar = A_ring[slat, None]
    X[0, slat] = Ar * K[0, slat]
    X[1, slat] = Ar * (K[1, slat] - K[0, slat])
    X[2, slat] = Ar * (K[2, slat] - K[1, slat])
    X[3, slat] = -Ar * K[2, slat]
    X = 0.5 * (X + np.transpose(X, (0, 2, 1)))
    a = np.zeros((4, N))
    a[0] = A_ring
    a[1, slat] = -A_ring[slat]
    resid = a - X.sum(2)
    for p in range(4):
        X[p][np.diag_indices(N)] += resid[p]
    C = np.zeros_like(X)
    C[:, ns] = X[:, ns] / A_ring[None, ns, None]
    # slat rows: X(tau) = (1 - tau) Y(tau)  ->  synthetic division by (1 - tau)
    Y0 = X[0, slat]
    Y1 = X[1, slat] + Y0
    Y2 = X[2, slat] + Y1
    rem = X[3, slat] + Y2
    if np.max(np.abs(rem)) > 1e-9 * np.max(np.abs(X)):
        raise RuntimeError("porous shade exchange: slat rows not divisible by (1 - tau)")
    C[0, slat], C[1, slat], C[2, slat] = Y0 / Ar, Y1 / Ar, Y2 / Ar
    return C


def porous_shade_view_factors(g, n_rays=DEFAULT_RAYS, seed=12345, cache_dir=None, sampler='sobol'):
    """Merged-cavity view factors with the shade as a porous sheet (straight-through
    transmission, open fraction tau). Returns SimpleNamespace with C (4, N, N)
    such that F(tau) = sum_p tau^p C[p] (rows per unit area of the emitter; slat
    faces per unit slat area), A_ring (full ring areas), slat mask, names, idx.
    Limits: tau = 1 -> unshaded single enclosure; tau = 0 -> the two separate
    enclosures (block-shade gap, shade-liner gap)."""
    if g.get('r_sh') is None:
        raise ValueError("porous shade view factors need r_sh")
    path = None
    if cache_dir:
        key = _cache_key(g, 'porous', 'montecarlo', n_rays, seed, sampler)
        path = os.path.join(cache_dir, f"vfp_{key}.npz")
        if os.path.exists(path):
            d = np.load(path, allow_pickle=False)
            names = [str(x) for x in d['names']]
            return SimpleNamespace(C=d['C'], A_ring=d['A'], slat=d['slat'], names=names,
                                   idx={nm: i for i, nm in enumerate(names)}, method='montecarlo-porous')
    K, A, names, slat = _mc_porous(g, n_rays, seed, sampler)
    C = _porous_symmetrise(K, A, slat)
    if path:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez_compressed(path, C=C, A=A, slat=slat, names=np.array(names))
    return SimpleNamespace(C=C, A_ring=A, slat=slat, names=list(names),
                           idx={nm: i for i, nm in enumerate(names)}, method='montecarlo-porous')


def porous_F(vp, tau):
    """F(tau) and dF/dtau for a porous_shade_view_factors result."""
    C = vp.C
    F = C[0] + tau * (C[1] + tau * (C[2] + tau * C[3]))
    dF = C[1] + tau * (2 * C[2] + 3 * tau * C[3])
    return F, dF
