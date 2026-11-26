"""
Hot-zone layouts: how powersticks and heaters are arranged in the graphite matrix.

Every stick must have its full ring of heater neighbours (perimeter heaters required), so the
plant has N_h != N_ps heaters in general. Four patterns (unit stick-heater spacing d = 1):

  checkerboard  sticks and heaters alternate on a square grid; 4 heaters around each stick,
                4 sticks around each heater; bulk 1 heater per stick
  hex6          sticks on a triangular lattice, heaters at the triangle centres; 6 heaters around
                each stick, 3 sticks around each heater; bulk 2 heaters per stick
  honeycomb     sticks on one honeycomb sublattice, heaters on the other; 3 + 3; bulk 1:1
  hex3          heaters on a triangular lattice, sticks at the triangle centres; 3 heaters around
                each stick, 6 sticks around each heater; bulk 1 heater per 2 sticks

Stick counts (plan(pattern, N_req)):
  N_req < SMALL_N (20): the smallest "Tier A" count >= N_req: clusters with every stick fully
      surrounded, no empty slot enclosed by the heaters, every slot inside the cluster's convex
      hull filled, and at least three mirror axes (D3 / D4 / D6, round). Tier A clusters are
      enumerated once (enumerate_tier_a, saved in layout_tier_a.json) up to TIER_A_NMAX sticks.
  N_req >= SMALL_N: N = N_req, a round cluster: the N stick sites nearest a high-symmetry centre
      (the centre giving the fewest heaters), heaters = every heater site next to a stick.

Geometry (spacing d): the plan area per stick is the stick cell (pi r_out^2, its graphite and
cavity) plus its bulk share of heater cavities: A_s = pi r_out^2 + rho_bulk pi r_hcav^2 = k d^2 with
k the pattern's lattice area per stick at unit spacing (checkerboard 2: d = the old pitch).
Hot zone: the circle about the cluster centroid through the outermost cavity centres plus d / 2.
"""
import itertools
import json
import math
import os

import numpy as np

S3 = math.sqrt(3.0)
SMALL_N = 20
TIER_A_NMAX = 40
_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'layout_tier_a.json')

PATTERNS = {
    'checkerboard': dict(rho=1.0, k=2.0, around_stick=4, around_heater=4, stick_rot=45.0,
                         desc='checkerboard: 4 heaters around each stick, 1 heater per stick in bulk'),
    'hex6': dict(rho=2.0, k=1.5 * S3, around_stick=6, around_heater=3, stick_rot=None,
                 desc='hexagonal: 6 heaters around each stick, 2 heaters per stick in bulk'),
    'honeycomb': dict(rho=1.0, k=1.5 * S3, around_stick=3, around_heater=3, stick_rot=None,
                      desc='honeycomb: 3 heaters around each stick, 3 sticks around each heater, 1:1 in bulk'),
    'hex3': dict(rho=0.5, k=0.75 * S3, around_stick=3, around_heater=6, stick_rot=None,
                 desc='hexagonal: 3 heaters around each stick, 6 sticks around each heater, 1 heater per 2 sticks in bulk'),
}


def pattern(name):
    n = str(name or 'checkerboard').lower()
    if n not in PATTERNS:
        raise ValueError(f"layout pattern must be one of {sorted(PATTERNS)}, not {name!r}")
    return n


# --------------------------------------------------------------------------
# lattices (unit stick-heater spacing)
# --------------------------------------------------------------------------
def lattice(name, R=12.0):
    """Stick and heater sites within radius R, and extra high-symmetry points (symmetry centres).
    -> (sticks (n, 2), heaters (m, 2), extra (k, 2), point-group order 4 or 6)"""
    name = pattern(name)
    S, H, X = [], [], []
    n = int(R) + 3
    if name == 'checkerboard':
        for x in range(-n, n + 1):
            for y in range(-n, n + 1):
                (S if (x + y) % 2 == 0 else H).append((x, y))
                X.append((x + .5, y + .5))
        order = 4
    else:
        a1, a2 = np.array([S3, 0.0]), np.array([S3 / 2, 1.5])      # triangular lattice, spacing sqrt(3)
        V, U, D = [], [], []
        for i in range(-n, n + 1):
            for j in range(-n, n + 1):
                p = i * a1 + j * a2
                V.append(tuple(p)); U.append(tuple(p + (a1 + a2) / 3)); D.append(tuple(p + 2 * (a1 + a2) / 3))
        S, H = {'hex6': (V, U + D), 'honeycomb': (U, D), 'hex3': (U + D, V)}[name]
        X = V + U + D
        order = 6
    S, H, X = (np.array(v, float) for v in (S, H, X))
    return S[np.hypot(*S.T) <= R], H[np.hypot(*H.T) <= R + 1.5], X[np.hypot(*X.T) <= 2], order


def _key(p):
    return (round(float(p[0]) * 1e6), round(float(p[1]) * 1e6))


def _neighbours(S, H):
    """Heater neighbours (unit distance) of each stick -> list of frozensets of heater indices."""
    out = []
    for p in S:
        d = np.hypot(*(H - p).T)
        out.append(frozenset(np.nonzero(np.abs(d - 1.0) < 1e-6)[0]))
    return out


def counts_bulk(name):
    """Bulk heaters per stick and the number of heaters around each stick / sticks around each heater."""
    p = PATTERNS[pattern(name)]
    return p['rho'], p['around_stick'], p['around_heater']


# --------------------------------------------------------------------------
# Tier A clusters (symmetric, compact) -- enumerated once
# --------------------------------------------------------------------------
def _convex_ok(S, idx, others):
    from scipy.spatial import ConvexHull
    P = S[idx]
    if len(idx) < 3 or np.linalg.matrix_rank(P - P[0]) < 2:
        return len(idx) <= 2
    hull = ConvexHull(P)
    Q = S[others]
    return not np.all(Q @ hull.equations[:, :2].T + hull.equations[:, 2] <= 1e-7, axis=1).any()


def enumerate_tier_a(name, n_max=TIER_A_NMAX, R=9.0):
    """All Tier A clusters of `name` up to n_max sticks (>= 3 mirror axes about a high-symmetry
    centre; every stick fully surrounded; no enclosed empty slot; convex). For each N the one
    with the fewest heaters. -> {N: dict(sticks, heaters (unit-spacing xy lists), group)}"""
    S, H, X, order = lattice(name, R)
    nbh = _neighbours(S, H)
    full = max(len(s) for s in nbh)
    inner = [i for i, p in enumerate(S) if np.hypot(*p) <= R - 2 and len(nbh[i]) == full]
    inner_set = set(inner)
    sidx = {_key(p): i for i, p in enumerate(S)}
    h2s = {}
    for i in inner:
        for h in nbh[i]:
            h2s.setdefault(h, set()).add(i)
    share = {i: set() for i in inner}
    for ss in h2s.values():
        for i in ss:
            share[i] |= ss - {i}
    # symmetry centres: sites and midpoints near the origin
    pts = np.vstack([S[np.hypot(*S.T) <= 1.1], H[np.hypot(*H.T) <= 1.1], X[np.hypot(*X.T) <= 1.1]])
    cands = list(pts) + [(a + b) / 2 for a, b in itertools.combinations(pts, 2) if np.hypot(*(a - b)) <= 1.8]
    centres, seen = [], set()
    for c in cands:
        if np.hypot(*c) <= 1.1 and _key(c) not in seen:
            seen.add(_key(c)); centres.append(np.asarray(c, float))
    ops = []
    for k in range(order):
        t = 2 * math.pi * k / order
        ops.append(('r', np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])))
        ops.append(('m', np.array([[math.cos(t), math.sin(t)], [math.sin(t), -math.cos(t)]])))
    best = {}
    for c in centres:
        near_s = S[np.hypot(*(S - c).T) <= 3]
        near_h = H[np.hypot(*(H - c).T) <= 3]
        hidx = {_key(p) for p in H}
        valid = [M for kind, M in ops
                 if all(_key(M @ (p - c) + c) in sidx for p in near_s) and all(_key(M @ (p - c) + c) in hidx for p in near_h)]
        mirrors = [M for M in valid if np.linalg.det(M) < 0]
        groups = {}                                  # every subgroup generated by two mirrors, >= 3 mirrors
        for m1, m2 in itertools.combinations(mirrors, 2):
            Gs = [np.eye(2), m1, m2]
            grew = True
            while grew:
                grew = False
                for A, B in itertools.product(list(Gs), repeat=2):
                    C = A @ B
                    if not any(np.allclose(C, X) for X in Gs):
                        Gs.append(C); grew = True
            if sum(1 for X in Gs if np.linalg.det(X) < 0) >= 3:
                groups[tuple(sorted(tuple(np.round(X.ravel(), 6)) for X in Gs))] = Gs
        for G in groups.values():
            _grow(G, c, S, H, nbh, inner, inner_set, sidx, share, n_max, best)
    return dict(sorted(best.items()))


def _grow(G, c, S, H, nbh, inner, inner_set, sidx, share, n_max, best):
    """Grow every connected cluster invariant under group G (about c) orbit by orbit; keep the
    Tier A one with the fewest heaters per stick count in `best`."""
    gname = f"D{sum(1 for M in G if np.linalg.det(M) < 0)}"
    orb_of, orbits = {}, []
    for i in sorted(inner, key=lambda i: np.hypot(*(S[i] - c))):
        if i in orb_of:
            continue
        o = set()
        for M in G:
            j = sidx.get(_key(M @ (S[i] - c) + c))
            if j is None or j not in inner_set:
                o = None
                break
            o.add(j)
        if o is None:
            continue
        for j in o:
            orb_of[j] = len(orbits)
        orbits.append(frozenset(o))
    if not orbits:
        return
    dmin = min(min(np.hypot(*(S[j] - c)) for j in o) for o in orbits)
    stack = [frozenset([k]) for k, o in enumerate(orbits)
             if min(np.hypot(*(S[j] - c)) for j in o) <= dmin + 1e-6 and len(o) <= n_max]
    visited = set(stack)
    while stack:
        stt = stack.pop()
        sticks = frozenset().union(*(orbits[k] for k in stt))
        n = len(sticks)
        hs = frozenset().union(*(nbh[j] for j in sticks))
        comp, todo = {next(iter(sticks))}, [next(iter(sticks))]
        while todo:
            a = todo.pop()
            for b in share[a] & sticks:
                if b not in comp:
                    comp.add(b); todo.append(b)
        if len(comp) == n and (n not in best or len(hs) < len(best[n]['heaters'])):
            others = [j for j in inner if j not in sticks]
            if not any(nbh[j] <= hs for j in others) and _convex_ok(S, list(sticks), others):
                best[n] = dict(sticks=[list(map(float, S[j])) for j in sorted(sticks)],
                               heaters=[list(map(float, H[h])) for h in sorted(hs)], group=gname)
        adj = set()
        for j in sticks:
            adj |= share[j]
        for j in adj - sticks:
            k = orb_of.get(j)
            if k is None or k in stt or n + len(orbits[k]) > n_max:
                continue
            ns = stt | {k}
            if ns not in visited:
                visited.add(ns); stack.append(ns)


def build_tier_a_file(n_max=TIER_A_NMAX, path=_DATA):
    """Enumerate the Tier A clusters of every pattern and save them (run once; minutes)."""
    data = {name: {str(n): v for n, v in enumerate_tier_a(name, n_max).items()} for name in PATTERNS}
    with open(path, 'w') as fh:
        json.dump(data, fh)
    return data


_TIER_A = None


def tier_a(name):
    """{N: dict(sticks, heaters, group)} of the Tier A clusters of a pattern (from layout_tier_a.json)."""
    global _TIER_A
    if _TIER_A is None:
        with open(_DATA) as fh:
            _TIER_A = {k: {int(n): v for n, v in d.items()} for k, d in json.load(fh).items()}
    return _TIER_A[pattern(name)]


def feasible_counts(name, n_max=SMALL_N - 1):
    """Tier A stick counts (and their heater counts) up to n_max. -> {N: N_h}"""
    return {n: len(v['heaters']) for n, v in tier_a(name).items() if n <= n_max}


# --------------------------------------------------------------------------
# round clusters (N >= SMALL_N)
# --------------------------------------------------------------------------
def round_cluster(name, N):
    """N stick sites nearest a high-symmetry centre (the centre with the fewest heaters) and every
    heater next to them. -> dict(sticks, heaters, group='round')"""
    R = max(6.0, 1.6 * math.sqrt(N * PATTERNS[pattern(name)]['k'] / math.pi) + 4)
    S, H, X, _ = lattice(name, R)
    nbh = _neighbours(S, H)
    full = max(len(s) for s in nbh)
    ok = np.array([len(s) == full for s in nbh])
    cands = list(S[np.hypot(*S.T) <= 1.1]) + list(H[np.hypot(*H.T) <= 1.1]) + list(X[np.hypot(*X.T) <= 1.1])
    best = None
    for c in cands:
        d = np.hypot(*(S - c).T) + np.where(ok, 0.0, 1e9)
        idx = np.argsort(d, kind='stable')[:N]
        hs = frozenset().union(*(nbh[j] for j in idx))
        if best is None or len(hs) < len(best[1]):
            best = (idx, hs)
    idx, hs = best
    return dict(sticks=S[sorted(idx)].tolist(), heaters=H[sorted(hs)].tolist(), group='round')


# --------------------------------------------------------------------------
# plant layout
# --------------------------------------------------------------------------
def plan(name, N_req):
    """Stick count and cluster for at least N_req sticks. -> dict(pattern, N, N_h, sticks, heaters
    (unit-spacing xy, centred on the cluster centroid), group, tier ('A' or 'round'))"""
    name = pattern(name)
    N_req = max(int(N_req), 1)
    if N_req < SMALL_N:
        ta = tier_a(name)
        n = min((n for n in ta if n >= N_req), default=None)
        if n is None:
            raise ValueError(f"no Tier A {name} cluster with >= {N_req} sticks in layout_tier_a.json")
        cl, tier = ta[n], 'A'
    else:
        n, cl, tier = N_req, round_cluster(name, N_req), 'round'
    st, he = np.array(cl['sticks'], float), np.array(cl['heaters'], float)
    c = np.vstack([st, he]).mean(0)
    return dict(pattern=name, N=n, N_h=len(he), sticks=(st - c).tolist(), heaters=(he - c).tolist(),
                group=cl['group'], tier=tier)


def spacing(name, r_out, r_hcav):
    """Stick-heater centre spacing d [m]: plan area per stick pi r_out^2 + rho pi r_hcav^2 = k d^2."""
    p = PATTERNS[pattern(name)]
    return math.sqrt((math.pi * r_out ** 2 + p['rho'] * math.pi * r_hcav ** 2) / p['k'])


def footprint(lay, d):
    """Hot-zone diameter [m]: circle about the cluster centroid through the outermost cavity centres + d / 2."""
    P = np.vstack([np.array(lay['sticks']), np.array(lay['heaters'])])
    return 2.0 * (float(np.hypot(*P.T).max()) * d + 0.5 * d)


# --------------------------------------------------------------------------
# graphite walls between cavities
# --------------------------------------------------------------------------
def _square(S, rot_deg):
    t = math.radians(rot_deg)
    R = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    return (0.5 * S * np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]], float)) @ R.T


def _pt_seg(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0.0, 1.0)
    return float(np.hypot(*(a + t * ab - p)))


def _pt_poly(p, P):
    """Distance from point p to convex polygon P (0 inside)."""
    inside = all(np.cross(b - a, p - a) >= 0 for a, b in zip(P, np.roll(P, -1, axis=0)))
    return 0.0 if inside else min(_pt_seg(p, a, b) for a, b in zip(P, np.roll(P, -1, axis=0)))


def _poly_poly(P, Q):
    d1 = min(_pt_poly(q, P) for q in Q)
    d2 = min(_pt_poly(p, Q) for p in P)
    return min(d1, d2)


def _nearest_vectors(name):
    """Unit-spacing vectors from a stick to its nearest heaters / sticks and from a heater to its
    nearest heaters (bulk lattice)."""
    S, H, _, _ = lattice(name, 6.0)
    s0 = S[np.argmin(np.hypot(*S.T))]
    h0 = H[np.argmin(np.hypot(*H.T))]

    def near(P, p0):
        d = np.hypot(*(P - p0).T)
        d[d < 1e-9] = np.inf
        return P[np.abs(d - d.min()) < 1e-6] - p0
    return near(H, s0), near(S, s0), near(H, h0)


def webs(name, S_cav, r_hcav, d, rot=None):
    """Graphite wall thickness [m] between neighbouring cavities: square stick cavity of side S_cav
    turned `rot` deg (None: the pattern's default; for the hexagonal patterns the rotation with the
    thickest thinnest wall), round heater cavity r_hcav, stick-heater spacing d.
    -> (rot used, {'stick-heater', 'stick-stick', 'heater-heater'})"""
    name = pattern(name)
    vsh, vss, vhh = (v * d for v in _nearest_vectors(name))

    def walls(r):
        sq = _square(S_cav, r)
        sh = min(_pt_poly(v, sq) for v in vsh) - r_hcav
        ss = min(_poly_poly(sq, sq + v) for v in vss)
        hh = float(np.hypot(*vhh.T).min()) - 2 * r_hcav
        return {'stick-heater': sh, 'stick-stick': ss, 'heater-heater': hh}
    if rot is None:
        rot = PATTERNS[name]['stick_rot']
    if rot is None:
        grid = np.arange(0.0, 90.0, 0.5)
        rot = float(max(grid, key=lambda r: min(walls(r).values())))
    return rot, walls(rot)


# --------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------
def table(n_max=SMALL_N - 1):
    """Text table of the feasible (Tier A) stick counts and heater counts per pattern."""
    L = [f"Feasible stick counts below {n_max + 1} (heaters): every stick fully surrounded, symmetric (>= 3 mirror "
         f"axes), compact; from {n_max + 1} sticks any count (round cluster)"]
    for name, p in PATTERNS.items():
        fc = feasible_counts(name, n_max)
        L.append(f"  {name:13s} bulk {p['rho']:g} heater(s) per stick: " + ', '.join(f"{n} ({h})" for n, h in fc.items()))
    return '\n'.join(L)


def plot_patterns(names=None, Ns=None, axes=None):
    """Tier A clusters of each pattern (blue squares sticks, red circles heaters). -> fig"""
    import matplotlib.pyplot as plt
    names = list(names or PATTERNS)
    rows = []
    for nm in names:
        fc = feasible_counts(nm)
        rows.append((nm, [n for n in (Ns or fc) if n in fc]))
    ncol = max(len(r[1]) for r in rows)
    fig, axs = plt.subplots(len(rows), ncol, figsize=(1.9 * ncol, 2.1 * len(rows)), squeeze=False)
    for (nm, ns), arow in zip(rows, axs):
        for ax in arow:
            ax.axis('off')
        for ax, n in zip(arow, ns):
            lay = plan(nm, n)
            st, he = np.array(lay['sticks']), np.array(lay['heaters'])
            ax.scatter(he[:, 0], he[:, 1], s=14, c='#d43c2c', marker='o')
            ax.scatter(st[:, 0], st[:, 1], s=22, c='#2f6aa8', marker='s')
            ax.set_aspect('equal')
            ax.set_title(f"{n} + {lay['N_h']}", fontsize=8)
        arow[0].text(-0.15, 0.5, nm, transform=arow[0].transAxes, rotation=90, va='center', ha='right', fontsize=9)
    fig.suptitle('Feasible layouts: N sticks (blue squares) + N_h heaters (red circles)', fontsize=10)
    fig.tight_layout()
    return fig
