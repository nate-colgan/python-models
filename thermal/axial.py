"""
Full-height cavity radiation model (decoupled block height brief, §3.3 / §3.6).

One gray enclosure over the full block height with ring-to-ring view factors
(thermal.viewfactors). Shaded: the shade is a porous sheet at r_sh with
straight-through transmission (open fraction τ = 1 − f): a ray crossing it hits
a slat with probability f, otherwise continues undeflected, so the view-factor
matrix is a polynomial F(τ) = C0 + τC1 + τ²C2 (+τ³C3 = 0). τ = 1 reproduces
the unshaded cavity; τ = 0 the two separate gaps. Unshaded: F fixed.

Unknowns, solved together by Newton with an analytic Jacobian and warm start
from the previous call:
  * radiosity J of every surface
  * emissive power E = σT⁴ of each shade ring's outer and inner slat face
  * emissive power of each liner outer-wall ring (active zone)
  * optionally the shade blocked fraction f (cell-flux control; replaces the
    brentq search of the local model)

Surface models
  block ring B_k     gray, T from the conduction solution, ε = gr_emis(T)
  slat faces So/Si   gray RGF faces (ε_s), area f·A_ring; faces of a ring are
                     joined by exact 1-D RGF conduction q = (I(T_o) − I(T_i))/t_sh,
                     I(T) = ∫k dT (any k(T); no layer discretisation needed)
  liner ring L_k     gray outer face (ε = gr_emis(T_Lo)); absorbed flux is
                     conducted through the wall and radiated to the stick in
                     two bands, locally per segment (pre-tabulated vs T_Lo)
  floor / ceiling    reradiating (net flux 0), split at r_sh; loss hook not wired
  liner edge E, pad  reradiating in Phase 3 (adiabatic liner bottom); Phase 4
  underside R        replaces R with the RGF pad / liner-bottom model

The block-wall flux sensitivity dQ/dT_wall (implicit-function theorem on the
Newton system) is returned for the implicit wall coupling in update_temps.
Simplifications are those of the brief §4. Out of scope here: W foils.
"""
import re

import numpy as np

from .properties import gr_emis, gr_tcx, ins_tcx, panel_efficiency

SIGMA = 5.67036713e-8          # same constant as tinless.py
_H, _C, _KB = 6.62607015e-34, 2.99792458e8, 1.380649e-23


def _dgr_emis(T):
    return (gr_emis(T + 0.5) - gr_emis(T - 0.5))


class Kirchhoff:
    """I(T) = ∫ k(T) dT for an insulation k(T) (default RGF ins_tcx), tabulated.
    Steady 1-D planar conduction with k(T): q = (I(T1) − I(T2)) / t, exactly."""

    def __init__(self, k_fn=ins_tcx, T_lo=150.0, T_hi=6000.0, dT=0.25):
        self.k_fn = k_fn
        self.T = np.arange(T_lo, T_hi + dT, dT)
        k = np.asarray(k_fn(self.T), float)
        self.I = np.concatenate([[0.0], np.cumsum(0.5 * (k[1:] + k[:-1]) * dT)])

    def I_of(self, T):
        return np.interp(T, self.T, self.I)

    def k_of(self, T):
        return np.asarray(self.k_fn(T), float)


def layered_conduction_flux(T_o, T_i, t, n_layers, k_fn=ins_tcx):
    """Flux through a slab of n equal layers, each with k at its own mean T
    (the classic layered approximation). n_layers=1 is the legacy model
    (k at the face-mean temperature). Used for the sensitivity study only;
    the solver uses the exact Kirchhoff form (the n → ∞ limit)."""
    from scipy.optimize import brentq
    T_o, T_i = float(T_o), float(T_i)
    if abs(T_o - T_i) < 1e-9:
        return 0.0
    dx = t / n_layers

    def end_T(q):
        T = T_o
        for _ in range(n_layers):
            # layer: q = k((T+T')/2)(T - T')/dx  -> solve for T' by fixed point
            Tn = T - q * dx / k_fn(T)
            for _ in range(30):
                Tn_new = T - q * dx / k_fn(0.5 * (T + Tn))
                if abs(Tn_new - Tn) < 1e-9:
                    break
                Tn = Tn_new
            T = Tn
        return T - T_i

    k_lo = min(k_fn(T_o), k_fn(T_i)) * 0.5
    k_hi = max(k_fn(T_o), k_fn(T_i)) * 2.0
    q_a, q_b = k_lo * (T_o - T_i) / t, k_hi * (T_o - T_i) / t
    return brentq(end_T, min(q_a, q_b), max(q_a, q_b), xtol=1e-10, rtol=1e-12)


class LinerTable:
    """Liner ring: outer-face temperature → (flux into stick, flux to cells,
    liner inner-face T), per unit liner OUTER area. Reproduces the local
    model's liner → stick chain (spectral, Gauss quadrature, wall conduction
    with gr_tcx at the outer-face T) without crosstalk along the stick."""

    def __init__(self, sim, T_in_grid=None):
        ps = sim.powerstick
        if not ps.liner_bool:
            raise NotImplementedError("axial radiation model requires liner_bool: true")
        if sim.liner_tungsten_foil:
            raise NotImplementedError("axial radiation model: liner tungsten_foil not supported yet")
        tab = ps.emissivity_table
        wl, bw = sim._spectral_grid(tab)
        wl_m = wl * 1e-6
        f_bare, f_active = ps.f_bare, ps.f_active
        avg, act = np.zeros(wl.size), np.zeros(wl.size)
        for (rng, e_act, e_bare, e_inact) in tab:
            m = (wl >= rng[0]) & (wl < rng[1])
            ep = f_bare * e_bare + (1 - f_bare) * f_active * e_act + (1 - f_bare) * (1 - f_active) * e_inact
            avg[m] = ep
            act[m] = (1 - f_bare) * f_active * e_act / ep
        T = np.arange(350.0, 3600.0, 2.0) if T_in_grid is None else np.asarray(T_in_grid, float)
        Tc = T[:, None]
        with np.errstate(over='ignore'):
            B = np.pi * (2 * _H * _C ** 2) / wl_m ** 5 / np.expm1(_H * _C / (wl_m * _KB * Tc))
        mat = sim.liner_material
        if mat == 'tac':
            eps_in = np.clip(-0.0351 + 8.15e-5 * Tc + (0.3008 - 4.46e-5 * Tc) / wl, 0.0, 1.0)
        elif mat == 'zrc':
            eps_in = np.clip(0.70 - 1e-4 * Tc + 0.02 * (wl - 0.665), 0.0, 1.0)
        else:
            eps_in = np.broadcast_to(gr_emis(Tc), B.shape)
        eps_in = np.maximum(eps_in, 1e-9)
        r_ps = 2 * ps.width / np.pi
        r_l = 2 * ps.liner_width / np.pi
        denom = (r_ps / r_l) * (1 - eps_in) / eps_in + 1 + (1 - avg) / avg
        q_ps = (B * bw / denom).sum(1)                 # per unit stick area
        q_cell = (B * bw * act / denom).sum(1)
        ratio = (4 * ps.width) / (2 * np.pi * r_l)     # stick area / liner outer area
        q_out, q_c = q_ps * ratio, q_cell * ratio      # per unit liner outer area
        t = sim.liner_thickness
        T_o = T.copy()
        if t > 0:
            for _ in range(50):
                T_new = T + q_out * t / gr_tcx(T_o)
                if np.max(np.abs(T_new - T_o)) < 1e-9:
                    T_o = T_new
                    break
                T_o = T_new
        if np.any(np.diff(T_o) <= 0):
            raise RuntimeError("liner table: T_out(T_in) not monotone")
        self.T_out, self.T_in = T_o, T
        self.q_out, self.q_cell = q_out, q_c
        self.dq_out = np.gradient(q_out, T_o)
        self.dq_cell = np.gradient(q_c, T_o)
        self.band_gap = tab[0][0][1]
        self.f_cell_area = (1 - f_bare) * f_active
        self.ratio = ratio
        # spectral data kept for the liner-bottom model (per quadrature node)
        e_bare = np.zeros(wl.size)
        for (rng, e_act, e_b, e_inact) in tab:
            e_bare[(wl >= rng[0]) & (wl < rng[1])] = e_b
        self.wl, self.wl_m, self.bw = wl, wl_m, bw
        self.eps_cells, self.act, self.eps_bare = avg, act, e_bare
        self.material = mat
        self.r_ratio = r_ps / r_l

    def B(self, T):
        """Band-integrated blackbody emissive power per node [W/m2] at T."""
        with np.errstate(over='ignore'):
            return np.pi * (2 * _H * _C ** 2) / self.wl_m ** 5 / np.expm1(_H * _C / (self.wl_m * _KB * T)) * self.bw

    def eps_in(self, T):
        """Spectral emissivity of the liner inner (stick-facing) face at T, per node."""
        if self.material == 'tac':
            e = np.clip(-0.0351 + 8.15e-5 * T + (0.3008 - 4.46e-5 * T) / self.wl, 0.0, 1.0)
        elif self.material == 'zrc':
            e = np.clip(0.70 - 1e-4 * T + 0.02 * (self.wl - 0.665), 0.0, 1.0)
        else:
            e = np.full(self.wl.size, float(gr_emis(T)))
        return np.maximum(e, 1e-9)

    def q_in_cells_nodes(self, T_in):
        """Per-node net flux into the stick per unit stick area from a liner inner
        face at T_in (local two-surface exchange, as in the table)."""
        e = self.eps_in(T_in)
        denom = self.r_ratio * (1 - e) / e + 1 + (1 - self.eps_cells) / self.eps_cells
        return self.B(T_in) / denom

    def __call__(self, T_out):
        x = self.T_out
        return (np.interp(T_out, x, self.q_out), np.interp(T_out, x, self.dq_out),
                np.interp(T_out, x, self.q_cell), np.interp(T_out, x, self.dq_cell),
                np.interp(T_out, x, self.T_in))


class LinerBottom:
    """Liner bottom + RGF pad (brief §3.5), quasi-steady.

        R (pad underside, in the cavity enclosure) -> RGF pad [∫k dT, t_rgf]
          -> liner graphite plate [k = gr_tcx, t_lb] -> LB (liner-bottom inner face)

    LB exchanges radiation, per spectral quadrature node, with:
      bottom cap  : F_cap->LB = 1 (F_LB->cap = A_cap/A_LB), gas-director emissivity
                    (emis_bare per band), temperature T_cap
      stick sides : lowest active segment; radiosity = reflected liner radiation,
                    J = (1 - eps)/eps * q_in (cells at ~T_cap emit ~nothing)
      liner walls : lowest segment's inner face, treated as black at T_Li (near-black
                    liner channel; the far end of the channel is lumped in here)
    The split of the LB annulus between stick sides and liner walls uses coaxial-
    cylinder view factors over the stick length.
    Areas: pad/plate conduction over A_R (the liner outer footprint), LB face A_LB
    (liner inner footprint), cap A_cap (stick footprint); square proportions kept
    when the enclosure uses equivalent cylinders."""

    def __init__(self, table, sim, A_R, t_rgf, t_lb, eps_R, T_cap, include_cells, L, kir):
        from .viewfactors import coax_F_outer_to_inner
        ps = sim.powerstick
        s_l, s_ps, t_w = ps.liner_width, ps.width, sim.liner_thickness
        s_in = s_l - 2 * t_w
        if s_in <= s_ps:
            raise ValueError("liner inner side must exceed the stick side for the liner-bottom model")
        self.tab, self.kir = table, kir
        self.A_R = float(A_R)
        self.A_LB = self.A_R * (s_in / s_l) ** 2
        self.A_cap = self.A_R * (s_ps / s_l) ** 2
        self.t_rgf, self.t_lb = float(t_rgf), float(t_lb)
        self.eps_R, self.T_cap = float(eps_R), float(T_cap)
        self.include_cells = bool(include_cells)
        F1 = self.A_cap / self.A_LB
        r1, r2 = 2 * s_ps / np.pi, 2 * s_in / np.pi
        F_in_out = (r2 / r1) * float(coax_F_outer_to_inner(r1, r2, L))      # stick side -> liner wall
        G_side_end = 2 * np.pi * r1 * L * 0.5 * (1 - F_in_out)               # stick side -> one end ring
        F_ann_side = G_side_end / (np.pi * (r2 ** 2 - r1 ** 2))
        self.F = (F1, (1 - F1) * F_ann_side, (1 - F1) * (1 - F_ann_side))   # cap, cells, liner walls
        self.E_cap = table.B(self.T_cap)

    def _surround(self, T_Li):
        """Stick-side and liner-wall radiosities seen by LB (depend on T_Li only; cached)."""
        if getattr(self, '_sur_T', None) != T_Li:
            tb = self.tab
            self._sur = ((1 - tb.eps_cells) / tb.eps_cells * tb.q_in_cells_nodes(T_Li), tb.B(T_Li))
            self._sur_T = T_Li
        return self._sur

    def _rad(self, T_LB, T_Li):
        """Net radiation leaving LB [W] and per-node LB radiosity."""
        tb = self.tab
        F1, F2, F3 = self.F
        e = tb.eps_in(T_LB)
        ec = tb.eps_bare
        Jc, Jw = self._surround(float(T_Li))
        E = tb.B(T_LB)
        J = (e * E + (1 - e) * (F1 * ec * self.E_cap + F2 * Jc + F3 * Jw)) / (1 - (1 - e) * F1 * (1 - ec))
        G = F1 * (ec * self.E_cap + (1 - ec) * J) + F2 * Jc + F3 * Jw
        return float(self.A_LB * np.sum(J - G)), J

    def solve(self, T_R, T_Li):
        """Heat through the pad Q [W] (+ = from R into LB) and the internal state."""
        from scipy.optimize import brentq
        I_R = float(self.kir.I_of(T_R))

        def T_mid_of(T_LB, Q):
            return T_LB + Q * self.t_lb / (self.A_R * float(gr_tcx(T_LB))) if self.t_lb > 0 else T_LB

        def res(T_LB):
            Q, _ = self._rad(T_LB, T_Li)
            return float(self.kir.I_of(T_mid_of(T_LB, Q))) + Q * self.t_rgf / self.A_R - I_R
        prev = getattr(self, '_T_LB_prev', None)
        T_LB = None
        if prev is not None:                           # warm start: narrow bracket
            a, b = prev - 40.0, prev + 40.0
            ra, rb = res(a), res(b)
            if ra * rb < 0:
                T_LB = brentq(res, a, b, xtol=1e-6)
        if T_LB is None:
            T_LB = brentq(res, 250.0, max(T_R, T_Li) + 100.0, xtol=1e-6)
        self._T_LB_prev = T_LB
        self._res = res
        Q, J = self._rad(T_LB, T_Li)
        return Q, T_LB, T_mid_of(T_LB, Q), J

    def outputs(self, T_R, T_Li):
        """Energy bookkeeping. LB is fed by the pad (Q_pad) and by radiation from the
        liner walls; the latter is liner-ring heat that the local liner->stick model
        already credits to the stick sides. So the liner bottom's NET extra heat to
        the stick sides is Q_pad - Q_cap (the cap's share comes out of it), and
        total heat to the stick = liner rings + Q_pad exactly.
          Q_LB_cells        net extra heat to the stick sides [W] (may be < 0)
          Q_LB_cells_active its cell share (lowest segment's cell/total ratio)
          Q_LB_sides_gross  radiation absorbed by the stick sides from LB (mostly
                            liner-wall radiation bounced off LB; information only)"""
        Q, T_LB, T_mid, J = self.solve(T_R, T_Li)
        tb = self.tab
        F2 = self.F[1]
        Q_cap = float(self.A_cap * np.sum(tb.eps_bare * J))
        net = Q - Q_cap
        cell_frac = float(np.interp(T_Li, tb.T_in, tb.q_cell) / np.interp(T_Li, tb.T_in, tb.q_out))
        return dict(Q_pad=Q, T_LB=T_LB, T_pad_mid=T_mid, T_R=float(T_R),
                    Q_cap=Q_cap,                                               # absorbed by the bottom cap [W]
                    q_cap=float(np.sum(tb.eps_bare * J)) / 1e4,                 # [W/cm2]
                    Q_LB_cells=net, Q_LB_cells_active=net * cell_frac,
                    Q_LB_sides_gross=float(self.A_LB * F2 * np.sum(tb.eps_cells * J)))

    def Q_and_dQ(self, T_R, T_Li, dT=0.05):
        """Q and dQ/dT_R from one solve (implicit function theorem on the pad residual:
        res(T_LB) = I(T_mid) + Q t_rgf/A_R - I(T_R) = 0 -> dT_LB/dT_R = k(T_R)/res'(T_LB))."""
        Q0, T_LB, _, _ = self.solve(T_R, T_Li)
        dres = (self._res(T_LB + dT) - self._res(T_LB - dT)) / (2 * dT)
        dQ_dTLB = (self._rad(T_LB + dT, T_Li)[0] - self._rad(T_LB - dT, T_Li)[0]) / (2 * dT)
        return Q0, dQ_dTLB * float(self.kir.k_of(T_R)) / dres


# surface kinds
_BLOCK, _RERAD, _LINER, _SLAT_O, _SLAT_I, _PAD = range(6)
_OUTER_SIDE = ('B', 'So', 'floor_out', 'ceil_out')


class AxialRadiation:
    """Coupled full-height radiation solve. Build with thermal.build.build_tinless
    (radiation.axial: true), which attaches it as sim.axial_model.

    vfs: {'porous': porous_shade_view_factors(...)} (shaded) or
         {'single': cavity_view_factors(..., shaded=False)['single']}."""

    def __init__(self, sim, g, vfs, tol=1e-9, max_iter=40, pad=None):
        """pad: None (pad underside R reradiating = adiabatic liner bottom) or a dict
        t_rgf, t_lb, eps_R, T_cap, include_cells -> liner-bottom model (LinerBottom)."""
        if sim.shade_bool and sim.shade_outer_foil:
            raise NotImplementedError("axial radiation model: shade outer_foil not supported yet")
        self.sim, self.g = sim, g
        self.tol, self.max_iter = tol, max_iter
        self.n = n = len(g['edges']) - 1
        self.n_act = na = g['n_act']
        self.n_low = g['n_low']
        self.liner = LinerTable(sim)
        self.kir = Kirchhoff()
        self.eps_s = float(sim.shade_surface_emissivity)
        self.t_sh = float(sim.shade_thickness)
        if 'porous' in vfs:
            vp = vfs['porous']
            self.shaded = True
            self.C = np.asarray(vp.C)
            names, A = list(vp.names), np.asarray(vp.A_ring)
        elif 'single' in vfs:
            en = vfs['single']
            self.shaded = False
            self.C = np.asarray(en.F)[None]
            names, A = list(en.names), np.asarray(en.A)
        else:
            raise ValueError("vfs must contain 'porous' (shaded) or 'single' (unshaded)")
        self.names, self.A = names, A           # A: full ring area for slat faces
        self.N = N = len(names)
        idx = {nm: i for i, nm in enumerate(names)}
        self.idx = idx
        kind = np.full(N, _RERAD)
        self.iB = np.array([idx[f'B{k}'] for k in range(n)])
        self.iL = np.array([idx[f'L{k}'] for k in range(na)])
        kind[self.iB], kind[self.iL] = _BLOCK, _LINER
        Eidx = np.full(N, -1)
        if self.shaded:
            # conducting face pairs (outer, inner); default one slat pair per ring.
            # vp.pairs may give any list, e.g. explicit panel faces split in bins.
            pairs = getattr(vp, 'pairs', None) or [(f'So{k}', f'Si{k}') for k in range(n)]
            self.iSo = np.array([idx[a] for a, _ in pairs])
            self.iSi = np.array([idx[b] for _, b in pairs])
            m_ = len(pairs)
            kind[self.iSo], kind[self.iSi] = _SLAT_O, _SLAT_I
            Eidx[self.iSo], Eidx[self.iSi] = np.arange(m_), m_ + np.arange(m_)
            self.eL = 2 * m_ + np.arange(na)
        else:
            self.iSo = self.iSi = np.array([], int)
            self.eL = np.arange(na)
        Eidx[self.iL] = self.eL
        self.n_sl = len(self.iSo)
        self.nE = len(self.eL) + 2 * self.n_sl
        self.pad = None
        if pad is not None and 'R' in idx:
            self.iR = idx['R']
            self.eR = self.nE
            self.nE += 1
            kind[self.iR] = _PAD
            Eidx[self.iR] = self.eR
            self.pad = LinerBottom(self.liner, sim, A[self.iR], pad['t_rgf'], pad['t_lb'], pad['eps_R'],
                                   pad['T_cap'], pad.get('include_cells', False),
                                   g['edges_act'][-1] - g['edges_act'][0], self.kir)
        self.iS = np.r_[self.iSo, self.iSi].astype(int)
        self.eS = Eidx[self.iS] if self.shaded else np.array([], int)
        self.kind, self.Eidx = kind, Eidx
        self.slat = np.isin(kind, (_SLAT_O, _SLAT_I))
        self.outer_side = np.array([nm.rstrip('0123456789') in _OUTER_SIDE for nm in names])
        self.A_L = A[self.iL]
        self.A_B = A[self.iB]
        ps = sim.powerstick
        self.A_cell_cm2_seg = ps.A_step * (1 - ps.f_bare) * ps.f_active * 1e4
        self.A_cell_cm2 = self.A_cell_cm2_seg * na
        self._x = None
        self._f = None
        self.stats = dict(calls=0, newton_its=0, fixed_solves=0)

    # ------------------------------------------------------------------
    def F_of(self, f):
        """View factors and d/dtau at blocked fraction f (tau = 1 - f)."""
        C = self.C
        if C.shape[0] == 1:
            return C[0], np.zeros_like(C[0])
        tau = 1.0 - f
        F = C[0] + tau * (C[1] + tau * (C[2] + tau * C[3]))
        dF = C[1] + tau * (2 * C[2] + 3 * tau * C[3])
        return F, dF

    def area(self, f):
        """Actual surface areas at blocked fraction f (slat faces f·A_ring)."""
        return np.where(self.slat, f * self.A, self.A)

    def _resjac(self, x, f, E_B, eps_B, Q_tgt):
        """Residual and Jacobian. If Q_tgt is not None, f is the last unknown."""
        N, nE = self.N, self.nE
        free = Q_tgt is not None
        F, dF = self.F_of(f)
        J = x[:N]
        Eu = np.maximum(x[N:N + nE], 1e-3)
        Tu = (Eu / SIGMA) ** 0.25
        dTdE = Tu / (4 * Eu)
        G = F @ J
        dGdf = -(dF @ J) if free else None             # d tau / d f = -1
        iB, iL, iS, eL, eS = self.iB, self.iL, self.iS, self.eL, self.eS
        es = self.eps_s
        nx = N + nE + (1 if free else 0)
        r = np.empty(nx)
        Jac = np.zeros((nx, nx))

        # --- radiosity rows: J - c G - a = 0
        c = np.ones(N)                                  # reradiating default
        a = np.zeros(N)
        c[iB] = 1 - eps_B
        a[iB] = eps_B * E_B
        T_L = Tu[eL]
        eps_L = gr_emis(T_L)
        deps_L = _dgr_emis(T_L) * dTdE[eL]
        c[iL] = 1 - eps_L
        a[iL] = eps_L * Eu[eL]
        if self.shaded:
            c[iS] = 1 - es
            a[iS] = es * Eu[eS]
            Jac[iS, N + eS] = -es
        r[:N] = J - c * G - a
        Jac[:N, :N] = np.eye(N) - c[:, None] * F
        Jac[iL, N + eL] = -eps_L - deps_L * (Eu[eL] - G[iL])
        if free:
            Jac[:N, -1] = -c * dGdf

        # --- slat face balances (per slat area; f cancels)
        if self.shaded:
            n = self.n_sl
            eo, ei = np.arange(n), n + np.arange(n)
            iSo, iSi = self.iSo, self.iSi
            To, Ti = Tu[eo], Tu[ei]
            qc = (self.kir.I_of(To) - self.kir.I_of(Ti)) / self.t_sh
            ko = self.kir.k_of(To) / self.t_sh * dTdE[eo]
            ki = self.kir.k_of(Ti) / self.t_sh * dTdE[ei]
            ro, ri = N + eo, N + ei
            r[ro] = es * (G[iSo] - Eu[eo]) - qc
            r[ri] = es * (G[iSi] - Eu[ei]) + qc
            Jac[ro, :N] = es * F[iSo]
            Jac[ri, :N] = es * F[iSi]
            Jac[ro, ro] = -es - ko
            Jac[ro, ri] = ki
            Jac[ri, ri] = -es - ki
            Jac[ri, ro] = ko
            if free:
                Jac[ro, -1] = es * dGdf[iSo]
                Jac[ri, -1] = es * dGdf[iSi]

        # --- liner balances
        q, dq, qc_, dqc_, T_Li = self.liner(T_L)
        if self.pad is not None:          # liner bottom: R gray (eps_R), heat through the pad
            pd, iR, eR = self.pad, self.iR, self.eR
            T_R = Tu[eR]
            r[iR] = J[iR] - (1 - pd.eps_R) * G[iR] - pd.eps_R * Eu[eR]
            Jac[iR, :N] = -(1 - pd.eps_R) * F[iR]
            Jac[iR, iR] += 1.0
            Jac[iR, N + eR] = -pd.eps_R
            Qp, dQp = pd.Q_and_dQ(T_R, T_Li[0])
            r[N + eR] = pd.eps_R * (G[iR] - Eu[eR]) - Qp / pd.A_R
            Jac[N + eR, :N] = pd.eps_R * F[iR]
            Jac[N + eR, N + eR] = -pd.eps_R - dQp / pd.A_R * dTdE[eR]
            if free:
                Jac[iR, -1] = -(1 - pd.eps_R) * dGdf[iR]
                Jac[N + eR, -1] = pd.eps_R * dGdf[iR]
        rl = N + eL
        r[rl] = eps_L * (G[iL] - Eu[eL]) - q
        Jac[rl, :N] = eps_L[:, None] * F[iL]
        Jac[rl, rl] = -eps_L + deps_L * (G[iL] - Eu[eL]) - dq * dTdE[eL]
        if free:
            Jac[rl, -1] = eps_L * dGdf[iL]
            At = self.A_L.sum()
            Q_lb = 0.0
            if self.pad is not None and self.pad.include_cells:     # lagged (no derivative)
                Q_lb = self.pad.outputs(Tu[self.eR], T_Li[0])['Q_LB_cells_active']
            r[-1] = ((qc_ * self.A_L).sum() + Q_lb - Q_tgt) / At
            Jac[-1, rl] = self.A_L * dqc_ * dTdE[eL] / At
        return r, Jac

    def _newton(self, x, f, E_B, eps_B, Q_tgt):
        N, nE = self.N, self.nE
        free = Q_tgt is not None
        scale = float(E_B.max())
        for it in range(1, self.max_iter + 1):
            r, Jac = self._resjac(x, f, E_B, eps_B, Q_tgt)
            if np.max(np.abs(r)) < self.tol * scale and it > 1:
                return x, f, it - 1, True
            d = np.linalg.solve(Jac, -r)
            lam = 1.0
            dE = d[N:N + nE]
            Eu = x[N:N + nE]
            neg = dE < 0
            if np.any(neg):          # keep E > 0: at most halve any E per step
                lam = min(1.0, float(np.min(0.5 * Eu[neg] / -dE[neg])))
            if free and abs(d[-1]) > 0.25:
                lam = min(lam, 0.25 / abs(d[-1]))
            self.stats['newton_its'] += 1
            x = x + lam * d[:N + nE]
            if free:
                f = float(np.clip(f + lam * d[-1], 0.0, 1.0))
        r, _ = self._resjac(x, f, E_B, eps_B, Q_tgt)
        return x, f, self.max_iter, np.max(np.abs(r)) < 1e3 * self.tol * scale

    def _guess(self, E_B):
        Em = float(E_B.mean())
        x = np.empty(self.N + self.nE)
        x[:self.N] = Em
        x[self.N:] = 0.5 * Em
        if self.shaded:
            x[self.N:self.N + self.n_sl] = 0.9 * Em
        return x

    # ------------------------------------------------------------------
    def solve(self, T_wall, f=None, control=False):
        """Coupled solve at block wall temperatures T_wall (n rings, bottom-up).
        control=True: find f in [f_shade_min, f_shade_max] holding the cell-flux
        setpoint (clamped at the bounds, like the local model); else use f
        (default 0; ignored for the unshaded cavity)."""
        sim = self.sim
        self.stats['calls'] += 1
        T_wall = np.asarray(T_wall, float)
        E_B = SIGMA * T_wall ** 4
        eps_B = gr_emis(T_wall)
        x0 = self._x if self._x is not None else self._guess(E_B)
        if not self.shaded:
            f, control = 0.0, False
        if control:
            Q_tgt = sim.nominal_cell_flux * self.A_cell_cm2
            f_lo, f_hi = sim.f_shade_min, sim.f_shade_max
            f0 = self._f if self._f is not None else 0.5 * (f_lo + f_hi)
            # last step sat on a bound: try that bound first (cheap, usual case)
            if self._f is not None and (self._f <= f_lo + 1e-12 or self._f >= f_hi - 1e-12):
                fb = f_lo if self._f <= f_lo + 1e-12 else f_hi
                x, _, _, ok = self._newton(x0.copy(), fb, E_B, eps_B, None)
                if ok:
                    qc = self._qcell(x)
                    if (fb == f_lo and qc <= Q_tgt) or (fb == f_hi and qc >= Q_tgt):
                        return self._finish(x, fb, E_B, eps_B)
                    x0 = x
            x, fs, _, ok = self._newton(x0.copy(), f0, E_B, eps_B, Q_tgt)
            if ok and f_lo <= fs <= f_hi:
                return self._finish(x, fs, E_B, eps_B, Q_tgt)
            self.stats['fixed_solves'] += 1
            fb = f_lo if (ok and fs < f_lo) else f_hi if (ok and fs > f_hi) else None
            if fb is None:          # robust fallback: bracket like the local model
                return self._bracket(x0, E_B, eps_B, Q_tgt, f_lo, f_hi)
            x, _, _, ok = self._newton(x0.copy(), fb, E_B, eps_B, None)
            if not ok:
                raise RuntimeError("axial radiation: Newton failed at shade bound")
            return self._finish(x, fb, E_B, eps_B)
        f = 0.0 if f is None else float(f)
        x, _, _, ok = self._newton(x0.copy(), f, E_B, eps_B, None)
        if not ok:
            x, _, _, ok = self._newton(self._guess(E_B), f, E_B, eps_B, None)
            if not ok:
                raise RuntimeError("axial radiation: Newton did not converge")
        return self._finish(x, f, E_B, eps_B)

    def _bracket(self, x0, E_B, eps_B, Q_tgt, f_lo, f_hi):
        from scipy.optimize import brentq
        cache = {}

        def flux(fs):
            x, _, _, ok = self._newton(x0.copy(), fs, E_B, eps_B, None)
            if not ok:
                x, _, _, ok = self._newton(self._guess(E_B), fs, E_B, eps_B, None)
            cache[fs] = x
            return self._qcell(x) - Q_tgt
        if flux(f_lo) <= 0:
            return self._finish(cache[f_lo], f_lo, E_B, eps_B)
        if flux(f_hi) >= 0:
            return self._finish(cache[f_hi], f_hi, E_B, eps_B)
        fs = brentq(flux, f_lo, f_hi, xtol=1e-6)
        flux(fs)
        x, fs2, _, ok = self._newton(cache[fs], fs, E_B, eps_B, Q_tgt)
        return self._finish(x if ok else cache[fs], fs2 if ok else fs, E_B, eps_B, Q_tgt if ok else None)

    def _qcell(self, x):
        T_L = (np.maximum(x[self.N + self.eL], 1e-3) / SIGMA) ** 0.25
        return float((self.liner(T_L)[2] * self.A_L).sum())

    def _dQ_dTw(self, x, f, E_B, eps_B, Q_tgt):
        """d(Q_block)/d(T_wall), n x n, at a converged state (implicit function
        theorem on the Newton system; includes the shade-control response when
        f is free)."""
        _, Jac = self._resjac(x, f, E_B, eps_B, Q_tgt)
        N = self.N
        F, _ = self.F_of(f)
        T = (E_B / SIGMA) ** 0.25
        G = F[self.iB] @ x[:N]
        dr = np.zeros((Jac.shape[0], self.n))
        dr[self.iB, np.arange(self.n)] = -(_dgr_emis(T) * (E_B - G) + gr_emis(T) * 4 * SIGMA * T ** 3)
        dx = -np.linalg.solve(Jac, dr)
        dJ = dx[:N]
        return self.A_B[:, None] * (dJ[self.iB] - F[self.iB] @ dJ)

    def _finish(self, x, f, E_B, eps_B, Q_tgt=None):
        self._x, self._f = x.copy(), f
        N = self.N
        F, _ = self.F_of(f)
        J = x[:N]
        G = F @ J
        Eu = np.maximum(x[N:N + self.nE], 1e-3)
        Tu = (Eu / SIGMA) ** 0.25
        T_Lo = Tu[self.eL]
        q, _, qc, _, T_Li = self.liner(T_Lo)
        Q_block = self.A_B * (J[self.iB] - G[self.iB])          # W leaving each ring
        Q_ps = q * self.A_L
        Q_cell = qc * self.A_L
        out = dict(f=f, J=J, G=G, F=F, Q_block=Q_block, Q_ps=Q_ps, Q_cell=Q_cell,
                   T_liner_out=T_Lo, T_liner_in=T_Li)
        rerad = self.kind == _RERAD
        out['T_rerad'] = {self.names[i]: (max(G[i], 0) / SIGMA) ** 0.25 for i in np.where(rerad)[0]}
        if self.shaded:
            n = self.n_sl
            T_so, T_si = Tu[:n], Tu[n:2 * n]
            X = self.area(f)[:, None] * F
            o, i_ = self.outer_side, ~self.outer_side
            out.update(T_shade_out=T_so, T_shade_in=T_si,
                       Q_shade_cond=f * self.A[self.iSo] * (self.kir.I_of(T_so) - self.kir.I_of(T_si)) / self.t_sh,
                       Q_through_openings=float((X[np.ix_(o, i_)] * (J[o][:, None] - J[i_][None, :])).sum()))
        Q_sink = Q_ps.sum()
        if self.pad is not None:
            po = self.pad.outputs(Tu[self.eR], T_Li[0])
            out.update(po)
            Q_sink += po['Q_pad']
            if self.pad.include_cells:
                out['Q_cell'] = Q_cell.copy()
                out['Q_cell'][0] += po['Q_LB_cells_active']
                out['Q_ps'] = Q_ps.copy()
                out['Q_ps'][0] += po['Q_LB_cells']
        out['Q_sink'] = float(Q_sink)
        out['balance_err'] = abs(Q_block.sum() - Q_sink) / max(abs(Q_block.sum()), 1e-30)
        out['dQ_dTw'] = self._dQ_dTw(x, f, E_B, eps_B, Q_tgt)
        self.last = out
        return out

    # ------------------------------------------------------------------
    def vertical_transport(self, sol=None, z_plane=None):
        """Net radiative power crossing the horizontal plane z_plane (default: the
        liner-bottom plane) upward: Σ_{i below, j above} A_i F_ij (J_i − J_j), split
        by path: 'outer' (both surfaces in the block-shade gap), 'inner' (both in
        the shade-liner space), 'through' (exchange through the shade openings).
        Floor/pad count as below, ceiling as above."""
        sol = sol or self.last
        e, ea = self.g['edges'], self.g['edges_act']
        zp = self.g['z_act0'] if z_plane is None else z_plane
        J, F = sol['J'], sol['F']
        X = self.area(sol['f'])[:, None] * F
        zc = np.full(self.N, np.nan)
        for nm, i in self.idx.items():
            head = nm.rstrip('0123456789')
            mv = re.match(r'^[fd](\d+)(p\d+|m\d+|ep|em)$', nm)   # vane faces / ends, ring k
            if mv:
                k = int(mv.group(1)); zc[i] = 0.5 * (e[k] + e[k + 1])
            elif head in ('B', 'So', 'Si'):
                k = int(nm[len(head):]); zc[i] = 0.5 * (e[k] + e[k + 1])
            elif head == 'L':
                k = int(nm[1:]); zc[i] = 0.5 * (ea[k] + ea[k + 1])
            elif nm.startswith('floor') or nm in ('R', 'E'):
                zc[i] = -1.0
            elif nm.startswith('ceil'):
                zc[i] = np.inf
        lo, hi = zc < zp, zc > zp
        T = X * (J[:, None] - J[None, :])
        o = self.outer_side
        pair_o = o[:, None] & o[None, :]
        pair_i = ~o[:, None] & ~o[None, :]
        m = lo[:, None] & hi[None, :]
        res = {'outer': float(T[m & pair_o].sum()), 'inner': float(T[m & pair_i].sum()),
               'through': float(T[m & ~pair_o & ~pair_i].sum())}
        if not self.shaded:
            res = {'cavity': res['outer'] + res['inner'] + res['through']}
        return res

    # ------------------------------------------------------------------
    def step(self, T_dist, control):
        """Drop-in for TinlessSimulation._compute_f_shade + _step_calc.
        Returns f, (Q_rad[n rings, + = leaving block], Q_elec, Q_cell, Q_waste, T_liner)
        with the last four over the active rings; sets sim._step_temps."""
        sol = self.solve(T_dist[:, 0], control=control and self.shaded)
        Q_cell = sol['Q_cell']
        flux = Q_cell / self.A_cell_cm2_seg
        eff = panel_efficiency(self.liner.band_gap, sol['T_liner_in'], flux=flux)
        Q_elec = Q_cell * eff
        Q_waste = sol['Q_ps'] - Q_elec
        nanv = np.full(self.n, np.nan)
        self.sim._step_temps = {
            'T_liner_out': sol['T_liner_out'],
            'T_foil_inner': np.full(self.n_act, np.nan),
            'T_sh_in': sol.get('T_shade_out', nanv),     # legacy naming: heat-in face
            'T_sh_out': sol.get('T_shade_in', nanv),     # (block side / liner side)
        }
        return sol['f'], (sol['Q_block'], Q_elec, Q_cell, Q_waste, sol['T_liner_in'])


class VaneHybridRadiation:
    """Rotating-vane shade (square cavity, octagon of 8 centre-pivoted vanes),
    represented by three ray-traced vane positions:

        closed : 0 deg (closed octagon with the real corner gaps) -> minimum flux
        mid    : THETA_MID deg (10)                                -> where w(theta) bends most
        open   : 90 deg (edge-on)                                  -> maximum flux

    Each is an exact coupled radiation solve (AxialRadiation) on view factors
    ray-traced once per geometry in the real square geometry
    (thermal.correlations._vane_enclosure), so temperature level, gradient and
    the per-ring split of block heat come from physics at every step.

    Control: if the cell-flux setpoint lies between the closed and open limits it is
    held exactly, with the per-ring block and liner heat blended piecewise-linearly
    along s in [0, 2] (0 closed, 1 mid, 2 open); s is chosen to hit the setpoint.
    Against ray-traced 2/5/30 deg cases the piecewise blend is within ~1.1 % of total
    heat per ring (lower-zone share within ~0.03); a closed/open-only blend was 2.6 %.
    Otherwise the vanes sit fully closed (s = 0) or open (s = 2).
    The vane angle is estimated from s with the vane table (plotting only).
    Reported f_shade = 1 - s/2."""

    THETA_MID = 10.0
    KEYS = ('closed', 'mid', 'open')

    def __init__(self, sim, g, lay, n_rays=2 ** 14, n_bins=2, cache_dir=None, pad=None):
        from types import SimpleNamespace
        from . import correlations as C
        self.sim, self.g, self.lay = sim, g, lay
        self.n = len(g['edges']) - 1
        self.n_act, self.n_low = g['n_act'], g['n_low']
        self.anchors = {}
        for key, theta in zip(self.KEYS, (0.0, self.THETA_MID, 90.0)):
            enc = self._enclosure(C, lay, theta, n_bins, n_rays, cache_dir)
            vp = SimpleNamespace(C=enc.F[None], A_ring=enc.A, slat=enc.slat, names=enc.names,
                                 idx=enc.idx, pairs=enc.pairs)
            self.anchors[key] = AxialRadiation(sim, g, {'porous': vp}, pad=pad)
        ax = self.anchors['open']
        self.pad = ax.pad
        self.liner = ax.liner
        self.A_L = ax.A_L
        self.A_cell_cm2_seg = ax.A_cell_cm2_seg
        self.A_cell_cm2 = ax.A_cell_cm2
        self.shaded = True
        self.stats = dict(calls=0)
        self.last = None
        self._angle_curve(C)

    def _angle_curve(self, C):
        """Vane angle vs s from the ray-traced vane table (thermal.correlations.VANE_GRID):
        in each segment, the normalised table heat fraction between its two anchor
        angles, at this vane's R'' and block H/L (nearly temperature independent:
        ~1-3 deg). Plotting estimate only."""
        import warnings
        e, ea = self.g['edges'], self.g['edges_act']
        self.H_over_L = float(e[-1] / (ea[-1] - ea[0]))
        self.R_cond = float(C.vane_R_cond(self.lay['t']))
        Rg, hg = C.VANE_GRID['R_cond'], C.VANE_GRID['H_over_L']
        if not (min(Rg) * 0.999 <= self.R_cond <= max(Rg) * 1.001 and min(hg) - 1e-3 <= self.H_over_L <= max(hg) + 1e-3):
            warnings.warn(f"vane angle estimate: R''={self.R_cond:.3g} m2K/W or H/L={self.H_over_L:.2f} outside the "
                          "vane table; nearest data used", stacklevel=3)
        self._seg = []
        for a, b in ((0.0, self.THETA_MID), (self.THETA_MID, 90.0)):
            th = np.linspace(a, b, 401)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                ph = np.asarray(C.vane_heat_fraction(th, self.H_over_L, R_cond=self.R_cond))
            self._seg.append((np.maximum.accumulate((ph - ph[0]) / (ph[-1] - ph[0])), th))

    def angle_of(self, s):
        """Approximate vane angle [deg] (0 closed, 90 edge-on) for blend parameter s in [0, 2]."""
        k = 0 if s <= 1.0 else 1
        w, th = self._seg[k]
        return float(np.interp(s - k, w, th))

    def _enclosure(self, C, lay, theta, n_bins, n_rays, cache_dir):
        import hashlib
        import os
        import pickle
        g = self.g
        path = None
        if cache_dir:
            parts = [f"{lay[k]:.6f}" for k in ('A', 'l', 'liner_half', 't', 'gap')] + \
                    [",".join(f"{x:.6f}" for x in g['edges']), ",".join(f"{x:.6f}" for x in g['edges_act']),
                     f"{g['z_act0']:.6f}", f"{g['H_low']:.6f}", f"{theta}", str(n_bins), str(n_rays), "vane-v1"]
            path = os.path.join(cache_dir, "vane_" + hashlib.sha1("|".join(parts).encode()).hexdigest()[:16] + ".pkl")
            if os.path.exists(path):
                return pickle.load(open(path, 'rb'))
        enc = C._vane_enclosure(lay, g['edges'], g['edges_act'], g['z_act0'], g['H_low'], theta,
                                n_bins, n_rays)
        if path:
            os.makedirs(cache_dir, exist_ok=True)
            pickle.dump(enc, open(path, 'wb'))
        return enc

    def _liner_from_Q(self, Q_ps):
        """Heat into each liner ring -> (T_out, q_cell per liner area, T_in)."""
        tb = self.liner
        T_out = np.interp(Q_ps / self.A_L, tb.q_out, tb.T_out)
        return T_out, np.interp(T_out, tb.T_out, tb.q_cell), np.interp(T_out, tb.T_out, tb.T_in)

    @staticmethod
    def _pw(vals, s):
        """Piecewise-linear blend of three anchor values at s in [0, 2]."""
        k = 0 if s <= 1.0 else 1
        return vals[k] + (s - k) * (vals[k + 1] - vals[k])

    def solve(self, T_wall, control=False, s=None):
        """control=True: hold the setpoint if feasible; else s (default 2 = fully open)."""
        from scipy.optimize import brentq
        self.stats['calls'] += 1
        sols = [self.anchors[k].solve(T_wall, f=1.0) for k in self.KEYS]
        # liner-ring heat only (the anchors' Q_ps may include liner-bottom -> cells)
        Qp = [self.anchors[k].A_L * self.liner(so['T_liner_out'])[0] for k, so in zip(self.KEYS, sols)]
        Qb = [so['Q_block'] for so in sols]
        pad_keys = ('Q_pad', 'T_LB', 'T_pad_mid', 'T_R', 'Q_cap', 'q_cap', 'Q_LB_cells', 'Q_LB_cells_active',
                    'Q_LB_sides_gross')
        has_pad = self.pad is not None
        inc = has_pad and self.pad.include_cells
        lbc = (lambda x: self._pw([so['Q_LB_cells_active'] for so in sols], x)) if inc else (lambda x: 0.0)

        def qcell_tot(x):
            return float((self._liner_from_Q(self._pw(Qp, x))[1] * self.A_L).sum()) + lbc(x)
        Q_lo, Q_hi = float(sols[0]['Q_cell'].sum()), float(sols[2]['Q_cell'].sum())
        if control:
            Q_tgt = self.sim.nominal_cell_flux * self.A_cell_cm2
            if Q_hi <= Q_tgt:
                s = 2.0
            elif Q_lo >= Q_tgt:
                s = 0.0
            else:
                s = brentq(lambda x: qcell_tot(x) - Q_tgt, 0.0, 2.0, xtol=1e-9)
        else:
            s = 2.0 if s is None else float(np.clip(s, 0.0, 2.0))
        Q_block = self._pw(Qb, s)
        Q_ps = self._pw(Qp, s)
        T_out, qc, T_in = self._liner_from_Q(Q_ps)
        Q_cell = qc * self.A_L
        pad_out = {k: self._pw([so[k] for so in sols], s) for k in pad_keys} if has_pad else {}
        Q_sink = float(Q_ps.sum()) + pad_out.get('Q_pad', 0.0)
        if inc:
            Q_cell = Q_cell.copy(); Q_cell[0] += pad_out['Q_LB_cells_active']
            Q_ps = Q_ps.copy(); Q_ps[0] += pad_out['Q_LB_cells']
        out = dict(s=s, w=s / 2.0, f=1.0 - s / 2.0, vane_angle=self.angle_of(s),
                   Q_block=Q_block, Q_ps=Q_ps, Q_cell=Q_cell, Q_sink=Q_sink,
                   Q_cell_min=Q_lo, Q_cell_max=Q_hi, Q_cell_mid=float(sols[1]['Q_cell'].sum()),
                   T_liner_out=T_out, T_liner_in=T_in,
                   dQ_dTw=self._pw([so['dQ_dTw'] for so in sols], s),
                   T_vane_out=float(np.mean(self._pw([so['T_shade_out'] for so in sols], s))),
                   T_vane_in=float(np.mean(self._pw([so['T_shade_in'] for so in sols], s))),
                   balance_err=abs(Q_block.sum() - Q_sink) / max(abs(Q_block.sum()), 1e-30),
                   anchors=tuple(sols), **pad_out)
        self.last = out
        return out

    def vertical_transport(self, sol=None):
        """Net upward radiative transfer across the liner-bottom plane, blended
        piecewise between the three vane positions like everything else."""
        sol = sol or self.last
        v = [sum(self.anchors[k].vertical_transport(so).values()) for k, so in zip(self.KEYS, sol['anchors'])]
        return {'cavity': float(self._pw(np.array(v), sol['s']))}

    def step(self, T_dist, control):
        """Drop-in for the TinlessSimulation radiation step (see AxialRadiation.step)."""
        sol = self.solve(T_dist[:, 0], control=control)
        Q_cell = sol['Q_cell']
        eff = panel_efficiency(self.liner.band_gap, sol['T_liner_in'], flux=Q_cell / self.A_cell_cm2_seg)
        Q_elec = Q_cell * eff
        self.sim._step_temps = {
            'T_liner_out': sol['T_liner_out'],
            'T_foil_inner': np.full(self.n_act, np.nan),
            'T_sh_in': np.full(self.n, sol['T_vane_out']),
            'T_sh_out': np.full(self.n, sol['T_vane_in']),
        }
        return sol['f'], (sol['Q_block'], Q_elec, Q_cell, sol['Q_ps'] - Q_elec, sol['T_liner_in'])
