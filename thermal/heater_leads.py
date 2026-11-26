"""
Heater current-lead optimizer (separate from the system optimizer).

Each heater has n_leads current leads (2 for legs = 1: supply + return) that carry the heater
current I = P / V from the element, at the bottom face of the top Thermax (x = 0), up through
the Thermax (thickness L = tank.t_top) to the terminal at its top face (x = L). A lead is up to
`zones` material zones in series, hot -> cold (default graphite / tungsten / copper), each a
solid round bar of its own diameter and length. The optimizer picks the diameters and lengths
that minimise the lifetime cost of one lead:

    capex     material (volume x $/m3) + joints + heat rejection ($/kW_th on the peak load)
              + heater supply ($/kW on the lead's Joule power), x (1 + soft_cost_frac)
    energy    lifetime x elec_price x [Joule heat + storage heat drained / eta_in
              + heat-rejection parasitics (kW_e per kW_th rejected)]

as the system optimizer prices energy (economics.elec_price, no discounting, cycles_per_year x
lifetime_yr cycles).

1-D steady model along the lead, solved for two states per cycle:
    charge  current I for charge_hr; hot end at T_hot_charge (heater max_temperature)
    idle    no current for the rest of the cycle (hold + discharge); hot end at the block mean
    Thermax hot (bottom) face at the block T_max in both states (T_ins_hot), top face T_ins_cold.
    d/dx(k A dT/dx) + I^2 rho_e(T) / A - q_ex = 0
    q_ex = exchange with the surrounding insulation, per unit length: across the annular gap to the
           bore wall (T_b), gas conduction 2 pi k_gas (T - T_b) / ln(r_b / r) + radiation between
           concentric cylinders 2 pi r sigma (T^4 - T_b^4) / (1/eps + r/r_b (1/eps_w - 1)); in series
           with radial conduction through the Thermax, 2 pi k_thermax (T_b - T_w) / ln(r_far / r_b),
           out to r_far where the Thermax is at its undisturbed temperature T_w(x) (r_far None: the
           bore wall itself at T_w).
    T_w(x): the undisturbed Thermax temperature: the exact steady 1-D profile with the Thermax
            k(T) = K0 exp(B T_C) (properties.thermax_tcx). The heat flux through the slab is uniform,
            so the Kirchhoff transform phi = int k dT is linear in x, i.e. exp(B T_w) is linear in x:
            T_w = ln(e_h + (e_c - e_h) x/L) / B, e = exp(B T_C) at the faces. Hotter than a linear
            profile over most of the thickness (k rises steeply with T).
Boundary conditions: T(0) = T_hot; cold end T(L) = T_term (water-cooled terminal clamp) with the
last zone uncooled ('passive'); or with the last zone water-cooled along its length ('cooled'),
the 1-D solve ends at its start, which is held at T_cool.

Energy accounting. Heat the lead puts into the insulation at height x goes partly up to the deck
and partly back down to the storage: the split of a source in a 1-D slab with both faces held,
w(x) = R(0..x) / R(0..L) up, 1 - w down; for k = K0 exp(B T) this is exactly
w = (T_ins_hot - T_w(x)) / (T_ins_hot - T_ins_cold). Then
    heat rejected   Q_rej  = heat out of the cold end (+ the cooled zone's Joule heat and the
                             heat it absorbs from the insulation)
    storage drain   Q_drain = Q_hot (heat into the lead at x = 0) - sum q_ex (1 - w)
    Q_J + Q_drain = Q_rej + sum q_ex w            (energy balance; checked)
Joule heat that flows back down the lead into the storage counts as delivered.

Limits: each zone's material max service temperature (T_max) over its length, and each joint
below the lower T_max of its two materials (the colder side); a violation adds a penalty of
penalty_per_K $ per K. Every zone is at least min_zone_len long; optimize_with_skips re-runs the
optimization without any zone an optimum leaves at that minimum (graphite -> copper, tungsten ->
copper, graphite only, ...) and keeps the cheapest.

Not modelled: contact resistance at the joints, the lead above the insulation top face, the
current falling once the heater holds max_temperature (the charge state uses the rated current
throughout: conservative for Joule heat), insulation disturbed by the lead, axial variation of
the heater current, W-C reaction at a graphite / tungsten joint (W2C / WC form above ~1400 C).
"""
import copy
import math

import numpy as np
from scipy.linalg import solve_banded

from .properties import gr_tcx, heater_graphite_resistivity, thermax_tcx, THERMAX_B
from .heater import heater_max_power, element as heater_element

_trapz = getattr(np, 'trapezoid', None) or np.trapz
SIGMA = 5.670374419e-8
C2K = 273.15
UOHM_CM = 1e-8          # micro-ohm cm -> ohm m


def _table(T_K, vals, scale=1.0):
    T_K, vals = np.asarray(T_K, float), np.asarray(vals, float) * scale
    return lambda T: np.interp(T, T_K, vals)


# Material library: k(T) [W/m-K], rho_e(T) [ohm m], T in K (clamped outside the tables);
# density [kg/m3]; default T_max [K] (max service temperature), emissivity of the lead surface,
# cost_per_m3 [$/m3] (PLACEHOLDERS; graphite None -> heater graphite $/kg x density);
# coolable: may be the water-cooled last zone (others are only run passive as the last zone).
MATERIALS = {
    # heater-element graphite: Mersen 2020 resistivity (properties.heater_graphite_resistivity);
    # k from the properties.gr_tcx table read as C (its first column runs 20-2400)
    'graphite': dict(
        k=lambda T: gr_tcx(np.asarray(T, float) - C2K), rho_e=heater_graphite_resistivity,
        density=1850.0, T_max=3273.15, emissivity=0.85, cost_per_m3=None),
    # k: Incropera Table A.1; rho_e: Desai et al. 1984 (J. Phys. Chem. Ref. Data 13), approx.
    'tungsten': dict(
        k=_table([300, 400, 600, 800, 1000, 1200, 1500, 2000, 2500, 3000],
                 [174, 159, 137, 125, 118, 113, 108, 100, 95, 95]),
        rho_e=_table([300, 400, 500, 600, 800, 1000, 1200, 1500, 2000, 2500, 3000, 3500],
                     [5.44, 7.83, 10.3, 12.9, 18.3, 24.9, 31.0, 40.4, 56.7, 73.4, 90.9, 108.6], UOHM_CM),
        density=19300.0, T_max=2773.15, emissivity=0.25, cost_per_m3=2.5e6),
    'molybdenum': dict(
        k=_table([300, 400, 600, 800, 1000, 1200, 1500, 2000, 2500],
                 [138, 134, 126, 118, 112, 105, 98, 90, 86]),
        rho_e=_table([300, 400, 500, 600, 800, 1000, 1200, 1500, 2000, 2500, 2890],
                     [5.52, 7.86, 10.2, 12.6, 17.6, 22.9, 28.3, 36.6, 50.9, 66.0, 77.7], UOHM_CM),
        density=10220.0, T_max=2073.15, emissivity=0.2, cost_per_m3=6.0e5),
    # k: Incropera; rho_e: Matula 1979 (J. Phys. Chem. Ref. Data 8)
    'copper': dict(
        k=_table([300, 400, 600, 800, 1000, 1200, 1358], [401, 393, 379, 366, 352, 339, 330]),
        rho_e=_table([273, 293, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300],
                     [1.543, 1.678, 1.725, 2.402, 3.090, 3.792, 4.514, 5.262, 6.041, 6.858, 7.717,
                      8.626, 9.592], UOHM_CM),
        density=8960.0, T_max=473.15, emissivity=0.1, cost_per_m3=1.1e5, coolable=True),
    'aluminum': dict(
        k=_table([300, 400, 600, 800, 933], [237, 240, 231, 218, 218]),
        rho_e=_table([293, 400, 500, 600, 700, 800, 900], [2.65, 3.87, 4.99, 6.13, 7.35, 8.70, 10.18], UOHM_CM),
        density=2700.0, T_max=423.15, emissivity=0.1, cost_per_m3=1.4e4, coolable=True),
    # AISI 304. k: Incropera Table A.1; rho_e: typical 304 data (72 micro-ohm cm at 20 C, ~121 at
    # 1000 C; weakly T-dependent, ~40x copper at room temperature). T_max: PLACEHOLDER service limit
    # in argon (strength / creep, not scaling)
    'stainless': dict(
        k=_table([300, 400, 600, 800, 1000, 1200, 1500], [14.9, 16.6, 19.8, 22.6, 25.4, 28.0, 31.7]),
        rho_e=_table([293, 373, 473, 673, 873, 1073, 1273, 1473], [72, 78, 85, 98, 108, 116, 121, 125], UOHM_CM),
        density=7900.0, T_max=1073.15, emissivity=0.35, cost_per_m3=5.0e4, coolable=True),
}


def k_argon(T):
    """Argon thermal conductivity [W/m-K] (power-law fit: 0.0177 at 300 K, 0.043 at 1000 K)."""
    return 0.0177 * (np.asarray(T, float) / 300.0) ** 0.73


LEAD_DEFAULTS = {
    'zones': ['graphite', 'tungsten', 'copper'],   # hot -> cold; any MATERIALS names, 1-4 zones
    'n_leads': 2,                  # leads per heater (supply + return)
    # design input: None -> the YAML (block T_min / swing, heater power from the YAML heater
    # section, t_top); or a design row from the optimizer log (e.g. best[0]) -> its t_top, T_min,
    # T_max, block height, heater power and number of heaters
    'design': None,
    'P_heater': None,              # W override of the heater power (current I = P / heater.voltage)
    'voltage': None,               # V override of heater.voltage
    'L': None,                     # m lead length in the insulation; None -> tank.t_top
    # temperatures [K]; None -> from the design
    'T_hot_charge': None,          # lead hot end while charging; None -> heater.max_temperature
    'T_ins_hot': None,             # Thermax hot (bottom) face, both states; None -> block T_max
    'T_idle': None,                # lead hot end, no current; None -> block mean
    'T_ins_cold': 80.0 + C2K,      # Thermax top face (roof insulation -> headspace)
    'T_term': 40.0 + C2K,          # water-cooled terminal clamp at the top face (passive last zone)
    'T_cool': 40.0 + C2K,          # water-cooled last zone ('cooled')
    'gap': 0.005,                  # m radial gap lead -> bore in the insulation (argon)
    # radial conduction into the Thermax, in series with the gap: from the bore to r_far, where the
    # Thermax is at its undisturbed temperature. None -> the bore wall itself is at the
    # undisturbed temperature (no insulation resistance: overstates the exchange of a hot lead)
    'r_far': 0.5,                  # m
    'k_gap': None,                 # W/m-K gap gas; None -> argon k(T) at the film temperature
    'eps_wall': 0.9,               # bore wall (Thermax) emissivity
    'materials': {},               # per-material overrides: {'copper': {'T_max': 523.15, 'cost_per_m3': ...}}
    'joint_cost': 0.0,             # $ per joint between zones (PLACEHOLDER)
    # every zone of a lead is at least min_zone_len long; a zone that the optimum leaves at this
    # minimum (within at_min_tol) is then tried removed (optimize_with_skips): graphite -> copper,
    # tungsten -> copper, graphite only, ...
    'min_zone_len': 0.05,          # m
    'at_min_tol': 0.003,           # m
    'd_bounds': (0.02, 0.50),      # m body (largest) diameter bounds per zone
    # joints: no necks -- the contact diameter is the smaller of the two body diameters, and the larger
    # side widens from it at no more than taper_deg (half-angle from the lead axis) to its body diameter.
    # taper_deg None -> constant diameter per zone (steps at the joints)
    'taper_deg': 60.0,
    'sub_pts': 4,                  # integration sub-points per segment / control volume
    'penalty_per_K': 1.0e3,        # $ per K over a zone's T_max
    'T_tol': 1.0,                  # K over a limit still counted feasible (optima ride on the limits)
    # idle disconnect: with a material name (e.g. 'graphite'), the joint on the cold side of that zone is
    # opened while idle (no current: hold + discharge) and closed for charging. Open, the faces exchange
    # only by radiation (parallel faces) and argon conduction across break_gap. None = always connected
    'idle_break_after': None,
    'break_gap': 0.01,             # m open gap
    'break_cost': 0.0,             # $ per lead for the disconnect mechanism (PLACEHOLDER)
    'nodes': 80,                   # 1-D grid (lead cost converged to <0.1% at 60)
    'de': {'popsize': 12, 'maxiter': 150, 'tol': 1e-7, 'seed': 0},   # ~2-3 min per cooling mode
}


# --------------------------------------------------------------------------
# context: design, operating states, economics
# --------------------------------------------------------------------------
def lead_context(cfg, opt, inputs=None):
    """Everything the lead model needs for one system design: cfg = the YAML config (dict),
    opt = optimizer inputs (cost weights; scenario applied), inputs = LEAD_INPUTS overrides."""
    from . import optimize as O
    li = O.merge(LEAD_DEFAULTS, inputs or {})
    opt = O.apply_scenario(O.merge(O.OPT_DEFAULTS, opt or {}))
    eco, cost, par = opt['economics'], opt['costs'], opt['parasitics']
    h = cfg['heater']
    st = cfg['storage']
    row = li['design']
    if row:
        d = row['design']
        T_min, T_max = row['T_min_C'] + C2K, d['T_max_C'] + C2K
        H = d['H_block']
        t_top = d['t_top']
        P = (row.get('heater') or {}).get('P_heater')
        N = row.get('N_h', row.get('N_ps'))           # heaters
        src = 'optimizer design row'
    else:
        T_min = float(st['T_min'])
        T_max = T_min + float(st['block_temp_range'])
        H = float(st.get('block_height') or st['L'])
        t_top = float(cfg['tank']['t_top'])
        P, N = None, None
        src = 'YAML'
    P = float(li['P_heater'] or P or heater_max_power(h, H))
    V = float(li['voltage'] or h.get('voltage') or 0.0)
    if V <= 0:
        raise ValueError("give the heater voltage (heater.voltage or LEAD_INPUTS['voltage'])")
    el = heater_element(dict(h, max_power=P, voltage=V), H, T_min, P) if h.get('type') == 'tube' else None

    for name in li['zones']:
        if name not in MATERIALS:
            raise KeyError(f"zones: unknown material {name!r}; MATERIALS has {sorted(MATERIALS)}")
    mats = {}
    for name in MATERIALS:                 # every material, so any zone set can be run on this context
        m = dict(MATERIALS[name], **(li['materials'].get(name) or {}))
        if m['cost_per_m3'] is None:          # graphite: heater graphite $/kg x density
            per_kg = cost.get('heater_graphite_per_kg') or cost.get('iso_graphite_per_kg') or \
                cfg['tank'].get('costs', {}).get('column_graphite_per_kg', [10, 10, 10])[1]
            m['cost_per_m3'] = float(per_kg) * m['density']
        mats[name] = m

    cycle_hr = 8760.0 / eco['cycles_per_year']
    charge_hr = float(opt.get('charge_hr') or h.get('charge_hr') or 30.0)
    T_idle = li['T_idle'] or 0.5 * (T_min + T_max)
    T_ins_hot = li['T_ins_hot'] or T_max
    return dict(
        li=li, mats=mats, source=src, P=P, V=V, I=P / V, N_heaters=N, element=el,
        L=float(li['L'] or t_top), H_block=H, T_min=T_min, T_max=T_max,
        charge=dict(I=P / V, T_hot=li['T_hot_charge'] or float(h['max_temperature']),
                    T_ins_hot=T_ins_hot, hours=charge_hr),
        idle=dict(I=0.0, T_hot=T_idle, T_ins_hot=T_ins_hot, hours=cycle_hr - charge_hr,
                  break_after=li['idle_break_after']),
        discharge_hr=min(float(opt.get('duration_hr') or 0.0), cycle_hr - charge_hr),
        cycles_life=eco['cycles_per_year'] * eco['lifetime_yr'],
        elec_price=eco['elec_price'], eta_in=eco['eta_in'], soft=eco['soft_cost_frac'],
        hr_per_kWth=cost['heat_rejection_per_kWth'], elec_per_kW=cost['charge_electrics_per_kW'],
        rej_kWe_per_kWth=par['heat_rejection_kWe_per_kWth'], scenario=O.scenario_name(opt),
    )


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------
def make_lead(ctx, d, fracs, cooled, zones=None, fj=None):
    """Lead from the optimizer variables: body (largest) diameters d per zone, length fractions
    fracs and joint-diameter fractions fj. Every zone gets min_zone_len; zone i then takes
    fracs[i] of the free length still left, the last zone the rest. Joint j (between zones j and
    j+1) has one contact diameter for both sides, fj[j] x the smaller of the two body diameters
    (fj None -> 1: the smaller body diameter); from each joint a zone widens at no more than
    taper_deg (half-angle from the lead axis) up to its body diameter (lead_profile).
    zones: the materials, hot -> cold (default LEAD_INPUTS zones)."""
    zones = list(zones or ctx['li']['zones'])
    L, n, mz = ctx['L'], len(zones), ctx['li']['min_zone_len']
    free = L - n * mz
    if free < 0:
        raise ValueError(f"{n} zones x min_zone_len {mz} m exceed the lead length {L:.2f} m")
    lens, rem = [], free
    for f in list(fracs)[:n - 1]:
        lens.append(mz + float(f) * rem)
        rem -= float(f) * rem
    lens.append(mz + rem)
    d = [float(v) for v in d]
    fj = [1.0] * (n - 1) if fj is None else [float(v) for v in fj]
    d_joint = [fj[j] * min(d[j], d[j + 1]) for j in range(n - 1)]
    return dict(mats=zones, lens=lens, d=d, d_joint=d_joint, cooled=bool(cooled),
                taper_deg=ctx['li'].get('taper_deg'))


def lead_from_x(ctx, x, cooled, zones):
    """make_lead from an optimizer vector [body diameters (n), length fractions (n-1)]. No necks:
    each joint's contact diameter is the smaller of the two body diameters."""
    n = len(zones)
    return make_lead(ctx, x[:n], x[n:2 * n - 1], cooled, zones)


def lead_profile(lead, x):
    """Zone index and diameter D [m] at heights x [m] (array). Constant body diameter per zone,
    narrowing at no more than taper_deg (half-angle) to each joint's contact diameter; no
    d_joint / taper_deg None -> constant diameter per zone (steps at the joints)."""
    x = np.asarray(x, float)
    lens, D = lead['lens'], np.asarray(lead['d'], float)
    n = len(lens)
    edges = np.concatenate([[0.0], np.cumsum(lens)])
    z = np.clip(np.searchsorted(edges, x, 'right') - 1, 0, n - 1)
    dj, taper = lead.get('d_joint'), lead.get('taper_deg')
    if not dj or taper is None or n == 1:
        return z, D[z]
    t2 = 2.0 * math.tan(math.radians(float(taper)))
    dj = np.asarray(dj, float)
    dL = np.where(z > 0, dj[np.maximum(z - 1, 0)], D[z])          # hot-side joint of the zone
    dR = np.where(z < n - 1, dj[np.minimum(z, n - 2)], D[z])      # cold-side joint
    Dx = np.minimum(D[z], np.minimum(dL + t2 * (x - edges[z]), dR + t2 * (edges[z + 1] - x)))
    return z, Dx


def zone_volumes(lead, n_pts=401):
    """Volume [m3] of each zone (integral of the profile area)."""
    edges = np.concatenate([[0.0], np.cumsum(lead['lens'])])
    out = []
    for i in range(len(lead['lens'])):
        if lead['lens'][i] <= 0:
            out.append(0.0)
            continue
        xs = np.linspace(edges[i], edges[i + 1], n_pts)
        xs[0] += 1e-12; xs[-1] -= 1e-12                              # stay inside the zone
        out.append(float(_trapz(math.pi * lead_profile(lead, xs)[1] ** 2 / 4, xs)))
    return out


def _overlap(a, b, s, e):
    return np.clip(np.minimum(b, e) - np.maximum(a, s), 0.0, None)


def _ins(ctx, T_hot, x):
    """Undisturbed Thermax temperature T_w(x) [K] (exact 1-D profile, k = K0 exp(B T_C)) and the
    up-fraction w(x) = R(0..x) / R(0..L) at heights x; T_hot = Thermax hot (bottom) face."""
    L, Tc = ctx['L'], ctx['li']['T_ins_cold']
    s = np.clip(np.asarray(x, float) / L, 0.0, 1.0)
    eh, ec = np.exp(THERMAX_B * (T_hot - C2K)), np.exp(THERMAX_B * (Tc - C2K))
    Tw = C2K + np.log(eh + (ec - eh) * s) / THERMAX_B          # phi = int k dT linear in x
    # int dx / k ~ int dx / exp(B T_w) = ln(e(x) / e_h) / (e_c - e_h)  ->  w = (T_hot - T_w) / (T_hot - Tc)
    w = (T_hot - Tw) / (T_hot - Tc) if abs(T_hot - Tc) > 1e-9 else s
    return Tw, w


# --------------------------------------------------------------------------
# 1-D steady solve
# --------------------------------------------------------------------------
def solve_state(ctx, lead, state, max_iter=300, tol=1e-3):
    """Steady temperature along the lead for one state (dict I, T_hot, T_ins_hot; optional
    break_after: the joint on the cold side of that material's zone is open). The cross-section
    varies along the lead (lead_profile): conduction, Joule heat and the exchange with the
    insulation are integrated over sub-points (li sub_pts per segment / control volume)."""
    li, mats = ctx['li'], ctx['mats']
    I, T_hot, T_ih = state['I'], state['T_hot'], state['T_ins_hot']
    L, gap, eps_w = ctx['L'], li['gap'], li['eps_wall']
    names, lens = lead['mats'], lead['lens']
    nz = len(names)
    edges = np.concatenate([[0.0], np.cumsum(lens)])
    cooled = lead['cooled'] and lens[-1] > 0
    x_end = edges[-2] if cooled else L
    T_end = li['T_cool'] if cooled else li['T_term']
    eps = np.array([mats[m]['emissivity'] for m in names])
    k_gas = (lambda T: np.full_like(T, li['k_gap'])) if li['k_gap'] else k_argon
    r_far = li['r_far']

    def coeffs(z, D):
        """Per-unit-length exchange coefficients at points with zone z and diameter D."""
        r = D / 2
        rb = r + gap
        cg = 2 * math.pi / np.log(rb / r)                                   # x k_gas -> W/m-K
        cr = 2 * math.pi * r * SIGMA / (1 / eps[z] + r / rb * (1 / eps_w - 1))   # x (T^4 - Tb^4) -> W/m
        ci = 2 * math.pi / np.log(np.maximum(r_far / rb, 1.0 + 1e-6)) if r_far else None   # x k_thermax
        return cg, cr, ci

    def exch(cf, T, Tw, Tb):
        """Exchange lead -> far-field insulation per unit length: gap (gas + radiation, wall at
        Tb) in series with radial Thermax conduction to r_far. -> (q [W/m], dq/dT, new Tb)."""
        cg, cr, ci = cf
        kg = k_gas(0.5 * (T + Tb))
        hg = cg * kg + cr * (T * T + Tb * Tb) * (T + Tb)
        hd = cg * kg + 4 * cr * T ** 3
        if ci is None:
            return cg * kg * (T - Tw) + cr * (T ** 4 - Tw ** 4), hd, Tw
        hi = ci * thermax_tcx(0.5 * (Tb + Tw))
        h = hg * hi / (hg + hi)
        q = h * (T - Tw)
        Tbn = np.clip(Tw + q / hi, np.minimum(T, Tw), np.maximum(T, Tw))    # wall between lead and far field
        return q, hd * hi / (hd + hi), 0.5 * (Tb + Tbn)

    def prop(key, z, T, masks):
        out = np.empty_like(T)
        for i, m in masks.items():
            out[m] = mats[names[i]][key](T[m])
        return out

    out = dict(lead=lead, state=state, x_end=x_end, cooled=cooled)
    if x_end < 1e-3:                       # cooled zone at the hot face: no steady answer worth having
        out.update(ok=False, reason='cooled zone reaches the hot face')
        return out

    N, M = int(li['nodes']), int(li.get('sub_pts', 4))
    x = np.linspace(0.0, x_end, N + 1)
    dx = x[1] - x[0]
    a, b = np.maximum(x - dx / 2, 0.0), np.minimum(x + dx / 2, x_end)       # control volumes
    fr = (np.arange(M) + 0.5) / M
    xc = a[:, None] + (b - a)[:, None] * fr[None, :]                         # CV sub-points
    wc = np.repeat(((b - a) / M)[:, None], M, 1)
    xs = x[:-1, None] + dx * fr[None, :]                                     # segment sub-points
    ws = dx / M
    zc, Dc = lead_profile(lead, xc)
    zs, Ds = lead_profile(lead, xs)
    Ac, As = math.pi * Dc ** 2 / 4, math.pi * Ds ** 2 / 4
    cfc = coeffs(zc, Dc)
    Twc, wcf = _ins(ctx, T_ih, xc)
    Tw, w = _ins(ctx, T_ih, x)                                               # at the nodes (plots)
    live = sorted(set(np.unique(zc)) | set(np.unique(zs)))
    mc = {i: zc == i for i in live if (zc == i).any()}
    ms = {i: zs == i for i in live if (zs == i).any()}
    brk = None                             # open joint: (segment index, face area, emissivity, x)
    after = state.get('break_after')
    if after in names:
        i = names.index(after)
        if i < nz - 1 and lens[i] > 0 and edges[i + 1] <= x_end + 1e-9:
            kseg = int(min(max(np.searchsorted(x, edges[i + 1], 'right') - 1, 0), N - 1))
            dj = (lead.get('d_joint') or [None] * (nz - 1))[i] or min(lead['d'][i], lead['d'][i + 1])
            brk = (kseg, math.pi * dj ** 2 / 4, 1.0 / (1.0 / eps[i] + 1.0 / eps[i + 1] - 1.0), float(edges[i + 1]))

    def seg_G(T):
        """Segment conductances [W/K] at node temperatures T (an open joint in series where set)."""
        Tsub = T[:-1, None] + fr[None, :] * (T[1:] - T[:-1])[:, None]
        G = 1.0 / (ws / (prop('k', zs, Tsub, ms) * As)).sum(1)
        if brk:
            k, Ab, eb, _ = brk
            Ta, Tc_ = T[k], T[k + 1]
            Gb = Ab * (float(k_gas(np.array(0.5 * (Ta + Tc_)))) / li['break_gap']
                       + SIGMA * eb * (Ta * Ta + Tc_ * Tc_) * (Ta + Tc_))
            G[k] = 1.0 / (1.0 / G[k] + 1.0 / Gb)
        return G

    T = T_hot + (T_end - T_hot) * x / x_end
    Tb = Twc.copy()                        # bore wall temperature at the CV sub-points
    T_runaway = max(T_hot, max(mats[names[i]]['T_max'] for i in live)) + 1500.0
    ok = False
    for it in range(max_iter):
        G = seg_G(T)
        TT = np.repeat(T[:, None], M, 1)
        if I:
            r0 = prop('rho_e', zc, TT, mc)
            S = I * I * (wc * r0 / Ac).sum(1)
            dS = I * I * (wc * (prop('rho_e', zc, TT + 1.0, mc) - r0) / Ac).sum(1)
        else:
            S = dS = np.zeros_like(T)
        q_s, dq_s, Tb = exch(cfc, TT, Twc, Tb)
        Q, dQ = (wc * q_s).sum(1), (wc * dq_s).sum(1)
        # Newton on the sources (Joule, exchange), Picard on k(T)
        diag = np.zeros_like(T)
        diag[1:] += G; diag[:-1] += G
        diag += dQ
        dS = np.minimum(dS, 0.5 * diag)        # keep the diagonal positive (runaway shows as T rising)
        diag -= dS
        rhs = S - dS * T - Q + dQ * T
        ab = np.zeros((3, N + 1))
        ab[0, 1:] = -G; ab[1] = diag; ab[2, :-1] = -G
        ab[1, 0], ab[0, 1], rhs[0] = 1.0, 0.0, T_hot
        ab[1, -1], ab[2, -2], rhs[-1] = 1.0, 0.0, T_end
        try:
            Tn = solve_banded((1, 1), ab, rhs)
        except (np.linalg.LinAlgError, ValueError):
            break
        if not np.all(np.isfinite(Tn)):
            break
        dT = np.abs(Tn - T).max()
        T = np.clip(T + np.clip(Tn - T, -300.0, 300.0), 1.0, None)          # step limit
        if T.max() > T_runaway:
            break
        if dT < tol:
            ok = True
            break
    if not ok:
        out.update(ok=False, reason='no steady solution (thermal runaway or no convergence)', T=T, x=x)
        return out

    # converged: exact sources and boundary flows (bore wall Tb from the last iteration)
    G = seg_G(T)
    TT = np.repeat(T[:, None], M, 1)
    S = I * I * (wc * prop('rho_e', zc, TT, mc) / Ac).sum(1) if I else np.zeros_like(T)
    q_s, _, Tb = exch(cfc, TT, Twc, Tb)
    q = (wc * q_s).sum(1)
    q_up = float((wc * q_s * wcf).sum())                         # deck share of the heat into the insulation
    Q_hot = G[0] * (T[0] - T[1]) + q[0] - S[0]
    Q_end = G[-1] * (T[-2] - T[-1]) + S[-1] - q[-1]
    Q_J = float(S.sum())
    Q_rej, Q_J_cool, absorb, absorb_w = Q_end, 0.0, 0.0, 0.0
    cool = None
    if cooled:                             # water-cooled last zone at T_cool
        i, Tc = nz - 1, li['T_cool']
        xk = np.linspace(edges[-2], L, 81)
        xk[0] += 1e-12
        zk, Dk = lead_profile(lead, xk)
        cfk = coeffs(zk, Dk)
        Twk, wk = _ins(ctx, T_ih, xk)
        Tc_arr, Tbk = np.full_like(xk, Tc), Twk.copy()
        for _ in range(60):                # bore wall of the cooled zone (fixed point)
            qk, _, Tbn = exch(cfk, Tc_arr, Twk, Tbk)
            if np.abs(Tbn - Tbk).max() < 1e-3:
                break
            Tbk = Tbn
        qa = -qk                           # absorbed by the cooled zone [W/m]
        absorb = float(_trapz(qa, xk))
        absorb_w = float(_trapz(qa * wk, xk))
        jk = I * I * mats[names[i]]['rho_e'](Tc) / (math.pi * Dk ** 2 / 4)   # W/m
        Q_J_cool = float(_trapz(jk, xk))
        Q_rej = Q_end + Q_J_cool + absorb
        cool = dict(x=xk, q_ex_per_m=-qa, joule_per_m=jk, Tw=Twk)
    Q_J_tot = Q_J + Q_J_cool
    q_tot = float(q.sum())
    drain = Q_hot - (q_tot - q_up) + (absorb - absorb_w)
    loss = Q_rej + q_up - absorb_w                               # = Q_J_tot + drain
    T_zone = []
    for i in range(nz):
        if lens[i] <= 0:
            T_zone.append(None)
        elif cooled and i == nz - 1:
            T_zone.append(li['T_cool'])
        else:
            m = (x >= edges[i] - 1e-9) & (x <= edges[i + 1] + 1e-9)
            T_zone.append(float(T[m].max()) if m.any() else float(np.interp(edges[i], x, T)))
    viol = sum(max(0.0, Tz - ctx['mats'][names[i]]['T_max']) for i, Tz in enumerate(T_zone) if Tz is not None)
    # joints: temperature at each zone boundary vs the lower T_max of the two materials (the colder side)
    T_joint = []
    for j in range(1, nz):
        if lens[j - 1] <= 0 or lens[j] <= 0:
            continue
        Tj = float(np.interp(edges[j], x, T)) if edges[j] <= x_end + 1e-12 else li['T_cool']
        lim = min(ctx['mats'][names[j - 1]]['T_max'], ctx['mats'][names[j]]['T_max'])
        T_joint.append(dict(between=(names[j - 1], names[j]), x=float(edges[j]), T=Tj, limit=lim))
        viol += max(0.0, Tj - lim)
    cv = b - a                             # control-volume lengths: per-metre profiles = S / cv, q / cv
    out.update(ok=True, iters=it + 1, x=x, T=T, Tw=Tw, w=w, q_ex=q, S=S, cv=cv, joule_per_m=S / cv,
               q_ex_per_m=q / cv, cool=cool, Q_hot=Q_hot, Q_end=Q_end,
               Q_J=Q_J_tot, Q_rej=Q_rej, Q_cool_absorb=absorb, drain=drain, loss=loss,
               balance=Q_J_tot + drain - loss, Q_ins_up=q_up - absorb_w,
               T_zone_max=T_zone, T_joint=T_joint, viol_K=viol, open_joint=brk[3] if brk else None,
               R_lead=Q_J_tot / I ** 2 if I else None)
    return out


# --------------------------------------------------------------------------
# cost
# --------------------------------------------------------------------------
def evaluate(ctx, lead):
    """Lifetime cost of one lead (charge + idle states). -> dict (cost per lead, $)."""
    ch = solve_state(ctx, lead, ctx['charge'])
    idle = solve_state(ctx, lead, ctx['idle'])
    names, lens = lead['mats'], lead['lens']
    vols = zone_volumes(lead)
    mat_cost = [v * ctx['mats'][m]['cost_per_m3'] for v, m in zip(vols, names)]
    masses = [v * ctx['mats'][m]['density'] for v, m in zip(vols, names)]
    n_present = sum(1 for v in lens if v > 0)
    joints = max(0, n_present - 1) * ctx['li']['joint_cost']
    res = dict(lead=lead, charge=ch, idle=idle, vol_m3=vols, mass_kg=masses, mat_cost=mat_cost, joints=joints)
    if not (ch['ok'] and idle['ok']):
        res.update(ok=False, objective=1e9, reason=ch.get('reason') or idle.get('reason'))
        return res
    soft = 1 + ctx['soft']
    P_rej_peak = max(ch['Q_rej'], idle['Q_rej'], 0.0)
    brk_cost = ctx['li']['break_cost'] if idle.get('open_joint') is not None else 0.0
    capex_parts = dict(material=sum(mat_cost) * soft, joints=(joints + brk_cost) * soft,
                       heat_rejection=ctx['hr_per_kWth'] * P_rej_peak / 1e3 * soft,
                       supply=ctx['elec_per_kW'] * ch['Q_J'] / 1e3 * soft)
    t_ch, t_id = ctx['charge']['hours'] * 3600, ctx['idle']['hours'] * 3600
    E_J = ch['Q_J'] * t_ch
    E_drain = ch['drain'] * t_ch + idle['drain'] * t_id
    E_rej = max(ch['Q_rej'], 0) * t_ch + max(idle['Q_rej'], 0) * t_id
    E_par = ctx['rej_kWe_per_kWth'] * E_rej
    price = ctx['elec_price'] / 3.6e9 * ctx['cycles_life']                 # $ per J per cycle, lifetime
    energy_parts = dict(joule=price * E_J, storage_heat=price * E_drain / ctx['eta_in'],
                        rejection_parasitic=price * E_par)
    capex, energy = sum(capex_parts.values()), sum(energy_parts.values())
    viol = ch['viol_K'] + idle['viol_K']
    res.update(ok=True, capex=capex, capex_parts=capex_parts, energy_cost=energy, energy_parts=energy_parts,
               total=capex + energy, viol_K=viol, feasible=viol <= ctx['li'].get('T_tol', 1.0),
               objective=capex + energy + ctx['li']['penalty_per_K'] * viol,
               P_rej_peak=P_rej_peak, E_cycle_MWh=dict(joule=E_J / 3.6e9, drain=E_drain / 3.6e9,
                                                        rejected=E_rej / 3.6e9))
    return res


def baseline_lead(ctx, d=None):
    """Graphite only, the element's cross-section (the system model's assumption: the element
    continues up through the top insulation), passive, terminal at the top face."""
    if d is None:
        A = ctx['element']['A_cross'] if ctx['element'] else math.pi * 0.1 ** 2 / 4
        d = math.sqrt(4 * A / math.pi)
    return dict(mats=['graphite'], lens=[ctx['L']], d=[d], d_joint=[], cooled=False, taper_deg=None)


# --------------------------------------------------------------------------
# optimizer
# --------------------------------------------------------------------------
def optimize_leads(ctx, cooled, zones=None, verbose=True, **de_kw):
    """Differential evolution over the zone body diameters and zone lengths (no necks: each joint's
    contact diameter is the smaller body diameter) for one set of zones (hot -> cold; default LEAD_INPUTS zones) and cooling mode, then a
    bounded Nelder-Mead polish. -> evaluate() dict of the best lead (+ 'x', 'nfev', 'at_min':
    zones left at min_zone_len)."""
    from scipy.optimize import differential_evolution, minimize
    zones = list(zones or ctx['li']['zones'])
    nz = len(zones)
    dlo, dhi = ctx['li']['d_bounds']
    bounds = [(dlo, dhi)] * nz + [(0.0, 1.0)] * (nz - 1)

    def f(x):
        return evaluate(ctx, lead_from_x(ctx, x, cooled, zones))['objective']

    kw = dict(ctx['li']['de'], **de_kw)
    r = differential_evolution(f, bounds, init='sobol', polish=False, **kw)
    p = minimize(f, r.x, method='Nelder-Mead', bounds=bounds,
                 options=dict(xatol=1e-5, fatol=1e-2, maxiter=4000, maxfev=4000))
    x = p.x if p.fun < r.fun else r.x
    best = evaluate(ctx, lead_from_x(ctx, x, cooled, zones))
    lim = ctx['li']['min_zone_len'] + ctx['li']['at_min_tol']
    at_min = [m for m, ln in zip(zones, best['lead']['lens']) if ln <= lim] if nz > 1 else []
    best.update(x=list(map(float, x)), nfev=r.nfev + p.nfev, at_min=at_min)
    if verbose:
        print(f"  {' / '.join(zones):<30s} {'cooled' if cooled else 'passive'}: "
              + (f"${best['total']:,.0f} per lead" if best.get('ok') else 'no solution')
              + ('' if best.get('feasible') else '  (T limit exceeded)')
              + (f"  [{', '.join(at_min)} at the {ctx['li']['min_zone_len'] * 100:.0f} cm minimum]" if at_min else ''))
    return best


def optimize_with_skips(ctx, cooled, zones=None, cache=None, verbose=True, **de_kw):
    """optimize_leads on `zones` (hot -> cold; default the LEAD_INPUTS zones); whenever an
    optimum leaves zones at min_zone_len, also optimize the leads with those zones removed (each
    one, and all of them), and so on, down to a single material. A water-cooled run needs a
    coolable last zone. cache: dict {(zones, cooled): result} shared between calls, so a zone set
    reached from two starting sets (e.g. graphite only) is optimized once.
    -> (best result, {zones tuple: result} of every set tried)."""
    from itertools import combinations
    cache = {} if cache is None else cache
    tried, todo = {}, [tuple(zones or ctx['li']['zones'])]
    while todo:
        z = todo.pop(0)
        if z in tried or not z:
            continue
        if cooled and not ctx['mats'][z[-1]].get('coolable'):
            continue
        if (z, bool(cooled)) in cache:
            r = cache[(z, bool(cooled))]
            if verbose:
                print(f"  {' / '.join(z):<30s} {'cooled' if cooled else 'passive'}: (already run) "
                      + (f"${r['total']:,.0f} per lead" if r.get('ok') else 'no solution'))
        else:
            r = optimize_leads(ctx, cooled, zones=z, verbose=verbose, **de_kw)
            cache[(z, bool(cooled))] = r
        tried[z] = r
        for k in range(1, len(r['at_min']) + 1):
            for drop in combinations(r['at_min'], k):
                rest = tuple(m for m in z if m not in drop)
                if rest and rest not in tried:
                    todo.append(rest)
    if not tried:
        raise ValueError("no zone set to run: a water-cooled run needs a coolable last zone (copper, aluminum)")
    ok = [r for r in tried.values() if r.get('ok')]
    best = min(ok or tried.values(), key=lambda r: (not r.get('feasible', False), r['objective']))
    return best, tried


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
def describe(ctx, res, label=''):
    """Text summary of one evaluated lead."""
    lead, ch, idle = res['lead'], res['charge'], res['idle']
    L = []
    mode = 'cooled last zone' if lead['cooled'] else 'passive'
    L.append(f"{label} ({mode}): ${res.get('total', float('nan')):,.0f} per lead, "
             f"${res.get('total', float('nan')) * ctx['li']['n_leads']:,.0f} per heater"
             + ('' if res.get('feasible', False) else '  ** NOT FEASIBLE **'))
    if not res.get('ok'):
        L.append(f"  {res.get('reason')}")
        return '\n'.join(L)
    x0 = 0.0
    for i, (m, ln, d) in enumerate(zip(lead['mats'], lead['lens'], lead['d'])):
        if ln <= 0:
            continue
        A = math.pi * d * d / 4
        cool = ' water-cooled' if lead['cooled'] and i == len(lead['mats']) - 1 else ''
        Tz = ch['T_zone_max'][i]
        L.append(f"  {m:<10s} x {x0:5.2f}-{x0 + ln:5.2f} m  D {d*1e3:6.1f} mm (body)  J {ctx['I']/A/1e6:5.2f} A/mm2  "
                 f"{res['mass_kg'][i]:7.1f} kg  ${res['mat_cost'][i]:>8,.0f}  T max {Tz - C2K:6.0f} C "
                 f"(limit {ctx['mats'][m]['T_max'] - C2K:.0f}){cool}")
        x0 += ln
    if lead.get('d_joint') and lead.get('taper_deg') is not None:
        L.append("  joint contact diameters: " + ", ".join(
            f"{lead['mats'][j]}/{lead['mats'][j + 1]} {dj * 1e3:.0f} mm ({ctx['I'] / (math.pi * dj * dj / 4) / 1e6:.2f} A/mm2)"
            for j, dj in enumerate(lead['d_joint'])) + f"; taper <= {lead['taper_deg']:g} deg")
    if ch.get('T_joint'):
        L.append("  joints (charge): " + ", ".join(
            f"{j['between'][0]}/{j['between'][1]} at {j['x']:.2f} m {j['T'] - C2K:.0f} C (limit {j['limit'] - C2K:.0f})"
            for j in ch['T_joint']))
    if idle.get('open_joint') is not None:
        L.append(f"  idle: joint at {idle['open_joint']:.2f} m OPEN ({ctx['li']['break_gap'] * 1e3:.0f} mm gap: "
                 f"radiation + argon); closed while charging")
    for nm, s in (('charge', ch), ('idle', idle)):
        L.append(f"  {nm:<6s}: Joule {s['Q_J']/1e3:6.2f} kW, from storage at hot end {s['Q_hot']/1e3:6.2f} kW, "
                 f"rejected (cold end{' + cooling' if lead['cooled'] else ''}) {s['Q_rej']/1e3:6.2f} kW, "
                 f"to insulation (deck share) {s['Q_ins_up']/1e3:5.2f} kW; storage drain {s['drain']/1e3:6.2f} kW")
    L.append(loss_line(ctx, res))
    R = ch['R_lead']
    L.append(f"  lead R {R*1e6:.1f} uOhm, drop {ctx['I']*R:.2f} V of {ctx['V']:g} V "
             f"({ctx['I']*R/ctx['V']*100:.1f}%)")
    cp, ep = res['capex_parts'], res['energy_parts']
    L.append("  capex " + ", ".join(f"{k} ${v:,.0f}" for k, v in cp.items()) + f"  = ${res['capex']:,.0f}")
    L.append("  lifetime energy " + ", ".join(f"{k} ${v:,.0f}" for k, v in ep.items()) + f"  = ${res['energy_cost']:,.0f}")
    return '\n'.join(L)


def losses(ctx, res):
    """Energy lost through one lead, per phase, as % of the heater power P (and per cycle as % of
    the heater's charge energy P x charge_hr). Lost = Joule heat + heat conducted out of the
    storage (net; < 0 while charging when Joule heat flows down into the block) = heat leaving
    the system (cooling water / terminal + the deck share of the heat put into the insulation).
    Hold and discharge are the same no-current state; they are split by hours (discharge_hr =
    the optimizer duration_hr)."""
    P = ctx['P']
    ch, idle = res['charge'], res['idle']
    t_ch, t_id = ctx['charge']['hours'], ctx['idle']['hours']
    t_dis = ctx.get('discharge_hr', 0.0)
    rows = {
        'charging': dict(hours=t_ch, joule=ch['Q_J'], conduction=ch['drain']),
        'hold': dict(hours=t_id - t_dis, joule=0.0, conduction=idle['drain']),
        'discharging': dict(hours=t_dis, joule=0.0, conduction=idle['drain']),
    }
    E = 0.0
    for r in rows.values():
        r['lost'] = r['joule'] + r['conduction']
        r.update({k + '_pct': 100 * r[k] / P for k in ('joule', 'conduction', 'lost')})
        r['E_kWh'] = r['lost'] * r['hours'] / 1e3
        E += r['E_kWh']
    return dict(rows=rows, P=P, E_cycle_kWh=E, E_cycle_pct=100 * E * 1e3 / (P * t_ch),
                mean_pct=100 * E * 1e3 / (P * (t_ch + t_id)))


def loss_line(ctx, res):
    """One-line summary of losses()."""
    lo = losses(ctx, res)
    r = lo['rows']
    c = r['charging']
    return (f"  losses, % of heater power {lo['P'] / 1e3:,.0f} kW: charging {c['lost_pct']:.2f}% "
            f"(Joule {c['joule_pct']:.2f}% + conduction from storage {c['conduction_pct']:+.2f}%), "
            f"hold {r['hold']['lost_pct']:.2f}% ({r['hold']['hours']:.0f} h), discharging "
            f"{r['discharging']['lost_pct']:.2f}% ({r['discharging']['hours']:.0f} h); per cycle "
            f"{lo['E_cycle_kWh']:,.0f} kWh = {lo['E_cycle_pct']:.2f}% of the heater's charge energy")


def loss_table(results):
    """Text table of losses() for {label: (ctx, res)}: per phase, % of heater power."""
    out = [f"{'':<48s}{'charging':>28s}{'hold':>10s}{'discharging':>13s}{'per cycle':>24s}",
           f"{'% of heater power':<48s}{'Joule':>9s}{'conduction':>11s}{'total':>8s}{'':>10s}{'':>13s}"
           f"{'kWh':>10s}{'% of charge':>14s}"]
    for lab, (ctx, res) in results.items():
        if not res.get('ok'):
            out.append(f"{lab:<48s}  no solution")
            continue
        lo = losses(ctx, res)
        r = lo['rows']
        c = r['charging']
        out.append(f"{lab:<48s}{c['joule_pct']:9.2f}{c['conduction_pct']:11.2f}{c['lost_pct']:8.2f}"
                   f"{r['hold']['lost_pct']:10.2f}{r['discharging']['lost_pct']:13.2f}"
                   f"{lo['E_cycle_kWh']:10,.0f}{lo['E_cycle_pct']:13.2f}%")
    return '\n'.join(out)


def plot_lead(ctx, res, axes=None, title=''):
    """Temperature along the lead, charge and idle, with the undisturbed insulation."""
    import matplotlib.pyplot as plt
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    lead = res['lead']
    edges = np.concatenate([[0.0], np.cumsum(lead['lens'])])
    shades = {'graphite': '#d9d9d9', 'tungsten': '#c6dbef', 'molybdenum': '#dadaeb', 'copper': '#fdd0a2',
              'aluminum': '#e5f5e0'}
    for ax, nm in zip(axes, ('charge', 'idle')):
        s = res[nm]
        for i, m in enumerate(lead['mats']):
            if lead['lens'][i] > 0:
                ax.axhspan(edges[i], edges[i + 1], color=shades.get(m, '#eeeeee'), alpha=0.6, lw=0)
                ax.text(0.98, 0.5 * (edges[i] + edges[i + 1]), m + (' (cooled)' if lead['cooled'] and
                        i == len(lead['mats']) - 1 else ''), transform=ax.get_yaxis_transform(),
                        ha='right', va='center', fontsize=8, color='#444444')
        if s.get('ok'):
            ax.plot(s['T'] - C2K, s['x'], color='#1f4e9e', lw=2, label='lead')
            ax.plot(s['Tw'] - C2K, s['x'], color='#888888', lw=1.5, ls='--', label='insulation (undisturbed)')
            if lead['cooled'] and lead['lens'][-1] > 0:
                ax.plot([ctx['li']['T_cool'] - C2K] * 2, [edges[-2], edges[-1]], color='#1f4e9e', lw=2)
        ax.set_title(f"{nm}" + (f", {ctx['I']/1e3:.1f} kA" if nm == 'charge' else ', no current'), fontsize=10)
        ax.set_xlabel('temperature [C]')
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('height above the hot face of the top insulation [m]')
    axes[0].legend(loc='lower left', fontsize=8, frameon=False)
    if title:
        axes[0].figure.suptitle(title, fontsize=11)
    return axes


MAT_COLORS = {'graphite': '#5a5a5a', 'tungsten': '#7f9cc0', 'molybdenum': '#9d8fc4', 'copper': '#d0782a',
              'aluminum': '#a9b8c2', 'stainless': '#9aa5a8'}


def _with_cool(s, key):
    """Height and a per-metre profile along the whole lead, incl. a water-cooled last zone."""
    x, v = s['x'], s[key]
    if s.get('cool') is not None:
        x, v = np.concatenate([x, s['cool']['x']]), np.concatenate([v, s['cool'][key]])
    return x, v


def plot_lead_section(ctx, res, title=''):
    """One lead: to-scale cross-section through the top insulation (material zones, bar
    diameters, gap, the Thermax around it) beside the temperature and the heat per metre along
    it (Joule heating and exchange with the insulation), all on the same height axis."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    if res['charge'].get('ok') and 'joule_per_m' not in res['charge']:
        res = evaluate(ctx, res['lead'])          # result from an older version of this module: re-solve
    lead, ch, idle = res['lead'], res['charge'], res['idle']
    li = ctx['li']
    L, gap = ctx['L'], li['gap']
    names, lens, d = lead['mats'], lead['lens'], lead['d']
    edges = np.concatenate([[0.0], np.cumsum(lens)])
    present = [i for i in range(len(names)) if lens[i] > 0]
    cooled_i = len(names) - 1 if lead['cooled'] and lens[-1] > 0 else None
    r_ins = max(li['r_far'] or 0.0, max(d[i] for i in present) / 2 + gap + 0.15)
    pad = 0.12 * L
    T_ins_hot = ctx['charge']['T_ins_hot']

    fig = plt.figure(figsize=(14, 5.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.0, 1.0], wspace=0.10)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharey=ax0)
    ax2 = fig.add_subplot(gs[2], sharey=ax0)

    # -- cross-section, to scale --
    ax0.add_patch(Rectangle((-r_ins, 0), 2 * r_ins, L, facecolor='#efe4cc', edgecolor='#b9a67c',
                            hatch='..', lw=0.8))
    ax0.text(-r_ins + 0.02, 0.02, 'Thermax', fontsize=7.5, color='#7a6a45', va='bottom')
    ax0.add_patch(Rectangle((-r_ins, -pad), 2 * r_ins, pad, facecolor='#f3c9a8', edgecolor='#c08a5a', lw=0.8))
    ax0.text(0, -pad / 2, f"block / heater cavity: Thermax bottom face {T_ins_hot - C2K:.0f} C",
             ha='center', va='center', fontsize=7.5, color='#5a3a1a')
    ax0.add_patch(Rectangle((-r_ins, L), 2 * r_ins, pad, facecolor='#e8eef4', edgecolor='#9fb0c0', lw=0.8))
    T_top = li['T_cool'] if cooled_i is not None else li['T_term']
    ax0.text(0, L + pad / 2, f"terminal {T_top - C2K:.0f} C; Thermax top face {li['T_ins_cold'] - C2K:.0f} C",
             ha='center', va='center', fontsize=7.5, color='#2a3a4a')
    from matplotlib.patches import Polygon
    for i in present:
        y0, h, r = edges[i], lens[i], d[i] / 2
        yy = np.linspace(y0 + 1e-9, y0 + h - 1e-9, 121)
        rr = lead_profile(lead, yy)[1] / 2
        ax0.add_patch(Polygon(np.c_[np.r_[rr + gap, -(rr + gap)[::-1]], np.r_[yy, yy[::-1]]],
                              facecolor='white', lw=0))                                        # bore
        ax0.add_patch(Polygon(np.c_[np.r_[rr, -rr[::-1]], np.r_[yy, yy[::-1]]],
                              facecolor=MAT_COLORS.get(names[i], '#999999'), edgecolor='black', lw=0.8,
                              hatch='////' if i == cooled_i else None))
        lab = (f"{names[i]}{' (water-cooled)' if i == cooled_i else ''}\n"
               f"D {d[i] * 1e3:.0f} mm (body), L {h:.2f} m\n"
               f"{ctx['I'] / (math.pi * r * r) / 1e6:.2f} A/mm2, {res['mass_kg'][i]:.0f} kg")
        ax0.annotate(lab, xy=(r, y0 + h / 2), xytext=(r + gap + 0.05, y0 + h / 2), va='center', ha='left',
                     fontsize=7.5, arrowprops=dict(arrowstyle='-', color='#333333', lw=0.6),
                     bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#999999', lw=0.5, alpha=0.9))
    for j, dj in enumerate(lead.get('d_joint') or []):
        if lead.get('taper_deg') is not None and lens[j] > 0 and lens[j + 1] > 0:
            ax0.annotate(f"joint {dj * 1e3:.0f} mm", xy=(-dj / 2, edges[j + 1]), xytext=(-r_ins + 0.02, edges[j + 1]),
                         va='center', ha='left', fontsize=7, color='#333333',
                         arrowprops=dict(arrowstyle='-', color='#666666', lw=0.5))
    ax0.axvline(0, color='#333333', lw=0.5, ls=(0, (6, 3, 1, 3)))
    ax0.set_xlim(-r_ins, r_ins)
    ax0.set_ylim(-pad, L + pad)
    ax0.set_aspect('equal', adjustable='box')
    ax0.set_xlabel('radius [m]')
    ax0.set_ylabel('height above the bottom face of the top insulation [m]')
    ax0.set_title(f"Section, to scale (gap {gap * 1e3:.0f} mm; Thermax undisturbed at r_far "
                  f"{li['r_far'] or 0:.2f} m)", fontsize=9.5)

    # -- zone bands on the profile panels --
    for i in present:
        for ax in (ax1, ax2):
            ax.axhspan(edges[i], edges[i + 1], color=MAT_COLORS.get(names[i], '#999999'), alpha=0.10, lw=0)
            ax.axhline(edges[i + 1], color='#999999', lw=0.6)
        if i != cooled_i:
            T_lim = ctx['mats'][names[i]]['T_max'] - C2K
            ax1.plot([T_lim, T_lim], [edges[i], edges[i + 1]], color='#b2182b', lw=1.2, ls=':')
        ax2.text(0.98, 0.5 * (edges[i] + edges[i + 1]), names[i], transform=ax2.get_yaxis_transform(),
                 ha='right', va='center', fontsize=7.5, color='#555555')

    # -- temperature --
    for s, nm, ls in ((ch, f"lead, charge ({ctx['I'] / 1e3:.1f} kA)", '-'), (idle, 'lead, idle (no current)', '--')):
        if not s.get('ok'):
            continue
        x, T = s['x'], s['T'] - C2K
        if s.get('cool') is not None:
            x = np.concatenate([x, s['cool']['x']])
            T = np.concatenate([T, np.full_like(s['cool']['x'], li['T_cool'] - C2K)])
        ax1.plot(T, x, color='#1f4e9e', lw=2, ls=ls, label=nm)
    xs = np.linspace(0, L, 200)
    ax1.plot(_ins(ctx, T_ins_hot, xs)[0] - C2K, xs, color='#8c8c8c', lw=1.5, ls='-.', label='Thermax, undisturbed')
    ax1.plot([], [], color='#b2182b', ls=':', label='zone T_max')
    ax1.set_xlabel('temperature [C]')
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=7.5, loc='upper right', frameon=False)
    ax1.set_title('Temperature', fontsize=9.5)

    # -- heat per metre --
    if ch.get('ok'):
        x, J = _with_cool(ch, 'joule_per_m')
        ax2.plot(J / 1e3, x, color='#d0782a', lw=2, label=f"Joule heating, charge ({ch['Q_J'] / 1e3:.1f} kW)")
        x, q = _with_cool(ch, 'q_ex_per_m')
        ax2.plot(q / 1e3, x, color='#1f4e9e', lw=1.5, ls='--', label='to insulation, charge (< 0: from it)')
    if idle.get('ok'):
        x, q = _with_cool(idle, 'q_ex_per_m')
        ax2.plot(q / 1e3, x, color='#1f4e9e', lw=1.0, ls=':', label='to insulation, idle')
    ax2.axvline(0, color='#444444', lw=0.7)
    ax2.set_xlabel('heat per metre of lead [kW/m]')
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=7.5, loc='upper left', frameon=False)
    ax2.set_title('Heat generation and exchange', fontsize=9.5)
    for ax in (ax1, ax2):
        plt.setp(ax.get_yticklabels(), visible=False)
    sub = ''
    if ch.get('ok') and idle.get('ok'):
        sub = (f"charge: {ch['Q_J'] / 1e3:.1f} kW Joule, {ch['Q_rej'] / 1e3:.1f} kW rejected at the cold end"
               f"{' + cooling' if cooled_i is not None else ''}, {ch['drain'] / 1e3:+.1f} kW from storage;  "
               f"idle: {idle['Q_rej'] / 1e3:.1f} kW rejected, {idle['drain'] / 1e3:+.1f} kW from storage")
    fig.suptitle((title + '\n' if title else '') + sub, fontsize=10)
    return fig


def design_lead_context(cfg, opt, row, lead_inputs=None, table=None):
    """Lead context for one optimizer design row with the lead table's settings (idle disconnect,
    d_bounds scaled with sqrt(I / I_ref)); the heater current is row's heater power / cfg's
    heater.voltage. -> (ctx, table settings)"""
    from . import optimize as O
    tab = O.merge(TABLE_DEFAULTS, table or {})
    li = O.merge(LEAD_DEFAULTS, lead_inputs or {})
    li['design'] = row
    I_kA = lead_context(cfg, opt, li)['I'] / 1e3
    s = math.sqrt(I_kA / tab['I_ref_kA'])
    dlo, dhi = li['d_bounds']
    li.update(idle_break_after=tab['idle_break_after'], d_bounds=(dlo * s, dhi * s))
    return lead_context(cfg, opt, li), tab


def voltage_case(V, design, base_cfg, opt, fidelity=None, lead_inputs=None, table=None, full_leads=True):
    """One case of a heater-supply voltage study (process-pool worker): the system design (physical
    design dict, e.g. best[0]['design']) re-evaluated by optimize.evaluate with heater.voltage = V
    (heater search, element, leads from the run's lead table, capex, RTE), and with full_leads its
    leads optimized in full with the table's settings at that design's heater power and current.
    Picklable result (no lead context; rebuild it with design_lead_context(cfg, opt, res, ...)):
    dict(V, cfg, res (evaluate dict), lead (best lead), lead_losses, lead_table (interpolated
    per lead), lead_err (table / full - 1), minutes)."""
    import time
    from . import optimize as O
    t0 = time.perf_counter()
    cfg = copy.deepcopy(base_cfg)
    cfg['heater']['voltage'] = float(V)
    res = O.evaluate(design, cfg, opt, fidelity=fidelity)
    out = dict(V=float(V), cfg=cfg, res=res, lead=None, lead_losses=None, lead_table=None, lead_err=None)
    lt = O.lead_table(opt)
    if full_leads and lt is not None and res.get('feasible'):
        ld = lead_for_design(cfg, opt, res, lead_inputs, table, lt=lt, verbose=False)
        out.update(lead=ld['res'], lead_losses=ld['losses'], lead_table=ld['table'], lead_err=ld['err'])
    out['minutes'] = (time.perf_counter() - t0) / 60
    return out


def lead_for_design(cfg, opt, row, lead_inputs=None, table=None, lt=None, verbose=True):
    """Full lead optimization for one optimizer design row (e.g. best[0]) with the lead table's
    settings (zones, water-cooled last zone, idle disconnect, d_bounds scaled with sqrt(I / I_ref)),
    at the design's own heater power, t_top, T_min and T_max (the table assumed the block mean
    T_max - swing_guess / 2 for the idle hot end). lt: the run's LeadTable -> also the table's
    interpolated values at this design and their error against the full optimization.
    -> dict(ctx, res (best lead), tried {zones: result}, losses, table (interpolated per lead, or
    None), err {total, energy_cost, capex, E_cycle_kWh: table / full - 1})"""
    ctx, tab = design_lead_context(cfg, opt, row, lead_inputs, table)
    best, tried = optimize_with_skips(ctx, tab['cooled'], zones=tab['zones'], verbose=verbose)
    out = dict(ctx=ctx, res=best, tried=tried, losses=losses(ctx, best) if best.get('ok') else None,
               table=None, err=None, settings=dict(zones=tab['zones'], cooled=tab['cooled'],
                                                    idle_break_after=tab['idle_break_after']))
    if lt is not None and best.get('ok'):
        tv = lt(ctx['I'], ctx['L'], ctx['T_max'] - C2K)
        full = dict(total=best['total'], energy_cost=best['energy_cost'], capex=best['capex'],
                    E_cycle_kWh=out['losses']['E_cycle_kWh'])
        out.update(table=tv, err={k: tv[k] / v - 1 for k, v in full.items() if v})
    return out


# --------------------------------------------------------------------------
# lead table: optimal leads precomputed for the system optimizer
# --------------------------------------------------------------------------
# The optimal lead depends, within one optimizer run (prices, scenario, cycle, heater max
# temperature fixed), on the heater current I, the top insulation t_top (lead length) and the block
# T_max (Thermax hot face). Its lifetime cost and each loss per ampere are nearly independent of I
# (3-12 kA scan, 2026-11-24: $/kA flat within 3.5%, loss components within ~20%), so the table is
# t_top x T_max at a few currents, interpolated linearly in I (values per ampere; held flat outside
# the current range).
TABLE_DEFAULTS = {
    't_top': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],        # m (system design variable bounds 0.5-3)
    'T_max_C': [1800.0, 2033.3, 2266.7, 2500.0],    # C block T_max (design variable bounds 1800-2500)
    'I_kA': [4.0, 10.0],
    'zones': ['graphite', 'tungsten', 'copper'],
    'cooled': True,                                  # water-cooled copper
    'idle_break_after': 'graphite',                  # graphite joint open when idle (option 2)
    'swing': None,                                   # K block swing for the idle temperature T_max - swing/2;
                                                     #   None -> opt sizing.swing_guess
    'I_ref_kA': 9.0,                                 # d_bounds of the lead inputs apply at this current;
                                                     #   scaled with sqrt(I / I_ref) (optimal diameters ~ sqrt(I))
}
# per-lead quantities stored per point (W, $); the interpolator returns them at the asked current
TABLE_KEYS = ('total', 'capex', 'energy_cost', 'material', 'joints', 'Q_J_charge', 'drain_charge',
              'drain_idle', 'Q_rej_charge', 'Q_rej_idle', 'P_rej_peak', 'E_cycle_kWh')


def _table_point(args):
    """One table point (module level: runs in a worker process)."""
    import warnings
    cfg, opt, li, tab, I_kA, t_top, T_max_C = args
    warnings.simplefilter('ignore')
    from . import optimize as O
    s = math.sqrt(I_kA / tab['I_ref_kA'])
    dlo, dhi = li.get('d_bounds', LEAD_DEFAULTS['d_bounds'])
    swing = tab['swing'] or O.merge(O.OPT_DEFAULTS, opt)['sizing']['swing_guess']
    V = float(li.get('voltage') or cfg['heater'].get('voltage') or 60.0)
    T_max = T_max_C + C2K
    pt = O.merge(li, {'design': None, 'P_heater': I_kA * 1e3 * V, 'voltage': V, 'L': t_top,
                      'T_ins_hot': T_max, 'T_idle': T_max - swing / 2, 'idle_break_after': tab['idle_break_after'],
                      'd_bounds': (dlo * s, dhi * s)})
    ctx = lead_context(cfg, opt, pt)
    best, tried = optimize_with_skips(ctx, tab['cooled'], zones=tab['zones'], verbose=False)
    row = dict(I_kA=I_kA, t_top=t_top, T_max_C=T_max_C, ok=bool(best.get('ok')),
               feasible=bool(best.get('feasible')), zones=best['lead']['mats'], lens=best['lead']['lens'],
               d=best['lead']['d'], tried={' / '.join(z): r.get('total') for z, r in tried.items()})
    if best.get('ok'):
        lo = losses(ctx, best)
        brk = ctx['li']['break_cost'] if best['idle'].get('open_joint') is not None else 0.0
        row.update(total=best['total'], capex=best['capex'], energy_cost=best['energy_cost'],
                   material=float(sum(best['mat_cost'])), joints=best['joints'] + brk,
                   Q_J_charge=best['charge']['Q_J'], drain_charge=best['charge']['drain'],
                   drain_idle=best['idle']['drain'], Q_rej_charge=best['charge']['Q_rej'],
                   Q_rej_idle=best['idle']['Q_rej'], P_rej_peak=best['P_rej_peak'], E_cycle_kWh=lo['E_cycle_kWh'])
    return row


def _table_key(cfg, opt, li, tab):
    """Hash of everything a table depends on (cache file name)."""
    import hashlib
    import json
    from . import optimize as O
    o = O.apply_scenario(O.merge(O.OPT_DEFAULTS, opt or {}))
    dep = dict(li={k: v for k, v in li.items() if k not in ('design', 'P_heater', 'L', 'T_ins_hot', 'T_idle',
                                                            'zones', 'idle_break_after')},
               eco=o['economics'], costs=o['costs'], par=o['parasitics'], charge_hr=o.get('charge_hr'),
               duration_hr=o.get('duration_hr'), sizing=o['sizing'], tab=tab,
               heater={k: cfg['heater'].get(k) for k in ('max_temperature', 'voltage', 'density')},
               tank_graphite=cfg['tank'].get('costs', {}).get('column_graphite_per_kg'))
    return hashlib.sha1(json.dumps(dep, sort_keys=True, default=str).encode()).hexdigest()[:12]


def table_key(cfg, opt, lead_inputs=None, table=None):
    """Cache key of the lead table these inputs would build (hash of every input it depends on)."""
    from . import optimize as O
    return _table_key(cfg, O.apply_scenario(O.merge(O.OPT_DEFAULTS, opt or {})),
                      O.merge(LEAD_DEFAULTS, lead_inputs or {}), O.merge(TABLE_DEFAULTS, table or {}))


def build_lead_table(cfg, opt, lead_inputs=None, table=None, workers=None, cache_dir='opt_runs/_lead_tables',
                     verbose=True, force=False):
    """Optimal leads on the grid table['t_top'] x table['T_max_C'] x table['I_kA'] (TABLE_DEFAULTS)
    for the YAML config cfg, the optimizer inputs opt (cost weights, scenario) and the lead inputs.
    Runs in a process pool; cached in cache_dir by a hash of all inputs (same inputs -> loaded,
    unless force: always rebuilt, and the cache file overwritten). -> LeadTable."""
    import json
    import os
    import time
    from concurrent.futures import ProcessPoolExecutor
    from . import optimize as O
    tab = O.merge(TABLE_DEFAULTS, table or {})
    li = O.merge(LEAD_DEFAULTS, lead_inputs or {})
    opt = O.apply_scenario(O.merge(O.OPT_DEFAULTS, opt or {}))
    key = _table_key(cfg, opt, li, tab)
    path = os.path.join(cache_dir, f"lead_table_{key}.json") if cache_dir else None
    if path and os.path.exists(path) and not force:
        if verbose:
            print(f"lead table: loaded {path}")
        return LeadTable(json.load(open(path)), path)
    pts = [(cfg, opt, li, tab, I, t, T) for I in tab['I_kA'] for t in tab['t_top'] for T in tab['T_max_C']]
    workers = workers or min(len(pts), max(1, (os.cpu_count() or 2) - 2))
    if verbose:
        print(f"lead table: {len(pts)} lead optimizations on {workers} workers ...", flush=True)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(_table_point, pts))
    data = dict(key=key, grid=dict(I_kA=tab['I_kA'], t_top=tab['t_top'], T_max_C=tab['T_max_C']), table=tab,
                scenario=O.scenario_name(opt), elec_price=opt['economics']['elec_price'], rows=rows,
                wall_min=(time.perf_counter() - t0) / 60)
    if path:
        os.makedirs(cache_dir, exist_ok=True)
        json.dump(data, open(path, 'w'), indent=1, default=float)
    if verbose:
        bad = [r for r in rows if not r['feasible']]
        print(f"lead table: done in {data['wall_min']:.1f} min" + (f"; saved {path}" if path else '')
              + (f"; {len(bad)} points over a T limit" if bad else ''))
    return LeadTable(data, path)


class LeadTable:
    """Interpolates a lead table: lt(I [A], t_top [m], T_max_C) -> dict of per-lead quantities
    (TABLE_KEYS: lifetime $ total / capex / energy_cost, material and joint $ before soft costs,
    loss powers [W]: Q_J_charge, drain_charge, drain_idle, Q_rej_charge, Q_rej_idle, P_rej_peak;
    E_cycle_kWh). Linear in (t_top, T_max) and, per ampere, in I; inputs outside the grid are
    clamped (per-ampere values held flat)."""

    def __init__(self, data, path=None):
        from scipy.interpolate import RegularGridInterpolator
        self.data, self.path = data, path
        g = data['grid']
        self.axes = [np.asarray(g['I_kA'], float), np.asarray(g['t_top'], float), np.asarray(g['T_max_C'], float)]
        idx = {(r['I_kA'], r['t_top'], r['T_max_C']): r for r in data['rows']}
        self.missing = [k for k, r in idx.items() if not r.get('ok')]
        self._f = {}
        for key in TABLE_KEYS:
            v = np.empty([len(a) for a in self.axes])
            for i, I in enumerate(g['I_kA']):
                for j, t in enumerate(g['t_top']):
                    for k, T in enumerate(g['T_max_C']):
                        r = idx[(I, t, T)]
                        v[i, j, k] = r[key] / I if r.get('ok') else np.nan
            if len(self.axes[0]) == 1:               # one current: per-A values held flat in I
                v = np.concatenate([v, v]); ax0 = np.array([self.axes[0][0], self.axes[0][0] + 1.0])
            else:
                ax0 = self.axes[0]
            self._f[key] = RegularGridInterpolator((ax0, self.axes[1], self.axes[2]), v)
        self._ax0 = ax0

    def __call__(self, I, t_top, T_max_C):
        I_kA = float(I) / 1e3
        p = [float(np.clip(I_kA, self._ax0[0], self._ax0[-1])),
             float(np.clip(t_top, self.axes[1][0], self.axes[1][-1])),
             float(np.clip(T_max_C, self.axes[2][0], self.axes[2][-1]))]
        out = {key: float(f(p)[0]) * I_kA for key, f in self._f.items()}
        out.update(I_kA=I_kA, clamped=(p[0] != I_kA and len(self.axes[0]) > 1) or p[1] != t_top or p[2] != T_max_C)
        return out

    def summary(self):
        """Text table: lifetime $/kA and energy lost per cycle per kA on the grid."""
        g = self.data['grid']
        rows = {(r['I_kA'], r['t_top'], r['T_max_C']): r for r in self.data['rows']}
        L = [f"Lead table {self.data['key']} (scenario {self.data['scenario'] or 'none'}, "
             f"{self.data['elec_price']:g} $/MWh): lifetime $ per kA per lead; kWh lost per cycle per kA"]
        for I in g['I_kA']:
            L.append(f"  I = {I:g} kA      " + ''.join(f"T_max {T:>6.0f} C   " for T in g['T_max_C']))
            for t in g['t_top']:
                cells = []
                for T in g['T_max_C']:
                    r = rows[(I, t, T)]
                    cells.append(f"{r['total'] / I:7,.0f} {r['E_cycle_kWh'] / I:5.1f}{'*' if not r['feasible'] else ' '} "
                                 if r.get('ok') else f"{'--':>14s} ")
                L.append(f"  t_top {t:4.1f} m     " + ''.join(f"{c:<17s}" for c in cells))
        if any(not r['feasible'] for r in self.data['rows']):
            L.append("  * over a temperature limit")
        return '\n'.join(L)
