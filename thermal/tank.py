"""System-scale steel tank model (steel_tank_system_brief.md).

Takes per-stick results of the tinless model and builds the full plant around them:
checkerboard hot zone, API 650-class tank, Thermax insulation, graded graphite ->
oxide support columns, forced-air finned floor, foundation, heat losses by path and
a low / central / high capex breakdown. Closed-form or 1-D integrals only (ms per
call), all inputs from the YAML `tank:` block (defaults in TANK_DEFAULTS).

Temperatures in this module are in degC unless a name ends in _K.

Entry points
    stick_from_run(m, r)                 per-stick inputs from a build_tinless model + run
    tank_system(stick, cfg)              full system from per-stick inputs
    tank_system_reference(hz, cfg)       same, hot zone given directly (verification, §7.2)
    cell_loss_sink(m, cfg)               roof + floor loss sink for a built model, before its run
    CellLossSink.for_storage(storage)    -> Q_sink_fn for run_simulation / run_heating_simulation
    calibrate_f3d(...)                   column lateral-pickup factor vs the v3 FEA point
"""
import copy
import math
import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from .properties import thermax_tcx, thermax_tcx_integral

SIGMA = 5.670374419e-8
G = 9.81
R_GAS = 8.314462618
C2K = 273.15
RHO_STEEL = 7850.0
KCAL = 4184.0

# --------------------------------------------------------------------------
# Defaults (brief §8). Every cost is [low, central, high]; all are placeholders.
# --------------------------------------------------------------------------
TANK_DEFAULTS = {
    'P_rated': 25.0e6, 'duration_hr': 50.0,
    'T_amb': 15.0, 'T_air_design': 40.0,
    't_side': 1.0, 't_top': 1.0, 'L_bot': 2.0,
    'H_wd': None, 'H_deck': 2.0, 'c_shell': 0.05,
    'soil_allowable': 225e3,
    'thermax': {'rho': 640.0},           # k(T) in properties.thermax_tcx
    'columns': {'pitch': None, 'thickness': 0.23, 'w_max': 0.513, 'w_min': 0.135, 'sigma_ref': 2.81e6, 'T_ref': 2400.0, 'n': 3.8, 'Q_kcal': 50.0,
                'sigma_cap': 20e6, 'T1': 1200.0, 'sigma_ox': 3.0e6, 'L_graphite': None,
                'rho_graphite': 1850.0, 'rho_oxide': 3000.0, 'f_3D': 1.0, 'plate_area': 1.0},
    'floor': {'plate_t': 0.020, 'k_steel': 45.0, 'fin_h': 0.14, 'fin_t': 0.010, 'fin_pitch': 0.14,
              'channel_clearance': 0.05, 'v_air': 6.0, 'K_minor': 3.0, 'eta_fan': 0.6,
              'T_limit_normal': 93.0, 'T_limit_annexM': 120.0},
    'shell': {'h_out': 10.0, 'emissivity': 0.9, 'h_headspace': 3.0, 'emis_insulation': 0.9},
    'costs': {
        'steel_per_t': [5000, 7500, 10000], 't_shell': 0.011, 't_bottom': 0.008,
        'bottom_annular_frac': 0.10, 'roof_kg_m2': 88.0,
        'deck_kg_m2': [80, 120, 160], 'deck_per_t': [5000, 6000, 8000],
        'gas_tight_fixed': [0.15e6, 0.3e6, 0.45e6], 'per_penetration': [750, 1500, 2250],
        'thermax_per_kg': [1.5, 2.5, 4.0],
        'column_graphite_per_kg': [5, 10, 15], 'oxide_per_kg': [1, 2, 4],
        'bearing_plate': [700, 900, 1100], 'tie_grid': [700, 1500, 2300],
        'floor_per_m2': [720, 935, 1150], 'slab_per_m2': [330, 470, 610],
        'cellular_glass_per_m2': [110, 165, 220], 'mat_per_m3': [870, 1300, 1800], 'mat_thickness': 1.0,
        'fans_per_m3s': [7500, 10000, 15000],
        'storage_graphite_per_kg': [1.0, 1.0, 1.0],
    },
    'economics': {'elec_price': 87.0, 'heater_eff': 0.9},
    'block_axial_losses': True,   # roof + floor losses in the stick / heater runs (CellLossSink)
}

# Air at ~60 C (brief §5.4)
AIR = dict(rho=1.060, mu=2.008e-5, k=0.02808, cp=1007.0, Pr=0.720)


def merge_defaults(cfg_tank=None):
    """Deep-merge a YAML `tank:` dict over TANK_DEFAULTS."""
    out = copy.deepcopy(TANK_DEFAULTS)

    def _num(v):
        # PyYAML (YAML 1.1) reads 25.0e6 as a string; accept it as a number
        if isinstance(v, str):
            try:
                return float(v)
            except ValueError:
                return v
        if isinstance(v, list):
            return [_num(x) for x in v]
        return v

    def _m(d, s):
        for k, v in (s or {}).items():
            if isinstance(v, dict) and isinstance(d.get(k), dict):
                _m(d[k], v)
            else:
                d[k] = _num(v)
    _m(out, cfg_tank)
    return out


# --------------------------------------------------------------------------
# Materials
# --------------------------------------------------------------------------
def k_thermax(T):
    """Thermax [W/m-K], T in C (properties.thermax_tcx)."""
    return thermax_tcx(np.asarray(T, float) + C2K)


def phi_thermax(Tc, Th):
    """Kirchhoff integral of Thermax k from Tc to Th [W/m], T in C (exact)."""
    return thermax_tcx_integral(np.asarray(Tc, float) + C2K, np.asarray(Th, float) + C2K)


_KG_T = np.array([20, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400], float)
_KG_K = np.array([115, 100, 85, 73, 62, 54, 49, 45, 42, 40, 39, 37, 34], float)


def k_graphite(T):
    """Column graphite k [W/m-K], T in C. Mersen 2020 datasheet table from
    Tank_Enclosure_Model.ipynb, used as the ISO-68 curve (flag: not an ISO-68 datasheet)."""
    return np.interp(np.clip(T, _KG_T[0], _KG_T[-1]), _KG_T, _KG_K)


# exact cumulative integral of the piecewise-linear table (constant beyond the ends)
_KG_CUM = np.concatenate([[0.0], np.cumsum(0.5 * (_KG_K[1:] + _KG_K[:-1]) * np.diff(_KG_T))])


_KG_SLOPE = np.diff(_KG_K) / np.diff(_KG_T)


def _cum_kg(T):
    """int_{T_0}^{T} k_graphite (quadratic within each linear-k segment)."""
    T = np.asarray(T, float)
    Tc = np.clip(T, _KG_T[0], _KG_T[-1])
    i = np.clip(np.searchsorted(_KG_T, Tc, side='right') - 1, 0, len(_KG_T) - 2)
    d = Tc - _KG_T[i]
    inside = _KG_CUM[i] + _KG_K[i] * d + 0.5 * _KG_SLOPE[i] * d ** 2
    return inside + _KG_K[-1] * np.maximum(T - _KG_T[-1], 0) - _KG_K[0] * np.maximum(_KG_T[0] - T, 0)


def psi_graphite(a, b):
    """int_a^b k_graphite dT [W/m] (exact)."""
    return float(_cum_kg(b) - _cum_kg(a))


def k_oxide(T):
    """High-Al refractory / alumina posts [W/m-K], T in C: 2.8 + 1000/T[K]."""
    return 2.8 + 1000.0 / (np.asarray(T, float) + C2K)


def psi_oxide(a, b):
    """int_a^b k_oxide dT [W/m] (exact)."""
    return 2.8 * (b - a) + 1000.0 * math.log((b + C2K) / (a + C2K))


def sigma_allow(T, col):
    """Creep-limited column allowable [Pa], T in C (brief §5.3):
    sigma_ref * exp[Q/(nR) (1/T - 1/T_ref)], capped at sigma_cap."""
    a = col['Q_kcal'] * KCAL / (col['n'] * R_GAS)
    s = col['sigma_ref'] * np.exp(a * (1.0 / (np.asarray(T, float) + C2K) - 1.0 / (col['T_ref'] + C2K)))
    return np.minimum(s, col['sigma_cap'])


# --------------------------------------------------------------------------
# Inputs from the per-stick model (brief §2)
# --------------------------------------------------------------------------
@dataclass
class StickInputs:
    P_stick: float          # W_e, discharge-average electrical power (Nate, 2026-11-18)
    E_stick_e: float        # J_e delivered over the discharge
    E_stick_th: float       # J_th stored between T_min and T_max
    V_block: float          # m3 graphite per stick
    r_out: float
    r_in: float
    H_block: float
    L_ps: float
    T_min: float            # C
    T_max: float            # C
    rho_g: float            # storage graphite density used by the stick model [kg/m3]
    t_discharge: float      # s
    eta_block: float = 0.45  # electricity / heat leaving the block (post-hoc loss correction)
    t: np.ndarray = None    # s, discharge history
    T_avg: np.ndarray = None  # C, volume-weighted block average
    T_top: np.ndarray = None  # C, top ring average (roof face)
    T_bot: np.ndarray = None  # C, bottom ring average (floor face)
    A_hcav_default: float = None  # m2 heater cavity, from heater.side_len and heater.area_ratio
    E_sink: float = 0.0           # J removed by Q_sink_fn during the discharge (roof + floor losses)
    cell_losses_in_run: bool = False  # roof + floor losses already in the stick run (CellLossSink)


def _ring_weights(storage):
    b = storage.block
    nR = storage.num_rad_segments
    r = b.rad_in + np.arange(nR + 1) * (b.rad_out - b.rad_in) / nR
    return np.diff(r ** 2)                      # radial ring area weights


def ring_means(T_dist, storage):
    """Volume-weighted mean temperature of each axial ring [same units as T_dist]."""
    w = _ring_weights(storage)
    return (np.asarray(T_dist) * w).sum(axis=1) / w.sum()


def _stick_geometry(m):
    """Per-stick geometry and stored heat from a built (not necessarily run) model."""
    from .properties import storage_rho, storage_cp
    st = m.storage
    rin, rout = st.block.rad_in, st.block.rad_out
    H = float(m.H_block)
    V = math.pi * (rout ** 2 - rin ** 2) * H
    T_lo, T_hi = float(m.T_min), float(m.T_max)
    rho = float(storage_rho(0.5 * (T_lo + T_hi)))
    E_th = rho * V * quad(lambda T: float(storage_cp(T)), T_lo, T_hi)[0]
    hc = m.cfg.get('heater', {}) if isinstance(m.cfg, dict) else {}
    A_h = None
    if hc.get('side_len') and hc.get('area_ratio'):
        A_h = math.pi * (2 * hc['side_len'] / math.pi * hc['area_ratio']) ** 2
    return dict(r_in=rin, r_out=rout, H=H, V=V, T_lo=T_lo, T_hi=T_hi, rho=rho, E_th=E_th, A_h=A_h,
                L=float(m.L))


def stick_from_run(m, r):
    """Per-stick inputs from a build_tinless model `m` and its discharge results `r`
    (run with snap_every set for the top/bottom ring histories; else start/end only)."""
    st = m.storage
    t, dt = r['time'], np.diff(r['time'])
    on = np.asarray(r['ps_active'][1:], bool)
    E_e = float(np.sum((r['P_elec'][1:] * dt)[on]))
    E_rad = float(np.sum((r['Q_rad'][1:] * dt)[on]))
    qs = np.asarray(r.get('Q_sink', np.zeros_like(t)), float)
    if len(qs) != len(t):
        raise ValueError(f"results['Q_sink'] has {len(qs)} entries vs {len(t)} time points")
    E_sink = float(np.sum((qs[1:] * dt)[on]))
    t_end = float(r['t_ps_removed'] if r['t_ps_removed'] is not None else t[-1])
    g = _stick_geometry(m)
    rin, rout, H, V, T_lo, T_hi, rho, E_th, A_h = (g[k] for k in
                                                   ('r_in', 'r_out', 'H', 'V', 'T_lo', 'T_hi', 'rho', 'E_th', 'A_h'))
    # volume-weighted block average and top / bottom ring histories from the snapshots
    snaps = np.asarray(r['temp_snapshots'])
    ts = np.asarray(r['snap_times'])
    dz = st.axial_dz()
    w = _ring_weights(st)
    rm = np.array([ring_means(s, st) for s in snaps])
    Tv = (rm * dz).sum(axis=1) / dz.sum()
    keep = ts <= t_end + 1e-6
    t_on = t[: int(np.searchsorted(t, t_end, side='right'))]
    # time base = solver steps; T_avg from the step record corrected to a volume weighting
    T_step = np.interp(t_on, ts[keep], Tv[keep]) if keep.sum() > 2 else np.asarray(r['T_avg'])[: len(t_on)]
    del w
    return StickInputs(
        P_stick=E_e / t_end, E_stick_e=E_e, E_stick_th=E_th, V_block=V, r_out=rout, r_in=rin,
        H_block=H, L_ps=float(m.L), T_min=T_lo - C2K, T_max=T_hi - C2K, rho_g=rho, t_discharge=t_end,
        eta_block=E_e / E_rad if E_rad else 0.45,
        t=t_on, T_avg=T_step - C2K,
        T_top=np.interp(t_on, ts[keep], rm[keep, -1]) - C2K,
        T_bot=np.interp(t_on, ts[keep], rm[keep, 0]) - C2K,
        A_hcav_default=A_h, E_sink=E_sink, cell_losses_in_run=E_sink > 0)


# --------------------------------------------------------------------------
# §3 hot zone
# --------------------------------------------------------------------------
@dataclass
class HotZone:
    N_ps: int
    N_h: int
    D_hot: float
    H_block: float
    m_graphite: float       # kg storage graphite
    E_th: float             # J stored (swing)
    A_c: float = np.nan     # catchment area per stick
    pitch: float = np.nan   # stick-heater spacing
    fill: float = np.nan
    E_e: float = np.nan     # J_e delivered (N * E_stick_e)
    E_required: float = np.nan
    energy_mismatch: float = np.nan   # N E_stick_e / (P_rated * duration) - 1
    A_hcav: float = np.nan
    L_ps: float = 3.0

    @property
    def A_hot(self):
        return math.pi * self.D_hot ** 2 / 4


def cavity_webs(r_in, r_out, r_hcav, rotated=True):
    """Graphite wall ('web') thickness between neighbouring cavities on the checkerboard
    (pitch p = sqrt(A_c / 2), A_c = pi r_out^2 + pi r_hcav^2, as size_hot_zone).
    Stick cavity: square of side S = pi r_in / 2 (r_in is its equivalent radius); heater
    cavity: circle r_hcav. rotated=True: stick squares turned 45 deg to the grid, corners
    toward the heaters, faces toward the diagonal stick neighbours.
    -> (pitch, {'stick-heater', 'stick-stick', 'heater-heater'} [m])"""
    p = math.sqrt((math.pi * r_out ** 2 + math.pi * r_hcav ** 2) / 2.0)
    S = math.pi * r_in / 2.0
    if rotated:
        sh, ss = p - S / math.sqrt(2.0) - r_hcav, math.sqrt(2.0) * p - S
    else:
        sh, ss = p - S / 2.0 - r_hcav, math.sqrt(2.0) * (p - S)
    return p, {'stick-heater': sh, 'stick-stick': ss, 'heater-heater': math.sqrt(2.0) * p - 2.0 * r_hcav}


def size_hot_zone(st, P_rated, duration_hr, A_hcav):
    N = int(math.ceil(P_rated / st.P_stick))
    E_req = P_rated * duration_hr * 3600.0
    A_c = math.pi * st.r_out ** 2 + A_hcav
    p = math.sqrt(A_c / 2.0)
    D = math.sqrt(4 * N * A_c / math.pi) + p
    V_g = N * st.V_block
    fill = V_g / (math.pi * D ** 2 / 4 * st.H_block)
    return HotZone(N_ps=N, N_h=N, D_hot=D, H_block=st.H_block, m_graphite=V_g * st.rho_g,
                   E_th=N * st.E_stick_th, A_c=A_c, pitch=p, fill=fill, E_e=N * st.E_stick_e,
                   E_required=E_req, energy_mismatch=N * st.E_stick_e / E_req - 1.0, A_hcav=A_hcav,
                   L_ps=st.L_ps)


# --------------------------------------------------------------------------
# §4 tank geometry
# --------------------------------------------------------------------------
@dataclass
class TankGeometry:
    D_tank: float
    H_tank: float
    r_i: float
    r_o: float
    H_wd: float
    A_hot: float
    A_tank: float
    p_bear: float           # Pa on the column grid
    warnings: list = field(default_factory=list)


def tank_geometry(hz, c):
    r_i = hz.D_hot / 2
    D_t = hz.D_hot + 2 * c['t_side'] + 2 * c['c_shell']
    H_wd = c['H_wd'] if c['H_wd'] is not None else hz.L_ps + c['t_top']
    H_t = c['L_bot'] + hz.H_block + c['t_top'] + H_wd + c['H_deck']
    A_hot = math.pi * r_i ** 2
    # storage graphite + top Thermax, carried by the columns (sticks, liners and
    # heaters hang from the deck; shade vanes, ~1% of block mass, are neglected)
    p = G * (hz.m_graphite / A_hot + c['thermax']['rho'] * c['t_top'])
    w = []
    if not 10 <= H_t <= 30:
        w.append(f"H_tank {H_t:.1f} m outside the usual API 650 range 10-30 m")
    if not 15 <= D_t <= 80:
        w.append(f"D_tank {D_t:.1f} m outside the usual API 650 range 15-80 m")
    if p > 225e3:
        w.append(f"column-grid bearing {p / 1e3:.0f} kPa > 225 kPa (beyond CSP hot-tank precedent)")
    return TankGeometry(D_tank=D_t, H_tank=H_t, r_i=r_i, r_o=r_i + c['t_side'], H_wd=H_wd,
                        A_hot=A_hot, A_tank=math.pi * D_t ** 2 / 4, p_bear=p, warnings=w)


# --------------------------------------------------------------------------
# §5.2 walls, edges, roof
# --------------------------------------------------------------------------
def _surface_q(Ts, Ta, h, eps):
    return h * (Ts - Ta) + eps * SIGMA * ((Ts + C2K) ** 4 - (Ta + C2K) ** 4)


def side_wall(T_hot, geo, hz, c):
    """Cylindrical Thermax wall over H_block; shell cold face to ambient. -> (Q, T_shell)."""
    sh = c['shell']
    ln = math.log(geo.r_o / geo.r_i)
    A_o = 2 * math.pi * geo.r_o * hz.H_block

    def res(Ts):
        return 2 * math.pi * hz.H_block * phi_thermax(Ts, T_hot) / ln \
            - A_o * _surface_q(Ts, c['T_amb'], sh['h_out'], sh['emissivity'])
    Ts = brentq(res, c['T_amb'], T_hot, xtol=1e-6)
    return 2 * math.pi * hz.H_block * phi_thermax(Ts, T_hot) / ln, Ts


def roof(T_hot, geo, c):
    """Planar top Thermax -> headspace (argon convection + radiation) -> deck -> ambient.
    -> (Q, T_insulation_top, T_deck)."""
    sh = c['shell']
    A_h, A_d = geo.A_hot, geo.A_tank
    eps_hs = 1.0 / (1.0 / sh['emis_insulation'] + A_h / A_d * (1.0 / sh['emissivity'] - 1.0))

    def Tc_of(Td):          # insulation top face for a given deck temperature
        f = lambda Tc: phi_thermax(Tc, T_hot) / c['t_top'] - _surface_q(Tc, Td, sh['h_headspace'], eps_hs)
        return brentq(f, Td, T_hot, xtol=1e-6)

    def res(Td):
        Tc = Tc_of(Td)
        return A_h * phi_thermax(Tc, T_hot) / c['t_top'] \
            - A_d * _surface_q(Td, c['T_amb'], sh['h_out'], sh['emissivity'])
    Td = brentq(res, c['T_amb'] + 1e-9, T_hot - 1.0, xtol=1e-6)
    Tc = Tc_of(Td)
    return A_h * phi_thermax(Tc, T_hot) / c['t_top'], Tc, Td


# --------------------------------------------------------------------------
# §5.3 bottom stack: graded graphite -> oxide columns in parallel with Thermax
# --------------------------------------------------------------------------
@dataclass
class ColumnDesign:
    N_col: float
    pitch: float            # m square grid
    F: float                # N per column
    A_top: float            # m2 graphite at T_max
    A_joint: float          # m2 graphite at T1
    A_ox: float
    L_g: float
    L_ox: float
    V_g: float              # m3 graphite per column
    V_ox: float
    A_mean: float
    S_g: float              # shape integral int dz/A over the graphite segment [1/m]
    S_ox: float
    T_design: float
    T_floor_design: float
    warnings: list = field(default_factory=list)


def column_pitch(p_bear, T_hot, col):
    """Square-grid pitch [m]: tributary area = column capacity / areal load, where the
    capacity is the largest graphite section (w_max x thickness, at the hot face) at its
    creep allowable. columns.pitch (if set) overrides."""
    if col.get('pitch') is not None:
        return float(col['pitch'])
    return math.sqrt(float(sigma_allow(T_hot, col)) * col['w_max'] * col['thickness'] / p_bear)


def design_columns(A_hot, p_bear, T_hot, T_floor, L_bot, col, pitch=None):
    """Size the columns at the peak hot-face temperature (area follows local T).
    pitch None -> column_pitch(p_bear, T_hot, col).

    Joint placement:
      columns.L_graphite None -> joint AT T1 (brief §5.3); segment lengths follow from
                                 1-D continuity (L_g = int k A dT / Q).
      columns.L_graphite = x  -> graphite segment length fixed (e.g. a built design);
                                 the joint temperature is an output, warned if > T1
                                 (T1 is then the oxide's maximum service temperature)."""
    p = column_pitch(p_bear, T_hot, col) if pitch is None else float(pitch)
    N = A_hot / p ** 2
    F = p_bear * p ** 2
    T1 = col['T1']
    A_min = col['w_min'] * col['thickness']      # rectangular section: width varies, thickness fixed
    A_ox = F / col['sigma_ox']
    w = []
    Tg = np.linspace(T_floor, T_hot, 4001)
    Ag = np.maximum(F / sigma_allow(Tg, col), A_min)
    kA = k_graphite(Tg) * Ag
    # cumulative integrals measured from the hot end: I(Tj) = int_Tj^T_hot k A dT
    cI = np.concatenate([[0.0], np.cumsum(0.5 * (kA[1:] + kA[:-1]) * np.diff(Tg))])
    cJ = np.concatenate([[0.0], np.cumsum(0.5 * (kA[1:] * Ag[1:] + kA[:-1] * Ag[:-1]) * np.diff(Tg))])
    I_of = lambda Tj: float(cI[-1] - np.interp(Tj, Tg, cI))
    J_of = lambda Tj: float(cJ[-1] - np.interp(Tj, Tg, cJ))
    L_fix = col.get('L_graphite')
    if L_fix is None:
        if T_hot <= T1:
            raise ValueError(f"hot face {T_hot:.0f} C <= graphite/oxide transition T1 {T1:.0f} C")
        Tj = T1
        Q1 = (I_of(Tj) + A_ox * psi_oxide(T_floor, Tj)) / L_bot     # 1-D column heat [W]
        L_g = I_of(Tj) / Q1
    else:
        L_g = float(L_fix)
        if not 0 < L_g < L_bot:
            raise ValueError(f"columns.L_graphite {L_g} must lie in (0, L_bot = {L_bot})")
        Tj = brentq(lambda x: (L_bot - L_g) * I_of(x) - L_g * A_ox * psi_oxide(T_floor, x),
                    T_floor + 1e-6, T_hot - 1e-6, xtol=1e-6)
        Q1 = I_of(Tj) / L_g
        if Tj > T1:
            w.append(f"graphite/oxide joint {Tj:.0f} C > oxide limit T1 {T1:.0f} C")
    L_ox = L_bot - L_g
    if L_ox <= 0:
        w.append(f"oxide segment length {L_ox:.3f} m <= 0")
    V_g = J_of(Tj) / Q1
    V_ox = A_ox * max(L_ox, 0.0)
    A_g = lambda T: max(F / float(sigma_allow(T, col)), A_min)
    return ColumnDesign(N_col=N, pitch=p, F=F, A_top=A_g(T_hot), A_joint=A_g(Tj), A_ox=A_ox, L_g=L_g, L_ox=L_ox,
                        V_g=V_g, V_ox=V_ox, A_mean=(V_g + V_ox) / L_bot, S_g=psi_graphite(Tj, T_hot) / Q1,
                        S_ox=max(L_ox, 1e-9) / A_ox, T_design=T_hot, T_floor_design=T_floor, warnings=w)


def column_heat(cd, T_hot, T_floor):
    """1-D heat through one built column (fixed geometry) for any hot/cold face:
    Kirchhoff in each segment (exact: Q dz/A = -k dT), joint temperature from
    continuity. -> (Q_1D, T_joint)."""
    f = lambda Tj: psi_graphite(Tj, T_hot) / cd.S_g - psi_oxide(T_floor, Tj) / cd.S_ox
    Tj = brentq(f, T_floor, T_hot, xtol=1e-6)
    return psi_graphite(Tj, T_hot) / cd.S_g, Tj


def bottom_heat(cd, A_hot, T_hot, T_floor, c):
    """Total bottom heat [W] at the given faces -> dict."""
    col = c['columns']
    Q1, Tj = column_heat(cd, T_hot, T_floor)
    Q_col = col['f_3D'] * Q1
    Q_thx = (A_hot - cd.N_col * cd.A_mean) * phi_thermax(T_floor, T_hot) / c['L_bot']
    Q = cd.N_col * Q_col + Q_thx
    return dict(Q_bot=Q, Q_col_each=Q_col, Q_col_total=cd.N_col * Q_col, Q_thx=Q_thx,
                q_bot=Q / A_hot, col_share=cd.N_col * Q_col / Q, T_joint=Tj)


# --------------------------------------------------------------------------
# §5.4 finned floor, forced air
# --------------------------------------------------------------------------
def floor_air(Q_bot, q_bot, Q_col_each, D_tank, T_in, c):
    f = c['floor']
    s = f['fin_pitch'] - f['fin_t']
    H_ch = f['fin_h'] + f['channel_clearance']
    D_h = 4 * s * H_ch / (2 * (s + H_ch))
    Re = AIR['rho'] * f['v_air'] * D_h / AIR['mu']
    ff = (0.79 * math.log(Re) - 1.64) ** -2
    Nu = (ff / 8) * (Re - 1000) * AIR['Pr'] / (1 + 12.7 * math.sqrt(ff / 8) * (AIR['Pr'] ** (2 / 3) - 1))
    h = Nu * AIR['k'] / D_h
    mH = math.sqrt(2 * h / (f['k_steel'] * f['fin_t'])) * f['fin_h']
    eta = math.tanh(mH) / mH
    h_eff = h * ((f['fin_pitch'] - f['fin_t']) + eta * 2 * f['fin_h']) / f['fin_pitch']
    Vdot = f['v_air'] * D_tank * s * H_ch / f['fin_pitch']
    dT = Q_bot / (AIR['rho'] * Vdot * AIR['cp'])
    T_max = T_in + dT + q_bot / h_eff
    q_col = Q_col_each / c['columns']['plate_area']
    dp = (ff * D_tank / D_h + f['K_minor']) * AIR['rho'] * f['v_air'] ** 2 / 2
    return dict(h=h, Re=Re, eta_fin=eta, h_eff=h_eff, Vdot=Vdot, dT_air=dT,
                T_floor_avg=T_in + 0.5 * dT + q_bot / h_eff, T_floor_max=T_max,
                T_floor_col=T_max + (q_col - q_bot) / h_eff, dp=dp, P_fan=dp * Vdot / f['eta_fan'],
                margin_normal=f['T_limit_normal'] - T_max, margin_annexM=f['T_limit_annexM'] - T_max)


def bottom_and_floor(cd, A_hot, D_tank, T_hot, T_in, c, tol=1e-3, max_iter=60, T_floor=None):
    """Bottom stack + floor solved together (floor average = cold face). T_floor pins it."""
    if T_floor is not None:
        b = bottom_heat(cd, A_hot, T_hot, T_floor, c)
        fl = floor_air(b['Q_bot'], b['q_bot'], b['Q_col_each'], D_tank, T_in, c)
        return b, fl, dict(converged=True, n_iter=0, T_floor=T_floor)
    Tf = T_in + 30.0
    for it in range(max_iter):
        b = bottom_heat(cd, A_hot, T_hot, Tf, c)
        fl = floor_air(b['Q_bot'], b['q_bot'], b['Q_col_each'], D_tank, T_in, c)
        if abs(fl['T_floor_avg'] - Tf) < tol:
            return b, fl, dict(converged=True, n_iter=it + 1, T_floor=Tf)
        Tf = fl['T_floor_avg']
    return b, fl, dict(converged=False, n_iter=max_iter, T_floor=Tf)


# --------------------------------------------------------------------------
# Losses
# --------------------------------------------------------------------------
def losses(T_side, T_top, T_bot, geo, hz, cd, c, T_air=None):
    """All loss paths [W] for given hot-face temperatures (C); air inlet = T_air (default T_amb)."""
    T_air = c['T_amb'] if T_air is None else T_air
    Q_side, T_shell = side_wall(T_side, geo, hz, c)
    Q_edge = 2 * 0.54 * math.pi * hz.D_hot * phi_thermax(T_shell, T_side)
    Q_top, T_ins_top, T_deck = roof(T_top, geo, c)
    b, fl, it = bottom_and_floor(cd, geo.A_hot, geo.D_tank, T_bot, T_air, c)
    tot = Q_side + Q_edge + Q_top + b['Q_bot']
    return dict(Q_side=Q_side, Q_edge=Q_edge, Q_top=Q_top, Q_bot=b['Q_bot'], Q_bot_col=b['Q_col_total'],
                Q_bot_thx=b['Q_thx'], Q_total=tot, q_bot=b['q_bot'], T_shell=T_shell,
                T_ins_top=T_ins_top, T_deck=T_deck, T_joint=b['T_joint'], T_floor_avg=fl['T_floor_avg'],
                T_floor_max=fl['T_floor_max'], P_fan=fl['P_fan'], floor_converged=it['converged'])


# --------------------------------------------------------------------------
# §6 costs
# --------------------------------------------------------------------------
def costs(hz, geo, cd, fl_design, c):
    k = c['costs']
    th = c['thermax']
    lines = {}
    i3 = range(3)

    def add(name, qty, rate, unit):
        rate = np.broadcast_to(np.asarray(rate, float), (3,))
        qty = np.broadcast_to(np.asarray(qty, float), (3,))
        lines[name] = dict(qty=qty.tolist(), unit=unit, cost=(qty * rate).tolist())

    D, A_t = geo.D_tank, geo.A_tank
    m_shell = math.pi * D * geo.H_tank * k['t_shell'] * RHO_STEEL / 1e3
    m_bot = A_t * k['t_bottom'] * RHO_STEEL * (1 + k['bottom_annular_frac']) / 1e3
    m_roof = A_t * k['roof_kg_m2'] / 1e3
    add('tank shell', m_shell, k['steel_per_t'], 't')
    add('tank bottom', m_bot, k['steel_per_t'], 't')
    add('roof, ring girder, wind girders', m_roof, k['steel_per_t'], 't')
    m_deck = np.asarray(k['deck_kg_m2'], float) * A_t / 1e3
    add('deck trusses', m_deck, k['deck_per_t'], 't')
    n_pen = hz.N_ps + hz.N_h
    lines['gas tightness, nozzles'] = dict(qty=[n_pen] * 3, unit='penetrations',
                                           cost=[k['gas_tight_fixed'][i] + k['per_penetration'][i] * n_pen for i in i3])
    H_side = c['L_bot'] + hz.H_block + c['t_top']
    V_side = math.pi * (geo.r_o ** 2 - geo.r_i ** 2) * H_side
    V_top = geo.A_hot * c['t_top']
    V_bot = geo.A_hot * c['L_bot'] - cd.N_col * (cd.V_g + cd.V_ox)
    for nm, V in (('Thermax side', V_side), ('Thermax top', V_top), ('Thermax bottom', V_bot)):
        add(nm, V * th['rho'], k['thermax_per_kg'], 'kg')
    col = c['columns']
    add('column graphite', cd.N_col * cd.V_g * col['rho_graphite'], k['column_graphite_per_kg'], 'kg')
    add('oxide posts', cd.N_col * cd.V_ox * col['rho_oxide'], k['oxide_per_kg'], 'kg')
    add('bearing plates', cd.N_col, k['bearing_plate'], 'columns')
    add('column tie grid', cd.N_col, k['tie_grid'], 'columns')
    add('floor plate, fins, welds', A_t, k['floor_per_m2'], 'm2')
    add('HT slab, rails, firebrick', A_t, k['slab_per_m2'], 'm2')
    add('cellular glass', A_t, k['cellular_glass_per_m2'], 'm2')
    add('structural mat', A_t * k['mat_thickness'], k['mat_per_m3'], 'm3')
    add('fans, louvres, exhaust, controls', fl_design['Vdot'], k['fans_per_m3s'], 'm3/s')
    tot = np.sum([v['cost'] for v in lines.values()], axis=0)
    groups = {
        'containment': ('tank shell', 'tank bottom', 'roof, ring girder, wind girders', 'deck trusses',
                        'gas tightness, nozzles'),
        'insulation': ('Thermax side', 'Thermax top', 'Thermax bottom'),
        'supports': ('column graphite', 'oxide posts', 'bearing plates', 'column tie grid'),
        'floor & foundation': ('floor plate, fins, welds', 'HT slab, rails, firebrick', 'cellular glass',
                               'structural mat', 'fans, louvres, exhaust, controls'),
    }
    by_group = {g: np.sum([lines[n]['cost'] for n in ns], axis=0).tolist() for g, ns in groups.items()}
    storage_g = (hz.m_graphite * np.asarray(k['storage_graphite_per_kg'], float)).tolist()
    weights = dict(steel_t=m_shell + m_bot + m_roof, deck_t=float(m_deck[1]),
                   thermax_t=(V_side + V_top + V_bot) * th['rho'] / 1e3,
                   column_graphite_t=cd.N_col * cd.V_g * col['rho_graphite'] / 1e3,
                   oxide_t=cd.N_col * cd.V_ox * col['rho_oxide'] / 1e3)
    return dict(lines=lines, groups=by_group, total=tot.tolist(), storage_graphite=storage_g, weights=weights)


# --------------------------------------------------------------------------
# Full system
# --------------------------------------------------------------------------
@dataclass
class TankResult:
    hz: HotZone
    geo: TankGeometry
    cols: ColumnDesign
    design: dict            # peak (T_max, design-day air): losses + floor
    floor_design: dict
    loss_peak: dict         # T_max, annual-mean air
    loss_cycle: dict        # time-average over the discharge history
    cost: dict
    loss_fn: object         # Q_loss(T_hot) [W], uniform hot face, annual-mean air
    foundation_pressure: float
    economics: dict
    stick_correction: dict
    warnings: list
    cfg: dict
    cell_sink: object = None  # CellLossSink of this system (roof + floor loss per cell)

    def summary(self):
        """Printable report (multi-line str)."""
        hz, g, cd, L = self.hz, self.geo, self.cols, self.loss_cycle
        _t = self.cfg['columns']['thickness']
        E = hz.E_th
        lo, mid, hi = self.cost['total']
        e_th = E / 3.6e6
        e_e = (hz.E_e / 3.6e6) if np.isfinite(hz.E_e) else np.nan
        lines = [
            f"Hot zone: {hz.N_ps} sticks + {hz.N_h} heaters, D_hot {hz.D_hot:.1f} m x H {hz.H_block:.1f} m, "
            f"fill {hz.fill:.2f}, graphite {hz.m_graphite / 1e3:,.0f} t, stored {E / 3.6e12:.2f} GWh_th",
        ]
        if np.isfinite(hz.energy_mismatch):
            lines.append(f"  energy: N x E_stick_e = {hz.E_e / 3.6e12:.3f} GWh_e vs required "
                         f"{hz.E_required / 3.6e12:.3f} GWh_e ({hz.energy_mismatch * 100:+.1f}%, not resized)")
        lines += [
            f"Tank: D {g.D_tank:.1f} m x H {g.H_tank:.1f} m (withdrawal {g.H_wd:.1f} m), bearing "
            f"{g.p_bear / 1e3:.0f} kPa, foundation {self.foundation_pressure / 1e3:.0f} kPa "
            f"(allowable {self.cfg['soil_allowable'] / 1e3:.0f})",
            f"Columns: {cd.N_col:.0f} @ {cd.pitch:.2f} m pitch ({cd.F / 1e3:.0f} kN each), graphite width "
            f"{cd.A_top / _t * 1e3:.0f} -> {cd.A_joint / _t * 1e3:.0f} mm x {_t * 1e3:.0f} mm ({cd.L_g:.2f} m) + oxide "
            f"{cd.A_ox / _t * 1e3:.0f} mm wide ({cd.L_ox:.2f} m), graphite {self.cost['weights']['column_graphite_t']:.0f} t, f_3D "
            f"{self.cfg['columns']['f_3D']:.2f}",
            "Losses [kW]       side   edges     top  bottom (col / Thermax)   total   %/day",
        ]
        for nm, d in (('peak (T_max)', self.loss_peak), ('cycle average', L)):
            lines.append(f"  {nm:14s} {d['Q_side'] / 1e3:6.0f} {d['Q_edge'] / 1e3:7.0f} {d['Q_top'] / 1e3:7.0f} "
                         f"{d['Q_bot'] / 1e3:7.0f} ({d['Q_bot_col'] / 1e3:4.0f} / {d['Q_bot_thx'] / 1e3:4.0f}) "
                         f"{d['Q_total'] / 1e3:8.0f} {d['Q_total'] * 86400 / E * 100:7.2f}")
        d, fl = self.design, self.floor_design
        lines += [
            f"Temperatures (peak, design-day air): shell {d['T_shell']:.0f} C, deck {d['T_deck']:.0f} C, "
            f"joint {d['T_joint']:.0f} C, floor avg {fl['T_floor_avg']:.0f} / max {fl['T_floor_max']:.0f} / "
            f"column spot {fl['T_floor_col']:.0f} C (Annex M limit {self.cfg['floor']['T_limit_annexM']:.0f} C)",
            f"Floor air: {fl['Vdot']:.1f} m3/s, dT {fl['dT_air']:.1f} K, h_eff {fl['h_eff']:.1f} W/m2K, "
            f"fan {fl['P_fan'] / 1e3:.1f} kW; peak q_bot {d['q_bot']:.0f} W/m2, cycle {L['q_bot']:.0f} W/m2",
            f"Capex (containment + insulation + supports + foundation): ${lo / 1e6:.1f} / {mid / 1e6:.1f} / "
            f"{hi / 1e6:.1f} M  = ${mid / e_th:.2f}/kWh_th"
            + (f", ${mid / e_e:.2f}/kWh_e" if np.isfinite(e_e) else ''),
            "  " + ', '.join(f"{k} ${v[1] / 1e6:.1f}M" for k, v in self.cost['groups'].items())
            + f"; storage graphite (pass-through) ${self.cost['storage_graphite'][1] / 1e6:.1f}M",
            f"Lost heat: ${self.economics['annual_cost'] / 1e6:.2f}M/yr "
            f"({self.economics['annual_MWh']:,.0f} MWh_e/yr at ${self.cfg['economics']['elec_price']:.0f}/MWh)",
        ]
        sc = self.stick_correction
        if sc:
            if sc['cell_losses_in_run']:
                lines.append(f"Roof + floor losses in the stick run: {sc['E_sink_kWh']:.0f} kWh_th per stick "
                             "over the discharge")
            lines.append(f"Stick correction (post-hoc, {sc['paths']}): {sc['E_loss_per_stick_kWh']:.0f} kWh_th lost "
                         f"per stick over the discharge -> E_stick_e {sc['E_stick_e_kWh']:.0f} -> "
                         f"{sc['E_stick_e_net_kWh']:.0f} kWh ({sc['penalty_frac'] * 100:.1f}%)")
        lines += [f"WARNING: {w}" for w in self.warnings]
        return '\n'.join(lines)


def _design_bottom(geo, c, T_max):
    """Columns (pitch from capacity / areal load, then the graded section) sized at T_max
    with the design-day floor: fixed point on the floor temperature. The governing
    section is at the hot face, so the pitch itself does not move with the floor."""
    Tf = c['T_air_design'] + 40.0
    for _ in range(30):
        cd = design_columns(geo.A_hot, geo.p_bear, T_max, Tf, c['L_bot'], c['columns'])
        b, fl, it = bottom_and_floor(cd, geo.A_hot, geo.D_tank, T_max, c['T_air_design'], c)
        if abs(it['T_floor'] - Tf) < 1e-3:
            break
        Tf = it['T_floor']
    return cd, b, fl, it


def _system(hz, c, T_max, hist=None, stick=None):
    """Shared core. hist = (t, T_side, T_top, T_bot) in C for the cycle average."""
    geo = tank_geometry(hz, c)
    warn = list(geo.warnings)
    cd, b, fl, it = _design_bottom(geo, c, T_max)
    warn += cd.warnings
    design = losses(T_max, T_max, T_max, geo, hz, cd, c, T_air=c['T_air_design'])
    if not it['converged']:
        warn.append("floor / bottom-stack iteration did not converge")
    T_lim = c['floor']['T_limit_annexM']          # design basis: API 650 Annex M (Nate, 2026-11-18)
    if fl['T_floor_max'] > T_lim:
        warn.append(f"floor max {fl['T_floor_max']:.0f} C > {T_lim:.0f} C (API 650 Annex M)")
    if fl['T_floor_col'] > T_lim:
        warn.append(f"column spot {fl['T_floor_col']:.0f} C > {T_lim:.0f} C (API 650 Annex M; "
                    "known fix: concentrate fins under column lines; not modelled)")
    peak = losses(T_max, T_max, T_max, geo, hz, cd, c)

    def loss_fn(T_hot, T_top=None, T_bot=None):
        return losses(T_hot, T_hot if T_top is None else T_top, T_hot if T_bot is None else T_bot,
                      geo, hz, cd, c)['Q_total']

    # cycle average: time-average of every path over the history (losses are nonlinear in T)
    t, Ts, Tt, Tb = hist
    # evaluate on <= 60 points (smooth histories) then trapezoid in time
    idx = np.unique(np.linspace(0, len(t) - 1, min(len(t), 60)).astype(int))
    recs = [losses(Ts[i], Tt[i], Tb[i], geo, hz, cd, c) for i in idx]
    keys = ('Q_side', 'Q_edge', 'Q_top', 'Q_bot', 'Q_bot_col', 'Q_bot_thx', 'Q_total', 'q_bot')
    tt = t[idx]
    span = tt[-1] - tt[0]
    cyc = {k: (float(np.trapz([r[k] for r in recs], tt) / span) if span > 0 else recs[0][k]) for k in keys}
    E_loss_hist = float(np.trapz([r['Q_total'] for r in recs], tt))
    E_side_hist = float(np.trapz([r['Q_side'] + r['Q_edge'] for r in recs], tt))

    cost = costs(hz, geo, cd, fl, c)
    w = cost['weights']
    W = (hz.m_graphite + 1e3 * (w['thermax_t'] + w['column_graphite_t'] + w['oxide_t'] + w['steel_t'] + w['deck_t'])
) * G
    p_found = W / geo.A_tank
    if p_found > c['soil_allowable']:
        warn.append(f"foundation pressure {p_found / 1e3:.0f} kPa > soil allowable {c['soil_allowable'] / 1e3:.0f} kPa")
    eco = c['economics']
    MWh = cyc['Q_total'] * 8760 / 1e6
    econ = dict(annual_MWh=MWh, annual_cost=MWh * eco['elec_price'] / eco['heater_eff'])
    corr = {}
    if stick is not None:
        # roof + floor losses act on every cell; when they were in the stick run
        # (CellLossSink) only the perimeter paths (side + edges) are corrected here
        inc = stick.cell_losses_in_run
        e_loss = (E_side_hist if inc else E_loss_hist) / hz.N_ps   # J_th per stick, post-hoc part
        pen = e_loss * stick.eta_block
        corr = dict(E_loss_per_stick_kWh=e_loss / 3.6e6, E_stick_e_kWh=stick.E_stick_e / 3.6e6,
                    E_stick_e_net_kWh=(stick.E_stick_e - pen) / 3.6e6, penalty_frac=pen / stick.E_stick_e,
                    P_stick_net=(stick.E_stick_e - pen) / stick.t_discharge,
                    cell_losses_in_run=inc, E_sink_kWh=stick.E_sink / 3.6e6,
                    paths='side + edges' if inc else 'all paths',
                    note='electricity lost ~ eta_block x heat lost')
    T_all = np.concatenate([np.atleast_1d(x) for x in hist[1:]])
    sink = CellLossSink.from_system(hz, geo, cd, c, float(T_all.min()), max(float(T_all.max()), T_max))
    return TankResult(hz=hz, geo=geo, cols=cd, design=design, floor_design=fl, loss_peak=peak, loss_cycle=cyc,
                      cost=cost, loss_fn=loss_fn, foundation_pressure=p_found, economics=econ,
                      stick_correction=corr, warnings=warn, cfg=c, cell_sink=sink)


def tank_system(stick, cfg_tank=None):
    """Full system from per-stick inputs (StickInputs). The heater cavity comes from the
    heater block (pi (2 side_len / pi * area_ratio)^2, via stick_from_run)."""
    c = merge_defaults(cfg_tank)
    A_h = stick.A_hcav_default
    if A_h is None:
        raise ValueError("heater cavity area unknown: set heater.side_len and heater.area_ratio")
    hz = size_hot_zone(stick, c['P_rated'], c['duration_hr'], A_h)
    hist = (stick.t, stick.T_avg, stick.T_top, stick.T_bot)
    res = _system(hz, c, stick.T_max, hist, stick)
    res.warnings.insert(0, "cycle-average losses are the mean over the discharge only (no charge/hold cycle defined)")
    if abs(hz.energy_mismatch) > 0.01:
        res.warnings.insert(0, f"energy mismatch {hz.energy_mismatch * 100:+.1f}%: N_ps x E_stick_e vs "
                               "P_rated x duration (not resized)")
    return res


def tank_system_reference(cfg_tank=None, D_hot=35.4, H_block=10.0, m_graphite=10.49e6, E_th=3.0e9 * 3600,
                          T_min=1900.0, T_max=2400.0, N_ps=0, L_ps=3.0):
    """Hot zone given directly (brief §7.2 v3 design point). Cycle average = linear
    T sweep T_max -> T_min (flagged: no history)."""
    c = merge_defaults(cfg_tank)
    hz = HotZone(N_ps=N_ps, N_h=N_ps, D_hot=D_hot, H_block=H_block, m_graphite=m_graphite, E_th=E_th,
                 fill=m_graphite / 1450.0 / (math.pi * D_hot ** 2 / 4 * H_block), L_ps=L_ps)
    T = np.linspace(T_max, T_min, 41)
    res = _system(hz, c, T_max, (np.linspace(0, 1, 41), T, T, T))
    res.warnings.insert(0, "reference mode: cycle average = uniform sweep T_max -> T_min")
    return res


class CellLossSink:
    """Roof and floor insulation losses as a per-node sink for run_simulation /
    run_heating_simulation (Q_sink_fn). Every cell loses heat straight up through the
    roof and down through the bottom stack, so the plant's per-area fluxes q_top(T),
    q_bot(T) act on the top / bottom block rings of every stick and heater cell.

    Tabulated vs a uniform hot-face temperature (annual-mean air). The loss through a
    node's top (bottom) face is q(T_node) x ring area x f_open, with f_open = 1 / fill
    (hot-zone plan area / graphite plan area): the cavities' share of the roof and
    floor is drawn from the graphite around them (cavity ceilings and floors re-radiate
    to the block). The same f_open serves any block model of the layout (stick block,
    heater block). Side wall and edge losses only reach perimeter cells: post hoc."""

    def __init__(self, T_C, q_top, q_bot, f_open, info=None):
        self.T_C, self.q_top, self.q_bot = np.asarray(T_C), np.asarray(q_top), np.asarray(q_bot)
        self.f_open = float(f_open)
        self.info = info or {}

    @classmethod
    def from_system(cls, hz, geo, cd, c, T_lo, T_hi, n=40):
        T = np.linspace(T_lo - 150.0, T_hi + 100.0, n)
        qt = np.array([roof(x, geo, c)[0] for x in T]) / geo.A_hot
        qb = np.array([bottom_and_floor(cd, geo.A_hot, geo.D_tank, x, c['T_amb'], c)[0]['Q_bot']
                       for x in T]) / geo.A_hot
        return cls(T, qt, qb, 1.0 / hz.fill, info=dict(D_hot=hz.D_hot, N_ps=hz.N_ps, pitch=cd.pitch))

    def q(self, T_C):
        """(q_top, q_bot) [W/m2 of hot-zone plan area] at a uniform hot face T [C]."""
        return np.interp(T_C, self.T_C, self.q_top), np.interp(T_C, self.T_C, self.q_bot)

    def for_storage(self, storage):
        """f(T_dist [K]) -> (nA, nR) sink [W] for this block model."""
        b = storage.block
        nR = storage.num_rad_segments
        r = b.rad_in + np.arange(nR + 1) * (b.rad_out - b.rad_in) / nR
        a = np.pi * np.diff(r ** 2) * self.f_open
        nA = storage.num_ax_segments

        def fn(T_dist):
            T = np.asarray(T_dist, float) - C2K
            Q = np.zeros((nA, nR))
            Q[-1] += np.interp(T[-1], self.T_C, self.q_top) * a     # top ring (index -1) -> roof
            Q[0] += np.interp(T[0], self.T_C, self.q_bot) * a       # bottom ring -> bottom stack
            return Q
        return fn


def cell_loss_sink(m, cfg_tank=None, eta_e=0.45):
    """CellLossSink for a built model BEFORE its run, from a provisional plant: N_ps from
    the required energy with E_stick_e ~ eta_e x stored heat. The per-area fluxes hardly
    depend on plant size (floor-air heating moves q_bot ~1%); tank_system gives the
    final plant's sink (TankResult.cell_sink) for comparison."""
    c = merge_defaults(cfg_tank)
    g = _stick_geometry(m)
    if g['A_h'] is None:
        raise ValueError("heater cavity area unknown: set heater.side_len and heater.area_ratio")
    E_req = c['P_rated'] * c['duration_hr'] * 3600.0
    N = int(math.ceil(E_req / (eta_e * g['E_th'])))
    st = StickInputs(P_stick=c['P_rated'] / N, E_stick_e=eta_e * g['E_th'], E_stick_th=g['E_th'], V_block=g['V'],
                     r_out=g['r_out'], r_in=g['r_in'], H_block=g['H'], L_ps=g['L'], T_min=g['T_lo'] - C2K,
                     T_max=g['T_hi'] - C2K, rho_g=g['rho'], t_discharge=c['duration_hr'] * 3600.0)
    hz = size_hot_zone(st, c['P_rated'], c['duration_hr'], g['A_h'])
    geo = tank_geometry(hz, c)
    cd = _design_bottom(geo, c, st.T_max)[0]
    sink = CellLossSink.from_system(hz, geo, cd, c, st.T_min, st.T_max)
    sink.info['provisional'] = True
    return sink


# --------------------------------------------------------------------------
# f_3D calibration (v3 FEA point)
# --------------------------------------------------------------------------
def calibrate_f3d(cfg_tank=None, Q_target=774e3, A_hot=982.0, pitch=1.70, L_bot=2.0, T_hot=2400.0,
                  T1=1100.0, T_floor=85.0, m_graphite=10.49e6, t_top=1.03, L_graphite=None):
    """f_3D that makes 1-D columns + Thermax hit the FEA bottom heat (v3: 774 kW over
    982 m2, pitch 1.70 m, L_bot 2.0 m, 2400 C hot face, T1 ~1100 C, floor pinned 85 C).
    L_graphite: v3 graphite segment length if known (joint T becomes an output).
    Valid only near the v3 column geometry. -> dict."""
    c = merge_defaults(cfg_tank)
    c['columns'] = dict(c['columns'], pitch=pitch, T1=T1, L_graphite=L_graphite)   # FEA pitch, not computed
    c['L_bot'] = L_bot
    p_bear = G * (m_graphite / A_hot + c['thermax']['rho'] * t_top)
    cd = design_columns(A_hot, p_bear, T_hot, T_floor, L_bot, c['columns'])

    def Q(f):
        cc = dict(c, columns=dict(c['columns'], f_3D=f))
        return bottom_heat(cd, A_hot, T_hot, T_floor, cc)
    f = brentq(lambda f: Q(f)['Q_bot'] - Q_target, 0.1, 20.0, xtol=1e-6)
    b1, bf = Q(1.0), Q(f)
    return dict(f_3D=f, Q_1D=b1['Q_bot'], Q_col_1D=b1['Q_col_total'], Q_thx=b1['Q_thx'],
                Q_col_cal=bf['Q_col_total'], T_joint=bf['T_joint'], p_bear=p_bear, N_col=cd.N_col,
                A_top=cd.A_top, A_joint=cd.A_joint, L_g=cd.L_g, L_ox=cd.L_ox)
