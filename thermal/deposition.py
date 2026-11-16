import numpy as np


def vapor_pressure_graphite(T):
    """Carbon sublimation pressure [Pa]. Brewer et al."""
    return 10 ** (14.284 - 3.911e4 / T)


def vapor_pressure_tac_ta(T):
    """Ta vapour pressure over carbon-saturated TaC [Pa]."""
    return 10 ** (12.67 - 48789 / T)


def vapor_pressure_zrc_zr(T):
    """Zr vapour pressure over carbon-saturated ZrC [Pa]. Pollock."""
    return 10 ** (8.3846 - 43875 / T) * 101000.0


def vapor_pressure_tungsten(T):
    """W vapour pressure [Pa]. Plante & Sessoms."""
    return 10 ** (7.781 - 45385 / T) * 101000


def diffusion_coefficient_carbon_argon(T):
    """Effective C-in-Ar diffusivity [m²/s]. conv_factor=3.5 from CFD studies."""
    conv_factor = 3.5
    return 1e-5 * (T / 298) ** 1.75 * conv_factor


def diffusion_coefficient_tungsten_argon(T):
    """Effective W-in-Ar diffusivity [m²/s]. conv_factor=3.5 from CFD studies."""
    conv_factor = 3.5
    return 1e-6 * (T / 298) ** 1.75 * conv_factor


def diffusion_coefficient_carbon_tungsten(T):
    """C diffusivity through W solid [m²/s]. Buhsmer & Crayton."""
    return 9.22e-7 * np.exp(-169805 / (8.314 * T))


def gr_rho_molar(T):
    """Graphite molar density [mol/m³]."""
    return 155000.0


def tac_rho_molar(T):
    """TaC molar density [mol/m³]."""
    return 92166.0


def zrc_rho_molar(T):
    """ZrC molar density [mol/m³]."""
    return 64020.0


def w_rho_molar(T):
    """W molar density [mol/m³]."""
    return 105263.0


def deposition_rate_tinless(
    T_emitter,
    T_ps=600.0,
    d=0.08,
    area_ratio=2.0,
    surface_material='graphite',
    th_tac=0.0,
    th_w=0.0,
    f_C_sol_W=0.003,
    P_total=101325,
    T_graphite=None,
):
    """
    Instantaneous deposition rate on the powerstick cells for the tinless geometry.

    Physics basis
    -------------
    Carbon-saturated carbide on graphite ('tac' case):
        Solid-state C supply rate through the carbide exceeds gas-phase removal by
        ~3-5 orders of magnitude (D_solid * C_solid >> D_gas * C_gas adjusted for
        path length). The carbide surface therefore maintains a_C = 1 (carbon-saturated)
        and P_C at the carbide inner face = P_C(graphite). A carbide coating does NOT
        reduce carbon deposition relative to bare graphite.

    Tungsten foil ('tungsten' case):
        W's low C solubility (~0.3 at% at a_C = 1) is the actual barrier mechanism,
        not its diffusivity. C behind the foil is at P_graphite (set by the
        carbon-saturated TaC/graphite behind it). The coupled solid-gas series model
        uses a partition coefficient k_s = C_W_solid / C_gas to consistently combine
        the W solid diffusion resistance with the argon gas resistance.
        Note: the foil carburises in service; steady-state barrier performance degrades
        once the W converts to WC and becomes carbon-saturated itself.

    Tungsten shell liner ('w_shell' case):
        The liner itself is a solid W tube (thickness th_w). The outer face is at
        T_graphite and is in direct contact with the graphite C environment; P_C(T_graphite)
        drives carbon into the outer face. The outer surface develops a WC layer (WC
        emissivity applies on that side) but the shell is assumed thick enough to remain
        predominantly W metal, so C diffusivity through W (not WC) applies throughout.
        The inner face at T_emitter faces the PS and emits W vapour. Same series-resistance
        model as 'tungsten', but driven by the graphite outer-face C loading rather than an
        inner liner surface. Requires T_graphite to be provided.

    Parameters
    ----------
    T_emitter : float
        Temperature of the innermost hot surface facing the PS [K].
        For 'w_shell': temperature of the W shell inner face.
    T_ps : float
        Powerstick/cell temperature [K]. Default 600 K.
    d : float
        Effective emitter-to-PS transport distance [m].
    area_ratio : float
        A_emitter / A_ps.
    surface_material : str
        Innermost surface: 'graphite', 'tac', 'zrc', 'tungsten', or 'w_shell'.
    th_tac : float
        TaC liner wall thickness [m] (informational — does not affect C deposition rate).
    th_w : float
        W foil or shell thickness [m].
    f_C_sol_W : float
        Carbon mole-fraction solubility in W at a_C = 1. Default 0.003 (~0.3 at%).
    P_total : float
        Total argon pressure [Pa]. Default 101325.
    T_graphite : float or None
        Temperature of the graphite block inner face [K]. Required when
        surface_material='w_shell' (outer face of W shell is in contact with graphite).

    Returns
    -------
    dict
        'total'   : total deposition rate on PS [nm/yr]
        'C'       : carbon deposition rate [nm/yr]
        'metal'   : metal species (Ta or W) deposition rate [nm/yr]
        'P_C'     : C vapour pressure driving deposition [Pa]  (diagnostic)
        'P_metal' : metal vapour pressure at emitter surface [Pa]  (diagnostic)
    """
    R     = 8.314
    nm_yr = 3.154e16

    T_avg   = (T_emitter + T_ps) / 2
    D_C_gas = diffusion_coefficient_carbon_argon(T_avg)
    D_W_gas = diffusion_coefficient_tungsten_argon(T_avg)
    h_C     = D_C_gas / d
    h_W     = D_W_gas / d

    P_C_sat = vapor_pressure_graphite(T_emitter)
    C_C_sat = P_C_sat / (R * T_emitter)

    P_C_ret = P_C_sat   # diagnostic return value; overridden for w_shell

    # ── Carbon ──────────────────────────────────────────────────────────────────
    if surface_material in ('graphite', 'tac', 'zrc'):
        # ZrC is carbon-saturated (like TaC): C activity = 1, same C deposition as graphite.
        flux_C = C_C_sat * h_C

    elif surface_material == 'tungsten':
        D_C_W    = diffusion_coefficient_carbon_tungsten(T_emitter)
        C_W_back = w_rho_molar(T_emitter) * f_C_sol_W
        k_s      = C_W_back / C_C_sat if C_C_sat > 0 else 1.0

        if th_w > 0:
            flux_C = h_C * D_C_W * C_W_back / (th_w * h_C + D_C_W * k_s)
        else:
            flux_C = C_C_sat * h_C

    elif surface_material == 'w_shell':
        # Outer face at T_graphite: C loading in W set by P_C(T_graphite).
        # Inner face at T_emitter: emits W vapour, drains C to gas gap.
        # Same series model as 'tungsten' but C_W_outer driven by the hotter graphite
        # outer face rather than the inner (cooler) emitter surface.
        if T_graphite is None:
            raise ValueError("T_graphite is required for surface_material='w_shell'")
        P_C_outer     = vapor_pressure_graphite(T_graphite)
        C_C_outer_gas = P_C_outer / (R * T_graphite)           # mol/m³ at outer face
        C_W_outer     = w_rho_molar(T_graphite) * f_C_sol_W    # C dissolved in W solid at outer face
        k_s           = C_W_outer / C_C_outer_gas if C_C_outer_gas > 0 else 1.0
        D_C_W         = diffusion_coefficient_carbon_tungsten(T_emitter)
        P_C_ret       = P_C_outer
        if th_w > 0:
            flux_C = h_C * D_C_W * C_W_outer / (th_w * h_C + D_C_W * k_s)
        else:
            flux_C = C_C_outer_gas * h_C

    dep_C_nm_yr = flux_C * area_ratio / gr_rho_molar(T_emitter) * nm_yr

    # ── Metal vapour ─────────────────────────────────────────────────────────────
    dep_metal_nm_yr = 0.0
    P_metal_surface = 0.0

    if surface_material == 'tac':
        P_metal_surface = vapor_pressure_tac_ta(T_emitter)
        flux_metal      = (P_metal_surface / (R * T_emitter)) * h_W
        dep_metal_nm_yr = flux_metal * area_ratio / tac_rho_molar(T_emitter) * nm_yr

    elif surface_material == 'zrc':
        P_metal_surface = vapor_pressure_zrc_zr(T_emitter)
        flux_metal      = (P_metal_surface / (R * T_emitter)) * h_W
        dep_metal_nm_yr = flux_metal * area_ratio / zrc_rho_molar(T_emitter) * nm_yr

    elif surface_material in ('tungsten', 'w_shell'):
        P_metal_surface = vapor_pressure_tungsten(T_emitter)
        flux_metal      = (P_metal_surface / (R * T_emitter)) * h_W
        dep_metal_nm_yr = flux_metal * area_ratio / w_rho_molar(T_emitter) * nm_yr

    return {
        'total':   dep_C_nm_yr + dep_metal_nm_yr,
        'C':       dep_C_nm_yr,
        'metal':   dep_metal_nm_yr,
        'P_C':     P_C_ret,
        'P_metal': P_metal_surface,
    }
