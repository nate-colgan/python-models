import numpy as np


def Sn_rho(temp):
    #return liquid Tin density, Stankus
    return 6982.2 - 0.75643 * (temp - 505) + 1.074e-4 * (temp - 505) ** 2 - 2.194e-8 * (temp - 505) ** 3

def Sn_tcx(temp):
    #return liquid Tin thermal conductivity [W/mK] fix this one?
    return 13.9 + 0.02868 * temp

def Sn_mu(temp):
    #return liquid Tin viscosity [Pas]
    return 3.908e-4 * np.exp(790.7 / temp)

def Sn_cp(temp):
    #return liquid Tin cp [J/kgK]
    #set constant cp for now, need to tract enthalpy instead of temp if using varying cp
    return 300
    #return 182.7 + 0.05177 * temp + 1.086e7 * temp ** (-2)

def Sn_Pr(temp):
    #return Prandtl number
    return Sn_cp(temp) * Sn_mu(temp) / Sn_tcx(temp)

def gr_cp(temp):
    #return graphite CP, from Entegris [J/kgK]
    return (0.44391 + 0.30795e-4 * temp - 0.61257e5 * temp ** (-2) + 0.10795e8 * temp ** (-3)) * 4184

def gr_rho(temp):
    #return graphite density [kg/m^3] fix
    return 1860

def tac_emis_temp(temp):
    #return Tac thermal emissivity
    return 0.0238 + 1.084e-4 * temp

def tac_emis_wavelength_temp(wavelength, temp):
    #return Tac thermal emissivity as a function of wavelength and temperature
    return -0.0351 + 8.15e-5 * temp + (0.3008 - 4.46e-5 * temp) / (1e6 * wavelength)

def tac_emis_binned_1eV(temp):
    #return Tac thermal emissivity as a function of temperature, binned to above and below 1.24 um
    return np.array([0.204 + 7.18e-5 * temp, 0.0685 + 7.58e-5 * temp])


def zrc_emis_temp(temp: float) -> float:
    """ZrC total hemispherical emissivity vs temperature [K].

    Fit to Table A.6 of Shabalin 'Ultra-High Temperature Materials II':
    εT ≈ 0.68 @ 1000 K, 0.59 @ 2000 K, 0.55 @ 2500 K, 0.50 @ 3000 K.
    """
    return float(np.clip(0.77 - 9e-5 * temp, 0.35, 1.0))


def zrc_emis_wavelength_temp(wavelength: float, temp: float) -> float:
    """ZrC spectral emissivity vs wavelength [µm] and temperature [K].

    ZrC is near-grey with a slight increase toward longer wavelengths (the integral
    emissivity exceeds the monochromatic value at 0.665 µm at all temperatures).
    Calibrated to Table A.6: ε(0.665 µm) ≈ 0.60 @ 1000 K, 0.48 @ 2000 K,
    0.42 @ 2500 K, 0.40 @ 3000 K. Shabalin 'Ultra-High Temperature Materials II'.
    """
    return float(np.clip(0.70 - 1e-4 * temp + 0.02 * (wavelength - 0.665), 0.0, 1.0))


def gr_emis(temp: float) -> float:
    """
    Calculate the emissivity of graphite as a function of temperature.

    Parameters:
    temp (float): Temperature (K).

    Returns:
    float: Emissivity of graphite.
    """
    return 0.679 + 6e-5 * temp
    #return 0.8

def wc_emis_grey(temp: float) -> float:
    """
    Calculate the emissivity of tungsten as a function of temperature.

    Parameters:
    temp (float): Temperature (K).

    Returns:
    float: Emissivity of tungsten.
    """
    return 0.4


def w_emis_total(temp: float) -> float:
    """
    Planck-weighted total hemispherical emissivity of tungsten.

    Degree-5 polynomial fit in t = T/1000 to the Planck-weighted integral
    of the spectral emissivity table in w_emis_spectral:

        ε_total(T) = ∫ ε(λ,T) · B(λ,T) dλ  /  ∫ B(λ,T) dλ

    Fit range 200–3100 K; max error ±0.004, rms 0.001.
    Both numerator and denominator integrated over the w_emis_spectral table range (0.01–999 um).
    Clamped at the table boundaries outside that range.
    """
    t = float(np.clip(temp, 200.0, 3100.0)) / 1000.0
    return float(np.polyval([
         5.7075219580e-03,
        -3.5937520870e-02,
         4.0250389734e-02,
         1.1090626888e-01,
        -9.8495584433e-02,
         6.8779609057e-02,
    ], t))

def w_emis_spectral_vec(temperature: float, wavelengths_um: np.ndarray) -> np.ndarray:
    """
    Vectorized W spectral emissivity for a scalar temperature and array of wavelengths.

    wavelengths_um : wavelengths in microns (numpy array)
    Returns        : emissivity array, same shape as wavelengths_um
    Out-of-range wavelengths are clamped to the table boundary values.
    """
    _T  = np.array([0., 700., 1600., 2600., 3100.])
    _WL = np.array([0., 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.65, 0.7,
                    0.75, 0.8, 0.85, 0.9, 0.95, 1., 1.5, 2., 2.5, 3.,
                    3.5, 4., 4.5, 5., 1000.])
    _EV = np.array([
        [0.539, 0.539, 0.539, 0.541, 0.542, 0.54,  0.533, 0.508, 0.497, 0.486,
         0.474, 0.463, 0.452, 0.441, 0.43,  0.419, 0.262, 0.141, 0.101, 0.082,
         0.069, 0.062, 0.056, 0.05,  0.05],   # T=0
        [0.539, 0.539, 0.539, 0.541, 0.542, 0.54,  0.533, 0.508, 0.497, 0.486,
         0.474, 0.463, 0.452, 0.441, 0.43,  0.419, 0.262, 0.141, 0.101, 0.082,
         0.069, 0.062, 0.056, 0.05,  0.05],   # T=700
        [0.557, 0.557, 0.573, 0.578, 0.564, 0.546, 0.532, 0.507, 0.495, 0.485,
         0.474, 0.462, 0.451, 0.44,  0.429, 0.417, 0.299, 0.219, 0.171, 0.137,
         0.114, 0.096, 0.08,  0.071, 0.071],  # T=1600
        [0.383, 0.383, 0.453, 0.465, 0.467, 0.466, 0.461, 0.448, 0.44,  0.433,
         0.425, 0.417, 0.409, 0.4,   0.39,  0.383, 0.314, 0.265, 0.23,  0.211,
         0.197, 0.187, 0.183, 0.183, 0.183], # T=2600
        [0.41,  0.41,  0.456, 0.461, 0.46,  0.456, 0.451, 0.436, 0.428, 0.42,
         0.413, 0.405, 0.398, 0.39,  0.384, 0.378, 0.32,  0.291, 0.276, 0.26,
         0.25,  0.24,  0.24,  0.24,  0.24],  # T=3100
    ])
    T_c   = float(np.clip(temperature, _T[0], _T[-1]))
    ti    = int(np.clip(np.searchsorted(_T, T_c, side='right') - 1, 0, len(_T) - 2))
    t_frac = (T_c - _T[ti]) / (_T[ti + 1] - _T[ti])
    row_lo = np.interp(wavelengths_um, _WL, _EV[ti])
    row_hi = np.interp(wavelengths_um, _WL, _EV[ti + 1])
    return row_lo + t_frac * (row_hi - row_lo)


def w_emis_spectral(temperature: float, wavelength: float) -> float:
    """
    Calculate the emissivity of tungsten as a function of temperature and wavelength.

    Parameters:
    temperature (float): Temperature (K).
    wavelength(float): wavelength (micron)

    Returns:
    float: Emissivity of tungsten.
    """
    et = [
        [0, 0.0, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 1000],
        [0, 0.539, 0.539, 0.539, 0.541, 0.542, 0.54, 0.533, 0.508, 0.497, 0.486, 0.474, 0.463, 0.452, 0.441,
         0.43, 0.419, 0.262, 0.141, 0.101, 0.082, 0.069, 0.062, 0.056, 0.05, 0.05],
        [700, 0.539, 0.539, 0.539, 0.541, 0.542, 0.54, 0.533, 0.508, 0.497, 0.486, 0.474, 0.463, 0.452, 0.441,
         0.43, 0.419, 0.262, 0.141, 0.101, 0.082, 0.069, 0.062, 0.056, 0.05, 0.05],
        [1600, 0.557, 0.557, 0.573, 0.578, 0.564, 0.546, 0.532, 0.507, 0.495, 0.485, 0.474, 0.462, 0.451, 0.44,
         0.429, 0.417, 0.299, 0.219, 0.171, 0.137, 0.114, 0.096, 0.08, 0.071, 0.071],
        [2600, 0.383, 0.383, 0.453, 0.465, 0.467, 0.466, 0.461, 0.448, 0.44, 0.433, 0.425, 0.417, 0.409, 0.4, 0.39,
         0.383, 0.314, 0.265, 0.23, 0.211, 0.197, 0.187, 0.183, 0.183, 0.183],
        [3100, 0.41, 0.41, 0.456, 0.461, 0.46, 0.456, 0.451, 0.436, 0.428, 0.42, 0.413, 0.405, 0.398, 0.39, 0.384,
         0.378, 0.32, 0.291, 0.276, 0.26, 0.25, 0.24, 0.24, 0.24, 0.24]
    ]
    temperatures = np.array([row[0] for row in et[1:]])
    wavelengths = np.array(et[0][1:])
    ev = np.array([row[1:] for row in et[1:]])
    if not (min(temperatures) <= temperature <= max(temperatures)):
        print("temp:", temperature)
        raise ValueError("Temperature out of range.")
    if not (min(wavelengths) <= wavelength <= max(wavelengths)):
        raise ValueError("Wavelength out of range.")

    temp_idx = np.searchsorted(temperatures, temperature) - 1
    wl_idx = np.searchsorted(wavelengths, wavelength) - 1

    T1, T2 = temperatures[temp_idx], temperatures[temp_idx + 1]
    WL1, WL2 = wavelengths[wl_idx], wavelengths[wl_idx + 1]
    Q11 = ev[temp_idx, wl_idx]
    Q12 = ev[temp_idx, wl_idx + 1]
    Q21 = ev[temp_idx + 1, wl_idx]
    Q22 = ev[temp_idx + 1, wl_idx + 1]

    interpolated_emissivity = (
        Q11 * (T2 - temperature) * (WL2 - wavelength) +
        Q21 * (temperature - T1) * (WL2 - wavelength) +
        Q12 * (T2 - temperature) * (wavelength - WL1) +
        Q22 * (temperature - T1) * (wavelength - WL1)
    ) / ((T2 - T1) * (WL2 - WL1))

    return interpolated_emissivity

def gr_tcx(temp: float) -> float:
    """
    Calculate the thermal conductivity of graphite as a function of temperature.

    Parameters:
    temp (float): Temperature (K).

    Returns:
    float: Thermal conductivity of graphite (W/mK).
    """
    tcx_arr = np.array([[20,	115],
    [200,	100],
    [400,	85],
    [600,	73],
    [800,	62],
    [1000,	54],
    [1200,	49],
    [1400,	45],
    [1600,	42],
    [1800,	40],
    [2000,	39],
    [2200,	37],
    [2400,	34]]
                     )
    return np.interp(temp, tcx_arr[:,0], tcx_arr[:,1])

def w_tcx(temp: float) -> float:
    """
    Calculate the thermal conductivity of tungsten as a function of temperature.

    Parameters:
    temp (float): Temperature (K).

    Returns:
    float: Thermal conductivity of tungsten (W/mK).
    """
    if temp < 300:
        return 173
    elif temp < 1000:
        return 173 - 0.03 * (temp - 300)
    else:
        return 152 - 0.01 * (temp - 1000)

def storage_rho(temp):
    #storage graphite density
    #return 1743
    return 1450

def storage_cp(temp):
    #storage graphite specific heat
    #replaced with fixed cp until better enthalpy handling implemented
    cp_arr = np.array([1.27, 1.92, 1.99, 2.13])
    temp_C = np.array([300, 750, 1200, 1995])
    temp_C = np.append(temp_C, 2400)
    cp_arr = np.append(cp_arr, 2.13)
    temp_C_interp = np.minimum(temp - 273.15, 2400)   # array-safe clamp
    return 2000
    #return np.interp(temp_C_interp, temp_C, cp_arr) * 1000

def storage_tcx(temp):
    #storage graphite conductivity
    temp_C = np.array([300, 750, 1200, 1995])
    conductivity = np.array([31.5, 28.7, 24.8, 23.1])
    temp_C = np.append(temp_C, 2400)
    conductivity = np.append(conductivity, conductivity[-1] + (conductivity[-1] - conductivity[-2]) / (temp_C[-2] - temp_C[-3]) * (temp_C[-1] - temp_C[-2]))
    temp_C_interp = np.minimum(temp - 273.15, 2400)   # array-safe clamp
    return 20
    #return np.interp(temp_C_interp, temp_C, conductivity)

# Thermax (640 kg/m3), measured: k = K0 * exp(B * T[C])  [W/m-K]
THERMAX_K0 = 0.04916
THERMAX_B = 0.0011


def thermax_tcx(temp):
    """Thermax thermal conductivity [W/m-K], temp in K."""
    return THERMAX_K0 * np.exp(THERMAX_B * (np.asarray(temp, float) - 273.15))


def thermax_tcx_integral(T_lo, T_hi):
    """Exact Kirchhoff integral of thermax_tcx from T_lo to T_hi [W/m], temps in K."""
    return THERMAX_K0 / THERMAX_B * (np.exp(THERMAX_B * (np.asarray(T_hi, float) - 273.15))
                                     - np.exp(THERMAX_B * (np.asarray(T_lo, float) - 273.15)))


def ins_tcx(temp):
    #Carbon insulation thermal conductivity in Argon
    temps = np.array([0, 400, 800, 1000, 2000, 2500])
    tcx = np.array([0.09, 0.09, 0.176, 0.35, 0.94, 0.94 + (0.94 - 0.35) * 500 / 1000])
    temp_C = temp - 273.15
    return np.interp(temp_C, temps, tcx)

# Parabolic fit coefficients: η = a·T² + b·T + c  (T in K, η dimensionless)
# Columns: (a, b, c) for bandgap wavelengths 1.24, 1.38, 1.55 μm
_PANEL_EFF_COEFFS = {
    1.55: (-2.4659e-7,  8.7159e-4, -0.33288),
    1.38: (-2.7235e-7,  1.0632e-3, -0.57389),
    1.24: (-2.9534e-7,  1.2511e-3, -0.83965),
}

def panel_efficiency(bandgap: float, T_emitter: float, flux: float = None) -> float:
    """
    Return TPV panel efficiency as a parabolic function of emitter temperature,
    with an optional flux-dependent degradation factor accounting for cell
    heating at high irradiance:

        η = η_0(T) · 0.998^(2 · flux · η_0)

    Parameters
    ----------
    bandgap   : PV cell bandgap wavelength [μm] — must be 1.24, 1.38, or 1.55
    T_emitter : emitter temperature [K]
    flux      : incident cell flux [W/cm²] — degradation applied when provided

    Returns
    -------
    float : panel efficiency η (dimensionless)

    Raises
    ------
    ValueError if bandgap is not one of the three supported values.
    """
    coeffs = _PANEL_EFF_COEFFS.get(bandgap)
    if coeffs is None:
        raise ValueError(
            f"Invalid bandgap {bandgap} μm — must be one of "
            f"{sorted(_PANEL_EFF_COEFFS)}."
        )
    a, b, c = coeffs
    eta_0 = a * T_emitter**2 + b * T_emitter + c
    if flux is not None:
        return eta_0 * (0.998 ** (2 * flux * (1-eta_0)))
    return eta_0
