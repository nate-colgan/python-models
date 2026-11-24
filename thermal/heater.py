"""
Heater element geometry and electrical sizing (YAML `heater` section).

Two element types (heater.type):

  rod   square graphite rod of side side_len (the original model). Radiating radius
        r_h = 2 side_len / pi (same perimeter); cross-section side_len^2.

  tube  tubular graphite resistive element, outer diameter D, wall t = wall_ratio x D,
        sized so the DC supply voltage V drives the heater power P through it:

            R = V^2 / P,   R = rho_e legs^2 H / A,   A = pi D^2 w (1 - w),   w = t / D
            ->  D = sqrt(rho_e legs^2 H P / (V^2 pi w (1 - w)))

        H   heated length = block height (the heater spans the block)
        legs  the tube slit lengthwise into `legs` equal strips in series (1 = current down
            the tube and back through a separate return; 2 = U-element, both terminals on
            top). Each strip has A / legs, so R grows with legs^2.
        rho_e  the highest resistivity over the heater's operating range [T_lo,
            max_temperature] (properties.heater_graphite_resistivity: Mersen 2020 data to
            2002 C, extrapolated above), so the supply delivers P everywhere in that range; at lower resistance the
            supply turns the voltage down.
        P   the heater power: heater.max_power when set (the optimizer's power search), else
            the cap max_power_per_m x H. The element follows the power, so the charge run
            that tests a power also tests the element sized for it: a lower power gives a
            thinner tube with less radiating area, which runs hotter, and the run checks
            it still charges within max_temperature (view factor to the cavity, emissivity).

The element spans the block height and passes up through the top insulation to the
deck, so its graphite mass is A x (H_block + t_top) x density.
"""
import math

import numpy as np

from .properties import heater_graphite_resistivity


def heater_max_power(h, H):
    """Heater max electrical power [W] for a heater of length H [m] (it spans the block
    height): heater.max_power if given (absolute override, used by the optimizer's
    power search), else heater.max_power_per_m x H."""
    if h.get('max_power') is not None:
        return float(h['max_power'])
    if h.get('max_power_per_m') is None:
        raise KeyError("heater: give max_power_per_m [W/m] (or an absolute max_power [W])")
    return float(h['max_power_per_m']) * float(H)


def element(h, H, T_lo, P=None):
    """Heater element for the YAML heater dict h, heated length H [m], lowest operating
    temperature T_lo [K] (the block T_min) and power P [W] (default heater_max_power).
    -> dict(type, P, r_h (radiating radius), D (outer diameter; rod: 2 r_h), t (wall),
            A_cross [m2], density, and for a tube V, R [ohm], I [A], rho_e [ohm m], legs)."""
    P = float(heater_max_power(h, H) if P is None else P)
    kind = h.get('type', 'rod')
    dens = float(h.get('density', 1850.0))
    if kind == 'rod':
        s = float(h['side_len'])
        return dict(type='rod', P=P, r_h=2 * s / math.pi, D=4 * s / math.pi, t=s / 2, A_cross=s * s,
                    density=dens)
    if kind != 'tube':
        raise ValueError(f"heater.type must be 'tube' or 'rod', not {kind!r}")
    V = float(h['voltage'])
    w = float(h['wall_ratio'])
    legs = int(h.get('legs', 1))
    if not 0 < w <= 0.5:
        raise ValueError(f"heater.wall_ratio {w} must be in (0, 0.5]")
    T_hi = max(float(h['max_temperature']), float(T_lo))
    rho_e = float(np.max(heater_graphite_resistivity(np.linspace(float(T_lo), T_hi, 41))))
    R = V * V / P
    A = rho_e * legs ** 2 * float(H) / R
    D = math.sqrt(A / (math.pi * w * (1 - w)))
    return dict(type='tube', P=P, r_h=D / 2, D=D, t=w * D, A_cross=A, density=dens, V=V, R=R, I=P / V,
                rho_e=rho_e, legs=legs)


def element_mass(el, L_total):
    """Graphite mass [kg] of an element over a length L_total [m]."""
    return el['A_cross'] * float(L_total) * el['density']


def describe(el):
    """One-line description of an element dict."""
    if el['type'] == 'tube':
        return (f"tube OD {el['D']*1e3:.0f} mm x {el['t']*1e3:.1f} mm wall, {el['R']*1e3:.2f} mOhm, "
                f"{el['I']/1e3:,.1f} kA at {el['V']:g} V")
    return f"rod {el['A_cross'] ** 0.5 * 1e3:.0f} mm square"
