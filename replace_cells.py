import json

with open("DemoThermalModel.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# ── Cell 29: updated function cell ───────────────────────────────────────────
func_source = '''\
def spectral_net_radiation(T_hot, T_cold, emissivity_table, VF, A_hot, emis_hot, f_bare, f_active):
    """
    Net spectral radiative heat transfer [W] from hot to cold surface for a 2-surface
    enclosure.  Reuses the same Planck integration pattern as PowerStick.incident_energy:
    - logspace wavelength grid, forward differences below bandgap, backward above
    - denominator: VF*(1-e_hot)/e_hot + 1 + (1-e_cold)/e_cold  (per-band e_cold from table)

    Parameters
    ----------
    T_hot          : float  Hot surface temperature [K]
    T_cold         : float  Cold surface temperature [K]
    emissivity_table : list  Same format as PowerStick.emissivity_table:
                             [(wavelength_range_tuple, emis_active, emis_bare, emis_inactive), ...]
                             wavelengths in microns
    VF             : float  View factor from hot to cold (A_cold/A_hot for concentric cylinders)
    A_hot          : float  Hot surface area [m^2]
    emis_hot       : float  Hot surface emissivity (spectrally uniform)
    f_bare         : float  Fraction of cold surface with no PV cell
    f_active       : float  Fraction of non-bare area that is active PV

    Returns
    -------
    float  Net heat transfer [W], hot -> cold
    """
    h   = 6.62607015e-34
    c   = 2.99792458e8
    k_B = 1.380649e-23

    wavelength_min = emissivity_table[0][0][0]
    wavelength_max = emissivity_table[-1][0][1]
    band_gap       = emissivity_table[0][0][1]   # boundary between first and second band
    n_bins = 500

    wavelengths   = np.logspace(np.log10(wavelength_min), np.log10(wavelength_max), n_bins)
    wavelengths_m = wavelengths * 1e-6
    planck = lambda T: np.pi * (2*h*c**2) / wavelengths_m**5 / np.expm1(h*c / (wavelengths_m * k_B * T))
    E_hot  = planck(T_hot)
    E_cold = planck(T_cold)

    Q_net = 0.0
    for i, wavelength in enumerate(wavelengths):
        # bandwidth (matches PowerStick.incident_energy convention)
        if wavelength < band_gap:
            dl = wavelengths_m[i+1] - wavelengths_m[i]
        else:
            dl = wavelengths_m[i] - wavelengths_m[i-1]

        # effective cold-surface emissivity for this wavelength band
        for (wavelength_range, emis_active, emis_bare, emis_inactive) in emissivity_table:
            if wavelength_range[0] <= wavelength < wavelength_range[1]:
                emis_c = (f_bare * emis_bare
                          + (1 - f_bare) * f_active       * emis_active
                          + (1 - f_bare) * (1 - f_active) * emis_inactive)
                break

        denom  = VF * (1 - emis_hot) / emis_hot + 1 + (1 - emis_c) / emis_c
        Q_net += (E_hot[i] - E_cold[i]) * dl * A_hot / denom

    return Q_net


def graphite_to_powerstick_heat_transfer(T_outer, T_ps, k_gr, emis_gr,
                                          emissivity_table, f_bare, f_active,
                                          D_in, D_out, D_ps, L,
                                          tol=1e-2, max_iter=50):
    """
    Heat transfer from an annular graphite region to a powerstick via series
    conduction (outer wall -> inner wall) and multi-band spectral radiation
    (inner wall -> powerstick).

    Parameters
    ----------
    T_outer          : float  Temperature at outer wall of graphite annulus [K]
    T_ps             : float  Powerstick temperature [K]
    k_gr             : float  Graphite thermal conductivity [W/mK]
    emis_gr          : float  Graphite inner-surface emissivity (spectrally uniform)
    emissivity_table : list   Band emissivities for powerstick surface:
                              [(wavelength_range, emis_active, emis_bare, emis_inactive), ...]
                              wavelength_range in microns, first band is below bandgap
    f_bare           : float  Bare (no PV) area fraction of powerstick
    f_active         : float  Active PV fraction of non-bare powerstick area
    D_in             : float  Inner diameter of graphite annulus [m]
    D_out            : float  Outer diameter of graphite annulus [m]
    D_ps             : float  Powerstick outer diameter [m]
    L                : float  Length [m]
    tol              : float  Convergence tolerance on T_inner [K]
    max_iter         : int    Maximum iterations

    Returns
    -------
    Q        : float  Heat transfer rate [W]
    T_inner  : float  Converged graphite inner wall temperature [K]
    """
    A_gr_inner = np.pi * D_in * L
    VF_gr_ps   = D_ps / D_in          # view factor: graphite inner wall -> powerstick
    R_cond     = RprimeCondCyl(D_in, D_out, k_gr) / L

    T_inner = (T_outer + T_ps) / 2    # initial guess

    for _ in range(max_iter):
        Q = spectral_net_radiation(T_inner, T_ps, emissivity_table, VF_gr_ps,
                                   A_gr_inner, emis_gr, f_bare, f_active)
        T_inner_new = T_outer - Q * R_cond
        if abs(T_inner_new - T_inner) < tol:
            return Q, T_inner_new
        T_inner = T_inner_new

    return Q, T_inner
'''

# ── Cell 30: updated sweep cell ───────────────────────────────────────────────
sweep_source = '''\
# ── Fixed inputs ──────────────────────────────────────────────────────────────
V_graphite =       # total graphite annulus volume [m^3] — fill in
L          =       # length [m] — fill in
D_ps       =       # powerstick outer diameter [m] — fill in
T_ps       =       # powerstick temperature [K] — fill in

# Multi-band powerstick emissivity definition
bandgap          =       # bandgap wavelength [microns], e.g. 1.77 for GaSb
# Below-bandgap emissivities (band 1)
emis_active_1    =       # active PV cell emissivity below bandgap
emis_bare_1      =       # bare surface emissivity below bandgap
emis_inactive_1  =       # inactive cell emissivity below bandgap
# Above-bandgap emissivities (band 2)
emis_active_2    =       # active PV cell emissivity above bandgap
emis_bare_2      =       # bare surface emissivity above bandgap
emis_inactive_2  =       # inactive cell emissivity above bandgap
# Fractional areas
f_bare           =       # fraction of powerstick with no PV cell
f_active         =       # fraction of non-bare area that is active PV

emissivity_table = [
    ((0.0, bandgap), emis_active_1, emis_bare_1, emis_inactive_1),
    ((bandgap, 1e4), emis_active_2, emis_bare_2, emis_inactive_2),
]

# ── Sweep parameters ──────────────────────────────────────────────────────────
k_gr_vals    = [15, 30]           # graphite thermal conductivity [W/mK]
T_outer_vals = [2273, 2473, 2673] # outer graphite wall temperature [K]
D_in_vals    = [0.75, 1, 1.25]    # inner graphite diameter [m] — adjust units to match above

# ── Sweep ─────────────────────────────────────────────────────────────────────
results = {}
for k_gr in k_gr_vals:
    for D_in in D_in_vals:
        D_out   = np.sqrt(4 * V_graphite / (np.pi * L) + D_in**2)
        emis_gr = gr_emis(np.mean(T_outer_vals))
        Q_vals, T_inner_vals = [], []
        for T_outer in T_outer_vals:
            Q, T_inner = graphite_to_powerstick_heat_transfer(
                T_outer, T_ps, k_gr, emis_gr,
                emissivity_table, f_bare, f_active,
                D_in, D_out, D_ps, L
            )
            Q_vals.append(Q)
            T_inner_vals.append(T_inner)
        results[(k_gr, D_in)] = (Q_vals, T_inner_vals)

# ── Plot ──────────────────────────────────────────────────────────────────────
T_outer_C = np.array(T_outer_vals) - 273.15
colors    = plt.cm.plasma(np.linspace(0.15, 0.85, len(D_in_vals)))
ls_map    = {k: style for k, style in zip(k_gr_vals, ['-', '--'])}

fig_Q, ax_Q = plt.subplots(figsize=(8, 5))
fig_T, ax_T = plt.subplots(figsize=(8, 5))

for ci, D_in in enumerate(D_in_vals):
    D_out = np.sqrt(4 * V_graphite / (np.pi * L) + D_in**2)
    for k_gr in k_gr_vals:
        Q_vals, T_inner_vals = results[(k_gr, D_in)]
        lbl = f"D_in={D_in}, k={k_gr} W/mK"
        ax_Q.plot(T_outer_C, np.array(Q_vals) / 1e3,
                  color=colors[ci], linestyle=ls_map[k_gr], marker='o', label=lbl)
        ax_T.plot(T_outer_C, np.array(T_inner_vals) - 273.15,
                  color=colors[ci], linestyle=ls_map[k_gr], marker='o', label=lbl)

for ax, ylabel, title in [
    (ax_Q, "Q [kW]",       "Power output"),
    (ax_T, "T_inner [°C]", "Graphite inner wall temperature"),
]:
    ax.set_xlabel("T_outer [°C]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    ax.legend(fontsize=8, ncol=2)

fig_Q.tight_layout()
fig_T.tight_layout()
plt.show()
'''

replacements = {
    "05cdebfe-308e-4fb9-a19a-fa64a308fe85": func_source,
    "d67a3d14-9873-4714-88d5-e04a0361fea6": sweep_source,
}

for cell in nb["cells"]:
    cid = cell.get("id", "")
    if cid in replacements:
        cell["source"] = replacements[cid]
        cell["outputs"] = []
        cell["execution_count"] = None
        print(f"Replaced cell {cid[:8]}...")

with open("DemoThermalModel.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Done.")
