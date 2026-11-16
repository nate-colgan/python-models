import json

with open('tinless system model.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

# ── 1. Fix YAML default: shield_side_len 0.18 → 0.25 ─────────────────────────
for cell in nb['cells']:
    if not cell.get('id', '').startswith('a8c3d501'):
        continue
    src = cell['source'] if isinstance(cell['source'], str) else ''.join(cell['source'])
    old = '  shield_side_len: 0.18    # square side length of W shield [m]; r_shield = 2*side_len/pi'
    new = '  shield_side_len: 0.25    # square side length of W shield [m]; r_shield = 2*side_len/pi  (must give r_shield > r_liner)'
    if old in src:
        cell['source'] = src.replace(old, new, 1)
        cell['outputs'] = []
        cell['execution_count'] = None
        print('Fixed YAML shield_side_len default in a8c3d501')
    else:
        print('WARNING: YAML pattern not found')
    break

# ── 2. Add scratch cell at end of notebook ────────────────────────────────────
scratch_src = """\
# ── Scratch: shield sensitivity ── incident cell flux vs f_shield ─────────────
# All inputs are defined here. Change them to explore the physics.
# Only dependency: spectral_net_radiation must be defined in the kernel (cell 9494082a).
import numpy as np
import matplotlib.pyplot as plt

# ── Inputs ────────────────────────────────────────────────────────────────────
side_len        = 0.13    # powerstick square side [m]
liner_side_len  = 0.20    # liner square side [m]
shield_side_len = 0.25    # shield square side [m]  ← tune this  (r_shield must be > r_liner)
r_in            = 0.20    # graphite inner-wall effective radius [m]  (= storage.block.rad_in)
L               = 0.30    # axial length [m]
f_bare          = 0.20    # bare (non-cell) fraction of powerstick surface
f_active        = 0.83    # active PV fraction of non-bare surface

band_gap = 1.55           # PV bandgap wavelength [μm]
emissivity_table = [
    ((0.3,      band_gap), 0.85, 0.10, 0.10),  # below bandgap: cells absorb strongly
    ((band_gap, 8.0     ), 0.15, 0.10, 0.15),  # above bandgap: low absorption
]

T_hot_vals   = [1800, 1900, 2000, 2100, 2173]  # graphite inner-wall temps to sweep [K]
f_shield_arr = np.linspace(0, 0.9, 60)          # shield area fractions to evaluate

# ── Derived geometry ──────────────────────────────────────────────────────────
r_ps_eff = 2 * side_len / np.pi
r_liner  = 2 * liner_side_len / np.pi
r_shield = 2 * shield_side_len / np.pi

assert r_liner < r_shield < r_in, (
    f"Shield must sit between liner (r={r_liner*100:.1f} cm) and graphite inner wall "
    f"(r={r_in*100:.1f} cm), but r_shield = {r_shield*100:.1f} cm"
)

VF_gr_liner   = r_liner  / r_in
VF_liner_ps   = r_ps_eff / r_liner
VF_gr_shield  = r_shield / r_in
VF_shield_lin = r_liner  / r_shield

A_ps         = 4 * side_len * L
A_liner_seg  = 2 * np.pi * r_liner  * L
A_shield_seg = 2 * np.pi * r_shield * L
A_cell_cm2   = 4 * side_len * L * (1 - f_bare) * f_active * 1e4

print(f"r_ps_eff  = {r_ps_eff*100:.2f} cm")
print(f"r_liner   = {r_liner*100:.2f} cm")
print(f"r_shield  = {r_shield*100:.2f} cm")
print(f"r_in      = {r_in*100:.2f} cm")
print(f"A_cell    = {A_cell_cm2:.1f} cm²")

# ── Sweep ─────────────────────────────────────────────────────────────────────
sweep = {}
for T_hot in T_hot_vals:
    fluxes = []
    for fs in f_shield_arr:
        Q_ps, _ = spectral_net_radiation(
            T_hot, emissivity_table,
            VF_gr_liner, A_ps, f_bare, f_active,
            liner_bool    = True,
            VF_gr_liner   = VF_gr_liner,
            VF_liner_ps   = VF_liner_ps,
            A_liner_seg   = A_liner_seg,
            shield_bool   = True,
            f_shield      = fs,
            VF_gr_shield  = VF_gr_shield,
            VF_shield_lin = VF_shield_lin,
            A_shield_seg  = A_shield_seg,
        )
        fluxes.append(Q_ps / A_cell_cm2)
    sweep[T_hot] = np.array(fluxes)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
for T_hot, fluxes in sweep.items():
    ax.plot(f_shield_arr, fluxes, label=f'T_inner = {T_hot} K')
ax.set_xlabel('Shield area fraction  f_shield')
ax.set_ylabel('Incident cell flux  [W/cm\\u00b2]')
ax.set_title('Incident cell flux vs shield fraction')
ax.legend()
ax.grid(True)
fig.tight_layout()
plt.show()

# ── Summary table ─────────────────────────────────────────────────────────────
ref_fracs = [0.0, 0.2, 0.4, 0.6, 0.8]
header = '  '.join(f'f={f:.1f}' for f in ref_fracs)
print(f"\\n{'T_hot [K]':>10}  {header}  [W/cm\\u00b2]")
for T_hot, fluxes in sweep.items():
    vals = [fluxes[np.argmin(np.abs(f_shield_arr - fs))] for fs in ref_fracs]
    row  = '  '.join(f'{v:>7.2f}' for v in vals)
    print(f"{T_hot:>10}  {row}")
"""

scratch_cell = {
    "id": "scratch_shield_01",
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": scratch_src,
}

nb['cells'].append(scratch_cell)
print('Added scratch cell at end of notebook')

with open('tinless system model.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Saved.')
