import json

NB = 'tinless system model.ipynb'
with open(NB, encoding='utf-8') as f:
    nb = json.load(f)

NEW_SRC = '''\
# ── Scratch: shield sensitivity ── incident cell flux vs f_shield ─────────────
# Self-contained: adjust any input below.
# Depends on: TinlessSimulation class defined in cell 03dd9a23 (run that cell first).
import numpy as np
import matplotlib.pyplot as plt

# ── Inputs ────────────────────────────────────────────────────────────────────
side_len        = 0.13    # powerstick square side [m]
liner_side_len  = 0.30    # liner square side [m]
shield_side_len = 0.32    # shield square side [m]  (r_shield must lie between r_liner and r_in)
r_in            = 0.50    # graphite inner-wall effective radius [m]
L               = 3.0     # axial length [m]
f_bare          = 0.0
f_active        = 1.0

band_gap = 1.24           # PV bandgap [μm]
emissivity_table = [
    ((0.1,      band_gap), 0.9,  0.10, 0.10),
    ((band_gap, 20.0    ), 0.05, 0.10, 0.10),
]

T_hot_vals     = [1873, 2073, 2273, 2473, 2673]  # graphite inner-wall temperatures [K]
N_shields_vals = [0, 1, 2, 3]                    # number of shield layers
f_shield_arr   = np.linspace(0, 1, 60)            # shield area fractions to sweep

# ── Derived geometry ──────────────────────────────────────────────────────────
r_ps_eff = 2 * side_len / np.pi
r_liner  = 2 * liner_side_len / np.pi
r_shield = 2 * shield_side_len / np.pi

assert r_liner < r_shield < r_in, (
    f"Need r_liner ({r_liner*100:.1f} cm) < r_shield ({r_shield*100:.1f} cm)"
    f" < r_in ({r_in*100:.1f} cm)"
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
print(f"A_cell    = {A_cell_cm2:.1f} cm\\u00b2")

# ── Mock sim object for TinlessSimulation.spectral_net_radiation ──────────────
# spectral_net_radiation is an instance method; calling it as
# TinlessSimulation.spectral_net_radiation(mock, T_hot, f_shield=fs)
# is idiomatic Python for unbound method calls.
class _M:
    pass

def _make_sim(N):
    m = _M()
    m.powerstick              = _M()
    m.powerstick.emissivity_table = emissivity_table
    m.powerstick.liner_bool   = True
    m.powerstick.f_bare       = f_bare
    m.powerstick.f_active     = f_active
    m.VF                      = VF_gr_liner
    m.A_ps_seg                = A_ps
    m.N_shields               = max(N, 1)   # formula needs >= 1 when shield is active
    m.shield_bool             = (N > 0)
    m.VF_gr_liner             = VF_gr_liner
    m.VF_liner_ps             = VF_liner_ps
    m.A_liner_seg             = A_liner_seg
    m.VF_gr_shield            = VF_gr_shield  if N > 0 else None
    m.VF_shield_lin           = VF_shield_lin if N > 0 else None
    m.A_shield_seg            = A_shield_seg  if N > 0 else None
    return m

# ── Sweep ─────────────────────────────────────────────────────────────────────
flux = {}   # (N, T_hot) -> flux array over f_shield_arr
for N in N_shields_vals:
    sim = _make_sim(N)
    for T_hot in T_hot_vals:
        vals = []
        for fs in f_shield_arr:
            Q_ps, _ = TinlessSimulation.spectral_net_radiation(sim, T_hot, f_shield=fs)
            vals.append(Q_ps / A_cell_cm2)
        flux[(N, T_hot)] = np.array(vals)

# ── Plot: one subplot per N_shields ──────────────────────────────────────────
colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(T_hot_vals)))
fig, axes = plt.subplots(1, len(N_shields_vals), figsize=(16, 5), sharey=True,
                         constrained_layout=True)

for ax, N in zip(axes, N_shields_vals):
    for color, T_hot in zip(colors, T_hot_vals):
        ax.plot(f_shield_arr, flux[(N, T_hot)], color=color, label=f'{T_hot} K')
    title = 'No shield' if N == 0 else f'{N} shield layer{"s" if N > 1 else ""}'
    ax.set_title(title)
    ax.set_xlabel('Shield area fraction  $f_\\\\mathrm{shield}$')
    ax.grid(True, alpha=0.4)

axes[0].set_ylabel('Incident cell flux  [W/cm\\u00b2]')
axes[0].legend(title='$T_\\\\mathrm{inner}$', fontsize=8)
fig.suptitle('Incident cell flux vs shield fraction  |  W liner, W shield(s)', fontsize=11)
plt.show()

# ── Effect of N_shields at fixed f_shield ────────────────────────────────────
# Cross-section: flux vs N_shields at a few representative f_shield values
f_refs = [0.25, 0.50, 0.75]
fig2, ax2 = plt.subplots(figsize=(8, 4))
line_colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(f_refs)))
for color, fs_ref in zip(line_colors, f_refs):
    i_ref = np.argmin(np.abs(f_shield_arr - fs_ref))
    for T_hot in T_hot_vals:
        y = [flux[(N, T_hot)][i_ref] for N in N_shields_vals]
        ax2.plot(N_shields_vals, y, color=color, marker='o',
                 label=f'$f_{{shield}}={fs_ref}$' if T_hot == T_hot_vals[0] else None,
                 alpha=0.4 + 0.15 * T_hot_vals.index(T_hot))

ax2.set_xlabel('Number of shield layers')
ax2.set_ylabel('Incident cell flux  [W/cm\\u00b2]')
ax2.set_title('Effect of shield count at fixed f_shield  (lighter = higher T)')
ax2.set_xticks(N_shields_vals)
ax2.legend(title='f_shield')
ax2.grid(True, alpha=0.4)
fig2.tight_layout()
plt.show()

# ── Summary table ─────────────────────────────────────────────────────────────
fs_ref = 0.5
i_ref  = np.argmin(np.abs(f_shield_arr - fs_ref))
print(f"\\nCell flux [W/cm\\u00b2] at f_shield = {fs_ref}")
header = '  '.join(f'N={N}' for N in N_shields_vals)
print(f"{'T_hot':>10}  {header}")
for T_hot in T_hot_vals:
    vals = '  '.join(f'{flux[(N,T_hot)][i_ref]:>5.2f}' for N in N_shields_vals)
    print(f"{T_hot:>10}  {vals}")
'''

for cell in nb['cells']:
    if cell.get('id','') == 'scratch_shield_01':
        cell['source'] = NEW_SRC
        cell['outputs'] = []
        cell['execution_count'] = None
        print(f'Replaced scratch_shield_01 ({len(NEW_SRC.splitlines())} lines)')
        break

with open(NB, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Saved.')
