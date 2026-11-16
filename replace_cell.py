import json

with open("DemoThermalModel.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

new_source = '''\
# ── Fixed inputs ──────────────────────────────────────────────────────────────
V_graphite =       # total graphite annulus volume [m^3] — fill in
L          =       # length [m] — fill in
D_ps       =       # powerstick outer diameter [m] — fill in
T_ps       =       # powerstick temperature [K] — fill in
emis_ps    =       # powerstick emissivity [-] — fill in

# ── Sweep parameters ──────────────────────────────────────────────────────────
k_gr_vals    = [15, 30]           # graphite thermal conductivity [W/mK]
T_outer_vals = [2273, 2473, 2673] # outer graphite wall temperature [K]
D_in_vals    = [0.75, 1, 1.25]    # inner graphite diameter [m] — adjust units to match above

# ── Sweep ─────────────────────────────────────────────────────────────────────
results = {}
for k_gr in k_gr_vals:
    for D_in in D_in_vals:
        D_out = np.sqrt(4 * V_graphite / (np.pi * L) + D_in**2)
        emis_gr = gr_emis(np.mean(T_outer_vals))
        Q_vals, T_inner_vals = [], []
        for T_outer in T_outer_vals:
            Q, T_inner = graphite_to_powerstick_heat_transfer(
                T_outer, T_ps, k_gr, emis_gr, emis_ps, D_in, D_out, D_ps, L
            )
            Q_vals.append(Q)
            T_inner_vals.append(T_inner)
        results[(k_gr, D_in)] = (Q_vals, T_inner_vals)

# ── Plot ──────────────────────────────────────────────────────────────────────
T_outer_C  = np.array(T_outer_vals) - 273.15
colors     = plt.cm.plasma(np.linspace(0.15, 0.85, len(D_in_vals)))
ls_map     = {k: style for k, style in zip(k_gr_vals, ['-', '--'])}

fig_Q,  ax_Q = plt.subplots(figsize=(8, 5))
fig_T,  ax_T = plt.subplots(figsize=(8, 5))

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
    (ax_Q, "Q [kW]",        "Power output"),
    (ax_T, "T_inner [°C]",  "Graphite inner wall temperature"),
]:
    ax.set_xlabel("T_outer [°C]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    # Build a tidy legend: color = D_in, linestyle = k_gr
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=8, ncol=2)

fig_Q.tight_layout()
fig_T.tight_layout()
plt.show()
'''

target_id = "d67a3d14-9873-4714-88d5-e04a0361fea6"
for cell in nb["cells"]:
    if cell.get("id") == target_id:
        cell["source"] = new_source
        cell["outputs"] = []
        cell["execution_count"] = None
        print("Cell replaced.")
        break

with open("DemoThermalModel.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
