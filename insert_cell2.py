import json, uuid

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
        # Size D_out so that the annulus cross-section gives the target volume
        D_out = np.sqrt(4 * V_graphite / (np.pi * L) + D_in**2)
        emis_gr = gr_emis(np.mean(T_outer_vals))  # evaluated at a representative temp
        Q_vals, T_inner_vals = [], []
        for T_outer in T_outer_vals:
            Q, T_inner = graphite_to_powerstick_heat_transfer(
                T_outer, T_ps, k_gr, emis_gr, emis_ps, D_in, D_out, D_ps, L
            )
            Q_vals.append(Q)
            T_inner_vals.append(T_inner)
        results[(k_gr, D_in)] = (Q_vals, T_inner_vals)

# ── Plot ──────────────────────────────────────────────────────────────────────
T_outer_C = np.array(T_outer_vals) - 273.15
colors     = plt.cm.plasma(np.linspace(0.15, 0.85, len(D_in_vals)))
linestyles = ['-', '--']

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
fig.suptitle("Graphite-to-powerstick heat transfer parametric sweep")

for row, k_gr in enumerate(k_gr_vals):
    ax_Q = axes[row, 0]
    ax_T = axes[row, 1]
    for ci, D_in in enumerate(D_in_vals):
        D_out = np.sqrt(4 * V_graphite / (np.pi * L) + D_in**2)
        Q_vals, T_inner_vals = results[(k_gr, D_in)]
        lbl = f"D_in={D_in}, D_out={D_out:.3f}"
        ax_Q.plot(T_outer_C, np.array(Q_vals) / 1e3, marker='o', color=colors[ci],
                  linestyle=linestyles[row], label=lbl)
        ax_T.plot(T_outer_C, np.array(T_inner_vals) - 273.15, marker='o', color=colors[ci],
                  linestyle=linestyles[row], label=lbl)

    ax_Q.set_ylabel("Q [kW]")
    ax_Q.set_title(f"Power output  (k_gr = {k_gr} W/mK)")
    ax_Q.legend(fontsize=8)
    ax_Q.grid(True)

    ax_T.set_ylabel("T_inner [°C]")
    ax_T.set_title(f"Inner wall temperature  (k_gr = {k_gr} W/mK)")
    ax_T.legend(fontsize=8)
    ax_T.grid(True)

for ax in axes[-1]:
    ax.set_xlabel("T_outer [°C]")

fig.tight_layout()
plt.show()
'''

# Find the cell we just inserted (the graphite_to_powerstick function)
target_id = "05cdebfe-308e-4fb9-a19a-fa64a308fe85"
insert_idx = next((i for i, c in enumerate(nb["cells"]) if c.get("id") == target_id), len(nb["cells"]) - 1)

new_cell = {
    "cell_type": "code",
    "id": str(uuid.uuid4()),
    "metadata": {},
    "outputs": [],
    "source": new_source,
    "execution_count": None,
}

nb["cells"].insert(insert_idx + 1, new_cell)

with open("DemoThermalModel.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Inserted sweep cell at index {insert_idx + 1}")
