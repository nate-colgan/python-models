import json

with open('tinless system model.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

patches = []

# ── cell 9494082a: spectral_net_radiation — add Returns ──────────────────────
patches.append(('9494082a',
    "    liner_bool     : bool   True = tungsten liner (w_emis_spectral), False = graphite (gr_emis)\n"
    "    \"\"\"\n",
    "    liner_bool     : bool   True = tungsten liner (w_emis_spectral), False = graphite (gr_emis)\n"
    "\n"
    "    Returns\n"
    "    -------\n"
    "    float -- total radiative power absorbed by the powerstick segment [W],\n"
    "             summed across all wavelength bands\n"
    "    \"\"\"\n"
))

# ── cell 9494082a: graphite_to_powerstick_heat_transfer — complete docstring ──
patches.append(('9494082a',
    "    D_ps     : float  Powerstick outer diameter [m]\n"
    "    L        : float  Length [m]\n"
    "    \"\"\"\n",
    "    D_ps     : float  Powerstick outer diameter [m]\n"
    "    L        : float  Length [m]\n"
    "    liner_bool : bool   True = tungsten liner emissivity, False = bare graphite\n"
    "    tol      : float  Absolute temperature tolerance for brentq root-find [K]\n"
    "    max_iter : int    Maximum brentq iterations\n"
    "\n"
    "    Returns\n"
    "    -------\n"
    "    Q       : float -- net radiative heat transfer from graphite inner wall\n"
    "                       to powerstick [W]\n"
    "    T_inner : float -- graphite inner wall temperature [K] at thermal equilibrium\n"
    "    \"\"\"\n"
))

# ── cell 03dd9a23: TinlessSimulation.__init__ ─────────────────────────────────
patches.append(('03dd9a23',
    "    def __init__(self, storage, powerstick, T_sur=300, block_heat_loss=False):\n",
    "    def __init__(self, storage, powerstick, T_sur=300, block_heat_loss=False):\n"
    "        \"\"\"\n"
    "        Initialise the simulation and pre-compute geometry-dependent quantities.\n"
    "\n"
    "        The view factor VF = r_ps_eff / r_in is derived here once so it does not\n"
    "        need to be recomputed every timestep.  r_ps_eff = 2*width/pi converts the\n"
    "        square powerstick cross-section to an equivalent cylinder with the same\n"
    "        lateral surface area.\n"
    "\n"
    "        Parameters\n"
    "        ----------\n"
    "        storage         : StorageModule -- graphite block + pipe assembly\n"
    "        powerstick      : PowerStick    -- TPV panel assembly\n"
    "        T_sur           : float         -- surroundings temperature for outer-wall\n"
    "                                           radiation loss [K] (default 300)\n"
    "        block_heat_loss : bool          -- if True, outer wall radiates to T_sur;\n"
    "                                           if False, outer wall is adiabatic\n"
    "        \"\"\"\n"
))

# ── cell 03dd9a23: TinlessSimulation._step_calc ───────────────────────────────
patches.append(('03dd9a23',
    "    def _step_calc(self, T_dist):\n",
    "    def _step_calc(self, T_dist):\n"
    "        \"\"\"\n"
    "        Compute radiation and electricity for one timestep given a temperature field.\n"
    "\n"
    "        Loops over each axial segment, reads the inner-wall temperature, and calls\n"
    "        spectral_net_radiation to get the total radiated power and\n"
    "        generate_electricity_step to get the cell heat flux.  Q_waste is the\n"
    "        portion of absorbed radiation that does not become electricity.\n"
    "\n"
    "        Parameters\n"
    "        ----------\n"
    "        T_dist : ndarray, shape (n_ax, n_rad) -- current block temperature field [K]\n"
    "\n"
    "        Returns\n"
    "        -------\n"
    "        Q_rad   : ndarray (n_ax,) -- total spectral radiation from inner wall per\n"
    "                                     axial segment [W]\n"
    "        Q_cell  : ndarray (n_ax,) -- raw cell heat flux (third return of\n"
    "                                     generate_electricity_step) per segment [W]\n"
    "        Q_waste : ndarray (n_ax,) -- Q_rad minus electrical output per segment [W]\n"
    "        \"\"\"\n"
))

# ── cell 03dd9a23: TinlessSimulation.run_simulation ───────────────────────────
patches.append(('03dd9a23',
    "    def run_simulation(self, end_time, dt, sub_steps, snap_every=None, print_interval=0):\n",
    "    def run_simulation(self, end_time, dt, sub_steps, snap_every=None, print_interval=0):\n"
    "        \"\"\"\n"
    "        Run the full transient discharge simulation.\n"
    "\n"
    "        At each timestep: (1) compute Q_rad and Q_cell via _step_calc;\n"
    "        (2) call StorageModule.update_temps with -Q_rad as the heat source\n"
    "        (heat leaving the block); (3) record scalar diagnostics.  Optional\n"
    "        temperature snapshots of the full 2-D field are stored every snap_every\n"
    "        seconds.\n"
    "\n"
    "        Parameters\n"
    "        ----------\n"
    "        end_time       : float      -- total simulation duration [s]\n"
    "        dt             : float      -- timestep size [s]\n"
    "        sub_steps      : int        -- finite-difference sub-steps per dt for\n"
    "                                       numerical stability of the conduction solve\n"
    "        snap_every     : float|None -- interval between full-field temperature\n"
    "                                       snapshots [s]; None records end state only\n"
    "        print_interval : int        -- print a status line every N timesteps;\n"
    "                                       0 = silent\n"
    "\n"
    "        Returns\n"
    "        -------\n"
    "        dict with keys:\n"
    "          'time'           : ndarray (n_steps+1,) -- time [s]\n"
    "          'Q_rad'          : ndarray (n_steps+1,) -- total radiated power [W]\n"
    "          'Q_cell'         : ndarray (n_steps+1,) -- total cell heat flux [W]\n"
    "          'P_elec'         : ndarray (n_steps+1,) -- electrical output [W]\n"
    "                                                      (= Q_cell * panel_efficiency)\n"
    "          'T_avg'          : ndarray (n_steps+1,) -- volume-average block temp [K]\n"
    "          'T_inner'        : ndarray (n_steps+1,) -- axial-avg inner wall temp [K]\n"
    "          'T_outer'        : ndarray (n_steps+1,) -- axial-avg outer wall temp [K]\n"
    "          'snap_times'     : ndarray -- times of full-field snapshots [s]\n"
    "          'temp_snapshots' : ndarray -- (n_ax, n_rad) temperature fields [K]\n"
    "        \"\"\"\n"
))

# ── cell c7d9e412: simulate_pss_avg_power ─────────────────────────────────────
patches.append(('c7d9e412',
    "def simulate_pss_avg_power(area_ratio):\n",
    "def simulate_pss_avg_power(area_ratio):\n"
    "    \"\"\"\n"
    "    Build and run a single-case simulation, then return the average cell heat\n"
    "    flux over the pseudo-steady-state (PSS) window minus the design target.\n"
    "\n"
    "    Intended as the objective function for the bisection search in this cell.\n"
    "    A return value of zero means the geometry delivers exactly avg_power during\n"
    "    the pseudo-steady portion of the discharge.\n"
    "\n"
    "    PSS onset is the timestep where (T_inner - T_avg) is most negative — the\n"
    "    point of maximum radial gradient, after which conduction begins to catch up\n"
    "    with the discharge rate.  The window ends at SOC = 0 (T_avg <= T_min).\n"
    "    Q_cell (not Q_rad) is averaged because the bisection target is the\n"
    "    cell-area-weighted power, not total thermal emission.\n"
    "\n"
    "    Parameters\n"
    "    ----------\n"
    "    area_ratio : float -- r_in / r_ps_eff; controls the view factor\n"
    "                          VF = 1 / area_ratio.  Larger values move the inner\n"
    "                          wall further from the powerstick, reducing VF and\n"
    "                          increasing the radiation exchange.\n"
    "\n"
    "    Returns\n"
    "    -------\n"
    "    float -- avg_Q_cell_pss - avg_power [W]; positive means the geometry\n"
    "             delivers too much heat (area_ratio should decrease), negative\n"
    "             means too little (area_ratio should increase)\n"
    "    \"\"\"\n"
))

# ── apply all patches ─────────────────────────────────────────────────────────
for cell_prefix, old, new in patches:
    for cell in nb['cells']:
        if cell.get('id', '').startswith(cell_prefix):
            src = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
            if old not in src:
                print(f"WARNING: patch target not found in cell {cell_prefix}:")
                print(repr(old[:100]))
            else:
                src = src.replace(old, new, 1)
                cell['source'] = src
                cell['outputs'] = []
                cell['execution_count'] = None
                print(f"Patched cell {cell_prefix}")
            break

with open('tinless system model.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Saved.")
