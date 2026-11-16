import json, uuid

with open("DemoThermalModel.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

new_source = '''def graphite_to_powerstick_heat_transfer(T_outer, T_ps, k_gr, emis_gr, emis_ps,
                                          D_in, D_out, D_ps, L,
                                          tol=1e-2, max_iter=50):
    """
    Heat transfer from an annular graphite region to a powerstick via series
    conduction (outer wall -> inner wall) and radiation (inner wall -> powerstick).

    Parameters
    ----------
    T_outer  : float  Temperature at outer wall of graphite annulus [K]
    T_ps     : float  Powerstick temperature [K]
    k_gr     : float  Graphite thermal conductivity [W/mK]
    emis_gr  : float  Graphite inner-surface emissivity [-]
    emis_ps  : float  Powerstick emissivity [-]
    D_in     : float  Inner diameter of graphite annulus [m]
    D_out    : float  Outer diameter of graphite annulus [m]
    D_ps     : float  Powerstick outer diameter [m]
    L        : float  Length [m]
    tol      : float  Convergence tolerance on T_inner [K]
    max_iter : int    Maximum iterations

    Returns
    -------
    Q        : float  Heat transfer rate [W]
    T_inner  : float  Converged graphite inner wall temperature [K]
    """
    A_gr_inner = np.pi * D_in * L       # graphite inner surface area [m^2]
    A_ps       = np.pi * D_ps * L       # powerstick surface area [m^2]
    VF_gr_ps   = D_ps / D_in           # view factor: graphite inner wall -> powerstick (by reciprocity)
    R_cond     = RprimeCondCyl(D_in, D_out, k_gr) / L  # cylindrical conduction resistance [K/W]

    T_inner = (T_outer + T_ps) / 2     # initial guess

    for _ in range(max_iter):
        R_rad = Rrad_2surfaces(T_inner, T_ps, A_gr_inner, A_ps, emis_gr, emis_ps, VF_gr_ps)
        Q     = (T_outer - T_ps) / (R_cond + R_rad)
        T_inner_new = T_outer - Q * R_cond
        if abs(T_inner_new - T_inner) < tol:
            return Q, T_inner_new
        T_inner = T_inner_new

    return Q, T_inner'''

new_cell = {
    "cell_type": "code",
    "id": str(uuid.uuid4()),
    "metadata": {},
    "outputs": [],
    "source": new_source,
    "execution_count": None,
}

# Insert after cell with id acc13fc7-64a9-47b6-9357-d9c389eb81e9
target_id = "acc13fc7-64a9-47b6-9357-d9c389eb81e9"
insert_idx = next((i for i, c in enumerate(nb["cells"]) if c.get("id") == target_id), len(nb["cells"]) - 1)
nb["cells"].insert(insert_idx + 1, new_cell)

with open("DemoThermalModel.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Inserted new cell at index {insert_idx + 1}, id={new_cell['id']}")
