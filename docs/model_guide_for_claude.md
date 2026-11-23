# Tinless thermal battery model: working guide for Claude

This file is context for Claude (desktop app / claude.ai project). Read it together with
`docs/tinless_model_documentation.docx`. The documentation covers the physics, equations,
inputs and numerical settings in detail. This guide covers the rest:

- how the code and notebooks are used day to day;
- why the model is set up the way it is: decisions and their reasons;
- what changed after the documentation was generated;
- results so far;
- what is deliberately not modelled;
- pitfalls.

The user is Nate, who owns the model. Treat his stated decisions below as settled unless he reopens them.
Last updated 2026-11-23 (commit after `ca392810`).

---

## 1. What the system is (one paragraph)

The plant is a 25 MW AC, 50 h thermal battery:

- **Storage:** a large graphite block in an insulated steel tank, under argon.
- **Charging:** resistive heaters in round cavities heat the block, up to about 2400 °C.
- **Discharging:** thermophotovoltaic (TPV) "powersticks" are inserted into square cavities. Each stick is surrounded by a graphite liner and 8 rotating graphite-felt shade vanes. The vanes throttle the radiation so the cells see a constant target flux.
- **Layout:** stick and heater cavities sit on a checkerboard in one continuous graphite matrix. The stick cavities are rotated 45° to the grid.
- **Weekly cycle:** a 50 h charge, about 68 h hold and a 50 h discharge (168 h).
- **Stick position:** the stick is inserted only during discharge. During charge and hold it is retracted and a door closes the hole.

## 2. Repository map (what matters)

| Path | Role |
|---|---|
| `tinless_model_input.yaml` | All physics inputs. **Generated** by cell `a8c3d501` of `tinless system model.ipynb`; never edit the file directly (see §6). |
| `tinless system model.ipynb` | Single-design work. `a8c3d501` writes the YAML, `b2e4f810` builds and runs one discharge, `shade_open_resize`, `full_cycle` (charge→hold→discharge ledger), `heat_sim_01` / `heat_bisect_01` (heater), `tank_sys01` (plant), plus `scratch_*` study cells. |
| `tinless system optimizer.ipynb` | System optimizer driver. Cells run in order: `opt_setup`, `opt_inputs` (OPT_INPUTS overrides), `opt_baseline`, `opt_run` (differential evolution), `opt_results`, `opt_scatter`, `opt_flux`, `opt_verify` (top designs at detailed fidelity), `opt_cycle` (full cycle of the optimum at detailed fidelity, 10–20 min), `opt_report` (Word run report), `opt_docs` (regenerates the model documentation). |
| `thermal/build.py` | `load_config`, `build_tinless(cfg, overrides)` → model namespace `m` (stick cell); `build_heater(m, area_ratio)` (heater cell). One entry point for notebook, sweeps and optimizer. |
| `thermal/tinless.py` | `TinlessSimulation`: `run_simulation` (discharge loop) and `run_heating_simulation` (charge loop). |
| `thermal/axial.py` | Axial radiation model in use: ring-resolved cavity radiation with vane shades and the liner / liner bottom; supplies `dQ/dT_wall` for the implicit wall coupling. |
| `thermal/viewfactors.py` | Cavity view factors (Sobol Monte Carlo), vane ray trace, axial mesh. Disk cache in `radiation.vf_cache_dir`, which the optimizer turns off. |
| `thermal/components.py` | `StorageModule` / `GraphiteBlock` (radial–axial conduction), heater, etc. Block temperature: `storage.block.temp_dist[i_ax, i_rad]`, where `[:, 0]` is the inner-wall node. |
| `thermal/properties.py` | Material properties, `panel_efficiency` (TPV fit), `LINER_GRAY` (grafoil). |
| `thermal/deposition.py` | Carbon / tungsten vapour deposition on the cells and bare areas. |
| `thermal/sizing.py` | `run_discharge`, `shade_open_point`, `shade_open_resize` (block sizing). |
| `thermal/tank.py` | Plant around the per-stick results: hot-zone layout, tank, insulation, columns, floor cooling, foundation, losses, capex. Closed-form, milliseconds. **Temperatures in °C here.** |
| `thermal/cycle.py` | `run_cycle`: one full cycle with a closed energy ledger; `plot_layout`, `plot_timeseries`, `plot_losses`. |
| `thermal/optimize.py` | `OPT_DEFAULTS`, `DESIGN_VARS`, `decode`/`encode`, `evaluate_design`, `run_de`, `summary`, `pareto`, `DETAILED` fidelity. |
| `thermal/report.py` | Word run report (`write_report`) and `FigureStore`. |
| `thermal/docs*.py` | Generator for the model documentation (.docx). |
| `tests/` | pytest suite, 40 tests. `test_regression.py` compares against `baseline_local_radiation.npz`, regenerated with `tests/make_baseline.py` when physics changes on purpose. |
| `opt_runs/<timestamp>_p<price>/` | One folder per optimizer run: `evals_<pid>.jsonl` (one JSON record per design), `inputs.json`, and after the report cell `report.docx`, `figures/`, `report_inputs.json`. Not committed. |
| `patch_*.py`, `insert_*.py`, older notebooks | Historical one-off scripts and earlier studies; not part of the current model. |

Units: the transient model works in K. `tank.py` and optimizer design fields ending in `_C`
are °C. Energies in the optimizer records are MWh per cycle (`*_MWh`) or kWh per stick.

## 3. How a design is evaluated (the optimizer pipeline in practice)

`evaluate_design(d, base_cfg, opt)` (about 70–90 s per design at screening fidelity, one core):

1. **Design to config.** `design_cfg` copies the YAML and overwrites the design variables and the screening fidelity: 0.5 m rings, 5 radial nodes, timestep ×2, 2^12 vane rays, no view-factor cache.
2. **Shade-open sizing** (`sizing.shade_open_resize`). T_max is a design variable.
   - T_min is the volume-weighted block average at which the shades reach fully open.
   - The block volume is scaled so that this happens at 50 h.
   - The sizing iterates (about 3–4 discharges) because T_min and the volume change the radial profile.
   - An infeasible flux raises `FluxUnreachable`.
3. **Heater search.** Find the smallest heater power that charges in `charge_hr` (50 h) with the cavity at its minimum gap (0.25 m). The cavity grows only if the power cap is not enough. The cap is `heater.max_power_per_m` × block height, 150 kW/m in the YAML.
4. **Graphite wall check.** The minimum web between neighbouring cavities must be at least 0.30 m.
5. **Tank.** `tank_system` sized on the clean stick's discharge-average AC power: N_ps = ceil(P_rated / P_stick_AC).
6. **Lifetime-average wear.** A second discharge at twice the lifetime-mean degradation (`lifetime_mean_D`). Performance is the average of the clean and worn runs, which is the exact average over every cycle of the life for performance that is linear in wear.
7. **Energy per cycle, RTE and capex.** Objective = lifetime capex + electricity price × energy lost over the life.

`OPT_DEFAULTS` holds everything that is not physics (economics, costs, limits, fidelity). The
notebook's `OPT_INPUTS` overrides it. Physics inputs come from the YAML.

Design variables (10):

| Variable | Range |
|---|---|
| liner side | 0.2–0.6 m |
| powerstick side | as a fraction of the room in the liner |
| stick length L | 3–10 m |
| block height | H = L + 0.25 + frac × (14 − L − 0.25) m, so the block is always at least 0.25 m taller than the stick |
| flux target | 5–20 W/cm² |
| shade thickness | 2/8–48/8 in (integer eighths) |
| side insulation | 0.5–3 m |
| top insulation | 0.5–3 m |
| bottom stack L_bot | 2–4 m |
| T_max | 1800–2500 °C |

Constraints:

- T_min ≥ 1400 °C;
- swing 200–1000 °C;
- tank D ≤ 80 m and H ≤ 30 m;
- floor maximum and column hot spot ≤ `tank.floor.T_limit_design` (120 °C);
- foundation pressure ≤ soil allowable;
- energy target met (within the 1% sizing tolerance);
- the cycle fits in 168 h;
- graphite wall ≥ 0.30 m.

Differential evolution settings: scipy, Sobol initialisation, deferred updating, 20 workers,
popsize 10 (100 designs per generation), 25 generations. A full run is about 3300 designs and about 120 CPU-hours.

## 4. Decisions and their reasons (settled unless Nate reopens them)

**Objective and accounting**
- Minimise lifetime capex + `elec_price` × (AC in − net AC out) × cycles × life.
  - One price is used for charge and discharge. Its only job is to put a dollar value on efficiency.
  - Runs have used $75/MWh and $150/MWh; the run folder suffix `_p75` / `_p150` records which.
  - Capex is reported separately, without the electricity term.
  - Don't reintroduce separate buy/sell prices or revenue.
- Net RTE = (AC out − floor-cooling fans − heat-rejection fans and pumps) / AC in. Hold losses are included.
  - "Before parasitic loads" (`RTE_gross`) excludes the fans.
  - The floor fans run all cycle. Heat rejection uses 0.03 kW_e per kW_th.
- P_stick is the discharge-average power, not the setpoint power.
- Inverter efficiency is 98% each way. Life is 30 years with 52.1 cycles/yr. Soft costs are +30% on all capex, including replacements.
- Wear: D = deposit / 10 nm, linear.
  - Cells are replaced at D = 0.1. Bare areas are cleaned at D = 0.01.
  - The emissivity of the inactive and bare areas rises linearly to 0.8 at D = 1.
  - No deposition while the stick is retracted.
- Performance = the average over every cycle of the life (see §3 step 6). This holds both for designs whose cells are never replaced and for those with a partial last interval.

**Sizing**
- T_max is the variable; T_min and the swing are sizing outputs (shade-open sizing).
- The discharge cutoff and T_avg are volume-weighted block averages.
- The charge buys back the heat the discharge released plus the losses; the block is not reset to a uniform T_min.
- Heater power is sized to be as small as possible, because peak charging power sets the grid interconnect (charging power > output power). The interconnect is costed on max(peak charging AC, peak output AC).

**Geometry and mesh**
- `storage.num_ax_segments` = the total number of block rings over the block height (not stick steps).
- The accuracy target is 2%: speed is preferred over sub-2% accuracy. Energy-conservation checks (0.5%) are bug checks and stay tight.
- Shades are rotating vanes. Only flux feasibility matters, not the vane angle. The closed and open limits are ray-traced once per geometry, and the setpoint is held between them.
- With shades on, the stick cavity is the minimum that fits the vanes around the liner; YAML `powerstick.area_ratio` is ignored.
- Shade transmission uses straight-through (porous sheet) transmission, not diffuse.
- Heater cavity area always comes from `heater.side_len` × `heater.area_ratio`.

**Tank / plant**
- Column creep basis: sigma_ref = 2.81 MPa at 2400 °C (1% strain in 30 yr). Column pitch is computed. Columns are rectangular, 230 mm thick.
- Heat spreading: f_3D = 0.85 for the total column heat (confirmed). The floor hot spot uses f_foot = 1.21 × the 1-D column heat over a 0.36 m² bearing plate.
  - Scaling the spot by f_3D under-predicted it by about 40 K against the v3 FEA.
- The hot spot is checked under the column at the **air-outlet end**. The air warms along a single-pass path across a rectangular floor; optimizing flow paths for the round floor is out of scope for now.
- Floor limit 120 °C (`T_limit_design`). Nate lowered it from the Annex M 200 °C to keep headroom for a cooling-fan failure.
  - Floor air velocity is 10 m/s, raised from 6 m/s to meet the limit.
- Foundation pressure includes:
  - graphite;
  - hung equipment;
  - the concrete mat and the high-temperature slab;
  - cellular glass;
  - the floor plate and its fins.
- Roof and floor insulation losses act on every cell. They are a heat sink on the top and bottom block rings inside the transient runs (`tank.block_axial_losses: true`). Side and edge losses reach only perimeter cells, so they are accounted after the run.
- Storage graphite density 1450 kg/m³, graphite pass-through cost $1/kg. Argon cost is ignored.

**Deposition**
- The ×3.5 `conv_factor` in `deposition.py` is Nate's own correction from his deposition CFD runs (convection-enhanced transport). It is shape-dependent and probably conservative. Leave it alone.

**Liner**
- Current YAML: `liner_material: graphite`, 5 mm thick.
- Grafoil is available as an option: emissivity 0.5 on both faces, k = 4 W/mK, 0.5 mm. It is set aside for now.
- Liner cost is costed as grafoil walls at $55/m² plus a CFC floor at $1000/m² × 2 × side² in either case.

**Placeholder costs**

| Item | Cost |
|---|---|
| Heater | $5000 each |
| Charge electrics | $75/kW |
| Inverter | $100/kW (peak AC) |
| Dry cooler | $125/kW_th |
| Grid interconnect | $150/kW |
| Shade RGF | $15k/m³ |
| Shade axle | $300/m |
| Shade motor | $1500 |
| Shade linkage | $2000 per stick |
| Cells | $1/cm² of cell area |
| Stick | $100/m + $2000 each |
| Structural steel | $3 / 5 / 8 per kg (low / central / high) |

## 5. Changes made after the documentation .docx was generated (commit `ca392810`)

The documentation predates these changes. Several of its "Items to review" are now resolved.

- **Wall surface vs cell-centre temperature (fixed).**
  - Radiation is now solved at the cavity **surface** temperature, coupled to the first conduction node through a half-cell resistance (`StorageModule.wall_half_resistance`).
  - Discharge: a linearised surface solve with an effective Jacobian `D_eff = D (I + R_w D)^-1` in the implicit wall update.
  - Charge: a per-ring Newton solve.
  - The surface runs about 21 K below the node centre in discharge. Stick energy changed +0.8%; the charge took 51.6 h instead of 49.8 h at the same power.
  - Discharge results carry a new `T_wall` key.
- **Floor:** f_foot hot spot at the outlet column, `floor_limit()` (the design limit if set, else Annex M), v_air 10 m/s, T_limit_design 120 °C.
- **Foundation pressure** now includes the masses listed in §4.
- **Bearing plates** are costed as structural steel by mass (N_col × 0.36 m² × 50 mm × 7850 kg/m³ × $/kg).
- **L_bot** (bottom stack height) became an optimizer variable (2–4 m).
- **H_max** raised to 14 m. The block is always at least `min_lower_zone` = 0.25 m taller than the stick: a floor under the stick, the liner bottom, the felt pad and lower-zone radiation always exist, so there is no special case when L = H.
- **Run report** (`thermal/report.py`):
  - the model-settings table shows **as-run** values with a source column (YAML / optimizer input / design variable / sizing output);
  - a new section, "Round-trip efficiency vs cell efficiency";
  - wording "before parasitic loads".
- **Cell-efficiency statistics.** Optimizer records now include `eta_cell` (DC / heat absorbed by the active cells, life average), `RTE_frac_cell`, `E_cell_abs_MWh` and `E_dc_MWh`. The cycle result has the same quantities.
  - The ratio splits exactly: RTE = delivery (absorbed / AC bought) × cell efficiency × output (net AC / DC).
  - Runs logged before 2026-11-23 lack these fields.

## 6. Workflow rules and pitfalls

- **YAML:** edit only the `yaml_content = """..."""` string in cell `a8c3d501` of `tinless system model.ipynb`, then run that cell. It writes the file with UTF-8 encoding, and the file has non-ASCII comments. Editing the .yaml directly gets overwritten.
- **Scratch and sweep cells** must load their inputs from the YAML themselves, not from globals defined in other cells.
- **When rewriting a notebook cell,** keep parameter values Nate has changed. He sets them on purpose to match the physical system.
- **Optimizer notebook, `opt_run` cell:** `LOG_DIR` is a fresh timestamped folder every time the cell runs.
  - To work with a finished run, set `RELOAD_RUN = 'opt_runs/<folder>'`, or the results cell loads an empty folder and `best` is empty (IndexError further down).
  - The setup cell recreates `FIGS`, so re-run the plot cells before the report cell.
- **Module reloads:** the setup cell reloads `thermal.*`; `opt_cycle` reloads `thermal.cycle`; `opt_report` reloads `thermal.report`. Changes elsewhere need the setup cell or a kernel restart.
- **`report.docx` open in Word** locks the file. `write_report` then saves `report_<timestamp>.docx` instead.
- **`panel_efficiency`** is always called with a flux argument by the model. An override placed after the `if flux is not None` branch never takes effect.
  - On 2026-11-23 a flat-55% override (the engineering lead's cell-efficiency target) was used for a what-if rerun of the optimum. It is commented out now.
  - Make sure no override is active before an optimizer run. The regression test fails if one is.
- **`print_outputs_bool`** only controls printing; it doesn't affect optimizer runtime.
- **Screening vs detailed fidelity:** the optimizer ranks designs at screening fidelity. The verify and cycle cells re-run the top designs at `O.DETAILED` (1/3 m rings, 2^14 rays, full timestep), and the report tabulates the difference.
  - In a timestep study on one design, screening stick energy was within about 1% of detailed (see §9 for the known step-size errors).
- **Life-average vs clean-cycle RTE:** the report's full-cycle RTE (clean cells) is higher than the optimizer's life-average RTE, which includes wear.

## 7. Results so far

| Run | Price | Designs (feasible) | Best: lifetime cost / capex | Net RTE (life avg) | Notes |
|---|---|---|---|---|---|
| `20261119_143649_p75` | $75 | 376 (159) | $378M / $82.5M | 36.7% | early, short |
| `20261119_164422_p75` | $75 | 3328 (2855) | $314M / $78.9M | 36.8% | L pinned at the old 6 m bound and shade at 3 in; bounds widened after |
| `20261120_150244_p75` | $75 | 614 (209) | $321M / $88.0M | 37.1% | stopped early |
| `20261120_161611_p150` | $150 | 3328 (2137) | $541M / $91.3M | 37.9% | current model (wall fix, L_bot, H_max 14, 120 °C floor) |

The p150 optimum:

- 107 sticks + 107 heaters;
- stick 204 mm × 9.45 m in a 315 mm liner, block 10.86 m tall;
- flux target 8.0 W/cm²;
- block 1772–2356 °C;
- shade 3.375 in;
- insulation 2.19 m side / 1.78 m top, L_bot 3.33 m;
- tank 39.2 m diameter × 29.2 m;
- heater 585 kW each, interconnect 63.9 MW.

Full cycle at screening fidelity, clean cells:

| Quantity | Value |
|---|---|
| Net RTE | 38.9% |
| Cell efficiency (DC / absorbed) | 46.1% |
| RTE / cell efficiency | about 85% |
| Delivery (absorbed / AC bought) | about 89% |
| Output (net AC / DC) | about 95% |

Observations worth knowing:

- **Liner temperature.** During discharge the liner inner face runs at about 1605–1650 °C, mostly near 1605 °C. The TPV fit peaks at 1845 °C (48.5%), so the cells run about 2 points below their best efficiency. Liner temperature is tied to the flux target through the shade control.
- **55% what-if.** With a flat 55% cell efficiency, the same layout re-sized gives about 47.0% net RTE on the clean cycle and 45.8% life average. RTE / cell efficiency stays at about 85% because delivery and output don't depend on the cells.
- **Floor study** (previous optimum):
  - at 6 m/s the column hot spot was about 191 °C;
  - at 10 m/s it was 148 °C (+$1.2M objective);
  - L_bot 3 m brings it to about 112 °C.

  That is why L_bot became a variable.

## 8. Known simplifications and things not modelled

- All cells run in step: synchronous operation, no edge-cell effects in the grid.
- No heat loss through the hold-phase door or the roof penetrations (sticks, heaters, shade axles), and no gas-tightness cost for the penetrations.
- The round floor is treated as a rectangle with single-pass air flow.
- Charging and discharging cells are annular analysis cells with an adiabatic outer radius; the real layout is a checkerboard.
- Argon cost is ignored. Several costs are placeholders (§4).
- The deposition convection factor is a fixed ×3.5 (shape-dependent; see §4).

## 9. Deferred work (agreed, not started)

Nate deferred these on 2026-11-20; together they would roughly halve optimizer run time and improve accuracy:

1. Interpolate the discharge step that crosses the T_min cutoff. Today that step is dropped from the stick's energy but counted in the discharge time.
2. Fix the shade-opening lag. `sizing.shade_open_point` pairs the shade fraction with the end-of-step temperature, so the opening is detected about one step late.
3. Then switch screening to one uniform step of about 3600 s, replacing the three-tier dt_short / medium / long.
4. Cache the vane view factors per design, in memory. The ray trace is about 40% of a screening discharge, and each design re-traces the same cavity about 4 times.

Other open items: the optimal flow-path layout for the round floor; heater power / charge time as an
optimizer variable; the door and penetration losses.

## 10. Glossary

| Term | Meaning |
|---|---|
| stick / powerstick | Square TPV cell assembly inserted into the cavity during discharge |
| liner | Graphite (or grafoil) square tube around the stick; its inner face is the emitter the cells see |
| shade / vanes | 8 rotating graphite-felt vanes around the liner that throttle radiation to hold the flux target |
| f_shade | Shade open fraction |
| flux target / avg_flux | Heat absorbed by the active cells per cm², held constant by the shades |
| shade-open sizing | Block sizing in which the shades reach fully open exactly at the end of the 50 h discharge |
| swing | T_max − T_min of the block |
| D, D_cell, D_bare | Degradation = deposit / 10 nm |
| delivery | Heat absorbed by the active cells / AC bought |
| L_bot | Bottom stack height under the hot zone (support columns + Thermax) |
| f_3D, f_foot | Column heat-spreading factors: total heat and foot hot spot |
| screening / DETAILED | Optimizer fidelity levels (§3, §6) |
| RGF | Rigid graphite felt |
| CFC | Carbon-fibre-reinforced carbon |
| Thermax | Tank insulation material (side, top and bottom stack; k(T) in `properties.thermax_tcx`) |
