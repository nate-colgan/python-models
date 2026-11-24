# Cost scenarios for the Tinless optimizer: Pilot, FOAK, NOAK

**Source:** `20260511_FP-system-cost-model_v6.xlsx`, a cost model by another Fourth Power engineer.

The spreadsheet's three later cost columns map to the scenarios here:

| Scenario | Spreadsheet column | Header on the Inputs sheet |
|---|---|---|
| **Pilot** | 2027 Pilot | "High RTE" (a stale label; the block sheets use this column as 2027) |
| **FOAK** | 2028 Phase 1 | "Low RTE" |
| **NOAK** | 2034 Phase 3 | 2034 |

**Who sets this up:** Nate. **For:** Claude Code. **Date:** 2026-11-24.

## Read first

**The spreadsheet models a different architecture:** a tin loop, 10 h, daily cycle, in a warehouse.

- Only its **unit costs and efficiencies** are used here. Its plant sizes, CPE/CPP and RTE outputs are not.
- Nate's model keeps its own product definition: `P_rated` 25 MW, 50 h, weekly cycle.

**Key names** follow `docs/tinless_model_documentation.docx` §9.5 and §5.8 and `model_guide_for_claude.md` §4.

- Confirm each name against `thermal/optimize.py` (`OPT_DEFAULTS`) and `thermal/tank.py` (`TANK_DEFAULTS`) before wiring it in.
- If a key differs, map to the existing key. Do not add duplicates.

**Where each kind of input goes:**

- **Optimizer inputs** (economics, costs, parasitics) go in `OPT_DEFAULTS` / `OPT_INPUTS`.
- **Tank and graphite rates** live in the YAML. Change them **only** by editing the `yaml_content` string in cell `a8c3d501` of `tinless system model.ipynb` (see guide §6).
  - The optimizer uses the central value [1] of each low/central/high triple.
  - Set the central value as listed and leave low/high alone unless noted.

**Add a scenario selector.** For example `OPT_INPUTS['cost_scenario'] in {'pilot', 'foak', 'noak', None}`, applied as an override layer on top of `OPT_DEFAULTS` and the YAML tank costs.

- `None` must reproduce current behaviour exactly, so the regression tests still pass.
- Record the active scenario in each run's `inputs.json` and in the run-folder name, e.g. `_p75_noak`.

**No cell-efficiency override.** The spreadsheet's TPV efficiency is reference data only (see §5). Do not turn on a flat `panel_efficiency` override as part of a scenario.

## 1. Optimizer inputs (`OPT_DEFAULTS` / `OPT_INPUTS`)

| Key | Current | Pilot | FOAK | NOAK | Unit |
|---|---|---|---|---|---|
| `economics.eta_in` | 0.98 | **0.95** | 0.98 | 0.98 | - |
| `economics.eta_out` | 0.98 | **0.95** | 0.98 | 0.98 | - |
| `economics.soft_cost_frac` | 0.30 | **0.55** | **0.375** | **0.25** | - (excludes O&M, see note) |
| `costs.cell_per_cm2` | 1 | **36.52** | **4.557** | **1.872** | $/cm2 cell area, all-in module cost |
| `costs.stick_per_m` | 100 | **339** | **263** | **217** | $/m |
| `costs.stick_fixed` | 2,000 | **7,510** | **5,110** | **4,510** | $/stick |
| `costs.heater_fixed` | 5,000 | replaced by the §3 model | | | |
| `costs.interconnect_per_kW` | 150 | **357.8** | **286.2** | **257.6** | $/kW, on max(peak charge AC, peak output AC) |
| `costs.charge_electrics_per_kW` | 75 | **0** | **0** | **0** | $/kW (included in interconnect) |
| `costs.inverter_per_kW` | 100 | **0** | **0** | **0** | $/kW (included in interconnect) |
| `costs.heat_rejection_per_kWth` | 125 | **780** | **567** | **496** | $/kWth peak stick heat rejection |
| `parasitics.heat_rejection_kWe_per_kWth` | 0.03 | **0.05** | **0.05** | **0.05** | kWe/kWth |

Unchanged in all scenarios (no spreadsheet equivalent; keep the current values):

- **Liner:** `liner_wall_per_m2`, `liner_floor_per_m2`, `liner_floor_area_factor`.
- **Shades:** `shade_rgf_per_m3`, `shade_axle_per_m`, `shade_axle_extra`, `shade_motor`, `shade_linkage`, `shade_drive`.
- **Other:** `bare_clean_per_stick`, `elec_price`, `cycles_per_year`, `lifetime_yr`.

### Derivation notes

**`soft_cost_frac`** = soft costs + contingency + controls:

| Scenario | Soft costs | Contingency | Controls (% of project) | Total |
|---|---|---|---|---|
| Pilot | 0.40 | 0.05 | 0.10 | 0.55 |
| FOAK | 0.25 | 0.05 | 0.075 | 0.375 |
| NOAK | 0.15 | 0.05 | 0.05 | 0.25 |

- O&M is **not** included. The spreadsheet's maintenance is 5% / 3% / 2% per year of "active capex", plus staffing (§5).
- Nate's current 0.30 was meant to cover soft costs, controls and O&M at NOAK maturity. 0.25 + O&M on the spreadsheet basis (~6% of capex over life) ≈ 0.31, so it is consistent.
- **Do not add an O&M model.** Nate decides how O&M is treated.

**`cell_per_cm2`** is the TPV module total cost ÷ cell area per module. Each module is 120 cells × 0.8560 × 0.8760 cm = 89.98 cm². The module total is cells + metal-core PCB + cold plate, divided by 0.85 (the assembly and test cost is 15% of the module total).

| Scenario | Cell $/cm2 | $/cell | PCB | Cold plate | Materials | Module total | $/cm2 all-in |
|---|---|---|---|---|---|---|---|
| Pilot | 30 | 22.496 | 3.951 | 90 | 2,793.43 | 3,286.39 | 36.52 |
| FOAK | 3 | 2.2496 | 3.556 | 75 | 348.50 | 410.00 | 4.557 |
| NOAK | 1 | 0.74986 | 3.200 | 50 | 143.18 | 168.45 | 1.872 |

The all-in value is used because the model also charges `cell_per_cm2` for every cell replacement, and replacements are module-level.

**`stick_per_m`** = aluminium core + gas directors, per metre.

- **Core:** 16.17 kg/m × aluminium $/kg (4.545 / 3.636 / 3.182) + fabrication ($150 / $75 / $30 over 3.41 m) → 117.5 / 80.8 / 60.3 $/m.
- **Gas directors:** 4 gold-plated directors over 3.08 m. Per stick $682.5 / $562.5 / $482.5 → 221.5 / 182.5 / 156.6 $/m.

**`stick_fixed`** = bottom cap + take-up/actuation + cavity assembly and test + on-site assembly. The spreadsheet's emitter-cavity walls and tungsten liner are tin-design items and are excluded; Nate's liner and shades are costed separately.

| Scenario | Bottom cap | Take-up / actuation | Cavity assembly and test | On-site assembly | Total |
|---|---|---|---|---|---|
| Pilot | 509.6 | 5,000 | 500 | 1,500 | 7,510 |
| FOAK | 509.6 | 3,000 | 400 | 1,200 | 5,110 |
| NOAK | 509.6 | 3,000 | 250 | 750 | 4,510 |

On-site assembly is 10 / 8 / 5 h at $150/h.

**`interconnect_per_kW`** = 2 × electrical balance of system (178.89 / 143.12 / 128.80 $/kW equipment, plus an equal labour and cabling allowance).

- The spreadsheet sizes all electrical on peak charge power. That matches this model's interconnect basis, max(charge, output).
- `charge_electrics_per_kW` and `inverter_per_kW` are therefore 0, to avoid double counting.

**`heat_rejection_per_kWth`** = 2 × dry-cooler cost (300 / 270 / 243 $/kW, plus an equal distribution and install allowance) × the "pass-through" cooling margin (1.30 / 1.05 / 1.02).

- Without the margin it would be 600 / 540 / 486.
- The spreadsheet's own risk log notes that dry coolers may cost more than this.

**`heat_rejection_kWe_per_kWth`** = 1/20, the spreadsheet's dry-cooling ratio (BAC Trillium, 95 °F dry bulb).

## 2. YAML tank and graphite rates (central values)

| Line (doc §5.8) | Current central | Pilot | FOAK | NOAK | Unit |
|---|---|---|---|---|---|
| Storage graphite (pass-through) | 1.0 | **1.50** | **0.75** | **0.50** | $/kg |
| Column graphite (iso-molded) | 10 | **20** | **16** | **2** | $/kg |
| Thermax side / top / bottom | 2.5 | 2.5 | 2.5 | 2.5 | $/kg (unchanged, see note) |
| Structural steel: shell, bottom, roof, deck trusses, bearing plates | 5 | **5.03** | **5.03** | **5.03** | $/kg (= $5,030/t installed) |

- **Storage graphite** for FOAK and NOAK is baked aluminium-anode-type carbon, not graphite. Its properties differ (density, conductivity, purity). Treat those cases as a material change, not only a price.
- **Thermax:** the spreadsheet's FOAK/NOAK insulation prices ($0.31, $0.279/kg) are for **petcoke**, not Thermax. Nate excluded them, so Thermax stays $2.50/kg (the pilot quote) in all scenarios.
- **Steel:** the spreadsheet rate is bolted structural framing. If the tank steel lines are a low/central/high triple, set only the central value and keep the existing low/high.
- **Unchanged** (no spreadsheet equivalent): gas tightness, oxide posts, column tie grid, floor plate/fins/welds, heat-resistant slab/rails/firebrick, cellular glass, structural mat, fans/louvres.

### Shared iso-molded graphite price

Add one scenario parameter, `iso_graphite_per_kg` = 20 / 16 / 2 $/kg. Drive both of these from it:

- the column graphite rate in the tank cost, and
- the heater graphite term in §3.

In `None` (default) mode it must equal the current column-graphite central rate (10 $/kg), so nothing changes.

## 3. Heater cost model (replaces `costs.heater_fixed` when a scenario is active)

The spreadsheet costs a 1 MW_th panel heater (1.5 m × 1.016 m × 12.7 mm graphite body, 34.84 kg at 1,800 kg/m³) with two tungsten rod elements. Its cost is built from the iso-molded graphite price, CNC time, tungsten and assembly labour.

Implement it per heater, scaled with the heater's rated thermal power `P_h` (W):

```
f   = P_h / 1e6                                     # spreadsheet heater = 1 MW_th

C_heater = f * ( m_g * p_iso                        # graphite body material
               + h_body * r_cnc                     # body machining
               + n_el * (m_W * p_W + h_el * r_cnc)  # tungsten elements
               + h_at * r_asm )                     # assembly and test
         + h_site * r_site                          # on-site install, per heater (not scaled)
```

| Parameter | Meaning | Pilot | FOAK | NOAK | Unit |
|---|---|---|---|---|---|
| `m_g` | graphite body mass per MW_th | 34.84 | 34.84 | 34.84 | kg |
| `p_iso` | iso-molded graphite price (shared, §2) | 20 | 16 | 2 | $/kg |
| `h_body` | body machining time per MW_th | 20 | 18 | 15 | h |
| `r_cnc` | CNC rate | 200 | 150 | 100 | $/h |
| `n_el` | elements per MW_th | 2 | 2 | 2 | - |
| `m_W` | tungsten per element (Ø50.8 mm × 0.5 m) | 19.56 | 19.56 | 19.56 | kg |
| `p_W` | tungsten rod price | 50 | 50 | 50 | $/kg |
| `h_el` | element machining time | 0.5 | 0.5 | 0.5 | h |
| `h_at` | assembly and test time | 0.5 | 0.5 | 0.5 | h |
| `r_asm` | assembly labour rate | 50 | 50 | 50 | $/h |
| `h_site` | on-site install time per heater | 10 | 8 | 5 | h |
| `r_site` | on-site construction rate | 150 | 150 | 150 | $/h |

**Check values** for a 1 MW_th heater. These must match the spreadsheet's Heater Block "Heater Total Cost" plus on-site assembly:

| Scenario | Heater ex-site | + on-site | Total |
|---|---|---|---|
| Pilot | 6,877.66 | 1,500 | **8,377.66** |
| FOAK | 5,388.31 | 1,200 | **6,588.31** |
| NOAK | 3,650.57 | 750 | **4,400.57** |

Add a unit test that reproduces these three numbers.

### Notes for implementation

- **Use the existing heater power.** `P_h` is the per-heater rated power the optimizer already sizes in step 3 of the guide (the minimum power that charges in `charge_hr`). With the scenario active, charge electrics are folded into the interconnect (§1), so there is no separate per-kW heater-electrics term.
- **Linear scaling is a choice.** Scaling body, machining and elements linearly with `P_h` keeps the spreadsheet's cost per MW_th. Nate's heaters are long rods in round cavities, not 1 m panels.
  - If the model has a heater graphite volume, a geometry-based alternative is `m_g` = actual heater graphite mass, with `h_body` scaled by mass.
  - Implement the per-MW version first and put the geometry option behind a flag. Ask Nate before making it the default.
- **Size of the lever:** the graphite term is a small part of heater cost (~8% Pilot, ~2% NOAK). Machining time and the tungsten elements dominate. Coupling the heater to the iso-molded price is correct, but it will barely move the optimum.

## 4. Things that stay as they are

- **Product definition:** `P_rated`, `duration_hr`, `charge_hr`, cycle length, `elec_price`, `lifetime_yr`, `cycles_per_year`.
- **Physics:** all YAML physics, including the TPV efficiency fit and the deposition `conv_factor`.
- **Constraints:** all optimizer constraints and design-variable bounds.
  - Recommendation for the Pilot scenario only: cells at ~$36.5/cm² will push the optimum hard toward high flux. Consider raising the `avg_flux` upper bound above 20 W/cm² for Pilot runs. Nate decides.

## 5. Reference data from the spreadsheet (not inputs; do not wire in without Nate)

| Item | Pilot | FOAK | NOAK | Note |
|---|---|---|---|---|
| TPV net cell efficiency | 0.20 | 0.50 | 0.55 | Spreadsheet target. The model uses its own `panel_efficiency` fit. |
| Insulation efficiency (per 24 h cycle) | 0.80 | 0.99 | 0.975 | Sizes the spreadsheet's insulation. The optimizer chooses thickness itself. |
| Maintenance materials | 5% | 3% | 2% | Per year of "active capex". The OPEX sheet shows 2.5% for 2034, which is inconsistent. |
| Staffing | 0.5 engineers + 2 techs per 100 MW | 0.2 + 0.5 | 0.2 + 0.5 | $200k / $100k per FTE-year |
| Electronics parasitics | 0.05 MW idle, 0.1 MW discharging | same | same | 0.2% / 0.4% of 25 MW. The sheet comment says 2%, but the values are 0.2%. Not in the model. |
| Argon | $0.567/yd³, 1%/day leakage | same | same | Negligible |
| Labour | assembly $50/h, CNC $200/h, on-site $150/h | CNC $150/h | CNC $100/h | Used in §3 |
| Sales tax / freight | blank | blank | blank | Only filled for the 2026 Tech Demo column (6.25% / 10%) |

## 6. Checks before handing back

1. `cost_scenario=None` reproduces the current baseline. The regression test passes, and no `panel_efficiency` override is active.
2. The heater unit test reproduces the three check values in §3.
3. One design evaluated in each scenario prints a cost breakdown. The changed lines move in the expected direction:
   - cells, sticks and interconnect go up for Pilot and down for NOAK;
   - heat rejection goes up in every scenario;
   - storage graphite goes down for FOAK and NOAK.
4. The run report's model-settings table shows the scenario name and the as-run values with their source.
