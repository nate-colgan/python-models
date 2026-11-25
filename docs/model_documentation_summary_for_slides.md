# Tinless thermal battery model: documentation summary for a slide deck

**Purpose of this file.** Source material for Claude (desktop app) to build a PowerPoint summary of
the tinless system model documentation, with key takeaways. It condenses every section of
`docs/tinless_model_documentation.docx` (generated from the code by `thermal/docs*.py`) and adds the
headline results of the latest optimizer run. It includes the new work on heater sizing and
heater current leads (documentation section 4.8).

**Sources and dates.**
- Model documentation as built on 2026-11-24/25 from branch `tank-system-model` (commit d734970b plus
  uncommitted heater-lead changes). The copy of the .docx in `docs/` dated 2026-11-20 predates
  section 4.8; re-running the `opt_docs` cell regenerates it.
- Results: optimizer run `opt_runs/20261125_001641_p87_pilot` (Pilot cost scenario, $87/MWh,
  1 MW x 50 h plant). The run report is `report.docx` in that folder.
- Context on decisions and earlier runs: `docs/model_guide_for_claude.md`.

**Notes for the deck builder.**
- Audience: engineers and technical leads at the company who know the concept but not the model's
  internals. Keep the physics at the level of "what is modelled and how", with numbers.
- Suggested length 15-20 slides; an outline is at the end (section 15).
- Figures worth reusing are in the .docx and the run report (listed in section 13). Equations in the
  .docx are images; the key ones are written out below in plain text.
- The numbers in the "Items to review" list are the model's own self-critique, not confirmed
  errors. Present them as an open-issues list.

---

## 1. Key takeaways (for the summary slide)

1. **One model chain from physics to dollars.** A single stick cell and a single heater cell are
   simulated in detail (radiation, shades, conduction, cells), scaled to a plant in an insulated
   steel tank, run through a weekly cycle with a closed energy ledger, costed, and searched by a
   differential-evolution optimizer for the lowest lifetime cost.
2. **Objective = lifetime capex + value of the electricity lost** (one price, $87/MWh, for buying
   and selling). This makes round-trip efficiency (RTE) and capital cost directly tradeable.
3. **Latest pilot optimum (1 MW x 50 h, Pilot costs):** 7 powersticks + 7 heaters, block
   1789-2247 C, 8.1 m tall, tank 14.2 m dia x 22.1 m. **Net RTE 27.0 % (life average)**,
   initial capex $26.4M, objective $42.7M.
4. **Where the energy goes:** RTE = delivery (71.6 %) x cell efficiency (44.9 %) x output (84.1 %).
   Cell conversion is the dominant loss (39 % of the electricity bought); insulation losses over the
   week come second (about 17 %, 2.8 %/day of stored heat). The detailed full-cycle check with clean
   cells gives 29.0 %.
5. **Powersticks dominate capex** (58 % of initial capex at $36.5/cm2 cells), then the tank (22 %),
   heat rejection (8 %) and grid interconnect (6 %).
6. **Heaters are sized, not assumed.** Each design gets the smallest heater power that charges the
   block in 50 h (366 kW for the optimum), and a graphite tube element sized so 60 V DC drives
   that power (188 mm OD, 6.1 kA).
7. **Heater current leads are now in the model.** The leads (graphite / tungsten / water-cooled
   copper, through the roof insulation) are optimized for lowest lifetime cost and fed into every
   design's capex, energy ledger, RTE and objective. For the optimum they cost **0.8 RTE points**
   (27.8 % -> 27.0 %), **3.8 MWh per cycle** (2.5 % of the electricity bought) and about **$0.7M
   over the life**, about 80 % of it energy, not hardware.
8. **The lead table is accurate.** A full lead optimization of the best design agrees with the
   fast table the optimizer used to within 1.3 % on lifetime cost and 0.4 % on energy lost.
9. **Lead cost and losses scale with heater current**, so a higher heater supply voltage gives
   proportionally cheaper, lower-loss leads. 60 V DC means 6.1 kA per heater; the voltage is a lever
   worth studying.
10. **Known risks are listed in the model itself.** The documentation carries 74 review items
    (section 11). The most important open physics questions are flagged in section 13 of this file.
    Among them: the tungsten-graphite lead joint running at tungsten's 2500 C limit (carbide
    formation), unsourced block graphite properties, and the TPV efficiency fit, which sets every
    electrical output.

---

## 2. The system (doc section 1.1)

- **Charge:** resistive graphite heaters in round cavities heat a large graphite block (up to
  about 2250-2500 C).
- **Hold:** the block is held hot; the heaters make up the losses.
- **Discharge:** water-cooled square **powersticks** covered in TPV cells are lowered into square
  cavities only while discharging.
  - Each stick hangs in a thin **graphite liner**, which keeps graphite vapour off the cells and is
    the emitter the cells see.
  - **Eight rotating vanes (shades)** of rigid graphite felt surround the liner. They throttle
    radiation from the cavity wall so the cells see a constant flux while the block cools.
- **Layout:** sticks and heaters alternate on a checkerboard in one continuous graphite matrix
  (stick cavities turned 45 deg). The hot zone sits in a gas-tight, argon-filled, insulated
  (Thermax) steel tank, on graphite-to-oxide support columns over an air-cooled finned floor.
- **Weekly cycle:** about 50 h charge, about 68 h hold, about 50 h discharge (168 h).

## 3. How the model is organised (doc 1.2, Figure 1.1)

| Level | What it does | Code |
|---|---|---|
| Stick cell | Annulus of graphite (one stick + heater pair's share) around one stick cavity; full-height radiation enclosure with vanes, liner and stick; discharge transient | `build.py`, `tinless.py`, `axial.py`, `viewfactors.py`, `correlations.py` |
| Heater cell | Same graphite volume around one heater cavity; charge transient | `build.py`, `tinless.py` |
| Heater and leads | Element sizing from power and voltage; current-lead model, optimizer and lead table | `heater.py`, `heater_leads.py` |
| Plant | Hot-zone layout, tank, insulation losses, columns, floor cooling, foundation, costs | `tank.py` |
| Cycle | Charge -> hold -> discharge with a closed energy ledger | `cycle.py` |
| Optimizer | 10 design variables, evaluation pipeline, objective, differential evolution | `optimize.py` |
| Reports | Run report and this documentation (Word) | `report.py`, `docs*.py` |

Conventions: K inside the cell model, C in the tank model and optimizer outputs; flux in W/cm2 of
active cell area; "screening" vs "detailed" are the optimizer's two fidelity levels.

## 4. Inputs (doc section 2)

- One YAML file, `tinless_model_input.yaml`, drives the physics. It is written by a notebook cell and
  never edited directly. The optimizer overwrites about 20 of its keys for every design (geometry,
  flux, temperatures, mesh, tank insulation, plant rating).
- The documentation lists every YAML key with its value, which keys are ignored or unused in the
  current configuration, the code defaults used when a key is missing, and the YAML comments that
  disagree with the code. Example: `end_time` says 50 h but is 66.7 h; the `dt_long` comment
  describes a different rule from the one the code uses.
- All cost and economic inputs for the optimizer live in the optimizer notebook (`opt_inputs`), with
  Pilot / FOAK / NOAK cost scenarios applied on top.

## 5. Geometry and discretisation (doc section 3)

- **Equivalent circles:** every square section (stick, liner, cavity) becomes the circle with the
  same perimeter, so radiating areas are exact. The block volume is set from the energy need, so
  graphite mass is right; the conduction path is an annulus rather than a square cell. That is
  review item 2.
- **Cavity size** is the smallest square that lets the 8 vanes swing fully open with 5 mm clearance
  (the YAML area ratio is ignored).
- **Block volume** comes from the heat the stick must absorb over the discharge at the target flux:
  V = E_req / (rho c_p dT). A sizing loop then corrects it (section 7 below).
- **Lower zone:** the block is always taller than the stick (at least 0.25 m). Stick and liner hang
  top-aligned. A liner-bottom plate with a felt pad closes the liner; the lower block rings still
  radiate into the cavity.
- **Mesh:** total axial rings over the block height (0.5 m rings in optimizer screening, 1/3 m
  detailed), split between the lower and active zones; 5 radial rings of equal thickness.
- **Heater cell:** same graphite volume, height and mesh around a round heater cavity; starts at a
  uniform T_min. The stick cell starts at a uniform T_max.

## 6. Component physics (doc section 4)

### 6.1 Block conduction (4.1)
- Axisymmetric r-z finite volume, explicit forward Euler with automatic sub-stepping for stability.
- Cavity-wall radiation is very stiff (equivalent axial k of about 1000 W/m-K), so the wall node is
  coupled implicitly through the radiation model's sensitivity matrix dQ/dT_wall (IMEX scheme).
- Axial conduction through graphite is negligible over a cycle (axial diffusion time 168 days);
  axial heat redistribution happens by radiation across the cavity.
- Block properties are constants: rho 1450 kg/m3, c_p 2000 J/kg-K, k 20 W/m-K.

### 6.2 Cavity radiation (4.2)
- Gray, diffuse radiosity enclosure over the full block height: 272 surfaces per vane position
  (20 block rings, 6 liner rings, 160 vane faces, 80 vane end faces, pad, liner edge,
  floor / ceiling).
- View factors are ray-traced (Sobol quasi-Monte Carlo, 16,384 rays per surface detailed,
  4,096 screening) in the real square geometry with real vane thickness, gaps and shadowing.
  They are then corrected to satisfy reciprocity and summation exactly (Sinkhorn scaling).
- Solved by Newton's method (about 3.4 iterations per call, about 150 ms per controlled solve).
- The lower zone supplies a large share of the heat: 41 % with vanes closed, 7 % open, at a
  uniform 2423 K wall.

### 6.3 Shades / rotating vanes (4.3)
- Eight vanes of graphite felt on an octagon, full block height, all turning by the same angle.
- Three vane angles are fully traced and solved (0, 10 and 90 deg); the actual position is a blend of
  the three (blend parameter s from 0 = closed to 2 = open). Totals are exact; the per-ring split is
  within about 1 % of a direct trace.
- **Control law:** each step, find the vane position that holds the total cell power at the flux
  target. If even closed vanes deliver too much (hot block early in discharge), they sit closed
  and over-deliver. If even open vanes deliver too little, the stick is near the end of its
  discharge.
- Vane conduction uses the exact integral of the felt's k(T); vanes are quasi-steady.

### 6.4 Liner and liner bottom (4.4)
- Each liner ring is a quasi-steady chain: absorbs on its outer face, conducts through the 5 mm
  wall, radiates spectrally from its inner face to the stick opposite. The chain is pre-tabulated
  against temperature (LinerTable).
- Spectral exchange uses 20 Gauss-Legendre wavelength nodes (10 per band), exact for the stick's
  two-band emissivity.
- The liner bottom (felt pad, graphite plate, stick bottom cap) has its own quasi-steady model.

### 6.5 Powerstick and cells (4.5)
- Stick surface: 5 % bare, 85.5 % active cells, 9.5 % inactive (gaps, busbars).
- Two-band emissivity: active cells absorb 0.9 below the 1.24 um bandgap and 0.015 above it;
  bare and inactive areas absorb 0.02.
- Electrical power = absorbed cell heat x eta(T_liner, flux) x (1 - wear). eta is a quadratic fit in
  the emitter (liner) temperature, peaking at 48.5 % at 2118 K (1845 C), with a mild flux derate.

### 6.6 Heater and the charge cell (4.6)
- Heater and cavity wall per ring as concentric gray cylinders; one heater temperature for the
  whole element.
- **Phase 1 (power-limited):** the heater runs at its full power and its temperature floats below
  the limit.
- **Phase 2 (temperature-limited):** the heater sits at its 2600 C limit and the power falls as the
  wall heats up.
- The charge ends when the block's volume-average temperature reaches T_max.

### 6.7 Deposition and degradation (4.7)
- Graphite evaporates from the hot liner and diffuses across the argon gap to the cold stick
  (stagnant-film diffusion with a x3.5 convection factor from in-house CFD).
- The deposit darkens bare and inactive areas and reduces cell efficiency. Wear D = deposit / 10 nm,
  linear. Cells are replaced at D = 0.1; bare areas are cleaned at D = 0.01.
- For the pilot optimum: 0.46 pm per cycle, 0 cell replacements over 30 years (interval 42 years),
  7 bare-area cleanings.

## 7. Heater sizing, element and current leads (doc section 4.8, NEW)

This is the new section of the documentation. It covers how the heater power, the heater element and
the current leads are sized, and how their cost and losses enter the plant.

### 7.1 The chain for every design (Figure 4.8.1)
1. **Heater search:** find the smallest heater power that charges the block in the target time
   (50 h) with the cavity at its minimum clearance. The cavity grows only if the power cap is not
   enough.
2. **Element:** a graphite tube sized so the supply voltage drives that power through it. This sets
   the heater current I = P / V.
3. **Leads:** two per heater (supply + return) carry I from the top of the element up through the
   roof insulation (thickness t_top) to the deck. Their geometry is optimized for lowest lifetime
   cost.
4. **Plant:** a lead table of optimal leads, built once per optimizer session with that session's
   costs, gives every design its leads' cost, Joule heat, heat drawn from storage and heat
   rejected. These enter capex, the energy ledger, RTE and the objective.

### 7.2 Heater search (optimizer)
- Reject if the heater limit (2600 C) is not above the block T_max, or if even the power cap
  (53.9 kW per m of block height) cannot charge in time.
- Charge at the cap with the cavity at its minimum (element radius + 0.25 m). If that meets the
  target, bisect down to the smallest power that still does (3 % tolerance screening, 1 % detailed).
  **The element is resized for every power tried**, so each charge run tests the element that power
  needs.
- Otherwise grow the cavity (area ratio x 1.6 per try, up to 8 tries). A bigger cavity spreads the
  power over more wall, so the heater stays below its limit for longer.
- Why minimise heater power: the peak charging draw sizes the grid interconnect, which is a large
  cost line ($358/kW in the Pilot scenario).

### 7.3 Heater element
- Tube of outer diameter D and wall t = 0.1 D (`heater.type: tube`). Sized at the highest graphite
  resistivity over the operating range, so the supply can always deliver the power:
  - R = V^2 / P; R = rho_e x legs^2 x H / A; A = pi D^2 w (1 - w), with w = t / D
  - D = sqrt(rho_e legs^2 H P / (V^2 pi w (1 - w)))
- Resistivity from Mersen 2020 data to 2002 C, extrapolated above: 12.2 uOhm m at the 2600 C
  limit.
- Element graphite = A x block height x 1850 kg/m3, costed at the iso-molded graphite price. With
  the leads modelled, the element no longer runs up through the roof insulation; the leads do.
- **Pilot optimum:** 366 kW per heater -> tube OD 188 mm x 18.8 mm wall, 9.83 mOhm, 6.11 kA at
  60 V, 150 kg graphite, $10,494 per heater.

### 7.4 Current leads: geometry
- Up to four material zones in series, hot to cold. The system uses **graphite / tungsten / copper**,
  with the copper **water-cooled** along its length. Each zone is a solid round bar.
- Each zone has a body diameter (bounds scale with sqrt(I)) and a length of at least 5 cm.
- **No necks:** two bars at a joint share one contact diameter (the smaller body diameter); the
  larger bar widens from it at no more than 60 deg. Without this rule the optimizer picked
  unbuildable step changes at joints.
- Each lead sits in a bore in the Thermax with a 5 mm argon gap.
- **Idle disconnect:** the graphite-tungsten joint is opened (10 mm gap) whenever the heater is off,
  during hold and discharge, and closed for charging. This cuts the heat the leads leak out of the
  storage.

### 7.5 Current leads: 1-D thermal model
- Steady conduction along the lead with Joule heating and exchange with the insulation:
  d/dx(k A dT/dx) + I^2 rho_e(T) / A - q_ex = 0
- Exchange q_ex: gas conduction and radiation across the bore gap, in series with radial
  conduction through the Thermax out to 0.5 m, where the Thermax is at its undisturbed temperature.
- Undisturbed Thermax profile: exact for k = K0 exp(B T); exp(B T_w) is linear in height. It is
  hotter than a linear profile over most of the thickness.
- **Two states per cycle:**

| State | Current | Hot end | Hours per cycle |
|---|---|---|---|
| Charging | rated I = P / V | heater limit, 2600 C | charge time (about 50 h) |
| Heater off (hold + discharge) | 0 | block mean temperature | rest of the week (about 118 h) |

- Cold end: water-cooled copper held at 40 C.
- Numerics: 80 finite-volume segments, Newton on the sources and Picard on k(T), converged to
  1e-3 K. A lead whose temperature runs away (Joule heat outruns conduction) is infeasible.

### 7.6 Energy accounting and lifetime cost per lead
- Heat the lead puts into the insulation splits between the deck (up) and the storage (down) in the
  ratio of the slab resistances on either side. This is exact for the exponential k(T).
- Per lead and per cycle: Joule heat while charging (bought), net heat drained from the storage
  (negative while charging, when Joule heat flows down into the block), heat rejected to the
  cooling water.
- Lifetime cost of one lead, priced like the system optimizer (no discounting, 52.1 cycles/yr x
  30 yr, $87/MWh):
  - capex = [material volume x $/m3 + joints + dry cooler $/kWth x peak rejected heat + supply
    $/kW x Joule power] x (1 + soft costs)
  - energy = lifetime cycles x price x [Joule heat + storage drain / eta_in + 0.05 kWe/kWth x
    rejected heat]
- Temperature limits: each zone below its material limit; each joint below the colder material's
  limit. Every kelvin over costs a $1,000 penalty.

### 7.7 Lead optimizer
- Differential evolution over body diameters and length fractions (popsize 12, 150 generations,
  then a Nelder-Mead polish). Every point in the search box is a valid lead.
- **Zone skipping:** if an optimum leaves a zone at its 5 cm minimum, the optimization is re-run
  without it (graphite / copper, tungsten / copper, ...). The cheapest feasible set wins.
- Materials library: graphite, tungsten, molybdenum, copper, aluminum, stainless. Copper, aluminum
  and stainless are coolable.

| Material | k at 1000 K [W/m-K] | rho_e at 1000 K [uOhm cm] | T_max [C] | $/m3 |
|---|---|---|---|---|
| graphite | 66 | 1049 | 3000 | 37,000 |
| tungsten | 118 | 24.9 | 2500 | 2,500,000 |
| molybdenum | 112 | 22.9 | 1800 | 600,000 |
| copper | 352 | 6.9 | 200 | 110,000 |
| aluminum | 218 | 10.2 | 150 | 14,000 |
| stainless | 25 | 113 | 800 | 50,000 |

### 7.8 Lead table (how the system optimizer uses the leads)
- Optimizing leads for every design would take minutes each. Within one session the prices and
  limits are fixed, and the optimal lead depends on only three things: **heater current, lead length
  (t_top) and block T_max**.
- A scan over 3-12 kA showed **lifetime cost per ampere flat within 3.5 %**, so the table stores
  per-ampere values.
- Grid: t_top 0.5-3.0 m (6 points) x T_max 1800-2500 C (4 points) x I = 4 and 10 kA. That is 48
  lead optimizations, about 10-25 min on 20 workers, rebuilt at the start of every optimizer
  session and cached under a hash of its inputs.
- Each design's values are interpolated linearly per ampere at its own current, t_top and T_max.
- Typical values (Pilot, $87/MWh): **$5,700-10,700 lifetime per kA per lead** and **34-56 kWh lost
  per cycle per kA per lead**. Both rise with t_top and with T_max.

### 7.9 Leads in the plant model
- n = 2 x N leads. Per cycle: E_joule = n x P_J x t_charge; E_drain = n x (drain_charge x t_charge
  + drain_idle x t_off); E_rej = n x (rejected heat while charging and idle, where positive).
- Electricity bought = (heater heat + E_drain + E_joule) / eta_in. The rejected heat loads the dry
  coolers (fan power and capacity).
- Capex: a "heater leads" line; the lead Joule power is added to the heater supply and to the grid
  interconnect sizing; heat-rejection capacity covers the leads.
- The full-cycle ledger has a "heater leads" group (Joule heat, net heat from storage).
- With no lead table, none of this applies and the element runs through the roof as before.

### 7.10 Results for the pilot optimum (run 20261125_001641_p87_pilot)

From the optimizer (lead table):

| Item | Value |
|---|---|
| Leads | 2 per heater, 14 in the plant, 6.11 kA each |
| Per lead, lifetime | $47,420 = capex $9,635 + energy $37,785 (80 % energy) |
| Per lead, charging | Joule 7.4 kW; -3.1 kW from storage (heat flows into the block); 4.1 kW rejected |
| Per lead, heater off | +0.45 kW drawn from storage; 0.51 kW rejected |
| Plant energy per cycle | Joule 5.1 MWh - 1.4 MWh returned to storage = **3.8 MWh lost (2.5 % of AC bought)** |
| Effect on net RTE | **27.00 % with leads vs 27.82 % without** |
| Plant lifetime cost | about $0.7M; initial capex line about $0.1M (0.2 % of capex) |

Full lead optimization of the same design (Figure "heater_lead_section" in the run report):

| Zone (hot -> cold) | x [m] | Body dia [mm] | Current density [A/mm2] | T max / limit [C] |
|---|---|---|---|---|
| graphite | 0.00-0.70 | 384 | 0.05 | 2776 / 3000 |
| tungsten | 0.70-0.79 | 37 | 5.78 | 2372 / 2500 |
| copper, water-cooled | 0.79-1.40 | 71 | 1.55 | 40 / 200 |

- The graphite-tungsten joint runs at **2500 C, exactly the tungsten limit**, while charging. The
  optimum rides on that constraint.
- Lead resistance 199 uOhm: 1.2 V drop, 2.0 % of the 60 V supply.
- Losses: 1.19 % of heater power while charging, 0.12 % while the heater is off; 271 kWh per lead
  per cycle = 1.48 % of the heater's charge energy.
- **Table vs full optimization:** lifetime cost -1.3 %, capex -4.2 %, energy -0.5 %, energy lost per
  cycle -0.4 %. The table is good enough for the optimizer.

### 7.11 Review items specific to the leads (doc items 37-40)
- Not modelled: contact resistance at joints, the lead above the insulation, the disturbance of
  the Thermax by the lead, and the **tungsten-graphite reaction (W2C / WC above about 1400 C) at a
  joint that runs at 2500 C**. The effect on joint design is possibly large; on cost, probably small.
- Charging assumes rated current for the whole charge. Once the heater hits its temperature limit
  the current falls, so the Joule heat is overstated (conservative).
- The table uses a guessed block mean for the heater-off state; the lead-detail check measures the
  error (1.3 % here). Rebuild the table if it exceeds a few percent.
- Lead material, joint and disconnect-mechanism costs are placeholders; the idle disconnect has no
  cost or reliability penalty.

## 8. Tank and plant (doc section 5)

- **Sizing:** N sticks = ceil(P_rated / P_stick), with P_stick the discharge-average AC power.
  Heaters = sticks.
- **Hot zone:** checkerboard pitch from the cell area; minimum graphite wall between cavities
  0.3 m (constraint).
- **Insulation losses:** steady 1-D conduction with temperature-dependent k (exact Kirchhoff
  integral) through side wall, roof (plus argon headspace and deck), edges and the bottom stack
  (columns in parallel with Thermax).
  - Roof and floor losses act on every cell, so they are applied as a heat sink inside the cell
    transients.
  - Side and edge losses reach only perimeter cells, so they are spread over all sticks after the
    run.
- **Support columns:** graded graphite (width follows the creep allowable along the temperature
  profile: 2.81 MPa at 2400 C for 1 % strain in 30 yr) on oxide posts. A 3-D correction f_3D = 0.85
  is calibrated to an FEA of the v3 design; the floor hot spot under the column foot uses 1.21x the
  1-D heat.
- **Floor cooling:** forced air at 10 m/s through steel fins under the floor plate. Floor maximum
  and column hot spot must stay at or below 120 C (design limit, below API 650 Annex M's 200 C to
  keep margin for a fan failure).
- **Foundation:** pressure of everything over the tank footprint must stay at or below 225 kPa.
- **Costs:** about 18 quantity x rate lines (steel, Thermax, column graphite, oxide, floor, slab,
  cellular glass, mat, fans); all rates are placeholders, with low / central / high values.
- **Pilot optimum:** hot zone 11.3 m, tank 14.2 x 22.1 m (below the usual API 650 range of 15-80 m
  diameter), floor max 46 C, column hot spot 99 C, foundation 119 kPa, insulation losses 2.78 % of
  stored heat per day.

## 9. Transient simulation (doc section 6)

- **Discharge loop** (stick cell), each step:
  1. radiation and vane control at the start-of-step wall temperatures;
  2. electrical power per ring;
  3. choose the time step from the vane state;
  4. roof / floor sink;
  5. conduction update with the implicit wall coupling;
  6. test the cutoff.
- **Time steps:** 600 s at first and while the vanes are closed, 900 s while they regulate, 1800 s
  when fully open. Screening doubles these. There is no error control.
- **Three phases of a discharge:** (1) hot block, vanes closed, cells over the target flux;
  (2) vanes regulating, constant power at target; (3) vanes open, power falls. The stick is
  withdrawn when the volume-average block temperature reaches T_min.
- **Charge loop** (heater cell): fixed 600 s step, stops when the volume-average temperature reaches
  T_max.
- **Shade-open sizing loop:** with T_max fixed, iterate T_min and the block volume until the vanes
  reach fully open exactly when the block reaches T_min, at exactly the target duration. This sets
  the swing and the block size. It typically takes 3-4 discharge runs; the pilot optimum converged
  in 4 runs with volume factor 1.017.

## 10. Material properties (doc section 7)

- **Block graphite:** constants rho 1450 kg/m3, c_p 2000 J/kg-K, k 20 W/m-K, with no source. The code
  holds unused tables that disagree (c_p 2130, k 22.8 at 2150 C).
- **Graphite emissivity:** linear fit 0.679 + 6e-5 T, unclipped.
- **Insulation:** graphite felt (vanes, pad) table to 2500 C; Thermax k = 0.0492 exp(0.0011 T[C])
  W/m-K.
- **TPV efficiency:** quadratic in emitter temperature for 1.24 / 1.38 / 1.55 um bandgaps. The
  1.24 um fit peaks at 48.5 % at 1845 C. No source and no clamp.
- **Heater graphite resistivity:** Mersen 2020 data, 16.5 uOhm m at room temperature, minimum near
  950 C, 11.4 at 2002 C, extrapolated to 12.2 at 2600 C.
- **Lead materials:** see section 7.7.

## 11. Optimizer (doc section 9)

### 11.1 Objective
- f = (1 + soft costs) x lifetime capex + price x (electricity bought - net electricity sold) x
  cycles per year x lifetime, plus a penalty for constraint violations.
- Soft costs 55 %, $87/MWh, 52.1 cycles/yr, 30 years.

### 11.2 Design variables (10)

| Variable | Range |
|---|---|
| Liner side | 0.2-0.6 m |
| Powerstick side | fraction of the room in the liner |
| Stick length L | 3-6 m |
| Block height | L + 0.25 m up to 14 m |
| Cell flux target | 5-20 W/cm2 |
| Shade thickness | 1/4-6 in (integer eighths) |
| Side / top insulation | 0.5-3 m each |
| Bottom stack | 2-4 m |
| Block T_max | 1800-2500 C |

T_min, the swing, the block volume, heater power and heater cavity are sizing outputs, not
variables.

### 11.3 Constraints
- flux reachable;
- T_min at or above 1400 C;
- swing 200-1000 C;
- charge in 50 h;
- 0.3 m minimum graphite wall;
- energy target met;
- tank at most 80 m diameter and 30 m tall;
- floor and column hot spot at or below 120 C;
- foundation at or below 225 kPa;
- charge + discharge fit in 168 h.

### 11.4 One design evaluation (Figure 9.1, about 1-2 min on one core)
1. Shade-open sizing.
2. Heater search.
3. Graphite wall check.
4. Plant and tank (two passes).
5. Lifetime-average wear (clean run averaged with one run at twice the life-mean wear).
6. Energy per cycle including the heater leads.
7. RTE, capex and objective.

### 11.5 Capex lines
- tank;
- storage graphite;
- powersticks (cells + stick);
- liners;
- shades;
- heaters (fixed + element graphite + supply);
- heater leads;
- inverters;
- grid interconnect (on max of the charging draw incl. lead Joule power, and the peak output);
- heat rejection (on the peak per stick slot: stick discharge + idle leads, or leads while
  charging).

All lines are multiplied by (1 + soft costs).

### 11.6 Search
- scipy differential evolution, 20 workers, about 90-100 designs per generation, 25 generations.
- A full run is about 3,300 designs. The pilot run took 82 CPU-hours: 3,328 designs, 2,210
  feasible.
- Screening fidelity noise is a few tenths of a percent, comparable to the spread among the best
  designs. The top designs are re-checked at detailed fidelity (verify cell).

## 12. Latest results (pilot run, $87/MWh)

### 12.1 Optimum

| Item | Value |
|---|---|
| Plant | 1 MW x 50 h; 7 powersticks + 7 heaters |
| Powerstick / liner | 169 mm square x 5.96 m (at the 6 m upper bound) / 369 mm |
| Flux target | 9.9 W/cm2 |
| Shade | 2.75 in vanes |
| Block | 8.08 m tall, 1789-2247 C (swing 458 C) |
| Insulation | side 1.45 m, top 1.40 m, bottom stack 3.23 m |
| Heater | 366 kW each, 688 mm cavity; grid interconnect 2.8 MW |
| Tank | 14.2 m dia x 22.1 m |
| Cycle | charge 49.7 h, hold 68.8 h, discharge 50.5 h (clean) |
| Per cycle | 157 MWh bought (14 MWh of it hold losses), 48 MWh sold gross, 42 MWh net of fans |
| **Net RTE (life average)** | **27.0 %** (30.5 % before parasitic loads; 27.8 % without lead losses) |

### 12.2 RTE chain
- RTE = delivery 71.6 % x cell efficiency 44.9 % x output 84.1 %.
  - Delivery is the heat absorbed by the active cells / AC bought; it carries the heater supply,
    insulation, bare and inactive-area absorption and lead losses.
  - Output is net AC / DC; it carries the inverter, floor fans and heat-rejection fans.
- Across all 2,210 feasible designs RTE ranges from 20.8 to 29.8 % (median 26.6 %). The 10 best
  by objective sit within 26.1-27.1 %.

### 12.3 Costs

| Item | Value |
|---|---|
| Objective | $42.7M = lifetime capex $27.2M + energy lost $15.6M |
| Initial capex | $26.4M (incl. 55 % soft costs) |

Initial capex breakdown:

| Line | Cost | Share |
|---|---|---|
| Powersticks | $15.3M | 58 % |
| Tank | $5.8M | 22 % |
| Heat rejection | $2.1M | 8 % |
| Grid interconnect | $1.6M | 6 % |
| Storage graphite | $1.1M | 4 % |
| Shades | $0.4M | 1.4 % |
| Heaters | $0.1M | 0.4 % |
| Heater leads | $0.1M | 0.2 % |
| Liners and inverters | about $0 | |

Cells are $36.5/cm2 in the Pilot scenario, which pushes the optimum toward high flux. The report
suggests considering a higher flux bound.

### 12.4 Full-cycle check (detailed settings, clean cells)
`cycle.run_cycle` re-runs the optimum through a full charge-hold-discharge week at detailed settings
(finer rings, more rays, full timestep) with clean cells, including the heater leads. The result is
**29.0 % net RTE** (32.4 % before parasitic loads). That is higher than the 27.0 % life average
because the cells are fresh.

| Quantity | Value |
|---|---|
| AC in / net AC out | 157 / 45 MWh per cycle |
| Cycle | charge 49.7 h + hold 68.0 h + discharge 50.2 h |
| Cell efficiency (DC / absorbed) | 46.5 % |
| Delivery / output | 73.3 % / 85.0 % |
| Liner inner face during discharge | 1653-1689 C (TPV fit peaks at 1845 C, 48.5 %) |

Where the electricity bought goes (% of AC in; the slide-ready loss waterfall is the report's
"cycle_losses" figure):

| Group | Item | MWh / cycle | % of AC in |
|---|---|---|---|
| Power electronics | heater supply (1 - eta_in) | 7.8 | 5.0 % |
| Heater leads | Joule heat (charging) | 5.2 | 3.3 % |
| | heat returned to the storage (net) | -1.4 | -0.9 % |
| Insulation: charge | roof + floor | 2.4 | 1.6 % |
| | side + edges | 4.7 | 3.0 % |
| Insulation: hold | side / edges / roof / floor | 7.3 / 1.3 / 2.3 / 1.9 | 8.1 % in total |
| Insulation: discharge | roof + floor | 2.4 | 1.5 % |
| | side + edges | 4.8 | 3.1 % |
| Stick | absorbed on bare + inactive area | 3.1 | 2.0 % |
| | cell conversion loss | 61.5 | 39.2 % |
| Power electronics | inverter (1 - eta_out) | 2.7 | 1.7 % |
| Parasitics | floor-cooling fans / heat-rejection fans + pumps | 1.8 / 3.5 | 3.4 % in total |
| **Output** | **net AC out** | **45.4** | **29.0 %** |

Takeaways from the ledger:
- Insulation losses over the whole week total about 17 % of the AC bought (27 MWh), the largest
  loss after cell conversion. The hold alone is about 8 %. At this 1 MW pilot scale the tank's surface-to-volume
  ratio is high.
- The heater leads lose about 2.4 % net.

Detailed-settings check of the top 3 designs: RTE within 0.1 point of screening, and all three
remain feasible. Screening ranks the designs reliably.

## 13. Items to review (doc section 11): 74 items, prioritised for a slide

The documentation flags 74 items with confidence and estimated effect. Grouped and ranked for a
slide (doc item numbers in brackets):

**Likely to move results by percent-level or more**
- TPV efficiency fit: unsourced and unclamped; ignores cell temperature and spectrum; multiplies
  out-of-band absorption too. It sets every electrical output [28]. Docstring and code disagree on
  the flux derate [27].
- Block graphite properties: unsourced constants (density 1450 vs 1743 kg/m3 is a 20 % volume
  difference) [69].
- Charge and discharge are simulated from uniform temperatures in separate cells; the cycle is not
  thermally closed. Effect on T_min possibly tens of K [5, 58].
- Closed vanes cannot hold the flux setpoint early in the discharge (1.1-2.5x the target), so the
  cells are over-driven while the block is hot [19].
- Roof penetration and door losses are not modelled (sticks, shade axles, doors). Possibly tens of %
  of the roof loss [45]. The heater leads are now modelled.
- Tungsten-graphite reaction at the lead joint, which runs at 2500 C [37].

**Numerical / bookkeeping (known, mostly small)**
- The step crossing the T_min cutoff is dropped from stick energy (0.5-2 % low) [63]. The shade
  opening is detected one step late [65]. Both fixes are agreed and deferred with a speed-up plan.
- No time-step error control; screening at 2x steps [61, 74].
- Radiation energy recorded at the start of the step vs heat actually removed [10, 59].

**Plant / structure**
- Columns are checked for creep only: no buckling, oxide creep, seismic or thermal expansion [51].
- f_3D calibrated to one FEA point [50].
- Foundation load simplifications [55].
- All tank costs are placeholders [57].

**Deposition**
- Rate evaluated at the mean liner temperature underestimates it, because the rate is strongly
  Arrhenius [32].
- x3.5 factor, no pressure dependence [33].
- Carbon vapour pressure correlation about 3x the handbook value (conservative) [34].

## 14. Figures to reuse

From `docs/tinless_model_documentation.docx` (after rebuilding with `opt_docs`):
- Figure 1.1 model hierarchy;
- 3.2 stick-cell plan section;
- 3.3 r-z mesh;
- 4.2.1 radiation and vane control flowchart;
- 4.3.1 vane positions;
- 4.3.2 control window;
- 4.6.1 heater power vs wall temperature;
- 4.8.1 heater and lead sizing flowchart;
- 4.8.2 example lead section / temperature / heat;
- 5.1 checkerboard plan;
- 5.2 tank half-section with heat paths;
- 6.1 how the transients fit together;
- 6.4 shade-open sizing loop;
- 9.1 one design evaluation;
- section 8 cycle plots and loss ledger (only when built with the example design).

From the run report `opt_runs/20261125_001641_p87_pilot/report.docx`:
- layout to scale;
- cycle time series;
- loss waterfall;
- RTE vs cell efficiency;
- cell efficiency vs liner temperature;
- **heater_lead_section** (lead to scale, temperatures, heat generation);
- scatter plots of the run.

## 15. Suggested slide outline

1. **Title:** Tinless thermal battery system model: what it does and what it says (date,
   branch).
2. **Key takeaways:** the 10 points of section 1, cut to 5-6 bullets.
3. **The system:** concept sketch (stick, liner, vanes, heater, block, tank); weekly cycle.
4. **Model architecture:** cell -> plant -> cycle -> optimizer (Figure 1.1); what each level adds.
5. **Discharge physics:** ray-traced cavity with rotating vanes, liner, two-band cells; the three
   discharge phases; shade control window (Figures 4.2.1, 4.3.2).
6. **Charge physics and heater sizing:** phase 1 / phase 2 heater, smallest-power search, tube
   element (Figure 4.6.1, element numbers).
7. **NEW: heater current leads, model:** geometry, 1-D thermal model, two states, idle
   disconnect, lifetime cost (Figure 4.8.2).
8. **NEW: heater current leads, results:** pilot optimum zones table, 0.8-point RTE cost,
   3.8 MWh/cycle, cost is 80 % energy, table accurate to 1.3 %, joint at the tungsten limit
   (heater_lead_section figure).
9. **NEW: lead table and plant integration:** why per-ampere works, grid, how it enters capex,
   energy, RTE and objective; the voltage lever.
10. **Plant model:** tank, insulation paths, columns, floor cooling, foundation, constraints
    (Figures 5.1, 5.2).
11. **Sizing and transient:** shade-open sizing loop (Figure 6.4).
12. **Optimizer:** objective, 10 variables, constraints, evaluation pipeline (Figure 9.1), run
    size.
13. **Pilot results, design:** optimum table, layout figure.
14. **Pilot results, energy and cost:** RTE chain, ledger, capex breakdown.
15. **Open issues:** prioritised review items (section 13 of this file).
16. **Next steps:**
    - regenerate the model documentation (`opt_docs`) so it carries section 4.8 and this run's
      example design;
    - deferred speed-ups (cutoff interpolation, shade-lag fix, uniform step, view-factor cache,
      about 2x faster);
    - heater voltage / current study for the leads;
    - lead joint materials (W-graphite reaction);
    - penetration and door losses;
    - sourced graphite and TPV data.

## 16. Glossary

| Term | Meaning |
|---|---|
| Powerstick / stick | Water-cooled square tube covered in TPV cells, inserted only during discharge |
| Liner | Thin graphite square tube around the stick; its inner face is the emitter the cells see |
| Shade / vanes | 8 rotating graphite-felt vanes that throttle radiation to hold the flux target |
| Flux target | Heat absorbed by the active cells per cm2, held constant by the vanes |
| Shade-open sizing | Block sizing so the vanes reach fully open exactly at T_min at the end of the target duration |
| Swing | T_max - T_min of the block |
| RTE | Round-trip efficiency: net AC sold / AC bought per cycle |
| Delivery / output | Absorbed cell heat / AC bought; net AC / DC |
| Lead | Current conductor from the heater element up through the roof insulation to the deck |
| Lead table | Precomputed optimal leads (per ampere) on a t_top x T_max x current grid |
| Idle disconnect | Graphite-tungsten lead joint opened while the heater is off |
| Screening / detailed | Optimizer fidelity levels (coarse for the search, fine for verification) |
| D (wear) | Deposit / 10 nm; cells replaced at 0.1, bare areas cleaned at 0.01 |
| Thermax | Tank insulation board (side, roof, bottom stack) |
| RGF | Rigid graphite felt (vanes, liner-bottom pad) |
| f_3D, f_foot | Column heat-spreading factors (total heat, foot hot spot) |
