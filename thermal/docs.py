"""Model documentation generator: a Word document describing every part of the tinless
system model (inputs, geometry, component physics, material properties, transient
simulation, plant / cycle accounting and the optimizer), with live tables of the current
inputs and code constants, flowcharts, property plots and example charge / discharge runs.

    from thermal.docs import build_docs
    build_docs('docs/tinless_model_documentation.docx', base_cfg, opt, example_dir='opt_runs/<run>')

Prose lives in the section functions below (thermal/docs_sections.py holds the component
physics); numbers in tables are read from the code / YAML when the document is built.
"""
import datetime as _dt
import os

import numpy as np

from .docs_base import Doc, flowchart, yaml_with_comments, _fmt, DIM
from .report import DESC as OPT_DESC, _flatten, _git, VAR_PHYS

C2K = 273.15
INCH = 0.0254


# --------------------------------------------------------------------------
# Example design (best of an optimizer run, detailed settings)
# --------------------------------------------------------------------------
def run_example(base_cfg, opt, log_dir, verbose=False):
    """Best feasible design of the run in log_dir, re-run as a full cycle at the optimizer's
    DETAILED fidelity, sized the same way. -> (row, cycle result)"""
    from . import optimize as O
    from .cycle import run_cycle
    rows = O.load_log(log_dir)
    ok = [r for r in rows if r.get('feasible')]
    if not ok:
        raise ValueError(f"no feasible design in {log_dir}")
    top = min(ok, key=lambda r: r['lifetime_cost'])
    o = O.merge(opt, {'fidelity': O.DETAILED})
    d = dict(top['design'], T_min_guess_C=top['T_min_C'])
    cfg = O.design_cfg(base_cfg, d, o)
    sz = o['sizing']
    cy = run_cycle(cfg, opt=o, cycle_hr=8760.0 / o['economics']['cycles_per_year'], charge_hr=o['charge_hr'],
                   resize_block=True, size_heater_cavity=True, verbose=verbose,
                   resize_kw=dict(tol_T=sz['tol_T'], tol_t=sz['tol_t'], max_iter=sz['max_iter'],
                                  T_min_floor=o['geometry']['T_min_floor_C'] + C2K))
    return top, cy


def latest_run(root='opt_runs'):
    """Most recent run folder under root that has a feasible design."""
    from . import optimize as O
    if not os.path.isdir(root):
        return None
    for d in sorted(os.listdir(root), reverse=True):
        p = os.path.join(root, d)
        if os.path.isdir(p) and any(r.get('feasible') for r in O.load_log(p)):
            return p
    return None


# --------------------------------------------------------------------------
# Shared tables
# --------------------------------------------------------------------------
def yaml_table(doc, yaml_path, prefix=None, exclude=(), size=8):
    from .report import SET_BY_OPTIMIZER
    rows = []
    for k, v, c in yaml_with_comments(yaml_path):
        if prefix and not (k == prefix or k.startswith(prefix + '.')):
            continue
        if any(k == e or k.startswith(e + '.') for e in exclude):
            continue
        if v == '' and not c:
            continue
        note = 'replaced per design by the optimizer' if k in SET_BY_OPTIMIZER else ''
        rows.append([k, v, c, (note, {'color': DIM, 'italic': True}) if note else ''])
    doc.table(['YAML key', 'value', 'meaning (YAML comment)', ''], rows, widths=[1.9, 0.9, 3.3, 1.0], size=size)


def opt_inputs_table(doc, opt, groups=None, size=8):
    flat = _flatten({k: v for k, v in opt.items() if not k.startswith('_') and k != 'design_vars'})
    rows = []
    for k, v in flat.items():
        g = k.split('.')[0] if '.' in k else 'plant'
        if groups and g not in groups:
            continue
        unit, sc, desc = OPT_DESC.get(k, ('', 1, ''))
        vv = v * sc if isinstance(v, (int, float)) and not isinstance(v, bool) and sc != 1 else v
        est = 'placeholder' in desc.lower()
        rows.append([k, _fmt(vv), unit, (desc.replace(' (placeholder)', '') + (' [estimate]' if est else ''),
                                         {'color': 'B5520A'} if est else {})])
    doc.table(['input', 'value', 'unit', 'description'], rows, widths=[2.4, 0.8, 0.7, 3.2], size=size)


# --------------------------------------------------------------------------
# Optimizer
# --------------------------------------------------------------------------
def sec_optimizer(doc, opt):
    from . import optimize as O
    doc.h1('9. The optimizer')
    doc.p("The optimizer (thermal/optimize.py, driven by tinless system optimizer.ipynb) searches the design "
          "variables below for the plant with the lowest lifetime cost. Each candidate is a complete plant: the cell "
          "model is sized, run, scaled to a plant in a tank, costed and scored. Every evaluation is logged (one JSON "
          "line per design) so runs can be reloaded and reported.")
    doc.h2('9.1 Objective')
    doc.p("One electricity price is used for buying and selling, so lost energy is valued in dollars and traded "
          "directly against capital cost:")
    doc.eq(r'\mathrm{minimise}\;\; C_{life} = (1+f_{soft})\,(C_0 + n_{rep}C_{cells} + n_{clean}C_{clean})'
           r' + p_e\,(E_{AC,in} - E_{AC,net})\,N_{cyc/yr}\,Y',
           where='first term: lifetime capex; second: value of the electricity lost over the life')
    doc.p(f"with f_soft = {opt['economics']['soft_cost_frac']}, p_e = ${opt['economics']['elec_price']:.0f}/MWh, "
          f"N_cyc/yr = {opt['economics']['cycles_per_year']:.2f} (one-week cycle) and Y = "
          f"{opt['economics']['lifetime_yr']:.0f} years. C_0 is the sum of the initial capex lines, E_AC,in the "
          "electricity bought per cycle (charge + hold) and E_AC,net the electricity sold per cycle net of fans, "
          "both at life-average cell condition. Capex is reported without the electricity term. An optional floor "
          f"RTE >= rte_min ({opt['objective']['rte_min']}) can be imposed.")
    doc.p("Differential evolution minimises f:")
    doc.eq(r'f = C_{life}/10^6\;\;\mathrm{(feasible)},\qquad f = ' + f"{O.INFEASIBLE:g}" +
           r' + 10^3\sum_i v_i\;\;\mathrm{(infeasible)},\qquad f = ' + f"{O.FAILED:g}" + r'\;\;\mathrm{(model\ error)}',
           where='v_i: normalised constraint violations (table below).')
    doc.review("The penalty is a large constant plus a slope, so infeasible designs are ranked among themselves by "
               "total violation but carry no information about cost; DE recovers by replacing them. A design that "
               "is barely infeasible is never compared with feasible ones on cost.")

    doc.h2('9.2 Optimized variables and bounds')
    doc.p("The variables are re-parameterised so that every point in the search box is a buildable geometry "
          "(the powerstick side is a fraction of the room left in the liner; the block height a fraction between "
          "the stick length and H_max). Shade thickness is an integer number of 1/8 in.")
    rows = []
    for name, lo, hi, integer, desc in O.design_vars(opt):
        rows.append([name, f"{lo:g}", f"{hi:g}", 'yes' if integer else 'no', desc])
    doc.table(['variable', 'lower', 'upper', 'integer', 'meaning'], rows, widths=[0.9, 0.6, 0.6, 0.6, 4.4])
    g = opt['geometry']
    doc.p("Decoding (optimize.decode):")
    doc.eq(r's_{ps} = 0.10 + \phi_{ps}\,\left(\min(0.40,\; s_{liner} - 2g_{ps}) - 0.10\right),\qquad '
           r'H = L + \Delta_0 + \phi_H\,(H_{max} - L - \Delta_0)')
    doc.p(f"with g_ps = {g['ps_liner_gap']} m (minimum stick-to-liner clearance), H_max = {g['H_max']} m and "
          f"Delta_0 = min_lower_zone = {g.get('min_lower_zone', 0.25)} m (there is always a lower zone: a floor below "
          f"the stick, the liner bottom and pad, and radiation through the lower zone). "
          f"T_max is capped at {g['T_cap_C']:.0f} C. T_min and the swing are not variables: they come out of the "
          "shade-open sizing (section 6.5).")

    doc.h2('9.3 Evaluation of one design')
    doc.p("optimize.evaluate_design runs these steps; any failed step returns the design as infeasible with the "
          "violation that stopped it.")
    fig = flowchart({
        'd': (3.6, 9.4, 'Design vector x  ->  decode  ->  model config\n(YAML + design + fidelity settings)', 'io'),
        's': (3.6, 8.2, '1. Shade-open sizing (discharge runs, clean cells)\nT_min, block volume -> shades fully open '
                        'at duration', 'loop'),
        'c1': (3.6, 7.0, 'T_min >= floor?  swing within limits?\nflux target reachable?', 'dec'),
        'h': (3.6, 5.9, '2. Heater search (charge runs)\nsmallest power at min cavity; grow cavity if capped', 'loop'),
        'w': (3.6, 4.85, 'graphite walls >= min_wall?', 'dec'),
        't': (3.6, 3.85, '3. Tank: pass 1 (side/edge share) -> pass 2 sized on AC stick\nN sticks, tank, losses, '
                         'floor, columns, costs', 'proc'),
        'g': (3.6, 2.75, '4. Discharge at 2x life-mean wear (no deposition)', 'proc'),
        'e': (3.6, 1.75, '5. Energy per cycle (+ heater leads), RTE, capex,\nmaintenance, energy lost, constraints', 'proc'),
        'f': (3.6, 0.6, 'f  (logged with all metrics)', 'out'),
        'x': (6.4, 5.4, 'infeasible:\nf = 1e4 + 1e3 sum(v)', 'out'),
    }, [('d', 's'), ('s', 'c1'), ('c1', 'h', 'ok'), ('h', 'w', 'ok'), ('w', 't', 'ok'), ('t', 'g'), ('g', 'e'),
        ('e', 'f'), ('c1', 'x', 'no'), ('h', 'x', 'not met'), ('w', 'x', 'no')], size=(7.2, 10.0))
    doc.fig(fig, 'Steps of one design evaluation (optimize.evaluate_design).', width=5.6, name='Figure 9.1')
    doc.h3('Constraints')
    lim = opt['limits']
    doc.table(['constraint', 'value', 'normalised violation'], [
        ['target cell flux reachable with the shades fully open', '-', '1 (FluxUnreachable)'],
        ['T_min >= T_min_floor', f"{g['T_min_floor_C']:.0f} C", '(floor - T) / 100'],
        ['swing_min <= swing <= swing_max', f"{g['swing_min']:.0f} - {g['swing_max']:.0f} C", 'excess / 100'],
        ['charge in charge_hr within the heater power cap', f"{opt['charge_hr']:g} h", 't_charge / target - 1'],
        ['graphite wall between any two cavities', f"{g['min_wall']} m", '(min_wall - wall) / min_wall'],
        ['plant energy >= P_rated x duration (clean)', f"within {opt['sizing']['tol_t']:.0%}", '-mismatch'],
        ['tank diameter / height', f"{lim['D_tank_max']:g} / {lim['H_tank_max']:g} m", 'value / limit - 1'],
        ['floor max / column hot spot <= floor design limit', 'tank.floor.T_limit_design (else Annex M)', 'T / limit - 1'],
        ['foundation pressure', 'tank.soil_allowable', 'p / allowable - 1'],
        ['charge + discharge fit in the cycle', f"{8760 / opt['economics']['cycles_per_year']:.0f} h", 'excess'],
        ['net RTE >= rte_min (optional)', f"{opt['objective']['rte_min']}", '1 - RTE / rte_min'],
    ], widths=[3.4, 1.4, 2.2])

    doc.h2('9.4 Energy accounting per cycle')
    doc.p("All terms per cycle for the plant of N sticks (N heaters). E_rel is the heat released from the block "
          "during a discharge (heat into the stick + roof/floor losses while the stick is in), averaged over clean "
          "and worn cells; E_sink,ch the roof/floor loss of one cell during its charge; Q_side, Q_edge the plant "
          "side and edge losses at the mid temperature; Q_loss(T_max) the total loss while holding at T_max:")
    doc.eq(r'E_{AC,in} = \frac{N\,(E_{rel} + E_{sink,ch}) + (Q_{side}+Q_{edge})\,t_{ch} + Q_{loss}(T_{max})\,t_{hold}'
           r' + E_{drain} + E_J}{\eta_{in}}',
           where='E_J, E_drain: heater-lead Joule heat while charging and net heat drawn from the storage (section 4.8; '
                 'zero when no lead table is used).')
    doc.eq(r'E_{AC,out} = N\,\eta_{out}\,(1-\pi_{se})\,\frac{E_{e,clean} + E_{e,worn}}{2},\qquad '
           r'E_{AC,net} = E_{AC,out} - P_{floor\,fan}\,t_{cycle} - c_{rej}\,(N\,\overline{E_{rej}} + E_{rej,lead})')
    doc.eq(r'\mathrm{RTE_{net}} = E_{AC,net}/E_{AC,in},\qquad t_{hold} = t_{cycle} - t_{ch} - t_{dis}')
    doc.p(f"eta_in = eta_out = {opt['economics']['eta_in']}; pi_se is the fraction of the stick's energy lost to "
          f"side + edge losses (tank.stick_correction); c_rej = {opt['parasitics']['heat_rejection_kWe_per_kWth']} "
          "kWe per kWth of stick heat rejection (dry coolers); floor fans run the whole cycle. The charge buys back "
          "the heat the discharge released (steady cycle), not a uniform T_min -> T_max swing, because the stick is "
          "removed when the volume-average block temperature reaches T_min with a non-uniform field.")
    doc.h3('Degradation over the life')
    doc.p("Deposition per cycle d (nm) comes from the clean discharge. With D = deposit / d_dead, cells are replaced "
          f"at D_c = {opt['deposition']['cell_replace_D']} and bare areas cleaned at D_b = "
          f"{opt['deposition']['bare_clean_D']} (also reset at cell replacement). Deposit grows linearly with cycles, "
          "so D_c and D_b are saw-tooth functions of time. Performance is taken as linear in D, and the life average "
          "is the clean run averaged with one run at twice the life-mean D's:")
    doc.eq(r'\bar{X} = \frac{X(0,0) + X(2\bar{D}_c,\,2\bar{D}_b)}{2},\qquad '
           r'\bar{D} = \frac{1}{d_{life}}\int_0^{d_{life}} D(x)\,dx')
    doc.p("The number of replacements and cleanings over the life is counted from the same saw-tooth; replacement "
          "cost = cells of all sticks; cleaning cost per stick (bare_clean_per_stick).")
    doc.review("Performance is assumed linear in D for the averaging. Cell efficiency scales linearly with D, but "
               "the emissivity rise of bare/inactive areas changes the heat balance and the discharge length "
               "non-linearly; the error is small for small D but has not been checked at D ~ 0.1.")

    doc.h2('9.5 Capital cost model')
    doc.p("Initial capex lines (all x (1 + soft costs)):")
    doc.bl([
        'tank: containment, insulation, supports, floor and foundation (section 5.8, central values of the YAML cost '
        'triples, replaced by the optimizer inputs tank_costs and costs.iso_graphite_per_kg where given);',
        'storage graphite: graphite mass x storage_graphite_per_kg;',
        'powersticks: cells (cell_per_cm2 x 4 s L (1 - f_bare)) + stick_per_m x L + stick_fixed, per stick;',
        'liners: walls 4 s L x liner_wall_per_m2 + floor liner_floor_area_factor s^2 x liner_floor_per_m2;',
        'shades: 8 RGF vanes (l x t x H) x shade_rgf_per_m3 + 8 axles (t_top + axle_extra) x shade_axle_per_m + '
        'drive (one motor + linkage per stick, or one motor per vane);',
        'heaters, per heater: heater_fixed + element graphite A_cross x H_block (x (H_block + t_top) when no lead '
        'table is used) x heater.density x heater_graphite_per_kg (default the iso-molded graphite price) + '
        'charge_electrics_per_kW x (heater power + lead Joule power);',
        'heater leads (with a lead table): leads per heater x N x (lead material + joints) from the lead table '
        '(section 4.8);',
        'inverters: inverter_per_kW x max(P_rated, peak plant AC output);',
        "grid interconnect: interconnect_per_kW x max(all heaters + their leads' Joule power at full power / "
        "eta_in, peak AC output);",
        'heat rejection: heat_rejection_per_kWth x N x peak heat rejection per stick slot (stick discharge peak + '
        'idle leads, or the leads while charging, whichever is larger).',
    ])
    doc.h3('Cost and economic inputs')
    opt_inputs_table(doc, opt, groups=('costs', 'economics', 'tank_costs'))

    doc.h2('9.6 Differential evolution')
    doc.p("scipy.optimize.differential_evolution with: strategy 'best1bin' (scipy default), mutation factor "
          "dithered in [0.5, 1.0) each generation, crossover probability 0.7, Sobol initial population, deferred "
          "updating (the whole generation is evaluated in parallel, then the population is updated), integer handling "
          "for the shade thickness (integrality), no polishing, the YAML design as one initial member (x0). Population "
          "= popsize x " + str(len(O.design_vars(opt))) + " variables; stop after maxiter generations or when the population's objective spread "
          "std(f) <= tol x |mean(f)| (tol = 1e-3). Designs are evaluated in a process pool (one thread per worker).")
    doc.p("Typical settings in the notebook: WORKERS 20, POPSIZE 10 (90 designs per generation), MAXITER 25, SEED 0; "
          "one design takes about 1-2 min on one core (screening settings), so a run is roughly 2-4 hours.")
    doc.review("Screening fidelity (0.5 m rings, 2^12 vane rays, 2x timestep, 3% heater-cavity tolerance, 3 K / 1% "
               "sizing tolerances) gives numerical noise of a few tenths of a percent in the objective, comparable to "
               "the spread among the best designs; the top designs should be re-checked at detailed settings "
               "(verify cell) and their differences treated as within noise.")
    doc.h3('Fixed (non-optimized) inputs of the optimizer')
    opt_inputs_table(doc, opt, groups=('plant', 'objective', 'cells', 'parasitics', 'deposition', 'flux_band_tol', 'sizing',
                                       'geometry', 'limits', 'fidelity'))
    doc.p(f"Detailed settings used for verification: {O.DETAILED}.")


# --------------------------------------------------------------------------
# Cycle / plant accounting
# --------------------------------------------------------------------------
def sec_cycle(doc, cy, top, opt):
    from .cycle import plot_layout, plot_timeseries, plot_losses, J_MWh
    doc.h1('8. Plant cycle and example design')
    doc.p("thermal/cycle.py runs one full charge - hold - discharge cycle for a design and splits the electricity "
          "bought into where it ends up (the ledger). The optimizer uses the same accounting (section 9.4). This "
          "section shows it for the example design: the best design of the optimizer run "
          f"(objective rank 1), re-run at detailed settings with clean cells.")
    d = top['design']
    doc.kv([
        ('Liner / powerstick', f"{d['liner_side'] * 1e3:.0f} mm / {d['ps_side'] * 1e3:.0f} mm square, stick "
                               f"length {d['L']:.2f} m"),
        ('Block', f"height {cy.m.H_block:.2f} m, r {cy.m.r_in:.3f} - {cy.m.r_out:.3f} m, "
                  f"{cy.T_min_C:.0f} - {cy.T_max_C:.0f} C"),
        ('Shade', f"{d['shade_thickness'] / INCH:.3f} in"),
        ('Flux target', f"{d['avg_flux']:.2f} W/cm2"),
        ('Heater', f"{cy.hm.heater.max_power / 1e3:,.0f} kW, cavity {2 * cy.hm.r_in * 1e3:.0f} mm"),
        ('Plant', f"{cy.N} sticks + {cy.N} heaters, tank {cy.tank.geo.D_tank:.1f} x {cy.tank.geo.H_tank:.1f} m"),
        ('Cycle', f"charge {cy.t_charge_hr:.1f} h, hold {cy.t_hold_hr:.1f} h, discharge {cy.t_discharge_hr:.1f} h"),
        ('Net round trip (clean cells)', f"{cy.RTE * 100:.1f}%"),
    ])
    fig, _ = plot_layout(cy)
    doc.fig(fig, 'Example design to scale. Plan: sticks (blue) in square cavities turned 45 deg with liner and '
                 'octagonal shade vanes (solid closed, dashed open), heaters (red) in round cavities, on a '
                 'checkerboard in one graphite matrix; dashed circles are the annular cells the model analyses. '
                 'Profile: vertical section through a stick and a heater.', name='Figure 8.1')
    fig, _ = plot_timeseries(cy)
    doc.fig(fig, 'One-week cycle of the example design: temperatures, power per stick / heater and, during the '
                 'discharge, cell flux and shade position.', name='Figure 8.2')
    fig, _ = plot_losses(cy)
    doc.fig(fig, 'Where the electricity bought in one cycle goes (% of AC in).', name='Figure 8.3')
    rows, last = [], None
    for g, lab, v in cy.ledger:
        rows.append([g if g != last else '', lab, f"{v * J_MWh:,.1f}", f"{v / cy.AC_in * 100:.1f}%"])
        last = g
    doc.table(['group', 'item', 'MWh / cycle', '% of AC in'], rows, widths=[1.6, 3.2, 1.1, 1.1])
    doc.h2('8.1 Ledger definitions')
    doc.bl([
        'power electronics: heater supply (1 - eta_in) on everything bought; inverter (1 - eta_out) on the DC output.',
        'insulation, charge: roof + floor loss of every cell during its charge run (cell loss sink) + side + edge '
        'losses of the plant at the mid temperature for the charge time.',
        'insulation, hold: every loss path at T_max for the hold time (the heaters make it up; stick retracted, '
        'door closed, door loss not modelled).',
        'insulation, discharge: roof + floor (sink in the discharge run) + side + edge (taken out of the heat the '
        'sticks would otherwise absorb).',
        'stick: heat absorbed on the bare and inactive areas + the cells\' conversion loss.',
        'heater leads (with a lead table): Joule heat while charging and the net heat the leads draw from the storage '
        '(section 4.8).',
        'parasitics: floor-cooling fans for the whole cycle; dry-cooler fans and pumps on the heat the sticks (and '
        'the water-cooled leads) reject.',
        'The ledger closes against the AC bought (checked in code).',
    ])


# --------------------------------------------------------------------------
# Overview, numerics summary, review list, code map
# --------------------------------------------------------------------------
def sec_overview(doc, ctx):
    doc.h1('1. Overview')
    doc.p("This document describes the tinless thermal-battery system model in enough detail to be critiqued: "
          "what goes in, how each component is modelled, which numbers are fixed in the code, how the transient "
          "runs, how a plant is assembled and costed, and how the optimizer searches designs. Statements marked "
          "Review flag approximations, placeholders or possible errors found while writing it; they are collected "
          "in section 11. Numbers in tables are read from the code and the YAML when the document is generated, "
          "so they match the code version on the title page.")
    doc.h2('1.1 The system')
    doc.p("Electricity heats a large graphite block through resistive heaters (charge). The block is held hot "
          "(hold) and discharged through thermophotovoltaic (TPV) powersticks: water-cooled square sticks covered "
          "in PV cells, lowered into square cavities in the block only while discharging. Each stick hangs inside "
          "a thin graphite liner (which keeps graphite vapour off the cells) and is surrounded by eight rotating "
          "vanes (the shades) that throttle radiation from the hot cavity wall so the cells see a constant flux as "
          "the block cools. Heaters sit in their own round cavities. Sticks and heaters alternate on a checkerboard "
          "in one continuous graphite matrix (the hot zone), inside a gas-tight, insulated steel tank standing on "
          "graphite / oxide columns over an air-cooled floor.")
    doc.h2('1.2 How the model is organised')
    doc.p("The plant is built up from one repeating cell. One stick and one heater are each analysed as an annulus "
          "of graphite with an adiabatic outer radius holding one stick + heater pair's worth of graphite (the "
          "checkerboard's symmetry lines replaced by a circle). The stick cell gives the discharge; the heater cell "
          "the charge. The tank model multiplies the cells into a plant and adds the plant-scale losses, structure "
          "and costs; the cycle accounting turns it into a round-trip efficiency; the optimizer wraps it all.")
    fig = flowchart({
        'y': (1.6, 8.6, 'YAML inputs\n(tinless_model_input.yaml)', 'io'),
        'o': (5.4, 8.6, 'Optimizer inputs + design\nvariables (optimize.py)', 'io'),
        'b': (3.5, 7.3, 'Geometry and mesh (build.py)\nstick cell + heater cell', 'proc'),
        'r': (5.2, 5.9, 'Discharge transient (tinless.py)\nradiation enclosure, vanes,\nliner, stick, block '
                        'conduction', 'loop'),
        'z': (5.2, 4.3, 'Shade-open sizing (sizing.py)\nT_min and block volume', 'loop'),
        'h': (1.6, 5.9, 'Charge transient (heater cell)\nheater power / cavity search', 'loop'),
        'd': (5.2, 2.9, 'Deposition per cycle\n-> degradation, maintenance', 'proc'),
        't': (1.8, 2.9, 'Tank and plant (tank.py)\nN sticks, layout, losses,\ncolumns, floor cooling, costs', 'proc'),
        'c': (3.5, 1.6, 'Cycle accounting (cycle.py / optimize.py)\nRTE, energy lost, lifetime capex', 'proc'),
        'f': (3.5, 0.5, 'Objective  ->  differential evolution', 'out'),
    }, [('y', 'b'), ('o', 'b'), ('b', 'r'), ('r', 'z'), ('z', 'r', 'resize', -0.5), ('b', 'h'),
        ('z', 'h', 'T_min, block'), ('z', 'd'), ('z', 't', 'stick energy, power'), ('h', 't', 'heater cavity'),
        ('d', 'c'), ('t', 'c'), ('c', 'f')], size=(7.0, 9.2))
    doc.fig(fig, 'Model hierarchy: what feeds what.', width=5.4, name='Figure 1.1')
    doc.h2('1.3 Conventions')
    doc.bl([
        'Temperatures are in K inside the cell model and in C in the tank model and optimizer outputs; tables say '
        'which.',
        'Cell flux and target flux are in W/cm2 of active cell area.',
        "'Screening' and 'detailed' settings refer to the optimizer's fidelity levels (section 9.6).",
        'file.py:line references point at the code version on the title page.',
    ])
    doc.h2('1.4 Document map')
    doc.p('2 Inputs; 3 Geometry and discretisation; 4 Component physics (block, radiation, shades, liner, stick and '
          'cells, heater, deposition, heater sizing and current leads); 5 Tank and plant; 6 Transient simulation; 7 Material properties; 8 Plant '
          'cycle and example design; 9 Optimizer; 10 Numerical settings; 11 Items to review; Appendix A '
          'alternative model options; Appendix B code map.')


def sec_numerics(doc, rows):
    doc.h1('10. Numerical settings summary')
    doc.p("Discretisation, iteration counts, tolerances, initial guesses and clamps used by the in-use model "
          "path, gathered from the sections above.")
    doc.table(['area', 'setting', 'value', 'where'], [list(r) for r in rows], widths=[1.2, 2.6, 1.6, 1.7], size=8)


def sec_review(doc):
    doc.h1('11. Items to review')
    doc.p("Every Review note in the document, in order, with the section it appears in. Confidence and "
          "estimated effect are given in the note where known. These are candidates for refinement, not "
          "confirmed errors.")
    rows = [[i + 1, sec, txt] for i, (sec, txt) in enumerate(doc.reviews)]
    doc.table(['#', 'section', 'item'], rows, widths=[0.35, 1.6, 5.15], size=8)


CODE_MAP = [
    ('thermal/build.py', 'Builds the stick-cell and heater-cell models from the YAML: geometry, mesh, components, '
                         'radiation model wiring, degradation, heater power cap.'),
    ('thermal/components.py', 'Powerstick, storage block (conduction), heater and related component classes.'),
    ('thermal/tinless.py', 'TinlessSimulation: discharge (run_simulation) and charge (run_heating_simulation) '
                           'transients, time stepping, legacy local radiation model.'),
    ('thermal/axial.py', 'Full-height radiation enclosure: AxialRadiation (Newton radiosity solve), '
                         'VaneHybridRadiation (vane positions), LinerTable, LinerBottom.'),
    ('thermal/viewfactors.py', 'Analytic and Monte Carlo view factors, axial mesh, enclosure assembly.'),
    ('thermal/correlations.py', 'Vane geometry and ray tracing (square cavity), vane studies.'),
    ('thermal/properties.py', 'Material properties (graphite, storage, insulation, emissivities, cell efficiency).'),
    ('thermal/deposition.py', 'Carbon deposition on the cells (vapour transport across the liner gap).'),
    ('thermal/sizing.py', 'run_discharge, shade-open sizing of T_min and block volume.'),
    ('thermal/tank.py', 'Plant: hot-zone layout, tank, insulation losses, columns, floor cooling, foundation, costs, '
                        'per-cell roof / floor loss sink.'),
    ('thermal/heater.py', 'Heater element sizing (tube from power, voltage and resistivity; rod), element mass.'),
    ('thermal/heater_leads.py', 'Heater current leads: material library, 1-D lead model, lifetime cost, lead '
                                'optimizer, lead table (build + interpolation), full lead optimization of a design.'),
    ('thermal/cycle.py', 'Full charge-hold-discharge cycle, energy ledger, layout and cycle plots.'),
    ('thermal/optimize.py', 'Design variables, evaluation pipeline, objective, differential evolution, run log.'),
    ('thermal/report.py', 'Run report (Word) of an optimizer run.'),
    ('thermal/docs*.py', 'This document.'),
]


def build_docs(path, base_cfg, opt, yaml_path='tinless_model_input.yaml', example_dir=None, cy=None, top=None,
               verbose=True, run_example_design=True, lead_detail=None):
    """Write the model documentation to `path` (.docx). example_dir: optimizer run whose best
    design is re-run as the example (default: the latest run in opt_runs/); pass cy / top to
    reuse an example already run; run_example_design=False skips it (quick check). lead_detail:
    heater_leads.lead_for_design result for the example lead figure (else a lead-table point). -> path"""
    import importlib
    from . import docs_secA, docs_secB, docs_secC, docs_secD, docs_secE
    for mod in (docs_secA, docs_secB, docs_secC, docs_secD, docs_secE):
        importlib.reload(mod)
    say = print if verbose else (lambda *a, **k: None)
    if cy is None and run_example_design:
        example_dir = example_dir or latest_run()
        if example_dir:
            say(f"example: best design of {example_dir}, detailed settings ...", flush=True)
            top, cy = run_example(base_cfg, opt, example_dir)
    ctx = dict(base_cfg=base_cfg, opt=opt, yaml_path=yaml_path, cy=cy, top=top, example_dir=example_dir,
               lead_detail=lead_detail)
    doc = Doc()
    now = _dt.datetime.now().strftime('%Y-%m-%d')
    doc.title('Tinless thermal battery: system model documentation',
              f"Generated {now} from the code ({_git()}) and {yaml_path}. For review: please comment directly in "
              f"the document.")
    doc.toc()
    doc.page_break()
    numerics = []
    sec_overview(doc, ctx)
    say('sections 2-4 ...', flush=True)
    numerics += docs_secA.sec_inputs_geometry(doc, ctx) or []          # 2, 3, 4.1
    numerics += docs_secB.sec_radiation(doc, ctx) or []                # 4.2 - 4.4
    numerics += docs_secA.sec_stick(doc, ctx) or []                    # 4.5
    numerics += docs_secC.sec_heater_deposition(doc, ctx) or []        # 4.6, 4.7
    numerics += docs_secE.sec_heater_leads(doc, ctx) or []             # 4.8
    say('section 5 ...', flush=True)
    numerics += docs_secD.sec_tank(doc, ctx) or []                     # 5
    say('section 6 ...', flush=True)
    numerics += docs_secC.sec_transient(doc, ctx) or []                # 6
    say('section 7 ...', flush=True)
    numerics += docs_secA.sec_properties(doc, ctx) or []               # 7
    if cy is not None:
        sec_cycle(doc, cy, top, opt)                                   # 8
    else:
        doc.h1('8. Plant cycle and example design')
        doc.p('No example design was run for this build.')
    sec_optimizer(doc, opt)                                            # 9
    sec_numerics(doc, numerics)                                        # 10
    sec_review(doc)                                                    # 11
    doc.h1('Appendix A. Alternative model options')
    for mod in (docs_secA, docs_secB, docs_secC, docs_secD):
        f = getattr(mod, 'alternatives', None)
        if f:
            f(doc, ctx)
    doc.h1('Appendix B. Code map')
    doc.table(['file', 'contents'], [list(r) for r in CODE_MAP], widths=[1.8, 5.3])
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    doc.save(path)
    say(f"written {path} ({len(doc.reviews)} review items)")
    return path
