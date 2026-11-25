"""Model documentation: section 4.8 -- heater sizing, the heater element, the heater current
leads (1-D model, lifetime cost, optimizer), the lead table and how leads enter the plant model.

Numbers are read from the code (thermal/heater.py, thermal/heater_leads.py, thermal/optimize.py),
the YAML (ctx['base_cfg']), the optimizer settings (ctx['opt']) and, when the run uses one, the
lead table, when the document is built. file:line references follow the code.
"""
import math
import os

import numpy as np

from .docs_base import flowchart, DIM

C2K = 273.15
_SRC = {}
_FIG = {'n': 0}


def _ref(fname, pat, nth=1):
    """'fname:line' of the nth line of thermal/fname containing pat (live from the source)."""
    if fname not in _SRC:
        try:
            _SRC[fname] = open(os.path.join(os.path.dirname(__file__), fname), encoding='utf-8').read().splitlines()
        except OSError:
            _SRC[fname] = []
    k = 0
    for i, line in enumerate(_SRC[fname]):
        if pat in line:
            k += 1
            if k == nth:
                return f"{fname}:{i + 1}"
    return fname


def _figname():
    _FIG['n'] += 1
    return f"Figure 4.8.{_FIG['n']}"


def _example_lead(ctx):
    """(lead ctx, evaluated lead, label) for the figure: ctx['lead_detail'] (full optimization of
    the run's top design) if given, else the lead-table point nearest the example design (or the
    middle of the grid), re-evaluated from its stored geometry. None if neither is available."""
    from . import heater_leads as HL
    from . import optimize as O
    det = ctx.get('lead_detail')
    if det and det.get('res', {}).get('ok'):
        return det['ctx'], det['res'], (f"full lead optimization of design #{det.get('rank', 0) + 1} of the run "
                                        f"(lead-detail cell)")
    lt = O.lead_table(ctx['opt'])
    if lt is None:
        return None
    try:
        g, tab = lt.data['grid'], O.merge(HL.TABLE_DEFAULTS, lt.data.get('table') or {})
        top = ctx.get('top')
        t_top = top['design']['t_top'] if top else float(np.median(g['t_top']))
        T_max = top['design']['T_max_C'] if top else float(np.median(g['T_max_C']))
        pick = lambda vals, v: min(vals, key=lambda a: abs(a - v))
        I_kA, t, T = g['I_kA'][-1], pick(g['t_top'], t_top), pick(g['T_max_C'], T_max)
        row = next(r for r in lt.data['rows'] if (r['I_kA'], r['t_top'], r['T_max_C']) == (I_kA, t, T))
        if not row.get('ok'):
            return None
        cfg, opt = ctx['base_cfg'], ctx['opt']
        V = float(cfg['heater'].get('voltage') or 60.0)
        swing = tab['swing'] or O.merge(O.OPT_DEFAULTS, opt)['sizing']['swing_guess']
        li = dict(design=None, P_heater=I_kA * 1e3 * V, voltage=V, L=t, T_ins_hot=T + C2K,
                  T_idle=T + C2K - swing / 2, idle_break_after=tab['idle_break_after'])
        lc = HL.lead_context(cfg, opt, li)
        d = row['d']
        lead = dict(mats=row['zones'], lens=row['lens'], d=d, d_joint=[min(d[j], d[j + 1]) for j in range(len(d) - 1)],
                    cooled=tab['cooled'], taper_deg=lc['li']['taper_deg'])
        res = HL.evaluate(lc, lead)
        if not res.get('ok'):
            return None
        return lc, res, f"lead-table point I = {I_kA:g} kA, t_top = {t:g} m, block T_max = {T:.0f} C"
    except Exception as e:                            # the figure is optional
        print(f"[docs] example lead not available ({type(e).__name__}: {e})")
        return None


def sec_heater_leads(doc, ctx):
    from . import heater as H
    from . import heater_leads as HL
    from . import optimize as O
    cfg, opt = ctx['base_cfg'], ctx['opt']
    h = cfg['heater']
    st = cfg['storage']
    Hb = float(st.get('block_height') or st['L'])
    T_min = float(st['T_min'])
    el = H.element(h, Hb, T_min)
    li = HL.LEAD_DEFAULTS
    lt = O.lead_table(opt)
    g = opt['geometry']
    fid = opt['fidelity']
    _FIG['n'] = 0
    rows = []

    doc.h2('4.8 Heater sizing, element and current leads')
    doc.p("Section 4.6 describes how a heater of given power heats the block. This section covers how the power, "
          "the element and the current leads that feed it are sized, and how their cost and losses enter the plant. "
          "The chain, for every design the optimizer evaluates:")
    doc.nl([
        'Heater search (optimizer): the smallest heater power that charges the block in charge_hr with the cavity at '
        'its minimum clearance; the cavity grows only if the power cap is not enough.',
        'Element: a graphite tube sized so the supply voltage drives that power through it (thermal/heater.py). It '
        'sets the heater current I = P / V.',
        'Current leads: two per heater (supply + return) carry I from the top of the element up through the top '
        'Thermax (thickness t_top) to the deck. Each is graphite / tungsten / copper in series, the copper '
        'water-cooled (thermal/heater_leads.py). Their geometry is optimized for the lowest lifetime cost per lead.',
        'Plant: a lead table of optimal leads (built once per optimizer session with that session\'s costs) gives '
        'every design its leads\' cost, Joule heat, heat drawn from the storage and heat rejected, interpolated at '
        'its current, t_top and block T_max. These enter the capex, the energy ledger, the RTE and the objective.',
    ])
    fig = flowchart({
        'a': (3.6, 5.4, 'Design: block, T_min / T_max, t_top\n(after shade-open sizing)', 'io'),
        'b': (3.6, 4.2, 'Heater search: power P (+ cavity) for charge_hr\n(charge runs, element resized per P)', 'loop'),
        'c': (3.6, 3.0, 'Element: tube D, wall, R = V^2 / P;  I = P / V', 'proc'),
        'd': (3.6, 1.8, 'Lead table: per-lead cost + loss powers\nat (I, t_top, T_max)', 'proc'),
        'e': (3.6, 0.6, 'Plant: capex lines, energy ledger, RTE, objective', 'out'),
        't': (7.0, 1.8, 'Lead table, built before the run:\noptimal leads on t_top x T_max x I\n'
                        '(1-D lead model + DE per point)', 'loop'),
    }, [('a', 'b'), ('b', 'c'), ('c', 'd'), ('d', 'e'), ('t', 'd')], size=(8.6, 6.0))
    doc.fig(fig, 'Heater and lead sizing for one design, and the lead table built before the run.', width=5.4,
            name=_figname())

    # ------------------------------------------------------------------ heater search
    doc.h3('Heater power and cavity (optimizer heater search)')
    doc.p(("optimize.heater_search", {'mono': True}), f" ({_ref('optimize.py', 'def heater_search')}) runs on the "
          "block left by the shade-open sizing, with charge runs of the heater cell (section 6.6) up to 1.5 x the "
          f"target charge time charge_hr = {opt['charge_hr']:g} h:")
    doc.nl([
        f"Reject if the heater limit max_temperature ({h['max_temperature'] - C2K:.0f} C) is not above the block T_max "
        "+ 1 K (violation (T_max + 1 - T_h,max) / 100).",
        "Power cap P_cap = max_power_per_m x H_block (or heater.max_power). If even a perfect charge at P_cap, "
        "E_block / P_cap with E_block = rho cp V (T_max - T_min) at T_max, takes longer than charge_hr, reject "
        "(violation t / charge_hr - 1).",
        f"Charge at P_cap with the cavity at its minimum: radius = element radius + the minimum heater gap "
        + (f"(geometry.heater_min_gap, fixed at {g['heater_min_gap']} m)" if g.get('heater_min_gap') is not None else
           f"(the arcing clearance of the heater supply voltage: gap = V / {g.get('heater_arc_V_per_in', 7.5):g} V/in, "
           f"{O.heater_gap(opt, cfg) * 1e3:.0f} mm at the YAML {cfg['heater']['voltage']:g} V)")
        + f". If that meets charge_hr, bisect the power (geometrically) between E_block / "
        f"charge_hr and P_cap to the smallest that still meets it, to a ratio of 1 + heater_ar_tol "
        f"({fid['heater_ar_tol']} screening, {O.DETAILED['heater_ar_tol']} detailed). The element and the minimum "
        "cavity are resized for every power tried, so each charge run tests the element that power needs.",
        "Otherwise keep P_cap and grow the cavity: area ratio x 1.6 per try (up to 8 tries) until the charge meets "
        "charge_hr, then bisect the area ratio to the same tolerance. A bigger cavity spreads the power over more "
        "wall, lowering the wall-to-bulk temperature difference so the heater stays below its limit longer.",
        "If no cavity meets it: reject with violation t_charge / charge_hr - 1.",
    ])
    doc.p("The design keeps the heater power and cavity found; the heater supply, the grid interconnect and the "
          "element are costed on that power.")

    # ------------------------------------------------------------------ element
    doc.h3('Heater element')
    doc.p(f"heater.type = {h.get('type', 'rod')!r} ({_ref('heater.py', 'def element')}). A tube element of outer "
          "diameter D and wall t = w D is sized so the DC supply voltage V drives the heater power P through it at "
          "the highest resistivity over its operating range (so the supply can always deliver P; where the "
          "graphite conducts better the supply turns the voltage down):")
    doc.eq(r'R = \frac{V^2}{P},\qquad R = \frac{\rho_e\,n_{leg}^2\,H}{A},\qquad A = \pi D^2 w(1-w)'
           r'\quad\Rightarrow\quad D = \sqrt{\frac{\rho_e\,n_{leg}^2\,H\,P}{V^2\,\pi\,w(1-w)}},\qquad I = \frac{P}{V}',
           where='H: heated length = block height; n_leg: strips in series (1 = down the tube and back through a '
                 'separate return; 2 = U-element); rho_e = max of properties.heater_graphite_resistivity over '
                 '[T_min, max_temperature] (Mersen 2020 data to 2002 C, extrapolated above).')
    if el['type'] == 'tube':
        doc.table(['quantity', 'YAML heater at its power cap', 'note'], [
            ['power', f"{el['P'] / 1e3:,.0f} kW", f"max_power_per_m {h['max_power_per_m'] / 1e3:.0f} kW/m x H {Hb:g} m"],
            ['voltage / current', f"{el['V']:g} V / {el['I'] / 1e3:,.2f} kA", 'heater.voltage; I = P / V'],
            ['resistivity used', f"{el['rho_e'] * 1e6:.2f} uOhm m", 'highest over the operating range'],
            ['element', H.describe(el), f"wall_ratio {h.get('wall_ratio')}, legs {el.get('legs', 1)}"],
            ['cross-section', f"{el['A_cross'] * 1e4:.1f} cm2", f"density {el['density']:g} kg/m3"],
        ], widths=[1.4, 2.6, 3.1])
    doc.p("Element graphite mass = A x length x density, costed at heater_graphite_per_kg (default the iso-molded "
          "graphite price) plus heater_fixed per heater. With the heater leads modelled (optimizer input "
          "leads.table set) the length is the block height only: the leads carry the current through the roof "
          "insulation. Without them the element is taken on up through the top insulation (H_block + t_top), the "
          "older assumption, and its conduction and Joule losses there are not counted.")
    doc.p("The heater current matters for the leads: lead cost and losses scale roughly with I, so for a given "
          "power a higher supply voltage means proportionally cheaper, lower-loss leads (and a thinner, longer-"
          "resistance element). The lead table is built per ampere for this reason (below).")

    # ------------------------------------------------------------------ lead geometry
    doc.h3('Current leads: geometry')
    doc.p(f"Each heater has n_leads = {li['n_leads']} leads (optimizer input leads.n_per_heater). A lead runs from "
          "the element at the bottom face of the top Thermax (x = 0) up through it to the deck terminal at its top "
          "face (x = L = t_top). It is up to four material zones in series, hot to cold; the system uses graphite / "
          "tungsten / copper ("
          + ' / '.join(HL.TABLE_DEFAULTS['zones']) + " in the lead table), each a solid round bar:")
    doc.bl([
        f"Each zone has a body (largest) diameter D_i within d_bounds {li['d_bounds']} m (scaled with sqrt(I / "
        f"{HL.TABLE_DEFAULTS['I_ref_kA']:g} kA) in the lead table, since optimal diameters go as sqrt(I)) and a length "
        f"of at least min_zone_len = {li['min_zone_len']} m.",
        "No necks: the two bars at a joint share one contact diameter, the smaller of the two body diameters. "
        f"From the joint the larger bar widens at no more than taper_deg = {li['taper_deg']:g} deg (half-angle "
        "from the axis) to its body diameter (heater_leads.lead_profile). Without this the optimizer chose very "
        "different diameters on the two sides of a joint, which is not buildable.",
        f"Each lead sits in a bore in the Thermax with a radial argon gap of {li['gap'] * 1e3:.0f} mm.",
        "Optional idle disconnect (idle_break_after = a material): the joint on the cold side of that zone is "
        "opened while the heater is off (hold and discharge) and closed for charging. The lead table uses the "
        f"graphite joint ({HL.TABLE_DEFAULTS['idle_break_after']!r}).",
    ])

    # ------------------------------------------------------------------ lead thermal model
    doc.h3('Current leads: 1-D thermal model')
    doc.p(f"Steady conduction along the lead with Joule heating and exchange with the surrounding insulation "
          f"(heater_leads.solve_state, {_ref('heater_leads.py', 'def solve_state')}):")
    doc.eq(r'\frac{d}{dx}\left(k(T)\,A(x)\,\frac{dT}{dx}\right) + \frac{I^2\,\rho_e(T)}{A(x)} - q_{ex}(x) = 0',
           where='A(x) = pi D(x)^2 / 4 from the zone profile; k and rho_e per material (table below); q_ex [W/m] heat '
                 'from the lead into the insulation.')
    doc.p("Exchange with the insulation: across the gap to the bore wall at T_b by gas conduction and radiation "
          "between concentric cylinders, in series with radial conduction through the Thermax out to r_far = "
          f"{li['r_far']} m, where the Thermax is at its undisturbed temperature T_w(x):")
    doc.eq(r'q_{gap} = \frac{2\pi k_{Ar}}{\ln(r_b/r)}(T - T_b) + \frac{2\pi r\,\sigma\,(T^4 - T_b^4)}'
           r'{1/\varepsilon + (r/r_b)(1/\varepsilon_w - 1)},\qquad '
           r'q_{ins} = \frac{2\pi k_{tx}}{\ln(r_{far}/r_b)}(T_b - T_w),\qquad q_{ex} = q_{gap} = q_{ins}',
           where=f"r = D / 2, r_b = r + gap; k_Ar = 0.0177 (T / 300)^0.73 W/m-K at the film temperature; eps: lead "
                 f"material emissivity; eps_w = {li['eps_wall']} (Thermax bore); k_tx = Thermax k at the mean of T_b "
                 "and T_w. T_b is found by fixed-point iteration within the solve.")
    doc.p("Undisturbed Thermax profile. The Thermax conductivity is k = K0 exp(B T_C) (properties.thermax_tcx). "
          "Through a slab with both faces held, the heat flux is uniform, so the Kirchhoff transform "
          "phi = int k dT is linear in x; with this k that makes exp(B T_w) linear in x:")
    doc.eq(r'T_w(x) = \frac{1}{B}\ln\left(e_h + (e_c - e_h)\,\frac{x}{L}\right),\qquad e = e^{B\,T_C}\ '
           r'\mathrm{at\ the\ hot\ (bottom)\ and\ cold\ (top)\ faces}',
           where=f"bottom face at the block T_max (T_ins_hot), top face T_ins_cold = {li['T_ins_cold'] - C2K:.0f} C. "
                 "The profile is hotter than a linear one over most of the thickness because k rises steeply "
                 "with T.")
    doc.p("Two operating states per cycle:")
    doc.table(['state', 'current', 'hot end T(0)', 'Thermax faces', 'hours per cycle'], [
        ['charging', 'rated I = P / V', 'heater max_temperature', 'T_max (bottom) / T_ins_cold (top)', 'charge_hr'],
        ['heater off (hold + discharge)', '0', 'block mean (T_min + T_max) / 2', 'same', 'cycle - charge_hr'],
    ], widths=[1.6, 1.1, 1.6, 1.8, 1.0])
    doc.p("Cold end: with the last zone uncooled ('passive') the 1-D solve runs to the top face, where a water-"
          f"cooled terminal clamp holds T_term = {li['T_term'] - C2K:.0f} C. With the last zone water-cooled along "
          f"its length ('cooled', the system choice) the solve ends at the start of that zone, held at T_cool = "
          f"{li['T_cool'] - C2K:.0f} C; the cooled zone's own Joule heat and the heat it absorbs from the insulation "
          "are added to the heat rejected. An open joint (idle disconnect) is a conductance in series in its "
          f"segment: argon conduction across break_gap = {li['break_gap'] * 1e3:.0f} mm plus radiation between "
          "parallel faces of the contact area.")
    doc.p(f"Numerics: {li['nodes']} finite-volume segments; conduction, Joule heat and exchange integrated over "
          f"{li['sub_pts']} sub-points per segment and control volume (the diameter varies within them); Newton on "
          "the Joule and exchange sources, Picard on k(T); tridiagonal solve; temperature steps limited to 300 K "
          "per iteration; converged when the largest change is below 1e-3 K (at most 300 iterations). A lead whose "
          "temperature rises past the materials' limits + 1500 K is reported as thermal runaway (no steady "
          "solution: Joule heating outruns conduction and exchange) and is infeasible.")
    rows += [
        ['leads', 'lead grid', f"{li['nodes']} segments x {li['sub_pts']} sub-points", _ref('heater_leads.py', "'nodes':")],
        ['leads', 'lead solve', 'Newton (sources) + Picard (k), tol 1e-3 K, max 300 it., step <= 300 K',
         _ref('heater_leads.py', 'def solve_state')],
        ['leads', 'lead optimizer', f"DE popsize {li['de']['popsize']}, maxiter {li['de']['maxiter']}, tol "
                                    f"{li['de']['tol']:g} + Nelder-Mead polish", _ref('heater_leads.py', 'def optimize_leads')],
    ]

    # ------------------------------------------------------------------ energy and cost
    doc.h3('Current leads: energy accounting and lifetime cost')
    doc.p("Heat the lead puts into the insulation at height x splits between the deck (up) and the storage "
          "(down) in the ratio of the slab resistances on either side; with k = K0 exp(B T) the up-fraction is "
          "exactly w(x) = (T_ins_hot - T_w(x)) / (T_ins_hot - T_ins_cold). Then per lead:")
    doc.eq(r'Q_{drain} = Q_{hot} - \int q_{ex}\,(1 - w)\,dx,\qquad Q_{rej} = Q_{end} + Q_{J,cool} + Q_{abs,cool},\qquad '
           r'Q_J + Q_{drain} = Q_{rej} + \int q_{ex}\,w\,dx',
           where='Q_hot: heat into the lead from the element at x = 0; Q_drain: net heat drawn from the storage '
                 '(< 0 while charging when Joule heat flows down into the block, which then counts as delivered); '
                 'Q_rej: heat to the cooling water / terminal. The balance is checked in the tests.')
    doc.p("Lifetime cost of one lead, priced as the system optimizer prices energy (no discounting, "
          "cycles_per_year x lifetime_yr cycles at elec_price):")
    doc.eq(r'C_{lead} = (1+f_{soft})\left[\sum_i V_i c_i + C_{joints} + c_{rej}\,\max(Q_{rej}) + c_{sup}\,Q_{J,ch}\right]'
           r' + p_e N_{cyc}\left[E_J + \frac{E_{drain}}{\eta_{in}} + k_{rej}\,E_{rej}\right]',
           where='V_i c_i: zone volume (integral of the profile area) x $/m3; c_rej: heat_rejection_per_kWth on the '
                 'peak rejected heat; c_sup: charge_electrics_per_kW on the lead Joule power; E per cycle: Joule '
                 'while charging, storage drain (put back by the heaters through eta_in), rejected heat x '
                 'heat_rejection_kWe_per_kWth (fans and pumps).')
    doc.p("Temperature limits: each zone's maximum temperature must stay below its material T_max, and each joint "
          "below the lower T_max of its two materials (the colder side). Every kelvin over adds penalty_per_K = "
          f"${li['penalty_per_K']:,.0f} to the objective; a lead within T_tol = {li['T_tol']} K counts as feasible "
          "(optima ride on the limits).")

    # ------------------------------------------------------------------ optimizer
    doc.h3('Current leads: optimizer')
    doc.p(f"heater_leads.optimize_leads ({_ref('heater_leads.py', 'def optimize_leads')}): for one zone set and "
          "cooling mode, differential evolution over x = [D_1..D_n, f_1..f_(n-1)]: body diameters within d_bounds and "
          "length fractions in [0, 1]. Every zone first gets min_zone_len; zone i then takes f_i of the free "
          "length still left and the last zone the rest, so every point of the search box is a valid lead. "
          f"Settings: Sobol initialisation, popsize {li['de']['popsize']}, maxiter {li['de']['maxiter']}, tol "
          f"{li['de']['tol']:g}, seed {li['de']['seed']}, no DE polish; then a bounded Nelder-Mead polish, keeping "
          "the better of the two.")
    doc.p(f"heater_leads.optimize_with_skips ({_ref('heater_leads.py', 'def optimize_with_skips')}): when an "
          f"optimum leaves zones at the minimum length (within {li['at_min_tol'] * 1e3:.0f} mm), the optimization is "
          "re-run without each of them and without all of them (graphite / copper, tungsten / copper, graphite "
          "only, ...), recursively; a water-cooled run needs a coolable last zone. The cheapest feasible set wins.")

    # ------------------------------------------------------------------ materials
    doc.h3('Lead materials')
    mats = HL.lead_context(cfg, opt, {'design': None})['mats'] if h.get('voltage') else None
    mrows = []
    for name, m0 in HL.MATERIALS.items():
        m = (mats or {}).get(name, m0)
        kk = [float(m['k'](np.array([T]))[0]) for T in (300.0, 1000.0, 2000.0)]
        rr = [float(m['rho_e'](np.array([T]))[0]) * 1e8 for T in (300.0, 1000.0, 2000.0)]
        mrows.append([name, ' / '.join(f"{v:.0f}" for v in kk), ' / '.join(f"{v:.1f}" for v in rr),
                      f"{m['density']:,.0f}", f"{m['T_max'] - C2K:.0f}", f"{m['emissivity']:g}",
                      f"{m['cost_per_m3']:,.0f}" if m.get('cost_per_m3') else 'heater graphite $/kg x density',
                      'yes' if m.get('coolable') else ''])
    doc.table(['material', 'k [W/m-K] 300 / 1000 / 2000 K', 'rho_e [uOhm cm] 300 / 1000 / 2000 K', 'kg/m3',
               'T_max [C]', 'emis.', '$/m3', 'cool-able'], mrows, widths=[0.8, 1.2, 1.3, 0.6, 0.6, 0.5, 1.4, 0.5], size=8)
    doc.p("Sources: graphite resistivity Mersen 2020 (properties.heater_graphite_resistivity), k from "
          "properties.gr_tcx; tungsten, molybdenum, copper, aluminum k from Incropera Table A.1; resistivities from "
          "Desai et al. 1984 (W), Matula 1979 (Cu) and typical data (Mo, Al, 304 stainless). Tables are clamped "
          "outside their range. The notebook's LEAD_INPUTS can override T_max, cost and emissivity per material.")

    # ------------------------------------------------------------------ table
    doc.h3('Lead table (how the system optimizer uses the leads)')
    doc.p("Optimizing the leads for every design (minutes each) would multiply the optimizer's cost. Within one "
          "optimizer session the prices, scenario, cycle and heater limit are fixed, and the optimal lead depends "
          "on three things: the heater current I, the lead length t_top and the block T_max (the Thermax hot face). "
          "A scan over 3 - 12 kA showed the optimal lifetime cost per ampere flat within 3.5% and the loss "
          "components per ampere within about 20%, so the table holds per-ampere values at a few currents.")
    tab = O.merge(HL.TABLE_DEFAULTS, (lt.data.get('table') if lt else None) or {})
    doc.bl([
        f"Grid: t_top {tab['t_top']} m x block T_max {[round(v) for v in tab['T_max_C']]} C x I {tab['I_kA']} kA "
        f"({len(tab['t_top']) * len(tab['T_max_C']) * len(tab['I_kA'])} lead optimizations in a process pool, "
        "about 10 - 25 min on 20 workers).",
        f"Leads: {' / '.join(tab['zones'])}, last zone {'water-cooled' if tab['cooled'] else 'passive'}, "
        + (f"{tab['idle_break_after']} joint open when the heater is off" if tab['idle_break_after'] else 'always connected')
        + "; heater-off hot end at T_max - swing_guess / 2 (the actual T_min is not known before the run).",
        "Built at the start of every optimizer session from that session's cost inputs, scenario and lead inputs "
        "(opt_leads cell, REBUILD_LEAD_TABLE = True), saved in opt_runs/_lead_tables under a hash of all of them "
        "and copied into the run folder; the run cell checks the hash against the current inputs before the run.",
        "Interpolation (heater_leads.LeadTable): each stored quantity divided by its current, linear in (I, t_top, "
        "T_max), times the design's current; designs outside the grid are clamped to its edge and flagged.",
        "Stored per point: lifetime $ (total, capex, energy), material and joint $, Joule power while charging, "
        "storage drain while charging and idle, heat rejected while charging and idle, peak rejection, energy "
        "lost per cycle.",
    ])
    if lt is not None:
        doc.p(f"Lead table of this build: {os.path.basename(str(lt.path))}.")
        doc.code(lt.summary())

    # ------------------------------------------------------------------ plant integration
    doc.h3('Leads in the plant model')
    doc.p(f"optimize.lead_energies ({_ref('optimize.py', 'def lead_energies')}), used by evaluate_design and by "
          "the full-cycle accounting (cycle.run_cycle), takes the table values at I = P_heater / V, t_top and "
          "T_max for the n = n_per_heater x N leads of the plant:")
    doc.eq(r'E_{J} = n\,Q_{J,ch}\,t_{ch},\qquad E_{drain} = n\,(Q_{drain,ch}\,t_{ch} + Q_{drain,off}\,t_{off}),\qquad '
           r'E_{rej} = n\,(Q_{rej,ch}\,t_{ch} + Q_{rej,off}\,t_{off})',
           where='t_ch: the design\'s charge time; t_off = cycle - t_ch. Joule heat is bought; the drain is put back by '
                 'the heaters; the rejected heat loads the dry coolers.')
    doc.eq(r'E_{AC,in} = \frac{N(E_{rel} + E_{sink,ch}) + (Q_{side}+Q_{edge})\,t_{ch} + Q_{loss}(T_{max})\,t_{hold}'
           r' + E_{drain} + E_J}{\eta_{in}},\qquad E_{AC,net} = E_{AC,out} - P_{fan}\,t_{cyc} - c_{rej}\,(N\,\overline{E_{rej}}'
           r' + E_{rej,lead})')
    doc.bl([
        "capex: a 'heater leads' line = n x (lead material + joints), x (1 + soft costs) with the rest; the lead "
        "Joule power at full current is added to each heater's supply rating (charge_electrics_per_kW) and to the "
        "plant charging draw that sizes the grid interconnect;",
        "heat rejection capacity per stick slot = max(stick discharge peak + the idle leads' rejection, the leads' "
        "rejection while charging);",
        "element graphite over the block height only (the leads replace the length through the roof);",
        "outputs: leads (the per-lead table values), E_lead_joule_MWh, E_lead_drain_MWh, E_lead_lost_MWh per cycle;",
        "full-cycle ledger: group 'heater leads' with the Joule heat and the net heat drawn from the storage; the "
        "heat-rejection fan line includes the leads.",
    ])
    doc.p("Without a lead table (leads.table = None) none of this is applied and the element runs through the top "
          "insulation, as before the lead model existed.")

    # ------------------------------------------------------------------ top designs
    doc.h3('Full lead optimization of a top design and the table check')
    doc.p(f"heater_leads.lead_for_design ({_ref('heater_leads.py', 'def lead_for_design')}), run by the optimizer "
          "notebook's lead-detail cell for the best design: the leads optimized in full with the table's settings at "
          "the design's own heater power, t_top, T_min and T_max (the table used the swing guess for the heater-off "
          "temperature), and the table's interpolated values compared with it. The run report shows the lead, its "
          "temperatures and heat generation, and the table error.")
    ex = _example_lead(ctx)
    if ex is not None:
        lc, res, lab = ex
        doc.code(HL.describe(lc, res, f"Example lead ({lab})"))
        fig = HL.plot_lead_section(lc, res, title=f"Example heater lead: {' / '.join(res['lead']['mats'])}, "
                                                  f"${res['total']:,.0f} per lead (lifetime)")
        doc.fig(fig, f"Example lead ({lab}). Left: section through the top insulation to scale, with the material "
                     "zones, bar diameters, bore gap and Thermax. Middle: temperature along the lead while charging "
                     "and with the heater off, the undisturbed Thermax profile and each zone's limit. Right: heat per "
                     "metre: Joule heating while charging and exchange with the insulation (< 0: taken from it).",
                width=6.8, name=_figname())

    # ------------------------------------------------------------------ review
    doc.review("Lead model omissions: contact resistance at the joints, the lead above the insulation top face, "
               "the disturbance of the Thermax by the lead, and the tungsten-graphite reaction (W2C / WC form above "
               "~1400 C) at the graphite / tungsten joint, which runs near the tungsten limit while charging. "
               "Confidence high (code); effect on the joint design possibly large, on cost probably small.")
    doc.review("The charging state uses the rated current for the whole charge_hr; once the heater reaches its "
               "temperature limit its power (and current) falls. Conservative for the Joule heat (overstated); the "
               "lead table also uses charge_hr rather than each design's charge time for the lead's own cost.")
    doc.review("The lead table assumes the heater-off hot end at T_max - swing_guess / 2, not each design's mean "
               "block temperature, and is interpolated linearly between grid points; the lead-detail cell measures "
               "the resulting error for the best design. Re-build the table if the error is more than a few percent.")
    doc.review("Lead material costs, joint and disconnect-mechanism costs are placeholders; the idle disconnect "
               "is costed at break_cost (0 by default) with no reliability penalty.")
    return rows
