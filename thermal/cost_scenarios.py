"""
Cost scenarios for the optimizer: Pilot, FOAK, NOAK.

Source: docs/cost_scenarios_pilot_foak_noak.md (from 20260511_FP-system-cost-model_v6.xlsx,
the Fourth Power system cost model: Pilot = its 2027 Pilot column, FOAK = 2028 Phase 1,
NOAK = 2034 Phase 3). Only unit costs and efficiencies are taken from it; the product
definition (P_rated, duration, cycle) and all physics stay the model's own.

Select one with OPT_INPUTS['economics']['cost_scenario'] = 'Pilot' | 'FOAK' | 'NOAK'
(case-insensitive; None = the inputs cell as written). The scenario is an override layer
applied last: OPT_DEFAULTS < the notebook inputs cell < the scenario. Every key a scenario
sets is listed below; anything it does not set keeps the inputs-cell value (liner, shades,
bare cleaning, elec_price, cycles, lifetime, the other tank rates, ...).

Each entry has the same nesting as the optimizer inputs (see thermal.optimize.OPT_DEFAULTS):
economics, costs, parasitics and tank_costs (central values
of the YAML tank.costs rates, applied to every design by optimize.design_cfg).
"""

# Heater: costs.heater_fixed per heater (Nate's values, 2026-11-24) + the element's graphite (element
# mass x iso_graphite_per_kg, added by the optimizer); not scaled with heater power. For reference, the
# spreadsheet's 1 MW_th heater without its graphite body (machining, two tungsten elements, assembly,
# on-site install) is 7,681 / 6,031 / 4,331 $.

SCENARIOS = {
    'pilot': {
        'economics': {'eta_in': 0.95, 'eta_out': 0.95, 'soft_cost_frac': 0.55},
        'costs': {
            'cell_per_cm2': 36.52, 'stick_per_m': 339.0, 'stick_fixed': 7510.0,
            'interconnect_per_kW': 357.8, 'charge_electrics_per_kW': 0.0, 'inverter_per_kW': 0.0,
            'heat_rejection_per_kWth': 780.0,
            'iso_graphite_per_kg': 20.0, 'heater_graphite_per_kg': None,
            'heater_fixed': 7500.0,
        },
        'parasitics': {'heat_rejection_kWe_per_kWth': 0.05},
        'tank_costs': {'storage_graphite_per_kg': 1.50, 'steel_per_t': 5030.0, 'deck_per_t': 5030.0,
                       'structural_steel_per_kg': 5.03},
    },
    'foak': {
        'economics': {'eta_in': 0.98, 'eta_out': 0.98, 'soft_cost_frac': 0.375},
        'costs': {
            'cell_per_cm2': 4.557, 'stick_per_m': 263.0, 'stick_fixed': 5110.0,
            'interconnect_per_kW': 286.2, 'charge_electrics_per_kW': 0.0, 'inverter_per_kW': 0.0,
            'heat_rejection_per_kWth': 567.0,
            'iso_graphite_per_kg': 16.0, 'heater_graphite_per_kg': None,
            'heater_fixed': 6000.0,
        },
        'parasitics': {'heat_rejection_kWe_per_kWth': 0.05},
        'tank_costs': {'storage_graphite_per_kg': 0.75, 'steel_per_t': 5030.0, 'deck_per_t': 5030.0,
                       'structural_steel_per_kg': 5.03},
    },
    'noak': {
        'economics': {'eta_in': 0.98, 'eta_out': 0.98, 'soft_cost_frac': 0.25},
        'costs': {
            'cell_per_cm2': 1.872, 'stick_per_m': 217.0, 'stick_fixed': 4510.0,
            'interconnect_per_kW': 257.6, 'charge_electrics_per_kW': 0.0, 'inverter_per_kW': 0.0,
            'heat_rejection_per_kWth': 496.0,
            'iso_graphite_per_kg': 2.0, 'heater_graphite_per_kg': None,
            'heater_fixed': 4500.0,
        },
        'parasitics': {'heat_rejection_kWe_per_kWth': 0.05},
        'tank_costs': {'storage_graphite_per_kg': 0.50, 'steel_per_t': 5030.0, 'deck_per_t': 5030.0,
                       'structural_steel_per_kg': 5.03},
    },
}

# what a scenario changes, for the notes in the run report
NOTES = {
    'pilot': "Pilot (2027): eta_in / eta_out 0.95; cells $36.5/cm2 all-in module cost (cells push the optimum "
             "toward high flux: consider raising the avg_flux bound).",
    'foak': "FOAK (2028 Phase 1). Storage 'graphite' at $0.75/kg is baked anode-type carbon: a different "
            "material (density, conductivity, purity), priced here only.",
    'noak': "NOAK (2034 Phase 3). Storage 'graphite' at $0.50/kg is baked anode-type carbon: a different "
            "material (density, conductivity, purity), priced here only.",
}
