import numpy as np
import copy

from .components import (
    Fluid, Pipe, Pump, HeatingModule, StorageModule, PowerConversionBlock,
    Sn_rho, Sn_cp,
)
from .properties import storage_rho, storage_cp


class SystemSimulation:
    def __init__(self, pump: Pump, heater: HeatingModule, storage: StorageModule, power_conversion: PowerConversionBlock, pipes: list,
                 heat_loss_bool, block_heat_loss_bool, bypass_bool, powerstick_bypass_bool, vary_flowrate_bool, print_outputs_bool, liner_bool, N_loops_step: int,
                 T_tin_max_heat: float):
        """
        Initialize the SystemSimulation object with its components.

        Parameters:
        pump (Pump): Pump object.
        heater (HeatingModule): HeatingModule object.
        storage (StorageModule): StorageModule object.
        power_conversion (PowerConversionBlock): PowerConversionBlock object.
        pipes (list): List of Pipe objects connecting the components.
        """
        self.pump = pump
        self.heater = heater
        self.storage = storage
        self.power_conversion = power_conversion
        self.pipes = pipes
        self.T_tin_max_heat = T_tin_max_heat
        self.heat_loss_bool = heat_loss_bool
        self.bypass_bool = bypass_bool
        self.powerstick_bypass_bool = powerstick_bypass_bool
        self.vary_flowrate_bool = vary_flowrate_bool
        self.N_loops_step = N_loops_step
        self.liner_bool = liner_bool
        self.block_heat_loss_bool = block_heat_loss_bool
        self.print_outputs_bool = print_outputs_bool

    def run_simulation(self, initial_fluids: list, end_time: int, heat_inputs: list, flow_rates: list, emissivity_tables: list,
                       T_sur: float, T_batt_empty: float, DT_batt: float, init_batt_state: float = 1.0):
        """
        Run the system simulation over the specified time steps.

        Parameters:
        initial_fluids (list): List of Fluid objects containing the initial conditions at each location.
        end_time (int): Total runtime for the simulation (s)
        heat_inputs (list): List of heat inputs for the heating module at each time step.
        flow_rates (list): List of flow rates for the pump at each time step.
        emissivity_tables (list): List of emissivity tables for the powerblock at each timestep
        T_sur (float): Ambient temperature for heat loss calcs [K]

        Returns:
        dict: Dictionary containing the simulation results.
        """
        results = {
            "time": [],
            "pump_power": [],
            "heater_temp": [],
            "storage_temp_dist": [],
            "storage_bypass": [],
            "power_generated": [],
            "waste_heat": [],
            "batt_level": [],
            "T_av_storage": [],
            "Q_storage": [],
            "Q_tin": [],
            "Pipe_heat_loss": [],
            "heater_power": [],
            "flowrate": [],
            "cell_flux": [],
            "fluid_conditions": {
                "pump_outlet": [],
                "heater_inlet": [],
                "heater_outlet": [],
                "storage_inlet": [],
                "storage_outlet": [],
                "power_conversion_inlet": [],
                "power_conversion_outlet": [],
                "pump_inlet": [],
            }
        }
        T_max_storage = T_batt_empty + DT_batt
        print(T_batt_empty, DT_batt)
        current_time = 0
        discharge_iter = 0
        last_discharge_bool = False
        batt_empty_bool = False
        iteration = 0
        last_storage_bypass_fraction = 0.5
        last_loop_steps = 1
        Q_tin = 0
        T_tin_max_ramp_rate = 0.3333
        T_av_block_init = np.mean(self.storage.block.temp_dist)
        V_storage = self.storage.N_pipes * np.pi * (self.storage.block.rad_out**2 - self.storage.block.rad_in**2) * self.storage.pipe.length
        Q_capacity_full = V_storage * DT_batt * storage_cp(T_av_block_init) * storage_rho(T_av_block_init) / 3.6e9
        batt_charge = init_batt_state * Q_capacity_full
        Q_store = 0
        Q_loss_tot = 0
        while current_time < end_time:
            Q_store_check = 0
            iteration += 1
            flowrate = flow_rates[current_time]
            heat_input = heat_inputs[current_time]
            emissivity_table = emissivity_tables[current_time]
            discharge_bool = heat_input < 10 and emissivity_table[0][1] > 0.01
            if discharge_bool != last_discharge_bool:
                discharge_iter = iteration
            self.power_conversion.powerstick.update_emissivity_table(emissivity_table)
            for fluid in initial_fluids:
                fluid.flow_rate = flowrate

            V_heat = self.heater.N_pipes * self.heater.pipe.volume
            V_store = self.storage.N_pipes * self.storage.pipe.volume
            V_PCB = 4 * self.power_conversion.N_cells * self.power_conversion.pipe.volume
            V_pipes = sum(pipe.volume for pipe in self.pipes)
            V_tot = V_heat + V_store + V_PCB + V_pipes
            mass_tot = V_tot * Sn_rho(initial_fluids[0].temperature)
            if iteration - discharge_iter < self.N_loops_step:
                loop_steps = 1
            else:
                loop_steps = self.N_loops_step

            time_step_duration = loop_steps * mass_tot / flowrate
            current_time += int(time_step_duration)
            if current_time > end_time:
                break

            if current_time == 0:
                P_out = initial_fluids[0].pressure + 1e4
            else:
                P_out = initial_fluids[0].pressure

            pump_in_fluid = Fluid(min(initial_fluids[-1].temperature + (last_loop_steps - 1) * Q_tin / (flowrate *
                    Sn_cp(initial_fluids[-1].temperature)), self.T_tin_max_heat), initial_fluids[-1].pressure, flowrate)
            last_loop_steps = loop_steps
            pump_power, fluid = self.pump.required_power(pump_in_fluid, P_out)
            results["pump_power"].append(pump_power)
            results["fluid_conditions"]["pump_outlet"].append(fluid)
            initial_fluids[0] = fluid
            if self.heat_loss_bool:
                Q_loss_0, fluid, _ = self.pipes[0].heat_loss(initial_fluids[0], T_sur)
            else:
                Q_loss_0 = 0.0
                fluid = initial_fluids[0]

            fluid = Fluid(fluid.temperature, fluid.pressure - self.pipes[0].pressure_drop(initial_fluids[0]), fluid.flow_rate)
            results["fluid_conditions"]["heater_inlet"].append(fluid)
            initial_fluids[1] = fluid
            T_av_storage = T_batt_empty + batt_charge * 3.6e9 / (V_storage * storage_rho(T_av_block_init) * storage_cp(T_av_block_init))
            results["T_av_storage"].append(T_av_storage)
            print("T_av_stor:", T_av_storage)
            T_storage_inner_first = self.storage.block.temp_dist[0][0]
            if self.vary_flowrate_bool and not discharge_bool:
                mdot_min = 0.2 * fluid.flow_rate
                mdot_max = fluid.flow_rate
                new_flowrate = min(mdot_max, max(mdot_min, heat_input / (Sn_cp(initial_fluids[1].temperature) * max(0.01, (self.T_tin_max_heat - initial_fluids[1].temperature)))))
                for fluid in initial_fluids:
                    fluid.flow_rate = new_flowrate
            last_discharge_bool = discharge_bool

            results["flowrate"].append(initial_fluids[1].flow_rate)
            Q_max_ramp_rate = mass_tot * Sn_cp(initial_fluids[1].temperature) * T_tin_max_ramp_rate + Q_store + Q_loss_tot
            Q_max_store = V_storage * storage_rho(T_av_storage) * storage_cp(T_av_storage) * (T_max_storage - T_av_storage) / time_step_duration
            Q_max_heat = initial_fluids[1].flow_rate * Sn_cp(initial_fluids[1].temperature) * (self.T_tin_max_heat - initial_fluids[1].temperature)
            Q_heat = max(0, min(Q_max_ramp_rate, Q_max_heat, Q_max_store, heat_input))

            self.heater.heat_input = Q_heat
            heater_temp, _, _ = self.heater.solve_heater_temperature(initial_fluids[1])
            results["heater_temp"].append(heater_temp)
            fluid, Q_heat = self.heater.outlet_condition(initial_fluids[1])
            print("Q_heat", Q_heat, Q_max_store, Q_max_heat, Q_max_ramp_rate, heat_input)
            results["heater_power"].append(Q_heat)
            results["fluid_conditions"]["heater_outlet"].append(fluid)
            initial_fluids[2] = fluid

            if self.heat_loss_bool:
                Q_loss_1, fluid, _ = self.pipes[1].heat_loss(initial_fluids[2], T_sur)
            else:
                Q_loss_1 = 0.0
                fluid = initial_fluids[2]

            fluid = Fluid(fluid.temperature, fluid.pressure - self.pipes[0].pressure_drop(initial_fluids[2]), fluid.flow_rate)
            results["fluid_conditions"]["storage_inlet"].append(fluid)
            initial_fluids[3] = fluid

            if discharge_bool and self.bypass_bool:
                if iteration > 1:
                    storage_bypass_fraction = self.storage.optim_bypass_ratio(0.5, initial_fluids[3], self.power_conversion.T_tin_max_PCB)
                else:
                    if initial_fluids[3].temperature > self.power_conversion.T_tin_max_PCB:
                        storage_bypass_fraction = self.storage.bypass_range[1]
                    else:
                        storage_bypass_fraction = 0.5
            else:
                storage_bypass_fraction = self.storage.bypass_range[0]

            initial_fluids[4], Q_store, Q_rad_dist = self.storage.bypass_outlet_condition(storage_bypass_fraction, initial_fluids[3])
            last_storage_bypass_frac = storage_bypass_fraction
            results["storage_bypass"].append(storage_bypass_fraction)
            results["storage_temp_dist"].append(self.storage.block.temp_dist)
            results["Q_storage"].append(Q_store)
            results["fluid_conditions"]["storage_outlet"].append(initial_fluids[4])

            sub_step_dur = 4.5
            num_sub_steps = int(np.ceil(time_step_duration / sub_step_dur))
            self.storage.block.temp_dist, Q_store_check_step, surf_loss_step = self.storage.update_temps(self.storage.block.temp_dist, Q_rad_dist, time_step_duration, num_sub_steps, T_sur, self.block_heat_loss_bool)
            Q_store_check = Q_store_check_step * self.storage.N_pipes / time_step_duration
            surf_loss = surf_loss_step * self.storage.N_pipes / time_step_duration
            print("surf_loss:", surf_loss)
            batt_charge += Q_store_check_step * self.storage.N_pipes / 3600000000
            results["batt_level"].append(batt_charge)
            if self.heat_loss_bool:
                Q_loss_2, fluid, _ = self.pipes[2].heat_loss(initial_fluids[4], T_sur)
            else:
                Q_loss_2 = 0.0
                fluid = initial_fluids[4]

            fluid = Fluid(fluid.temperature, fluid.pressure - self.pipes[0].pressure_drop(initial_fluids[4]), fluid.flow_rate)
            results["fluid_conditions"]["power_conversion_inlet"].append(fluid)
            initial_fluids[5] = fluid

            Q_cell = 0
            print(batt_empty_bool, self.powerstick_bypass_bool, discharge_bool)
            if batt_empty_bool == False:
                if batt_charge < 0 and discharge_bool:
                    batt_empty_bool = True
                    fluid = copy.deepcopy(initial_fluids[5])
                    electricity_generated = 0
                    waste_heat = 0
                if self.powerstick_bypass_bool and not discharge_bool:
                    power_conversion_bypass_fraction = power_conversion_bypass_fractions[t]
                    power_conversion_inlet_main = initial_fluids[5]
                    power_conversion_inlet_bypass = initial_fluids[5]
                    power_conversion_inlet_main.flow_rate *= (1 - power_conversion_bypass_fraction)
                    power_conversion_inlet_bypass.flow_rate *= power_conversion_bypass_fraction
                    power_conversion_outlet_main, electricity_generated, waste_heat = self.power_conversion.determine_outlet_conditions(power_conversion_inlet_main)
                    power_conversion_outlet_bypass = power_conversion_inlet_bypass
                    power_conversion_outlet_flow_rate = initial_fluids[5].flow_rate
                    power_conversion_outlet_temp = (power_conversion_outlet_main.temperature * power_conversion_outlet_main.flow_rate + power_conversion_outlet_bypass.temperature * power_conversion_outlet_bypass.flow_rate) / power_conversion_outlet_flow_rate
                    power_conversion_outlet_pressure = power_conversion_outlet_main.pressure
                    fluid = Fluid(power_conversion_outlet_temp, power_conversion_outlet_pressure, power_conversion_outlet_flow_rate)
                else:
                    fluid = self.power_conversion.determine_outlet_conditions(initial_fluids[5])
                    temperatures, electricity_generated, waste_heat, Q_cell = self.power_conversion.power_conversion(initial_fluids[5])
            elif discharge_bool:
                discharge_bool = False
                electricity_generated = 0
                waste_heat = 0

            else:
                fluid = copy.deepcopy(initial_fluids[5])
                electricity_generated = 0
                waste_heat = 0

            Q_out_PCB = electricity_generated + waste_heat
            results["power_generated"].append(electricity_generated)
            results["waste_heat"].append(waste_heat)
            results["cell_flux"].append(Q_cell)
            results["fluid_conditions"]["power_conversion_outlet"].append(fluid)
            initial_fluids[6] = fluid

            if self.heat_loss_bool:
                Q_loss_3, fluid, _ = self.pipes[3].heat_loss(initial_fluids[6], T_sur)
            else:
                Q_loss_3 = 0.0
                fluid = initial_fluids[6]

            fluid = Fluid(fluid.temperature, fluid.pressure - self.pipes[0].pressure_drop(initial_fluids[6]), fluid.flow_rate)
            results["fluid_conditions"]["pump_inlet"].append(fluid)
            Q_tin = (fluid.temperature - initial_fluids[7].temperature) * fluid.flow_rate * Sn_cp(fluid.temperature) / loop_steps
            results["Q_tin"].append(Q_tin)
            initial_fluids[7] = fluid

            Q_loss_tot = Q_loss_0 + Q_loss_1 + Q_loss_2 + Q_loss_3 + surf_loss
            results["Pipe_heat_loss"].append(Q_loss_tot)
            if self.print_outputs_bool:
                print("Iteration: ", iteration,
                      "Time: ", np.round(current_time / 3600, 3),
                      "Storage Bypass: ", np.round(storage_bypass_fraction, 3),
                      "heat_input: ", np.round(Q_heat, 0),
                      "flow rate: ", np.round(fluid.flow_rate, 3),
                      "T_heat_in:", np.round(initial_fluids[1].temperature, 2),
                      "T_heat_out:", np.round(initial_fluids[2].temperature, 2),
                      "T_av_storage", np.round(T_av_storage, 2),
                      "T_storage_inner_first", np.round(T_storage_inner_first, 2),
                      "Storage energy", np.round(batt_charge, 2),
                      "T_PCB_in:", np.round(initial_fluids[5].temperature, 2),
                      "T_PCB_out: ", np.round(initial_fluids[6].temperature, 2),
                      "Q_store: ", np.round(Q_store, 0),
                      "Q_store_check: ", np.round(Q_store_check, 0),
                      "Q_PCB: ", np.round(Q_out_PCB, 0),
                      "Q_tin:", np.round(Q_tin, 0),
                      "elec gen:", np.round(electricity_generated, 0))

            results["time"].append(current_time)

        return results
