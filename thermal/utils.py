import numpy as np
import yaml
import os
import shutil
import re
import matplotlib.pyplot as plt

from .components import GraphiteBlock
from .properties import storage_cp, storage_rho


def generate_time_series(range_definitions, time_steps):
    series = np.zeros(time_steps)
    for item in range_definitions:
        start, end = item['range']
        series[start:end+1] = item['value']
    return series


def generate_emissivity_table_series(range_definitions, time_steps):
    series = [[] for _ in range(time_steps)]
    for item in range_definitions:
        start, end = item['range']
        for i in range(start, end+1):
            series[i] = [(tuple(pair[0]), float(pair[1]), float(pair[2]), float(pair[3])) for pair in item['value']]
    return series


def generate_2d_temp_distribution(initial_temp, num_ax_segments, num_rad_segments):
    return np.full((num_ax_segments, num_rad_segments), initial_temp)


def create_new_folder(base_folder='plots', label='plot'):
    """Creates a new folder with an incremented number in the base folder."""
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)

    existing_folders = [f for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))]
    existing_numbers = [int(folder.split('_')[-1]) for folder in existing_folders if folder.startswith(f'{label}_')]
    new_number = max(existing_numbers, default=0) + 1

    new_folder = os.path.join(base_folder, f'{label}_{new_number}')
    os.makedirs(new_folder)

    return new_folder


def copy_yaml_file(folder, yaml_file='simulation_input_template.yaml'):
    """Copies the YAML file to the specified folder."""
    if os.path.exists(yaml_file):
        shutil.copy(yaml_file, folder)
    else:
        print(f"YAML file '{yaml_file}' not found.")


def save_input_config(file_path, input_data):
    """
    Save the input configuration to a YAML file.

    Parameters:
    file_path (str): Path to the input configuration file.
    input_data (dict): Input configuration data.
    """
    with open(file_path, 'w') as file:
        yaml.dump(input_data, file)


def save_results(file_path, results):
    """
    Save the simulation results to a YAML file.

    Parameters:
    file_path (str): Path to the results file.
    results (dict): Simulation results data.
    """
    with open(file_path, 'w') as file:
        yaml.dump(results, file)


def plot_temperature_distribution(temp_dist, time, length, r_inner, r_outer, vmin, vmax, filename):
    """
    Plot the temperature distribution as a 2D colormap.

    Parameters:
    temp_dist (2D array): Temperature distribution in the storage block (K).
    length (float): Length of the storage block (m).
    r_inner (float): Internal radius of the storage block (m).
    r_outer (float): External radius of the storage block (m).
    vmin (float): Minimum temperature value for the colormap.
    vmax (float): Maximum temperature value for the colormap.
    """
    temp_dist = np.transpose(temp_dist)
    num_rad_segments, num_ax_segments = temp_dist.shape
    dr = (r_outer - r_inner) / num_rad_segments
    dz = length / num_ax_segments
    r = np.linspace(r_inner + dr / 2, r_outer - dr / 2, num_rad_segments)
    z = np.linspace(dz / 2, length - dz / 2, num_ax_segments)
    R, Z = np.meshgrid(r, z, indexing='ij')
    cbarticks = np.arange(np.floor(vmin/100) * 100, np.ceil(vmax / 100) * 100, (np.ceil(vmax / 100) - np.floor(vmin/100)) * 5)
    plt.figure(figsize=(10, 6))
    plt.contourf(Z, R, temp_dist, cbarticks, cmap='hot', vmin=vmin, vmax=vmax)
    plt.colorbar(label='Temperature (C)', ticks=cbarticks)
    plt.xlabel('Axial Position (m)')
    plt.ylabel('Radial Position (m)')
    plt.title(f'Temperature Distribution in the Storage Block at t={time: .2f} hrs')
    plt.savefig(filename)
    plt.close()


def calculate_stored_energy(temp_dist_1: np.ndarray, temp_dist_2: np.ndarray, block: GraphiteBlock, num_rad_segments: int, num_ax_segments: int) -> float:
    """
    Calculate the total stored energy in the storage block given a temperature distribution.

    Parameters:
    temp_dist_1 (2D array): Reference temperature distribution (K).
    temp_dist_2 (2D array): Current temperature distribution (K).
    block (GraphiteBlock): GraphiteBlock object.
    num_rad_segments (int): Number of radial segments.
    num_ax_segments (int): Number of axial segments.

    Returns:
    float: Total stored energy in the storage block (J).
    """
    r_inner = block.rad_in
    r_outer = block.rad_out
    dr = (r_outer - r_inner) / num_rad_segments
    dz = block.length / num_ax_segments
    total_energy = 0.0

    for i in range(num_ax_segments):
        for j in range(num_rad_segments):
            r_in = r_inner + j * dr
            r_out = r_in + dr
            volume = np.pi * (r_out ** 2 - r_in ** 2) * dz
            dT = temp_dist_2[i, j] - temp_dist_1[i, j]
            energy_density = storage_cp(temp_dist_2[i, j]) * storage_rho(temp_dist_2[i, j]) * dT
            total_energy += energy_density * volume

    return total_energy


def process_results_matrix(results, end_time, target_state, V_storage, Q_block_full):
    """
    Process raw simulation results into plottable data streams.

    V_storage and Q_block_full must be passed explicitly (they are not globals here).
    """
    time = np.array(results["time"]) / 3600
    num_iterations = len(results["time"])
    data_streams = {
        "time": time,
        "pump_power": np.array(results["pump_power"]),
        "heater_temp": np.array(results["heater_temp"]) - 273.15,
        "heater_power": np.array(results["heater_power"]) / 1000,
        "power_generated": np.array(results["power_generated"]) / 1000,
        "waste_heat": np.array(results["waste_heat"]) / 1000,
        "pipe_heat_loss": np.array(results["Pipe_heat_loss"]) / 1000,
        "Q_storage": np.array(results["Q_storage"]) / 1000,
        "Q_tin": np.array(results["Q_tin"]) / 1000,
        "Batt_level": np.array(results["batt_level"]),
    }
    data_streams["innermost_radial_last_axial_temp"] = np.array([results["storage_temp_dist"][i][-1, 0] for i in range(num_iterations)]) - 273.15
    data_streams["average_storage_temp"] = np.array([np.mean(results["storage_temp_dist"][i]) for i in range(num_iterations)]) - 273.15

    iter_full = np.where(np.array(results["batt_level"]) / Q_block_full > target_state)[0]
    time_full = np.array(data_streams["time"])[iter_full[0]] if iter_full.size > 0 else 0
    print("batt charge time: ", time_full)

    iteration_durations = np.diff(np.array(results["time"]))
    iteration_durations = np.append(iteration_durations, end_time - results["time"][-1])
    Q_block_init = 0
    data_streams["Q_storage_integral"] = (Q_block_init + np.cumsum(data_streams["Q_storage"] * iteration_durations)) / Q_block_full
    data_streams["Electricity_integral"] = np.cumsum(data_streams["power_generated"] * iteration_durations)
    data_streams["Batt_level"] = (Q_block_init + data_streams["Batt_level"]) / Q_block_full

    return data_streams, time_full


def downsample_data(data_streams, num_points=100):
    indices = np.linspace(0, len(data_streams["time"]) - 1, num_points).astype(int)
    downsampled_streams = {}
    for key, value in data_streams.items():
        if isinstance(value, np.ndarray) and len(value) > 1:
            downsampled_streams[key] = value[indices]
        else:
            downsampled_streams[key] = value
    return downsampled_streams


def generate_handles(vary_param, param_values, fixed_params, all_results):
    handles = []
    for val in param_values:
        for result in all_results:
            if precise_match(result["label"], {**fixed_params, vary_param: val}):
                handles.append(plt.Line2D([0], [0], linestyle=result["line_style"], color=result["color"],
                                          marker=result["shape"], markersize=result["size"], label=f'{vary_param}={val}'))
                break
    return handles


def precise_match(label, fixed_params):
    for k, v in fixed_params.items():
        pattern = rf"{k}={v},"
        if not re.search(pattern, label):
            return False
    return True


def plot_results(all_params, vary_param, file_suffix, all_results, param_values, new_folder):
    fixed_params = {k: v for k, v in all_params.items() if k != vary_param}
    fixed_params_str = ', '.join([f"{k}={v}" for k, v in fixed_params.items()])

    plt.figure()
    first_entry = True
    for result in all_results:
        if precise_match(result["label"], fixed_params):
            time = result["data_streams"]["time"]
            Batt_level = result["data_streams"]["Batt_level"]
            plt.plot(time, Batt_level, linestyle=result["line_style"], color=result["color"],
                     marker=result["shape"], markersize=result["size"], label=result["label"] if first_entry else "")
            first_entry = False

    handles = generate_handles(vary_param, param_values, fixed_params, all_results)
    plt.legend(handles=handles, loc='best')
    plt.xlabel("Time (hours)")
    plt.ylabel("Battery Charge State")
    plt.title(f"Battery State: {fixed_params_str}")
    plt.savefig(os.path.join(new_folder, f'Battery_Charge_State_{file_suffix}.png'))

    plt.figure()
    first_entry = True
    for result in all_results:
        if precise_match(result["label"], fixed_params):
            print(f"Matched result with label: {result['label']}")
            time = result["data_streams"]["time"]
            Q_powerstick = result["data_streams"]["power_generated"] + result["data_streams"]["waste_heat"]
            plt.plot(time, Q_powerstick, linestyle=result["line_style"], color=result["color"],
                     marker=result["shape"], markersize=result["size"], label=result["label"] if first_entry else "")
            first_entry = False

    plt.legend(handles=handles, loc='best')
    plt.xlabel("Time (hours)")
    plt.ylabel("Q_powerstick")
    plt.title("Q_powerstick vs. Time")
    plt.savefig(os.path.join(new_folder, f'Q_powerstick_{file_suffix}.png'))


def plot_battery_full_times(time_full, heating_rates, num_storage_blocks, block_in_rads, output_folder, input_data, linestyles, colors):
    """
    Plot the times at which the battery becomes full for each condition.

    Parameters:
        time_full (numpy.ndarray): Array of times (indexed by heating rate and block in radius).
        heating_rates (list): List of heating rates.
        num_storage_blocks (list): List of storage block counts.
        block_in_rads (list): List of block inner radii.
        output_folder (str): Folder to save the output plot.
        input_data (dict): Input configuration data (for pipe OD).
        linestyles (list): Line style list.
        colors (list): Color list.
    """
    plt.figure()
    for i, heating_rate in enumerate(heating_rates):
        for j, num_blocks in enumerate(num_storage_blocks):
            times = time_full[i, j]
            print(heating_rate, num_blocks, times)
            area_ratios = np.array(block_in_rads) / (0.5 * input_data["storage"]["Pipe"]["OD"])
            valid_indices = times > 0
            filtered_times = times[valid_indices]
            filtered_AR = np.array(area_ratios)[valid_indices]
            plt.plot(filtered_AR, filtered_times, label=f'HR={heating_rate}, SB={num_blocks}',
                     marker='o', linestyle=linestyles[j], color=colors[i])

    plt.xlabel("Area Ratio")
    plt.ylabel("Time to Full Battery (hours)")
    plt.title("Time to Full Battery vs. Area Ratio")
    plt.legend(fontsize=8, loc='best')
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, 'Time_to_Full_Battery.png'))
    print(f"Plot saved as 'Time_to_Full_Battery.png' in {output_folder}")


def is_fully_charged(results, target_state, V_storage, DT_batt):
    """
    Check if the system reaches the full charge condition.

    V_storage and DT_batt must be passed explicitly.
    """
    final_batt_level = np.array(results["batt_level"])[-1] / (V_storage * DT_batt * storage_cp(2448) * storage_rho(2448) / 3.6e9)
    print("final batt level: ", final_batt_level)
    return final_batt_level >= target_state


def find_min_heating_rate(simulation, initial_fluids, end_time, flow_rates, emissivity_tables, T_sur,
                          time_steps, input_data, target_state, V_storage, DT_batt,
                          generate_simulation_fn):
    """
    Binary search to find the minimum heating rate to fully charge the system.

    generate_simulation_fn: callable(mid, input_data) -> (simulation, initial_fluids, heat_inputs, storage)
    that rebuilds the simulation objects for each candidate heating rate.
    """
    low = 1e5
    high = 1e6
    tolerance = 1e3
    best_heating_rate = None

    while high - low > tolerance:
        mid = (low + high) / 2
        sim, fluids, heat_inputs, storage = generate_simulation_fn(mid, input_data)
        results = sim.run_simulation(fluids, end_time, heat_inputs, flow_rates, emissivity_tables, T_sur)
        Q_block_full = V_storage * 500 * storage_cp(2448) * storage_rho(2448) / 3.6e9
        final_batt_level = np.array(results["batt_level"])[-1] / Q_block_full
        print(low, mid, high, final_batt_level, target_state)

        if is_fully_charged(results, target_state, V_storage, DT_batt):
            best_heating_rate = mid
            high = mid
        else:
            error = target_state - final_batt_level
            step_size = max(tolerance, abs(error) * 1e5)
            low = min(mid + step_size, high)

    if best_heating_rate is None:
        print(f"Warning: No heating rate found to fully charge the system within the allowable range.")
        return high, results

    return best_heating_rate or low, results
