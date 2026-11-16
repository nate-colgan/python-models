import numpy as np
import copy

from .properties import (
    Sn_rho, Sn_mu, Sn_cp, Sn_tcx, Sn_Pr,
    gr_rho, gr_cp, gr_emis, gr_tcx,
    w_emis_spectral, w_tcx, tac_emis_temp,
    storage_rho, storage_cp, storage_tcx, ins_tcx,
    panel_efficiency,
)
from .correlations import (
    CircPipeFF, RectPipeFF, CircPipeNu, RectPipeNu,
    RprimeCondCyl, RCondWall, RprimeConvCyl, RConvWall,
    Rrad, Rrad_2surfaces,
)


class Fluid:
    def __init__(self, temperature: float, pressure: float, flow_rate: float):
        """
        Initialize the Fluid object with its properties.

        Parameters:
        temperature (float): Temperature of the fluid (K).
        pressure (float): Pressure of the fluid (Pa).
        flow_rate (float): Flow rate of the fluid (kg/s).
        """
        self.temperature = temperature
        self.pressure = pressure
        self.flow_rate = flow_rate

class Pipe:
    def __init__(self, OD: float, length: float, wall_thickness: float, roughness: float, th_ins: float):
        """
        Initialize the Pipe object with its physical properties.

        Parameters:
        diameter (float): Inner diameter of the pipe (meters).
        length (float): Length of the pipe (meters).
        wall_thickness (float): Wall thickness of the pipe (meters).
        roughness (float): Surface roughness of the pipe (meters).
        th_ins (float): Insulation thickness
        """
        self.OD = float(OD)
        self.length = float(length)
        self.wall_thickness = float(wall_thickness)
        self.roughness = float(roughness)
        self.ID = self.OD - 2 * self.wall_thickness
        self.A_s = np.pi * self.OD * self.length
        self.A_c = self.OD * self.length
        self.th_ins = th_ins
        self.volume = self.length * np.pi * self.ID ** 2 / 4

    def pressure_drop(self, fluid: Fluid) -> float:
        """
        Calculate the pressure drop of the pipe using the Darcy-Weisbach equation.

        Parameters:
        fluid (Fluid): Fluid object containing the flow conditions.

        Returns:
        float: Pressure drop (Pa).
        """
        mdot = fluid.flow_rate
        fluid_viscosity = Sn_mu(fluid.temperature)
        fluid_density = Sn_rho(fluid.temperature)
        Vdot = mdot / fluid_density
        velocity = Vdot / (np.pi * (self.ID / 2) ** 2)
        reynolds_number = 4 * mdot / (np.pi * fluid_viscosity * self.ID)
        friction_factor = CircPipeFF(reynolds_number, self.ID, self.roughness)
        pressure_drop = 0.5 * fluid_density * velocity ** 2 * friction_factor * self.length / self.ID
        return pressure_drop

    def heat_loss(self, fluid: Fluid, T_sur: float, num_shells: int = 10, tol: float = 1e-3, max_iter: int = 100) -> tuple:
        """
        Calculate the heat loss of the pipe to ambient.

        Parameters:
        fluid (Fluid): Fluid object containing the flow conditions.
        T_sur (float): Surrounding temperature (K).
        num_shells (int): Number of discretized shells for the insulation.
        tol (float): Tolerance for convergence.
        max_iter (int): Maximum number of iterations.

        Returns:
        tuple: Heat loss (W) and Fluid object containing the outlet conditions.
        """
        mdot = fluid.flow_rate
        T_in = fluid.temperature
        fluid_viscosity = Sn_mu(T_in)
        fluid_density = Sn_rho(T_in)
        fluid_cp = Sn_cp(T_in)
        fluid_tcx = Sn_tcx(T_in)
        h_out = 100  #pipe surface heat transfer coefficient estimate, W/m^2K
        r_inner = self.OD / 2
        r_outer = r_inner + self.th_ins
        dr = (r_outer - r_inner) / num_shells

        reynolds_number = 4 * mdot / (np.pi * fluid_viscosity * self.ID)
        friction_factor = CircPipeFF(reynolds_number, self.ID, self.roughness)
        Nu_pipe = CircPipeNu(reynolds_number, friction_factor, Sn_Pr(T_in))
        R_cond_pipe = RprimeCondCyl(self.ID, self.OD, gr_tcx(T_in)) / self.length
        R_conv_in = RprimeConvCyl(Nu_pipe, fluid_tcx, self.ID) / self.length
        R_conv_out = 1 / (h_out * np.pi * (self.OD + 2 * self.th_ins) * self.length)
        T_shells = np.linspace(T_in, T_sur + (T_in - T_sur) / num_shells, num_shells)

        for iteration in range(max_iter):
            R_shells = temperatures = np.zeros(num_shells)
            R_tin_to_shells = temperatures = np.zeros(num_shells)
            T_old = np.copy(T_shells)
            R_cond_ins = 0
            for j in range(num_shells):
                r_i = r_inner + j * dr
                R_shells[j] = RprimeCondCyl(2 * r_i, 2 * (r_i + dr), ins_tcx(T_shells[j])) / self.length

            R_cond_ins = sum(R_shells)
            R_tot = R_cond_pipe + R_cond_ins + R_conv_in + R_conv_out
            Q_loss = (T_in - T_sur) / R_tot
            for j in range(num_shells):
                R_tin_to_shells[j] = R_conv_in + R_cond_pipe + sum(R_shells[0:j])
                T_shells[j] = T_in - Q_loss * R_tin_to_shells[j]

            if np.max(np.abs(T_shells - T_old)) < tol:
                break

        T_out = T_in - Q_loss / (mdot * fluid_cp)
        return Q_loss, Fluid(T_out, fluid.pressure, mdot), T_shells


class RectangularPipe:
    def __init__(self, H: float, W: float, length: float, roughness: float):
        """
        Initialize the RectangularPipe object with its physical properties.

        Parameters:
        H (float): Height of the rectangular pipe (m).
        W (float): Width of the rectangular pipe (m).
        length (float): Length of the pipe (m).
        roughness (float): Roughness of the pipe (m).
        """
        self.H = H
        self.W = W
        self.length = length
        self.roughness = roughness
        self.D_h = 2 * H * W / (H + W)
        self.A_s = 2 * (H + W) * length
        self.volume = self.length * self.H * self.W

    def pressure_drop(self, fluid: Fluid) -> float:
        """
        Calculate the pressure drop of the pipe.

        Parameters:
        fluid (Fluid): Fluid object containing the flow conditions.

        Returns:
        float: Pressure drop (Pa).
        """
        mdot = fluid.flow_rate
        fluid_viscosity = Sn_mu(fluid.temperature)
        fluid_density = Sn_rho(fluid.temperature)
        velocity = mdot / (fluid_density * self.H * self.W)
        Re = (fluid_density * velocity * self.D_h) / fluid_viscosity
        ff = RectPipeFF(Re, self.H, self.W, self.roughness)
        dp = ff * (self.length / self.D_h) * (fluid_density * velocity ** 2) / 2
        return dp


class GraphiteBlock:
    def __init__(self, initial_temp_distribution: np.ndarray, inner_radius: float, outer_radius: float, length: float, th_ins: float):
        """
        Initialize the GraphiteBlock object with its physical properties.

        Parameters:
        initial_temp_distribution (array): Initial temperature distribution in the block (K).
        inner_radius (float): Inner radius of the block (meters).
        outer_radius (float): Outer radius of the block (meters).
        length (float): Length of the block (meters).
        """
        self.temp_dist = initial_temp_distribution.astype(float)
        self.rad_in = float(inner_radius)
        self.rad_out = float(outer_radius)
        self.length = float(length)
        self.th_ins = float(th_ins)


class PowerStick:
    def __init__(self, axial_steps: int, width: float, height: float, cooling_temp: float, f_bare: float, f_active: float, liner_bool: bool, liner_width: float = None, emissivity_table: list = None):
        """
        Initialize the PowerStick object with its physical properties.

        Parameters:
        width (float): Width of the power stick (m).
        height (float): Height of the power stick (m).
        cooling_temp (float): Cooling temperature to maintain the power stick (K).
        f_bare (float): Fraction of powerstick area not covered by SNGC
        f_active (float): Fraction of SNGC-covered area that produces electricity
        liner_bool (bool): Whether a tungsten liner is present.
        liner_width (float): Square side length of the tungsten liner [m]; r_liner = 2*liner_width/pi.
        emissivity_table (list): List of tuples containing wavelength ranges and corresponding emissivities.
        """
        self.num_ax_steps = axial_steps
        self.width = width
        self.height = height
        self.height_step = self.height / self.num_ax_steps
        self.A_step = 4 * self.height_step * self.width
        self.cooling_temp = cooling_temp
        self.emissivity_table = emissivity_table if emissivity_table is not None else []
        self.f_bare = f_bare
        self.f_active = f_active
        self.liner_bool = liner_bool
        self.liner_width = liner_width

    def update_emissivity_table(self, new_emissivity_table):
        self.emissivity_table = new_emissivity_table

    def generate_electricity_step(self, T_emitter: float, view_factor: float) -> tuple:
        """
        Calculate the electricity generated by the power stick based on incident energy.

        Panel efficiency is computed per-step from the emitter temperature and the cell
        bandgap wavelength using the parabolic fit in panel_efficiency(), with a
        flux-dependent degradation factor: η = η_0 · 0.998^(2 · flux · η_0).

        Parameters:
        T_emitter (float): Temperature of the emitter surface (K).
        view_factor (float): View factor between the emitter and the powerstick.

        Returns:
        tuple: (electricity [W], waste_heat [W], cell_flux [W])
        """
        incident_energy_bins, emissivity_bins, incident_photon_bins = self.incident_energy(T_emitter, view_factor)

        bandgap = self.emissivity_table[0][0][1]
        absorbed_energy_below_bandgap = 0
        absorbed_energy_above_bandgap = 0
        absorbed_energy_bare = 0
        absorbed_energy_inactive = 0
        elec_flux = 0
        e_charge = 1.60218e-19  #electron charge, Coulombs
        P_collect = 1  #probability of photon above bandgap being absorbed
        V_panel = 1  #voltage on panel, update

        for i, (wavelength_range, emis_active, emis_bare, emis_inactive) in enumerate(self.emissivity_table):
            avg_panel_emis = self.f_bare * emis_bare + (1 - self.f_bare) * self.f_active * emis_active + (1 - self.f_bare) * (1 - self.f_active) * emis_inactive
            absorbed_energy_bare += incident_energy_bins[wavelength_range] * self.f_bare * emis_bare / avg_panel_emis
            absorbed_energy_inactive += incident_energy_bins[wavelength_range] * (1 - self.f_bare) * (1 - self.f_active) * emis_inactive / avg_panel_emis
            if i == 0:  # First bin is below the bandgap
                absorbed_energy_below_bandgap += incident_energy_bins[wavelength_range] * (1 - self.f_bare) * self.f_active * emis_active / avg_panel_emis
                elec_flux += incident_photon_bins[wavelength_range] * P_collect * e_charge * V_panel
            elif i == 1:  # Second bin is above the bandgap
                absorbed_energy_above_bandgap += incident_energy_bins[wavelength_range] * (1 - self.f_bare) * self.f_active * emis_active / avg_panel_emis

        absorbed_heat = (absorbed_energy_below_bandgap + absorbed_energy_above_bandgap + absorbed_energy_bare + absorbed_energy_inactive) * self.A_step
        cell_flux = (absorbed_energy_below_bandgap + absorbed_energy_above_bandgap) * self.A_step
        flux_wcm2 = cell_flux / (self.A_step * (1 - self.f_bare) * self.f_active * 1e4)
        eff = panel_efficiency(bandgap, T_emitter, flux=flux_wcm2)
        electricity = cell_flux * eff
        waste_heat = absorbed_heat - electricity
        return electricity, waste_heat, cell_flux

    def incident_energy(self, T_emitter: float, view_factor: float) -> tuple:
        """
        Calculate the total amount of incident energy on the powerstick surface in each energy bin.

        Parameters:
        T_emitter (float): Temperature of the emitter surface (K).
        view_factor (float): View factor between the emitter and the powerstick.

        Returns:
        tuple: Total amount of incident energy in each energy bin (W/m^2), and corresponding powerstick emissivities.
        """
        h = 6.62607015e-34  # Planck constant (J·s)
        c = 2.99792458e8    # Speed of light (m/s)
        k = 1.380649e-23    # Boltzmann constant (J/K)

        incident_energy_bins = {range_: 0.0 for range_, _, _, _ in self.emissivity_table}
        incident_photon_bins = {range_: 0.0 for range_, _, _, _ in self.emissivity_table}
        emissivity_bins = {range_: emissivity for range_, emissivity, _, _ in self.emissivity_table}

        wavelength_min = self.emissivity_table[0][0][0]
        wavelength_max = self.emissivity_table[1][0][1]
        band_gap = self.emissivity_table[0][0][1]
        n_bins = 300

        wavelengths = np.logspace(np.log10(wavelength_min), np.log10(wavelength_max), n_bins)
        wavelengths_m = wavelengths * 1e-6  # Convert to meters
        spectral_radiance = np.pi * (2 * h * c**2) / (wavelengths_m**5) * (1 / (np.expm1((h * c) / (wavelengths_m * k * T_emitter))))
        spectral_count = spectral_radiance / (h * c / wavelengths_m)

        for i, wavelength in enumerate(wavelengths):
            if self.liner_bool:
                #emitter_emis = w_emis_spectral(T_emitter, wavelength)
                emitter_emis = tac_emis_temp(T_emitter)
            else:
                emitter_emis = gr_emis(T_emitter)
            if wavelength < band_gap:
                band_width = wavelengths_m[i+1] - wavelengths_m[i]
            else:
                band_width = wavelengths_m[i] - wavelengths_m[i-1]

            for (wavelength_range, emis_active, emis_bare, emis_inactive) in self.emissivity_table:
                if wavelength_range[0] <= wavelength < wavelength_range[1]:
                    avg_panel_emis = self.f_bare * emis_bare + (1 - self.f_bare) * self.f_active * emis_active + (1 - self.f_bare) * (1 - self.f_active) * emis_inactive
                    incident_energy_bins[tuple(wavelength_range)] += (
                        spectral_radiance[i] * (band_width) / (view_factor *
                                    (1 - emitter_emis) / emitter_emis + 1 + (1 - avg_panel_emis) / avg_panel_emis)
                    )
                    incident_photon_bins[tuple(wavelength_range)] += spectral_count[i]
                    break

        return incident_energy_bins, emissivity_bins, incident_photon_bins


class Emitter:
    def __init__(self, width: float, height: float, wall_thickness: float, pipe: RectangularPipe, liner_bool: bool):
        """
        Initialize the Emitter object with its physical properties.

        Parameters:
        width (float): Width of the emitter (m).
        height (float): Height of the emitter (m).
        wall_thickness (float): Thickness of the graphite wall (m).
        pipe (RectangularPipe): Rectangular pipe object liquid tin channels
        """
        self.width = width
        self.height = height
        self.wall_thickness = wall_thickness
        self.pipe = pipe
        self.liner_bool = liner_bool

    def R_tin_to_tungsten(self, fluid: Fluid, T_graphite: float, T_tungsten: float):
        """
        Calculate thermal resistance per meter between liquid tin and tungsten liner.

        Parameters:
        fluid (Fluid): Fluid object containing the flow conditions.
        T_graphite (float): Graphite surface temperature guess (K)
        T_tungsten (float): Tungsten liner temperature guess (K)
        """
        mdot = fluid.flow_rate
        T_tin = fluid.temperature
        A_emitter = 4 * self.width  #emitter area per meter length
        u = mdot / (4 * Sn_rho(T_tin) * self.pipe.W * self.pipe.H)
        Re = Sn_rho(T_tin) * u * self.pipe.D_h / Sn_mu(T_tin)
        ff_pipe = RectPipeFF(Re, self.pipe.H, self.pipe.W, self.pipe.roughness)
        Nu_pipe = RectPipeNu(Re, self.pipe.H, self.pipe.W, ff_pipe, Sn_Pr(T_tin))
        h_conv = Nu_pipe * Sn_tcx(T_tin) / self.pipe.D_h
        R_conv = 1 / (h_conv * A_emitter)

        R_cond_graphite = self.wall_thickness / (gr_tcx(T_graphite) * A_emitter)
        R_rad_graphite_tungsten = Rrad_2surfaces(T_graphite, T_tungsten, A_emitter, A_emitter,
                                                     gr_emis(T_graphite), w_emis_spectral(T_tungsten, 1.0), 1.0)
        if self.liner_bool:
            R_tot = R_conv + R_cond_graphite + R_rad_graphite_tungsten
        else:
            R_tot = R_conv + R_cond_graphite
        return R_tot, R_conv, R_cond_graphite, R_rad_graphite_tungsten


class HeatingModule:
    def __init__(self, heat_input: float, N_pipes: int, N_heat: int, H_panel: float, H_heat: float, Pipe: Pipe, simple_heat: bool, max_temperature: float):
        """
        Initialize the HeatingModule object with its physical properties.

        Parameters:
        heat_input (float): Heat input to the heating module (W).
        N_pipes (int): Number of pipes in the heating module.
        Pipe (Pipe): Pipe object representing the physical properties of the pipe.
        max_temperature (float): Maximum allowable heater temperature (K).
        """
        self.heat_input = heat_input
        self.N_pipes = N_pipes
        self.N_heat = N_heat
        self.H_panel = H_panel
        self.H_heat = H_heat
        self.pipe = Pipe
        self.max_temperature = max_temperature
        self.simple_heat = simple_heat

    def heat_transfer_residual(self, T_heat: float, fluid: Fluid) -> float:
        """
        Calculate heat transfer to liquid tin for a guessed heater element temperature.

        Parameters:
        T_heat (float): Temperature of the heater (K).
        fluid (Fluid): Fluid object containing the inlet conditions.

        Returns:
        float: Residual (W).
        """
        T_heat = float(T_heat)
        T_tin_in = fluid.temperature
        mdot = fluid.flow_rate
        VF = 1  #using Tyler's heater model of rectangular heaters and tin manifold plate
        A_heat = self.pipe.length * 2 * self.H_heat * self.N_heat
        A_panel = self.pipe.length * self.H_panel * 2 * self.N_heat
        Cdot_in = mdot * Sn_cp(T_tin_in)
        T_tin_av = T_tin_in + self.heat_input / (2 * Cdot_in)
        Re = 4 * mdot / (self.N_pipes * np.pi * Sn_mu(T_tin_av) * self.pipe.ID)
        ff_pipe = CircPipeFF(Re, self.pipe.ID, self.pipe.roughness)
        Nu_pipe = CircPipeNu(Re, ff_pipe, Sn_Pr(T_tin_av))
        R_cond = RprimeCondCyl(self.pipe.ID, self.pipe.OD, gr_tcx(T_tin_av)) / self.pipe.length / self.N_pipes
        R_conv = RprimeConvCyl(Nu_pipe, Sn_tcx(T_tin_av), self.pipe.ID) / self.pipe.length / self.N_pipes

        T_pipe_surface = T_tin_in

        for _ in range(100):
            R_rad = Rrad_2surfaces(T_heat, T_pipe_surface, A_heat, A_panel, gr_emis(T_heat), gr_emis(T_pipe_surface), VF)
            R_tot = R_cond + R_conv + R_rad
            T_pipe_surface_new = T_tin_in + self.heat_input * (R_cond + R_conv)

            if abs(T_pipe_surface_new - T_pipe_surface) < 1e-2:
                break

            T_pipe_surface = T_pipe_surface_new

        UA = 1 / R_tot
        Cdot = mdot * Sn_cp(T_tin_av)
        NTU = UA / Cdot
        eps_hx = 1 - np.exp(-1 * NTU)
        Q_max = Cdot * (T_heat - T_tin_in)
        Q_calc = Q_max * eps_hx
        residual = Q_calc - self.heat_input
        return residual

    def solve_heater_temperature(self, fluid: Fluid, abs_tol: float = 1e-3, max_iter: int = 100) -> float:
        """
        Iteratively solve for the heater temperature using the Newton-Raphson method.

        Parameters:
        fluid (Fluid): Fluid object containing the inlet conditions.
        abs_tol (float): Tolerance for the residual (W).
        max_iter (int): Maximum number of iterations.

        Returns:
        float: Solved heater temperature (K).
        """
        if self.simple_heat:
            return 0.0, 0.0, 0.0
        else:
            T_tin_in = fluid.temperature
            mdot = fluid.flow_rate
            heater_temp_guess = T_tin_in + 49
            for i in range(max_iter):
                residual = self.heat_transfer_residual(heater_temp_guess, fluid)
                if abs(residual) < abs_tol:
                    if heater_temp_guess > self.max_temperature:
                        self.heat_input = max(0.0, self.heat_input + self.heat_transfer_residual(self.max_temperature, fluid))
                        heater_temp_guess = self.max_temperature
                        T_pipe_outer, T_pipe_inner = self.calculate_pipe_temperatures(self.max_temperature, fluid)

                    T_pipe_outer, T_pipe_inner = self.calculate_pipe_temperatures(heater_temp_guess, fluid)
                    return heater_temp_guess, T_pipe_outer, T_pipe_inner
                delta_temp = 1e-3
                residual_plus = self.heat_transfer_residual(heater_temp_guess + delta_temp, fluid)
                derivative = (residual_plus - residual) / delta_temp
                heater_temp_guess -= residual / derivative

            raise RuntimeError("Failed to converge to a solution.")

    def calculate_pipe_temperatures(self, T_heat: float, fluid: Fluid) -> tuple:
        """
        Calculate the pipe surface and inner temperatures.

        Parameters:
        T_heat (float): Temperature of the heater (K).
        fluid (Fluid): Fluid object containing the inlet conditions.

        Returns:
        tuple: Pipe outer temperature (K) and pipe inner temperature (K).
        """
        T_heat = float(T_heat)
        T_tin_in = fluid.temperature
        mdot = fluid.flow_rate
        VF = 0.6928
        Cdot_in = mdot * Sn_cp(T_tin_in)
        T_tin_av = T_tin_in + self.heat_input / (2 * Cdot_in)
        Re = 4 * mdot / (self.N_pipes * np.pi * Sn_mu(T_tin_av) * self.pipe.ID)
        ff_pipe = CircPipeFF(Re, self.pipe.ID, self.pipe.roughness)
        Nu_pipe = CircPipeNu(Re, ff_pipe, Sn_Pr(T_tin_av))
        R_cond = RprimeCondCyl(self.pipe.ID, self.pipe.OD, gr_tcx(T_tin_av)) / self.pipe.length / self.N_pipes
        R_conv = RprimeConvCyl(Nu_pipe, Sn_tcx(T_tin_av), self.pipe.ID) / self.pipe.length / self.N_pipes
        T_pipe_surface = T_tin_in

        for _ in range(100):
            emissivity = gr_emis(T_pipe_surface)
            R_rad = Rrad(T_heat, T_pipe_surface, self.pipe.A_s * self.N_pipes, VF, emissivity)
            R_tot = R_cond + R_conv + R_rad
            T_pipe_surface_new = T_tin_in + self.heat_input * R_conv

            if abs(T_pipe_surface_new - T_pipe_surface) < 1e-6:
                break

            T_pipe_surface = T_pipe_surface_new

        T_pipe_outer = T_heat - self.heat_input * R_rad
        T_pipe_inner = T_pipe_outer - self.heat_input * R_cond

        return T_pipe_outer, T_pipe_inner

    def outlet_condition(self, fluid: Fluid) -> Fluid:
        """
        Calculate heater outlet conditions.

        Parameters:
        fluid (Fluid): Fluid object containing the inlet conditions.

        Returns:
        Fluid: Fluid object containing the outlet conditions.
        """
        T_tin_in = fluid.temperature
        P_tin_in = fluid.pressure
        mdot = fluid.flow_rate
        T_tin_out = T_tin_in + self.heat_input / (mdot * Sn_cp(T_tin_in))
        dP_tin = self.pipe.pressure_drop(fluid) / self.N_pipes
        P_tin_out = P_tin_in - dP_tin
        return Fluid(T_tin_out, P_tin_out, mdot), self.heat_input


class StorageModule:
    def __init__(self, N_pipes: int, Pipe: Pipe, Block: GraphiteBlock, num_ax_segments: int, num_rad_segments: int, bypass_range: list):
        """
        Initialize the StorageModule object with its physical properties.

        Parameters:
        N_pipes (int): Number of pipes in the storage module.
        Pipe (Pipe): Pipe object representing the physical properties of the pipe.
        Block (GraphiteBlock): GraphiteBlock object representing the physical properties of the storage block.
        num_ax_segments (int): Number of axial segments in the storage block.
        num_rad_segments (int): Number of radial segments in the storage block.
        bypass_range (list): Lower and upper bypass ratio limits
        """
        self.pipe = Pipe
        self.N_pipes = N_pipes
        self.block = Block
        self.num_ax_segments = num_ax_segments
        self.segment_length = self.pipe.length / num_ax_segments
        self.num_rad_segments = num_rad_segments
        self.bypass_range = bypass_range

    def heat_transfer_rate(self, fluid: Fluid, T_block_surface: float) -> float:
        """
        Calculate the heat transfer rate from the pipe segment to the storage block surface.

        Parameters:
        fluid (Fluid): Fluid object containing the inlet conditions to each block.
        T_block_surface (float): Temperature of the storage block surface (K).

        Returns:
        float: Radiative heat transfer rate (W).
        """
        T_tin = fluid.temperature
        mdot = fluid.flow_rate
        T_block_surface = float(T_block_surface)
        VF = 1.0

        A_pipe = np.pi * self.pipe.OD * self.segment_length
        A_block_surface = np.pi * self.block.rad_in * 2 * self.segment_length
        Re = 4 * mdot / (np.pi * Sn_mu(T_tin) * self.pipe.ID)
        ff_pipe = CircPipeFF(Re, self.pipe.ID, self.pipe.roughness)
        Nu_pipe = CircPipeNu(Re, ff_pipe, Sn_Pr(T_tin))
        R_cond = RprimeCondCyl(self.pipe.ID, self.pipe.OD, gr_tcx(T_tin)) / self.segment_length
        R_conv = RprimeConvCyl(Nu_pipe, Sn_tcx(T_tin), self.pipe.ID) / self.segment_length

        T_pipe_surface = T_tin
        C_min = fluid.flow_rate * Sn_cp(fluid.temperature)

        for _ in range(100):
            emis_pipe = gr_emis(T_pipe_surface)
            emis_block = gr_emis(T_block_surface)
            R_rad = Rrad_2surfaces(T_pipe_surface, T_block_surface, A_pipe, A_block_surface, emis_pipe, emis_block, VF)
            R_tot = R_cond + R_conv + R_rad
            UA = 1 / R_tot
            NTU = UA / C_min
            effectiveness = 1 - np.exp(-NTU)
            Q_rad = effectiveness * C_min * (fluid.temperature - T_block_surface)
            T_pipe_surface_new = T_tin - Q_rad * (R_cond + R_conv)
            if abs(T_pipe_surface_new - T_pipe_surface) < 1e-2:
                break

            T_pipe_surface = T_pipe_surface_new

        return Q_rad

    def radial_conduction(self, T_inner: float, T_outer: float) -> float:
        """
        Calculate radial conduction through the storage block at a given axial location.

        Parameters:
        T_inner (float): Temperature at the inner radius (K).
        T_outer (float): Temperature at the outer radius (K).

        Returns:
        float: Radial conduction heat transfer rate (W).
        """
        r_inner = self.graphite_block.rad_in
        r_outer = self.graphite_block.rad_out
        k = storage_tcx(T_inner)
        L = self.segment_length
        R_cond = np.log(r_outer / r_inner) / (2 * np.pi * k * L)
        Q_cond = (T_inner - T_outer) / R_cond
        return Q_cond

    def update_temps(self, init_temp_dist: np.ndarray, Q_rad: np.ndarray, time_step: float, sub_steps: int, T_sur: float, block_heat_loss: bool) -> np.ndarray:
        """
        Update the temperature distribution in the storage block.

        Parameters:
        init_temp_dist (2D array): Initial temperature distribution in the block (K).
        Q_rad (1D array): Radiative heat flux for each axial segment (W) per block.
        time_step (float): Time step for the update (s).
        sub_steps (int): Number of sub-steps to divide the time step into for stability.

        Returns:
        2D array: Updated temperature distribution (K).
        """
        r_inner = self.block.rad_in
        r_outer = self.block.rad_out
        dr = (r_outer - r_inner) / self.num_rad_segments
        dz = self.segment_length
        T = np.copy(init_temp_dist).astype(float)
        sub_time_step = time_step / sub_steps
        vol_tot = 0
        int_energy_tot = 0
        int_energy_check = 0
        surf_heat_loss = 0
        for k in range(sub_steps):
            net_heat_step = 0
            net_heat = np.zeros_like(T)
            for i in range(self.num_ax_segments):
                for j in range(self.num_rad_segments):
                    r_in = r_inner + j * dr
                    r_c = r_in + 0.5 * dr
                    r_out = r_in + dr
                    r_c_next = r_in + 1.5 * dr
                    r_c_last = r_in - 0.5 * dr
                    if j == 0:
                        heat_in_rad = Q_rad[i]
                        heat_out_rad = storage_tcx(T[i, j]) * (T[i, j] - T[i, j + 1]) * 2 * np.pi * dz / np.log(r_c_next / r_c)
                    elif j == self.num_rad_segments - 1:
                        heat_in_rad = storage_tcx(T[i, j]) * (T[i, j - 1] - T[i, j]) * 2 * np.pi * dz / np.log(r_c / r_c_last)
                        if block_heat_loss:
                            h_out = 100
                            R_ins = RprimeCondCyl(2*r_c, 2*(self.block.rad_out + self.block.th_ins), ins_tcx((T_sur + T[i, j]) / 2)) / dz
                            R_conv_out = 1 / (h_out * np.pi * (2 * self.block.rad_out + 2 * self.block.th_ins) * dz)
                            R_tot_block = R_ins + R_conv_out
                            heat_out_rad = (T[i, j] - T_sur) / R_tot_block
                            surf_heat_loss += heat_out_rad * sub_time_step
                        else:
                            heat_out_rad = 0.0
                    else:
                        heat_in_rad = storage_tcx(T[i, j]) * (T[i, j - 1] - T[i, j]) * 2 * np.pi * dz / np.log(r_c / r_c_last)
                        heat_out_rad = storage_tcx(T[i, j]) * (T[i, j] - T[i, j + 1]) * 2 * np.pi * dz / np.log(r_c_next / r_c)

                    if i == 0:
                        heat_in_ax = 0.0
                        heat_out_ax = storage_tcx(T[i, j]) * (T[i, j] - T[i + 1, j]) / dz * np.pi * (r_out ** 2 - r_in ** 2)
                    elif i == self.num_ax_segments - 1:
                        heat_in_ax = storage_tcx(T[i, j]) * (T[i - 1, j] - T[i, j]) / dz * np.pi * (r_out ** 2 - r_in ** 2)
                        heat_out_ax = 0.0
                    else:
                        heat_in_ax = storage_tcx(T[i, j]) * (T[i - 1, j] - T[i, j]) / dz * np.pi * (r_out ** 2 - r_in ** 2)
                        heat_out_ax = storage_tcx(T[i, j]) * (T[i, j] - T[i + 1, j]) / dz * np.pi * (r_out ** 2 - r_in ** 2)

                    net_heat[i,j] = (heat_in_rad - heat_out_rad) + (heat_in_ax - heat_out_ax)
                    net_heat_step += net_heat[i,j]
                    int_energy_tot += net_heat[i,j] * sub_time_step

            for i in range(self.num_ax_segments):
                for j in range(self.num_rad_segments):
                    r_in = r_inner + j * dr
                    r_out = r_in + dr
                    T[i, j] += net_heat[i, j] * sub_time_step / (storage_cp(T[i, j]) * storage_rho(T[i, j]) * np.pi * (r_out ** 2 - r_in ** 2) * dz)

        for i in range(self.num_ax_segments):
            for j in range(self.num_rad_segments):
                r_in = r_inner + j * dr
                r_out = r_in + dr
                int_energy_check += (T[i, j] - init_temp_dist[i, j]) * np.pi * (r_out ** 2 - r_in ** 2) * dz * storage_cp(init_temp_dist[i, j]) * storage_rho(init_temp_dist[i, j])
                vol_tot += np.pi * (r_out ** 2 - r_in ** 2) * dz

        return T, int_energy_tot, surf_heat_loss

    def update_temps_matrix(self, init_temp_dist: np.ndarray, Q_rad: np.ndarray, time_step: float, sub_steps: int = 10) -> np.ndarray:
        #attempt to rewrite above using linear algebra, in progress
        r_inner = self.block.rad_in
        r_outer = self.block.rad_out
        dr = (r_outer - r_inner) / self.num_rad_segments
        dz = self.segment_length
        sub_time_step = time_step / sub_steps

        T = np.copy(init_temp_dist).astype(float)

        for _ in range(sub_steps):
            num_nodes = self.num_ax_segments * self.num_rad_segments
            A = np.zeros((num_nodes, num_nodes))
            b = np.zeros(num_nodes)

            for i in range(self.num_ax_segments):
                for j in range(self.num_rad_segments):
                    index = i * self.num_rad_segments + j
                    r_in = r_inner + j * dr
                    r_out = r_in + dr

                    k = storage_tcx(T[i, j])
                    cp = storage_cp(T[i, j])
                    rho = storage_rho(T[i, j])
                    alpha = k / (cp * rho)

                    if j == 0:
                        A[index, index] = 1 + sub_time_step * (2 * np.pi * dz / np.log(r_out / r_in)) * alpha / (np.pi * (r_out**2 - r_in**2) * dz)
                        if j + 1 < self.num_rad_segments:
                            A[index, index + 1] = -sub_time_step * (2 * np.pi * dz / np.log(r_out / r_in)) * alpha / (np.pi * (r_out**2 - r_in**2) * dz)
                        b[index] = T[i, j] + sub_time_step * Q_rad[i] / (cp * rho * np.pi * (r_out**2 - r_in**2) * dz)
                    elif j == self.num_rad_segments - 1:
                        A[index, index] = 1 + sub_time_step * (2 * np.pi * dz / np.log(r_out / r_in)) * alpha / (np.pi * (r_out**2 - r_in**2) * dz)
                        if j - 1 >= 0:
                            A[index, index - 1] = -sub_time_step * (2 * np.pi * dz / np.log(r_out / r_in)) * alpha / (np.pi * (r_out**2 - r_in**2) * dz)
                        b[index] = T[i, j]
                    else:
                        A[index, index] = 1 + sub_time_step * (2 * np.pi * dz / np.log(r_out / r_in)) * alpha / (np.pi * (r_out**2 - r_in**2) * dz)
                        if j - 1 >= 0:
                            A[index, index - 1] = -sub_time_step * (2 * np.pi * dz / np.log(r_out / r_in)) * alpha / (np.pi * (r_out**2 - r_in**2) * dz)
                        if j + 1 < self.num_rad_segments:
                            A[index, index + 1] = -sub_time_step * (2 * np.pi * dz / np.log(r_out / r_in)) * alpha / (np.pi * (r_out**2 - r_in**2) * dz)
                        b[index] = T[i, j]

                    if i == 0:
                        if i + 1 < self.num_ax_segments:
                            A[index, index + self.num_rad_segments] = -sub_time_step * (storage_tcx(init_temp_dist[i, j]) / dz) / (storage_cp(init_temp_dist[i, j]) * storage_rho(init_temp_dist[i, j]) * np.pi * (r_out**2 - r_in**2))
                    elif i == self.num_ax_segments - 1:
                        if i - 1 >= 0:
                            A[index, index - self.num_rad_segments] = -sub_time_step * (storage_tcx(init_temp_dist[i, j]) / dz) / (storage_cp(init_temp_dist[i, j]) * storage_rho(init_temp_dist[i, j]) * np.pi * (r_out**2 - r_in**2))
                    else:
                        A[index, index - self.num_rad_segments] = -sub_time_step * (storage_tcx(init_temp_dist[i, j]) / dz) / (storage_cp(init_temp_dist[i, j]) * storage_rho(init_temp_dist[i, j]) * np.pi * (r_out**2 - r_in**2))
                        A[index, index + self.num_rad_segments] = -sub_time_step * (storage_tcx(init_temp_dist[i, j]) / dz) / (storage_cp(init_temp_dist[i, j]) * storage_rho(init_temp_dist[i, j]) * np.pi * (r_out**2 - r_in**2))

            T_new = np.linalg.solve(A, b)
            T = T_new.reshape((self.num_ax_segments, self.num_rad_segments))

        return T

    def bypass_outlet_condition(self, storage_bypass_fraction, inlet_fluid):
        #Calculate resulting temperature from storage block given a guess bypass ratio
        storage_in_main = copy.deepcopy(inlet_fluid)
        storage_in_bypass = copy.deepcopy(inlet_fluid)
        storage_in_main.flow_rate *= (1 - storage_bypass_fraction)
        storage_in_bypass.flow_rate *= storage_bypass_fraction
        storage_out_main, Q_store, Q_rad_dist, _ = self.outlet_conditions(storage_in_main)
        storage_out_bypass = copy.deepcopy(storage_in_bypass)
        storage_out_temp = (storage_out_main.temperature * storage_out_main.flow_rate + storage_out_bypass.temperature
                               * storage_out_bypass.flow_rate) / (inlet_fluid.flow_rate)
        outlet_fluid = Fluid(storage_out_temp, storage_out_main.pressure, inlet_fluid.flow_rate)
        return outlet_fluid, Q_store, Q_rad_dist

    def optim_bypass_ratio(self, init_bypass_fraction, inlet_fluid, target_outlet_temp, abs_tol=1e-2, deriv_tol=1e-8, max_iterations=100):
        # find optimal bypass ratio using Newton-Raphson
        storage_bypass_fraction = init_bypass_fraction
        derivative = 1
        last_derivative = 2

        if inlet_fluid.temperature > target_outlet_temp:
            return self.bypass_range[1]
        else:
            storage_out_min, _, _ = self.bypass_outlet_condition(self.bypass_range[0], inlet_fluid)
            if storage_out_min.temperature < target_outlet_temp:
                return self.bypass_range[0]
            else:
                storage_out_max, _, _ = self.bypass_outlet_condition(self.bypass_range[1], inlet_fluid)
                if storage_out_max.temperature > target_outlet_temp:
                    return self.bypass_range[1]
            for _ in range(max_iterations):
                storage_bypass_fraction = min(self.bypass_range[1], max(self.bypass_range[0], storage_bypass_fraction))
                storage_out, _, _ = self.bypass_outlet_condition(storage_bypass_fraction, inlet_fluid)
                residual = storage_out.temperature - target_outlet_temp
                if abs(residual) < abs_tol or abs(derivative) < deriv_tol:
                    return storage_bypass_fraction
                elif storage_bypass_fraction == self.bypass_range[0] and (residual < 0 or derivative == last_derivative):
                    return self.bypass_range[0]
                elif storage_bypass_fraction == self.bypass_range[1] and (residual > 0 or derivative == last_derivative):
                    return self.bypass_range[1]

                if storage_bypass_fraction > self.bypass_range[1] - 1e-3:
                    delta_frac = -1e-3
                else:
                    delta_frac = 1e-3

                last_derivative = derivative
                res_plus_fluid, _, _ = self.bypass_outlet_condition(storage_bypass_fraction + delta_frac, inlet_fluid)
                residual_plus = res_plus_fluid.temperature - target_outlet_temp
                derivative = (residual_plus - residual) / delta_frac
                if derivative != 0:
                    storage_bypass_fraction -= residual / derivative

    def outlet_conditions(self, fluid: Fluid) -> tuple:
        """
        Calculate outlet temperature and heat transfer rate.

        Parameters:
        fluid (Fluid): Fluid object containing the inlet conditions.

        Returns:
        tuple: Outlet fluid conditions and heat transfer information.
        """
        T_tin_in = fluid.temperature
        P_tin_in = fluid.pressure
        mdot = fluid.flow_rate
        fluid_pipe = Fluid(T_tin_in, P_tin_in, mdot / self.N_pipes)
        fluid_pipe_loc = fluid_pipe
        T_block_surface = self.block.temp_dist[:,0]
        T_tin = np.zeros(self.num_ax_segments + 1)
        T_tin[0] = T_tin_in
        dP_tot = 0
        Q_total = 0
        Q_rad_dist = []
        for i in range(self.num_ax_segments):
            Q_rad = self.heat_transfer_rate(fluid_pipe_loc, T_block_surface[i])
            Q_total += Q_rad
            Q_rad_dist.append(Q_rad)

            T_tin[i + 1] = T_tin[i] - Q_rad / (fluid_pipe_loc.flow_rate * Sn_cp(T_tin[i]))
            fluid_pipe_loc.temperature = T_tin[i+1]

        T_tin_out = T_tin[-1]
        Q_total *= self.N_pipes
        P_tin_out = P_tin_in - self.pipe.pressure_drop(fluid_pipe)
        return Fluid(T_tin_out, P_tin_out, mdot), Q_total, Q_rad_dist, T_tin


class PowerConversionBlock:
    def __init__(self, axial_steps: int, N_cells: int,
                     height_pipe: float, width_emitter: float, length: float, roughness_pipe: float, width_powerstick: float,
                     wall_thickness: float, cooling_temp: float, panel_efficiency=None, T_tin_max_PCB: float = None, f_bare: float = 0.0,
                     f_active: float = 1.0, liner_bool: bool = False, emissivity_table: list = None):
        """
        Initialize the PowerConversionBlock object with its physical properties and components.

        Parameters:
        axial_steps (int): number of axial discretizations
        N_cells (int): number of powerstick/emitter assemblies
        height_pipe (float): Height of the rectangular pipe (m).
        width_emitter (float): Width of the emitter (m).
        length (float): Length of the components (m).
        roughness_pipe (float): Roughness of the pipe (m).
        width_powerstick (float): Width of the power stick (m).
        wall_thickness (float): Thickness of the graphite wall (m).
        cooling_temp (float): Cooling temperature to maintain the power stick (K).
        panel_efficiency: Ignored — efficiency is computed dynamically from emitter temperature and flux.
        T_tin_max_PCB (float): Max permitted Tin inlet temperature (K).
        emissivity_table (list): List of tuples containing wavelength ranges and corresponding emissivities.
        """
        self.pipe = RectangularPipe(H=height_pipe, W=width_emitter, length=length, roughness=roughness_pipe)
        self.emitter = Emitter(width=width_emitter, height=height_pipe, wall_thickness=wall_thickness, pipe=self.pipe, liner_bool=liner_bool)
        self.powerstick = PowerStick(axial_steps=axial_steps, width=width_powerstick,
                                     height=length, cooling_temp=cooling_temp, f_bare=f_bare, f_active=f_active, liner_bool=liner_bool, emissivity_table=emissivity_table)
        self.num_ax_steps = axial_steps
        self.N_cells = N_cells
        self.step_length = self.powerstick.height / self.num_ax_steps
        self.VF = self.powerstick.width / self.emitter.width
        self.T_tin_max_PCB = T_tin_max_PCB

    def calculate_temperatures(self, fluid: Fluid, T_guess: float, tol_TW: float = 1, max_iter: int = 100):
        """
        Iteratively solve for the graphite and tungsten temperatures.

        Parameters:
        fluid: (Fluid): Fluid object containing the flow conditions.
        T_guess (float): Initial guess for the tungsten temperature (K).
        tol_TW (float): Tolerance for temperature convergence (K).
        max_iter (int): Maximum number of iterations.

        Returns:
        tuple: Converged temperatures of graphite wall and tungsten liner (K).
        """
        T_tin = fluid.temperature
        P_tin = fluid.pressure
        mdot = fluid.flow_rate
        T_graphite = T_tin
        T_emit = T_guess
        A_emitter = 4 * self.emitter.width * self.step_length
        A_module = self.powerstick.A_step
        for _ in range(max_iter):
            R_total_emitter, R_conv, R_cond_graphite, R_rad_graphite_tungsten = tuple(x / self.step_length for x in
                                                    self.emitter.R_tin_to_tungsten(fluid, T_graphite, T_emit))

            Q_emitter = (T_tin - T_emit) / R_total_emitter

            electricity_step, waste_heat_step, _ = self.powerstick.generate_electricity_step(T_emit, self.VF)
            Q_emit_to_powerstick = electricity_step + waste_heat_step
            R_emit_to_powerstick = (T_emit - self.powerstick.cooling_temp) / Q_emit_to_powerstick
            T_emit_new = T_emit + (Q_emitter - Q_emit_to_powerstick) * R_total_emitter
            T_graphite_new = T_tin - Q_emitter * (R_cond_graphite + R_conv)
            if abs(T_emit_new - T_emit) < tol_TW:
                return T_emit_new, T_graphite_new

            T_emit = T_emit_new
            T_graphite = T_graphite_new

        raise RuntimeError("Failed to converge to a solution for temperatures.")

    def power_conversion(self, fluid: Fluid):
        """
        Perform the power conversion calculations by discretizing the block axially.

        Parameters:
        fluid (Fluid): Fluid object containing the flow conditions.

        Returns:
        tuple: Arrays containing the temperatures, electricity generated, and waste heat for each step.
        """
        T_tin_in = fluid.temperature
        mdot = fluid.flow_rate
        temperatures = np.zeros(self.num_ax_steps + 1)
        electricity_generated = 0
        waste_heat = 0
        Q_cell = 0
        Q_rad = np.zeros(self.num_ax_steps)

        temperatures[0] = T_tin_in

        for i in range(self.num_ax_steps):
            T_tin = temperatures[i]
            T_guess = T_tin - 100
            fluid_cell_loc = Fluid(T_tin, fluid.pressure, mdot / self.N_cells)
            T_emit, _ = self.calculate_temperatures(fluid_cell_loc, T_guess)
            electricity_step, waste_heat_step, cell_flux_step = self.powerstick.generate_electricity_step(T_emit, self.VF)
            electricity_generated += electricity_step
            waste_heat += waste_heat_step
            Q_cell += cell_flux_step
            Q_rad[i] = electricity_step + waste_heat_step

            delta_T_tin = -Q_rad[i] / (mdot / self.N_cells * Sn_cp(temperatures[i]))
            temperatures[i + 1] = T_tin + delta_T_tin

        electricity_generated *= self.N_cells
        waste_heat *= self.N_cells
        Q_cell *= self.N_cells
        return temperatures, electricity_generated, waste_heat, Q_cell

    def determine_outlet_conditions(self, fluid: Fluid) -> Fluid:
        """
        Determine the outlet conditions of the power conversion block.

        Parameters:
        fluid (Fluid): Fluid object containing the inlet conditions.

        Returns:
        Fluid: Fluid object containing the outlet conditions.
        """
        temperatures, _, _, _ = self.power_conversion(fluid)
        T_out = temperatures[-1]
        fluid_pipe = Fluid(fluid.temperature, fluid.pressure, fluid.flow_rate / self.N_cells / 4)
        dp = self.pipe.pressure_drop(fluid_pipe)
        P_out = fluid.pressure - dp
        return Fluid(T_out, P_out, fluid.flow_rate)


class Pump:
    def __init__(self, pump_efficiency: float):
        """
        Initialize the Pump object with its physical properties and components.

        Parameters:
        pump_efficiency (float): Efficiency of the pump (decimal).
        """
        self.pump_efficiency = pump_efficiency

    def outlet_conditions(self, pump_power: float, fluid: Fluid) -> Fluid:
        """
        Calculate the outlet conditions of the pump.

        Parameters:
        pump_power (float): Power of the pump (W).
        fluid (Fluid): Fluid object containing the inlet conditions.

        Returns:
        Fluid: Fluid object containing the outlet conditions.
        """
        T_in = fluid.temperature
        P_in = fluid.pressure
        mdot = fluid.flow_rate
        fluid_density = Sn_rho(T_in)
        fluid_cp = Sn_cp(T_in)
        power_added = pump_power * self.pump_efficiency
        delta_T = pump_power / (mdot * fluid_cp)
        T_out = T_in + delta_T
        delta_P = power_added / (mdot / fluid_density)
        P_out = P_in + delta_P
        return Fluid(T_out, P_out, mdot)

    def required_power(self, fluid_in: Fluid, P_out: float) -> tuple:
        """
        Calculate the required power and outlet temperature for the pump.

        Parameters:
        fluid_in (Fluid): Fluid object containing the inlet conditions.
        P_out (float): Outlet pressure (Pa).

        Returns:
        tuple: Required power (W) and Fluid object containing the outlet conditions.
        """
        T_in = fluid_in.temperature
        P_in = fluid_in.pressure
        mdot = fluid_in.flow_rate
        fluid_density = Sn_rho(T_in)
        fluid_cp = Sn_cp(T_in)
        delta_P = P_out - P_in
        power_required = (mdot / fluid_density) * delta_P / self.pump_efficiency
        delta_T = power_required / (mdot * fluid_cp)
        T_out = T_in + delta_T
        return power_required, Fluid(T_out, P_out, mdot)
