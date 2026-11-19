import numpy as np
from scipy.optimize import brentq
import warnings
from .properties import (gr_emis, gr_tcx, ins_tcx, panel_efficiency,
                          w_emis_total, w_emis_spectral_vec,
                          zrc_emis_wavelength_temp)


class HeatingElement:
    """
    Cylindrical resistive heating element at the centre of the graphite annulus.
    """
    def __init__(self, diameter, max_temperature, max_power, emissivity):
        self.diameter        = float(diameter)
        self.max_temperature = float(max_temperature)
        self.max_power       = float(max_power)
        self.emissivity      = float(emissivity)


class TinlessSimulation:
    """
    Transient model: graphite annulus radiating to a central cylindrical powerstick.
    Each timestep reads inner wall temps, computes radiation + electricity, and updates
    the block temperature distribution via StorageModule.update_temps.
    """

    def __init__(self, storage, powerstick, T_sur=300, block_heat_loss=False,
                 shade_bool=False, shade_side_len=None,
                 shade_thickness=0.05,
                 shade_surface_emissivity=0.8,
                 shade_num_conduction_layers=5,
                 f_shade_min=0.0, f_shade_max=0.9, nominal_cell_flux=None,
                 liner_material='tac', shade_outer_foil=False,
                 liner_thickness=0.0, liner_tungsten_foil=False):
        self.storage         = storage
        self.powerstick      = powerstick
        self.T_sur           = T_sur
        self.block_heat_loss = block_heat_loss
        self.liner_material  = liner_material.lower()

        r_ps_eff      = 2 * powerstick.width / np.pi
        r_in          = storage.block.rad_in
        self.VF       = r_ps_eff / r_in
        self.A_ps_seg = powerstick.A_step

        if powerstick.liner_bool:
            if powerstick.liner_width is None:
                raise ValueError("PowerStick.liner_width must be set when liner_bool=True")
            r_liner          = 2 * powerstick.liner_width / np.pi
            self.VF_gr_liner = r_liner / r_in
            self.VF_liner_ps = r_ps_eff / r_liner
            self.A_liner_seg = 2 * np.pi * r_liner * powerstick.height_step
        else:
            self.VF_gr_liner = None
            self.VF_liner_ps = None
            self.A_liner_seg = None

        self.shade_bool                  = shade_bool
        self.shade_thickness             = shade_thickness
        self.shade_surface_emissivity    = shade_surface_emissivity
        self.shade_num_conduction_layers = shade_num_conduction_layers
        self.f_shade_min                 = f_shade_min
        self.f_shade_max                 = f_shade_max
        self.nominal_cell_flux            = nominal_cell_flux
        self.shade_outer_foil            = shade_outer_foil
        self.liner_thickness              = liner_thickness
        self.liner_tungsten_foil          = liner_tungsten_foil
        self.A_gr_seg = 2 * np.pi * r_in * powerstick.height_step
        if shade_bool:
            if not powerstick.liner_bool:
                raise ValueError("shade_bool=True requires liner_bool=True")
            if shade_side_len is None:
                raise ValueError("shade_side_len must be set when shade_bool=True")
            r_shade = 2 * shade_side_len / np.pi
            gap_outer = r_in    - r_shade   # clearance between shade and cavity wall
            gap_inner = r_shade - r_liner   # clearance between shade and liner
            if r_shade >= r_in:
                raise ValueError(
                    f"shade geometry infeasible: r_shade={r_shade*1e3:.2f} mm >= "
                    f"r_in={r_in*1e3:.2f} mm — shade would touch the cavity wall. "
                    f"Reduce shade_side_len (currently {shade_side_len*1e3:.2f} mm)."
                )
            if r_shade <= r_liner:
                raise ValueError(
                    f"shade geometry infeasible: r_shade={r_shade*1e3:.2f} mm <= "
                    f"r_liner={r_liner*1e3:.2f} mm — shade must sit outside the liner. "
                    f"Increase shade_side_len or reduce liner_side_len."
                )
            # When the shade rotates to the fully-open (perpendicular) position its
            # panel swings through the outer gap. The panel width equals gap_inner
            # (liner-to-shade distance), so it must fit in gap_outer without the
            # outer edge hitting the cavity wall and the inner edge hitting the liner.
            if gap_outer < gap_inner:
                raise ValueError(
                    f"shade geometry infeasible: outer gap ({gap_outer*1e3:.2f} mm) < "
                    f"inner gap ({gap_inner*1e3:.2f} mm) — when fully open the shade "
                    f"panel would hit the cavity wall. "
                    f"Move shade_side_len closer to the liner "
                    f"(max shade_side_len ≈ {(shade_side_len - (gap_inner - gap_outer)/2 * np.pi/2)*1e3:.2f} mm)."
                )
            # Minimum shade size: shade circumference >= (1 + sqrt(2)/2) x liner circumference.
            # When the shade panels open, they must clear the liner corners (which project
            # to r_liner * sqrt(2) diagonally). This sets a hard lower bound on r_shade.
            r_shade_min = (1.0 + np.sqrt(2) / 2.0) * r_liner
            if r_shade < r_shade_min:
                raise ValueError(
                    f"shade geometry infeasible: r_shade={r_shade*1e3:.2f} mm < "
                    f"minimum {r_shade_min*1e3:.2f} mm (= (1+√2/2) × r_liner). "
                    f"Open shade panels would clip the liner corners. "
                    f"Set shade_side_len ≥ {r_shade_min * np.pi / 2.0 * 1e3:.2f} mm."
                )
            # Minimum cavity size: cavity circumference >= (1 + sqrt(2)) x liner circumference.
            # The shade+liner system needs enough radial clearance in the cavity to operate.
            r_in_min = (1.0 + np.sqrt(2)) * r_liner
            if r_in < r_in_min:
                raise ValueError(
                    f"shade geometry infeasible: cavity r_in={r_in*1e3:.2f} mm < "
                    f"minimum {r_in_min*1e3:.2f} mm (= (1+√2) × r_liner). "
                    f"Cavity is too small for this liner+shade configuration."
                )
            self.VF_gr_shade  = r_shade / r_in
            self.VF_shade_lin = r_liner / r_shade
            self.A_shade_seg  = 2 * np.pi * r_shade * powerstick.height_step
        else:
            self.VF_gr_shade  = None
            self.VF_shade_lin = None
            self.A_shade_seg  = None

    def _spectral_grid(self, emissivity_table):
        """Gauss-Legendre nodes/weights in log-wavelength, per emissivity-table band.

        Band edges (incl. the bandgap) are never straddled, so the piecewise-constant
        panel emissivities are integrated exactly and the smooth Planck / TaC
        spectra converge spectrally fast. Default 10 nodes per band (2 bands -> 20
        nodes) is exact to ~1e-6 relative vs. a 20,000-bin reference.

        Returns (wavelengths [µm], weights [m]) so that  ∫ f(λ) dλ ≈ Σ f(λ_i) w_i.
        Cached per (table, n_per_band).
        """
        m = int(getattr(self, 'n_gauss_per_band', 10))
        key = (tuple(tuple(r[0]) for r in emissivity_table), m)
        cache = getattr(self, '_grid_cache', None)
        if cache is not None and cache[0] == key:
            return cache[1], cache[2]
        x, wt = np.polynomial.legendre.leggauss(m)
        wl, w = [], []
        for (lo, hi), *_ in emissivity_table:
            la, lb = np.log(lo), np.log(hi)
            lam = np.exp(0.5 * (lb - la) * x + 0.5 * (lb + la))      # µm
            wl.append(lam)
            w.append(wt * 0.5 * (lb - la) * lam * 1e-6)              # dλ = λ dlnλ  [m]
        wl, w = np.concatenate(wl), np.concatenate(w)
        self._grid_cache = (key, wl, w)
        return wl, w

    def spectral_net_radiation(self, T_hot, f_shade=0.0):
        h   = 6.62607015e-34
        c   = 2.99792458e8
        k_B = 1.380649e-23

        emissivity_table = self.powerstick.emissivity_table
        liner_bool       = self.powerstick.liner_bool
        f_bare           = self.powerstick.f_bare
        f_active         = self.powerstick.f_active
        VF               = self.VF
        A_ps             = self.A_ps_seg
        shade_bool      = self.shade_bool
        VF_gr_liner      = self.VF_gr_liner
        VF_liner_ps      = self.VF_liner_ps
        A_liner_seg      = self.A_liner_seg
        VF_gr_shade     = self.VF_gr_shade
        VF_shade_lin    = self.VF_shade_lin
        A_shade_seg     = self.A_shade_seg

        wavelength_min = emissivity_table[0][0][0]
        wavelength_max = emissivity_table[-1][0][1]
        band_gap       = emissivity_table[0][0][1]

        if getattr(self, 'spectral_quadrature', 'gauss') == 'gauss':
            wavelengths, bw = self._spectral_grid(emissivity_table)
        else:
            # Legacy uniform-log grid (kept for comparison only): bin edges are not
            # aligned to the emissivity-band edges, giving ~2% error in Q_cell.
            n_bins = getattr(self, 'n_bins', 300)
            wavelengths = np.logspace(np.log10(wavelength_min), np.log10(wavelength_max), n_bins)
            _wm = wavelengths * 1e-6
            bw_fwd = np.concatenate([_wm[1:] - _wm[:-1], [_wm[-1] - _wm[-2]]])
            bw_bwd = np.concatenate([[_wm[1] - _wm[0]], _wm[1:] - _wm[:-1]])
            bw = np.where(wavelengths < band_gap, bw_fwd, bw_bwd)
        n_bins        = wavelengths.size
        wavelengths_m = wavelengths * 1e-6

        def _B(T):
            return np.pi * (2*h*c**2) / wavelengths_m**5 / np.expm1(h*c / (wavelengths_m * k_B * T))

        avg_ps_emis      = np.zeros(n_bins)
        active_frac_per_wl = np.zeros(n_bins)
        for (wl_range, emis_active, emis_bare, emis_inactive) in emissivity_table:
            mask = (wavelengths >= wl_range[0]) & (wavelengths < wl_range[1])
            ep = (f_bare * emis_bare
                  + (1 - f_bare) * f_active       * emis_active
                  + (1 - f_bare) * (1 - f_active) * emis_inactive)
            avg_ps_emis[mask]       = ep
            active_frac_per_wl[mask] = (1 - f_bare) * f_active * emis_active / ep

        if not liner_bool:
            eps_gr = gr_emis(T_hot)
            B_hot  = _B(T_hot)
            denom  = VF * (1 - eps_gr) / eps_gr + 1 + (1 - avg_ps_emis) / avg_ps_emis
            Q_ps   = float(np.sum(B_hot * bw / denom)) * A_ps
            Q_cell = float(np.sum(B_hot * bw * active_frac_per_wl / denom)) * A_ps
            return Q_ps, None, Q_cell

        sigma         = 5.67036713e-8
        eps_gr        = gr_emis(T_hot)
        B_hot         = _B(T_hot)
        liner_material = self.liner_material

        if liner_material == 'tac' and abs(band_gap - 1.24) > 1e-6:
            warnings.warn(
                f"tac_emis_binned_1eV is calibrated for a 1.24 µm bandgap but "
                f"the emissivity table has band_gap={band_gap} µm — "
                "liner emissivities may be mismatched.",
                UserWarning, stacklevel=2,
            )

        def _eps_liner_spectral(T_lin):
            """Return (eps_in_arr [n_bins], eps_out_scalar) for the liner at T_lin.

            eps_in  : emissivity of the liner surface facing the powerstick (inner face)
            eps_out : emissivity of the liner surface facing the graphite wall (outer face)

            TaC liner : inner face = TaC (binned spectral), outer face = graphite
            Graphite  : both faces = graphite
            """
            eps_out = gr_emis(T_lin)
            if liner_material == 'tac':
                eps_in = np.clip(-0.0351 + 8.15e-5 * T_lin + (0.3008 - 4.46e-5 * T_lin) / wavelengths, 0.0, 1.0)
            elif liner_material == 'zrc':
                eps_in = np.clip(0.70 - 1e-4 * T_lin + 0.02 * (wavelengths - 0.665), 0.0, 1.0)
            else:
                eps_in = np.full(n_bins, eps_out)
            return eps_in, eps_out

        _shade_active = shade_bool and f_shade > 0 and A_shade_seg is not None
        _f_direct      = 1.0 - f_shade if _shade_active else 1.0
        if _shade_active:
            eps_sh           = self.shade_surface_emissivity
            w_foil_bool      = self.shade_outer_foil
            A_shade_sh      = f_shade * A_shade_seg
            A_liner_sh       = f_shade * A_liner_seg
            eps_gr_liner_out = gr_emis(T_hot)  # liner outer face is always graphite
            # R_rad_in: graphite wall → shade outer face (no-foil path; also used as _Q_max lower-bound)
            R_rad_in  = (VF_gr_shade * (1 - eps_gr) / eps_gr + 1 + (1 - eps_sh) / eps_sh) / A_shade_sh
            R_rad_out = (VF_shade_lin * (1 - eps_sh) / eps_sh
                         + 1 + (1 - eps_gr_liner_out) / eps_gr_liner_out) / A_liner_sh

        liner_thickness    = self.liner_thickness
        liner_tungsten_foil = self.liner_tungsten_foil
        _T_liner_in  = [None]        # written by _balance; read after brentq converges
        _T_foil_inner = [None]       # W foil inner face temp (when liner_tungsten_foil=True)
        _T_sh        = [np.nan, np.nan]  # [T_sh_in, T_sh_out] shade outer/inner face

        def _balance(T_liner_out):
            _, eps_L_out = _eps_liner_spectral(T_liner_out)
            B_lin_out    = _B(T_liner_out)
            denom_in     = VF_gr_liner * (1 - eps_gr) / eps_gr + 1 + (1 - eps_L_out) / eps_L_out
            Q_in         = _f_direct * float(np.sum((B_hot - B_lin_out) * bw / denom_in)) * A_liner_seg
            if _shade_active:
                if w_foil_bool:
                    # Brentq on T_foil directly — eliminates nested solve.
                    # Given T_foil, Q is explicit from the graphite→foil step,
                    # then the rest of the chain (foil→RGF, conduction, RGF→liner)
                    # is propagated forward. Residual = conduction balance.
                    eps_RGF_outer = eps_sh
                    def _foil_residual(T_foil):
                        eps_W    = w_emis_total(T_foil)
                        R_gf     = (VF_gr_shade * (1 - eps_gr) / eps_gr + 1 + (1 - eps_W) / eps_W) / A_shade_sh
                        Q        = sigma * (T_hot**4 - T_foil**4) / R_gf
                        R_fs     = ((1 - eps_W) / eps_W + 1 + (1 - eps_RGF_outer) / eps_RGF_outer) / A_shade_sh
                        T_sh_in  = max(T_foil**4 - Q * R_fs / sigma, 0.0)**0.25
                        T_sh_out = (T_liner_out**4 + Q * R_rad_out / sigma)**0.25
                        k_sh     = ins_tcx((T_sh_in + T_sh_out) / 2)
                        R_cond   = self.shade_thickness / (k_sh * A_shade_sh)
                        return T_sh_in - T_sh_out - Q * R_cond
                    try:
                        _T_foil = brentq(_foil_residual, T_liner_out + 1.0, T_hot - 0.5, xtol=0.1, maxiter=50)
                        eps_W   = w_emis_total(_T_foil)
                        R_gf    = (VF_gr_shade * (1 - eps_gr) / eps_gr + 1 + (1 - eps_W) / eps_W) / A_shade_sh
                        _Q_sh   = sigma * (T_hot**4 - _T_foil**4) / R_gf
                    except ValueError:
                        _Q_sh = 0.0
                else:
                    def _shade_residual(Q):
                        T_sh_in  = max(T_hot**4 - Q * R_rad_in / sigma, 0.0)**0.25
                        T_sh_out = (T_liner_out**4 + Q * R_rad_out / sigma)**0.25
                        k_sh     = ins_tcx((T_sh_in + T_sh_out) / 2)
                        R_cond   = self.shade_thickness / (k_sh * A_shade_sh)
                        return T_sh_in - T_sh_out - Q * R_cond
                    _Q_max = sigma * (T_hot**4 - T_liner_out**4) / R_rad_in * 0.9999
                    try:
                        _Q_sh    = brentq(_shade_residual, 0.0, _Q_max, xtol=0.1, maxiter=50)
                        _T_sh[0] = max(T_hot**4 - _Q_sh * R_rad_in / sigma, 0.0)**0.25
                        _T_sh[1] = (T_liner_out**4 + _Q_sh * R_rad_out / sigma)**0.25
                    except ValueError:
                        _Q_sh    = 0.0
                        _T_sh[:] = [T_liner_out, T_liner_out]
                Q_in += _Q_sh
            # Conduction through liner wall: graphite outer face → TaC inner face
            if liner_thickness > 0:
                R_cond_liner = liner_thickness / (gr_tcx(T_liner_out) * A_liner_seg)
                T_liner_in   = max(T_liner_out - Q_in * R_cond_liner, 400.0)
            else:
                T_liner_in = T_liner_out
            _T_liner_in[0] = T_liner_in
            # Spectral TaC inner-face emissivity — used by both branches
            eps_L_in, _ = _eps_liner_spectral(T_liner_in)
            B_tac_in     = _B(T_liner_in)
            if liner_material == 'tac' and liner_tungsten_foil:
                # Both legs spectral: TaC → W foil, then W foil → PS
                def _foil_inner_res(T_foil):
                    eps_W    = w_emis_spectral_vec(T_foil, wavelengths)
                    B_foil   = _B(T_foil)
                    # TaC → foil: net spectral radiation, bin by bin
                    denom_tf = VF_liner_ps * (1 - eps_L_in) / eps_L_in + 1 + (1 - eps_W) / eps_W
                    Q_tf     = float(np.sum((B_tac_in - B_foil) * bw / denom_tf)) * A_ps
                    # foil → PS: spectral
                    denom_fps = VF_liner_ps * (1 - eps_W) / eps_W + 1 + (1 - avg_ps_emis) / avg_ps_emis
                    Q_fps     = float(np.sum(B_foil * bw / denom_fps)) * A_ps
                    return Q_tf - Q_fps
                try:
                    _T_foil = brentq(_foil_inner_res, 400.0, T_liner_in - 0.5, xtol=0.5, maxiter=50)
                except ValueError:
                    _T_foil = T_liner_in
                _T_foil_inner[0] = _T_foil
                eps_W     = w_emis_spectral_vec(_T_foil, wavelengths)
                B_foil    = _B(_T_foil)
                denom_fps = VF_liner_ps * (1 - eps_W) / eps_W + 1 + (1 - avg_ps_emis) / avg_ps_emis
                Q_out     = float(np.sum(B_foil * bw / denom_fps)) * A_ps
            else:
                denom_out = VF_liner_ps * (1 - eps_L_in) / eps_L_in + 1 + (1 - avg_ps_emis) / avg_ps_emis
                Q_out     = float(np.sum(B_tac_in * bw / denom_out)) * A_ps
            return Q_in - Q_out

        T_liner_out   = brentq(_balance, 400.0, T_hot - 1.0, xtol=1.0, maxiter=50)
        T_liner_in    = _T_liner_in[0]
        T_foil_inner  = _T_foil_inner[0]

        if liner_material == 'tac' and liner_tungsten_foil and T_foil_inner is not None:
            eps_W_ps   = w_emis_spectral_vec(T_foil_inner, wavelengths)
            B_foil     = _B(T_foil_inner)
            denom_foil = VF_liner_ps * (1 - eps_W_ps) / eps_W_ps + 1 + (1 - avg_ps_emis) / avg_ps_emis
            Q_ps   = float(np.sum(B_foil * bw / denom_foil)) * A_ps
            Q_cell = float(np.sum(B_foil * bw * active_frac_per_wl / denom_foil)) * A_ps
        else:
            eps_L_in, _ = _eps_liner_spectral(T_liner_in)
            B_lin_in    = _B(T_liner_in)
            denom_out   = VF_liner_ps * (1 - eps_L_in) / eps_L_in + 1 + (1 - avg_ps_emis) / avg_ps_emis
            Q_ps   = float(np.sum(B_lin_in * bw / denom_out)) * A_ps
            Q_cell = float(np.sum(B_lin_in * bw * active_frac_per_wl / denom_out)) * A_ps
        self._last_temps = {
            'T_liner_out':  T_liner_out,
            'T_liner_in':   T_liner_in,
            'T_foil_inner': T_foil_inner,
            'T_sh_in':      _T_sh[0],
            'T_sh_out':     _T_sh[1],
        }
        return Q_ps, T_liner_in, Q_cell

    def _step_calc(self, T_dist, f_shade=0.0):
        ps         = self.powerstick
        n_ax       = self.storage.num_ax_segments
        if n_ax != ps.num_ax_steps:
            raise NotImplementedError(
                "local per-segment radiation needs one block ring per powerstick step "
                "(block_height == L); taller blocks need the axial enclosure model")
        band_gap   = ps.emissivity_table[0][0][1]
        A_cell_cm2 = ps.A_step * (1 - ps.f_bare) * ps.f_active * 1e4
        Q_rad       = np.zeros(n_ax)
        Q_elec      = np.zeros(n_ax)
        Q_cell      = np.zeros(n_ax)
        Q_waste     = np.zeros(n_ax)
        T_liner     = np.zeros(n_ax)
        T_liner_out   = np.zeros(n_ax)
        T_foil_inner  = np.full(n_ax, np.nan)
        T_sh_in       = np.full(n_ax, np.nan)
        T_sh_out      = np.full(n_ax, np.nan)
        for i in range(n_ax):
            T_wall_i = T_dist[i, 0]
            Q_rad[i], T_liner_i, Q_cell[i] = self.spectral_net_radiation(T_wall_i, f_shade=f_shade)
            T_liner[i] = T_liner_i if ps.liner_bool else 0.0
            T_emit_i   = T_liner_i if ps.liner_bool else T_wall_i
            flux_wcm2  = Q_cell[i] / A_cell_cm2
            eff_i      = panel_efficiency(band_gap, T_emit_i, flux=flux_wcm2)
            eff_i     *= getattr(self, 'cell_eff_factor', 1.0)   # cell degradation (deposition)
            Q_elec[i]  = Q_cell[i] * eff_i
            Q_waste[i] = Q_rad[i] - Q_elec[i]
            lt = getattr(self, '_last_temps', {})
            T_liner_out[i]  = lt.get('T_liner_out') or T_liner_i
            T_foil_inner[i] = lt.get('T_foil_inner', np.nan) or np.nan
            T_sh_in[i]      = lt.get('T_sh_in', np.nan)
            T_sh_out[i]     = lt.get('T_sh_out', np.nan)
        self._step_temps = {
            'T_liner_out':  T_liner_out,
            'T_foil_inner': T_foil_inner,
            'T_sh_in':      T_sh_in,
            'T_sh_out':     T_sh_out,
        }
        return Q_rad, Q_elec, Q_cell, Q_waste, T_liner

    def _compute_f_shade(self, T_dist):
        ps = self.powerstick
        A_cell_cm2 = 4 * ps.width * ps.height * (1 - ps.f_bare) * ps.f_active * 1e4

        def _flux(fs):
            _, _, Q_cell, _, _ = self._step_calc(T_dist, f_shade=fs)
            return Q_cell.sum() / A_cell_cm2

        f_lo, f_hi = self.f_shade_min, self.f_shade_max
        flux_lo = _flux(f_lo)
        flux_hi = _flux(f_hi)

        if flux_lo <= self.nominal_cell_flux:
            return f_lo
        if flux_hi >= self.nominal_cell_flux:
            return f_hi
        return brentq(lambda fs: _flux(fs) - self.nominal_cell_flux,
                      f_lo, f_hi, xtol=1e-4, maxiter=30)

    def _radiation_step(self, T_dist, use_shade):
        """f_shade search + radiation for one step. Local per-segment model, or the
        full-height enclosure model when build_tinless attached sim.axial_model."""
        ax = getattr(self, 'axial_model', None)
        if ax is not None:
            return ax.step(T_dist, control=use_shade)
        f_shade = self._compute_f_shade(T_dist) if use_shade else 0.0
        return f_shade, self._step_calc(T_dist, f_shade=f_shade)

    def _axial_diag(self, T, Q_rad, active):
        """Per-step diagnostics for the decoupled-height model (brief §5):
        Q_rad_low   heat leaving the lower-zone block rings [W]
        Q_vert_rad  net upward radiative transfer across the liner-bottom plane [W]
                    (dict split: outer gap / inner space / through the shade, porous model)
        Q_vert_cond net upward block conduction across that plane [W]
        T_wall_spread  block inner-wall max - min [K]"""
        from .properties import storage_tcx
        st = self.storage
        out = dict(T_wall_spread=float(T[:, 0].max() - T[:, 0].min()), Q_rad_low=np.nan,
                   Q_vert_rad=np.nan, Q_vert_cond=np.nan, Q_vert_rad_outer=np.nan,
                   Q_vert_rad_inner=np.nan, Q_vert_rad_through=np.nan)
        ax = getattr(self, 'axial_model', None)
        n_low = getattr(ax, 'n_low', 0) if ax is not None else 0
        if ax is None or not n_low:
            return out
        out['Q_rad_low'] = float(np.sum(Q_rad[:n_low]))
        dz = st.axial_dz()
        dz_c = 0.5 * (dz[n_low - 1] + dz[n_low])
        nR = st.num_rad_segments
        r_lo = st.block.rad_in + np.arange(nR) * (st.block.rad_out - st.block.rad_in) / nR
        r_hi = r_lo + (st.block.rad_out - st.block.rad_in) / nR
        Tl, Tu = T[n_low - 1], T[n_low]
        k = np.asarray(storage_tcx(0.5 * (Tl + Tu)), float) * np.ones(nR)
        out['Q_vert_cond'] = float(np.sum(k * np.pi * (r_hi ** 2 - r_lo ** 2) * (Tl - Tu) / dz_c))
        if active and getattr(ax, 'last', None) is not None and hasattr(ax, 'vertical_transport'):
            vt = ax.vertical_transport()
            out['Q_vert_rad'] = float(sum(vt.values()))
            for kk in ('outer', 'inner', 'through'):
                if kk in vt:
                    out[f'Q_vert_rad_{kk}'] = float(vt[kk])
        return out

    def _dep_T_emitter(self, T_inner_mean, T_liner_mean, T_foil_inner_mean):
        """Pick the correct emitter temperature for deposition based on configuration.

        The innermost surface facing the PS is always T_liner when any liner is
        present (VF_gr_liner is not None), regardless of liner material — because
        the liner inner face is the actual gas-phase emitter even for a graphite liner.
        """
        if self.liner_tungsten_foil:
            return T_foil_inner_mean
        elif self.VF_gr_liner is not None:
            return T_liner_mean
        else:
            return T_inner_mean

    def run_simulation(self, end_time, dt_short, dt_medium, dt_long, sub_steps,
                       dt_short_duration=3600, snap_every=None, print_interval=0,
                       deposition_fn=None, deposition_kwargs=None, T_ps_cutoff=None,
                       stop_at_cutoff=False, dt_at_shade_limit=None, Q_sink_fn=None):
        """dt_at_shade_limit: time step while the shade sits at its most-closed limit
        (f_shade == f_shade_max); default None -> dt_short (original rule).
        Q_sink_fn: optional f(T_dist) -> heat [W] removed per ring (nA,) at the outer
        node, or per node (nA, nR); e.g. thermal.tank.CellLossSink (roof / floor
        insulation losses on the top / bottom rings). Default None = adiabatic.
        The applied total is recorded as 'Q_sink' [W]."""
        storage  = self.storage
        n_ax     = storage.num_ax_segments
        _use_shade = self.shade_bool and self.nominal_cell_flux is not None
        _dep_kw = deposition_kwargs or {}
        # volume weights (radial rings x ring heights): T_avg and the T_ps_cutoff test use
        # the volume-weighted block average; T_inner / T_outer are ring-height weighted
        _nR = storage.num_rad_segments
        _r = storage.block.rad_in + np.arange(_nR + 1) * (storage.block.rad_out - storage.block.rad_in) / _nR
        _sl = storage.axial_dz()
        _W = np.diff(_r ** 2)[None, :] * _sl[:, None]
        _W = _W / _W.sum()
        _wz = _sl / _sl.sum()
        _Tavg = lambda T_d: float((T_d * _W).sum())

        T = storage.block.temp_dist
        f_shade, (Q0, Q_elec0, Q_cell0, _, T_liner0) = self._radiation_step(T, _use_shade)

        _st0 = self._step_temps
        time_arr        = [0.0]
        Q_arr           = [Q0.sum()]
        Q_cell_arr      = [Q_cell0.sum()]
        Q_elec_arr      = [Q_elec0.sum()]
        T_avg_arr       = [_Tavg(T)]
        T_inner_arr     = [float(T[:, 0] @ _wz)]
        T_outer_arr     = [float(T[:, -1] @ _wz)]
        T_liner_arr     = [T_liner0.mean()]
        T_liner_out_arr   = [np.nanmean(_st0['T_liner_out'])]
        T_foil_inner_arr  = [np.nanmean(_st0['T_foil_inner'])]
        T_sh_in_arr       = [np.nanmean(_st0['T_sh_in'])]
        T_sh_out_arr      = [np.nanmean(_st0['T_sh_out'])]
        f_shade_arr    = [f_shade]
        _lim = lambda k: float((getattr(getattr(self, 'axial_model', None), 'last', None) or {}).get(k, np.nan))
        # axial-model extras recorded per step (NaN when not provided / stick removed):
        # vane limits [W]; liner bottom (Phase 4): heat through pad [W], T_R / T_LB [K],
        # bottom-cap absorbed power [W] and flux [W/cm2], liner bottom -> stick sides [W]
        _AX_KEYS = ('Q_cell_min', 'Q_cell_max', 'Q_pad', 'T_R', 'T_LB', 'Q_cap', 'q_cap', 'Q_LB_cells',
                    'vane_angle')      # approximate vane angle [deg] (vane model)
        ax_rec = {k: [_lim(k)] for k in _AX_KEYS}
        _dg0 = self._axial_diag(T, Q0, True)
        diag_rec = {k: [v] for k, v in _dg0.items()}
        _lo_key = 'T_liner_out'
        liner_out_snaps = [np.array(_st0[_lo_key], float).copy()]
        _n_lo = liner_out_snaps[0].size

        def _lo_snap(st):
            a = np.array(st[_lo_key], float)
            return a.copy() if a.size == _n_lo else np.full(_n_lo, np.nan)
        ps_active_arr  = [True]
        Q_sink_arr     = [0.0 if Q_sink_fn is None else float(np.sum(Q_sink_fn(T)))]
        snap_times   = [0.0]
        temp_snaps   = [T.copy()]

        if deposition_fn is not None:
            _T_e0 = self._dep_T_emitter(T[:, 0].mean(), T_liner0.mean(),
                                         np.nanmean(_st0['T_foil_inner']))
            _d0 = deposition_fn(_T_e0, **_dep_kw)
            dep_rate_arr  = [_d0['total']]
            dep_C_arr     = [_d0['C']]
            dep_metal_arr = [_d0['metal']]
        else:
            dep_rate_arr = dep_C_arr = dep_metal_arr = []

        t            = 0.0
        next_snap    = snap_every if snap_every else float('inf')
        last_print_t = -float('inf')
        _ps_active   = True
        t_ps_removed = None

        while t < end_time:
            T = storage.block.temp_dist

            if _ps_active:
                f_shade, (Q_rad, Q_elec, Q_cell, Q_waste, T_liner) = self._radiation_step(T, _use_shade)
                if t < dt_short_duration:
                    dt = dt_short
                elif self.shade_bool and f_shade == self.f_shade_max:
                    dt = dt_short if dt_at_shade_limit is None else dt_at_shade_limit
                elif self.shade_bool and f_shade > self.f_shade_min:
                    dt = dt_medium
                else:
                    dt = dt_long
                dt = min(dt, end_time - t)
                _st = self._step_temps
            else:
                # Conduction-only phase: powerstick removed, no radiation exchange.
                # Liner stays in thermal contact with the block inner wall.
                f_shade = 0.0
                dt      = min(dt_long, end_time - t)
                Q_rad   = np.zeros(n_ax)
                Q_elec  = np.zeros(n_ax)
                Q_cell  = np.zeros(n_ax)
                T_liner = T[:, 0].copy()   # equilibrated with block inner wall
                _st     = {k: np.full(n_ax, np.nan)
                           for k in ('T_liner_out', 'T_foil_inner', 'T_sh_in', 'T_sh_out')}

            _ax = getattr(self, 'axial_model', None)
            _Qs = None if Q_sink_fn is None else np.asarray(Q_sink_fn(T), float)
            if _ax is not None and _ps_active:
                T_new, _, _ = storage.update_temps(
                    T, -Q_rad, dt, sub_steps, self.T_sur, self.block_heat_loss,
                    dQ_dTw=-_ax.last['dQ_dTw'], Q_out=_Qs)
            else:
                T_new, _, _ = storage.update_temps(
                    T, -Q_rad, dt, sub_steps, self.T_sur, self.block_heat_loss, Q_out=_Qs
                )
            storage.block.temp_dist = T_new
            t += dt

            # Check powerstick cutoff after this step.
            if T_ps_cutoff is not None and _ps_active and _Tavg(T_new) <= T_ps_cutoff:
                _ps_active   = False
                t_ps_removed = t
                print(f"[discharge] T_avg={_Tavg(T_new):.0f} K reached cutoff "
                      f"{T_ps_cutoff:.0f} K at t={t/3600:.4f} hr ({t:.1f} s) — "
                      f"powerstick removed, conduction-only phase begins.")
                if stop_at_cutoff:
                    break

            time_arr.append(t)
            Q_sink_arr.append(0.0 if _Qs is None else float(_Qs.sum()))
            Q_arr.append(Q_rad.sum())
            Q_cell_arr.append(Q_cell.sum())
            Q_elec_arr.append(Q_elec.sum())
            T_avg_arr.append(_Tavg(T_new))
            T_inner_arr.append(float(T_new[:, 0] @ _wz))
            T_outer_arr.append(float(T_new[:, -1] @ _wz))
            T_liner_arr.append(T_liner.mean())
            T_liner_out_arr.append(np.nanmean(_st['T_liner_out']))
            T_foil_inner_arr.append(np.nanmean(_st['T_foil_inner']))
            T_sh_in_arr.append(np.nanmean(_st['T_sh_in']))
            T_sh_out_arr.append(np.nanmean(_st['T_sh_out']))
            f_shade_arr.append(f_shade)
            _on = _ps_active or t_ps_removed == t
            for k in _AX_KEYS:
                ax_rec[k].append(_lim(k) if _on else np.nan)
            _dg = self._axial_diag(T, Q_rad, _on)
            for k in diag_rec:
                diag_rec[k].append(_dg[k])
            ps_active_arr.append(_ps_active)

            if deposition_fn is not None:
                if _ps_active:
                    _T_e = self._dep_T_emitter(T_new[:, 0].mean(), T_liner.mean(),
                                                np.nanmean(_st['T_foil_inner']))
                    _d = deposition_fn(_T_e, **_dep_kw)
                    dep_rate_arr.append(_d['total'])
                    dep_C_arr.append(_d['C'])
                    dep_metal_arr.append(_d['metal'])
                else:
                    dep_rate_arr.append(0.0)
                    dep_C_arr.append(0.0)
                    dep_metal_arr.append(0.0)

            if snap_every and t >= next_snap:
                snap_times.append(t)
                temp_snaps.append(T_new.copy())
                liner_out_snaps.append(_lo_snap(_st))
                next_snap += snap_every

            if print_interval and (t - last_print_t >= print_interval):
                _tag = '' if _ps_active else '  [cond-only]'
                print(f"t={t/3600:.2f} hr  T_avg={_Tavg(T_new):.0f} K  "
                      f"Q_cell={Q_cell.sum()/1e3:.1f} kW  f_sh={f_shade:.3f}  "
                      f"dt={dt:.0f} s{_tag}")
                last_print_t = t

        if snap_times[-1] != t:
            snap_times.append(t)
            temp_snaps.append(storage.block.temp_dist.copy())
            liner_out_snaps.append(_lo_snap(self._step_temps) if _ps_active else np.full(_n_lo, np.nan))

        _S_PER_YR = 3.154e7
        if dep_rate_arr:
            _ta = np.array(time_arr)
            dep_C_nm     = np.trapz(dep_C_arr,     _ta) / _S_PER_YR
            dep_metal_nm = np.trapz(dep_metal_arr, _ta) / _S_PER_YR
            dep_total_nm = dep_C_nm + dep_metal_nm
        else:
            dep_C_nm = dep_metal_nm = dep_total_nm = 0.0

        return {
            'time':           np.array(time_arr),
            'Q_rad':          np.array(Q_arr),
            'Q_cell':         np.array(Q_cell_arr),
            'P_elec':         np.array(Q_elec_arr),
            'T_avg':          np.array(T_avg_arr),
            'T_inner':        np.array(T_inner_arr),
            'T_outer':        np.array(T_outer_arr),
            'T_liner':        np.array(T_liner_arr),
            'T_liner_out':    np.array(T_liner_out_arr),
            'T_foil_inner':   np.array(T_foil_inner_arr),
            'T_sh_in':        np.array(T_sh_in_arr),
            'T_sh_out':       np.array(T_sh_out_arr),
            'f_shade':        np.array(f_shade_arr),
            **{k: np.array(v) for k, v in ax_rec.items()},   # axial-model extras (see _AX_KEYS)
            **{k: np.array(v) for k, v in diag_rec.items()},  # see _axial_diag
            'liner_out_snaps': np.array(liner_out_snaps),      # liner outer-face T per active ring at snap_times
            'ps_active':      np.array(ps_active_arr),
            'Q_sink':         np.array(Q_sink_arr),
            't_ps_removed':   t_ps_removed,
            'snap_times':     np.array(snap_times),
            'temp_snapshots': np.array(temp_snaps),
            'dep_rate':       np.array(dep_rate_arr),
            'dep_C_nm':       dep_C_nm,
            'dep_metal_nm':   dep_metal_nm,
            'dep_total_nm':   dep_total_nm,
        }

    def run_heating_simulation(self, heater, T_block_max, end_time, dt, sub_steps,
                               snap_every=None, print_interval=0, Q_sink_fn=None):
        """
        Transient charging simulation with a central resistive heating element.

        The heater radiates outward to the graphite inner wall.  At each timestep
        the heater temperature is solved from a two-surface coaxial radiation
        enclosure; the resulting per-segment heat flux is passed to
        StorageModule.update_temps as a positive source at the inner wall.

        Stops when the volume-weighted T_avg >= T_block_max or after int(end_time/dt)
        steps. T_avg, T_inner and T_outer are volume- / height-weighted.

        Q_sink_fn: optional f(T_dist) -> heat [W] removed per ring (nA,) or per node
        (nA, nR), held over each step (e.g. tank.CellLossSink: roof / floor losses).
        The applied total is recorded as 'Q_sink' [W].

        Returns dict with keys:
          'time', 'Q_heater', 'T_heater', 'T_avg', 'T_inner', 'T_outer',
          'T_liner', 'snap_times', 'temp_snapshots'
        """
        sigma   = 5.67036713e-8
        storage = self.storage
        r_in    = storage.block.rad_in
        n_ax    = storage.num_ax_segments
        # Heater spans the full block height; per-ring area follows ring height
        # (non-uniform when the block is taller than the powerstick).
        seg_len = storage.axial_dz()

        A_h_seg  = np.pi * heater.diameter * seg_len
        A_gr_seg = np.pi * 2.0 * r_in * seg_len
        eps_h    = heater.emissivity

        def _q_segs(T_h, T_inner_arr):
            Q_seg = np.zeros(n_ax)
            for i in range(n_ax):
                eps_gr   = gr_emis(T_inner_arr[i])
                R_enc    = ((1.0 - eps_h) / eps_h + 1.0
                            + (A_h_seg[i] / A_gr_seg[i]) * (1.0 - eps_gr) / eps_gr)
                Q_seg[i] = sigma * (T_h**4 - T_inner_arr[i]**4) * A_h_seg[i] / R_enc
            return Q_seg

        def _total_q(T_h, T_inner_arr):
            return float(_q_segs(T_h, T_inner_arr).sum())

        def _solve_T_heater(T_inner_arr):
            T_inner_max   = float(T_inner_arr.max())
            if T_inner_max >= heater.max_temperature:
                return T_inner_max, np.zeros(n_ax)
            q_at_max_temp = _total_q(heater.max_temperature, T_inner_arr)
            if q_at_max_temp >= heater.max_power:
                T_lo = T_inner_max + 0.1
                T_hi = heater.max_temperature
                try:
                    T_h = brentq(
                        lambda Th: _total_q(Th, T_inner_arr) - heater.max_power,
                        T_lo, T_hi, xtol=1.0, maxiter=50,
                    )
                except ValueError:
                    T_h = heater.max_temperature
            else:
                T_h = heater.max_temperature
            return T_h, _q_segs(T_h, T_inner_arr)

        # Rings facing the liner: the top num_ax_steps rings (bottom-up indexing;
        # equals all rings when block height == powerstick length).
        _act = range(n_ax - self.powerstick.num_ax_steps, n_ax)

        def _liner_avg(T_d):
            if not self.powerstick.liner_bool:
                return 0.0
            tl = np.zeros(len(_act))
            for n, i in enumerate(_act):
                _, tli, _ = self.spectral_net_radiation(float(T_d[i, 0]))
                tl[n] = tli
            return float(tl.mean())

        # volume weights (radial rings x ring heights) for the block average
        _nR = storage.num_rad_segments
        _r = r_in + np.arange(_nR + 1) * (storage.block.rad_out - r_in) / _nR
        _W = np.diff(_r ** 2)[None, :] * seg_len[:, None]
        _W = _W / _W.sum()
        _wz = seg_len / seg_len.sum()
        _Tavg = lambda T_d: float((T_d * _W).sum())

        T        = storage.block.temp_dist.copy()
        T_h_0, _ = _solve_T_heater(T[:, 0])

        n_steps    = int(end_time / dt)
        snap_steps = max(1, round(snap_every / dt)) if snap_every else n_steps

        time_list    = [0.0]
        Q_list       = [0.0]
        T_h_list     = [T_h_0]
        T_avg_list   = [_Tavg(T)]
        T_inner_list = [float(T[:, 0] @ _wz)]
        T_outer_list = [float(T[:, -1] @ _wz)]
        T_liner_list = [_liner_avg(T)]
        Q_sink_list  = [0.0 if Q_sink_fn is None else float(np.sum(Q_sink_fn(T)))]
        snap_times   = [0.0]
        temp_snaps   = [T.copy()]

        for step in range(n_steps):
            T_h, Q_segs = _solve_T_heater(T[:, 0])
            _Qs = None if Q_sink_fn is None else np.asarray(Q_sink_fn(T), float)
            T_new, _, _ = storage.update_temps(
                T, Q_segs, dt, sub_steps, self.T_sur, self.block_heat_loss, Q_out=_Qs,
            )
            Q_sink_list.append(0.0 if _Qs is None else float(_Qs.sum()))
            storage.block.temp_dist = T_new
            T = T_new

            idx = step + 1
            t   = idx * dt
            time_list.append(t)
            Q_list.append(float(Q_segs.sum()))
            T_h_list.append(T_h)
            T_avg_list.append(_Tavg(T_new))
            T_inner_list.append(float(T_new[:, 0] @ _wz))
            T_outer_list.append(float(T_new[:, -1] @ _wz))
            T_liner_list.append(_liner_avg(T_new))

            if idx % snap_steps == 0:
                snap_times.append(t)
                temp_snaps.append(T_new.copy())

            if print_interval and (idx % print_interval == 0):
                phase = 1 if T_h < heater.max_temperature - 1.0 else 2
                print(
                    f"t={t/3600:.2f} hr  T_inner={T_inner_list[-1]:.0f} K  "
                    f"T_avg={T_avg_list[-1]:.0f} K  T_liner={T_liner_list[-1]:.0f} K  "
                    f"T_heater={T_h:.0f} K  Q={Q_segs.sum()/1e3:.1f} kW  [phase {phase}]"
                )

            if T_avg_list[-1] >= T_block_max:
                print(f"Block fully charged: T_avg={T_avg_list[-1]:.1f} K >= {T_block_max} K "
                      f"at t={t/3600:.2f} hr")
                break

        if snap_times[-1] != time_list[-1]:
            snap_times.append(time_list[-1])
            temp_snaps.append(storage.block.temp_dist.copy())

        return {
            'time':           np.array(time_list),
            'Q_heater':       np.array(Q_list),
            'T_heater':       np.array(T_h_list),
            'T_avg':          np.array(T_avg_list),
            'T_inner':        np.array(T_inner_list),
            'T_outer':        np.array(T_outer_list),
            'T_liner':        np.array(T_liner_list),
            'Q_sink':         np.array(Q_sink_list),
            'snap_times':     np.array(snap_times),
            'temp_snapshots': np.array(temp_snaps),
        }
