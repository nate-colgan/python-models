"""
Build a tinless simulation from a YAML config dict.

Reproduces the setup in the main cell of `tinless system model.ipynb`
(cell b2e4f810): input parsing, block sizing from the required discharge
energy, shade/liner geometry auto-correction, component construction and
deposition kwargs. One entry point for the notebook, sweeps and the optimizer.

Usage
-----
    from thermal.build import build_tinless, load_config
    cfg = load_config('tinless_model_input.yaml')
    m = build_tinless(cfg, {'storage.L': 4.0, 'powerstick.liner_thickness': 0.01})
    results = m.sim.run_simulation(**m.run_kwargs)
    hm = build_heater(m)                       # matching single-heater charge model
    charge = hm.sim.run_heating_simulation(hm.heater, end_time=..., **hm.run_kwargs)

`overrides` uses dotted keys into the config; the input dict is not modified.
"""
import copy
from types import SimpleNamespace

import numpy as np

from .components import GraphiteBlock, Pipe, StorageModule, PowerStick
from .properties import storage_rho, storage_cp
from .tinless import TinlessSimulation, HeatingElement
from .viewfactors import axial_mesh, cavity_view_factors, porous_shade_view_factors, DEFAULT_RAYS
from .axial import AxialRadiation, VaneHybridRadiation


def load_config(path='tinless_model_input.yaml'):
    """Read the YAML input, accepting UTF-8 or the Windows code page (the notebook's
    YAML cell writes with the platform default encoding)."""
    import yaml
    raw = open(path, 'rb').read()
    for enc in ('utf-8', 'cp1252', 'latin-1'):
        try:
            return yaml.safe_load(raw.decode(enc))
        except UnicodeDecodeError:
            continue


def build_heater(m, area_ratio=None):
    """Single-heater charge model matching a build_tinless model `m` (checkerboard: one
    heater per stick, N_h = N_ps). The heater cell holds the same graphite as the stick
    cell (same volume, height, axial mesh and radial rings) around the heater cavity:

        r_hcav = heater.area_ratio * r_heater_eff,   r_heater_eff = 2 side_len / pi
        r_out  = sqrt(V_block / (pi H_block) + r_hcav^2)

    The heater spans the full block height. Block starts at T_min. Outer wall adiabatic
    (symmetry line between heater and stick); roof / floor losses via Q_sink_fn.
    -> SimpleNamespace(sim, storage, block, heater, r_in, r_out, r_heater, T_min, T_max,
       run_kwargs)."""
    cfg = m.cfg
    h = cfg['heater']
    ar = float(h['area_ratio'] if area_ratio is None else area_ratio)
    r_h = 2 * h['side_len'] / np.pi
    r_in = ar * r_h
    st = m.storage
    H = float(m.H_block)
    V = np.pi * (st.block.rad_out ** 2 - st.block.rad_in ** 2) * H
    r_out = float(np.sqrt(V / (np.pi * H) + r_in ** 2))
    nA, nR = st.num_ax_segments, st.num_rad_segments
    block = GraphiteBlock(initial_temp_distribution=np.full((nA, nR), float(m.T_min)),
                          inner_radius=r_in, outer_radius=r_out, length=H, th_ins=st.block.th_ins)
    storage = StorageModule(N_pipes=1, Pipe=Pipe(OD=0.02, length=H, wall_thickness=0.001, roughness=1e-5,
                                                 th_ins=0.0),
                            Block=block, num_ax_segments=nA, num_rad_segments=nR, bypass_range=[0, 1])
    if getattr(st, 'axial_edges', None) is not None:
        storage.set_axial_mesh(st.axial_edges)
    # TinlessSimulation needs a powerstick; unused in the charge (no liner -> no liner solve)
    ps = PowerStick(axial_steps=nA, width=m.powerstick.width, height=H, cooling_temp=m.powerstick.cooling_temp,
                    f_bare=m.powerstick.f_bare, f_active=m.powerstick.f_active, liner_bool=False,
                    liner_width=None, emissivity_table=m.emissivity_table)
    sim = TinlessSimulation(storage, ps, T_sur=cfg['T_sur'], block_heat_loss=cfg['block_heat_loss_bool'])
    heater = HeatingElement(diameter=2 * r_h, max_temperature=h['max_temperature'], max_power=h['max_power'],
                            emissivity=h['emissivity'])
    if r_h >= r_in:
        raise ValueError(f"heater radius {r_h:.3f} m >= cavity radius {r_in:.3f} m (heater.area_ratio {ar})")
    kw = dict(T_block_max=float(m.T_max), dt=cfg['dt_short'], sub_steps=cfg['sub_steps'])
    return SimpleNamespace(sim=sim, storage=storage, block=block, heater=heater, r_in=r_in, r_out=r_out,
                           r_heater=r_h, area_ratio=ar, T_min=float(m.T_min), T_max=float(m.T_max), V=V,
                           H=H, run_kwargs=kw)


def _apply_overrides(cfg, overrides):
    cfg = copy.deepcopy(cfg)
    for dotted, val in (overrides or {}).items():
        node = cfg
        keys = dotted.split('.')
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = val
    return cfg


def build_tinless(cfg, overrides=None, verbose=False, geometry_only=False):
    """geometry_only=True: build geometry/mesh/components but do not require a
    runnable radiation model (used while the axial model is under development)."""
    cfg = _apply_overrides(cfg, overrides)
    say = print if verbose else (lambda *a, **k: None)

    liner_bool = cfg['liner_bool']
    liner_material = cfg.get('liner_material', 'tac')
    raw_emis = cfg['emissivity_tables'][0]['value']
    emissivity_table = [(tuple(r[0]), float(r[1]), float(r[2]), float(r[3])) for r in raw_emis]

    s, p, sh = cfg['storage'], cfg['powerstick'], cfg.get('shade', {})
    avg_flux, duration_hr = s['avg_flux'], s['duration_hr']
    T_min, block_temp_range = s['T_min'], s['block_temp_range']
    L, th_ins = s['L'], s['th_ins']
    num_ax, num_rad = s['num_ax_segments'], s['num_rad_segments']
    H_block = s.get('block_height', None)
    H_block = L if H_block is None else float(H_block)
    if H_block < L - 1e-12:
        raise ValueError(f"storage.block_height ({H_block}) < storage.L ({L})")
    rad_cfg = cfg.get('radiation', {}) or {}
    axial = bool(rad_cfg.get('axial', False))
    lb_cfg = cfg.get('liner_bottom', {}) or {}
    lb_enabled = bool(lb_cfg.get('enabled', False))
    ce_cfg = cfg.get('cavity_ends', {}) or {}

    side_len, area_ratio = p['side_len'], p['area_ratio']
    f_bare, f_active, T_ps = p['f_bare'], p['f_active'], p['cooling_temp']
    liner_side_len = p['liner_side_len']
    liner_thickness = p.get('liner_thickness', 0.0)
    liner_tungsten_foil = p.get('tungsten_foil', False)

    shade_bool = sh.get('shade_bool', cfg.get('shade_bool', False))
    ratio = sh.get('liner_to_shade_perimeter_ratio', None)
    shade_side_len = ratio * liner_side_len if ratio else None
    shade_thickness = sh.get('thickness', 0.05)
    shade_eps = sh.get('surface_emissivity', 0.8)
    shade_outer_foil = sh.get('outer_foil', False)
    shade_nlayers = sh.get('num_conduction_layers', 5)
    corner_gap = sh.get('corner_gap', 0.003175)
    per = ratio * 4 * liner_side_len if ratio else None
    f_shade_min = 8 * shade_thickness / per if per else 0.03
    f_shade_max = 1 - 8 * corner_gap / per if per else 0.99

    # ── Cavity size ──────────────────────────────────────────────────────────
    # Without a shade: r_in = area_ratio * r_ps_eff (YAML). With a shade (and liner)
    # area_ratio is IGNORED: the cavity is the smallest that fits the shade around
    # the liner, and area_ratio becomes an output (r_in / r_ps_eff).
    #   shade.model: vanes  -> square cavity side S = 4 (liner side + √2·c) + 2 c
    #                          (open face vanes clear the wall, open corner vanes the
    #                          liner corners; c = shade.vane_clearance); r_in = 2 S / π
    #   shade.model: porous -> r_in = (1 + √2) r_liner + 0.01 m (original notebook rule)
    r_ps_eff = 2 * side_len / np.pi
    shade_model = str(sh.get('model', 'vanes')).lower()
    if shade_model not in ('vanes', 'porous'):
        raise ValueError(f"shade.model must be 'vanes' or 'porous', got {shade_model!r}")
    vane_clear = float(sh.get('vane_clearance', 0.005))
    if shade_bool and liner_bool:
        if shade_model == 'vanes':
            from .correlations import vane_layout_for_liner
            _, S_min = vane_layout_for_liner(liner_side_len, shade_thickness, corner_gap, vane_clear)
            r_in = 2.0 * S_min / np.pi
            say(f"[geometry] shade: cavity sized for vanes around a {liner_side_len*1e3:.0f} mm liner: "
                f"side {S_min*1e3:.0f} mm (r_in {r_in*1e3:.0f} mm); area_ratio input ignored")
        else:
            r_in = (1.0 + np.sqrt(2)) * (2 * liner_side_len / np.pi) + 0.01
            say(f"[geometry] shade: cavity r_in = (1+sqrt2) r_liner + 10 mm = {r_in*1e3:.0f} mm; "
                f"area_ratio input ignored")
        area_ratio = r_in / r_ps_eff
    else:
        r_in = r_ps_eff * area_ratio

    # ── Sizing: block volume from required discharge energy ──────────────────
    A_ps = 4 * side_len * L
    T_max = T_min + block_temp_range
    T_avg_ = T_min + block_temp_range / 2
    f_cell = (1.0 - f_bare) * f_active
    f_inact = (1.0 - f_bare) * (1.0 - f_active)
    ps_tmp = PowerStick(axial_steps=1, width=side_len, height=L, cooling_temp=T_ps,
                        f_bare=f_bare, f_active=f_active, liner_bool=liner_bool,
                        liner_width=liner_side_len, emissivity_table=emissivity_table)
    inc_bins, _, _ = ps_tmp.incident_energy(T_min - 100, 1 / area_ratio)
    inc_tot = sum(inc_bins.values())
    f_Q = sum(inc_bins[r[0]] / inc_tot * f_cell * r[1] / (f_bare * r[2] + f_cell * r[1] + f_inact * r[3])
              for r in emissivity_table)
    E_req = avg_flux * 1e4 * A_ps * f_cell / f_Q * duration_hr * 3600
    e_dens = storage_rho(T_avg_) * storage_cp(T_avg_) * block_temp_range
    V_block = E_req / e_dens * float(s.get('volume_factor', 1.0) or 1.0)   # optimizer: clean-run correction
    r_out = np.sqrt(V_block / (np.pi * H_block) + r_in ** 2)
    say(f"Block: r_in={r_in:.3f} m  r_out={r_out:.3f} m  L={L} m  V={V_block:.2f} m^3")

    # ── Shade / liner geometry auto-correction (as in notebook) ──────────────
    if shade_bool and liner_bool:
        shade_min = (1.0 + np.sqrt(2) / 2.0) * liner_side_len
        if shade_side_len is None or shade_side_len < shade_min:
            shade_side_len = shade_min
            say(f"[geometry] shade_side_len auto-set to {shade_side_len*1e3:.1f} mm")
        per = 4.0 * shade_side_len
        f_shade_min = 8.0 * shade_thickness / per
        f_shade_max = 1.0 - 8.0 * corner_gap / per

    # ── Rotating-vane shade (axial model): square cavity, octagon of 8 vanes ──
    # shade.model: vanes (default) | porous (thin porous sheet, Phase 3 model).
    # Vane length l = liner side + √2·clearance (clears the liner corners when
    # open); the cavity side must be >= 4 l + 2·clearance (open face vanes clear
    # the wall). The block bore is enlarged to fit if needed.
    vane_lay = None
    if shade_bool and liner_bool and axial and shade_model == 'vanes':
        _, S_min = vane_layout_for_liner(liner_side_len, shade_thickness, corner_gap, vane_clear)
        S_cav = np.pi * r_in / 2.0                      # square cavity with the bore's perimeter
        if S_cav < S_min - 1e-9:
            S_cav = S_min
            r_in = 2.0 * S_cav / np.pi
            area_ratio = r_in / r_ps_eff
            r_out = np.sqrt(V_block / (np.pi * H_block) + r_in ** 2)
            say(f"[geometry] vanes: cavity side increased to {S_cav*1e3:.0f} mm (r_in {r_in*1e3:.0f} mm, "
                f"r_out {r_out*1e3:.0f} mm)")
        vane_lay, _ = vane_layout_for_liner(liner_side_len, shade_thickness, corner_gap, vane_clear, S=S_cav)
        shade_side_len = vane_lay['l'] * (1 + np.sqrt(2))   # octagon perimeter / 4 (for the sim's checks)

    # ── Axial mesh (z = 0 at cavity floor, bottom-up) ───────────────────────
    # Powerstick + liner (length L) hang top-aligned in a block of height H_block.
    # storage.num_ax_segments = TOTAL block rings over H_block, split between the
    # lower zone and the active zone by height with a ring edge exactly at the
    # liner-bottom plane (see viewfactors.axial_mesh). One powerstick step per
    # active ring.
    t_lb = liner_thickness                      # liner-bottom plate = liner wall thickness
    t_rgf = float(lb_cfg.get('rgf_thickness', 0.025) or 0.0)
    has_low = H_block - L > 1e-12
    if lb_enabled and not has_low:
        say("[liner_bottom] block_height == L: no space below the liner; liner-bottom model ignored")
        lb_enabled = False
    if has_low:
        H_low = H_block - L - t_lb - t_rgf
        if H_low < 0:
            raise ValueError(f"H_low = H_block - L - t_lb - t_rgf = {H_low:.4f} m < 0: "
                             "block_height too small for liner bottom + RGF pad")
    else:
        H_low = 0.0
    if has_low and not axial:
        raise ValueError("block_height > L needs radiation.axial: true "
                         "(the local per-segment model cannot represent the lower zone)")
    n_act, n_low, edges, edges_act, z_act0 = axial_mesh(L, H_block, num_ax)
    n_ax_tot = n_act + n_low
    act_slice = slice(n_low, n_ax_tot)          # block rings facing the powerstick
    mesh = SimpleNamespace(edges=edges, edges_act=edges_act, n_act=n_act, n_low=n_low,
                           n_ax=n_ax_tot, act_slice=act_slice, z_act0=z_act0,
                           H_low=H_low, t_lb=t_lb, t_rgf=t_rgf, dz=np.diff(edges))
    say(f"Mesh: H_block={H_block} m, L={L} m, {n_low} lower + {n_act} active rings "
        f"(dz_low={z_act0/max(n_low,1):.3f} m, dz_act={L/n_act:.3f} m)" if n_low else
        f"Mesh: H_block={H_block} m = L, {n_act} rings (dz={L/n_act:.3f} m)")

    # ── Deposition degradation (optimizer) ──────────────────────────────────
    # cfg['degradation'] = {'cell': D_c, 'bare': D_b, 'emis_end': 0.8}
    #   D_c: cell efficiency x (1 - D_c); inactive (on-cell) emissivity rises linearly to emis_end
    #   D_b: bare-area emissivity rises linearly to emis_end (bare areas are cleaned separately)
    # {'frac': D} sets both. The sizing above keeps the fresh table, so a degraded model has
    # exactly the fresh geometry.
    deg = cfg.get('degradation') or {}
    D_c = float(deg.get('cell', deg.get('frac', 0.0)) or 0.0)
    D_b = float(deg.get('bare', deg.get('frac', 0.0)) or 0.0)
    D_deg = D_c
    emis_run = emissivity_table
    if D_c > 0 or D_b > 0:
        e_end = float(deg.get('emis_end', 0.8))
        emis_run = [(b, a, e_b + D_b * (e_end - e_b), e_i + D_c * (e_end - e_i))
                    for (b, a, e_b, e_i) in emissivity_table]

    # ── Components ───────────────────────────────────────────────────────────
    block = GraphiteBlock(initial_temp_distribution=np.full((n_ax_tot, num_rad), float(T_max)),
                          inner_radius=r_in, outer_radius=r_out, length=H_block, th_ins=th_ins)
    storage = StorageModule(N_pipes=1, Pipe=Pipe(OD=0.02, length=H_block, wall_thickness=0.001,
                                                 roughness=1e-5, th_ins=0.0),
                            Block=block, num_ax_segments=n_ax_tot, num_rad_segments=num_rad,
                            bypass_range=[0, 1])
    if has_low:
        storage.set_axial_mesh(edges)
    powerstick = PowerStick(axial_steps=n_act, width=side_len, height=L, cooling_temp=T_ps,
                            f_bare=f_bare, f_active=f_active, liner_bool=liner_bool,
                            liner_width=liner_side_len, emissivity_table=emis_run)
    # Liner bottom + RGF pad (axial model only). Disabled: pad underside reradiating
    # (adiabatic liner bottom).
    pad = None
    if axial and lb_enabled:
        eps_R = lb_cfg.get('rgf_outer_emissivity', None)
        T_cap = (cfg.get('bottom_cap', {}) or {}).get('temperature', None)
        pad = dict(t_rgf=t_rgf, t_lb=t_lb,
                   eps_R=float(sh.get('surface_emissivity', 0.8) if eps_R is None else eps_R),
                   T_cap=float(T_ps if T_cap is None else T_cap),
                   include_cells=bool(lb_cfg.get('include_flux_to_cells_in_power', False)))
    sim = TinlessSimulation(storage, powerstick, T_sur=cfg['T_sur'],
                            block_heat_loss=cfg['block_heat_loss_bool'],
                            shade_bool=shade_bool, shade_side_len=shade_side_len,
                            shade_thickness=shade_thickness, shade_surface_emissivity=shade_eps,
                            shade_outer_foil=shade_outer_foil, shade_num_conduction_layers=shade_nlayers,
                            f_shade_min=f_shade_min, f_shade_max=f_shade_max,
                            nominal_cell_flux=avg_flux, liner_material=liner_material,
                            liner_thickness=liner_thickness, liner_tungsten_foil=liner_tungsten_foil)
    sim.cell_eff_factor = 1.0 - D_deg

    # ── Deposition kwargs (as in notebook) ───────────────────────────────────
    r_liner_eff = 2 * liner_side_len / np.pi
    if liner_bool:
        d_dep, ar_dep = r_liner_eff - r_ps_eff, r_liner_eff / r_ps_eff
    else:
        d_dep, ar_dep = r_in - r_ps_eff, r_in / r_ps_eff
    surf_mat = ('tungsten' if liner_tungsten_foil else
                'tac' if (liner_bool and liner_material == 'tac') else 'graphite')
    dep_kwargs = dict(T_ps=T_ps, d=d_dep, area_ratio=ar_dep, surface_material=surf_mat,
                      th_tac=liner_thickness,
                      th_w=p.get('tungsten_foil_thickness', 25e-6) if liner_tungsten_foil else 0.0)

    run_kwargs = dict(end_time=cfg['end_time'], dt_short=cfg['dt_short'], dt_medium=cfg['dt_medium'],
                      dt_long=cfg['dt_long'], sub_steps=cfg['sub_steps'],
                      dt_short_duration=cfg.get('dt_short_duration', 3600), T_ps_cutoff=T_min)

    # ── Full-height enclosure radiation (radiation.axial: true) ──────────────
    vfs = None
    if axial and not geometry_only:
        g = dict(r_in=r_in, r_sh=2 * shade_side_len / np.pi if shade_bool else None,
                 r_liner=r_liner_eff, L_ps=L, H=H_block, t_lb=t_lb, t_rgf=t_rgf, H_low=H_low,
                 z_act0=z_act0, edges=edges, edges_act=edges_act, n_act=n_act, n_low=n_low)
        n_rays = int(rad_cfg.get('vf_rays', DEFAULT_RAYS) or DEFAULT_RAYS)
        cache_dir = rad_cfg.get('vf_cache_dir', None)
        if vane_lay is not None:   # rotating vanes: ray-traced closed / 10 deg / open positions
            sim.axial_model = VaneHybridRadiation(
                sim, g, vane_lay, n_rays=int(rad_cfg.get('vane_rays', 2 ** 14) or 2 ** 14),
                cache_dir=cache_dir, pad=pad)
            ax = sim.axial_model
            f_shade_min, f_shade_max = 0.0, 1.0                 # f = 1 - w (0 open, 1 closed)
            sim.f_shade_min, sim.f_shade_max = f_shade_min, f_shade_max
            say(f"Axial radiation: vanes (l = {vane_lay['l']*100:.1f} / {vane_lay['l']*np.sqrt(2)*100:.1f} cm, "
                f"t = {vane_lay['t']*1e3:.1f} mm), closed / 10 deg / open positions, "
                f"{ax.anchors['open'].N} surfaces per limit")
        elif shade_bool:     # shade = porous sheet, straight-through transmission (always ray-traced)
            vfs = {'porous': porous_shade_view_factors(g, n_rays=n_rays, cache_dir=cache_dir)}
        else:
            vfs = cavity_view_factors(g, shaded=False,
                                      method=rad_cfg.get('vf_method', 'analytic') or 'analytic',
                                      n_rays=n_rays, cache_dir=cache_dir)
        if vane_lay is None:
            sim.axial_model = AxialRadiation(sim, g, vfs, pad=pad)
            say(f"Axial radiation: {sim.axial_model.N} surfaces, view factors "
                f"{'/'.join(v.method for v in vfs.values())}")

    return SimpleNamespace(
        vfs=vfs, cfg=cfg, sim=sim, storage=storage, block=block, powerstick=powerstick, mesh=mesh,
        axial=axial, liner_bottom=dict(lb_cfg, enabled=lb_enabled, thickness=t_lb, rgf_thickness=t_rgf),
        cavity_ends=dict(ce_cfg), radiation=dict(rad_cfg),
        emissivity_table=emissivity_table, run_kwargs=run_kwargs, dep_kwargs=dep_kwargs,
        L=L, H_block=H_block, r_ps_eff=r_ps_eff, r_in=r_in, r_out=r_out, area_ratio=area_ratio,
        r_liner_eff=r_liner_eff, shade_side_len=shade_side_len, shade_bool=shade_bool,
        liner_bool=liner_bool, liner_material=liner_material,
        f_shade_min=f_shade_min, f_shade_max=f_shade_max,
        T_min=T_min, T_max=T_max, E_req=E_req, V_block=V_block, f_Q=f_Q, A_ps=A_ps,
        avg_flux=avg_flux, degradation=dict(cell=D_c, bare=D_b),
    )
