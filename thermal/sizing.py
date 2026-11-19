"""Block sizing so the shades just reach fully open as the discharge ends.

The stick holds the target cell flux while the shades regulate. Once they are fully open
the flux, and the power, fall. `shade_open_resize` moves T_min to the block temperature
at which the shades open, and rescales the block volume so that point is reached at the
target duration. Both change the radial temperature profile, and so the shade-open
temperature, so it iterates. T_max is held fixed.

Every pass is the same discharge run as the model notebook's main cell (build_tinless,
roof + floor cell losses, deposition, stick out at T_min). When the shades are still
regulating at T_min, the opening is extrapolated from the shade position vs block
temperature over the last regulating steps; the extrapolation shrinks to zero as the
iteration converges.

Block temperature = the volume-weighted block average (r['T_avg'], which the discharge
cutoff also uses).
"""
import copy
from types import SimpleNamespace

import numpy as np

from .build import build_tinless
from .deposition import deposition_rate_tinless
from . import tank as tk


class FluxUnreachable(ValueError):
    """The shades are fully open from the start: the target cell flux is never reached."""


def run_discharge(cfg, print_interval=0, dt_scale=1.0, dt_at_shade_limit=None, deposition=True,
                  stop_at_cutoff=False):
    """One discharge as the model notebook's main cell runs it -> (m, results).
    dt_scale / dt_at_shade_limit coarsen the time steps and stop_at_cutoff skips the
    conduction-only tail after the stick comes out (optimizer screening); the defaults
    reproduce the notebook run."""
    m = build_tinless(cfg)
    t_cfg = cfg.get('tank')
    sink = tk.cell_loss_sink(m, t_cfg) if t_cfg and t_cfg.get('block_axial_losses', True) else None
    kw = dict(m.run_kwargs, snap_every=600, print_interval=print_interval, stop_at_cutoff=stop_at_cutoff)
    for k in ('dt_short', 'dt_medium', 'dt_long'):
        kw[k] = kw[k] * dt_scale
    if dt_at_shade_limit is not None:
        kw['dt_at_shade_limit'] = dt_at_shade_limit
    if deposition:
        kw.update(deposition_fn=deposition_rate_tinless, deposition_kwargs=m.dep_kwargs)
    r = m.sim.run_simulation(**kw, Q_sink_fn=None if sink is None else sink.for_storage(m.storage))
    return m, r


def cell_flux(m, r):
    """Cell flux on the active area [W/cm2] per recorded step (as the shade controller)."""
    ps = m.powerstick
    A_act = 4 * ps.width * ps.height * (1 - ps.f_bare) * ps.f_active * 1e4
    return np.asarray(r['Q_cell'], float) / A_act


def shade_open_point(m, r, tol=1e-6, n_fit=8):
    """Time [s] and block average [K] at which the shades become fully open for good.

    Seen in the run: placed between the last regulating step and the first open one, where
    the falling cell flux, extrapolated back from the first two open steps, equals the target.
    Not seen (still regulating when the stick comes out): extrapolated linearly from the
    last `n_fit` regulating steps, shade position vs block temperature and block temperature
    vs time (constant power while regulating).

    -> dict(t_open, T_open, seen) or None if the shades never regulate (closed to the end).
    Raises FluxUnreachable if the shades are fully open from the start."""
    fs = np.asarray(r['f_shade'], float)
    on = np.asarray(r['ps_active'], bool)
    t, Ta = np.asarray(r['time'], float), np.asarray(r['T_avg'], float)
    is_open = on & (fs <= m.f_shade_min + tol)
    shut = np.where(on & ~is_open)[0]
    if len(shut) == 0:
        raise FluxUnreachable("shades fully open from the start: the target flux is never reached")
    i = shut[-1] + 1
    if i < len(fs) and is_open[i]:
        t_open = t[i]
        q, q_t = cell_flux(m, r), float(m.avg_flux)
        if i + 1 < len(q) and on[i + 1] and q[i] > q[i + 1] and q[i] < q_t:
            slope = (q[i] - q[i + 1]) / (t[i + 1] - t[i])
            t_open = float(np.clip(t[i] - (q_t - q[i]) / slope, t[i - 1], t[i]))
        return dict(t_open=t_open, T_open=float(np.interp(t_open, t[i - 1:i + 1], Ta[i - 1:i + 1])), seen=True)
    reg = np.where(on & (fs > m.f_shade_min + tol) & (fs < m.f_shade_max - tol))[0]
    if len(reg) < 2:
        return None
    k = reg[-n_fit:]
    b, a = np.polyfit(Ta[k], fs[k], 1)                 # f_shade = a + b T
    T_open = (m.f_shade_min - a) / b
    d, c = np.polyfit(t[k], Ta[k], 1)                  # T_avg = c + d t
    return dict(t_open=float((T_open - c) / d), T_open=float(T_open), seen=False)


def shade_open_resize(cfg, tol_T=2.0, tol_t=0.005, max_iter=8, step_down=100.0, T_min_floor=None,
                      verbose=True, **run_kw):
    """Iterate storage.T_min and storage.volume_factor (T_max fixed) until the shades reach
    fully open when the block average reaches T_min (within tol_T [K]) at the target
    duration (within tol_t, fractional). storage.T_min in `cfg` is the starting guess.

    Each pass: run the discharge, find the opening (T_open, t_open), set T_min = T_open and
    scale the block volume by duration / t_open (power is constant while the shades
    regulate). If the shades are still fully closed at T_min, T_min drops by `step_down`.
    Stops early (below_floor=True) if T_min would go below T_min_floor [K].
    run_kw go to run_discharge (dt_scale, dt_at_shade_limit, deposition).

    -> SimpleNamespace(cfg, m, r, history, converged, below_floor, T_min, T_max, swing,
                       volume_factor, t_open_hr, T_open, r_out)"""
    c = copy.deepcopy(cfg)
    s = c['storage']
    s['volume_factor'] = float(s.get('volume_factor', 1.0) or 1.0)
    T_max = float(s['T_min']) + float(s['block_temp_range'])
    t_req = float(s['duration_hr']) * 3600
    hist = []
    m = r = None
    converged = below = False
    for it in range(1, max_iter + 1):
        T_min = float(s['T_min'])
        m, r = run_discharge(c, **run_kw)
        pt = shade_open_point(m, r)
        rec = dict(it=it, T_min=T_min, swing=T_max - T_min, volume_factor=s['volume_factor'], r_out=m.r_out,
                   V_block=m.V_block, t_open_hr=np.nan, T_open=np.nan, seen=None)
        hist.append(rec)
        if pt is None:
            T_next = T_min - step_down
            if verbose:
                print(f"pass {it}: shades still fully closed at T_min = {T_min:.0f} K; lowering T_min {step_down:.0f} K")
        else:
            t_open, T_open = pt['t_open'], pt['T_open']
            rec.update(t_open_hr=t_open / 3600, T_open=T_open, seen=pt['seen'])
            dT, dt_frac = T_open - T_min, t_open / t_req - 1
            if verbose:
                print(f"pass {it}: T_min {T_min:.1f} K, volume x{s['volume_factor']:.4f} (r_out {m.r_out:.4f} m) -> "
                      f"shades open at {T_open:.1f} K ({dT:+.1f}), {t_open/3600:.2f} h ({dt_frac*100:+.2f}%)"
                      f"{'' if pt['seen'] else ' [extrapolated past T_min]'}")
            if abs(dT) <= tol_T and abs(dt_frac) <= tol_t:
                converged = True
                break
            T_next = T_open
        if T_min_floor is not None and T_next < T_min_floor:
            below = True
            break
        if it == max_iter:
            break
        V_target = m.V_block * (t_req / pt['t_open'] if pt is not None else 1.0)
        s['T_min'], s['block_temp_range'] = T_next, T_max - T_next
        s['volume_factor'] = 1.0
        V0 = build_tinless(c, geometry_only=True).V_block
        # shades still closed: keep the volume build gives for the new swing (duration basis)
        s['volume_factor'] = V_target / V0 if pt is not None else 1.0
    last = hist[-1]
    return SimpleNamespace(cfg=c, m=m, r=r, history=hist, converged=converged, below_floor=below,
                           T_min=last['T_min'], T_max=T_max, swing=T_max - last['T_min'],
                           volume_factor=last['volume_factor'], t_open_hr=last['t_open_hr'],
                           T_open=last['T_open'], r_out=m.r_out)
