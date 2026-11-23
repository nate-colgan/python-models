"""Word (.docx) report of an optimizer run (tinless system optimizer.ipynb).

    FIGS = FigureStore()                  # in the setup cell
    ...; FIGS.keep('pareto', 'caption')   # after each figure, before plt.show()
    write_report(LOG_DIR, dict(opt=..., opt_inputs=..., base_cfg=..., rows=..., best=..., ...))

Writes <run dir>/report.docx (figures embedded), <run dir>/figures/*.png and
<run dir>/report_inputs.json (everything needed to rerun). Needs python-docx.
"""
import datetime as _dt
import io
import json
import os
import subprocess

import numpy as np

INCH = 0.0254
C2K = 273.15


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
class FigureStore:
    """Keeps a PNG of each notebook figure (rendered when kept, so plt.show() closing the
    figure does not matter). Re-keeping a name replaces it, so re-running a cell is fine."""

    def __init__(self):
        self.items = {}

    def keep(self, name, caption='', fig=None, dpi=150):
        import matplotlib.pyplot as plt
        fig = fig or plt.gcf()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
        self.items[name] = dict(caption=caption, png=buf.getvalue())

    def __len__(self):
        return len(self.items)


# --------------------------------------------------------------------------
# Descriptions of the non-optimized inputs (thermal.optimize.OPT_DEFAULTS keys)
# --------------------------------------------------------------------------
DESC = {
    'P_rated': ('MW', 1e-6, 'plant AC export power'),
    'duration_hr': ('h', 1, 'discharge duration at P_rated (energy target, enforced on clean performance)'),
    'charge_hr': ('h', 1, 'design charge time: heater power, then heater cavity, sized to meet it'),
    'economics.eta_out': ('-', 1, 'DC -> AC inverter efficiency'),
    'economics.eta_in': ('-', 1, 'AC -> heater supply efficiency'),
    'economics.cycles_per_year': ('1/yr', 1, 'cycles per year (1-week cycle: charge + hold + discharge)'),
    'economics.lifetime_yr': ('yr', 1, 'plant life'),
    'economics.soft_cost_frac': ('-', 1, 'soft costs (tax, freight, EPC) added to all capex'),
    'economics.elec_price': ('$/MWh', 1, 'one price for charge and discharge: values the energy lost'),
    'objective.rte_min': ('-', 1, 'optional floor on net RTE (0 = none)'),
    'costs.cell_per_cm2': ('$/cm2', 1, 'cells, per cm2 of cell-covered stick face (also each replacement)'),
    'costs.stick_per_m': ('$/m', 1, 'powerstick, per m of length'),
    'costs.stick_fixed': ('$', 1, 'powerstick, fixed per stick'),
    'costs.heater_fixed': ('$', 1, 'heater element, per heater (placeholder)'),
    'costs.charge_electrics_per_kW': ('$/kW', 1, 'heater power supply (placeholder)'),
    'costs.inverter_per_kW': ('$/kW', 1, 'inverters, sized on peak plant AC output'),
    'costs.heat_rejection_per_kWth': ('$/kWth', 1, 'dry coolers, sized on peak stick heat rejection'),
    'costs.bare_clean_per_stick': ('$', 1, 'per stick per bare-area cleaning (count tracked)'),
    'costs.interconnect_per_kW': ('$/kW', 1, 'grid connection on max(peak charge, peak output) AC (placeholder)'),
    'costs.liner_wall_per_m2': ('$/m2', 1, 'liner walls, 4 x side x L'),
    'costs.liner_floor_per_m2': ('$/m2', 1, 'CFC liner floor'),
    'costs.liner_floor_area_factor': ('-', 1, 'CFC area = factor x side^2 (extra covers upper features)'),
    'costs.shade_rgf_per_m3': ('$/m3', 1, 'RGF shade vanes, 8 per stick, full block height (placeholder)'),
    'costs.shade_axle_per_m': ('$/m', 1, 'graphite / CFC axle per vane (placeholder)'),
    'costs.shade_axle_extra': ('m', 1, 'axle length above the top insulation'),
    'costs.shade_drive': ('', 1, "'linkage': 1 motor + linkage per stick, 'per_vane': 1 motor per vane"),
    'costs.shade_motor': ('$', 1, 'per motor incl. driver (placeholder)'),
    'costs.shade_linkage': ('$', 1, 'per stick, linkage drive (placeholder)'),
    'parasitics.heat_rejection_kWe_per_kWth': ('kWe/kWth', 1, 'dry-cooler fans + pumps per kW rejected'),
    'deposition.d_dead_nm': ('nm', 1, 'deposit at which cells are dead (D = deposit / d_dead)'),
    'deposition.emis_end': ('-', 1, 'bare / inactive area emissivity at D = 1'),
    'deposition.cell_replace_D': ('-', 1, 'replace cells at this degradation'),
    'deposition.bare_clean_D': ('-', 1, 'clean bare areas at this degradation'),
    'flux_band_tol': ('-', 1, '|flux / target - 1| within this counts as at target'),
    'sizing.tol_T': ('K', 1, 'shade-open sizing: shades open within this of T_min'),
    'sizing.tol_t': ('-', 1, 'shade-open sizing: and within this fraction of the duration'),
    'sizing.max_iter': ('', 1, 'shade-open sizing passes, max'),
    'sizing.swing_guess': ('K', 1, 'first guess of the swing'),
    'geometry.ps_liner_gap': ('m', 1, 'min powerstick -> liner clearance per side'),
    'geometry.heater_min_gap': ('m', 1, 'min heater -> cavity wall clearance'),
    'geometry.H_max': ('m', 1, 'block height upper bound'),
    'geometry.min_lower_zone': ('m', 1, 'block always this much taller than the stick (lower zone)'),
    'geometry.T_cap_C': ('C', 1, 'max block temperature'),
    'geometry.T_min_floor_C': ('C', 1, 'lowest allowed T_min (constraint)'),
    'geometry.min_wall': ('m', 1, 'min graphite wall between neighbouring cavities (constraint)'),
    'geometry.swing_min': ('C', 1, 'min block temperature swing (constraint)'),
    'geometry.swing_max': ('C', 1, 'max block temperature swing (constraint)'),
    'limits.D_tank_max': ('m', 1, 'max tank diameter (constraint)'),
    'limits.H_tank_max': ('m', 1, 'max tank height (constraint)'),
    'fidelity.dz': ('m', 1, 'block ring height (screening)'),
    'fidelity.nrad': ('', 1, 'radial block segments'),
    'fidelity.dt_scale': ('', 1, 'timestep multiplier (screening)'),
    'fidelity.vane_rays': ('', 1, 'rays per surface, vane view factors'),
    'fidelity.heater_ar_tol': ('', 1, 'heater cavity search tolerance'),
}
COST_GROUPS = ('costs', 'economics')        # shown with the cost results

# optimizer variable -> (design key, unit, scale, format)
VAR_PHYS = {
    'liner_side': ('liner_side', 'mm', 1e3, '.0f'),
    'ps_frac': ('ps_side', 'mm (powerstick side)', 1e3, '.0f'),
    'L': ('L', 'm', 1, '.2f'),
    'H_frac': ('H_block', 'm (block height)', 1, '.2f'),
    'avg_flux': ('avg_flux', 'W/cm2', 1, '.2f'),
    'shade_8th': ('shade_thickness', 'in', 1 / INCH, '.3f'),
    't_side': ('t_side', 'm', 1, '.2f'),
    't_top': ('t_top', 'm', 1, '.2f'),
    'L_bot': ('L_bot', 'm', 1, '.2f'),
    'T_max_C': ('T_max_C', 'C', 1, '.0f'),
}

# YAML keys the optimizer sets for every design (the YAML value is not used)
SET_BY_OPTIMIZER = {
    'end_time', 'storage.L', 'storage.block_height', 'storage.avg_flux', 'storage.duration_hr', 'storage.T_min',
    'storage.block_temp_range', 'storage.num_ax_segments', 'storage.num_rad_segments', 'storage.volume_factor',
    'powerstick.liner_side_len', 'powerstick.side_len', 'shade.thickness', 'tank.t_side', 'tank.t_top', 'tank.L_bot',
    'tank.P_rated', 'tank.duration_hr', 'heater.area_ratio', 'radiation.vane_rays', 'radiation.vf_cache_dir',
}

ASSUMPTIONS = [
    "Each stick and each heater is analysed as an annulus of graphite with an adiabatic outer radius "
    "(one stick + heater pair's graphite per cavity); in the plant they sit on a checkerboard in one "
    "continuous graphite matrix (layout figure). All cells are assumed to run in step.",
    "Block sizing (shade-open): T_max is a design variable; T_min is the volume-average block temperature at "
    "which the shades are fully open, and the block volume is set so that this happens at the target duration.",
    "Charge: heater power is the smallest that charges in charge_hr with the cavity at its minimum gap (the "
    "cavity grows only if the power cap, max_power_per_m x block height, is not enough). Grid interconnect is "
    "sized on max(all heaters at full power, peak output).",
    "Steady weekly cycle: charge, hold at T_max (heaters make up all losses, stick retracted and a door closed; "
    "door loss not modelled), discharge with the stick inserted. No deposition while the stick is retracted.",
    "Degradation: deposition per cycle from the clean discharge; cells replaced and bare areas cleaned at the "
    "set thresholds. Performance is the average over every cycle of the life.",
    "Objective = lifetime capex (initial + replacements + cleanings, all incl. soft costs) + value of the "
    "electricity lost over the life (AC bought - net AC sold, at the electricity price). Capex is reported "
    "without the electricity term.",
    "Not modelled: heat leak through roof penetrations (sticks, heaters, shade axles), the hold-phase door, "
    "edge cells of the grid. Costs marked estimate are placeholders to be replaced.",
]

WARN = 'B5520A'
DIM = '6B7385'
HEAD_FILL = 'DCE3EE'


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _fmt(v, f=None):
    if v is None:
        return '-'
    if isinstance(v, (bool, np.bool_)):
        return 'yes' if v else 'no'
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    if isinstance(v, (float, np.floating)):
        v = float(v)
        if f:
            return format(v, f)
        if not np.isfinite(v):
            return str(v)
        a = abs(v)
        if a == 0:
            return '0'
        if a >= 1e4:
            return f"{v:,.0f}"
        if a >= 100:
            return f"{v:,.1f}"
        if a >= 10:
            return f"{v:.1f}"
        return f"{v:.3g}"
    if isinstance(v, (list, tuple, dict)):
        return json.dumps(v, default=str)
    return str(v)


def _flatten(d, pre=''):
    out = {}
    for k, v in d.items():
        key = f"{pre}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + '.'))
        else:
            out[key] = v
    return out


def _git():
    try:
        rev = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()
        br = subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(['git', 'status', '--porcelain', '--', 'thermal'], capture_output=True,
                               text=True).stdout.strip()
        return f"{br} @ {rev}" + (' (thermal/ has uncommitted changes)' if dirty else '')
    except Exception:
        return 'unknown'


def _M(v):
    return f"${v / 1e6:,.1f}M"


class _Doc:
    """Thin python-docx wrapper: headings, paragraphs, bullets, tables, figures."""

    def __init__(self):
        from docx import Document
        from docx.shared import Pt, Inches
        from docx.enum.section import WD_ORIENT
        self.Pt, self.Inches = Pt, Inches
        d = Document()
        s = d.sections[0]
        s.orientation = WD_ORIENT.PORTRAIT
        s.page_width, s.page_height = Inches(8.5), Inches(11)
        for side in ('left_margin', 'right_margin'):
            setattr(s, side, Inches(0.7))
        s.top_margin = s.bottom_margin = Inches(0.7)
        self.width = 8.5 - 1.4
        st = d.styles['Normal']
        st.font.name, st.font.size = 'Calibri', Pt(10)
        st.paragraph_format.space_after = Pt(4)
        for name, size, before in (('Title', 22, 0), ('Heading 1', 15, 16), ('Heading 2', 12, 10)):
            h = d.styles[name]
            h.font.name, h.font.size = 'Calibri', Pt(size)
            h.paragraph_format.space_before, h.paragraph_format.space_after = Pt(before), Pt(4)
            h.paragraph_format.keep_with_next = True
        self.d = d

    def title(self, text, sub):
        self.d.add_paragraph(text, style='Title')
        self.para(sub, size=9, color=DIM)

    def h1(self, text):
        self.d.add_heading(text, level=1)

    def h2(self, text):
        self.d.add_heading(text, level=2)

    def para(self, text, size=None, color=None, italic=False, bold_prefix=None):
        from docx.shared import RGBColor
        p = self.d.add_paragraph()
        if bold_prefix:
            r = p.add_run(bold_prefix)
            r.bold = True
        r = p.add_run(text)
        if size:
            r.font.size = self.Pt(size)
        if color:
            r.font.color.rgb = RGBColor.from_string(color)
        r.italic = italic
        return p

    def bullets(self, items):
        for it in items:
            self.d.add_paragraph(it, style='List Bullet')

    def mono(self, text):
        for ln in text.splitlines():
            p = self.d.add_paragraph()
            p.paragraph_format.space_after = self.Pt(0)
            r = p.add_run(ln)
            r.font.name, r.font.size = 'Consolas', self.Pt(7.5)

    @staticmethod
    def _shade(cell, fill):
        from docx.oxml import OxmlElement
        from docx.oxml.ns import qn
        tcPr = cell._tc.get_or_add_tcPr()
        shd = OxmlElement('w:shd')
        shd.set(qn('w:val'), 'clear')
        shd.set(qn('w:color'), 'auto')
        shd.set(qn('w:fill'), fill)
        tcPr.append(shd)

    @staticmethod
    def _repeat_header(row):
        from docx.oxml import OxmlElement
        from docx.oxml.ns import qn
        trPr = row._tr.get_or_add_trPr()
        el = OxmlElement('w:tblHeader')
        el.set(qn('w:val'), 'true')
        trPr.append(el)

    def table(self, head, rows, widths=None, size=8.5, header=True):
        """rows: lists of cells; a cell is text or (text, {'bold', 'color', 'italic'})."""
        from docx.shared import RGBColor
        n = len(head) if head else len(rows[0])
        t = self.d.add_table(rows=0, cols=n)
        t.style = 'Table Grid'
        t.autofit = widths is None
        all_rows = ([head] if header and head else []) + list(rows)
        for i, r in enumerate(all_rows):
            cells = t.add_row().cells
            for j, v in enumerate(r):
                txt, fmt = (v if isinstance(v, tuple) else (v, {}))
                c = cells[j]
                c.text = ''
                p = c.paragraphs[0]
                p.paragraph_format.space_after = self.Pt(0)
                run = p.add_run(str(txt))
                run.font.size = self.Pt(size)
                run.bold = bool(fmt.get('bold')) or (header and i == 0)
                run.italic = bool(fmt.get('italic'))
                if fmt.get('color'):
                    run.font.color.rgb = RGBColor.from_string(fmt['color'])
                if header and i == 0:
                    self._shade(c, HEAD_FILL)
                if widths:
                    c.width = self.Inches(widths[j])
        if header and head:
            self._repeat_header(t.rows[0])
        self.d.add_paragraph().paragraph_format.space_after = self.Pt(2)
        return t

    def kv(self, rows, w0=2.6):
        self.table(None, [(k, v) for k, v in rows], widths=[w0, self.width - w0], header=False)

    def figure(self, png, caption, name):
        self.d.add_picture(io.BytesIO(png), width=self.Inches(self.width))
        p = self.para(caption, size=9, color=DIM, italic=True, bold_prefix=f"{name}: ")
        p.paragraph_format.space_after = self.Pt(10)

    def page_break(self):
        from docx.enum.text import WD_BREAK
        self.d.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    def save(self, path):
        self.d.save(path)


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------
def _key_results(doc, b, ctx):
    d = b['design']
    hs = b.get('heater', {})
    cy = ctx.get('cycle')
    rows = [
        ('Net round-trip efficiency (life average)', f"{b['RTE'] * 100:.1f}%"),
        ('Plant', f"{ctx['opt']['P_rated'] / 1e6:.0f} MW x {ctx['opt']['duration_hr']:.0f} h: {b['N_ps']} powersticks "
                  f"+ {b['N_ps']} heaters"),
        ('Block', f"{b['T_min_C']:.0f} - {d['T_max_C']:.0f} C (swing {b['swing']:.0f} C), {d['H_block']:.2f} m tall"),
        ('Powerstick / liner', f"{d['ps_side'] * 1e3:.0f} mm x {d['L']:.2f} m / {d['liner_side'] * 1e3:.0f} mm, "
                               f"flux target {d['avg_flux']:.1f} W/cm2"),
        ('Tank', f"{b['D_tank']:.1f} m dia x {b['H_tank']:.1f} m"),
        ('Heater', f"{hs.get('P_heater', np.nan) / 1e3:,.0f} kW each; grid interconnect "
                   f"{b['interconnect_kW'] / 1e3:.1f} MW"),
    ]
    if cy is not None:
        rows.append(('Full-cycle check (detailed settings, clean cells)', f"net RTE {cy.RTE * 100:.1f}%"))
    for k, p in _rte_parts(b, cy).items():
        rows.append((f"RTE / average cell efficiency ({k})",
                     f"{p['frac'] * 100:.1f}% (RTE {p['RTE'] * 100:.1f}% / cell {p['eta'] * 100:.1f}%)"))
    doc.kv(rows)


def _variables(doc, b):
    from . import optimize as O
    x = b.get('x')
    d = b['design']
    rows = []
    for i, (name, lo, hi, integer, desc) in enumerate(O.DESIGN_VARS):
        key, unit, sc, f = VAR_PHYS.get(name, (name, '', 1, '.3g'))
        xv = x[i] if x is not None else None
        at = ''
        if xv is not None and hi > lo:
            u = (xv - lo) / (hi - lo)
            at = 'at lower bound' if u < 0.02 else 'at upper bound' if u > 0.98 else ''
        rows.append([name, desc, f"{lo:g} - {hi:g}" + (' (integer)' if integer else ''), _fmt(xv),
                     format(d[key] * sc, f) + ' ' + unit if key in d else '-',
                     (at, {'bold': True, 'color': WARN}) if at else ''])
    doc.table(['variable', 'description', 'range searched', 'optimum (search value)', 'optimum (physical)', ''],
              rows, widths=[0.8, 2.3, 0.9, 0.9, 1.3, 0.9])
    h = b.get('heater', {})
    doc.h2('Sizing outputs (not search variables)')
    doc.kv([
        ('T_min (from shade-open sizing)', f"{b['T_min_C']:.0f} C"),
        ('Swing', f"{b['swing']:.0f} C"),
        ('Block volume factor (sizing)', f"{b['volume_factor']:.3f} ({b['sizing_runs']} sizing runs, "
                                         f"{'converged' if b['sizing_converged'] else 'NOT converged'})"),
        ('Heater power (sized for charge_hr)', f"{h.get('P_heater', np.nan) / 1e3:,.0f} kW"),
        ('Heater cavity (sized)', f"{b['heater_cavity_D'] * 1e3:.0f} mm dia (area ratio {b['heater_area_ratio']:.2f})"),
    ])


def _design(doc, b):
    d = b['design']
    doc.h2('Layout')
    doc.kv([
        ('Liner', f"{d['liner_side'] * 1e3:.0f} mm square"),
        ('Powerstick', f"{d['ps_side'] * 1e3:.0f} mm square x {d['L']:.2f} m"),
        ('Stick cavity (equiv. radius) / analysis outer radius', f"{b['r_in']:.3f} m / {b['r_out']:.3f} m"),
        ('Shade', f"{d['shade_thickness'] / INCH:.3f} in thick, octagonal vanes, full block height"),
        ('Block height', f"{d['H_block']:.2f} m"),
        ('Heater cavity', f"{b['heater_cavity_D'] * 1e3:.0f} mm dia"),
        ('Grid pitch (stick to heater)', f"{b['pitch']:.3f} m"),
        ('Graphite walls', ', '.join(f"{k} {v * 1e3:.0f} mm" for k, v in b['webs'].items())),
        ('Sticks / heaters', f"{b['N_ps']} / {b['N_ps']}"),
        ('Hot zone diameter', f"{b['D_hot']:.1f} m"),
        ('Tank', f"{b['D_tank']:.1f} m dia x {b['H_tank']:.1f} m tall"),
        ('Insulation side / top / bottom stack', f"{d['t_side']:.2f} m / {d['t_top']:.2f} m / "
                                                 f"{d.get('L_bot', float('nan')):.2f} m"),
        ('Floor (design-day air)', f"max {b.get('T_floor_max_C', float('nan')):.0f} C, column hot spot "
                                   f"{b.get('T_floor_col_C', float('nan')):.0f} C at {b.get('v_air', float('nan')):.0f} m/s; "
                                   f"foundation {b.get('foundation_kPa', float('nan')):.0f} kPa"),
        ('Inverters / interconnect', f"{b['inverter_kW'] / 1e3:.1f} MW / {b['interconnect_kW'] / 1e3:.1f} MW "
                                     f"(charging draw {b['P_charge_ac_kW'] / 1e3:.1f} MW)"),
    ])
    doc.h2('Performance (life average)')
    doc.kv([
        ('Net round-trip efficiency', f"{b['RTE'] * 100:.2f}% (before parasitic loads {b['RTE_gross'] * 100:.2f}%)"),
        ('AC bought per cycle', f"{b['E_in_MWh']:,.0f} MWh (of which hold losses {b['E_hold_MWh']:,.0f} MWh)"),
        ('AC sold per cycle', f"{b['E_out_MWh']:,.0f} MWh gross, {b['E_net_MWh']:,.0f} MWh net of parasitic loads"),
        ('Fans per cycle', f"floor cooling {b['E_floor_fan_MWh']:.1f} MWh, heat rejection {b['E_rej_fan_MWh']:.1f} MWh"),
        ('Cycle', f"charge {b['t_charge_hr']:.1f} h, hold {b['t_hold_hr']:.1f} h, discharge {b['t_discharge_hr']:.1f} h "
                  f"clean / {b['t_discharge_deg_hr']:.1f} h at 2x life-mean wear"),
        ('Energy vs target (clean)', f"{b['energy_mismatch'] * 100:+.1f}%"),
        ('Insulation losses', f"{b['loss_pct_day']:.2f} %/day of stored heat"),
        ('Per stick per discharge', f"{b['E_stick_fresh_kWh']:,.0f} kWh clean, {b['E_stick_deg_kWh']:,.0f} kWh at 2x "
                                    f"life-mean wear; peak {b['P_stick_fresh_kW']:.0f} kW"),
        ('Peak heat rejection per stick', f"{b['Q_rej_peak_kW']:.0f} kWth"),
    ])
    doc.h2('Deposition and maintenance')
    doc.kv([
        ('Deposition', f"{b['dep_nm_per_cycle'] * 1e3:.3f} pm per cycle"),
        ('Cell replacements over the life', f"{b['n_rep']} (interval {b['rep_interval_yr']:.1f} yr)"),
        ('Bare-area cleanings over the life', f"{b['n_clean']} (interval {b['clean_interval_yr']:.1f} yr)"),
        ('Wear state of the averaging run', f"D_cell {b['D_cell_end']:.4f}, D_bare {b['D_bare_end']:.4f} "
                                            f"(2x the life-mean)"),
    ])
    lab = {'flux_target': 'target [W/cm2]', 'frac_above': 'time above target band', 'frac_band': 'time in target band',
           'frac_below': 'time below target band', 'flux_std': 'flux std [W/cm2]',
           'flux_rms': 'rms about target [W/cm2]', 'flux_max': 'max flux [W/cm2]', 'flux_end': 'flux at end [W/cm2]'}
    ff = b.get('flux_fresh') or {}
    doc.h2('Cell flux, clean discharge')
    doc.kv([(lab[k], f"{ff[k] * 100:.0f}%" if k.startswith('frac') else _fmt(ff[k])) for k in lab if k in ff])
    if b.get('tank_warnings'):
        doc.h2('Model notes for this design')
        doc.bullets(b['tank_warnings'])


def _cycle(doc, cy):
    A = cy.AC_in
    J = 1 / 3.6e9
    s = cy.cfg['storage']
    doc.kv([
        ('Net round-trip efficiency (clean cells)', f"{cy.RTE * 100:.1f}% ({cy.RTE_gross * 100:.1f}% before parasitic loads)"),
        ('AC in / net out per cycle', f"{A * J:,.0f} / {cy.net * J:,.0f} MWh"),
        ('Cycle', f"{cy.cycle_hr:.0f} h = charge {cy.t_charge_hr:.1f} + hold {cy.t_hold_hr:.1f} + "
                  f"discharge {cy.t_discharge_hr:.1f} h"),
        ('Block', f"{cy.T_min_C:.0f} - {cy.T_max_C:.0f} C, r {cy.m.r_in:.3f} - {cy.m.r_out:.3f} m x "
                  f"{cy.m.H_block:.2f} m (volume factor {float(s.get('volume_factor', 1) or 1):.3f})"),
        ('Heater', f"{cy.hm.heater.max_power / 1e3:,.0f} kW, cavity {2 * cy.hm.r_in * 1e3:.0f} mm"),
        ('Sticks', f"{cy.N}"),
    ])
    rows, last = [], None
    for g, lab, v in cy.ledger:
        bold = {'bold': True} if g == 'output' else {}
        rows.append([g if g != last else '', (lab, bold), (f"{v * J:,.1f}", bold), f"{v / cy.N / 3.6e6:,.0f}",
                     (f"{v / A * 100:.1f}%", bold)])
        last = g
    doc.table(['group', 'where the electricity goes', 'MWh / cycle', 'kWh / stick', '% of AC in'], rows,
              widths=[1.5, 2.6, 1.0, 1.0, 1.0])


def _png(fig, out_dir, fname):
    """Render a report-made figure to PNG (also saved to <run dir>/figures) and close it."""
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    png = buf.getvalue()
    fdir = os.path.join(out_dir, 'figures')
    os.makedirs(fdir, exist_ok=True)
    with open(os.path.join(fdir, f"{fname}.png"), 'wb') as fh:
        fh.write(png)
    return png


def _rte_parts(b, cy):
    """RTE = delivery x eta_cell x out, with eta_cell = DC / absorbed by the active cells,
    delivery = absorbed / AC bought, out = net AC / DC (inverter + parasitic loads); for the life
    average (optimizer record, if it has the cell energies) and the clean full cycle.
    -> {label: dict}"""
    out = {}
    if b is not None and 'E_cell_abs_MWh' in b and np.isfinite(b.get('eta_cell', np.nan)):
        out['life average (optimizer, screening)'] = dict(
            RTE=b['RTE'], eta=b['eta_cell'], frac=b['RTE_frac_cell'],
            deliv=b['E_cell_abs_MWh'] / b['E_in_MWh'], out=b['E_net_MWh'] / b['E_dc_MWh'])
    if cy is not None and hasattr(cy, 'E_cell_abs'):
        out['clean cells, full cycle (detailed)'] = dict(
            RTE=cy.RTE, eta=cy.eta_cell, frac=cy.RTE_frac_cell,
            deliv=cy.E_cell_abs / cy.AC_in, out=cy.net / cy.DC)
    return out


def _liner_trajectory(cy):
    """Stick-in discharge steps of the full cycle: axial-mean liner inner-face temperature
    [C], instantaneous cell efficiency P_elec / Q_cell, DC energy per step [J]."""
    r = cy.r_dis
    on = np.asarray(r['ps_active'][1:], bool)
    dt = np.diff(np.asarray(r['time'], float))
    Q = np.asarray(r['Q_cell'][1:], float)
    P = np.asarray(r['P_elec'][1:], float)
    T = np.asarray(r['T_liner'][1:], float)[: len(Q)] - C2K
    ok = on & (Q > 0)
    return T[ok], P[ok] / Q[ok], (P * dt)[ok], np.asarray(r['time'][1:], float)[ok] / 3600


def _cell_efficiency(doc, b, cy, rows, cfg, out_dir):
    import matplotlib.pyplot as plt
    from .properties import panel_efficiency, _PANEL_EFF_COEFFS
    doc.para("Average cell efficiency here = DC out / heat absorbed by the active cells, over the discharge. "
             "RTE / cell efficiency is the fraction of the cells' own efficiency that the plant keeps; it splits "
             "exactly into RTE = delivery x cell efficiency x output, with delivery = heat absorbed by the active "
             "cells / AC bought (heater supply, insulation losses, absorption on bare and inactive area) and "
             "output = net AC / DC (inverter and parasitic loads: floor-cooling and heat-rejection fans, pumps).")
    parts = _rte_parts(b, cy)
    if not parts:
        doc.para("The optimizer record has no cell energies (run before they were logged) and no full cycle was "
                 "run, so this section is empty.", color=WARN)
        return
    cols = list(parts)
    lab = [('Net round-trip efficiency', 'RTE', '%'), ('Average cell efficiency, DC / absorbed', 'eta', '%'),
           ('RTE / cell efficiency', 'frac', '%'),
           ('  delivery: absorbed by active cells / AC bought', 'deliv', '%'),
           ('  output: net AC / DC (inverter + parasitic loads)', 'out', '%')]
    doc.table(['quantity'] + cols, [[(n, {'bold': True}) if k == 'frac' else n]
                                    + [(f"{parts[c][k] * 100:.1f}%", {'bold': True} if k == 'frac' else {})
                                       for c in cols] for n, k, _ in lab],
              widths=[2.8] + [(doc.width - 2.8) / len(cols)] * len(cols))
    if b is not None and 'E_cell_abs_MWh' not in b:
        doc.para("The optimizer record of this run has no cell energies (logged from 2026-11-23 on), so only the "
                 "full-cycle column is shown and the population statistics below are not available.",
                 size=9, color=DIM)

    pop = [r for r in rows if r.get('feasible') and 'E_cell_abs_MWh' in r and np.isfinite(r.get('eta_cell', np.nan))]
    if pop:
        top = sorted(pop, key=lambda r: r['lifetime_cost'])[:10]

        def st(rs, k):
            v = np.array([r[k] for r in rs]) * 100
            return f"{v.min():.1f} / {np.median(v):.1f} / {v.max():.1f}%"
        doc.h2('Across the run (feasible designs, life average)')
        doc.table(['quantity', f"all feasible (n = {len(pop):,}): min / median / max",
                   f"10 best: min / median / max"],
                  [[n, st(pop, k), st(top, k)] for n, k in
                   (('RTE / cell efficiency', 'RTE_frac_cell'), ('Average cell efficiency', 'eta_cell'),
                    ('Net RTE', 'RTE'))], widths=[2.0, 2.6, doc.width - 4.6])

    # figure 1: decomposition of the optimum + the run's designs on RTE vs cell efficiency
    key = cols[-1]
    p = parts[key]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw=dict(width_ratios=[1.1, 1]))
    a = ax[0]
    steps = [('AC bought', 1.0), ('absorbed by active cells', p['deliv']), ('DC out', p['deliv'] * p['eta']),
             ('net AC out (= RTE)', p['RTE'])]
    y = np.arange(len(steps))[::-1]
    a.barh(y, [v * 100 for _, v in steps], color=['#9aa5b1', '#5b8fc7', '#2f6aa8', '#1f3f8f'], height=0.6)
    for yi, (n, v) in zip(y, steps):
        a.text(v * 100 + 1, yi, f"{v * 100:.1f}%", va='center', fontsize=9)
    for (yi, (_, v0)), (_, v1), f in zip(zip(y, steps), steps[1:], ('delivery', 'cell efficiency', 'inverter + parasitic loads')):
        a.annotate(f"x {v1 / v0:.3f} ({f})", xy=(102, yi - 0.5), fontsize=8, color='#555555', va='center')
    a.set_yticks(y, [n for n, _ in steps])
    a.set_xlim(0, 135)
    a.set_xlabel('% of AC bought')
    a.set_title(f"Optimum ({key.split(' (')[0]}): RTE = {p['frac'] * 100:.1f}% of cell efficiency", fontsize=10)
    a.grid(alpha=.3, axis='x')
    a = ax[1]
    if pop:
        e = np.array([r['eta_cell'] for r in pop]) * 100
        R = np.array([r['RTE'] for r in pop]) * 100
        c = np.array([r['lifetime_cost'] for r in pop]) / 1e6
        sc = a.scatter(e, R, c=c, s=10, cmap='viridis_r', alpha=.7, lw=0)
        fig.colorbar(sc, ax=a, label='lifetime cost [$M]')
    pts = [(parts[k]['eta'] * 100, parts[k]['RTE'] * 100, k) for k in cols]
    for (xe, yr, k), mk in zip(pts, ('*', 'D')):
        a.plot(xe, yr, mk, ms=12 if mk == '*' else 7, mfc='#d43c2c' if mk == '*' else 'white', mec='#d43c2c',
               mew=1.5, ls='none', label=f"optimum: {k}")
    lo_x = min([x for x, _, _ in pts] + ([e.min()] if pop else [])) - 2
    hi_x = max([x for x, _, _ in pts] + ([e.max()] if pop else [])) + 2
    lo_y = min([y for _, y, _ in pts] + ([R.min()] if pop else [])) - 2
    hi_y = max([y for _, y, _ in pts] + ([R.max()] if pop else [])) + 2
    xs = np.array([lo_x, hi_x])
    for f in np.arange(0.50, 1.0, 0.05):
        if f * hi_x < lo_y or f * lo_x > hi_y:
            continue                                    # line not in view
        a.plot(xs, f * xs, ls=':', c='#888888', lw=1)
        xl, yl = (hi_x, f * hi_x) if f * hi_x <= hi_y else (hi_y / f, hi_y)    # where it leaves the box
        a.text(xl, yl, f"{f:.0%} ", fontsize=8, color='#666666', ha='right', va='top' if yl == hi_y else 'bottom')
    a.set_xlim(lo_x, hi_x)
    a.set_ylim(lo_y, hi_y)
    a.set_xlabel('average cell efficiency, DC / absorbed [%]')
    a.set_ylabel('net RTE [%]')
    a.set_title('Net RTE vs cell efficiency' + (' (feasible, life average)' if pop else ''), fontsize=10)
    a.legend(fontsize=8, loc='lower right')
    a.grid(alpha=.3)
    fig.tight_layout()
    doc.figure(_png(fig, out_dir, 'rte_vs_cell_efficiency'),
               "Left: the optimum's round trip as a chain, RTE = delivery x cell efficiency x output (inverter + parasitic loads). Right: net RTE "
               "against average cell efficiency; dotted lines = constant RTE / cell efficiency.", 'rte_vs_cell_efficiency')

    # figure 2: cell efficiency vs liner temperature (the fit the model uses)
    bg = float(cfg['emissivity_tables'][0]['value'][0][0][1])
    _ce = _PANEL_EFF_COEFFS[bg]
    flux_t = float((b or {}).get('design', {}).get('avg_flux', cfg['storage']['avg_flux']))
    T = np.array([])
    if cy is not None:
        T, eff, E, th = _liner_trajectory(cy)
    fig, axs = plt.subplots(1, 2 if len(T) else 1, figsize=(11, 4.4) if len(T) else (7, 4.4), squeeze=False)
    rows_t = []
    for i, a in enumerate(axs[0]):
        zoom = i == 1
        if zoom:
            pad = max(25.0, 0.3 * (T.max() - T.min()))
            TC = np.linspace(T.min() - pad, T.max() + pad, 200)
        else:
            TC = np.linspace(1000, 2600, 400)
        a.plot(TC, panel_efficiency(bg, TC + C2K) * 100, c='#555555', lw=2, label='fit, no flux derate')
        for fl, ls in ((flux_t, '-'), (2 * flux_t, '--')):
            a.plot(TC, panel_efficiency(bg, TC + C2K, flux=fl) * 100, c='#2f6aa8', lw=2, ls=ls,
                   label=f"fit at {fl:.1f} W/cm2" + (' (flux target)' if fl == flux_t else ''))
        if len(T):
            a.axvspan(T.min(), T.max(), color='#f2c14e', alpha=.2, lw=0, label='liner range, optimum discharge')
        if zoom:
            sc = a.scatter(T, eff * 100, c=th, s=14, cmap='plasma', zorder=3, lw=0,
                           label='optimum, clean discharge: step values')
            fig.colorbar(sc, ax=a, label='hours into discharge')
            a.set_title('Zoom on the operating range', fontsize=10)
        else:
            a.set_title(f"Fit used in the model (band gap {bg:g} um)", fontsize=10)
            a.legend(fontsize=8, loc='lower center')
        a.set_xlabel('liner inner-face (emitter) temperature [C]')
        a.set_ylabel('cell efficiency, DC / absorbed [%]')
        a.grid(alpha=.3)
    if len(T):
        Tm = float((T * E).sum() / E.sum())
        rows_t = [('Liner inner face during the discharge (axial mean)', f"{T.min():.0f} - {T.max():.0f} C, "
                                                                        f"output-weighted mean {Tm:.0f} C"),
                  ('Cell efficiency at the steps (DC / absorbed)', f"{eff.min() * 100:.1f} - {eff.max() * 100:.1f}%"),
                  ('Peak of the fit (no derate)', f"{-_ce[1] / (2 * _ce[0]) - C2K:.0f} C, "
                                                  f"{panel_efficiency(bg, -_ce[1] / (2 * _ce[0])) * 100:.1f}%")]
    fig.tight_layout()
    doc.h2('Cell efficiency vs liner temperature')
    doc.para(f"The model's cell efficiency is a quadratic fit in the emitter temperature (the liner inner face; "
             f"thermal.properties.panel_efficiency, band gap {bg:g} um), derated for flux: "
             f"eta = eta_0(T) x 0.998^(2 x flux x (1 - eta_0)), then x (1 - D_cell) for deposition wear. "
             f"The points are the optimum's clean discharge; they sit a little off the target-flux curve because "
             f"the flux and the liner temperature vary along the stick and over time.")
    if rows_t:
        doc.kv(rows_t)
    doc.figure(_png(fig, out_dir, 'cell_efficiency_vs_liner_T'),
               "Cell efficiency vs liner inner-face temperature: the fit used in this study (no derate, at the "
               "optimum's flux target and twice it) and the optimum's discharge steps.", 'cell_efficiency_vs_liner_T')


def _verified(doc, ver):
    rows = []
    for i, (r, rv) in enumerate(ver):
        rows.append([i + 1, f"{r['RTE'] * 100:.1f}%", f"{rv.get('RTE', np.nan) * 100:.1f}%", r['N_ps'],
                     rv.get('N_ps', '-'), f"{r['T_min_C']:.0f}", f"{rv.get('T_min_C', np.nan):.0f}",
                     'yes' if rv.get('feasible') else f"no {rv.get('violations')}"])
    doc.table(['rank', 'RTE (screening)', 'RTE (detailed)', 'sticks (screening)', 'sticks (detailed)',
               'T_min C (screening)', 'T_min C (detailed)', 'feasible (detailed)'], rows)


def _top(doc, best, n=10):
    rows = []
    for i, r in enumerate(best[:n]):
        d = r['design']
        rows.append([i + 1, f"{r['RTE'] * 100:.1f}", r['N_ps'], f"{d['liner_side'] * 1e3:.0f}",
                     f"{d['ps_side'] * 1e3:.0f}", f"{d['L']:.2f}", f"{d['H_block']:.2f}", f"{d['avg_flux']:.1f}",
                     f"{d['shade_thickness'] / INCH:.3f}", f"{d['t_side']:.2f} / {d['t_top']:.2f}",
                     f"{d['T_max_C']:.0f}", f"{r['T_min_C']:.0f}"])
    doc.table(['rank', 'RTE %', 'sticks', 'liner mm', 'stick mm', 'L m', 'block H m', 'flux W/cm2', 'shade in',
               'insul. side / top m', 'T_max C', 'T_min C'], rows, size=8)


def _costs(doc, b, opt, opt_inputs, verified):
    soft = 1 + opt['economics']['soft_cost_frac']
    doc.h2('Cost results for the optimum')
    doc.kv([
        ('Objective: lifetime capex + value of energy lost', _M(b['lifetime_cost'])),
        ('Lifetime capex (initial + replacements + cleanings)', _M(b['capex_total'])),
        ('Initial capex', _M(b['capex0'])),
        ('Value of the energy lost over the life', f"{_M(b['loss_cost'])} at "
                                                     f"${opt['economics']['elec_price']:.0f}/MWh"),
        ('Cell replacement', f"{b['n_rep']} x {_M(b['rep_cost'])}"),
        ('Bare-area cleaning', f"{b['n_clean']} x {_M(b['clean_cost'])}"),
    ], w0=3.2)
    cl = sorted(b['capex_lines'].items(), key=lambda kv: -kv[1])
    tot = sum(v for _, v in cl)
    rows = [[k, _M(v * soft), f"{v / tot * 100:.1f}%"] for k, v in cl]
    rows.append([('initial capex', {'bold': True}), (_M(b['capex0']), {'bold': True}), ''])
    doc.h2('Initial capex breakdown (incl. soft costs)')
    doc.table(['item', 'cost', 'share'], rows, widths=[4.0, 1.4, 1.0])
    if verified:
        doc.h2('Detailed-settings check of the top designs: costs')
        doc.table(['rank', 'objective (screening)', 'objective (detailed)', 'capex (screening)', 'capex (detailed)'],
                  [[i + 1, _M(r['lifetime_cost']), _M(rv.get('lifetime_cost', np.nan)), _M(r['capex_total']),
                    _M(rv.get('capex_total', np.nan))] for i, (r, rv) in enumerate(verified)])
    doc.h2('Cost and economic assumptions')
    _input_tables(doc, opt, opt_inputs, include=COST_GROUPS)


def _input_tables(doc, opt, opt_inputs, include=None, exclude=()):
    flat = _flatten(opt)
    given = set(_flatten(opt_inputs or {}))
    groups = {}
    for k, v in flat.items():
        g = k.split('.')[0] if '.' in k else 'plant'
        if (include and g not in include) or g in exclude:
            continue
        groups.setdefault(g, []).append((k, v))
    for g, items in groups.items():
        rows = []
        for k, v in items:
            unit, sc, desc = DESC.get(k, ('', 1, ''))
            vv = v * sc if isinstance(v, (int, float)) and not isinstance(v, bool) and sc != 1 else v
            est = 'placeholder' in desc.lower()
            rows.append([k.split('.', 1)[-1] if '.' in k else k, _fmt(vv), unit,
                         (desc.replace(' (placeholder)', '') + (' [estimate]' if est else ''),
                          {'color': WARN} if est else {}),
                         '' if k in given else ('default', {'color': DIM, 'italic': True})])
        doc.para(g, size=10).runs[0].bold = True
        doc.table(['input', 'value', 'unit', 'description', ''], rows, widths=[1.7, 0.8, 0.7, 3.3, 0.6])


DESIGN_KEYS = {'storage.L', 'storage.block_height', 'storage.avg_flux', 'powerstick.liner_side_len',
               'powerstick.side_len', 'shade.thickness', 'tank.t_side', 'tank.t_top', 'tank.L_bot'}
SIZED_KEYS = {'storage.T_min', 'storage.block_temp_range', 'storage.volume_factor', 'heater.area_ratio',
              'heater.max_power'}


def as_run_cfg(base_cfg, best, opt):
    """Model config of the optimum as the optimizer ran it (optimize.design_cfg with the
    screening fidelity, plus the sizing outputs: T_min / swing / volume factor from the
    shade-open sizing, heater cavity and power from the heater search) and the source of
    every value. -> (cfg, {dotted key: source})"""
    from . import optimize as O
    d = dict(best['design'], T_min_guess_C=best['T_min_C'])
    cfg = O.design_cfg(base_cfg, d, opt)
    s, h = cfg['storage'], cfg['heater']
    s['T_min'] = best['T_min_C'] + C2K
    s['block_temp_range'] = best['swing']
    s['volume_factor'] = best['volume_factor']
    h['area_ratio'] = best['heater_area_ratio']
    P = (best.get('heater') or {}).get('P_heater')
    if P is not None:
        h['max_power'] = P
    base_flat = _flatten(base_cfg)
    src = {}
    for k, v in _flatten(cfg).items():
        if k in DESIGN_KEYS and not (k == 'tank.L_bot' and 'L_bot' not in best['design']):
            src[k] = 'design variable'
        elif k in SIZED_KEYS:
            src[k] = 'sizing output'
        elif k not in base_flat or base_flat[k] != v:
            src[k] = 'optimizer input'
        else:
            src[k] = 'YAML'
    return cfg, src


def _model_settings(doc, cfg, src=None):
    """Switches, emissivity table and every model setting; with src (as_run_cfg) the values
    are the optimum's as run, labelled by source."""
    src = src or {}
    flat = _flatten({k: v for k, v in cfg.items() if k != 'emissivity_tables'})
    lab = lambda k: (src.get(k, 'YAML'), {'color': DIM, 'italic': True}) if src else ''
    doc.h2('Switches')
    doc.table(['option', 'value', 'source'] if src else ['option', 'value'],
              [[k, (_fmt(v), {'bold': True})] + ([lab(k)] if src else []) for k, v in flat.items()
               if isinstance(v, bool)], widths=[3.5, 1.0, 1.6] if src else [3.5, 1.0])
    try:
        tab = cfg['emissivity_tables'][0]['value']
        doc.h2('Stick emissivity table')
        doc.table(['wavelength band [um]', 'active cells', 'bare', 'inactive'],
                  [[f"{r[0][0]:g} - {r[0][1]:g}", r[1], r[2], r[3]] for r in tab], widths=[1.8, 1.2, 1.2, 1.2])
    except Exception:
        pass
    if src:
        doc.h2('All model settings of the optimum, as run')
        doc.para("Sources: YAML = tinless_model_input.yaml; optimizer input = replaced by the optimizer for every "
                 "design (targets, screening mesh / rays / timestep limits); design variable = the optimum's value; "
                 "sizing output = from the shade-open sizing or the heater search. Temperatures in K (model) and C "
                 "(tank). heater.max_power is the searched heater power (the YAML gives max_power_per_m as the cap).",
                 size=9, color=DIM)
        doc.table(['key', 'value', 'source'],
                  [[k, _fmt(v), lab(k)] for k, v in flat.items() if not isinstance(v, bool)],
                  widths=[3.0, 2.3, 1.8], size=8)
    else:
        doc.h2('All model (YAML) settings')
        doc.table(['YAML key', 'value', ''],
                  [[k, _fmt(v), ('replaced per design by the optimizer', {'color': DIM, 'italic': True})
                    if k in SET_BY_OPTIMIZER else ''] for k, v in flat.items() if not isinstance(v, bool)],
                  widths=[3.0, 2.0, 2.1], size=8)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def write_report(out_dir, ctx):
    """ctx keys: opt, opt_inputs, base_cfg, rows, best, figs (FigureStore), run (dict of DE
    settings), de (scipy result or None), verified [(screening row, detailed res)], cycle
    (thermal.cycle result or None), yaml_path, notebook. -> path of report.docx"""
    from . import optimize as O
    os.makedirs(out_dir, exist_ok=True)
    opt, rows, best = ctx['opt'], ctx.get('rows') or [], ctx.get('best') or []
    figs = ctx.get('figs')
    run = ctx.get('run') or {}
    de = ctx.get('de')
    name = os.path.basename(os.path.normpath(out_dir))
    now = _dt.datetime.now().strftime('%Y-%m-%d %H:%M')
    n_ok = sum(1 for r in rows if r.get('feasible'))
    n_fail = sum(1 for r in rows if 'error' in r)
    cpu_h = sum(float(r.get('wall_s', 0) or 0) for r in rows) / 3600

    doc = _Doc()
    doc.title(f"Tinless system optimizer: run {name}",
              f"Report generated {now}. Model YAML {ctx.get('yaml_path', '')}; notebook {ctx.get('notebook', '')}; "
              f"code {_git()}.")
    sec = iter(range(1, 20))
    if best:
        doc.h1('Key results')
        _key_results(doc, best[0], ctx)
    else:
        doc.para('No feasible design in this run.', color=WARN)

    doc.h1(f"{next(sec)}. What was optimized")
    doc.para(f"A {opt['P_rated'] / 1e6:.0f} MW, {opt['duration_hr']:.0f} h thermal battery: a graphite block heated by "
             f"resistive heaters and discharged through thermophotovoltaic powersticks behind rotating graphite "
             f"shades, in an insulated steel tank. The optimizer picks the design with the lowest lifetime capex plus "
             f"value of the electricity lost, which pushes toward high round-trip efficiency.")
    doc.h2('Method and assumptions')
    doc.bullets(ASSUMPTIONS)
    doc.h2('Run')
    doc.kv([
        ('Search', 'scipy differential evolution (Sobol initialisation, deferred updating, integer shade thickness)'),
        ('Settings', ', '.join(f"{k} {v}" for k, v in run.items())),
        ('Designs evaluated', f"{len(rows):,} ({n_ok:,} feasible, {n_fail} failed); {cpu_h:,.1f} CPU-hours"),
        ('DE result', f"{getattr(de, 'nit', '-')} generations, {getattr(de, 'nfev', '-')} evaluations: "
                      f"{getattr(de, 'message', '-')}" if de is not None else 'not run in this session (log reloaded)'),
    ])

    if best:
        b = best[0]
        doc.h1(f"{next(sec)}. Optimized variables: ranges and optimum")
        _variables(doc, b)
        doc.h1(f"{next(sec)}. Optimal design")
        _design(doc, b)
        cy = ctx.get('cycle')
        if cy is not None:
            doc.h1(f"{next(sec)}. Full charge - hold - discharge cycle of the optimum")
            doc.para("Re-run of the optimal design at detailed settings (finer rings, more rays, full timestep), "
                     "sized the same way, clean cells. The optimizer's efficiency is the life average (clean and "
                     "worn cells), so it is a little lower.")
            _cycle(doc, cy)
        doc.h1(f"{next(sec)}. Round-trip efficiency vs cell efficiency")
        _cell_efficiency(doc, b, cy, rows, cy.cfg if cy is not None else ctx['base_cfg'], out_dir)
        if ctx.get('verified'):
            doc.h1(f"{next(sec)}. Detailed-settings check of the top designs")
            _verified(doc, ctx['verified'])
        doc.h1(f"{next(sec)}. Best designs of the run")
        _top(doc, best)

    if figs is not None and len(figs):
        doc.h1(f"{next(sec)}. Figures")
        fdir = os.path.join(out_dir, 'figures')
        os.makedirs(fdir, exist_ok=True)
        for fname, it in figs.items.items():
            with open(os.path.join(fdir, f"{fname}.png"), 'wb') as fh:
                fh.write(it['png'])
            doc.figure(it['png'], it['caption'], fname)

    if best:
        doc.h1(f"{next(sec)}. Costs")
        _costs(doc, best[0], opt, ctx.get('opt_inputs'), ctx.get('verified'))
    doc.h1(f"{next(sec)}. Other inputs (not optimized)")
    doc.para("'default' = value from thermal/optimize.py OPT_DEFAULTS, not set in the notebook's OPT_INPUTS.",
             size=9, color=DIM)
    _input_tables(doc, opt, ctx.get('opt_inputs'), exclude=COST_GROUPS)
    if best:
        doc.h1(f"{next(sec)}. Model settings of the optimum (as run)")
        _model_settings(doc, *as_run_cfg(ctx['base_cfg'], best[0], opt))
    else:
        doc.h1(f"{next(sec)}. Model settings (YAML)")
        _model_settings(doc, ctx['base_cfg'])
    if best:
        doc.h1('Appendix: optimum, full text summary')
        doc.mono(O.summary(best[0]))

    path = os.path.join(out_dir, 'report.docx')
    try:
        doc.save(path)
    except PermissionError:                     # report.docx open in Word: save beside it
        path = os.path.join(out_dir, f"report_{_dt.datetime.now():%Y%m%d_%H%M%S}.docx")
        doc.save(path)
        print(f"report.docx is open (locked by Word?); saved as {os.path.basename(path)} instead")
    with open(os.path.join(out_dir, 'report_inputs.json'), 'w', encoding='utf-8') as fh:
        json.dump(dict(generated=now, code=_git(), run=run, opt=opt, opt_inputs=ctx.get('opt_inputs'),
                       base_yaml=ctx['base_cfg'], best=best[0] if best else None), fh, indent=1, default=str)
    return path
