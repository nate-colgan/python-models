"""Building blocks for the model documentation (thermal/docs.py): a Word document wrapper
with equations (rendered with matplotlib mathtext), flowcharts, review call-outs, and a
YAML reader that keeps the inline comments."""
import io
import re

import numpy as np

from .report import _Doc, WARN, DIM, HEAD_FILL, _fmt

REVIEW = 'B5520A'


class Doc(_Doc):
    """Word document for the model documentation."""

    def __init__(self):
        super().__init__()
        from docx.shared import Pt
        h = self.d.styles['Heading 3']
        h.font.name, h.font.size = 'Calibri', Pt(10.5)
        h.paragraph_format.space_before, h.paragraph_format.space_after = Pt(8), Pt(3)
        h.paragraph_format.keep_with_next = True
        self.reviews = []            # (section, text) collected for the review section
        self.section = ''

    def h1(self, text):
        self.section = text
        super().h1(text)

    def h2(self, text):
        self.section = text
        super().h2(text)

    def h3(self, text):
        self.d.add_heading(text, level=3)

    def p(self, *parts):
        """Paragraph from text parts; a part may be (text, {'bold', 'italic', 'mono', 'color'})."""
        from docx.shared import RGBColor
        para = self.d.add_paragraph()
        for part in parts:
            txt, fmt = part if isinstance(part, tuple) else (part, {})
            r = para.add_run(txt)
            r.bold = bool(fmt.get('bold'))
            r.italic = bool(fmt.get('italic'))
            if fmt.get('mono'):
                r.font.name = 'Consolas'
                r.font.size = self.Pt(9)
            if fmt.get('color'):
                r.font.color.rgb = RGBColor.from_string(fmt['color'])
        return para

    def bl(self, items, level=0):
        style = 'List Bullet' if level == 0 else 'List Bullet 2'
        for it in items:
            if isinstance(it, (list, tuple)) and it and isinstance(it[0], tuple):
                para = self.d.add_paragraph(style=style)
                for txt, fmt in it:
                    r = para.add_run(txt)
                    r.bold = bool(fmt.get('bold'))
                    r.italic = bool(fmt.get('italic'))
            else:
                self.d.add_paragraph(str(it), style=style)

    def nl(self, items):
        for it in items:
            self.d.add_paragraph(str(it), style='List Number')

    def eq(self, tex, size=12, where=None):
        """Display equation (matplotlib mathtext), optional 'where' line below."""
        import matplotlib
        import matplotlib.pyplot as plt
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        fig = plt.figure(figsize=(0.01, 0.01))
        fig.text(0, 0, f"${tex}$", fontsize=size)
        buf = io.BytesIO()
        try:
            with matplotlib.rc_context({'mathtext.fontset': 'cm'}):
                fig.savefig(buf, format='png', dpi=220, bbox_inches='tight', pad_inches=0.03, transparent=False,
                            facecolor='white')
        except Exception as e:                       # unsupported mathtext: show the source instead
            plt.close(fig)
            print(f"[docs] equation not rendered ({type(e).__name__}): {tex[:80]}")
            self.mono(tex)
            if where:
                self.para(where, size=9, color=DIM)
            return
        plt.close(fig)
        from PIL import Image
        im = Image.open(io.BytesIO(buf.getvalue()))
        w_in = min(im.size[0] / 220 * 0.95, self.width - 0.3)
        self.d.add_picture(io.BytesIO(buf.getvalue()), width=self.Inches(w_in))
        self.d.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        if where:
            self.para(where, size=9, color=DIM)

    def review(self, text):
        """Inline 'Review' call-out; also collected for the review section."""
        from docx.shared import RGBColor
        para = self.d.add_paragraph()
        r = para.add_run('Review: ')
        r.bold = True
        r.font.color.rgb = RGBColor.from_string(REVIEW)
        r2 = para.add_run(text)
        r2.italic = True
        para.paragraph_format.left_indent = self.Inches(0.25)
        self.reviews.append((self.section, text))

    def code(self, text):
        self.mono(text)

    def img(self, png, caption=None, width=None, name=None):
        self.d.add_picture(io.BytesIO(png), width=self.Inches(width or self.width))
        if caption:
            p = self.para(caption, size=9, color=DIM, italic=True, bold_prefix=f"{name}: " if name else None)
            p.paragraph_format.space_after = self.Pt(10)

    def fig(self, fig, caption=None, width=None, name=None, dpi=150):
        import matplotlib.pyplot as plt
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        self.img(buf.getvalue(), caption, width, name)

    def toc(self):
        """Word table-of-contents field (Word fills it in: right-click -> Update field)."""
        from docx.oxml import OxmlElement
        from docx.oxml.ns import qn
        para = self.d.add_paragraph()
        r = para.add_run()
        for tag, attr in (('w:fldChar', 'begin'), ('w:instrText', None), ('w:fldChar', 'separate'),
                          ('w:t', None), ('w:fldChar', 'end')):
            el = OxmlElement(tag)
            if tag == 'w:fldChar':
                el.set(qn('w:fldCharType'), attr)
            elif tag == 'w:instrText':
                el.set(qn('xml:space'), 'preserve')
                el.text = 'TOC \\o "1-2" \\h \\z \\u'
            else:
                el.text = 'Right-click here and choose Update Field to build the table of contents.'
            r._r.append(el)


# --------------------------------------------------------------------------
# Flowcharts
# --------------------------------------------------------------------------
_KIND = {
    'proc': dict(boxstyle='round,pad=0.45', fc='#E8EEF7', ec='#2F5F9E'),
    'io': dict(boxstyle='round,pad=0.45', fc='#F3F3F3', ec='#555555'),
    'dec': dict(boxstyle='round4,pad=0.45', fc='#FFF4E0', ec='#B5520A'),
    'loop': dict(boxstyle='round,pad=0.45', fc='#EAF5EA', ec='#2E7D32'),
    'out': dict(boxstyle='round,pad=0.45', fc='#F7E8F2', ec='#8E2C6E'),
}


def flowchart(boxes, arrows, size=(7.0, 6.0), groups=(), fontsize=8.0, title=None):
    """boxes: {id: (x, y, text, kind)} in axes units of size; arrows: (a, b, label, rad) with
    rad the arc (0 straight). groups: (x0, y0, x1, y1, label) dashed frames. -> Figure."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    W, H = size
    fig, ax = plt.subplots(figsize=size)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis('off')
    for x0, y0, x1, y1, lab in groups:
        ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle='round,pad=0.05', fc='none',
                                    ec='#999999', ls='--', lw=0.9))
        ax.text(x0 + 0.08, y1 - 0.08, lab, ha='left', va='top', fontsize=fontsize - 0.5, color='#666666',
                style='italic')
    T = {}
    for k, (x, y, txt, kind) in boxes.items():
        T[k] = ax.text(x, y, txt, ha='center', va='center', fontsize=fontsize, bbox=_KIND.get(kind, _KIND['proc']),
                       wrap=False, zorder=3)
    fig.canvas.draw()
    inv = ax.transData.inverted()
    half = {}                                     # box half-sizes in data units (incl. the bbox pad)
    for k, t in T.items():
        ext = t.get_bbox_patch().get_window_extent()
        (x0, y0), (x1, y1) = inv.transform([(ext.x0, ext.y0), (ext.x1, ext.y1)])
        half[k] = ((x1 - x0) / 2, (y1 - y0) / 2, (x0 + x1) / 2, (y0 + y1) / 2)

    def edge(k, dx, dy, gap=0.04):
        """Point on box k's edge in direction (dx, dy) from its centre (plus a small gap)."""
        hw, hh, cx, cy = half[k]
        n = np.hypot(dx, dy) or 1.0
        ux, uy = dx / n, dy / n
        t = min(hw / abs(ux) if abs(ux) > 1e-12 else np.inf, hh / abs(uy) if abs(uy) > 1e-12 else np.inf)
        return cx + (t + gap) * ux, cy + (t + gap) * uy

    for a in arrows:
        a_, b_, lab = a[0], a[1], a[2] if len(a) > 2 else ''
        rad = a[3] if len(a) > 3 else 0.0
        xa, ya = half[a_][2], half[a_][3]
        xb, yb = half[b_][2], half[b_][3]
        dx, dy = xb - xa, yb - ya
        # leave / arrive along the arc's tangent (arc3: control point offset rad * |d| normal to the chord)
        ox, oy = rad * dy, -rad * dx
        sa = edge(a_, dx / 2 + ox, dy / 2 + oy)
        sb = edge(b_, -dx / 2 + ox, -dy / 2 + oy)
        ax.annotate('', xy=sb, xytext=sa,
                    arrowprops=dict(arrowstyle='-|>', color='#333333', lw=1.0, shrinkA=0, shrinkB=0,
                                    mutation_scale=10, connectionstyle=f'arc3,rad={rad}'), zorder=3.5)
        if lab:
            xm, ym = (xa + xb) / 2, (ya + yb) / 2
            if rad:
                dx, dy = xb - xa, yb - ya
                xm, ym = xm + rad * dy * 0.5, ym - rad * dx * 0.5
            ax.text(xm, ym, lab, ha='center', va='center', fontsize=fontsize - 1, color='#333333',
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.9), zorder=4)
    if title:
        ax.set_title(title, fontsize=fontsize + 2)
    return fig


# --------------------------------------------------------------------------
# YAML with comments
# --------------------------------------------------------------------------
def yaml_with_comments(path):
    """[(dotted key, raw value text, comment)] for every scalar / list leaf of a block-style
    YAML file; comment-only lines indented under a key are appended to its comment."""
    out, stack = [], []
    last = None
    for raw in open(path, encoding='utf-8').read().splitlines():
        if not raw.strip():
            continue
        ind = len(raw) - len(raw.lstrip(' '))
        s = raw.strip()
        if s.startswith('#'):
            if last is not None and ind > last[1]:
                out[last[0]][2] = (out[last[0]][2] + ' ' + s.lstrip('#').strip()).strip()
            continue
        if s.startswith('- '):
            continue
        m = re.match(r'^([A-Za-z0-9_]+):\s*(.*)$', s)
        if not m:
            continue
        key, rest = m.group(1), m.group(2)
        val, com = rest, ''
        if '#' in rest:
            i = rest.index('#')
            val, com = rest[:i].strip(), rest[i + 1:].strip()
        while stack and stack[-1][1] >= ind:
            stack.pop()
        path_ = '.'.join([k for k, _ in stack] + [key])
        if val == '' or val.startswith('{') and False:
            stack.append((key, ind))
            if com:
                out.append([path_, '', com])
                last = (len(out) - 1, ind)
            else:
                last = None
            continue
        out.append([path_, val, com])
        last = (len(out) - 1, ind)
    return [tuple(r) for r in out]
