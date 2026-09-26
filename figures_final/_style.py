"""Shared figure style for the SAKIKO paper. Deterministic; no randomness.

The palette and tone come from the paper's main framework figure: purple for the
intervention and its injection site, green for observation and for arrival at the
required action, blue for the tool-call side and for a comparison arm, orange for
request-information and for a wrong destination, rose for cannot-answer and for a
correct decision broken. Panels follow ggplot2 `theme_bw()` in the manuscript's serif
face, with a centred "(a) Title" heading tinted in the panel's accent colour.

Figures are drawn at print size for the ICLR text width. At most two panels per
figure, and no two panels use the same chart type where a different one carries the
same evidence."""
import csv, colorsys, json, os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, to_hex
from matplotlib.patches import Rectangle
from matplotlib.ticker import AutoMinorLocator, NullLocator

EV_CSV = 'final_evidence/FINAL_PAPER_EVIDENCE.csv'
TEXT_W = 5.5          # ICLR text width in inches
BASE = 8.5            # base font size in points, at print size
OUT_DIR = 'figures'   # what the manuscript's \includegraphics paths point at

# ---- the framework figure's accents ----
PURPLE,  PURPLE_L  = '#7b5ea7', '#e4dcf0'    # intervention · injection site
GREEN,   GREEN_L   = '#4a9e6f', '#d8eee1'    # observation · arrival · a property that holds
BLUE,    BLUE_L    = '#4a7fc1', '#dce9f7'    # tool call · a comparison arm
ORANGE,  ORANGE_L  = '#e8944a', '#fde4d9'    # request information · a wrong destination
ROSE,    ROSE_L    = '#d4646a', '#f7dcdd'    # cannot answer · a decision broken
TAN,     CREAM     = '#c7a95b', '#fdf6e3'    # frozen weights · panel tint
GREY, GREY_L, GREY_XL = '#8c8c8c', '#bfbfbf', '#dcdcdc'
SALMON = ROSE                                 # older name

# ---- semantic names used by the panels ----
ARRIVE = GREEN            # arrival at the required action · a property that holds · ADMIT
OTHER = ORANGE            # moved, but not to the required action
BROKEN = ROSE             # a correct decision broken · a property that fails
REAL = PURPLE             # the real direction, applied at its own site and channel
COMPARE = CYAN = BLUE     # score-space comparator, ungated arm
DECLINE = PURPLE_L        # insufficient evidence, deliberately not a failure colour
PALEBLUE = PURPLE_L       # partial
UNCHANGED, UNEXPOSED, NULLGREY = GREY_L, GREY_XL, GREY

# ---- greys for ink and furniture ----
PANEL, GRID, GRIDM, FRAME = '#fafafa', '#e9e9e9', '#f2f2f2', '#cbcbcb'
GREY92, GREY85, GREY30, GREY20, GREY10 = '#ebebeb', '#d9d9d9', '#4d4d4d', '#333333', '#1a1a1a'
INK = '#1a1a1a'

def dark(c, light=0.34, sat=0.60):
    """Same hue, darker: palette colours used as small text stay legible."""
    h, l, s = colorsys.rgb_to_hls(*to_rgb(c))
    return to_hex(colorsys.hls_to_rgb(h, light, min(s, sat)))

ARRIVE_TXT = dark(GREEN, 0.28, 0.65)
OTHER_TXT  = dark(ORANGE, 0.34, 0.70)
BROKEN_TXT = dark(ROSE, 0.36, 0.60)
REAL_TXT   = dark(PURPLE, 0.34, 0.55)
BLUE_TXT = COMPARE_TXT = CYAN_TXT = dark(BLUE, 0.34, 0.60)
DECLINE_TXT = REAL_TXT
GREEN_TXT = ARRIVE_TXT

# ---- evidence ----
def load():
    """variable -> value (str). The evidence index is the single source of truth."""
    return {r['variable']: r['value'] for r in csv.DictReader(open(EV_CSV))}

def load_rows():
    return {r['variable']: r for r in csv.DictReader(open(EV_CSV))}

def num(ev, k):
    return float(ev[k])

def triple(ev, k):
    """'107/52/+55' -> (107, 52, 55)"""
    f, b, n = ev[k].split('/')
    return int(f), int(b), int(n.replace('+', ''))

def frozen(path):
    """Read a frozen verdict artifact. Read-only; never written."""
    with open(path) as fh:
        return json.load(fh)

# ---- theme ----
def theme_bw(base=BASE):
    """ggplot2::theme_bw(), in the manuscript's serif face."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'STIXGeneral', 'DejaVu Serif'],
        'font.size': base, 'text.color': INK,
        'mathtext.fontset': 'stix',
        'axes.facecolor': PANEL, 'axes.edgecolor': FRAME, 'axes.linewidth': 0.7,
        'axes.spines.left': True, 'axes.spines.right': True,
        'axes.spines.top': True, 'axes.spines.bottom': True,
        'axes.grid': True, 'axes.axisbelow': True,
        'grid.color': GRID, 'grid.linewidth': 0.6,
        'axes.labelsize': base, 'axes.labelcolor': INK, 'axes.labelpad': 3.0,
        'axes.titlesize': base * 1.05, 'axes.titlelocation': 'center', 'axes.titlepad': 6.0,
        'axes.titlecolor': INK,
        'axes.xmargin': 0.05, 'axes.ymargin': 0.05,
        'xtick.color': '#9a9a9a', 'ytick.color': '#9a9a9a',
        'xtick.labelcolor': GREY30, 'ytick.labelcolor': GREY30,
        'xtick.labelsize': 0.85 * base, 'ytick.labelsize': 0.85 * base,
        'xtick.direction': 'out', 'ytick.direction': 'out',
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.minor.size': 0, 'ytick.minor.size': 0,
        'xtick.major.pad': 2.5, 'ytick.major.pad': 2.5,
        'legend.frameon': True, 'legend.facecolor': 'white', 'legend.edgecolor': FRAME,
        'legend.framealpha': 0.92, 'legend.fontsize': 0.85 * base,
        'legend.title_fontsize': 0.85 * base, 'legend.handlelength': 1.1,
        'legend.handleheight': 1.0, 'legend.handletextpad': 0.5,
        'legend.borderaxespad': 0.4, 'legend.borderpad': 0.4, 'legend.columnspacing': 1.2,
        'lines.linewidth': 1.1, 'lines.solid_capstyle': 'round',
        'patch.linewidth': 0,
        'figure.facecolor': 'white', 'savefig.facecolor': 'white',
        'figure.dpi': 300, 'savefig.dpi': 300,
        'savefig.bbox': 'tight', 'savefig.pad_inches': 0.03,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
    })

setup = theme_grey = theme_bw      # older scripts call setup()

def gg(ax, x='continuous', y='continuous', tint=None):
    """Panel furniture: light grid on both axes, minor grid on continuous ones."""
    ax.set_facecolor(tint or PANEL)
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(FRAME); s.set_linewidth(0.7)
    ax.grid(True, which='major', color=GRID, linewidth=0.6)
    for axis, kind in ((ax.xaxis, x), (ax.yaxis, y)):
        if kind == 'continuous':
            axis.set_minor_locator(AutoMinorLocator(2))
        else:
            axis.set_minor_locator(NullLocator())
            if kind == 'none':
                axis.set_major_locator(NullLocator())
    ax.grid(True, which='minor', color=GRIDM, linewidth=0.4)
    ax.tick_params(which='minor', length=0)
    return ax

def panel_title(ax, text, colour=INK, pad=6.0):
    """Centred '(a) Title' heading, tinted like the framework figure's section headings."""
    ax.set_title(text, fontsize=BASE * 1.05, color=colour, loc='center', pad=pad)

def note(ax, text, x=1.0, y=-0.02, ha='right', va='top', size=None, **kw):
    """Small italic provenance note, in axes coordinates."""
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=size or 0.78 * BASE,
            color=GREY30, style='italic', clip_on=False, **kw)

def key(ax, x, y, color, text, size=None, bold=False, tcolor=None, box=5.2,
        transform=None, ec='none'):
    """A legend key with its label, placed in axes coordinates."""
    tr = transform or ax.transAxes
    ax.scatter([x], [y], s=box ** 2, marker='s', color=color, edgecolors=ec,
               linewidths=0.4, transform=tr, clip_on=False, zorder=6)
    ax.annotate(text, xy=(x, y), xycoords=tr, xytext=(box * 0.8 + 1.5, 0),
                textcoords='offset points', ha='left', va='center',
                fontsize=size or 0.85 * BASE, color=tcolor or GREY20,
                fontweight='bold' if bold else 'normal', annotation_clip=False)

def save(fig, name, out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    out = f'{out_dir}/{name}'
    fig.savefig(out + '.pdf', metadata={'CreationDate': None})   # byte-stable output
    fig.savefig(out + '.png')
    print(f'  wrote {out}.pdf / .png')
