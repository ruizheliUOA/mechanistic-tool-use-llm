"""SAKIKO figure visual system v2.

One palette, one type scale, one stroke vocabulary, shared by every figure.
Colour is semantic: a hue means the same thing in Fig 1 as in Fig 4.
Designed for two-column conference width; all sizes are chosen so that
nothing falls below ~5.5pt at final insertion scale.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from matplotlib.path import Path
import matplotlib.patches as mpatches

# ---------------------------------------------------------------- palette
INK      = "#1B1F26"   # primary type
SLATE    = "#33475B"   # frozen transformer / structural spine
BLUE     = "#3D6E9C"   # observation, router, representation modules
TEAL     = "#2F7D6A"   # gold arrival / supported / repair
VERM     = "#A8453B"   # other-wrong / broken / failure
GREY     = "#8B9199"   # source-retained / inactive / controls
FAINT    = "#E6EAEE"   # panel fills
HAIR     = "#C3CAD1"   # hairline rules
PAPER    = "#FFFFFF"

# tinted fills (used sparingly, never as decoration)
SLATE_F  = "#EDF1F5"
BLUE_F   = "#E7EEF5"
TEAL_F   = "#E6F0ED"
VERM_F   = "#F6E9E7"
GREY_F   = "#F0F2F4"

# ---------------------------------------------------------------- type scale
T_PANEL  = 7.2    # dashed-panel titles
T_LABEL  = 6.4    # box labels
T_MATH   = 6.6    # inline math
T_SMALL  = 5.6    # annotations
T_TINY   = 5.0    # sub-annotations
T_PL     = 8.0    # panel letters (a) (b)

LW_SPINE = 0.9    # backbone strokes
LW_BOX   = 0.7    # ordinary box strokes
LW_ARROW = 0.75
LW_HAIR  = 0.5


def apply() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": T_LABEL,
        "text.color": INK,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "mathtext.fontset": "dejavusans",
        "axes.linewidth": LW_BOX,
        "lines.linewidth": LW_ARROW,
    })


def canvas(w: float, h: float, xlim: tuple[float, float], ylim: tuple[float, float]):
    """Blank drawing surface in explicit data coordinates."""
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    return fig, ax


# ---------------------------------------------------------------- primitives
def box(ax, x, y, w, h, label="", *, fc=PAPER, ec=INK, lw=LW_BOX, fs=T_LABEL,
        tc=None, r=0.6, ls="-", z=3, va="center", pad_top=None, weight="normal"):
    """Rounded rectangle with a centred label."""
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0,rounding_size={r}",
                       linewidth=lw, edgecolor=ec, facecolor=fc,
                       linestyle=ls, zorder=z)
    ax.add_patch(p)
    if label:
        ty = y + h / 2 if pad_top is None else y + h - pad_top
        ax.text(x + w / 2, ty, label, ha="center", va=va,
                fontsize=fs, color=tc or INK, zorder=z + 1, weight=weight,
                linespacing=1.35)
    return p


def panel(ax, x, y, w, h, title="", *, ec=GREY, fc="none", lw=LW_HAIR,
          ls=(0, (3.2, 2.2)), tc=None, fs=T_PANEL, z=1, title_x=None):
    """Dashed grouping region with a title sitting on its top-left."""
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0,rounding_size=0.9",
                       linewidth=lw, edgecolor=ec, facecolor=fc,
                       linestyle=ls, zorder=z)
    ax.add_patch(p)
    if title:
        tx = x + 1.6 if title_x is None else title_x
        ax.text(tx, y + h - 0.05, title, ha="left", va="center",
                fontsize=fs, color=tc or ec, weight="bold", zorder=z + 2,
                bbox=dict(fc=PAPER, ec="none", pad=1.4))
    return p


def arrow(ax, xy1, xy2, *, color=INK, lw=LW_ARROW, style="-|>", ms=2.6,
          conn="arc3,rad=0", ls="-", z=4, alpha=1.0):
    a = FancyArrowPatch(xy1, xy2, arrowstyle=style, mutation_scale=ms * 3.2,
                        linewidth=lw, color=color, connectionstyle=conn,
                        linestyle=ls, zorder=z, alpha=alpha,
                        shrinkA=0.6, shrinkB=0.6)
    ax.add_patch(a)
    return a


def label(ax, x, y, s, *, fs=T_SMALL, color=INK, ha="center", va="center",
          weight="normal", z=6, style="normal", bg=False, rot=0):
    kw = dict(ha=ha, va=va, fontsize=fs, color=color, weight=weight,
              zorder=z, style=style, rotation=rot, linespacing=1.35)
    if bg:
        kw["bbox"] = dict(fc=PAPER, ec="none", pad=1.0)
    return ax.text(x, y, s, **kw)


def oplus(ax, x, y, r=0.95, *, color=INK, lw=LW_BOX, z=6, fc=PAPER):
    """Residual-add / injection operator."""
    ax.add_patch(Circle((x, y), r, fc=fc, ec=color, lw=lw, zorder=z))
    ax.plot([x - r * 0.55, x + r * 0.55], [y, y], color=color, lw=lw, zorder=z + 1)
    ax.plot([x, x], [y - r * 0.55, y + r * 0.55], color=color, lw=lw, zorder=z + 1)


def step_badge(ax, x, y, n, *, r=1.05, color=INK, fs=T_SMALL, z=8):
    """Numbered pass marker."""
    ax.add_patch(Circle((x, y), r, fc=color, ec="none", zorder=z))
    ax.text(x, y, str(n), ha="center", va="center", fontsize=fs,
            color=PAPER, weight="bold", zorder=z + 1)


def snowflake(ax, x, y, *, s=0.85, color=SLATE, lw=0.6, z=7):
    """Frozen-parameter glyph (geometric, not a cartoon icon)."""
    for ang in (0, 60, 120):
        import numpy as np
        a = np.deg2rad(ang)
        dx, dy = s * np.cos(a), s * np.sin(a)
        ax.plot([x - dx, x + dx], [y - dy, y + dy], color=color, lw=lw, zorder=z,
                solid_capstyle="round")


def bracket(ax, x0, x1, y, *, depth=0.8, color=INK, lw=LW_BOX, z=5, down=False):
    """Thin square bracket used for the evidential wrapper."""
    d = -depth if down else depth
    ax.plot([x0, x0, x1, x1], [y, y + d, y + d, y], color=color, lw=lw,
            zorder=z, solid_capstyle="butt", solid_joinstyle="miter")


def save(fig, stem, outdir):
    import os
    os.makedirs(outdir, exist_ok=True)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 400})):
        fig.savefig(f"{outdir}/{stem}.{ext}", bbox_inches="tight",
                    pad_inches=0.02, **kw)
    plt.close(fig)
    return f"{outdir}/{stem}.pdf"
