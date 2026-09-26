"""Unified visual system for all SAKIKO figures. Import before plotting."""
import matplotlib as mpl, matplotlib.pyplot as plt
# low-saturation, colour-blind conscious, legible in grayscale (distinct L*)
C = dict(
    structure = "#2E4057",   # muted navy  — framework / structure
    pass_     = "#3F7D6E",   # muted teal  — pass / licensed
    fail      = "#A8433B",   # muted verm. — fail / stop
    caution   = "#B8873A",   # muted amber — inconclusive / retrospective
    neutral   = "#8A8F98",   # grey        — not reached / unavailable
    faint     = "#DDE1E6",   # rules, fills
    ink       = "#1C1F24",   # text
    bg        = "#FFFFFF",
)
GREY_L = {"structure":0.34,"pass_":0.46,"fail":0.40,"caution":0.58,"neutral":0.62}
def apply():
    mpl.rcParams.update({
        "figure.dpi":160, "savefig.dpi":300, "figure.facecolor":C["bg"], "savefig.facecolor":C["bg"],
        "font.family":"DejaVu Sans", "font.size":8.5,
        "axes.titlesize":9.5, "axes.labelsize":8.5, "axes.edgecolor":C["ink"],
        "axes.linewidth":0.7, "axes.facecolor":C["bg"], "axes.grid":False,
        "xtick.labelsize":7.8, "ytick.labelsize":7.8,
        "xtick.color":C["ink"], "ytick.color":C["ink"], "text.color":C["ink"],
        "xtick.major.width":0.7, "ytick.major.width":0.7,
        "legend.fontsize":7.5, "legend.frameon":False,
        "savefig.bbox":"tight", "savefig.pad_inches":0.02,
        "pdf.fonttype":42, "ps.fonttype":42,   # embed as TrueType, editable
    })
def save(fig, stem, outdir):
    import os
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(f"{outdir}/{stem}.pdf")
    fig.savefig(f"{outdir}/{stem}.png", dpi=300)
    plt.close(fig)
    return f"{outdir}/{stem}.pdf"
