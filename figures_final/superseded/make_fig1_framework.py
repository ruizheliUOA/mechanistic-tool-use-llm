"""Figure 1 — the unified SAKIKO framework. Conceptual; contains no measured data.
Three layers: model flow (top), SAKIKO control flow (middle), evidence semantics
(bottom). Spec: figures_final/FIGURE_ROLE_MAP.md and the main-figure specification.
Manifest: F1."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import *
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

setup()
fig, ax = plt.subplots(figsize=(7.2, 5.9))
ax.set_xlim(-3.5, 101.5); ax.set_ylim(0, 118); ax.axis('off')

def box(x, y, w, h, ec=MUTED, fc='white', lw=0.9, z=3, ls='solid'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=0.8",
                 linewidth=lw, edgecolor=ec, facecolor=fc, zorder=z, linestyle=ls))
def arr(x1, y1, x2, y2, c=MUTED, lw=1.0, ms=8.0, ls='-'):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=ms,
                 linewidth=lw, color=c, zorder=4, linestyle=ls))
def layer_label(y, txt):
    ax.text(-1.4, y, ' '.join(txt), rotation=90, ha='center', va='center', fontsize=5.4,
            color=FAINT, fontweight='bold')

W, GAP = 17.4, 3.15          # stage geometry, reused by every layer
sx = lambda i: i * (W + GAP)  # left edge of stage i

# ══ LAYER 1 — model flow ═══════════════════════════════════════════════════
MY, MH = 101.5, 14.0
layer_label(MY + MH/2, "MODEL")
box(0, MY, 20.5, MH)
ax.text(10.2, MY + MH/2, "prompt +\navailable tools", ha='center', va='center',
        fontsize=6.0, color=MUTED, linespacing=1.4, zorder=5)
arr(21.1, MY + MH/2, 25.4, MY + MH/2)
box(26, MY, 21, MH)
ax.text(36.5, MY + MH - 4.0, "forward pass", ha='center', va='center', fontsize=6.2,
        color=INK, zorder=5)
ax.text(36.5, MY + 4.6, "observe  $h_{obs}$  at  $\\ell_{obs}$", ha='center', va='center',
        fontsize=5.8, color=MUTED, zorder=5)
arr(47.6, MY + MH/2, 51.9, MY + MH/2)
box(52.5, MY, 22, MH, ec=INK, lw=1.1)
ax.text(63.5, MY + MH - 4.0, "baseline decision  $\\hat{y}_0$", ha='center', va='center',
        fontsize=6.2, color=INK, zorder=5)
ax.text(63.5, MY + 6.2, "over  {tool call, request info,\nanswer directly, abstain}",
        ha='center', va='center', fontsize=4.9, color=INK, linespacing=1.4, zorder=5)
ax.text(63.5, MY + 1.9, "$\\hat{y}_0 \\neq y^*$ : error   ·   $\\hat{y}_0 = y^*$ : correct",
        ha='center', va='center', fontsize=4.7, color=MUTED, zorder=5)
arr(74.7, MY + MH/2, 79.0, MY + MH/2)
box(79.6, MY, 20.4, MH, ec=REAL, lw=1.1, fc='#f2f7f9')
ax.text(89.8, MY + MH - 4.0, "post-intervention  $\\hat{y}_1$", ha='center', va='center',
        fontsize=6.2, color=REAL, zorder=5)
ax.text(89.8, MY + 4.6, "$h' = h + q_c\\, s_c\\, d_c$  at  $\\ell_{inj}$", ha='center',
        va='center', fontsize=5.4, color=MUTED, zorder=5)
ax.text(0, MY + MH + 1.4, "SAKIKO acts on this decision — before any tool executes",
        ha='left', va='bottom', fontsize=6.2, color=REAL, style='italic')

# ══ LAYER 2 — SAKIKO control flow ══════════════════════════════════════════
SY, SH = 72.0, 22.0
layer_label(SY + SH/2, "SAKIKO")
ax.add_patch(Rectangle((sx(0) - 1.2, SY - 2.2), 3*W + 2*GAP + 2.4, SH + 8.6,
             facecolor='#f7f6f3', edgecolor='none', zorder=0))
ax.add_patch(Rectangle((sx(3) - 1.2, SY - 2.2), 2*W + GAP + 2.4, SH + 8.6,
             facecolor='#f1f5f7', edgecolor='none', zorder=0))
ax.text(sx(0) + (3*W + 2*GAP)/2, SY + SH + 4.2, "CORRECTION   ·   can we intervene?",
        ha='center', fontsize=6.2, color=MUTED, fontweight='bold')
ax.text(sx(3) + (2*W + GAP)/2, SY + SH + 4.2, "ADJUDICATION   ·   what did it accomplish?",
        ha='center', fontsize=6.2, color=REAL, fontweight='bold')
stages = [
    ("Discover", "which directional errors\ndoes this model make?", "directional error channels"),
    ("Detect",   "is this input in one of\nthose error states?",     "Readable $\\neq$ Steerable"),
    ("Correct",  "can it be moved toward\nthe intended action?",     "channel · estimator\nsite · dose"),
    ("Verify",   "where did it arrive, and\nwhat did that cost?",    "movement $\\neq$ repair"),
    ("License",  "what may we claim\nfrom this evidence?",           "claim follows evidence"),
]
for i, (name, q, note) in enumerate(stages):
    x = sx(i); emph = name == "Verify"
    box(x, SY, W, SH, ec=REAL if emph else MUTED, fc='#e6eff3' if emph else 'white',
        lw=1.7 if emph else 0.9)
    ax.text(x + W/2, SY + SH - 3.6, name, ha='center', va='center', fontsize=9.2,
            color=REAL if emph else INK, fontweight='bold' if emph else 'normal', zorder=5)
    ax.text(x + W/2, SY + 11.8, q, ha='center', va='center', fontsize=5.6, color=INK,
            style='italic', linespacing=1.45, zorder=5)
    ax.text(x + W/2, SY + 4.0, note, ha='center', va='center', fontsize=5.3, color=MUTED,
            linespacing=1.4, zorder=5)
    if i < 4: arr(x + W + 0.35, SY + SH/2, x + W + GAP - 0.35, SY + SH/2)
ax.text(sx(3) - GAP/2, SY - 4.4, "behaviour changed — but was it repaired?", ha='center',
        va='top', fontsize=6.0, color=REAL, style='italic')

# ══ LAYER 3 — evidence semantics ═══════════════════════════════════════════
NY = 60.0
layer_label(31.0, "EVIDENCE")
for i, txt in enumerate([r"$c = (y^* \rightarrow \hat{y}_0)$",
                         r"$r(h_{obs}) \rightarrow (\hat{c},\, p_c,\, \mathrm{fire?})$",
                         r"$\theta_c = \{d_c,\, \ell_{obs},\, \ell_{inj},\, q_c,\, \mathrm{gate}\}$"]):
    ax.text(sx(i) + W/2, NY, txt, ha='center', va='center', fontsize=5.4, color=INK, zorder=5)
ax.text(sx(1) + W/2, NY - 4.6, "no channel fires → unchanged", ha='center',
        va='center', fontsize=5.1, color=MUTED, style='italic')
ax.text(sx(2) + W/2, NY - 4.6, "matched-control specificity", ha='center',
        va='center', fontsize=5.1, color=MUTED, style='italic')

PY_, PH = 3.0, 48.0
box(0, PY_, 56.5, PH, ec=REAL, fc='#f7fafb', ls=(0, (3.5, 2.5)), lw=0.9, z=1)
arr(sx(3) + W/2, SY - 8.0, 40, PY_ + PH + 0.5, c=REAL, lw=0.8, ms=7, ls=(0, (2, 2)))
ax.text(2.8, PY_ + PH - 4.2, "Verify — every outcome resolved, not merely counted",
        fontsize=6.9, color=REAL, style='italic', va='center')
groups = [("was a baseline ERROR   $\\hat{y}_0 \\neq y^*$",
           [("SOURCE\nRETAINED", CONTROL, "$\\hat{y}_1 = \\hat{y}_0$"),
            ("GOLD\nARRIVAL", REAL, "$\\hat{y}_1 = y^*$"),
            ("OTHER\nWRONG", DAMAGE, "left source,\nstill wrong")]),
          ("was already CORRECT   $\\hat{y}_0 = y^*$",
           [("CORRECT\nRETAINED", REAL, "preserved"),
            ("BROKEN", DAMAGE, "damaged")])]
for (lbl, cats), cy in zip(groups, [32.0, 15.0]):
    ax.text(2.8, cy + 8.4, lbl, fontsize=5.6, color=MUTED, va='center')
    cx = 2.8
    for name, col, note in cats:
        ax.add_patch(FancyBboxPatch((cx, cy), 15.2, 6.0, boxstyle="round,pad=0.12,rounding_size=0.5",
                     linewidth=0, facecolor=col, alpha=0.9, zorder=3))
        ax.text(cx + 7.6, cy + 3.0, name, ha='center', va='center', fontsize=5.3,
                color='white', fontweight='bold', linespacing=1.3, zorder=4)
        ax.text(cx + 7.6, cy - 1.0, note, ha='center', va='top', fontsize=4.9, color=MUTED,
                linespacing=1.35, zorder=4)
        cx += 17.0
ax.text(2.8, PY_ + 4.4, r"$\mathrm{target\ hit} = \mathrm{GOLD\_ARRIVAL} \;/\; "
        r"(\mathrm{GOLD\_ARRIVAL} + \mathrm{OTHER\_WRONG})$", fontsize=5.3, color=INK, va='center')
ax.text(2.8, PY_ + 1.0, "collateral is reported on a named denominator; zero exposure is vacuous",
        fontsize=5.0, color=MUTED, style='italic', va='center')

box(60, PY_, 40, PH, ec=MUTED, lw=0.9, z=1)
ax.text(62.6, PY_ + PH - 4.2, "License — a property of the evidence", fontsize=6.9,
        color=INK, style='italic', va='center')
ladder = [("Adjudicable", "measurable at all"), ("Readable", "the state is decodable"),
          ("Steerable", "specific beyond controls"), ("Correctable", "arrivals concentrate on gold"),
          ("Preserving", "correct behaviour survives"), ("Licensable", "evidence supports the claim")]
ly = PY_ + PH - 9.4
for j, (r_, note) in enumerate(ladder):
    top = r_ == "Licensable"
    ax.text(63.4, ly, ("$\\rightarrow$ " if j else "") + r_, fontsize=6.0,
            color=REAL if top else INK, va='center', fontweight='bold' if top else 'normal')
    ax.text(79.0, ly, note, fontsize=4.8, color=MUTED, va='center')
    ly -= 4.15
ax.text(62.6, PY_ + 12.8, "Correctable $\\neq$ Preserving $\\neq$ Licensable",
        fontsize=5.7, color=INK, va='top', fontweight='bold')
ax.text(62.6, PY_ + 8.4, "a favourable estimate can still be declined", fontsize=5.4,
        color=MUTED, va='top', style='italic')
ax.text(62.6, PY_ + 5.6, "DECLINE = insufficient evidence,\nnot intrinsic uncorrectability",
        fontsize=5.4, color=DECLINE, va='top', linespacing=1.55)

out = 'figures_final/fig1_framework.png'
plt.savefig(out); plt.savefig(out.replace('.png', '.pdf'))
print(f"  wrote {out}  (conceptual — no measured data)")
