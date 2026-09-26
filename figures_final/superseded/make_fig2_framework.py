"""Figure 2 — the SAKIKO framework. Conceptual; contains no measured data.

Top: the model's pre-execution decision. Bottom: the five stages as facet-style
panels, each headed by the question it answers (Section 3). Dashed links mark where
SAKIKO reads the forward pass (Detect), writes to it (Correct) and resolves the
resulting action (Verify). The rung ladder is not repeated here: Figure 4 owns it.
Outcome classes use the same colours as every data figure."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import *
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

theme_grey()
FW, FH = TEXT_W, 2.46
fig = plt.figure(figsize=(FW, FH))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, FW); ax.set_ylim(0, FH); ax.axis('off')

S = 0.8 * BASE          # strip and body text size
STRIP_H = STRIP_PT / 72

def box(x, y, w, h, fc=GREY92, ec='none', lw=0.0, r=0.03, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}',
                                facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z))

def arrow(x0, y0, x1, y1, ls='-', color=GREY30, lw=0.8, ms=6):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=ms,
                                 linewidth=lw, color=color, linestyle=ls, zorder=4,
                                 shrinkA=0, shrinkB=0))

def chip(x, y, w, h, fc, text, tc):
    box(x, y, w, h, fc=fc, r=0.02, z=3)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=0.72 * BASE,
            color=tc, zorder=4)

# ══ the model's decision ══════════════════════════════════════════════════
X0, GAP = 0.05, 0.09
W = [0.84, 0.84, 0.96, 1.44, 0.96]                      # Discover … License
XS = [X0 + sum(W[:i]) + GAP * i for i in range(5)]
MY, MH = 1.93, 0.33
ax.text(X0, MY + MH + 0.06, 'the model: one pre-execution decision, before any tool executes',
        fontsize=0.75 * BASE, color=GREY30, style='italic', va='bottom')
model = [(XS[0], W[0], 'prompt +\navailable tools'),
         (XS[1], W[1], 'forward pass\nread $h_{\\mathrm{obs}}$ at $\\ell_{\\mathrm{obs}}$'),
         (XS[2], W[2], "$h' = h + q_c\\, s_c\\, d_c$\nat $\\ell_{\\mathrm{inj}}$, if the Router fires"),
         (XS[3], W[3] + GAP + W[4],
          'action $\\hat{y}_0$ (baseline) or $\\hat{y}_1$ (after intervention)\n'
          'tool call · request info · direct answer · cannot answer')]
for x, w, t in model:
    box(x, MY, w, MH, fc=GREY92)
    ax.text(x + w / 2, MY + MH / 2, t, ha='center', va='center', fontsize=0.72 * BASE,
            color=GREY10, linespacing=1.25, zorder=4)
for (x, w, _), (xn, _, _) in zip(model[:-1], model[1:]):
    arrow(x + w + 0.005, MY + MH / 2, xn - 0.005, MY + MH / 2, ms=5)

# ══ the five stages ═══════════════════════════════════════════════════════
PY, PH = 0.34, 1.10
TOP = PY + PH + STRIP_H
stages = [('Discover', 'Which errors\ndoes this model\nactually make?'),
          ('Detect', 'Is this input\nin one of those\nstates?'),
          ('Correct', 'Can the decision\nbe moved toward\nthe required action?'),
          ('Verify', 'Where did the decision go,\nand what did it cost?'),
          ('License', 'What may this\nevidence support?')]
for (name, q), x, w in zip(stages, XS, W):
    box(x, PY, w, PH, fc=GREY92, r=0.0)
    box(x, PY + PH, w, STRIP_H, fc=GREY85, r=0.0)
    ax.text(x + w / 2, PY + PH + STRIP_H / 2, name, ha='center', va='center', fontsize=S,
            color=GREY10, zorder=4)
    ax.text(x + 0.06, PY + PH - 0.07, q, ha='left', va='top', fontsize=0.78 * BASE,
            style='italic', color='black', linespacing=1.15, zorder=4)
for i in range(4):
    arrow(XS[i] + W[i] + 0.008, PY + PH / 2, XS[i + 1] - 0.008, PY + PH / 2, ms=5)

def body(i, lines, y0=PY + 0.40):
    for k, (t, c) in enumerate(lines):
        ax.text(XS[i] + 0.06, y0 - k * 0.135, t, ha='left', va='center', fontsize=0.72 * BASE,
                color=c, zorder=4)

body(0, [('$c = (y^\\star \\rightarrow \\hat{y}_0)$', GREY10), ('per model, per task', GREY30),
         ('sparse ones set aside', GREY30)])
body(1, [('$r(h_{\\mathrm{obs}}) \\rightarrow (\\hat{c},\\, p_{\\hat{c}},\\, \\mathrm{fire})$', GREY10),
         ('per-channel Router', GREY30), ('no fire: baseline kept', GREY30)])
body(2, [('$\\theta_c$: estimator, sites, dose', GREY10), ('frozen before evaluation', GREY30),
         ('tested against controls', GREY30)])

# Verify: every outcome resolved, in the colours of the data figures
vx, vw = XS[3] + 0.06, W[3] - 0.12
ax.text(vx, PY + 0.66, 'baseline error', fontsize=0.66 * BASE, color=GREY30, va='bottom')
cw = (vw - 0.06) / 3
for k, (fc, t, tc) in enumerate([(UNCHANGED, 'retained', GREY10), (ARRIVE, 'gold arrival', 'white'),
                                 (OTHER, 'other wrong', GREY10)]):
    chip(vx + k * (cw + 0.03), PY + 0.47, cw, 0.17, fc, t, tc)
ax.text(vx, PY + 0.30, 'baseline-correct, Router fired', fontsize=0.66 * BASE, color=GREY30,
        va='bottom')
for k, (fc, t, tc) in enumerate([(UNCHANGED, 'kept', GREY10), (BROKEN, 'broken', 'white')]):
    chip(vx + k * (cw + 0.03), PY + 0.11, cw, 0.17, fc, t, tc)
ax.text(vx + 2 * (cw + 0.03) + cw / 2, PY + 0.195, 'on a named\ndenominator', ha='center',
        va='center', fontsize=0.62 * BASE, color=GREY30, linespacing=1.1)

# License: a frozen verdict
lx, lw_ = XS[4] + 0.06, W[4] - 0.12
chip(lx, PY + 0.47, lw_ / 2 - 0.015, 0.17, ARRIVE, 'ADMIT', 'white')
chip(lx + lw_ / 2 + 0.015, PY + 0.47, lw_ / 2 - 0.015, 0.17, DECLINE, 'DECLINE', DECLINE_TXT)
ax.text(lx, PY + 0.41, 'ten criteria fixed\nbefore evaluation', fontsize=0.66 * BASE,
        color=GREY30, va='top', linespacing=1.15)
ax.text(lx, PY + 0.035, 'DECLINE: the evidence\nis insufficient', fontsize=0.66 * BASE,
        color=DECLINE_TXT, va='bottom', linespacing=1.15)

# ══ where SAKIKO touches the model ════════════════════════════════════════
arrow(XS[1] + W[1] / 2, MY - 0.01, XS[1] + W[1] / 2, TOP + 0.01, ls=(0, (2, 1.6)), color=GREEN_TXT)
ax.text(XS[1] + W[1] / 2 + 0.04, (MY + TOP) / 2, 'reads', fontsize=0.66 * BASE, color=GREEN_TXT,
        va='center')
arrow(XS[2] + W[2] / 2, TOP + 0.01, XS[2] + W[2] / 2, MY - 0.01, ls=(0, (2, 1.6)), color=ARRIVE_TXT)
ax.text(XS[2] + W[2] / 2 + 0.04, (MY + TOP) / 2, 'writes', fontsize=0.66 * BASE, color=ARRIVE_TXT,
        va='center')
arrow(XS[3] + W[3] / 2, MY - 0.01, XS[3] + W[3] / 2, TOP + 0.01, ls=(0, (2, 1.6)), color=GREY30)
ax.text(XS[3] + W[3] / 2 + 0.04, (MY + TOP) / 2, 'resolves $\\hat{y}_1$ against $\\hat{y}_0$ and $y^\\star$',
        fontsize=0.66 * BASE, color=GREY30, va='center')

# ══ what the stages do ════════════════════════════════════════════════════
def group(x0, x1, text):
    y = PY - 0.07
    ax.plot([x0, x0, x1, x1], [y + 0.03, y, y, y + 0.03], color=GREY30, lw=0.6)
    ax.text((x0 + x1) / 2, y - 0.03, text, ha='center', va='top', fontsize=0.75 * BASE,
            color=GREY30, style='italic')
group(XS[0], XS[2] + W[2], 'produce a correction')
group(XS[3], XS[4] + W[4], 'interpret it: is the correction a repair?')

save(fig, 'fig2_framework')
print('    conceptual — no measured data')
