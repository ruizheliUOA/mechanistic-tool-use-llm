"""Figure 1 candidate A — two-pass method figure.

Architecture-forward, in the idiom of dense CV/ML method figures: an offline
preparation strip, the frozen backbone as the visual centre drawn across the
two passes the implementation actually performs (L_inj = 21 precedes
L_obs = 26, so diagnosis and intervention cannot share one forward pass),
outcome accounting on the right over two disjoint populations, and the
evidential structure as a thin wrapper rather than a row of boxes.

No model names, no empirical numbers: this is the conceptual contribution.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sakiko_v2_style import *   # noqa: F403
import sakiko_v2_style as S

S.apply()
fig, ax = S.canvas(7.2, 4.9, (0, 100), (0, 66))

def stack(ax, x0, y0, w, h, n, *, mark=None, gap=0.42):
    bw = (w - gap * (n - 1)) / n
    out = []
    for i in range(n):
        x = x0 + i * (bw + gap)
        hit = mark is not None and i == mark
        S.box(ax, x, y0, bw, h, fc=BLUE_F if hit else SLATE_F,
              ec=BLUE if hit else SLATE, lw=1.0 if hit else 0.6, r=0.3, z=3)
        out.append(x + bw / 2)
    return out

NB, I_INJ, I_OBS = 9, 3, 6

# ==================================================== OFFLINE
S.panel(ax, 1.0, 54.4, 55.6, 11.2,
        "OFFLINE   ·   channel discovery and candidate construction", tc=SLATE)
S.box(ax, 2.8, 56.6, 10.0, 6.4, "baseline\npredictions\n+ gold",
      fc=GREY_F, ec=GREY, fs=T_TINY)
S.arrow(ax, (13.0, 59.8), (15.4, 59.8), color=GREY)
S.box(ax, 15.6, 56.6, 11.0, 6.4, "directed error\ntopology",
      fc=GREY_F, ec=GREY, fs=T_TINY)
S.arrow(ax, (26.8, 59.8), (29.2, 59.8), color=GREY)
S.box(ax, 29.4, 56.6, 12.2, 6.4, "supported\nchannel\n$c=(g\\rightarrow s)$",
      fc=BLUE_F, ec=BLUE, fs=T_TINY)
S.arrow(ax, (41.8, 61.4), (44.2, 61.6), color=BLUE)
S.arrow(ax, (41.8, 58.2), (44.2, 58.0), color=BLUE)
S.box(ax, 44.4, 59.8, 10.6, 3.4, "router  $r_c$", fc=BLUE_F, ec=BLUE, fs=T_TINY)
S.box(ax, 44.4, 56.2, 10.6, 3.4, "direction  $d_c$", fc=BLUE_F, ec=BLUE, fs=T_TINY)
S.label(ax, 26.0, 55.2, "channels are discovered from the model's own errors, not assumed;"
        "   $d_c$ is channel-specific", fs=T_TINY, color=GREY, style="italic")

# ==================================================== PASS 1
S.panel(ax, 1.0, 40.0, 55.6, 13.2, "", ec=HAIR)
S.step_badge(ax, 3.5, 51.7, 1, color=SLATE)
S.label(ax, 5.4, 51.7, "DIAGNOSTIC PASS   ·   read", fs=T_PANEL,
        weight="bold", color=SLATE, ha="left")
S.box(ax, 2.2, 42.6, 8.4, 6.2, "user request\n+ tool context", fs=T_TINY)
S.arrow(ax, (10.8, 45.7), (12.4, 45.7))
b1 = stack(ax, 12.6, 42.6, 21.0, 6.2, NB, mark=I_OBS)
S.snowflake(ax, 17.4, 50.3, color=SLATE)
S.label(ax, 18.9, 50.3, r"$\theta_{\mathrm{LLM}}$  frozen", fs=T_TINY,
        color=SLATE, ha="left")
S.label(ax, b1[0], 41.4, "$B_1$", fs=T_TINY, color=SLATE)
S.label(ax, b1[I_OBS], 41.4, "$L_{\\mathrm{obs}}$", fs=T_TINY, color=BLUE, weight="bold")
ax.plot([b1[I_OBS], b1[I_OBS]], [48.8, 50.0], color=BLUE, lw=LW_ARROW, zorder=4)
ax.plot([b1[I_OBS], 41.5], [50.0, 50.0], color=BLUE, lw=LW_ARROW, zorder=4)
S.arrow(ax, (41.5, 50.0), (41.5, 48.1), color=BLUE)
S.label(ax, 34.6, 50.9, "$h_{\\mathrm{obs}}\\in\\mathbb{R}^{d}$", fs=T_MATH,
        color=BLUE, bg=True)
S.box(ax, 36.6, 43.4, 9.8, 4.6, "router\n$r_c(h_{\\mathrm{obs}})$",
      fc=BLUE_F, ec=BLUE, fs=T_TINY)
S.arrow(ax, (46.6, 45.7), (48.6, 45.7), color=BLUE)
S.box(ax, 48.8, 43.4, 7.4, 4.6, "$p_c \\gtrless \\tau$", ec=BLUE, fs=T_MATH)
S.label(ax, 52.5, 41.8, "gate", fs=T_TINY, color=BLUE)

# ==================================================== PASS 2
S.panel(ax, 1.0, 24.4, 55.6, 14.6, "", ec=HAIR)
S.step_badge(ax, 3.5, 37.4, 2, color=SLATE)
S.label(ax, 5.4, 37.4, "INTERVENTION PASS   ·   gated write", fs=T_PANEL,
        weight="bold", color=SLATE, ha="left")
S.arrow(ax, (6.4, 42.5), (6.4, 32.4), color=GREY, ls=(0, (2, 1.7)))
S.label(ax, 6.4, 30.6, "same\ninput", fs=T_TINY, color=GREY)
S.arrow(ax, (10.8, 29.5), (12.4, 29.5))
b2 = stack(ax, 12.6, 26.4, 21.0, 6.2, NB, mark=I_INJ)
S.label(ax, b2[I_INJ], 25.2, "$L_{\\mathrm{inj}}$", fs=T_TINY, color=BLUE, weight="bold")
S.label(ax, b2[-1], 25.2, "$B_L$", fs=T_TINY, color=SLATE)
S.oplus(ax, b2[I_INJ], 33.8, r=0.95, color=BLUE)
S.arrow(ax, (b2[I_INJ], 32.8), (b2[I_INJ], 32.7), color=BLUE)
S.label(ax, b2[I_INJ] + 1.7, 33.8, "$h' = h + \\alpha\\, d_c$", fs=T_MATH,
        color=BLUE, ha="left")
S.arrow(ax, (33.8, 29.5), (35.5, 29.5))
S.box(ax, 35.7, 26.4, 9.6, 6.2, "action\nreadout", fs=T_TINY)
S.arrow(ax, (45.5, 29.5), (47.3, 29.5))
S.box(ax, 47.5, 27.4, 6.2, 4.2, "$\\hat{a}'$", fs=T_MATH)
S.label(ax, 40.5, 25.2, "$[z_{\\mathrm{tool}},z_{\\mathrm{rfi}},"
        "z_{\\mathrm{cannot}},z_{\\mathrm{direct}}]$", fs=T_TINY, color=GREY)

# gate -> injection routing
for seg in ([(52.5, 43.3), (52.5, 40.7)], [(52.5, 40.7), (b2[I_INJ], 40.7)]):
    ax.plot([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]], color=BLUE,
            lw=LW_ARROW, zorder=4)
S.arrow(ax, (b2[I_INJ], 40.7), (b2[I_INJ], 34.9), color=BLUE)
S.label(ax, 36.6, 41.6, "fire  if  $p_c\\geq\\tau$", fs=T_TINY, color=BLUE, bg=True)
S.label(ax, 2.6, 23.2, "otherwise the forward pass is left untouched — no prompt edit, "
        "no weight update, no rewriting of the emitted answer",
        fs=T_TINY, color=GREY, style="italic", ha="left")

# ==================================================== EXPANDED BLOCK
S.panel(ax, 1.0, 16.9, 55.6, 5.6, "", ec=HAIR)
S.label(ax, 2.6, 21.9, "one block, expanded", fs=T_TINY, weight="bold",
        color=SLATE, ha="left", bg=True)
bx = 12.0
for lab, wdt in (("Norm", 4.2), ("Attn", 4.8), ("Norm", 4.2), ("MLP", 4.8)):
    S.box(ax, bx, 17.8, wdt, 2.9, lab, fc=SLATE_F, ec=SLATE, fs=T_TINY, r=0.3)
    bx += wdt + 2.6
ax.plot([12.0, 34.6], [21.4, 21.4], color=SLATE, lw=LW_HAIR, zorder=2)
S.oplus(ax, 19.4, 21.4, r=0.58, color=SLATE)
S.oplus(ax, 31.0, 21.4, r=0.58, color=SLATE)
S.label(ax, 25.2, 22.4, "residual stream", fs=T_TINY, color=SLATE, bg=True)
S.arrow(ax, (39.0, 19.3), (41.0, 19.3), color=BLUE, ms=2.0)
S.label(ax, 41.6, 19.3, "$h$ is read and written at the MLP output\n"
        "of the final prompt token", fs=T_TINY, color=BLUE, ha="left")
for xs in (b2[I_INJ], b1[I_OBS]):
    ax.plot([xs, xs], [16.9, 15.9], color=HAIR, lw=0.0)

# ==================================================== OUTCOME BUS
ax.plot([53.7, 57.0], [29.5, 29.5], color=INK, lw=LW_ARROW, zorder=4)
ax.plot([57.0, 57.0], [29.5, 54.0], color=INK, lw=LW_ARROW, zorder=4)
S.arrow(ax, (57.0, 54.0), (59.8, 54.0), color=INK)
ax.add_patch(__import__("matplotlib.patches", fromlist=["Circle"])
             .Circle((60.1, 54.0), 0.32, fc=INK, ec="none", zorder=6))
S.arrow(ax, (57.0, 34.6), (60.4, 34.6), color=INK)

# ==================================================== DESTINATIONS
S.panel(ax, 58.4, 44.6, 40.8, 21.0,
        "DESTINATION-RESOLVED OUTCOME", tc=SLATE)
S.label(ax, 60.2, 63.6, "for a baseline error on channel $c=(g\\rightarrow s)$, exactly three\n"
        "mutually exclusive post-intervention outcomes:", fs=T_TINY, color=GREY,
        ha="left", style="italic")
rows = [
    (58.6, GREY, GREY_F, "SOURCE_RETAINED", "$\\hat{a}'=s$", "error persists"),
    (54.0, TEAL, TEAL_F, "GOLD_ARRIVAL", "$\\hat{a}'=g$", "genuine repair"),
    (49.4, VERM, VERM_F, "OTHER_WRONG",
     "$\\hat{a}'=w\\in A_D\\setminus\\{g,s\\}$", "error redistributed"),
]
for yy, col, fcol, nm, ms, note in rows:
    S.box(ax, 62.6, yy - 1.85, 26.4, 3.7, "", fc=fcol, ec=col, lw=0.7, r=0.4)
    S.label(ax, 63.6, yy + 0.55, nm, fs=T_TINY, color=col, weight="bold", ha="left")
    S.label(ax, 63.6, yy - 0.95, ms + "    " + note, fs=T_TINY, ha="left")
    S.arrow(ax, (60.1, 54.0), (62.5, yy), color=col, lw=0.7)

ax.plot([90.2, 90.2], [60.8, 47.2], color=HAIR, lw=LW_HAIR, zorder=2)
S.label(ax, 93.6, 61.6, "under conventional\naccuracy", fs=T_TINY, color=GREY,
        style="italic")
for yy, col, verdict in ((58.6, GREY, "still wrong"), (54.0, TEAL, "correct"),
                         (49.4, VERM, "still wrong")):
    ax.plot([89.1, 90.2], [yy, yy], color=col, lw=LW_HAIR, zorder=4)
    S.label(ax, 90.8, yy, verdict, fs=T_TINY, color=col, ha="left",
            weight="bold" if col is TEAL else "normal")
S.label(ax, 78.8, 46.0, "conventional accuracy collapses  {SOURCE_RETAINED, OTHER_WRONG};\n"
        "destination resolution separates all three:    "
        "source exit  =  gold arrival  +  other-wrong", fs=T_TINY)

# ==================================================== PRESERVATION
S.panel(ax, 58.4, 24.4, 40.8, 18.6,
        "SECOND POPULATION   ·   preservation", tc=SLATE)
S.box(ax, 60.6, 32.4, 13.4, 4.6, "baseline-correct,\nrouter-exposed", fs=T_TINY)
S.arrow(ax, (74.2, 36.0), (77.4, 37.4), color=TEAL, conn="arc3,rad=0.14")
S.arrow(ax, (74.2, 33.4), (77.4, 31.8), color=VERM, conn="arc3,rad=-0.14")
S.box(ax, 77.6, 35.9, 11.0, 3.1, "retained", fc=TEAL_F, ec=TEAL, fs=T_TINY, tc=TEAL)
S.box(ax, 77.6, 30.3, 11.0, 3.1, "broken", fc=VERM_F, ec=VERM, fs=T_TINY, tc=VERM)
S.label(ax, 78.8, 27.4, "collateral damage is a distinct estimand, measured on a\n"
        "disjoint population — it is never netted against repairs",
        fs=T_TINY, color=GREY, style="italic")

# ==================================================== EVIDENTIAL WRAPPER
S.panel(ax, 1.0, 1.0, 98.2, 14.6, "", ec=SLATE, ls=(0, (1.5, 1.7)))
S.label(ax, 2.8, 14.6, "SAKIKO CLAIM STRUCTURE", fs=T_PANEL, weight="bold",
        color=SLATE, ha="left", bg=True)
S.box(ax, 3.0, 8.6, 13.0, 4.4, "Adjudicable", ec=SLATE, lw=1.0)
S.label(ax, 9.5, 7.4, "entry: can the study\nanswer the question?", fs=T_TINY, color=GREY)
S.arrow(ax, (16.2, 10.8), (19.4, 10.8), color=SLATE)
S.panel(ax, 19.6, 8.0, 34.6, 5.6, "", ec=BLUE, ls="-", lw=0.7, fc=BLUE_F)
for nm, xx in (("Readable", 21.0), ("Steerable", 31.4), ("Correctable", 41.8)):
    S.box(ax, xx, 9.0, 10.0, 3.6, nm, ec=BLUE, fs=T_LABEL, r=0.4)
S.arrow(ax, (31.0, 10.8), (31.3, 10.8), color=BLUE, ms=2.2)
S.arrow(ax, (41.4, 10.8), (41.7, 10.8), color=BLUE, ms=2.2)
S.label(ax, 36.9, 13.6, "SCIENTIFIC PROPERTY LAYER", fs=T_TINY, weight="bold",
        color=BLUE, bg=True)
S.label(ax, 36.9, 6.4, "a property claim about the tested setting and intervention",
        fs=T_TINY, color=BLUE, style="italic")
S.arrow(ax, (54.4, 10.8), (57.8, 10.8), color=SLATE)
S.box(ax, 58.0, 8.4, 22.6, 4.8, "", ec=SLATE, lw=1.5, r=0.5)
S.box(ax, 58.5, 8.8, 21.6, 4.0, "Licensable", ec=SLATE, lw=0.6, fs=T_LABEL, r=0.4)
S.label(ax, 69.3, 13.6, "EVIDENTIAL DECISION", fs=T_TINY, weight="bold",
        color=SLATE, bg=True)
S.label(ax, 82.0, 10.8, "ADMIT  /  DECLINE  /  NO-GO", fs=T_TINY, ha="left")
S.label(ax, 82.0, 8.8, "DECLINE $\\neq$ intrinsically\nuncorrectable", fs=T_TINY,
        color=VERM, ha="left")
cx = 22.0
for nm in ("Specificity", "Destination", "Preservation", "Yield", "Evidence"):
    S.box(ax, cx, 1.9, 9.4, 2.8, nm, fc=GREY_F, ec=GREY, fs=T_TINY, r=0.3)
    cx += 10.0
ax.plot([26.7, 62.7], [5.4, 5.4], color=GREY, lw=LW_HAIR, zorder=4)
for i in range(5):
    ax.plot([26.7 + i * 10.0, 26.7 + i * 10.0], [4.7, 5.4], color=GREY, lw=0.6, zorder=4)
S.arrow(ax, (62.7, 5.4), (66.0, 5.4), color=GREY, ms=2.2)
S.arrow(ax, (66.0, 5.4), (66.0, 8.3), color=GREY, ms=2.2)
S.label(ax, 11.2, 3.3, "EVIDENTIAL\nQUALIFICATION", fs=T_TINY, weight="bold", color=GREY)

S.save(fig, "fig1_sakiko_core_A", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "candidates"))
print("A rendered")
