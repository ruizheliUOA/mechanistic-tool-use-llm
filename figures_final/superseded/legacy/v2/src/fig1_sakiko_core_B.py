"""Figure 1 candidate B — nested claim architecture.

Same science as A, different grammar. Panel (a) compresses the mechanism into
one spine with an explicit re-entry arc for the second pass. Panel (b) replaces
the property chain with nested regions: each stronger claim is a strict subset
of the weaker one it presupposes, which a left-to-right flowchart cannot show.
Licensing is drawn as a gate the innermost claim must pass through, not as a
fourth ring, so it stays a different object type.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sakiko_v2_style import *   # noqa: F403
import sakiko_v2_style as S

S.apply()
fig, ax = S.canvas(7.2, 4.6, (0, 100), (0, 62))

# ==================================================== (a) MECHANISM
S.label(ax, 1.4, 60.4, "(a)", fs=T_PL, weight="bold", ha="left")
S.label(ax, 5.4, 60.4, "conditional intervention in a frozen model",
        fs=T_PANEL, weight="bold", color=SLATE, ha="left")

S.box(ax, 1.4, 49.6, 8.6, 5.8, "user request\n+ tool context", fs=T_TINY)
S.arrow(ax, (10.2, 52.5), (12.0, 52.5))
n, gap, x0, w = 11, 0.42, 12.2, 24.0
bw = (w - gap * (n - 1)) / n
cx = []
for i in range(n):
    x = x0 + i * (bw + gap)
    hit = i in (3, 7)
    S.box(ax, x, 49.4, bw, 6.2, fc=BLUE_F if hit else SLATE_F,
          ec=BLUE if hit else SLATE, lw=1.0 if hit else 0.6, r=0.3)
    cx.append(x + bw / 2)
S.snowflake(ax, 17.0, 57.6, color=SLATE)
S.label(ax, 18.4, 57.6, r"$\theta_{\mathrm{LLM}}$ frozen", fs=T_TINY, color=SLATE, ha="left")
S.label(ax, cx[3], 48.2, "$L_{\\mathrm{inj}}$", fs=T_TINY, color=BLUE, weight="bold")
S.label(ax, cx[7], 48.2, "$L_{\\mathrm{obs}}$", fs=T_TINY, color=BLUE, weight="bold")

# pass 1: read at L_obs
S.step_badge(ax, cx[7], 58.4, 1, r=0.95, color=BLUE)
ax.plot([cx[7], cx[7]], [55.7, 57.4], color=BLUE, lw=LW_ARROW, zorder=4)
S.arrow(ax, (cx[7] + 1.0, 58.4), (39.8, 58.4), color=BLUE)
S.box(ax, 40.0, 56.4, 9.0, 4.0, "router $r_c$", fc=BLUE_F, ec=BLUE, fs=T_TINY)
S.arrow(ax, (49.2, 58.4), (51.0, 58.4), color=BLUE)
S.box(ax, 51.2, 56.4, 6.6, 4.0, "$p_c\\!\\gtrless\\!\\tau$", ec=BLUE, fs=T_TINY)
S.label(ax, 35.0, 57.0, "$h_{\\mathrm{obs}}$", fs=T_MATH, color=BLUE, bg=True)

# pass 2: re-entry arc back to the earlier layer
S.step_badge(ax, cx[3], 45.2, 2, r=0.95, color=BLUE)
ax.plot([54.5, 54.5], [56.3, 45.2], color=BLUE, lw=LW_ARROW, zorder=4)
S.arrow(ax, (54.5, 45.2), (cx[3] + 1.1, 45.2), color=BLUE)
ax.plot([cx[3], cx[3]], [46.2, 47.4], color=BLUE, lw=LW_ARROW, zorder=4)
S.arrow(ax, (cx[3], 47.4), (cx[3], 49.3), color=BLUE)
S.label(ax, 41.0, 44.0, "fire: re-run the same input, writing "
        "$h'=h+\\alpha d_c$ at $L_{\\mathrm{inj}}$", fs=T_TINY, color=BLUE, ha="center")
S.label(ax, 24.0, 41.8, "$L_{\\mathrm{inj}} < L_{\\mathrm{obs}}$ — diagnosis and intervention are separate passes",
        fs=T_TINY, color=GREY, style="italic")

S.arrow(ax, (36.4, 52.5), (38.2, 52.5))
S.box(ax, 38.4, 50.0, 8.6, 5.0, "action\nreadout", fs=T_TINY)
S.arrow(ax, (47.2, 52.5), (49.0, 52.5))

# destinations, compact
S.panel(ax, 58.8, 43.0, 40.4, 18.0, "DESTINATION-RESOLVED OUTCOME", tc=SLATE)
S.label(ax, 60.4, 58.6, "one directed error $c=(g\\rightarrow s)$, three outcomes",
        fs=T_TINY, color=GREY, ha="left", style="italic")
for yy, col, fc_, nm, ms in ((55.8, GREY, GREY_F, "SOURCE_RETAINED", "$\\hat a'=s$"),
                             (51.6, TEAL, TEAL_F, "GOLD_ARRIVAL", "$\\hat a'=g$"),
                             (47.4, VERM, VERM_F, "OTHER_WRONG",
                              "$\\hat a'=w\\notin\\{g,s\\}$")):
    S.box(ax, 63.0, yy - 1.7, 22.0, 3.4, "", fc=fc_, ec=col, lw=0.7, r=0.4)
    S.label(ax, 64.0, yy + 0.5, nm, fs=T_TINY, color=col, weight="bold", ha="left")
    S.label(ax, 64.0, yy - 1.0, ms, fs=T_TINY, ha="left")
    S.arrow(ax, (61.2, 51.6), (62.9, yy), color=col, lw=0.7)
    ax.plot([85.4, 86.4], [yy, yy], color=col, lw=LW_HAIR, zorder=4)
    S.label(ax, 86.9, yy, "correct" if col is TEAL else "still wrong", fs=T_TINY,
            color=col, ha="left", weight="bold" if col is TEAL else "normal")
ax.add_patch(__import__("matplotlib.patches", fromlist=["Circle"])
             .Circle((61.2, 51.6), 0.3, fc=INK, ec="none", zorder=6))
ax.plot([53.0, 53.0], [52.5, 51.6], color=INK, lw=LW_ARROW, zorder=4)
ax.plot([49.0, 53.0], [52.5, 52.5], color=INK, lw=LW_ARROW, zorder=4)
S.arrow(ax, (53.0, 51.6), (60.9, 51.6), color=INK)
S.label(ax, 79.4, 44.4, "conventional accuracy collapses the two wrong outcomes",
        fs=T_TINY, color=GREY, style="italic")

# ==================================================== (b) NESTED CLAIMS
S.label(ax, 1.4, 39.6, "(b)", fs=T_PL, weight="bold", ha="left")
S.label(ax, 5.4, 39.6, "what was shown, and what may be asserted",
        fs=T_PANEL, weight="bold", color=SLATE, ha="left")

# outer entry condition
S.panel(ax, 2.0, 3.0, 50.0, 33.4, "", ec=SLATE, ls=(0, (3.0, 2.2)))
S.label(ax, 4.0, 36.4, "ADJUDICABLE", fs=T_TINY, weight="bold", color=SLATE,
        ha="left", bg=True)
S.label(ax, 27.0, 4.2, "entry condition — the study can answer the question at all:\n"
        "$|A_D|\\geq 3$, a supported channel, sufficient rows in every split",
        fs=T_TINY, color=SLATE, style="italic")

nest = [
    (4.6, 8.4, 44.8, 25.6, BLUE, "#F2F6FA", "Readable",
     "the error is linearly decodable from $h_{\\mathrm{obs}}$"),
    (9.6, 11.6, 34.8, 19.2, BLUE, "#E7EEF5", "Steerable",
     "the write moves behaviour, direction-specifically"),
    (14.6, 14.8, 24.8, 12.8, BLUE, "#D6E3EF", "Correctable",
     "the movement lands on $g$"),
]
for x, y, w_, h_, ec, fc_, nm, sub in nest:
    S.box(ax, x, y, w_, h_, "", fc=fc_, ec=ec, lw=0.8, r=1.2)
    S.label(ax, x + w_ / 2, y + h_ - 1.9, nm, fs=T_LABEL, weight="bold", color=SLATE)
    S.label(ax, x + w_ / 2, y + h_ - 4.0, sub, fs=T_TINY, color=SLATE)
S.label(ax, 27.0, 6.9, "Correctable $\\subset$ Steerable $\\subset$ Readable — "
        "each stronger claim is a strict subset\nof the weaker one it presupposes",
        fs=T_TINY, color=SLATE, style="italic")

# the licensing gate — deliberately not a fourth ring
S.arrow(ax, (52.4, 20.0), (57.6, 20.0), color=SLATE)
S.label(ax, 55.0, 21.2, "assert?", fs=T_TINY, color=SLATE, bg=True)
S.panel(ax, 57.8, 3.0, 41.4, 33.4, "", ec=GREY)
S.label(ax, 59.8, 36.4, "LICENSABLE  —  an evidential decision, not a property",
        fs=T_TINY, weight="bold", color=SLATE, ha="left", bg=True)
crit = [("Specificity", "does the effect survive matched controls?"),
        ("Destination", "does gold arrival exceed other-wrong?"),
        ("Preservation", "is collateral damage bounded?"),
        ("Yield", "is the licensed set non-vacuous?"),
        ("Evidence", "do the finite-sample bounds hold?")]
yy = 31.4
for nm, q in crit:
    S.box(ax, 59.6, yy - 1.5, 12.4, 3.0, nm, fc=GREY_F, ec=GREY, fs=T_TINY, r=0.3)
    S.label(ax, 73.0, yy, q, fs=T_TINY, color=GREY, ha="left")
    yy -= 4.4
ax.plot([65.8, 65.8], [9.6, 30.0], color=GREY, lw=LW_HAIR, zorder=2)
S.arrow(ax, (65.8, 11.4), (65.8, 9.8), color=GREY, ms=2.2)
S.box(ax, 60.4, 6.0, 36.0, 3.8, "", ec=SLATE, lw=1.5, r=0.5)
for i, (nm, col) in enumerate((("ADMIT", TEAL), ("DECLINE", GREY), ("NO-GO", VERM))):
    S.label(ax, 66.4 + i * 12.0, 7.9, nm, fs=T_TINY, weight="bold", color=col)
S.label(ax, 78.4, 4.2, "DECLINE reports the state of the evidence — "
        "not that the setting is uncorrectable", fs=T_TINY, color=VERM, style="italic")

S.save(fig, "fig1_sakiko_core_B", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "candidates"))
print("B rendered")
