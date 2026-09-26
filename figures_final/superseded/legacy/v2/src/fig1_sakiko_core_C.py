"""Figure 1 candidate C — claim ladder with mechanism and destination insets.

The most restrained option: the ladder READ / CONTROL / REPAIR carries the
composition, Adjudicability is drawn as the ground the ladder stands on, and
Licensability as a confidence bracket held above the final rung rather than as
a further rung. Mechanism and destination geometry are demoted to insets.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sakiko_v2_style import *   # noqa: F403
import sakiko_v2_style as S

S.apply()
fig, ax = S.canvas(7.2, 3.85, (0, 100), (0, 52))

# ==================================================== (a) MECHANISM INSET
S.label(ax, 1.4, 50.4, "(a)", fs=T_PL, weight="bold", ha="left")
S.label(ax, 5.0, 50.4, "conditional intervention in a frozen model",
        fs=T_PANEL, weight="bold", color=SLATE, ha="left")
S.box(ax, 1.4, 39.6, 8.0, 5.2, "request\n+ tools", fs=T_TINY)
S.arrow(ax, (9.6, 42.2), (11.2, 42.2))
n, gap, x0, w = 10, 0.4, 11.4, 18.0
bw = (w - gap * (n - 1)) / n
cx = []
for i in range(n):
    x = x0 + i * (bw + gap)
    hit = i in (3, 6)
    S.box(ax, x, 39.4, bw, 5.6, fc=BLUE_F if hit else SLATE_F,
          ec=BLUE if hit else SLATE, lw=1.0 if hit else 0.6, r=0.3)
    cx.append(x + bw / 2)
S.snowflake(ax, 14.4, 46.6, s=0.7, color=SLATE)
S.label(ax, 15.6, 46.6, r"$\theta$ frozen", fs=T_TINY, color=SLATE, ha="left")
S.label(ax, cx[3]+1.2, 38.0, "$L_{\\mathrm{inj}}$", fs=T_TINY, color=BLUE, weight="bold", ha="left")
S.label(ax, cx[6]+1.2, 38.0, "$L_{\\mathrm{obs}}$", fs=T_TINY, color=BLUE, weight="bold", ha="left")
ax.plot([cx[6], cx[6]], [39.3, 36.2], color=BLUE, lw=LW_ARROW, zorder=4)
S.arrow(ax, (cx[6], 36.2), (25.8, 36.2), color=BLUE)
S.box(ax, 26.0, 34.6, 10.4, 3.2, "$r_c(h_{\\mathrm{obs}})\\!\\gtrless\\!\\tau$",
      fc=BLUE_F, ec=BLUE, fs=T_TINY)
ax.plot([31.2, 31.2], [34.5, 32.4], color=BLUE, lw=LW_ARROW, zorder=4)
ax.plot([cx[3], 31.2], [32.4, 32.4], color=BLUE, lw=LW_ARROW, zorder=4)
S.arrow(ax, (cx[3], 32.4), (cx[3], 39.3), color=BLUE)
S.label(ax, 24.0, 31.4, "if the router fires, re-run the same input writing "
        "$h'=h+\\alpha d_c$ at $L_{\\mathrm{inj}}<L_{\\mathrm{obs}}$",
        fs=T_TINY, color=BLUE)
S.arrow(ax, (29.8, 42.2), (31.4, 42.2))
S.box(ax, 31.6, 39.8, 7.4, 4.8, "$\\hat a'$", fs=T_MATH)

# ==================================================== destination inset
S.label(ax, 43.0, 50.4, "(b)", fs=T_PL, weight="bold", ha="left")
S.label(ax, 46.6, 50.4, "movement is not correction",
        fs=T_PANEL, weight="bold", color=SLATE, ha="left")
S.arrow(ax, (39.2, 42.2), (44.2, 42.2))
ax.add_patch(__import__("matplotlib.patches", fromlist=["Circle"])
             .Circle((44.4, 42.2), 0.32, fc=INK, ec="none", zorder=6))
S.label(ax, 41.6, 43.4, "$c=(g\\rightarrow s)$", fs=T_TINY, color=GREY)
for yy, col, fc_, nm, ms, vd in (
        (46.6, GREY, GREY_F, "SOURCE_RETAINED", "$\\hat a'=s$", "still wrong"),
        (42.2, TEAL, TEAL_F, "GOLD_ARRIVAL", "$\\hat a'=g$", "correct"),
        (37.8, VERM, VERM_F, "OTHER_WRONG", "$\\hat a'=w\\notin\\{g,s\\}$", "still wrong")):
    S.box(ax, 46.4, yy - 1.6, 24.6, 3.2, "", fc=fc_, ec=col, lw=0.7, r=0.4)
    S.label(ax, 47.3, yy + 0.45, nm, fs=T_TINY, color=col, weight="bold", ha="left")
    S.label(ax, 47.3, yy - 0.95, ms, fs=T_TINY, ha="left")
    S.arrow(ax, (44.6, 42.2), (46.3, yy), color=col, lw=0.7)
    ax.plot([71.4, 72.4], [yy, yy], color=col, lw=LW_HAIR, zorder=4)
    S.label(ax, 72.9, yy, vd, fs=T_TINY, color=col, ha="left",
            weight="bold" if col is TEAL else "normal")
S.label(ax, 84.6, 48.6, "under conventional\naccuracy", fs=T_TINY, color=GREY,
        style="italic")
S.label(ax, 59.0, 33.6, "conventional accuracy collapses the two wrong outcomes;\n"
        "source exit  =  gold arrival  +  other-wrong", fs=T_TINY)

ax.plot([1.0, 99.0], [30.0, 30.0], color=HAIR, lw=LW_HAIR, zorder=1)

# ==================================================== (c) CLAIM LADDER
S.label(ax, 1.4, 28.0, "(c)", fs=T_PL, weight="bold", ha="left")
S.label(ax, 5.0, 28.0, "the claim ladder: what was shown, and what may be asserted",
        fs=T_PANEL, weight="bold", color=SLATE, ha="left")

rungs = [("READ", "Readable", "the error is linearly\ndecodable from $h_{\\mathrm{obs}}$", 17.0),
         ("CONTROL", "Steerable", "the write moves behaviour,\ndirection-specifically", 42.0),
         ("REPAIR", "Correctable", "the movement lands\non the gold action", 67.0)]
for i, (cap, formal, sub, xx) in enumerate(rungs):
    S.box(ax, xx, 17.6, 21.0, 6.4, "", fc=BLUE_F if i == 2 else PAPER,
          ec=BLUE, lw=1.0 if i == 2 else 0.7, r=0.5)
    S.label(ax, xx + 10.5, 21.9, cap, fs=T_PANEL, weight="bold", color=SLATE)
    S.label(ax, xx + 10.5, 19.4, formal, fs=T_LABEL, color=BLUE)
    S.label(ax, xx + 10.5, 15.4, sub, fs=T_TINY, color=GREY)
    if i:
        S.arrow(ax, (xx - 4.0, 20.8), (xx - 0.2, 20.8), color=BLUE)

# adjudicability: the ground the ladder stands on
S.box(ax, 12.0, 9.2, 76.0, 3.2, "", fc=GREY_F, ec=GREY, lw=0.7, r=0.4)
S.label(ax, 50.0, 10.8, "ADJUDICABLE   —   entry condition: $|A_D|\\geq 3$, a supported "
        "channel, sufficient rows in every split", fs=T_TINY, weight="bold", color=SLATE)
for xx in (27.5, 52.5, 77.5):
    ax.plot([xx, xx], [12.4, 13.2], color=GREY, lw=0.6, zorder=2)
S.label(ax, 50.0, 7.4, "without it, no rung on the ladder can be evaluated at all",
        fs=T_TINY, color=GREY, style="italic")

# licensability: a bracket held above the final rung, not a fourth rung
S.bracket(ax, 67.0, 88.0, 24.6, depth=1.6, color=SLATE, lw=1.1)
S.label(ax, 77.5, 27.6, "LICENSABLE", fs=T_LABEL, weight="bold", color=SLATE)
ax.plot([88.0, 92.0], [26.2, 26.2], color=SLATE, lw=LW_HAIR, zorder=4)
S.label(ax, 92.6, 26.2, "ADMIT\nDECLINE\nNO-GO", fs=T_TINY, ha="left")
S.label(ax, 41.6, 43.4, "", fs=T_TINY)
S.label(ax, 77.5, 28.0, "an evidential decision about the final rung — not a further rung",
        fs=T_TINY, color=SLATE, style="italic")
crit = ["Specificity", "Destination", "Preservation", "Yield", "Evidence"]
cx2 = 13.0
for nm in crit:
    S.box(ax, cx2, 1.4, 12.4, 3.0, nm, fc=GREY_F, ec=GREY, fs=T_TINY, r=0.3)
    cx2 += 13.2
ax.plot([19.2, 71.6], [5.6, 5.6], color=GREY, lw=LW_HAIR, zorder=2)
for i in range(5):
    ax.plot([19.2 + i * 13.2, 19.2 + i * 13.2], [4.4, 5.6], color=GREY, lw=0.6, zorder=2)
ax.plot([71.6, 77.5], [5.6, 5.6], color=GREY, lw=LW_HAIR, zorder=2)
ax.plot([77.5, 77.5], [5.6, 8.0], color=GREY, lw=LW_HAIR, zorder=2)
S.arrow(ax, (77.5, 13.4), (77.5, 17.5), color=GREY, ms=2.2, ls=(0, (2, 1.6)))
S.label(ax, 5.6, 2.9, "EVIDENTIAL\nQUALIFICATION", fs=T_TINY, weight="bold", color=GREY)
S.label(ax, 5.0, 26.2, "DECLINE reports the state of the evidence, "
        "not that the setting is uncorrectable", fs=T_TINY, color=VERM, ha="left")

S.save(fig, "fig1_sakiko_core_C", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "candidates"))
print("C rendered")
