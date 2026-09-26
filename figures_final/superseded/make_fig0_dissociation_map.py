"""Figure 0 — Introduction teaser: the four questions an intervention must pass.
A single changed action is followed through four distinct questions; each one is a
place where "it worked" can fail, annotated with the observed evidence.
Manifest: F0. Values from FINAL_PAPER_EVIDENCE.csv."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import *
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

setup(); ev = load()
fig, ax = plt.subplots(figsize=(7.2, 3.05))
ax.set_xlim(0, 100); ax.set_ylim(0, 44); ax.axis('off')

SPINE_Y = 30.0
gates = [
    ("improved?", "behavioural\nimprovement", None, None),
    ("arrived?", "destination\ncorrectness",
     f"{ev['phi_other_wrong']} of {ev['phi_source_exits']} exits reach\na third wrong action", "Phi-3.5"),
    ("preserved?", "preservation",
     f"{ev['phi_broken_e2']} of {ev['phi_exposed_correct']} correct rows\nare broken", "Phi-3.5"),
    ("established?", "evidential\nsufficiency",
     f"target-hit {ev['q4b_target_hit']}, {ev['gemma_target_hit']}\nformally declined", "Qwen3-4B · Gemma-2-9b"),
]
GW, X0, GAP = 14.0, 13.0, 5.5

ax.text(0.5, SPINE_Y, "an intervention\nchanges\nan action", ha='left', va='center',
        fontsize=6.2, color=REAL, fontweight='bold', linespacing=1.5, zorder=4)

for i, (q, prop, drop, who) in enumerate(gates):
    x = X0 + i * (GW + GAP)
    if i:
        ax.add_patch(FancyArrowPatch((x - GAP + 0.6, SPINE_Y), (x - 0.6, SPINE_Y),
                     arrowstyle='-|>', mutation_scale=9, linewidth=1.1, color=MUTED, zorder=2))
    ax.add_patch(FancyBboxPatch((x, SPINE_Y - 4.6), GW, 9.2,
                 boxstyle="round,pad=0.3,rounding_size=0.8", linewidth=1.0,
                 edgecolor=MUTED, facecolor='white', zorder=3))
    ax.text(x + GW/2, SPINE_Y + 2.0, q, ha='center', va='center', fontsize=8.0,
            color=INK, zorder=4)
    ax.text(x + GW/2, SPINE_Y - 2.4, prop, ha='center', va='center', fontsize=5.9,
            color=MUTED, linespacing=1.35, zorder=4)
    if drop:
        ax.add_patch(FancyArrowPatch((x + GW/2, SPINE_Y - 5.0), (x + GW/2, SPINE_Y - 11.2),
                     arrowstyle='-|>', mutation_scale=8, linewidth=0.9,
                     color=DAMAGE, linestyle=(0, (2.5, 1.8)), zorder=2))
        ax.text(x + GW/2, SPINE_Y - 12.4, drop, ha='center', va='top', fontsize=5.8,
                color=DAMAGE, linespacing=1.45, zorder=4)
        ax.text(x + GW/2, SPINE_Y - 19.2, who, ha='center', va='top', fontsize=5.3,
                color=MUTED, style='italic', zorder=4)

xe = X0 + 4 * (GW + GAP)
ax.add_patch(FancyArrowPatch((xe - GAP + 0.6, SPINE_Y), (xe - 0.6, SPINE_Y),
             arrowstyle='-|>', mutation_scale=9, linewidth=1.1, color=MUTED, zorder=2))
ax.text(xe + 0.3, SPINE_Y, "a licensed\nrepair claim", ha='left', va='center',
        fontsize=6.4, color=REAL, fontweight='bold', linespacing=1.4, zorder=4)

ax.text(0.5, 41.5, "Four questions, not one. An intervention that improves behaviour "
        "may still fail at any later question.", fontsize=8.0, color=INK, va='top')
ax.text(0.5, 5.2, "observed in this work", fontsize=5.8, color=DAMAGE, style='italic')
out = 'figures_final/fig0_dissociation_map.png'
plt.savefig(out); plt.savefig(out.replace('.png', '.pdf'))
print(f"  wrote {out}")
for q, p, d, w in gates:
    print(f"    {q:<14} {(d or '(passes)').replace(chr(10),' ')}")
