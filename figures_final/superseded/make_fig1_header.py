"""Figure 1 (Introduction header) — one intervention, three views.
Every number recomputed from p0_final_test_details.jsonl at render time.
Manifest: F1-header."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import *
import matplotlib.pyplot as plt, numpy as np, json
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

setup(); ev = load()

# ---- recompute straight from row level so the figure cannot drift ----
rows = [json.loads(l) for l in open('final/results/clean/p0_final_test_details.jsonl')]
fired = [r for r in rows if r['route'] != 'none']
err   = [r for r in fired if r['error_type'] != 'correct']
exits = [r for r in err if r['int_pred'] != r['clean_pred']]
gold  = sum(1 for r in exits if r['int_pred'] == r['gold'])
other = len(exits) - gold
ec    = [r for r in fired if r['error_type'] == 'correct']
brk   = sum(1 for r in ec if r['int_pred'] != r['gold'])
kept  = len(ec) - brk
net   = int(ev['phi_net'])
assert (len(exits), gold, other, len(ec), brk) == (153, 107, 46, 93, 52)

fig, ax = plt.subplots(figsize=(7.2, 3.35))
ax.set_xlim(0, 100); ax.set_ylim(-4, 100); ax.axis('off')

GOLD_C, BAD_C, KEEP_C = REAL, DAMAGE, '#d8d5cf'
TOP, BASE = 78, 20                      # panel title band / dot-grid baseline

def panel_head(x, w, n, kicker, sub):
    ax.text(x, TOP + 12, n, fontsize=6.4, color=MUTED, fontweight='bold')
    ax.text(x, TOP + 6.0, kicker, fontsize=8.6, color=INK, va='baseline')
    ax.text(x, TOP + 1.0, sub, fontsize=6.3, color=MUTED, va='baseline', style='italic')

def dots(x0, y0, vals, cols, ncol=17, pitch=1.62, size=11):
    xs, ys, cs = [], [], []
    i = 0
    for v, c in zip(vals, cols):
        for _ in range(v):
            xs.append(x0 + (i % ncol) * pitch)
            ys.append(y0 - (i // ncol) * pitch * 2.15)   # compensate axis aspect
            cs.append(c); i += 1
    ax.scatter(xs, ys, s=size, c=cs, marker='o', linewidths=0, zorder=3)

# ══ 1 · the aggregate ═════════════════════════════════════════════════════
X1 = 0
panel_head(X1, 20, "1", "What the number says", "aggregate accuracy")
ax.text(X1 + 1.5, 46, f"+{net}", fontsize=30, color=GOLD_C, fontweight='bold', va='center')
ax.text(X1 + 1.5, 33, "net corrections", fontsize=7.0, color=MUTED)
ax.text(X1 + 1.5, 27.5, "on a held-out test set", fontsize=6.2, color=MUTED, style='italic')
ax.text(X1 + 1.5, 12, "reads as success", fontsize=7.2, color=GOLD_C, style='italic')

# ══ 2 · where the corrections went ════════════════════════════════════════
X2 = 30
panel_head(X2, 30, "2", "Where they went", f"{len(exits)} decisions left the error")
dots(X2 + 1.0, 66, [gold, other], [GOLD_C, BAD_C], ncol=18, pitch=1.55, size=10)
ax.text(X2 + 1.0, 30, f"{gold}", fontsize=13, color=GOLD_C, fontweight='bold', va='center')
ax.text(X2 + 9.5, 30.6, "reached the\nrequired action", fontsize=6.2, color=MUTED,
        va='center', linespacing=1.4)
ax.text(X2 + 1.0, 20, f"{other}", fontsize=13, color=BAD_C, fontweight='bold', va='center')
ax.text(X2 + 9.5, 20.6, "landed on a\ndifferent error", fontsize=6.2, color=MUTED,
        va='center', linespacing=1.4)
ax.text(X2 + 1.0, 8.5, "movement is not arrival", fontsize=7.2, color=BAD_C, style='italic')

# ══ 3 · what it cost ══════════════════════════════════════════════════════
X3 = 63
panel_head(X3, 24, "3", "What it cost", f"{len(ec)} already-correct decisions it touched")
dots(X3 + 1.0, 66, [kept, brk], [KEEP_C, BAD_C], ncol=13, pitch=1.72, size=13)
ax.text(X3 + 1.0, 30, f"{kept}", fontsize=13, color=MUTED, fontweight='bold', va='center')
ax.text(X3 + 8.5, 30.6, "survived", fontsize=6.2, color=MUTED, va='center')
ax.text(X3 + 1.0, 20, f"{brk}", fontsize=13, color=BAD_C, fontweight='bold', va='center')
ax.text(X3 + 8.5, 20.6, "were broken", fontsize=6.2, color=BAD_C, va='center',
        fontweight='bold')
ax.text(X3 + 1.0, 8.5, "most of what it touched", fontsize=7.2, color=BAD_C,
        style='italic')

# ══ connective tissue ═════════════════════════════════════════════════════
for xa, xb in [(26.0, 29.0), (59.5, 62.5)]:
    ax.add_patch(FancyArrowPatch((xa, 50), (xb, 50), arrowstyle='-|>', mutation_scale=10,
                 linewidth=1.1, color=FAINT, zorder=2))
ax.plot([0, 87], [TOP + 16.5]*2, color=FAINT, linewidth=0.8)
ax.text(0, TOP + 18.5, "ONE INTERVENTION,  THREE VIEWS   ·   Phi-3.5, frozen locked test",
        fontsize=6.6, color=MUTED, fontweight='bold')

# ══ the ladder ════════════════════════════════════════════════════════════
LX = 90.5
rungs = [("Adjudicable", "y"), ("Readable", "y"), ("Steerable", "y"),
         ("Correctable", "y"), ("Preserving", "n"), ("Licensable", "-")]
ax.text(LX - 1.5, TOP + 6.0, "what the\nevidence licenses", fontsize=6.6, color=INK,
        va='baseline', linespacing=1.4)
ly = 62
ax.plot([LX, LX], [ly + 1.6, ly - 5*8.2 - 1.6], color=FAINT, linewidth=1.0, zorder=1)
for name, st in rungs:
    c = {"y": GOLD_C, "n": BAD_C, "-": FAINT}[st]
    ax.scatter([LX], [ly], s=46, c=c, marker='o', edgecolors='white',
               linewidths=0.9, zorder=4)
    ax.text(LX + 3.2, ly, name, fontsize=6.5, va='center',
            color=BAD_C if st == "n" else INK,
            fontweight='bold' if st == "n" else 'normal')
    ly -= 8.2
ax.text(LX - 1.5, 15.0, "Steerable and Correctable —\nbut not Preserving.",
        fontsize=6.4, color=MUTED, linespacing=1.5, va='top')

out = 'figures_final/fig1_header.png'
plt.savefig(out); plt.savefig(out.replace('.png', '.pdf'))
print(f"  wrote {out}")
print(f"    verified from row level: exits {len(exits)} = {gold} gold + {other} other-wrong")
print(f"                             exposed-correct {len(ec)} = {kept} kept + {brk} broken")
