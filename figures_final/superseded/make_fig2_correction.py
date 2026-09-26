"""Figure 2 — Act 1: is the correction real, stable and structurally specific?
All values read from FINAL_PAPER_EVIDENCE.csv. Manifest: F2 panels A, B, C."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import *
import matplotlib.pyplot as plt, numpy as np

setup(); ev = load()
fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(7.2, 2.75),
                                    gridspec_kw={'width_ratios': [0.95, 1.35, 1.0]})

# ---------- A: correction is real and stable ----------
phi_nets = [int(x) for x in ev['phi_multiseed_nets'].split(';')]
q25_net = int(ev['qwen25_real_net'])
axA.scatter(phi_nets, [1] * len(phi_nets), s=26, color=REAL, zorder=3, clip_on=False)
axA.scatter([q25_net], [0], s=34, color=REAL, marker='D', zorder=3, clip_on=False)
axA.axvline(0, color=MUTED, linewidth=0.7, zorder=1)
axA.text(np.mean(phi_nets), 1.30, f"{min(phi_nets)} to {max(phi_nets)}  ·  5/5 seeds positive",
         ha='center', fontsize=6.4, color=MUTED)
axA.text(q25_net, 0.30, f"+{q25_net}   (92 fixed / 13 broken)", ha='center',
         fontsize=6.4, color=MUTED)
axA.text(-4, 1, "Phi-3.5", ha='right', va='center', fontsize=7.2, color=INK)
axA.text(-4, 0, "Qwen2.5-7B", ha='right', va='center', fontsize=7.2, color=INK)
axA.set_xlim(-32, 96); axA.set_ylim(-0.55, 1.75)
axA.set_yticks([]); axA.spines['left'].set_visible(False)
axA.set_xlabel("net corrections", fontsize=7.2)
axA.set_title("a   real and stable", fontsize=7.8, loc='left', pad=6)
provenance_tag(axA, "correction stage · locked tests", y=-0.30)

# ---------- B: structural specificity, both settings ----------
phi = [("real", triple(ev, 'phi_e2_real_locked')[2]),
       ("random", triple(ev, 'phi_e2_random_dir')[2]),
       ("reverse", triple(ev, 'phi_e2_reverse_dir')[2]),
       ("wrong layer", triple(ev, 'phi_e2_wrong_layer')[2]),
       ("mismatched", triple(ev, 'phi_e2_mismatched')[2])]
q25 = [("real", q25_net), ("reverse", int(ev['qwen25_reverse_net'])),
       ("random mean", float(ev['qwen25_random_mean_net']))]
xs = list(range(len(phi))) + [len(phi) + 0.9 + i for i in range(len(q25))]
vals = [v for _, v in phi] + [v for _, v in q25]
labs = [l for l, _ in phi] + [l for l, _ in q25]
cols = [REAL if l == "real" else CONTROL for l in labs]
axB.bar(xs, vals, color=cols, width=0.66, zorder=3)
axB.axhline(0, color=MUTED, linewidth=0.7, zorder=2)
for x, v in zip(xs, vals):
    axB.text(x, v + (2.2 if v >= 0 else -2.2), f"{v:+.0f}" if v == int(v) else f"{v:+.1f}",
             ha='center', va='bottom' if v >= 0 else 'top', fontsize=6.5, color=INK)
axB.set_xticks(xs); axB.set_xticklabels(labs, fontsize=6.0, rotation=38, ha='right',
                    rotation_mode='anchor')
axB.set_ylabel("net corrections", fontsize=7.2)
axB.set_ylim(-30, 100)
axB.text(2.0, 93, "Phi-3.5", ha='center', fontsize=6.8, color=MUTED, style='italic')
axB.text(6.9, 93, "Qwen2.5-7B", ha='center', fontsize=6.8, color=MUTED, style='italic')
axB.axvline(4.95, color=FAINT, linewidth=0.8, zorder=1)
axB.set_title("b   the effect needs the real direction and site",
              fontsize=7.8, loc='left', pad=6)
provenance_tag(axB, "correction stage · aggregate-only controls", y=-0.34)

# ---------- C: modern matched-random nulls ----------
mods = [("Qwen3-8B", 'q8b_gold_arrival', 'q8b_random_gold_max', 'q8b_matched_random_exceed'),
        ("Gemma-2-9b", 'gemma_gold_arrival', 'gemma_random_gold_max', 'gemma_matched_random_exceed')]
y = np.arange(len(mods))[::-1]
for yi, (name, kr, km, ke) in zip(y, mods):
    real, mx, exc = int(ev[kr]), int(ev[km]), int(ev[ke])
    axC.barh(yi + 0.16, real, color=REAL, height=0.30, zorder=3)
    axC.barh(yi - 0.20, max(mx, 0.4), color=FAINT, height=0.30, zorder=3)
    axC.text(real + 1.2, yi + 0.16, f"{real}", va='center', fontsize=7.2, color=INK, fontweight='bold')
    axC.text(max(mx, 0.4) + 1.2, yi - 0.20, f"max {mx}", va='center', fontsize=6.2, color=MUTED)
    axC.text(-1.5, yi + 0.42, name, ha='right', va='center', fontsize=7.0, color=INK)
    axC.text(-1.5, yi - 0.20, f"{exc}/59 exceed", ha='right', va='center',
             fontsize=6.0, color=MUTED, style='italic')
axC.set_xlim(0, 48); axC.set_ylim(-0.70, 1.72)
axC.set_yticks([]); axC.spines['left'].set_visible(False)
axC.set_xlabel("gold arrivals", fontsize=7.2)
axC.set_title("c   no matched random comes close", fontsize=7.8, loc='left', pad=6)
provenance_tag(axC, "verification stage · sealed · row-level", y=-0.30)

plt.tight_layout(w_pad=2.0)
out = 'figures_final/fig2_correction.png'
plt.savefig(out); plt.savefig(out.replace('.png', '.pdf'))
print(f"  wrote {out}")
print(f"    a: Phi {phi_nets}  Qwen2.5 +{q25_net}")
print(f"    b: Phi {[v for _,v in phi]} | Qwen2.5 {[v for _,v in q25]}")
print(f"    c: 8B {ev['q8b_gold_arrival']}/max {ev['q8b_random_gold_max']} · Gemma {ev['gemma_gold_arrival']}/max {ev['gemma_random_gold_max']}")
