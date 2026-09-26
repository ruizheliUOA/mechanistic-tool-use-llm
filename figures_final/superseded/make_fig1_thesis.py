"""Figure 1 (Section 3) — correction is not repair.

  (a) bar chart   one intervention, every outcome resolved      Phi-3.5, historical
  (b) forest      what the evidence licenses                    sealed, three settings
Two panels only. Values from final_evidence/FINAL_PAPER_EVIDENCE.csv and the frozen
verdict artifacts, loaded and asserted by `_panels.load_all()`."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import *
from _panels import load_all, draw_outcomes, draw_forest, FOREST, VERDICT

theme_bw(); D = load_all()

FW, FH = TEXT_W, 2.36
fig = plt.figure(figsize=(FW, FH))
def axes_in(x, y, w, h):
    return fig.add_axes([x / FW, y / FH, w / FW, h / FH])

axa = axes_in(1.16, 0.62, 1.50, 1.34)
draw_outcomes(axa, D)
panel_title(axa, '(a) One intervention, every outcome resolved')
note(axa, 'Phi-3.5 · historical protocol · row-level records', x=0.0, y=-0.30, ha='left')

axb = axes_in(3.92, 0.62, 1.45, 1.34)
draw_forest(axb, D)
panel_title(axb, '(b) What the evidence licenses')
note(axb, 'sealed protocol · criteria frozen before evaluation', x=1.0, y=-0.30)

save(fig, 'fig1_thesis')
p = D['phi']
print(f'    a: {p["exits"]} exits = {p["gold"]} arrivals + {p["oth"]} other wrong | '
      f'{p["expo"]} exposed = {p["brk"]} broken + {p["kept"]} kept | net +{p["net"]}')
ev = D['ev']
for name, pfx in FOREST:
    print(f'    b: {name:<11} {ev[pfx + "_target_hit"]} '
          f'[{ev[pfx + "_target_hit_ci_lo"]}, {ev[pfx + "_target_hit_ci_hi"]}] '
          f'{VERDICT[ev[pfx + "_formal_verdict"]]}')
