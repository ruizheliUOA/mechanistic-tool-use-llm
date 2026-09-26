"""Figure D1 (Appendix D) — the historical record behind Sections 4 and 5.

  (a) dot plot   E1 collateral of every artifact-backed Phi-3.5 seed, against the 5%
                 bound, with each run's net gain
  (b) lollipop   the Qwen2.5-7B locked battery
Two panels only. Both historical; the Phi-3.5 control battery is Figure 3(a) and is
not repeated here."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import *
from _panels import load_all, draw_seeds, draw_q25_battery

theme_bw(); D = load_all()

FW, FH = TEXT_W, 2.30
fig = plt.figure(figsize=(FW, FH))
def axes_in(x, y, w, h):
    return fig.add_axes([x / FW, y / FH, w / FW, h / FH])

axa = axes_in(0.80, 0.66, 1.85, 1.22)
draw_seeds(axa, D)
panel_title(axa, '(a) Every realization exceeds the bound')
note(axa, 'Phi-3.5 · five seeds · historical', x=0.0, y=-0.34, ha='left')

axb = axes_in(3.72, 0.66, 1.60, 1.22)
draw_q25_battery(axb, D)
panel_title(axb, '(b) Real above every control')
note(axb, 'Qwen2.5-7B · locked battery · aggregate records', x=1.0, y=-0.34)

save(fig, 'figD1_historical')
for s in D['seeds']:
    print(f'    a: seed {s["seed"]:<5} broke {s["broke"]:>3}/{D["phi"]["e1_n"]} '
          f'= {int(s["broke"]) / D["phi"]["e1_n"]:.1%}  net +{s["net"]}')
q = D['q25']
print(f'    b: real +{q["real"]} · reversed +{q["rev"]} · random mean +{q["rmean"]} '
      f'(max {q["rmax"]})')
