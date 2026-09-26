"""Figure 4 (Section 5) — destination correctness and preservation are separate.

  (a) dumbbell   two corrections of the same size, landing differently  Qwen3-8B, sealed
  (b) dot plot   collateral on both denominators, against the 5% bound  all settings with
                 an exposure record, protocol labelled per row
Two panels only. The intervals that decide the licensing outcome are in Figure 1(b);
they are not repeated here."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import *
from _panels import load_all, draw_dumbbell, draw_preservation

theme_bw(); D = load_all()

FW, FH = TEXT_W, 2.46
fig = plt.figure(figsize=(FW, FH))
def axes_in(x, y, w, h):
    return fig.add_axes([x / FW, y / FH, w / FW, h / FH])

axa = axes_in(1.06, 0.70, 1.56, 1.28)
draw_dumbbell(axa, D)
panel_title(axa, '(a) Similar gain, different destination')
note(axa, 'Qwen3-8B · sealed · descriptive comparison', x=0.0, y=-0.32, ha='left')

axb = axes_in(3.78, 0.70, 1.58, 1.28)
draw_preservation(axb, D)
panel_title(axb, '(b) Preservation, on both denominators')
note(axb, 'E2 $\\geq$ E1 by construction', x=1.0, y=-0.32)

save(fig, 'fig_chain')
for lab, a, s in D['arms']['metrics']:
    print(f'    a: {lab:<18} activation {a} · score-space {s}')
for name, prov, e2, e1, vac in D['preservation']:
    print(f'    b: {name:<11} {prov:<10} E2 {e2[0]}/{e2[1]}'
          f'{" (vacuous)" if vac else ""} · E1 {e1[0]}/{e1[1]}')
