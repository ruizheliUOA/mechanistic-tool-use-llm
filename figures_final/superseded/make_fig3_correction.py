"""Figure 3 (Section 4) — correction is direction-specific.

  (a) bar chart    Phi-3.5 structural controls                   historical, locked test
  (b) strip plot   the real direction against 59 matched random directions,
                   in each sealed setting                        sealed
Two panels only; historical and sealed evidence are never drawn on a common scale."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import *
from _panels import load_all, draw_phi_controls, draw_nulls

theme_bw(); D = load_all()

FW, FH = TEXT_W, 2.42
fig = plt.figure(figsize=(FW, FH))
def axes_in(x, y, w, h):
    return fig.add_axes([x / FW, y / FH, w / FW, h / FH])

axa = axes_in(1.06, 0.68, 1.50, 1.28)
draw_phi_controls(axa, D)
panel_title(axa, '(a) Corrupted variants do not reproduce it')
note(axa, 'Phi-3.5 · historical · locked test', x=0.0, y=-0.34, ha='left')

axb = axes_in(3.62, 0.68, 1.72, 1.28)
draw_nulls(axb, D)
panel_title(axb, '(b) No matched random direction reaches it')
note(axb, 'sealed · add-one $p$ = 0.017 in each setting', x=1.0, y=-0.34)

save(fig, 'fig3_correction')
for lab, k in D['phi_arms']:
    print(f'    a: {lab:<19} {D["ev"][k]}')
for d in D['nulls']:
    print(f'    b: {d["name"]:<11} real target gain {d["tgr"]:.4f}; '
          f'random max {d["rmax"]:.4f}; {d["ge_real"]} of 59 ≥ real')
