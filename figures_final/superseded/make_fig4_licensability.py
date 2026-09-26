"""Figure 5 (Section 5) — Correctable is not Licensable.

One panel: a tile map of the six properties and the frozen verdict for every evaluated
setting, sealed settings above historical ones. Fill carries the status and the label
the evidence behind it: for Steerable the random directions that reach the real effect,
for Correctable target-hit, for Preserving broken decisions among those the Router
fired on (E2). Intervals are in Figure 1(b) and are not repeated here."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import *
from _panels import load_all, VERDICT

theme_bw(); D = load_all()
ev, rows = D['ev'], D['rows']

H, N, P, X = 'holds', 'fails', 'partial', 'n/e'
YES = ('yes', H)
th = lambda p: (f'{num(ev, p + "_target_hit"):.3f}', H)
e2 = {name: f'{b}/{n}' for name, prov, (b, n), e1, vac in D['preservation']}
llama_max = max(int(x.split('/')[0]) for x in ev['llama_randoms_ge_real'].split(';'))
nulls = {d['name']: f'{d["ge_real"]}/59' for d in D['nulls']}

groups = [
 ('sealed', [
  ('Qwen3-8B',   [YES, YES, (nulls['Qwen3-8B'], H), th('q8b'), (e2['Qwen3-8B'], P)],
   VERDICT[ev['q8b_formal_verdict']]),
  ('Qwen3-4B',   [YES, YES, (nulls['Qwen3-4B'], H), th('q4b'), (e2['Qwen3-4B'], H)],
   VERDICT[ev['q4b_formal_verdict']]),
  ('Gemma-2-9b', [YES, YES, (nulls['Gemma-2-9b'], H), th('gemma'), (e2['Gemma-2-9b'], N)],
   VERDICT[ev['gemma_formal_verdict']])]),
 ('historical', [
  ('Phi-3.5',      [YES, YES, ('0/1', H), th('phi'), (e2['Phi-3.5'], N)], None),
  ('Qwen2.5-7B',   [YES, YES, ('0/10', H), ('n/a', X), ('–', X)], None),
  ('Llama-3.1-8B', [YES, YES, (f'{llama_max}/20', N), ('–', X), ('–', X)], None),
  ('Mistral-7B',   [YES, YES, (ev['mistral_randoms_ge_real'], N), ('–', X), ('–', X)], None)])]
props = ['Adjudicable', 'Readable', 'Steerable', 'Correctable', 'Preserving', 'Licensable']

FILL = {H: GREEN_L, P: PURPLE_L, N: ROSE_L, X: '#f4f4f4'}
TXT = {H: ARRIVE_TXT, P: REAL_TXT, N: BROKEN_TXT, X: '#a0a0a0'}
VFILL = {'ADMIT': GREEN, 'DECLINE': PURPLE_L, None: '#f4f4f4'}
VTXT = {'ADMIT': 'white', 'DECLINE': REAL_TXT, None: '#a0a0a0'}

FW, FH = TEXT_W, 2.42
fig = plt.figure(figsize=(FW, FH))
ax = fig.add_axes([0.95 / FW, 0.52 / FH, 4.35 / FW, 1.58 / FH])
gg(ax, x='discrete', y='discrete'); ax.grid(False)

rows_all = [(name, cells, verdict, prov) for prov, items in groups
            for name, cells, verdict in items]
GAP = 0.32                                   # space between the two protocol bands
ypos, y = {}, 0.0
for i, (name, cells, verdict, prov) in enumerate(reversed(rows_all)):
    ypos[name] = y
    y += 1 + (GAP if i == 3 else 0)          # a gap between historical and sealed
for name, cells, verdict, prov in rows_all:
    y = ypos[name]
    for j, (label, state) in enumerate(cells):
        ax.add_patch(Rectangle((j - 0.45, y - 0.40), 0.90, 0.80, facecolor=FILL[state],
                               edgecolor='white', linewidth=1.2, zorder=3))
        ax.text(j, y, label, ha='center', va='center', fontsize=0.82 * BASE,
                color=TXT[state], zorder=4)
    ax.add_patch(Rectangle((5 - 0.45, y - 0.40), 0.90, 0.80, facecolor=VFILL[verdict],
                           edgecolor='white', linewidth=1.2, zorder=3))
    ax.text(5, y, verdict or 'not sealed', ha='center', va='center', fontsize=0.82 * BASE,
            color=VTXT[verdict], fontweight='bold' if verdict else 'normal', zorder=4)
ax.set_xticks(range(6)); ax.set_xticklabels(props, fontsize=0.82 * BASE)
ax.set_yticks([ypos[n] for n, *_ in rows_all])
ax.set_yticklabels([n for n, *_ in rows_all])
ax.set_xlim(-0.62, 5.62); ax.set_ylim(-0.62, max(ypos.values()) + 0.62)
for prov, items in groups:
    ys = [ypos[n] for n, *_ in items]
    ax.text(5.70, sum(ys) / len(ys), prov, ha='left', va='center', fontsize=0.82 * BASE,
            color=GREY30, rotation=-90)
panel_title(ax, 'Where each evaluated setting stops')
for x0, c, t in zip([0.0, 0.17, 0.33, 0.47], [GREEN_L, PURPLE_L, ROSE_L, '#f4f4f4'],
                    ['holds', 'partial', 'fails', 'not evaluated']):
    key(ax, x0, -0.30, c, t, box=5.4, size=0.8 * BASE,
        ec='#d5d5d5' if c == '#f4f4f4' else 'none')

save(fig, 'fig4_licensability')
for name, cells, verdict, prov in rows_all:
    print(f'    {prov:<10} {name:<13} ' + ' | '.join(f'{l}' for l, s in cells)
          + f'  -> {verdict}')
