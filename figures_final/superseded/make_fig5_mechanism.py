"""Figure 5 — why the stages are separated.

  a  small multiples (facet_wrap, free y) over the layer sweep: detection stays flat
     while the recovered direction and the correction change      Qwen2.5-7B, historical
  b  lollipop: the estimator changes the locked-test net           Qwen2.5-7B, historical
  c  slope chart in two facets, ungated -> gated: precision rises in every comparable
     setting, net gain does not; line type marks the protocol       mixed, labelled
Values from final_evidence/FINAL_PAPER_EVIDENCE.csv. The Router AUC is recorded only as
a single value flat across layers, so it is drawn as a dashed reference, not as points."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import *
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator, LogLocator, NullFormatter

theme_grey(); ev = load()

FW, FH = TEXT_W, 3.56
fig = plt.figure(figsize=(FW, FH))
def axes_in(x, y, w, h):
    return fig.add_axes([x / FW, y / FH, w / FW, h / FH])

def head(ax, letter, title, dx=-26):
    tag(ax, letter, title, dx=dx, strip_above=True)

# ═════════ (a) readable is not steerable ═════════
L = [int(x) for x in ev['mech_ca_direct_layers'].split(';')]
nets = [int(x) for x in ev['mech_ca_direct_best_val_net'].split(';')]
norms = [float(x) for x in ev['mech_ca_direct_dm_norm'].split(';')]
auc = num(ev, 'mech_ca_direct_auc_flat')
cos16 = ev['mech_ca_direct_cos_dm_pc1_l16']

YA, HA, WA, XA0, SP = 2.40, 0.78, 1.40, 0.44, 0.30
facets = []
for k in range(3):
    ax = axes_in(XA0 + k * (WA + SP), YA, WA, HA)
    gg(ax, x='continuous', y='continuous')
    ax.set_xlim(15.4, 22.6)
    ax.xaxis.set_major_locator(FixedLocator(L)); ax.xaxis.set_minor_locator(FixedLocator([17, 19, 21]))
    facets.append(ax)
axA1, axA2, axA3 = facets
axA1.plot([L[0], L[-1]], [auc, auc], color=GREEN, lw=1.3, ls=(0, (4, 2)))
axA1.set_ylim(0.5, 1.02)
axA1.yaxis.set_major_locator(FixedLocator([0.5, 0.75, 1.0]))
axA1.text(19, auc - 0.07, f'≈ {auc:.2f} at every layer', ha='center', va='top',
          fontsize=0.72 * BASE, color=GREEN_TXT)
axA2.plot(L, norms, color=GREY30, lw=1.0, zorder=3)
axA2.scatter(L, norms, s=16, color=GREY30, zorder=4, linewidths=0)
axA2.set_ylim(0, 17.5); axA2.yaxis.set_major_locator(FixedLocator([0, 5, 10, 15]))
axA2.annotate(f'{norms[0]}', xy=(L[0], norms[0]), xytext=(5, -1), textcoords='offset points',
              fontsize=0.72 * BASE, color=GREY30, va='top', ha='left')
axA2.annotate(f'{norms[-1]}', xy=(L[-1], norms[-1]), xytext=(-5, 1), textcoords='offset points',
              fontsize=0.72 * BASE, color=GREY30, va='bottom', ha='right')
axA3.plot(L, nets, color=ARRIVE, lw=1.0, zorder=3)
axA3.scatter(L, nets, s=16, color=ARRIVE, zorder=4, linewidths=0)
axA3.set_ylim(0, 56); axA3.yaxis.set_major_locator(FixedLocator([0, 20, 40]))
for x, v in zip(L, nets):
    axA3.annotate(f'+{v}', xy=(x, v), xytext=(0, 4), textcoords='offset points', ha='center',
                  va='bottom', fontsize=0.72 * BASE, color=ARRIVE_TXT)
axA3.annotate(f'cos with PC1 = {cos16}', xy=(L[0], nets[0]), xytext=(L[0] + 0.1, 4.5),
              textcoords='data', fontsize=0.66 * BASE, color=BROKEN_TXT, ha='left', va='center')
for ax, s in zip(facets, ['Router AUC', 'direction norm', 'best validation net']):
    strip(ax, s)
axA2.set_xlabel('observation and injection layer', fontsize=0.9 * BASE)
head(axA1, 'a', 'Readable is not steerable')
note(axA3, 'Qwen2.5-7B · cannot answer → direct answer · historical, development data',
     x=1.0, y=1.0 + (STRIP_PT + 4.5) / 72 / HA, ha='right', va='bottom')

# ═════════ (b) the estimator matters for this channel ═════════
est = ev['mech_estimator_pca1_vs_diffmean'].replace('+', '').split(' vs ')
pca, dm = int(est[0]), int(est[1])
YB = 0.40
HB = (1 * 0.27 + 0.16) + 0.06 + (2 * 0.27 + 0.16)   # match panel c's two facet rows
axB = axes_in(0.96, YB, 1.02, HB)
gg(axB, x='continuous', y='discrete')
for y, (lab, v) in enumerate([('difference\nin means', dm), ('PCA-1', pca)]):
    axB.plot([0, v], [y, y], color=ARRIVE, lw=1.4, zorder=3)
    axB.scatter([v], [y], s=34, color=ARRIVE, zorder=4, linewidths=0)
    axB.annotate(f'+{v}', xy=(v, y), xytext=(0, 6), textcoords='offset points', ha='center',
                 va='bottom', fontsize=0.8 * BASE, color=ARRIVE_TXT, fontweight='bold')
axB.set_yticks([0, 1]); axB.set_yticklabels(['difference\nin means', 'PCA-1'], linespacing=1.1)
axB.set_ylim(-0.6, 1.75); axB.set_xlim(0, 60)
axB.xaxis.set_major_locator(FixedLocator([0, 20, 40, 60]))
axB.set_xlabel('locked-test net gain', fontsize=0.9 * BASE)
strip(axB, 'cannot answer → tool call')
head(axB, 'b', 'Estimator choice', dx=-62)

# ═════════ (c) gating buys precision, not always net ═════════
names = ['Qwen2.5-7B', 'Qwen3-8B', 'Gemma-2-9b']
g = ev['gating_precision_gated'].split(';'); u = ev['gating_precision_ungated'].split(';')
prec_g = [np.inf if x == 'inf' else float(x) for x in g]; prec_u = [float(x) for x in u]
net_g = [int(ev['qwen25_real_net']), int(ev['q8b_gated_net']), int(ev['gemma_gated_net'])]
net_u = [int(ev['qwen25_ungated_net']), int(ev['q8b_ungated_net']), int(ev['gemma_ungated_net'])]
f25, b25 = (int(x) for x in ev['qwen25_real_fixed_broke'].split('/'))
fu25, bu25 = (int(x) for x in ev['qwen25_ungated_fixed_broke'].split('/'))
assert abs(f25 / b25 - prec_g[0]) < 0.005 and abs(fu25 / bu25 - prec_u[0]) < 0.005
LS = {'Qwen2.5-7B': (0, (3, 1.6)), 'Qwen3-8B': 'solid', 'Gemma-2-9b': 'solid'}

PROT = {'Qwen2.5-7B': 'historical', 'Qwen3-8B': 'sealed', 'Gemma-2-9b': 'sealed'}
XC1, WC, GAPC, RH = 3.16, 1.02, 0.12, 0.27
XC2 = XC1 + WC + GAPC
TOP = 40
groups = [('sealed', ['Qwen3-8B', 'Gemma-2-9b']), ('historical', ['Qwen2.5-7B'])]
vals = {nm: (pu, pg, nu, ng) for nm, pu, pg, nu, ng in zip(names, prec_u, prec_g, net_u, net_g)}

def arrow(ax, x0, x1, y):
    ax.annotate('', xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle='-|>', color=GREY30, lw=0.9, mutation_scale=6,
                                shrinkA=3.2, shrinkB=3.2), zorder=3)

yc = YB
panels = []
for prov, members in reversed(groups):
    h = len(members) * RH + 0.16
    a1 = axes_in(XC1, yc, WC, h); gg(a1, x='continuous', y='discrete')
    a2 = axes_in(XC2, yc, WC, h); gg(a2, x='continuous', y='discrete')
    a1.set_xscale('log'); a1.set_xlim(1, TOP)
    a1.xaxis.set_major_locator(FixedLocator([1, 3, 10, 30]))
    a1.xaxis.set_minor_locator(FixedLocator([2, 5, 20]))
    a1.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:g}'))
    a1.xaxis.set_minor_formatter(NullFormatter())
    a2.set_xlim(0, 105); a2.xaxis.set_major_locator(FixedLocator([0, 25, 50, 75, 100]))
    for ax in (a1, a2):
        ax.set_ylim(-0.5 - 0.08 / RH, len(members) - 0.5 + 0.08 / RH)
    for i, nm in enumerate(members):
        y = len(members) - 1 - i
        pu, pg, nu, ng = vals[nm]
        top = np.isinf(pg)
        xg = TOP * 0.97 if top else pg
        arrow(a1, pu, xg, y)
        a1.scatter([pu], [y], s=22, color=CYAN, zorder=4, linewidths=0)
        a1.annotate(f'{pu:g}', xy=(pu, y), xytext=(0, -4.0), textcoords='offset points',
                    ha='center', va='top', fontsize=0.66 * BASE, color=CYAN_TXT)
        if top:
            a1.annotate('no broken decision', xy=(TOP * 0.97, y), xytext=(-2, 4.2),
                        textcoords='offset points', ha='right', va='bottom',
                        fontsize=0.66 * BASE, color=ARRIVE_TXT)
        else:
            a1.scatter([pg], [y], s=22, color=ARRIVE, zorder=4, linewidths=0)
            a1.annotate(f'{pg:.2f}'.rstrip('0').rstrip('.'), xy=(pg, y), xytext=(0, 4.2),
                        textcoords='offset points', ha='center', va='bottom',
                        fontsize=0.66 * BASE, color=ARRIVE_TXT)
        arrow(a2, nu, ng, y)
        a2.scatter([nu], [y], s=22, color=CYAN, zorder=4, linewidths=0)
        a2.scatter([ng], [y], s=22, color=ARRIVE, zorder=4, linewidths=0)
        for v, col, dy, va in ((nu, CYAN_TXT, -4.0, 'top'), (ng, ARRIVE_TXT, 4.2, 'bottom')):
            a2.annotate(f'{v}', xy=(v, y), xytext=(0, dy), textcoords='offset points',
                        ha='center', va=va, fontsize=0.66 * BASE, color=col)
    a1.set_yticks(range(len(members))); a1.set_yticklabels(list(reversed(members)))
    a2.set_yticks(range(len(members))); a2.tick_params(axis='y', labelleft=False, length=0)
    strip(a2, prov, side='right')
    panels.append((a1, a2))
    yc += h + 0.06
(c1_low, c2_low), (c1_top, c2_top) = panels
for ax in (c1_top, c2_top):
    ax.tick_params(axis='x', labelbottom=False, length=0)
c1_low.set_xlabel('arrivals per broken decision', fontsize=0.9 * BASE)
c2_low.set_xlabel('net gain', fontsize=0.9 * BASE)
strip(c1_top, 'precision (log scale)')
strip(c2_top, 'net gain')
head(c1_top, 'c', 'Gating buys precision, not always net', dx=-70)
ky = 1.0 + (STRIP_PT + 7.5) / 72 / (c2_top.get_position().height * FH)
key(c2_top, 0.02, ky, CYAN, 'ungated', box=4.4)
key(c2_top, 0.58, ky, ARRIVE, 'gated', box=4.4)

save(fig, 'fig5_mechanism')
print(f'    a: layers {L} nets {nets} norms {norms} auc {auc} cos16 {cos16}')
print(f'    b: PCA-1 +{pca} vs difference-in-means +{dm}')
for nm, pu, pg, nu, ng in zip(names, prec_u, prec_g, net_u, net_g):
    print(f'    c: {nm:<11} precision {pu} -> {pg} | net {nu} -> {ng}')
