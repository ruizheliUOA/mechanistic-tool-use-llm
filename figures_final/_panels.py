"""Panel builders shared by the manuscript figures.

Every panel reads the evidence index and the frozen verdict artifacts once, through
`load_all()`, and asserts the identities that the captions state, so a panel drawn in
two figures cannot drift. Panels set their own scales and labels; the caller places
the axes and adds the "(a) ..." heading. No figure uses more than two panels."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _style import *
import csv
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator

Q8_PATH = 'final/results/qwen3_stage2_formal/QWEN3_STAGE2_FORMAL_RESULTS.json'
Q4N_PATH = 'final/results/qwen3_4b_w2c_formal_v1/FORMAL_RANDOM_NULL_RESULTS.json'
Q4_PATH = 'final/results/qwen3_4b_w2c_formal_v1/QWEN3_4B_FORMAL_RESULTS.json'
GE_PATH = ('research_exploration/sakiko_final_model_breadth_panel_v1/gemma/'
           'formal_freeze/FORMAL_RESULTS.json')
SEEDS_PATH = 'final/results/clean/p1_multiseed_table.csv'
VERDICT = {'FORMAL_CONFIRMATORY_SUCCESS': 'ADMIT', 'QWEN3_4B_FORMAL_DECLINE': 'DECLINE',
           'GEMMA_FORMAL_DECLINE': 'DECLINE'}
FOREST = [('Qwen3-8B', 'q8b'), ('Qwen3-4B', 'q4b'), ('Gemma-2-9B', 'gemma')]


def load_all():
    """Every value the manuscript figures draw, with the captions' identities asserted."""
    ev, rows = load(), load_rows()
    Q8, Q4N, Q4, GE = (frozen(p) for p in (Q8_PATH, Q4N_PATH, Q4_PATH, GE_PATH))
    D = {'ev': ev, 'rows': rows}

    # --- sealed random nulls: the real direction against 59 matched random ones ---
    nulls = []
    for name, p, vals, real, ge_real in [
            ('Qwen3-8B', 'q8b', Q8['random_distribution'], Q8['primary']['real'],
             Q8['primary']['randoms_ge_real']),
            ('Qwen3-4B', 'q4b', Q4N['random_target_gain_rates'], Q4['arms']['real_d_grad'],
             Q4N['randoms_ge_real']),
            ('Gemma-2-9B', 'gemma', GE['random_tgr'],
             {'n_channel_error': GE['real']['n'], 'gold_arrivals': GE['real']['gold'],
              'wrong_to_wrong': GE['real']['other']}, GE['n_random_ge_real'])]:
        n, gold, other = real['n_channel_error'], real['gold_arrivals'], real['wrong_to_wrong']
        assert gold == int(ev[f'{p}_gold_arrival']) and other == int(ev[f'{p}_other_wrong'])
        tgr = (gold - other) / n
        assert len(vals) == 59 and sum(v >= tgr for v in vals) == ge_real == 0
        if f'{p}_matched_random_exceed' in ev:
            assert int(ev[f'{p}_matched_random_exceed']) == ge_real
        nulls.append({'name': name, 'vals': list(vals), 'n': n, 'tgr': tgr,
                      'ge_real': ge_real, 'rmax': max(vals)})
    assert abs(Q4N['real_target_gain_rate'] - nulls[1]['tgr']) < 1e-12
    D['nulls'] = nulls

    # --- Qwen3-8B: the activation arm against the score-space comparator ---
    act, ss = Q8['primary']['real'], Q8['secondary']['score_space_comparator']
    assert (act['gold_arrivals'], act['wrong_to_wrong']) == (int(ev['q8b_gold_arrival']),
                                                             int(ev['q8b_other_wrong']))
    assert (ss['gold_arrivals'], ss['wrong_to_wrong']) == (int(ev['q8b_score_space_gold']),
                                                           int(ev['q8b_score_space_other_wrong']))
    th_a, th_s = num(ev, 'q8b_target_hit'), num(ev, 'q8b_score_space_target_hit')
    assert abs(act['target_hit'] - th_a) < 5e-5 and abs(ss['target_hit'] - th_s) < 5e-5
    act_net = act['gold_arrivals'] - act['broke']          # channel-level Net (S-3)
    assert act_net == int(ev['q8b_gated_net'])
    D['arms'] = {'metrics': [('other wrong', act['wrong_to_wrong'], ss['wrong_to_wrong']),
                             ('gold arrivals', act['gold_arrivals'], ss['gold_arrivals']),
                             ('channel-level net', act_net, ss['gold_arrivals'] - ss['broke'])],
                 'th_a': th_a, 'th_s': th_s}

    # --- Phi-3.5: the outcomes of one intervention, on both populations ---
    exits, gold, oth = (int(ev[k]) for k in ('phi_source_exits', 'phi_gold_arrival',
                                             'phi_other_wrong'))
    expo, brk = int(ev['phi_exposed_correct']), int(ev['phi_broken_e2'])
    e1_n = int(rows['phi_collateral_e1']['denominator'].split('=')[1])
    assert gold + oth == exits and gold - brk == int(ev['phi_net'])
    assert abs(brk / expo - num(ev, 'phi_collateral_e2')) < 5e-5
    assert abs(brk / e1_n - num(ev, 'phi_collateral_e1')) < 5e-5
    assert abs(gold / exits - num(ev, 'phi_target_hit')) < 5e-5
    D['phi'] = {'exits': exits, 'gold': gold, 'oth': oth, 'expo': expo, 'brk': brk,
                'kept': expo - brk, 'e1_n': e1_n, 'net': int(ev['phi_net']),
                'th': num(ev, 'phi_target_hit')}

    # --- preservation on both denominators, per setting ---
    q8_e1 = (Q8['primary']['real']['collateral_count'], Q8['primary']['real']['baseline_correct_n'])
    q4_e1 = (Q4['arms']['real_d_grad']['broke'], Q4['arms']['real_d_grad']['baseline_correct_n'])
    ge_e1 = (GE['collateral_all'][0], GE['collateral_all'][1])
    assert q8_e1 == (0, 211) and abs(num(ev, 'q8b_collateral_e1')) < 1e-9
    e2 = {}
    for p in ('q4b', 'gemma'):
        n = int(rows[f'{p}_collateral_e2']['denominator'].split('=')[1])
        assert n == int(ev[f'{p}_exposed_correct'])
        e2[p] = (round(num(ev, f'{p}_collateral_e2') * n), n)
    assert ev['q8b_collateral_e2'] == 'VACUOUS'
    D['preservation'] = [
        ('Phi-3.5', 'historical', (brk, expo), (brk, e1_n), False),
        ('Qwen3-8B', 'sealed', (0, int(ev['q8b_exposed_correct'])), q8_e1, True),
        ('Qwen3-4B', 'sealed', e2['q4b'], q4_e1, False),
        ('Gemma-2-9B', 'sealed', e2['gemma'], ge_e1, False)]

    # --- historical control batteries ---
    D['phi_arms'] = [('real', 'phi_e2_real_locked'), ('random direction', 'phi_e2_random_dir'),
                     ('reversed', 'phi_e2_reverse_dir'), ('wrong layer', 'phi_e2_wrong_layer'),
                     ('mismatched channel', 'phi_e2_mismatched')]
    assert triple(ev, 'phi_e2_random_dir')[2] < int(ev['phi_net'])
    D['q25'] = {'real': int(ev['qwen25_real_net']), 'rev': int(ev['qwen25_reverse_net']),
                'rmean': num(ev, 'qwen25_random_mean_net'),
                'rmax': int(ev['qwen25_random_max_net'])}
    assert ev['qwen25_real_ge_all_random'] == 'True'

    # --- Phi-3.5 realizations (Appendix D) ---
    seeds = list(csv.DictReader(open(SEEDS_PATH)))
    assert [int(s['net']) for s in seeds] == [int(x) for x in ev['phi_multiseed_nets'].split(';')]
    assert all(int(s['fixed']) - int(s['broke']) == int(s['net']) for s in seeds)
    assert abs(int(seeds[0]['broke']) / e1_n - num(ev, 'phi_collateral_e1')) < 5e-5
    D['seeds'] = seeds
    return D


# ═════════ what one intervention did, on both populations ═════════
def draw_outcomes(ax, D):
    """Phi-3.5: the 153 source exits and the 93 exposed correct decisions, resolved."""
    gg(ax, x='continuous', y='discrete')
    p = D['phi']
    rows_ = [(0, 'baseline-correct,\nRouter fired', [(p['kept'], UNCHANGED, 'kept', GREY20),
                                                     (p['brk'], BROKEN, 'broken', 'white')]),
             (1, 'baseline errors\nmoved off source', [(p['gold'], ARRIVE, 'arrive', 'white'),
                                                       (p['oth'], OTHER, 'elsewhere', GREY20)])]
    for y, lab, parts in rows_:
        left = 0
        for n_, colour, text, tc in parts:
            ax.barh(y, n_, left=left, height=0.5, color=colour, zorder=3)
            ax.text(left + n_ / 2, y, f'{n_} {text}', ha='center', va='center',
                    fontsize=0.75 * BASE, color=tc, fontweight='bold', zorder=4)
            left += n_
        ax.text(left + 13, y, f'of {left}', ha='left', va='center', fontsize=0.8 * BASE,
                color=GREY30)
    ax.set_yticks([1, 0]); ax.set_yticklabels([rows_[1][1], rows_[0][1]], linespacing=1.2)
    ax.set_ylim(-0.55, 1.85); ax.set_xlim(0, 186)
    ax.xaxis.set_major_locator(FixedLocator([0, 50, 100, 150]))
    ax.set_xlabel('decisions')
    ax.text(0, 1.60, f'aggregate: net gain +{p["net"]}   ·   target-hit {p["th"]:.2f}',
            fontsize=0.85 * BASE, color=GREY20)


def draw_forest(ax, D):
    """Target-hit with 95% bootstrap intervals against the 0.50 criterion."""
    gg(ax, x='continuous', y='discrete')
    ev = D['ev']
    ax.axvline(0.50, color=GREY30, lw=0.9, ls=(0, (3.5, 2.2)), zorder=2)
    for y, (name, p) in zip([2, 1, 0], FOREST):
        pt, lo, hi = (num(ev, f'{p}_target_hit{s}') for s in ('', '_ci_lo', '_ci_hi'))
        v = VERDICT[ev[f'{p}_formal_verdict']]
        fill, tc = (ARRIVE, ARRIVE_TXT) if v == 'ADMIT' else (DECLINE, DECLINE_TXT)
        ax.plot([lo, hi], [y, y], color=GREY20, lw=1.1, zorder=3, solid_capstyle='butt')
        for b in (lo, hi):
            ax.plot([b, b], [y - 0.07, y + 0.07], color=GREY20, lw=1.0, zorder=3)
        ax.scatter([pt], [y], s=46, marker='o', facecolor=fill, edgecolor=GREY20,
                   linewidths=0.8, zorder=4)
        ax.annotate(f'{pt:.3f}', xy=(pt, y), xytext=(0, 6.0), textcoords='offset points',
                    ha='center', va='bottom', fontsize=0.8 * BASE, color=GREY20)
        ax.text(1.105, y, v, ha='right', va='center', fontsize=0.85 * BASE, color=tc,
                fontweight='bold')
    ax.set_yticks([2, 1, 0])
    ax.set_yticklabels([f'{nm}\n{ev[p + "_source_exits"]} source exits' for nm, p in FOREST],
                       linespacing=1.2)
    ax.set_xlim(0.38, 1.12); ax.set_ylim(-0.62, 2.72)
    ax.xaxis.set_major_locator(FixedLocator([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]))
    ax.text(0.507, 2.60, 'criterion 0.50', ha='left', va='top', fontsize=0.8 * BASE,
            color=GREY30)
    ax.set_xlabel('target-hit, 95\\% bootstrap interval' if False else
                  'target-hit, 95% bootstrap interval')


# ═════════ correction is direction-specific ═════════
def draw_phi_controls(ax, D):
    """Phi-3.5 structural controls, as a diverging plot: what each arm corrects and
    what it costs, with the net marked. The real arm carries the intervention colour;
    each control is tinted by the component it corrupts."""
    gg(ax, x='continuous', y='discrete')
    ev = D['ev']
    labels = ['real', 'random direction', 'reversed', 'wrong layer', 'mismatched channel']
    accent = [REAL, BLUE, ORANGE, TAN, ROSE]          # direction · sign · site · channel
    for y, ((lab, k), text, acc) in zip(range(4, -1, -1),
                                        zip(D['phi_arms'], labels, accent)):
        f, b, n = triple(ev, k)
        assert f - b == n
        ax.barh(y, f, height=0.52, color=ARRIVE, alpha=0.85, zorder=3)
        ax.barh(y, -b, height=0.52, color=BROKEN, alpha=0.85, zorder=3)
        ax.plot([n, n], [y - 0.30, y + 0.30], color=acc, lw=2.4, zorder=5,
                solid_capstyle='round')
        ax.annotate(f'net {n:+d}', xy=(f, y), xytext=(6, 0), textcoords='offset points',
                    ha='left', va='center', fontsize=0.8 * BASE, color=dark(acc, 0.34),
                    fontweight='bold' if lab == 'real' else 'normal')
    ax.axvline(0, color=GREY30, lw=0.7, zorder=4)
    ax.set_yticks(range(5)); ax.set_yticklabels(list(reversed(labels)))
    ax.get_yticklabels()[-1].set_color(REAL_TXT)
    ax.set_ylim(-0.62, 5.45); ax.set_xlim(-82, 152)
    ax.xaxis.set_major_locator(FixedLocator([-50, 0, 50, 100]))
    ax.set_xticklabels(['50', '0', '50', '100'])
    ax.set_xlabel('broken  ←  decisions  →  corrected')
    key(ax, 0.02, 0.955, ARRIVE, 'corrected', box=4.4, size=0.76 * BASE)
    key(ax, 0.33, 0.955, BROKEN, 'broken', box=4.4, size=0.76 * BASE)
    ax.plot([0.60, 0.60], [0.938, 0.972], color=REAL, lw=2.2,
            transform=ax.transAxes, clip_on=False, zorder=6)
    ax.annotate('net, in the colour of the corrupted component', xy=(0.60, 0.955),
                xycoords=ax.transAxes, xytext=(5, 0), textcoords='offset points',
                ha='left', va='center', fontsize=0.7 * BASE, color=GREY20,
                annotation_clip=False)


def draw_nulls(ax, D):
    """Every sealed setting's matched-random null, with the real direction marked."""
    gg(ax, x='continuous', y='discrete')
    rng = np.random.default_rng(20260913)          # jitter only; deterministic seed
    for y, d in zip([2, 1, 0], D['nulls']):
        jit = rng.uniform(-0.13, 0.13, len(d['vals']))
        ax.scatter(d['vals'], y + jit, s=9, color=NULLGREY, alpha=0.55, linewidths=0,
                   zorder=3)
        ax.scatter([d['tgr']], [y], s=52, marker='D', color=REAL, edgecolor='white',
                   linewidths=0.6, zorder=5)
        ax.annotate(f'{d["tgr"]:.3f}', xy=(d['tgr'], y), xytext=(0, 7.5),
                    textcoords='offset points', ha='center', va='bottom',
                    fontsize=0.8 * BASE, color=REAL_TXT, fontweight='bold')
        ax.text(0.985, y - 0.30, f'{d["ge_real"]} of 59 reach it', ha='right', va='center',
                fontsize=0.78 * BASE, color=GREY30, transform=ax.get_yaxis_transform())
    ax.axvline(0, color=GREY30, lw=0.7, ls=(0, (3.5, 2.2)), zorder=2)
    ax.set_yticks([2, 1, 0]); ax.set_yticklabels([d['name'] for d in D['nulls']])
    ax.set_ylim(-0.62, 3.18); ax.set_xlim(-0.125, 0.33)
    ax.xaxis.set_major_locator(FixedLocator([-0.1, 0.0, 0.1, 0.2, 0.3]))
    ax.set_xlabel('target gain')
    key(ax, 0.015, 0.935, NULLGREY, '59 matched random', box=4.4, size=0.76 * BASE)
    ax.scatter([0.565], [0.935], s=30, marker='D', color=REAL, edgecolor='white',
               linewidths=0.5, transform=ax.transAxes, clip_on=False, zorder=6)
    ax.annotate('real direction', xy=(0.565, 0.935), xycoords=ax.transAxes,
                xytext=(5.2, 0), textcoords='offset points', ha='left', va='center',
                fontsize=0.76 * BASE, color=GREY20, annotation_clip=False)


# ═════════ destination and preservation ═════════
def draw_dumbbell(ax, D):
    """Qwen3-8B: the activation arm against the score-space comparator."""
    gg(ax, x='continuous', y='discrete')
    a_ = D['arms']
    for y, (lab, a, s) in enumerate(a_['metrics']):
        ax.plot([a, s], [y, y], color=GREY_L, lw=2.6, solid_capstyle='round', zorder=2)
        ax.scatter([s], [y], s=40, color=COMPARE, zorder=3, linewidths=0)
        ax.scatter([a], [y], s=44, marker='D', color=REAL, zorder=4, linewidths=0)
        for v, col, sgn in ((a, REAL_TXT, 1 if a >= s else -1),
                            (s, COMPARE_TXT, 1 if s > a else -1)):
            ax.annotate(f'{v}', xy=(v, y), xytext=(sgn * 6.0, 0), textcoords='offset points',
                        ha='left' if sgn > 0 else 'right', va='center', fontsize=0.9 * BASE,
                        color=col, fontweight='bold')
    ax.set_yticks(range(3)); ax.set_yticklabels([m[0] for m in a_['metrics']])
    ax.set_xlim(0, 48); ax.set_ylim(-0.6, 3.15)
    ax.xaxis.set_major_locator(FixedLocator([0, 10, 20, 30, 40]))
    ax.set_xlabel('count, of 72 routed channel errors')
    ax.legend(handles=[Line2D([], [], marker='D', ls='', color=REAL, ms=5.0,
                              label=f'activation   target-hit {a_["th_a"]:.3f}'),
                       Line2D([], [], marker='o', ls='', color=COMPARE, ms=5.0,
                              label=f'score-space   target-hit {a_["th_s"]:.3f}')],
              loc='upper left', bbox_to_anchor=(0.005, 1.0), ncol=1, handletextpad=0.4,
              borderaxespad=0.3, labelspacing=0.3)


def draw_preservation(ax, D):
    """Collateral on both denominators, per setting, against the 5% bound."""
    gg(ax, x='continuous', y='discrete')
    ax.axvline(5, color=GREY30, lw=0.9, ls=(0, (3.5, 2.2)), zorder=2)
    ys = list(range(len(D['preservation']) - 1, -1, -1))
    for y, (name, prov, (b2, n2), (b1, n1), vacuous) in zip(ys, D['preservation']):
        r2, r1 = 100 * b2 / n2, 100 * b1 / n1
        ax.plot([r1, r2], [y, y], color=GREY_L, lw=2.2, solid_capstyle='round', zorder=3)
        ax.scatter([r1], [y], s=34, facecolor='white', edgecolor=BROKEN, linewidths=1.2,
                   zorder=4)
        ax.scatter([r2], [y], s=40, color=BROKEN if not vacuous else GREY_L, zorder=5,
                   linewidths=0)
        label = (f'E2 {b2} of {n2}' + (' (vacuous)' if vacuous else '')
                 + f'   ·   E1 {b1} of {n1}')
        ax.annotate(label, xy=(max(min(r1, r2), 9.0), y), xytext=(0, 7.5),
                    textcoords='offset points', ha='left', va='bottom',
                    fontsize=0.76 * BASE, color=GREY30 if vacuous else BROKEN_TXT)
    ax.set_yticks(ys)
    ax.set_yticklabels([f'{nm}\n({prov})' for nm, prov, *_ in D['preservation']],
                       linespacing=1.2)
    ax.set_ylim(-0.75, len(D['preservation']) + 1.20); ax.set_xlim(-2, 74)
    ax.xaxis.set_major_locator(FixedLocator([0, 5, 20, 40, 60]))
    ax.set_xlabel('collateral: correct decisions broken (%)')
    ax.text(6.2, -0.45, 'bound 5%', ha='left', va='center', fontsize=0.78 * BASE,
            color=GREY30)
    ax.scatter([0.035], [0.955], s=30, facecolor='white', edgecolor=BROKEN, linewidths=1.1,
               transform=ax.transAxes, clip_on=False, zorder=6)
    ax.annotate('E1: of all baseline-correct', xy=(0.035, 0.955), xycoords=ax.transAxes,
                xytext=(5.5, 0), textcoords='offset points', ha='left', va='center',
                fontsize=0.76 * BASE, color=GREY20, annotation_clip=False)
    ax.scatter([0.035], [0.865], s=34, color=BROKEN, transform=ax.transAxes, clip_on=False,
               zorder=6, linewidths=0)
    ax.annotate('E2: of those the Router fired on', xy=(0.035, 0.865), xycoords=ax.transAxes,
                xytext=(5.5, 0), textcoords='offset points', ha='left', va='center',
                fontsize=0.76 * BASE, color=GREY20, annotation_clip=False)


# ═════════ appendix: historical realizations and battery ═════════
def draw_seeds(ax, D):
    """Phi-3.5 realizations: E1 collateral of every seed against the 5% bound."""
    gg(ax, x='continuous', y='discrete')
    seeds, e1_n = D['seeds'], D['phi']['e1_n']
    ax.axvline(5, color=GREY30, lw=0.9, ls=(0, (3.5, 2.2)), zorder=2)
    for y, s in zip(range(len(seeds) - 1, -1, -1), seeds):
        e1 = 100 * int(s['broke']) / e1_n
        ax.plot([0, e1], [y, y], color=SALMON, lw=1.2, alpha=0.45, zorder=3)
        ax.scatter([e1], [y], s=34, color=BROKEN, zorder=4, linewidths=0)
        ax.annotate(f'{int(s["broke"])} broken', xy=(e1, y), xytext=(6, 0),
                    textcoords='offset points', ha='left', va='center',
                    fontsize=0.78 * BASE, color=BROKEN_TXT)
        ax.text(0.985, y, f'net +{s["net"]}', ha='right', va='center', fontsize=0.78 * BASE,
                color=ARRIVE_TXT, transform=ax.get_yaxis_transform())
    ax.set_yticks(range(len(seeds)))
    ax.set_yticklabels([f'seed {s["seed"]}' for s in reversed(seeds)])
    ax.set_xlim(0, 42); ax.set_ylim(-0.7, len(seeds) - 0.3)
    ax.xaxis.set_major_locator(FixedLocator([0, 5, 10, 20, 30, 40]))
    ax.set_xlabel(f'E1 collateral: broken of {e1_n} baseline-correct (%)')
    ax.text(5.8, len(seeds) - 0.42, 'bound 5%', ha='left', va='top', fontsize=0.78 * BASE,
            color=GREY30)


def draw_q25_battery(ax, D):
    """Qwen2.5-7B locked battery: the real direction against its controls."""
    gg(ax, x='continuous', y='discrete')
    q = D['q25']
    items = [('real', q['real'], REAL), ('random,\nmean of 10', q['rmean'], NULLGREY),
             ('reversed', q['rev'], ORANGE)]
    for y, (lab, v, c) in zip([2, 1, 0], items):
        ax.plot([0, v], [y, y], color=c, lw=1.4, zorder=3)
        ax.scatter([v], [y], s=44 if c == REAL else 40, marker='D' if c == REAL else 'o',
                   color=c, zorder=4, linewidths=0)
        ax.annotate(f'+{v:g}', xy=(v, y), xytext=(0, 6), textcoords='offset points',
                    ha='center', va='bottom', fontsize=0.8 * BASE,
                    color=dark(c, 0.34) if c != NULLGREY else GREY20)
    ax.plot([q['rmax'], q['rmax']], [0.76, 1.24], color=NULLGREY, lw=1.1, zorder=3)
    ax.annotate(f'largest of the ten: +{q["rmax"]}', xy=(q['rmax'], 0.76), xytext=(4, -1),
                textcoords='offset points', ha='left', va='top', fontsize=0.78 * BASE,
                color=GREY30)
    ax.set_yticks([2, 1, 0]); ax.set_yticklabels([i[0] for i in items], linespacing=1.15)
    ax.set_ylim(-0.65, 2.7); ax.set_xlim(0, 95)
    ax.xaxis.set_major_locator(FixedLocator([0, 20, 40, 60, 80]))
    ax.set_xlabel('net gain')
