"""Export the figure data as tidy CSVs for the R scripts.

The evidence stays where it is: `_panels.load_all()` reads the evidence index and the
frozen verdict artifacts and asserts every identity the captions state. This script
only reshapes what it returns into long-form tables, one per panel, under
`figures_final/data/`. The R scripts read those tables and do nothing but plot, so
there is exactly one place where a number can enter a figure.

Run:  python figures_final/export_figure_data.py
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import csv
from _panels import load_all, FOREST, VERDICT
from _style import num, triple

OUT = 'figures_final/data'


def write(name, fieldnames, rows):
    os.makedirs(OUT, exist_ok=True)
    path = f'{OUT}/{name}.csv'
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f'  wrote {path}  ({len(rows)} rows)')


def main():
    D = load_all()
    ev, p = D['ev'], D['phi']

    # ---- Figure 1(a): one intervention, every outcome resolved ----------------
    write('outcomes', ['population', 'population_order', 'outcome', 'outcome_order',
                       'count', 'total'],
          [{'population': 'baseline errors moved off source', 'population_order': 2,
            'outcome': 'arrive', 'outcome_order': 1, 'count': p['gold'], 'total': p['exits']},
           {'population': 'baseline errors moved off source', 'population_order': 2,
            'outcome': 'elsewhere', 'outcome_order': 2, 'count': p['oth'], 'total': p['exits']},
           {'population': 'baseline-correct, Router fired', 'population_order': 1,
            'outcome': 'kept', 'outcome_order': 3, 'count': p['kept'], 'total': p['expo']},
           {'population': 'baseline-correct, Router fired', 'population_order': 1,
            'outcome': 'broken', 'outcome_order': 4, 'count': p['brk'], 'total': p['expo']}])

    # ---- Figure 1(b) / Section 5: target-hit with bootstrap intervals ---------
    write('forest', ['setting', 'setting_order', 'point', 'lo', 'hi', 'exits', 'verdict'],
          [{'setting': name, 'setting_order': i,
            'point': num(ev, f'{k}_target_hit'),
            'lo': num(ev, f'{k}_target_hit_ci_lo'), 'hi': num(ev, f'{k}_target_hit_ci_hi'),
            'exits': int(ev[f'{k}_source_exits']),
            'verdict': VERDICT[ev[f'{k}_formal_verdict']]}
           for i, (name, k) in enumerate(reversed(FOREST), start=1)])

    # ---- Figure 3(a): Phi-3.5 structural controls -----------------------------
    # `component` names what each arm corrupts; the R theme maps it to a colour.
    components = {'real': 'real', 'random direction': 'direction', 'reversed': 'sign',
                  'wrong layer': 'site', 'mismatched channel': 'channel'}
    labels = ['real', 'random direction', 'reversed', 'wrong layer', 'mismatched channel']
    rows = []
    for i, ((lab, key), text) in enumerate(zip(D['phi_arms'], labels)):
        f, b, n = triple(ev, key)
        assert f - b == n
        rows.append({'arm': text, 'arm_order': len(labels) - i, 'corrected': f,
                     'broken': b, 'net': n, 'component': components[text]})
    write('phi_controls', ['arm', 'arm_order', 'corrected', 'broken', 'net', 'component'], rows)

    # ---- Figure 3(b): the matched-random nulls, one row per direction ----------
    rows = []
    for i, d in enumerate(D['nulls']):
        order = len(D['nulls']) - i
        rows += [{'setting': d['name'], 'setting_order': order, 'kind': 'random',
                  'target_gain': v} for v in d['vals']]
        rows.append({'setting': d['name'], 'setting_order': order, 'kind': 'real',
                     'target_gain': d['tgr']})
    write('nulls', ['setting', 'setting_order', 'kind', 'target_gain'], rows)
    write('nulls_summary', ['setting', 'setting_order', 'real', 'random_max', 'ge_real', 'k'],
          [{'setting': d['name'], 'setting_order': len(D['nulls']) - i, 'real': d['tgr'],
            'random_max': d['rmax'], 'ge_real': d['ge_real'], 'k': len(d['vals'])}
           for i, d in enumerate(D['nulls'])])

    # ---- Figure 4(a): activation against the score-space comparator -----------
    rows = []
    for i, (metric, a, s) in enumerate(D['arms']['metrics']):
        rows.append({'metric': metric, 'metric_order': i + 1, 'arm': 'activation', 'count': a})
        rows.append({'metric': metric, 'metric_order': i + 1, 'arm': 'score-space', 'count': s})
    write('arms', ['metric', 'metric_order', 'arm', 'count'], rows)
    write('arms_target_hit', ['arm', 'target_hit'],
          [{'arm': 'activation', 'target_hit': D['arms']['th_a']},
           {'arm': 'score-space', 'target_hit': D['arms']['th_s']}])

    # ---- Figure 4(b): collateral on both denominators -------------------------
    rows = []
    for i, (name, prov, (b2, n2), (b1, n1), vacuous) in enumerate(D['preservation']):
        order = len(D['preservation']) - i
        rows.append({'setting': name, 'setting_order': order, 'protocol': prov,
                     'estimand': 'E2', 'broken': b2, 'n': n2, 'rate': 100 * b2 / n2,
                     'vacuous': int(vacuous)})
        rows.append({'setting': name, 'setting_order': order, 'protocol': prov,
                     'estimand': 'E1', 'broken': b1, 'n': n1, 'rate': 100 * b1 / n1,
                     'vacuous': 0})
    write('preservation', ['setting', 'setting_order', 'protocol', 'estimand', 'broken',
                           'n', 'rate', 'vacuous'], rows)

    # ---- Figure 5: the property and verdict map -------------------------------
    H, N, P, X = 'holds', 'fails', 'partial', 'not evaluated'
    e2 = {name: f'{b}/{n}' for name, prov, (b, n), e1, vac in D['preservation']}
    nulls = {d['name']: f'{d["ge_real"]}/{len(d["vals"])}' for d in D['nulls']}
    llama_max = max(int(x.split('/')[0]) for x in ev['llama_randoms_ge_real'].split(';'))
    th = lambda k: f'{num(ev, f"{k}_target_hit"):.3f}'
    grid = [
        ('Qwen3-8B', 'sealed', [('yes', H), ('yes', H), (nulls['Qwen3-8B'], H),
                                (th('q8b'), H), (e2['Qwen3-8B'], P)],
         VERDICT[ev['q8b_formal_verdict']]),
        ('Qwen3-4B', 'sealed', [('yes', H), ('yes', H), (nulls['Qwen3-4B'], H),
                                (th('q4b'), H), (e2['Qwen3-4B'], H)],
         VERDICT[ev['q4b_formal_verdict']]),
        ('Gemma-2-9B', 'sealed', [('yes', H), ('yes', H), (nulls['Gemma-2-9B'], H),
                                  (th('gemma'), H), (e2['Gemma-2-9B'], N)],
         VERDICT[ev['gemma_formal_verdict']]),
        ('Phi-3.5', 'historical', [('yes', H), ('yes', H), ('0/1', H), (th('phi'), H),
                                   (e2['Phi-3.5'], N)], 'not sealed'),
        ('Qwen2.5-7B', 'historical', [('yes', H), ('yes', H), ('0/10', H), ('n/a', X),
                                      ('--', X)], 'not sealed'),
        ('Llama-3.1-8B', 'historical', [('yes', H), ('yes', H), (f'{llama_max}/20', N),
                                        ('--', X), ('--', X)], 'not sealed'),
        ('Mistral-7B', 'historical', [('yes', H), ('yes', H),
                                      (ev['mistral_randoms_ge_real'], N), ('--', X),
                                      ('--', X)], 'not sealed'),
    ]
    props = ['Adjudicable', 'Readable', 'Steerable', 'Correctable', 'Preserving']
    rows = []
    for i, (name, prov, cells, verdict) in enumerate(grid):
        order = len(grid) - i
        for j, (label, status) in enumerate(cells):
            rows.append({'setting': name, 'setting_order': order, 'protocol': prov,
                         'property': props[j], 'property_order': j + 1,
                         'label': label, 'status': status})
        rows.append({'setting': name, 'setting_order': order, 'protocol': prov,
                     'property': 'Licensable', 'property_order': 6, 'label': verdict,
                     'status': {'ADMIT': H, 'DECLINE': P}.get(verdict, X)})
    write('ladder', ['setting', 'setting_order', 'protocol', 'property', 'property_order',
                     'label', 'status'], rows)

    # ---- Figure D1: historical realizations and battery -----------------------
    write('seeds', ['seed', 'seed_order', 'broken', 'n', 'rate', 'net'],
          [{'seed': s['seed'], 'seed_order': len(D['seeds']) - i, 'broken': int(s['broke']),
            'n': p['e1_n'], 'rate': 100 * int(s['broke']) / p['e1_n'], 'net': int(s['net'])}
           for i, s in enumerate(D['seeds'])])
    q = D['q25']
    write('q25_battery', ['arm', 'arm_order', 'net', 'kind', 'annotation'],
          [{'arm': 'real', 'arm_order': 3, 'net': q['real'], 'kind': 'real', 'annotation': ''},
           {'arm': 'random, mean of 10', 'arm_order': 2, 'net': q['rmean'], 'kind': 'control',
            'annotation': f'largest of the ten: +{q["rmax"]}'},
           {'arm': 'reversed', 'arm_order': 1, 'net': q['rev'], 'kind': 'control',
            'annotation': ''}])

    # ---- constants the captions and criteria depend on ------------------------
    write('constants', ['name', 'value', 'meaning'],
          [{'name': 'target_hit_criterion', 'value': 0.50,
            'meaning': 'the only destination boundary, frozen before any sealed evaluation'},
           {'name': 'collateral_bound_pct', 'value': 5.0,
            'meaning': 'preservation bound, adjudicated on E1'},
           {'name': 'add_one_p', 'value': 0.017,
            'meaning': 'add-one Monte Carlo p against 59 matched random directions'},
           {'name': 'phi_net', 'value': p['net'], 'meaning': 'Phi-3.5 net gain'},
           {'name': 'phi_target_hit', 'value': p['th'], 'meaning': 'Phi-3.5 target-hit'},
           {'name': 'q8b_channel_errors', 'value': int(ev['q8b_routed_channel_errors']),
            'meaning': 'routed channel errors behind Figure 4(a)'}])


if __name__ == '__main__':
    main()
