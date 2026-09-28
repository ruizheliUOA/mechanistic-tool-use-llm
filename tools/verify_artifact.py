#!/usr/bin/env python3
"""Read-only artifact checks. Standard library only; no model, network, or writes.

This checks preserved results, not a new execution of any sealed experiment.
The release manifest describes delivered bytes; it does not replace frozen hashes.
"""
import argparse
import ast
import collections
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

POINTER = re.compile(rb'version https://git-lfs.github.com/spec/v1\noid sha256:([0-9a-f]{64})\nsize (\d+)\n?\Z')

def payload(path):
    data = path.read_bytes()
    if POINTER.fullmatch(data):
        raise ValueError(f'MISSING_PAYLOAD: {path.name} is a Git LFS pointer, not evidence')
    return data

def file_at(root, rel):
    mapping = json.loads((root / "manifests/paths.json").read_text())["paths"]
    return root / mapping.get(rel, rel)

def js(root, rel):
    return json.loads(payload(file_at(root, rel)))

def jl(root, rel):
    return [json.loads(line) for line in payload(file_at(root, rel)).splitlines() if line.strip()]

def counts(rows):
    errors = [r for r in rows if r['gold'] == 'cannot_answer' and r['pred_base'] == 'tool_call']
    exits = [r for r in errors if r['pred_int'] != 'tool_call']
    gold = sum(r['pred_int'] == r['gold'] for r in exits)
    correct = [r for r in rows if r['gold'] == r['pred_base']]
    broke = sum(r['pred_int'] != r['gold'] for r in correct)
    fixed = sum(r['pred_base'] != r['gold'] and r['pred_int'] == r['gold'] for r in rows)
    return dict(routed_errors=len(errors), exits=len(exits), gold=gold, other=len(exits)-gold,
                exposed_correct=len(correct), broke=broke, fixed=fixed, net=fixed-broke)

def check(root):
    checks = []
    def require(ok, description):
        if not ok: raise AssertionError(description)
        checks.append(description)
    release = js(root, 'manifests/release.json')
    for item in release['files']:
        p = root / item['path']
        require(p.is_file(), 'exists: ' + item['path'])
        data = p.read_bytes()
        require(len(data) == item['bytes'] and hashlib.sha256(data).hexdigest() == item['sha256'],
                'release bytes: ' + item['path'])
    paths = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()
             and '.git' not in p.relative_to(root).parts and '__pycache__' not in p.parts}
    require(paths == {f['path'] for f in release['files']} | {'manifests/release.json'},
            'release file list is complete (manifest excludes only itself)')
    for item in js(root, 'manifests/payloads.json')['bundled_lfs_payloads']:
        data = payload(root / item['path'])
        require(len(data) == item['bytes'] and hashlib.sha256(data).hexdigest() == item['sha256'],
                'recovered payload: ' + item['path'])
    py = sorted(root.rglob('*.py'))
    for p in py:
        if '.git' not in p.parts: ast.parse(p.read_text(), filename=str(p.relative_to(root)))
    checks.append(f'Python syntax: {len(py)} files, parsed without import or execution')

    q8dir = 'final/results/qwen3_stage2_formal/'
    q4dir = 'final/results/qwen3_4b_w2c_formal_v1/'
    gedir = 'research_exploration/sakiko_final_model_breadth_panel_v1/gemma/formal_freeze/'
    summaries = {}
    specs = [('q8b', q8dir+'QWEN3_STAGE2_FORMAL_RECORDS.jsonl', 'd_grad', 6372, (72,52,38,14,6,0,40,40)),
             ('q4b', q4dir+'QWEN3_4B_FORMAL_RECORDS.jsonl', 'real_d_grad', 14404, (118,64,37,27,50,1,40,39)),
             ('gemma', gedir+'FORMAL_RECORDS.jsonl', 'real_d_grad', 7731, (96,27,17,10,11,1,17,16))]
    evidence = {r['variable']: r for r in csv.DictReader(file_at(root, 'final_evidence/FINAL_PAPER_EVIDENCE.csv').open())}
    for name, rel, real_arm, total, expected in specs:
        rows = jl(root, rel)
        require(len(rows) == total, name+': record count')
        grouped = collections.defaultdict(list)
        for r in rows: grouped[r['arm']].append(r)
        real = counts(grouped[real_arm])
        require(tuple(real.values()) == expected, name+': raw destination / exposure / net counts')
        for arm, records in grouped.items():
            ids = [r.get('sample_id', r.get('uuid')) for r in records]
            require(len(ids) == len(set(ids)), name+': unique samples in '+arm)
        require(all(r['pred_base'] == r['pred_int'] and r['scores_base'] == r['scores_int']
                    for r in grouped['zero']), name+': zero control is exactly baseline')
        randoms = [v for k,v in grouped.items() if k.startswith('formal_random_')]
        require(len(randoms) == 59, name+': 59 random directions')
        require(sum(counts(v)['gold']-counts(v)['other'] >= real['gold']-real['other'] for v in randoms) == 0,
                name+': no random direction reaches real target-gain numerator')
        for suffix, value in [('gold_arrival',real['gold']),('other_wrong',real['other']),('source_exits',real['exits'])]:
            require(int(evidence[name+'_'+suffix]['value']) == value, name+': evidence index '+suffix)
        summaries[name] = real
        baseline_union = {r.get('sample_id',r.get('uuid')): r for r in rows}
        if name in ('q8b','q4b'):
            expected_n = {'q8b':87,'q4b':124}[name]
            require(sum(r['gold']=='cannot_answer' and r['pred_base']=='tool_call'
                        for r in baseline_union.values()) == expected_n,
                    name+': channel-error denominator from archived baseline identities')

    q8 = js(root, q8dir+'QWEN3_STAGE2_FORMAL_RESULTS.json')
    q4 = js(root, q4dir+'QWEN3_4B_FORMAL_RESULTS.json')
    ge = js(root, gedir+'FORMAL_PRINCIPAL_VERDICT.json')
    require(q8['outcome']=='FORMAL_CONFIRMATORY_SUCCESS' and len(q8['primary_conjunction'])==10
            and all(q8['primary_conjunction'].values()), 'q8b: preserved ten-condition success')
    require(file_at(root, q4dir+'FORMAL_PRINCIPAL_VERDICT.txt').read_text().strip()=='QWEN3_4B_FORMAL_DECLINE',
            'q4b: preserved principal DECLINE')
    require(len(q4['conjunction'])==10 and sum(q4['conjunction'].values())==8,
            'q4b: preserved 8/10 conjunction')
    require(ge['principal_verdict']=='GEMMA_FORMAL_DECLINE' and ge['n_pass']==8,
            'gemma: preserved principal DECLINE and 8/10 conjunction')
    for name, real in [('q8b',q8['primary']['real']),('q4b',q4['arms']['real_d_grad'])]:
        c=summaries[name]
        require((real['gold_arrivals'],real['wrong_to_wrong'],real['source_exits'],real['fixed'],real['broke'],real['net'])
                == (c['gold'],c['other'],c['exits'],c['fixed'],c['broke'],c['net']), name+': raw versus frozen summary')
    q4base=jl(root,q4dir+'FORMAL_BASELINE_RECORDS.jsonl')
    require(len(q4base)==452 and sum(r['gold']==r['pred_base'] for r in q4base)==186,
            'q4b: bundled source-prediction baseline subset 452 rows / 186 correct')
    gebase=js(root,gedir+'SEALED_BASELINE.json')
    require(len(gebase)==548 and sum(r['gold']==r['pred'] for r in gebase)==223,
            'gemma: full baseline population denominator 223 / 548')
    ge_all_errors=sum(r['gold']=='cannot_answer' and r['pred']=='tool_call' for r in gebase)
    require(ge_all_errors==112 and ge['headline']['channel_errors']==96,
            'gemma: retain distinction between 112 full-baseline errors and 96 routed errors in frozen headline')
    for name, ci in [('q8b',q8['primary']['real']['target_hit_ci95']),('q4b',q4['arms']['real_d_grad']['target_hit_ci95']),('gemma',ge['headline']['target_hit_ci95'])]:
        for suffix,value in zip(('lo','hi'),ci):
            require(abs(float(evidence[name+'_target_hit_ci_'+suffix]['value'])-value)<0.00005,
                    name+': index agrees with frozen target-hit interval '+suffix)
    phi=jl(root,'final/results/clean/p0_final_test_details.jsonl')
    fixed=sum(r['clean_pred']!=r['gold'] and r['int_pred']==r['gold'] for r in phi)
    broke=sum(r['clean_pred']==r['gold'] and r['int_pred']!=r['gold'] for r in phi)
    exposed=[r for r in phi if r['route']!='none' and r['clean_pred']==r['gold']]
    require((len(phi),fixed,broke,len(exposed),fixed-broke)==(548,107,52,93,55),
            'Phi historical seed 42: 548 rows, 107 fixed, 52 broken, 93 exposed, net +55')
    seeds=list(csv.DictReader(file_at(root, 'final/results/clean/p1_multiseed_table.csv').open()))
    require(all(int(r['fixed'])-int(r['broke'])==int(r['net']) for r in seeds), 'Phi multiseed conservation')
    split=list(csv.DictReader(file_at(root, 'final/results/qwen3_stage0_1_v2/QWEN3_V2_SPLIT_INDEX.csv').open()))
    require(len(split)==3104 and len({r['sample_id'] for r in split})==3104,'canonical non-test split uniqueness and count')
    # Historical tables recovered from existing records: arithmetic only, no new evaluation.
    table=list(csv.DictReader(file_at(root,'final/results/channel_adaptive/placebo_control_summary_qwen7b.csv').open()))
    table={r['variant']:r for r in table}
    for arm,expected in [('real',(92,13,79)),('reverse',(17,1,16)),('ungated',(116,22,94))]:
        row=table[arm]
        require(tuple(int(row[k]) for k in ('fixed','broke','net'))==expected,
                'Qwen2.5 archived control counts: '+arm)
        require(int(row['fixed'])-int(row['broke'])==int(row['net']),
                'Qwen2.5 control conservation: '+arm)
    independent=js(root,'sakiko/results/p2_independent_baselines.json')
    require(independent['REAL_LOCKED']['net']==62 and independent['UNGATED_TOP1_INDEP']['net']==-32,
            'Phi independent gated/ungated comparison is +62/-32, distinct from original +55 run')
    for key,row in independent.items():
        if isinstance(row,dict) and all(k in row for k in ('fixed','broke','net')):
            require(row['fixed']-row['broke']==row['net'],'Phi independent conservation: '+key)
    meta=js(root,'final/results/metatool_qwen7b_sakiko_ca/metatool_qwen7b_sakiko_ca_summary.json')
    for name,mean in [('B_overcall_only',8.6),('C_undercall_only',13.0),('D_dual',21.6)]:
        arm=meta['arms_5seed'][name]
        require(len(arm['nets'])==5 and all(n>0 for n in arm['nets']) and
                abs(sum(arm['nets'])/5-mean)<1e-9 and arm['mean']==mean,
                'MetaTool preserved five-seed aggregate: '+name)
    for a,b,c in zip(*(meta['arms_5seed'][name]['nets'] for name in ('B_overcall_only','C_undercall_only','D_dual'))):
        require(a+b==c,'MetaTool disjoint-channel additive count check (not independent replication)')
    return {'executed_checks':len(checks),'result':'AVAILABLE_EVIDENCE_CHECKS_PASSED','scope':'bundled bytes and reported result arithmetic; historical limitations in docs/limitations.md','destination_counts':summaries,
            'gemma_full_baseline_channel_errors':ge_all_errors,
            'limitations':[
                'This is not release approval: historical frozen-manifest discrepancies remain in docs/limitations.md.',
                'Qwen3-8B population denominator 211 and Qwen3-4B denominator 214 are retained from frozen summaries; complete 548-row baselines are not separately bundled for these settings.',
                'Gemma frozen target-gain uses 96 routed channel errors; its full baseline has 112. No denominator or verdict was changed.',
                'No model inference, bootstrap rerun, sealed experiment, or historical missing-destination reconstruction was performed.'
            ]}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument('--require-payload',type=Path,help='Check one file and fail explicitly on a pointer')
    args=ap.parse_args()
    try:
        if args.require_payload:
            data=payload(args.require_payload);print(json.dumps({'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}));return 0
        print(json.dumps(check(args.root.resolve()),indent=2));return 0
    except (AssertionError,ValueError,OSError,KeyError) as e:
        print('FAIL: '+str(e),file=sys.stderr);return 1

if __name__=='__main__':sys.exit(main())
