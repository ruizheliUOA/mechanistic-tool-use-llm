#!/usr/bin/env python3
"""Apply the exchange rate to VERIFIED published sample sizes."""
import sys; sys.path.insert(0,'.')
from exchange_rate import min_certifiable

ROWS = [
 # (paper, label_space, N, N_source, dest_reported, controls)
 ("CAA (Rimsky et al. 2024)", "BINARY (A/B)", 50,
  "Sec 4.1 + App F Table 9: 50 held-out MC questions per behaviour, 7 behaviours",
  "no", "not reported"),
 ("ITI (Li et al., NeurIPS 2023)", "PER-ITEM options (TruthfulQA MC)", 817,
  "Sec 4.1: '817 questions spanning 38 subcategories'; 2-fold CV (Sec 4.3)",
  "no", "YES - random directions, Table 3"),
 ("Tan et al. 2024 (steering reliability)", "BINARY (A/B)", 500,
  "Sec 4.1: 40-10-50 train-val-test split of 1000 samples/dataset, 40 datasets",
  "no", "not reported"),
 ("AxBench (Wu et al. 2025)", "OPEN-ENDED (judge 0/1/2)", 10,
  "Sec 3.3: 10 instructions sampled per concept; 500 concepts (Concept500)",
  "no", "not reported"),
]
NOT_ACCESSED = [
 ("RepE (Zou et al.)", "abstract only in fetch budget; full text not retrieved"),
 ("CAST (Lee et al., ICLR 2025)", "arXiv page returned abstract only; full text not retrieved"),
]
print(f"{'paper':40}{'label space':34}{'N':>6}{'min cert.':>11}  {'dest?':6}{'controls'}")
print("-"*118)
for p,ls,n,src,dest,ctrl in ROWS:
    m = min_certifiable(n)
    ms = f"{m:.4f}" if m else "none"
    print(f"{p:40}{ls:34}{n:>6}{ms:>11}  {dest:6}{ctrl}")
print("\nN SOURCE REFERENCES (Rule 1 — every N quoted, none inferred)")
for p,_,n,src,_,_ in ROWS:
    print(f"  {p}\n    N={n}  <- {src}")
print("\nNOT ACCESSED (full text not retrieved; NOT audited from abstract)")
for p,why in NOT_ACCESSED:
    print(f"  {p}: {why}")
print("\nCOUNTS")
print(f"  papers with VERIFIED N        : {len(ROWS)}")
print(f"  papers NOT ACCESSED           : {len(NOT_ACCESSED)}")
print(f"  BINARY label space            : {sum(1 for r in ROWS if r[1].startswith('BINARY'))}/{len(ROWS)}")
print(f"  shared >=3-way action ontology: 0/{len(ROWS)}")
print(f"  destination outcomes reported : {sum(1 for r in ROWS if r[4]=='yes')}/{len(ROWS)}")
print(f"  random/reverse controls run   : {sum(1 for r in ROWS if r[5].startswith('YES'))}/{len(ROWS)}")
