"""CPU-only operating characteristics of the frozen SAKIKO licence.
Uses the exact frozen endpoint definitions and CI procedures."""
import csv,itertools,json
import numpy as np
from scipy.stats import beta
RNG=np.random.default_rng(20260813); B=600; DRAWS=1000
def cp_upper(k,n):
    return 1.0 if n==0 or k==n else float(beta.ppf(.975,k+1,n-k))
def one(n_err,p_exit,p_hit,n_exp,p_coll,rng):
    """Simulate one formal trial under the frozen endpoints."""
    ex=rng.binomial(n_err,p_exit)
    gold=rng.binomial(ex,p_hit) if ex>0 else 0
    other=ex-gold
    if ex==0: return False,"no_exits"
    th=gold/ex; tg=(gold-other)/n_err
    # frozen: bootstrap CI on target gain (per-row +1/-1/0) and target hit (per-exit 0/1)
    v=np.array([1]*gold+[-1]*other+[0]*(n_err-ex))
    bs=rng.choice(v,size=(B,n_err),replace=True).sum(1)/n_err
    tg_lo=np.percentile(bs,2.5)
    hv=np.array([1]*gold+[0]*other)
    hbs=rng.choice(hv,size=(B,ex),replace=True).mean(1)
    th_lo=np.percentile(hbs,2.5)
    brk=rng.binomial(n_exp,p_coll) if n_exp>0 else 0
    den=223  # all-baseline-correct scale, frozen primary denominator
    cr=brk/den; cr_hi=cp_upper(brk,den)
    ok=(n_err>=30 and tg>0 and tg_lo>0 and th>0.50 and th_lo>0.50
        and cr<=0.05 and cr_hi<=0.05 and n_exp>0)
    if ok: return True,"admit"
    if tg_lo<=0 or th_lo<=0.50: return False,"precision"
    if th<=0.50: return False,"destination_point"
    if cr>0.05 or cr_hi>0.05: return False,"collateral"
    if n_exp==0: return False,"vacuous"
    return False,"other"
rows=[]
grid=list(itertools.product([30,50,75,100,150,200,300],[0.55,0.60,0.65,0.70,0.75],
                            [0.28,0.45],[0,11,50],[0.0,0.02]))
for n_err,p_hit,p_exit,n_exp,p_coll in grid:
    res=[one(n_err,p_exit,p_hit,n_exp,p_coll,RNG) for _ in range(120)]
    adm=sum(1 for a,_ in res if a)/len(res)
    reasons={}
    for a,r in res:
        if not a: reasons[r]=reasons.get(r,0)+1
    tg_true=p_exit*(2*p_hit-1)
    rows.append({"n_channel_error":n_err,"true_exit_rate":p_exit,"true_target_hit":p_hit,
      "implied_true_target_gain":round(tg_true,4),"n_exposed_correct":n_exp,"true_collateral":p_coll,
      "P_ADMIT":round(adm,4),"dominant_failure":max(reasons,key=reasons.get) if reasons else "none",
      "frac_precision_fail":round(reasons.get("precision",0)/len(res),4),
      "frac_collateral_fail":round(reasons.get("collateral",0)/len(res),4),
      "frac_vacuous":round(reasons.get("vacuous",0)/len(res),4)})
with open("LICENCE_OPERATING_CHARACTERISTICS.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f"grid cells: {len(rows)}")
# --- power counterfactuals at the observed effects ---
print("\n=== POWER COUNTERFACTUALS at each setting's observed point effect ===")
cases={"Qwen3-8B (ADMIT)":(87,52/87,38/52,6,0.0),
       "Qwen3-4B (DECLINE)":(124,64/124,37/64,50,1/50),
       "Gemma-2-9B (DECLINE)":(96,27/96,17/27,11,1/11)}
out={}
for name,(n,pe,ph,ne,pc) in cases.items():
    at_n=np.mean([one(n,pe,ph,ne,pc,RNG)[0] for _ in range(200)])
    need=None
    for m in [50,75,100,150,200,250,300,400,500,700,1000]:
        p=np.mean([one(m,pe,ph,max(ne,int(ne*m/n)),pc,RNG)[0] for _ in range(120)])
        if p>=0.80 and need is None: need=m
    out[name]={"observed_n":n,"exit_rate":round(pe,4),"target_hit":round(ph,4),
      "implied_target_gain":round(pe*(2*ph-1),4),"P_ADMIT_at_observed_n":round(float(at_n),3),
      "n_for_80pct_certification":need}
    print(f"  {name:22s} n={n:3d} exit={pe:.3f} t-hit={ph:.3f} TG={pe*(2*ph-1):+.4f} "
          f"P(ADMIT|observed n)={at_n:.3f}  n for 80% power: {need}")
json.dump(out,open("POWER_COUNTERFACTUALS.json","w"),indent=2)
