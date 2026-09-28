"""Gemma Router readability - frozen modern Router family, verbatim hyperparameters."""
import json, csv, hashlib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
R="/root/autodl-tmp/sakiko-followup"
ROUTER_GRID=[0.4,0.5,0.6,0.7,0.8]; ROUTER_ROC_MIN=0.75; ROUTER_PRECISION_MIN=0.50; SEED=42
rows=[json.loads(l) for l in open("BASELINE_ROWS.jsonl")]
v2={r['sample_id']:r['v2_split'] for r in csv.DictReader(open(f"{R}/final/results/qwen3_stage0_1_v2/QWEN3_V2_SPLIT_INDEX.csv"))}
for r in rows: r['split']=v2.get(r['uuid'])
z=np.load("BASELINE_LOBS_ACTIVATIONS.npz"); A=z['activations']; idx=list(z['raw_idx'])
at={int(v):i for i,v in enumerate(idx)}
tr=[r for r in rows if r['split']=='train']; dv=[r for r in rows if r['split']=='dev']
def X(rs): return A[[at[r['raw_idx']] for r in rs]].astype(np.float64)
def boot_auc(y,p,n=2000,seed=42):
    rng=np.random.default_rng(seed); out=[]
    y=np.asarray(y); p=np.asarray(p)
    for _ in range(n):
        i=rng.integers(0,len(y),len(y))
        if len(set(y[i]))<2: continue
        out.append(roc_auc_score(y[i],p[i]))
    return (float(np.percentile(out,2.5)),float(np.percentile(out,97.5))) if out else (float('nan'),)*2
out={}
for c in json.load(open("CHANNEL_FREEZE.json"))["candidate_channels"]:
    ch,gold,src=c["channel"],c["gold"],c["source"]
    pos=[r for r in tr if r['gold']==gold and r['pred']==src]
    neg=[r for r in tr if r['gold']==r['pred']]
    Xtr=X(neg+pos); ytr=np.array([0]*len(neg)+[1]*len(pos))
    sc=StandardScaler().fit(Xtr)
    clf=LogisticRegression(C=1.0,penalty="l2",solver="liblinear",max_iter=2000,tol=1e-4,
                           random_state=SEED,class_weight=None,fit_intercept=True).fit(sc.transform(Xtr),ytr)
    assert int(clf.n_iter_[0])<2000, f"Router did not converge {ch}"
    tr_auc=float(roc_auc_score(ytr,clf.predict_proba(sc.transform(Xtr))[:,1]))
    dvp=[r for r in dv if r['gold']==gold and r['pred']==src]
    dvc=[r for r in dv if r['gold']==r['pred']]
    comp=dvc+dvp; cy=np.array([0]*len(dvc)+[1]*len(dvp))
    cp=clf.predict_proba(sc.transform(X(comp)))[:,1]
    roc=float(roc_auc_score(cy,cp)); pr=float(average_precision_score(cy,cp)); lo,hi=boot_auc(cy,cp)
    op=[r for r in dv if r['pred']==src]
    oy=np.array([int(r['gold']==gold) for r in op]); op_=clf.predict_proba(sc.transform(X(op)))[:,1]
    grid=[]; tau=None
    for t in ROUTER_GRID:
        pd=op_>=t; tp=int((pd&(oy==1)).sum()); fp=int((pd&(oy==0)).sum())
        tn=int(((~pd)&(oy==0)).sum()); fn=int(((~pd)&(oy==1)).sum())
        prec=tp/(tp+fp) if tp+fp else 0.0; rec=tp/(tp+fn) if tp+fn else 0.0
        grid.append({"tau":t,"precision":round(prec,4),"recall":round(rec,4),"routed":int(pd.sum()),
                     "tp":tp,"fp":fp,"tn":tn,"fn":fn})
        if tau is None and prec>=ROUTER_PRECISION_MIN: tau=t
    sel=next((g for g in grid if g["tau"]==tau),None)
    elig=roc>=ROUTER_ROC_MIN and sel is not None and sel["precision"]>=ROUTER_PRECISION_MIN
    # exposure: dev baseline-correct rows the router would fire on at tau
    fired_correct=0; fired_offchannel=0
    if tau is not None:
        allp=clf.predict_proba(sc.transform(X(dv)))[:,1]
        for r,p_ in zip(dv,allp):
            if p_>=tau:
                if r['gold']==r['pred']: fired_correct+=1
                elif not (r['gold']==gold and r['pred']==src): fired_offchannel+=1
    out[ch]={"train_positive_n":len(pos),"train_negative_n":len(neg),"train_auc":round(tr_auc,4),
      "dev_positive_n":len(dvp),"dev_negative_n":len(dvc),"dev_comparable_ROC_AUC":round(roc,4),
      "dev_ROC_AUC_95CI":[round(lo,4),round(hi,4)],"dev_PR_AUC":round(pr,4),
      "tau":tau,"selected":sel,"tau_grid":grid,"router_eligible":bool(elig),
      "fired_channel_errors":(sel or {}).get("tp"),"fired_baseline_correct_dev":fired_correct,
      "fired_off_channel_errors":fired_offchannel,
      "operational_dev_rows_predicting_source":len(op),
      "decline_reason":None if elig else ("DEV_ROC_AUC_LT_0.75" if roc<ROUTER_ROC_MIN else "NO_TAU_WITH_PRECISION_GE_0.50")}
    np.savez(f"router_{ch}.npz",mean=sc.mean_,scale=sc.scale_,coef=clf.coef_[0],intercept=clf.intercept_)
json.dump(out,open("ROUTER_RESULTS.json","w"),indent=2,sort_keys=True)
for ch,m in out.items():
    print(f"\n{ch}")
    print(f"   TRAIN AUC {m['train_auc']:.4f} (pos {m['train_positive_n']}, neg {m['train_negative_n']})")
    print(f"   DEV comparable ROC-AUC {m['dev_comparable_ROC_AUC']:.4f}  95%CI {m['dev_ROC_AUC_95CI']}  PR-AUC {m['dev_PR_AUC']:.4f}")
    print(f"   tau={m['tau']}  selected={m['selected']}")
    print(f"   fired: channel-errors={m['fired_channel_errors']} baseline-correct(dev)={m['fired_baseline_correct_dev']} off-channel={m['fired_off_channel_errors']}")
    print(f"   ROUTER_ELIGIBLE = {m['router_eligible']}" + (f"   [{m['decline_reason']}]" if not m['router_eligible'] else ""))
print("\nrouter-eligible channels:",sum(m['router_eligible'] for m in out.values()),"of 3")
