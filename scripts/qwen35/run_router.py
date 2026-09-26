import json,numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
GRID=[0.4,0.5,0.6,0.7,0.8]; ROC_MIN=0.75; PREC_MIN=0.50; SEED=42
rows=[json.loads(l) for l in open("BASELINE_ROWS.jsonl")]
z=np.load("BASELINE_LOBS_ACTIVATIONS.npz"); A=z['activations']; at={int(v):i for i,v in enumerate(z['raw_idx'])}
tr=[r for r in rows if r['split']=='train']; dv=[r for r in rows if r['split']=='dev']
def X(rs): return A[[at[r['raw_idx']] for r in rs]].astype(np.float64)
def boot(y,p,n=2000,seed=42):
    rng=np.random.default_rng(seed); y=np.asarray(y); p=np.asarray(p); o=[]
    for _ in range(n):
        i=rng.integers(0,len(y),len(y))
        if len(set(y[i]))>1: o.append(roc_auc_score(y[i],p[i]))
    return (float(np.percentile(o,2.5)),float(np.percentile(o,97.5)))
out={}
for c in [c for c in json.load(open("QWEN35_CHANNEL_DISCOVERY.json")) if c['support_eligible']]:
    ch,gold,src=c['channel'],c['gold'],c['source']
    pos=[r for r in tr if r['gold']==gold and r['pred']==src]
    neg=[r for r in tr if r['gold']==r['pred']]
    Xtr=X(neg+pos); ytr=np.array([0]*len(neg)+[1]*len(pos))
    sc=StandardScaler().fit(Xtr)
    clf=LogisticRegression(C=1.0,penalty="l2",solver="liblinear",max_iter=2000,tol=1e-4,
        random_state=SEED,class_weight=None,fit_intercept=True).fit(sc.transform(Xtr),ytr)
    assert int(clf.n_iter_[0])<2000,"Router did not converge"
    tra=float(roc_auc_score(ytr,clf.predict_proba(sc.transform(Xtr))[:,1]))
    dvp=[r for r in dv if r['gold']==gold and r['pred']==src]; dvc=[r for r in dv if r['gold']==r['pred']]
    comp=dvc+dvp; cy=np.array([0]*len(dvc)+[1]*len(dvp)); cp=clf.predict_proba(sc.transform(X(comp)))[:,1]
    roc=float(roc_auc_score(cy,cp)); pr=float(average_precision_score(cy,cp)); lo,hi=boot(cy,cp)
    op=[r for r in dv if r['pred']==src]; oy=np.array([int(r['gold']==gold) for r in op])
    opp=clf.predict_proba(sc.transform(X(op)))[:,1]
    grid=[]; tau=None
    for t in GRID:
        pd=opp>=t; tp=int((pd&(oy==1)).sum()); fp=int((pd&(oy==0)).sum())
        tn=int(((~pd)&(oy==0)).sum()); fn=int(((~pd)&(oy==1)).sum())
        prec=tp/(tp+fp) if tp+fp else 0.0; rec=tp/(tp+fn) if tp+fn else 0.0
        grid.append({"tau":t,"precision":round(prec,4),"recall":round(rec,4),"routed":int(pd.sum()),"tp":tp,"fp":fp,"tn":tn,"fn":fn})
        if tau is None and prec>=PREC_MIN: tau=t
    sel=next((g for g in grid if g['tau']==tau),None)
    elig=roc>=ROC_MIN and sel is not None and sel['precision']>=PREC_MIN
    fc=fo=0
    if tau is not None:
        allp=clf.predict_proba(sc.transform(X(dv)))[:,1]
        for r,p_ in zip(dv,allp):
            if p_>=tau:
                if r['gold']==r['pred']: fc+=1
                elif not (r['gold']==gold and r['pred']==src): fo+=1
    out[ch]={"train_pos":len(pos),"train_neg":len(neg),"train_auc":round(tra,4),
      "dev_pos":len(dvp),"dev_neg":len(dvc),"dev_roc_auc":round(roc,4),"dev_roc_auc_ci95":[round(lo,4),round(hi,4)],
      "dev_pr_auc":round(pr,4),"tau":tau,"selected":sel,"tau_grid":grid,"router_eligible":bool(elig),
      "fired_channel_errors":(sel or {}).get("tp"),"fired_baseline_correct_dev":fc,"fired_off_channel":fo,
      "operational_dev_rows":len(op),
      "decline_reason":None if elig else ("DEV_ROC_AUC_LT_0.75" if roc<ROC_MIN else "NO_TAU_PRECISION_GE_0.50")}
    np.savez(f"router_{ch}.npz",mean=sc.mean_,scale=sc.scale_,coef=clf.coef_[0],intercept=clf.intercept_)
    m=out[ch]
    print(f"\n{ch}")
    print(f"   TRAIN AUC {m['train_auc']:.4f} (pos {m['train_pos']}, neg {m['train_neg']})")
    print(f"   DEV ROC-AUC {m['dev_roc_auc']:.4f} CI {m['dev_roc_auc_ci95']} | PR-AUC {m['dev_pr_auc']:.4f}")
    print(f"   tau={m['tau']} selected={m['selected']}")
    print(f"   fired: ch-errors={m['fired_channel_errors']} baseline-correct={m['fired_baseline_correct_dev']} off-channel={m['fired_off_channel']}")
    print(f"   ROUTER_ELIGIBLE = {m['router_eligible']}" + (f"  [{m['decline_reason']}]" if not m['router_eligible'] else ""))
json.dump(out,open("QWEN35_ROUTER_METRICS.json","w"),indent=2,sort_keys=True)
