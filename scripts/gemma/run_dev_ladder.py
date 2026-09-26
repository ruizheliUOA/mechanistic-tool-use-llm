"""Gemma frozen DEV ladder. L_obs=30 L_inj=24 WRONG_LAYER=L_obs=30.
h' = h + q*s_c*d at every sequence position of L. Q_GRID frozen, not extendable.
DEV population = actual 1101 rows."""
import json, csv, os, sys, time, hashlib, itertools
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
import numpy as np, torch
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from smoke_test_vram_v4 import (MODEL_DIR, RAW, MODES, set_determinism, render_prompt_gemma, candidate_ids)
from transformers import AutoModelForCausalLM, AutoTokenizer
R="/root/autodl-tmp/sakiko-followup"
L_OBS, L_INJ, WRONG_LAYER = 30, 24, 30
Q_GRID=[0.0,0.125,0.25,0.5,1.0,2.0]
DEV_RANDOM_SEEDS=[20260811,20260812,20260813,20260814,20260815,20260816,20260817,20260818]
GATE={"source_exits_min":10,"target_hit_min":0.50,"clean_collateral_max":0.05}
def sha_arr(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

torch.cuda.init(); set_determinism()
tok=AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
model=AutoModelForCausalLM.from_pretrained(MODEL_DIR, local_files_only=True,
    torch_dtype=torch.bfloat16, attn_implementation="eager", device_map="cuda:0")
model.eval(); model.config.use_cache=False
for p in model.parameters(): p.requires_grad_(False)
dev=next(model.parameters()).device

rows=[json.loads(l) for l in open(f"{HERE}/BASELINE_ROWS.jsonl")]
v2={r['sample_id']:r['v2_split'] for r in csv.DictReader(open(f"{R}/final/results/qwen3_stage0_1_v2/QWEN3_V2_SPLIT_INDEX.csv"))}
for r in rows: r['split']=v2.get(r['uuid'])
raw=[json.loads(l) for l in open(RAW)]
TR=[r for r in rows if r['split']=='train']; DV=[r for r in rows if r['split']=='dev']
print(f"TRAIN {len(TR)} DEV {len(DV)}",flush=True)
assert len(DV)==1101, f"DEV denominator {len(DV)} != 1101"

# ---------- capture L_inj activations (last prompt token) for all non-sealed rows ----------
CACHE=f"{HERE}/LINJ_ACTIVATIONS.npz"
if not os.path.exists(CACHE):
    acts={}; t0=time.perf_counter()
    mod=model.model.layers[L_INJ].mlp
    for n,r in enumerate(rows):
        s=raw[r['raw_idx']]; prompt=render_prompt_gemma(tok,s)
        pids,cids=candidate_ids(tok,prompt,s["answers"][MODES[0]])
        cap={}
        def hook(_m,_i,out,pl=len(pids)):
            t=out[0] if isinstance(out,tuple) else out
            cap['x']=t[0,pl-1].detach().float().cpu().numpy()
        h=mod.register_forward_hook(hook)
        try:
            with torch.inference_mode():
                model(torch.tensor([pids+cids],device=dev), use_cache=False,
                      logits_to_keep=torch.arange(len(pids)-1,len(pids),device=dev))
        finally: h.remove()
        acts[r['raw_idx']]=cap['x'].astype(np.float32)
        if (n+1)%400==0: print(f"  Linj cap {n+1}/{len(rows)} {time.perf_counter()-t0:.0f}s",flush=True)
    np.savez_compressed(CACHE, idx=np.array(list(acts)), act=np.stack(list(acts.values())))
    print("Linj capture done",flush=True)
z=np.load(CACHE); AI={int(i):a for i,a in zip(z['idx'],z['act'])}
zo=np.load(f"{HERE}/BASELINE_LOBS_ACTIVATIONS.npz"); AO={int(i):a for i,a in zip(zo['raw_idx'],zo['activations'])}

router=json.load(open(f"{HERE}/ROUTER_RESULTS.json"))
live=[c for c in json.load(open(f"{HERE}/CHANNEL_FREEZE.json"))["candidate_channels"]
      if router[c["channel"]]["router_eligible"]]

def rprob(ch, rs):
    z=np.load(f"{HERE}/router_{ch}.npz")
    X=np.stack([AO[r['raw_idx']] for r in rs]).astype(np.float64)
    return 1/(1+np.exp(-(((X-z['mean'])/z['scale'])@z['coef'].reshape(-1)+float(z['intercept'][0]))))

def score_int(sample, layer, delta):
    prompt=render_prompt_gemma(tok,sample); sc={}
    mod=model.model.layers[layer].mlp
    def hook(_m,_i,out):
        if delta is None: return out
        t=out[0] if isinstance(out,tuple) else out
        new=t+delta.to(dtype=t.dtype,device=t.device)
        return (new,)+out[1:] if isinstance(out,tuple) else new
    h=mod.register_forward_hook(hook)
    try:
        for m in MODES:
            pids,cids=candidate_ids(tok,prompt,sample["answers"][m])
            with torch.inference_mode():
                lg=model(torch.tensor([pids+cids],device=dev),use_cache=False,
                    logits_to_keep=torch.arange(len(pids)-1,len(pids)+len(cids)-1,device=dev)).logits[0]
                lp=torch.log_softmax(lg.float(),-1)
                lab=torch.tensor(cids,device=dev)
                sc[m]=float(lp[torch.arange(len(cids),device=dev),lab].mean().item())
    finally: h.remove()
    pred=max(MODES,key=lambda m:sc[m])
    return sc,pred

RECS=open(f"{HERE}/DEV_INTERVENTION_RECORDS.jsonl","a")
def arm(ch,gold,src,name,d,layer,absn,pop,order):
    delta=None
    if d is not None:
        d64=np.ascontiguousarray(d.astype(np.float64)); d64=d64/np.linalg.norm(d64)
        delta=torch.from_numpy((d64*absn).astype(np.float32)).to(dev)
        ach=float(torch.linalg.vector_norm(delta.double()).item())
        assert abs(ach-absn)/max(1.0,absn)<1e-5, "DOSE_MISMATCH"
    out={}
    for n,r in enumerate(pop):
        sc,pred=score_int(raw[r['raw_idx']],layer,delta)
        out[r['uuid']]=pred
        RECS.write(json.dumps({"channel":ch,"arm":name,"q":absn/ (SC[ch] or 1),"abs_delta_norm":absn,
            "layer":layer,"sample_id":r['uuid'],"gold":r['gold'],"pred_base":r['pred'],
            "pred_int":pred,"scores_int":{k:round(v,6) for k,v in sc.items()},
            "direction_sha256":(sha_arr(d) if d is not None else None),
            "destination":("SOURCE_RETAINED" if pred==src else "GOLD_ARRIVAL" if pred==gold else "OTHER_WRONG"),
            "batch_index":n,"global_execution_order":order},sort_keys=True)+"\n")
    RECS.flush()
    return out

SC={}
plan=json.load(open(f"{HERE}/DEV_PLAN.json")) if os.path.exists(f"{HERE}/DEV_PLAN.json") else {}
order=0
for c in live:
    ch,gold,src=c["channel"],c["gold"],c["source"]
    tau=router[ch]["tau"]
    tr_pop=[r for r in TR if r['pred']==src]
    tr_routed=[r for r,p in zip(tr_pop,rprob(ch,tr_pop)) if p>=tau]
    norms=np.linalg.norm(np.stack([AI[r['raw_idx']] for r in tr_routed]).astype(np.float64),axis=1)
    SC[ch]=float(np.median(norms))
    dv_pop=[r for r in DV if r['pred']==src]
    pr=rprob(ch,dv_pop); dv_routed=[r for r,p in zip(dv_pop,pr) if p>=tau]
    print(f"\n{ch}: tau={tau} s_c={SC[ch]:.6f} train_routed={len(tr_routed)} dev_pred_eq_source={len(dv_pop)} dev_routed={len(dv_routed)}",flush=True)
    dg=np.load(f"{HERE}/d_grad__{ch}.npy")
    err_tr=[r for r in TR if r['gold']==gold and r['pred']==src]
    ref_tr=[r for r in TR if r['gold']==gold and r['pred']==gold]
    def dm(A):
        v=np.stack([A[r['raw_idx']] for r in ref_tr]).mean(0)-np.stack([A[r['raw_idx']] for r in err_tr]).mean(0)
        return (v/np.linalg.norm(v)).astype(np.float32)
    d_obs, d_inj = dm(AO), dm(AI)
    json.dump({"s_c":SC[ch],"tau":tau,"train_routed":len(tr_routed),"dev_pred_eq_source":len(dv_pop),
        "dev_routed":len(dv_routed),"d_grad_sha":sha_arr(dg),"diffmean_Lobs_sha":sha_arr(d_obs),
        "diffmean_Linj_sha":sha_arr(d_inj),"absolute_delta_norm_by_q":{str(q):q*SC[ch] for q in Q_GRID}},
        open(f"{HERE}/DOSE_DECLARATION__{ch}.json","w"),indent=2,sort_keys=True)
    # zero arm + exact equality gate
    e0=arm(ch,gold,src,"zero",None,L_INJ,0.0,dv_routed,order); order+=1
    bad=[u for u,p in e0.items() if p!=next(r['pred'] for r in dv_routed if r['uuid']==u)]
    print(f"   zero-control exact equality: {'PASS' if not bad else 'FAIL n=%d'%len(bad)}",flush=True)
    for q in [x for x in Q_GRID if x>0]:
        arm(ch,gold,src,"d_grad",dg,L_INJ,q*SC[ch],dv_routed,order); order+=1
        arm(ch,gold,src,"d_Lobs_diffmean",d_obs,L_INJ,q*SC[ch],dv_routed,order); order+=1
        print(f"   q={q} calibration arms done",flush=True)
    np.save(f"{HERE}/d_diffmean_Lobs__{ch}.npy",d_obs); np.save(f"{HERE}/d_diffmean_Linj__{ch}.npy",d_inj)
RECS.close()
json.dump(SC,open(f"{HERE}/S_C.json","w"),indent=2)
print("DOSE CALIBRATION COMPLETE",flush=True)
