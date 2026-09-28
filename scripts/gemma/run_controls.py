"""Gemma frozen DEV control battery at each channel's selected dose."""
import json, csv, os, sys, time, hashlib
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
import numpy as np, torch
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from smoke_test_vram_v4 import (MODEL_DIR, RAW, MODES, set_determinism, render_prompt_gemma, candidate_ids)
from transformers import AutoModelForCausalLM, AutoTokenizer
R="/root/autodl-tmp/sakiko-followup"
L_OBS,L_INJ,WRONG_LAYER=30,24,30
DEV_RANDOM_SEEDS=[20260811,20260812,20260813,20260814,20260815,20260816,20260817,20260818]
def sha_arr(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
torch.cuda.init(); set_determinism()
tok=AutoTokenizer.from_pretrained(MODEL_DIR,local_files_only=True)
model=AutoModelForCausalLM.from_pretrained(MODEL_DIR,local_files_only=True,
    torch_dtype=torch.bfloat16,attn_implementation="eager",device_map="cuda:0")
model.eval(); model.config.use_cache=False
for p in model.parameters(): p.requires_grad_(False)
dev=next(model.parameters()).device
rows=[json.loads(l) for l in open(f"{HERE}/BASELINE_ROWS.jsonl")]
v2={r['sample_id']:r['v2_split'] for r in csv.DictReader(open(f"{R}/final/results/qwen3_stage0_1_v2/QWEN3_V2_SPLIT_INDEX.csv"))}
for r in rows: r['split']=v2.get(r['uuid'])
DV=[r for r in rows if r['split']=='dev']
raw=[json.loads(l) for l in open(RAW)]
zo=np.load(f"{HERE}/BASELINE_LOBS_ACTIVATIONS.npz"); AO={int(i):a for i,a in zip(zo['raw_idx'],zo['activations'])}
router=json.load(open(f"{HERE}/ROUTER_RESULTS.json")); SC=json.load(open(f"{HERE}/S_C.json"))
sel=json.load(open(f"{HERE}/DOSE_SELECTION.json"))
def rprob(ch,rs):
    z=np.load(f"{HERE}/router_{ch}.npz")
    X=np.stack([AO[r['raw_idx']] for r in rs]).astype(np.float64)
    return 1/(1+np.exp(-(((X-z['mean'])/z['scale'])@z['coef'].reshape(-1)+float(z['intercept'][0]))))
def score_int(sample,layer,delta):
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
                lp=torch.log_softmax(lg.float(),-1); lab=torch.tensor(cids,device=dev)
                sc[m]=float(lp[torch.arange(len(cids),device=dev),lab].mean().item())
    finally: h.remove()
    return sc,max(MODES,key=lambda m:sc[m])
FH=open(f"{HERE}/DEV_CONTROL_RECORDS.jsonl","a")
def arm(ch,gold,src,name,d,layer,absn,pop,order):
    delta=None
    if d is not None:
        d64=np.ascontiguousarray(d.astype(np.float64)); d64=d64/np.linalg.norm(d64)
        delta=torch.from_numpy((d64*absn).astype(np.float32)).to(dev)
    t0=time.perf_counter()
    for n,r in enumerate(pop):
        sc,pred=score_int(raw[r['raw_idx']],layer,delta)
        FH.write(json.dumps({"channel":ch,"arm":name,"abs_delta_norm":absn,"layer":layer,
          "sample_id":r['uuid'],"gold":r['gold'],"pred_base":r['pred'],"pred_int":pred,
          "direction_sha256":(sha_arr(d) if d is not None else None),
          "destination":("SOURCE_RETAINED" if pred==src else "GOLD_ARRIVAL" if pred==gold else "OTHER_WRONG"),
          "batch_index":n,"global_execution_order":order},sort_keys=True)+"\n")
    FH.flush(); print(f"   {name:22s} n={len(pop):4d} {time.perf_counter()-t0:.0f}s",flush=True)
order=100
for ch in ["cannot_answer__to__tool_call","cannot_answer__to__direct"]:
    q=sel[ch]["selected_q"]
    if q is None: print(f"{ch}: no selected dose, controls skipped"); continue
    q=float(q); gold,src=ch.split("__to__"); tau=router[ch]["tau"]; absn=q*SC[ch]
    pop=[r for r in DV if r['pred']==src]
    routed=[r for r,p in zip(pop,rprob(ch,pop)) if p>=tau]
    print(f"\n{ch} q={q} absn={absn:.6f} routed={len(routed)} ungated_pop={len(pop)}",flush=True)
    dg=np.load(f"{HERE}/d_grad__{ch}.npy"); dinj=np.load(f"{HERE}/d_diffmean_Linj__{ch}.npy")
    arm(ch,gold,src,"reverse_d_grad",-dg,L_INJ,absn,routed,order); order+=1
    arm(ch,gold,src,"wrong_layer_d_grad",dg,WRONG_LAYER,absn,routed,order); order+=1
    arm(ch,gold,src,"d_Linj_diffmean_samelayer",dinj,L_INJ,absn,routed,order); order+=1
    for i,s in enumerate(DEV_RANDOM_SEEDS):
        v=np.random.default_rng(s).normal(size=dg.shape[0]); v=(v/np.linalg.norm(v)).astype(np.float32)
        arm(ch,gold,src,f"dev_random_{i}",v,L_INJ,absn,routed,order); order+=1
    arm(ch,gold,src,"ungated_d_grad",dg,L_INJ,absn,pop,order); order+=1
FH.close(); print("CONTROLS COMPLETE")
