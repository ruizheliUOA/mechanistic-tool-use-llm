"""Gemma TRAIN-only gradient-direction estimation at L_inj=24.
Estimator verbatim: d_grad = unit( mean_i unit( G_i,gold - G_i,source ) ),
G = 4-mode full-effect gradient stack (per-position gradient summed over prompt+candidate).
final_logit_softcapping stays ACTIVE. No compensation of any kind."""
import json, csv, os, sys, time, hashlib
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
import numpy as np, torch
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from smoke_test_vram_v4 import (MODEL_DIR, RAW, MODES, L_INJ, set_determinism,
                                render_prompt_gemma, candidate_ids, gradient_candidate)
from transformers import AutoModelForCausalLM, AutoTokenizer
R="/root/autodl-tmp/sakiko-followup"; TOL=1e-8
def sha_arr(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

torch.cuda.init(); set_determinism()
tok=AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
model=AutoModelForCausalLM.from_pretrained(MODEL_DIR, local_files_only=True,
    torch_dtype=torch.bfloat16, attn_implementation="eager", device_map="cuda:0")
model.eval(); model.config.use_cache=False
for p in model.parameters(): p.requires_grad_(False)
assert not any(p.requires_grad for p in model.parameters()), "parameter grad leak"
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False}); model.train(True)
dev=next(model.parameters()).device

rows=[json.loads(l) for l in open(f"{HERE}/BASELINE_ROWS.jsonl")]
v2={r['sample_id']:r['v2_split'] for r in csv.DictReader(open(f"{R}/final/results/qwen3_stage0_1_v2/QWEN3_V2_SPLIT_INDEX.csv"))}
for r in rows: r['split']=v2.get(r['uuid'])
raw=[json.loads(l) for l in open(RAW)]
router=json.load(open(f"{HERE}/ROUTER_RESULTS.json"))
qual=[c for c in json.load(open(f"{HERE}/CHANNEL_FREEZE.json"))["candidate_channels"]
      if router[c["channel"]]["router_eligible"]]
print("router-qualified:",[c["channel"] for c in qual], flush=True)

def G_of(sample):
    prompt=render_prompt_gemma(tok,sample); out=[]
    for m in MODES:
        pids,cids=candidate_ids(tok,prompt,sample["answers"][m])
        _,g,_=gradient_candidate(model,dev,pids,cids)
        out.append(g.sum(axis=0)); torch.cuda.empty_cache()
    return np.stack(out).astype(np.float32)

res={}
fh=open(f"{HERE}/DIRECTION_RECORDS.jsonl","w")
for c in qual:
    ch,gold,src=c["channel"],c["gold"],c["source"]
    gi,pi=MODES.index(gold),MODES.index(src)
    pop=sorted([r for r in rows if r['split']=='train' and r['gold']==gold and r['pred']==src],
               key=lambda r:r['uuid'])           # immutable UUID order
    assert len(pop)==c["train_error_support"], f"population drift {ch}: {len(pop)} != {c['train_error_support']}"
    W=[]; t0=time.perf_counter()
    for o,r in enumerate(pop):
        G=G_of(raw[r['raw_idx']])
        finite=bool(np.all(np.isfinite(G)))
        z=G[gi].astype(np.float64)-G[pi].astype(np.float64); nz=float(np.linalg.norm(z))
        st="VALID" if (finite and nz>TOL) else ("NON_FINITE_GRADIENT" if not finite else "CONTRAST_NORM_BELOW_TOLERANCE")
        if st!="VALID": raise RuntimeError(f"{st} for {r['uuid']}")
        w=z/nz; W.append(w)
        fh.write(json.dumps({"channel":ch,"sample_id":r['uuid'],"ordinal":o,"status":st,
            "target_contrast_norm":nz,"unit_contrast_norm":float(np.linalg.norm(w)),
            "per_mode_gradient_norms":{m:float(np.linalg.norm(G[k])) for k,m in enumerate(MODES)},
            "layer":L_INJ,"split":"train"})+"\n")
        if (o+1)%50==0: print(f"  {ch} {o+1}/{len(pop)} {time.perf_counter()-t0:.0f}s peak {torch.cuda.max_memory_allocated(0)/2**30:.2f} GiB",flush=True)
    Wm=np.stack(W); wbar=Wm.mean(0); wn=float(np.linalg.norm(wbar))
    d=(wbar/wn).astype(np.float32)
    assert abs(float(np.linalg.norm(d.astype(np.float64)))-1.0)<1e-6, "unit norm failure"
    res[ch]={"gold":gold,"source":src,"n":len(pop),"wbar_norm":wn,
        "chance_reference_one_over_sqrt_n":1.0/float(np.sqrt(len(pop))),
        "d_grad_norm":float(np.linalg.norm(d.astype(np.float64))),"d_grad_sha256":sha_arr(d),
        "population_sha256":hashlib.sha256("|".join(r['uuid'] for r in pop).encode()).hexdigest(),
        "target_contrast_norm_median":float(np.median([float(np.linalg.norm(x)) for x in W])),
        "layer":L_INJ,"hidden":int(d.shape[0]),"train_only":True,"dev_leakage":False}
    np.save(f"{HERE}/d_grad__{ch}.npy",d)
    print(f"  {ch}: n={len(pop)} wbar_norm={wn:.4f} (chance {res[ch]['chance_reference_one_over_sqrt_n']:.4f}) sha={res[ch]['d_grad_sha256'][:16]}",flush=True)
fh.close()
json.dump(res,open(f"{HERE}/DIRECTION_RESULTS.json","w"),indent=2,sort_keys=True)
print("DIRECTIONS DONE")
