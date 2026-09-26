#!/usr/bin/env python3
"""Stage D — TRAIN-only direction estimation for Qwen3-4B.

Primary estimator, recovered verbatim from the committed Qwen3-8B Stage 1.5:
    d_grad = unit( mean_i unit( G_i,gold - G_i,source ) )
with G_i,m the full-effect gradient of the mean candidate log-probability w.r.t. the
L_inj MLP output, summed over positions. No optimisation, no PCA/LDA/low-rank/nonlinear.

Prespecified diagnostic comparator: DiffMean at the same L_inj injection site
(mean(reference activations) - mean(error activations), L2-normalised), which removes the
estimator/site confound present in an L_obs DiffMean.

TRAIN only. DEV and SEALED are never touched here.
"""
import json,os,sys,time
from pathlib import Path
os.environ.setdefault("PYTHONHASHSEED","0"); os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
ROOT=Path("/root/autodl-tmp/sakiko-followup"); HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"scripts")); sys.path.insert(0,str(HERE))
import numpy as np, torch
import qwen3_8b_stage0_1 as V1
import qwen3_8b_stage0_1_v2 as V2
from run_qwen3_4b_stage_a import load_population, MODEL_DIR, L_INJ

CONTRAST_TOL=1e-8
CH=[("cannot_answer","tool_call"),("cannot_answer","direct")]

def gradient_object(model,tok,dev,sample):
    prompt,_,_=V1.render_prompt(tok,sample); G={}
    for m in V1.MODES:
        pids,cids=V1.candidate_ids(tok,prompt,sample["answers"][m])
        _s,grad,_st=V1.gradient_candidate(model,dev,pids,cids)
        G[m]=grad.sum(axis=0).astype(np.float32)
    return np.stack([G[m] for m in V1.MODES]).astype(np.float32)

rows,sealed,perm,raw=load_population()
base=[json.loads(l) for l in open(HERE/'QWEN3_4B_BASELINE_ROWS.jsonl')]
by={b['project_index']:b for b in base}
A_inj=np.load(HERE/'QWEN3_4B_ACT_L21.npy')
order={r['project_index']:i for i,r in enumerate(rows)}

V1.set_determinism(); V1.install_long_sequence_attention_guard()
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained(MODEL_DIR,local_files_only=True,trust_remote_code=False)
model=AutoModelForCausalLM.from_pretrained(MODEL_DIR,local_files_only=True,trust_remote_code=False,
        torch_dtype=torch.bfloat16,attn_implementation="eager",device_map="cuda:0").eval()
gc=V2.install_gradient_checkpointing(model,L_INJ)
# The committed Stage 1.5 installs BOTH memory strategies; the first attempt installed
# only gradient checkpointing and OOM'd in backward. Both are engineering-only: chunked
# eager attention performs the same computation with the same shapes and chunk size,
# checkpointing each chunk so only one chunk's probabilities are live at a time.
ca=V2.install_chunk_checkpointed_attention(V1)
dev=next(model.parameters()).device
MI={m:i for i,m in enumerate(V1.MODES)}
res={}
for g,s in CH:
    pop=[r for r in rows if r['split']=='train' and by[r['project_index']]['gold']==g
         and by[r['project_index']]['prediction']==s]
    print("channel %s->%s : %d TRAIN errors"%(g,s,len(pop)),flush=True)
    Z=[];t0=time.perf_counter();bad=0
    for k,r in enumerate(pop):
        G=gradient_object(model,tok,dev,r['sample'])
        z=G[MI[g]]-G[MI[s]]; n=float(np.linalg.norm(z))
        if not np.isfinite(n) or n<CONTRAST_TOL: bad+=1; continue
        Z.append(z/n)
        torch.cuda.empty_cache()
        if (k+1)%50==0: print("   %d/%d %.0fs"%(k+1,len(pop),time.perf_counter()-t0),flush=True)
    Z=np.stack(Z); wbar=Z.mean(0); wn=float(np.linalg.norm(wbar))
    d=(wbar/wn).astype(np.float32)
    # prespecified comparator: same-site (L_inj) DiffMean
    err=[order[r['project_index']] for r in pop]
    ref=[order[r['project_index']] for r in rows if r['split']=='train'
         and by[r['project_index']]['gold']==g and by[r['project_index']]['prediction']==g]
    dm=A_inj[ref].mean(0)-A_inj[err].mean(0); dm=(dm/np.linalg.norm(dm)).astype(np.float32)
    np.savez(HERE/('DIRECTION_%s__%s.npz'%(g,s)),d_grad=d,d_diffmean_Linj=dm)
    res["%s->%s"%(g,s)]={"n_train_errors":len(pop),"invalid_gradients":bad,
      "wbar_norm":wn,"chance_norm":1/np.sqrt(len(Z)),
      "concentration_ratio":wn/(1/np.sqrt(len(Z))),
      "d_grad_norm":float(np.linalg.norm(d.astype(np.float64))),
      "cos_dgrad_diffmean":float(d@dm),
      "n_reference":len(ref),"chunk_checkpointed_attention":ca,"elapsed_s":round(time.perf_counter()-t0,1)}
    print("  ||wbar||=%.6f chance=%.6f ratio=%.2f cos(d_grad,diffmean_Linj)=%.4f"%(
        wn,1/np.sqrt(len(Z)),res["%s->%s"%(g,s)]["concentration_ratio"],float(d@dm)),flush=True)
json.dump(res,open(HERE/'DIRECTION_ESTIMATION.json','w',encoding='utf-8',newline='\n'),indent=2,sort_keys=True)
print("DIRECTIONS_DONE")
