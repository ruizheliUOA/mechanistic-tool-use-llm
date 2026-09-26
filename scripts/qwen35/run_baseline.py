"""Qwen3.5-9B TRAIN+DEV baseline. Frozen modern four-mode readout reused verbatim.
L_obs=23 activations captured in the same pass. SEALED untouched."""
import os,json,csv,time,hashlib
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
import numpy as np, torch
from transformers import Qwen3_5ForCausalLM, AutoTokenizer
R="/root/autodl-tmp/sakiko-followup"; M="/root/autodl-tmp/models/qwen3.5-9b-c2022362"
MODES=["tool_call","direct","request_for_info","cannot_answer"]
L_OBS,L_INJ,MAX_TOKENS=23,18,8192
SYSTEM=("You are a helpful assistant.\n\n"
        "Given the user's question and the available tools, choose the most "
        "appropriate response from the provided options.")
torch.manual_seed(42); np.random.seed(42)
torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
torch.backends.cudnn.deterministic=True
tok=AutoTokenizer.from_pretrained(M,local_files_only=True)
model=Qwen3_5ForCausalLM.from_pretrained(M,local_files_only=True,dtype=torch.bfloat16,device_map={"":0})
model.eval(); model.config.use_cache=False
for p in model.parameters(): p.requires_grad_(False)
dev=next(model.parameters()).device
LAYERS=model.model.layers
def parse_tools(t):
    out=[]
    for x in t or []:
        if isinstance(x,str):
            try: out.append(json.loads(x))
            except json.JSONDecodeError: out.append({"raw":x})
        else: out.append(x)
    return out
def render_prompt(s):
    msgs=[{"role":"system","content":SYSTEM},{"role":"user","content":s["question"]}]
    text=tok.apply_chat_template(msgs,tools=parse_tools(s.get("tools")),tokenize=False,
                                 add_generation_prompt=True,enable_thinking=False)
    req="<|im_start|>assistant\n<think>\n\n</think>\n\n"
    if not text.endswith(req): raise RuntimeError("READOUT_INVALID_THINKING_MODE: suffix absent")
    if text.rsplit("<think>\n",1)[-1].split("\n</think>",1)[0]!="": raise RuntimeError("READOUT_INVALID_THINKING_MODE: nonempty reasoning")
    return text
def candidate_ids(prompt,cand):
    pids=tok.encode(prompt,add_special_tokens=False); cids=tok.encode(cand,add_special_tokens=False)
    if not pids or not cids: raise RuntimeError("empty tokens")
    if len(pids)+len(cids)>MAX_TOKENS: raise RuntimeError("STRUCTURAL_EXCLUSION_GT_8192")
    return pids,cids
def score_candidate(pids,cids,capture=None):
    ids=torch.tensor([pids+cids],dtype=torch.long,device=dev)
    positions=torch.arange(len(pids)-1,len(pids)+len(cids)-1,device=dev)
    cap={}; handles=[]
    for key,mod in (capture or {}).items():
        def hook(_m,_i,out,k=key):
            t=out[0] if isinstance(out,tuple) else out
            cap[k]=t[0,len(pids)-1].detach().float().cpu().numpy()
        handles.append(mod.register_forward_hook(hook))
    try:
        with torch.inference_mode():
            lg=model(ids,use_cache=False,logits_to_keep=positions).logits[0]
            lp=torch.log_softmax(lg.float(),dim=-1)
            lab=torch.tensor(cids,dtype=torch.long,device=dev)
            sc=float(lp[torch.arange(len(cids),device=dev),lab].mean().item())
    finally:
        for h in handles: h.remove()
    return sc,cap
raw=[json.loads(l) for l in open(f"{R}/data/processed/qwen25_7b_w2c/w2c_test_mcq.jsonl")]
v2={r['sample_id']:r['v2_split'] for r in csv.DictReader(open(f"{R}/final/results/qwen3_stage0_1_v2/QWEN3_V2_SPLIT_INDEX.csv"))}
todo=[(i,s) for i,s in enumerate(raw) if s['uuid'] in v2]
print(f"non-sealed rows: {len(todo)}",flush=True)
rows=[]; acts=[]; idxs=[]; excl=[]; t0=time.perf_counter()
fh=open("BASELINE_ROWS.jsonl","w")
for n,(i,s) in enumerate(todo):
    try:
        prompt=render_prompt(s)
        sc={}; cap0=None
        for m in MODES:
            pids,cids=candidate_ids(prompt,s["answers"][m])
            capture={"obs":LAYERS[L_OBS].mlp} if m==MODES[0] else None
            sc[m],c=score_candidate(pids,cids,capture)
            if m==MODES[0]: cap0=c["obs"]
        order=sorted(sc,key=lambda k:-sc[k])
        pred=order[0]; gold=s["correct_answer"]
        rank=order.index(gold)
        rec={"raw_idx":i,"uuid":s['uuid'],"split":v2[s['uuid']],"gold":gold,"pred":pred,
             "correct":pred==gold,"scores":{k:round(v,6) for k,v in sc.items()},
             "top1":order[0],"top2":order[1],"gold_rank":rank,
             "margin":round(sc[order[0]]-sc[order[1]],8),
             "gold_minus_pred":round(sc[gold]-sc[pred],8),
             "prompt_tokens":len(tok.encode(prompt,add_special_tokens=False))}
        fh.write(json.dumps(rec,sort_keys=True)+"\n"); rows.append(rec)
        acts.append(cap0.astype(np.float32)); idxs.append(i)
    except RuntimeError as e:
        excl.append({"raw_idx":i,"uuid":s['uuid'],"split":v2[s['uuid']],"reason":str(e)[:80]})
    except torch.cuda.OutOfMemoryError as e:
        excl.append({"raw_idx":i,"uuid":s['uuid'],"split":v2[s['uuid']],"reason":"CUDA_OOM"}); torch.cuda.empty_cache()
    if (n+1)%250==0:
        el=time.perf_counter()-t0
        print(f"  {n+1}/{len(todo)} {el:.0f}s {el/(n+1):.3f}s/row eta {(len(todo)-n-1)*el/(n+1)/60:.1f}min peak {torch.cuda.max_memory_allocated(0)/2**30:.2f}GiB",flush=True)
fh.close()
np.savez_compressed("BASELINE_LOBS_ACTIVATIONS.npz",raw_idx=np.array(idxs),activations=np.stack(acts))
json.dump(excl,open("BASELINE_EXCLUSIONS.json","w"),indent=2)
print(f"DONE rows={len(rows)} excluded={len(excl)} {(time.perf_counter()-t0)/60:.1f}min",flush=True)
