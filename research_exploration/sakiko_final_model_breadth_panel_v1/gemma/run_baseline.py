"""Gemma-2-9B-it - four-mode decision readout baseline over TRAIN + DEV.
SEALED is NOT touched. Frozen scoring semantics copied from the committed Qwen3 protocol."""
import json, os, sys, time
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np, torch
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from smoke_test_vram_v4 import (MODEL_DIR, RAW, MODES, L_OBS, set_determinism,
                                render_prompt_gemma, candidate_ids)
from transformers import AutoModelForCausalLM, AutoTokenizer

EXCLUDED = {217, 985, 1487, 2078, 2574, 3342}
ROOT="/root/autodl-tmp/sakiko-followup"

def score_candidate(model, device, pids, cids, capture_modules=None):
    ids = torch.tensor([pids + cids], dtype=torch.long, device=device)
    positions = torch.arange(len(pids)-1, len(pids)+len(cids)-1, device=device)
    captured={}; handles=[]
    for key, module in (capture_modules or {}).items():
        def hook(_m,_i,out,capture_key=key):
            t = out[0] if isinstance(out,tuple) else out
            captured[capture_key]=t[0, len(pids)-1].detach().float().cpu().numpy()
        handles.append(module.register_forward_hook(hook))
    try:
        with torch.inference_mode():
            logits = model(ids, use_cache=False, logits_to_keep=positions).logits[0]
            lp = torch.log_softmax(logits.float(), dim=-1)
            labels = torch.tensor(cids, dtype=torch.long, device=device)
            score = float(lp[torch.arange(len(cids), device=device), labels].mean().item())
    finally:
        for h in handles: h.remove()
    return score, captured

def main():
    torch.cuda.init(); det=set_determinism()
    tok=AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(MODEL_DIR, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation="eager", device_map="cuda:0")
    model.eval(); model.config.use_cache=False
    for p in model.parameters(): p.requires_grad_(False)
    device=next(model.parameters()).device
    obs_mod = model.model.layers[L_OBS].mlp

    perm=np.random.default_rng(42).permutation(3652)
    tr=json.load(open(f"{ROOT}/final/results/splits/train_idx.json"))
    dv=json.load(open(f"{ROOT}/final/results/splits/val_idx.json"))
    tr=tr if isinstance(tr,list) else tr["indices"]; dv=dv if isinstance(dv,list) else dv["indices"]
    split_of={}
    for p in tr: split_of[int(perm[p])]="train"
    for p in dv: split_of[int(perm[p])]="dev"
    rows=[json.loads(l) for l in open(RAW)]

    out=open(f"{HERE}/BASELINE_ROWS.jsonl","w")
    acts=[]; act_idx=[]
    n=0; t0=time.perf_counter()
    for raw_i, s in enumerate(rows):
        sp=split_of.get(raw_i)
        if sp is None or raw_i in EXCLUDED: continue
        prompt=render_prompt_gemma(tok, s)
        scores={}; cap=None
        for m in MODES:
            pids,cids=candidate_ids(tok,prompt,s["answers"][m])
            sc,c=score_candidate(model,device,pids,cids,
                                 capture_modules={"obs":obs_mod} if m==MODES[0] else None)
            scores[m]=sc
            if m==MODES[0]: cap=c["obs"]
        pred=max(MODES,key=lambda m:scores[m])
        srt=sorted(scores.values(),reverse=True)
        out.write(json.dumps({"raw_idx":raw_i,"uuid":s.get("uuid"),"split":sp,
            "gold":s["correct_answer"],"pred":pred,"correct":pred==s["correct_answer"],
            "scores":{m:round(scores[m],6) for m in MODES},
            "margin":round(srt[0]-srt[1],6),"prompt_tokens":len(tok.encode(prompt,add_special_tokens=False))})+"\n")
        acts.append(cap.astype(np.float32)); act_idx.append(raw_i)
        n+=1
        if n%250==0:
            el=time.perf_counter()-t0
            print(f"  {n} rows | {el:.0f}s | {el/n:.3f} s/row | eta {(3104-n)*el/n/60:.1f} min", flush=True)
    out.close()
    np.savez_compressed(f"{HERE}/BASELINE_LOBS_ACTIVATIONS.npz",
                        activations=np.stack(acts), raw_idx=np.array(act_idx))
    print(f"DONE {n} rows in {(time.perf_counter()-t0)/60:.1f} min")

if __name__=="__main__": main()
