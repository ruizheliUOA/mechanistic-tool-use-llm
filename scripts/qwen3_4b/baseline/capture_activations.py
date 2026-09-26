#!/usr/bin/env python3
"""Activation-capture pass for Qwen3-4B TRAIN+DEV.

Stage A scored all 3,104 rows successfully but died on a path-formatting bug at the
activation-save line, losing the in-memory arrays. The baseline rows are intact and are
NOT recomputed.

The captured feature is the L_obs / L_inj MLP output at the FINAL PROMPT TOKEN. The prompt
is identical across the four candidates, so this state does not depend on which candidate
is being scored; the committed runner captures it during the first candidate's forward.
One forward per row is therefore exactly equivalent to what Stage A would have stored, and
costs a quarter of the compute.

Also runs the committed readout-validity pilot: 8 ascending-UUID TRAIN rows, 2 repetitions,
score replay atol 1e-5.
"""
import json, os, sys, time
from pathlib import Path
os.environ.setdefault("PYTHONHASHSEED","0"); os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
ROOT=Path("/root/autodl-tmp/sakiko-followup"); HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"scripts")); sys.path.insert(0,str(HERE))
import numpy as np, torch
import qwen3_8b_stage0_1 as V1
from run_qwen3_4b_stage_a import load_population, MODEL_DIR, L_OBS, L_INJ

rows,sealed,perm,raw=load_population()
for r in rows:
    if r["project_index"] in sealed: print("QWEN3_4B_TEST_CONTAMINATION"); raise SystemExit(2)
V1.set_determinism(); guard=V1.install_long_sequence_attention_guard()
from transformers import AutoModelForCausalLM, AutoTokenizer
tok=AutoTokenizer.from_pretrained(MODEL_DIR,local_files_only=True,trust_remote_code=False)
model=AutoModelForCausalLM.from_pretrained(MODEL_DIR,local_files_only=True,trust_remote_code=False,
        torch_dtype=torch.bfloat16,attn_implementation="eager",device_map="cuda:0").eval()
dev=next(model.parameters()).device; H=model.config.hidden_size
cap={"obs":model.model.layers[L_OBS].mlp,"inj":model.model.layers[L_INJ].mlp}

# --- committed readout-validity pilot: 8 rows, 2 reps, atol 1e-5 ---
sel=[r for r in sorted((x for x in rows if x["split"]=="train"), key=lambda x:x["sample_id"])[:8]]
reps=[]
for _ in range(2):
    reps.append([V1.score_sample(model,tok,dev,r["sample"],None)[0] for r in sel])
maxdev=max(abs(reps[0][i]["scores"][m]-reps[1][i]["scores"][m]) for i in range(8) for m in V1.MODES)
same_pred=all(reps[0][i]["prediction"]==reps[1][i]["prediction"] and
              reps[0][i]["runner_up"]==reps[1][i]["runner_up"] for i in range(8))
print("pilot: max score deviation %.3e | identical prediction+runner_up: %s"%(maxdev,same_pred))

A_obs=np.zeros((len(rows),H),dtype=np.float32); A_inj=np.zeros((len(rows),H),dtype=np.float32)
t0=time.perf_counter()
for i,r in enumerate(rows):
    prompt,_,_=V1.render_prompt(tok,r["sample"])
    pids,cids=V1.candidate_ids(tok,prompt,r["sample"]["answers"][V1.MODES[0]])
    _,c=V1.score_candidate(model,dev,pids,cids,cap)
    A_obs[i]=c["obs"]; A_inj[i]=c["inj"]
    if (i+1)%500==0: print("  %d/%d  %.0fs"%(i+1,len(rows),time.perf_counter()-t0),flush=True)
np.save(HERE/("QWEN3_4B_ACT_L%d.npy"%L_OBS),A_obs)
np.save(HERE/("QWEN3_4B_ACT_L%d.npy"%L_INJ),A_inj)
json.dump({"rows":len(rows),"hidden":H,"L_obs":L_OBS,"L_inj":L_INJ,
  "readout_validity_pilot":{"rows":8,"repetitions":2,"max_score_deviation":maxdev,
    "atol":1e-5,"within_atol":bool(maxdev<=1e-5),
    "prediction_and_runner_up_identical":bool(same_pred)},
  "attention_guard":guard,"elapsed_s":round(time.perf_counter()-t0,1)},
  open(HERE/"QWEN3_4B_ACTIVATION_MANIFEST.json","w",encoding="utf-8",newline="\n"),indent=2,sort_keys=True)
print("ACTIVATIONS_DONE %.0fs"%(time.perf_counter()-t0))
