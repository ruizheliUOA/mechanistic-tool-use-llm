import os,json,time,traceback
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
import torch
from transformers import Qwen3_5ForCausalLM, AutoTokenizer
M="/root/autodl-tmp/models/qwen3.5-9b-c2022362"
R={"stage":None,"markers":[]}
def mem():
    return {"alloc_gb":round(torch.cuda.memory_allocated(0)/1e9,3),"reserved_gb":round(torch.cuda.memory_reserved(0)/1e9,3),
            "peak_alloc_gb":round(torch.cuda.max_memory_allocated(0)/1e9,3),"peak_reserved_gb":round(torch.cuda.max_memory_reserved(0)/1e9,3)}
try:
    torch.cuda.init(); torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    R["gpu"]=torch.cuda.get_device_name(0); R["gpu_total_gb"]=round(torch.cuda.get_device_properties(0).total_memory/1e9,3)
    R["mem_pre"]=mem()
    R["stage"]="weight_loading"; t0=time.perf_counter()
    tok=AutoTokenizer.from_pretrained(M,local_files_only=True)
    model=Qwen3_5ForCausalLM.from_pretrained(M,local_files_only=True,dtype=torch.bfloat16,device_map={"":0})
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    R["load_s"]=round(time.perf_counter()-t0,1); R["mem_after_load"]=mem()
    R["params_b"]=round(sum(p.numel() for p in model.parameters())/1e9,3)
    R["all_params_requires_grad_false"]=not any(p.requires_grad for p in model.parameters())
    R["dtypes"]=sorted({str(p.dtype) for p in model.parameters()})
    R["markers"].append("MODEL_LOAD_OK"); print("MODEL_LOAD_OK",flush=True)
    # locate text stack + L_inj module
    base=model.model
    lang=getattr(base,"language_model",base)
    R["text_stack_path"]="model.language_model" if hasattr(base,"language_model") else "model"
    layers=lang.layers; R["n_layers"]=len(layers)
    R["L_inj_module"]=type(layers[18].mlp).__name__
    R["L_obs_module"]=type(layers[23].mlp).__name__
    R["L_inj_mixer"]=type(getattr(layers[18],'linear_attn',getattr(layers[18],'self_attn',None))).__name__
    R["L_obs_mixer"]=type(getattr(layers[23],'self_attn',getattr(layers[23],'linear_attn',None))).__name__
    R["hidden_size"]=lang.config.hidden_size if hasattr(lang,'config') else model.config.text_config.hidden_size
    # tiny forward, use_cache=False
    R["stage"]="forward"; torch.cuda.reset_peak_memory_stats(0)
    msgs=[{"role":"user","content":"Reply with one short word."}]
    text=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    ids=tok(text,return_tensors="pt").to("cuda:0")
    with torch.inference_mode():
        out=model(**ids,use_cache=False)
    R["forward_logits_shape"]=list(out.logits.shape)
    R["forward_finite"]=bool(torch.isfinite(out.logits).all().item())
    R["mem_after_forward"]=mem()
    R["markers"].append("FORWARD_OK"); print("FORWARD_OK",flush=True)
    # gradient wrt L_inj leaf
    R["stage"]="gradient_leaf"; torch.cuda.reset_peak_memory_stats(0)
    leaf={}
    def hook(_m,_i,o):
        t=o[0] if isinstance(o,tuple) else o
        d=t.detach().requires_grad_(True); leaf["x"]=d
        return (d,)+o[1:] if isinstance(o,tuple) else d
    h=layers[18].mlp.register_forward_hook(hook)
    try:
        lg=model(**ids,use_cache=False).logits[0,-1]
        s=torch.log_softmax(lg.float(),-1).max()
        g=torch.autograd.grad(s,leaf["x"])[0]
        R["grad_shape"]=list(g.shape); R["grad_finite"]=bool(torch.isfinite(g).all().item())
        R["grad_l2"]=float(g.float().norm().item())
        R["leaf_requires_grad_only"]=leaf["x"].requires_grad
    finally: h.remove()
    R["mem_after_grad"]=mem()
    # deterministic 4-token generation
    R["stage"]="generate"; torch.cuda.reset_peak_memory_stats(0)
    with torch.inference_mode():
        g1=model.generate(**ids,max_new_tokens=4,do_sample=False,use_cache=True)
        g2=model.generate(**ids,max_new_tokens=4,do_sample=False,use_cache=True)
    R["gen_ids"]=g1[0,-4:].tolist(); R["gen_text"]=tok.decode(g1[0,ids['input_ids'].shape[1]:])
    R["deterministic_replay"]=bool(torch.equal(g1,g2))
    R["mem_after_generate"]=mem()
    R["markers"].append("GENERATE_OK"); print("GENERATE_OK",flush=True)
    ok=(R["forward_finite"] and R["grad_finite"] and R["deterministic_replay"] and R["all_params_requires_grad_false"])
    if ok: R["markers"].append("QWEN35_BF16_SMOKE_PASS"); print("QWEN35_BF16_SMOKE_PASS",flush=True)
    R["verdict"]="QWEN35_BF16_SMOKE_PASS" if ok else "QWEN35_BF16_SMOKE_FAIL"
    R["stage"]="complete"
except torch.cuda.OutOfMemoryError as e:
    R["verdict"]="QWEN35_HARDWARE_NO_GO"; R["oom_stage"]=R["stage"]; R["error"]=str(e)[:600]; R["mem_at_oom"]=mem()
    print("OOM at stage:",R["stage"],flush=True)
except Exception as e:
    R["verdict"]="QWEN35_BF16_SMOKE_ERROR"; R["error_stage"]=R["stage"]; R["error"]=f"{type(e).__name__}: {e}"; R["traceback"]=traceback.format_exc()[-1500:]
    print("ERROR at stage:",R["stage"],type(e).__name__,flush=True)
json.dump(R,open("QWEN35_BF16_SMOKE.json","w"),indent=2,default=str)
print("VERDICT:",R.get("verdict"))
