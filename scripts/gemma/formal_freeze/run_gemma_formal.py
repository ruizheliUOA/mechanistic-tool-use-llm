#!/usr/bin/env python3
"""Gemma-2-9B one-shot SEALED formal runner. cannot_answer -> tool_call @ q=0.125.

Two phases: --gate-only (no SEALED access) and --run-formal (one shot).
SEALED is unreachable without a fsynced access marker. A second scientific
invocation is refused. Frozen constants are read from FORMAL_CONFIG.json and
verified by hash; nothing scientific is defined in this file.
"""
import argparse, hashlib, json, os, sys, time
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GEM = os.path.dirname(HERE)
REPO = "/root/autodl-tmp/sakiko-followup"
MARKER = os.path.join(HERE, "FORMAL_ACCESS_STARTED.json")
LEDGER = os.path.join(HERE, "ONE_SHOT_LEDGER.json")
OUT = os.path.join(HERE, "FORMAL_RECORDS.jsonl")

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def sha_arr(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def fail(code):
    print("GATE_FAIL: " + code)
    sys.exit(2)

def run_gate(phase):
    """phase in {'pre-access','formal-run'}; affects checks 12 and 13 ONLY."""
    cfg = json.load(open(os.path.join(HERE, "FORMAL_CONFIG.json")))
    rnd = json.load(open(os.path.join(HERE, "FORMAL_RANDOM_NULL.json")))
    hsh = json.load(open(os.path.join(HERE, "FORMAL_HASHES.json")))
    ck = []
    M = "/root/autodl-tmp/models/gemma-2-9b-it-11c9b309"
    for f, v in hsh["weight_shards"].items():
        if sha(os.path.join(M, f)) != v: fail("WEIGHT_SHARD_MISMATCH:" + f)
    ck.append("1 weight shards (4) match")
    if sha(os.path.join(M, "tokenizer.json")) != hsh["tokenizer"]["tokenizer.json"]: fail("TOKENIZER_MISMATCH")
    ck.append("2 tokenizer matches")
    if sha(os.path.join(M, "config.json")) != hsh["config"]: fail("MODEL_CONFIG_MISMATCH")
    ck.append("3 model config matches")
    d = np.load(os.path.join(GEM, "d_grad__cannot_answer__to__tool_call.npy"))
    if sha_arr(d) != cfg["direction"]["sha256"]: fail("DIRECTION_MISMATCH")
    if abs(float(np.linalg.norm(d.astype(np.float64))) - 1.0) > 1e-6: fail("DIRECTION_NOT_UNIT")
    ck.append("4 direction hash + unit norm")
    R = np.load(os.path.join(HERE, "formal_randoms_K59.npy"))
    if R.shape != (59, 3584): fail("RANDOM_SHAPE")
    if sha_arr(R) != rnd["matrix_sha256"]: fail("RANDOM_MATRIX_REPLACED")
    ck.append("5 K=59 random matrix hash")
    if set(rnd["seeds"]) & set(rnd["seed_disjointness"]["dev_development_seeds"]): fail("SEED_COLLISION_DEV")
    for k, v in rnd["seed_disjointness"]["prior_formal_matrices"].items():
        if v["seed_intersection"]: fail("SEED_COLLISION_PRIOR:" + k)
    ck.append("6 random seed disjointness")
    if sha(os.path.join(GEM, "router_cannot_answer__to__tool_call.npz")) != hsh["artifacts"]["router_cannot_answer__to__tool_call.npz"]:
        fail("ROUTER_MISMATCH")
    ck.append("7 Router hash")
    if cfg["dose"]["q"] != 0.125 or cfg["router"]["tau"] != 0.4: fail("DOSE_OR_TAU_DRIFT")
    if cfg["representation"]["L_inj"] != 24 or cfg["representation"]["L_obs"] != 30: fail("LAYER_DRIFT")
    ck.append("8 q/tau/layers frozen")
    ex = json.load(open(os.path.join(HERE, "FORMAL_EXCLUSION_MANIFEST.json")))
    if ex["mechanical_exclusions"]["sealed"] != 0 or ex["sealed"] != 548: fail("EXCLUSION_DRIFT")
    ck.append("9 exclusion manifest (0 SEALED)")
    cj = json.load(open(os.path.join(GEM, "DEV_CONJUNCTION.json")))["cannot_answer__to__tool_call"]
    if cj["n_pass"] != 14 or cj["decision"] != "FORMAL_ADVANCEMENT_ELIGIBLE": fail("DEV_CONJUNCTION_DRIFT")
    ck.append("10 DEV conjunction 14/14")
    inv = json.load(open(os.path.join(HERE, "FORMAL_ARM_INVENTORY.json")))
    if inv["total_formal_arms"] != 65: fail("ARM_INVENTORY_DRIFT")
    ck.append("11 arm inventory = 65")
    led = json.load(open(LEDGER))
    if phase == "pre-access":
        if os.path.exists(MARKER): fail("ACCESS_MARKER_ALREADY_EXISTS")
        if led["formal_run_count"] != 0 or led["scientific_endpoint_observed"] or led["sealed_rows_read"] != 0:
            fail("LEDGER_NOT_PRISTINE")
        ck.append("12 no access marker; ledger pristine")
        if os.path.exists(OUT): fail("FORMAL_RECORDS_ALREADY_EXIST")
        ck.append("13 no formal records exist")
    else:
        if not os.path.exists(MARKER): fail("ACCESS_MARKER_ABSENT")
        if led["formal_run_count"] != 0: fail("SECOND_SCIENTIFIC_INVOCATION_REFUSED")
        ck.append("12 access marker present; run_count still 0")
        ck.append("13 first scientific invocation")
    for c in ck: print("  PASS " + c)
    print("GATE_PASS (%d checks, phase=%s)" % (len(ck), phase))
    return cfg, rnd

def run_formal():
    """One shot. Gate -> marker (fsync) -> baseline -> Router -> 65 arms -> ledger."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, GEM)
    from smoke_test_vram_v4 import (MODEL_DIR, RAW, MODES, set_determinism,
                                    render_prompt_gemma, candidate_ids)
    cfg, rnd = run_gate("formal-run")
    L_INJ = cfg["representation"]["L_inj"]; L_OBS = cfg["representation"]["L_obs"]
    TAU = cfg["router"]["tau"]; ABSN = cfg["dose"]["absolute_delta_norm"]
    GOLD, SRC = cfg["gold"], cfg["source"]
    import csv
    v2 = {r["sample_id"] for r in csv.DictReader(
        open(REPO + "/final/results/qwen3_stage0_1_v2/QWEN3_V2_SPLIT_INDEX.csv"))}
    raw = [json.loads(l) for l in open(RAW)]
    sealed = [(i, s) for i, s in enumerate(raw) if s["uuid"] not in v2]
    if len(sealed) != 548:
        fail("SEALED_COUNT_MISMATCH:%d" % len(sealed))
    torch.cuda.init(); set_determinism()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation="eager", device_map="cuda:0")
    model.eval(); model.config.use_cache = False
    for p in model.parameters(): p.requires_grad_(False)
    dev = next(model.parameters()).device
    def score(sample, layer, delta, cap_obs=False):
        prompt = render_prompt_gemma(tok, sample); sc = {}; obs = {}
        mods = [model.model.layers[layer].mlp]
        def hk(_m, _i, out):
            if delta is None: return out
            t_ = out[0] if isinstance(out, tuple) else out
            nw = t_ + delta.to(dtype=t_.dtype, device=t_.device)
            return (nw,) + out[1:] if isinstance(out, tuple) else nw
        h = mods[0].register_forward_hook(hk)
        ho = None
        if cap_obs:
            def hko(_m, _i, out, pl=None):
                t_ = out[0] if isinstance(out, tuple) else out
                obs["x"] = t_[0, obs["pl"] - 1].detach().float().cpu().numpy()
            ho = model.model.layers[L_OBS].mlp.register_forward_hook(hko)
        try:
            for m in MODES:
                pids, cids = candidate_ids(tok, prompt, sample["answers"][m])
                obs["pl"] = len(pids)
                with torch.inference_mode():
                    lg = model(torch.tensor([pids + cids], device=dev), use_cache=False,
                        logits_to_keep=torch.arange(len(pids) - 1, len(pids) + len(cids) - 1,
                                                    device=dev)).logits[0]
                    lp = torch.log_softmax(lg.float(), -1)
                    lab = torch.tensor(cids, device=dev)
                    sc[m] = float(lp[torch.arange(len(cids), device=dev), lab].mean().item())
        finally:
            h.remove()
            if ho is not None: ho.remove()
        return sc, max(MODES, key=lambda m: sc[m]), obs.get("x")
    # ---- access marker, fsynced BEFORE the first SEALED row is loaded ----
    mk = {"authorised_head": cfg["repository_head"], "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
          "runner_sha256": sha(os.path.abspath(__file__)),
          "random_matrix_sha256": rnd["matrix_sha256"], "sealed_count_declared": 548}
    with open(MARKER, "w") as f:
        json.dump(mk, f, indent=2, sort_keys=True); f.flush(); os.fsync(f.fileno())
    print("ACCESS MARKER CREATED AND FSYNCED")
    led = json.load(open(LEDGER)); led["access_marker_created"] = True
    with open(LEDGER, "w") as f:
        json.dump(led, f, indent=2, sort_keys=True); f.flush(); os.fsync(f.fileno())
    # ---- SEALED baseline ----
    t0 = time.time(); rows = []; A = {}
    for n, (ri, s) in enumerate(sealed):
        sc, pred, ob = score(s, L_INJ, None, cap_obs=True)
        rows.append({"raw_idx": ri, "uuid": s["uuid"], "gold": s["correct_answer"],
                     "pred": pred, "scores": sc}); A[ri] = ob
        if (n + 1) % 100 == 0:
            print("  sealed baseline %d/548 %.0fs" % (n + 1, time.time() - t0), flush=True)
    led = json.load(open(LEDGER)); led["sealed_rows_read"] = len(rows)
    json.dump(led, open(LEDGER, "w"), indent=2, sort_keys=True)
    json.dump(rows, open(os.path.join(HERE, "SEALED_BASELINE.json"), "w"), indent=2, sort_keys=True)
    # ---- Router ----
    z = np.load(os.path.join(GEM, "router_cannot_answer__to__tool_call.npz"))
    pes = [r for r in rows if r["pred"] == SRC]
    X = np.stack([A[r["raw_idx"]] for r in pes]).astype(np.float64)
    pr = 1 / (1 + np.exp(-(((X - z["mean"]) / z["scale"]) @ z["coef"].reshape(-1) + float(z["intercept"][0]))))
    routed = [r for r, p in zip(pes, pr) if p >= TAU]
    rp = {r["uuid"]: float(p) for r, p in zip(pes, pr)}
    print("SEALED: pred==source %d, routed %d" % (len(pes), len(routed)), flush=True)
    dg = np.load(os.path.join(GEM, "d_grad__cannot_answer__to__tool_call.npy"))
    dmo = np.load(os.path.join(GEM, "d_diffmean_Lobs__cannot_answer__to__tool_call.npy"))
    Rm = np.load(os.path.join(HERE, "formal_randoms_K59.npy"))
    fh = open(OUT, "a", newline="\n")
    def arm(name, d, layer, pop, order):
        delta = None
        if d is not None:
            d64 = np.ascontiguousarray(d.astype(np.float64)); d64 = d64 / np.linalg.norm(d64)
            delta = torch.from_numpy((d64 * ABSN).astype(np.float32)).to(dev)
            if abs(float(torch.linalg.vector_norm(delta.double()).item()) - ABSN) / max(1.0, ABSN) > 1e-5:
                fail("DOSE_MISMATCH")
        for n, r in enumerate(pop):
            sc, pred, _ = score(raw[r["raw_idx"]], layer, delta)
            fh.write(json.dumps({"arm": name, "uuid": r["uuid"], "raw_idx": r["raw_idx"],
                "gold": r["gold"], "pred_base": r["pred"], "scores_base": r["scores"],
                "pred_int": pred, "scores_int": sc, "router_probability": rp.get(r["uuid"]),
                "router_decision": r in routed, "layer": layer, "abs_delta_norm": ABSN,
                "direction_sha256": (sha_arr(d) if d is not None else None),
                "destination": ("SOURCE_RETAINED" if pred == SRC else
                                "GOLD_ARRIVAL" if pred == GOLD else "OTHER_WRONG"),
                "batch_index": n, "global_execution_order": order}, sort_keys=True) + "\n")
        fh.flush(); os.fsync(fh.fileno())
        print("  arm %-22s n=%d" % (name, len(pop)), flush=True)
    o = 0
    arm("zero", None, L_INJ, routed, o); o += 1
    for nm, d, ly, pp in [("real_d_grad", dg, L_INJ, routed),
                          ("d_Lobs_diffmean", dmo, L_INJ, routed),
                          ("reverse_d_grad", -dg, L_INJ, routed),
                          ("wrong_layer_d_grad", dg, L_OBS, routed),
                          ("ungated_d_grad", dg, L_INJ, pes)]:
        arm(nm, d, ly, pp, o); o += 1
    for i in range(59):
        arm("formal_random_%d" % i, Rm[i], L_INJ, routed, o); o += 1
    fh.close()
    led = json.load(open(LEDGER))
    led.update({"formal_run_count": 1, "scientific_endpoint_observed": True,
                "arms_executed": 65, "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    with open(LEDGER, "w") as f:
        json.dump(led, f, indent=2, sort_keys=True); f.flush(); os.fsync(f.fileno())
    print("FORMAL EXECUTION COMPLETE: 65 arms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-only", action="store_true")
    ap.add_argument("--run-formal", action="store_true")
    a = ap.parse_args()
    if a.gate_only == a.run_formal:
        print("choose exactly one of --gate-only / --run-formal"); sys.exit(2)
    if a.gate_only:
        run_gate("pre-access")
        print("\nSEALED NOT ACCESSED. No marker created. Authorisation still required.")
        return
    run_formal()

if __name__ == "__main__":
    main()
