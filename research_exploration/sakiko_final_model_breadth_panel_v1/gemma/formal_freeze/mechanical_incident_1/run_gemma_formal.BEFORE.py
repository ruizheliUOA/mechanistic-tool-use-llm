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
    print("--run-formal requires the access marker, which is created only under explicit")
    print("human authorisation by a separate deliberate step. Refusing to self-authorise.")
    sys.exit(3)

if __name__ == "__main__":
    main()
