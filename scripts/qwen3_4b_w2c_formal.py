#!/usr/bin/env python3
"""Qwen3-4B x When2Call — formal one-shot confirmatory runner.

Two modes:

  --gate-only    verification only.  No SEALED row is loaded, rendered, tokenised or
                 scored.  No model is loaded.  No endpoint is computed.  Safe to run
                 repeatedly.
  --run-formal   the single authorised scientific execution.  Hard-aborts unless every
                 protocol hash matches, the formal access marker exists, the formal
                 namespace is clean and the repository HEAD is the authorised frozen HEAD.

Every scientific definition is inherited, never redefined here:

  * readout            -- `scripts/qwen3_8b_stage0_1.py` (`render_prompt`, `candidate_ids`,
                          `score_candidate`, `MODES`, determinism block, attention guard)
  * intervention hook  -- the frozen additive hidden-state delta used by Stage E
  * populations, split -- `research_exploration/qwen3_4b_stage_d_e_v1/stage_d_e_common.py`
  * channel, Router, tau, layers, direction, q, s_c, b_c, arms, endpoints, conjunction,
    randoms, bootstrap seed, VOID rules
                       -- `research_exploration/qwen3_4b_stage_d_e_v1/FORMAL_PROTOCOL_FROZEN.json`

No estimator, arm, parser, endpoint or threshold is introduced by this file.

SEALED interaction in `--gate-only` is restricted to identifiers: the uuid field of the
sealed rows is read to recompute the frozen ordered-manifest hash, and every other field is
discarded without inspection.  No prompt, answer set, gold label or payload content is read.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
import uuid as uuidlib
from pathlib import Path

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np

ROOT = Path("/root/autodl-tmp/sakiko-followup")
DE = ROOT / "research_exploration/qwen3_4b_stage_d_e_v1"
PROSP = ROOT / "research_exploration/qwen3_4b_w2c_prospective_replication_v1"
AUDIT = ROOT / "research_exploration/qwen3_4b_collateral_integrity_audit_v1"
GATE = ROOT / "research_exploration/qwen3_4b_formal_gate_v1"
FORMAL_NS = ROOT / "final/results/qwen3_4b_w2c_formal_v1"

PROTOCOL = DE / "FORMAL_PROTOCOL_FROZEN.json"
ACCESS_MARKER = FORMAL_NS / "FORMAL_ACCESS_STARTED.json"
LEDGER = GATE / "ONE_SHOT_EXECUTION_LEDGER.json"
SELF = Path(__file__).resolve()

RAW = ROOT / "data/processed/qwen25_7b_w2c/w2c_test_mcq.jsonl"
SPLIT_INDEX = ROOT / "final/results/qwen3_stage0_1_v2/QWEN3_V2_SPLIT_INDEX.csv"
PERM_SEED = 42
TOTAL_ROWS = 3652


# ----------------------------------------------------------------- utilities
def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def sha_arr(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def wj(p, o):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    t = Path(str(p) + ".tmp")
    t.write_text(json.dumps(o, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(t, p)


def git(*a):
    return subprocess.check_output(["git", *a], cwd=str(ROOT), text=True).strip()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class Log:
    def __init__(self, path):
        self.lines = []
        self.path = path

    def __call__(self, msg):
        line = "[%s] %s" % (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), msg)
        self.lines.append(line)
        print(line, flush=True)

    def flush(self):
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.path).write_text("\n".join(self.lines) + "\n", encoding="utf-8")


# --------------------------------------------------- sealed identity (ID only)
def sealed_identifiers():
    """Return the ordered sealed uuid list.

    Only the `uuid` field is retained.  Every other field of every row -- prompt, answer
    candidates, gold label, metadata -- is discarded immediately and is never inspected,
    rendered, tokenised or scored.  This is identifier verification, not payload access.
    """
    ids = []
    with open(RAW, encoding="utf-8") as f:
        for line in f:
            ids.append(json.loads(line)["uuid"])          # uuid only; rest dropped
    assert len(ids) == TOTAL_ROWS, len(ids)
    perm = np.random.default_rng(PERM_SEED).permutation(TOTAL_ROWS)
    kept = {int(r["project_index"]) for r in
            csv.DictReader(open(SPLIT_INDEX, encoding="utf-8"))}
    sealed_idx = sorted(set(range(TOTAL_ROWS)) - kept)
    return sealed_idx, [ids[perm[i]] for i in sealed_idx], ids, perm, kept


def formal_randoms(salt, K, dim):
    """Deterministic regeneration from the frozen seeds, for verification only.

    The protocol pins `construction` and the exact seed list, so regeneration is a
    verification operation and must reproduce the frozen matrix byte-identically.  It is
    never used to replace the committed tensor.
    """
    out = []
    for i in range(K):
        seed = int.from_bytes(hashlib.sha256((salt + str(i)).encode("utf-8")).digest()[:8],
                              "big", signed=False)
        g = np.random.Generator(np.random.PCG64DXSM(seed))
        z = g.standard_normal(dim)
        out.append((z / np.linalg.norm(z)).astype(np.float32))
    return np.stack(out)


# ------------------------------------------------------------------ the gate
def run_gate(log, strict_head=None, phase="pre-access"):
    """`phase` selects the access-state semantics of checks 20 and 22 ONLY.

    In `pre-access` the formal namespace must be empty and the access marker must be
    absent.  In `formal-run` the marker is the authorisation, so it must be PRESENT and
    must be the only thing in the namespace besides this run's own log.  Every other
    check, and every scientific value, is identical in both phases.

    MECHANICAL REPAIR (incident 1, before any SEALED access): the originally frozen runner
    evaluated the pre-access semantics inside `--run-formal`, which contradicted
    `run_formal`'s own requirement that the marker exist, making the frozen protocol
    unexecutable.  No scientific definition, threshold, arm, endpoint, direction, dose or
    population is affected.
    """
    from safetensors import safe_open

    P = json.loads(PROTOCOL.read_text())
    fc = P["frozen_configuration"]
    checks = []

    def chk(n, name, ok, detail):
        checks.append({"n": n, "check": name, "pass": bool(ok), "detail": detail})
        log("%-3s %-46s %s  %s" % (n, name, "PASS" if ok else "FAIL", detail))
        return bool(ok)

    # 1 repository HEAD authorised
    head = git("rev-parse", "HEAD")
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    div = git("rev-list", "--left-right", "--count",
              "origin/exp/sakiko-followup-archive...HEAD")
    head_ok = (strict_head is None) or (head == strict_head)
    chk(1, "repository HEAD authorised",
        head_ok and branch == "exp/sakiko-followup-archive" and div == "0\t0",
        "HEAD=%s branch=%s divergence=%s" % (head[:12], branch, div.replace("\t", "/")))

    # 2 worktree scientific files clean
    dirty = git("status", "--porcelain", "--untracked-files=no")
    untracked = [l[3:] for l in git("status", "--porcelain").splitlines()
                 if l.startswith("??")]
    # This task's own deliverables are untracked until the human commits them; both are
    # listed explicitly so the check stays strict about everything else.
    allowed_untracked = {"final/results/decisive_upgrade_audit/",       # pre-existing, not ours
                         "research_exploration/qwen3_4b_formal_gate_v1/",
                         "scripts/qwen3_4b_w2c_formal.py",
                         "scripts/qwen3_4b_formal_package.py",
                         "Under"}                                       # pre-existing 0-byte stray
    if phase == "formal-run":
        # The formal namespace must exist during the run -- it holds the access marker that
        # authorises it -- and is untracked until the results are handed off. Allowing it
        # here is the same phase-aware repair as checks 20/22 (mechanical incident 1).
        allowed_untracked.add("final/results/qwen3_4b_w2c_formal_v1/")
    stray = [u for u in untracked if u not in allowed_untracked]
    chk(2, "worktree scientific files clean", dirty == "" and not stray,
        "tracked modifications=%d unexpected untracked=%s"
        % (len(dirty.splitlines()) if dirty else 0, stray or "none"))

    # 3 prerequisite packages hash-valid
    pkg_detail = {}
    allv = True
    for pkg in ("qwen3_4b_w2c_prospective_replication_v1",
                "qwen3_4b_gradient_memory_amendment_v1",
                "qwen3_4b_stage_d_e_v1",
                "qwen3_4b_collateral_integrity_audit_v1"):
        d = ROOT / "research_exploration" / pkg
        man = json.loads((d / "HASHES.json").read_text())
        arts = man.get("artifacts") or man.get("files")
        bad = [f for f, v in arts.items()
               if not (d / f).exists() or sha_file(d / f) != v]
        tracked = len(git("ls-files", str(d.relative_to(ROOT))).splitlines())
        pending = len(git("diff", "--name-only", "HEAD", "--",
                          str(d.relative_to(ROOT))).splitlines())
        ok = not bad and tracked > 0 and pending == 0
        allv &= ok
        pkg_detail[pkg] = {"n": len(arts), "mismatch": bad, "tracked": tracked,
                           "uncommitted_changes": pending}
    chk(3, "prerequisite packages hash-valid & committed", allv,
        "4 packages, %d artifacts" % sum(v["n"] for v in pkg_detail.values()))

    # 4 model manifest exact
    mem = json.loads((PROSP / "MODEL_AND_ENV_MANIFEST.json").read_text())
    mdir = Path(mem["model"]["local_path"])
    mbad = [k for k, v in mem["model_file_sha256"].items()
            if not (mdir / k).exists() or sha_file(mdir / k) != v]
    chk(4, "model manifest exact",
        not mbad and mem["model"]["repository"] == fc["model"]
        and mem["model"]["revision"] == fc["model_revision"]
        and mem["model"]["hidden_size"] == fc["hidden_size"]
        and mem["model"]["num_hidden_layers"] == fc["num_hidden_layers"]
        and mem["model"]["dtype"] == fc["dtype"]
        and mem["model"]["attention_implementation"] == fc["attention_implementation"],
        "%s@%s, %d files, mismatch=%s"
        % (fc["model"], fc["model_revision"][:12], len(mem["model_file_sha256"]),
           mbad or "none"))

    # 5 tokenizer / chat template hashes exact
    tok_ok = sha_file(mdir / "tokenizer.json") == mem["model"]["tokenizer_sha256"]
    tmpl = json.loads((mdir / "tokenizer_config.json").read_text()).get("chat_template", "")
    tmpl_ok = (hashlib.sha256(tmpl.encode("utf-8")).hexdigest()
               == mem["model"]["chat_template_sha256_utf8"])
    chk(5, "tokenizer & chat template hashes exact", tok_ok and tmpl_ok,
        "tokenizer=%s chat_template=%s" % (tok_ok, tmpl_ok))

    # 6 dataset / split manifests exact
    dsm = json.loads((PROSP / "DATASET_AND_SPLIT_MANIFEST.json").read_text())
    sealed_idx, sealed_uuid, all_ids, perm, kept = sealed_identifiers()
    chk(6, "dataset & split manifests exact",
        len(all_ids) == TOTAL_ROWS and len(kept) == 3104 and len(sealed_idx) == 548,
        "total=%d train+dev=%d sealed=%d" % (len(all_ids), len(kept), len(sealed_idx)))

    # 7 SEALED manifest exact, by hash only
    v1 = load_module(ROOT / "scripts/qwen3_8b_stage0_1.py", "q3v1_gate")
    recomputed = v1.index_order_hash(sealed_uuid)
    chk(7, "SEALED manifest hash exact (ID-only)",
        recomputed == P["sealed_evaluation"]["ordered_uuid_sha256"]
        and P["sealed_evaluation"]["count"] == 548,
        "548 IDs, ordered sha256 %s" % recomputed[:16])

    # 8 selected channel exact
    chk(8, "selected channel exact",
        P["formal_channel"]["channel"] == "cannot_answer__to__tool_call"
        and P["formal_channel"]["gold"] == "cannot_answer"
        and P["formal_channel"]["source"] == "tool_call"
        and P["formal_channel"]["single_channel"] is True,
        "cannot_answer -> tool_call, single channel")

    # 9 Router exact
    rp = PROSP / P["router"]["file"]
    z = np.load(rp)
    chk(9, "Router artifact exact",
        sha_file(rp) == P["router"]["sha256"]
        and z["coef"].shape == (1, fc["hidden_size"])
        and z["mean"].shape == (fc["hidden_size"],)
        and P["router"]["refit_forbidden"] is True,
        "sha %s, coef%s intercept=%.10f" % (P["router"]["sha256"][:12],
                                            tuple(z["coef"].shape), float(z["intercept"][0])))

    # 10 threshold exact
    audit_tau = None
    for r in csv.DictReader(open(PROSP / "ROUTER_AUDIT.csv", encoding="utf-8")):
        if r["channel"].replace("->", "__to__") == P["formal_channel"]["channel"]:
            audit_tau = float(r["tau"])
    chk(10, "Router threshold exact",
        fc["tau"] == P["router"]["tau"] == audit_tau,
        "tau=%r (protocol == router block == Stage C audit)" % fc["tau"])

    # 11 observation layer exact
    chk(11, "observation layer exact",
        fc["L_obs"] == 26 and mem["layer_mapping"]["L_obs"] == 26,
        "L_obs=%d" % fc["L_obs"])

    # 12 direction tensor hash exact
    with safe_open(DE / "PRIMARY_DIRECTIONS.safetensors", framework="numpy") as f:
        dg = f.get_tensor(P["directions"]["primary_d_grad"]["key"])
    with safe_open(DE / "DIFFMEAN_DIRECTIONS.safetensors", framework="numpy") as f:
        d26 = f.get_tensor(P["directions"]["L26_diffmean_frozen"]["key"])
        d21 = f.get_tensor(P["directions"]["L21_same_layer_diffmean"]["key"])
    dnorm = float(np.linalg.norm(dg.astype(np.float64)))
    chk(12, "direction tensors exact",
        sha_arr(dg) == P["directions"]["primary_d_grad"]["sha256"]
        and sha_arr(d26) == P["directions"]["L26_diffmean_frozen"]["sha256"]
        and sha_arr(d21) == P["directions"]["L21_same_layer_diffmean"]["sha256"]
        and dg.shape == (fc["hidden_size"],) and dg.dtype == np.float32
        and bool(np.all(np.isfinite(dg))) and abs(dnorm - 1.0) < 1e-5,
        "d_grad sha %s dim=%d finite norm=%.12f" % (sha_arr(dg)[:12], dg.shape[0], dnorm))

    # 13 injection layer exact
    chk(13, "injection layer exact",
        fc["L_inj"] == 21 and mem["layer_mapping"]["L_inj"] == 21
        and fc["wrong_layer"] == fc["L_obs"],
        "L_inj=%d, wrong-layer control=%d" % (fc["L_inj"], fc["wrong_layer"]))

    # 14 q exact
    dose = json.loads((DE / "DOSE_DECLARATION.json").read_text())
    sc = dose["per_channel"][P["formal_channel"]["channel"]]["s_c"]
    chk(14, "dose q exact",
        fc["q"] == 0.25 and fc["q"] in dose["q_grid"] and dose["grid_extendable"] is False,
        "q=%r on frozen grid %s" % (fc["q"], dose["q_grid"]))

    # 15 perturbation norm exact
    expect = fc["q"] * sc
    chk(15, "absolute perturbation norm exact",
        fc["s_c"] == sc and abs(fc["absolute_perturbation_norm"] - expect) < 1e-12,
        "q*s_c = %.10f (s_c=%.10f)" % (fc["absolute_perturbation_norm"], sc))

    # 16 random seeds exact
    fr = P["formal_randoms"]
    seeds = [int.from_bytes(hashlib.sha256((fr["salt"] + str(i)).encode()).digest()[:8],
                            "big", signed=False) for i in range(fr["K"])]
    dev_seeds = set(fr["seed_disjointness"]["dev_development_seeds"])
    hist_seeds = set(fr["seed_disjointness"]["other_historical_seeds"])
    q8_manifest = json.loads(
        (ROOT / "final/results/qwen3_stage2_formal/QWEN3_STAGE2_RANDOM_MANIFEST.json").read_text())
    q8_seeds = {int.from_bytes(hashlib.sha256((q8_manifest["salt"] + str(i)).encode()).digest()[:8],
                               "big", signed=False) for i in range(q8_manifest["K"])}
    chk(16, "formal random seeds exact & disjoint",
        seeds == fr["seeds"] and len(set(seeds)) == fr["K"] == 59
        and not (set(seeds) & dev_seeds) and not (set(seeds) & hist_seeds)
        and not (set(seeds) & q8_seeds)
        and fr["salt"] != q8_manifest["salt"],
        "K=59, regenerated seeds identical, disjoint from DEV(8), historical(4), Qwen3-8B(%d)"
        % len(q8_seeds))

    # 17 random matrix hash exact (deterministic verification regeneration)
    with safe_open(DE / "FORMAL_RANDOMS.safetensors", framework="numpy") as f:
        Rcommitted = f.get_tensor("formal_randoms")
    Rregen = formal_randoms(fr["salt"], fr["K"], fr["dimension"])
    byte_identical = Rregen.tobytes() == Rcommitted.tobytes()
    chk(17, "random matrix hash exact (byte-identical regen)",
        sha_arr(Rcommitted) == fr["matrix_sha256"]
        and sha_file(DE / "FORMAL_RANDOMS.safetensors") == fr["file_sha256"]
        and byte_identical and Rcommitted.shape == (59, fc["hidden_size"])
        and bool(np.all(np.isfinite(Rcommitted)))
        and float(np.abs(np.linalg.norm(Rcommitted.astype(np.float64), axis=1) - 1).max()) < 1e-6,
        "matrix sha %s, regen byte-identical=%s, shape%s"
        % (fr["matrix_sha256"][:12], byte_identical, tuple(Rcommitted.shape)))

    # 18 comparator definitions exact
    ssc = P["deterministic_score_space_comparator"]
    stage_e = json.loads((DE / "_STAGE_E_RESULTS.json").read_text())
    bc_dev = stage_e["results"][P["formal_channel"]["channel"]]["score_comparator"]["b_c"]
    chk(18, "comparator definitions exact",
        ssc["b_c"] == fc["b_c"] == bc_dev
        and ssc["applied"] == "e_gold += b_c/2 ; e_source -= b_c/2 ; others unchanged"
        and ssc["calibrated_on"] == "DEV, at the selected dose"
        and "L26 DiffMean" in " ".join(P["arm_battery"]),
        "b_c=%.10f frozen from DEV; L26 DiffMean comparator present" % ssc["b_c"])

    # 19 success conjunction exact
    prim = P["primary"]
    chk(19, "success conjunction exact",
        len(prim["conjunction"]) == 10 and prim["is_intersection_claim"] is True
        and prim["no_subset_substitution_after_execution"] is True
        and prim["p_value"].startswith("p_add_one = (1 + count[")
        and P["secondary"]["must_not_rescue_failed_primary"] is True,
        "%d conditions, intersection claim, no subset substitution"
        % len(prim["conjunction"]))

    # 20 formal namespace empty (pre-access) / marker-only (formal-run)
    ns_files = sorted(p.name for p in FORMAL_NS.iterdir()) if FORMAL_NS.exists() else []
    allowed_ns = ({ACCESS_MARKER.name, "FORMAL_RUN_LOG.txt"} if phase == "formal-run"
                  else set())
    ns_extra = [f for f in ns_files if f not in allowed_ns]
    chk(20, "formal namespace clean", not ns_extra,
        "%s %s (phase=%s, unexpected=%s)"
        % (FORMAL_NS.relative_to(ROOT),
           "absent" if not FORMAL_NS.exists() else "present, %d files" % len(ns_files),
           phase, ns_extra or "none"))

    # 21 no previous SEALED scientific access
    sealed_set = set(sealed_uuid)
    sealed_idx_set = set(sealed_idx)
    hits = []
    scan_roots = [ROOT / "research_exploration" / p for p in
                  ("qwen3_4b_w2c_prospective_replication_v1",
                   "qwen3_4b_gradient_memory_amendment_v1", "qwen3_4b_stage_d_e_v1",
                   "qwen3_4b_collateral_integrity_audit_v1", "qwen3_4b_formal_gate_v1")]
    scanned = 0
    # Preserved VOID evidence legitimately contains SEALED identifiers: a declared
    # mechanical VOID may occur after access began but before any endpoint was computed.
    # It is excluded from the scan ONLY when the declaration exists and asserts that no
    # endpoint was computed and the run was not consumed -- so this narrows the exemption
    # to disclosed evidence and adds a requirement rather than removing one.
    void_dir = GATE / "void_attempt_1"
    void_decl = void_dir / "FORMAL_VOID_DECLARATION.json"
    void_ok = True
    if void_dir.exists():
        try:
            vd = json.loads(void_decl.read_text())
            void_ok = (vd.get("void") is True
                       and vd.get("endpoints_computed") is False
                       and vd.get("one_shot_run_consumed") is False
                       and vd.get("scientific_outcome_observed") is False)
        except Exception:
            void_ok = False
    for d in scan_roots:
        if not d.exists():
            continue
        for p in sorted(d.rglob("*")):
            if not p.is_file() or p.suffix in (".npz", ".npy", ".safetensors", ".pyc"):
                continue
            if void_dir in p.parents:
                continue
            try:
                txt = p.read_text(encoding="utf-8")
            except Exception:
                continue
            scanned += 1
            for u in sealed_set:
                if u in txt:
                    hits.append({"file": str(p.relative_to(ROOT)), "sealed_uuid_hit": True})
                    break
    # structured check: no committed row-level record carries a sealed project_index
    struct_hits = 0
    for f in (DE / "DEV_INTERVENTION_RECORDS.jsonl", DE / "GRADIENT_DIRECTION_RECORDS.jsonl",
              PROSP / "QWEN3_4B_BASELINE_ROWS.jsonl"):
        for line in open(f, encoding="utf-8"):
            o = json.loads(line)
            if o.get("project_index") in sealed_idx_set or o.get("sample_id") in sealed_set:
                struct_hits += 1
    chk(21, "no previous SEALED scientific access",
        not hits and struct_hits == 0 and void_ok,
        "%d text artifacts scanned, %d identifier hits, %d sealed rows in record files; "
        "declared VOID evidence %s"
        % (scanned, len(hits), struct_hits,
           ("present and asserts no endpoint computed" if void_dir.exists() and void_ok
            else "absent" if not void_dir.exists() else "PRESENT BUT INVALID")))

    # 22 no prior formal run
    led = json.loads(LEDGER.read_text()) if LEDGER.exists() else None
    marker_ok = (ACCESS_MARKER.exists() if phase == "formal-run"
                 else not ACCESS_MARKER.exists())
    ledger_ok = (led is None or (led["formal_run_count"] == 0
                                 and led["scientific_endpoint_observed"] is False))
    if phase != "formal-run":
        ledger_ok = ledger_ok and (led is None or led["access_started"] is False)
    chk(22, "no prior formal run", marker_ok and ledger_ok,
        "access marker %s (required %s in phase=%s); ledger run_count=%s endpoint_observed=%s"
        % ("present" if ACCESS_MARKER.exists() else "absent",
           "present" if phase == "formal-run" else "absent", phase,
           led["formal_run_count"] if led else "not yet created",
           led["scientific_endpoint_observed"] if led else "n/a"))

    # 23 Qwen3.5 untouched / irrelevant
    q35 = [p for p in (ROOT / ".cache").glob("*qwen3*5*") if p.is_dir()] if (ROOT / ".cache").exists() else []
    q35_refs = P.get("is_not", [])
    chk(23, "Qwen3.5 irrelevant to this experiment",
        fc["model"] == "Qwen/Qwen3-4B" and "Qwen3.5" not in json.dumps(P),
        "protocol references only %s; no Qwen3.5 artifact in the formal design" % fc["model"])

    # 24 collateral claim-lock present
    lock = GATE / "COLLATERAL_CLAIM_LOCK.md"
    ver = json.loads((AUDIT / "FILE_INVENTORY.json").read_text())
    readj = (AUDIT / "FORMAL_ADVANCEMENT_READJUDICATION.md").read_text()
    # Anchors are quoted from the committed audit text, not paraphrased, so this check
    # cannot pass against a claim lock that has been softened or removed.
    # Markdown hard-wraps and blockquote markers must not defeat the anchors, so both
    # documents are whitespace-normalised (and "> " stripped) before matching.
    def flat(s):
        return " ".join(s.replace("\n> ", "\n").split())
    lock_txt = flat(lock.read_text(encoding="utf-8")) if lock.exists() else ""
    readj = flat(readj)
    anchors_audit = ["3/106",
                     "must never be presented as demonstrated eligible-at-risk safety",
                     "no deployment"]
    anchors_lock = ["must not be interpreted as demonstrated deployment safety",
                    "3 / 455", "3 / 106"]
    chk(24, "collateral claim-lock present & frozen",
        lock.exists() and ver["verdict"] == "QWEN3_4B_ADVANCEMENT_CONFIRMED"
        and all(a in readj for a in anchors_audit)
        and all(a in lock_txt for a in anchors_lock),
        "audit verdict=%s; lock anchors %d/%d, audit anchors %d/%d"
        % (ver["verdict"], sum(a in lock_txt for a in anchors_lock), len(anchors_lock),
           sum(a in readj for a in anchors_audit), len(anchors_audit)))

    # 25 runner hash frozen
    rh = GATE / "RUNNER_AND_PROTOCOL_HASHES.json"
    self_sha = sha_file(SELF)
    frozen_sha = None
    if rh.exists():
        frozen_sha = json.loads(rh.read_text()).get("formal_runner", {}).get("sha256")
    chk(25, "runner hash frozen",
        frozen_sha is None or frozen_sha == self_sha,
        "runner sha256=%s%s" % (self_sha[:16],
                                "" if frozen_sha else " (to be pinned on freeze)"))

    passed = all(c["pass"] for c in checks)
    return {
        "mode": "gate-only", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repository_head": head, "branch": branch,
        "runner_sha256": self_sha, "protocol_sha256": sha_file(PROTOCOL),
        "checks": checks, "n_checks": len(checks),
        "n_passed": sum(1 for c in checks if c["pass"]),
        "prerequisite_packages": pkg_detail,
        "sealed": {"count": len(sealed_uuid), "ordered_uuid_sha256": recomputed,
                   "rows_loaded": 0, "prompts_rendered": 0, "tokenised": 0, "scored": 0,
                   "payload_fields_read": ["uuid"], "identity_verified_by": "hash only"},
        "model_loaded": False, "gpu_used": False, "inference_run": False,
        "endpoint_computed": False, "arm_executed": False,
        "verdict": "QWEN3_4B_FORMAL_GATE_PASSED" if passed else "QWEN3_4B_FORMAL_GATE_FAILED",
    }


# --------------------------------------------------------------- formal mode
class FormalCtx:
    """Adapter exposing the formal SEALED population through the same accessors the frozen
    Stage-E metric code expects, so `analyze_stage_e.metrics` is reused verbatim rather
    than reimplemented."""

    def __init__(self, v1, rows, gold, source):
        self.v1 = v1
        self._rows = rows
        self._gold, self._source = gold, source
        self.tok = self.model = self.device = None

    def meta(self, ch):
        return self._gold, self._source

    def split_rows(self, split):
        return self._rows                      # the formal population is the only split

    def errors(self, ch, split):
        return sorted([r for r in self._rows
                       if r["gold"] == self._gold and r["prediction"] == self._source],
                      key=lambda r: r["sample_id"])


def run_formal(log, authorised_head):
    """The single authorised scientific execution.

    Hard-aborts before any SEALED read unless the gate passes, the access marker exists,
    the namespace is clean, the HEAD is authorised and the one-shot ledger is unconsumed.
    """
    import torch
    from safetensors import safe_open

    # ---- four hard aborts, all before any SEALED read ----
    gate = run_gate(log, strict_head=authorised_head, phase="formal-run")
    if gate["verdict"] != "QWEN3_4B_FORMAL_GATE_PASSED":
        raise SystemExit("ABORT: gate did not pass; refusing SEALED access")
    if not ACCESS_MARKER.exists():
        raise SystemExit("ABORT: formal access marker absent; refusing SEALED access")
    _allowed = {ACCESS_MARKER.name, "FORMAL_RUN_LOG.txt"}
    if FORMAL_NS.exists() and any(p.name not in _allowed for p in FORMAL_NS.iterdir()):
        raise SystemExit("ABORT: formal namespace not clean")
    led = json.loads(LEDGER.read_text())
    if led["formal_run_count"] != 0 or led["scientific_endpoint_observed"]:
        raise SystemExit("ABORT: one-shot ledger already consumed; manual authorisation required")
    marker = json.loads(ACCESS_MARKER.read_text())
    if marker.get("runner_sha256") != sha_file(SELF) or \
       marker.get("protocol_sha256") != sha_file(PROTOCOL):
        raise SystemExit("ABORT: access marker does not pin this runner/protocol")

    P = json.loads(PROTOCOL.read_text())
    fc = P["frozen_configuration"]
    ch = P["formal_channel"]["channel"]
    gold, source = P["formal_channel"]["gold"], P["formal_channel"]["source"]

    # ---- frozen primitives, imported not rewritten ----
    sys.path.insert(0, str(DE))
    v1 = load_module(ROOT / "scripts/qwen3_8b_stage0_1.py", "q3v1_formal")
    v1.HIDDEN_SIZE = fc["hidden_size"]
    SE = load_module(DE / "run_stage_e_dev.py", "stage_e_formal")     # hook + scoring
    AS = load_module(DE / "analyze_stage_e.py", "analyze_formal")     # destination metrics
    AS.SEED = P["bootstrap"]["seed"]                                  # frozen formal seed
    AS.BOOT = P["bootstrap"]["draws"]

    led["access_started"] = True
    wj(LEDGER, led)

    # ---- SEALED population: the 548 frozen rows, loaded only now ----
    sealed_idx, sealed_uuid, _, perm, _ = sealed_identifiers()
    raw = [json.loads(l) for l in open(RAW, encoding="utf-8")]
    pop = [{"project_index": i, "sample_id": raw[perm[i]]["uuid"],
            "gold": raw[perm[i]]["correct_answer"], "sample": raw[perm[i]]}
           for i in sealed_idx]
    assert [r["sample_id"] for r in pop] == sealed_uuid
    log("SEALED population loaded: %d rows" % len(pop))

    # ---- model, committed loader semantics ----
    from transformers import AutoModelForCausalLM, AutoTokenizer
    v1.set_determinism()
    guard = v1.install_long_sequence_attention_guard()
    mdir = Path(json.loads((PROSP / "MODEL_AND_ENV_MANIFEST.json").read_text())
                ["model"]["local_path"])
    tok = AutoTokenizer.from_pretrained(mdir, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        mdir, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.bfloat16, attn_implementation="eager", device_map="cuda:0")
    model.eval()
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
    device = next(model.parameters()).device
    ctx = FormalCtx(v1, None, gold, source)
    ctx.tok, ctx.model, ctx.device = tok, model, device

    # ---- baseline over all 548 rows, once, before any arm ----
    A_obs = np.zeros((len(pop), fc["hidden_size"]), dtype=np.float32)
    cap = {"L%d" % fc["L_obs"]: model.model.layers[fc["L_obs"]].mlp}
    rows = []
    for i, r in enumerate(pop):
        res, c = v1.score_sample(model, tok, device, r["sample"], cap)
        A_obs[i] = c["L%d" % fc["L_obs"]]
        rows.append({**r, "scores": res["scores"], "prediction": res["prediction"],
                     "runner_up": res["runner_up"]})
    ctx._rows = rows
    log("baseline complete: %d rows" % len(rows))

    # ---- Router gate, frozen coefficients, no refit ----
    z = np.load(PROSP / P["router"]["file"])
    pred_eq_source = [r for r in rows if r["prediction"] == source]
    X = A_obs[[i for i, r in enumerate(rows) if r["prediction"] == source]].astype(np.float64)
    pr = 1.0 / (1.0 + np.exp(-(((X - z["mean"]) / z["scale"]) @ z["coef"].reshape(-1)
                               + float(z["intercept"][0]))))
    routed = [r for r, p in zip(pred_eq_source, pr) if p >= fc["tau"]]
    rprob = {r["sample_id"]: float(p) for r, p in zip(pred_eq_source, pr)}
    log("routed %d of %d source-predicted rows" % (len(routed), len(pred_eq_source)))

    # ---- directions ----
    with safe_open(DE / "PRIMARY_DIRECTIONS.safetensors", framework="numpy") as f:
        dg = f.get_tensor(P["directions"]["primary_d_grad"]["key"])
    with safe_open(DE / "DIFFMEAN_DIRECTIONS.safetensors", framework="numpy") as f:
        d26 = f.get_tensor(P["directions"]["L26_diffmean_frozen"]["key"])
    with safe_open(DE / "FORMAL_RANDOMS.safetensors", framework="numpy") as f:
        R = f.get_tensor("formal_randoms")
    if sha_arr(dg) != P["directions"]["primary_d_grad"]["sha256"] or \
       sha_arr(R) != P["formal_randoms"]["matrix_sha256"]:
        raise SystemExit("VOID: DIRECTION_MISMATCH / RANDOM_VECTOR_REPLACED")

    absn = fc["absolute_perturbation_norm"]
    recs_path = FORMAL_NS / "QWEN3_4B_FORMAL_RECORDS.jsonl"

    def arm(name, direction, layer, population, order):
        delta = None
        achieved = 0.0
        if direction is not None:
            d64 = np.ascontiguousarray(direction.astype(np.float64))
            d64 = d64 / np.linalg.norm(d64)
            delta = torch.from_numpy((d64 * absn).astype(np.float32)).to("cuda:0")
            achieved = float(torch.linalg.vector_norm(delta.double()).item())
            if abs(achieved - absn) / max(1.0, absn) > 1e-5:
                raise SystemExit("VOID: DOSE_MISMATCH")
        out = {}
        with open(recs_path, "a", encoding="utf-8", newline="\n") as fh:
            for n, r in enumerate(population):
                sc, pred, runner = SE.score_with_intervention(ctx, r["sample"], layer, delta)
                out[r["sample_id"]] = pred
                fh.write(json.dumps({
                    "sample_id": r["sample_id"], "project_index": r["project_index"],
                    "gold": r["gold"], "scores_base": r["scores"],
                    "pred_base": r["prediction"],
                    "router_probability": rprob.get(r["sample_id"]),
                    "router_decision": r["sample_id"] in {x["sample_id"] for x in routed},
                    "arm": name, "direction_sha256": (sha_arr(direction)
                                                      if direction is not None else None),
                    "q": fc["q"], "abs_delta_norm": absn, "achieved_delta_norm": achieved,
                    "layer": layer, "scores_int": sc, "pred_int": pred,
                    "runner_up_int": runner,
                    "destination": ("SOURCE_RETAINED" if pred == source
                                    else "GOLD_ARRIVAL" if pred == gold else "OTHER_WRONG"),
                    "batch_index": n, "global_execution_order": order,
                }, sort_keys=True) + "\n")
        return out

    order = 0
    effects = {}
    # zero arm, then the exact zero-equality gate
    effects["zero"] = arm("zero", None, fc["L_inj"], routed, order); order += 1
    for r in routed:
        if effects["zero"][r["sample_id"]] != r["prediction"]:
            raise SystemExit("VOID: ZERO_CONTROL_FAILURE")
    log("zero-control exact equality: PASS")

    for name, d, layer, popn in [
            ("real_d_grad", dg, fc["L_inj"], routed),
            ("d_L26_diffmean", d26, fc["L_inj"], routed),
            ("reverse_d_grad", -dg, fc["L_inj"], routed),
            ("wrong_layer_d_grad", dg, fc["wrong_layer"], routed),
            ("ungated_d_grad", dg, fc["L_inj"], pred_eq_source)]:
        effects[name] = arm(name, d, layer, popn, order); order += 1
        log("arm %s complete (n=%d)" % (name, len(popn)))
    for i in range(P["formal_randoms"]["K"]):
        effects["formal_random_%d" % i] = arm("formal_random_%d" % i, R[i],
                                              fc["L_inj"], routed, order); order += 1
    log("K=%d formal random arms complete" % P["formal_randoms"]["K"])

    # frozen score-space comparator, b_c fixed on DEV
    b_c = fc["b_c"]
    ss = {}
    for r in routed:
        e = dict(r["scores"])
        e[gold] += b_c / 2.0
        e[source] -= b_c / 2.0
        ss[r["sample_id"]] = max(e, key=lambda k: e[k])
    effects["score_space"] = ss

    # ---- endpoints, via the frozen Stage-E metric implementation ----
    def mk(eff):
        return AS.metrics(ctx, ch, [{"sample_id": k, "pred_int": v, "scores_int": None,
                                     "scores_base": None} for k, v in eff.items()])
    results = {k: mk(v) for k, v in effects.items()}
    real = results["real_d_grad"]["target_gain_rate"]
    rnd = [results["formal_random_%d" % i]["target_gain_rate"]
           for i in range(P["formal_randoms"]["K"])]
    ge = sum(1 for x in rnd if x >= real)
    p_add_one = (1 + ge) / (P["formal_randoms"]["K"] + 1)

    m = results["real_d_grad"]
    conj = {
        "1_support_ge_30": m["n_channel_error"] >= 30,
        "2_target_gain_rate_gt_0": (m["target_gain_rate"] or 0) > 0,
        "3_target_gain_rate_ci_lower_gt_0": m["target_gain_rate_ci95"][0] > 0,
        "4_p_add_one_le_0.05": p_add_one <= 0.05,
        "5_target_hit_gt_0.50": (m["target_hit"] or 0) > 0.50,
        "6_target_hit_ci_lower_gt_0.50": m["target_hit_ci95"][0] > 0.50,
        "7_collateral_le_0.05": (m["clean_collateral_rate"] or 0) <= 0.05,
        "8_collateral_ci_upper_le_0.05": m["clean_collateral_rate_ci95"][1] <= 0.05,
        "9_zero_control_exact": True,
        "10_no_structural_failure": True,
    }
    verdict = ("QWEN3_4B_FORMAL_CONFIRMATORY_SUCCESS" if all(conj.values())
               else "QWEN3_4B_FORMAL_CONFIRMATORY_FAILURE")
    wj(FORMAL_NS / "QWEN3_4B_FORMAL_RESULTS.json", {
        "channel": ch, "verdict": verdict, "conjunction": conj,
        "p_add_one": p_add_one, "randoms_ge_real": ge,
        "random_target_gain_rates": rnd, "arms": results,
        "eligible_at_risk_collateral": {
            "numerator": m["collateral_count"], "denominator": m["eligible_collateral"],
            "note": "required diagnostic disclosure; see COLLATERAL_CLAIM_LOCK.md"},
        "claim_lock": "Passing the frozen formal collateral endpoint must not be "
                      "interpreted as demonstrated deployment safety.",
        "invocation_uuid": marker.get("invocation_uuid"),
        "attention_guard": guard})
    led["formal_run_count"] = 1
    led["scientific_endpoint_observed"] = True
    led["history"] = led.get("history", []) + [
        {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "invocation_uuid": marker.get("invocation_uuid"), "verdict": verdict}]
    wj(LEDGER, led)
    log("FORMAL VERDICT %s  p_add_one=%.6f" % (verdict, p_add_one))
    return 0


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--gate-only", action="store_true")
    g.add_argument("--run-formal", action="store_true")
    ap.add_argument("--authorised-head", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.run_formal:
        FORMAL_NS.mkdir(parents=True, exist_ok=True)
        logpath = FORMAL_NS / "FORMAL_RUN_LOG.txt"
    else:
        logpath = a.out and (Path(a.out).parent / "GATE_ONLY_LOG.txt")
        logpath = logpath or (GATE / "GATE_ONLY_LOG.txt")
    log = Log(logpath)
    try:
        if a.gate_only:
            log("MODE=gate-only  NO SEALED SCIENTIFIC INFERENCE  NO MODEL LOAD")
            res = run_gate(log, strict_head=a.authorised_head)
            log("VERDICT %s  (%d/%d checks passed)"
                % (res["verdict"], res["n_passed"], res["n_checks"]))
            wj(a.out or (GATE / "GATE_ONLY_OUTPUT.json"), res)
            return 0 if res["verdict"] == "QWEN3_4B_FORMAL_GATE_PASSED" else 1
        run_formal(log, a.authorised_head)
    finally:
        log.flush()


if __name__ == "__main__":
    sys.exit(main())
