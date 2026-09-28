#!/usr/bin/env python3
"""Shared frozen context for the Qwen3-4B Stage D/E continuation.

Every scientific definition is inherited by importing the committed Qwen3-8B modules
(`scripts/qwen3_8b_stage0_1.py`, `scripts/qwen3_8b_stage0_1_v2.py`). Nothing here
redefines an estimator, a metric, a gate or a population rule.

The only Qwen3-4B-specific facts are the checkpoint directory and the model width, which
is a property of the host model, not a protocol choice. `V1.HIDDEN_SIZE` is rebound to
2560 before any geometry call because the frozen `geometry_metrics` derives its random
null references from the model width; leaving it at the Qwen3-8B 4096 would evaluate a
2560-dimensional direction against the wrong null.

Loader: the committed `load_model()` semantics are replicated exactly, including the
parameter-gradient disabling verified bitwise-equivalent in
`research_exploration/qwen3_4b_gradient_memory_amendment_v1`.

SEALED evaluation is never loaded, rendered, tokenised or scored.
"""

import csv
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np

ROOT = Path("/root/autodl-tmp/sakiko-followup")
HERE = Path(__file__).resolve().parent
PROSP = ROOT / "research_exploration/qwen3_4b_w2c_prospective_replication_v1"
AMEND = ROOT / "research_exploration/qwen3_4b_gradient_memory_amendment_v1"
SCRATCH = Path("/path/to/experiment-cache")
BIG = SCRATCH / "stage_d_e"

MODEL_REPO = "Qwen/Qwen3-4B"
MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
MODEL_DIR = ROOT / ".cache/qwen3_4b" / MODEL_REVISION

RAW = ROOT / "data/processed/qwen25_7b_w2c/w2c_test_mcq.jsonl"
SPLIT_INDEX = ROOT / "final/results/qwen3_stage0_1_v2/QWEN3_V2_SPLIT_INDEX.csv"
PERM_SEED = 42
TOTAL_ROWS = 3652
N_TRAIN_DEV = 3104
N_SEALED = 548

L_OBS = 26
L_INJ = 21
HIDDEN = 2560
N_LAYERS = 36

# ---- frozen constants inherited from the modern Qwen3 protocol -------------
CHANNELS = [
    # (channel key, gold, source)  -- both support-eligible + readable channels,
    # carried forward verbatim from Stage C. Neither may be dropped or reordered.
    ("cannot_answer__to__tool_call", "cannot_answer", "tool_call"),
    ("cannot_answer__to__direct", "cannot_answer", "direct"),
]
Q_GRID = [0.0, 0.125, 0.25, 0.5, 1.0, 2.0]
GEOMETRY_BOOTSTRAP_DRAWS = 10000
GEOMETRY_BOOTSTRAP_SEED = 20260731          # frozen Stage 1.5 seed
DEV_BOOTSTRAP_DRAWS = 10000
DEV_BOOTSTRAP_SEED = 20260801               # frozen Stage 2A seed
DEV_RANDOM_SEEDS = [20260811, 20260812, 20260813, 20260814,
                    20260815, 20260816, 20260817, 20260818]
WRONG_LAYER = L_OBS                         # the other frozen site
GATE = {"source_exits_min": 10, "target_hit_min": 0.50, "clean_collateral_max": 0.05}
PERTURBATION_RTOL = 1e-5

# Stage C values, resolved from the archived package rather than from any prompt.
EXPECTED = {
    "cannot_answer__to__tool_call": {"train_err": 436, "dev_err": 220,
                                     "train_ref": 68, "dev_ref": 40},
    "cannot_answer__to__direct": {"train_err": 174, "dev_err": 114,
                                  "train_ref": 68, "dev_ref": 40},
}


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def sha_arr(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def wj(p, obj):
    t = Path(str(p) + ".tmp")
    t.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(t, p)


def load_v1v2():
    """Import the committed frozen modules and bind the host model width."""
    sys.path.insert(0, str(ROOT / "scripts"))

    def _load(path, name):
        spec = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    v1 = _load(ROOT / "scripts/qwen3_8b_stage0_1.py", "q3v1")
    v2 = _load(ROOT / "scripts/qwen3_8b_stage0_1_v2.py", "q3v2")
    assert v1.L_OBS == L_OBS and v1.L_INJ == L_INJ, (v1.L_OBS, v1.L_INJ)
    assert v1.MODES == ["tool_call", "direct", "request_for_info", "cannot_answer"]
    # host-model width; a property of Qwen3-4B, not a protocol choice
    v1.HIDDEN_SIZE = HIDDEN
    return v1, v2


def load_population():
    """Frozen TRAIN/DEV reconstruction: project_index i -> raw[perm[i]]."""
    raw = [json.loads(l) for l in open(RAW, encoding="utf-8")]
    assert len(raw) == TOTAL_ROWS, len(raw)
    perm = np.random.default_rng(PERM_SEED).permutation(TOTAL_ROWS)
    rows = []
    for r in csv.DictReader(open(SPLIT_INDEX, encoding="utf-8")):
        pi = int(r["project_index"])
        s = raw[perm[pi]]
        assert s["uuid"] == r["sample_id"], (pi, s["uuid"], r["sample_id"])
        assert s["correct_answer"] == r["gold"]
        rows.append({"project_index": pi, "sample_id": s["uuid"], "split": r["v2_split"],
                     "sample": s})
    assert len(rows) == N_TRAIN_DEV, len(rows)
    sealed = sorted(set(range(TOTAL_ROWS)) - {r["project_index"] for r in rows})
    assert len(sealed) == N_SEALED
    return rows, set(sealed)


class Ctx:
    """Frozen Stage A-C state plus the two eligible channels."""

    def __init__(self, need_model=False):
        self.v1, self.v2 = load_v1v2()
        self.base = [json.loads(l) for l in
                     open(PROSP / "QWEN3_4B_BASELINE_ROWS.jsonl", encoding="utf-8")]
        assert len(self.base) == N_TRAIN_DEV
        self.by_pi = {r["project_index"]: r for r in self.base}
        self.by_id = {r["sample_id"]: r for r in self.base}
        self.rowpos = {r["project_index"]: i for i, r in enumerate(self.base)}
        pop, self.sealed = load_population()
        self.sample_of = {r["project_index"]: r["sample"] for r in pop}
        for r in self.base:                       # baseline rows agree with reconstruction
            assert self.sample_of[r["project_index"]]["uuid"] == r["sample_id"]
        self.A = {}
        for L, expect in ((L_OBS, "38487e7472552aa81c6a9a2b2a2c5dd3b11af9f7744f9bc8108405ba189dd9cd"),
                          (L_INJ, "18b1c5cc43743bf397aef9ccfba2f5828292c0c05f4d9fd5195c8a4fcffd7922")):
            p = SCRATCH / ("QWEN3_4B_ACT_L%d.npy" % L)
            assert sha_file(p) == expect, "activation cache drift at L%d" % L
            a = np.load(p)
            assert a.shape == (N_TRAIN_DEV, HIDDEN), a.shape
            self.A[L] = a
        self.routers = {}
        for ch, _, _ in CHANNELS:
            z = np.load(PROSP / ("ROUTER_%s.npz" % ch.replace("__to__", "__")))
            self.routers[ch] = {k: z[k] for k in z.files}
        self.tau = {}
        for r in csv.DictReader(open(PROSP / "ROUTER_AUDIT.csv", encoding="utf-8")):
            key = r["channel"].replace("->", "__to__")
            self.tau[key] = float(r["tau"])
        self.model = self.tok = self.device = None
        if need_model:
            self.load_model()

    # ---------------- populations (frozen definitions) ----------------
    def split_rows(self, split):
        return [r for r in self.base if r["split"] == split]

    def errors(self, ch, split):
        g, s = self.meta(ch)
        sel = [r for r in self.split_rows(split) if r["gold"] == g and r["prediction"] == s]
        return sorted(sel, key=lambda r: r["sample_id"])

    def reference(self, ch, split):
        g, _ = self.meta(ch)
        sel = [r for r in self.split_rows(split) if r["gold"] == g and r["prediction"] == g]
        return sorted(sel, key=lambda r: r["sample_id"])

    def meta(self, ch):
        return next((g, s) for c, g, s in CHANNELS if c == ch)

    def router_prob(self, ch, rows):
        R = self.routers[ch]
        X = self.A[L_OBS][[self.rowpos[r["project_index"]] for r in rows]].astype(np.float64)
        z = ((X - R["mean"]) / R["scale"]) @ R["coef"].reshape(-1) + float(R["intercept"][0])
        return 1.0 / (1.0 + np.exp(-z))

    def pred_eq_source(self, ch, split="dev"):
        _, s = self.meta(ch)
        return sorted([r for r in self.split_rows(split) if r["prediction"] == s],
                      key=lambda r: r["sample_id"])

    def routed(self, ch, split="dev"):
        pop = self.pred_eq_source(ch, split)
        pr = self.router_prob(ch, pop)
        return [r for r, p in zip(pop, pr) if p >= self.tau[ch]], pop, pr

    # ---------------- model (committed loader semantics) ----------------
    def load_model(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.v1.set_determinism()
        guard = self.v1.install_long_sequence_attention_guard()
        torch.cuda.init()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)
        self.tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True,
                                                 trust_remote_code=False)
        m = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR, local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.bfloat16, attn_implementation="eager", device_map="cuda:0")
        # ---- the amendment: committed load_model() semantics, verbatim ----
        m.eval()
        m.config.use_cache = False
        for p in m.parameters():
            p.requires_grad_(False)
        # -------------------------------------------------------------------
        assert m.config.model_type == "qwen3"
        assert getattr(m.config, "_attn_implementation", None) == "eager"
        assert m.config.num_hidden_layers == N_LAYERS
        assert m.config.hidden_size == HIDDEN
        assert sum(1 for p in m.parameters() if p.requires_grad) == 0
        self.model = m
        self.device = next(m.parameters()).device
        self.guard = guard
        return m


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    return v / n, n


def diffmean_direction(ctx, ch, layer):
    """Frozen DiffMean: unit(mean(correct gold reference) - mean(gold->source error)),
    TRAIN only, sign convention positive intended goldward.  Identical formula at either
    site; the layer is the only argument."""
    err = ctx.errors(ch, "train")
    ref = ctx.reference(ch, "train")
    A = ctx.A[layer]
    ea = A[[ctx.rowpos[r["project_index"]] for r in err]].astype(np.float64)
    ra = A[[ctx.rowpos[r["project_index"]] for r in ref]].astype(np.float64)
    raw = ra.mean(0) - ea.mean(0)
    d, n = unit(raw)
    assert np.all(np.isfinite(d)) and n > 1e-8
    return d.astype(np.float32), n, len(err), len(ref)


def random_unit(seed, dim=HIDDEN):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim)
    return (v / np.linalg.norm(v)).astype(np.float32)


def assert_sealed_clean(ctx, rows):
    for r in rows:
        if r["project_index"] in ctx.sealed:
            print("QWEN3_4B_TEST_CONTAMINATION: %d" % r["project_index"], flush=True)
            raise SystemExit(2)
