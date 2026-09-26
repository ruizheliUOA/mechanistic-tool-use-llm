"""
discover_sakiko_channels.py — data-driven channel discovery for SAKIKO (CPU-only, no model).
============================================================================================
Frames SAKIKO as *channel-adaptive*: instead of assuming three fixed channels, it DISCOVERS
the dominant baseline error transitions (gold_label -> pred_label) from per-sample baseline
labels, applies configurable selection thresholds, and tags each candidate channel with a
status. It also consolidates already-archived diagnostics (ca_direct layer scan, ca_tc
DiffMean-vs-PCA, placebo controls) into flat, provenance-carrying CSV/JSON tables, and runs a
Net = Fixed - Broke consistency audit across all summary files that expose those fields.

STRICT SCOPE: reads only existing archived summaries / real per-sample files / small committed
vectors. Loads NO model weights and runs NO inference. Geometry that needs the local activation
caches lives in the companion `qwen7b_channel_geometry.py`.

Channel status vocabulary:
  stable            — recovered from exact per-sample data, count >= COUNT_MIN (or train >= TRAIN_MIN)
  weak              — recovered from data but below the stable thresholds
  insufficient      — present but far too few samples to support a direction
  summary_only      — real number, but from a summary file (per-sample not on disk / LFS pointer)
  qualitative_only  — no reliable per-sample response-mode gold labels available
  TODO_needs_activations — analysis blocked until activations are extracted

Usage:  python scripts/discover_sakiko_channels.py
"""
from __future__ import annotations
import json, csv, sys, logging
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "final" / "results" / "channel_adaptive"
OUT.mkdir(parents=True, exist_ok=True)

# ── inputs (all read-only) ────────────────────────────────────────────────────
QWEN_BASELINE = ROOT / "final/results/7b_w2c_baseline/qwen25_7b_baseline_details.jsonl"
SPLIT_DIR = ROOT / "sakiko_v3/results/splits"
METATOOL_SUM = ROOT / "final/results/dataset_extension/metatool_binary_baseline_summary.json"
METATOOL_DET = ROOT / "final/results/dataset_extension/metatool_binary_baseline_details.jsonl"
PHI_EVAL = ROOT / "sakiko_v3/results/p0_final_test_eval.json"
PHI_W2C_SUM = ROOT / "trace_db/w2c_phi35_summary.json"
TOOLSANDBOX_CAND = ROOT / "data/annotation/toolsandbox_decision_candidates.csv"

PARTA = ROOT / "final/results/7b_w2c_sakiko/method_diagnostics/gpu_confirmation/partA_ca_direct_layer_scan.json"
PARTC = ROOT / "final/results/7b_w2c_sakiko/method_diagnostics/gpu_confirmation/partC_direction_comparison.json"
TASKA = ROOT / "final/results/7b_w2c_sakiko/method_diagnostics/gpu_confirmation/followup_locked_tests/taskA_ca_direct_l20_cascade.json"
TASKB = ROOT / "final/results/7b_w2c_sakiko/method_diagnostics/gpu_confirmation/followup_locked_tests/taskB_ca_tc_pca_vs_diffmean.json"
FIG5_GPU = ROOT / "final/results/figures/fig5_ca_direct_layer_sensitivity_gpu_confirmed.csv"
PLACEBO = ROOT / "final/results/7b_w2c_sakiko/qwen25_7b_placebo_controls.json"

# files scanned for Net = Fixed - Broke consistency
NET_FILES = [
    ("qwen_locked_test", ROOT / "final/results/7b_w2c_sakiko/qwen25_7b_locked_test_summary.json"),
    ("qwen_placebo_real", PLACEBO),
    ("taskA_cascade", TASKA),
    ("taskB_ca_tc_pca", TASKB),
    ("partC_directions", PARTC),
    ("multiseed_per_seed", ROOT / "final/results/7b_w2c_sakiko/multiseed/qwen25_7b_multiseed_results.json"),
]

# selection thresholds (configurable)
COUNT_MIN = 50   # overall error count for "stable"
TRAIN_MIN = 30   # train error count for "stable"

GOLD_LABELS = ["tool_call", "request_for_info", "cannot_answer"]   # W2C golds ("direct" never gold)
PRED_LABELS = ["tool_call", "direct", "request_for_info", "cannot_answer"]
# canonical over/under-calling channel names used elsewhere in SAKIKO
NAMED = {("request_for_info", "tool_call"): "rfi_tc",
         ("cannot_answer", "tool_call"): "ca_tc",
         ("cannot_answer", "direct"): "ca_direct"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("discover")
WARNINGS: list[str] = []


def status_for(count_all: int, count_train: int | None) -> str:
    if count_all >= COUNT_MIN or (count_train is not None and count_train >= TRAIN_MIN):
        return "stable"
    if count_all >= 15:
        return "weak"
    return "insufficient"


def w(msg): WARNINGS.append(msg); log.warning(msg)


# ══════════════════════════════════════════════════════════════════════════════
# A. Qwen2.5-7B W2C — EXACT from per-sample baseline labels
# ══════════════════════════════════════════════════════════════════════════════
def discover_qwen():
    if not QWEN_BASELINE.exists():
        w("Qwen baseline details missing -> Qwen channel discovery TODO"); return None
    det = [json.loads(l) for l in open(QWEN_BASELINE)]
    n = len(det)
    # per-sample gold/pred
    gold = [d["gold"] for d in det]
    pred = [d["pred"] for d in det]
    # splits (optional)
    split_of = {}
    have_splits = all((SPLIT_DIR / f"{s}_idx.json").exists() for s in ("train", "val", "test"))
    if have_splits:
        for s in ("train", "val", "test"):
            for i in map(int, json.load(open(SPLIT_DIR / f"{s}_idx.json"))):
                split_of[i] = s
    else:
        w("W2C split idx files missing -> per-split channel counts unavailable")

    # 4x4 confusion (gold x pred)
    conf = {g: {p: 0 for p in PRED_LABELS} for g in GOLD_LABELS}
    for g, p in zip(gold, pred):
        if g in conf:
            conf[g][p] = conf[g].get(p, 0) + 1

    # error transitions gold->pred (g != p)
    trans_all = Counter()
    trans_split = defaultdict(Counter)
    for i, (g, p) in enumerate(zip(gold, pred)):
        if g == p:
            continue
        trans_all[(g, p)] += 1
        if i in split_of:
            trans_split[split_of[i]][(g, p)] += 1

    channels = []
    for (g, p), c in sorted(trans_all.items(), key=lambda kv: -kv[1]):
        ctr = trans_split["train"][(g, p)] if have_splits else None
        cv = trans_split["val"][(g, p)] if have_splits else None
        cte = trans_split["test"][(g, p)] if have_splits else None
        channels.append({
            "channel_id": NAMED.get((g, p), f"{g}__{p}"),
            "gold": g, "pred": p, "is_named_sakiko_channel": (g, p) in NAMED,
            "count_all": c, "count_train": ctr, "count_val": cv, "count_test": cte,
            "pct_of_errors": round(100 * c / sum(trans_all.values()), 2),
            "status": status_for(c, ctr),
        })

    acc = sum(1 for g, p in zip(gold, pred) if g == p) / n
    result = {
        "dataset": "w2c_qwen25_7b", "model": "Qwen2.5-7B-Instruct",
        "provenance": {"source_file": str(QWEN_BASELINE.relative_to(ROOT)),
                       "recompute_type": "EXACT_per_sample",
                       "splits": str(SPLIT_DIR.relative_to(ROOT)) if have_splits else None},
        "n_samples": n, "baseline_accuracy": round(acc, 5),
        "gold_distribution": dict(Counter(gold)), "pred_distribution": dict(Counter(pred)),
        "label_space": {"n_gold_classes": len(GOLD_LABELS), "gold_labels": GOLD_LABELS,
                        "pred_labels": PRED_LABELS, "note": "'direct' is a distractor pred, never gold"},
        "total_errors": sum(trans_all.values()),
        "selection_thresholds": {"count_min_overall": COUNT_MIN, "train_min": TRAIN_MIN},
        "discovered_channels": channels,
        "dominant_channels_stable": [c["channel_id"] for c in channels if c["status"] == "stable"],
        "confusion_matrix_gold_by_pred": conf,
    }
    json.dump(result, open(OUT / "channel_discovery_qwen25_7b_w2c.json", "w"), indent=2)

    # error transition matrix CSV (gold x pred), long form with provenance
    with open(OUT / "error_transition_matrix_w2c_qwen7b.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["gold", "pred", "count", "is_error", "is_named_channel", "channel_id",
                     "count_train", "count_val", "count_test", "source_file"])
        for g in GOLD_LABELS:
            for p in PRED_LABELS:
                c = conf[g][p]; is_err = int(g != p)
                cid = NAMED.get((g, p), f"{g}__{p}") if g != p else "-"
                ctr = trans_split["train"][(g, p)] if have_splits else ""
                cv = trans_split["val"][(g, p)] if have_splits else ""
                cte = trans_split["test"][(g, p)] if have_splits else ""
                wr.writerow([g, p, c, is_err, int((g, p) in NAMED), cid,
                             ctr, cv, cte, QWEN_BASELINE.name])
    log.info("Qwen: n=%d acc=%.4f | stable channels=%s", n, acc, result["dominant_channels_stable"])
    return result


# ══════════════════════════════════════════════════════════════════════════════
# B. MetaTool-Binary — SUMMARY-ONLY (per-sample details are an LFS pointer)
# ══════════════════════════════════════════════════════════════════════════════
def _is_real_jsonl(path: Path) -> bool:
    """True if the .jsonl holds real content (not a ~130B git-lfs pointer)."""
    if not path.exists():
        return False
    if path.stat().st_size < 500:
        with open(path) as f:
            head = f.read(64)
        return "git-lfs" not in head
    return True


def discover_metatool():
    if not METATOOL_SUM.exists():
        w("MetaTool summary missing -> MetaTool channel TODO"); return None
    s = json.load(open(METATOOL_SUM))
    details_real = _is_real_jsonl(METATOOL_DET)
    if not details_real:
        w("MetaTool per-sample details are an LFS pointer -> summary_only (retrievable via git lfs pull)")
    ep = s.get("error_pool", {})
    over = ep.get("no_tool_to_tool_call")   # the over-calling channel analogue
    under = ep.get("tool_call_to_no_tool")
    channels = [
        {"channel_id": "no_tool__tool_call", "gold": "no_tool", "pred": "tool_call",
         "kind": "over_calling", "count_all": over,
         "status": "summary_only" if not details_real else status_for(over or 0, None)},
        {"channel_id": "tool_call__no_tool", "gold": "tool_call", "pred": "no_tool",
         "kind": "under_calling", "count_all": under,
         "status": "summary_only" if not details_real else status_for(under or 0, None)},
    ]
    result = {
        "dataset": "metatool_binary", "model": s.get("model_id"),
        "provenance": {"source_file": str(METATOOL_SUM.relative_to(ROOT)),
                       "recompute_type": "EXACT_per_sample" if details_real else "SUMMARY_ONLY",
                       "per_sample_details": str(METATOOL_DET.relative_to(ROOT)),
                       "per_sample_available": details_real},
        "n_evaluated": s.get("n_evaluated"), "accuracy": s.get("accuracy"),
        "label_space": {"n_gold_classes": 2, "gold_labels": ["tool_call", "no_tool"],
                        "candidate_strings": s.get("candidate_strings")},
        "confusion_matrix": s.get("confusion_matrix"),
        "over_calling_channel": "no_tool__tool_call",
        "discovered_channels": channels,
        "decision_verdict": s.get("decision_verdict"),
    }
    json.dump(result, open(OUT / "channel_discovery_metatool_binary_summary.json", "w"), indent=2)
    with open(OUT / "metatool_binary_channel_summary.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["channel_id", "gold", "pred", "kind", "count_all", "status",
                     "recompute_type", "source_file"])
        rt = "EXACT_per_sample" if details_real else "SUMMARY_ONLY"
        for c in channels:
            wr.writerow([c["channel_id"], c["gold"], c["pred"], c["kind"], c["count_all"],
                         c["status"], rt, METATOOL_SUM.name])
    log.info("MetaTool-Binary: over-call no_tool->tool_call=%s under-call=%s (%s)", over, under, rt)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# C. Phi-3.5 W2C — SUMMARY-ONLY
# ══════════════════════════════════════════════════════════════════════════════
def discover_phi():
    if not PHI_EVAL.exists():
        w("Phi eval summary missing -> Phi channel TODO"); return None
    e = json.load(open(PHI_EVAL))
    bd = e.get("breakdown", {})
    name_map = {"request_for_info→tool_call": "rfi_tc",
                "cannot_answer→tool_call": "ca_tc",
                "cannot_answer→direct": "ca_direct"}
    channels = []
    for k, v in bd.items():
        g, p = k.split("→")
        channels.append({"channel_id": name_map.get(k, k.replace("→", "__")),
                         "gold": g, "pred": p, "count_test": v.get("n"),
                         "corrected": v.get("corrected"), "correction_rate": v.get("correction_rate"),
                         "status": "summary_only"})
    w2c_sum = json.load(open(PHI_W2C_SUM)) if PHI_W2C_SUM.exists() else None
    result = {
        "dataset": "w2c_phi35", "model": "Phi-3.5-mini-instruct",
        "provenance": {"source_files": [str(PHI_EVAL.relative_to(ROOT))] +
                       ([str(PHI_W2C_SUM.relative_to(ROOT))] if w2c_sum else []),
                       "recompute_type": "SUMMARY_ONLY",
                       "note": "per-sample Phi labels are LFS pointers / W2C source (read-only)"},
        "test_baseline_accuracy_pct": e.get("baseline_accuracy"),
        "test_intervened_accuracy_pct": e.get("intervened_accuracy"),
        "net_benefit": e.get("net_benefit"),
        "discovered_channels": channels,
        "dominant_channels": [c["channel_id"] for c in channels],
        "note": "Phi confirms the SAME 3 dominant channels as Qwen (rfi_tc, ca_tc, ca_direct), summary-level.",
    }
    json.dump(result, open(OUT / "channel_discovery_phi35_summary.json", "w"), indent=2)
    with open(OUT / "phi35_channel_summary.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["channel_id", "gold", "pred", "count_test", "corrected",
                     "correction_rate_pct", "status", "recompute_type", "source_file"])
        for c in channels:
            wr.writerow([c["channel_id"], c["gold"], c["pred"], c["count_test"], c["corrected"],
                         c["correction_rate"], c["status"], "SUMMARY_ONLY", PHI_EVAL.name])
    log.info("Phi-3.5: channels(summary)=%s", result["dominant_channels"])
    return result


# ══════════════════════════════════════════════════════════════════════════════
# D. ToolSandbox — qualitative_only
# ══════════════════════════════════════════════════════════════════════════════
def discover_toolsandbox():
    has_cand = TOOLSANDBOX_CAND.exists()
    result = {"dataset": "toolsandbox", "status": "qualitative_only",
              "provenance": {"candidate_file": str(TOOLSANDBOX_CAND.relative_to(ROOT)) if has_cand else None,
                             "recompute_type": "QUALITATIVE_ONLY"},
              "note": ("Decision candidates exist but there is no reliable per-turn response-mode gold "
                       "label file; ToolSandbox is qualitative-only, not a quantitative multi-class channel benchmark."),
              "discovered_channels": []}
    log.info("ToolSandbox: qualitative_only (candidate file present=%s)", has_cand)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Consolidation of archived diagnostics -> flat CSVs
# ══════════════════════════════════════════════════════════════════════════════
def consolidate_diagnostics():
    # ca_direct layer comparison (obs L16/18/20/22)
    if PARTA.exists():
        a = json.load(open(PARTA))
        best_by_obs = {}
        for r in a["scan"]:
            o = r["obs"]
            if o not in best_by_obs or (r["net"], -r["broke"]) > (best_by_obs[o]["net"], -best_by_obs[o]["broke"]):
                best_by_obs[o] = r
        with open(OUT / "ca_direct_layer_comparison_qwen7b.csv", "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["channel", "obs_layer", "direction_norm", "router_val_auc", "best_inj",
                         "best_alpha", "best_thr", "val_net", "val_fixed", "val_broke",
                         "ca_direct_before", "ca_direct_after", "test_reduction", "source_file"])
            lt = a.get("locked_test", {})
            for L, v in a["obs_layers"].items():
                b = best_by_obs[int(L)]
                tred = lt.get("reduction") if int(L) == lt.get("obs") else ""
                wr.writerow(["ca_direct", L, v["dir_norm"], v["router_val_auc"], b["inj"], b["alpha"],
                             b["thr"], b["net"], b["fixed"], b["broke"], b["ch_before"], b["ch_after"],
                             tred, PARTA.name])

    # direction method comparison (DiffMean vs PCA) — from partC (val) + taskB (val/test)
    rows = []
    if PARTC.exists():
        c = json.load(open(PARTC))
        for ch, v in c.items():
            for m in ("diffmean", "pca1", "pca_top10"):
                d = v["detail"][m]
                rows.append(["val", ch, m, v["obs"], v["inj"], v["alpha"], v["dir_norm"][m],
                             v["cos_diffmean_pca1"], d["net"], d["fixed"], d["broke"],
                             d["ch_before"], d["ch_after"], PARTC.name])
    if TASKB.exists():
        b = json.load(open(TASKB))
        for m in ("diffmean", "pca1"):
            for split in ("val", "test"):
                d = b[m][split]
                rows.append([split, "ca_tc", m + "_locked", b["config"]["obs"], b["config"]["inj"],
                             b["config"]["alpha"], b["direction_norm"].get(m, ""),
                             b["cos_diffmean_pca1"], d["net"], d["fixed"], d["broke"],
                             d["before"], d["after"], TASKB.name])
    with open(OUT / "direction_method_comparison_qwen7b.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["split", "channel", "direction_method", "obs_layer", "inj_layer", "alpha",
                     "direction_norm", "cos_diffmean_pca1", "net", "fixed", "broke",
                     "before", "after", "source_file"])
        wr.writerows(rows)

    # ca_tc PCA vs DiffMean focused CSV (val + test + reverse controls)
    if TASKB.exists():
        b = json.load(open(TASKB))
        with open(OUT / "ca_tc_pca_vs_diffmean_qwen7b.csv", "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["direction_method", "split", "locked_thr", "net", "fixed", "broke",
                         "before", "after", "direction_norm", "cos_diffmean_pca1", "source_file"])
            for m, tag in [("diffmean", "diffmean"), ("pca1", "pca1"),
                           ("pca1_reverse_control", "pca1_reverse"),
                           ("diffmean_reverse_control", "diffmean_reverse")]:
                for split in ("val", "test"):
                    d = b[m][split]
                    dn = b["direction_norm"].get(tag.replace("_reverse", ""), "")
                    wr.writerow([tag, split, b[m]["locked_thr"], d["net"], d["fixed"], d["broke"],
                                 d["before"], d["after"], dn, b["cos_diffmean_pca1"], TASKB.name])

    # placebo control summary
    if PLACEBO.exists():
        p = json.load(open(PLACEBO))
        with open(OUT / "placebo_control_summary_qwen7b.csv", "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["variant", "net", "fixed", "broke", "n_touched", "detail", "source_file"])
            for k in ("real", "reverse", "ungated"):
                if k in p:
                    v = p[k]
                    wr.writerow([k, v.get("net"), v.get("fixed"), v.get("broke"),
                                 v.get("n_touched"), "", PLACEBO.name])
            rd = p.get("random_distribution", {})
            if rd:
                wr.writerow(["random_mean", rd.get("mean"), "", "", "",
                             f"std={rd.get('std')} max={rd.get('max')} real_ge_all={rd.get('real_ge_all_random')}",
                             PLACEBO.name])
    log.info("consolidated diagnostics -> ca_direct layer / direction method / ca_tc pca / placebo CSVs")


# ══════════════════════════════════════════════════════════════════════════════
# Net = Fixed - Broke consistency audit
# ══════════════════════════════════════════════════════════════════════════════
def net_consistency():
    checks = []

    def add(name, src, fixed, broke, net):
        ok = (fixed is not None and broke is not None and net is not None and fixed - broke == net)
        checks.append({"item": name, "source_file": src, "fixed": fixed, "broke": broke,
                       "net": net, "fixed_minus_broke": (fixed - broke) if (fixed is not None and broke is not None) else None,
                       "consistent": ok})

    for name, path in NET_FILES:
        if not path.exists():
            w(f"net-check file missing: {path}"); continue
        d = json.load(open(path))
        rel = str(path.relative_to(ROOT))
        if name == "multiseed_per_seed":
            for s in d.get("per_seed", []):
                add(f"multiseed_seed{s.get('seed')}", rel, s.get("fixed"), s.get("broke"), s.get("net"))
        elif name == "qwen_placebo_real":
            for k in ("real", "reverse", "ungated"):
                if k in d:
                    add(f"placebo_{k}", rel, d[k].get("fixed"), d[k].get("broke"), d[k].get("net"))
        elif name == "taskA_cascade":
            for k in ("archived_cascade", "updated_cascade"):
                c = d[k]; add(f"taskA_{k}", rel, c["fixed"], c["broke"], c["net"])
        elif name == "taskB_ca_tc_pca":
            for m in ("diffmean", "pca1"):
                for split in ("val", "test"):
                    add(f"taskB_{m}_{split}", rel, d[m][split]["fixed"], d[m][split]["broke"], d[m][split]["net"])
        elif name == "partC_directions":
            for ch, v in d.items():
                for m, dd in v["detail"].items():
                    add(f"partC_{ch}_{m}", rel, dd["fixed"], dd["broke"], dd["net"])
        else:
            add(name, rel, d.get("fixed"), d.get("broke"), d.get("net"))

    with open(OUT / "net_consistency_checks.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["item", "source_file", "fixed", "broke", "net",
                                           "fixed_minus_broke", "consistent"])
        wr.writeheader()
        for c in checks:
            wr.writerow(c)
    n_ok = sum(1 for c in checks if c["consistent"])
    log.info("net-consistency: %d/%d rows satisfy Net = Fixed - Broke", n_ok, len(checks))
    return checks, n_ok


# ══════════════════════════════════════════════════════════════════════════════
def write_summary_csv(qwen, meta, phi, ts):
    with open(OUT / "channel_discovery_summary.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["dataset", "model", "channel_id", "gold", "pred", "count_all", "count_train",
                     "count_test", "status", "recompute_type", "source_file"])
        if qwen:
            src = qwen["provenance"]["source_file"]
            for c in qwen["discovered_channels"]:
                wr.writerow(["w2c_qwen25_7b", "Qwen2.5-7B", c["channel_id"], c["gold"], c["pred"],
                             c["count_all"], c["count_train"], c["count_test"], c["status"],
                             "EXACT_per_sample", src])
        if meta:
            src = meta["provenance"]["source_file"]
            rt = meta["provenance"]["recompute_type"]
            for c in meta["discovered_channels"]:
                wr.writerow(["metatool_binary", meta.get("model"), c["channel_id"], c["gold"],
                             c["pred"], c["count_all"], "", "", c["status"], rt, src])
        if phi:
            src = phi["provenance"]["source_files"][0]
            for c in phi["discovered_channels"]:
                wr.writerow(["w2c_phi35", "Phi-3.5", c["channel_id"], c["gold"], c["pred"], "",
                             "", c["count_test"], c["status"], "SUMMARY_ONLY", src])
        if ts:
            wr.writerow(["toolsandbox", "-", "-", "-", "-", "", "", "", ts["status"],
                         "QUALITATIVE_ONLY", ts["provenance"].get("candidate_file") or "-"])


def main():
    log.info("=== SAKIKO channel discovery (CPU-only, no model) ===")
    qwen = discover_qwen()
    meta = discover_metatool()
    phi = discover_phi()
    ts = discover_toolsandbox()
    write_summary_csv(qwen, meta, phi, ts)
    consolidate_diagnostics()
    checks, n_ok = net_consistency()
    log.info("=== discovery complete; outputs under %s ===", OUT.relative_to(ROOT))
    if WARNINGS:
        log.info("%d warning(s) recorded", len(WARNINGS))


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 (cross-dataset): generalized, mode-aware channel discovery
# ══════════════════════════════════════════════════════════════════════════════
# A candidate channel is an ordered baseline error transition gold_label -> pred_label.
# The channel count is NOT fixed; it is selected by observable evidence under
# configurable thresholds, with bootstrap stability, threshold sensitivity, and
# (where activations exist) geometry-only and hybrid modes.
#
# Usage:  python scripts/discover_sakiko_channels.py --cross-dataset
# Legacy consolidation mode (unchanged):  python scripts/discover_sakiko_channels.py

OUT2 = ROOT / "final" / "results" / "cross_dataset_channel_discovery"

PHI_TEST_DETAILS = ROOT / "final/results/clean/p0_final_test_details.jsonl"
METATOOL_SPLIT_DIR = ROOT / "data/processed/metatool_binary"
QWEN_ACT_DIR = ROOT / "data/processed/qwen25_7b_w2c/cache"

# Configurable threshold settings. "default" follows the Phase-1 suggested values;
# the two alternatives exist for sensitivity analysis and are NOT treated as truth.
THRESHOLD_SETTINGS = {
    "default": {"stable_train_min": 50, "stable_eval_min": 15, "weak_train_min": 20},
    "strict":  {"stable_train_min": 75, "stable_eval_min": 25, "weak_train_min": 30},
    "lenient": {"stable_train_min": 30, "stable_eval_min": 10, "weak_train_min": 10},
}
N_BOOTSTRAP = 200
# R0 — baseline-validity gate (added after the ACEBench pilot exposed readout collapse):
# a transition analysis is only meaningful if the baseline readout is not degenerate.
# Flag collapse when BOTH hold: one predicted class exceeds COLLAPSE_PRED_SHARE of predictions
# AND accuracy is below the trivial majority-gold baseline.
COLLAPSE_PRED_SHARE = 0.60
GEOM_KS = [2, 3, 4, 5, 6]
GEOM_LAYERS = [16, 20]           # activations cached for L12/16/18/20/22; 16+20 span the story
GEOM_MIN_CLUSTER = 20            # minimum cluster size for a meaningful subgroup
GEOM_SPLIT_SIL = 0.15            # within-transition split supported only above this silhouette
GEOM_SPLIT_ARI = 0.5             # ...and only if split-half stable
GEOM_MERGE_AUC = 0.65            # pairwise probe AUC below this = geometrically indistinct
GEOM_MERGE_COS = 0.90            # ...and direction cosine above this

# hand-selected channels actually used by archived SAKIKO interventions
ARCHIVED_CHANNELS = {
    "w2c_qwen25_7b": {"rfi_tc", "ca_tc", "ca_direct"},
    "w2c_phi35": {"rfi_tc", "ca_tc", "ca_direct"},
    "metatool_binary": {"no_tool__tool_call"},
}


def classify(train_n, eval_n, th, split_known=True):
    """Status under one threshold setting. eval_n = min(val,test) support where known."""
    if not split_known:
        # no per-split counts -> classify on total with an explicit flag upstream
        if train_n is None:
            return "unknown_split"
    if train_n >= th["stable_train_min"] and (eval_n is None or eval_n >= th["stable_eval_min"]):
        return "stable"
    if train_n >= th["weak_train_min"]:
        return "weak"
    return "insufficient"


def transition_discovery(rows, dataset, model, archived, split_field="split",
                         gold_field="gold", pred_field="pred", note=""):
    """Mode 1 (transition-only). rows: list of dicts with gold/pred (+ optional split).
    Returns full result dict incl. bootstrap stability + threshold sensitivity + P/R."""
    import random
    golds = [r[gold_field] for r in rows]
    preds = [r[pred_field] for r in rows]
    has_split = all(split_field in r and r[split_field] for r in rows)
    gold_labels = sorted(set(golds))
    pred_labels = sorted(set(preds) | set(golds))

    conf = {g: {p: 0 for p in pred_labels} for g in gold_labels}
    trans_all, trans_split = Counter(), defaultdict(Counter)
    for r in rows:
        g, p = r[gold_field], r[pred_field]
        conf[g][p] += 1
        if g != p:
            trans_all[(g, p)] += 1
            if has_split:
                trans_split[r[split_field]][(g, p)] += 1

    total_err = sum(trans_all.values())

    def counts_of(t):
        tr = trans_split["train"].get(t) if has_split else None
        va = trans_split["val"].get(t, 0) if has_split else None
        te = trans_split["test"].get(t, 0) if has_split else None
        ev = min(va, te) if has_split else None
        return tr or 0 if has_split else None, va, te, ev

    # bootstrap frequency stability (resample rows; how often is each transition stable@default?)
    rng = random.Random(0)
    boot_stable = Counter()
    idx = list(range(len(rows)))
    for _ in range(N_BOOTSTRAP):
        samp = [rows[rng.randrange(len(rows))] for _ in idx]
        ta, ts_ = Counter(), defaultdict(Counter)
        for r in samp:
            g, p = r[gold_field], r[pred_field]
            if g != p:
                ta[(g, p)] += 1
                if has_split:
                    ts_[r[split_field]][(g, p)] += 1
        for t in ta:
            if has_split:
                st = classify(ts_["train"].get(t, 0),
                              min(ts_["val"].get(t, 0), ts_["test"].get(t, 0)),
                              THRESHOLD_SETTINGS["default"])
            else:
                st = "stable" if ta[t] >= THRESHOLD_SETTINGS["default"]["stable_train_min"] else "sub"
            if st == "stable":
                boot_stable[t] += 1

    channels, sens_rows = [], []
    for (g, p), c in sorted(trans_all.items(), key=lambda kv: -kv[1]):
        tr, va, te, ev = counts_of((g, p))
        statuses = {}
        for name, th in THRESHOLD_SETTINGS.items():
            if has_split:
                statuses[name] = classify(tr, ev, th)
            else:  # no split info: use total count against train_min (flagged)
                statuses[name] = ("stable" if c >= th["stable_train_min"]
                                  else "weak" if c >= th["weak_train_min"] else "insufficient")
        cid = NAMED.get((g, p), f"{g}__{p}")
        channels.append({
            "channel_id": cid, "gold": g, "pred": p,
            "count_all": c, "count_train": tr, "count_val": va, "count_test": te,
            "pct_of_errors": round(100 * c / total_err, 2) if total_err else 0,
            "status_default": statuses["default"], "status_strict": statuses["strict"],
            "status_lenient": statuses["lenient"],
            "bootstrap_stable_freq": round(boot_stable[(g, p)] / N_BOOTSTRAP, 3),
            "in_archived_set": cid in archived,
        })
        for name in THRESHOLD_SETTINGS:
            sens_rows.append([dataset, cid, name, statuses[name]])

    stable_default = [c["channel_id"] for c in channels if c["status_default"] == "stable"]
    n_by_setting = {name: sum(1 for c in channels if c[f"status_{name}"] == "stable")
                    for name in THRESHOLD_SETTINGS}

    # precision/recall of automatic discovery vs archived hand-selected channels
    auto, arch = set(stable_default), set(archived)
    tp = auto & arch
    pr = {
        "auto_stable": sorted(auto), "archived": sorted(arch),
        "true_positive": sorted(tp),
        "extra_auto_channels": sorted(auto - arch),
        "missed_archived": sorted(arch - auto),
        "precision_vs_archived": round(len(tp) / len(auto), 3) if auto else None,
        "recall_vs_archived": round(len(tp) / len(arch), 3) if arch else None,
        "topk_by_count_equals_archived":
            set(c["channel_id"] for c in channels[:len(arch)]) == arch if arch else None,
    }

    acc = sum(1 for g, p in zip(golds, preds) if g == p) / len(rows)

    # R0 baseline-validity gate
    gold_c, pred_c = Counter(golds), Counter(preds)
    majority_gold_share = max(gold_c.values()) / len(rows)
    majority_pred_share = max(pred_c.values()) / len(rows)
    collapsed = majority_pred_share > COLLAPSE_PRED_SHARE and acc < majority_gold_share
    validity = {
        "majority_gold_share": round(majority_gold_share, 4),
        "majority_pred_share": round(majority_pred_share, 4),
        "majority_pred_class": max(pred_c, key=pred_c.get),
        "acc_minus_majority_gold": round(acc - majority_gold_share, 4),
        "readout_collapse_flag": collapsed,
        "rule": f"collapse iff majority_pred_share > {COLLAPSE_PRED_SHARE} AND acc < majority_gold_share",
    }

    n_stable_vals = set(n_by_setting.values())
    if collapsed:
        mode = "INVALID_READOUT_no_channel_claims"
    elif len(stable_default) == 0:
        mode = "zero_channel_or_qualitative"
    elif len(stable_default) == 1:
        mode = "one_channel"
    else:
        mode = "multi_channel"
    return {
        "dataset": dataset, "model": model, "note": note,
        "n_samples": len(rows), "baseline_accuracy": round(acc, 5),
        "total_errors": total_err, "has_split_info": has_split,
        "native_gold_labels": gold_labels, "native_pred_labels": pred_labels,
        "confusion_matrix_gold_by_pred": conf,
        "threshold_settings": THRESHOLD_SETTINGS, "n_bootstrap": N_BOOTSTRAP,
        "baseline_validity": validity,
        "discovered_channels": channels,
        "stable_channels_default": stable_default,
        "n_stable_by_threshold_setting": n_by_setting,
        "channel_count_stable_across_thresholds": len(n_stable_vals) == 1,
        "precision_recall_vs_archived": pr,
        "recommended_sakiko_mode": mode,
    }, sens_rows


def geometry_and_hybrid_qwen(result):
    """Modes 2 (geometry-only) and 3 (hybrid) for W2C+Qwen, the only dataset with
    cached activations. CPU-only (sklearn on the local .npy caches)."""
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    det = [json.loads(l) for l in open(QWEN_BASELINE)]
    golds = np.array([d["gold"] for d in det])
    preds = np.array([d["pred"] for d in det])
    err_mask = golds != preds
    trans_lbl = np.array([f"{g}__{p}" if g != p else "correct" for g, p in zip(golds, preds)])

    geometry = {"n_errors": int(err_mask.sum()), "layers": {}}
    hybrid_rows = []
    stable = [c for c in result["discovered_channels"] if c["status_default"] == "stable"]

    for L in GEOM_LAYERS:
        acts = np.load(QWEN_ACT_DIR / f"acts_L{L}.npy", mmap_mode="r")
        E = np.asarray(acts[err_mask], dtype=np.float32)
        elbl = trans_lbl[err_mask]
        layer = {"kmeans": []}

        # ── mode 2: unsupervised clustering of error activations ──
        rng = np.random.RandomState(0)
        for K in GEOM_KS:
            km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(E)
            sil = float(silhouette_score(E, km.labels_, sample_size=min(1500, len(E)),
                                         random_state=0))
            sizes = np.bincount(km.labels_)
            # split-half stability: cluster each half, compare on half B via ARI
            aris = []
            for rep in range(5):
                perm = rng.permutation(len(E))
                A, B = perm[: len(E) // 2], perm[len(E) // 2:]
                kA = KMeans(n_clusters=K, n_init=5, random_state=rep).fit(E[A])
                kB = KMeans(n_clusters=K, n_init=5, random_state=rep + 100).fit(E[B])
                aris.append(adjusted_rand_score(kA.predict(E[B]), kB.labels_))
            nmi = float(normalized_mutual_info_score(elbl, km.labels_))
            layer["kmeans"].append({
                "K": K, "silhouette": round(sil, 4),
                "min_cluster_size": int(sizes.min()), "max_cluster_size": int(sizes.max()),
                "split_half_ari_mean": round(float(np.mean(aris)), 4),
                "nmi_vs_transition_labels": round(nmi, 4),
                "nmi_vs_gold_label": round(float(
                    normalized_mutual_info_score(golds[err_mask], km.labels_)), 4),
                "nmi_vs_pred_label": round(float(
                    normalized_mutual_info_score(preds[err_mask], km.labels_)), 4),
                "meets_min_cluster": bool(sizes.min() >= GEOM_MIN_CLUSTER),
            })

        # ── mode 3a: within-transition split test ──
        ref_mean = {}
        for g in set(golds):
            m = (golds == g) & ~err_mask
            if m.sum() >= 10:
                ref_mean[g] = np.asarray(acts[m], dtype=np.float32).mean(0)
        for c in stable:
            m = elbl == f"{c['gold']}__{c['pred']}"   # elbl uses raw transition strings
            X = E[m]
            if len(X) < 2 * GEOM_MIN_CLUSTER:
                continue
            km2 = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)
            sil2 = float(silhouette_score(X, km2.labels_))
            sizes2 = np.bincount(km2.labels_)
            aris = []
            for rep in range(5):
                perm = np.random.RandomState(rep).permutation(len(X))
                A, B = perm[: len(X) // 2], perm[len(X) // 2:]
                if len(A) < 4 or len(B) < 4:
                    continue
                kA = KMeans(n_clusters=2, n_init=5, random_state=rep).fit(X[A])
                kB = KMeans(n_clusters=2, n_init=5, random_state=rep + 100).fit(X[B])
                aris.append(adjusted_rand_score(kA.predict(X[B]), kB.labels_))
            supported = (sil2 >= GEOM_SPLIT_SIL and sizes2.min() >= GEOM_MIN_CLUSTER
                         and float(np.mean(aris)) >= GEOM_SPLIT_ARI)
            hybrid_rows.append(["split_test", L, c["channel_id"], "",
                                round(sil2, 4), int(sizes2.min()),
                                round(float(np.mean(aris)), 4), "", "",
                                "SPLIT_SUPPORTED" if supported else "keep_single"])

        # ── mode 3b: pairwise merge test between stable transitions ──
        for i in range(len(stable)):
            for j in range(i + 1, len(stable)):
                a, b = stable[i], stable[j]
                Xa = E[elbl == f"{a['gold']}__{a['pred']}"]
                Xb = E[elbl == f"{b['gold']}__{b['pred']}"]
                if len(Xa) < GEOM_MIN_CLUSTER or len(Xb) < GEOM_MIN_CLUSTER:
                    continue
                X = np.vstack([Xa, Xb]); y = np.r_[np.zeros(len(Xa)), np.ones(len(Xb))]
                auc = float(np.mean(cross_val_score(
                    LogisticRegression(max_iter=2000), X, y, cv=5, scoring="roc_auc")))
                cos = ""
                if a["gold"] in ref_mean and b["gold"] in ref_mean:
                    da = ref_mean[a["gold"]] - Xa.mean(0)
                    db = ref_mean[b["gold"]] - Xb.mean(0)
                    cos = round(float(np.dot(da, db) /
                                      (np.linalg.norm(da) * np.linalg.norm(db) + 1e-9)), 4)
                merge = (auc < GEOM_MERGE_AUC and isinstance(cos, float) and cos > GEOM_MERGE_COS)
                hybrid_rows.append(["merge_test", L, a["channel_id"], b["channel_id"],
                                    "", "", "", round(auc, 4), cos,
                                    "MERGE_CANDIDATE" if merge else "keep_separate"])
        geometry["layers"][str(L)] = layer
        log.info("geometry L%d done (best sil across K: %.3f)", L,
                 max(k["silhouette"] for k in layer["kmeans"]))

    geometry["criteria"] = {"min_cluster": GEOM_MIN_CLUSTER, "split_sil": GEOM_SPLIT_SIL,
                            "split_ari": GEOM_SPLIT_ARI, "merge_auc": GEOM_MERGE_AUC,
                            "merge_cos": GEOM_MERGE_COS,
                            "note": ("silhouette ~ 0 everywhere = no cluster structure; channels are "
                                     "linear directions. Do NOT claim semantic clusters unless "
                                     "silhouette, stability, and size criteria all pass.")}
    return geometry, hybrid_rows


def cross_dataset_main():
    OUT2.mkdir(parents=True, exist_ok=True)
    all_sens, comparison, pr_rows = [], [], []

    # ── 1. W2C + Qwen2.5-7B (EXACT, full, with activations) ──
    det = [json.loads(l) for l in open(QWEN_BASELINE)]
    split_of = {}
    for s in ("train", "val", "test"):
        for i in map(int, json.load(open(SPLIT_DIR / f"{s}_idx.json"))):
            split_of[i] = s
    rows = [{"gold": d["gold"], "pred": d["pred"], "split": split_of.get(i)}
            for i, d in enumerate(det)]
    qwen, sens = transition_discovery(rows, "w2c_qwen25_7b", "Qwen2.5-7B-Instruct",
                                      ARCHIVED_CHANNELS["w2c_qwen25_7b"],
                                      note="EXACT full per-sample + seed-42 splits + activations")
    all_sens += sens
    geom, hybrid_rows = geometry_and_hybrid_qwen(qwen)
    qwen["geometry"] = geom
    json.dump(qwen, open(OUT2 / "w2c_qwen_channel_discovery.json", "w"), indent=2)
    with open(OUT2 / "hybrid_channel_tests.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["test", "layer", "channel_a", "channel_b", "within_silhouette",
                     "min_subcluster", "split_half_ari", "pairwise_probe_auc",
                     "direction_cosine", "verdict"])
        wr.writerows(hybrid_rows)

    # ── 2. W2C + Phi-3.5 (EXACT, test split only) ──
    pdet = [json.loads(l) for l in open(PHI_TEST_DETAILS)]
    # all rows are test split -> omit split field entirely so classification falls back to
    # total-count-vs-train_min (single-split data has no train/val support by construction)
    prows = [{"gold": d["gold"], "pred": d["clean_pred"]} for d in pdet]
    phi, sens = transition_discovery(
        prows, "w2c_phi35", "Phi-3.5-mini-instruct", ARCHIVED_CHANNELS["w2c_phi35"],
        note="EXACT but TEST SPLIT ONLY (548 rows); train/val per-sample never archived -> "
             "status uses total(=test) count vs train_min; treat as test_split_exact")
    phi["recompute_type"] = "test_split_exact"
    json.dump(phi, open(OUT2 / "w2c_phi_channel_discovery.json", "w"), indent=2)
    all_sens += sens

    # ── 3. MetaTool-Binary + Phi (EXACT, full, with archived splits) ──
    mdet = [json.loads(l) for l in open(METATOOL_DET)]
    msplit = {}
    for s in ("train", "val", "test"):
        for l in open(METATOOL_SPLIT_DIR / f"metatool_binary_{s}.jsonl"):
            msplit[json.loads(l)["sample_id"]] = s
    mrows = [{"gold": d["gold_response_mode"], "pred": d["pred_response_mode"],
              "split": msplit.get(d["sample_id"])} for d in mdet]
    meta, sens = transition_discovery(mrows, "metatool_binary", "Phi-3.5-mini-instruct",
                                      ARCHIVED_CHANNELS["metatool_binary"],
                                      note="EXACT full per-sample (upgraded from summary via small LFS pull)")
    json.dump(meta, open(OUT2 / "metatool_channel_discovery.json", "w"), indent=2)
    all_sens += sens

    # ── 4. ToolSandbox (qualitative-only negative control) — assessment md written by caller ──

    # comparison CSV + precision/recall CSV
    with open(OUT2 / "channel_discovery_comparison.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["dataset", "model", "recompute_type", "channel_id", "gold", "pred",
                     "count_all", "count_train", "count_val", "count_test", "pct_of_errors",
                     "status_default", "status_strict", "status_lenient",
                     "bootstrap_stable_freq", "in_archived_set"])
        for res, rt in [(qwen, "EXACT_full"), (phi, "test_split_exact"), (meta, "EXACT_full")]:
            for c in res["discovered_channels"]:
                wr.writerow([res["dataset"], res["model"], rt, c["channel_id"], c["gold"],
                             c["pred"], c["count_all"], c["count_train"], c["count_val"],
                             c["count_test"], c["pct_of_errors"], c["status_default"],
                             c["status_strict"], c["status_lenient"],
                             c["bootstrap_stable_freq"], c["in_archived_set"]])
        wr.writerow(["toolsandbox", "-", "QUALITATIVE_ONLY", "-", "-", "-",
                     0, "", "", "", "", "zero_channel", "zero_channel", "zero_channel", "", ""])

    with open(OUT2 / "threshold_sensitivity.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["dataset", "channel_id", "threshold_setting", "status"])
        wr.writerows(all_sens)

    with open(OUT2 / "discovery_precision_recall.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["dataset", "auto_stable_channels", "archived_channels", "true_positive",
                     "extra_auto", "missed_archived", "precision", "recall",
                     "topk_by_count_equals_archived", "n_stable_default", "n_stable_strict",
                     "n_stable_lenient", "count_stable_across_thresholds",
                     "recommended_mode"])
        for res in (qwen, phi, meta):
            pr = res["precision_recall_vs_archived"]
            ns = res["n_stable_by_threshold_setting"]
            wr.writerow([res["dataset"], ";".join(pr["auto_stable"]), ";".join(pr["archived"]),
                         ";".join(pr["true_positive"]), ";".join(pr["extra_auto_channels"]),
                         ";".join(pr["missed_archived"]), pr["precision_vs_archived"],
                         pr["recall_vs_archived"], pr["topk_by_count_equals_archived"],
                         ns["default"], ns["strict"], ns["lenient"],
                         res["channel_count_stable_across_thresholds"],
                         res["recommended_sakiko_mode"]])
        wr.writerow(["toolsandbox", "", "", "", "", "", "", "", "", 0, 0, 0, True,
                     "zero_channel_or_qualitative"])

    for res in (qwen, phi, meta):
        log.info("%s: stable=%s mode=%s P=%.3s R=%.3s", res["dataset"],
                 res["stable_channels_default"], res["recommended_sakiko_mode"],
                 str(res["precision_recall_vs_archived"]["precision_vs_archived"]),
                 str(res["precision_recall_vs_archived"]["recall_vs_archived"]))
    log.info("=== cross-dataset discovery complete -> %s ===", OUT2.relative_to(ROOT))


if __name__ == "__main__":
    if "--cross-dataset" in sys.argv:
        cross_dataset_main()
    else:
        main()
