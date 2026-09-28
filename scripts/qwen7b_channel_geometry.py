"""
qwen7b_channel_geometry.py — Qwen2.5-7B channel geometry from LOCAL activation caches (CPU-only).
=================================================================================================
Computes small, committable geometry summaries that back the channel-adaptive story:
  * per-layer linear-probe AUC (channel-error vs correct)         -> qwen7b_probe_auc_by_layer.csv
  * per-layer silhouette of the channel/correct label structure    -> qwen7b_silhouette_by_layer.csv
  * per-channel per-layer direction geometry (DiffMean norm, etc.)  -> channel_layer_geometry_qwen7b.csv
  * DiffMean direction cosine matrix (recomputed + committed .npz)  -> qwen7b_direction_cosine_matrix.csv
  * 2-D PCA coordinates of channel/correct samples (for a scatter)  -> qwen7b_pca2d_channel_points.csv

Reads the LOCAL, UNCOMMITTED activation caches (data/processed/qwen25_7b_w2c/cache/acts_L*.npy,
3652 x 3584, row-aligned to qwen25_7b_baseline_details.jsonl). It NEVER writes activation arrays
out — only small derived numbers. Loads NO model weights. If the caches are absent, every geometry
output is skipped and marked TODO_needs_activations.

Usage:  python scripts/qwen7b_channel_geometry.py
"""
from __future__ import annotations
import json, csv, sys, logging
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "final" / "results" / "channel_adaptive"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = ROOT / "data/processed/qwen25_7b_w2c/cache"
BASELINE = ROOT / "final/results/7b_w2c_baseline/qwen25_7b_baseline_details.jsonl"
COMMITTED_NPZ = ROOT / "final/results/7b_w2c_sakiko/qwen25_7b_directions.npz"
SPLIT_DIR = ROOT / "sakiko_v3/results/splits"
LAYERS = [12, 16, 18, 20, 22]
CHANNELS = {  # channel -> (gold, pred, reference gold for the DiffMean "correct" pool)
    "rfi_tc": ("request_for_info", "tool_call", "request_for_info"),
    "ca_tc": ("cannot_answer", "tool_call", "cannot_answer"),
    "ca_direct": ("cannot_answer", "direct", "cannot_answer"),
}
SEED = 42
PCA2D_LAYER = 20                 # layer for the 2-D PCA scatter
PCA2D_MAX_CORRECT = 400          # subsample the correct pool to keep the CSV small

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("geometry")


def load_meta():
    det = [json.loads(l) for l in open(BASELINE)]
    meta = []
    for d in det:
        g, p = d["gold"], d["pred"]
        if g == p:
            et = "correct"
        elif (g, p) == ("request_for_info", "tool_call"):
            et = "rfi_tc"
        elif (g, p) == ("cannot_answer", "tool_call"):
            et = "ca_tc"
        elif (g, p) == ("cannot_answer", "direct"):
            et = "ca_direct"
        else:
            et = f"{g}__{p}"
        meta.append({"gold": g, "pred": p, "correct": g == p, "etype": et})
    return meta


def main():
    if not all((CACHE / f"acts_L{L}.npy").exists() for L in LAYERS):
        log.warning("Activation caches missing -> ALL geometry outputs TODO_needs_activations")
        json.dump({"status": "TODO_needs_activations",
                   "reason": "local acts_L*.npy caches absent"},
                  open(OUT / "GEOMETRY_STATUS.json", "w"), indent=2)
        return
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score, silhouette_samples

    meta = load_meta()
    n = len(meta)
    acts = {L: np.load(CACHE / f"acts_L{L}.npy", mmap_mode="r") for L in LAYERS}
    for L in LAYERS:
        assert acts[L].shape[0] == n, (L, acts[L].shape, n)
    log.info("Loaded %d-row activations at layers %s (dim=%d)", n, LAYERS, acts[LAYERS[0]].shape[1])

    idx_correct = [i for i in range(n) if meta[i]["correct"]]
    idx_ch = {ch: [i for i in range(n) if meta[i]["etype"] == ch] for ch in CHANNELS}
    ref_ch = {ch: [i for i in range(n) if meta[i]["correct"] and meta[i]["gold"] == g]
              for ch, (g, p, g_ref) in CHANNELS.items()}

    # ── 1. per-layer linear-probe AUC (pos=channel error, neg=ALL correct) — matches router def ──
    with open(OUT / "qwen7b_probe_auc_by_layer.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["channel", "obs_layer", "cv_auc_5fold", "n_pos", "n_neg", "method", "source"])
        for ch in CHANNELS:
            pos = idx_ch[ch]; neg = idx_correct
            y = np.array([1] * len(pos) + [0] * len(neg))
            for L in LAYERS:
                X = np.asarray(acts[L][pos + neg], dtype=np.float32)
                Xs = StandardScaler().fit_transform(X)
                clf = LogisticRegression(max_iter=2000, C=1.0, solver="liblinear", random_state=SEED)
                cv = float(cross_val_score(clf, Xs, y, cv=5, scoring="roc_auc").mean())
                wr.writerow([ch, L, round(cv, 4), len(pos), len(neg), "LR_liblinear_5foldCV",
                             "local acts_L%d.npy (uncommitted)" % L])
                log.info("  probe %s L%d cv_auc=%.4f", ch, L, cv)

    # ── 2. per-channel per-layer direction geometry (DiffMean norm + PCA-1 alignment) ──
    def diffmean(L, err, ref):
        A = np.asarray(acts[L]); raw = A[ref].mean(0) - A[err].mean(0)
        nrm = float(np.linalg.norm(raw)); return raw / (nrm + 1e-12), nrm

    units_native = {}
    with open(OUT / "channel_layer_geometry_qwen7b.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["channel", "obs_layer", "diffmean_norm", "median_act_norm", "n_err", "n_ref",
                     "cos_diffmean_pca1", "is_native_layer", "source"])
        native = {"rfi_tc": 20, "ca_tc": 20, "ca_direct": 16}
        for ch, (g, p, g_ref) in CHANNELS.items():
            err, ref = idx_ch[ch], ref_ch[ch]
            for L in LAYERS:
                unit, nrm = diffmean(L, err, ref)
                A = np.asarray(acts[L])
                med = float(np.median(np.linalg.norm(A[err + ref], axis=1)))
                # PCA-1 alignment
                sub = np.asarray(acts[L][err + ref], dtype=np.float32)
                sc = StandardScaler().fit(sub)
                pc1 = PCA(n_components=1, random_state=SEED).fit(sc.transform(sub)).components_[0]
                pc1_raw = pc1 / sc.scale_; pc1_raw = pc1_raw / (np.linalg.norm(pc1_raw) + 1e-12)
                cos = abs(float(unit @ pc1_raw))
                wr.writerow([ch, L, round(nrm, 4), round(med, 4), len(err), len(ref),
                             round(cos, 4), int(L == native[ch]), "local acts_L%d.npy" % L])
                if L == native[ch]:
                    units_native[ch] = unit
        log.info("  channel_layer_geometry written (5 layers x 3 channels)")

    # ── 3. direction cosine matrix (recomputed native-layer units + committed .npz cross-check) ──
    chs = list(CHANNELS)
    with open(OUT / "qwen7b_direction_cosine_matrix.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["method", "channel_a", "channel_b", "layer_a", "layer_b", "cosine",
                     "same_layer", "source"])
        native = {"rfi_tc": 20, "ca_tc": 20, "ca_direct": 16}
        # (a) native-layer units (rfi_tc/ca_tc at L20, ca_direct at L16): cross-layer flagged
        for i in range(len(chs)):
            for j in range(i, len(chs)):
                a, b = chs[i], chs[j]
                cos = float(units_native[a] @ units_native[b])
                wr.writerow(["recomputed_native", a, b, native[a], native[b], round(cos, 4),
                             int(native[a] == native[b]), "local acts (native layer)"])
        # (b) all-at-L20 units (apples-to-apples, incl. ca_direct recomputed at L20)
        u20 = {ch: diffmean(20, idx_ch[ch], ref_ch[ch])[0] for ch in chs}
        for i in range(len(chs)):
            for j in range(i, len(chs)):
                a, b = chs[i], chs[j]
                wr.writerow(["recomputed_L20_common", a, b, 20, 20, round(float(u20[a] @ u20[b]), 4),
                             1, "local acts_L20.npy"])
        # (c) committed .npz cross-check (native layers as stored)
        if COMMITTED_NPZ.exists():
            z = np.load(COMMITTED_NPZ)
            zu = {ch: z[f"{ch}_unit"] for ch in chs if f"{ch}_unit" in z}
            for i in range(len(chs)):
                for j in range(i, len(chs)):
                    a, b = chs[i], chs[j]
                    if a in zu and b in zu and zu[a].shape == zu[b].shape:
                        wr.writerow(["committed_npz", a, b, native[a], native[b],
                                     round(float(zu[a] @ zu[b]), 4), int(native[a] == native[b]),
                                     "qwen25_7b_directions.npz"])
    log.info("  direction cosine matrix written (recomputed + committed cross-check)")

    # ── 4. silhouette by layer (labels: 3 channels + correct, on PCA-50) ──
    labeled = idx_correct + [i for ch in CHANNELS for i in idx_ch[ch]]
    lab = (["correct"] * len(idx_correct) +
           [ch for ch in CHANNELS for _ in idx_ch[ch]])
    lab = np.array(lab)
    with open(OUT / "qwen7b_silhouette_by_layer.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["obs_layer", "silhouette_overall", "sil_rfi_tc", "sil_ca_tc", "sil_ca_direct",
                     "sil_correct", "n_labeled", "pca_dims", "source"])
        for L in LAYERS:
            X = np.asarray(acts[L][labeled], dtype=np.float32)
            Xs = StandardScaler().fit_transform(X)
            Xp = PCA(n_components=50, random_state=SEED).fit_transform(Xs)
            overall = float(silhouette_score(Xp, lab))
            sv = silhouette_samples(Xp, lab)
            per = {c: round(float(sv[lab == c].mean()), 4) for c in ["rfi_tc", "ca_tc", "ca_direct", "correct"]}
            wr.writerow([L, round(overall, 4), per["rfi_tc"], per["ca_tc"], per["ca_direct"],
                         per["correct"], len(labeled), 50, "local acts_L%d.npy" % L])
            log.info("  silhouette L%d overall=%.4f", L, overall)

    # ── 5. 2-D PCA scatter points (subsample correct) ──
    rng = np.random.RandomState(SEED)
    corr_sub = list(rng.choice(idx_correct, min(PCA2D_MAX_CORRECT, len(idx_correct)), replace=False))
    pts_idx = corr_sub + [i for ch in CHANNELS for i in idx_ch[ch]]
    pts_lab = (["correct"] * len(corr_sub) + [ch for ch in CHANNELS for _ in idx_ch[ch]])
    split_of = {}
    if all((SPLIT_DIR / f"{s}_idx.json").exists() for s in ("train", "val", "test")):
        for s in ("train", "val", "test"):
            for i in map(int, json.load(open(SPLIT_DIR / f"{s}_idx.json"))):
                split_of[i] = s
    Xp = PCA(n_components=2, random_state=SEED).fit_transform(
        StandardScaler().fit_transform(np.asarray(acts[PCA2D_LAYER][pts_idx], dtype=np.float32)))
    with open(OUT / "qwen7b_pca2d_channel_points.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["row_idx", "channel_or_correct", "split", "obs_layer", "pc1", "pc2", "source"])
        for k, i in enumerate(pts_idx):
            wr.writerow([i, pts_lab[k], split_of.get(i, ""), PCA2D_LAYER,
                         round(float(Xp[k, 0]), 4), round(float(Xp[k, 1]), 4),
                         "local acts_L%d.npy" % PCA2D_LAYER])
    log.info("  pca2d points written (correct subsample=%d, layer L%d)", len(corr_sub), PCA2D_LAYER)
    log.info("=== geometry complete; outputs under %s ===", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
