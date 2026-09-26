#!/usr/bin/env python3
"""Data-free Qwen vocabulary-effect audit for recovered SAKIKO directions.

This is an explicitly non-equivalent alternative to the historical four-mode
alignment audit.  It reads three recovered unit directions plus only
``model.norm.weight`` and chunked slices of ``lm_head.weight`` directly from
safetensors on CPU.  It does not load a model, run a forward pass, open a
dataset/split, or reconstruct a direction.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open
from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
OUT = (
    ROOT
    / "final"
    / "results"
    / "static_output_head_alignment"
    / "remote_recovery"
)
SUMMARY_CSV = OUT / "qwen_static_vocabulary_effect_summary.csv"
EXTREMA_CSV = OUT / "qwen_static_vocabulary_effect_extrema.csv"
REPORT = OUT / "QWEN_DIRECTION_REMOTE_RECOVERY_AND_ALTERNATIVE.md"
MANIFEST = OUT / "qwen_static_vocabulary_effect_manifest.json"

MODEL_DIR = (
    ROOT / ".cache" / "modelscope" / "Qwen" / "Qwen2___5-7B-Instruct"
)
CONFIG = MODEL_DIR / "config.json"
INDEX = MODEL_DIR / "model.safetensors.index.json"
TOKENIZER_JSON = MODEL_DIR / "tokenizer.json"
TOKENIZER_CONFIG = MODEL_DIR / "tokenizer_config.json"
POINTER = (
    ROOT
    / "final"
    / "results"
    / "7b_w2c_sakiko"
    / "qwen25_7b_directions.npz"
)

EXPECTED_DIRECTION_OID = (
    "6d765083b1bb9d9127b8c03946b4ed2c7a402b51213c748fe37fae024ed6b34c"
)
EXPECTED_DIRECTION_SIZE = 44586
EXPECTED_HIDDEN = 3584
EXPECTED_VOCAB = 152064
CHUNK_ROWS = 4096
TOP_K = 25

CHANNEL_META = {
    "rfi_tc": {
        "direction_key": "rfi_tc_unit",
        "gold": "request_for_info",
        "source": "tool_call",
        "observation_layer": 20,
        "injection_layer": 18,
    },
    "ca_tc": {
        "direction_key": "ca_tc_unit",
        "gold": "cannot_answer",
        "source": "tool_call",
        "observation_layer": 20,
        "injection_layer": 16,
    },
    "ca_direct": {
        "direction_key": "ca_direct_unit",
        "gold": "cannot_answer",
        "source": "direct",
        "observation_layer": 16,
        "injection_layer": 16,
    },
}


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.rstrip()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def parse_lfs_pointer() -> tuple[str, int]:
    text = POINTER.read_text(encoding="ascii")
    match = re.fullmatch(
        r"version https://git-lfs\.github\.com/spec/v1\n"
        r"oid sha256:([0-9a-f]{64})\n"
        r"size ([0-9]+)\n?",
        text,
    )
    if not match:
        raise RuntimeError(f"not a canonical LFS pointer: {POINTER}")
    return match.group(1), int(match.group(2))


def direction_object_path(oid: str) -> Path:
    git_dir = Path(git("rev-parse", "--git-dir"))
    if not git_dir.is_absolute():
        git_dir = ROOT / git_dir
    return git_dir / "lfs" / "objects" / oid[:2] / oid[2:4] / oid


def load_directions(path: Path) -> tuple[list[str], np.ndarray, dict[str, Any]]:
    channels = list(CHANNEL_META)
    vectors = []
    details: dict[str, Any] = {}
    with np.load(path, allow_pickle=False) as archive:
        expected_keys = {
            *(meta["direction_key"] for meta in CHANNEL_META.values()),
            "rfi_tc_meta",
            "ca_tc_meta",
            "ca_direct_meta",
        }
        if set(archive.files) != expected_keys:
            raise RuntimeError(
                f"unexpected NPZ keys: {archive.files}; expected {expected_keys}"
            )
        for channel in channels:
            key = CHANNEL_META[channel]["direction_key"]
            vector = np.asarray(archive[key], dtype=np.float32)
            if vector.shape != (EXPECTED_HIDDEN,):
                raise RuntimeError(f"{key} shape {vector.shape}")
            if not np.isfinite(vector).all():
                raise RuntimeError(f"{key} contains non-finite values")
            norm = float(np.linalg.norm(vector))
            if abs(norm - 1.0) > 2e-6:
                raise RuntimeError(f"{key} is not unit length: {norm}")
            vectors.append(vector)
            meta = np.asarray(archive[f"{channel}_meta"], dtype=np.float32)
            details[channel] = {
                **CHANNEL_META[channel],
                "direction_l2_norm": norm,
                "stored_observation_layer": int(meta[0]),
                "stored_median_activation_norm": float(meta[1]),
            }
            if int(meta[0]) != CHANNEL_META[channel]["observation_layer"]:
                raise RuntimeError(f"{channel} observation-layer mismatch")
    matrix = np.stack(vectors, axis=1)
    return channels, matrix, details


def locate_output_parameters() -> tuple[Path, str, str]:
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    weight_map = index["weight_map"]
    head_name = "lm_head.weight"
    norm_name = "model.norm.weight"
    head_shard = weight_map.get(head_name)
    norm_shard = weight_map.get(norm_name)
    if head_shard is None or norm_shard is None:
        raise RuntimeError("lm_head.weight/model.norm.weight missing from index")
    if head_shard != norm_shard:
        raise RuntimeError("audit expects head and norm in one shard")
    return MODEL_DIR / head_shard, head_name, norm_name


def compute_scores(
    shard: Path,
    head_name: str,
    norm_name: str,
    directions: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    torch.set_grad_enabled(False)
    torch.set_num_threads(max(1, min(16, torch.get_num_threads())))
    with safe_open(shard, framework="pt", device="cpu") as tensors:
        head_slice = tensors.get_slice(head_name)
        head_shape = tuple(head_slice.get_shape())
        norm_shape = tuple(tensors.get_slice(norm_name).get_shape())
        if head_shape != (EXPECTED_VOCAB, EXPECTED_HIDDEN):
            raise RuntimeError(f"unexpected head shape: {head_shape}")
        if norm_shape != (EXPECTED_HIDDEN,):
            raise RuntimeError(f"unexpected norm shape: {norm_shape}")

        gain = tensors.get_tensor(norm_name).to(torch.float32)
        if not torch.isfinite(gain).all():
            raise RuntimeError("non-finite final norm gain")
        direction_tensor = torch.from_numpy(directions)
        effective_directions = gain[:, None] * direction_tensor
        scores = np.empty(
            (EXPECTED_VOCAB, directions.shape[1]), dtype=np.float32
        )
        for start in range(0, EXPECTED_VOCAB, CHUNK_ROWS):
            stop = min(start + CHUNK_ROWS, EXPECTED_VOCAB)
            head_chunk = head_slice[start:stop, :].to(torch.float32)
            chunk_scores = head_chunk @ effective_directions
            scores[start:stop] = chunk_scores.numpy()

    parameter_info = {
        "head_parameter": head_name,
        "norm_parameter": norm_name,
        "head_shape": list(head_shape),
        "norm_shape": list(norm_shape),
        "head_dtype": "BF16",
        "norm_dtype": "BF16",
        "tied_branch": "untied",
        "norm_branch": "RMSNorm",
        "mean_centered": False,
        "gain_min": float(gain.min().item()),
        "gain_max": float(gain.max().item()),
        "gain_mean": float(gain.mean().item()),
    }
    return scores, parameter_info


def quantile_dict(values: np.ndarray) -> dict[str, float]:
    quantiles = np.quantile(
        values,
        [0.0001, 0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999, 0.9999],
    )
    names = [
        "p0001",
        "p001",
        "p01",
        "p05",
        "p50",
        "p95",
        "p99",
        "p999",
        "p9999",
    ]
    return {name: float(value) for name, value in zip(names, quantiles)}


def safe_token_text(tokenizer: Tokenizer, token_id: int) -> str:
    try:
        text = tokenizer.decode([token_id], skip_special_tokens=False)
    except Exception as exc:
        return f"<DECODE_ERROR:{type(exc).__name__}>"
    return text.encode("unicode_escape").decode("ascii")


def summarize_scores(
    channels: list[str],
    scores: np.ndarray,
    tokenizer: Tokenizer,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows = []
    extrema_rows = []
    for column, channel in enumerate(channels):
        values = scores[:, column].astype(np.float64)
        summary = {
            "model": "Qwen2.5-7B-Instruct",
            "channel": channel,
            "gold": CHANNEL_META[channel]["gold"],
            "source": CHANNEL_META[channel]["source"],
            "direction_layer": CHANNEL_META[channel]["observation_layer"],
            "injection_layer": CHANNEL_META[channel]["injection_layer"],
            "hidden_dimension": EXPECTED_HIDDEN,
            "vocabulary_dimension": EXPECTED_VOCAB,
            "score_mean": float(values.mean()),
            "score_std": float(values.std()),
            "score_min": float(values.min()),
            "score_max": float(values.max()),
            "fraction_positive": float(np.mean(values > 0)),
            **quantile_dict(values),
        }
        summary_rows.append(summary)

        low_ids = np.argpartition(values, TOP_K)[:TOP_K]
        high_ids = np.argpartition(values, -TOP_K)[-TOP_K:]
        low_ids = low_ids[np.argsort(values[low_ids])]
        high_ids = high_ids[np.argsort(values[high_ids])[::-1]]
        for tail, ids in (("negative", low_ids), ("positive", high_ids)):
            for rank, token_id in enumerate(ids, start=1):
                extrema_rows.append(
                    {
                        "model": "Qwen2.5-7B-Instruct",
                        "channel": channel,
                        "tail": tail,
                        "rank": rank,
                        "token_id": int(token_id),
                        "token_text_escaped": safe_token_text(
                            tokenizer, int(token_id)
                        ),
                        "static_coefficient": float(values[token_id]),
                    }
                )
    return summary_rows, extrema_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(map(cell, headers)) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(cell(value) for value in row) + " |"
        for row in rows
    )
    return "\n".join(lines)


def build_report(
    *,
    generated_utc: str,
    source_head: str,
    direction_path: Path,
    direction_details: dict[str, Any],
    parameter_info: dict[str, Any],
    summary_rows: list[dict[str, Any]],
    extrema_rows: list[dict[str, Any]],
    pairwise_cosines: dict[str, float],
    shard_sha256: str,
) -> str:
    recovery_rows = []
    for channel, details in direction_details.items():
        recovery_rows.append(
            [
                channel,
                details["stored_observation_layer"],
                details["injection_layer"],
                f"{details['direction_l2_norm']:.9f}",
                f"{details['stored_median_activation_norm']:.6f}",
                "stored in valid local LFS object",
            ]
        )
    distribution_rows = []
    for row in summary_rows:
        distribution_rows.append(
            [
                row["channel"],
                f"{row['score_mean']:.6g}",
                f"{row['score_std']:.6g}",
                f"{row['p01']:.6g}",
                f"{row['p50']:.6g}",
                f"{row['p99']:.6g}",
                f"{row['score_min']:.6g}",
                f"{row['score_max']:.6g}",
                f"{row['fraction_positive']:.6f}",
            ]
        )
    extrema_by_channel: dict[str, list[dict[str, Any]]] = {
        channel: [] for channel in CHANNEL_META
    }
    for row in extrema_rows:
        if row["rank"] <= 10:
            extrema_by_channel[row["channel"]].append(row)
    extrema_sections = []
    for channel, rows in extrema_by_channel.items():
        extrema_sections.append(
            f"### {channel}\n\n"
            + markdown_table(
                ["tail", "rank", "token ID", "decoded token", "coefficient"],
                [
                    [
                        row["tail"],
                        row["rank"],
                        row["token_id"],
                        f"`{row['token_text_escaped']}`",
                        f"{row['static_coefficient']:.7g}",
                    ]
                    for row in rows
                ],
            )
        )

    extrema_markdown = "\n\n".join(extrema_sections)

    return f"""# Qwen Direction Remote Recovery and Static Vocabulary-Effect Audit

**Tier:** POST-HOC DESCRIPTIVE / LIGHT
**Generated UTC:** `{generated_utc}`
**Source Git HEAD:** `{source_head}`
**Mechanical outcome:** `QWEN_DIRECTIONS_RECOVERED_ALTERNATIVE_AUDIT_COMPLETE`

## Result

The exact Qwen direction LFS object is present on the GitHub remote and was
fetched narrowly into the local Git LFS object cache. Its SHA256 equals the
committed pointer OID:

- object path: `{direction_path}`
- SHA256: `{EXPECTED_DIRECTION_OID}`
- size: `{EXPECTED_DIRECTION_SIZE}` bytes
- worktree checkout performed: `false`

All three deployed Qwen vectors are finite `float32[3584]` unit vectors:

{markdown_table(
    ["channel", "obs", "inj", "L2 norm", "stored median act norm", "status"],
    recovery_rows,
)}

Pairwise direction cosines: `{json.dumps(pairwise_cosines, sort_keys=True)}`.

## Why the original four-mode decomposition still stops

The historical scorer selects `sample["answers"][mode]`, tokenizes that
sample-specific answer, and averages log-probability across all answer tokens.
It does not define four fixed one-token verbalizers. Therefore exact mode token
IDs, a four-mode output-head subspace, `a_contrast`, `a_offaxis`, `a_dec`, and
`a_orth` do not exist as data-free quantities for that historical readout.

Using the strings `tool_call`, `request_for_info`, `cannot_answer`, and
`direct` as replacement token targets would be a new readout, not a recovery.
This audit does not do that.

## Alternative computed here

For each vocabulary row `t` and recovered unit direction `d`, this audit
computes the static RMSNorm-gain proxy coefficient:

```text
q_t = gain ⊙ W_U[t]
s_t(d) = q_t^T d
```

`lm_head.weight` and `model.norm.weight` were read directly and only from:

- shard: `{MODEL_DIR / "model-00004-of-00004.safetensors"}`
- shard SHA256: `{shard_sha256}`
- head parameter/shape: `{parameter_info['head_parameter']}`
  `{parameter_info['head_shape']}` BF16
- norm parameter/shape: `{parameter_info['norm_parameter']}`
  `{parameter_info['norm_shape']}` BF16
- branch: untied output head + RMSNorm; no mean removal

The shard was streamed in `{CHUNK_ROWS}`-row slices. No model object or forward
pass was created.

## Vocabulary-wide coefficient distributions

{markdown_table(
    [
        "channel",
        "mean",
        "std",
        "p01",
        "p50",
        "p99",
        "min",
        "max",
        "fraction > 0",
    ],
    distribution_rows,
)}

These are static output-row coefficients per unit residual perturbation, not
probabilities, generated predictions, or behavioral effects.

## Largest-magnitude vocabulary rows

{extrema_markdown}

The full top/bottom `{TOP_K}` rows per channel are in
`qwen_static_vocabulary_effect_extrema.csv`.

## Interpretation boundary

Allowed interpretation:

- a positive coefficient means the static gain-weighted unembedding row has a
  positive inner product with the stored direction;
- the extrema identify vocabulary rows most aligned or anti-aligned with that
  data-free proxy.

Not established:

- historical four-mode alignment;
- sequence-level score changes;
- post-intervention logits or predictions;
- any causal mechanism or behavioral correction;
- propagation from the injection layer through later transformer blocks.

This is a static parameter-space proxy. The true activation-conditioned
Jacobian of RMSNorm depends on the hidden activation and is not computed here.

## Execution exclusions

- CPU only;
- zero full-model loading;
- zero forward/backward pass;
- zero inference or prediction generation;
- zero dataset, split, sealed-index, or activation-cache access;
- zero direction reconstruction;
- zero additional LFS object download beyond the exact 44,586-byte direction
  object;
- zero Qwen3 or Phase 10.3 work.
"""


def main() -> None:
    dirty = git("status", "--short").splitlines()
    allowed = {
        str(SCRIPT.relative_to(ROOT)),
        str(OUT.relative_to(ROOT)) + "/",
    }
    unrelated = [
        line
        for line in dirty
        if line[3:] not in allowed
        and not line[3:].startswith(str(OUT.relative_to(ROOT)) + "/")
    ]
    if unrelated:
        raise RuntimeError(f"unrelated dirty worktree entries: {unrelated}")

    oid, declared_size = parse_lfs_pointer()
    if oid != EXPECTED_DIRECTION_OID or declared_size != EXPECTED_DIRECTION_SIZE:
        raise RuntimeError("direction pointer identity changed")
    direction_path = direction_object_path(oid)
    if not direction_path.is_file():
        raise RuntimeError(
            "exact LFS object absent; fetch only the committed Qwen direction path"
        )
    if (
        direction_path.stat().st_size != EXPECTED_DIRECTION_SIZE
        or sha256(direction_path) != EXPECTED_DIRECTION_OID
    ):
        raise RuntimeError("local LFS direction object failed integrity check")

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config.get("hidden_size") != EXPECTED_HIDDEN:
        raise RuntimeError("hidden-size mismatch")
    if config.get("vocab_size") != EXPECTED_VOCAB:
        raise RuntimeError("vocabulary-size mismatch")
    if config.get("tie_word_embeddings") is not False:
        raise RuntimeError("expected untied output head")
    if "rms_norm_eps" not in config:
        raise RuntimeError("expected RMSNorm configuration")

    channels, directions, direction_details = load_directions(direction_path)
    shard, head_name, norm_name = locate_output_parameters()
    scores, parameter_info = compute_scores(
        shard, head_name, norm_name, directions
    )
    if not np.isfinite(scores).all():
        raise RuntimeError("non-finite vocabulary coefficients")

    tokenizer = Tokenizer.from_file(str(TOKENIZER_JSON))
    summary_rows, extrema_rows = summarize_scores(channels, scores, tokenizer)
    pairwise_cosines = {
        f"{channels[i]}|{channels[j]}": float(
            directions[:, i] @ directions[:, j]
        )
        for i in range(len(channels))
        for j in range(i + 1, len(channels))
    }

    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(SUMMARY_CSV, summary_rows)
    write_csv(EXTREMA_CSV, extrema_rows)

    generated_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    source_head = git("rev-parse", "HEAD")
    shard_sha256 = sha256(shard)
    report_text = build_report(
        generated_utc=generated_utc,
        source_head=source_head,
        direction_path=direction_path,
        direction_details=direction_details,
        parameter_info=parameter_info,
        summary_rows=summary_rows,
        extrema_rows=extrema_rows,
        pairwise_cosines=pairwise_cosines,
        shard_sha256=shard_sha256,
    )
    REPORT.write_text(report_text, encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "title": "Qwen recovered-direction static vocabulary-effect audit",
        "tier": "POST-HOC DESCRIPTIVE / LIGHT",
        "generated_utc": generated_utc,
        "git": {
            "head": source_head,
            "branch": git("branch", "--show-current"),
        },
        "mechanical_outcome": (
            "QWEN_DIRECTIONS_RECOVERED_ALTERNATIVE_AUDIT_COMPLETE"
        ),
        "non_equivalence": {
            "replaces_historical_four_mode_readout": False,
            "computes_a_contrast_or_a_dec": False,
            "reason": (
                "historical mode scores are sample-dependent multi-token "
                "candidate-sequence mean log-probabilities"
            ),
        },
        "direction_object": {
            "pointer_path": str(POINTER.relative_to(ROOT)),
            "local_lfs_object_path": str(direction_path),
            "sha256": EXPECTED_DIRECTION_OID,
            "size_bytes": EXPECTED_DIRECTION_SIZE,
            "remote_fetch_scope": str(POINTER.relative_to(ROOT)),
            "worktree_checkout_performed": False,
            "channels": direction_details,
            "pairwise_cosines": pairwise_cosines,
        },
        "output_parameters": {
            **parameter_info,
            "shard_path": str(shard),
            "shard_size_bytes": shard.stat().st_size,
            "shard_sha256": shard_sha256,
            "config_path": str(CONFIG),
            "config_sha256": sha256(CONFIG),
            "index_path": str(INDEX),
            "index_sha256": sha256(INDEX),
        },
        "tokenizer": {
            "implementation": "tokenizers.Tokenizer.from_file; decode only",
            "tokenizer_json_path": str(TOKENIZER_JSON),
            "tokenizer_json_sha256": sha256(TOKENIZER_JSON),
            "tokenizer_config_path": str(TOKENIZER_CONFIG),
            "tokenizer_config_sha256": sha256(TOKENIZER_CONFIG),
            "model_or_auto_model_loaded": False,
        },
        "method": {
            "proxy": "q_t = gain * W_U[t]; s_t(d) = q_t dot d",
            "head_chunk_rows": CHUNK_ROWS,
            "top_k_each_tail": TOP_K,
            "summary_rows": summary_rows,
        },
        "software_versions": {
            "python": sys.version.split()[0],
            "numpy": package_version("numpy"),
            "torch": package_version("torch"),
            "safetensors": package_version("safetensors"),
            "tokenizers": package_version("tokenizers"),
        },
        "artifacts": {
            str(SCRIPT.relative_to(ROOT)): {"sha256": sha256(SCRIPT)},
            str(SUMMARY_CSV.relative_to(ROOT)): {
                "sha256": sha256(SUMMARY_CSV),
                "rows": len(summary_rows),
            },
            str(EXTREMA_CSV.relative_to(ROOT)): {
                "sha256": sha256(EXTREMA_CSV),
                "rows": len(extrema_rows),
            },
            str(REPORT.relative_to(ROOT)): {"sha256": sha256(REPORT)},
        },
        "consistency_checks": {
            "direction_oid_matches_pointer": True,
            "direction_object_size_matches": True,
            "three_unit_directions": True,
            "all_direction_values_finite": True,
            "head_and_direction_dimensions_match": True,
            "all_vocabulary_coefficients_finite": True,
            "summary_has_three_rows": len(summary_rows) == 3,
            "extrema_has_expected_rows": (
                len(extrema_rows) == len(CHANNEL_META) * TOP_K * 2
            ),
            "worktree_pointer_not_checked_out": POINTER.stat().st_size == 130,
        },
        "constraint_attestations": {
            "cpu_only": True,
            "gpu_used": False,
            "full_model_loaded": False,
            "forward_pass": False,
            "backward_pass": False,
            "inference_or_prediction": False,
            "dataset_accessed": False,
            "split_or_sealed_index_accessed": False,
            "activation_cache_accessed": False,
            "direction_reconstructed": False,
            "qwen3_work": False,
            "phase10_3_modified": False,
        },
    }
    MANIFEST.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "outcome": manifest["mechanical_outcome"],
                "summary_rows": len(summary_rows),
                "extrema_rows": len(extrema_rows),
                "report": str(REPORT.relative_to(ROOT)),
                "manifest": str(MANIFEST.relative_to(ROOT)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
