#!/usr/bin/env python3
"""Gated Qwen2.5 Stage A pilot and dev-only Jacobian geometry.

The script has two explicit stages:

* ``pilot``: 4 archived-error dev samples per channel, selected by ascending
  UUID, to measure numerical noise, validate hooks/gradients, and project cost.
* ``full``: only runs after a passing pilot whose projected GPU time is <=2 h.

It never opens the project test index. Dataset access is guarded by the frozen
train/dev index allow lists. JSONL files spanning all project splits decode
only allow-listed line numbers.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import io
import json
import math
import os
import platform
import subprocess
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from datasets import load_dataset
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "final" / "results" / "qwen25_stage_a_jacobian_geometry"
MODEL_DIR = ROOT / ".cache" / "modelscope" / "Qwen" / "Qwen2___5-7B-Instruct"
TRAIN_INDEX = ROOT / "final" / "results" / "splits" / "train_idx.json"
DEV_INDEX = ROOT / "final" / "results" / "splits" / "val_idx.json"
ACT_DIR = ROOT / "data" / "processed" / "qwen25_7b_w2c" / "cache"

DIRECTION_OID = (
    "6d765083b1bb9d9127b8c03946b4ed2c7a402b51213c748fe37fae024ed6b34c"
)
BASELINE_OID = (
    "e66a2faff706e44aff1370df79663aaf9257c3699f10b9b3d3d3731fb5903a79"
)
SPLIT_BASELINE_OID = (
    "d6d7c409e388d0debf6d7c60653e3149999429116e5d48fb14c777b507eca6b0"
)
DATASET_REVISION = "0582f7749df63a96fdc3070932e83e72396ace53"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"

SEED = 42
NULL_SEED = 20260728
PILOT_PER_CHANNEL = 4
MAX_PROMPT_TOKENS = 8192
MAX_GPU_SECONDS = 2 * 60 * 60
DIMENSION = 3584
MODES = ["tool_call", "direct", "request_for_info", "cannot_answer"]
CHANNELS = {
    "rfi_tc": {
        "gold": "request_for_info",
        "source": "tool_call",
        "obs_layer": 20,
        "inj_layer": 18,
        "rho": 2.0,
        "threshold": 0.7,
        "median_norm": 39.23229217529297,
    },
    "ca_tc": {
        "gold": "cannot_answer",
        "source": "tool_call",
        "obs_layer": 20,
        "inj_layer": 16,
        "rho": 4.0,
        "threshold": 0.8,
        "median_norm": 39.23229217529297,
    },
    "ca_direct": {
        "gold": "cannot_answer",
        "source": "direct",
        "obs_layer": 16,
        "inj_layer": 16,
        "rho": 6.0,
        "threshold": 0.4,
        "median_norm": 19.090181350708008,
    },
}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def lfs_object(oid: str) -> Path:
    path = ROOT / ".git" / "lfs" / "objects" / oid[:2] / oid[2:4] / oid
    if not path.is_file():
        raise FileNotFoundError(f"missing LFS object {oid}")
    return path


def load_indices() -> tuple[list[int], list[int], set[int]]:
    train = list(map(int, json.loads(TRAIN_INDEX.read_text())))
    dev = list(map(int, json.loads(DEV_INDEX.read_text())))
    if set(train) & set(dev):
        raise ValueError("train/dev overlap")
    return train, dev, set(train) | set(dev)


def safe_jsonl(path: Path, allowed: set[int]) -> dict[int, dict[str, Any]]:
    """JSON-decode only allow-listed line numbers."""
    rows: dict[int, dict[str, Any]] = {}
    with path.open("rb") as handle:
        for index, line in enumerate(handle):
            if index in allowed:
                rows[index] = json.loads(line)
    if set(rows) != allowed:
        raise ValueError(f"safe JSONL selection incomplete: {path}")
    return rows


def load_directions() -> dict[str, np.ndarray]:
    payload = lfs_object(DIRECTION_OID).read_bytes()
    if sha256_bytes(payload) != DIRECTION_OID:
        raise ValueError("direction SHA256 mismatch")
    archive = np.load(io.BytesIO(payload), allow_pickle=False)
    output: dict[str, np.ndarray] = {}
    for channel in CHANNELS:
        value = archive[f"{channel}_unit"]
        if (
            value.shape != (DIMENSION,)
            or value.dtype != np.float32
            or not np.isfinite(value).all()
            or abs(float(np.linalg.norm(value)) - 1.0) > 1e-5
        ):
            raise ValueError(f"invalid direction {channel}")
        output[channel] = value.copy()
    return output


class SealedData:
    """Allow-list wrapper around the cached, shuffled upstream dataset."""

    def __init__(self) -> None:
        self.train, self.dev, self.allowed = load_indices()
        self._dataset = load_dataset(
            "nvidia/When2Call", "test", split="mcq"
        ).shuffle(seed=SEED)
        if len(self._dataset) != 3652:
            raise ValueError(f"unexpected upstream row count {len(self._dataset)}")
        self.archived = safe_jsonl(lfs_object(BASELINE_OID), self.allowed)
        self.split_archived = safe_jsonl(
            lfs_object(SPLIT_BASELINE_OID), self.allowed
        )
        for index in self.allowed:
            if (
                self.archived[index]["uuid"]
                != self.split_archived[index]["uuid"]
            ):
                raise ValueError(f"archived UUID mismatch at allowed row {index}")

    def sample(self, index: int) -> dict[str, Any]:
        if index not in self.allowed:
            raise PermissionError(f"non-train/dev row access denied: {index}")
        sample = self._dataset[index]
        expected = self.archived[index]["uuid"]
        if sample["uuid"] != expected:
            raise ValueError(f"dataset UUID mismatch at {index}")
        return sample


def parse_tools(tools: list[Any] | None) -> list[Any]:
    output: list[Any] = []
    for tool in tools or []:
        if isinstance(tool, str):
            try:
                output.append(json.loads(tool))
            except json.JSONDecodeError:
                output.append({"raw": tool})
        else:
            output.append(tool)
    return output


def prompt_text(tokenizer: Any, sample: dict[str, Any]) -> str:
    parts = ["You are a helpful assistant."]
    tools = parse_tools(sample["tools"])
    if tools:
        parts.append(
            "You have access to the following tools:\n"
            + json.dumps(tools, indent=2, ensure_ascii=False)
        )
    else:
        parts.append("No tools are available.")
    parts.append(
        "Given the user's question, choose the most appropriate response "
        "from the provided options."
    )
    messages = [
        {"role": "system", "content": "\n\n".join(parts)},
        {"role": "user", "content": sample["question"]},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        fallback = "".join(
            f"<|{message['role']}|>\n{message['content']}\n"
            for message in messages
        )
        return fallback + "<|assistant|>\n"


def configure_determinism() -> dict[str, Any]:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    return {
        "seed": SEED,
        "numpy_seed": SEED,
        "torch_seed": SEED,
        "cuda_seed_all": SEED,
        "CUBLAS_WORKSPACE_CONFIG": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "torch_deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "allow_tf32": False,
    }


def load_model() -> tuple[Any, Any, torch.device, dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"expected one visible GPU, found {torch.cuda.device_count()}"
        )
    deterministic = configure_determinism()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()
    model.requires_grad_(False)
    model.config.use_cache = False
    device = next(model.parameters()).device
    environment = {
        "model_repository": "Qwen/Qwen2.5-7B-Instruct",
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "model_path": str(MODEL_DIR.relative_to(ROOT)),
        "dtype": "torch.bfloat16",
        "batch_size": 1,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "datasets_version": __import__("datasets").__version__,
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "attention_implementation": getattr(
            model.config, "_attn_implementation", "unknown"
        ),
        "model_load_seconds": time.perf_counter() - started,
        "determinism": deterministic,
    }
    return model, tokenizer, device, environment


def fit_reconstructed_routers(
    data: SealedData,
) -> tuple[dict[str, Any], dict[str, Any]]:
    routers: dict[str, Any] = {}
    audit: dict[str, Any] = {}
    for channel, config in CHANNELS.items():
        layer = config["obs_layer"]
        path = ACT_DIR / f"acts_L{layer}.npy"
        acts = np.load(path, mmap_mode="r")
        if acts.shape != (3652, DIMENSION) or acts.dtype != np.float32:
            raise ValueError(f"activation cache mismatch {path}: {acts.shape}")
        positives = [
            index
            for index in data.train
            if (
                data.archived[index]["gold"] == config["gold"]
                and data.archived[index]["pred"] == config["source"]
            )
        ]
        negatives = [
            index for index in data.train if data.archived[index]["correct"]
        ]
        row_order = negatives + positives
        x = np.asarray(acts[row_order], dtype=np.float32)
        y = np.asarray(
            [0] * len(negatives) + [1] * len(positives), dtype=np.int64
        )
        scaler = StandardScaler()
        xs = scaler.fit_transform(x)
        classifier = LogisticRegression(
            max_iter=2000,
            C=1.0,
            solver="liblinear",
            random_state=SEED,
        )
        classifier.fit(xs, y)
        dev_x = np.asarray(acts[data.dev], dtype=np.float32)
        dev_probability = classifier.predict_proba(
            scaler.transform(dev_x)
        )[:, 1]
        probabilities = {
            index: float(probability)
            for index, probability in zip(data.dev, dev_probability)
        }
        routers[channel] = {
            "scaler": scaler,
            "classifier": classifier,
            "probability": probabilities,
        }
        train_slice_hash = sha256_bytes(
            np.asarray(acts[data.train], dtype=np.float32).tobytes(order="C")
        )
        dev_slice_hash = sha256_bytes(dev_x.tobytes(order="C"))
        audit[channel] = {
            "status": "RECONSTRUCTED_NOT_ORIGINAL_SERIALIZED_ROUTER",
            "obs_layer": layer,
            "threshold": config["threshold"],
            "binary_gate": True,
            "n_train_positive": len(positives),
            "n_train_correct_negative": len(negatives),
            "train_row_order": "correct negatives, then channel-error positives",
            "train_activation_slice_sha256": train_slice_hash,
            "dev_activation_slice_sha256": dev_slice_hash,
            "scaler_mean_sha256": sha256_bytes(
                scaler.mean_.astype(np.float64).tobytes()
            ),
            "scaler_scale_sha256": sha256_bytes(
                scaler.scale_.astype(np.float64).tobytes()
            ),
            "classifier_coef_sha256": sha256_bytes(
                classifier.coef_.astype(np.float64).tobytes()
            ),
            "classifier_intercept_sha256": sha256_bytes(
                classifier.intercept_.astype(np.float64).tobytes()
            ),
        }
        del acts, x, xs, dev_x
    return routers, audit


def top_prediction(scores: dict[str, float]) -> tuple[str, str, float]:
    ordered = sorted(
        MODES, key=lambda mode: (-scores[mode], MODES.index(mode))
    )
    return (
        ordered[0],
        ordered[1],
        float(scores[ordered[0]] - scores[ordered[1]]),
    )


def score_candidate(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    prompt: str,
    candidate: str,
    inj_layer: int | None = None,
    correction: np.ndarray | None = None,
    jacobian: bool = False,
    capture_prompt_state: bool = False,
) -> dict[str, Any]:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    candidate_ids = tokenizer.encode(candidate, add_special_tokens=False)
    if not candidate_ids:
        raise ValueError("empty candidate")
    if len(prompt_ids) > MAX_PROMPT_TOKENS:
        raise ValueError(f"prompt length {len(prompt_ids)} exceeds historical cap")
    input_ids = torch.tensor(
        [prompt_ids + candidate_ids], device=device, dtype=torch.long
    )
    prompt_len = len(prompt_ids)
    candidate_len = len(candidate_ids)
    leaf_box: dict[str, torch.Tensor] = {}
    state_box: dict[str, torch.Tensor] = {}
    handle = None
    if inj_layer is not None:
        module = model.model.layers[inj_layer].mlp

        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            is_tuple = isinstance(output, tuple)
            tensor = output[0] if is_tuple else output
            if capture_prompt_state:
                state_box["prompt"] = (
                    tensor[0, :prompt_len, :].detach().to("cpu").float()
                )
            replacement = tensor
            if jacobian:
                replacement = tensor.detach().requires_grad_(True)
                leaf_box["value"] = replacement
            if correction is not None:
                correction_tensor = torch.as_tensor(
                    correction, dtype=torch.bfloat16, device=device
                )
                replacement = replacement + correction_tensor
            if is_tuple:
                return (replacement,) + output[1:]
            return replacement

        handle = module.register_forward_hook(hook)

    started = time.perf_counter()
    grad_context = nullcontext() if jacobian else torch.no_grad()
    try:
        with grad_context:
            logits = model(input_ids, use_cache=False).logits
            score_logits = logits[
                0,
                prompt_len - 1: prompt_len + candidate_len - 1,
                :,
            ]
            log_probs = torch.log_softmax(score_logits.float(), dim=-1)
            targets = torch.tensor(
                candidate_ids, dtype=torch.long, device=device
            )
            score_tensor = log_probs[
                torch.arange(candidate_len, device=device), targets
            ].mean()
            if jacobian:
                if "value" not in leaf_box:
                    raise RuntimeError("injection leaf was not captured")
                gradient = torch.autograd.grad(
                    score_tensor,
                    leaf_box["value"],
                    retain_graph=False,
                    create_graph=False,
                )[0]
                if gradient is None or not torch.isfinite(gradient).all():
                    raise RuntimeError("missing or non-finite Jacobian")
                gradient_f32 = gradient[0].detach().float().cpu().numpy()
                g_prompt = gradient_f32[:prompt_len].sum(axis=0)
                g_full = gradient_f32.sum(axis=0)
                gradient_shape = list(gradient.shape)
                gradient_finite = True
            else:
                g_prompt = None
                g_full = None
                gradient_shape = None
                gradient_finite = None
            score = float(score_tensor.detach().cpu())
    finally:
        if handle is not None:
            handle.remove()
    elapsed = time.perf_counter() - started
    prompt_state = state_box.get("prompt")
    del input_ids
    if "logits" in locals():
        del logits, score_logits, log_probs, score_tensor
    if "gradient" in locals():
        del gradient
    return {
        "score": score,
        "runtime_seconds": elapsed,
        "prompt_tokens": prompt_len,
        "candidate_tokens": candidate_len,
        "actual_positions": prompt_len + candidate_len,
        "g_prompt": g_prompt,
        "g_full": g_full,
        "gradient_shape": gradient_shape,
        "gradient_finite": gradient_finite,
        "prompt_state": prompt_state,
    }


def score_sample(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    sample: dict[str, Any],
    inj_layer: int | None = None,
    correction: np.ndarray | None = None,
) -> dict[str, Any]:
    prompt = prompt_text(tokenizer, sample)
    scores: dict[str, float] = {}
    lengths: dict[str, int] = {}
    runtimes: dict[str, float] = {}
    prompt_tokens = None
    for mode in MODES:
        result = score_candidate(
            model,
            tokenizer,
            device,
            prompt,
            sample["answers"].get(mode, ""),
            inj_layer=inj_layer,
            correction=correction,
        )
        scores[mode] = result["score"]
        lengths[mode] = result["candidate_tokens"]
        runtimes[mode] = result["runtime_seconds"]
        prompt_tokens = result["prompt_tokens"]
    prediction, runner_up, margin = top_prediction(scores)
    return {
        "scores": scores,
        "prediction": prediction,
        "runner_up": runner_up,
        "top1_top2_margin": margin,
        "candidate_tokens": lengths,
        "prompt_tokens": prompt_tokens,
        "mode_runtime_seconds": runtimes,
        "runtime_seconds": sum(runtimes.values()),
    }


def jacobian_sample(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    sample: dict[str, Any],
    inj_layer: int,
    capture_prompt_states: bool,
) -> dict[str, Any]:
    prompt = prompt_text(tokenizer, sample)
    scores: dict[str, float] = {}
    g_prompt: dict[str, np.ndarray] = {}
    g_full: dict[str, np.ndarray] = {}
    candidate_tokens: dict[str, int] = {}
    actual_positions: dict[str, int] = {}
    gradient_shapes: dict[str, list[int]] = {}
    mode_runtime: dict[str, float] = {}
    reference_state: torch.Tensor | None = None
    max_prompt_deviation = 0.0
    max_prompt_reference_magnitude = 0.0
    prompt_difference_sum = 0.0
    prompt_difference_elements = 0
    prompt_nonzero_difference_elements = 0
    for mode in MODES:
        result = score_candidate(
            model,
            tokenizer,
            device,
            prompt,
            sample["answers"].get(mode, ""),
            inj_layer=inj_layer,
            jacobian=True,
            capture_prompt_state=capture_prompt_states,
        )
        scores[mode] = result["score"]
        g_prompt[mode] = result["g_prompt"]
        g_full[mode] = result["g_full"]
        candidate_tokens[mode] = result["candidate_tokens"]
        actual_positions[mode] = result["actual_positions"]
        gradient_shapes[mode] = result["gradient_shape"]
        mode_runtime[mode] = result["runtime_seconds"]
        state = result["prompt_state"]
        if capture_prompt_states:
            if reference_state is None:
                reference_state = state
                max_prompt_reference_magnitude = float(
                    torch.max(torch.abs(reference_state))
                )
            else:
                difference = torch.abs(state - reference_state)
                deviation = float(torch.max(difference))
                max_prompt_deviation = max(max_prompt_deviation, deviation)
                prompt_difference_sum += float(torch.sum(difference))
                prompt_difference_elements += difference.numel()
                prompt_nonzero_difference_elements += int(
                    torch.count_nonzero(difference)
                )
        del state
        gc.collect()
        torch.cuda.empty_cache()
    return {
        "scores": scores,
        "g_prompt": g_prompt,
        "g_full": g_full,
        "candidate_tokens": candidate_tokens,
        "actual_positions": actual_positions,
        "gradient_shapes": gradient_shapes,
        "gradient_finite": True,
        "max_prompt_hidden_deviation": max_prompt_deviation,
        "max_prompt_reference_magnitude": max_prompt_reference_magnitude,
        "relative_max_prompt_hidden_deviation": (
            max_prompt_deviation
            / max(max_prompt_reference_magnitude, np.finfo(np.float32).tiny)
        ),
        "mean_abs_prompt_hidden_deviation": (
            prompt_difference_sum / prompt_difference_elements
            if prompt_difference_elements else 0.0
        ),
        "prompt_hidden_nonzero_difference_fraction": (
            prompt_nonzero_difference_elements / prompt_difference_elements
            if prompt_difference_elements else 0.0
        ),
        "mode_runtime_seconds": mode_runtime,
        "runtime_seconds": sum(mode_runtime.values()),
    }


def centered(scores: dict[str, float]) -> dict[str, float]:
    mean = sum(scores.values()) / len(scores)
    return {mode: scores[mode] - mean for mode in MODES}


def pilot_sample_ids(data: SealedData) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for channel, config in CHANNELS.items():
        population = [
            index
            for index in data.dev
            if (
                data.archived[index]["gold"] == config["gold"]
                and data.archived[index]["pred"] == config["source"]
            )
        ]
        ordered = sorted(
            population, key=lambda index: data.archived[index]["uuid"]
        )
        if len(ordered) < PILOT_PER_CHANNEL:
            raise ValueError(f"insufficient archived pilot pool {channel}")
        result[channel] = ordered[:PILOT_PER_CHANNEL]
    return result


def run_pilot() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    data = SealedData()
    directions = load_directions()
    routers, router_audit = fit_reconstructed_routers(data)
    sample_ids = pilot_sample_ids(data)
    model, tokenizer, device, environment = load_model()
    rows: list[dict[str, Any]] = []
    all_score_diffs: list[float] = []
    all_centered_diffs: list[float] = []
    all_margin_diffs: list[float] = []
    prediction_flips = 0
    runner_up_flips = 0
    prompt_state_deviation = 0.0
    prompt_state_relative_deviation = 0.0
    jacobian_seconds: list[float] = []
    intervention_seconds: list[float] = []
    baseline_seconds: list[float] = []
    gradient_norms: list[float] = []
    contrast_norms: list[float] = []

    pilot_started = time.perf_counter()
    for channel in CHANNELS:
        config = CHANNELS[channel]
        correction = (
            config["rho"] * config["median_norm"] * directions[channel]
        ).astype(np.float32)
        for index in sample_ids[channel]:
            sample = data.sample(index)
            first = score_sample(model, tokenizer, device, sample)
            second = score_sample(model, tokenizer, device, sample)
            baseline_seconds.append(
                0.5 * (first["runtime_seconds"] + second["runtime_seconds"])
            )
            score_diffs = {
                mode: abs(first["scores"][mode] - second["scores"][mode])
                for mode in MODES
            }
            centered_first = centered(first["scores"])
            centered_second = centered(second["scores"])
            centered_diffs = {
                mode: abs(centered_first[mode] - centered_second[mode])
                for mode in MODES
            }
            margin_diff = abs(
                first["top1_top2_margin"] - second["top1_top2_margin"]
            )
            all_score_diffs.extend(score_diffs.values())
            all_centered_diffs.extend(centered_diffs.values())
            all_margin_diffs.append(margin_diff)
            prediction_flip = first["prediction"] != second["prediction"]
            runner_up_flip = first["runner_up"] != second["runner_up"]
            prediction_flips += int(prediction_flip)
            runner_up_flips += int(runner_up_flip)

            jacobian = jacobian_sample(
                model,
                tokenizer,
                device,
                sample,
                config["inj_layer"],
                capture_prompt_states=True,
            )
            jacobian_seconds.append(jacobian["runtime_seconds"])
            prompt_state_deviation = max(
                prompt_state_deviation,
                jacobian["max_prompt_hidden_deviation"],
            )
            prompt_state_relative_deviation = max(
                prompt_state_relative_deviation,
                jacobian["relative_max_prompt_hidden_deviation"],
            )
            for mode in MODES:
                gradient_norms.extend([
                    float(np.linalg.norm(jacobian["g_prompt"][mode])),
                    float(np.linalg.norm(jacobian["g_full"][mode])),
                ])
            contrast_norms.extend([
                float(np.linalg.norm(
                    jacobian["g_prompt"][config["gold"]]
                    - jacobian["g_prompt"][config["source"]]
                )),
                float(np.linalg.norm(
                    jacobian["g_full"][config["gold"]]
                    - jacobian["g_full"][config["source"]]
                )),
            ])

            probability = routers[channel]["probability"][index]
            router_active = (
                first["prediction"] == config["source"]
                and probability >= config["threshold"]
            )
            if router_active:
                intervened = score_sample(
                    model,
                    tokenizer,
                    device,
                    sample,
                    inj_layer=config["inj_layer"],
                    correction=correction,
                )
                intervention_seconds.append(intervened["runtime_seconds"])
                intervention_prediction = intervened["prediction"]
            else:
                intervention_prediction = first["prediction"]

            rows.append({
                "channel": channel,
                "project_index": index,
                "sample_id": sample["uuid"],
                "selection_rule": "ascending UUID within archived dev channel error",
                "archived_prediction": data.archived[index]["pred"],
                "current_prediction_first": first["prediction"],
                "current_prediction_second": second["prediction"],
                "prediction_flip": prediction_flip,
                "runner_up_flip": runner_up_flip,
                "max_abs_score_difference": max(score_diffs.values()),
                "mean_abs_score_difference": float(np.mean(list(score_diffs.values()))),
                "max_abs_centered_score_difference": max(centered_diffs.values()),
                "margin_difference": margin_diff,
                "baseline_top1_top2_margin": first["top1_top2_margin"],
                "router_status": "RECONSTRUCTED_NOT_ARCHIVED",
                "router_probability": probability,
                "router_active": router_active,
                "intervention_prediction": intervention_prediction,
                "prompt_tokens": first["prompt_tokens"],
                "candidate_tokens": first["candidate_tokens"],
                "gradient_shapes": jacobian["gradient_shapes"],
                "gradient_finite": jacobian["gradient_finite"],
                "max_prompt_hidden_deviation": jacobian[
                    "max_prompt_hidden_deviation"
                ],
                "max_prompt_reference_magnitude": jacobian[
                    "max_prompt_reference_magnitude"
                ],
                "relative_max_prompt_hidden_deviation": jacobian[
                    "relative_max_prompt_hidden_deviation"
                ],
                "mean_abs_prompt_hidden_deviation": jacobian[
                    "mean_abs_prompt_hidden_deviation"
                ],
                "prompt_hidden_nonzero_difference_fraction": jacobian[
                    "prompt_hidden_nonzero_difference_fraction"
                ],
                "baseline_runtime_seconds": first["runtime_seconds"],
                "jacobian_runtime_seconds": jacobian["runtime_seconds"],
                "intervention_runtime_seconds": (
                    intervened["runtime_seconds"] if router_active else 0.0
                ),
            })
            del first, second, jacobian
            if router_active:
                del intervened
            gc.collect()
            torch.cuda.empty_cache()

    observed_max_score = max(all_score_diffs, default=0.0)
    observed_max_centered = max(all_centered_diffs, default=0.0)
    observed_max_margin = max(all_margin_diffs, default=0.0)
    score_noise_tolerance = max(1e-6, 10.0 * observed_max_score)
    centered_noise_tolerance = max(1e-6, 10.0 * observed_max_centered)
    margin_noise_tolerance = max(1e-6, 10.0 * observed_max_margin)
    near_boundary_margin = 2.0 * score_noise_tolerance
    prompt_state_tolerance = 1e-6
    prompt_state_shared_samples = sum(
        row["max_prompt_hidden_deviation"] <= prompt_state_tolerance
        for row in rows
    )
    median_gradient_norm = float(np.median(gradient_norms))
    contrast_norm_tolerance = max(
        1e-10,
        10.0 * np.finfo(np.float32).eps * max(median_gradient_norm, 1e-12),
    )
    svd_rank_rule = (
        "tol=max(4,3584)*eps_float64*largest_singular_value"
    )
    offaxis_tolerance = 1e-8
    mean_baseline = float(np.mean(baseline_seconds))
    mean_jacobian = float(np.mean(jacobian_seconds))
    mean_intervention = (
        float(np.mean(intervention_seconds))
        if intervention_seconds else mean_baseline
    )
    archived_channel_count = sum(
        1
        for channel, config in CHANNELS.items()
        for index in data.dev
        if (
            data.archived[index]["gold"] == config["gold"]
            and data.archived[index]["pred"] == config["source"]
        )
    )
    projected = (
        len(data.dev) * mean_baseline
        + archived_channel_count * (mean_jacobian + mean_intervention)
    )
    peak_memory = int(torch.cuda.max_memory_allocated(0))
    pilot_elapsed = time.perf_counter() - pilot_started
    outcome = (
        "PASS"
        if (
            all(row["gradient_finite"] for row in rows)
            and prediction_flips == 0
            and projected <= MAX_GPU_SECONDS
        )
        else (
            "FULL_DEV_GEOMETRY_COST_BLOCKED"
            if projected > MAX_GPU_SECONDS
            else "ENGINEERING_RETRY_REQUIRED"
        )
    )
    result = {
        "schema_version": 1,
        "label": "CURRENT_ENVIRONMENT_GEOMETRY_ONLY",
        "gate_b_outcome": outcome,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": git("rev-parse", "HEAD"),
        "environment": environment,
        "selection": {
            channel: [
                {
                    "project_index": index,
                    "sample_id": data.archived[index]["uuid"],
                }
                for index in indices
            ]
            for channel, indices in sample_ids.items()
        },
        "noise_floor": {
            "observed_max_abs_score_difference": observed_max_score,
            "observed_mean_abs_score_difference": float(
                np.mean(all_score_diffs)
            ),
            "observed_max_abs_centered_score_difference": observed_max_centered,
            "observed_max_margin_difference": observed_max_margin,
            "prediction_flips": prediction_flips,
            "runner_up_flips": runner_up_flips,
            "frozen_score_noise_tolerance": score_noise_tolerance,
            "frozen_centered_noise_tolerance": centered_noise_tolerance,
            "frozen_margin_noise_tolerance": margin_noise_tolerance,
            "frozen_near_boundary_rule": (
                "baseline top1/top2 margin <= "
                "2 * frozen_score_noise_tolerance"
            ),
            "frozen_near_boundary_margin": near_boundary_margin,
        },
        "hook_validation": {
            "module": "model.model.layers[L_inj].mlp",
            "tensor": "MLP forward output before decoder residual addition",
            "positions": "all concatenated prompt+candidate positions",
            "schedule": "active throughout each candidate scoring forward",
            "batch_size": 1,
            "direction_hash": DIRECTION_OID,
            "all_gradients_finite": all(
                row["gradient_finite"] for row in rows
            ),
            "gradient_shapes": sorted({
                tuple(shape)
                for row in rows
                for shape in row["gradient_shapes"].values()
            }),
            "memory_released_between_modes": True,
        },
        "well_posedness": {
            "result": "SAMPLE_CONDITIONAL",
            "prompt_positions_shared_for_all_pilot_samples": (
                prompt_state_shared_samples == len(rows)
            ),
            "frozen_prompt_state_absolute_tolerance": prompt_state_tolerance,
            "prompt_state_shared_sample_count": prompt_state_shared_samples,
            "prompt_state_not_shared_sample_count": (
                len(rows) - prompt_state_shared_samples
            ),
            "max_abs_prompt_hidden_deviation": prompt_state_deviation,
            "max_relative_prompt_hidden_deviation": (
                prompt_state_relative_deviation
            ),
            "candidate_positions_shared": False,
            "candidate_position_status": "CANDIDATE_CONDITIONED",
            "common_prompt_policy_for_gate_c": (
                "compute common-prompt projection only when the four "
                "candidate-forward prompt states agree within the frozen "
                "absolute tolerance; otherwise label "
                "COMMON_PROMPT_STATE_NOT_SHARED"
            ),
        },
        "frozen_geometry_tolerances": {
            "contrast_norm_tolerance": contrast_norm_tolerance,
            "contrast_norm_pilot_min": min(contrast_norms),
            "svd_rank_rule": svd_rank_rule,
            "offaxis_numerical_tolerance": offaxis_tolerance,
        },
        "cost": {
            "pilot_wall_seconds_excluding_model_load": pilot_elapsed,
            "mean_baseline_four_mode_seconds_per_sample": mean_baseline,
            "mean_jacobian_four_mode_seconds_per_sample": mean_jacobian,
            "mean_active_intervention_four_mode_seconds_per_sample": mean_intervention,
            "projected_dev_baseline_samples": len(data.dev),
            "projected_channel_error_samples": archived_channel_count,
            "projected_total_gpu_seconds": projected,
            "projected_total_gpu_hours": projected / 3600.0,
            "peak_gpu_memory_bytes": peak_memory,
            "expected_jsonl_storage_bytes": archived_channel_count * 8000,
            "two_hour_limit_seconds": MAX_GPU_SECONDS,
        },
        "router_reconstruction": router_audit,
        "archive_pilot_comparison": {
            "baseline_prediction_disagreements": [
                {
                    "channel": row["channel"],
                    "project_index": row["project_index"],
                    "sample_id": row["sample_id"],
                    "archived": row["archived_prediction"],
                    "current": row["current_prediction_first"],
                    "baseline_margin": row["baseline_top1_top2_margin"],
                }
                for row in rows
                if row["archived_prediction"]
                != row["current_prediction_first"]
            ],
            "intervention_comparison": (
                "UNAVAILABLE_NO_ARCHIVED_DEV_PER_SAMPLE_INTERVENTION_RECORDS"
            ),
            "routed_status_comparison": (
                "UNAVAILABLE_ORIGINAL_SERIALIZED_ROUTER_AND_ROUTED_IDS_MISSING"
            ),
        },
        "samples": rows,
        "test_access": {
            "project_test_index_opened": False,
            "project_test_rows_materialized": False,
            "project_test_labels_scores_or_aggregates_read": False,
        },
    }
    (OUT / "gate_b_engineering_pilot.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    sample_lines = "\n".join(
        f"| {row['channel']} | {row['project_index']} | `{row['sample_id']}` | "
        f"{row['max_abs_score_difference']:.3g} | "
        f"{str(row['prediction_flip']).lower()} | "
        f"{row['max_prompt_hidden_deviation']:.3g} | "
        f"{row['jacobian_runtime_seconds']:.3f} |"
        for row in rows
    )
    (OUT / "GATE_B_ENGINEERING_PILOT.md").write_text(
        "# Gate B — Engineering pilot\n\n"
        f"**Outcome:** `{outcome}`\n\n"
        "**Scientific label:** `CURRENT_ENVIRONMENT_GEOMETRY_ONLY`\n\n"
        "Four samples per channel were selected by ascending UUID within the "
        "archived dev channel-error pool. No outcome, score margin, or archive "
        "agreement was used for selection.\n\n"
        "## Numerical noise floor\n\n"
        f"- Maximum absolute score difference: `{observed_max_score:.12g}`\n"
        f"- Mean absolute score difference: `{np.mean(all_score_diffs):.12g}`\n"
        f"- Maximum centered-score difference: `{observed_max_centered:.12g}`\n"
        f"- Maximum margin difference: `{observed_max_margin:.12g}`\n"
        f"- Prediction flips: `{prediction_flips}`\n"
        f"- Runner-up flips: `{runner_up_flips}`\n"
        f"- Frozen score tolerance: `{score_noise_tolerance:.12g}`\n"
        f"- Frozen near-boundary rule: top1/top2 margin <= "
        f"`{near_boundary_margin:.12g}`\n\n"
        "## Hook and common-state validation\n\n"
        "The hook intercepts `model.model.layers[L_inj].mlp` output, detaches "
        "it, marks the detached tensor as a leaf, and backpropagates only through "
        "layers above the injection point. Gradients were finite for all four "
        "candidate-specific graphs. Prompt positions were compared across those "
        "graphs; candidate positions remain candidate-conditioned.\n\n"
        f"- Max prompt-position hidden-state deviation: "
        f"`{prompt_state_deviation:.12g}`\n"
        f"- Max relative prompt-position deviation: "
        f"`{prompt_state_relative_deviation:.12g}`\n"
        f"- Shared within frozen `1e-6` tolerance: "
        f"`{prompt_state_shared_samples} / {len(rows)}` pilot samples\n"
        f"- Peak allocated GPU memory: `{peak_memory}` bytes\n"
        f"- Projected full-dev time: `{projected / 3600:.4f}` GPU-hours\n"
        f"- Two-hour gate: `{'PASS' if projected <= MAX_GPU_SECONDS else 'BLOCKED'}`\n\n"
        "## Sample diagnostics\n\n"
        "| channel | index | UUID | max score Δ | pred flip | prompt hidden max Δ | Jacobian s |\n"
        "|---|---:|---|---:|---|---:|---:|\n"
        f"{sample_lines}\n\n"
        "Original Router weights and archived routed IDs are absent. Router "
        "statuses shown here come from a deterministic reconstruction of the "
        "frozen historical recipe and are not represented as the original "
        "serialized Router.\n"
    )
    print(json.dumps({
        "gate_b_outcome": outcome,
        "pilot_samples": len(rows),
        "prediction_flips": prediction_flips,
        "max_score_noise": observed_max_score,
        "max_prompt_hidden_deviation": prompt_state_deviation,
        "max_relative_prompt_hidden_deviation": prompt_state_relative_deviation,
        "projected_gpu_hours": projected / 3600.0,
        "peak_gpu_memory_bytes": peak_memory,
    }, indent=2))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def geometry_metrics(
    gradients: dict[str, np.ndarray],
    direction: np.ndarray,
    gold: str,
    source: str,
    contrast_tolerance: float,
    offaxis_tolerance: float,
) -> tuple[dict[str, Any], np.ndarray | None]:
    matrix = np.stack([gradients[mode] for mode in MODES]).astype(np.float64)
    mean_gradient = matrix.mean(axis=0)
    centered_matrix = matrix - mean_gradient
    _, singular_values, vh = np.linalg.svd(
        centered_matrix, full_matrices=False
    )
    largest = float(singular_values[0]) if len(singular_values) else 0.0
    rank_tolerance = (
        max(centered_matrix.shape)
        * np.finfo(np.float64).eps
        * largest
    )
    rank = int(np.sum(singular_values > rank_tolerance))
    contrast = (
        gradients[gold].astype(np.float64)
        - gradients[source].astype(np.float64)
    )
    contrast_norm = float(np.linalg.norm(contrast))
    result: dict[str, Any] = {
        "observed_rank": rank,
        "singular_values": [float(value) for value in singular_values],
        "svd_rank_tolerance": rank_tolerance,
        "contrast_norm": contrast_norm,
        "gradient_norms": {
            mode: float(np.linalg.norm(gradients[mode])) for mode in MODES
        },
        "status": "VALID",
        "offaxis_clipped": False,
    }
    if contrast_norm <= contrast_tolerance:
        result.update({
            "status": "CONTRAST_AXIS_UNDEFINED",
            "a_contrast": None,
            "a_dec": None,
            "a_offaxis": None,
            "a_orth": None,
        })
        return result, None
    if rank == 0:
        result.update({
            "status": "DECISION_SUBSPACE_DEGENERATE",
            "a_contrast": None,
            "a_dec": None,
            "a_offaxis": None,
            "a_orth": None,
        })
        return result, None
    axis = contrast / contrast_norm
    unit_direction = direction.astype(np.float64)
    a_contrast = float(np.dot(unit_direction, axis) ** 2)
    basis = vh[:rank]
    a_dec = float(np.sum((basis @ unit_direction) ** 2))
    a_offaxis = a_dec - a_contrast
    if a_offaxis < 0 and a_offaxis >= -offaxis_tolerance:
        a_offaxis = 0.0
        result["offaxis_clipped"] = True
    elif a_offaxis < -offaxis_tolerance:
        result["status"] = "GEOMETRY_INCONSISTENCY_NEGATIVE_OFFAXIS"
        result.update({
            "a_contrast": a_contrast,
            "a_dec": a_dec,
            "a_offaxis": a_offaxis,
            "a_orth": 1.0 - a_dec,
        })
        return result, None
    result.update({
        "a_contrast": a_contrast,
        "a_dec": a_dec,
        "a_offaxis": a_offaxis,
        "a_orth": 1.0 - a_dec,
        "identity_contrast_plus_offaxis_minus_dec": (
            a_contrast + a_offaxis - a_dec
        ),
        "identity_dec_plus_orth_minus_one": a_dec + (1.0 - a_dec) - 1.0,
    })
    return result, axis


def distribution(values: Iterable[float | int | None]) -> dict[str, Any]:
    array = np.asarray(
        [value for value in values if value is not None and np.isfinite(value)],
        dtype=np.float64,
    )
    if not len(array):
        return {
            "n": 0, "mean": None, "std": None, "min": None, "p25": None,
            "median": None, "p75": None, "max": None,
        }
    return {
        "n": int(len(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "min": float(np.min(array)),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "p75": float(np.quantile(array, 0.75)),
        "max": float(np.max(array)),
    }


def metric_pair(predicted: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    mask = np.isfinite(predicted) & np.isfinite(actual)
    x = predicted[mask].astype(np.float64)
    y = actual[mask].astype(np.float64)
    if not len(x):
        return {
            "n": 0, "pearson": None, "spearman": None, "slope": None,
            "intercept": None, "rmse": None, "mae": None,
        }
    if len(x) >= 2 and np.std(x) > 0 and np.std(y) > 0:
        pearson = float(stats.pearsonr(x, y).statistic)
        spearman = float(stats.spearmanr(x, y).statistic)
        slope, intercept = np.polyfit(x, y, 1)
        slope = float(slope)
        intercept = float(intercept)
    else:
        pearson = spearman = slope = intercept = None
    residual = y - x
    return {
        "n": int(len(x)),
        "pearson": pearson,
        "spearman": spearman,
        "slope": slope,
        "intercept": intercept,
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "mae": float(np.mean(np.abs(residual))),
    }


def safe_spearman(x: Iterable[float], y: Iterable[float]) -> float | None:
    xa = np.asarray(list(x), dtype=np.float64)
    ya = np.asarray(list(y), dtype=np.float64)
    mask = np.isfinite(xa) & np.isfinite(ya)
    xa, ya = xa[mask], ya[mask]
    if len(xa) < 2 or np.std(xa) == 0 or np.std(ya) == 0:
        return None
    return float(stats.spearmanr(xa, ya).statistic)


def resultant_null(
    n: int, trials: int = 10000, seed: int = NULL_SEED
) -> np.ndarray:
    """Exact rotational Markov simulation without allocating n*d vectors."""
    if n <= 0:
        return np.asarray([], dtype=np.float64)
    rng = np.random.default_rng(seed + n)
    resultant = np.ones(trials, dtype=np.float64)
    beta_shape = (DIMENSION - 1) / 2.0
    for _ in range(1, n):
        cosine = 2.0 * rng.beta(beta_shape, beta_shape, size=trials) - 1.0
        resultant = np.sqrt(
            np.maximum(
                0.0,
                resultant * resultant + 1.0 + 2.0 * resultant * cosine,
            )
        )
    return resultant / n


def summarize_geometry(
    records: list[dict[str, Any]],
    axes: dict[tuple[str, str], list[np.ndarray]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    summaries: dict[str, Any] = {}
    concentration: dict[str, Any] = {
        "schema_version": 1,
        "dimension": DIMENSION,
        "seed": NULL_SEED,
        "null_trials": 10000,
        "null_method": (
            "exact rotational Markov simulation of sums of independent uniform "
            "unit axes; cosine increments sampled from transformed "
            "Beta((d-1)/2,(d-1)/2)"
        ),
        "groups": {},
    }
    for channel in CHANNELS:
        for geometry_type in ("common_prompt", "full_effect"):
            group = [
                row
                for row in records
                if row["channel"] == channel
                and row["geometry_type"] == geometry_type
            ]
            key = f"{channel}|{geometry_type}"
            summaries[key] = {
                "channel": channel,
                "geometry_type": geometry_type,
                "n_total": len(group),
                "n_valid": sum(row["geometry_status"] == "VALID" for row in group),
                "n_contrast_undefined": sum(
                    row["geometry_status"] == "CONTRAST_AXIS_UNDEFINED"
                    for row in group
                ),
                "n_rank_zero": sum(
                    row["geometry_status"] == "DECISION_SUBSPACE_DEGENERATE"
                    for row in group
                ),
                "n_geometry_inconsistency": sum(
                    row["geometry_status"].startswith("GEOMETRY_INCONSISTENCY")
                    for row in group
                ),
                "n_common_prompt_state_not_shared": sum(
                    row["geometry_status"] == "COMMON_PROMPT_STATE_NOT_SHARED"
                    for row in group
                ),
                "a_contrast": distribution(row["a_contrast"] for row in group),
                "a_dec": distribution(row["a_dec"] for row in group),
                "a_offaxis": distribution(row["a_offaxis"] for row in group),
                "a_orth": distribution(row["a_orth"] for row in group),
                "observed_rank": distribution(
                    row["observed_rank"] for row in group
                ),
                "contrast_norm": distribution(
                    row["contrast_norm"] for row in group
                ),
                "singular_value_1": distribution(
                    row["singular_values"][0] for row in group
                ),
                "singular_value_2": distribution(
                    row["singular_values"][1] for row in group
                ),
                "singular_value_3": distribution(
                    row["singular_values"][2] for row in group
                ),
                "singular_value_4": distribution(
                    row["singular_values"][3] for row in group
                ),
                "gradient_norm_by_mode": {
                    mode: distribution(
                        row["gradient_norms"][mode] for row in group
                    )
                    for mode in MODES
                },
                "bootstrap_ci": "NOT_USED_DESCRIPTIVE_DISTRIBUTIONS_REPORTED",
            }
            group_axes = axes.get((channel, geometry_type), [])
            if group_axes:
                matrix = np.stack(group_axes)
                resultant = float(np.linalg.norm(matrix.mean(axis=0)))
                gram = matrix @ matrix.T
                tri = gram[np.triu_indices(len(matrix), k=1)]
                null = resultant_null(len(matrix))
                histogram_counts, histogram_edges = np.histogram(
                    tri, bins=np.linspace(-1.0, 1.0, 41)
                )
                concentration["groups"][key] = {
                    "n_axes": len(matrix),
                    "resultant_length": resultant,
                    "pairwise_cosine": {
                        "mean": float(np.mean(tri)) if len(tri) else None,
                        "median": float(np.median(tri)) if len(tri) else None,
                        "p05": float(np.quantile(tri, 0.05)) if len(tri) else None,
                        "p25": float(np.quantile(tri, 0.25)) if len(tri) else None,
                        "p75": float(np.quantile(tri, 0.75)) if len(tri) else None,
                        "p95": float(np.quantile(tri, 0.95)) if len(tri) else None,
                        "histogram_edges": histogram_edges.tolist(),
                        "histogram_counts": histogram_counts.tolist(),
                    },
                    "chance_null": {
                        "mean": float(np.mean(null)),
                        "p95": float(np.quantile(null, 0.95)),
                        "p99": float(np.quantile(null, 0.99)),
                        "empirical_one_sided_p": float(
                            (1 + np.sum(null >= resultant)) / (len(null) + 1)
                        ),
                    },
                }
            else:
                concentration["groups"][key] = {
                    "n_axes": 0,
                    "status": "NO_VALID_CONTRAST_AXES",
                }
    return summaries, concentration


def build_rank_null(records: list[dict[str, Any]]) -> dict[str, Any]:
    ranks = sorted({
        int(row["observed_rank"])
        for row in records
        if row["observed_rank"] > 0
    })
    output = {
        "schema_version": 1,
        "dimension": DIMENSION,
        "monte_carlo_seed": NULL_SEED,
        "monte_carlo_draws_per_rank": 100000,
        "ranks": {},
    }
    for rank in ranks:
        a = rank / 2.0
        b = (DIMENSION - rank) / 2.0
        rng = np.random.default_rng(NULL_SEED + rank)
        draws = rng.beta(a, b, size=100000)
        output["ranks"][str(rank)] = {
            "beta_parameters": [a, b],
            "theoretical_mean": rank / DIMENSION,
            "theoretical_p50": float(stats.beta.ppf(0.50, a, b)),
            "theoretical_p95": float(stats.beta.ppf(0.95, a, b)),
            "theoretical_p99": float(stats.beta.ppf(0.99, a, b)),
            "monte_carlo_mean": float(np.mean(draws)),
            "monte_carlo_p50": float(np.quantile(draws, 0.50)),
            "monte_carlo_p95": float(np.quantile(draws, 0.95)),
            "monte_carlo_p99": float(np.quantile(draws, 0.99)),
        }
    return output


def build_linearity(samples: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": 1,
        "label": "CURRENT_ENVIRONMENT_GEOMETRY_ONLY",
        "dose_grid": "NOT_RUN_SINGLE_HISTORICAL_OPERATING_DOSE",
        "channels": {},
    }
    for channel in CHANNELS:
        channel_samples = [
            sample for sample in samples if sample["channel"] == channel
        ]
        channel_result: dict[str, Any] = {}
        for population_name, subset in (
            ("all_current_channel_errors", channel_samples),
            (
                "reconstructed_router_active",
                [sample for sample in channel_samples if sample["router_active"]],
            ),
        ):
            raw_predicted: list[float] = []
            raw_actual: list[float] = []
            centered_predicted: list[float] = []
            centered_actual: list[float] = []
            margin_predicted: list[float] = []
            margin_actual: list[float] = []
            residual_context = {
                "baseline_margin": [],
                "predicted_effect_magnitude": [],
                "candidate_length": [],
                "injected_position_count": [],
                "router_probability": [],
                "raw_residual": [],
            }
            for sample in subset:
                pred = sample["linear_predicted_score_change"]
                actual = sample["actual_score_change"]
                pred_mean = float(np.mean([pred[mode] for mode in MODES]))
                actual_mean = float(np.mean([actual[mode] for mode in MODES]))
                for mode in MODES:
                    raw_predicted.append(pred[mode])
                    raw_actual.append(actual[mode])
                    centered_predicted.append(pred[mode] - pred_mean)
                    centered_actual.append(actual[mode] - actual_mean)
                    residual_context["baseline_margin"].append(
                        sample["baseline_source_gold_margin"]
                    )
                    residual_context["predicted_effect_magnitude"].append(
                        abs(pred[mode])
                    )
                    residual_context["candidate_length"].append(
                        sample["candidate_tokens"][mode]
                    )
                    residual_context["injected_position_count"].append(
                        sample["actual_position_counts"][mode]
                    )
                    residual_context["router_probability"].append(
                        sample["router_probability"]
                    )
                    residual_context["raw_residual"].append(
                        actual[mode] - pred[mode]
                    )
                gold = sample["gold"]
                source = sample["source"]
                margin_predicted.append(pred[gold] - pred[source])
                margin_actual.append(actual[gold] - actual[source])
            raw_x = np.asarray(raw_predicted, dtype=np.float64)
            raw_y = np.asarray(raw_actual, dtype=np.float64)
            centered_x = np.asarray(centered_predicted, dtype=np.float64)
            centered_y = np.asarray(centered_actual, dtype=np.float64)
            margin_x = np.asarray(margin_predicted, dtype=np.float64)
            margin_y = np.asarray(margin_actual, dtype=np.float64)
            channel_result[population_name] = {
                "n_samples": len(subset),
                "raw_score_change": metric_pair(raw_x, raw_y),
                "centered_score_change": metric_pair(centered_x, centered_y),
                "decision_margin_change_primary": metric_pair(
                    margin_x, margin_y
                ),
                "raw_residual_spearman": {
                    key: safe_spearman(
                        residual_context["raw_residual"],
                        residual_context[key],
                    )
                    for key in (
                        "baseline_margin",
                        "predicted_effect_magnitude",
                        "candidate_length",
                        "injected_position_count",
                        "router_probability",
                    )
                },
            }
        result["channels"][channel] = channel_result
    return result


def run_full() -> dict[str, Any]:
    pilot_path = OUT / "gate_b_engineering_pilot.json"
    if not pilot_path.is_file():
        raise RuntimeError("Gate B pilot report missing")
    pilot = json.loads(pilot_path.read_text())
    if pilot["gate_b_outcome"] != "PASS":
        raise RuntimeError(f"Gate B did not pass: {pilot['gate_b_outcome']}")
    if pilot["cost"]["projected_total_gpu_seconds"] > MAX_GPU_SECONDS:
        raise RuntimeError("Gate B cost projection exceeds two hours")
    tolerances = pilot["frozen_geometry_tolerances"]
    score_noise = pilot["noise_floor"]["frozen_score_noise_tolerance"]
    near_boundary = pilot["noise_floor"]["frozen_near_boundary_margin"]
    prompt_state_tolerance = pilot["well_posedness"][
        "frozen_prompt_state_absolute_tolerance"
    ]

    data = SealedData()
    directions = load_directions()
    routers, router_audit = fit_reconstructed_routers(data)
    model, tokenizer, device, environment = load_model()
    full_started = time.perf_counter()

    current_baseline: dict[int, dict[str, Any]] = {}
    for ordinal, index in enumerate(data.dev, start=1):
        sample = data.sample(index)
        current_baseline[index] = score_sample(
            model, tokenizer, device, sample
        )
        if ordinal % 25 == 0:
            print(f"baseline {ordinal}/{len(data.dev)}", flush=True)
            gc.collect()
            torch.cuda.empty_cache()

    prediction_vector = [
        current_baseline[index]["prediction"] for index in data.dev
    ]
    prediction_sha256 = sha256_bytes(canonical_bytes(prediction_vector))
    prediction_records_sha256 = sha256_bytes(canonical_bytes([
        {
            "project_index": index,
            "sample_id": data.archived[index]["uuid"],
            "prediction": current_baseline[index]["prediction"],
        }
        for index in data.dev
    ]))
    current_populations: dict[str, list[int]] = {}
    for channel, config in CHANNELS.items():
        current_populations[channel] = [
            index
            for index in data.dev
            if (
                data.archived[index]["gold"] == config["gold"]
                and current_baseline[index]["prediction"] == config["source"]
            )
        ]

    sample_results: list[dict[str, Any]] = []
    geometry_records: list[dict[str, Any]] = []
    axes: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    processed = 0
    total_geometry_samples = sum(map(len, current_populations.values()))
    for channel, config in CHANNELS.items():
        direction = directions[channel]
        correction = (
            config["rho"] * config["median_norm"] * direction
        ).astype(np.float32)
        for index in current_populations[channel]:
            sample = data.sample(index)
            baseline = current_baseline[index]
            jacobian = jacobian_sample(
                model,
                tokenizer,
                device,
                sample,
                config["inj_layer"],
                capture_prompt_states=True,
            )
            jacobian_score_deviation = max(
                abs(jacobian["scores"][mode] - baseline["scores"][mode])
                for mode in MODES
            )
            probability = routers[channel]["probability"][index]
            router_active = (
                baseline["prediction"] == config["source"]
                and probability >= config["threshold"]
            )
            if router_active:
                intervened = score_sample(
                    model,
                    tokenizer,
                    device,
                    sample,
                    inj_layer=config["inj_layer"],
                    correction=correction,
                )
            else:
                intervened = {
                    **baseline,
                    "scores": dict(baseline["scores"]),
                    "runtime_seconds": 0.0,
                }
            actual_change = {
                mode: (
                    intervened["scores"][mode] - baseline["scores"][mode]
                )
                for mode in MODES
            }
            per_position_scale = (
                config["rho"] * config["median_norm"]
                if router_active else 0.0
            )
            linear_change = {
                mode: float(
                    per_position_scale
                    * np.dot(
                        jacobian["g_full"][mode].astype(np.float64),
                        direction.astype(np.float64),
                    )
                )
                for mode in MODES
            }
            source_gold_margin = (
                baseline["scores"][config["source"]]
                - baseline["scores"][config["gold"]]
            )
            sample_result = {
                "channel": channel,
                "project_index": index,
                "sample_id": sample["uuid"],
                "population_source": "CURRENT_ENVIRONMENT",
                "gold": config["gold"],
                "source": config["source"],
                "baseline_scores": baseline["scores"],
                "baseline_prediction": baseline["prediction"],
                "baseline_non_source_runner_up": max(
                    [mode for mode in MODES if mode != config["source"]],
                    key=lambda mode: baseline["scores"][mode],
                ),
                "baseline_source_gold_margin": source_gold_margin,
                "baseline_top1_top2_margin": baseline["top1_top2_margin"],
                "near_boundary": (
                    baseline["top1_top2_margin"] <= near_boundary
                ),
                "candidate_tokens": jacobian["candidate_tokens"],
                "prompt_tokens": baseline["prompt_tokens"],
                "actual_position_counts": jacobian["actual_positions"],
                "router_status": "RECONSTRUCTED_NOT_ARCHIVED",
                "router_probability": probability,
                "router_threshold": config["threshold"],
                "router_active": router_active,
                "rho": config["rho"],
                "median_norm": config["median_norm"],
                "actual_per_position_scaling": per_position_scale,
                "intervention_scores": intervened["scores"],
                "intervention_prediction": intervened["prediction"],
                "actual_score_change": actual_change,
                "linear_predicted_score_change": linear_change,
                "jacobian_score_max_deviation_from_baseline": (
                    jacobian_score_deviation
                ),
                "max_prompt_hidden_deviation": jacobian[
                    "max_prompt_hidden_deviation"
                ],
                "relative_max_prompt_hidden_deviation": jacobian[
                    "relative_max_prompt_hidden_deviation"
                ],
                "prompt_state_shared_within_frozen_tolerance": (
                    jacobian["max_prompt_hidden_deviation"]
                    <= prompt_state_tolerance
                ),
                "jacobian_runtime_seconds": jacobian["runtime_seconds"],
                "intervention_runtime_seconds": intervened["runtime_seconds"],
            }
            sample_results.append(sample_result)

            for geometry_type, gradients in (
                ("common_prompt", jacobian["g_prompt"]),
                ("full_effect", jacobian["g_full"]),
            ):
                metrics, axis = geometry_metrics(
                    gradients,
                    direction,
                    config["gold"],
                    config["source"],
                    tolerances["contrast_norm_tolerance"],
                    tolerances["offaxis_numerical_tolerance"],
                )
                if (
                    geometry_type == "common_prompt"
                    and jacobian["max_prompt_hidden_deviation"]
                    > prompt_state_tolerance
                ):
                    metrics["status"] = "COMMON_PROMPT_STATE_NOT_SHARED"
                    metrics["a_contrast"] = None
                    metrics["a_dec"] = None
                    metrics["a_offaxis"] = None
                    metrics["a_orth"] = None
                    axis = None
                record = {
                    "label": "CURRENT_ENVIRONMENT_GEOMETRY_ONLY",
                    "channel": channel,
                    "project_index": index,
                    "sample_id": sample["uuid"],
                    "population_source": "CURRENT_ENVIRONMENT",
                    "geometry_type": geometry_type,
                    "geometry_label": (
                        "COMMON_PROMPT_GEOMETRY"
                        if geometry_type == "common_prompt"
                        else "CANDIDATE_CONDITIONED_FULL_EFFECT_GEOMETRY"
                    ),
                    "gold": config["gold"],
                    "source": config["source"],
                    "baseline_scores": baseline["scores"],
                    "baseline_source_gold_margin": source_gold_margin,
                    "baseline_non_source_runner_up": sample_result[
                        "baseline_non_source_runner_up"
                    ],
                    "candidate_tokens": jacobian["candidate_tokens"],
                    "router_status": "RECONSTRUCTED_NOT_ARCHIVED",
                    "router_probability": probability,
                    "router_active": router_active,
                    "injected_position_counts": (
                        {mode: baseline["prompt_tokens"] for mode in MODES}
                        if geometry_type == "common_prompt"
                        else jacobian["actual_positions"]
                    ),
                    "actual_per_position_scaling": per_position_scale,
                    "observed_rank": metrics["observed_rank"],
                    "singular_values": metrics["singular_values"],
                    "svd_rank_tolerance": metrics["svd_rank_tolerance"],
                    "contrast_norm": metrics["contrast_norm"],
                    "a_contrast": metrics["a_contrast"],
                    "a_dec": metrics["a_dec"],
                    "a_offaxis": metrics["a_offaxis"],
                    "a_orth": metrics["a_orth"],
                    "gradient_norms": metrics["gradient_norms"],
                    "geometry_status": metrics["status"],
                    "offaxis_clipped": metrics["offaxis_clipped"],
                    "max_prompt_hidden_deviation": jacobian[
                        "max_prompt_hidden_deviation"
                    ],
                    "frozen_prompt_state_absolute_tolerance": (
                        prompt_state_tolerance
                    ),
                }
                geometry_records.append(record)
                if axis is not None:
                    axes[(channel, geometry_type)].append(axis)
            processed += 1
            print(
                f"geometry {processed}/{total_geometry_samples} "
                f"{channel} index={index}",
                flush=True,
            )
            del jacobian, intervened
            gc.collect()
            torch.cuda.empty_cache()

    summary, concentration = summarize_geometry(geometry_records, axes)
    rank_null = build_rank_null(geometry_records)
    concentration["random_orientation_a_dec_null"] = rank_null
    linearity = build_linearity(sample_results)

    archive_disagreements: list[dict[str, Any]] = []
    rounded_score_differences: list[float] = []
    for index in data.dev:
        archived = data.archived[index]
        current = current_baseline[index]
        rounded_score_diff = max(
            abs(current["scores"][mode] - archived["avg_logp"][mode])
            for mode in MODES
        )
        rounded_score_differences.append(rounded_score_diff)
        if archived["pred"] != current["prediction"]:
            archive_disagreements.append({
                "project_index": index,
                "sample_id": archived["uuid"],
                "archived_prediction": archived["pred"],
                "current_prediction": current["prediction"],
                "current_top1_top2_margin": current["top1_top2_margin"],
                "archived_rounded_score_max_abs_difference": rounded_score_diff,
                "frozen_noise_tolerance": score_noise,
                "near_boundary": current["top1_top2_margin"] <= near_boundary,
                "router_status_differs": None,
                "diagnosis": (
                    "current Transformers/PyTorch execution differs from "
                    "historical archived rounded-score environment"
                ),
            })

    max_jacobian_score_deviation = max(
        result["jacobian_score_max_deviation_from_baseline"]
        for result in sample_results
    ) if sample_results else 0.0
    geometry_inconsistencies = [
        row for row in geometry_records
        if row["geometry_status"].startswith("GEOMETRY_INCONSISTENCY")
    ]
    replay = {
        "schema_version": 1,
        "gate_c_outcome": "CURRENT_ENVIRONMENT_GEOMETRY_ONLY",
        "reason": (
            "Gate A lacked original Router weights, archived routed-dev order, "
            "and dev per-sample post-intervention records. Current model execution "
            "and Jacobians are valid; no historical continuity claim is made."
        ),
        "population": {
            "source": "CURRENT_ENVIRONMENT",
            "dev_count": len(data.dev),
            "prediction_vector_sha256": prediction_sha256,
            "prediction_records_sha256": prediction_records_sha256,
            "channel_counts": {
                channel: len(indices)
                for channel, indices in current_populations.items()
            },
            "channel_order": {
                channel: [
                    {
                        "project_index": index,
                        "sample_id": data.archived[index]["uuid"],
                    }
                    for index in indices
                ]
                for channel, indices in current_populations.items()
            },
        },
        "baseline_archive_comparison": {
            "allowed_dev_rows": len(data.dev),
            "prediction_agreement_count": (
                len(data.dev) - len(archive_disagreements)
            ),
            "prediction_disagreement_count": len(archive_disagreements),
            "prediction_agreement_rate": (
                (len(data.dev) - len(archive_disagreements)) / len(data.dev)
            ),
            "archived_scores_are_rounded_to_4_decimals": True,
            "max_abs_current_vs_archived_rounded_score_difference": max(
                rounded_score_differences
            ),
            "mean_abs_current_vs_archived_rounded_score_difference": float(
                np.mean(rounded_score_differences)
            ),
            "disagreements": archive_disagreements,
        },
        "routed_set_agreement": (
            "NOT_COMPUTABLE_ORIGINAL_ROUTER_AND_ROUTED_ID_ORDER_MISSING"
        ),
        "intervention_agreement": (
            "NOT_COMPUTABLE_NO_ARCHIVED_DEV_PER_SAMPLE_INTERVENTION_RECORDS"
        ),
        "execution_validity": {
            "max_abs_jacobian_forward_score_deviation_from_baseline": (
                max_jacobian_score_deviation
            ),
            "frozen_gate_b_score_noise_tolerance": score_noise,
            "geometry_inconsistency_count": len(geometry_inconsistencies),
            "all_gradients_finite": True,
        },
        "router_reconstruction": router_audit,
        "environment": environment,
        "runtime": {
            "full_wall_seconds_excluding_model_load": (
                time.perf_counter() - full_started
            ),
            "model_load_seconds": environment["model_load_seconds"],
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(0)),
        },
        "test_access": {
            "project_test_index_opened": False,
            "project_test_rows_materialized": False,
            "project_test_labels_scores_or_aggregates_read": False,
        },
    }

    with (OUT / "jacobian_geometry.jsonl").open("w") as handle:
        for row in geometry_records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (OUT / "jacobian_channel_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    summary_csv_fields = [
        "channel", "geometry_type", "n_total", "n_valid",
        "n_contrast_undefined", "n_rank_zero", "n_geometry_inconsistency",
        "n_common_prompt_state_not_shared", "a_contrast_mean", "a_dec_mean",
        "a_offaxis_mean", "a_orth_mean",
        "rank_mean", "contrast_norm_median",
    ]
    with (OUT / "jacobian_channel_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=summary_csv_fields, lineterminator="\n"
        )
        writer.writeheader()
        for value in summary.values():
            writer.writerow({
                "channel": value["channel"],
                "geometry_type": value["geometry_type"],
                "n_total": value["n_total"],
                "n_valid": value["n_valid"],
                "n_contrast_undefined": value["n_contrast_undefined"],
                "n_rank_zero": value["n_rank_zero"],
                "n_geometry_inconsistency": value["n_geometry_inconsistency"],
                "n_common_prompt_state_not_shared": value[
                    "n_common_prompt_state_not_shared"
                ],
                "a_contrast_mean": value["a_contrast"]["mean"],
                "a_dec_mean": value["a_dec"]["mean"],
                "a_offaxis_mean": value["a_offaxis"]["mean"],
                "a_orth_mean": value["a_orth"]["mean"],
                "rank_mean": value["observed_rank"]["mean"],
                "contrast_norm_median": value["contrast_norm"]["median"],
            })
    (OUT / "concentration_null.json").write_text(
        json.dumps(concentration, indent=2) + "\n"
    )
    (OUT / "linearity.json").write_text(
        json.dumps(linearity, indent=2) + "\n"
    )
    (OUT / "gate_c_replay_report.json").write_text(
        json.dumps(replay, indent=2) + "\n"
    )

    summary_lines = "\n".join(
        f"| {value['channel']} | {value['geometry_type']} | "
        f"{value['n_total']} | {value['n_valid']} | "
        f"{value['observed_rank']['median']} | "
        f"{value['a_contrast']['median']} | {value['a_dec']['median']} | "
        f"{value['a_offaxis']['median']} | {value['a_orth']['median']} |"
        for value in summary.values()
    )
    (OUT / "GATE_C_REPLAY_REPORT.md").write_text(
        "# Gate C — Dev-only reproduction\n\n"
        "**Outcome:** `CURRENT_ENVIRONMENT_GEOMETRY_ONLY`\n\n"
        f"The current un-intervened dev prediction vector is pinned by SHA256 "
        f"`{prediction_sha256}`. Current channel-error counts are "
        f"`{json.dumps(replay['population']['channel_counts'], sort_keys=True)}`.\n\n"
        f"Archived/current baseline predictions agree on "
        f"{replay['baseline_archive_comparison']['prediction_agreement_count']} / "
        f"{len(data.dev)} allowed dev rows; all "
        f"{len(archive_disagreements)} disagreements are listed in the JSON report. "
        "The archived scores are rounded to four decimals, and original Router/"
        "routed-order plus dev intervention records are missing, so archived replay "
        "cannot be asserted or fully tested.\n\n"
        "No project test index, row, label, score, or aggregate was opened.\n"
    )
    (OUT / "JACOBIAN_GEOMETRY_SUMMARY.md").write_text(
        "# Sample-conditioned sequence-score Jacobian geometry\n\n"
        "**Label:** `CURRENT_ENVIRONMENT_GEOMETRY_ONLY`\n\n"
        "| channel | geometry | n total | n valid | median rank | median a_contrast | median a_dec | median a_offaxis | median a_orth |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|---:|\n"
        f"{summary_lines}\n\n"
        "Common-prompt geometry sums gradients only over shared prompt positions. "
        "Common-prompt projection is omitted sample-by-sample when the four "
        "candidate-forward prompt states exceed the frozen Gate B sharing "
        "tolerance. Full-effect geometry includes all historically injected "
        "prompt and candidate positions and is labeled "
        "`CANDIDATE_CONDITIONED_FULL_EFFECT_GEOMETRY`. Observed rank is determined "
        "from the centered four-gradient matrix and is never forced to three. "
        "Undefined contrasts, rank-zero cases, and numerical clipping are counted "
        "explicitly in the JSON/CSV summaries.\n"
    )
    concentration_lines = "\n".join(
        f"| {key} | {value.get('n_axes', 0)} | "
        f"{value.get('resultant_length')} | "
        f"{value.get('chance_null', {}).get('mean')} | "
        f"{value.get('chance_null', {}).get('p95')} | "
        f"{value.get('chance_null', {}).get('p99')} | "
        f"{value.get('chance_null', {}).get('empirical_one_sided_p')} |"
        for key, value in concentration["groups"].items()
    )
    rank_lines = "\n".join(
        f"| {rank} | {value['theoretical_mean']:.9g} | "
        f"{value['theoretical_p50']:.9g} | {value['theoretical_p95']:.9g} | "
        f"{value['theoretical_p99']:.9g} | {value['monte_carlo_mean']:.9g} |"
        for rank, value in rank_null["ranks"].items()
    )
    (OUT / "CONCENTRATION_NULL_REPORT.md").write_text(
        "# Contrast-axis concentration and orientation nulls\n\n"
        "| channel/geometry | n axes | resultant | null mean | null p95 | null p99 | empirical p |\n"
        "|---|---:|---:|---:|---:|---:|---:|\n"
        f"{concentration_lines}\n\n"
        "The 10,000-draw resultant null uses an exact rotational Markov simulation "
        "in dimension 3584. Pairwise-cosine summaries and histograms are in the "
        "machine-readable report.\n\n"
        "## Random-orientation null for a_dec\n\n"
        "| observed rank | theoretical mean | p50 | p95 | p99 | MC mean |\n"
        "|---:|---:|---:|---:|---:|---:|\n"
        f"{rank_lines}\n\n"
        "Each rank uses `Beta(r/2, (3584-r)/2)` and a deterministic 100,000-draw "
        "Monte Carlo implementation check. No fixed-rank-3 null is imposed.\n"
    )
    linearity_lines: list[str] = []
    for channel, channel_result in linearity["channels"].items():
        for population_name, value in channel_result.items():
            for object_name in (
                "raw_score_change",
                "centered_score_change",
                "decision_margin_change_primary",
            ):
                metric = value[object_name]
                linearity_lines.append(
                    f"| {channel} | {population_name} | {object_name} | "
                    f"{metric['n']} | {metric['pearson']} | "
                    f"{metric['spearman']} | {metric['slope']} | "
                    f"{metric['intercept']} | {metric['rmse']} | "
                    f"{metric['mae']} |"
                )
    (OUT / "LINEARITY_REPORT.md").write_text(
        "# Historical-dose first-order linearity\n\n"
        "The prediction is computed once as "
        "`sum_p alpha_i,p <J_i,m,p, d_hat>` using the actual all-position "
        "historical perturbation. The number of positions is not multiplied a "
        "second time. Inactive reconstructed Router rows have alpha=0.\n\n"
        "| channel | population | object | n | Pearson | Spearman | slope | intercept | RMSE | MAE |\n"
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|\n"
        + "\n".join(linearity_lines)
        + "\n\nThe decision-margin change is the primary decision-relevant object. "
        "Residual correlations are reported in `linearity.json`. Only the single "
        "historical operating dose was used; no dose grid or tuning was performed.\n"
    )
    print(json.dumps({
        "gate_c_outcome": replay["gate_c_outcome"],
        "prediction_vector_sha256": prediction_sha256,
        "channel_counts": replay["population"]["channel_counts"],
        "archive_prediction_disagreements": len(archive_disagreements),
        "geometry_records": len(geometry_records),
        "geometry_inconsistencies": len(geometry_inconsistencies),
        "runtime_seconds": replay["runtime"]["full_wall_seconds_excluding_model_load"],
        "peak_gpu_memory_bytes": replay["runtime"]["peak_gpu_memory_bytes"],
    }, indent=2))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return replay


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["pilot", "full"])
    args = parser.parse_args()
    if args.stage == "pilot":
        run_pilot()
    else:
        run_full()


if __name__ == "__main__":
    main()
