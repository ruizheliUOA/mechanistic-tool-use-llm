# Experiment code map

| Experiment | Main code / local modules |
|---|---|
| Qwen3-8B sealed | `qwen3_stage2_formal_v4.py`; `qwen3_8b_stage0_1{,_v2}.py`; `qwen3_stage1_5_gradient_mean.py`; `qwen3_stage2a_v2_dev_calibration.py` |
| Qwen3-4B sealed | `qwen3_4b_w2c_formal.py`, `qwen3_4b_formal_package.py`; `qwen3_4b/stage_d_e/` and `qwen3_4b/baseline/` |
| Gemma sealed | `gemma/formal_freeze/run_gemma_formal.py`; development modules under `gemma/` |
| Phi historical | `phi35/sakiko_v3/p0_run_locked_eval.py`, `p1_multiseed_locked_eval.py`, `p2_placebo_controls.py`, `run_v31.py`; independent controls in `phi35/phi_placebo/` with `phi_mainline/` dependencies |
| Qwen2.5 historical | `qwen7b_native_sakiko*.py`, `qwen7b_followup_locked_tests.py`, `phase3_*.py` |
| Mistral historical | `phase5_mistral_*.py` |
| Llama historical | `phase7_*.py`, `phase8_*.py` |
| MetaTool | `convert_metatool_binary.py`, `metatool_*.py` |
| ACEBench | `convert_acebench_decision.py`, `acebench_*.py` |
| Qwen3.5 screening | `qwen35/` |

Module names/bytes are retained for provenance. Original relative layout can be
materialized with `tools/prepare_legacy_layout.py`; historical machine roots and
external inputs still require the recorded environment. Some files load models
or create outputs at import time. Do not execute/import them as a safety check.
Use `tools/verify_artifact.py` to inspect data without rerunning experiments.
