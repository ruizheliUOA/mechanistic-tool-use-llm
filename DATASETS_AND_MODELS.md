# Public inputs, licenses, and runtime records

No model checkpoint or large raw dataset is distributed in this artifact. Preserved
outcome records are evidence, not a substitute for acquiring the benchmark input.
The commands below are preparation examples, not executed formal experiments or
promises that current downloads reproduce historical serialization.

## When2Call

Official source: [NVIDIA/When2Call](https://github.com/NVIDIA/When2Call) and
[nvidia/When2Call](https://huggingface.co/datasets/nvidia/When2Call).
The [dataset card at the recorded revision](https://huggingface.co/datasets/nvidia/When2Call/blob/0582f7749df63a96fdc3070932e83e72396ace53/README.md)
states **CC BY 4.0**, synthetic generation and automated labeling. Preserve NVIDIA
attribution and comply with the upstream license when obtaining or distributing it.

The recorded Qwen3 input revision is
`0582f7749df63a96fdc3070932e83e72396ace53`, configuration `test`, split `mcq`,
3,652 examples. The project uses these benchmark examples for its own development
and sealed allocations; it does not equate the upstream split name with the
project's sealed split. With `datasets` installed in a separate environment:

```python
from datasets import load_dataset
rows = load_dataset("nvidia/When2Call", "test", split="mcq",
                    revision="0582f7749df63a96fdc3070932e83e72396ace53")
assert len(rows) == 3652
```

Historical runners expect `data/processed/qwen25_7b_w2c/w2c_test_mcq.jsonl`.
The expected serialized file hash is
`8c3694e583eeeb8dbc297e6cd90da70efc68efa4b6adb7227523e828c6b7b14c`.
A generic JSON export is not guaranteed to match the original ordering/bytes.
The recovered `final/results/splits/{train_idx,val_idx}.json` and the canonical
`final/results/qwen3_stage0_1_v2/QWEN3_V2_SPLIT_INDEX.csv` preserve the archived
allocation. Do not generate a new split to resolve a missing-input mismatch.
Some earlier baseline scripts load the dataset without a revision; their exact
upstream bytes cannot be inferred from this newer recorded revision.

## MetaTool

Official source: [HowieHwong/MetaTool](https://github.com/HowieHwong/MetaTool).
The [upstream MIT license](https://github.com/HowieHwong/MetaTool/blob/master/LICENSE)
retains copyright attribution to Yue Huang (2023); this is third-party attribution,
not an identity to remove. This project uses the binary tool-need Task 1 data,
currently at `dataset/tmp_dataset/Task1.json`, not the whole ToolE benchmark.

The original historical upstream revision is **not recorded** in the inspected
conversion script. The repository revision observed during this audit is
`35e81bb7576826e980c80fed8f8c0a2b4a1e6fbb`; it is not asserted to be the experimental
revision. No raw Task1 dataset is bundled. After obtaining the correct upstream
file, in a separate staging copy:

```sh
python3 scripts/convert_metatool_binary.py --task1_json /path/to/Task1.json --out_dir data/processed/metatool_binary --seed 42
```

The converter maps positive/negative to tool_call/no_tool and writes all/train/val/test
JSONL files with its archived stratified 70/15/15 rule. Seed 42 does not establish
identity of the input or of an unrecorded upstream revision. See `metatool_audit/`
for partition comparisons and historical rescorability limits.

## ACEBench

Official source: [chenchen0103/ACEBench](https://github.com/chenchen0103/ACEBench),
the current destination of the historical `ACEBench/ACEBench` URL, under its
[MIT license](https://github.com/chenchen0103/ACEBench/blob/main/LICENSE.md).
The project uses six single-turn categories in English and Chinese (900 rows),
not the multi-agent benchmark. Exact source-file SHA-256 values are preserved in
`final/results/acebench_generation_readout/dataset_manifest.json`; no repository
revision is recorded there. Current upstream files must match those hashes before
claiming historical byte identity.

Stage the six English JSONL files directly in `/path/to/acebench-input/` and the
Chinese counterparts in its `zh/` subdirectory (upstream uses `data_all/data_en`
and `data_all/data_zh`). In a separate copy of this artifact:

```sh
python3 scripts/convert_acebench_decision.py --raw-dir /path/to/acebench-input --seed 42
```

Outputs go to `data/processed/acebench_decision/`. The archived converter uses
native category labels and stratifies by label/language at 70/15/15. Generation,
parsing, and the readout protocol are separate stages. This support-stopped setting
is framework transfer, not an additional formal licence.

## Models and libraries

| Model | Recorded revision / access | Source of exact configuration |
|---|---|---|
| [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | `b968826d9c46dd6066d109eabc6255188de91218` | `final/results/qwen3_stage2_formal/QWEN3_STAGE2_INPUT_MANIFEST.json` |
| [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | `1cfa9a7208912126459214e8b04321603b3df60c` | `final/results/qwen3_4b_w2c_dose/FORMAL_PROTOCOL_FROZEN.json` |
| [Gemma-2-9b-it](https://huggingface.co/google/gemma-2-9b-it) | `11c9b309abf73637e4b6f9a3fa1e92e615547819`; upstream access/license acceptance may be required | `research_exploration/sakiko_final_model_breadth_panel_v1/gemma/formal_freeze/FORMAL_CONFIG.json` |
| [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | `0e9e39f249a16976918f6564b8830bc894c89659`; upstream access/license acceptance may be required | `final/results/phase8_prospective_llama/PHASE8_EXECUTION_MANIFEST.md` |

Other historical model names and any recorded revisions remain in their package
protocols and manifests. Missing historical revisions are not replaced with current
ones. Model licenses remain upstream; this artifact's code license does not grant
rights to model weights. Existing package manifests record shard hashes, tokenizer,
attention, dtype, and runtime assumptions. None of the model shards was downloaded
or rehashed for this audit. `requirements.txt` lists optional imports; a complete
historical dependency lock is unavailable. Safe bundled-evidence checks use only
the Python standard library.
