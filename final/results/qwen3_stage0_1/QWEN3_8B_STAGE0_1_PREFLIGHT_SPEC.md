# Qwen3-8B Stage 0/1 configuration-only preflight

Run only after the Block 0 commit is pushed:

`python scripts/qwen3_8b_stage0_1.py --preflight`

The preflight verifies the committed protocol/runner hashes, exact official model and tokenizer files at immutable revision, all five weight SHA256 values and sizes, fixed software versions, train/dev selective index, structural evaluation firewall, empty Stage 1 output namespace, normalized layer mapping, LFS attributes, four-candidate mask schema, and thinking-disabled template. It loads the tokenizer only. It renders and archives the lexicographically first authorized TRAIN prompt, including exact messages, tool placement, deterministic empty-think boundary, candidates, token IDs and scoring-mask positions.

It must not load the full model, execute a forward, compute any Qwen3 score, count a channel, open an evaluation manifest/index/payload, create a Stage 2 artifact, or expose a scientific result. PASS outputs are committed and pushed separately before `--run-stage1`. Any model/tokenizer identity, thinking-mode, data-firewall, layer, hash, version, LFS, or output-namespace failure blocks Stage 1.
