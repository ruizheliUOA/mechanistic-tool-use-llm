# Preflight specification

`python scripts/qwen3_stage2_formal.py --preflight`

Configuration-only. Verifies every frozen artifact hash, model and tokenizer file
identity at the frozen revision, software versions, that the preregistered
configuration matches the committed artifacts exactly, direction validity and
unit norm, the formal random count and every vector hash, seed disjointness from
all development and pilot seeds, the execution freeze (arm order, batch size,
single process, single model load, no resume), an empty formal output namespace,
disk headroom, and the sealed-evaluation identity inherited by reference.

It must not open the evaluation payload, enumerate evaluation IDs, load the full
model, compute evaluation predictions, or compute endpoints.

The result is committed and pushed separately.
