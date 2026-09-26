# Gate-only specification

`python scripts/qwen3_stage2_formal.py --gate-only`

Run after the final runner version is frozen. Verifies, without opening the
evaluation payload: all artifact hashes; model and tokenizer identity; software
versions; the selected channel and configuration; direction hashes; formal random
count and hashes; seed disjointness; arm completeness; fixed arm order; batch
size 1; the same-run single-load path; that the zero gate precedes endpoint
computation; the raw record schema; the primary/secondary hierarchy; the
bootstrap specification; the support-outcome branch; the VOID rules; the
sealed-evaluation firewall; and an empty formal output namespace.

Outputs are committed and pushed separately.
