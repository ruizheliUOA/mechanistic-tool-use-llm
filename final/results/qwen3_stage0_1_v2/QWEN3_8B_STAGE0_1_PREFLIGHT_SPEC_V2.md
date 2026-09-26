# Qwen3 V2 configuration-only preflight specification

Run after the V2 freeze commit is pushed:

`python scripts/qwen3_8b_stage0_1_v2.py --preflight`

The preflight verifies the committed V2 hash manifest and every artifact in it,
the V2 runner hash, the V1 runner hash, the attention-backend audit hash and its
`ATTENTION_BACKEND_ACCEPTED` disposition, the exact model and tokenizer identity
at the frozen immutable revision including all weight sizes and SHA256 values,
the frozen software versions, the inherited non-test population identity by
baseline and activation hash with 3,104 unique immutable UUIDs, the inherited
sealed-evaluation identity **by frozen manifest reference only**, the presence
of both frozen activation sites `L_obs` and `L_inj`, the artifact-audit outcome,
an empty V2 output namespace, and Git LFS availability for V2 numerical outputs.

It must not compute the V2 split, compute V2 support counts, load test data,
load the full model, or compute geometry. It loads no model weights at all.

PASS output is committed and pushed separately before Block 4 executes.
