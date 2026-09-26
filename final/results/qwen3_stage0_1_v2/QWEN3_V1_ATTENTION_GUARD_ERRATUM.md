# Erratum: V1 long-sequence attention guard

Status: **V1 remains immutable.** This erratum documents a runner change made
*during* V1 execution under the Stage 0/1 engineering-retry provision. It does
not modify any V1 artifact, result or conclusion.

## Why the guard was necessary

The frozen protocol fixes `attention_implementation: "eager"`
and `dtype: bfloat16`. Eager attention materialises a
`[1, 32, S, S]` float32 softmax buffer. On the single 24 GiB RTX 4090 D used
for this line, that buffer plus the 15.25 GiB of BF16 weights exceeds capacity
for the longest authorized sequences.

Three Stage 1 attempts established this ordering:

1. `RuntimeError` — deterministic algorithms require `CUBLAS_WORKSPACE_CONFIG`
   to be set before process start (environment fix, no code change).
2. `OutOfMemoryError` with 1.89 GiB reserved-but-unallocated — allocator
   fragmentation (environment fix `expandable_segments:True`, no code change).
3. `OutOfMemoryError` with 0.11 GiB reserved-but-unallocated — a genuine
   capacity ceiling, not fragmentation. No environment-only remedy remained.

The frozen protocol forbids skipping rows or shrinking populations, and no
authorized sequence reaches the frozen 8192-token blocking threshold, so no
predeclared structural exclusion applied. The alternatives were to change the
frozen attention implementation (a scientific-definition change requiring a new
protocol version) or to make the existing eager computation fit. The latter was
chosen because it changes no frozen scientific value.

## What the guard does

- Sequences with query length <= **4096** are dispatched to the
  **unmodified upstream** `eager_attention_forward`. This path is unchanged.
- Longer sequences evaluate the identical eager expression in chunks of
  **8 attention heads**. Softmax is taken over the key axis independently
  for each `(head, query)` row, so partitioning the head axis changes no value
  analytically.
- On the chunked path the attention-probability tensor is returned as `None`.
  The runner never sets `output_attentions`, and `Qwen3DecoderLayer.forward`
  discards that value unless it is requested. Rebuilding the full
  `[1, 32, S, S]` tensor would defeat the guard.

## Files and hashes

| item | SHA256 |
|---|---|
| superseded Block 0 runner | `297680d19010caddd521aedfce453976cc477898295d8c7dffe52ed0d57121b4` |
| current runner (with guard) | `b1830b4be291c1adfe7e843e8de7fdea0ee1e4e1b6c10c25c6fc40465073f4ac` |

The V1 hash manifest `QWEN3_8B_STAGE0_1_HASHES.json` records the current runner
hash and retains the superseded value in its `runner_revision_note`. The V1
runner file itself was never overwritten in history; both versions are
reachable in the commit graph.

## Scope of the deviation in V1

- affected rows: **6** of 3,104 authorized non-test rows
- affected scored sequences: **24** of 12,416 (0.193%)
- affected DEV rows in V1: **0**

All six affected rows were V1 TRAIN rows. No V1 DEV row used the chunked path,
so no V1 dev-side quantity was computed on it.

| V1 project index | immutable UUID | V1 split | gold | prompt tokens | guarded sequences (mode, length) |
|---:|---|---|---|---:|---|
| 410 | `ad0f48a6-bc9b-42b8-b487-0c21012e70ac` | train | request_for_info | 4625 | direct 4670, cannot_answer 4649, tool_call 4642, request_for_info 4641 |
| 723 | `4d92a312-65f4-4732-961a-a4caaf4b84d7` | train | request_for_info | 5172 | direct 5218, cannot_answer 5199, request_for_info 5198, tool_call 5189 |
| 1066 | `b505311d-27b6-45f1-b09d-effa9ead72e5` | train | cannot_answer | 5118 | direct 5165, cannot_answer 5145, request_for_info 5139, tool_call 5135 |
| 1741 | `3bff4027-c393-41af-97d6-5502925dfd7d` | train | cannot_answer | 4548 | direct 4593, cannot_answer 4575, request_for_info 4571, tool_call 4565 |
| 2376 | `eabd1570-92b2-4022-b73e-3603ed49fa65` | train | tool_call | 4627 | direct 4672, request_for_info 4654, cannot_answer 4651, tool_call 4644 |
| 2690 | `58bcb8cc-2520-4275-8f49-be930e5144c7` | train | tool_call | 5185 | direct 5231, cannot_answer 5212, request_for_info 5206, tool_call 5202 |

Every mode of each affected row exceeds the threshold, because in each case the
prompt alone already exceeds it.

## Empirical module-level equivalence evidence

All comparisons use the **same GPU**, the **same BF16 dtype**, and
**byte-identical Q/K/V and mask tensors** captured from the real model. There
is no CPU/GPU and no FP32/BF16 confound. Native attention was evaluated after
the model was freed so that it fits at every tested length.

### A. Synthetic length sweep (real captured tensors from row 2690, sliced)

| layer | S | path | bitwise | max abs dev | mean abs dev | max ULP | deterministic |
|---:|---:|---|---|---:|---:|---:|---|
| 0 | 512 | native-passthrough | True | 0.000e+00 | 0.000e+00 | 0 | True |
| 0 | 1024 | native-passthrough | True | 0.000e+00 | 0.000e+00 | 0 | True |
| 0 | 2048 | native-passthrough | True | 0.000e+00 | 0.000e+00 | 0 | True |
| 0 | 4095 | native-passthrough | True | 0.000e+00 | 0.000e+00 | 0 | True |
| 0 | 4096 | native-passthrough | True | 0.000e+00 | 0.000e+00 | 0 | True |
| 0 | 4097 | chunked | True | 0.000e+00 | 0.000e+00 | 0 | True |
| 0 | 4200 | chunked | False | 9.766e-04 | 1.621e-09 | 17 | True |
| 0 | 4608 | chunked | True | 0.000e+00 | 0.000e+00 | 0 | True |
| 0 | 5000 | chunked | True | 0.000e+00 | 0.000e+00 | 0 | True |
| 0 | 5231 | chunked | True | 0.000e+00 | 0.000e+00 | 0 | True |
| 21 | 512 | native-passthrough | True | 0.000e+00 | 0.000e+00 | 0 | True |
| 21 | 1024 | native-passthrough | True | 0.000e+00 | 0.000e+00 | 0 | True |
| 21 | 2048 | native-passthrough | True | 0.000e+00 | 0.000e+00 | 0 | True |
| 21 | 4095 | native-passthrough | True | 0.000e+00 | 0.000e+00 | 0 | True |
| 21 | 4096 | native-passthrough | True | 0.000e+00 | 0.000e+00 | 0 | True |
| 21 | 4097 | chunked | True | 0.000e+00 | 0.000e+00 | 0 | True |
| 21 | 4200 | chunked | False | 7.812e-03 | 1.402e-08 | 14 | True |
| 21 | 4608 | chunked | True | 0.000e+00 | 0.000e+00 | 0 | True |
| 21 | 5000 | chunked | True | 0.000e+00 | 0.000e+00 | 0 | True |
| 21 | 5231 | chunked | True | 0.000e+00 | 0.000e+00 | 0 | True |
| 26 | 512 | native-passthrough | True | 0.000e+00 | 0.000e+00 | 0 | True |
| 26 | 1024 | native-passthrough | True | 0.000e+00 | 0.000e+00 | 0 | True |
| 26 | 2048 | native-passthrough | True | 0.000e+00 | 0.000e+00 | 0 | True |
| 26 | 4095 | native-passthrough | True | 0.000e+00 | 0.000e+00 | 0 | True |
| 26 | 4096 | native-passthrough | True | 0.000e+00 | 0.000e+00 | 0 | True |
| 26 | 4097 | chunked | True | 0.000e+00 | 0.000e+00 | 0 | True |
| 26 | 4200 | chunked | False | 1.562e-02 | 1.265e-08 | 14 | True |
| 26 | 4608 | chunked | True | 0.000e+00 | 0.000e+00 | 0 | True |
| 26 | 5000 | chunked | True | 0.000e+00 | 0.000e+00 | 0 | True |
| 26 | 5231 | chunked | True | 0.000e+00 | 0.000e+00 | 0 | True |

The single non-bitwise length in this sweep is **S = 4200**, which is *not* one
of the 24 production lengths. At that length, at layer 26, 286 of 17,203,200
output elements differ (0.0017%), mean absolute deviation is 1.27e-08, and the
output-norm relative difference is about 8e-08. The deviation arises from
cuBLAS selecting a different GEMM algorithm for the 8-head batched shape.

### B. The 24 real production lengths, both frozen sites

48 comparisons (24 sequences x layers 21 and
26), each against that row's own captured tensors:

- bitwise identical: **48/48**
- max absolute deviation: **0.000e+00**
- max relative output-norm difference: **0.000e+00**
- max ULP: **0**
- all outputs finite: **True**
- deterministic on repeat: **True**

### C. All-layer induction over every affected sequence

For each of the 24 affected sequences, the guarded and native
implementations were compared at **every one of the 36 decoder layers** on
identical inputs:

- total layer comparisons: **864**
- bitwise identical: **864**
- all sequences all layers bitwise identical: **True**

Because layer 0 receives the token embeddings, which do not depend on the
attention implementation, and because every layer maps identical inputs to
bitwise-identical outputs, it follows by induction that the complete forward
pass — and therefore the candidate log-probability score — is **bitwise
identical** between the guarded and native implementations *for these 24
sequences* on this hardware and software stack.

## Limitations of this evidence

These limits are stated explicitly and are not to be elided:

1. **The equivalence is not universal across lengths.** At S = 4200 the two
   implementations differ. The claim "bitwise identical to upstream at all
   lengths" is **false** and is not made.
2. **The induction result is specific to the 24 tested sequences**, this GPU,
   this dtype, this CUDA/cuBLAS/PyTorch/Transformers stack, and these
   deterministic settings. It is an exact statement about those runs, not a
   general theorem about the two implementations.
3. **No analytic propagation bound is claimed.** No mathematically valid bound
   on downstream score deviation as a function of local attention deviation was
   derived, so for any length where outputs are not bitwise identical, an
   empirical local deviation must not be treated as a certified end-to-end score
   bound.
4. **The captured Q/K/V were produced under the guarded path.** For the
   induction this is sound, because the hypothesis being propagated is exactly
   that both paths agree bitwise; but it means the tensors are not independent
   of the implementation being audited.

## Invariants checked

- causal semantics: perturbing key/value positions from 4000 onward left all
  output positions before 4000 bitwise unchanged, and changed positions at and
  after 4000 — the guarded path does not leak future positions into earlier
  ones.
- finiteness: all outputs finite at every tested length and layer.
- determinism: repeated guarded evaluation on identical inputs is bitwise
  reproducible at every tested length and layer.
- masking: identical mask slice supplied to both implementations; no semantic
  masking discrepancy observed.

## V2 disposition

**ATTENTION_BACKEND_ACCEPTED.**

The guarded implementation is adopted as the formal V2 execution backend for
every sequence exceeding the frozen threshold of 4096 tokens. Sequences at or
below the threshold continue to use the unmodified upstream implementation.

This adoption is a disclosed engineering formalisation. It does not alter the
frozen attention implementation (`eager`), dtype, batch size, populations,
candidate construction, scoring, layers, support gate, direction estimator,
Router specification or geometry definitions.
