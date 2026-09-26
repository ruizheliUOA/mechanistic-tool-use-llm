# Integrity findings and release blockers

**Full release checklist: FAIL / incomplete.** This is an inspected private-source
candidate, not a verified anonymous submission artifact. No frozen verdict or protocol was changed to make a check pass.

## Gemma earlier-manifest discrepancy: provenance reconciled

See [PROVENANCE_RECONCILIATION.md](PROVENANCE_RECONCILIATION.md). The current
conjunction bytes are already present and correctly hashed by the formal manifest
in the pre-execution commit, before authorization and sealed access. That chain
has now been verified through the subsequent re-freeze and execution commits.
The parent-directory older digest still fails and its preimage remains unavailable.
It is an openly retained historical hash discrepancy, not evidence that the current
conjunction was introduced after the sealed run.

## Two unresolved inventory bookkeeping discrepancies

In `final/results/qwen3_4b_w2c_formal_v1/FILE_INVENTORY.json`:

| Referenced file | Recorded SHA-256 | Actual SHA-256 |
|---|---|---|
| `FILE_INVENTORY.json` (self-entry) | `5d048588b40c43541ebcaf37ed49e4d06005f6a169b4512346b26a124ae6387c` | `e0a22196e94aefea329f8792fa23dae7ca24578d6ee7b4952b12692e7ba19dec` |
| `HASHES.json` | `b840a98fdd1f151f40a2b570f700d25321e2bb9a9e02b25d47ef4d9637732318` | `c1e186dcc12b436a7b8d52d426d25fb9f54747018179d1c2c8de4622f3914c4f` |

Both discrepancies are already present in the source. No matching historical
preimages were found. The current `HASHES.json` independently validates the actual
`FILE_INVENTORY.json` and the listed available result payloads, but that does not
make the inventory's older assertions true. They remain failed hash assertions;
no circular/self hash was silently rewritten.

## Explained pre/post-execution differences

Two original hash references identify earlier states of mutable execution records:

- Llama `phase8_part0_hashes.json` predates resolving primary-model access in
  `PHASE8_EXECUTION_MANIFEST.md`.
- Gemma `FORMAL_HASHES.json` describes the pre-execution `ONE_SHOT_LEDGER.json`,
  while the current ledger records the completed one-shot run.

For both, exact matching historical bytes were recovered to `provenance_snapshots/`.
The mapping records both hashes. Historical and completed records remain distinct;
the old hash is not described as matching the completed file.

## Denominator and availability limits

- Qwen3-8B arm records cover 392 distinct baseline sample IDs, including all 87
  channel errors, but not a separate full 548-row baseline. Its 211 population-correct
  denominator is preserved from the frozen summary, not independently recounted.
- Qwen3-4B's bundled baseline file contains 452 source-prediction rows, with 186
  correct. It is not the complete 548-row baseline; its population denominator 214
  is preserved from the summary. The exposed denominator 50 and one break are
  directly reconstructed.
- Gemma's complete 548-row baseline has 112 channel errors and 223 correct; its
  real arm has 96 routed channel errors and 11 exposed-correct rows. The frozen
  formal summary's target-gain denominator is 96. The artifact explicitly preserves
  this scope difference and makes no substitution of 112 for 96.
- Model checkpoints, some caches, historical missing destinations, and unrecorded
  dataset revisions cannot be verified from this package. The unavailable-reference
  ledger distinguishes missing bytes from verified hashes.

## Checks actually executed

The standalone verifier checks delivered-byte hashes, 40 recovered LFS payloads,
Python syntax, per-arm sample uniqueness, zero-control equality, 59 random arms in
each sealed setting, destination/exposure/net conservation, principal verdicts,
frozen target-hit interval/index consistency, the historical Phi seed-42 counts,
and canonical non-test split uniqueness. The source's claim-tier and figure-source
consistency checks were also run. No model inference or sealed experiment ran.

Identity and secret checks cover the candidate's files, filenames, text, recovered
JSONL payloads, numeric-container metadata, PDF text/metadata/annotations, and PNG
metadata/OCR. Git history and remote/fresh-clone checks are reported separately in AUDIT_STATUS.md.
Anonymous mirror verification remains a distinct requirement.

## Legacy external-file hash literals

The three historical `phase10_2_expanded_random_null{,_v2,_v3}.py` scripts
contain a tokenizer-config hash literal with 62 hexadecimal characters, rather
than a valid 64-character SHA-256. It was unchanged from the source and was not
validated as a model-file hash. These are the three hash-like gitleaks findings
that do not have the length of a complete digest. None is an account credential;
these historical runners are not advertised as runnable reproduction checks.
