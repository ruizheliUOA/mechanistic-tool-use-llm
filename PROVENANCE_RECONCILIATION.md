# Historical provenance reconciliation

A follow-up inspection of the preserved Git history resolves the earlier uncertainty
about the bytes used for the Gemma development conjunction. No scientific field,
protocol, verdict or existing frozen manifest was changed.

## Gemma pre-execution chain

The initial pre-SEALED commit contains all three of:

1. `DEV_CONJUNCTION.json` with SHA-256
   `52644754757e71acfe2fcd90bff81ca797ee46de7dd4a13d2050b9888f123780`.
2. `formal_freeze/FORMAL_HASHES.json` recording that exact hash for both
   `DEV_CONJUNCTION.json` and `DEV_CHANNEL_VERDICTS.json`.
3. A one-shot ledger with authorization not granted, zero runs, zero sealed rows
   read and no observed scientific endpoint; no access-marker file exists then.

The subsequent runner re-freeze and completed-run commits descend from that commit.
The conjunction bytes and their digest in the formal manifest are unchanged through
all three. Exact initial snapshots are provided in `provenance_snapshots/gemma_presealed/`;
`VERIFICATION.json` states the observed stages. This confirms preserved pre-execution
provenance from Git, although Git history alone is not an independent trusted timestamp.

The earlier parent-directory `FORMAL_FREEZE.json` still has a different digest,
`5dc32e1b04cb91f69639855aeb8aede09e85d3d3cc54c9605733ba8342c66f9a`.
Its preimage was not found. That individual assertion **fails**; it is not silently
repaired or described as passing. The stronger claim made in the first packaging
audit—that the currently available conjunction had no established pre-execution
provenance—was incomplete, because it overlooked the contemporaneous formal manifest.
This document corrects that packaging interpretation without changing research evidence.

## Qwen3-4B baseline scope

The preserved `scripts/qwen3_4b_formal_package.py` explicitly creates
`FORMAL_BASELINE_RECORDS.jsonl` from the union of the arms' baseline fields. Its
`baseline_records_note` documents that rows whose baseline prediction is not the
source never enter an arm, and that the full 548-row denominator is retained only
in aggregate. Thus the 452-row baseline export is intentional and does not establish
a missing 548-row file that can be filled with other experiments' records.
The population denominator 214 remains an aggregate-only assertion; this audit
cannot independently reconstruct all its rows.

## What is still not a full PASS

The two Qwen3-4B inventory self/cross-hash assertions still fail. Missing external
inputs and complete Qwen population baselines remain explicitly limited. The
available destination counts, exposed-correct denominators, control records and
principal verdicts pass their stated read-only checks. A private source repository
can preserve these disclosures; it is not a claim that every historical hash or
reproduction promise has passed, nor a verified Anonymous GitHub mirror.
