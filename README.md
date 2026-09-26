# SAKIKO: code and preserved research evidence

Companion artifact for *When Does Correction Become Repair?* It contains code,
frozen protocols, outcome records, summaries, and the evidence audit trail.
Historical analyses and prospective sealed evaluations retain their original scope.
Manuscript drafts and private development history are outside this artifact.

**Candidate status:** preserved evidence and runnable inspection checks, with known
historical hash and availability limitations in [INTEGRITY_FINDINGS.md](INTEGRITY_FINDINGS.md).
The Gemma pre-execution chain is reconciled in [PROVENANCE_RECONCILIATION.md](PROVENANCE_RECONCILIATION.md).
A private source repository is not a verified anonymous mirror or a blanket full
reproducibility PASS. See [AUDIT_STATUS.md](AUDIT_STATUS.md).

## Shortest reading path

1. [ARTIFACT_INDEX.md](ARTIFACT_INDEX.md): claims, protocols, evidence, and scripts.
2. [REPRODUCIBILITY.md](REPRODUCIBILITY.md): safe checks and their actual limits.
3. [LFS_MANIFEST.md](LFS_MANIFEST.md): payloads and unavailable inputs.
4. [final_evidence/FINAL_PAPER_EVIDENCE.csv](final_evidence/FINAL_PAPER_EVIDENCE.csv)
   and [EVIDENCE_CORRECTIONS.md](final_evidence/EVIDENCE_CORRECTIONS.md): the index
   and appended corrections. Older audit prose can predate these corrections.

## Verify the bundled evidence

Python 3.10+ is sufficient. Run from the artifact root; no GPU, model, account, or
third-party package is needed:

```sh
python3 tools/verify_artifact.py
python3 final_evidence/tier_check.py
```

The first checks delivered files and hashes, recovered payloads, Python syntax,
destination counts, zero controls, random-arm counts, selected denominators, and
preserved formal verdicts. Missing files and LFS pointers cause explicit failure.
The second checks the existing claim-to-evidence tier mapping; it does not establish
the truth or completeness of every claim.

## Scientific boundaries

- Qwen3-8B retains `FORMAL_CONFIRMATORY_SUCCESS`; Qwen3-4B and Gemma retain their
  principal `DECLINE` verdicts. Positive point estimates do not replace the gate.
- Aggregate net gains +40 and +39 differ from channel gold arrivals 38 and 37.
- Population collateral and collateral among exposed baseline-correct decisions
  are separate estimands: Qwen3-8B's recorded 0/211 is not tight 0/6 evidence.
- Gemma's frozen summary uses 96 routed channel errors; the full baseline contains
  112. These denominators are not interchangeable.
- Historical Phi-3.5 results and exploratory work acquire no prospective licence
  through packaging. No experiment was rerun.

## Availability

All 40 formerly pointer-only files now contain verified full payloads (87,789,671
bytes in total). Their original payload OIDs and sizes are recorded in
[ARTIFACT_AVAILABILITY.json](ARTIFACT_AVAILABILITY.json). Model checkpoints and
large raw datasets are not bundled. [DATASETS_AND_MODELS.md](DATASETS_AND_MODELS.md)
records official sources, known revisions, licenses, and reconstruction limits.

Frozen runners retain their original environment assumptions, including some
historical absolute paths. They are not the quick-start interface. In particular,
`--gate-only` does not guarantee a read-only operation; see the command classes in
[REPRODUCIBILITY.md](REPRODUCIBILITY.md).

[ANON_RELEASE_NOTES.md](ANON_RELEASE_NOTES.md) distinguishes byte-preserved evidence,
operational edits, recovered dependencies, and the separately named anonymous
derivative. Original frozen hash manifests have not been rewritten.
