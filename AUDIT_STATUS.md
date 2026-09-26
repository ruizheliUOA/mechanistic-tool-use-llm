# Candidate audit status

Final Anonymous URL: **尚未產生 / not generated**.

This report concerns the current local candidate, not an approved submission.
PASS applies only to each stated scope; omitted future stages are FAIL.

| Check | Result | Evidence / limitation |
|---|---|---|
| Original repository unchanged | PASS | During this resumed task: all 900 source-tree file hashes/status matched; all captured Git storage files, including added/deleted-file detection, matched across the three local source stores. |
| Separate repository | FAIL | Separate sibling candidate directory, no symlink/shared inode/object database. New Git metadata is to be audited separately from this content snapshot. |
| Fresh Git history | FAIL | To be checked after this content snapshot is committed; final administrative audit is kept outside this snapshot. |
| Anonymous author / committer metadata | FAIL | No artifact commit exists yet; anonymous commit metadata is therefore not verified. |
| Identity scan | PASS | Current candidate: source-derived private dictionary, all file names/text, recovered records, PDF content/metadata, PNG metadata/OCR, numeric-container metadata. No confirmed author identity hit. |
| Email scan | PASS | Candidate matches reviewed: reserved example-domain benchmark addresses and anonymous/SSH placeholders; no personal email retained. |
| Local path scan | PASS | Private home paths removed from the prior seven code edits; generic historical provider roots retained in frozen records and classified explicitly. Historical runners are not advertised as portable quick-start commands. |
| Secret scan | PASS | Candidate regex/context scan and redacted gitleaks scan; detected generic-key matches reviewed as recorded file-hash literals (including three malformed legacy tokenizer digests). No confirmed credential. Commit/history scan remains unexecuted because no Git history exists. |
| Original/private repo URL scan | PASS | Candidate scan found no original/private source remote or personal source URL; dictionary and current source commit remain private. |
| Metadata / external-link audit | PASS | 27 PDFs and 32 PNGs inspected, PNG OCR completed, three representative figures manually viewed; software attribution/XML identifiers and benchmark/model links classified. No mirror rendering was available. |
| Frozen evidence / hash integrity | FAIL | 40 recovered payloads verify; 23 historical dependencies match recorded hashes. Three original hash assertions remain unresolved, including the older Gemma parent-manifest entry; formal pre-execution conjunction provenance is reconciled; two execution-record transitions have exact recovered preimages. |
| LFS / large-file audit | FAIL | 40/40 inventoried LFS objects recovered and verified (87,789,671 bytes), no remaining pointer among those files. Other required provenance/complete-baseline availability is limited; no blanket claim-complete availability PASS. |
| Dataset / license documentation | PASS | Official When2Call, MetaTool and ACEBench sources/licenses documented; known versus unrecorded revisions, preprocessing locations and limitations distinguished. Raw data/model weights not bundled. |
| Reproducibility consistency | FAIL | Quick-start verifier, claim-tier and figure-source checks pass within documented scope. Historical hash assertions and incomplete Qwen baseline denominators prevent full reproduction approval. |
| GitHub private repo / branch settings | FAIL | Private source creation/fresh clone are administrative steps to be checked outside this immutable content snapshot. Existing CLI authentication was verified without disclosing credentials. |
| Fresh clone audit | FAIL | No release commit/remote exists to clone. A separately extracted delivery archive is checked as a local transport test, not a remote clone. |
| Anonymous mirror audit | FAIL | No mirror created, no anonymous URL, no logged-out reviewer verification. Browser inventory returned zero available sessions. Official platform docs were inspected but are not evidence of deployed mirror behavior. |

No sealed experiment, model inference, verdict revision or source write occurred.
The blocking original-record discrepancy is detailed in `INTEGRITY_FINDINGS.md`.
Private administrative/source records are stored outside this artifact.

This in-repository report records content preparation, before Git commit/remote operations.
A later administrative audit can verify those operations without modifying this content
snapshot or pretending that unresolved historical assertions passed. See
`PROVENANCE_RECONCILIATION.md` for the follow-up provenance finding.
