# Anonymous packaging changes

No measured value, sample, label, split, seed, threshold, direction, dose, statistical
rule, preregistration or formal verdict was changed. Original frozen manifests are
preserved, including their discrepancies. Full-checklist completion and anonymous mirror verification remain outstanding.

## Preserved and recovered

- Code, protocols, preregistrations, principal verdicts, summaries, claim/evidence indices,
  statistical scripts, figures and necessary audits are retained.
- All 40 inventoried LFS payloads were recovered and checked against original OIDs/sizes.
- 23 additional referenced dependencies were recovered from history only when their
  complete file hashes matched hashes already recorded in the source manifests.
- Two separately stored pre-execution snapshots explain historical execution-record
  transitions; see `provenance_snapshots/SNAPSHOT_MAPPING.json`.
- Both previously edited frozen `.gitattributes` files were restored byte-for-byte.

## Operational edits

README, the reproduction guide, the LFS guide, the dependency inventory and release
documentation now describe what is actually bundled and distinguish safe checks from
historical experiment commands. The new `tools/verify_artifact.py` is read-only.

The following 11 operational/code files retain prior identity-only edits. For all
seven Python scripts, AST structure and non-string constants were checked unchanged.
Only private-path string values were altered; these edits do not change parameters.

| File |
|---|
| `constraints/STANDING_CONSTRAINTS.md` |
| `final/results/acebench_generation_readout/HUMAN_GIT_HANDOFF.md` |
| `final/results/mistral7b_w2c_sakiko_ca/HUMAN_GIT_HANDOFF.md` |
| `final/results/phase8_prospective_llama/HUMAN_GIT_HANDOFF.md` |
| `scripts/eval_w2c_7b_baseline.py` |
| `scripts/extract_toolsandbox_decision_candidates.py` |
| `scripts/metatool_binary_fp16_confirmation.py` |
| `scripts/metatool_binary_robustness_audit.py` |
| `scripts/metatool_binary_sakiko_pilot.py` |
| `scripts/qwen7b_native_sakiko.py` |
| `scripts/qwen7b_native_sakiko_multiseed.py` |

## Anonymous derivatives

Two documents with identity content are available under separately named `.anonymised.md`
paths. `ANON_DERIVATIONS.json` records original/derived hashes and transformations.
Neither derivative is the original frozen artifact; an original hash must not be used
to validate the derivative. No original manifest was rewritten.

## Excluded material

Manuscript drafts, submission assembly copies, narrative drafts, private author hooks
and one unmarked superseded draft claim matrix are outside this reviewer package.
Necessary manuscript provenance/sufficiency audits remain under `_evidence_audit/`.
An editable superseded presentation duplicate was excluded; its corresponding SVG,
PDF and PNG remain. Operating-system debris and caches are excluded. No missing
research outcome was manufactured, and necessary evidence was not removed to pass a scan.

## Retained contextual strings

Generic historical GPU-provider roots in frozen protocols and runners remain
byte-identical. They contain no observed personal username and are not runnable
path recommendations. Scientific references to AI models/libraries, third-party
copyrights, software documentation URLs, and reserved example-domain addresses in
benchmark examples are retained. The source-derived private identity dictionary,
source remote, current source commit and audit raw matches are not distributed.

## Verification scope

The release manifest identifies delivered bytes only. CPU checks validate available
outcome arithmetic, control inventories, selected denominators, summary/index links
and preserved verdicts. They do not repair the original frozen-record discrepancy,
verify missing model weights, establish absent historical dataset revisions, or
constitute a reviewer-side anonymous mirror audit.

## Follow-up provenance review

Exact Gemma pre-execution snapshots were added under `provenance_snapshots/gemma_presealed/`.
`PROVENANCE_RECONCILIATION.md` corrects the earlier packaging interpretation: the formal
manifest already validates the available conjunction before sealed execution. The
older parent-manifest hash remains a disclosed failed assertion. No frozen file changed.
