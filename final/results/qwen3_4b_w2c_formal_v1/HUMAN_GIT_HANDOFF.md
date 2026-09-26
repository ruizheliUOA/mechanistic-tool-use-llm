# Human Git Handoff — Qwen3-4B one-shot formal SAKIKO result

**One authorised SEALED run, completed. Result final. Nothing staged, committed or pushed.**

## Verdict

`QWEN3_4B_FORMAL_DECLINE`

The frozen ten-condition conjunction failed on conditions 3 and 6 — the two
confidence-interval conditions. Point estimates passed; the intervals did not exclude their
thresholds on 124 SEALED channel errors.

Direction specificity was **not** the failure: 0 of 59 fresh matched-random directions
reached the real direction, `p_add_one = 0.016667`, real rank 1 of 60.

## Staging

```
git add -- final/results/qwen3_4b_w2c_formal_v1
```

Nothing else. No `git add .`. Do not stage `final/results/decisive_upgrade_audit/` or the
stray root-level `Under` file.

Verify before committing:

```
git status --porcelain -- final/results/qwen3_4b_w2c_formal_v1
git diff --cached --stat
```

No file in this package is git-ignored, so no `git add -f` is required.

## Proposed commit

```
experiment: add Qwen3-4B one-shot formal SAKIKO result
```

The scientific verdict is deliberately not in the commit message, per repository convention.

## Also uncommitted

`research_exploration/qwen3_4b_formal_gate_v1/GATE_ONLY_LOG.txt` and
`GATE_ONLY_OUTPUT.json` may show as modified from the final gate replay, and the ledger now
records `formal_run_count = 1`. Stage the gate package separately if you want that recorded:

```
git add -- research_exploration/qwen3_4b_formal_gate_v1
```

## Size

29 files, 24.4 MB. The bulk is `QWEN3_4B_FORMAL_RECORDS.jsonl` (14,404 raw per-sample,
per-arm records) plus the per-arm splits derived from it.

No model weights. No activation caches. No dataset payload — records carry immutable sample
ids, four-mode log-probability scores, predictions, Router probabilities, arm identity, dose
and destination, and no prompt, answer-candidate string, dialogue or generated text.
`SAFETY_SCAN_CLEAN` across all nine categories.

## Integrity

One authorised run, invocation `e3d9d0b5-92f9-42da-86c2-d8ca2606466c`. Access marker
fsynced before the first SEALED load. 65 arms, 0 duplicate samples within an arm, 0
non-finite scores, all 59 randoms present with no extras, destinations partition every
population exactly, zero-control exact on all 218 routed rows.

The earlier attempt was terminated by host teardown mid-random-battery, declared
`QWEN3_4B_FORMAL_VOID` / `INCOMPLETE_ALL_ARM_EXECUTION` before any endpoint existed,
preserved in `research_exploration/qwen3_4b_formal_gate_v1/void_attempt_1/`, and **not
resumed** — every arm was re-executed from scratch under explicit human authorisation.

## Claim lock — travels with any use of this result

> Passing the frozen formal collateral endpoint must not be interpreted as demonstrated
> deployment safety.

Report collateral on both denominators: frozen 1/214 = 0.0047, eligible-at-risk 1/50 = 0.0200.

And: the frozen score-space comparator **outperformed** the activation intervention here
(Target Gain 0.1048 vs 0.0806). Do not claim activation provides behaviour unavailable to a
simple frozen score-space shift.
