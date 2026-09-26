# Gemma-2-9B-it — terminal DEV verdict

**DEV-stage only. SEALED never accessed. No formal ADMIT or DECLINE exists.**

| channel | terminal verdict | failed layer |
|---|---|---|
| `request_for_info → tool_call` | `READABILITY_ROUTER_DECLINE` | readability — DEV ROC-AUC 0.7067, CI [0.6666, 0.7456] entirely below the 0.75 gate |
| `cannot_answer → tool_call` | **`FORMAL_ADVANCEMENT_ELIGIBLE` (14/14)** | none |
| `cannot_answer → direct` | `DECLINE_COLLATERAL` (13/14) | condition 11 — `non_vacuous_collateral_evaluation`, 0 exposed baseline-correct rows |

## The advancing channel

`cannot_answer → tool_call` @ q = 0.125 (min admissible; the max-Net dose would have been 2.0).
193 routed channel errors: 142 SOURCE_RETAINED, 51 exits, **34 GOLD_ARRIVAL**, 17 OTHER_WRONG.
target-hit 0.6667, Target Gain **+0.0881** CI [+0.016, +0.161], Fixed 34 / Broke 0 / Net +34.
Collateral 0/471 all-correct, 0/23 exposed-correct.

Specificity is unusually clean: **none of the 8 matched-random directions produced a single
gold arrival**, reverse produced zero exits, and zero reproduced baseline exactly.

## The declining channel, and why it matters

`cannot_answer → direct` has the **better** destination evidence — target-hit 0.8636, Target
Gain +0.1860 CI [+0.093, +0.291], 19 gold against 3 other-wrong — and passes 13 of 14
conditions including every control. It declines because the Router fires on **zero**
baseline-correct rows, so its 0/471 collateral could not have failed.

**Its 0/471 must never be described as safety.** It is a structurally vacuous test. This is the
second time the frozen protocol has refused a channel on this exact structure; Qwen3-4B's
`cannot_answer → direct` declined the same way.

## Standing caution on the advancing channel

The collateral pass rests on **23 exposed-correct rows**. That is better than the Qwen3-8B
ADMIT's 6, which the final audit judged an exposure artifact, but it remains thin, and sealed
exposure is likely to be smaller still. Deployment safety may not be inferred from it.
