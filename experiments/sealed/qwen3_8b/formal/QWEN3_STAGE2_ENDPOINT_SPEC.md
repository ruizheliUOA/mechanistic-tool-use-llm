# Endpoint specification

## Formal definitions

    source_exits    = count(baseline pred = source AND intervention pred != source)
    gold_arrivals   = count(baseline pred = source AND intervention pred = gold)
    wrong_to_wrong  = count(baseline pred = source AND intervention pred != source
                                                   AND intervention pred != gold)
    target_hit      = gold_arrivals / source_exits            (when source_exits > 0)
    target_gain_count = gold_arrivals - wrong_to_wrong
    target_gain_rate  = target_gain_count / n_channel_error
    clean_collateral_rate = baseline-correct rows made incorrect / baseline-correct rows

## Primary

`target_gain_rate` on the frozen channel-error population, compared against the
K=59 fresh matched-random directions via
`p_add_one = (1 + count[random >= real]) / (K + 1)`.

## Populations

Baseline is computed once over all 548 evaluation rows **before** any arm. The
channel-error population and the safety denominator are then fixed from that
baseline and are never redefined using intervention results.

## Bootstrap

10000 resamples, deterministic seed from
`SAKIKO_QWEN3_CA_TC_STAGE2_BOOTSTRAP_V1`. Method frozen; not changed after execution.

## Raw record schema

immutable ID, gold, baseline four-mode scores, baseline prediction, Router probability and decision, arm identity, direction seed/hash, dose and achieved perturbation norm, intervention four-mode scores, intervention prediction, destination transition, batch index, global execution order.
