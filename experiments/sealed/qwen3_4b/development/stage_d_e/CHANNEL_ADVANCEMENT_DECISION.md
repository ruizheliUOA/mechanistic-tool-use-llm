# Channel advancement decision — Qwen3-4B × When2Call, Stage D/E

Both support-eligible channels completed the full frozen DEV ladder. Neither is hidden.

| Channel | Support | Readable | Geometry | Steerable | Destination-correct | Collateral | Decision |
|---|---|---|---|---|---|---|---|
| `cannot_answer→tool_call` | ELIGIBLE (TRAIN 436/68, DEV 220/40) | READABLE (DEV AUC 0.9237, τ=0.011564, prec 0.500) | **GEOMETRY_QUALIFIED** (9/9) | **YES** — real 0.1227 vs every random ≤0.0091; reverse 0 source exits | target_gain_rate 0.1227 [CI 0.0409, 0.2091]; target_hit 0.6484 [0.5500]; 59 gold vs 32 other-wrong | 3/455 = **0.0066** on 106 eligible exposed rows | **FORMAL_ADVANCEMENT_ELIGIBLE** |
| `cannot_answer→direct` | ELIGIBLE (TRAIN 174/68, DEV 114/40) | READABLE (DEV AUC 0.8799, τ=0.0, prec 0.864) | **GEOMETRY_QUALIFIED** (9/9) | **YES** — real 0.1754 vs every random ≤ −0.0175; reverse −0.2105 | target_gain_rate 0.1754 [CI 0.0789]; target_hit 0.7941 [0.6486]; 27 gold vs 7 other-wrong | 0/455 = 0.0000 but **eligible exposure is 0** | **DECLINE_COLLATERAL** |

Selected dose for both channels: **q = 0.25** (smallest admissible; the frozen rule takes
the smallest behaviourally sufficient dose, never the best-looking cell).

## Why `cannot_answer→direct` declines

It is not a weak channel. On every criterion that carries information it is the *stronger*
of the two: higher target hit (0.7941 vs 0.6484), higher target gain (0.1754 vs 0.1227),
cleaner exit profile (27 gold vs 7 other-wrong), and a reverse control that inverts the
sign outright (−0.2105, with all 24 reverse-induced exits landing in `OTHER_WRONG` and
none in gold).

It declines on frozen condition 11, **non-vacuous collateral evaluation**. Its safety gate
carries no information:

- the channel source mode is `direct`;
- the pinned When2Call split contains **zero** gold `direct` rows in TRAIN and in DEV;
- so no row whose baseline prediction is `direct` can ever be baseline-correct;
- so the routed-and-baseline-correct population is **0**, and the observed collateral rate
  of 0.0000 is a structural identity, not a measurement.

A channel cannot demonstrate safety-bounded correction when its safety denominator is
empty by construction. This is the same exclusion the frozen Qwen3-8B program applied to
its own `cannot_answer→direct` channel
(`QWEN3_STAGE2_CHANNEL_SELECTION_DISCLOSURE.md`, criterion 5), reached here independently
from Qwen3-4B's own split and not copied.

The channel is recorded as `DECLINE_COLLATERAL`, not as a negative result. Its
steerability evidence is real and is preserved in full.

## Direction-specificity, kept distinct

The three concepts are not collapsed:

- **Readable** — the Router identifies the error state. Established at Stage C for both
  channels and unchanged here.
- **Steerable** — the specific primary direction changes behaviour beyond matched
  controls. Established for **both** channels: real target gain exceeds all eight
  matched-random directions at the identical perturbation norm (rank 1 of 9, `p_add_one`
  not yet applicable — this is the DEV development budget, not the formal null), the
  reverse direction fails to reproduce it, the zero arm moves nothing, and the wrong-layer
  arm at the same absolute norm produces 15 exits with target gain 0.0045 against the real
  0.1227.
- **Correctable-candidate** — direction-specific change preferentially reaches gold with
  acceptable wrong-to-wrong and collateral behaviour. Established for
  `cannot_answer→tool_call` only, because `cannot_answer→direct` has no evaluable
  collateral surface.

Steerability without an evaluable safety surface is not correctability, and this decision
does not upgrade it.

## Estimator dependence

At the selected dose, both DiffMean constructions are near-inert on this channel:

| arm | source exits | gold | other-wrong | target gain |
|---|---:|---:|---:|---:|
| `d_grad` | 91 | 59 | 32 | **+0.1227** |
| L26 DiffMean (mandatory comparator) | 2 | 1 | 1 | 0.0000 |
| L21 same-layer DiffMean (diagnostic) | 1 | 1 | 0 | +0.0045 |
| wrong-layer d_grad at L26 | 15 | 8 | 7 | +0.0045 |
| reverse −d_grad | 0 | 0 | 0 | 0.0000 |
| zero | 0 | 0 | 0 | 0.0000 |
| frozen score-space bias comparator | 117 | 69 | 48 | +0.0955 |

Report this as **estimator dependence** on this channel at this dose. It is not a claim
that DiffMean is universally invalid.

The score-space comparator is the one that comes closest. It reaches more gold arrivals
(69 vs 59) but pays for them with substantially more wrong-to-wrong traffic (48 vs 32) and
more collateral (10 vs 3), so its target gain is lower (0.0955 vs 0.1227). The activation
intervention beats it, but not by a wide margin, and the formal claim must be worded to
say only that d_grad exceeds a simple frozen two-mode score shift — not that a score shift
cannot move this channel at all.

## Not reduced to Net

Net is reported (+69 for the advancing channel, +31 for the declining one) but is not the
decision variable. Advancement is conjunctive and lexicographic over the frozen
fourteen-condition list; no weighted actionability score was constructed.
