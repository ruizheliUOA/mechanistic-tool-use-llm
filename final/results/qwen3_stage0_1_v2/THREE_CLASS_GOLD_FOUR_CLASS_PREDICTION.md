# Three-class gold against four-class prediction

## Prediction space

The frozen readout scores four complete candidates per sample, in the frozen
mode order read from the committed V1 protocol `readout.mode_order`:

`tool_call`, `direct`, `request_for_info`, `cannot_answer`

Every one of the 3104 authorized non-test rows carries all four
candidate answers; this was asserted when the selective byte index was built.

## Gold space

Gold labels actually present in the pinned population
(`nvidia/When2Call` revision
`0582f7749df63a96fdc3070932e83e72396ace53`, config `test`,
split `mcq`):

| gold mode | TRAIN | DEV | total |
|---|---:|---:|---:|
| tool_call | 898 | 203 | 1101 |
| direct | 0 | 0 | 0 |
| request_for_info | 745 | 161 | 906 |
| cannot_answer | 913 | 184 | 1097 |

**Gold `direct` has zero rows.** The gold
space is three-class while the prediction space is four-class.

This was verified independently, not merely observed: the historical Qwen2.5
baseline over the identical 3104 authorized rows reproduces the
same three-way gold distribution exactly, which cross-validates both the byte
index and the gold-label extraction.

## Consequences

1. **`direct` remains a valid predicted destination.** It is predicted on
   418 TRAIN and 99 DEV rows. It is
   a real behavioural outcome of the model, not an unreachable label.

2. **`direct`-related wrong-to-wrong flows remain scientifically relevant.**
   Movement of a routed source error *into* or *out of* `direct` is a
   meaningful redistribution and would be measured by any future Stage 2
   endpoint that counts destinations, since destination selectivity is defined
   over arrivals rather than over gold classes.

3. **All `direct -> *` source-error channels are structurally empty.** A channel
   `g -> p` requires gold `g`. With no gold `direct` rows, all such channels
   have zero error support and zero reference support by construction, not by
   model behaviour:

| channel | train error | train ref | dev error | dev ref |
|---|---:|---:|---:|---:|
| direct__to__tool_call | 0 | 0 | 0 | 0 |
| direct__to__request_for_info | 0 | 0 | 0 | 0 |
| direct__to__cannot_answer | 0 | 0 | 0 | 0 |

These 3 transitions can never pass any support gate on this
dataset regardless of model, accuracy or allocation. They are retained in the
complete ledger with their exclusion codes rather than silently dropped.

4. **The effective candidate channel count is 9, not 12**, for
   any analysis on this pinned population. This is a property of the dataset,
   fixed before any Qwen3 measurement, and it is not an amendment.
