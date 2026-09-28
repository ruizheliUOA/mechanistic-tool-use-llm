# Formal channel selection disclosure

The single confirmatory channel is **`cannot_answer → tool_call`**.

## Why this channel

1. passed the frozen V2 support gate;
2. passed the frozen Router gate;
3. passed the Stage 1.5 geometric advancement gate;
4. passed the Stage 2A behavioural dose gate;
5. **has non-vacuous collateral evaluation** — 368 DEV rows are both
   routed-eligible and baseline-correct, so its collateral bound is a real
   constraint;
6. d_grad materially exceeded the frozen score-space comparator on DEV
   (target gain +0.3114 versus +0.1018);
7. original DiffMean, reverse and wrong-layer controls did not reproduce the DEV
   result;
8. selecting one channel avoids an under-resolved multi-channel random null.

## The disclosure that matters

**This channel was selected using TRAIN/DEV developmental evidence while the
formal evaluation partition remained sealed.** It was not an untouched choice
among all possible channels. Any future claim must carry that qualification.

Two of the three eligible channels were deliberately excluded from the formal
primary analysis:

- `request_for_info → tool_call` — on DEV its frozen score-space comparator
  *outperformed* d_grad on target gain (+0.1844 versus +0.1311), so it is the
  weakest case for a claim that the activation intervention does something a
  simple two-mode score shift cannot;
- `cannot_answer → direct` — its clean collateral rate is **structurally** zero,
  because the source mode is `direct` and the pinned split contains no gold
  `direct` rows, so no routed row can ever be baseline-correct. Its safety gate
  carries no information.

Neither is added to the formal primary analysis. Reporting them post hoc as
confirmatory would be selection after the fact.
