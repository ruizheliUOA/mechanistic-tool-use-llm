# SAKIKO Figure 1 — authoritative caption

*Detailed enough that the method survives even if the graphic is simplified.*

> **Figure 1: SAKIKO separates a behavioural change from a licensed correction
> claim.** A frozen model makes a pre-execution decision over a multiclass action
> set `A_D` with `|A_D| = K >= 3`; when it errs, the error is *directed*, written
> `c = (g -> s)` for gold action `g` and the specific wrong action `s` the
> baseline emits. **Offline**, channels are discovered from the model's own error
> topology rather than assumed, and a supported channel yields both a per-channel
> logistic-regression router `r_c` and a channel-specific direction `d_c`; there
> is no universal steering vector. **Online**, the parameters are frozen
> throughout. Because the injection site precedes the observation site in
> execution order (`L_inj = 21 < L_obs = 26`, zero-based, on a 36-layer
> backbone), diagnosis and intervention occupy two separate forward passes over
> the same input. Pass 1 caches the MLP output at `L_obs` for the final prompt
> token; the router scores that cached activation offline in NumPy, so no second
> network is evaluated inside the model graph. If `p_c >= tau`, pass 2 re-runs the
> same input while adding `h' = h + alpha d_c` at `L_inj`, and the remaining
> frozen layers produce the action; otherwise the forward pass is left untouched —
> no prompt edit, no weight update, no rewriting of the emitted answer. Each
> channel error then resolves into exactly one of three mutually exclusive
> destinations: `SOURCE_RETAINED` (`a' = s`, the error persists), `GOLD_ARRIVAL`
> (`a' = g`, genuine repair), or `OTHER_WRONG` (`a' = w` for some other wrong
> action, the error redistributed), with `source exit = gold arrival +
> other-wrong`. Conventional accuracy counts only `GOLD_ARRIVAL` as repaired and
> collapses `SOURCE_RETAINED` with `OTHER_WRONG` as still wrong, so it cannot
> distinguish an error that persists from one that has merely moved;
> destination-resolved accounting separates all three. Collateral risk is measured
> on a **disjoint** population of baseline-correct, router-exposed rows and is
> never netted against repairs. Finally, the claim structure: adjudicability is an
> entry condition on the study, not a property of the model; `Readable ->
> Steerable -> Correctable` are property claims about the tested setting and
> intervention; and licensability is a distinct object — an evidential decision,
> qualified by specificity, destination, preservation, yield and evidence
> precision, about whether the available evidence justifies asserting the
> correction claim. A DECLINE reports the state of that evidence and does not
> assert that the setting is intrinsically uncorrectable.

## Notes for the editor

- ~300 words. If the page budget bites, cut in this order: (1) the "otherwise the
  forward pass is left untouched" clause; (2) the collateral sentence; (3) the
  "there is no universal steering vector" clause.
- The two-pass sentence and the correctness-collapse sentence are **not**
  cuttable. They are the two things reviewers most reliably get wrong.
- Never restore the withdrawn sentence *"Aggregate metrics score SOURCE_RETAINED
  against the other two."* It is backwards.
