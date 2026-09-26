# SAKIKO general API

What a researcher supplies:

```python
actions: list[str]                                  # A_D, |A_D| >= 3
gold_action(sample) -> str
model_action_scores(sample) -> dict[str, float]     # score per action
hidden_representation(sample, site) -> np.ndarray
intervention(sample, candidate) -> dict[str, float] # post-intervention scores
is_exposed(sample, candidate) -> bool               # REQUIRED, surface-specific
```

What SAKIKO returns:

```python
{
 "channels":      [(g, s, train_err, train_ref, dev_err, dev_ref, adjudicable)],
 "readability":   {channel: (auc, ci, tau, exposure)},
 "specificity":   {channel: (p_add_one, reverse_reproduces, wrong_site)},
 "destination":   {channel: (source_retained, gold_arrival, other_wrong, target_hit, yield)},
 "preservation":  {channel: (broken, exposed, rate, ci)},   # exposure-conditional
 "evidence":      {channel: (interval or risk bound at delta)},
 "stop_stage":    "ADJUDICABILITY | READABILITY | SPECIFICITY | DESTINATION | PRESERVATION | EVIDENCE | NONE",
 "next_action":   str,
}
```

The API is coherent for any `K`-class action space and any intervention surface, because
`is_exposed` is supplied rather than assumed. **`stop_stage` and `next_action` are the outputs
that make this a diagnostic protocol rather than a binary judge** — and they are exactly what
the previous audit found the binary ADMIT/DECLINE was collapsing.
