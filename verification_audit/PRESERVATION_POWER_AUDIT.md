# Preservation power audit — is the ADMIT's preservation arm vacuous?

**No. But a reported interval is wrong and must be corrected.**

## The premise, and why it is wrong

The concern: if the ADMIT's exposed-correct population was ~6–50 rows, then
"no correct sample was broken" certifies approximately nothing.

**The ADMIT's conditions 7 and 8 were adjudicated on E1 — all baseline-correct
rows, n = 211 — not on E2 (n = 6).** E2 is the *retrospective* exposure-conditional
re-analysis performed later in this project; it was never the licensing endpoint.

## Design sensitivity at 80% power

Smallest true break rate detectable, i.e. P(≥1 break) ≥ 0.80:

| exposed-correct n | min detectable rate | what it is |
|---:|---:|---|
| 6 | **0.2353** | E2 for the ADMIT — **vacuous** |
| 11 | 0.1361 | Gemma E2 |
| 50 | 0.0317 | Qwen3-4B E2 |
| 93 | 0.0172 | V2 design target |
| **211** | **0.0076** | **E1 — what the ADMIT actually used** |

**At n = 211 the study could detect a true break rate of 0.76%, against a gate
set at 5%.** That is roughly 6.6x more sensitive than the criterion requires. The
preservation arm of the ADMIT is well-powered, not vacuous.

The E2 figure (0/6, detectable only above 23.5%) **is** vacuous — and that is
exactly why `PRESERVATION_STATUS_FINAL.md` already reports it as *not*
certifying, and why the manuscript does not claim exposure-conditional
preservation for any setting.

Both statements are true and answer different questions:

- **E1, n=211:** the intervention did not measurably damage the baseline-correct
  population. Well-powered. **This is what the ADMIT licensed.**
- **E2, n=6:** whether the intervention damages the rows the Router actually
  fires on is **unresolved**, and no re-analysis can resolve it.

## The real defect this surfaced

```
frozen  clean_collateral_rate_ci95 = [0.0, 0.0]
```

**This interval is degenerate.** A bootstrap over 0 breaks in 211 draws returns
zero in every resample, so the procedure reports [0.0, 0.0] regardless of
uncertainty. It is not a valid upper bound.

The correct one-sided 95% Clopper–Pearson upper for 0/211 is **0.014097**.

**Verdict impact: none** — both 0.0 and 0.0141 clear the 0.05 gate, so condition 8
passes either way. **Reporting impact: real** — [0.0, 0.0] must never be quoted in
the paper. It reads as "zero risk with certainty", which is false, and a reviewer
recomputing it will flag it immediately.

**Required manuscript correction:** wherever the ADMIT's collateral interval
appears, report **0/211, one-sided 95% upper 0.0141 (Clopper–Pearson)**, and state
that the bootstrap is degenerate at zero events. This is the same class of defect
as the earlier CP-vs-bootstrap confusion, in the opposite direction: the frozen
pipeline used a bootstrap where an exact method was required.

## Net effect on the paper

**Strengthens it.** The preservation claim is better supported than the challenge
assumed — 0.76% detectable against a 5% gate — and the one genuinely vacuous
number (0/6) is already disclosed as non-certifying. What changes is one
interval, from an impossible [0.0, 0.0] to a defensible 0.0141.
