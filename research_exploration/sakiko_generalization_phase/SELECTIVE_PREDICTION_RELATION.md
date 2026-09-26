# Relation to selective prediction / conformal risk control

## The scientific object is different, and the difference is precise

| | selective prediction / CRC | SAKIKO |
|---|---|---|
| what is selected | which **outputs** to emit or abstain on | whether an **intervention** may be called a correction |
| the null | none required — risk is bounded distribution-free | **matched-random and reverse interventions** |
| the risk | error rate on retained predictions | destination misdirection + exposure-conditional damage |
| what is abstained from | answering | **making a scientific claim** |
| causal intervention specificity | **not evaluated** | central |
| source→gold vs source→other-wrong | **not distinguished** | central |
| channel discovery | roles predefined | automatic over all `K(K−1)` transitions |

**SAKIKO is not selective prediction applied to interventions.** Selective prediction withholds
outputs; SAKIKO withholds a *claim about a causal manipulation*. The object being certified is a
counterfactual property of an edit, not the correctness of a prediction — which is why the null
must be another intervention rather than a calibration set.

## But the honest direction of borrowing runs the other way

The CRC literature has finite-sample distribution-free guarantees. SAKIKO has bootstrap
intervals against conventional thresholds. **The strongest single improvement available to this
project is to re-express constructs D (preservation) and C (destination) as conformal
risk-controlled quantities**, giving `P(correct→wrong | exposed) ≤ α` with a finite-sample
guarantee instead of a Clopper–Pearson interval against a 0.05 convention.

That is not a small reframing — it would replace the least defensible part of the framework
(threshold provenance) with the most defensible machinery available, and it is **exactly the
part the hostile panel's Reviewer C attacked**.

It is also **not free**: CRC needs an exchangeable calibration population, and SAKIKO's exposed-
correct populations were 0, 6, 11 and 23 rows. The impossibility result in 2606.29054 says that
below a support threshold, meaningful certification forces heavy abstention. SAKIKO's high
refusal rate may therefore be **structurally necessary**, not a tuning artifact.
