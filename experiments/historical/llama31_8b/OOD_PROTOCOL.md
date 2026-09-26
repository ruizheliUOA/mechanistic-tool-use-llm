# Strict OOD Protocol — feasibility established, execution GATED (not run)

**Status: written before any OOD run; NOT executed.** Per the frozen protocol, OOD may begin
only *after* the prospective Llama result is frozen and fully written **and** multiseed
confirms ≥1 channel in ≥4/5 seeds with per-seed z ≥ 2. This document records (a) that a
defensible held-out grouping exists, and (b) the exact design that would be run if — and only
if — the prerequisite is met.

## 1. Feasibility inspection (metadata only; no intervention, no test outcome)

W2C carries pre-existing grouping metadata not derived from any intervention outcome:

| field | distinct values | usable as a group? |
|---|---|---|
| `source` | **2** (`BFCL v2 Live Multiple` 2901, `BFCL v2 Live Simple` 751) | coarse but valid (2-way) |
| `target_tool` | 515 (1295 `None` = the `cannot_answer` gold set) | too fine; leaks the gold class |
| tool **domain** (prefix of the first tool's name) | **156** (top: Movies 239, Events 238, Buses 189, Flights 168, Media 160, Alarm 158, Payment 123, Hotels 120, Music 112, Homes 96, `get_*` 392, NO_TOOL 258) | **yes — the defensible choice** |
| `held_out_param` | 213 (2590 `None`) | a property of the RFI construction, not a domain |

**Chosen grouping (pre-registered): tool domain.** It is a genuine task/tool-family split,
exists in the source data, and is independent of any intervention result.

**Candidate held-out domain set** `{Travel, Weather, Music, Payment, Restaurants, RideSharing}`
gives adequate support on both sides:

| group | n | rfi_tc | ca_tc | ca_direct |
|---|---|---|---|---|
| SEEN | 3266 | 605 | 456 | 416 |
| **UNSEEN** | **386** | **73** | **79** | **47** |

All three channels clear the frozen support floors on the unseen side (≥30 train-equivalent
support; ≥8 eval), so **strict OOD is feasible in principle**. If a future run's admitted
channel has smaller support, the rule below applies: report OOD infeasibility rather than
construct an artificial split.

## 2. Strict OOD design (frozen; would be run only if the prerequisite is met)

1. **Learn on SEEN only:** discovery, reference pools, routers, DiffMean/PCA-1 directions,
   activation scale (`median_train‖h_inj‖`) — all estimated from SEEN-domain training rows.
2. **Select on SEEN validation only:** obs/inj layer, method, threshold, ρ — the entire frozen
   Gate-v2 Stage-1→5 pipeline, run exactly as in the pilot, using only SEEN validation rows.
3. **Freeze** the configuration and the ADMIT/REJECT decision; hash the decision file.
4. **Evaluate once on UNSEEN domains** with the full Stage-6 battery (real / reverse / 20
   randoms seed-block 1000+k / ungated / wrong-layer). **No** target-domain tuning of
   threshold, direction, layer, or ρ. One shot; no re-run after seeing the result.
5. **Report** with the same confirmation rule (z ≥ 2 ∧ n_ge/N ≤ 0.05 ∧ real > reverse ∧
   Broke ≤ Fixed/2 ∧ own residual −25%).

**Support requirement:** both SEEN and UNSEEN sides must independently satisfy the frozen
Stage-1 floors (ref-pool ≥ 80, own-val-errors ≥ 30, val/test ≥ 8). If not → **report OOD
infeasibility**; do not merge domains to manufacture support.

## 3. Claim boundary for OOD (stated in advance)

Strict OOD generalization is a **separate claim** from cross-dataset target-native adaptation
(which the archive already reports for MetaTool). A positive OOD result would license only:
*"a channel selected on seen tool domains retains direction-specific corrective utility on
unseen tool domains of the same dataset."* It would **not** license cross-dataset,
cross-architecture, or scale claims.

## 4. Execution status

**NOT RUN.** The prerequisite (multiseed confirmation of ≥1 admitted channel in ≥4/5 seeds
with z ≥ 2) is not satisfied by the Phase-8 prospective run — see
`LLAMA_PROSPECTIVE_RESULTS.md` and `PHASE8_CLAIM_BOUNDARY.md`. Running OOD without a validated
mechanism would reproduce exactly the error Phase 5 exposed (behaviourally impressive,
mechanistically uninterpretable numbers). This file exists so that the design is fixed *before*
any future run, not fitted afterwards.
