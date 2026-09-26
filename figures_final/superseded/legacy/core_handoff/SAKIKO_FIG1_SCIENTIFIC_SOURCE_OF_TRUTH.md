# SAKIKO Figure 1 — scientific source of truth

**This document is authoritative for content. A designer may change every visual
choice; nothing here may be changed without a scientist's sign-off.**

---

## 1. Problem definition

A frozen language model must make a **pre-execution decision**: given a user
request and a tool context, which of several response modes should it take —
before any tool is actually called. The decision space is a discrete multiclass
action set `A_D` with `|A_D| = K ≥ 3`. In the tested setting:

```
A_D = { tool_call, request_for_info, cannot_answer, direct_answer }
```

These four are an *example* of a dataset-native multiclass action space, not a
SAKIKO-specific ontology. Do not present them as part of the framework.

When `K ≥ 3`, a wrong answer can be wrong in more than one way. That is the entire
reason this work exists.

## 2. The directed error channel

The unit of analysis is a **directed channel**:

```
c = (g → s)
```

- `g` — the gold action
- `s` — the specific wrong action the frozen baseline actually emits

A channel is not "the model gets this wrong". It is "the model gets this wrong
*in this particular direction*". `(tool_call → cannot_answer)` and
`(cannot_answer → tool_call)` are different channels with different geometry.

`|A_D| ≥ 3` is required: with only two actions, leaving `s` is identical to
arriving at `g`, and the central distinction collapses.

## 3. Offline channel discovery

Performed once, before any intervention, from the model's own behaviour:

```
baseline predictions + gold
  → directed error topology (which g→s transitions actually occur, and how often)
  → candidate channels
  → support gate  →  supported channel c
```

Channels are **discovered, not assumed**. No channel is chosen because it is
convenient or historically favourable. Support gates (TRAIN ≥ 60 errors and
≥ 60 references; DEV ≥ 30 / ≥ 30) are applied before selection.

## 4. Router construction

For the supported channel, a lightweight per-channel detector is fit on TRAIN:

- Input: `h_obs`, the hidden state at the observation site
- Model: `StandardScaler` → `LogisticRegression(C=1.0, L2, liblinear, max_iter=2000, tol=1e-4, seed=42)`
- Output: `p_c` — probability that this row belongs to channel `c`
- Threshold `τ` chosen from a frozen grid as the smallest value achieving DEV
  source-matched precision ≥ 0.50
- Eligibility: DEV ROC-AUC ≥ 0.75 **and** τ-precision ≥ 0.50

The router is **one logistic regression**, not a second neural network. This
matters for the cost claim and must not be drawn as a model.

## 5. Direction / representation estimation

A channel-specific candidate direction `d_c` is estimated on TRAIN only, from
gradients of the gold-vs-source candidate scores at the injection site:

```
d_grad = unit( mean_i unit( G_i,gold − G_i,source ) )
```

`d_c` is **specific to channel `c`**. There is no universal steering vector. Two
channels on the same model yield different directions.

## 6. Frozen transformer topology

`θ_LLM` is frozen throughout. No weight is updated, ever. The prompt is not
edited. The emitted answer is not rewritten. The only thing that changes is one
hidden state, conditionally.

Verified sites (36-layer backbone, zero-based indexing):

- **Observation:** `layers[26].mlp` output, final prompt token
- **Injection:** `layers[21].mlp` output

**`L_inj = 21` executes BEFORE `L_obs = 26`.** See §7–8 and
`SAKIKO_FIG1_TOPOLOGY_LOCK.md`. This is load-bearing.

## 7. Pass 1 — diagnose

```
input → frozen model → forward to completion
      → cache h_obs (MLP output at layer 26, final prompt token)
```

No intervention occurs in pass 1. The activation is **cached to disk**.

The router then scores the cached activation **offline, in NumPy**. It is not
part of the model graph and does not run inside a forward pass.

## 8. Pass 2 — intervene

Only if `p_c ≥ τ`:

```
same input → frozen model
           → at layer 21: h' = h + α·d_c
           → remaining frozen layers 22…35
           → action readout
```

If `p_c < τ`, pass 2 does not happen at all and the baseline output stands.

## 9. Intervention site and dose

Additive, constant, at the MLP output of layer 21. Dose `α = q · s_c`, where
`s_c` is the median activation norm over TRAIN router-active rows and `q` comes
from a frozen grid `{0, 0.125, 0.25, 0.5, 1.0, 2.0}`; the selected `q` is the
minimum admissible value.

## 10. Final action readout

The four candidate continuations are scored under the intervened forward pass and
the argmax is taken. The intervention therefore returns to the **actual
multiclass decision space** — it is not evaluated in a latent space.

## 11. Destination-resolved outcomes

For each baseline error on channel `c = (g → s)`, the post-intervention action
`â′` falls into exactly one of three mutually exclusive, exhaustive classes:

| Outcome | Condition | Meaning |
|---|---|---|
| `SOURCE_RETAINED` | `â′ = s` | the original error persists |
| `GOLD_ARRIVAL` | `â′ = g` | genuine repair |
| `OTHER_WRONG` | `â′ = w ∈ A_D \ {g,s}` | the error is redistributed to a different wrong action |

Identity: **`source exit = gold arrival + other-wrong`**.

**The correctness collapse, stated correctly.** Under conventional accuracy,
`fixed = gold_arrivals`. For a baseline error, `GOLD_ARRIVAL` becomes correct,
while `SOURCE_RETAINED` **and** `OTHER_WRONG` both remain wrong. Aggregate
metrics collapse **{SOURCE_RETAINED, OTHER_WRONG}**; they cannot distinguish an
error that persists from one that has merely moved. Destination-resolved
accounting separates all three.

Verified on the Qwen3-8B primary channel: `source_exits = 52`,
`gold_arrivals = 38`, `wrong_to_wrong = 14`, and `52 = 38 + 14`.

## 12. Second population — preservation

A **disjoint** population: rows the baseline already gets right, on which the
router nevertheless fires.

```
baseline-correct ∩ router-exposed  →  CORRECT_RETAINED  or  BROKEN
```

Collateral damage is a separate estimand on a separate population. It is
**never netted against repairs**. Two estimands are distinguished: E1 (frozen,
all baseline-correct rows) and E2 (exposure-conditional, router-fired ∩
baseline-correct). E2 is the deployment-relevant one and has a much smaller
denominator — which is why preservation is usually the binding constraint.

## 13. Causal controls

The real direction is compared against a frozen control set: zero, matched-random
(K = 59 salt-derived directions), reverse, wrong layer, ungated, and a
score-space comparator. The comparator is **derived**, not an arm.

The question these answer: does the real intervention have a *direction-specific*
causal effect, or would any perturbation of similar magnitude do the same?

## 14. Readable → Steerable → Correctable

Three **property claims about the tested setting and intervention** — not about
the model as an object, and not about the framework.

- **Readable** — the channel error is linearly decodable from `h_obs`. Internal
  error information is detectable.
- **Steerable** — the intervention moves behaviour, and the movement is
  direction-specific: it survives matched controls.
- **Correctable** — the movement is *target-directed*. It lands on `g` rather
  than merely departing from `s`.

Each strictly presupposes the one before. **Readable ≠ Steerable ≠ Correctable.**
A setting can be readable and steerable while being uncorrectable — that is
exactly what "movement is not correction" means.

## 15. Adjudicable

An **entry condition on the study**, not a property of the model. Can this study
answer the question at all? Requires `|A_D| ≥ 3`, a supported channel, and
sufficient rows in every split. Without it, no rung can be evaluated — a
non-adjudicable setting yields no information about the model whatsoever.

## 16. Licensable

A **different kind of object from the three properties above.** It is an
epistemic decision about whether the accumulated evidence justifies *asserting*
the Correctable claim. It is qualified by:

specificity · destination · preservation · yield · evidence precision

**`Correctable` is a claim. `Licensable` is a judgement about that claim.** They
must never be drawn as consecutive homogeneous nodes.

## 17. ADMIT / DECLINE / NO-GO

| Verdict | Means | Does **not** mean |
|---|---|---|
| `ADMIT` | the evidence supports asserting the correction claim under the frozen conjunction | the model is safe, certified, or fixed |
| `DECLINE` | the evidence does not support the assertion | the setting is intrinsically uncorrectable |
| `NO-GO` | the study could not be adjudicated | the model failed |

A DECLINE is a statement about an evidence budget, not about a model's nature.
Settings that DECLINE frequently have *positive* Correctable point evidence and
fail only on preservation power or yield.
