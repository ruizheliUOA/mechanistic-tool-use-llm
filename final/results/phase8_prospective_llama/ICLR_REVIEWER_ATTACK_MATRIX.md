# ICLR Reviewer-Attack Matrix

Each row: the exact claim, the strongest skeptical alternative, prior evidence, what Phase 8
adds, what remains unresolved, and where the claim belongs.

---

## 1. "The Gate was designed after seeing the answers."

**Claim:** Gate v2's decisions on Llama are prospective, not retrofitted.
**Strongest attack:** every threshold was chosen on Phi/Qwen/Mistral *after* their outcomes
were known; the "prospective" run is theatre.
**Prior evidence:** Phase-6 dev/prospective split; Phase-7 provenance ledger labelling every
threshold (inherited / Phase-5-motivated / Phase-7-dev-selected).
**Phase 8 adds:** `GATE_V2_LOCK_MANIFEST.json` verified **15/15 artifacts unchanged** at HEAD
`1ff1f6f` *before* Llama was touched; `PHASE8_PRE_RESULT_PREDICTIONS.json` +
`PHASE8_EXECUTION_MANIFEST.md` hashed before any model output; per-channel fragility
predictions hashed **before** activation extraction; decision file hashed (`45a3f8a7…`) and
hash-verified before the test driver would run (it *refuses* otherwise — self-tested).
**Unresolved:** the thresholds are still dev-fitted; only their *application* was prospective.
The Stage-1 power floor (≥30) has single-channel provenance (tc_rfi) and is disclosed as such.
**Placement:** **main paper** (the freeze/hash chain is a core methodological contribution);
provenance ledger in appendix.

## 2. "Positive Net is only generic perturbation."

**Claim:** Net without matched-norm controls is not evidence of mechanism.
**Attack:** you report Nets of +16…+23; those are just perturbation effects.
**Prior:** Mistral Phase 5 (+82 cascade, random max +119); Phase-7 ρ study.
**Phase 8 adds:** *we make the attack ourselves and act on it.* Llama's largest effects
(+18/+23/+23 val; +16/+19 test) were **all rejected**. At ρ=8 rfi_tc's **reverse equals real
(+18/+18)** and **random mean (24.1) exceeds real**. `ca_direct`'s **wrong-layer control (+61)
is 3× its real effect (+19)**. We never report a Net as a result.
**Unresolved:** nothing — this is our own finding, replicated on a third architecture.
**Placement:** **main paper** (headline: Net is not evidence).

## 3. "The result is a Qwen-specific accident."

**Claim:** actionability is architecture-specific; only Qwen ca_rfi is test-confirmed.
**Attack:** n=1 positive architecture; the method may simply not work.
**Prior:** Mistral A-global (Phase 7).
**Phase 8 adds:** a third architecture, independently rejected. This *strengthens* the
skeptic: 2 of 3 intervention-bearing architectures yield **zero** actionable channels. We
state this plainly rather than hiding it.
**Unresolved:** whether *any* second architecture has an actionable channel is **open**. The
honest reading is that Qwen ca_rfi may be the exception, not the rule.
**Placement:** **main paper + limitations** (this is the central limitation of the programme).

## 4. "High AUC already explains everything."

**Claim:** detectability is orthogonal to actionability.
**Attack:** your routers are just good/bad classifiers; AUC is the real variable.
**Prior:** tc_rfi AUC 0.849 → test-null; Mistral ca_direct AUC 0.938 → non-specific;
router–steering cos ≈ 0 universally (Phase 7).
**Phase 8 adds:** Llama AUCs **0.837 / 0.959 / 0.937** — all comfortably above the floor, all
**rejected**. The highest-AUC Llama channel (ca_tc, 0.959) is also the one whose apparent
specificity evaporated at test. AUC now spans 0.837–0.964 across confirmed-actionable and
confirmed-non-actionable channels **with no separation**.
**Unresolved:** none — refuted on 4 architectures.
**Placement:** **main paper** (fig3 detectability-vs-actionability).

## 5. "Random controls are too few."

**Claim:** 20 matched-norm randoms suffice to reject; z and the non-parametric fraction agree.
**Attack:** n=20 gives a coarse 5% resolution; and the archived Qwen multiseed used n=1 on
3/5 seeds.
**Prior:** the n=1 weakness is one we *surfaced ourselves* in the Phase-6 audit.
**Phase 8 adds:** **n=20 at every ρ on validation and n=20 at test**, paired across ρ (same 20
unit vectors), seed blocks disjoint (val 2000+k, test 1000+k) and recorded. Rejections used
both parametric (z) and non-parametric (n≥real) criteria, which **agreed in every case**.
**Unresolved:** n=20 bounds resolution at 5%; a borderline channel (z≈2) would need more. None
of ours was borderline at test (z = 0.879, 1.256).
**Placement:** appendix (method), with the n=1 archived caveat stated in the main text.

## 6. "The method only suppresses tool calls."

**Claim:** the framework targets response-mode choice, not tool-call suppression.
**Attack:** every channel is `*→tool_call`; you are just adding an anti-tool bias.
**Prior:** ca_direct (`cannot_answer→direct`) is not a tool-call channel; the archived global
tool_call-penalty control gave Net +3 vs SAKIKO's +79.
**Phase 8 adds:** Llama's `ca_direct` (from_pred = `direct`, no tool call involved) behaves
identically to the tool-call channels — boundary-seeking, non-specific, wrong-layer +61. So
the failure mode is **not** about tool calls; it is about magnitude vs direction. Conversely
`ca_tc`@ρ=4 shows a direction that *specifically* pushes off tool_call (z=5.46) yet lands on
the wrong label — suppression is real but is **not correction**.
**Unresolved:** the archived penalty-control is single-model.
**Placement:** appendix (with the ca_direct evidence promoted to main text if space).

## 7. "The test split influenced configuration selection."

**Claim:** no Phase-8 configuration was selected using test data.
**Attack:** you computed baseline predictions on all 3652 rows, which includes test.
**Phase 8 answer (explicit):** baseline predictions on the full set are the **R0 requirement**
and firewall rule #5's own first precondition. Everything that *selects* — router fit,
direction estimation, activation scale, obs/inj layer, method, threshold, ρ — used
**train+val only**, enforced in code by `assert_no_test()` on every intervention index set
(unit-tested: a single test index raises). Activations were extracted for **train+val only**
(3104 rows). The test driver **refuses to run** unless the decision file exists *and* its
SHA256 matches. Discovery consumed test *support counts* (how many test rows fall in each
transition) — a pre-registered part of the frozen Stage-1 filter, never an outcome.
**Unresolved:** a purist could object to test support counts in Stage 1. Documented in the
integrity audit; the alternative (blind support) was not the frozen rule.
**Placement:** appendix (reproducibility), with the firewall design in the main method.

## 8. "Negative Mistral/Llama results only reflect the wrong layer."

**Claim:** the negatives are not an artifact of one arbitrary injection site.
**Attack:** you inject at one MLP output; the writable representation may be elsewhere.
**Prior:** Phase-7 named this **H2** as the single live, untested hypothesis.
**Phase 8 adds:** partial coverage only — the frozen grid searched obs ∈ {13,18,22,27}
(4 depths) × inj ∈ {obs, −2, −4} (12 layer-channel combinations), and a 32× ρ range at each.
All three Llama channels selected the **same** depth (0.688) as Mistral, and all failed.
**Unresolved — and I will not overstate this:** the search covers **MLP-output sites only**.
Residual-stream and attention-output sites are **untested on any architecture**. This remains
the strongest open alternative to "no linear correction direction exists".
**Placement:** **limitations (main text)** — a reviewer will ask, and the honest answer is
"not tested".

## 9. "The framework has limited practical utility if many channels are rejected."

**Claim:** a reliable rejector has real value — it saves the test budget and prevents false
mechanistic claims.
**Attack:** a method that admits nothing on 2 of 3 architectures is not a method.
**Phase 8 adds:** the gate's rejections would have **saved the entire Phase-5 Mistral test
budget** and prevented the +82/"bigger number, worse mechanism" trap; on Llama it correctly
refused three channels whose raw Nets (+16…+23) would otherwise have been reported as wins.
**Unresolved — concede this:** as a *corrective tool* the framework currently has **one**
demonstrated actionable channel across four architectures. The contribution is therefore
**evaluative/diagnostic**, not a deployable intervention method. We should say exactly that.
**Placement:** **limitations + discussion** (reframe the contribution honestly).

## 10. "The contribution is only another steering vector."

**Claim:** the contribution is the *actionability distinction* and its measurement protocol,
not a steering direction.
**Attack:** DiffMean steering + an LR probe is standard; you have added hyperparameters.
**Phase 8 adds:** what is new is not the vector but the **decision procedure and its
prospective test**: (i) ρ, an architecture-normalized magnitude (identical α maps to ρ
spanning 0.84–2.06× across channels — raw α is not a physical quantity); (ii) the
**interior-vs-boundary** response signature; (iii) matched-norm randoms as a *selection-time
rejector*; (iv) a hash-chained freeze → prospective decision → bounded adversarial audit
protocol; (v) a pre-intervention statistic (fragility) that made a **correct falsifiable call
on an unseen architecture, 2/2**.
**Unresolved:** the admission half is unvalidated, so we cannot claim a *predictive* gate.
**Placement:** **main paper** (framing), with (v) as the most novel and most fragile claim
(n=2 audited channels — explicitly not a statistic).

---

## Cross-cutting weakness a reviewer will find first

**The programme has one test-confirmed actionable channel (Qwen ca_rfi) across four
architectures and ~8 controlled channels.** Everything else is a rejection. The paper must
lead with the *evaluative* contribution ("Net and AUC do not establish mechanism; here is a
protocol that says so prospectively") and must **not** be framed as a working intervention
method. Any framing that implies broad corrective capability will not survive review — and
would not deserve to.
