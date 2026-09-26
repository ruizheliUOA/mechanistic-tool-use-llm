# Phase-5 Figures — captions & provenance

Only claim-bearing figures are included.

## fig1_specificity_qwen_vs_mistral.{png,pdf}

**Caption.** *Direction-specificity does not transfer to a second architecture, even as raw
Net rises.* **(A) Cascade-level placebo, matched protocol (n_random = 10), locked test
n = 548.** Grey = matched-norm random directions; green ★ = the real learned cascade; red ▼ =
reverse; dashed = random mean. On **Qwen2.5-7B** (archived) the real cascade (+79) lies far
above its entire random distribution (mean 25.2; **0/10 random ≥ real**; margin **+53.8**) and
reverse collapses to +16 (20% retention). On **Mistral-7B-v0.3** the real cascade scores
*higher* in absolute terms (+82) yet lies **inside** its random distribution (mean 70.1;
**1/10 random ≥ real**, max random **+119 > real**; margin only **+11.9**), and reverse retains
**88%** (+72). **(B) Mistral per-channel placebo (n_random = 20).** Only `ca_tc` (real +62)
clears its random cloud (95th percentile, 1/20 ≥ real); `rfi_tc`'s real (+12) falls **below**
its random mean (+29.1, 17/20 ≥ real) — worse than noise; `ca_direct` (+10) sits inside its
cloud with reverse (+12) above it. Conclusion: on a `tool_call`-saturated model, most of the
behavioral gain is **gated magnitude**, not learned direction — a Net-only evaluation would
have reported Mistral as the better result.

**Data provenance.**
- Qwen: `final/results/7b_w2c_sakiko/qwen25_7b_placebo_controls.json` (archived, committed).
- Mistral cascade: `final/results/mistral7b_w2c_sakiko_ca/placebos/cascade_placebo_armF.json`.
- Mistral per-channel: `final/results/mistral7b_w2c_sakiko_ca/placebos/placebo_all_controls.json`.
- Generator: `scripts/phase5_mistral_cascade_placebo.py` (control) + figure script (plot only).

**Claim supported:** PHASE5_CLAIM_BOUNDARY.md rows **B1, B3, C1, C2, D1, D1b**.
**Claim NOT supported by this figure:** any multi-seed statement (multi-seed was not run).
