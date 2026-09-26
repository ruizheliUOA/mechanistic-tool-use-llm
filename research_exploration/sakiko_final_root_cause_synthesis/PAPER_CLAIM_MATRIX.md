# Paper claim matrix

`SUPPORTED` · `QUALIFIED` (supported with stated qualification) · `HYPOTHESIS` (hypothesis-
generating only) · `NOT ESTABLISHED` · `CONTRADICTED`

| # | claim | status | primary evidence |
|---|---|---|---|
| 1 | Structured tool-decision error channels exist | **SUPPORTED** | 12-transition ledgers, 4 models; 3 support-eligible channels in Qwen3-8B/Gemma |
| 2 | Some channels are internally readable from activations | **SUPPORTED** | `ca→*` DEV ROC-AUC 0.9190–0.9387 across 2 families |
| 3 | Readable ⇒ steerable | **CONTRADICTED** | Mistral `rfi_tc` readable but real +12 < random mean +29, 17/20 randoms ≥ real |
| 4 | Steerable ⇒ correctable | **NOT ESTABLISHED** | the only candidate (Llama redistribution) failed specificity first; no licensed example |
| 5 | Some modern setting satisfies the full correction licence | **SUPPORTED** | Qwen3-8B × When2Call × `ca→tc`, 10/10, prospective one-shot |
| 6 | The Qwen family is more correctable | **NOT ESTABLISHED** | Qwen3-8B ADMIT, Qwen3-4B DECLINE, Qwen3.5 no actionable channel — non-monotonic within one family |
| 7 | Larger models are more correctable | **CONTRADICTED** | Qwen3.5-9B > Qwen3-8B in size, terminated earliest |
| 8 | Readability is channel-dependent | **QUALIFIED** | `SUGGESTIVE_CHANNEL_ASSOCIATION`; confounded with benchmark, one Llama counterexample |
| 9 | Correctability is channel-dependent | **NOT ESTABLISHED** | only one channel ever reached a formal ADMIT |
| 10 | Correctability is model-dependent | **QUALIFIED** | same channel, same dataset, 3 different outcomes across models — but confounded with n and exposure |
| 11 | A licence DECLINE means the intervention failed | **CONTRADICTED** | Gemma and Qwen3-4B DECLINEs have direction-specific positive point effects, 0/59 randoms, p=0.0167 |
| 12 | The licence is calibrated | **NOT ESTABLISHED** | 3 formal settings; no basis to estimate P(reliable correction \| ADMIT) |
| 13 | Current thresholds are optimal | **NOT ESTABLISHED** | provenance is convention/pilot, not derived from decision utility |
| 14 | The current gate is conservative | **SUPPORTED** | simulated P(ADMIT) 0.93 at the ADMIT effect vs 0.19–0.25 at the DECLINE effects |
| 15 | Destination-resolved accounting adds information beyond Net | **SUPPORTED** | Qwen3-4B Net +39 and Gemma Net +16 both refused on destination; Gemma score-space TG −0.0814 with *more* exits |
| 16 | SAKIKO generalises beyond When2Call | **NOT ESTABLISHED** | all formal evidence on one benchmark; every alternative population failed eligibility |
| 17 | Activation is superior to score-space | **NOT ESTABLISHED** | locked paired test inconclusive (p=0.3770); activation won on DEV destination in Gemma only |
| 18 | Correction failures can be localised by stage | **SUPPORTED** | 19 ledger units resolve to 9 distinct failure classes at 6 distinct rungs |
| 19 | Higher baseline accuracy improves the correction substrate | **CONTRADICTED** | Qwen3.5 has the best accuracy and least collapse, and no actionable channel |
| 20 | The DECLINEs were "really correctable, just under-powered" | **CONTRADICTED** | simulation: with exposure scaled, P(ADMIT) peaks ≈0.46–0.49 near n=400 then falls as collateral binds |
| 21 | The frozen all-correct collateral denominator measures deployment risk | **CONTRADICTED** | Gemma 1/223 = 0.45% passes while 1/11 = 9.1% exposure-conditional exceeds the bound |
| 22 | A conservative evidential rule is being applied consistently | **SUPPORTED** | 8 refusals across 4 rungs incl. the authors' own prior headline result and a one-row support miss |
