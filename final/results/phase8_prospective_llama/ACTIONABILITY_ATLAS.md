# Cross-Architecture Actionability Atlas

Explanatory atlas over every channel with adequate controls. **Not** a fitted predictive model — the channel set is far too small (n≈8). Archived rows are development evidence (Phase 5/7); Llama rows are the Phase-8 prospective run.

| model | channel | role | supp(tr) | ref | AUC | obs (depth) | est | best ρ | shape | real | rev | rand μ±σ (max) | n≥real | z | own before→after | gate | truth |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-7B | ca_rfi | dev positive control | 153 | 79 | 0.918 | 20 (0.714) | diffmean | 4.0 | interior-specific | 17 | 0 | 2.8±2.0 (8) | 0 | 6.96 | 39→17 | **ADMIT(dev)** | actionable |
| Qwen2.5-7B | tc_rfi | dev false-admission | 89 | 771 | 0.849 | 24 (0.857) | diffmean | 1.0 | interior (val) / null (test) | 6 | -1 | 1.2±1.4 (4) | 0 | 3.29 | 21→15 | **REJECT (Stage-1 power floor: 21<30)** | non-actionable |
| Mistral-7B-v0.3 | ca_tc | dev discriminator | 541 | 149 | 0.964 | 22 (0.688) | diffmean | 8.0 | boundary-seeking | 65 | 20 | 33.2±18.9 (68) | 2 | 1.68 | 117→23 | **REJECT (no rho passes)** | non-actionable |
| Mistral-7B-v0.3 | rfi_tc | dev negative control | 662 | 51 | 0.878 | 22 (0.688) | diffmean | None | pending/near-null (Phase-7 partial) | None | None | — | None | None | 141→None | **REJECT (ref-pool 51<80)** | non-actionable |
| Mistral-7B-v0.3 | ca_direct | dev negative control | 205 | 149 | 0.938 | 22 (0.688) | pca1 | None | non-specific (archived) | None | None | — | None | None | 43→None | **REJECT** | non-actionable |
| Llama-3.1-8B | rfi_tc | PROSPECTIVE | 471 | 202 | 0.8367 | 22 (0.688) | diffmean | 8.0 | boundary-seeking | 18 | 18 | 24.15±8.708 (38) | 16 | -0.706 | 101→30 | **REJECT** | not tested (no audit slot) |
| Llama-3.1-8B | ca_tc | PROSPECTIVE | 383 | 146 | 0.9585 | 22 (0.688) | diffmean | 8.0 | boundary-seeking | 23 | 0 | 4.5±7.075 (20) | 0 | 2.615 | 79→12 | **REJECT** | non-actionable |
| Llama-3.1-8B | ca_direct | PROSPECTIVE | 316 | 146 | 0.9366 | 22 (0.688) | diffmean | 8.0 | boundary-seeking | 23 | 1 | 7.05±5.472 (21) | 0 | 2.915 | 73→33 | **REJECT** | non-actionable |

### Descriptive hypothesis check (no statistics claimed on n≈8)

| # | hypothesis | verdict |
|---|---|---|
| H1 | interior-specific response ⇔ true actionability | **supported (weakly, n=1 positive)** — the only test-confirmed actionable channel (Qwen ca_rfi) is the only interior-specific one
| H2 | boundary-seeking ⇒ generic perturbation/saturation | **supported** — Mistral ca_tc (boundary-seeking, random μ +27..+33) is non-actionable despite the largest Net |
| H3 | static detectability cannot determine steerability | **supported** — AUC 0.84–0.96 spans both actionable and non-actionable; Mistral ca_tc has the highest AUC and fails |
| H4 | actionability is architecture/channel-specific | **supported** — same channel id (ca_tc) is partially specific on Qwen and non-specific on Mistral |
| H5 | rejection evidence is easier than admission | **supported** — every rejection with ground truth was correct; the one dev admission (tc_rfi, pre-power-floor) was wrong |
| H6 | normalized ρ transfers better than raw α | **supported (mechanistically)** — identical α maps to ρ spanning 0.84–2.06× across channels; ρ makes curves comparable across architectures. Not an actionability claim. |
