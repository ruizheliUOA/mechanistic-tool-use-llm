# Mistral-7B — per-channel geometry & R1/R2 selection (seed-42 split)

| channel | scope | R1 obs (depth) | norm@R1 | ratio@R1 | AUC@R1 | cos(DM,PC1)@R1 | methods | inj |
|---|---|---|---|---|---|---|---|---|
| rfi_tc | Y | L22 (0.69) | 0.723 | 0.247 | 0.8777 | 0.7877 | ['diffmean'] | [18, 20, 22] |
| ca_tc | Y | L22 (0.69) | 0.794 | 0.2712 | 0.9636 | 0.7351 | ['diffmean'] | [18, 20, 22] |
| ca_direct | Y | L22 (0.69) | 0.515 | 0.1759 | 0.938 | 0.5601 | ['diffmean', 'pca1'] | [18, 20, 22] |
