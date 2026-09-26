# Mistral-7B — Placebo & Specificity Controls (seed-42, locked config)

| channel | real Net | reverse Net | random mean±std (max) | real pct | #rand≥real | ungated Net(broke) | wrong-layer Net |
|---|---|---|---|---|---|---|---|
| rfi_tc | **+12** | +19 | 29.1±14.0 (54) | 15 | 17 | +18(30) | -13 |
| ca_tc | **+62** | +41 | 35.5±13.9 (67) | 95 | 1 | -97(191) | +53 |
| ca_direct | **+10** | +12 | 8.7±4.3 (20) | 65 | 7 | +12(0) | +24 |
