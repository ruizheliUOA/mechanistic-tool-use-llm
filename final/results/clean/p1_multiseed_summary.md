# P1: Multi-Seed Stability Report

**Date**: 2026-03-20
**Seeds**: 42, 123, 456, 789, 2024
**Method**: Full clean pipeline per seed (split → val sweep → lock → test)

## 1. Per-Seed Results

| seed | thr | α | baseline | acc% | Δacc | F1% | ΔF1 | fixed | broke | net | rfi_tc | ca_tc | ca_dir |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | 0.40 | 10.0 | 48.18 | 58.21 | +10.04 | 46.03 | +12.23 | 107 | 52 | +55 | 22 | 11 | 22 |
| 123 | 0.40 | 10.0 | 48.18 | 58.03 | +9.85 | 45.71 | +11.71 | 105 | 51 | +54 | 21 | 9 | 24 |
| 456 | 0.60 | 8.0 | 48.18 | 57.85 | +9.67 | 45.02 | +10.26 | 94 | 41 | +53 | 17 | 13 | 23 |
| 789 | 0.90 | 8.0 | 48.18 | 59.12 | +10.95 | 45.22 | +11.37 | 76 | 16 | +60 | 27 | 11 | 22 |
| 2024 | 0.60 | 8.0 | 48.18 | 60.40 | +12.23 | 47.16 | +14.14 | 107 | 40 | +67 | 27 | 18 | 22 |

## 2. Summary Statistics

| Metric | Mean | Std | Min | Max |
|---|---|---|---|---|
| accuracy | 58.72 | 1.06 | 57.85 | 60.40 |
| delta_acc | 10.55 | 1.06 | 9.67 | 12.23 |
| macro_f1 | 45.83 | 0.84 | 45.02 | 47.16 |
| delta_f1 | 11.94 | 1.43 | 10.26 | 14.14 |
| net | 57.80 | 5.81 | 53.00 | 67.00 |
| fixed | 97.80 | 13.33 | 76.00 | 107.00 |
| broke | 40.00 | 14.51 | 16.00 | 52.00 |

## 3. Channel-Level Stability

| Channel | Mean net | Std | Min | Max | #Positive / #Seeds |
|---|---|---|---|---|---|
| rfi_tc | +22.8 | 4.3 | +17 | +27 | 5/5 |
| ca_tc | +12.4 | 3.4 | +9 | +18 | 5/5 |
| ca_direct | +22.6 | 0.9 | +22 | +24 | 5/5 |

## 4. Conclusion

### Pass Criteria Check

- 5/5 seeds overall net > 0: ✅ (5/5)
- 5/5 seeds delta_acc > 0: ✅ (5/5)
- Mean net clearly positive: ✅ (mean=+57.8, std=5.8)
- ≥2 channels mostly positive: ✅ (3/3 channels)

### Stability Ranking (mean/std ratio)

- **ca_direct**: mean=+22.6, std=0.9, ratio=25.39 ← most stable
- **rfi_tc**: mean=+22.8, std=4.3, ratio=5.34 
- **ca_tc**: mean=+12.4, std=3.4, ratio=3.60 ← least stable

### Overall Verdict

**✅ P1 PASSED.** The method is stable across 5 seeds.

Strongest claim upgrade: "SAKIKO V3.1 achieves Δacc=+10.55% ± 1.06% and net=+57.8 ± 5.8 across 5 independent splits (seeds: 42, 123, 456, 789, 2024)."

**Recommendation**: Ready to proceed to P3 cross-model.
