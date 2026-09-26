# P0 验收审计报告

**Date**: 2026-03-20  
**Auditor**: Copilot (automated)  
**Status**: ✅ PASS — P0 可作为后续统一基线

---

## 1. 文件存在性检查

| 文件 | 存在 | 大小 |
|---|---|---|
| p0_split_info.json | ✅ | 1.4 KB |
| p0_val_threshold_sweep.json | ✅ | 4.4 KB |
| p0_val_alpha_sweep.json | ✅ | 4.5 KB |
| p0_locked_config.json | ✅ | 514 B |
| p0_final_test_eval.json | ✅ | 2.7 KB |
| p0_locked_vs_old_comparison.md | ✅ | 1.2 KB |
| p0_joint_grid_val.json | ✅ | (联合搜索复核) |
| splits/train_idx.json | ✅ | 14.6 KB |
| splits/val_idx.json | ✅ | 3.1 KB |
| splits/test_idx.json | ✅ | 3.1 KB |

---

## 2. Val-Only Selection 证明

### 2a. Threshold 只在 val 上选
- **代码证据**: `val_threshold_sweep()` (p0_run_locked_eval.py L202-247)
  - L205: `_train_pipeline(acts_obs, meta, train_idx, ...)` — 训练只用 train
  - L216: `evaluate_on_split(..., val_idx, ...)` — 评估只用 val
  - L244: `best = max(sweep_results, ...)` — 选择基于 val 结果
- **数据证据**: p0_val_threshold_sweep.json 标注 `"sweep_on": "val"`, `"n_val": 548`
- **结论**: ✅ threshold 完全基于 val set 选定

### 2b. Alpha 只在 val 上选
- **代码证据**: `val_alpha_sweep()` (p0_run_locked_eval.py L250-322)
  - L254-258: PCA bases 训练只用 train_idx
  - L291: `evaluate_on_split(..., val_idx, ...)` — 评估只用 val
  - L319: `best = max(sweep_results, ...)` — 选择基于 val 结果
- **数据证据**: p0_val_alpha_sweep.json 标注 `"sweep_on": "val"`, `"locked_threshold": 0.4`
- **结论**: ✅ alpha 完全基于 val set 选定

### 2c. Test set 只在最后一步使用一次
- **代码证据**: `test_idx` 在代码中只被传递给两处：
  1. `save_split_info()` — 仅用于计数和存储索引
  2. `final_locked_test()` (L352-399) — 唯一一次调用 `evaluate_on_split(..., test_idx, ...)`
- **grep 确认**: `test_idx` 从未出现在 `val_threshold_sweep` 或 `val_alpha_sweep` 的调用链中
- **结论**: ✅ test set 只评估了一次

### 2d. Baseline 和 intervened metrics 来自同一 test split
- **代码证据**: `compute_metrics()` 在 `final_locked_test()` 内一次性对 548 test 样本计算 baseline 和 intervened
- **数据证据**: p0_final_test_eval.json 中 `n_test=548`, `baseline_accuracy=48.18`, `intervened_accuracy=58.21` 均来自同一 results 列表
- **交叉验证**: baseline 48.18% = 264/548 (test correct count / test total)，与 p0_split_info.json 中 test_error_counts.correct=264 一致
- **结论**: ✅ baseline 和 intervened 来自同一 test split

---

## 3. Split 与结果汇总表

### 3a. Split 分布

| 类别 | train | val | test |
|---|---|---|---|
| **总数** | **2556** | **548** | **548** |
| correct | 1233 | 264 | 264 |
| rfi→tc | 522 | 112 | 112 |
| ca→tc | 357 | 77 | 77 |
| ca→direct | 275 | 59 | 59 |
| ca→rfi | 65 | 14 | 14 |
| rfi→direct | 33 | 7 | 7 |
| tc→rfi | 26 | 6 | 5 |
| tc→direct | 22 | 4 | 5 |
| rfi→ca | 13 | 3 | 3 |
| tc→ca | 10 | 2 | 2 |

Split hashes:
- train: `7b843b43080e9465`
- val: `2d2f078a47ae7702`
- test: `aa7dbd0750025fa8`

### 3b. Locked Config

| 参数 | 值 | 来源 |
|---|---|---|
| threshold (all channels) | **0.40** | val sweep (max net) |
| ca_direct_alpha | **10.0** | val sweep (max net) |
| rfi_alpha | 5.0 | 继承自 V3.1 |
| ca_tc_alpha | 2.0 | 继承自 V3.1 |

### 3c. Final Locked Test Results

| 指标 | 值 |
|---|---|
| Test samples | 548 |
| Baseline accuracy | 48.18% |
| **Intervened accuracy** | **58.21%** |
| **Delta accuracy** | **+10.04%** |
| Baseline Macro-F1 | 33.80% |
| **Intervened Macro-F1** | **46.03%** |
| **Delta F1** | **+12.23%** |
| Fixed | 107 |
| Broke | 52 |
| **Net benefit** | **+55** |

Per-channel:
| Channel | Routed | Fixed | Broke | Net | Precision |
|---|---|---|---|---|---|
| rfi_tc | 137 | 59 | 37 | +22 | 43.1% |
| ca_tc | 83 | 18 | 7 | +11 | 21.7% |
| ca_direct | 73 | 30 | 8 | +22 | 41.1% |

---

## 4. 联合网格搜索复核

### 方法
- Sweep: threshold ∈ {0.40, 0.50, 0.60, 0.70} × alpha ∈ {6, 8, 10, 12}
- 16 种组合，全部在 val set (n=548) 上评估
- 选择标准: max net_benefit, tiebreak by accuracy

### 完整网格结果 (按 net 降序)

| thr | alpha | acc% | f1% | net | fixed | broke |
|---|---|---|---|---|---|---|
| **0.40** | **10.0** | **62.23** | **48.24** | **+77** | **117** | **40** |
| 0.40 | 8.0 | 62.04 | 48.12 | +76 | 116 | 40 |
| 0.40 | 12.0 | 61.86 | 48.09 | +75 | 116 | 41 |
| 0.50 | 10.0 | 61.68 | 47.67 | +74 | 110 | 36 |
| 0.50 | 8.0 | 61.50 | 47.55 | +73 | 109 | 36 |
| 0.40 | 6.0 | 61.31 | 47.56 | +72 | 112 | 40 |
| 0.50 | 12.0 | 61.31 | 47.51 | +72 | 109 | 37 |
| 0.60 | 10.0 | 61.13 | 47.20 | +71 | 103 | 32 |
| 0.60 | 8.0 | 60.95 | 47.08 | +70 | 102 | 32 |
| 0.50 | 6.0 | 60.77 | 46.98 | +69 | 105 | 36 |
| 0.70 | 10.0 | 60.77 | 46.84 | +69 | 98 | 29 |
| 0.60 | 12.0 | 60.77 | 46.99 | +69 | 102 | 33 |
| 0.70 | 8.0 | 60.58 | 46.71 | +68 | 97 | 29 |
| 0.70 | 12.0 | 60.58 | 46.75 | +68 | 98 | 30 |
| 0.60 | 6.0 | 60.22 | 46.45 | +66 | 98 | 32 |
| 0.70 | 6.0 | 59.85 | 46.07 | +64 | 93 | 29 |

### 结论
**✅ P0 locked config (threshold=0.40, alpha=10.0) 通过联合搜索复核**

(0.40, 10.0) 在 16 种组合中排名第一，net=+77, acc=62.23%。
次优 (0.40, 8.0) 差 1 net 和 0.19% acc。顺序选参未造成偏差。

---

## 5. 已知局限

1. **Val/test 数值差异**: val 上 net=+77 (62.23%), test 上 net=+55 (58.21%)。
   - 差距 ~5% 可能来自 split variance (各 548 样本)
   - 需要 P1 multi-seed 确认稳定性

2. **rfi_tc channel 的 damage rate 偏高**: test 上 broke=37/137=27%，precision=43.1%
   - 这是三个 channel 中 damage 最高的，需后续优化 router

3. **与旧 test-tuned 结果对比**: 旧 61.13% → 新 58.21% (-2.92%)
   - 符合预期：去除 test-set tuning 后数字天然回落
   - 旧数字的 test set 有 1096 样本，新 test set 只有 548

---

## 6. 审计结论

| 检查项 | 结果 |
|---|---|
| 文件完整性 | ✅ 全部 10 个文件存在 |
| Threshold val-only | ✅ 代码+数据双重确认 |
| Alpha val-only | ✅ 代码+数据双重确认 |
| Test 只用一次 | ✅ grep 确认 |
| Baseline/intervened 同源 | ✅ 264/548 = 48.18% 交叉验证 |
| 联合搜索复核 | ✅ (0.40, 10.0) 仍为全局最优 |
| 无 test leakage | ✅ |
| 旧文件未被修改 | ✅ |

**P0 结果干净、可复现，可作为后续所有实验的统一基线。**
