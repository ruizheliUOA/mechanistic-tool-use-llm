# E2+ — full historical control recovery

```
E2_REMAINS_REDUCED_FIVE_CONDITION
```

The five-condition E-2 result is unchanged and not reopened.

## Search executed

`git rev-list --all --objects` across all three branches and all 169 archive
commits, plus `--diff-filter=D` for deleted paths, matching:
`ungated|tau0|ablation|remove_rfi|remove_ca|no_rfi|no_ca`.

## What exists

| artifact | model | population | verdict |
|---|---|---|---|
| `final/results/exploratory/v31_v31_ablation_2ch.json` | Phi-3.5 | **n=1096** | **INELIGIBLE — wrong population** |
| `sakiko/results/v31_v31_ablation_2ch_details.jsonl` | Phi-3.5 | n=1096 | INELIGIBLE, same |
| `qwen3_4b_w2c_formal_v1/FORMAL_UNGATED_RECORDS.jsonl` | Qwen3-4B | modern | not Phi; already in modern block |
| `gemma/UNGATED_CONTROL.json` | Gemma | modern | not Phi; already in modern block |
| `qwen3_4b_stage_d_e_v1/DEV_UNGATED_CONTROL.json` | Qwen3-4B | DEV | development only |
| `cross_model/p3_qwen25_ablation_2ch.json` | Qwen2.5-7B | different model | not the Phi ladder |

## Why the Phi ablation cannot be merged

`p0_split_info.json` (seed 42, 70/15/15): **n_train 2556 / n_val 548 / n_test 548**.

The v31 ablation reports **n_test = 1096 = 548 + 548** — it was run on
**validation + test combined**. Its baseline accuracy (48.27) also differs from
the locked-test baseline (48.18).

Merging it into the five-condition table would (a) change the population
mid-table and (b) import validation rows that were used for hyperparameter
selection. Both are prohibited.

**The four missing arms — Ungated, remove_rfi_tc, remove_ca_tc, remove_ca_direct
— do not exist on the locked 548-row test population in any commit, on any
branch, in any LFS object.** They were never logged there.

## Consequence

E-2 stands at five conditions. Claim C3-1 is unchanged. The nine-condition
ladder is permanently unavailable, by absence rather than by inaccessibility.
