# Qwen3 V2 Router table

STAGE_2_NOT_AUTHORIZED

| channel | train pos | train neg | dev ROC-AUC | dev PR-AUC | tau | precision | recall | specificity | routed | eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| cannot_answer__to__direct | 269 | 848 | 0.9329 | 0.8362 | 0.4 | 0.9366 | 0.8365 | 0.7692 | 142 | True |
| cannot_answer__to__tool_call | 314 | 848 | 0.9387 | 0.8676 | 0.4 | 0.8294 | 0.8443 | 0.9526 | 170 | True |
| request_for_info__to__tool_call | 448 | 848 | 0.7700 | 0.6292 | 0.4 | 0.5789 | 0.6762 | 0.7757 | 285 | True |
