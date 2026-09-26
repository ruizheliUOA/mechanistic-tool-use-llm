# Qwen3 V2 prediction table

STAGE_2_NOT_AUTHORIZED

One row per support-eligible channel.

| channel | train err | train ref | dev err | dev ref | ROC-AUC | PR-AUC | tau | precision | Q_sum full | Q_sum common | neg-sign full | neg-sign common | concentration | concentration null p | direction sha256 | eligible | exclusion |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| request_for_info__to__tool_call | 448 | 102 | 244 | 52 | 0.769961 | 0.629204 | 0.4 | 0.578947 | 0.710761 | — | 0.557377 | — | 0.529780 | 0.000100 | `4d7743af6844359cc4803f1f5952a25cf58cf8d0278368d67740b3f473ddd433` | True | — |
| cannot_answer__to__tool_call | 314 | 75 | 167 | 43 | 0.938736 | 0.867632 | 0.4 | 0.829412 | 0.724542 | — | 0.808383 | — | 0.619150 | 0.000100 | `1ec06c367d00c5aba6016d6c78f7721e5c7688401e584cefea4aedd58b0ab172` | True | — |
| cannot_answer__to__direct | 269 | 75 | 159 | 43 | 0.932910 | 0.836155 | 0.4 | 0.936620 | 0.611052 | — | 0.522013 | — | 0.450134 | 0.000100 | `a136818a657e67cdd3c5b41417d8514b3016ad1b8e20717d28638c80c3836c4d` | True | — |
