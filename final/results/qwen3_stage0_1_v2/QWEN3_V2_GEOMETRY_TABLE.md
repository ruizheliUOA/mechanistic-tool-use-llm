# Qwen3 V2 dev geometry table

STAGE_2_NOT_AUTHORIZED

Developmental train/dev result. The evaluation split was not accessed.

| channel | n total | n valid | Q_sum full | Q_sum common | neg-sign full | neg-sign common | resultant full | null p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| request_for_info__to__tool_call | 244 | 244 | 0.710761 | — | 0.557377 | — | 0.529780 | 0.000100 |
| cannot_answer__to__tool_call | 167 | 167 | 0.724542 | — | 0.808383 | — | 0.619150 | 0.000100 |
| cannot_answer__to__direct | 159 | 159 | 0.611052 | — | 0.522013 | — | 0.450134 | 0.000100 |

Full/common ordering consistency: `{"common_Q_sum_order_low_to_high": [], "comparable_channels": [], "exact_order_equal": true, "full_Q_sum_order_low_to_high": [], "kendall_p_two_sided_descriptive": null, "kendall_tau_two_sided_descriptive": null, "pairwise": {"concordant": 0, "discordant": 0, "ties": 0}}`

Geometry never adds or removes a channel.
