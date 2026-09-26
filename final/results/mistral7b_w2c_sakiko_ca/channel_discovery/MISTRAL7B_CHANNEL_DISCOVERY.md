# Mistral-7B-Instruct-v0.3 — Automatic W2C Channel Discovery

- baseline acc 0.4269; total errors 2093; N=3652
- R0 validity: readout_collapse_flag=False (majority_pred 0.821 class tool_call, acc−majority_gold +0.0723)

## Discovered transitions (gold→pred)

| channel | gold→pred | all | tr | val | te | %err | status | boot | in manual | scope |
|---|---|---|---|---|---|---|---|---|---|---|
| rfi_tc | request_for_info→tool_call | 948 | 662 | 141 | 145 | 45.29 | stable | 1.0 | True | PASS |
| ca_tc | cannot_answer→tool_call | 773 | 541 | 117 | 115 | 36.93 | stable | 1.0 | True | PASS |
| ca_direct | cannot_answer→direct | 291 | 205 | 43 | 43 | 13.9 | stable | 1.0 | True | PASS |
| ca_rfi | cannot_answer→request_for_info | 26 | 18 | 2 | 6 | 1.24 | insufficient | 0.0 | False | — |
| rfi_direct | request_for_info→direct | 21 | 17 | 2 | 2 | 1.0 | insufficient | 0.0 | False | — |
| rfi_ca | request_for_info→cannot_answer | 17 | 15 | 0 | 2 | 0.81 | insufficient | 0.0 | False | — |
| tc_direct | tool_call→direct | 15 | 9 | 4 | 2 | 0.72 | insufficient | 0.0 | False | — |
| tc_ca | tool_call→cannot_answer | 1 | 1 | 0 | 0 | 0.05 | insufficient | 0.0 | False | — |
| tc_rfi | tool_call→request_for_info | 1 | 1 | 0 | 0 | 0.05 | insufficient | 0.0 | False | — |

**Scope-passing channels (intervention-eligible):** ['rfi_tc', 'ca_tc', 'ca_direct']

**Auto-discovered stable channels beyond the manual {rfi_tc, ca_tc, ca_direct}:** []
