# Llama-3.1-8B — W2C Channel Topology (Phase 8)

- baseline acc 0.4403; total errors 2044; collapse flag False (majority pred 0.640 class tool_call)

## Discovered transitions

| channel | gold→pred | all | tr | val | te | status | boot | ref | own_val_err | Stage-1 |
|---|---|---|---|---|---|---|---|---|---|---|
| rfi_tc | request_for_info→tool_call | 678 | 471 | 101 | 106 | stable | 1.0 | 202 | 101 | ELIGIBLE |
| ca_tc | cannot_answer→tool_call | 535 | 383 | 79 | 73 | stable | 1.0 | 146 | 79 | ELIGIBLE |
| ca_direct | cannot_answer→direct | 463 | 316 | 73 | 74 | stable | 1.0 | 146 | 73 | ELIGIBLE |
| ca_rfi | cannot_answer→request_for_info | 99 | 68 | 10 | 21 | weak | 0.06 | 146 | 10 | own_val_err=10<30 |
| tc_direct | tool_call→direct | 91 | 66 | 13 | 12 | weak | 0.09 | 776 | 13 | own_val_err=13<30 |
| rfi_direct | request_for_info→direct | 73 | 54 | 11 | 8 | weak | 0.0 | 202 | 11 | own_val_err=11<30 |
| tc_rfi | tool_call→request_for_info | 66 | 45 | 9 | 12 | weak | 0.0 | 776 | 9 | own_val_err=9<30 |
| rfi_ca | request_for_info→cannot_answer | 25 | 18 | 5 | 2 | insufficient | 0.0 | 202 | 5 | train<30; eval<8; status=insufficient; own_val_err=5<30 |
| tc_ca | tool_call→cannot_answer | 14 | 11 | 2 | 1 | insufficient | 0.0 | 776 | 2 | train<30; eval<8; status=insufficient; own_val_err=2<30 |

**Stage-1-eligible (proceed to gate):** ['rfi_tc', 'ca_tc', 'ca_direct']

## Cross-architecture topology (stable channel counts)

| channel | Phi | Qwen | Mistral | Llama |
|---|---|---|---|---|
| ca_direct | 393 | 465 | 291 | 463 |
| ca_rfi |  | 219 |  |  |
| ca_tc | 511 | 513 | 773 | 535 |
| rfi_tc | 746 | 622 | 948 | 678 |
| tc_rfi |  | 133 |  |  |

*(Cross-model counts are context only; they do not alter the Llama rules.)*
