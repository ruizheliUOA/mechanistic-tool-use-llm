# LFS and artifact availability

All 40 originally inventoried LFS pointers have been recovered. Each file below now
contains full payload bytes, verified against its original pointer OID and size.
The total is 87,789,671 bytes. No model checkpoint or raw dataset was added.

The original frozen `.gitattributes` files are preserved because their bytes are
referenced by frozen hashes. A future publication must store these recovered
payloads as ordinary Git blobs and verify them after cloning; these attribute
files are historical evidence, not proof of an LFS server holding the payloads.

## Bundled full payloads

For every row: status = full bytes bundled; acquisition = included in this artifact.
A recovered historical payload is not a newly rerun experiment. Re-run the read-only
checker to verify actual payload hashes; never hash a pointer as if it were its object.

| Relative path | Bytes | Payload SHA-256 |
|---|---:|---|
| `research_exploration/sakiko_final_model_breadth_panel_v1/gemma/d_diffmean_Linj__cannot_answer__to__direct.npy` | 14464 | `4efdb5568ee7f989440da09858f04b4a15f9b0ae91468da3ab04d5142826024c` |
| `research_exploration/sakiko_final_model_breadth_panel_v1/gemma/DEV_INTERVENTION_RECORDS.jsonl` | 1887981 | `3c1b36458692d188f5fdc10f5c80cf376741b7b618ead85f60b772313419c788` |
| `research_exploration/sakiko_final_model_breadth_panel_v1/gemma/d_diffmean_Lobs__cannot_answer__to__tool_call.npy` | 14464 | `3333b9f019122e5a0ccea34e585c928f418d2fe37404b105e8c41c9551a7d595` |
| `research_exploration/sakiko_final_model_breadth_panel_v1/gemma/router_request_for_info__to__tool_call.npz` | 87018 | `7b139deb08973cf2a8bce97c0dbf0874ceda896b3f7544d2408ff6c0bc5a9482` |
| `research_exploration/sakiko_final_model_breadth_panel_v1/gemma/BASELINE_LOBS_ACTIVATIONS.npz` | 20815417 | `539dae88390b6279838ecc273e735746c19793ecd7dad2884a2e231895420c01` |
| `research_exploration/sakiko_final_model_breadth_panel_v1/gemma/DIRECTION_RECORDS.jsonl` | 246671 | `63f932b3cf400f0bcadf7edc97a13e176407f96f3acebd4b6a59e864736bea76` |
| `research_exploration/sakiko_final_model_breadth_panel_v1/gemma/router_cannot_answer__to__tool_call.npz` | 87018 | `0c34252a02cb94ee4a8eb44e856ff0c15d7b9b23eee65f52dd357c1b717c8619` |
| `research_exploration/sakiko_final_model_breadth_panel_v1/gemma/d_grad__cannot_answer__to__direct.npy` | 14464 | `59e3b0a2ddb2efd29d1c09ffa3c49cef7cff315bb666f80fde79df8a0f8b8d7e` |
| `research_exploration/sakiko_final_model_breadth_panel_v1/gemma/BASELINE_ROWS.jsonl` | 942518 | `b91469538bd0db33d7d771991b544edb92900a850c05f67720b521301c45d288` |
| `research_exploration/sakiko_final_model_breadth_panel_v1/gemma/DEV_CONTROL_RECORDS.jsonl` | 1894998 | `4e57a48dd62da2c255be7c5decf35bb324d56dd6d7a183ad85a0bef749cd0037` |
| `research_exploration/sakiko_final_model_breadth_panel_v1/gemma/d_diffmean_Linj__cannot_answer__to__tool_call.npy` | 14464 | `da28f1bceda22ea8c73737d89bd6f735026d8acfc165b5c0d2ee65b165ab58a7` |
| `research_exploration/sakiko_final_model_breadth_panel_v1/gemma/LINJ_ACTIVATIONS.npz` | 20797882 | `ccf1bab3b538a2c3abd3f564d31f5468a3361420352857a3274a2e43ec4ccc38` |
| `research_exploration/sakiko_final_model_breadth_panel_v1/gemma/router_cannot_answer__to__direct.npz` | 87018 | `7dce9794ad58137318c7f7955621749f312b812a8b8f4e6be477661c4d367566` |
| `research_exploration/sakiko_final_model_breadth_panel_v1/gemma/d_diffmean_Lobs__cannot_answer__to__direct.npy` | 14464 | `69c3b811f8cb07cc2103c2f90f43a6010018519bfa003dd794968ca41a7e795a` |
| `research_exploration/sakiko_final_model_breadth_panel_v1/gemma/d_grad__cannot_answer__to__tool_call.npy` | 14464 | `cd4f75260b8dab35da6ef271b36e2b569afd434cb2a227ec50c2f2d733a81ae4` |
| `research_exploration/sakiko_final_model_breadth_panel_v1/gemma/formal_freeze/formal_randoms_K59.npy` | 845952 | `a3a6a4b0b07ce08eb2fc9c05c39a9a268bc6c3b7fb49fad49954528426285b02` |
| `research_exploration/sakiko_final_model_breadth_panel_v1/gemma/formal_freeze/FORMAL_RECORDS.jsonl` | 5989063 | `f05b75f471da5c8a503fb79fb62b958905e0dcfb25ef0261ba95765c263cfb1e` |
| `research_exploration/qwen35_sakiko/BASELINE_ROWS.jsonl` | 1226102 | `011068fbc939bff0ad8478623e5f78438b92d72ce0037028bac6580821d3a09a` |
| `final/results/qwen3_stage2_formal/QWEN3_STAGE2_FORMAL_RANDOMS.safetensors` | 971816 | `11055724067f46c66a0acebadece01424cd77588c58e8a87c7bca054e403356e` |
| `final/results/qwen3_stage2_formal/QWEN3_STAGE2_FORMAL_RECORDS.jsonl` | 5055748 | `d8f622d4ed08e4ef6395b20c6af3ad5287f8cd2f89af9286608a20c06e8d8819` |
| `final/results/mistral7b_w2c_sakiko_ca/baseline/mistral7b_w2c_baseline_details.jsonl` | 1098088 | `6f6f0fc39562cfd1d2d266cf09f258939ceeb795552ab0bd96ae7339fd23ac1c` |
| `final/results/qwen3_4b_w2c_formal_v1/FORMAL_ZERO_RECORDS.jsonl` | 170322 | `d061538bcd96716251a94140890c3a6b52f7b8e88f70fce151f8151ac4ce2215` |
| `final/results/qwen3_4b_w2c_formal_v1/FORMAL_RANDOM_RECORDS.jsonl` | 11190963 | `4a1cb5089c280e707f5eb13311b11d514bff88c82c0ed735a12854897b2ab9b7` |
| `final/results/qwen3_4b_w2c_formal_v1/FORMAL_REAL_RECORDS.jsonl` | 188663 | `e8b3339511b75380bf6d286abc01959023372cafb973a8f6b96149ea80924111` |
| `final/results/qwen3_4b_w2c_formal_v1/FORMAL_BASELINE_RECORDS.jsonl` | 172587 | `b9651d2bc9a1128967111df3aaa3bf3d48d0c6da209871446b234e0473dfc3f9` |
| `final/results/qwen3_4b_w2c_formal_v1/FORMAL_UNGATED_RECORDS.jsonl` | 393578 | `17f72196483a0f863424ca155d96590ddb75e11b1cac2e30024fb5afc8151f35` |
| `final/results/qwen3_4b_w2c_formal_v1/FORMAL_WRONG_LAYER_RECORDS.jsonl` | 190011 | `ebcf49dea3b5d89f8bd75397723cf7d323a8e0b4e068cadbe81e7a37c124f376` |
| `final/results/qwen3_4b_w2c_formal_v1/FORMAL_DIFFMEAN_RECORDS.jsonl` | 189329 | `13c6520ae1f6afa1a1c670aaf300f1db71ef2101cb06cc9fc5b9cde7729fa876` |
| `final/results/qwen3_4b_w2c_formal_v1/FORMAL_SCORE_SPACE_RECORDS.jsonl` | 149898 | `883ad9824614aafc09fc1fcdc6b59c613760a63bea1fd59e1e9a16876ae2cddc` |
| `final/results/qwen3_4b_w2c_formal_v1/QWEN3_4B_FORMAL_RECORDS.jsonl` | 12511539 | `aed60b18f4dab6f6475426317755440b04ecae09e87e79631bb53a453fe6b84d` |
| `final/results/qwen3_4b_w2c_formal_v1/FORMAL_REVERSE_RECORDS.jsonl` | 188673 | `987d8822d1d71798cd74473ac7dada97c97496f46707eb5e979e005c4796edcb` |
| `final/results/clean/p0_final_test_details.jsonl` | 91258 | `c5478da97b0c6134b1b35e321c23957c13181dbf4769a346ecbd015b42ae3b58` |
| `final/results/acebench_generation_readout/audit/audit_judgments.jsonl` | 30541 | `d409fd35d093dbd2d306ed56ad5078503e4cd61dea987e9d3fa239ad5ac4eb0d` |
| `final/results/qwen3_stage0_1_v2/QWEN3_V2_LOG.jsonl` | 1791 | `657a21d42ceaa3fa087c96e03812b30afd383cbda77bb6d80b3e46c5d37d5bc2` |
| `final/results/qwen3_stage0_1_v2/routers/cannot_answer__to__direct.safetensors` | 49684 | `b1ebad7270971f830d3f3505eceab3c8f448148583502f2104e56c3e207db0b1` |
| `final/results/qwen3_stage0_1_v2/routers/cannot_answer__to__tool_call.safetensors` | 49684 | `5d7000b3952ce97b78c93e636a99e73125b15b9138d21a613f1ca3a3a8313b74` |
| `final/results/qwen3_stage0_1_v2/routers/request_for_info__to__tool_call.safetensors` | 49692 | `3a7086718f703d2c7be272c6f950657c2bf11d2aa44eaa1f01ee4e136e4cc06b` |
| `final/results/qwen3_stage0_1_v2/directions/cannot_answer__to__direct.safetensors` | 17120 | `db986511c918eea56ce5f3f7dc82fbace2c17a8fe9bbdd62fd8ba6cb0948e10a` |
| `final/results/qwen3_stage0_1_v2/directions/cannot_answer__to__tool_call.safetensors` | 17128 | `94c0302a015baad96b5e6b692342f2ef758b53325559a47514ddfe54aa866ce6` |
| `final/results/qwen3_stage0_1_v2/directions/request_for_info__to__tool_call.safetensors` | 17136 | `8774e34f3d286be9f1e330bf88bfd0164d1cf5f8ad148af5c000c8359e883431` |

## Other availability classes

- **Pointer only:** none among the 40 inventoried files; the checker rejects pointer stubs.
- **Unbundled large products:** model checkpoints, nonessential activation caches, and some historical arrays. Their historical manifests do not establish present availability.
- **Public inputs / reconstructable processing:** see `DATASETS_AND_MODELS.md`. A command and public source do not promise bit-identical historical bytes unless a matching original hash is verified.
- **Historical hash only:** see `UNAVAILABLE_REFERENCES.csv` for file-hash references whose bytes are not bundled. Some are model files, external inputs, mutable pre-execution records, or superseded versions. They are not counted as verified.
- **Unavailable historical outcomes:** missing placebo destinations and Qwen2.5 row-level destinations cannot be recovered from aggregate Fixed/Broke/Net.

## Limits that remain

The complete Qwen3-8B and Qwen3-4B baselines are not separately bundled; recorded
population denominators 211 and 214 are not independently recounted here. Qwen3-8B
routed results, exposed denominator 6, destination outcomes and zero control are
checkable. Gemma has a complete 548-row baseline; Qwen3-4B has a 452-row subset. Original-manifest
discrepancies remain unresolved in `INTEGRITY_FINDINGS.md`; recovering these 40
payloads does not erase those discrepancies.
