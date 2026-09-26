# SAKIKO Benchmark Adapter Contract

What any benchmark must supply before SAKIKO-Core can be instantiated. Ten requirements; each
maps to a construct and to a real historical rejection.

| # | requirement | construct | rejected in practice by |
|---|---|---|---|
| 1 | deterministic or deterministically recoverable first-decision gold | A | ToolDial (confounded), LLM-judge benchmarks |
| 2 | `\|A_D\| ≥ 3` with ≥1 meaningful alternative beyond `{g,s}` | C | **MetaTool** |
| 3 | stable model-side action scoring / readout | A | **WildToolBench (G3)** |
| 4 | directed error support ≥ frozen floor, TRAIN and DEV | A | **Qwen3.5 `ca→*` (29 vs 30)** |
| 5 | valid reference population (`gold = pred = g`) | A, readability | **Qwen3-8B V1**, Qwen3.5 |
| 6 | baseline-correct population | D | — |
| 7 | **defined intervention exposure set** | D | **Gemma & Qwen3-4B `ca→direct`** (exposure = 0) |
| 8 | sample IDs and grouping metadata | evidence | — |
| 9 | no contamination with SAKIKO development if used confirmatorily | licensing | ACEBench, ToolSandbox, MetaTool |
| 10 | **measured evaluator/gold validity** | A | **new — motivated by arXiv 2607.02577** |

Requirement 10 is added this phase. Every requirement has rejected a real candidate, so the
contract is empirically grounded rather than aspirational.

## Note on requirement 7

Exposure is intervention-surface-dependent: for Router-gated activation editing it is
Router-fired baseline-correct rows; for score-space it is rows whose margin is within the bias;
for direct mode selection it is all rows the policy touches. **The contract requires the adapter
to declare the exposure set, not to assume the activation one** — which is what makes construct
D surface-agnostic.
