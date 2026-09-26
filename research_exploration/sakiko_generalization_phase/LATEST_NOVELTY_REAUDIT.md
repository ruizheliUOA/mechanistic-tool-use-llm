# Fresh novelty re-audit — the search the last phase did not run

Primary sources read this phase, not abstracts alone.

## The near-collision, and exactly where it lands

**arXiv 2607.24343 — "Beyond Aggregate Risk: Role-Stratified Conformal Risk Control for LLM
Tool Calls" (July 2026).** The closest conceptual neighbour located to date, and it is *not* a
steering paper — confirming the previous phase's suspicion that the nearest prior would come
from the reliability literature.

| dimension | 2607.24343 | SAKIKO |
|---|---|---|
| domain | LLM tool calls | LLM tool calls |
| allow/block decision rule | **YES** — per-field threshold | YES — ADMIT/DECLINE |
| **collateral on benign cases** | **YES** — "over-intervention: fraction of benign fields modified or blocked" (30.6%) | YES — exposure-conditional collateral |
| **finite-sample risk guarantee** | **YES — conformal risk control** | **NO — conventions** |
| causal intervention on internals | **NO** — post-process / abstain only | **YES** |
| destination among K≥3 wrong actions | **NO** — binary violated/allowed | **YES** |
| automatic error-channel discovery | **NO** — 6 predefined semantic roles | **YES** |
| readable→steerable→correctable staging | NO | YES |

**What this costs SAKIKO:** the "we measure collateral on exposed benign cases" contribution is
**no longer unique**, and the neighbour does it with a distribution-free guarantee where SAKIKO
has a convention. The previous audit's `CONSERVATIVE_BUT_UNCALIBRATED` verdict is now worse than
"a known limitation" — it is a limitation the adjacent literature has already solved for its
own object.

**What survives:** causal intervention specificity, destination resolution among K≥3 wrong
actions, and automatic channel discovery. None of the three appears in 2607.24343.

## Other 2026 findings

- **arXiv 2606.29054** — impossibility result: when base risk exceeds a threshold, any
  distribution-free certification must abstain on a large fraction. **This is a formal reason to
  expect DECLINEs to dominate**, and it makes SAKIKO's 8-of-9 refusal rate theoretically
  expected rather than embarrassing. A genuinely useful citation.
- **arXiv 2604.09155 (CORA)** — conformal risk control for GUI agent harmful-action rates.
  Same statistical machinery, execution-stage object, no representation intervention.
- **arXiv 2603.22161** — activation steering of confidence changes abstention behaviour in
  Gemma-3-27B. Closest *steering* neighbour: steers a latent property to move abstention, but
  binary, no destination resolution, no licensing.
- **arXiv 2607.02577 — "Benchmarking the Benchmarks: A Validity Audit of Tool-Calling
  Evaluation" (July 2026).** Trace-level review of 496 tasks finds evaluator error rates of
  **BFCL v4 20%, τ²-Bench Retail 9.8%, MCP-Atlas 13.5%, LiveMCPBench 30.5%**, with
  LiveMCPBench scores ranging 57.9–76.8% across 23 repeats.

## The re-audit's most consequential finding

That last paper **independently validates SAKIKO's benchmark-eligibility discipline** — the
ToolDial confound rejection and the WildToolBench interface rejection were exactly the checks
the field now demonstrably needs — and simultaneously **raises the bar for any second
benchmark**: adopting BFCL v4 or an MCP benchmark would import 20–30% label noise into a study
whose endpoints are differences of a few dozen samples.

## Three-layer novelty verdict

| layer | verdict | why |
|---|---|---|
| A — `readable ⇏ steerable ⇏ correctable` in tool use | **COLLIDED / MODERATE** | 2605.05715 owns decodable-not-corrected; 2603.22161 owns steering-abstention |
| B — staged protocol (support→specificity→destination→preservation→evidence) | **MODERATE** | structure resembles selective-prediction pipelines; the *destination* rung is not present in any located work |
| C — tool-use reliability system: automatic channel discovery + causal-intervention adjudication | **STRONG** | no located work discovers error channels and adjudicates causal interventions on them |

Novelty after generalization: **`DEFENSIBLE_AND_NARROWER_THAN_HOPED`** — better than the
previous `DEFENSIBLE_BUT_NARROW` at Layer C, worse at the collateral contribution.
