# Tool-use benchmark landscape, August 2026

Fresh search this phase. Eligibility categories: **A** native fit · **B** lossless adapter ·
**C** new instrument required · **D** unsuitable.

| benchmark | first-action gold | `\|A_D\|≥3`? | category | note |
|---|---|:-:|:-:|---|
| **When2Call** | explicit | yes (3 gold + distractor) | **A** | consumed by 3 formal trials |
| **RUT-Bench** (2606.03318) | multi-turn, user-behaviour derived | plausible | **C** | 1638 samples, 59 executable environments, ideal + non-ideal user patterns, single- and multi-turn; code released May 2026. **Highest-information candidate located.** |
| ACEBench | partial (Normal/Special/Agent) | Special has ambiguous/incomplete instructions | **C** | prior SAKIKO readout-audit population — **not independent** |
| WildToolBench | native explicit | yes | **C** | previously rejected at G3 interface non-equivalence |
| τ²-Bench | dual-control user/agent | plausible | **C** | **evaluator error 9.8%** (2607.02577) |
| BFCL v4 | partial | partial | **D/C** | **evaluator error 20%**; upstream of W2C — shared lineage |
| MCP-Atlas | trajectory | — | **C** | **evaluator error 13.5%** |
| LiveMCPBench | trajectory | — | **D** | **evaluator error 30.5%**, 23 repeats span 57.9–76.8% |
| MCP-Bench / TRAJECT-Bench | trajectory-level | — | **C** | decision-point, not first-action; long-term extension only |
| ToolHop | multi-hop | — | **D** | wrong construct |
| ToolDial | nominal | — | **D** | G4 confound: 15 action sequences / 11,111 dialogues |
| MetaTool | binary | **no** | **D** | `\|A_D\{g,s}\|=0` — Correctable undefined |
| ToolSandbox | partial | — | **D** | prior development corpus |
| ToolFailBench | post-execution | — | **D** | wrong construct |
| UserToolBench (2608.10042) | personalisation, profile-hidden | unclear | **C** | new Aug 2026; unscreened |

## The finding that changes the second-benchmark calculus

**arXiv 2607.02577** audited 496 tasks and found evaluator error rates of **9.8%–30.5%** across
BFCL v4, τ²-Bench, MCP-Atlas and LiveMCPBench, with LiveMCPBench irreproducible across repeats.

SAKIKO's formal endpoints are differences of tens of samples (Gemma: 27 exits, 17 gold).
**Importing a benchmark with 20% label noise would swamp the effect being measured.** This is
not a reason to avoid a second benchmark; it is a reason that the adapter contract's gold-
validity gate is load-bearing rather than bureaucratic — and it independently vindicates the
ToolDial and WildToolBench rejections.

## Ranking by information gain

1. **RUT-Bench** — different data source (real user–LLM interactions), different user-side
   distribution, native non-ideal behaviours that plausibly instantiate ask/act/defer, released
   code, not consumed. Differs from W2C on data source, interaction structure and label
   generation simultaneously.
2. **WildToolBench** — re-openable only if a prospective instrument can recover the first-decision
   construct without altering native semantics.
3. **τ²-Bench** — richest decision ontology, but 9.8% evaluator error must be handled.

**Not recommended:** anything with a measured evaluator error rate above ~10% until that is
addressed upstream.
