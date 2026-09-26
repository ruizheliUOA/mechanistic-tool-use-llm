# E3 — MetaTool re-adjudication

```
METATOOL_BIDIRECTIONAL_EFFECT_SUPPORTED_BUT_NOT_FULLY_ADJUDICABLE
```

Post-hoc historical re-adjudication from recovered primary artifacts. Modern
sealed criteria are **not** imposed retroactively.

---

## Provenance

| field | value |
|---|---|
| phase | Phase 2 — MetaTool x Qwen2.5-7B target-native SAKIKO-CA |
| model | **Qwen2.5-7B-Instruct**, bf16 (not Phi) |
| dataset | MetaTool-Binary, 1,040 rows, native Yes/No readout |
| splits | train / val / test, committed under `data/processed/metatool_binary/` |
| seeds | **5** — 42, 123, 456, 789, 2024 |
| baseline | acc **0.7731**, balanced, deterministic re-score 50/50, 0 skips |
| channels | `no_tool→tool_call` (over-call, n=107) · `tool_call→no_tool` (under-call, n=129) |
| observation | L24, 5/5 seeds |
| injection | L20 x3 / L22 x2 |
| Router AUC | nt_tc 0.927–0.979 · tc_nt 0.947–0.991 |
| direction | **PCA-1** for under-call 5/5 · **DiffMean** for over-call 5/5, chosen on validation |
| row-level | **config-level sweeps only** (`pilot_details.jsonl` n=300, `multiseed_rows.jsonl` n=4) — LFS-fetched and verified; **no per-sample records exist** |

Primary artifacts were read directly; `PHASE2_CLAIM_BOUNDARY.md` was consulted
only after the numbers, and it agrees with them.

---

## Bidirectionality — the load-bearing result

| arm | Net per seed (42/123/456/789/2024) | mean | wins | losses |
|---|---|---:|---:|---:|
| **B over-call only** | 3 · 11 · 15 · 9 · 5 | **+8.6** ± 4.3 | **5/5** | 0 |
| **C under-call only** | 11 · 11 · 12 · 16 · 15 | **+13.0** ± 2.1 | **5/5** | 0 |
| **D dual** | 14 · 22 · 27 · 25 · 20 | **+21.6** ± 4.5 | **5/5** | 0 |

```
additive_all_seeds          : true      (8.6 + 13.0 = 21.6, exact)
partition_overlap_all_seeds : 0         (disjoint on EVERY seed)
```

Accuracy A→D rises from ~0.74–0.81 to **0.89–0.92** across seeds.

**The two directions act on disjoint populations, by construction and in fact.**
The channels are defined by opposite gold labels, and measured eligibility
overlap is exactly zero on all five seeds.

### Why this refutes global suppression

A global tool-call suppressor would improve the over-call channel and
**necessarily worsen** the under-call channel — those are opposite behavioural
directions. Instead both channels improve, in all five seeds, with zero
population overlap and exact additivity. A single global bias cannot produce
that pattern. This is the strongest available rebuttal of the "SAKIKO is just
tool-call suppression" objection, and it is artifact-backed.

### Specificity controls (seed 42 only)

```
reverse direction  : net 0        (changes zero predictions)
random draws       : real Net >= 95th pct (over-call), 100th pct (under-call), K=20
ungated            : nt_tc -39, tc_nt -34   vs gated +3 / +11
```

The ungated result **independently replicates the Phi UNGATED finding** on a
different model and corpus: removing the gate inverts the sign. Router gating is
necessary in both settings.

---

## Why it is not fully adjudicable

1. **Destination resolution is degenerate by construction.** In a binary action
   space, for any channel `g→s`, `|A \ {g,s}| = 0`. `OTHER_WRONG` is empty, so
   `target_hit ≡ 1` whenever any exit occurs and Target Gain collapses to the
   exit rate. Level-1 cannot be *earned* here — it is undefined, not failed.
2. **No collateral evidence.** No per-sample records survive, so
   Router-exposed baseline-correct is unavailable and no collateral rate can be
   computed. Only the coarse bookkeeping survives: `Broke <= Fixed/2` in arm D on
   5/5, and arm B at seed 42 is marginal (Broke 8 vs Fixed 11).
3. **Specificity is seed-42 scope only.** Per-seed random distributions were
   never run.
4. **Small per-channel test support** — 13–22 rows per channel per seed. The sign
   is stable 5/5; the per-channel magnitudes carry wide uncertainty.

**Highest supportable rung: `Steerable`, with direction-specificity at seed-42
scope.** Not Level-1 (structurally undefined), not preservation-verified.

---

## C — Claim matrix updates

Only claims touched by Parts A and B. The frozen matrix is otherwise unchanged.

### C4-1 — preservation turning point · **SURVIVES, point estimate downgraded**

- **Old:** aggregate Net conceals a preservation failure (52/93 = 55.9%).
- **New evidence:** Broke 16–52, Net +53 to +67 across surviving Phi TEST runs;
  every realization violates the preservation budget by >= 1.6x (see
  `preservation_audit/PHI_PRESERVATION_SENSITIVITY.md`).
- **Revised:** *the exact collateral point estimate varies across historical
  executions, but the qualitative preservation failure is robust to the
  realization.*
- **Prohibited:** 52/93 as a stable constant; Broke=16 as improved preservation.

### C2 — correction breadth · **STRENGTHENED, historical/post-hoc label required**

- **Old:** a meaningful subset of tool-decision errors is correctable.
- **New evidence:** a second corpus (MetaTool-Binary) and second model
  (Qwen2.5-7B), 5 seeds, 15/15 arm-runs positive, accuracy 0.77 → ~0.90.
- **Revised strongest:** *channel-specific intervention produces robust positive
  effects on a second benchmark and model, at the Steerable rung; destination
  resolution is undefined in a binary action space, so this does not extend the
  correctability claim.*
- **Prohibited:** counting MetaTool as a correctability or Level-1 result;
  quoting its `target_hit` (trivially 1).

### NEW — C6-1, not global suppression · **ADD as SUPPORTING (historical)**

- **Evidence:** both directions positive 5/5, overlap 0, exact additivity,
  reverse net 0, ungated negative on both channels.
- **Strongest permitted:** *on MetaTool-Binary with Qwen2.5-7B, channel-specific
  interventions improved tool over-calling and under-calling simultaneously, on
  disjoint sample populations, with exactly additive net effects across five
  seeds. A single global shift in tool-call propensity cannot produce this
  pattern.*
- **Prohibited:** any universal mechanism claim; any transfer of this to Phi,
  When2Call, or the multi-class setting; "SAKIKO is never suppression".

### Cross-benchmark breadth · unchanged in kind

MetaTool adds a second corpus at the **Steerable** rung only. The benchmark
concentration objection (reviewer objection 7) is **partly** eased, not resolved:
the licensing ladder still rests on When2Call.
