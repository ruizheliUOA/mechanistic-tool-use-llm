# Final model and benchmark evidence map

## Models — seven, one framework, provenance preserved

| model | scale | family | dataset | protocol | era | Act-1 | Act-2 | **strongest rung** |
|---|---|---|---|---|---|---|---|---|
| **Qwen3-8B** | 8B | Qwen3 | W2C | sealed frozen | modern | 72 ch. errors, 38 gold | t-hit **0.7308**, CI95 **[0.6042, 0.8462]**; E1 0/211 | **`FORMAL_CONFIRMATORY_SUCCESS`** — Licensable |
| **Qwen3-4B** | 4B | Qwen3 | W2C | sealed frozen | modern | 118 ch. errors, 37 gold | t-hit **0.5781**, CI95 lower **0.456** — crosses 0.50 | **`FORMAL_DECLINE`** — Correctable (point), not Licensable |
| **Gemma-2-9b-it** | 9B | Gemma-2 | W2C | sealed frozen | modern | 96 ch. errors, 17 gold | t-hit **0.6296**, CI95 lower **0.444** — crosses 0.50 | **`FORMAL_DECLINE`** — fails conditions 3 & 6 |
| **Phi-3.5-mini** | 3.8B | Phi-3 | W2C | historical locked | historical | Net **+55**, 5 seeds +53…+67 | t-hit 0.6993; **E2 52/93 = 55.9%** | **Steerable** — preservation fails |
| **Qwen2.5-7B** | 7B | Qwen2.5 | W2C + MetaTool | historical | historical | +79 real vs +16 reverse; MetaTool 15/15 | binary space: destination degenerate | **Steerable** |
| **Llama-3.1-8B** | 8B | Llama-3.1 | W2C | historical locked | historical | descriptive redistribution | 19 gold vs **44 other-wrong**, t-hit ≈0.30 | **Effect only** — fails specificity (5/20, 3/20 randoms ≥ real) |
| **Mistral-7B-v0.3** | 7B | Mistral | W2C | historical | historical | real +12 | — | **Effect only** — fails specificity (17/20 randoms ≥ real; reverse +19 > real +12) |

**Provenance rule:** modern/sealed rows (Qwen3-8B, Qwen3-4B, Gemma) and
historical rows (Phi, Qwen2.5-7B, Llama, Mistral) must be **visually separated** in
every figure and table. They belong to one framework, not one protocol.

## What the rung distribution actually shows

This is the empirical content of C5-1 — **not the count of Level-2 licences**:

- **One formal ADMIT** — Qwen3-8B, `FORMAL_CONFIRMATORY_SUCCESS`.
- **Two formal DECLINEs** — Qwen3-4B and Gemma. Both have *positive destination
  point estimates* (0.5781, 0.6296) and both are declined because the interval
  crosses the frozen boundary. This is the gate working, not the method failing:
  `DECLINE != intrinsically uncorrectable` is already frozen in the definitions.
  Support is the binding constraint — 64 and 27 exits respectively.
- **Steerable and Correctable, but not Preserving** — Phi is the important case:
  structurally specific, aggregate-positive, destination-concentrated (107 gold vs
  46 other-wrong), and **preservation-failing**. *(Amended 2026-09-10: previously
  "Steerable but not correctable", which contradicted `FINAL_FORMAL_STATUS_TABLE.md`.)*
- **Effect but not steerable** — Llama and Mistral. Llama is the cleanest
  descriptive example of *movement without arrival* (19 gold vs 44 other-wrong),
  but because it failed specificity first it cannot serve as a licensed example of
  "steerable but not correctable."
- **Not adjudicable** — Qwen3.5 `ca→*` failed channel support (DEV reference 29 vs
  the required 30). A rung the framework can also return.

The hierarchy discriminates in **both** directions. That is the result.

## Benchmarks — classified by what each supports

| benchmark | supports | does **not** support |
|---|---|---|
| **When2Call (W2C)** | correction, destination verification, preservation, full licensing | — |
| **MetaTool-Binary** | correction breadth; bidirectionality; gating necessity replication | any Correctable or licensing claim — `OTHER_WRONG` empty by construction |
| **BFCL** | OOD/supporting only, where artifact-backed | licensing of any kind |
| ToolSandbox, ACEBench | **excluded** — no quantitative channel structure / readout unsolved | — |

**Never imply every benchmark validates full licensing.** Only W2C does.
