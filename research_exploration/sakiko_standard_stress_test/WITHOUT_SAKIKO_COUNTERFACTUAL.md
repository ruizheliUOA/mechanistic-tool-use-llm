# What a competent conventional paper would have concluded

Best-practice conventional reporting assumed: accuracy, Fixed/Broke/Net, a matched-random
control, a reverse control, and a collateral number. Not a strawman.

| setting | conventional conclusion | SAKIKO conclusion | changed? |
|---|---|---|:-:|
| **Qwen3-4B `ca→tc`** | "Net **+39**, Fixed 40 / Broke 1, **0 of 59** random directions match, reverse dead, 37 errors repaired. A clean cross-model replication." | DECLINE — destination intervals fail (t-hit CI lower 0.4545, TG CI lower −0.0484) | **YES** |
| **Gemma `ca→tc`** | "Net **+16**, Fixed 17 / Broke 1, **0 of 59** randoms, reverse produced zero exits, collateral **0.45%**. Cross-family replication." | DECLINE — same two intervals | **YES** |
| **Phi-3.5** | "Net **+55**, +10.04 accuracy points, beats all four placebos." *(this is what the project itself published in 2026-03)* | WOULD-DECLINE — collateral 52/264 = **19.7%**, 3.9× the bound | **YES** |
| **Gemma `ca→direct`** | "target-hit **0.86**, TG +0.186, 19 repaired, **zero** breakage. The best result in the panel." | DECLINE — collateral untestable, 0 exposed rows | **YES** |
| **Mistral `rfi_tc`** | Careful reporting already rejects: real +12 < random mean +29 | rejected at specificity | no |
| **Llama locked TEST** | Careful reporting already rejects: 5/20 randoms ≥ real | rejected at specificity | no |
| **Qwen3.5** | likely: "no usable channel" — or worse, intervene on `ca→*` anyway with 29 reference rows | NO-GO at support, prospectively | **YES (partially)** |
| **Qwen3-8B** | "successful correction" | ADMIT | no |

## The count

**Four of eight** settings would have been reported differently — and in every one of those four
the conventional reading is *more* favourable. Three would have been published as successes.

The Phi case is not hypothetical: **the project itself published Net +55 as a success in
2026-03**, with `clean_damage: {n: 264, damaged: 52, rate: 19.7}` sitting in the same results
file, measured and not treated as decisive.

## The honest counterweight

Two of the four changes (Qwen3-4B, Gemma) come from the **certification** layer, not from
destination measurement. A conventional paper reporting bootstrap CIs on its endpoints — which
good practice arguably already requires — would have caught both without any of SAKIKO's
apparatus.

So the incremental value decomposes as:

- **Genuinely novel contribution:** collateral estimand discipline (Phi), non-vacuity (Gemma
  `ca→direct`), prospective support gating (Qwen3.5). Three settings.
- **Enforcing existing best practice:** reporting CIs on destination endpoints. Two settings.

The framework's defensible claim is that it makes all of this **prespecified and mandatory**
rather than optional and post-hoc — which is exactly what failed in the Phi case, where the
number existed and was ignored.
