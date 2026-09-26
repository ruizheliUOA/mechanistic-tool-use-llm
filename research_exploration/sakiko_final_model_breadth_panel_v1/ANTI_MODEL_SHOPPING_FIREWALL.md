# Anti-model-shopping firewall

The panel is **closed before inference**. Exactly two checkpoints may be studied:

| slot | checkpoint | role |
|---|---|---|
| MODEL G | `google/gemma-2-9b-it` | primary cross-family replication |
| MODEL Q | `Qwen/Qwen3.5-9B` | secondary cross-generation / architecture stress |

## Binding rules

1. **No replacement on a negative outcome.** A `DECLINE`, `NO_ELIGIBLE_CHANNEL`,
   `BASELINE_INVALID`, `READOUT_INVALID`, `ARCHITECTURE_INTERFACE_NO_GO` or `HARDWARE_NO_GO`
   from either slot does **not** authorise substituting another checkpoint into that slot.
2. **Both slots are reported regardless of outcome**, including outcomes that weaken the paper.
3. **The Gemma fallback** `google/gemma-7b-it` may be triggered **only** by a pre-inference
   mechanical memory/compatibility gate, and only before any scientific baseline is observed.
   It may never be triggered by unattractive scientific results. If used, the record must state
   `HARDWARE_TRIGGERED_MODEL_FALLBACK` and that Gemma-7B is an older-generation architecture.
4. **No quantisation, no precision change, no GGUF, no unofficial conversion, no merged
   finetune, no model substitution** — as a fallback or otherwise.
5. **No third family, no larger Qwen, no larger Gemma, no Llama, no Mistral, no Phi**, and no
   model selected after observing results.
6. After both slots are adjudicated, **model expansion terminates permanently** for the ICLR
   manuscript.

## Access-blocked ≠ scientific outcome

If a slot cannot be executed because of an **access or licensing** condition external to the
science (for example a gated-repository authorisation the operator has not granted), that is a
**mechanical block**, not one of the eight per-model scientific verdicts. It must be reported
as blocked and pending, and it does **not** license a replacement model under rule 1 — the slot
stays assigned to that checkpoint.
