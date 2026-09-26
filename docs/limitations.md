# Evidence scope and known limitations

## Preserved hash discrepancies

Three old hash assertions remain false in the original scientific archive:

| Preserved manifest | Reference | Recorded / actual SHA-256 |
|---|---|---|
| Gemma `development/FORMAL_FREEZE.json` | `DEV_CONJUNCTION.json` | `5dc32e1b04cb91f69639855aeb8aede09e85d3d3cc54c9605733ba8342c66f9a` / `52644754757e71acfe2fcd90bff81ca797ee46de7dd4a13d2050b9888f123780` |
| Qwen3-4B `formal/FILE_INVENTORY.json` | self-entry | `5d048588b40c43541ebcaf37ed49e4d06005f6a169b4512346b26a124ae6387c` / `e0a22196e94aefea329f8792fa23dae7ca24578d6ee7b4952b12692e7ba19dec` |
| Same inventory | `HASHES.json` | `b840a98fdd1f151f40a2b570f700d25321e2bb9a9e02b25d47ef4d9637732318` / `c1e186dcc12b436a7b8d52d426d25fb9f54747018179d1c2c8de4622f3914c4f` |

These bytes were not rewritten. The correct Gemma conjunction hash already appears
in the formal manifest of the pre-execution Git commit, with authorization not
granted and sealed rows read = 0. It remains identical through re-freeze and
execution. Exact pre-execution bytes are retained in
`manifests/provenance/gemma_presealed/`; the earlier parent-directory digest remains
unresolved. This establishes the available conjunction's historical provenance,
not a claim that the older digest passed or an independent trusted timestamp.

The current Qwen3-4B `HASHES.json` validates the delivered inventory and available
results, but does not make that inventory's stale self/cross-hash entries true.

## Availability and scientific scope

- Every one of the 40 recovered LFS payloads is included and verified, including
  development arrays, formal records, random directions and historical rows.
- Full Qwen3-8B and Qwen3-4B 548-row baseline exports are not separately available.
  The population-correct denominators 211/214 remain aggregate-only assertions.
  Exposed-correct denominators 6/50 and their 0/1 breaks are directly checked.
- Gemma's full 548-row baseline has 112 channel errors and 223 correct; the
  frozen formal target-gain denominator is the 96 routed channel errors. Its
  one break among 11 exposed-correct rows is 9.1%, above 5%.
- Historical non-recorded destination outcomes cannot be reconstructed from
  Fixed/Broke/Net. Historical results are not relabeled prospective.
- Some recorded model hashes, activation caches and dataset revisions cannot be
  verified from this distribution. No weights or large raw datasets are bundled.
- Old frozen manifests include references to excluded administrative/writing files,
  earlier operational versions and external inputs. Those references are not
  described as verified. Retained locations are in `manifests/paths.json`.

The runtime code is preserved scientific code, not a newly tested portable GPU
application. The supported portable entry point verifies bundled data only.

## Private runtime-path derivatives

Four recovered Qwen3-4B code/summary files contain private session/cache locations.
They are distributed as explicitly named `.anonymised.py` / `.anonymised.json`
derivatives. Only cache-location string fields were replaced; code AST structure,
non-path fields, experiment numbers and rules were verified unchanged. Original
and derivative hashes are distinct in `manifests/anonymous_derivatives.json`.
The historical-layout preparation tool uses the derived code at its original
module location; it does not claim the old frozen source hash still matches.
