# Phase 4 — Step 15 · Reproducibility & Safety Checklist

Scanned: `final/results/acebench_generation_readout/**` (27 files, **660 KB**, all text/PNG) and the
8 `scripts/acebench_*.py` + `scripts/make_acebench_readout_figures.py`. Verdict at the bottom.

## 1. Secrets & credentials — **PASS**
| Check | Result |
|---|---|
| API keys / tokens / passwords / bearer / cookies | **none**. Regex hits for `token` are all benign: `max_new_tokens`, `max_prompt_tokens`, "token log-probability", "lang token". |
| SSH keys / `BEGIN … PRIVATE KEY` / `ssh-rsa` | none |
| Cloud keys (`AKIA…`), `sk-…`, `hf_…`, `ghp_…` | none |
| `.env` / credential files | none created or referenced |

*(Note: `token123` / `access_tokens` appear inside **ACEBench dataset** example schemas quoted in
generations — third-party fixture data in gitignored `.cache/`, never in a committed file.)*

## 2. Privacy & environment leakage — **PASS**
| Check | Result |
|---|---|
| Private hosts / IPs (`127.*`, `192.168.*`, `10.*`, `localhost`) | none |
| Proxy / `clash` / port 7890 / socks5 / `network_turbo` | none in any deliverable |
| Absolute local paths (`/root/autodl-tmp`, `/home/<user>`) | **none** — all paths repo-relative |
| PII | none. Names in generations (`John Doe`, `Jane Smith`) are ACEBench synthetic fixtures and live only in gitignored `.cache/`. |
| Machine identity / user identity | not embedded |

## 3. Large / binary artefacts — **PASS**
| Check | Result |
|---|---|
| Files > 1 MB in the deliverable tree | **none** (largest: 72 KB fig; total 660 KB) |
| Model weights (`*.safetensors`, `*.bin`) | none — model read from gitignored `.cache/modelscope/` |
| Activation caches / `*.npy` / `*.pkl` / `*.pth` | **none — this phase extracts no activations at all** |
| Raw generations | **gitignored**. `git check-ignore` confirms `.cache/acebench_phase4/generation_details_{full,paraphrase}.jsonl` and `audit_sample.jsonl` are excluded via `.gitignore:2 (.cache/)`. |
| Temp logs / task-runner files | none in the tree (run logs live in the session scratchpad, outside the repo) |

**Policy fix applied during this scan (disclosed).** `audit/audit_sample.jsonl` (72 KB) was initially
written into the deliverable tree, but it carries **raw generations embedding ACEBench dataset text**
— contrary to this phase's own locked artifact policy (`PHASE4_GENERATION_PROTOCOL.md`, *Compute &
artefact policy*). It was **moved to gitignored `.cache/acebench_phase4/`**. The committed audit
evidence is `audit/audit_judgments.jsonl` (per-row judgments, **metadata only, no raw text**) plus
`audit/audit_summary.json`; the sample is byte-exactly regenerable (seed 7) via the command in
`GENERATION_READOUT_HUMAN_AUDIT.md §4`.

**Gitignore gap — reported, NOT unilaterally changed (rationale).** `.gitignore` covers `.cache/`,
`*.pt`, `*.bin`, `*.safetensors`, but **not** `*.pth`, `*.npy`, `*.pkl`, `clash/`.
- **No Phase-4 risk:** this phase produces zero files matching those patterns (verified by `find`).
- **Adding them would be a no-op for the existing files:** `git ls-files '*.npy'` shows **44 `.npy`
  files already tracked** from earlier phases. `.gitignore` does not untrack tracked files, so adding
  `*.npy` would neither remove nor exclude them — it would only create a false impression that it had.
- Therefore this is **pre-existing repo hygiene outside Phase-4 scope**, left as a recommendation for
  the maintainer rather than a silent repo-wide edit by me. *(Recommended if desired:
  append `*.pth`, `*.npy`, `*.pkl`, `clash/` to `.gitignore` — future-only protection.)*

## 4. Scope & isolation — **PASS**
| Check | Result |
|---|---|
| No archived artifact overwritten | **confirmed** — `git status --porcelain` shows **zero modified files**; only untracked additions |
| Failed candidate-scoring pilot preserved as negative reference | **yes**, untouched; cited in `PHASE4_EXISTING_EVIDENCE_AUDIT.md` and §2 of the R0 decision |
| Forbidden work not performed | **confirmed** — no activation extraction, router training, direction extraction, alpha tuning, SAKIKO / SAKIKO-CA, placebo intervention, or multi-seed intervention was run. GPU was used **only** for black-box generation (baseline / replay / paraphrase). |
| Git write operations | **none executed.** Only read-only git (`status`, `ls-files`, `check-ignore`) was used. No add/commit/push/pull/fetch/merge/rebase/reset/clean/branch/checkout/switch/remote/config; no `ssh -T`; no GitHub auth; branch untouched. |

## 5. Reproducibility — **PASS**
| Element | Status |
|---|---|
| Readout locked **before** the full run | `READOUT_LOCK_MANIFEST.json` — 8 sha256 hashes (protocol, parser, generation script, dataset, both official prompts, model config + index), protocol v1.0 / parser v1.0 |
| Thresholds fixed pre-hoc | R0.1–R0.10 + verdict rule in `PHASE4_GENERATION_PROTOCOL.md §17`, hashed pre-run. **No threshold moved after results were seen.** |
| Determinism | greedy (`do_sample=False`, `num_beams=1`), batch 1, ascending `sample_id`; **50-row replay = 100% identical** |
| Parser determinism | pure string processing; byte-identical over 3× repeats (30/30 unit tests) |
| Dataset provenance | `dataset_manifest.json` — 12 raw files sha256-verified; regenerated 800-row set verified **cell-exact** vs archived split_info (24/24 cells, gold distribution, first-200 order) |
| Environment | torch 2.1.2+cu121 · transformers 4.49.0 · RTX 4090 D · bf16 (asserted unquantized at load: `dtypes={torch.bfloat16}`, `quantized=False`, 15.41 GB) |
| Figures traceable to source tables | `figures/figure_manifest.csv` maps each of 5 figures → its source JSON/CSV |
| Re-run commands | documented in each report; all scripts CLI-invocable |

## 6. Honesty ledger (things counted against the result)
- **R0.9 fails** and is reported as a failure — no threshold relaxation, no re-split.
- **Paraphrase check fails** (0.56 vs 0.80) and was **run once, not re-rolled**; its own construction
  weakness (a weaker format directive than the official prompt) is disclosed in
  `PARAPHRASE_ROBUSTNESS_REPORT.md §3`, and the strong claim it might support is explicitly *not* made.
- **§21 taxonomy gap** (CONDITIONAL PASS + paraphrase FAIL was never enumerated) is disclosed in the
  decision rather than silently forced into a category.
- **1 parser mislabel** (`0499`) and **2 flagged conflicts** are counted against the audit.
- **R0.7 passes narrowly** (EOS 95.87% vs 95%) — disclosed as a sensitivity, cap not changed post-hoc.
- The R0-passing readout is **not** spun as a positive result for SAKIKO-CA: no transfer claim is made
  in either direction.

---

# VERDICT: **PASS**

No secrets, no PII, no private-network or path leakage, no weights/activations/large raw generations,
no archived artifact modified, no forbidden experiment run, no git write operation executed. One
self-caught artifact-policy violation was corrected (raw-generation audit sample moved to gitignored
cache) and one pre-existing gitignore gap is reported to the maintainer with reasons rather than
silently patched. The Phase-4 package is safe for the human to review, stage, and commit.
