# Qwen3-8B Stage 0/1 prospective protocol

Status: **frozen before any Qwen3 model load or Qwen3 score**. Tier: developmental, train/dev-only, retryable.

## Scientific role and claim boundary

This starts a prospectively isolated Qwen3 model line. It measures baseline response-mode decisions, automatically discovers Qwen3 TRAIN error channels, estimates train-only DiffMean directions and Routers, and computes dev-only sample-conditioned geometry. It produces no evaluation result and no Stage 2 intervention result. Because future Qwen3 evaluation may reuse rows seen by older models, the strongest future framing is **prospective held-out-model validation**, not new-data-held-out confirmation.

## Model and thinking mode

- Official repository: `Qwen/Qwen3-8B`; immutable revision `b968826d9c46dd6066d109eabc6255188de91218`.
- Architecture: Qwen3ForCausalLM, 36 layers, hidden size 4096, BF16, eager attention.
- Fixed environment: Transformers 4.51.0, tokenizers 0.21.1, Torch 2.1.2+cu121.
- tokenizer_config SHA256: `d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101`; chat-template UTF-8 SHA256: `a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8`.
- The only template call passes parsed tools separately and sets `tokenize=False, add_generation_prompt=True, enable_thinking=False`. The candidate starts immediately after Qwen3's deterministic empty `<think>...</think>` block. No generated reasoning precedes it. Failure is `READOUT_INVALID_THINKING_MODE`; no template sweep is allowed.

## Test firewall and data

The pinned `nvidia/When2Call` revision is `0582f7749df63a96fdc3070932e83e72396ace53`, config/split `test/mcq`. The original 3,652-row order is mapped through `numpy.random.default_rng(42).permutation(3652)`. Authorized project populations are train 2,556 and dev 548; the remaining 548 are sealed evaluation. The committed selective byte index emits only train/dev raw offsets. It was validated against all 3,104 authorized UUIDs with zero mismatch. It does not open the evaluation manifest, emit evaluation offsets, or parse non-allowlisted payloads. Every consumer must direct-seek through this index.

## Readout

The frozen mode order is tool_call, direct, request_for_info, cannot_answer. Each sample uses four separate teacher-forced forwards and its own complete multi-token candidate. Prompt tokens are excluded. Candidate tokens are separately encoded without BOS/EOS and averaged over logits at positions `prompt_len-1 ... prompt_len+candidate_len-2`. There is no generation, single-token proxy, padding, calibration, or truncation. Any sequence above 8,192 tokens blocks execution rather than being skipped. Eight TRAIN rows selected by ascending immutable UUID are replayed twice; scores must agree within 1e-5 and predictions/runner-ups exactly. Score gaps <=1e-8 are unresolved ties and block the readout.

## Automatic channels and support gate

All 12 directed Qwen3 TRAIN gold→prediction errors are ledgered. A channel is SUPPORT_ELIGIBLE iff train error >=60, train correct-gold reference >=60, dev error >=30, and dev correct-gold reference >=30. Every passing channel is retained. Channels are never merged, split, inherited from Qwen2.5, or selected by Router/geometry favorability.

## Direction, layers, and Router

The inherited estimator is `mean(TRAIN correct gold reference activation)-mean(TRAIN gold→source error activation)`, normalized only after storing its raw norm. Each float32 vector is saved directly as safetensors through Git LFS with sign and population hashes. Qwen2.5 had 28 layers and the pre-existing common/modal rule L_obs=20, L_inj=16. The frozen normalized mapping gives Qwen3 L_obs=26 and L_inj=21, zero-based, at `model.model.layers[L].mlp` output. The baseline pass captures the final-prompt-token activation at both L26 and L21; direction and Router estimation use L26, while exact dev gradients use L21. There is no layer sweep.

Each Router uses the L26 last-prompt-token MLP output, TRAIN channel errors as positives, and all correct TRAIN rows as negatives. The estimator is StandardScaler plus LogisticRegression(C=1, L2, liblinear, max_iter=2000, tol=1e-4, seed=42, no class weighting). The dev threshold grid is [0.4,0.5,0.6,0.7,0.8]; select the smallest threshold with source-matched precision >=0.50. Router eligibility additionally requires comparable dev ROC-AUC >=0.75.

## Dev geometry

For every support-eligible channel and DEV error row, candidate-conditioned gradients are taken at L21. Full effect sums every prompt-plus-candidate position; common prompt sums prompt positions only when all four prompt states agree within 1e-6. The contrast axis is `unit(G_gold-G_source)` and signed contrast is its dot product with the train direction. Centering the four mode gradients defines the rank<=3 decision subspace. The primary predictor is `Q_sum=sum(a_offaxis)/sum(a_dec)`; negative-sign fraction is key secondary. Frozen tolerances and null seeds are in the JSON protocol. Raw `[4,4096]` float32 gradients are stored via LFS. No dev behavioral correction is claimed, and no first-order destination rate is computed because no Stage 2 dose is frozen.

## Future Stage 2 branch
## Frozen output schemas

Baseline rows are deterministic JSONL with authorized project index, immutable UUID, split, gold, four finite scores, prediction, runner-up, candidate lengths, prompt length, and all 12 ordered margins; both the canonical schema and split prediction vectors are SHA256-hashed. The activation LFS object contains float32 `[3104,4096]` tensors for L26 and L21 in baseline row order. Each direction LFS object contains one float32 `[4096]` unit vector plus channel/layer/raw-norm/sign/population metadata. Each Router LFS object contains StandardScaler and logistic-regression numeric tensors plus the selected tau. Raw dev gradients are float32 `[4,4096]` tensors keyed by channel, variant, and ordinal.

The complete 12-transition ledger, retry ledger, structured JSONL event log, environment manifest, runtime/memory manifest, failure manifest, geometry rows/analysis, handoff, and path-to-SHA256 manifests use the exact schemas in the JSON protocol. Hash manifests exclude themselves. Engineering retries append the command, script commit/hash, exit code, reason, and invalidated outputs. Any scientific-definition change requires a newly committed protocol version before rerunning.


K>=5 uses one-sided exact Kendall tau between lower dev Q_sum and higher evaluation destination selectivity. K=4 uses the same test labeled LOW_POWER. K=1..3 withdraws predictor validation and makes real-versus-prospectively-generated-matched-random destination selectivity the future primary for every channel. K=0 stops with NO_ELIGIBLE_CHANNELS. Qwen2.5 thresholds are exploratory transfer checks only. This protocol creates no Stage 2 runner, preregistration, freeze, preflight, or evaluation prediction.
